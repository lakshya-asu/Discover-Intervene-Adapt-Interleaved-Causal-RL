# src/dia/evgs_procgen.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import numpy as np

try:
    import gym
except Exception:
    gym = None

from .evgs_adapters import make_coinrun_evgs, InfoObsWrapper


@dataclass
class CoinRunDetectorConfig:
    yellow_r_min: int = 200
    yellow_g_min: int = 180
    yellow_b_max: int = 60
    min_coin_pixels: int = 8
    # centroid_x < frame_w * coin_close_frac  →  coin "close" (agent nearly there)
    # 0.80 means the coin just needs to be in the left 80 % of the frame.
    # Since the coin enters from the right (~centroid_x 55-60 in a 64px frame)
    # this fires within a few steps of the coin becoming visible.
    coin_close_frac: float = 0.80


class ProcgenCoinRunInfoWrapper(gym.Wrapper if gym else object):
    """
    Wraps a ProcGen CoinRun env and injects three semantic variables into info:

      coin_visible  (0/1) – yellow coin pixels present in the current frame
      coin_close    (0/1) – coin centroid in the left 55 % of the frame (player almost there)
      coin_collected(0/1) – coin was collected this episode (set on prev_level_complete)

    soft_continue is True while the episode is running, False on done so that
    PixelStackPPOOption terminates cleanly at episode boundaries.
    """

    def __init__(self, env, cfg: Optional[CoinRunDetectorConfig] = None):
        if gym:
            super().__init__(env)
        self.cfg = cfg or CoinRunDetectorConfig()
        self._coin_visible   = 0.0
        self._coin_close     = 0.0
        self._coin_collected = 0.0
        self._last_obs: Optional[Dict] = None

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _extract_frame(obs) -> Optional[np.ndarray]:
        """Unwrap nested dict obs until we reach a (H, W, 3) uint8 ndarray."""
        raw = obs
        for _ in range(4):
            if isinstance(raw, np.ndarray):
                break
            if isinstance(raw, dict):
                raw = raw.get("obs", raw)
            else:
                break
        return raw if (isinstance(raw, np.ndarray) and raw.ndim == 3 and raw.shape[-1] == 3) else None

    def _detect_coin(self, frame: Optional[np.ndarray]) -> Tuple[float, float]:
        """Return (coin_visible, coin_close) from a raw RGB frame."""
        if frame is None:
            return 0.0, 0.0
        r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
        mask = (r >= self.cfg.yellow_r_min) & (g >= self.cfg.yellow_g_min) & (b <= self.cfg.yellow_b_max)
        npix = int(np.count_nonzero(mask))
        if npix < self.cfg.min_coin_pixels:
            return 0.0, 0.0
        xs = np.where(mask)[1]
        cx = float(np.mean(xs))
        W = frame.shape[1]
        coin_close = 1.0 if cx < W * self.cfg.coin_close_frac else 0.0
        return 1.0, coin_close

    def _reset_episode(self):
        self._coin_visible   = 0.0
        self._coin_close     = 0.0
        self._coin_collected = 0.0

    # ------------------------------------------------------------------ gym API

    def reset(self, **kwargs):
        self._reset_episode()
        obs = self.env.reset(**kwargs)
        frame = self._extract_frame(obs)
        self._coin_visible, self._coin_close = self._detect_coin(frame)
        info = {
            "coin_visible":   self._coin_visible,
            "coin_close":     self._coin_close,
            "coin_collected": self._coin_collected,
            "soft_continue":  True,
        }
        self._last_obs = {"obs": obs, "info": info}
        return self._last_obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        frame = self._extract_frame(obs)
        vis, close = self._detect_coin(frame)

        # Only update visible/close while coin is still present
        if not self._coin_collected:
            self._coin_visible = vis
            self._coin_close   = close

        # Collect via ProcGen's level-complete flag (reliable even if frame is the new level)
        if info and info.get("prev_level_complete", False):
            self._coin_collected = 1.0

        vis_now   = self._coin_visible
        close_now = self._coin_close
        coin_now  = self._coin_collected

        # Reset tracking state for the new episode that ProcGen will start on next step
        if done:
            self._reset_episode()

        out_info = dict(info or {})
        out_info.update({
            "coin_visible":   vis_now,
            "coin_close":     close_now,
            "coin_collected": coin_now,
            "soft_continue":  not done,   # ← option breaks cleanly at episode boundary
        })
        self._last_obs = {"obs": obs, "info": out_info}
        return self._last_obs, rew, done, out_info

    def get_obs(self):
        if self._last_obs is None:
            return self.reset()
        return self._last_obs


def wrap_procgen_coinrun_env(env, cfg: Optional[CoinRunDetectorConfig] = None, return_evgs: bool = False):
    env = InfoObsWrapper(env)
    env = ProcgenCoinRunInfoWrapper(env, cfg or CoinRunDetectorConfig())
    if return_evgs:
        return env, make_coinrun_evgs()
    return env
