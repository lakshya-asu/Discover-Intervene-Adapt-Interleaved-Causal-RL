# src/dia/evgs_procgen.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np

try:
    import gym
except Exception:
    gym = None

from .evgs_adapters import make_coinrun_evgs, InfoObsWrapper


@dataclass
class CoinRunDetectorConfig:
    # ---- Coin (yellow) ----
    yellow_r_min: int = 200
    yellow_g_min: int = 180
    yellow_b_max: int = 60
    min_coin_pixels: int = 3
    # coin_close: centroid_x < W * coin_close_frac
    coin_close_frac: float = 0.80
    # coin_elevated: centroid_y < H * elevated_y_frac  (upper portion = elevated)
    elevated_y_frac: float = 0.45

    # ---- Saw blades (circular gray hazards) ----
    saw_gray_min: int = 80
    saw_gray_max: int = 195
    saw_max_channel_diff: int = 40   # R,G,B must be within this of each other
    min_saw_pixels: int = 8

    # ---- Elevated platform (brown tiles in upper half) ----
    plat_r_min: int = 90            # brown: red-dominant
    plat_r_g_gap: int = 12          # R must exceed G by this
    plat_r_b_gap: int = 28          # R must exceed B by this
    plat_b_max: int = 135           # blue cap (not sky-blue)
    min_platform_pixels: int = 20

    # ---- Creature / running enemy (colorful sprite in mid-frame, right side) ----
    creature_sat_min: int = 45      # minimum color saturation (max - min channel)
    creature_bright_min: int = 70   # minimum brightness
    min_creature_pixels: int = 20


class ProcgenCoinRunInfoWrapper(gym.Wrapper if gym else object):
    """
    Wraps a ProcGen CoinRun env and injects 7 semantic variables into info:

      coin_visible    (0/1) – yellow coin pixels present in the current frame
      coin_close      (0/1) – coin centroid in the left 80% of the frame
      coin_collected  (0/1) – coin was collected this episode
      coin_elevated   (0/1) – coin centroid in upper 45% of frame (high platform)
      platform_above  (0/1) – climbable brown platform in upper half of screen
      saw_visible     (0/1) – saw-blade (gray circular) hazard in mid-frame
      creature_visible(0/1) – colorful enemy creature in right portion of mid-frame

    soft_continue is True while the episode is running so PixelStackPPOOption
    terminates cleanly at episode boundaries.
    """

    def __init__(self, env, cfg: Optional[CoinRunDetectorConfig] = None):
        if gym:
            super().__init__(env)
        self.cfg = cfg or CoinRunDetectorConfig()
        self._coin_visible    = 0.0
        self._coin_close      = 0.0
        self._coin_elevated   = 0.0
        self._coin_collected  = 0.0
        self._platform_above  = 0.0
        self._saw_visible     = 0.0
        self._creature_visible = 0.0
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

    def _detect_coin(self, frame: Optional[np.ndarray]) -> Tuple[float, float, float]:
        """Return (coin_visible, coin_close, coin_elevated) from a raw RGB frame."""
        if frame is None:
            return 0.0, 0.0, 0.0
        r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
        mask = (r >= self.cfg.yellow_r_min) & (g >= self.cfg.yellow_g_min) & (b <= self.cfg.yellow_b_max)
        npix = int(np.count_nonzero(mask))
        if npix < self.cfg.min_coin_pixels:
            return 0.0, 0.0, 0.0
        xs = np.where(mask)[1]
        ys = np.where(mask)[0]
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        W, H = frame.shape[1], frame.shape[0]
        coin_close    = 1.0 if cx < W * self.cfg.coin_close_frac else 0.0
        coin_elevated = 1.0 if cy < H * self.cfg.elevated_y_frac else 0.0
        return 1.0, coin_close, coin_elevated

    def _detect_platform_above(self, frame: Optional[np.ndarray]) -> float:
        """Detect brown platform/floor tiles in the upper two-thirds of the frame."""
        if frame is None:
            return 0.0
        H = frame.shape[0]
        r = frame[..., 0].astype(np.int32)
        g = frame[..., 1].astype(np.int32)
        b = frame[..., 2].astype(np.int32)
        brown = (
            (r >= self.cfg.plat_r_min) &
            ((r - g) >= self.cfg.plat_r_g_gap) &
            ((r - b) >= self.cfg.plat_r_b_gap) &
            (b <= self.cfg.plat_b_max)
        )
        # Exclude bottom third (main ground) and top strip (sky)
        brown[2 * H // 3:, :] = False
        brown[:H // 6, :] = False
        # Exclude coin pixels
        coin_m = (
            (frame[..., 0] >= self.cfg.yellow_r_min) &
            (frame[..., 1] >= self.cfg.yellow_g_min) &
            (frame[..., 2] <= self.cfg.yellow_b_max)
        )
        brown &= ~coin_m
        return 1.0 if int(np.count_nonzero(brown)) >= self.cfg.min_platform_pixels else 0.0

    def _detect_saw(self, frame: Optional[np.ndarray]) -> float:
        """Detect saw-blade hazards (medium-brightness gray circular objects) in mid-frame."""
        if frame is None:
            return 0.0
        H = frame.shape[0]
        r = frame[..., 0].astype(np.int32)
        g = frame[..., 1].astype(np.int32)
        b = frame[..., 2].astype(np.int32)
        brightness = (r + g + b) // 3
        max_diff = np.maximum(np.abs(r - g), np.maximum(np.abs(g - b), np.abs(r - b)))
        gray = (
            (max_diff < self.cfg.saw_max_channel_diff) &
            (brightness >= self.cfg.saw_gray_min) &
            (brightness <= self.cfg.saw_gray_max)
        )
        # Exclude very top and very bottom strips
        gray[:H // 6, :] = False
        gray[5 * H // 6:, :] = False
        # Exclude coin pixels
        coin_m = (
            (frame[..., 0] >= self.cfg.yellow_r_min) &
            (frame[..., 1] >= self.cfg.yellow_g_min) &
            (frame[..., 2] <= self.cfg.yellow_b_max)
        )
        gray &= ~coin_m
        return 1.0 if int(np.count_nonzero(gray)) >= self.cfg.min_saw_pixels else 0.0

    def _detect_creature(self, frame: Optional[np.ndarray]) -> float:
        """Detect running enemy creatures (colorful sprites) in the right part of mid-frame.

        Heuristic: high-saturation pixels that are not sky-blue, not coin-yellow, and not
        gray (saw-blade).  We only inspect the right half of the frame since enemies approach
        from the right while the player is roughly centred-left.
        """
        if frame is None:
            return 0.0
        H, W = frame.shape[:2]
        r = frame[..., 0].astype(np.int32)
        g = frame[..., 1].astype(np.int32)
        b = frame[..., 2].astype(np.int32)
        max_c = np.maximum(r, np.maximum(g, b))
        min_c = np.minimum(r, np.minimum(g, b))
        sat   = max_c - min_c
        bright = (r + g + b) // 3
        colorful = (sat >= self.cfg.creature_sat_min) & (bright >= self.cfg.creature_bright_min)
        # Exclude sky (blue dominant)
        colorful &= ~((b > r + 25) & (b > g + 15) & (b > 110))
        # Exclude coin
        colorful &= ~(
            (frame[..., 0] >= self.cfg.yellow_r_min) &
            (frame[..., 1] >= self.cfg.yellow_g_min) &
            (frame[..., 2] <= self.cfg.yellow_b_max)
        )
        # Exclude gray (saw-blades already detected separately)
        max_diff = np.maximum(np.abs(r - g), np.maximum(np.abs(g - b), np.abs(r - b)))
        colorful &= ~(max_diff < self.cfg.saw_max_channel_diff)
        # Only mid-frame height zone
        colorful[:H // 5, :] = False
        colorful[4 * H // 5:, :] = False
        # Only right half of frame (enemies come from right; player is in centre-left)
        colorful[:, :W // 3] = False
        return 1.0 if int(np.count_nonzero(colorful)) >= self.cfg.min_creature_pixels else 0.0

    def _reset_episode(self):
        self._coin_visible     = 0.0
        self._coin_close       = 0.0
        self._coin_elevated    = 0.0
        self._coin_collected   = 0.0
        self._platform_above   = 0.0
        self._saw_visible      = 0.0
        self._creature_visible = 0.0

    # ------------------------------------------------------------------ gym API

    def reset(self, **kwargs):
        self._reset_episode()
        obs = self.env.reset(**kwargs)
        frame = self._extract_frame(obs)
        self._coin_visible, self._coin_close, self._coin_elevated = self._detect_coin(frame)
        self._platform_above   = self._detect_platform_above(frame)
        self._saw_visible      = self._detect_saw(frame)
        self._creature_visible = self._detect_creature(frame)
        info = {
            "coin_visible":    self._coin_visible,
            "coin_close":      self._coin_close,
            "coin_elevated":   self._coin_elevated,
            "coin_collected":  self._coin_collected,
            "platform_above":  self._platform_above,
            "saw_visible":     self._saw_visible,
            "creature_visible": self._creature_visible,
            "soft_continue":   True,
        }
        self._last_obs = {"obs": obs, "info": info}
        return self._last_obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        frame = self._extract_frame(obs)

        vis, close, elevated = self._detect_coin(frame)
        platform  = self._detect_platform_above(frame)
        saw       = self._detect_saw(frame)
        creature  = self._detect_creature(frame)

        # Only update coin tracking while coin is still present
        if not self._coin_collected:
            self._coin_visible  = vis
            self._coin_close    = close
            self._coin_elevated = elevated

        self._platform_above   = platform
        self._saw_visible      = saw
        self._creature_visible = creature

        # Level-complete flag from ProcGen (reliable even on the new-level frame)
        if info and info.get("prev_level_complete", False):
            self._coin_collected = 1.0

        vis_now       = self._coin_visible
        close_now     = self._coin_close
        elevated_now  = self._coin_elevated
        coin_now      = self._coin_collected
        platform_now  = self._platform_above
        saw_now       = self._saw_visible
        creature_now  = self._creature_visible

        # Reset for the new episode ProcGen will start on the very next step
        if done:
            self._reset_episode()

        out_info = dict(info or {})
        out_info.update({
            "coin_visible":    vis_now,
            "coin_close":      close_now,
            "coin_elevated":   elevated_now,
            "coin_collected":  coin_now,
            "platform_above":  platform_now,
            "saw_visible":     saw_now,
            "creature_visible": creature_now,
            "soft_continue":   not done,
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
