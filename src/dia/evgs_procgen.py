# src/dia/evgs_procgen.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

try:
    import gym
except Exception:
    gym = None

from .evgs_adapters import make_coinrun_evgs, InfoObsWrapper


@dataclass
class CoinRunDetectorConfig:
    horizon: int = 512
    yellow_r_min: int = 200
    yellow_g_min: int = 180
    yellow_b_max: int = 60
    min_coin_pixels: int = 12


class _CoinPixelHeuristics:
    def __init__(self, cfg: CoinRunDetectorConfig):
        self.cfg = cfg
    def coin_present(self, frame: Optional[np.ndarray]) -> bool:
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[-1] != 3:
            return False
        r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
        mask = (r >= self.cfg.yellow_r_min) & (g >= self.cfg.yellow_g_min) & (b <= self.cfg.yellow_b_max)
        return bool(np.count_nonzero(mask) >= self.cfg.min_coin_pixels)


class ProcgenCoinRunInfoWrapper(gym.Wrapper if gym else object):
    def __init__(self, env, cfg: Optional[CoinRunDetectorConfig] = None):
        if gym:
            super().__init__(env)
        self.cfg = cfg or CoinRunDetectorConfig()
        self.pix = _CoinPixelHeuristics(self.cfg)
        self._t = 0
        self._prev_coin_present: Optional[bool] = None
        self._coin_collected = 0.0
        self._last_obs = None

    def reset(self, **kwargs):
        self._t = 0
        self._prev_coin_present = None
        self._coin_collected = 0.0
        obs = self.env.reset(**kwargs)
        frame = obs if isinstance(obs, np.ndarray) else None
        self._prev_coin_present = self.pix.coin_present(frame)
        info = {"coin_collected": self._coin_collected, "progress": 0.0, "enemy_near": 0.0, "soft_continue": True}
        self._last_obs = {"obs": obs, "info": info}
        return self._last_obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        frame = obs if isinstance(obs, np.ndarray) else None
        cur_present = self.pix.coin_present(frame)
        if self._prev_coin_present is True and cur_present is False:
            self._coin_collected = 1.0
        if info and info.get("level_complete", False):
            self._coin_collected = 1.0
        self._prev_coin_present = cur_present
        self._t += 1
        progress = float(min(1.0, self._t / max(1, self.cfg.horizon)))
        inj = {"coin_collected": self._coin_collected, "progress": progress, "enemy_near": 0.0, "soft_continue": True}
        out_info = dict(info or {})
        out_info.update(inj)
        self._last_obs = {"obs": obs, "info": out_info}
        return self._last_obs, rew, done, out_info

    # NEW
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
