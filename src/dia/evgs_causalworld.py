# src/dia/evgs_causalworld.py
from __future__ import annotations

from typing import Any, Dict, Optional, List
from dataclasses import dataclass
import numpy as np

try:
    import gym
except Exception:
    gym = None

from .evgs_adapters import make_causalworld_evgs, InfoObsWrapper


@dataclass
class CausalWorldAdapterConfig:
    height_key: str = "stack_height"
    goal_dist_key: str = "distance_to_goal"
    grasped_key: str = "grasped"


class CausalWorldInfoWrapper(gym.Wrapper if gym else object):
    def __init__(self, env, cfg: Optional[CausalWorldAdapterConfig] = None):
        if gym:
            super().__init__(env)
        self.cfg = cfg or CausalWorldAdapterConfig()
        self._last = {"stack_height": 0.0, "distance_to_goal": 1.0, "grasped": 0.0}
        self._last_obs = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        info = dict(self._last)
        self._last_obs = {"obs": obs, "info": info}
        return self._last_obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        info = dict(info or {})
        out = {
            "stack_height": float(info.get(self.cfg.height_key, self._last["stack_height"])),
            "distance_to_goal": float(info.get(self.cfg.goal_dist_key, self._last["distance_to_goal"])),
            "grasped": float(1.0 if info.get(self.cfg.grasped_key, False) else 0.0),
            "soft_continue": True,
        }
        self._last = out
        self._last_obs = {"obs": obs, "info": out}
        return self._last_obs, rew, done, out

    # NEW
    def get_obs(self):
        if self._last_obs is None:
            return self.reset()
        return self._last_obs


def wrap_causalworld_env(env, cfg: Optional[CausalWorldAdapterConfig] = None, return_evgs: bool = False):
    env = InfoObsWrapper(env)
    env = CausalWorldInfoWrapper(env, cfg or CausalWorldAdapterConfig())
    if return_evgs:
        return env, make_causalworld_evgs()
    return env
