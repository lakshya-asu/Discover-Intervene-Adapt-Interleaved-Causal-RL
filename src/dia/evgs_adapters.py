# src/dia/evgs_adapters.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

try:
    import gym
except Exception:
    gym = None

from .evgs import EVGS


# ------------------------------ Helpers ------------------------------

def _ensure_dict_obs(obs: Any) -> Dict[str, Any]:
    """If observation is array-like, wrap as {'obs': array}. If dict, return as-is."""
    if isinstance(obs, dict):
        return obs
    return {"obs": obs, "info": {}}


# ------------------------------ Montezuma ------------------------------

def make_montezuma_evgs(var_names: Optional[List[str]] = None) -> EVGS:
    names = var_names or ["has_key", "door_open", "room_id", "skull_near"]

    def obs_to_vars(obs):
        d = _ensure_dict_obs(obs)
        info = d.get("info", {}) or {}
        has_key = float(bool(info.get("has_key", 0)))
        door_open = float(bool(info.get("door_open", 0)))
        room_id = float(int(info.get("room_id", 0)))
        skull_near = float(bool(info.get("skull_near", 0)))
        return np.array([has_key, door_open, room_id / 100.0, skull_near], dtype=float)

    return EVGS(var_names=names, obs_to_vars=obs_to_vars)


# ------------------------------ ProcGen CoinRun ------------------------------

def make_coinrun_evgs(var_names: Optional[List[str]] = None) -> EVGS:
    names = var_names or ["coin_collected", "progress", "enemy_near"]

    def obs_to_vars(obs):
        d = _ensure_dict_obs(obs)
        info = d.get("info", {}) or {}
        coin = float(bool(info.get("coin_collected", info.get("level_complete", 0))))
        progress = float(info.get("progress", 0.0))
        enemy_near = float(bool(info.get("enemy_near", 0)))
        return np.array([coin, progress, enemy_near], dtype=float)

    return EVGS(var_names=names, obs_to_vars=obs_to_vars)


# ------------------------------ CausalWorld ------------------------------

def make_causalworld_evgs(var_names: Optional[List[str]] = None) -> EVGS:
    names = var_names or ["tower_height", "distance_to_goal", "grasped"]

    def obs_to_vars(obs):
        d = _ensure_dict_obs(obs)
        info = d.get("info", {}) or {}
        height = float(info.get("stack_height", 0.0))
        dist = float(info.get("distance_to_goal", 1.0))
        grasped = float(bool(info.get("grasped", 0)))
        return np.array([height, dist, grasped], dtype=float)

    return EVGS(var_names=names, obs_to_vars=obs_to_vars)


# ------------------------------ Info wrapper ------------------------------

class InfoObsWrapper(gym.Wrapper if gym else object):
    """
    Packs step() info into the observation (as {'obs': raw_obs, 'info': info})
    and exposes get_obs() to avoid resetting between macro-steps.
    Handles both 4-tuple (obs, r, done, info) and 5-tuple Gym/Gymnasium API.
    """

    def __init__(self, env):
        if gym:
            super().__init__(env)
        self._last_obs: Optional[Dict[str, Any]] = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._last_obs = {"obs": obs, "info": {}}
        return self._last_obs

    def step(self, action):
        out = self.env.step(action)
        # Accept both APIs
        if isinstance(out, tuple) and len(out) == 5:
            obs, rew, terminated, truncated, info = out
            done = bool(terminated or truncated)
        else:
            obs, rew, done, info = out
        info = dict(info or {})
        self._last_obs = {"obs": obs, "info": info}
        return self._last_obs, rew, done, info

    # Let runner/options fetch the last obs without calling reset()
    def get_obs(self):
        if self._last_obs is None:
            return self.reset()
        return self._last_obs
