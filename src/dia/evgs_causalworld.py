# src/dia/evgs_causalworld.py
from __future__ import annotations

from typing import Any, Dict, Optional
from dataclasses import dataclass
import numpy as np

try:
    import gym
except Exception:
    gym = None

from .evgs_adapters import make_causalworld_evgs, InfoObsWrapper


@dataclass
class CausalWorldAdapterConfig:
    # Block z-thresholds (meters). Block center is at ~0.065m when resting on table.
    table_z: float = 0.065            # block z at rest (observed from env)
    lifted_z_threshold: float = 0.085 # block z > this = lifted off table
    high_lifted_z_threshold: float = 0.12  # block z > this = clears max obstacle (0.10m)
    # Goal proximity threshold (xy distance in meters)
    above_goal_xy_threshold: float = 0.03
    # Fractional success threshold
    success_threshold: float = 0.9


class CausalWorldInfoWrapper(gym.Wrapper if gym else object):
    """
    Extracts task-relevant scalars from CausalWorld info dict and exposes them
    under stable keys for the EVGS to read.

    CausalWorld provides per-step info:
        achieved_goal: shape (1, 2, 3) - [finger_tips_pos, block_pos]
        desired_goal:  shape (1, 2, 3) - [finger_tips_goal, block_goal]
        success: bool
        fractional_success: float [0, 1]

    This wrapper computes:
        block_z:           block center z-coordinate
        dist_xy:           xy distance from block to goal
        fractional_success: pass-through from env
        success:           pass-through from env
    """

    def __init__(self, env, cfg: Optional[CausalWorldAdapterConfig] = None):
        if gym:
            super().__init__(env)
        self.cfg = cfg or CausalWorldAdapterConfig()
        self._last: Dict[str, Any] = {
            "block_z": self.cfg.table_z,
            "dist_xy": 1.0,
            "success": False,
            "fractional_success": 0.0,
        }
        self._last_obs = None

    def _extract(self, info: Dict[str, Any]) -> Dict[str, Any]:
        try:
            achieved = np.array(info["achieved_goal"])[0, 1]  # block position [x,y,z]
            desired = np.array(info["desired_goal"])[0, 1]    # block goal [x,y,z]
            block_z = float(achieved[2])
            dist_xy = float(np.linalg.norm(achieved[:2] - desired[:2]))
        except (KeyError, IndexError, TypeError):
            block_z = self._last["block_z"]
            dist_xy = self._last["dist_xy"]
        return {
            "block_z": block_z,
            "dist_xy": dist_xy,
            "success": bool(info.get("success", False)),
            "fractional_success": float(info.get("fractional_success", 0.0)),
        }

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._last_obs = {"obs": obs, "info": dict(self._last)}
        return self._last_obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        extracted = self._extract(dict(info or {}))
        self._last = extracted
        self._last_obs = {"obs": obs, "info": extracted}
        return self._last_obs, rew, done, extracted

    def get_obs(self):
        if self._last_obs is None:
            return self.reset()
        return self._last_obs


def wrap_causalworld_env(
    env,
    cfg: Optional[CausalWorldAdapterConfig] = None,
    return_evgs: bool = False,
):
    env = InfoObsWrapper(env)
    env = CausalWorldInfoWrapper(env, cfg or CausalWorldAdapterConfig())
    if return_evgs:
        return env, make_causalworld_evgs(cfg=cfg)
    return env
