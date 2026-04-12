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
    """
    Seven semantic variables injected by ProcgenCoinRunInfoWrapper:
      coin_visible    (0/1) – yellow coin pixels present in frame
      coin_close      (0/1) – coin centroid in left 80% of frame
      coin_collected  (0/1) – coin was collected this episode
      coin_elevated   (0/1) – coin centroid in upper 45% of frame (high platform)
      platform_above  (0/1) – climbable brown platform visible in upper half of screen
      saw_visible     (0/1) – saw-blade (gray circular) hazard in mid-frame
      creature_visible(0/1) – colorful enemy creature in right portion of mid-frame
    """
    names = var_names or [
        "coin_visible", "coin_close", "coin_collected",
        "coin_elevated", "platform_above", "saw_visible", "creature_visible",
    ]

    def obs_to_vars(obs):
        d = _ensure_dict_obs(obs)
        info = d.get("info", {}) or {}
        vis      = float(bool(info.get("coin_visible",     0)))
        close    = float(info.get("coin_close",      0.0))
        coin     = float(bool(info.get("coin_collected",   0)))
        elevated = float(info.get("coin_elevated",   0.0))
        platform = float(info.get("platform_above",  0.0))
        saw      = float(bool(info.get("saw_visible",      0)))
        creature = float(bool(info.get("creature_visible", 0)))
        return np.array([vis, close, coin, elevated, platform, saw, creature], dtype=float)

    return EVGS(var_names=names, obs_to_vars=obs_to_vars)


# ------------------------------ CausalWorld ------------------------------
#
# 5-variable EVGS for pick_and_place task.
# Reads from info keys produced by CausalWorldInfoWrapper:
#   block_z, dist_xy, success, fractional_success
#
# Ground-truth causal DAGs:
#   T0 (small obstacle, h=0.02):
#       grasped -> target_lifted -> target_above_goal -> task_success
#   T1 (large obstacle, h=0.10, structural change):
#       grasped -> target_lifted -> target_high_lifted -> target_above_goal -> task_success
#   T2 (large tool_block, motor change):
#       same DAG as T0; only pi_grasp motor policy needs retraining
#
# Intervention API:
#   T1: env.do_intervention({"obstacle": {"size": np.array([0.5, 0.015, 0.10])}})
#   T2: env.do_intervention({"tool_block": {"size": np.array([0.085, 0.085, 0.085])}})
#
# Obstacle bounds in pick_and_place:
#   x=0.5 (fixed), y=0.015 (fixed), z: space_a=[0.02,0.065], space_b=[0.065,0.10]

def make_causalworld_evgs(
    var_names: Optional[List[str]] = None,
    cfg=None,
) -> EVGS:
    """5-variable EVGS for CausalWorld pick_and_place.

    Variables (indexed 0-4):
        0  grasped            block_z > table_z + 0.01 (any lift from rest)
        1  target_lifted      block_z > lifted_z_threshold  (~0.085m)
        2  target_high_lifted block_z > high_lifted_z_threshold (~0.12m)
        3  target_above_goal  xy dist to goal < above_goal_xy_threshold (~0.03m)
        4  task_success       fractional_success > success_threshold (~0.9)
    """
    names = var_names or [
        "grasped",
        "target_lifted",
        "target_high_lifted",
        "target_above_goal",
        "task_success",
    ]

    # Use defaults if no cfg provided
    table_z = getattr(cfg, "table_z", 0.065)
    lifted_thresh = getattr(cfg, "lifted_z_threshold", 0.085)
    high_thresh = getattr(cfg, "high_lifted_z_threshold", 0.12)
    xy_thresh = getattr(cfg, "above_goal_xy_threshold", 0.03)
    succ_thresh = getattr(cfg, "success_threshold", 0.9)

    def obs_to_vars(obs):
        d = _ensure_dict_obs(obs)
        info = d.get("info", {}) or {}
        block_z = float(info.get("block_z", table_z))
        dist_xy = float(info.get("dist_xy", 1.0))
        frac = float(info.get("fractional_success", 0.0))
        grasped = float(block_z > table_z + 0.01)
        lifted = float(block_z > lifted_thresh)
        high_lifted = float(block_z > high_thresh)
        above_goal = float(dist_xy < xy_thresh)
        success = float(frac > succ_thresh)
        return np.array([grasped, lifted, high_lifted, above_goal, success], dtype=float)

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
