# src/dia/shaping.py
from __future__ import annotations

from typing import Any, Callable, Optional
import numpy as np

try:
    import gym
except Exception:
    gym = None

from .evgs import EVGS
from .types import Subgoal


class PredicateShapingEnv(gym.Wrapper if gym else object):
    """
    Generic shaping wrapper: gives +1 and terminates when EVGS predicate holds (x_t -> x_t+1) for a Subgoal.
    - Does NOT alter observations; only reward/done for shaping PPO options.
    - Works when you can compute X from obs or info via a provided extractor.
    """
    def __init__(self, env, extractor: Callable[[Any], np.ndarray], subgoal: Subgoal):
        if gym:
            super().__init__(env)
        self.extractor = extractor
        self.subgoal = subgoal
        self._last_x = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._last_x = self.extractor(obs)
        return obs

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        x = self.extractor(obs)
        reward = 0.0
        if self._last_x is not None:
            # note: EVGS.predicate_holds is static so we call it via the class
            from .evgs import EVGS as _EVGS
            if _EVGS.predicate_holds(self._last_x, x, self.subgoal):
                reward = 1.0
                done = True
        self._last_x = x
        info = dict(info or {})
        info.setdefault("soft_continue", True)
        return obs, float(reward), done, info


def make_shaped_env(env, evgs: EVGS, subgoal: Subgoal):
    """Convenience: build PredicateShapingEnv using EVGS.extract as the extractor."""
    return PredicateShapingEnv(env, extractor=lambda obs: evgs.extract(obs), subgoal=subgoal)


# --- CoinRun-specific simple shaping (coin collected): reward 1 when level complete ---

class CoinCollectedShaping(gym.Wrapper if gym else object):
    """Reward +1 (terminate) when info['level_complete'] or info['coin_collected'] is True."""
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        info = dict(info or {})
        if info.get("level_complete", False) or bool(info.get("coin_collected", False)):
            return obs, 1.0, True, info
        return obs, 0.0, done, info
