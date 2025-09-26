from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .base import EnvAPI

class ProcgenCoinRunEnv(EnvAPI):
    """Gymnasium-style wrapper for CoinRun (procgen2)."""

    def __init__(self, start_level=0, num_levels=1):
        # procgen2 registers Gymnasium envs; this ID works with gym.make(...)
        self._env = gym.make("procgen:procgen-coinrun-v0",
                             start_level=start_level, num_levels=num_levels)
        # Observation is an RGB image; tests expect (64, 64, 3) uint8 and obs as a dict with key "rgb"
        self._observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        # CoinRun has 15 discrete actions
        self._action_space = spaces.Discrete(15)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = self._env.reset(seed=seed, options=options)
        # Wrap raw array into dict so tests can do obs["rgb"]
        return {"rgb": obs}, info or {}

    def step(self, action):
        obs, rew, terminated, truncated, info = self._env.step(action)
        return {"rgb": obs}, float(rew), bool(terminated), bool(truncated), info or {}

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass
