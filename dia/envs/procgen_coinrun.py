from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from procgen import ProcgenGym3Env
from .base import EnvAPI

class ProcgenCoinRunEnv(EnvAPI):
    """Wraps ProcgenGym3Env to present a Gymnasium-style single-env API."""

    def __init__(self, start_level=0, num_levels=1, render=False):
        self._env = ProcgenGym3Env(num=1, env_name="coinrun",
                                   start_level=start_level, num_levels=num_levels,
                                   render=render)
        # Procgen returns dict of arrays; we pick 'rgb' channel
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(15)

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._env.seed(seed)
        obs = self._env.reset()
        rgb = obs["rgb"][0]  # (1, 64, 64, 3)
        info = {"level_seed": self._env.get_info()[0].get("level_seed", None)}
        return rgb, info

    def step(self, action):
        act = np.array([action], dtype=np.int32)
        self._env.step(act)
        obs = self._env.observe()
        rew = self._env.get_reward()[0]
        first = self._env.get_first()[0]   # episode start flag
        done = self._env.get_done()[0]     # episode end flag
        rgb = obs["rgb"][0]
        info = self._env.get_info()[0]
        terminated = bool(done)
        truncated = False  # Procgen doesn't separate; treat done as terminal
        return rgb, float(rew), terminated, truncated, info
