from __future__ import annotations
import numpy as np
from gymnasium import spaces
from procgen import ProcgenGym3Env
from .base import EnvAPI

class ProcgenCoinRunEnv(EnvAPI):
    """Direct Procgen2 (gym3) wrapper for CoinRun (returns uint8 images)."""

    def __init__(self, start_level=0, num_levels=1):
        self._env = ProcgenGym3Env(
            num=1,
            env_name="coinrun",
            start_level=start_level,
            num_levels=num_levels,
        )
        self._t = 0
        self._observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self._action_space = spaces.Discrete(15)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def _cast_obs(self, obs):
        # procgen returns float32 [0,1]; convert to uint8 [0,255]
        if obs.dtype != np.uint8:
            obs = (obs * 255).clip(0, 255).astype(np.uint8)
        return obs

    def reset(self, seed: int | None = None, options: dict | None = None):
        self._t = 0
        obs, _, _, _ = self._env.observe()
        obs = self._cast_obs(obs[0])  # drop batch dimension
        return obs, {"stub": True}

    def step(self, action):
        self._t += 1
        self._env.act(np.array([action]))
        obs, rew, first, done = self._env.observe()
        obs = self._cast_obs(obs[0])  # drop batch dimension
        return obs, float(rew[0]), bool(done[0]), False, {"stub": True}

    def close(self):
        self._env.close()
