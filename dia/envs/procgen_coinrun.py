from __future__ import annotations
import numpy as np
from gymnasium import spaces
from procgen import ProcgenGym3Env
from .base import EnvAPI

class ProcgenCoinRunEnv(EnvAPI):
    """Robust wrapper for Procgen2 CoinRun that auto-detects image slot."""

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

    def _extract_obs(self, tup):
        """Find the dict with 'rgb' and return the image array."""
        for part in tup:
            if isinstance(part, dict) and "rgb" in part:
                rgb = part["rgb"][0]  # drop batch
                return rgb.astype(np.uint8)
        raise RuntimeError(f"No rgb found in observe() output: {tup}")

    def reset(self, seed: int | None = None, options: dict | None = None):
        self._t = 0
        tup = self._env.observe()
        obs = self._extract_obs(tup)
        return obs, {"stub": True}

    def step(self, action):
        self._t += 1
        self._env.act(np.array([action]))
        tup = self._env.observe()
        obs = self._extract_obs(tup)
        reward = float(tup[0][0]) if isinstance(tup[0], np.ndarray) else 0.0
        done = bool(tup[-1][0]) if isinstance(tup[-1], np.ndarray) else False
        return obs, reward, done, False, {"stub": True}

    def close(self):
        self._env.close()
