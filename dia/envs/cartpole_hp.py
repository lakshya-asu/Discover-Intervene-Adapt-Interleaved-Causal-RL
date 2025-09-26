from __future__ import annotations
import math, random
import gymnasium as gym
import numpy as np
from .base import EnvAPI

class CartPoleHiddenParamsEnv(EnvAPI):
    """Gymnasium CartPole with per-episode hidden pole length/mass."""

    def __init__(self, length_range=(0.4, 1.2), mass_range=(0.05, 0.3)):
        self._length_range = length_range
        self._mass_range = mass_range
        self._env = gym.make("CartPole-v1")
        self._factors = {}

        # expose spaces
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def _sample_factors(self, seed=None):
        rng = random.Random(seed)
        length = rng.uniform(*self._length_range)
        mass = rng.uniform(*self._mass_range)
        self._factors = {"pole_length": length, "pole_mass": mass}

        # set on underlying env (CartPole uses constants; we lightly hack dynamics)
        # We adjust length via _polelength if present; otherwise store for eval only.
        if hasattr(self._env.unwrapped, "length"):
            self._env.unwrapped.length = length
        if hasattr(self._env.unwrapped, "polemass"):
            self._env.unwrapped.polemass = mass
        elif hasattr(self._env.unwrapped, "masspole"):
            self._env.unwrapped.masspole = mass

    def reset(self, seed: int | None = None, options: dict | None = None):
        self._sample_factors(seed)
        obs, info = self._env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        return self._env.step(action)

    def get_factors(self) -> dict:
        return dict(self._factors)
