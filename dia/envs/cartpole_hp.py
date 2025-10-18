from __future__ import annotations
import random
import gymnasium as gym
from .base import EnvAPI

class CartPoleHiddenParamsEnv(EnvAPI):
    """Gymnasium CartPole with per-episode hidden pole length/mass."""

    def __init__(self, length_range=(0.4, 1.2), mass_range=(0.05, 0.3)):
        self._length_range = length_range
        self._mass_range = mass_range
        self._env = gym.make("CartPole-v1")
        self._factors = {}
        self._observation_space = self._env.observation_space
        self._action_space = self._env.action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def _sample_factors(self, seed=None):
        rng = random.Random(seed)
        length = rng.uniform(*self._length_range)
        mass = rng.uniform(*self._mass_range)
        self._factors = {"pole_length": length, "pole_mass": mass}
        # Best-effort: adjust underlying env if attributes exist
        un = self._env.unwrapped
        if hasattr(un, "length"):
            un.length = length
        if hasattr(un, "polemass"):
            un.polemass = mass
        elif hasattr(un, "masspole"):
            un.masspole = mass

    def reset(self, seed: int | None = None, options: dict | None = None):
        self._sample_factors(seed)
        obs, info = self._env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        return self._env.step(action)

    def get_factors(self) -> dict:
        return dict(self._factors)

    def close(self):
        self._env.close()
