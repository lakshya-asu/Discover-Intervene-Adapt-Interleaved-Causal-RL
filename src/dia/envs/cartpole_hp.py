"""CartPole wrapper that randomizes hidden dynamics parameters per episode."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class CartPoleHiddenParamsEnv(gym.Env):
    """Expose CartPole-v1 while sampling hidden pole parameters each reset."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        *,
        length_range: Tuple[float, float] = (0.4, 0.8),
        mass_range: Tuple[float, float] = (0.05, 0.2),
    ) -> None:
        super().__init__()
        self._env = gym.make("CartPole-v1")
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

        self._length_range = length_range
        self._mass_range = mass_range
        self._rng = np.random.default_rng()
        self._hidden_factors: Dict[str, float] = {}

    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        self._sample_hidden_factors(seed=seed)
        obs, info = self._env.reset(seed=seed, options=options)
        return np.asarray(obs, dtype=np.float32), info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, float]]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        return (
            np.asarray(obs, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    # ------------------------------------------------------------------
    def get_factors(self) -> Dict[str, float]:
        """Return the latent parameters for evaluation purposes."""
        return dict(self._hidden_factors)

    def close(self) -> None:
        self._env.close()

    # ------------------------------------------------------------------
    def _sample_hidden_factors(self, seed: Optional[int]) -> None:
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        length = float(rng.uniform(*self._length_range))
        mass = float(rng.uniform(*self._mass_range))
        self._apply_hidden_parameters(length=length, mass=mass)
        self._hidden_factors = {
            "pole_length": length,
            "pole_mass": mass,
        }

    def _apply_hidden_parameters(self, *, length: float, mass: float) -> None:
        env = getattr(self._env, "unwrapped", self._env)
        env.length = length / 2.0  # underlying implementation uses half-length
        env.masspole = mass
        env.total_mass = env.masspole + env.masscart
        env.polemass_length = env.masspole * env.length


__all__ = ["CartPoleHiddenParamsEnv"]
