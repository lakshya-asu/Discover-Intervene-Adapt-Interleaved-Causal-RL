from __future__ import annotations
from abc import ABC, abstractmethod

class EnvAPI(ABC):
    """Minimal interface we standardize on."""

    @abstractmethod
    def reset(self, seed: int | None = None, options: dict | None = None):
        """Return (obs, info) like Gymnasium."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        """Return (obs, reward, terminated, truncated, info) like Gymnasium."""
        raise NotImplementedError

    @property
    @abstractmethod
    def observation_space(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def action_space(self):
        raise NotImplementedError

    # sim-only helpers (no-op by default)
    def get_factors(self) -> dict:
        return {}

    def intervene(self, spec: dict) -> bool:
        return False
