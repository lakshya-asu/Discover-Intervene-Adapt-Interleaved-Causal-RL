"""Abstract environment API for DIA experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class EnvAPI(ABC):
    """Common interface for DIA-compatible environments."""

    @abstractmethod
    def reset(self, seed: Optional[int] = None, factors: Optional[Dict[str, Any]] = None) -> Any:
        """Reset environment state and return the initial observation."""

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Advance one step using `action` and return (obs, reward, done, info)."""

    @property
    @abstractmethod
    def action_space(self) -> Any:
        """Return the environment's action space descriptor."""

    @property
    @abstractmethod
    def observation_space(self) -> Any:
        """Return the environment's observation space descriptor."""

    def get_factors(self) -> Dict[str, Any]:
        """Return the latent causal factors, if available (sim-only)."""
        return {}

    def intervene(self, spec: Dict[str, Any]) -> bool:
        """Apply an intervention described by `spec`; override when supported."""
        return False
