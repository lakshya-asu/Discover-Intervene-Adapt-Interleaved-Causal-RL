from __future__ import annotations
import numpy as np
from gymnasium import spaces
from .base import EnvAPI

try:
    import causal_world as cw  # noqa: F401
    _HAS_CW = True
except Exception:
    _HAS_CW = False


class CausalWorldPushingEnv(EnvAPI):
    """Minimal stub so tests can reset/step/close; real CW wiring in Sprint-2."""

    def __init__(self, obs_shape=(64, 64, 3), act_dim=4):
        if not _HAS_CW:
            raise ImportError("causal_world not installed. Install from source and retry.")
        self._observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        self._t = 0

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self, seed: int | None = None, options: dict | None = None):
        self._t = 0
        rgb = np.zeros(self._observation_space.shape, dtype=self._observation_space.dtype)
        # Test expects obs as a dict, and info to contain "factors"
        info = {"stub": True, "factors": {}}
        return {"rgb": rgb}, info

    def step(self, action):
        self._t += 1
        rgb = np.zeros(self._observation_space.shape, dtype=self._observation_space.dtype)
        reward = 0.0
        terminated = self._t >= 1
        truncated = False
        info = {"stub": True, "factors": {}}
        return {"rgb": rgb}, reward, terminated, truncated, info

    def close(self):
        pass
