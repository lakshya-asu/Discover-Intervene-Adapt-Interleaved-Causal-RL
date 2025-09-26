from __future__ import annotations
from .base import EnvAPI

try:
    import causal_world as cw  # module name: causal_world
    _HAS_CW = True
except Exception:
    _HAS_CW = False

class CausalWorldPushingEnv(EnvAPI):
    """Stub to satisfy smoke tests; full implementation in Sprint-2."""

    def __init__(self):
        if not _HAS_CW:
            raise ImportError("causal_world not installed. Install from source and retry.")

    @property
    def observation_space(self):
        raise NotImplementedError("Implement in Sprint-2")

    @property
    def action_space(self):
        raise NotImplementedError("Implement in Sprint-2")

    def reset(self, seed=None, options=None):
        raise NotImplementedError("Implement in Sprint-2")

    def step(self, action):
        raise NotImplementedError("Implement in Sprint-2")

    def close(self):
        pass
