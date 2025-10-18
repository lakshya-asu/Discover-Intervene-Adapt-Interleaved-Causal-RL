# src/dia/evgs_cartpole.py
from __future__ import annotations

from typing import Any, List, Optional
import numpy as np
from .evgs import EVGS

CARTPOLE_VARS = ["x", "x_dot", "theta", "theta_dot"]

def make_cartpole_evgs() -> EVGS:
    def obs_to_vars(obs: Any) -> np.ndarray:
        arr = np.array(obs if not isinstance(obs, dict) else obs.get("obs", []), dtype=float).reshape(-1)
        out = np.zeros(len(CARTPOLE_VARS), dtype=float)
        out[:min(len(arr), len(out))] = arr[:len(out)]
        return out
    return EVGS(var_names=CARTPOLE_VARS, obs_to_vars=obs_to_vars)
