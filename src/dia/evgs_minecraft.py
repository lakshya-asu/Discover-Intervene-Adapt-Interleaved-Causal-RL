# src/dia/evgs_minecraft.py
from __future__ import annotations

from typing import Any, List, Optional
import numpy as np

from .evgs import EVGS

VAR_NAMES = [
    "wood", "stone", "coal", "ironore", "furnace", "stonepickaxe",
    "iron", "ironpickaxe", "diamond"
]

def make_minecraft_evgs(var_names: Optional[List[str]] = None) -> EVGS:
    names = var_names or VAR_NAMES

    def obs_to_vars(obs: Any) -> np.ndarray:
        # Expect obs to be dict {"obs": np.array([...]), "info": {...}}
        if isinstance(obs, dict) and "obs" in obs:
            arr = np.array(obs["obs"], dtype=float).reshape(-1)
            if arr.shape[0] < len(names):
                pad = np.zeros(len(names), dtype=float)
                pad[:arr.shape[0]] = arr
                return pad
            return arr[:len(names)]
        # If it's already an array, assume it's the variable vector
        arr = np.array(obs, dtype=float).reshape(-1)
        if arr.shape[0] < len(names):
            pad = np.zeros(len(names), dtype=float)
            pad[:arr.shape[0]] = arr
            return pad
        return arr[:len(names)]

    return EVGS(var_names=names, obs_to_vars=obs_to_vars)

