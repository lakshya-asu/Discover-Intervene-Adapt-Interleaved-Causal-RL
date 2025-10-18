from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from .types import Predicate, Subgoal


@dataclass
class EVGSConfig:
    var_names: List[str]


class EVGS:
    """Environment-Variable Goal Space utilities.
    - Extracts variable vector X from raw env observations (via provided adapter)
    - Evaluates subgoal completion predicates
    """
    def __init__(self, var_names: List[str], obs_to_vars: Callable[[object], np.ndarray]):
        self.var_names = var_names
        self.obs_to_vars = obs_to_vars

    def names(self) -> List[str]:
        return list(self.var_names)

    def extract(self, obs) -> np.ndarray:
        """Map observation -> variable vector X (np.ndarray shape [M])."""
        x = self.obs_to_vars(obs)
        x = np.asarray(x, dtype=float).reshape(-1)
        if len(x) != len(self.var_names):
            raise ValueError(f"obs_to_vars produced {len(x)} variables; expected {len(self.var_names)}")
        return x

    @staticmethod
    def predicate_holds(x_t: np.ndarray, x_tp1: np.ndarray, sg: Subgoal) -> bool:
        if sg.predicate == Predicate.UP:
            return x_tp1[sg.var_index] > x_t[sg.var_index]
        if sg.predicate == Predicate.DOWN:
            return x_tp1[sg.var_index] < x_t[sg.var_index]
        if sg.predicate == Predicate.EQUAL:
            if sg.value is None:
                raise ValueError("EQUAL predicate requires a target value")
            return float(x_tp1[sg.var_index]) == float(sg.value)
        if sg.predicate == Predicate.REACH:
            # generic reachability: within tolerance of target value
            if sg.value is None:
                raise ValueError("REACH predicate requires a target value")
            return abs(float(x_tp1[sg.var_index]) - float(sg.value)) <= 1e-6
        raise ValueError(f"Unknown predicate: {sg.predicate}")
