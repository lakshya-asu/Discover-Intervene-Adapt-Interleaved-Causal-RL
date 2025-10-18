# Copyright (c) 2025
# DIA types & dataclasses
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Any


class Predicate(str, Enum):
    """Predicates over environment variables (EVGS)."""
    UP = "UP"        # X_{t+1} > X_t
    DOWN = "DOWN"    # X_{t+1} < X_t
    EQUAL = "EQUAL"  # X_{t+1} == v (with value)
    REACH = "REACH"  # generic reachability predicate


@dataclass(frozen=True)
class Subgoal:
    """A subgoal is a (variable_index, predicate[, value]) triple."""
    var_index: int
    predicate: Predicate
    value: Optional[float] = None

    def pretty(self, var_names: Optional[List[str]] = None) -> str:
        vname = f"X[{self.var_index}]"
        if var_names and 0 <= self.var_index < len(var_names):
            vname = var_names[self.var_index]
        if self.predicate == Predicate.EQUAL:
            return f"{vname} == {self.value}"
        if self.predicate == Predicate.UP:
            return f"{vname} ↑"
        if self.predicate == Predicate.DOWN:
            return f"{vname} ↓"
        if self.predicate == Predicate.REACH:
            return f"reach({vname}={self.value})"
        return f"{vname}:{self.predicate}"
