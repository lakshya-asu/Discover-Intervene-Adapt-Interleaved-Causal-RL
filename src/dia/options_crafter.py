# src/dia/options_crafter.py
"""
Crafter skill options for DIA.
Each CrafterOption targets a single inventory variable and runs
a flat PPO policy until that variable increases or max_steps is reached.
"""
from __future__ import annotations
from typing import Optional
import numpy as np


CRAFTER_SKILLS = [
    "wood", "stone", "coal", "iron", "diamond",
    "sapling", "wood_pickaxe", "stone_pickaxe",
    "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword",
]


class CrafterOption:
    """
    A skill option for the Crafter environment.
    target_var: name of the inventory variable this skill aims to increase.
    max_steps: episode steps before the option terminates regardless.
    """

    def __init__(self, target_var: str, max_steps: int = 200):
        if target_var not in CRAFTER_SKILLS:
            raise ValueError(f"Unknown skill target: {target_var}. "
                             f"Choose from {CRAFTER_SKILLS}")
        self.target_var = target_var
        self.max_steps = max_steps
        self.policy = None  # SB3 PPO model; trained separately via train_crafter_skill()

    def act(self, obs: np.ndarray) -> int:
        """Select an action. Falls back to random if policy not yet trained."""
        if self.policy is None:
            return int(np.random.randint(17))
        action, _ = self.policy.predict(obs, deterministic=False)
        return int(action)

    def is_trained(self) -> bool:
        return self.policy is not None


def make_crafter_options(max_steps: int = 200) -> list[CrafterOption]:
    """Create one CrafterOption per skill variable."""
    return [CrafterOption(v, max_steps=max_steps) for v in CRAFTER_SKILLS]
