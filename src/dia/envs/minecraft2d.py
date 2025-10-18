# src/dia/envs/minecraft2d.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

try:
    import gym
    from gym import spaces
except Exception:
    gym = None
    spaces = None


def _maybe_spaces_discrete(n: int):
    if spaces is not None:
        return spaces.Discrete(n)
    # Fallback minimal action space with sample()
    class _Discrete:
        def __init__(self, n):
            self.n = int(n)
        def sample(self):
            return int(np.random.randint(0, self.n))
    return _Discrete(n)


@dataclass
class ChainConfig:
    # Consumption & crafting rules
    stone_for_furnace: int = 8
    wood_for_stick: int = 2          # simplified stick cost folded into pickaxe costs
    stone_for_stone_pick: int = 3
    wood_for_stone_pick: int = 2
    iron_for_iron_pick: int = 3
    wood_for_iron_pick: int = 2
    smelt_coal_per_iron: int = 1
    smelt_ore_per_iron: int = 1

    # Limits
    max_count: int = 999

    # Rewards (optional extrinsic; DIA uses subgoal success internally)
    rew_diamond: float = 10.0
    rew_shaping: float = 0.0  # set >0 if you want shaping on other events


class MinecraftChainEnv(gym.Env if gym else object):
    """
    A tiny symbolic 2D-Minecraft causal chain:

        wood, stone -> craft stone_pickaxe
        stone -> craft furnace
        coal + ironore + furnace -> smelt iron
        iron + wood -> craft iron_pickaxe
        iron_pickaxe -> mine diamond

    Observation: dict {"obs": np.array([wood, stone, coal, ironore, furnace, stonepickaxe,
                                        iron, ironpickaxe, diamond], dtype=float),
                       "info": {...}}
    Actions (Discrete 10):
      0: gather wood (+1)
      1: gather stone (+1)
      2: gather coal (+1)
      3: mine iron ore (+1)              [requires stonepickaxe or ironpickaxe]
      4: craft furnace (stone-8 => furnace=1)
      5: craft stone pickaxe (wood-2, stone-3 => stonepickaxe=1)
      6: smelt iron (furnace & ironore>=1 & coal>=1 => iron+1, ironore-1, coal-1)
      7: craft iron pickaxe (iron-3, wood-2 => ironpickaxe=1)
      8: mine diamond (+1)               [requires ironpickaxe]
      9: noop
    """
    metadata = {"render.modes": []}

    VAR_NAMES = [
        "wood", "stone", "coal", "ironore", "furnace", "stonepickaxe",
        "iron", "ironpickaxe", "diamond"
    ]

    def __init__(self, cfg: Optional[ChainConfig] = None, seed: Optional[int] = None):
        super().__init__()
        self.cfg = cfg or ChainConfig()
        self.rng = np.random.RandomState(seed or 0)

        self.n_vars = len(self.VAR_NAMES)
        self.state = np.zeros(self.n_vars, dtype=float)

        self.action_space = _maybe_spaces_discrete(10)
        if spaces is not None:
            self.observation_space = spaces.Box(low=0.0, high=float(self.cfg.max_count),
                                                shape=(self.n_vars,), dtype=np.float32)

    # ------------- Helpers -------------

    def _clip(self):
        np.clip(self.state, 0.0, float(self.cfg.max_count), out=self.state)

    def get_obs(self):
        # Return dict observation compatible with EVGS adapters
        return {"obs": self.state.copy(), "info": {}}

    # ------------- Gym API -------------

    def reset(self, **kwargs):
        self.state[:] = 0.0
        return self.get_obs()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        s = self.state
        cfg = self.cfg
        rew = 0.0

        if action == 0:  # + wood
            s[0] += 1.0

        elif action == 1:  # + stone
            s[1] += 1.0

        elif action == 2:  # + coal
            s[2] += 1.0

        elif action == 3:  # + iron ore (requires pickaxe)
            if s[5] >= 1.0 or s[7] >= 1.0:  # stonepickaxe or ironpickaxe
                s[3] += 1.0

        elif action == 4:  # craft furnace
            if s[4] < 1.0 and s[1] >= cfg.stone_for_furnace:
                s[1] -= cfg.stone_for_furnace
                s[4] = 1.0

        elif action == 5:  # craft stone pickaxe
            if s[5] < 1.0 and s[0] >= cfg.wood_for_stone_pick and s[1] >= cfg.stone_for_stone_pick:
                s[0] -= cfg.wood_for_stone_pick
                s[1] -= cfg.stone_for_stone_pick
                s[5] = 1.0

        elif action == 6:  # smelt iron
            if s[4] >= 1.0 and s[3] >= cfg.smelt_ore_per_iron and s[2] >= cfg.smelt_coal_per_iron:
                s[3] -= cfg.smelt_ore_per_iron
                s[2] -= cfg.smelt_coal_per_iron
                s[6] += 1.0

        elif action == 7:  # craft iron pickaxe
            if s[7] < 1.0 and s[6] >= cfg.iron_for_iron_pick and s[0] >= cfg.wood_for_iron_pick:
                s[6] -= cfg.iron_for_iron_pick
                s[0] -= cfg.wood_for_iron_pick
                s[7] = 1.0

        elif action == 8:  # mine diamond
            if s[7] >= 1.0:
                s[8] += 1.0
                rew += cfg.rew_diamond

        elif action == 9:
            pass  # noop

        # Optional shaping (off by default)
        rew += cfg.rew_shaping * 0.0

        self._clip()
        info = {"soft_continue": True}
        return self.get_obs(), float(rew), False, info  # never 'done' in smoke test (runner handles horizons)
