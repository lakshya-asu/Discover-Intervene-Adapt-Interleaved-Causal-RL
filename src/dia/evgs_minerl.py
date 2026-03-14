# src/dia/evgs_minerl.py
"""
EVGS adapter for MineRL 3D Minecraft environment.

Maps MineRL's inventory observation (flat dict: item_name → count)
to the same 9 binary variables used in the 2D symbolic Minecraft chain:

  0  wood           — any log variant in inventory (>= 1)
  1  stone          — cobblestone in inventory (>= 1)
  2  coal           — coal in inventory (>= 1)
  3  ironore        — iron_ore in inventory (>= 1)
  4  furnace        — furnace in inventory (>= 1)
  5  stonepickaxe   — stone_pickaxe in inventory (>= 1)
  6  iron           — iron_ingot in inventory (>= 1)
  7  ironpickaxe    — iron_pickaxe in inventory (>= 1)
  8  diamond        — diamond in inventory (>= 1)

MineRL observation format:
  obs['pov']               — np.ndarray (H, W, 3) uint8  (default 360×640)
  obs['inventory']         — dict {item_name: scalar_count}
                             e.g. {'oak_log': 3, 'cobblestone': 0, ...}

Usage
-----
    import gym
    env  = gym.make("MineRLObtainDiamondShovel-v0")
    evgs = make_minerl_evgs()

    obs = env.reset()
    x   = evgs.extract(obs)   # → np.ndarray shape (9,), values 0.0 or 1.0
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from .evgs import EVGS

# ── Variable names (identical to 2D Minecraft) ───────────────────────────────
VAR_NAMES: List[str] = [
    "wood", "stone", "coal", "ironore", "furnace",
    "stonepickaxe", "iron", "ironpickaxe", "diamond",
]

# ── Minecraft item-name → DIA variable mapping ───────────────────────────────
# MineRL inventory keys are full Minecraft item IDs (snake_case, no namespace).
_ITEM_MAP: Dict[str, List[str]] = {
    "wood": [
        "oak_log", "birch_log", "spruce_log", "jungle_log",
        "acacia_log", "dark_oak_log", "mangrove_log", "cherry_log",
        "log",  # some versions use generic "log"
        "oak_wood", "birch_wood", "spruce_wood", "jungle_wood",
        "acacia_wood", "dark_oak_wood",
    ],
    "stone": [
        "cobblestone",
        "cobbled_deepslate",  # deepslate cobblestone also counts
    ],
    "coal": [
        "coal",
    ],
    "ironore": [
        "iron_ore",
        "deepslate_iron_ore",
    ],
    "furnace": [
        "furnace",
    ],
    "stonepickaxe": [
        "stone_pickaxe",
    ],
    "iron": [
        "iron_ingot",
    ],
    "ironpickaxe": [
        "iron_pickaxe",
    ],
    "diamond": [
        "diamond",
    ],
}

# Build reverse lookup: item_name → variable_index
_ITEM_TO_VAR: Dict[str, int] = {}
for _var_name, _items in _ITEM_MAP.items():
    _var_idx = VAR_NAMES.index(_var_name)
    for _item in _items:
        _ITEM_TO_VAR[_item.lower()] = _var_idx


def _extract_from_flat_dict(inv_dict: Dict[str, Any]) -> np.ndarray:
    """
    Extract DIA variable counts from MineRL's flat inventory dict.

    inv_dict looks like: {'oak_log': 3, 'cobblestone': 0, 'coal': 2, ...}
    Values may be ints, floats, or 0-d numpy arrays.
    """
    counts = np.zeros(len(VAR_NAMES), dtype=float)
    for item_name, qty in inv_dict.items():
        var_idx = _ITEM_TO_VAR.get(str(item_name).lower())
        if var_idx is not None:
            counts[var_idx] += float(qty)
    return counts


def _obs_to_vars(obs: Any) -> np.ndarray:
    """
    Convert a MineRL step observation to the 9-dim binary variable vector.

    Binary threshold: variable = 1.0 iff count >= 1.
    """
    inv = None
    if isinstance(obs, dict):
        inv = obs.get("inventory")

    if inv is None or not isinstance(inv, dict):
        return np.zeros(len(VAR_NAMES), dtype=float)

    counts = _extract_from_flat_dict(inv)
    return (counts >= 1.0).astype(float)


def make_minerl_evgs(var_names: Optional[List[str]] = None) -> EVGS:
    """
    Create an EVGS instance for MineRL 3D Minecraft.

    Returns
    -------
    EVGS with:
      .extract(obs) → binary np.ndarray shape (9,)
      .names()      → ['wood', 'stone', 'coal', 'ironore', 'furnace',
                        'stonepickaxe', 'iron', 'ironpickaxe', 'diamond']
    """
    names = var_names or VAR_NAMES
    return EVGS(var_names=names, obs_to_vars=_obs_to_vars)


def inventory_summary(obs: Any) -> str:
    """Human-readable inventory summary from a MineRL obs dict."""
    if not isinstance(obs, dict):
        return "(no obs)"
    inv = obs.get("inventory", {})
    if not isinstance(inv, dict):
        return "(no inventory)"
    counts = _extract_from_flat_dict(inv)
    parts = [f"{VAR_NAMES[i]}={int(counts[i])}" for i in range(len(VAR_NAMES))
             if counts[i] > 0]
    return "  ".join(parts) if parts else "(empty)"
