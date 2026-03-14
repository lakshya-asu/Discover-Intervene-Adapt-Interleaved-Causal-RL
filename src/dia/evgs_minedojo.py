# src/dia/evgs_minedojo.py
"""
EVGS adapter for MineDojo 3D Minecraft environment.

Maps MineDojo's inventory observation (36 item slots with names + quantities)
to the same 9 binary variables used in the 2D symbolic Minecraft chain:

  0  wood           — any log variant in inventory (>= 1)
  1  stone          — cobblestone in inventory (>= 1)
  2  coal           — coal in inventory (>= 1)
  3  ironore        — iron_ore in inventory (>= 1)
  4  furnace        — furnace in inventory or placed (>= 1)
  5  stonepickaxe   — stone_pickaxe in inventory (>= 1)
  6  iron           — iron_ingot in inventory (>= 1)
  7  ironpickaxe    — iron_pickaxe in inventory (>= 1)
  8  diamond        — diamond in inventory (>= 1)

This shared variable space enables direct PCG transfer:
  pcg_3d.apply_update(np.load("pcg_2d.npy"))
  → 3D agent starts with 2D-learned causal structure as prior.

MineDojo observation format (from minedojo.core.obs_spaces):
  obs['inventory']['name']     — np.ndarray of strings, shape (36,)
  obs['inventory']['quantity'] — np.ndarray of ints,    shape (36,)

Usage
-----
    env  = minedojo.make("open-ended", image_size=(160, 256))
    evgs = make_minedojo_evgs()

    obs  = env.reset()
    x    = evgs.extract(obs)    # → np.ndarray shape (9,), values 0.0 or 1.0
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
# MineDojo item names come from Minecraft's registry (snake_case, no namespace).
# Multiple names per variable handle wood-type variants and spelling differences.
_ITEM_MAP: Dict[str, List[str]] = {
    "wood": [
        "log", "oak_log", "birch_log", "spruce_log", "jungle_log",
        "acacia_log", "dark_oak_log", "mangrove_log", "cherry_log",
        # some MineDojo versions use un-prefixed names
        "minecraft:log", "minecraft:oak_log",
    ],
    "stone": [
        "cobblestone", "minecraft:cobblestone",
        # some tasks use "stone" directly (not cobblestone)
        "stone",
    ],
    "coal": [
        "coal", "minecraft:coal",
    ],
    "ironore": [
        "iron_ore", "deepslate_iron_ore", "minecraft:iron_ore",
    ],
    "furnace": [
        "furnace", "minecraft:furnace",
        "lit_furnace",  # furnace that is currently active
    ],
    "stonepickaxe": [
        "stone_pickaxe", "minecraft:stone_pickaxe",
    ],
    "iron": [
        "iron_ingot", "minecraft:iron_ingot",
    ],
    "ironpickaxe": [
        "iron_pickaxe", "minecraft:iron_pickaxe",
    ],
    "diamond": [
        "diamond", "minecraft:diamond",
    ],
}

# Build reverse lookup: item_name → variable_index
_ITEM_TO_VAR: Dict[str, int] = {}
for _var_name, _items in _ITEM_MAP.items():
    _var_idx = VAR_NAMES.index(_var_name)
    for _item in _items:
        _ITEM_TO_VAR[_item.lower()] = _var_idx


def _extract_inventory_counts(inv_obs: Any) -> np.ndarray:
    """
    Extract inventory item counts from MineDojo's inventory observation.

    Handles several formats MineDojo may return:
    - Dict with 'name' and 'quantity' arrays
    - Structured numpy array (dtype with 'name'/'quantity' fields)
    - List/array of dicts
    """
    counts = np.zeros(len(VAR_NAMES), dtype=float)

    try:
        if isinstance(inv_obs, dict):
            names  = inv_obs.get("name", [])
            qtys   = inv_obs.get("quantity", [])
        elif hasattr(inv_obs, "dtype") and hasattr(inv_obs.dtype, "names"):
            # structured numpy array
            names  = inv_obs["name"]  if "name"  in inv_obs.dtype.names else []
            qtys   = inv_obs["quantity"] if "quantity" in inv_obs.dtype.names else []
        else:
            return counts  # unknown format — return zeros

        for name, qty in zip(names, qtys):
            name_str = str(name).lower().strip()
            # strip surrounding quotes / null bytes
            name_str = name_str.strip("'\"").rstrip("\x00")
            if not name_str or name_str in ("air", "none", ""):
                continue
            var_idx = _ITEM_TO_VAR.get(name_str)
            if var_idx is not None:
                counts[var_idx] += float(qty)

    except Exception:
        pass  # return zeros on any parse error

    return counts


def _obs_to_vars(obs: Any) -> np.ndarray:
    """
    Convert a MineDojo step observation to the 9-dim binary variable vector.

    Binary threshold: variable = 1.0 iff count >= 1.
    (The 2D symbolic env also uses binary presence, so transfer is clean.)
    """
    if isinstance(obs, dict):
        inv = obs.get("inventory", obs.get("obs", None))
    else:
        inv = obs

    if inv is None:
        return np.zeros(len(VAR_NAMES), dtype=float)

    counts = _extract_inventory_counts(inv)
    return (counts >= 1.0).astype(float)


def make_minedojo_evgs(var_names: Optional[List[str]] = None) -> EVGS:
    """
    Create an EVGS instance for MineDojo 3D Minecraft.

    var_names   — optionally override variable names (default: 9-item chain)

    Returns
    -------
    EVGS with:
      .extract(obs) → binary np.ndarray shape (9,)
      .names()      → ['wood', 'stone', 'coal', 'ironore', 'furnace',
                        'stonepickaxe', 'iron', 'ironpickaxe', 'diamond']
    """
    names = var_names or VAR_NAMES
    return EVGS(var_names=names, obs_to_vars=_obs_to_vars)


# ── Utility: inspect inventory for debugging ─────────────────────────────────

def inventory_summary(obs: Any) -> str:
    """Human-readable inventory summary (for debugging)."""
    if isinstance(obs, dict):
        inv = obs.get("inventory", {})
    else:
        inv = obs

    counts = _extract_inventory_counts(inv)
    parts  = [f"{VAR_NAMES[i]}={int(counts[i])}" for i in range(len(VAR_NAMES))
              if counts[i] > 0]
    return "  ".join(parts) if parts else "(empty)"
