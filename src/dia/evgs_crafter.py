# src/dia/evgs_crafter.py
"""
EVGS adapter for the Crafter environment (Hafner, 2021).

Maps Crafter's structured observation (flat inventory dict containing all items
and survival stats) to a 15-dimensional variable vector used by DIA's PCG, SIG,
and planner.

Variable groups
---------------
Crafting inventory (indices 0-11, raw integer counts):
    wood, stone, coal, iron, diamond, sapling,
    wood_pickaxe, stone_pickaxe, iron_pickaxe,
    wood_sword, stone_sword, iron_sword

Survival status (indices 12-14, float 0-9 → kept as-is):
    health, food, drink

Note: In Crafter, all items including health/food/drink are stored in a single
flat inventory dict (no separate status dict).

Predicate semantics
-------------------
  UP   (↑) — variable increased since last option start: item collected,
              status improved (e.g. ate food → food↑, health↑)
  DOWN (↓) — variable decreased: status depleted (food↓ means hunger)

The survival variables intentionally use raw float values (0–9) rather than
binary thresholds so the PCG can detect both increase (recovery) and decrease
(depletion) as distinct causal signals.

Crafter observation format
--------------------------
  crafter.Env() returns a raw numpy image from reset()/step().
  The inventory is available in the info dict from step():
    info['inventory'] — dict {item_name: int/float}  (includes health/food/drink)

  Use CrafterInfoWrapper to produce wrapped dict observations:
    {"obs": <image>, "info": {"inventory": {...}, ...}}

Usage
-----
    import crafter
    from dia.evgs_crafter import make_crafter_evgs
    from dia.evgs_adapters import InfoObsWrapper

    env  = CrafterInfoWrapper(crafter.Env())
    evgs = make_crafter_evgs()

    obs = env.reset()          # {"obs": image, "info": {"inventory": {...}}}
    x   = evgs.extract(obs)    # → np.ndarray shape (15,)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from .evgs import EVGS

# ---------------------------------------------------------------------------
# Variable definitions
# ---------------------------------------------------------------------------

VAR_NAMES: List[str] = [
    # Crafting inventory (0-11)
    # Note: Crafter has no smelting step; mining iron gives "iron" directly.
    # There is no separate iron_ore variable.
    "wood",
    "stone",
    "coal",
    "iron",
    "diamond",
    "sapling",
    "wood_pickaxe",
    "stone_pickaxe",
    "iron_pickaxe",
    "wood_sword",
    "stone_sword",
    "iron_sword",
    # Survival status (12-14)
    "health",
    "food",
    "drink",
]

# Indices for quick reference
VAR_INDEX: Dict[str, int] = {name: i for i, name in enumerate(VAR_NAMES)}

# Crafter inventory key → DIA variable name
# Crafter uses underscore_case item names matching the achievement names.
# NOTE: In Crafter, ALL items (including health/food/drink) are stored in a single
# flat inventory dict, not split into inventory + status dicts.
_INVENTORY_MAP: Dict[str, str] = {
    "wood":          "wood",
    "stone":         "stone",
    "coal":          "coal",
    "iron":          "iron",       # Crafter: mining iron block → iron directly (no smelting)
    "diamond":       "diamond",
    "sapling":       "sapling",
    "wood_pickaxe":  "wood_pickaxe",
    "stone_pickaxe": "stone_pickaxe",
    "iron_pickaxe":  "iron_pickaxe",
    "wood_sword":    "wood_sword",
    "stone_sword":   "stone_sword",
    "iron_sword":    "iron_sword",
    # Survival stats: in Crafter these live in inventory (same flat dict)
    "health":        "health",
    "food":          "food",
    "drink":         "drink",
    # "energy" intentionally omitted: correlates with day/night cycle,
    # not causally driven by agent interventions.
}

# _STATUS_MAP is kept for backward compatibility but is empty:
# Crafter has no separate status dict; all fields are in inventory.
_STATUS_MAP: Dict[str, str] = {}

# Ground-truth causal DAG for Crafter (used to compute SHD)
# Each tuple is (cause_var, effect_var) meaning cause_var↑ enables effect_var↑.
CRAFTER_CAUSAL_EDGES: List[tuple] = [
    # Crafting prerequisites (Crafter has no smelting; iron is collected directly)
    # All crafting is done at a placed crafting table (not modelled as a variable here).
    ("wood",          "wood_pickaxe"),    # wood needed to craft wood pickaxe
    ("wood",          "wood_sword"),
    ("wood",          "stone_pickaxe"),   # wood handle for stone pickaxe
    ("wood",          "stone_sword"),
    ("wood",          "iron_pickaxe"),
    ("wood",          "iron_sword"),
    ("stone",         "stone_pickaxe"),   # stone head for stone pickaxe
    ("stone",         "stone_sword"),
    ("iron",          "iron_pickaxe"),    # iron head for iron pickaxe
    ("iron",          "iron_sword"),
    ("wood_pickaxe",  "stone"),           # need wood pickaxe to mine stone
    ("stone_pickaxe", "coal"),            # need stone pickaxe to mine coal
    ("stone_pickaxe", "iron"),            # need stone pickaxe to mine iron
    ("iron_pickaxe",  "diamond"),         # need iron pickaxe to mine diamond
    # Survival causal links
    ("food",          "health"),          # low food → health decreases over time
    ("drink",         "health"),          # dehydration → health decreases over time
]


# ---------------------------------------------------------------------------
# Observation extractor
# ---------------------------------------------------------------------------

def _obs_to_vars(obs: Any) -> np.ndarray:
    """Extract 15-dim variable vector from a Crafter observation.

    Accepts two observation formats:
    1. Wrapped dict: {"obs": <image>, "info": {"inventory": {...}, ...}}
       (produced by InfoObsWrapper + CrafterInfoWrapper)
    2. Raw info dict: {"inventory": {...}}
       (e.g. the info dict directly from env.step())

    Returns zeros when obs is a raw numpy image (no inventory data available).
    """
    x = np.zeros(len(VAR_NAMES), dtype=float)

    if not isinstance(obs, dict):
        # Raw numpy image from crafter.Env.reset() / render() — no state available
        return x

    # Unwrap the {"obs": ..., "info": {...}} format used by InfoObsWrapper
    if "info" in obs and isinstance(obs["info"], dict):
        inventory = obs["info"].get("inventory", {})
    else:
        # Fallback: obs may itself be the info dict (e.g. passed directly)
        inventory = obs.get("inventory", {})

    # -- All Crafter items including health/food/drink are in the inventory dict --
    if isinstance(inventory, dict):
        for crafter_key, dia_name in _INVENTORY_MAP.items():
            val = inventory.get(crafter_key, 0)
            idx = VAR_INDEX.get(dia_name)
            if idx is not None:
                x[idx] = float(val)

    return x


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def make_crafter_evgs(var_names: Optional[List[str]] = None) -> EVGS:
    """
    Build an EVGS adapter for Crafter.

    Args:
        var_names: Override the default 16 variable names.  Rarely needed;
                   pass None to use the standard set.

    Returns:
        EVGS instance ready for use with SimplePCG, SIGraph, and DIARunner.
    """
    names = var_names or VAR_NAMES
    return EVGS(var_names=names, obs_to_vars=_obs_to_vars)


# ---------------------------------------------------------------------------
# Utility: format variable vector as readable summary
# ---------------------------------------------------------------------------

def inventory_summary(obs: Any) -> str:
    """Return a compact string summary of the Crafter variable state."""
    x = _obs_to_vars(obs)
    parts = []
    for i, name in enumerate(VAR_NAMES):
        v = x[i]
        if v > 0:
            if name in ("health", "food", "drink"):
                parts.append(f"{name}:{v:.1f}")
            else:
                parts.append(f"{name}:{int(v)}")
    return "{" + ", ".join(parts) + "}" if parts else "{empty}"
