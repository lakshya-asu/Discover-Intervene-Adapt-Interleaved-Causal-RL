#!/usr/bin/env python3
"""
ROCKET-1 + DIA Hybrid: MineStudio-native implementation of the ROCKET-1 paper
(arXiv 2410.17856), with DIA's PCG as high-level causal planner.

Modes:
  rocket1    -- pure ROCKET-1, fixed skill order (baseline)
  dia        -- ROCKET-1 execution + DIA PCG/SIG ordering (transfer)
  dia_online -- dia + online PCG fine-tuning from 3D experience

Usage:
  conda run -n dia-minecraft python scripts/run_rocket1_minestudio.py \\
      --mode dia --seed 0 --out /tmp/dia_rocket1_s0.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rocket1_minestudio")

# DIA imports
try:
    from dia.evgs_minedojo import make_minedojo_evgs, VAR_NAMES
    from dia.sig import SIGraph, Skill
    from dia.types import Subgoal, Predicate
    from dia.pcg import SimplePCG, PCGConfig
    _DIA_OK = True
except ImportError as _e:
    logger.error("DIA import failed: %s", _e)
    _DIA_OK = False
    VAR_NAMES: List[str] = [
        "wood", "stone", "coal", "ironore", "furnace",
        "stonepickaxe", "iron", "ironpickaxe", "diamond",
    ]

try:
    from dia.sam2_tracker import SAM2Tracker
    _SAM2_IMPORT_OK = True
except ImportError:
    _SAM2_IMPORT_OK = False

try:
    from dia.florence_grounder import FlorenceGrounder
    _FLORENCE_IMPORT_OK = True
except ImportError:
    _FLORENCE_IMPORT_OK = False

try:
    from dia.options_rocket1 import _load_rocket1, _OBJ_ID_MINE, _OBJ_ID_APPROACH
    _ROCKET1_IMPORT_OK = True
except ImportError as _e:
    logger.error("options_rocket1 import failed: %s", _e)
    _ROCKET1_IMPORT_OK = False

# Interaction type IDs (from MineStudio SEGMENT_MAPPING)
_OBJ_ID_CRAFT = 4   # Craft — opens crafting table / inventory grid
_OBJ_ID_USE   = 3   # Use  — interact with placed block (furnace, chest)

# SAM-2 is called only every N frames; the cached mask is reused in between.
# Target blocks are static world objects — a 1.5s-stale mask is fine and cuts
# GPU cost dramatically (30× fewer SAM-2 forward passes).
_SAM2_INTERVAL = 30

# Per-skill ROCKET-1 craft step budget (before /give fallback)
_CRAFT_ROCKET1_STEPS: Dict[str, int] = {
    "woodpickaxe":  300,   # inventory 2x2 + crafting table, simple pattern
    "furnace":      350,   # crafting table required, 3x3 stone pattern
    "stonepickaxe": 350,   # crafting table required
    "iron":         400,   # furnace interaction (Use) + wait for smelt
    "ironpickaxe":  350,   # crafting table required
}

# Skill classification
# woodpickaxe is NOT in DIA's 9-var PCG (to keep PCG transfer clean),
# but IS in the execution chain — it is a hard prerequisite for mining stone
# in 3D Minecraft that the 2D symbolic env abstracts away.
GATHER_SKILLS: frozenset = frozenset({"wood", "stone", "coal", "ironore", "diamond"})
CRAFT_SKILLS:  frozenset = frozenset({
    "woodpickaxe", "furnace", "stonepickaxe", "iron", "ironpickaxe",
})
_VAR_IDX: Dict[str, int] = {n: i for i, n in enumerate(VAR_NAMES)}

_CRAFT_GIVE: Dict[str, str] = {
    "woodpickaxe":  "minecraft:wooden_pickaxe",
    "furnace":      "minecraft:furnace",
    "stonepickaxe": "minecraft:stone_pickaxe",
    "iron":         "minecraft:iron_ingot",
    "ironpickaxe":  "minecraft:iron_pickaxe",
}
# Gather skill /give fallback items.  Used when ROCKET-1 exhausts all attempts
# without collecting the resource (e.g. underground ores not visible on the surface).
# Keeps the planning comparison valid: both DIA and ROCKET-1 modes get the same items;
# the metric is ordering efficiency (steps wasted on prerequisite-missing attempts).
_GATHER_GIVE: Dict[str, str] = {
    "wood":    "minecraft:oak_log",
    "stone":   "minecraft:cobblestone",
    "coal":    "minecraft:coal",
    "ironore": "minecraft:iron_ore",   # raw_iron is 1.17+; MineRL uses 1.16
    "diamond": "minecraft:diamond",
}
# Unified lookup for _give_item (craft + gather)
_ALL_GIVE: Dict[str, str] = {**_CRAFT_GIVE, **_GATHER_GIVE}
_CRAFT_PREREQS: Dict[str, List[str]] = {
    "woodpickaxe":  ["wood"],
    "furnace":      ["stone"],
    "stonepickaxe": ["wood", "stone"],
    "iron":         ["ironore", "coal", "furnace"],
    "ironpickaxe":  ["iron", "wood"],
}

# Tool prerequisites for gather skills — in 3D Minecraft you can't yield cobblestone
# without a pickaxe, iron ore without a stone pickaxe, etc.
# These are enforced by the reactive planner before attempting any gather skill.
_GATHER_PREREQS: Dict[str, List[str]] = {
    "wood":    [],
    "stone":   ["woodpickaxe"],
    "coal":    ["woodpickaxe"],
    "ironore": ["stonepickaxe"],
    "diamond": ["ironpickaxe"],
}

# Block types to watch in info["mine_block"] for post-mine drop collection
_MINE_BLOCK_KEYS: Dict[str, List[str]] = {
    "wood":    ["log", "oak_log", "birch_log", "spruce_log",
                "jungle_log", "acacia_log", "dark_oak_log"],
    "stone":   ["stone", "cobblestone"],
    "coal":    ["coal_ore"],
    "ironore": ["iron_ore"],
    "diamond": ["diamond_ore"],
}


def _all_prereqs(skill_name: str) -> List[str]:
    """Return all hard prerequisites (tool + material) for a skill."""
    return _GATHER_PREREQS.get(skill_name, []) + _CRAFT_PREREQS.get(skill_name, [])


def _pick_next_skill(
    skill_order: List[str],
    achieved: List[str],
    attempted: Dict[str, int],
    max_attempts: int = 2,
) -> Optional[str]:
    """Reactive planner: pick the best next skill to attempt given current state.

    Uses the PCG/SIG topological order as a priority ranking.  Among skills
    whose prerequisites are all satisfied, returns the highest-priority one
    that hasn't exceeded max_attempts.

    If *all* remaining skills have unsatisfied prereqs (e.g. woodpickaxe craft
    failed and stone is blocked), falls back to the first skill with the fewest
    missing prerequisites so the agent tries to unblock itself.

    Returns None only when every skill in skill_order has been achieved or
    exhausted its attempt budget.
    """
    achieved_set = set(achieved)
    unachieved = [s for s in skill_order if s not in achieved_set]

    if not unachieved:
        return None

    # Partition by prereqs met / not met
    ready, blocked = [], []
    for skill in unachieved:
        missing = [p for p in _all_prereqs(skill) if p not in achieved_set]
        if missing:
            blocked.append((skill, len(missing)))
        else:
            ready.append(skill)

    # Among ready skills, pick first under attempt budget
    for skill in ready:
        if attempted.get(skill, 0) < max_attempts:
            return skill

    # All ready skills exhausted — try blocked skill with fewest missing prereqs
    if blocked:
        blocked.sort(key=lambda x: x[1])
        for skill, _ in blocked:
            if attempted.get(skill, 0) < max_attempts:
                logger.warning(
                    "[planner] all ready skills exhausted; attempting blocked skill %s "
                    "(missing: %s)", skill,
                    [p for p in _all_prereqs(skill) if p not in achieved_set],
                )
                return skill

    return None  # every skill achieved or attempt budget exhausted

_REWARDS_CFG = [
    {"event": "pickup",     "identity": "pickup_log",      "reward": 1.0, "max_reward_times": 1,
     "objects": ["log", "oak_log", "birch_log", "spruce_log", "jungle_log", "acacia_log", "dark_oak_log"]},
    {"event": "mine_block", "identity": "mine_log",        "reward": 0.5, "max_reward_times": 1,
     "objects": ["log", "oak_log", "birch_log", "spruce_log"]},
    {"event": "pickup",     "identity": "pickup_cobble",   "reward": 1.0, "max_reward_times": 1,
     "objects": ["cobblestone"]},
    {"event": "pickup",     "identity": "pickup_coal",     "reward": 1.0, "max_reward_times": 1,
     "objects": ["coal"]},
    {"event": "pickup",     "identity": "pickup_iron_ore", "reward": 1.0, "max_reward_times": 1,
     "objects": ["iron_ore", "raw_iron"]},
    {"event": "pickup",     "identity": "pickup_diamond",  "reward": 5.0, "max_reward_times": 1,
     "objects": ["diamond"]},
]

_SKILL_ITEMS: Dict[str, List[str]] = {
    "wood":         ["log", "oak_log", "birch_log", "spruce_log",
                     "jungle_log", "acacia_log", "dark_oak_log"],
    "woodpickaxe":  ["wooden_pickaxe", "minecraft:wooden_pickaxe"],
    "stone":        ["cobblestone", "stone"],
    "coal":         ["coal"],
    "ironore":      ["iron_ore", "raw_iron"],
    "furnace":      ["furnace"],
    "stonepickaxe": ["stone_pickaxe"],
    "iron":         ["iron_ingot"],
    "ironpickaxe":  ["iron_pickaxe"],
    "diamond":      ["diamond"],
}


def build_transfer_sig() -> "SIGraph":
    sig = SIGraph()
    for var_idx, var_name in enumerate(VAR_NAMES):
        sig.add_skill(Skill(skill_id=var_idx,
                            subgoal=Subgoal(var_index=var_idx, predicate=Predicate.UP),
                            name=var_name))
    idx = _VAR_IDX
    sig.add_prerequisite(idx["stone"],        idx["furnace"])
    sig.add_prerequisite(idx["wood"],         idx["stonepickaxe"])
    sig.add_prerequisite(idx["stone"],        idx["stonepickaxe"])
    sig.add_prerequisite(idx["stonepickaxe"], idx["ironore"])
    sig.add_prerequisite(idx["furnace"],      idx["iron"])
    sig.add_prerequisite(idx["coal"],         idx["iron"])
    sig.add_prerequisite(idx["ironore"],      idx["iron"])
    sig.add_prerequisite(idx["iron"],         idx["ironpickaxe"])
    sig.add_prerequisite(idx["wood"],         idx["ironpickaxe"])
    sig.add_prerequisite(idx["ironpickaxe"],  idx["diamond"])
    return sig


def topo_order_from_sig(sig: "SIGraph") -> List[str]:
    try:
        order = sig.toposort()
    except Exception:
        order = list(range(len(VAR_NAMES)))
    names = [VAR_NAMES[i] for i in order]
    # woodpickaxe is not in the 9-var PCG but is a hard 3D prerequisite:
    # stone blocks in vanilla Minecraft require a wooden pickaxe to yield cobblestone.
    # Insert it immediately after wood in the execution order.
    if "wood" in names:
        names.insert(names.index("wood") + 1, "woodpickaxe")
    return names


def baseline_order() -> List[str]:
    """Skill order for the rocket1 baseline (VAR_NAMES order = wrong causal sequence).

    Uses the same VAR_NAMES traversal as the original ROCKET-1 paper but inserts
    woodpickaxe after wood (both modes need the pickaxe to mine stone; the
    planning comparison is about the ordering of the remaining skills).
    """
    names = list(VAR_NAMES)
    if "wood" in names:
        names.insert(names.index("wood") + 1, "woodpickaxe")
    return names


def _inv_has(info: Dict, item_names: List[str]) -> bool:
    """True if any item_name is in inventory with qty >= 1."""
    item_set = {n.lower() for n in item_names}
    inv = info.get("inventory", {})
    if isinstance(inv, dict):
        names = inv.get("name", [])
        qtys  = inv.get("quantity", [])
        if hasattr(names, "__len__") and len(names) > 0:
            for n, q in zip(names, qtys):
                if str(n).strip().lower().rstrip("\x00") in item_set and int(q) > 0:
                    return True
        for v in inv.values():
            if isinstance(v, dict):
                if str(v.get("type", "")).lower() in item_set and int(v.get("quantity", 0)) > 0:
                    return True
    pickup = info.get("pickup", {})
    return any(pickup.get(name, 0) > 0 for name in item_names)


def _skill_achieved(skill_name: str, info: Dict) -> bool:
    return _inv_has(info, _SKILL_ITEMS.get(skill_name, [skill_name]))


def _inventory_summary(info: Dict) -> str:
    found = [s for s in VAR_NAMES if _skill_achieved(s, info)]
    return "[" + ", ".join(found) + "]" if found else "[]"


def _update_pcg_online(
    pcg: "SimplePCG",
    evgs: Any,
    trajectory: List[Tuple],
    skill_var_idx: int,
    lr: float = 0.05,
) -> None:
    if not trajectory:
        return
    j = skill_var_idx
    delta = np.zeros_like(pcg.probs)
    for obs, _act, next_obs, _done in trajectory:
        try:
            x_t   = evgs.extract(obs)
            x_tp1 = evgs.extract(next_obs)
        except Exception:
            continue
        effect = float(x_tp1[j]) - float(x_t[j])
        if abs(effect) < 1e-6:
            continue
        for i in range(len(VAR_NAMES)):
            if i != j and float(x_t[i]) > 0.5:
                delta[i, j] += effect
    pcg.conservative_update(delta / max(1, len(trajectory)), lr=lr)


def _show_minecraft_window() -> None:
    """Map and raise the Minecraft LWJGL window on the current X display.

    LWJGL starts unmapped (hidden) unless a window manager raises it.
    We search for it by name and move it to the right monitor.
    """
    try:
        import subprocess, time
        display_env = os.environ.get("DISPLAY", ":1")
        result = subprocess.run(
            ["xwininfo", "-root", "-children", "-display", display_env],
            capture_output=True, text=True, timeout=5,
        )
        win_id = None
        for line in result.stdout.splitlines():
            if "Minecraft" in line and "0x" in line:
                win_id = line.strip().split()[0]
                break
        if win_id is None:
            logger.debug("_show_minecraft_window: no Minecraft window found yet")
            return
        from Xlib import display as _xdisp
        d = _xdisp.Display(display_env)
        win = d.create_resource_object("window", int(win_id, 16))
        win.map()
        d.sync()
        time.sleep(0.2)
        # Position on DP-4 (offset 3000+323, 2560×1440) — centred
        win.configure(x=3800, y=773, width=1280, height=720)
        win.raise_window()
        d.sync()
        logger.info("Minecraft window %s mapped at 3800,773 (1280×720)", win_id)
    except Exception as exc:
        logger.debug("_show_minecraft_window: %s", exc)


def make_minestudio_env(seed: int) -> Any:
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import RewardsCallback
    from minestudio.simulator.entry import check_engine
    check_engine(skip_confirmation=True)   # auto-download engine if missing
    env = MinecraftSim(
        action_type="agent",
        obs_size=(224, 224),
        preferred_spawn_biome="forest",
        callbacks=[RewardsCallback(_REWARDS_CFG)],
        seed=seed,
    )
    return env


def _run_gather_skill(
    skill_name: str,
    env: Any,
    obs: Dict,
    info: Dict,
    policy: Any,
    memory: Any,
    sam2: Optional[Any],
    max_steps: int,
    device: Any,
    grounder: Optional[Any] = None,
) -> Tuple[bool, int, Dict, Dict, Any, List]:
    """Run ROCKET-1 gather loop for one skill, threading recurrent memory through.

    Seed SAM-2 via Florence-2 open-vocabulary detection when available, falling
    back to a skill-specific heuristic screen fraction.  Florence-2 is only
    called when SAM-2 needs a new seed (skill start or tracking collapse), not
    every frame.

    Mine(2) + SAM-2 mask when mask is non-zero; Approach(6) + zeros mask otherwise.
    Returns (success, steps_taken, obs, info, memory, trajectory).
    """
    import torch
    H, W = obs["image"].shape[:2]
    trajectory: List = []

    # Heuristic fallback seed — approximate screen position of target block.
    _SEED_FRACTIONS: Dict[str, Tuple[float, float]] = {
        "wood":    (0.5, 0.45),   # tree trunks at eye level, center-top half
        "stone":   (0.5, 0.55),   # cliff/cave wall, center
        "coal":    (0.5, 0.50),
        "ironore": (0.5, 0.55),
        "diamond": (0.5, 0.60),   # deep cave walls, lower center
    }
    fx, fy = _SEED_FRACTIONS.get(skill_name, (0.5, 0.5))
    heuristic_seed: Tuple[int, int] = (int(fx * W), int(fy * H))
    # seed_xy is updated each time SAM-2 needs a new seed point
    seed_xy: Tuple[int, int] = heuristic_seed

    # ── Tuning constants ────────────────────────────────────────────────────
    _APPROACH_PATIENCE = 150   # steps of zero SAM-2 mask before forcing Mine
    _COLLECT_STEPS     = 30    # approach steps after each mine_block event
    _AIR_THRESHOLD     = 100   # Minecraft air units (max=300); surface if below
    _SURFACE_STEPS     = 40    # approach steps to escape water
    log_interval = 300

    zero_streak = 0
    # Cumulative mine_block count for this skill (to detect newly broken blocks)
    mine_keys = _MINE_BLOCK_KEYS.get(skill_name, [])
    last_mine_count = sum(
        int(info.get("mine_block", {}).get(k, 0)) for k in mine_keys
    )
    # SAM-2 mask cache — reused between calls to avoid per-frame GPU overhead
    cached_obj_mask = np.zeros((H, W), dtype=np.uint8)

    for step in range(max_steps):
        rgb = obs["image"]

        # ── Survival: escape water if air is running out ─────────────────
        air = int(info.get("life_stats", {}).get("air", 300))
        if air < _AIR_THRESHOLD:
            logger.info("[%s] step=%d LOW AIR (%d) — surfacing for %d steps",
                        skill_name, step, air, _SURFACE_STEPS)
            if sam2 is not None:
                sam2.reset()
            zero_mask = torch.zeros((H, W), dtype=torch.uint8, device=device)
            for _ in range(_SURFACE_STEPS):
                surface_input = {
                    "image": obs["image"],
                    "segment": {
                        "obj_id":   torch.tensor(_OBJ_ID_APPROACH, dtype=torch.int64, device=device),
                        "obj_mask": zero_mask,
                    },
                }
                action_dict, memory = policy.get_action(
                    input=surface_input, state_in=memory, input_shape="*", deterministic=False,
                )
                obs, _, terminated, truncated, info = env.step(action_dict)
                trajectory.append((obs, action_dict, obs, terminated or truncated))
                if terminated or truncated:
                    return False, len(trajectory), obs, info, memory, trajectory
            continue  # resume normal loop after surfacing

        # ── SAM-2 mask (throttled to every _SAM2_INTERVAL frames) ────────
        # Blocks are static — a cached mask from 30 frames ago is still valid.
        # This cuts SAM-2 GPU cost by 30× with negligible tracking quality loss.
        #
        # When SAM-2 needs a new seed (skill start or tracking collapse),
        # try Florence-2 open-vocabulary detection first; fall back to the
        # heuristic fraction if Florence-2 is unavailable or returns nothing.
        if sam2 is not None and step % _SAM2_INTERVAL == 0:
            if grounder is not None and not sam2._initialized:
                fl_pt = grounder.point(rgb, skill_name)
                seed_xy = fl_pt if fl_pt is not None else heuristic_seed
            else:
                seed_xy = heuristic_seed
            cached_obj_mask = sam2.update(rgb, seed_xy=seed_xy)
        obj_mask = cached_obj_mask

        mask_active = bool(obj_mask.sum() > 0)
        if mask_active:
            zero_streak = 0
            active_obj_id = _OBJ_ID_MINE
        else:
            zero_streak += 1
            if zero_streak > _APPROACH_PATIENCE:
                active_obj_id = _OBJ_ID_MINE
                if sam2 is not None and zero_streak % 50 == 0:
                    sam2.reset()
            else:
                active_obj_id = _OBJ_ID_APPROACH

        if step % log_interval == 0:
            mb = {k: v for k, v in info.get("mine_block", {}).items() if int(v) > 0}
            pk = {k: v for k, v in info.get("pickup", {}).items() if int(v) > 0}
            logger.info(
                "[%s] step=%d obj_id=%d mask_px=%d zero_streak=%d air=%d inv=%s"
                " mine_block=%s pickup=%s",
                skill_name, step, active_obj_id, int(obj_mask.sum()),
                zero_streak, air, _inventory_summary(info), mb, pk,
            )

        rocket_input = {
            "image": rgb,
            "segment": {
                "obj_id":   torch.tensor(active_obj_id, dtype=torch.int64).to(device),
                "obj_mask": torch.tensor(obj_mask,      dtype=torch.uint8).to(device),
            },
        }
        action_dict, memory = policy.get_action(
            input=rocket_input, state_in=memory, input_shape="*", deterministic=False,
        )
        obs_prev = obs
        obs, reward, terminated, truncated, info = env.step(action_dict)
        trajectory.append((obs_prev, action_dict, obs, terminated or truncated))

        if _skill_achieved(skill_name, info):
            return True, len(trajectory), obs, info, memory, trajectory
        if terminated or truncated:
            return False, len(trajectory), obs, info, memory, trajectory

        # ── Post-mine drop collection ────────────────────────────────────
        # After a block break, the drop sits on the ground within ~1 block.
        # Mine(2) mode keeps the agent facing the now-empty spot; switch to
        # Approach briefly so the agent drifts over the item to auto-collect.
        current_mine_count = sum(
            int(info.get("mine_block", {}).get(k, 0)) for k in mine_keys
        )
        if current_mine_count > last_mine_count:
            last_mine_count = current_mine_count
            logger.info("[%s] step=%d block broken — %d approach steps to collect drop",
                        skill_name, step, _COLLECT_STEPS)
            if sam2 is not None:
                sam2.reset()          # re-seed next frame on new target
            zero_mask = torch.zeros((H, W), dtype=torch.uint8, device=device)
            for _ in range(_COLLECT_STEPS):
                collect_input = {
                    "image": obs["image"],
                    "segment": {
                        "obj_id":   torch.tensor(_OBJ_ID_APPROACH, dtype=torch.int64, device=device),
                        "obj_mask": zero_mask,
                    },
                }
                action_dict, memory = policy.get_action(
                    input=collect_input, state_in=memory, input_shape="*", deterministic=False,
                )
                obs, _, terminated, truncated, info = env.step(action_dict)
                trajectory.append((obs, action_dict, obs, terminated or truncated))
                if _skill_achieved(skill_name, info):
                    return True, len(trajectory), obs, info, memory, trajectory
                if terminated or truncated:
                    return False, len(trajectory), obs, info, memory, trajectory

    return False, len(trajectory), obs, info, memory, trajectory


def _give_item(skill_name: str, env: Any, obs: Dict, info: Dict) -> Tuple[bool, Dict, Dict]:
    """Execute /give for skill_name (craft or gather). Returns (success, obs, info)."""
    inner = getattr(env, "env", None)
    if inner is not None and hasattr(inner, "execute_cmd"):
        try:
            cmd = f"/give @p {_ALL_GIVE[skill_name]} 1"
            raw_obs, _rw, _dn, raw_info = inner.execute_cmd(cmd)
            obs, info = env._wrap_obs_info(raw_obs, raw_info)
            noop = env.noop_action()
            obs, _, _t, _tr, info = env.step(noop)
            ok = _skill_achieved(skill_name, info)
            if not ok:
                logger.warning(
                    "[%s] /give cmd sent (%s) but item not detected in inventory "
                    "(wrong item name for this MC version?)",
                    skill_name, cmd,
                )
            return ok, obs, info
        except Exception as exc:
            logger.warning("[%s] /give failed: %s", skill_name, exc)
    else:
        logger.warning("[%s] execute_cmd not available", skill_name)
    return False, obs, info


def _run_rocket1_craft_skill(
    skill_name: str,
    env: Any,
    obs: Dict,
    info: Dict,
    policy: Any,
    memory: Any,
    achieved: List[str],
    device: Any,
) -> Tuple[bool, Dict, Dict, Any, str, int]:
    """Try ROCKET-1 Craft/Use interaction, fall back to /give on timeout.

    Parameters
    ----------
    skill_name : str  — one of CRAFT_SKILLS
    policy / memory  — ROCKET-1 policy + recurrent state (updated in-place)
    achieved         — skills already completed this episode

    Returns
    -------
    (success, obs, info, memory, craft_method, steps_taken)
    craft_method in {"rocket1", "give_fallback", "prereqs_missing", "already_had", "failed"}
    """
    import torch

    missing = [p for p in _CRAFT_PREREQS.get(skill_name, []) if p not in achieved]
    if missing:
        logger.info("[%s] prerequisites not met: %s", skill_name, missing)
        return False, obs, info, memory, "prereqs_missing", 0

    if _skill_achieved(skill_name, info):
        return True, obs, info, memory, "already_had", 0

    H, W = obs["image"].shape[:2]
    max_steps = _CRAFT_ROCKET1_STEPS.get(skill_name, 300)
    # iron smelting = interact with furnace (Use/3); everything else = Craft/4
    obj_id = _OBJ_ID_USE if skill_name == "iron" else _OBJ_ID_CRAFT
    zero_mask = torch.zeros((H, W), dtype=torch.uint8, device=device)

    logger.info("[%s] ROCKET-1 craft (%d steps, obj_id=%d)", skill_name, max_steps, obj_id)
    for step in range(max_steps):
        rocket_input = {
            "image": obs["image"],
            "segment": {
                "obj_id":   torch.tensor(obj_id, dtype=torch.int64, device=device),
                "obj_mask": zero_mask,
            },
        }
        action_dict, memory = policy.get_action(
            input=rocket_input, state_in=memory, input_shape="*", deterministic=False,
        )
        obs, _, terminated, truncated, info = env.step(action_dict)
        if _skill_achieved(skill_name, info):
            logger.info("[%s] ROCKET-1 craft succeeded in %d steps", skill_name, step + 1)
            return True, obs, info, memory, "rocket1", step + 1
        if terminated or truncated:
            logger.info("[%s] episode ended during craft at step %d", skill_name, step + 1)
            break

    # ── /give fallback ────────────────────────────────────────────────────
    logger.warning(
        "[%s] ROCKET-1 craft did not succeed in %d steps — /give fallback",
        skill_name, max_steps,
    )
    ok, obs, info = _give_item(skill_name, env, obs, info)
    if ok:
        return True, obs, info, memory, "give_fallback", max_steps
    return False, obs, info, memory, "failed", max_steps


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    np.random.seed(args.seed)

    if not _DIA_OK:
        return {"error": "DIA modules not importable", "mode": args.mode, "seed": args.seed}

    M    = len(VAR_NAMES)
    evgs = make_minedojo_evgs()

    pcg = SimplePCG(PCGConfig(num_vars=M, init_edge_prob=0.05, seed=args.seed))
    if args.mode in ("dia", "dia_online") and args.pcg_path and os.path.exists(args.pcg_path):
        probs = np.load(args.pcg_path).astype(float)
        np.fill_diagonal(probs, 0.0)
        pcg.state.edge_probs = np.clip(probs, 0.0, 1.0)
        off_diag = np.ones((M, M), dtype=bool)
        np.fill_diagonal(off_diag, False)
        logger.info("PCG loaded from %s  (entropy=%.3f, edges>0.5: %d)",
                    args.pcg_path, pcg.entropy(), int((pcg.probs[off_diag] > 0.5).sum()))
    elif args.mode != "rocket1":
        logger.info("PCG file not found (%s) — using uniform prior", args.pcg_path)

    if args.mode == "rocket1":
        skill_order = baseline_order()
    else:
        skill_order = topo_order_from_sig(build_transfer_sig())
    logger.info("Skill order (%s): %s", args.mode, skill_order)

    if args.dry_run:
        print("\n[DRY RUN] Init complete. Skipping env steps.")
        return {
            "mode": args.mode, "seed": args.seed, "dry_run": True,
            "skill_order": skill_order, "pcg_entropy": float(pcg.entropy()),
            "rocket1_ok": _ROCKET1_IMPORT_OK, "sam2_import_ok": _SAM2_IMPORT_OK,
            "status": "dry_run_ok",
        }

    if not _ROCKET1_IMPORT_OK:
        return {"error": "ROCKET-1 not available", "mode": args.mode, "seed": args.seed}

    import torch
    policy = _load_rocket1()
    if policy is None:
        logger.error("ROCKET-1 model failed to load — exiting")
        sys.exit(1)
    device = next(policy.parameters()).device
    logger.info("ROCKET-1 ready on %s", device)

    sam2: Optional[Any] = None
    if _SAM2_IMPORT_OK:
        _dev_str = "cuda" if str(device) != "cpu" else "cpu"
        sam2 = SAM2Tracker.build(device=_dev_str)
        logger.info("SAM-2: %s", "loaded" if sam2 else "unavailable — zeros mask fallback")

    grounder: Optional[Any] = None
    if _FLORENCE_IMPORT_OK:
        _dev_str = "cuda" if str(device) != "cpu" else "cpu"
        grounder = FlorenceGrounder.build(device=_dev_str)
        logger.info(
            "Florence-2: %s",
            "loaded" if grounder else "unavailable — heuristic fallback",
        )

    # MinecraftSim reset is intermittently flaky due to JVM malloc corruption
    # during ResourceManager loading (see logs/mc_*.log: "malloc(): unsorted double
    # linked list corrupted"). Observed ~60% failure rate. Retry up to 3 times,
    # killing orphaned Java processes between attempts.
    # Also kill before attempt 1: a stale process from a prior crashed session would
    # respond with its old mission XML (wrong env spec, missing ObservationFromFullStats)
    # causing pickup stats to always be empty for the entire new run.
    import subprocess as _sp
    _MAX_RESET_TRIES = 3
    env = None
    obs = info = None
    for _reset_attempt in range(1, _MAX_RESET_TRIES + 1):
        logger.info("Creating MinecraftSim (seed=%d, attempt=%d)...", args.seed, _reset_attempt)
        try:
            # Always kill stale Java before creating a new env — a reused JVM
            # serves the old mission XML (missing ObservationFromFullStats) which
            # causes pickup stats to always be 0 for the entire episode.
            _java_kill = _sp.run(
                ["pkill", "-f", "mcprec-6.13.jar"],
                capture_output=True,
            )
            if _java_kill.returncode == 0:
                logger.info("Killed stale mcprec Java process before attempt %d", _reset_attempt)
                time.sleep(3)  # let OS reclaim ports + shared memory
            env = make_minestudio_env(args.seed)
            obs, info = env.reset()
            logger.info("Env reset OK. Inventory: %s", _inventory_summary(info))
            _show_minecraft_window()
            break
        except Exception as exc:
            logger.warning("Reset attempt %d/%d failed: %s", _reset_attempt, _MAX_RESET_TRIES, exc)
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass
            env = None
            if _reset_attempt == _MAX_RESET_TRIES:
                logger.error("All %d reset attempts failed — giving up", _MAX_RESET_TRIES)
                return {"error": str(exc), "mode": args.mode, "seed": args.seed}
            time.sleep(5)

    if obs is None:
        return {"error": "env reset returned None obs", "mode": args.mode, "seed": args.seed}

    memory: Any = None
    global_step = 0
    achieved: List[str] = []
    attempted: Dict[str, int] = {}   # how many times each skill has been tried
    results: Dict[str, Any] = {}
    t_start = time.time()

    print(f"\n{'='*65}")
    print(f"ROCKET-1 + DIA  mode={args.mode}  seed={args.seed}")
    print(f"Skill order (priority): {skill_order}")
    print(f"{'='*65}\n")

    # ── Reactive planner loop ────────────────────────────────────────────────
    # Instead of iterating a fixed list, we ask the planner at each step:
    # "given what I've achieved so far, what's the highest-priority skill
    #  whose prerequisites are satisfied?"  This is DIA's core contribution —
    # causal knowledge (from the PCG/SIG) actively guides execution, not just
    # pre-sorts a static list.
    while global_step < args.max_total_steps:
        # Scan inventory for any skills already in inventory (picked up passively)
        for sn in skill_order:
            if sn not in achieved and _skill_achieved(sn, info):
                logger.info("[planner] %s already in inventory — marking achieved", sn)
                achieved.append(sn)
                results[sn] = {"success": True, "steps": 0,
                               "step_global": global_step, "reason": "already_in_inv"}

        skill_name = _pick_next_skill(skill_order, achieved, attempted, max_attempts=2)
        if skill_name is None:
            logger.info("[planner] all skills achieved or exhausted — done")
            break

        var_idx = _VAR_IDX.get(skill_name, -1)
        missing_prereqs = [p for p in _all_prereqs(skill_name) if p not in set(achieved)]

        # Note: attempted is incremented AFTER the skill call so that zero-step
        # prereqs_missing returns don't consume the attempt budget.  This prevents
        # the Run-10 bug where iron's attempted counter hit max_attempts=2 from
        # two consecutive prereqs_missing skips (ironore not yet detected) before
        # ironore was caught by the already_in_inv scan on the next outer iteration.
        print(f"\n--- [{skill_name}] prev_attempts={attempted.get(skill_name, 0)}  "
              f"global_step={global_step}  inv={_inventory_summary(info)} ---")
        if missing_prereqs:
            logger.warning("[planner] %s has unsatisfied prereqs %s — attempting anyway",
                           skill_name, missing_prereqs)

        if skill_name in GATHER_SKILLS:
            # Gather skills always run — increment before calling.
            attempted[skill_name] = attempted.get(skill_name, 0) + 1
            if sam2 is not None:
                sam2.reset()
            success, steps, obs, info, memory, traj = _run_gather_skill(
                skill_name=skill_name, env=env, obs=obs, info=info,
                policy=policy, memory=memory, sam2=sam2,
                max_steps=args.max_steps_per_skill, device=device,
                grounder=grounder,
            )
            global_step += steps
            if args.mode == "dia_online" and traj and var_idx >= 0:
                _update_pcg_online(pcg, evgs, traj, var_idx, lr=0.05)
                logger.info("[%s] PCG updated (entropy=%.3f)", skill_name, pcg.entropy())
            # /give fallback: if ROCKET-1 exhausted all attempts (e.g. underground ore
            # not visible in the surface-forest world), give the item so the planning
            # ordering comparison remains valid for downstream prerequisite chains.
            if not success and attempted.get(skill_name, 0) >= 2:
                give_ok, obs, info = _give_item(skill_name, env, obs, info)
                if give_ok:
                    success = True
                    logger.info(
                        "[%s] ROCKET-1 gather exhausted — /give %s fallback",
                        skill_name, _GATHER_GIVE.get(skill_name, skill_name),
                    )
        else:
            success, obs, info, memory, craft_method, steps = _run_rocket1_craft_skill(
                skill_name=skill_name, env=env, obs=obs, info=info,
                policy=policy, memory=memory, achieved=achieved, device=device,
            )
            global_step += steps
            # Only count as a real attempt when the skill was actually executed.
            # prereqs_missing (0 steps) and already_had must NOT burn the budget.
            if craft_method not in ("prereqs_missing", "already_had"):
                attempted[skill_name] = attempted.get(skill_name, 0) + 1

        status = "SUCCESS" if success else "FAILED"
        if skill_name in CRAFT_SKILLS:
            extra = f"  method={craft_method}"
        elif success and attempted.get(skill_name, 0) >= 2 and skill_name != "wood":
            extra = "  method=give_fallback"
        else:
            extra = ""
        print(f"    {status}  steps={steps}  total={global_step}"
              f"  inv={_inventory_summary(info)}{extra}")

        rec: Dict[str, Any] = {"success": bool(success), "steps": int(steps),
                               "step_global": int(global_step),
                               "attempt": attempted.get(skill_name, 0)}
        if skill_name in CRAFT_SKILLS:
            rec["craft_method"] = craft_method
        results[skill_name] = rec

        if success and skill_name not in achieved:
            achieved.append(skill_name)

        logger.info("[planner] achieved=%s  remaining=%s",
                    achieved,
                    [s for s in skill_order if s not in set(achieved)])

    t_elapsed = time.time() - t_start
    try:
        env.close()
    except Exception:
        pass

    diamond_reached = "diamond" in achieved
    pcg_entropy_final = float(pcg.entropy())

    print(f"\n{'='*65}")
    print(f"Episode done  t={t_elapsed:.1f}s  steps={global_step}")
    print(f"Achieved ({len(achieved)}/9): {achieved}")
    print(f"diamond_reached={diamond_reached}  pcg_entropy={pcg_entropy_final:.4f}")
    print(f"{'='*65}\n")

    return {
        "mode":               args.mode,
        "seed":               args.seed,
        "skill_order":        skill_order,
        "results":            results,
        "skills_achieved":    achieved,
        "n_achieved":         len(achieved),
        "diamond_reached":    diamond_reached,
        "total_steps":        int(global_step),
        "elapsed_s":          round(t_elapsed, 1),
        "pcg_entropy_final":  pcg_entropy_final,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "ROCKET-1 + DIA Hybrid (MineStudio native)\n"
            "  rocket1    — fixed VAR_NAMES order, ROCKET-1 execution only\n"
            "  dia        — DIA PCG/SIG topo order + ROCKET-1 execution\n"
            "  dia_online — dia + online PCG update from 3D trajectories\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mode",                default="dia",
                    choices=["rocket1", "dia", "dia_online"])
    ap.add_argument("--seed",                type=int, default=0)
    ap.add_argument("--pcg_path",            default="pcg_2d.npy")
    ap.add_argument("--max_steps_per_skill", type=int, default=3000)
    ap.add_argument("--max_total_steps",     type=int, default=50000)
    ap.add_argument("--out",                 default="/tmp/rocket1_minestudio_result.json")
    ap.add_argument("--dry_run",             action="store_true")
    args = ap.parse_args()

    print("run_rocket1_minestudio.py")
    for k in ("mode", "seed", "pcg_path", "max_steps_per_skill",
              "max_total_steps", "out", "dry_run"):
        print(f"  {k:24s}: {getattr(args, k)}")
    print()

    result = run_experiment(args)

    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result saved -> {out_path}")
    print(f"n_achieved={result.get('n_achieved', 0)}/9  "
          f"diamond_reached={result.get('diamond_reached', False)}")


if __name__ == "__main__":
    main()
