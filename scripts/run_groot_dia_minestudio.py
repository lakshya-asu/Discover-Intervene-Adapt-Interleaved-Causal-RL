#!/usr/bin/env python3
"""
DIA + GROOT Hybrid: MineStudio-native implementation using GROOT (goal-conditioned IL)
as the execution backbone and DIA's PCG as the causal planner.

Architecture:
  DIA PCG    — learned causal prerequisite graph → picks next skill to attempt
  GrootPolicy — goal-conditioned IL, conditioned on a reference video clip per skill

Modes:
  groot      — pure GROOT, fixed skill order (baseline)
  dia        — DIA PCG/SIG topo order + GROOT execution
  dia_online — dia + online PCG fine-tuning from trajectories

Usage:
  conda run -n dia-minecraft python scripts/run_groot_dia_minestudio.py \\
      --mode dia --seed 0 --out /tmp/dia_groot_s0.json

Key difference from run_rocket1_minestudio.py:
  - No SAM-2 / Florence-2 (GROOT uses reference video for visual goal)
  - No /give fallback for gather skills (GROOT must succeed on its own)
  - /give fallback kept for craft skills to maintain planning comparison validity
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
logger = logging.getLogger("groot_dia_minestudio")

# ---------------------------------------------------------------------------
# DIA imports
# ---------------------------------------------------------------------------
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
    from dia.groot_executor import GrootExecutor
    _GROOT_IMPORT_OK = True
except ImportError as _e:
    logger.error("GrootExecutor import failed: %s", _e)
    _GROOT_IMPORT_OK = False

# ---------------------------------------------------------------------------
# Skill taxonomy (same as run_rocket1_minestudio.py)
# ---------------------------------------------------------------------------
GATHER_SKILLS: frozenset = frozenset({"wood", "stone", "coal", "ironore", "diamond"})
# Underground gather skills need a staircase-descent primer (dark cave BC frames).
# Surface skills only need a horizontal yaw sweep.
UNDERGROUND_GATHER: frozenset = frozenset({"coal", "ironore", "diamond"})
CRAFT_SKILLS: frozenset = frozenset({
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

_CRAFT_PREREQS: Dict[str, List[str]] = {
    # woodpickaxe needs 3 planks + 2 sticks → 2 logs minimum; require 3 to be safe
    "woodpickaxe":  ["wood"],
    "furnace":      ["stone"],
    "stonepickaxe": ["wood", "stone"],
    "iron":         ["ironore", "coal", "furnace"],
    "ironpickaxe":  ["iron", "wood"],
}

_GATHER_PREREQS: Dict[str, List[str]] = {
    # Tool-tier unlock hierarchy:
    #   tier-0 (bare hands): wood
    #   tier-1 (wooden pickaxe): stone, coal
    #   tier-2 (stone pickaxe): ironore
    #   tier-3 (iron pickaxe): diamond
    # Stone and coal also transitively need wood (to craft the wooden pickaxe),
    # but woodpickaxe is listed as the direct prerequisite here.
    "wood":    [],
    "stone":   ["woodpickaxe"],
    "coal":    ["woodpickaxe"],
    "ironore": ["stonepickaxe"],
    "diamond": ["ironpickaxe"],
}

# Minimum quantities to gather before declaring a gather skill "done".
# These reflect downstream craft requirements:
#   wood   → 3 logs: 12 planks → enough handles for all pickaxes + buffer
#            (was 5, but 2883-step over-farming wasted budget — 3 is sufficient)
#   stone  → 11 cobblestone: furnace (8) + stonepickaxe (3); mostly found as
#            byproduct so min_qty rarely limits explicit stone attempts
#   ironore→ 3 ore → 3 ingots → iron_pickaxe (3 ingots needed)
#   coal   → 1 is fine (one coal = 8 smelt operations)
#   diamond→ rare; 1 is the target
_GATHER_MIN_QTY: Dict[str, int] = {
    "wood":    3,
    "stone":   11,
    "coal":    1,
    "ironore": 3,
    "diamond": 1,
}

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

# ---------------------------------------------------------------------------
# Skill clips — loaded at module level from manifest.json
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT_DEFAULT = os.path.join(_SCRIPT_DIR, "..")

_manifest_path = os.path.join(_SCRIPT_DIR, "..", "skill_clips", "manifest.json")
if os.path.exists(_manifest_path):
    with open(_manifest_path) as _f:
        SKILL_CLIPS: Dict[str, List[str]] = json.load(_f)
    logger.info("skill_clips/manifest.json loaded (%d skills)", len(SKILL_CLIPS))
else:
    SKILL_CLIPS = {}
    logger.warning("skill_clips/manifest.json not found — GROOT will use a fallback")


def get_clip_path(skill_name: str, project_root: str) -> Optional[str]:
    """Return absolute path to first available clip for skill_name."""
    clips = SKILL_CLIPS.get(skill_name, [])
    for rel_path in clips:
        abs_path = os.path.join(project_root, rel_path)
        if os.path.isfile(abs_path):
            return abs_path
    return None


# ---------------------------------------------------------------------------
# Prerequisite helpers (copied verbatim from run_rocket1_minestudio.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Inventory helpers (copied verbatim from run_rocket1_minestudio.py)
# ---------------------------------------------------------------------------

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


def _inv_count(info: Dict, item_names: List[str]) -> int:
    """Total inventory count across all matching item_names."""
    item_set = {n.lower() for n in item_names}
    total = 0
    inv = info.get("inventory", {})
    if isinstance(inv, dict):
        names = inv.get("name", [])
        qtys  = inv.get("quantity", [])
        if hasattr(names, "__len__") and len(names) > 0:
            for n, q in zip(names, qtys):
                if str(n).strip().lower().rstrip("\x00") in item_set:
                    total += max(0, int(q))
        for v in inv.values():
            if isinstance(v, dict):
                if str(v.get("type", "")).lower() in item_set:
                    total += max(0, int(v.get("quantity", 0)))
    return total


def _skill_achieved(skill_name: str, info: Dict) -> bool:
    return _inv_has(info, _SKILL_ITEMS.get(skill_name, [skill_name]))


def _inventory_summary(info: Dict) -> str:
    found = [s for s in VAR_NAMES if _skill_achieved(s, info)]
    return "[" + ", ".join(found) + "]" if found else "[]"


# ---------------------------------------------------------------------------
# /give fallback for CRAFT skills only
# ---------------------------------------------------------------------------

def _give_item(skill_name: str, env: Any, obs: Dict, info: Dict) -> Tuple[bool, Dict, Dict]:
    """Execute /give for craft skill_name.  Returns (success, obs, info)."""
    inner = getattr(env, "env", None)
    if inner is not None and hasattr(inner, "execute_cmd"):
        try:
            cmd = f"/give @p {_CRAFT_GIVE[skill_name]} 1"
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


def _apply_experiment_setup(env: Any, obs: Dict, info: Dict) -> Tuple[Dict, Dict]:
    """Apply persistent experiment-wide settings after each env reset.

    - Hunger disabled: saturation effect for 1,000,000 s (≈11.5 days) so the
      agent never slows down or dies from starvation mid-experiment.
    - Keeps time at day: prevents hostile mob spawning from interfering with
      surface skill collection (wood, stone).

    Safe to call even if execute_cmd is unavailable — degrades gracefully.
    """
    inner = getattr(env, "env", None)
    if inner is None or not hasattr(inner, "execute_cmd"):
        logger.debug("_apply_experiment_setup: execute_cmd unavailable, skipping")
        return obs, info

    cmds = [
        # Permanent saturation — hunger bar freezes, no exhaustion
        "/effect give @p minecraft:saturation 1000000 255 true",
        # Water-breathing — prevents drowning in underground streams/flooded caves
        "/effect give @p minecraft:water_breathing 1000000 255 true",
        # Fire resistance — prevents burning in lava pockets underground
        "/effect give @p minecraft:fire_resistance 1000000 255 true",
        # Keep time locked to day (6000 = noon); prevents hostile mob spawns
        "/time set 6000",
        "/gamerule doDaylightCycle false",
        # Give 64 torches — placed during underground primer for ironore/diamond
        "/give @p minecraft:torch 64",
    ]
    for cmd in cmds:
        try:
            raw_obs, _rw, _dn, raw_info = inner.execute_cmd(cmd)
            obs, info = env._wrap_obs_info(raw_obs, raw_info)
            logger.debug("experiment setup cmd OK: %s", cmd)
        except Exception as exc:
            logger.warning("experiment setup cmd failed (%s): %s", cmd, exc)

    # One noop step to let effects register
    try:
        noop = env.noop_action()
        obs, _, _t, _tr, info = env.step(noop)
    except Exception as exc:
        logger.warning("experiment setup noop step failed: %s", exc)

    logger.info("Experiment setup applied: hunger/water/fire effects, time locked to day, torches given.")
    return obs, info


# ---------------------------------------------------------------------------
# PCG online update — inventory-state causal learning
# ---------------------------------------------------------------------------

def _inv_state_vector(info: Dict) -> np.ndarray:
    """Binary vector: x[i]=1 if VAR_NAMES[i] item is in current inventory."""
    return np.array([
        float(_skill_achieved(name, info))
        for name in VAR_NAMES
    ], dtype=float)


def _update_pcg_from_inventory(
    pcg: "SimplePCG",
    skill_var_idx: int,
    inv_before: np.ndarray,
    success: bool,
    lr: float = 0.05,
) -> None:
    """Update PCG edges from a single (inventory_state, skill_outcome) observation.

    Causal learning rules:
      Success: items present in inv_before are potential enablers of skill j
               → strengthen edge i→j for all i where inv_before[i] = 1.
      Failure: items absent from inv_before are candidate missing prerequisites
               → small negative signal for i→j where inv_before[i] = 0,
               meaning "not having i correlates with failing j".

    This mirrors the DIA "Intervene" step: each skill attempt is an intervention,
    and the inventory before the attempt is the context.  Over many attempts the
    PCG converges toward the true tool-tier causal graph.
    """
    j = skill_var_idx
    if j < 0:
        return
    M = len(VAR_NAMES)
    delta = np.zeros((M, M), dtype=float)

    if success:
        for i, had_it in enumerate(inv_before):
            if i != j and had_it > 0.5:
                # Had item i → skill j succeeded: positive causal evidence
                delta[i, j] += 1.0
    else:
        for i, had_it in enumerate(inv_before):
            if i != j and had_it < 0.5:
                # Lacked item i → skill j failed: weak prerequisite signal
                delta[i, j] -= 0.2

    pcg.conservative_update(delta / M, lr=lr)
    logger.info(
        "[pcg] online update: skill=%s success=%s inv=%s → Δmax=%.4f entropy=%.3f",
        VAR_NAMES[j], success,
        [VAR_NAMES[i] for i, v in enumerate(inv_before) if v > 0.5],
        float(np.abs(delta).max()),
        pcg.entropy(),
    )


def _update_pcg_online(
    pcg: "SimplePCG",
    evgs: Any,
    trajectory: List[Tuple],
    skill_var_idx: int,
    lr: float = 0.05,
) -> None:
    """Legacy trajectory-based PCG update (kept for dia_online compatibility)."""
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


# ---------------------------------------------------------------------------
# Env creation and window management (copied verbatim from run_rocket1_minestudio.py)
# ---------------------------------------------------------------------------

def make_minestudio_env(seed: int) -> Any:
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import RewardsCallback
    from minestudio.simulator.entry import check_engine
    check_engine(skip_confirmation=True)
    env = MinecraftSim(
        action_type="agent",
        obs_size=(224, 224),
        preferred_spawn_biome="forest",
        callbacks=[RewardsCallback(_REWARDS_CFG)],
        seed=seed,
    )
    return env


def _raise_minecraft_window_once() -> Optional[str]:
    """Find and raise every Minecraft window.  Returns the last win_id found or None."""
    try:
        import subprocess
        from Xlib import display as _xdisp, X
        from Xlib.protocol import event as _xevent

        display_env = os.environ.get("DISPLAY", ":1")
        result = subprocess.run(
            ["xwininfo", "-root", "-children", "-display", display_env],
            capture_output=True, text=True, timeout=5,
        )
        win_ids = [
            line.strip().split()[0]
            for line in result.stdout.splitlines()
            if "Minecraft" in line and "0x" in line
        ]
        if not win_ids:
            return None

        d = _xdisp.Display(display_env)
        root = d.screen().root
        NET_ACTIVE = d.intern_atom("_NET_ACTIVE_WINDOW")
        for win_id in win_ids:
            win = d.create_resource_object("window", int(win_id, 16))
            win.map()
            d.sync()
            win.configure(x=0, y=0, width=1280, height=720)
            d.sync()
            ev = _xevent.ClientMessage(
                window=win,
                client_type=NET_ACTIVE,
                data=(32, [2, X.CurrentTime, 0, 0, 0]),
            )
            root.send_event(ev, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
        d.sync()
        return win_ids[-1]
    except Exception as exc:
        logger.debug("_raise_minecraft_window_once: %s", exc)
        return None


# Module-level flag so only one keeper thread runs per process.
_WINDOW_KEEPER_RUNNING = False


def _show_minecraft_window(interval: int = 30) -> None:
    """Map, resize, and persistently keep all Minecraft windows on top.

    Raises every Minecraft window immediately, then spawns a daemon thread
    that re-raises them every *interval* seconds for the lifetime of the
    process.  Safe to call multiple times — only one keeper thread is started.
    """
    import threading, time

    # Immediate raise (poll up to 5 s for window to appear at episode start)
    win_id = None
    for _ in range(10):
        win_id = _raise_minecraft_window_once()
        if win_id:
            break
        time.sleep(0.5)

    if win_id:
        logger.info("Minecraft window(s) raised to 0,0 (1280x720)")
    else:
        logger.debug("_show_minecraft_window: no Minecraft window found at startup")

    # Persistent keeper — re-raises whenever another window covers the game
    global _WINDOW_KEEPER_RUNNING
    if _WINDOW_KEEPER_RUNNING:
        return
    _WINDOW_KEEPER_RUNNING = True

    def _keeper():
        while True:
            time.sleep(interval)
            try:
                _raise_minecraft_window_once()
            except Exception:
                pass

    t = threading.Thread(target=_keeper, daemon=True, name="mc-window-keeper")
    t.start()
    logger.info("Minecraft window keeper started (re-raises every %ds)", interval)


# ---------------------------------------------------------------------------
# Video recording
# ---------------------------------------------------------------------------

class _VideoRecordingEnv:
    """Thin env wrapper that appends obs frames to a list after every step."""

    def __init__(self, env: Any, frame_buffer: List) -> None:
        self._env = env
        self._buf = frame_buffer

    def step(self, action: Any) -> Any:
        result = self._env.step(action)
        obs = result[0]
        if isinstance(obs, dict) and "image" in obs:
            self._buf.append(obs["image"].copy())
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


def _save_video(frames: List, out_path: str, fps: int = 20) -> bool:
    """Save a list of uint8 (H,W,3) frames as an MP4.

    Tries cv2 first (fast), falls back to imageio if cv2 is unavailable.
    Returns True on success, False otherwise.
    """
    if not frames:
        logger.warning("_save_video: no frames to save")
        return False
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    try:
        import cv2
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        logger.info("Video saved: %s  (%d frames @ %dfps)", out_path, len(frames), fps)
        return True
    except Exception as exc:
        logger.debug("cv2 video save failed (%s) — trying imageio", exc)
    try:
        import imageio
        imageio.mimwrite(out_path, frames, fps=fps, quality=7)
        logger.info("Video saved (imageio): %s  (%d frames @ %dfps)", out_path, len(frames), fps)
        return True
    except Exception as exc:
        logger.warning("_save_video failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# SIG / PCG ordering helpers (copied verbatim from run_rocket1_minestudio.py)
# ---------------------------------------------------------------------------

def build_transfer_sig() -> "SIGraph":
    """Build the transfer SIG encoding the Minecraft tech-tree causal structure.

    Tool-tier hierarchy (what the PCG needs to encode):
      Tier 0 — bare hands: wood
      Tier 1 — wooden pickaxe (crafted from wood): stone, coal
        wood → [woodpickaxe] → stone
        wood → [woodpickaxe] → coal
      Tier 2 — stone pickaxe (crafted from stone + wood): ironore
        stone + wood → [stonepickaxe] → ironore
      Tier 3 — iron pickaxe (crafted from iron + wood): diamond
        iron + wood → [ironpickaxe] → diamond

    woodpickaxe is not a VAR_NAME (it's inserted separately in topo order),
    so its tier-1 unlock is encoded transitively via wood → {stone, coal}.
    """
    sig = SIGraph()
    for var_idx, var_name in enumerate(VAR_NAMES):
        sig.add_skill(Skill(skill_id=var_idx,
                            subgoal=Subgoal(var_index=var_idx, predicate=Predicate.UP),
                            name=var_name))
    idx = _VAR_IDX

    # Tier-0 → Tier-1: wood enables stone and coal (via wooden pickaxe)
    sig.add_prerequisite(idx["wood"],         idx["stone"])   # transitive: wood→wpk→stone
    sig.add_prerequisite(idx["wood"],         idx["coal"])    # transitive: wood→wpk→coal

    # Tier-1 → Tier-2: stone + wood enables stonepickaxe → ironore
    sig.add_prerequisite(idx["wood"],         idx["stonepickaxe"])
    sig.add_prerequisite(idx["stone"],        idx["stonepickaxe"])
    sig.add_prerequisite(idx["stonepickaxe"], idx["ironore"])

    # Crafting causal links
    sig.add_prerequisite(idx["stone"],        idx["furnace"])
    sig.add_prerequisite(idx["coal"],         idx["iron"])    # fuel for smelting
    sig.add_prerequisite(idx["ironore"],      idx["iron"])
    sig.add_prerequisite(idx["furnace"],      idx["iron"])

    # Tier-2 → Tier-3: iron pickaxe → diamond
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
    if "wood" in names:
        names.insert(names.index("wood") + 1, "woodpickaxe")
    return names


def baseline_order() -> List[str]:
    """Skill order for the groot baseline (VAR_NAMES order, woodpickaxe inserted after wood)."""
    names = list(VAR_NAMES)
    if "wood" in names:
        names.insert(names.index("wood") + 1, "woodpickaxe")
    return names


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    np.random.seed(args.seed)

    project_root = os.path.abspath(args.project_root)

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
    elif args.mode != "groot":
        logger.info("PCG file not found (%s) — using uniform prior", args.pcg_path)

    if args.mode == "groot":
        skill_order = baseline_order()
    else:
        skill_order = topo_order_from_sig(build_transfer_sig())
    logger.info("Skill order (%s): %s", args.mode, skill_order)

    if args.dry_run:
        print("\n[DRY RUN] Init complete. Skipping env steps.")
        return {
            "mode": args.mode, "seed": args.seed, "dry_run": True,
            "skill_order": skill_order, "pcg_entropy": float(pcg.entropy()),
            "groot_import_ok": _GROOT_IMPORT_OK,
            "status": "dry_run_ok",
        }

    if not _GROOT_IMPORT_OK:
        return {"error": "GrootExecutor not available", "mode": args.mode, "seed": args.seed}

    executor = GrootExecutor.build(device="cuda", use_ft_ckpt=not args.no_ft_ckpt)
    if executor is None:
        logger.error("GrootExecutor failed to load — exiting")
        sys.exit(1)
    logger.info("GrootExecutor ready.")

    import subprocess as _sp
    _MAX_RESET_TRIES = 3
    env = None
    obs = info = None
    for _reset_attempt in range(1, _MAX_RESET_TRIES + 1):
        logger.info("Creating MinecraftSim (seed=%d, attempt=%d)...", args.seed, _reset_attempt)
        try:
            _java_kill = _sp.run(
                ["pkill", "-f", "mcprec-6.13.jar"],
                capture_output=True,
            )
            if _java_kill.returncode == 0:
                logger.info("Killed stale mcprec Java process before attempt %d", _reset_attempt)
                time.sleep(3)
            env = make_minestudio_env(args.seed)
            obs, info = env.reset()
            logger.info("Env reset OK. Inventory: %s", _inventory_summary(info))
            obs, info = _apply_experiment_setup(env, obs, info)
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

    # Optionally wrap env with video recorder
    frame_buffer: List = []
    if getattr(args, "video_dir", None):
        env = _VideoRecordingEnv(env, frame_buffer)
        logger.info("Video recording enabled → %s", args.video_dir)

    # Reset GROOT recurrent memory once per episode
    executor.reset()

    global_step = 0
    achieved: List[str] = []
    attempted: Dict[str, int] = {}
    results: Dict[str, Any] = {}
    t_start = time.time()

    print(f"\n{'='*65}")
    print(f"DIA + GROOT  mode={args.mode}  seed={args.seed}")
    print(f"Skill order (priority): {skill_order}")
    print(f"{'='*65}\n")

    # ── Reactive planner loop ────────────────────────────────────────────────
    while global_step < args.max_total_steps:
        # Scan inventory for any skills already collected passively
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

        print(f"\n--- [{skill_name}] prev_attempts={attempted.get(skill_name, 0)}  "
              f"global_step={global_step}  inv={_inventory_summary(info)} ---")
        if missing_prereqs:
            logger.warning("[planner] %s has unsatisfied prereqs %s — attempting anyway",
                           skill_name, missing_prereqs)

        # ── Determine clip ───────────────────────────────────────────────────
        clip_path = get_clip_path(skill_name, project_root)
        if clip_path is None:
            logger.warning("[%s] No reference clip found — skipping (not counted as attempt)",
                           skill_name)
            results[skill_name] = {
                "success": False, "steps": 0,
                "step_global": global_step, "attempt": attempted.get(skill_name, 0),
                "reason": "no_clip",
            }
            continue

        # Increment attempt counter only when GROOT will actually run
        attempted[skill_name] = attempted.get(skill_name, 0) + 1

        # ── Run GROOT for this skill ─────────────────────────────────────────
        # Gather skills get a navigation primer before GROOT takes over.
        # Surface skills (wood, stone): horizontal yaw sweep to find trees/outcrops.
        # Underground skills (coal, ironore, diamond): diagonal staircase descent
        #   (~150 steps at 45° pitch, forced attack+forward) then wall sweep.
        #   BC demonstrations for these skills start in dark caves (mean brightness
        #   ~13/255) so the agent must descend before GROOT's visual goal-following
        #   becomes effective.
        is_gather = skill_name in GATHER_SKILLS
        primer_type = "underground" if skill_name in UNDERGROUND_GATHER else "surface"
        primer_n = (200 if primer_type == "underground" else args.primer_steps) if is_gather else 0

        # Snapshot inventory state before attempt (for online causal learning)
        inv_before = _inv_state_vector(info)

        # Minimum quantity to farm — keeps agent mining after first pickup
        # rather than exiting the cave to dig a new one later.
        min_qty = _GATHER_MIN_QTY.get(skill_name, 1) if is_gather else 1
        qty_before = _inv_count(info, _SKILL_ITEMS.get(skill_name, [skill_name]))
        logger.info("[%s] starting attempt=%d  qty_in_inv=%d  target=%d",
                    skill_name, attempted[skill_name], qty_before, min_qty)

        success, steps, obs, info, _groot_mem = executor.run_skill(
            skill_name=skill_name,
            clip_path=clip_path,
            env=env,
            obs=obs,
            info=info,
            max_steps=args.max_steps_per_skill,
            primer_steps=primer_n,
            primer_type=primer_type,
            min_qty=min_qty,
        )
        global_step += steps

        # ── Online causal learning (all modes) ──────────────────────────────
        # Update PCG from the (inventory_before, skill_outcome) observation.
        # This implements the DIA "Intervene → Adapt" loop: each skill attempt
        # is a causal intervention and we update edge beliefs accordingly.
        # woodpickaxe is not a VAR_NAME, so its causal role propagates through
        # the wood→stone/coal transitive edges added to the SIG.
        if var_idx >= 0:
            _update_pcg_from_inventory(pcg, var_idx, inv_before, success, lr=0.03)
            logger.info("[%s] PCG entropy=%.3f", skill_name, pcg.entropy())

        # ── Craft-only /give fallback (NOT for gather skills) ────────────────
        craft_method = "groot"
        if not success and skill_name in CRAFT_SKILLS:
            give_ok, obs, info = _give_item(skill_name, env, obs, info)
            if give_ok:
                success = True
                craft_method = "give_fallback"
                logger.info("[%s] GROOT craft did not succeed — /give fallback used", skill_name)
            else:
                craft_method = "failed"

        status = "SUCCESS" if success else "FAILED"
        if skill_name in CRAFT_SKILLS:
            extra = f"  method={craft_method}"
        else:
            extra = ""
        print(f"    {status}  steps={steps}  total={global_step}"
              f"  inv={_inventory_summary(info)}{extra}")

        rec: Dict[str, Any] = {
            "success": bool(success),
            "steps": int(steps),
            "step_global": int(global_step),
            "attempt": attempted[skill_name],
        }
        if skill_name in CRAFT_SKILLS:
            rec["craft_method"] = craft_method
        results[skill_name] = rec

        if success and skill_name not in achieved:
            achieved.append(skill_name)

        logger.info("[planner] achieved=%s  remaining=%s",
                    achieved,
                    [s for s in skill_order if s not in set(achieved)])

    t_elapsed = time.time() - t_start

    # Save video if recording was enabled
    if getattr(args, "video_dir", None) and frame_buffer:
        tag = f"it_s{args.seed}_{args.mode}"
        video_path = os.path.join(args.video_dir, f"{tag}.mp4")
        _save_video(frame_buffer, video_path, fps=20)

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
        "mode":              args.mode,
        "seed":              args.seed,
        "skill_order":       skill_order,
        "results":           results,
        "skills_achieved":   achieved,
        "n_achieved":        len(achieved),
        "diamond_reached":   diamond_reached,
        "total_steps":       int(global_step),
        "elapsed_s":         round(t_elapsed, 1),
        "pcg_entropy_final": pcg_entropy_final,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "DIA + GROOT Hybrid (MineStudio native)\n"
            "  groot      — fixed VAR_NAMES order, GROOT execution only\n"
            "  dia        — DIA PCG/SIG topo order + GROOT execution\n"
            "  dia_online — dia + online PCG update from trajectories\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mode",                default="dia",
                    choices=["groot", "dia", "dia_online"])
    ap.add_argument("--seed",                type=int, default=0)
    ap.add_argument("--pcg_path",            default="pcg_2d.npy")
    ap.add_argument("--max_steps_per_skill", type=int, default=2000)
    ap.add_argument("--max_total_steps",     type=int, default=40000)
    ap.add_argument("--out",                 default="/tmp/groot_minestudio_result.json")
    ap.add_argument("--dry_run",             action="store_true")
    ap.add_argument("--no_ft_ckpt",          action="store_true",
                    help="Use original pretrained GROOT weights (skip fine-tuned ckpt)")
    ap.add_argument("--primer_steps",        type=int, default=80,
                    help="Camera-sweep navigation steps before GROOT per gather attempt")
    ap.add_argument("--project_root",        default=_PROJECT_ROOT_DEFAULT)
    ap.add_argument("--video_dir",           default=None,
                    help="Directory to save per-run MP4 videos (disabled if not set)")
    args = ap.parse_args()

    print("run_groot_dia_minestudio.py")
    for k in ("mode", "seed", "pcg_path", "max_steps_per_skill",
              "max_total_steps", "no_ft_ckpt", "primer_steps",
              "out", "dry_run", "project_root", "video_dir"):
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
