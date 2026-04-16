#!/usr/bin/env python3
"""
Full tech-tree playthrough demo recorder.

Records human demonstrations of every skill in the DIA Minecraft tech tree,
one skill at a time, in the correct causal order. Each segment is saved as:

  <out_dir>/<skill>/<tag>.mp4               — reference video clip (GROOT conditioning)
  <out_dir>/<skill>/<tag>_bc.npz           — obs + action arrays (BC fine-tuning)
  <out_dir>/<skill>/<tag>_annotations.json — frame-level bookmarks (optional)

The player completes each skill in a single live session. Between skills the
environment is reset and pre-loaded with the required prerequisites (crafted
tools given via /give) so the player can focus on the skill being demonstrated.

Controls:
  W/A/S/D        — move
  Space          — jump
  Left Shift     — sneak
  Left Ctrl      — sprint
  Left mouse     — attack / break block
  Right mouse    — use / interact / place
  E              — inventory
  1-9            — hotbar slot
  Q              — drop item
  ---
  F5             — drop annotation bookmark at current frame (can press multiple times)
  F10            — start recording this skill's segment
  F11            — stop + save, advance to next skill
  F12            — quit

Annotation bookmarks (F5):
  Each press logs {"frame": N, "time_sec": T, "label": "mark_N"} to a JSON sidecar.
  Useful for marking moments like "approaching ore", "initiating attack", "first pickup".
  The JSON is saved alongside the MP4 and BC npz for post-processing / clip slicing.

Usage:
  conda run -n dia-minecraft python scripts/record_playthrough_demos.py \\
      --out_dir data/playthrough_demos \\
      --seed 0

  # Resume from a specific skill if a session was interrupted:
  conda run -n dia-minecraft python scripts/record_playthrough_demos.py \\
      --out_dir data/playthrough_demos \\
      --seed 0 --start_skill ironore
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("record_playthrough")

# ---------------------------------------------------------------------------
# Tech-tree skill sequence with prerequisites to /give before each demo
# ---------------------------------------------------------------------------

# Full ordered sequence matching DIA's topo order
SKILL_SEQUENCE: List[str] = [
    "wood",
    "woodpickaxe",
    "stone",
    "coal",
    "furnace",
    "stonepickaxe",
    "ironore",
    "iron",
    "ironpickaxe",
    "diamond",
]

# Prerequisites to /give at the start of each skill's demo so the player
# starts with the right tools and materials.
# Also sets the right pickaxe into hotbar slot 1 for mining skills.
_SKILL_SETUP: Dict[str, List[str]] = {
    "wood": [
        # Bare hands — no prerequisites
    ],
    "woodpickaxe": [
        "/give @p minecraft:oak_log 8",
    ],
    "stone": [
        "/give @p minecraft:wooden_pickaxe 1",
        "/item replace entity @p hotbar.0 minecraft:wooden_pickaxe 1",
        "/replaceitem entity @p slot.hotbar.0 minecraft:wooden_pickaxe 1",  # 1.16 fallback
    ],
    "coal": [
        "/give @p minecraft:stone_pickaxe 1",
        "/item replace entity @p hotbar.0 minecraft:stone_pickaxe 1",
        "/replaceitem entity @p slot.hotbar.0 minecraft:stone_pickaxe 1",
    ],
    "furnace": [
        "/give @p minecraft:cobblestone 16",
    ],
    "stonepickaxe": [
        "/give @p minecraft:cobblestone 8",
        "/give @p minecraft:oak_log 4",
    ],
    "ironore": [
        "/give @p minecraft:stone_pickaxe 1",
        "/item replace entity @p hotbar.0 minecraft:stone_pickaxe 1",
        "/replaceitem entity @p slot.hotbar.0 minecraft:stone_pickaxe 1",
        "/give @p minecraft:torch 32",
    ],
    "iron": [
        "/give @p minecraft:iron_ore 4",
        "/give @p minecraft:coal 4",
        "/give @p minecraft:furnace 1",
    ],
    "ironpickaxe": [
        "/give @p minecraft:iron_ingot 6",
        "/give @p minecraft:oak_log 4",
    ],
    "diamond": [
        "/give @p minecraft:iron_pickaxe 1",
        "/item replace entity @p hotbar.0 minecraft:iron_pickaxe 1",
        "/replaceitem entity @p slot.hotbar.0 minecraft:iron_pickaxe 1",
        "/give @p minecraft:torch 32",
    ],
}

# Common experiment setup applied to every skill reset
_COMMON_SETUP: List[str] = [
    "/effect give @p minecraft:saturation 1000000 255 true",
    "/effect give @p minecraft:water_breathing 1000000 255 true",
    "/effect give @p minecraft:fire_resistance 1000000 255 true",
    "/time set 6000",
    "/gamerule doDaylightCycle false",
]

_SKILL_DESCRIPTIONS: Dict[str, str] = {
    "wood":        "Punch/chop trees to collect logs (bare hands or any tool)",
    "woodpickaxe": "Open crafting table → craft wooden pickaxe from planks + sticks",
    "stone":       "Mine cobblestone with wooden pickaxe (given in slot 1)",
    "coal":        "Mine coal ore underground with stone pickaxe (given in slot 1)",
    "furnace":     "Open crafting table → craft furnace from 8 cobblestone",
    "stonepickaxe":"Open crafting table → craft stone pickaxe",
    "ironore":     "Mine iron ore underground with stone pickaxe (given in slot 1)",
    "iron":        "Smelt iron ore in furnace using coal as fuel",
    "ironpickaxe": "Open crafting table → craft iron pickaxe from ingots + sticks",
    "diamond":     "Mine diamond ore deep underground with iron pickaxe (given in slot 1)",
}

# ---------------------------------------------------------------------------
# Input capture (pynput)
# ---------------------------------------------------------------------------

from pynput import keyboard as _kb, mouse as _ms  # noqa: E402

_KEY_STATE: Dict[str, bool] = {
    "forward": False, "back": False, "left": False, "right": False,
    "jump": False, "sneak": False, "sprint": False,
    "attack": False, "use": False, "drop": False, "inventory": False,
}
_HOTBAR: int = 1
_mouse_dx: float = 0.0
_mouse_dy: float = 0.0
_mouse_lock = threading.Lock()
_recording: bool = False
_advance: bool = False   # F11 — save and go to next skill
_quit: bool = False
_prev_mouse: Optional[Tuple[float, float]] = None
_annotations: List[Dict] = []     # accumulated F5 bookmarks for current segment
_annot_lock = threading.Lock()
_current_frame: int = 0           # frame counter written by the main loop


def _on_key_press(key: Any) -> None:
    global _recording, _advance, _quit, _HOTBAR
    try:
        if key == _kb.Key.f5:
            if _recording:
                with _annot_lock:
                    idx = len(_annotations)
                    mark = {
                        "frame":    _current_frame,
                        "time_sec": round(time.monotonic(), 3),
                        "label":    f"mark_{idx}",
                    }
                    _annotations.append(mark)
                logger.info("[recorder] Annotation %d at frame %d", idx, _current_frame)
            else:
                logger.info("[recorder] F5 pressed but not recording — ignored")
            return
        if key == _kb.Key.f10:
            if not _recording:
                _recording = True
                logger.info("[recorder] *** RECORDING STARTED ***")
            return
        if key == _kb.Key.f11:
            _recording = False
            _advance = True
            logger.info("[recorder] *** RECORDING STOPPED — saving & advancing ***")
            return
        if key == _kb.Key.f12:
            _recording = False
            _quit = True
            logger.info("[recorder] *** QUIT ***")
            return
    except Exception:
        pass

    try:
        ch = key.char.lower() if hasattr(key, "char") and key.char else None
    except Exception:
        ch = None

    if ch == "w":   _KEY_STATE["forward"] = True
    elif ch == "s": _KEY_STATE["back"]    = True
    elif ch == "a": _KEY_STATE["left"]    = True
    elif ch == "d": _KEY_STATE["right"]   = True
    elif ch == "e": _KEY_STATE["inventory"] = True
    elif ch == "q": _KEY_STATE["drop"]    = True
    elif ch in "123456789": _HOTBAR = int(ch)

    try:
        if key == _kb.Key.space: _KEY_STATE["jump"]   = True
        elif key == _kb.Key.shift: _KEY_STATE["sneak"] = True
        elif key == _kb.Key.ctrl:  _KEY_STATE["sprint"] = True
    except Exception:
        pass


def _on_key_release(key: Any) -> None:
    try:
        ch = key.char.lower() if hasattr(key, "char") and key.char else None
    except Exception:
        ch = None

    if ch == "w":   _KEY_STATE["forward"] = False
    elif ch == "s": _KEY_STATE["back"]    = False
    elif ch == "a": _KEY_STATE["left"]    = False
    elif ch == "d": _KEY_STATE["right"]   = False
    elif ch == "e": _KEY_STATE["inventory"] = False
    elif ch == "q": _KEY_STATE["drop"]    = False

    try:
        if key == _kb.Key.space: _KEY_STATE["jump"]   = False
        elif key == _kb.Key.shift: _KEY_STATE["sneak"] = False
        elif key == _kb.Key.ctrl:  _KEY_STATE["sprint"] = False
    except Exception:
        pass


def _on_mouse_click(x: float, y: float, button: Any, pressed: bool) -> None:
    if button == _ms.Button.left:  _KEY_STATE["attack"] = pressed
    elif button == _ms.Button.right: _KEY_STATE["use"]  = pressed


def _on_mouse_move(x: float, y: float) -> None:
    global _mouse_dx, _mouse_dy, _prev_mouse
    if _prev_mouse is not None:
        dx = x - _prev_mouse[0]
        dy = y - _prev_mouse[1]
        with _mouse_lock:
            _mouse_dx += dx * 0.1
            _mouse_dy += dy * 0.1
    _prev_mouse = (x, y)


def _build_action(noop: Dict) -> Dict:
    import copy
    act = copy.deepcopy(noop)
    for k in ("forward", "back", "left", "right", "jump", "sneak",
              "sprint", "attack", "use", "drop", "inventory"):
        if k in act:
            act[k] = int(_KEY_STATE[k])

    with _mouse_lock:
        dx, dy = _mouse_dx, _mouse_dy
        globals()["_mouse_dx"] = 0.0
        globals()["_mouse_dy"] = 0.0

    dx = float(np.clip(dx, -30.0, 30.0))
    dy = float(np.clip(dy, -30.0, 30.0))
    if "camera" in act and hasattr(act["camera"], "__len__"):
        act["camera"] = np.array([dy, dx], dtype=np.float32)

    slot_key = f"hotbar.{_HOTBAR}"
    if slot_key in act:
        act[slot_key] = 1

    return act

# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

_WINDOW_KEEPER_RUNNING = False


# Center monitor geometry (DP-0: 1920x1080 at offset 1080,546 on DISPLAY=:0)
_MC_WIN_X = 1080
_MC_WIN_Y = 546
_MC_WIN_W = 1920
_MC_WIN_H = 1080


def _raise_minecraft_window_once() -> Optional[str]:
    """Find and raise every Minecraft window, positioning it on the center monitor."""
    try:
        import subprocess
        display_env = os.environ.get("DISPLAY", ":0")

        # Try wmctrl first (simpler, handles window manager decorations)
        r = subprocess.run(
            ["wmctrl", "-r", "Minecraft", "-e",
             f"0,{_MC_WIN_X},{_MC_WIN_Y},{_MC_WIN_W},{_MC_WIN_H}"],
            capture_output=True, timeout=3,
        )
        if r.returncode == 0:
            # Also raise to front
            subprocess.run(["wmctrl", "-r", "Minecraft", "-b", "add,above"],
                           capture_output=True, timeout=3)
            return "wmctrl"

        # Fallback: Xlib
        from Xlib import display as _xdisp, X
        from Xlib.protocol import event as _xevent

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
            win.configure(x=_MC_WIN_X, y=_MC_WIN_Y, width=_MC_WIN_W, height=_MC_WIN_H)
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


def _show_minecraft_window(interval: int = 15) -> None:
    """Raise the Minecraft window now and keep re-raising it every *interval* seconds."""
    global _WINDOW_KEEPER_RUNNING

    win_id = None
    for _ in range(20):          # poll up to 10 s for the window to appear
        win_id = _raise_minecraft_window_once()
        if win_id:
            break
        time.sleep(0.5)

    if win_id:
        logger.info("Minecraft window raised to 0,0 (1280x720)")
    else:
        logger.warning("Minecraft window not found — may need to view DISPLAY=%s manually",
                       os.environ.get("DISPLAY", ":1"))

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
    logger.info("Window keeper started (re-raises every %ds)", interval)


def _make_env(seed: int) -> Any:
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.entry import check_engine
    check_engine(skip_confirmation=True)
    env = MinecraftSim(
        action_type="agent",
        obs_size=(224, 224),
        preferred_spawn_biome="forest",
        seed=seed,
    )
    return env


def _exec_cmds(env: Any, obs: Dict, info: Dict, cmds: List[str]) -> Tuple[Dict, Dict]:
    """Run a list of /commands, skip silently on error."""
    _real = getattr(env, "_env", env)
    inner = getattr(_real, "env", None)
    if inner is None or not hasattr(inner, "execute_cmd"):
        logger.warning("execute_cmd unavailable — setup skipped")
        return obs, info
    for cmd in cmds:
        try:
            raw_obs, _rw, _dn, raw_info = inner.execute_cmd(cmd)
            obs, info = env._wrap_obs_info(raw_obs, raw_info)
        except Exception as exc:
            logger.debug("cmd failed (%s): %s", cmd, exc)
    try:
        noop = env.noop_action()
        obs, _, _t, _tr, info = env.step(noop)
    except Exception:
        pass
    return obs, info


def _setup_skill(env: Any, obs: Dict, info: Dict, skill: str) -> Tuple[Dict, Dict]:
    """Apply common + skill-specific setup commands."""
    cmds = list(_COMMON_SETUP) + _SKILL_SETUP.get(skill, [])
    return _exec_cmds(env, obs, info, cmds)

# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_segment(
    skill: str,
    seg_idx: int,
    frames: List[np.ndarray],
    actions: List[Dict],
    annotations: List[Dict],
    out_dir: str,
    fps: int = 20,
) -> str:
    skill_dir = os.path.join(out_dir, skill)
    os.makedirs(skill_dir, exist_ok=True)
    tag = f"demo_{seg_idx:03d}"
    mp4 = os.path.join(skill_dir, f"{tag}.mp4")
    npz = os.path.join(skill_dir, f"{tag}_bc.npz")

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()

    BOOL_KEYS = ["forward", "back", "left", "right", "jump", "sneak",
                 "sprint", "attack", "use", "drop", "inventory"]
    n = len(actions)
    bool_arr   = np.zeros((n, len(BOOL_KEYS)), dtype=np.float32)
    camera_arr = np.zeros((n, 2), dtype=np.float32)
    for t, act in enumerate(actions):
        for ki, k in enumerate(BOOL_KEYS):
            bool_arr[t, ki] = float(act.get(k, 0))
        cam = act.get("camera", [0.0, 0.0])
        camera_arr[t] = [float(cam[0]), float(cam[1])] if hasattr(cam, "__len__") else [0.0, 0.0]

    np.savez_compressed(npz, obs=np.stack(frames), bool_actions=bool_arr,
                        bool_keys=np.array(BOOL_KEYS), camera=camera_arr)

    if annotations:
        annot_path = os.path.join(skill_dir, f"{tag}_annotations.json")
        with open(annot_path, "w") as fh:
            json.dump({"skill": skill, "segment": seg_idx,
                       "total_frames": len(frames), "fps": fps,
                       "annotations": annotations}, fh, indent=2)
        logger.info("Saved: %s  (%d annotations)", annot_path, len(annotations))

    logger.info("Saved: %s  (%d frames)", mp4, len(frames))
    return mp4

# ---------------------------------------------------------------------------
# Main session loop
# ---------------------------------------------------------------------------

def _next_seg_idx(out_dir: str, skill: str) -> int:
    skill_dir = os.path.join(out_dir, skill)
    if not os.path.isdir(skill_dir):
        return 0
    existing = [f for f in os.listdir(skill_dir) if f.endswith(".mp4")]
    return len(existing)


def _load_manifest(out_dir: str) -> List[Dict]:
    path = os.path.join(out_dir, "manifest.json")
    if os.path.isfile(path):
        with open(path) as fh:
            return json.load(fh)
    return []


def _save_manifest(out_dir: str, entries: List[Dict]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w") as fh:
        json.dump(entries, fh, indent=2)


def _record_seed(
    seed: int,
    seed_label: str,
    start_skill_idx: int,
    args: argparse.Namespace,
    manifest: List[Dict],
) -> bool:
    """Record all skills for one seed in a single continuous world session.

    The world is loaded ONCE per seed.  Between skills only /give commands
    run — no world reset — so the player stays in the same terrain throughout
    the full wood→diamond playthrough.  The world only reloads when moving to
    the next seed.

    Returns False if the user pressed F12 (quit).
    """
    global _recording, _advance, _quit

    logger.info("\n%s\n%s  seed=%d\n%s", "="*70, seed_label, seed, "="*70)
    env = _make_env(seed)
    obs, info = env.reset()

    # Apply common setup once for this world (hunger off, time lock, effects)
    obs, info = _exec_cmds(env, obs, info, list(_COMMON_SETUP))
    _show_minecraft_window()

    try:
        noop_action = env.noop_action()
    except AttributeError:
        noop_action = {}

    tick = 1.0 / args.fps

    for skill_idx in range(start_skill_idx, len(SKILL_SEQUENCE)):
        if _quit:
            break

        skill = SKILL_SEQUENCE[skill_idx]
        seg_idx = _next_seg_idx(args.out_dir, skill)

        # Apply only skill-specific /give commands — NO env.reset() between skills
        skill_cmds = _SKILL_SETUP.get(skill, [])
        if skill_cmds:
            obs, info = _exec_cmds(env, obs, info, skill_cmds)

        # Reset per-skill recording state
        _recording = False
        _advance = False
        with _annot_lock:
            _annotations.clear()
        globals()["_current_frame"] = 0

        logger.info(
            "\n%s\n%s  Skill %d/%d: %s\n  %s\n%s",
            "="*70, seed_label,
            skill_idx + 1, len(SKILL_SEQUENCE), skill.upper(),
            _SKILL_DESCRIPTIONS.get(skill, ""),
            "="*70,
        )
        logger.info(
            "\n"
            "  Will save → %s/%s/demo_%03d.mp4\n"
            "  F5  = bookmark / annotate current frame\n"
            "  F10 = START recording\n"
            "  F11 = STOP + SAVE → next skill\n"
            "  F12 = QUIT session\n",
            args.out_dir, skill, seg_idx,
        )

        frames: List[np.ndarray] = []
        actions: List[Dict] = []

        while not _quit and not _advance:
            t0 = time.perf_counter()

            act = _build_action(noop_action)
            try:
                obs, _r, terminated, truncated, info = env.step(act)
            except Exception as exc:
                logger.warning("env.step error: %s", exc)
                terminated = True

            if _recording and "image" in obs:
                frames.append(obs["image"].copy())
                actions.append(act)
                globals()["_current_frame"] = len(frames) - 1

            # If the episode ends unexpectedly (death etc.), reset and re-apply setup
            if terminated or truncated:
                logger.warning("[%s] Unexpected episode end — respawning in same world.", skill)
                obs, info = env.reset()
                obs, info = _exec_cmds(env, obs, info, list(_COMMON_SETUP))
                if skill_cmds:
                    obs, info = _exec_cmds(env, obs, info, skill_cmds)

            elapsed = time.perf_counter() - t0
            wait = tick - elapsed
            if wait > 0:
                time.sleep(wait)

        if frames:
            with _annot_lock:
                annots = list(_annotations)
            mp4 = _save_segment(skill, seg_idx, frames, actions, annots, args.out_dir, args.fps)
            entry = {
                "seed": seed, "skill": skill, "segment": seg_idx,
                "frames": len(frames), "annotations": len(annots),
                "mp4": mp4,
            }
            manifest.append(entry)
            _save_manifest(args.out_dir, manifest)
            logger.info("[%s] seed=%d demo_%03d saved (%d frames, %d annotations).",
                        skill, seed, seg_idx, len(frames), len(annots))
        else:
            logger.info("[%s] No frames recorded — skipping save.", skill)

    try:
        env.close()
    except Exception:
        pass

    return not _quit


def record(args: argparse.Namespace) -> None:
    global _quit

    seeds: List[int] = args.seeds
    start_seed_val = args.start_seed if args.start_seed is not None else seeds[0]
    start_skill_idx = SKILL_SEQUENCE.index(args.start_skill) if args.start_skill in SKILL_SEQUENCE else 0

    # Find where to resume in the seed list
    if start_seed_val in seeds:
        seed_start = seeds.index(start_seed_val)
    else:
        seed_start = 0

    manifest = _load_manifest(args.out_dir)

    logger.info(
        "Recording %d playthroughs: seeds %s",
        len(seeds), seeds,
    )
    logger.info("Resuming from seed=%d, skill=%s", start_seed_val, args.start_skill)

    kb = _kb.Listener(on_press=_on_key_press, on_release=_on_key_release)
    ms = _ms.Listener(on_move=_on_mouse_move, on_click=_on_mouse_click)
    kb.start(); ms.start()

    for s_idx in range(seed_start, len(seeds)):
        if _quit:
            break

        seed = seeds[s_idx]
        seed_label = f"[Playthrough {s_idx + 1}/{len(seeds)}]"
        # start_skill only applies to the first (possibly resumed) seed
        skill_start = start_skill_idx if s_idx == seed_start else 0

        ok = _record_seed(seed, seed_label, skill_start, args, manifest)
        if not ok:
            break

        logger.info("\n%s\nPlaythrough %d/%d complete (seed=%d).\n%s",
                    "="*70, s_idx + 1, len(seeds), seed, "="*70)

    kb.stop(); ms.stop()

    demos_per_skill = {
        skill: len([e for e in manifest if e["skill"] == skill])
        for skill in SKILL_SEQUENCE
    }
    logger.info("\nSession complete.")
    logger.info("Demos collected per skill: %s", demos_per_skill)
    logger.info("Manifest: %s/manifest.json (%d entries)", args.out_dir, len(manifest))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Record full tech-tree playthrough demos for GROOT fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--out_dir",     default="data/playthrough_demos")
    ap.add_argument("--seeds",       type=int, nargs="+", default=list(range(20)),
                    help="Seed values to record, in order (default: 0–19)")
    ap.add_argument("--fps",         type=int, default=20)
    ap.add_argument("--start_seed",  type=int, default=None,
                    help="Resume from this seed (must be in --seeds list)")
    ap.add_argument("--start_skill", default="wood", choices=SKILL_SEQUENCE,
                    help="Resume from this skill within --start_seed's playthrough")
    args = ap.parse_args()

    print("record_playthrough_demos.py")
    print(f"  {'out_dir':12s}: {args.out_dir}")
    print(f"  {'seeds':12s}: {args.seeds}")
    print(f"  {'fps':12s}: {args.fps}")
    print(f"  {'start_seed':12s}: {args.start_seed}")
    print(f"  {'start_skill':12s}: {args.start_skill}")
    print()
    print("Skill sequence:", " → ".join(SKILL_SEQUENCE))
    print()
    print("Controls:")
    print("  F5  = annotate / bookmark current frame")
    print("  F10 = start recording")
    print("  F11 = stop + save → next skill")
    print("  F12 = quit session")
    print()

    record(args)


if __name__ == "__main__":
    main()
