#!/usr/bin/env python3
"""
Human crafting demonstration recorder for GROOT BC fine-tuning.

Launches a MineStudio env with the required ingredients pre-given via /give,
then records keyboard+mouse input at 20 Hz while the human crafts the target
item.  Saves:

  <out_dir>/<skill>/<tag>.mp4          — reference video clip (for GROOT conditioning)
  <out_dir>/<skill>/<tag>_bc.npz       — obs + named-action-dict arrays (BC training)

Controls (game window must be focused):
  W / S / A / D  — forward / back / left / right
  Space          — jump
  Left Shift     — sneak
  Left Ctrl      — sprint
  Left mouse     — attack / break block
  Right mouse    — use / interact / place block
  E              — toggle inventory
  1-9            — hotbar slot
  Q key          — drop item
  ---
  F10            — start recording a segment
  F11            — stop and save current segment
  F12            — quit recorder

Usage:
  conda run -n dia-minecraft python scripts/record_crafting_demos.py \\
      --skill woodpickaxe --seed 0 \\
      --out_dir data/crafting_demos

  conda run -n dia-minecraft python scripts/record_crafting_demos.py \\
      --skill stonepickaxe --seed 0 \\
      --out_dir data/crafting_demos

Supported skills: woodpickaxe, stonepickaxe, ironpickaxe, furnace, iron
"""
from __future__ import annotations

import argparse
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
logger = logging.getLogger("record_crafting")

# ---------------------------------------------------------------------------
# Ingredients to /give per skill
# ---------------------------------------------------------------------------

_GIVE_CMDS: Dict[str, List[str]] = {
    "woodpickaxe": [
        "/give @p minecraft:oak_log 6",        # 6 logs → plenty of planks+sticks
    ],
    "stonepickaxe": [
        "/give @p minecraft:cobblestone 16",
        "/give @p minecraft:oak_log 4",
    ],
    "ironpickaxe": [
        "/give @p minecraft:iron_ingot 6",
        "/give @p minecraft:oak_log 4",
    ],
    "furnace": [
        "/give @p minecraft:cobblestone 16",
    ],
    "iron": [
        "/give @p minecraft:iron_ore 4",
        "/give @p minecraft:coal 4",
        "/give @p minecraft:furnace 1",
    ],
}

# Also place a crafting table and furnace nearby so the player can craft
_SETUP_CMDS: List[str] = [
    "/setblock ~ ~ ~3 minecraft:crafting_table",  # 3 blocks ahead
    "/effect give @p minecraft:saturation 1000000 255 true",
    "/time set 6000",
    "/gamerule doDaylightCycle false",
]

# ---------------------------------------------------------------------------
# Keyboard state tracker (pynput global listener)
# ---------------------------------------------------------------------------

from pynput import keyboard as _kb, mouse as _ms  # noqa: E402

_KEY_STATE: Dict[str, bool] = {
    "forward": False, "back": False, "left": False, "right": False,
    "jump": False, "sneak": False, "sprint": False,
    "attack": False, "use": False, "drop": False, "inventory": False,
}
_HOTBAR: int = 1  # current slot 1-9

# Mouse delta accumulator — read and reset at each action tick
_mouse_dx: float = 0.0
_mouse_dy: float = 0.0
_mouse_lock = threading.Lock()

# Control flags
_recording: bool = False
_quit: bool = False

_HOTBAR_KEYS: Dict[_kb.Key, int] = {}


def _on_key_press(key: Any) -> None:
    global _recording, _quit, _HOTBAR

    # Special keys
    try:
        if key == _kb.Key.f10:
            _recording = True
            logger.info("[recorder] recording STARTED")
            return
        if key == _kb.Key.f11:
            _recording = False
            logger.info("[recorder] recording STOPPED — saving...")
            return
        if key == _kb.Key.f12:
            _quit = True
            logger.info("[recorder] quit requested")
            return
    except Exception:
        pass

    # Movement / action keys
    try:
        ch = key.char.lower() if hasattr(key, "char") and key.char else None
    except Exception:
        ch = None

    if ch == "w":
        _KEY_STATE["forward"] = True
    elif ch == "s":
        _KEY_STATE["back"] = True
    elif ch == "a":
        _KEY_STATE["left"] = True
    elif ch == "d":
        _KEY_STATE["right"] = True
    elif ch == "e":
        _KEY_STATE["inventory"] = True
    elif ch == "q":
        _KEY_STATE["drop"] = True
    elif ch in "123456789":
        _HOTBAR = int(ch)

    try:
        if key == _kb.Key.space:
            _KEY_STATE["jump"] = True
        elif key == _kb.Key.shift:
            _KEY_STATE["sneak"] = True
        elif key == _kb.Key.ctrl:
            _KEY_STATE["sprint"] = True
    except Exception:
        pass


def _on_key_release(key: Any) -> None:
    try:
        ch = key.char.lower() if hasattr(key, "char") and key.char else None
    except Exception:
        ch = None

    if ch == "w":
        _KEY_STATE["forward"] = False
    elif ch == "s":
        _KEY_STATE["back"] = False
    elif ch == "a":
        _KEY_STATE["left"] = False
    elif ch == "d":
        _KEY_STATE["right"] = False
    elif ch == "e":
        _KEY_STATE["inventory"] = False
    elif ch == "q":
        _KEY_STATE["drop"] = False

    try:
        if key == _kb.Key.space:
            _KEY_STATE["jump"] = False
        elif key == _kb.Key.shift:
            _KEY_STATE["sneak"] = False
        elif key == _kb.Key.ctrl:
            _KEY_STATE["sprint"] = False
    except Exception:
        pass


def _on_mouse_click(x: float, y: float, button: Any, pressed: bool) -> None:
    if button == _ms.Button.left:
        _KEY_STATE["attack"] = pressed
    elif button == _ms.Button.right:
        _KEY_STATE["use"] = pressed


def _on_mouse_move(x: float, y: float) -> None:
    # We get absolute positions; compute delta vs screen centre
    # Minecraft locks cursor to centre, so raw absolute coords won't work.
    # Instead we rely on _on_mouse_scroll / _on_mouse_move giving us deltas
    # via the accumulated difference from the previous call.
    pass  # handled by _on_mouse_delta


_prev_mouse: Optional[Tuple[float, float]] = None


def _on_mouse_move_delta(x: float, y: float) -> None:
    global _mouse_dx, _mouse_dy, _prev_mouse
    if _prev_mouse is not None:
        dx = x - _prev_mouse[0]
        dy = y - _prev_mouse[1]
        with _mouse_lock:
            _mouse_dx += dx * 0.1   # scale: raw pixels → degrees-ish
            _mouse_dy += dy * 0.1
    _prev_mouse = (x, y)


def _build_action_dict(env_noop: Dict) -> Dict:
    """Construct a MineStudio agent action dict from current input state."""
    import copy
    act = copy.deepcopy(env_noop)

    act["forward"] = int(_KEY_STATE["forward"])
    act["back"]    = int(_KEY_STATE["back"])
    act["left"]    = int(_KEY_STATE["left"])
    act["right"]   = int(_KEY_STATE["right"])
    act["jump"]    = int(_KEY_STATE["jump"])
    act["sneak"]   = int(_KEY_STATE["sneak"])
    act["sprint"]  = int(_KEY_STATE["sprint"])
    act["attack"]  = int(_KEY_STATE["attack"])
    act["use"]     = int(_KEY_STATE["use"])
    act["drop"]    = int(_KEY_STATE["drop"])
    act["inventory"] = int(_KEY_STATE["inventory"])

    # Mouse look
    with _mouse_lock:
        dx = _mouse_dx
        dy = _mouse_dy
        # reset after read
        globals()["_mouse_dx"] = 0.0
        globals()["_mouse_dy"] = 0.0

    # MineStudio camera: [pitch_delta, yaw_delta] in degrees
    # Clamp to ±30° per step (20Hz — smooth movement)
    dx = float(np.clip(dx, -30.0, 30.0))
    dy = float(np.clip(dy, -30.0, 30.0))
    if "camera" in act and hasattr(act["camera"], "__len__"):
        act["camera"] = np.array([dy, dx], dtype=np.float32)  # pitch=dy, yaw=dx

    # Hotbar
    slot_key = f"hotbar.{_HOTBAR}"
    if slot_key in act:
        act[slot_key] = 1

    return act


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

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


def _apply_setup(env: Any, obs: Dict, info: Dict, skill: str) -> Tuple[Dict, Dict]:
    inner = getattr(env, "env", None)
    if inner is None or not hasattr(inner, "execute_cmd"):
        logger.warning("execute_cmd unavailable — setup commands skipped")
        return obs, info

    cmds = list(_SETUP_CMDS)
    cmds.extend(_GIVE_CMDS.get(skill, []))

    for cmd in cmds:
        try:
            raw_obs, _rw, _dn, raw_info = inner.execute_cmd(cmd)
            obs, info = env._wrap_obs_info(raw_obs, raw_info)
            logger.info("setup OK: %s", cmd)
        except Exception as exc:
            logger.warning("setup failed (%s): %s", cmd, exc)

    # One noop step to let effects register
    try:
        noop = env.noop_action()
        obs, _, _t, _tr, info = env.step(noop)
    except Exception as exc:
        logger.warning("noop step failed: %s", exc)

    return obs, info


def _save_segment(
    skill: str,
    seg_idx: int,
    frames: List[np.ndarray],
    actions: List[Dict],
    out_dir: str,
    fps: int = 20,
) -> str:
    """Save frames as MP4 and actions as .npz.  Returns the MP4 path."""
    skill_dir = os.path.join(out_dir, skill)
    os.makedirs(skill_dir, exist_ok=True)

    tag = f"demo_{seg_idx:03d}"
    mp4_path = os.path.join(skill_dir, f"{tag}.mp4")
    npz_path = os.path.join(skill_dir, f"{tag}_bc.npz")

    # Write MP4
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    logger.info("Saved clip: %s  (%d frames)", mp4_path, len(frames))

    # Write BC arrays — save obs and a compressed action representation
    obs_arr = np.stack(frames, axis=0)  # (N, H, W, 3) uint8

    # Serialize action dicts as structured arrays (forward, back, ..., camera_pitch, camera_yaw)
    BOOL_KEYS = ["forward", "back", "left", "right", "jump", "sneak",
                 "sprint", "attack", "use", "drop", "inventory"]
    n = len(actions)
    bool_arr = np.zeros((n, len(BOOL_KEYS)), dtype=np.float32)
    camera_arr = np.zeros((n, 2), dtype=np.float32)
    for t, act in enumerate(actions):
        for k_i, k in enumerate(BOOL_KEYS):
            bool_arr[t, k_i] = float(act.get(k, 0))
        cam = act.get("camera", [0.0, 0.0])
        camera_arr[t] = [float(cam[0]), float(cam[1])] if hasattr(cam, "__len__") else [0.0, 0.0]

    np.savez_compressed(
        npz_path,
        obs=obs_arr,
        bool_actions=bool_arr,
        bool_keys=np.array(BOOL_KEYS),
        camera=camera_arr,
    )
    logger.info("Saved BC data: %s", npz_path)
    return mp4_path


# ---------------------------------------------------------------------------
# Main recording loop
# ---------------------------------------------------------------------------

def record(args: argparse.Namespace) -> None:
    global _recording, _quit

    skill = args.skill
    if skill not in _GIVE_CMDS:
        logger.error("Unknown skill '%s'. Supported: %s", skill, list(_GIVE_CMDS))
        sys.exit(1)

    logger.info("Setting up env (seed=%d, skill=%s)...", args.seed, skill)
    env = _make_env(args.seed)
    obs, info = env.reset()
    obs, info = _apply_setup(env, obs, info, skill)

    # Get a baseline noop action for constructing action dicts
    try:
        noop_action = env.noop_action()
    except AttributeError:
        noop_action = {}

    # Start input listeners
    kb_listener = _kb.Listener(on_press=_on_key_press, on_release=_on_key_release)
    ms_listener = _ms.Listener(
        on_move=_on_mouse_move_delta,
        on_click=_on_mouse_click,
    )
    kb_listener.start()
    ms_listener.start()

    logger.info(
        "\n"
        "=== CRAFTING DEMO RECORDER ===\n"
        "  Skill: %s\n"
        "  Items given — craft the item in the crafting table 3 blocks ahead\n"
        "  F10  = start recording\n"
        "  F11  = stop + save segment\n"
        "  F12  = quit\n"
        "==============================\n",
        skill,
    )

    seg_frames: List[np.ndarray] = []
    seg_actions: List[Dict] = []
    seg_idx = args.start_idx
    tick = 1.0 / args.fps

    try:
        while not _quit:
            t0 = time.perf_counter()

            # Build action from current key state
            action_dict = _build_action_dict(noop_action)

            # Step env
            try:
                obs, _r, terminated, truncated, info = env.step(action_dict)
            except Exception as exc:
                logger.warning("env.step error: %s", exc)
                terminated = True

            # Record frame if active
            if _recording and "image" in obs:
                seg_frames.append(obs["image"].copy())
                seg_actions.append(action_dict)

            # Save when recording just stopped (flag just turned False)
            if not _recording and seg_frames:
                mp4 = _save_segment(skill, seg_idx, seg_frames, seg_actions, args.out_dir, args.fps)
                seg_frames = []
                seg_actions = []
                seg_idx += 1
                logger.info("Segment %d saved → %s", seg_idx - 1, mp4)
                logger.info("Press F10 to record another segment, F12 to quit.")

            if terminated or truncated:
                logger.info("Episode ended — resetting env...")
                obs, info = env.reset()
                obs, info = _apply_setup(env, obs, info, skill)

            # Rate-limit to target FPS
            elapsed = time.perf_counter() - t0
            wait = tick - elapsed
            if wait > 0:
                time.sleep(wait)

    finally:
        if _recording and seg_frames:
            logger.info("Auto-saving in-progress segment (%d frames)...", len(seg_frames))
            _save_segment(skill, seg_idx, seg_frames, seg_actions, args.out_dir, args.fps)

        kb_listener.stop()
        ms_listener.stop()
        try:
            env.close()
        except Exception:
            pass

    logger.info("Recording session complete.  %d segments saved to %s/%s/",
                seg_idx - args.start_idx, args.out_dir, skill)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Record human crafting demonstrations for GROOT BC fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--skill",     required=True,
                    choices=list(_GIVE_CMDS), help="Crafting skill to demonstrate")
    ap.add_argument("--seed",      type=int, default=0, help="Env seed")
    ap.add_argument("--out_dir",   default="data/crafting_demos",
                    help="Directory to save clips and BC data")
    ap.add_argument("--fps",       type=int, default=20, help="Recording frame rate")
    ap.add_argument("--start_idx", type=int, default=0,
                    help="Starting segment index (to avoid overwriting existing demos)")
    args = ap.parse_args()

    print("record_crafting_demos.py")
    for k in ("skill", "seed", "out_dir", "fps", "start_idx"):
        print(f"  {k:12s}: {getattr(args, k)}")
    print()

    record(args)


if __name__ == "__main__":
    main()
