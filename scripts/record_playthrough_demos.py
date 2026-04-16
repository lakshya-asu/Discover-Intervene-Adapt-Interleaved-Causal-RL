#!/usr/bin/env python3
"""
Full tech-tree playthrough demo recorder.

Records one continuous human demonstration per seed.  Play the full tech tree
from bare hands to diamond — the world advances to the next seed automatically
when a diamond appears in your inventory.  Frames, actions, and language
annotations are saved as one bundle per seed.

Output per seed (in --out_dir/seed_<N>/):
  playthrough.mp4               — full recording
  playthrough_bc.npz            — obs + action arrays (BC fine-tuning)
  playthrough_annotations.json  — labeled segments (F5 annotations)

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
  F5             — annotate: describe what you're about to do
  F12            — quit

Annotation segments (F5):
  Press F5 to END the current labeled segment and START a new one.
  The game pauses, the terminal prompts you to type a description of what you're
  about to do (e.g. "walking toward coal vein", "mining coal with pickaxe").
  Press Enter — game resumes and all frames until the next F5 are tagged with that label.
  Segments are saved as {"start_frame": N, "end_frame": M, "label": "..."} in a JSON
  sidecar alongside the MP4 and BC npz.  Use them to slice targeted fine-tuning clips.

Seed advancement:
  The seed advances automatically when a diamond appears in your inventory.
  Press F12 to quit at any time.

Usage:
  conda run -n dia-minecraft python scripts/record_playthrough_demos.py \\
      --out_dir data/playthrough_demos

  # Resume from a specific seed:
  conda run -n dia-minecraft python scripts/record_playthrough_demos.py \\
      --out_dir data/playthrough_demos \\
      --start_seed 3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
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
# Input capture — MineStudio PlayCallback (exclusive mouse, proper key events)
# ---------------------------------------------------------------------------
# PlayCallback creates a separate pyglet window that:
#   - Captures the mouse exclusively (cursor locked to window)
#   - Reads WASD / Space / Shift / Ctrl / mouse buttons / mouse delta
#   - Returns actions in env format (individual key dict)
# We subclass it to add F5 (annotate) and F12 (quit) detection.
#
# Controls printed on startup:
#   C            — capture / release mouse in the game window
#   Left Ctrl+C  — close window
#   Esc          — enter chat/command mode
#   F5           — annotate: describe what you're doing
#   F12          — quit recording session

class _RecordingPlayCallback:
    """Wraps PlayCallback's GUI for direct use from the game loop.

    NOT registered with MinecraftSim as a callback — env.reset() and
    env.step() run their own internal callback chains unmodified.
    We drive the GUI manually each tick: dispatch events, read human
    actions, push rendered frames.

    Controls printed by PlayCallback on startup:
      C            — capture / release mouse
      Left Ctrl+C  — close window / quit
      F5           — annotate current segment
      F12          — quit recording session
    """

    def __init__(self) -> None:
        from minestudio.simulator.callbacks.play import PlayCallback
        self._cb = PlayCallback(agent_generator=None)
        self.f5_pressed: bool = False
        self.quit_pressed: bool = False

    # ------------------------------------------------------------------
    # Per-tick interface (called from game loop)
    # ------------------------------------------------------------------

    def get_action(self) -> Dict:
        """Dispatch GUI events and return the current human action dict."""
        gui = self._cb.gui
        gui.window.dispatch_events()

        released = gui._capture_all_keys()
        if "F5" in released:
            self.f5_pressed = True
        if "F12" in released:
            self.quit_pressed = True
            logger.info("[recorder] *** QUIT (F12) ***")
        # 'C' toggles exclusive mouse; Ctrl+C quits
        if "C" in released:
            if not (gui.modifiers & gui.key.MOD_CTRL):
                gui.capture_mouse = not gui.capture_mouse
                gui.window.set_mouse_visible(not gui.capture_mouse)
                gui.window.set_exclusive_mouse(gui.capture_mouse)
            else:
                self.quit_pressed = True
                logger.info("[recorder] *** QUIT (Ctrl-C) ***")

        return gui._get_human_action()

    def update_display(self, info: Dict) -> None:
        """Push the latest observation frame to the GUI."""
        try:
            self._cb.gui._update_image(info)
        except Exception:
            pass

    def show_resetting(self) -> None:
        """Show 'Resetting environment...' message (call before env.reset())."""
        try:
            self._cb.gui.reset_gui()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Window management
    # ------------------------------------------------------------------

    def position_window(self, x: int = 1080, y: int = 546,
                        w: int = 1280, h: int = 720) -> None:
        """Move and resize the pyglet game window onto the center monitor."""
        try:
            win = self._cb.gui.window
            win.set_size(w, h)
            win.set_location(x, y)
            logger.info("PlayCallback window at (%d,%d) %dx%d", x, y, w, h)
        except Exception as exc:
            logger.warning("Could not position game window: %s", exc)

    def release_mouse(self) -> None:
        """Release exclusive mouse capture (call before terminal input())."""
        try:
            self._cb.gui.window.set_exclusive_mouse(False)
            self._cb.gui.window.set_mouse_visible(True)
        except Exception:
            pass

    def capture_mouse(self) -> None:
        """Re-acquire exclusive mouse capture (call after terminal input())."""
        try:
            self._cb.gui.window.set_exclusive_mouse(True)
            self._cb.gui.window.set_mouse_visible(False)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> Any:
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.entry import check_engine
    check_engine(skip_confirmation=True)
    env = MinecraftSim(
        action_type="env",
        obs_size=(224, 224),
        preferred_spawn_biome="forest",
        seed=seed,
        # No callbacks — PlayCallback GUI is driven manually in the game loop
        # so it cannot interfere with env.reset() internals.
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


def _inv_has_diamond(info: Dict) -> bool:
    """Return True if a diamond appears in the agent's inventory."""
    diamond_names = {"diamond", "minecraft:diamond"}
    inv = info.get("inventory", {})
    if isinstance(inv, dict):
        names = inv.get("name", [])
        qtys  = inv.get("quantity", [])
        if hasattr(names, "__len__") and len(names) > 0:
            for n, q in zip(names, qtys):
                if str(n).strip().lower().rstrip("\x00") in diamond_names and int(q) > 0:
                    return True
        for v in inv.values():
            if isinstance(v, dict):
                if str(v.get("type", "")).lower() in diamond_names and int(v.get("quantity", 0)) > 0:
                    return True
    pickup = info.get("pickup", {})
    return any(pickup.get(name, 0) > 0 for name in diamond_names)

# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_segment(
    seed: int,
    frames: List[np.ndarray],
    actions: List[Dict],
    annotations: List[Dict],
    out_dir: str,
    fps: int = 20,
) -> str:
    seed_dir = os.path.join(out_dir, f"seed_{seed:03d}")
    os.makedirs(seed_dir, exist_ok=True)
    mp4 = os.path.join(seed_dir, "playthrough.mp4")
    npz = os.path.join(seed_dir, "playthrough_bc.npz")

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
        annot_path = os.path.join(seed_dir, "playthrough_annotations.json")
        with open(annot_path, "w") as fh:
            json.dump({
                "seed": seed,
                "total_frames": len(frames),
                "fps": fps,
                "labeled_segments": annotations,
                "note": (
                    "Each entry has start_frame, end_frame (inclusive), and label (str). "
                    "Use these to slice obs/bc arrays for targeted fine-tuning."
                ),
            }, fh, indent=2)
        logger.info("Saved: %s  (%d labeled segments)", annot_path, len(annotations))

    logger.info("Saved: %s  (%d frames)", mp4, len(frames))
    return mp4

# ---------------------------------------------------------------------------
# Main session loop
# ---------------------------------------------------------------------------

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
    args: argparse.Namespace,
    manifest: List[Dict],
    play_cb: _RecordingPlayCallback,
) -> bool:
    """Record one full free-play session for a seed.

    The world loads once. Recording starts immediately. The session ends when:
      - A diamond appears in the inventory  → auto-advance to next seed
      - The user presses F12               → quit

    Returns False if the user pressed F12 (quit).
    """
    logger.info("\n%s\n%s  seed=%d\n%s", "="*70, seed_label, seed, "="*70)
    logger.info(
        "\n"
        "  Recording to → %s/seed_%03d/\n"
        "  F5  = annotate (describe what you're about to do)\n"
        "  F12 = QUIT session\n"
        "  Recording will stop automatically when you collect a diamond.\n",
        args.out_dir, seed,
    )

    env = _make_env(seed)

    # Show "Resetting..." in the PlayCallback window while waiting for Malmo.
    play_cb.show_resetting()

    # Retry env.reset() with a 25-second gap between attempts.
    # Why 25s: after Malmo restarts internally, the JVM says "process ready"
    # after ~15s but the Malmo socket needs ~10s more to start accepting
    # connections.  A 5s delay (previous value) was consistently too short;
    # 25s gives the full ~25s window after "process ready".
    for _attempt in range(5):
        try:
            obs, info = env.reset()
            break
        except (ConnectionRefusedError, ConnectionResetError, OSError) as _exc:
            if _attempt == 4:
                raise
            logger.warning("env.reset() failed (%s), retry %d/5 in 25 s…", _exc, _attempt + 1)
            play_cb.show_resetting()
            time.sleep(25)

    # Apply common setup: hunger off, time lock to day, survival effects, torches
    obs, info = _exec_cmds(env, obs, info, list(_COMMON_SETUP) + [
        "/give @p minecraft:torch 64",
    ])

    # Position window and push the first frame to the GUI
    play_cb.position_window(x=1080, y=546, w=1280, h=720)
    play_cb.update_display(info)

    # Reset per-seed flags
    play_cb.f5_pressed = False
    play_cb.quit_pressed = False

    # Annotation state (local to this seed)
    annotation_segments: List[Dict] = []
    current_label: Optional[str] = None
    label_start_frame: int = 0
    current_frame: int = 0

    tick = 1.0 / args.fps
    frames: List[np.ndarray] = []
    actions: List[Dict] = []
    diamond_found = False

    try:
        while not play_cb.quit_pressed and not diamond_found:
            # --- Handle F5 annotation prompt ---
            if play_cb.f5_pressed:
                play_cb.f5_pressed = False
                play_cb.release_mouse()

                # Close the previous labeled segment
                if current_label is not None:
                    annotation_segments.append({
                        "start_frame": label_start_frame,
                        "end_frame":   current_frame,
                        "label":       current_label,
                    })
                    logger.info("[annot] Closed: '%s' frames %d–%d",
                                current_label, label_start_frame, current_frame)
                    current_label = None

                print("\n" + "─"*60)
                print(f"  Annotate — seed {seed}")
                print("  Describe what you are about to do (or press Enter to skip):")
                print("  Examples: 'chopping oak tree', 'crafting wooden pickaxe',")
                print("            'mining coal', 'digging to diamond level', 'placing torch'")
                label = input("  >>> ").strip()

                if label:
                    current_label = label
                    label_start_frame = current_frame
                    logger.info("[annot] Started: '%s' at frame %d", label, current_frame)
                else:
                    logger.info("[annot] No label — continuing unlabeled")

                play_cb.capture_mouse()
                print("─"*60 + "\n")

            t0 = time.perf_counter()

            # Read human input from PlayCallback GUI (dispatches events too)
            action = play_cb.get_action()

            try:
                obs, _r, terminated, truncated, info = env.step(action)
            except Exception as exc:
                logger.warning("env.step error: %s — skipping frame", exc)
                elapsed = time.perf_counter() - t0
                wait = tick - elapsed
                if wait > 0:
                    time.sleep(wait)
                continue

            if "image" in obs:
                frames.append(obs["image"].copy())
                actions.append(action)
                play_cb.update_display(info)
                current_frame = len(frames) - 1

            # Check for diamond — advance seed automatically
            if _inv_has_diamond(info):
                diamond_found = True
                logger.info("[seed=%d] Diamond found at frame %d! Advancing to next seed.",
                            seed, current_frame)
                break

            # Respawn on death without resetting the world
            if terminated or truncated:
                logger.warning("[seed=%d] Episode ended — respawning.", seed)
                play_cb.show_resetting()
                for _ra in range(3):
                    try:
                        obs, info = env.reset()
                        break
                    except (ConnectionRefusedError, ConnectionResetError, OSError) as _exc:
                        if _ra == 2:
                            break
                        logger.warning("respawn reset failed (%s), retry %d/3…", _exc, _ra + 1)
                        play_cb.show_resetting()
                        time.sleep(25)
                obs, info = _exec_cmds(env, obs, info, list(_COMMON_SETUP) + [
                    "/give @p minecraft:torch 64",
                ])
                play_cb.update_display(info)

            elapsed = time.perf_counter() - t0
            wait = tick - elapsed
            if wait > 0:
                time.sleep(wait)

    finally:
        # Close the last open annotation segment
        if current_label is not None:
            annotation_segments.append({
                "start_frame": label_start_frame,
                "end_frame":   current_frame,
                "label":       current_label,
            })

        if frames:
            mp4 = _save_segment(seed, frames, actions, annotation_segments, args.out_dir, args.fps)
            entry = {
                "seed": seed,
                "frames": len(frames),
                "diamond_found": diamond_found,
                "annotations": len(annotation_segments),
                "mp4": mp4,
            }
            manifest.append(entry)
            _save_manifest(args.out_dir, manifest)
            logger.info("seed=%d saved: %d frames, diamond=%s, %d annotations.",
                        seed, len(frames), diamond_found, len(annotation_segments))
        else:
            logger.info("seed=%d: no frames recorded.", seed)

        try:
            env.close()
        except Exception:
            pass

    return not play_cb.quit_pressed


def record(args: argparse.Namespace) -> None:
    seeds: List[int] = args.seeds
    start_seed_val = args.start_seed if args.start_seed is not None else seeds[0]

    # Find where to resume in the seed list
    if start_seed_val in seeds:
        seed_start = seeds.index(start_seed_val)
    else:
        seed_start = 0

    manifest = _load_manifest(args.out_dir)

    logger.info("Recording %d playthroughs: seeds %s", len(seeds), seeds)
    logger.info("Resuming from seed=%d", start_seed_val)

    # Create one PlayCallback for the whole session — its pyglet window persists
    # across seeds and provides exclusive mouse + keyboard capture.
    play_cb = _RecordingPlayCallback()

    for s_idx in range(seed_start, len(seeds)):
        if play_cb.quit_pressed:
            break

        seed = seeds[s_idx]
        seed_label = f"[Playthrough {s_idx + 1}/{len(seeds)}]"

        ok = _record_seed(seed, seed_label, args, manifest, play_cb)
        if not ok:
            break

        logger.info("\n%s\nPlaythrough %d/%d complete (seed=%d).\n%s",
                    "="*70, s_idx + 1, len(seeds), seed, "="*70)

    diamond_seeds = [e["seed"] for e in manifest if e.get("diamond_found")]
    logger.info("\nSession complete.")
    logger.info("Seeds with diamond: %s", diamond_seeds)
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
    ap.add_argument("--out_dir",    default="data/playthrough_demos")
    ap.add_argument("--seeds",      type=int, nargs="+", default=list(range(20)),
                    help="Seed values to record, in order (default: 0–19)")
    ap.add_argument("--fps",        type=int, default=20)
    ap.add_argument("--start_seed", type=int, default=None,
                    help="Resume from this seed (must be in --seeds list)")
    args = ap.parse_args()

    print("record_playthrough_demos.py")
    print(f"  {'out_dir':12s}: {args.out_dir}")
    print(f"  {'seeds':12s}: {args.seeds}")
    print(f"  {'fps':12s}: {args.fps}")
    print(f"  {'start_seed':12s}: {args.start_seed}")
    print()
    print("Controls:")
    print("  F5  = annotate (describe what you're about to do)")
    print("  F12 = quit session")
    print()
    print("Seed advances automatically when you collect a diamond.")
    print()

    record(args)


if __name__ == "__main__":
    main()
