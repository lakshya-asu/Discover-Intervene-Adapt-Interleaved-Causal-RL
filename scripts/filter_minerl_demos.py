#!/usr/bin/env python3
"""
scripts/filter_minerl_demos.py  —  Extract per-skill BC windows from MineRL demos.

Scans MineRL human demonstration episodes for moments where a specific inventory
item count increases, then extracts the preceding --window steps as behavioral
cloning (BC) training data for that skill.

Data format supported (MineRL 1.0.x / BASALT 2022):
  <data_dir>/<EnvName>/<episode_id>/
      *.jsonl     — recorded actions + state (one JSON object per line)
      *.mp4       — video recording (frames = observations)

Each JSONL line contains at minimum:
    inventory : list of {type: str, quantity: int}
    keyboard  : {keys: [str]}
    mouse     : {buttons: [int], dx: float, dy: float, dwheel: float,
                  x: float, y: float}
    pitch, yaw, xpos, ypos, zpos : floats
    tick, isGuiOpen : int/bool

Output (per skill):
    <out_dir>/<skill_name>/obs.npy     — uint8 array (N, H, W, 3)
    <out_dir>/<skill_name>/actions.npy — int32 array (N,)

Actions are encoded as indices into the DIA discrete action set (N_ACTIONS=25)
defined in src/dia/options_minerl.py.  The conversion uses a heuristic mapping
from the raw minerec keyboard+mouse state to the closest discrete action index.

Usage
-----
  # First, download the dataset (see instructions below if data is missing)
  conda run -n dia-minecraft python scripts/filter_minerl_demos.py \\
      --data_dir ~/minerl_data \\
      --out_dir  data/minerl_bc \\
      --window   100 \\
      --max_windows 500

  # Quick smoke test (5 windows max per skill):
  conda run -n dia-minecraft python scripts/filter_minerl_demos.py --max_windows 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Repo path setup ───────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ── Skill → inventory item keys ───────────────────────────────────────────────
SKILL_VARS: Dict[str, List[str]] = {
    "wood":         ["oak_log", "birch_log", "spruce_log", "jungle_log",
                     "acacia_log", "dark_oak_log", "log", "log2",
                     "oak_wood", "birch_wood", "spruce_wood", "jungle_wood",
                     "acacia_wood", "dark_oak_wood"],
    "stone":        ["cobblestone", "cobbled_deepslate", "stone"],
    "coal":         ["coal"],
    "ironore":      ["iron_ore"],
    "furnace":      ["furnace"],
    "stonepickaxe": ["stone_pickaxe"],
    "iron":         ["iron_ingot"],
    "ironpickaxe":  ["iron_pickaxe"],
    "diamond":      ["diamond"],
}

SKILL_NAMES = list(SKILL_VARS.keys())

# ── MineRL env names (priority order) ─────────────────────────────────────────
# MineRL 1.0.x available envs that have full tech-tree demos:
_CANDIDATE_ENV_NAMES = [
    "MineRLObtainDiamondShovel-v0",   # best: full diamond tech-tree
    "MineRLHumanSurvival-v0",          # fallback: general survival
    "MineRLBasaltFindCave-v0",         # partial tech-tree
    "MineRLBasaltBuildVillageHouse-v0",
    "MineRLBasaltMakeWaterfall-v0",
    "MineRLBasaltCreateVillageAnimalPen-v0",
    "MineRLTreechop-v0",               # wood only
]

# ── Frame size ─────────────────────────────────────────────────────────────────
OBS_H = 64
OBS_W = 64

# ── DIA discrete action definitions (mirrors options_minerl.py) ──────────────
# Each entry: list of (key, value) pairs that are set to 1 in the action dict.
# camera entries are [pitch_delta, yaw_delta] in degrees.
_ACTION_DEFS: List[List[Tuple[str, Any]]] = [
    [],                                               # 0:  noop
    [("forward", 1)],                                 # 1:  forward
    [("back", 1)],                                    # 2:  back
    [("left", 1)],                                    # 3:  strafe left
    [("right", 1)],                                   # 4:  strafe right
    [("jump", 1)],                                    # 5:  jump
    [("attack", 1)],                                  # 6:  attack
    [("use", 1)],                                     # 7:  use
    [("forward", 1), ("attack", 1)],                  # 8:  forward + attack
    [("forward", 1), ("jump", 1)],                    # 9:  forward + jump
    [("forward", 1), ("sprint", 1)],                  # 10: sprint forward
    [("camera", np.array([-15.0, 0.0]))],             # 11: camera up (coarse)
    [("camera", np.array([15.0, 0.0]))],              # 12: camera down (coarse)
    [("camera", np.array([0.0, -15.0]))],             # 13: camera left (coarse)
    [("camera", np.array([0.0, 15.0]))],              # 14: camera right (coarse)
    [("forward", 1), ("sprint", 1), ("attack", 1)],   # 15: sprint + attack
    [("camera", np.array([-5.0, 0.0]))],              # 16: camera up (fine)
    [("camera", np.array([5.0, 0.0]))],               # 17: camera down (fine)
    [("camera", np.array([0.0, -5.0]))],              # 18: camera left (fine)
    [("camera", np.array([0.0, 5.0]))],               # 19: camera right (fine)
    [("hotbar.1", 1)],                                # 20: equip slot 1
    [("hotbar.2", 1)],                                # 21: equip slot 2
    [("forward", 1), ("use", 1)],                     # 22: forward + use
    [("jump", 1), ("attack", 1)],                     # 23: jump + attack
    [("forward", 1), ("jump", 1), ("attack", 1)],     # 24: fwd + jump + attack
]
N_ACTIONS = len(_ACTION_DEFS)

# Camera threshold (degrees) to decide "coarse vs fine" camera move
_CAM_COARSE_THRESH = 10.0
_CAM_FINE_THRESH = 2.0  # below this → noop for camera


def _minerec_to_discrete(step: Dict[str, Any]) -> int:
    """Convert a minerec JSONL step dict to a DIA discrete action index.

    Uses a heuristic rule-based mapping from keyboard/mouse state.
    Priority: attack > use > movement > camera.
    """
    kb_keys = set(step.get("keyboard", {}).get("keys", []))
    mouse_buttons = set(step.get("mouse", {}).get("buttons", []))

    forward = "key.keyboard.w" in kb_keys
    back    = "key.keyboard.s" in kb_keys
    left    = "key.keyboard.a" in kb_keys
    right   = "key.keyboard.d" in kb_keys
    jump    = "key.keyboard.space" in kb_keys
    sprint  = "key.keyboard.left.control" in kb_keys
    attack  = 0 in mouse_buttons   # left-click
    use     = 1 in mouse_buttons   # right-click

    # Camera: compute pitch/yaw delta from mouse dx/dy
    mouse = step.get("mouse", {})
    camera_scaler = 360 / 2400
    pitch_d = float(mouse.get("dy", 0.0)) * camera_scaler
    yaw_d   = float(mouse.get("dx", 0.0)) * camera_scaler

    # ── Rule-based mapping (highest-priority first) ──────────────────────────
    if forward and sprint and attack:
        return 15
    if forward and jump and attack:
        return 24
    if jump and attack:
        return 23
    if forward and attack:
        return 8
    if forward and use:
        return 22
    if attack:
        return 6
    if use:
        return 7
    if forward and sprint:
        return 10
    if forward and jump:
        return 9
    if forward:
        return 1
    if back:
        return 2
    if left:
        return 3
    if right:
        return 4
    if jump:
        return 5

    # Camera movement (check coarse then fine)
    if abs(pitch_d) >= _CAM_COARSE_THRESH:
        return 11 if pitch_d < 0 else 12  # up / down
    if abs(yaw_d) >= _CAM_COARSE_THRESH:
        return 13 if yaw_d < 0 else 14   # left / right
    if abs(pitch_d) >= _CAM_FINE_THRESH:
        return 16 if pitch_d < 0 else 17
    if abs(yaw_d) >= _CAM_FINE_THRESH:
        return 18 if yaw_d < 0 else 19

    return 0  # noop


def _inventory_count(step: Dict[str, Any], item_keys: List[str]) -> int:
    """Sum the inventory quantities for the given item keys in one step."""
    total = 0
    inv = step.get("inventory", [])
    if isinstance(inv, list):
        # BASALT 2022 format: list of {type: str, quantity: int}
        for entry in inv:
            if entry.get("type", "") in item_keys:
                total += int(entry.get("quantity", 0))
    elif isinstance(inv, dict):
        # Alternative flat dict format
        for k in item_keys:
            total += int(inv.get(k, 0))
    return total


def _load_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load all steps from a JSONL file. Each line is one step dict."""
    steps = []
    with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                steps.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return steps


def _load_mp4_frames(mp4_path: str) -> Optional[np.ndarray]:
    """Load all frames from an MP4 as uint8 (N, H, W, 3) BGR→RGB.

    Returns None if cv2 is unavailable or the file cannot be read.
    """
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return None
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Convert BGR→RGB and resize to OBS_H×OBS_W
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_small = cv2.resize(frame_rgb, (OBS_W, OBS_H),
                                 interpolation=cv2.INTER_AREA)
        frames.append(frame_small)
    cap.release()
    if not frames:
        return None
    return np.stack(frames, axis=0)  # (N, OBS_H, OBS_W, 3)


def _find_episode_dirs(data_dir: str) -> List[str]:
    """Return all episode directories under data_dir.

    An episode directory contains at least one .jsonl file.
    Searches recursively up to depth 3 to handle:
        data_dir/                     (flat)
        data_dir/<EnvName>/           (one level)
        data_dir/<EnvName>/<episode>/ (two levels)
    """
    episode_dirs = []
    data_path = Path(data_dir)
    if not data_path.exists():
        return episode_dirs

    # Collect all dirs that contain a .jsonl file
    for jsonl_file in data_path.rglob("*.jsonl"):
        ep_dir = str(jsonl_file.parent)
        if ep_dir not in episode_dirs:
            episode_dirs.append(ep_dir)
    return sorted(episode_dirs)


def _find_paired_files(ep_dir: str) -> List[Tuple[str, Optional[str]]]:
    """Return list of (jsonl_path, mp4_path_or_None) pairs in ep_dir.

    Handles VPT and BASALT formats:
      VPT format:   episode.jsonl + episode.mp4
      BASALT format: episode.jsonl + episode.mp4
    Both use stem + .mp4 for the video.
    """
    ep_path = Path(ep_dir)
    pairs = []
    for jsonl_file in sorted(ep_path.glob("*.jsonl")):
        # Skip .mp4.jsonl files — these are binary metadata, not action data
        if jsonl_file.name.endswith(".mp4.jsonl"):
            continue
        stem = jsonl_file.stem  # e.g. "episode" (before the final .jsonl)
        mp4_file = ep_path / (stem + ".mp4")
        pairs.append((
            str(jsonl_file),
            str(mp4_file) if mp4_file.exists() else None,
        ))
    return pairs


def extract_skill_windows(
    ep_dir: str,
    window: int,
    skill_name: str,
    item_keys: List[str],
    max_windows: int,
    current_count: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Extract BC windows from one episode for one skill.

    Scans the JSONL for timesteps where the total inventory count of
    `item_keys` increases. Extracts the preceding `window` steps (obs
    frames + discrete actions).

    Returns list of (obs_window, action_window) tuples where:
        obs_window    : (window, OBS_H, OBS_W, 3) uint8
        action_window : (window,) int32

    Stops early if current_count + len(results) >= max_windows.
    """
    remaining = max_windows - current_count
    if remaining <= 0:
        return []

    pairs = _find_paired_files(ep_dir)
    if not pairs:
        return []

    results = []

    for jsonl_path, mp4_path in pairs:
        if remaining <= 0:
            break

        # Load action/state sequence
        steps = _load_jsonl(jsonl_path)
        if len(steps) < window + 1:
            continue  # episode too short

        # Load video frames (may be None if cv2 not available or mp4 missing)
        frames: Optional[np.ndarray] = None
        if mp4_path is not None:
            frames = _load_mp4_frames(mp4_path)
            if frames is not None and len(frames) != len(steps):
                # Frame count mismatch: interpolate or trim
                if len(frames) < len(steps):
                    frames = None  # too few frames, skip video
                else:
                    frames = frames[: len(steps)]

        # Convert all steps to discrete actions
        actions = np.array(
            [_minerec_to_discrete(s) for s in steps], dtype=np.int32
        )

        # Scan for inventory increase events
        prev_count = _inventory_count(steps[0], item_keys)
        for t in range(1, len(steps)):
            curr_count = _inventory_count(steps[t], item_keys)
            if curr_count > prev_count:
                # Acquisition event at step t — extract preceding window
                start = max(0, t - window)
                end = t  # exclusive; we include step t's action too if possible

                if frames is not None:
                    obs_window = frames[start:end].copy()
                else:
                    # Fallback: create placeholder zero frames
                    length = end - start
                    obs_window = np.zeros(
                        (length, OBS_H, OBS_W, 3), dtype=np.uint8
                    )

                act_window = actions[start:end].copy()

                # Pad to exactly `window` steps if near episode start
                if len(act_window) < window:
                    pad = window - len(act_window)
                    obs_window = np.concatenate(
                        [np.zeros((pad, OBS_H, OBS_W, 3), dtype=np.uint8),
                         obs_window], axis=0
                    )
                    act_window = np.concatenate(
                        [np.zeros(pad, dtype=np.int32), act_window], axis=0
                    )

                results.append((obs_window, act_window))
                remaining -= 1
                if remaining <= 0:
                    break

            prev_count = curr_count

    return results


def discover_env_names(data_dir: str) -> List[str]:
    """Return candidate env-named subdirectories found in data_dir."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    found = []
    for entry in sorted(data_path.iterdir()):
        if entry.is_dir() and any(
            entry.name.startswith(prefix)
            for prefix in ["MineRL", "minerl", "basalt", "Basalt"]
        ):
            found.append(entry.name)
    return found


def print_download_instructions(data_dir: str) -> None:
    """Print instructions for downloading MineRL demo data."""
    print("\n" + "=" * 72)
    print("NO MINERL DEMO DATA FOUND")
    print("=" * 72)
    print(f"\nExpected data directory: {data_dir}")
    print(
        "\nMineRL 1.0.x (BASALT 2022) demo data can be downloaded from:"
    )
    print("  https://github.com/minerllabs/basalt-2022-competition-dataset")
    print("\nDownload commands (requires ~50 GB free disk space):")
    print()
    print("  # Option A: Download BASALT 2022 competition dataset via aicrowd CLI")
    print("  pip install aicrowd-cli")
    print("  aicrowd login")
    print(f"  mkdir -p {data_dir}")
    print(
        "  aicrowd dataset download --challenge basalt-2022-behavioural-cloning-baseline"
        f" -o {data_dir}"
    )
    print()
    print("  # Option B: Download only MineRLObtainDiamondShovel dataset")
    print("  #   (approx. 35 GB, ~2200 episodes, best coverage of all 9 skills)")
    print(
        "  # See: https://github.com/minerllabs/basalt-2022-competition-dataset"
        "#downloading-the-basalt-dataset"
    )
    print()
    print(
        "  # Option C: Use the OpenAI Video PreTraining (VPT) contractor data"
    )
    print(
        "  #   https://github.com/openai/Video-PreTraining"
        "#downloading-the-dataset"
    )
    print(
        "  # Store as: {data_dir}/MineRLObtainDiamondShovel-v0/<episode>/*.jsonl"
    )
    print()
    print(
        "After downloading, re-run:"
    )
    print(
        f"  conda run -n dia-minecraft python scripts/filter_minerl_demos.py"
        f" --data_dir {data_dir}"
    )
    print("=" * 72 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Extract per-skill BC training windows from MineRL human demo data.\n"
            "Scans episodes for inventory acquisition events, extracts preceding\n"
            f"--window steps as (obs, action) BC training data for each of the\n"
            f"9 DIA skills: {', '.join(SKILL_NAMES)}."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--data_dir",
        type=str,
        default=os.path.expanduser("~/minerl_data"),
        help=(
            "Root directory containing MineRL demo episodes. "
            "Expected layout: <data_dir>/<EnvName>/<episode>/*.jsonl  "
            "(default: ~/minerl_data)"
        ),
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="data/minerl_bc",
        help=(
            "Output directory. Per-skill arrays saved as "
            "<out_dir>/<skill>/obs.npy and <out_dir>/<skill>/actions.npy  "
            "(default: data/minerl_bc)"
        ),
    )
    ap.add_argument(
        "--window",
        type=int,
        default=100,
        help="Number of steps to extract before each acquisition event (default: 100)",
    )
    ap.add_argument(
        "--max_windows",
        type=int,
        default=500,
        help="Maximum BC windows to collect per skill (default: 500)",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Scan episodes and print statistics without saving any files",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-episode progress",
    )
    args = ap.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    out_dir = os.path.expanduser(args.out_dir)

    print(f"MineRL BC Data Extractor")
    print(f"  data_dir   : {data_dir}")
    print(f"  out_dir    : {out_dir}")
    print(f"  window     : {args.window}")
    print(f"  max_windows: {args.max_windows} per skill")
    print()

    # ── Discover episode directories ─────────────────────────────────────────
    episode_dirs = _find_episode_dirs(data_dir)
    if not episode_dirs:
        print_download_instructions(data_dir)
        sys.exit(0)

    print(f"Found {len(episode_dirs)} episode director(y/ies) under {data_dir}")
    env_names_found = discover_env_names(data_dir)
    if env_names_found:
        print(f"  Env subdirs: {env_names_found}")
    print()

    # ── Check cv2 availability ────────────────────────────────────────────────
    try:
        import cv2 as _cv2
        _CV2_OK = True
    except ImportError:
        _CV2_OK = False
        print(
            "WARNING: cv2 (opencv-python) not found. "
            "Video frames will be saved as zero arrays.\n"
            "  Install with: pip install opencv-python-headless\n"
        )

    # ── Extract windows per skill ─────────────────────────────────────────────
    skill_windows: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
        skill: [] for skill in SKILL_NAMES
    }

    for ep_idx, ep_dir in enumerate(episode_dirs):
        # Check if all skills have enough windows
        all_done = all(
            len(skill_windows[s]) >= args.max_windows for s in SKILL_NAMES
        )
        if all_done:
            print("All skills have reached max_windows. Stopping early.")
            break

        if args.verbose:
            print(f"  [{ep_idx + 1}/{len(episode_dirs)}] {ep_dir}")

        for skill_name in SKILL_NAMES:
            if len(skill_windows[skill_name]) >= args.max_windows:
                continue
            item_keys = SKILL_VARS[skill_name]
            new_windows = extract_skill_windows(
                ep_dir=ep_dir,
                window=args.window,
                skill_name=skill_name,
                item_keys=item_keys,
                max_windows=args.max_windows,
                current_count=len(skill_windows[skill_name]),
            )
            skill_windows[skill_name].extend(new_windows)
            if new_windows and args.verbose:
                print(
                    f"    {skill_name}: +{len(new_windows)} windows "
                    f"(total {len(skill_windows[skill_name])})"
                )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\nExtraction summary:")
    print(f"  {'Skill':<16}  {'Windows':>8}  {'Obs shape':<22}  {'Actions shape'}")
    print("  " + "-" * 64)
    for skill_name in SKILL_NAMES:
        windows = skill_windows[skill_name]
        if windows:
            obs_shape = (len(windows) * args.window,) + windows[0][0].shape[1:]
            act_shape = (len(windows) * args.window,)
            print(
                f"  {skill_name:<16}  {len(windows):>8}  "
                f"{str(obs_shape):<22}  {act_shape}"
            )
        else:
            print(f"  {skill_name:<16}  {'0':>8}  {'—':<22}  —")

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n[dry_run] No files saved.")
        return

    total_saved = 0
    for skill_name in SKILL_NAMES:
        windows = skill_windows[skill_name]
        if not windows:
            continue

        skill_dir = os.path.join(out_dir, skill_name)
        os.makedirs(skill_dir, exist_ok=True)

        # Stack all windows into single arrays
        all_obs = np.concatenate([w[0] for w in windows], axis=0)   # (N, H, W, 3)
        all_act = np.concatenate([w[1] for w in windows], axis=0)   # (N,)

        obs_path = os.path.join(skill_dir, "obs.npy")
        act_path = os.path.join(skill_dir, "actions.npy")
        np.save(obs_path, all_obs)
        np.save(act_path, all_act)

        total_saved += 1
        print(
            f"  Saved {skill_name}: obs={all_obs.shape} → {obs_path}\n"
            f"           actions={all_act.shape} → {act_path}"
        )

    if total_saved == 0:
        print(
            "\nNo windows were extracted. "
            "Check that your data_dir contains .jsonl episode files."
        )
        print_download_instructions(data_dir)
    else:
        print(f"\nDone. Saved BC data for {total_saved} skill(s) to: {out_dir}")


if __name__ == "__main__":
    main()
