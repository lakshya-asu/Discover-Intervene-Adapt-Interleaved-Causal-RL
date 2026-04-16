#!/usr/bin/env python3
"""Build GROOT reference clips directly from BC obs data.

Unlike the heuristic build_skill_clips.py (which extracts time-windows
from contractor .mp4 files), this script extracts clips from the
MineRL BC observation arrays (data/minerl_bc/<skill>/obs.npy).
These arrays contain frames from actual successful demonstrations,
so the clips are guaranteed to show skill-relevant behavior.

Strategy per skill:
  - Find windows with high *activity* (many non-noop actions)
  - Export the N_CLIPS best windows as 64×64 MP4 clips (upsampled to 320×180)
  - Also export 1 full-trajectory clip from the middle of the episode

Output: skill_clips/<skill>/bc_clip_{i}.mp4  (overwrites nothing else)
        skill_clips/manifest.json  (updated to include bc_clips)

Usage:
    python scripts/build_bc_clips.py [--skill wood] [--n_clips 5]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import av
    _AV_OK = True
except ImportError:
    _AV_OK = False

REPO_ROOT   = Path(__file__).resolve().parent.parent
BC_ROOT     = REPO_ROOT / "data" / "minerl_bc"
CLIP_ROOT   = REPO_ROOT / "skill_clips"
MANIFEST    = CLIP_ROOT / "manifest.json"

SKILLS = [
    "wood", "stone", "coal", "ironore",
    "furnace", "stonepickaxe", "iron", "ironpickaxe", "diamond",
]

# Clip parameters
CLIP_FRAMES  = 400   # 20 s @ 20 fps
CLIP_FPS     = 20
OUT_W, OUT_H = 320, 180   # GROOT prefers 320x180

# Non-noop action ids
NOOP_ACTIONS = {0}


def _activity_score(act_window: np.ndarray) -> float:
    """Fraction of non-noop frames — higher = more interesting."""
    return float(np.mean(~np.isin(act_window, list(NOOP_ACTIONS))))


def _find_best_windows(
    acts: np.ndarray,
    n_clips: int,
    clip_len: int,
    stride: int = 50,
) -> List[int]:
    """Return start indices of the n_clips most active windows."""
    n = len(acts)
    if n < clip_len:
        return [0]

    scores: List[Tuple[float, int]] = []
    for s in range(0, n - clip_len, stride):
        scores.append((_activity_score(acts[s:s + clip_len]), s))

    scores.sort(reverse=True)
    # Deduplicate: skip windows that overlap with already-selected ones
    selected: List[int] = []
    used = set()
    for _, start in scores:
        if any(abs(start - s) < clip_len // 2 for s in used):
            continue
        selected.append(start)
        used.add(start)
        if len(selected) >= n_clips:
            break

    return selected


def _write_clip_av(frames: np.ndarray, out_path: Path, fps: int = CLIP_FPS) -> bool:
    """Write (T, H, W, 3) uint8 array to MP4 using PyAV."""
    if not _AV_OK:
        return False
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(out_path), mode="w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.width  = OUT_W
            stream.height = OUT_H
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "23", "preset": "fast"}

            for frame_arr in frames:
                # Resize 64×64 → OUT_W×OUT_H using numpy (no cv2)
                from PIL import Image
                img = Image.fromarray(frame_arr).resize((OUT_W, OUT_H), Image.BILINEAR)
                frame = av.VideoFrame.from_ndarray(
                    np.array(img), format="rgb24"
                )
                for pkt in stream.encode(frame):
                    container.mux(pkt)
            for pkt in stream.encode():
                container.mux(pkt)
        return True
    except Exception as e:
        print(f"    [av] write failed: {e}", file=sys.stderr)
        return False


def build_skill_clips(
    skill: str,
    n_clips: int = 5,
    force: bool = False,
) -> List[str]:
    """Build n_clips MP4 reference clips for skill from BC obs data.

    Returns list of relative paths (relative to repo root) of written clips.
    """
    obs_path = BC_ROOT / skill / "obs.npy"
    act_path = BC_ROOT / skill / "actions.npy"

    if not obs_path.exists():
        print(f"  [skip] {skill}: obs.npy not found at {obs_path}")
        return []
    if not act_path.exists():
        print(f"  [skip] {skill}: actions.npy not found at {act_path}")
        return []

    obs  = np.load(str(obs_path),  mmap_mode="r")   # (N, 64, 64, 3)
    acts = np.load(str(act_path),  mmap_mode="r")   # (N,)
    n    = len(obs)
    print(f"  {skill}: {n} frames")

    out_dir = CLIP_ROOT / skill
    out_dir.mkdir(parents=True, exist_ok=True)

    starts = _find_best_windows(acts, n_clips, CLIP_FRAMES, stride=100)
    written: List[str] = []

    for i, start in enumerate(starts):
        end      = min(start + CLIP_FRAMES, n)
        frames   = obs[start:end].copy()   # (T, 64, 64, 3)
        out_path = out_dir / f"bc_clip_{i}.mp4"

        if out_path.exists() and not force:
            print(f"    exists (skip): {out_path.name}")
            written.append(str(out_path.relative_to(REPO_ROOT)))
            continue

        ok = _write_clip_av(frames, out_path)
        if ok:
            sz = out_path.stat().st_size / 1024
            print(f"    wrote {out_path.name}  "
                  f"({len(frames)} frames  act_score={_activity_score(acts[start:end]):.2f}  {sz:.0f}KB)")
            written.append(str(out_path.relative_to(REPO_ROOT)))
        else:
            print(f"    FAILED to write {out_path.name}")

    return written


def update_manifest(new_clips: Dict[str, List[str]]) -> None:
    """Merge new_clips into the existing manifest, prepending bc_clips."""
    existing: Dict[str, List[str]] = {}
    if MANIFEST.exists():
        try:
            existing = json.loads(MANIFEST.read_text())
        except Exception:
            pass

    for skill, paths in new_clips.items():
        old = existing.get(skill, [])
        # Prepend BC clips (higher priority), keep old heuristic clips after
        bc_paths = [p for p in paths if "bc_clip" in p]
        other    = [p for p in old if "bc_clip" not in p]
        existing[skill] = bc_paths + other

    CLIP_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(existing, indent=2))
    print(f"\nManifest updated: {MANIFEST}")
    for sk, clips in existing.items():
        print(f"  {sk}: {len(clips)} clips  (bc={sum(1 for c in clips if 'bc_clip' in c)})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build BC-derived GROOT reference clips")
    ap.add_argument("--skill",   default=None, help="Single skill (default: all)")
    ap.add_argument("--n_clips", type=int, default=5, help="Clips per skill")
    ap.add_argument("--force",   action="store_true", help="Overwrite existing clips")
    args = ap.parse_args()

    if not _AV_OK:
        print("ERROR: PyAV not installed. Run: pip install av", file=sys.stderr)
        sys.exit(1)

    skills = [args.skill] if args.skill else SKILLS
    all_new: Dict[str, List[str]] = {}

    for skill in skills:
        print(f"\n[{skill}]")
        paths = build_skill_clips(skill, n_clips=args.n_clips, force=args.force)
        if paths:
            all_new[skill] = paths

    if all_new:
        update_manifest(all_new)
    else:
        print("\nNo clips written.")


if __name__ == "__main__":
    main()
