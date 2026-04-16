#!/usr/bin/env python3
"""
scripts/build_skill_clips.py — Build GROOT skill-clip reference database.

Extracts 20-second (400-frame @ 20fps) clips from VPT contractor mp4 files
using time-window heuristics that follow typical Minecraft tech-tree progression.
Writes clips as 320x180 h264 mp4s and generates a manifest.json.

Skill windows (frame-offset : start_frame – end_frame):
    wood          :    0 –   400   (  0 – 20 s)
    woodpickaxe   : 1200 –  1600   ( 60 – 80 s)
    stone         : 2000 –  2400   (100 – 120 s)
    coal          : 2400 –  2800   (120 – 140 s)
    furnace       : 3600 –  4000   (180 – 200 s)
    stonepickaxe  : 3200 –  3600   (160 – 180 s)
    ironore       : 4400 –  4800   (220 – 240 s)
    iron          : 5000 –  5400   (250 – 270 s)
    ironpickaxe   : 5400 –  5800   (270 – 290 s)
    diamond       : 5600 –  6000   (280 – 300 s)

Source mp4s: sorted alphabetically, pick indices 0, 40, 80, 120, 160.
If a source mp4 is too short, fall back to the next mp4 in the full pool.

Usage
-----
  # Normal run (from project root)
  conda run -n dia-minecraft python scripts/build_skill_clips.py

  # Dry run (just print, no encoding)
  conda run -n dia-minecraft python scripts/build_skill_clips.py --dry_run

  # Custom paths
  conda run -n dia-minecraft python scripts/build_skill_clips.py \\
      --src_dir /path/to/contractor/data \\
      --out_dir skill_clips \\
      --manifest skill_clips/manifest.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Project root (two levels up from this script) ────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]

# ── Contractor data source ────────────────────────────────────────────────────
# _REPO_ROOT = .../DIA/Discover-Intervene-Adapt-Interleaved-Causal-RL
# parents[0] = .../DIA/Discover-Intervene-Adapt-Interleaved-Causal-RL (same)
# parents[1] = .../DIA
_DEFAULT_SRC = (
    _REPO_ROOT.parent
    / "baselines" / "vpt" / "data" / "contractor" / "data" / "10.0"
)

# ── Output defaults ───────────────────────────────────────────────────────────
_DEFAULT_OUT = _REPO_ROOT / "skill_clips"
_DEFAULT_MANIFEST = _REPO_ROOT / "skill_clips" / "manifest.json"

# ── Video encoding parameters ─────────────────────────────────────────────────
OUT_W = 320
OUT_H = 180
FPS = 20
CLIP_FRAMES = 400   # 20 s × 20 fps

# ── Primary source indices (alphabetically sorted mp4 list) ───────────────────
PRIMARY_INDICES = [0, 40, 80, 120, 160]

# ── Skill definitions: name → (start_frame, end_frame) ───────────────────────
SKILLS: List[Tuple[str, int, int]] = [
    ("wood",         0,    400),
    ("woodpickaxe",  1200, 1600),
    ("stone",        2000, 2400),
    ("coal",         2400, 2800),
    ("furnace",      3600, 4000),
    ("stonepickaxe", 3200, 3600),
    ("ironore",      4400, 4800),
    ("iron",         5000, 5400),
    ("ironpickaxe",  5400, 5800),
    ("diamond",      5600, 6000),
]


# ── av import (lazy so --dry_run works without encoding) ─────────────────────
def _import_av():
    """Import pyav, raising ImportError with a helpful message if missing."""
    try:
        import av
        return av
    except ImportError as exc:
        raise ImportError(
            "pyav is required: conda run -n dia-minecraft pip install av"
        ) from exc


# ── Frame count probe ─────────────────────────────────────────────────────────

def probe_frame_count(mp4_path: Path) -> int:
    """Return the number of video frames in mp4_path using pyav metadata.

    Falls back to iterating frames if the container reports 0.
    """
    av = _import_av()
    try:
        with av.open(str(mp4_path)) as container:
            stream = container.streams.video[0]
            n = stream.frames
            if n and n > 0:
                return int(n)
            # Fallback: count by seeking to end
            # This is slow but reliable for variable-frame containers.
            count = 0
            for _ in container.decode(video=0):
                count += 1
            return count
    except Exception as exc:
        logger.warning("Could not probe %s: %s", mp4_path.name, exc)
        return 0


# ── Clip extraction ───────────────────────────────────────────────────────────

def extract_clip(
    src: Path,
    start: int,
    end: int,
    dst: Path,
    dry_run: bool = False,
) -> bool:
    """Extract frames [start, end) from src, write as h264 mp4 to dst.

    Returns True on success, False on failure.
    """
    if dry_run:
        logger.info("[dry_run] Would extract frames %d-%d from %s → %s",
                    start, end, src.name, dst)
        return True

    av = _import_av()

    frames_needed = end - start
    collected: list = []

    try:
        with av.open(str(src)) as in_c:
            in_stream = in_c.streams.video[0]
            in_stream.thread_type = "AUTO"

            frame_idx = 0
            for packet in in_c.demux(in_stream):
                if len(collected) >= frames_needed:
                    break
                try:
                    for frame in packet.decode():
                        if frame_idx < start:
                            frame_idx += 1
                            continue
                        if frame_idx >= end:
                            break
                        # Resize to OUT_W × OUT_H
                        img = frame.to_image().resize(
                            (OUT_W, OUT_H)
                        )
                        collected.append(img)
                        frame_idx += 1
                except Exception:
                    pass
                if len(collected) >= frames_needed:
                    break
    except Exception as exc:
        logger.error("Failed reading %s: %s", src, exc)
        return False

    if not collected:
        logger.warning("No frames collected from %s (start=%d, end=%d)",
                       src.name, start, end)
        return False

    # Write output clip
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with av.open(str(dst), "w", format="mp4") as out_c:
            out_stream = out_c.add_stream("h264", rate=FPS)
            out_stream.width = OUT_W
            out_stream.height = OUT_H
            out_stream.pix_fmt = "yuv420p"
            out_stream.options = {"crf": "23", "preset": "fast"}

            from av import VideoFrame
            for img in collected:
                vf = VideoFrame.from_image(img)
                vf = vf.reformat(format="yuv420p")
                for pkt in out_stream.encode(vf):
                    out_c.mux(pkt)
            # Flush
            for pkt in out_stream.encode(None):
                out_c.mux(pkt)
    except Exception as exc:
        logger.error("Failed writing %s: %s", dst, exc)
        if dst.exists():
            dst.unlink()
        return False

    logger.info("Wrote %d frames → %s", len(collected), dst)
    return True


# ── Main build logic ──────────────────────────────────────────────────────────

def build_skill_clips(
    src_dir: Path,
    out_dir: Path,
    manifest_path: Path,
    dry_run: bool = False,
) -> Dict[str, List[str]]:
    """Build the full skill-clip database.

    Returns the manifest dict (skill → list of relative clip paths).
    """
    # Collect all mp4s, sorted alphabetically
    all_mp4s: List[Path] = sorted(src_dir.glob("*.mp4"))
    if not all_mp4s:
        logger.error("No mp4 files found in %s", src_dir)
        sys.exit(1)

    logger.info("Found %d mp4 files in %s", len(all_mp4s), src_dir)

    # Build primary pool: 5 mp4s at specified indices
    primary_pool: List[Path] = []
    for idx in PRIMARY_INDICES:
        if idx < len(all_mp4s):
            primary_pool.append(all_mp4s[idx])
        else:
            logger.warning(
                "Primary index %d exceeds pool size (%d), using last mp4",
                idx, len(all_mp4s)
            )
            primary_pool.append(all_mp4s[-1])

    logger.info(
        "Primary pool (%d files): %s",
        len(primary_pool),
        [p.name for p in primary_pool],
    )

    # Fallback pool: all mp4s not already in primary pool
    primary_set = set(p.name for p in primary_pool)
    fallback_pool: List[Path] = [p for p in all_mp4s if p.name not in primary_set]

    # Cache probed frame counts to avoid repeated seeks
    frame_count_cache: Dict[str, int] = {}

    def get_frame_count(p: Path) -> int:
        if p.name not in frame_count_cache:
            if dry_run:
                # In dry_run, assume 6000 frames for primaries
                frame_count_cache[p.name] = 6000
            else:
                frame_count_cache[p.name] = probe_frame_count(p)
        return frame_count_cache[p.name]

    manifest: Dict[str, List[str]] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    for skill_name, start_frame, end_frame in SKILLS:
        logger.info(
            "Processing skill: %s (frames %d–%d)",
            skill_name, start_frame, end_frame,
        )
        skill_dir = out_dir / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)

        clip_paths: List[str] = []
        clip_slot = 0

        # Try each primary source in turn
        for primary_mp4 in primary_pool:
            if clip_slot >= 5:
                break

            dst = skill_dir / f"clip_{clip_slot}.mp4"

            # Check if primary has enough frames
            n_frames = get_frame_count(primary_mp4)
            if n_frames < end_frame:
                logger.info(
                    "  %s only has %d frames (need %d for %s) — trying fallbacks",
                    primary_mp4.name, n_frames, end_frame, skill_name,
                )
                # Search fallback pool
                found_fallback = False
                for fb_mp4 in fallback_pool:
                    fb_frames = get_frame_count(fb_mp4)
                    if fb_frames >= end_frame:
                        logger.info(
                            "  Fallback: %s (%d frames)", fb_mp4.name, fb_frames
                        )
                        ok = extract_clip(
                            fb_mp4, start_frame, end_frame, dst, dry_run=dry_run
                        )
                        if ok:
                            rel = str(dst.relative_to(_REPO_ROOT))
                            clip_paths.append(rel)
                            clip_slot += 1
                            found_fallback = True
                            break

                if not found_fallback:
                    logger.warning(
                        "  No mp4 with >= %d frames found for %s clip_%d — skipping",
                        end_frame, skill_name, clip_slot,
                    )
                continue

            # Primary has enough frames
            ok = extract_clip(
                primary_mp4, start_frame, end_frame, dst, dry_run=dry_run
            )
            if ok:
                rel = str(dst.relative_to(_REPO_ROOT))
                clip_paths.append(rel)
                clip_slot += 1
            else:
                logger.warning(
                    "  extract_clip failed for %s clip_%d", skill_name, clip_slot
                )

        manifest[skill_name] = clip_paths
        logger.info(
            "  %s: %d/%d clips written", skill_name, len(clip_paths), 5
        )

    # Write manifest
    if not dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Manifest written to %s", manifest_path)
    else:
        logger.info("[dry_run] Manifest would be written to %s", manifest_path)
        logger.info("[dry_run] Manifest preview:\n%s", json.dumps(manifest, indent=2))

    return manifest


# ── Verification ──────────────────────────────────────────────────────────────

def verify_clips(manifest: Dict[str, List[str]]) -> None:
    """Open each clip with pyav and confirm it is a valid mp4."""
    av = _import_av()
    print("\nVerification:")
    all_ok = True
    for skill, paths in manifest.items():
        for rel_path in paths:
            abs_path = _REPO_ROOT / rel_path
            try:
                with av.open(str(abs_path)) as c:
                    n = c.streams.video[0].frames
                print(f"  OK   {rel_path}  ({n} frames reported)")
            except Exception as exc:
                print(f"  FAIL {rel_path}: {exc}")
                all_ok = False
    if all_ok:
        print("All clips verified OK.")
    else:
        print("Some clips failed verification.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(
        description="Build GROOT skill-clip reference database from VPT contractor mp4s.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--src_dir",
        type=Path,
        default=_DEFAULT_SRC,
        help=f"Directory containing contractor mp4 files (default: {_DEFAULT_SRC})",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output directory for skill clips (default: {_DEFAULT_OUT})",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=_DEFAULT_MANIFEST,
        help=f"Path for manifest.json (default: {_DEFAULT_MANIFEST})",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be extracted without actually encoding any video",
    )
    ap.add_argument(
        "--no_verify",
        action="store_true",
        help="Skip post-build verification of written clips",
    )
    args = ap.parse_args()

    print("Build Skill Clips")
    print(f"  src_dir  : {args.src_dir}")
    print(f"  out_dir  : {args.out_dir}")
    print(f"  manifest : {args.manifest}")
    print(f"  dry_run  : {args.dry_run}")
    print(f"  skills   : {[s for s, _, _ in SKILLS]}")
    print(f"  clips/skill: 5 (indices {PRIMARY_INDICES})")
    print()

    if not args.src_dir.exists():
        logger.error("src_dir does not exist: %s", args.src_dir)
        sys.exit(1)

    manifest = build_skill_clips(
        src_dir=args.src_dir,
        out_dir=args.out_dir,
        manifest_path=args.manifest,
        dry_run=args.dry_run,
    )

    # Summary
    print("\nSummary:")
    total_clips = 0
    skills_under_3: List[str] = []
    for skill, paths in manifest.items():
        n = len(paths)
        total_clips += n
        flag = "" if n >= 3 else "  *** FEWER THAN 3 CLIPS ***"
        print(f"  {skill:<16}  {n} clip(s){flag}")
        if n < 3:
            skills_under_3.append(skill)

    print(f"\nTotal clips: {total_clips}")
    if skills_under_3:
        print(f"WARNING: skills with fewer than 3 clips: {skills_under_3}")
    else:
        print("All skills have >= 3 clips.")

    if not args.dry_run and not args.no_verify:
        verify_clips(manifest)


if __name__ == "__main__":
    main()
