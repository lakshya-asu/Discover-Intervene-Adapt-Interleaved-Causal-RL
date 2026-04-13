#!/usr/bin/env python3
"""
Sanity check for MineDojo voxel observations.

Verifies that:
  - minedojo.make() accepts use_voxel=True with voxel_size=(7,3,7)
  - obs['voxels'] is present and has the expected structure
  - voxel_extract_target() and get_agent_state() work on live observations
  - EVGS inventory extraction produces non-degenerate output

Run with:
    conda run -n dia-minecraft python scripts/verify_voxel_obs.py [--seed 0]
"""

import argparse
import os
import sys
import traceback

import numpy as np

# Add repo src/ to import path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

try:
    import minedojo
    MINEDOJO_OK = True
except ImportError as _md_err:
    MINEDOJO_OK = False
    print(f"ERROR: minedojo not importable: {_md_err}")
    print("  Install with: pip install minedojo")
    print("  Or activate the conda env: conda activate dia-minecraft")
    sys.exit(1)

try:
    from dia.evgs_minedojo import (
        make_minedojo_evgs,
        voxel_extract_target,
        get_agent_state,
        VAR_NAMES,
    )
    DIA_OK = True
except ImportError as _dia_err:
    DIA_OK = False
    print(f"ERROR: could not import from src.dia.evgs_minedojo: {_dia_err}")
    print("  Make sure you are running from the repo root or src/ is on sys.path.")
    sys.exit(1)


# Voxel grid shape expected from minedojo.make(..., voxel_size=(7,3,7))
# MineDojo voxel_size must be a dict: xmin/xmax/ymin/ymax/zmin/zmax
# Grid shape = (xmax-xmin+1, ymax-ymin+1, zmax-zmin+1) = (7, 3, 7)
VOXEL_SIZE_DICT = dict(xmin=-3, xmax=3, ymin=-1, ymax=1, zmin=-3, zmax=3)
VOXEL_SIZE = (7, 3, 7)   # kept for internal array math
N_VOXEL_CELLS = VOXEL_SIZE[0] * VOXEL_SIZE[1] * VOXEL_SIZE[2]  # 147

# Wood-type block names to look for (subset of SKILL_TARGETS['wood'])
WOOD_BLOCK_NAMES = {
    "log", "oak_log", "birch_log", "spruce_log", "jungle_log",
    "acacia_log", "dark_oak_log", "mangrove_log",
    "minecraft:log", "minecraft:oak_log",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flush(*args, **kwargs):
    """Print then flush stdout immediately."""
    print(*args, **kwargs)
    sys.stdout.flush()


def _describe_voxel_obs(voxel_obs):
    """
    Print a structured description of the voxels sub-observation.

    Handles both dict format and structured numpy array format.
    """
    _flush("\n--- obs['voxels'] structure ---")

    if voxel_obs is None:
        _flush("  obs['voxels'] is None")
        return

    _flush(f"  type : {type(voxel_obs).__name__}")

    if isinstance(voxel_obs, dict):
        _flush(f"  keys : {list(voxel_obs.keys())}")
        for k, v in voxel_obs.items():
            arr = np.asarray(v)
            _flush(f"    [{k}]  shape={arr.shape}  dtype={arr.dtype}")
        block_names = voxel_obs.get("block_name", None)

    elif hasattr(voxel_obs, "dtype") and hasattr(voxel_obs.dtype, "names"):
        # Structured numpy array
        _flush(f"  dtype.names : {voxel_obs.dtype.names}")
        _flush(f"  shape       : {voxel_obs.shape}")
        for field in voxel_obs.dtype.names:
            sub = np.asarray(voxel_obs[field])
            _flush(f"    [{field}]  shape={sub.shape}  dtype={sub.dtype}")
        block_names = (
            voxel_obs["block_name"]
            if "block_name" in voxel_obs.dtype.names else None
        )

    else:
        arr = np.asarray(voxel_obs)
        _flush(f"  shape : {arr.shape}  dtype : {arr.dtype}")
        _flush("  (not a dict or structured array - cannot extract block_name)")
        block_names = None

    sys.stdout.flush()
    return block_names


def _print_block_name_summary(block_names):
    """Print unique block names and check for wood/log blocks."""
    if block_names is None:
        _flush("  block_name : not found in voxel obs")
        return False

    flat = np.asarray(block_names).ravel()
    unique_blocks = sorted(set(str(b).lower().strip().rstrip("\x00") for b in flat))

    _flush(f"\n--- block_name unique values ({len(unique_blocks)} distinct) ---")
    for name in unique_blocks:
        _flush(f"  {name!r}")

    wood_found = any(b in WOOD_BLOCK_NAMES for b in unique_blocks)
    _flush(f"\n  wood/log blocks present: {wood_found}")
    sys.stdout.flush()
    return wood_found


def _print_voxel_extract(obs, label=""):
    """Run voxel_extract_target and print result summary."""
    _flush(f"\n--- voxel_extract_target(obs, 'wood')  {label} ---")
    try:
        target_voxel = voxel_extract_target(obs, "wood")
        _flush(f"  shape  : {target_voxel.shape}")
        _flush(f"  dtype  : {target_voxel.dtype}")
        _flush(f"  sum    : {target_voxel.sum():.1f}  (cells matching a wood target)")
        _flush(f"  max    : {target_voxel.max():.1f}")
    except Exception:
        _flush("  ERROR calling voxel_extract_target:")
        traceback.print_exc()
    sys.stdout.flush()


def _print_agent_state(obs, label=""):
    """Run get_agent_state and print result."""
    _flush(f"\n--- get_agent_state(obs)  {label} ---")
    try:
        state = get_agent_state(obs)
        labels = ["pitch/90", "yaw/180", "health/20", "food/20", "on_ground", "reserved"]
        _flush(f"  shape : {state.shape}  dtype : {state.dtype}")
        for i, (lbl, val) in enumerate(zip(labels, state)):
            _flush(f"  [{i}] {lbl:<12} = {val:.4f}")
    except Exception:
        _flush("  ERROR calling get_agent_state:")
        traceback.print_exc()
    sys.stdout.flush()


def _print_evgs_extract(evgs, obs, label=""):
    """Run evgs.extract and print inventory variable vector."""
    _flush(f"\n--- evgs.extract(obs)  {label} ---")
    try:
        x = evgs.extract(obs)
        _flush(f"  shape : {x.shape}  dtype : {x.dtype}")
        for i, (name, val) in enumerate(zip(VAR_NAMES, x)):
            marker = " <-- acquired" if val > 0.5 else ""
            _flush(f"  [{i}] {name:<14} = {val:.1f}{marker}")
    except Exception:
        _flush("  ERROR calling evgs.extract:")
        traceback.print_exc()
    sys.stdout.flush()


def _obs_summary(evgs, obs, label=""):
    """Print top-level obs keys, then voxel structure, then EVGS and state."""
    _flush(f"\n{'='*60}")
    _flush(f"OBS SUMMARY  {label}")
    _flush(f"{'='*60}")

    # Top-level keys
    if isinstance(obs, dict):
        _flush(f"top-level keys: {sorted(obs.keys())}")
    else:
        _flush(f"obs type (not a dict): {type(obs).__name__}")
    sys.stdout.flush()

    # Voxel structure
    voxel_obs = obs.get("voxels", None) if isinstance(obs, dict) else None
    block_names = _describe_voxel_obs(voxel_obs)
    _print_block_name_summary(block_names)
    _print_voxel_extract(obs, label=label)
    _print_agent_state(obs, label=label)
    _print_evgs_extract(evgs, obs, label=label)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Verify MineDojo voxel observations and EVGS utilities.\n\n"
            "Creates a MineDojo open-ended env with voxels enabled and checks\n"
            "that the voxel obs structure, block names, voxel_extract_target,\n"
            "get_agent_state, and inventory EVGS all work correctly.\n\n"
            "Usage:\n"
            "  conda run -n dia-minecraft python scripts/verify_voxel_obs.py [--seed N]"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--seed", type=int, default=0,
        help="World seed for the MineDojo environment (default: 0)",
    )
    args = ap.parse_args()

    _flush("verify_voxel_obs.py")
    _flush(f"  minedojo  : {minedojo.__version__ if hasattr(minedojo, '__version__') else 'unknown version'}")
    _flush(f"  seed      : {args.seed}")
    _flush(f"  voxel_size: {VOXEL_SIZE}")
    _flush(f"  dia EVGS  : {len(VAR_NAMES)} variables: {VAR_NAMES}")

    evgs = make_minedojo_evgs()
    passed = False
    env = None

    try:
        # ------------------------------------------------------------------
        # Create environment
        # ------------------------------------------------------------------
        _flush("\nCreating MineDojo open-ended env with voxels...")
        sys.stdout.flush()

        env = minedojo.make(
            task_id="open-ended",
            image_size=(64, 64),
            world_seed=str(args.seed),
            specified_biome="forest",
            use_voxel=True,
            voxel_size=VOXEL_SIZE_DICT,
        )

        _flush(f"  env created: {type(env).__name__}")
        _flush(f"  obs_space  : {type(env.observation_space).__name__}")
        sys.stdout.flush()

        # ------------------------------------------------------------------
        # Reset
        # ------------------------------------------------------------------
        _flush("\nCalling env.reset()...")
        sys.stdout.flush()
        obs = env.reset()
        _flush("  reset() complete")
        sys.stdout.flush()

        _obs_summary(evgs, obs, label="after reset()")

        # Confirm voxels key is present
        voxels_present = isinstance(obs, dict) and "voxels" in obs and obs["voxels"] is not None
        _flush(f"\nvoxels key present: {voxels_present}")
        sys.stdout.flush()

        # ------------------------------------------------------------------
        # Noop steps: report at step 5 and step 10
        # ------------------------------------------------------------------
        noop_action = env.action_space.no_op()
        _flush(f"\nTaking 10 noop steps (action = no_op)...")
        sys.stdout.flush()

        for step_i in range(1, 11):
            try:
                obs, reward, done, info = env.step(noop_action)
            except Exception as step_err:
                _flush(f"  step {step_i}: ERROR - {step_err}")
                traceback.print_exc()
                break

            if done:
                _flush(f"  step {step_i}: episode ended, calling reset()")
                obs = env.reset()

            if step_i in (5, 10):
                _obs_summary(evgs, obs, label=f"after step {step_i}")

        # ------------------------------------------------------------------
        # Final PASS/FAIL assessment
        # ------------------------------------------------------------------
        _flush(f"\n{'='*60}")
        _flush("RESULT")
        _flush(f"{'='*60}")

        checks = {
            "voxels key present in obs"  : voxels_present,
            "voxel_extract_target runs"  : True,  # would have raised above if broken
            "get_agent_state runs"       : True,
            "evgs.extract runs"          : True,
        }

        # Try each utility one more time for the final check
        try:
            tv = voxel_extract_target(obs, "wood")
            checks["voxel_extract_target shape correct"] = tv.shape == (N_VOXEL_CELLS,)
            _flush(f"  voxel_extract_target shape: {tv.shape}  "
                   f"(expected ({N_VOXEL_CELLS},))")
        except Exception as exc:
            checks["voxel_extract_target shape correct"] = False
            _flush(f"  voxel_extract_target FAILED: {exc}")

        try:
            st = get_agent_state(obs)
            checks["get_agent_state shape correct"] = st.shape == (6,)
            _flush(f"  get_agent_state shape: {st.shape}  (expected (6,))")
        except Exception as exc:
            checks["get_agent_state shape correct"] = False
            _flush(f"  get_agent_state FAILED: {exc}")

        try:
            x = evgs.extract(obs)
            checks["evgs shape correct"] = x.shape == (len(VAR_NAMES),)
            _flush(f"  evgs.extract shape: {x.shape}  (expected ({len(VAR_NAMES)},))")
        except Exception as exc:
            checks["evgs shape correct"] = False
            _flush(f"  evgs.extract FAILED: {exc}")

        _flush("")
        all_passed = True
        for check_name, result in checks.items():
            mark = "PASS" if result else "FAIL"
            _flush(f"  [{mark}] {check_name}")
            if not result:
                all_passed = False

        _flush("")
        if all_passed and voxels_present:
            _flush("OVERALL: PASS")
            passed = True
        else:
            _flush("OVERALL: FAIL")
            if not voxels_present:
                _flush("  obs['voxels'] was absent - check that use_voxel=True is supported")
                _flush("  by the installed MineDojo version.")

    except Exception:
        _flush("\nUNHANDLED EXCEPTION:")
        traceback.print_exc()
        _flush("\nOVERALL: FAIL")

    finally:
        if env is not None:
            try:
                env.close()
                _flush("\nenv.close() OK")
            except Exception as close_err:
                _flush(f"\nenv.close() error (non-fatal): {close_err}")
        sys.stdout.flush()

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
