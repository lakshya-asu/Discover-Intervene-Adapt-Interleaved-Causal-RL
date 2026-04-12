#!/usr/bin/env python3
"""
scripts/run_transfer_pool_minedojo.py
  — Pool runner for the MineDojo 3D transfer experiment.

Runs run_transfer_minedojo.py for all seed×mode combinations in parallel,
limited to --workers N at a time (default: 2, because each job needs an
independent Minecraft server process).

Jobs
----
  transfer : seeds 0-4  (5 jobs)
  baseline : seeds 0-4  (5 jobs)
  Total    : 10 jobs

Output layout
-------------
  results/logs/transfer3d_{mode}_seed{seed}.json   — per-run JSON result
  results/logs/transfer3d_{mode}_seed{seed}.log    — per-run stdout/stderr
  results/logs/transfer3d_pool.log                 — pool-level summary

Usage
-----
  # Default: 2 workers, seeds 0-4 for both modes
  conda run -n dia-minecraft python scripts/run_transfer_pool_minedojo.py

  # 4 workers (if enough RAM for 4 Minecraft servers):
  conda run -n dia-minecraft python scripts/run_transfer_pool_minedojo.py --workers 4

  # Custom seeds / extra args forwarded to the child script:
  conda run -n dia-minecraft python scripts/run_transfer_pool_minedojo.py \\
      --workers 2 --seeds 0 1 2 --modes transfer \\
      --max_steps_per_skill 2000 --max_total_steps 30000 \\
      --pcg_path pcg_2d.npy --bc_dir data/minerl_policies
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import List, Tuple

OUT_DIR  = "results/logs"
POOL_LOG = os.path.join(OUT_DIR, "transfer3d_pool.log")

# ── Script path (relative to this file, resolved at runtime) ────────────────
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_CHILD_SCRIPT = os.path.join(_THIS_DIR, "run_transfer_minedojo.py")


# ---------------------------------------------------------------------------
# Job building
# ---------------------------------------------------------------------------

def build_jobs(
    modes: List[str],
    seeds: List[int],
    pcg_path: str,
    bc_dir: str,
    max_steps_per_skill: int,
    max_total_steps: int,
    dry_run: bool,
    verbose: bool,
    skip_existing: bool,
) -> List[Tuple[str, int, str, str]]:
    """
    Build the list of (mode, seed, out_json, log_path) tuples.

    If skip_existing is True, jobs whose output JSON already exists are
    printed as [skip] and excluded from the run list.
    """
    jobs = []
    for mode in modes:
        for seed in seeds:
            out  = os.path.join(OUT_DIR, f"transfer3d_{mode}_seed{seed}.json")
            log  = os.path.join(OUT_DIR, f"transfer3d_{mode}_seed{seed}.log")
            if skip_existing and os.path.exists(out):
                print(f"[skip] {out}")
                continue
            jobs.append((mode, seed, out, log))
    return jobs


def build_cmd(
    mode: str,
    seed: int,
    out: str,
    pcg_path: str,
    bc_dir: str,
    max_steps_per_skill: int,
    max_total_steps: int,
    dry_run: bool,
    verbose: bool,
) -> List[str]:
    """Construct the subprocess command for a single job."""
    cmd = [
        "conda", "run", "--no-capture-output", "-n", "dia-minecraft",
        sys.executable, _CHILD_SCRIPT,
        "--mode",                str(mode),
        "--seed",                str(seed),
        "--pcg_path",            pcg_path,
        "--bc_dir",              bc_dir,
        "--max_steps_per_skill", str(max_steps_per_skill),
        "--max_total_steps",     str(max_total_steps),
        "--out",                 out,
    ]
    if dry_run:
        cmd.append("--dry_run")
    if verbose:
        cmd.append("--verbose")
    return cmd


# ---------------------------------------------------------------------------
# Pool runner
# ---------------------------------------------------------------------------

def run_pool(
    jobs: List[Tuple[str, int, str, str]],
    n_workers: int,
    pcg_path: str,
    bc_dir: str,
    max_steps_per_skill: int,
    max_total_steps: int,
    dry_run: bool,
    verbose: bool,
    pool_log_path: str,
) -> None:
    """Launch jobs with bounded parallelism; poll until all complete."""
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[pool] {len(jobs)} jobs, {n_workers} workers")
    print(f"[pool] log → {pool_log_path}")

    with open(pool_log_path, "w") as plog:
        plog.write(f"MineDojo transfer pool  workers={n_workers}\n")
        plog.write(f"jobs={len(jobs)}\n\n")

    # running: list of (Popen, mode, seed)
    running: List[Tuple[subprocess.Popen, str, int]] = []
    idx = 0

    while idx < len(jobs) or running:
        # ── Reap finished processes ────────────────────────────────────────
        still = []
        for p, mode, seed in running:
            if p.poll() is not None:
                rc = p.returncode
                status = "OK" if rc == 0 else f"FAIL(rc={rc})"
                msg = f"[done {status}] {mode} seed={seed}"
                print(msg)
                with open(pool_log_path, "a") as plog:
                    plog.write(msg + "\n")
            else:
                still.append((p, mode, seed))
        running = still

        # ── Launch up to n_workers ─────────────────────────────────────────
        while len(running) < n_workers and idx < len(jobs):
            mode, seed, out, log = jobs[idx]
            cmd = build_cmd(
                mode=mode,
                seed=seed,
                out=out,
                pcg_path=pcg_path,
                bc_dir=bc_dir,
                max_steps_per_skill=max_steps_per_skill,
                max_total_steps=max_total_steps,
                dry_run=dry_run,
                verbose=verbose,
            )
            with open(log, "w") as lf:
                p = subprocess.Popen(cmd, stdout=lf, stderr=lf)
            msg = f"[launch] {mode} seed={seed} pid={p.pid}  log={log}"
            print(msg)
            with open(pool_log_path, "a") as plog:
                plog.write(msg + "\n")
            running.append((p, mode, seed))
            idx += 1

        time.sleep(2)

    msg = "[pool] all jobs complete"
    print(msg)
    with open(pool_log_path, "a") as plog:
        plog.write(msg + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Pool runner for run_transfer_minedojo.py.\n"
            "Runs transfer (seeds 0-4) + baseline (seeds 0-4) = 10 jobs,\n"
            "limited to --workers N at a time.\n\n"
            "Default: 2 workers (each needs a Minecraft server process)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--workers", type=int, default=2,
        help="Maximum number of parallel jobs  (default: 2)",
    )
    ap.add_argument(
        "--modes", nargs="+", default=["transfer", "baseline"],
        choices=["transfer", "baseline"],
        help="Modes to run  (default: transfer baseline)",
    )
    ap.add_argument(
        "--seeds", nargs="+", type=int, default=list(range(5)),
        help="Seeds to run  (default: 0 1 2 3 4)",
    )
    ap.add_argument(
        "--pcg_path", type=str, default="pcg_2d.npy",
        help="Path to 2D PCG .npy file  (default: pcg_2d.npy)",
    )
    ap.add_argument(
        "--bc_dir", type=str, default="data/minerl_policies",
        help="Directory with <skill>.pt BC policy files  "
             "(default: data/minerl_policies)",
    )
    ap.add_argument(
        "--max_steps_per_skill", type=int, default=3000,
        help="Max env steps per skill  (default: 3000)",
    )
    ap.add_argument(
        "--max_total_steps", type=int, default=40_000,
        help="Hard budget on total env steps  (default: 40000)",
    )
    ap.add_argument(
        "--no_skip", action="store_true",
        help="Re-run jobs even if the output JSON already exists",
    )
    ap.add_argument(
        "--dry_run", action="store_true",
        help="Pass --dry_run to each child (no Minecraft server required)",
    )
    ap.add_argument(
        "--verbose", action="store_true",
        help="Pass --verbose to each child",
    )
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    jobs = build_jobs(
        modes=args.modes,
        seeds=args.seeds,
        pcg_path=args.pcg_path,
        bc_dir=args.bc_dir,
        max_steps_per_skill=args.max_steps_per_skill,
        max_total_steps=args.max_total_steps,
        dry_run=args.dry_run,
        verbose=args.verbose,
        skip_existing=not args.no_skip,
    )

    if not jobs:
        print("[pool] nothing to do (all outputs already exist)")
        print("       Use --no_skip to force re-runs.")
        return

    run_pool(
        jobs=jobs,
        n_workers=args.workers,
        pcg_path=args.pcg_path,
        bc_dir=args.bc_dir,
        max_steps_per_skill=args.max_steps_per_skill,
        max_total_steps=args.max_total_steps,
        dry_run=args.dry_run,
        verbose=args.verbose,
        pool_log_path=POOL_LOG,
    )


if __name__ == "__main__":
    main()
