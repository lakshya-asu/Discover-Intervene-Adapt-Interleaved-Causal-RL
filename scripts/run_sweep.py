#!/usr/bin/env python3
# scripts/run_sweep.py
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime

DEFAULT_PRESETS = {
    # A longer, sequential run across multiple envs
    "long": [
        {"env": "minecraft2d", "pcg": "notears", "steps": 2000, "goal": "diamond"},
        {"env": "coinrun",     "pcg": "variational", "steps": 2000, "goal": "coin_collected"},
        {"env": "cartpole",    "pcg": "notears", "steps": 1000, "goal": "x"},
        # Montezuma last (optional; requires ROMs)
        {"env": "montezuma",   "pcg": "notears", "steps": 1000, "goal": "has_key"},
    ],
    # A shorter sanity sweep
    "short": [
        {"env": "minecraft2d", "pcg": "notears", "steps": 300, "goal": "diamond"},
        {"env": "coinrun",     "pcg": "variational", "steps": 300, "goal": "coin_collected"},
    ],
}

def run_one(run, logroot, fit_every=10, pcg_epochs=200, buffer_recent=1024, min_buffer=256):
    env = run["env"]; pcg = run["pcg"]; steps = int(run["steps"]); goal = run.get("goal")
    env_id = run.get("env_id")
    logdir = os.path.join(logroot, f"{env}_{pcg}_{steps}")
    cmd = [
        sys.executable, "scripts/dia_cli.py", "run",
        "--env", env, "--pcg", pcg, "--steps", str(steps),
        "--fit_every", str(fit_every), "--pcg_epochs", str(pcg_epochs),
        "--buffer_recent", str(buffer_recent), "--min_buffer", str(min_buffer),
        "--logdir", logdir
    ]
    if goal: cmd += ["--goal", goal]
    if env_id: cmd += ["--env_id", env_id]
    print(">>>", " ".join(cmd))
    return subprocess.call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", type=str, default="long", choices=list(DEFAULT_PRESETS.keys()))
    ap.add_argument("--logroot", type=str, default=f"runs/sweeps/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--fit_every", type=int, default=10)
    ap.add_argument("--pcg_epochs", type=int, default=200)
    ap.add_argument("--buffer_recent", type=int, default=1024)
    ap.add_argument("--min_buffer", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.logroot, exist_ok=True)
    suite = DEFAULT_PRESETS[args.preset]
    results = []
    for run in suite:
        rc = run_one(run, logroot=args.logroot, fit_every=args.fit_every,
                     pcg_epochs=args.pcg_epochs, buffer_recent=args.buffer_recent, min_buffer=args.min_buffer)
        results.append((run["env"], rc))
    print("\nSweep complete:")
    for env, rc in results:
        print(f"  {env}: exit {rc}")

if __name__ == "__main__":
    main()
