#!/usr/bin/env python3
"""Rerun any crafter jobs whose JSON output is missing (e.g., killed by OOM).

Usage:
  python3 scripts/run_crafter_mopup.py --workers 4
"""
import subprocess, os, sys, time, argparse

METHODS = ["ppo", "ride", "icm", "dia_no_ig", "dia_no_sig", "dia", "dia_oracle"]
SEEDS = list(range(5))
STEPS = 200_000
OUT_DIR = "results/logs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=STEPS)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    jobs = []
    for method in METHODS:
        for seed in SEEDS:
            out = f"{OUT_DIR}/crafter_{method}_seed{seed}.json"
            if os.path.exists(out):
                print(f"[skip] {out}")
                continue
            jobs.append((method, seed, out))

    if not jobs:
        print("[mopup] all 35 jobs already complete — nothing to do")
        return

    print(f"[mopup] {len(jobs)} missing jobs, {args.workers} workers")

    running = []
    idx = 0
    while idx < len(jobs) or running:
        still = []
        for p, m, s in running:
            if p.poll() is not None:
                rc = p.returncode
                if rc != 0:
                    print(f"[fail rc={rc}] {m} seed={s}")
                else:
                    print(f"[done rc=0] {m} seed={s}")
            else:
                still.append((p, m, s))
        running = still

        while len(running) < args.workers and idx < len(jobs):
            method, seed, out = jobs[idx]
            log = out.replace(".json", ".log")
            cmd = [
                "conda", "run", "-n", "dia-minecraft",
                "python3", "scripts/run_baseline_crafter.py",
                "--method", method, "--seed", str(seed),
                "--steps", str(args.steps), "--out", out,
            ]
            with open(log, "w") as lf:
                p = subprocess.Popen(cmd, stdout=lf, stderr=lf)
            print(f"[launch] {method} seed={seed} pid={p.pid}")
            running.append((p, method, seed))
            idx += 1

        time.sleep(2)

    print("[mopup] all done")


if __name__ == "__main__":
    main()
