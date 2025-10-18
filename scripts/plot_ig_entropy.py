#!/usr/bin/env python3
# scripts/plot_ig_entropy.py
from __future__ import annotations

import argparse
import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

def load_scalar_series(run_dir, tag):
    ea = EventAccumulator(run_dir)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None
    events = ea.Scalars(tag)
    xs = [e.step for e in events]
    ys = [e.value for e in events]
    return xs, ys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logroot", type=str, default="runs")
    ap.add_argument("--prefix", type=str, default="")
    args = ap.parse_args()

    run_dirs = sorted([d for d in glob.glob(os.path.join(args.logroot, "**"), recursive=True) if os.path.isdir(d)])
    if not run_dirs:
        print("No runs found.")
        return

    for tag in ["mc2d/pcg_entropy", "mc2d/ig_update", "coinrun/pcg_entropy", "coinrun/ig_update",
                "monte/pcg_entropy", "monte/ig_update", "cp/pcg_entropy", "cp/ig_update", "cw/pcg_entropy", "cw/ig_update"]:
        plt.figure()
        any_plotted = False
        for rd in run_dirs:
            res = load_scalar_series(rd, tag)
            if res is None: continue
            xs, ys = res
            plt.plot(xs, ys, label=os.path.basename(rd))
            any_plotted = True
        if any_plotted:
            plt.title(tag)
            plt.xlabel("macro step")
            plt.ylabel(tag.split("/")[-1])
            plt.legend(fontsize=6)
            out = os.path.join(args.logroot, f"{args.prefix}{tag.replace('/', '_')}.png")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            print("Saved", out)

if __name__ == "__main__":
    main()
