#!/usr/bin/env python3
"""
Analyse ROCKET-1 experiment results and print comparison table.

Usage:
  python scripts/analyse_rocket1.py --dir results/rocket1
  python scripts/analyse_rocket1.py --dir results/rocket1 --latex
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

VAR_NAMES = ["wood", "stone", "coal", "ironore", "furnace",
             "stonepickaxe", "iron", "ironpickaxe", "diamond"]


def load_results(directory: str) -> Dict[str, List[Dict]]:
    """Load all JSON result files, grouped by mode."""
    by_mode: Dict[str, List[Dict]] = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        try:
            with open(path) as f:
                r = json.load(f)
            mode = r.get("mode", "unknown")
            by_mode[mode].append(r)
        except Exception as e:
            print(f"  skip {path}: {e}")
    return dict(by_mode)


def summarise_mode(results: List[Dict]) -> Dict[str, Any]:
    if not results:
        return {}

    n = len(results)
    n_achieved   = [r.get("n_achieved", 0)       for r in results]
    diamond      = [int(r.get("diamond_reached", False)) for r in results]
    total_steps  = [r.get("total_steps", 0)       for r in results]

    # Per-skill success rates
    skill_success: Dict[str, List[int]] = defaultdict(list)
    skill_steps:   Dict[str, List[int]] = defaultdict(list)
    for r in results:
        for skill in VAR_NAMES:
            sr = r.get("results", {}).get(skill, {})
            skill_success[skill].append(int(sr.get("success", False)))
            skill_steps[skill].append(sr.get("steps", 0))

    return {
        "n_runs":        n,
        "n_achieved":    {"mean": float(np.mean(n_achieved)),  "std": float(np.std(n_achieved))},
        "diamond_rate":  float(np.mean(diamond)),
        "total_steps":   {"mean": float(np.mean(total_steps)), "std": float(np.std(total_steps))},
        "skill_success": {s: float(np.mean(v)) for s, v in skill_success.items()},
        "skill_steps":   {s: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                          for s, v in skill_steps.items()},
    }


def print_comparison(by_mode: Dict[str, List[Dict]], latex: bool = False) -> None:
    summaries = {mode: summarise_mode(results) for mode, results in by_mode.items()}

    print("\n" + "=" * 70)
    print("ROCKET-1 Results Summary")
    print("=" * 70)

    # Main metrics table
    modes = sorted(summaries.keys())
    col_w = 18
    header = f"{'Metric':<25}" + "".join(f"{m:>{col_w}}" for m in modes)
    print(header)
    print("-" * (25 + col_w * len(modes)))

    def row(label, vals):
        return f"{label:<25}" + "".join(f"{v:>{col_w}}" for v in vals)

    def fmt(mean, std=None):
        if std is None:
            return f"{mean:.3f}"
        return f"{mean:.2f}±{std:.2f}"

    for metric, key in [("n_achieved (mean±std)", "n_achieved"),
                         ("diamond rate",          "diamond_rate"),
                         ("total_steps (mean)",    "total_steps")]:
        vals = []
        for m in modes:
            s = summaries.get(m, {})
            if key == "diamond_rate":
                vals.append(fmt(s.get("diamond_rate", float("nan"))))
            elif key in s:
                v = s[key]
                if isinstance(v, dict):
                    vals.append(fmt(v["mean"], v.get("std")))
                else:
                    vals.append(fmt(v))
            else:
                vals.append("N/A")
        print(row(metric, vals))

    # Per-skill success rates
    print("\nPer-skill success rates (fraction of seeds):")
    header2 = f"{'Skill':<18}" + "".join(f"{m:>{col_w}}" for m in modes)
    print(header2)
    print("-" * (18 + col_w * len(modes)))
    for skill in VAR_NAMES:
        vals = []
        for m in modes:
            rate = summaries.get(m, {}).get("skill_success", {}).get(skill, float("nan"))
            vals.append(f"{rate:.2f}")
        print(f"  {skill:<16}" + "".join(f"{v:>{col_w}}" for v in vals))

    print("=" * 70)

    if latex:
        print("\n% LaTeX table fragment:")
        print("\\begin{tabular}{l" + "r" * len(modes) + "}")
        print("\\toprule")
        print("Metric & " + " & ".join(modes) + " \\\\")
        print("\\midrule")
        for skill in VAR_NAMES:
            row_vals = []
            for m in modes:
                rate = summaries.get(m, {}).get("skill_success", {}).get(skill, float("nan"))
                row_vals.append(f"{rate:.2f}")
            print(f"{skill} & " + " & ".join(row_vals) + " \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyse ROCKET-1 experiment results.")
    ap.add_argument("--dir",   default="results/rocket1", help="Results directory")
    ap.add_argument("--latex", action="store_true",       help="Print LaTeX table")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        print(f"Directory not found: {args.dir}")
        return

    by_mode = load_results(args.dir)
    if not by_mode:
        print(f"No JSON result files found in {args.dir}")
        return

    for mode, results in by_mode.items():
        print(f"  {mode}: {len(results)} runs (seeds: {[r.get('seed') for r in results]})")

    print_comparison(by_mode, latex=args.latex)


if __name__ == "__main__":
    main()
