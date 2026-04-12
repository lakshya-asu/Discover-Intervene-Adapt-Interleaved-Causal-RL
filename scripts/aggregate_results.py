#!/usr/bin/env python3
"""
scripts/aggregate_results.py — Aggregate 2D Minecraft sweep results into a Markdown table.

Usage:
  python3 scripts/aggregate_results.py
  python3 scripts/aggregate_results.py --pattern "results/logs/2d_*.json"
  python3 scripts/aggregate_results.py --pattern "results/logs/2d_*.json" --csv results/table2.csv

Output (stdout): a Markdown table with mean ± std across seeds.

JSON result schema expected (written by run_baseline_2d.py):
  {
    "method": str,
    "seed": int,
    "steps_to_diamond": int | null,
    "final_shd": float | null,
    "final_ece": float | null,
    "success_rate": float | null,
    "completed": bool
  }

Incomplete runs (completed=false) are skipped.
Methods are ordered by canonical order if present, then alphabetically.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np


# Canonical display order and labels for the 7 sweep methods
METHOD_ORDER = [
    "ppo",
    "ppo_options",
    "ride",
    "icm",
    "dia_no_ig",
    "dia_no_sig",
    "dia",
]

METHOD_LABELS: Dict[str, str] = {
    "ppo":         "PPO flat",
    "ppo_options": "PPO + Options",
    "ride":        "RIDE",
    "icm":         "ICM",
    "dia_no_ig":   "DIA -- no IG",
    "dia_no_sig":  "DIA -- no SIG",
    "dia":         "DIA (ours)",
}


def load_results(pattern: str) -> List[Dict]:
    """Load all JSON files matching the glob pattern."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"[aggregate_results] No files found matching: {pattern}", file=sys.stderr)
        return []

    records = []
    skipped = 0
    for path in paths:
        try:
            with open(path) as fh:
                rec = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[aggregate_results] Skipping {path}: {exc}", file=sys.stderr)
            skipped += 1
            continue

        if not rec.get("completed", False):
            skipped += 1
            continue

        records.append(rec)

    print(
        f"[aggregate_results] Loaded {len(records)} completed runs "
        f"({skipped} skipped) from {len(paths)} files.",
        file=sys.stderr,
    )
    return records


def aggregate(records: List[Dict]) -> Dict[str, Dict]:
    """Group records by method and compute per-metric statistics."""
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for rec in records:
        grouped[rec["method"]].append(rec)

    stats: Dict[str, Dict] = {}
    for method, runs in grouped.items():
        n = len(runs)

        # steps_to_diamond: only count runs where diamond was reached
        std_vals = [r["steps_to_diamond"] for r in runs if r.get("steps_to_diamond") is not None]
        sr_mean: Optional[float] = float(np.mean(std_vals)) if std_vals else None
        sr_std: Optional[float]  = float(np.std(std_vals))  if len(std_vals) > 1 else (0.0 if std_vals else None)

        # success_rate: fraction of option executions where subgoal succeeded
        sr_list = [r["success_rate"] for r in runs if r.get("success_rate") is not None]
        success_mean = float(np.mean(sr_list)) if sr_list else None

        # SHD and ECE (only for PCG-based methods; null otherwise)
        shd_list = [r["final_shd"] for r in runs if r.get("final_shd") is not None]
        ece_list = [r["final_ece"] for r in runs if r.get("final_ece") is not None]

        shd_mean = float(np.mean(shd_list)) if shd_list else None
        shd_std  = float(np.std(shd_list))  if len(shd_list) > 1 else (0.0 if shd_list else None)
        ece_mean = float(np.mean(ece_list)) if ece_list else None

        stats[method] = {
            "n_seeds":          n,
            "steps_mean":       sr_mean,
            "steps_std":        sr_std,
            "n_diamond_seeds":  len(std_vals),
            "shd_mean":         shd_mean,
            "shd_std":          shd_std,
            "ece_mean":         ece_mean,
            "success_mean":     success_mean,
        }

    return stats


def _fmt(val: Optional[float], decimals: int = 0, pct: bool = False) -> str:
    if val is None:
        return "--"
    if pct:
        return f"{val * 100:.1f}%"
    if decimals == 0:
        return f"{int(round(val))}"
    return f"{val:.{decimals}f}"


def render_markdown(stats: Dict[str, Dict]) -> str:
    """Render a Markdown table from aggregated stats."""
    # Column widths (minimum)
    col_method    = 14
    col_steps_m   = 24
    col_steps_s   = 24
    col_shd       = 7
    col_ece       = 8
    col_succ      = 10

    header = (
        f"| {'Method':<{col_method}} "
        f"| {'Steps-to-Diamond (mean)':<{col_steps_m}} "
        f"| {'Steps-to-Diamond (std)':<{col_steps_s}} "
        f"| {'SHD':<{col_shd}} "
        f"| {'ECE':<{col_ece}} "
        f"| {'Success%':<{col_succ}} |"
    )
    sep = (
        f"|{'-'*(col_method+2)}"
        f"|{'-'*(col_steps_m+2)}"
        f"|{'-'*(col_steps_s+2)}"
        f"|{'-'*(col_shd+2)}"
        f"|{'-'*(col_ece+2)}"
        f"|{'-'*(col_succ+2)}|"
    )

    lines = [header, sep]

    # Sort: canonical order first, then alphabetical for any extra methods
    all_methods = list(stats.keys())
    ordered = [m for m in METHOD_ORDER if m in stats]
    extras  = sorted(m for m in all_methods if m not in METHOD_ORDER)
    for method in ordered + extras:
        s = stats[method]
        label = METHOD_LABELS.get(method, method)
        n_info = f"(n={s['n_seeds']})"
        row = (
            f"| {label:<{col_method}} "
            f"| {_fmt(s['steps_mean']):<{col_steps_m}} "
            f"| {_fmt(s['steps_std']):<{col_steps_s}} "
            f"| {_fmt(s['shd_mean'], decimals=1):<{col_shd}} "
            f"| {_fmt(s['ece_mean'], decimals=4):<{col_ece}} "
            f"| {_fmt(s['success_mean'], pct=True):<{col_succ}} |"
        )
        lines.append(row)

    return "\n".join(lines)


def write_csv(stats: Dict[str, Dict], csv_path: str) -> None:
    """Write aggregated stats to a CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    fieldnames = [
        "method", "n_seeds", "steps_to_diamond_mean", "steps_to_diamond_std",
        "n_diamond_seeds", "shd_mean", "shd_std", "ece_mean", "success_mean",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        all_methods = list(stats.keys())
        ordered = [m for m in METHOD_ORDER if m in stats]
        extras  = sorted(m for m in all_methods if m not in METHOD_ORDER)
        for method in ordered + extras:
            s = stats[method]
            writer.writerow({
                "method":                  method,
                "n_seeds":                 s["n_seeds"],
                "steps_to_diamond_mean":   s["steps_mean"],
                "steps_to_diamond_std":    s["steps_std"],
                "n_diamond_seeds":         s["n_diamond_seeds"],
                "shd_mean":                s["shd_mean"],
                "shd_std":                 s["shd_std"],
                "ece_mean":                s["ece_mean"],
                "success_mean":            s["success_mean"],
            })
    print(f"[aggregate_results] CSV written to {csv_path}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Aggregate 2D Minecraft sweep results into a Markdown table.",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="results/logs/2d_*.json",
        help='Glob pattern for result JSON files (default: "results/logs/2d_*.json").',
    )
    ap.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to also write a CSV summary.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    records = load_results(args.pattern)
    if not records:
        print("No completed runs found.  Nothing to aggregate.")
        sys.exit(0)

    stats = aggregate(records)
    table = render_markdown(stats)

    print("\n## Table 2 (partial): Sample Efficiency -- 2D Minecraft\n")
    print(table)
    print()

    # Per-method seed count summary
    print("### Run completion summary")
    all_methods = list(stats.keys())
    ordered = [m for m in METHOD_ORDER if m in stats]
    extras  = sorted(m for m in all_methods if m not in METHOD_ORDER)
    for method in ordered + extras:
        s = stats[method]
        label = METHOD_LABELS.get(method, method)
        diamond_rate = s["n_diamond_seeds"] / max(1, s["n_seeds"]) * 100
        print(
            f"  {label:<20} : {s['n_seeds']:2d} seeds  "
            f"diamond reached in {s['n_diamond_seeds']:2d}/{s['n_seeds']} seeds "
            f"({diamond_rate:.0f}%)"
        )
    print()

    if args.csv:
        write_csv(stats, args.csv)


if __name__ == "__main__":
    main()
