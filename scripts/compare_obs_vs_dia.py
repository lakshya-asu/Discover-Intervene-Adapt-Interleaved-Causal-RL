#!/usr/bin/env python3
# scripts/compare_obs_vs_dia.py
"""
Aggregate and compare DIA (interventional) vs Observational baseline results
on CausalWorld.

Reads:
  results/logs/cw_{T0,T1,T2}_seed{0..4}.json    -- DIA interventional
  results/logs/obs_{T0,T1,T2}_seed{0..4}.json   -- Observational baseline

Prints comparison table and returns exit code 0 if obs SHD > dia SHD
(expected direction for Claim 1), else exits with code 1.

Usage:
  conda run -n dia-minecraft python3 scripts/compare_obs_vs_dia.py
  conda run -n dia-minecraft python3 scripts/compare_obs_vs_dia.py \
      --logs_dir results/logs --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


def load_results(logs_dir: Path, prefix: str, conditions: List[str],
                 seeds: List[int]) -> Dict[str, List[float]]:
    """Load SHD and ECE for each condition, returning {condition: [shd_seed0, ...]}.

    Missing files are reported as NaN so the script doesn't crash mid-sweep.
    """
    shd_by_cond: Dict[str, List[float]] = {c: [] for c in conditions}
    ece_by_cond: Dict[str, List[float]] = {c: [] for c in conditions}
    for cond in conditions:
        for seed in seeds:
            path = logs_dir / f"{prefix}_{cond}_{seed}.json"
            if not path.exists():
                shd_by_cond[cond].append(float("nan"))
                ece_by_cond[cond].append(float("nan"))
                continue
            with open(path) as f:
                data = json.load(f)
            shd_by_cond[cond].append(float(data.get("final_shd", float("nan"))))
            ece_by_cond[cond].append(float(data.get("final_ece", float("nan"))))
    return shd_by_cond, ece_by_cond


def mean_std(values: List[float]) -> str:
    arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
    n_missing = sum(1 for v in values if np.isnan(v))
    if len(arr) == 0:
        return "  N/A  "
    m = np.mean(arr)
    s = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
    suffix = f" ({n_missing} missing)" if n_missing else ""
    return f"{m:.2f} ± {s:.2f}{suffix}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", type=str, default="results/logs")
    ap.add_argument("--seeds",    type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    logs_dir   = Path(args.logs_dir)
    seeds      = args.seeds
    conditions = ["T0", "T1", "T2"]

    dia_shd, dia_ece = load_results(logs_dir, "cw",  conditions, seeds)
    obs_shd, obs_ece = load_results(logs_dir, "obs", conditions, seeds)

    # ── Summary table ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CausalWorld: DIA (interventional) vs Observational Baseline — SHD")
    print("=" * 70)
    header = f"{'Method':<28} | {'T0 SHD':>14} | {'T1 SHD':>14} | {'T2 SHD':>14}"
    print(header)
    print("-" * 70)
    print(f"{'DIA (interventional)':<28} | {mean_std(dia_shd['T0']):>14} | "
          f"{mean_std(dia_shd['T1']):>14} | {mean_std(dia_shd['T2']):>14}")
    print(f"{'Observational baseline':<28} | {mean_std(obs_shd['T0']):>14} | "
          f"{mean_std(obs_shd['T1']):>14} | {mean_std(obs_shd['T2']):>14}")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("  CausalWorld: DIA (interventional) vs Observational Baseline — ECE")
    print("=" * 70)
    header2 = f"{'Method':<28} | {'T0 ECE':>14} | {'T1 ECE':>14} | {'T2 ECE':>14}"
    print(header2)
    print("-" * 70)
    print(f"{'DIA (interventional)':<28} | {mean_std(dia_ece['T0']):>14} | "
          f"{mean_std(dia_ece['T1']):>14} | {mean_std(dia_ece['T2']):>14}")
    print(f"{'Observational baseline':<28} | {mean_std(obs_ece['T0']):>14} | "
          f"{mean_std(obs_ece['T1']):>14} | {mean_std(obs_ece['T2']):>14}")
    print("=" * 70)

    # ── Per-seed detail ──────────────────────────────────────────────────────────
    print("\n--- Per-seed SHD detail ---")
    for cond in conditions:
        print(f"\n  Condition {cond}:")
        print(f"  {'seed':<6} {'DIA SHD':>9} {'Obs SHD':>9} {'Δ (Obs−DIA)':>12}")
        print(f"  {'-'*6} {'-'*9} {'-'*9} {'-'*12}")
        for i, seed in enumerate(seeds):
            d_v = dia_shd[cond][i]
            o_v = obs_shd[cond][i]
            delta = o_v - d_v if not (np.isnan(d_v) or np.isnan(o_v)) else float("nan")
            d_str = f"{d_v:.1f}" if not np.isnan(d_v) else "N/A"
            o_str = f"{o_v:.1f}" if not np.isnan(o_v) else "N/A"
            d_str_delta = f"{delta:+.1f}" if not np.isnan(delta) else "N/A"
            print(f"  {seed:<6} {d_str:>9} {o_str:>9} {d_str_delta:>12}")

    # ── Claim 1 verdict ──────────────────────────────────────────────────────────
    all_obs_vals = [v for c in conditions for v in obs_shd[c] if not np.isnan(v)]
    all_dia_vals = [v for c in conditions for v in dia_shd[c] if not np.isnan(v)]

    if len(all_obs_vals) == 0 or len(all_dia_vals) == 0:
        print("\n[CLAIM 1] Insufficient data — not all runs complete.")
        sys.exit(2)

    obs_mean = float(np.mean(all_obs_vals))
    dia_mean = float(np.mean(all_dia_vals))
    delta    = obs_mean - dia_mean

    print("\n" + "=" * 70)
    print(f"  Overall mean SHD — DIA: {dia_mean:.2f}  |  Obs: {obs_mean:.2f}  "
          f"|  Δ = {delta:+.2f}")
    if obs_mean > dia_mean:
        print("  [CLAIM 1] SUPPORTED: obs SHD > DIA SHD  "
              "(observational is worse, as expected)")
        verdict = 0
    else:
        print("  [CLAIM 1] NOT SUPPORTED: obs SHD <= DIA SHD  -- investigate!")
        verdict = 1
    print("=" * 70 + "\n")
    sys.exit(verdict)


if __name__ == "__main__":
    main()
