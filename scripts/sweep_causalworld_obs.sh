#!/bin/bash
# Run CausalWorld observational PCG baseline sweep: 3 conditions x 5 seeds = 15 runs
#
# Usage:
#   bash scripts/sweep_causalworld_obs.sh
#   STEPS=5000 OUT_DIR=results/logs bash scripts/sweep_causalworld_obs.sh
#
# Environment variables (all optional):
#   STEPS    : number of option-level training steps per run (default: 5000)
#   OUT_DIR  : directory for JSON result files (default: results/logs)
#
# This baseline corresponds to CDL/CARL-style observational causal discovery.
# Compare results against cw_*.json (DIA interventional) to validate Claim 1.

set -e

STEPS="${STEPS:-5000}"
OUT_DIR="${OUT_DIR:-results/logs}"
mkdir -p "$OUT_DIR"

CONDITIONS="T0 T1 T2"
SEEDS="0 1 2 3 4"

echo "============================================================"
echo "  CausalWorld Observational Baseline Sweep"
echo "  conditions : $CONDITIONS"
echo "  seeds      : $SEEDS"
echo "  steps      : $STEPS"
echo "  out_dir    : $OUT_DIR"
echo "  compare to : $OUT_DIR/cw_{T0,T1,T2}_*.json  (DIA interventional)"
echo "============================================================"

launched=0
skipped=0

for cond in $CONDITIONS; do
  for seed in $SEEDS; do
    out="$OUT_DIR/obs_${cond}_${seed}.json"
    if [ -f "$out" ]; then
      echo "Skipping $out (already exists)"
      skipped=$((skipped + 1))
      continue
    fi
    log="$OUT_DIR/obs_${cond}_${seed}.log"
    conda run -n dia-minecraft python3 scripts/run_causalworld_obs_baseline.py \
      --condition "$cond" \
      --seed "$seed" \
      --steps "$STEPS" \
      --out "$out" \
      >"$log" 2>&1 &
    echo "Launched: condition=$cond seed=$seed  pid=$!  log=$log"
    launched=$((launched + 1))
  done
done

echo ""
echo "All runs launched: $launched new, $skipped skipped."
echo "Monitor with: ls $OUT_DIR/obs_*.json | wc -l"
echo "Waiting for all background jobs to complete..."
wait
echo ""
echo "All done. Run comparison:"
echo "  conda run -n dia-minecraft python3 scripts/compare_obs_vs_dia.py"
