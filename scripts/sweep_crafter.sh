#!/bin/bash
# Run all Crafter baseline comparisons: 7 methods x 5 seeds = 35 runs
# Usage:
#   bash scripts/sweep_crafter.sh
#   STEPS=200000 OUT_DIR=results/logs bash scripts/sweep_crafter.sh
#
# Environment variables (all optional):
#   STEPS    : number of option-level training steps per run (default: 200000)
#   OUT_DIR  : directory for JSON result files (default: results/logs)
#
# Runs all jobs in the background in parallel.
# Monitor progress:
#   ls results/logs/crafter_*.json | wc -l    # completed runs (out of 35)
#   grep '"completed": true' results/logs/crafter_*.json | wc -l
#
# Wait for all to finish:
#   jobs -l    # shows running background jobs
#   wait       # blocks until all background jobs complete

set -e

STEPS="${STEPS:-200000}"
OUT_DIR="${OUT_DIR:-results/logs}"
mkdir -p "$OUT_DIR"

METHODS="ppo ride icm dia_no_ig dia_no_sig dia dia_oracle"
SEEDS="0 1 2 3 4"

echo "============================================================"
echo "  Crafter Sweep"
echo "  methods : $METHODS"
echo "  seeds   : $SEEDS"
echo "  steps   : $STEPS"
echo "  out_dir : $OUT_DIR"
echo "============================================================"

launched=0
skipped=0

for method in $METHODS; do
  for seed in $SEEDS; do
    out="$OUT_DIR/crafter_${method}_seed${seed}.json"
    if [ -f "$out" ]; then
      echo "Skipping $out (already exists)"
      skipped=$((skipped + 1))
      continue
    fi
    log="$OUT_DIR/crafter_${method}_seed${seed}.log"
    conda run -n dia-minecraft python3 scripts/run_baseline_crafter.py \
      --method "$method" \
      --seed "$seed" \
      --steps "$STEPS" \
      --out "$out" \
      >"$log" 2>&1 &
    echo "Launched: method=$method seed=$seed  pid=$!  log=$log"
    launched=$((launched + 1))
  done
done

echo ""
echo "All runs launched: $launched new, $skipped skipped."
echo "Monitor with: ls $OUT_DIR/crafter_*.json | wc -l"
echo "Waiting for all background jobs to complete..."
wait
echo "All done."
