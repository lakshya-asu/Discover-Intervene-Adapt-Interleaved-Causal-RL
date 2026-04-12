#!/bin/bash
# Run all 2D Minecraft baseline comparisons: 8 methods x 10 seeds = 80 runs
# Usage:
#   bash scripts/sweep_2d.sh
#   STEPS=200000 OUT_DIR=results/logs bash scripts/sweep_2d.sh
#
# Environment variables (all optional):
#   STEPS    : number of option-level training steps per run (default: 500000)
#   OUT_DIR  : directory for JSON result files (default: results/logs)
#
# Runs all jobs in the background in parallel.
# Monitor progress:
#   ls results/logs/2d_*.json | wc -l   # completed runs (out of 80)
#   grep '"completed": true' results/logs/2d_*.json | wc -l  # successful runs
#
# Wait for all to finish:
#   jobs -l     # shows running background jobs
#   wait        # blocks until all background jobs complete

set -e

STEPS="${STEPS:-500000}"
OUT_DIR="${OUT_DIR:-results/logs}"
mkdir -p "$OUT_DIR"

METHODS="ppo ppo_options ride icm dia_no_ig dia_no_sig dia dia_oracle"
SEEDS="0 1 2 3 4 5 6 7 8 9"

echo "============================================================"
echo "  2D Minecraft Sweep"
echo "  methods : $METHODS"
echo "  seeds   : $SEEDS"
echo "  steps   : $STEPS"
echo "  out_dir : $OUT_DIR"
echo "============================================================"

launched=0
skipped=0

for method in $METHODS; do
  for seed in $SEEDS; do
    out="$OUT_DIR/2d_${method}_seed${seed}.json"
    if [ -f "$out" ]; then
      echo "Skipping $out (already exists)"
      skipped=$((skipped + 1))
      continue
    fi
    log="$OUT_DIR/2d_${method}_seed${seed}.log"
    conda run -n dia-minecraft python3 scripts/run_baseline_2d.py \
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
echo "Monitor with: ls $OUT_DIR/2d_*.json | wc -l"
echo "Waiting for all background jobs to complete..."
wait
echo "All done."
