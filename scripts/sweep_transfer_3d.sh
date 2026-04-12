#!/bin/bash
# Run DIA 3D Transfer sweep: 2 conditions × 5 seeds = 10 runs
# Usage:
#   bash scripts/sweep_transfer_3d.sh
#   PCG_PATH=pcg_2d.npy BC_DIR=data/minerl_policies OUT_DIR=results/logs bash scripts/sweep_transfer_3d.sh
#
# Environment variables (all optional):
#   PCG_PATH : path to 2D PCG edge-prob matrix (default: pcg_2d.npy)
#   BC_DIR   : directory of BC policy .pt files   (default: data/minerl_policies)
#   OUT_DIR  : directory for JSON result files    (default: results/logs)

set -e

PCG_PATH="${PCG_PATH:-pcg_2d.npy}"
BC_DIR="${BC_DIR:-data/minerl_policies}"
OUT_DIR="${OUT_DIR:-results/logs}"
mkdir -p "$OUT_DIR"

SEEDS="0 1 2 3 4"

echo "============================================================"
echo "  DIA 3D Transfer Sweep"
echo "  conditions : baseline transfer"
echo "  seeds      : $SEEDS"
echo "  pcg_path   : $PCG_PATH"
echo "  bc_dir     : $BC_DIR"
echo "  out_dir    : $OUT_DIR"
echo "============================================================"

launched=0
skipped=0

# ── Baseline runs (no transfer knowledge) ──────────────────────────────────
for seed in $SEEDS; do
  out="$OUT_DIR/transfer_baseline_seed${seed}.json"
  if [ -f "$out" ]; then
    echo "Skipping $out (already exists)"
    skipped=$((skipped + 1))
    continue
  fi
  log="$OUT_DIR/transfer_baseline_seed${seed}.log"
  conda run -n dia-minecraft python3 scripts/run_transfer_3d.py \
    --mode baseline \
    --seed "$seed" \
    --out "$out" \
    >"$log" 2>&1 &
  echo "Launched: mode=baseline seed=$seed  pid=$!  log=$log"
  launched=$((launched + 1))
done

# ── Transfer runs (2D PCG + BC warm-start) ─────────────────────────────────
for seed in $SEEDS; do
  out="$OUT_DIR/transfer_dia_seed${seed}.json"
  if [ -f "$out" ]; then
    echo "Skipping $out (already exists)"
    skipped=$((skipped + 1))
    continue
  fi
  log="$OUT_DIR/transfer_dia_seed${seed}.log"
  conda run -n dia-minecraft python3 scripts/run_transfer_3d.py \
    --mode transfer \
    --seed "$seed" \
    --pcg_path "$PCG_PATH" \
    --bc_dir "$BC_DIR" \
    --out "$out" \
    >"$log" 2>&1 &
  echo "Launched: mode=transfer seed=$seed  pid=$!  log=$log"
  launched=$((launched + 1))
done

echo ""
echo "All runs launched: $launched new, $skipped skipped."
echo "Monitor with: ls $OUT_DIR/transfer_*.json | wc -l"
echo "Waiting for all background jobs to complete..."
wait
echo "All done."
