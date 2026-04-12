#!/bin/bash
# Complete DIA Claim 3: BC pretraining + transfer sweep
# Run this AFTER BC extraction is complete (data/minerl_bc/diamond/obs.npy exists)
#
# Usage:
#   bash scripts/run_bc_then_transfer.sh
#
# Prerequisites:
#   - data/minerl_bc/  populated by scripts/filter_minerl_demos.py
#   - pcg_2d.npy       trained PCG matrix (already exists)

set -e

REPO="/home/flux/DIA/Discover-Intervene-Adapt-Interleaved-Causal-RL"
cd "$REPO"

BC_DIR="data/minerl_bc"
POLICIES_DIR="data/minerl_policies"
PCG_PATH="pcg_2d.npy"
OUT_DIR="results/logs"

echo "============================================================"
echo "  DIA Claim 3: BC Pretraining + Transfer Sweep"
echo "============================================================"

# ── Check BC data exists ──────────────────────────────────────────────────
if [ ! -f "$BC_DIR/diamond/obs.npy" ]; then
  echo "ERROR: BC data not ready. Run filter_minerl_demos.py first."
  echo "  Check: ls $BC_DIR/diamond/obs.npy"
  exit 1
fi
echo "BC data found: $BC_DIR"

# ── Check PCG matrix ──────────────────────────────────────────────────────
if [ ! -f "$PCG_PATH" ]; then
  echo "ERROR: PCG matrix not found: $PCG_PATH"
  exit 1
fi
echo "PCG matrix found: $PCG_PATH"

# ── Step 1: BC Pretraining ────────────────────────────────────────────────
echo ""
echo "Step 1: BC Pretraining (50 epochs, CUDA auto)"
mkdir -p "$POLICIES_DIR"

conda run -n dia-minecraft python3 scripts/pretrain_bc_minerl.py \
  --bc_dir "$BC_DIR" \
  --out_dir "$POLICIES_DIR" \
  --epochs 50 \
  --device auto \
  2>&1 | tee "$OUT_DIR/bc_pretrain.log"

echo ""
echo "BC pretraining done. Policies in: $POLICIES_DIR"
ls "$POLICIES_DIR"/*.pt 2>/dev/null | wc -l | xargs echo "  .pt files:"

# ── Step 2: Transfer runs (seeds 0-4) ────────────────────────────────────
echo ""
echo "Step 2: Transfer runs (5 seeds, parallel)"

mkdir -p "$OUT_DIR"
launched=0
skipped=0

for seed in 0 1 2 3 4; do
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
    --bc_dir "$POLICIES_DIR" \
    --out "$out" \
    >"$log" 2>&1 &
  echo "Launched: mode=transfer seed=$seed  pid=$!  log=$log"
  launched=$((launched + 1))
done

echo ""
echo "Transfer runs launched: $launched new, $skipped skipped."
echo "Waiting for all background jobs..."
wait

echo ""
echo "All transfer runs complete. Results:"
ls "$OUT_DIR"/transfer_dia_seed*.json 2>/dev/null | wc -l | xargs echo "  transfer_dia results:"
ls "$OUT_DIR"/transfer_baseline_seed*.json 2>/dev/null | wc -l | xargs echo "  transfer_baseline results:"

# ── Step 3: Print summary ─────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Summary: Transfer vs Baseline"
echo "============================================================"

conda run -n dia-minecraft python3 -c "
import json
import os

results = {}
for mode in ['baseline', 'dia']:
    results[mode] = []
    for seed in range(5):
        f = os.path.join('$OUT_DIR', f'transfer_{mode}_seed{seed}.json')
        if os.path.exists(f):
            with open(f) as fp:
                d = json.load(fp)
            results[mode].append(d)

print('Mode       | Seed | steps_to_diamond | n_achieved | completed')
print('-----------+------+-----------------+------------+----------')
for mode in ['baseline', 'dia']:
    for d in results[mode]:
        stps = d.get('steps_to_diamond_3d', 'null')
        n = d.get('n_achieved', 0)
        c = d.get('completed', False)
        print(f'{mode:<10} | {d[\"seed\"]}    | {str(stps):<16} | {n:<10} | {c}')
" 2>/dev/null

echo ""
echo "Done. Run scripts/append_transfer_metrics.py to update results/metrics_log.md"
