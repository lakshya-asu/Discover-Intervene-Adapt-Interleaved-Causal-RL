#!/bin/bash
# scripts/sweep_exp5.sh
# Sweep Experiment 5: T0 -> T1/T2 adaptive retraining (Claim 4)
# Runs 2 conditions × 5 seeds = 10 runs
#
# Usage:
#   bash scripts/sweep_exp5.sh
#   bash scripts/sweep_exp5.sh --phase1 5000 --phase2 2000   (override steps)
#
# Output: results/logs/exp5_{T1,T2}_seed{0..4}.json

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
CONDITIONS="T1 T2"
SEEDS="0 1 2 3 4"
PHASE1=5000
PHASE2=2000
OUT_DIR="results/logs"
LOGDIR="runs/exp5"
CONDA_ENV="dia-minecraft"
SCRIPT="scripts/experiment5_causalworld_adapt.py"

# ── Parse optional overrides ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase1) PHASE1="$2"; shift 2 ;;
        --phase2) PHASE2="$2"; shift 2 ;;
        --out_dir) OUT_DIR="$2"; shift 2 ;;
        --logdir) LOGDIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "  Exp5 Sweep: T0 -> {T1, T2} adaptive retraining"
echo "  Conditions : ${CONDITIONS}"
echo "  Seeds      : ${SEEDS}"
echo "  Phase1     : ${PHASE1} steps"
echo "  Phase2     : ${PHASE2} steps"
echo "  Output dir : ${OUT_DIR}"
echo "============================================================"

TOTAL=0
DONE=0
FAILED=0

for COND in ${CONDITIONS}; do
    for SEED in ${SEEDS}; do
        TOTAL=$((TOTAL + 1))
        OUT="${OUT_DIR}/exp5_${COND}_seed${SEED}.json"

        if [[ -f "${OUT}" ]]; then
            echo "[sweep] Skipping ${COND} seed=${SEED}: ${OUT} already exists"
            DONE=$((DONE + 1))
            continue
        fi

        echo ""
        echo "------------------------------------------------------------"
        echo "[sweep] Running condition=${COND} seed=${SEED}"
        echo "        -> ${OUT}"
        echo "------------------------------------------------------------"

        if conda run -n "${CONDA_ENV}" python "${SCRIPT}" \
                --condition "${COND}" \
                --seed "${SEED}" \
                --phase1_steps "${PHASE1}" \
                --phase2_steps "${PHASE2}" \
                --logdir "${LOGDIR}" \
                --out "${OUT}"; then
            DONE=$((DONE + 1))
            echo "[sweep] DONE: ${COND} seed=${SEED} -> ${OUT}"
        else
            FAILED=$((FAILED + 1))
            echo "[sweep] FAILED: ${COND} seed=${SEED}" >&2
        fi
    done
done

echo ""
echo "============================================================"
echo "  Sweep complete: ${DONE}/${TOTAL} succeeded, ${FAILED} failed"
echo "============================================================"

# ── Quick summary of results ─────────────────────────────────────────────────
echo ""
echo "Results summary:"
for COND in ${CONDITIONS}; do
    echo "  Condition ${COND}:"
    for SEED in ${SEEDS}; do
        OUT="${OUT_DIR}/exp5_${COND}_seed${SEED}.json"
        if [[ -f "${OUT}" ]]; then
            python3 -c "
import json, sys
with open('${OUT}') as f:
    d = json.load(f)
print(f'    seed={d[\"seed\"]}  phase1_shd={d[\"phase1_shd\"]}  '
      f'phase2_shd={d[\"phase2_shd\"]}  delta_shd={d[\"delta_shd\"]:+.1f}  '
      f'steps_to_detect={d[\"steps_to_detect\"]}  '
      f'structural_detected={d[\"structural_detected\"]}')
" 2>/dev/null || echo "    seed=${SEED}: parse error"
        else
            echo "    seed=${SEED}: MISSING"
        fi
    done
done
