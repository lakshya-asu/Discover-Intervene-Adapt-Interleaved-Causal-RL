#!/usr/bin/env bash
# Run all DIA paper figure scripts and report which files were generated.
# Usage: bash scripts/plot_all.sh [--conda-env <name>]
#        (default conda env: dia-minecraft)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
FIGURES_DIR="$REPO_ROOT/results/figures"
CONDA_ENV="${1:-dia-minecraft}"

# Resolve python interpreter
if command -v conda &>/dev/null; then
    PYTHON="$(conda run -n "$CONDA_ENV" which python 2>/dev/null || true)"
fi
if [[ -z "${PYTHON:-}" ]]; then
    PYTHON="$(command -v python3 || command -v python)"
fi
echo "Using Python: $PYTHON"

mkdir -p "$FIGURES_DIR"

SCRIPTS=(
    "plot_2d_results.py"
    "plot_causalworld_results.py"
    "plot_crafter_results.py"
)

FAILED=()

for script in "${SCRIPTS[@]}"; do
    echo ""
    echo "========================================"
    echo "Running: $script"
    echo "========================================"
    if "$PYTHON" "$SCRIPT_DIR/$script"; then
        echo "  [OK] $script completed."
    else
        echo "  [FAIL] $script exited with error."
        FAILED+=("$script")
    fi
done

echo ""
echo "========================================"
echo "Generated figures:"
echo "========================================"
for ext in pdf png; do
    for f in "$FIGURES_DIR"/*."$ext"; do
        [[ -f "$f" ]] && echo "  $f"
    done
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""
    echo "FAILED scripts:"
    for s in "${FAILED[@]}"; do
        echo "  $s"
    done
    exit 1
else
    echo ""
    echo "All scripts completed successfully."
fi
