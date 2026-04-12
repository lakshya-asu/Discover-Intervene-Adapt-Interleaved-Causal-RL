#!/usr/bin/env bash
# scripts/setup_vpt_data.sh
#
# Download VPT foundation model weights and a small contractor data sample
# for BC warm-start and baseline experiments.
#
# Usage:
#   bash scripts/setup_vpt_data.sh
#
# Idempotent: skips files that already exist.
# Run from the repo root or any directory; paths are absolute.
#
# Data source: https://github.com/openai/Video-Pre-Training
# Model host:  https://openaipublic.blob.core.windows.net/minecraft-rl/

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VPT_DIR="/home/flux/DIA/baselines/vpt"
MODELS_DIR="${VPT_DIR}/models"
DATA_DIR="${VPT_DIR}/data/contractor"

# Base URL for all VPT assets
BLOB_BASE="https://openaipublic.blob.core.windows.net/minecraft-rl"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { echo "[setup_vpt_data] $*"; }
skip() { echo "[setup_vpt_data] SKIP (already exists): $1"; }

download_if_missing() {
    local url="$1"
    local dest="$2"
    local label="${3:-$dest}"
    if [[ -f "$dest" ]]; then
        skip "$label"
    else
        log "Downloading $label ..."
        mkdir -p "$(dirname "$dest")"
        wget --quiet --show-progress -O "$dest" "$url"
        log "Done: $dest"
    fi
}

# ---------------------------------------------------------------------------
# Step 1: Clone VPT repo if not present
# ---------------------------------------------------------------------------
if [[ -d "${VPT_DIR}/.git" ]]; then
    log "VPT repo already cloned at ${VPT_DIR}"
else
    log "Cloning openai/Video-Pre-Training ..."
    mkdir -p "$(dirname "$VPT_DIR")"
    git clone https://github.com/openai/Video-Pre-Training "${VPT_DIR}"
fi
mkdir -p "${MODELS_DIR}"

# ---------------------------------------------------------------------------
# Step 2: Foundation model weights
#
# Size estimates (approximate, from public data):
#   1x model: ~900 MB weights + ~400 KB .model file
#   2x model: ~1.7 GB weights + ~400 KB .model file
#   3x model: ~2.4 GB weights + ~400 KB .model file
#
# We default to 3x (best quality). If disk is constrained, uncomment 1x block.
# ---------------------------------------------------------------------------
log "--- Foundation model weights ---"

# -- 3x model (preferred, ~2.4 GB weights) --
download_if_missing \
    "${BLOB_BASE}/models/foundation-model-3x.model" \
    "${MODELS_DIR}/foundation-model-3x.model" \
    "foundation-model-3x.model (~400 KB)"

download_if_missing \
    "${BLOB_BASE}/models/foundation-model-3x.weights" \
    "${MODELS_DIR}/foundation-model-3x.weights" \
    "foundation-model-3x.weights (~2.4 GB)"

# -- 1x model (fallback, ~900 MB, uncomment if disk is tight) --
# download_if_missing \
#     "${BLOB_BASE}/models/foundation-model-1x.model" \
#     "${MODELS_DIR}/foundation-model-1x.model" \
#     "foundation-model-1x.model (~400 KB)"
# download_if_missing \
#     "${BLOB_BASE}/models/foundation-model-1x.weights" \
#     "${MODELS_DIR}/foundation-model-1x.weights" \
#     "foundation-model-1x.weights (~900 MB)"

# ---------------------------------------------------------------------------
# Step 3: Contractor data - small sample
#
# The VPT dataset uses index JSON files that list (basedir, relpaths[]).
# Each relpath has four associated files:
#   <basedir>/<relpath>.mp4          video frames
#   <basedir>/<relpath>.jsonl        per-frame action dicts
#   <basedir>/<relpath>-options.json metadata / item options
#   <basedir>/<relpath>.zip          full checkpoint (optional, large)
#
# Index file structure:
#   { "basedir": "https://...", "relpaths": ["<version>/<alias>-<uuid>-<date>-<time>", ...] }
#
# There is NO ObtainDiamond-specific subset in the VPT release; the contractor
# data is general survival gameplay. To obtain diamond, contractors played
# open-ended survival, so version 10.x data (newest, best quality) is the
# recommended source.
#
# Size estimates:
#   Version 10.x full snapshot: ~2,000 episodes, estimated 200-400 GB total
#   Each episode: ~100-200 MB video + ~10 MB jsonl (720p, ~15-30 min per session)
#   5 episodes: ~750 MB - 1.5 GB
#
# We download the index file, then fetch 5 episodes (mp4 + jsonl only, skip .zip).
# ---------------------------------------------------------------------------
log "--- Contractor data sample (5 episodes from v10.x) ---"
mkdir -p "${DATA_DIR}"

INDEX_URL="${BLOB_BASE}/snapshots/all_10xx_Jun_29.json"
INDEX_FILE="${DATA_DIR}/all_10xx_Jun_29.json"
download_if_missing "$INDEX_URL" "$INDEX_FILE" "v10.x index JSON"

# Parse the first 5 relpaths and download mp4 + jsonl for each.
log "Extracting first 5 episode relpaths from index ..."

# Export INDEX_FILE so the Python heredoc can read it from os.environ.
export INDEX_FILE="${INDEX_FILE}"

# Use Python (always available in the conda env) to parse the index.
python3 - <<'PYEOF'
import json, os, subprocess, sys

index_path = os.environ.get("INDEX_FILE", "/home/flux/DIA/baselines/vpt/data/contractor/all_10xx_Jun_29.json")
data_dir   = os.path.dirname(index_path)
n_episodes = 200

with open(index_path) as f:
    idx = json.load(f)

basedir  = idx["basedir"].rstrip("/")
relpaths = idx["relpaths"][:n_episodes]

for relpath in relpaths:
    # relpath ends in .mp4 (e.g. data/10.0/episode.mp4)
    # video URL: basedir/relpath          -> dest: data_dir/relpath
    # jsonl URL: basedir/relpath[:-4]+.jsonl -> dest: data_dir/relpath[:-4]+.jsonl
    file_pairs = [
        (f"{basedir}/{relpath}",                    relpath),
        (f"{basedir}/{relpath[:-4]}.jsonl",         relpath[:-4] + ".jsonl"),
    ]
    for url, rel_dest in file_pairs:
        dest = os.path.join(data_dir, rel_dest)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"[setup_vpt_data] SKIP (already exists): {dest}")
        else:
            print(f"[setup_vpt_data] Downloading {rel_dest} ...")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            ret = subprocess.run(
                ["wget", "--quiet", "--show-progress", "-O", dest, url],
                check=False
            )
            if ret.returncode != 0:
                print(f"[setup_vpt_data] WARNING: failed to download {url}", file=sys.stderr)
PYEOF

# ---------------------------------------------------------------------------
# Step 4: Summary
# ---------------------------------------------------------------------------
log ""
log "=== Setup complete ==="
log "VPT repo:       ${VPT_DIR}"
log "Models dir:     ${MODELS_DIR}"
log "Contractor data:${DATA_DIR}"
log ""
log "To run inference with the 3x model:"
log "  conda run -n dia-minecraft python ${VPT_DIR}/run_agent.py \\"
log "    --model ${MODELS_DIR}/foundation-model-3x.model \\"
log "    --weights ${MODELS_DIR}/foundation-model-3x.weights"
log ""
log "Note: run_agent.py uses an infinite loop; use Ctrl+C to stop."
