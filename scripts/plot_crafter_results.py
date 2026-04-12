"""
Plot Crafter results for DIA paper (NeurIPS 2026).

Output: results/figures/fig3_crafter.{pdf,png}

Behaviour:
  - If ≥5 crafter_*.json files are found, generate a full bar chart
    (steps to diamond + SHD) mirroring the 2D Minecraft layout.
  - Otherwise, write a placeholder PNG/PDF with "Crafter sweep running —
    N/35 complete" and the list of completed seeds.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = REPO_ROOT / "results" / "logs"
OUT_DIR = REPO_ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid")
PALETTE = sns.color_palette("muted")

# Expected 7 methods × 5 seeds = 35 jobs
CRAFTER_METHODS = ["dia_oracle", "dia", "dia_no_sig", "dia_no_ig", "ride", "icm", "ppo"]
DIA_METHODS = ["dia_oracle", "dia", "dia_no_sig", "dia_no_ig"]
TOTAL_JOBS = 35
N_SEEDS = 5

METHOD_COLORS: dict[str, tuple] = {
    "dia_oracle":  PALETTE[0],
    "dia":         PALETTE[1],
    "dia_no_sig":  PALETTE[2],
    "dia_no_ig":   PALETTE[3],
    "ride":        (0.5, 0.5, 0.5),
    "icm":         (0.65, 0.65, 0.65),
    "ppo":         (0.78, 0.78, 0.78),
}

METHOD_LABELS: dict[str, str] = {
    "dia_oracle":  "DIA (oracle)",
    "dia":         "DIA",
    "dia_no_sig":  "DIA–sig",
    "dia_no_ig":   "DIA–IG",
    "ride":        "RIDE",
    "icm":         "ICM",
    "ppo":         "PPO",
}

AXIS_LABEL_SIZE = 11
TICK_SIZE = 9
TITLE_SIZE = 12


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def find_crafter_files(logs_dir: Path) -> list[Path]:
    """Return all crafter_*.json files present in logs_dir."""
    return sorted(logs_dir.glob("crafter_*.json"))


def load_crafter_data(logs_dir: Path) -> dict[str, list[dict]]:
    """Load crafter JSON files. Returns {method: [record, ...]}."""
    data: dict[str, list[dict]] = {}
    for method in CRAFTER_METHODS:
        records = []
        for seed in range(N_SEEDS):
            path = logs_dir / f"crafter_{method}_seed{seed}.json"
            if path.exists():
                with open(path) as fh:
                    records.append(json.load(fh))
        if records:
            data[method] = records
    return data


# ---------------------------------------------------------------------------
# Placeholder figure
# ---------------------------------------------------------------------------
def write_placeholder(n_complete: int) -> None:
    """Write a placeholder figure showing sweep progress."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_axis_off()

    ax.text(
        0.5,
        0.65,
        f"Crafter sweep running\n{n_complete}/{TOTAL_JOBS} complete",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="dimgray",
    )
    ax.text(
        0.5,
        0.35,
        "This panel will be populated once ≥5 result files are available.\n"
        "Re-run this script after the sweep completes.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color="gray",
        style="italic",
    )
    # Simple progress bar
    bar_width = 0.6
    bar_x = (1.0 - bar_width) / 2
    progress = n_complete / TOTAL_JOBS
    ax.add_patch(
        plt.Rectangle(
            (bar_x, 0.15),
            bar_width,
            0.07,
            transform=ax.transAxes,
            facecolor="lightgray",
            edgecolor="gray",
            linewidth=1,
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (bar_x, 0.15),
            bar_width * progress,
            0.07,
            transform=ax.transAxes,
            facecolor=PALETTE[1],
            edgecolor="none",
        )
    )

    fig.suptitle("Crafter: DIA vs. Baselines (pending)", fontsize=14, fontweight="bold", color="dimgray")

    for ext in ("pdf", "png"):
        out_path = OUT_DIR / f"fig3_crafter.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        log.info("Saved placeholder %s", out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Full figure (when data is available)
# ---------------------------------------------------------------------------
def _bar_stats(
    data: dict[str, list[dict]],
    methods: list[str],
    key: str,
    fallback: Optional[float] = None,
) -> tuple[list[float], list[float]]:
    means, stds = [], []
    for m in methods:
        if m not in data:
            means.append(np.nan)
            stds.append(0.0)
            continue
        vals = []
        for r in data[m]:
            v = r.get(key)
            if v is not None:
                vals.append(float(v))
            elif fallback is not None:
                vals.append(fallback)
        if vals:
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=0)))
        else:
            means.append(np.nan)
            stds.append(0.0)
    return means, stds


def write_full_figure(data: dict[str, list[dict]]) -> None:
    """Generate full 1×2 bar chart figure."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Crafter: DIA vs. Baselines", fontsize=14, fontweight="bold")

    colors = [METHOD_COLORS[m] for m in CRAFTER_METHODS]
    x = np.arange(len(CRAFTER_METHODS))
    width = 0.65

    # --- Steps to diamond ---
    means_s, stds_s = _bar_stats(data, CRAFTER_METHODS, "steps_to_diamond")
    ax_l.bar(
        x, means_s, width, yerr=stds_s, capsize=4,
        color=colors, edgecolor="white",
        error_kw={"elinewidth": 1.2, "ecolor": "dimgray"},
    )
    ax_l.set_xticks(x)
    ax_l.set_xticklabels([METHOD_LABELS[m] for m in CRAFTER_METHODS], rotation=30, ha="right", fontsize=TICK_SIZE)
    ax_l.set_ylabel("Steps to First Diamond", fontsize=AXIS_LABEL_SIZE)
    ax_l.set_title("(A) Steps to Diamond", fontsize=TITLE_SIZE, fontweight="bold")
    ax_l.tick_params(axis="y", labelsize=TICK_SIZE)
    ax_l.spines["top"].set_visible(False)
    ax_l.spines["right"].set_visible(False)

    # --- SHD (DIA methods only) ---
    x_c = np.arange(len(DIA_METHODS))
    colors_c = [METHOD_COLORS[m] for m in DIA_METHODS]
    means_c, stds_c = _bar_stats(data, DIA_METHODS, "final_shd")
    ax_r.bar(
        x_c, means_c, width, yerr=stds_c, capsize=4,
        color=colors_c, edgecolor="white",
        error_kw={"elinewidth": 1.2, "ecolor": "dimgray"},
    )
    ax_r.axhline(0, color="black", linestyle="--", linewidth=1.1, label="GT SHD = 0")
    ax_r.set_xticks(x_c)
    ax_r.set_xticklabels([METHOD_LABELS[m] for m in DIA_METHODS], rotation=30, ha="right", fontsize=TICK_SIZE)
    ax_r.set_ylabel("Structural Hamming Distance (SHD)", fontsize=AXIS_LABEL_SIZE)
    ax_r.set_title("(B) PCG Accuracy (lower is better)", fontsize=TITLE_SIZE, fontweight="bold")
    ax_r.tick_params(axis="y", labelsize=TICK_SIZE)
    ax_r.legend(fontsize=8, loc="upper right")
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)

    fig.tight_layout(pad=2.5)
    for ext in ("pdf", "png"):
        out_path = OUT_DIR / f"fig3_crafter.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        log.info("Saved %s", out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    crafter_files = find_crafter_files(LOGS_DIR)
    n_complete = len(crafter_files)
    log.info("Found %d/%d Crafter result files.", n_complete, TOTAL_JOBS)

    if n_complete >= 5:
        data = load_crafter_data(LOGS_DIR)
        write_full_figure(data)
        print(f"fig3_crafter generated from {n_complete}/{TOTAL_JOBS} files.")
    else:
        write_placeholder(n_complete)
        print(f"fig3_crafter placeholder written ({n_complete}/{TOTAL_JOBS} complete).")


if __name__ == "__main__":
    main()
