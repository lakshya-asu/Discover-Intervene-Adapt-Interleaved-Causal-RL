"""
Plot 2D Minecraft results for DIA paper (NeurIPS 2026).

Output: results/figures/fig1_2d_minecraft.{pdf,png}

Panels:
  A - Steps to Diamond (bar chart, ±1 std, DNF annotation for ppo/ppo_options)
  B - Diamond Success Rate (bar chart, ±1 std)
  C - SHD / PCG accuracy, DIA methods only (lower is better, GT=0 dashed line)
  D - Diamond acquisition curve (cumulative fraction vs. option step)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# DIA family → first 4 muted colours; baselines → grays
METHOD_ORDER = ["dia_oracle", "dia", "dia_no_sig", "dia_no_ig", "ride", "icm", "ppo", "ppo_options"]
DIA_METHODS = ["dia_oracle", "dia", "dia_no_sig", "dia_no_ig"]
CURVE_METHODS = ["dia_oracle", "dia", "dia_no_sig", "dia_no_ig", "ride", "icm"]

METHOD_COLORS: dict[str, tuple] = {
    "dia_oracle":  PALETTE[0],
    "dia":         PALETTE[1],
    "dia_no_sig":  PALETTE[2],
    "dia_no_ig":   PALETTE[3],
    "ride":        (0.5, 0.5, 0.5),
    "icm":         (0.65, 0.65, 0.65),
    "ppo":         (0.78, 0.78, 0.78),
    "ppo_options": (0.88, 0.88, 0.88),
}

METHOD_LABELS: dict[str, str] = {
    "dia_oracle":  "DIA (oracle)",
    "dia":         "DIA",
    "dia_no_sig":  "DIA–sig",
    "dia_no_ig":   "DIA–IG",
    "ride":        "RIDE",
    "icm":         "ICM",
    "ppo":         "PPO",
    "ppo_options": "PPO+Options",
}

DNF_METHODS = {"ppo", "ppo_options"}

AXIS_LABEL_SIZE = 11
TICK_SIZE = 9
TITLE_SIZE = 12
N_SEEDS = 10


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_2d_data(logs_dir: Path) -> dict[str, list[dict]]:
    """Load all 2d_*.json files. Returns {method: [record, ...]}."""
    data: dict[str, list[dict]] = {}
    for method in METHOD_ORDER:
        records = []
        for seed in range(N_SEEDS):
            path = logs_dir / f"2d_{method}_seed{seed}.json"
            if path.exists():
                with open(path) as fh:
                    records.append(json.load(fh))
            else:
                log.warning("Missing: %s", path)
        if records:
            data[method] = records
        else:
            log.warning("No data found for method '%s'", method)
    return data


def extract_metric(
    records: list[dict],
    key: str,
    fallback: Optional[float] = None,
) -> list[float]:
    """Extract a numeric metric from records, skipping None values."""
    values = []
    for r in records:
        v = r.get(key)
        if v is not None:
            values.append(float(v))
        elif fallback is not None:
            values.append(fallback)
    return values


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------
def _bar_stats(
    data: dict[str, list[dict]],
    methods: list[str],
    key: str,
    fallback: Optional[float] = None,
) -> tuple[list[str], list[float], list[float]]:
    """Return (labels, means, stds) for the given metric and method list."""
    labels, means, stds = [], [], []
    for m in methods:
        if m not in data:
            labels.append(METHOD_LABELS[m])
            means.append(np.nan)
            stds.append(0.0)
            continue
        vals = extract_metric(data[m], key, fallback)
        if vals:
            labels.append(METHOD_LABELS[m])
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=0)))
        else:
            labels.append(METHOD_LABELS[m])
            means.append(np.nan)
            stds.append(0.0)
    return labels, means, stds


def _draw_bar_panel(
    ax: plt.Axes,
    methods: list[str],
    means: list[float],
    stds: list[float],
    ylabel: str,
    title: str,
    dnf_methods: Optional[set[str]] = None,
    hline: Optional[float] = None,
    hline_label: str = "",
) -> None:
    """Draw a bar chart panel with error bars."""
    x = np.arange(len(methods))
    colors = [METHOD_COLORS[m] for m in methods]

    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        error_kw={"elinewidth": 1.2, "ecolor": "dimgray"},
    )

    # Horizontal reference line
    if hline is not None:
        ax.axhline(hline, color="black", linestyle="--", linewidth=1.1, label=hline_label)
        if hline_label:
            ax.legend(fontsize=8, loc="upper right")

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=30, ha="right", fontsize=TICK_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Panel D — acquisition curve
# ---------------------------------------------------------------------------
def _build_acquisition_curve(
    records: list[dict],
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (step_grid, cumulative_fraction) from per-seed diamond_steps_list."""
    all_first_steps = []
    for r in records:
        dsl = r.get("diamond_steps_list")
        if dsl and len(dsl) > 0:
            all_first_steps.append(float(dsl[0]))
        else:
            std = r.get("steps_to_diamond")
            if std is not None:
                all_first_steps.append(float(std))
            # if neither is available, seed never reached diamond → not included

    n_seeds = len(records)
    if n_seeds == 0 or not all_first_steps:
        grid = np.linspace(0, n_steps, 500)
        return grid, np.zeros_like(grid)

    all_first_steps.sort()
    grid = np.linspace(0, n_steps, 500)
    frac = np.array([np.mean(np.array(all_first_steps) <= s) for s in grid])
    return grid, frac


def _draw_acquisition_panel(
    ax: plt.Axes,
    data: dict[str, list[dict]],
    methods: list[str],
    n_steps: int = 10000,
) -> None:
    for m in methods:
        if m not in data:
            log.warning("Acquisition curve: no data for '%s'", m)
            continue
        grid, frac = _build_acquisition_curve(data[m], n_steps)
        ax.plot(grid, frac, color=METHOD_COLORS[m], label=METHOD_LABELS[m], linewidth=1.8)

    # PPO flat at 0
    if "ppo" in data:
        ax.plot(
            [0, n_steps],
            [0, 0],
            color=METHOD_COLORS["ppo"],
            linewidth=1.2,
            linestyle="--",
            label=METHOD_LABELS["ppo"],
        )

    ax.set_xlabel("Option steps", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Cumulative fraction\nseeds reaching diamond", fontsize=AXIS_LABEL_SIZE)
    ax.set_title("(D) Diamond Acquisition Curve", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlim(0, n_steps)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=7.5, ncol=2, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    data = load_2d_data(LOGS_DIR)

    # -----------------------------------------------------------------------
    # Panel A: steps_to_diamond (DNF methods excluded from bar, annotated)
    # -----------------------------------------------------------------------
    labels_a, means_a, stds_a = _bar_stats(data, METHOD_ORDER, "steps_to_diamond")
    # For DNF methods (ppo / ppo_options) steps_to_diamond is null → nan → bar absent
    # Annotate them separately after axis limits set

    # -----------------------------------------------------------------------
    # Panel B: success_rate
    # -----------------------------------------------------------------------
    labels_b, means_b, stds_b = _bar_stats(data, METHOD_ORDER, "success_rate", fallback=0.0)

    # -----------------------------------------------------------------------
    # Panel C: final_shd (DIA methods only)
    # -----------------------------------------------------------------------
    labels_c, means_c, stds_c = _bar_stats(data, DIA_METHODS, "final_shd")

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("2D Minecraft: DIA vs. Baselines", fontsize=14, fontweight="bold", y=1.01)
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # --- Panel A ---
    _draw_bar_panel(
        ax_a,
        METHOD_ORDER,
        means_a,
        stds_a,
        ylabel="Steps to First Diamond",
        title="(A) Steps to Diamond",
        dnf_methods=DNF_METHODS,
    )
    # After bar is drawn, set y-limit headroom for DNF text
    valid_a = [v for v in means_a if not np.isnan(v)]
    if valid_a:
        ax_a.set_ylim(0, max(valid_a) * 1.25)

    # Re-draw DNF labels now that ylim is set
    y_top_a = ax_a.get_ylim()[1]
    x_pos = np.arange(len(METHOD_ORDER))
    for i, m in enumerate(METHOD_ORDER):
        if m in DNF_METHODS:
            ax_a.text(
                x_pos[i],
                y_top_a * 0.93,
                "DNF",
                ha="center",
                va="top",
                fontsize=8,
                color="dimgray",
                style="italic",
                fontweight="bold",
            )

    # --- Panel B ---
    _draw_bar_panel(
        ax_b,
        METHOD_ORDER,
        means_b,
        stds_b,
        ylabel="Success Rate",
        title="(B) Diamond Success Rate",
    )
    ax_b.set_ylim(0, 1.1)

    # --- Panel C ---
    _draw_bar_panel(
        ax_c,
        DIA_METHODS,
        means_c,
        stds_c,
        ylabel="Structural Hamming Distance (SHD)",
        title="(C) PCG Accuracy (lower is better)",
        hline=0.0,
        hline_label="GT SHD = 0",
    )

    # --- Panel D ---
    _draw_acquisition_panel(ax_d, data, CURVE_METHODS, n_steps=10000)

    fig.tight_layout(pad=2.5)

    for ext in ("pdf", "png"):
        out_path = OUT_DIR / f"fig1_2d_minecraft.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        log.info("Saved %s", out_path)

    plt.close(fig)
    print("fig1_2d_minecraft generated successfully.")


if __name__ == "__main__":
    main()
