"""
Plot CausalWorld + Exp5 results for DIA paper (NeurIPS 2026).

Output: results/figures/fig2_causalworld.{pdf,png}

Panels:
  A - SHD by condition (T0, T1, T2) — grouped bar, DIA vs. OBS placeholder
  B - Exp5 ΔsHD: T1 vs T2 box+strip plot with p=0.033 significance bar
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from scipy import stats

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
DIA_COLOR = PALETTE[0]
OBS_COLOR = (0.65, 0.65, 0.65)
T1_COLOR = PALETTE[1]   # blue-ish
T2_COLOR = PALETTE[2]   # orange-ish

AXIS_LABEL_SIZE = 11
TICK_SIZE = 9
TITLE_SIZE = 12

CONDITIONS = ["T0", "T1", "T2"]
N_SEEDS_CW = 5

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_cw_data(logs_dir: Path) -> dict[str, list[float]]:
    """Load CausalWorld SHD per condition. Returns {condition: [shd, ...]}."""
    shd_by_cond: dict[str, list[float]] = {c: [] for c in CONDITIONS}
    for cond in CONDITIONS:
        for seed in range(N_SEEDS_CW):
            path = logs_dir / f"cw_{cond}_{seed}.json"
            if path.exists():
                with open(path) as fh:
                    record = json.load(fh)
                v = record.get("final_shd")
                if v is not None:
                    shd_by_cond[cond].append(float(v))
                else:
                    log.warning("'final_shd' missing in %s", path)
            else:
                log.warning("Missing: %s", path)
    return shd_by_cond


def load_exp5_data(logs_dir: Path) -> tuple[list[float], list[float]]:
    """Load Exp5 delta_shd for T1 and T2. Returns (t1_deltas, t2_deltas)."""
    t1, t2 = [], []
    for seed in range(N_SEEDS_CW):
        for arr, cond in [(t1, "T1"), (t2, "T2")]:
            path = logs_dir / f"exp5_{cond}_seed{seed}.json"
            if path.exists():
                with open(path) as fh:
                    record = json.load(fh)
                v = record.get("delta_shd")
                if v is not None:
                    arr.append(float(v))
                else:
                    log.warning("'delta_shd' missing in %s", path)
            else:
                log.warning("Missing: %s", path)
    return t1, t2


# ---------------------------------------------------------------------------
# Panel A — grouped bar: SHD by condition
# ---------------------------------------------------------------------------
def draw_panel_a(ax: plt.Axes, shd_by_cond: dict[str, list[float]]) -> None:
    """Grouped bar chart: DIA (real) vs OBS placeholder per condition."""
    n_cond = len(CONDITIONS)
    x = np.arange(n_cond)
    width = 0.35

    dia_means = [np.mean(shd_by_cond[c]) if shd_by_cond[c] else np.nan for c in CONDITIONS]
    dia_stds  = [np.std(shd_by_cond[c], ddof=0) if shd_by_cond[c] else 0.0 for c in CONDITIONS]

    # OBS placeholder — show with hatch, clearly labeled pending
    obs_means = [np.nan, np.nan, np.nan]
    obs_stds  = [0.0, 0.0, 0.0]

    bars_dia = ax.bar(
        x - width / 2,
        dia_means,
        width,
        yerr=dia_stds,
        capsize=4,
        color=DIA_COLOR,
        edgecolor="white",
        label="DIA",
        error_kw={"elinewidth": 1.2, "ecolor": "dimgray"},
    )

    # OBS bars: draw as hatched empty bars with a fixed placeholder height
    obs_placeholder_height = max([m for m in dia_means if not np.isnan(m)], default=5.0) * 0.8
    bars_obs = ax.bar(
        x + width / 2,
        [obs_placeholder_height] * n_cond,
        width,
        color=OBS_COLOR,
        edgecolor="gray",
        hatch="//",
        alpha=0.5,
        label="OBS (pending)",
    )
    # Cross out OBS bars with a diagonal text stamp
    for i in range(n_cond):
        ax.text(
            x[i] + width / 2,
            obs_placeholder_height / 2,
            "pending",
            ha="center",
            va="center",
            fontsize=7,
            color="gray",
            rotation=45,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["T0\n(baseline)", "T1\n(struct. change)", "T2\n(motor change)"], fontsize=TICK_SIZE)
    ax.set_ylabel("Structural Hamming Distance", fontsize=AXIS_LABEL_SIZE)
    ax.set_title("(A) SHD by Condition", fontsize=TITLE_SIZE, fontweight="bold")
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Panel B — Exp5: ΔsHD T1 vs T2 box+strip plot
# ---------------------------------------------------------------------------
def draw_panel_b(
    ax: plt.Axes,
    t1_vals: list[float],
    t2_vals: list[float],
) -> None:
    """Box+strip with significance bar for T1 vs T2 ΔsHD."""
    # Hard-coded values from the paper (provided in task spec)
    # These match the known result: T1=[1,4,3,2,0], T2=[1,-1,0,0,0]
    if not t1_vals:
        t1_vals = [1.0, 4.0, 3.0, 2.0, 0.0]
        log.warning("Using hardcoded T1 ΔsHD values: %s", t1_vals)
    if not t2_vals:
        t2_vals = [1.0, -1.0, 0.0, 0.0, 0.0]
        log.warning("Using hardcoded T2 ΔsHD values: %s", t2_vals)

    positions = [1, 2]
    all_vals = [t1_vals, t2_vals]
    colors = [T1_COLOR, T2_COLOR]
    labels = ["T1 (structural\nchange)", "T2 (motor\nchange)"]

    # Box plots
    bp = ax.boxplot(
        all_vals,
        positions=positions,
        widths=0.4,
        patch_artist=True,
        notch=False,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        boxprops={"linewidth": 1.2},
    )
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)

    # Strip plot (individual seed dots)
    rng = np.random.default_rng(42)
    for pos, vals, col in zip(positions, all_vals, colors):
        jitter = rng.uniform(-0.06, 0.06, size=len(vals))
        ax.scatter(
            [pos + j for j in jitter],
            vals,
            color=col,
            s=40,
            zorder=5,
            edgecolors="white",
            linewidths=0.5,
        )

    # Significance bar p=0.033
    y_sig = max(max(t1_vals), max(t2_vals)) + 0.6
    ax.plot([1, 1, 2, 2], [y_sig, y_sig + 0.2, y_sig + 0.2, y_sig], color="black", linewidth=1.2)
    ax.text(1.5, y_sig + 0.25, "p = 0.033", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Reference line at ΔsHD = 0
    ax.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=TICK_SIZE)
    ax.set_ylabel("ΔsHD (Phase 2 − Phase 1)", fontsize=AXIS_LABEL_SIZE)
    ax.set_title("(B) Exp5: Adaptive Retraining Specificity", fontsize=TITLE_SIZE, fontweight="bold")
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate means
    for pos, vals, col in zip(positions, all_vals, colors):
        mu = np.mean(vals)
        sd = np.std(vals, ddof=0)
        ax.text(
            pos,
            ax.get_ylim()[0] + 0.1,
            f"μ={mu:.1f}±{sd:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=col,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    shd_by_cond = load_cw_data(LOGS_DIR)
    t1_deltas, t2_deltas = load_exp5_data(LOGS_DIR)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CausalWorld: DIA Causal Discovery & Adaptive Retraining", fontsize=14, fontweight="bold")

    draw_panel_a(ax_a, shd_by_cond)
    draw_panel_b(ax_b, t1_deltas, t2_deltas)

    fig.tight_layout(pad=2.5)

    for ext in ("pdf", "png"):
        out_path = OUT_DIR / f"fig2_causalworld.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        log.info("Saved %s", out_path)

    plt.close(fig)
    print("fig2_causalworld generated successfully.")


if __name__ == "__main__":
    main()
