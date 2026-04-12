"""
PCG evaluation metrics.

All public functions accept:
  probs: np.ndarray, shape (M, M) -- predicted edge probabilities (probs[i,j] = P(i->j))
  true_edges: list of (i, j) int tuples -- ground-truth directed edges
  threshold: float -- probability cutoff to binarize probs (default 0.5)
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np


def binarize(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert probability matrix to binary adjacency matrix."""
    return (probs >= threshold).astype(int)


def shd(probs: np.ndarray,
        true_edges: List[Tuple[int, int]],
        threshold: float = 0.5) -> int:
    """
    Structural Hamming Distance between predicted and ground-truth DAG.
    Counts the number of edge additions and deletions needed to match ground truth.
    """
    M = probs.shape[0]
    pred = binarize(probs, threshold)
    true = np.zeros((M, M), dtype=int)
    for i, j in true_edges:
        true[i, j] = 1
    # Zero out diagonal (no self-loops)
    np.fill_diagonal(pred, 0)
    np.fill_diagonal(true, 0)
    return int(np.sum(pred != true))


def ece(probs: np.ndarray,
        true_edges: List[Tuple[int, int]],
        n_bins: int = 10) -> float:
    """
    Expected Calibration Error for edge probabilities.
    Measures how well the predicted probabilities are calibrated against ground truth.
    """
    M = probs.shape[0]
    # Flatten, excluding diagonal
    mask = ~np.eye(M, dtype=bool)
    flat_probs = probs[mask]
    flat_true = np.zeros(M * M, dtype=float)
    for i, j in true_edges:
        flat_true[i * M + j] = 1.0
    flat_true = flat_true.reshape(M, M)[mask]

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    n = len(flat_probs)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        # Include the right endpoint in the last bin so prob=1.0 is captured
        if b == n_bins - 1:
            in_bin = (flat_probs >= lo) & (flat_probs <= hi)
        else:
            in_bin = (flat_probs >= lo) & (flat_probs < hi)
        if in_bin.sum() == 0:
            continue
        avg_conf = flat_probs[in_bin].mean()
        avg_acc = flat_true[in_bin].mean()
        ece_val += (in_bin.sum() / n) * abs(avg_conf - avg_acc)
    return float(ece_val)


def pcg_summary(probs: np.ndarray,
                true_edges: List[Tuple[int, int]],
                var_names: List[str] | None = None,
                threshold: float = 0.5) -> dict:
    """
    Compute SHD and ECE together and return as dict.
    Optionally print a human-readable summary.
    """
    s = shd(probs, true_edges, threshold)
    e = ece(probs, true_edges)
    result = {"shd": s, "ece": e, "threshold": threshold}
    if var_names:
        M = probs.shape[0]
        pred = binarize(probs, threshold)
        true_mat = np.zeros((M, M), dtype=int)
        for i, j in true_edges:
            true_mat[i, j] = 1
        result["false_positives"] = [
            (var_names[i], var_names[j])
            for i in range(M) for j in range(M)
            if pred[i, j] == 1 and true_mat[i, j] == 0 and i != j
        ]
        result["false_negatives"] = [
            (var_names[i], var_names[j])
            for i in range(M) for j in range(M)
            if pred[i, j] == 0 and true_mat[i, j] == 1
        ]
    return result
