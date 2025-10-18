# src/dia/sem_utils.py
from __future__ import annotations

import numpy as np
from typing import Tuple

def random_dag(num_vars: int, edge_prob: float = 0.2, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    A = rng.rand(num_vars, num_vars) < edge_prob
    # make DAG by upper-triangularizing after a random permutation
    perm = rng.permutation(num_vars)
    A = A[perm][:, perm]
    A = np.triu(A, k=1).astype(float)
    # unpermute back to original order if desired; here we keep permuted order
    return A

def random_weights_from_dag(A: np.ndarray, w_min: float = 0.5, w_max: float = 2.0, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed + 1)
    W = np.zeros_like(A, dtype=float)
    nonzero = np.where(A > 0)
    signs = rng.choice([-1.0, 1.0], size=len(nonzero[0]))
    mags = rng.uniform(w_min, w_max, size=len(nonzero[0]))
    W[nonzero] = signs * mags
    return W

def sample_linear_sem(W: np.ndarray, N: int = 2000, noise_scale: float = 1.0, seed: int = 0) -> np.ndarray:
    """
    Linear SEM: X = X W + Z  => (I - W) X^T = Z^T  => X = Z (I - W)^{-1}
    Assumes W corresponds to a DAG (I - W) is invertible and acyclic.
    """
    rng = np.random.RandomState(seed + 2)
    d = W.shape[0]
    Z = rng.normal(0.0, noise_scale, size=(N, d))
    M = np.eye(d) - W
    X = Z @ np.linalg.inv(M)
    return X

def intervention_mask(N: int, d: int, prob: float = 0.1, seed: int = 0) -> np.ndarray:
    """
    Create a mask[n, j]=1 indicating variable j is intervened in sample n.
    Used to ignore residuals for intervened variables during fitting.
    """
    rng = np.random.RandomState(seed + 3)
    mask = (rng.rand(N, d) < prob).astype(float)
    np.fill_diagonal(mask[:d], 0.0)  # ensure at least some purely observational at start
    return mask
