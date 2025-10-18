from __future__ import annotations

from typing import Tuple
import numpy as np


def bernoulli_kl(p_new: np.ndarray, p_old: np.ndarray, eps: float = 1e-8) -> float:
    """KL(Bern(p_new) || Bern(p_old)) summed over entries.
    Args:
        p_new, p_old: arrays in [0,1] of same shape
    """
    p_new = np.clip(p_new, eps, 1 - eps)
    p_old = np.clip(p_old, eps, 1 - eps)
    kl = p_new * np.log(p_new / p_old) + (1 - p_new) * np.log((1 - p_new) / (1 - p_old))
    return float(np.sum(kl))


def entropy_bernoulli(p: np.ndarray, eps: float = 1e-8) -> float:
    p = np.clip(p, eps, 1 - eps)
    h = -p * np.log(p) - (1 - p) * np.log(1 - p)
    return float(np.sum(h))


class BetaScheduler:
    """Simple schedule for beta (IG coefficient): high -> low as entropy drops."""
    def __init__(self, beta_max: float = 1.0, beta_min: float = 0.0, h_ref: float = 10.0):
        self.beta_max = beta_max
        self.beta_min = beta_min
        self.h_ref = max(h_ref, 1e-6)

    def __call__(self, current_entropy: float) -> float:
        # Linear ramp: beta = beta_min + (beta_max - beta_min) * min(1, H/H_ref)
        w = min(1.0, current_entropy / self.h_ref)
        return self.beta_min + (self.beta_max - self.beta_min) * w
