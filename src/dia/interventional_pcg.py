# src/dia/interventional_pcg.py
"""
InterventionalPCG: causal graph learned from targeted skill interventions.

Key distinction from the old transition-frequency estimator:

  OLD (observational):  P(j↑ | state[i]=1 at time t)
                        Computed over mixed transitions from ALL skills.
                        Correlates state variables — not causal.

  NEW (interventional): P(j↑ | skill_k executes)
                        Updated ONLY from the one skill that targeted variable k.
                        Clean signal: "did my specific action cause this change?"

probs[k, j] = P(variable j increases when skill targeting k is executed).
This is what gets fed into expand_sig_from_pcg for prerequisite discovery.

Usage:
    ipcg = InterventionalPCG(num_vars=7)
    ...
    ipcg.update(cause_var=2, x_before=x0, x_after=x1)
    p = ipcg.probs         # [7, 7] — used for heatmap and SIG expansion
    H = ipcg.entropy()     # total uncertainty
    e = ipcg.row_entropy(k)  # uncertainty about effects of skill k
"""
from __future__ import annotations

import numpy as np


class InterventionalPCG:
    """
    Bayesian causal edge probability matrix updated from interventional data.

    For each skill k (targeting variable k) and each other variable j:
      hits_up[k, j]  = # times skill k fired AND variable j went from 0→1
      hits_dn[k, j]  = # times skill k fired AND variable j went from 1→0
      exec_count[k]  = # times skill k was executed

    Edge probability:
      probs[k, j] = P(j↑ | skill_k executes)
                  = (hits_up[k, j] + alpha) / (exec_count[k] + 2*alpha)

    Stays at 0.5 (maximum uncertainty) until exec_count[k] >= min_obs.
    """

    def __init__(self, num_vars: int, alpha: float = 1.0, min_obs: int = 5):
        self.d        = int(num_vars)
        self.alpha    = float(alpha)
        self.min_obs  = int(min_obs)

        self.hits_up   = np.zeros((self.d, self.d), dtype=float)
        self.hits_dn   = np.zeros((self.d, self.d), dtype=float)
        self.exec_count = np.zeros(self.d, dtype=float)

        # probs[k, j] = P(j↑ | skill_k executed); diagonal = 0
        self.probs = np.full((self.d, self.d), 0.5, dtype=float)
        np.fill_diagonal(self.probs, 0.0)

    # ------------------------------------------------------------------ update

    def update(self, cause_var: int, x_before: np.ndarray, x_after: np.ndarray):
        """
        Record the effect of executing the skill that targets cause_var.

        Call this ONCE per macro-step immediately after the option finishes,
        using the EVGS state before and after the entire option run.
        """
        x_before = np.asarray(x_before, dtype=float).reshape(-1)
        x_after  = np.asarray(x_after,  dtype=float).reshape(-1)
        k = int(cause_var)

        self.exec_count[k] += 1
        n = float(self.exec_count[k])

        for j in range(self.d):
            if j == k:
                continue
            diff = float(x_after[j]) - float(x_before[j])
            if diff > 0.5:
                self.hits_up[k, j] += 1
            elif diff < -0.5:
                self.hits_dn[k, j] += 1

            # Only publish a meaningful estimate once we have enough data
            if n >= self.min_obs:
                a = self.alpha
                self.probs[k, j] = (self.hits_up[k, j] + a) / (n + 2.0 * a)
            # else stays at 0.5 (max-entropy prior)

        self.probs[k, k] = 0.0

    # ------------------------------------------------------------------ entropy

    def entropy(self) -> float:
        """Total Shannon entropy of all off-diagonal edge probabilities."""
        eps = 1e-8
        p = np.clip(self.probs, eps, 1.0 - eps)
        h = -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)
        np.fill_diagonal(h, 0.0)
        return float(np.sum(h))

    def row_entropy(self, var_k: int) -> float:
        """
        Entropy of row var_k: how uncertain are the causal effects of skill k?
        High value → we still need to probe skill k more.
        """
        eps = 1e-8
        row = self.probs[int(var_k)].copy()
        row[int(var_k)] = 0.5  # treat self as neutral (excluded)
        p = np.clip(row, eps, 1.0 - eps)
        h = -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)
        return float(np.sum(h))

    # ------------------------------------------------------------------ compat

    def apply_update(self, new_probs: np.ndarray):
        """Compatibility shim for code that calls pcg.apply_update(probs)."""
        self.probs = np.asarray(new_probs, dtype=float).copy()
        np.fill_diagonal(self.probs, 0.0)
