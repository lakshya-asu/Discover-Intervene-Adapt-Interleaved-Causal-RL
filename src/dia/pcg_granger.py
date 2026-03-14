# src/dia/pcg_granger.py
"""
GrangerPCG: Intervention-aware causal structure learner for DIA.

For each directed edge (i → j) this estimator combines two evidence streams:

  1. Interventional lift
     When skill-i is executed (mask[:, i] = 1), variable j co-changes at some
     rate P_int(j).  Compare to P_base(j) = rate of j changing when we were NOT
     targeting i.  Lift = P_int - P_base.  A positive lift is causal evidence.

  2. Observational Granger score
     Among steps where neither i nor j was the intervention target (clean
     observational data), does X_t[i] being high predict X_{t+1}[j] changing
     more than when X_t[i] is low?

Both lifts are mapped to [0, 1] via a sigmoid centred at 0
(lift = 0  →  score = 0.5, meaning "no evidence").  The two scores are then
blended with configurable weights (lambda_int, lambda_obs).

The resulting probability matrix is blended into the existing beliefs with a
momentum term so that a single noisy update does not swamp prior knowledge.

API is a superset of SimplePCG:
  .probs          – [d, d] edge-probability matrix
  .entropy()      – total Bernoulli entropy (used by the planner for phase switching)
  .apply_update() – standard apply when probs were computed externally
  .fit_from_transitions(X_t, X_tp1, mask) – mask-aware fitting (main new method)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .intrinsic import bernoulli_kl, entropy_bernoulli


@dataclass
class GrangerPCGConfig:
    num_vars: int
    # Prior / initialisation
    init_edge_prob: float = 0.5    # start at maximum uncertainty so entropy > entropy_high

    # Bayesian smoothing
    alpha: float = 2.0             # Beta pseudo-count added to numerator & denominator

    # What counts as a "change event" for a variable
    change_threshold: float = 0.05  # |X_tp1[j] - X_t[j]| > this → j "changed"

    # Evidence blending weights
    lambda_int: float = 0.6        # weight for interventional lift
    lambda_obs: float = 0.4        # weight for observational Granger score

    # Sigmoid steepness: larger k → sharper 0/1 boundary around zero lift
    sigmoid_k: float = 6.0

    # Minimum observations required in each condition before computing a score
    min_obs: int = 4

    # Momentum: new_probs = momentum * old + (1-momentum) * computed
    # Prevents single updates from overcorrecting; set to 0 to disable.
    momentum: float = 0.2

    seed: Optional[int] = None


def _sigmoid(x: float, k: float = 1.0) -> float:
    """Numerically stable logistic: sigmoid(k * x)."""
    kx = k * x
    if kx >= 0:
        return 1.0 / (1.0 + np.exp(-kx))
    e = np.exp(kx)
    return e / (1.0 + e)


class GrangerPCG:
    """
    Intervention-aware Granger-causal Probabilistic Causal Graph.

    Fixes the following weaknesses of SimplePCG._transition_fit_probs:

    * Uses the PCGBuffer intervention mask so intervened transitions are
      separated from observational ones.
    * Measures actual *change events* (|Δx| > threshold) rather than the
      direction of change, which is better for binary / near-binary variables.
    * Applies a sigmoid to turn lift scores into calibrated probabilities,
      avoiding the degenerate P=0.5 floor that keeps entropy artificially high.
    * Starts at init_edge_prob = 0.5 (maximum uncertainty) so that the planner
      correctly begins in the "novel" exploration phase.
    """

    def __init__(self, cfg: GrangerPCGConfig):
        self.cfg = cfg
        d = cfg.num_vars
        p0 = np.full((d, d), float(cfg.init_edge_prob), dtype=float)
        np.fill_diagonal(p0, 0.0)
        self._probs = p0
        self._step = 0

    # ------------------------------------------------------------------ #
    #  Standard SimplePCG-compatible API                                   #
    # ------------------------------------------------------------------ #

    @property
    def probs(self) -> np.ndarray:
        return self._probs.copy()

    def entropy(self) -> float:
        """Total Bernoulli entropy over all off-diagonal edges."""
        p = self._probs.copy()
        np.fill_diagonal(p, 0.0)
        return entropy_bernoulli(p)

    def apply_update(self, new_probs: np.ndarray) -> float:
        """
        Blend externally-computed probability estimates into current beliefs.

        Uses momentum to smooth out noisy single-batch updates.
        Returns the realised KL-divergence (information gain).
        """
        new_p = np.clip(np.asarray(new_probs, dtype=float), 0.0, 1.0)
        np.fill_diagonal(new_p, 0.0)
        ig = float(bernoulli_kl(new_p, self._probs))
        m = self.cfg.momentum
        self._probs = m * self._probs + (1.0 - m) * new_p
        np.fill_diagonal(self._probs, 0.0)
        self._step += 1
        return ig

    def suggest_edges_by_uncertainty(self, k: int = 5) -> np.ndarray:
        """Return indices of k edges with highest Bernoulli entropy (most uncertain)."""
        p = self._probs.copy()
        np.fill_diagonal(p, 0.0)
        eps = 1e-8
        h = -p * np.log(np.clip(p, eps, 1 - eps)) - (1 - p) * np.log(np.clip(1 - p, eps, 1 - eps))
        np.fill_diagonal(h, -np.inf)
        flat_idx = np.argsort(h.ravel())[::-1][:k]
        return np.array([divmod(int(fi), p.shape[1]) for fi in flat_idx], dtype=int)

    # ------------------------------------------------------------------ #
    #  Mask-aware causal discovery                                         #
    # ------------------------------------------------------------------ #

    def fit_from_transitions(
        self,
        X_t: np.ndarray,    # [N, d] pre-option state snapshot
        X_tp1: np.ndarray,  # [N, d] post-option state snapshot
        mask: np.ndarray,   # [N, d] mask[k, i] = 1 iff step k targeted var i
    ) -> np.ndarray:
        """
        Compute a new [d, d] matrix of edge probabilities from buffered transitions.

        Does NOT update self._probs in-place; call apply_update() on the result.

        Three complementary signals per edge (i → j):

        Signal 1 – Prerequisite score  [weight: lambda_int]
          Among transitions where skill-j was explicitly targeted (mask[:,j]=1),
          does X_t[i]=1 make success (j actually changing) more likely?

              p_succ | i=1 = P(j changes | targeting j, X_t[i]=1)
              p_succ | i=0 = P(j changes | targeting j, X_t[i]=0)
              lift = p_succ|i=1 − p_succ|i=0
              prereq_score = sigmoid(sigmoid_k * lift)

          This detects prerequisite relationships: "you need i to succeed at j".

        Signal 2 – Interventional co-change  [weight: lambda_int * 0.5]
          When targeting i (mask[:,i]=1), does j co-change more than baseline?

              lift_co = P(j changes|target i) − P(j changes|not target i)
              co_score = sigmoid(sigmoid_k * lift_co)

          Detects direct effects that propagate within one option step.

        Signal 3 – Observational Granger  [weight: lambda_obs]
          In clean observational steps (neither i nor j targeted), does
          X_t[i]=1 predict X_{t+1}[j] changing?

              lift_obs = P(j changes | X_t[i]=1) − P(j changes | X_t[i]=0)
              obs_score = sigmoid(sigmoid_k * lift_obs)

          Captures correlational regularities not explained by interventions.

        If any signal lacks sufficient data it contributes 0.5 (no information).
        Informed signals are averaged with their weights.
        """
        N = X_t.shape[0]
        d = X_t.shape[1]
        cfg = self.cfg

        if N < cfg.min_obs * 2:
            return self._probs.copy()

        # Binary change indicator per variable
        change = (np.abs(X_tp1 - X_t) > cfg.change_threshold).astype(float)  # [N, d]

        new_probs = np.full((d, d), 0.5, dtype=float)

        for i in range(d):
            int_i: np.ndarray = mask[:, i] > 0.5   # [N] bool: steps targeting var i
            obs_i: np.ndarray = ~int_i              # [N] bool
            n_int = int(int_i.sum())
            n_obs = int(obs_i.sum())

            for j in range(d):
                if i == j:
                    new_probs[i, j] = 0.0
                    continue

                scores: list = []
                weights: list = []

                # ── Signal 1: Prerequisite (conditional success rate) ──────────
                # Among steps that explicitly targeted var j, does pre-state i=1
                # increase the probability of j actually changing?
                tgt_j: np.ndarray = mask[:, j] > 0.5
                n_tgt_j = int(tgt_j.sum())

                if n_tgt_j >= 2 * cfg.min_obs:
                    ch_j_tgt = change[tgt_j, j]
                    Xi_at_tgt = X_t[tgt_j, i]
                    hi_prereq: np.ndarray = Xi_at_tgt > 0.5
                    lo_prereq: np.ndarray = ~hi_prereq
                    n_hi_p = int(hi_prereq.sum())
                    n_lo_p = int(lo_prereq.sum())
                    if n_hi_p >= cfg.min_obs and n_lo_p >= cfg.min_obs:
                        p_s1 = (ch_j_tgt[hi_prereq].sum() + cfg.alpha) / (n_hi_p + 2.0 * cfg.alpha)
                        p_s0 = (ch_j_tgt[lo_prereq].sum() + cfg.alpha) / (n_lo_p + 2.0 * cfg.alpha)
                        lift_prereq = float(p_s1 - p_s0)
                        scores.append(_sigmoid(lift_prereq, cfg.sigmoid_k))
                        weights.append(cfg.lambda_int)

                # ── Signal 2: Interventional co-change ────────────────────────
                if n_int >= cfg.min_obs and n_obs >= cfg.min_obs:
                    p_co  = (change[int_i, j].sum() + cfg.alpha) / (n_int + 2.0 * cfg.alpha)
                    p_base = (change[obs_i, j].sum() + cfg.alpha) / (n_obs + 2.0 * cfg.alpha)
                    lift_co = float(p_co - p_base)
                    scores.append(_sigmoid(lift_co, cfg.sigmoid_k))
                    weights.append(cfg.lambda_int * 0.5)

                # ── Signal 3: Observational Granger ───────────────────────────
                clean_obs: np.ndarray = obs_i & (mask[:, j] < 0.5)
                n_clean = int(clean_obs.sum())
                if n_clean >= 2 * cfg.min_obs:
                    X_i_clean = X_t[clean_obs, i]
                    ch_j_clean = change[clean_obs, j]
                    hi: np.ndarray = X_i_clean > 0.5
                    lo: np.ndarray = ~hi
                    n_hi = int(hi.sum())
                    n_lo = int(lo.sum())
                    if n_hi >= cfg.min_obs and n_lo >= cfg.min_obs:
                        p_hi = (ch_j_clean[hi].sum() + cfg.alpha) / (n_hi + 2.0 * cfg.alpha)
                        p_lo = (ch_j_clean[lo].sum() + cfg.alpha) / (n_lo + 2.0 * cfg.alpha)
                        lift_obs = float(p_hi - p_lo)
                        scores.append(_sigmoid(lift_obs, cfg.sigmoid_k))
                        weights.append(cfg.lambda_obs)

                # ── Combine available signals ─────────────────────────────────
                if scores:
                    total_w = sum(weights)
                    combined = sum(s * w for s, w in zip(scores, weights)) / total_w
                    new_probs[i, j] = float(combined)
                # else: leave at 0.5 (insufficient data for any signal)

        np.fill_diagonal(new_probs, 0.0)
        return new_probs

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                         #
    # ------------------------------------------------------------------ #

    def top_edges(self, k: int = 10, var_names=None):
        """Return the k highest-probability edges as a list of (prob, i, j, name) tuples."""
        p = self._probs.copy()
        np.fill_diagonal(p, -1.0)
        flat = np.argsort(p.ravel())[::-1][:k]
        d = p.shape[0]
        out = []
        for fi in flat:
            i, j = divmod(int(fi), d)
            pi = float(self._probs[i, j])
            if pi <= 0:
                break
            ni = var_names[i] if var_names and i < len(var_names) else f"X{i}"
            nj = var_names[j] if var_names and j < len(var_names) else f"X{j}"
            out.append((pi, i, j, f"{ni} → {nj}"))
        return out
