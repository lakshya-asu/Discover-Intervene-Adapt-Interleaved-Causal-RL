# src/dia/pcg_learner.py
"""
Differentiable PCG (NOTEARS-style) learner with interventional weighting.

Implements a lightweight version of NOTEARS (Zheng et al., 2018) using PyTorch
to learn a weighted adjacency matrix W (shape [d,d]) that is acyclic via an
augmented Lagrangian constraint h(W) = trace(expm(W * W)) - d = 0.

This module exposes a probabilistic view of the learned graph:
  p(edge_ij) = sigmoid( alpha * |W_ij| - bias ) / temp

API
---
DifferentiablePCG(cfg):
    - cfg.num_vars
    - cfg.device (optional)
    - prob mapping knobs: alpha, sigmoid_bias, sigmoid_temp

Methods:
    - fit(X, mask=None, epochs=1000): fits W from data X (numpy [N, d]).
      mask: optional numpy [N, d] where mask[n, j]==1 indicates that variable j
            was intervened in sample n (so its residual term is ignored).
    - probs -> np.ndarray [d,d]: current Bernoulli-like probs for edges
    - entropy() -> float: sum Bernoulli entropies across edges (excl diag)
    - expected_ig_from_update(new_probs) -> float: KL(new||old)
    - apply_update(new_probs) -> float: adjust weights to match probs (inverse map)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

try:
    from .intrinsic import bernoulli_kl, entropy_bernoulli
except Exception:
    def bernoulli_kl(p_new: np.ndarray, p_old: np.ndarray, eps: float = 1e-8) -> float:
        p_new = np.clip(p_new, eps, 1 - eps)
        p_old = np.clip(p_old, eps, 1 - eps)
        kl = p_new * np.log(p_new / p_old) + (1 - p_new) * np.log((1 - p_new) / (1 - p_old))
        return float(np.sum(kl))

    def entropy_bernoulli(p: np.ndarray, eps: float = 1e-8) -> float:
        p = np.clip(p, eps, 1 - eps)
        h = -p * np.log(p) - (1 - p) * np.log(1 - p)
        return float(np.sum(h))


@dataclass
class DifferentiablePCGConfig:
    num_vars: int
    device: Optional[str] = None
    init_scale: float = 0.01
    l1_penalty: float = 0.01
    l2_penalty: float = 0.01
    alpha: float = 1.5
    sigmoid_bias: float = 0.5
    sigmoid_temp: float = 1.0
    lr: float = 1e-2
    max_iter: int = 1000
    h_tol: float = 1e-8
    rho_max: float = 1e16
    rho_init: float = 1.0
    w_decay: float = 0.0
    verbose: bool = False


def _h_func(W: Tensor) -> Tensor:
    """Acyclicity: h(W) = trace(expm(W * W)) - d."""
    d = W.shape[0]
    WW = W * W
    expm = torch.matrix_exp(WW)
    return torch.trace(expm) - float(d)


class DifferentiablePCG:
    """NOTEARS-style differentiable learner with optional interventional mask."""

    def __init__(self, cfg: DifferentiablePCGConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.d = cfg.num_vars

        W = (torch.randn(self.d, self.d, device=self.device) * cfg.init_scale)
        with torch.no_grad():
            W.fill_diagonal_(0.0)
        self.W = nn.Parameter(W)
        self._probs_cache = None
        self.step = 0

    def _weights_numpy(self) -> np.ndarray:
        return self.W.detach().cpu().numpy()

    def _probs_from_weights(self, W_np: Optional[np.ndarray] = None) -> np.ndarray:
        """Map weights -> Bernoulli-like probs via sigmoid(alpha * |W| - bias)."""
        if W_np is None:
            W_np = self._weights_numpy()
        W_abs = np.abs(W_np)
        logits = (self.cfg.alpha * W_abs - self.cfg.sigmoid_bias) / max(1e-8, self.cfg.sigmoid_temp)
        probs = 1.0 / (1.0 + np.exp(-logits))
        np.fill_diagonal(probs, 0.0)
        return probs

    @property
    def probs(self) -> np.ndarray:
        if self._probs_cache is None:
            self._probs_cache = self._probs_from_weights()
        return self._probs_cache

    def entropy(self) -> float:
        p = self.probs.copy()
        np.fill_diagonal(p, 0.0)
        return entropy_bernoulli(p)

    def expected_ig_from_update(self, new_probs: np.ndarray) -> float:
        old = self.probs
        return bernoulli_kl(new_probs, old)

    def apply_update(self, new_probs: np.ndarray) -> float:
        """Adjust weights to approximately match target probabilities."""
        eps = 1e-8
        p = np.clip(new_probs, eps, 1 - eps)
        inv = np.log(p / (1 - p))
        W_abs = (inv * self.cfg.sigmoid_temp + self.cfg.sigmoid_bias) / max(1e-8, self.cfg.alpha)
        np.fill_diagonal(W_abs, 0.0)
        W_np = self._weights_numpy()
        signs = np.sign(W_np)
        new_W = signs * W_abs
        with torch.no_grad():
            self.W.copy_(torch.tensor(new_W, device=self.device, dtype=self.W.dtype))
            self.W.data.diagonal().zero_()
        self._probs_cache = None
        ig = self.expected_ig_from_update(new_probs)
        return float(ig)

    def fit(self, X: np.ndarray, mask: Optional[np.ndarray] = None, epochs: Optional[int] = None) -> dict:
        """
        Fit W from data X using augmented Lagrangian NOTEARS.
        Args:
          X: numpy [N, d], mean-centered internally
          mask: optional numpy [N, d], 1 if variable j was intervened in sample n (residual ignored)
        """
        if X.ndim != 2 or X.shape[1] != self.d:
            raise ValueError(f"X must be shape [N, {self.d}]")

        X = np.asarray(X, dtype=float)
        X = X - X.mean(axis=0, keepdims=True)

        n, d = X.shape
        X_t = torch.tensor(X, dtype=torch.get_default_dtype(), device=self.device)

        if mask is None:
            Wmask = torch.ones((n, d), device=self.device, dtype=X_t.dtype)
        else:
            mask = np.asarray(mask, dtype=float)
            if mask.shape != (n, d):
                raise ValueError(f"mask must be shape [N, d] = [{n}, {d}]")
            Wmask = torch.tensor(1.0 - mask, dtype=X_t.dtype, device=self.device)  # 1 means keep, 0 ignore

        rho = float(self.cfg.rho_init)
        alpha_lag = 0.0
        optimizer = optim.Adam([self.W], lr=self.cfg.lr, weight_decay=self.cfg.w_decay)

        max_iter = int(epochs or self.cfg.max_iter)
        l1 = float(self.cfg.l1_penalty)
        l2 = float(self.cfg.l2_penalty)
        h_tol = float(self.cfg.h_tol)
        rho_max = float(self.cfg.rho_max)

        last_loss = None
        prev_h = None

        if self.cfg.verbose:
            print(f"[DifferentiablePCG] fit: n={n} d={d} max_iter={max_iter} device={self.device}")

        for it in range(max_iter):
            optimizer.zero_grad()
            W = self.W
            W = W * (1.0 - torch.eye(d, device=self.device))  # zero diag in forward too
            X_pred = X_t @ W

            resid = (X_t - X_pred)  # [N, d]
            # Mask residuals for intervened (weight 0): elementwise multiply
            resid2 = (resid * Wmask) ** 2
            loss_reg = 0.5 * resid2.sum() / float(torch.count_nonzero(Wmask))

            loss_l1 = l1 * torch.sum(torch.abs(W))
            loss_l2 = 0.5 * l2 * torch.sum(W ** 2)

            h = _h_func(W)
            lagrangian = loss_reg + loss_l1 + loss_l2 + alpha_lag * h + 0.5 * rho * (h ** 2)

            lagrangian.backward()
            optimizer.step()

            with torch.no_grad():
                self.W.data.diagonal().zero_()

            last_loss = float(lagrangian.item())
            h_val = float(h.item())

            if self.cfg.verbose and (it % max(1, max_iter // 10) == 0):
                print(f"iter {it:4d} loss={last_loss:.6e} h={h_val:.3e} rho={rho:.3e}")

            # Update rho and alpha per augmented Lagrangian heuristic
            if prev_h is None:
                prev_h = abs(h_val)
            # If not improving enough, increase rho
            if abs(h_val) > 0.25 * prev_h:
                rho = min(rho * 10.0, rho_max)
            alpha_lag = alpha_lag + rho * h_val
            prev_h = abs(h_val)

            if abs(h_val) <= h_tol and it > 0:
                if self.cfg.verbose:
                    print(f"Converged at iter {it}, h={h_val:.3e}")
                break

        self._probs_cache = None
        self.step += 1
        W_np = self._weights_numpy()
        probs = self._probs_from_weights(W_np)
        return {
            "W": W_np,
            "probs": probs,
            "h": float(h.item()) if isinstance(h, torch.Tensor) else float(h),
            "last_loss": last_loss,
            "steps": it + 1,
        }

    def suggest_edges_by_uncertainty(self, k: int = 5) -> np.ndarray:
        p = self.probs.copy()
        np.fill_diagonal(p, 0.0)
        eps = 1e-8
        h = -p * np.log(np.clip(p, eps, 1 - eps)) - (1 - p) * np.log(np.clip(1 - p, eps, 1 - eps))
        flat_idx = np.argsort(h.ravel())[::-1][:k]
        edges = np.array([divmod(int(i), p.shape[1]) for i in flat_idx], dtype=int)
        return edges
