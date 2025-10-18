# src/dia/pcg_variational.py
"""
Variational (Gumbel-Concrete) PCG with acyclicity on mean probabilities.

We maintain:
  - logits L (learned): edge probabilities P = sigmoid(L)
  - continuous strengths B (learned): effective adjacency per sample = A_tilde ⊙ B
  - A_tilde ~ RelaxedBernoulli(Gumbel-Concrete) with temperature tau

Loss per iteration (using K Monte Carlo samples):
  - Regression: mean_k [ || X - X (A_tilde_k ⊙ B) ||_F^2 / (2 * |mask_keep|) ]
    (with optional interventional mask to ignore residuals for intervened variables)
  - Regularizers: L1(B), L2(B)
  - Acyclicity: h(P) = trace(expm(P ⊙ P)) - d
  - Optional entropy encouragement on P early in training (omitted here by default)

API mirrors pcg_learner where reasonable:
  - probs -> numpy [d,d]
  - entropy(), expected_ig_from_update(), apply_update()
  - fit(X, mask=None, epochs=...)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

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


def _h_func(P: torch.Tensor) -> torch.Tensor:
    """Acyclicity on mean probabilities P (continuous DAG proxy)."""
    d = P.shape[0]
    PP = P * P
    expm = torch.matrix_exp(PP)
    return torch.trace(expm) - float(d)


@dataclass
class VariationalPCGConfig:
    num_vars: int
    device: Optional[str] = None
    init_scale: float = 0.01
    tau: float = 0.5           # Gumbel-Concrete temperature
    l1_penalty: float = 0.01   # on B
    l2_penalty: float = 0.01   # on B
    lr: float = 1e-2
    max_iter: int = 1000
    h_tol: float = 1e-8
    rho_max: float = 1e16
    rho_init: float = 1.0
    w_decay: float = 0.0
    K: int = 4                 # MC samples per iteration
    verbose: bool = False


class VariationalPCG:
    """Variational relaxed-Bernoulli PCG with acyclicity penalty on mean probs."""

    def __init__(self, cfg: VariationalPCGConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.d = cfg.num_vars

        L = torch.zeros((self.d, self.d), device=self.device)  # logits init at 0 -> P=0.5
        with torch.no_grad():
            L.fill_diagonal_(-1e9)  # force no self-loops
        self.L = nn.Parameter(L)

        B = (torch.randn(self.d, self.d, device=self.device) * cfg.init_scale)
        with torch.no_grad():
            B.fill_diagonal_(0.0)
        self.B = nn.Parameter(B)

        self._probs_cache = None
        self.step = 0

    @torch.no_grad()
    def _probs_torch(self) -> torch.Tensor:
        P = torch.sigmoid(self.L)
        P.fill_diagonal_(0.0)
        return P

    @property
    def probs(self) -> np.ndarray:
        if self._probs_cache is None:
            self._probs_cache = self._probs_torch().detach().cpu().numpy()
        return self._probs_cache

    def entropy(self) -> float:
        p = self.probs.copy()
        np.fill_diagonal(p, 0.0)
        return entropy_bernoulli(p)

    def expected_ig_from_update(self, new_probs: np.ndarray) -> float:
        old = self.probs
        return bernoulli_kl(new_probs, old)

    def apply_update(self, new_probs: np.ndarray) -> float:
        """Set mean probabilities to new_probs by adjusting logits."""
        eps = 1e-8
        p = np.clip(new_probs, eps, 1 - eps)
        inv = np.log(p / (1 - p))
        inv[np.eye(self.d, dtype=bool)] = -1e9  # diag -inf
        with torch.no_grad():
            self.L.copy_(torch.tensor(inv, device=self.device, dtype=self.L.dtype))
            self.B.data.diagonal().zero_()
        self._probs_cache = None
        return self.expected_ig_from_update(new_probs)

    def _sample_relaxed(self, P: torch.Tensor, K: int) -> torch.Tensor:
        """Sample K relaxed Bernoulli matrices (Gumbel-Concrete). Returns [K, d, d]."""
        U = torch.rand((K, self.d, self.d), device=self.device)
        g = -torch.log(-torch.log(U + 1e-8) + 1e-8)
        logits = torch.log(P + 1e-8) - torch.log(1 - P + 1e-8)
        A_tilde = torch.sigmoid((logits + g) / max(1e-6, self.cfg.tau))
        # zero diagonal
        eye = torch.eye(self.d, device=self.device).unsqueeze(0)  # [1, d, d]
        A_tilde = A_tilde * (1.0 - eye)
        return A_tilde

    def fit(self, X: np.ndarray, mask: Optional[np.ndarray] = None, epochs: Optional[int] = None) -> dict:
        """
        Fit variational PCG by minimizing relaxed loss.
        Args:
          X: numpy [N, d], mean-centered internally
          mask: optional numpy [N, d], 1 if var j intervened in sample n (ignore residual there)
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
            Wmask = torch.tensor(1.0 - mask, dtype=X_t.dtype, device=self.device)

        optimizer = optim.Adam([self.L, self.B], lr=self.cfg.lr, weight_decay=self.cfg.w_decay)

        rho = float(self.cfg.rho_init)
        alpha_lag = 0.0
        rho_max = float(self.cfg.rho_max)
        h_tol = float(self.cfg.h_tol)
        max_iter = int(epochs or self.cfg.max_iter)
        l1 = float(self.cfg.l1_penalty)
        l2 = float(self.cfg.l2_penalty)

        last_loss = None
        prev_h = None

        if self.cfg.verbose:
            print(f"[VariationalPCG] fit: n={n} d={d} K={self.cfg.K} max_iter={max_iter} device={self.device}")

        for it in range(max_iter):
            optimizer.zero_grad()

            P = torch.sigmoid(self.L)
            P = P * (1.0 - torch.eye(d, device=self.device))
            A_samples = self._sample_relaxed(P, self.cfg.K)  # [K, d, d]
            B = self.B * (1.0 - torch.eye(d, device=self.device))

            reg_losses = []
            for k in range(self.cfg.K):
                A_eff = A_samples[k] * B  # [d, d]
                X_pred = X_t @ A_eff
                resid = (X_t - X_pred)
                resid2 = (resid * Wmask) ** 2
                reg_losses.append(0.5 * resid2.sum() / float(torch.count_nonzero(Wmask)))
            loss_reg = torch.stack(reg_losses).mean()

            loss_l1 = l1 * torch.sum(torch.abs(B))
            loss_l2 = 0.5 * l2 * torch.sum(B ** 2)

            h = _h_func(P)
            lagrangian = loss_reg + loss_l1 + loss_l2 + alpha_lag * h + 0.5 * rho * (h ** 2)

            lagrangian.backward()
            optimizer.step()

            with torch.no_grad():
                self.B.data.diagonal().zero_()
                self.L.data.diagonal().fill_(-1e9)

            last_loss = float(lagrangian.item())
            h_val = float(h.item())

            if self.cfg.verbose and (it % max(1, max_iter // 10) == 0):
                print(f"iter {it:4d} loss={last_loss:.6e} h={h_val:.3e} rho={rho:.3e}")

            if prev_h is None:
                prev_h = abs(h_val)
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
        return {
            "probs": self.probs,
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
