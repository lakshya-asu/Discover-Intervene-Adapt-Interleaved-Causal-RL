# src/dia/cwm.py
"""
Causal World Model (CWM) for DIA — PCG-masked transition predictor.

Architecture
------------
    x̂_{t+1}^j = sigmoid( Σ_i A_{ij} · g_{ij}(x_t^i) + h_j(skill_k) )

Where:
  A_{ij}  = PCG soft edge weight (SimplePCG.probs[i,j])
  g_{ij}  = per-edge scalar MLP: Linear(1→edge_hidden→1)  (M² networks)
  h_j     = skill-conditioned bias: Embedding(K, skill_emb_dim) → Linear(→M)
  Loss    = BCE(x̂_{t+1}, x_{t+1})  per-variable binary cross-entropy

Usage
-----
Offline pre-training (from 2D symbolic transitions):
    trainer = CWMTrainer(CWMConfig(), pcg=my_pcg)
    trainer.fit_offline(transitions)        # list[(x_t, skill_idx, x_tp1)]
    trainer.save("cwm_2d.pt")

Online fine-tuning (during 3D transfer):
    trainer.load("cwm_2d.pt")
    # after each skill option completes:
    for (x_t, k, x_tp1) in skill_transitions:
        trainer.update_online(x_t, k, x_tp1)
    delta = trainer.pcg_delta(x_t, k, x_tp1, pcg.probs)
    pcg.conservative_update(delta, lr=cfg.pcg_lr)
    trainer.prune_pcg(pcg, threshold=cfg.prune_threshold)
"""
from __future__ import annotations

import collections
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CWMConfig:
    num_vars: int = 9           # DIA variable dimension (M)
    num_skills: int = 9         # number of distinct skill indices (K = M for Minecraft)
    edge_hidden: int = 16       # hidden units in per-edge MLP g_{ij}
    skill_emb_dim: int = 16     # skill embedding dimension
    offline_lr: float = 1e-3   # Adam lr for offline pre-training
    offline_epochs: int = 50    # epochs over 2D buffer
    offline_batch: int = 64     # mini-batch size for offline training
    online_lr: float = 1e-4    # Adam lr for online fine-tuning
    online_batch: int = 32      # mini-batch size for online update
    online_steps: int = 4       # gradient steps per online call
    buffer_maxlen: int = 10_000 # max replay buffer size
    prune_threshold: float = 0.05   # mean contribution below this → prune PCG edge
    pcg_lr: float = 0.05        # step size for SimplePCG.conservative_update()
    min_transitions_to_prune: int = 20  # wait for this many 3D transitions before pruning


# ---------------------------------------------------------------------------
# Neural Network
# ---------------------------------------------------------------------------

class _EdgeNet(nn.Module):
    """Per-edge MLP: scalar input (x_t^i ∈ {0,1}) → scalar contribution."""
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (...,) scalar or (...,1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.net(x).squeeze(-1)   # (...,)


class CausalWorldModel(nn.Module):
    """
    PCG-masked transition model over M binary DIA variables.

    forward(x_t, skill_idx, pcg_probs) → logits [B, M]
    """

    def __init__(self, cfg: CWMConfig):
        super().__init__()
        self.cfg = cfg
        M, K = cfg.num_vars, cfg.num_skills

        # M × M per-edge networks stored as a flat ModuleList (index = i*M + j)
        self.edge_nets = nn.ModuleList(
            [_EdgeNet(cfg.edge_hidden) for _ in range(M * M)]
        )

        # Skill-conditioned bias: Embedding → Linear → M values
        self.skill_emb  = nn.Embedding(K, cfg.skill_emb_dim)
        self.skill_head = nn.Linear(cfg.skill_emb_dim, M)

        # Learnable per-variable bias
        self.var_bias = nn.Parameter(torch.zeros(M))

    def forward(
        self,
        x_t: "torch.Tensor",           # [B, M] float in {0,1}
        skill_idx: "torch.Tensor",      # [B] int64
        pcg_probs: "torch.Tensor",      # [M, M] float in [0,1]  (soft PCG mask)
    ) -> "torch.Tensor":
        """Return logits (pre-sigmoid) for x_{t+1}. Shape: [B, M]."""
        B, M = x_t.shape

        # ── Skill contribution: [B, M] ────────────────────────────────────────
        skill_bias = self.skill_head(self.skill_emb(skill_idx))   # [B, M]

        # ── Edge contributions: Σ_i A_{ij} · g_{ij}(x_t^i) ──────────────────
        # Compute all edge outputs at once
        # edge_out[b, i, j] = g_{ij}(x_t[b, i])
        # Stack edge_nets outputs into [B, M, M] tensor
        x_col = x_t.unsqueeze(2).expand(B, M, M).contiguous()   # [B, M, M]
        edge_out = torch.zeros(B, M, M, device=x_t.device)
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                net_ij = self.edge_nets[i * M + j]
                edge_out[:, i, j] = net_ij(x_col[:, i, j])      # [B]

        # Mask by PCG probabilities: A_{ij} ∈ [0,1]
        # pcg_probs: [M, M] → broadcast to [B, M, M]
        masked = edge_out * pcg_probs.unsqueeze(0)               # [B, M, M]

        # Sum over cause dimension i → [B, M]
        causal_contrib = masked.sum(dim=1)                        # [B, M]

        # ── Combine ────────────────────────────────────────────────────────────
        logits = causal_contrib + skill_bias + self.var_bias     # [B, M]
        return logits

    def predict_proba(
        self,
        x_t: "torch.Tensor",
        skill_idx: "torch.Tensor",
        pcg_probs: "torch.Tensor",
    ) -> "torch.Tensor":
        """Sigmoid probabilities for x_{t+1}. Shape: [B, M]."""
        return torch.sigmoid(self.forward(x_t, skill_idx, pcg_probs))

    def loss(
        self,
        x_t: "torch.Tensor",       # [B, M]
        skill_idx: "torch.Tensor", # [B]
        x_tp1: "torch.Tensor",     # [B, M]  target (binary)
        pcg_probs: "torch.Tensor", # [M, M]
    ) -> "torch.Tensor":
        """Mean BCE loss over all variables and batch."""
        logits = self.forward(x_t, skill_idx, pcg_probs)
        return F.binary_cross_entropy_with_logits(logits, x_tp1)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class CWMTrainer:
    """
    Manages CausalWorldModel training (offline + online) and PCG interaction.

    Parameters
    ----------
    cfg   : CWMConfig
    pcg   : SimplePCG (for reading probs during training; updated via pcg_delta)
    device: 'cpu' or 'cuda' (default: auto)
    """

    def __init__(self, cfg: CWMConfig, pcg=None, device: Optional[str] = None):
        if not TORCH_OK:
            raise RuntimeError("PyTorch is required for CWMTrainer. Install torch.")
        self.cfg = cfg
        self.pcg = pcg

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = CausalWorldModel(cfg).to(self.device)
        self._offline_opt = torch.optim.Adam(self.model.parameters(), lr=cfg.offline_lr)
        self._online_opt  = torch.optim.Adam(self.model.parameters(), lr=cfg.online_lr)

        # Online replay buffer: deque of (x_t, skill_idx, x_tp1) numpy arrays
        self._buffer: collections.deque = collections.deque(maxlen=cfg.buffer_maxlen)

        # Running mean edge contributions for pruning: shape [M, M]
        M = cfg.num_vars
        self._contrib_ema = np.zeros((M, M), dtype=float)
        self._contrib_n   = 0
        self._online_n    = 0   # total online transitions seen

    # ── Utilities ────────────────────────────────────────────────────────────

    def _pcg_tensor(self, pcg_probs: Optional[np.ndarray] = None) -> "torch.Tensor":
        """Get PCG probs as a float32 tensor on device."""
        if pcg_probs is None:
            if self.pcg is None:
                # Uniform: all edges equally likely
                p = np.full((self.cfg.num_vars, self.cfg.num_vars), 0.5)
                np.fill_diagonal(p, 0.0)
            else:
                p = self.pcg.probs
        else:
            p = pcg_probs
        return torch.tensor(p, dtype=torch.float32, device=self.device)

    def _to_tensor(self, arr: np.ndarray, dtype=torch.float32) -> "torch.Tensor":
        return torch.tensor(arr, dtype=dtype, device=self.device)

    def _batch_from_list(
        self,
        transitions: List[Tuple[np.ndarray, int, np.ndarray]],
        batch_size: int,
    ):
        """Yield random mini-batches of (x_t, skill_idx, x_tp1) tensors."""
        n = len(transitions)
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = idx[start: start + batch_size]
            xs  = np.stack([transitions[i][0] for i in batch_idx])
            ks  = np.array([transitions[i][1] for i in batch_idx], dtype=np.int64)
            xps = np.stack([transitions[i][2] for i in batch_idx])
            yield (
                self._to_tensor(xs),
                torch.tensor(ks, device=self.device),
                self._to_tensor(xps),
            )

    # ── Offline pre-training ──────────────────────────────────────────────────

    def fit_offline(
        self,
        transitions: List[Tuple[np.ndarray, int, np.ndarray]],
        pcg_probs: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> List[float]:
        """
        Train the CWM on 2D symbolic transitions.

        Parameters
        ----------
        transitions : list of (x_t [M], skill_idx int, x_tp1 [M])
        pcg_probs   : [M, M] edge weights to use during training.
                      If None, uses self.pcg.probs (or uniform if no pcg).
        verbose     : print loss every 10 epochs

        Returns
        -------
        List of per-epoch mean losses.
        """
        if not transitions:
            return []

        A = self._pcg_tensor(pcg_probs)
        cfg = self.cfg
        losses = []

        self.model.train()
        for epoch in range(cfg.offline_epochs):
            epoch_losses = []
            for x_t, skill_idx, x_tp1 in self._batch_from_list(transitions, cfg.offline_batch):
                self._offline_opt.zero_grad()
                loss = self.model.loss(x_t, skill_idx, x_tp1, A)
                loss.backward()
                self._offline_opt.step()
                epoch_losses.append(loss.item())
            mean_loss = float(np.mean(epoch_losses))
            losses.append(mean_loss)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  [CWM offline] epoch {epoch+1}/{cfg.offline_epochs}  loss={mean_loss:.4f}")

        self.model.eval()
        return losses

    # ── Online fine-tuning ────────────────────────────────────────────────────

    def update_online(
        self,
        x_t: np.ndarray,
        skill_idx: int,
        x_tp1: np.ndarray,
        pcg_probs: Optional[np.ndarray] = None,
    ) -> float:
        """
        Add one transition to the replay buffer and do a mini-batch update.

        Returns the mean loss over online_steps gradient steps.
        """
        self._buffer.append((x_t.copy(), int(skill_idx), x_tp1.copy()))
        self._online_n += 1

        if len(self._buffer) < self.cfg.online_batch:
            return 0.0

        A = self._pcg_tensor(pcg_probs)
        buf = list(self._buffer)
        cfg = self.cfg
        step_losses = []

        self.model.train()
        for _ in range(cfg.online_steps):
            batch_idx = np.random.choice(len(buf), size=cfg.online_batch, replace=False)
            xs  = self._to_tensor(np.stack([buf[i][0] for i in batch_idx]))
            ks  = torch.tensor([buf[i][1] for i in batch_idx], device=self.device)
            xps = self._to_tensor(np.stack([buf[i][2] for i in batch_idx]))

            self._online_opt.zero_grad()
            loss = self.model.loss(xs, ks, xps, A)
            loss.backward()
            self._online_opt.step()
            step_losses.append(loss.item())

        self.model.eval()
        return float(np.mean(step_losses))

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        x_t: np.ndarray,
        skill_idx: int,
        pcg_probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return binary predictions for x_{t+1}. Shape: [M]."""
        A = self._pcg_tensor(pcg_probs)
        xt = self._to_tensor(x_t[None])          # [1, M]
        k  = torch.tensor([skill_idx], device=self.device)
        proba = self.model.predict_proba(xt, k, A)  # [1, M]
        return (proba[0].cpu().numpy() >= 0.5).astype(float)

    @torch.no_grad()
    def predict_proba(
        self,
        x_t: np.ndarray,
        skill_idx: int,
        pcg_probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return probability predictions for x_{t+1}. Shape: [M]."""
        A = self._pcg_tensor(pcg_probs)
        xt = self._to_tensor(x_t[None])
        k  = torch.tensor([skill_idx], device=self.device)
        proba = self.model.predict_proba(xt, k, A)
        return proba[0].cpu().numpy()

    def prediction_error(
        self,
        x_t: np.ndarray,
        skill_idx: int,
        x_tp1: np.ndarray,
        pcg_probs: Optional[np.ndarray] = None,
    ) -> float:
        """
        Mean absolute prediction error |x̂_{t+1} - x_{t+1}|.
        Use as curiosity / intrinsic reward signal.
        """
        proba = self.predict_proba(x_t, skill_idx, pcg_probs)
        return float(np.mean(np.abs(proba - x_tp1.astype(float))))

    # ── PCG update signal ─────────────────────────────────────────────────────

    @torch.no_grad()
    def pcg_delta(
        self,
        x_t: np.ndarray,
        skill_idx: int,
        x_tp1: np.ndarray,
        pcg_probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute a [M, M] delta matrix to feed into SimplePCG.conservative_update().

        Logic:
          err_j = |x̂_{t+1}^j - x_{t+1}^j|   (how wrong was the model about var j)
          For edge (i→j): if A_{ij} was high AND err_j is high → edge caused misprediction
            → push down: delta[i,j] = -err_j * A_{ij}
          If A_{ij} was high AND err_j is low → edge is genuinely predictive
            → push up slightly: delta[i,j] = +0.1 * A_{ij} * (1 - err_j)
          Net: delta[i,j] = A_{ij} * (0.1*(1-err_j) - err_j)
                           = A_{ij} * (0.1 - 1.1*err_j)

        Returns delta [M, M] for use as:
            pcg.conservative_update(delta, lr=cfg.pcg_lr)
        """
        A_np = self.pcg.probs if pcg_probs is None else pcg_probs
        proba = self.predict_proba(x_t, skill_idx, A_np)
        err   = np.abs(proba - x_tp1.astype(float))   # [M]

        # err broadcast: each column j gets err_j
        # A_np[i,j] = edge weight i→j
        A_np = A_np.copy()
        np.fill_diagonal(A_np, 0.0)

        # Update rule: push down edges that caused prediction errors
        delta = A_np * (0.1 - 1.1 * err[np.newaxis, :])   # [M, M]
        np.fill_diagonal(delta, 0.0)
        return delta

    # ── PCG pruning ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _update_contrib_ema(
        self,
        x_t: np.ndarray,
        skill_idx: int,
        pcg_probs: Optional[np.ndarray] = None,
        alpha: float = 0.1,
    ):
        """
        Update exponential moving average of edge contributions.
        contrib[i,j] = |A_{ij} · g_{ij}(x_t^i)|   (how much edge i→j contributes)
        """
        A_np = self.pcg.probs if pcg_probs is None else pcg_probs
        A    = self._pcg_tensor(A_np)
        M    = self.cfg.num_vars

        xt = self._to_tensor(x_t[None])   # [1, M]
        k  = torch.tensor([skill_idx], device=self.device)

        # Compute per-edge contribution magnitude
        contrib = np.zeros((M, M), dtype=float)
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                net_ij  = self.model.edge_nets[i * M + j]
                g_val   = net_ij(xt[:, i]).item()         # scalar
                contrib[i, j] = abs(float(A_np[i, j]) * g_val)

        # EMA update
        if self._contrib_n == 0:
            self._contrib_ema = contrib
        else:
            self._contrib_ema = (1 - alpha) * self._contrib_ema + alpha * contrib
        self._contrib_n += 1

    def prune_pcg(self, pcg, threshold: Optional[float] = None):
        """
        Zero out PCG edges whose mean contribution is below `threshold`.

        Only runs when >= cfg.min_transitions_to_prune online transitions seen.
        Modifies pcg.state.edge_probs in-place.
        """
        if self._online_n < self.cfg.min_transitions_to_prune:
            return
        if threshold is None:
            threshold = self.cfg.prune_threshold

        mask = self._contrib_ema < threshold
        np.fill_diagonal(mask, False)
        pruned = int(mask.sum())
        if pruned > 0:
            pcg.state.edge_probs[mask] = 0.0
            np.fill_diagonal(pcg.state.edge_probs, 0.0)

    # ── Convenience: process full skill trajectory ────────────────────────────

    def process_skill_trajectory(
        self,
        trajectory: list,
        skill_idx: int,
        evgs,
        pcg,
        verbose: bool = False,
    ) -> dict:
        """
        Process all transitions from a completed skill option.run() trajectory.

        Parameters
        ----------
        trajectory  : list of (obs_t, action, obs_tp1, succ) from option.run()
        skill_idx   : DIA variable index of the skill that ran
        evgs        : EVGS extractor
        pcg         : SimplePCG (updated in-place)
        verbose     : print per-step summary

        Returns
        -------
        dict with:
          'mean_error'   : float — mean prediction error (curiosity signal)
          'online_loss'  : float — mean fine-tuning loss
          'pcg_delta'    : np.ndarray [M,M] — aggregate delta applied to PCG
          'n_transitions': int
        """
        errors     = []
        online_losses = []
        agg_delta  = np.zeros((self.cfg.num_vars, self.cfg.num_vars))
        n = 0

        for (obs_t, _act, obs_tp1, _succ) in trajectory:
            x_t  = evgs.extract(obs_t)
            x_tp1 = evgs.extract(obs_tp1)

            if np.allclose(x_t, x_tp1):
                continue   # no variable change — low information

            err  = self.prediction_error(x_t, skill_idx, x_tp1, pcg.probs)
            errors.append(err)

            delta = self.pcg_delta(x_t, skill_idx, x_tp1, pcg.probs)
            agg_delta += delta

            online_loss = self.update_online(x_t, skill_idx, x_tp1, pcg.probs)
            online_losses.append(online_loss)

            self._update_contrib_ema(x_t, skill_idx, pcg.probs)
            n += 1

        # Apply aggregated PCG delta (normalised by n_transitions to avoid overcorrection)
        if n > 0:
            norm_delta = agg_delta / n
            pcg.conservative_update(norm_delta, lr=self.cfg.pcg_lr)
            self.prune_pcg(pcg)

        return {
            "mean_error":    float(np.mean(errors))   if errors        else 0.0,
            "online_loss":   float(np.mean(online_losses)) if online_losses else 0.0,
            "pcg_delta":     agg_delta,
            "n_transitions": n,
        }

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save model weights and EMA state."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "contrib_ema": self._contrib_ema,
            "contrib_n":   self._contrib_n,
            "online_n":    self._online_n,
            "cfg":         self.cfg,
        }, path)

    def load(self, path: str):
        """Load model weights and EMA state."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self._contrib_ema = ckpt.get("contrib_ema", self._contrib_ema)
        self._contrib_n   = ckpt.get("contrib_n",   0)
        self._online_n    = ckpt.get("online_n",    0)
        self.model.eval()
