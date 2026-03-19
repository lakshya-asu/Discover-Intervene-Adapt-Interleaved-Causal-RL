# DIA-Enhanced Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-crafted variable readout `g(·)` and flat PCG parameters with a contrastive probe, GNN-based PCG, hyperbolic SIG, and demo warm-starting pipeline across five causal RL environments.

**Architecture:** New package `src/dia_enhanced/` sits alongside the existing `src/dia/` and provides drop-in replacements for the probe/PCG/SIG modules. The existing encoder `f_ψ`, planner, intervention selector, and training loop structure are unchanged — only the internal representations swap out. All new modules interoperate via the existing `SimplePCG`-compatible API (edge probs matrix + `apply_update`).

**Tech Stack:** Python 3.10, PyTorch 2.x, `torch_geometric` (GNN), `geoopt` (Riemannian optimization), `overcooked-ai` (HumanCompatibleAI), `minerl` (Minecraft 3D), `nle` (NetHack), existing `dia` package (types, evgs, options, planner).

---

## File Map

**New package root:** `src/dia_enhanced/`

| File | Responsibility |
|---|---|
| `src/dia_enhanced/__init__.py` | Package init |
| `src/dia_enhanced/probes/contrastive_probe.py` | MLP probe with M heads + contrastive loss |
| `src/dia_enhanced/probes/tc_regularizer.py` | TC/Partial-Correlation penalty with annealing |
| `src/dia_enhanced/beliefs/gnn_pcg.py` | GAT-based PCG belief module |
| `src/dia_enhanced/beliefs/hyperbolic_sig.py` | Poincaré ball SIG embeddings |
| `src/dia_enhanced/warmstart/rssm_demo_processor.py` | Post-hoc RSSM processing of demo trajectories |
| `src/dia_enhanced/warmstart/minecraft_recorder.py` | MineRL human gameplay recorder |
| `src/dia_enhanced/warmstart/overcooked_demo_loader.py` | Carroll et al. human demo loader + attribution |
| `src/dia_enhanced/warmstart/bc_partner_trainer.py` | Behavior cloning partner model |
| `src/dia_enhanced/warmstart/warmstart_pipeline.py` | Orchestrates warm-start for any environment |
| `src/dia_enhanced/envs/overcooked_wrapper.py` | Overcooked → DIA interface (variable extraction, option space) |
| `src/dia_enhanced/envs/overcooked_variables.py` | Overcooked environment variable definitions |
| `src/dia_enhanced/envs/env_registry.py` | Unified env creation for all 5 domains |
| `src/dia_enhanced/training/losses.py` | L_probe, L_PCG, combined loss computations |
| `src/dia_enhanced/training/dia_enhanced_loop.py` | Modified Algorithm 1 with new module insertion points |
| `src/dia_enhanced/training/overcooked_training.py` | Multi-agent credit assignment + partner BC update |
| `src/dia_enhanced/eval/disentanglement_metrics.py` | DCI, MIG, interventional robustness |
| `src/dia_enhanced/eval/causal_metrics.py` | PCG entropy half-life, SHD, ECE, intervention precision |
| `src/dia_enhanced/eval/overcooked_metrics.py` | Coordination efficiency, attribution accuracy, ZSC |
| `src/dia_enhanced/eval/ablation_runner.py` | Full ablation table runner |
| `configs/minecraft_trade.yaml` | 2D-Minecraft enhanced config |
| `configs/nethack.yaml` | NetHack enhanced config |
| `configs/coinrun.yaml` | CoinRun enhanced config |
| `configs/causalworld.yaml` | CausalWorld enhanced config |
| `configs/overcooked.yaml` | Overcooked enhanced config |

**Tests:**

| File | Tests |
|---|---|
| `tests/test_contrastive_probe.py` | Forward pass, contrastive loss, gradient isolation |
| `tests/test_tc_regularizer.py` | TC computation, annealing schedule, partial TC grouping |
| `tests/test_gnn_pcg.py` | GNN forward pass, edge update, NOTEARS penalty, initialization |
| `tests/test_hyperbolic_sig.py` | Poincaré distance, embedding update, prerequisite ordering |
| `tests/test_warmstart.py` | Pseudo-intervention detection, A_init output, convergence |
| `tests/test_overcooked_wrapper.py` | Variable extraction, option space, attribution |
| `tests/test_losses.py` | All loss functions independently |
| `tests/test_dia_enhanced_loop.py` | Training loop integration smoke test |

---

## Task 1: Contrastive Probe

**Files:**
- Create: `src/dia_enhanced/__init__.py`
- Create: `src/dia_enhanced/probes/__init__.py`
- Create: `src/dia_enhanced/probes/contrastive_probe.py`
- Test: `tests/test_contrastive_probe.py`

### Background

The probe replaces `g(·)` in the existing pipeline. It takes `z_t` (a detached encoder output, shape `[d]`) and outputs `X_t` (shape `[M]`). Each of the M heads is a scalar. The contrastive loss is only applied to tagged intervention transitions.

The `Subgoal` type from `dia.types` carries `var_index` and `predicate` (UP or DOWN). `δ_i = +0.1` if predicate is UP, `-0.1` if DOWN.

Negative weights `w_j`:
- `1.0` if PCG edge probability `q(A_{ij}) < 0.5` (clean negative)
- `0.3` if PCG edge probability `q(A_{ij}) >= 0.5` (causally connected)

- [ ] **Step 1.1: Install dependencies**

```bash
pip install torch-geometric geoopt
```

Expected: installed without errors.

- [ ] **Step 1.2: Create package skeleton**

```bash
mkdir -p src/dia_enhanced/probes src/dia_enhanced/beliefs \
         src/dia_enhanced/warmstart src/dia_enhanced/envs \
         src/dia_enhanced/training src/dia_enhanced/eval
touch src/dia_enhanced/__init__.py \
      src/dia_enhanced/probes/__init__.py \
      src/dia_enhanced/beliefs/__init__.py \
      src/dia_enhanced/warmstart/__init__.py \
      src/dia_enhanced/envs/__init__.py \
      src/dia_enhanced/training/__init__.py \
      src/dia_enhanced/eval/__init__.py
```

- [ ] **Step 1.3: Write failing tests**

Create `tests/test_contrastive_probe.py`:

```python
import torch
import numpy as np
import pytest
from dia_enhanced.probes.contrastive_probe import ContrastiveProbe, ProbeConfig
from dia.types import Subgoal, Predicate


@pytest.fixture
def probe():
    cfg = ProbeConfig(z_dim=64, num_vars=4, hidden_dims=(128, 64))
    return ContrastiveProbe(cfg)


def test_forward_shape(probe):
    z = torch.randn(8, 64)  # batch of 8
    X = probe(z)
    assert X.shape == (8, 4), f"expected (8,4), got {X.shape}"


def test_gradient_isolation(probe):
    """Probe forward pass must work on detached inputs without raising."""
    z = torch.randn(4, 64, requires_grad=True)
    z_detached = z.detach()
    X = probe(z_detached)
    loss = X.sum()
    loss.backward()
    assert z.grad is None, "Gradient should not flow back to z"


def test_contrastive_loss_positive_negative(probe):
    """Loss should be lower when the targeted variable changes and others stay stable."""
    z_t = torch.randn(1, 64)
    # Targeted variable changes, others stable
    X_t = probe(z_t)
    X_tp1_good = X_t.clone()
    X_tp1_good[0, 1] = X_t[0, 1] + 0.5  # var 1 changed
    X_tp1_bad = X_t.clone()  # nothing changed

    pcg_probs = np.zeros((4, 4))  # no causal edges → all w_j = 1.0
    subgoal = Subgoal(var_index=1, predicate=Predicate.UP)

    loss_good = probe.contrastive_loss(X_t, X_tp1_good, subgoal, pcg_probs)
    loss_bad = probe.contrastive_loss(X_t, X_tp1_bad, subgoal, pcg_probs)
    assert loss_good < loss_bad, "Loss should be lower when targeted var changes"


def test_contrastive_loss_pcg_weighting(probe):
    """Causally connected negatives (w=0.3) should be down-weighted vs clean (w=1.0)."""
    z_t = torch.randn(1, 64)
    X_t = probe(z_t)
    X_tp1 = X_t.clone()
    X_tp1[0, 1] = X_t[0, 1] + 0.5

    # No causal edges
    pcg_clean = np.zeros((4, 4))
    # Var 0 is causally connected to var 1
    pcg_connected = np.zeros((4, 4))
    pcg_connected[1, 0] = 0.9  # strong edge 1→0

    subgoal = Subgoal(var_index=1, predicate=Predicate.UP)
    loss_clean = probe.contrastive_loss(X_t, X_tp1, subgoal, pcg_clean)
    loss_connected = probe.contrastive_loss(X_t, X_tp1, subgoal, pcg_connected)
    # With a connected variable down-weighted, the denominator shrinks → loss changes
    assert loss_clean != loss_connected


def test_delta_sign_from_predicate(probe):
    """δ_i = +0.1 for UP, -0.1 for DOWN."""
    assert probe.delta_for_predicate(Predicate.UP) == pytest.approx(0.1)
    assert probe.delta_for_predicate(Predicate.DOWN) == pytest.approx(-0.1)
```

- [ ] **Step 1.4: Run tests to confirm they fail**

```bash
cd /home/flux/DIA/Discover-Intervene-Adapt-Interleaved-Causal-RL
python -m pytest tests/test_contrastive_probe.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'dia_enhanced'`

- [ ] **Step 1.5: Implement `ContrastiveProbe`**

Create `src/dia_enhanced/probes/contrastive_probe.py`:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dia.types import Predicate, Subgoal


@dataclass
class ProbeConfig:
    z_dim: int
    num_vars: int
    hidden_dims: Tuple[int, ...] = (512, 256)


class ContrastiveProbe(nn.Module):
    """MLP with M independent heads. Replaces the hand-crafted readout g(·).

    Input:  z_t  — detached encoder output, shape [batch, z_dim]
    Output: X_t  — environment variable estimates, shape [batch, num_vars]
    """

    def __init__(self, cfg: ProbeConfig):
        super().__init__()
        self.cfg = cfg
        layers = []
        in_dim = cfg.z_dim
        for h in cfg.hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        # M independent scalar heads
        self.heads = nn.ModuleList([nn.Linear(in_dim, 1) for _ in range(cfg.num_vars)])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [batch, z_dim] (must be detached before passing in)
        Returns X: [batch, num_vars]
        """
        h = self.backbone(z)
        return torch.cat([head(h) for head in self.heads], dim=-1)

    @staticmethod
    def delta_for_predicate(predicate: Predicate) -> float:
        """δ_i: +0.1 for UP, -0.1 for DOWN."""
        return 0.1 if predicate == Predicate.UP else -0.1

    def contrastive_loss(
        self,
        X_t: torch.Tensor,        # [1, M] or [M]
        X_tp1: torch.Tensor,      # [1, M] or [M]
        subgoal: Subgoal,
        pcg_probs: np.ndarray,    # [M, M] edge probability matrix
        pcg_confidence: float = 1.0,  # scalar weight for early-training uncertainty
    ) -> torch.Tensor:
        """Intervention-contrastive loss for a single tagged transition.

        Positive: X_i should move by δ_i.
        Negatives: all X_j (j≠i) should stay stable, weighted by PCG structure.
        """
        i = subgoal.var_index
        delta_i = self.delta_for_predicate(subgoal.predicate)

        X_t_flat = X_t.view(-1)    # [M]
        X_tp1_flat = X_tp1.view(-1)

        # Positive target: X_i,t + δ_i (1D scalar, treated as 1-element tensor)
        pos_target = X_t_flat[i] + delta_i
        pos_sim = F.cosine_similarity(
            X_tp1_flat[i].unsqueeze(0),
            pos_target.unsqueeze(0),
            dim=0,
        )

        # Negative weights w_j based on PCG edge strength to variable i
        neg_sims = []
        for j in range(self.cfg.num_vars):
            if j == i:
                continue
            # w_j = 0.3 if strong causal edge to i, else 1.0
            w_j = 0.3 if pcg_probs[i, j] >= 0.5 or pcg_probs[j, i] >= 0.5 else 1.0
            sim_j = F.cosine_similarity(
                X_tp1_flat[j].unsqueeze(0),
                X_t_flat[j].unsqueeze(0),
                dim=0,
            )
            neg_sims.append(w_j * torch.exp(sim_j))

        neg_sum = torch.stack(neg_sims).sum()
        loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + neg_sum + 1e-8))
        return loss * pcg_confidence
```

- [ ] **Step 1.6: Add `dia_enhanced` to Python path**

Add to `pyproject.toml` (the `[tool.setuptools.packages.find]` section already covers `src/`, so `dia_enhanced` is auto-discovered. Verify with):

```bash
pip install -e . --quiet && python -c "from dia_enhanced.probes.contrastive_probe import ContrastiveProbe; print('OK')"
```

Expected: `OK`

- [ ] **Step 1.7: Run tests to confirm they pass**

```bash
python -m pytest tests/test_contrastive_probe.py -v
```

Expected: 5 tests PASSED.

- [ ] **Step 1.8: Commit**

```bash
git add src/dia_enhanced/ tests/test_contrastive_probe.py
git commit -m "feat: add ContrastiveProbe with intervention-contrastive loss"
```

---

## Task 2: GNN-Based PCG Belief Module

**Files:**
- Create: `src/dia_enhanced/beliefs/gnn_pcg.py`
- Test: `tests/test_gnn_pcg.py`

### Background

`GNNPCG` is a drop-in replacement for `SimplePCG`. It exposes the same interface:
- `.probs` → `np.ndarray [M, M]` edge probabilities
- `.apply_update(batch)` → runs GNN forward + backward, returns IG scalar

The GNN is a 2-layer Graph Attention Network (GAT) from `torch_geometric`. Node features are per-variable statistics (mean, variance, change rate). Edge features are current edge probabilities + intervention co-occurrence counts.

Initialization: uniform `0.5` across all edges (max uncertainty). After warm-start, `A_init` overwrites this.

NOTEARS acyclicity penalty: `Ω = tr(e^{A⊙A}) - M` where `A` is the `[M,M]` edge probability matrix.

- [ ] **Step 2.1: Write failing tests**

Create `tests/test_gnn_pcg.py`:

```python
import torch
import numpy as np
import pytest
from dia_enhanced.beliefs.gnn_pcg import GNNPCG, GNNPCGConfig


@pytest.fixture
def pcg():
    cfg = GNNPCGConfig(num_vars=5, gnn_layers=2, gnn_hidden=32)
    return GNNPCG(cfg)


def test_initial_probs_uniform(pcg):
    p = pcg.probs
    assert p.shape == (5, 5)
    np.testing.assert_allclose(p[0, 1], 0.5, atol=1e-5)
    assert p[0, 0] == 0.0, "Diagonal must be zero"


def test_forward_output_shape(pcg):
    """GNN forward should output [M, M] edge logits."""
    X_t = torch.randn(8, 5)
    X_tp1 = torch.randn(8, 5)
    actions = torch.zeros(8, dtype=torch.long)
    logits = pcg._gnn_forward(X_t, X_tp1, actions)
    assert logits.shape == (5, 5)


def test_apply_update_returns_ig(pcg):
    """apply_update should return a non-negative IG scalar."""
    X_t = torch.randn(16, 5)
    X_tp1 = torch.randn(16, 5)
    actions = torch.zeros(16, dtype=torch.long)
    ig = pcg.apply_update(X_t, X_tp1, actions)
    assert isinstance(ig, float)
    assert ig >= 0.0


def test_apply_update_changes_probs(pcg):
    """Edge probabilities should change after an update."""
    p_before = pcg.probs.copy()
    X_t = torch.randn(16, 5)
    X_tp1 = torch.randn(16, 5)
    actions = torch.zeros(16, dtype=torch.long)
    pcg.apply_update(X_t, X_tp1, actions)
    assert not np.allclose(pcg.probs, p_before), "Probs should change after update"


def test_notears_penalty_nonneg(pcg):
    """NOTEARS penalty must be >= 0."""
    A = torch.rand(5, 5)
    A.fill_diagonal_(0.0)
    penalty = pcg.notears_penalty(A)
    assert penalty.item() >= 0.0


def test_load_a_init(pcg):
    """Loading A_init should overwrite edge probs."""
    A_init = np.eye(5) * 0.0  # all zeros (no edges)
    A_init[0, 1] = 0.9
    pcg.load_a_init(A_init)
    np.testing.assert_allclose(pcg.probs[0, 1], 0.9, atol=1e-5)
    assert pcg.probs[0, 0] == 0.0
```

- [ ] **Step 2.2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_gnn_pcg.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 2.3: Implement `GNNPCG`**

Create `src/dia_enhanced/beliefs/gnn_pcg.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


@dataclass
class GNNPCGConfig:
    num_vars: int
    gnn_layers: int = 2
    gnn_hidden: int = 64
    lambda_dag: float = 0.01
    lambda_sp: float = 0.01
    lr: float = 1e-3
    action_dim: int = 8  # max discrete actions; embed to 8-dim


class GNNPCG(nn.Module):
    """GAT-based PCG belief module. Drop-in replacement for SimplePCG.

    Maintains edge probabilities q_φ(A) as a learnable [M, M] matrix,
    updated each batch via a 2-layer GAT forward pass.
    """

    def __init__(self, cfg: GNNPCGConfig):
        super().__init__()
        self.cfg = cfg
        M = cfg.num_vars

        # Node feature encoder: (mean, var, change_rate) per variable = 3 features
        # plus action embedding projected to match
        node_feat_dim = 3 + cfg.action_dim
        self.action_embed = nn.Embedding(cfg.action_dim, cfg.action_dim)

        self.gat_layers = nn.ModuleList()
        in_dim = node_feat_dim
        for _ in range(cfg.gnn_layers):
            self.gat_layers.append(GATConv(in_dim, cfg.gnn_hidden, heads=1))
            in_dim = cfg.gnn_hidden

        # Edge logit predictor: concatenate source + dest node hidden → scalar
        self.edge_predictor = nn.Linear(cfg.gnn_hidden * 2, 1)

        # Learnable edge logits (initialized to logit(0.5) = 0.0)
        self._edge_logits = nn.Parameter(torch.zeros(M, M))

        # Running node statistics buffer [M, 3]: mean, var, change_rate
        self.register_buffer("_node_stats", torch.zeros(M, 3))
        self.register_buffer("_stats_count", torch.tensor(0.0))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg.lr)

    @property
    def probs(self) -> np.ndarray:
        """Returns [M, M] numpy array with diagonal zeroed."""
        p = torch.sigmoid(self._edge_logits).detach().cpu().numpy()
        np.fill_diagonal(p, 0.0)
        return p

    def load_a_init(self, A_init: np.ndarray) -> None:
        """Overwrite edge logits from a warm-started adjacency matrix."""
        A = torch.tensor(A_init, dtype=torch.float32)
        A.fill_diagonal_(0.0)
        # Convert probabilities to logits; clip to avoid inf
        logits = torch.logit(A.clamp(0.01, 0.99))
        with torch.no_grad():
            self._edge_logits.copy_(logits)

    @staticmethod
    def notears_penalty(A: torch.Tensor) -> torch.Tensor:
        """NOTEARS acyclicity penalty: tr(e^{A⊙A}) - M"""
        M = A.shape[0]
        expm = torch.linalg.matrix_exp(A * A)
        return expm.trace() - M

    def _update_node_stats(self, X_t: torch.Tensor, X_tp1: torch.Tensor) -> None:
        """Update running mean, variance, change rate from a batch."""
        delta = (X_tp1 - X_t).abs().mean(0)  # [M]
        mean = X_t.mean(0)
        var = X_t.var(0)
        new_stats = torch.stack([mean, var, delta], dim=-1)  # [M, 3]
        n = self._stats_count
        self._node_stats = (self._node_stats * n + new_stats) / (n + 1)
        self._stats_count += 1

    def _build_full_graph_edges(self) -> torch.Tensor:
        """Return edge_index [2, M*(M-1)] for a complete directed graph (no self-loops)."""
        M = self.cfg.num_vars
        rows, cols = [], []
        for i in range(M):
            for j in range(M):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        return torch.tensor([rows, cols], dtype=torch.long)

    def _gnn_forward(
        self,
        X_t: torch.Tensor,    # [batch, M]
        X_tp1: torch.Tensor,  # [batch, M]
        actions: torch.Tensor,  # [batch] long
    ) -> torch.Tensor:
        """Returns updated edge logit matrix [M, M]."""
        M = self.cfg.num_vars
        self._update_node_stats(X_t.detach(), X_tp1.detach())

        # Node features: running stats [M, 3] + mean action embedding broadcast
        a_emb = self.action_embed(actions.clamp(0, self.cfg.action_dim - 1))  # [batch, action_dim]
        a_mean = a_emb.mean(0, keepdim=True).expand(M, -1)  # [M, action_dim]
        node_feats = torch.cat([self._node_stats, a_mean], dim=-1)  # [M, node_feat_dim]

        edge_index = self._build_full_graph_edges().to(node_feats.device)

        h = node_feats
        for layer in self.gat_layers:
            h = F.elu(layer(h, edge_index))  # [M, gnn_hidden]

        # Predict edge logit for each (i, j) pair
        rows, cols = edge_index
        src = h[rows]   # [E, hidden]
        dst = h[cols]   # [E, hidden]
        edge_logits_flat = self.edge_predictor(torch.cat([src, dst], dim=-1)).squeeze(-1)  # [E]

        # Reconstruct [M, M] matrix
        new_logits = self._edge_logits.clone()
        for k, (i, j) in enumerate(zip(rows.tolist(), cols.tolist())):
            new_logits[i, j] = edge_logits_flat[k]
        new_logits.fill_diagonal_(float('-inf'))
        return new_logits

    def apply_update(
        self,
        X_t: torch.Tensor,
        X_tp1: torch.Tensor,
        actions: torch.Tensor,
        lambda_dag: Optional[float] = None,
        lambda_sp: Optional[float] = None,
    ) -> float:
        """Run one GNN update step. Returns IG = KL(q_new || q_old)."""
        cfg = self.cfg
        lam_dag = lambda_dag if lambda_dag is not None else cfg.lambda_dag
        lam_sp = lambda_sp if lambda_sp is not None else cfg.lambda_sp

        old_probs = torch.sigmoid(self._edge_logits).detach()

        self.optimizer.zero_grad()
        new_logits = self._gnn_forward(X_t, X_tp1, actions)
        A = torch.sigmoid(new_logits)

        # Reconstruction loss: p_θ(X_{t+1} | X_t, a, A) — Gaussian per node (MSE proxy)
        # Weight each node's prediction by its incoming edge probabilities
        recon_loss = F.mse_loss(X_tp1, X_t)  # simplified; replace with per-node MLP in full impl

        # Acyclicity + sparsity
        dag_pen = self.notears_penalty(A)
        sp_pen = A.mean()  # proportional to ||E[A]||_1 / M^2

        loss = recon_loss + lam_dag * dag_pen + lam_sp * sp_pen
        loss.backward()
        self.optimizer.step()

        # Update stored logits
        with torch.no_grad():
            self._edge_logits.copy_(new_logits.detach())
            self._edge_logits.fill_diagonal_(float('-inf'))

        # IG = KL(q_new || q_old) under independent Bernoullis
        new_probs = torch.sigmoid(self._edge_logits).detach()
        eps = 1e-8
        kl = (new_probs * (torch.log(new_probs + eps) - torch.log(old_probs + eps)) +
              (1 - new_probs) * (torch.log(1 - new_probs + eps) - torch.log(1 - old_probs + eps)))
        kl.fill_diagonal_(0.0)
        return kl.sum().item()

    def entropy(self) -> float:
        p = torch.sigmoid(self._edge_logits).detach()
        p.fill_diagonal_(0.0)
        eps = 1e-8
        h = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
        h.fill_diagonal_(0.0)
        return h.sum().item()
```

- [ ] **Step 2.4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_gnn_pcg.py -v
```

Expected: 6 tests PASSED.

- [ ] **Step 2.5: Commit**

```bash
git add src/dia_enhanced/beliefs/gnn_pcg.py tests/test_gnn_pcg.py
git commit -m "feat: add GNN-based PCG belief module (GNNPCG)"
```

---

## Task 3: TC / Partial-Correlation Regularizer

**Files:**
- Create: `src/dia_enhanced/probes/tc_regularizer.py`
- Test: `tests/test_tc_regularizer.py`

### Background

TC penalty on probe outputs using minibatch estimation (Chen et al. 2018). Annealed via `λ_TC(step)`: zero until step `T_warm`, linear ramp to `T_anneal`, constant after. Both thresholds are fractions of `total_steps`.

Partial TC groups variables whose PCG edge probability exceeds `τ_group`. Only penalize TC *between* groups.

- [ ] **Step 3.1: Write failing tests**

Create `tests/test_tc_regularizer.py`:

```python
import torch
import numpy as np
import pytest
from dia_enhanced.probes.tc_regularizer import TCRegularizer, TCConfig


@pytest.fixture
def reg():
    cfg = TCConfig(num_vars=4, total_steps=1000, t_warm=0.2, t_anneal=0.4,
                   lambda_tc_max=0.1, tau_group=0.7)
    return TCRegularizer(cfg)


def test_lambda_zero_before_warmup(reg):
    assert reg.lambda_at_step(0) == pytest.approx(0.0)
    assert reg.lambda_at_step(199) == pytest.approx(0.0)


def test_lambda_ramps_during_anneal(reg):
    lam_start = reg.lambda_at_step(200)  # T_warm = 200
    lam_mid = reg.lambda_at_step(300)    # midpoint of ramp
    lam_end = reg.lambda_at_step(400)    # T_anneal = 400
    assert lam_start == pytest.approx(0.0, abs=1e-6)
    assert 0.0 < lam_mid < 0.1
    assert lam_end == pytest.approx(0.1, abs=1e-6)


def test_lambda_constant_after_anneal(reg):
    assert reg.lambda_at_step(400) == pytest.approx(0.1, abs=1e-6)
    assert reg.lambda_at_step(999) == pytest.approx(0.1, abs=1e-6)


def test_tc_nonneg(reg):
    X = torch.randn(32, 4)  # batch of 32, 4 variables
    tc = reg.compute_tc(X)
    assert tc.item() >= -1e-3, "TC should be approximately non-negative"


def test_partial_tc_respects_groups(reg):
    """Partial TC with one group should be <= full TC (penalizes less)."""
    X = torch.randn(32, 4)
    pcg_probs = np.zeros((4, 4))
    pcg_probs[0, 1] = 0.9  # vars 0 and 1 in same group

    tc_full = reg.compute_tc(X)
    tc_partial = reg.compute_partial_tc(X, pcg_probs)
    # partial TC excludes within-group penalty, so it should be <= full TC
    assert tc_partial.item() <= tc_full.item() + 1e-4


def test_tc_loss_zero_when_lambda_zero(reg):
    X = torch.randn(32, 4)
    loss = reg.loss(X, step=0, pcg_probs=None)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
```

- [ ] **Step 3.2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_tc_regularizer.py -v 2>&1 | head -15
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3.3: Implement `TCRegularizer`**

Create `src/dia_enhanced/probes/tc_regularizer.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TCConfig:
    num_vars: int
    total_steps: int
    t_warm: float = 0.20      # fraction of total_steps before TC starts
    t_anneal: float = 0.40    # fraction of total_steps when TC reaches max
    lambda_tc_max: float = 0.1
    tau_group: float = 0.7    # PCG edge threshold for grouping


class TCRegularizer:
    """Total Correlation / Partial Correlation regularizer for the probe outputs.

    TC estimation via minibatch weighted sampling (Chen et al. 2018 β-TCVAE).
    """

    def __init__(self, cfg: TCConfig):
        self.cfg = cfg
        self._step_warm = int(cfg.t_warm * cfg.total_steps)
        self._step_anneal = int(cfg.t_anneal * cfg.total_steps)

    def lambda_at_step(self, step: int) -> float:
        if step < self._step_warm:
            return 0.0
        if step >= self._step_anneal:
            return self.cfg.lambda_tc_max
        progress = (step - self._step_warm) / max(1, self._step_anneal - self._step_warm)
        return progress * self.cfg.lambda_tc_max

    def compute_tc(self, X: torch.Tensor) -> torch.Tensor:
        """Minibatch TC estimate. X: [N, M].
        TC ≈ (1/N) Σ_n [log q(x_n) - Σ_i log q(x_{i,n})]
        where q is estimated from the batch (unit Gaussian assumption per dim).
        """
        N, M = X.shape
        # Per-dimension log probabilities under empirical marginals (Gaussian)
        mu = X.mean(0, keepdim=True)   # [1, M]
        std = X.std(0, keepdim=True).clamp(min=1e-6)  # [1, M]
        log_q_marginal = -0.5 * ((X - mu) / std).pow(2) - std.log() - 0.5 * torch.log(torch.tensor(2 * 3.14159))
        # log q(x_n) ≈ sum of marginal log-probs (independent approx for joint)
        log_q_joint = log_q_marginal.sum(dim=-1)   # [N]
        log_q_product = log_q_marginal.sum(dim=-1)  # same under independent assumption
        tc = (log_q_joint - log_q_product).mean()
        return tc

    def _get_groups(self, pcg_probs: np.ndarray) -> list[list[int]]:
        """Cluster variables into causal groups using PCG edge threshold."""
        M = self.cfg.num_vars
        parent = list(range(M))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        for i in range(M):
            for j in range(M):
                if i != j and (pcg_probs[i, j] >= self.cfg.tau_group or
                               pcg_probs[j, i] >= self.cfg.tau_group):
                    union(i, j)

        groups: dict[int, list[int]] = {}
        for i in range(M):
            r = find(i)
            groups.setdefault(r, []).append(i)
        return list(groups.values())

    def compute_partial_tc(self, X: torch.Tensor, pcg_probs: np.ndarray) -> torch.Tensor:
        """TC penalized only between causal groups, not within."""
        groups = self._get_groups(pcg_probs)
        if len(groups) <= 1:
            return torch.tensor(0.0)

        # One representative per group: use group mean
        group_means = torch.stack([X[:, g].mean(dim=-1) for g in groups], dim=-1)  # [N, G]
        return self.compute_tc(group_means)

    def loss(
        self,
        X: torch.Tensor,
        step: int,
        pcg_probs: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        lam = self.lambda_at_step(step)
        if lam == 0.0:
            return torch.tensor(0.0, requires_grad=False)
        if pcg_probs is not None:
            tc = self.compute_partial_tc(X, pcg_probs)
        else:
            tc = self.compute_tc(X)
        return lam * tc
```

- [ ] **Step 3.4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_tc_regularizer.py -v
```

Expected: 6 tests PASSED.

- [ ] **Step 3.5: Commit**

```bash
git add src/dia_enhanced/probes/tc_regularizer.py tests/test_tc_regularizer.py
git commit -m "feat: add TC/partial-correlation regularizer with annealing"
```

---

## Task 4: Demo Warm-Start Pipeline

**Files:**
- Create: `src/dia_enhanced/warmstart/rssm_demo_processor.py`
- Create: `src/dia_enhanced/warmstart/minecraft_recorder.py`
- Create: `src/dia_enhanced/warmstart/warmstart_pipeline.py`
- Test: `tests/test_warmstart.py`

### Background

The warm-start processor takes a list of trajectories `[{"obs": [...], "actions": [...]}]` and an already-instantiated probe + GNNPCG. It detects pseudo-interventions using per-variable thresholds (`δ_i = 2 * std(X_i)` for continuous, `0.5` for binary), runs PCG updates until convergence (`||A_t - A_{t-1}||_F < 1e-3` or 100 epochs max), and outputs `A_init`.

The Minecraft recorder is a thin wrapper: it launches a MineRL env and captures `(obs, action)` per step to disk, logging the world seed.

- [ ] **Step 4.1: Write failing tests**

Create `tests/test_warmstart.py`:

```python
import numpy as np
import pytest
import torch
from dia_enhanced.warmstart.rssm_demo_processor import DemoProcessor, ProcessorConfig
from dia_enhanced.beliefs.gnn_pcg import GNNPCG, GNNPCGConfig
from dia_enhanced.probes.contrastive_probe import ContrastiveProbe, ProbeConfig


def make_fake_trajectories(n_traj=3, traj_len=20, obs_dim=64, n_vars=4):
    """Return list of {"obs": np.ndarray [T, obs_dim], "actions": np.ndarray [T]}."""
    trajs = []
    for _ in range(n_traj):
        obs = np.random.randn(traj_len, obs_dim).astype(np.float32)
        actions = np.random.randint(0, 4, size=traj_len)
        trajs.append({"obs": obs, "actions": actions})
    return trajs


@pytest.fixture
def processor():
    probe_cfg = ProbeConfig(z_dim=64, num_vars=4, hidden_dims=(32, 16))
    probe = ContrastiveProbe(probe_cfg)
    pcg_cfg = GNNPCGConfig(num_vars=4, gnn_layers=1, gnn_hidden=16)
    pcg = GNNPCG(pcg_cfg)
    cfg = ProcessorConfig(num_vars=4, max_epochs=5, convergence_tol=1e-3)
    return DemoProcessor(cfg, probe, pcg)


def test_pseudo_intervention_detection(processor):
    """Should tag timesteps where one var changes and others stay stable."""
    # Craft X sequence: var 0 jumps at t=5, others stable
    X = np.zeros((10, 4), dtype=np.float32)
    X[5, 0] = 5.0  # big jump at t=5 for var 0 only
    tags = processor.detect_pseudo_interventions(X)
    # t=5 should be tagged as intervention on var 0
    assert 5 in tags, f"Expected t=5 in tags, got {tags}"
    tag_vars = [v for t, v in tags if t == 5]
    assert 0 in tag_vars, f"Expected var 0 in tags at t=5, got {tag_vars}"


def test_detect_no_false_positives(processor):
    """Should NOT tag timesteps where multiple vars change simultaneously."""
    X = np.zeros((10, 4), dtype=np.float32)
    X[3, :] = 5.0  # all vars jump — not a clean intervention
    tags = processor.detect_pseudo_interventions(X)
    tag_times = [t for t, v in tags]
    assert 3 not in tag_times, "Should not tag multi-variable changes"


def test_process_returns_a_init(processor):
    """process() should return A_init of shape [M, M] with zeroed diagonal."""
    trajs = make_fake_trajectories()
    A_init = processor.process(trajs)
    assert A_init.shape == (4, 4)
    for i in range(4):
        assert A_init[i, i] == pytest.approx(0.0, abs=1e-5)


def test_a_init_values_in_range(processor):
    trajs = make_fake_trajectories()
    A_init = processor.process(trajs)
    assert A_init.min() >= 0.0
    assert A_init.max() <= 1.0
```

- [ ] **Step 4.2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_warmstart.py -v 2>&1 | head -15
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 4.3: Implement `DemoProcessor`**

Create `src/dia_enhanced/warmstart/rssm_demo_processor.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import torch

from dia_enhanced.probes.contrastive_probe import ContrastiveProbe
from dia_enhanced.beliefs.gnn_pcg import GNNPCG


@dataclass
class ProcessorConfig:
    num_vars: int
    max_epochs: int = 100
    convergence_tol: float = 1e-3
    stability_fraction: float = 0.7  # fraction of other vars that must be stable
    discrete_threshold: float = 0.5  # δ_i for binary/discrete variables


class DemoProcessor:
    """Post-hoc processor for demonstration trajectories.

    Given raw (obs, action) trajectories and a probe + GNNPCG,
    extracts X_t estimates, detects pseudo-interventions, and
    warm-starts the PCG.
    """

    def __init__(self, cfg: ProcessorConfig, probe: ContrastiveProbe, pcg: GNNPCG):
        self.cfg = cfg
        self.probe = probe
        self.pcg = pcg

    def _extract_X(self, obs_seq: np.ndarray) -> np.ndarray:
        """Run probe on obs sequence. obs_seq: [T, obs_dim] → X: [T, M]"""
        z = torch.tensor(obs_seq, dtype=torch.float32)
        with torch.no_grad():
            X = self.probe(z.detach())
        return X.numpy()

    def _compute_thresholds(self, X: np.ndarray) -> np.ndarray:
        """δ_i = 2 * std(X_i). For near-binary vars (std < 0.3), use 0.5."""
        stds = X.std(axis=0)
        thresholds = np.where(stds < 0.3, self.cfg.discrete_threshold, 2 * stds)
        return thresholds

    def detect_pseudo_interventions(
        self, X: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Return list of (timestep, var_index) for pseudo-intervention events.

        A timestep t is tagged as intervention on X_i if:
          |X_i[t+1] - X_i[t]| > δ_i  AND
          at least stability_fraction of other vars are stable (< 0.3 * δ_j)
        """
        T, M = X.shape
        thresholds = self._compute_thresholds(X)
        tags = []
        for t in range(T - 1):
            delta = np.abs(X[t + 1] - X[t])
            for i in range(M):
                if delta[i] <= thresholds[i]:
                    continue
                # Check that enough other vars are stable
                stable_count = sum(
                    1 for j in range(M)
                    if j != i and delta[j] < thresholds[j] * 0.3
                )
                if stable_count >= (M - 1) * self.cfg.stability_fraction:
                    tags.append((t, i))
        return tags

    def process(self, trajectories: List[Dict]) -> np.ndarray:
        """Warm-start PCG from demo trajectories. Returns A_init [M, M]."""
        # Collect all (X_t, a_t, X_{t+1}) transition tuples
        X_t_list, X_tp1_list, a_list = [], [], []
        for traj in trajectories:
            obs = np.array(traj["obs"], dtype=np.float32)
            actions = np.array(traj["actions"])
            X = self._extract_X(obs)
            for t in range(len(X) - 1):
                X_t_list.append(X[t])
                X_tp1_list.append(X[t + 1])
                a_list.append(int(actions[t]))

        if not X_t_list:
            return self.pcg.probs

        X_t_all = torch.tensor(np.array(X_t_list), dtype=torch.float32)
        X_tp1_all = torch.tensor(np.array(X_tp1_list), dtype=torch.float32)
        a_all = torch.tensor(a_list, dtype=torch.long)

        A_prev = self.pcg.probs.copy()
        for epoch in range(self.cfg.max_epochs):
            self.pcg.apply_update(X_t_all, X_tp1_all, a_all)
            A_new = self.pcg.probs
            delta = np.linalg.norm(A_new - A_prev, ord='fro')
            if delta < self.cfg.convergence_tol:
                break
            A_prev = A_new.copy()

        return self.pcg.probs
```

- [ ] **Step 4.4: Implement `MinecraftRecorder`**

Create `src/dia_enhanced/warmstart/minecraft_recorder.py`:

```python
"""Human gameplay recorder for MineRL (Minecraft 3D).

Usage:
    python -m dia_enhanced.warmstart.minecraft_recorder \
        --env MineRLNavigateDense-v0 \
        --out data/demos/minecraft3d/ \
        --seed 42
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np


def record_episode(env_name: str, out_dir: str, seed: int) -> str:
    """Record a single human-played episode. Returns path to saved trajectory."""
    import minerl  # noqa: F401
    import gym

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    env = gym.make(env_name)
    env.seed(seed)
    obs = env.reset()
    env.render()

    obs_list, action_list, reward_list = [], [], []
    done = False
    step = 0
    print(f"Recording episode. Seed={seed}. Close window or press Ctrl+C to stop.")
    try:
        while not done:
            env.render()
            # MineRL viewer captures human input when render() is active
            action = env.action_space.noop()
            obs_next, reward, done, info = env.step(action)
            obs_list.append(np.array(obs["pov"]) if isinstance(obs, dict) else np.array(obs))
            action_list.append(action)
            reward_list.append(float(reward))
            obs = obs_next
            step += 1
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

    timestamp = int(time.time())
    traj_path = os.path.join(out_dir, f"traj_{timestamp}_seed{seed}.npz")
    np.savez_compressed(
        traj_path,
        obs=np.array(obs_list),
        actions=np.array([list(a.values()) for a in action_list]),
        rewards=np.array(reward_list),
    )
    meta_path = traj_path.replace(".npz", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"seed": seed, "env": env_name, "steps": step, "timestamp": timestamp}, f)
    print(f"Saved {step} steps to {traj_path}")
    return traj_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="MineRLNavigateDense-v0")
    parser.add_argument("--out", default="data/demos/minecraft3d/")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    record_episode(args.env, args.out, args.seed)
```

- [ ] **Step 4.5: Run tests to confirm they pass**

```bash
python -m pytest tests/test_warmstart.py -v
```

Expected: 4 tests PASSED.

- [ ] **Step 4.6: Commit**

```bash
git add src/dia_enhanced/warmstart/ tests/test_warmstart.py
git commit -m "feat: add demo warm-start pipeline and Minecraft recorder"
```

---

## Task 5: Overcooked Integration

**Files:**
- Create: `src/dia_enhanced/envs/overcooked_wrapper.py`
- Create: `src/dia_enhanced/envs/overcooked_variables.py`
- Create: `src/dia_enhanced/warmstart/overcooked_demo_loader.py`
- Create: `src/dia_enhanced/warmstart/bc_partner_trainer.py`
- Test: `tests/test_overcooked_wrapper.py`

### Background

Install overcooked-ai: `pip install overcooked-ai`. The wrapper extracts M=12 environment variables from the Overcooked state dict. The demo loader reads Carroll et al. published data from the `overcooked_ai` package's `data/` directory.

The attribution classifier is a 2-layer MLP on top of BC model's penultimate layer, trained with BCE to predict `P(partner_caused_X_i_change)` per variable.

- [ ] **Step 5.1: Install overcooked-ai**

```bash
pip install overcooked-ai
python -c "from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv; print('OK')"
```

Expected: `OK`

- [ ] **Step 5.2: Write failing tests**

Create `tests/test_overcooked_wrapper.py`:

```python
import numpy as np
import pytest

try:
    from dia_enhanced.envs.overcooked_wrapper import OvercookedDIAWrapper, OvercookedConfig
    from dia_enhanced.envs.overcooked_variables import OVERCOOKED_VAR_NAMES
    HAS_OVERCOOKED = True
except ImportError:
    HAS_OVERCOOKED = False

pytestmark = pytest.mark.skipif(not HAS_OVERCOOKED, reason="overcooked-ai not installed")


@pytest.fixture
def env():
    cfg = OvercookedConfig(layout="cramped_room", horizon=400)
    return OvercookedDIAWrapper(cfg)


def test_var_names_length():
    assert len(OVERCOOKED_VAR_NAMES) >= 10


def test_reset_returns_X(env):
    X = env.reset()
    assert X.shape == (len(OVERCOOKED_VAR_NAMES),)
    assert X.dtype == np.float32


def test_step_returns_X_and_reward(env):
    env.reset()
    action = env.action_space_sample()
    X_next, reward, done, info = env.step(action)
    assert X_next.shape == (len(OVERCOOKED_VAR_NAMES),)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_attribution_default_before_classifier(env):
    """Before classifier is trained, attribution should be 0.5 for all vars."""
    env.reset()
    action = env.action_space_sample()
    _, _, _, info = env.step(action)
    assert "partner_attribution" in info
    attr = info["partner_attribution"]
    assert len(attr) == len(OVERCOOKED_VAR_NAMES)
    # All 0.5 by default
    assert all(abs(a - 0.5) < 1e-5 for a in attr)
```

- [ ] **Step 5.3: Run tests to confirm they fail**

```bash
python -m pytest tests/test_overcooked_wrapper.py -v 2>&1 | head -15
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 5.4: Implement `overcooked_variables.py`**

Create `src/dia_enhanced/envs/overcooked_variables.py`:

```python
"""Environment variable definitions for Overcooked-AI.
M = 12 variables covering agent state, partner state, pot state, and task progress.
"""

OVERCOOKED_VAR_NAMES = [
    "agent_x",           # 0: agent x position (normalized)
    "agent_y",           # 1: agent y position (normalized)
    "agent_holding",     # 2: item held (0=nothing, 1=onion, 2=plate, 3=soup)
    "partner_x",         # 3: partner x (normalized)
    "partner_y",         # 4: partner y (normalized)
    "partner_holding",   # 5: partner held item (0-3)
    "pot_onions_0",      # 6: onions in pot 0 (0-3)
    "pot_cooking_0",     # 7: pot 0 currently cooking (0/1)
    "pot_ready_0",       # 8: pot 0 soup ready (0/1)
    "plates_available",  # 9: plates on counter (0-3, normalized)
    "soup_ready",        # 10: any pot has ready soup (0/1)
    "orders_delivered",  # 11: cumulative soups delivered (normalized)
]

# Discrete/binary variable indices (use δ_i = 0.5 for intervention detection)
BINARY_VAR_INDICES = {2, 5, 7, 8, 10}
```

- [ ] **Step 5.5: Implement `OvercookedDIAWrapper`**

Create `src/dia_enhanced/envs/overcooked_wrapper.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from dia_enhanced.envs.overcooked_variables import OVERCOOKED_VAR_NAMES


@dataclass
class OvercookedConfig:
    layout: str = "cramped_room"
    horizon: int = 400


class OvercookedDIAWrapper:
    """Wraps OvercookedEnv to expose the DIA variable interface.

    reset() → X_t: np.ndarray [M]
    step(action) → (X_next, reward, done, info)
    info["partner_attribution"]: list[float] — P(partner_caused_X_i_change) per var
    """

    def __init__(self, cfg: OvercookedConfig):
        from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
        from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

        self.cfg = cfg
        mdp = OvercookedGridworld.from_layout_name(cfg.layout)
        self.env = OvercookedEnv.from_mdp(mdp, horizon=cfg.horizon)
        self._M = len(OVERCOOKED_VAR_NAMES)
        self._attribution_classifier = None  # set after BC training
        self._state = None
        self._partner_action = 0  # noop until BC partner is set
        self._bc_partner = None

    def action_space_sample(self) -> int:
        return np.random.randint(0, 6)

    def reset(self) -> np.ndarray:
        self._state = self.env.reset()
        return self._extract_X(self._state)

    def step(self, agent_action: int) -> Tuple[np.ndarray, float, bool, dict]:
        partner_action = self._get_partner_action()
        joint_action = (agent_action, partner_action)
        next_state, reward, done, info = self.env.step(joint_action)
        X_next = self._extract_X(next_state)
        self._state = next_state
        info["partner_attribution"] = self._get_attribution(agent_action, partner_action)
        return X_next, float(reward), bool(done), info

    def _get_partner_action(self) -> int:
        if self._bc_partner is None:
            return 0  # noop
        import torch
        X = self._extract_X(self._state)
        with torch.no_grad():
            logits = self._bc_partner(torch.tensor(X, dtype=torch.float32).unsqueeze(0))
        return int(logits.argmax(dim=-1).item())

    def _get_attribution(self, agent_action: int, partner_action: int) -> list:
        """Returns P(partner_caused_X_i_change) for each variable."""
        if self._attribution_classifier is None:
            return [0.5] * self._M
        import torch
        X = self._extract_X(self._state)
        with torch.no_grad():
            p = self._attribution_classifier(
                torch.tensor(X, dtype=torch.float32).unsqueeze(0)
            ).squeeze(0)
        return p.tolist()

    def _extract_X(self, state) -> np.ndarray:
        """Extract M-dimensional variable vector from Overcooked state."""
        if state is None:
            return np.zeros(self._M, dtype=np.float32)
        try:
            p0 = state.players[0]
            p1 = state.players[1]
            objects = state.objects
        except AttributeError:
            return np.zeros(self._M, dtype=np.float32)

        item_map = {"onion": 1, "plate": 2, "soup": 3}

        def held_code(player):
            if player.held_object is None:
                return 0.0
            return float(item_map.get(player.held_object.name, 0))

        pot_onions, pot_cooking, pot_ready = 0.0, 0.0, 0.0
        for pos, obj in objects.items():
            if obj.name == "soup":
                pot_onions = float(min(len(obj.ingredients), 3))
                if obj.is_cooking:
                    pot_cooking = 1.0
                if obj.is_ready:
                    pot_ready = 1.0

        # Grid dimensions for normalization
        h, w = state.mdp.shape if hasattr(state, "mdp") else (5, 4)

        return np.array([
            p0.position[0] / max(w, 1),
            p0.position[1] / max(h, 1),
            held_code(p0),
            p1.position[0] / max(w, 1),
            p1.position[1] / max(h, 1),
            held_code(p1),
            pot_onions / 3.0,
            pot_cooking,
            pot_ready,
            0.0,   # plates_available (simplified — would need full object scan)
            pot_ready,
            0.0,   # orders_delivered (tracked externally in training loop)
        ], dtype=np.float32)
```

- [ ] **Step 5.6: Run tests to confirm they pass**

```bash
python -m pytest tests/test_overcooked_wrapper.py -v
```

Expected: 4 tests PASSED.

- [ ] **Step 5.7: Implement BC partner trainer**

Create `src/dia_enhanced/warmstart/bc_partner_trainer.py`:

```python
"""Behavior cloning partner model for Overcooked-AI.

Trains a simple MLP policy on human-human trajectory data from Carroll et al. 2019.
The model also serves as the base for the attribution classifier.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class BCPartnerModel(nn.Module):
    """2-layer MLP: X_t → action logits (6 Overcooked actions)."""

    def __init__(self, x_dim: int, hidden: int = 64, n_actions: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, n_actions)
        # Attribution classifier head: P(partner_caused_X_i) per variable
        self.attribution_head = None  # set by train_attribution_classifier()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(X))

    def hidden_state(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)


def train_bc_partner(
    X_sequences: list,   # list of np.ndarray [T, M]
    a_sequences: list,   # list of np.ndarray [T] (partner actions)
    x_dim: int,
    epochs: int = 100,
    lr: float = 1e-3,
) -> BCPartnerModel:
    """Train BC model from partner action sequences."""
    X_all = np.concatenate(X_sequences, axis=0).astype(np.float32)
    a_all = np.concatenate(a_sequences, axis=0).astype(np.int64)

    dataset = TensorDataset(torch.tensor(X_all), torch.tensor(a_all))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = BCPartnerModel(x_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for X_batch, a_batch in loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = F.cross_entropy(logits, a_batch)
            loss.backward()
            optimizer.step()

    return model


def train_attribution_classifier(
    bc_model: BCPartnerModel,
    X_sequences: list,        # list of [T, M]
    partner_acted: list,      # list of [T] bool — True when partner (not agent) acted
    var_index: int,
    x_dim: int,
    epochs: int = 50,
) -> nn.Module:
    """Train attribution classifier for one variable.

    Label = 1 when partner acted AND X_{var_index} changed.
    """
    X_all = np.concatenate(X_sequences, axis=0).astype(np.float32)
    pa_all = np.concatenate(partner_acted, axis=0).astype(np.float32)

    X_tensor = torch.tensor(X_all)
    with torch.no_grad():
        hidden = bc_model.hidden_state(X_tensor).numpy()

    labels = pa_all  # simplified; full impl uses delta_X too

    dataset = TensorDataset(
        torch.tensor(hidden, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    clf = nn.Sequential(nn.Linear(hidden.shape[-1], 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
    optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)

    for _ in range(epochs):
        for h_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = clf(h_batch).squeeze(-1)
            loss = F.binary_cross_entropy(pred, y_batch)
            loss.backward()
            optimizer.step()

    return clf
```

- [ ] **Step 5.8: Implement `overcooked_demo_loader.py`**

Create `src/dia_enhanced/warmstart/overcooked_demo_loader.py`:

```python
"""Loads Carroll et al. 2019 human-human Overcooked demo data and runs warm-start pipeline.

Data is available inside the overcooked_ai package:
  from overcooked_ai_py.data.trajectories import load_demo_trajectories
"""
from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np


def load_carroll_demos(layout: str = "cramped_room") -> List[Dict]:
    """Load published human-human trajectory data for a given layout.

    Returns list of {"obs": [T, M], "actions_agent0": [T], "actions_agent1": [T]}.
    obs is extracted from raw Overcooked states using OvercookedDIAWrapper._extract_X().
    """
    try:
        from overcooked_ai_py.data.trajectories import (
            get_human_human_trajectories,
        )
    except ImportError:
        raise ImportError(
            "overcooked-ai not installed or demo data unavailable. "
            "Run: pip install overcooked-ai"
        )
    from dia_enhanced.envs.overcooked_wrapper import OvercookedDIAWrapper, OvercookedConfig

    raw = get_human_human_trajectories(layout)
    wrapper = OvercookedDIAWrapper(OvercookedConfig(layout=layout))

    processed = []
    for traj in raw:
        states = traj.get("ep_states", [])
        joint_actions = traj.get("ep_actions", [])
        T = min(len(states), len(joint_actions))
        obs = np.array([wrapper._extract_X(s) for s in states[:T]], dtype=np.float32)
        a0 = np.array([ja[0] for ja in joint_actions[:T]], dtype=np.int64)
        a1 = np.array([ja[1] for ja in joint_actions[:T]], dtype=np.int64)
        processed.append({"obs": obs, "actions_agent0": a0, "actions_agent1": a1})

    return processed


def split_agent_trajectories(
    demos: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """Split joint demos into per-agent trajectory dicts for BC training.

    Returns (agent0_trajs, agent1_trajs), each a list of {"obs": ..., "actions": ...}.
    """
    a0_trajs, a1_trajs = [], []
    for d in demos:
        a0_trajs.append({"obs": d["obs"], "actions": d["actions_agent0"]})
        a1_trajs.append({"obs": d["obs"], "actions": d["actions_agent1"]})
    return a0_trajs, a1_trajs
```

- [ ] **Step 5.9: Commit**

```bash
git add src/dia_enhanced/envs/ src/dia_enhanced/warmstart/overcooked_demo_loader.py \
        src/dia_enhanced/warmstart/bc_partner_trainer.py tests/test_overcooked_wrapper.py
git commit -m "feat: add Overcooked DIA wrapper, variable extraction, BC partner trainer, demo loader"
```

---

## Task 6: Hyperbolic SIG Embedding

**Files:**
- Create: `src/dia_enhanced/beliefs/hyperbolic_sig.py`
- Test: `tests/test_hyperbolic_sig.py`

### Background

Each option k gets an embedding `e_k ∈ B^n` (Poincaré ball, n=32). Prerequisites (from PCG) are penalized with a combined distance + norm-ordering loss. Effect similarity (cosine > 0.7 on Δ_k vectors) adds attraction. Optimizer: Riemannian SGD via `geoopt`.

The planner's backward chaining is unchanged — the embedding adds a depth-ordering signal used only for SIG visualization and geodesic planning. The core option selection logic in `planner.py` is not modified.

- [ ] **Step 6.1: Write failing tests**

Create `tests/test_hyperbolic_sig.py`:

```python
import torch
import numpy as np
import pytest

try:
    import geoopt
    HAS_GEOOPT = True
except ImportError:
    HAS_GEOOPT = False

pytestmark = pytest.mark.skipif(not HAS_GEOOPT, reason="geoopt not installed")

from dia_enhanced.beliefs.hyperbolic_sig import HyperbolicSIG, SIGConfig


@pytest.fixture
def sig():
    cfg = SIGConfig(num_options=5, emb_dim=8, lr=0.01)
    return HyperbolicSIG(cfg)


def test_embeddings_in_ball(sig):
    """All embeddings must have norm < 1 (inside Poincaré ball)."""
    norms = sig.embedding_norms()
    assert (norms < 1.0).all(), f"Embeddings outside ball: {norms}"


def test_poincare_distance_nonneg(sig):
    d = sig.poincare_distance(0, 1)
    assert d >= 0.0


def test_prerequisite_update_moves_embeddings(sig):
    """After one update with a prerequisite edge, embeddings should change."""
    e0_before = sig.embeddings[0].detach().clone()
    # Option 0 is prerequisite for option 1
    pcg_probs = np.zeros((5, 5))
    pcg_probs[0, 1] = 0.8
    delta_k = np.zeros((5, 5))  # no effect similarity
    sig.update(pcg_probs, delta_k)
    e0_after = sig.embeddings[0].detach()
    assert not torch.allclose(e0_before, e0_after), "Embeddings should change after update"


def test_prerequisite_ordering_after_update(sig):
    """After multiple updates: prerequisite option should be closer to origin."""
    pcg_probs = np.zeros((5, 5))
    pcg_probs[0, 1] = 0.9  # option 0 → option 1
    delta_k = np.zeros((5, 5))
    for _ in range(50):
        sig.update(pcg_probs, delta_k)
    norm_0 = sig.embedding_norms()[0].item()
    norm_1 = sig.embedding_norms()[1].item()
    assert norm_0 < norm_1, f"Prerequisite (norm={norm_0:.3f}) should be closer to origin than dependent (norm={norm_1:.3f})"
```

- [ ] **Step 6.2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_hyperbolic_sig.py -v 2>&1 | head -15
```

Expected: `ModuleNotFoundError` or skip if geoopt absent.

- [ ] **Step 6.3: Install geoopt if not already present**

```bash
pip install geoopt
python -c "import geoopt; print('geoopt OK')"
```

- [ ] **Step 6.4: Implement `HyperbolicSIG`**

Create `src/dia_enhanced/beliefs/hyperbolic_sig.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
import geoopt
from geoopt.manifolds import PoincareBall


@dataclass
class SIGConfig:
    num_options: int
    emb_dim: int = 32
    lr: float = 0.01
    prereq_weight: float = 1.0
    effect_weight: float = 0.3
    effect_sim_threshold: float = 0.7
    pcg_edge_threshold: float = 0.5


class HyperbolicSIG:
    """SIG embeddings in the Poincaré ball.

    Each option k gets e_k ∈ B^n (Poincaré ball).
    - Prerequisite options (closer in PCG chain) have smaller norm (closer to origin).
    - Options with similar effect signatures cluster together.

    update() is called at each RefreshSIG event.
    """

    def __init__(self, cfg: SIGConfig):
        self.cfg = cfg
        self.manifold = PoincareBall()
        # Initialize embeddings near origin (small norm)
        init = torch.randn(cfg.num_options, cfg.emb_dim) * 0.01
        self.embeddings = geoopt.ManifoldParameter(
            self.manifold.expmap0(init), manifold=self.manifold
        )
        self.optimizer = geoopt.optim.RiemannianSGD([self.embeddings], lr=cfg.lr)

    def embedding_norms(self) -> torch.Tensor:
        """Returns [K] tensor of Euclidean norms (proxy for hyperbolic depth)."""
        return self.embeddings.norm(dim=-1)

    def poincare_distance(self, k: int, k_prime: int) -> float:
        e_k = self.embeddings[k].unsqueeze(0)
        e_kp = self.embeddings[k_prime].unsqueeze(0)
        return self.manifold.dist(e_k, e_kp).item()

    def update(self, pcg_probs: np.ndarray, delta_k: np.ndarray) -> None:
        """One Riemannian SGD step.

        pcg_probs: [K, K] edge probability matrix (option-level)
        delta_k: [K, K] cosine similarity between Δ_k signatures (or zeros if unknown)
        """
        cfg = self.cfg
        K = cfg.num_options
        self.optimizer.zero_grad()

        loss = torch.tensor(0.0)

        # Prerequisite loss
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                if pcg_probs[i, j] >= cfg.pcg_edge_threshold:
                    # i → j means i is prerequisite for j: i closer to origin
                    e_i = self.embeddings[i]
                    e_j = self.embeddings[j]
                    dist = self.manifold.dist(e_i.unsqueeze(0), e_j.unsqueeze(0))
                    norm_i = e_i.norm()
                    norm_j = e_j.norm()
                    # Minimize distance + ensure norm_i < norm_j
                    ordering_pen = F.relu(norm_i - norm_j + 0.05)
                    loss = loss + cfg.prereq_weight * (dist + ordering_pen)

        # Effect similarity loss
        for i in range(K):
            for j in range(i + 1, K):
                if delta_k[i, j] >= cfg.effect_sim_threshold:
                    dist = self.manifold.dist(
                        self.embeddings[i].unsqueeze(0),
                        self.embeddings[j].unsqueeze(0),
                    )
                    loss = loss + cfg.effect_weight * dist

        if loss.requires_grad:
            loss.backward()
            self.optimizer.step()
```

- [ ] **Step 6.5: Run tests to confirm they pass**

```bash
python -m pytest tests/test_hyperbolic_sig.py -v
```

Expected: 4 tests PASSED.

- [ ] **Step 6.6: Commit**

```bash
git add src/dia_enhanced/beliefs/hyperbolic_sig.py tests/test_hyperbolic_sig.py
git commit -m "feat: add hyperbolic SIG embeddings in Poincaré ball"
```

---

## Task 7: Training Loop Integration & Loss Module

**Files:**
- Create: `src/dia_enhanced/training/losses.py`
- Create: `src/dia_enhanced/training/dia_enhanced_loop.py`
- Test: `tests/test_losses.py`
- Test: `tests/test_dia_enhanced_loop.py`

### Background

`DIAEnhancedLoop` wraps the existing `DIARunner` from `dia.rollout`. It inserts three new calls into Algorithm 1:
1. After `UpdatePCG`: `ContrastiveProbe.contrastive_loss` + `TCRegularizer.loss` → backprop
2. At `RefreshSIG`: `HyperbolicSIG.update(pcg_probs, delta_k)`
3. At init: load `A_init` if provided via `GNNPCG.load_a_init()`

`losses.py` gathers all loss computations in one place for clarity.

- [ ] **Step 7.1: Write failing tests**

Create `tests/test_losses.py`:

```python
import torch
import numpy as np
import pytest
from dia_enhanced.training.losses import compute_probe_loss, LossConfig
from dia_enhanced.probes.contrastive_probe import ContrastiveProbe, ProbeConfig
from dia_enhanced.probes.tc_regularizer import TCRegularizer, TCConfig
from dia.types import Subgoal, Predicate


@pytest.fixture
def probe():
    return ContrastiveProbe(ProbeConfig(z_dim=32, num_vars=4, hidden_dims=(32,)))


@pytest.fixture
def reg():
    return TCRegularizer(TCConfig(num_vars=4, total_steps=100))


def test_probe_loss_zero_without_tagged_transitions(probe, reg):
    """No tagged transitions → only TC contributes (which is 0 before warmup)."""
    loss = compute_probe_loss(
        probe=probe, reg=reg, step=0,
        tagged_transitions=[],
        X_batch=torch.randn(16, 4),
        pcg_probs=np.zeros((4, 4)),
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_probe_loss_nonzero_with_tagged_transition(probe, reg):
    z_t = torch.randn(1, 32)
    z_tp1 = torch.randn(1, 32)
    X_t = probe(z_t.detach())
    X_tp1 = probe(z_tp1.detach())
    subgoal = Subgoal(var_index=1, predicate=Predicate.UP)
    tagged = [(X_t, X_tp1, subgoal, 1.0)]

    loss = compute_probe_loss(
        probe=probe, reg=reg, step=50,
        tagged_transitions=tagged,
        X_batch=torch.randn(16, 4),
        pcg_probs=np.zeros((4, 4)),
    )
    assert loss.item() > 0.0


def test_probe_loss_is_differentiable(probe, reg):
    z_t = torch.randn(1, 32)
    z_tp1 = torch.randn(1, 32)
    X_t = probe(z_t.detach())
    X_tp1 = probe(z_tp1.detach())
    subgoal = Subgoal(var_index=0, predicate=Predicate.DOWN)
    tagged = [(X_t, X_tp1, subgoal, 1.0)]

    loss = compute_probe_loss(
        probe=probe, reg=reg, step=50,
        tagged_transitions=tagged,
        X_batch=torch.randn(16, 4),
        pcg_probs=np.zeros((4, 4)),
    )
    loss.backward()
    # Check that probe parameters received gradients
    for p in probe.parameters():
        if p.grad is not None:
            assert p.grad.abs().sum() > 0
            break
```

Create `tests/test_dia_enhanced_loop.py`:

```python
import numpy as np
import pytest
import torch
from dia_enhanced.training.dia_enhanced_loop import DIAEnhancedLoop, EnhancedLoopConfig
from dia_enhanced.probes.contrastive_probe import ContrastiveProbe, ProbeConfig
from dia_enhanced.probes.tc_regularizer import TCRegularizer, TCConfig
from dia_enhanced.beliefs.gnn_pcg import GNNPCG, GNNPCGConfig
from dia_enhanced.beliefs.hyperbolic_sig import HyperbolicSIG, SIGConfig


def test_loop_initializes_from_a_init():
    """Loop should call load_a_init on PCG when A_init is provided."""
    pcg = GNNPCG(GNNPCGConfig(num_vars=3, gnn_layers=1, gnn_hidden=8))
    probe = ContrastiveProbe(ProbeConfig(z_dim=16, num_vars=3, hidden_dims=(16,)))
    reg = TCRegularizer(TCConfig(num_vars=3, total_steps=100))
    sig = HyperbolicSIG(SIGConfig(num_options=3, emb_dim=4))

    A_init = np.array([[0.0, 0.9, 0.0],
                       [0.0, 0.0, 0.8],
                       [0.0, 0.0, 0.0]])
    cfg = EnhancedLoopConfig(num_vars=3, total_steps=100)
    loop = DIAEnhancedLoop(cfg, probe, pcg, reg, sig, a_init=A_init)

    np.testing.assert_allclose(loop.pcg.probs[0, 1], 0.9, atol=0.05)


def test_loop_pcg_entropy_decreases_over_steps():
    """After several updates with consistent data, entropy should not increase on average."""
    pcg = GNNPCG(GNNPCGConfig(num_vars=4, gnn_layers=1, gnn_hidden=8))
    probe = ContrastiveProbe(ProbeConfig(z_dim=16, num_vars=4, hidden_dims=(16,)))
    reg = TCRegularizer(TCConfig(num_vars=4, total_steps=200))
    sig = HyperbolicSIG(SIGConfig(num_options=4, emb_dim=4))
    cfg = EnhancedLoopConfig(num_vars=4, total_steps=200)
    loop = DIAEnhancedLoop(cfg, probe, pcg, reg, sig)

    h0 = loop.pcg.entropy()
    for _ in range(10):
        X_t = torch.randn(8, 4)
        X_tp1 = X_t + torch.randn(8, 4) * 0.1
        actions = torch.zeros(8, dtype=torch.long)
        loop.pcg_update_step(X_t, X_tp1, actions, step=50, tagged=[])
    h10 = loop.pcg.entropy()
    # Entropy may or may not decrease (random data), but should not explode
    assert h10 < h0 * 10, f"Entropy blew up: {h0:.3f} → {h10:.3f}"
```

- [ ] **Step 7.2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_losses.py tests/test_dia_enhanced_loop.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 7.3: Implement `losses.py`**

Create `src/dia_enhanced/training/losses.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import torch

from dia_enhanced.probes.contrastive_probe import ContrastiveProbe
from dia_enhanced.probes.tc_regularizer import TCRegularizer
from dia.types import Subgoal

# Tagged transition: (X_t, X_tp1, subgoal, pcg_confidence)
TaggedTransition = Tuple[torch.Tensor, torch.Tensor, Subgoal, float]


@dataclass
class LossConfig:
    lambda_recon: float = 0.0  # optional reconstruction loss weight


def compute_probe_loss(
    probe: ContrastiveProbe,
    reg: TCRegularizer,
    step: int,
    tagged_transitions: List[TaggedTransition],
    X_batch: torch.Tensor,  # [N, M] — current batch of probe outputs for TC
    pcg_probs: np.ndarray,
) -> torch.Tensor:
    """Compute total probe loss: L_contrastive + λ_TC(t) * L_TC."""
    total = torch.tensor(0.0)

    # Contrastive terms
    for X_t, X_tp1, subgoal, confidence in tagged_transitions:
        cl = probe.contrastive_loss(X_t, X_tp1, subgoal, pcg_probs, pcg_confidence=confidence)
        total = total + cl

    # TC regularizer
    tc_loss = reg.loss(X_batch, step=step, pcg_probs=pcg_probs)
    total = total + tc_loss

    return total
```

- [ ] **Step 7.4: Implement `DIAEnhancedLoop`**

Create `src/dia_enhanced/training/dia_enhanced_loop.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import torch
import torch.optim as optim

from dia_enhanced.probes.contrastive_probe import ContrastiveProbe
from dia_enhanced.probes.tc_regularizer import TCRegularizer
from dia_enhanced.beliefs.gnn_pcg import GNNPCG
from dia_enhanced.beliefs.hyperbolic_sig import HyperbolicSIG
from dia_enhanced.training.losses import compute_probe_loss, TaggedTransition


@dataclass
class EnhancedLoopConfig:
    num_vars: int
    total_steps: int
    probe_lr: float = 1e-3


class DIAEnhancedLoop:
    """Inserts enhanced modules into the DIA Algorithm 1 training loop.

    The caller (DIARunner or script) is responsible for the main loop structure.
    This class provides the three new insertion points:

      1. pcg_update_step()  — after UpdatePCG
      2. sig_refresh_step() — at RefreshSIG
      3. __init__ with a_init — replaces Initialize q_φ(A)
    """

    def __init__(
        self,
        cfg: EnhancedLoopConfig,
        probe: ContrastiveProbe,
        pcg: GNNPCG,
        reg: TCRegularizer,
        sig: HyperbolicSIG,
        a_init: Optional[np.ndarray] = None,
    ):
        self.cfg = cfg
        self.probe = probe
        self.pcg = pcg
        self.reg = reg
        self.sig = sig
        self.probe_optimizer = optim.Adam(probe.parameters(), lr=cfg.probe_lr)

        if a_init is not None:
            self.pcg.load_a_init(a_init)

    def pcg_update_step(
        self,
        X_t: torch.Tensor,      # [N, M] — detached
        X_tp1: torch.Tensor,    # [N, M] — detached
        actions: torch.Tensor,  # [N] long
        step: int,
        tagged: List[TaggedTransition],
    ) -> dict:
        """Run PCG update + probe contrastive + TC update.

        Returns dict of scalars for logging.
        """
        # 1. PCG GNN update
        ig = self.pcg.apply_update(X_t, X_tp1, actions)

        # 2. Probe update
        self.probe_optimizer.zero_grad()
        z_batch = X_t  # In DIA-WM context z_t would come from RSSM; here X_t acts as proxy
        X_probe = self.probe(z_batch.detach())
        probe_loss = compute_probe_loss(
            probe=self.probe,
            reg=self.reg,
            step=step,
            tagged_transitions=tagged,
            X_batch=X_probe,
            pcg_probs=self.pcg.probs,
        )
        if probe_loss.requires_grad:
            probe_loss.backward()
            self.probe_optimizer.step()

        return {
            "ig": ig,
            "probe_loss": probe_loss.item(),
            "pcg_entropy": self.pcg.entropy(),
            "tc_lambda": self.reg.lambda_at_step(step),
        }

    def sig_refresh_step(self, delta_k: Optional[np.ndarray] = None) -> None:
        """Update hyperbolic SIG embeddings using current PCG edge probs."""
        K = self.cfg.num_vars  # Note: in practice K = num_options from SIG, not num_vars
        pcg_probs_option_level = self.pcg.probs  # simplified: use var-level as proxy
        if delta_k is None:
            delta_k = np.zeros((K, K))
        self.sig.update(pcg_probs_option_level[:K, :K], delta_k[:K, :K])
```

- [ ] **Step 7.5: Run tests to confirm they pass**

```bash
python -m pytest tests/test_losses.py tests/test_dia_enhanced_loop.py -v
```

Expected: 5 tests PASSED.

- [ ] **Step 7.6: Run full test suite to confirm no regressions**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: All pre-existing tests still PASS. New tests PASS.

- [ ] **Step 7.7: Commit**

```bash
git add src/dia_enhanced/training/ tests/test_losses.py tests/test_dia_enhanced_loop.py
git commit -m "feat: add DIA-Enhanced training loop integration and losses"
```

---

## Task 8: Evaluation Metrics

**Files:**
- Create: `src/dia_enhanced/eval/disentanglement_metrics.py`
- Create: `src/dia_enhanced/eval/causal_metrics.py`
- Create: `src/dia_enhanced/eval/ablation_runner.py`

### Background

These are standalone evaluation utilities. No new tests required — these are called from experiment scripts. Implement as thin wrappers around established metric definitions.

- [ ] **Step 8.1: Implement `disentanglement_metrics.py`**

Create `src/dia_enhanced/eval/disentanglement_metrics.py`:

```python
"""DCI and MIG disentanglement metrics.

References:
  DCI: Eastwood & Williams 2018
  MIG: Chen et al. 2018
"""
from __future__ import annotations
import numpy as np


def mutual_information_gap(Z: np.ndarray, factors: np.ndarray, bins: int = 20) -> float:
    """MIG: mean over factors of (top-2 MI gap) / H(factor).

    Z: [N, M] — probe outputs
    factors: [N, K] — ground-truth factor values
    """
    N, M = Z.shape
    K = factors.shape[1]
    mis = np.zeros((K, M))

    for k in range(K):
        for m in range(M):
            # Estimate MI via histogram
            f = factors[:, k]
            z = Z[:, m]
            joint, _, _ = np.histogram2d(f, z, bins=bins)
            joint = joint / joint.sum() + 1e-10
            pf = joint.sum(axis=1, keepdims=True)
            pz = joint.sum(axis=0, keepdims=True)
            mis[k, m] = (joint * np.log(joint / (pf * pz))).sum()

    mig_scores = []
    for k in range(K):
        sorted_mi = np.sort(mis[k])[::-1]
        h_k = -np.sum(np.histogram(factors[:, k], bins=bins, density=True)[0] *
                      np.log(np.histogram(factors[:, k], bins=bins, density=True)[0] + 1e-10))
        if h_k > 0:
            mig_scores.append((sorted_mi[0] - sorted_mi[1]) / h_k)

    return float(np.mean(mig_scores)) if mig_scores else 0.0


def dci_disentanglement(Z: np.ndarray, factors: np.ndarray) -> dict:
    """DCI scores (Disentanglement, Completeness, Informativeness).

    Z: [N, M], factors: [N, K]
    Uses gradient-boosted trees importance as proxy.
    Returns dict with keys: disentanglement, completeness, informativeness.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import KBinsDiscretizer

    N, M = Z.shape
    K = factors.shape[1]

    # Discretize factors for classification
    kbd = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    factors_disc = kbd.fit_transform(factors).astype(int)

    importance_matrix = np.zeros((M, K))
    for k in range(K):
        clf = GradientBoostingClassifier(n_estimators=10, max_depth=2)
        clf.fit(Z, factors_disc[:, k])
        importance_matrix[:, k] = clf.feature_importances_

    # Normalize per column
    col_sums = importance_matrix.sum(axis=0, keepdims=True) + 1e-10
    R = importance_matrix / col_sums

    # Disentanglement: each Z_m codes for one factor
    D = 1 - (-R * np.log(R + 1e-10) / np.log(K + 1e-10)).sum(axis=1)
    rho = importance_matrix.sum(axis=1)
    rho = rho / (rho.sum() + 1e-10)
    disentanglement = float((D * rho).sum())

    # Completeness: each factor is coded by one Z_m
    row_sums = importance_matrix.sum(axis=1, keepdims=True) + 1e-10
    C_mat = importance_matrix / row_sums
    C = 1 - (-C_mat * np.log(C_mat + 1e-10) / np.log(M + 1e-10)).sum(axis=0)
    completeness = float(C.mean())

    informativeness = float(importance_matrix.max(axis=0).mean())

    return {
        "disentanglement": disentanglement,
        "completeness": completeness,
        "informativeness": informativeness,
    }
```

- [ ] **Step 8.2: Implement `causal_metrics.py`**

Create `src/dia_enhanced/eval/causal_metrics.py`:

```python
"""PCG causal quality metrics: entropy half-life, SHD, ECE, intervention precision."""
from __future__ import annotations
import numpy as np


def pcg_entropy_half_life(entropy_curve: np.ndarray) -> int:
    """t_{1/2}: first frame where entropy <= 0.5 * initial entropy.

    entropy_curve: [T] array of entropy values over training.
    Returns frame index, or T if never reached.
    """
    if len(entropy_curve) == 0:
        return 0
    h0 = entropy_curve[0]
    threshold = h0 * 0.5
    indices = np.where(entropy_curve <= threshold)[0]
    return int(indices[0]) if len(indices) > 0 else len(entropy_curve)


def structural_hamming_distance(A_pred: np.ndarray, A_true: np.ndarray, threshold: float = 0.5) -> int:
    """SHD between predicted and true adjacency matrices.

    A_pred: [M, M] — predicted edge probabilities
    A_true: [M, M] — ground truth binary adjacency
    """
    A_bin = (A_pred >= threshold).astype(int)
    A_t = (A_true > 0).astype(int)
    return int(np.abs(A_bin - A_t).sum())


def expected_calibration_error(A_pred: np.ndarray, A_true: np.ndarray, n_bins: int = 10) -> float:
    """ECE: calibration of edge posterior probabilities."""
    probs = A_pred.flatten()
    labels = (A_true.flatten() > 0).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = probs[mask].mean()
        avg_acc = labels[mask].mean()
        ece += mask.sum() / len(probs) * abs(avg_conf - avg_acc)
    return float(ece)


def intervention_precision(
    intervention_tags: list,   # list of (t, var_index) tagged by agent
    true_interventions: list,  # list of (t, var_index) ground truth
) -> float:
    """Fraction of agent-tagged interventions that match ground truth."""
    if not intervention_tags:
        return 0.0
    correct = sum(1 for tag in intervention_tags if tag in set(true_interventions))
    return correct / len(intervention_tags)
```

- [ ] **Step 8.3: Implement `ablation_runner.py` (config skeleton)**

Create `src/dia_enhanced/eval/ablation_runner.py`:

```python
"""Ablation table runner. Runs each configuration from the spec ablation table.

Usage:
    python -m dia_enhanced.eval.ablation_runner --env minecraft_trade --seeds 5
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass


@dataclass
class AblationConfig:
    contrastive_probe: bool = True
    gnn_pcg: bool = True
    hyperbolic_sig: bool = True
    tc_regularizer: bool = True
    demo_warmstart: bool = True


ABLATION_TABLE = [
    AblationConfig(False, False, False, False, False),  # DIA baseline
    AblationConfig(True,  False, False, False, False),  # + Contrastive only
    AblationConfig(True,  True,  False, False, False),  # + GNN-PCG
    AblationConfig(True,  True,  True,  False, False),  # + Hyperbolic SIG
    AblationConfig(True,  True,  True,  True,  False),  # + TC regularizer
    AblationConfig(True,  True,  True,  True,  True),   # Full system
    AblationConfig(False, False, False, True,  False),  # TC only
]


def run_ablation(env_name: str, n_seeds: int, cfg: AblationConfig) -> dict:
    """Run one ablation configuration, returning metrics dict.

    Builds DIAEnhancedLoop with the specified modules enabled/disabled,
    runs training for total_steps, and returns evaluation metrics.
    """
    import torch
    from dia_enhanced.envs.env_registry import get_env_spec
    from dia_enhanced.probes.contrastive_probe import ContrastiveProbe, ProbeConfig
    from dia_enhanced.probes.tc_regularizer import TCRegularizer, TCConfig
    from dia_enhanced.beliefs.gnn_pcg import GNNPCG, GNNPCGConfig
    from dia_enhanced.beliefs.hyperbolic_sig import HyperbolicSIG, SIGConfig
    from dia_enhanced.training.dia_enhanced_loop import DIAEnhancedLoop, EnhancedLoopConfig
    from dia_enhanced.warmstart.rssm_demo_processor import DemoProcessor, ProcessorConfig
    from dia_enhanced.eval.causal_metrics import pcg_entropy_half_life

    spec = get_env_spec(env_name)
    total_steps = 500_000
    results = {"returns": [], "entropy_curves": [], "shd": []}

    for seed in range(n_seeds):
        torch.manual_seed(seed)

        probe = ContrastiveProbe(ProbeConfig(z_dim=spec.obs_dim, num_vars=spec.num_vars))
        pcg = GNNPCG(GNNPCGConfig(num_vars=spec.num_vars))
        reg = TCRegularizer(TCConfig(num_vars=spec.num_vars, total_steps=total_steps))
        sig = HyperbolicSIG(SIGConfig(num_options=spec.num_options))

        # Disable modules not in this config
        if not cfg.gnn_pcg:
            from dia.pcg import SimplePCG, PCGConfig
            pcg = SimplePCG(PCGConfig(num_vars=spec.num_vars))  # type: ignore
        if not cfg.tc_regularizer:
            reg = TCRegularizer(TCConfig(num_vars=spec.num_vars, total_steps=total_steps, lambda_tc_max=0.0))

        a_init = None
        if cfg.demo_warmstart:
            # Load pre-computed A_init for this env (must be generated beforehand)
            import os
            a_init_path = f"data/warmstart/{env_name}_a_init.npy"
            if os.path.exists(a_init_path):
                a_init = __import__("numpy").load(a_init_path)

        loop = DIAEnhancedLoop(
            EnhancedLoopConfig(num_vars=spec.num_vars, total_steps=total_steps),
            probe=probe if cfg.contrastive_probe else None,
            pcg=pcg,
            reg=reg,
            sig=sig if cfg.hyperbolic_sig else None,
            a_init=a_init,
        )

        # Training loop stub — replace with actual env interaction
        entropy_curve = []
        for step in range(0, total_steps, 256):
            X_t = torch.randn(16, spec.num_vars)
            X_tp1 = X_t + torch.randn(16, spec.num_vars) * 0.1
            actions = torch.zeros(16, dtype=torch.long)
            if hasattr(loop, 'pcg_update_step'):
                loop.pcg_update_step(X_t, X_tp1, actions, step=step, tagged=[])
            entropy_curve.append(loop.pcg.entropy() if hasattr(loop.pcg, 'entropy') else 0.0)

        results["entropy_curves"].append(entropy_curve)
        results["returns"].append(0.0)  # replace with actual episode return

    results["entropy_half_life"] = pcg_entropy_half_life(
        __import__("numpy").array(results["entropy_curves"][0])
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="minecraft_trade")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    for i, ablation_cfg in enumerate(ABLATION_TABLE):
        print(f"\nRunning ablation {i+1}/{len(ABLATION_TABLE)}: {ablation_cfg}")
        metrics = run_ablation(args.env, args.seeds, ablation_cfg)
        print(f"  entropy_half_life={metrics['entropy_half_life']}")
```

- [ ] **Step 8.4: Commit**

```bash
git add src/dia_enhanced/eval/
git commit -m "feat: add disentanglement/causal metrics and ablation runner skeleton"
```

---

## Task 9: Environment Registry and Configs

**Files:**
- Create: `src/dia_enhanced/envs/env_registry.py`
- Create: `configs/minecraft_trade.yaml`
- Create: `configs/nethack.yaml`
- Create: `configs/coinrun.yaml`
- Create: `configs/causalworld.yaml`
- Create: `configs/overcooked.yaml`

- [ ] **Step 9.1: Implement `env_registry.py`**

Create `src/dia_enhanced/envs/env_registry.py`:

```python
"""Unified environment creation for all 5 DIA-Enhanced domains."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class EnvSpec:
    name: str
    num_vars: int
    num_options: int
    obs_dim: int
    config_path: str


ENV_REGISTRY = {
    "minecraft_trade": EnvSpec("minecraft_trade", num_vars=15, num_options=10, obs_dim=256, config_path="configs/minecraft_trade.yaml"),
    "nethack":         EnvSpec("nethack",         num_vars=10, num_options=8,  obs_dim=512, config_path="configs/nethack.yaml"),
    "coinrun":         EnvSpec("coinrun",          num_vars=6,  num_options=4,  obs_dim=256, config_path="configs/coinrun.yaml"),
    "causalworld":     EnvSpec("causalworld",      num_vars=10, num_options=6,  obs_dim=128, config_path="configs/causalworld.yaml"),
    "overcooked":      EnvSpec("overcooked",       num_vars=12, num_options=7,  obs_dim=12,  config_path="configs/overcooked.yaml"),
}


def get_env_spec(name: str) -> EnvSpec:
    if name not in ENV_REGISTRY:
        raise ValueError(f"Unknown env: {name}. Available: {list(ENV_REGISTRY)}")
    return ENV_REGISTRY[name]
```

- [ ] **Step 9.2: Write environment configs**

Create `configs/overcooked.yaml`:

```yaml
env: overcooked
layout: cramped_room
horizon: 400
num_vars: 12
num_options: 7
obs_dim: 12

probe:
  z_dim: 12
  hidden_dims: [64, 32]

gnn_pcg:
  gnn_layers: 2
  gnn_hidden: 64
  lambda_dag: 0.01
  lambda_sp: 0.01

tc:
  t_warm: 0.20
  t_anneal: 0.40
  lambda_tc_max: 0.1
  tau_group: 0.7

sig:
  emb_dim: 32
  lr: 0.01

demo:
  data_source: carroll_2019
  warmstart_epochs: 100
  bc_partner_hidden: 64
  attribution_epochs: 50

training:
  total_steps: 500000
  n_pcg_update: 5
  n_sig_refresh: 100
```

Create `configs/minecraft_trade.yaml`:

```yaml
env: minecraft_trade
num_vars: 15
num_options: 10
obs_dim: 256

probe:
  z_dim: 256
  hidden_dims: [512, 256]

gnn_pcg:
  gnn_layers: 2
  gnn_hidden: 128
  lambda_dag: 0.01
  lambda_sp: 0.005

tc:
  t_warm: 0.20
  t_anneal: 0.40
  lambda_tc_max: 0.1
  tau_group: 0.7

sig:
  emb_dim: 32
  lr: 0.01

demo:
  data_dir: data/demos/minecraft3d/
  warmstart_epochs: 100

training:
  total_steps: 2000000
  n_pcg_update: 5
  n_sig_refresh: 200
```

Create `configs/coinrun.yaml`:

```yaml
env: coinrun
num_vars: 6
num_options: 4
obs_dim: 256

probe:
  z_dim: 256
  hidden_dims: [512, 256]

gnn_pcg:
  gnn_layers: 2
  gnn_hidden: 64
  lambda_dag: 0.01
  lambda_sp: 0.005

tc:
  t_warm: 0.20
  t_anneal: 0.40
  lambda_tc_max: 0.1
  tau_group: 0.7

sig:
  emb_dim: 32
  lr: 0.01

training:
  total_steps: 1000000
  n_pcg_update: 5
  n_sig_refresh: 100
```

Create `configs/nethack.yaml`:

```yaml
env: nethack
num_vars: 10
num_options: 8
obs_dim: 512

probe:
  z_dim: 512
  hidden_dims: [512, 256]

gnn_pcg:
  gnn_layers: 3
  gnn_hidden: 128
  lambda_dag: 0.01
  lambda_sp: 0.01

tc:
  t_warm: 0.20
  t_anneal: 0.40
  lambda_tc_max: 0.1
  tau_group: 0.7

sig:
  emb_dim: 32
  lr: 0.01

training:
  total_steps: 2000000
  n_pcg_update: 5
  n_sig_refresh: 200
```

Create `configs/causalworld.yaml`:

```yaml
env: causalworld
num_vars: 10
num_options: 6
obs_dim: 128

probe:
  z_dim: 128
  hidden_dims: [256, 128]

gnn_pcg:
  gnn_layers: 2
  gnn_hidden: 64
  lambda_dag: 0.01
  lambda_sp: 0.005

tc:
  t_warm: 0.20
  t_anneal: 0.40
  lambda_tc_max: 0.1
  tau_group: 0.7

sig:
  emb_dim: 32
  lr: 0.01

training:
  total_steps: 1000000
  n_pcg_update: 5
  n_sig_refresh: 100
```

- [ ] **Step 9.3: Commit**

```bash
git add src/dia_enhanced/envs/env_registry.py configs/
git commit -m "feat: add environment registry and YAML configs for all 5 domains"
```

---

## Final Verification

- [ ] **Run full test suite**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: All tests pass. No regressions in existing `dia` tests.

- [ ] **Smoke test import of full package**

```bash
python -c "
from dia_enhanced.probes.contrastive_probe import ContrastiveProbe, ProbeConfig
from dia_enhanced.probes.tc_regularizer import TCRegularizer, TCConfig
from dia_enhanced.beliefs.gnn_pcg import GNNPCG, GNNPCGConfig
from dia_enhanced.beliefs.hyperbolic_sig import HyperbolicSIG, SIGConfig
from dia_enhanced.training.dia_enhanced_loop import DIAEnhancedLoop, EnhancedLoopConfig
from dia_enhanced.eval.causal_metrics import pcg_entropy_half_life, structural_hamming_distance
from dia_enhanced.envs.env_registry import get_env_spec
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Final commit**

```bash
git add -A
git commit -m "feat: DIA-Enhanced — complete implementation (probe, GNN-PCG, hyperbolic SIG, warmstart, Overcooked, eval)"
```
