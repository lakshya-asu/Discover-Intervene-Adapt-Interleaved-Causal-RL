from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np

from .intrinsic import bernoulli_kl, entropy_bernoulli


@dataclass
class PCGState:
    """Holds edge probabilities and caches for IG/entropy computations."""
    edge_probs: np.ndarray  # shape [M, M], diag=0, entries in [0,1]
    step: int = 0


@dataclass
class PCGConfig:
    num_vars: int
    init_edge_prob: float = 0.05
    seed: Optional[int] = None


class SimplePCG:
    """A lightweight probabilistic causal graph with independent Bernoulli edges.
    Notes:
      - This is a practical placeholder that supports info-gain and entropy tracking.
      - A full differentiable DAG learner (e.g., NOTEARS-like) can be wired under the same API.
    """
    def __init__(self, cfg: PCGConfig):
        self.cfg = cfg
        rng = np.random.RandomState(cfg.seed or 0)
        p0 = np.full((cfg.num_vars, cfg.num_vars), float(cfg.init_edge_prob), dtype=float)
        np.fill_diagonal(p0, 0.0)
        self.state = PCGState(edge_probs=p0, step=0)

    @property
    def probs(self) -> np.ndarray:
        return self.state.edge_probs

    def entropy(self) -> float:
        # sum of Bernoulli entropies over edges (excluding diag)
        p = self.state.edge_probs.copy()
        np.fill_diagonal(p, 0.0)
        return entropy_bernoulli(p)

    def expected_ig_from_update(self, new_probs: np.ndarray) -> float:
        """Compute IG = KL( q_new || q_old ) under independent Bernoullis."""
        old = self.state.edge_probs
        return bernoulli_kl(new_probs, old)

    def apply_update(self, new_probs: np.ndarray) -> float:
        """Apply update and return realized info gain."""
        ig = self.expected_ig_from_update(new_probs)
        self.state.edge_probs = np.clip(new_probs, 0.0, 1.0)
        np.fill_diagonal(self.state.edge_probs, 0.0)
        self.state.step += 1
        return ig

    def conservative_update(self, delta: np.ndarray, lr: float = 0.1) -> Tuple[np.ndarray, float]:
        """Convenience: new_probs = probs + lr * delta (clipped), returns (new_probs, ig)."""
        new_p = np.clip(self.state.edge_probs + lr * delta, 0.0, 1.0)
        np.fill_diagonal(new_p, 0.0)
        ig = self.expected_ig_from_update(new_p)
        self.state.edge_probs = new_p
        self.state.step += 1
        return new_p, ig

    def suggest_edges_by_uncertainty(self, k: int = 5) -> np.ndarray:
        """Return indices of k edges with highest Bernoulli entropy (i.e., most uncertain)."""
        p = self.state.edge_probs.copy()
        np.fill_diagonal(p, 0.0)
        h = -p * np.log(np.clip(p, 1e-8, 1-1e-8)) - (1-p) * np.log(np.clip(1-p, 1e-8, 1-1e-8))
        # mask diag
        np.fill_diagonal(h, -np.inf)
        flat_idx = np.argsort(h.ravel())[::-1][:k]
        edges = np.array([divmod(i, p.shape[1]) for i in flat_idx], dtype=int)
        return edges
