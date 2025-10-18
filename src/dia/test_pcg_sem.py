# tests/test_pcg_sem.py
import numpy as np

from dia.sem_utils import random_dag, random_weights_from_dag, sample_linear_sem  # noqa: E402
from dia.pcg_learner import DifferentiablePCG, DifferentiablePCGConfig           # noqa: E402
from dia.pcg_variational import VariationalPCG, VariationalPCGConfig             # noqa: E402

# fix spaced import typo
from dia.sem_utils import random_dag, random_weights_from_dag, sample_linear_sem

def topk_hit_rate(P_est: np.ndarray, A_true: np.ndarray, k: int) -> float:
    d = P_est.shape[0]
    idx = np.argwhere(~np.eye(d, dtype=bool))
    scores = P_est[~np.eye(d, dtype=bool)]
    order = np.argsort(scores)[::-1]
    top_idx = idx[order][:k]
    hits = 0
    for (i, j) in top_idx:
        if A_true[i, j] > 0.0:
            hits += 1
    return hits / max(1, k)

def test_differentiablepcg_topk():
    d = 6
    A = random_dag(d, edge_prob=0.3, seed=0)
    W = random_weights_from_dag(A, seed=0)
    X = sample_linear_sem(W, N=3000, noise_scale=0.5, seed=0)

    cfg = DifferentiablePCGConfig(num_vars=d, max_iter=500, lr=5e-3, verbose=False)
    pcg = DifferentiablePCG(cfg)
    pcg.fit(X, mask=None, epochs=500)

    P = pcg.probs
    k = int(A.sum())
    hr = topk_hit_rate(P, A, k=k if k > 0 else 1)
    assert hr >= 0.5  # at least half the true edges appear in top-k

def test_variationalpcg_topk():
    d = 6
    A = random_dag(d, edge_prob=0.3, seed=42)
    W = random_weights_from_dag(A, seed=42)
    X = sample_linear_sem(W, N=3000, noise_scale=0.5, seed=42)

    cfg = VariationalPCGConfig(num_vars=d, max_iter=500, lr=5e-3, K=4, verbose=False)
    vpcg = VariationalPCG(cfg)
    vpcg.fit(X, mask=None, epochs=500)

    P = vpcg.probs
    k = int(A.sum())
    hr = topk_hit_rate(P, A, k=k if k > 0 else 1)
    assert hr >= 0.5
