# scripts/synth_sem.py
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np

from dia.sem_utils import random_dag, random_weights_from_dag, sample_linear_sem, intervention_mask
from dia.pcg_learner import DifferentiablePCG, DifferentiablePCGConfig
from dia.pcg_variational import VariationalPCG, VariationalPCGConfig

def precision_at_k(P: np.ndarray, A_true: np.ndarray, k: int) -> float:
    d = P.shape[0]
    mask = ~np.eye(d, dtype=bool)
    scores = P[mask]
    idxs = np.argwhere(mask)
    order = np.argsort(scores)[::-1]
    top = idxs[order][:k]
    hits = sum(A_true[i, j] > 0 for i, j in top)
    return hits / max(1, k)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=8, help="num variables")
    ap.add_argument("--edges", type=float, default=0.25, help="edge probability in DAG")
    ap.add_argument("--N", type=int, default=4000, help="num samples")
    ap.add_argument("--interv_prob", type=float, default=0.0, help="probability a var is intervened per-sample")
    ap.add_argument("--learner", type=str, default="both", choices=["nt", "var", "both"])
    ap.add_argument("--epochs", type=int, default=800)
    args = ap.parse_args()

    A = random_dag(args.d, edge_prob=args.edges, seed=0)
    W = random_weights_from_dag(A, seed=0)
    X = sample_linear_sem(W, N=args.N, noise_scale=0.5, seed=0)

    mask = intervention_mask(args.N, args.d, prob=args.interv_prob, seed=0) if args.interv_prob > 0 else None

    k = int(A.sum()) if A.sum() >= 1 else 1
    print(f"[SEM] d={args.d} true_edges={int(A.sum())} N={args.N} interv_prob={args.interv_prob}")

    if args.learner in ("nt", "both"):
        cfg = DifferentiablePCGConfig(num_vars=args.d, max_iter=args.epochs, lr=5e-3, verbose=True)
        pcg = DifferentiablePCG(cfg)
        res = pcg.fit(X, mask=mask, epochs=args.epochs)
        P = pcg.probs
        print("[NOTEARS-like] h=", res["h"], " entropy=", pcg.entropy())
        print("[NOTEARS-like] P@k=", precision_at_k(P, A, k))

    if args.learner in ("var", "both"):
        cfg = VariationalPCGConfig(num_vars=args.d, max_iter=args.epochs, lr=5e-3, K=4, verbose=True)
        vpcg = VariationalPCG(cfg)
        res = vpcg.fit(X, mask=mask, epochs=args.epochs)
        P = vpcg.probs
        print("[Variational] h=", res["h"], " entropy=", vpcg.entropy())
        print("[Variational] P@k=", precision_at_k(P, A, k))

if __name__ == "__main__":
    main()
