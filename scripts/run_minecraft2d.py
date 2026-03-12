#!/usr/bin/env python3
"""
scripts/run_minecraft2d.py — DIA on 2D-Minecraft with full metrics.

Metrics tracked
---------------
  - PCG entropy H[q_φ] over time
  - Information gain (IG) per PCG update
  - Structural Hamming Distance (SHD) vs. ground-truth causal graph
  - Edge precision / recall vs. ground-truth
  - Intervention precision  (fraction of skill executions that succeed)
  - Per-skill success rates
  - Chain milestones (first step each resource is acquired)
  - SIG edge count (how many prerequisites learned)

Output
------
  runs/dia/  (or --logdir)
    metrics.csv       — per-step CSV
    final_pcg.json    — final PCG probabilities + causal metrics
    events.out.*      — TensorBoard events

Run
---
  cd Discover-Intervene-Adapt-Interleaved-Causal-RL
  python scripts/run_minecraft2d.py --steps 600 --pcg notears
  tensorboard --logdir runs/dia
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from dia.envs.minecraft2d import MinecraftChainEnv, ChainConfig
from dia.evgs_minecraft import make_minecraft_evgs, VAR_NAMES
from dia.evgs import EVGS
from dia.sig import SIGraph, Skill
from dia.types import Subgoal, Predicate
from dia.planner import PlannerConfig, InterventionSelector
from dia.rollout import DIARunner, RunnerConfig
from dia.logging_utils import TBLogger
from dia.pcg_learner import DifferentiablePCG, DifferentiablePCGConfig
from dia.pcg_variational import VariationalPCG, VariationalPCGConfig
from dia.pcg import SimplePCG, PCGConfig
from dia.options import OptionConfig
from dia.options_minecraft import ResourceOption


# ── Ground-truth causal graph ─────────────────────────────────────────────────
# Variables:  wood(0) stone(1) coal(2) ironore(3) furnace(4)
#             stonepickaxe(5) iron(6) ironpickaxe(7) diamond(8)
#
# Edge (i→j): Xi causally enables Xj (prerequisite in crafting chain).

GROUND_TRUTH_EDGES: List[Tuple[int, int]] = [
    (1, 4),  # stone      -> furnace
    (0, 5),  # wood       -> stonepickaxe
    (1, 5),  # stone      -> stonepickaxe
    (5, 3),  # stonepickaxe -> ironore
    (4, 6),  # furnace    -> iron
    (2, 6),  # coal       -> iron
    (3, 6),  # ironore    -> iron
    (6, 7),  # iron       -> ironpickaxe
    (0, 7),  # wood       -> ironpickaxe
    (7, 8),  # ironpickaxe -> diamond
]

# Short display codes for the 9 variables (for ASCII matrix)
VAR_CODES = ["WD", "ST", "CL", "IO", "FN", "SP", "IR", "IP", "DM"]


# ── Causal metrics ────────────────────────────────────────────────────────────

def _gt_adjacency(n_vars: int) -> np.ndarray:
    A = np.zeros((n_vars, n_vars), dtype=float)
    for i, j in GROUND_TRUTH_EDGES:
        A[i, j] = 1.0
    return A


def shd(probs: np.ndarray, A_gt: np.ndarray, threshold: float = 0.5) -> int:
    A_pred = (probs >= threshold).astype(float)
    np.fill_diagonal(A_pred, 0.0)
    return int(np.sum(np.abs(A_pred - A_gt)))


def precision_recall(probs: np.ndarray, A_gt: np.ndarray, threshold: float = 0.5
                     ) -> Tuple[float, float, int, int, int]:
    A_pred = (probs >= threshold).astype(float)
    np.fill_diagonal(A_pred, 0.0)
    tp = int(np.sum((A_gt > 0) & (A_pred > 0)))
    fp = int(np.sum((A_gt == 0) & (A_pred > 0)))
    fn = int(np.sum((A_gt > 0) & (A_pred == 0)))
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return p, r, tp, fp, fn


# ── ASCII PCG display ─────────────────────────────────────────────────────────

def pcg_ascii(probs: np.ndarray, threshold: float = 0.5) -> str:
    n = probs.shape[0]
    codes = VAR_CODES[:n]
    header = "       " + " ".join(f"{c:>4}" for c in codes)
    lines = [header]
    for i in range(n):
        row = f"  {codes[i]}  "
        for j in range(n):
            if i == j:
                row += "   - "
            else:
                p = probs[i, j]
                star = "*" if p >= threshold else " "
                row += f"{p:.2f}{star}"
        lines.append(row)
    return "\n".join(lines)


# ── SIG builder ───────────────────────────────────────────────────────────────

def build_sig(evgs: EVGS) -> SIGraph:
    idx = {name: i for i, name in enumerate(evgs.names())}
    sig = SIGraph()
    for name in VAR_NAMES:
        sid = idx[name]
        sig.add_skill(Skill(skill_id=sid,
                            subgoal=Subgoal(var_index=sid, predicate=Predicate.UP),
                            name=f"{name}↑"))
    sig.add_prerequisite(idx["stone"],        idx["furnace"])
    sig.add_prerequisite(idx["wood"],         idx["stonepickaxe"])
    sig.add_prerequisite(idx["stone"],        idx["stonepickaxe"])
    sig.add_prerequisite(idx["stonepickaxe"], idx["ironore"])
    sig.add_prerequisite(idx["furnace"],      idx["iron"])
    sig.add_prerequisite(idx["coal"],         idx["iron"])
    sig.add_prerequisite(idx["ironore"],      idx["iron"])
    sig.add_prerequisite(idx["iron"],         idx["ironpickaxe"])
    sig.add_prerequisite(idx["wood"],         idx["ironpickaxe"])
    sig.add_prerequisite(idx["ironpickaxe"],  idx["diamond"])
    return sig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="DIA 2D-Minecraft run with metrics")
    ap.add_argument("--steps",        type=int,   default=600)
    ap.add_argument("--pcg",          type=str,   default="notears",
                    choices=["notears", "variational", "simple"])
    ap.add_argument("--fit_every",    type=int,   default=10)
    ap.add_argument("--pcg_epochs",   type=int,   default=200)
    ap.add_argument("--min_buffer",   type=int,   default=64)
    ap.add_argument("--buffer_recent",type=int,   default=512)
    ap.add_argument("--option_steps", type=int,   default=64,
                    help="Max primitive steps per option execution")
    ap.add_argument("--task_goal",    type=str,   default="diamond")
    ap.add_argument("--logdir",       type=str,   default="runs/dia")
    ap.add_argument("--print_every",  type=int,   default=20)
    ap.add_argument("--pcg_every",    type=int,   default=100,
                    help="Steps between PCG matrix printouts")
    ap.add_argument("--seed",         type=int,   default=0)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    os.makedirs(args.logdir, exist_ok=True)
    metrics_path = os.path.join(args.logdir, "metrics.csv")

    # ── Environment & EVGS ──────────────────────────────────────────────────
    env  = MinecraftChainEnv(ChainConfig(), seed=args.seed)
    evgs = make_minecraft_evgs()
    var_names = evgs.names()
    M = len(var_names)
    name_to_idx = {n: i for i, n in enumerate(var_names)}
    A_gt = _gt_adjacency(M)

    # ── PCG backend ─────────────────────────────────────────────────────────
    if args.pcg == "simple":
        pcg = SimplePCG(PCGConfig(num_vars=M, init_edge_prob=0.05, seed=args.seed))
    elif args.pcg == "variational":
        pcg = VariationalPCG(VariationalPCGConfig(
            num_vars=M, max_iter=args.pcg_epochs, lr=5e-3, K=4, verbose=False))
    else:
        pcg = DifferentiablePCG(DifferentiablePCGConfig(
            num_vars=M, max_iter=args.pcg_epochs, lr=5e-3, verbose=False))

    # ── SIG & planner ───────────────────────────────────────────────────────
    sig          = build_sig(evgs)
    planner_cfg  = PlannerConfig(entropy_high=25.0, entropy_low=3.0)
    selector     = InterventionSelector(pcg, sig, planner_cfg)

    # ── Runner ──────────────────────────────────────────────────────────────
    logger = TBLogger(args.logdir)
    rcfg = RunnerConfig(
        buffer_size=20_000,
        min_buffer=args.min_buffer,
        batch_recent=args.buffer_recent,
        fit_every=args.fit_every,
        pcg_epochs=args.pcg_epochs,
        log_prefix="mc2d",
        option_max_steps=args.option_steps,
        terminate_on_success=True,
        auto_expand_sig=True,
        add_threshold=0.75,
        remove_threshold=0.55,
    )

    def option_factory(skill: Skill) -> ResourceOption:
        return ResourceOption(
            skill.subgoal,
            OptionConfig(max_steps=args.option_steps, terminate_on_success=True))

    runner = DIARunner(env, evgs, pcg, sig, selector, rcfg,
                       logger=logger, option_factory=option_factory)

    # ── Task goal ────────────────────────────────────────────────────────────
    goal_var  = args.task_goal if args.task_goal in name_to_idx else "diamond"
    task_goal = Subgoal(var_index=name_to_idx[goal_var], predicate=Predicate.UP)

    # ── Metric accumulators ──────────────────────────────────────────────────
    achieved:        List[int]          = []
    skill_attempts:  Dict[int, int]     = {i: 0 for i in range(M)}
    skill_successes: Dict[int, int]     = {i: 0 for i in range(M)}
    milestones:      Dict[str, Optional[int]] = {v: None for v in var_names}
    total_int   = 0
    total_succ  = 0

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_fh = open(metrics_path, "w", newline="")
    csv_w  = csv.DictWriter(csv_fh, fieldnames=[
        "step", "explore", "phase", "skill", "success",
        "ig", "pcg_entropy", "shd", "causal_prec", "causal_rec",
        "int_precision", "sig_edges", "achieved_count",
    ])
    csv_w.writeheader()

    # ── Header ───────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  DIA 2D-Minecraft  |  PCG={args.pcg}  |  goal={goal_var}  |  steps={args.steps}")
    print(f"  vars : {', '.join(var_names)}")
    print(f"  codes: {', '.join(f'{v}={c}' for v, c in zip(var_names, VAR_CODES))}")
    print(f"  GT edges ({len(GROUND_TRUTH_EDGES)}): "
          + "  ".join(f"{VAR_CODES[i]}→{VAR_CODES[j]}" for i, j in GROUND_TRUTH_EDGES))
    print(f"  logdir: {args.logdir}")
    print(f"{'='*72}\n")

    t0 = time.time()

    for t in range(args.steps):

        # ── Exploration gate  ρ = min(1, H / entropy_high) ──────────────────
        H   = pcg.entropy() if hasattr(pcg, "entropy") else 0.0
        rho = min(1.0, H / planner_cfg.entropy_high)
        explore = bool(rng.random() < rho)
        tg = None if explore else task_goal

        rec = runner.step(achieved, task_goal=tg)

        # ── Collect step info ────────────────────────────────────────────────
        sid     = rec["skill_id"]
        sname   = rec["skill_name"]
        success = rec["success"]
        ig      = rec["ig_update"]
        h       = rec["pcg_entropy"] if not np.isnan(rec["pcg_entropy"]) else H

        skill_attempts[sid]  = skill_attempts.get(sid, 0) + 1
        skill_successes[sid] = skill_successes.get(sid, 0) + (1 if success else 0)
        total_int  += 1
        total_succ += 1 if success else 0

        if success and milestones.get(var_names[sid]) is None:
            milestones[var_names[sid]] = t + 1
            print(f"  ★ [{t+1:04d}] MILESTONE: {var_names[sid]} acquired for the first time!")

        if success and sid not in achieved:
            achieved.append(sid)

        # ── Causal metrics ───────────────────────────────────────────────────
        probs = np.array(pcg.probs) if hasattr(pcg, "probs") else np.full((M, M), 0.5)
        np.fill_diagonal(probs, 0.0)
        shd_val          = shd(probs, A_gt)
        cprec, crec, *_  = precision_recall(probs, A_gt)
        int_prec         = total_succ / max(1, total_int)
        sig_edges        = sum(len(v) for v in sig.edges.values())

        # ── TensorBoard ──────────────────────────────────────────────────────
        logger.add_scalar("mc2d/pcg_entropy",        h,        t)
        logger.add_scalar("mc2d/ig",                 ig,       t)
        logger.add_scalar("mc2d/shd",                shd_val,  t)
        logger.add_scalar("mc2d/causal_precision",   cprec,    t)
        logger.add_scalar("mc2d/causal_recall",      crec,     t)
        logger.add_scalar("mc2d/int_precision",      int_prec, t)
        logger.add_scalar("mc2d/achieved_count",     len(achieved), t)
        logger.add_scalar("mc2d/sig_edges",          sig_edges, t)
        logger.add_scalar(f"mc2d/succ/{var_names[sid]}", float(success), t)

        # ── CSV ──────────────────────────────────────────────────────────────
        csv_w.writerow(dict(
            step=t + 1, explore=int(explore), phase=rec["phase"],
            skill=sname, success=int(success),
            ig=f"{ig:.6f}", pcg_entropy=f"{h:.4f}",
            shd=shd_val, causal_prec=f"{cprec:.4f}", causal_rec=f"{crec:.4f}",
            int_precision=f"{int_prec:.4f}",
            sig_edges=sig_edges, achieved_count=len(achieved),
        ))

        # ── Console row ───────────────────────────────────────────────────────
        if (t + 1) % args.print_every == 0:
            exp_flag = "E" if explore else "G"
            print(
                f"[{t+1:04d}|{exp_flag}] {rec['phase']:<8} {sname:<16}"
                f" succ={int(success)}"
                f" H={h:5.2f} IG={ig:.4f}"
                f" SHD={shd_val:3d} P={cprec:.2f}/R={crec:.2f}"
                f" ach={len(achieved)}/{M}"
                f" t={time.time()-t0:.0f}s"
            )

        # ── PCG matrix printout ───────────────────────────────────────────────
        if args.pcg_every > 0 and (t + 1) % args.pcg_every == 0:
            print(f"\n  ── PCG edge probs @ step {t+1}  (threshold=0.5, *=edge predicted) ──")
            print(pcg_ascii(probs))
            discovered = [(VAR_CODES[i], VAR_CODES[j]) for (i, j) in GROUND_TRUTH_EDGES
                          if probs[i, j] >= 0.5]
            print(f"  GT edges found: {len(discovered)}/{len(GROUND_TRUTH_EDGES)}"
                  + (f"  {discovered}" if discovered else "  (none yet)"))
            print(f"  SIG edges ({sig_edges}): ", end="")
            for u, vs in sig.edges.items():
                for v in vs:
                    if u < M and v < M:
                        print(f"{var_names[u]}→{var_names[v]}", end="  ")
            print("\n")

    # ── Final summary ─────────────────────────────────────────────────────────
    probs = np.array(pcg.probs) if hasattr(pcg, "probs") else np.full((M, M), 0.5)
    np.fill_diagonal(probs, 0.0)
    shd_f             = shd(probs, A_gt)
    cprec_f, crec_f, tp_f, fp_f, fn_f = precision_recall(probs, A_gt)
    h_f               = pcg.entropy() if hasattr(pcg, "entropy") else float("nan")
    int_prec_f        = total_succ / max(1, total_int)

    print(f"\n{'='*72}")
    print(f"  FINAL SUMMARY  (steps={args.steps}  PCG={args.pcg}  seed={args.seed})")
    print(f"{'='*72}")
    print(f"  PCG entropy (final)      : {h_f:.4f}")
    print(f"  SHD vs ground truth      : {shd_f}  (0 = perfect)")
    print(f"  Edge precision / recall  : {cprec_f:.3f} / {crec_f:.3f}"
          f"  (TP={tp_f} FP={fp_f} FN={fn_f})")
    print(f"  Intervention precision   : {total_succ}/{total_int} = {int_prec_f:.3f}")
    print(f"  Resources achieved       : {len(achieved)}/{M}")
    print(f"    {[var_names[i] for i in sorted(achieved)]}")

    print(f"\n  Chain milestones (first acquisition):")
    for vname in var_names:
        ms = milestones[vname]
        print(f"    {vname:<16}: {'step ' + str(ms) if ms else 'not reached'}")

    print(f"\n  Per-skill success rates:")
    for i, vname in enumerate(var_names):
        att = skill_attempts.get(i, 0)
        suc = skill_successes.get(i, 0)
        bar = "#" * int(20 * suc / max(1, att))
        print(f"    {vname:<16}: {suc:3d}/{att:3d}  {bar}")

    print(f"\n  Final PCG edge probabilities  (* = predicted edge ≥ 0.5)")
    print(pcg_ascii(probs))

    print(f"\n  Ground-truth edges discovered:")
    for (i, j) in GROUND_TRUTH_EDGES:
        p = probs[i, j]
        tick = "✓" if p >= 0.5 else "✗"
        print(f"    {tick} {var_names[i]:>14} → {var_names[j]:<14}  p={p:.3f}")

    # ── Save JSON ────────────────────────────────────────────────────────────
    json_path = os.path.join(args.logdir, "final_pcg.json")
    with open(json_path, "w") as fh:
        json.dump({
            "var_names": var_names,
            "probs": probs.tolist(),
            "ground_truth_edges": GROUND_TRUTH_EDGES,
            "shd": shd_f,
            "causal_precision": cprec_f,
            "causal_recall": crec_f,
            "tp": tp_f, "fp": fp_f, "fn": fn_f,
            "pcg_entropy_final": h_f,
            "intervention_precision": int_prec_f,
            "achieved": sorted(achieved),
            "achieved_names": [var_names[i] for i in sorted(achieved)],
            "milestones": {k: v for k, v in milestones.items()},
            "args": vars(args),
        }, fh, indent=2)

    print(f"\n  CSV    : {metrics_path}")
    print(f"  JSON   : {json_path}")
    print(f"  TBoard : tensorboard --logdir {args.logdir}")
    print(f"{'='*72}\n")

    logger.flush()
    logger.close()
    csv_fh.close()


if __name__ == "__main__":
    main()
