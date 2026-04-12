#!/usr/bin/env python3
# scripts/experiment5_causalworld_adapt.py
"""
Experiment 5: T0 -> T1/T2 Adaptive Retraining (Claim 4)

Tests whether DIA discriminates structural change (T1) from motor change (T2)
during online adaptation from a T0 baseline checkpoint.

Phase 1 (T0, phase1_steps):
    Train DIA on CausalWorld T0 (small obstacle).
    Save PCG checkpoint and learned SIG.

Phase 2 (adaptation, phase2_steps):
    Apply T1 or T2 intervention.
    Continue from Phase 1 checkpoint.
    Track entropy spikes, edge changes, and SHD deltas.

Usage:
    python experiment5_causalworld_adapt.py \
        --condition T1 --seed 0 \
        --phase1_steps 5000 --phase2_steps 2000 \
        --out results/logs/exp5_T1_seed0.json

Output JSON:
    {
      "condition": "T1",
      "seed": 0,
      "phase1_shd": float,
      "phase2_shd": float,
      "delta_shd": float,
      "steps_to_detect": int|null,
      "structural_detected": bool,
      "grasped_success_before": float,
      "grasped_success_after": float,
      "completed": true
    }
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# ── CausalWorld import (graceful fallback) ────────────────────────────────────
_env_ctor = None
try:
    from causal_world.envs.causalworld import CausalWorld  # type: ignore
    _env_ctor = CausalWorld
except Exception:
    try:
        from causal_world.envs import CausalWorld  # type: ignore
        _env_ctor = CausalWorld
    except Exception:
        _env_ctor = None

_task_ctor = None
try:
    from causal_world.task_generators import generate_task  # type: ignore
    _task_ctor = generate_task
except Exception:
    _task_ctor = None

# ── DIA imports ───────────────────────────────────────────────────────────────
from dia.evgs_causalworld import wrap_causalworld_env, CausalWorldAdapterConfig
from dia.evgs_adapters import make_causalworld_evgs
from dia.evgs import EVGS
from dia.sig import SIGraph, Skill
from dia.types import Subgoal, Predicate
from dia.planner import PlannerConfig, InterventionSelector
from dia.rollout import DIARunner, RunnerConfig
from dia.logging_utils import TBLogger
from dia.checkpoint import save_checkpoint, load_checkpoint

from dia.pcg_learner import DifferentiablePCG, DifferentiablePCGConfig
from dia.options import RandomOption, OptionConfig

from dia.eval.pcg_metrics import shd as eval_shd


# ── Ground-truth edge definitions ─────────────────────────────────────────────
# Variables (5): grasped(0)  target_lifted(1)  target_high_lifted(2)
#                target_above_goal(3)  task_success(4)
#
# T0 (small obstacle h=0.02):
#   Normal pick-and-place chain. target_high_lifted is NOT required.
#   grasped -> target_lifted -> target_above_goal -> task_success
#
# T1 (large obstacle h=0.10, structural change):
#   Block must clear the obstacle to reach goal. target_high_lifted becomes
#   a required intermediate: its causal path inserts into the chain.
#   grasped -> target_lifted -> target_high_lifted -> target_above_goal -> task_success
#
# T2 (large tool_block, motor change):
#   Same causal structure as T0; only the grasp motor policy is harder
#   (bigger block is more difficult to grasp reliably).
#   grasped -> target_lifted -> target_above_goal -> task_success

GT_EDGES: Dict[str, List[Tuple[int, int]]] = {
    # (src_idx, dst_idx) using the 5-variable ordering above
    "T0": [(0, 1), (1, 3), (3, 4)],           # no high_lifted needed
    "T1": [(0, 1), (1, 2), (2, 3), (3, 4)],   # high_lifted required
    "T2": [(0, 1), (1, 3), (3, 4)],           # same structure as T0
}

# Variable index for target_high_lifted (the structurally differentiating variable)
IDX_HIGH_LIFTED = 2
# Variable index for grasped skill (motor-change affected)
IDX_GRASPED = 0

# Interventions mirrored from train_causalworld_dia.py
INTERVENTIONS = {
    "T0": {"obstacle": {"size": np.array([0.5, 0.015, 0.02])}},
    "T1": {"obstacle": {"size": np.array([0.5, 0.015, 0.10])}},
    "T2": {"tool_block": {"size": np.array([0.085, 0.085, 0.085])}},
}

# Entropy-spike threshold: if |H_new - H_old| > this, record steps_to_detect
ENTROPY_SPIKE_THRESHOLD = 2.0


# ── Env builder ───────────────────────────────────────────────────────────────

def _build_raw_env():
    """Build a fresh CausalWorld pick_and_place env (uninterventioned)."""
    if _env_ctor is None:
        raise RuntimeError(
            "CausalWorld not found. Install 'causalworld' and 'pybullet'."
        )
    if _task_ctor is not None:
        task = _task_ctor(task_generator_id="pick_and_place")
        return _env_ctor(task=task, enable_visualization=False)
    return _env_ctor()


def _apply_intervention(env, condition: str) -> bool:
    """Walk wrapper chain and apply intervention. Returns True on success."""
    intervention = INTERVENTIONS[condition]
    raw_env = env
    while hasattr(raw_env, "env"):
        raw_env = raw_env.env
    if hasattr(raw_env, "do_intervention"):
        raw_env.do_intervention(intervention)
        return True
    # Fallback: try top-level wrapper
    try:
        env.do_intervention(intervention)
        return True
    except Exception:
        return False


def _build_wrapped_env(condition: str):
    """Build CausalWorld env and apply intervention."""
    raw = _build_raw_env()
    env = wrap_causalworld_env(raw, CausalWorldAdapterConfig())
    ok = _apply_intervention(env, condition)
    print(f"[exp5] Applied condition={condition} (success={ok})")
    return env


# ── SIG builder (mirrors train_causalworld_dia.py) ────────────────────────────

def build_cw_sig(evgs: EVGS) -> SIGraph:
    """Build the T0 prior SIG (4-skill chain, no target_high_lifted skill)."""
    names = evgs.names()
    idx = {n: i for i, n in enumerate(names)}
    sig = SIGraph()
    s_g = Skill(skill_id=idx["grasped"],
                subgoal=Subgoal(idx["grasped"], Predicate.UP), name="grasp↑")
    s_l = Skill(skill_id=idx["target_lifted"],
                subgoal=Subgoal(idx["target_lifted"], Predicate.UP), name="lifted↑")
    s_a = Skill(skill_id=idx["target_above_goal"],
                subgoal=Subgoal(idx["target_above_goal"], Predicate.UP), name="above_goal↑")
    s_s = Skill(skill_id=idx["task_success"],
                subgoal=Subgoal(idx["task_success"], Predicate.UP), name="success↑")
    for s in (s_g, s_l, s_a, s_s):
        sig.add_skill(s)
    sig.add_prerequisite(s_g.skill_id, s_l.skill_id)
    sig.add_prerequisite(s_l.skill_id, s_a.skill_id)
    sig.add_prerequisite(s_a.skill_id, s_s.skill_id)
    return sig


# ── PCG builder ───────────────────────────────────────────────────────────────

def _build_pcg(num_vars: int, seed: int = 0, pcg_epochs: int = 200):
    return DifferentiablePCG(
        DifferentiablePCGConfig(num_vars=num_vars, max_iter=pcg_epochs, lr=5e-3, verbose=False)
    )


# ── Option factory ────────────────────────────────────────────────────────────

def _make_option_factory(env, action_space):
    def option_factory(skill: Skill):
        return RandomOption(
            skill.subgoal,
            OptionConfig(max_steps=200, terminate_on_success=True),
            action_space,
        )
    return option_factory


# ── Runner builder ────────────────────────────────────────────────────────────

def _build_runner(
    env,
    evgs: EVGS,
    pcg,
    sig: SIGraph,
    logdir: str,
    log_prefix: str,
    min_buffer: int = 256,
    fit_every: int = 10,
    pcg_epochs: int = 200,
    buffer_recent: int = 1024,
) -> DIARunner:
    selector = InterventionSelector(pcg, sig, PlannerConfig())
    logger = TBLogger(logdir)
    rcfg = RunnerConfig(
        buffer_size=50_000,
        min_buffer=min_buffer,
        batch_recent=buffer_recent,
        fit_every=fit_every,
        pcg_epochs=pcg_epochs,
        log_prefix=log_prefix,
        option_max_steps=200,
        terminate_on_success=True,
        auto_expand_sig=True,
        add_threshold=0.75,
        remove_threshold=0.55,
    )
    option_factory = _make_option_factory(env, env.action_space)
    return DIARunner(
        env, evgs, pcg, sig, selector, rcfg,
        logger=logger, option_factory=option_factory,
    )


# ── Training loop with optional entropy-spike tracking ───────────────────────

def _run_phase(
    runner: DIARunner,
    evgs: EVGS,
    n_steps: int,
    task_goal: Subgoal,
    track_entropy_spike: bool = False,
    entropy_threshold: float = ENTROPY_SPIKE_THRESHOLD,
    prev_entropy: Optional[float] = None,
) -> Tuple[Dict, Optional[int], float]:
    """
    Run `n_steps` DIA steps.

    Returns:
        (last_rec, steps_to_detect, final_entropy)
        steps_to_detect: step index (1-indexed) at which first entropy spike was
            detected, or None if no spike observed.
        final_entropy: PCG entropy at end of phase.
    """
    achieved: List[int] = []
    steps_to_detect: Optional[int] = None
    last_rec: Dict = {}
    last_entropy = prev_entropy  # for spike detection

    name_to_idx = {n: i for i, n in enumerate(evgs.names())}

    for t in range(n_steps):
        rec = runner.step(achieved, task_goal=task_goal)
        last_rec = rec

        if rec.get("success") and rec.get("skill_id") not in achieved:
            achieved.append(rec["skill_id"])

        current_entropy = rec.get("pcg_entropy", float("nan"))

        # Entropy-spike detection
        if (
            track_entropy_spike
            and steps_to_detect is None
            and rec.get("did_fit_pcg", False)
            and last_entropy is not None
            and not np.isnan(last_entropy)
            and not np.isnan(current_entropy)
        ):
            delta_h = abs(current_entropy - last_entropy)
            if delta_h > entropy_threshold:
                steps_to_detect = t + 1  # 1-indexed
                print(f"[exp5] Entropy spike detected at step {steps_to_detect}: "
                      f"delta_H={delta_h:.4f} (threshold={entropy_threshold})")

        if rec.get("did_fit_pcg", False):
            last_entropy = current_entropy

        if (t + 1) % 50 == 0:
            print(
                f"[exp5] [{t+1:04d}/{n_steps}] "
                f"phase={rec['phase']} skill={rec['skill_name']:<12} "
                f"succ={int(rec['success'])} H={current_entropy:.4f} "
                f"buf={rec['buffer_size']}"
            )

    final_entropy = float(last_rec.get("pcg_entropy", float("nan")))
    return last_rec, steps_to_detect, final_entropy


# ── Structural-change detection logic ────────────────────────────────────────

def _check_structural_detected(
    condition: str,
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    threshold: float = 0.5,
    edge_change_threshold: float = 0.15,
) -> bool:
    """
    T1: structural_detected = True if edges involving target_high_lifted (idx=2)
        changed significantly (|p_after - p_before| > edge_change_threshold for
        at least one edge incident to var 2).
    T2: structural_detected = False if edges to/from target_high_lifted did NOT
        change significantly (all |delta| < edge_change_threshold).
    """
    j = IDX_HIGH_LIFTED
    m = probs_before.shape[0]
    deltas = []
    for i in range(m):
        if i == j:
            continue
        deltas.append(abs(float(probs_after[i, j]) - float(probs_before[i, j])))
        deltas.append(abs(float(probs_after[j, i]) - float(probs_before[j, i])))

    max_delta = max(deltas) if deltas else 0.0
    print(
        f"[exp5] high_lifted edge max-delta={max_delta:.4f} "
        f"(threshold={edge_change_threshold})"
    )

    if condition == "T1":
        # Structural: expect edge delta at high_lifted
        return max_delta > edge_change_threshold
    else:
        # T2 motor: expect NO structural edge change
        return max_delta <= edge_change_threshold


# ── Grasped skill success rate helper ────────────────────────────────────────

def _get_grasped_success_rate(sig: SIGraph) -> float:
    sk = sig.skills.get(IDX_GRASPED)
    if sk is None:
        return float("nan")
    return float(sk.success_rate)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="DIA Exp5: T0->T1/T2 adaptive retraining")
    ap.add_argument("--condition", type=str, required=True, choices=["T1", "T2"],
                    help="Intervention condition to adapt to (T1=structural, T2=motor)")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for reproducibility")
    ap.add_argument("--phase1_steps", type=int, default=5000,
                    help="Steps for Phase 1 T0 baseline training")
    ap.add_argument("--phase2_steps", type=int, default=2000,
                    help="Steps for Phase 2 adaptation after intervention")
    ap.add_argument("--fit_every", type=int, default=10,
                    help="Fit PCG every N option executions")
    ap.add_argument("--pcg_epochs", type=int, default=200,
                    help="PCG optimization steps per fit")
    ap.add_argument("--buffer_recent", type=int, default=1024,
                    help="Recent buffer size for PCG fitting")
    ap.add_argument("--min_buffer", type=int, default=256,
                    help="Minimum buffer size before first PCG fit")
    ap.add_argument("--entropy_threshold", type=float, default=ENTROPY_SPIKE_THRESHOLD,
                    help="Entropy-spike detection threshold (|H_new - H_old|)")
    ap.add_argument("--logdir", type=str, default="runs/exp5",
                    help="TensorBoard log directory")
    ap.add_argument("--out", type=str, default=None,
                    help="Output JSON path. Defaults to results/logs/exp5_{condition}_seed{seed}.json")
    args = ap.parse_args()

    # Resolve output path
    if args.out is None:
        args.out = f"results/logs/exp5_{args.condition}_seed{args.seed}.json"

    # Reproducibility
    np.random.seed(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[exp5] === Experiment 5: T0 -> {args.condition} adaptation ===")
    print(f"[exp5] seed={args.seed}  phase1={args.phase1_steps}  phase2={args.phase2_steps}")

    # ── Phase 1: T0 baseline ──────────────────────────────────────────────────
    print("\n[exp5] --- Phase 1: T0 baseline training ---")

    env_t0 = _build_wrapped_env("T0")
    evgs = make_causalworld_evgs()
    M = len(evgs.names())
    name_to_idx = {n: i for i, n in enumerate(evgs.names())}
    goal_idx = name_to_idx["task_success"]
    task_goal = Subgoal(var_index=goal_idx, predicate=Predicate.UP)

    pcg_p1 = _build_pcg(M, seed=args.seed, pcg_epochs=args.pcg_epochs)
    sig_p1 = build_cw_sig(evgs)

    runner_p1 = _build_runner(
        env_t0, evgs, pcg_p1, sig_p1,
        logdir=f"{args.logdir}/{args.condition}_seed{args.seed}/phase1",
        log_prefix="p1",
        min_buffer=args.min_buffer,
        fit_every=args.fit_every,
        pcg_epochs=args.pcg_epochs,
        buffer_recent=args.buffer_recent,
    )

    _run_phase(runner_p1, evgs, args.phase1_steps, task_goal,
               track_entropy_spike=False)

    # Capture Phase 1 metrics
    probs_p1 = np.array(pcg_p1.probs).copy()
    phase1_shd = float(eval_shd(probs_p1, GT_EDGES["T0"]))
    phase1_entropy = float(pcg_p1.entropy()) if hasattr(pcg_p1, "entropy") else float("nan")
    grasped_success_before = _get_grasped_success_rate(sig_p1)

    print(f"[exp5] Phase 1 done: SHD(T0-GT)={phase1_shd}  H={phase1_entropy:.4f}  "
          f"grasped_sr={grasped_success_before:.4f}")

    # Save Phase 1 checkpoint
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        ckpt_path = f.name
    save_checkpoint(ckpt_path, pcg_p1, sig_p1, metadata={"condition": "T0", "seed": args.seed})
    print(f"[exp5] Phase 1 checkpoint saved to {ckpt_path}")

    # ── Phase 2: adaptation under T1 or T2 ───────────────────────────────────
    print(f"\n[exp5] --- Phase 2: adaptation under {args.condition} ---")

    # Build a fresh env and apply the target intervention
    env_t_x = _build_wrapped_env(args.condition)
    evgs2 = make_causalworld_evgs()   # fresh extractor (same config)

    pcg_p2 = _build_pcg(M, seed=args.seed, pcg_epochs=args.pcg_epochs)
    sig_p2 = build_cw_sig(evgs2)

    # Restore Phase 1 state
    load_checkpoint(ckpt_path, pcg_p2, sig_p2)
    os.unlink(ckpt_path)
    print("[exp5] Phase 1 checkpoint restored into Phase 2 runner")

    runner_p2 = _build_runner(
        env_t_x, evgs2, pcg_p2, sig_p2,
        logdir=f"{args.logdir}/{args.condition}_seed{args.seed}/phase2",
        log_prefix="p2",
        min_buffer=args.min_buffer,
        fit_every=args.fit_every,
        pcg_epochs=args.pcg_epochs,
        buffer_recent=args.buffer_recent,
    )

    _last_rec, steps_to_detect, phase2_entropy = _run_phase(
        runner_p2, evgs2, args.phase2_steps, task_goal,
        track_entropy_spike=True,
        entropy_threshold=args.entropy_threshold,
        prev_entropy=phase1_entropy,
    )

    # Phase 2 metrics
    probs_p2 = np.array(pcg_p2.probs).copy()
    # Evaluate against the GT of the new condition
    phase2_shd = float(eval_shd(probs_p2, GT_EDGES[args.condition]))
    delta_shd = phase2_shd - phase1_shd
    grasped_success_after = _get_grasped_success_rate(sig_p2)

    structural_detected = _check_structural_detected(
        args.condition, probs_p1, probs_p2
    )

    print(f"\n[exp5] === Results for condition={args.condition} seed={args.seed} ===")
    print(f"  phase1_shd              : {phase1_shd}")
    print(f"  phase2_shd              : {phase2_shd}")
    print(f"  delta_shd               : {delta_shd:+.1f}")
    print(f"  steps_to_detect         : {steps_to_detect}")
    print(f"  structural_detected     : {structural_detected}")
    print(f"  grasped_success_before  : {grasped_success_before:.4f}")
    print(f"  grasped_success_after   : {grasped_success_after:.4f}")

    result = {
        "condition": args.condition,
        "seed": args.seed,
        "phase1_steps": args.phase1_steps,
        "phase2_steps": args.phase2_steps,
        "phase1_shd": phase1_shd,
        "phase2_shd": phase2_shd,
        "delta_shd": delta_shd,
        "steps_to_detect": steps_to_detect,
        "structural_detected": structural_detected,
        "grasped_success_before": grasped_success_before,
        "grasped_success_after": grasped_success_after,
        "phase1_entropy": phase1_entropy,
        "phase2_entropy": phase2_entropy,
        "completed": True,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[exp5] Results written to {out_path}")

    return result


if __name__ == "__main__":
    main()
