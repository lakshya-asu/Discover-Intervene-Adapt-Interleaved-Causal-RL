#!/usr/bin/env python3
"""
scripts/run_transfer_minedojo.py  —  DIA Claim 3: 2D PCG + SIG transfer to 3D MineDojo.

MineDojo port of run_transfer_3d.py (MineRL).  Evaluates whether causal
structure learned cheaply in 2D symbolic Minecraft can guide skill sequencing
in a 3D MineDojo open-ended environment, reducing the number of 3D steps
needed to reach diamond.

Conditions
----------
  --mode baseline
      Empty SIG, uniform PCG prior (0.05), SkillScriptedOption fallback.
      No transfer knowledge.

  --mode transfer
      Load PCG edge-probs from --pcg_path  (pcg_2d.npy or similar).
      Load BC warm-started policies from --bc_dir (pretrain_bc_minerl.py output).
      Build SIG from hardcoded 2D causal graph (same as train_minecraft2d_dia.py).
      Online PCG fine-tune from 3D interventional transitions.

Key measurement
---------------
  steps_to_diamond_3d — total env steps until diamond variable > 0.
  Logged to --out as JSON; --dry_run skips all env steps for import/init testing.

Env creation strategy
---------------------
  Primary:  minedojo.make(task_id="open-ended", image_size=(64, 64))
  Fallback: gym.make("MineRLObtainDiamondShovel-v0")  (if MineDojo unavailable)
  Both are wrapped with MinedojoObsWrapper / MineRLObsWrapper respectively.

Usage
-----
  # Transfer condition, seed 0:
  conda run -n dia-minecraft python scripts/run_transfer_minedojo.py \\
      --mode transfer --seed 0 \\
      --pcg_path pcg_2d.npy \\
      --bc_dir data/minerl_policies \\
      --out results/logs/transfer3d_transfer_seed0.json

  # Baseline condition, seed 0:
  conda run -n dia-minecraft python scripts/run_transfer_minedojo.py \\
      --mode baseline --seed 0 \\
      --out results/logs/transfer3d_baseline_seed0.json

  # Dry-run (no Minecraft server needed): verify imports + init
  conda run -n dia-minecraft python scripts/run_transfer_minedojo.py \\
      --mode transfer --seed 0 --dry_run \\
      --pcg_path pcg_2d.npy \\
      --bc_dir /tmp/minerl_policies_test \\
      --out /tmp/transfer_minedojo_test.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

# ── Repo path setup ─────────────────────────────────────────────────────────
_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, _REPO_ROOT)

from dia.evgs_minedojo import make_minedojo_evgs, VAR_NAMES
from dia.sig import SIGraph, Skill
from dia.types import Subgoal, Predicate
from dia.pcg import SimplePCG, PCGConfig
from dia.options import OptionConfig
from dia.options_minedojo import (
    MinedojoObsWrapper,
    MineDojoObsWrapper,           # alias — same class
    InventoryConditionedCraftOption,
    BCOptionWrapper,
    load_bc_option,
    CRAFT_SKILLS,
)
from dia.options_minerl import (
    SkillScriptedOption,
    MineRLObsWrapper,
)

# ── Variable index mapping ───────────────────────────────────────────────────
_VAR_IDX: Dict[str, int] = {n: i for i, n in enumerate(VAR_NAMES)}
DIAMOND_IDX: int = _VAR_IDX["diamond"]

# ── Env ID reported in JSON output ──────────────────────────────────────────
_MINEDOJO_ENV_ID  = "minedojo-open-ended"
_MINERL_FALLBACK_ENVS = [
    "MineRLObtainDiamondShovel-v0",
    "MineRLObtainDiamond-v0",
    "MineRLObtainIronPickaxeDense-v0",
]


# ---------------------------------------------------------------------------
# Build 2D-derived SIG (same prerequisite graph as 2D symbolic training)
# ---------------------------------------------------------------------------

def build_transfer_sig() -> SIGraph:
    """
    Reconstruct the known Minecraft prerequisite SIG.

    This is the same graph hardcoded in train_minecraft2d_dia.py.
    The edges represent what the 2D DIA agent discovers (used as ground truth
    for the transfer condition).
    """
    sig = SIGraph()
    for var_idx, var_name in enumerate(VAR_NAMES):
        sg = Subgoal(var_index=var_idx, predicate=Predicate.UP)
        sig.add_skill(Skill(skill_id=var_idx, subgoal=sg, name=f"{var_name}+"))

    idx = _VAR_IDX
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


def build_empty_sig() -> SIGraph:
    """Baseline SIG with no prerequisite edges (empty prior)."""
    sig = SIGraph()
    for var_idx, var_name in enumerate(VAR_NAMES):
        sg = Subgoal(var_index=var_idx, predicate=Predicate.UP)
        sig.add_skill(Skill(skill_id=var_idx, subgoal=sg, name=f"{var_name}+"))
    return sig


# ---------------------------------------------------------------------------
# Execution plan from SIG topology
# ---------------------------------------------------------------------------

def topo_order_from_sig(sig: SIGraph) -> List[int]:
    """Topological execution order; falls back to natural index order on cycle."""
    try:
        order = sig.toposort()
    except Exception:
        order = sorted(sig.skills.keys())
    return order


# ---------------------------------------------------------------------------
# Online PCG update from 3D interventional transitions
# ---------------------------------------------------------------------------

def update_pcg_from_trajectory(
    pcg: SimplePCG,
    trajectory: list,
    skill_var_idx: int,
    evgs,
    lr: float = 0.05,
) -> None:
    """
    Lightweight online PCG update using interventional transitions.

    For each transition (x_t, x_{t+1}) where skill j was the intervened variable:
    - edge (i -> j) is strengthened if x_t[i] > 0 and x_{t+1}[j] > x_t[j]
    - edge (i -> j) is weakened   if x_t[i] > 0 and x_{t+1}[j] unchanged
    Only updates for j == skill_var_idx since that was the executed skill.
    """
    if not trajectory:
        return
    j = skill_var_idx
    delta = np.zeros_like(pcg.probs)

    for obs, _act, next_obs, _succ in trajectory:
        try:
            x_t   = evgs.extract(obs)
            x_tp1 = evgs.extract(next_obs)
        except Exception:
            continue

        effect = float(x_tp1[j]) - float(x_t[j])
        if abs(effect) < 1e-6:
            continue   # no change in target variable

        for i in range(len(VAR_NAMES)):
            if i == j:
                continue
            if float(x_t[i]) > 0.5:
                delta[i, j] += effect

    n = max(1, len(trajectory))
    pcg.conservative_update(delta / n, lr=lr)


# ---------------------------------------------------------------------------
# Environment creation with MineDojo primary, MineRL fallback
# ---------------------------------------------------------------------------

def make_env_minedojo(evgs) -> tuple[Any, str]:
    """
    Create and wrap a 3D Minecraft environment.

    Attempts MineDojo first; falls back to MineRL if MineDojo is unavailable.

    Returns
    -------
    (wrapped_env, env_id_string)
      wrapped_env — MinedojoObsWrapper or MineRLObsWrapper
      env_id_string — human-readable ID for JSON output
    """
    # ── Try MineDojo ─────────────────────────────────────────────────────────
    try:
        import minedojo
        raw = minedojo.make(task_id="open-ended", image_size=(64, 64))
        wrapped = MinedojoObsWrapper(raw, evgs)
        return wrapped, _MINEDOJO_ENV_ID
    except Exception as md_exc:
        print(f"  [env] MineDojo unavailable ({md_exc}); trying MineRL fallback")

    # ── Try MineRL fallback ───────────────────────────────────────────────────
    try:
        import gym
        import minerl  # noqa: registers MineRL envs
        for env_id in _MINERL_FALLBACK_ENVS:
            try:
                raw = gym.make(env_id)
                wrapped = MineRLObsWrapper(raw, evgs)
                print(f"  [env] MineRL fallback: {env_id}")
                return wrapped, env_id
            except Exception:
                continue
    except Exception as rl_exc:
        print(f"  [env] MineRL also unavailable: {rl_exc}")

    raise RuntimeError(
        "No 3D Minecraft environment available.  "
        "Install MineDojo (pip install minedojo) or MineRL and ensure a "
        "Minecraft server can start.  Use --dry_run for import-only testing."
    )


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(args) -> Dict[str, Any]:
    """
    Core experiment.  Returns result dict regardless of --dry_run.
    In dry-run mode no env steps are taken; the result contains metadata only.
    """
    np.random.seed(args.seed)

    M    = len(VAR_NAMES)
    evgs = make_minedojo_evgs()

    # ── Build PCG ──────────────────────────────────────────────────────────
    pcg = SimplePCG(PCGConfig(num_vars=M, init_edge_prob=0.05, seed=args.seed))
    if args.mode == "transfer" and args.pcg_path and os.path.exists(args.pcg_path):
        probs = np.load(args.pcg_path).astype(float)
        np.fill_diagonal(probs, 0.0)
        pcg.state.edge_probs = np.clip(probs, 0.0, 1.0)
        mask = np.ones((M, M), dtype=bool)
        np.fill_diagonal(mask, False)
        n_strong = int((pcg.probs[mask] > 0.5).sum())
        print(f"PCG loaded from {args.pcg_path}  "
              f"(entropy={pcg.entropy():.3f}, edges>0.5: {n_strong})")
    else:
        if args.mode == "transfer":
            print(f"PCG file not found ({args.pcg_path}); using uniform init")
        else:
            print("Baseline mode: uniform PCG prior")

    # ── Build SIG ──────────────────────────────────────────────────────────
    if args.mode == "transfer":
        sig = build_transfer_sig()
        print(f"Transfer SIG: {len(sig.skills)} skills, "
              f"{sum(len(v) for v in sig.edges.values())} edges")
    else:
        sig = build_empty_sig()
        print("Baseline SIG: empty (no transfer edges)")

    # ── Execution plan ────────────────────────────────────────────────────
    plan = topo_order_from_sig(sig)
    print(f"Execution plan: {[VAR_NAMES[i] for i in plan]}")

    # ── Load skill options ────────────────────────────────────────────────
    options: Dict[int, Any] = {}

    for var_idx, var_name in enumerate(VAR_NAMES):
        sg  = Subgoal(var_index=var_idx, predicate=Predicate.UP)
        cfg = OptionConfig(max_steps=args.max_steps_per_skill, terminate_on_success=True)

        if args.mode == "transfer":
            if var_name in CRAFT_SKILLS:
                opt = InventoryConditionedCraftOption(sg, cfg, var_name)
                print(f"  [{var_name}] craft option (deterministic)")
            else:
                pt_path = os.path.join(args.bc_dir, f"{var_name}.pt")
                opt     = load_bc_option(var_name, pt_path, cfg)
                if opt is None:
                    opt = SkillScriptedOption(sg, cfg, var_name)
                    if args.verbose:
                        print(f"  [{var_name}] BC not found — using SkillScriptedOption")
        else:
            # Baseline: scripted options only
            opt = SkillScriptedOption(sg, cfg, var_name)
            if args.verbose:
                print(f"  [{var_name}] SkillScriptedOption (baseline)")

        options[var_idx] = opt

    # ── Dry run: stop here ────────────────────────────────────────────────
    if args.dry_run:
        print("\n[DRY RUN] Init complete. Skipping env steps.")
        result = {
            "mode":          args.mode,
            "seed":          args.seed,
            "dry_run":       True,
            "pcg_entropy":   float(pcg.entropy()),
            "plan":          [VAR_NAMES[i] for i in plan],
            "options_ready": {VAR_NAMES[i]: type(options[i]).__name__ for i in plan},
            "steps_to_diamond_3d": None,
            "status":        "dry_run_ok",
        }
        return result

    # ── Create environment ────────────────────────────────────────────────
    print(f"\nCreating 3D Minecraft env (MineDojo primary, MineRL fallback)...")
    try:
        env, env_id = make_env_minedojo(evgs)
    except Exception as exc:
        print(f"ERROR: could not create any 3D env: {exc}")
        return {
            "mode":                args.mode,
            "seed":                args.seed,
            "env_id":              "none",
            "error":               str(exc),
            "steps_to_diamond_3d": None,
        }

    print(f"  [env] created: {env_id}")

    # ── Reset ─────────────────────────────────────────────────────────────
    obs = env.reset()
    global_step                = 0
    achieved: List[int]        = []
    steps_to_diamond_3d: Optional[int] = None
    skill_log: List[Dict]      = []

    print(f"\n{'='*60}")
    print(f"DIA 3D MineDojo Transfer Experiment  mode={args.mode}  seed={args.seed}")
    print(f"{'='*60}")

    t_start = time.time()

    for skill_id in plan:
        var_idx  = skill_id
        var_name = VAR_NAMES[var_idx]
        option   = options[var_idx]

        # Check if already acquired as side effect of a prior skill
        x_now = evgs.extract(env.get_obs())
        if x_now[var_idx] > 0.5:
            print(f"  [{var_name}] already acquired — skipping")
            achieved.append(var_idx)
            if var_idx == DIAMOND_IDX and steps_to_diamond_3d is None:
                steps_to_diamond_3d = global_step
            continue

        print(f"  [{var_name}] executing  (max {args.max_steps_per_skill} steps, "
              f"option={type(option).__name__})")
        out     = option.run(env, evgs)
        success = out["success"]
        steps   = out["steps"]
        global_step += steps

        skill_log.append({
            "skill":     var_name,
            "success":   bool(success),
            "steps":     int(steps),
            "cum_steps": int(global_step),
        })

        if success:
            achieved.append(var_idx)
            print(f"  [{var_name}] SUCCESS  ({steps} steps, total={global_step})")
            if var_idx == DIAMOND_IDX and steps_to_diamond_3d is None:
                steps_to_diamond_3d = global_step
        else:
            print(f"  [{var_name}] FAILED   ({steps} steps, total={global_step})")

        # ── Reset env if episode ended mid-sequence ────────────────────────
        if out.get("episode_done"):
            print(f"  [reset] episode terminated after {var_name} — resetting env")
            try:
                env.reset()
            except Exception as _e:
                print(f"  [reset] warning: {_e}")

        # ── Online PCG fine-tune from 3D transitions (transfer mode only) ──
        if args.mode == "transfer" and out.get("trajectory"):
            update_pcg_from_trajectory(
                pcg=pcg,
                trajectory=out["trajectory"],
                skill_var_idx=var_idx,
                evgs=evgs,
                lr=0.05,
            )

        # Early exit if all 9 skills achieved
        if len(set(achieved)) >= len(VAR_NAMES):
            break

        # Hard budget: stop if max total steps exceeded
        if global_step >= args.max_total_steps:
            print(f"  [budget] reached max_total_steps={args.max_total_steps}, stopping")
            break

    t_elapsed = time.time() - t_start

    # ── Post-loop: final diamond check ────────────────────────────────────
    try:
        x_final = evgs.extract(env.get_obs())
        if x_final[DIAMOND_IDX] > 0.5 and steps_to_diamond_3d is None:
            steps_to_diamond_3d = global_step
    except Exception:
        pass

    # ── Cleanup ──────────────────────────────────────────────────────────
    try:
        # MinedojoObsWrapper stores env in self.env
        inner = getattr(env, "env", env)
        inner.close()
    except Exception:
        pass

    # ── Final PCG state ───────────────────────────────────────────────────
    strong_edges = [
        (VAR_NAMES[i], VAR_NAMES[j], float(pcg.probs[i, j]))
        for i in range(M) for j in range(M)
        if i != j and pcg.probs[i, j] >= 0.5
    ]

    result: Dict[str, Any] = {
        "mode":                 args.mode,
        "seed":                 args.seed,
        "env_id":               env_id,
        "steps_to_diamond_3d":  steps_to_diamond_3d,
        "total_steps":          int(global_step),
        "skills_achieved":      [VAR_NAMES[i] for i in achieved],
        "n_achieved":           len(set(achieved)),
        "elapsed_s":            round(t_elapsed, 1),
        "skill_log":            skill_log,
        "completed":            len(set(achieved)) >= len(VAR_NAMES),
        "pcg_entropy_final":    float(pcg.entropy()),
        "pcg_strong_edges":     strong_edges,
    }

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> Dict[str, Any]:
    ap = argparse.ArgumentParser(
        description=(
            "DIA Claim 3 transfer experiment — MineDojo backend.\n"
            "  --mode baseline : flat scripted skills, no transfer\n"
            "  --mode transfer : 2D PCG + BC warm-start → guide 3D skill sequencing\n\n"
            "Primary env:  minedojo.make(task_id='open-ended', image_size=(64,64))\n"
            "Fallback env: gym.make('MineRLObtainDiamondShovel-v0')\n"
            "Use --dry_run to verify imports and init without a Minecraft server."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--mode", type=str, default="transfer", choices=["baseline", "transfer"],
        help="Experiment condition: 'baseline' or 'transfer'  (default: transfer)",
    )
    ap.add_argument(
        "--seed", type=int, default=0,
        help="Random seed  (default: 0)",
    )
    ap.add_argument(
        "--pcg_path", type=str, default="pcg_2d.npy",
        help="Path to 2D PCG edge-prob matrix (.npy). Transfer mode only.  "
             "(default: pcg_2d.npy)",
    )
    ap.add_argument(
        "--bc_dir", type=str, default="data/minerl_policies",
        help="Directory containing <skill>.pt BC policy files. Transfer mode only.  "
             "(default: data/minerl_policies)",
    )
    ap.add_argument(
        "--max_steps_per_skill", type=int, default=3000,
        help="Max env steps per skill  (default: 3000)",
    )
    ap.add_argument(
        "--max_total_steps", type=int, default=40_000,
        help="Hard budget on total env steps  (default: 40000)",
    )
    ap.add_argument(
        "--out", type=str, default=None,
        help="JSON output path  "
             "(default: results/logs/transfer3d_{mode}_seed{seed}.json)",
    )
    ap.add_argument(
        "--dry_run", action="store_true",
        help="Verify imports and PCG/policy loading without taking any env steps",
    )
    ap.add_argument(
        "--verbose", action="store_true",
        help="Extra debug logging",
    )
    args = ap.parse_args()

    # Default output path
    if args.out is None:
        args.out = (
            f"results/logs/transfer3d_{args.mode}_seed{args.seed}.json"
        )

    print("run_transfer_minedojo.py")
    print(f"  mode                : {args.mode}")
    print(f"  seed                : {args.seed}")
    print(f"  pcg_path            : {args.pcg_path}")
    print(f"  bc_dir              : {args.bc_dir}")
    print(f"  max_steps_per_skill : {args.max_steps_per_skill}")
    print(f"  max_total_steps     : {args.max_total_steps}")
    print(f"  out                 : {args.out}")
    print(f"  dry_run             : {args.dry_run}")
    print()

    result = run_experiment(args)

    # ── Save result ───────────────────────────────────────────────────────
    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved → {out_path}")

    # ── Print key metric ──────────────────────────────────────────────────
    stps = result.get("steps_to_diamond_3d")
    if stps is not None:
        print(f"steps_to_diamond_3d = {stps}")
    else:
        err = result.get("error")
        if err:
            print(f"ERROR: {err}")
        elif result.get("dry_run"):
            print("Dry run complete. No steps taken.")
        else:
            achieved_names = result.get("skills_achieved", [])
            print(f"Diamond not reached.  Skills achieved: {achieved_names} "
                  f"({result.get('n_achieved', 0)}/9)")

    return result


if __name__ == "__main__":
    main()
