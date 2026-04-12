#!/usr/bin/env python3
"""
scripts/run_transfer_3d.py  —  DIA Claim 3: 2D PCG + SIG transfer to 3D MineRL.

Evaluates whether causal structure learned cheaply in 2D symbolic Minecraft can
guide skill sequencing in 3D (MineRL), reducing the number of 3D steps needed
to reach diamond.

Conditions
----------
  --mode baseline
      Empty SIG, uniform PCG prior (0.05), SkillScriptedOption fallback.
      No transfer knowledge.

  --mode transfer
      Load PCG edge-probs from --pcg_path  (pcg_2d.npy or similar).
      Load BC-warm-started policies from --bc_dir (pretrain_bc_minerl.py output).
      Build SIG from hardcoded 2D causal graph (same as train_minecraft2d_dia.py).
      Online PCG fine-tune from 3D interventional transitions.

Key measurement
---------------
  steps_to_diamond_3d — total env steps until inventory["diamond"] > 0.
  Logged to --out as JSON; --dry_run skips all env steps for import/init testing.

Usage
-----
  # Transfer condition, seed 0:
  conda run -n dia-minecraft python scripts/run_transfer_3d.py \\
      --mode transfer --seed 0 \\
      --pcg_path pcg_2d.npy \\
      --bc_dir data/minerl_policies \\
      --out results/transfer_3d_seed0.json

  # Baseline condition, seed 0:
  conda run -n dia-minecraft python scripts/run_transfer_3d.py \\
      --mode baseline --seed 0 \\
      --out results/baseline_3d_seed0.json

  # Dry-run (no Minecraft server needed): verify imports + init
  conda run -n dia-minecraft python scripts/run_transfer_3d.py \\
      --mode transfer --seed 0 --dry_run \\
      --pcg_path pcg_2d.npy \\
      --bc_dir /tmp/minerl_policies_test \\
      --out /tmp/transfer_test.json
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

from dia.evgs_minerl import make_minerl_evgs, VAR_NAMES
from dia.sig import SIGraph, Skill
from dia.types import Subgoal, Predicate
from dia.pcg import SimplePCG, PCGConfig
from dia.options import OptionConfig
from dia.options_minerl import (
    MineRLObsWrapper,
    MineRLPPOOption,
    SkillScriptedOption,
    N_ACTIONS,
)

# ── Variable index mapping ───────────────────────────────────────────────────
_VAR_IDX: Dict[str, int] = {n: i for i, n in enumerate(VAR_NAMES)}
DIAMOND_IDX: int = _VAR_IDX["diamond"]

# ── MineRL environment name ──────────────────────────────────────────────────
# MineRL / BASALT 2022 available envs (priority order):
_CANDIDATE_ENVS = [
    "MineRLObtainDiamondShovel-v0",
    "MineRLBasaltFindCave-v0",
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

    wood -> stonepickaxe, stone_pickaxe -> ironore, etc.
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
# Load BC-warm-started option policies
# ---------------------------------------------------------------------------

def _load_bc_policy_from_pt(pt_path: str):
    """
    Load a BCPolicy checkpoint from a .pt file.

    The BCPolicy architecture must match pretrain_bc_minerl.py:
      conv(3→32,8×8,s4) → conv(32→64,4×4,s2) → conv(64→64,3×3,s1)
      → flatten → fc(1024→512) → fc(512→n_actions)

    Returns (policy, metadata_dict).
    """
    import torch
    import torch.nn as nn

    class _BCPolicy(nn.Module):
        def __init__(self, n_actions: int = N_ACTIONS) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 512),
                nn.ReLU(),
                nn.Linear(512, n_actions),
            )
            self.n_actions = n_actions

        def forward(self, x):
            return self.fc(self.conv(x))

        def predict(self, obs_hwc: np.ndarray) -> int:
            """obs_hwc: uint8 (H, W, 3)."""
            self.eval()
            with torch.no_grad():
                x = (
                    torch.from_numpy(obs_hwc.astype(np.float32) / 255.0)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                )
                return int(self(x).argmax(dim=-1).item())

    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    policy = _BCPolicy(n_actions=ckpt.get("n_actions", N_ACTIONS))
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    return policy, ckpt


def load_ppo_option(skill: str, zip_path: str, cfg: OptionConfig) -> Optional[Any]:
    """Load a trained SB3 PPO model (.zip) and wrap as MineRLPPOOption.

    Returns None if file is missing or SB3 is not installed.
    """
    if not os.path.exists(zip_path):
        return None
    try:
        var_idx = _VAR_IDX[skill]
        sg = Subgoal(var_index=var_idx, predicate=Predicate.UP)
        opt = MineRLPPOOption(sg, cfg, zip_path)
        print(f"  [{skill}] PPO model loaded from {zip_path}")
        return opt
    except Exception as exc:
        print(f"  [{skill}] WARNING: could not load PPO model from {zip_path}: {exc}")
        return None


def load_bc_option(skill: str, pt_path: str, cfg: OptionConfig) -> Optional[Any]:
    """
    Load a BCPolicy from disk and wrap it as a lightweight OptionPolicy.

    Returns None if the file is missing (caller falls back to SkillScriptedOption).
    """
    if not os.path.exists(pt_path):
        return None
    try:
        var_idx = _VAR_IDX[skill]
        sg = Subgoal(var_index=var_idx, predicate=Predicate.UP)
        policy, meta = _load_bc_policy_from_pt(pt_path)
        acc_str = f"{meta.get('train_acc', 0.0):.3f}" if isinstance(
            meta.get("train_acc"), float) else "?"
        print(f"  [{skill}] BC policy loaded  "
              f"(train_acc={acc_str}  epochs={meta.get('epochs', '?')})")
        return BCOptionWrapper(sg, cfg, policy)
    except Exception as exc:
        print(f"  [{skill}] WARNING: could not load BC policy from {pt_path}: {exc}")
        return None


class BCOptionWrapper:
    """Thin OptionPolicy wrapper around a BCPolicy (no SB3 dependency)."""

    def __init__(self, subgoal: Subgoal, cfg: OptionConfig, bc_policy):
        self.subgoal = subgoal
        self.cfg = cfg
        self._policy = bc_policy

    def act(self, obs: Dict[str, np.ndarray]) -> int:
        """obs: wrapped dict with 'rgb' key (H, W, 3) uint8."""
        rgb = obs.get("rgb")
        if rgb is None or not hasattr(self._policy, "predict"):
            return int(np.random.randint(N_ACTIONS))
        return self._policy.predict(rgb)

    def run(self, env, evgs) -> Dict[str, Any]:
        """Execute until success, timeout, or env done."""
        obs = env.get_obs() if hasattr(env, "get_obs") else {}
        steps = 0
        success = False
        trajectory = []

        while steps < self.cfg.max_steps:
            action = self.act(obs)
            next_obs, _rew, done, _info = env.step(action)

            x_curr = evgs.extract(obs)
            x_next = evgs.extract(next_obs)
            succ = (x_next[self.subgoal.var_index] > x_curr[self.subgoal.var_index])
            trajectory.append((obs, action, next_obs, succ))
            obs = next_obs
            steps += 1

            if succ:
                success = True
                if self.cfg.terminate_on_success:
                    break
            if done:
                break

        return {"success": success, "steps": steps,
                "trajectory": trajectory, "final_obs": obs,
                "episode_done": done}


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
    (Only updates for j == skill_var_idx since that was the executed skill.)
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
            # Observational signal: i present at time t, j changed
            if float(x_t[i]) > 0.5:
                delta[i, j] += effect   # positive if j went up, negative if down

    n = max(1, len(trajectory))
    pcg.conservative_update(delta / n, lr=lr)


# ---------------------------------------------------------------------------
# MineRL environment helpers
# ---------------------------------------------------------------------------

def find_available_env() -> Optional[str]:
    """Return the first available MineRL env ID, or None."""
    try:
        import gym
        import minerl  # noqa: registers envs
        for env_id in _CANDIDATE_ENVS:
            if env_id in gym.envs.registry.env_specs:
                return env_id
        # Fallback: try to find any MineRL env
        for key in gym.envs.registry.env_specs:
            if "MineRL" in key or "minerl" in key:
                return key
        return None
    except Exception:
        return None


def make_env(env_id: str, evgs):
    """Create and wrap a MineRL environment."""
    import gym
    import minerl  # noqa: registers envs
    raw = gym.make(env_id)
    return MineRLObsWrapper(raw, evgs)


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
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(args) -> Dict[str, Any]:
    """
    Core experiment.  Returns result dict regardless of --dry_run.
    In dry-run mode, no env steps are taken; the result contains metadata only.
    """
    np.random.seed(args.seed)

    M = len(VAR_NAMES)
    evgs = make_minerl_evgs()

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

    # ── Load BC skill options (transfer mode only) ─────────────────────────
    opt_cfg = OptionConfig(max_steps=args.max_steps_per_skill, terminate_on_success=True)
    options: Dict[int, Any] = {}

    for var_idx, var_name in enumerate(VAR_NAMES):
        sg = Subgoal(var_index=var_idx, predicate=Predicate.UP)
        cfg = OptionConfig(max_steps=args.max_steps_per_skill, terminate_on_success=True)

        if args.mode == "transfer":
            # Priority: PPO (.zip) > BC (.pt) > scripted macro
            ppo_path = os.path.join(args.ppo_dir, f"skill_{var_name}.zip")
            opt = load_ppo_option(var_name, ppo_path, cfg)
            if opt is None and args.bc_dir:
                pt_path = os.path.join(args.bc_dir, f"{var_name}.pt")
                opt = load_bc_option(var_name, pt_path, cfg)
        else:
            opt = None

        if opt is None:
            opt = SkillScriptedOption(sg, cfg, var_name)
            if args.mode == "baseline" or args.verbose:
                print(f"  [{var_name}] using SkillScriptedOption (macro fallback)")

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

    # ── Find available MineRL env ─────────────────────────────────────────
    env_id = args.env_id or find_available_env()
    if env_id is None:
        print("ERROR: No MineRL environment found in gym registry.")
        print("  Install with:  conda activate dia-minecraft")
        print("  Requires Minecraft server to be running for actual steps.")
        return {
            "mode": args.mode, "seed": args.seed,
            "error": "no_env_found", "steps_to_diamond_3d": None,
        }

    print(f"\nCreating MineRL env: {env_id}")
    try:
        env = make_env(env_id, evgs)
    except Exception as exc:
        print(f"ERROR: could not create env '{env_id}': {exc}")
        return {
            "mode": args.mode, "seed": args.seed,
            "error": str(exc), "steps_to_diamond_3d": None,
        }

    # ── Reset ─────────────────────────────────────────────────────────────
    obs = env.reset()
    global_step = 0
    achieved: List[int] = []
    steps_to_diamond_3d: Optional[int] = None
    skill_log: List[Dict] = []

    print(f"\n{'='*60}")
    print(f"DIA 3D Transfer Experiment  mode={args.mode}  seed={args.seed}")
    print(f"{'='*60}")

    t_start = time.time()

    for skill_id in plan:
        var_idx  = skill_id
        var_name = VAR_NAMES[var_idx]
        option   = options[var_idx]

        # Check if already achieved as a side effect of a prior skill
        x_now = evgs.extract(env.get_obs())
        if x_now[var_idx] > 0.5:
            print(f"  [{var_name}] already acquired — skipping")
            achieved.append(var_idx)
            if var_idx == DIAMOND_IDX and steps_to_diamond_3d is None:
                steps_to_diamond_3d = global_step
            continue

        print(f"  [{var_name}] executing  (max {args.max_steps_per_skill} steps)")
        out     = option.run(env, evgs)
        success = out["success"]
        steps   = out["steps"]
        global_step += steps

        skill_log.append({
            "skill":   var_name,
            "success": bool(success),
            "steps":   int(steps),
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

        # ── Online PCG fine-tune from 3D transitions (transfer mode) ──────
        if args.mode == "transfer" and out.get("trajectory"):
            update_pcg_from_trajectory(
                pcg=pcg,
                trajectory=out["trajectory"],
                skill_var_idx=var_idx,
                evgs=evgs,
                lr=0.05,
            )

        # Early exit if we have all 9 skills
        if len(set(achieved)) >= len(VAR_NAMES):
            break

        # Hard budget: if we have already exceeded max total steps, stop
        if global_step >= args.max_total_steps:
            print(f"  [budget] reached max_total_steps={args.max_total_steps}, stopping")
            break

    t_elapsed = time.time() - t_start

    # ── Post-loop: check if diamond was in inventory at any point ──────────
    try:
        x_final = evgs.extract(env.get_obs())
        if x_final[DIAMOND_IDX] > 0.5 and steps_to_diamond_3d is None:
            steps_to_diamond_3d = global_step
    except Exception:
        pass

    try:
        env.env.close()
    except Exception:
        pass

    # ── Final PCG state ────────────────────────────────────────────────────
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
        "pcg_entropy_final":    float(pcg.entropy()),
        "pcg_strong_edges":     strong_edges,
        "completed":            len(set(achieved)) >= len(VAR_NAMES),
    }

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "DIA Claim 3 transfer experiment.\n"
            "  --mode baseline : flat scripted skills, no transfer\n"
            "  --mode transfer : 2D PCG + BC warm-start → guide 3D skill sequencing\n\n"
            "Requires Minecraft server for actual env steps.\n"
            "Use --dry_run to verify imports and initialisation without the server."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--mode", type=str, required=True, choices=["baseline", "transfer"],
        help="Experiment condition: 'baseline' or 'transfer'",
    )
    ap.add_argument(
        "--seed", type=int, default=0,
        help="Random seed (default: 0)",
    )
    ap.add_argument(
        "--pcg_path", type=str, default="pcg_2d.npy",
        help="Path to 2D PCG edge-prob matrix (.npy). Transfer mode only. "
             "(default: pcg_2d.npy)",
    )
    ap.add_argument(
        "--bc_dir", type=str, default="data/minerl_policies",
        help="Directory containing <skill>.pt BC policy files. Transfer mode only. "
             "(default: data/minerl_policies)",
    )
    ap.add_argument(
        "--ppo_dir", type=str, default="models/minerl",
        help="Directory containing skill_<var>.zip PPO models. Checked before BC. "
             "(default: models/minerl)",
    )
    ap.add_argument(
        "--env_id", type=str, default=None,
        help="MineRL env ID to use (default: auto-detect from registry)",
    )
    ap.add_argument(
        "--max_steps_per_skill", type=int, default=500,
        help="Max env steps allocated to each skill (default: 500)",
    )
    ap.add_argument(
        "--max_total_steps", type=int, default=10_000,
        help="Hard budget on total env steps per episode (default: 10000)",
    )
    ap.add_argument(
        "--out", type=str, default=None,
        help="JSON output path (default: results/transfer_3d_{mode}_seed{seed}.json)",
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
        os.makedirs("results", exist_ok=True)
        args.out = f"results/transfer_3d_{args.mode}_seed{args.seed}.json"

    print(f"run_transfer_3d.py")
    print(f"  mode       : {args.mode}")
    print(f"  seed       : {args.seed}")
    print(f"  pcg_path   : {args.pcg_path}")
    print(f"  bc_dir     : {args.bc_dir}")
    print(f"  out        : {args.out}")
    print(f"  dry_run    : {args.dry_run}")
    print()

    result = run_experiment(args)

    # ── Save result ────────────────────────────────────────────────────────
    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved → {out_path}")

    # ── Print key metric ───────────────────────────────────────────────────
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
            print("Diamond not reached within budget.")

    return result


if __name__ == "__main__":
    main()
