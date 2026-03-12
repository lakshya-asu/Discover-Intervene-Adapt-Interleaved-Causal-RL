#!/usr/bin/env python3
"""
scripts/watch_coinrun_dia.py — Run the DIA agent on CoinRun and record to MP4.

Skill hierarchy (3-step, each backed by the trained PPO):
  see_coin↑   – navigate until the yellow coin appears in frame
  approach↑   – move until coin centroid is in the left 55 % of frame
  collect↑    – collect the coin

DIA discovers the causal dependencies between these variables via PCG and uses
that structure to plan: see → approach → collect.

Usage:
    conda activate dia
    cd Discover-Intervene-Adapt-Interleaved-Causal-RL
    python scripts/train_coinrun_ppo.py --timesteps 1000000 --n_envs 4
    python scripts/watch_coinrun_dia.py --model models/coinrun_cnn_ppo.zip --macro_steps 120
"""
from __future__ import annotations

import argparse
import numpy as np
import imageio

import gym

from dia.evgs_procgen import wrap_procgen_coinrun_env, CoinRunDetectorConfig
from dia.evgs_adapters import make_coinrun_evgs
from dia.sig import SIGraph, Skill
from dia.types import Subgoal, Predicate
from dia.planner import PlannerConfig, InterventionSelector
from dia.rollout import DIARunner, RunnerConfig
from dia.pcg_learner import DifferentiablePCG, DifferentiablePCGConfig
from dia.pcg import SimplePCG, PCGConfig
from dia.options import RandomOption, OptionConfig
from dia.options_coinrun import PixelStackPPOOption

try:
    from PIL import Image, ImageDraw
    PIL_OK = True
except Exception:
    PIL_OK = False


# ---------------------------------------------------------------------------
# Env wrapper: intercepts step() to capture raw RGB frames
# ---------------------------------------------------------------------------

class FrameCapture(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.frames: list = []
        self._pending: list = []

    def _grab(self, obs):
        raw = obs
        for _ in range(8):
            if isinstance(raw, np.ndarray):
                break
            if isinstance(raw, dict):
                raw = raw.get("obs", raw)
            else:
                break
        if isinstance(raw, np.ndarray) and raw.ndim == 3:
            self._pending.append(raw.copy())

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        obs = result[0] if isinstance(result, tuple) else result
        self._grab(obs)
        return result

    def step(self, action):
        result = self.env.step(action)
        self._grab(result[0])
        return result

    def get_obs(self):
        return self.env.get_obs()

    def flush(self, meta: dict):
        for f in self._pending:
            self.frames.append((f, meta))
        self._pending = []


# ---------------------------------------------------------------------------
# Frame annotation overlay
# ---------------------------------------------------------------------------

_PHASE_COLOR = {
    "novel":   (80,  200, 255),
    "confirm": (255, 200,  80),
    "goal":    (80,  255, 120),
}


def add_overlay(raw: np.ndarray, meta: dict, scale: int) -> np.ndarray:
    frame = raw.astype(np.uint8).repeat(scale, axis=0).repeat(scale, axis=1)
    if not PIL_OK:
        return frame
    img  = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    H, W = frame.shape[:2]
    bar  = max(16, H // 9)
    draw.rectangle([0, 0, W, bar], fill=(0, 0, 0, 210))
    phase   = meta.get("phase",      "?")
    skill   = meta.get("skill_name", "?")
    step    = meta.get("macro_step", 0)
    H_val   = meta.get("H",          0.0)
    success = meta.get("success",    False)
    score   = meta.get("score",      0.0)
    ig      = meta.get("ig",         0.0)
    fits    = meta.get("fits",       0)
    color = _PHASE_COLOR.get(phase, (200, 200, 200))
    tick  = "✓" if success else "·"
    text  = (f"[{step:03d}] {phase:<7} {skill:<14} H={H_val:4.2f}  "
             f"IG={ig:.3f}  {tick}  score={score:.0f}  PCG_fits={fits}")
    draw.text((3, 2), text, fill=color)
    return np.array(img)


# ---------------------------------------------------------------------------
# SIG: 3-step causal hierarchy (see → approach → collect)
# ---------------------------------------------------------------------------

def build_coinrun_sig(evgs) -> SIGraph:
    """
    Two-skill hierarchy:  see_coin↑  →  collect↑
    coin_close is kept in the EVGS/PCG for causal structure learning but is
    not exposed as a targeted skill (its UP predicate has a state-aliasing
    issue when the coin first becomes visible while already close).
    """
    names = evgs.names()
    idx   = {n: i for i, n in enumerate(names)}
    sig   = SIGraph()

    s_see     = Skill(skill_id=idx["coin_visible"],   subgoal=Subgoal(idx["coin_visible"],   Predicate.UP), name="see_coin↑")
    s_collect = Skill(skill_id=idx["coin_collected"], subgoal=Subgoal(idx["coin_collected"], Predicate.UP), name="collect↑")

    sig.add_skill(s_see)
    sig.add_skill(s_collect)

    # Must see the coin before collecting
    sig.add_prerequisite(s_see.skill_id, s_collect.skill_id)
    return sig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--macro_steps",  type=int, default=120)
    ap.add_argument("--option_steps", type=int, default=256,
                    help="Max env steps per option (episode break also terminates)")
    ap.add_argument("--out",          type=str, default="coinrun_dia_trained.mp4")
    ap.add_argument("--fps",          type=int, default=15)
    ap.add_argument("--seed",         type=int, default=0)
    ap.add_argument("--scale",        type=int, default=6)
    ap.add_argument("--pcg",          type=str, default="notears", choices=["notears", "simple"])
    ap.add_argument("--min_buffer",   type=int, default=15)
    ap.add_argument("--fit_every",    type=int, default=5)
    ap.add_argument("--pcg_epochs",   type=int, default=100)
    ap.add_argument("--model",        type=str, default=None)
    ap.add_argument("--n_stack",      type=int, default=4)
    ap.add_argument("--num_levels",   type=int, default=200)
    args = ap.parse_args()

    # --- env ---
    base_env = gym.make("procgen:procgen-coinrun-v0",
                        num_levels=args.num_levels, start_level=args.seed)
    wrapped  = wrap_procgen_coinrun_env(base_env, CoinRunDetectorConfig())
    env      = FrameCapture(wrapped)

    # --- EVGS + PCG ---
    evgs = make_coinrun_evgs()
    M    = len(evgs.names())

    if args.pcg == "simple":
        pcg = SimplePCG(PCGConfig(num_vars=M, init_edge_prob=0.5, seed=args.seed))
    else:
        pcg = DifferentiablePCG(DifferentiablePCGConfig(
            num_vars=M, max_iter=args.pcg_epochs, lr=5e-3, verbose=False))

    # --- SIG ---
    sig = build_coinrun_sig(evgs)

    # --- Planner (entropy thresholds calibrated for M=3: max H ≈ 4.16 nats) ---
    plan_cfg = PlannerConfig(
        entropy_high=3.8,   # above → "novel"  (explore / discover structure)
        entropy_low=0.5,    # below → "goal"   (exploit discovered structure)
    )
    selector = InterventionSelector(pcg, sig, plan_cfg)

    # --- Options: same PPO model, one option object per skill with its own subgoal ---
    name_to_idx = {n: i for i, n in enumerate(evgs.names())}
    see_id      = name_to_idx["coin_visible"]
    coin_id     = name_to_idx["coin_collected"]

    ppo_model = None
    if args.model:
        print(f"  Loading PPO model: {args.model}")
        from stable_baselines3 import PPO as SB3PPO
        ppo_model = SB3PPO.load(args.model)
        print(f"  PPO loaded  (n_stack={args.n_stack}, option_steps={args.option_steps})")
    else:
        print("  No --model: all skills use random actions")

    # see_coin: shorter horizon (just need coin to enter frame)
    # collect:  longer horizon (may need full episode to navigate & collect)
    see_cfg  = OptionConfig(max_steps=min(args.option_steps, 256), terminate_on_success=True)
    coll_cfg = OptionConfig(max_steps=args.option_steps,           terminate_on_success=True)

    if ppo_model is not None:
        skill_to_opt = {
            see_id:  PixelStackPPOOption(Subgoal(see_id,  Predicate.UP), see_cfg,  ppo_model, args.n_stack),
            coin_id: PixelStackPPOOption(Subgoal(coin_id, Predicate.UP), coll_cfg, ppo_model, args.n_stack),
        }
    else:
        skill_to_opt = {}

    def option_factory(skill):
        if skill.skill_id in skill_to_opt:
            return skill_to_opt[skill.skill_id]
        return RandomOption(skill.subgoal, opt_cfg, env.action_space)

    # --- DIA runner ---
    runner = DIARunner(
        env, evgs, pcg, sig, selector,
        RunnerConfig(
            buffer_size          = 10_000,
            min_buffer           = args.min_buffer,
            fit_every            = args.fit_every,
            pcg_epochs           = args.pcg_epochs,
            option_max_steps     = args.option_steps,
            terminate_on_success = True,
            auto_expand_sig      = False,  # keep hard-coded hierarchy intact for demo
        ),
        option_factory=option_factory,
    )

    env.reset()
    env._pending = []

    task_goal = Subgoal(var_index=coin_id, predicate=Predicate.UP)
    achieved: list = []
    score    = 0.0
    pcg_fits = 0

    policy_str = f"PPO({args.model})" if args.model else "random"
    print("=" * 72)
    print(f"  DIA CoinRun  |  PCG={args.pcg}  |  macro_steps={args.macro_steps}")
    print(f"  policy={policy_str}  option_steps={args.option_steps}")
    print(f"  vars: {evgs.names()}")
    print(f"  H thresholds: high={plan_cfg.entropy_high}  low={plan_cfg.entropy_low}")
    print("=" * 72)

    for t in range(args.macro_steps):
        rec = runner.step(achieved, task_goal=task_goal)
        H   = float(pcg.entropy()) if hasattr(pcg, "entropy") else 0.0

        if rec["success"]:
            sid = rec["skill_id"]
            if sid == coin_id:
                score    += 10.0
                # Reset achieved so the hierarchy restarts for the next episode
                achieved  = []
            elif sid not in achieved:
                achieved.append(sid)

        if rec["did_fit_pcg"]:
            pcg_fits += 1

        meta = {
            "macro_step": t + 1,
            "phase":      rec["phase"],
            "skill_name": rec["skill_name"],
            "success":    rec["success"],
            "H":          H,
            "score":      score,
            "ig":         rec["ig_update"],
            "fits":       pcg_fits,
        }
        env.flush(meta)

        print(f"  [{t+1:03d}] {rec['phase']:<7} {rec['skill_name']:<14} "
              f"succ={int(rec['success'])}  H={H:5.2f}  "
              f"buf={rec['buffer_size']:4d}  score={score:.0f}  "
              f"frames={len(env.frames)}")

    env.close()

    if not env.frames:
        print("ERROR: no frames captured.")
        return

    print(f"\nWriting {len(env.frames)} frames → {args.out}  (fps={args.fps})")
    with imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8) as writer:
        for raw_frame, meta in env.frames:
            writer.append_data(add_overlay(raw_frame, meta, args.scale))

    print(f"Done.  score={score:.0f}  PCG fits={pcg_fits}")
    print(f"Video: {args.out}")


if __name__ == "__main__":
    main()
