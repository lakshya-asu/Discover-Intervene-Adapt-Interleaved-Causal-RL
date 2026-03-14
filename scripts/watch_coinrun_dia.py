#!/usr/bin/env python3
"""
scripts/watch_coinrun_dia.py — Run the DIA agent on CoinRun and record to MP4.

Layout: game (384×384) | info panel showing:
  • PCG  — 7×7 causal-edge probability heatmap
  • SIG  — skill plan with auto-discovered prerequisites
  • EVGS — live variable bars
  • Metrics — H, IG, score, phase

EVGS variables (7):
  0  coin_visible    – yellow coin detected in frame
  1  coin_close      – coin in left 80% of frame
  2  coin_collected  – level completed
  3  coin_elevated   – coin on a high platform (upper 45% of frame)
  4  platform_above  – climbable platform visible in upper half
  5  saw_visible     – saw-blade hazard (gray circular) in mid-frame
  6  creature_visible– running enemy (colorful sprite) in right mid-frame

Skills (auto-prerequisite discovery via PCG):
  see_coin↑    – press RIGHT until coin enters frame
  approach↑    – press RIGHT until coin is close
  collect↑     – PPO policy to collect the coin  ← task goal
  level_coin↓  – jump RIGHT+UP to climb to coin's platform level
  climb↑       – jump to reveal / reach elevated platform
  dodge↓       – jump RIGHT+UP to clear saw blades
  evade↓       – jump RIGHT+UP to clear running enemies

Usage:
    conda activate dia
    python scripts/watch_coinrun_dia.py --model models/coinrun_cnn_ppo_v2.zip
"""
from __future__ import annotations

import argparse
import numpy as np
import imageio_ffmpeg

import gym

from dia.evgs_procgen import wrap_procgen_coinrun_env, CoinRunDetectorConfig
from dia.evgs_adapters import make_coinrun_evgs
from dia.sig import SIGraph, Skill
from dia.types import Subgoal, Predicate
from dia.planner import PlannerConfig, InterventionSelector
from dia.rollout import DIARunner, RunnerConfig
from dia.pcg_learner import DifferentiablePCG, DifferentiablePCGConfig
from dia.pcg import SimplePCG, PCGConfig
from dia.options import RandomOption, OptionConfig, OptionPolicy, FixedActionOption
from dia.options_coinrun import PixelStackPPOOption

try:
    from PIL import Image, ImageDraw
    PIL_OK = True
except Exception:
    PIL_OK = False

# ---------------------------------------------------------------------------
# ProcGen CoinRun action indices (15-action Discrete space)
# grid: {L, NOOP, R} × {D, NOOP, U} + 6 button actions
# ---------------------------------------------------------------------------
ACTION_NOOP     = 4
ACTION_RIGHT    = 7
ACTION_RIGHT_UP = 8   # jump right
ACTION_UP       = 5   # jump in place
ACTION_LEFT     = 1


# ---------------------------------------------------------------------------
# Phase-aware option dispatcher
# ---------------------------------------------------------------------------

class PhaseAwareOption(OptionPolicy):
    """NOVEL phase → stochastic PPO; CONFIRM/GOAL → deterministic PPO."""
    def __init__(self, subgoal, cfg, explore_opt, exploit_opt, selector):
        super().__init__(subgoal, cfg)
        self.explore_opt = explore_opt
        self.exploit_opt = exploit_opt
        self.selector    = selector

    def act(self, obs):
        phase = self.selector.phase()
        return (self.explore_opt if phase == "novel" else self.exploit_opt).act(obs)

    def run(self, env, evgs):
        phase = self.selector.phase()
        return (self.explore_opt if phase == "novel" else self.exploit_opt).run(env, evgs)


# ---------------------------------------------------------------------------
# Frame capture wrapper
# ---------------------------------------------------------------------------

class FrameCapture(gym.Wrapper):
    """Intercepts step/reset to record frames + per-frame EVGS snapshots."""
    def __init__(self, env, evgs=None, var_names=None):
        super().__init__(env)
        self.frames: list = []
        self._pending: list = []
        self._pending_evgs: list = []
        self._evgs = evgs
        self._var_names = var_names or []

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
            if self._evgs is not None:
                try:
                    self._pending_evgs.append(self._evgs.extract(obs).copy())
                except Exception:
                    self._pending_evgs.append(None)
            else:
                self._pending_evgs.append(None)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self._grab(result[0] if isinstance(result, tuple) else result)
        return result

    def step(self, action):
        result = self.env.step(action)
        self._grab(result[0])
        return result

    def get_obs(self):
        return self.env.get_obs()

    def flush(self, meta: dict):
        for i, f in enumerate(self._pending):
            fm = dict(meta)
            if i < len(self._pending_evgs) and self._pending_evgs[i] is not None:
                x = self._pending_evgs[i]
                if self._var_names and len(x) == len(self._var_names):
                    fm["var_vals"] = dict(zip(self._var_names, x.tolist()))
            self.frames.append((f, fm))
        self._pending = []
        self._pending_evgs = []


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_PHASE_COLOR = {
    "novel":   (80,  200, 255),
    "confirm": (255, 200,  80),
    "goal":    (80,  255, 120),
}


def _short(name: str) -> str:
    parts = name.split("_")
    return "".join(p[0] for p in parts)[:4].upper()


def make_frame(raw: np.ndarray, meta: dict, scale: int, var_names: list) -> np.ndarray:
    game = raw.astype(np.uint8).repeat(scale, axis=0).repeat(scale, axis=1)
    GH, GW = game.shape[:2]
    PANEL_W = 210

    if not PIL_OK:
        return game

    canvas = Image.new("RGB", (GW + PANEL_W, GH), color=(12, 12, 20))
    canvas.paste(Image.fromarray(game), (0, 0))
    draw = ImageDraw.Draw(canvas)

    # HUD bar
    phase   = meta.get("phase",      "?")
    skill   = meta.get("skill_name", "?")
    step    = meta.get("macro_step", 0)
    H_val   = meta.get("H",          0.0)
    success = meta.get("success",    False)
    score   = meta.get("score",      0.0)
    bar_h   = max(14, GH // 24)
    draw.rectangle([0, 0, GW, bar_h], fill=(0, 0, 0, 200))
    hud_color = _PHASE_COLOR.get(phase, (200, 200, 200))
    tick = "✓" if success else "·"
    draw.text((3, 2),
              f"[{step:03d}] {phase:<7} {skill:<16} H={H_val:.2f}  {tick}  score={score:.0f}",
              fill=hud_color)

    M   = len(var_names)
    PX  = GW + 4
    PY  = 6
    short = [_short(n) for n in var_names]

    # PCG section
    draw.text((PX, PY), "PCG  causal edges", fill=(100, 180, 255)); PY += 12
    pcg_probs = meta.get("pcg_probs")
    LABEL  = 22
    avail  = PANEL_W - LABEL - 6
    cell_w = max(1, avail // M)
    cell_h = 17

    for j in range(M):
        draw.text((PX + LABEL + j * cell_w + 1, PY), short[j], fill=(80, 180, 130))
    PY += 10

    for i in range(M):
        draw.text((PX, PY + 3), short[i], fill=(80, 180, 130))
        for j in range(M):
            x0 = PX + LABEL + j * cell_w
            y0 = PY
            x1 = x0 + cell_w - 2
            y1 = y0 + cell_h - 2
            if i == j:
                draw.rectangle([x0, y0, x1, y1], fill=(20, 20, 30))
            else:
                if pcg_probs is not None:
                    p = float(np.clip(pcg_probs[i, j], 0, 1))
                    if p >= 0.5:
                        r_c = int(60 + 195 * (p - 0.5) * 2)
                        g_c = int(60 - 40  * (p - 0.5) * 2)
                        b_c = 60
                    else:
                        r_c = 60
                        g_c = int(60 - 40  * p * 2)
                        b_c = int(60 + 195 * (0.5 - p) * 2)
                    draw.rectangle([x0, y0, x1, y1], fill=(r_c, g_c, b_c))
                    draw.text((x0 + 1, y0 + 3), f"{p:.2f}", fill=(240, 240, 240))
                else:
                    draw.rectangle([x0, y0, x1, y1], fill=(50, 50, 70))
                    draw.text((x0 + 1, y0 + 3), "?", fill=(160, 160, 160))
        PY += cell_h

    # PCG entropy bar
    PY += 3
    max_H = M * (M - 1) * 0.693
    frac  = min(1.0, H_val / max(1e-6, max_H))
    blen  = PANEL_W - 12
    draw.rectangle([PX, PY, PX + blen, PY + 5], fill=(30, 30, 45))
    draw.rectangle([PX, PY, PX + int(blen * frac), PY + 5], fill=(100, 180, 255))
    draw.text((PX, PY + 7), f"H={H_val:.3f}  max≈{max_H:.1f}", fill=(140, 160, 200))
    PY += 19

    # SIG section
    PY += 3
    draw.line([(PX, PY), (PX + PANEL_W - 10, PY)], fill=(35, 35, 50)); PY += 4
    draw.text((PX, PY), "SIG  skill plan", fill=(100, 180, 255)); PY += 12
    for skill_name, is_achieved, is_active in meta.get("sig_skills", []):
        if is_active:
            fg, mk = (80, 255, 120), "▶"
        elif is_achieved:
            fg, mk = (60, 160, 60), "✓"
        else:
            fg, mk = (100, 100, 120), "○"
        draw.text((PX, PY), f"{mk} {skill_name}", fill=fg); PY += 13

    # EVGS state
    PY += 3
    draw.line([(PX, PY), (PX + PANEL_W - 10, PY)], fill=(35, 35, 50)); PY += 4
    draw.text((PX, PY), "EVGS  state", fill=(100, 180, 255)); PY += 12
    bar_max = PANEL_W - 12
    for vname, vval in meta.get("var_vals", {}).items():
        v  = float(vval)
        fg = (80, 230, 80) if v > 0.5 else (110, 110, 130)
        bl = int(bar_max * min(1.0, v))
        draw.rectangle([PX, PY + 10, PX + bar_max, PY + 13], fill=(25, 25, 35))
        if bl > 0:
            draw.rectangle([PX, PY + 10, PX + bl, PY + 13], fill=fg)
        label = vname.split("_")[-1][:7]
        draw.text((PX, PY), f"{label}: {v:.2f}", fill=fg); PY += 16

    # Discovered causal edges
    PY += 3
    draw.line([(PX, PY), (PX + PANEL_W - 10, PY)], fill=(35, 35, 50)); PY += 4
    draw.text((PX, PY), "Causal edges", fill=(100, 180, 255)); PY += 12
    edges_str = meta.get("sig_edges", "none yet")
    line_buf = ""
    for w in edges_str.split(" "):
        if len(line_buf) + len(w) + 1 > 25:
            draw.text((PX, PY), line_buf, fill=(200, 160, 80)); PY += 12; line_buf = w
        else:
            line_buf = (line_buf + " " + w).strip()
    if line_buf:
        draw.text((PX, PY), line_buf, fill=(200, 160, 80)); PY += 12

    # Bottom metrics
    PY += 3
    draw.line([(PX, PY), (PX + PANEL_W - 10, PY)], fill=(35, 35, 50)); PY += 5
    ig   = meta.get("ig", 0.0)
    fits = meta.get("fits", 0)
    draw.text((PX, PY), f"IG    {ig:.4f}",  fill=(160, 160, 190)); PY += 13
    draw.text((PX, PY), f"score {score:.0f}", fill=(200, 220, 100)); PY += 13
    draw.text((PX, PY), f"fits  {fits}",      fill=(160, 160, 190)); PY += 13
    draw.text((PX, PY), f"phase {phase}",     fill=_PHASE_COLOR.get(phase, (180, 180, 180)))

    return np.array(canvas)


# ---------------------------------------------------------------------------
# SIG: 7-skill hierarchy (all prerequisites auto-discovered by PCG)
# ---------------------------------------------------------------------------

def build_coinrun_sig(evgs) -> SIGraph:
    """
    Bootstrap SIG with 7 skills and NO hard-coded prerequisites.

    PCG discovers causal structure from transitions, e.g.:
      coin_visible    → coin_collected  (must see coin to collect it)
      coin_elevated   → coin_close      (must climb before approaching)
      platform_above  → coin_elevated   (platform reveals elevated path)
      saw_visible     → coin_close↓     (saw blocks approach)
      creature_visible → coin_close↓   (enemy blocks approach)
    """
    names = evgs.names()
    idx   = {n: i for i, n in enumerate(names)}
    M     = len(names)   # 7
    sig   = SIGraph()

    # UP-predicate skills (skill_id == var_index)
    s_see    = Skill(skill_id=idx["coin_visible"],
                     subgoal=Subgoal(idx["coin_visible"],   Predicate.UP),
                     name="see_coin↑")
    s_close  = Skill(skill_id=idx["coin_close"],
                     subgoal=Subgoal(idx["coin_close"],     Predicate.UP),
                     name="approach↑")
    s_coll   = Skill(skill_id=idx["coin_collected"],
                     subgoal=Subgoal(idx["coin_collected"], Predicate.UP),
                     name="collect↑")
    s_climb  = Skill(skill_id=idx["platform_above"],
                     subgoal=Subgoal(idx["platform_above"], Predicate.UP),
                     name="climb↑")

    # DOWN-predicate skills (skill_id = var_index + M to avoid collision)
    s_level  = Skill(skill_id=idx["coin_elevated"]   + M,
                     subgoal=Subgoal(idx["coin_elevated"],   Predicate.DOWN),
                     name="level_coin↓")
    s_dodge  = Skill(skill_id=idx["saw_visible"]     + M,
                     subgoal=Subgoal(idx["saw_visible"],     Predicate.DOWN),
                     name="dodge↓")
    s_evade  = Skill(skill_id=idx["creature_visible"] + M,
                     subgoal=Subgoal(idx["creature_visible"], Predicate.DOWN),
                     name="evade↓")

    for s in [s_see, s_close, s_coll, s_climb, s_level, s_dodge, s_evade]:
        sig.add_skill(s)

    return sig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--macro_steps",  type=int, default=150)
    ap.add_argument("--option_steps", type=int, default=500)
    ap.add_argument("--out",          type=str, default="coinrun_dia_v3_rich.mp4")
    ap.add_argument("--fps",          type=int, default=30)
    ap.add_argument("--seed",         type=int, default=0)
    ap.add_argument("--scale",        type=int, default=6)
    ap.add_argument("--pcg",          type=str, default="notears",
                    choices=["notears", "simple"])
    ap.add_argument("--min_buffer",   type=int, default=15)
    ap.add_argument("--fit_every",    type=int, default=5)
    ap.add_argument("--pcg_epochs",   type=int, default=100)
    ap.add_argument("--model",        type=str, default=None)
    ap.add_argument("--n_stack",      type=int, default=4)
    ap.add_argument("--num_levels",   type=int, default=200)
    ap.add_argument("--start_level",  type=int, default=0)
    args = ap.parse_args()

    # env
    base_env = gym.make("procgen:procgen-coinrun-v0",
                        num_levels=args.num_levels, start_level=args.start_level)
    wrapped  = wrap_procgen_coinrun_env(base_env, CoinRunDetectorConfig())

    # EVGS (7 vars)
    evgs      = make_coinrun_evgs()
    var_names = evgs.names()
    M         = len(var_names)   # 7

    env = FrameCapture(wrapped, evgs=evgs, var_names=var_names)

    if args.pcg == "simple":
        pcg = SimplePCG(PCGConfig(num_vars=M, init_edge_prob=0.5, seed=args.seed))
    else:
        pcg = DifferentiablePCG(DifferentiablePCGConfig(
            num_vars=M, max_iter=args.pcg_epochs, lr=5e-3, verbose=False))

    # SIG
    sig = build_coinrun_sig(evgs)

    # Planner — scale entropy thresholds with number of variables
    plan_cfg = PlannerConfig(
        entropy_high=4.05 * (M / 3),
        entropy_low=2.0  * (M / 3),
    )
    selector = InterventionSelector(pcg, sig, plan_cfg)

    # Variable index shortcuts
    idx       = {n: i for i, n in enumerate(var_names)}
    see_id    = idx["coin_visible"]
    close_id  = idx["coin_close"]
    coin_id   = idx["coin_collected"]
    elev_id   = idx["coin_elevated"]
    plat_id   = idx["platform_above"]
    saw_id    = idx["saw_visible"]
    creat_id  = idx["creature_visible"]

    ppo_model = None
    if args.model:
        print(f"  Loading PPO model: {args.model}")
        from stable_baselines3 import PPO as SB3PPO
        ppo_model = SB3PPO.load(args.model)
        print(f"  PPO loaded  (n_stack={args.n_stack}, option_steps={args.option_steps})")
    else:
        print("  No --model: skills use fixed/random actions")

    ppo_cfg = OptionConfig(max_steps=args.option_steps, terminate_on_success=True)

    def option_factory(skill):
        sg    = skill.subgoal
        vname = var_names[sg.var_index] if sg.var_index < M else "?"

        # collect↑ — phase-aware PPO (complex navigation)
        if vname == "coin_collected" and ppo_model is not None:
            exp = PixelStackPPOOption(sg, ppo_cfg, ppo_model, args.n_stack, deterministic=False)
            expl= PixelStackPPOOption(sg, ppo_cfg, ppo_model, args.n_stack, deterministic=True)
            return PhaseAwareOption(sg, ppo_cfg, exp, expl, selector)

        # see_coin↑ — press RIGHT until coin enters frame
        if vname == "coin_visible":
            return FixedActionOption(sg, OptionConfig(max_steps=100, terminate_on_success=True),
                                     action=ACTION_RIGHT)

        # approach↑ — press RIGHT to get close
        if vname == "coin_close":
            return FixedActionOption(sg, OptionConfig(max_steps=150, terminate_on_success=True),
                                     action=ACTION_RIGHT)

        # climb↑ (platform_above↑) — jump to reveal/reach elevated platform
        if vname == "platform_above":
            return FixedActionOption(sg, OptionConfig(max_steps=60, terminate_on_success=True),
                                     action_cycle=[ACTION_RIGHT_UP, ACTION_RIGHT_UP,
                                                   ACTION_RIGHT, ACTION_UP])

        # level_coin↓ (coin_elevated↓) — jump right to climb to coin's level
        if vname == "coin_elevated":
            return FixedActionOption(sg, OptionConfig(max_steps=80, terminate_on_success=True),
                                     action_cycle=[ACTION_RIGHT_UP, ACTION_RIGHT_UP, ACTION_RIGHT])

        # dodge↓ (saw_visible↓) — jump right over the saw
        if vname == "saw_visible":
            return FixedActionOption(sg, OptionConfig(max_steps=60, terminate_on_success=True),
                                     action_cycle=[ACTION_RIGHT_UP, ACTION_RIGHT_UP,
                                                   ACTION_RIGHT, ACTION_RIGHT])

        # evade↓ (creature_visible↓) — jump right past the enemy
        if vname == "creature_visible":
            return FixedActionOption(sg, OptionConfig(max_steps=60, terminate_on_success=True),
                                     action_cycle=[ACTION_RIGHT_UP, ACTION_RIGHT_UP,
                                                   ACTION_RIGHT, ACTION_RIGHT])

        return RandomOption(sg, OptionConfig(max_steps=80), env.action_space)

    # DIA runner
    runner = DIARunner(
        env, evgs, pcg, sig, selector,
        RunnerConfig(
            buffer_size           = 10_000,
            min_buffer            = args.min_buffer,
            fit_every             = args.fit_every,
            pcg_epochs            = args.pcg_epochs,
            option_max_steps      = args.option_steps,
            terminate_on_success  = True,
            auto_expand_sig       = True,
            add_threshold         = 0.62,
            remove_threshold      = 0.40,
            create_missing_skills = False,
        ),
        option_factory=option_factory,
    )

    env.reset()
    env._pending = []

    task_goal      = Subgoal(var_index=coin_id, predicate=Predicate.UP)
    achieved: list = []
    score          = 0.0
    pcg_fits       = 0
    prev_sig_edges = set()

    policy_str = f"PPO({args.model})" if args.model else "fixed+random"
    print("=" * 72)
    print(f"  DIA CoinRun  |  PCG={args.pcg}  |  macro_steps={args.macro_steps}")
    print(f"  policy={policy_str}  option_steps={args.option_steps}")
    print(f"  vars ({M}): {var_names}")
    print(f"  H thresholds: high={plan_cfg.entropy_high:.2f}  low={plan_cfg.entropy_low:.2f}")
    print("=" * 72)

    for t in range(args.macro_steps):
        rec = runner.step(achieved, task_goal=task_goal)
        H   = float(pcg.entropy()) if hasattr(pcg, "entropy") else 0.0

        # Episode-reset: coin_visible drops when level changes → clear episode achievements
        delta_x = rec.get("delta_x")
        if delta_x is not None and delta_x[see_id] < -0.5:
            achieved = []

        if rec["success"]:
            sid = rec["skill_id"]
            if sid == coin_id:
                score   += 10.0
                achieved = []
            elif sid not in achieved:
                achieved.append(sid)

        if rec["did_fit_pcg"]:
            pcg_fits += 1

        # Detect newly auto-discovered SIG prerequisites
        cur_sig_edges = set()
        for sid, sk in sig.skills.items():
            for pre in sig.prerequisites(sid):
                cur_sig_edges.add((pre, sid))
        new_edges = cur_sig_edges - prev_sig_edges
        for (pre, post) in new_edges:
            pn = sig.skills[pre].name  if pre  in sig.skills else str(pre)
            pst= sig.skills[post].name if post in sig.skills else str(post)
            print(f"  *** PCG DISCOVERED edge: {pn} → {pst} (step {t+1}) ***")
        prev_sig_edges = cur_sig_edges

        try:
            x_now = evgs.extract(env.get_obs())
        except Exception:
            x_now = np.zeros(M)

        try:
            pcg_probs_now = np.array(pcg.probs).copy()
        except Exception:
            pcg_probs_now = None

        sig_skills_list = []
        for sid in sig.toposort():
            if sid not in sig.skills:
                continue
            sk    = sig.skills[sid]
            pres  = sig.prerequisites(sid)
            pre_names = "+".join(sig.skills[p].name for p in pres if p in sig.skills)
            label = sk.name + (f"  [needs:{pre_names}]" if pre_names else "  [free]")
            sig_skills_list.append((label, sid in achieved, rec["skill_id"] == sid))

        sig_edges_str = " → ".join(
            f"{sig.skills[p].name}→{sig.skills[s].name}"
            for (p, s) in sorted(cur_sig_edges)
            if p in sig.skills and s in sig.skills
        ) or "none yet"

        meta = {
            "macro_step": t + 1,
            "phase":      rec["phase"],
            "skill_name": rec["skill_name"],
            "success":    rec["success"],
            "H":          H,
            "score":      score,
            "ig":         rec["ig_update"],
            "fits":       pcg_fits,
            "pcg_probs":  pcg_probs_now,
            "sig_skills": sig_skills_list,
            "sig_edges":  sig_edges_str,
            "var_vals":   dict(zip(var_names, x_now.tolist())),
        }
        env.flush(meta)

        print(f"  [{t+1:03d}] {rec['phase']:<7} {rec['skill_name']:<16} "
              f"succ={int(rec['success'])}  H={H:5.2f}  "
              f"vis={x_now[see_id]:.0f} "
              f"elev={x_now[elev_id]:.0f} "
              f"plat={x_now[plat_id]:.0f} "
              f"saw={x_now[saw_id]:.0f} "
              f"creat={x_now[creat_id]:.0f} "
              f"buf={rec['buffer_size']:4d}  score={score:.0f}  "
              f"frames={len(env.frames)}")

    env.close()

    if not env.frames:
        print("ERROR: no frames captured.")
        return

    print(f"\nWriting {len(env.frames)} frames → {args.out}  (fps={args.fps})")
    first = make_frame(env.frames[0][0], env.frames[0][1], args.scale, var_names)
    H_px, W_px = first.shape[:2]
    W_px = (W_px // 16) * 16
    H_px = (H_px // 16) * 16
    gen = imageio_ffmpeg.write_frames(args.out, size=(W_px, H_px), fps=args.fps,
                                      quality=8, codec="libx264")
    gen.send(None)
    for raw_frame, frame_meta in env.frames:
        frame = make_frame(raw_frame, frame_meta, args.scale, var_names)
        frame = frame[:H_px, :W_px]
        gen.send(frame.tobytes())
    gen.close()

    print(f"Done.  score={score:.0f}  PCG fits={pcg_fits}")
    print(f"Video: {args.out}")


if __name__ == "__main__":
    main()
