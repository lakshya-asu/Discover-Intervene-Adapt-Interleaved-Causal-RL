#!/usr/bin/env python3
"""
scripts/watch_coinrun_dia_v4.py  —  DIA agent on CoinRun (interventional PCG).

New algorithm: Structured Interventional Probing (SIP)
------------------------------------------------------
Problems with v3:
  1. PCG was OBSERVATIONAL: mixed all skill transitions, found correlations not causes.
  2. Planner looped on dodge/evade because saw/creature are near-constant false positives.
  3. Premature SIG edges (added with only 15 data points) blocked direct coin collection.

SIP fixes:
  1. INTERVENTIONAL PCG: when skill k executes, record P(j↑ | skill_k did that).
     Each skill's row is updated only from that skill's own transitions.
  2. ROUND-ROBIN DISCOVERY: try every skill at least N times before trusting structure.
     Prevents any single skill (dodge/evade) from dominating.
  3. CONSERVATIVE EDGES: add SIG prereq only if exec_count >= min_obs AND p > 0.78.
     More data = more confident = fewer spurious lockouts.
  4. GOAL PHASE: once discovery complete, follow causal chain to collect coin.

Layout: game (384×384) | info panel
  PCG heatmap — P(var j↑ | skill k executes)  [rows = skills, cols = effect vars]
  SIG plan    — prerequisite chain
  EVGS state  — live variable bars
  Exec table  — how many times each skill was tried
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
from dia.plan_search import GoalPlanner
from dia.sig_auto import expand_sig_from_pcg, AutoSIGConfig
from dia.options import RandomOption, OptionConfig, OptionPolicy, FixedActionOption
from dia.options_coinrun import PixelStackPPOOption
from dia.interventional_pcg import InterventionalPCG

try:
    from PIL import Image, ImageDraw
    PIL_OK = True
except Exception:
    PIL_OK = False


# ---------------------------------------------------------------------------
# Action indices for ProcGen CoinRun (15-action Discrete)
# ---------------------------------------------------------------------------
ACTION_NOOP     = 4
ACTION_RIGHT    = 7
ACTION_RIGHT_UP = 8   # jump right
ACTION_UP       = 5
ACTION_LEFT     = 1


# ---------------------------------------------------------------------------
# Phase-aware PPO option
# ---------------------------------------------------------------------------

class PhaseAwareOption(OptionPolicy):
    """NOVEL → stochastic PPO;  CONFIRM/GOAL → deterministic PPO."""
    def __init__(self, subgoal, cfg, explore_opt, exploit_opt, selector):
        super().__init__(subgoal, cfg)
        self.explore_opt = explore_opt
        self.exploit_opt = exploit_opt
        self.selector    = selector

    def act(self, obs):
        return (self.explore_opt if self.selector.phase() == "discover"
                else self.exploit_opt).act(obs)

    def run(self, env, evgs):
        return (self.explore_opt if self.selector.phase() == "discover"
                else self.exploit_opt).run(env, evgs)


# ---------------------------------------------------------------------------
# Frame capture
# ---------------------------------------------------------------------------

class FrameCapture(gym.Wrapper):
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
# Information-directed skill selector
# ---------------------------------------------------------------------------

class InformationDirectedSelector:
    """
    Three-phase selection driven by InterventionalPCG:

      discover  — round-robin: pick the skill with fewest executions.
                  Ensures every skill gets tried equally. No skill can dominate.
      confirm   — pick the skill whose row has highest entropy (still uncertain effects).
      goal      — follow causal plan derived from confident PCG edges.

    Phase transition:
      discover → confirm  when all skills have >= discover_per_skill executions
      confirm  → goal     when total PCG entropy < goal_entropy_frac * max_entropy
    """

    def __init__(self, ipcg: InterventionalPCG, sig: SIGraph,
                 goal_planner: GoalPlanner,
                 var_to_skill: dict,          # var_index → skill_id
                 discover_per_skill: int = 8,
                 goal_entropy_frac: float = 0.30):
        self.ipcg                = ipcg
        self.sig                 = sig
        self.goal_planner        = goal_planner
        self.var_to_skill        = var_to_skill   # var_index -> skill_id
        self.discover_per_skill  = discover_per_skill
        self.goal_entropy_frac   = goal_entropy_frac

    def phase(self) -> str:
        # Still in discovery if any skill hasn't been tried enough
        for var_k in self.var_to_skill:
            if self.ipcg.exec_count[var_k] < self.discover_per_skill:
                return "discover"
        # Graduated to confirm or goal based on remaining uncertainty
        H     = self.ipcg.entropy()
        max_H = self.ipcg.d * (self.ipcg.d - 1) * 0.693
        if H < self.goal_entropy_frac * max(1e-6, max_H):
            return "goal"
        return "confirm"

    def select(self, achieved: list, task_goal=None) -> int:
        ph = self.phase()

        if ph == "discover":
            # Least-executed skill (ties broken by var_index order)
            return min(
                self.var_to_skill.values(),
                key=lambda sid: self.ipcg.exec_count[
                    self.sig.skills[sid].subgoal.var_index
                ],
            )

        if ph == "confirm":
            # Skill whose row still has highest entropy
            return max(
                self.var_to_skill.values(),
                key=lambda sid: self.ipcg.row_entropy(
                    self.sig.skills[sid].subgoal.var_index
                ),
            )

        # ── goal phase ──────────────────────────────────────────────────────
        if task_goal is not None:
            plan = self.goal_planner.plan_for_subgoal(task_goal, achieved)
            if plan and plan.skills:
                ready = set(self.sig.ready_skills(achieved))
                for sid in plan.skills:
                    if sid in ready:
                        return sid

        # Fallback: skill with highest success rate that is currently ready
        ready = self.sig.ready_skills(achieved)
        if not ready:
            ready = list(self.sig.skills.keys())
        return max(ready, key=lambda sid: self.sig.skills[sid].success_rate)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_PHASE_COLOR = {
    "discover": (80,  200, 255),
    "confirm":  (255, 200,  80),
    "goal":     (80,  255, 120),
}


def _short(name: str) -> str:
    parts = name.split("_")
    return "".join(p[0] for p in parts)[:4].upper()


def make_frame(raw: np.ndarray, meta: dict, scale: int, var_names: list) -> np.ndarray:
    game  = raw.astype(np.uint8).repeat(scale, axis=0).repeat(scale, axis=1)
    GH, GW = game.shape[:2]
    PANEL_W = 215

    if not PIL_OK:
        return game

    canvas = Image.new("RGB", (GW + PANEL_W, GH), color=(12, 12, 20))
    canvas.paste(Image.fromarray(game), (0, 0))
    draw = ImageDraw.Draw(canvas)

    # HUD bar
    phase   = meta.get("phase", "?")
    skill   = meta.get("skill_name", "?")
    step    = meta.get("macro_step", 0)
    H_val   = meta.get("H", 0.0)
    success = meta.get("success", False)
    score   = meta.get("score", 0.0)
    bar_h   = max(14, GH // 24)
    draw.rectangle([0, 0, GW, bar_h], fill=(0, 0, 0))
    hud_color = _PHASE_COLOR.get(phase, (200, 200, 200))
    tick = "\u2713" if success else "\u00b7"
    draw.text((3, 2),
              f"[{step:03d}] {phase:<8} {skill:<14} H={H_val:.2f} {tick} score={score:.0f}",
              fill=hud_color)

    M     = len(var_names)
    PX    = GW + 4
    PY    = 6
    short = [_short(n) for n in var_names]

    # PCG heatmap — rows = cause skill (var k), cols = effect variable j
    draw.text((PX, PY), "PCG  P(j\u2191 | skill_k executes)", fill=(100, 180, 255)); PY += 12
    LABEL  = 22
    avail  = PANEL_W - LABEL - 6
    cell_w = max(1, avail // M)
    cell_h = 17

    for j in range(M):
        draw.text((PX + LABEL + j * cell_w + 1, PY), short[j], fill=(80, 180, 130))
    PY += 10

    pcg_probs = meta.get("pcg_probs")
    exec_counts = meta.get("exec_counts", {})
    for i in range(M):
        execs = int(exec_counts.get(i, 0))
        draw.text((PX, PY + 3), f"{short[i]}:{execs}", fill=(80, 180, 130))
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
                        g_c = 60
                        b_c = int(60 + 195 * (0.5 - p) * 2)
                    draw.rectangle([x0, y0, x1, y1], fill=(r_c, g_c, b_c))
                    draw.text((x0 + 1, y0 + 3), f"{p:.2f}", fill=(240, 240, 240))
                else:
                    draw.rectangle([x0, y0, x1, y1], fill=(50, 50, 70))
        PY += cell_h

    # PCG entropy bar
    PY += 3
    max_H = M * (M - 1) * 0.693
    frac  = min(1.0, H_val / max(1e-6, max_H))
    blen  = PANEL_W - 12
    draw.rectangle([PX, PY, PX + blen, PY + 5], fill=(30, 30, 45))
    draw.rectangle([PX, PY, PX + int(blen * frac), PY + 5], fill=(100, 180, 255))
    draw.text((PX, PY + 7), f"H={H_val:.3f}  max\u2248{max_H:.1f}", fill=(140, 160, 200))
    PY += 20

    # SIG section
    draw.line([(PX, PY), (PX + PANEL_W - 10, PY)], fill=(35, 35, 50)); PY += 4
    draw.text((PX, PY), "SIG  skill plan", fill=(100, 180, 255)); PY += 12
    for skill_name, is_achieved, is_active in meta.get("sig_skills", []):
        if is_active:
            fg, mk = (80, 255, 120), "\u25b6"
        elif is_achieved:
            fg, mk = (60, 160, 60), "\u2713"
        else:
            fg, mk = (100, 100, 120), "\u25cb"
        draw.text((PX, PY), f"{mk} {skill_name}", fill=fg); PY += 13

    # EVGS state bars
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
        if len(line_buf) + len(w) + 1 > 26:
            draw.text((PX, PY), line_buf, fill=(200, 160, 80)); PY += 12; line_buf = w
        else:
            line_buf = (line_buf + " " + w).strip()
    if line_buf:
        draw.text((PX, PY), line_buf, fill=(200, 160, 80)); PY += 12

    # Bottom metrics
    PY += 3
    draw.line([(PX, PY), (PX + PANEL_W - 10, PY)], fill=(35, 35, 50)); PY += 5
    draw.text((PX, PY), f"score {score:.0f}", fill=(200, 220, 100)); PY += 13
    draw.text((PX, PY), f"phase {phase}",
              fill=_PHASE_COLOR.get(phase, (180, 180, 180)))

    return np.array(canvas)


# ---------------------------------------------------------------------------
# SIG: 7-skill hierarchy (no hard-coded prerequisites)
# ---------------------------------------------------------------------------

def build_coinrun_sig(evgs) -> SIGraph:
    names = evgs.names()
    idx   = {n: i for i, n in enumerate(names)}
    M     = len(names)
    sig   = SIGraph()

    # UP-predicate skills (skill_id == var_index)
    s_see   = Skill(skill_id=idx["coin_visible"],
                    subgoal=Subgoal(idx["coin_visible"],   Predicate.UP), name="see_coin\u2191")
    s_close = Skill(skill_id=idx["coin_close"],
                    subgoal=Subgoal(idx["coin_close"],     Predicate.UP), name="approach\u2191")
    s_coll  = Skill(skill_id=idx["coin_collected"],
                    subgoal=Subgoal(idx["coin_collected"], Predicate.UP), name="collect\u2191")
    s_climb = Skill(skill_id=idx["platform_above"],
                    subgoal=Subgoal(idx["platform_above"], Predicate.UP), name="climb\u2191")

    # DOWN-predicate skills (skill_id = var_index + M to avoid collision)
    s_level = Skill(skill_id=idx["coin_elevated"]    + M,
                    subgoal=Subgoal(idx["coin_elevated"],    Predicate.DOWN), name="level_coin\u2193")
    s_dodge = Skill(skill_id=idx["saw_visible"]      + M,
                    subgoal=Subgoal(idx["saw_visible"],      Predicate.DOWN), name="dodge\u2193")
    s_evade = Skill(skill_id=idx["creature_visible"] + M,
                    subgoal=Subgoal(idx["creature_visible"], Predicate.DOWN), name="evade\u2193")

    for s in [s_see, s_close, s_coll, s_climb, s_level, s_dodge, s_evade]:
        sig.add_skill(s)

    return sig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--macro_steps",       type=int,   default=150)
    ap.add_argument("--option_steps",      type=int,   default=500)
    ap.add_argument("--out",               type=str,   default="coinrun_dia_v4_sip.mp4")
    ap.add_argument("--fps",               type=int,   default=30)
    ap.add_argument("--seed",              type=int,   default=0)
    ap.add_argument("--scale",             type=int,   default=6)
    ap.add_argument("--model",             type=str,   default=None)
    ap.add_argument("--n_stack",           type=int,   default=4)
    ap.add_argument("--num_levels",        type=int,   default=200)
    ap.add_argument("--start_level",       type=int,   default=0)
    ap.add_argument("--discover_per_skill",type=int,   default=8,
                    help="Executions of each skill before leaving discovery phase")
    ap.add_argument("--add_threshold",     type=float, default=0.78,
                    help="PCG edge probability threshold to add SIG prerequisite")
    ap.add_argument("--min_obs",           type=int,   default=5,
                    help="Minimum skill executions before any edge can be confirmed")
    args = ap.parse_args()

    # ── environment ────────────────────────────────────────────────────────
    base_env = gym.make("procgen:procgen-coinrun-v0",
                        num_levels=args.num_levels, start_level=args.start_level)
    wrapped  = wrap_procgen_coinrun_env(base_env, CoinRunDetectorConfig())

    evgs      = make_coinrun_evgs()
    var_names = evgs.names()
    M         = len(var_names)
    idx       = {n: i for i, n in enumerate(var_names)}

    see_id   = idx["coin_visible"]
    coin_id  = idx["coin_collected"]
    elev_id  = idx["coin_elevated"]
    plat_id  = idx["platform_above"]
    saw_id   = idx["saw_visible"]
    creat_id = idx["creature_visible"]

    env = FrameCapture(wrapped, evgs=evgs, var_names=var_names)

    # ── InterventionalPCG ──────────────────────────────────────────────────
    ipcg = InterventionalPCG(num_vars=M, alpha=1.0, min_obs=args.min_obs)

    # ── SIG ────────────────────────────────────────────────────────────────
    sig = build_coinrun_sig(evgs)

    # Map: var_index → skill_id  (for all 7 skills)
    var_to_skill = {sk.subgoal.var_index: sid for sid, sk in sig.skills.items()}

    # ── Goal planner ───────────────────────────────────────────────────────
    goal_planner = GoalPlanner(sig)

    # ── Selector ───────────────────────────────────────────────────────────
    selector = InformationDirectedSelector(
        ipcg, sig, goal_planner, var_to_skill,
        discover_per_skill=args.discover_per_skill,
        goal_entropy_frac=0.30,
    )

    # ── PPO model ──────────────────────────────────────────────────────────
    ppo_model = None
    if args.model:
        print(f"  Loading PPO model: {args.model}")
        from stable_baselines3 import PPO as SB3PPO
        ppo_model = SB3PPO.load(args.model)
        print(f"  PPO loaded  (n_stack={args.n_stack})")
    else:
        print("  No --model: skills use fixed/random actions")

    ppo_cfg = OptionConfig(max_steps=args.option_steps, terminate_on_success=True)

    # ── Option factory ─────────────────────────────────────────────────────
    def option_factory(skill):
        sg    = skill.subgoal
        vname = var_names[sg.var_index] if sg.var_index < M else "?"

        if vname == "coin_collected" and ppo_model is not None:
            exp  = PixelStackPPOOption(sg, ppo_cfg, ppo_model, args.n_stack, deterministic=False)
            expl = PixelStackPPOOption(sg, ppo_cfg, ppo_model, args.n_stack, deterministic=True)
            return PhaseAwareOption(sg, ppo_cfg, exp, expl, selector)

        if vname == "coin_visible":
            return FixedActionOption(sg, OptionConfig(max_steps=100, terminate_on_success=True),
                                     action=ACTION_RIGHT)
        if vname == "coin_close":
            return FixedActionOption(sg, OptionConfig(max_steps=150, terminate_on_success=True),
                                     action=ACTION_RIGHT)
        if vname == "platform_above":
            return FixedActionOption(sg, OptionConfig(max_steps=60, terminate_on_success=True),
                                     action_cycle=[ACTION_RIGHT_UP, ACTION_RIGHT_UP,
                                                   ACTION_RIGHT, ACTION_UP])
        if vname == "coin_elevated":
            return FixedActionOption(sg, OptionConfig(max_steps=80, terminate_on_success=True),
                                     action_cycle=[ACTION_RIGHT_UP, ACTION_RIGHT_UP, ACTION_RIGHT])
        if vname == "saw_visible":
            return FixedActionOption(sg, OptionConfig(max_steps=60, terminate_on_success=True),
                                     action_cycle=[ACTION_RIGHT_UP, ACTION_RIGHT_UP,
                                                   ACTION_RIGHT, ACTION_RIGHT])
        if vname == "creature_visible":
            return FixedActionOption(sg, OptionConfig(max_steps=60, terminate_on_success=True),
                                     action_cycle=[ACTION_RIGHT_UP, ACTION_RIGHT_UP,
                                                   ACTION_RIGHT, ACTION_RIGHT])

        return RandomOption(sg, OptionConfig(max_steps=80), env.action_space)

    # Option registry (lazy)
    options: dict = {}

    def get_option(skill):
        if skill.skill_id not in options:
            options[skill.skill_id] = option_factory(skill)
        return options[skill.skill_id]

    # ── SIG auto-expansion config (conservative) ───────────────────────────
    sig_expand_cfg = AutoSIGConfig(
        add_threshold=args.add_threshold,
        remove_threshold=args.add_threshold - 0.20,
        create_missing_skills=False,   # don't auto-create; we have all 7
        verbose=False,
    )

    # ── Reset ──────────────────────────────────────────────────────────────
    env.reset()
    env._pending = []

    task_goal      = Subgoal(var_index=coin_id, predicate=Predicate.UP)
    achieved: list = []
    score          = 0.0
    prev_sig_edges = set()
    edge_log: list = []

    policy_str = f"PPO({args.model})" if args.model else "fixed/random"
    print("=" * 72)
    print(f"  DIA CoinRun v4 (SIP)  |  macro_steps={args.macro_steps}")
    print(f"  policy={policy_str}  option_steps={args.option_steps}")
    print(f"  discover_per_skill={args.discover_per_skill}  "
          f"add_threshold={args.add_threshold}  min_obs={args.min_obs}")
    print(f"  vars ({M}): {var_names}")
    print("=" * 72)

    for t in range(args.macro_steps):
        # ── Select skill ────────────────────────────────────────────────────
        phase    = selector.phase()
        skill_id = selector.select(achieved, task_goal)
        skill    = sig.skills[skill_id]
        option   = get_option(skill)

        # ── Execute option ──────────────────────────────────────────────────
        obs = env.get_obs()
        x0  = evgs.extract(obs)
        out = option.run(env, evgs)
        success: bool = bool(out["success"])
        x1  = evgs.extract(out["final_obs"])

        # Skill stats
        delta_x = x1 - x0
        skill.update_stats(success=success, delta_x=delta_x)

        # ── Interventional PCG update ──────────────────────────────────────
        # Only update the row for the variable this skill targeted.
        # This is the key change: each skill execution is a targeted intervention.
        cause_var = skill.subgoal.var_index
        ipcg.update(cause_var, x0, x1)

        # ── Conservative SIG auto-expansion ───────────────────────────────
        # Gate on minimum observation count before trusting any edge.
        if float(ipcg.exec_count[cause_var]) >= args.min_obs:
            expand_sig_from_pcg(sig, evgs, ipcg.probs, sig_expand_cfg)

        # ── Score + episode reset ──────────────────────────────────────────
        if rec_success := success:
            if skill_id == coin_id:
                score   += 10.0
                achieved = []
            elif skill_id not in achieved:
                achieved.append(skill_id)

        # coin_visible drops when a new level starts → reset episode tracking
        if delta_x[see_id] < -0.5:
            achieved = []

        # ── Detect new SIG edges ───────────────────────────────────────────
        cur_sig_edges = set()
        for sid, sk in sig.skills.items():
            for pre in sig.prerequisites(sid):
                cur_sig_edges.add((pre, sid))
        for (pre, post) in cur_sig_edges - prev_sig_edges:
            pn  = sig.skills[pre].name  if pre  in sig.skills else str(pre)
            pst = sig.skills[post].name if post in sig.skills else str(post)
            msg = f"  *** DISCOVERED: {pn} → {pst} (step {t+1}) ***"
            print(msg)
            edge_log.append(f"step {t+1}: {pn}→{pst}")
        prev_sig_edges = cur_sig_edges

        # ── Assemble frame metadata ────────────────────────────────────────
        H = ipcg.entropy()
        try:
            x_now = evgs.extract(env.get_obs())
        except Exception:
            x_now = np.zeros(M)

        sig_skills_list = []
        for sid in sig.toposort():
            if sid not in sig.skills:
                continue
            sk    = sig.skills[sid]
            pres  = sig.prerequisites(sid)
            pre_names = "+".join(sig.skills[p].name for p in pres if p in sig.skills)
            label = sk.name + (f"  [{pre_names}]" if pre_names else "  [free]")
            sig_skills_list.append((label, sid in achieved, skill_id == sid))

        sig_edges_str = " | ".join(
            f"{sig.skills[p].name}\u2192{sig.skills[s].name}"
            for (p, s) in sorted(cur_sig_edges)
            if p in sig.skills and s in sig.skills
        ) or "none yet"

        meta = {
            "macro_step":  t + 1,
            "phase":       phase,
            "skill_name":  skill.name,
            "success":     success,
            "H":           H,
            "score":       score,
            "pcg_probs":   ipcg.probs.copy(),
            "exec_counts": {i: int(ipcg.exec_count[i]) for i in range(M)},
            "sig_skills":  sig_skills_list,
            "sig_edges":   sig_edges_str,
            "var_vals":    dict(zip(var_names, x_now.tolist())),
        }
        env.flush(meta)

        # ── Console log ────────────────────────────────────────────────────
        execs_str = " ".join(f"{_short(var_names[i])}={int(ipcg.exec_count[i])}"
                             for i in range(M))
        print(f"  [{t+1:03d}] {phase:<8} {skill.name:<16} "
              f"succ={int(success)}  H={H:5.2f}  "
              f"vis={x_now[see_id]:.0f} saw={x_now[saw_id]:.0f} "
              f"creat={x_now[creat_id]:.0f}  "
              f"score={score:.0f}  | {execs_str}")

    env.close()

    if not env.frames:
        print("ERROR: no frames captured.")
        return

    print("\n" + "=" * 72)
    print(f"Causal edges discovered ({len(edge_log)}):")
    for e in edge_log:
        print(f"  {e}")
    print("=" * 72)

    print(f"\nWriting {len(env.frames)} frames → {args.out}  (fps={args.fps})")
    first  = make_frame(env.frames[0][0], env.frames[0][1], args.scale, var_names)
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

    print(f"Done.  score={score:.0f}")
    print(f"Video: {args.out}")


if __name__ == "__main__":
    main()
