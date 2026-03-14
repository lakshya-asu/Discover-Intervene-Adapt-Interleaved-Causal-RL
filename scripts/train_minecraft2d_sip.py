#!/usr/bin/env python3
"""
scripts/train_minecraft2d_sip.py  —  Phase 1: DIA on 2D Minecraft (SIP algorithm).

Discovers the crafting causal chain via Structured Interventional Probing:
  wood, stone → stonepickaxe → ironore → furnace + coal + ironore → iron
  iron + wood → ironpickaxe → diamond

Key differences from train_minecraft2d_dia.py:
  1. Round-robin discovery: each skill tried equally before trusting PCG structure.
  2. Empty SIG: no hardcoded edges — all discovered from PCG via expand_sig_from_pcg.
  3. Per-step trajectory buffer: all primitive transitions within each option are
     collected, giving SimplePCG clean observational data.
  4. Saves pcg_2d.npy + sig_2d.json for transfer to 3D Minecraft.
  5. PIL-rendered video showing PCG heatmap, EVGS state, and discovered edges.

Output
------
  pcg_2d.npy      — 9×9 float32 edge-probability matrix
  sig_2d.json     — discovered prerequisite edges {edges: [[pre, post], ...], names: [...]}
  minecraft2d_sip.mp4  — video of learning progress
"""
from __future__ import annotations

import argparse
import json
import numpy as np

from dia.envs.minecraft2d import MinecraftChainEnv, ChainConfig
from dia.evgs_minecraft import make_minecraft_evgs, VAR_NAMES
from dia.sig import SIGraph, Skill
from dia.types import Subgoal, Predicate
from dia.pcg import SimplePCG, PCGConfig
from dia.sig_auto import expand_sig_from_pcg, AutoSIGConfig
from dia.options import OptionConfig
from dia.options_minecraft import ResourceOption

try:
    from PIL import Image, ImageDraw
    PIL_OK = True
except Exception:
    PIL_OK = False

try:
    import imageio_ffmpeg
    FFMPEG_OK = True
except Exception:
    FFMPEG_OK = False


# ---------------------------------------------------------------------------
# PCG fitting from transition buffer (observational: P(j↑ | i=1))
# ---------------------------------------------------------------------------

def fit_probs_from_buffer(buffer: list, num_vars: int, alpha: float = 1.0) -> np.ndarray:
    """
    Compute observational edge probabilities from (x_t, x_tp1) transition pairs.

    probs[i, j] = P(j increases | i was 1 at time t)
                = Bayesian estimate with Laplace smoothing alpha.

    For 2D Minecraft's deterministic crafting chain, this correctly identifies
    prerequisites: when stone=1, stonepickaxe gets crafted → probs[stone, pickaxe] ↑.
    """
    if len(buffer) < 2:
        return None

    X_t   = np.stack([x  for x, _  in buffer], axis=0).astype(float)   # (N, M)
    X_tp1 = np.stack([x_ for _,  x_ in buffer], axis=0).astype(float)  # (N, M)
    M = num_vars

    probs = np.full((M, M), 0.5, dtype=float)
    np.fill_diagonal(probs, 0.0)

    for i in range(M):
        mask_i1 = X_t[:, i] > 0.5   # rows where var i was present
        mask_i0 = ~mask_i1
        for j in range(M):
            if i == j:
                continue
            j_up = (X_tp1[:, j] - X_t[:, j]) > 0.5

            n1 = float(mask_i1.sum())
            s1 = float((mask_i1 & j_up).sum())
            n0 = float(mask_i0.sum())

            # Bayesian estimate: P(j↑ | i=1)
            probs[i, j] = (s1 + alpha) / (n1 + 2.0 * alpha)

    return probs


# ---------------------------------------------------------------------------
# SIP selector (round-robin discover → confirm → goal)
# ---------------------------------------------------------------------------

class MinecraftSIPSelector:
    """
    Three-phase SIP selector for 2D Minecraft.

    Maintain exec_count separately from the PCG (SimplePCG has no exec_count).
    Phase transitions:
      discover → confirm:  all skills tried >= discover_per_skill times
      confirm  → goal:     total PCG entropy < goal_entropy_frac × max_entropy
    """

    def __init__(self, pcg: SimplePCG, sig: SIGraph, var_to_skill: dict,
                 discover_per_skill: int = 10,
                 goal_entropy_frac: float = 0.25):
        self.pcg                = pcg
        self.sig                = sig
        self.var_to_skill       = var_to_skill      # var_index → skill_id
        self.exec_count: dict   = {k: 0 for k in var_to_skill}
        self.discover_per_skill = discover_per_skill
        self.goal_entropy_frac  = goal_entropy_frac

    def mark_executed(self, var_k: int):
        self.exec_count[var_k] = self.exec_count.get(var_k, 0) + 1

    def phase(self) -> str:
        for var_k in self.var_to_skill:
            if self.exec_count.get(var_k, 0) < self.discover_per_skill:
                return "discover"
        M   = len(self.var_to_skill)
        H   = self.pcg.entropy()
        Hmax = M * (M - 1) * 0.693
        if H < self.goal_entropy_frac * max(1e-6, Hmax):
            return "goal"
        return "confirm"

    def _row_entropy(self, k: int) -> float:
        """Shannon entropy of PCG row k (uncertainty about what var_k enables)."""
        eps = 1e-8
        row = self.pcg.probs[k].copy()
        row[k] = 0.5
        p = np.clip(row, eps, 1.0 - eps)
        return float(np.sum(-p * np.log(p) - (1.0 - p) * np.log(1.0 - p)))

    def select(self, achieved: list, task_goal_var: int = 8) -> int:
        ph = self.phase()

        if ph == "discover":
            # Least-executed skill (ties → lowest var_index)
            return min(
                self.var_to_skill.values(),
                key=lambda sid: self.exec_count.get(
                    self.sig.skills[sid].subgoal.var_index, 0),
            )

        if ph == "confirm":
            # Skill whose row has highest entropy (we're most uncertain about its effects)
            return max(
                self.var_to_skill.values(),
                key=lambda sid: self._row_entropy(
                    self.sig.skills[sid].subgoal.var_index),
            )

        # ── goal phase ────────────────────────────────────────────────────────
        # Follow topological order from SIG; pick the first ready skill
        # that is not yet achieved.
        achieved_set = set(achieved)
        topo = self.sig.toposort()
        for sid in topo:
            if sid not in self.sig.skills:
                continue
            if sid in achieved_set:
                continue
            prereqs = self.sig.prerequisites(sid)
            if all(p in achieved_set for p in prereqs):
                return sid

        # Fallback: skill with highest success_rate among ready skills
        ready = self.sig.ready_skills(achieved) or list(self.sig.skills.keys())
        return max(ready, key=lambda sid: self.sig.skills[sid].success_rate)


# ---------------------------------------------------------------------------
# PIL rendering — PCG heatmap + EVGS + SIG edges
# ---------------------------------------------------------------------------

_PHASE_COL = {
    "discover": (80,  200, 255),
    "confirm":  (255, 200,  80),
    "goal":     (80,  255, 120),
}

_SHORT = {n: n[:4].upper() for n in VAR_NAMES}

W_CANVAS, H_CANVAS = 720, 600


def _short(name: str) -> str:
    return name[:4].upper()


def render_frame(pcg: SimplePCG, var_names: list, exec_count: dict,
                 meta: dict) -> np.ndarray:
    if not PIL_OK:
        return np.zeros((H_CANVAS, W_CANVAS, 3), dtype=np.uint8)

    canvas = Image.new("RGB", (W_CANVAS, H_CANVAS), color=(12, 12, 20))
    draw   = ImageDraw.Draw(canvas)
    M      = len(var_names)
    short  = [_short(n) for n in var_names]

    phase   = meta.get("phase", "?")
    step    = meta.get("step", 0)
    skill   = meta.get("skill_name", "?")
    success = meta.get("success", False)
    H_val   = meta.get("H", 0.0)
    col     = _PHASE_COL.get(phase, (200, 200, 200))
    tick    = "\u2713" if success else "\u00b7"

    # ── title bar ─────────────────────────────────────────────────────────────
    draw.rectangle([0, 0, W_CANVAS, 20], fill=(0, 0, 0))
    draw.text((4, 3),
              f"[{step:03d}] {phase:<8}  {skill:<18}  H={H_val:.3f}  {tick}",
              fill=col)

    # ── PCG heatmap ──────────────────────────────────────────────────────────
    PY = 28
    draw.text((4, PY), "PCG  P(j\u2191 | var_i active)   [rows=cause, cols=effect]",
              fill=(100, 180, 255))
    PY += 14

    LABEL  = 52
    avail  = W_CANVAS // 2 - LABEL - 8
    cell_w = max(1, avail // M)
    cell_h = 26

    # column headers
    for j in range(M):
        draw.text((4 + LABEL + j * cell_w + 1, PY), short[j], fill=(80, 180, 130))
    PY += 12

    probs = pcg.probs
    for i in range(M):
        execs = int(exec_count.get(i, 0))
        draw.text((4, PY + 5), f"{short[i]}:{execs:2d}", fill=(80, 180, 130))
        for j in range(M):
            x0 = 4 + LABEL + j * cell_w
            y0 = PY
            x1 = x0 + cell_w - 2
            y1 = y0 + cell_h - 2
            if i == j:
                draw.rectangle([x0, y0, x1, y1], fill=(20, 20, 30))
            else:
                p = float(np.clip(probs[i, j], 0, 1))
                if p >= 0.5:
                    r_c = int(60 + 195 * (p - 0.5) * 2)
                    g_c = int(60 - 40  * (p - 0.5) * 2)
                    b_c = 60
                else:
                    r_c = 60
                    g_c = 60
                    b_c = int(60 + 195 * (0.5 - p) * 2)
                draw.rectangle([x0, y0, x1, y1], fill=(r_c, g_c, b_c))
                draw.text((x0 + 1, y0 + 4), f"{p:.2f}", fill=(220, 220, 220))
        PY += cell_h

    # ── PCG entropy bar ───────────────────────────────────────────────────────
    PY += 4
    max_H  = M * (M - 1) * 0.693
    frac   = min(1.0, H_val / max(1e-6, max_H))
    blen   = W_CANVAS // 2 - 12
    draw.rectangle([4, PY, 4 + blen, PY + 6], fill=(30, 30, 45))
    draw.rectangle([4, PY, 4 + int(blen * frac), PY + 6], fill=(100, 180, 255))
    draw.text((4, PY + 8), f"H={H_val:.3f}  max\u2248{max_H:.1f}", fill=(140, 160, 200))
    PY += 22

    # ── right panel: EVGS state ───────────────────────────────────────────────
    RX  = W_CANVAS // 2 + 8
    RPY = 28
    draw.text((RX, RPY), "EVGS  state", fill=(100, 180, 255)); RPY += 14
    bar_max = W_CANVAS // 2 - 20
    var_vals = meta.get("var_vals", {})
    for vname in var_names:
        v  = float(var_vals.get(vname, 0.0))
        fg = (80, 230, 80) if v > 0.5 else (80, 80, 100)
        bl = int(bar_max * min(1.0, v))
        draw.rectangle([RX, RPY + 12, RX + bar_max, RPY + 15], fill=(25, 25, 35))
        if bl > 0:
            draw.rectangle([RX, RPY + 12, RX + bl, RPY + 15], fill=fg)
        draw.text((RX, RPY), f"{vname}: {v:.0f}", fill=fg)
        RPY += 20

    # ── right panel: SIG edges ────────────────────────────────────────────────
    RPY += 8
    draw.line([(RX, RPY), (W_CANVAS - 4, RPY)], fill=(35, 35, 50)); RPY += 6
    draw.text((RX, RPY), "SIG  discovered edges", fill=(100, 180, 255)); RPY += 14
    for edge_str in meta.get("sig_edges_list", []):
        draw.text((RX, RPY), edge_str, fill=(200, 160, 80)); RPY += 13
    if not meta.get("sig_edges_list"):
        draw.text((RX, RPY), "(none yet)", fill=(80, 80, 100)); RPY += 13

    # ── right panel: phase indicator ─────────────────────────────────────────
    RPY += 6
    draw.line([(RX, RPY), (W_CANVAS - 4, RPY)], fill=(35, 35, 50)); RPY += 6
    draw.text((RX, RPY), f"Phase: {phase}", fill=col); RPY += 14
    diamonds = meta.get("diamonds", 0)
    draw.text((RX, RPY), f"Diamonds mined: {diamonds}", fill=(200, 220, 100))

    return np.array(canvas)


# ---------------------------------------------------------------------------
# SIG build (empty — no hardcoded edges; all discovered from PCG)
# ---------------------------------------------------------------------------

def build_empty_minecraft_sig(evgs) -> SIGraph:
    names = evgs.names()
    sig   = SIGraph()
    for i, name in enumerate(names):
        sg  = Subgoal(var_index=i, predicate=Predicate.UP)
        sk  = Skill(skill_id=i, subgoal=sg, name=f"{name}\u2191")
        sig.add_skill(sk)
    # NO hardcoded prerequisite edges — all discovered from PCG
    return sig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Phase 1: learn Minecraft crafting PCG via SIP on 2D symbolic env.")
    ap.add_argument("--macro_steps",        type=int,   default=200,
                    help="Number of macro-steps (one option per step)")
    ap.add_argument("--option_steps",       type=int,   default=50,
                    help="Max primitive steps per option execution")
    ap.add_argument("--discover_per_skill", type=int,   default=12,
                    help="Times each skill must be tried before leaving discover phase")
    ap.add_argument("--fit_every",          type=int,   default=5,
                    help="Refit PCG every N macro-steps")
    ap.add_argument("--min_buffer",         type=int,   default=50,
                    help="Min transitions before PCG fitting starts")
    ap.add_argument("--reset_every",        type=int,   default=25,
                    help="Reset env every N macro-steps so crafting items respawn "
                         "(prevents ResourceOption from becoming a noop after first acquisition)")
    ap.add_argument("--add_threshold",      type=float, default=0.72,
                    help="PCG edge threshold for adding SIG prerequisite")
    ap.add_argument("--min_obs",            type=int,   default=8,
                    help="Min skill executions before any SIG edge is confirmed")
    ap.add_argument("--pcg_out",            type=str,   default="pcg_2d.npy",
                    help="Output file for PCG edge-probability matrix")
    ap.add_argument("--sig_out",            type=str,   default="sig_2d.json",
                    help="Output file for discovered SIG edges")
    ap.add_argument("--out",                type=str,   default="minecraft2d_sip.mp4",
                    help="Output video file")
    ap.add_argument("--fps",                type=int,   default=10)
    ap.add_argument("--seed",               type=int,   default=0)
    ap.add_argument("--use_known_edges",    action="store_true",
                    help="After PCG fitting, also add the known ground-truth crafting "
                         "edges to the SIG. Use for demos when PCG signal is too weak "
                         "to cross the threshold alone (observational dilution in count envs). "
                         "PCG learning still runs; this just ensures a valid plan is exported.")
    args = ap.parse_args()

    # ── env + EVGS ────────────────────────────────────────────────────────────
    env       = MinecraftChainEnv(ChainConfig(), seed=args.seed)
    evgs      = make_minecraft_evgs()
    var_names = evgs.names()
    M         = len(var_names)
    idx       = {n: i for i, n in enumerate(var_names)}

    # ── PCG (empty, observational, will be fit from buffer) ──────────────────
    pcg = SimplePCG(PCGConfig(num_vars=M, init_edge_prob=0.05, seed=args.seed))

    # ── SIG (empty — no hardcoded edges) ────────────────────────────────────
    sig          = build_empty_minecraft_sig(evgs)
    var_to_skill = {sk.subgoal.var_index: sid for sid, sk in sig.skills.items()}

    # ── SIP selector ─────────────────────────────────────────────────────────
    selector = MinecraftSIPSelector(
        pcg, sig, var_to_skill,
        discover_per_skill=args.discover_per_skill,
        goal_entropy_frac=0.25,
    )

    # ── Options (ResourceOption per skill) ───────────────────────────────────
    options = {
        i: ResourceOption(Subgoal(i, Predicate.UP),
                          OptionConfig(max_steps=args.option_steps, terminate_on_success=True))
        for i in range(M)
    }

    # ── SIG auto-expansion config ─────────────────────────────────────────────
    sig_cfg = AutoSIGConfig(
        add_threshold=args.add_threshold,
        remove_threshold=args.add_threshold - 0.20,
        create_missing_skills=False,
        verbose=False,
    )

    # ── Transition buffer ─────────────────────────────────────────────────────
    buffer: list = []   # list of (x_t, x_tp1) from ALL primitive steps in all options
    MAX_BUFFER = 5_000

    # ── State ─────────────────────────────────────────────────────────────────
    obs       = env.reset()
    achieved: list = []
    diamonds  = 0
    frames:   list = []
    edge_log: list = []
    prev_sig_edges: set = set()
    last_reset = 0

    print("=" * 70)
    print(f"  DIA 2D Minecraft SIP  |  macro_steps={args.macro_steps}")
    print(f"  discover_per_skill={args.discover_per_skill}  "
          f"add_threshold={args.add_threshold}  min_obs={args.min_obs}")
    print(f"  vars ({M}): {var_names}")
    print("=" * 70)

    for t in range(args.macro_steps):
        phase    = selector.phase()
        skill_id = selector.select(achieved, task_goal_var=idx["diamond"])
        skill    = sig.skills[skill_id]
        option   = options[skill_id]
        var_k    = skill.subgoal.var_index

        # ── Periodic env reset (forces fresh crafting trajectories) ──────────
        # Without resets, ResourceOption becomes noop once items are acquired;
        # the buffer never sees "stonepickaxe crafted from wood+stone" again.
        if args.reset_every > 0 and (t - last_reset) >= args.reset_every:
            env.reset()
            # Do NOT reset achieved: PCG/SIG learning is cumulative across episodes
            last_reset = t

        # ── Get state before option ──────────────────────────────────────────
        x0  = evgs.extract(env.get_obs())

        # ── Execute option (ResourceOption handles prerequisite gathering) ───
        out     = option.run(env, evgs)
        success = bool(out["success"])
        x1      = evgs.extract(out["final_obs"])

        # ── Collect per-step transitions from trajectory ───────────────────
        # Each (obs_t, action, obs_tp1, succ) yields one (x_t, x_tp1) pair.
        # This gives SimplePCG the intermediate transitions where prerequisites
        # are present just before effects occur — the correct observational signal.
        for (obs_t, _act, obs_tp1, _succ) in out["trajectory"]:
            xt  = evgs.extract(obs_t)
            xtp1 = evgs.extract(obs_tp1)
            if not np.allclose(xt, xtp1):   # only store actual changes
                buffer.append((xt, xtp1))
        # Trim buffer
        if len(buffer) > MAX_BUFFER:
            buffer = buffer[-MAX_BUFFER:]

        # ── Refit PCG from buffer ─────────────────────────────────────────
        if len(buffer) >= args.min_buffer and (t + 1) % args.fit_every == 0:
            new_probs = fit_probs_from_buffer(buffer, M, alpha=1.0)
            if new_probs is not None:
                ig = pcg.apply_update(new_probs)

        # ── Skill stats + exec_count ──────────────────────────────────────
        delta_x = x1 - x0
        skill.update_stats(success=success, delta_x=delta_x)
        selector.mark_executed(var_k)

        # ── Conservative SIG expansion ────────────────────────────────────
        if selector.exec_count.get(var_k, 0) >= args.min_obs:
            expand_sig_from_pcg(sig, evgs, pcg.probs, sig_cfg)

        # ── Track achievements ────────────────────────────────────────────
        if success:
            if skill_id not in achieved:
                achieved.append(skill_id)
            if var_k == idx["diamond"]:
                diamonds += 1

        # ── Detect new SIG edges ──────────────────────────────────────────
        cur_edges: set = set()
        for sid in sig.skills:
            for pre in sig.prerequisites(sid):
                cur_edges.add((pre, sid))
        for (pre, post) in cur_edges - prev_sig_edges:
            pname  = sig.skills[pre].name  if pre  in sig.skills else str(pre)
            ptname = sig.skills[post].name if post in sig.skills else str(post)
            msg    = f"  *** DISCOVERED: {pname} \u2192 {ptname} (step {t+1}) ***"
            print(msg)
            edge_log.append(f"step {t+1}: {pname}\u2192{ptname}")
        prev_sig_edges = cur_edges

        # ── Render frame ──────────────────────────────────────────────────
        H     = pcg.entropy()
        x_now = evgs.extract(env.get_obs())
        edges_list = [
            f"{sig.skills[p].name}\u2192{sig.skills[s].name}"
            for (p, s) in sorted(cur_edges)
            if p in sig.skills and s in sig.skills
        ]
        meta = {
            "step":          t + 1,
            "phase":         phase,
            "skill_name":    skill.name,
            "success":       success,
            "H":             H,
            "var_vals":      dict(zip(var_names, x_now.tolist())),
            "sig_edges_list": edges_list,
            "diamonds":      diamonds,
        }
        frame = render_frame(pcg, var_names, selector.exec_count, meta)
        frames.append(frame)

        # ── Console log ───────────────────────────────────────────────────
        execs_str = " ".join(f"{var_names[i][:4]}={int(selector.exec_count.get(i, 0))}"
                             for i in range(M))
        print(f"  [{t+1:03d}] {phase:<8} {skill.name:<18} "
              f"succ={int(success)}  H={H:5.3f}  buf={len(buffer):4d}  "
              f"diamonds={diamonds} | {execs_str}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Discovered edges ({len(edge_log)}):")
    for e in edge_log:
        print(f"  {e}")

    print("\nFinal PCG (top edges with prob >= 0.5):")
    for i in range(M):
        for j in range(M):
            if i != j and pcg.probs[i, j] >= 0.5:
                print(f"  {var_names[i]} \u2192 {var_names[j]}: {pcg.probs[i,j]:.3f}")

    # ── Optionally supplement SIG with known ground-truth edges ──────────────
    # The observational PCG correctly RANKS the causal edges (coal→iron highest)
    # but diluted buffer values may not cross the threshold for crafting edges.
    # This flag adds the known Minecraft crafting prerequisites explicitly,
    # complementing what PCG learning found via the data-driven signal.
    if args.use_known_edges:
        print("\nAdding known Minecraft crafting edges to SIG:")
        gt_edges = [
            ("stone", "furnace"), ("wood", "stonepickaxe"), ("stone", "stonepickaxe"),
            ("stonepickaxe", "ironore"), ("furnace", "iron"), ("coal", "iron"),
            ("ironore", "iron"), ("iron", "ironpickaxe"), ("wood", "ironpickaxe"),
            ("ironpickaxe", "diamond"),
        ]
        for (pre_name, post_name) in gt_edges:
            pre_id  = idx[pre_name]
            post_id = idx[post_name]
            sig.add_prerequisite(pre_id, post_id)
            print(f"  {pre_name} → {post_name}  (pcg={pcg.probs[pre_id, post_id]:.3f})")

    # ── Save PCG probs ────────────────────────────────────────────────────────
    np.save(args.pcg_out, pcg.probs.astype(np.float32))
    print(f"\nSaved PCG probs: {args.pcg_out}  shape={pcg.probs.shape}")

    # ── Save SIG edges ────────────────────────────────────────────────────────
    edges_list_final = []
    for sid in sig.skills:
        for pre in sig.prerequisites(sid):
            edges_list_final.append([pre, sid])
    sig_data = {
        "names":    var_names,
        "edges":    edges_list_final,
        "edge_log": edge_log,
    }
    with open(args.sig_out, "w") as f:
        json.dump(sig_data, f, indent=2)
    print(f"Saved SIG:       {args.sig_out}  ({len(edges_list_final)} edges)")

    # ── Write video ───────────────────────────────────────────────────────────
    if not frames:
        print("No frames captured.")
        return

    if not FFMPEG_OK:
        print(f"imageio_ffmpeg not available; skipping video write.")
        return

    H_px, W_px = frames[0].shape[:2]
    W_px = (W_px // 16) * 16
    H_px = (H_px // 16) * 16
    print(f"\nWriting {len(frames)} frames → {args.out}  (fps={args.fps})")
    gen = imageio_ffmpeg.write_frames(args.out, size=(W_px, H_px), fps=args.fps,
                                      quality=8, codec="libx264")
    gen.send(None)
    for frame in frames:
        gen.send(frame[:H_px, :W_px].tobytes())
    gen.close()
    print(f"Done.  Video: {args.out}")
    print(f"Diamonds mined: {diamonds}")


if __name__ == "__main__":
    main()
