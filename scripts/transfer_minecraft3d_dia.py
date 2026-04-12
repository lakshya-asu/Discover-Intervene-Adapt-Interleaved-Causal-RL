#!/usr/bin/env python3
"""
scripts/transfer_minecraft3d_dia.py  —  Phase 2: PCG-guided execution in 3D Minecraft.

Loads the causal structure discovered in Phase 1 (2D symbolic Minecraft) and uses
it to guide a 3D MineRL agent that executes PPO skills (or random placeholders).

Transfer mechanism
------------------
1. Load sig_2d.json  → SIG (which skills are prerequisites for which)
2. Build SIG in memory; topological sort gives the causal execution order
3. For each skill in topo order (wood → stone → stonepickaxe → ... → diamond):
   a. Load pre-trained PPO model from models/minerl/skill_{var}.zip  (if available)
      or fall back to RandomMineRLOption as a placeholder
   b. Run option until item acquired or timeout
4. Record video with PCG plan overlay + EVGS inventory state

Why FIXED PRIOR?
----------------
The 2D PCG is used as a frozen plan, not as a live learner.
  - 2D causal structure is the same as 3D (same crafting rules)
  - 2D learning is cheap (symbolic env, no Java server needed)
  - 3D agent just follows the topological execution order from the 2D SIG

Usage
-----
  # Prerequisites:
  #   1. Run Phase 1 to get sig_2d.json:
  #      conda run -n dia python scripts/train_minecraft2d_sip.py
  #   2. (Optional) Train PPO skills:
  #      conda run -n dia-minecraft python scripts/train_minerl_skill.py --var wood
  #   3. Run Phase 2:
  conda run -n dia-minecraft python scripts/transfer_minecraft3d_dia.py \\
      --sig sig_2d.json \\
      --model_dir models/minerl \\
      --out transfer_minecraft3d.mp4

  # Quick smoke test with random policies (no PPO training required):
  conda run -n dia-minecraft python scripts/transfer_minecraft3d_dia.py \\
      --sig sig_2d.json --model_dir models/minerl \\
      --max_steps_per_skill 200 --out transfer_test.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from dia.evgs_minerl import make_minerl_evgs, VAR_NAMES, inventory_summary
from dia.options_minerl import (MineRLObsWrapper, MineRLPPOOption,
                                RandomMineRLOption, SkillScriptedOption,
                                load_skill_option, make_random_option)
from dia.sig import SIGraph, Skill
from dia.types import Subgoal, Predicate
from dia.options import OptionConfig
from dia.pcg import SimplePCG, PCGConfig
from dia.intrinsic import BetaScheduler

try:
    from dia.cwm import CWMConfig, CWMTrainer
    CWM_OK = True
except Exception:
    CWM_OK = False

try:
    from dia.vlm_minerl import MineRLVLMAnalyzer, MineRLVLMConfig, fuse_with_inventory
    VLM_OK = True
except Exception:
    VLM_OK = False

try:
    from dia.clip_skill import CLIPGoalEncoder, SKILL_TEXTS
    CLIP_OK = True
except Exception:
    CLIP_OK = False

try:
    import gym
    GYM_OK = True
except ImportError:
    GYM_OK = False

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


_IMAGE_H, _IMAGE_W = 360, 640   # MineRL default resolution
_PANEL_W   = 280
_PHASE_COL = (80, 255, 120)
_VAR_IDX   = {n: i for i, n in enumerate(VAR_NAMES)}


# ---------------------------------------------------------------------------
# Build 3D SIG from saved 2D sig_2d.json
# ---------------------------------------------------------------------------

def load_sig_from_json(sig_path: str) -> SIGraph:
    """Reconstruct a SIGraph from the JSON file saved by train_minecraft2d_sip.py."""
    with open(sig_path) as f:
        data = json.load(f)

    names = data.get("names", VAR_NAMES)
    edges = data.get("edges", [])   # list of [pre_id, post_id]

    sig = SIGraph()
    for i, name in enumerate(names):
        sg = Subgoal(var_index=i, predicate=Predicate.UP)
        sig.add_skill(Skill(skill_id=i, subgoal=sg, name=f"{name}↑"))

    for (pre, post) in edges:
        sig.add_prerequisite(int(pre), int(post))

    print(f"Loaded SIG from {sig_path}:")
    print(f"  Variables: {names}")
    print(f"  Edges ({len(edges)}): {edges}")
    return sig


# ---------------------------------------------------------------------------
# Load PPO option for a given variable
# ---------------------------------------------------------------------------

def _model_path(model_dir: str, var_name: str) -> str:
    return os.path.join(model_dir, f"skill_{var_name}.zip")


def load_options(model_dir: str, sig: SIGraph,
                 max_steps: int = 2000, deterministic: bool = True,
                 use_scripted_fallback: bool = True) -> dict:
    """
    Load PPO skill options from model_dir.

    Falls back to SkillScriptedOption (task-specific macro behaviors) when no
    trained model exists. Use use_scripted_fallback=False for pure random.

    Returns dict: var_idx → option (MineRLPPOOption or SkillScriptedOption).
    """
    options = {}
    for var_idx, var_name in enumerate(VAR_NAMES):
        path = _model_path(model_dir, var_name)
        sg   = Subgoal(var_index=var_idx, predicate=Predicate.UP)
        cfg  = OptionConfig(max_steps=max_steps, terminate_on_success=True)
        if os.path.exists(path):
            print(f"  [{var_name}] loading PPO model: {path}")
            opt = MineRLPPOOption(sg, cfg, path, deterministic=deterministic)
        elif use_scripted_fallback:
            print(f"  [{var_name}] no model found → using SkillScriptedOption (macro)")
            opt = SkillScriptedOption(sg, cfg, var_name)
        else:
            print(f"  [{var_name}] no model found → using RandomMineRLOption")
            opt = RandomMineRLOption(sg, cfg)
        options[var_idx] = opt
    return options


# ---------------------------------------------------------------------------
# Frame rendering — game frame + info panel
# ---------------------------------------------------------------------------

def make_transfer_frame(rgb_frame: np.ndarray, meta: dict, var_names: list) -> np.ndarray:
    """Render one video frame: game (left) + info panel (right)."""
    GH, GW = rgb_frame.shape[:2]

    if not PIL_OK:
        return rgb_frame

    canvas = Image.new("RGB", (GW + _PANEL_W, GH), color=(12, 12, 20))
    canvas.paste(Image.fromarray(rgb_frame.astype(np.uint8)), (0, 0))
    draw   = ImageDraw.Draw(canvas)

    step      = meta.get("step", 0)
    var_name  = meta.get("var_name", "?")
    success   = meta.get("success", False)
    tick      = "✓" if success else "·"

    # ── title bar ─────────────────────────────────────────────────────────────
    draw.rectangle([0, 0, GW, 18], fill=(0, 0, 0))
    draw.text((3, 2), f"[{step:03d}] goal  skill={var_name}  {tick}",
              fill=_PHASE_COL)

    PX = GW + 4
    PY = 6

    # ── 2D SIG plan ───────────────────────────────────────────────────────────
    draw.text((PX, PY), "Causal Plan  (from 2D PCG)", fill=(100, 180, 255)); PY += 14
    for entry in meta.get("plan_entries", []):
        name, is_done, is_active = entry
        if is_active:
            fg, mk = (80, 255, 120), "▶"
        elif is_done:
            fg, mk = (60, 160, 60),  "✓"
        else:
            fg, mk = (100, 100, 120), "○"
        draw.text((PX, PY), f"{mk} {name}", fill=fg); PY += 13

    # ── EVGS inventory state ──────────────────────────────────────────────────
    PY += 6
    draw.line([(PX, PY), (PX + _PANEL_W - 8, PY)], fill=(35, 35, 50)); PY += 6
    draw.text((PX, PY), "EVGS  (3D inventory)", fill=(100, 180, 255)); PY += 14
    bar_max = _PANEL_W - 12
    for vname, vval in meta.get("var_vals", {}).items():
        v  = float(vval)
        fg = (80, 230, 80) if v > 0.5 else (80, 80, 100)
        bl = int(bar_max * min(1.0, v))
        draw.rectangle([PX, PY + 11, PX + bar_max, PY + 14], fill=(25, 25, 35))
        if bl > 0:
            draw.rectangle([PX, PY + 11, PX + bl, PY + 14], fill=fg)
        draw.text((PX, PY), f"{vname}: {v:.0f}", fill=fg); PY += 18

    # ── Progress ──────────────────────────────────────────────────────────────
    PY += 4
    draw.line([(PX, PY), (PX + _PANEL_W - 8, PY)], fill=(35, 35, 50)); PY += 6
    achieved = meta.get("achieved_names", [])
    draw.text((PX, PY), f"Achieved: {len(achieved)}/9", fill=(200, 220, 100)); PY += 14
    for aname in achieved:
        draw.text((PX + 8, PY), f"✓ {aname}", fill=(80, 200, 80)); PY += 12

    return np.array(canvas)


# ---------------------------------------------------------------------------
# Main execution loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Phase 2: Execute 2D-PCG-guided plan in 3D MineRL.")
    ap.add_argument("--sig",               type=str, default="sig_2d.json",
                    help="Path to sig_2d.json from Phase 1")
    ap.add_argument("--model_dir",         type=str, default="models/minerl",
                    help="Directory containing skill_*.zip PPO models")
    ap.add_argument("--env_id",            type=str,
                    default="MineRLObtainDiamondShovel-v0",
                    help="MineRL environment ID")
    ap.add_argument("--max_steps_per_skill", type=int, default=500,
                    help="Max MineRL steps per skill before giving up")
    ap.add_argument("--out",               type=str, default="transfer_minecraft3d.mp4")
    ap.add_argument("--fps",               type=int, default=20)
    ap.add_argument("--deterministic",     action="store_true", default=True)
    ap.add_argument("--random_fallback",   action="store_true", default=False,
                    help="Use random actions as fallback instead of scripted macros")
    ap.add_argument("--cwm",               type=str, default="",
                    help="Path to pre-trained CWM weights (cwm_2d.pt). "
                         "If provided, enables online PCG adaptation and curiosity logging.")
    ap.add_argument("--pcg",               type=str, default="pcg_2d.npy",
                    help="Path to pcg_2d.npy (PCG edge probs from Phase 1). "
                         "Used to initialise online SimplePCG for CWM-driven pruning.")
    ap.add_argument("--cwm_online_steps",  type=int, default=4,
                    help="Gradient steps per skill for online CWM fine-tuning")
    ap.add_argument("--vlm",               action="store_true", default=False,
                    help="Enable Claude Opus 4.6 visual consultant. "
                         "Calls API at skill start and on failure to get visual scan "
                         "and directive action sequence. Requires ANTHROPIC_API_KEY.")
    ap.add_argument("--vlm_verbose",       action="store_true", default=False,
                    help="Print VLM analysis results (reason, directive, latency)")
    ap.add_argument("--use_clip",          action="store_true", default=False,
                    help="Load CLIP (ViT-B/32) and inject goal embeddings into PPO options. "
                         "At skill start, encodes VLM reason text (or default skill text) "
                         "→ 512-dim vector → option.set_goal_embedding(). Requires that "
                         "skills were trained with --use_clip.")
    ap.add_argument("--clip_device",       type=str, default="cpu",
                    help="Device for CLIP: 'cpu' or 'cuda' (default cpu)")
    args = ap.parse_args()

    if not GYM_OK:
        print("ERROR: gym not installed. Run:  conda activate dia-minecraft")
        return

    # ── Load 2D SIG ───────────────────────────────────────────────────────────
    if not os.path.exists(args.sig):
        print(f"ERROR: sig file not found: {args.sig}")
        print("  Run Phase 1 first:  conda run -n dia python scripts/train_minecraft2d_sip.py")
        return
    sig = load_sig_from_json(args.sig)

    # ── Online PCG (initialised from 2D probs if available) ───────────────────
    M   = len(VAR_NAMES)
    pcg = SimplePCG(PCGConfig(num_vars=M, init_edge_prob=0.05))
    if os.path.exists(args.pcg):
        probs = np.load(args.pcg).astype(float)
        np.fill_diagonal(probs, 0.0)
        pcg.state.edge_probs = np.clip(probs, 0.0, 1.0)
        print(f"Loaded PCG probs from {args.pcg}")
    else:
        print(f"  PCG file not found ({args.pcg}); using uniform init")

    # ── CWM (optional — load pre-trained weights if provided) ────────────────
    cwm_trainer = None
    if args.cwm and CWM_OK:
        cwm_path = args.cwm
        if os.path.exists(cwm_path):
            cfg_cwm = CWMConfig(
                num_vars=M, num_skills=M,
                online_steps=args.cwm_online_steps,
            )
            cwm_trainer = CWMTrainer(cfg_cwm, pcg=pcg)
            cwm_trainer.load(cwm_path)
            print(f"Loaded CWM from {cwm_path}")
        else:
            print(f"  CWM file not found ({cwm_path}); running without CWM")
    elif args.cwm and not CWM_OK:
        print("  WARNING: CWM requested but PyTorch not available — skipping CWM")

    # ── CLIP goal encoder (optional) ──────────────────────────────────────────
    clip_encoder = None
    if args.use_clip:
        if CLIP_OK:
            clip_encoder = CLIPGoalEncoder.get(device=args.clip_device)
            print(f"CLIP encoder loaded (ViT-B/32, device={args.clip_device})")
        else:
            print("WARNING: --use_clip requested but clip_skill module unavailable "
                  "(pip install openai-clip)")

    # ── VLM visual consultant (optional) ─────────────────────────────────────
    vlm_analyzer = None
    if args.vlm:
        if VLM_OK:
            vlm_analyzer = MineRLVLMAnalyzer(
                MineRLVLMConfig(verbose=args.vlm_verbose),
                min_interval_s=1.0,
            )
            print("VLM consultant enabled (Claude Opus 4.6)")
        else:
            print("WARNING: --vlm requested but vlm_minerl module unavailable")

    # PCG entropy scheduler for curiosity scaling
    _beta_sched = BetaScheduler(beta_max=1.0, beta_min=0.0,
                                h_ref=M * (M - 1) * 0.693)

    # ── MineRL environment ────────────────────────────────────────────────────
    import gym as gym_mod
    import minerl   # noqa: registers envs
    evgs    = make_minerl_evgs()
    raw_env = gym_mod.make(args.env_id)
    env     = MineRLObsWrapper(raw_env, evgs)

    # ── Load PPO options (or random placeholders) ─────────────────────────────
    os.makedirs(args.model_dir, exist_ok=True)
    print(f"\nLoading skill options from {args.model_dir}/ ...")
    options = load_options(args.model_dir, sig,
                           max_steps=args.max_steps_per_skill,
                           deterministic=args.deterministic,
                           use_scripted_fallback=not args.random_fallback)

    # ── Execution plan: topological order from 2D SIG ─────────────────────────
    topo_order  = sig.toposort()
    plan_names  = [sig.skills[sid].name for sid in topo_order if sid in sig.skills]
    print(f"\n2D-derived execution plan (topological order):")
    for i, name in enumerate(plan_names):
        print(f"  {i+1}. {name}")

    # ── Reset env ─────────────────────────────────────────────────────────────
    obs      = env.reset()
    achieved: list = []
    frames:  list = []
    global_step = 0

    print("\n" + "=" * 65)
    print("Phase 2: Executing causal plan in MineDojo 3D")
    print("=" * 65)

    for skill_id in topo_order:
        if skill_id not in sig.skills:
            continue
        skill    = sig.skills[skill_id]
        var_idx  = skill.subgoal.var_index
        var_name = VAR_NAMES[var_idx]
        option   = options.get(var_idx)

        # Check if already achieved (from previous option side-effects)
        x_now = evgs.extract(env.get_obs())
        if x_now[var_idx] > 0.5:
            print(f"  [{var_name}] already achieved — skipping")
            if skill_id not in achieved:
                achieved.append(skill_id)
            continue

        # ── VLM: visual scan before execution ─────────────────────────────────
        _vlm_goal_text: Optional[str] = None   # populated by VLM if available
        if vlm_analyzer is not None:
            raw_obs = env.get_raw_obs()
            pov_frame = None
            if isinstance(raw_obs, dict):
                pov_frame = raw_obs.get("pov")
                if pov_frame is None:
                    pov_frame = raw_obs.get("rgb")
            if pov_frame is not None:
                vlm_pre = vlm_analyzer.analyze(
                    np.asarray(pov_frame, dtype=np.uint8),
                    var_name=var_name,
                    context="start",
                )
                if not vlm_pre.error:
                    _vlm_goal_text = vlm_pre.reason  # encode via CLIP below
                    if vlm_pre.directive_actions:
                        # Feed directive into SkillScriptedOption if applicable
                        if isinstance(option, SkillScriptedOption):
                            option.set_directive(vlm_pre.directive_actions)
                    if not args.vlm_verbose:
                        print(f"  [{var_name}] VLM: {vlm_pre.reason[:80]}  "
                              f"directive={vlm_pre.directive_actions}")

        # ── CLIP: encode goal text → embedding → inject into option ───────────
        if clip_encoder is not None and hasattr(option, "set_goal_embedding"):
            # Prefer VLM-generated scene-specific text; fall back to fixed skill text
            goal_text = _vlm_goal_text or SKILL_TEXTS.get(
                var_name, f"gathering {var_name} in Minecraft"
            )
            goal_embed = clip_encoder.encode_vlm_instruction(goal_text)
            option.set_goal_embedding(goal_embed)

        # ── Run skill ─────────────────────────────────────────────────────────
        print(f"  [{var_name}] executing skill  (max {args.max_steps_per_skill} steps) ...")
        out     = option.run(env, evgs)
        success = out["success"]
        steps   = out["steps"]
        global_step += steps

        if success:
            achieved.append(skill_id)
            print(f"  [{var_name}] SUCCESS  ({steps} steps)")
        else:
            print(f"  [{var_name}] FAILED   ({steps} steps, moving on)")
            # ── VLM: failure diagnosis ─────────────────────────────────────
            if vlm_analyzer is not None:
                raw_obs = env.get_raw_obs()
                pov_frame = None
                if isinstance(raw_obs, dict):
                    pov_frame = raw_obs.get("pov")
                    if pov_frame is None:
                        pov_frame = raw_obs.get("rgb")
                if pov_frame is not None:
                    vlm_fail = vlm_analyzer.analyze(
                        np.asarray(pov_frame, dtype=np.uint8),
                        var_name=var_name,
                        context="failed",
                    )
                    if not vlm_fail.error:
                        print(f"  [{var_name}] VLM diagnosis: {vlm_fail.reason[:100]}")

        # ── CWM: online fine-tuning + PCG update ──────────────────────────────
        if cwm_trainer is not None and out.get("trajectory"):
            cwm_result = cwm_trainer.process_skill_trajectory(
                trajectory=out["trajectory"],
                skill_idx=var_idx,
                evgs=evgs,
                pcg=pcg,
                verbose=False,
            )
            H_pcg = pcg.entropy()
            beta  = _beta_sched(H_pcg)
            intrinsic = cwm_result["mean_error"] * beta
            print(f"  [{var_name}] CWM: err={cwm_result['mean_error']:.3f}  "
                  f"loss={cwm_result['online_loss']:.4f}  "
                  f"intrinsic_r={intrinsic:.3f}  "
                  f"n_trans={cwm_result['n_transitions']}  "
                  f"PCG_H={H_pcg:.3f}")

        # ── Collect frames from trajectory ────────────────────────────────────
        x_curr = evgs.extract(env.get_raw_obs() or {})
        for (traj_obs, _act, _nobs, _succ) in out["trajectory"]:
            # traj_obs is already in wrapped format {'rgb': ..., 'inventory': ...}
            rgb = None
            if isinstance(traj_obs, dict):
                raw_rgb = traj_obs.get("rgb")
                if raw_rgb is not None:
                    rgb = np.asarray(raw_rgb, dtype=np.uint8)
            if rgb is None:
                rgb = np.zeros((_IMAGE_H, _IMAGE_W, 3), dtype=np.uint8)

            # Build plan entries for panel
            plan_entries = []
            for sid in topo_order:
                if sid not in sig.skills:
                    continue
                sk     = sig.skills[sid]
                is_done   = sid in achieved
                is_active = (sid == skill_id)
                plan_entries.append((sk.name, is_done, is_active))

            frame_meta = {
                "step":           global_step,
                "var_name":       var_name,
                "success":        success,
                "plan_entries":   plan_entries,
                "var_vals":       dict(zip(VAR_NAMES, x_curr.tolist())),
                "achieved_names": [VAR_NAMES[sig.skills[s].subgoal.var_index]
                                   for s in achieved if s in sig.skills],
            }
            frame = make_transfer_frame(rgb, frame_meta, VAR_NAMES)
            frames.append(frame)

        raw_obs = env.get_raw_obs()
        print(f"  inventory: {inventory_summary(raw_obs) if raw_obs else '?'}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"Phase 2 complete.  Total steps: {global_step}")
    print(f"Achieved ({len(achieved)}/{len(sig.skills)}):")
    for sid in achieved:
        if sid in sig.skills:
            print(f"  ✓ {sig.skills[sid].name}")
    diamond_id = _VAR_IDX["diamond"]
    if diamond_id in achieved:
        print("\n  *** DIAMOND MINED — GOAL ACHIEVED ***")

    # ── CWM: print final PCG state after online adaptation ───────────────────
    if cwm_trainer is not None:
        print("\nFinal PCG after CWM online adaptation (edges >= 0.5):")
        for i in range(M):
            for j in range(M):
                if i != j and pcg.probs[i, j] >= 0.5:
                    print(f"  {VAR_NAMES[i]} → {VAR_NAMES[j]}: {pcg.probs[i,j]:.3f}")
        print(f"PCG entropy: {pcg.entropy():.3f}")

    # ── VLM stats ─────────────────────────────────────────────────────────────
    if vlm_analyzer is not None:
        s = vlm_analyzer.stats()
        print(f"\nVLM stats: calls={s['total_calls']:.0f}  "
              f"avg_latency={s['avg_latency_ms']:.0f}ms  "
              f"error_rate={s['error_rate']:.2f}")

    # ── Write video ───────────────────────────────────────────────────────────
    try:
        env.env.close()
    except Exception:
        pass

    if not frames:
        print("No frames to write.")
        return
    if not FFMPEG_OK:
        print("imageio_ffmpeg not available — skipping video.")
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
    print(f"Video: {args.out}")


if __name__ == "__main__":
    main()
