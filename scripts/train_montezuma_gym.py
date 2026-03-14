#!/usr/bin/env python3
"""
train_montezuma_gym.py
======================
DIA with Granger-causal structure learning on Montezuma's Revenge.

Outputs (all in --outdir / monterun1/):
  videos/episode_stepNNNN.mp4   – game footage recorded at each checkpoint
  pcg/pcg_stepNNNN.png          – labelled edge-probability heatmap
  sig/sig_stepNNNN.png          – labelled prerequisite-graph visualisation
  tensorboard/                  – scalar logs (TBLogger)
  summary.txt                   – end-of-run statistics

Run:
    conda activate dia
    cd Discover-Intervene-Adapt-Interleaved-Causal-RL
    pip install -e .
    python scripts/train_montezuma_gym.py --steps 400 --outdir monterun1
"""

from __future__ import annotations

import argparse, os, textwrap
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Environment setup — gymnasium first (gym 0.26 lacks register_envs)
# ---------------------------------------------------------------------------

def _make_raw_env(env_id: Optional[str] = None) -> Any:
    preferred = env_id or "ALE/MontezumaRevenge-v5"
    errors: List[str] = []

    try:
        import ale_py
        import gymnasium as gym
        gym.register_envs(ale_py)
        env = gym.make(preferred, render_mode="rgb_array")
        print(f"  Created env via gymnasium: {preferred}")
        return env
    except Exception as e:
        errors.append(f"gymnasium: {e}")

    try:
        import gym as gym_legacy  # type: ignore
        try:
            import ale_py  # noqa: F401 registers ALE with gym
        except ImportError:
            pass
        env = gym_legacy.make(preferred, render_mode="rgb_array")
        print(f"  Created env via gym (legacy): {preferred}")
        return env
    except Exception as e:
        errors.append(f"gym-legacy: {e}")

    raise RuntimeError(
        "Could not create Montezuma env:\n" + "\n".join(errors) +
        "\n\nInstall: pip install gymnasium ale-py && python -m AutoROM --accept-license"
    )


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _setup_outdir(outdir: str) -> Tuple[str, str, str, str]:
    """Create subdirectories; return (videos_dir, pcg_dir, sig_dir, tb_dir)."""
    vdir = os.path.join(outdir, "videos")
    pdir = os.path.join(outdir, "pcg")
    sdir = os.path.join(outdir, "sig")
    tdir = os.path.join(outdir, "tensorboard")
    for d in (vdir, pdir, sdir, tdir):
        os.makedirs(d, exist_ok=True)
    return vdir, pdir, sdir, tdir


def save_pcg_heatmap(
    pcg, var_names: List[str], path: str,
    step: int = 0, vlm_context: str = "",
) -> None:
    """Save a labelled heatmap of PCG edge probabilities."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        probs = np.array(pcg.probs)
        d = len(var_names)
        # Mask diagonal
        diag_mask = np.eye(d, dtype=bool)

        fig, ax = plt.subplots(figsize=(max(7, d), max(6, d - 1)))
        cmap = plt.cm.RdYlGn
        im = ax.imshow(probs, vmin=0.0, vmax=1.0, cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, label="P(row → col)")

        # Cell annotations
        for i in range(d):
            for j in range(d):
                if i == j:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               color="lightgrey", zorder=2))
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=8, color="grey", zorder=3)
                else:
                    p = probs[i, j]
                    txt_color = "white" if (p > 0.75 or p < 0.25) else "black"
                    ax.text(j, i, f"{p:.2f}", ha="center", va="center",
                            fontsize=7, color=txt_color, zorder=3,
                            fontweight="bold" if p >= 0.65 else "normal")
                    # Highlight strong edges
                    if p >= 0.65:
                        ax.add_patch(plt.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1,
                            fill=False, edgecolor="navy", linewidth=2, zorder=4))

        ax.set_xticks(range(d))
        ax.set_yticks(range(d))
        ax.set_xticklabels(var_names, rotation=35, ha="right", fontsize=9)
        ax.set_yticklabels(var_names, fontsize=9)
        ax.set_xlabel("Effect variable (j)", fontsize=10)
        ax.set_ylabel("Cause variable (i)", fontsize=10)

        H = float(pcg.entropy()) if hasattr(pcg, "entropy") else float("nan")
        title = (
            f"GrangerPCG edge probabilities  [step {step}]\n"
            f"P(row causes col)   entropy={H:.2f}"
        )
        if vlm_context:
            title += f"\nVLM: {vlm_context}"
        ax.set_title(title, fontsize=11)

        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
    except Exception as e:
        print(f"  [warn] PCG heatmap save failed: {e}")


def save_sig_graph(
    sig, var_names: List[str], path: str,
    step: int = 0, vlm_context: str = "",
) -> None:
    """Save a labelled directed-graph visualisation of the learned SIG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx

        G = nx.DiGraph()
        for sid, skill in sig.skills.items():
            label = skill.name or (var_names[skill.subgoal.var_index]
                                   if skill.subgoal.var_index < len(var_names)
                                   else f"s{sid}")
            G.add_node(sid, label=label,
                       sr=float(getattr(skill, "success_rate", 0.0)))
        for u, vs in sig.edges.items():
            for v in vs:
                G.add_edge(u, v)

        if len(G.nodes) == 0:
            return

        fig, ax = plt.subplots(figsize=(max(8, len(G.nodes) * 1.4), 5))

        try:
            pos = nx.spring_layout(G, seed=42, k=1.8)
        except Exception:
            pos = nx.circular_layout(G)

        labels = nx.get_node_attributes(G, "label")
        sr_vals = nx.get_node_attributes(G, "sr")
        node_colors = [plt.cm.YlOrRd(min(1.0, sr_vals.get(n, 0.0))) for n in G.nodes]

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=1600, alpha=0.92)
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                                font_size=8, font_weight="bold")
        nx.draw_networkx_edges(G, pos, ax=ax, arrowsize=18,
                               edge_color="steelblue", width=2.0,
                               connectionstyle="arc3,rad=0.1")

        sig_title = (
            f"Skill-Intervention Graph (SIG)  [step {step}]\n"
            f"Nodes = skills  ·  Edges = learned prerequisites  "
            f"·  Colour = success rate"
        )
        if vlm_context:
            sig_title += f"\nVLM: {vlm_context}"
        ax.set_title(sig_title, fontsize=11)
        ax.axis("off")

        sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd,
                                   norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="success rate", shrink=0.6)

        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
    except Exception as e:
        print(f"  [warn] SIG graph save failed: {e}")


def save_video(frames: List, path: str, fps: int = 15) -> bool:
    """Save a list of RGB numpy frames to an mp4 file via imageio."""
    if not frames:
        return False
    try:
        import imageio
        valid = [f for f in frames if f is not None and hasattr(f, "shape")]
        if not valid:
            return False
        writer = imageio.get_writer(path, fps=fps, macro_block_size=1)
        for f in valid:
            writer.append_data(f)
        writer.close()
        return True
    except Exception as e:
        # Fallback: try saving as GIF
        try:
            import imageio
            gif_path = path.replace(".mp4", ".gif")
            imageio.mimwrite(gif_path, valid, fps=fps)
            print(f"  [info] Saved GIF fallback: {gif_path}")
            return True
        except Exception as e2:
            print(f"  [warn] Video save failed: {e}  (GIF: {e2})")
            return False


# ---------------------------------------------------------------------------
# Directed option — biased random actions toward each zone
# ---------------------------------------------------------------------------

_NOOP, _FIRE = 0, 1
_UP, _RIGHT, _LEFT, _DOWN = 2, 3, 4, 5
_UPRIGHT, _UPLEFT, _DOWNRIGHT, _DOWNLEFT = 6, 7, 8, 9

_PREFERRED_ACTIONS: Dict[str, List[int]] = {
    "at_key_zone":    [_RIGHT, _UPRIGHT, _UP, _RIGHT, _UPLEFT],
    "at_door_zone":   [_LEFT,  _DOWNLEFT, _DOWN, _LEFT, _DOWNLEFT],
    "on_upper_level": [_UP,    _UPRIGHT, _UPLEFT, _UP],
    "near_rope":      [_RIGHT, _UP, _NOOP, _RIGHT, _NOOP],
    "has_key":        [_RIGHT, _UPRIGHT, _UP, _FIRE, _RIGHT],
    "door_open":      [_LEFT,  _DOWNLEFT, _FIRE, _LEFT, _DOWN],
    "skull_near":     [_NOOP,  _RIGHT, _LEFT, _UP, _DOWN],
    "score_gained":   [_UP,    _RIGHT, _UPRIGHT, _LEFT, _NOOP],
}

_EXPLORE_PROB = 0.25


class DirectedOption:
    """
    Option policy biased toward actions appropriate for a given target variable.
    75 % of actions come from a preferred set; 25 % are random exploration.
    """

    def __init__(self, subgoal, action_space, var_name: str,
                 max_steps: int = 160, explore_prob: float = _EXPLORE_PROB):
        self.subgoal      = subgoal
        self.action_space = action_space
        self.preferred    = _PREFERRED_ACTIONS.get(var_name, [_NOOP])
        self.max_steps    = max_steps
        self.explore_prob = explore_prob

    def _act(self, _obs) -> int:
        if np.random.random() < self.explore_prob:
            return int(self.action_space.sample())
        return int(np.random.choice(self.preferred))

    def run(self, env, evgs) -> Dict[str, Any]:
        obs = env.get_obs() if hasattr(env, "get_obs") else env.reset()
        x_t = evgs.extract(obs)
        success = False
        step_pairs: List[Tuple[np.ndarray, np.ndarray]] = []

        for _s in range(self.max_steps):
            action = self._act(obs)
            result = env.step(action)
            if isinstance(result, tuple) and len(result) == 4:
                next_obs, _rew, done, info = result
            else:
                next_obs, _rew, done, info = result[0], result[1], result[2], result[3]

            x_tp1 = evgs.extract(next_obs)
            if np.any(x_tp1 != x_t):
                step_pairs.append((x_t.copy(), x_tp1.copy()))
            if evgs.predicate_holds(x_t, x_tp1, self.subgoal):
                success = True
                break
            x_t   = x_tp1
            obs   = next_obs
            if done and not info.get("soft_continue", False):
                break

        return {"success": success, "steps": _s + 1,
                "final_obs": obs, "step_pairs": step_pairs}


# ---------------------------------------------------------------------------
# GrangerInterventionSelector — exploration-friendly skill selection
# ---------------------------------------------------------------------------

from dia.planner import InterventionSelector, PlannerConfig   # noqa: E402
from dia.rollout import DIARunner, RunnerConfig               # noqa: E402


class GrangerInterventionSelector(InterventionSelector):
    """
    Overrides InterventionSelector to fix two problems in the "novel" phase:

    1. Tie-breaking:  all novelty scores start equal (all edges at 0.5),
       so base argmax always returns skill 0.  We add Dirichlet-softmax
       sampling that spreads selection uniformly when scores are tied.

    2. GoalPlanner short-circuit:  when task_goal is provided, the base
       select() immediately returns the task-goal skill (which is always
       "ready" because there are no prerequisites yet), bypassing
       exploration entirely.  We suppress task_goal during "novel" phase.
    """

    def __init__(self, *args, explore_eps: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.explore_eps = explore_eps   # fraction of steps that are fully random

    def _select_non_goal(self, achieved, task_goal):
        phase = self.phase()
        candidates = self.sig.ready_skills(achieved)
        if not candidates:
            candidates = list(self.sig.skills.keys())
        skills = [self.sig.skills[c] for c in candidates]

        if phase == "novel":
            # ε-greedy exploration so every skill gets turns in the buffer
            if np.random.random() < self.explore_eps:
                return int(np.random.choice([s.skill_id for s in skills]))
            scores = [self.score_novelty(s) + np.random.uniform(0, 0.05)
                      for s in skills]
        elif phase == "confirm":
            scores = [self.score_confirm(s) for s in skills]
        else:
            scores = [self.score_goal(s, task_goal) for s in skills]

        best_idx = int(np.argmax(scores))
        return skills[best_idx].skill_id

    def select(self, achieved, task_goal=None):
        # During novel phase suppress the goal so GoalPlanner cannot
        # short-circuit and force-select skill-0 every single step.
        if self.phase() == "novel":
            return self._select_non_goal(achieved, task_goal=None)
        return super().select(achieved, task_goal)


class GrangerDIARunner(DIARunner):
    """
    Subclass of DIARunner that overrides PCG fitting to use the intervention
    mask from the PCGBuffer — the key fix enabling proper causal discovery.
    """

    def _maybe_fit_pcg(self):
        self.steps += 1
        if self.steps % max(1, self.cfg.fit_every) != 0:
            return False, 0.0, 0.0
        if len(self.buffer) < self.cfg.min_buffer:
            return False, 0.0, 0.0

        packed, mask = self.buffer.recent(self.cfg.batch_recent)
        if packed.shape[0] == 0:
            return False, 0.0, 0.0

        d = len(self.evgs.var_names)
        X_t   = packed[:, :d]
        X_tp1 = packed[:, d:]

        old_probs   = np.array(self.pcg.probs).copy()
        old_entropy = float(self.pcg.entropy()) if hasattr(self.pcg, "entropy") else float("nan")

        if hasattr(self.pcg, "fit_from_transitions"):
            new_probs = self.pcg.fit_from_transitions(X_t, X_tp1, mask)
            self.pcg.apply_update(new_probs)
        else:
            new_probs = self._transition_fit_probs(X_t, X_tp1)
            if hasattr(self.pcg, "apply_update"):
                self.pcg.apply_update(new_probs)

        new_probs   = np.array(self.pcg.probs)
        new_entropy = float(self.pcg.entropy()) if hasattr(self.pcg, "entropy") else float("nan")
        ig_update    = self._bernoulli_kl(new_probs, old_probs)
        entropy_drop = (old_entropy - new_entropy) if not (
            np.isnan(old_entropy) or np.isnan(new_entropy)) else 0.0

        if self.logger:
            self.logger.add_scalar(f"{self.cfg.log_prefix}/pcg_entropy",  new_entropy)
            self.logger.add_scalar(f"{self.cfg.log_prefix}/ig_update",     ig_update)
            self.logger.add_scalar(f"{self.cfg.log_prefix}/entropy_drop",  entropy_drop)

        if self.cfg.auto_expand_sig and hasattr(self.pcg, "probs"):
            from dia.sig_auto import expand_sig_from_pcg, AutoSIGConfig
            stats = expand_sig_from_pcg(
                self.sig, self.evgs, self.pcg.probs,
                AutoSIGConfig(
                    add_threshold=self.cfg.add_threshold,
                    remove_threshold=self.cfg.remove_threshold,
                    create_missing_skills=self.cfg.create_missing_skills,
                    verbose=False,
                ),
            )
            if self.logger:
                self.logger.add_scalar(f"{self.cfg.log_prefix}/sig_added",   float(stats["added"]))
                self.logger.add_scalar(f"{self.cfg.log_prefix}/sig_removed", float(stats["removed"]))

        return True, ig_update, entropy_drop


# ---------------------------------------------------------------------------
# Console diagnostic helpers
# ---------------------------------------------------------------------------

def _print_top_edges(pcg, var_names: List[str], k: int = 8) -> None:
    edges = pcg.top_edges(k=k, var_names=var_names) if hasattr(pcg, "top_edges") else []
    print(f"  Top-{k} edges:")
    for rank, (prob, i, j, name) in enumerate(edges, 1):
        bar = "█" * int(prob * 20)
        marker = " ◄ STRONG" if prob >= 0.65 else ""
        print(f"    {rank:2d}. {bar:<20} {prob:.3f}  {name}{marker}")


def _print_sig(sig, var_names: List[str]) -> None:
    edges = []
    for u, vs in sig.edges.items():
        for v in vs:
            un = sig.skills[u].name if u in sig.skills else str(u)
            vn = sig.skills[v].name if v in sig.skills else str(v)
            edges.append(f"{un} → {vn}")
    if edges:
        print("  SIG prerequisites:", ", ".join(edges))
    else:
        print("  SIG: no prerequisites discovered yet")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__ or ""),
    )
    parser.add_argument("--env_id",           default=None,             type=str)
    parser.add_argument("--steps",            default=400,              type=int,
                        help="Macro-steps (option executions)")
    parser.add_argument("--fit_every",        default=20,               type=int)
    parser.add_argument("--min_buffer",       default=80,               type=int)
    parser.add_argument("--batch_recent",     default=1024,             type=int)
    parser.add_argument("--option_steps",     default=160,              type=int,
                        help="Max primitive steps per option")
    parser.add_argument("--goal",             default="has_key",        type=str)
    parser.add_argument("--outdir",           default="monterun1",      type=str)
    parser.add_argument("--checkpoint_every", default=50,               type=int,
                        help="Save PCG/SIG/video every N macro-steps")
    parser.add_argument("--video_frames",     default=600,              type=int,
                        help="Max frames per video clip")
    parser.add_argument("--seed",             default=0,                type=int)
    # ── VLM enrichment ──────────────────────────────────────────────────────
    parser.add_argument("--use_vlm",          action="store_true",
                        help="Enrich EVGS with Claude claude-opus-4-6 VLM (needs CLAUDE_API_KEY)")
    parser.add_argument("--vlm_call_every",   default=200,              type=int,
                        help="Call VLM every N primitive steps (~1 per option at default 200)")
    parser.add_argument("--vlm_alpha",        default=0.6,              type=float,
                        help="RAM weight in fusion: enriched = alpha*RAM + (1-alpha)*VLM")
    parser.add_argument("--vlm_verbose",      action="store_true",
                        help="Print one-line VLM summary after each API call")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # ── Output structure ─────────────────────────────────────────────────────
    outdir = args.outdir
    videos_dir, pcg_dir, sig_dir, tb_dir = _setup_outdir(outdir)
    print(f"Output directory: {os.path.abspath(outdir)}/")

    # ── Environment ──────────────────────────────────────────────────────────
    print("Creating Montezuma's Revenge environment…")
    raw_env = _make_raw_env(args.env_id)

    from dia.evgs_montezuma import MontezumaRichWrapper, make_montezuma_evgs_rich
    env = MontezumaRichWrapper(raw_env)
    evgs = make_montezuma_evgs_rich()
    var_names = evgs.names()
    M = len(var_names)
    print(f"  EVGS variables ({M}): {var_names}")

    # ── Optional VLM enrichment ───────────────────────────────────────────────
    vlm_analyzer  = None
    if args.use_vlm:
        from dia.vlm_scene import VLMSceneAnalyzer, VLMSceneConfig
        from dia.evgs_vlm  import VLMEnrichedEVGS, VLMFusionConfig
        vlm_analyzer = VLMSceneAnalyzer(VLMSceneConfig(
            call_every=args.vlm_call_every,
            verbose=args.vlm_verbose,
        ))
        evgs = VLMEnrichedEVGS(
            evgs, vlm_analyzer,
            VLMFusionConfig(alpha=args.vlm_alpha),
        )
        print(
            f"  VLM enrichment ON  (claude-opus-4-6, "
            f"call_every={args.vlm_call_every}, alpha={args.vlm_alpha})"
        )
    else:
        print("  VLM enrichment OFF  (pass --use_vlm to enable)")

    # ── GrangerPCG ───────────────────────────────────────────────────────────
    from dia.pcg_granger import GrangerPCG, GrangerPCGConfig
    pcg = GrangerPCG(GrangerPCGConfig(
        num_vars=M,
        init_edge_prob=0.5,    # max uncertainty → entropy ≈ 38.8 >> entropy_high=25
        alpha=2.0,
        change_threshold=0.05,
        lambda_int=0.6,
        lambda_obs=0.4,
        sigmoid_k=6.0,
        min_obs=4,
        momentum=0.2,
    ))
    print(f"  Initial PCG entropy: {pcg.entropy():.2f}  "
          f"(entropy_high=25 → starts in 'novel' phase)")

    # ── SIG: one Predicate.UP skill per variable, no initial prerequisites ───
    from dia.sig import SIGraph, Skill
    from dia.types import Subgoal, Predicate
    sig = SIGraph()
    for vi, vname in enumerate(var_names):
        sig.add_skill(Skill(skill_id=vi,
                            subgoal=Subgoal(var_index=vi, predicate=Predicate.UP),
                            name=vname))
    print(f"  SIG: {len(sig.skills)} skills initialised, 0 prerequisites")

    # ── Planner ──────────────────────────────────────────────────────────────
    selector = GrangerInterventionSelector(
        pcg, sig,
        PlannerConfig(entropy_high=25.0, entropy_low=5.0),
        explore_eps=0.6,    # 60 % random during novel phase for mask diversity
    )

    # ── Logger ───────────────────────────────────────────────────────────────
    from dia.logging_utils import TBLogger
    logger = TBLogger(tb_dir)

    # ── Runner ───────────────────────────────────────────────────────────────
    rcfg = RunnerConfig(
        buffer_size=20_000,
        min_buffer=args.min_buffer,
        batch_recent=args.batch_recent,
        fit_every=args.fit_every,
        pcg_epochs=200,
        log_prefix="granger",
        option_max_steps=args.option_steps,
        terminate_on_success=True,
        auto_expand_sig=True,
        add_threshold=0.72,
        remove_threshold=0.52,
        create_missing_skills=False,
    )

    def option_factory(skill: Skill) -> DirectedOption:
        vname = var_names[skill.subgoal.var_index] if skill.subgoal.var_index < M else "score_gained"
        return DirectedOption(skill.subgoal, env.action_space, vname,
                              max_steps=args.option_steps)

    runner = GrangerDIARunner(
        env=env, evgs=evgs, pcg=pcg, sig=sig, selector=selector,
        cfg=rcfg, logger=logger, option_factory=option_factory,
    )

    # ── Task goal ─────────────────────────────────────────────────────────────
    name_to_idx = {n: i for i, n in enumerate(var_names)}
    goal_var = args.goal if args.goal in name_to_idx else "has_key"
    task_goal = Subgoal(var_index=name_to_idx[goal_var], predicate=Predicate.UP)
    print(f"  Task goal: '{goal_var}' ↑  (var index {name_to_idx[goal_var]})\n")

    # ── Warmup: reset env and run 60 NOOP frames to skip startup animation ────
    print("Warming up environment (60 NOOP frames)…")
    env.reset()
    for _ in range(60):
        env.step(0)   # NOOP
    print("  Warmup complete – player now controllable\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    achieved: List[int] = []
    successes_by_var: Dict[str, int] = {n: 0 for n in var_names}
    ckpt_every = max(1, args.checkpoint_every)

    # Start recording for the first video clip
    env.start_recording()

    for t in range(args.steps):
        rec = runner.step(achieved, task_goal=task_goal)

        if rec["success"] and rec["skill_id"] not in achieved:
            achieved.append(rec["skill_id"])
        if rec["success"]:
            successes_by_var[rec["skill_name"]] = \
                successes_by_var.get(rec["skill_name"], 0) + 1

        # ── Console progress every 10 steps ──────────────────────────────────
        if (t + 1) % 10 == 0:
            H     = rec["pcg_entropy"]
            phase = rec["phase"]
            sym   = {"novel": "?", "confirm": "~", "goal": "★"}.get(phase, phase)
            print(
                f"[{t+1:04d}/{args.steps}] {sym} {phase:<7}  "
                f"skill={rec['skill_name']:<16}  succ={int(rec['success'])}  "
                f"H={H:6.2f}  IG={rec['ig_update']:.4f}  "
                f"buf={rec['buffer_size']:4d}  "
                f"achieved={[var_names[i] for i in achieved]}"
            )

        # ── Checkpoint: save PCG heatmap, SIG graph, video ───────────────────
        if (t + 1) % ckpt_every == 0:
            step_tag = f"step{t+1:04d}"
            print(f"\n  ── Checkpoint {step_tag} ──────────────────────────────")

            # Grab latest VLM scene description for plot annotations
            vlm_ctx = ""
            if vlm_analyzer is not None and hasattr(evgs, "last_vlm"):
                lr = evgs.last_vlm
                if lr is not None and not lr.error:
                    vlm_ctx = lr.scene_description
                    if lr.player_action and lr.player_action != "unknown":
                        vlm_ctx += f"  [{lr.player_action}]"

            # PCG heatmap
            pcg_path = os.path.join(pcg_dir, f"pcg_{step_tag}.png")
            save_pcg_heatmap(pcg, var_names, pcg_path, step=t + 1,
                             vlm_context=vlm_ctx)
            print(f"  PCG heatmap → {pcg_path}")
            _print_top_edges(pcg, var_names, k=6)

            # SIG graph
            sig_path = os.path.join(sig_dir, f"sig_{step_tag}.png")
            save_sig_graph(sig, var_names, sig_path, step=t + 1,
                           vlm_context=vlm_ctx)
            print(f"  SIG graph   → {sig_path}")
            _print_sig(sig, var_names)

            # Video clip from accumulated frames
            frames = env.get_frames()
            if frames:
                # Trim to max_frames
                if len(frames) > args.video_frames:
                    frames = frames[:args.video_frames]
                vid_path = os.path.join(videos_dir, f"episode_{step_tag}.mp4")
                ok = save_video(frames, vid_path, fps=15)
                if ok:
                    print(f"  Video       → {vid_path}  ({len(frames)} frames)")
                else:
                    print(f"  Video save skipped (no frames or encoder issue)")
            else:
                print("  Video: no frames accumulated yet")

            # Restart recording for next clip
            env.start_recording()
            print()

    # ── Final checkpoint ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    for label, path_fn, save_fn, extra_kw in [
        ("PCG heatmap", lambda: os.path.join(pcg_dir, "pcg_final.png"),
         save_pcg_heatmap, {"step": args.steps, "vlm_context": ""}),
        ("SIG graph",   lambda: os.path.join(sig_dir,  "sig_final.png"),
         save_sig_graph,  {"step": args.steps, "vlm_context": ""}),
    ]:
        p = path_fn()
        save_fn(pcg if "PCG" in label else sig,
                var_names, p, **extra_kw)
        print(f"  {label} → {p}")

    # Final video
    frames = env.get_frames()
    if frames:
        if len(frames) > args.video_frames:
            frames = frames[:args.video_frames]
        vid_path = os.path.join(videos_dir, "episode_final.mp4")
        save_video(frames, vid_path, fps=15)
        print(f"  Final video  → {vid_path}")

    # Summary text
    summary_path = os.path.join(outdir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("DIA Granger-Causal Montezuma Run Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Steps:     {args.steps}\n")
        f.write(f"Goal:      {goal_var}\n")
        f.write(f"Achieved:  {[var_names[i] for i in achieved]}\n\n")
        f.write(f"Final PCG entropy: {pcg.entropy():.3f}\n\n")
        f.write("Skill success counts:\n")
        for vname in var_names:
            f.write(f"  {vname:<18} {successes_by_var.get(vname, 0)}\n")
        f.write("\nTop edges (P > 0.55):\n")
        for prob, i, j, name in pcg.top_edges(k=15, var_names=var_names):
            if prob > 0.55:
                f.write(f"  {name}: {prob:.3f}\n")
        f.write("\nSIG prerequisites:\n")
        for u, vs in sig.edges.items():
            for v in vs:
                un = sig.skills[u].name if u in sig.skills else str(u)
                vn = sig.skills[v].name if v in sig.skills else str(v)
                f.write(f"  {un} → {vn}\n")
    print(f"  Summary     → {summary_path}")

    # Final console summary
    print("\nSkill success counts:")
    for vname in var_names:
        cnt = successes_by_var.get(vname, 0)
        bar = "█" * min(cnt, 40)
        print(f"  {vname:<18} {bar:<40} ({cnt})")

    print("\nTop learned edges:")
    _print_top_edges(pcg, var_names, k=10)
    print("\nFinal SIG:")
    _print_sig(sig, var_names)

    achieved_names = [var_names[i] for i in achieved if i < len(var_names)]
    print(f"\nVariables achieved at least once: {achieved_names}")

    # ── VLM stats ─────────────────────────────────────────────────────────────
    if vlm_analyzer is not None:
        vstats = vlm_analyzer.stats()
        print(f"\nVLM API usage:  calls={vstats['total_vlm_calls']:.0f}  "
              f"avg_latency={vstats['avg_latency_ms']:.0f} ms  "
              f"error_rate={vstats['error_rate']:.2%}")
        with open(summary_path, "a") as f:
            f.write(f"\nVLM API usage:\n")
            for k, v in vstats.items():
                f.write(f"  {k}: {v}\n")

    logger.flush()
    logger.close()
    print(f"\nAll outputs in: {os.path.abspath(outdir)}/")
    print(f"TensorBoard:    tensorboard --logdir {tb_dir}")


if __name__ == "__main__":
    main()
