# src/dia/planner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from .pcg import SimplePCG
from .sig import SIGraph, Skill
from .types import Subgoal
from .intrinsic import BetaScheduler
from .plan_search import GoalPlanner  # <- goal-aware planner over SIG


@dataclass
class PlannerConfig:
    novel_fraction: float = 0.6       # (kept for reference; phase is derived from entropy)
    confirm_fraction: float = 0.2
    goal_fraction: float = 0.2
    entropy_high: float = 25.0        # thresholds to switch phases
    entropy_low: float = 5.0


class InterventionSelector:
    """
    Chooses next skill to execute.
    - If a task_goal is provided, use GoalPlanner to get an ordered plan and pick the first 'ready' skill.
    - Else use a novelty -> confirmatory -> goal-directed heuristic based on PCG entropy and skill stats.
    """
    def __init__(self, pcg: SimplePCG, sig: SIGraph, cfg: PlannerConfig,
                 use_ig_bonus: bool = True, use_sig: bool = True):
        self.pcg = pcg
        self.sig = sig
        self.cfg = cfg
        self.use_ig_bonus = use_ig_bonus  # False => skip novelty/entropy phase
        self.use_sig = use_sig            # False => ignore SIG prerequisites
        self.beta_sched = BetaScheduler(beta_max=1.0, beta_min=0.0, h_ref=cfg.entropy_high)
        self.goal_planner = GoalPlanner(sig)

    def phase(self) -> str:
        H = self.pcg.entropy()
        if not self.use_ig_bonus:
            # No IG-based novelty seeking; collapse to confirm or goal only.
            return "goal" if H <= self.cfg.entropy_low else "confirm"
        if H >= self.cfg.entropy_high:
            return "novel"
        if H <= self.cfg.entropy_low:
            return "goal"
        return "confirm"

    # ----------------- scoring (non-goal mode) -----------------

    def score_novelty(self, skill: Skill) -> float:
        """Score by entropy of incoming edges to the target variable."""
        p = self.pcg.probs.copy()
        j = skill.subgoal.var_index
        incoming = p[:, j]
        incoming[j] = 0.0
        eps = 1e-8
        h = -incoming * np.log(np.clip(incoming, eps, 1 - eps)) - (1 - incoming) * np.log(np.clip(1 - incoming, eps, 1 - eps))
        return float(np.sum(h))

    def score_confirm(self, skill: Skill) -> float:
        # prefer mid success rates (uncertain)
        s = skill.success_rate
        return float(1.0 - abs(0.5 - s))

    def score_goal(self, skill: Skill, goal_subgoal: Optional[Subgoal]) -> float:
        if goal_subgoal and (skill.subgoal == goal_subgoal):
            return 1.0 + skill.success_rate
        return skill.success_rate

    # ----------------- selection -----------------

    def _select_non_goal(self, achieved: List[int], task_goal: Optional[Subgoal]) -> int:
        phase = self.phase()
        if self.use_sig:
            candidates = self.sig.ready_skills(achieved)
            if not candidates:
                candidates = list(self.sig.skills.keys())
        else:
            candidates = list(self.sig.skills.keys())
        skills = [self.sig.skills[c] for c in candidates]

        if phase == "novel":
            scores = [self.score_novelty(s) for s in skills]
        elif phase == "confirm":
            scores = [self.score_confirm(s) for s in skills]
        else:
            scores = [self.score_goal(s, task_goal) for s in skills]

        scores_arr = np.array(scores, dtype=float)
        exp_s = np.exp(scores_arr - scores_arr.max())
        probs_s = exp_s / exp_s.sum()
        best_idx = int(np.random.choice(len(skills), p=probs_s))
        return skills[best_idx].skill_id

    def select(self, achieved: List[int], task_goal: Optional[Subgoal] = None) -> int:
        """
        Select the next skill to execute.

        Goal plan is only activated in "goal" phase (low entropy = confident causal model).
        In "novel" / "confirm" phases, falls through to _select_non_goal() so that DIA
        explores broadly to BUILD the causal model first.

        Rationale: with an empty SIG at initialisation, plan_for_subgoal returns [target]
        (no prerequisites), so the goal plan would always pick the target skill directly.
        That collapses exploration to a single skill that never succeeds, giving
        success_rate ≈ 0 and SHD stuck at the initial value.  By gating on phase="goal"
        we ensure goal-directed execution only kicks in AFTER the PCG has been learned.

        When use_sig=False, SIG prerequisite filtering is bypassed entirely.
        """
        if task_goal is not None and self.use_sig and self.phase() == "goal":
            plan = self.goal_planner.plan_for_subgoal(task_goal, achieved)
            if plan is not None and plan.skills:
                # pick first ready skill from the plan
                ready = set(self.sig.ready_skills(achieved))
                for sid in plan.skills:
                    if sid in ready:
                        return sid
        # fallback (also handles novel/confirm phases)
        return self._select_non_goal(achieved, task_goal)
