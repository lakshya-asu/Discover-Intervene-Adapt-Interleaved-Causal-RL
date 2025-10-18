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
    def __init__(self, pcg: SimplePCG, sig: SIGraph, cfg: PlannerConfig):
        self.pcg = pcg
        self.sig = sig
        self.cfg = cfg
        self.beta_sched = BetaScheduler(beta_max=1.0, beta_min=0.0, h_ref=cfg.entropy_high)
        self.goal_planner = GoalPlanner(sig)

    def phase(self) -> str:
        H = self.pcg.entropy()
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
        candidates = self.sig.ready_skills(achieved)
        if not candidates:
            candidates = list(self.sig.skills.keys())
        skills = [self.sig.skills[c] for c in candidates]

        if phase == "novel":
            scores = [self.score_novelty(s) for s in skills]
        elif phase == "confirm":
            scores = [self.score_confirm(s) for s in skills]
        else:
            scores = [self.score_goal(s, task_goal) for s in skills]

        best_idx = int(np.argmax(scores))
        return skills[best_idx].skill_id

    def select(self, achieved: List[int], task_goal: Optional[Subgoal] = None) -> int:
        """
        If task_goal is provided and corresponds to a known skill, return the first 'ready'
        skill in its prerequisite plan. Otherwise fallback to non-goal selection.
        """
        if task_goal is not None:
            plan = self.goal_planner.plan_for_subgoal(task_goal, achieved)
            if plan is not None and plan.skills:
                # pick first ready skill from the plan
                ready = set(self.sig.ready_skills(achieved))
                for sid in plan.skills:
                    if sid in ready:
                        return sid
        # fallback
        return self._select_non_goal(achieved, task_goal)
