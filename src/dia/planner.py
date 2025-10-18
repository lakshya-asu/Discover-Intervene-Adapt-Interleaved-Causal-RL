from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np

from .pcg import SimplePCG
from .sig import SIGraph, Skill
from .types import Subgoal
from .intrinsic import entropy_bernoulli, BetaScheduler


@dataclass
class PlannerConfig:
    novel_fraction: float = 0.6       # early training: prefer novelty (IG) over exploitation
    confirm_fraction: float = 0.2     # mid: repeat to confirm
    goal_fraction: float = 0.2        # late: prioritize task-directed
    entropy_high: float = 25.0        # thresholds to switch phases
    entropy_low: float = 5.0


class InterventionSelector:
    """Chooses next skill to execute using a novelty->confirmatory->goal-directed schedule."""
    def __init__(self, pcg: SimplePCG, sig: SIGraph, cfg: PlannerConfig):
        self.pcg = pcg
        self.sig = sig
        self.cfg = cfg
        self.beta_sched = BetaScheduler(beta_max=1.0, beta_min=0.0, h_ref=cfg.entropy_high)

    def phase(self) -> str:
        H = self.pcg.entropy()
        if H >= self.cfg.entropy_high:
            return "novel"
        if H <= self.cfg.entropy_low:
            return "goal"
        return "confirm"

    def score_novelty(self, skill: Skill) -> float:
        """Heuristic: score by edge uncertainty around skill target var."""
        # simplistic: sum entropy of incoming edges to the target's variable index
        p = self.pcg.probs.copy()
        j = skill.subgoal.var_index
        incoming = p[:, j]
        incoming[j] = 0.0
        eps = 1e-8
        h = -incoming * np.log(np.clip(incoming, eps, 1-eps)) - (1-incoming) * np.log(np.clip(1-incoming, eps, 1-eps))
        return float(np.sum(h))

    def score_confirm(self, skill: Skill) -> float:
        # prefer skills with mid success rates (uncertain)
        s = skill.success_rate
        return float(1.0 - abs(0.5 - s))

    def score_goal(self, skill: Skill, goal_subgoal: Optional[Subgoal]) -> float:
        # if a specific task subgoal is given, prioritize matching skills; otherwise use success rate
        if goal_subgoal and (skill.subgoal == goal_subgoal):
            return 1.0 + skill.success_rate
        return skill.success_rate

    def select(self, achieved: List[int], task_goal: Optional[Subgoal] = None) -> int:
        phase = self.phase()
        candidates = self.sig.ready_skills(achieved)
        if not candidates:
            # fallback: any skill
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
