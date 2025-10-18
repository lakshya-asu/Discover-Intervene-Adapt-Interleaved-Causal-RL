from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .types import Subgoal


@dataclass
class Skill:
    skill_id: int
    subgoal: Subgoal
    name: str
    effect_signature: Optional[np.ndarray] = None  # E[ X_{t+1} - X_t | execute skill ]
    success_rate: float = 0.0
    attempts: int = 0
    successes: int = 0

    def update_stats(self, success: bool, delta_x: Optional[np.ndarray] = None):
        self.attempts += 1
        if success:
            self.successes += 1
        self.success_rate = self.successes / max(1, self.attempts)
        if delta_x is not None:
            if self.effect_signature is None:
                self.effect_signature = np.array(delta_x, dtype=float)
            else:
                # running average
                self.effect_signature = 0.9 * self.effect_signature + 0.1 * np.array(delta_x, dtype=float)


class SIGraph:
    """Skill–Intervention Graph over learned options (skills)."""
    def __init__(self):
        # adjacency: u -> v if u is a prerequisite for v
        self.skills: Dict[int, Skill] = {}
        self.edges: Dict[int, List[int]] = {}         # forward adjacency
        self.rev_edges: Dict[int, List[int]] = {}     # reverse adjacency (predecessors)

    def add_skill(self, skill: Skill):
        self.skills[skill.skill_id] = skill
        self.edges.setdefault(skill.skill_id, [])
        self.rev_edges.setdefault(skill.skill_id, [])

    def add_prerequisite(self, pre_id: int, post_id: int):
        self.edges.setdefault(pre_id, [])
        if post_id not in self.edges[pre_id]:
            self.edges[pre_id].append(post_id)
        self.rev_edges.setdefault(post_id, [])
        if pre_id not in self.rev_edges[post_id]:
            self.rev_edges[post_id].append(pre_id)

    def prerequisites(self, skill_id: int) -> List[int]:
        return list(self.rev_edges.get(skill_id, []))

    def successors(self, skill_id: int) -> List[int]:
        return list(self.edges.get(skill_id, []))

    def ready_skills(self, achieved: List[int]) -> List[int]:
        """Return skills whose prerequisites are all in achieved."""
        achieved_set = set(achieved)
        ready = []
        for sid in self.skills:
            pres = self.prerequisites(sid)
            if all(p in achieved_set for p in pres):
                ready.append(sid)
        return ready

    def toposort(self) -> List[int]:
        """Topological order (Kahn)."""
        indeg = {sid: len(self.prerequisites(sid)) for sid in self.skills}
        q = [sid for sid, d in indeg.items() if d == 0]
        out = []
        while q:
            u = q.pop(0)
            out.append(u)
            for v in self.successors(u):
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        return out
