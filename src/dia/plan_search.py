# src/dia/plan_search.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Set

from .sig import SIGraph, Skill
from .types import Subgoal


def ancestors(sig: SIGraph, target_id: int) -> Set[int]:
    """Return all prerequisite (ancestor) skills for target_id."""
    visited: Set[int] = set()
    stack = [target_id]
    while stack:
        u = stack.pop()
        for pre in sig.prerequisites(u):
            if pre not in visited:
                visited.add(pre)
                stack.append(pre)
    return visited


def plan_from_prereqs(sig: SIGraph, target_id: int, achieved: List[int]) -> List[int]:
    """
    Return an ordered list of skill_ids to execute to enable and then execute `target_id`.
    Ordering is topological over the induced subgraph of required ancestors.
    Excludes skills already in `achieved`. Includes `target_id` last.
    """
    required = ancestors(sig, target_id)
    required = [sid for sid in required if sid not in set(achieved)]
    if not required and target_id in achieved:
        return []  # already done
    # restrict to the subgraph on required + target
    sub_nodes = set(required) | {target_id}

    indeg = {sid: 0 for sid in sub_nodes}
    for u in list(sub_nodes):
        for v in sig.successors(u):
            if v in sub_nodes:
                indeg[v] += 1

    # Kahn's algorithm
    q = [u for u, d in indeg.items() if d == 0]
    order: List[int] = []
    while q:
        u = q.pop(0)
        order.append(u)
        for v in sig.successors(u):
            if v in sub_nodes:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

    # remove those already achieved; ensure target is last
    plan = [sid for sid in order if sid != target_id and sid not in set(achieved)]
    plan.append(target_id)
    return plan


def find_skill_for_subgoal(sig: SIGraph, subgoal: Subgoal) -> Optional[int]:
    """Return the skill_id that exactly matches the given subgoal, if any."""
    for sid, sk in sig.skills.items():
        if sk.subgoal == subgoal:
            return sid
    return None


@dataclass
class GoalPlan:
    skills: List[int]          # ordered sequence of skill_ids to execute
    target_skill: int          # equals skills[-1]


class GoalPlanner:
    """
    Lightweight goal-directed planner over the SIG.
    Given a target Subgoal, it finds the corresponding skill and returns
    a topologically ordered plan of prerequisites ending in that skill.
    """

    def __init__(self, sig: SIGraph):
        self.sig = sig

    def plan_for_subgoal(self, subgoal: Subgoal, achieved: List[int]) -> Optional[GoalPlan]:
        sid = find_skill_for_subgoal(self.sig, subgoal)
        if sid is None:
            return None
        seq = plan_from_prereqs(self.sig, sid, achieved)
        return GoalPlan(skills=seq, target_skill=sid)

    def plan_for_skill(self, target_skill_id: int, achieved: List[int]) -> GoalPlan:
        seq = plan_from_prereqs(self.sig, target_skill_id, achieved)
        return GoalPlan(skills=seq, target_skill=target_skill_id)
