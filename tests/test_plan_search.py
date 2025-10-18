# tests/test_plan_search.py
from __future__ import annotations

from dia.sig import SIGraph, Skill
from dia.types import Subgoal, Predicate
from dia.plan_search import plan_from_prereqs, find_skill_for_subgoal, GoalPlanner


def build_sig_chain():
    # 0 -> 1 -> 2 (0 prerequisite for 1, 1 prerequisite for 2)
    sig = SIGraph()
    s0 = Skill(skill_id=0, subgoal=Subgoal(0, Predicate.UP), name="s0")
    s1 = Skill(skill_id=1, subgoal=Subgoal(1, Predicate.UP), name="s1")
    s2 = Skill(skill_id=2, subgoal=Subgoal(2, Predicate.UP), name="s2")
    for s in (s0, s1, s2):
        sig.add_skill(s)
    sig.add_prerequisite(0, 1)
    sig.add_prerequisite(1, 2)
    return sig, s0, s1, s2


def test_plan_from_prereqs():
    sig, s0, s1, s2 = build_sig_chain()
    achieved = [0]
    plan = plan_from_prereqs(sig, target_id=2, achieved=achieved)
    assert plan == [1, 2], f"expected [1,2], got {plan}"


def test_goalplanner_find_skill():
    sig, s0, s1, s2 = build_sig_chain()
    gp = GoalPlanner(sig)
    sid = find_skill_for_subgoal(sig, s2.subgoal)
    assert sid == 2
    plan = gp.plan_for_skill(2, achieved=[])
    assert plan.skills[-1] == 2
    assert plan.skills[:2] == [0, 1]
