# src/dia/sig_auto.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np

from .sig import SIGraph, Skill
from .types import Subgoal, Predicate
from .evgs import EVGS


@dataclass
class AutoSIGConfig:
    add_threshold: float = 0.75     # add prereq if p(i->j) >= add_threshold
    remove_threshold: float = 0.55  # remove prereq if p(i->j) < remove_threshold
    create_missing_skills: bool = True
    default_predicate: Predicate = Predicate.UP  # default predicate for auto-created skills
    verbose: bool = False


def _find_skill_with_subgoal(sig: SIGraph, sg: Subgoal) -> Optional[int]:
    for sid, sk in sig.skills.items():
        if sk.subgoal == sg:
            return sid
    return None


def _get_or_create_skill(sig: SIGraph, var_index: int, evgs: EVGS,
                         pred: Predicate, name_prefix: str = "") -> int:
    sg = Subgoal(var_index=var_index, predicate=pred)
    sid = _find_skill_with_subgoal(sig, sg)
    if sid is not None:
        return sid
    # create a new skill id
    sid = 0 if not sig.skills else (max(sig.skills.keys()) + 1)
    name = f"{name_prefix}{evgs.names()[var_index]}:{pred.value}"
    sig.add_skill(Skill(skill_id=sid, subgoal=sg, name=name))
    return sid


def _remove_prerequisite(sig: SIGraph, u: int, v: int):
    # safe removal on both adjacency structures
    if u in sig.edges:
        sig.edges[u] = [x for x in sig.edges[u] if x != v]
    if v in sig.rev_edges:
        sig.rev_edges[v] = [x for x in sig.rev_edges[v] if x != u]


def expand_sig_from_pcg(sig: SIGraph, evgs: EVGS, probs: np.ndarray, cfg: AutoSIGConfig) -> Dict[str, int]:
    """
    Expand/trim SIG prerequisites based on PCG edge probabilities.
    Adds prerequisites u->v when p(i->j) high, mapping i to a (possibly auto-created) skill on var i,
    and v to the skill that changes var j (creating it if missing and allowed).

    Returns counts: {"added": n, "removed": m, "created_skills": k}
    """
    assert probs.ndim == 2 and probs.shape[0] == probs.shape[1], "probs must be [d,d]"
    d = probs.shape[0]
    added = removed = created = 0

    # ensure at least one skill per variable if configured
    for j in range(d):
        sg_j = Subgoal(var_index=j, predicate=cfg.default_predicate)
        if _find_skill_with_subgoal(sig, sg_j) is None and cfg.create_missing_skills:
            _get_or_create_skill(sig, j, evgs, cfg.default_predicate, name_prefix="")
            created += 1

    # map var->skill id for target change (predicate default)
    var_to_skill = {}
    for sid, sk in sig.skills.items():
        if sk.subgoal.predicate == cfg.default_predicate:
            var_to_skill.setdefault(sk.subgoal.var_index, sid)

    # add/remove edges based on thresholds
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            pij = float(probs[i, j])
            # identify prerequisite and target skills
            pre_sid = var_to_skill.get(i)
            if pre_sid is None and cfg.create_missing_skills:
                pre_sid = _get_or_create_skill(sig, i, evgs, cfg.default_predicate, name_prefix="")
                var_to_skill[i] = pre_sid
                created += 1

            tgt_sid = var_to_skill.get(j)
            if tgt_sid is None and cfg.create_missing_skills:
                tgt_sid = _get_or_create_skill(sig, j, evgs, cfg.default_predicate, name_prefix="")
                var_to_skill[j] = tgt_sid
                created += 1

            if pre_sid is None or tgt_sid is None:
                continue

            # is edge present?
            present = tgt_sid in sig.edges.get(pre_sid, [])
            if pij >= cfg.add_threshold and not present:
                sig.add_prerequisite(pre_sid, tgt_sid)
                added += 1
                if cfg.verbose:
                    print(f"[SIG-AUTO] add prereq {pre_sid} -> {tgt_sid} (p={pij:.2f})")
            elif pij < cfg.remove_threshold and present:
                _remove_prerequisite(sig, pre_sid, tgt_sid)
                removed += 1
                if cfg.verbose:
                    print(f"[SIG-AUTO] remove prereq {pre_sid} -> {tgt_sid} (p={pij:.2f})")

    return {"added": added, "removed": removed, "created_skills": created}
