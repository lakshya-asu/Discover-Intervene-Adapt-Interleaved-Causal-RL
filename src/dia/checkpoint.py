# src/dia/checkpoint.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from .sig import SIGraph, Skill
from .types import Subgoal, Predicate


# ---------------------------- SIG serialization ----------------------------

def _sig_to_dict(sig: SIGraph) -> Dict[str, Any]:
    skills = []
    for sid, sk in sig.skills.items():
        skills.append({
            "skill_id": sid,
            "name": sk.name,
            "subgoal": {
                "var_index": sk.subgoal.var_index,
                "predicate": sk.subgoal.predicate.value,
                "value": sk.subgoal.value,
            },
            "success_rate": sk.success_rate,
            "attempts": sk.attempts,
            "successes": sk.successes,
            "effect_signature": None if sk.effect_signature is None else sk.effect_signature.tolist(),
        })
    edges = []
    for u, outs in sig.edges.items():
        for v in outs:
            edges.append([u, v])
    return {"skills": skills, "edges": edges}


def _sig_from_dict(sig: SIGraph, data: Dict[str, Any]) -> SIGraph:
    sig.skills.clear()
    sig.edges.clear()
    sig.rev_edges.clear()

    for s in data.get("skills", []):
        sg = Subgoal(
            var_index=int(s["subgoal"]["var_index"]),
            predicate=Predicate(s["subgoal"]["predicate"]),
            value=s["subgoal"]["value"],
        )
        sk = Skill(
            skill_id=int(s["skill_id"]),
            subgoal=sg,
            name=s.get("name", f"skill_{int(s['skill_id'])}"),
        )
        sk.success_rate = float(s.get("success_rate", 0.0))
        sk.attempts = int(s.get("attempts", 0))
        sk.successes = int(s.get("successes", 0))
        eff = s.get("effect_signature", None)
        if eff is not None:
            sk.effect_signature = np.array(eff, dtype=float)
        sig.add_skill(sk)

    for u, v in data.get("edges", []):
        sig.add_prerequisite(int(u), int(v))

    return sig


# ---------------------------- PCG serialization ----------------------------

def _pcg_to_dict(pcg: Any) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"pcg_type": pcg.__class__.__name__}
    # Always store probs if available
    if hasattr(pcg, "probs"):
        meta["probs"] = getattr(pcg, "probs").tolist()

    # DifferentiablePCG (NOTEARS-like): has W
    if hasattr(pcg, "W"):
        try:
            W = pcg.W.detach().cpu().numpy()
            meta["W"] = W.tolist()
        except Exception:
            pass

    # VariationalPCG: has L (logits) and B (strengths)
    if hasattr(pcg, "L") and hasattr(pcg, "B"):
        try:
            L = pcg.L.detach().cpu().numpy()
            B = pcg.B.detach().cpu().numpy()
            meta["L"] = L.tolist()
            meta["B"] = B.tolist()
        except Exception:
            pass

    return meta


def _pcg_from_dict(pcg: Any, data: Dict[str, Any]) -> Any:
    # Try to restore parameters if shapes match
    if "W" in data and hasattr(pcg, "W"):
        W = np.array(data["W"], dtype=float)
        try:
            import torch
            with torch.no_grad():
                pcg.W.copy_(torch.tensor(W, device=pcg.W.device, dtype=pcg.W.dtype))
                pcg.W.data.diagonal().zero_()
        except Exception:
            pass

    if "L" in data and hasattr(pcg, "L"):
        L = np.array(data["L"], dtype=float)
        try:
            import torch
            with torch.no_grad():
                pcg.L.copy_(torch.tensor(L, device=pcg.L.device, dtype=pcg.L.dtype))
                pcg.L.data.diagonal().fill_(-1e9)
        except Exception:
            pass

    if "B" in data and hasattr(pcg, "B"):
        B = np.array(data["B"], dtype=float)
        try:
            import torch
            with torch.no_grad():
                pcg.B.copy_(torch.tensor(B, device=pcg.B.device, dtype=pcg.B.dtype))
                pcg.B.data.diagonal().zero_()
        except Exception:
            pass

    # If only probs exist and the PCG supports apply_update(), use it to set probs
    if "probs" in data and hasattr(pcg, "apply_update"):
        probs = np.array(data["probs"], dtype=float)
        try:
            pcg.apply_update(probs)
        except Exception:
            pass

    return pcg


# ---------------------------- Options serialization (lightweight) ----------------------------

def _options_to_dict(options: Dict[int, Any]) -> Dict[str, Any]:
    """
    Save lightweight metadata. If an option exposes `.save(path)`, you can save models separately.
    Here we only record class names and subgoal spec; model files are out-of-scope.
    """
    out = {}
    for sid, opt in options.items():
        out[str(sid)] = {
            "class": opt.__class__.__name__,
            "subgoal": {
                "var_index": opt.subgoal.var_index,
                "predicate": opt.subgoal.predicate.value,
                "value": opt.subgoal.value,
            },
        }
    return out


# ---------------------------- Public API ----------------------------

def save_checkpoint(path: str, pcg: Any, sig: SIGraph, options: Optional[Dict[int, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save PCG+SIG (and lightweight option metadata) to a JSON file.
    (Models for PPO options should be saved separately via their .save() method.)
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "pcg": _pcg_to_dict(pcg),
        "sig": _sig_to_dict(sig),
        "options": _options_to_dict(options or {}),
        "metadata": metadata or {},
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_checkpoint(path: str, pcg: Any, sig: SIGraph) -> Dict[str, Any]:
    """
    Load from JSON checkpoint and restore into provided pcg and sig objects.
    Returns the raw dict (including options metadata) for the caller to inspect.
    """
    with open(path, "r") as f:
        data = json.load(f)
    _pcg_from_dict(pcg, data.get("pcg", {}))
    _sig_from_dict(sig, data.get("sig", {}))
    return data
