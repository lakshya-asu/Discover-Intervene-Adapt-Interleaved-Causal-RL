from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np

from .pcg import SimplePCG
from .sig import SIGraph, Skill
from .evgs import EVGS
from .types import Subgoal
from .planner import InterventionSelector, PlannerConfig
from .intrinsic import entropy_bernoulli


class DIARunner:
    """High-level training orchestrator for Discover–Intervene–Adapt."""
    def __init__(self, env, evgs: EVGS, pcg: SimplePCG, sig: SIGraph, selector: InterventionSelector):
        self.env = env
        self.evgs = evgs
        self.pcg = pcg
        self.sig = sig
        self.selector = selector
        self.history = []

    def step(self, achieved: List[int], task_goal: Optional[Subgoal] = None) -> Dict:
        skill_id = self.selector.select(achieved, task_goal)
        skill = self.sig.skills[skill_id]

        # Retrieve or construct an option policy for the skill.
        # In a full system, you'd map skill_id -> OptionPolicy instance in a registry.
        # For bootstrapping, we assume env has action_space and we use RandomOption.
        from .options import RandomOption, OptionConfig
        option = RandomOption(skill.subgoal, OptionConfig(max_steps=64), self.env.action_space)

        # Execute the option
        obs0 = self.env.get_obs() if hasattr(self.env, "get_obs") else self.env.reset()
        x0 = self.evgs.extract(obs0)
        out = option.run(self.env, self.evgs)
        x1 = self.evgs.extract(out["final_obs"])
        success = out["success"]

        # Update SIG stats
        delta_x = x1 - x0
        skill.update_stats(success=success, delta_x=delta_x)

        # Very simple PCG update heuristic: if a skill targeting X_j succeeded, increase prob of some incoming edges.
        # This is placeholder logic; replace with a proper learner.
        j = skill.subgoal.var_index
        delta = np.zeros_like(self.pcg.probs)
        # Encourage edges from vars that changed when executing the skill
        changed = np.where(np.abs(delta_x) > 1e-6)[0]
        for i in changed:
            if i != j:
                delta[i, j] += 0.05
        new_p, ig = self.pcg.conservative_update(delta, lr=0.5)

        rec = {
            "skill_id": skill_id,
            "success": success,
            "delta_x": delta_x,
            "ig": ig,
            "pcg_entropy": self.pcg.entropy(),
        }
        self.history.append(rec)
        return rec
