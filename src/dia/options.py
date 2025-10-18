from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple
import time

import numpy as np

from .types import Subgoal
from .evgs import EVGS


@dataclass
class OptionConfig:
    max_steps: int = 128
    terminate_on_success: bool = True


class OptionPolicy:
    """Abstract option policy: executes low-level actions to achieve a subgoal."""
    def __init__(self, subgoal: Subgoal, cfg: OptionConfig):
        self.subgoal = subgoal
        self.cfg = cfg

    def act(self, obs) -> Any:
        """Return a primitive action for the environment. Override in subclasses."""
        raise NotImplementedError

    def update(self, transition: Dict[str, Any]):
        """Update internal policy from a transition. Override in subclasses if learning online."""
        pass

    def run(self, env, evgs: EVGS) -> Dict[str, Any]:
        """Execute until success or horizon, returning a summary dict."""
        obs = env.get_obs() if hasattr(env, "get_obs") else env.reset()
        x_t = evgs.extract(obs)
        steps = 0
        success = False
        trajectory = []
        while steps < self.cfg.max_steps:
            action = self.act(obs)
            next_obs, ext_rew, done, info = env.step(action)
            x_tp1 = evgs.extract(next_obs)
            succ_this = evgs.predicate_holds(x_t, x_tp1, self.subgoal)
            trajectory.append((obs, action, next_obs, succ_this))
            self.update({"obs": obs, "action": action, "next_obs": next_obs, "info": info})
            obs = next_obs
            steps += 1
            if succ_this:
                success = True
                if self.cfg.terminate_on_success:
                    break
            if done:
                # allow continuing if environment exposes "soft_done" to ignore episodic boundaries
                if not info.get("soft_continue", False):
                    break
            x_t = x_tp1
        return {"success": success, "steps": steps, "trajectory": trajectory, "final_obs": obs}


class RandomOption(OptionPolicy):
    """Fallback option that samples random actions (for bootstrapping skills)."""
    def __init__(self, subgoal: Subgoal, cfg: OptionConfig, action_space):
        super().__init__(subgoal, cfg)
        self.action_space = action_space

    def act(self, obs):
        return self.action_space.sample()
