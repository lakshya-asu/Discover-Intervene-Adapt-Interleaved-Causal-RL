# tests/test_runner_checkpoint.py
from __future__ import annotations

import os
import numpy as np

from dia.evgs import EVGS
from dia.types import Subgoal, Predicate
from dia.sig import SIGraph, Skill
from dia.planner import PlannerConfig, InterventionSelector
from dia.rollout import DIARunner, RunnerConfig
from dia.pcg_learner import DifferentiablePCG, DifferentiablePCGConfig
from dia.checkpoint import save_checkpoint, load_checkpoint


class _ActionSpace:
    def __init__(self):
        self.n = 2
    def sample(self):
        return np.random.randint(0, self.n)


class DummyEnv:
    """
    Minimal gym-like env:
      - obs is a 3D vector
      - action 1 increases X0; action 0 decreases X1
    """
    def __init__(self):
        self.action_space = _ActionSpace()
        self.state = np.zeros(3, dtype=float)

    def reset(self):
        self.state[:] = 0.0
        return self.state.copy()

    def step(self, action):
        if action == 1:
            self.state[0] += 1.0
        else:
            self.state[1] -= 1.0
        obs = self.state.copy()
        reward = 0.0
        done = False
        info = {"soft_continue": True}
        return obs, reward, done, info

    def get_obs(self):
        return self.state.copy()


def test_runner_checkpoint(tmp_path):
    # EVGS: identity for 3 variables
    evgs = EVGS(var_names=["X0", "X1", "X2"], obs_to_vars=lambda o: np.array(o, dtype=float))

    # PCG learner
    pcg = DifferentiablePCG(DifferentiablePCGConfig(num_vars=3, max_iter=50, lr=1e-2, verbose=False))

    # SIG: two skills
    sig = SIGraph()
    s_increase_x0 = Skill(skill_id=0, subgoal=Subgoal(0, Predicate.UP), name="inc_X0")
    s_decrease_x1 = Skill(skill_id=1, subgoal=Subgoal(1, Predicate.DOWN), name="dec_X1")
    sig.add_skill(s_increase_x0)
    sig.add_skill(s_decrease_x1)
    sig.add_prerequisite(0, 1)

    selector = InterventionSelector(pcg, sig, PlannerConfig())
    env = DummyEnv()

    cfg = RunnerConfig(fit_every=2, pcg_epochs=50, min_buffer=4, batch_recent=16)
    runner = DIARunner(env, evgs, pcg, sig, selector, cfg, logger=None, option_factory=None)

    achieved = []
    for _ in range(10):
        rec = runner.step(achieved, task_goal=None)

    # Save checkpoint
    ckpt_path = os.path.join(tmp_path, "ckpt.json")
    save_checkpoint(ckpt_path, pcg, sig, options=runner.options)

    # Capture old probs
    old_probs = pcg.probs.copy()

    # Load into a fresh PCG+SIG
    pcg2 = DifferentiablePCG(DifferentiablePCGConfig(num_vars=3, max_iter=1, lr=1e-2, verbose=False))
    sig2 = SIGraph()
    load_checkpoint(ckpt_path, pcg2, sig2)
    new_probs = pcg2.probs

    assert np.allclose(old_probs, new_probs, atol=1e-6), "PCG probabilities should round-trip via checkpoint"
    assert len(sig2.skills) == len(sig.skills) and len(sig2.edges) == len(sig.edges), "SIG structure should restore"
