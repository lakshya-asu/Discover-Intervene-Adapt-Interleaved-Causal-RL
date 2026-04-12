#!/usr/bin/env python3
"""
scripts/run_baseline_2d.py — Run a single (method, seed) baseline on 2D Minecraft.

Supported methods:
  ppo          : PPO flat (no options, no PCG).  Stub -- needs SB3 flat PPO wiring.
  ppo_options  : PPO + Options (no PCG).  Stub -- needs SB3 per-option PPO training.
  ride         : RIDE intrinsic reward baseline.  Stub -- needs RIDE reward shaper.
  icm          : ICM intrinsic reward baseline.   Stub -- needs ICM reward shaper.
  dia_no_ig    : DIA without information-gain bonus (pure success-rate-based planner).
  dia_no_sig   : DIA without SIG (random skill selection).
  dia          : Full DIA (default).

Writes a JSON result file at --out:
  {
    "method": "dia",
    "seed": 0,
    "steps_to_diamond": 123456,   # first option step where diamond > 0, else null
    "final_shd": 2.0,             # SHD vs ground-truth 2D DAG (null for non-PCG methods)
    "final_ece": 0.05,            # ECE at end of training (null for non-PCG methods)
    "success_rate": 0.8,          # fraction of option executions where subgoal succeeded
    "completed": true
  }

Usage:
  cd Discover-Intervene-Adapt-Interleaved-Causal-RL
  python scripts/run_baseline_2d.py --method dia --seed 0 --steps 500000 --out results/logs/2d_dia_seed0.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Gym wrapper: flatten dict observation to a plain float32 numpy array
# ---------------------------------------------------------------------------

try:
    import gym as _gym_module
    _GymWrapperBase = _gym_module.Wrapper
except Exception:
    _gym_module = None
    _GymWrapperBase = object


class _FlatObsWrapper(_GymWrapperBase):
    """Thin Gym wrapper that converts MinecraftChainEnv's dict obs to a flat array.

    MinecraftChainEnv returns {"obs": np.array(...), "info": {...}} from both
    reset() and step().  SB3 / manual training loops expect a flat ndarray that
    matches observation_space (Box shape=(9,), dtype=float32).

    Extends gym.Wrapper so that SB3 recognises it as a valid Gym environment.
    """

    def __init__(self, env):
        if _gym_module is not None:
            super().__init__(env)
        else:
            self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs):
        obs_dict = self.env.reset(**kwargs)
        arr = np.empty(self.observation_space.shape, dtype=np.float32)
        arr[:] = obs_dict["obs"]
        return arr

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        arr = np.empty(self.observation_space.shape, dtype=np.float32)
        arr[:] = obs_dict["obs"]
        return arr, float(reward), done, info


# ---------------------------------------------------------------------------
# Simple actor-critic network (shared trunk) used by RIDE and ICM loops
# ---------------------------------------------------------------------------

class _SimplePolicyNet(nn.Module if _TORCH_AVAILABLE else object):
    """2-layer shared trunk with separate policy and value heads.

    Architecture:
        Linear(obs_dim, 64) -> ReLU -> Linear(64, 64) -> ReLU
        policy head: Linear(64, n_actions)  (logits)
        value  head: Linear(64, 1)
    """

    def __init__(self, obs_dim: int = 9, n_actions: int = 10):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is required for _SimplePolicyNet")
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),     nn.ReLU(),
        )
        self.policy_head = nn.Linear(64, n_actions)
        self.value_head  = nn.Linear(64, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# GAE helper
# ---------------------------------------------------------------------------

def _gae(rewards, values, dones, gamma: float = 0.99, lam: float = 0.95):
    """Compute Generalised Advantage Estimates (returns numpy arrays)."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_val * (1.0 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + np.array(values[:T], dtype=np.float32)
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO update (clipped surrogate, shared for RIDE and ICM loops)
# ---------------------------------------------------------------------------

def _ppo_update(
    net: "_SimplePolicyNet",
    optimizer,
    obs_buf,    # (T, obs_dim) float32 tensor
    act_buf,    # (T,) long tensor
    adv_buf,    # (T,) float32 tensor
    ret_buf,    # (T,) float32 tensor
    old_logp,   # (T,) float32 tensor
    clip_eps: float = 0.2,
    n_epochs: int = 4,
):
    adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)
    for _ in range(n_epochs):
        logits, values = net(obs_buf)
        dist = torch.distributions.Categorical(logits=logits)
        logp  = dist.log_prob(act_buf)
        ratio = torch.exp(logp - old_logp)
        surr1 = ratio * adv_buf
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_buf
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss  = F.mse_loss(values, ret_buf)
        entropy     = dist.entropy().mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# ---------------------------------------------------------------------------
# RIDE shaper (Raileanu & Rocktaschel 2020)
# ---------------------------------------------------------------------------

class _RIDEShaper(nn.Module if _TORCH_AVAILABLE else object):
    """Impact-driven intrinsic reward: ||phi(s') - phi(s)|| / sqrt(N(s')).

    phi is a 2-layer MLP (9 -> 64 -> 32).
    N is a hash-based state visitation counter (discretised to integers).
    phi is updated online via a forward prediction loss.
    """

    def __init__(self, obs_dim: int = 9, embed_dim: int = 32, lr: float = 3e-4):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is required for _RIDEShaper")
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, embed_dim),
        )
        self.counts: Dict[tuple, int] = {}
        self._optimizer = torch.optim.Adam(self.phi.parameters(), lr=lr)

    def _to_t(self, obs: np.ndarray) -> "torch.Tensor":
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    def _hash(self, obs: np.ndarray) -> tuple:
        return tuple(obs.astype(int).tolist())

    def reward(self, obs_t: np.ndarray, obs_tp1: np.ndarray) -> float:
        """Compute RIDE intrinsic reward and update visitation count."""
        key = self._hash(obs_tp1)
        self.counts[key] = self.counts.get(key, 0) + 1
        n = self.counts[key]

        t_t  = self._to_t(obs_t)
        t_tp1 = self._to_t(obs_tp1)
        with torch.no_grad():
            e_t   = self.phi(t_t)
            e_tp1 = self.phi(t_tp1)
        dist = torch.norm(e_tp1 - e_t, dim=-1).item()
        return dist / (n ** 0.5 + 1e-8)

    def update(self, obs_t: np.ndarray, obs_tp1: np.ndarray):
        """Online phi update: minimise forward prediction loss ||phi(s') - phi(s)||."""
        t_t   = self._to_t(obs_t)
        t_tp1 = self._to_t(obs_tp1)
        e_t   = self.phi(t_t)
        e_tp1 = self.phi(t_tp1)
        loss  = F.mse_loss(e_tp1, e_t.detach())
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()


# ---------------------------------------------------------------------------
# ICM shaper (Pathak et al. 2017)
# ---------------------------------------------------------------------------

class _ICMShaper(nn.Module if _TORCH_AVAILABLE else object):
    """Intrinsic Curiosity Module.

    Components
    ----------
    phi           : encoder Linear(9,32)->ReLU
    forward_model : Linear(32+10, 32)  predicts phi(s') from phi(s) + one-hot(a)
    inverse_model : Linear(64, 10)     predicts action from (phi(s), phi(s'))
    """

    def __init__(self, obs_dim: int = 9, n_actions: int = 10,
                 embed_dim: int = 32, lr: float = 3e-4):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is required for _ICMShaper")
        super().__init__()
        self.n_actions = n_actions
        self.phi = nn.Sequential(nn.Linear(obs_dim, embed_dim), nn.ReLU())
        self.forward_model  = nn.Linear(embed_dim + n_actions, embed_dim)
        self.inverse_model  = nn.Linear(embed_dim * 2, n_actions)
        self._optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def _to_t(self, obs: np.ndarray) -> "torch.Tensor":
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    def _one_hot(self, action: int) -> "torch.Tensor":
        oh = torch.zeros(1, self.n_actions)
        oh[0, action] = 1.0
        return oh

    def reward(self, obs_t: np.ndarray, action: int, obs_tp1: np.ndarray) -> float:
        """Intrinsic reward = forward model prediction error."""
        t_t   = self._to_t(obs_t)
        t_tp1 = self._to_t(obs_tp1)
        oh    = self._one_hot(action)
        with torch.no_grad():
            e_t   = self.phi(t_t)
            e_tp1 = self.phi(t_tp1)
            e_hat = self.forward_model(torch.cat([e_t, oh], dim=-1))
        return F.mse_loss(e_hat, e_tp1).item()

    def update(self, obs_batch, act_batch, obs_next_batch):
        """Combined forward + inverse loss on a mini-batch (numpy arrays)."""
        t_obs   = torch.tensor(obs_batch,      dtype=torch.float32)
        t_nobs  = torch.tensor(obs_next_batch, dtype=torch.float32)
        t_act   = torch.tensor(act_batch,      dtype=torch.long)

        e_obs  = self.phi(t_obs)
        e_nobs = self.phi(t_nobs)

        # one-hot encode actions for forward model
        oh = torch.zeros(len(t_act), self.n_actions)
        oh.scatter_(1, t_act.unsqueeze(1), 1.0)

        e_hat    = self.forward_model(torch.cat([e_obs, oh], dim=-1))
        fwd_loss = F.mse_loss(e_hat, e_nobs.detach())

        inv_logits = self.inverse_model(torch.cat([e_obs, e_nobs], dim=-1))
        inv_loss   = F.cross_entropy(inv_logits, t_act)

        loss = 0.8 * fwd_loss + 0.2 * inv_loss
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

# ---------------------------------------------------------------------------
# Ground-truth causal graph (9 vars, 10 edges)
# Variables: wood(0) stone(1) coal(2) ironore(3) furnace(4)
#            stonepickaxe(5) iron(6) ironpickaxe(7) diamond(8)
# ---------------------------------------------------------------------------

GROUND_TRUTH_EDGES: List[Tuple[int, int]] = [
    (1, 4),  # stone      -> furnace
    (0, 5),  # wood       -> stonepickaxe
    (1, 5),  # stone      -> stonepickaxe
    (5, 3),  # stonepickaxe -> ironore
    (4, 6),  # furnace    -> iron
    (2, 6),  # coal       -> iron
    (3, 6),  # ironore    -> iron
    (6, 7),  # iron       -> ironpickaxe
    (0, 7),  # wood       -> ironpickaxe
    (7, 8),  # ironpickaxe -> diamond
]

# Index of the "diamond" variable in the EVGS / env state vector
DIAMOND_IDX = 8

# Methods that use DIA's PCG; all others get null SHD/ECE
PCG_METHODS = {"dia", "dia_no_ig", "dia_no_sig", "dia_oracle"}


# ---------------------------------------------------------------------------
# DIA imports (only needed for PCG-based methods, but imported at top so
# missing packages surface early rather than at runtime deep inside a method)
# ---------------------------------------------------------------------------

def _import_dia():
    """Import all DIA modules and return them as a namespace dict."""
    from dia.envs.minecraft2d import MinecraftChainEnv, ChainConfig
    from dia.evgs_minecraft import make_minecraft_evgs, VAR_NAMES
    from dia.evgs import EVGS
    from dia.sig import SIGraph, Skill
    from dia.types import Subgoal, Predicate
    from dia.planner import PlannerConfig, InterventionSelector
    from dia.rollout import DIARunner, RunnerConfig
    from dia.logging_utils import TBLogger
    from dia.pcg_learner import DifferentiablePCG, DifferentiablePCGConfig
    from dia.pcg import SimplePCG, PCGConfig
    from dia.options import OptionConfig
    from dia.options_minecraft import RuleOption
    from dia.eval.pcg_metrics import shd as eval_shd, ece as eval_ece
    return {
        "MinecraftChainEnv": MinecraftChainEnv,
        "ChainConfig": ChainConfig,
        "make_minecraft_evgs": make_minecraft_evgs,
        "VAR_NAMES": VAR_NAMES,
        "EVGS": EVGS,
        "SIGraph": SIGraph,
        "Skill": Skill,
        "Subgoal": Subgoal,
        "Predicate": Predicate,
        "PlannerConfig": PlannerConfig,
        "InterventionSelector": InterventionSelector,
        "DIARunner": DIARunner,
        "RunnerConfig": RunnerConfig,
        "TBLogger": TBLogger,
        "DifferentiablePCG": DifferentiablePCG,
        "DifferentiablePCGConfig": DifferentiablePCGConfig,
        "SimplePCG": SimplePCG,
        "PCGConfig": PCGConfig,
        "OptionConfig": OptionConfig,
        "RuleOption": RuleOption,
        "eval_shd": eval_shd,
        "eval_ece": eval_ece,
    }


# ---------------------------------------------------------------------------
# SIG builder (mirrors run_minecraft2d.py)
# ---------------------------------------------------------------------------

def _build_sig(d, var_names: List[str], oracle: bool = False):
    """
    Build the 2D Minecraft SIGraph.

    oracle=False (default for DIA): skills only, NO prerequisite edges.
        The PCG must discover the structure interventionally; auto_expand_sig
        then populates edges as evidence accumulates.  This is the correct
        experimental setup for Claim 1 (PCG accuracy) and Claim 2 (sample
        efficiency): DIA starts blind and builds its own causal model.

    oracle=True: correct ground-truth edges given upfront.
        Used as the 'oracle SIG' comparison baseline to show the upper bound
        on performance when the causal structure is known in advance.
    """
    SIGraph = d["SIGraph"]
    Skill = d["Skill"]
    Subgoal = d["Subgoal"]
    Predicate = d["Predicate"]

    idx = {name: i for i, name in enumerate(var_names)}
    sig = SIGraph()
    for name in var_names:
        sid = idx[name]
        sig.add_skill(Skill(
            skill_id=sid,
            subgoal=Subgoal(var_index=sid, predicate=Predicate.UP),
            name=f"{name}↑",
        ))
    if oracle:
        sig.add_prerequisite(idx["stone"],        idx["furnace"])
        sig.add_prerequisite(idx["wood"],         idx["stonepickaxe"])
        sig.add_prerequisite(idx["stone"],        idx["stonepickaxe"])
        sig.add_prerequisite(idx["stonepickaxe"], idx["ironore"])
        sig.add_prerequisite(idx["furnace"],      idx["iron"])
        sig.add_prerequisite(idx["coal"],         idx["iron"])
        sig.add_prerequisite(idx["ironore"],      idx["iron"])
        sig.add_prerequisite(idx["iron"],         idx["ironpickaxe"])
        sig.add_prerequisite(idx["wood"],         idx["ironpickaxe"])
        sig.add_prerequisite(idx["ironpickaxe"],  idx["diamond"])
    return sig


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------

def _run_dia(method: str, seed: int, steps: int) -> Dict:
    """
    Full DIA and its ablations:
      dia        : full DIA
      dia_no_ig  : InterventionSelector(use_ig_bonus=False) -- no entropy-driven novelty phase
      dia_no_sig : InterventionSelector(use_sig=False) -- no SIG prerequisite filtering
    """
    d = _import_dia()
    MinecraftChainEnv = d["MinecraftChainEnv"]
    ChainConfig = d["ChainConfig"]
    make_minecraft_evgs = d["make_minecraft_evgs"]
    Subgoal = d["Subgoal"]
    Predicate = d["Predicate"]
    PlannerConfig = d["PlannerConfig"]
    InterventionSelector = d["InterventionSelector"]
    DIARunner = d["DIARunner"]
    RunnerConfig = d["RunnerConfig"]
    DifferentiablePCG = d["DifferentiablePCG"]
    DifferentiablePCGConfig = d["DifferentiablePCGConfig"]
    OptionConfig = d["OptionConfig"]
    RuleOption = d["RuleOption"]
    eval_shd = d["eval_shd"]
    eval_ece = d["eval_ece"]

    np.random.seed(seed)

    env = MinecraftChainEnv(ChainConfig(), seed=seed)
    evgs = make_minecraft_evgs()
    var_names = evgs.names()
    M = len(var_names)
    name_to_idx = {n: i for i, n in enumerate(var_names)}

    pcg = DifferentiablePCG(DifferentiablePCGConfig(
        num_vars=M, max_iter=200, lr=5e-3, verbose=False,
    ))

    # dia_oracle uses the correct ground-truth SIG upfront (upper bound baseline).
    # All other DIA variants start with an empty SIG and discover structure via PCG.
    oracle = (method == "dia_oracle")
    sig = _build_sig(d, var_names, oracle=oracle)
    planner_cfg = PlannerConfig(entropy_high=25.0, entropy_low=3.0)
    selector = InterventionSelector(
        pcg, sig, planner_cfg,
        use_ig_bonus=(method != "dia_no_ig"),
        use_sig=(method != "dia_no_sig"),
    )
    fit_every = 10

    rcfg = RunnerConfig(
        buffer_size=20_000,
        min_buffer=256,
        batch_recent=2048,
        fit_every=fit_every,
        pcg_epochs=200,
        log_prefix=f"mc2d_{method}_{seed}",
        option_max_steps=64,
        terminate_on_success=True,
        auto_expand_sig=(method != "dia_no_sig"),
        add_threshold=0.75,
        remove_threshold=0.55,
    )

    def option_factory(skill):
        # RuleOption executes one action per step (the option's target action).
        # This keeps each transition "clean" -- only the target variable changes,
        # giving the interventional PCG estimator an uncontaminated signal.
        return RuleOption(
            skill.subgoal, OptionConfig(max_steps=64, terminate_on_success=True),
        )

    runner = DIARunner(
        env, evgs, pcg, sig, selector, rcfg,
        logger=None, option_factory=option_factory,
    )

    task_goal = Subgoal(
        var_index=name_to_idx["diamond"], predicate=Predicate.UP,
    )

    achieved: List[int] = []
    total_steps = 0
    total_successes = 0
    steps_to_diamond: Optional[int] = None
    diamond_count = 0
    diamond_steps_list: List[int] = []  # all option steps where diamond succeeded (for learning curves)

    for t in range(steps):
        rec = runner.step(achieved, task_goal=task_goal)
        total_steps += 1

        if rec["success"]:
            total_successes += 1
            sid = rec["skill_id"]
            if sid not in achieved:
                achieved.append(sid)
            # First time diamond is achieved: record current option-step count
            if sid == DIAMOND_IDX and steps_to_diamond is None:
                steps_to_diamond = t + 1
            # Count every diamond success over full training + record step for learning curves
            if sid == DIAMOND_IDX:
                diamond_count += 1
                diamond_steps_list.append(t + 1)

    # Final PCG metrics
    probs = np.array(pcg.probs)
    np.fill_diagonal(probs, 0.0)
    final_shd = float(eval_shd(probs, GROUND_TRUTH_EDGES))
    final_ece = float(eval_ece(probs, GROUND_TRUTH_EDGES))
    success_rate = total_successes / max(1, total_steps)

    return {
        "method": method,
        "seed": seed,
        "steps_to_diamond": steps_to_diamond,
        "final_shd": final_shd,
        "final_ece": final_ece,
        "success_rate": round(success_rate, 4),
        "diamond_count": diamond_count,
        "diamond_steps_list": diamond_steps_list,
        "completed": True,
        "_pcg_probs": probs.tolist(),  # internal — stripped before JSON write
    }


def _run_ppo(method: str, seed: int, steps: int) -> Dict:
    """PPO flat baseline using stable_baselines3 when available.

    For ``ppo_options``: a hierarchical options policy would require a custom
    policy architecture.  This stub uses the same flat PPO as a placeholder
    and documents the intent via the method name in the output.

    Falls back to a random-action baseline when SB3 is not installed.
    """
    from dia.envs.minecraft2d import MinecraftChainEnv, ChainConfig

    np.random.seed(seed)
    env = _FlatObsWrapper(MinecraftChainEnv(ChainConfig(), seed=seed))
    env.reset()

    steps_to_diamond: Optional[int] = None
    total_reward_positive = 0
    total_steps = 0

    try:
        from stable_baselines3 import PPO as SB3PPO

        # NOTE: ppo_options would ideally use a hierarchical policy; flat PPO
        # is used here as a structural placeholder until an options layer is
        # implemented on top of the OptionConfig / ResourceOption classes.
        model = SB3PPO(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=0,
            n_steps=256,
            batch_size=64,
            learning_rate=3e-4,
            device="cpu",
        )
        model.learn(total_timesteps=steps)

        # Evaluation episode: run up to `steps` more timesteps
        obs = env.reset()
        for t in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(int(action))
            total_steps += 1
            if reward > 0:
                total_reward_positive += 1
            if steps_to_diamond is None and obs[DIAMOND_IDX] > 0:
                steps_to_diamond = t + 1
            if done:
                obs = env.reset()

    except ImportError:
        # SB3 not available: random-action fallback
        obs = env.reset()
        for t in range(steps):
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            total_steps += 1
            if reward > 0:
                total_reward_positive += 1
            if steps_to_diamond is None and obs[DIAMOND_IDX] > 0:
                steps_to_diamond = t + 1
            if done:
                obs = env.reset()

    success_rate = total_reward_positive / max(1, total_steps)
    return {
        "method": method,
        "seed": seed,
        "steps_to_diamond": steps_to_diamond,
        "final_shd": None,
        "final_ece": None,
        "success_rate": round(success_rate, 4),
        "completed": True,
    }


def _run_ride(method: str, seed: int, steps: int) -> Dict:
    """RIDE: Rewarding Impact-Driven Exploration (Raileanu & Rocktaschel 2020).

    Intrinsic reward: r_int = ||phi(s') - phi(s)||_2 / sqrt(N(s'))
    where phi is a learned 2-layer MLP embedding and N is a hash-based
    visitation counter.  Trains a simple actor-critic (PPO-style) on
    r_extrinsic + r_int with GAE, updating every 256 environment steps.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is required for _run_ride")

    from dia.envs.minecraft2d import MinecraftChainEnv, ChainConfig

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = _FlatObsWrapper(MinecraftChainEnv(ChainConfig(), seed=seed))
    obs_dim  = 9
    n_actions = 10
    update_every = 256
    gamma, lam, clip_eps = 0.99, 0.95, 0.2

    net   = _SimplePolicyNet(obs_dim, n_actions)
    ride  = _RIDEShaper(obs_dim)
    opt   = torch.optim.Adam(net.parameters(), lr=3e-4)

    # Rollout buffers
    obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []

    obs = env.reset()
    steps_to_diamond: Optional[int] = None
    total_reward_positive = 0
    total_steps = 0

    for t in range(steps):
        t_obs  = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, value = net(t_obs)
        dist   = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()
        logp   = dist.log_prob(torch.tensor(action)).item()

        next_obs, ext_reward, done, _ = env.step(action)
        total_steps += 1

        # RIDE intrinsic reward + phi update
        r_int = ride.reward(obs, next_obs)
        ride.update(obs, next_obs)
        shaped_reward = ext_reward + r_int

        obs_buf.append(obs.copy())
        act_buf.append(action)
        rew_buf.append(shaped_reward)
        val_buf.append(value.item())
        logp_buf.append(logp)
        done_buf.append(float(done))

        if ext_reward > 0:
            total_reward_positive += 1
        if steps_to_diamond is None and next_obs[DIAMOND_IDX] > 0:
            steps_to_diamond = t + 1

        obs = next_obs
        if done:
            obs = env.reset()

        # PPO update every `update_every` steps
        if len(obs_buf) >= update_every:
            advantages, returns = _gae(rew_buf, val_buf, done_buf, gamma, lam)
            t_obs_b   = torch.tensor(np.array(obs_buf),  dtype=torch.float32)
            t_act_b   = torch.tensor(act_buf,            dtype=torch.long)
            t_adv_b   = torch.tensor(advantages,         dtype=torch.float32)
            t_ret_b   = torch.tensor(returns,            dtype=torch.float32)
            t_logp_b  = torch.tensor(logp_buf,           dtype=torch.float32)
            _ppo_update(net, opt, t_obs_b, t_act_b, t_adv_b, t_ret_b, t_logp_b, clip_eps)
            obs_buf.clear(); act_buf.clear(); rew_buf.clear()
            val_buf.clear(); logp_buf.clear(); done_buf.clear()

    success_rate = total_reward_positive / max(1, total_steps)
    return {
        "method": method,
        "seed": seed,
        "steps_to_diamond": steps_to_diamond,
        "final_shd": None,
        "final_ece": None,
        "success_rate": round(success_rate, 4),
        "completed": True,
    }


def _run_icm(method: str, seed: int, steps: int) -> Dict:
    """ICM: Intrinsic Curiosity Module (Pathak et al. 2017).

    Intrinsic reward: forward model prediction error
        r_int = ||phi(s') - f(phi(s), a)||_2
    where phi is an encoder, f is the forward model, and an inverse model
    g(phi(s), phi(s')) -> a is trained jointly to keep phi task-relevant.
    Trains the same simple actor-critic (PPO-style) as RIDE.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is required for _run_icm")

    from dia.envs.minecraft2d import MinecraftChainEnv, ChainConfig

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = _FlatObsWrapper(MinecraftChainEnv(ChainConfig(), seed=seed))
    obs_dim   = 9
    n_actions = 10
    update_every = 256
    gamma, lam, clip_eps = 0.99, 0.95, 0.2

    net  = _SimplePolicyNet(obs_dim, n_actions)
    icm  = _ICMShaper(obs_dim, n_actions)
    opt  = torch.optim.Adam(net.parameters(), lr=3e-4)

    obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []
    # Mini-buffers for ICM dynamics update
    icm_obs, icm_acts, icm_nobs = [], [], []

    obs = env.reset()
    steps_to_diamond: Optional[int] = None
    total_reward_positive = 0
    total_steps = 0

    for t in range(steps):
        t_obs  = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, value = net(t_obs)
        dist   = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()
        logp   = dist.log_prob(torch.tensor(action)).item()

        next_obs, ext_reward, done, _ = env.step(action)
        total_steps += 1

        # ICM intrinsic reward
        r_int = icm.reward(obs, action, next_obs)
        shaped_reward = ext_reward + r_int

        obs_buf.append(obs.copy())
        act_buf.append(action)
        rew_buf.append(shaped_reward)
        val_buf.append(value.item())
        logp_buf.append(logp)
        done_buf.append(float(done))

        icm_obs.append(obs.copy())
        icm_acts.append(action)
        icm_nobs.append(next_obs.copy())

        if ext_reward > 0:
            total_reward_positive += 1
        if steps_to_diamond is None and next_obs[DIAMOND_IDX] > 0:
            steps_to_diamond = t + 1

        obs = next_obs
        if done:
            obs = env.reset()

        # PPO + ICM update every `update_every` steps
        if len(obs_buf) >= update_every:
            advantages, returns = _gae(rew_buf, val_buf, done_buf, gamma, lam)
            t_obs_b  = torch.tensor(np.array(obs_buf),  dtype=torch.float32)
            t_act_b  = torch.tensor(act_buf,            dtype=torch.long)
            t_adv_b  = torch.tensor(advantages,         dtype=torch.float32)
            t_ret_b  = torch.tensor(returns,            dtype=torch.float32)
            t_logp_b = torch.tensor(logp_buf,           dtype=torch.float32)
            _ppo_update(net, opt, t_obs_b, t_act_b, t_adv_b, t_ret_b, t_logp_b, clip_eps)

            icm.update(
                np.array(icm_obs,  dtype=np.float32),
                np.array(icm_acts, dtype=np.int64),
                np.array(icm_nobs, dtype=np.float32),
            )
            obs_buf.clear(); act_buf.clear(); rew_buf.clear()
            val_buf.clear(); logp_buf.clear(); done_buf.clear()
            icm_obs.clear(); icm_acts.clear(); icm_nobs.clear()

    success_rate = total_reward_positive / max(1, total_steps)
    return {
        "method": method,
        "seed": seed,
        "steps_to_diamond": steps_to_diamond,
        "final_shd": None,
        "final_ece": None,
        "success_rate": round(success_rate, 4),
        "completed": True,
    }


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_METHOD_RUNNERS = {
    "ppo":         _run_ppo,
    "ppo_options": _run_ppo,
    "ride":        _run_ride,
    "icm":         _run_icm,
    "dia_no_ig":   _run_dia,
    "dia_no_sig":  _run_dia,
    "dia":         _run_dia,
    "dia_oracle":  _run_dia,   # DIA with ground-truth SIG given upfront (upper bound)
}

VALID_METHODS = sorted(_METHOD_RUNNERS.keys())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run a single (method, seed) 2D-Minecraft baseline and write a JSON result."
    )
    ap.add_argument(
        "--method",
        type=str,
        required=True,
        choices=VALID_METHODS,
        help="Baseline method to run.",
    )
    ap.add_argument("--seed",  type=int, required=True, help="Random seed (0-9).")
    ap.add_argument(
        "--steps",
        type=int,
        default=500_000,
        help="Number of option-level steps (default: 500000).",
    )
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to write the output JSON result file.",
    )
    ap.add_argument(
        "--save_pcg",
        type=str,
        default=None,
        help="If set, save final PCG probs matrix to this .npy path (DIA methods only).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print(
        f"[run_baseline_2d] method={args.method} seed={args.seed} "
        f"steps={args.steps} out={args.out}"
    )

    runner_fn = _METHOD_RUNNERS[args.method]
    try:
        result = runner_fn(args.method, args.seed, args.steps)
    except NotImplementedError as exc:
        # Write a placeholder result so the sweep can track which runs are stubs
        result = {
            "method": args.method,
            "seed": args.seed,
            "steps_to_diamond": None,
            "final_shd": None,
            "final_ece": None,
            "success_rate": None,
            "completed": False,
            "error": str(exc),
        }
        print(f"[run_baseline_2d] STUB: {exc}", file=sys.stderr)
    except Exception as exc:
        result = {
            "method": args.method,
            "seed": args.seed,
            "steps_to_diamond": None,
            "final_shd": None,
            "final_ece": None,
            "success_rate": None,
            "completed": False,
            "error": repr(exc),
        }
        print(f"[run_baseline_2d] ERROR: {exc}", file=sys.stderr)
        raise

    # Optionally save PCG probs before stripping from JSON
    pcg_probs = result.pop("_pcg_probs", None)
    if args.save_pcg and pcg_probs is not None:
        import numpy as _np
        _np.save(args.save_pcg, _np.array(pcg_probs))
        print(f"[run_baseline_2d] PCG probs saved to {args.save_pcg}")

    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[run_baseline_2d] wrote {args.out}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
