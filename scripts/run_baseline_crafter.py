#!/usr/bin/env python3
"""
scripts/run_baseline_crafter.py — Run a single (method, seed) baseline on Crafter.

Supported methods:
  ppo, ride, icm, dia_no_ig, dia_no_sig, dia, dia_oracle

Output JSON schema (same as run_baseline_2d.py):
  steps_to_diamond, final_shd, final_ece, success_rate,
  diamond_count, diamond_steps_list, completed

Usage:
  python scripts/run_baseline_crafter.py --method dia --seed 0 --steps 200000 \
      --out results/logs/crafter_dia_seed0.json
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
# Constants
# ---------------------------------------------------------------------------

# 15-var inventory state vector (matches evgs_crafter.py VAR_NAMES order)
_INV_KEYS = [
    "wood", "stone", "coal", "iron", "diamond", "sapling",
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
    "health", "food", "drink",
]
OBS_DIM   = 15
N_ACTIONS = 17
DIAMOND_IDX = 4   # index of "diamond" in VAR_NAMES

# Ground-truth causal DAG (16 edges)
GROUND_TRUTH_EDGES: List[Tuple[int, int]] = [
    (0, 6), (0, 9), (0, 7), (1, 7), (0, 8), (3, 8),
    (0, 10), (1, 10), (0, 11), (3, 11),
    (7, 1), (7, 2), (7, 3), (8, 4),
    (13, 12), (14, 12),
]

PCG_METHODS = {"dia", "dia_no_ig", "dia_no_sig", "dia_oracle"}


def _info_to_flat(info: dict) -> np.ndarray:
    inv = info.get("inventory", {})
    return np.array([float(inv.get(k, 0.0)) for k in _INV_KEYS], dtype=np.float32)


# ---------------------------------------------------------------------------
# Shared neural modules (PPO actor-critic, RIDE, ICM)
# ---------------------------------------------------------------------------

class _Net(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, obs_dim=OBS_DIM, n_act=N_ACTIONS):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch required")
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(),
                                   nn.Linear(64, 64), nn.ReLU())
        self.pi = nn.Linear(64, n_act)
        self.v  = nn.Linear(64, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.pi(h), self.v(h).squeeze(-1)


def _gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    g = 0.0
    for t in reversed(range(T)):
        nv = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * nv * (1 - dones[t]) - values[t]
        g = delta + gamma * lam * (1 - dones[t]) * g
        adv[t] = g
    return adv, adv + np.array(values[:T], dtype=np.float32)


def _ppo_update(net, opt, obs_b, act_b, adv_b, ret_b, logp_b, clip=0.2, epochs=4):
    adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
    for _ in range(epochs):
        logits, vals = net(obs_b)
        d = torch.distributions.Categorical(logits=logits)
        ratio = torch.exp(d.log_prob(act_b) - logp_b)
        loss = (-torch.min(ratio * adv_b, torch.clamp(ratio, 1 - clip, 1 + clip) * adv_b).mean()
                + 0.5 * F.mse_loss(vals, ret_b)
                - 0.01 * d.entropy().mean())
        opt.zero_grad(); loss.backward(); opt.step()


class _RIDE(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, obs_dim=OBS_DIM, lr=3e-4):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch required")
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, 32))
        self.counts: dict = {}
        self.opt = torch.optim.Adam(self.phi.parameters(), lr=lr)

    def _t(self, x): return torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    def _key(self, x): return tuple(x.astype(int).tolist())

    def reward(self, s, s2):
        k = self._key(s2)
        self.counts[k] = self.counts.get(k, 0) + 1
        with torch.no_grad():
            d = torch.norm(self.phi(self._t(s2)) - self.phi(self._t(s)), dim=-1).item()
        return d / (self.counts[k] ** 0.5 + 1e-8)

    def update(self, s, s2):
        loss = F.mse_loss(self.phi(self._t(s2)), self.phi(self._t(s)).detach())
        self.opt.zero_grad(); loss.backward(); self.opt.step()


class _ICM(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, obs_dim=OBS_DIM, n_act=N_ACTIONS, lr=3e-4):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch required")
        super().__init__()
        self.n_act = n_act
        self.phi = nn.Sequential(nn.Linear(obs_dim, 32), nn.ReLU())
        self.fwd = nn.Linear(32 + n_act, 32)
        self.inv = nn.Linear(64, n_act)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def _t(self, x): return torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    def _oh(self, a):
        oh = torch.zeros(1, self.n_act); oh[0, a] = 1.0; return oh

    def reward(self, s, a, s2):
        with torch.no_grad():
            e = self.phi(self._t(s))
            e2 = self.phi(self._t(s2))
            eh = self.fwd(torch.cat([e, self._oh(a)], -1))
        return F.mse_loss(eh, e2).item()

    def update(self, obs, acts, nobs):
        to = torch.tensor(obs, dtype=torch.float32)
        tn = torch.tensor(nobs, dtype=torch.float32)
        ta = torch.tensor(acts, dtype=torch.long)
        e = self.phi(to); en = self.phi(tn)
        oh = torch.zeros(len(ta), self.n_act).scatter_(1, ta.unsqueeze(1), 1.0)
        fl = F.mse_loss(self.fwd(torch.cat([e, oh], -1)), en.detach())
        il = F.cross_entropy(self.inv(torch.cat([e, en], -1)), ta)
        loss = 0.8 * fl + 0.2 * il
        self.opt.zero_grad(); loss.backward(); self.opt.step()


# ---------------------------------------------------------------------------
# DIA imports and SIG builder
# ---------------------------------------------------------------------------

def _import_dia():
    from dia.evgs_crafter import make_crafter_evgs
    from dia.evgs_adapters import InfoObsWrapper
    from dia.sig import SIGraph, Skill
    from dia.types import Subgoal, Predicate
    from dia.planner import PlannerConfig, InterventionSelector
    from dia.rollout import DIARunner, RunnerConfig
    from dia.pcg_learner import DifferentiablePCG, DifferentiablePCGConfig
    from dia.options import OptionConfig, RandomOption
    from dia.eval.pcg_metrics import shd as eval_shd, ece as eval_ece
    return locals()


# Oracle SIG edges (name pairs)
_ORACLE_EDGES = [
    ("wood", "wood_pickaxe"), ("wood", "wood_sword"),
    ("wood", "stone_pickaxe"), ("stone", "stone_pickaxe"),
    ("wood", "iron_pickaxe"), ("iron", "iron_pickaxe"),
    ("wood", "stone_sword"), ("stone", "stone_sword"),
    ("wood", "iron_sword"), ("iron", "iron_sword"),
    ("stone_pickaxe", "stone"), ("stone_pickaxe", "coal"),
    ("stone_pickaxe", "iron"), ("iron_pickaxe", "diamond"),
    ("food", "health"), ("drink", "health"),
]


def _build_sig(d, var_names, oracle=False):
    idx = {n: i for i, n in enumerate(var_names)}
    sig = d["SIGraph"]()
    for name in var_names:
        sid = idx[name]
        sig.add_skill(d["Skill"](
            skill_id=sid,
            subgoal=d["Subgoal"](var_index=sid, predicate=d["Predicate"].UP),
            name=f"{name}↑",
        ))
    if oracle:
        for src, dst in _ORACLE_EDGES:
            if src in idx and dst in idx:
                sig.add_prerequisite(idx[src], idx[dst])
    return sig


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------

def _run_dia(method: str, seed: int, steps: int) -> Dict:
    import crafter
    d = _import_dia()
    np.random.seed(seed)

    env  = d["InfoObsWrapper"](crafter.Env(seed=seed))
    evgs = d["make_crafter_evgs"]()
    var_names = evgs.names()
    M = len(var_names)
    name_to_idx = {n: i for i, n in enumerate(var_names)}

    pcg = d["DifferentiablePCG"](d["DifferentiablePCGConfig"](
        num_vars=M, max_iter=200, lr=5e-3, verbose=False,
    ))
    sig = _build_sig(d, var_names, oracle=(method == "dia_oracle"))

    selector = d["InterventionSelector"](
        pcg, sig, d["PlannerConfig"](entropy_high=25.0, entropy_low=3.0),
        use_ig_bonus=(method != "dia_no_ig"),
        use_sig=(method != "dia_no_sig"),
    )

    rcfg = d["RunnerConfig"](
        buffer_size=20_000, min_buffer=256, batch_recent=2048, fit_every=10,
        pcg_epochs=200, log_prefix=f"crafter_{method}_{seed}",
        option_max_steps=200, terminate_on_success=True,
        auto_expand_sig=(method != "dia_no_sig"),
        add_threshold=0.75, remove_threshold=0.55,
    )

    def option_factory(skill):
        return d["RandomOption"](
            subgoal=skill.subgoal,
            cfg=d["OptionConfig"](max_steps=200, terminate_on_success=True),
            action_space=env.action_space,
        )

    runner = d["DIARunner"](env, evgs, pcg, sig, selector, rcfg,
                            logger=None, option_factory=option_factory)
    task_goal = d["Subgoal"](var_index=name_to_idx["diamond"], predicate=d["Predicate"].UP)

    achieved: List[int] = []
    total_successes = 0
    steps_to_diamond: Optional[int] = None
    diamond_count = 0
    diamond_steps_list: List[int] = []

    for t in range(steps):
        rec = runner.step(achieved, task_goal=task_goal)
        if rec["success"]:
            total_successes += 1
            sid = rec["skill_id"]
            if sid not in achieved:
                achieved.append(sid)
            if sid == DIAMOND_IDX:
                if steps_to_diamond is None:
                    steps_to_diamond = t + 1
                diamond_count += 1
                diamond_steps_list.append(t + 1)

    probs = np.array(pcg.probs)
    np.fill_diagonal(probs, 0.0)
    return {
        "method": method, "seed": seed,
        "steps_to_diamond": steps_to_diamond,
        "final_shd": float(d["eval_shd"](probs, GROUND_TRUTH_EDGES)),
        "final_ece": float(d["eval_ece"](probs, GROUND_TRUTH_EDGES)),
        "success_rate": round(total_successes / max(1, steps), 4),
        "diamond_count": diamond_count,
        "diamond_steps_list": diamond_steps_list,
        "completed": True,
    }


def _flat_loop(method, seed, steps, intrinsic_cls=None):
    """Shared flat-env loop for ppo/ride/icm."""
    import crafter
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)

    raw_env = crafter.Env(seed=seed)
    raw_env.reset()
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    net = _Net() if _TORCH_AVAILABLE else None
    shaper = intrinsic_cls() if (intrinsic_cls and _TORCH_AVAILABLE) else None
    opt = torch.optim.Adam(net.parameters(), lr=3e-4) if net else None

    obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []
    icm_obs, icm_acts, icm_nobs = [], [], []
    update_every = 256

    steps_to_diamond: Optional[int] = None
    diamond_count = 0
    diamond_steps_list: List[int] = []
    total_pos_rew = 0

    for t in range(steps):
        if net:
            t_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, value = net(t_obs)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()
            logp = dist.log_prob(torch.tensor(action)).item()
        else:
            action = np.random.randint(N_ACTIONS)
            value = logp = 0.0

        _, ext_rew, done, info = raw_env.step(action)
        nobs = _info_to_flat(info)

        r_int = 0.0
        if shaper:
            if isinstance(shaper, _RIDE):
                r_int = shaper.reward(obs, nobs)
                shaper.update(obs, nobs)
            else:  # _ICM
                r_int = shaper.reward(obs, action, nobs)
                icm_obs.append(obs.copy())
                icm_acts.append(action)
                icm_nobs.append(nobs.copy())

        if net:
            obs_buf.append(obs.copy()); act_buf.append(action)
            rew_buf.append(ext_rew + r_int)
            val_buf.append(value.item() if hasattr(value, "item") else float(value))
            logp_buf.append(logp); done_buf.append(float(done))

        if ext_rew > 0:
            total_pos_rew += 1
        if nobs[DIAMOND_IDX] > obs[DIAMOND_IDX]:
            if steps_to_diamond is None:
                steps_to_diamond = t + 1
            diamond_count += 1
            diamond_steps_list.append(t + 1)

        obs = nobs
        if done:
            raw_env.reset()
            obs = np.zeros(OBS_DIM, dtype=np.float32)

        if net and len(obs_buf) >= update_every:
            adv, ret = _gae(rew_buf, val_buf, done_buf)
            _ppo_update(net, opt,
                        torch.tensor(np.array(obs_buf),  dtype=torch.float32),
                        torch.tensor(act_buf,             dtype=torch.long),
                        torch.tensor(adv,                 dtype=torch.float32),
                        torch.tensor(ret,                 dtype=torch.float32),
                        torch.tensor(logp_buf,            dtype=torch.float32))
            if isinstance(shaper, _ICM) and icm_obs:
                shaper.update(np.array(icm_obs, dtype=np.float32),
                              np.array(icm_acts, dtype=np.int64),
                              np.array(icm_nobs, dtype=np.float32))
                icm_obs.clear(); icm_acts.clear(); icm_nobs.clear()
            obs_buf.clear(); act_buf.clear(); rew_buf.clear()
            val_buf.clear(); logp_buf.clear(); done_buf.clear()

    return {
        "method": method, "seed": seed,
        "steps_to_diamond": steps_to_diamond,
        "final_shd": None, "final_ece": None,
        "success_rate": round(total_pos_rew / max(1, steps), 4),
        "diamond_count": diamond_count,
        "diamond_steps_list": diamond_steps_list,
        "completed": True,
    }


def _run_ppo(method, seed, steps):
    return _flat_loop(method, seed, steps, intrinsic_cls=None)

def _run_ride(method, seed, steps):
    return _flat_loop(method, seed, steps, intrinsic_cls=_RIDE)

def _run_icm(method, seed, steps):
    return _flat_loop(method, seed, steps, intrinsic_cls=_ICM)


# ---------------------------------------------------------------------------
# Dispatch / CLI
# ---------------------------------------------------------------------------

_METHOD_RUNNERS = {
    "ppo":        _run_ppo,
    "ride":       _run_ride,
    "icm":        _run_icm,
    "dia_no_ig":  _run_dia,
    "dia_no_sig": _run_dia,
    "dia":        _run_dia,
    "dia_oracle": _run_dia,
}

VALID_METHODS = sorted(_METHOD_RUNNERS.keys())


def parse_args():
    ap = argparse.ArgumentParser(
        description="Run a single (method, seed) Crafter baseline."
    )
    ap.add_argument("--method", required=True, choices=VALID_METHODS)
    ap.add_argument("--seed",   type=int, required=True)
    ap.add_argument("--steps",  type=int, default=200_000)
    ap.add_argument("--out",    required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"[run_baseline_crafter] method={args.method} seed={args.seed} "
          f"steps={args.steps} out={args.out}")
    try:
        result = _METHOD_RUNNERS[args.method](args.method, args.seed, args.steps)
    except Exception as exc:
        result = {
            "method": args.method, "seed": args.seed,
            "steps_to_diamond": None, "final_shd": None, "final_ece": None,
            "success_rate": None, "diamond_count": 0, "diamond_steps_list": [],
            "completed": False, "error": repr(exc),
        }
        print(f"[run_baseline_crafter] ERROR: {exc}", file=sys.stderr)
        raise
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[run_baseline_crafter] wrote {args.out}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
