#!/usr/bin/env python3
# scripts/run_carl_baseline.py
"""
SAC baseline for the DIA CausalWorld T0/T1/T2 comparison.

CARL (Causal Augmented RL, Ahmed et al. NeurIPS 2020) is NOT shipped as a
standalone importable module inside the causal_world pip package -- the paper
described the benchmark framework, but CARL's RL training code was never merged
into the public package.  The closest sanctioned comparison from the CausalWorld
paper is a standard RL agent (SAC) trained from scratch (or from a checkpoint)
under the intervention condition.  This is the relevant ablation for DIA's claim:
DIA recovers in fewer environment steps by detecting and isolating the structural
change, whereas a non-causal agent must retrain from its current policy.

This script implements a minimal SAC (Soft Actor-Critic) loop using only PyTorch
and NumPy to avoid SB3 version/numpy ABI issues in the dia conda environment.

Usage
-----
  conda run -n dia python scripts/run_carl_baseline.py \
      --condition T0 --seed 0 --steps 500 --out /tmp/carl_T0.json

Output JSON
-----------
  {
    "method": "sac_baseline",
    "carl_available": false,
    "condition": "T0",
    "seed": 0,
    "total_steps": 500,
    "episodes": <int>,
    "episode_returns": [<float>, ...],
    "episode_successes": [<bool>, ...],
    "steps_to_first_success": <int|null>,
    "final_success_rate": <float>,
    "mean_return": <float>
  }
"""
from __future__ import annotations

# Path guard: the dia conda env has mixed numpy 1.23/2.x dist-info alongside a
# ~/.local installation of numpy 2.x.  Move ~/.local paths to the end so the
# conda env's numpy (1.23.5) is imported first, preventing ABI mismatches with
# CausalWorld's compiled pybullet extensions and internal numpy calls.
import sys as _sys
_sys.path = (
    [p for p in _sys.path if ".local" not in p]
    + [p for p in _sys.path if ".local" in p]
)

import argparse
import collections
import json
import logging
import math
import os
import random
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intervention conditions -- must match train_causalworld_dia.py exactly
# ---------------------------------------------------------------------------
INTERVENTIONS: Dict[str, Dict[str, Any]] = {
    "T0": {"obstacle": {"size": np.array([0.5, 0.015, 0.02])}},
    "T1": {"obstacle": {"size": np.array([0.5, 0.015, 0.10])}},
    "T2": {"tool_block": {"size": np.array([0.085, 0.085, 0.085])}},
}


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Simple uniform-sampling replay buffer backed by numpy arrays."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.ptr % self.capacity
        self.obs[idx] = obs.astype(np.float32)
        self.actions[idx] = action.astype(np.float32)
        self.rewards[idx] = float(reward)
        self.next_obs[idx] = next_obs.astype(np.float32)
        self.dones[idx] = float(done)
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.actions[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.next_obs[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.dones[idx], dtype=torch.float32, device=device),
        )


# ---------------------------------------------------------------------------
# SAC networks
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, out_dim: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class _QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = _mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1))


class _Actor(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (tanh-squashed action, log_prob)."""
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        raw = mu + eps * std
        action = torch.tanh(raw)
        # Log-prob with tanh correction
        log_prob = (
            -0.5 * ((raw - mu) / (std + 1e-8)) ** 2
            - log_std
            - math.log(math.sqrt(2 * math.pi))
        ).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return action, log_prob


# ---------------------------------------------------------------------------
# SAC agent
# ---------------------------------------------------------------------------

class SACAgent:
    """
    Minimal Soft Actor-Critic (Haarnoja et al. 2018).

    Hyper-parameters are deliberately set close to CausalWorld paper defaults.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        hidden: int = 256,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha
        self.act_dim = act_dim

        self.actor = _Actor(obs_dim, act_dim, hidden).to(device)
        self.q1 = _QNet(obs_dim, act_dim, hidden).to(device)
        self.q2 = _QNet(obs_dim, act_dim, hidden).to(device)
        self.q1_target = _QNet(obs_dim, act_dim, hidden).to(device)
        self.q2_target = _QNet(obs_dim, act_dim, hidden).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=lr)

        if auto_alpha:
            self.target_entropy = -float(act_dim)
            self.log_alpha = torch.tensor(
                math.log(alpha), dtype=torch.float32, device=device, requires_grad=True
            )
            self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.actor.sample(obs_t)
        # Explicitly convert to a fresh numpy 1.x float64 array so CausalWorld's
        # pybullet-based np.clip() call doesn't hit numpy 2.x ABI paths.
        return np.array(action.squeeze(0).cpu().tolist(), dtype=np.float64)

    def update(self, buffer: ReplayBuffer, batch_size: int = 256) -> None:
        if buffer.size < batch_size:
            return

        obs, act, rew, next_obs, done = buffer.sample(batch_size, self.device)

        with torch.no_grad():
            next_act, next_log_prob = self.actor.sample(next_obs)
            q1_next = self.q1_target(next_obs, next_act)
            q2_next = self.q2_target(next_obs, next_act)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            q_target = rew + self.gamma * (1.0 - done) * q_next

        # Critic update
        q1_loss = F.mse_loss(self.q1(obs, act), q_target)
        q2_loss = F.mse_loss(self.q2(obs, act), q_target)
        self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()

        # Actor update
        new_act, log_prob = self.actor.sample(obs)
        q1_val = self.q1(obs, new_act)
        q2_val = self.q2(obs, new_act)
        actor_loss = (self.alpha * log_prob - torch.min(q1_val, q2_val)).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        # Entropy temperature update
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()

        # Soft target update
        for p, pt in zip(self.q1.parameters(), self.q1_target.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
        for p, pt in zip(self.q2.parameters(), self.q2_target.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _make_env(condition: str, seed: int, max_episode_length: Optional[int] = None):
    """Return a raw CausalWorld pick_and_place env with intervention applied.

    max_episode_length controls the episode horizon passed to CausalWorld.
    The default (None) uses CausalWorld's internal formula:
      rigid_objects * 10 / dt  =  2 * 10 / 0.01  =  2000 steps per episode.
    For short smoke tests pass a smaller value (e.g. 200) via --max_ep_len.
    """
    from causal_world.envs.causalworld import CausalWorld
    from causal_world.task_generators import generate_task

    task = generate_task(task_generator_id="pick_and_place")
    env = CausalWorld(
        task=task,
        enable_visualization=False,
        seed=seed,
        normalize_observations=True,
        normalize_actions=True,
        max_episode_length=max_episode_length,
    )
    if condition in INTERVENTIONS:
        success, _ = env.do_intervention(INTERVENTIONS[condition])
        logger.info(
            "Applied intervention condition=%s: %s (success=%s)",
            condition, INTERVENTIONS[condition], success,
        )
    return env


# ---------------------------------------------------------------------------
# SAC training loop
# ---------------------------------------------------------------------------

def run_sac(
    condition: str,
    seed: int,
    total_steps: int,
    learning_starts: int = 500,
    batch_size: int = 256,
    buffer_size: int = 50_000,
    update_freq: int = 1,
    log_every: int = 500,
    max_episode_length: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Train SAC from scratch on the pick_and_place task under `condition`.

    Returns a dict of metrics compatible with the DIA comparison format.
    """
    # Fix seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cpu")  # CausalWorld pybullet is CPU-bound anyway

    env = _make_env(condition=condition, seed=seed, max_episode_length=max_episode_length)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    logger.info("obs_dim=%d  act_dim=%d", obs_dim, act_dim)

    # Adjust learning_starts to not exceed total_steps
    learning_starts = min(learning_starts, total_steps // 4)

    agent = SACAgent(obs_dim=obs_dim, act_dim=act_dim, device=device)
    buffer = ReplayBuffer(capacity=buffer_size, obs_dim=obs_dim, act_dim=act_dim)

    episode_returns: List[float] = []
    episode_successes: List[bool] = []
    ep_return: float = 0.0
    ep_len: int = 0
    steps_to_first_success: Optional[int] = None

    obs = np.array(env.reset(), dtype=np.float32)
    t0 = time.time()

    for step in range(total_steps):
        if step < learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        next_obs_raw, reward, done, info = env.step(action)
        next_obs = np.array(next_obs_raw, dtype=np.float32)
        buffer.add(obs, action, reward, next_obs, done)
        ep_return += float(reward)
        ep_len += 1

        if done:
            # Parse success signal from CausalWorld info dict
            success = bool(
                info.get("task_solved", False)
                or info.get("fractional_success", 0.0) >= 0.9
            )
            episode_returns.append(ep_return)
            episode_successes.append(success)
            if success and steps_to_first_success is None:
                steps_to_first_success = step + 1
            ep_return = 0.0
            ep_len = 0
            obs = np.array(env.reset(), dtype=np.float32)
        else:
            obs = next_obs

        if step >= learning_starts and step % update_freq == 0:
            agent.update(buffer, batch_size=batch_size)

        if (step + 1) % log_every == 0:
            recent_sr = (
                float(np.mean(episode_successes[-20:])) if episode_successes else 0.0
            )
            recent_ret = (
                float(np.mean(episode_returns[-20:])) if episode_returns else 0.0
            )
            logger.info(
                "step=%d/%d | episodes=%d | recent_sr=%.3f | recent_ret=%.3f | alpha=%.4f",
                step + 1, total_steps, len(episode_returns), recent_sr, recent_ret, agent.alpha,
            )

    elapsed = time.time() - t0

    try:
        env.close()
    except Exception:
        pass

    final_success_rate = (
        float(np.mean(episode_successes[-20:])) if episode_successes else 0.0
    )
    mean_return = float(np.mean(episode_returns)) if episode_returns else 0.0

    logger.info(
        "SAC done in %.1fs | episodes=%d | final_success_rate=%.3f | mean_return=%.3f",
        elapsed, len(episode_returns), final_success_rate, mean_return,
    )

    return {
        "method": "sac_baseline",
        # CARL is not available as an importable module in the causal_world pip
        # package (Ahmed et al. NeurIPS 2020 released the benchmark env but not a
        # ready-to-import CARL training loop).  SAC from scratch under the same
        # intervention condition is the equivalent comparison: it measures how many
        # environment steps a non-causal agent needs to solve the task after the
        # change is applied, directly matching DIA's "recovery steps" metric.
        "carl_available": False,
        "carl_note": (
            "CARL is not shipped as an importable module in causal_world. "
            "SAC-from-scratch is the standard non-causal RL comparison used "
            "in the CausalWorld benchmark paper (Table 2, Ahmed et al. 2020)."
        ),
        "condition": condition,
        "seed": seed,
        "total_steps": total_steps,
        "episodes": len(episode_returns),
        "episode_returns": [round(float(r), 4) for r in episode_returns],
        "episode_successes": [bool(s) for s in episode_successes],
        "steps_to_first_success": steps_to_first_success,
        "final_success_rate": round(final_success_rate, 4),
        "mean_return": round(mean_return, 4),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="SAC baseline for DIA CausalWorld T0/T1/T2 comparison."
    )
    ap.add_argument(
        "--condition",
        type=str,
        default="T0",
        choices=["T0", "T1", "T2"],
        help=(
            "Intervention condition: "
            "T0=small obstacle (baseline), "
            "T1=tall obstacle (structural change), "
            "T2=larger tool_block (motor/size change)"
        ),
    )
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument(
        "--steps",
        type=int,
        default=10_000,
        help="Total environment steps for SAC training",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to write JSON results (default: stdout only)",
    )
    ap.add_argument(
        "--max_ep_len",
        type=int,
        default=None,
        help=(
            "Override CausalWorld episode length (steps). "
            "Default: CausalWorld internal (2000 steps = 20s at skip_frame=10). "
            "Set to e.g. 200 for faster smoke tests."
        ),
    )
    args = ap.parse_args()

    if args.condition not in INTERVENTIONS:
        ap.error(f"Unknown condition '{args.condition}'. Choose from T0, T1, T2.")

    logger.info(
        "Starting SAC baseline | condition=%s | seed=%d | steps=%d",
        args.condition, args.seed, args.steps,
    )

    results = run_sac(
        condition=args.condition,
        seed=args.seed,
        total_steps=args.steps,
        max_episode_length=args.max_ep_len,
    )

    json_str = json.dumps(results, indent=2, default=str)
    print(json_str)

    if args.out is not None:
        out_path = os.path.abspath(args.out)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(json_str + "\n")
        logger.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
