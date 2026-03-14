#!/usr/bin/env python3
"""
scripts/train_coinrun_ppo.py — Train a CNN PPO policy on ProcGen CoinRun.

The trained model is saved to --out (default: models/coinrun_cnn_ppo.zip)
and can be loaded by watch_coinrun_dia.py --model models/coinrun_cnn_ppo.zip

Shaped reward (default on):
  - Raw game reward: +10 when coin collected (level complete)
  - One-shot: +3.0 first time coin becomes visible in an episode (seek reward)
  - Dense: +0.05 each step the coin is visible in-frame
  - Dense: +0.40 each step coin pixel-centroid moves left (agent approaching)

Usage:
    conda activate dia
    cd Discover-Intervene-Adapt-Interleaved-Causal-RL
    # quick test (few minutes):
    python scripts/train_coinrun_ppo.py --timesteps 200000 --n_envs 4
    # full run (~30-60 min on CPU, ~5 min on GPU):
    python scripts/train_coinrun_ppo.py --timesteps 1000000 --n_envs 8
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback
)


# ---------------------------------------------------------------------------
# Dense shaping wrapper
# ---------------------------------------------------------------------------

class CoinRunShapingWrapper(gym.Wrapper):
    """
    Adds dense reward signals on top of the sparse game reward.
    Operates on raw (64, 64, 3) uint8 pixel observations.

      +first_sight_bonus  one-shot when coin becomes visible for the first time
      +0.05               per step if coin (yellow) pixels visible
      +approach_bonus     per step if coin centroid moves left (agent approaching)
      +10.0               game reward when coin collected (passed through from procgen)
    """
    YELLOW_R, YELLOW_G, YELLOW_B = 200, 180, 60
    MIN_COIN_PX = 10

    def __init__(self, env, shaping: bool = True,
                 first_sight_bonus: float = 3.0,
                 approach_bonus: float = 0.40):
        super().__init__(env)
        self._shaping = shaping
        self._first_sight_bonus = first_sight_bonus
        self._approach_bonus = approach_bonus
        self._prev_cx: float | None = None
        self._coin_ever_seen: bool = False

    @staticmethod
    def _coin_centroid(frame: np.ndarray):
        r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
        mask = (r >= 200) & (g >= 180) & (b <= 60)
        npix = int(np.count_nonzero(mask))
        if npix < 10:
            return None, npix
        xs = np.where(mask)[1]
        return float(np.mean(xs)), npix

    def seed(self, seed=None):
        # procgen ignores runtime seeds (uses start_level at creation); silently accept
        try:
            return self.env.seed()
        except Exception:
            return []

    def reset(self, **kwargs):
        # strip seed kwarg that SB3/shimmy may pass; procgen doesn't support it
        kwargs.pop("seed", None)
        kwargs.pop("options", None)
        obs = self.env.reset(**kwargs)
        self._prev_cx = None
        self._coin_ever_seen = False
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if self._shaping and isinstance(obs, np.ndarray) and obs.ndim == 3:
            cx, npix = self._coin_centroid(obs)
            if cx is not None:
                # One-time bonus the first moment coin becomes visible this episode
                if not self._coin_ever_seen:
                    rew += self._first_sight_bonus
                    self._coin_ever_seen = True
                rew += 0.05   # coin visible (per-step)
                if self._prev_cx is not None and cx < self._prev_cx:
                    rew += self._approach_bonus  # centroid moved left → approaching
            self._prev_cx = cx
            if done:
                self._coin_ever_seen = False
        return obs, rew, done, info


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def make_env(rank: int, seed: int, num_levels: int, start_level: int,
             shaping: bool, first_sight_bonus: float = 3.0,
             approach_bonus: float = 0.40) -> callable:
    def _init():
        env = gym.make(
            "procgen:procgen-coinrun-v0",
            num_levels=num_levels,
            start_level=start_level,
            distribution_mode="easy",
        )
        env = CoinRunShapingWrapper(env, shaping=shaping,
                                    first_sight_bonus=first_sight_bonus,
                                    approach_bonus=approach_bonus)
        return env
    return _init


# ---------------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------------

class EpStats(BaseCallback):
    """Logs win-rate and mean reward every --log_every steps using done flags."""
    def __init__(self, log_every: int = 10_000):
        super().__init__()
        self._log_every = log_every
        self._ep_game_rewards: list[float] = []
        self._step_rewards: list[float] = []
        self._ep_lengths: list[int] = []
        self._cur_len = [0] * 1  # will resize on first call
        self._cur_rew = [0.0] * 1

    def _on_step(self) -> bool:
        n_envs = len(self.locals.get("dones", []))
        if len(self._cur_len) != n_envs:
            self._cur_len = [0] * n_envs
            self._cur_rew = [0.0] * n_envs

        rewards = self.locals.get("rewards", [0.0] * n_envs)
        dones   = self.locals.get("dones",   [False] * n_envs)
        infos   = self.locals.get("infos",   [{}] * n_envs)

        for i, (r, done, info) in enumerate(zip(rewards, dones, infos)):
            self._cur_rew[i] += float(r)
            self._cur_len[i] += 1
            if done:
                # game_rew = raw game reward (10 for win, 0 otherwise)
                game_r = float(info.get("prev_level_complete", False)) * 10.0
                self._ep_game_rewards.append(game_r)
                self._ep_lengths.append(self._cur_len[i])
                self._cur_rew[i] = 0.0
                self._cur_len[i] = 0

        if (self.num_timesteps % self._log_every < n_envs
                and self._ep_game_rewards):
            n     = len(self._ep_game_rewards)
            win_r = np.mean([r >= 9.0 for r in self._ep_game_rewards[-100:]])
            mean_l = np.mean(self._ep_lengths[-100:])
            print(f"  [{self.num_timesteps:>8,d}]  ep={n:5d}  "
                  f"win%={win_r*100:5.1f}  mean_len={mean_l:.0f}")
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps",   type=int,   default=1_000_000)
    ap.add_argument("--n_envs",      type=int,   default=4)
    ap.add_argument("--n_stack",     type=int,   default=4)
    ap.add_argument("--num_levels",  type=int,   default=200,
                    help="ProcGen levels to train on (lower = easier to overfit/memorize)")
    ap.add_argument("--start_level", type=int,   default=0)
    ap.add_argument("--no_shaping",        action="store_true")
    ap.add_argument("--first_sight_bonus", type=float, default=3.0,
                    help="One-shot reward the first time coin becomes visible per episode")
    ap.add_argument("--approach_bonus",    type=float, default=0.40,
                    help="Per-step reward when coin centroid moves left (agent approaching)")
    ap.add_argument("--out",               type=str,   default="models/coinrun_cnn_ppo")
    ap.add_argument("--seed",        type=int,   default=0)
    ap.add_argument("--lr",          type=float, default=5e-4)
    ap.add_argument("--n_steps",     type=int,   default=512,
                    help="Rollout steps per env per update (total = n_steps * n_envs)")
    ap.add_argument("--batch_size",  type=int,   default=256)
    ap.add_argument("--ent_coef",    type=float, default=0.01)
    ap.add_argument("--resume",      type=str,   default=None,
                    help="Path to existing model zip to resume from")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or "models", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)

    shaping = not args.no_shaping

    # --- training envs ---
    # DummyVecEnv (single-process) is used throughout: SubprocVecEnv conflicts with
    # the numpy 1.26/torch 2.5 dtype patch needed on this system.
    env_fns = [make_env(i, args.seed, args.num_levels, args.start_level, shaping,
                        first_sight_bonus=args.first_sight_bonus,
                        approach_bonus=args.approach_bonus)
               for i in range(args.n_envs)]
    venv = DummyVecEnv(env_fns)
    venv = VecFrameStack(venv, n_stack=args.n_stack)

    # --- eval envs (unseen levels) ---
    eval_fns = [make_env(0, args.seed + 999, 0, 2000, shaping=False)]
    eval_env = DummyVecEnv(eval_fns)
    eval_env = VecFrameStack(eval_env, n_stack=args.n_stack)

    # --- callbacks ---
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, 100_000 // args.n_envs),
        save_path="models/checkpoints",
        name_prefix="coinrun_ppo",
        verbose=0,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models/",
        log_path="models/",
        eval_freq=max(1, 50_000 // args.n_envs),
        n_eval_episodes=20,
        deterministic=True,
        verbose=0,
    )
    stats_cb = EpStats(log_every=10_000)

    # --- model ---
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=venv, verbose=0)
    else:
        model = PPO(
            "CnnPolicy",
            venv,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=4,
            gamma=0.999,
            gae_lambda=0.95,
            ent_coef=args.ent_coef,
            clip_range=0.2,
            max_grad_norm=0.5,
            verbose=0,
            seed=args.seed,
        )

    print("=" * 64)
    print(f"  CoinRun CNN PPO training")
    print(f"  timesteps={args.timesteps:,}  n_envs={args.n_envs}  "
          f"n_stack={args.n_stack}  num_levels={args.num_levels}")
    print(f"  shaping={'yes' if shaping else 'no'}  "
          f"first_sight={args.first_sight_bonus}  approach={args.approach_bonus}")
    print(f"  lr={args.lr}  n_steps={args.n_steps}  batch={args.batch_size}")
    print(f"  out={args.out}.zip")
    print("=" * 64)

    model.learn(
        total_timesteps=args.timesteps,
        callback=[ckpt_cb, eval_cb, stats_cb],
        reset_num_timesteps=args.resume is None,
    )

    model.save(args.out)
    venv.close()
    eval_env.close()

    print(f"\nSaved → {args.out}.zip")
    print(f"Best model (by eval) → models/best_model.zip")
    print(f"\nTo watch DIA with this policy:")
    print(f"  python scripts/watch_coinrun_dia.py --model {args.out}.zip --macro_steps 100")


if __name__ == "__main__":
    main()
