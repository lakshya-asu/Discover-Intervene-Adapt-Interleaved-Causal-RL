# src/dia/options.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable
import os

import numpy as np

from .types import Subgoal
from .evgs import EVGS

# try to import stable_baselines3; if missing, provide a fallback stub that instructs the user
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
    SB3_AVAILABLE = True
except Exception:  # pragma: no cover - user will install SB3 for full functionality
    PPO = None  # type: ignore
    ActorCriticPolicy = object  # type: ignore
    DummyVecEnv = None  # type: ignore
    VecEnv = object  # type: ignore
    SB3_AVAILABLE = False


@dataclass
class OptionConfig:
    max_steps: int = 128
    terminate_on_success: bool = True
    # PPO training params (used when training an option)
    ppo_learning_rate: float = 3e-4
    ppo_n_steps: int = 2048
    ppo_batch_size: int = 64
    ppo_n_epochs: int = 10
    ppo_gamma: float = 0.99
    ppo_verbose: int = 0
    ppo_total_timesteps: int = 10000
    model_dir: Optional[str] = None


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
        """Execute until success or horizon, returning a summary dict.

        This default implementation executes actions from act() repeatedly. When integrating a
        SB3 model, `act()` will call the loaded/trained model's predict method.
        """
        obs = env.reset() if not hasattr(env, "get_obs") else env.get_obs()
        x_t = evgs.extract(obs)
        steps = 0
        success = False
        trajectory = []
        while steps < self.cfg.max_steps:
            action = self.act(obs)
            next_obs, ext_rew, done, info = env.step(action)
            x_tp1 = evgs.extract(next_obs)
            succ_this = EVGS.predicate_holds(x_t, x_tp1, self.subgoal)
            trajectory.append((obs, action, next_obs, succ_this))
            self.update({"obs": obs, "action": action, "next_obs": next_obs, "info": info})
            obs = next_obs
            steps += 1
            if succ_this:
                success = True
                if self.cfg.terminate_on_success:
                    break
            if done:
                # allow continuing if environment exposes "soft_continue" to ignore episodic boundaries
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


class PPOOption(OptionPolicy):
    """
    Option policy implemented using Stable‑Baselines3 PPO.
    - Call `train(env)` to train the option to achieve its subgoal using shaped reward.
    - `load(path)` and `save(path)` are provided for persistence.
    - `act(obs)` uses the loaded SB3 model if available; otherwise raises error.
    """

    def __init__(self, subgoal: Subgoal, cfg: OptionConfig, model: Optional[Any] = None):
        super().__init__(subgoal, cfg)
        self.model = model
        self.env_for_training = None  # optionally store a training env

    def _make_vec_env(self, env):
        """Wrap env into a vectorized DummyVecEnv if SB3 is available"""
        if not SB3_AVAILABLE:
            raise RuntimeError("stable_baselines3 is not installed. Install it to use PPOOption.")
        if isinstance(env, VecEnv):
            return env
        # DummyVecEnv expects a callable that returns an env
        return DummyVecEnv([lambda: env])

    def train(self, env, reward_wrapper: Optional[Callable] = None, total_timesteps: Optional[int] = None):
        """Train a PPO model to perform this option.
        Args:
            env: gym.Env (or VecEnv). If reward_wrapper is provided, it should wrap env to give +1 for subgoal.
            reward_wrapper: optional function that, given env, returns a wrapped env with shaped rewards.
            total_timesteps: override cfg.ppo_total_timesteps
        """
        if not SB3_AVAILABLE:
            raise RuntimeError("stable_baselines3 is not installed. Please install stable-baselines3 to train PPO options.")

        if reward_wrapper is not None:
            env_train = reward_wrapper(env)
        else:
            env_train = env

        vec_env = self._make_vec_env(env_train)
        model = PPO("MlpPolicy", vec_env,
                    learning_rate=self.cfg.ppo_learning_rate,
                    n_steps=self.cfg.ppo_n_steps,
                    batch_size=self.cfg.ppo_batch_size,
                    n_epochs=self.cfg.ppo_n_epochs,
                    gamma=self.cfg.ppo_gamma,
                    verbose=self.cfg.ppo_verbose)
        total = int(total_timesteps or self.cfg.ppo_total_timesteps)
        model.learn(total_timesteps=total)
        self.model = model
        self.env_for_training = env_train
        return model

    def act(self, obs):
        if self.model is None or not SB3_AVAILABLE:
            raise RuntimeError("No PPO model available; call train() or load() to initialize a model.")
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("No model to save")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def load(self, path: str, env=None):
        if not SB3_AVAILABLE:
            raise RuntimeError("stable_baselines3 is not installed.")
        if env is not None:
            vec_env = self._make_vec_env(env)
            model = PPO.load(path, env=vec_env)
        else:
            model = PPO.load(path)
        self.model = model
        return model
