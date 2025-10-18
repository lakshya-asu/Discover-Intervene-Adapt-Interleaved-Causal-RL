# src/dia/rollout.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np

from .evgs import EVGS
from .types import Subgoal
from .sig import SIGraph, Skill
from .planner import InterventionSelector
from .options import OptionPolicy, RandomOption, OptionConfig
from .logging_utils import TBLogger
from .sig_auto import expand_sig_from_pcg, AutoSIGConfig


# ----------------------------- Buffer for PCG fitting -----------------------------

class PCGBuffer:
    """
    Stores macro-transitions at the option level:
      (X_t, X_{t+1}, mask_row)
    where mask_row[j] = 1 indicates variable j was intervened in this sample (ignore residuals for j).
    """
    def __init__(self, num_vars: int, capacity: int = 10_000):
        self.d = int(num_vars)
        self.cap = int(capacity)
        self.X_t = np.zeros((self.cap, self.d), dtype=float)
        self.X_tp1 = np.zeros((self.cap, self.d), dtype=float)
        self.mask = np.zeros((self.cap, self.d), dtype=float)
        self.ptr = 0
        self.full = False

    def __len__(self) -> int:
        return self.cap if self.full else self.ptr

    def add(self, x_t: np.ndarray, x_tp1: np.ndarray, intervened_idx: Optional[int] = None,
            mask_row: Optional[np.ndarray] = None):
        x_t = np.asarray(x_t, dtype=float).reshape(-1)
        x_tp1 = np.asarray(x_tp1, dtype=float).reshape(-1)
        if x_t.shape[0] != self.d or x_tp1.shape[0] != self.d:
            raise ValueError(f"PCGBuffer.add: expected vectors of length {self.d}, got {x_t.shape}, {x_tp1.shape}")

        i = self.ptr
        self.X_t[i] = x_t
        self.X_tp1[i] = x_tp1
        m = np.zeros(self.d, dtype=float) if mask_row is None else np.asarray(mask_row, dtype=float).reshape(-1)
        if intervened_idx is not None and 0 <= int(intervened_idx) < self.d:
            m[int(intervened_idx)] = 1.0
        if m.shape[0] != self.d:
            raise ValueError(f"PCGBuffer.add: mask_row wrong shape {m.shape}, expected ({self.d},)")
        self.mask[i] = m

        self.ptr += 1
        if self.ptr >= self.cap:
            self.ptr = 0
            self.full = True

    def recent(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self)
        if n == 0:
            return np.zeros((0, self.d * 2)), np.zeros((0, self.d))
        k = int(min(k, n))
        idx = (np.arange(k) + (self.ptr - k)) % (self.cap if self.full else self.ptr)
        X_inputs = self.X_t[idx]
        X_targets = self.X_tp1[idx]
        mask = self.mask[idx]
        packed = np.hstack([X_inputs, X_targets])
        return packed, mask


# ----------------------------- Runner configuration -----------------------------

@dataclass
class RunnerConfig:
    buffer_size: int = 10_000
    min_buffer: int = 256
    batch_recent: int = 2048
    fit_every: int = 10            # fit PCG every N option executions
    pcg_epochs: int = 200          # number of optimization steps per fit() call
    log_prefix: str = "dia"
    terminate_on_success: bool = True  # pass to options
    option_max_steps: int = 64

    # Auto-SIG expansion
    auto_expand_sig: bool = True
    add_threshold: float = 0.75
    remove_threshold: float = 0.55
    expand_every: int = 0  # 0 => expand right after a fit


# ----------------------------- DIA Runner -----------------------------

class DIARunner:
    """
    High-level Discover–Intervene–Adapt loop with:
      - goal-aware selection (if task_goal provided)
      - PCG dataset buffer and periodic fit()
      - IG logging
      - optional Auto-SIG expansion from PCG posteriors
    """

    def __init__(
        self,
        env: Any,
        evgs: EVGS,
        pcg: Any,                    # SimplePCG or DifferentiablePCG/VariationalPCG
        sig: SIGraph,
        selector: InterventionSelector,
        cfg: RunnerConfig,
        logger: Optional[TBLogger] = None,
        option_factory: Optional[Callable[[Skill], OptionPolicy]] = None,
    ):
        self.env = env
        self.evgs = evgs
        self.pcg = pcg
        self.sig = sig
        self.selector = selector
        self.cfg = cfg
        self.logger = logger
        self.option_factory = option_factory
        self.options: Dict[int, OptionPolicy] = {}
        self.buffer = PCGBuffer(num_vars=len(evgs.var_names), capacity=cfg.buffer_size)
        self.steps = 0

    # -------- option registry --------

    def get_option(self, skill: Skill) -> OptionPolicy:
        sid = skill.skill_id
        if sid in self.options:
            return self.options[sid]
        if self.option_factory is not None:
            opt = self.option_factory(skill)
        else:
            opt = RandomOption(
                subgoal=skill.subgoal,
                cfg=OptionConfig(max_steps=self.cfg.option_max_steps, terminate_on_success=self.cfg.terminate_on_success),
                action_space=self.env.action_space,
            )
        self.options[sid] = opt
        return opt

    # -------- PCG fitting / SIG expansion --------

    def _maybe_fit_pcg(self) -> Tuple[bool, float]:
        self.steps += 1
        if self.steps % max(1, self.cfg.fit_every) != 0:
            return False, 0.0
        if len(self.buffer) < self.cfg.min_buffer:
            return False, 0.0
        if not hasattr(self.pcg, "fit"):
            return False, 0.0

        packed, mask = self.buffer.recent(self.cfg.batch_recent)
        if packed.shape[0] == 0:
            return False, 0.0
        d = len(self.evgs.var_names)
        X_targets = packed[:, d:]  # learners expect X (targets) and regress X on X @ W internally

        old_probs = getattr(self.pcg, "probs").copy()
        res = self.pcg.fit(X_targets, mask=mask, epochs=self.cfg.pcg_epochs)
        new_probs = getattr(self.pcg, "probs")

        # IG
        try:
            ig_update = float(self.pcg.expected_ig_from_update(new_probs))
        except Exception:
            eps = 1e-8
            p_new = np.clip(new_probs, eps, 1 - eps)
            p_old = np.clip(old_probs, eps, 1 - eps)
            kl = p_new * np.log(p_new / p_old) + (1 - p_new) * np.log((1 - p_new) / (1 - p_old))
            ig_update = float(np.sum(kl))

        if self.logger:
            self.logger.add_scalar(f"{self.cfg.log_prefix}/pcg_entropy", float(self.pcg.entropy()))
            self.logger.add_scalar(f"{self.cfg.log_prefix}/ig_update", ig_update)

        # Maybe expand SIG
        if self.cfg.auto_expand_sig and hasattr(self.pcg, "probs"):
            should_expand = (self.cfg.expand_every == 0) or (self.steps % max(1, self.cfg.expand_every) == 0)
            if should_expand:
                stats = expand_sig_from_pcg(
                    self.sig, self.evgs, self.pcg.probs,
                    AutoSIGConfig(
                        add_threshold=self.cfg.add_threshold,
                        remove_threshold=self.cfg.remove_threshold,
                        create_missing_skills=True,
                        verbose=False,
                    )
                )
                if self.logger:
                    self.logger.add_scalar(f"{self.cfg.log_prefix}/sig_added", float(stats["added"]))
                    self.logger.add_scalar(f"{self.cfg.log_prefix}/sig_removed", float(stats["removed"]))
                    self.logger.add_scalar(f"{self.cfg.log_prefix}/sig_created", float(stats["created_skills"]))

        return True, ig_update

    # -------- main step --------

    def step(self, achieved: List[int], task_goal: Optional[Subgoal] = None) -> Dict[str, Any]:
        # Select a skill (goal-aware if task_goal provided)
        phase = self.selector.phase()
        skill_id = self.selector.select(achieved, task_goal)
        skill = self.sig.skills[skill_id]
        option = self.get_option(skill)

        # Execute option
        out = option.run(self.env, self.evgs)
        success: bool = bool(out["success"])
        final_obs = out["final_obs"]
        traj = out.get("trajectory", [])
        if len(traj) == 0:
            obs0 = final_obs
            x0 = self.evgs.extract(obs0)
            x1 = x0.copy()
        else:
            obs0 = traj[0][0]
            x0 = self.evgs.extract(obs0)
            x1 = self.evgs.extract(final_obs)

        # SIG stats
        delta_x = x1 - x0
        skill.update_stats(success=success, delta_x=delta_x)
        if self.logger:
            self.logger.add_scalar(f"{self.cfg.log_prefix}/skill_success_{skill_id}", float(success))

        # Push into PCG buffer, mark target var as intervened
        self.buffer.add(x0, x1, intervened_idx=skill.subgoal.var_index)

        # Maybe fit PCG & expand SIG
        did_fit, ig_update = self._maybe_fit_pcg()

        rec = {
            "phase": phase,
            "skill_id": skill_id,
            "skill_name": skill.name,
            "success": success,
            "delta_x": delta_x,
            "pcg_entropy": float(self.pcg.entropy()) if hasattr(self.pcg, "entropy") else np.nan,
            "ig_update": ig_update if did_fit else 0.0,
            "did_fit_pcg": did_fit,
            "buffer_size": len(self.buffer),
        }

        if self.logger:
            self.logger.add_scalar(f"{self.cfg.log_prefix}/success", float(success))
            if hasattr(self.pcg, "entropy"):
                self.logger.add_scalar(f"{self.cfg.log_prefix}/pcg_entropy_live", float(self.pcg.entropy()))

        return rec
