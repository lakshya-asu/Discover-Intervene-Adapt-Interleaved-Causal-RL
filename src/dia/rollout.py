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
    expand_every: int = 0          # 0 => expand right after a fit
    create_missing_skills: bool = True  # auto-create skills for new variables in SIG expansion


# ----------------------------- DIA Runner -----------------------------

class DIARunner:
    """
    High-level Discover–Intervene–Adapt loop with:
      - goal-aware selection (if task_goal provided)
      - PCG dataset buffer and periodic fit()
      - IG logging (always KL between old/new edge-probabilities)
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

    @staticmethod
    def _bernoulli_kl(p_new: np.ndarray, p_old: np.ndarray, eps: float = 1e-8) -> float:
        p_new = np.clip(p_new, eps, 1 - eps)
        p_old = np.clip(p_old, eps, 1 - eps)
        kl = p_new * np.log(p_new / p_old) + (1 - p_new) * np.log((1 - p_new) / (1 - p_old))
        return float(np.sum(kl))

    @staticmethod
    def _interventional_fit_probs(X_t: np.ndarray, X_tp1: np.ndarray,
                                   mask: np.ndarray, alpha: float = 5.0) -> np.ndarray:
        """
        Interventional causal-edge estimation using the skill execution mask.

        For edge (i -> j) the question is:
          "When skill j was executed (do(j)), did having i already done help j succeed?"

        Only transitions where skill j was the intervened variable (mask[:, j] = 1) are
        used to estimate edge i -> j.  This isolates the interventional signal:
          P(j↑ | do(j), i=1 at t)  vs  P(j↑ | do(j), i=0 at t)

        In a prerequisite chain this cleanly recovers the structure:
          - ironpickaxe -> diamond: diamond only increases when ironpickaxe >= 1
          - observational correlations between non-adjacent variables are suppressed

        Returns NaN for edges with insufficient data so callers can preserve
        the old PCG probability for those edges rather than overwriting with 0.5.
        Only edges with sufficient interventional evidence get a real estimate.

        Returns a [d, d] matrix: estimated edge probabilities where observed,
        NaN where there is insufficient data (diagonal is 0).
        """
        if X_t.shape[0] == 0:
            d = X_t.shape[1] if X_t.ndim > 1 else 1
            return np.full((d, d), np.nan)

        N, d = X_t.shape
        probs = np.full((d, d), np.nan, dtype=float)
        min_samples = 3

        for j in range(d):
            # Only use transitions where skill j was the intervened skill
            int_idx = (mask[:, j] > 0.5) if mask.ndim == 2 else np.zeros(N, dtype=bool)
            n_int = int(np.sum(int_idx))
            if n_int < min_samples:
                continue

            x_t_j   = X_t[int_idx]
            x_tp1_j = X_tp1[int_idx]
            j_up    = (x_tp1_j[:, j] > x_t_j[:, j]).astype(float)  # did j increase?

            for i in range(d):
                if i == j:
                    continue
                active_i   = (x_t_j[:, i] > 0.5).astype(float)
                inactive_i = 1.0 - active_i

                n_active   = float(np.sum(active_i))
                n_inactive = float(np.sum(inactive_i))

                # Require both sufficient inactive samples AND a minimum balance ratio.
                # Without this, early training is dominated by n_inactive=2-3 samples
                # whose Beta posteriors are prior-dominated (alpha=5), causing 40+ FPs
                # the moment the buffer first fills (step ~260).
                min_inactive = 15
                balance_min  = 0.25   # inactive must be ≥ 25% of intervention samples
                n_total = n_active + n_inactive
                if (n_active < min_samples or n_inactive < min_inactive
                        or n_inactive / n_total < balance_min):
                    # insufficient contrast — leave as NaN (caller keeps old prob)
                    continue

                # Beta posterior: (successes + alpha) / (trials + 2*alpha)
                p_up_active   = (float(np.sum(active_i   * j_up)) + alpha) / (n_active   + 2 * alpha)
                p_up_inactive = (float(np.sum(inactive_i * j_up)) + alpha) / (n_inactive + 2 * alpha)

                denom = p_up_active + p_up_inactive
                probs[i, j] = p_up_active / denom if denom > 1e-10 else np.nan

        np.fill_diagonal(probs, 0.0)
        return probs

    def _maybe_fit_pcg(self) -> Tuple[bool, float, float]:
        self.steps += 1
        if self.steps % max(1, self.cfg.fit_every) != 0:
            return False, 0.0, 0.0
        if len(self.buffer) < self.cfg.min_buffer:
            return False, 0.0, 0.0

        packed, mask = self.buffer.recent(self.cfg.batch_recent)
        if packed.shape[0] == 0:
            return False, 0.0, 0.0

        d = len(self.evgs.var_names)
        X_t_data   = packed[:, :d]   # state BEFORE the option
        X_tp1_data = packed[:, d:]   # state AFTER the option

        old_probs   = np.array(getattr(self.pcg, "probs")).copy()
        old_entropy = float(getattr(self.pcg, "entropy")()) if hasattr(self.pcg, "entropy") else np.nan

        # ── Causal discovery: interventional edge estimation ─────────────────
        # For edge (i -> j): only uses transitions where skill j was executed.
        # Asks "when we tried to achieve j, did having i active help?"
        # This suppresses observational correlations and recovers prerequisites.
        # Returns NaN for edges with insufficient data; we preserve old probs there.
        est_probs = self._interventional_fit_probs(X_t_data, X_tp1_data, mask)

        # Merge: only update edges where we have real signal; keep old probs elsewhere
        new_probs = old_probs.copy()
        observed = ~np.isnan(est_probs)
        new_probs[observed] = est_probs[observed]
        np.fill_diagonal(new_probs, 0.0)

        if hasattr(self.pcg, "apply_update"):
            self.pcg.apply_update(new_probs)
        elif hasattr(self.pcg, "fit"):
            # Fallback for PCGs without apply_update (legacy or custom)
            try:
                _ = self.pcg.fit(X_tp1_data, mask=mask, epochs=self.cfg.pcg_epochs)
                new_probs = np.array(getattr(self.pcg, "probs"))
            except Exception:
                pass

        new_probs   = np.array(getattr(self.pcg, "probs"))
        new_entropy = float(getattr(self.pcg, "entropy")()) if hasattr(self.pcg, "entropy") else np.nan

        ig_update    = self._bernoulli_kl(new_probs, old_probs)
        entropy_drop = float(old_entropy - new_entropy) if (not np.isnan(old_entropy) and not np.isnan(new_entropy)) else 0.0

        if self.logger:
            self.logger.add_scalar(f"{self.cfg.log_prefix}/pcg_entropy", new_entropy)
            self.logger.add_scalar(f"{self.cfg.log_prefix}/ig_update", ig_update)
            self.logger.add_scalar(f"{self.cfg.log_prefix}/entropy_drop", entropy_drop)

        # ── SIG auto-expansion from updated PCG probs ────────────────────────
        if self.cfg.auto_expand_sig and hasattr(self.pcg, "probs"):
            should_expand = (self.cfg.expand_every == 0) or (self.steps % max(1, self.cfg.expand_every) == 0)
            if should_expand:
                stats = expand_sig_from_pcg(
                    self.sig, self.evgs, self.pcg.probs,
                    AutoSIGConfig(
                        add_threshold=self.cfg.add_threshold,
                        remove_threshold=self.cfg.remove_threshold,
                        create_missing_skills=self.cfg.create_missing_skills,
                        verbose=False,
                    )
                )
                if self.logger:
                    self.logger.add_scalar(f"{self.cfg.log_prefix}/sig_added", float(stats["added"]))
                    self.logger.add_scalar(f"{self.cfg.log_prefix}/sig_removed", float(stats["removed"]))
                    self.logger.add_scalar(f"{self.cfg.log_prefix}/sig_created", float(stats["created_skills"]))

        return True, ig_update, entropy_drop

    # -------- checkpoint save/load --------

    def save_checkpoint(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save PCG + SIG (and lightweight option metadata) to a JSON file."""
        from .checkpoint import save_checkpoint as _save
        _save(path, self.pcg, self.sig, self.options, metadata)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Restore PCG + SIG state from a JSON checkpoint. Returns the raw dict."""
        from .checkpoint import load_checkpoint as _load
        return _load(path, self.pcg, self.sig)

    # -------- main step --------

    def step(self, achieved: List[int], task_goal: Optional[Subgoal] = None) -> Dict[str, Any]:
        # Select a skill (goal-aware if task_goal provided)
        phase = self.selector.phase()
        skill_id = self.selector.select(achieved, task_goal)
        skill = self.sig.skills[skill_id]
        option = self.get_option(skill)

        # Execute option
        obs = self.env.get_obs() if hasattr(self.env, "get_obs") else self.env.reset()
        x0 = self.evgs.extract(obs)
        out = option.run(self.env, self.evgs)
        success: bool = bool(out["success"])
        final_obs = out["final_obs"]
        x1 = self.evgs.extract(final_obs)

        # SIG stats
        delta_x = x1 - x0
        skill.update_stats(success=success, delta_x=delta_x)
        if self.logger:
            self.logger.add_scalar(f"{self.cfg.log_prefix}/skill_success_{skill_id}", float(success))

        # Push option-level transition (x_start → x_end)
        self.buffer.add(x0, x1, intervened_idx=skill.subgoal.var_index)

        # Also push any per-step informative transitions returned by the option
        # (state changes that happened mid-option, e.g. coin becoming visible)
        for xt_s, xtp1_s in out.get("step_pairs", []):
            self.buffer.add(xt_s, xtp1_s, intervened_idx=skill.subgoal.var_index)

        # Maybe fit PCG & expand SIG
        did_fit, ig_update, entropy_drop = self._maybe_fit_pcg()

        rec = {
            "phase": phase,
            "skill_id": skill_id,
            "skill_name": skill.name,
            "success": success,
            "delta_x": delta_x,
            "pcg_entropy": float(self.pcg.entropy()) if hasattr(self.pcg, "entropy") else np.nan,
            "ig_update": ig_update if did_fit else 0.0,
            "did_fit_pcg": did_fit,
            "entropy_drop": entropy_drop if did_fit else 0.0,
            "buffer_size": len(self.buffer),
        }

        if self.logger:
            self.logger.add_scalar(f"{self.cfg.log_prefix}/success", float(success))
            if hasattr(self.pcg, "entropy"):
                self.logger.add_scalar(f"{self.cfg.log_prefix}/pcg_entropy_live", float(self.pcg.entropy()))

        return rec
