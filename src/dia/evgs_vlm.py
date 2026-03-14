"""
evgs_vlm.py
===========
VLM-enriched EVGS wrapper for Montezuma's Revenge.

Fuses two independent evidence streams:
  ① RAM-based hard binary values  (always available, 0 or 1)
  ② VLM soft-probability estimates (Claude claude-opus-4-6, every N steps)

Fusion formula (default α = 0.6):
  enriched[i] = α · ram[i] + (1 − α) · vlm[i]

This produces continuous values in [0, 1] that reflect both the precise
RAM sensor and the model's visual interpretation.  The soft values give
GrangerPCG a richer gradient signal compared to hard binary transitions.

Drop-in replacement for EVGS:
  - same .var_names, .extract(), .names(), .predicate_holds() interface
  - adds  .last_vlm  property returning the most recent VLMSceneResult
  - adds  .vlm_stats()  for API usage diagnostics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from dia.evgs import EVGS
from dia.types import Subgoal


@dataclass
class VLMFusionConfig:
    """Controls how VLM and RAM signals are combined."""

    alpha: float = 0.6
    """Weight for RAM-based (hard) values.
    0.6 → 60 % RAM + 40 % VLM.  Higher = trust RAM more."""

    frame_key: str = "obs"
    """Key inside the obs-dict that holds the RGB frame (numpy array)."""

    passthrough_info: bool = True
    """If True, write ``vlm_*`` soft values back into ``obs["info"]`` so
    downstream logging and TensorBoard can inspect raw VLM outputs."""


class VLMEnrichedEVGS:
    """
    Wraps any EVGS instance and enriches its variable vector with
    soft VLM predictions from a ``VLMSceneAnalyzer``.

    Compatible with the full EVGS protocol used by DIARunner / DirectedOption:
        .var_names              list[str]
        .extract(obs)       →   np.ndarray shape [M]   (soft, values in [0,1])
        .names()            →   list[str]
        .predicate_holds()      static; uses base EVGS logic unchanged
    """

    def __init__(
        self,
        base_evgs: EVGS,
        vlm_analyzer,                  # VLMSceneAnalyzer (imported lazily)
        cfg: Optional[VLMFusionConfig] = None,
    ) -> None:
        self.base = base_evgs
        self.vlm  = vlm_analyzer
        self.cfg  = cfg or VLMFusionConfig()

        # Mirror the attribute the rest of DIA code reads
        self.var_names: List[str] = list(base_evgs.var_names)

        # Cache for the most recent VLM result (exposed as .last_vlm)
        self._last_vlm_result = None

    # ------------------------------------------------------------------ #
    # EVGS interface                                                       #
    # ------------------------------------------------------------------ #

    def names(self) -> List[str]:
        return self.var_names

    def extract(self, obs) -> np.ndarray:
        """
        Fuse RAM-based EVGS extraction with VLM soft estimates.

        Parameters
        ----------
        obs : dict  {"obs": np.ndarray (H×W×3), "info": {...}}
            Standard MontezumaRichWrapper observation dict.

        Returns
        -------
        np.ndarray shape [M], values in [0, 1]
        """
        # ── ① RAM-based hard values ────────────────────────────────────
        ram_vec: np.ndarray = self.base.extract(obs)

        # ── ② Extract the RGB frame ────────────────────────────────────
        frame: Optional[np.ndarray] = None
        if isinstance(obs, dict):
            frame = obs.get(self.cfg.frame_key)

        # ── ③ Call / cache VLM ────────────────────────────────────────
        vlm_result = self.vlm.analyze(frame)
        self._last_vlm_result = vlm_result

        if vlm_result.error:
            # VLM unavailable — return pure RAM values unchanged
            return ram_vec

        # ── ④ Fuse ────────────────────────────────────────────────────
        vlm_vec = vlm_result.to_array(self.var_names)
        alpha   = self.cfg.alpha
        fused   = alpha * ram_vec + (1.0 - alpha) * vlm_vec

        # ── ⑤ Optionally inject VLM values into obs["info"] ───────────
        if self.cfg.passthrough_info and isinstance(obs, dict):
            info = obs.get("info")
            if isinstance(info, dict):
                info.update(vlm_result.as_info_addendum())

        return fused

    @staticmethod
    def predicate_holds(
        x_t: np.ndarray, x_tp1: np.ndarray, sg: Subgoal
    ) -> bool:
        """Delegate to the base EVGS static method unchanged."""
        return EVGS.predicate_holds(x_t, x_tp1, sg)

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    @property
    def last_vlm(self):
        """Most recent ``VLMSceneResult`` (or None before first call)."""
        return self._last_vlm_result

    def vlm_stats(self) -> dict:
        """API usage statistics delegated to the underlying analyzer."""
        return self.vlm.stats()
