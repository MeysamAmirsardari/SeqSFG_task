"""Simulated listeners: the hypothesis and its rivals, each as something that can be run.

Nothing in this module is data. It exists so that four questions can be answered before a
person is asked to sit through an evening of trials:

1. Does the pipeline work end to end -- design, tracks, logging, analysis -- without a human?
2. Does the analysis find the effect when the effect is there?  (power)
3. Does the analysis fail to find it when it is not?  (false positive rate)
4. Can the analysis tell the hypothesis apart from the rivals that predict a rising curve for
   uninteresting reasons?  (this is the one that matters, and the answer is not automatically
   yes -- `tcoh.analysis.discriminability` reports how often it succeeds)

Every listener here is a psychometric function whose threshold depends on the condition. The
four differ only in that dependence:

    coherence  log threshold interpolates between a floor and the B-only ceiling in proportion
               to the model's own segregation index at that dT. The hypothesis.
    pedestal   threshold grows with the INTERVAL the listener has to judge -- the lag between
               the final A and B tones -- with no reference to whether an A stream exists. The
               Weber rival. It predicts the same rising curve in the coherent condition AND in
               the jittered control, because the controls have the same final interval.
    overlap    threshold is linear in the acoustic overlap of the two tones. The simplest
               sensory rival, and nearly indistinguishable from `coherence` on the coherent
               conditions alone -- which is exactly why the controls exist.
    null       threshold does not depend on dT at all. Used to check the false positive rate.

Any figure or table produced from these must say SIMULATED on its face. The one in
`tcoh/notebooks` does.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .config import Condition, Config, validate
from .psychometric import PF

MODES = ("coherence", "pedestal", "overlap", "null")


@dataclass
class SimulatedListener:
    cfg: Config
    mode: str = "coherence"
    floor_ms: float = 3.0        # threshold when the two tones are one object (Elhilali: 2-4 ms)
    ceiling_ms: float = 15.0     # threshold from the B rhythm alone (Elhilali: 10-20 ms)
    sigma: float = 0.6           # slope of the psychometric function, natural log units
    lapse: float = 0.02
    seed: int = 20260913

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self._d = validate(self.cfg)
        self._rng = np.random.default_rng(self.seed)
        self._cond = {c.name: c for c in self._d.conditions}

    # -- where the threshold sits ------------------------------------------------
    def kappa(self, cond: Condition) -> float:
        """The fraction of the way from the floor to the ceiling this condition should sit."""
        if cond.a_kind == "absent":
            return 1.0
        if self.mode == "null":
            return 0.0
        if self.mode == "coherence":
            if cond.a_kind in ("scrambled", "pair_only", "nopartner"):
                return 1.0          # no coherent A stream to bind to, whatever the lag is
            return float(self._d.model_curve.get(cond.lag_pct, 0.0))
        if self.mode == "overlap":
            if cond.a_kind in ("scrambled", "pair_only", "nopartner"):
                return 1.0
            return float(cond.lag_pct) / 100.0      # overlap falls linearly with dT at duty 0.5
        if self.mode == "pedestal":
            # the judged interval is the final A-B lag, which the jittered control shares, so
            # this rival makes no distinction between them at all.
            if cond.a_kind in ("nopartner", "pair_only"):
                return 1.0
            lag = self.cfg.lag_ms(cond.lag_pct)
            return float(np.clip(lag / (self.cfg.soa_ms / 2.0), 0.0, 1.0))
        raise AssertionError(self.mode)

    def threshold_ms(self, cond: Condition) -> float:
        k = self.kappa(cond)
        return float(math.exp(math.log(self.floor_ms)
                              + k * (math.log(self.ceiling_ms) - math.log(self.floor_ms))))

    def pf(self, cond: Condition) -> PF:
        return PF(self.threshold_ms(cond), self.sigma, self.lapse)

    # -- responding ---------------------------------------------------------------
    def respond(self, condition, delta_ms: float) -> bool:
        c = condition if isinstance(condition, Condition) else self._cond[condition]
        return bool(self.pf(c).respond(delta_ms, self._rng))

    def truth(self) -> Dict[str, float]:
        """The thresholds this listener was built with, for checking what came back out."""
        return {c.name: self.threshold_ms(c) for c in self._d.conditions}
