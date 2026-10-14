"""Cues, the observers that use them, and the ceilings that make the design interpretable.

A cue is a number computed from one interval's gain matrix. An observer compares that number
across the two intervals and picks the larger. Two cues are the design:

  timbre       the target re the mean of the other tones, on the probe       exact in A
  own          the target on the probe re its own earlier tones              exact in B

and every other cue a listener could plausibly use is a candidate shortcut. `ceilings` computes,
for each trial type, the threshold of an ideal observer of each single cue. A cue that is exact
has no ceiling; every other cue must need a large increment, or the type does not force the
comparison it is meant to force.

What a ceiling is not
---------------------
It is not the performance of a listener with every piece of information. Such a listener could
read each tone's level separately and do both types perfectly; that is what hearing every
component as its own stream, and comparing across streams at will, would amount to. The design
rests on the perceptual premise that this is hard: comparing levels across components at one
moment is easy when they are bound (profile analysis) and hard across streams; following one
component over time is easy when it is a stream of its own and hard inside a fused chord. The
ceilings rule out the shortcuts that need neither, which is what a stimulus can rule out.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .config import Condition, Config, delta_db, srs_db
from .stimulus import draw_latent, gains_from_latent

P_TARGET = 0.5 ** (1.0 / 3.0)       # 3-down 1-up: 0.7937


def _pw(g: np.ndarray) -> np.ndarray:
    """Power sum in dB over the last axis, ignoring absent tones."""
    return 10.0 * np.log10(np.nansum(10.0 ** (g / 10.0), axis=-1))


def cue_functions(cfg: Config) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    t, oth = cfg.target_index, cfg.others
    near = [t - 1, t + 1]

    def timbre_seq(g):
        return g[..., :, t] - np.mean(g[..., :, oth], axis=-1)

    return {
        "timbre": lambda g: timbre_seq(g)[..., -1],
        "own": lambda g: g[..., -1, t] - np.mean(g[..., :-1, t], axis=-1),
        "timbre_change": lambda g: timbre_seq(g)[..., -1] - np.mean(timbre_seq(g)[..., :-1],
                                                                    axis=-1),
        "nearest": lambda g: g[..., -1, t] - np.mean(g[..., -1, near], axis=-1),
        "target_level": lambda g: g[..., -1, t],
        "target_re_earlier_others": lambda g: g[..., -1, t] - np.mean(g[..., :-1, oth],
                                                                      axis=(-1, -2)),
        "overall_level": lambda g: _pw(g[..., -1, :]),
        "overall_change": lambda g: _pw(g[..., -1, :]) - np.mean(_pw(g[..., :-1, :]), axis=-1),
    }


CUE_LABELS = {
    "timbre": "target re others, probe (timbre)",
    "own": "target re its own past (own stream)",
    "timbre_change": "change of timbre into the probe",
    "nearest": "target re its two neighbours, probe",
    "target_level": "target level, across intervals",
    "target_re_earlier_others": "target re the others' earlier tones",
    "overall_level": "loudness of the probe, across intervals",
    "overall_change": "loudness change into the probe",
}
SURVIVING = {"A": "timbre", "B": "own"}


def cues(cfg: Config, gains: np.ndarray) -> Dict[str, np.ndarray]:
    """Every cue for one gain matrix or a stack of them. NaN where a cue needs absent tones."""
    g = np.asarray(gains, dtype=float)
    with np.errstate(invalid="ignore"):
        return {k: f(g) for k, f in cue_functions(cfg).items()}


def draw_gains(cfg: Config, cond: Condition, delta: float, signal: bool, n: int,
               rng: np.random.Generator) -> np.ndarray:
    return np.stack([gains_from_latent(cfg, cond.task, cond.kind, draw_latent(cfg, rng),
                                       delta, signal)[0] for _ in range(n)])


def _pc(diff: np.ndarray) -> float:
    return float(np.mean(diff > 0) + 0.5 * np.mean(diff == 0))


def ideal_threshold(cfg: Config, cond: Condition, cue: str, n: int = 20000,
                    seed: int = 0) -> float:
    """Increment (dL, dB) at which an ideal observer of ONE cue is 79.4% correct.

    0 for an exact cue; inf if it never gets there below 40 dB. Common random numbers: the rove
    draws are fixed and only the increment moves, so the bisection is on a monotone function.
    """
    rng = np.random.default_rng(seed)
    gs = draw_gains(cfg, cond, 0.0, False, n, rng)
    gn = draw_gains(cfg, cond, 0.0, False, n, rng)
    f = cue_functions(cfg)[cue]
    with np.errstate(invalid="ignore"):
        base_n = f(gn)
        if not np.all(np.isfinite(base_n)):
            return float("nan")
        t = cfg.target_index

        def pc(delta):
            s = gs.copy()
            s[:, -1, t] += delta
            return _pc(f(s) - base_n)

        if pc(1e-3) >= P_TARGET:
            return 0.0
        lo, hi = 0.0, 40.0
        if pc(hi) < P_TARGET:
            return float("inf")
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if pc(mid) < P_TARGET else (lo, mid)
    return 0.5 * (lo + hi)


@lru_cache(maxsize=64)
def _ceilings_cached(cfg: Config, task: str, kind: str, n: int, seed: int) -> Tuple:
    # the gains do not depend on the step, so neither do the ceilings
    cond = Condition(f"{task}_{kind}", task, 0.0, kind)
    return tuple((cue, ideal_threshold(cfg, cond, cue, n, seed)) for cue in cue_functions(cfg))


def ceilings(cfg: Config, cond: Condition, n: int = 20000, seed: int = 0) -> dict:
    """Ideal single-cue thresholds for one condition, and the lowest one that is not exact.

    `ceiling_db` is the smallest increment at which ANY shortcut reaches 79.4%: a threshold
    well below it means the listener used the surviving cue. Expressed in dL and in SRS.
    """
    rows = dict(_ceilings_cached(cfg, cond.task, cond.kind, n, seed))
    surviving = [k for k, v in rows.items() if v == 0.0]
    finite = {k: v for k, v in rows.items() if np.isfinite(v) and v > 0}
    lowest = min(finite, key=finite.get) if finite else None
    c = finite[lowest] if lowest else float("inf")
    return {"condition": cond.name, "task": cond.task, "per_cue_db": rows,
            "exact": surviving, "lowest_shortcut": lowest, "ceiling_db": c,
            "ceiling_srs_db": srs_db(c) if np.isfinite(c) else float("inf")}


# ----------------------------------------------------------------------------
# a listener with no ears, for exercising the pipeline and for power
# ----------------------------------------------------------------------------
MODES = ("coherence", "late", "null", "within", "fused")


@dataclass
class SimulatedListener:
    """A weighted-cue observer whose weight on timbre is its binding state.

    decision = b * (timbre + noise_t) + (1 - b) * (own + noise_o), per interval

    b is 1 for a listener who hears one object and 0 for one who hears the target as its own
    stream. The modes say how b depends on the stimulus:

      coherence  b is the model's target-to-figure coherence for that stimulus
      late       the same curve stretched to twice the shear: binding persists longer
      null       b = 0.5 everywhere: the shear changes nothing
      within     b = 0: always the target's own stream
      fused      b = 1: always the timbre

    Internal noise is in dB on each cue. The defaults put an unroved increment at threshold
    near 3.5 dB for timbre and 2.4 dB for the own stream, the order of profile-analysis and
    intensity-discrimination thresholds for brief tones.
    """
    cfg: Config
    mode: str = "coherence"
    noise_timbre_db: float = 3.0
    noise_own_db: float = 2.0
    lapse: float = 0.02
    seed: int = 0

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.rng = np.random.default_rng(self.seed)

    def binding(self, cond: Condition) -> float:
        if cond.kind == "alone":
            return 0.0
        if self.mode == "null":
            return 0.5
        if self.mode == "within":
            return 0.0
        if self.mode == "fused":
            return 1.0
        from .model import target_coherence
        if cond.kind == "capture":
            return target_coherence(self.cfg, cond)
        s = cond.step_ms / (2.0 if self.mode == "late" else 1.0)
        return target_coherence(self.cfg, Condition("tmp", cond.task, s, "figure"))

    def decision(self, gains: np.ndarray, b: float, rng: np.random.Generator) -> float:
        cz = cues(self.cfg, gains)
        own = cz["own"] + rng.normal(0.0, self.noise_own_db)
        if not np.isfinite(cz["timbre"]):
            return float(own)
        tim = cz["timbre"] + rng.normal(0.0, self.noise_timbre_db)
        return float(b * tim + (1.0 - b) * own)

    def respond(self, trial, cond: Condition, rng: Optional[np.random.Generator] = None) -> int:
        """Which interval (1 or 2) the listener picks.

        The runner passes a generator keyed on the trial's slot, so a simulated session that is
        interrupted and resumed answers exactly as an uninterrupted one would.
        """
        rng = self.rng if rng is None else rng
        if rng.random() < self.lapse:
            return int(rng.integers(1, 3))
        b = self.binding(cond) if cond.task != "none" else 0.5
        d1 = self.decision(trial.first.gains_db, b, rng)
        d2 = self.decision(trial.second.gains_db, b, rng)
        return 1 if d1 > d2 else 2
