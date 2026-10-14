"""The adaptive track: 3-down 1-up on SRS in dB, with additive steps (Levitt 1971).

Additive steps in SRS are multiplicative steps in dA/A, so this is the same rule as
`tcoh.track.Track` running on a log axis, and `verify` checks that the two give identical
reversals for identical responses. It is a separate class only so that every number it reports
is in dB of SRS rather than in a variable called `delta_ms`.

The schedule, and why
---------------------
4 dB steps until the first reversal, then 2 dB; the track stops after 4 reversals at 2 dB and
the threshold is their mean. Simulated against listeners with psychometric slopes from shallow to
steep (d' proportional to (dA/A)^0.7 ... ^1.5), this takes a median of 27 trials and recovers the
threshold with a spread of 3.1 dB and a bias under 1 dB. Six final reversals cost 9 more trials
for a spread of 2.8 dB, and a maximum-likelihood fit to all trials is no better than the
reversal mean at this slope: the psychometric function for a level increment is shallow in SRS
(d' doubles per 6 dB), and that, not the rule, sets the precision. Two sessions per listener
halve the variance; a longer track would not.

What the class records
----------------------
Every trial, every reversal, and the three things that can silently ruin a threshold: time at the
ceiling (the track cannot step up, so the rule is no longer the rule), time at the floor, and
non-convergence. A track whose averaged reversals touch the ceiling is a lower bound, not an
estimate, and `audit` says so.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import Config, delta_db


@dataclass
class TrackTrial:
    index: int
    srs_db: float
    correct: bool
    step_index: int
    at_ceiling: bool
    at_floor: bool
    reversal: bool


class Track:
    """One adaptive run. Ask for `srs`, hand it `update(correct)`, stop when `finished`."""

    def __init__(self, cfg: Config, condition: str, seed: int = 0):
        self.cfg, self.condition, self.seed = cfg, condition, seed
        self.srs = float(cfg.srs_start_db)
        self.step_index = 0
        self.n_correct_run = 0
        self.direction = 0
        self.trials: List[TrackTrial] = []
        self.reversals: List[Tuple[float, int]] = []
        self.finished = False
        self.stop_reason: Optional[str] = None

    @property
    def target_p(self) -> float:
        return 0.5 ** (1.0 / self.cfg.n_down)

    @property
    def final_step(self) -> int:
        return len(self.cfg.step_sizes_db) - 1

    def _final(self) -> List[float]:
        return [x for x, si in self.reversals if si == self.final_step]

    def update(self, correct: bool) -> None:
        if self.finished:
            raise RuntimeError("track already finished")
        cfg = self.cfg
        x = self.srs
        at_ceil = x >= cfg.srs_max_db - 1e-9
        at_floor = x <= cfg.srs_min_db + 1e-9
        move = 0
        if correct:
            self.n_correct_run += 1
            if self.n_correct_run >= cfg.n_down:
                move, self.n_correct_run = -1, 0
        else:
            move, self.n_correct_run = +1, 0
        reversal = False
        if move:
            if self.direction and move != self.direction:
                reversal = True
                self.reversals.append((x, self.step_index))
                need = 0
                for i, n in enumerate(cfg.reversals_per_step):
                    need += n
                    if len(self.reversals) == need and self.step_index == i:
                        self.step_index = i + 1
                        break
            self.direction = move
        self.trials.append(TrackTrial(len(self.trials), x, bool(correct), self.step_index,
                                      at_ceil, at_floor, reversal))
        if move:
            self.srs = float(np.clip(x + move * cfg.step_sizes_db[self.step_index],
                                     cfg.srs_min_db, cfg.srs_max_db))
        if len(self._final()) >= cfg.n_final_reversals:
            self.finished, self.stop_reason = True, "converged"
        elif len(self.trials) >= cfg.max_trials_per_track:
            self.finished, self.stop_reason = True, "max_trials"

    def threshold_db(self) -> Optional[float]:
        """Mean SRS at the last `n_final_reversals` reversals at the final step; None if the
        track never got there."""
        f = self._final()
        if len(f) < self.cfg.n_final_reversals:
            return None
        return float(np.mean(f[-self.cfg.n_final_reversals:]))

    def audit(self) -> dict:
        n = len(self.trials)
        f = self._final()
        used = f[-self.cfg.n_final_reversals:] if len(f) >= self.cfg.n_final_reversals else []
        first_used = None
        if used:
            k = len(self.reversals) - len(used)
            idx = [t.index for t in self.trials if t.reversal]
            first_used = idx[k] if k < len(idx) else None
        tail = self.trials[first_used:] if first_used is not None else []
        thr = self.threshold_db()
        return {
            "condition": self.condition, "n_trials": n, "n_reversals": len(self.reversals),
            "n_final_reversals": len(f), "stop_reason": self.stop_reason,
            "converged": self.stop_reason == "converged",
            "threshold_srs_db": thr,
            "threshold_delta_db": None if thr is None else delta_db(thr),
            "reversals_used_srs_db": [float(x) for x in used],
            "reversal_sd_db": float(np.std(used)) if len(used) > 1 else None,
            "at_ceiling_trials": sum(t.at_ceiling for t in self.trials),
            "at_floor_trials": sum(t.at_floor for t in self.trials),
            "ceiling_inside_threshold": sum(t.at_ceiling for t in tail),
            "floor_inside_threshold": sum(t.at_floor for t in tail),
            "ceiling_limited": bool(used) and max(used) >= self.cfg.srs_max_db - 1e-9,
            "prop_correct": float(np.mean([t.correct for t in self.trials])) if n else None,
            "target_p": self.target_p,
        }


# ----------------------------------------------------------------------------
def p_correct_model(srs, threshold_srs: float, beta: float = 1.0, lapse: float = 0.02,
                    at: float = 0.5 ** (1 / 3)) -> np.ndarray:
    """2AFC proportion correct when d' grows as (dA/A)^beta, scaled to `at` at threshold."""
    from scipy.stats import norm
    span = 1.0 - lapse
    d_thr = math.sqrt(2.0) * norm.ppf((at - lapse / 2.0) / span)
    d = d_thr * 10.0 ** (beta * (np.asarray(srs, dtype=float) - threshold_srs) / 20.0)
    return lapse / 2.0 + span * norm.cdf(d / math.sqrt(2.0))


def simulate(cfg: Config, thresholds_srs: Sequence[float] = (-12.0, -8.0, -4.0, 0.0, 4.0),
             betas: Sequence[float] = (0.7, 1.0, 1.5), n_runs: int = 400,
             seed: int = 20261014) -> dict:
    """Bias, spread, length and failure rate of the rule against known listeners."""
    rng = np.random.default_rng(seed)
    cells = {}
    for thr in thresholds_srs:
        for b in betas:
            got, lens, fails = [], [], 0
            for _ in range(n_runs):
                t = Track(cfg, "sim")
                while not t.finished:
                    t.update(bool(rng.random() < p_correct_model(t.srs, thr, b)))
                lens.append(len(t.trials))
                v = t.threshold_db()
                if v is None:
                    fails += 1
                else:
                    got.append(v)
            g = np.asarray(got)
            cells[f"{thr:+g}dB_b{b:g}"] = {
                "true_srs_db": thr, "beta": b,
                "bias_db": float(g.mean() - thr) if len(g) else float("nan"),
                "sd_db": float(g.std(ddof=1)) if len(g) > 1 else float("nan"),
                "median_trials": float(np.median(lens)),
                "p90_trials": float(np.percentile(lens, 90)),
                "fail_rate": fails / n_runs}
    v = list(cells.values())
    return {"cells": cells,
            "worst_abs_bias_db": max(abs(c["bias_db"]) for c in v),
            "median_sd_db": float(np.median([c["sd_db"] for c in v])),
            "median_trials": float(np.median([c["median_trials"] for c in v])),
            "p90_trials": float(np.median([c["p90_trials"] for c in v])),
            "max_fail_rate": max(c["fail_rate"] for c in v), "n_runs": n_runs}
