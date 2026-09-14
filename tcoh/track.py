"""The adaptive track: an n-down 1-up rule on a multiplicative step, as in Elhilali et al. (2009).

The rule, quoted from the paper's Methods and implemented literally
--------------------------------------------------------------------
"the tracking variable, dT, was set to 20 ms. It was divided by a factor c after two
consecutive correct responses, and multiplied by that same factor c after each incorrect
response. The value of c was set to 4 at the beginning of the adaptive run; it was reduced to 2
after the first reversal in the direction of tracking (from decreasing to increasing), and to
sqrt(2) after a further two reversals. The procedure stopped after the sixth reversal with the
sqrt(2) step size. Threshold was computed as the geometric mean of dT at the last six reversal
points."

One discrepancy in that passage is worth flagging rather than quietly resolving. The same
paragraph calls the rule "three-down one-up" and says it "tracked the 79.4%-correct point",
but then describes stepping down after TWO consecutive correct responses, which converges on
70.7%. The two cannot both be true. This implementation follows the stated target: `n_down`
defaults to 3, which is the rule that converges on 79.4%, and the convergence point is computed
from `n_down` rather than hard-coded, so a run with `n_down=2` reports 70.7% and no comparison
is made across the two by accident.

What the class records, and why all of it
------------------------------------------
A threshold with no diagnostics is an assertion. Each track keeps every trial, every reversal,
every step-size change, and three things that can silently ruin a threshold:

* *ceiling time* -- trials spent pinned at `delta_max_ms`. The track cannot step up from there,
  so the rule is no longer the rule. Harmless before the first reversal, fatal inside the
  averaged ones, and `Track.audit` distinguishes the two.
* *floor time* -- the same at `delta_min_ms`, which means the listener is better than the
  procedure can measure and the threshold is an upper bound, not an estimate.
* *non-convergence* -- hitting `max_trials` before the final reversals. The threshold is
  reported as missing, not as whatever the track happened to be sitting on.

`simulate` runs the whole rule against a listener whose threshold is known, which is the only
way to find out what this procedure actually measures: its bias, its spread, how long it takes,
and how often it fails. That spread is also the input the power analysis needs, so it is
measured rather than assumed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Config
from .psychometric import PF, target_proportion


@dataclass
class TrackTrial:
    index: int
    delta_ms: float
    correct: bool
    step_index: int
    at_ceiling: bool
    at_floor: bool
    reversal: bool


class Track:
    """One adaptive run. Ask it for `delta`, hand it `update(correct)`, stop when `finished`."""

    def __init__(self, cfg: Config, condition: str, seed: int = 0):
        self.cfg, self.condition, self.seed = cfg, condition, seed
        self.delta = float(cfg.delta_start_ms)
        self.step_index = 0
        self.n_correct_run = 0
        self.direction = 0                   # -1 going down, +1 going up, 0 not yet moved
        self.trials: List[TrackTrial] = []
        self.reversals: List[Tuple[float, int]] = []   # (delta at the reversal, step index)
        self.finished = False
        self.stop_reason: Optional[str] = None

    # -- the rule ---------------------------------------------------------------
    @property
    def target_p(self) -> float:
        return target_proportion(self.cfg.n_down)

    @property
    def final_step(self) -> int:
        return len(self.cfg.step_factors) - 1

    def _reversals_at_final(self) -> int:
        return sum(1 for _, si in self.reversals if si == self.final_step)

    def update(self, correct: bool) -> None:
        if self.finished:
            raise RuntimeError("track already finished")
        cfg = self.cfg
        delta = self.delta
        at_ceil = delta >= cfg.delta_max_ms - 1e-12
        at_floor = delta <= cfg.delta_min_ms + 1e-12

        move = 0
        if correct:
            self.n_correct_run += 1
            if self.n_correct_run >= cfg.n_down:
                move, self.n_correct_run = -1, 0
        else:
            move, self.n_correct_run = +1, 0

        reversal = False
        if move != 0:
            if self.direction != 0 and move != self.direction:
                reversal = True
                self.reversals.append((delta, self.step_index))
                # the step size shrinks on a schedule counted in reversals, not in trials
                done = len(self.reversals)
                need = 0
                for i, n in enumerate(cfg.reversals_per_step):
                    need += n
                    if done == need and self.step_index == i:
                        self.step_index = i + 1
                        break
            self.direction = move

        self.trials.append(TrackTrial(len(self.trials), delta, bool(correct), self.step_index,
                                      at_ceil, at_floor, reversal))

        if move != 0:
            f = cfg.step_factors[self.step_index]
            self.delta = float(np.clip(delta * (f ** move), cfg.delta_min_ms, cfg.delta_max_ms))

        if self._reversals_at_final() >= cfg.n_final_reversals:
            self.finished, self.stop_reason = True, "converged"
        elif len(self.trials) >= cfg.max_trials_per_track:
            self.finished, self.stop_reason = True, "max_trials"

    # -- the result -------------------------------------------------------------
    def threshold_ms(self) -> Optional[float]:
        """Geometric mean of the last `n_final_reversals` reversals at the final step size.

        None when the track did not get there: a number produced by a track that never
        converged is not a threshold and is not reported as one.
        """
        final = [d for d, si in self.reversals if si == self.final_step]
        if len(final) < self.cfg.n_final_reversals:
            return None
        return float(np.exp(np.mean(np.log(final[-self.cfg.n_final_reversals:]))))

    def audit(self) -> dict:
        """Everything that could make the threshold above wrong, as numbers."""
        n = len(self.trials)
        final = [d for d, si in self.reversals if si == self.final_step]
        used = final[-self.cfg.n_final_reversals:] if len(final) >= self.cfg.n_final_reversals else []
        ceil_all = sum(t.at_ceiling for t in self.trials)
        floor_all = sum(t.at_floor for t in self.trials)
        # trials from the first averaged reversal onwards -- the ones the threshold depends on
        first_used = next((t.index for t in self.trials if t.reversal
                           and t.step_index == self.final_step
                           and t.delta_ms in used), None)
        tail = self.trials[first_used:] if first_used is not None else []
        return {
            "condition": self.condition, "n_trials": n, "n_reversals": len(self.reversals),
            "n_final_reversals": len(final), "stop_reason": self.stop_reason,
            "converged": self.stop_reason == "converged",
            "threshold_ms": self.threshold_ms(),
            "reversals_used_ms": [float(x) for x in used],
            "reversal_spread_log": float(np.std(np.log(used))) if len(used) > 1 else None,
            "at_ceiling_trials": ceil_all, "at_floor_trials": floor_all,
            "ceiling_inside_threshold": sum(t.at_ceiling for t in tail),
            "floor_inside_threshold": sum(t.at_floor for t in tail),
            "prop_correct": float(np.mean([t.correct for t in self.trials])) if n else None,
            "target_p": self.target_p,
        }

    def deltas_and_correct(self) -> Tuple[np.ndarray, np.ndarray]:
        return (np.array([t.delta_ms for t in self.trials]),
                np.array([t.correct for t in self.trials]))


# ----------------------------------------------------------------------------
# validating the procedure itself
# ----------------------------------------------------------------------------
def run_simulated(cfg: Config, pf: PF, rng: np.random.Generator, condition: str = "sim") -> Track:
    t = Track(cfg, condition)
    while not t.finished:
        t.update(bool(pf.respond(t.delta, rng)))
    return t


def simulate(cfg: Config, thresholds_ms: Sequence[float] = (2.0, 4.0, 8.0, 16.0, 24.0),
             sigmas: Sequence[float] = (0.35, 0.6, 0.9), lapse: float = 0.02,
             n_runs: int = 400, seed: int = 20260913) -> dict:
    """Does this rule recover a threshold it was given? Bias, spread, length, failure rate.

    The spread it reports -- the standard deviation of log threshold across repeated runs of
    the same simulated listener -- is the measurement noise of a single track, and is what the
    power analysis in `tcoh.analysis.power` consumes. Measuring it here means the power
    statement rests on this procedure rather than on a textbook figure for a different one.
    """
    rng = np.random.default_rng(seed)
    cells = {}
    for thr in thresholds_ms:
        for sg in sigmas:
            pf = PF(float(thr), float(sg), lapse)
            got, lens, fails = [], [], 0
            for _ in range(n_runs):
                t = run_simulated(cfg, pf, rng)
                lens.append(len(t.trials))
                v = t.threshold_ms()
                if v is None:
                    fails += 1
                else:
                    got.append(v)
            lg = np.log(got) if got else np.array([np.nan])
            cells[(float(thr), float(sg))] = {
                "true_ms": float(thr), "sigma": float(sg),
                "recovered_geomean_ms": float(np.exp(np.mean(lg))),
                "bias_log": float(np.mean(lg) - math.log(thr)),
                "bias_pct": float(100.0 * (math.exp(np.mean(lg) - math.log(thr)) - 1.0)),
                "sd_log": float(np.std(lg, ddof=1)) if len(got) > 1 else float("nan"),
                "sd_pct": float(100.0 * (math.exp(np.std(lg, ddof=1)) - 1.0)) if len(got) > 1 else float("nan"),
                "median_trials": float(np.median(lens)),
                "fail_rate": fails / n_runs,
            }
    worst_bias = max(abs(c["bias_log"]) for c in cells.values())
    return {"cells": {f"{k[0]:g}ms_s{k[1]:g}": v for k, v in cells.items()},
            "worst_abs_bias_log": worst_bias,
            "worst_abs_bias_pct": float(100.0 * (math.exp(worst_bias) - 1.0)),
            "median_sd_log": float(np.median([c["sd_log"] for c in cells.values()])),
            "median_trials": float(np.median([c["median_trials"] for c in cells.values()])),
            "max_fail_rate": max(c["fail_rate"] for c in cells.values()),
            "n_runs": n_runs, "target_p": target_proportion(cfg.n_down)}
