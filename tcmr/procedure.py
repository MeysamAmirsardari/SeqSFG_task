"""Adaptive tracks, the interleaved order, and the threshold fit. No sound, no screen.

Tracks follow Christiansen & Oxenham (2014): 2-down 1-up on the tone level (70.7% correct), steps
8, 4, 2 dB, shrinking after the 2nd and 4th reversal. Here every track runs a fixed number of
trials and all tracks are interleaved round by round (each round gives every track one trial in
random order), so conditions share fatigue and learning equally and the session length is fixed.

Thresholds: each condition's level at 70.7% correct from a maximum-likelihood fit of every trial
(cumulative Gaussian in dB, guess 1/3, lapse 0.02, one slope per listener shared by all
conditions); the mean of the last reversals at 2 dB is reported alongside.

The level range is limited (min_db to max_db). When the rule asks for a level outside it, the track
gets the limit instead and the step is marked (`Step.limit`). A track on which that happened
CEILING_CALLS times or more at the top is "at ceiling": its threshold is at or above the highest
level that can be played. In simulated exp1 tracks (32 trials, start 75, maximum 85 dB) that is
under 5% of tracks when the true threshold is 80 dB or lower, about half at 84 dB and about 90%
at 86 dB or higher. The fit itself stays within about 2 dB up to a true threshold of 86 dB and
loses precision above it, so a fitted threshold above max_db is not used as a number (analysis.py).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, stats

from .config import Condition, Config

GUESS = 1.0 / 3.0
CEILING_CALLS = 3        # this many calls above max_db in one track: the track is at ceiling


@dataclass
class Step:
    index: int
    level_db: float
    correct: bool
    step_db: float
    reversal: bool
    next_level_db: float
    limit: int = 0       # +1: the rule asked for more than max_db, -1: for less than min_db


class Track:
    def __init__(self, cfg: Config, cond: Condition, track_id: int):
        self.cfg, self.cond, self.id = cfg, cond, track_id
        self.level = float(cfg.start_db)
        self.run, self.direction, self.n_rev = 0, 0, 0
        self.steps: List[Step] = []
        self.reversal_levels: List[Tuple[float, float]] = []   # (level, step in force)

    @property
    def step_db(self) -> float:
        i = sum(1 for r in self.cfg.step_change_after if self.n_rev >= r)
        return self.cfg.steps_db[i]

    @property
    def finished(self) -> bool:
        return len(self.steps) >= self.cfg.trials_per_track

    def update(self, correct: bool) -> Step:
        cfg, lvl, st = self.cfg, self.level, self.step_db
        move = 0
        if correct:
            self.run += 1
            if self.run >= cfg.n_down:
                move, self.run = -1, 0
        else:
            move, self.run = +1, 0
        rev, limit = False, 0
        if move:
            if self.direction and move != self.direction:
                rev = True
                self.n_rev += 1
                self.reversal_levels.append((lvl, st))
            self.direction = move
            want = lvl + move * self.step_db
            self.level = float(np.clip(want, cfg.min_db, cfg.max_db))
            limit = int(want > cfg.max_db) - int(want < cfg.min_db)
        s = Step(len(self.steps), lvl, bool(correct), st, rev, self.level, limit)
        self.steps.append(s)
        return s

    @property
    def calls_above_max(self) -> int:
        """Trials after which the rule asked for a level above max_db (and got max_db)."""
        return sum(1 for s in self.steps if s.limit > 0)

    @property
    def at_ceiling(self) -> bool:
        return self.calls_above_max >= CEILING_CALLS

    @classmethod
    def replay(cls, cfg, cond, track_id, responses: Sequence[bool]) -> "Track":
        t = cls(cfg, cond, track_id)
        for c in responses:
            t.update(bool(c))
        return t

    def reversal_threshold(self, n: int = 6) -> Optional[float]:
        final = [l for l, s in self.reversal_levels if s == self.cfg.steps_db[-1]]
        return float(np.mean(final[-n:])) if len(final) >= 2 else None


def make_tracks(cfg: Config) -> List[Track]:
    out = []
    for c in cfg.condition_list():
        for _ in range(c.tracks):
            out.append(Track(cfg, c, len(out)))
    return out


def make_order(cfg: Config, seed: int) -> List[int]:
    """Track id of every main trial: rounds, each a random permutation of all tracks."""
    rng = np.random.default_rng([seed, 41])
    n = cfg.n_tracks
    order: List[int] = []
    for _ in range(cfg.trials_per_track):
        perm = list(rng.permutation(n))
        if order and perm[0] == order[-1]:          # no track twice in a row across rounds
            perm[0], perm[-1] = perm[-1], perm[0]
        order += [int(p) for p in perm]
    return order


# ----------------------------------------------------------------------------------------------
def _p(level, mu, sigma, lapse):
    return GUESS + (1 - GUESS - lapse) * stats.norm.cdf((np.asarray(level) - mu) / sigma)


def fit(trials: Dict[str, Tuple[Sequence[float], Sequence[bool]]], target_p: float = 0.707,
        lapse: float = 0.02) -> dict:
    """Shared-slope fit across conditions: {condition: (levels, correct)} -> thresholds in dB."""
    names = [k for k, (l, c) in trials.items() if len(l) >= 8]
    if not names:
        return {"thresholds_db": {}, "sigma_db": None}
    L = [np.asarray(trials[k][0], float) for k in names]
    C = [np.asarray(trials[k][1], bool) for k in names]

    def nll(x):
        s = math.exp(x[-1])
        tot = 0.0
        for i in range(len(names)):
            p = np.clip(_p(L[i], x[i], s, lapse), 1e-9, 1 - 1e-9)
            tot -= np.sum(np.where(C[i], np.log(p), np.log(1 - p)))
        return tot
    x0 = [float(np.mean(l)) for l in L] + [math.log(4.0)]
    bounds = [(0.0, 100.0)] * len(names) + [(math.log(0.5), math.log(20.0))]
    r = optimize.minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    s = math.exp(r.x[-1])
    z = stats.norm.ppf((target_p - GUESS) / (1 - GUESS - lapse))
    return {"thresholds_db": {k: float(r.x[i] + s * z) for i, k in enumerate(names)},
            "sigma_db": s, "converged": bool(r.success)}


@dataclass
class SimListener:
    """3AFC observer whose tone threshold (70.7%) depends on the condition."""
    thresholds_db: Dict[str, float]
    sigma_db: float = 4.0
    lapse: float = 0.02

    def p(self, cond: str, level: float) -> float:
        z = stats.norm.ppf((0.707 - GUESS) / (1 - GUESS - self.lapse))
        mu = self.thresholds_db[cond] - self.sigma_db * z
        return float(_p(level, mu, self.sigma_db, self.lapse))
