"""Psychometric functions: the simulated listener, and the fit that turns trials into a threshold.

Two things live here.

`PF` is a psychometric function on log delta -- a cumulative Gaussian between a fixed guess
rate (0.5, because the task is two-interval forced choice) and a lapse rate. It is used to
*simulate* a listener when the tracker, the design or the power of the experiment is being
checked, and to *fit* one when real trials come back.

`fit` is a maximum-likelihood fit of that function to every trial a condition produced,
including the ones an adaptive track spends far from threshold. Averaging reversals, which is
what the published procedure does and what `track.threshold` reports, throws most of the data
away and gives a number with no confidence interval. Fitting all of it gives both, plus a
slope, plus a lapse rate -- and if the two disagree that is worth knowing before anything is
concluded. The report prints both, always, side by side.

Why a cumulative Gaussian on log delta rather than on delta
------------------------------------------------------------
Because the tracked variable is multiplicative: the step rule divides and multiplies, the
published threshold is a geometric mean, and thresholds that range from 2 ms to 20 ms across
conditions are not plausibly homoscedastic on a linear axis. Everything downstream -- the
coherence index, the bootstrap, the model comparison -- is therefore in log units too.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, stats

GUESS = 0.5                 # two-interval forced choice
TARGET_P = 0.5 ** (1.0 / 3.0)   # what 3-down 1-up converges on: 0.7937


def target_proportion(n_down: int) -> float:
    """The proportion correct an n-down 1-up rule converges on."""
    return 0.5 ** (1.0 / n_down)


def _logistic(z: float) -> float:
    """1/(1+exp(-z)), without overflowing.

    Nelder-Mead is unconstrained, so it will happily try z = -750 on the way to an optimum, and
    the textbook form raises OverflowError there rather than returning the 0 it is converging
    on. Splitting on the sign keeps every exponent negative.
    """
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass(frozen=True)
class PF:
    """P(correct) = guess + (1 - guess - lapse) * Phi((log d - m) / sigma)."""
    threshold_ms: float         # the delta at which P(correct) = `at`
    sigma: float                # slope, in natural-log units of delta
    lapse: float = 0.02
    at: float = TARGET_P
    guess: float = GUESS

    @property
    def m(self) -> float:
        span = 1.0 - self.guess - self.lapse
        z = stats.norm.ppf(np.clip((self.at - self.guess) / span, 1e-9, 1 - 1e-9))
        return math.log(self.threshold_ms) - self.sigma * z

    def p_correct(self, delta_ms) -> np.ndarray:
        d = np.clip(np.asarray(delta_ms, dtype=float), 1e-12, None)
        span = 1.0 - self.guess - self.lapse
        return self.guess + span * stats.norm.cdf((np.log(d) - self.m) / self.sigma)

    def respond(self, delta_ms, rng: np.random.Generator) -> np.ndarray:
        return rng.random(np.shape(delta_ms)) < self.p_correct(delta_ms)

    def threshold_at(self, p: float) -> float:
        span = 1.0 - self.guess - self.lapse
        z = stats.norm.ppf(np.clip((p - self.guess) / span, 1e-9, 1 - 1e-9))
        return float(math.exp(self.m + self.sigma * z))


SIGMA_BOUNDS = (0.08, 2.5)
# A psychometric function with sigma below about 0.08 log units is a step: correct below
# threshold, correct above, nothing in between. No listener is that sharp, and the only way the
# fit reaches there is a degenerate optimum -- which adaptive data invites, because the trials
# pile up at one delta and the slope is barely constrained. Bounding it keeps the threshold
# estimate sane; `Fit.reliable` says whether to believe the slope at all.


@dataclass(frozen=True)
class Fit:
    threshold_ms: float
    sigma: float
    lapse: float
    n_trials: int
    loglik: float
    converged: bool
    at: float
    ci: Optional[Tuple[float, float]] = None      # bootstrap CI on the threshold
    deviance: Optional[float] = None              # fit vs saturated, for goodness of fit
    deviance_p: Optional[float] = None
    n_distinct: int = 0
    delta_range: float = 1.0                      # max delta / min delta actually presented
    sigma_at_bound: bool = False

    @property
    def reliable(self) -> bool:
        """Is the SLOPE worth quoting?

        An adaptive track concentrates its trials near threshold by design, so it can pin a
        threshold down while saying almost nothing about the slope. The fit is called reliable
        only when the deltas presented span at least a factor of three over at least six
        distinct values and the slope did not run into its bound.
        """
        return (not self.sigma_at_bound) and self.n_distinct >= 6 and self.delta_range >= 3.0

    def as_dict(self) -> dict:
        return {"threshold_ms": self.threshold_ms, "sigma": self.sigma, "lapse": self.lapse,
                "n_trials": self.n_trials, "converged": self.converged, "at": self.at,
                "ci_lo": None if self.ci is None else self.ci[0],
                "ci_hi": None if self.ci is None else self.ci[1],
                "deviance": self.deviance, "deviance_p": self.deviance_p,
                "n_distinct": self.n_distinct, "delta_range": self.delta_range,
                "sigma_at_bound": self.sigma_at_bound, "reliable": self.reliable}


def fit(deltas_ms: Sequence[float], correct: Sequence[bool], at: float = TARGET_P,
        lapse_max: float = 0.1, n_boot: int = 0, seed: int = 0,
        guess: float = GUESS) -> Optional[Fit]:
    """Maximum-likelihood fit to every trial. None if there is nothing to fit.

    The lapse rate is free but bounded: leaving it at zero biases the slope, and leaving it
    unbounded lets a handful of lucky trials at the bottom of a track buy an implausible one.
    `lapse_max` is the usual 0.1.
    """
    d = np.asarray(deltas_ms, dtype=float)
    y = np.asarray(correct, dtype=bool)
    ok = np.isfinite(d) & (d > 0)
    d, y = d[ok], y[ok]
    if d.size < 8 or len(np.unique(d)) < 3 or y.all() or (~y).all():
        return None
    x = np.log(d)

    lo_s, hi_s = SIGMA_BOUNDS

    def _sigma(z):
        return lo_s + (hi_s - lo_s) * _logistic(z)

    def nll(theta):
        m, z, lg = theta
        if not (np.isfinite(m) and np.isfinite(z) and np.isfinite(lg)):
            return np.inf
        s = _sigma(z)
        lam = lapse_max * _logistic(lg)
        p = guess + (1.0 - guess - lam) * stats.norm.cdf((x - m) / s)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -float(np.sum(np.log(np.where(y, p, 1.0 - p))))

    def _z(sig):
        q = (sig - lo_s) / (hi_s - lo_s)
        return math.log(q / (1.0 - q))


    best, best_v = None, np.inf
    for m0 in (np.median(x), np.percentile(x, 25), np.percentile(x, 75)):
        for s0 in (0.3, 0.7, 1.2):
            r = optimize.minimize(nll, [m0, _z(s0), -2.0], method="Nelder-Mead",
                                  options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
            if r.fun < best_v:
                best, best_v = r, r.fun
    m, s = best.x[0], _sigma(best.x[1])
    lam = lapse_max * _logistic(best.x[2])
    thr = float(math.exp(m + s * stats.norm.ppf(np.clip((at - guess) / (1 - guess - lam), 1e-9, 1 - 1e-9))))

    dev, dev_p = _deviance(x, y, m, s, lam, guess)
    uniq = np.unique(np.round(d, 6))
    rng_ratio = float(uniq.max() / uniq.min()) if uniq.min() > 0 else 1.0
    at_bound = bool(s <= lo_s * 1.02 or s >= hi_s * 0.98)
    ci = None
    if n_boot:
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, d.size, d.size)
            f = fit(d[idx], y[idx], at=at, lapse_max=lapse_max, n_boot=0, guess=guess)
            if f is not None and np.isfinite(f.threshold_ms):
                vals.append(f.threshold_ms)
        if len(vals) > 20:
            ci = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    return Fit(thr, s, lam, int(d.size), -best_v, bool(best.success), at, ci, dev, dev_p,
               int(uniq.size), rng_ratio, at_bound)


def _deviance(x, y, m, s, lam, guess) -> Tuple[float, float]:
    """Goodness of fit, binned by unique delta. A tiny p here means the shape is wrong."""
    p = guess + (1.0 - guess - lam) * stats.norm.cdf((x - m) / s)
    dev, df = 0.0, 0
    for u in np.unique(x):
        sel = x == u
        n, k, ph = int(sel.sum()), int(y[sel].sum()), float(p[sel][0])
        if n == 0:
            continue
        for obs, exp in ((k, n * ph), (n - k, n * (1 - ph))):
            if obs > 0 and exp > 0:
                dev += 2.0 * obs * math.log(obs / exp)
        df += 1
    df = max(df - 3, 1)
    return float(dev), float(stats.chi2.sf(dev, df))
