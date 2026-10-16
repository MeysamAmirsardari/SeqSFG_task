"""Two listeners with no ears, who look at the actual onsets of each trial.

They exist to test the STIMULUS, not to model people. If the design is right, a listener who
uses only B must be bad at every lag and equally bad at all of them, and a listener who uses the
A-B lag must be good at small lags and worse at large ones. Both observers decide from the onsets
`build_trial` actually produced, so a cue that leaks into the stimulus by accident shows up as a
single-channel observer doing better than it should.

single  -- sees B only. Estimates the 3-Hz grid from the pairs that can never carry the target,
           then picks the interval whose candidate B tone is most early against it.
pair    -- sees the A-B lag of every pair, with timing noise that grows with the lag (a Weber
           fraction plus a floor). Picks the interval whose candidate pair departs most from that
           interval's own median lag. With no A tone it falls back on the single strategy.
"""
from __future__ import annotations

import numpy as np

from .config import Config, conditions
from .stimulus import Interval, build_trial

SINGLE_NOISE_MS = 8.0
PAIR_FLOOR_MS = 1.5
PAIR_WEBER = 0.15


def _single_score(cfg: Config, iv: Interval, rng) -> float:
    b = iv.b_onsets_ms + rng.normal(0, SINGLE_NOISE_MS, iv.b_onsets_ms.size)
    k = np.arange(cfg.n_pairs)
    safe = [i for i in range(cfg.n_pairs) if i not in cfg.target_pairs]
    phase = np.median(b[safe] - k[safe] * cfg.slot_ms)
    res = b - (k * cfg.slot_ms + phase)
    return float(-min(res[i] for i in cfg.target_pairs))       # earliness of the candidates


def _pair_score(cfg: Config, iv: Interval, rng) -> float:
    lag = iv.a_onsets_ms - iv.b_onsets_ms
    sd = PAIR_FLOOR_MS + PAIR_WEBER * np.abs(np.nanmedian(lag))
    lag = lag + rng.normal(0, sd, lag.size)
    ref = np.median([lag[i] for i in range(cfg.n_pairs) if i not in cfg.target_pairs])
    return float(max(lag[i] - ref for i in cfg.target_pairs))  # B early -> lag grows


def choose(cfg: Config, tr, mode: str, rng) -> int:
    """Which interval this observer says held the displacement (1 or 2)."""
    if mode == "pair" and tr.first.kind == "pair":
        s = [_pair_score(cfg, iv, rng) for iv in (tr.first, tr.second)]
    else:
        s = [_single_score(cfg, iv, rng) for iv in (tr.first, tr.second)]
    return 1 if s[0] > s[1] else 2


def p_correct(cfg: Config, cond, delta_ms: float, mode: str, n: int = 2000,
              seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    ok = 0
    for _ in range(n):
        tr = build_trial(cfg, cond, delta_ms, rng)
        ok += choose(cfg, tr, mode, rng) == tr.target_position
    return ok / n


def threshold(cfg: Config, cond, mode: str, n: int = 1500, seed: int = 0,
              target: float = 0.7937) -> float:
    """The 79.4% point of this observer in this condition, by interpolation."""
    deltas = np.geomspace(cfg.delta_min_ms * 2, cfg.delta_max_ms, 16)
    pc = np.array([p_correct(cfg, cond, d, mode, n, seed + i) for i, d in enumerate(deltas)])
    pc = np.maximum.accumulate(pc)
    if pc[-1] < target:
        return float("inf")
    return float(np.exp(np.interp(target, pc, np.log(deltas))))


def single_channel_threshold(cfg: Config, n: int = 1500) -> float:
    """What a listener ignoring A can reach -- the floor the jitter has to keep high."""
    c = next(x for x in conditions(cfg) if x.kind == "pair" and x.lag_ms == 0.0)
    return threshold(cfg, c, "single", n=n)
