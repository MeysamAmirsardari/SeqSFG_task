"""Listeners with no ears, who look at the actual onsets of each trial.

They exist to test the STIMULUS, not to model people. If the design is right, a listener who
judges the tones' own timing must be bad at every lag and equally bad at all of them, and a
listener who uses the A-B lag must be good at small lags and worse at large ones. Every observer
decides from the onsets `build_trial` actually produced, so a cue that leaks into the stimulus by
accident shows up as a tone-alone observer doing better than it should.

pair      sees the A-B lag of every token, with timing noise that grows with the lag (a Weber
          fraction plus a floor). Picks the interval whose candidate token departs most from that
          interval's own median lag. With no A tone it falls back on `single`.
single    sees B only, and judges each candidate B against its own neighbours: against the
          grid fitted to the other tokens (grid timing), or against the tones either side of it
          (scatter timing, where there is no grid to fit).
tones     does the same for A and for B separately and goes with whichever looks more displaced:
          a listener hearing two streams and watching each for a hiccup.
boundary  an ideal observer of scatter timing, with perfect timing and exact knowledge of the
          minimum gap. Says "target" when either channel shows two tones closer than the
          minimum, which no standard interval can do, and guesses otherwise. No person can do
          this; it bounds what one tone's timing could ever give.

With a context (tpair.context) the token lags differ before and after the switch, and how the
listener forms the reference lag matters:

pair      takes the reference from the test tokens only (it knows where the switch is): the
          context changes nothing for it. The null for every context effect.
pair_all  takes the reference from every token that cannot be the target, context included. In
          a two-interval comparison a shifted reference shifts both intervals alike and cancels,
          so this listener shows no context effect either: a polluted reference is not, by
          itself, a cost.
change    does not know where the switch is: it listens for the largest change of the lag from
          one token to the next, anywhere. The switch is such a change, in both intervals, and
          when it runs the target's way it hides the target (the decoy).
persist   like `pair`, but its timing noise is set by a lag that still carries the context,
          decaying with the time since the switch (CONTEXT_TAU_MS): a bound context helps, a
          segregated one hurts. Illustrative: it exists to size the effect the design can see.
contrast  the same with the sign reversed: the context repels, as an adapting system would.

All are told which way B moves relative to A, as a listener in a block of one direction is.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import Config, conditions, scatter_geometry, shifts, timing_faults
from .stimulus import Interval, build_trial, token_lags

SINGLE_NOISE_MS = 8.0
PAIR_FLOOR_MS = 1.5
PAIR_WEBER = 0.15
CONTEXT_TAU_MS = 500.0
MODES = ("pair", "single", "tones", "boundary", "pair_all", "change", "persist", "contrast")


def tone_evidence(cfg: Config, x: np.ndarray, direction: float) -> float:
    """How far the most-displaced candidate tone of one channel looks moved in `direction`
    (+1 later, -1 earlier), judged against that channel alone. Noise-free; observers add their
    own, and the analysis uses it as is."""
    x = np.asarray(x, float)
    k = np.arange(cfg.n_pairs)
    cand = list(cfg.target_pairs)
    if cfg.timing == "grid":
        safe = [i for i in range(cfg.n_pairs) if i not in cand]
        phase = np.median(x[safe] - k[safe] * cfg.slot_ms)
        res = x - (k * cfg.slot_ms + phase)
        return float(max(direction * res[i] for i in cand))
    out = []
    for i in cand:
        if i == cfg.n_pairs - 1:
            typical = np.median(np.diff(x[:i]))
            out.append(direction * (x[i] - x[i - 1] - typical))
        else:
            out.append(direction * (x[i] - (x[i - 1] + x[i + 1]) / 2.0))
    return float(max(out))


def _moves(cfg: Config, tr) -> tuple:
    """Which way each channel moved in the target: (A, B), each -1, 0 or +1."""
    a, b = shifts(cfg, tr.delta_signed_ms)
    return float(np.sign(a)), float(np.sign(b))


def _tone_score(cfg: Config, iv: Interval, rng, channel: str, direction: float,
                noise: float) -> float:
    x = iv.b_onsets_ms if channel == "b" else iv.a_onsets_ms
    if not np.isfinite(x).all() or direction == 0:
        return -np.inf
    return tone_evidence(cfg, x + rng.normal(0, noise, x.size), direction)


def _pair_score(cfg: Config, iv: Interval, rng, sign: float = -1.0, mode: str = "pair") -> float:
    lag = token_lags(cfg, iv)
    test = iv.lag_ms if iv.context_lag_ms is not None else np.nanmedian(lag)
    judged = abs(test)
    if mode in ("persist", "contrast") and iv.context_lag_ms is not None:
        # the lag that sets the noise still carries the context, fading after the switch
        t_switch = np.nanmin(np.vstack([iv.a_onsets_ms, iv.b_onsets_ms])[:, cfg.n_context])
        t_now = max(np.nanmin([iv.a_onsets_ms[i], iv.b_onsets_ms[i]]) for i in cfg.target_pairs)
        w = np.exp(-max(t_now - t_switch, 0.0) / CONTEXT_TAU_MS)
        k = 1.0 if mode == "persist" else -1.0
        judged = max(abs(test + k * (iv.context_lag_ms - test) * w), 0.0)
    sd = PAIR_FLOOR_MS + PAIR_WEBER * judged
    lag = lag + rng.normal(0, sd, lag.size)
    s = sign if cfg.a_leads else -sign                          # an earlier B shrinks a trailing lag
    if mode == "change":
        return float(np.max(s * np.diff(lag)))
    others = [i for i in range(cfg.n_pairs) if i not in cfg.target_pairs]
    if iv.context_lag_ms is not None and mode != "pair_all":
        others = [i for i in others if i >= cfg.n_context]
    ref = np.median([lag[i] for i in others]) if others else test
    return float(max(s * (lag[i] - ref) for i in cfg.target_pairs))


def _violates(cfg: Config, iv: Interval) -> bool:
    """Does either channel, on its own, show timing no untouched interval could have: two tones
    closer than the minimum gap, or anything the timing screen throws away?"""
    g = scatter_geometry(cfg)["min_ioi_ms"]
    for x in (iv.a_onsets_ms, iv.b_onsets_ms):
        if np.isfinite(x).all() and ((np.diff(x) < g - 1e-6).any() or timing_faults(cfg, x)):
            return True
    return False


def choose(cfg: Config, tr, mode: str, rng, noise: float = SINGLE_NOISE_MS) -> int:
    """Which interval this observer says held the change (1 or 2)."""
    ivs = (tr.first, tr.second)
    sign = 1.0 if tr.delta_signed_ms > 0 else -1.0
    da, db = _moves(cfg, tr)
    if mode in ("pair", "pair_all", "change", "persist", "contrast") and tr.first.kind != "b_alone":
        s = [_pair_score(cfg, iv, rng, sign, mode) for iv in ivs]
    elif mode == "boundary" and cfg.timing == "scatter":
        v = [_violates(cfg, iv) for iv in ivs]
        if v[0] != v[1]:
            return 1 if v[0] else 2
        return int(rng.integers(1, 3))
    elif mode in ("tones", "boundary"):
        s = [max(_tone_score(cfg, iv, rng, "a", da, noise),
                 _tone_score(cfg, iv, rng, "b", db, noise)) for iv in ivs]
    else:
        s = [_tone_score(cfg, iv, rng, "b", db, noise) for iv in ivs]
    if s[0] == s[1]:
        return int(rng.integers(1, 3))
    return 1 if s[0] > s[1] else 2


def p_correct(cfg: Config, cond, delta_ms: float, mode: str, n: int = 2000,
              seed: int = 0, noise: float = SINGLE_NOISE_MS) -> float:
    rng = np.random.default_rng(seed)
    ok = 0
    for _ in range(n):
        tr = build_trial(cfg, cond, delta_ms, rng)
        ok += choose(cfg, tr, mode, rng, noise) == tr.target_position
    return ok / n


def threshold(cfg: Config, cond, mode: str, n: int = 1500, seed: int = 0,
              target: float = 0.7937, noise: float = SINGLE_NOISE_MS) -> float:
    """The 79.4% point of this observer in this condition, by interpolation; inf if it never
    gets there below the ceiling."""
    deltas = np.geomspace(cfg.delta_min_ms * 2, cfg.ceiling_for(cond.kind), 16)
    pc = np.array([p_correct(cfg, cond, d, mode, n, seed + i, noise) for i, d in enumerate(deltas)])
    pc = np.maximum.accumulate(pc)
    if pc[-1] < target:
        return float("inf")
    return float(np.exp(np.interp(target, pc, np.log(deltas))))


def single_channel_threshold(cfg: Config, n: int = 1500) -> float:
    """What a listener ignoring A can reach -- the floor the timing has to keep high."""
    c = next(x for x in conditions(cfg) if x.kind == "pair")             # first direction, lag
    return threshold(cfg, c, "single", n=n)
