"""The temporal-coherence prediction for jittered A-B pairs.

Two channels, so the published index applies directly: lambda2/lambda1 of the coherence matrix
(Elhilali et al. 2009, with the half-wave reading `tcoh.model` documents), 0 when the channels
move together and approaching 1 as they decouple.

Because the timing moves A and B together, it barely changes the prediction: scatter timing,
grid jitter and a regular grid all give the same rising curve, a little lower or higher. That is
what it should do: it randomises each channel's own rhythm and leaves their relation exactly
where the lag puts it. Averaged over `n_seq` random sequences drawn with the session's own
timing, since any one sequence is a single draw.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Sequence

import numpy as np

from tcoh.model import coherence_matrix

MODEL_FS = 1000.0


def _env(onsets, total, tone_ms, ramp_ms):
    e = np.zeros(total)
    w = int(round(tone_ms)); r = max(int(round(ramp_ms)), 1)
    sh = np.ones(w); sh[:r] = np.linspace(0, 1, r); sh[-r:] = np.linspace(1, 0, r)
    for o in onsets:
        i = int(round(o)); j = min(total, i + w)
        if 0 <= i < total:
            e[i:j] += sh[:j - i]
    return e


def _onsets(rng, timing, n_pairs, slot_ms, jitter_ms, min_ioi_ms, max_ioi_ms):
    if timing == "grid":
        return np.arange(n_pairs) * slot_ms + rng.uniform(-jitter_ms, jitter_ms, n_pairs)
    slack = (n_pairs + 1) * (slot_ms - min_ioi_ms)
    while True:
        r = np.arange(n_pairs) * min_ioi_ms + np.sort(rng.uniform(0.0, slack, n_pairs))
        if max_ioi_ms is None or np.diff(r).max() <= max_ioi_ms:
            return r


@lru_cache(maxsize=512)
def _index(lag_ms, slot_ms, tone_ms, ramp_ms, jitter_ms, n_pairs, n_seq, seed, variant,
           timing="grid", min_ioi_ms=None, max_ioi_ms=None):
    kw = dict(variant)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_seq):
        b = 200.0 + _onsets(rng, timing, n_pairs, slot_ms, jitter_ms, min_ioi_ms, max_ioi_ms)
        total = int(b[-1] + lag_ms + tone_ms + 300)
        e = np.stack([_env(b, total, tone_ms, ramp_ms), _env(b + lag_ms, total, tone_ms, ramp_ms)])
        vals.append(coherence_matrix(e, MODEL_FS, **kw).ratio)
    return float(np.mean(vals))


@lru_cache(maxsize=512)
def _index_screened(cfg, lag_ms, n_seq, seed, variant):
    from .stimulus import reference_onsets
    kw = dict(variant)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_seq):
        b = 200.0 + reference_onsets(cfg, rng) - cfg.lead_ms
        total = int(b[-1] + lag_ms + cfg.tone_ms + 300)
        e = np.stack([_env(b, total, cfg.tone_ms, cfg.ramp_ms),
                      _env(b + lag_ms, total, cfg.tone_ms, cfg.ramp_ms)])
        vals.append(coherence_matrix(e, MODEL_FS, **kw).ratio)
    return float(np.mean(vals))


def lag_index(cfg, lag_ms: float, n_seq: int = 40, n_pairs: int = 12, seed: int = 0,
              **variant) -> float:
    """Mean lambda2/lambda1 at this lag. A longer sequence than the trial, for a stable estimate."""
    timing = getattr(cfg, "timing", "grid")
    if timing == "scatter" and (getattr(cfg, "regular_tol_ms", None) is not None
                                or getattr(cfg, "pair_clear_ms", None) is not None):
        # the session's own screened draws, lengthened to n_pairs tokens
        return _index_screened(cfg.replace(n_pairs=n_pairs), float(lag_ms), n_seq, seed,
                               tuple(sorted(variant.items())))
    g = None
    if timing == "scatter":
        from .config import scatter_geometry
        g = scatter_geometry(cfg)["min_ioi_ms"]
    return _index(float(lag_ms), cfg.slot_ms, cfg.tone_ms, cfg.ramp_ms, cfg.pair_jitter_ms,
                  n_pairs, n_seq, seed, tuple(sorted(variant.items())), timing, g,
                  getattr(cfg, "max_ioi_ms", None) if timing == "scatter" else None)


VARIANTS: Dict[str, dict] = {
    "default": {},
    "trim_startup": {"trim_s": 0.6},
    "magnitude": {"rectify": "magnitude"},
    "loose_significance": {"significance": 0.05},
    "tight_significance": {"significance": 0.4},
}


def prediction_band(cfg, lags: Sequence[float], n_seq: int = 20) -> dict:
    """The curve under several readings of an under-specified filter bank."""
    curves = {k: [lag_index(cfg, L, n_seq=n_seq, **v) for L in lags] for k, v in VARIANTS.items()}
    arr = np.array(list(curves.values()))
    dec = float(max(0.0, -np.diff(arr, axis=1).min())) if arr.shape[1] > 1 else 0.0
    return {"lags": list(map(float, lags)), "curves": curves, "mid": arr.mean(0).tolist(),
            "lo": arr.min(0).tolist(), "hi": arr.max(0).tolist(),
            "max_spread": float((arr.max(0) - arr.min(0)).max()),
            "max_decrease": dec, "all_monotone": dec <= 0.01}
