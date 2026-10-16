"""The temporal-coherence prediction for jittered A-B pairs.

Two channels, so the published index applies directly: lambda2/lambda1 of the coherence matrix
(Elhilali et al. 2009, with the half-wave reading `tcoh.model` documents), 0 when the channels
move together and approaching 1 as they decouple.

Because the jitter moves A and B together, it barely changes the prediction -- 0.794 at a
100 ms lag without jitter, 0.804 with it. That is what it should do: it randomises each
channel's own rhythm and leaves their relation exactly where the lag puts it. Averaged over
`n_seq` random sequences, since any one sequence is a single draw of the jitter.
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


@lru_cache(maxsize=512)
def _index(lag_ms, slot_ms, tone_ms, ramp_ms, jitter_ms, n_pairs, n_seq, seed, variant):
    kw = dict(variant)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_seq):
        b = 200.0 + np.arange(n_pairs) * slot_ms + rng.uniform(-jitter_ms, jitter_ms, n_pairs)
        total = int(b[-1] + lag_ms + tone_ms + 300)
        e = np.stack([_env(b, total, tone_ms, ramp_ms), _env(b + lag_ms, total, tone_ms, ramp_ms)])
        vals.append(coherence_matrix(e, MODEL_FS, **kw).ratio)
    return float(np.mean(vals))


def lag_index(cfg, lag_ms: float, n_seq: int = 40, n_pairs: int = 12, seed: int = 0,
              **variant) -> float:
    """Mean lambda2/lambda1 at this lag. A longer sequence than the trial, for a stable estimate."""
    return _index(float(lag_ms), cfg.slot_ms, cfg.tone_ms, cfg.ramp_ms, cfg.pair_jitter_ms,
                  n_pairs, n_seq, seed, tuple(sorted(variant.items())))


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
