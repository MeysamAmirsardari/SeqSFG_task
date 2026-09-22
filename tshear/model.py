"""What the temporal-coherence model predicts for a sheared figure of N tones.

lambda2/lambda1 does not generalise
------------------------------------
Elhilali et al. (2009) read segregation off a two-channel coherence matrix as the ratio of its
two eigenvalues: 0 when the channels move together and 1 when they are independent. With four
channels that ratio only asks whether there is a SECOND object; it says nothing about whether
there are two, three or four, and it is not even monotone in the shear -- measured here it
rises to 0.654 at 70% shear, falls to 0.615 at 80%, and rises again. A design whose axis is
built on a non-monotone index cannot interpret an ordered behavioural result.

The participation ratio does
-----------------------------
    PR = (sum_i lambda_i)^2 / sum_i lambda_i^2

is the standard effective-rank of a spectrum: 1 when one eigenvalue carries everything, N when
all N are equal. Read on the coherence matrix it is the EFFECTIVE NUMBER OF OBJECTS, and
normalised as (PR - 1) / (N - 1) it runs 0 to 1 like the published index. For N = 2 it is a
monotone function of lambda2/lambda1 -- at lambda2/lambda1 = r it equals (1+r)^2/(1+r^2) - 1 --
so it is a strict generalisation of the published reading rather than a different quantity.

Measured on the default stimulus it runs from exactly 1.000 effective objects at step 0 to
3.839 at step 100%, monotonically. That is the axis this task is built on.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from tcoh.model import coherence_matrix, filter_bank

MODEL_FS = 1000.0          # the envelope-domain sample rate the rate filters are defined at


def figure_envelopes(cfg, step_pct: float, n_rep: Optional[int] = None,
                     fs: float = MODEL_FS) -> np.ndarray:
    """One envelope per tone, shape (n_tones, n_samples), at the model's own sample rate."""
    n_rep = n_rep if n_rep is not None else max(cfg.n_repeats, 6)
    step = cfg.step_ms(step_pct)
    total = int(round((n_rep * cfg.period_ms + (cfg.n_tones - 1) * step
                       + cfg.tone_ms + 100.0) * fs / 1000.0))
    w = int(round(cfg.tone_ms * fs / 1000.0))
    r = max(int(round(cfg.ramp_ms * fs / 1000.0)), 1)
    shape = np.ones(w)
    shape[:r] = np.linspace(0.0, 1.0, r)
    shape[-r:] = np.linspace(1.0, 0.0, r)
    env = np.zeros((cfg.n_tones, total))
    for k in range(cfg.n_tones):
        for rep in range(n_rep):
            i = int(round((rep * cfg.period_ms + k * step) * fs / 1000.0))
            j = min(total, i + w)
            env[k, i:j] += shape[: j - i]
    return env


def effective_objects(cfg, step_pct: float, n_rep: Optional[int] = None,
                      **kw) -> dict:
    """The effective number of objects the model sees, and the 0-1 normalisation of it."""
    c = coherence_matrix(figure_envelopes(cfg, step_pct, n_rep), MODEL_FS, **kw)
    ev = np.clip(np.sort(np.abs(np.asarray(c.eigenvalues)))[::-1], 0.0, None)
    tot = float(ev.sum())
    pr = float(tot ** 2 / np.sum(ev ** 2)) if tot > 0 else 1.0
    n = cfg.n_tones
    return {"step_pct": float(step_pct), "eigenvalues": ev.tolist(),
            "participation_ratio": pr, "normalised": float((pr - 1.0) / (n - 1.0)),
            "lambda2_over_lambda1": float(c.ratio)}


def predicted_curve(cfg, pcts: Optional[Sequence[float]] = None, **kw) -> Dict[float, dict]:
    pcts = list(cfg.step_pcts if pcts is None else pcts)
    return {float(p): effective_objects(cfg, p, **kw) for p in pcts}


def prediction_band(cfg, pcts: Optional[Sequence[float]] = None) -> dict:
    """The curve under several defensible readings of the filter bank, and what survives them.

    Same discipline as the two-tone task: a prediction that holds for one arbitrary setting of
    an under-specified filter bank is not a prediction, so the spread across readings is
    reported and the claim is made at the strength the model actually supports.
    """
    variants = {"default": {}, "halfwave_trim": {"trim_s": 0.6},
                "magnitude": {"rectify": "magnitude"},
                "loose_significance": {"significance": 0.05},
                "tight_significance": {"significance": 0.4}}
    pcts = list(cfg.step_pcts if pcts is None else pcts)
    curves = {}
    for name, over in variants.items():
        try:
            curves[name] = [effective_objects(cfg, p, **over)["normalised"] for p in pcts]
        except TypeError:
            continue
    arr = np.array([curves[k] for k in curves])
    dec = float(max(0.0, -np.diff(arr, axis=1).min())) if arr.shape[1] > 1 else 0.0
    return {"pcts": [float(p) for p in pcts], "curves": curves,
            "lo": arr.min(0).tolist(), "hi": arr.max(0).tolist(), "mid": arr.mean(0).tolist(),
            "max_spread": float((arr.max(0) - arr.min(0)).max()),
            "max_decrease": dec, "all_monotone": bool(dec <= 1e-9),
            "n_variants": len(curves)}


def two_channel_check() -> dict:
    """For N = 2 the normalised participation ratio is a monotone function of lambda2/lambda1.

    Stated in the docstring above; computed here so the claim is checked rather than asserted.
    """
    out = []
    for r in np.linspace(0.0, 1.0, 21):
        ev = np.array([1.0, r])
        pr = ev.sum() ** 2 / np.sum(ev ** 2)
        out.append((float(r), float(pr - 1.0)))
    vals = [v for _, v in out]
    return {"ratios": [r for r, _ in out], "normalised_pr": vals,
            "monotone": bool(all(b - a >= -1e-12 for a, b in zip(vals, vals[1:]))),
            "at_0": vals[0], "at_1": vals[-1]}
