"""What the temporal-coherence model predicts for every condition, and where it puts the bound.

The quantity
-------------
The task asks whether ONE tone, the target, is bound with the rest. So the model output used
here is the target's coherence with the figure: the mean normalised coherence between the
target's channel and each other channel,

    rho_T = mean_k C_tk / sqrt(C_tt C_kk),     k != target

from the coherence matrix of `tcoh.model` (rate filters 2-32 Hz, half-wave coincidence, as
validated against Elhilali et al. 2009 there). It is 1 at step 0 by construction and falls as
the shear grows. `tshear.model` reads the whole matrix as an effective number of objects; that
is reported too, but it answers a different question (how many objects, not whether this one is
bound) and the two need not agree.

Two readings, two bounds
------------------------
The half-wave reading puts the half-way point of rho_T at about 20 ms of shear: the tones' 60 ms
envelopes stop overlapping and the fast rate filters lose them. The magnitude reading weights
the slow 3 Hz repetition, which a 64 ms shift barely dephases, and puts it beyond 40 ms. The
experiment's range, 0 to 64 ms, is chosen so that the two predict visibly different curves. That
is the sense in which the measured bound is a test of the model and not only a number.

The capture control
-------------------
Captor tokens make the other tones repeat at twice the target's rate with no change in onset
asynchrony. The model puts rho_T for that stimulus at 0.71 (half-wave), the value it reaches at
about 13 ms of shear. If binding is what the model says it is, the capture condition should sit
where a 13 ms shear sits on the measured curve: an out-of-sample check, because nothing in the
shear conditions was fitted to it.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional, Sequence

import numpy as np

from tcoh.model import coherence_matrix

from .config import Condition, Config, captor_onsets, conditions, onsets

MODEL_FS = 1000.0
VARIANTS = {"halfwave": {}, "magnitude": {"rectify": "magnitude"}, "trimmed": {"trim_s": 0.6}}
# `significance` only changes how the matrix is decomposed, not rho_T, so it is not a reading
# of the model for this purpose.


def envelopes(cfg: Config, cond: Condition, n_rep: Optional[int] = None,
              fs: float = MODEL_FS) -> np.ndarray:
    """(n_tones, n_samples) envelopes at the model's sample rate, captors included.

    Six repetitions rather than five by default: the matrix integrates over the whole input,
    and one extra period keeps the estimate from being dominated by the onset of the sequence.
    """
    n_rep = n_rep if n_rep is not None else max(cfg.n_repeats, 6)
    c2 = cfg.replace(n_repeats=n_rep)
    o = onsets(c2, cond) - cfg.lead_ms
    co = captor_onsets(c2, cond)
    total = int(round((n_rep * cfg.period_ms + (cfg.n_tones - 1) * cond.step_ms
                       + cfg.tone_ms + 100.0) * fs / 1000.0))
    w = int(round(cfg.tone_ms * fs / 1000.0))
    r = max(int(round(cfg.ramp_ms * fs / 1000.0)), 1)
    shape = np.ones(w)
    shape[:r] = np.linspace(0.0, 1.0, r)
    shape[-r:] = np.linspace(1.0, 0.0, r)
    env = np.zeros((cfg.n_tones, total))

    def put(k, on):
        i = int(round(on * fs / 1000.0))
        j = min(total, i + w)
        env[k, i:j] += shape[: j - i]

    for rep in range(n_rep):
        for k in range(cfg.n_tones):
            if np.isfinite(o[rep, k]):
                put(k, o[rep, k])
    if co is not None:
        co = co - cfg.lead_ms
        for rep in range(co.shape[0]):
            for k in range(cfg.n_tones):
                if np.isfinite(co[rep, k]):
                    put(k, co[rep, k])
    return env


@lru_cache(maxsize=8192)
def _coh(cfg: Config, cond: Condition, variant: str) -> Dict[str, float]:
    env = envelopes(cfg, cond)
    c = coherence_matrix(env, MODEL_FS, **VARIANTS[variant])
    M = np.asarray(c.matrix, dtype=float)
    d = np.sqrt(np.clip(np.diag(M), 1e-300, None))
    Rm = M / np.outer(d, d)
    t, oth = cfg.target_index, cfg.others
    ev = np.clip(np.sort(np.abs(np.asarray(c.eigenvalues)))[::-1], 0.0, None)
    pr = float(ev.sum() ** 2 / np.sum(ev ** 2)) if ev.sum() > 0 else 1.0
    return {"target_coherence": float(np.mean(Rm[t, oth])),
            "effective_objects": pr,
            "normalised_objects": (pr - 1.0) / (cfg.n_tones - 1.0)}


def target_coherence(cfg: Config, cond: Condition, variant: str = "halfwave") -> float:
    """rho_T for one stimulus. Undefined (NaN) for the target alone."""
    if cond.kind == "alone":
        return float("nan")
    key = Condition("m", "A", float(cond.step_ms), cond.kind)   # task does not change timing
    return _coh(cfg, key, variant)["target_coherence"]


def effective_objects(cfg: Config, cond: Condition, variant: str = "halfwave") -> float:
    if cond.kind == "alone":
        return 1.0
    key = Condition("m", "A", float(cond.step_ms), cond.kind)
    return _coh(cfg, key, variant)["effective_objects"]


GRID_MAX_MS = 90.0
GRID_RES_MS = 1.0


@lru_cache(maxsize=32)
def _grid(cfg: Config, variant: str) -> tuple:
    """rho_T on a fine grid of shears, computed once per configuration and reading.

    The grid stops at 90 ms: beyond that the four tones no longer fit in one 333 ms period at
    the default geometry, and the stimulus the model would be describing does not exist.
    """
    top = min(GRID_MAX_MS, (cfg.period_ms - cfg.tone_ms) / (cfg.n_tones - 1) - 1.0)
    g = np.arange(0.0, top + 1e-9, GRID_RES_MS)
    v = np.array([target_coherence(cfg, Condition("m", "A", float(x)), variant) for x in g])
    return g, v


def curve(cfg: Config, steps: Sequence[float], variant: str = "halfwave") -> np.ndarray:
    """rho_T at arbitrary shears, interpolated on the cached grid (clamped at its end)."""
    g, v = _grid(cfg, variant)
    return np.interp(np.asarray(steps, dtype=float), g, v)


def half_way_ms(cfg: Config, variant: str = "halfwave") -> float:
    """The shear at which rho_T has fallen half-way from 1 to its value at the largest step
    of the session. NaN if the grid never gets there."""
    floor = float(curve(cfg, [max(cfg.steps_ms)], variant)[0])
    return equivalent_shear_ms(cfg, 1.0 - 0.5 * (1.0 - floor), variant)


def equivalent_shear_ms(cfg: Config, rho: float, variant: str = "halfwave") -> float:
    """The smallest shear that gives this rho_T, e.g. for the capture condition."""
    g, v = _grid(cfg, variant)
    below = np.where(v <= rho)[0]
    if not len(below) or below[0] == 0:
        return float("nan")
    i = below[0]
    return float(np.interp(rho, [v[i], v[i - 1]], [g[i], g[i - 1]]))


def predictions(cfg: Config) -> dict:
    """rho_T for every condition under every reading, with the bound each reading implies."""
    out = {"variants": {}, "steps_ms": list(cfg.steps_ms)}
    for v in VARIANTS:
        per = {c.name: target_coherence(cfg, c, v) for c in conditions(cfg)
               if c.kind != "alone"}
        cap = Condition("A_cap", "A", 0.0, "capture")
        rho_cap = target_coherence(cfg, cap, v)
        out["variants"][v] = {
            "per_condition": per,
            "curve": [target_coherence(cfg, Condition("m", "A", float(x)), v)
                      for x in cfg.steps_ms],
            "half_way_ms": half_way_ms(cfg, v),
            "capture_rho": rho_cap,
            "capture_equivalent_ms": equivalent_shear_ms(cfg, rho_cap, v),
        }
    return out
