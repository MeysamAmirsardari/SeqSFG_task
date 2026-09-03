"""ERB-spaced tone pool and small auditory-model helpers used only for diagnostics.

ERB formulas: Glasberg & Moore (1990). Absolute threshold: Terhardt (1979)
approximation. Roex excitation: Patterson et al. (1982) rounded-exponential
filter with p = 4f/ERB(f). A-weighting: IEC 61672 closed form. None of these
decide any stimulus parameter; they are used to *check* the level decisions.
"""
from __future__ import annotations

import numpy as np


def erb_width_hz(f_hz):
    f = np.asarray(f_hz, dtype=float)
    return 24.7 * (4.37 * f / 1000.0 + 1.0)


def erb_number(f_hz):
    f = np.asarray(f_hz, dtype=float)
    return 21.4 * np.log10(4.37 * f / 1000.0 + 1.0)


def erb_number_inv(e):
    e = np.asarray(e, dtype=float)
    return (10.0 ** (e / 21.4) - 1.0) / 4.37 * 1000.0


def make_pool(low_hz: float, high_hz: float, spacing_erb: float) -> np.ndarray:
    """Channel centre frequencies from low_hz upward in steps of spacing_erb ERB, <= high_hz."""
    e0 = float(erb_number(low_hz))
    e1 = float(erb_number(high_hz))
    n = int(np.floor((e1 - e0) / spacing_erb + 1e-9)) + 1
    es = e0 + spacing_erb * np.arange(n)
    return erb_number_inv(es)


def abs_threshold_db_spl(f_hz):
    """Terhardt (1979) approximation of the free-field absolute threshold."""
    fk = np.asarray(f_hz, dtype=float) / 1000.0
    return 3.64 * fk ** -0.8 - 6.5 * np.exp(-0.6 * (fk - 3.3) ** 2) + 1e-3 * fk ** 4


def a_weighting_db(f_hz):
    f = np.asarray(f_hz, dtype=float)
    f2 = f ** 2
    num = 12194.0 ** 2 * f2 ** 2
    den = (f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194.0 ** 2)
    return 20.0 * np.log10(num / den) + 2.00


def roex_weight(f_probe_hz, f_masker_hz):
    """Attenuation (linear power) of a masker at f_masker as seen by a roex filter centred at f_probe."""
    fp = np.asarray(f_probe_hz, dtype=float)
    fm = np.asarray(f_masker_hz, dtype=float)
    g = np.abs(fm - fp) / fp
    p = 4.0 * fp / erb_width_hz(fp)
    return (1.0 + p * g) * np.exp(-p * g)


def excitation_from_pool(freqs_hz, tone_level_db, occupancy):
    """Per-channel long-term excitation (dB) produced by all OTHER channels of the pool.

    Each other channel sounds a fraction `occupancy` of the time at `tone_level_db`.
    Returns (excitation_db, own_level_db) where own_level is the tone's own level.
    """
    freqs = np.asarray(freqs_hz, dtype=float)
    I = 10.0 ** (tone_level_db / 10.0) * occupancy
    exc = np.zeros_like(freqs)
    for i, fp in enumerate(freqs):
        w = roex_weight(fp, freqs)
        w[i] = 0.0
        exc[i] = 10.0 * np.log10(np.sum(w * I) + 1e-30)
    return exc, np.full_like(freqs, tone_level_db)
