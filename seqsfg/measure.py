"""Measurements on rendered audio (and exact counterparts on the schedule).

Everything here is a pure function of an audio buffer (plus the pool frequencies)
or of a schedule. The verification battery decides what to compare.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.signal.windows import hann

from .config import Config, Derived
from .stimulus import Interval, FIGURE, BACKGROUND, render_interval, tone_envelope

EPS = 1e-12


def db(x):
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=float), EPS))


# ----------------------------------------------------------------------------
# broadband envelope
# ----------------------------------------------------------------------------
def frame_rms(x: np.ndarray, sr: int, frame_ms: float = 2.0) -> np.ndarray:
    hop = int(round(sr * frame_ms / 1000.0))
    n = (len(x) // hop) * hop
    return np.sqrt(np.mean(x[:n].reshape(-1, hop).astype(float) ** 2, axis=1))


def envelope_features(env: np.ndarray, frame_ms: float, lag_lo_ms: float, lag_hi_ms: float) -> Dict[str, float]:
    """Summary statistics of a broadband envelope that need no knowledge of the schedule."""
    e = env / max(env.mean(), EPS)
    dev = e - 1.0
    fr = 1000.0 / frame_ms
    feats = {
        "mod_depth": float(np.std(e)),
        "crest": float(np.max(env) / max(np.sqrt(np.mean(env ** 2)), EPS)),
        "kurtosis": float(np.mean(dev ** 4) / max(np.mean(dev ** 2) ** 2, EPS)),
        "level_db": float(20.0 * np.log10(max(np.sqrt(np.mean(env ** 2)), EPS))),
    }
    spec = np.abs(np.fft.rfft(dev * hann(len(dev)))) ** 2
    freqs = np.fft.rfftfreq(len(dev), d=1.0 / fr)
    for lo, hi, name in ((0.5, 3.0, "mod_0.5_3Hz"), (3.0, 10.0, "mod_3_10Hz"), (10.0, 30.0, "mod_10_30Hz"),
                         (30.0, 150.0, "mod_30_150Hz")):
        sel = (freqs >= lo) & (freqs < hi)
        feats[name] = float(db(np.mean(spec[sel]) if sel.any() else EPS))
    ac = np.correlate(dev, dev, mode="full")[len(dev) - 1:]
    ac = ac / max(ac[0], EPS)
    lags = np.arange(len(ac)) * frame_ms
    sel = (lags >= lag_lo_ms) & (lags <= lag_hi_ms)
    feats["ac_peak_iei"] = float(np.max(ac[sel])) if sel.any() else 0.0
    sel2 = (lags >= 100.0) & (lags <= 1000.0)
    feats["ac_peak_100_1000"] = float(np.max(ac[sel2])) if sel2.any() else 0.0
    return feats


def element_locked_envelope(env: np.ndarray, frame_ms: float, element_onsets_ms: Sequence[float],
                            span_ms: float, pre_ms: float = 100.0, post_ms: float = 150.0) -> np.ndarray:
    """Envelope locked to element onsets, averaged over elements in the linear domain, in dB re the
    interval RMS. (Averaging in dB would let a momentary silence in one element dominate.)"""
    ref = max(np.sqrt(np.mean(env ** 2)), EPS)
    n_pre, n_post = int(round(pre_ms / frame_ms)), int(round((span_ms + post_ms) / frame_ms))
    segs = []
    for t in element_onsets_ms:
        c = int(round(t / frame_ms))
        lo, hi = c - n_pre, c + n_post
        if lo < 0 or hi > len(env):
            continue
        segs.append(env[lo:hi] / ref)
    if not segs:
        return np.zeros(n_pre + n_post)
    return 20.0 * np.log10(np.maximum(np.mean(segs, axis=0), EPS))


def per_element_rms_db(x: np.ndarray, sr: int, element_onsets_ms: Sequence[float], span_ms: float) -> np.ndarray:
    out = []
    for t in element_onsets_ms:
        a, b = int(round(t * sr / 1000.0)), int(round((t + span_ms) * sr / 1000.0))
        out.append(20.0 * np.log10(max(np.sqrt(np.mean(x[a:b].astype(float) ** 2)), EPS)))
    return np.array(out)


# ----------------------------------------------------------------------------
# channel-resolved measurements by complex demodulation at each pool frequency
# ----------------------------------------------------------------------------
def channel_envelopes(x: np.ndarray, sr: int, freqs: Sequence[float], win_ms: float = 40.0,
                      hop_ms: float = 1.0) -> np.ndarray:
    """Narrowband amplitude envelope of each channel, (n_channels, n_frames), frame rate 1/hop_ms.

    Demodulate at the exact channel frequency, box-average over one hop, then smooth with a
    Hann window of win_ms. With channels >= 1 ERB apart (>= ~50 Hz) and a 40 ms window the
    first null of the Hann lies at 50 Hz and neighbours are attenuated by > 30 dB.
    """
    hop = int(round(sr * hop_ms / 1000.0))
    n_frames = len(x) // hop
    xs = x[:n_frames * hop].astype(float)
    t = np.arange(n_frames * hop) / sr
    w = hann(int(round(win_ms / hop_ms)), sym=True)
    w = w / w.sum()
    env = np.empty((len(freqs), n_frames))
    for i, f in enumerate(freqs):
        y = xs * np.exp(-2j * np.pi * f * t)
        yb = y.reshape(n_frames, hop).mean(axis=1)
        env[i] = 2.0 * np.abs(np.convolve(yb, w, mode="same"))
    return env


def single_tone_reference(cfg: Config, d: Derived, win_ms: float = 40.0, hop_ms: float = 1.0) -> float:
    """Plateau envelope value one isolated tone produces under channel_envelopes (per-channel identical)."""
    sr = cfg.sample_rate
    env = tone_envelope(cfg)
    f = float(d.channel_freqs_hz[len(d.channel_freqs_hz) // 2])
    n = cfg.ms_to_samples(200.0)
    x = np.zeros(n)
    start = cfg.ms_to_samples(50.0)
    x[start:start + env.size] = cfg.tone_amplitude * env * np.sin(2 * np.pi * f * np.arange(env.size) / sr)
    e = channel_envelopes(x, sr, [f], win_ms, hop_ms)[0]
    return float(e.max())


def on_states(env: np.ndarray, ref: float, frac: float = 0.5) -> np.ndarray:
    return env > frac * ref


def onsets_from_states(on: np.ndarray) -> List[np.ndarray]:
    """Rising edges per channel, in frames."""
    out = []
    for row in on:
        d = np.diff(row.astype(np.int8), prepend=0)
        out.append(np.flatnonzero(d == 1))
    return out


def channel_power_db(env: np.ndarray) -> np.ndarray:
    return db(np.mean(env ** 2, axis=1))


def peakedness(v: np.ndarray, n_top: int) -> float:
    """How far the most prominent channels stand above the rest: mean of top-n minus median."""
    s = np.sort(v)[::-1]
    return float(np.mean(s[:n_top]) - np.median(v))


# ----------------------------------------------------------------------------
# exact schedule-based measures
# ----------------------------------------------------------------------------
def count_trace(onsets: np.ndarray, n_grid: int, dur: int) -> np.ndarray:
    """Number of tones sounding at each grid instant (exact, from onsets)."""
    diff = np.zeros(n_grid + 1, dtype=int)
    np.add.at(diff, onsets, 1)
    np.add.at(diff, np.minimum(onsets + dur, n_grid), -1)
    return np.cumsum(diff)[:n_grid]


def channel_on_matrix(iv: Interval, n_channels: int, n_grid: int, dur: int) -> np.ndarray:
    on = np.zeros((n_channels, n_grid), dtype=bool)
    for c in range(n_channels):
        o = iv.onset[iv.channel == c]
        for t in o:
            on[c, t:t + dur] = True
    return on


def pair_count_in_lag_range(onsets_ms: np.ndarray, lo: float, hi: float, T: float) -> float:
    """Number of onset pairs whose lag is in [lo, hi], divided by its expectation for uniform placement."""
    n = onsets_ms.size
    if n < 2:
        return 1.0
    diffs = onsets_ms[None, :] - onsets_ms[:, None]
    obs = np.sum((diffs >= lo) & (diffs <= hi))
    p = ((T - lo) ** 2 - (T - hi) ** 2) / T ** 2 if hi < T else (T - lo) ** 2 / T ** 2
    exp = n * (n - 1) / 2.0 * p
    return float(obs / max(exp, EPS))


def single_channel_stats(onsets_ms_per_channel: List[np.ndarray], cfg: Config) -> Dict[str, np.ndarray]:
    """Per-channel statistics from onset times (ms), any source (schedule or audio)."""
    T = cfg.interval_dur_ms
    keys = ["count", "ioi_mean", "ioi_sd", "ioi_cv", "ioi_min", "ioi_max", "frac_ioi_in_iei", "pairs_iei_norm"]
    out = {k: np.zeros(len(onsets_ms_per_channel)) for k in keys}
    for c, o in enumerate(onsets_ms_per_channel):
        o = np.sort(np.asarray(o, dtype=float))
        out["count"][c] = o.size
        if o.size >= 2:
            ioi = np.diff(o)
            out["ioi_mean"][c] = ioi.mean()
            out["ioi_sd"][c] = ioi.std()
            out["ioi_cv"][c] = ioi.std() / max(ioi.mean(), EPS)
            out["ioi_min"][c] = ioi.min()
            out["ioi_max"][c] = ioi.max()
            out["frac_ioi_in_iei"][c] = np.mean((ioi >= cfg.iei_min_ms) & (ioi <= cfg.iei_max_ms))
        out["pairs_iei_norm"][c] = pair_count_in_lag_range(o, cfg.iei_min_ms, cfg.iei_max_ms, T)
    return out
