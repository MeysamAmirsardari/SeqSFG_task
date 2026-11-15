"""Noise bands, bursts and trials, deterministic from a seed.

Each burst is a fresh 20-Hz-wide noise built in the spectral domain (1-Hz resolution, the first
187.5 ms of a 1-s token). Comodulated bands use the SAME complex spectrum shifted to each centre
frequency, so their envelopes are identical; random bands each get their own.

A burst is a stretch of noise at the band's LONG-TERM level, so its own level varies from burst to
burst as in running noise (about 2 dB sd at 187.5 ms, 4 dB at 40 ms), together across comodulated
bands and independently across random ones. That variation is part of what comodulation offers the
listener, and for bursts shorter than about 1/bandwidth it is all of it. Rescaling every burst to
the same rms (`burst_level = "equalised"`, the behaviour before 2026-11-14) removes it: random
bands then have identical energy in every burst and the masker energy in the tone's band is the
same in every interval, which makes the random and band-alone conditions easier and CMR smaller.

Bursts are ramped and placed at their own onsets. A burst's comodulated partners
in other bands are the same noise shifted in time by the shear, exactly as when the burst of one
band is delayed in Christiansen & Oxenham (2014).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import ARRANGEMENTS, FLANK_SLOTS, PATTERNS, Condition, Config


def envelope(t: np.ndarray, dur: float, ramp: float) -> np.ndarray:
    env = np.zeros_like(t)
    on = (t >= 0) & (t < ramp)
    top = (t >= ramp) & (t <= dur - ramp)
    off = (t > dur - ramp) & (t <= dur)
    env[on] = np.sin(0.5 * math.pi * t[on] / ramp) ** 2
    env[top] = 1.0
    env[off] = np.sin(0.5 * math.pi * (dur - t[off]) / ramp) ** 2
    return env


def band_bursts(cfg: Config, rng: np.random.Generator, comodulated: bool) -> List[np.ndarray]:
    """One burst for every band, before ramps: unit LONG-TERM rms (natural), or unit rms over the
    burst itself (equalised). The random draws are the same either way."""
    sr = cfg.sample_rate
    n = int(round(cfg.burst_ms / 1000 * sr))
    nfft = sr                                         # 1-Hz bins
    half = int(round(cfg.bandwidth_hz / 2))
    k = np.arange(-half, half + 1)
    shared = rng.normal(size=k.size) + 1j * rng.normal(size=k.size)
    out = []
    for f in cfg.bands_hz:
        spec = shared if comodulated else rng.normal(size=k.size) + 1j * rng.normal(size=k.size)
        full = np.zeros(nfft // 2 + 1, complex)
        full[int(round(f)) + k] = spec
        x = np.fft.irfft(full, nfft)[:n]
        if cfg.burst_level == "equalised":
            out.append(x / np.sqrt(np.mean(x ** 2)))
        else:                                         # E[x^2] = 4 K / nfft^2 for K unit-variance bins
            out.append(x * (nfft / (2.0 * math.sqrt(k.size))))
    return out


MAX_REDRAWS = 4          # a trial whose noise happens to exceed full scale is drawn again
REDRAW_TAG = 990000      # ... with this (+ attempt) appended to its seed, which is what gets logged


@dataclass
class Trial:
    phase: str
    condition: str
    envelope: str
    arrangement: str
    dt_ms: float
    level_db: Optional[float]        # tone level; None = no tone (ratings)
    signal_interval: int             # 1-3, or 0 for none
    n_intervals: int
    seed: Tuple[int, ...]
    sample_rate: int
    n_samples: int
    interval_on_s: List[float] = field(default_factory=list)
    interval_off_s: List[float] = field(default_factory=list)
    target_on_s: List[float] = field(default_factory=list)     # final burst onset per interval
    offsets_ms: List[float] = field(default_factory=list)
    tone_band: int = 2
    tone_hz: float = 1000.0
    slots: List[List[List[int]]] = field(default_factory=list)   # patterns: interval, repetition,
                                                                 # band -> slot (offset / dT)

    def summary(self) -> dict:
        return asdict(self)


def order_code(slots: Sequence[Sequence[Sequence[int]]]) -> str:
    """Compact text of a trial's orders: slots by band (low to high) per repetition; repetitions
    of one interval joined by '/', intervals by '|'. A frozen interval is written once."""
    out = []
    for reps in slots:
        if all(list(r) == list(reps[0]) for r in reps):
            reps = reps[:1]
        out.append("/".join(",".join(f"{v:+d}" if v else "0" for v in r) for r in reps))
    return "|".join(out)


def _reordered_interval(cfg: Config, dt: float, rng: np.random.Generator, orders) -> List[List[int]]:
    """A new order for every repetition: the next order is drawn among those that differ from the
    last one and keep every band at least min_gap_ms of silence after its previous burst."""
    reps = [list(orders[int(rng.integers(len(orders)))])]
    for _ in range(cfg.n_precursors - 1):
        ok = [o for o in orders if list(o) != reps[-1] and all(
            cfg.burst_gap_ms(a, b, dt) >= cfg.min_gap_ms for a, b in zip(reps[-1], o))]
        reps.append(list(ok[int(rng.integers(len(ok)))]))
    return reps


def build(cfg: Config, cond: Condition, level_db: Optional[float], seed: Sequence[int], *,
          phase: str = "main", n_intervals: int = 3, signal_interval: Optional[int] = None,
          amplitude=None, orders: Optional[Sequence[Sequence[int]]] = None, stems: bool = False,
          _redraw: int = 0):
    """Render one trial. `amplitude(freq_hz, db)` is the sine peak for a level (the calibration).

    Natural-level noise has no hard ceiling: if a draw exceeds full scale the whole trial is drawn
    again from `seed + (REDRAW_TAG + attempt,)`, and the returned trial carries that seed, so the
    logged seed always regenerates exactly what was played. `_redraw=MAX_REDRAWS` turns this off.

    Patterns: `orders` gives one order (slot by band) per interval for a frozen trial (the session
    balances them over a track); without it they are drawn here, three different ones. With
    `stems=True` the per-band signals (and the tone, last) are returned too, for checking."""
    seed = tuple(int(s) for s in seed)
    asked = signal_interval
    rng = np.random.default_rng(list(seed))
    sr = cfg.sample_rate
    if signal_interval is None:
        signal_interval = int(rng.integers(1, n_intervals + 1)) if level_db is not None else 0
    else:
        rng.integers(1, n_intervals + 1)             # keep the draw order identical
    if cfg.tone_bands:                               # drawn only when randomised, so older
        tone_band = int(rng.choice(cfg.tone_bands))  # configurations regenerate bit for bit
        tone_hz = float(cfg.bands_hz[tone_band])
    else:
        tone_band, tone_hz = cfg.target_band, cfg.tone_hz
    nb_ = len(cfg.bands_hz)
    if cond.arrangement in PATTERNS:              # drawn only for patterns: fixed arrangements
        pool = cfg.orders()                       # regenerate exactly as before
        if cond.arrangement == "frozen":
            if orders is None:
                pick = rng.permutation(len(pool))[:n_intervals]
                orders = [pool[i] for i in pick]
            slots = [[list(orders[iv])] * cfg.n_precursors for iv in range(n_intervals)]
        else:
            slots = [_reordered_interval(cfg, cond.dt_ms, rng, pool) for _ in range(n_intervals)]
        off = cfg.offsets_ms(cond.arrangement, cond.dt_ms)
    else:
        off = cfg.offsets_ms(cond.arrangement, cond.dt_ms)
        slots = [[list(ARRANGEMENTS[cond.arrangement])] * cfg.n_precursors] * n_intervals
    if cfg.lead_ms is not None:
        lead = cfg.lead_ms / 1000
    else:
        lead = -min(off) / 1000 if min(off) < 0 else 0.0
    span = lead + cfg.n_precursors * cfg.noi_ms / 1000 + cfg.burst_ms / 1000
    pre, post, isi = 0.05, 0.1, cfg.isi_ms / 1000
    n_samples = int(math.ceil((pre + n_intervals * span + (n_intervals - 1) * isi + post) * sr))
    x = np.zeros(n_samples)
    dur, ramp = cfg.burst_ms / 1000, cfg.ramp_ms / 1000
    nb = int(round(dur * sr))
    env = envelope(np.arange(nb) / sr, dur, ramp)
    amp = amplitude or (lambda f, db: cfg.uncal_peak_60db * 10 ** ((db - 60.0) / 20))
    rms = [amp(f, cfg.band_db) / math.sqrt(2) for f in cfg.bands_hz]
    tr = Trial(phase, cond.name, cond.envelope, cond.arrangement, cond.dt_ms, level_db,
               signal_interval, n_intervals, seed, sr, n_samples, offsets_ms=list(off),
               tone_band=tone_band, tone_hz=tone_hz,
               slots=[[list(r) for r in reps] for reps in slots] if cond.arrangement in PATTERNS else [])
    stem = np.zeros((nb_ + 1, n_samples)) if stems else None
    phases = rng.uniform(0, 2 * math.pi, size=n_intervals)
    for iv in range(n_intervals):
        t0 = pre + iv * (span + isi) + lead          # time of the unshifted first precursor
        target = t0 + cfg.n_precursors * cfg.noi_ms / 1000
        for j in range(cfg.n_precursors + 1):
            bursts = band_bursts(cfg, rng, cond.envelope == "co")
            for b, sig in enumerate(bursts):
                if cond.envelope == "ref" and b != tone_band:   # reference: the tone's band alone
                    continue
                if j == cfg.n_precursors:
                    on = target
                elif cond.arrangement in PATTERNS:
                    on = t0 + j * cfg.noi_ms / 1000 + slots[iv][j][b] * cond.dt_ms / 1000
                else:
                    on = t0 + j * cfg.noi_ms / 1000 + off[b] / 1000
                i0 = int(round(on * sr))
                x[i0:i0 + nb] += rms[b] * env * sig
                if stems:
                    stem[b, i0:i0 + nb] += rms[b] * env * sig
        if iv + 1 == signal_interval:
            t = np.arange(nb) / sr
            i0 = int(round(target * sr))
            tone = amp(tone_hz, level_db) * env * np.sin(2 * math.pi * tone_hz * t + phases[iv])
            x[i0:i0 + nb] += tone
            if stems:
                stem[nb_, i0:i0 + nb] += tone
        if cfg.lead_ms is not None:
            tr.interval_on_s.append(t0 - lead)    # the same window in every condition
        else:
            tr.interval_on_s.append(t0 + min(0.0, min(off) / 1000))
        tr.interval_off_s.append(target + dur)
        tr.target_on_s.append(target)
    peak = float(np.abs(x).max())
    if peak > 0.95:
        if cfg.burst_level == "natural" and _redraw < MAX_REDRAWS:
            return build(cfg, cond, level_db, seed + (REDRAW_TAG + _redraw,), phase=phase,
                         n_intervals=n_intervals, signal_interval=asked, amplitude=amplitude,
                         orders=orders, stems=stems, _redraw=_redraw + 1)
        raise RuntimeError(f"peak {peak:.2f}: lower the levels or raise the system volume")
    buf = np.repeat(x[:, None], cfg.channels, 1).astype(np.float32)
    return (tr, buf, stem) if stems else (tr, buf)


def sha256(buf: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(buf, "<f4").tobytes()).hexdigest()


def band_envelopes(cfg: Config, buf: np.ndarray, t0: float, dur: float) -> np.ndarray:
    """Hilbert envelopes of each band over [t0, t0+dur] (for verification)."""
    from scipy import signal
    sr = cfg.sample_rate
    x = buf[:, 0].astype(float)
    out = []
    for f in cfg.bands_hz:
        sos = signal.butter(4, [f - 15, f + 15], btype="bandpass", fs=sr, output="sos")
        e = np.abs(signal.hilbert(signal.sosfiltfilt(sos, x)))
        out.append(e[int(t0 * sr):int((t0 + dur) * sr)])
    return np.array(out)
