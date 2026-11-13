"""Noise bands, bursts and trials, deterministic from a seed.

Each burst is a fresh 20-Hz-wide noise built in the spectral domain (1-Hz resolution, the first
187.5 ms of a 1-s token). Comodulated bands use the SAME complex spectrum shifted to each centre
frequency, so their envelopes are identical; random bands each get their own. Every band burst is
scaled to the band level (rms), ramped, and placed at its own onset. A burst's comodulated partners
in other bands are the same noise shifted in time by the shear, exactly as when the burst of one
band is delayed in Christiansen & Oxenham (2014).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import Condition, Config


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
    """One burst for every band, unit rms over the burst, before ramps."""
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
        out.append(x / np.sqrt(np.mean(x ** 2)))
    return out


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

    def summary(self) -> dict:
        return asdict(self)


def build(cfg: Config, cond: Condition, level_db: Optional[float], seed: Sequence[int], *,
          phase: str = "main", n_intervals: int = 3, signal_interval: Optional[int] = None,
          amplitude=None) -> Tuple[Trial, np.ndarray]:
    """Render one trial. `amplitude(freq_hz, db)` is the sine peak for a level (the calibration)."""
    seed = tuple(int(s) for s in seed)
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
    off = cfg.offsets_ms(cond.arrangement, cond.dt_ms)
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
               tone_band=tone_band, tone_hz=tone_hz)
    phases = rng.uniform(0, 2 * math.pi, size=n_intervals)
    for iv in range(n_intervals):
        t0 = pre + iv * (span + isi) + lead          # time of the unshifted first precursor
        target = t0 + cfg.n_precursors * cfg.noi_ms / 1000
        for j in range(cfg.n_precursors + 1):
            bursts = band_bursts(cfg, rng, cond.envelope == "co")
            for b, sig in enumerate(bursts):
                if cond.envelope == "ref" and b != tone_band:   # reference: the tone's band alone
                    continue
                on = target if j == cfg.n_precursors else t0 + j * cfg.noi_ms / 1000 + off[b] / 1000
                i0 = int(round(on * sr))
                x[i0:i0 + nb] += rms[b] * env * sig
        if iv + 1 == signal_interval:
            t = np.arange(nb) / sr
            i0 = int(round(target * sr))
            x[i0:i0 + nb] += amp(tone_hz, level_db) * env * np.sin(
                2 * math.pi * tone_hz * t + phases[iv])
        tr.interval_on_s.append(t0 + min(0.0, min(off) / 1000))
        tr.interval_off_s.append(target + dur)
        tr.target_on_s.append(target)
    peak = float(np.abs(x).max())
    if peak > 0.95:
        raise RuntimeError(f"peak {peak:.2f}: lower the levels or raise the system volume")
    return tr, np.repeat(x[:, None], cfg.channels, 1).astype(np.float32)


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
