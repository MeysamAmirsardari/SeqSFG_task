"""Pure tones for one cup: a sine at the requested peak, raised-cosine on and off ramps, the other
channel exactly silent.

Ramps: 50-ms raised-cosine (sin^2) ramps. The level rises from -20 dB to -1 dB of its steady value
in 0.581 x 50 = 29 ms and falls back in the same time, so switching makes no click. (IEC 60645-1
clause 8.6.3 sets rise and fall times for audiometers; we could not read that clause, so it is not
claimed here. tests/test_audiometry.py measures what these ramps actually do.)
"""
from __future__ import annotations

import math

import numpy as np

from .rig import CHANNELS, CUPS, MAX_PEAK, SAMPLE_RATE

RAMP_S = 0.050


def envelope(n: int, sr: int, ramp_s: float = RAMP_S) -> np.ndarray:
    nr = int(round(ramp_s * sr))
    if 2 * nr > n:
        raise ValueError("tone shorter than its two ramps")
    e = np.ones(n)
    r = np.sin(0.5 * np.pi * (np.arange(nr) + 0.5) / nr) ** 2
    e[:nr] = r
    e[n - nr:] = r[::-1]
    return e


def tone(freq: float, dur_s: float, peak: float, cup: str, sr: int = SAMPLE_RATE,
         channels: int = CHANNELS, ramp_s: float = RAMP_S) -> np.ndarray:
    """(n, channels) float32: the tone in `cup`'s channel, zeros elsewhere."""
    if cup not in CUPS:
        raise ValueError(f"cup must be one of {CUPS}")
    if not (0.0 < peak <= MAX_PEAK) or not math.isfinite(peak):
        raise ValueError(f"peak {peak} outside (0, {MAX_PEAK}]")
    if not (20.0 <= freq <= sr / 2.5):
        raise ValueError(f"frequency {freq} Hz out of range")
    n = int(round(dur_s * sr))
    t = np.arange(n) / sr
    x = peak * np.sin(2.0 * np.pi * freq * t) * envelope(n, sr, ramp_s)
    out = np.zeros((n, channels), dtype=np.float32)
    out[:, CUPS.index(cup)] = x.astype(np.float32)
    return out


def band_noise(freqs, seconds: float, nominal_db: float, cup: str, rng: np.random.Generator,
               bandwidth_hz: float = 20.0, sr: int = SAMPLE_RATE, channels: int = CHANNELS,
               ramp_s: float = RAMP_S) -> np.ndarray:
    """Continuous noise bands of 20 Hz around each frequency, each with the long-term rms tcmr
    gives a band at `nominal_db` (the rms of a sine of peak `peak_for(nominal_db)`), built the
    way tcmr builds them: random complex bins across the band, inverse FFT, scaled to unit
    long-term power by nfft / (2 sqrt K). Several frequencies add independent bands."""
    from .rig import peak_for
    if cup not in CUPS:
        raise ValueError(f"cup must be one of {CUPS}")
    n = int(round(seconds * sr))
    df = sr / n
    half = int(round(bandwidth_hz / 2 / df))
    k = np.arange(-half, half + 1)
    x = np.zeros(n)
    for f in np.atleast_1d(freqs):
        full = np.zeros(n // 2 + 1, complex)
        full[int(round(float(f) / df)) + k] = rng.normal(size=k.size) + 1j * rng.normal(size=k.size)
        x += np.fft.irfft(full, n) * (n / (2.0 * np.sqrt(k.size)))
    x *= peak_for(nominal_db) / np.sqrt(2.0)
    x *= envelope(n, sr, ramp_s)
    if np.max(np.abs(x)) > MAX_PEAK:
        raise ValueError("band noise would exceed the peak limit")
    out = np.zeros((n, channels), dtype=np.float32)
    out[:, CUPS.index(cup)] = x.astype(np.float32)
    return out


def ramp_times_ms(ramp_s: float = RAMP_S, sr: int = SAMPLE_RATE) -> tuple:
    """(rise, fall): time from -20 dB to -1 dB of the steady amplitude and back, measured on the
    envelope itself."""
    n = int(round(4 * ramp_s * sr))
    e = envelope(n, sr, ramp_s)
    lo, hi = 10 ** (-20 / 20), 10 ** (-1 / 20)
    nr = int(round(ramp_s * sr))
    up = e[:nr]
    down = e[n - nr:]
    rise = (np.argmax(up >= hi) - np.argmax(up >= lo)) / sr * 1000
    fall = (np.argmax(down < lo) - np.argmax(down < hi)) / sr * 1000
    return float(rise), float(fall)
