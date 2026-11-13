"""Sequences: deterministic tone lists, rendered in continuous time (onsets need not fall on samples)."""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import List, Tuple

import numpy as np

from .config import Condition, Config


@dataclass(frozen=True)
class Tone:
    role: str          # flank1 | mid_a | b | flank2
    triplet: int
    freq_hz: float
    onset_s: float
    dur_s: float
    phase_rad: float


def build(cfg: Config, cond: Condition, seconds: float, seed) -> List[Tone]:
    rng = np.random.default_rng(list(seed) if isinstance(seed, (list, tuple)) else seed)
    T, soa, P = cfg.tone_ms / 1000, cfg.soa_ms / 1000, cfg.period_ms / 1000
    n = max(1, int(seconds / P + 1e-9))
    f_flank = {"same": cfg.f_a_hz, "off": cfg.f_off_hz, "none": None}[cond.flank]
    tones: List[Tone] = []
    for k in range(n):
        t0 = k * P
        ph = rng.uniform(0, 2 * math.pi, size=4)
        if f_flank is not None:
            tones.append(Tone("flank1", k, f_flank, t0, T, ph[0]))
            tones.append(Tone("flank2", k, f_flank, t0 + 2 * soa, T, ph[3]))
        tones.append(Tone("mid_a", k, cfg.f_a_hz, t0 + soa, T, ph[1]))
        tones.append(Tone("b", k, cfg.f_b_hz, t0 + soa + cond.d_ms / 1000, T, ph[2]))
    return tones


def envelope(t: np.ndarray, dur: float, ramp: float) -> np.ndarray:
    env = np.zeros_like(t)
    on = (t >= 0) & (t < ramp)
    top = (t >= ramp) & (t <= dur - ramp)
    off = (t > dur - ramp) & (t <= dur)
    env[on] = np.sin(0.5 * math.pi * t[on] / ramp) ** 2
    env[top] = 1.0
    env[off] = np.sin(0.5 * math.pi * (dur - t[off]) / ramp) ** 2
    return env


def render(cfg: Config, tones: List[Tone], amplitude, pre_s: float = 0.05,
           post_s: float = 0.2) -> np.ndarray:
    """(n, channels) float32; `amplitude(freq_hz, level_db)` is the calibration."""
    sr = cfg.sample_rate
    end = max(t.onset_s + t.dur_s for t in tones)
    x = np.zeros(int(math.ceil((pre_s + end + post_s) * sr)), np.float64)
    for tn in tones:
        a = float(amplitude(tn.freq_hz, cfg.level_db))
        on = pre_s + tn.onset_s
        i0 = int(math.floor(on * sr))
        i1 = min(len(x), int(math.ceil((on + tn.dur_s) * sr)) + 1)
        t = np.arange(i0, i1) / sr - on
        x[i0:i1] += a * envelope(t, tn.dur_s, cfg.ramp_ms / 1000) * np.sin(
            2 * math.pi * tn.freq_hz * t + tn.phase_rad)
    peak = float(np.abs(x).max())
    if peak > 0.95:
        raise RuntimeError(f"peak {peak:.2f}: lower the level")
    return np.repeat(x[:, None], cfg.channels, 1).astype(np.float32)


def sha256(buf: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(buf, "<f4").tobytes()).hexdigest()


def table(tones: List[Tone]) -> List[dict]:
    return [asdict(t) for t in tones]


def sound_end(tones: List[Tone]) -> float:
    """Last offset, on the same clock as the reports (0 = start of the first triplet)."""
    return max(t.onset_s + t.dur_s for t in tones)
