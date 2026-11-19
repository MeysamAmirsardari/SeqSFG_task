"""One trial: a single A-B sequence, its levels, and its sound, deterministic from a seed.

Every random number is drawn from one generator per trial, in a fixed order and always the same
number of draws, so a trial regenerates bit for bit from (config, delay, target, seed). Whether the
trial carries the change is decided by the session (half the trials of every block) and passed in.

Levels, per tone pair k (B_k at k x SOA, A_k at k x SOA + D):
    common       A_k = L + r_k + s_k           B_k = L + r_k + s_k + t_k
    independent  A_k = L + r_k + s_k           B_k = L + t_k               (Micheyl et al. 2013)
with r_k uniform in +-rove_db, s_k = decoy_db on one or two pairs (else 0), t_k = target_db on
the one target B tone of a change trial (never among the first `target_from`).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import Config


@dataclass(frozen=True)
class Tone:
    stream: str            # "A" or "B"
    index: int             # 0 .. n_tones-1
    freq_hz: float
    onset_s: float         # from the first sample of the buffer
    level_db: float
    rove_db: float
    decoy: bool
    target: bool
    phase_rad: float


@dataclass
class Trial:
    phase: str             # example | practice | main
    delay_ms: float
    levels: str
    target_present: bool
    target_index: int      # B tone carrying the change, -1 if none
    decoys: List[int]
    seed: Tuple[int, ...]
    sample_rate: int
    n_samples: int
    block: int = -1
    n_intervals: int = 1
    interval_on_s: List[float] = field(default_factory=list)
    interval_off_s: List[float] = field(default_factory=list)
    tones: List[Tone] = field(default_factory=list)

    @property
    def signal_interval(self) -> int:           # for the shared app: 1 = the change is there
        return 1 if self.target_present else 0

    @property
    def condition(self) -> str:
        return f"D{self.delay_ms:g}"

    def summary(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "tones"}
        d["a_levels_db"] = [t.level_db for t in self.tones if t.stream == "A"]
        d["b_levels_db"] = [t.level_db for t in self.tones if t.stream == "B"]
        return d


def _draws(rng: np.random.Generator, cfg: Config) -> dict:
    n = cfg.n_tones
    k = int(rng.choice(cfg.decoy_counts))
    return {"rove": rng.uniform(-cfg.rove_db, cfg.rove_db, size=n),
            "decoys": sorted(int(x) for x in rng.choice(n, size=k, replace=False)),
            "target": int(rng.integers(cfg.target_from, n)),
            "phase_b": rng.uniform(0.0, 2.0 * math.pi, size=n),
            "phase_a": rng.uniform(0.0, 2.0 * math.pi, size=n)}


def n_samples(cfg: Config) -> int:
    """One buffer length for every trial: the longest sequence (largest delay) plus margins."""
    longest = cfg.sequence_ms(max(cfg.delays_ms))
    return int(round((cfg.pre_ms + longest + cfg.post_ms) / 1000.0 * cfg.sample_rate))


def build(cfg: Config, delay_ms: float, target_present: bool, seed: Sequence[int], *,
          phase: str = "main", block: int = -1, amplitude=None, stems: bool = False,
          target_db: float = None):
    """Render one trial. `amplitude(freq_hz, db)` is the sine peak for a level (calibration);
    without it 60 dB SPL is `uncal_peak_60db`. With stems=True the B and A signals are returned
    separately too (rows 0 and 1), for checking."""
    seed = tuple(int(s) for s in seed)
    r = _draws(np.random.default_rng(list(seed)), cfg)
    L, sr = cfg.level_db, cfg.sample_rate
    t_idx = r["target"] if target_present else -1
    tdb = cfg.target_db if target_db is None else float(target_db)   # practice may use a bigger one
    tones: List[Tone] = []
    pre = cfg.pre_ms / 1000.0
    for k in range(cfg.n_tones):
        rove = float(r["rove"][k])
        dec = k in r["decoys"]
        s = cfg.decoy_db if dec else 0.0
        tgt = k == t_idx
        if cfg.levels == "common":
            la, lb = L + rove + s, L + rove + s + (tdb if tgt else 0.0)
        else:
            la, lb = L + rove + s, L + (tdb if tgt else 0.0)
        on_b = pre + k * cfg.soa_ms / 1000.0
        tones.append(Tone("B", k, cfg.f_b_hz, on_b, lb, rove if cfg.levels == "common" else 0.0,
                          dec and cfg.levels == "common", tgt, float(r["phase_b"][k])))
        tones.append(Tone("A", k, cfg.f_a_hz, on_b + delay_ms / 1000.0, la, rove, dec, False,
                          float(r["phase_a"][k])))
    n = n_samples(cfg)
    amp = amplitude or (lambda f, db: cfg.uncal_peak_60db * 10 ** ((db - 60.0) / 20))
    dur, ramp = cfg.tone_ms / 1000.0, cfg.ramp_ms / 1000.0
    m = int(round(dur * sr))
    t = np.arange(m) / sr
    env = np.ones(m)
    on, off = t < ramp, t > dur - ramp
    env[on] = np.sin(0.5 * math.pi * t[on] / ramp) ** 2
    env[off] = np.sin(0.5 * math.pi * (dur - t[off]) / ramp) ** 2
    stem = np.zeros((2, n))
    for tn in tones:
        i0 = int(round(tn.onset_s * sr))
        stem[0 if tn.stream == "B" else 1, i0:i0 + m] += \
            amp(tn.freq_hz, tn.level_db) * env * np.sin(2 * math.pi * tn.freq_hz * t + tn.phase_rad)
    x = stem.sum(0)
    peak = float(np.abs(x).max())
    if peak > cfg.max_peak:
        raise RuntimeError(f"peak {peak:.2f}: lower the levels or the calibration amplitude")
    first = min(tn.onset_s for tn in tones)
    last = max(tn.onset_s for tn in tones) + dur
    tr = Trial(phase, float(delay_ms), cfg.levels, bool(target_present), t_idx, r["decoys"], seed,
               sr, n, block, 1, [first], [last], tones)
    buf = np.repeat(x[:, None], cfg.channels, 1).astype(np.float32)
    return (tr, buf, stem) if stems else (tr, buf)


def sha256(buf: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(buf, "<f4").tobytes()).hexdigest()
