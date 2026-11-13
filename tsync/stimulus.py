"""The stimulus: geometry, randomisation and rendering, all deterministic from a seed.

Every tone is synthesised in continuous time and sampled, so an onset can fall between samples
and delta is exact at any value, not rounded to 1/48000 s. A trial is rendered once into a
single buffer (sequence 1, roved silence, sequence 2), so the timing inside a trial is fixed by
the samples and nothing the audio driver does can change it.

Randomness is drawn from one generator per trial, in a fixed order and always the same number of
draws, so a trial can be regenerated bit-for-bit from (config, condition, delta, seed, phase,
catch) and its SHA-256 compared with the one logged when it was played (`tsync audit`).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Condition, Config


# ----------------------------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------------------------
def ending_reference_ms(cfg: Config, cond: Condition) -> Tuple[float, float]:
    """Where the ending's B and A tones sit before delta is applied (ms from the first onset)."""
    grid_b = (cfg.n_tones - 1) * cfg.soa_ms
    if cfg.ending == "sync":
        ref = grid_b + cond.context_ms / 2.0 + cond.ending_shift_ms
        return ref, ref
    return grid_b + cond.ending_shift_ms, grid_b + cond.context_ms + cond.ending_shift_ms


def ending_onsets_ms(cfg: Config, cond: Condition, delta_ms: float, sign: int) -> Tuple[float, float]:
    """Onsets of the ending's B and A tones. sign +1: B first (B earlier, A later); -1: A first."""
    rb, ra = ending_reference_ms(cfg, cond)
    sb = cfg.delta_split_b
    return rb - sign * delta_ms * sb, ra + sign * delta_ms * (1.0 - sb)


def sequence_onsets_ms(cfg: Config, cond: Condition, delta_ms: float, sign: int
                       ) -> Tuple[List[float], List[float]]:
    """All B and A onsets of one sequence, ms from its first B onset."""
    b = [k * cfg.soa_ms for k in range(cfg.n_tones - 1)]
    a = [k * cfg.soa_ms + cond.context_ms for k in range(cfg.n_tones - 1)]
    eb, ea = ending_onsets_ms(cfg, cond, delta_ms, sign)
    return b + [eb], a + [ea]


def nominal_end_ms(cfg: Config, cond: Condition) -> float:
    """End of the last tone of a sequence whose ending is synchronous (the standard)."""
    b, a = sequence_onsets_ms(cfg, cond, 0.0, +1)
    last = max(b + (a if cond.has_a else []))
    return last + cfg.tone_ms


# ----------------------------------------------------------------------------------------------
# the trial
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Tone:
    interval: int          # 1 or 2
    stream: str            # "A" or "B"
    index: int             # 0 .. n_tones-1; the last one is the ending
    freq_hz: float
    onset_s: float         # from the first sample of the trial buffer
    dur_s: float
    level_db: float        # dB SPL, nominal + rove + decoy
    rove_db: float
    decoy: bool
    phase_rad: float


@dataclass
class Trial:
    phase: str             # demo | practice | exposure | main
    condition: str
    context_ms: float
    ending_shift_ms: float
    delta_ms: float        # asynchrony of the target ending (0 for none)
    sign: int              # +1 B first, -1 A first
    target_interval: int   # 1 or 2
    is_catch: bool
    seed: Tuple[int, ...]
    isi_ms: float          # silence from the standard's last offset to sequence 2's first onset
    sample_rate: int
    n_samples: int
    seq_start_s: Tuple[float, float]       # first onset of each sequence
    interval_on_s: Tuple[float, float]     # first sound of each interval
    interval_off_s: Tuple[float, float]    # last sound of each interval
    ending_b_s: Tuple[float, float]        # onset of the ending's B tone, per interval
    ending_a_s: Tuple[float, float]
    tones: List[Tone] = field(default_factory=list)

    @property
    def asynchrony_ms(self) -> Tuple[float, float]:
        """A onset minus B onset at the ending, per interval (positive: B first)."""
        return tuple((a - b) * 1000.0 for a, b in zip(self.ending_a_s, self.ending_b_s))

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate

    def summary(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "tones"}
        d["asynchrony_ms"] = list(self.asynchrony_ms)
        return d


def _draws(rng: np.random.Generator, cfg: Config) -> dict:
    """Every random number a trial uses, in one fixed order, whatever the condition."""
    n = cfg.n_tones
    out = {
        "target_interval": int(rng.integers(1, 3)),
        "sign": int(rng.choice((1, -1))),
        "isi_ms": float(rng.uniform(cfg.isi_ms[0], cfg.isi_ms[1])),
    }
    for iv in (1, 2):
        out[f"rove{iv}"] = rng.uniform(-cfg.a_rove_db, cfg.a_rove_db, size=n)
        k = int(rng.choice(cfg.a_decoy_counts))
        out[f"decoy{iv}"] = sorted(int(x) for x in rng.choice(n, size=k, replace=False))
        out[f"phase_b{iv}"] = rng.uniform(0.0, 2.0 * math.pi, size=n)
        out[f"phase_a{iv}"] = rng.uniform(0.0, 2.0 * math.pi, size=n)
    return out


def build_trial(cfg: Config, cond: Condition, delta_ms: float, seed: Sequence[int], *,
                phase: str = "main", is_catch: bool = False,
                target_interval: Optional[int] = None, sign: Optional[int] = None) -> Trial:
    """One trial: which interval holds the target, its order, the roves, the phases, the timing."""
    seed = tuple(int(s) for s in seed)
    rng = np.random.default_rng(list(seed))
    r = _draws(rng, cfg)
    ti = int(target_interval) if target_interval is not None else r["target_interval"]
    if sign is None:
        sign = {"random": r["sign"], "b_first": 1, "a_first": -1}[cfg.delta_sign]
    if ti not in (1, 2) or sign not in (1, -1):
        raise ValueError("target_interval must be 1 or 2 and sign +1 or -1")

    T = cfg.tone_ms / 1000.0
    pre = cfg.pre_ms / 1000.0
    nominal_end = nominal_end_ms(cfg, cond) / 1000.0
    starts = (pre, pre + nominal_end + r["isi_ms"] / 1000.0)
    tones: List[Tone] = []
    ends_b, ends_a, ons, offs = [], [], [], []
    for iv, t0 in zip((1, 2), starts):
        d = delta_ms if iv == ti else 0.0
        b_ms, a_ms = sequence_onsets_ms(cfg, cond, d, sign)
        for k, on in enumerate(b_ms):
            tones.append(Tone(iv, "B", k, cfg.f_b_hz, t0 + on / 1000.0, T, cfg.level_b_db,
                              0.0, False, float(r[f"phase_b{iv}"][k])))
        ends_b.append(t0 + b_ms[-1] / 1000.0)
        ends_a.append(t0 + a_ms[-1] / 1000.0)
        if cond.has_a:
            for k, on in enumerate(a_ms):
                rove = float(r[f"rove{iv}"][k])
                decoy = k in r[f"decoy{iv}"]
                lvl = cfg.level_a_db + rove + (cfg.a_decoy_db if decoy else 0.0)
                tones.append(Tone(iv, "A", k, cfg.f_a_hz, t0 + on / 1000.0, T, lvl, rove, decoy,
                                  float(r[f"phase_a{iv}"][k])))
        mine = [t for t in tones if t.interval == iv]
        ons.append(min(t.onset_s for t in mine))
        offs.append(max(t.onset_s + t.dur_s for t in mine))

    extent = nominal_end + cfg.delta_max_ms / 1000.0
    n_samples = int(math.ceil((starts[1] + extent + cfg.post_ms / 1000.0) * cfg.sample_rate))
    return Trial(phase=phase, condition=cond.name, context_ms=cond.context_ms,
                 ending_shift_ms=cond.ending_shift_ms, delta_ms=float(delta_ms), sign=int(sign),
                 target_interval=ti, is_catch=bool(is_catch), seed=seed, isi_ms=r["isi_ms"],
                 sample_rate=cfg.sample_rate, n_samples=n_samples, seq_start_s=starts,
                 interval_on_s=tuple(ons), interval_off_s=tuple(offs),
                 ending_b_s=tuple(ends_b), ending_a_s=tuple(ends_a), tones=tones)


# ----------------------------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------------------------
def envelope(t: np.ndarray, dur: float, ramp: float) -> np.ndarray:
    """Raised-cosine onset and offset ramps on a flat top; zero outside [0, dur]."""
    env = np.zeros_like(t)
    on = (t >= 0) & (t < ramp)
    top = (t >= ramp) & (t <= dur - ramp)
    off = (t > dur - ramp) & (t <= dur)
    env[on] = np.sin(0.5 * math.pi * t[on] / ramp) ** 2
    env[top] = 1.0
    env[off] = np.sin(0.5 * math.pi * (dur - t[off]) / ramp) ** 2
    return env


class Clipping(RuntimeError):
    pass


def render(cfg: Config, trial: Trial, amplitude) -> Tuple[np.ndarray, List[float]]:
    """The trial as a (n, channels) float32 buffer, and the linear amplitude of every tone.

    `amplitude(freq_hz, level_db)` maps a level to a linear peak amplitude (the calibration).
    """
    sr = cfg.sample_rate
    ramp = cfg.ramp_ms / 1000.0
    mono = np.zeros(trial.n_samples, dtype=np.float64)
    amps = []
    for tn in trial.tones:
        a = float(amplitude(tn.freq_hz, tn.level_db))
        amps.append(a)
        i0 = max(0, int(math.floor(tn.onset_s * sr)))
        i1 = min(trial.n_samples, int(math.ceil((tn.onset_s + tn.dur_s) * sr)) + 1)
        t = np.arange(i0, i1, dtype=np.float64) / sr - tn.onset_s
        mono[i0:i1] += a * envelope(t, tn.dur_s, ramp) * np.sin(2.0 * math.pi * tn.freq_hz * t
                                                                 + tn.phase_rad)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak > cfg.max_peak:
        raise Clipping(f"trial peak {peak:.3f} is above max_peak {cfg.max_peak}: lower the "
                       "levels or raise the system volume and recalibrate")
    out = np.repeat(mono[:, None], cfg.channels, axis=1).astype(np.float32)
    return out, amps


def buffer_sha256(buf: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(buf, dtype="<f4").tobytes()).hexdigest()


def tone_table(trial: Trial, amps: Optional[List[float]] = None) -> List[dict]:
    rows = []
    for i, t in enumerate(trial.tones):
        d = asdict(t)
        d["onset_ms"] = round(t.onset_s * 1000.0, 6)
        if amps is not None:
            d["amplitude"] = amps[i]
        rows.append(d)
    return rows
