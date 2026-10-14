"""The level-probe version of the sheared figure: is the target bound with the other tones?

The jitter task probes timing, and timing is also what the step manipulates, so a step changes
the physics of the judgment as well as the perceptual organisation. Here the probe is level
instead. The target's last tone is `delta_db` louder in one interval, and the listener says
which. A level change is the same physical event at every step.

What makes it a binding measure is the level ROVE, which differs between two trial types and
leaves the listener exactly one usable cue in each:

  A  common rove, new on every repetition, all four tones together. The target's loudness
     relative to its own earlier tones is now noise; what survives exactly is its loudness
     relative to the other three AT THE SAME MOMENT, i.e. the timbre of the last token.
     Easy if the four tones are heard as one thing, hard if they are separate streams.

  B  the target's whole stream shifted by one amount per interval, and the three other tones
     shifted independently on the last token. What survives exactly is the target relative to
     its own earlier tones (the stream shift cancels); the timbre of the last token is noise.
     Easy if the target is its own stream, hard if it is fused into the chord.

Without a rove both cues are available and the change is audible either way, which is why the
rove is the design and not a detail. No rove of finite size destroys a cue completely, so each
trial type has a ceiling: the threshold of an ideal observer that uses ONLY the destroyed cue.
`ceiling` computes it, and at the default 10 dB both sit near 8 dB.

This module is the prototype behind `notebooks/level_probe.ipynb`. For experiments use `tcbound`:
here a fused listener can still do type B from the change of timbre into the last repetition
(an ideal observer of that cue alone needs only 4 dB), which `tcbound` closes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .config import DEFAULT, Config
from .stimulus import tone_envelope

TASKS = ("A", "B", "none")
STEPS_MS: Tuple[float, ...] = (0.0, 10.0, 25.0, 45.0, 75.0)
# In milliseconds rather than as a percentage of the isochronous step: grouping by onset
# asynchrony is a matter of milliseconds whatever the repetition rate. 75 ms keeps clear of the
# 83.3 ms step at which the combined onset train becomes isochronous.
ROVE_DB = 10.0
AMPLITUDE = 0.04
# Peak of ONE tone before gains. At step 0 all four tones overlap; with a +10 dB rove and a
# 12 dB increment on top, the in-phase worst case is 0.04 * 3.16 * (3 + 3.98) = 0.88 of full
# scale. At the tshear default of 0.05 the same case clips.
TARGET_P = 0.7937          # 3-down 1-up


@dataclass(frozen=True)
class LevelInterval:
    onsets_ms: np.ndarray      # (n_repeats, n_tones)
    gains_db: np.ndarray       # (n_repeats, n_tones), re AMPLITUDE
    is_signal: bool


@dataclass(frozen=True)
class LevelTrial:
    first: LevelInterval
    second: LevelInterval
    target_position: int       # 1 or 2: which interval has the louder target
    task: str
    step_ms: float
    delta_db: float
    rove_db: float
    phases: np.ndarray         # (n_tones, n_repeats), shared by both intervals

    def intervals(self) -> Tuple[LevelInterval, LevelInterval]:
        return self.first, self.second


# ----------------------------------------------------------------------------
def onsets(cfg: Config, step_ms: float) -> np.ndarray:
    rep = np.arange(cfg.n_repeats, dtype=float)[:, None]
    k = np.arange(cfg.n_tones, dtype=float)[None, :]
    return cfg.lead_ms + rep * cfg.period_ms + k * float(step_ms)


def interval_ms(cfg: Config, max_step_ms: float = max(STEPS_MS)) -> float:
    """One length for every step up to `max_step_ms`, so duration says nothing about the step."""
    return (cfg.lead_ms + (cfg.n_repeats - 1) * cfg.period_ms
            + (cfg.n_tones - 1) * max_step_ms + cfg.tone_ms + cfg.tail_ms)


def gains(cfg: Config, task: str, delta_db: float, rove_db: float, signal: bool,
          rng: np.random.Generator) -> np.ndarray:
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    R, K, t = cfg.n_repeats, cfg.n_tones, cfg.target_index
    g = np.zeros((R, K))
    if task == "A":
        g += rng.uniform(-rove_db, rove_db, size=(R, 1))
    elif task == "B":
        g[:, t] += rng.uniform(-rove_db, rove_db)
        others = [j for j in range(K) if j != t]
        g[-1, others] += rng.uniform(-rove_db, rove_db, size=len(others))
    if signal:
        g[-1, t] += delta_db
    return g


def make_trial(task: str, step_ms: float, delta_db: float, rng: np.random.Generator,
               rove_db: float = ROVE_DB, cfg: Config = DEFAULT,
               target_position: Optional[int] = None) -> LevelTrial:
    """One trial. Every draw happens here, so the two intervals share everything but gains."""
    if target_position is None:
        target_position = int(rng.integers(1, 3))
    o = onsets(cfg, step_ms)
    phases = rng.uniform(0.0, 2 * math.pi, size=(cfg.n_tones, cfg.n_repeats))
    sig = LevelInterval(o, gains(cfg, task, delta_db, rove_db, True, rng), True)
    std = LevelInterval(o, gains(cfg, task, delta_db, rove_db, False, rng), False)
    first, second = (sig, std) if target_position == 1 else (std, sig)
    return LevelTrial(first, second, target_position, task, float(step_ms), float(delta_db),
                      float(rove_db), phases)


def render(cfg: Config, iv: LevelInterval, phases: np.ndarray,
           total_ms: Optional[float] = None, amplitude: float = AMPLITUDE,
           only_tone: Optional[int] = None) -> np.ndarray:
    fs = cfg.sample_rate
    total_ms = interval_ms(cfg) if total_ms is None else total_ms
    n = int(round(total_ms * fs / 1000.0))
    x = np.zeros(n)
    env = tone_envelope(cfg)
    t = np.arange(env.size) / fs
    for rep in range(iv.onsets_ms.shape[0]):
        for k, f in enumerate(cfg.freqs_hz):
            if only_tone is not None and k != only_tone:
                continue
            i = int(round(iv.onsets_ms[rep, k] * fs / 1000.0))
            if i < 0 or i >= n:
                raise ValueError(f"a tone at {iv.onsets_ms[rep, k]:.1f} ms falls outside "
                                 f"the {total_ms:.0f} ms interval")
            j = min(n, i + env.size)
            a = amplitude * 10.0 ** (iv.gains_db[rep, k] / 20.0)
            x[i:j] += a * env[: j - i] * np.sin(2 * np.pi * f * t[: j - i] + phases[k, rep])
    return x


# ----------------------------------------------------------------------------
# the two cues, and observers that use them
# ----------------------------------------------------------------------------
def cues(cfg: Config, gains_db: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(timbre, own_stream) for one interval or a stack of them, in dB.

    timbre      the target's last tone against the mean of the other three, same moment
    own_stream  the target's last tone against the mean of its own earlier tones
    """
    g = np.asarray(gains_db, dtype=float)
    t = cfg.target_index
    others = [j for j in range(cfg.n_tones) if j != t]
    timbre = g[..., -1, t] - g[..., -1, others].mean(axis=-1)
    own = g[..., -1, t] - g[..., :-1, t].mean(axis=-1)
    return timbre, own


def _draws(cfg: Config, task: str, delta_db: float, rove_db: float, n: int, seed: int):
    rng = np.random.default_rng(seed)
    sig = np.stack([gains(cfg, task, delta_db, rove_db, True, rng) for _ in range(n)])
    std = np.stack([gains(cfg, task, delta_db, rove_db, False, rng) for _ in range(n)])
    return sig, std


def p_correct(cfg: Config, task: str, delta_db: float, rove_db: float = ROVE_DB,
              w_timbre: float = 1.0, noise_timbre_db: float = 0.0,
              noise_own_db: float = 0.0, n: int = 20000, seed: int = 0) -> float:
    """Proportion correct for a listener who weighs the two cues and adds internal noise.

    w_timbre = 1 reads only the timbre of the last token, 0 only the target's own stream.
    With no internal noise this is an ideal observer of that mixture.
    """
    sig, std = _draws(cfg, task, delta_db, rove_db, n, seed)
    rng = np.random.default_rng(seed + 1)
    ts, os_ = cues(cfg, sig)
    tn, on = cues(cfg, std)
    ds = (w_timbre * (ts + rng.normal(0, noise_timbre_db, n))
          + (1 - w_timbre) * (os_ + rng.normal(0, noise_own_db, n)))
    dn = (w_timbre * (tn + rng.normal(0, noise_timbre_db, n))
          + (1 - w_timbre) * (on + rng.normal(0, noise_own_db, n)))
    return float(np.mean(ds > dn) + 0.5 * np.mean(ds == dn))


def ceiling(cfg: Config, task: str, rove_db: float = ROVE_DB, n: int = 20000,
            seed: int = 0) -> float:
    """Threshold of an ideal observer using ONLY the cue this trial type destroys.

    A listener who cannot use the surviving cue falls back to this. A threshold well below it
    means the surviving cue was used; that is the inference the design rests on.
    """
    w = {"A": 0.0, "B": 1.0}[task]          # the destroyed cue for each type
    sig0, std0 = _draws(cfg, task, 0.0, rove_db, n, seed)
    ts, os_ = cues(cfg, sig0)
    tn, on = cues(cfg, std0)
    base = w * ts + (1 - w) * os_ - (w * tn + (1 - w) * on)
    # adding delta to the signal's target last tone adds delta to BOTH cues
    lo, hi = 0.0, 80.0
    for _ in range(50):
        mid = (lo + hi) / 2
        pc = float(np.mean(base + mid > 0))
        lo, hi = (mid, hi) if pc < TARGET_P else (lo, mid)
    return (lo + hi) / 2


def worst_case_peak(cfg: Config, rove_db: float = ROVE_DB, max_delta_db: float = 12.0,
                    amplitude: float = AMPLITUDE) -> float:
    """Every tone at the top of the rove, the target at the top of the track, all in phase."""
    g = 10 ** (rove_db / 20.0)
    return amplitude * g * ((cfg.n_tones - 1) + 10 ** (max_delta_db / 20.0))
