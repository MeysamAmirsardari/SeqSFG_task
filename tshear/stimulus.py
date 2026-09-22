"""Building and rendering one trial of the sheared-figure task.

An *interval* is `n_repeats` repetitions of the four-tone figure, the last of which may have
one tone displaced. A *trial* is two intervals -- one standard, one target -- in a randomised
order, separated by silence.

The invariants, which are the experiment
-----------------------------------------
1. *The two intervals of a trial differ in the position of ONE tone in ONE repetition and in
   nothing else.* Same frequencies, same count, same durations, same ramps, same starting
   phases, same step, same onsets everywhere else. Not matched on average -- equal, sample for
   sample, apart from the displaced tone.

2. *Every non-target channel is identical across step sizes in its own timing.* Each tone
   repeats isochronously at `rate_hz` whatever the shear; what the step changes is only the
   PHASE of one channel relative to another. So the within-channel rhythm available to a
   listener is the same at every step, which is what makes a flat curve mean 'they used the
   within-channel rhythm' rather than 'the manipulation did nothing'.

3. *The displacement is the same physical operation at every step.* One tone moves by delta.
   Nothing is scaled, stretched or substituted.

4. *Duration and total energy are constant.* Intervals are padded to a length computed from the
   widest step and the largest jitter, and moving a tone cannot change how much energy the
   interval contains.

`tshear.verify` checks all four numerically rather than taking this docstring's word for it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import Condition, Config, ConfigError, onsets_ms, target_clearance


@dataclass(frozen=True)
class Interval:
    """One sequence. `onsets` is (n_repeats, n_tones) in ms from the interval start."""
    onsets_ms: np.ndarray
    delta_ms: float               # signed displacement applied to the target tone (0 = standard)
    level_db: float
    step_pct: float
    kind: str                     # figure | single

    @property
    def is_target(self) -> bool:
        return self.delta_ms != 0.0


@dataclass(frozen=True)
class Trial:
    first: Interval
    second: Interval
    target_position: int          # 1 or 2 -- which interval carries the displacement
    delta_ms: float               # unsigned, as the track asked for
    delta_signed_ms: float
    delta_realised_ms: float      # after rounding the onset to a whole sample
    condition: str
    phases: np.ndarray            # (n_tones, n_repeats) starting phases
    crosses_neighbour: bool
    is_catch: bool = False


def figure_onsets(cfg: Config, cond: Condition) -> np.ndarray:
    """(n_repeats, n_tones) onsets with no jitter. Tones absent from `single` are NaN."""
    o = np.stack([onsets_ms(cfg, cond.step_pct, r) for r in range(cfg.n_repeats)])
    if cond.kind == "single":
        # only the target's own channel: the within-channel limit, with no figure at all
        mask = np.full(cfg.n_tones, np.nan)
        mask[cfg.target_index] = 0.0
        o = o + mask[None, :]
    return o


def build_interval(cfg: Config, cond: Condition, delta_signed_ms: float,
                   level_db: float) -> Interval:
    o = figure_onsets(cfg, cond)
    if delta_signed_ms:
        o = o.copy()
        o[-1, cfg.target_index] += delta_signed_ms
    return Interval(o, float(delta_signed_ms), float(level_db), cond.step_pct, cond.kind)


def build_trial(cfg: Config, cond: Condition, delta_ms: float, rng: np.random.Generator,
                target_position: Optional[int] = None, direction: Optional[int] = None,
                is_catch: bool = False) -> Trial:
    """One trial. Every random draw happens HERE and is handed to both intervals.

    Drawing inside `build_interval` instead would make invariant 1 a hope about generator
    state; drawing once and passing the values makes it a property of the code.
    """
    delta_ms = float(np.clip(delta_ms, cfg.delta_min_ms, cfg.delta_max_ms))
    if target_position is None:
        target_position = int(rng.integers(1, 3))
    if direction is None:
        direction = {"random": 0, "forward": +1, "backward": -1}[cfg.delta_direction]
        if direction == 0:
            direction = int(rng.choice([-1, 1]))
    signed = direction * delta_ms

    phases = rng.uniform(0.0, 2 * math.pi, size=(cfg.n_tones, cfg.n_repeats))
    rove = (rng.uniform(-cfg.level_rove_db, cfg.level_rove_db, size=2)
            if cfg.level_rove_db else np.zeros(2))

    std = build_interval(cfg, cond, 0.0, float(rove[0]))
    tgt = build_interval(cfg, cond, signed, float(rove[1]))
    first, second = (tgt, std) if target_position == 1 else (std, tgt)
    clear = target_clearance(cfg, cond.step_pct, delta_ms, direction)
    return Trial(first, second, int(target_position), float(delta_ms), float(signed),
                 _realised_shift_ms(cfg, std, tgt), cond.name, phases,
                 bool(clear["crosses_neighbour"]) and cond.kind == "figure", is_catch)


def _realised_shift_ms(cfg: Config, std: Interval, tgt: Interval) -> float:
    """The shift that survives rounding the onset to a whole sample -- what was heard."""
    k = cfg.target_index
    a = int(round(std.onsets_ms[-1, k] * cfg.sample_rate / 1000.0))
    b = int(round(tgt.onsets_ms[-1, k] * cfg.sample_rate / 1000.0))
    return (b - a) * 1000.0 / cfg.sample_rate


# ----------------------------------------------------------------------------
def tone_envelope(cfg: Config, n: Optional[int] = None) -> np.ndarray:
    n = n if n is not None else int(round(cfg.tone_ms * cfg.sample_rate / 1000.0))
    r = min(int(round(cfg.ramp_ms * cfg.sample_rate / 1000.0)), n // 2)
    w = np.ones(n)
    if r > 0:
        up = 0.5 * (1.0 - np.cos(np.pi * np.arange(r) / r))
        w[:r], w[n - r:] = up, up[::-1]
    return w


def render_interval(cfg: Config, iv: Interval, phases: Optional[np.ndarray] = None,
                    only_tone: Optional[int] = None,
                    total_ms: Optional[float] = None) -> np.ndarray:
    """One interval as a mono waveform, padded to the length the configuration calls for.

    `only_tone` renders a single channel on its own, so `verify` can compare channels
    separately; the LENGTH still comes from the whole interval.
    """
    fs = cfg.sample_rate
    total_ms = cfg.interval_ms() if total_ms is None else total_ms
    n_total = int(round(total_ms * fs / 1000.0))
    x = np.zeros(n_total)
    env = tone_envelope(cfg)
    amp = cfg.tone_amplitude * 10.0 ** (iv.level_db / 20.0)
    t = np.arange(env.size) / fs
    for k, f in enumerate(cfg.freqs_hz):
        if only_tone is not None and k != only_tone:
            continue
        for rep in range(iv.onsets_ms.shape[0]):
            on = iv.onsets_ms[rep, k]
            if not np.isfinite(on):
                continue                      # absent in the single-tone control
            i = int(round(on * fs / 1000.0))
            if i < 0 or i >= n_total:
                raise ConfigError(
                    f"a tone at {on:.2f} ms falls outside the {total_ms:.0f} ms interval")
            j = min(n_total, i + env.size)
            ph = 0.0 if phases is None else float(phases[k, rep])
            x[i:j] += amp * env[: j - i] * np.sin(2 * np.pi * f * t[: j - i] + ph)
    return x


def render_trial(cfg: Config, tr: Trial) -> np.ndarray:
    gap = np.zeros(int(round(cfg.isi_ms * cfg.sample_rate / 1000.0)))
    return np.concatenate([render_interval(cfg, tr.first, tr.phases), gap,
                           render_interval(cfg, tr.second, tr.phases)])


def to_output(cfg: Config, x: np.ndarray) -> np.ndarray:
    if not cfg.monaural:
        return np.stack([x, x], axis=-1)
    return np.stack([x, np.zeros_like(x)], axis=-1)
