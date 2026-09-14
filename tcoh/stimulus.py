"""Building and rendering one trial: two intervals that differ in the position of a single tone.

An *interval* is a pair of onset lists, one per channel, plus the shift applied to the last B
tone. A *trial* is two intervals -- one standard (shift 0) and one target (shift +-delta) --
in a randomised order, separated by silence.

The invariants, which are the experiment
-----------------------------------------
Everything the design claims rests on four properties of what this module builds, and each one
is checked numerically by ``tcoh.verify`` rather than asserted here:

1. *The two intervals of a trial are identical apart from one tone's position.* Same
   frequencies, same count, same durations, same ramps, same starting phases, same A onsets.
   The only difference is where the final B tone starts. Not "statistically matched" -- equal,
   sample for sample, everywhere except in the neighbourhood of that tone.

2. *The B channel does not know what condition it is in.* B's onsets are the same grid in every
   condition, so the cue available from B alone -- is the B rhythm regular? -- is constant
   across the whole dT sweep. Any dependence of performance on dT must come from A.

3. *The displacement is the same physical operation at every dT.* The final B tone moves by
   delta. Nothing else is scaled, stretched or substituted, so a difference between conditions
   cannot be a difference in what the listener was asked to detect.

4. *Duration and energy are constant.* Every interval is padded to the same length, computed
   from the widest lag and the largest shift the configuration allows, and shifting a tone
   cannot change how much energy either channel contains.

Who moves, and which tone
--------------------------
The upper tone (B) is the one displaced, and the lower tone (A) never moves, following Elhilali
et al. (2009). The final A tone is present in every condition except `absent` and `nopartner`,
and it sits wherever the reference mode puts it -- the point of the `sync` and `tempo` modes is
precisely that they put it somewhere else.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import Condition, Config, ConfigError, Derived, validate


@dataclass(frozen=True)
class Interval:
    """One sequence of the pair. Times in milliseconds from the start of the interval."""
    a_onsets_ms: np.ndarray          # may be empty (a_kind='absent')
    b_onsets_ms: np.ndarray
    delta_ms: float                  # signed displacement applied to the last B tone (0 = standard)
    level_db: float                  # rove applied to this interval, dB re the nominal amplitude
    a_kind: str
    lag_pct: float

    @property
    def is_target(self) -> bool:
        return self.delta_ms != 0.0

    def b_target_onset_ms(self) -> float:
        return float(self.b_onsets_ms[-1])


@dataclass(frozen=True)
class Trial:
    first: Interval
    second: Interval
    target_position: int             # 1 or 2 -- which interval carries the shift
    delta_ms: float                  # unsigned, as requested by the track
    delta_signed_ms: float           # what was actually applied, sign included
    delta_realised_ms: float         # after rounding the onset to a whole sample
    condition: str
    phases: Tuple[np.ndarray, np.ndarray]
    is_catch: bool = False

    def intervals(self) -> Tuple[Interval, Interval]:
        return self.first, self.second


# ----------------------------------------------------------------------------
# onsets
# ----------------------------------------------------------------------------
def b_onsets(cfg: Config, n_precursor: Optional[int] = None) -> np.ndarray:
    """The reference grid. Identical in every condition -- see invariant 2."""
    n = (cfg.n_precursor if n_precursor is None else n_precursor) + 1
    return cfg.lead_ms + np.arange(n, dtype=float) * cfg.soa_ms


def n_a_tones(cfg: Config, cond: Condition) -> int:
    """How many A tones this condition has, without building them."""
    n = (cfg.n_precursor if cond.n_precursor is None else cond.n_precursor) + 1
    if cond.a_kind == "absent":
        return 0
    if cond.a_kind == "nopartner":
        return n - 1
    if cond.a_kind == "pair_only":
        return 1
    return n


def scramble_onsets(rng: np.random.Generator, lo_ms: float, hi_ms: float, n: int,
                    min_gap_ms: float, max_tries: int = 500) -> np.ndarray:
    """`n` random onsets in [lo, hi] with at least `min_gap` between consecutive ones.

    Rejection sampling with a guaranteed fallback: if the window is too tight the tones are
    laid out evenly with a small random offset, which is still aperiodic relative to B but
    never fails. `validate` sizes the window so the fallback is not reached in practice.
    """
    if n <= 0:
        return np.zeros(0, dtype=float)
    for _ in range(max_tries):
        t = np.sort(rng.uniform(lo_ms, hi_ms, size=n))
        if n == 1 or float(np.diff(t).min()) >= min_gap_ms:
            return t
    step = (hi_ms - lo_ms) / max(n - 1, 1)
    return lo_ms + np.arange(n) * step + rng.uniform(-step / 8, step / 8, size=n)


def a_onsets(cfg: Config, cond: Condition, scramble: Optional[np.ndarray] = None) -> np.ndarray:
    """Where the A tones go, given the condition's kind and reference mode.

    `scramble` is required for a_kind='scrambled' and holds the PRECURSOR onsets, drawn once
    per trial by `build_trial` and passed in for both intervals. Passing the values rather than
    a generator is what makes "the two intervals differ in one tone and nothing else" a
    property of the code and not a hope about generator state.
    """
    b = b_onsets(cfg, cond.n_precursor)
    lag = cfg.lag_ms(cond.lag_pct)

    if cond.a_kind == "absent":
        return np.zeros(0, dtype=float)

    if cond.reference == "tempo":
        # A is isochronous at its own rate, phase-set so the final A lands on the unshifted
        # final B. This is Elhilali's manipulation: no pedestal, no rhythmic anomaly, but the
        # precursor lag ramps from tone to tone instead of holding at one value.
        soa_a = cfg.tone_ms + float(cond.tempo_gap_ms)
        k = np.arange(b.size, dtype=float)[::-1]
        out = b[-1] - k * soa_a
    else:
        out = b + lag
        if cond.reference == "sync":
            # pull the final A back onto the unshifted final B, so the judged interval is
            # always "synchronous versus delta" and no pedestal grows with dT.
            out = out.copy()
            out[-1] = b[-1]

    if cond.a_kind == "scrambled":
        # precursors only: the final A tone stays exactly where the coherent condition puts it,
        # so the interval the listener judges at the end has the same geometry in both. What is
        # destroyed is the regularity of the A sequence, not the final pair.
        if scramble is None or len(scramble) != out.size - 1:
            raise ValueError(f"a_kind='scrambled' needs {out.size - 1} precursor onsets")
        out = np.concatenate([np.asarray(scramble, dtype=float), out[-1:]])

    if cond.a_kind == "pair_only":
        out = out[-1:]

    if cond.a_kind == "nopartner":
        out = out[:-1]

    return out


def scramble_window(cfg: Config, cond: Condition) -> Tuple[float, float]:
    """The span the scrambled precursors are drawn from: the one the coherent ones occupy."""
    b = b_onsets(cfg, cond.n_precursor)
    lag = cfg.lag_ms(cond.lag_pct)
    return float(b[0] + lag), float(b[-2] + lag)


def build_interval(cfg: Config, cond: Condition, delta_signed_ms: float, level_db: float,
                   scramble: Optional[np.ndarray] = None) -> Interval:
    b = b_onsets(cfg, cond.n_precursor)
    if delta_signed_ms:
        b = b.copy()
        b[-1] += delta_signed_ms
    return Interval(a_onsets(cfg, cond, scramble), b, float(delta_signed_ms), float(level_db),
                    cond.a_kind, cond.lag_pct)


def build_trial(cfg: Config, cond: Condition, delta_ms: float, rng: np.random.Generator,
                target_position: Optional[int] = None, direction: Optional[int] = None,
                is_catch: bool = False) -> Trial:
    """One trial: a standard and a target interval, in a randomised order.

    The jitter displacements, the starting phases and the two level roves are drawn ONCE here
    and handed to both intervals, which is what makes invariant 1 hold exactly rather than on
    average. Deliberately cheap: this runs on every trial of a session, so it does no
    validation and computes no model predictions.
    """
    delta_ms = float(np.clip(delta_ms, cfg.delta_min_ms, cfg.delta_max_ms))
    if target_position is None:
        target_position = int(rng.integers(1, 3))
    if direction is None:
        direction = {"random": 0, "forward": +1, "backward": -1}[cfg.delta_direction]
        if direction == 0:
            direction = int(rng.choice([-1, 1]))
    signed = direction * delta_ms

    n_a = n_a_tones(cfg, cond)
    n_b = (cfg.n_precursor if cond.n_precursor is None else cond.n_precursor) + 1

    # Independent sub-streams, so that a draw whose SIZE depends on the condition -- the A
    # phases, the scrambled onsets -- cannot shift the stream the B phases come from. Without
    # this the B channel is not bit-identical across conditions for a given seed, which is an
    # invariant the design claims and `verify.invariants` checks.
    ss = np.random.SeedSequence(int(rng.integers(0, 2 ** 63)))
    r_b, r_a, r_s, r_l = (np.random.default_rng(c) for c in ss.spawn(4))
    phases = (r_a.uniform(0, 2 * math.pi, size=max(n_a, 1)),
              r_b.uniform(0, 2 * math.pi, size=n_b))
    scramble = None
    if cond.a_kind == "scrambled":
        lo, hi = scramble_window(cfg, cond)
        scramble = scramble_onsets(r_s, lo, hi, max(n_a - 1, 0), cfg.scramble_min_gap)
    rove = (r_l.uniform(-cfg.level_rove_db, cfg.level_rove_db, size=2)
            if cfg.level_rove_db else np.zeros(2))

    std = build_interval(cfg, cond, 0.0, rove[0], scramble)
    tgt = build_interval(cfg, cond, signed, rove[1], scramble)
    first, second = (tgt, std) if target_position == 1 else (std, tgt)

    realised = _realised_shift_ms(cfg, std, tgt)
    return Trial(first, second, int(target_position), float(delta_ms), float(signed), realised,
                 cond.name, phases, is_catch)


def _realised_shift_ms(cfg: Config, std: Interval, tgt: Interval) -> float:
    """The shift that survives rounding the onset to a whole sample.

    At the bottom of a track delta is a fraction of a millisecond and the sample grid is
    1/48 ms, so nominal and realised differ. The analysis uses the realised value: it is what
    the listener heard.
    """
    a = int(round(std.b_onsets_ms[-1] * cfg.sample_rate / 1000.0))
    b = int(round(tgt.b_onsets_ms[-1] * cfg.sample_rate / 1000.0))
    return (b - a) * 1000.0 / cfg.sample_rate


# ----------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------
def tone_envelope(cfg: Config, n: Optional[int] = None) -> np.ndarray:
    """Raised-cosine-gated unit envelope of one tone, in samples."""
    n = n if n is not None else int(round(cfg.tone_ms * cfg.sample_rate / 1000.0))
    r = min(int(round(cfg.ramp_ms * cfg.sample_rate / 1000.0)), n // 2)
    w = np.ones(n)
    if r > 0:
        up = 0.5 * (1.0 - np.cos(np.pi * np.arange(r) / r))
        w[:r], w[n - r:] = up, up[::-1]
    return w


def render_interval(cfg: Config, iv: Interval, d: Optional[Derived] = None,
                    phases: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                    total_ms: Optional[float] = None) -> np.ndarray:
    """One interval as a mono waveform, padded to the length its precursor count calls for.

    `total_ms` overrides that length. It exists for callers that render ONE channel of an
    interval by blanking the other: the length has to come from the whole interval, not from
    the stripped copy, whose B array may be empty.
    """
    d = d or validate(cfg)
    fs = cfg.sample_rate
    from .config import interval_ms
    if total_ms is None:
        total_ms = interval_ms(cfg, iv.b_onsets_ms.size - 1)
    n_total = int(round(total_ms * fs / 1000.0))
    x = np.zeros(n_total)
    env = tone_envelope(cfg)
    amp = cfg.tone_amplitude * 10.0 ** (iv.level_db / 20.0)
    t = np.arange(env.size) / fs
    for onsets, f, ph in ((iv.a_onsets_ms, cfg.f_a_hz, None if phases is None else phases[0]),
                          (iv.b_onsets_ms, cfg.f_b_hz, None if phases is None else phases[1])):
        for k, on in enumerate(onsets):
            i = int(round(on * fs / 1000.0))
            if i < 0 or i >= n_total:
                raise ConfigError(f"a tone at {on:.2f} ms falls outside the {total_ms:.0f} ms interval")
            j = min(n_total, i + env.size)
            p = 0.0 if ph is None else float(ph[k])
            x[i:j] += amp * env[: j - i] * np.sin(2 * np.pi * f * t[: j - i] + p)
    return x


def render_trial(cfg: Config, tr: Trial, d: Optional[Derived] = None) -> np.ndarray:
    """The whole trial: interval, silence, interval. Mono, or duplicated to stereo by the caller."""
    d = d or validate(cfg)
    gap = np.zeros(int(round(cfg.isi_ms * cfg.sample_rate / 1000.0)))
    return np.concatenate([render_interval(cfg, tr.first, d, tr.phases), gap,
                           render_interval(cfg, tr.second, d, tr.phases)])


def to_output(cfg: Config, x: np.ndarray) -> np.ndarray:
    """Mono to what the sound card is given: diotic by default, left-only if monaural."""
    if not cfg.monaural:
        return np.stack([x, x], axis=-1)
    return np.stack([x, np.zeros_like(x)], axis=-1)


def channel_envelopes(cfg: Config, iv: Interval, d: Optional[Derived] = None,
                      fs_env: float = 1000.0) -> np.ndarray:
    """The two channel envelopes the coherence model consumes, shape (2, n).

    Built from the onset lists rather than by demodulating the waveform, because that is what
    the model's first stage is defined on and because it keeps the model's input free of the
    level rove, which is not part of the hypothesis.
    """
    d = d or validate(cfg)
    from .config import interval_ms
    from .model import sequence_envelope
    n_ms = interval_ms(cfg, iv.b_onsets_ms.size - 1)
    return np.stack([sequence_envelope(iv.a_onsets_ms, cfg.tone_ms, cfg.ramp_ms, n_ms, fs_env),
                     sequence_envelope(iv.b_onsets_ms, cfg.tone_ms, cfg.ramp_ms, n_ms, fs_env)])
