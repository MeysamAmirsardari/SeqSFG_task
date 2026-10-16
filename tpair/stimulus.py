"""Building and rendering one trial of the jittered-pair task.

What `tpair.verify` checks, and why each matters
-------------------------------------------------
1. *The lag is exact in every pair of every interval*, apart from the one displaced B tone.
   The jitter moves pairs, never the relation inside one.

2. *The jitter is drawn independently for the two intervals.* So neither channel can be compared
   across intervals, and the only consistent structure in either interval is the A-B lag. This is
   the same reasoning as a level rove, applied to time.

3. *A carries no information.* No A tone ever moves relative to its own pair, so the A channel of
   the target interval is a draw from exactly the distribution the standard's A channel comes
   from. A listener who attends only to A is at chance, by construction.

4. *Only the B tone of pair N or N-1 moves, and only early.* Everything else in the target
   interval is where its pair's jitter puts it.

5. *Duration is fixed and nothing overlaps its own channel.* Every interval is exactly
   `interval_ms` long whatever the condition, and the geometry is validated so that no two tones
   of one frequency can sum, whatever the jitter and the displacement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import Condition, Config, ConfigError


@dataclass(frozen=True)
class Interval:
    b_onsets_ms: np.ndarray
    a_onsets_ms: np.ndarray       # NaN throughout for the B-alone control
    jitter_ms: np.ndarray         # the whole-pair displacement each pair was given
    target_pair: Optional[int]    # which pair's B was displaced, or None for the standard
    delta_ms: float               # signed displacement of that B (negative = early)
    level_db: float
    lag_ms: float
    kind: str

    @property
    def is_target(self) -> bool:
        return self.target_pair is not None


@dataclass(frozen=True)
class Trial:
    first: Interval
    second: Interval
    target_position: int
    delta_ms: float
    delta_signed_ms: float
    delta_realised_ms: float
    target_pair: int
    pairing_preserved: bool       # the displaced B is still nearer its own A than any other
    condition: str
    phases: Tuple[np.ndarray, np.ndarray]     # per interval, shape (2, n_pairs): B row, A row
    is_catch: bool = False

    @property
    def target(self) -> Interval:
        return self.first if self.target_position == 1 else self.second

    @property
    def standard(self) -> Interval:
        return self.second if self.target_position == 1 else self.first


def build_interval(cfg: Config, cond: Condition, jitter: np.ndarray, level_db: float,
                   target_pair: Optional[int] = None, delta_signed_ms: float = 0.0) -> Interval:
    b = cfg.nominal_onsets() + np.asarray(jitter, float)
    a = b + cond.lag_ms if cond.kind == "pair" else np.full(cfg.n_pairs, np.nan)
    if target_pair is not None:
        b = b.copy()
        b[target_pair] += delta_signed_ms
    return Interval(b, a, np.asarray(jitter, float), target_pair, float(delta_signed_ms),
                    float(level_db), cond.lag_ms, cond.kind)


def build_trial(cfg: Config, cond: Condition, delta_ms: float, rng: np.random.Generator,
                target_position: Optional[int] = None, target_pair: Optional[int] = None,
                direction: Optional[int] = None, jitters=None, is_catch: bool = False) -> Trial:
    """One trial. Every random draw happens here, once."""
    delta_ms = float(np.clip(delta_ms, cfg.delta_min_ms, cfg.delta_max_ms))
    if target_position is None:
        target_position = int(rng.integers(1, 3))
    if target_pair is None:
        target_pair = int(rng.choice(cfg.target_pairs))
    if direction is None:
        direction = {"backward": -1, "forward": +1, "random": 0}[cfg.delta_direction]
        if direction == 0:
            direction = int(rng.choice([-1, 1]))
    signed = direction * delta_ms
    J = cfg.pair_jitter_ms
    if jitters is None:
        # independently for each interval: see invariant 2
        jitters = (rng.uniform(-J, J, cfg.n_pairs), rng.uniform(-J, J, cfg.n_pairs))
    phases = (rng.uniform(0, 2 * math.pi, (2, cfg.n_pairs)),
              rng.uniform(0, 2 * math.pi, (2, cfg.n_pairs)))
    rove = (rng.uniform(-cfg.level_rove_db, cfg.level_rove_db, 2)
            if cfg.level_rove_db else np.zeros(2))

    # jitters/phases/rove are indexed by INTERVAL position, so which one is the target is
    # decided only by target_position, never by which draw came first
    ivs = []
    for pos in (1, 2):
        is_tgt = pos == target_position
        ivs.append(build_interval(cfg, cond, jitters[pos - 1], float(rove[pos - 1]),
                                  target_pair if is_tgt else None, signed if is_tgt else 0.0))
    tgt = ivs[target_position - 1]
    nominal_b = tgt.b_onsets_ms[target_pair] - signed
    fs = cfg.sample_rate
    realised = (int(round(tgt.b_onsets_ms[target_pair] * fs / 1000.0))
                - int(round(nominal_b * fs / 1000.0))) * 1000.0 / fs
    return Trial(ivs[0], ivs[1], int(target_position), float(delta_ms), float(signed),
                 float(realised), int(target_pair), pairing_preserved(tgt), cond.name,
                 phases, is_catch)


def pairing_preserved(iv: Interval) -> bool:
    """Is the displaced B still nearer its own A than any other A in the interval?"""
    if iv.target_pair is None or iv.kind != "pair":
        return True
    t = iv.target_pair
    d = np.abs(iv.a_onsets_ms - iv.b_onsets_ms[t])
    return int(np.argmin(d)) == t


# ----------------------------------------------------------------------------
def tone_envelope(cfg: Config) -> np.ndarray:
    n = int(round(cfg.tone_ms * cfg.sample_rate / 1000.0))
    r = min(int(round(cfg.ramp_ms * cfg.sample_rate / 1000.0)), n // 2)
    w = np.ones(n)
    if r > 0:
        up = 0.5 * (1.0 - np.cos(np.pi * np.arange(r) / r))
        w[:r], w[n - r:] = up, up[::-1]
    return w


def render_interval(cfg: Config, iv: Interval, phases: Optional[np.ndarray] = None,
                    which: str = "both", level: bool = True) -> np.ndarray:
    """One interval as a mono waveform. `which` = both | a | b renders one channel alone.

    `level=False` drops the rove, for checks that compare channels across intervals.
    """
    fs = cfg.sample_rate
    n_total = int(round(cfg.interval_ms * fs / 1000.0))
    x = np.zeros(n_total)
    env = tone_envelope(cfg)
    amp = cfg.tone_amplitude * (10.0 ** (iv.level_db / 20.0) if level else 1.0)
    t = np.arange(env.size) / fs
    chans = []
    if which in ("both", "b"):
        chans.append((iv.b_onsets_ms, cfg.f_b_hz, 0))
    if which in ("both", "a"):
        chans.append((iv.a_onsets_ms, cfg.f_a_hz, 1))
    for onsets, f, row in chans:
        for k, on in enumerate(onsets):
            if not np.isfinite(on):
                continue
            i = int(round(on * fs / 1000.0))
            if i < 0 or i + env.size > n_total:
                raise ConfigError(f"a tone at {on:.1f} ms does not fit the "
                                  f"{cfg.interval_ms:g} ms interval")
            ph = 0.0 if phases is None else float(phases[row, k])
            x[i:i + env.size] += amp * env * np.sin(2 * np.pi * f * t + ph)
    return x


def render_trial(cfg: Config, tr: Trial) -> np.ndarray:
    gap = np.zeros(int(round(cfg.isi_ms * cfg.sample_rate / 1000.0)))
    return np.concatenate([render_interval(cfg, tr.first, tr.phases[0]), gap,
                           render_interval(cfg, tr.second, tr.phases[1])])


def to_output(cfg: Config, x: np.ndarray) -> np.ndarray:
    if not cfg.monaural:
        return np.stack([x, x], axis=-1)
    return np.stack([x, np.zeros_like(x)], axis=-1)
