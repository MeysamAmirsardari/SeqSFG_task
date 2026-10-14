"""Building and rendering one trial: the rove scheme is the experiment.

Notation, per interval, with R repetitions and the probe on the last:

    c   ~ U(+-interval_rove)   everything in the interval            (0 by default)
    d   ~ U(+-rove)            the target's offset from the others, all repetitions
    v   ~ U(+-rove)            A only: the level all four tones jump to on the probe
    w   ~ U(+-rove)            B only: the jump of the three other tones on the probe
    u_k ~ U(+-flanker_jitter)  B only: an extra, independent jump per other tone (0 by default)

                       repetitions 1..R-1        probe, type A          probe, type B
    other tones        c                         c + v                  c + w + u_k
    target             c + d                     c + v  (+ delta)       c + d  (+ delta)

Every quantity is drawn afresh for each interval, so nothing about the standard interval
predicts the signal interval. Read off the table:

    target re the others on the probe     A: delta exactly           B: d - w - mean(u) + delta
    target re its own earlier tones       A: v - d + delta           B: delta exactly

and the same cue that is exact in one type is noise of +-rove in the other. The repetitions
before the probe are drawn by the same code in both types, the jump of the other three tones on
the probe has the same distribution in both (v and w), and so does the target's jump relative to
them (-d in A, -w in B). Up to the probe a listener cannot tell the types apart.

An earlier version (`tshear.levelprobe`) roved A differently on every repetition and left the
other tones of B flat until the probe. Its B trials kept a third cue nearly intact: the change in
timbre from the earlier repetitions to the probe varied only by the mean of three independent
flanker jumps, so an ideal observer of that cue reached 79% at 4 dB, half the intended ceiling.
Here that cue varies by w and needs 8.5 dB. `observer.ceilings` lists every single-cue
observer for both types.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import Condition, Config, ConfigError, captor_onsets, delta_db, onsets


@dataclass(frozen=True)
class Interval:
    onsets_ms: np.ndarray                     # (R, K), NaN where a tone is absent
    gains_db: np.ndarray                      # (R, K) re tone_amplitude, NaN where absent
    captor_onsets_ms: Optional[np.ndarray]    # (R-1, K) or None
    captor_gains_db: Optional[np.ndarray]
    is_signal: bool
    latent: Tuple[float, ...]                 # (c, d, v, w, *u): what was drawn


@dataclass(frozen=True)
class Trial:
    first: Interval
    second: Interval
    target_position: int                      # 1 or 2: which interval has the louder target
    condition: str
    task: str
    kind: str
    step_ms: float
    srs_db: float
    delta_db: float
    phases: np.ndarray                        # (K, R), shared by both intervals
    captor_phases: np.ndarray                 # (K, R-1), shared by both intervals
    total_ms: float
    is_catch: bool = False

    def intervals(self) -> Tuple[Interval, Interval]:
        return self.first, self.second

    @property
    def signal(self) -> Interval:
        return self.first if self.target_position == 1 else self.second

    @property
    def standard(self) -> Interval:
        return self.second if self.target_position == 1 else self.first


# ----------------------------------------------------------------------------
def draw_latent(cfg: Config, rng: np.random.Generator) -> Tuple[float, ...]:
    """Every rove variable for one interval, always drawn and always in this order.

    Drawing the same number of values whatever the task keeps the generator's state independent
    of the condition, so a resumed session and a fresh one see the same draws.
    """
    c = float(rng.uniform(-cfg.interval_rove_db, cfg.interval_rove_db))
    d = float(rng.uniform(-cfg.rove_db, cfg.rove_db))
    v = float(rng.uniform(-cfg.rove_db, cfg.rove_db))
    w = float(rng.uniform(-cfg.rove_db, cfg.rove_db))
    u = rng.uniform(-cfg.flanker_jitter_db, cfg.flanker_jitter_db, size=cfg.n_tones - 1)
    return (c, d, v, w, *[float(x) for x in u])


def gains_from_latent(cfg: Config, task: str, kind: str, latent: Tuple[float, ...],
                      delta: float, signal: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """The (R, K) gain matrix and, for `capture`, the (R-1, K) captor gains."""
    R, K, t = cfg.n_repeats, cfg.n_tones, cfg.target_index
    oth = cfg.others
    c, d, v, w, *u = latent
    if task == "none":                        # catch: no rove at all, both cues intact
        c = d = v = w = 0.0
        u = [0.0] * len(oth)
    g = np.full((R, K), c, dtype=float)
    g[:-1, t] += d
    if task == "A":
        g[-1, :] = c + v
    elif task == "B":
        g[-1, t] = c + d
        g[-1, oth] = c + w + np.asarray(u, dtype=float)
    elif task == "none":
        g[-1, :] = c
    else:
        raise ConfigError(f"unknown task {task!r}")
    if signal:
        g[-1, t] += float(delta)
    cap = None
    if kind == "capture":
        cap = np.full((R - 1, K), c, dtype=float)
        cap[:, t] = np.nan
    if kind == "alone":
        g[:, oth] = np.nan
    return g, cap


def build_trial(cfg: Config, cond: Condition, srs: float, rng: np.random.Generator,
                target_position: Optional[int] = None, is_catch: bool = False,
                delta_override_db: Optional[float] = None) -> Trial:
    """One trial. Every draw happens here, in a fixed order, and both intervals share the
    timing and the starting phases: they differ only in their gains."""
    srs = float(np.clip(srs, cfg.srs_min_db, cfg.srs_max_db))
    delta = delta_db(srs) if delta_override_db is None else float(delta_override_db)
    if target_position is None:
        target_position = int(rng.integers(1, 3))
    R, K = cfg.n_repeats, cfg.n_tones
    phases = rng.uniform(0.0, 2 * math.pi, size=(K, R))
    captor_phases = rng.uniform(0.0, 2 * math.pi, size=(K, R - 1))
    lat_sig, lat_std = draw_latent(cfg, rng), draw_latent(cfg, rng)
    o = onsets(cfg, cond)
    co = captor_onsets(cfg, cond)
    gs, cs = gains_from_latent(cfg, cond.task, cond.kind, lat_sig, delta, True)
    gn, cn = gains_from_latent(cfg, cond.task, cond.kind, lat_std, delta, False)
    sig = Interval(o, gs, co, cs, True, lat_sig)
    std = Interval(o, gn, co, cn, False, lat_std)
    first, second = (sig, std) if target_position == 1 else (std, sig)
    return Trial(first, second, int(target_position), cond.name, cond.task, cond.kind,
                 float(cond.step_ms), srs, float(delta), phases, captor_phases,
                 cfg.interval_ms(cond.step_ms), bool(is_catch))


# ----------------------------------------------------------------------------
def tone_envelope(cfg: Config) -> np.ndarray:
    n = int(round(cfg.tone_ms * cfg.sample_rate / 1000.0))
    r = min(int(round(cfg.ramp_ms * cfg.sample_rate / 1000.0)), n // 2)
    w = np.ones(n)
    if r > 0:
        up = 0.5 * (1.0 - np.cos(np.pi * np.arange(r) / r))
        w[:r], w[n - r:] = up, up[::-1]
    return w


def _events(iv: Interval, phases: np.ndarray, captor_phases: np.ndarray):
    """(onset_ms, tone, gain_db, phase, is_captor) for every tone the interval contains."""
    R, K = iv.onsets_ms.shape
    for r in range(R):
        for k in range(K):
            if np.isfinite(iv.onsets_ms[r, k]):
                yield iv.onsets_ms[r, k], k, iv.gains_db[r, k], phases[k, r], False
    if iv.captor_onsets_ms is not None:
        for r in range(iv.captor_onsets_ms.shape[0]):
            for k in range(K):
                if np.isfinite(iv.captor_onsets_ms[r, k]):
                    yield (iv.captor_onsets_ms[r, k], k, iv.captor_gains_db[r, k],
                           captor_phases[k, r], True)


def render_interval(cfg: Config, iv: Interval, phases: np.ndarray,
                    captor_phases: Optional[np.ndarray] = None, total_ms: Optional[float] = None,
                    only_tone: Optional[int] = None, captors: bool = True) -> np.ndarray:
    fs = cfg.sample_rate
    if total_ms is None:
        finite = iv.onsets_ms[np.isfinite(iv.onsets_ms)]
        total_ms = float(finite.max()) + cfg.tone_ms + cfg.tail_ms
    n = int(round(total_ms * fs / 1000.0))
    x = np.zeros(n)
    env = tone_envelope(cfg)
    t = np.arange(env.size) / fs
    cph = captor_phases if captor_phases is not None else np.zeros((cfg.n_tones, cfg.n_repeats))
    for on, k, g, ph, is_captor in _events(iv, phases, cph):
        if (only_tone is not None and k != only_tone) or (is_captor and not captors):
            continue
        i = int(round(on * fs / 1000.0))
        if i < 0 or i + env.size > n:
            raise ConfigError(f"a tone at {on:.1f} ms does not fit in the {total_ms:.0f} ms "
                              "interval")
        a = cfg.tone_amplitude * 10.0 ** (g / 20.0)
        x[i:i + env.size] += a * env * np.sin(2 * np.pi * cfg.freqs_hz[k] * t + ph)
    return x


def render_trial(cfg: Config, tr: Trial) -> np.ndarray:
    gap = np.zeros(int(round(cfg.isi_ms * cfg.sample_rate / 1000.0)))
    return np.concatenate([
        render_interval(cfg, tr.first, tr.phases, tr.captor_phases, tr.total_ms), gap,
        render_interval(cfg, tr.second, tr.phases, tr.captor_phases, tr.total_ms)])


def to_output(cfg: Config, x: np.ndarray) -> np.ndarray:
    if not cfg.monaural:
        return np.stack([x, x], axis=-1)
    return np.stack([x, np.zeros_like(x)], axis=-1)


def gains_json(g: Optional[np.ndarray]) -> str:
    """A gain matrix as compact JSON for the trial log; NaN (absent tone) becomes null."""
    if g is None:
        return ""
    import json
    return json.dumps([[None if not np.isfinite(x) else round(float(x), 4) for x in row]
                       for row in np.asarray(g)], separators=(",", ":"))


def gains_from_json(s: str) -> Optional[np.ndarray]:
    if not s:
        return None
    import json
    return np.array([[np.nan if x is None else float(x) for x in row] for row in json.loads(s)])
