"""Building and rendering one trial of the jittered-pair task.

What `tpair.verify` checks, and why each matters
-------------------------------------------------
1. *The lag is exact in every token of every interval* except the target token, whose lag is
   the standard lag plus the signed change. The timing moves tokens, never the relation inside
   one.

2. *The two intervals' timing is drawn independently*, so neither channel can be compared across
   intervals, and the only consistent structure in either interval is the A-B lag. The same
   reasoning as a level rove, applied to time.

3. *One tone's own timing says nothing.* Under scatter timing each channel's onsets are spread
   with a flat density over everything the minimum gap allows, in both intervals. Moving one tone
   of each channel by delta/2 leaves that density where it was except at the minimum gap, so a
   listener judging either tone against its own neighbours, or both, is left with that boundary
   and nothing else.

4. *Only the target token changes, and only the way its condition says.* Everything else in the
   target interval is where the timing put it.

5. *Duration is fixed and nothing overlaps its own channel.* Every interval is the same length
   whatever the condition and wherever the tokens fall, and the geometry is validated so that no
   two tones of one frequency can sum, and no two tokens come closer than the configured gap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import SIGN, Condition, Config, ConfigError, scatter_geometry, shifts, timing_faults


@dataclass(frozen=True)
class Interval:
    b_onsets_ms: np.ndarray
    a_onsets_ms: np.ndarray       # NaN throughout for the B-alone control
    jitter_ms: np.ndarray         # grid timing: each pair's displacement; scatter: NaN
    target_pair: Optional[int]    # which token changed, or None for the standard
    delta_ms: float               # signed change of B relative to A there (negative = in)
    level_db: float
    lag_ms: float
    kind: str
    ref_ms: Optional[np.ndarray] = None      # each token's reference onset, before any move
    ref_b_ms: Optional[np.ndarray] = None    # decoupled probes: B's own reference onsets
    token_db: Optional[np.ndarray] = None    # per-token level, on top of level_db
    a_shift_ms: float = 0.0                  # how far the target's A moved
    b_shift_ms: float = 0.0                  # and its B
    context_lag_ms: Optional[float] = None   # lag of the first n_context tokens, if any

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
    pairing_preserved: bool       # the target's B is still nearer its own A than any other
    passes_a_onset: bool          # the change put B at or before its own A's onset
    condition: str
    phases: Tuple[np.ndarray, np.ndarray]     # per interval, shape (2, n_pairs): B row, A row
    is_catch: bool = False
    opens_gap: bool = False       # the change left a silent gap between A and B
    kind: str = "pair"

    @property
    def target(self) -> Interval:
        return self.first if self.target_position == 1 else self.second

    @property
    def standard(self) -> Interval:
        return self.second if self.target_position == 1 else self.first


# ---------------------------------------------------------------------------- timing
def reference_onsets(cfg: Config, rng: np.random.Generator,
                     log: Optional[list] = None) -> np.ndarray:
    """Each token's reference onset (A's, before any move) for one interval.

    grid     the nominal grid plus U(+-pair_jitter_ms) per pair
    scatter  N points uniform in the window subject to the minimum gap: sort N uniform draws
             over the slack and add k times the gap to the k-th. That is exactly the uniform
             density on the set of allowed configurations. The longest-gap cap is applied by
             rejection, which keeps the density flat on what remains.
    """
    n = cfg.n_pairs
    if cfg.timing == "grid":
        return cfg.nominal_onsets() + rng.uniform(-cfg.pair_jitter_ms, cfg.pair_jitter_ms, n)
    geo = scatter_geometry(cfg)
    base = cfg.lead_ms + np.arange(n) * geo["min_ioi_ms"]
    for _ in range(10000):
        r = base + np.sort(rng.uniform(0.0, geo["slack_ms"], n))
        bad = timing_faults(cfg, r)
        if not bad:
            return r
        if log is not None:
            log.append(tuple(bad))          # one entry per draw thrown away
    raise ConfigError("the timing screen rejects nearly every draw")


def _draw_refs(cfg: Config, kind: str, rng: np.random.Generator, log: Optional[list] = None):
    a = reference_onsets(cfg, rng, log)
    return a, (reference_onsets(cfg, rng, log) if kind == "decoupled" else a)


def build_interval(cfg: Config, cond: Condition, ref_a: np.ndarray, level_db: float,
                   target_pair: Optional[int] = None, delta_signed_ms: float = 0.0,
                   ref_b: Optional[np.ndarray] = None, jitter: Optional[np.ndarray] = None,
                   token_db: Optional[np.ndarray] = None) -> Interval:
    n = cfg.n_pairs
    ref_a = np.asarray(ref_a, float)
    ref_b = ref_a if ref_b is None else np.asarray(ref_b, float)
    ctx = cond.context_lag_ms if (cfg.n_context and cond.kind == "pair") else None
    if cond.kind == "pair":
        lags = np.full(n, float(cond.lag_ms))
        if ctx is not None:
            lags[:cfg.n_context] = ctx
        lead, trail = ref_a, ref_a + lags
        a, b = (lead.copy(), trail.copy()) if cfg.a_leads else (trail.copy(), lead.copy())
    elif cond.kind == "b_alone":
        a, b = np.full(n, np.nan), ref_a.copy()
    elif cond.kind == "decoupled":
        a, b = ref_a.copy(), ref_b.copy()
    else:
        raise ConfigError(f"unknown condition kind {cond.kind}")
    a_shift = b_shift = 0.0
    if target_pair is not None:
        a_shift, b_shift = shifts(cfg, delta_signed_ms)
        if np.isfinite(a[target_pair]):
            a[target_pair] += a_shift
        b[target_pair] += b_shift
    return Interval(b, a, np.full(n, np.nan) if jitter is None else np.asarray(jitter, float),
                    target_pair, float(delta_signed_ms), float(level_db), cond.lag_ms, cond.kind,
                    ref_a, ref_b if cond.kind == "decoupled" else None,
                    np.zeros(n) if token_db is None else np.asarray(token_db, float),
                    float(a_shift), float(b_shift), ctx)


def build_trial(cfg: Config, cond: Condition, delta_ms: float, rng: np.random.Generator,
                target_position: Optional[int] = None, target_pair: Optional[int] = None,
                direction: Optional[int] = None, jitters=None, refs=None,
                is_catch: bool = False, timing_log: Optional[list] = None) -> Trial:
    """One trial. Every random draw happens here, once.

    `jitters` (grid) or `refs` (scatter: one array per interval, or an (A, B) pair of arrays for
    a decoupled probe) fix the timing instead of drawing it.
    """
    delta_ms = float(np.clip(delta_ms, cfg.delta_min_ms, cfg.ceiling_for(cond.kind)))
    if target_position is None:
        target_position = int(rng.integers(1, 3))
    if target_pair is None:
        target_pair = int(rng.choice(cfg.target_pairs))
    if direction is None:
        direction = SIGN.get(cond.direction, 0)
        if direction == 0:
            direction = int(rng.choice([-1, 1]))
    signed = direction * delta_ms
    n = cfg.n_pairs
    if cfg.timing == "grid":
        J = cfg.pair_jitter_ms
        if jitters is None:
            # independently for each interval: see invariant 2
            jitters = (rng.uniform(-J, J, n), rng.uniform(-J, J, n))
        pairs = [(cfg.nominal_onsets() + np.asarray(j, float),) * 2 for j in jitters]
    else:
        jitters = (None, None)
        if refs is None:
            pairs = [_draw_refs(cfg, cond.kind, rng, timing_log) for _ in (1, 2)]
        else:
            pairs = [tuple(r) if isinstance(r, (tuple, list)) else (r, r) for r in refs]
    phases = (rng.uniform(0, 2 * math.pi, (2, n)), rng.uniform(0, 2 * math.pi, (2, n)))
    rove = (rng.uniform(-cfg.level_rove_db, cfg.level_rove_db, 2)
            if cfg.level_rove_db else np.zeros(2))
    T = cfg.token_rove_db
    tok = ((rng.uniform(-T, T, n), rng.uniform(-T, T, n)) if T else (np.zeros(n), np.zeros(n)))

    # timing/phases/roves are indexed by INTERVAL position, so which one is the target is
    # decided only by target_position, never by which draw came first
    ivs = []
    for pos in (1, 2):
        is_tgt = pos == target_position
        ra, rb = pairs[pos - 1]
        ivs.append(build_interval(cfg, cond, ra, float(rove[pos - 1]),
                                  target_pair if is_tgt else None, signed if is_tgt else 0.0,
                                  ref_b=rb, jitter=jitters[pos - 1], token_db=tok[pos - 1]))
    tgt = ivs[target_position - 1]
    t = target_pair
    fs = cfg.sample_rate
    q = lambda x: int(round(x * fs / 1000.0))
    b_now, b_was = tgt.b_onsets_ms[t], tgt.b_onsets_ms[t] - tgt.b_shift_ms
    if np.isfinite(tgt.a_onsets_ms[t]):
        a_now, a_was = tgt.a_onsets_ms[t], tgt.a_onsets_ms[t] - tgt.a_shift_ms
        realised = ((q(b_now) - q(a_now)) - (q(b_was) - q(a_was))) * 1000.0 / fs
    else:
        realised = (q(b_now) - q(b_was)) * 1000.0 / fs
    pair = cond.kind == "pair"
    passes = bool(pair and signed < 0 and cond.lag_ms > 0
                  and tgt.b_onsets_ms[t] <= tgt.a_onsets_ms[t] + 1e-9)
    return Trial(ivs[0], ivs[1], int(target_position), float(delta_ms), float(signed),
                 float(realised), int(t), pairing_preserved(tgt), passes, cond.name, phases,
                 is_catch, opens_gap=bool(pair and token_gaps(cfg, tgt)[t] > 1e-9),
                 kind=cond.kind)


def token_lags(cfg: Config, iv: Interval) -> np.ndarray:
    """Onset of the trailing tone minus the leading one, per token, as the design defines it."""
    return (iv.b_onsets_ms - iv.a_onsets_ms) if cfg.a_leads else (iv.a_onsets_ms - iv.b_onsets_ms)


def token_gaps(cfg: Config, iv: Interval) -> np.ndarray:
    """Silence between the end of the earlier tone of each token and the start of the later one.
    Negative = they overlap, 0 = they abut. NaN for the B-alone control."""
    a, b = iv.a_onsets_ms, iv.b_onsets_ms
    return np.maximum(a, b) - (np.minimum(a, b) + cfg.tone_ms)


def between_token_silence(cfg: Config, iv: Interval) -> np.ndarray:
    """Silence from the last offset of each token to the first onset of the next (pairs only)."""
    a, b = iv.a_onsets_ms, iv.b_onsets_ms
    start, end = np.minimum(a, b), np.maximum(a, b) + cfg.tone_ms
    return start[1:] - end[:-1]


def pairing_preserved(iv: Interval) -> bool:
    """Is the target's B still nearer its own A than any other A in the interval?"""
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

    `level=False` drops both roves, for checks that compare channels across intervals.
    """
    fs = cfg.sample_rate
    n_total = int(round(cfg.interval_len_ms * fs / 1000.0))
    x = np.zeros(n_total)
    env = tone_envelope(cfg)
    tok = iv.token_db if iv.token_db is not None else np.zeros(len(iv.b_onsets_ms))
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
                                  f"{cfg.interval_len_ms:g} ms interval")
            ph = 0.0 if phases is None else float(phases[row, k])
            amp = cfg.tone_amplitude * (10.0 ** ((iv.level_db + tok[k]) / 20.0) if level else 1.0)
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
