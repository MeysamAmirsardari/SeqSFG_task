"""Parameters and geometry for the jittered-pair task.

The stimulus
-------------
Six AB tokens. A comes first; B starts `lag` ms after it, from 0 (synchronous) to `tone_ms` (B
starts exactly at A's offset): the dT axis of Elhilali et al. (2009), Figure 8B, 0% to 100%. In
one of the two intervals the lag of the last or second-to-last token changes by delta, and the
listener says which interval.

The cue the design is built to remove
--------------------------------------
Whatever moves to change the lag also moves against the OTHER tokens. A listener who ignores the
A-B relation and only hears "that tone was off the rhythm" can do the task, and that strategy is
blind to the thing the task is about. L01's second session shows it happening: with tokens
jittered +-35 ms around a 3 Hz grid and B moving alone, the answers followed B's own
irregularity (z = 2.55), and a near-ideal listener using only that cue needs 37 ms, which is what
B alone measured (37.8 ms). Two changes close it.

  timing = "scatter"  The six tokens are placed uniformly at random in a fixed window, subject
                      only to a minimum onset-to-onset gap. The joint density of the onsets is
                      then flat, so moving a tone leaves the sequence exactly as probable as it
                      was, unless the move takes it closer to a neighbour than the minimum gap.
                      Nothing about one tone's timing can be learned or predicted beyond that
                      boundary, and the flat density is the one that leaves least of it: for a
                      shift, the information an ideal listener gains is the density's total
                      variation along the shift, which on a given range is smallest when the
                      density is constant. The most common gap is the minimum itself, so a short
                      gap is never suspicious. ("grid", the original jitter around a grid, is
                      kept so old sessions read correctly.)

  shift = "split"     The lag changes by delta, but A moves delta/2 one way and B delta/2 the
                      other. The token stays centred, and each tone moves half as far against its
                      neighbours as the relation changes. ("b", B moving alone, is the original.)

At 60 ms tones and 3 Hz, a listener judging either tone's own timing, or both, is at 52-53%
correct at the largest shift, and an ideal observer who knows the minimum gap to the millisecond
reaches 60%. The staircase needs 79.4%. The lag inside every other token stays exact, so the A-B relation
is the only thing in the sound that says which interval changed. Decoupled probes (A and B
scattered independently, the same shifts, no relation) measure what is left in every session.

Which way the lag changes
--------------------------
  backward  "in":  B comes earlier relative to A. Cannot open a gap inside the token while delta
            stays below lag + tone_ms, which `validate` enforces. At small lags a large change
            puts B before A (`passes_a_onset`).
  forward   "out": B comes later relative to A. Once delta exceeds tone_ms - lag a silent gap
            opens between A's offset and B's onset; at 100% any change does, so there the task
            is between-channel gap detection. Recorded on every trial (`opens_gap`).
  both      separate "in" and "out" tracks at every lag, run in blocks by direction (the
            listener is told which way to listen) and ordered ABBA across rounds.
  random    one track per lag, the direction drawn per trial (kept for completeness).

The target sits in token N or N-1 with equal probability, so the listener cannot wait for the
very last tone. Thresholds by position are reported as a diagnostic.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class ConfigError(ValueError):
    pass


CORE_LAGS: Tuple[float, ...] = (0.0, 15.0, 30.0, 45.0, 60.0)
# 0, 25, 50, 75 and 100% of a 60 ms tone: the five-level version of Elhilali's dT axis.

FUSION_MARGIN_MS = 25.0
FUSION_TERM_MAX = 8
# A frequency ratio counts as one the pair could fuse at when it is within reach of p/q with
# both terms up to this value. 700/1977 Hz sits midway between 8:3 and 3:1, 5.9% from each.


DIRECTIONS = {"backward": ("in",), "forward": ("out",), "both": ("in", "out"),
              "random": ("random",)}
SIGN = {"in": -1, "out": +1}            # sign of the change of B relative to A
WORD = {"in": "early", "out": "late", "random": "early or late"}            # shift = "b"
REL_WORD = {"in": "earlier", "out": "later", "random": "earlier or later"}  # shift = "split"
TIMINGS = ("scatter", "grid")
CLEARANCES = ("token", "channel")
SHIFTS = ("split", "b")

LEGACY = {"timing": "grid", "shift": "b", "token_gap_ms": 25.0, "max_ioi_ms": None,
          "clearance": "token", "regular_tol_ms": None, "pair_clear_ms": None,
          "token_rove_db": 0.0, "probe_rate": 0.0, "n_context": 0, "context_lags_ms": [],
          "practice_lag_ms": None}
# What every session recorded before scatter timing existed ran under. A stored config without
# these keys reads with these values, and a key holding its legacy value is left out of the
# hash, so the hash of every old preset and session is unchanged.


@dataclass(frozen=True)
class Condition:
    name: str
    lag_ms: float
    kind: str = "pair"          # pair | b_alone | decoupled
    role: str = "main"          # main | control | probe
    direction: str = "in"       # in | out | random
    context_lag_ms: Optional[float] = None    # lag of the first n_context tokens, if any

    def label(self) -> str:
        what = {"b_alone": "B alone", "decoupled": "decoupled probe"}.get(
            self.kind, f"lag {self.lag_ms:g} ms")
        if self.context_lag_ms is not None:
            what = f"context {self.context_lag_ms:g} ms -> {what}"
        return f"{what}, {self.direction}"

    @property
    def stationary(self) -> bool:
        return self.context_lag_ms is None or abs(self.context_lag_ms - self.lag_ms) < 1e-9


@dataclass(frozen=True)
class Config:
    # ---- audio ---------------------------------------------------------------
    sample_rate: int = 48000
    monaural: bool = False
    tone_amplitude: float = 0.05
    tone_level_db_spl: float = 65.0
    level_rove_db: float = 3.0      # each interval moved by U(+-)
    token_rove_db: float = 3.0      # and each token, both tones together, by U(+-) on top
    # The per-token rove makes "one token sounded louder or fuller" useless: a change of the lag
    # changes how long A and B overlap, and so the token's loudness, by about a dB.

    # ---- the tones -----------------------------------------------------------
    f_a_hz: float = 700.0           # the LOW tone, A
    f_b_hz: float = 1977.0          # the HIGH tone, B; 18.0 semitones above A
    tone_ms: float = 60.0
    ramp_ms: float = 5.0
    a_leads: bool = True            # A first, B trails by `lag`

    # ---- the sequence --------------------------------------------------------
    rate_hz: float = 3.0            # mean onset-to-onset rate of the tokens
    n_pairs: int = 6
    timing: str = "scatter"         # scatter | grid; see the module docstring
    token_gap_ms: float = 25.0      # scatter: least silence between two tokens, after any shift
    clearance: str = "token"
    # What token_gap_ms keeps apart under scatter timing.
    #   token    whole tokens: B of one token ends before A of the next starts. Tokens never
    #            interleave, so the duty cycle stays low and lag 100% (B at A's offset) is only a
    #            third of the way to alternation at 3 Hz.
    #   channel  tones of the same frequency only. B of one token may sound while A of the next
    #            starts, as in an ABAB sequence, so the tone can fill half the mean gap and lag 0
    #            B of a token still starts at least token_gap_ms before A of the next, so the
    #            tokens keep their order. Only the same-channel gap limits the jitter, which at
    #            5 Hz leaves about twice the room the token rule would (or any room at all).
    #            Lags stay within the tone: with jittered timing B can sit nearer the NEXT token's
    #            A once it starts after its own A has ended, and the coherence index turns over
    #            there (0.82 at the tone, 0.76 a quarter-tone later, for 75 ms tones at 5 Hz).
    max_ioi_ms: Optional[float] = 700.0    # scatter: no two tokens further apart than this
    regular_tol_ms: Optional[float] = None  # scatter: no 3 gaps in a row within this of each other
    pair_clear_ms: Optional[float] = None   # scatter: every gap >= 2 x largest lag + this
    # The screen. Every interval's timing is drawn from the session seed and checked against these
    # rules; a draw that breaks one is thrown away and redrawn from the same seeded stream, so a
    # session is still exactly reproducible. The rules apply to the timing before any change, in
    # both intervals and every condition alike, so they never depend on which interval changed.
    #   max_ioi_ms      a hole: a gap long enough to split the sequence in two and restart
    #                   build-up
    #   regular_tol_ms  an accidental rhythm: four tokens evenly spaced to within this, a local
    #                   grid against which one moved tone would stand out
    #   pair_clear_ms   an ambiguous pairing: at the largest lag, B starting nearer the NEXT
    #                   token's A than its own. With a whole gap of 2 x lag + this, B is always
    #                   nearer its own A by at least this much before the change
    # A rule has an edge, and a change can carry the target across it where no standard ever
    # goes. The ideal observer in tpair.observer knows every rule, so verify measures that.
    pair_jitter_ms: float = 35.0    # grid only: each pair moves as a unit by U(-J, +J)
    interval_ms: Optional[float] = None    # grid: required; scatter: None = derived
    lead_ms: float = 20.0           # silence before the first possible onset
    isi_ms: float = 500.0

    # ---- conditions ----------------------------------------------------------
    lags_ms: Tuple[float, ...] = CORE_LAGS
    include_b_alone: bool = False
    # B on its own, the same timing, B moving as it does in the token conditions. Replaced by
    # the decoupled probes, which keep A in the sound and remove only the relation.
    probe_rate: float = 0.10
    n_context: int = 0
    context_lags_ms: Tuple[float, ...] = ()
    # Context mode (see tpair.context). The first n_context tokens of every interval run at a
    # context lag, the rest at the test lag (`lags_ms`), and the target is always a test token.
    # One condition per (context lag, test lag, direction). Both intervals of a trial share the
    # context, so the switch itself never says which interval changed.
    # Share of main-block slots that play a decoupled probe at the ceiling: A and B scattered
    # independently, moved exactly as in a target token. Each tone on its own carries what it
    # carries in the real trials; the relation is gone. Never touches a staircase.

    # ---- the target ----------------------------------------------------------
    target_from_end: Tuple[int, ...] = (1, 2)       # 1 = last pair, 2 = second-to-last
    delta_direction: str = "both"                   # in and out tracks; see the module docstring
    shift: str = "split"                            # split | b; see the module docstring

    # ---- the tracked variable ------------------------------------------------
    delta_start_ms: float = 10.0
    delta_min_ms: float = 0.25
    delta_max_ms: float = 55.0
    # "In" opens a gap inside the token at lag 0 once delta reaches tone_ms; 55 keeps every "in"
    # token whole. The same ceiling serves "out", where the gap is part of the stimulus.
    delta_max_b_alone_ms: float = 120.0

    # ---- the adaptive rule (names match tcoh.track.Track) --------------------
    n_down: int = 3
    step_factors: Tuple[float, ...] = (4.0, 2.0, math.sqrt(2.0))
    reversals_per_step: Tuple[int, ...] = (1, 2)
    n_final_reversals: int = 6
    max_trials_per_track: int = 90
    tracks_per_condition: int = 2

    # ---- catch trials --------------------------------------------------------
    catch_rate: float = 0.07
    catch_delta_ms: float = 45.0
    catch_at_lag_ms: Optional[float] = 0.0
    max_catch_miss_rate: float = 0.15

    # ---- procedure -----------------------------------------------------------
    feedback: bool = True
    practice_trials: int = 12
    practice_criterion: float = 0.75
    practice_delta_ms: float = 45.0
    practice_lag_ms: Optional[float] = None     # None = 0, the easiest token
    familiarise: bool = True
    response_timeout_s: float = 6.0
    break_every: int = 1
    break_every_trials: Optional[int] = 70
    max_same_condition_run: int = 2
    tracks_per_block: Optional[int] = None

    # ---- analysis ------------------------------------------------------------
    n_boot: int = 4000
    lapse_max: float = 0.1

    # -------------------------------------------------------------------------
    def replace(self, **kw) -> "Config":
        return dataclasses.replace(self, **kw)

    def to_dict(self) -> dict:
        return {f.name: (list(v) if isinstance(v := getattr(self, f.name), tuple) else v)
                for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ConfigError(f"unknown parameter(s): {sorted(unknown)}")
        kw = dict(d)
        # sessions recorded before A-first existed had B leading; keep reading them correctly
        kw.setdefault("a_leads", False)
        kw.setdefault("delta_max_b_alone_ms", kw.get("delta_max_ms", 120.0))
        for k, v in LEGACY.items():
            kw.setdefault(k, v)
        for name in ("lags_ms", "target_from_end", "step_factors", "reversals_per_step",
                     "context_lags_ms"):
            if kw.get(name) is not None:
                kw[name] = tuple(kw[name])
        return cls(**kw)

    def hash(self) -> str:
        d = {k: v for k, v in self.to_dict().items() if not (k in LEGACY and v == LEGACY[k])}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    @property
    def slot_ms(self) -> float:
        return 1000.0 / self.rate_hz

    @property
    def target_pairs(self) -> Tuple[int, ...]:
        """0-based indices of the pairs that may carry the displacement."""
        return tuple(self.n_pairs - k for k in self.target_from_end)

    def nominal_onsets(self) -> np.ndarray:
        """Grid timing only: where each token sits before its jitter."""
        return self.lead_ms + np.arange(self.n_pairs) * self.slot_ms

    def track_cfg(self, kind: str) -> "Config":
        """The config a track of this kind runs under: only the ceiling differs."""
        return self.replace(delta_max_ms=self.delta_max_b_alone_ms) if kind == "b_alone" else self

    def ceiling_for(self, kind: str) -> float:
        return self.delta_max_b_alone_ms if kind == "b_alone" else self.delta_max_ms

    @property
    def ceiling_max_ms(self) -> float:
        return max(self.delta_max_ms, self.delta_max_b_alone_ms)

    @property
    def directions(self) -> Tuple[str, ...]:
        return DIRECTIONS[self.delta_direction]

    @property
    def can_move_early(self) -> bool:
        return any(d in ("in", "random") for d in self.directions)

    @property
    def can_move_late(self) -> bool:
        return any(d in ("out", "random") for d in self.directions)

    def lag_pct(self, lag_ms: float) -> float:
        return 100.0 * lag_ms / self.tone_ms

    @property
    def interval_len_ms(self) -> float:
        if self.interval_ms is not None:
            return float(self.interval_ms)
        return scatter_geometry(self)["interval_ms"]

    def trial_ms(self) -> float:
        return 2 * self.interval_len_ms + self.isi_ms


DEFAULT = Config()


def conditions(cfg: Config) -> Tuple[Condition, ...]:
    """Every lag, then B alone, for each direction. Names carry the direction only when there is
    more than one, so single-direction sessions keep the names they were recorded under."""
    dirs = DIRECTIONS.get(cfg.delta_direction, ("in",))
    out = []
    for d in dirs:
        sfx = f"_{d}" if len(dirs) > 1 else ""
        if cfg.context_lags_ms:
            out += [Condition(f"ctx_{_tag(C)}_lag_{_tag(L)}{sfx}", L, "pair", "main", d, C)
                    for L in cfg.lags_ms for C in cfg.context_lags_ms]
        else:
            out += [Condition(f"lag_{_tag(L)}{sfx}", L, "pair", "main", d) for L in cfg.lags_ms]
        if cfg.include_b_alone:
            out.append(Condition(f"b_alone{sfx}", 0.0, "b_alone", "control", d))
    return tuple(out)


def probe_conditions(cfg: Config) -> Tuple[Condition, ...]:
    """One decoupled probe per direction; empty when probes are off."""
    if not cfg.probe_rate:
        return ()
    dirs = DIRECTIONS.get(cfg.delta_direction, ("in",))
    return tuple(Condition(f"probe{f'_{d}' if len(dirs) > 1 else ''}", 0.0, "decoupled", "probe",
                           d) for d in dirs)


def _tag(v: float) -> str:
    return f"{v:g}".replace(".", "p")


def shifts(cfg: Config, signed_ms: float) -> Tuple[float, float]:
    """How a change of `signed_ms` in B's timing relative to A's is shared: (A moves, B moves)."""
    if cfg.shift == "split":
        return -signed_ms / 2.0, signed_ms / 2.0
    return 0.0, float(signed_ms)


def ratio_audit(f_a: float, f_b: float, term_max: int = FUSION_TERM_MAX) -> dict:
    """How far the pair sits from every ratio p/q with both terms up to `term_max`."""
    r = max(f_a, f_b) / min(f_a, f_b)
    best = None
    for q in range(1, term_max + 1):
        for p in range(q, term_max + 1):
            if math.gcd(p, q) != 1:
                continue
            m = abs(r - p / q) / (p / q)
            if best is None or m < best[0]:
                best = (m, f"{p}:{q}")
    from tcoh.model import channel_crosstalk
    return {"ratio": r, "octaves": math.log2(r), "semitones": 12 * math.log2(r),
            "nearest": best[1], "mistuning_pct": 100 * best[0],
            "erbs": channel_crosstalk(f_a, f_b)["erbs"]}


# ---------------------------------------------------------------------------- geometry
def token_extent(cfg: Config, kind: str, lag: float, signed_ms: float = 0.0) -> Tuple[float, float]:
    """First onset and last offset of a token's sound, relative to its reference onset, after
    the lag has changed by `signed_ms`. One channel of a probe counts as a token of one tone."""
    a_shift, b_shift = shifts(cfg, signed_ms)
    if kind == "pair":
        a, b = (0.0, lag) if cfg.a_leads else (lag, 0.0)
        a, b = a + a_shift, b + b_shift
        return min(a, b), max(a, b) + cfg.tone_ms
    if kind == "b_alone":
        return b_shift, b_shift + cfg.tone_ms
    s = max(abs(a_shift), abs(b_shift))                     # decoupled: the larger single move
    return -s, s + cfg.tone_ms


def _plan(cfg: Config):
    """(kind, target lag, every lag that shares an interval with it, ceiling)."""
    ctx = tuple(cfg.context_lags_ms) if cfg.n_context else ()
    plan = [("pair", L, (L,) + ctx, cfg.delta_max_ms) for L in cfg.lags_ms]
    if cfg.include_b_alone:
        plan.append(("b_alone", 0.0, (0.0,), cfg.delta_max_b_alone_ms))
    if cfg.probe_rate:
        plan.append(("decoupled", 0.0, (0.0,), cfg.delta_max_ms))
    return plan


def _signs(cfg: Config):
    signs = sorted({SIGN[d] for d in cfg.directions if d in SIGN}) or [-1, 1]
    return [-1, 1] if "random" in cfg.directions else signs


def _moves(cfg: Config):
    """Every (kind, lag, signed change) the session can play, at the extremes of delta."""
    for kind, lag, _, ceiling in _plan(cfg):
        for s in _signs(cfg):
            for d in np.linspace(0.0, ceiling, 23):
                yield kind, lag, s * d


@lru_cache(maxsize=64)
def scatter_geometry(cfg: Config) -> dict:
    """Scatter timing: the minimum onset-to-onset gap, the free slack, and the interval length.

    The minimum gap keeps `token_gap_ms` of silence between any two neighbouring tokens at every
    lag, in every direction, with delta anywhere up to its ceiling; only one token per interval
    ever moves. N onsets placed uniformly subject to that gap need (N - 1) gaps plus slack; the
    slack is chosen so the mean onset-to-onset interval is 1/rate. The interval holds the whole
    window plus the furthest any token can reach past its onset.
    """
    need, reach = 0.0, 0.0
    for kind, lag, present, ceiling in _plan(cfg):
        # any token of the interval can neighbour any other, and the target any of them; with
        # one lag per interval this is the target against its own unmoved shape
        still = [token_extent(cfg, kind, x, 0.0) for x in present]
        for s0, e0 in still:
            for s0b, e0b in still:
                need = max(need, e0 - s0b)
            reach = max(reach, e0)
        for sgn in _signs(cfg):
            for d in np.linspace(0.0, ceiling, 23):
                s1, e1 = token_extent(cfg, kind, lag, sgn * d)
                for s0, e0 in still:
                    need = max(need, e1 - s0, e0 - s1)
                reach = max(reach, e1)
    if getattr(cfg, "clearance", "token") == "channel":
        # one channel's tones: a tone may move its largest single step towards its neighbour,
        # which has not moved (only the target token moves); and a token's trailing onset must
        # stay before the next token's leading one
        step = max(max(abs(x) for x in shifts(cfg, s)) for _, _, s in _moves(cfg))
        trail = 0.0
        for kind, lag, present, ceiling in _plan(cfg):
            for x in present:
                trail = max(trail, x)
            for sgn in _signs(cfg):
                a_sh, b_sh = shifts(cfg, sgn * ceiling)
                trail = max(trail, lag + max(b_sh, -a_sh, 0.0))
        g = max(cfg.tone_ms + step + cfg.token_gap_ms, trail + cfg.token_gap_ms)
    else:
        g = need + cfg.token_gap_ms
    slack = (cfg.n_pairs + 1) * (cfg.slot_ms - g)
    window = max(slack, 0.0) + (cfg.n_pairs - 1) * g
    return {"min_ioi_ms": g, "slack_ms": slack, "window_ms": window, "reach_ms": reach,
            "interval_ms": float(math.ceil(cfg.lead_ms + window + reach + cfg.lead_ms))}


def timing_faults(cfg: Config, x: np.ndarray) -> List[str]:
    """Which screening rules one channel's onsets break (see Config). Empty = a good draw.
    The minimum gap is not here: the scatter construction guarantees it."""
    d = np.diff(np.asarray(x, float))
    out = []
    if d.size == 0:
        return out
    if cfg.max_ioi_ms is not None and d.max() > cfg.max_ioi_ms + 1e-9:
        out.append("hole")
    if cfg.pair_clear_ms is not None:
        need = 2.0 * max(tuple(cfg.lags_ms) + tuple(cfg.context_lags_ms or ())) + cfg.pair_clear_ms
        if d.min() < need - 1e-9:
            out.append("pairing")
    if cfg.regular_tol_ms is not None and d.size >= 3:
        w = np.lib.stride_tricks.sliding_window_view(d, 3)
        if (w.max(1) - w.min(1) < cfg.regular_tol_ms).any():
            out.append("rhythm")
    return out


@lru_cache(maxsize=64)
def realised_timing(cfg: Config, n: int = 4000, seed: int = 11) -> dict:
    """What the scatter actually produces, after the longest-gap cap: mean and spread of the
    onset-to-onset intervals, and the rate that implies."""
    from .stimulus import reference_onsets
    rng = np.random.default_rng(seed)
    log: List[str] = []
    ioi = np.concatenate([np.diff(reference_onsets(cfg, rng, log)) for _ in range(n)])
    return {"mean_ioi_ms": float(ioi.mean()), "sd_ioi_ms": float(ioi.std()),
            "min_ioi_ms": float(ioi.min()), "max_ioi_ms": float(ioi.max()),
            "rate_hz": 1000.0 / float(ioi.mean()),
            "accepted": n / (n + len(log)) if log else 1.0,
            "rejected_by": {k: sum(k in b for b in log) / (n + len(log))
                            for k in sorted({x for b in log for x in b})}}


def extremes_ms(cfg: Config) -> dict:
    """The earliest any sound can start and the latest any can end, over every condition."""
    if cfg.timing == "scatter":
        # the first token never carries the target (validate keeps two tokens before it), so
        # nothing sounds before the window opens
        geo = scatter_geometry(cfg)
        return {"earliest_onset": cfg.lead_ms,
                "latest_offset": cfg.lead_ms + geo["window_ms"] + geo["reach_ms"]}
    J = cfg.pair_jitter_ms
    early = cfg.ceiling_max_ms if (0 in cfg.target_pairs and cfg.can_move_early) else 0.0
    first = cfg.lead_ms - J - early
    end = cfg.nominal_onsets()[-1] + J + cfg.tone_ms
    late = cfg.can_move_late
    last = end + max(cfg.lags_ms or (0.0,)) + (cfg.delta_max_ms if late else 0.0)
    if cfg.include_b_alone:
        last = max(last, end + (cfg.delta_max_b_alone_ms if late else 0.0))
    return {"earliest_onset": first, "latest_offset": last}


def same_channel_clearance_ms(cfg: Config) -> float:
    """Smallest silent gap between two tones of the SAME channel that any trial can produce.

    Grid: the target B comes delta_max early, its own pair was jittered early and the pair
    before late (or the mirror image). Scatter: two neighbouring onsets at the minimum gap, one
    tone moved its largest single step towards the other. Two tones of one frequency closer than
    this would sum.
    """
    if cfg.timing == "scatter":
        step = max(max(abs(x) for x in shifts(cfg, s)) for _, _, s in _moves(cfg))
        return scatter_geometry(cfg)["min_ioi_ms"] - cfg.tone_ms - step
    J = cfg.pair_jitter_ms
    return cfg.slot_ms - 2 * J - cfg.ceiling_max_ms - cfg.tone_ms


@dataclass(frozen=True)
class Derived:
    conditions: Tuple[Condition, ...]
    n_tracks: int
    ratio: dict
    extremes: dict
    same_channel_gap_ms: float
    model_index: Dict[str, float]
    notes: Tuple[str, ...]
    probes: Tuple[Condition, ...] = ()
    geometry: Optional[dict] = None


def validate(cfg: Config) -> Derived:
    from .model import lag_index
    notes: List[str] = []
    if cfg.n_pairs < 4:
        raise ConfigError("n_pairs must be at least 4: the listener needs pairs before the target")
    if any(not (1 <= k <= cfg.n_pairs - 2) for k in cfg.target_from_end):
        raise ConfigError("target_from_end must leave at least two pairs before the target")
    if bool(cfg.n_context) != bool(cfg.context_lags_ms):
        raise ConfigError("context mode needs both n_context and context_lags_ms")
    if cfg.n_context:
        n_test = cfg.n_pairs - cfg.n_context
        if n_test < max(cfg.target_from_end) + 1:
            raise ConfigError(f"{n_test} test tokens cannot hold a target at "
                              f"{max(cfg.target_from_end)} from the end and a test token before "
                              "it: the listener needs the test lag before the change")
        if cfg.n_context < 2:
            raise ConfigError("a context of fewer than two tokens is not a context")
        if any(C < 0 or C > cfg.tone_ms + 1e-9 for C in cfg.context_lags_ms):
            raise ConfigError("context lags must lie between 0 and tone_ms")
        if not any(abs(C - L) < 1e-9 for L in cfg.lags_ms for C in cfg.context_lags_ms):
            notes.append("no context equals a test lag: there is no stationary baseline")
    if cfg.delta_direction not in DIRECTIONS:
        raise ConfigError(f"delta_direction must be one of {sorted(DIRECTIONS)}")
    if cfg.timing not in TIMINGS:
        raise ConfigError(f"timing must be one of {TIMINGS}")
    if cfg.clearance not in CLEARANCES:
        raise ConfigError(f"clearance must be one of {CLEARANCES}")
    if (cfg.regular_tol_ms is not None or cfg.pair_clear_ms is not None) and cfg.timing != "scatter":
        raise ConfigError("the timing screen is defined for scatter timing only")
    if cfg.pair_clear_ms is not None and cfg.pair_clear_ms < 0:
        raise ConfigError("pair_clear_ms must be at least 0")
    if cfg.clearance == "channel" and cfg.timing != "scatter":
        raise ConfigError("clearance = channel is defined for scatter timing only")
    if cfg.clearance == "channel" and cfg.token_gap_ms < FUSION_MARGIN_MS:
        raise ConfigError(f"with clearance = channel, token_gap_ms is the silence between two "
                          f"tones of one frequency and must be at least {FUSION_MARGIN_MS:g}")
    if cfg.shift not in SHIFTS:
        raise ConfigError(f"shift must be one of {SHIFTS}")
    if cfg.delta_direction == "random":
        notes.append("random direction: one track per lag mixes earlier and later changes, and "
                     "later ones open a gap inside the token at large lags")
    if cfg.a_leads and cfg.can_move_early:
        room = min(cfg.lags_ms or (0.0,)) + cfg.tone_ms
        if cfg.delta_max_ms >= room:
            raise ConfigError(
                f"delta_max_ms={cfg.delta_max_ms:g} lets an early B finish before its own A starts "
                f"at lag {min(cfg.lags_ms):g} ms, opening a gap inside the token (limit {room:g}).")
    if any(L > cfg.tone_ms + 1e-9 for L in cfg.lags_ms):
        raise ConfigError("a lag beyond tone_ms leaves a gap inside the token")
    if cfg.delta_min_ms <= 0 or cfg.delta_max_ms <= cfg.delta_min_ms:
        raise ConfigError("need 0 < delta_min_ms < delta_max_ms")
    for name in ("catch_delta_ms", "practice_delta_ms"):
        if getattr(cfg, name) > cfg.delta_max_ms:
            raise ConfigError(f"{name}={getattr(cfg, name):g} is above delta_max_ms; the geometry "
                              "is only checked up to the ceiling")
    if len(cfg.step_factors) != len(cfg.reversals_per_step) + 1:
        raise ConfigError("reversals_per_step must have one entry fewer than step_factors")
    if not 0.0 <= cfg.probe_rate < 0.5 or cfg.catch_rate + cfg.probe_rate >= 0.5:
        raise ConfigError("catch and probe trials together must stay well under half the trials")

    ra = ratio_audit(cfg.f_a_hz, cfg.f_b_hz)
    if ra["octaves"] <= 1.0:
        raise ConfigError(f"A and B are {ra['octaves']:.2f} octaves apart; the design needs more "
                          "than one octave")
    if ra["mistuning_pct"] < 3.0:
        raise ConfigError(
            f"f_b/f_a = {ra['ratio']:.4f} is {ra['mistuning_pct']:.2f}% from {ra['nearest']}. A "
            "simultaneous pair that close to a simple ratio fuses partly through harmonicity, which "
            "at lag 0 would add a grouping cue the design does not manipulate.")
    if ra["erbs"] < 6.0:
        notes.append(f"A and B are only {ra['erbs']:.1f} ERB apart")

    geo = None
    if cfg.timing == "scatter":
        geo = scatter_geometry(cfg)
        if cfg.slot_ms - geo["min_ioi_ms"] < 30.0:
            raise ConfigError(
                f"at {cfg.rate_hz:g} Hz the mean gap between tokens ({cfg.slot_ms:.0f} ms) leaves "
                f"almost nothing above the {geo['min_ioi_ms']:.1f} ms minimum the geometry needs; "
                "the timing would be nearly regular. Lower the rate, the ceiling or the tone.")
        if cfg.max_ioi_ms is not None and cfg.max_ioi_ms < cfg.slot_ms + 100.0:
            raise ConfigError("max_ioi_ms must sit well above the mean gap, or the cap reshapes "
                              "the timing instead of trimming its tail")
        if cfg.interval_ms is not None and cfg.interval_ms < geo["interval_ms"]:
            raise ConfigError(f"interval_ms={cfg.interval_ms:g} is shorter than the "
                              f"{geo['interval_ms']:g} ms the scatter needs")
    elif cfg.interval_ms is None:
        raise ConfigError("grid timing needs interval_ms")
    if cfg.timing == "grid" and cfg.probe_rate:
        raise ConfigError("decoupled probes are defined for scatter timing only")

    ext = extremes_ms(cfg)
    if ext["earliest_onset"] < 5.0:
        raise ConfigError(f"a tone can start at {ext['earliest_onset']:.1f} ms; raise lead_ms")
    if ext["latest_offset"] > cfg.interval_len_ms - 5.0 + 1e-9:
        raise ConfigError(f"a tone can end at {ext['latest_offset']:.1f} ms, past the "
                          f"{cfg.interval_len_ms:g} ms interval")
    gap = same_channel_clearance_ms(cfg)
    if gap < FUSION_MARGIN_MS:
        raise ConfigError(
            f"in the worst case two tones of the same frequency come within {gap:.1f} ms of each "
            f"other (limit {FUSION_MARGIN_MS:g}). Lower delta_max_ms or pair_jitter_ms.")
    if cfg.delta_start_ms * cfg.step_factors[0] > cfg.delta_max_ms:
        notes.append("the first error would call for more than delta_max_ms; the track sits at "
                     "the ceiling until it gets three right")
    peak = 2 * cfg.tone_amplitude * 10 ** ((cfg.level_rove_db + cfg.token_rove_db) / 20.0)
    if peak >= 1.0:
        raise ConfigError(f"two overlapping tones at the maximum rove peak at {peak:.2f} FS")
    if not cfg.include_b_alone and not cfg.probe_rate:
        notes.append("neither B alone nor decoupled probes: nothing in the session measures what "
                     "a listener ignoring the A-B relation could do")
    passing = [L for L in cfg.lags_ms if 0 < L < cfg.delta_max_ms] if cfg.can_move_early else []
    if passing:
        notes.append(
            f"at lag {', '.join(f'{L:g}' for L in passing)} ms a large enough change puts B before "
            "A. Recorded on every trial as passes_a_onset.")
    if cfg.can_move_late and cfg.a_leads:
        opening = [(L, cfg.tone_ms - L) for L in cfg.lags_ms if cfg.tone_ms - L < cfg.delta_max_ms]
        if opening:
            notes.append(
                "a later B opens a silent gap after its A once delta exceeds "
                + ", ".join(f"{max(o, 0):g} ms at {cfg.lag_pct(L):g}%" for L, o in opening)
                + ". Recorded on every trial as opens_gap.")

    conds = conditions(cfg)
    if cfg.n_context:
        from .context import context_index
        model = {c.name: context_index(cfg, c.context_lag_ms, c.lag_ms) for c in conds
                 if c.kind == "pair"}
    else:
        model = {c.name: (float("nan") if c.kind == "b_alone" else lag_index(cfg, c.lag_ms))
                 for c in conds}
    return Derived(conds, len(conds) * cfg.tracks_per_condition, ra, ext, gap, model,
                   tuple(notes), probe_conditions(cfg), geo)


def describe(cfg: Config) -> str:
    d = validate(cfg)
    ra = d.ratio
    word = REL_WORD if cfg.shift == "split" else WORD
    if cfg.timing == "scatter":
        g, t = d.geometry, realised_timing(cfg)
        seq = (f"  sequence   {cfg.n_pairs} tokens scattered uniformly at random in each interval, "
               f"onsets at least {g['min_ioi_ms']:.1f} ms apart"
               + (f" and at most {cfg.max_ioi_ms:g}" if cfg.max_ioi_ms else "")
               + f"; drawn afresh for every interval\n"
               f"             gaps between tokens {t['mean_ioi_ms']:.0f} +- {t['sd_ioi_ms']:.0f} ms "
               f"({t['rate_hz']:.2f} Hz on average); the shortest gap is also the commonest")
    else:
        seq = (f"  sequence   {cfg.n_pairs} pairs at {cfg.rate_hz:g} Hz on average, each moved as "
               f"a unit by U(+-{cfg.pair_jitter_ms:g} ms), drawn afresh for each interval")
    how = ("A and B each move delta/2, opposite ways" if cfg.shift == "split"
           else "B moves, A stays")
    L = [f"jittered A-B pairs  |  config {cfg.hash()}",
         f"  tones      A {cfg.f_a_hz:.0f} Hz, B {cfg.f_b_hz:.0f} Hz: {ra['semitones']:.1f} st "
         f"({ra['octaves']:.2f} oct, {ra['erbs']:.1f} ERB), {ra['mistuning_pct']:.1f}% from "
         f"{ra['nearest']}, the nearest ratio with terms up to {FUSION_TERM_MAX}",
         f"             {cfg.tone_ms:g} ms tones, {cfg.ramp_ms:g} ms ramps; "
         f"{'A first, B trails by the lag' if cfg.a_leads else 'B first, A trails by the lag'}",
         seq,
         f"  level      +-{cfg.level_rove_db:g} dB per interval"
         + (f", +-{cfg.token_rove_db:g} dB per token" if cfg.token_rove_db else ""),
         f"  interval   {cfg.interval_len_ms:g} ms; trial {cfg.interval_len_ms:g} + {cfg.isi_ms:g} + "
         f"{cfg.interval_len_ms:g} ms = {cfg.trial_ms() / 1000:.2f} s",
         f"  target     token "
         + " or ".join(f"{'N' if k == 1 else f'N-{k - 1}'}" for k in cfg.target_from_end)
         + ": B comes " + " / ".join(f"{word[x]} ({x})" for x in cfg.directions)
         + f" by delta relative to A; {how}"
         + ("; blocked by direction, ABBA over rounds" if len(cfg.directions) > 1 else ""),
         f"  delta      starts at {cfg.delta_start_ms:g} ms, capped at {cfg.delta_max_ms:g} ms"
         + (f" (B alone {cfg.delta_max_b_alone_ms:g})" if cfg.include_b_alone else "")
         + f"; same-pitch tones at least {d.same_channel_gap_ms:.0f} ms apart",
         ]
    if d.probes:
        L.append(f"  probes     {100 * cfg.probe_rate:g}% of slots: A and B scattered independently, "
                 f"moved as a target token at {cfg.delta_max_ms:g} ms; never touch a staircase")
    if cfg.n_context:
        L.append(f"  context    the first {cfg.n_context} of {cfg.n_pairs} tokens at the context lag "
                 f"({', '.join(f'{x:g}' for x in cfg.context_lags_ms)} ms), the rest at the test "
                 f"lag; the target is one of the last {max(cfg.target_from_end)}. Model column: "
                 "l2/l1 over the 600 ms ending at the last token")
        L += ["", f"  {'condition':<20} {'dir':>4} {'context':>8} {'test':>6} {'model l2/l1':>12}"]
        for c in d.conditions:
            L.append(f"  {c.name:<20} {c.direction:>4} {cfg.lag_pct(c.context_lag_ms):7g}% "
                     f"{cfg.lag_pct(c.lag_ms):5g}% {d.model_index[c.name]:12.3f}"
                     + ("   stationary" if c.stationary else ""))
        for n in d.notes:
            L.append(f"\n  note: {n}")
        return "\n".join(L)
    L += ["", f"  {'condition':<14} {'dir':>4} {'dT':>6} {'lag ms':>7} {'model l2/l1':>12}  role"]
    for c in d.conditions:
        mv = d.model_index[c.name]
        L.append(f"  {c.name:<14} {c.direction:>4} "
                 f"{('-' if c.kind == 'b_alone' else f'{cfg.lag_pct(c.lag_ms):g}%'):>6} "
                 f"{('-' if c.kind == 'b_alone' else f'{c.lag_ms:g}'):>7} "
                 f"{('-' if mv != mv else f'{mv:.3f}'):>12}  {c.role}"
                 + ("  (what A adds, measured)" if c.kind == "b_alone" else ""))
    for n in d.notes:
        L.append(f"\n  note: {n}")
    return "\n".join(L)
