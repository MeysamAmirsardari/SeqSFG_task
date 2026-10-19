"""Parameters and geometry for the jittered-pair task.

The stimulus
-------------
Six AB tokens at an average of 3 Hz fill a 2-second interval. A always comes first; B starts
`lag` ms after it, from 0 (synchronous) to `tone_ms` (B starts exactly at A's offset). That is
the dT axis of Elhilali et al. (2009), Figure 8B, 0% to 100%, with 75 ms tones as in their
simulation. There is never a gap between A and B inside a token. Each token is jittered AS A
WHOLE by an independent uniform draw of +-`pair_jitter_ms`, so the lag inside every token is
exact while neither tone on its own keeps a regular rhythm. In one of the two intervals, the B
tone of the last or second-to-last token comes `delta` ms early, and the listener says which.

Why the whole-pair jitter is the point
---------------------------------------
Both earlier tasks left the target's OWN rhythm intact. In tcoh B ran on a fixed grid; in
tshear every channel was isochronous at 3 Hz. Either way a listener could judge the displaced
tone against its own previous repetitions and ignore the other tone entirely -- a strategy that
is blind to the very manipulation the task is about. Here that reference is randomised away. A
B-only listener has to spot one early tone in a sequence whose tones already wander by
+-40 ms, which takes about 43 ms of displacement (see `observer.single_channel_threshold`); the
A-B lag, meanwhile, is exact in every pair. Below that level the only route to the answer runs
through the relation between the two tones.

The jitter is drawn independently for each interval of a trial. Sharing it would make the two
intervals identical apart from the target, and the B channel could then be compared across
intervals directly -- the cue the jitter exists to remove, put back.

Which tone moves, and which way
--------------------------------
B moves EARLY, into more overlap with its own A. Moving it late would open a silent gap inside
the token at 100%, where B already starts at A's offset, and the design never has one. Early
shifts cannot open a gap as long as delta stays below lag + tone_ms, which `validate` enforces
for every token condition. The price: at small lags a large enough shift carries B to A's onset
and past it. Every trial records whether that happened (`passes_a_onset`).

The target sits in pair N or N-1 with equal probability, so the listener cannot simply wait for
the very last tone, and the last pair's special status -- nothing follows it -- is diluted
rather than built into every trial. Thresholds by position are reported as a diagnostic.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, fields
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class ConfigError(ValueError):
    pass


CORE_LAGS: Tuple[float, ...] = (0.0, 18.75, 37.5, 56.25, 75.0)
# 0, 25, 50, 75 and 100% of a 75 ms tone: the five-level version of Elhilali's dT axis.

FUSION_MARGIN_MS = 25.0
FUSION_TERM_MAX = 8
# A frequency ratio counts as one the pair could fuse at when it is within reach of p/q with
# both terms up to this value. The chosen pair is 7.7% from the nearest of them.


@dataclass(frozen=True)
class Condition:
    name: str
    lag_ms: float
    kind: str = "pair"          # pair | b_alone
    role: str = "main"          # main | control

    def label(self) -> str:
        return "B alone" if self.kind == "b_alone" else f"lag {self.lag_ms:g} ms"


@dataclass(frozen=True)
class Config:
    # ---- audio ---------------------------------------------------------------
    sample_rate: int = 48000
    monaural: bool = False
    tone_amplitude: float = 0.05
    tone_level_db_spl: float = 65.0
    level_rove_db: float = 3.0

    # ---- the tones -----------------------------------------------------------
    f_a_hz: float = 700.0           # the LOW tone, A
    f_b_hz: float = 2262.0          # the HIGH tone, B; 20.3 semitones above A
    tone_ms: float = 75.0
    ramp_ms: float = 5.0
    a_leads: bool = True            # A first, B trails by `lag`

    # ---- the sequence --------------------------------------------------------
    rate_hz: float = 3.0            # average pair rate
    n_pairs: int = 6
    pair_jitter_ms: float = 35.0    # each pair moves as a unit by U(-J, +J)
    interval_ms: float = 2000.0
    lead_ms: float = 50.0           # nominal onset of the first pair's B tone
    isi_ms: float = 500.0

    # ---- conditions ----------------------------------------------------------
    lags_ms: Tuple[float, ...] = CORE_LAGS
    include_b_alone: bool = True
    # B on its own, same jitter, same displacement: what a listener ignoring A could do. Every
    # pair condition is read against it.

    # ---- the target ----------------------------------------------------------
    target_from_end: Tuple[int, ...] = (1, 2)       # 1 = last pair, 2 = second-to-last
    delta_direction: str = "backward"               # B comes EARLY; see the module docstring

    # ---- the tracked variable ------------------------------------------------
    delta_start_ms: float = 10.0
    delta_min_ms: float = 0.25
    delta_max_ms: float = 70.0
    # For token conditions. An early B opens a gap inside its token once delta exceeds
    # lag + tone_ms; at lag 0 that is 75 ms, so 70 keeps every token gap-free.
    delta_max_b_alone_ms: float = 120.0
    # B alone has no A and so no gap to open; its only limit is two B tones running into each
    # other. It needs the room: its threshold sits far above the token conditions by design,
    # and a ceiling near it censors the control (3 of 4 simulated tracks at a 100 ms ceiling).

    # ---- the adaptive rule (names match tcoh.track.Track) --------------------
    n_down: int = 3
    step_factors: Tuple[float, ...] = (4.0, 2.0, math.sqrt(2.0))
    reversals_per_step: Tuple[int, ...] = (1, 2)
    n_final_reversals: int = 6
    max_trials_per_track: int = 90
    tracks_per_condition: int = 2

    # ---- catch trials --------------------------------------------------------
    catch_rate: float = 0.07
    catch_delta_ms: float = 60.0
    catch_at_lag_ms: Optional[float] = 0.0
    max_catch_miss_rate: float = 0.15

    # ---- procedure -----------------------------------------------------------
    feedback: bool = True
    practice_trials: int = 12
    practice_criterion: float = 0.75
    practice_delta_ms: float = 60.0
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
        for name in ("lags_ms", "target_from_end", "step_factors", "reversals_per_step"):
            if kw.get(name) is not None:
                kw[name] = tuple(kw[name])
        return cls(**kw)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]

    @property
    def slot_ms(self) -> float:
        return 1000.0 / self.rate_hz

    @property
    def target_pairs(self) -> Tuple[int, ...]:
        """0-based indices of the pairs that may carry the displacement."""
        return tuple(self.n_pairs - k for k in self.target_from_end)

    def nominal_onsets(self) -> np.ndarray:
        return self.lead_ms + np.arange(self.n_pairs) * self.slot_ms

    def track_cfg(self, kind: str) -> "Config":
        """The config a track of this kind runs under: only the ceiling differs."""
        return self.replace(delta_max_ms=self.delta_max_b_alone_ms) if kind == "b_alone" else self

    @property
    def ceiling_max_ms(self) -> float:
        return max(self.delta_max_ms, self.delta_max_b_alone_ms)

    def lag_pct(self, lag_ms: float) -> float:
        return 100.0 * lag_ms / self.tone_ms

    def trial_ms(self) -> float:
        return 2 * self.interval_ms + self.isi_ms


DEFAULT = Config()


def conditions(cfg: Config) -> Tuple[Condition, ...]:
    out = [Condition(f"lag_{_tag(L)}", L, "pair", "main") for L in cfg.lags_ms]
    if cfg.include_b_alone:
        out.append(Condition("b_alone", 0.0, "b_alone", "control"))
    return tuple(out)


def _tag(v: float) -> str:
    return f"{v:g}".replace(".", "p")


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


def extremes_ms(cfg: Config) -> dict:
    """The earliest any sound can start and the latest any can end, over every condition."""
    J = cfg.pair_jitter_ms
    first = cfg.lead_ms - J - (cfg.ceiling_max_ms if 0 in cfg.target_pairs else 0.0)
    last = cfg.nominal_onsets()[-1] + J + max(cfg.lags_ms or (0.0,)) + cfg.tone_ms
    return {"earliest_onset": first, "latest_offset": last}


def same_channel_clearance_ms(cfg: Config) -> float:
    """Smallest silent gap between two tones of the SAME channel that any trial can produce.

    Worst case: the target B comes delta_max early, its own pair was jittered early and the
    pair before late. Two tones of one frequency closer than this would sum.
    """
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


def validate(cfg: Config) -> Derived:
    from .model import lag_index
    notes: List[str] = []
    if cfg.n_pairs < 4:
        raise ConfigError("n_pairs must be at least 4: the listener needs pairs before the target")
    if any(not (1 <= k <= cfg.n_pairs - 2) for k in cfg.target_from_end):
        raise ConfigError("target_from_end must leave at least two pairs before the target")
    if cfg.delta_direction not in ("backward", "forward", "random"):
        raise ConfigError("delta_direction must be backward, forward or random")
    if cfg.delta_direction != "backward":
        notes.append("late shifts open a silent gap inside the token at 100%; the design moves B "
                     "early only")
    if cfg.a_leads and cfg.delta_direction == "backward":
        room = min(cfg.lags_ms or (0.0,)) + cfg.tone_ms
        if cfg.delta_max_ms >= room:
            raise ConfigError(
                f"delta_max_ms={cfg.delta_max_ms:g} lets an early B finish before its own A starts "
                f"at lag {min(cfg.lags_ms):g} ms, opening a gap inside the token (limit {room:g}).")
    if any(L > cfg.tone_ms + 1e-9 for L in cfg.lags_ms):
        raise ConfigError("a lag beyond tone_ms leaves a gap inside the token")
    if cfg.delta_min_ms <= 0 or cfg.delta_max_ms <= cfg.delta_min_ms:
        raise ConfigError("need 0 < delta_min_ms < delta_max_ms")
    if len(cfg.step_factors) != len(cfg.reversals_per_step) + 1:
        raise ConfigError("reversals_per_step must have one entry fewer than step_factors")

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

    ext = extremes_ms(cfg)
    if ext["earliest_onset"] < 5.0:
        raise ConfigError(f"a tone can start at {ext['earliest_onset']:.1f} ms; raise lead_ms")
    if ext["latest_offset"] > cfg.interval_ms - 5.0:
        raise ConfigError(f"a tone can end at {ext['latest_offset']:.1f} ms, past the "
                          f"{cfg.interval_ms:g} ms interval")
    gap = same_channel_clearance_ms(cfg)
    if gap < FUSION_MARGIN_MS:
        raise ConfigError(
            f"in the worst case two tones of the same frequency come within {gap:.1f} ms of each "
            f"other (limit {FUSION_MARGIN_MS:g}). Lower delta_max_ms or pair_jitter_ms.")
    if cfg.delta_start_ms * cfg.step_factors[0] > cfg.delta_max_ms:
        notes.append("the first error would call for more than delta_max_ms; the track sits at "
                     "the ceiling until it gets three right")
    peak = 2 * cfg.tone_amplitude * 10 ** (cfg.level_rove_db / 20.0)
    if peak >= 1.0:
        raise ConfigError(f"two overlapping tones at the maximum rove peak at {peak:.2f} FS")
    if not cfg.include_b_alone:
        notes.append("include_b_alone is off, so nothing shows that the listener used A at all")
    passing = [L for L in cfg.lags_ms if 0 < L < cfg.delta_max_ms]
    if passing:
        notes.append(
            f"at lag {', '.join(f'{L:g}' for L in passing)} ms a large enough early shift carries B "
            "to A's onset and past it. Recorded on every trial as passes_a_onset.")

    conds = conditions(cfg)
    model = {c.name: (float("nan") if c.kind == "b_alone" else lag_index(cfg, c.lag_ms))
             for c in conds}
    return Derived(conds, len(conds) * cfg.tracks_per_condition, ra, ext, gap, model,
                   tuple(notes))


def describe(cfg: Config) -> str:
    d = validate(cfg)
    ra = d.ratio
    L = [f"jittered A-B pairs  |  config {cfg.hash()}",
         f"  tones      A {cfg.f_a_hz:.0f} Hz, B {cfg.f_b_hz:.0f} Hz: {ra['semitones']:.1f} st "
         f"({ra['octaves']:.2f} oct, {ra['erbs']:.1f} ERB), {ra['mistuning_pct']:.1f}% from "
         f"{ra['nearest']}, the nearest ratio with terms up to {FUSION_TERM_MAX}",
         f"             {cfg.tone_ms:g} ms tones, {cfg.ramp_ms:g} ms ramps; "
         f"{'A first, B trails by the lag' if cfg.a_leads else 'B first, A trails by the lag'}",
         f"  sequence   {cfg.n_pairs} pairs at {cfg.rate_hz:g} Hz on average, each moved as a "
         f"unit by U(+-{cfg.pair_jitter_ms:g} ms), drawn afresh for each interval",
         f"  interval   {cfg.interval_ms:g} ms; trial {cfg.interval_ms:g} + {cfg.isi_ms:g} + "
         f"{cfg.interval_ms:g} ms = {cfg.trial_ms() / 1000:.1f} s",
         f"  target     the B tone of pair "
         + " or ".join(f"{'N' if k == 1 else f'N-{k - 1}'}" for k in cfg.target_from_end)
         + f" comes delta ms {'early' if cfg.delta_direction == 'backward' else cfg.delta_direction}",
         f"  delta      starts at {cfg.delta_start_ms:g} ms, capped at {cfg.delta_max_ms:g} ms "
         f"(B alone {cfg.delta_max_b_alone_ms:g}); same-pitch tones at least "
         f"{d.same_channel_gap_ms:.0f} ms apart",
         "",
         f"  {'condition':<10} {'dT':>6} {'lag ms':>7} {'model l2/l1':>12}  role"]
    for c in d.conditions:
        mv = d.model_index[c.name]
        L.append(f"  {c.name:<10} {('-' if c.kind == 'b_alone' else f'{cfg.lag_pct(c.lag_ms):g}%'):>6} "
                 f"{('-' if c.kind == 'b_alone' else f'{c.lag_ms:g}'):>7} "
                 f"{('-' if mv != mv else f'{mv:.3f}'):>12}  {c.role}"
                 + ("  (what A adds, measured)" if c.kind == "b_alone" else ""))
    for n in d.notes:
        L.append(f"\n  note: {n}")
    return "\n".join(L)
