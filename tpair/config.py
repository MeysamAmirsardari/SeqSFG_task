"""Parameters and geometry for the jittered-pair task.

The stimulus
-------------
Six pairs of pure tones at an average of 3 Hz fill a 2-second interval. In every pair the low
tone A starts `lag` ms after the high tone B, with `lag` between 0 (synchronous) and 100 ms
(the tones abut). Each pair is jittered AS A WHOLE by an independent uniform draw of
+-`pair_jitter_ms`: A and B move together, so the lag inside every pair is exact while neither
channel on its own is regular. In one of the two intervals of a trial, the B tone of the last
or the second-to-last pair comes `delta` ms early, and the listener says which interval.

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
B moves, as in the two earlier tasks and in Elhilali et al. (2009), and it moves EARLY, so the
lag grows from L to L + delta. Moving it late would shrink the lag, and at delta = L the target
B would land exactly on its own A, producing a synchronous chord the standard never contains --
a categorical cue rather than a larger version of the same one. That is the trap the two-tone
task fell into before its pilot was restricted to early shifts.

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


CORE_LAGS: Tuple[float, ...] = (0.0, 10.0, 25.0, 50.0, 75.0, 100.0)
# Close spacing near synchrony, where every behavioural result in this repository has changed
# fastest (the sheared figure lost 79% of its synchrony advantage within 12.5 ms), and roughly
# even spacing in the model index above it: 0.000, 0.047, 0.173, 0.399, 0.635, 0.804.

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
    tone_ms: float = 100.0
    ramp_ms: float = 10.0

    # ---- the sequence --------------------------------------------------------
    rate_hz: float = 3.0            # average pair rate
    n_pairs: int = 6
    pair_jitter_ms: float = 40.0    # each pair moves as a unit by U(-J, +J)
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
    delta_start_ms: float = 20.0
    delta_min_ms: float = 0.25
    delta_max_ms: float = 120.0
    # High enough to MEASURE B-alone, whose threshold the jitter puts near 46-60 ms by design.
    # At 100 ms the control's upward reversals kept landing on the clamp, and a reversal at a
    # clamp is not a reversal: 3 of 4 simulated B-alone tracks came back censored. The geometry
    # allows up to 128 ms (same-frequency tones then still 25 ms apart in the worst case); 120
    # keeps 33.

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
    first = cfg.lead_ms - J - (cfg.delta_max_ms if 0 in cfg.target_pairs else 0.0)
    last = cfg.nominal_onsets()[-1] + J + max(cfg.lags_ms or (0.0,)) + cfg.tone_ms
    return {"earliest_onset": first, "latest_offset": last}


def same_channel_clearance_ms(cfg: Config) -> float:
    """Smallest silent gap between two tones of the SAME channel that any trial can produce.

    Worst case: the target B comes delta_max early, its own pair was jittered early and the
    pair before late. Two tones of one frequency closer than this would sum.
    """
    J = cfg.pair_jitter_ms
    return cfg.slot_ms - 2 * J - cfg.delta_max_ms - cfg.tone_ms


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
    if any(L < 0 or L > cfg.tone_ms + 1e-9 for L in cfg.lags_ms):
        notes.append("a lag beyond tone_ms leaves a silent gap inside the pair, which the design "
                     "does not otherwise contain")
    if cfg.delta_direction not in ("backward", "forward", "random"):
        raise ConfigError("delta_direction must be backward, forward or random")
    if cfg.delta_direction != "backward":
        notes.append(
            "late shifts shrink the lag, and at delta = lag the target B lands exactly on its own A: "
            "a synchronous chord the standard never contains, i.e. a different cue rather than a "
            "bigger one. The design's default is early shifts only.")
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
    notes.append(
        f"at lag {max(cfg.lags_ms):g} ms with a large delta and an unlucky jitter draw, the early "
        "B can sit nearer the PREVIOUS pair's A than its own. That is inherent to moving a tone "
        "away from its partner; every trial records whether it happened.")

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
         f"             {cfg.tone_ms:g} ms tones, {cfg.ramp_ms:g} ms ramps",
         f"  sequence   {cfg.n_pairs} pairs at {cfg.rate_hz:g} Hz on average, each moved as a "
         f"unit by U(+-{cfg.pair_jitter_ms:g} ms), drawn afresh for each interval",
         f"  interval   {cfg.interval_ms:g} ms; trial {cfg.interval_ms:g} + {cfg.isi_ms:g} + "
         f"{cfg.interval_ms:g} ms = {cfg.trial_ms() / 1000:.1f} s",
         f"  target     the B tone of pair "
         + " or ".join(f"{'N' if k == 1 else f'N-{k - 1}'}" for k in cfg.target_from_end)
         + f" comes delta ms {'early' if cfg.delta_direction == 'backward' else cfg.delta_direction}",
         f"  delta      starts at {cfg.delta_start_ms:g} ms, capped at {cfg.delta_max_ms:g} ms; "
         f"closest same-frequency tones in the worst case {d.same_channel_gap_ms:.0f} ms apart",
         "",
         f"  {'condition':<10} {'lag ms':>7} {'model l2/l1':>12}  role"]
    for c in d.conditions:
        mv = d.model_index[c.name]
        L.append(f"  {c.name:<10} {('-' if c.kind == 'b_alone' else f'{c.lag_ms:g}'):>7} "
                 f"{('-' if mv != mv else f'{mv:.3f}'):>12}  {c.role}"
                 + ("  (what A adds, measured)" if c.kind == "b_alone" else ""))
    for n in d.notes:
        L.append(f"\n  note: {n}")
    return "\n".join(L)
