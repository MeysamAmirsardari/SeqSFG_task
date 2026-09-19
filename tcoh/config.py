"""Every parameter of the two-tone coherence experiment, and the validator that refuses
configurations the design cannot support.

Nothing numerical is chosen anywhere else in this package. Modules ask for ``cfg.<name>`` or
for a field of ``Derived``, which is computed FROM the config and never decided independently.
That is the same discipline the seqsfg package uses, for the same reason: a parameter that can
be set in two places will eventually be set differently in two places.

The stimulus
------------
Two pure tones, A (low) and B (high), each an isochronous sequence of ``n_precursor + 1``
tones. B's sequence is the reference grid; A's is the same grid displaced by a lag. The lag is
expressed as ``dT%``, a percentage of HALF the period, so 0% is exact synchrony and 100% is
exact alternation -- the axis of Figure 8B in Elhilali et al. (2009).

    dT = 0%     A  A  A  A  A  A         100%     A  A  A  A  A  A
                B  B  B  B  B  B                   B  B  B  B  B  B

On every trial the listener hears two such sequences separated by a silent gap. They are
identical except that in one of them -- chosen at random -- the LAST B tone is displaced by
+-delta. The listener says which one. delta is what the adaptive track measures.

Why tone_ms must be exactly half of soa_ms
------------------------------------------
Because otherwise dT is not an ordered axis. With a silent gap inside each channel the model's
own segregation index is not monotone in dT: it peaks near 75%, where the two channels
interdigitate most thoroughly, and falls again at full alternation (``model.duty_cycle_scan``).
A monotone behavioural result would then confirm nothing. ``validate`` refuses any other duty
cycle unless ``allow_nonmonotone_duty`` is set, which exists so the planned follow-up -- where
breaking the confound between onset lag and acoustic overlap is the point -- can be built
without editing this file.

What is deliberately held constant, and why it matters
-------------------------------------------------------
The B sequence is bit-identical in every condition: same frequency, same onsets, same level,
same ramps. The difference between the standard and target interval is a displacement of one
B tone and nothing else, and that displacement is the same physical operation at every dT.
So the cue available from the B channel alone -- "is the B rhythm regular?" -- cannot vary
with dT, and any dependence of performance on dT has to come from the A channel. That
invariant is the spine of the whole argument; ``tcoh.verify`` checks it numerically rather
than trusting this paragraph.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from functools import lru_cache as _lru_cache

A_KINDS = ("coherent", "scrambled", "pair_only", "absent", "nopartner")
# coherent  : the A sequence is isochronous at lag dT. The condition of interest.
# scrambled : the A PRECURSORS are placed at random times across the span they would have
#             occupied, subject to never overlapping each other, while the final A tone stays
#             exactly where the coherent condition puts it. Same tones, same count, same level,
#             same final A-B interval, no isochronous A sequence to stream. Control 1.
#             (An earlier version jittered the onsets around the grid by a bounded amount. The
#             model said no: a bounded jitter leaves the channels substantially co-modulated at
#             dT = 0 and its own coherence then TRACKS dT, which makes it useless as a
#             comparison line. Free placement is measurably more decoherent -- see
#             verify.model_checks -- and adds no new regularity of its own.)
# pair_only : the A PRECURSORS are removed and only the final A tone remains, exactly where the
#             coherent condition puts it. Control 2, and the cleaner logic of the two: with a
#             single A tone there is no A sequence to stream, by construction and not by
#             degree, while the interval the listener judges is identical. Its cost is that it
#             holds five fewer tones, so energy and masking are not matched -- but that
#             mismatch is the same at every dT and cannot produce an interaction with dT.
#             The two controls fail in different directions, which is why both are worth running.
# absent    : no A tones at all. The ceiling: performance from the B rhythm alone. This is
#             Elhilali's third control (her crosses in Figure 2).
# nopartner : A precursors present and coherent, final A tone removed. Separates what the A
#             CONTEXT contributes from what the simultaneous final A PARTNER contributes.

REFERENCES = ("yoked", "sync", "tempo")
# yoked : both sequences isochronous at the same rate, A lagging B by dT throughout, so the
#         final pair also sits at dT. This is exactly the stimulus Figure 8 simulates, which
#         is why it is the primary. Its cost is that the interval the listener judges has a
#         pedestal that grows with dT, so interval discrimination alone could produce a rising
#         threshold; that is what the `jittered` control is for.
# sync  : the final A tone is moved onto the unshifted final B tone, so the judged interval is
#         always "synchronous versus delta" and no pedestal grows. Its cost is a rhythmic
#         anomaly in the A stream that grows with dT, and at dT = 100% the last two A tones
#         become contiguous and fuse into one long tone -- validate() refuses that case.
# tempo : Elhilali's own manipulation. A and B are isochronous at DIFFERENT rates and the whole
#         A sequence is phase-shifted so the final pair is synchronous. No pedestal, no
#         anomaly, but the precursor lag ramps instead of holding at one value, so dT is not
#         single-valued. Kept for direct replication of her Figure 2, not for the sweep.


class ConfigError(ValueError):
    """Raised by :func:`validate` with a message that says what to change."""


@dataclass(frozen=True)
class Condition:
    """One adaptive track's worth of stimulus: what A does, and where it sits relative to B."""
    name: str
    lag_pct: float                       # dT%, 0 = synchronous, 100 = alternating
    a_kind: str = "coherent"
    reference: str = "yoked"
    n_precursor: Optional[int] = None    # None -> cfg.n_precursor. Set to vary build-up.
    tempo_gap_ms: Optional[float] = None # reference='tempo' only: A's silent gap (Elhilali: 30/50/70)
    interleaved: bool = False            # complex tones only: see Config.tone_freqs
    role: str = "main"                   # main | control | anchor -- reporting only

    def label(self) -> str:
        bits = [f"dT={self.lag_pct:g}%"]
        if self.a_kind != "coherent":
            bits.append(self.a_kind)
        if self.interleaved:
            bits.append("interleaved")
        if self.reference != "yoked":
            bits.append(self.reference)
        if self.n_precursor is not None:
            bits.append(f"{self.n_precursor}pre")
        if self.tempo_gap_ms is not None:
            bits.append(f"gap{self.tempo_gap_ms:g}")
        return " ".join(bits)


CORE_PCTS: Tuple[float, ...] = (0.0, 25.0, 50.0, 75.0, 100.0)
FULL_PCTS: Tuple[float, ...] = (0.0, 12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5, 100.0)


@dataclass(frozen=True)
class Config:
    # ---- audio ---------------------------------------------------------------
    sample_rate: int = 48000
    monaural: bool = False               # True reproduces Elhilali's left-earpiece-only presentation
    tone_amplitude: float = 0.05         # linear peak amplitude of ONE tone
    tone_level_db_spl: float = 65.0      # what that amplitude should measure at the ear
    level_rove_db: float = 3.0           # +-, drawn independently for each interval of a trial

    # ---- the two tones -------------------------------------------------------
    f_a_hz: float = 1000.0               # the LOW tone; Elhilali's A
    df_semitones: float = 15.0           # B is this far above A. Her largest separation (1.25 oct)
    partials_hz: Tuple[float, ...] = ()  # empty -> A and B are pure tones at f_a_hz / f_b_hz
    # Non-empty makes each tone a complex of several partials: an even number of frequencies,
    # split between A and B by `tone_freqs`. It changes nothing about what the model predicts
    # -- partials within a tone are perfectly coherent with each other, so the coherence matrix
    # has the same leading two eigenvalues as the pure-tone case, verified to three decimals --
    # and everything about what the listener can do with a single auditory filter. The point is
    # to remove the one-channel solution and, with an inharmonic set, harmonic fusion too.
    tone_ms: float = 75.0                # must be soa_ms / 2; see the module docstring
    ramp_ms: float = 10.0                # raised-cosine onset and offset
    soa_ms: float = 150.0                # onset-to-onset within one channel

    # ---- the sequence --------------------------------------------------------
    n_precursor: int = 5                 # tones per channel BEFORE the target pair (Elhilali: 5)
    lead_ms: float = 250.0               # silence before the first tone
    tail_ms: float = 250.0               # silence after the last tone ends, beyond any shift
    isi_ms: float = 500.0                # silence between the two intervals of a trial (Elhilali)
    scramble_min_gap_ms: Optional[float] = None
    # 'scrambled' A precursors: random onsets with at least this much between them. None ->
    # tone_ms, which is the smallest separation for which two A tones can never overlap, so the
    # control can never contain a summed pair of tones the coherent condition does not have.

    # ---- the tracked variable ------------------------------------------------
    delta_start_ms: float = 20.0         # Elhilali's starting value
    delta_min_ms: float = 0.25           # below one sample period at 48 kHz there is nothing to render
    delta_max_ms: float = 45.0           # ceiling; a track resting here is reported, not hidden
    delta_direction: str = "random"      # random | forward | backward  (forward = the B tone is late)

    # ---- the adaptive rule ---------------------------------------------------
    n_down: int = 3                      # 3-down 1-up tracks the 79.4% point
    step_factors: Tuple[float, ...] = (4.0, 2.0, math.sqrt(2.0))
    reversals_per_step: Tuple[int, ...] = (1, 2)   # change step after the 1st, then 2 more reversals
    n_final_reversals: int = 6           # threshold = geometric mean of this many, at the final step
    max_trials_per_track: int = 90       # a track that cannot converge is stopped and flagged
    tracks_per_condition: int = 3

    # ---- conditions ----------------------------------------------------------
    sweep_pcts: Tuple[float, ...] = CORE_PCTS
    interleaved_pcts: Tuple[float, ...] = ()
    # dT levels to repeat with the partials interleaved instead of separated. Needs
    # `partials_hz`. Same tones, same spectrum, same level; only which partial belongs to which
    # tone changes. If the curve is the same in both, frequency region was not what grouped
    # the partials -- and if it is not, the separated curve owed something to spectral
    # proximity that the coherence account does not claim.
    scrambled_pcts: Tuple[float, ...] = CORE_PCTS
    # Matched at EVERY dT of the sweep, not at a subset. Measured power for the test that
    # actually separates the accounts (H2) is 97% with five matched levels and 70% with three,
    # for a session that is 20% longer -- by far the best return on the listener's time this
    # design has. `tcoh.analysis.power` is where those numbers come from.
    pair_only_pcts: Tuple[float, ...] = ()
    nopartner_pcts: Tuple[float, ...] = ()
    include_b_only: bool = True
    buildup_pcts: Tuple[float, ...] = ()     # dT levels to repeat with buildup_n_precursor
    buildup_n_precursor: int = 13
    include_tempo_replication: bool = False  # Elhilali's 30 / 50 / 70 ms controls, reference='tempo'
    allow_nonmonotone_duty: bool = False

    # ---- catch trials --------------------------------------------------------
    catch_rate: float = 0.06             # fraction of main trials replaced by a supra-threshold one
    catch_delta_ms: float = 45.0         # the shift used on those; must be detectable by anyone
    catch_at_pct: Optional[float] = None
    # Which condition a catch trial is built from. None means "whichever track hosts the slot",
    # which is what the first design did and what the first real session showed to be wrong: a
    # fixed 50 ms probe is eight times threshold at dT = 0% and barely above it at dT = 100%, so
    # its difficulty tracks the condition and a miss says nothing about attention. Set it to a
    # lag_pct -- 0.0 is the obvious choice -- and every catch trial is the same easy stimulus
    # wherever it lands, which is the only way a lapse probe measures lapses. The listener
    # cannot pick them out, because trials at that lag occur normally anyway.
    max_catch_miss_rate: float = 0.15    # above this the session is flagged as inattentive

    # ---- procedure -----------------------------------------------------------
    feedback: bool = True                # 2AFC: the measure is criterion-free, so feedback only helps
    practice_trials: int = 12
    practice_criterion: float = 0.75     # proportion correct on easy practice before the main block
    practice_delta_ms: float = 45.0
    familiarise: bool = True
    response_timeout_s: float = 6.0
    break_every: int = 8                 # tracks between offered breaks
    max_same_condition_run: int = 2      # consecutive trials from one track when interleaving
    tracks_per_block: Optional[int] = None
    # How many tracks are interleaved at once. None picks it: the block size has to divide the
    # number of conditions, or some condition sits in the short block every round and ends up
    # systematically late. With five conditions in blocks of three the serial-position spread
    # was 0.185 of the session; interleaving all five gives 0.

    # ---- streams block (secondary, subjective) -------------------------------
    streams_block: bool = False          # "one sound or two?" at each dT, no timing manipulation
    streams_trials_per_pct: int = 10
    streams_duration_ms: float = 3000.0

    # ---- analysis ------------------------------------------------------------
    n_boot: int = 4000
    lapse_max: float = 0.1               # upper bound on the fitted lapse rate

    # ---------------------------------------------------------------------------
    def replace(self, **kw) -> "Config":
        return dataclasses.replace(self, **kw)

    def to_dict(self) -> dict:
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = list(v) if isinstance(v, tuple) else v
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ConfigError(f"unknown parameter(s): {sorted(unknown)}")
        kw = dict(d)
        for name in ("step_factors", "reversals_per_step", "sweep_pcts", "scrambled_pcts",
                     "pair_only_pcts", "nopartner_pcts", "buildup_pcts", "partials_hz",
                     "interleaved_pcts"):
            if name in kw and kw[name] is not None:
                kw[name] = tuple(kw[name])
        return cls(**kw)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]

    # -- quantities every module needs and nobody should recompute --------------
    @property
    def f_b_hz(self) -> float:
        return self.f_a_hz * 2.0 ** (self.df_semitones / 12.0)

    @property
    def is_complex(self) -> bool:
        return bool(self.partials_hz)

    def tone_freqs(self, interleaved: bool = False) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """The frequencies of A and B, as (a_partials, b_partials).

        With `partials_hz` empty this is the pure-tone pair and `interleaved` is meaningless.
        Otherwise the sorted set is split in half: SEPARATED gives A the lower half and B the
        upper half, which is the direct analogue of the pure-tone design and keeps the two
        curves comparable; INTERLEAVED alternates them, so no frequency boundary separates A
        from B and common onset is the only thing left that can group a tone's partials
        together. Same partials, same long-term spectrum, same level -- only the assignment
        changes, which is what makes the pair of conditions a test of spectral proximity.
        """
        if not self.partials_hz:
            return (self.f_a_hz,), (self.f_b_hz,)
        p = tuple(sorted(float(f) for f in self.partials_hz))
        if interleaved:
            return p[0::2], p[1::2]
        h = len(p) // 2
        return p[:h], p[h:]

    @property
    def partial_amplitude_scale(self) -> float:
        """Per-partial amplitude divisor that holds the POWER of one tone constant.

        `tone_amplitude` stays the amplitude of a whole tone whatever it is made of, so a
        level calibration measured with pure tones carries over to the complex version
        unchanged and the two can be compared on absolute level.
        """
        n = max(len(self.partials_hz) // 2, 1)
        return math.sqrt(float(n))

    @property
    def n_tones(self) -> int:
        return self.n_precursor + 1

    @property
    def duty(self) -> float:
        return self.tone_ms / self.soa_ms

    @property
    def scramble_min_gap(self) -> float:
        return self.scramble_min_gap_ms if self.scramble_min_gap_ms is not None else self.tone_ms

    def lag_ms(self, pct: float) -> float:
        """dT% as milliseconds. 100% is half the period, i.e. exact alternation."""
        return float(pct) / 100.0 * self.soa_ms / 2.0

    def pct(self, lag_ms: float) -> float:
        return 200.0 * float(lag_ms) / self.soa_ms


@dataclass(frozen=True)
class Derived:
    f_b_hz: float
    lag_ms_by_pct: Dict[float, float]
    interval_ms: float            # for a sequence of cfg.n_precursor; see config.interval_ms
    longest_interval_ms: float
    trial_ms: float
    mean_trial_ms: float
    conditions: Tuple[Condition, ...]
    n_tracks: int
    matched_control_levels: int
    est_trials: int
    est_minutes: float
    collision_headroom_ms: float
    erbs_apart: float
    roex_attenuation_db: float
    roex_leak_is_floor: bool
    harmonic: dict
    model_curve: Dict[float, float]            # by dT%, for the coherent sequence
    model_by_condition: Dict[str, float]       # by condition name, controls included
    notes: Tuple[str, ...]


# ----------------------------------------------------------------------------
# the condition set
# ----------------------------------------------------------------------------
def conditions(cfg: Config) -> Tuple[Condition, ...]:
    """The tracks this configuration will run, in a fixed order that does not depend on a seed.

    The set is built here rather than in the runner so that `tcoh-design` can print it, the
    verification battery can audit every member of it, and the session file can record it,
    all from the same function.

    The three controls earn their place as follows.

    `absent` (B only) is the ceiling. Because the B channel is bit-identical in every
    condition, this one number is the performance available without any A tone at all, and
    every other condition has to be read against it.

    `jittered` is the one that separates the hypothesis from its most serious rival. In the
    yoked reference the interval the listener judges sits on a pedestal that grows with dT, and
    discriminating a change in a longer interval is harder for reasons that have nothing to do
    with streaming. The jittered control holds that pedestal -- the final A tone is in exactly
    the position the coherent condition puts it -- along with the number, level, duration and
    frequency of every A tone, and destroys only the temporal coherence of the A sequence. A
    pedestal account predicts the two are equal at every dT. A coherence account predicts they
    differ most at dT = 0 and converge at dT = 100%. The shapes are different, so the data can
    choose.

    `nopartner` asks a further question the paper's own data raise: Elhilali found that a
    physically present, synchronous final A tone bought the listener nothing when the
    precursors placed it in another stream. Removing that tone entirely should then cost
    nothing either. It is off by default because it doubles as a manipulation check rather
    than a test, and session time is the scarce resource.
    """
    out: List[Condition] = []
    for p in cfg.sweep_pcts:
        out.append(Condition(f"coh_{_tag(p)}", p, "coherent", "yoked", role="main"))
    for p in cfg.scrambled_pcts:
        out.append(Condition(f"scr_{_tag(p)}", p, "scrambled", "yoked", role="control"))
    for p in cfg.pair_only_pcts:
        out.append(Condition(f"par_{_tag(p)}", p, "pair_only", "yoked", role="control"))
    for p in cfg.nopartner_pcts:
        out.append(Condition(f"nop_{_tag(p)}", p, "nopartner", "yoked", role="control"))
    for p in cfg.interleaved_pcts:
        out.append(Condition(f"int_{_tag(p)}", p, "coherent", "yoked",
                             interleaved=True, role="control"))
    for p in cfg.buildup_pcts:
        out.append(Condition(f"bld_{_tag(p)}", p, "coherent", "yoked",
                             n_precursor=cfg.buildup_n_precursor, role="control"))
    if cfg.include_b_only:
        out.append(Condition("b_only", 0.0, "absent", "yoked", role="control"))
        if cfg.buildup_pcts:
            # A build-up condition has a LONGER B sequence, not just a longer A sequence, and a
            # longer B rhythm is easier to judge whether or not anything streams -- a confound
            # running in the same direction as the prediction. So the long precursor gets its own
            # ceiling, and H3 is scored as an interaction: does lengthening the precursor help the
            # coherent condition MORE than it helps the listener with no low tone at all?
            out.append(Condition("b_only_long", 0.0, "absent", "yoked",
                                 n_precursor=cfg.buildup_n_precursor, role="control"))
    if cfg.include_tempo_replication:
        for gap in (30.0, 50.0, 70.0):
            out.append(Condition(f"tempo_{gap:g}", 0.0, "coherent", "tempo",
                                 tempo_gap_ms=gap, role="control"))
    return tuple(out)


SIMPLE_RATIO_MAX_ORDER = 5
# p:q counts as a 'simple' ratio when p + q <= this: 1:1, 2:1, 3:1, 3:2, 4:1. Those are the
# ones two simultaneous components actually fuse or beat at. Searching further out finds
# ratios like 19:8 that are arithmetically close to everything and perceptually close to
# nothing, which is why `harmonic_proximity` caps its own search too.

COMMON_F0_RANGE_HZ = (120.0, 600.0)
COMMON_F0_MAX_HARMONIC = 12
COMMON_F0_TOLERANCE = 0.04
# For a complex of several partials the fusion cue that matters is a shared fundamental, not
# any one pairwise ratio: that is what makes a set of components one harmonic object instead
# of several. A set is inharmonic when no f0 in this range explains every partial as a
# low-numbered harmonic to within this tolerance.


def _simple_ratio_mistuning(x: float, y: float) -> Tuple[float, str]:
    """Distance from the nearest genuinely simple frequency ratio, as a percentage."""
    r = max(x, y) / min(x, y)
    best = (float("inf"), "")
    for q in range(1, SIMPLE_RATIO_MAX_ORDER):
        for p in range(q, SIMPLE_RATIO_MAX_ORDER + 1):
            if math.gcd(p, q) != 1 or p + q > SIMPLE_RATIO_MAX_ORDER:
                continue
            err = abs(r - p / q) / (p / q)
            if err < best[0]:
                best = (err, f"{p}:{q}")
    return 100.0 * best[0], best[1]


def _common_f0(freqs: Sequence[float]) -> Optional[dict]:
    """The best low-order harmonic template for this set, if one fits within tolerance.

    Meaningless below three components: ANY two frequencies are some n:m, so a pair always
    "fits" a fundamental and the test would fire on every set. What makes a pair fuse is a
    LOW-order ratio, and `_simple_ratio_mistuning` is the test for that. Returns None rather
    than a false positive.
    """
    if len(freqs) < 3:
        return None
    lo, hi = COMMON_F0_RANGE_HZ
    best = None
    f0 = lo
    while f0 <= hi:
        ns = [round(f / f0) for f in freqs]
        if all(1 <= n <= COMMON_F0_MAX_HARMONIC for n in ns) and len(set(ns)) == len(ns):
            err = max(abs(f - n * f0) / (n * f0) for f, n in zip(freqs, ns))
            if best is None or err < best["error"]:
                best = {"f0_hz": float(f0), "harmonics": list(ns), "error": float(err)}
        f0 += 0.5
    if best is None or best["error"] >= COMMON_F0_TOLERANCE:
        return None
    return best


def partial_audit(cfg: Config, interleaved: bool = False) -> dict:
    """Is this partial set resolved, inharmonic, and free of a common fundamental?

    Three separate things, easy to conflate:

    * RESOLVED -- every pair at least 3 ERBs apart, so each partial owns an auditory filter and
      the model's one-channel-per-component reduction still applies.
    * NOT A SIMPLE RATIO -- no pair close to 1:1, 2:1, 3:1, 3:2 or 4:1, which fuse or beat.
    * NO COMMON FUNDAMENTAL -- the set as a whole is not a low-numbered harmonic series, which
      is the cue that actually welds several components into one perceived object and the one
      thing a pairwise test cannot see.

    Reported for one arrangement at a time, because interleaving changes which partials are
    simultaneous within a tone but not which pairs exist.
    """
    from .model import channel_crosstalk
    a_f, b_f = cfg.tone_freqs(interleaved)
    allf = sorted(a_f + b_f)
    worst_erb, worst_erb_pair = float("inf"), None
    worst_mis, worst_mis_pair, worst_mis_ratio = float("inf"), None, ""
    for i, x in enumerate(allf):
        for y in allf[i + 1:]:
            e = channel_crosstalk(x, y)["erbs"]
            if e < worst_erb:
                worst_erb, worst_erb_pair = e, (x, y)
            m, lbl = _simple_ratio_mistuning(x, y)
            if m < worst_mis:
                worst_mis, worst_mis_pair, worst_mis_ratio = m, (x, y), lbl
    return {"arrangement": "interleaved" if interleaved else "separated",
            "a_hz": list(a_f), "b_hz": list(b_f), "n_partials_per_tone": len(a_f),
            "min_erbs": float(worst_erb), "min_erb_pair": worst_erb_pair,
            "min_simple_ratio_mistuning_pct": float(worst_mis),
            "min_simple_ratio_pair": worst_mis_pair, "min_simple_ratio": worst_mis_ratio,
            "common_f0": _common_f0(allf),
            "within_a_f0": _common_f0(list(a_f)), "within_b_f0": _common_f0(list(b_f))}


FUSION_MARGIN_MS = 25.0
# How close two tones may come before they stop being two events. Onset-asynchrony DETECTION
# runs to a few milliseconds, but two tones still fuse into one perceived event well beyond
# that; 25 ms is a deliberately conservative reading of that literature and is used only to
# bound delta, never to model anything.


def displaced_tone_clearance(cfg: Config, lag_pct: float, delta_ms: float,
                             direction: int) -> dict:
    """How close the displaced final B tone comes to anything else, and whether that is new.

    Two separate hazards, and only the second is subtle.

    The obvious one: shifted early, the tone walks back towards the PRECEDING B tone in its own
    channel. `b_gap_ms` is what is left; at zero they abut and fuse into one tone of twice the
    duration.

    The subtle one: shifted LATE, it walks towards its own A partner, which lags it by exactly
    `lag`. At delta = lag the two coincide and the listener hears a chord that the standard
    interval did not contain. That is not a larger version of the same cue, it is a different
    cue, and it is available at every dT above zero -- worst at small lags, where it sits right
    where the staircase starts. Shifted EARLY the same tone moves AWAY from its partner and
    towards the previous A instead, which is `soa - lag` away, so the clean range is much wider.

    dT = 0% is the exception that proves it: there the standard IS the chord and any shift
    breaks it, so `makes_more_synchronous` is False and the shrinking separation is the signal.
    """
    lag = cfg.lag_ms(lag_pct)
    sign = 1 if direction >= 0 else -1
    own = abs(lag - sign * delta_ms)          # to its own A partner after the shift
    prev = abs((cfg.soa_ms - lag) + sign * delta_ms)   # to the A tone before that
    return {"a_sep_ms": min(own, prev),
            "b_gap_ms": cfg.soa_ms - cfg.tone_ms - (delta_ms if sign < 0 else 0.0),
            "makes_more_synchronous": bool(own < lag - 1e-9),
            "own_a_sep_ms": own, "prev_a_sep_ms": prev}


def max_safe_delta_ms(cfg: Config, pcts: Sequence[float], direction: int,
                      margin_ms: float = FUSION_MARGIN_MS) -> float:
    """The largest delta that keeps every condition clear of both hazards, or 0 if none does."""
    best = 0.0
    for dl in np.arange(1.0, cfg.soa_ms - cfg.tone_ms, 0.5):
        ok = True
        for p in pcts:
            g = displaced_tone_clearance(cfg, p, float(dl), direction)
            if g["b_gap_ms"] < margin_ms:
                ok = False
            if p > 0 and (g["makes_more_synchronous"] or g["a_sep_ms"] < margin_ms):
                ok = False
            if not ok:
                break
        if ok:
            best = float(dl)
    return best


def interval_ms(cfg: Config, n_precursor: Optional[int] = None) -> float:
    """How long one interval lasts, for a sequence with this many precursors.

    Computed from the widest lag the dT axis allows (100%) and the largest shift the track
    allows, NOT from the conditions present, so that a condition's duration does not depend on
    what else is in the configuration.

    It is deliberately a function of precursor count rather than one number for the whole
    session. Duration has to be constant between the standard and the target interval of a
    trial, and it is -- that is what stops it being a cue. It does NOT have to be constant
    across conditions: the listener always knows which condition they are in (the lag is
    audible) and cannot use that to answer a within-trial question. Padding every condition to
    the longest one cost the full configuration about a hundred minutes of pure silence.
    """
    n = (cfg.n_precursor if n_precursor is None else n_precursor) + 1
    return (cfg.lead_ms + (n - 1) * cfg.soa_ms + cfg.tone_ms
            + cfg.soa_ms / 2.0 + cfg.delta_max_ms + cfg.tail_ms)


def sync_reference_gap_ms(cfg: Config, lag_pct: float) -> float:
    """Silence between the last A precursor and the final A tone under reference='sync'.

    That mode drags the final A tone back onto the unshifted final B so that the judged
    interval is always "synchronous versus delta". The cost is that the final A tone leaves its
    own grid, and at a large enough dT it meets the A tone before it: at zero gap the two
    become one continuous tone of twice the duration, which is not the stimulus the condition
    name describes. Zero is already too far, not merely negative.
    """
    return cfg.soa_ms - cfg.lag_ms(lag_pct) - cfg.tone_ms


def _tag(pct: float) -> str:
    return f"{pct:g}".replace(".", "p")


# ----------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------
def validate(cfg: Config) -> Derived:
    """Refuse impossible or self-defeating configurations, and return what follows from a legal one."""
    notes: List[str] = []
    if cfg.tone_ms <= 0 or cfg.soa_ms <= 0:
        raise ConfigError("tone_ms and soa_ms must be positive")
    if cfg.ramp_ms * 2 > cfg.tone_ms:
        raise ConfigError(f"ramp_ms={cfg.ramp_ms} twice over does not fit in tone_ms={cfg.tone_ms}")
    if abs(cfg.duty - 0.5) > 1e-9 and not cfg.allow_nonmonotone_duty:
        raise ConfigError(
            f"tone_ms must be exactly half of soa_ms (got duty={cfg.duty:.3f}). At any other duty "
            "cycle the model's own segregation index is not monotone in dT (see "
            "model.duty_cycle_scan), so an ordered behavioural result could not be interpreted. "
            "Set allow_nonmonotone_duty=true only for the planned follow-up that separates "
            "onset lag from acoustic overlap, and say so in the write-up.")
    if cfg.n_precursor < 1:
        raise ConfigError("n_precursor must be at least 1: the sequence needs a context")
    if cfg.df_semitones <= 0:
        raise ConfigError("df_semitones must be positive; B is defined as the upper tone")
    if not 0.0 <= cfg.catch_rate < 0.5:
        raise ConfigError("catch_rate must be in [0, 0.5)")
    if cfg.delta_min_ms <= 0 or cfg.delta_max_ms <= cfg.delta_min_ms:
        raise ConfigError("need 0 < delta_min_ms < delta_max_ms")
    if cfg.delta_direction not in ("random", "forward", "backward"):
        raise ConfigError("delta_direction must be random, forward or backward")
    if cfg.n_down < 1:
        raise ConfigError("n_down must be at least 1")
    if cfg.scramble_min_gap < cfg.tone_ms:
        raise ConfigError(
            f"scramble_min_gap_ms={cfg.scramble_min_gap:g} is below tone_ms={cfg.tone_ms:g}, so two "
            "scrambled A tones could overlap and sum. The control would then hold a louder event "
            "the coherent condition never contains, and would stop being matched.")
    if len(cfg.step_factors) != len(cfg.reversals_per_step) + 1:
        raise ConfigError("reversals_per_step must have one entry fewer than step_factors")
    if any(f <= 1.0 for f in cfg.step_factors):
        raise ConfigError("every step factor must exceed 1 (the track is multiplicative)")
    if cfg.n_final_reversals < 2:
        raise ConfigError("n_final_reversals < 2 gives a threshold with no averaging at all")

    conds = conditions(cfg)
    if not conds:
        raise ConfigError("no conditions: sweep_pcts is empty and every control is switched off")
    names = [c.name for c in conds]
    if len(set(names)) != len(names):
        raise ConfigError(f"duplicate condition names: {sorted({n for n in names if names.count(n) > 1})}")
    for c in conds:
        if c.a_kind not in A_KINDS:
            raise ConfigError(f"{c.name}: a_kind must be one of {A_KINDS}")
        if c.reference not in REFERENCES:
            raise ConfigError(f"{c.name}: reference must be one of {REFERENCES}")
        if not 0.0 <= c.lag_pct <= 100.0:
            raise ConfigError(f"{c.name}: lag_pct must be in [0, 100]")
        if c.reference == "sync":
            gap = sync_reference_gap_ms(cfg, c.lag_pct)
            if gap <= 0:
                raise ConfigError(
                    f"{c.name}: reference='sync' at dT={c.lag_pct:g}% leaves {gap:.1f} ms between "
                    "the final A tone and the A tone before it, so the two meet or overlap and "
                    "merge into a single long tone -- a different stimulus wearing the same name. "
                    "Use reference='yoked', or a dT strictly below "
                    f"{cfg.pct(cfg.soa_ms - cfg.tone_ms):.0f}%.")
        if c.reference == "tempo" and c.tempo_gap_ms is None:
            raise ConfigError(f"{c.name}: reference='tempo' needs tempo_gap_ms")

    # a shift must not carry the final B tone into the tone before it, nor onto an A tone
    headroom = cfg.soa_ms - cfg.tone_ms
    sweep = sorted({c.lag_pct for c in conds if c.a_kind == "coherent"})
    if sweep:
        dirs = ((-1,), (1,), (-1, 1))[{"backward": 0, "forward": 1, "random": 2}[cfg.delta_direction]]
        safe = min(max_safe_delta_ms(cfg, sweep, s) for s in dirs)
        if cfg.delta_max_ms > safe:
            worst = None
            for p in sweep:
                for s in dirs:
                    g = displaced_tone_clearance(cfg, p, cfg.delta_max_ms, s)
                    if p > 0 and (g["makes_more_synchronous"] or g["a_sep_ms"] < FUSION_MARGIN_MS):
                        worst = (p, "late" if s > 0 else "early", g)
                        break
                if worst:
                    break
            where = ("" if worst is None else
                     f" At dT={worst[0]:g}% a {worst[1]} shift of {cfg.delta_max_ms:g} ms brings it "
                     f"within {worst[2]['a_sep_ms']:.1f} ms of an A tone"
                     + (", MORE synchronous with its partner than the standard is."
                        if worst[2]["makes_more_synchronous"] else "."))
            notes.append(
                f"delta_max_ms={cfg.delta_max_ms:g} exceeds the {safe:.0f} ms this geometry supports "
                f"with a {FUSION_MARGIN_MS:.0f} ms fusion margin.{where} Near that delta the "
                "listener is detecting a chord that appeared rather than a tone that moved. "
                "Lower delta_max_ms, or set delta_direction='backward', which moves the tone away "
                "from its partner instead of towards it.")
    if cfg.delta_max_ms >= headroom:
        raise ConfigError(
            f"delta_max_ms={cfg.delta_max_ms} is not less than soa_ms - tone_ms = {headroom:.1f} ms, "
            "so a backward shift would drive the final B tone into the one before it and the "
            "trial would contain a collision rather than a displacement.")
    if cfg.catch_delta_ms > cfg.delta_max_ms:
        raise ConfigError("catch_delta_ms exceeds delta_max_ms")
    if cfg.catch_at_pct is not None:
        if not any(abs(c.lag_pct - cfg.catch_at_pct) < 1e-9 and c.a_kind == "coherent"
                   for c in conds):
            raise ConfigError(
                f"catch_at_pct={cfg.catch_at_pct:g} names a coherent condition this design does "
                "not contain, so the probe would be a stimulus the listener never otherwise hears.")
    elif cfg.catch_rate > 0:
        notes.append(
            "catch trials are built from whichever condition hosts them, so a probe in a hard "
            "condition is not easy and a miss there does not mean inattention. Set catch_at_pct "
            "(0.0 is the obvious choice) to make every probe the same easy stimulus.")
    if cfg.delta_start_ms > cfg.delta_max_ms:
        raise ConfigError("delta_start_ms exceeds delta_max_ms")
    if cfg.delta_start_ms * cfg.step_factors[0] > cfg.delta_max_ms:
        notes.append(
            f"an error on the first trial would call for {cfg.delta_start_ms * cfg.step_factors[0]:.0f} ms, "
            f"above the {cfg.delta_max_ms:.0f} ms ceiling, so the track will sit at the ceiling until it "
            "gets three right. That is before the first reversal and outside the threshold average, but "
            "the audit reports how often it happened.")

    # geometry. The interval is padded to a length that does not depend on which conditions are
    # in this config, nor on the shift: it is computed from the largest lag the axis allows
    # (100%) and the largest shift the track allows. Duration is therefore constant across
    # conditions and between the standard and the target, and cannot be a cue.
    lag_by_pct = {float(p): cfg.lag_ms(p) for p in sorted({c.lag_pct for c in conds})}
    interval = interval_ms(cfg)
    longest = max(interval_ms(cfg, c.n_precursor) for c in conds)
    trial = 2 * interval + cfg.isi_ms
    mean_trial = 2 * float(np.mean([interval_ms(cfg, c.n_precursor) for c in conds])) + cfg.isi_ms

    # frequency separation
    from .model import channel_crosstalk, harmonic_proximity
    xt = channel_crosstalk(cfg.f_a_hz, cfg.f_b_hz)
    harm = harmonic_proximity(cfg.f_a_hz, cfg.f_b_hz)
    if xt["erbs"] < 3.0:
        raise ConfigError(
            f"the two tones are {xt['erbs']:.1f} ERBs apart. Below about 3 they share auditory "
            "filters, the 'one channel per tone' reduction the model prediction rests on stops "
            "holding, and a shift of B changes the envelope inside A's filter -- a peripheral "
            "cue the design does not want. Increase df_semitones.")
    if harm["mistuning_pct"] < 3.0:
        notes.append(
            f"f_b/f_a = {harm['frequency_ratio']:.4f} is only {harm['mistuning_pct']:.2f}% from "
            f"{harm['ratio']}. Two simultaneous tones in a simple ratio fuse partly through "
            "harmonicity, which is a grouping cue this experiment does not manipulate and cannot "
            "separate from synchrony. Consider an df_semitones further from a small-integer ratio.")

    if cfg.is_complex:
        if len(cfg.partials_hz) % 2:
            raise ConfigError(
                f"partials_hz has {len(cfg.partials_hz)} entries; it must be even so the set can "
                "be split evenly between A and B.")
        if len(set(cfg.partials_hz)) != len(cfg.partials_hz):
            raise ConfigError("partials_hz contains a repeated frequency")
        want = {c.interleaved for c in conds}
        for il in sorted(want):
            au = partial_audit(cfg, il)
            where = au["arrangement"]
            if au["min_erbs"] < 3.0:
                lo, hi = au["min_erb_pair"]
                raise ConfigError(
                    f"{where}: partials at {lo:.0f} and {hi:.0f} Hz are {au['min_erbs']:.1f} ERBs "
                    "apart. Below about 3 they share an auditory filter, so they are not two "
                    "channels and the whole point of using a complex tone is lost.")
            if au["common_f0"] is not None:
                f0 = au["common_f0"]
                raise ConfigError(
                    f"{where}: all partials fit harmonics {f0['harmonics']} of "
                    f"{f0['f0_hz']:.1f} Hz to within {100 * f0['error']:.1f}%. A harmonic set "
                    "fuses into one object through harmonicity, which is exactly the grouping "
                    "cue this experiment must not supply. Choose an inharmonic set.")
            if au["min_simple_ratio_mistuning_pct"] < 3.0:
                lo, hi = au["min_simple_ratio_pair"]
                notes.append(
                    f"{where}: {lo:.0f}/{hi:.0f} Hz is only "
                    f"{au['min_simple_ratio_mistuning_pct']:.1f}% from {au['min_simple_ratio']}, "
                    "close enough for those two partials to fuse or beat.")
            for lbl, key in (("A", "within_a_f0"), ("B", "within_b_f0")):
                if au[key] is not None:
                    notes.append(
                        f"{where}: tone {lbl}'s own partials are harmonics "
                        f"{au[key]['harmonics']} of {au[key]['f0_hz']:.1f} Hz. That tone will fuse "
                        "through harmonicity rather than through common onset.")
    elif cfg.interleaved_pcts:
        raise ConfigError("interleaved_pcts needs partials_hz; pure tones cannot interleave")

    mc = _model_curve(tuple(sorted(lag_by_pct)), cfg.tone_ms, cfg.soa_ms, cfg.n_tones)
    mbc = _model_by_condition(cfg, conds, interval)

    n_tracks = len(conds) * cfg.tracks_per_condition
    per_track = _expected_track_trials(cfg)
    est_trials = int(round(n_tracks * per_track * (1.0 + cfg.catch_rate)))
    est_minutes = est_trials * (mean_trial / 1000.0 + 1.6) / 60.0

    if cfg.level_rove_db > 0:
        notes.append(
            f"levels are roved by +-{cfg.level_rove_db:g} dB per interval. Shifting the B tone cannot "
            "change the energy in either channel, so the rove is insurance rather than a fix, and the "
            "audit confirms the roved difference carries no information about which interval was the "
            "target.")
    if not cfg.include_b_only:
        notes.append("include_b_only is off: without the ceiling there is nothing to normalise the "
                     "sweep against and the coherence index cannot be computed.")

    # H2 -- the only test that separates the hypothesis from its rivals -- is a slope across the
    # dT levels at which a control is matched to the coherent condition. Too few matched levels
    # and the slope has almost no power, while H1 still fires happily, so a short design can
    # look like a result and settle nothing. Measured: 5 matched levels give 87% power, 3 give
    # 9%. That is worth saying out loud rather than leaving in a table somewhere.
    sweep = {c.lag_pct for c in conds if c.a_kind == "coherent"}
    matched = max((len(sweep & {c.lag_pct for c in conds if c.a_kind == k})
                   for k in ("scrambled", "pair_only", "nopartner")), default=0)
    if matched < 4:
        notes.append(
            f"only {matched} dT level(s) have a matched control, so H2 -- the test that "
            "separates temporal coherence from interval discrimination and from local acoustic "
            "overlap -- has very little power (about 9% at three levels against 87% at five, "
            "from tcoh.analysis.power). H1 will still fire, and H1 on its own settles nothing. "
            "Treat this configuration as a screen, not as a test of the hypothesis.")
    if cfg.monaural:
        notes.append("monaural presentation (left only), as in Elhilali et al.")

    return Derived(cfg.f_b_hz, lag_by_pct, interval, longest, trial, mean_trial, conds,
                   n_tracks, matched,
                   est_trials, est_minutes,
                   headroom, xt["erbs"], xt["roex_attenuation_db"], xt["roex_is_floor"], harm, mc,
                   mbc, tuple(notes))


@_lru_cache(maxsize=64)
def _model_curve(pcts: Tuple[float, ...], tone_ms: float, soa_ms: float,
                 n_tones: int) -> Dict[float, float]:
    """The model prediction, memoised.

    validate() is called by anything that needs a derived quantity, including the runner
    between trials, and the coherence model is a bank of sixty convolutions. Caching it keeps
    validate() cheap enough to stay the single source of derived numbers instead of being
    something callers learn to avoid.
    """
    from .model import predicted_curve
    return {p: c.ratio for p, c in predicted_curve(
        pcts, tone_ms=tone_ms, soa_ms=soa_ms, n_tones=n_tones).items()}


def _model_by_condition(cfg: Config, conds: Tuple[Condition, ...],
                        _unused: float) -> Dict[str, float]:
    """The model's segregation index for each condition AS BUILT, controls included.

    Keying the prediction on dT alone would quietly give every control the prediction of the
    coherent sequence at the same lag, which is the opposite of what a control is for: the
    whole point of the scrambled and pair-only conditions is that their A sequence is NOT the
    coherent one, and the difference between their predicted indices and the coherent ones IS
    the interaction H2 tests. Stochastic conditions are averaged over several draws.
    """
    from .model import coherence_matrix, filter_bank, sequence_envelope
    from .stimulus import a_onsets, b_onsets, scramble_onsets, scramble_window
    fs = 1000.0
    bank = filter_bank(fs)
    out: Dict[str, float] = {}
    for c in conds:
        if c.a_kind == "absent":
            out[c.name] = float("nan")           # one channel: the index is not defined
            continue
        rng = np.random.default_rng(0xC0DE)
        draws = 8 if c.a_kind == "scrambled" else 1
        vals = []
        b = b_onsets(cfg, c.n_precursor)
        for _ in range(draws):
            sc = None
            if c.a_kind == "scrambled":
                lo, hi = scramble_window(cfg, c)
                sc = scramble_onsets(rng, lo, hi, b.size - 1, cfg.scramble_min_gap)
            a = a_onsets(cfg, c, sc)
            ms = interval_ms(cfg, c.n_precursor)
            env = np.stack([sequence_envelope(a, cfg.tone_ms, cfg.ramp_ms, ms, fs),
                            sequence_envelope(b, cfg.tone_ms, cfg.ramp_ms, ms, fs)])
            vals.append(coherence_matrix(env, fs, bank=bank).ratio)
        out[c.name] = float(np.mean(vals))
    return out


def _expected_track_trials(cfg: Config) -> float:
    """A rough but honest expectation for how long one 3-down-1-up track runs.

    Reversals need runs of `n_down` correct or one incorrect; near the 79.4% point a reversal
    costs about `n_down + 1` trials on average, plus the trials spent getting down from the
    starting value. Simulation in `tcoh.track.simulate` gives the real distribution; this is
    only for the duration estimate a user sees before committing an evening to it.
    """
    n_rev = sum(cfg.reversals_per_step) + cfg.n_final_reversals
    return min(cfg.max_trials_per_track, 6.0 + n_rev * (cfg.n_down + 1.0))


DEFAULT = Config()


def describe(cfg: Config) -> str:
    d = validate(cfg)
    L = [f"two-tone coherence  |  config {cfg.hash()}"]
    if cfg.is_complex:
        arrangements = sorted({c.interleaved for c in d.conditions})
        for il in arrangements:
            au = partial_audit(cfg, il)
            a_lbl = " + ".join(f"{f:.0f}" for f in au["a_hz"])
            b_lbl = " + ".join(f"{f:.0f}" for f in au["b_hz"])
            head = "  tones     " if il == arrangements[0] else "            "
            L.append(f"{head} {au['arrangement']:11} A {a_lbl} Hz   B {b_lbl} Hz")
            L.append(f"                         {au['min_erbs']:.1f} ERB apart at the closest, "
                     f"{au['min_simple_ratio_mistuning_pct']:.0f}% from {au['min_simple_ratio']}, "
                     f"{'no common f0' if au['common_f0'] is None else 'COMMON F0'}")
    else:
        L.append(f"  tones      A {cfg.f_a_hz:.0f} Hz, B {d.f_b_hz:.0f} Hz "
                 f"({cfg.df_semitones:g} st, {cfg.df_semitones / 12:.2f} oct, {d.erbs_apart:.1f} ERB)")
    L.append(f"             {cfg.tone_ms:g} ms tones, {cfg.ramp_ms:g} ms ramps, {cfg.soa_ms:g} ms SOA "
             f"(duty {cfg.duty:.2f}), {cfg.n_tones} per channel")
    if not cfg.is_complex:
        # for a complex set these are properties of every partial pair, and the audit lines
        # above already report the worst of them; f_a/f_b are not the tones being played
        L.append(f"             cross-channel leak {'<' if d.roex_leak_is_floor else ''}"
                 f"-{d.roex_attenuation_db:.0f} dB; nearest low-order ratio {d.harmonic['ratio']} "
                 f"({d.harmonic['mistuning_pct']:.1f}% away)")
    L += [
         f"  trial      {d.interval_ms:.0f} ms + {cfg.isi_ms:.0f} ms + {d.interval_ms:.0f} ms "
         f"= {d.trial_ms / 1000:.2f} s of sound",
         f"  track      {cfg.n_down}-down 1-up, steps x{'/x'.join(f'{s:g}' for s in cfg.step_factors)}, "
         f"threshold = geometric mean of the last {cfg.n_final_reversals} reversals",
         f"             delta starts at {cfg.delta_start_ms:g} ms, capped at {cfg.delta_max_ms:g} ms "
         f"({d.collision_headroom_ms:.0f} ms before a collision)",
         f"  conditions {len(d.conditions)} x {cfg.tracks_per_condition} tracks "
         f"= {d.n_tracks} tracks, about {d.est_trials} trials, {d.est_minutes:.0f} min",
         ""]
    L.append("  condition      dT%   A            reference   model l2/l1   role")
    for c in d.conditions:
        mc = d.model_by_condition.get(c.name, float("nan"))
        pred = "  -  " if mc != mc else f"{mc:.3f}"
        L.append(f"  {c.name:<13} {c.lag_pct:5.1f}  {c.a_kind:<11}  {c.reference:<9}   "
                 f"{pred:^11}   {c.role}")
    coh = [(c.lag_pct, d.model_by_condition[c.name]) for c in d.conditions if c.a_kind == "coherent"]
    for kind in ("scrambled", "pair_only"):
        ctl = {c.lag_pct: d.model_by_condition[c.name] for c in d.conditions if c.a_kind == kind}
        shared = [(p, v, ctl[p]) for p, v in coh if p in ctl]
        if len(shared) >= 2:
            L.append("")
            L.append(f"  predicted {kind} minus coherent (what H2 tests; it should fall with dT):")
            L.append("    " + "   ".join(f"dT={p:g}%: {b - a:+.2f}" for p, a, b in shared))
    if d.notes:
        L.append("")
        for n in d.notes:
            L.append("  note: " + n)
    return "\n".join(L)
