"""The parameters of the sheared-figure task, and the geometry that constrains them.

The stimulus
-------------
Four pure tones at fixed, inharmonic, well-resolved frequencies repeat together at `rate_hz`.
Within each repetition their onsets are SHEARED: tone k starts `k * step` after tone 0, so the
figure runs from a chord (step = 0) to an even arpeggio. On the last repetition one tone -- the
same one every time, see `target_index` -- is displaced in time by `delta`, and the listener
says which of two intervals contained the displacement.

The axis
---------
`step` is quoted as a percentage of the step at which the four onsets are evenly spaced over
the period, `iso_step = period / n_tones`. At 0% the four tones are simultaneous and the model
sees exactly one object; at 100% the combined four-channel onset train is isochronous at
`n_tones * rate_hz`, and the model sees very nearly four. The axis is therefore bounded and
interpretable at both ends without any free parameter.

Why 100% is not simply 'the most sheared'
------------------------------------------
At exactly 100% the combined train is perfectly regular, which hands the listener a rhythmic
reference that exists at no other step. This is the same trap that produced an unexplained dip
at dT = 100% in the two-tone task, and it is why the default step set samples 90% as well: a
discontinuity at exactly 100% and nowhere else is a rhythm cue, a smooth decline is not.

The two references, and why the design has its null built in
-------------------------------------------------------------
A listener can judge the displaced tone against two things:

  * its OWN previous repetitions, which are isochronous at `rate_hz` in that one channel. This
    reference does not depend on the step size at all -- every channel is isochronous whatever
    the shear.
  * the OTHER THREE TONES of the same repetition. This reference is the figure, and it degrades
    as the figure shears apart.

So a listener using only within-channel timing produces a FLAT curve, by construction, and a
listener using the figure produces a rising one. The null hypothesis is a property of the
stimulus rather than a control condition bolted on afterwards. `include_single_tone` measures
the within-channel limit directly, as a ceiling.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class ConfigError(ValueError):
    pass


# Four resolved, inharmonic tones, near-equally spaced in log frequency (about 10.2 semitones
# apart). Chosen by searching 300 000 random quadruples against three requirements: every pair
# at least 3.5 ERB apart so each tone owns an auditory channel, every pair at least 5% from any
# simple ratio, and no fundamental in 120-600 Hz explaining all four as harmonics of 12 or
# below. 501 quadruples passed; this one is 4.4 ERB apart at the closest and 8.0% from the
# nearest simple ratio.
DEFAULT_FREQS: Tuple[float, ...] = (683.0, 1229.0, 2241.0, 3985.0)

CORE_STEPS: Tuple[float, ...] = (0.0, 15.0, 30.0, 50.0, 75.0, 100.0)
# Spaced by the MODEL, not by the step. The effective number of objects runs 1.00, 1.69, 2.59,
# 3.41, 3.70, 3.84 across these six, which covers the model's dynamic range evenly; stepping
# uniformly in milliseconds instead would spend half the conditions on the flat top, where the
# index moves from 3.70 to 3.84 and nothing is being distinguished.
#
# 75% and 100% are both on that flat top, and that is the point of including both: the model
# says they are the same figure (0.901 against 0.947), so a behavioural difference between them
# cannot be a coherence effect. 100% is the step at which the combined onset train is exactly
# isochronous, and this pair is what separates that rhythmic cue from the binding the task is
# meant to measure.
FUSION_MARGIN_MS = 25.0


@dataclass(frozen=True)
class Condition:
    """One adaptive track's worth of stimulus."""
    name: str
    step_pct: float
    kind: str = "figure"          # figure | single  (single = the target tone alone)
    role: str = "main"            # main | control

    def label(self) -> str:
        if self.kind == "single":
            return "target tone alone"
        return f"step={self.step_pct:g}%"


@dataclass(frozen=True)
class Config:
    # ---- audio ---------------------------------------------------------------
    sample_rate: int = 48000
    monaural: bool = False
    tone_amplitude: float = 0.05          # peak amplitude of ONE tone
    tone_level_db_spl: float = 65.0
    level_rove_db: float = 3.0

    # ---- the figure ----------------------------------------------------------
    freqs_hz: Tuple[float, ...] = DEFAULT_FREQS
    rate_hz: float = 3.0                  # repetitions of the whole figure per second
    tone_ms: float = 60.0
    ramp_ms: float = 10.0
    n_repeats: int = 5                    # repetitions per interval; the last carries the jitter

    # ---- the sheared axis ----------------------------------------------------
    step_pcts: Tuple[float, ...] = CORE_STEPS
    target_index: int = 2                 # 0-based: the 3rd tone counting up in frequency

    # ---- the tracked variable ------------------------------------------------
    delta_start_ms: float = 12.0
    delta_min_ms: float = 0.25
    delta_max_ms: float = 50.0
    delta_direction: str = "random"       # random | forward | backward

    # ---- timing --------------------------------------------------------------
    lead_ms: float = 200.0
    tail_ms: float = 200.0
    isi_ms: float = 500.0

    # ---- the adaptive rule (names match tcoh.track.Track, which is reused) ----
    n_down: int = 3
    step_factors: Tuple[float, ...] = (4.0, 2.0, math.sqrt(2.0))
    reversals_per_step: Tuple[int, ...] = (1, 2)
    n_final_reversals: int = 6
    max_trials_per_track: int = 90
    tracks_per_condition: int = 2

    # ---- controls ------------------------------------------------------------
    include_single_tone: bool = True
    # The target tone on its own, same rate, same jitter. This is the WITHIN-CHANNEL limit: the
    # best a listener can do with no figure at all. Every figure condition is read against it,
    # and a figure condition that is no better than this one was not using the figure.

    # ---- catch trials --------------------------------------------------------
    catch_rate: float = 0.07
    catch_delta_ms: float = 45.0
    catch_at_pct: Optional[float] = 0.0   # every probe is the same easy stimulus, wherever it lands
    max_catch_miss_rate: float = 0.15

    # ---- procedure -----------------------------------------------------------
    feedback: bool = True
    practice_trials: int = 12
    practice_criterion: float = 0.75
    practice_delta_ms: float = 45.0
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
        for name in ("freqs_hz", "step_pcts", "step_factors", "reversals_per_step"):
            if kw.get(name) is not None:
                kw[name] = tuple(kw[name])
        return cls(**kw)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]

    # -- geometry every module needs and nobody should recompute ---------------
    @property
    def n_tones(self) -> int:
        return len(self.freqs_hz)

    @property
    def period_ms(self) -> float:
        return 1000.0 / self.rate_hz

    @property
    def iso_step_ms(self) -> float:
        """The step at which the combined onset train is evenly spaced over the period."""
        return self.period_ms / self.n_tones

    def step_ms(self, pct: float) -> float:
        return float(pct) / 100.0 * self.iso_step_ms

    @property
    def target_freq_hz(self) -> float:
        return self.freqs_hz[self.target_index]

    def figure_span_ms(self, pct: float) -> float:
        return (self.n_tones - 1) * self.step_ms(pct)

    def interval_ms(self) -> float:
        """Constant across conditions, computed from the widest step and the largest jitter.

        Equal duration in both intervals of a trial is what stops duration being a cue; equal
        duration ACROSS conditions is not required, but it costs nothing here and removes one
        thing a reader has to check.
        """
        widest = max(self.step_pcts) if self.step_pcts else 0.0
        return (self.lead_ms + (self.n_repeats - 1) * self.period_ms
                + self.figure_span_ms(widest) + self.tone_ms + self.delta_max_ms + self.tail_ms)

    def trial_ms(self) -> float:
        return 2 * self.interval_ms() + self.isi_ms


DEFAULT = Config()


# ----------------------------------------------------------------------------
def conditions(cfg: Config) -> Tuple[Condition, ...]:
    out = [Condition(f"step_{_tag(p)}", p, "figure", "main") for p in cfg.step_pcts]
    if cfg.include_single_tone:
        out.append(Condition("single", 0.0, "single", "control"))
    return tuple(out)


def _tag(p: float) -> str:
    return f"{p:g}".replace(".", "p")


def onsets_ms(cfg: Config, step_pct: float, rep: int) -> np.ndarray:
    """Onset of every tone in repetition `rep`, before any jitter."""
    return cfg.lead_ms + rep * cfg.period_ms + np.arange(cfg.n_tones) * cfg.step_ms(step_pct)


def combined_gaps_ms(cfg: Config, step_pct: float) -> np.ndarray:
    """Intervals of the combined onset train over one period, in order."""
    o = np.arange(cfg.n_tones) * cfg.step_ms(step_pct)
    return np.diff(np.concatenate([o, [cfg.period_ms]]))


def is_isochronous(cfg: Config, step_pct: float, tol: float = 1e-9) -> bool:
    g = combined_gaps_ms(cfg, step_pct)
    return bool(np.ptp(g) < tol)


def target_clearance(cfg: Config, step_pct: float, delta_ms: float, direction: int) -> dict:
    """How close the displaced tone comes to anything else, and to what.

    Three distances matter and they are not the same thing:

    * `own_channel_ms` -- to the target's own neighbouring repetitions, a whole period away.
      Nothing the staircase can reach comes near this, which is why the within-channel
      reference survives at every step and every delta.
    * `neighbour_ms` -- to the adjacent tones of the SAME repetition, `step` away in time but
      in a different frequency channel. There is no masking here and no overlap that matters;
      what changes is which tones are near-synchronous with which.
    * `crosses_neighbour` -- whether the jitter is larger than the step, so the target moves
      past a neighbour in the combined order. At step 0 this is meaningless, and at every step
      it is inherent to the manipulation rather than a fault: displacing a tone out of a chord
      IS the signal. It is reported per trial so the analysis can say how often a threshold
      was measured in that regime, never silently prevented.
    """
    step = cfg.step_ms(step_pct)
    sign = 1 if direction >= 0 else -1
    moved = delta_ms * sign
    k = cfg.target_index
    others = [(j - k) * step for j in range(cfg.n_tones) if j != k]
    neighbour = min(abs(o - moved) for o in others) if others else float("inf")
    return {"own_channel_ms": cfg.period_ms - abs(moved),
            "neighbour_ms": float(neighbour),
            "crosses_neighbour": bool(step > 0 and abs(moved) >= step),
            "step_ms": step}


def max_safe_delta_ms(cfg: Config, margin_ms: float = FUSION_MARGIN_MS) -> float:
    """The largest jitter that cannot make the target overlap ITS OWN neighbouring repetitions.

    This is the only hard acoustic constraint, and it is worth being precise about why the
    obvious-looking alternatives are not constraints at all.

    The target's neighbours WITHIN a repetition are a different frequency each, so a jitter
    that brings it close to one of them produces near-synchrony, not masking or summation. At
    step 0 the tones are exactly simultaneous already: that is the stimulus, not a collision.
    An earlier version of this function took the smallest gap in the combined onset train,
    which is zero at step 0 by construction, and therefore refused every configuration.

    What cannot be allowed is the target running into the same tone one period earlier or
    later, because that IS the same frequency and the two would sum. With a 333 ms period and a
    60 ms tone that allows a quarter of a second, so in practice nothing the staircase can
    reach comes near it. Crossing a neighbour in the combined order is a separate matter --
    inherent to the manipulation, reported per trial, never prevented. See `target_clearance`.
    """
    return max(0.0, cfg.period_ms - cfg.tone_ms - margin_ms)


@dataclass(frozen=True)
class Derived:
    conditions: Tuple[Condition, ...]
    n_tracks: int
    period_ms: float
    iso_step_ms: float
    interval_ms: float
    trial_ms: float
    min_erbs: float
    min_ratio_mistuning_pct: float
    model_index: Dict[str, float]
    isochronous_steps: Tuple[float, ...]
    max_safe_delta_ms: float
    notes: Tuple[str, ...]


def validate(cfg: Config) -> Derived:
    from .model import effective_objects
    notes: List[str] = []

    if cfg.n_tones < 2:
        raise ConfigError("the figure needs at least two tones")
    if not 0 <= cfg.target_index < cfg.n_tones:
        raise ConfigError(f"target_index {cfg.target_index} is outside 0..{cfg.n_tones - 1}")
    if cfg.target_index in (0, cfg.n_tones - 1):
        notes.append(
            f"target_index {cfg.target_index} is an EDGE tone. Displacing it changes when the "
            "figure begins or ends, which is a cue about the figure as a whole rather than "
            "about one component of it. An interior tone has a neighbour on each side.")
    if cfg.tone_ms >= cfg.period_ms:
        raise ConfigError(f"tone_ms {cfg.tone_ms:g} does not fit in the {cfg.period_ms:.1f} ms period")
    if any(p < 0 or p > 100 for p in cfg.step_pcts):
        raise ConfigError("step_pcts are percentages of the isochronous step, so 0..100")
    if cfg.delta_min_ms <= 0 or cfg.delta_max_ms <= cfg.delta_min_ms:
        raise ConfigError("need 0 < delta_min_ms < delta_max_ms")
    if cfg.delta_direction not in ("random", "forward", "backward"):
        raise ConfigError("delta_direction must be random, forward or backward")
    if len(cfg.step_factors) != len(cfg.reversals_per_step) + 1:
        raise ConfigError("reversals_per_step must have one entry fewer than step_factors")
    if cfg.n_repeats < 2:
        raise ConfigError("n_repeats must be at least 2: one repetition cannot establish a figure")

    # the tones have to be separate channels and not fuse
    from tcoh.model import channel_crosstalk
    from tcoh.config import _common_f0, _simple_ratio_mistuning
    import itertools
    f = sorted(cfg.freqs_hz)
    if len(set(f)) != len(f):
        raise ConfigError("freqs_hz contains a repeated frequency")
    erbs = min(channel_crosstalk(x, y)["erbs"] for x, y in itertools.combinations(f, 2))
    mis = min(_simple_ratio_mistuning(x, y)[0] for x, y in itertools.combinations(f, 2))
    if erbs < 3.0:
        raise ConfigError(
            f"the closest pair of tones is {erbs:.1f} ERB apart. Below about 3 they share an "
            "auditory filter, so they are not separate channels and 'four components' is a "
            "description of the synthesis rather than of what the listener receives.")
    f0 = _common_f0(f)
    if f0 is not None:
        raise ConfigError(
            f"all four tones fit harmonics {f0['harmonics']} of {f0['f0_hz']:.1f} Hz to within "
            f"{100 * f0['error']:.1f}%. A harmonic set fuses through harmonicity, which is a "
            "grouping cue this design does not manipulate and cannot separate from timing.")
    if mis < 3.0:
        notes.append(f"the closest pair is only {mis:.1f}% from a simple ratio; expect some fusion")

    safe = max_safe_delta_ms(cfg)
    if cfg.delta_max_ms > safe:
        raise ConfigError(
            f"delta_max_ms={cfg.delta_max_ms:g} exceeds the {safe:.1f} ms the geometry allows: "
            f"beyond that the displaced tone reaches its own repetition one period away, which "
            f"is the same frequency and would sum with it.")
    # a jitter larger than the step moves the target past a neighbour in the combined order
    crossing = [p for p in cfg.step_pcts if 0 < cfg.step_ms(p) <= cfg.delta_max_ms]
    if crossing:
        notes.append(
            f"at step {', '.join(f'{p:g}%' for p in crossing)} the ceiling jitter "
            f"({cfg.delta_max_ms:g} ms) is at least as large as the step itself, so a track "
            "visiting the ceiling moves the target past a neighbour rather than merely away "
            "from it. That is inherent to displacing a tone out of a chord, not a fault; every "
            "trial records whether it happened and the analysis reports the fraction.")

    iso = tuple(p for p in cfg.step_pcts if is_isochronous(cfg, p) and p > 0)
    if iso:
        notes.append(
            f"step {', '.join(f'{p:g}%' for p in iso)} makes the combined onset train exactly "
            f"isochronous at {cfg.n_tones * cfg.rate_hz:.0f} Hz. That is a rhythmic reference "
            "available at no other step, and it is the leading alternative explanation for any "
            "improvement there. Sample a step just below it as well, or the two cannot be told "
            "apart.")
    if cfg.delta_start_ms * cfg.step_factors[0] > cfg.delta_max_ms:
        notes.append(
            f"an error on the first trial would call for "
            f"{cfg.delta_start_ms * cfg.step_factors[0]:g} ms, above the {cfg.delta_max_ms:g} ms "
            "ceiling, so the track sits at the ceiling until it gets three right. That is "
            "before the first reversal and outside the threshold average.")
    if not cfg.include_single_tone:
        notes.append(
            "include_single_tone is off, so the within-channel limit is not measured and a "
            "figure condition cannot be shown to have beaten it.")

    # Overlap, and therefore the level of the whole figure, falls as the step grows: at step 0
    # the four tones sum, at 100% they are sequential. Within a trial both intervals have the
    # same step, so this is not a cue about which interval was displaced -- but it does mean the
    # scene level differs across conditions, and it has to clear the headroom at the loud end.
    n_simul = {}
    for p in cfg.step_pcts:
        step = cfg.step_ms(p)
        n_simul[p] = 1 if step >= cfg.tone_ms else int(min(cfg.n_tones,
                                                           math.floor(cfg.tone_ms / step) + 1)) \
            if step > 0 else cfg.n_tones
    loudest = max(n_simul.values())
    peak = cfg.tone_amplitude * loudest * 10 ** (cfg.level_rove_db / 20.0)
    if peak >= 1.0:
        raise ConfigError(
            f"at the smallest step {loudest} tones overlap, so the peak reaches {peak:.2f} FS "
            "with the rove at its maximum and the waveform would clip.")
    notes.append(
        f"the figure's own level varies with the step: up to {loudest} tones overlap at the "
        f"smallest step ({10 * math.log10(loudest):.0f} dB above one tone) and they are "
        "sequential at the largest. Both intervals of a trial share a step, so this is not a "
        "cue about which interval moved, but conditions are not equal-loudness and the "
        f"analysis reports it. Worst-case peak {peak:.2f} FS "
        f"({-20 * math.log10(peak):.0f} dB of headroom).")

    conds = conditions(cfg)
    model = {c.name: (float("nan") if c.kind == "single"
                      else effective_objects(cfg, c.step_pct)["normalised"]) for c in conds}
    return Derived(
        conditions=conds, n_tracks=len(conds) * cfg.tracks_per_condition,
        period_ms=cfg.period_ms, iso_step_ms=cfg.iso_step_ms,
        interval_ms=cfg.interval_ms(), trial_ms=cfg.trial_ms(),
        min_erbs=float(erbs), min_ratio_mistuning_pct=float(mis),
        model_index=model, isochronous_steps=iso, max_safe_delta_ms=safe,
        notes=tuple(notes))


def describe(cfg: Config) -> str:
    d = validate(cfg)
    L = [f"sheared four-tone figure  |  config {cfg.hash()}",
         f"  tones      " + " + ".join(f"{f:.0f}" for f in sorted(cfg.freqs_hz)) + " Hz"
         f"   ({d.min_erbs:.1f} ERB apart at the closest, "
         f"{d.min_ratio_mistuning_pct:.0f}% from the nearest simple ratio)",
         f"             {cfg.tone_ms:g} ms tones, {cfg.ramp_ms:g} ms ramps, "
         f"figure repeats at {cfg.rate_hz:g} Hz (period {cfg.period_ms:.1f} ms)",
         f"             {cfg.n_repeats} repetitions per interval; the last one carries the jitter",
         f"  target     tone {cfg.target_index + 1} of {cfg.n_tones} counting up "
         f"({cfg.target_freq_hz:.0f} Hz), the same one on every trial",
         f"  shear      step quoted as % of {cfg.iso_step_ms:.1f} ms, the step that makes the "
         f"combined train isochronous",
         f"  trial      {d.interval_ms:.0f} ms + {cfg.isi_ms:g} ms + {d.interval_ms:.0f} ms = "
         f"{d.trial_ms / 1000:.2f} s of sound",
         f"  jitter     starts at {cfg.delta_start_ms:g} ms, capped at {cfg.delta_max_ms:g} ms "
         f"(geometry allows {d.max_safe_delta_ms:.0f} ms)",
         "",
         f"  {'condition':<12} {'step%':>7} {'step ms':>8} {'span':>7} "
         f"{'model objects':>14}  role"]
    for c in d.conditions:
        if c.kind == "single":
            L.append(f"  {c.name:<12} {'-':>7} {'-':>8} {'-':>7} {'-':>14}  "
                     f"{c.role} (within-channel limit)")
            continue
        v = d.model_index[c.name]
        L.append(f"  {c.name:<12} {c.step_pct:7.4g} {cfg.step_ms(c.step_pct):8.2f} "
                 f"{cfg.figure_span_ms(c.step_pct):6.0f}ms {1 + v * (cfg.n_tones - 1):14.2f}  "
                 f"{c.role}")
    for n in d.notes:
        L.append(f"\n  note: {n}")
    return "\n".join(L)
