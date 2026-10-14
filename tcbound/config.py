"""Parameters of the binding-bound task, and the geometry and level limits that constrain them.

The stimulus
-------------
Four pure tones (683, 1229, 2241, 3985 Hz) repeat together at `rate_hz`, five times per
interval. Within each repetition their onsets are SHEARED: tone k starts `k * step` after tone
0, so the figure runs from a chord (step 0) to an arpeggio (64 ms, where adjacent tones no
longer overlap). The target is always tone 3 (2241 Hz). On the last repetition, the PROBE, the
target is louder in one of the two intervals, and the listener says which.

What makes this a binding measure is the level rove, which differs between two trial types and
leaves exactly one cue intact in each (see `stimulus.draw_gains` for the arithmetic):

  A  the probe token is reset to a flat profile: all four tones at one new level. What survives
     is the target re the other tones AT THE SAME MOMENT: the timbre of the probe token. It
     needs the four tones to be heard as one thing.

  B  the target keeps its own level and the other three jump together. What survives is the
     target re ITS OWN earlier tones. It needs the target to be heard as a stream of its own.

The first four repetitions are statistically identical in A and B. So is the jump of the other
three tones at the probe, and so is the jump of the target RELATIVE to them. The two types
differ only in which comparison stays exact. A listener cannot know the type in advance, and the
build-up of organisation before the probe is the same in both.

The tracked variable
--------------------
SRS, the signal-to-standard ratio of profile analysis: 20 log10(dA/A) in dB, where dA is the
amplitude added to the target on the probe. The level increment the listener hears is
dL = 20 log10(1 + 10^(SRS/20)) dB. SRS is tracked with additive steps, which are multiplicative
steps in dA/A, the same rule as the timing tasks run on their log axis.
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Tuple

import numpy as np

from tshear.config import DEFAULT_FREQS


class ConfigError(ValueError):
    pass


STEPS_MS: Tuple[float, ...] = (0.0, 8.0, 16.0, 32.0, 64.0)
# A doubling series in milliseconds of onset shear between adjacent tones. Asynchrony works in
# milliseconds whatever the repetition rate, and the plausible bound runs from about 10 to 45 ms
# (the half-wave and magnitude readings of the coherence model put it at 20 and 41 ms), so the
# series has to bracket both with room above. 64 ms is the largest step at which adjacent tones
# no longer overlap and the combined onset train is still far from the isochronous 83.3 ms.

TASKS = ("A", "B", "none")
KINDS = ("figure", "capture", "alone")


def delta_db(srs_db: float) -> float:
    """Level increment in dB of the target, for a signal-to-standard ratio in dB."""
    return float(20.0 * math.log10(1.0 + 10.0 ** (float(srs_db) / 20.0)))


def srs_db(delta_db_: float) -> float:
    """Signal-to-standard ratio in dB, for a level increment in dB."""
    r = 10.0 ** (float(delta_db_) / 20.0) - 1.0
    if r <= 0:
        return float("-inf")
    return float(20.0 * math.log10(r))


@dataclass(frozen=True)
class Condition:
    """One adaptive track's worth of stimulus."""
    name: str
    task: str                   # A | B | none
    step_ms: float
    kind: str = "figure"        # figure | capture (flanker captors between tokens) | alone
    role: str = "main"          # main | control | catch

    def label(self) -> str:
        if self.kind == "alone":
            return f"{self.task}, target alone"
        if self.kind == "capture":
            return f"{self.task}, step 0 + captors"
        return f"{self.task}, step {self.step_ms:g} ms"


@dataclass(frozen=True)
class Config:
    # ---- audio ---------------------------------------------------------------
    sample_rate: int = 48000
    monaural: bool = False
    tone_amplitude: float = 0.03          # peak amplitude of ONE tone at the nominal level
    tone_level_db_spl: float = 55.0       # nominal level of one tone; roves go either side

    # ---- the figure ----------------------------------------------------------
    freqs_hz: Tuple[float, ...] = DEFAULT_FREQS
    rate_hz: float = 3.0
    tone_ms: float = 60.0
    ramp_ms: float = 10.0
    n_repeats: int = 5                    # the last repetition is the probe
    target_index: int = 2                 # 0-based: the 3rd tone counting up (2241 Hz)

    # ---- the sheared axis ----------------------------------------------------
    steps_ms: Tuple[float, ...] = STEPS_MS

    # ---- level roves (uniform, half-ranges in dB) ----------------------------
    rove_db: float = 12.0                 # d (target offset), v (A probe), w (B flankers)
    interval_rove_db: float = 0.0         # c: one level for a whole interval
    flanker_jitter_db: float = 0.0        # u: B probe only, independent per flanker

    # ---- the tracked variable (SRS, dB) --------------------------------------
    srs_start_db: float = 4.0             # dL 8.3 dB
    srs_min_db: float = -30.0             # dL 0.27 dB
    srs_max_db: float = 9.5               # dL 12.0 dB
    step_sizes_db: Tuple[float, ...] = (4.0, 2.0)
    reversals_per_step: Tuple[int, ...] = (1,)
    n_final_reversals: int = 4
    max_trials_per_track: int = 50
    n_down: int = 3                       # 3-down 1-up: 79.4% correct
    tracks_per_condition: int = 1

    # ---- timing --------------------------------------------------------------
    lead_ms: float = 100.0
    tail_ms: float = 100.0
    isi_ms: float = 400.0

    # ---- controls ------------------------------------------------------------
    include_capture: bool = True          # A and B at step 0 with flanker captors
    include_alone: bool = True            # B with the target alone: no flankers at all
    captor_phase: float = 0.5             # captor tokens at this fraction of the period

    # ---- catch trials --------------------------------------------------------
    catch_rate: float = 0.06
    catch_delta_db: float = 8.0           # no rove, step 0: both cues intact
    max_catch_miss_rate: float = 0.15

    # ---- procedure -----------------------------------------------------------
    feedback: bool = True
    familiarise: bool = True
    practice_trials: int = 12
    practice_criterion: float = 0.75
    practice_srs_db: float = 6.0          # dL 9.9 dB
    practice_rounds: int = 3
    response_timeout_s: float = 6.0
    break_every_trials: Optional[int] = 60
    max_same_condition_run: int = 2
    max_same_task_run: int = 3
    fillers: bool = True                  # keep the order unpredictable as tracks finish

    # ---- analysis ------------------------------------------------------------
    near_ceiling_db: float = 2.0          # a threshold this close to the ceiling is not a
                                          # measurement of the surviving cue
    n_boot: int = 4000

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
        for name in ("freqs_hz", "steps_ms", "step_sizes_db", "reversals_per_step"):
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
        return self.period_ms / self.n_tones

    @property
    def target_freq_hz(self) -> float:
        return self.freqs_hz[self.target_index]

    @property
    def others(self) -> List[int]:
        return [k for k in range(self.n_tones) if k != self.target_index]

    @property
    def delta_max_db(self) -> float:
        return delta_db(self.srs_max_db)

    def interval_ms(self, step_ms: float) -> float:
        """Interval length for one step. Both intervals of a trial share it.

        It grows with the step by 3 x step, so the silence after the last tone is the same at
        every step. Padding every step to the widest would add a step-dependent silence before
        the response instead, and cost 0.4 s per trial.
        """
        return (self.lead_ms + (self.n_repeats - 1) * self.period_ms
                + (self.n_tones - 1) * float(step_ms) + self.tone_ms + self.tail_ms)

    def trial_ms(self, step_ms: float) -> float:
        return 2 * self.interval_ms(step_ms) + self.isi_ms


DEFAULT = Config()


# ----------------------------------------------------------------------------
def _tag(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def conditions(cfg: Config) -> Tuple[Condition, ...]:
    out = [Condition(f"{t}_{_tag(s)}", t, float(s), "figure", "main")
           for s in cfg.steps_ms for t in ("A", "B")]
    if cfg.include_capture:
        out += [Condition("A_cap", "A", 0.0, "capture", "control"),
                Condition("B_cap", "B", 0.0, "capture", "control")]
    if cfg.include_alone:
        out.append(Condition("B_alone", "B", 0.0, "alone", "control"))
    return tuple(out)


def catch_condition(cfg: Config) -> Condition:
    return Condition("catch", "none", 0.0, "figure", "catch")


def onsets(cfg: Config, cond: Condition) -> np.ndarray:
    """(n_repeats, n_tones) onsets in ms from the interval start; NaN where a tone is absent."""
    rep = np.arange(cfg.n_repeats, dtype=float)[:, None]
    k = np.arange(cfg.n_tones, dtype=float)[None, :]
    o = cfg.lead_ms + rep * cfg.period_ms + k * float(cond.step_ms)
    if cond.kind == "alone":
        o = o.copy()
        o[:, cfg.others] = np.nan
    return o


def captor_onsets(cfg: Config, cond: Condition) -> Optional[np.ndarray]:
    """(n_repeats - 1, n_tones) onsets of the captor tokens, or None.

    A captor token is the three other tones on their own, sheared like the figure, placed
    between consecutive repetitions. The other tones then repeat at twice the rate of the
    target, so the target is synchronous with every second one of them and their envelopes
    are no longer coherent with its own: segregation with no change in onset asynchrony.
    """
    if cond.kind != "capture":
        return None
    rep = np.arange(cfg.n_repeats - 1, dtype=float)[:, None]
    k = np.arange(cfg.n_tones, dtype=float)[None, :]
    o = (cfg.lead_ms + (rep + cfg.captor_phase) * cfg.period_ms + k * float(cond.step_ms))
    o[:, cfg.target_index] = np.nan
    return o


def worst_case_peak(cfg: Config) -> Dict[str, float]:
    """In-phase sum of every tone that can sound at once, each at the top of its rove.

    Assumes all four overlap, which is exact at step 0 and conservative elsewhere.
    """
    a = cfg.tone_amplitude
    c, r, u = cfg.interval_rove_db, cfg.rove_db, cfg.flanker_jitter_db
    g = lambda db: 10.0 ** (db / 20.0)
    n_oth = cfg.n_tones - 1
    dmax = cfg.delta_max_db
    out = {
        "earlier": a * (g(c + r) + n_oth * g(c)),
        "probe_A": a * g(c + r) * (n_oth + g(dmax)),
        "probe_B": a * (g(c + r + dmax) + n_oth * g(c + r + u)),
        "captor": a * n_oth * g(c),
        "catch": a * (n_oth + g(cfg.catch_delta_db)),
    }
    out["worst"] = max(out.values())
    return out


def level_range_db_spl(cfg: Config) -> Tuple[float, float]:
    """Quietest and loudest single tone the session can play, re the nominal level."""
    lo = cfg.tone_level_db_spl - cfg.interval_rove_db - cfg.rove_db - cfg.flanker_jitter_db
    hi = cfg.tone_level_db_spl + cfg.interval_rove_db + cfg.rove_db + cfg.delta_max_db
    return lo, hi


@dataclass(frozen=True)
class Derived:
    conditions: Tuple[Condition, ...]
    n_tracks: int
    period_ms: float
    trial_ms: Dict[float, float]
    min_erbs: float
    min_ratio_mistuning_pct: float
    peak: Dict[str, float]
    level_range_db_spl: Tuple[float, float]
    notes: Tuple[str, ...]


def validate(cfg: Config) -> Derived:
    notes: List[str] = []
    if cfg.n_tones < 3:
        raise ConfigError("the figure needs at least three tones: a target and two others")
    if not 0 < cfg.target_index < cfg.n_tones - 1:
        raise ConfigError("the target must be an interior tone: an edge tone's onset is also "
                          "the onset or the end of the whole figure")
    if cfg.n_repeats < 3:
        raise ConfigError("n_repeats must be at least 3: two earlier repetitions to establish "
                          "the figure and the target's own level, then the probe")
    if any(s < 0 for s in cfg.steps_ms) or not cfg.steps_ms:
        raise ConfigError("steps_ms must be non-negative and non-empty")
    if 0.0 not in cfg.steps_ms:
        notes.append("step 0 is not sampled, so there is no synchronous anchor for the binding "
                     "index and the capture control has nothing to be compared with")
    span = (cfg.n_tones - 1) * max(cfg.steps_ms) + cfg.tone_ms
    if span >= cfg.period_ms:
        raise ConfigError(f"at the largest step the figure spans {span:.0f} ms, which does not "
                          f"fit in the {cfg.period_ms:.0f} ms period")
    iso = [s for s in cfg.steps_ms if abs(s - cfg.iso_step_ms) < 0.1 * cfg.iso_step_ms]
    if iso:
        notes.append(f"step {iso} ms is within 10% of {cfg.iso_step_ms:.1f} ms, where the "
                     "combined onset train is isochronous; that rhythm exists at no other step")
    if cfg.include_capture:
        gap = cfg.captor_phase * cfg.period_ms - cfg.tone_ms
        gap2 = (1 - cfg.captor_phase) * cfg.period_ms - cfg.tone_ms
        if min(gap, gap2) < 40.0:
            raise ConfigError("captor tokens come within 40 ms of the figure's own tones in the "
                              "same channel; that is forward masking, not capture")

    # tracked variable and its schedule
    if not cfg.srs_min_db < cfg.srs_start_db <= cfg.srs_max_db:
        raise ConfigError("need srs_min_db < srs_start_db <= srs_max_db")
    if len(cfg.step_sizes_db) != len(cfg.reversals_per_step) + 1:
        raise ConfigError("reversals_per_step must have one entry fewer than step_sizes_db")
    if cfg.n_final_reversals % 2:
        notes.append("an odd number of final reversals averages one more peak than valleys "
                     "(or the reverse), which biases the threshold by half a step")

    # the tones are separate channels and do not fuse by harmonicity
    from tcoh.config import _common_f0, _simple_ratio_mistuning
    from tcoh.model import channel_crosstalk
    f = sorted(cfg.freqs_hz)
    if len(set(f)) != len(f):
        raise ConfigError("freqs_hz contains a repeated frequency")
    erbs = min(channel_crosstalk(x, y)["erbs"] for x, y in itertools.combinations(f, 2))
    mis = min(_simple_ratio_mistuning(x, y)[0] for x, y in itertools.combinations(f, 2))
    if erbs < 3.0:
        raise ConfigError(f"the closest pair of tones is {erbs:.1f} ERB apart; below about 3 "
                          "they share an auditory filter and are not separate components")
    if _common_f0(f) is not None:
        raise ConfigError("the tones fit a low-order harmonic series and would fuse by "
                          "harmonicity, which this design does not manipulate")

    # levels
    peak = worst_case_peak(cfg)
    if peak["worst"] >= 0.98:
        raise ConfigError(f"the worst case reaches {peak['worst']:.2f} FS and would clip. "
                          "Lower tone_amplitude and raise the system volume instead.")
    lo, hi = level_range_db_spl(cfg)
    if hi > 85.0:
        notes.append(f"the loudest tone can reach {hi:.0f} dB SPL at the nominal calibration")
    if lo < 30.0:
        notes.append(f"the quietest tone can fall to {lo:.0f} dB SPL at the nominal calibration")
    if cfg.rove_db < 10.0:
        notes.append(f"a rove of {cfg.rove_db:g} dB leaves the destroyed cues usable at about "
                     f"{0.71 * cfg.rove_db:.0f} dB; the design wants that well above threshold")

    conds = conditions(cfg)
    steps = sorted({c.step_ms for c in conds})
    return Derived(
        conditions=conds, n_tracks=len(conds) * cfg.tracks_per_condition,
        period_ms=cfg.period_ms, trial_ms={s: cfg.trial_ms(s) for s in steps},
        min_erbs=float(erbs), min_ratio_mistuning_pct=float(mis), peak=peak,
        level_range_db_spl=(lo, hi), notes=tuple(notes))


def describe(cfg: Config) -> str:
    d = validate(cfg)
    lo, hi = d.level_range_db_spl
    L = [f"binding bound in a sheared four-tone figure  |  config {cfg.hash()}",
         "  tones      " + " + ".join(f"{x:.0f}" for x in sorted(cfg.freqs_hz)) + " Hz"
         f"   ({d.min_erbs:.1f} ERB apart at the closest, "
         f"{d.min_ratio_mistuning_pct:.0f}% from the nearest simple ratio)",
         f"             {cfg.tone_ms:g} ms tones, {cfg.ramp_ms:g} ms ramps, repeating at "
         f"{cfg.rate_hz:g} Hz, {cfg.n_repeats} repetitions; the last is the probe",
         f"  target     tone {cfg.target_index + 1} of {cfg.n_tones} ({cfg.target_freq_hz:.0f} Hz)"
         ", the same on every trial",
         f"  shear      steps {', '.join(f'{s:g}' for s in cfg.steps_ms)} ms between adjacent "
         "tones",
         f"  roves      d, v, w uniform +/-{cfg.rove_db:g} dB; interval rove "
         f"+/-{cfg.interval_rove_db:g} dB; flanker jitter +/-{cfg.flanker_jitter_db:g} dB",
         f"  tracking   SRS from {cfg.srs_start_db:+g} dB (dL {delta_db(cfg.srs_start_db):.1f} "
         f"dB), steps {'/'.join(f'{s:g}' for s in cfg.step_sizes_db)} dB, 3-down 1-up, "
         f"{cfg.n_final_reversals} final reversals, cap {cfg.srs_max_db:+g} dB "
         f"(dL {cfg.delta_max_db:.1f} dB)",
         f"  levels     one tone {cfg.tone_level_db_spl:g} dB SPL nominal, range {lo:.0f} to "
         f"{hi:.0f} dB SPL; worst-case peak {d.peak['worst']:.2f} FS",
         f"  trial      {min(d.trial_ms.values()) / 1000:.2f} to "
         f"{max(d.trial_ms.values()) / 1000:.2f} s of sound",
         "",
         f"  {'condition':<10} {'task':>4} {'step ms':>8} {'kind':>8}  role"]
    for c in d.conditions:
        L.append(f"  {c.name:<10} {c.task:>4} {c.step_ms:8g} {c.kind:>8}  {c.role}")
    cc = catch_condition(cfg)
    L.append(f"  {'catch':<10} {cc.task:>4} {cc.step_ms:8g} {cc.kind:>8}  "
             f"{cfg.catch_rate:.0%} of trials, dL {cfg.catch_delta_db:g} dB, no rove")
    for n in d.notes:
        L.append(f"\n  note: {n}")
    return "\n".join(L)
