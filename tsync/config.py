"""Every number the experiment uses, in one frozen, hashable object.

The measure
-----------
Two isochronous pure-tone sequences, B (1000 Hz) and A (15 semitones lower, 420.45 Hz), twelve
100-ms tones each (20-ms raised-cosine ramps), one tone every 200 ms per channel: the sequences
of Micheyl, Hanson, Demany, Shamma & Oxenham (2013, JEP:HPP 39:1568), Experiment 3, including
their level statistics (A roved +-4 dB per tone, one or two A tones a further 4 dB up).

The first eleven pairs are the *context*: A starts D ms after B, D in {0, 35, 50, 71, 100} ms,
exactly as in Micheyl's Experiment 3. The twelfth pair is the *ending*, and it is the only thing
the listener judges:

* standard ending: A and B start together, at the midpoint between the two streams' slots
  (D/2 after B's slot, D/2 before A's);
* target ending:   A and B start delta apart, delta divided equally between them (one moves
  delta/2 earlier, the other delta/2 later), the order (B first or A first) random per trial.

The listener picks the interval whose ending was out of sync (2I-2AFC, 3-down 1-up).

Why the ending is re-synchronised instead of keeping the context's lag
-------------------------------------------------------------------------
If the ending kept the context's lag ("yoked"), the listener would have to tell D from D+delta.
Asynchrony discrimination gets worse as the reference asynchrony grows (a Weber effect), with or
without any change in grouping, so a rising threshold could not be attributed to segregation.
P01's tcoh curve (3.5, 8.4, 15.0, 22.4 ms at reference lags 0, 18.75, 37.5, 56.25 ms) fits a
Weber fraction near 0.4 almost exactly. Here the judgement is "synchronous versus delta" at
every D, so the reference never changes; only the context does.

Why delta is divided between A and B
-------------------------------------
Each stream's last tone moves by only delta/2, so a listener who ignores one stream and times
the other gets half the cue the A-B relation carries, and that within-stream cue also sits on a
growing pedestal (the D/2 displacement of the ending). Grouped listeners judge the A-B relation
directly; segregated listeners are left with the weaker within-stream cue. The random order
(B first or A first) stops a fixed "listen for the high one first" strategy.

The control
-----------
The resynchronised ending sits D/2 away from each stream's slot, so at D = 100 ms it is 50 ms
off both rhythms. `D0_late` asks whether that displacement alone hurts: a synchronous context
(D = 0) with the ending moved 50 ms late, exactly as far off the B rhythm as in D100. If it
behaves like D0, the rise with D comes from the context, not from the displaced ending.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields, replace
from typing import Dict, List, Tuple


class ConfigError(ValueError):
    """Raised by `validate` with a message that says what to change."""


@dataclass(frozen=True)
class Condition:
    name: str
    context_ms: float            # D: A starts this long after B in the context pairs
    ending_shift_ms: float = 0.0 # extra displacement of the ending's reference point
    role: str = "main"           # main | control
    has_a: bool = True

    @property
    def label(self) -> str:
        bits = [f"A lags B by {self.context_ms:g} ms"]
        if self.ending_shift_ms:
            bits.append(f"ending {self.ending_shift_ms:+g} ms")
        return ", ".join(bits)


@dataclass(frozen=True)
class Config:
    name: str = "tsync_main"
    version: int = 1

    # ---- audio ---------------------------------------------------------------------------
    sample_rate: int = 48000
    channels: int = 2                    # diotic: the same signal in both ears
    device_match: str = "Headphones"     # the output device must contain this (case-insensitive)
    blocksize: int = 256                 # frames per audio callback (5.3 ms at 48 kHz)
    latency: str = "low"                 # PortAudio suggested latency
    schedule_lead_ms: float = 120.0      # how far ahead of "now" a trial is scheduled

    # ---- tones (Micheyl et al. 2013, Exp. 3) -----------------------------------------------
    f_b_hz: float = 1000.0
    df_semitones: float = 15.0           # A is this far BELOW B
    tone_ms: float = 100.0
    ramp_ms: float = 20.0                # raised-cosine, onset and offset
    soa_ms: float = 200.0                # onset to onset within a channel (5 Hz)
    n_tones: int = 12                    # per channel; the last pair is the ending

    # ---- levels (dB SPL; amplitudes come from the calibration) ----------------------------
    level_b_db: float = 60.0
    level_a_db: float = 60.0
    a_rove_db: float = 4.0               # each A tone uniform in +-this
    a_decoy_db: float = 4.0              # added on top of the rove
    a_decoy_counts: Tuple[int, ...] = (1, 2)   # how many A tones get it, equiprobable
    max_peak: float = 0.9                # a rendered trial louder than this is refused

    # ---- conditions -------------------------------------------------------------------------
    context_ms: Tuple[float, ...] = (0.0, 35.0, 50.0, 71.0, 100.0)
    include_late_control: bool = True
    late_control_shift_ms: float = 50.0
    ending: str = "sync"                 # sync (pedestal-free, the design) | yoked
    delta_split_b: float = 0.5           # share of delta carried by B; the rest by A
    delta_sign: str = "random"           # random | b_first | a_first
    min_gap_ms: float = 15.0             # smallest silence allowed between tones of one stream

    # ---- trial ---------------------------------------------------------------------------
    isi_ms: Tuple[float, float] = (450.0, 550.0)   # roved silence between the two sequences
    pre_ms: float = 40.0                 # silence at the start of each trial buffer
    post_ms: float = 80.0                # silence at the end

    # ---- adaptive track (transformed up-down; Levitt 1971) ---------------------------------
    delta_start_ms: float = 20.0
    delta_min_ms: float = 0.25
    delta_max_ms: float = 60.0
    n_down: int = 3                      # 3-down 1-up -> 79.4% correct
    step_factors: Tuple[float, ...] = (4.0, 2.0, math.sqrt(2.0))
    reversals_per_step: Tuple[int, ...] = (1, 2)
    n_final_reversals: int = 6
    max_trials_per_track: int = 80
    tracks_per_condition: int = 1

    # ---- catch trials (do not move the track) ---------------------------------------------
    catch_per_track: int = 2
    catch_delta_ms: float = 60.0
    catch_positions: Tuple[int, int] = (5, 30)   # inserted before staircase trial k, k in range

    # ---- familiarisation and practice -----------------------------------------------------
    demo_delta_ms: float = 60.0
    practice_delta_ms: float = 30.0
    practice_n: int = 10
    practice_criterion: int = 8
    practice_max_attempts: int = 3
    exposure_delta_ms: float = 40.0
    exposure_conditions: Tuple[str, ...] = ("D50", "D100", "D0_late")
    exposure_per_condition: int = 2

    # ---- screen timing (ms) -------------------------------------------------------------
    ready_ms: int = 450                  # neutral screen before the first sound
    feedback: bool = True
    feedback_ms: int = 550
    blank_ms: int = 650
    reminder_after_s: float = 10.0       # a gentle prompt if no answer by then
    autopause_after_s: float = 60.0      # and a pause after this
    min_rest_s: float = 8.0              # between blocks
    long_rest_after_block: int = 3       # a longer suggested rest after this block
    long_rest_s: float = 30.0

    # ---- derived ---------------------------------------------------------------------------
    @property
    def f_a_hz(self) -> float:
        return self.f_b_hz * 2.0 ** (-self.df_semitones / 12.0)

    @property
    def target_p(self) -> float:
        return 0.5 ** (1.0 / self.n_down)

    def conditions(self) -> List[Condition]:
        out = [Condition(f"D{d:g}", float(d)) for d in self.context_ms]
        if self.include_late_control:
            out.append(Condition("D0_late", 0.0, float(self.late_control_shift_ms), "control"))
        return out

    def condition(self, name: str) -> Condition:
        for c in self.conditions():
            if c.name == name:
                return c
        raise KeyError(f"no condition {name!r} in this configuration")

    # ---- identity --------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        s = json.dumps(self.to_dict(), sort_keys=True, default=float)
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    def replace(self, **kw) -> "Config":
        return replace(self, **kw)

    @classmethod
    def from_dict(cls, d: Dict) -> "Config":
        names = {f.name: f for f in fields(cls)}
        unknown = set(d) - set(names)
        if unknown:
            raise ConfigError(f"unknown configuration keys: {sorted(unknown)}")
        kw = {}
        for k, v in d.items():
            kw[k] = tuple(v) if isinstance(v, list) else v
        return cls(**kw)

    @classmethod
    def load(cls, path) -> "Config":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def validate(cfg: Config) -> Config:
    """Refuse any configuration whose stimulus or procedure is not what it claims to be."""
    from .stimulus import ending_onsets_ms   # local: stimulus imports config

    if cfg.ending not in ("sync", "yoked"):
        raise ConfigError("ending must be 'sync' or 'yoked'")
    if cfg.delta_sign not in ("random", "b_first", "a_first"):
        raise ConfigError("delta_sign must be random, b_first or a_first")
    if not 0.0 <= cfg.delta_split_b <= 1.0:
        raise ConfigError("delta_split_b must be between 0 and 1")
    if cfg.ramp_ms * 2 > cfg.tone_ms:
        raise ConfigError("the two ramps are longer than the tone")
    if cfg.tone_ms >= cfg.soa_ms:
        raise ConfigError("tones are as long as the onset-to-onset time: no gap inside a stream")
    if cfg.n_tones < 3:
        raise ConfigError("need at least two context tones before the ending")
    if not 0 < cfg.delta_min_ms < cfg.delta_start_ms <= cfg.delta_max_ms:
        raise ConfigError("need 0 < delta_min < delta_start <= delta_max")
    if cfg.catch_delta_ms > cfg.delta_max_ms + 1e-9:
        raise ConfigError("catch_delta_ms is above delta_max_ms")
    if len(cfg.step_factors) != len(cfg.reversals_per_step) + 1:
        raise ConfigError("step_factors needs one more entry than reversals_per_step")
    if any(f <= 1.0 for f in cfg.step_factors):
        raise ConfigError("every step factor must be above 1")
    if cfg.isi_ms[0] > cfg.isi_ms[1] or cfg.isi_ms[0] < 2 * (cfg.soa_ms - cfg.tone_ms):
        raise ConfigError("the silence between the two sequences must be well above the silence "
                          "between tones inside a sequence, or the intervals run together")
    if cfg.practice_criterion > cfg.practice_n:
        raise ConfigError("practice_criterion is above practice_n")
    if not set(cfg.exposure_conditions) <= {c.name for c in cfg.conditions()}:
        raise ConfigError("exposure_conditions names a condition this design does not have")
    lo, hi = cfg.catch_positions
    if not 0 <= lo <= hi or hi - lo + 1 < cfg.catch_per_track:
        raise ConfigError("catch_positions must hold at least catch_per_track positions")

    # geometry: at every condition, both orders, delta from 0 to the ceiling, no tone may touch
    # or overlap the previous tone of its own stream, and nothing may start before t = 0.
    gap_s = cfg.min_gap_ms / 1000.0
    T = cfg.tone_ms / 1000.0
    for cond in cfg.conditions():
        for sign in (+1, -1):
            for delta in (0.0, cfg.delta_max_ms):
                b_ms, a_ms = ending_onsets_ms(cfg, cond, delta, sign)
                last_ctx = (cfg.n_tones - 2) * cfg.soa_ms
                prev_b_end = (last_ctx / 1000.0) + T
                prev_a_end = (last_ctx + cond.context_ms) / 1000.0 + T
                if b_ms / 1000.0 - prev_b_end < gap_s - 1e-9:
                    raise ConfigError(
                        f"{cond.name}: the ending's B tone comes within {cfg.min_gap_ms:g} ms of "
                        f"the previous B tone at delta={delta:g} ms. Lower delta_max_ms.")
                if cond.has_a and a_ms / 1000.0 - prev_a_end < gap_s - 1e-9:
                    raise ConfigError(
                        f"{cond.name}: the ending's A tone comes within {cfg.min_gap_ms:g} ms of "
                        f"the previous A tone at delta={delta:g} ms. Lower delta_max_ms.")
    return cfg
