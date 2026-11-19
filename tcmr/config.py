"""tcmr: comodulation masking release (CMR) after precursors sheared across frequency.

The stimulus is Christiansen & Oxenham (2014, JASA 135:3520, doi 10.1121/1.4872300), Exp. 1:
five 20-Hz noise bands (0.25, 0.5, 1, 2, 4 kHz) at 60 dB SPL each, 187.5-ms bursts with 20-ms
raised-cosine ramps, one burst every 250 ms, four precursor bursts and a final burst that is
always synchronous across bands and carries the 1-kHz tone in one of three intervals.
Comodulated bands share amplitudes and phases (identical envelopes); random bands do not.
CMR = threshold(random) - threshold(comodulated).

New here: instead of delaying only the 1-kHz band's precursors, every band's precursors are
offset, by k * dT for the band's position k in an arrangement:

    sync       (0, 0, 0, 0, 0)        no shear
    ascending  (-2, -1, 0, +1, +2)    low bands first: an ordered sweep
    zigzag     (+2, -1, 0, +1, -2)    the same offsets of the 1-kHz band to each flank at each
                                      frequency distance as ascending, but not monotonic

(bands listed low to high; the 1-kHz band is always 0, so its own timing never changes). The
final burst is brought back into synchrony by adjusting each band's last precursor-to-target
interval, as Christiansen & Oxenham did with +-dT/2.

Ascending vs zigzag at the same dT asks whether an ordered sweep holds the bands together better
than an equally asynchronous unordered pattern.

Bursts keep the natural level fluctuation of the noise (`burst_level`, see stimulus.py). Sessions
whose stored configuration has no `burst_level` were recorded with equalised bursts.

Patterns (2026-11-14, literature/tcmr_redesign_2026-11-14.md section 13). Two arrangements are not a
fixed offset list but drawn for every interval:

    frozen     a staircase of offsets (-2, -1, 0, +1, +2) x dT with the bands shuffled over it: the
               tone band keeps 0, the four others take -2, -1, +1, +2 in one of 24 orders. The order
               is the same in all precursors of an interval and differs between the three intervals
               of a trial (drawn without replacement), so how much of the pattern could be learned
               never depends on which interval holds the tone. Over a track the orders are
               balanced: each of the 24 once in every 8 trials.
    reordered  the same offsets with a different order in every repetition: nothing repeats.
               Consecutive bursts of one band keep at least `min_gap_ms` of silence (at dT = 40 ms a
               band may not jump from +2 to -2).

Practice gate (2026-11-15): with `practice_min_correct` set, a listener who gets fewer practice
trials right than that is stopped after the practice; the main part never starts (session.py).

The noise is never frozen: every burst is new noise at its natural level. With `lead_ms` set, every
interval opens `lead_ms` before the tone band's first burst whatever the condition, so the tone band
sits at the same times in every trial and every trial has the same length.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from itertools import permutations
from typing import Dict, List, Optional, Tuple

ARRANGEMENTS: Dict[str, Tuple[int, ...]] = {
    "sync": (0, 0, 0, 0, 0),
    "ascending": (-2, -1, 0, 1, 2),
    "zigzag": (2, -1, 0, 1, -2),
}
PATTERNS = ("frozen", "reordered")        # drawn per interval (frozen) or per repetition
FLANK_SLOTS = (-2, -1, 1, 2)
NEW_DEFAULTS = {"lead_ms": None,           # fields added after sessions were recorded: left out of
                "practice_min_correct": 0}  # the hash at these values, so old hashes stay the same


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Condition:
    envelope: str        # co | rand
    arrangement: str
    dt_ms: float
    tracks: int = 1

    @property
    def name(self) -> str:
        return f"{self.envelope}_{self.arrangement}_{self.dt_ms:g}"


@dataclass(frozen=True)
class Config:
    name: str = "tcmr_main"
    sample_rate: int = 48000
    channels: int = 2
    device_match: str = "Headphones"
    blocksize: int = 256
    latency: str = "low"

    # ---- stimulus (Christiansen & Oxenham 2014) ---------------------------------------------
    bands_hz: Tuple[float, ...] = (250.0, 500.0, 1000.0, 2000.0, 4000.0)
    target_band: int = 2
    bandwidth_hz: float = 20.0
    band_db: float = 60.0
    burst_level: str = "natural"          # natural: a burst is a stretch of noise at the long-term
                                          # band level, so its own level varies from burst to burst
                                          # (together across comodulated bands). equalised: every
                                          # burst rescaled to exactly the band level, as before
                                          # 2026-11-14 (removes that cue; see stimulus.py)
    burst_ms: float = 187.5
    ramp_ms: float = 20.0
    noi_ms: float = 250.0
    n_precursors: int = 4
    tone_hz: float = 1000.0
    tone_bands: Tuple[int, ...] = ()      # if set, the tone's band is drawn at random per trial
                                          # from these band indices (e.g. (2, 3): 1 or 2 kHz)
    isi_ms: float = 400.0                 # silence between the three intervals
    min_gap_ms: float = 15.0              # smallest silence between two bursts of one band
    lead_ms: Optional[float] = None       # fixed: every interval opens this long before the tone
                                          # band's first burst (None: the earliest band's onset)

    # ---- conditions -------------------------------------------------------------------------
    conditions: Tuple[Tuple, ...] = (
        ("co", "sync", 0.0, 1), ("rand", "sync", 0.0, 1),
        ("co", "ascending", 10.0, 1),
        ("co", "ascending", 20.0, 2), ("co", "zigzag", 20.0, 2),
        ("rand", "ascending", 20.0, 1),
    )

    # ---- adaptive tracks (Christiansen & Oxenham: 2-down 1-up, 8/4/2 dB) ---------------------
    start_db: float = 75.0
    min_db: float = 20.0
    max_db: float = 85.0
    steps_db: Tuple[float, ...] = (8.0, 4.0, 2.0)
    step_change_after: Tuple[int, ...] = (2, 4)    # reversals after which the step shrinks
    n_down: int = 2
    trials_per_track: int = 36
    lapse: float = 0.02

    # ---- practice and ratings ------------------------------------------------------------------
    practice_levels_db: Tuple[float, ...] = (80.0, 80.0, 78.0, 78.0, 75.0, 75.0, 72.0, 72.0)
    practice_min_correct: int = 0         # the practice gate: with fewer correct practice trials
                                          # than this the session stops before the main part
                                          # (0: no gate, as in every session before 2026-11-15)
    ratings: Tuple[Tuple, ...] = (("sync", 0.0), ("ascending", 10.0), ("ascending", 20.0),
                                  ("zigzag", 10.0), ("zigzag", 20.0))
    rating_reps: int = 2

    # ---- screen timing (ms) -------------------------------------------------------------------
    ready_ms: int = 400
    feedback_ms: int = 500
    blank_ms: int = 600
    break_every_rounds: int = 9            # one round = one trial of every track

    # ---- levels without a calibration --------------------------------------------------------
    uncal_peak_60db: float = 0.02          # sine peak that stands for 60 dB SPL

    # -- derived --------------------------------------------------------------------------------
    def condition_list(self) -> List[Condition]:
        return [Condition(e, a, float(d), int(n)) for e, a, d, n in self.conditions]

    def offsets_ms(self, arrangement: str, dt_ms: float) -> List[float]:
        """Offsets by band for a fixed arrangement. A pattern has no fixed assignment: this is its
        staircase before the bands are shuffled (the set of offsets every interval uses)."""
        if arrangement in PATTERNS:
            return [s * dt_ms for s in ARRANGEMENTS["ascending"]]
        return [s * dt_ms for s in ARRANGEMENTS[arrangement]]

    def orders(self) -> List[Tuple[int, ...]]:
        """The 24 orders: slot (-2..2) by band, the tone band at 0."""
        flank = [b for b in range(len(self.bands_hz)) if b != self.target_band]
        out = []
        for perm in permutations(FLANK_SLOTS):
            o = [0] * len(self.bands_hz)
            for b, sl in zip(flank, perm):
                o[b] = sl
            out.append(tuple(o))
        return out

    def burst_gap_ms(self, slot_from: int, slot_to: int, dt_ms: float) -> float:
        """Silence between a band's burst at slot_from and its next burst at slot_to, one onset
        interval later."""
        return self.noi_ms + (slot_to - slot_from) * dt_ms - self.burst_ms

    @property
    def n_tracks(self) -> int:
        return sum(c.tracks for c in self.condition_list())

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        d = {k: v for k, v in self.to_dict().items()
             if not (k in NEW_DEFAULTS and v == NEW_DEFAULTS[k])}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    def replace(self, **kw) -> "Config":
        return replace(self, **kw)

    @classmethod
    def from_dict(cls, d: Dict) -> "Config":
        names = {f.name for f in fields(cls)}
        bad = set(d) - names
        if bad:
            raise ConfigError(f"unknown keys {sorted(bad)}")

        def tup(v):
            return tuple(tup(x) for x in v) if isinstance(v, list) else v
        return cls(**{k: tup(v) for k, v in d.items()})

    @classmethod
    def load(cls, path) -> "Config":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def validate(cfg: Config) -> Config:
    if len(cfg.bands_hz) != 5:
        raise ConfigError("the arrangements are defined for five bands")
    if cfg.ramp_ms * 2 > cfg.burst_ms or cfg.burst_ms >= cfg.noi_ms:
        raise ConfigError("bursts must fit their ramps and be shorter than the onset interval")
    patterned = [c for c in cfg.condition_list() if c.arrangement in PATTERNS]
    patterned += [Condition("co", a, float(d)) for a, d in cfg.ratings if a in PATTERNS]
    for c in cfg.condition_list():
        if c.arrangement not in ARRANGEMENTS and c.arrangement not in PATTERNS:
            raise ConfigError(f"unknown arrangement {c.arrangement}")
        if c.envelope not in ("co", "rand", "ref"):
            raise ConfigError("envelope must be co, rand or ref (the tone's band alone)")
        for o in cfg.offsets_ms(c.arrangement, c.dt_ms):
            gap = cfg.noi_ms - o - cfg.burst_ms
            if gap < cfg.min_gap_ms:
                raise ConfigError(f"{c.name}: a band's last precursor ends {gap:.1f} ms before the "
                                  f"final burst (minimum {cfg.min_gap_ms:g}); lower dT")
    for a, d in cfg.ratings:
        if a not in ARRANGEMENTS and a not in PATTERNS:
            raise ConfigError(f"unknown rating arrangement {a}")
    if patterned:
        if cfg.tone_bands:
            raise ConfigError("patterns keep the tone band at offset 0: the tone's band cannot be "
                              "randomised with them")
        if cfg.lead_ms is None:
            raise ConfigError("patterns need lead_ms, so the tone band sits at the same times in "
                              "every condition")
        if cfg.burst_gap_ms(0, 0, 0.0) < cfg.min_gap_ms:
            raise ConfigError("bursts are too long for the onset interval")
        for c in patterned:
            if c.envelope == "ref":
                raise ConfigError("band alone has no flanking bands to arrange: use sync")
            if c.arrangement == "reordered":
                ok = [(a, b) for a in (-2, -1, 0, 1, 2) for b in (-2, -1, 0, 1, 2)
                      if cfg.burst_gap_ms(a, b, c.dt_ms) >= cfg.min_gap_ms]
                if not all(any(x == a for x, _ in ok) for a in FLANK_SLOTS):
                    raise ConfigError(f"{c.name}: some slot has no allowed next slot")
    if cfg.lead_ms is not None:
        need = max([2 * c.dt_ms for c in cfg.condition_list()] +
                   [2 * float(d) for a, d in cfg.ratings if a != "sync"] + [0.0])
        if cfg.lead_ms < need:
            raise ConfigError(f"lead_ms = {cfg.lead_ms:g} ms is shorter than the earliest band's "
                              f"{need:g} ms lead")
    if cfg.burst_level not in ("natural", "equalised"):
        raise ConfigError("burst_level must be natural or equalised")
    if not 0 <= cfg.practice_min_correct <= len(cfg.practice_levels_db):
        raise ConfigError("practice_min_correct must lie between 0 and the number of practice trials")
    if not cfg.min_db <= cfg.start_db <= cfg.max_db:
        raise ConfigError("start level outside the level range")
    if len(cfg.steps_db) != len(cfg.step_change_after) + 1:
        raise ConfigError("steps_db needs one more entry than step_change_after")
    if any(not 0 <= b < len(cfg.bands_hz) for b in cfg.tone_bands):
        raise ConfigError("tone_bands must index bands_hz")
    if not cfg.tone_bands and (ARRANGEMENTS["ascending"][cfg.target_band] != 0
                               or ARRANGEMENTS["zigzag"][cfg.target_band] != 0):
        raise ConfigError("the target band must keep offset 0")
    return cfg
