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
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from typing import Dict, List, Tuple

ARRANGEMENTS: Dict[str, Tuple[int, ...]] = {
    "sync": (0, 0, 0, 0, 0),
    "ascending": (-2, -1, 0, 1, 2),
    "zigzag": (2, -1, 0, 1, -2),
}


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
    burst_ms: float = 187.5
    ramp_ms: float = 20.0
    noi_ms: float = 250.0
    n_precursors: int = 4
    tone_hz: float = 1000.0
    tone_bands: Tuple[int, ...] = ()      # if set, the tone's band is drawn at random per trial
                                          # from these band indices (e.g. (2, 3): 1 or 2 kHz)
    isi_ms: float = 400.0                 # silence between the three intervals
    min_gap_ms: float = 15.0              # smallest silence before the final burst in any band

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
        return [s * dt_ms for s in ARRANGEMENTS[arrangement]]

    @property
    def n_tracks(self) -> int:
        return sum(c.tracks for c in self.condition_list())

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]

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
    for c in cfg.condition_list():
        if c.arrangement not in ARRANGEMENTS:
            raise ConfigError(f"unknown arrangement {c.arrangement}")
        if c.envelope not in ("co", "rand", "ref"):
            raise ConfigError("envelope must be co, rand or ref (the tone's band alone)")
        for o in cfg.offsets_ms(c.arrangement, c.dt_ms):
            gap = cfg.noi_ms - o - cfg.burst_ms
            if gap < cfg.min_gap_ms:
                raise ConfigError(f"{c.name}: a band's last precursor ends {gap:.1f} ms before the "
                                  f"final burst (minimum {cfg.min_gap_ms:g}); lower dT")
    for a, d in cfg.ratings:
        if a not in ARRANGEMENTS:
            raise ConfigError(f"unknown rating arrangement {a}")
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
