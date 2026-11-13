"""taba: A(AB)A triplets, with B delayed by D inside the middle sound.

The sequence repeats  A  [A+B]  A  -  : a low tone A in three slots, a high tone B added to the
middle slot, and a silent fourth slot (the ABA- format of van Noorden 1975). B starts D ms after
the middle A. The flanking A tones are at the SAME frequency as the A inside the middle sound, so
they pull it out of the chord (sequential capture; Bregman & Pinker 1978); the onset delay D
weakens the chord's own cohesion (onset synchrony; Darwin & Ciocca 1992). The listener reports,
continuously, whether they hear

    TOGETHER (left key):  the high beep is part of the middle sound (one rich sound)
    SEPARATE (right key): the high beep stands out as a sound of its own

Both percepts can alternate during a long sequence (bistability; Pressnitzer & Hupe 2006). The
dependent variables per condition are the proportion of time heard as separate, the first
percept, the latency of the first switch and the switch rate.

Controls, each at D = 0, 35, 100 ms so the flank x D interaction can be estimated:
  dyad_only   the middle sound alone (no flanking A): how much B and A fuse without capture
              (the report version of Micheyl et al. 2013 Exp. 3, same tones and D values)
  offfreq     flanking tones 9 st below A, as the remote captors of Micheyl, Hunter & Oxenham
              (2010): capture is weak, so a difference from the main condition at the same D is
              attributable to same-frequency capture
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from typing import Dict, List, Tuple


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Condition:
    name: str
    d_ms: float                 # B onset minus middle-A onset (positive: B late)
    flank: str = "same"         # same | none | off


@dataclass(frozen=True)
class Config:
    name: str = "taba_main"
    sample_rate: int = 48000
    channels: int = 2
    device_match: str = "Headphones"
    blocksize: int = 256
    latency: str = "low"

    f_b_hz: float = 1000.0
    df_semitones: float = 15.0          # A is this far below B (Micheyl et al. 2013)
    off_semitones: float = 9.0          # off-frequency flanks: this far below A
    tone_ms: float = 100.0
    ramp_ms: float = 20.0
    soa_ms: float = 200.0               # slot to slot; a triplet lasts 4 slots
    level_db: float = 60.0
    amplitude: float = 0.08             # per tone at level_db when uncalibrated

    d_ms: Tuple[float, ...] = (0.0, 20.0, 35.0, 50.0, 71.0, 100.0)
    controls: Tuple[str, ...] = ("dyad_only_0", "dyad_only_35", "dyad_only_100",
                                 "offfreq_0", "offfreq_35", "offfreq_100")
    reps: int = 2                       # sequences per main condition
    control_reps: int = 1
    trial_s: float = 30.0               # length of each sequence
    practice_s: float = 15.0
    rest_s: float = 4.0                 # silence between sequences
    long_rest_every: int = 5

    @property
    def f_a_hz(self) -> float:
        return self.f_b_hz * 2.0 ** (-self.df_semitones / 12.0)

    @property
    def f_off_hz(self) -> float:
        return self.f_a_hz * 2.0 ** (-self.off_semitones / 12.0)

    @property
    def period_ms(self) -> float:
        return 4.0 * self.soa_ms

    def conditions(self) -> List[Condition]:
        out = [Condition(f"D{d:g}", float(d)) for d in self.d_ms]
        for c in self.controls:
            kind, d = c.rsplit("_", 1)
            out.append(Condition(c, float(d), {"dyad_only": "none", "offfreq": "off"}[kind]))
        return out

    def condition(self, name: str) -> Condition:
        for c in self.conditions():
            if c.name == name:
                return c
        raise KeyError(name)

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
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})

    @classmethod
    def load(cls, path) -> "Config":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def validate(cfg: Config) -> Config:
    if cfg.ramp_ms * 2 > cfg.tone_ms:
        raise ConfigError("ramps longer than the tone")
    if cfg.tone_ms >= cfg.soa_ms:
        raise ConfigError("tones must be shorter than the slot so the A tones do not touch")
    if any(abs(d) > cfg.soa_ms for d in cfg.d_ms):
        raise ConfigError("|D| above one slot would move B into a neighbouring slot")
    if 0.0 not in cfg.d_ms:
        raise ConfigError("D = 0 is needed (practice and the reference condition)")
    if cfg.trial_s < 4 * cfg.period_ms / 1000.0:
        raise ConfigError("a sequence needs at least four triplets")
    cfg.conditions()
    return cfg
