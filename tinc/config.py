"""tinc: the level-increment task of Micheyl et al. (2013), with the opposite role for grouping.

Micheyl, Hanson, Demany, Shamma & Oxenham (2013), "Auditory stream segregation for alternating
and synchronous tones", JEP:HPP 39:1568 (doi 10.1037/a0032241), Experiments 2 and 3:

    twelve B tones (1000 Hz, the target stream) and twelve A tones (15 semitones lower,
    420.45 Hz, the distractor), 100 ms each with 20-ms raised-cosine ramps, one tone every 200 ms
    per stream (5 Hz); A starts D ms after B, D = 0, 35, 50, 71, 100 ms (Exp. 3); 60 dB SPL;
    one B tone, never one of the first four, 4 dB louder on half the trials; the A tones roved
    +-4 dB (uniform) and one or two of them 4 dB louder on top. Yes/no ("was there a level change
    in the B tones?"), feedback, d' per condition. Their d' rose with D (about 1.5 at 0 ms, 2.4
    at 100 ms): ignoring A is easier once A and B segregate.

This task keeps every one of those numbers and changes one thing: the A tones are roved TOGETHER
with the B tones, tone pair by tone pair (`levels = "common"`), and the one or two extra 4-dB
steps raise both tones of a pair. A and B then rise and fall together; only the target breaks
that, by raising one B tone alone. The B tones on their own are as unpredictable as Micheyl's A
tones were, so following B alone is the losing strategy; comparing each B with its A wins. That
comparison is what grouping supports, so d' should FALL with D: the mirror image of their curve.

`levels = "independent"` is their condition exactly (B fixed, A roved alone, decoys on A), for a
replication in the same listeners and code.

One sequence per trial (single interval). The listener answers yes or no: did one high tone get
louder on its own? Delays are blocked (blocks of 52 trials, as theirs were 51-52), each delay in
two blocks, the second round in reverse order.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from typing import Dict, List, Tuple


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    name: str = "tinc_main"

    # ---- audio ---------------------------------------------------------------------------
    sample_rate: int = 48000
    channels: int = 2                      # diotic
    device_match: str = "Headphones"
    blocksize: int = 256
    latency: str = "low"

    # ---- tones (Micheyl et al. 2013, Exp. 3) ---------------------------------------------
    f_b_hz: float = 1000.0
    df_semitones: float = 15.0             # A is this far below B: 420.45 Hz
    tone_ms: float = 100.0
    ramp_ms: float = 20.0
    soa_ms: float = 200.0                  # 5 Hz per stream
    n_tones: int = 12                      # per stream
    delays_ms: Tuple[float, ...] = (0.0, 35.0, 50.0, 71.0, 100.0)   # A starts D ms after B

    # ---- levels (dB SPL) -----------------------------------------------------------------
    levels: str = "common"                 # common: A and B roved together (this task);
                                           # independent: A roved alone, B fixed (Micheyl)
    level_db: float = 60.0
    rove_db: float = 4.0                   # uniform +-, per tone pair (common) or per A tone
    decoy_db: float = 4.0                  # added to one or two pairs (common) or A tones
    decoy_counts: Tuple[int, ...] = (1, 2)
    target_db: float = 4.0                 # the change: one B tone this much louder
    target_from: int = 4                   # 0-based: the first four B tones never carry it

    # ---- procedure -----------------------------------------------------------------------
    blocks_per_delay: int = 2
    trials_per_block: int = 52
    examples: Tuple[Tuple, ...] = ((0.0, 1), (0.0, 0), (100.0, 1), (100.0, 0))
    practice: Tuple[Tuple, ...] = ((0.0, 10), (100.0, 10))   # (delay, trials[, change dB])
    feedback: bool = True

    # ---- timing (ms) -----------------------------------------------------------------------
    pre_ms: float = 40.0                   # silence before the first tone in the buffer
    post_ms: float = 80.0
    ready_ms: int = 400
    feedback_ms: int = 500
    blank_ms: int = 600
    max_peak: float = 0.95
    uncal_peak_60db: float = 0.02          # sine peak that stands for 60 dB SPL uncalibrated

    @property
    def f_a_hz(self) -> float:
        return self.f_b_hz * 2.0 ** (-self.df_semitones / 12.0)

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

    @property
    def n_trials(self) -> int:
        return len(self.delays_ms) * self.blocks_per_delay * self.trials_per_block

    def sequence_ms(self, delay: float) -> float:
        return (self.n_tones - 1) * self.soa_ms + delay + self.tone_ms


def validate(cfg: Config) -> Config:
    if cfg.levels not in ("common", "independent"):
        raise ConfigError("levels must be common or independent")
    if 2 * cfg.ramp_ms > cfg.tone_ms or cfg.tone_ms >= cfg.soa_ms:
        raise ConfigError("tones must fit their ramps and be shorter than the onset interval")
    if any(d < 0 or d > cfg.soa_ms - cfg.tone_ms + 1e-9 for d in cfg.delays_ms):
        raise ConfigError("delays must lie between 0 and soa - tone, so an A tone never runs "
                          "into the next B tone")
    if not 0 <= cfg.target_from < cfg.n_tones:
        raise ConfigError("target_from must index a B tone")
    if cfg.trials_per_block % 2:
        raise ConfigError("trials_per_block must be even: half the trials carry the change")
    if any(k < 0 or k > cfg.n_tones for k in cfg.decoy_counts):
        raise ConfigError("decoy_counts out of range")
    for d, t in cfg.examples:
        if d not in cfg.delays_ms or t not in (0, 1):
            raise ConfigError("examples are (delay from delays_ms, 0 or 1)")
    for item in cfg.practice:
        d, n = item[0], item[1]
        if d not in cfg.delays_ms or n < 1 or len(item) not in (2, 3):
            raise ConfigError("practice items are (delay from delays_ms, trials[, change dB])")
    change = max([cfg.target_db] + [float(it[2]) for it in cfg.practice if len(it) == 3])
    a_db = cfg.level_db + cfg.rove_db + cfg.decoy_db           # only B carries the change
    b_db = a_db + change
    amp = lambda db: cfg.uncal_peak_60db * 10 ** ((db - 60.0) / 20)
    if amp(a_db) + amp(b_db) > cfg.max_peak:
        raise ConfigError("the loudest possible pair would exceed max_peak uncalibrated")
    return cfg
