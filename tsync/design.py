"""Who hears what, in which order, with which random numbers.

Order: one adaptive track per condition, the six conditions in the order given by one row of a
Williams Latin square (Williams 1949). Across the six rows every condition appears once in every
position and every condition follows every other exactly once, so position (practice, fatigue)
and first-order carry-over are balanced across listeners. Twelve listeners use each row twice.

Seeds: one 63-bit session seed, drawn fresh for every session and logged. Every trial's
generator is seeded with (session seed, phase, block, trial, attempt), so any trial can be
rebuilt exactly and nothing about one trial depends on what happened before it.
"""
from __future__ import annotations

import re
import secrets
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import Config

PHASE_CODE = {"check": 1, "demo": 2, "practice": 3, "exposure": 4, "warmup": 5, "main": 6}

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def clean_code(code: str) -> str:
    """A participant code in ASCII: Persian and Arabic-Indic digits mapped, anything else refused.

    The tcoh data already hold codes like '۰P01' typed with the wrong keyboard layout.
    """
    c = code.strip().translate(_DIGITS)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", c):
        raise ValueError(f"participant code {code!r} must be letters, digits, '-' or '_' only")
    return c.upper()


def participant_number(code: str) -> int:
    """P07 -> 7. Codes without digits get a stable number from their characters."""
    m = re.findall(r"\d+", code)
    if m:
        return int(m[-1])
    return sum(ord(ch) for ch in code) % 997 + 1


def williams_square(n: int) -> List[List[int]]:
    """Rows of a Williams design: n rows for even n, 2n (the square and its mirror) for odd n."""
    first, lo, hi = [0], 1, n - 1
    take_lo = True
    while len(first) < n:
        first.append(lo if take_lo else hi)
        if take_lo:
            lo += 1
        else:
            hi -= 1
        take_lo = not take_lo
    rows = [[(x + r) % n for x in first] for r in range(n)]
    if n % 2:
        rows += [list(reversed(r)) for r in rows]
    return rows


@dataclass
class Plan:
    session_seed: int
    participant_number: int
    order_row: int
    blocks: List[str]                         # condition of each main block, in run order
    catch_slots: List[List[int]]              # per block: staircase trial indices preceded by a catch
    exposure: List[str]                       # condition of each exposure trial
    conditions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Plan":
        return cls(**d)


def make_plan(cfg: Config, participant: int, session_seed: int | None = None,
              order_row: int | None = None) -> Plan:
    names = [c.name for c in cfg.conditions()]
    rows = williams_square(len(names))
    row = (participant - 1) % len(rows) if order_row is None else int(order_row) % len(rows)
    blocks: List[str] = []
    for rep in range(cfg.tracks_per_condition):
        blocks += [names[i] for i in rows[(row + rep) % len(rows)]]
    seed = int(session_seed) if session_seed is not None else secrets.randbits(63)
    rng = np.random.default_rng([seed, 99])
    lo, hi = cfg.catch_positions
    slots = [sorted(int(k) for k in rng.choice(np.arange(lo, hi + 1), cfg.catch_per_track,
                                                replace=False)) for _ in blocks]
    exposure = [c for c in cfg.exposure_conditions for _ in range(cfg.exposure_per_condition)]
    exposure = [exposure[i] for i in rng.permutation(len(exposure))]
    return Plan(seed, participant, row, blocks, slots, exposure, names)


def trial_seed(plan: Plan, phase: str, block: int, k: int, attempt: int = 0) -> Tuple[int, ...]:
    return (plan.session_seed, PHASE_CODE[phase], int(block), int(k), int(attempt))


def balance_report(rows: Sequence[Sequence[int]]) -> Dict[str, bool]:
    """Is every condition in every position once, and every ordered pair adjacent once?"""
    n = len(rows[0])
    reps = len(rows) // n
    pos_ok = all(sorted(r[p] for r in rows) == sorted(list(range(n)) * reps) for p in range(n))
    pairs: Dict[Tuple[int, int], int] = {}
    for r in rows:
        for a, b in zip(r, r[1:]):
            pairs[(a, b)] = pairs.get((a, b), 0) + 1
    want = {(a, b) for a in range(n) for b in range(n) if a != b}
    carry_ok = set(pairs) == want and len(set(pairs.values())) == 1
    return {"positions_balanced": pos_ok, "carryover_balanced": carry_ok}
