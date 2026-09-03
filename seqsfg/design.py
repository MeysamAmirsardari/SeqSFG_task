"""Trial lists: balancing, ordering constraints, and seeding.

A session is seeded on (participant code, session index) together, so the same
person run twice gets different orders and different stimuli, and any trial can
be rebuilt from its recorded seed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from .config import Config, validate


@dataclass(frozen=True)
class TrialSpec:
    index: int              # position within its block
    block: str              # practice | main | control
    variant: str
    step_ms: float
    target_position: int    # 1 or 2: which interval is the recurring one
    seed: int
    practice_round: int = 0   # 1-based attempt number within a practice stage
    practice_stage: int = 0   # 1-based index into cfg.practice_cells


def session_seed(participant_code: str, session_index: int) -> int:
    h = hashlib.sha256(f"{participant_code.strip()}|{int(session_index)}".encode()).digest()
    return int.from_bytes(h[:8], "big") & 0x7FFFFFFFFFFFFFFF


def _runs_ok(items: list, constraints: Sequence[Tuple[Callable, int]]) -> bool:
    for key, max_run in constraints:
        run = 1
        for i in range(1, len(items)):
            run = run + 1 if key(items[i]) == key(items[i - 1]) else 1
            if run > max_run:
                return False
    return True


def constrained_shuffle(rng: np.random.Generator, items: list, constraints: Sequence[Tuple[Callable, int]],
                        tries: int = 40000) -> list:
    """Random order satisfying every (key, max_run) constraint at once.

    Plain rejection sampling first (which leaves the order uniform over the admissible set);
    if the constraints are tight, fall back to repairing a shuffled list by local swaps.
    """
    items = list(items)
    for _ in range(tries):
        rng.shuffle(items)
        if _runs_ok(items, constraints):
            return items
    for _ in range(200):
        rng.shuffle(items)
        for _ in range(20000):
            if _runs_ok(items, constraints):
                return items
            i, j = int(rng.integers(len(items))), int(rng.integers(len(items)))
            items[i], items[j] = items[j], items[i]
    raise RuntimeError("could not satisfy the ordering constraints: raise max_condition_run or max_variant_run, "
                       "or add trials per condition")


def _balanced(cells: Sequence[tuple], n_per_cell: int) -> List[tuple]:
    """Every cell gets n_per_cell trials with the target interval balanced within the cell."""
    out = []
    for (variant, step) in cells:
        half = n_per_cell // 2
        out += [(variant, float(step), 1)] * half + [(variant, float(step), 2)] * (n_per_cell - half)
    return out


def make_design(cfg: Config, participant_code: str, session_index: int) -> dict:
    """The main block interleaves every ladder; practice runs its stages in order.

    Trials of the two ladders are shuffled together under both constraints, so the listener
    cannot tell from the recent history which foil the next trial carries.
    """
    d = validate(cfg)
    seed = session_seed(participant_code, session_index)
    rng = np.random.default_rng(seed)

    def specs(block: str, items: List[tuple], practice_round: int = 0, practice_stage: int = 0) -> List[TrialSpec]:
        return [TrialSpec(i, block, v, float(s), int(t), int(rng.integers(1, 2 ** 31 - 1)),
                          practice_round, practice_stage)
                for i, (v, s, t) in enumerate(items)]

    practice_stages = []
    for si, cell in enumerate(cfg.practice_cells):
        rounds = []
        for r in range(cfg.practice_max_rounds):
            items = _balanced([cell], cfg.practice_n)
            rng.shuffle(items)
            rounds.append(specs("practice", items, practice_round=r + 1, practice_stage=si + 1))
        practice_stages.append(rounds)

    constraints = [(lambda x: (x[0], x[1]), cfg.max_condition_run)]
    if len(cfg.main_variants) > 1:
        constraints.append((lambda x: x[0], cfg.max_variant_run))
    main_items = constrained_shuffle(rng, _balanced(list(d.main_cells), cfg.trials_per_condition), constraints)
    main = specs("main", main_items)
    ctrl_items = constrained_shuffle(rng, _balanced(list(cfg.control_cells), cfg.control_trials_per_cell),
                                     [(lambda x: (x[0], x[1]), cfg.max_condition_run)])
    control = specs("control", ctrl_items)
    design = {
        "participant_code": participant_code, "session_index": int(session_index), "session_seed": seed,
        "config_hash": cfg.hash(),
        "practice_stages": [[[asdict(t) for t in r] for r in stage] for stage in practice_stages],
        "main": [asdict(t) for t in main], "control": [asdict(t) for t in control],
    }
    design["design_hash"] = design_hash(design)
    return design


def design_hash(design: dict) -> str:
    core = {k: design[k] for k in ("participant_code", "session_index", "session_seed", "config_hash",
                                   "practice_stages", "main", "control")}
    s = json.dumps(core, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def specs_from(design: dict, block: str, practice_stage: int = 1, practice_round: int = 1) -> List[TrialSpec]:
    if block == "practice":
        return [TrialSpec(**t) for t in design["practice_stages"][practice_stage - 1][practice_round - 1]]
    return [TrialSpec(**t) for t in design[block]]
