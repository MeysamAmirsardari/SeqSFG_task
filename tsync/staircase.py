"""The adaptive track: transformed up-down (Levitt 1971) on a multiplicative step.

The rule is the one used by Elhilali et al. (2009) and by tcoh: delta starts at 20 ms, is divided
by a factor after `n_down` consecutive correct responses and multiplied by it after an error;
the factor is 4 until the first reversal, 2 for the next two, then sqrt(2); the track stops
after six reversals at sqrt(2), and the threshold is the geometric mean of those six. With
n_down = 3 it converges on 79.4% correct.

Everything that can quietly spoil a threshold is kept: every trial, every reversal, trials
pinned at the ceiling or the floor (and whether any of them fall inside the averaged reversals),
and tracks that hit `max_trials_per_track` before converging, which report no threshold at all.

A track is a pure function of its response sequence, so `Staircase.replay` rebuilds one exactly
from logged responses. That is what makes a session resumable after a crash.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

from .config import Config


@dataclass(frozen=True)
class Step:
    index: int
    delta_ms: float          # the delta this trial was run at
    correct: bool
    step_index: int          # which factor was in force when this trial was run
    at_ceiling: bool
    at_floor: bool
    reversal: bool           # this trial's response reversed the direction of travel
    next_delta_ms: float


class Staircase:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.delta = float(cfg.delta_start_ms)
        self.step_index = 0
        self._run = 0
        self._direction = 0              # -1 down, +1 up, 0 not yet moved
        self.steps: List[Step] = []
        self.reversals: List[Tuple[float, int]] = []   # (delta at the reversal, factor index)
        self.finished = False
        self.stop_reason: Optional[str] = None

    @classmethod
    def replay(cls, cfg: Config, responses: Iterable[bool]) -> "Staircase":
        s = cls(cfg)
        for c in responses:
            s.update(bool(c))
        return s

    @property
    def final_step(self) -> int:
        return len(self.cfg.step_factors) - 1

    @property
    def n_trials(self) -> int:
        return len(self.steps)

    def reversals_at_final(self) -> int:
        return sum(1 for _, si in self.reversals if si == self.final_step)

    def update(self, correct: bool) -> Step:
        if self.finished:
            raise RuntimeError("this track has already finished")
        cfg = self.cfg
        delta, si = self.delta, self.step_index
        at_ceiling = delta >= cfg.delta_max_ms - 1e-12
        at_floor = delta <= cfg.delta_min_ms + 1e-12

        move = 0
        if correct:
            self._run += 1
            if self._run >= cfg.n_down:
                move, self._run = -1, 0
        else:
            move, self._run = +1, 0

        reversal = False
        if move:
            if self._direction and move != self._direction:
                reversal = True
                self.reversals.append((delta, si))
                need = 0
                for i, n in enumerate(cfg.reversals_per_step):
                    need += n
                    if len(self.reversals) == need and self.step_index == i:
                        self.step_index = i + 1
                        break
            self._direction = move
            factor = cfg.step_factors[self.step_index]
            self.delta = float(np.clip(delta * factor ** move, cfg.delta_min_ms, cfg.delta_max_ms))

        step = Step(len(self.steps), delta, bool(correct), si, at_ceiling, at_floor, reversal,
                    self.delta)
        self.steps.append(step)

        if self.reversals_at_final() >= cfg.n_final_reversals:
            self.finished, self.stop_reason = True, "converged"
        elif len(self.steps) >= cfg.max_trials_per_track:
            self.finished, self.stop_reason = True, "max_trials"
        return step

    # -- results -------------------------------------------------------------------------------
    def averaged_reversals(self) -> List[float]:
        final = [d for d, si in self.reversals if si == self.final_step]
        n = self.cfg.n_final_reversals
        return final[-n:] if len(final) >= n else []

    def threshold_ms(self) -> Optional[float]:
        """Geometric mean of the last n reversals at the final step; None if not converged."""
        used = self.averaged_reversals()
        if not used:
            return None
        return float(math.exp(float(np.mean(np.log(used)))))

    def audit(self) -> dict:
        used = self.averaged_reversals()
        rev_trials = [s.index for s in self.steps if s.reversal and s.step_index == self.final_step]
        first = rev_trials[-len(used)] if used else None
        tail = self.steps[first:] if first is not None else []
        n = len(self.steps)
        return {
            "n_trials": n,
            "n_reversals": len(self.reversals),
            "n_final_reversals": self.reversals_at_final(),
            "stop_reason": self.stop_reason,
            "converged": self.stop_reason == "converged",
            "threshold_ms": self.threshold_ms(),
            "reversals_used_ms": [float(x) for x in used],
            "reversal_sd_log": float(np.std(np.log(used), ddof=1)) if len(used) > 1 else None,
            "trials_at_ceiling": int(sum(s.at_ceiling for s in self.steps)),
            "trials_at_floor": int(sum(s.at_floor for s in self.steps)),
            "ceiling_in_threshold": int(sum(s.at_ceiling for s in tail)),
            "floor_in_threshold": int(sum(s.at_floor for s in tail)),
            "prop_correct": float(np.mean([s.correct for s in self.steps])) if n else None,
            "target_p": self.cfg.target_p,
        }

    def to_rows(self) -> List[dict]:
        return [asdict(s) for s in self.steps]
