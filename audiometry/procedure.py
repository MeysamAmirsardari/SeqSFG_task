"""The threshold procedure, with no sound and no screen, so it can be tested on its own.

ThresholdSearch is BSA 2018 6.6-6.7 for one ear and one frequency, on a hearing-level scale:
  - start at a clearly audible level; if it is missed, go up 10 dB at a time (6.6);
  - after every response go down 10 dB; after every miss go up 5 dB (6.7, steps 1-4);
  - only presentations made while ascending count (a response right after a 10-dB drop sends the
    level down again, as in the Hughson-Westlake method);
  - threshold = the lowest level with responses on at least half of its ascending presentations
    and at least two responses: BSA 6.7's "two out of two, three or four". (ASHA 2005's minimum is
    two of three; ISO 8253-1:1989 asks three of up to five ascents, or two of three shortened.)
ScreenPlan is the test order of BSA 6.4, its 1-kHz retest, the repeat after a failed retest, and
the repeat of one frequency after too many false responses.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Deque, Dict, List, Optional, Tuple

from . import standards as S

OTHER = {"left": "right", "right": "left"}
FAIL_STATUSES = ("no_response_at_cap",)
UNRELIABLE_STATUSES = ("unreliable", "unreliable_false_responses")


@dataclass
class Step:
    level: float
    ascending: bool
    heard: Optional[bool] = None


class ThresholdSearch:
    def __init__(self, start: float, floor: float = S.FLOOR_DB_HL, cap: float = S.CAP_DB_HL,
                 max_presentations: int = S.MAX_PRESENTATIONS):
        if floor >= cap:
            raise ValueError("floor must be below cap")
        self.floor, self.cap, self.max_presentations = float(floor), float(cap), max_presentations
        self.level = min(max(float(start), self.floor), self.cap)
        self.start = self.level
        self.phase = "start"                 # start | descend | ascend | done
        self.steps: List[Step] = []
        self.asc_pres: Dict[float, int] = {}
        self.asc_resp: Dict[float, int] = {}
        self.threshold: Optional[float] = None
        self.status: Optional[str] = None    # found | at_floor | no_response_at_cap | unreliable

    @property
    def done(self) -> bool:
        return self.phase == "done"

    def current(self) -> Step:
        if self.done:
            raise RuntimeError("the search is finished")
        return Step(self.level, self.phase == "ascend")

    def record(self, heard: bool) -> None:
        if self.done:
            raise RuntimeError("the search is finished")
        lvl, asc = self.level, self.phase == "ascend"
        self.steps.append(Step(lvl, asc, bool(heard)))
        if asc:
            self.asc_pres[lvl] = self.asc_pres.get(lvl, 0) + 1
            if heard:
                self.asc_resp[lvl] = self.asc_resp.get(lvl, 0) + 1
        if self.phase == "start":
            if heard:
                self._down()
            elif lvl >= self.cap:
                self._finish(None, "no_response_at_cap")
            else:
                self.level = min(lvl + S.FAMILIARISATION_STEP_DB, self.cap)
        elif self.phase == "descend":
            if heard:
                self._down()
            else:
                self.phase = "ascend"
                self._up()
        else:                                # ascend
            if heard:
                if self.meets(lvl):
                    self._finish(lvl, "at_floor" if lvl <= self.floor else "found")
                else:
                    self._down()
            else:
                self._up()
        if not self.done and len(self.steps) >= self.max_presentations:
            self._finish(None, "unreliable")

    def meets(self, lvl: float) -> bool:
        r, n = self.asc_resp.get(lvl, 0), self.asc_pres.get(lvl, 0)
        return r >= S.MIN_RESPONSES and n > 0 and r / n >= S.MIN_FRACTION

    def _down(self) -> None:
        new = self.level - S.STEP_DOWN_DB
        if new <= self.floor:
            # cannot go below the floor: presentations there count as ascending ones, so a
            # listener who hears the floor level twice in up to four tries ends at the floor
            self.level, self.phase = self.floor, "ascend"
        else:
            self.level, self.phase = new, "descend"

    def _up(self) -> None:
        if self.level >= self.cap:           # missed at the cap: no threshold within the range
            self._finish(None, "no_response_at_cap")
            return
        self.level = min(self.level + S.STEP_UP_DB, self.cap)
        self.phase = "ascend"

    def _finish(self, threshold: Optional[float], status: str) -> None:
        self.threshold, self.status, self.phase = threshold, status, "done"

    def summary(self) -> dict:
        return {"start_db_hl": self.start, "threshold_db_hl": self.threshold, "status": self.status,
                "n_presentations": len(self.steps),
                "steps": [[s.level, int(s.ascending), int(bool(s.heard))] for s in self.steps],
                "ascending": {f"{k:g}": [self.asc_resp.get(k, 0), n]
                              for k, n in sorted(self.asc_pres.items())}}


# ---- the test order ---------------------------------------------------------------------------------
@dataclass
class Test:
    ear: str
    freq: float
    kind: str              # test | retest
    ear_pass: int          # 1, or 2 when the ear is repeated after a failed retest
    familiarise: bool = False
    start: float = S.FAMILIARISATION_DB_HL
    retry: int = 0         # 1 when repeated after too many false responses


@dataclass
class EarResult:
    ear: str
    ear_pass: int
    tests: Dict[float, dict] = field(default_factory=dict)      # freq -> search summary
    retest: Optional[dict] = None
    retest_diff_db: Optional[float] = None
    attempts: List[dict] = field(default_factory=list)          # tries replaced by a retry
    final: Dict[float, Optional[float]] = field(default_factory=dict)
    statuses: Dict[float, Optional[str]] = field(default_factory=dict)


class ScreenPlan:
    """BSA 2018 6.4: better ear first; 1000, 2000, 4000, 8000, 500, 250 Hz; then 1000 Hz again in
    the first ear. Retest within 5 dB: keep the more sensitive threshold. More than 5 dB (or a
    retest that cannot be compared): re-instruct and repeat that ear once; if it still differs,
    keep the lower threshold (ASHA 2005) and flag it. The second ear gets a retest only if the
    first ear showed that variation."""

    def __init__(self, first_ear: str, floor: float = S.FLOOR_DB_HL, cap: float = S.CAP_DB_HL):
        if first_ear not in OTHER:
            raise ValueError("first_ear must be left or right")
        self.floor, self.cap = floor, cap
        self.order = [first_ear, OTHER[first_ear]]
        self.ears: List[EarResult] = []
        self.flags: List[str] = []
        self.variation_seen = False
        self.events: List[str] = []           # for the app: "reinstruct", "other_ear"
        self._queue: Deque[Test] = deque()
        self._open_ear(first_ear, 1, retest=True, familiarise=True)

    # -- building --------------------------------------------------------------------------------
    def _open_ear(self, ear: str, ear_pass: int, retest: bool, familiarise: bool) -> None:
        self.ears.append(EarResult(ear, ear_pass))
        for i, f in enumerate(S.TEST_ORDER_HZ):
            self._queue.append(Test(ear, float(f), "test", ear_pass, familiarise and i == 0))
        if retest:
            self._queue.append(Test(ear, float(S.RETEST_HZ), "retest", ear_pass))

    @property
    def current_ear(self) -> EarResult:
        return self.ears[-1]

    def _adjacent(self, res: EarResult, freq: float) -> Optional[float]:
        done = [(abs(math.log2(f / freq)), f) for f, s in res.tests.items()
                if s["threshold_db_hl"] is not None and f != freq]
        if not done:
            return None
        return res.tests[min(done)[1]]["threshold_db_hl"]

    def _start(self, t: Test) -> float:
        res = self.current_ear
        if t.kind == "retest":
            first = res.tests.get(t.freq, {}).get("threshold_db_hl")
            base = None if first is None else first + S.NEXT_START_ABOVE_DB
        elif not res.tests:
            base = S.FAMILIARISATION_DB_HL
        else:
            adj = self._adjacent(res, t.freq)
            base = None if adj is None else adj + S.NEXT_START_ABOVE_DB
        if base is None:
            base = S.FAMILIARISATION_DB_HL
        return min(max(base, self.floor), self.cap)

    # -- running ---------------------------------------------------------------------------------
    def next(self) -> Optional[Test]:
        if not self._queue:
            return None
        t = self._queue[0]
        t.start = self._start(t)
        return t

    def search_for(self, t: Test) -> ThresholdSearch:
        return ThresholdSearch(t.start, self.floor, self.cap)

    def _pop(self, t: Test) -> None:
        if not self._queue or self._queue[0] is not t:
            raise RuntimeError("finish()/repeat() must get the test that next() returned")
        self._queue.popleft()

    def repeat(self, t: Test, search: ThresholdSearch, extra: Optional[dict] = None) -> None:
        """Too many false responses: keep this try for the record and measure the frequency again
        (once) after re-instruction."""
        if t.retry:
            raise RuntimeError("a frequency is repeated only once")
        self._pop(t)
        rec = {"freq": t.freq, "kind": t.kind, **search.summary(), **(extra or {})}
        self.current_ear.attempts.append(rec)
        self.flags.append(f"{t.ear} ear, {t.freq:g} Hz ({t.kind}): repeated after "
                          f"{(extra or {}).get('false_responses', '?')} false responses")
        self._queue.appendleft(replace(t, retry=1, familiarise=False))
        self.events.append("reinstruct")

    def finish(self, t: Test, search: ThresholdSearch, extra: Optional[dict] = None,
               unreliable: Optional[str] = None) -> None:
        self._pop(t)
        res = self.current_ear
        s = {**search.summary(), **(extra or {})}
        if unreliable:
            s["threshold_measured_db_hl"] = s["threshold_db_hl"]
            s["threshold_db_hl"], s["status"] = None, "unreliable_false_responses"
            self.flags.append(f"{t.ear} ear, {t.freq:g} Hz ({t.kind}): {unreliable}")
        if t.kind == "test":
            res.tests[t.freq] = s
        else:
            res.retest = s
        if not self._queue:
            self._close_ear(res)

    def _close_ear(self, res: EarResult) -> None:
        for f, s in res.tests.items():
            res.final[f] = s["threshold_db_hl"]
            res.statuses[f] = s["status"]
        repeat = False
        if res.retest is not None:
            a = res.tests.get(float(S.RETEST_HZ), {}).get("threshold_db_hl")
            b = res.retest["threshold_db_hl"]
            if a is None or b is None:
                res.retest_diff_db = None
                bad = True
            else:
                res.retest_diff_db = abs(a - b)
                bad = res.retest_diff_db > S.RETEST_TOLERANCE_DB
                res.final[float(S.RETEST_HZ)] = min(a, b)       # the more sensitive (BSA 6.4)
            if bad:
                self.variation_seen = True
                if res.ear_pass == 1:
                    repeat = True
                    self.flags.append(f"{res.ear} ear: 1-kHz retest differed by "
                                      f"{_fmt(res.retest_diff_db)}; ear re-instructed and repeated")
                else:
                    self.flags.append(f"{res.ear} ear: 1-kHz retest still differed by "
                                      f"{_fmt(res.retest_diff_db)} after the repeat; the lower "
                                      f"threshold is kept (ASHA 2005)")
        if res.ear_pass == 2:                                   # compare with the first pass
            first = next((r for r in self.ears if r.ear == res.ear and r.ear_pass == 1), None)
            if first is not None:
                for f, v in res.final.items():
                    u = first.final.get(f)
                    if u is not None and v is not None and abs(u - v) > S.PASS_DIFF_FLAG_DB:
                        self.flags.append(f"{res.ear} ear, {f:g} Hz: the two passes differ by "
                                          f"{abs(u - v):g} dB")
        for f, st in res.statuses.items():
            if st in ("unreliable", "no_response_at_cap"):
                self.flags.append(f"{res.ear} ear, {f:g} Hz: {st.replace('_', ' ')}")
        if repeat:
            self.events.append("reinstruct")
            self._open_ear(res.ear, 2, retest=True, familiarise=True)
            return
        if res.ear == self.order[0]:
            self.events.append("other_ear")
            self._open_ear(self.order[1], 1, retest=self.variation_seen, familiarise=False)

    @property
    def finished(self) -> bool:
        return not self._queue

    def final(self) -> Dict[str, Dict[float, Optional[float]]]:
        """The kept thresholds: each ear's last pass."""
        out: Dict[str, Dict[float, Optional[float]]] = {}
        for res in self.ears:
            out[res.ear] = dict(res.final)
        return out

    def final_status(self) -> Dict[str, Dict[float, Optional[str]]]:
        out: Dict[str, Dict[float, Optional[str]]] = {}
        for res in self.ears:
            out[res.ear] = dict(res.statuses)
        return out

    def summary(self) -> dict:
        return {"order": self.order, "flags": list(self.flags),
                "final_db_hl": {e: {f"{f:g}": v for f, v in d.items()} for e, d in self.final().items()},
                "final_status": {e: {f"{f:g}": v for f, v in d.items()}
                                 for e, d in self.final_status().items()},
                "ears": [{"ear": r.ear, "pass": r.ear_pass,
                          "tests": {f"{f:g}": s for f, s in r.tests.items()},
                          "retest": r.retest, "retest_diff_db": r.retest_diff_db,
                          "attempts": r.attempts,
                          "final_db_hl": {f"{f:g}": v for f, v in r.final.items()},
                          "status": {f"{f:g}": v for f, v in r.statuses.items()}}
                         for r in self.ears]}


def _fmt(d: Optional[float]) -> str:
    return "an uncomparable amount" if d is None else f"{d:g} dB"


# ---- scoring a press -----------------------------------------------------------------------------------
def response_window(on: float, off: float) -> Tuple[float, float]:
    return on + S.RESPONSE_FROM_S, off + S.RESPONSE_UNTIL_AFTER_OFFSET_S


def heard(on: float, off: float, presses: List[float]) -> Tuple[bool, Optional[float]]:
    """(heard, latency from onset) for one tone, from the key-down times."""
    a, b = response_window(on, off)
    for p in presses:
        if a <= p <= b:
            return True, p - on
    return False, None


def consistent(on: float, off: float, press: Optional[float], release: Optional[float]) -> bool:
    """BSA 6.6: the response matches the tone's onset and offset (press while it plays, release
    near its end)."""
    if press is None or release is None:
        return False
    lo, hi = S.RELEASE_WINDOW_S
    return on + S.RESPONSE_FROM_S <= press < off and off + lo <= release <= off + hi
