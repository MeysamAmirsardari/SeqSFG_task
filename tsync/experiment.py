"""The session, written once as a script, run by either the Qt app or the headless simulator.

`Experiment.script()` is a generator. It yields what should happen next (`Screen` or `Play`) and
receives what happened (a `ScreenResult` or a `PlayResult`). It owns all decisions (practice
criterion, the staircases, catch trials, repeating an aborted trial, what to log); the app owns
only timing, sound, pixels and keys. The same script therefore runs under test, in simulation
and in the booth, and the simulator exercises every line the participant's session will.

Rules the script enforces:

* A trial that did not play cleanly (paused, audio dropout, late start, volume or device change)
  is logged with its reason and run again with fresh random numbers; it never moves a track.
* Catch trials (delta at the ceiling) are logged but never move a track.
* Every presented trial, valid or not, is a row in trials.csv and trials.jsonl.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np

from .config import Config
from .design import Plan, trial_seed
from .levels import Calibration
from .logbook import SessionWriter, now_iso, read_jsonl
from .staircase import Staircase
from .stimulus import Trial, build_trial, buffer_sha256, render, tone_table
from .text import DEBRIEF, QUESTIONNAIRE, T


# ----------------------------------------------------------------------------------------------
# what the script asks for, and what it gets back
# ----------------------------------------------------------------------------------------------
@dataclass
class Screen:
    kind: str                         # message | form | rest | alert | end
    name: str
    title: str = ""
    body: str = ""
    keys: Tuple[str, ...] = ("space",)
    min_s: float = 0.0
    figure: str = ""                  # an illustration the app draws: sequences | keys | blocks
    fields: Tuple[dict, ...] = ()
    progress: Optional[float] = None
    data: dict = field(default_factory=dict)


@dataclass
class ScreenResult:
    key: str = "space"
    values: dict = field(default_factory=dict)
    shown_s: float = 0.0


@dataclass
class Play:
    trial: Trial
    buffer: np.ndarray
    sha256: str
    peak: float
    amps: List[float]
    mode: str = "respond"             # respond | listen
    feedback: bool = True
    progress: Optional[float] = None
    label: str = ""
    captions: Tuple[str, str] = ("", "")


@dataclass
class PlayResult:
    status: str = "ok"                # ok | aborted
    reason: str = ""                  # pause | operator | audio_underflow | audio_late |
                                      # audio_error | volume_changed | device_lost | timeout
    response: Optional[int] = None
    key: str = ""
    rt_ms: Optional[float] = None     # from the end of sequence 2 (DAC time) to the key press
    rt_from_onset2_ms: Optional[float] = None
    keys: List[dict] = field(default_factory=list)
    early_keys: int = 0
    reminder_shown: bool = False
    audio: dict = field(default_factory=dict)
    visual: dict = field(default_factory=dict)
    system: dict = field(default_factory=dict)
    wall_start: str = ""
    wall_response: str = ""


class SessionEnded(Exception):
    pass


# ----------------------------------------------------------------------------------------------
# resuming
# ----------------------------------------------------------------------------------------------
@dataclass
class ResumeState:
    resumes: int
    main_started: bool
    responses: Dict[int, List[bool]]      # per block: staircase responses so far, in order
    catches: Dict[int, List[int]]         # per block: catch slots already delivered
    practice_passed: bool

    @classmethod
    def from_folder(cls, folder, resumes: int) -> "ResumeState":
        rows = [r["row"] for r in read_jsonl(folder / "trials.jsonl")]
        responses: Dict[int, List[bool]] = {}
        catches: Dict[int, List[int]] = {}
        main = False
        practice_passed = False
        for r in rows:
            if r.get("phase") == "main":
                main = True
                if r.get("status") != "ok":
                    continue
                b = int(r["block"])
                if r.get("is_catch"):
                    catches.setdefault(b, []).append(int(r["staircase_trial"]))
                else:
                    responses.setdefault(b, []).append(bool(r["correct"]))
            if r.get("phase") == "exposure":
                practice_passed = True
        return cls(resumes, main, responses, catches, practice_passed)


# ----------------------------------------------------------------------------------------------
# the script
# ----------------------------------------------------------------------------------------------
class Experiment:
    def __init__(self, cfg: Config, cal: Calibration, plan: Plan, writer: SessionWriter,
                 resume: Optional[ResumeState] = None):
        self.cfg, self.cal, self.plan, self.w = cfg, cal, plan, writer
        self.conds = {c.name: c for c in cfg.conditions()}
        self.resume = resume
        self.attempt_base = 1000 * (resume.resumes if resume else 0)
        self.n_blocks = len(plan.blocks)
        self.tracks: Dict[int, dict] = dict(writer.meta.get("tracks", {}) or {})
        self.est_per_block = 42 + cfg.catch_per_track
        lo, hi = min(cfg.context_ms), max(cfg.context_ms)
        self.sync_name, self.async_name = f"D{lo:g}", f"D{hi:g}"

    # -- building and logging trials ---------------------------------------------------------
    def make_play(self, cond_name: str, delta: float, phase: str, block: int, k: int,
                  attempt: int, *, is_catch=False, target_interval=None, sign=None,
                  mode="respond", feedback=True, label="", captions=("", ""),
                  progress=None) -> Play:
        cond = self.conds[cond_name]
        seed = trial_seed(self.plan, phase, block, k, attempt + self.attempt_base)
        tr = build_trial(self.cfg, cond, delta, seed, phase=phase, is_catch=is_catch,
                         target_interval=target_interval, sign=sign)
        buf, amps = render(self.cfg, tr, self.cal.amplitude)
        return Play(tr, buf, buffer_sha256(buf), float(np.max(np.abs(buf))), amps, mode,
                    feedback, progress, label, captions)

    def log(self, play: Play, res: PlayResult, *, block: int, block_condition: str,
            trial_in_block: int, staircase_trial: Optional[int], attempt: int,
            step=None) -> Optional[bool]:
        tr = play.trial
        ok = res.status == "ok"
        correct = None
        if ok and play.mode == "respond" and res.response in (1, 2):
            correct = res.response == tr.target_interval
        a, v, s = res.audio, res.visual, res.system
        row = {
            "phase": tr.phase, "block": block, "block_condition": block_condition,
            "trial_in_block": trial_in_block, "staircase_trial": staircase_trial,
            "attempt": attempt, "status": res.status, "abort_reason": res.reason,
            "condition": tr.condition, "context_ms": tr.context_ms,
            "ending_shift_ms": tr.ending_shift_ms, "is_catch": tr.is_catch,
            "delta_ms": tr.delta_ms, "sign": tr.sign,
            "order": "B first" if tr.sign > 0 else "A first",
            "target_interval": tr.target_interval, "response": res.response, "correct": correct,
            "key": res.key, "rt_ms": res.rt_ms, "rt_from_onset2_ms": res.rt_from_onset2_ms,
            "n_early_keys": res.early_keys, "reminder_shown": res.reminder_shown,
            "step_index": step.step_index if step else None,
            "reversal": step.reversal if step else None,
            "next_delta_ms": step.next_delta_ms if step else None,
            "at_ceiling": step.at_ceiling if step else None,
            "at_floor": step.at_floor if step else None,
            "isi_ms": tr.isi_ms, "async1_ms": tr.asynchrony_ms[0], "async2_ms": tr.asynchrony_ms[1],
            "seed": "-".join(str(x) for x in tr.seed), "sha256": play.sha256, "peak": play.peak,
            "onset_error_ms": a.get("onset_error_ms"), "audio_late": a.get("late"),
            "underflows": a.get("underflows"), "callback_gap_max_ms": a.get("callback_gap_max_ms"),
            "visual_lag1_ms": v.get("lag1_ms"), "visual_lag2_ms": v.get("lag2_ms"),
            "volume": s.get("output_volume"), "muted": s.get("muted"),
            "alert_volume": s.get("alert_volume"), "device": s.get("device"),
            "wall_start": res.wall_start, "wall_response": res.wall_response,
            "t_session_s": self.w.t(),
        }
        detail = {"trial": tr.summary(), "tones": tone_table(tr, play.amps),
                  "result": asdict(res), "mode": play.mode, "feedback": play.feedback}
        self.w.trial(row, detail)
        return correct

    def run_trial(self, make, *, block: int, block_condition: str, trial_in_block: int,
                  staircase_trial: Optional[int], on_valid=None):
        """Present until one clean presentation; log every attempt. Returns (play, result, step)."""
        attempt = 0
        while True:
            play = make(attempt)
            res = yield play
            if res.status != "ok":
                self.log(play, res, block=block, block_condition=block_condition,
                         trial_in_block=trial_in_block, staircase_trial=staircase_trial,
                         attempt=attempt)
                self.w.event("trial_repeated", reason=res.reason, block=block,
                             trial_in_block=trial_in_block)
                attempt += 1
                continue
            step = on_valid(play, res) if on_valid else None
            self.log(play, res, block=block, block_condition=block_condition,
                     trial_in_block=trial_in_block, staircase_trial=staircase_trial,
                     attempt=attempt, step=step)
            return play, res, step

    # -- the session -------------------------------------------------------------------------
    def script(self) -> Generator:
        cfg, r = self.cfg, self.resume
        try:
            if r is not None and r.main_started:
                yield Screen("message", "resume", T["resume_title"], T["resume_body"])
                yield from self.warmup()
            else:
                if r is None:
                    yield from self.about()
                yield from self.instructions()
                yield from self.sound_check()
                yield from self.demos()
                passed = yield from self.practice()
                if not passed:
                    self.w.update_meta(practice_passed=False)
                    key = (yield Screen("alert", "practice_fail", T["practice_fail_title"],
                                        T["practice_fail_body"], keys=("c", "q"))).key
                    self.w.event("practice_decision", key=key)
                    if key == "q":
                        yield Screen("end", "end_practice", T["end_title"], T["end_body"],
                                     keys=("escape",))
                        raise SessionEnded("ended after practice")
                else:
                    self.w.update_meta(practice_passed=True)
                yield from self.exposure()
                yield Screen("message", "main_intro", T["main_title"],
                             T["main_body"].format(n=self.n_blocks), figure="blocks",
                             data={"blocks": self.n_blocks, "done": 0})
            for b, name in enumerate(self.plan.blocks):
                done = r.responses.get(b, []) if r else []
                if r is not None and Staircase.replay(cfg, done).finished:
                    continue
                yield from self.block(b, name, done, r.catches.get(b, []) if r else [])
            yield from self.debrief()
            self.w.close("complete")
            yield Screen("end", "end", T["end_title"], T["end_body"], keys=("escape",))
        except SessionEnded as e:
            self.w.event("session_ended", why=str(e))
            if self.w.meta.get("status") == "running":
                self.w.close("ended_early")

    def about(self):
        res = yield Screen("form", "about", T["about_title"], T["about_body"], keys=("return",),
                           fields=QUESTIONNAIRE)
        self.w.update_meta(participant_answers=res.values)

    def instructions(self):
        yield Screen("message", "intro1", T["intro1_title"], T["intro1_body"])
        yield Screen("message", "intro2", T["intro2_title"], T["intro2_body"], figure="sequences",
                     min_s=2.0)
        yield Screen("message", "intro3", T["intro3_title"], T["intro3_body"], figure="keys",
                     min_s=2.0)

    def sound_check(self):
        for attempt in range(5):
            yield Screen("message", "check", T["check_title"], T["check_body"])
            make = lambda a, at=attempt: self.make_play(
                self.sync_name, 0.0, "check", 0, at, a, mode="listen", feedback=False,
                label=T["check_title"])
            yield from self.run_trial(make, block=-1, block_condition="check",
                                      trial_in_block=attempt, staircase_trial=None)
            key = (yield Screen("message", "check_q", T["check_q_title"], T["check_q_body"],
                                keys=("y", "n"))).key
            self.w.event("sound_check", answer=key, attempt=attempt)
            if key == "y":
                return
            yield Screen("alert", "check_fail", T["check_fail_title"], T["check_fail_body"])
        self.w.event("sound_check_unresolved")

    def demos(self):
        cfg = self.cfg
        examples = [(self.sync_name, 2, +1), (self.async_name, 1, -1)]
        for i, (name, ti, sign) in enumerate(examples):
            caps = (T["caption_async"], T["caption_sync"]) if ti == 1 else \
                   (T["caption_sync"], T["caption_async"])
            while True:
                yield Screen("message", f"demo{i + 1}",
                             T["demo_title"].format(i=i + 1, n=len(examples)), T["demo_body"])
                make = lambda a, i=i, name=name, ti=ti, sign=sign: self.make_play(
                    name, cfg.demo_delta_ms, "demo", i, 0, a, target_interval=ti, sign=sign,
                    mode="listen", feedback=False, captions=caps, label="Example")
                yield from self.run_trial(make, block=-1, block_condition="demo",
                                          trial_in_block=i, staircase_trial=None)
                key = (yield Screen("message", f"demo{i + 1}_again", T["demo_again_title"],
                                    T["demo_again_body"], keys=("r", "space"))).key
                if key == "space":
                    break

    def practice(self):
        cfg = self.cfg
        n = cfg.practice_n
        for attempt in range(cfg.practice_max_attempts):
            yield Screen("message", f"practice{attempt + 1}", T["practice_title"],
                         T["practice_body"].format(n=n))
            k_ok = 0
            for k in range(n):
                make = lambda a, k=k, attempt=attempt: self.make_play(
                    self.sync_name, cfg.practice_delta_ms, "practice", attempt, k, a,
                    label=f"Practice {k + 1} of {n}", progress=(k + 1) / n)
                play, res, _ = yield from self.run_trial(
                    make, block=-1, block_condition="practice", trial_in_block=k,
                    staircase_trial=None)
                k_ok += int(res.response == play.trial.target_interval)
            self.w.event("practice_score", attempt=attempt, correct=k_ok, n=n)
            if k_ok >= cfg.practice_criterion:
                yield Screen("message", "practice_pass", T["practice_pass_title"],
                             T["practice_pass_body"].format(k=k_ok, n=n))
                return True
            if attempt + 1 < cfg.practice_max_attempts:
                yield Screen("message", "practice_retry",
                             T["practice_retry_title"].format(k=k_ok, n=n),
                             T["practice_retry_body"], min_s=2.0)
        return False

    def exposure(self):
        cfg = self.cfg
        yield Screen("message", "exposure", T["exposure_title"], T["exposure_body"])
        n = len(self.plan.exposure)
        for k, name in enumerate(self.plan.exposure):
            make = lambda a, k=k, name=name: self.make_play(
                name, cfg.exposure_delta_ms, "exposure", 0, k, a,
                label=f"Practice {k + 1} of {n}", progress=(k + 1) / n)
            yield from self.run_trial(make, block=-1, block_condition="exposure",
                                      trial_in_block=k, staircase_trial=None)

    def warmup(self):
        cfg = self.cfg
        for k in range(3):
            make = lambda a, k=k: self.make_play(self.sync_name, cfg.practice_delta_ms, "warmup",
                                                 0, k, a,
                                                 label="Warm-up")
            yield from self.run_trial(make, block=-1, block_condition="warmup",
                                      trial_in_block=k, staircase_trial=None)

    def block(self, b: int, name: str, done: List[bool], caught: List[int]):
        cfg = self.cfg
        nb = self.n_blocks
        if b == 0 and not done:
            yield Screen("rest", f"block{b + 1}", T["block_title"].format(i=1, n=nb),
                         T["block_body"], figure="blocks", data={"blocks": nb, "done": 0})
        elif not done:
            long = (b == cfg.long_rest_after_block)
            yield Screen("rest", f"rest{b}", T["rest_title"].format(i=b, n=nb),
                         T["rest_long_body"] if long else T["rest_body"],
                         min_s=cfg.long_rest_s if long else cfg.min_rest_s, figure="blocks",
                         data={"blocks": nb, "done": b})
        stair = Staircase.replay(cfg, done)
        slots = [k for k in self.plan.catch_slots[b] if k not in caught]
        self.w.event("block_start", block=b, condition=name, resumed_at=len(done))
        n_in_block = len(done) + len(caught)

        def progress(k):
            return (b + min(k / self.est_per_block, 0.98)) / nb

        while not stair.finished:
            k = stair.n_trials
            while slots and slots[0] < k:          # a slot passed while the session was down
                slots.pop(0)
            if slots and slots[0] == k:
                slots.pop(0)
                make = lambda a, k=k: self.make_play(
                    name, cfg.catch_delta_ms, "main", b, 500 + k, a, is_catch=True,
                    label=f"Part {b + 1} of {nb}", progress=progress(n_in_block))
                yield from self.run_trial(make, block=b, block_condition=name,
                                          trial_in_block=n_in_block, staircase_trial=k)
                n_in_block += 1
                continue

            def valid(play, res, stair=stair):
                return stair.update(res.response == play.trial.target_interval)

            make = lambda a, k=k: self.make_play(
                name, stair.delta, "main", b, k, a, label=f"Part {b + 1} of {nb}",
                progress=progress(n_in_block))
            yield from self.run_trial(make, block=b, block_condition=name,
                                      trial_in_block=n_in_block, staircase_trial=k, on_valid=valid)
            n_in_block += 1
        audit = dict(stair.audit(), block=b, condition=name)
        self.tracks[b] = audit
        self.w.event("track_done", **audit)
        self.w.update_meta(tracks=self.tracks)

    def debrief(self):
        res = yield Screen("form", "debrief", T["debrief_title"], T["debrief_body"],
                           keys=("return",), fields=DEBRIEF)
        self.w.update_meta(debrief=res.values)
