"""The session as a script (a generator), plus the on-disk log. Shared by the app and the simulator.

The script yields Screen or Play and receives what happened. It owns every decision (which track,
which level, repeats after a pause or an audio fault, the rating block) and writes every trial to
disk (fsync) before asking for the next one, so a crash loses at most the trial in flight and a
session can resume where it stopped.
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np

from tsync.logbook import now_iso, read_jsonl, write_json_atomic
from tsync.text import QUESTIONNAIRE

from .config import ARRANGEMENTS, Condition, Config
from .procedure import Track, make_order, make_tracks
from .stimulus import build, sha256

FIELDS = ["row", "phase", "index", "attempt", "status", "reason", "condition", "envelope",
          "arrangement", "dt_ms", "tone_hz", "track", "track_trial", "level_db", "signal_interval",
          "response", "correct", "rating", "rt_ms", "step_db", "reversal", "next_level_db",
          "seed", "sha256", "onset_error_ms", "underflows", "volume", "muted", "wall", "t_s"]


@dataclass
class Screen:
    name: str
    title: str = ""
    body: str = ""
    keys: Tuple[str, ...] = ("space",)
    kind: str = "message"            # message | form | alert | end
    fields: Tuple[dict, ...] = ()
    min_s: float = 0.0
    progress: Optional[float] = None


@dataclass
class Play:
    trial: object
    buffer: np.ndarray
    sha: str
    mode: str = "respond"            # respond | listen | rate
    feedback: bool = True
    label: str = ""
    progress: Optional[float] = None
    caption: str = ""


@dataclass
class Result:
    status: str = "ok"
    reason: str = ""
    response: Optional[int] = None
    rt_ms: Optional[float] = None
    keys: List[dict] = field(default_factory=list)
    audio: dict = field(default_factory=dict)
    system: dict = field(default_factory=dict)


class Writer:
    def __init__(self, folder: Path, meta: dict, resume: bool = False):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.t0 = time.perf_counter()
        mp = self.folder / "session.json"
        if resume:
            self.meta = json.loads(mp.read_text())
            self.meta.setdefault("resumes", []).append(now_iso())
            self.meta["status"] = "running"
        else:
            if mp.exists():
                raise FileExistsError(f"{self.folder} already holds a session")
            self.meta = dict(meta, status="running", created=now_iso())
        self.save()
        new = not (self.folder / "trials.csv").exists()
        self._csv_f = open(self.folder / "trials.csv", "a", newline="")
        self._csv = csv.DictWriter(self._csv_f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            self._csv.writeheader()
        self._js = open(self.folder / "trials.jsonl", "a")
        self._ev = open(self.folder / "events.jsonl", "a")
        self.rows = len(read_jsonl(self.folder / "trials.jsonl"))
        self.closed = False

    @staticmethod
    def _sync(f):
        f.flush()
        os.fsync(f.fileno())

    def save(self):
        write_json_atomic(self.folder / "session.json", self.meta)

    def event(self, what: str, /, **kw):
        if self.closed:
            return
        self._ev.write(json.dumps({"t": now_iso(), "event": what, **kw}, default=str) + "\n")
        self._sync(self._ev)

    def trial(self, row: dict, detail: dict):
        self.rows += 1
        row = dict(row, row=self.rows, wall=now_iso(), t_s=round(time.perf_counter() - self.t0, 3))
        self._js.write(json.dumps(dict(detail, row=row), default=float) + "\n")
        self._sync(self._js)
        self._csv.writerow({k: ("" if v is None else v) for k, v in row.items()})
        self._sync(self._csv_f)

    def close(self, status: str):
        if self.closed:
            return
        self.meta.update(status=status, ended=now_iso())
        self.save()
        self.closed = True
        for f in (self._csv_f, self._js, self._ev):
            try:
                self._sync(f)
                f.close()
            except (OSError, ValueError):
                pass


T = {
    "intro1": ("Listening for a tone in noise",
               "Each trial has three short bursts of rushing noise, one after the other. Each "
               "burst is a few seconds long and ends with a slightly louder part.\n\nIn ONE of the "
               "three, a steady beep (a pure tone) is hidden in that last part. Your task: say which "
               "of the three had the beep."),
    "intro2": ("How to answer",
               "The three boxes on the screen light up in turn, one for each sound.\n\n"
               "Press 1, 2 or 3 (or ← ↓ →) for the sound with the beep. After each "
               "answer you will see whether it was right.\n\nThe beep gets quieter when you are "
               "right and louder when you are wrong, so many trials will be hard. When you are not "
               "sure, guess. Press P at any time to pause."),
    "examples": ("Two examples", "Listen: the beep is easy to hear in these. The box with the "
                 "beep is marked after each example."),
    "practice": ("Practice", "Eight practice trials. Answer with 1, 2 or 3."),
    "main": ("The main part", "About 30 minutes, with three short breaks."),
    "rest": ("Short break", "Take a moment. Press Space when you are ready."),
    "rate": ("Last part: how did the noise sound?",
             "You will hear single noise sequences, without a beep. After each one, tell us how it "
             "sounded:\n\n1 = one single sound that pulses together\n5 = separate layers that "
             "pulse on their own\n\nUse 2, 3 and 4 for in between. There are no right answers."),
    "end": ("Thank you!", "That is the end. Please let the experimenter know."),
    "resume": ("Welcome back", "We will continue where you left off."),
}


class Experiment:
    def __init__(self, cfg: Config, writer: Writer, seed: int, amplitude=None,
                 resume_rows: Optional[List[dict]] = None):
        self.cfg, self.w, self.seed, self.amp = cfg, writer, int(seed), amplitude
        self.tracks = make_tracks(cfg)
        self.order = make_order(cfg, self.seed)
        self.done = 0
        self.rated = 0
        self.resumed = resume_rows is not None
        if resume_rows:
            per: Dict[int, List[bool]] = {}
            for r in resume_rows:
                if r["phase"] == "main" and r["status"] == "ok":
                    per.setdefault(int(r["track"]), []).append(bool(r["correct"]))
                if r["phase"] == "rating" and r["status"] == "ok":
                    self.rated += 1
            for t in self.tracks:
                t.__init__(cfg, t.cond, t.id)
                for c in per.get(t.id, []):
                    t.update(c)
            self.done = sum(len(v) for v in per.values())

    # ------------------------------------------------------------------------------------------
    def _play(self, phase, cond, level, sd, *, mode="respond", label="", progress=None,
              n_intervals=3, signal_interval=None, caption=""):
        tr, buf = build(self.cfg, cond, level, sd, phase=phase, n_intervals=n_intervals,
                        signal_interval=signal_interval, amplitude=self.amp)
        return Play(tr, buf, sha256(buf), mode, mode == "respond", label, progress, caption)

    def _log(self, play: Play, res: Result, index, attempt, track=None, step=None, rating=None):
        tr = play.trial
        correct = (res.response == tr.signal_interval) if (play.mode == "respond"
                                                           and res.status == "ok") else None
        a, s = res.audio, res.system
        row = {"phase": tr.phase, "index": index, "attempt": attempt, "status": res.status,
               "reason": res.reason, "condition": tr.condition, "envelope": tr.envelope,
               "arrangement": tr.arrangement, "dt_ms": tr.dt_ms,
               "tone_hz": getattr(tr, "tone_hz", None),
               "track": track.id if track else None,
               "track_trial": step.index if step else None, "level_db": tr.level_db,
               "signal_interval": tr.signal_interval, "response": res.response,
               "correct": correct, "rating": rating, "rt_ms": res.rt_ms,
               "step_db": step.step_db if step else None,
               "reversal": step.reversal if step else None,
               "next_level_db": step.next_level_db if step else None,
               "seed": "-".join(map(str, tr.seed)), "sha256": play.sha,
               "onset_error_ms": a.get("onset_error_ms"), "underflows": a.get("underflows"),
               "volume": s.get("output_volume"), "muted": s.get("muted")}
        self.w.trial(row, {"trial": tr.summary(), "result": asdict(res), "mode": play.mode})
        return correct

    def _until_ok(self, make, log):
        attempt = 0
        while True:
            play = make(attempt)
            res = yield play
            if res.status != "ok":
                log(play, res, attempt)
                self.w.event("repeat", reason=res.reason)
                attempt += 1
                continue
            return play, res, attempt

    # ------------------------------------------------------------------------------------------
    def script(self) -> Generator:
        cfg = self.cfg
        cmap = {c.name: c for c in cfg.condition_list()}
        co0 = next(c for c in cfg.condition_list() if c.envelope == "co" and c.dt_ms == 0)
        r0 = next(c for c in cfg.condition_list() if c.envelope == "rand" and c.dt_ms == 0)
        total = len(self.order)
        if not self.resumed:
            res = yield Screen("about", "A few questions first", "Press Enter when you are done.",
                               ("return",), "form", QUESTIONNAIRE)
            self.w.meta["participant_answers"] = (res or {}).get("values")
            self.w.save()
            for k in ("intro1", "intro2", "examples"):
                yield Screen(k, *T[k], min_s=1.5 if k != "examples" else 0)
            for i, (c, iv) in enumerate(((co0, 2), (r0, 3))):
                make = lambda a, i=i, c=c, iv=iv: self._play(
                    "example", c, 80.0, (self.seed, 1, i, a), mode="listen", label="Example",
                    signal_interval=iv, caption=f"The beep was in sound {iv}")
                play, res, att = yield from self._until_ok(make, lambda p, r, a: self._log(p, r, i, a))
                self._log(play, res, i, att)
            yield Screen("practice", *T["practice"])
            for i, lvl in enumerate(cfg.practice_levels_db):
                c = co0 if i % 2 == 0 else r0
                make = lambda a, i=i, c=c, lvl=lvl: self._play(
                    "practice", c, lvl, (self.seed, 2, i, a), label=f"Practice {i + 1} of "
                    f"{len(cfg.practice_levels_db)}", progress=(i + 1) / len(cfg.practice_levels_db))
                play, res, att = yield from self._until_ok(make, lambda p, r, a: self._log(p, r, i, a))
                self._log(play, res, i, att)
            yield Screen("main", *T["main"])
        else:
            yield Screen("resume", *T["resume"])
        per_round = cfg.n_tracks
        while self.done < total:
            i = self.done
            if i and i % (per_round * cfg.break_every_rounds) == 0 and \
                    self.w.meta.get("last_break") != i:
                self.w.meta["last_break"] = i
                yield Screen("rest", *T["rest"], min_s=5.0, progress=i / total)
            track = self.tracks[self.order[i]]
            k = len(track.steps)
            make = lambda a, track=track, k=k: self._play(
                "main", track.cond, track.level, (self.seed, 6, track.id, k, a),
                label="Main part", progress=i / total)
            play, res, att = yield from self._until_ok(
                make, lambda p, r, a: self._log(p, r, i, a, track))
            step = track.update(res.response == play.trial.signal_interval)
            self._log(play, res, i, att, track, step)
            self.done += 1
        self.w.meta["tracks"] = {t.id: {"condition": t.cond.name, "final_level_db": t.level,
                                       "reversal_threshold_db": t.reversal_threshold()}
                                 for t in self.tracks}
        self.w.save()
        # ratings: one sequence (precursors + final burst, no tone) per item
        items = [(a, d) for a, d in cfg.ratings for _ in range(cfg.rating_reps)]
        rng = np.random.default_rng([self.seed, 77])
        items = [items[j] for j in rng.permutation(len(items))]
        if self.rated < len(items):
            yield Screen("rate", *T["rate"], min_s=2.0)
        for j in range(self.rated, len(items)):
            a, d = items[j]
            c = Condition("co", a, d)
            make = lambda att, j=j, c=c: self._play(
                "rating", c, None, (self.seed, 8, j, att), mode="rate", n_intervals=1,
                label=f"Sound {j + 1} of {len(items)}")
            play, res, att = yield from self._until_ok(make, lambda p, r, a_: self._log(p, r, j, a_))
            self._log(play, res, j, att, rating=res.response)
        self.w.close("complete")
        yield Screen("end", *T["end"], ("escape",), "end")
