"""The hearing screen in the booth: a full-screen window that never changes while tones play, the
space bar as the response button (press and hold while the tone is heard, BSA 2018 6.1), one tone
at a time in one cup, scored automatically.

Experimenter keys: Enter starts (after the instructions are read aloud) and continues after a
message; Esc pauses (C continues, Q saves and stops). The system volume is read twice a second;
a tone is scored only after a reading taken after it ended, so a tone played at a changed or muted
volume is never scored: the screen pauses and the tone is voided.
"""
from __future__ import annotations

import concurrent.futures
import platform
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter

from . import procedure as P
from . import standards as S
from .rig import MAX_PEAK, NOMINAL_MAX_DB, nominal_from_hl, peak_for
from .tones import tone

LEAD_S = 0.3                   # a tone is scheduled this far ahead
SETTLE_S = 1.0                 # quiet before the first tone of a test and after any pause
KEY_WAIT_S = 0.25              # while the key is held, look again this often
EVAL_MARGIN_S = 0.10           # a presentation is scored this long after its window closes
GUARD_TICK_MS = 100            # the volume reader is looked at this often
READ_EVERY_S = 0.5             # a new volume reading is started this often
VOL_WAIT_MAX_S = 3.0           # no good reading this long after a tone: pause

INSTRUCTIONS = (
    "I am going to test your hearing by measuring the quietest sounds that you can hear. Each "
    "ear is tested separately, with tones of different pitches.\n\n"
    "As soon as you hear a tone, press the SPACE bar. Keep it pressed for as long as you hear "
    "the tone, no matter which ear you hear it in. Release it as soon as you no longer hear "
    "the tone.\n\n"
    "Whatever the sound, and no matter how faint, press as soon as you think you hear it, and "
    "release as soon as you think it stops.\n\n"
    "If you have ringing in your ears, try to ignore it and respond only to the tones; tell us if "
    "you cannot tell them apart. Please sit quietly. Tell us at any time if anything is "
    "uncomfortable.")
# BSA 2018 6.1 (with "button" made "SPACE bar"), its tinnitus instruction, and ASHA 2005's "each
# ear is to be tested separately with tones of different pitches".
READ_ALOUD = "Experimenter: read this aloud, ask whether it is clear, then press Enter"

BG, TEXT, MUTED, WARN = "#0E1116", "#E9EDF2", "#8F9AA8", "#F2B84B"
MAC = {49: "space", 36: "return", 76: "return", 53: "escape", 8: "c", 12: "q"}
QTK = {Qt.Key.Key_Space: "space", Qt.Key.Key_Return: "return", Qt.Key.Key_Enter: "return",
       Qt.Key.Key_Escape: "escape", Qt.Key.Key_C: "c", Qt.Key.Key_Q: "q"}


def key_name(e) -> str:
    if platform.system() == "Darwin" and e.nativeVirtualKey() in MAC:
        return MAC[e.nativeVirtualKey()]
    return QTK.get(e.key(), "")


@dataclass
class Context:
    code: str
    first_ear: str
    offsets: Dict[str, Dict[float, float]]   # cup -> freq -> provisional offset (jig baseline)
    locked_volume: dict
    seed: int
    log: Callable[[dict], None]              # appends one event to events.jsonl
    save: Callable[[dict], None]             # rewrites screen.json


class Canvas(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.title, self.body, self.footer, self.style = "", "", "", "text"

    def show_text(self, title="", body="", footer="", style="text"):
        self.title, self.body, self.footer, self.style = title, body, footer, style
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        u = h / 100
        p.fillRect(self.rect(), QColor(BG))
        cw = min(w * 0.64, 120 * u)
        x = (w - cw) / 2

        def text(r, s, px, color, weight=QFont.Weight.Normal, align=Qt.AlignmentFlag.AlignLeft):
            f = QFont()
            f.setPixelSize(max(10, int(px)))
            f.setWeight(weight)
            p.setFont(f)
            p.setPen(QColor(color))
            p.drawText(r, int(align | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap), s)

        text(QRectF(x, h * 0.10, cw, 10 * u), self.title, 4.0 * u,
             WARN if self.style == "warn" else TEXT, QFont.Weight.Medium)
        text(QRectF(x, h * 0.10 + 9 * u, cw, 66 * u), self.body, 2.4 * u, TEXT)
        text(QRectF(0, h * 0.89, w, 5 * u), self.footer, 2.0 * u, MUTED,
             align=Qt.AlignmentFlag.AlignHCenter)


class Window(QtWidgets.QMainWindow):
    def __init__(self, on_key):
        super().__init__()
        self.canvas = Canvas()
        self.setCentralWidget(self.canvas)
        self.setWindowTitle("Hearing screen")
        self.on_key, self.allow_close = on_key, False

    def keyPressEvent(self, e):
        if not e.isAutoRepeat():
            self.on_key(key_name(e), True, time.perf_counter())

    def keyReleaseEvent(self, e):
        if not e.isAutoRepeat():
            self.on_key(key_name(e), False, time.perf_counter())

    def closeEvent(self, e):
        if self.allow_close:
            e.accept()
        else:
            e.ignore()


class Screening(QtCore.QObject):
    def __init__(self, win: Window, engine, ctx: Context, on_exit: Callable[[str], None],
                 volume_reader: Callable[[], dict]):
        super().__init__()
        self.win, self.engine, self.ctx, self.on_exit = win, engine, ctx, on_exit
        self.read_volume = volume_reader
        self.rng = np.random.default_rng(ctx.seed)
        self.plan = P.ScreenPlan(ctx.first_ear)
        self.test: Optional[P.Test] = None
        self.search: Optional[P.ThresholdSearch] = None
        self.pres: Optional[dict] = None
        self.key_down, self.key_since, self.key_intervals = False, 0.0, []
        self.not_before = 0.0
        self.prev_off: Optional[float] = None
        self.prev_dur: Optional[float] = None
        self.next_dur: Optional[float] = None
        self.next_gap: Optional[float] = None
        self.last_was_catch = False
        self.state = "instructions"          # instructions | message | run | pause | end
        self.after_message: Optional[Callable[[], None]] = None
        self.held_message = None
        self.status = "stopped"
        self.familiarised = False
        self.fam_tries = 0
        self.false_since_instruct = 0
        self.audio_voids = 0
        self._reset_test_counts()
        self.n_false = self.n_void = self.n_tones = self.n_catch = 0
        self.timers: List[QtCore.QTimer] = []
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.vol_future = None
        self.vol_started = 0.0
        self.last_submit = -1e9
        self.last_ok_read_start: Optional[float] = None
        self.guard = QtCore.QTimer(self)
        self.guard.setTimerType(Qt.TimerType.PreciseTimer)
        self.guard.timeout.connect(self._check_volume)
        self.t0 = time.perf_counter()

    def _reset_test_counts(self) -> None:
        self.false_in_test = self.catch_in_test = self.catch_false_in_test = self.void_in_test = 0

    # -- plumbing ---------------------------------------------------------------------------------
    def _at(self, t: float, fn: Callable[[], None]) -> QtCore.QTimer:
        tm = QtCore.QTimer(self)
        tm.setSingleShot(True)
        tm.setTimerType(Qt.TimerType.PreciseTimer)
        tm.timeout.connect(lambda: self._fire(tm, fn))
        self.timers.append(tm)
        tm.start(max(0, int(round((t - time.perf_counter()) * 1000))))
        return tm

    def _fire(self, tm, fn):
        if tm in self.timers:
            self.timers.remove(tm)
        try:
            fn()
        except Exception as e:                    # never leave the listener in a silent loop
            self._fatal(e)

    def _cancel_timers(self) -> None:
        for tm in self.timers:
            tm.stop()
        self.timers = []

    def _event(self, name: str, /, **kw) -> None:
        self.ctx.log({"t": round(time.perf_counter() - self.t0, 4), "event": name, **kw})

    def _paint(self, title="", body="", footer="", style="text"):
        self.win.canvas.show_text(title, body, footer, style)

    # -- start and messages ------------------------------------------------------------------------
    def start(self) -> None:
        self.guard.start(GUARD_TICK_MS)
        self._event("start", first_ear=self.ctx.first_ear, seed=self.ctx.seed)
        self._message("Hearing check", INSTRUCTIONS, READ_ALOUD, self._next_test)

    def _message(self, title, body, footer, then, style="text") -> None:
        self.state, self.after_message = "message", then
        self._paint(title, body, footer, style)

    def _volume_ok(self, v: dict) -> bool:
        return (bool(v.get("readable")) and not v.get("muted")
                and v.get("output_volume") == self.ctx.locked_volume.get("output_volume"))

    def _read_now(self) -> dict:
        try:
            return self.read_volume() or {}
        except Exception as e:
            return {"readable": False, "error": repr(e)}

    def _volume_text(self, v: dict) -> str:
        return (f"The system volume reads {v.get('output_volume')} (muted: {v.get('muted')}); it "
                f"must be {self.ctx.locked_volume.get('output_volume')} and not muted.")

    def _running_screen(self) -> None:
        ear = self.test.ear if self.test else ""
        self._paint("Listening", "Press and hold SPACE while you hear a tone. Release it when the "
                    "tone stops.", f"{ear} ear")

    # -- tests ------------------------------------------------------------------------------------
    def _next_test(self) -> None:
        while self.plan.events:
            ev = self.plan.events.pop(0)
            self._event("plan", what=ev)
            if ev == "reinstruct":
                nxt = self.plan.next()
                if nxt is not None and nxt.familiarise:
                    self.familiarised, self.fam_tries = False, 0
                self.false_since_instruct = 0
                self._message("Once more", INSTRUCTIONS, READ_ALOUD, self._next_test)
                return
            if ev == "other_ear":
                self._message("The other ear", "The same again, now in the other ear.",
                              "Experimenter: press Enter to go on", self._next_test)
                return
        self.test = self.plan.next()
        if self.test is None:
            self._finish("complete")
            return
        self.search = self.plan.search_for(self.test)
        self._reset_test_counts()
        self._event("test", ear=self.test.ear, freq=self.test.freq, kind=self.test.kind,
                    ear_pass=self.test.ear_pass, retry=self.test.retry, start_db_hl=self.test.start)
        self._resume_run()

    def _resume_run(self) -> None:
        self.state = "run"
        self._running_screen()
        self.not_before = max(self.not_before, time.perf_counter() + SETTLE_S)
        self._schedule()

    def _familiarising(self) -> bool:
        return bool(self.test and self.test.familiarise and not self.familiarised)

    def _schedule(self) -> None:
        if self.state != "run" or self.pres is not None:
            return
        now = time.perf_counter()
        if self.key_down:
            if now - self.key_since > S.HELD_KEY_PAUSE_S:
                self._pause("The SPACE bar has been held down for a long time. Ask the listener to "
                            "release it (or check that the key is not stuck).")
                return
            self._at(now + KEY_WAIT_S, self._schedule)
            return
        if self.next_dur is None:
            self.next_dur = float(self.rng.uniform(*S.TONE_S))
            lo = max(self.prev_dur or 0.0, self.next_dur)
            self.next_gap = float(self.rng.uniform(lo, max(lo, S.INTERVAL_MAX_S)))
        earliest = self.not_before
        if self.prev_off is not None:
            earliest = max(earliest, self.prev_off + self.next_gap)
        if earliest - LEAD_S > now + 0.005:
            self._at(earliest - LEAD_S, self._schedule)
            return
        t_on = max(earliest, now + LEAD_S)
        t, dur = self.test, self.next_dur
        self.next_dur = self.next_gap = None
        catch = (not self._familiarising() and len(self.search.steps) >= 1
                 and not self.last_was_catch and self.rng.random() < S.CATCH_PROB)
        lvl = self.search.current().level
        nominal = nominal_from_hl(lvl, t.freq, self.ctx.offsets[t.ear][round(t.freq, 2)])
        peak = peak_for(nominal)
        if nominal > NOMINAL_MAX_DB + 1e-9 or peak > MAX_PEAK:
            raise RuntimeError(f"{lvl} dB HL at {t.freq:g} Hz would be {nominal:.1f} nominal dB")
        pb = None if catch else self.engine.play_at(tone(t.freq, dur, peak, t.ear), t_on)
        if catch:
            self.n_catch += 1
            self.catch_in_test += 1
        else:
            self.n_tones += 1
        after = S.RESPONSE_UNTIL_AFTER_OFFSET_S
        if self._familiarising():
            after = max(after, S.RELEASE_WINDOW_S[1])
        self.pres = {"t_on": t_on, "dur": dur, "level_db_hl": lvl, "nominal_db": nominal,
                     "peak": peak, "ascending": self.search.current().ascending, "pb": pb,
                     "catch": catch, "presses": [], "releases": []}
        self.pres["timer"] = self._at(t_on + dur + after + EVAL_MARGIN_S, self._evaluate)

    def _held_at(self, t: float) -> bool:
        return any(d < t and (u is None or u > t) for d, u in self.key_intervals)

    def _void(self, pres: dict, why: str, base: dict, quiet_s: float = 1.0) -> None:
        self.n_void += 1
        self.void_in_test += 1
        self._event("void", why=why, **base)
        self.not_before = max(self.not_before, time.perf_counter() + quiet_s)
        self.prev_off, self.prev_dur = pres["t_on"] + pres["dur"], pres["dur"]

    def _evaluate(self) -> None:
        pres = self.pres
        if pres is None or self.state != "run":
            return
        pb = pres["pb"]
        on = pb.onset_perf() if (pb is not None and pb.onset_perf() is not None) else pres["t_on"]
        off = on + pres["dur"]
        now = time.perf_counter()
        # score only after a good volume reading that began after this presentation ended
        if self.last_ok_read_start is None or self.last_ok_read_start < off:
            if now - off > VOL_WAIT_MAX_S:
                self._pause("The system volume could not be read, so the last tone does not "
                            "count. Check the volume and the sound settings.")
                return
            pres["timer"] = self._at(now + 0.05, self._evaluate)
            return
        self.pres = None
        rep = pb.report() if pb is not None else {}
        base = {"ear": self.test.ear, "freq": self.test.freq, "level_db_hl": pres["level_db_hl"],
                "nominal_db": round(pres["nominal_db"], 2), "ascending": pres["ascending"],
                "catch": pres["catch"], "dur_s": round(pres["dur"], 3),
                "onset_error_ms": rep.get("onset_error_ms"), "underflows": rep.get("underflows"),
                "presses": [round(p - on, 3) for p in pres["presses"]],
                "releases": [round(r - on, 3) for r in pres["releases"]]}
        if pb is not None:
            never = pb.onset_perf() is None
            if pb.late or rep.get("underflows") or pb.aborted or never:
                if never and not getattr(self.engine, "alive", True):
                    raise RuntimeError("the audio output stopped")
                self._void(pres, "audio", base)
                self.audio_voids += 1
                if self.audio_voids >= 3:
                    self.audio_voids = 0
                    self._pause("Tones are not playing correctly (late or broken up) three times in "
                                "a row. Check the audio output and that nothing else is running.")
                    return
                self._schedule()
                return
            self.audio_voids = 0
        a, b = P.response_window(on, off)
        if self._held_at(on):
            self._void(pres, "key held at onset", base)
            self._schedule()
            return
        early = [p for p in pres["presses"] if p < a]
        late = [p for p in pres["presses"] if p > b]
        inside = [p for p in pres["presses"] if a <= p <= b]
        if early:                                 # an anticipation: cannot tell, play it again
            self._false(early[0], during="before window")
            self._void(pres, "press before the response window", base, S.QUIET_AFTER_FALSE_S)
            self._continue_after_tone()
            return
        for p in late:
            self._false(p, during="after window")
        self.prev_off, self.prev_dur = off, pres["dur"]
        if pres["catch"]:
            self.last_was_catch = True
            if inside:
                self.catch_false_in_test += 1
                self._false(inside[0], during="catch trial")
            self._event("catch", responded=bool(inside), **base)
            self._continue_after_tone()
            return
        self.last_was_catch = False
        heard = bool(inside)
        lat = inside[0] - on if heard else None
        if self._familiarising() and heard:
            rel = next((r for r in sorted(pres["releases"]) if r > inside[0]), None)
            if not P.consistent(on, off, inside[0], rel):
                self.fam_tries += 1
                self.n_void += 1
                self._event("familiarisation", ok=False, tries=self.fam_tries, **base)
                if self.fam_tries >= S.FAMILIARISATION_TRIES:
                    self.fam_tries = 0
                    self._message("Once more", INSTRUCTIONS, READ_ALOUD, self._resume_run)
                    return
                self._schedule()
                return
            self.familiarised = True
            self._event("familiarisation", ok=True, **base)
        self.search.record(heard)
        self._event("tone", heard=heard, latency_s=None if lat is None else round(lat, 3), **base)
        self._continue_after_tone()

    def _continue_after_tone(self) -> None:
        if self.search.done:
            self._threshold_done()
            return
        if self.false_since_instruct >= S.FALSE_REINSTRUCT:
            self._reinstruct_now()
            return
        self._schedule()

    def _reinstruct_now(self) -> None:
        self.false_since_instruct = 0
        self._event("reinstruct", why="false responses")
        self._message("Only when you hear a tone", INSTRUCTIONS,
                      "Experimenter: remind the listener to press only when a tone is heard, "
                      "then press Enter", self._resume_run)

    def _threshold_done(self) -> None:
        t, s = self.test, self.search
        extra = {"false_responses": self.false_in_test, "catch_trials": self.catch_in_test,
                 "catch_false": self.catch_false_in_test, "voids": self.void_in_test,
                 "retry": t.retry}
        self._event("threshold", ear=t.ear, freq=t.freq, kind=t.kind, ear_pass=t.ear_pass,
                    threshold_db_hl=s.threshold, status=s.status, **extra)
        if self.false_in_test >= S.FALSE_REPEAT_AT and not t.retry:
            self.plan.repeat(t, s, extra)
        elif self.false_in_test >= S.FALSE_REPEAT_AT:
            self.plan.finish(t, s, extra, unreliable=f"{self.false_in_test} false responses "
                                                     f"again after re-instruction")
        else:
            self.plan.finish(t, s, extra)
        self._save("running")
        self._next_test()

    def _false(self, t: float, during: str = "between tones") -> None:
        self.n_false += 1
        self.false_in_test += 1
        self.false_since_instruct += 1
        self.not_before = max(self.not_before, t + S.QUIET_AFTER_FALSE_S)
        self._event("false_response", during=during, press_t=round(t - self.t0, 4))

    # -- keys -----------------------------------------------------------------------------------------
    def on_key(self, name: str, down: bool, t: float) -> None:
        try:
            self._key(name, down, t)
        except Exception as e:
            self._fatal(e)

    def _key(self, name: str, down: bool, t: float) -> None:
        if name == "space":
            if down:
                if self.key_down:
                    return
                self.key_down, self.key_since = True, t
                self.key_intervals.append([t, None])
                if self.state != "run":
                    return
                p = self.pres
                if p is not None and t < p["t_on"]:          # pressed before the tone started
                    if p["pb"] is not None:
                        self.engine.abort(p["pb"])
                    p["timer"].stop()
                    self.pres = None
                    self._false(t, during="before onset")
                    self._void(p, "press before onset", {"level_db_hl": p["level_db_hl"],
                                                         "catch": p["catch"]},
                               S.QUIET_AFTER_FALSE_S)
                    self.prev_off = self.prev_dur = None
                    if self.false_since_instruct >= S.FALSE_REINSTRUCT:
                        self._cancel_timers()
                        self._reinstruct_now()
                    else:
                        self._schedule()
                elif p is not None:
                    p["presses"].append(t)
                else:
                    self._false(t)
                    if self.false_since_instruct >= S.FALSE_REINSTRUCT:
                        self._cancel_timers()
                        self._reinstruct_now()
            else:
                if not self.key_down:
                    return
                self.key_down = False
                if self.key_intervals and self.key_intervals[-1][1] is None:
                    self.key_intervals[-1][1] = t
                self.not_before = max(self.not_before, t + S.KEY_QUIET_S)
                if self.state == "run" and self.pres is not None:
                    self.pres["releases"].append(t)
            return
        if not down:
            return
        if self.state == "message" and name == "return":
            v = self._read_now()
            if not self._volume_ok(v):
                self._event("volume_wrong_at_message", volume=v)
                self._paint("Volume", self._volume_text(v),
                            "Restore it, then press Enter again", "warn")
                return
            then, self.after_message = self.after_message, None
            if then:
                then()
        elif self.state == "message" and name == "escape":
            self.held_message = (self.win.canvas.title, self.win.canvas.body,
                                 self.win.canvas.footer, self.win.canvas.style, self.after_message)
            self.state = "pause"
            self._event("pause", why="experimenter, at a message")
            self._paint("Paused", "Paused by the experimenter.", "C goes back  ·  Q saves and stops",
                        "warn")
        elif self.state == "run" and name == "escape":
            self._pause("Paused by the experimenter.")
        elif self.state == "pause" and name == "c":
            if self.held_message is not None:
                title, body, footer, style, then = self.held_message
                self.held_message = None
                self._message(title, body, footer, then, style)
            else:
                self._continue()
        elif self.state == "pause" and name == "q":
            self._finish("stopped")
        elif self.state == "end" and name in ("return", "escape"):
            self.on_exit(self.status)

    # -- pause, volume, end ----------------------------------------------------------------------------
    def _abort_current(self) -> None:
        self._cancel_timers()
        p, self.pres = self.pres, None
        if p is not None:
            if p["pb"] is not None:
                self.engine.abort(p["pb"])
            self.n_void += 1
            self.void_in_test += 1
            try:
                self._event("void", why="paused", level_db_hl=p["level_db_hl"], catch=p["catch"])
            except Exception:
                pass
        self.next_dur = self.next_gap = None
        self.prev_off = self.prev_dur = None

    def _pause(self, why: str) -> None:
        self._abort_current()
        self.state = "pause"
        self._event("pause", why=why)
        self._paint("Paused", why, "C continues  ·  Q saves and stops", "warn")

    def _continue(self) -> None:
        v = self._read_now()
        if not self._volume_ok(v):
            self._paint("Paused", self._volume_text(v),
                        "Restore it, then C continues  ·  Q saves and stops", "warn")
            return
        self._event("continue")
        if self.test is None or self.search is None:
            self._next_test()
        else:
            self._resume_run()

    def _check_volume(self) -> None:
        now = time.perf_counter()
        if self.vol_future is not None and self.vol_future.done():
            try:
                v = self.vol_future.result() or {}
            except Exception as e:
                v = {"readable": False, "error": repr(e)}
            started, self.vol_future = self.vol_started, None
            if self._volume_ok(v):
                self.last_ok_read_start = started
            elif self.state == "run":
                self._event("volume_changed", volume=v)
                self._pause(self._volume_text(v) + " The tone that was playing does not count.")
        if self.vol_future is None and self.state != "end" and now - self.last_submit >= READ_EVERY_S:
            try:
                self.vol_started = self.last_submit = now
                self.vol_future = self.pool.submit(self.read_volume)
            except RuntimeError:                  # the pool is closed: the screen is over
                self.guard.stop()

    def _save(self, status: str) -> None:
        self.ctx.save({"status": status, "plan": self.plan.summary(),
                       "n_tones": self.n_tones, "n_catch_trials": self.n_catch,
                       "n_false_responses": self.n_false, "n_void": self.n_void,
                       "minutes": round((time.perf_counter() - self.t0) / 60, 2)})

    def _finish(self, status: str) -> None:
        self._abort_current()
        self.guard.stop()
        self.status, self.state = status, "end"
        self._save(status)
        self._event("end", status=status)
        if status == "complete":
            self._paint("Done", "Thank you. This part is finished.", "Experimenter: press Enter")
        else:
            self._paint("Stopped", "The screen was stopped; what was measured is saved.",
                        "Experimenter: press Enter")

    def _fatal(self, e: Exception) -> None:
        self.status, self.state = "error", "end"
        try:
            self._abort_current()
            self.guard.stop()
        except Exception:
            pass
        for fn in (lambda: self._event("error", error=repr(e), trace=traceback.format_exc()),
                   lambda: self._save("error")):
            try:
                fn()
            except Exception:
                pass
        self._paint("Stopped", f"Something went wrong: {e}", "Experimenter: press Enter", "warn")


def run_screen(engine, ctx: Context, volume_reader, fullscreen: bool = True, hook=None) -> str:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    holder = {}
    out = {"status": "stopped"}

    def on_key(name, down, t):
        if "s" in holder:
            holder["s"].on_key(name, down, t)

    win = Window(on_key)

    def done(status):
        out["status"] = status
        win.allow_close = True
        win.close()
        QtCore.QTimer.singleShot(100, app.quit)

    s = Screening(win, engine, ctx, done, volume_reader)
    holder["s"] = s
    if hook is not None:
        hook(s)
    if fullscreen:
        win.showFullScreen()
    else:
        win.resize(1100, 720)
        win.show()
    QtCore.QTimer.singleShot(300, lambda: s._fire(None, s.start))   # an error at start ends cleanly
    app.exec()
    s.guard.stop()
    s._cancel_timers()
    s.pool.shutdown(wait=False)
    return out["status"]
