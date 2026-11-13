"""The booth app: runs `Experiment.script()` with sound, a full-screen display and the keyboard.

Timing of one trial
-------------------
    neutral screen --ready_ms--> sequence 1 (left box lit) --roved 450-550 ms--> sequence 2 (right
    box lit) --> answer window (keys before it are logged and ignored) --> feedback (symbol + word)
    --> blank --> next trial.

The silence between two trials is at least ready + feedback + blank = 1.65 s plus the response
time, more than three times the gap inside a trial, and the screen changes in between, so the two
sequences of one trial cannot be confused with those of the next.

The sound is scheduled on the audio engine's sample clock (see audio.py) and the box changes are
scheduled on the same clock, so the boxes follow the sound to within a display frame. Key
presses are stamped on arrival; the response time is measured from the true end of sequence 2
(its DAC time), not from when the program asked for it.

Before every trial the system volume, mute state and audio stream are checked in a worker thread
during the neutral screen. If the volume is not the one the session started with, the stream has
stalled, or the output is muted, the trial is not played: an alert asks for the experimenter.

Keys: left arrow or 1 = first, right arrow or 2 = second (keypad too, and on macOS by physical
key so a Persian layout cannot break them), P = pause, Esc = experimenter menu (Q quits safely).
"""
from __future__ import annotations

import concurrent.futures
import platform
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from .audio import AudioEngine
from .config import Config
from .experiment import Experiment, Play, PlayResult, Screen, ScreenResult
from .logbook import SessionWriter, now_iso
from .system import volume_settings
from .text import T
from .ui import FormPage, Stage, View

MAC_KEYS = {35: "p", 12: "q", 16: "y", 45: "n", 15: "r", 8: "c", 18: "1", 19: "2", 83: "1",
            84: "2", 49: "space", 36: "return", 76: "return", 53: "escape", 123: "left",
            124: "right", 51: "backspace"}
QT_KEYS = {Qt.Key.Key_Left: "left", Qt.Key.Key_Right: "right", Qt.Key.Key_Space: "space",
           Qt.Key.Key_Return: "return", Qt.Key.Key_Enter: "return", Qt.Key.Key_Escape: "escape",
           Qt.Key.Key_Backspace: "backspace"}
RESPONSE = {"left": 1, "1": 1, "right": 2, "2": 2}
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def key_name(e: QtGui.QKeyEvent) -> str:
    if platform.system() == "Darwin" and e.nativeVirtualKey() in MAC_KEYS:
        return MAC_KEYS[e.nativeVirtualKey()]
    if e.key() in QT_KEYS:
        return QT_KEYS[e.key()]
    t = e.text().translate(_DIGITS).lower()
    return t if t and t.isprintable() else f"key{e.key()}"


@dataclass
class Guard:
    locked_volume: Optional[int]          # None: not readable on this machine
    device: str


class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Listening experiment")
        self.stack = QtWidgets.QStackedLayout(self)
        self.stage = Stage()
        self.form = FormPage()
        self.stack.addWidget(self.stage)
        self.stack.addWidget(self.form)
        self.runner: Optional["Runner"] = None
        self.setStyleSheet("background: #0E1116;")

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        if self.runner is not None:
            self.runner.on_key(e)

    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        if self.runner is not None and not self.runner.closing:
            self.runner.quit("window_closed")
        e.accept()


class Runner(QtCore.QObject):
    def __init__(self, win: Window, exp: Experiment, engine: Optional[AudioEngine],
                 writer: SessionWriter, guard: Guard, on_exit: Callable[[str], None]):
        super().__init__()
        self.win, self.exp, self.engine, self.w = win, exp, engine, writer
        self.cfg: Config = exp.cfg
        self.guard, self.on_exit = guard, on_exit
        self.gen = exp.script()
        self.started = False
        self.closing = False
        self.state = "idle"
        self.action = None
        self.timers: List[QtCore.QTimer] = []
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.poll = QtCore.QTimer(self)
        self.poll.setTimerType(Qt.TimerType.PreciseTimer)
        self.poll.timeout.connect(self._tick)
        self.poll.start(5)
        self.overlay_return: Optional[str] = None
        self.pause_pending = False
        self.t_overlay = 0.0
        win.runner = self
        win.form.submitted.connect(self._form_done)

    # -- driving the script --------------------------------------------------------------------
    def start(self) -> None:
        self._advance(None)

    def _advance(self, result) -> None:
        self._cancel_timers()
        try:
            if not self.started:
                self.started = True
                action = next(self.gen)
            else:
                action = self.gen.send(result)
        except StopIteration:
            self._exit("complete")
            return
        except Exception as e:                           # never lose a session silently
            self._fatal(e)
            return
        self.action = action
        if isinstance(action, Screen):
            self._screen(action)
        elif isinstance(action, Play):
            self._play(action)

    # -- screens -----------------------------------------------------------------------------
    def _screen(self, s: Screen) -> None:
        self.t_shown = time.perf_counter()
        self.w.event("screen", name=s.name, kind=s.kind)
        if s.kind == "form":
            self.state = "form"
            self.win.form.build(s.title, s.body, s.fields, self.win.height())
            self.win.stack.setCurrentIndex(1)
            return
        self.win.stack.setCurrentIndex(0)
        self.state = "screen"
        footer = ""
        if s.keys == ("space",):
            footer = T["space_hint"]
        elif s.kind == "end":
            footer = "Experimenter: press Esc to close"
        self.view = View(kind="message" if s.kind not in ("rest", "end") else s.kind,
                         title=s.title, body=s.body, footer=footer, figure=s.figure,
                         data=dict(s.data), progress=s.progress)
        if s.kind == "alert":
            self.view.kind = "message"
            self.view.overlay = ("alert", s.title, s.body)
        if s.min_s > 0 and s.keys == ("space",):
            self.view.footer = ""
            self._at(self.t_shown + s.min_s, self._screen_ready)
        self.win.stage.show_view(self.view)

    def _screen_ready(self) -> None:
        if self.state == "screen":
            self.view.footer = T["space_hint"]
            self.win.stage.show_view(self.view)

    def _form_done(self, values: dict) -> None:
        if self.state != "form":
            return
        self.win.stack.setCurrentIndex(0)
        self._advance(ScreenResult("return", values, time.perf_counter() - self.t_shown))

    # -- trials ------------------------------------------------------------------------------
    def _play(self, play: Play) -> None:
        self.play = play
        self.state = "prepare"
        self.win.stack.setCurrentIndex(0)
        self.res = PlayResult(wall_start=now_iso())
        self.keys: List[dict] = []
        self.pb = None
        self.t_ready = time.perf_counter()
        self.view = View(kind="trial", label=play.label, progress=play.progress)
        self.win.stage.show_view(self.view)
        self.sys_future = self.pool.submit(volume_settings)
        self.res.system = {}

    def _prepare_tick(self) -> None:
        f = self.sys_future
        waited = time.perf_counter() - self.t_ready
        if not f.done() and waited < 2.5:
            return
        sysinfo = f.result() if f.done() else {"readable": False, "timeout": True}
        sysinfo = dict(sysinfo, device=self.guard.device)
        self.res.system = sysinfo
        problem = self._guard_problem(sysinfo)
        if not problem:
            try:
                self._schedule()
                return
            except Exception as e:                  # the stream refused the buffer
                problem = f"The audio output failed: {e}"
        self.state = "blocked"
        self.w.event("guard_blocked", problem=problem, system=sysinfo)
        self.view.overlay = ("alert", "Please call the experimenter",
                             problem + "\n\nExperimenter: fix it, then press Space.")
        self.win.stage.show_view(self.view)

    def _guard_problem(self, s: dict) -> str:
        if self.engine is not None and not self.engine.alive:
            return f"The audio output stopped ({self.engine.error})."
        if self.engine is not None and self.engine._last is not None and \
                time.perf_counter() - self.engine._last[2] > 0.5:
            return "The audio output stalled (were the headphones unplugged?)."
        if self.guard.locked_volume is not None:
            if not s.get("readable"):
                return "The system volume could not be read."
            if s.get("output_volume") != self.guard.locked_volume:
                return (f"The system volume changed from {self.guard.locked_volume} to "
                        f"{s.get('output_volume')}. Set it back to {self.guard.locked_volume}.")
            if s.get("muted"):
                return "The sound output is muted. Unmute it."
        return ""

    def _schedule(self) -> None:
        cfg, tr = self.cfg, self.play.trial
        now = time.perf_counter()
        t_first = max(self.t_ready + cfg.ready_ms / 1000.0, now + cfg.schedule_lead_ms / 1000.0)
        t_buf = t_first - tr.interval_on_s[0]
        if self.engine is not None:
            self.pb = self.engine.play_at(self.play.buffer, t_buf)
        self.t_buf = t_buf
        self.state = "playing"
        on1, off1 = t_buf + tr.interval_on_s[0], t_buf + tr.interval_off_s[0]
        on2, off2 = t_buf + tr.interval_on_s[1], t_buf + tr.interval_off_s[1]
        self.sched = {"on1": on1, "off1": off1, "on2": on2, "off2": off2}
        self.visual = {}
        self.view.prompt, self.view.prompt_style = T["prompt_listen"], "muted"
        self._at(on1, lambda: self._panel(0, "active", "on1"))
        self._at(off1, lambda: self._panel(0, "idle", "off1", caption=0))
        self._at(on2, lambda: self._panel(1, "active", "on2"))
        self._at(off2, lambda: self._panel(1, "idle", "off2", caption=1))
        self._at(off2, self._sound_over)

    def _panel(self, i: int, state: str, tag: str, caption: Optional[int] = None) -> None:
        t = time.perf_counter()
        self.visual[tag] = (t - self.sched[tag]) * 1000.0
        self.view.panels[i] = state
        if caption is not None and self.play.captions[caption]:
            self.view.captions[caption] = self.play.captions[caption]
        self.win.stage.show_view(self.view)
        self.win.stage.repaint()

    def _sound_over(self) -> None:
        if self.state != "playing":
            return
        if self.play.mode == "listen":
            self.state = "listen_tail"
            self._at(time.perf_counter() + 0.9, lambda: self._finish_trial(None))
            return
        self.state = "respond"
        self.t_open = time.perf_counter()
        self.view.prompt, self.view.prompt_style = T["prompt_respond"], "text"
        self.win.stage.show_view(self.view)
        self._at(self.t_open + self.cfg.reminder_after_s, self._remind)
        self._at(self.t_open + self.cfg.autopause_after_s, lambda: self._interrupt("timeout"))

    def _remind(self) -> None:
        if self.state == "respond":
            self.res.reminder_shown = True
            self.view.prompt, self.view.prompt_style = T["prompt_reminder"], "warn"
            self.win.stage.show_view(self.view)

    def _respond(self, resp: int, key: str, t_key: float) -> None:
        self.state = "feedback"
        self._cancel_timers()
        tr = self.play.trial
        self.res.response, self.res.key = resp, key
        self.res.wall_response = now_iso()
        off2, on2 = self.sched["off2"], self.sched["on2"]
        if self.pb is not None and self.pb.onset_dac is not None:
            t0 = self.pb.onset_perf()
            off2, on2 = t0 + tr.interval_off_s[1], t0 + tr.interval_on_s[1]
        self.res.rt_ms = (t_key - off2) * 1000.0
        self.res.rt_from_onset2_ms = (t_key - on2) * 1000.0
        correct = resp == tr.target_interval
        if self.play.feedback:
            self.view.panels[resp - 1] = "good" if correct else "bad"
            self.view.prompt = ("✓  " + T["correct"]) if correct else ("✗  " + T["wrong"])
            self.view.prompt_style = "good" if correct else "bad"
            self.win.stage.show_view(self.view)
            self._at(t_key + self.cfg.feedback_ms / 1000.0, self._blank)
        else:
            self._blank()

    def _blank(self) -> None:
        self.state = "blank"
        self.view.panels = ["idle", "idle"]
        self.view.prompt = ""
        self.win.stage.show_view(self.view)
        self._at(time.perf_counter() + self.cfg.blank_ms / 1000.0, lambda: self._finish_trial(None))

    def _finish_trial(self, reason: Optional[str]) -> None:
        res = self.res
        if self.pb is not None:
            if not self.pb.finished and reason:
                self.engine.abort(self.pb)
            res.audio = self.pb.report()
            if reason is None and self.pb.late:
                reason = "audio_late"
            elif reason is None and res.audio["underflows"]:
                reason = "audio_underflow"
        res.keys = self.keys
        res.early_keys = sum(1 for k in self.keys if k["state"] in ("prepare", "playing"))
        res.visual = {"lag1_ms": self.visual.get("on1"), "lag2_ms": self.visual.get("on2"),
                      **{f"{k}_ms": v for k, v in self.visual.items()}}
        if reason:
            res.status, res.reason = "aborted", reason
        if self.pause_pending and not reason:
            self.pause_pending = False
            self._overlay("pause", T["paused_title"], T["paused_body"], then=("advance", res))
            return
        self._advance(res)

    def _interrupt(self, reason: str) -> None:
        """Pause or operator menu during a trial: stop the sound; the trial will be repeated."""
        if self.state in ("prepare", "blocked", "playing", "respond", "listen_tail"):
            self._cancel_timers()
            if self.pb is not None:
                self.engine.abort(self.pb)
            self.state = "interrupted"
            self.view.panels = ["idle", "idle"]
            self.view.prompt = ""
            self.res.status, self.res.reason = "aborted", reason
            if self.pb is not None:
                self._at(time.perf_counter() + 0.06, lambda: self._interrupted(reason))
            else:
                self._interrupted(reason)
        elif self.state in ("feedback", "blank"):
            self.pause_pending = True

    def _interrupted(self, reason: str) -> None:
        res = self.res
        if self.pb is not None:
            res.audio = self.pb.report()
        res.keys = self.keys
        if reason == "operator":
            self._overlay("operator", T["operator_title"], T["operator_body"],
                          then=("advance", res))
        else:
            self._overlay("pause", T["paused_title"], T["paused_body"], then=("advance", res))

    # -- overlays ------------------------------------------------------------------------------
    def _overlay(self, style: str, title: str, body: str, then) -> None:
        self.overlay_state = (self.state, then)
        self.state = "overlay"
        self.t_overlay = time.perf_counter()
        self.w.event("overlay_on", style=style)
        self.view.overlay = (style, title, body)
        self.win.stack.setCurrentIndex(0)
        self.win.stage.show_view(self.view)

    def _overlay_off(self) -> None:
        prev, then = self.overlay_state
        self.w.event("overlay_off", seconds=time.perf_counter() - self.t_overlay)
        self.view.overlay = None
        self.win.stage.show_view(self.view)
        if then[0] == "advance":
            self._advance(then[1])
        elif then[0] == "restore":
            self.state = prev
            if prev == "form":
                self.win.stack.setCurrentIndex(1)

    # -- keys ----------------------------------------------------------------------------------
    def on_key(self, e: QtGui.QKeyEvent) -> None:
        t = time.perf_counter()
        name = key_name(e)
        rec = {"key": name, "t_perf": t, "qt_ts_ms": e.timestamp(), "state": self.state,
               "autorepeat": e.isAutoRepeat()}
        if self.state in ("prepare", "blocked", "playing", "respond", "listen_tail", "feedback",
                          "blank"):
            rec["t_rel_on1_ms"] = (t - self.sched["on1"]) * 1000.0 if hasattr(self, "sched") \
                and self.state != "prepare" else None
            self.keys.append(rec)
        else:
            self.w.event("key", **rec)
        if e.isAutoRepeat():
            return
        st = self.state
        if st == "done":
            if name in ("q", "escape"):
                self.quit("after_error")
            return
        if st == "overlay":
            style = self.view.overlay[0] if self.view.overlay else ""
            if name == "q" and style == "operator":
                self.quit("operator_quit")
            elif name == "space" and time.perf_counter() - self.t_overlay > 0.5:
                self._overlay_off()
            return
        if name == "escape" and st not in ("done",):
            if isinstance(self.action, Screen) and self.action.kind == "end":
                self._advance(ScreenResult("escape", {}, time.perf_counter() - self.t_shown))
                return
            if isinstance(self.action, Play):
                self._interrupt("operator")
            else:
                self._overlay("operator", T["operator_title"], T["operator_body"],
                              then=("restore", None))
            return
        if name == "p" and st != "form":
            if isinstance(self.action, Play):
                self._interrupt("pause")
            else:
                self._overlay("pause", T["paused_title"], T["paused_body"], then=("restore", None))
            return
        if st == "blocked" and name == "space":
            self.view.overlay = None
            self._play(self.play)
            return
        if st == "screen":
            s: Screen = self.action
            if name in s.keys and time.perf_counter() - self.t_shown >= s.min_s:
                self._advance(ScreenResult(name, {}, time.perf_counter() - self.t_shown))
            return
        if st == "respond" and name in RESPONSE:
            self._respond(RESPONSE[name], name, t)

    # -- housekeeping --------------------------------------------------------------------------
    def _tick(self) -> None:
        if self.state == "prepare":
            self._prepare_tick()
        elif self.state == "playing" and self.pb is not None:
            if self.pb.late:
                self._finish_trial("audio_late")
            elif self.engine is not None and not self.engine.alive:
                self._finish_trial("audio_error")
        elif self.state == "blocked" and self.view.overlay is not None:
            pass

    def _at(self, t_perf: float, fn) -> None:
        tm = QtCore.QTimer(self)
        tm.setSingleShot(True)
        tm.setTimerType(Qt.TimerType.PreciseTimer)
        tm.timeout.connect(fn)
        tm.start(max(0, int(round((t_perf - time.perf_counter()) * 1000.0))))
        self.timers.append(tm)

    def _cancel_timers(self) -> None:
        for tm in self.timers:
            tm.stop()
            tm.deleteLater()
        self.timers = []

    def quit(self, why: str) -> None:
        if self.closing:
            return
        self.closing = True
        self._cancel_timers()
        if self.engine is not None:
            self.engine.silence()
        if isinstance(self.action, Play) and self.res.status == "aborted":
            try:
                self.gen.send(self.res)            # let the script log the interrupted trial
            except Exception:
                pass
        try:
            self.gen.close()
        except Exception:
            pass
        self.w.event("quit", why=why)
        if self.w.meta.get("status") == "running":
            self.w.close("quit" if why != "window_closed" else "window_closed")
        self._exit(why)

    def _fatal(self, e: Exception) -> None:
        tb = traceback.format_exc()
        try:
            self.w.event("fatal", error=repr(e), traceback=tb)
            if self.w.meta.get("status") == "running":
                self.w.close("error")
        finally:
            self.state = "done"
            self.view = View(kind="message", title="Something went wrong",
                             body="Please call the experimenter.\n\n" + repr(e),
                             footer="Experimenter: press Esc then Q")
            self.win.stage.show_view(self.view)
            self.closing = False

    def _exit(self, why: str) -> None:
        self.state = "done"
        self.closing = True
        self.poll.stop()
        self.pool.shutdown(wait=False)
        self.on_exit(why)


def run_app(exp: Experiment, engine: Optional[AudioEngine], writer: SessionWriter, guard: Guard,
            fullscreen: bool = True, screen_index: int = 0) -> str:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = Window()
    outcome = {"why": ""}

    def on_exit(why):
        outcome["why"] = why
        QtCore.QTimer.singleShot(200, app.quit)

    runner = Runner(win, exp, engine, writer, guard, on_exit)
    screens = app.screens()
    scr = screens[screen_index] if screen_index < len(screens) else app.primaryScreen()
    writer.update_meta(display={"name": scr.name(), "size": [scr.size().width(), scr.size().height()],
                                "dpr": scr.devicePixelRatio(), "refresh_hz": scr.refreshRate()})
    win.setGeometry(scr.geometry())
    if fullscreen:
        win.showFullScreen()
        app.setOverrideCursor(Qt.CursorShape.BlankCursor)
    else:
        win.resize(1280, 800)
        win.show()
    win.raise_()
    win.activateWindow()
    QtCore.QTimer.singleShot(300, runner.start)
    app.exec()
    app.restoreOverrideCursor()
    return outcome["why"]
