"""The booth app for tcmr: full-screen, three boxes that light up with the three sounds.

Reuses tsync's audio engine (one stream, sample-scheduled onsets, DAC stamps), its questionnaire
form and its colours. Keys: 1/2/3 or left/down/right arrows to answer, 1-5 for ratings, P to pause
(the trial is played again later), Esc for the experimenter menu (Q saves and quits). Before every
trial the system volume is read in a worker thread; if it differs from the session's starting
value, or the output is muted or stalled, the trial is not played and the experimenter is called.
"""
from __future__ import annotations

import concurrent.futures
import platform
import time
import traceback
from typing import Callable, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen

from tsync.system import volume_settings
from tsync.ui import C, FormPage, _font

from .session import Play, Result, Screen

MAC = {18: "1", 19: "2", 20: "3", 21: "4", 23: "5", 83: "1", 84: "2", 85: "3", 86: "4", 87: "5",
       123: "left", 125: "down", 124: "right", 49: "space", 36: "return", 76: "return",
       53: "escape", 35: "p", 12: "q"}
QTK = {Qt.Key.Key_Left: "left", Qt.Key.Key_Down: "down", Qt.Key.Key_Right: "right",
       Qt.Key.Key_Space: "space", Qt.Key.Key_Return: "return", Qt.Key.Key_Enter: "return",
       Qt.Key.Key_Escape: "escape"}
ANSWER = {"1": 1, "2": 2, "3": 3, "left": 1, "down": 2, "right": 3}
_DIG = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def key_name(e) -> str:
    if platform.system() == "Darwin" and e.nativeVirtualKey() in MAC:
        return MAC[e.nativeVirtualKey()]
    if e.key() in QTK:
        return QTK[e.key()]
    t = e.text().translate(_DIG).lower()
    return t if t else f"key{e.key()}"


def col(name, a=255):
    c = QColor(C.get(name, name))
    c.setAlpha(a)
    return c


class Canvas(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.v = {"kind": "blank"}

    def show(self, **v):
        self.v = v
        self.update()

    def _t(self, p, r, s, px, color, w=QFont.Weight.Normal, align=Qt.AlignmentFlag.AlignLeft):
        p.setFont(_font(px, w))
        p.setPen(col(color))
        p.drawText(r, int(align | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap), s)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h, v = self.width(), self.height(), self.v
        u = h / 100
        p.fillRect(self.rect(), col("bg"))
        if v.get("kind") == "message":
            cw = min(w * 0.62, 118 * u)
            x = (w - cw) / 2
            self._t(p, QRectF(x, h * 0.16, cw, 12 * u), v.get("title", ""), 4.4 * u, "text",
                    QFont.Weight.Medium)
            self._t(p, QRectF(x, h * 0.16 + 9 * u, cw, 55 * u), v.get("body", ""), 2.6 * u, "text")
            self._t(p, QRectF(0, h * 0.86, w, 5 * u), v.get("footer", ""), 2.1 * u, "muted",
                    align=Qt.AlignmentFlag.AlignHCenter)
        elif v.get("kind") == "trial":
            self._t(p, QRectF(4 * u, 3.5 * u, w / 2, 4 * u), v.get("label", ""), 1.9 * u, "muted")
            states = v.get("boxes", [])
            n = len(states)
            gap, bh = min(w * 0.05, 7 * u), 26 * u
            bw = min(30 * u, (w * 0.84 - (n - 1) * gap) / max(n, 1))
            x0 = (w - n * bw - (n - 1) * gap) / 2
            y = h * 0.42 - bh / 2
            hints = v.get("hints", [])
            for i, st in enumerate(states):
                r = QRectF(x0 + i * (bw + gap), y, bw, bh)
                fill, edge, lw = col("panel"), col("edge"), 0.25 * u
                if st == "active":
                    fill, edge, lw = col("active"), col("active_edge"), 0.45 * u
                elif st in ("good", "bad", "mark"):
                    c = "good" if st in ("good", "mark") else "bad"
                    fill, edge, lw = col(c, 40), col(c), 0.45 * u
                p.setPen(QPen(edge, lw))
                p.setBrush(fill)
                p.drawRoundedRect(r, 2.2 * u, 2.2 * u)
                self._t(p, QRectF(r.x(), r.y() + bh * 0.18, bw, 12 * u), str(i + 1), 10 * u,
                        "text" if st != "idle" else "muted", QFont.Weight.Medium,
                        Qt.AlignmentFlag.AlignHCenter)
                if i < len(hints):
                    self._t(p, QRectF(r.x(), r.y() + bh * 0.72, bw, 5 * u), hints[i], 2.0 * u,
                            "muted", align=Qt.AlignmentFlag.AlignHCenter)
            if v.get("prompt"):
                self._t(p, QRectF(0, y + bh + 7 * u, w, 6 * u), v["prompt"], 2.8 * u,
                        v.get("style", "text"), QFont.Weight.Medium, Qt.AlignmentFlag.AlignHCenter)
            if v.get("sub"):
                self._t(p, QRectF(0, y + bh + 13 * u, w, 6 * u), v["sub"], 2.2 * u, "muted",
                        align=Qt.AlignmentFlag.AlignHCenter)
        if v.get("progress") is not None:
            bw = min(w * 0.36, 60 * u)
            r = QRectF((w - bw) / 2, h - 5.5 * u, bw, 0.55 * u)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(col("edge"))
            p.drawRoundedRect(r, 0.3 * u, 0.3 * u)
            p.setBrush(col("accent", 200))
            p.drawRoundedRect(QRectF(r.x(), r.y(), bw * max(0, min(1, v["progress"])), r.height()),
                              0.3 * u, 0.3 * u)
        if v.get("overlay"):
            title, body, style = v["overlay"]
            p.fillRect(self.rect(), QColor(0, 0, 0, 185))
            cw, ch = min(w * 0.5, 80 * u), 30 * u
            r = QRectF((w - cw) / 2, (h - ch) / 2, cw, ch)
            p.setPen(QPen(col("warn" if style == "alert" else "edge"), 0.3 * u))
            p.setBrush(col("panel"))
            p.drawRoundedRect(r, 2.4 * u, 2.4 * u)
            self._t(p, QRectF(r.x() + 4 * u, r.y() + 4 * u, cw - 8 * u, 8 * u), title, 3.6 * u,
                    "warn" if style == "alert" else "text", QFont.Weight.Medium)
            self._t(p, QRectF(r.x() + 4 * u, r.y() + 12 * u, cw - 8 * u, 16 * u), body, 2.5 * u,
                    "text")
        p.end()


class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Listening experiment")
        self.stack = QtWidgets.QStackedLayout(self)
        self.canvas, self.form = Canvas(), FormPage()
        self.stack.addWidget(self.canvas)
        self.stack.addWidget(self.form)
        self.runner = None
        self.setStyleSheet(f"background: {C['bg']};")

    def keyPressEvent(self, e):
        if self.runner:
            self.runner.on_key(e)

    def closeEvent(self, e):
        if self.runner and not self.runner.closing:
            self.runner.quit("window_closed")
        e.accept()


class Runner(QtCore.QObject):
    def __init__(self, win, exp, engine, locked_volume, on_exit: Callable[[str], None],
                 read_volume=volume_settings, visual_offset_s: float = 0.0, set_volume=None):
        super().__init__()
        self.win, self.exp, self.eng, self.cfg = win, exp, engine, exp.cfg
        self.locked, self.on_exit, self.read_volume = locked_volume, on_exit, read_volume
        self.vis = float(visual_offset_s)        # extra output delay (Bluetooth) for the boxes
        self.set_volume = set_volume             # if given, drift is corrected, not alarmed
        self.gen = exp.script()
        self.started = self.closing = False
        self.state, self.action, self.timers = "idle", None, []
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.poll = QtCore.QTimer(self)
        self.poll.timeout.connect(self._tick)
        self.poll.start(5)
        self.view = {}
        win.runner = self
        win.form.submitted.connect(self._form_done)

    def start(self):
        self._advance(None)

    def _advance(self, result):
        self._cancel()
        try:
            act = self.gen.send(result) if self.started else next(self.gen)
            self.started = True
        except StopIteration:
            self._exit("complete")
            return
        except Exception as e:
            self._fatal(e)
            return
        self.action = act
        (self._screen if isinstance(act, Screen) else self._play)(act)

    def _paint(self, **kw):
        self.view.update(kw)
        self.win.canvas.show(**self.view)

    # -- screens ---------------------------------------------------------------------------------
    def _screen(self, s: Screen):
        self.t_shown = time.perf_counter()
        self.exp.w.event("screen", name=s.name)
        if s.kind == "form":
            self.state = "form"
            self.win.form.build(s.title, s.body, s.fields, self.win.height())
            self.win.stack.setCurrentIndex(1)
            return
        self.win.stack.setCurrentIndex(0)
        self.state = "screen"
        footer = "Press Space to continue" if s.keys == ("space",) else \
            ("Experimenter: press Esc to close" if s.kind == "end" else "")
        self.view = {"kind": "message", "title": s.title, "body": s.body,
                     "footer": "" if s.min_s else footer, "progress": s.progress}
        self.win.canvas.show(**self.view)
        if s.min_s:
            self._at(self.t_shown + s.min_s, lambda: self.state == "screen" and self._paint(footer=footer))

    def _form_done(self, values):
        if self.state == "form":
            self.win.stack.setCurrentIndex(0)
            self._advance({"key": "return", "values": values})

    # -- trials ------------------------------------------------------------------------------
    def _play(self, play: Play):
        self.play, self.pb, self.res, self.keys = play, None, Result(), []
        self.state, self.t_ready = "prepare", time.perf_counter()
        n = play.trial.n_intervals
        hints = ["←  or  1", "↓  or  2", "→  or  3"] if play.mode == "respond" else []
        self.view = {"kind": "trial", "label": play.label, "progress": play.progress,
                     "boxes": ["idle"] * n, "hints": hints, "prompt": ""}
        self.win.stack.setCurrentIndex(0)
        self.win.canvas.show(**self.view)
        self.fut = self.pool.submit(self.read_volume)

    def _prepare(self):
        if not self.fut.done() and time.perf_counter() - self.t_ready < 2.5:
            return
        sysinfo = self.fut.result() if self.fut.done() else {"readable": False}
        if self.set_volume is not None and self.locked is not None and sysinfo.get("readable") \
                and (abs((sysinfo.get("output_volume") or 0) - self.locked) > 1 or sysinfo.get("muted")):
            was = dict(sysinfo)
            sysinfo = self.set_volume(self.locked)          # put it back before this trial
            sysinfo = dict(sysinfo, corrected_from=was.get("output_volume"),
                           was_muted=was.get("muted"))
            self.exp.w.event("volume_reset", was=was.get("output_volume"), muted=was.get("muted"),
                             now=sysinfo.get("output_volume"), target=self.locked)
        self.res.system = sysinfo
        problem = ""
        if self.eng is not None and not self.eng.alive:
            problem = f"The audio output stopped ({self.eng.error})."
        elif self.locked is not None and (abs((sysinfo.get("output_volume") or -99) - self.locked) > 1
                                          or sysinfo.get("muted")):
            problem = (f"The system volume is {sysinfo.get('output_volume')} (muted "
                       f"{sysinfo.get('muted')}); it must be {self.locked}.")
        if not problem:
            try:
                return self._schedule()
            except Exception as e:
                problem = f"The sound could not be played: {e}"
        self.state = "blocked"
        self.exp.w.event("blocked", problem=problem)
        self._paint(overlay=("Please call the experimenter", problem +
                             "\n\nExperimenter: fix it, then press Space.", "alert"))

    def _schedule(self):
        tr = self.play.trial
        # schedule beyond the output's own latency: Bluetooth reports ~180 ms, more than a fixed
        # 120-ms lead, which made sounds start "late" and get repeated
        st = getattr(self.eng, "_stream", None)
        lead = 0.12 + (float(st.latency) if st is not None else 0.0)
        t_first = max(self.t_ready + self.cfg.ready_ms / 1000, time.perf_counter() + lead)
        t_buf = t_first - tr.interval_on_s[0]
        if self.eng is not None:
            self.pb = self.eng.play_at(self.play.buffer, t_buf)
        self.state = "playing"
        self.t_buf = t_buf
        for i, (on, off) in enumerate(zip(tr.interval_on_s, tr.interval_off_s)):
            self._at(t_buf + self.vis + on, lambda i=i: self._box(i, "active"))
            self._at(t_buf + self.vis + off, lambda i=i: self._box(i, "idle"))
        self.t_end = t_buf + self.vis + tr.interval_off_s[-1]
        self._at(self.t_end, self._sound_over)

    def _box(self, i, st):
        if self.state == "playing":
            boxes = list(self.view["boxes"])
            boxes[i] = st
            self._paint(boxes=boxes)

    def _sound_over(self):
        if self.state != "playing":
            return
        m = self.play.mode
        if m == "listen":
            boxes = ["idle"] * self.play.trial.n_intervals
            boxes[self.play.trial.signal_interval - 1] = "mark"
            self._paint(boxes=boxes, prompt=self.play.caption, style="good")
            self.state = "tail"
            self._at(time.perf_counter() + 1.6, lambda: self._finish(None))
        elif m == "rate":
            self.state = "rate"
            self.t_open = time.perf_counter()
            self._paint(boxes=["idle"] * 5, hints=["one sound", "", "", "", "separate"],
                        prompt="How did it sound?  Press 1 to 5", style="text")
        else:
            self.state = "respond"
            self.t_open = time.perf_counter()
            self._paint(prompt="Which sound had the beep?", style="text")

    def _answer(self, resp, t_key):
        self.state = "feedback"
        self._cancel()
        self.res.response = resp
        self.res.rt_ms = (t_key - self.t_end) * 1000.0
        if self.play.mode == "respond" and self.play.feedback:
            ok = resp == self.play.trial.signal_interval
            boxes = ["idle"] * 3
            boxes[resp - 1] = "good" if ok else "bad"
            self._paint(boxes=boxes, prompt=("✓  Correct" if ok else "✗  Wrong"),
                        style="good" if ok else "bad")
            self._at(t_key + self.cfg.feedback_ms / 1000, self._blank)
        else:
            self._blank()

    def _blank(self):
        self.state = "blank"
        self._paint(boxes=["idle"] * len(self.view["boxes"]), prompt="")
        self._at(time.perf_counter() + self.cfg.blank_ms / 1000, lambda: self._finish(None))

    def _finish(self, reason):
        if self.pb is not None:
            if reason and not self.pb.finished:
                self.eng.abort(self.pb)
            rep = self.pb.report()
            rep["onset_error_ms"] = rep.pop("onset_latency_ms", rep.get("onset_error_ms"))
            self.res.audio = rep
            if not reason and self.pb.late:
                reason = "audio_late"
            elif not reason and rep.get("underflows"):
                reason = "audio_underflow"
        self.res.keys = self.keys
        if reason:
            self.res.status, self.res.reason = "aborted", reason
        self._advance(self.res)

    def _interrupt(self, why):
        if self.state in ("prepare", "blocked", "playing", "respond", "rate", "tail"):
            self._cancel()
            if self.pb is not None:
                self.eng.abort(self.pb)
            self.res.status, self.res.reason = "aborted", why
            if self.pb is not None:
                self.res.audio = self.pb.report()
            self.res.keys = self.keys
            self._overlay(why, then=("advance", self.res))
        elif self.state in ("screen", "form"):
            self._overlay(why, then=("restore", self.state))

    def _overlay(self, why, then):
        self.over = then
        self.state = "overlay"
        self.t_over = time.perf_counter()
        self.exp.w.event("overlay", why=why)
        title, body = (("Experimenter menu", "Space = continue\nQ = save and quit")
                       if why == "operator" else ("Paused", "Press Space to continue."))
        self.win.stack.setCurrentIndex(0)
        self._paint(overlay=(title, body, "menu"))

    def on_key(self, e):
        t = time.perf_counter()
        k = key_name(e)
        self.keys.append({"key": k, "t_perf": t, "state": self.state}) if hasattr(self, "keys") \
            else None
        if e.isAutoRepeat():
            return
        st = self.state
        if st == "done":
            if k in ("q", "escape"):
                self.quit("after_error")
            return
        if st == "overlay":
            if k == "q" and self.view.get("overlay", ("",))[0] == "Experimenter menu":
                self.quit("operator_quit")
            elif k == "space" and t - self.t_over > 0.4:
                self._paint(overlay=None)
                kind, val = self.over
                if kind == "advance":
                    self._advance(val)
                else:
                    self.state = val
                    if val == "form":
                        self.win.stack.setCurrentIndex(1)
            return
        if k == "escape":
            if isinstance(self.action, Screen) and self.action.kind == "end":
                self._advance({"key": "escape"})
            else:
                self._interrupt("operator")
            return
        if k == "p" and st != "form":
            self._interrupt("pause")
            return
        if st == "blocked" and k == "space":
            self._paint(overlay=None)
            self._play(self.play)
        elif st == "screen" and k in self.action.keys and t - self.t_shown >= self.action.min_s:
            self._advance({"key": k})
        elif st == "respond" and k in ANSWER:
            self._answer(ANSWER[k], t)
        elif st == "rate" and k in ("1", "2", "3", "4", "5"):
            self._answer(int(k), t)

    def _tick(self):
        if self.state == "prepare":
            self._prepare()
        elif self.state == "playing" and self.pb is not None and (self.pb.late or not self.eng.alive):
            self._cancel()
            self._finish("audio_late" if self.pb.late else "audio_error")

    def _at(self, t, fn):
        tm = QtCore.QTimer(self)
        tm.setSingleShot(True)
        tm.setTimerType(Qt.TimerType.PreciseTimer)
        tm.timeout.connect(fn)
        tm.start(max(0, int(round((t - time.perf_counter()) * 1000))))
        self.timers.append(tm)

    def _cancel(self):
        for tm in self.timers:
            tm.stop()
        self.timers = []

    def quit(self, why):
        if self.closing:
            return
        self.closing = True
        self._cancel()
        if self.eng is not None and getattr(self, "pb", None) is not None and not self.pb.finished:
            self.eng.abort(self.pb)
        if isinstance(self.action, Play) and getattr(self, "res", None) and self.state in (
                "prepare", "playing", "respond", "rate", "tail", "overlay", "blocked"):
            self.res.status, self.res.reason = "aborted", why
            try:
                self.gen.send(self.res)               # the script logs the interrupted trial
            except Exception:
                pass
        try:
            self.gen.close()
        except Exception:
            pass
        self.exp.w.event("quit", why=why)
        if self.exp.w.meta.get("status") == "running":
            self.exp.w.close(why)
        self._exit(why)

    def _fatal(self, e):
        self.exp.w.event("fatal", error=repr(e), traceback=traceback.format_exc())
        if self.exp.w.meta.get("status") == "running":
            self.exp.w.close("error")
        self.state = "done"
        self.win.stack.setCurrentIndex(0)
        self.view = {"kind": "message", "title": "Something went wrong",
                     "body": "Please call the experimenter.\n\n" + repr(e),
                     "footer": "Experimenter: press Esc"}
        self.win.canvas.show(**self.view)

    def _exit(self, why):
        self.state = "done"
        self.closing = True
        self.poll.stop()
        self.pool.shutdown(wait=False)
        self.on_exit(why)


def run_app(exp, engine, locked_volume, fullscreen=True, visual_offset_s: float = 0.0,
            set_volume=None) -> str:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = Window()
    out = {}

    def done(why):
        out["why"] = why
        QtCore.QTimer.singleShot(200, app.quit)

    r = Runner(win, exp, engine, locked_volume, done, visual_offset_s=visual_offset_s,
               set_volume=set_volume)
    if fullscreen:
        win.showFullScreen()
        app.setOverrideCursor(Qt.CursorShape.BlankCursor)
    else:
        win.resize(1280, 800)
        win.show()
    QtCore.QTimer.singleShot(300, r.start)
    app.exec()
    app.restoreOverrideCursor()
    return out.get("why", "")
