"""The listening session: continuous percept report on A(AB)A sequences.

Reuses tsync's audio engine (one stream for the session, sample-scheduled onsets, DAC time stamps)
and its macOS key mapping. During a sequence the screen shows two boxes, TOGETHER (left) and
SEPARATE (right); the box of the last key pressed stays lit, so listeners can see what they are
reporting. Keys: left arrow or 1 = together, right arrow or 2 = separate, P = pause (the sequence
is discarded and run again at the end), Esc then Q = save and quit.
"""
from __future__ import annotations

import json
import time
from typing import Callable, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen

from tsync.app import key_name
from tsync.logbook import now_iso, write_json_atomic
from tsync.system import volume_settings

from .analysis import SEP, TOG, trial_stats

C = {"bg": "#0E1116", "panel": "#171C23", "edge": "#2A313B", "text": "#E9EDF2", "muted": "#8F9AA8",
     "tog": "#33C3B0", "sep": "#5B9BFF", "warn": "#F2B84B"}
KEYS = {"left": TOG, "1": TOG, "right": SEP, "2": SEP}


def _font(px, w=QFont.Weight.Normal):
    f = QFont()
    f.setPixelSize(max(9, int(px)))
    f.setWeight(w)
    return f


class Canvas(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.mode, self.title, self.body, self.footer = "message", "", "", ""
        self.current: Optional[str] = None
        self.progress: Optional[float] = None
        self.label = ""

    def set(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.update()

    def _text(self, p, r, s, px, color, w=QFont.Weight.Normal, align=Qt.AlignmentFlag.AlignLeft):
        p.setFont(_font(px, w))
        p.setPen(QColor(C[color]))
        p.drawText(r, int(align | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap), s)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        u = h / 100
        p.fillRect(self.rect(), QColor(C["bg"]))
        if self.mode == "message":
            cw = min(w * 0.62, 118 * u)
            x = (w - cw) / 2
            self._text(p, QRectF(x, h * 0.16, cw, 12 * u), self.title, 4.4 * u, "text",
                       QFont.Weight.Medium)
            self._text(p, QRectF(x, h * 0.16 + 9 * u, cw, 55 * u), self.body, 2.6 * u, "text")
            self._text(p, QRectF(0, h * 0.86, w, 5 * u), self.footer, 2.1 * u, "muted",
                       align=Qt.AlignmentFlag.AlignHCenter)
        else:
            self._text(p, QRectF(4 * u, 3.5 * u, w / 2, 4 * u), self.label, 1.9 * u, "muted")
            pw, ph, gap = min(w * 0.3, 50 * u), 30 * u, min(w * 0.07, 12 * u)
            y = h * 0.42 - ph / 2
            for i, (name, key, lab) in enumerate(((TOG, "←  or  1", "Together"),
                                                  (SEP, "2  or  →", "Separate"))):
                r = QRectF(w / 2 - gap / 2 - pw if i == 0 else w / 2 + gap / 2, y, pw, ph)
                on = self.current == name
                col = QColor(C["tog" if name == TOG else "sep"])
                fill = QColor(col)
                fill.setAlpha(60 if on else 0)
                p.setPen(QPen(col if on else QColor(C["edge"]), 0.45 * u if on else 0.25 * u))
                p.setBrush(fill if on else QColor(C["panel"]))
                p.drawRoundedRect(r, 2.4 * u, 2.4 * u)
                self._text(p, QRectF(r.x(), r.y() + ph * 0.22, pw, 10 * u), lab, 5.0 * u,
                           "text" if on else "muted", QFont.Weight.Medium, Qt.AlignmentFlag.AlignHCenter)
                self._text(p, QRectF(r.x(), r.y() + ph * 0.68, pw, 5 * u), key, 2.2 * u, "muted",
                           align=Qt.AlignmentFlag.AlignHCenter)
            msg = "Press when what you hear changes" if self.current else \
                "Press a key as soon as you know what you hear"
            self._text(p, QRectF(0, y + ph + 7 * u, w, 5 * u), msg, 2.6 * u,
                       "text" if not self.current else "muted", align=Qt.AlignmentFlag.AlignHCenter)
        if self.progress is not None:
            bw = min(w * 0.36, 60 * u)
            r = QRectF((w - bw) / 2, h - 5.5 * u, bw, 0.55 * u)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(C["edge"]))
            p.drawRoundedRect(r, 0.3 * u, 0.3 * u)
            p.setBrush(QColor(C["sep"]))
            p.drawRoundedRect(QRectF(r.x(), r.y(), bw * self.progress, r.height()), 0.3 * u, 0.3 * u)
        p.end()


INSTR = [
    ("What you will hear",
     "Repeating groups of three sounds: a low beep, a middle sound, a low beep, then a short pause."
     "\n\nThe middle sound has a low and a high beep in it. Sometimes the high beep is heard "
     "as PART of the middle sound, which then sounds like one rich sound. Sometimes the high "
     "beep stands out as a SEPARATE sound of its own, apart from the low beeps."),
    ("Your task",
     "There are no right or wrong answers. Tell us what you hear, as it happens.\n\n"
     "Press ← (or 1) for TOGETHER: the high beep is part of one rich middle sound.\n"
     "Press → (or 2) for SEPARATE: the high beep is a sound on its own.\n\n"
     "Press as soon as you know, then again whenever it changes. It may change several times, "
     "or not at all. Do not try to hear it one way: just report. Press P to pause."),
]


class Session(QtCore.QObject):
    def __init__(self, win, cfg, plan: List[dict], engine, writer, render: Callable,
                 locked_volume: Optional[int], on_exit: Callable[[str], None]):
        super().__init__()
        self.win, self.cfg, self.plan, self.eng = win, cfg, list(plan), engine
        self.w, self.render, self.locked, self.on_exit = writer, render, locked_volume, on_exit
        self.i, self.state, self.page = 0, "intro", 0
        self.timers: List[QtCore.QTimer] = []
        self.n_main = sum(1 for t in plan if t["phase"] == "main")
        self.done_main = 0
        self.last_break = 0
        self.pb = None

    # -- flow ------------------------------------------------------------------------------------
    def start(self):
        self._page()

    def _page(self):
        t, b = INSTR[self.page]
        self.state = "intro"
        self.win.canvas.set(mode="message", title=t, body=b, footer="Press Space to continue",
                            progress=None)

    def _message(self, title, body, footer="Press Space to continue", state="wait"):
        self.state = state
        self.win.canvas.set(mode="message", title=title, body=body, footer=footer)

    def _next(self):
        self._cancel()
        if self.state == "playing":           # a stray timer: never start a second sequence
            return
        if self.i >= len(self.plan):
            self.w["meta"]["status"] = "complete"
            self.w["save"]()
            self._message("Thank you!", "That is the end. Please tell the experimenter.",
                          "Experimenter: press Esc", state="end")
            return
        tr = self.plan[self.i]
        if tr["phase"] == "main" and self.done_main and self.done_main % self.cfg.long_rest_every == 0 \
                and self.last_break != self.done_main:
            self.last_break = self.done_main
            self._message("Short break", f"{self.done_main} of {self.n_main} done.")
            return
        self._play(tr)

    def _play(self, tr):
        vol = volume_settings()
        if self.locked is not None and (vol.get("output_volume") != self.locked or vol.get("muted")):
            self._message("Please call the experimenter",
                          f"System volume is {vol.get('output_volume')} (muted {vol.get('muted')}), "
                          f"it must be {self.locked}. Fix it, then press Space.")
            return
        self.tr, self.vol, self.reports, self.keys = tr, vol, [], []
        try:
            buf = self.render(tr)
            self.t_start = time.perf_counter() + 0.8
            self.pb = self.eng.play_at(buf, self.t_start - 0.05)
        except Exception as e:                # dead stream, unplugged headphones, clipping
            self.pb = None
            self._message("Please call the experimenter",
                          f"The sound could not be played ({e}). Fix it, then press Space.")
            return
        self.state = "playing"
        self.win.canvas.set(mode="trial", current=None,
                            label="Practice" if tr["phase"] == "practice" else
                            f"{self.done_main + 1} of {self.n_main}",
                            progress=self.done_main / max(1, self.n_main))
        self.dur = tr["sound_s"]               # the real span of the sound, not the nominal length
        self._at(self.t_start - 0.05 + buf.shape[0] / self.cfg.sample_rate, self._end)

    def _onset(self) -> float:
        on = self.pb.onset_perf() if self.pb is not None else None
        return (on if on is not None else self.t_start - 0.05) + 0.05

    def _end(self, status="ok", reason=""):
        self._cancel()
        if status == "ok" and self.pb is not None:
            r = self.pb.report()
            if self.pb.late:
                status, reason = "aborted", "audio_late"
            elif r.get("underflows"):
                status, reason = "aborted", "audio_underflow"
        on = self._onset()
        rep = [(t - on, p) for t, p in self.reports]
        rec = {"phase": self.tr["phase"], "condition": self.tr["condition"], "d_ms": self.tr["d_ms"],
               "flank": self.tr["flank"], "seed": self.tr["seed"], "seconds": self.dur,
               "status": status, "reason": reason, "reports": rep, "keys": self.keys,
               "stats": trial_stats(rep, self.dur), "sha256": self.tr.get("sha256"),
               "audio": self.pb.report() if self.pb is not None else {},
               "volume": self.vol, "wall": now_iso(), "index": self.i}
        self.w["trial"](rec)
        if status == "ok":
            if self.tr["phase"] == "main":
                self.done_main += 1
            self.i += 1
        else:                                  # run it again: practice right away, main at the end
            again = dict(self.tr, seed=list(self.tr["seed"]) + [len(self.plan)])
            self.i += 1
            if self.tr["phase"] == "practice":
                self.plan.insert(self.i, again)
            else:
                self.plan.append(again)
        self.state = "rest"
        self.win.canvas.set(mode="trial", current=None, label="")
        self._at(time.perf_counter() + self.cfg.rest_s, self._next)

    # -- keys --------------------------------------------------------------------------------
    def on_key(self, e):
        k = key_name(e)
        t = time.perf_counter()
        if e.isAutoRepeat():
            return
        if self.state == "playing":
            self.keys.append({"key": k, "t_perf": t})
            if k in KEYS:
                self.reports.append((t, KEYS[k]))
                self.win.canvas.set(current=KEYS[k])
            elif k == "p":
                self.eng.abort(self.pb)
                self._end("aborted", "pause")
                self._cancel()
                self._message("Paused", "Press Space to continue. That sequence will be "
                              "played again later.")
            elif k == "escape":
                self.eng.abort(self.pb)
                self._end("aborted", "operator")
                self._cancel()
                self._message("Experimenter menu", "Space = continue, Q = save and quit",
                              state="menu")
            return
        if self.state == "menu" and k == "q":
            self.quit("operator_quit")
        elif self.state == "end" and k == "escape":
            self.quit("complete")
        elif self.state == "intro" and k == "space":
            self.page += 1
            self._page() if self.page < len(INSTR) else self._next()
        elif self.state in ("wait", "menu") and k == "space":
            self._next()
        elif k == "escape" and self.state not in ("end", "menu"):
            self._cancel()
            self._message("Experimenter menu", "Space = continue, Q = save and quit", state="menu")

    def _at(self, t, fn):
        tm = QtCore.QTimer(self)
        tm.setSingleShot(True)
        tm.setTimerType(Qt.TimerType.PreciseTimer)
        tm.timeout.connect(fn)
        tm.start(max(0, int((t - time.perf_counter()) * 1000)))
        self.timers.append(tm)

    def _cancel(self):
        for tm in self.timers:
            tm.stop()
        self.timers = []

    def quit(self, why):
        if self.state == "playing":           # log the sequence in flight before leaving
            self.eng.abort(self.pb)
            self._end("aborted", why)
        self._cancel()
        if self.w["meta"].get("status") == "running":
            self.w["meta"]["status"] = why
        self.w["save"]()
        self.on_exit(why)


class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Listening experiment")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.canvas = Canvas()
        lay.addWidget(self.canvas)
        self.session: Optional[Session] = None

    def keyPressEvent(self, e):
        if self.session:
            self.session.on_key(e)

    def closeEvent(self, e):
        if self.session and self.session.state != "end":
            self.session.quit("window_closed")
        e.accept()


def run(cfg, plan, engine, writer, render, locked_volume, fullscreen=True) -> str:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = Window()
    out = {"why": ""}

    def done(why):
        out["why"] = why
        QtCore.QTimer.singleShot(200, app.quit)

    win.session = Session(win, cfg, plan, engine, writer, render, locked_volume, done)
    if fullscreen:
        win.showFullScreen()
        app.setOverrideCursor(Qt.CursorShape.BlankCursor)
    else:
        win.resize(1280, 800)
        win.show()
    QtCore.QTimer.singleShot(300, win.session.start)
    app.exec()
    app.restoreOverrideCursor()
    return out["why"]
