"""The booth app for tinc: tcmr's app with a yes/no answer.

One box lights while the sequence plays. Then two answer boxes: 1 = No (← or 1), 2 = Yes
(→ or 2). Feedback marks the box pressed and says Correct or Wrong. Pause (P), the experimenter
menu (Esc), the volume guard and the audio checks are tcmr's, unchanged.
"""
from __future__ import annotations

import time

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

from tcmr.app import Runner, Window, key_name

YESNO = {"1": 1, "left": 1, "2": 2, "right": 2}
HINTS = ["No   (←  or  1)", "Yes   (→  or  2)"]


class YesNoRunner(Runner):
    def _play(self, play):
        super()._play(play)
        if play.mode in ("yesno", "listen"):
            self._paint(hints=[])

    def _sound_over(self):
        if self.state != "playing":
            return
        m = self.play.mode
        if m == "listen":
            self._paint(boxes=["mark"], prompt=self.play.caption, style="good")
            self.state = "tail"
            self._at(time.perf_counter() + 1.6, lambda: self._finish(None))
        elif m == "yesno":
            self.state = "respond"
            self.t_open = time.perf_counter()
            self._paint(boxes=["idle", "idle"], hints=HINTS,
                        prompt="Did one high tone get louder on its own?", style="text")
        else:
            super()._sound_over()

    def _answer(self, resp, t_key):
        if self.play.mode != "yesno":
            return super()._answer(resp, t_key)
        self.state = "feedback"
        self._cancel()
        self.res.response = resp
        self.res.rt_ms = (t_key - self.t_end) * 1000.0
        if self.play.feedback:
            ok = (resp == 2) == self.play.trial.target_present
            boxes = ["idle", "idle"]
            boxes[resp - 1] = "good" if ok else "bad"
            self._paint(boxes=boxes, prompt=("✓  Correct" if ok else "✗  Wrong"),
                        style="good" if ok else "bad")
            self._at(t_key + self.cfg.feedback_ms / 1000, self._blank)
        else:
            self._blank()

    def on_key(self, e):
        if self.state == "respond" and getattr(self, "play", None) is not None \
                and self.play.mode == "yesno" and not e.isAutoRepeat():
            k = key_name(e)
            if k in YESNO:
                self.keys.append({"key": k, "t_perf": time.perf_counter(), "state": self.state})
                self._answer(YESNO[k], time.perf_counter())
                return
            if k in ("3", "down"):                 # not an answer here
                return
        super().on_key(e)


def run_app(exp, engine, locked_volume, fullscreen=True, visual_offset_s: float = 0.0,
            set_volume=None) -> str:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = Window()
    out = {}

    def done(why):
        out["why"] = why
        QtCore.QTimer.singleShot(200, app.quit)

    r = YesNoRunner(win, exp, engine, locked_volume, done, visual_offset_s=visual_offset_s,
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
