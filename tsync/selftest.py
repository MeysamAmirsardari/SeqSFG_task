"""Drive the real app end to end with a robot participant: no screen, no sound, real Qt timers.

The robot answers every screen, fills both forms, answers trials (mostly correctly so the tracks
converge), pauses once in the middle of a trial and once on a screen, and opens and closes the
experimenter menu. The audio engine is replaced by one that keeps time but makes no sound. A short
configuration keeps the whole run to a few seconds.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import Playback
from .config import Config, validate

FAST = dict(name="tsync_selftest", tone_ms=40.0, ramp_ms=10.0, soa_ms=80.0, n_tones=3,
            context_ms=(0.0, 20.0), include_late_control=True, late_control_shift_ms=10.0,
            exposure_conditions=("D20",), exposure_per_condition=1, delta_start_ms=8.0,
            delta_max_ms=20.0, catch_delta_ms=20.0, catch_per_track=1, catch_positions=(1, 2),
            max_trials_per_track=6, practice_n=2, practice_criterion=1, isi_ms=(100.0, 110.0),
            ready_ms=40, feedback_ms=40, blank_ms=40, min_rest_s=0.0, long_rest_s=0.0,
            demo_delta_ms=20.0, practice_delta_ms=15.0, exposure_delta_ms=15.0, n_final_reversals=2,
            reversals_per_step=(1, 1), reminder_after_s=5.0, autopause_after_s=20.0)


def fast_config() -> Config:
    return validate(Config(**FAST))


class SilentEngine:
    """Stands in for AudioEngine: schedules nothing, reports a perfect playback."""

    def __init__(self, sr: int):
        self.sr, self.error, self.played = sr, None, []

    @property
    def alive(self):
        return True

    @property
    def _last(self):
        return (0, 0.0, time.perf_counter())

    def play_at(self, buf, t_perf):
        pb = Playback(np.asarray(buf), 0, self.sr, t_scheduled_perf=t_perf, clock_offset=0.0)
        pb.started, pb.onset_dac = True, t_perf
        pb.end_dac = t_perf + pb.n / self.sr
        self.played.append(pb)
        return pb

    def abort(self, pb):
        pb.aborted = pb.finished = True

    def silence(self):
        pass


@dataclass
class Report:
    status: str
    why: str
    trials_played: int
    folder: Path
    actions: dict


def run_selftest(data_dir: Path, p_correct: float = 0.85, seed: int = 1,
                 timeout_s: float = 120.0, audio_device: int = None) -> Report:
    """`audio_device`: run the real audio engine on that output, playing silence."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Qt

    from .app import Guard, Runner, Window
    from .experiment import Experiment, Play, Screen
    from .levels import uncalibrated
    from .session import open_session

    cfg = fast_config()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    cal = uncalibrated(cfg, 0.0 if audio_device is not None else 0.05)
    w, plan = open_session(cfg, Path(data_dir), "BOT01", cal, simulated=True, session_seed=seed)
    if audio_device is not None:
        from .audio import AudioEngine
        eng = AudioEngine(audio_device, cfg.sample_rate, cfg.channels, cfg.blocksize, cfg.latency)
        eng.start()
        eng.played = []
        _orig = eng.play_at

        def _track(buf, t, _orig=_orig, eng=eng):
            pb = _orig(buf, t)
            eng.played.append(pb)
            return pb
        eng.play_at = _track
    else:
        eng = SilentEngine(cfg.sample_rate)
    win = Window()
    win.resize(1280, 800)
    win.show()
    done = {"why": None}
    runner = Runner(win, Experiment(cfg, cal, plan, w), eng, w,
                    Guard(None, "silent"), lambda why: done.update(why=why))
    rng = np.random.default_rng(seed)
    counts = {"screen_keys": 0, "answers": 0, "pause_trial": 0, "pause_screen": 0, "menu": 0,
              "forms": 0}
    keymap = {"space": Qt.Key.Key_Space, "return": Qt.Key.Key_Return, "escape": Qt.Key.Key_Escape,
              "left": Qt.Key.Key_Left, "right": Qt.Key.Key_Right, "p": Qt.Key.Key_P,
              "y": Qt.Key.Key_Y, "c": Qt.Key.Key_C, "r": Qt.Key.Key_R, "q": Qt.Key.Key_Q}

    def press(k):
        txt = k if len(k) == 1 else ""
        ev = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, keymap[k], Qt.KeyboardModifier.NoModifier,
                             txt)
        QtWidgets.QApplication.sendEvent(win, ev)

    state = {"last": None}

    def bot():
        st, act = runner.state, runner.action
        if st == "form":
            for f in win.form.fields:
                wdg = win.form.widgets[f["name"]]
                if f["type"] == "int":
                    wdg.setValue(int(f.get("min", 0)) + 3)
                elif f["type"] == "choice":
                    wdg.setCurrentIndex(1)
                elif f["type"] == "longtext":
                    wdg.setPlainText("robot")
                else:
                    wdg.setText("robot")
            counts["forms"] += 1
            win.form.try_submit()
        elif st == "overlay":
            press("space")
        elif st == "screen" and isinstance(act, Screen):
            if act.name == "main_intro" and not counts["pause_screen"]:
                counts["pause_screen"] += 1
                press("p")
                return
            if act.name == "main_intro" and not counts["menu"]:
                counts["menu"] += 1
                press("escape")
                return
            pref = {"check_q": "y", "practice_fail": "c"}.get(act.name)
            key = pref or ("space" if "space" in act.keys else act.keys[0])
            if key == "r":
                key = "space"
            counts["screen_keys"] += 1
            press(key)
        elif st == "playing" and isinstance(act, Play) and act.trial.phase == "main" \
                and not counts["pause_trial"]:
            counts["pause_trial"] += 1
            press("p")
        elif st == "respond" and isinstance(act, Play) and state["last"] is not act:
            state["last"] = act
            ok = rng.random() < p_correct
            resp = act.trial.target_interval if ok else 3 - act.trial.target_interval
            counts["answers"] += 1
            press("left" if resp == 1 else "right")

    t = QtCore.QTimer()
    t.timeout.connect(bot)
    t.start(15)
    QtCore.QTimer.singleShot(50, runner.start)
    t_end = time.time() + timeout_s
    while done["why"] is None and time.time() < t_end:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
        time.sleep(0.002)
    t.stop()
    win.close()
    if audio_device is not None:
        eng.stop()
    return Report(w.meta.get("status"), done["why"] or "timeout", len(eng.played), w.folder, counts)
