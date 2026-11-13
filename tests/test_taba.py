"""taba: A(AB)A geometry, percept statistics, plan, and a robot run of the app."""
import json
import os
import time

import numpy as np
import pytest

from taba.analysis import SEP, TOG, clean_reports, trial_stats
from taba.cli import PRESET, make_plan
from taba.config import Config, ConfigError, validate
from taba.stimulus import build, render, sha256

CFG = Config()


def test_preset_equals_defaults():
    assert Config.load(PRESET) == CFG
    validate(CFG)
    with pytest.raises(ConfigError):
        validate(CFG.replace(d_ms=(0.0, 250.0)))


def test_triplets_have_the_right_tones_and_lag():
    for c in CFG.conditions():
        tones = build(CFG, c, 8.0, [3])
        per = {}
        for t in tones:
            per.setdefault(t.triplet, {})[t.role] = t
        for k, d in per.items():
            assert d["b"].onset_s - d["mid_a"].onset_s == pytest.approx(c.d_ms / 1000)
            assert d["mid_a"].freq_hz == pytest.approx(CFG.f_a_hz)
            if c.flank == "none":
                assert "flank1" not in d
            else:
                f = CFG.f_a_hz if c.flank == "same" else CFG.f_off_hz
                assert d["flank1"].freq_hz == pytest.approx(f)
                assert d["flank2"].onset_s - d["flank1"].onset_s == pytest.approx(2 * CFG.soa_ms / 1000)


def test_rendering_is_reproducible():
    c = CFG.condition("D35")
    amp = lambda f, db: 0.08
    assert sha256(render(CFG, build(CFG, c, 5, [1]), amp)) == \
        sha256(render(CFG, build(CFG, c, 5, [1]), amp))


def test_percept_statistics():
    ev = [(2.0, TOG), (2.5, TOG), (10.0, SEP), (20.0, TOG), (40.0, SEP)]
    assert clean_reports(ev, 30.0) == [(2.0, TOG), (10.0, SEP), (20.0, TOG)]
    s = trial_stats(ev, 30.0)
    assert s["first_percept"] == TOG and s["first_latency_s"] == 2.0
    assert s["prop_separate"] == pytest.approx(10.0 / 28.0)
    assert s["switches"] == 2
    assert s["mean_phase_s"][SEP] == pytest.approx(10.0)
    assert trial_stats([], 30.0)["prop_separate"] is None


def test_plan_has_every_condition_and_no_back_to_back_repeats():
    plan = make_plan(CFG, 5)
    main = [t["condition"] for t in plan if t["phase"] == "main"]
    assert len(main) == 6 * CFG.reps + 6 * CFG.control_reps
    assert all(t["sound_s"] <= t["seconds"] + 0.2 for t in plan)
    assert all(a != b for a, b in zip(main, main[1:]))


def test_app_runs_with_a_robot(tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Qt
    from taba.app import Session, Window
    from tsync.selftest import SilentEngine

    cfg = CFG.replace(trial_s=3.2, practice_s=3.2, rest_s=0.05, reps=1, controls=("offfreq_0",))
    plan = make_plan(cfg, 9)
    recs, meta = [], {"status": "running"}
    writer = {"meta": meta, "save": lambda: None, "trial": recs.append}
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = Window()
    win.resize(1000, 700)
    win.show()
    done = {}
    rend = lambda tr: render(cfg, build(cfg, cfg.condition(tr["condition"]), tr["seconds"],
                                        tr["seed"]), lambda f, db: 0.0)
    win.session = s = Session(win, cfg, plan, SilentEngine(cfg.sample_rate), writer, rend, None,
                              lambda why: done.update(why=why))

    def press(k):
        QtWidgets.QApplication.sendEvent(win, QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress, k, Qt.KeyboardModifier.NoModifier, ""))

    state = {"n": 0, "last": None, "menu": 0}

    def bot():
        if s.state == "rest" and s.done_main == 2 and not state["menu"]:
            state["menu"] = 1                    # Esc in the gap between sequences
            press(Qt.Key.Key_Escape)
        elif s.state == "menu":
            press(Qt.Key.Key_Space)
        elif s.state in ("intro", "wait"):
            press(Qt.Key.Key_Space)
        elif s.state == "end":
            press(Qt.Key.Key_Escape)
        elif s.state == "playing":
            el = time.perf_counter() - s.t_start
            k = int(el // 1.0)
            if el > 0.3 and k != state["last"]:
                state["last"] = k
                press(Qt.Key.Key_Left if k % 2 == 0 else Qt.Key.Key_Right)

    t = QtCore.QTimer()
    t.timeout.connect(bot)
    t.start(20)
    QtCore.QTimer.singleShot(10, s.start)
    t_end = time.time() + 120
    while "why" not in done and time.time() < t_end:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
        time.sleep(0.002)
    t.stop()
    assert done.get("why") == "complete" and meta["status"] == "complete"
    main = [r for r in recs if r["phase"] == "main" and r["status"] == "ok"]
    assert len(main) == 7 and state["menu"] == 1
    assert len({r["index"] for r in recs}) == len(recs)        # no sequence played twice
    assert all(r["stats"]["n_reports"] >= 2 and r["stats"]["first_percept"] == TOG for r in main)
    assert all(0 <= r["reports"][0][0] < 1.5 for r in main)
