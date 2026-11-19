"""tinc: Micheyl et al. (2013)'s level-increment task with A and B roved together."""
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from tinc import audit
from tinc.cli import drive
from tinc.config import Config, ConfigError, validate
from tinc.observer import SimListener, d_prime
from tinc.session import FIELDS, Experiment, Writer, block_plan
from tinc.stimulus import build, sha256

CONFIGS = Path(__file__).resolve().parents[1] / "tinc" / "configs"
OURS = validate(Config.load(CONFIGS / "tinc_main.json"))
THEIRS = validate(Config.load(CONFIGS / "tinc_micheyl.json"))


def test_presets_are_micheyl_2013_experiment_3():
    for c in (OURS, THEIRS):
        assert (c.f_b_hz, round(c.f_a_hz, 2), c.tone_ms, c.ramp_ms, c.soa_ms, c.n_tones) == \
            (1000.0, 420.45, 100.0, 20.0, 200.0, 12)
        assert c.delays_ms == (0.0, 35.0, 50.0, 71.0, 100.0)
        assert (c.level_db, c.rove_db, c.decoy_db, c.decoy_counts, c.target_db, c.target_from) == \
            (60.0, 4.0, 4.0, (1, 2), 4.0, 4)
        assert c.trials_per_block == 52 and c.sequence_ms(0) == 2300 and c.sequence_ms(100) == 2400
    assert OURS.levels == "common" and THEIRS.levels == "independent"
    assert {k: v for k, v in OURS.to_dict().items() if k not in ("name", "levels")} == \
        {k: v for k, v in THEIRS.to_dict().items() if k not in ("name", "levels")}


def test_validation():
    with pytest.raises(ConfigError):
        validate(OURS.replace(delays_ms=(0.0, 120.0)))        # A would run into the next B
    with pytest.raises(ConfigError):
        validate(OURS.replace(levels="other"))
    with pytest.raises(ConfigError):
        validate(OURS.replace(trials_per_block=51))


def test_levels_follow_each_rule():
    for k in range(50):
        for present in (True, False):
            tr, _ = build(OURS, 35.0, present, (3, k))
            a = np.array([t.level_db for t in tr.tones if t.stream == "A"])
            b = np.array([t.level_db for t in tr.tones if t.stream == "B"])
            d = b - a                                          # common: 0 but the change
            assert (np.abs(d) > 1e-9).sum() == int(present)
            if present:
                assert d[tr.target_index] == pytest.approx(4.0) and tr.target_index >= 4
            assert len(tr.decoys) in (1, 2)
            tr2, _ = build(THEIRS, 35.0, present, (3, k))
            b2 = np.array([t.level_db for t in tr2.tones if t.stream == "B"])
            assert (np.abs(b2 - 60.0) > 1e-9).sum() == int(present)     # B fixed but the change


def test_timing_and_one_buffer_length():
    for d in OURS.delays_ms:
        tr, buf, stem = build(OURS, d, True, (1, int(d)), stems=True)
        assert buf.shape[0] == 120960
        sr, m = OURS.sample_rate, int(0.1 * OURS.sample_rate)
        for row, off in ((0, 0.0), (1, d)):
            starts = [int(round((40 + k * 200 + off) * sr / 1000)) for k in range(12)]
            mask = np.zeros(buf.shape[0], bool)
            for i0 in starts:
                mask[i0:i0 + m] = True
            assert not np.any(stem[row, ~mask])
        assert tr.interval_off_s[0] - tr.interval_on_s[0] == pytest.approx((2300 + d) / 1000)


def test_the_two_rules_reward_opposite_strategies():
    assert d_prime(OURS, 0.0, "pair", n=800) > 1.2 > 0.7 > d_prime(OURS, 0.0, "b_alone", n=800)
    assert d_prime(THEIRS, 0.0, "b_alone", n=800) > 2.0 > 0.8 > d_prime(THEIRS, 0.0, "pair", n=800)
    # alternation leaves a timing-only pairing rule guessing which A belongs to which B
    assert d_prime(OURS, 100.0, "nearest", n=800) < 0.8 < d_prime(OURS, 71.0, "nearest", n=800)


def test_blocks_balanced_and_fixed_by_the_seed():
    plan = block_plan(OURS, 20261115)
    order = [b["delay_ms"] for b in plan]
    assert order[:5] == order[5:][::-1] and sorted(set(order)) == sorted(OURS.delays_ms)
    assert all(sum(b["present"]) == 26 for b in plan)
    assert block_plan(OURS, 20261115) == plan and block_plan(OURS, 7) != plan


def test_seed_audit_passes_and_catches_a_fault(monkeypatch):
    small = OURS.replace(trials_per_block=8, practice=((0.0, 2),))
    rep = audit.audit(small, 5, attempts=(0,))
    assert rep["ok"], rep["problems"]
    import tinc.audit as A
    real = A.build
    def shifted(cfg, delay, present, seed, **kw):
        out = real(cfg, delay, present, seed, **kw)
        if kw.get("stems"):
            tr, buf, stem = out
            stem[1] = np.roll(stem[1], 48)                     # A tones 1 ms late
            return tr, stem.sum(0)[:, None].repeat(2, 1).astype(np.float32), stem
        return out
    monkeypatch.setattr(A, "build", shifted)
    assert not audit.audit(small, 5, attempts=(0,))["ok"]


def test_simulated_session_and_analysis(tmp_path):
    from tinc.analysis import figure, summarize
    w = Writer(tmp_path / "s", {"config": OURS.to_dict(), "code": "T1", "session_seed": 3}, fields=FIELDS)
    drive(Experiment(OURS, w, 3), SimListener(OURS, "pair", seed=1))
    rows = list(csv.DictReader(open(tmp_path / "s" / "trials.csv")))
    main = [r for r in rows if r["phase"] == "main"]
    assert len(main) == 520
    s = summarize(tmp_path / "s")
    assert set(s["by_delay"]) == {"0", "35", "50", "71", "100"}
    assert all(v["d_prime"] > 0.8 for v in s["by_delay"].values())   # the pair listener does well
    assert figure(tmp_path / "s", s).exists()


def test_app_runs_with_a_robot(tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Qt
    from tcmr.app import Window
    from tinc.app import YesNoRunner
    from tsync.selftest import SilentEngine

    cfg = OURS.replace(blocks_per_delay=1, trials_per_block=2, delays_ms=(0.0, 100.0),
                       examples=((0.0, 1),), practice=((0.0, 2),), ready_ms=30, feedback_ms=30,
                       blank_ms=30)
    w = Writer(tmp_path / "s", {"config": cfg.to_dict(), "code": "BOT"}, fields=FIELDS)
    exp = Experiment(cfg, w, 11)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = Window()
    win.resize(1000, 700)
    win.show()
    done = {}
    r = YesNoRunner(win, exp, SilentEngine(cfg.sample_rate), None, lambda why: done.update(why=why),
                    read_volume=lambda: {"output_volume": 50, "muted": False, "readable": True})
    keys = {"space": Qt.Key.Key_Space, "p": Qt.Key.Key_P, "1": Qt.Key.Key_1, "2": Qt.Key.Key_2,
            "3": Qt.Key.Key_3, "escape": Qt.Key.Key_Escape}

    def press(k):
        QtWidgets.QApplication.sendEvent(win, QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress, keys[k], Qt.KeyboardModifier.NoModifier,
            k if len(k) == 1 else ""))

    st = {"paused": 0, "last": None, "three": 0}

    def bot():
        s = r.state
        if s == "form":
            for f in win.form.fields:
                wd = win.form.widgets[f["name"]]
                if f["type"] == "int":
                    wd.setValue(int(f.get("min", 0)) + 4)
                elif f["type"] == "choice":
                    wd.setCurrentIndex(1)
            win.form.try_submit()
        elif s == "overlay":
            press("space")
        elif s == "screen":
            press(r.action.keys[0] if r.action.keys[0] in keys else "space")
        elif s == "playing" and r.play.trial.phase == "main" and not st["paused"]:
            st["paused"] = 1
            press("p")
        elif s == "respond" and st["last"] is not r.play:
            if not st["three"]:
                st["three"] = 1
                press("3")                                   # not an answer: ignored
            st["last"] = r.play
            press("2" if r.play.trial.target_present else "1")

    t = QtCore.QTimer()
    t.timeout.connect(bot)
    t.start(15)
    QtCore.QTimer.singleShot(10, r.start)
    t_end = time.time() + 150
    while "why" not in done and time.time() < t_end:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
        time.sleep(0.002)
    t.stop()
    assert done.get("why") == "complete"
    from tsync.logbook import read_jsonl
    rows = [x["row"] for x in read_jsonl(tmp_path / "s" / "trials.jsonl")]
    main = [x for x in rows if x["phase"] == "main" and x["status"] == "ok"]
    assert len(main) == 4 and all(x["correct"] for x in main)
    assert any(x["reason"] == "pause" for x in rows)
    assert {x["said_yes"] for x in main} == {True, False}


def test_pointer_shows_on_the_questionnaire_only(tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from PySide6.QtCore import Qt
    from tcmr.app import Window
    from tinc.app import YesNoRunner
    from tsync.selftest import SilentEngine
    w = Writer(tmp_path / "s", {"config": OURS.to_dict(), "code": "BOT"}, fields=FIELDS)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setOverrideCursor(Qt.CursorShape.BlankCursor)          # as run_app does full screen
    try:
        win = Window()
        r = YesNoRunner(win, Experiment(OURS, w, 1), SilentEngine(OURS.sample_rate), None,
                        lambda why: None)
        r.start()                                               # first screen: the questionnaire
        assert r.state == "form"
        assert QtWidgets.QApplication.overrideCursor().shape() == Qt.CursorShape.ArrowCursor
        r._form_done({})
        assert QtWidgets.QApplication.overrideCursor().shape() == Qt.CursorShape.BlankCursor
    finally:
        app.restoreOverrideCursor()
