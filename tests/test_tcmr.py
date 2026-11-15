"""tcmr: stimulus, tracks, order, fit, the session script, and a robot run of the app."""
import json
import os
import time

import numpy as np
import pytest

from tcmr.cli import DEFAULT_SIM, PRESET, drive
from tcmr.config import ARRANGEMENTS, Condition, Config, ConfigError, validate
from tcmr.procedure import SimListener, Track, fit, make_order, make_tracks
from tcmr.session import Experiment, Writer
from tcmr.stimulus import band_envelopes, build

CFG = Config()


def test_preset_and_validation():
    assert Config.load(PRESET) == CFG
    validate(CFG)
    with pytest.raises(ConfigError):     # 2 x 30 ms would run a band into the final burst
        validate(CFG.replace(conditions=(("co", "ascending", 30.0, 1),)))


def test_onsets_follow_the_arrangement_and_the_final_burst_is_synchronous():
    for arr in ("ascending", "zigzag"):
        c = Condition("co", arr, 20.0)
        tr, buf = build(CFG, c, None, (1,), n_intervals=1)
        assert tr.offsets_ms == [s * 20.0 for s in ARRANGEMENTS[arr]]
        assert tr.offsets_ms[CFG.target_band] == 0.0
        e = band_envelopes(CFG, buf, tr.target_on_s[0] + 0.03, 0.12)
        assert np.min(np.corrcoef(e)) > 0.9          # comodulated final burst


def test_tone_only_in_the_signal_interval():
    c = Condition("co", "sync", 0.0)
    tr, with_tone = build(CFG, c, 80.0, (3,), signal_interval=2)
    _, without = build(CFG.replace(), c, 80.0, (3,), signal_interval=2,
                       amplitude=lambda f, db: 0.0 if db > 70 else 0.02 * 10 ** ((db - 60) / 20))
    d = (with_tone - without)[:, 0]
    sr = CFG.sample_rate
    on = lambda i: slice(int(tr.target_on_s[i] * sr), int((tr.target_on_s[i] + 0.18) * sr))
    assert np.abs(d[on(1)]).max() > 0.1 and np.abs(d[on(0)]).max() == 0 and np.abs(d[on(2)]).max() == 0


def test_track_rule_and_order():
    t = Track(CFG, CFG.condition_list()[0], 0)
    for c in (True, True):
        t.update(c)
    assert t.level == 67.0                           # 75 - 8 after two correct
    t.update(False)
    assert t.level == 75.0 and t.n_rev == 1
    order = make_order(CFG, 3)
    assert len(order) == CFG.n_tracks * CFG.trials_per_track
    assert all(a != b for a, b in zip(order, order[1:]))
    assert all(order.count(i) == CFG.trials_per_track for i in range(CFG.n_tracks))


def test_fit_recovers_thresholds():
    rng = np.random.default_rng(0)
    lst = SimListener({"a": 50.0, "b": 58.0})
    data = {}
    for k in ("a", "b"):
        L = rng.uniform(40, 70, 400)
        data[k] = (L, [rng.random() < lst.p(k, l) for l in L])
    f = fit(data)
    assert f["thresholds_db"]["a"] == pytest.approx(50.0, abs=1.5)
    assert f["thresholds_db"]["b"] == pytest.approx(58.0, abs=1.5)


def test_simulated_session_and_resume(tmp_path):
    from tsync.logbook import read_jsonl
    from tcmr.analysis import summarize
    w = Writer(tmp_path / "s", {"config": CFG.to_dict(), "code": "T1", "session_seed": 5})
    drive(Experiment(CFG, w, 5), SimListener(DEFAULT_SIM), np.random.default_rng(1))
    rows = [r["row"] for r in read_jsonl(tmp_path / "s" / "trials.jsonl")]
    main = [r for r in rows if r["phase"] == "main"]
    assert len(main) == CFG.n_tracks * CFG.trials_per_track
    assert json.loads((tmp_path / "s" / "session.json").read_text())["status"] == "complete"
    s = summarize(tmp_path / "s")
    assert s["cmr_db"]["sync_0"] > 3 and s["contrasts_db"]["shear_cost_ascending_20_db"] > 0
    # resume: replay half the main trials into a fresh experiment, it continues from there
    half = [r for r in rows if r["phase"] != "main"] + main[:len(main) // 2]
    exp = Experiment(CFG, Writer(tmp_path / "r", {"config": CFG.to_dict()}), 5, resume_rows=half)
    assert exp.done == len(main) // 2
    assert sum(len(t.steps) for t in exp.tracks) == len(main) // 2


def test_app_runs_with_a_robot(tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Qt
    from tcmr.app import Runner, Window
    from tsync.selftest import SilentEngine

    cfg = CFG.replace(trials_per_track=2, n_precursors=1, practice_levels_db=(80.0,),
                      ratings=(("sync", 0.0),), rating_reps=1, ready_ms=30, feedback_ms=30,
                      blank_ms=30, isi_ms=100.0, break_every_rounds=1)
    w = Writer(tmp_path / "s", {"config": cfg.to_dict(), "code": "BOT"})
    exp = Experiment(cfg, w, 11)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = Window()
    win.resize(1000, 700)
    win.show()
    done = {}
    r = Runner(win, exp, SilentEngine(cfg.sample_rate), None, lambda why: done.update(why=why),
               read_volume=lambda: {"output_volume": 50, "muted": False, "readable": True})
    keys = {"space": Qt.Key.Key_Space, "escape": Qt.Key.Key_Escape, "p": Qt.Key.Key_P,
            "1": Qt.Key.Key_1, "2": Qt.Key.Key_2, "3": Qt.Key.Key_3}

    def press(k):
        QtWidgets.QApplication.sendEvent(win, QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress, keys[k], Qt.KeyboardModifier.NoModifier,
            k if len(k) == 1 else ""))

    st = {"paused": 0, "last": None}

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
            st["last"] = r.play
            press(str(r.play.trial.signal_interval))
        elif s == "rate" and st["last"] is not r.play:
            st["last"] = r.play
            press("3")

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
    assert len(main) == cfg.n_tracks * 2 and all(x["correct"] for x in main)
    assert any(x["reason"] == "pause" for x in rows)
    assert [x["rating"] for x in rows if x["phase"] == "rating"] == [3]


def test_ascending_preset_never_overlaps():
    from pathlib import Path
    cfg = validate(Config.load(Path(PRESET).parent / "tcmr_ascending.json"))
    assert {c.arrangement for c in cfg.condition_list()} == {"sync", "ascending"}
    assert max(c.dt_ms for c in cfg.condition_list()) == 25.0          # largest shift 2 x 25 = 50 ms
    for c in cfg.condition_list():
        tr, buf = build(cfg, c, 70.0, (4, int(c.dt_ms)), n_intervals=1, signal_interval=1)
        for b, o in enumerate(tr.offsets_ms):
            gap = cfg.noi_ms - o - cfg.burst_ms      # silence before the final burst in band b
            assert gap >= 10.0


def test_tone_band_is_randomised_and_band_alone_has_one_band():
    from pathlib import Path
    cfg = validate(Config.load(Path(PRESET).parent / "tcmr_ascending.json"))
    assert cfg.tone_bands == (2, 3)
    hz = {build(cfg, Condition("co", "sync", 0.0), 70.0, (k,))[0].tone_hz for k in range(40)}
    assert hz == {1000.0, 2000.0}
    tr, buf = build(cfg, Condition("ref", "sync", 0.0), None, (3,), n_intervals=1)
    e = band_envelopes(cfg, buf, tr.target_on_s[0] + 0.03, 0.12).mean(axis=1)
    others = [e[b] for b in range(5) if b != tr.tone_band]
    assert max(others) < 0.05 * e[tr.tone_band]          # only the tone's band is present
    # the default preset still regenerates exactly as before (no extra random draw)
    t1, b1 = build(CFG, Condition("co", "sync", 0.0), 66.0, (9,))
    assert t1.tone_hz == 1000.0


def test_bursts_keep_the_natural_level_of_the_noise():
    from tcmr.stimulus import band_bursts
    rng = np.random.default_rng(3)
    lev = lambda xs: 10 * np.log10([np.mean(x ** 2) for x in xs])
    co = np.array([lev(band_bursts(CFG, rng, True)) for _ in range(400)])
    rd = np.array([lev(band_bursts(CFG, rng, False)) for _ in range(400)])
    assert 1.5 < co[:, 2].std() < 3.2                       # about 2 dB for 20 Hz x 187.5 ms
    assert np.abs(co - co[:, [2]]).max() < 0.3              # comodulated bands share the level
    assert abs(np.corrcoef(rd[:, 1], rd[:, 2])[0, 1]) < 0.2  # random bands do not
    assert np.mean(10 ** (co[:, 2] / 10)) == pytest.approx(1.0, abs=0.1)   # long-term level kept
    short = CFG.replace(burst_ms=40.0, ramp_ms=10.0, noi_ms=200.0)
    s = np.array([lev(band_bursts(short, rng, True)) for _ in range(400)])
    assert s[:, 2].std() > co[:, 2].std() + 1.0             # shorter bursts vary more
    eq = np.array([lev(band_bursts(CFG.replace(burst_level="equalised"), rng, False))
                   for _ in range(20)])
    assert np.abs(eq).max() < 1e-9                          # the old behaviour is still available
    with pytest.raises(ConfigError):
        validate(CFG.replace(burst_level="loud"))


def test_a_tall_noise_peak_draws_the_trial_again():
    c = Condition("co", "sync", 0.0)
    tight = lambda f, db: 0.09                              # little headroom: some draws exceed 0.95
    out = [build(CFG, c, None, (7, k), n_intervals=1, amplitude=tight) for k in range(30)]
    again = [(t, b) for t, b in out if len(t.seed) > 2]
    assert again and all(np.abs(b).max() <= 0.95 for _, b in out)
    t, b = again[0]                                         # the logged seed regenerates the trial
    assert np.array_equal(build(CFG, c, None, t.seed, n_intervals=1, amplitude=tight)[1], b)
    with pytest.raises(RuntimeError):                       # a level that never fits still fails
        build(CFG, c, None, (7, 0), n_intervals=1, amplitude=lambda f, db: 1.0)
