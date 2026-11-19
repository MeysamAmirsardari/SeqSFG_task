"""tcmr, 2026-11-15: the practice gate, the top of the level range, the calibration routine and the
rules for continuing a session."""
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from tcmr.analysis import describe, summarize
from tcmr.cli import (_check_seed, cal_id, calibration_notes, drive, main, next_seed, parse_level,
                      sim_thresholds)
from tcmr.config import Config, ConfigError, validate
from tcmr.procedure import CEILING_CALLS, SimListener, Track
from tcmr.session import Experiment, Result, Writer, practice_state, resume_blocked
from tsync.levels import Calibration
from tsync.logbook import read_jsonl

CONFIGS = Path(__file__).resolve().parents[1] / "tcmr" / "configs"
EXP1 = validate(Config.load(CONFIGS / "tcmr_exp1.json"))
SMALL = EXP1.replace(name="small", trials_per_track=2)


class Scripted:
    """Right on the first `k` practice trials, wrong on the other practice trials; in the main
    part a simulated listener."""

    def __init__(self, k, cfg=EXP1):
        self.k, self.n, self.seen = k, len(cfg.practice_levels_db), 0
        self.sim = SimListener(sim_thresholds(cfg))

    def p(self, cond, level):
        self.seen += 1
        if self.seen <= self.n:
            return 1.0 if self.seen <= self.k else 0.0
        return self.sim.p(cond, level)


def _session(tmp_path, name, cfg, listener, seed=5):
    w = Writer(tmp_path / name, {"config": cfg.to_dict(), "code": "T1", "session_seed": seed})
    drive(Experiment(cfg, w, seed), listener, np.random.default_rng(seed))
    folder = tmp_path / name
    rows = [r["row"] for r in read_jsonl(folder / "trials.jsonl")]
    return folder, json.loads((folder / "session.json").read_text()), rows


# ---------------------------------------------------------------------------------- the gate
def test_the_gate_belongs_to_exp1_and_older_presets_have_none():
    assert EXP1.practice_min_correct == 6 and len(EXP1.practice_levels_db) == 8
    for name in ("tcmr_main", "tcmr_ascending", "tcmr_baseline", "tcmr_frozen"):
        assert Config.load(CONFIGS / f"{name}.json").practice_min_correct == 0
    old = {k: v for k, v in EXP1.to_dict().items() if k != "practice_min_correct"}
    assert Config.from_dict(old).practice_min_correct == 0          # a session stored before the gate
    assert Config.from_dict(old).hash() == "25ac4a4d62854cc1"       # ... keeps the hash it ran under
    for bad in (-1, 9):
        with pytest.raises(ConfigError):
            validate(EXP1.replace(practice_min_correct=bad))


def test_fewer_than_six_right_stops_the_session_after_the_practice(tmp_path):
    folder, meta, rows = _session(tmp_path, "fail", SMALL, Scripted(5))
    assert [r["phase"] for r in rows] == ["example"] * 2 + ["practice"] * 8      # nothing after it
    assert meta["status"] == "practice_failed"
    assert meta["practice"] == {"correct": 5, "trials": 8, "needed": 6, "passed": False}
    events = [e["event"] for e in read_jsonl(folder / "events.jsonl")]
    assert "practice_gate" in events
    s = summarize(folder)
    assert s["practice_correct"] == "5/8" and s["practice_gate_passed"] is False
    assert s["thresholds_db"] == {} and s["main_trials"] == [0, 22]
    assert any("practice gate not passed: 5 of 8" in f for f in s["flags"])
    assert "practice 5/8" in describe(s)


def test_six_right_goes_on_to_the_main_part(tmp_path):
    folder, meta, rows = _session(tmp_path, "pass", SMALL, Scripted(6))
    assert meta["status"] == "complete"
    assert meta["practice"] == {"correct": 6, "trials": 8, "needed": 6, "passed": True}
    assert sum(r["phase"] == "main" and r["status"] == "ok" for r in rows) == 22
    s = summarize(folder)
    assert s["practice_gate_passed"] is True
    assert not any("practice" in f or "not complete" in f for f in s["flags"])


def test_a_session_without_a_gate_never_stops_at_the_practice(tmp_path):
    cfg = SMALL.replace(practice_min_correct=0)
    folder, meta, rows = _session(tmp_path, "nogate", cfg, Scripted(0))
    assert meta["status"] == "complete" and meta["practice"]["passed"] is True
    assert summarize(folder)["practice_gate_passed"] is None


# ---------------------------------------------------------------------- continuing a session
def test_a_session_stopped_at_or_before_the_gate_cannot_be_continued(tmp_path, capsys):
    folder, meta, rows = _session(tmp_path, "fail", SMALL, Scripted(3))
    assert practice_state(SMALL, rows) == {"correct": 3, "trials": 8, "planned": 8, "needed": 6,
                                           "passed": False}
    assert "practice gate (3 of 8 correct, 6 needed)" in resume_blocked(SMALL, meta, rows)
    half = [r for r in rows if r["phase"] != "practice"] + \
        [r for r in rows if r["phase"] == "practice"][:4]
    assert "before the practice was finished (4 of 8" in resume_blocked(SMALL, {"status": "operator_quit"}, half)
    assert "before the practice was finished (0 of 8" in resume_blocked(SMALL, {"status": "interrupted"}, [])
    with pytest.raises(ValueError):                    # the script itself refuses as well
        Experiment(SMALL, Writer(tmp_path / "x", {"config": SMALL.to_dict()}), 5, resume_rows=rows)
    with pytest.raises(ValueError):
        Experiment(SMALL, Writer(tmp_path / "y", {"config": SMALL.to_dict()}), 5, resume_rows=[])
    # the command stops before it opens anything, and leaves the session as it was
    before = (folder / "session.json").read_text()
    with pytest.raises(SystemExit) as e:
        main(["run", "--code", "T1", "--resume", str(folder), "--uncalibrated"])
    assert e.value.code == 2 and "practice gate" in capsys.readouterr().out
    assert (folder / "session.json").read_text() == before
    # a passed gate, a session that is not finished: it may go on; a finished one may not
    ok_folder, ok_meta, ok_rows = _session(tmp_path, "pass", SMALL, Scripted(8))
    assert resume_blocked(SMALL, dict(ok_meta, status="operator_quit"), ok_rows) == ""
    assert resume_blocked(SMALL, ok_meta, ok_rows) == "this session is complete"
    older = SMALL.replace(practice_min_correct=0)       # sessions from before the gate: as before
    assert resume_blocked(older, {"status": "operator_quit"}, []) == ""


def test_the_trial_interrupted_by_a_quit_is_drawn_anew_when_the_session_continues(tmp_path):
    folder, meta, rows = _session(tmp_path, "full", SMALL, Scripted(8))
    main_rows = [r for r in rows if r["phase"] == "main"]
    cut = 5                                             # five trials done, the sixth interrupted
    aborted = dict(main_rows[cut], status="aborted", reason="operator_quit", attempt=0,
                   response=None, correct=None)
    kept = [r for r in rows if r["phase"] != "main"] + main_rows[:cut] + [aborted]
    exp = Experiment(SMALL, Writer(tmp_path / "again", {"config": SMALL.to_dict()}), 5,
                     resume_rows=kept)
    assert exp.done == cut and exp.first_attempt == 1
    gen = exp.script()
    assert next(gen).name == "resume"
    play = gen.send({"key": "space"})
    first = [int(v) for v in main_rows[cut]["seed"].split("-")]
    assert list(play.trial.seed[:4]) == first[:4]                       # the same trial of the same track
    assert first[4] == 0 and play.trial.seed[4] == 1                    # ... as a new draw
    assert play.sha != main_rows[cut]["sha256"]
    nxt = gen.send(Result(response=play.trial.signal_interval, rt_ms=500.0))
    assert nxt.trial.seed[4] == 0                                       # later trials start at 0 again
    # nothing interrupted: the next trial is simply the next one
    clean = [r for r in rows if r["phase"] != "main"] + main_rows[:cut]
    exp2 = Experiment(SMALL, Writer(tmp_path / "clean", {"config": SMALL.to_dict()}), 5,
                      resume_rows=clean)
    assert exp2.first_attempt == 0


# ------------------------------------------------------------------- the top of the level range
def test_a_track_marks_every_call_beyond_the_level_range():
    t = Track(EXP1, EXP1.condition_list()[1], 0)
    levels = []
    for _ in range(5):                                  # wrong every time: 75, 83, then held at 85
        levels.append(t.level)
        t.update(False)
    assert levels == [75.0, 83.0, 85.0, 85.0, 85.0] and t.level == 85.0
    assert [s.limit for s in t.steps] == [0, 1, 1, 1, 1]         # 83 + 8 already asked for 91
    assert t.calls_above_max == 4 and t.at_ceiling and CEILING_CALLS == 3
    easy = Track(EXP1, EXP1.condition_list()[0], 0)
    for _ in range(32):                                 # right every time: down to the floor
        easy.update(True)
    assert easy.level == EXP1.min_db and easy.calls_above_max == 0 and not easy.at_ceiling
    assert any(s.limit == -1 for s in easy.steps)
    replayed = Track.replay(EXP1, t.cond, 0, [False] * 5)        # the analysis counts it the same way
    assert replayed.calls_above_max == 4


def test_a_threshold_above_the_range_is_flagged_and_entered_as_a_lower_bound(tmp_path):
    th = dict(sim_thresholds(EXP1), ref_sync_0=64.0, rand_sync_0=96.0)   # cannot hear it at 85 dB
    w = Writer(tmp_path / "s", {"config": EXP1.to_dict(), "code": "T1", "session_seed": 7})
    drive(Experiment(EXP1, w, 7), Scripted(8), np.random.default_rng(7))  # practice passed ...
    folder, meta, rows = _session(tmp_path, "ceil", EXP1, _PracticeThen(SimListener(th)), seed=7)
    s = summarize(folder)
    assert s["at_ceiling"] == ["rand_sync_0"] and s["calls_above_max"]["rand_sync_0"][0] >= 10
    assert list(s["censored_at_max"]) == ["rand_sync_0"] and s["censored_at_max"]["rand_sync_0"] > 85.0
    assert s["thresholds_db"]["rand_sync_0"] == 85.0 and s["max_db"] == 85.0
    assert s["cmr_db"]["sync_0"] == pytest.approx(85.0 - s["thresholds_db"]["co_sync_0"])
    assert any("rand_sync_0 is at ceiling" in f for f in s["flags"])
    assert any("lower bound" in f for f in s["flags"])
    assert "AT LEAST" in describe(s)
    assert meta["tracks"]["1"]["calls_above_max"] == s["calls_above_max"]["rand_sync_0"][0]
    from tcmr.plots import session_figure
    assert session_figure(folder, s, tmp_path / "fig.png").exists()
    # an ordinary listener: nothing at the top of the range, nothing changed
    plain = summarize(tmp_path / "s")
    assert plain["at_ceiling"] == [] and plain["censored_at_max"] == {}
    assert not any("ceiling" in f or "bound" in f for f in plain["flags"])


class _PracticeThen:
    """Every practice trial right, then the given listener."""

    def __init__(self, listener, n=8):
        self.listener, self.n, self.seen = listener, n, 0

    def p(self, cond, level):
        self.seen += 1
        return 1.0 if self.seen <= self.n else self.listener.p(cond, level)


# ------------------------------------------------------------------------------ calibration
def test_meter_readings_are_read_forgivingly():
    assert parse_level("68.5") == 68.5 and parse_level(" 71 ") == 71.0
    assert parse_level("68,5") == 68.5                                  # decimal comma
    assert parse_level("۶۸٫۵") == 68.5 and parse_level("٧٢") == 72.0    # Persian, Arabic digits
    for bad in ("", "abc", "6.8", "680", "-70"):
        assert parse_level(bad) is None


def test_calibration_notes_warn_about_headroom_and_uneven_bands():
    flat = {f: 68.0 for f in EXP1.bands_hz}                             # 60 dB SPL = 0.0199
    notes = calibration_notes(EXP1, flat)
    assert [m for m, _ in notes] == [" "] and "0.35" in notes[0][1] and "0.63" in notes[0][1]
    quiet = {**flat, 1000.0: 62.0}                                # the tone band 6 dB weaker
    marks = [m for m, _ in calibration_notes(EXP1, quiet)]
    assert marks[0] == "!" and "raise the system volume" in calibration_notes(EXP1, quiet)[0][1]
    uneven = {**flat, 250.0: 50.0}
    assert any(m == "!" and "A weighting" in t for m, t in calibration_notes(EXP1, uneven))
    loud = {f: 74.0 for f in EXP1.bands_hz}
    assert calibration_notes(EXP1, loud) == [(" ", calibration_notes(EXP1, loud)[0][1])]


class _FakePlayback:
    finished = False


class _FakeEngine:
    played = []

    def __init__(self, *a, **k):
        pass

    def start(self):
        return {"latency_s": 0.01}

    def play_at(self, buf, t):
        f = int(round(np.argmax(np.abs(np.fft.rfft(buf[:48000, 0])))))   # which band is playing
        _FakeEngine.played.append(f)
        return _FakePlayback()

    def abort(self, pb):
        pb.finished = True

    def stop(self):
        pass


def _fake_audio(monkeypatch, answers, volumes=(60, 60)):
    import builtins
    import tsync.audio
    import tsync.system
    _FakeEngine.played = []
    vols = list(volumes)
    monkeypatch.setattr(tsync.audio, "AudioEngine", _FakeEngine)
    monkeypatch.setattr(tsync.audio, "find_output", lambda m: {"index": 0, "name": "External Headphones"})
    monkeypatch.setattr(tsync.system, "volume_settings", lambda: {
        "output_volume": vols.pop(0) if len(vols) > 1 else vols[0], "muted": False, "readable": True})
    it = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(it))


def test_calibrate_survives_typos_and_starts_with_the_tone_band(tmp_path, monkeypatch, capsys):
    who = ["HD 280", "meter X", "flat plate", "MA"]
    _fake_audio(monkeypatch, who + ["", "abc", "۷۲٫۵", "70", "71", "73", "69"])
    main(["calibrate", "--config", str(CONFIGS / "tcmr_exp1.json"), "--data", str(tmp_path)])
    assert _FakeEngine.played == [1000, 1000, 250, 500, 2000, 4000]      # 1 kHz first, played twice
    cal = Calibration.from_dict(json.loads((tmp_path / "calibration" / "tcmr_latest.json").read_text()))
    assert cal.device == "External Headphones" and cal.system_volume == 60
    assert cal.amplitude(1000.0, 60.0) == pytest.approx(0.05 * 10 ** (-12.5 / 20))
    assert cal.amplitude(250.0, 60.0) == pytest.approx(0.05 * 10 ** (-10.0 / 20))
    assert list(cal.amp_at_ref) == [f"{f:.6f}" for f in EXP1.bands_hz]
    assert len(list((tmp_path / "calibration").glob("tcmr_calibration_*.json"))) == 1   # a dated copy
    out = capsys.readouterr().out
    assert "not a level between 30 and 120" in out and "--calibrated" in out and "--volume 60" in out


def test_calibrate_saves_nothing_at_a_volume_that_is_too_low_or_that_changed(tmp_path, monkeypatch, capsys):
    who = ["HD 280", "meter X", "flat plate", "MA"]
    _fake_audio(monkeypatch, who + ["60"])                    # 1 kHz reads 60 dB: the 85-dB tone = 0.89
    with pytest.raises(SystemExit) as e:
        main(["calibrate", "--config", str(CONFIGS / "tcmr_exp1.json"), "--data", str(tmp_path)])
    assert e.value.code == 2 and _FakeEngine.played == [1000]
    assert "Raise the system volume" in capsys.readouterr().out
    _fake_audio(monkeypatch, who + ["72", "70", "71", "73", "69"], volumes=(60, 50))
    with pytest.raises(SystemExit) as e:
        main(["calibrate", "--config", str(CONFIGS / "tcmr_exp1.json"), "--data", str(tmp_path)])
    assert e.value.code == 2 and "volume changed" in capsys.readouterr().out
    assert not (tmp_path / "calibration").exists()


def _cal(db_1k, volume=60):
    amps = {f"{f:.6f}": 0.05 * 10 ** ((60.0 - (db_1k if f == 1000.0 else 70.0)) / 20)
            for f in EXP1.bands_hz}
    return Calibration(True, 60.0, amps, "External Headphones", volume)


def test_a_calibrated_seed_check_belongs_to_one_calibration(tmp_path, monkeypatch, capsys):
    tiny = EXP1.replace(name="tiny", trials_per_track=2, conditions=(
        ("co", "sync", 0.0, 1), ("rand", "sync", 0.0, 1), ("ref", "sync", 0.0, 1)))
    a, b = _cal(72.0), _cal(70.0)
    assert cal_id(a) == cal_id(_cal(72.0)) and cal_id(a) != cal_id(b) and cal_id(None) is None
    assert cal_id(a) != cal_id(_cal(72.0, volume=61))
    data = tmp_path / "data"
    rep, _ = _check_seed(tiny, 11, data, a)
    assert rep["ok"] and rep["amplitude"] == "calibrated" and rep["calibration"] == cal_id(a)
    assert next_seed(data, tiny, "calibrated", cal_id(a)) == 11
    assert next_seed(data, tiny, "calibrated", cal_id(b)) is None       # calibrated again: check again
    assert next_seed(data, tiny, "uncalibrated") is None
    # the run itself refuses the stale check, and accepts the matching one
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps(tiny.to_dict()))
    (data / "calibration").mkdir()
    import tcmr.app
    ran = []
    monkeypatch.setattr(tcmr.app, "run_app", lambda exp, *k, **kw: ran.append(exp) or "complete")
    args = ["run", "--config", str(path), "--code", "T1", "--seed", "11", "--data", str(data)]
    for cal, fine in ((b, False), (a, True)):
        (data / "calibration" / "tcmr_latest.json").write_text(json.dumps(cal.to_dict()))
        _fake_audio(monkeypatch, [])
        if fine:
            main(args)
        else:
            with pytest.raises(SystemExit) as e:
                main(args)
            assert e.value.code == 2 and "another calibration" in capsys.readouterr().out
    assert len(ran) == 1
    meta = json.loads((data / "T1" / "tcmr_session_01" / "session.json").read_text())
    assert meta["levels_calibrated"] and meta["seed_check"]["ok"] and meta["session_seed"] == 11
    # the session records the tcmr code itself (tsync's git record covers tsync alone)
    from tcmr.cli import source_state
    assert meta["source_tcmr"] == source_state()[0]["source_tcmr"] and len(meta["source_tcmr"]) == 64
    assert isinstance(meta["tcmr_dirty"], bool)


def test_a_session_continues_only_with_the_levels_it_started_with(tmp_path, monkeypatch, capsys):
    folder, meta, rows = _session(tmp_path, "s", SMALL, Scripted(8))
    main_rows = [r for r in rows if r["phase"] == "main"]
    keep = [r for r in rows if r["phase"] != "main"] + main_rows[:4]
    (folder / "trials.jsonl").write_text("".join(json.dumps({"row": r}) + "\n" for r in keep))
    meta.update(status="operator_quit", levels_calibrated=False, volume_locked=True,
                volume_at_start={"output_volume": 56})
    (folder / "session.json").write_text(json.dumps(meta))
    before = (folder / "session.json").read_text()
    import tcmr.app
    ran = []
    monkeypatch.setattr(tcmr.app, "run_app", lambda exp, *k, **kw: ran.append(exp) or "operator_quit")
    args = ["run", "--code", "T1", "--resume", str(folder), "--data", str(tmp_path / "data")]
    _fake_audio(monkeypatch, [], volumes=(40,))                 # the volume is not what it was
    with pytest.raises(SystemExit) as e:
        main(args + ["--uncalibrated"])
    assert e.value.code == 2 and "--volume 56" in capsys.readouterr().out
    assert (folder / "session.json").read_text() == before and not ran
    _fake_audio(monkeypatch, [], volumes=(56,))                 # same volume, same kind of levels
    main(args + ["--uncalibrated"])
    assert len(ran) == 1 and ran[0].done == 4


# ------------------------------------------------------------------------- the app, with a robot
def test_the_app_stops_after_a_failed_practice(tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Qt
    from tcmr.app import Runner, Window
    from tsync.selftest import SilentEngine

    cfg = Config().replace(trials_per_track=2, n_precursors=1, practice_levels_db=(80.0, 80.0, 80.0),
                           practice_min_correct=2, ratings=(), rating_reps=0, ready_ms=30,
                           feedback_ms=30, blank_ms=30, isi_ms=100.0, break_every_rounds=1)
    w = Writer(tmp_path / "s", {"config": cfg.to_dict(), "code": "BOT"})
    exp = Experiment(validate(cfg), w, 11)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = Window()
    win.resize(1000, 700)
    win.show()
    done = {}
    r = Runner(win, exp, SilentEngine(cfg.sample_rate), None, lambda why: done.update(why=why),
               read_volume=lambda: {"output_volume": 50, "muted": False, "readable": True})
    keys = {"space": Qt.Key.Key_Space, "escape": Qt.Key.Key_Escape,
            "1": Qt.Key.Key_1, "2": Qt.Key.Key_2, "3": Qt.Key.Key_3}

    def press(k):
        QtWidgets.QApplication.sendEvent(win, QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress, keys[k], Qt.KeyboardModifier.NoModifier,
            k if len(k) == 1 else ""))

    st = {"last": None, "n": 0, "screens": []}

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
        elif s == "screen":
            if not st["screens"] or st["screens"][-1] != r.action.name:
                st["screens"].append(r.action.name)
            press(r.action.keys[0] if r.action.keys[0] in keys else "space")
        elif s == "respond" and st["last"] is not r.play:
            st["last"] = r.play
            st["n"] += 1
            right = r.play.trial.signal_interval
            press(str(right if st["n"] == 1 else right % 3 + 1))    # 1 of 3 right: below the gate

    t = QtCore.QTimer()
    t.timeout.connect(bot)
    t.start(15)
    QtCore.QTimer.singleShot(10, r.start)
    t_end = time.time() + 120
    while "why" not in done and time.time() < t_end:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
        time.sleep(0.002)
    t.stop()
    assert done.get("why") == "practice_failed"
    assert st["screens"][-1] == "stopped" and "main" not in st["screens"]
    rows = [x["row"] for x in read_jsonl(tmp_path / "s" / "trials.jsonl")]
    assert [x["phase"] for x in rows] == ["example"] * 2 + ["practice"] * 3
    meta = json.loads((tmp_path / "s" / "session.json").read_text())
    assert meta["status"] == "practice_failed" and meta["practice"]["correct"] == 1
