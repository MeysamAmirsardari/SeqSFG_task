"""tsync: the stimulus, the procedure, the logging, resuming, the audio engine and the app."""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from tsync.config import Config, ConfigError, validate
from tsync.design import balance_report, clean_code, make_plan, williams_square
from tsync.experiment import Experiment
from tsync.levels import uncalibrated
from tsync.logbook import read_session
from tsync.session import open_session, resume_session
from tsync.sim import ModelListener, SimulatedCrash, drive, estimate_duration
from tsync.staircase import Staircase
from tsync.stimulus import (build_trial, buffer_sha256, ending_onsets_ms, render,
                            sequence_onsets_ms)
from tsync.text import DEBRIEF, QUESTIONNAIRE, T
from tsync.verify import PRESET, PRESET_HASH, measure_ending_onsets

CFG = Config()
CAL = uncalibrated(CFG)


def test_preset_is_frozen_and_equals_the_defaults():
    p = Config.load(PRESET)
    assert p == CFG
    assert p.hash() == PRESET_HASH
    validate(p)


def test_validate_refuses_a_ceiling_that_makes_tones_touch():
    with pytest.raises(ConfigError):
        validate(CFG.replace(delta_max_ms=150.0))
    with pytest.raises(ConfigError):
        validate(CFG.replace(isi_ms=(150.0, 200.0)))


def test_standard_ending_is_synchronous_and_delta_is_split_between_a_and_b():
    for c in CFG.conditions():
        for sign in (1, -1):
            b0, a0 = ending_onsets_ms(CFG, c, 0.0, sign)
            assert b0 == pytest.approx(a0)
            b, a = ending_onsets_ms(CFG, c, 12.0, sign)
            assert a - b == pytest.approx(12.0 * sign)
            assert b0 - b == pytest.approx(6.0 * sign)
            assert a - a0 == pytest.approx(6.0 * sign)


def test_ending_sits_midway_between_the_two_streams_and_the_control_matches_d100():
    grid = (CFG.n_tones - 1) * CFG.soa_ms
    for c in CFG.conditions():
        b, _ = ending_onsets_ms(CFG, c, 0.0, 1)
        assert b - grid == pytest.approx(c.context_ms / 2 + c.ending_shift_ms)
    assert ending_onsets_ms(CFG, CFG.condition("D0_late"), 0, 1) == \
        ending_onsets_ms(CFG, CFG.condition("D100"), 0, 1)


def test_delta_never_touches_the_context():
    c = CFG.condition("D71")
    b1, a1 = sequence_onsets_ms(CFG, c, 0.0, 1)
    b2, a2 = sequence_onsets_ms(CFG, c, 60.0, -1)
    assert b1[:-1] == b2[:-1] and a1[:-1] == a2[:-1]
    assert all(a - b == pytest.approx(71.0) for b, a in zip(b1[:-1], a1[:-1]))


@pytest.mark.parametrize("name,delta,sign", [("D0", 1.0, 1), ("D100", 7.5, -1),
                                              ("D0_late", 40.0, 1)])
def test_asynchrony_measured_in_the_waveform_equals_delta(name, delta, sign):
    tr = build_trial(CFG, CFG.condition(name), delta, (1, 2, 3), target_interval=1, sign=sign)
    buf, _ = render(CFG, tr, CAL.amplitude)
    mb, ma = measure_ending_onsets(CFG, buf, tr)
    got = ((ma[0] - mb[0]) - (ma[1] - mb[1])) * 1000.0
    assert got == pytest.approx(sign * delta, abs=0.02)


def test_trials_regenerate_bit_for_bit_from_their_seed():
    c = CFG.condition("D35")
    t1 = build_trial(CFG, c, 9.0, (4, 5, 6))
    t2 = build_trial(CFG, c, 9.0, (4, 5, 6), target_interval=t1.target_interval, sign=t1.sign)
    assert buffer_sha256(render(CFG, t1, CAL.amplitude)[0]) == \
        buffer_sha256(render(CFG, t2, CAL.amplitude)[0])
    t3 = build_trial(CFG, c, 9.0, (4, 5, 7))
    assert buffer_sha256(render(CFG, t1, CAL.amplitude)[0]) != \
        buffer_sha256(render(CFG, t3, CAL.amplitude)[0])


def test_a_levels_follow_micheyl():
    tr = build_trial(CFG, CFG.condition("D50"), 5.0, (8,))
    for iv in (1, 2):
        a = [t for t in tr.tones if t.stream == "A" and t.interval == iv]
        assert sum(t.decoy for t in a) in (1, 2)
        assert all(abs(t.rove_db) <= 4.0 for t in a)
        b = [t for t in tr.tones if t.stream == "B" and t.interval == iv]
        assert all(t.level_db == 60.0 for t in b)


def test_staircase_follows_the_published_rule():
    s = Staircase(CFG)
    for c in (True, True, True):
        s.update(c)
    assert s.delta == pytest.approx(5.0)              # 20 / 4
    s.update(False)                                    # first reversal: factor becomes 2
    assert s.step_index == 1 and s.delta == pytest.approx(10.0)
    assert s.reversals == [(5.0, 0)]
    rng = np.random.default_rng(0)
    lst = ModelListener({"x": 6.0})
    s = Staircase(CFG)
    while not s.finished:
        s.update(rng.random() < lst.p_correct("x", s.delta))
    assert s.stop_reason == "converged"
    used = s.averaged_reversals()
    assert len(used) == 6
    assert s.threshold_ms() == pytest.approx(math.exp(np.mean(np.log(used))))
    r = Staircase.replay(CFG, [st.correct for st in s.steps])
    assert [x.delta_ms for x in r.steps] == [x.delta_ms for x in s.steps]


def test_order_is_a_balanced_williams_square():
    rep = balance_report(williams_square(6))
    assert rep == {"positions_balanced": True, "carryover_balanced": True}
    assert make_plan(CFG, 1, 5).blocks != make_plan(CFG, 2, 5).blocks
    assert make_plan(CFG, 1, 5).blocks == make_plan(CFG, 7, 5).blocks


def test_participant_codes_are_cleaned():
    assert clean_code(" p۰۱ ") == "P01"
    with pytest.raises(ValueError):
        clean_code("P 01")


def test_participant_text_has_no_em_dashes():
    texts = list(T.values()) + [f["label"] for f in QUESTIONNAIRE + DEBRIEF]
    assert not any("—" in t for t in texts)


def _session(tmp_path, code="T01", **kw):
    w, plan = open_session(CFG, tmp_path, code, CAL, simulated=True, session_seed=11)
    return w, plan


def test_a_whole_simulated_session_is_logged_completely(tmp_path):
    from tsync.analysis import summarize
    w, plan = _session(tmp_path)
    drive(Experiment(CFG, CAL, plan, w), ModelListener(p_abort=0.03), np.random.default_rng(3))
    s = read_session(w.folder)
    rows = [t["row"] for t in s["trials"]]
    assert s["meta"]["status"] == "complete"
    assert sum(1 for _ in open(w.folder / "trials.csv")) - 1 == len(rows)
    assert {r["phase"] for r in rows} >= {"check", "demo", "practice", "exposure", "main"}
    assert any(r["status"] == "aborted" for r in rows)
    summ = summarize(w.folder, n_boot=20)
    assert len(summ["tracks"]) == 6
    assert all(t["converged"] and t["consistent"] for t in summ["tracks"])
    catch = [r for r in rows if r["is_catch"] and r["status"] == "ok"]
    assert len(catch) == 12 and all(r["step_index"] is None for r in catch)
    full = s["trials"][-1]
    assert len(full["tones"]) == 4 * CFG.n_tones


def test_a_crashed_session_resumes_where_it_stopped(tmp_path):
    w, plan = _session(tmp_path, "R01")
    rng = np.random.default_rng(5)
    with pytest.raises(SimulatedCrash):
        drive(Experiment(CFG, CAL, plan, w), ModelListener(), rng, crash_after_trials=110)
    before = [t["row"] for t in read_session(w.folder)["trials"]]
    w2, plan2, cal2, state = resume_session(w.folder, CFG)
    assert state.main_started and plan2.blocks == plan.blocks
    drive(Experiment(CFG, cal2, plan2, w2, resume=state), ModelListener(), rng)
    s = read_session(w.folder)
    rows = [t["row"] for t in s["trials"]]
    assert s["meta"]["status"] == "complete" and len(rows) > len(before)
    for b in range(6):
        rs = [r for r in rows if r["phase"] == "main" and r["block"] == b
              and r["status"] == "ok" and not r["is_catch"]]
        st = Staircase.replay(CFG, [r["correct"] for r in rs])
        assert st.finished and [x.delta_ms for x in st.steps] == [r["delta_ms"] for r in rs]
    seeds = [r["seed"] for r in rows]
    assert len(seeds) == len(set(seeds))


def test_session_fits_in_45_minutes():
    d = estimate_duration(CFG, n=24, seed=1)
    assert d["p90_min"] < 45.0


class _Info:
    def __init__(self, dac):
        self.outputBufferDacTime, self.currentTime = dac, dac - 0.01


def test_audio_callback_copies_stamps_and_aborts_cleanly():
    from tsync.audio import AudioEngine, Playback
    eng = AudioEngine(None, 1000, 1, 10)
    eng._stream = None
    buf = np.arange(1, 36, dtype=np.float32)[:, None]
    pb = Playback(buf, 25, 1000)
    eng._pb = pb
    outs = []
    for k in range(8):
        out = np.zeros((10, 1), np.float32)
        eng._callback(out, 10, _Info(1.0 + k * 0.01), 0)
        outs.append(out[:, 0].copy())
    sig = np.concatenate(outs)
    assert np.array_equal(sig[25:60], buf[:, 0]) and not sig[:25].any()
    assert pb.onset_dac == pytest.approx(1.025) and pb.end_dac == pytest.approx(1.060)
    late = Playback(buf, 5, 1000)
    eng._pb = late
    eng._callback(np.zeros((10, 1), np.float32), 10, _Info(2.0), 0)
    assert late.late and late.finished
    pb = Playback(np.ones((500, 1), np.float32), eng._frame + 10, 1000)
    eng._pb = pb
    eng._fade = 20
    out = np.zeros((10, 1), np.float32)
    eng._callback(out, 10, _Info(3.0), 0)
    pb.abort_frame = eng._frame
    tail = []
    for k in range(4):
        out = np.zeros((10, 1), np.float32)
        eng._callback(out, 10, _Info(3.01 + k * 0.01), 0)
        tail.append(out[:, 0].copy())
    tail = np.concatenate(tail)
    assert pb.finished and pb.aborted
    assert np.all(np.diff(tail[:20]) <= 1e-7) and not tail[20:].any()


def test_the_app_runs_a_whole_session_with_a_robot_participant(tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from tsync.selftest import run_selftest
    rep = run_selftest(tmp_path, timeout_s=150)
    assert rep.status == "complete" and rep.why == "complete"
    rows = [t["row"] for t in read_session(rep.folder)["trials"]]
    assert any(r["abort_reason"] == "pause" for r in rows)
    main = [r for r in rows if r["phase"] == "main" and r["status"] == "ok"]
    assert main and all(r["response"] in (1, 2) for r in main if not r["is_catch"])
    assert all(r["rt_ms"] is not None and r["rt_ms"] >= 0 for r in main)
    meta = json.loads((rep.folder / "session.json").read_text())
    assert meta["participant_answers"]["age"] is not None and "debrief" in meta
