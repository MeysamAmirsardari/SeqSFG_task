"""tcbound: the rove scheme keeps exactly one cue per type, the session machinery is exact."""
import contextlib
import csv
import io
import math

import numpy as np
import pytest

from tcbound.config import (DEFAULT as CFG, Condition, ConfigError, conditions, delta_db,
                            srs_db, validate, worst_case_peak)
from tcbound.observer import SURVIVING, ceilings, cues, draw_gains
from tcbound.stimulus import (build_trial, draw_latent, gains_from_json, gains_from_latent,
                              gains_json, render_interval, render_trial)
from tcbound.track import Track

T = CFG.target_index
OTH = CFG.others
COND = {c.name: c for c in conditions(CFG)}


def _quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


# ---- the rove scheme ------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["A_0", "A_32", "A_cap", "B_0", "B_64", "B_cap", "B_alone"])
def test_each_type_keeps_its_cue_exactly(name):
    c = COND[name]
    rng = np.random.default_rng(1)
    for delta in (0.5, 4.0, 11.0):
        s = cues(CFG, draw_gains(CFG, c, delta, True, 200, rng))[SURVIVING[c.task]]
        n = cues(CFG, draw_gains(CFG, c, delta, False, 200, rng))[SURVIVING[c.task]]
        assert np.allclose(s - n, delta, atol=1e-9)


@pytest.mark.parametrize("task,destroyed", [("A", "own"), ("B", "timbre"),
                                            ("B", "timbre_change"), ("A", "target_level")])
def test_the_other_cues_are_noise(task, destroyed):
    c = Condition("x", task, 0.0)
    g = draw_gains(CFG, c, 0.0, False, 3000, np.random.default_rng(2))
    assert np.std(cues(CFG, g)[destroyed]) > 6.0


def test_repetitions_before_the_probe_are_the_same_code_in_both_types():
    lat = draw_latent(CFG, np.random.default_rng(3))
    a, _ = gains_from_latent(CFG, "A", "figure", lat, 5.0, True)
    b, _ = gains_from_latent(CFG, "B", "figure", lat, 5.0, True)
    assert np.array_equal(a[:-1], b[:-1])
    c, d, v, w, *u = lat
    assert np.allclose(a[-1], [c + v, c + v, c + v + 5.0, c + v])            # flat, plus delta
    assert b[-1, T] == pytest.approx(c + d + 5.0)                           # the target holds
    assert np.allclose(b[-1, OTH], c + w + np.asarray(u))                   # the others jump


def test_controls_have_the_right_tones():
    cap = build_trial(CFG, COND["A_cap"], 0.0, np.random.default_rng(4))
    assert cap.first.captor_onsets_ms is not None
    assert np.all(np.isnan(cap.first.captor_onsets_ms[:, T]))
    assert np.all(np.isfinite(cap.first.captor_onsets_ms[:, OTH]))
    alone = build_trial(CFG, COND["B_alone"], 0.0, np.random.default_rng(4))
    assert np.all(np.isnan(alone.first.gains_db[:, OTH]))
    assert np.all(np.isfinite(alone.first.gains_db[:, T]))
    catch = build_trial(CFG, Condition("c", "none", 0.0), srs_db(8.0), np.random.default_rng(4))
    std = catch.standard.gains_db
    assert np.allclose(std, 0.0)
    assert catch.signal.gains_db[-1, T] == pytest.approx(8.0)


def test_srs_and_increment_convert_both_ways():
    for s in (-30.0, -6.0, 0.0, 9.5):
        assert srs_db(delta_db(s)) == pytest.approx(s)
    assert delta_db(9.5) == pytest.approx(12.0, abs=0.01)


def test_gain_json_round_trip_keeps_absent_tones_absent():
    g = build_trial(CFG, COND["B_alone"], 0.0, np.random.default_rng(0)).first.gains_db
    back = gains_from_json(gains_json(g))
    assert np.allclose(g, back, atol=1e-4, equal_nan=True)


# ---- the waveform -----------------------------------------------------------------------------
def test_the_waveform_carries_the_gains():
    tr = build_trial(CFG, COND["B_16"], 3.0, np.random.default_rng(5))
    fs, n = CFG.sample_rate, int(round(CFG.tone_ms * CFG.sample_rate / 1000))
    for iv in tr.intervals():
        for k in range(CFG.n_tones):
            x = render_interval(CFG, iv, tr.phases, tr.captor_phases, tr.total_ms, only_tone=k)
            for r in range(CFG.n_repeats):
                i = int(round(iv.onsets_ms[r, k] * fs / 1000))
                seg = x[i + n // 4: i + 3 * n // 4]
                db = 20 * np.log10(np.sqrt(2 * np.mean(seg ** 2)) / CFG.tone_amplitude)
                assert db == pytest.approx(iv.gains_db[r, k], abs=0.05)


def test_both_intervals_have_the_same_length_and_nothing_clips():
    assert worst_case_peak(CFG)["worst"] < 0.9
    rng = np.random.default_rng(6)
    for _ in range(60):
        tr = build_trial(CFG, COND["B_0"], CFG.srs_max_db, rng)
        x = render_trial(CFG, tr)
        assert np.abs(x).max() < worst_case_peak(CFG)["worst"] + 1e-9
        a = render_interval(CFG, tr.first, tr.phases, tr.captor_phases, tr.total_ms)
        b = render_interval(CFG, tr.second, tr.phases, tr.captor_phases, tr.total_ms)
        assert a.size == b.size


def test_validation_refuses_what_would_break_the_design():
    with pytest.raises(ConfigError):
        validate(CFG.replace(target_index=0))
    with pytest.raises(ConfigError):
        validate(CFG.replace(steps_ms=(0.0, 100.0)))                # does not fit the period
    with pytest.raises(ConfigError):
        validate(CFG.replace(tone_amplitude=0.05))                  # clips at the top of the rove
    with pytest.raises(ConfigError):
        validate(CFG.replace(freqs_hz=(500.0, 1000.0, 1500.0, 2000.0)))   # harmonic


def test_every_shortcut_needs_more_than_eight_db():
    for task, kind in (("A", "figure"), ("B", "figure"), ("B", "alone")):
        ce = ceilings(CFG, Condition("x", task, 0.0, kind), n=8000)
        assert ce["ceiling_db"] > 8.0
        assert SURVIVING[task] in ce["exact"]


# ---- the staircase -----------------------------------------------------------------------------
def test_the_track_is_additive_in_srs_and_averages_the_last_four_reversals():
    t = Track(CFG, "x")
    seq = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0]
    for ok in seq:
        if t.finished:
            break
        t.update(bool(ok))
    levels = [x.srs_db for x in t.trials]
    assert levels[:4] == [4.0, 4.0, 4.0, 0.0]
    steps = {round(abs(b - a), 6) for a, b in zip(levels, levels[1:]) if b != a}
    assert steps <= {4.0, 2.0}
    if t.threshold_db() is not None:
        final = [x for x, si in t.reversals if si == 1][-4:]
        assert t.threshold_db() == pytest.approx(np.mean(final))


def test_a_track_pinned_at_the_cap_is_flagged_as_a_lower_bound():
    t = Track(CFG, "x")
    rng = np.random.default_rng(0)
    while not t.finished:
        t.update(bool(rng.random() < 0.5))        # a listener at chance climbs to the cap
    a = t.audit()
    if a["converged"]:
        assert a["ceiling_limited"] == (max(a["reversals_used_srs_db"]) >= CFG.srs_max_db - 1e-9)


# ---- the session --------------------------------------------------------------------------------
def _run(tmp_path, code="T01", seed=1, mode="coherence", cfg=CFG, stop_after=None):
    from tcbound.runner import QuitRequested, Runner

    class Stopper(Runner):
        n = 0

        def _present(self, *a, **k):
            Stopper.n += 1
            if stop_after is not None and Stopper.n > stop_after:
                raise QuitRequested()
            return super()._present(*a, **k)
    r = Stopper(cfg, tmp_path, auto=mode, seed=seed)
    return _quiet(r.run, code=code), r


def _log(sdir):
    with open(sdir / "trials.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_a_simulated_session_completes_and_replays_from_its_log(tmp_path):
    from tcbound.analysis import analyse_session
    sdir, r = _run(tmp_path)
    a = analyse_session(sdir)
    dg = a["diagnostics"]
    assert dg["status"] == "complete" and dg["usable"]
    assert dg["log_replays_cleanly"]
    assert dg["n_converged"] >= 12
    w = a["wcurve"]["w"]
    assert w[0] > 0.7 and w[-1] < 0.45                 # the weights see the binding change


def test_a_resumed_session_is_the_same_session(tmp_path):
    full, _ = _run(tmp_path / "a", code="R01", seed=5)
    part, _ = _run(tmp_path / "b", code="R01", seed=5, stop_after=120)
    from tcbound.runner import Runner
    r = Runner(CFG, tmp_path / "b", auto="coherence", seed=5)
    _quiet(r.run, code="R01", resume=True, session_index=1)
    cols = ("phase", "slot_index", "condition", "srs_db", "gains_1", "gains_2", "response")
    a = [tuple(x[c] for c in cols) for x in _log(full)]
    b = [tuple(x[c] for c in cols) for x in _log(part)]
    assert a == b


def test_export_rebuilds_every_trial_exactly(tmp_path):
    from tcbound.export import export_session
    sdir, _ = _run(tmp_path, code="X01", seed=2)
    paths = export_session(sdir, tmp_path / "out")
    z = np.load(paths[0])
    assert len(z["condition"]) == sum(1 for r in _log(sdir)
                                      if r["phase"] in ("main", "catch", "filler"))


def test_the_order_never_runs_long_on_one_condition_or_one_type(tmp_path):
    sdir, _ = _run(tmp_path, code="O01", seed=3)
    rows = [r for r in _log(sdir) if r["phase"] in ("main", "catch", "filler")]
    run_c = run_t = 1
    for a, b in zip(rows, rows[1:]):
        run_c = run_c + 1 if a["condition"] == b["condition"] else 1
        run_t = run_t + 1 if a["task"] == b["task"] else 1
        assert run_c <= CFG.max_same_condition_run
        assert run_t <= CFG.max_same_task_run


def test_catch_trials_never_touch_a_staircase(tmp_path):
    from tcbound.analysis import replay_tracks
    sdir, _ = _run(tmp_path, code="C01", seed=4)
    rows = _log(sdir)
    assert any(r["phase"] == "catch" for r in rows)
    rep = replay_tracks(rows, CFG)
    assert all(a["logged_levels_match"] for a in rep.values())
