"""Tests for the sheared four-tone figure task.

The claims worth testing are the ones the design's argument rests on:
  * the two intervals of a trial differ in ONE tone of ONE repetition and nothing else;
  * the target's own channel is isochronous at every step, which is what makes a flat curve a
    positive result about strategy rather than a null result about binding;
  * the index the design uses is monotone in the step and reduces to the published one for two
    channels;
  * the analysis separates a figure listener from a within-channel listener.
"""
import csv
import io
import json
import contextlib
from pathlib import Path

import numpy as np
import pytest

from tshear import stimulus as S
from tshear import verify as V
from tshear.analysis import load, replay_tracks, synchrony_advantage, thresholds, trend_test
from tshear.config import (DEFAULT, Condition, Config, ConfigError, combined_gaps_ms,
                           conditions, is_isochronous, onsets_ms, validate)
from tshear.design import audit_design, duration_estimate, make_design, session_seed
from tshear.model import effective_objects, two_channel_check
from tshear.runner import Runner

CFG = DEFAULT
D = validate(CFG)
CONDS = D.conditions


# ============================================================ geometry
def test_the_isochronous_step_is_the_period_over_the_tone_count():
    assert CFG.iso_step_ms == pytest.approx(CFG.period_ms / CFG.n_tones)
    assert is_isochronous(CFG, 100.0)
    assert not any(is_isochronous(CFG, p) for p in CFG.step_pcts if 0 < p < 100)
    g = combined_gaps_ms(CFG, 100.0)
    assert np.allclose(g, CFG.iso_step_ms)


def test_every_channel_is_isochronous_at_every_step():
    """The reason a flat curve means 'the figure was ignored' rather than 'nothing happened'."""
    for c in CONDS:
        o = S.figure_onsets(CFG, c)
        for k in range(CFG.n_tones):
            col = o[:, k]
            if not np.isfinite(col).all():
                continue                      # absent in the single-tone control
            assert np.allclose(np.diff(col), CFG.period_ms)


def test_the_shear_puts_tone_k_exactly_k_steps_after_the_first():
    for p in (0.0, 15.0, 50.0, 100.0):
        o = onsets_ms(CFG, p, rep=0)
        assert np.allclose(np.diff(o), CFG.step_ms(p))


# ============================================================ stimulus invariants
def test_only_the_target_tone_of_the_last_repetition_moves():
    for c in CONDS:
        for direction in (+1, -1):
            tr = S.build_trial(CFG, c, 9.0, np.random.default_rng(2), target_position=1,
                               direction=direction)
            diff = np.nan_to_num(tr.first.onsets_ms - tr.second.onsets_ms, nan=0.0)
            moved = np.argwhere(np.abs(diff) > 1e-12)
            assert len(moved) == 1
            assert tuple(moved[0]) == (CFG.n_repeats - 1, CFG.target_index)
            assert diff[tuple(moved[0])] == pytest.approx(direction * 9.0)


def test_no_other_channel_differs_between_the_intervals():
    for c in CONDS:
        tr = S.build_trial(CFG, c, 9.0, np.random.default_rng(5), target_position=1)
        for j in range(CFG.n_tones):
            if j == CFG.target_index:
                continue
            a = S.Interval(tr.first.onsets_ms, tr.first.delta_ms, 0.0, tr.first.step_pct,
                           tr.first.kind)
            b = S.Interval(tr.second.onsets_ms, tr.second.delta_ms, 0.0, tr.second.step_pct,
                           tr.second.kind)
            assert np.array_equal(S.render_interval(CFG, a, tr.phases, only_tone=j),
                                  S.render_interval(CFG, b, tr.phases, only_tone=j))


def test_every_interval_is_the_same_length():
    lens = {S.render_interval(CFG, iv, tr.phases).size
            for c in CONDS
            for tr in [S.build_trial(CFG, c, CFG.delta_max_ms, np.random.default_rng(1))]
            for iv in (tr.first, tr.second)}
    assert len(lens) == 1


def test_the_single_tone_control_contains_only_the_target():
    c = next(x for x in CONDS if x.kind == "single")
    o = S.figure_onsets(CFG, c)
    present = [k for k in range(CFG.n_tones) if np.isfinite(o[:, k]).all()]
    assert present == [CFG.target_index]
    tr = S.build_trial(CFG, c, 9.0, np.random.default_rng(1))
    x = S.render_interval(CFG, tr.first, tr.phases)
    only = S.render_interval(CFG, tr.first, tr.phases, only_tone=CFG.target_index)
    assert np.array_equal(x, only)


def test_delta_is_clamped_and_the_realised_shift_is_what_was_rendered():
    q = 1000.0 / CFG.sample_rate
    for want, req in ((CFG.delta_max_ms, 1e6), (CFG.delta_min_ms, 1e-9)):
        tr = S.build_trial(CFG, CONDS[0], req, np.random.default_rng(1))
        assert tr.delta_ms == pytest.approx(want)
    for dd in (0.25, 1.0, 7.3, 50.0):
        tr = S.build_trial(CFG, CONDS[0], dd, np.random.default_rng(1), direction=+1)
        assert abs(abs(tr.delta_realised_ms) - dd) <= q


def test_crossing_a_neighbour_is_flagged_not_prevented():
    step15 = next(c for c in CONDS if c.name == "step_15")
    small = S.build_trial(CFG, step15, 2.0, np.random.default_rng(1), direction=+1)
    big = S.build_trial(CFG, step15, 40.0, np.random.default_rng(1), direction=+1)
    assert not small.crosses_neighbour
    assert big.crosses_neighbour          # 40 ms is well past the 12.5 ms step
    zero = next(c for c in CONDS if c.name == "step_0")
    assert not S.build_trial(CFG, zero, 40.0, np.random.default_rng(1)).crosses_neighbour


# ============================================================ config guards
def test_validate_rejects_unresolved_or_harmonic_tone_sets():
    with pytest.raises(ConfigError, match="ERB"):
        validate(CFG.replace(freqs_hz=(683.0, 700.0, 2241.0, 3985.0)))
    with pytest.raises(ConfigError, match="harmonic"):
        validate(CFG.replace(freqs_hz=(1000.0, 1750.0, 2750.0, 4250.0)))   # 4,7,11,17 of 250
    with pytest.raises(ConfigError, match="target_index"):
        validate(CFG.replace(target_index=9))


def test_an_edge_target_is_allowed_but_called_out():
    d = validate(CFG.replace(target_index=0))
    assert any("EDGE" in n for n in d.notes)
    assert not any("EDGE" in n for n in validate(CFG).notes)


def test_the_delta_ceiling_is_the_same_channel_constraint():
    from tshear.config import max_safe_delta_ms
    assert max_safe_delta_ms(CFG) == pytest.approx(CFG.period_ms - CFG.tone_ms - 25.0)
    # step 0 has zero-width gaps in the combined train by construction; that is the stimulus,
    # not a collision, and it must not shrink the usable range
    assert max_safe_delta_ms(CFG) > CFG.delta_max_ms


# ============================================================ model
def test_the_index_is_monotone_in_the_step_where_lambda2_is_not():
    pcts = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    vals = [effective_objects(CFG, p) for p in pcts]
    pr = [v["normalised"] for v in vals]
    assert all(b - a >= -1e-9 for a, b in zip(pr, pr[1:]))
    assert pr[0] == pytest.approx(0.0, abs=1e-6)
    l2 = [v["lambda2_over_lambda1"] for v in vals]
    assert not all(b - a >= -1e-9 for a, b in zip(l2, l2[1:]))     # the reason for the change


def test_the_index_reduces_to_the_published_one_for_two_channels():
    tc = two_channel_check()
    assert tc["monotone"]
    assert tc["at_0"] == pytest.approx(0.0)
    assert tc["at_1"] == pytest.approx(1.0)


def test_step_zero_is_exactly_one_object():
    assert effective_objects(CFG, 0.0)["participation_ratio"] == pytest.approx(1.0, abs=1e-6)


# ============================================================ design
def test_seeds_are_independent_across_participants_and_sessions():
    seeds = {(c, i): session_seed(CFG, c, i)
             for c in ("A", "B", "C", "D") for i in (1, 2, 3)}
    assert len(set(seeds.values())) == len(seeds)


def test_the_order_does_not_confound_condition_with_time():
    a = audit_design(CFG, make_design(CFG, "X", 1))
    assert a["balanced_track_count"]
    assert a["round_balance_spread"] == pytest.approx(0.0)
    assert a["serial_position_spread"] < 0.15
    assert a["run_constraint_respected"]


# ============================================================ end to end
@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    out = {}
    for mode in ("figure", "within"):
        d = tmp_path_factory.mktemp(mode)
        cfg = CFG.replace(tracks_per_condition=2)
        with contextlib.redirect_stdout(io.StringIO()):
            out[mode] = (cfg, Runner(cfg, d, audio=False, auto=mode, seed=9).run(code="SIM"))
    return out


def test_a_session_runs_converges_and_replays_from_its_own_log(sessions):
    cfg, sdir = sessions["figure"]
    rows, metas = load([sdir])
    assert metas[0]["status"] == "complete"
    rep = replay_tracks(rows, cfg)
    assert len(rep) == len(validate(cfg).conditions) * cfg.tracks_per_condition
    assert all(a["logged_deltas_match"] for a in rep.values())
    assert all(a["converged"] for a in rep.values())


def test_catch_trials_never_update_a_staircase(sessions):
    cfg, sdir = sessions["figure"]
    rows = list(csv.DictReader(open(sdir / "trials.csv")))
    catch = [r for r in rows if r["phase"] == "catch"]
    assert catch
    for r in catch:
        assert r["track_trial_index"] == "" and r["step_index"] == ""
        assert float(r["delta_ms"]) == pytest.approx(cfg.catch_delta_ms)


def test_a_figure_listener_and_a_within_channel_listener_are_told_apart(sessions):
    """The design's whole purpose, checked end to end."""
    cfg, sdir = sessions["figure"]
    rows, _ = load([sdir])
    res = thresholds(rows, cfg)
    t = trend_test(res, n_perm=2000, seed=1)
    assert t["ok"] and t["rho"] > 0.4 and t["p_one_sided"] < 0.05
    s = synchrony_advantage(res)
    assert s["ok"] and s["ratio"] > 1.0        # the figure beat the tone alone

    cfg2, sdir2 = sessions["within"]
    rows2, _ = load([sdir2])
    t2 = trend_test(thresholds(rows2, cfg2), n_perm=2000, seed=1)
    # The claim is that a within-channel strategy cannot produce a RISING curve, and the trend
    # test is one-sided for exactly that reason. Asserting a two-sided bound on rho instead
    # fails whenever twelve tracks happen to scatter downwards, which says nothing about the
    # design. Measured over twelve independent simulated sessions this test fires on 8% of
    # them, against its nominal 5%.
    assert t2["ok"] and t2["p_one_sided"] > 0.05


def test_resume_reproduces_an_uninterrupted_session(tmp_path):
    cfg = CFG.replace(tracks_per_condition=1)

    def run(**kw):
        with contextlib.redirect_stdout(io.StringIO()):
            return Runner(cfg, tmp_path, audio=False, auto="figure", seed=4).run(code="R", **kw)

    sdir = run()
    full = list(csv.DictReader(open(sdir / "trials.csv")))
    keep = full[: len(full) // 2]
    with open(sdir / "trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=full[0].keys()); w.writeheader(); w.writerows(keep)
    meta = json.loads((sdir / "session.json").read_text()); meta["status"] = "started"
    (sdir / "session.json").write_text(json.dumps(meta))
    run(resume=True, session_index=1)
    after = list(csv.DictReader(open(sdir / "trials.csv")))
    assert after[: len(keep)] == keep
    slots = [r["slot_index"] for r in after if r["slot_index"]]
    assert len(slots) == len(set(slots))
    # the stimulus draws are a function of the slot, so shared slots must agree exactly
    def stim(rs):
        return {r["slot_index"]: (r["target_position"], r["direction"], r["rove_db_1"])
                for r in rs if r["slot_index"]}
    a, b = stim(full), stim(after)
    shared = set(a) & set(b)
    assert len(shared) > len(keep) // 2
    assert all(a[s] == b[s] for s in shared)


def test_the_whole_battery_passes():
    battery = V.run_battery(CFG)
    failed = [c.name for v in battery.values() for c in v if not c.passed]
    assert not failed, failed


def test_duration_is_reported_with_its_parts():
    e = duration_estimate(CFG.replace(tracks_per_condition=1))
    assert e["total_minutes"] == pytest.approx(
        e["main_minutes"] + e["practice_minutes"] + e["break_minutes"] + e["setup_minutes"])
    assert e["worst_case_minutes"] > e["total_minutes"]
