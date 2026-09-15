"""Tests for the two-tone coherence task: stimulus, staircase, design, runner, analysis.

The claims worth testing are the ones the experiment's argument rests on, so most of this file
is about invariants rather than about return types:

  * the two intervals of a trial differ in one tone's position and nothing else;
  * the B channel is the same in every condition, so the single-channel cue cannot vary with dT;
  * the staircase implements the published rule and recovers a threshold it was given;
  * the order of the session does not confound condition with time;
  * the analysis finds the effect when it is there and not when it is not.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tcoh import stimulus as S
from tcoh import verify as V
from tcoh.analysis import (coherence_index, diagnostics, interaction_test, load, replay_tracks,
                           thresholds, trend_test)
from tcoh.config import DEFAULT, Config, conditions, validate
from tcoh.design import audit_design, duration_estimate, make_design, session_seed
from tcoh.observer import SimulatedListener
from tcoh.psychometric import PF, fit, target_proportion
from tcoh.runner import Runner
from tcoh.track import Track, simulate

CFG = DEFAULT
D = validate(CFG)
CONDS = conditions(CFG)


# ============================================================ stimulus
def test_b_grid_is_isochronous_and_independent_of_condition():
    b = S.b_onsets(CFG)
    assert b.size == CFG.n_precursor + 1
    assert np.allclose(np.diff(b), CFG.soa_ms)
    for c in CONDS:
        assert np.array_equal(S.b_onsets(CFG, c.n_precursor), b)


@pytest.mark.parametrize("pct", [0.0, 25.0, 50.0, 75.0, 100.0])
def test_coherent_a_lags_b_by_exactly_dT(pct):
    from tcoh.config import Condition
    c = Condition("x", pct, "coherent", "yoked")
    assert np.allclose(S.a_onsets(CFG, c) - S.b_onsets(CFG), CFG.lag_ms(pct))
    assert CFG.lag_ms(100.0) == CFG.soa_ms / 2      # 100% is exact alternation


def test_a_kinds_have_the_tone_counts_they_claim():
    from tcoh.config import Condition
    n = CFG.n_precursor + 1
    for kind, want in (("coherent", n), ("scrambled", n), ("pair_only", 1),
                       ("nopartner", n - 1), ("absent", 0)):
        c = Condition("x", 50.0, kind, "yoked")
        sc = (S.scramble_onsets(np.random.default_rng(0), *S.scramble_window(CFG, c), n - 1,
                                CFG.scramble_min_gap) if kind == "scrambled" else None)
        assert S.n_a_tones(CFG, c) == want
        assert S.a_onsets(CFG, c, sc).size == want


def test_pair_only_keeps_the_final_a_tone_exactly_where_coherent_puts_it():
    """The point of the control: the interval the listener judges is identical."""
    from tcoh.config import Condition
    for pct in (0.0, 50.0, 100.0):
        coh = S.a_onsets(CFG, Condition("c", pct, "coherent", "yoked"))
        par = S.a_onsets(CFG, Condition("p", pct, "pair_only", "yoked"))
        assert par.size == 1
        assert par[0] == pytest.approx(coh[-1])


def test_scrambled_keeps_the_final_a_tone_and_never_overlaps_itself():
    from tcoh.config import Condition
    rng = np.random.default_rng(7)
    for pct in (0.0, 50.0, 100.0):
        c = Condition("s", pct, "scrambled", "yoked")
        coh = S.a_onsets(CFG, Condition("c", pct, "coherent", "yoked"))
        for _ in range(50):
            sc = S.scramble_onsets(rng, *S.scramble_window(CFG, c), CFG.n_precursor,
                                   CFG.scramble_min_gap)
            a = S.a_onsets(CFG, c, sc)
            assert a[-1] == pytest.approx(coh[-1])
            assert np.diff(np.sort(a[:-1])).min() >= CFG.scramble_min_gap - 1e-9


def test_scrambled_is_not_secretly_isochronous():
    """A control that re-created a regular sequence would not be a control."""
    from tcoh.config import Condition
    rng = np.random.default_rng(3)
    c = Condition("s", 50.0, "scrambled", "yoked")
    spreads = []
    for _ in range(100):
        sc = S.scramble_onsets(rng, *S.scramble_window(CFG, c), CFG.n_precursor,
                               CFG.scramble_min_gap)
        spreads.append(np.std(np.diff(np.sort(sc))))
    assert np.mean(spreads) > 10.0     # ms; a grid would give exactly 0


def test_the_tempo_reference_lands_the_final_a_tone_on_the_final_b_tone():
    from tcoh.config import Condition
    b = S.b_onsets(CFG)
    for gap in (30.0, 50.0, 70.0):
        a = S.a_onsets(CFG, Condition("t", 0.0, "coherent", "tempo", tempo_gap_ms=gap))
        assert a[-1] == pytest.approx(b[-1])
        assert np.allclose(np.diff(a), CFG.tone_ms + gap)


def test_only_the_last_b_tone_moves_and_it_moves_by_delta():
    for c in CONDS:
        for direction in (+1, -1):
            tr = S.build_trial(CFG, c, 9.0, np.random.default_rng(2), target_position=1,
                               direction=direction)
            std, tgt = tr.second, tr.first
            assert np.array_equal(std.a_onsets_ms, tgt.a_onsets_ms)
            assert np.array_equal(std.b_onsets_ms[:-1], tgt.b_onsets_ms[:-1])
            assert tgt.b_onsets_ms[-1] - std.b_onsets_ms[-1] == pytest.approx(direction * 9.0)


def test_the_b_channel_renders_identically_in_every_condition():
    """The invariant that rules out any single-channel account of a dT effect."""
    ref = None
    for c in CONDS:
        tr = S.build_trial(CFG, c, 9.0, np.random.default_rng(11), target_position=1, direction=+1)
        b = V._render_channel(CFG, tr.second, D, "b", tr.phases)
        if ref is None:
            ref = b
        else:
            assert np.array_equal(b, ref), c.name


def test_the_a_channel_renders_identically_in_the_two_intervals():
    for c in CONDS:
        if c.a_kind == "absent":
            continue
        tr = S.build_trial(CFG, c, 9.0, np.random.default_rng(11), target_position=1)
        a1 = V._render_channel(CFG, tr.first, D, "a", tr.phases)
        a2 = V._render_channel(CFG, tr.second, D, "a", tr.phases)
        assert np.array_equal(a1, a2), c.name


def test_every_interval_is_the_same_length():
    lens = set()
    for c in CONDS:
        tr = S.build_trial(CFG, c, CFG.delta_max_ms, np.random.default_rng(1))
        for iv in (tr.first, tr.second):
            lens.add(S.render_interval(CFG, iv, D, tr.phases).size)
    assert len(lens) == 1


def test_delta_is_clamped_to_the_configured_range():
    for want, req in ((CFG.delta_max_ms, 1e6), (CFG.delta_min_ms, 1e-9)):
        tr = S.build_trial(CFG, CONDS[0], req, np.random.default_rng(1))
        assert tr.delta_ms == pytest.approx(want)


def test_the_realised_shift_is_the_one_that_was_rendered():
    q = 1000.0 / CFG.sample_rate
    for d in (0.25, 1.0, 3.7, 20.0, 45.0):
        tr = S.build_trial(CFG, CONDS[0], d, np.random.default_rng(1), direction=+1)
        assert abs(abs(tr.delta_realised_ms) - d) <= q


def test_monaural_output_is_silent_in_one_ear():
    x = np.ones(10)
    assert S.to_output(CFG.replace(monaural=True), x)[:, 1].max() == 0.0
    assert S.to_output(CFG.replace(monaural=False), x)[:, 1].min() == 1.0


def test_a_tone_outside_the_interval_is_an_error_not_a_silent_truncation():
    from tcoh.config import Condition
    iv = S.Interval(np.array([1e9]), S.b_onsets(CFG), 0.0, 0.0, "coherent", 0.0)
    with pytest.raises(Exception):
        S.render_interval(CFG, iv, D)


# ============================================================ staircase
def test_the_step_schedule_follows_the_published_rule():
    """x4 until the first reversal, x2 for two more, then sqrt(2)."""
    t = Track(CFG, "x")
    assert t.delta == CFG.delta_start_ms
    for _ in range(3):                       # three correct -> one step down at x4
        t.update(True)
    assert t.delta == pytest.approx(CFG.delta_start_ms / 4.0)
    t.update(False)                          # first reversal -> step becomes x2
    assert len(t.reversals) == 1
    assert t.step_index == 1
    assert t.delta == pytest.approx(CFG.delta_start_ms / 4.0 * 2.0)


def test_n_down_sets_the_proportion_the_track_converges_on():
    assert target_proportion(3) == pytest.approx(0.7937, abs=1e-4)
    assert target_proportion(2) == pytest.approx(0.7071, abs=1e-4)
    assert Track(CFG, "x").target_p == pytest.approx(0.7937, abs=1e-4)


def test_a_track_that_never_converges_reports_no_threshold():
    t = Track(CFG.replace(max_trials_per_track=10), "x")
    while not t.finished:
        t.update(True)
    assert t.stop_reason == "max_trials"
    assert t.threshold_ms() is None
    assert t.audit()["converged"] is False


def test_the_threshold_is_the_geometric_mean_of_the_last_reversals():
    rng = np.random.default_rng(0)
    t = Track(CFG, "x")
    while not t.finished:
        t.update(bool(PF(8.0, 0.6).respond(t.delta, rng)))
    final = [d for d, si in t.reversals if si == t.final_step][-CFG.n_final_reversals:]
    assert t.threshold_ms() == pytest.approx(float(np.exp(np.mean(np.log(final)))))


def test_delta_never_leaves_its_bounds_and_the_clamp_is_recorded():
    rng = np.random.default_rng(1)
    t = Track(CFG, "x")
    while not t.finished:
        assert CFG.delta_min_ms - 1e-9 <= t.delta <= CFG.delta_max_ms + 1e-9
        t.update(bool(rng.random() < 0.3))       # a poor listener, driven to the ceiling
    a = t.audit()
    assert a["at_ceiling_trials"] > 0
    assert "ceiling_inside_threshold" in a


def test_the_staircase_recovers_thresholds_it_was_given():
    r = simulate(CFG, thresholds_ms=(3.0, 12.0), sigmas=(0.6,), n_runs=200, seed=1)
    assert r["worst_abs_bias_pct"] < 20.0
    assert r["max_fail_rate"] < 0.05
    for v in r["cells"].values():
        assert v["bias_log"] < 0.0, "the reversal mean is known to under-estimate slightly"


def test_the_staircase_bias_is_similar_across_thresholds_so_it_cancels_in_a_ratio():
    r = simulate(CFG, thresholds_ms=(3.0, 8.0, 16.0), sigmas=(0.6,), n_runs=200, seed=2)
    biases = [v["bias_log"] for v in r["cells"].values()]
    assert max(biases) - min(biases) < 0.10     # log units, i.e. under 11% differential


# ============================================================ psychometric
def test_the_pf_hits_its_nominal_proportion_at_its_threshold():
    pf = PF(6.0, 0.5, 0.02)
    assert float(pf.p_correct(6.0)) == pytest.approx(target_proportion(3), abs=1e-6)
    assert float(pf.p_correct(1e-6)) == pytest.approx(0.5, abs=1e-3)
    assert float(pf.p_correct(1e6)) == pytest.approx(0.98, abs=1e-3)


def test_the_fit_recovers_a_known_threshold():
    rng = np.random.default_rng(0)
    pf = PF(6.0, 0.5, 0.02)
    d = np.exp(rng.uniform(np.log(1), np.log(40), 800))
    f = fit(d, pf.respond(d, rng))
    assert f.threshold_ms == pytest.approx(6.0, rel=0.2)
    assert f.reliable


def test_a_fit_from_data_with_no_spread_is_flagged_unreliable():
    rng = np.random.default_rng(0)
    d = np.full(300, 6.0) * rng.uniform(0.98, 1.02, 300)
    f = fit(d, rng.random(300) < 0.79)
    assert f is None or not f.reliable


def test_the_fit_returns_nothing_rather_than_nonsense_on_too_little_data():
    assert fit([1.0, 2.0], [True, False]) is None
    assert fit([1.0] * 20, [True] * 20) is None


# ============================================================ design
def test_the_design_is_deterministic_given_participant_and_session():
    a = make_design(CFG, "P01", 1)
    b = make_design(CFG, "P01", 1)
    assert a["design_hash"] == b["design_hash"]
    assert make_design(CFG, "P01", 2)["design_hash"] != a["design_hash"]
    assert make_design(CFG, "P02", 1)["design_hash"] != a["design_hash"]
    assert session_seed(CFG, "P01", 1) != session_seed(CFG, "P02", 1)


def test_every_condition_gets_the_same_number_of_tracks_one_per_round():
    a = audit_design(CFG, make_design(CFG, "P01", 1))
    assert a["balanced_track_count"]
    assert set(a["tracks_per_condition"].values()) == {CFG.tracks_per_condition}
    assert a["round_balance_spread"] == pytest.approx(0.0, abs=1e-9)


def test_no_condition_is_systematically_early_or_late():
    for code in ("P01", "P02", "P03"):
        a = audit_design(CFG, make_design(CFG, code, 1))
        assert a["serial_position_spread"] < 0.10


def test_the_run_length_constraint_holds_across_block_boundaries():
    for code in ("P01", "P02", "P03"):
        a = audit_design(CFG, make_design(CFG, code, 1))
        assert a["run_constraint_respected"], a["longest_condition_run"]


def test_no_block_holds_a_single_track_which_could_not_be_interleaved():
    from tcoh.config import CORE_PCTS
    cfg = CFG.replace(pair_only_pcts=CORE_PCTS)      # 16 conditions: the naive split leaves one
    des = make_design(cfg, "P01", 1)
    from collections import Counter
    per_block = Counter(t["block_index"] for t in des["tracks"])
    assert min(per_block.values()) >= 2


def test_the_duration_estimate_uses_simulated_track_lengths():
    e = duration_estimate(CFG)
    assert 25 <= e["median_trials_per_track"] <= 70
    assert e["total_minutes"] > e["main_minutes"]


# ============================================================ runner, end to end
@pytest.fixture(scope="module")
def simulated(tmp_path_factory):
    out = {}
    for mode in ("coherence", "pedestal", "null"):
        d = tmp_path_factory.mktemp(mode)
        r = Runner(CFG, d, audio=False, auto=mode, seed=5)
        out[mode] = (r, r.run(code="SIM"))
    return out


def test_a_full_session_runs_and_every_track_converges(simulated):
    r, sdir = simulated["coherence"]
    assert (sdir / "trials.csv").exists()
    assert (sdir / "session.json").exists()
    assert all(t.stop_reason == "converged" for t in r.tracks.values())
    meta = json.loads((sdir / "session.json").read_text())
    assert meta["status"] == "complete"
    assert meta["auto"] == "coherence"
    assert meta["rt_reference"]


def test_the_trial_log_has_every_column_and_no_blanks_where_it_matters(simulated):
    _, sdir = simulated["coherence"]
    rows = list(csv.DictReader(open(sdir / "trials.csv")))
    from tcoh.session import TRIAL_FIELDS
    assert list(rows[0].keys()) == TRIAL_FIELDS
    for r in rows:
        assert r["condition"] and r["phase"] in ("main", "catch")
        assert r["target_position"] in ("1", "2")
        assert r["correct"] in ("0", "1")
        assert float(r["delta_realised_ms"]) != 0.0


def test_catch_trials_are_logged_but_do_not_move_the_staircase(simulated):
    _, sdir = simulated["coherence"]
    rows = list(csv.DictReader(open(sdir / "trials.csv")))
    catch = [r for r in rows if r["phase"] == "catch"]
    assert catch
    assert all(abs(float(r["delta_ms"]) - CFG.catch_delta_ms) < 1e-9 for r in catch)
    assert all(r["track_trial_index"] == "" for r in catch)


def test_thresholds_can_be_recomputed_from_the_csv_alone(simulated):
    r, sdir = simulated["coherence"]
    rows, _ = load([sdir])
    rep = replay_tracks(rows, CFG)
    assert len(rep) == len(r.tracks)
    for tid, a in rep.items():
        assert a["threshold_ms"] == pytest.approx(r.tracks[tid].threshold_ms(), rel=1e-9)
        assert a["logged_deltas_match"]


def test_resume_refuses_when_the_config_has_changed(simulated, tmp_path):
    _, sdir = simulated["coherence"]
    from tcoh.session import DesignChanged, check_resumable, read_json
    meta = read_json(sdir / "session.json")
    other = CFG.replace(tracks_per_condition=CFG.tracks_per_condition + 1)
    with pytest.raises(DesignChanged):
        check_resumable(meta, other, make_design(other, "SIM", 1))


# ============================================================ analysis
def test_the_recovered_thresholds_track_the_ones_the_listener_was_given(simulated):
    r, sdir = simulated["coherence"]
    rows, _ = load([sdir])
    res = thresholds(rows, CFG)
    truth = r.sim.truth()
    got = {n: v.geomean_ms for n, v in res.items() if v.geomean_ms}
    assert len(got) == len(CONDS)
    lt = np.log([truth[n] for n in got])
    lg = np.log([got[n] for n in got])
    assert np.corrcoef(lt, lg)[0, 1] > 0.9
    assert np.max(np.abs(lt - lg)) < math.log(2.5)


def test_the_index_is_zero_at_synchrony_and_near_one_at_alternation(simulated):
    _, sdir = simulated["coherence"]
    rows, _ = load([sdir])
    idx = coherence_index(thresholds(rows, CFG), CFG, n_boot=600)
    assert idx["ok"]
    assert idx["pcts"][0] == 0.0
    assert abs(idx["kappa"][0]) < 0.25
    assert idx["kappa"][-1] > 0.5
    assert idx["span"] > 0.5


def test_the_index_refuses_itself_when_there_is_no_dynamic_range():
    from tcoh.analysis import CondResult
    res = {c.name: CondResult(c.name, c.lag_pct, c.a_kind, CFG.n_precursor,
                              [10.0, 10.1, 9.9], 3, 10.0, None, 0, 0.79, 0, 0)
           for c in CONDS}
    out = coherence_index(res, CFG, n_boot=400)
    assert not out["ok"]
    assert "dynamic range" in out["reason"]


def test_h1_fires_under_coherence_and_under_the_pedestal_rival_alike(simulated):
    """H1 is not diagnostic, and the test suite should pin that down rather than hope."""
    for mode in ("coherence", "pedestal"):
        _, sdir = simulated[mode]
        rows, _ = load([sdir])
        t = trend_test(thresholds(rows, CFG), n_perm=2000)
        assert t["ok"] and t["p_one_sided"] < 0.05, mode


def test_h2_separates_coherence_from_the_pedestal_rival(simulated):
    """The test the whole design exists for."""
    _, sc = simulated["coherence"]
    _, sp = simulated["pedestal"]
    rc = interaction_test(thresholds(load([sc])[0], CFG), CFG, n_boot=600, n_perm=4000)
    rp = interaction_test(thresholds(load([sp])[0], CFG), CFG, n_boot=600, n_perm=4000)
    assert rc["ok"] and rp["ok"]
    # the decision, not the sign of a null slope: under the pedestal truth the slope is zero, so
    # which side of zero it lands on is a coin flip and asserting it was over-specified.
    assert rc["p_slope_negative"] < 0.05
    assert rp["p_slope_negative"] > 0.05
    assert rc["slope_per_pct"] < 0
    assert rc["slope_per_pct"] < rp["slope_per_pct"]
    assert rc["p_method"].startswith("permutation")


def test_nothing_fires_under_the_null(simulated):
    _, sdir = simulated["null"]
    rows, _ = load([sdir])
    res = thresholds(rows, CFG)
    assert trend_test(res, n_perm=2000)["p_one_sided"] > 0.05
    assert interaction_test(res, CFG, n_boot=600, n_perm=4000)["p_slope_negative"] > 0.05


def test_the_diagnostics_notice_the_things_they_are_for(simulated):
    _, sdir = simulated["coherence"]
    rows, metas = load([sdir])
    dg = diagnostics(rows, metas, CFG, thresholds(rows, CFG))
    assert dg["catch_ok"]
    assert dg["convergence_rate"] == 1.0
    # Some censoring is normal: a staircase whose threshold is 15 ms still wanders, and with
    # delta_max at 45 ms the odd reversal lands on the clamp. What would not be normal is most
    # of them, which is what a real session showed and what prompted this check existing.
    assert dg["tracks_censored"] <= 0.15 * dg["tracks_attempted"]
    assert 0.4 < dg["rove_favours_target"] < 0.6
    assert dg["auto"] == ["coherence"]


def test_the_report_labels_simulated_data_as_simulated(simulated):
    from tcoh.analysis import analyse
    _, sdir = simulated["coherence"]
    text = analyse([sdir], n_boot=400)
    assert "SIMULATED DATA" in text
    assert "coherence" in text


# ============================================================ observers
def test_each_simulated_listener_makes_the_curve_its_name_promises():
    coh = SimulatedListener(CFG, mode="coherence")
    ped = SimulatedListener(CFG, mode="pedestal")
    nul = SimulatedListener(CFG, mode="null")
    by = {c.name: c for c in CONDS}
    # coherence: scrambled sits at the ceiling whatever dT is
    assert coh.threshold_ms(by["scr_0"]) == pytest.approx(coh.threshold_ms(by["scr_100"]))
    assert coh.threshold_ms(by["coh_0"]) < coh.threshold_ms(by["coh_100"])
    # pedestal: coherent and scrambled are identical at every dT
    for p in ("0", "50", "100"):
        assert ped.threshold_ms(by[f"coh_{p}"]) == pytest.approx(ped.threshold_ms(by[f"scr_{p}"]))
    # null: nothing depends on dT at all
    assert nul.threshold_ms(by["coh_0"]) == pytest.approx(nul.threshold_ms(by["coh_100"]))
    # every one of them puts b_only at the ceiling
    for o in (coh, ped, nul):
        assert o.threshold_ms(by["b_only"]) == pytest.approx(o.ceiling_ms)


# ============================================================ verification battery
def test_the_whole_verification_battery_passes_on_the_default_config():
    battery = V.run_battery(CFG, quick=True)
    failed = [c.name for v in battery.values() for c in v if not c.passed]
    assert not failed, failed


def test_the_battery_catches_a_broken_stimulus():
    """A config whose tones share auditory filters must fail, not quietly pass."""
    from tcoh.config import ConfigError
    with pytest.raises(ConfigError):
        V.run_battery(CFG.replace(df_semitones=1.0), quick=True)


# ============================================================ resume
def _run_partial(cfg, data_dir, n_trials, code="P", seed=3):
    """Run a simulated session and stop after n_trials, as a listener quitting mid-sitting."""
    from tcoh.runner import QuitRequested
    r = Runner(cfg, data_dir, audio=False, auto="coherence", seed=seed)
    sdir = r.start(code=code)
    r.calibrate()
    n = {"i": 0}
    orig = r._present

    def stopping(*a, **k):
        if n["i"] >= n_trials:
            raise QuitRequested()
        n["i"] += 1
        return orig(*a, **k)

    r._present = stopping
    r.main_block()
    return r, sdir


def _state(t):
    return (round(t.delta, 9), t.step_index, t.n_correct_run, t.direction,
            len(t.trials), len(t.reversals), t.finished, t.threshold_ms())


def test_resume_puts_every_staircase_back_exactly_where_it_was(tmp_path):
    """A 121-minute session has to be splittable, so resume has to actually resume.

    Before this was fixed the runner read the trial log and then built every Track from its
    starting delta, replaying the whole session and appending a second copy of every trial.
    """
    r1, sdir = _run_partial(CFG, tmp_path, 400)
    before = {tid: _state(t) for tid, t in r1.tracks.items()}
    assert before, "the partial run produced no tracks"

    r2 = Runner(CFG, tmp_path, audio=False, auto="coherence", seed=3)
    r2.start(code="P", resume=True)
    after = {tid: _state(t) for tid, t in r2.tracks.items()}
    assert after == before


def test_resume_does_not_repeat_a_trial_and_still_finishes(tmp_path):
    _run_partial(CFG, tmp_path, 400)
    r2 = Runner(CFG, tmp_path, audio=False, auto="coherence", seed=3)
    sdir = r2.run(code="P", resume=True)
    rows = list(csv.DictReader(open(sdir / "trials.csv")))
    slots = [int(float(x["slot_index"])) for x in rows if x["slot_index"]]
    assert len(slots) == len(set(slots)), "a resumed session re-ran trials it had already done"
    assert all(t.stop_reason == "converged" for t in r2.tracks.values())
    # and the thresholds are still recomputable from the log alone
    rep = replay_tracks(load([sdir])[0], CFG)
    assert all(a["threshold_ms"] is not None for a in rep.values())


def test_the_per_trial_stream_differs_between_participants_and_sessions(tmp_path):
    """Which interval holds the target must not be the same sequence for everyone.

    It was: the stimulus generator was seeded from a constant, so every participant got an
    identical run of target positions. Balanced, and no use to a listener, but any accidental
    structure would have been shared by the whole sample instead of averaging out.
    """
    seqs = {}
    for code, idx in (("AA", 1), ("BB", 1), ("AA", 2)):
        r = Runner(CFG, tmp_path, audio=False, auto="coherence", seed=0)
        r.start(code=code, session_index=idx)
        seqs[(code, idx)] = [int(r.rng.integers(1, 3)) for _ in range(30)]
    assert seqs[("AA", 1)] != seqs[("BB", 1)]
    assert seqs[("AA", 1)] != seqs[("AA", 2)]
    # but a given participant and session is reproducible
    r = Runner(CFG, tmp_path, audio=False, auto="coherence", seed=0)
    r.start(code="AA", session_index=1, resume=True)
    assert [int(r.rng.integers(1, 3)) for _ in range(30)] == seqs[("AA", 1)]


# ============================================================ the descriptive pilot
import json as _json

PILOT = Config.from_dict(_json.load(open("tcoh/configs/tcoh_pilot.json")))


def test_the_pilot_preset_is_five_coherent_conditions_and_nothing_else():
    d = validate(PILOT)
    assert [c.lag_pct for c in d.conditions] == [0.0, 25.0, 50.0, 75.0, 100.0]
    assert {c.a_kind for c in d.conditions} == {"coherent"}
    assert PILOT.tracks_per_condition == 2
    assert d.n_tracks == 10
    assert not PILOT.include_b_only
    assert PILOT.scrambled_pcts == () and PILOT.pair_only_pcts == () and PILOT.buildup_pcts == ()


def test_the_pilot_keeps_the_convergence_requirements_of_the_full_design():
    """Fewer conditions and fewer repeats -- not a cheaper threshold."""
    for f in ("n_down", "step_factors", "reversals_per_step", "n_final_reversals",
              "max_trials_per_track", "practice_criterion", "soa_ms", "tone_ms", "n_precursor"):
        assert getattr(PILOT, f) == getattr(CFG, f), f


def test_each_condition_is_measured_once_before_any_is_measured_twice():
    des = make_design(PILOT, "P01", 1)
    first, second = {}, {}
    for t in des["tracks"]:
        (first if t["round_index"] == 0 else second).setdefault(t["condition"], 0)
        (first if t["round_index"] == 0 else second)[t["condition"]] += 1
    assert set(first) == set(second) == set(des["conditions"])
    assert set(first.values()) == set(second.values()) == {1}
    # and every round-0 track is scheduled before every round-1 track
    by_track = {t["track_id"]: t["round_index"] for t in des["tracks"]}
    last0 = max(s["index"] for s in des["slot_plan"] if by_track[s["track_id"]] == 0)
    first1 = min(s["index"] for s in des["slot_plan"] if by_track[s["track_id"]] == 1)
    assert last0 < first1


def test_the_pilot_fits_in_under_an_hour_as_an_estimate():
    e = duration_estimate(PILOT)
    assert e["total_minutes"] < 60
    assert e["worst_case_minutes"] > e["total_minutes"], "a cap is not an estimate"


# ---- the delta geometry ------------------------------------------------------
def test_a_late_shift_moves_the_tone_towards_its_partner_and_an_early_one_away():
    from tcoh.config import displaced_tone_clearance
    lag = CFG.lag_ms(25.0)
    late = displaced_tone_clearance(CFG, 25.0, lag, +1)
    early = displaced_tone_clearance(CFG, 25.0, lag, -1)
    assert late["own_a_sep_ms"] == pytest.approx(0.0, abs=1e-9), "a late shift of exactly the lag lands ON the A tone"
    assert late["makes_more_synchronous"]
    assert early["own_a_sep_ms"] == pytest.approx(2 * lag)
    assert not early["makes_more_synchronous"]


def test_late_shifts_are_unusable_across_the_sweep_and_early_ones_reach_50ms():
    from tcoh.config import max_safe_delta_ms
    pcts = [0.0, 25.0, 50.0, 75.0, 100.0]
    assert max_safe_delta_ms(CFG, pcts, +1) == 0.0
    assert max_safe_delta_ms(CFG, pcts, -1) == pytest.approx(50.0)


def test_the_pilot_delta_ceiling_is_within_what_the_geometry_supports():
    from tcoh.config import max_safe_delta_ms
    pcts = sorted({c.lag_pct for c in validate(PILOT).conditions})
    assert PILOT.delta_direction == "backward"
    assert PILOT.delta_max_ms <= max_safe_delta_ms(PILOT, pcts, -1)
    assert PILOT.delta_max_ms < PILOT.soa_ms - PILOT.tone_ms      # no collision with the previous B
    assert not any("exceeds the" in n for n in validate(PILOT).notes)


def test_a_too_wide_ceiling_is_flagged_rather_than_accepted():
    notes = validate(CFG.replace(delta_max_ms=70.0, delta_direction="backward")).notes
    assert any("fusion margin" in n for n in notes)


def test_nothing_clips_at_the_pilot_ceiling():
    d = validate(PILOT)
    peak = 0.0
    for c in d.conditions:
        for dl in (PILOT.delta_min_ms, 20.0, PILOT.delta_max_ms):
            tr = S.build_trial(PILOT, c, dl, np.random.default_rng(1))
            iv = S.Interval(tr.first.a_onsets_ms, tr.first.b_onsets_ms, tr.first.delta_ms,
                            PILOT.level_rove_db, c.a_kind, c.lag_pct)
            peak = max(peak, float(np.max(np.abs(S.render_interval(PILOT, iv, d, tr.phases)))))
    assert peak < 0.95


# ---- catch trials ------------------------------------------------------------
def test_catch_trials_are_built_from_one_easy_condition_wherever_they_land(tmp_path):
    """A fixed shift is not an easy trial in a hard condition, so it cannot measure lapses."""
    assert PILOT.catch_at_pct == 0.0
    r = Runner(PILOT, tmp_path, audio=False, auto="coherence", seed=4)
    sdir = r.run(code="P")
    rows = list(csv.DictReader(open(sdir / "trials.csv")))
    catch = [x for x in rows if x["phase"] == "catch"]
    assert catch
    assert {x["condition"] for x in catch} == {"coh_0"}
    assert all(abs(float(x["delta_ms"]) - PILOT.catch_delta_ms) < 1e-9 for x in catch)
    assert all(x["track_trial_index"] == "" for x in catch)     # still no effect on any staircase


def test_a_catch_condition_the_design_does_not_contain_is_refused():
    from tcoh.config import ConfigError
    with pytest.raises(ConfigError, match="never otherwise hears"):
        validate(PILOT.replace(catch_at_pct=37.5))


def test_leaving_the_catch_condition_unset_is_flagged():
    assert any("hard condition is not easy" in n for n in validate(PILOT.replace(catch_at_pct=None)).notes)


# ---- the output --------------------------------------------------------------
def test_the_pilot_refuses_to_compute_kappa():
    """No B-only ceiling in this preset, so there is nothing to normalise against."""
    from tcoh.analysis import CondResult
    res = {c.name: CondResult(c.name, c.lag_pct, c.a_kind, PILOT.n_precursor,
                              [8.0, 9.0], 2, 8.5, None, 40, 0.79, 0, 0)
           for c in validate(PILOT).conditions}
    out = coherence_index(res, PILOT, n_boot=200)
    assert not out["ok"]
    assert "b_only" in out["reason"]


def test_the_pilot_curve_separates_usable_tracks_from_bounded_ones(tmp_path):
    from tcoh.analysis import CondResult
    from tcoh.plots import pilot_curve
    conds = validate(PILOT).conditions
    res = {}
    for i, c in enumerate(conds):
        cens = [PILOT.delta_max_ms] if c.lag_pct == 100.0 else []
        good = [] if cens else [4.0 + i, 5.0 + i]
        res[c.name] = CondResult(c.name, c.lag_pct, c.a_kind, PILOT.n_precursor, good, 2,
                                 (float(np.exp(np.mean(np.log(good)))) if good else None),
                                 None, 40, 0.79, 0, 0, cens, "ceiling" if cens else "")
    fig = pilot_curve(PILOT, res, path=tmp_path / "curve.png")
    ax = fig.axes[0]
    lo, hi = ax.get_xlim()
    assert lo > hi, "100% alternation must be on the LEFT and 0% synchrony on the right"
    labels = " ".join(t.get_text() for t in ax.get_legend().get_texts()).lower()
    assert "ceiling" in labels and "individual track" in labels
    assert "lambda" not in (ax.get_ylabel() + ax.get_title()).lower()
    assert "ms" in ax.get_ylabel()
    assert (tmp_path / "curve.png").exists()


def test_block_size_divides_the_conditions_so_none_is_systematically_late():
    from tcoh.design import choose_block_size
    assert choose_block_size(5) == 5
    assert choose_block_size(16) == 4
    a = audit_design(PILOT, make_design(PILOT, "P01", 1))
    assert a["serial_position_spread"] < 0.05


def test_the_feasibility_preset_admits_it_cannot_separate_condition_from_time():
    feas = Config.from_dict(_json.load(open("tcoh/configs/tcoh_pilot_feasibility.json")))
    assert feas.tracks_per_condition == 1
    checks = {c.name: c for v in V.run_battery(feas, quick=True).values() for c in v}
    c = checks["no condition is confounded with time in the session"]
    assert not c.passed
    assert "only a second track can" in c.detail


def test_the_older_presets_are_untouched():
    for name, want_tracks in (("tcoh_core", 33), ("tcoh_full", 57), ("tcoh_screen", 18)):
        cfg = Config.from_dict(_json.load(open(f"tcoh/configs/{name}.json")))
        assert validate(cfg).n_tracks == want_tracks, name
        assert cfg.catch_at_pct is None, f"{name} must keep its recorded behaviour"
        assert cfg.delta_max_ms == 45.0 and cfg.delta_direction == "random", name
