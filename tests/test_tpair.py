"""Tests for the jittered A-B pair task.

The claims the design rests on:
  * the lag is exact in every pair, and the jitter never touches it;
  * the jitter is independent between the two intervals, so no channel can be compared across them;
  * A carries no information at all, and B on its own carries very little;
  * the analysis tells a listener who uses A from one who does not.
"""
import contextlib
import csv
import io
import json

import numpy as np
import pytest

from tpair import stimulus as S
from tpair.analysis import (load, replay_tracks, strategy_check, thresholds, trend_test,
                            what_a_adds)
from tpair.config import DEFAULT, Condition, ConfigError, conditions, validate
from tpair.design import audit_design, duration_estimate, make_design, session_seed
from tpair.observer import threshold
from tpair.runner import Runner

CFG = DEFAULT
CONDS = conditions(CFG)
PAIR = [c for c in CONDS if c.kind == "pair"]


# ============================================================ geometry
def test_the_sequence_averages_three_hz_and_fills_two_seconds():
    assert CFG.slot_ms == pytest.approx(1000 / 3)
    assert CFG.interval_ms == 2000.0
    assert np.allclose(np.diff(CFG.nominal_onsets()), CFG.slot_ms)
    tr = S.build_trial(CFG, PAIR[0], 10.0, np.random.default_rng(1))
    assert S.render_interval(CFG, tr.first).size == int(2.0 * CFG.sample_rate)


def test_the_tones_are_more_than_an_octave_apart_and_far_from_simple_ratios():
    ra = validate(CFG).ratio
    assert ra["octaves"] > 1.0
    assert ra["mistuning_pct"] > 7.0          # 7.7% from both 3:1 and 7:2


def test_validate_rejects_close_or_harmonic_tones_and_impossible_geometry():
    with pytest.raises(ConfigError, match="octave"):
        validate(CFG.replace(f_b_hz=1300.0))
    with pytest.raises(ConfigError, match="simple ratio|from"):
        validate(CFG.replace(f_b_hz=2100.0))   # exactly 3:1
    with pytest.raises(ConfigError, match="same frequency"):
        validate(CFG.replace(delta_max_b_alone_ms=200.0))
    with pytest.raises(ConfigError, match="gap inside the token"):
        validate(CFG.replace(delta_max_ms=80.0))      # an early B would clear A at lag 0
    with pytest.raises(ConfigError, match="gap inside the token"):
        validate(CFG.replace(lags_ms=(0.0, 90.0)))    # beyond 100%
    with pytest.raises(ConfigError, match="interval"):
        validate(CFG.replace(interval_ms=1700.0))


# ============================================================ stimulus invariants
def test_the_five_levels_are_0_to_100_percent_of_the_tone_with_a_first():
    assert CFG.a_leads and CFG.tone_ms == 75.0
    assert [CFG.lag_pct(c.lag_ms) for c in PAIR] == [0, 25, 50, 75, 100]
    tr = S.build_trial(CFG, PAIR[2], 10.0, np.random.default_rng(1))
    assert np.all(tr.standard.b_onsets_ms - tr.standard.a_onsets_ms == pytest.approx(37.5))


def test_there_is_never_a_gap_inside_a_token():
    rng = np.random.default_rng(11)
    for c in PAIR:
        for _ in range(200):
            tr = S.build_trial(CFG, c, float(rng.uniform(CFG.delta_min_ms, CFG.delta_max_ms)), rng)
            for iv in (tr.first, tr.second):
                a, b = iv.a_onsets_ms, iv.b_onsets_ms
                assert np.all(np.maximum(a, b) <= np.minimum(a, b) + CFG.tone_ms + 1e-9)


def test_old_sessions_without_a_leads_still_read_as_b_first():
    from tpair.config import Config
    old = {k: v for k, v in CFG.to_dict().items() if k not in ("a_leads", "delta_max_b_alone_ms")}
    assert Config.from_dict(old).a_leads is False


@pytest.mark.parametrize("lag", [0.0, 18.75, 37.5, 75.0])
def test_the_lag_is_exact_in_every_pair_but_the_displaced_one(lag):
    c = next(x for x in PAIR if x.lag_ms == lag)
    rng = np.random.default_rng(int(lag) + 3)
    for _ in range(100):
        tr = S.build_trial(CFG, c, float(rng.uniform(1, CFG.delta_max_ms)), rng)
        for iv in (tr.first, tr.second):
            d = S.token_lags(CFG, iv)
            for k in range(CFG.n_pairs):
                want = lag + iv.delta_ms if k == iv.target_pair else lag
                assert d[k] == pytest.approx(want, abs=1e-9)


def test_the_jitter_moves_whole_pairs_and_is_independent_between_intervals():
    rng = np.random.default_rng(5)
    j1, j2 = [], []
    for _ in range(3000):
        tr = S.build_trial(CFG, PAIR[2], 10.0, rng)
        for iv in (tr.first, tr.second):
            assert np.allclose(iv.b_onsets_ms - CFG.nominal_onsets() - PAIR[2].lag_ms,
                               np.where(np.arange(CFG.n_pairs) == (iv.target_pair or -1),
                                        iv.jitter_ms + iv.delta_ms, iv.jitter_ms))
        j1.append(tr.first.jitter_ms); j2.append(tr.second.jitter_ms)
    j1, j2 = np.array(j1), np.array(j2)
    assert np.abs(j1).max() <= CFG.pair_jitter_ms + 1e-9
    for k in range(CFG.n_pairs):
        assert abs(np.corrcoef(j1[:, k], j2[:, k])[0, 1]) < 0.06


def test_a_never_moves_relative_to_its_pair():
    rng = np.random.default_rng(7)
    for c in PAIR:
        for _ in range(50):
            tr = S.build_trial(CFG, c, 30.0, rng)
            t = tr.target
            assert np.allclose(t.a_onsets_ms, CFG.nominal_onsets() + t.jitter_ms)


def test_only_b_of_pair_n_or_n_minus_1_moves_and_only_early():
    rng = np.random.default_rng(9)
    seen = []
    for _ in range(4000):
        tr = S.build_trial(CFG, PAIR[0], 20.0, rng)
        seen.append(tr.target_pair)
        assert tr.delta_signed_ms < 0
        assert tr.standard.target_pair is None
    for p in CFG.target_pairs:
        assert abs(seen.count(p) / len(seen) - 0.5) < 0.03


def test_no_tone_ever_sums_with_one_of_its_own_frequency():
    rng = np.random.default_rng(2)
    worst = np.inf
    for c in CONDS:
        for _ in range(300):
            tr = S.build_trial(CFG, c, CFG.track_cfg(c.kind).delta_max_ms, rng)
            for iv in (tr.first, tr.second):
                for ons in (iv.b_onsets_ms, iv.a_onsets_ms):
                    o = np.sort(ons[np.isfinite(ons)])
                    if o.size > 1:
                        worst = min(worst, float(np.diff(o).min() - CFG.tone_ms))
    assert worst >= 25.0


def test_the_b_alone_control_has_no_a_tones():
    c = next(x for x in CONDS if x.kind == "b_alone")
    tr = S.build_trial(CFG, c, 20.0, np.random.default_rng(1))
    assert np.isnan(tr.first.a_onsets_ms).all() and np.isnan(tr.second.a_onsets_ms).all()
    x = S.render_interval(CFG, tr.first, tr.phases[0])
    assert np.array_equal(x, S.render_interval(CFG, tr.first, tr.phases[0], which="b"))


def test_the_realised_shift_is_what_was_rendered():
    q = 1000.0 / CFG.sample_rate
    solo = next(x for x in CONDS if x.kind == "b_alone")
    for c, d in [(PAIR[0], d) for d in (0.25, 1.0, 7.3, 50.0, 70.0)] + [(solo, 120.0)]:
        tr = S.build_trial(CFG, c, d, np.random.default_rng(1))
        assert abs(abs(tr.delta_realised_ms) - d) <= q


# ============================================================ what a listener can use
def test_a_b_only_listener_is_flat_across_lags_and_far_above_the_pair_listener():
    single = [threshold(CFG, c, "single", n=500) for c in PAIR]
    pair = [threshold(CFG, c, "pair", n=500) for c in PAIR]
    assert max(single) / min(single) < 1.25           # cannot see the lag at all
    assert min(single) > 1.3 * max(pair)              # the jitter defeats it
    assert pair == sorted(pair)                       # the pair listener gets worse with lag


# ============================================================ design
def test_seeds_are_independent_and_the_order_is_balanced():
    seeds = {session_seed(CFG, c, i) for c in "ABCD" for i in (1, 2)}
    assert len(seeds) == 8
    a = audit_design(CFG, make_design(CFG, "X", 1))
    assert a["balanced_track_count"] and a["run_constraint_respected"]
    assert a["serial_position_spread"] < 0.15


def test_duration_is_reported_with_its_parts():
    e = duration_estimate(CFG.replace(tracks_per_condition=1))
    assert e["total_minutes"] == pytest.approx(
        e["main_minutes"] + e["practice_minutes"] + e["break_minutes"] + e["setup_minutes"])


# ============================================================ end to end
@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    out = {}
    for mode in ("pair", "single"):
        d = tmp_path_factory.mktemp(mode)
        with contextlib.redirect_stdout(io.StringIO()):
            out[mode] = Runner(CFG, d, audio=False, auto=mode, seed=21).run(code="SIM")
    return out


def test_a_session_converges_and_replays_from_its_own_log(sessions):
    rows, metas = load([sessions["pair"]])
    assert metas[0]["status"] == "complete"
    rep = replay_tracks(rows, CFG)
    assert len(rep) == len(CONDS) * CFG.tracks_per_condition
    assert all(a["converged"] and a["logged_deltas_match"] for a in rep.values())


def test_every_trial_logs_enough_to_rebuild_its_onsets(sessions):
    rows = list(csv.DictReader(open(sessions["pair"] / "trials.csv")))
    main = [r for r in rows if r["phase"] == "main"]
    assert all(len(r["jitter_1"].split(";")) == CFG.n_pairs for r in main)
    assert {int(r["target_pair"]) for r in main} == set(CFG.target_pairs)


def test_catch_trials_never_update_a_staircase(sessions):
    rows = list(csv.DictReader(open(sessions["pair"] / "trials.csv")))
    catch = [r for r in rows if r["phase"] == "catch"]
    assert catch and all(r["track_trial_index"] == "" for r in catch)
    assert all(float(r["delta_ms"]) == pytest.approx(CFG.catch_delta_ms) for r in catch)


def test_the_analysis_tells_the_two_listeners_apart(sessions):
    r1, _ = load([sessions["pair"]]); res1 = thresholds(r1, CFG)
    r2, _ = load([sessions["single"]]); res2 = thresholds(r2, CFG)
    assert trend_test(res1, n_perm=2000)["p_one_sided"] < 0.05
    assert trend_test(res2, n_perm=2000)["p_one_sided"] > 0.05
    # the pair listener beats B alone at every lag; the single listener does not, on average
    w1, w2 = what_a_adds(res1)["ratio_by_lag"], what_a_adds(res2)["ratio_by_lag"]
    assert min(w1.values()) > 1.2
    gm = lambda w: float(np.exp(np.mean(np.log(list(w.values())))))
    assert gm(w1) > 2.5 and gm(w2) < 1.6
    # and the strategy check flags only the one who used B alone
    assert abs(strategy_check(r1, CFG)["pair conditions"]["z"]) < 3.0
    assert strategy_check(r2, CFG)["pair conditions"]["z"] > 3.0
    assert strategy_check(r1, CFG)["B alone"]["coef_per_sd"] > 0     # the positive control


def test_resume_reproduces_the_stimulus_of_an_uninterrupted_session(tmp_path):
    cfg = CFG.replace(tracks_per_condition=1)
    def run(**kw):
        with contextlib.redirect_stdout(io.StringIO()):
            return Runner(cfg, tmp_path, audio=False, auto="pair", seed=4).run(code="R", **kw)
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
    key = lambda rs: {r["slot_index"]: (r["target_position"], r["target_pair"], r["jitter_1"],
                                        r["jitter_2"]) for r in rs if r["slot_index"]}
    a, b = key(full), key(after)
    shared = set(a) & set(b)
    assert len(shared) > len(keep) // 2 and all(a[s] == b[s] for s in shared)


def test_the_whole_battery_passes():
    from tpair.verify import run_battery
    failed = [c.name for v in run_battery(CFG).values() for c in v if not c.passed]
    assert not failed, failed
