"""Tests for the jittered A-B pair task.

The claims the design rests on:
  * the lag is exact in every token, and the timing never touches it;
  * the timing is independent between the two intervals, so no channel can be compared across them;
  * either tone's own timing carries next to nothing: tokens are scattered with a flat density and
    the change is split between A and B, so a listener judging one tone against its neighbours
    never reaches threshold;
  * the analysis tells a listener who uses the relation from one who does not;
  * "in" and "out" are separate tracks, and only "out" ever opens a gap inside a token;
  * every old preset and session still reads exactly as it did.
"""
import contextlib
import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest

from tpair import stimulus as S
from tpair.analysis import (channel_evidence, load, probe_summary, replay_tracks, strategy_check,
                            thresholds, trend_test, what_a_adds)
from tpair.config import (DEFAULT, Condition, Config, ConfigError, conditions, probe_conditions,
                          realised_timing, scatter_geometry, validate)
from tpair.design import audit_design, duration_estimate, make_design, session_seed
from tpair.observer import p_correct, threshold
from tpair.runner import Runner

CFG = DEFAULT
CONDS = conditions(CFG)
PAIR = [c for c in CONDS if c.kind == "pair" and c.direction == "in"]
OUT = [c for c in CONDS if c.kind == "pair" and c.direction == "out"]
PROBES = probe_conditions(CFG)
PRESETS = Path(__file__).resolve().parents[1] / "tpair" / "configs"
GEO = scatter_geometry(CFG)


def legacy(name="tpair_feasibility"):
    return Config.from_dict(json.loads((PRESETS / f"{name}.json").read_text()))


# ============================================================ geometry
def test_tokens_come_at_three_hz_or_faster_and_every_interval_is_the_same_length():
    assert realised_timing(CFG)["rate_hz"] >= 3.0
    lens = set()
    rng = np.random.default_rng(1)
    for c in list(CONDS) + list(PROBES):
        for _ in range(30):
            tr = S.build_trial(CFG, c, CFG.ceiling_for(c.kind), rng)
            lens.add(S.render_interval(CFG, tr.first).size)
    assert lens == {int(round(CFG.interval_len_ms * CFG.sample_rate / 1000))}


def test_the_tones_are_60_ms_more_than_an_octave_apart_and_far_from_simple_ratios():
    assert CFG.tone_ms == 60.0 and list(CFG.lags_ms) == [0, 15, 30, 45, 60]
    ra = validate(CFG).ratio
    assert 17.5 < ra["semitones"] < 18.5      # down from 20.3 in the first version
    assert ra["mistuning_pct"] > 5.8          # midway between 8:3 and 3:1


def test_validate_rejects_close_or_harmonic_tones_and_impossible_geometry():
    with pytest.raises(ConfigError, match="octave"):
        validate(CFG.replace(f_b_hz=1300.0))
    with pytest.raises(ConfigError, match="simple ratio|from"):
        validate(CFG.replace(f_b_hz=2100.0))   # exactly 3:1
    with pytest.raises(ConfigError, match="gap inside the token"):
        validate(CFG.replace(delta_max_ms=60.0))      # B earlier would clear A at lag 0
    with pytest.raises(ConfigError, match="gap inside the token"):
        validate(CFG.replace(lags_ms=(0.0, 70.0)))    # beyond 100%
    with pytest.raises(ConfigError, match="nearly regular"):
        validate(CFG.replace(rate_hz=5.0))            # no room left to scatter
    with pytest.raises(ConfigError, match="ceiling|delta_max"):
        validate(CFG.replace(catch_delta_ms=60.0))
    with pytest.raises(ConfigError, match="grid timing needs interval_ms"):
        validate(CFG.replace(timing="grid", probe_rate=0.0))
    with pytest.raises(ConfigError, match="scatter timing only"):
        validate(CFG.replace(timing="grid", interval_ms=2400.0))


def test_the_minimum_gap_is_what_the_worst_case_token_needs():
    # B later at 100%: A moves 27.5 ms early, B ends 60 + 27.5 + 60 ms after the reference
    assert GEO["min_ioi_ms"] == pytest.approx(60 + 60 + 55 / 2 + CFG.token_gap_ms)


# ============================================================ old sessions read as before
def test_old_presets_keep_their_hashes_names_timing_and_shift():
    """Recorded sessions were run under these; their hashes are the ones in session.json."""
    want = {"tpair_feasibility": "2749accdf2c94c57", "tpair_curve": "929e3c5e91e33c30"}
    for name, h in want.items():
        cfg = legacy(name)
        assert cfg.hash() == h
        assert (cfg.timing, cfg.shift, cfg.token_rove_db, cfg.probe_rate) == ("grid", "b", 0.0, 0.0)
        assert [c.name for c in conditions(cfg)] == ["lag_0", "lag_18p75", "lag_37p5", "lag_56p25",
                                                     "lag_75", "b_alone"]


def test_an_old_session_design_reproduces_exactly():
    # L01's second tpair session ran tpair_feasibility as session 2; this is its design hash
    assert make_design(legacy(), "L01", 2)["design_hash"] == "8e5f223cf756047c"


def test_old_sessions_without_a_leads_still_read_as_b_first():
    old = {k: v for k, v in legacy().to_dict().items()
           if k not in ("a_leads", "delta_max_b_alone_ms", "timing", "shift")}
    c = Config.from_dict(old)
    assert c.a_leads is False and c.timing == "grid" and c.shift == "b"


def test_a_grid_trial_is_built_as_before():
    """Same draws in the same order: B moves alone, A never moves, the jitter is the timing."""
    cfg = legacy()
    c = next(x for x in conditions(cfg) if x.lag_ms == 37.5)
    tr = S.build_trial(cfg, c, 20.0, np.random.default_rng(5))
    t = tr.target
    assert np.allclose(t.a_onsets_ms, cfg.nominal_onsets() + t.jitter_ms)
    assert t.b_onsets_ms[t.target_pair] == pytest.approx(
        cfg.nominal_onsets()[t.target_pair] + t.jitter_ms[t.target_pair] + 37.5 - 20.0)


def test_the_new_presets_are_the_default_design():
    for name, t in (("tpair_scatter_feasibility", 1), ("tpair_scatter_curve", 2)):
        cfg = Config.from_dict(json.loads((PRESETS / f"{name}.json").read_text()))
        assert cfg == CFG.replace(tracks_per_condition=t)


def test_conditions_cover_every_lag_in_both_directions_with_a_probe_each():
    assert [c.lag_ms for c in PAIR] == [c.lag_ms for c in OUT] == list(CFG.lags_ms)
    assert not any(c.kind == "b_alone" for c in CONDS)
    assert [(p.name, p.kind, p.direction) for p in PROBES] == [
        ("probe_in", "decoupled", "in"), ("probe_out", "decoupled", "out")]


# ============================================================ stimulus invariants
@pytest.mark.parametrize("direction", ["in", "out"])
@pytest.mark.parametrize("lag", [0.0, 15.0, 30.0, 60.0])
def test_the_lag_is_exact_in_every_token_but_the_target(lag, direction):
    c = next(x for x in PAIR + OUT if x.lag_ms == lag and x.direction == direction)
    rng = np.random.default_rng(int(lag) + 3)
    for _ in range(100):
        tr = S.build_trial(CFG, c, float(rng.uniform(1, CFG.delta_max_ms)), rng)
        for iv in (tr.first, tr.second):
            d = S.token_lags(CFG, iv)
            for k in range(CFG.n_pairs):
                want = lag + iv.delta_ms if k == iv.target_pair else lag
                assert d[k] == pytest.approx(want, abs=1e-9)


@pytest.mark.parametrize("conds,sign", [(PAIR, -1), (OUT, +1)], ids=["in", "out"])
def test_the_change_is_split_between_a_and_b_and_only_the_target_moves(conds, sign):
    rng = np.random.default_rng(9)
    seen = []
    for i in range(3000):
        tr = S.build_trial(CFG, conds[i % len(conds)], 20.0, rng)
        t, s = tr.target, tr.standard
        seen.append(tr.target_pair)
        assert (t.a_shift_ms, t.b_shift_ms) == (-sign * 10.0, sign * 10.0)
        assert s.target_pair is None and s.a_shift_ms == s.b_shift_ms == 0.0
        assert np.allclose(np.delete(t.a_onsets_ms, t.target_pair), np.delete(t.ref_ms, t.target_pair))
        assert t.a_onsets_ms[t.target_pair] == pytest.approx(t.ref_ms[t.target_pair] - sign * 10.0)
    for p in CFG.target_pairs:
        assert abs(seen.count(p) / len(seen) - 0.5) < 0.03


def test_an_in_token_never_has_a_gap_inside_it():
    rng = np.random.default_rng(11)
    for c in PAIR:
        for _ in range(200):
            tr = S.build_trial(CFG, c, float(rng.uniform(CFG.delta_min_ms, CFG.delta_max_ms)), rng)
            for iv in (tr.first, tr.second):
                assert np.all(S.token_gaps(CFG, iv) <= 1e-9)
            assert not tr.opens_gap


@pytest.mark.parametrize("lag", [0.0, 15.0, 30.0, 45.0, 60.0])
def test_an_out_token_opens_a_gap_exactly_when_lag_plus_delta_exceeds_the_tone(lag):
    c = next(x for x in OUT if x.lag_ms == lag)
    rng = np.random.default_rng(int(lag) + 1)
    for d in (1.0, 10.0, 15.5, 30.5, 45.5, 55.0):
        tr = S.build_trial(CFG, c, d, rng)
        g = S.token_gaps(CFG, tr.target)
        assert g[tr.target_pair] == pytest.approx(lag + d - CFG.tone_ms)
        assert tr.opens_gap == (lag + d > CFG.tone_ms)
        assert not tr.passes_a_onset
        assert np.all(np.delete(g, tr.target_pair) <= 1e-9)


def test_tokens_never_come_closer_than_the_configured_silence_and_never_too_far_apart():
    rng = np.random.default_rng(2)
    least, worst_same, longest = np.inf, np.inf, 0.0
    for c in list(CONDS) + list(PROBES):
        for _ in range(200):
            tr = S.build_trial(CFG, c, CFG.ceiling_for(c.kind), rng)
            for iv in (tr.first, tr.second):
                longest = max(longest, float(np.diff(iv.ref_ms).max()))
                if c.kind == "pair":
                    least = min(least, float(S.between_token_silence(CFG, iv).min()))
                for ons in (iv.b_onsets_ms, iv.a_onsets_ms):
                    o = np.sort(ons[np.isfinite(ons)])
                    worst_same = min(worst_same, float(np.diff(o).min() - CFG.tone_ms))
    assert least >= CFG.token_gap_ms - 1e-6
    assert worst_same >= 25.0
    assert longest <= CFG.max_ioi_ms + 1e-6


def test_the_timing_is_flat_so_no_position_and_no_gap_is_special():
    rng = np.random.default_rng(3)
    ioi = np.array([np.diff(S.reference_onsets(CFG, rng)) for _ in range(20000)])
    assert ioi.min() >= GEO["min_ioi_ms"] - 1e-9
    m = ioi.mean(0)
    assert np.ptp(m) / m.mean() < 0.03                 # every position alike
    hist, _ = np.histogram(ioi.ravel(), bins=12)
    assert np.argmax(hist) == 0                        # the commonest gap is the shortest
    assert ioi.std() > 100.0


def test_the_timing_is_independent_between_intervals():
    rng = np.random.default_rng(5)
    r1, r2 = [], []
    for _ in range(3000):
        tr = S.build_trial(CFG, PAIR[2], 10.0, rng)
        r1.append(tr.first.ref_ms); r2.append(tr.second.ref_ms)
    r1, r2 = np.array(r1), np.array(r2)
    for k in range(CFG.n_pairs):
        assert abs(np.corrcoef(r1[:, k], r2[:, k])[0, 1]) < 0.06


def test_a_probe_keeps_each_tones_timing_and_drops_the_relation():
    rng = np.random.default_rng(6)
    for p in PROBES:
        tr = S.build_trial(CFG, p, CFG.delta_max_ms, rng)
        assert tr.kind == "decoupled" and tr.delta_ms == CFG.delta_max_ms
        t = tr.target
        assert t.ref_b_ms is not None and not np.allclose(t.ref_b_ms, t.ref_ms)
        k = t.target_pair
        assert t.a_onsets_ms[k] == pytest.approx(t.ref_ms[k] + t.a_shift_ms)
        assert t.b_onsets_ms[k] == pytest.approx(t.ref_b_ms[k] + t.b_shift_ms)
        assert abs(t.a_shift_ms) == abs(t.b_shift_ms) == CFG.delta_max_ms / 2


def test_both_tones_of_a_token_share_its_level_and_the_waveform_carries_it():
    tr = S.build_trial(CFG, Condition("x", 60.0, "pair", "main", "in"), 20.0,
                       np.random.default_rng(4))
    iv = tr.first
    fs, n_env = CFG.sample_rate, int(round(CFG.tone_ms * CFG.sample_rate / 1000))
    for which, ons in (("a", iv.a_onsets_ms), ("b", iv.b_onsets_ms)):
        x = S.render_interval(CFG, iv, tr.phases[0], which=which)
        for k, on in enumerate(ons):
            i = int(round(on * fs / 1000))
            seg = x[i + n_env // 4: i + 3 * n_env // 4]
            db = 20 * np.log10(np.sqrt(2 * np.mean(seg ** 2)) / CFG.tone_amplitude)
            assert db == pytest.approx(iv.level_db + iv.token_db[k], abs=0.05)


def test_the_realised_change_is_what_was_rendered():
    q = 1000.0 / CFG.sample_rate
    for c, d in [(PAIR[0], d) for d in (0.25, 1.0, 7.3, 33.0, 55.0)] + [(PROBES[0], 55.0)]:
        tr = S.build_trial(CFG, c, d, np.random.default_rng(1))
        assert abs(abs(tr.delta_realised_ms) - d) <= 2 * q


# ============================================================ what a listener can use
@pytest.mark.parametrize("conds", [PAIR, OUT], ids=["in", "out"])
def test_judging_the_tones_own_timing_never_reaches_threshold(conds):
    for c in (conds[0], conds[-1]):
        for mode in ("single", "tones"):
            assert p_correct(CFG, c, CFG.delta_max_ms, mode, 3000, 2) < 0.58
        assert p_correct(CFG, c, CFG.delta_max_ms, "boundary", 3000, 2) < 0.68


@pytest.mark.parametrize("conds", [PAIR, OUT], ids=["in", "out"])
def test_the_relational_listener_converges_everywhere_and_gets_worse_with_the_lag(conds):
    pair = [threshold(CFG, c, "pair", n=500) for c in conds]
    assert max(pair) < 0.6 * CFG.delta_max_ms
    assert pair[0] < pair[2] < pair[-1]


def test_every_listener_is_near_chance_on_the_probes():
    for p in PROBES:
        for mode in ("single", "tones", "pair"):
            assert abs(p_correct(CFG, p, CFG.delta_max_ms, mode, 3000, 4) - 0.5) < 0.08


# ============================================================ design
def test_seeds_are_independent_and_the_order_is_balanced():
    seeds = {session_seed(CFG, c, i) for c in "ABCD" for i in (1, 2)}
    assert len(seeds) == 8
    a = audit_design(CFG, make_design(CFG, "X", 1))
    assert a["balanced_track_count"] and a["run_constraint_respected"]
    assert a["serial_position_spread"] < 0.15
    assert 0.08 < a["probe_rate_planned"] < 0.12


def test_directions_run_in_abba_blocks_that_alternate_across_sessions():
    orders = []
    for i in (1, 2):
        d = make_design(CFG, "X", i)
        orders.append(d["direction_order"])
        by = {t["track_id"]: t for t in d["tracks"]}
        for s_ in d["slot_plan"]:                 # every block holds one direction only
            assert by[s_["track_id"]]["direction"] == d["direction_order"][s_["block_index"]]
            assert not (s_["is_catch"] and s_["is_probe"])
    assert orders[0] in (["in", "out", "out", "in"], ["out", "in", "in", "out"])
    assert orders[1][0] != orders[0][0]               # the opening direction flips


def test_duration_is_reported_with_its_parts():
    e = duration_estimate(CFG.replace(tracks_per_condition=1))
    assert e["total_minutes"] == pytest.approx(
        e["main_minutes"] + e["practice_minutes"] + e["break_minutes"] + e["setup_minutes"])
    assert e["total_minutes"] < 65


# ============================================================ end to end
@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    out = {}
    for mode in ("pair", "tones"):
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


def test_every_trial_logs_enough_to_rebuild_its_sound(sessions):
    rows = list(csv.DictReader(open(sessions["pair"] / "trials.csv")))
    main = [r for r in rows if r["phase"] in ("main", "catch", "probe")]
    for r in main:
        tp = int(r["target_position"])
        a = np.array([float(v) for v in r[f"a_onsets_{tp}"].split(";")])
        b = np.array([float(v) for v in r[f"b_onsets_{tp}"].split(";")])
        assert len(a) == len(b) == len(r["token_db_1"].split(";")) == CFG.n_pairs
        if r["kind"] == "pair":
            lag = b - a
            k = int(r["target_pair"])
            assert np.allclose(np.delete(lag, k), float(r["lag_ms"]), atol=2e-3)
            assert lag[k] - float(r["lag_ms"]) == pytest.approx(float(r["delta_signed_ms"]), abs=2e-3)


def test_catch_trials_and_probes_never_update_a_staircase(sessions):
    rows = list(csv.DictReader(open(sessions["pair"] / "trials.csv")))
    catch = [r for r in rows if r["phase"] == "catch"]
    probes = [r for r in rows if r["phase"] == "probe"]
    assert catch and all(r["track_trial_index"] == "" for r in catch)
    assert all(float(r["delta_ms"]) == pytest.approx(CFG.catch_delta_ms) for r in catch)
    assert probes and all(r["track_trial_index"] == "" and r["kind"] == "decoupled" for r in probes)
    assert all(float(r["delta_ms"]) == pytest.approx(CFG.delta_max_ms) for r in probes)
    assert {r["direction"] for r in probes} == {"in", "out"}


def test_every_trial_logs_its_direction(sessions):
    rows = list(csv.DictReader(open(sessions["pair"] / "trials.csv")))
    main = [r for r in rows if r["phase"] in ("main", "catch")]
    for r in main:
        assert np.sign(float(r["delta_signed_ms"])) == (1 if r["direction"] == "out" else -1)
        assert r["condition"].endswith(r["direction"])
    assert any(r["opens_gap"] == "1" for r in main) and all(
        r["opens_gap"] == "0" for r in main if r["direction"] == "in")


def test_the_analysis_tells_the_two_listeners_apart(sessions):
    r1, _ = load([sessions["pair"]]); res1 = thresholds(r1, CFG)
    r2, _ = load([sessions["tones"]]); res2 = thresholds(r2, CFG)
    for d in ("in", "out"):
        assert trend_test(res1, n_perm=2000, direction=d)["p_one_sided"] < 0.05
    # the relational listener converges everywhere below the ceiling; the tone listener has no
    # threshold anywhere: every condition is at the ceiling, reported as a bound
    assert all(len(r.track_thresholds) == 2 and r.geomean_ms for r in res1.values())
    assert all(r.geomean_ms is None and r.censor_side == "ceiling" for r in res2.values())
    # the strategy check flags only the listener who judged the tones' own timing
    s1, s2 = strategy_check(r1, CFG), strategy_check(r2, CFG)
    for d in ("in", "out"):
        assert all(abs(w["z"]) < 3.0 for w in s1[f"pair conditions, {d}"]["channels"].values())
        assert s2[f"pair conditions, {d}"]["z"] > 3.0
    # and the probes are near chance for both: the tones carry nothing either way
    for rows in (r1, r2):
        p = probe_summary(rows, CFG)["all"]
        assert p["n"] >= 30 and p["p_below"] < 0.01 and abs(p["p"] - 0.5) < 0.15


def test_the_tone_evidence_is_rebuilt_for_both_channels(sessions):
    rows = [r for r in load([sessions["pair"]])[0] if r.get("phase") == "main"]
    ev = [channel_evidence(CFG, r) for r in rows[:50]]
    assert all(e is not None and set(e) == {"A", "B"} for e in ev)


def test_the_report_has_probe_and_in_against_out_sections(sessions):
    from tpair.analysis import analyse
    text = analyse([sessions["pair"]])
    assert "decoupled probes" in text and "-- in against out" in text
    assert "came through the relation" in text


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
    # everything drawn from the slot's own generator; the target's onsets also carry delta,
    # which follows the (simulated) answers and so may differ after a resume
    def key(rs):
        out = {}
        for r in rs:
            if r["slot_index"]:
                std = 3 - int(r["target_position"])
                out[r["slot_index"]] = (r["phase"], r["target_position"], r["target_pair"],
                                        r[f"a_onsets_{std}"], r[f"b_onsets_{std}"], r["token_db_1"])
        return out
    a, b = key(full), key(after)
    shared = set(a) & set(b)
    assert len(shared) > len(keep) // 2 and all(a[s] == b[s] for s in shared)


def test_a_session_under_an_old_preset_still_runs_and_reads(tmp_path):
    cfg = legacy()
    with contextlib.redirect_stdout(io.StringIO()):
        sdir = Runner(cfg, tmp_path, audio=False, auto="pair", seed=2).run(code="OLD")
    rows, _ = load([sdir])
    res = thresholds(rows, cfg)
    assert set(res) == {c.name for c in conditions(cfg)}
    assert what_a_adds(res)["ok"]
    # a log from before onsets were recorded: the strategy check falls back on the jitter
    old_cols = [k for k in rows[0] if k not in ("a_onsets_1", "b_onsets_1", "a_onsets_2",
                                                 "b_onsets_2", "token_db_1", "token_db_2",
                                                 "shift_a_ms", "shift_b_ms")]
    stripped = [{k: r[k] for k in old_cols} for r in rows]
    sc = strategy_check(stripped, cfg)
    assert sc["pair conditions"]["ok"] and set(sc["pair conditions"]["channels"]) == {"B"}


@pytest.mark.parametrize("which", ["new", "old"])
def test_the_whole_battery_passes(which):
    from tpair.verify import run_battery
    cfg = CFG if which == "new" else legacy()
    failed = [c.name for v in run_battery(cfg).values() for c in v if not c.passed]
    assert not failed, failed
