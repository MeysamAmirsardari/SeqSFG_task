"""The context-to-test design: the test tokens are the same whatever came before, both intervals
share the context, and the analysis separates persistence from the decoy the switch creates."""
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from tpair import stimulus as S
from tpair.analysis import load, probe_summary, thresholds
from tpair.config import Config, ConfigError, conditions, validate
from tpair.context import carry_over, context_index, effects, power, time_since_switch
from tpair.runner import Runner

PRESETS = Path(__file__).resolve().parents[1] / "tpair" / "configs"
CFG = Config.from_dict(json.loads((PRESETS / "tpair_context_main.json").read_text()))
CONDS = conditions(CFG)


def test_the_presets_are_the_context_design_and_old_hashes_are_untouched():
    feas = Config.from_dict(json.loads((PRESETS / "tpair_context_feasibility.json").read_text()))
    assert feas == CFG.replace(tracks_per_condition=1)
    assert (CFG.n_pairs, CFG.n_context, CFG.lags_ms, CFG.context_lags_ms) == (7, 4, (15.0,), (0.0, 15.0, 30.0))
    assert [c.name for c in CONDS][:3] == ["ctx_0_lag_15_in", "ctx_15_lag_15_in", "ctx_30_lag_15_in"]
    old = Config.from_dict(json.loads((PRESETS / "tpair_feasibility.json").read_text()))
    assert old.hash() == "2749accdf2c94c57" and old.n_context == 0
    scatter = Config.from_dict(json.loads((PRESETS / "tpair_scatter_feasibility.json").read_text()))
    assert scatter.hash() == "2823d28706a116c9"           # L01 session 5 ran under this


def test_validate_refuses_a_context_without_room_for_the_test():
    with pytest.raises(ConfigError, match="test tokens"):
        validate(CFG.replace(n_context=5))                # 2 test tokens: none before the target
    with pytest.raises(ConfigError, match="both n_context"):
        validate(CFG.replace(context_lags_ms=()))
    with pytest.raises(ConfigError, match="between 0 and tone"):
        validate(CFG.replace(context_lags_ms=(0.0, 15.0, 90.0)))


@pytest.mark.parametrize("cond", CONDS, ids=lambda c: c.name)
def test_context_tokens_carry_the_context_lag_and_test_tokens_the_test_lag(cond):
    rng = np.random.default_rng(1)
    for _ in range(50):
        tr = S.build_trial(CFG, cond, float(rng.uniform(1, CFG.delta_max_ms)), rng)
        for iv in (tr.first, tr.second):
            lag = S.token_lags(CFG, iv)
            want = np.r_[[cond.context_lag_ms] * CFG.n_context,
                         [cond.lag_ms] * (CFG.n_pairs - CFG.n_context)]
            k = iv.target_pair
            keep = np.arange(CFG.n_pairs) != (k if k is not None else -1)
            assert np.allclose(lag[keep], want[keep], atol=1e-9)
            if k is not None:
                assert k >= CFG.n_context + 1             # a test token precedes the target
                assert lag[k] == pytest.approx(cond.lag_ms + iv.delta_ms)


def test_both_intervals_share_the_context_so_the_switch_is_no_cue():
    rng = np.random.default_rng(2)
    for c in CONDS:
        tr = S.build_trial(CFG, c, 20.0, rng)
        assert tr.first.context_lag_ms == tr.second.context_lag_ms == c.context_lag_ms


def test_tokens_keep_their_distance_across_the_switch():
    rng = np.random.default_rng(3)
    least = np.inf
    for c in CONDS:
        for _ in range(200):
            tr = S.build_trial(CFG, c, CFG.delta_max_ms, rng)
            for iv in (tr.first, tr.second):
                least = min(least, float(S.between_token_silence(CFG, iv).min()))
    assert least >= CFG.token_gap_ms - 1e-6


def test_the_coherence_model_carries_almost_nothing_across_the_switch():
    vals = [context_index(CFG, C, 15.0, 600.0) for C in CFG.context_lags_ms]
    assert max(vals) - min(vals) < 0.05
    wide = [context_index(CFG, C, 15.0, 1000.0) for C in CFG.context_lags_ms]
    assert wide[0] < wide[1] < wide[2]                    # only when the window reaches the context


def test_power_grows_with_tracks_and_listeners():
    p = {(r["tracks"], r["listeners"]): r["power"] for r in power(0.36, 0.32)}
    assert p[(1, 1)] < p[(2, 1)] < p[(2, 6)] < p[(2, 12)]
    assert p[(2, 12)] > 0.95


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    out = {}
    for mode in ("persist", "change"):
        d = tmp_path_factory.mktemp(mode)
        with contextlib.redirect_stdout(io.StringIO()):
            out[mode] = Runner(CFG, d, audio=False, auto=mode, seed=21).run(code="SIM")
    return out


def test_the_analysis_reads_persistence_as_P_and_the_decoy_as_D(sessions):
    r1, _ = load([sessions["persist"]]); e1 = effects(thresholds(r1, CFG), CFG)[15.0]
    r2, _ = load([sessions["change"]]); e2 = effects(thresholds(r2, CFG), CFG)[15.0]
    assert e1["ok"] and e2["ok"] and e1["both_directions"]
    assert e1["P"] > 0 and abs(e1["D"]) < 0.4
    assert e2["D"] > 0.5 and abs(e2["P"]) < e2["D"]
    for rows in (r1, r2):                                 # ~60 probes: judge by the binomial
        assert probe_summary(rows, CFG)["all"]["p_vs_chance"] > 0.001


def test_every_main_trial_logs_its_context_and_time_since_the_switch(sessions):
    rows = [r for r in load([sessions["persist"]])[0] if r.get("phase") == "main"]
    assert {float(r["context_lag_ms"]) for r in rows} == set(CFG.context_lags_ms)
    t = [time_since_switch(CFG, r) for r in rows]
    assert all(x is not None and x > 0 for x in t)
    assert carry_over(rows)["ok"]


def test_catch_and_practice_use_the_stationary_test_condition(sessions):
    rows = load([sessions["persist"]])[0]
    catch = [r for r in rows if r.get("phase") == "catch"]
    assert catch and all(float(r["context_lag_ms"]) == 15.0 and float(r["lag_ms"]) == 15.0
                         for r in catch)


def test_the_report_has_the_context_section(sessions):
    from tpair.analysis import analyse
    text = analyse([sessions["persist"]])
    assert "-- does the context carry over?" in text and "P (persistence" in text
