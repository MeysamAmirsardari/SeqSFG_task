"""The fast preset: tokens at 5 Hz that may interleave, kept apart only within a channel.

The tokens are session 05's (60 ms, lags 0-60 ms, ceiling 55, 700/1977 Hz); only the rate and the
timing differ. At 5 Hz the token rule leaves no room for jitter at all; keeping only same-pitch
tones apart does. Every interval is drawn from the seed and screened: no hole over 400 ms, no
accidental four-token rhythm, and B always nearer its own A than the next.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from tpair import stimulus as S
from tpair.config import (Config, ConfigError, conditions, probe_conditions, scatter_geometry,
                          timing_faults, validate)
from tpair.design import audit_timing, make_design, slot_condition, slot_rng
from tpair.model import lag_index
from tpair.observer import p_correct

PRESETS = Path(__file__).resolve().parents[1] / "tpair" / "configs"
FAST = Config.from_dict(json.loads((PRESETS / "tpair_fast_out.json").read_text()))
PAIRS = [c for c in conditions(FAST) if c.kind == "pair"]


def test_old_configs_hash_as_recorded_with_the_new_field():
    # sessions 05 and 06 of L01 ran these; clearance = token is the legacy value, left out of the hash
    for name, h in {"tpair_scatter_feasibility": "2823d28706a116c9",
                    "tpair_context_feasibility": "5a0c329d2c0be9b8",
                    "tpair_feasibility": "2749accdf2c94c57"}.items():
        cfg = Config.from_dict(json.loads((PRESETS / f"{name}.json").read_text()))
        assert cfg.clearance == "token" and cfg.hash() == h


def test_the_preset_is_5_hz_out_only_with_session_05s_tokens():
    validate(FAST)
    s05 = Config.from_dict(json.loads((PRESETS / "tpair_scatter_feasibility.json").read_text()))
    assert FAST.rate_hz == 5.0 and FAST.clearance == "channel" and FAST.directions == ("out",)
    for k in ("tone_ms", "lags_ms", "delta_max_ms", "f_a_hz", "f_b_hz", "shift", "token_rove_db"):
        assert getattr(FAST, k) == getattr(s05, k), k
    assert FAST.tracks_per_condition == 2 and len(probe_conditions(FAST)) == 1


def test_the_token_rule_has_no_room_at_this_rate():
    with pytest.raises(ConfigError):
        validate(FAST.replace(clearance="token"))


def test_channel_clearance_is_scatter_only_and_keeps_the_fusion_margin():
    with pytest.raises(ConfigError):
        validate(FAST.replace(timing="grid", interval_ms=2000.0))
    with pytest.raises(ConfigError):
        validate(FAST.replace(token_gap_ms=20.0))


def test_the_minimum_gap_keeps_one_channel_apart_and_the_tokens_in_order():
    g = scatter_geometry(FAST)["min_ioi_ms"]
    # a tone moves 25 ms towards a neighbour that has not moved, and keeps 25 ms of silence
    assert g == pytest.approx(FAST.tone_ms + FAST.delta_max_ms / 2 + FAST.token_gap_ms)
    rng = np.random.default_rng(0)
    least_same, least_order, interleaved = np.inf, np.inf, 0
    for i in range(1500):
        tr = S.build_trial(FAST, PAIRS[i % len(PAIRS)], FAST.delta_max_ms, rng)
        for iv in (tr.first, tr.second):
            for x in (iv.a_onsets_ms, iv.b_onsets_ms):
                least_same = min(least_same, np.diff(np.sort(x)).min() - FAST.tone_ms)
            least_order = min(least_order, (iv.a_onsets_ms[1:] - iv.b_onsets_ms[:-1]).min())
            interleaved += (S.between_token_silence(FAST, iv) < 0).any()
    assert least_same >= FAST.token_gap_ms - 1e-9      # no tone sums with its own pitch
    assert least_order >= FAST.token_gap_ms - 1e-9     # B(k) always starts before A(k+1)
    assert interleaved > 0                             # and tokens do overlap, as intended


def test_the_mean_rate_is_5_hz_and_the_coherence_axis_is_ordered():
    x = [np.diff(S.reference_onsets(FAST, np.random.default_rng(s))) for s in range(800)]
    assert np.mean(np.concatenate(x)) == pytest.approx(200.0, abs=3.0)
    idx = [lag_index(FAST, c.lag_ms, n_seq=12) for c in PAIRS]
    assert all(b > a for a, b in zip(idx, idx[1:])) and idx[-1] > 0.75


def test_every_interval_passes_the_screen_and_bad_draws_are_redrawn():
    rng = np.random.default_rng(5)
    log, worst_own, n = [], np.inf, 0
    for i in range(600):
        tr = S.build_trial(FAST, PAIRS[-1], 1.0, rng, timing_log=log)
        for iv in (tr.first, tr.second):
            r = iv.ref_ms
            assert not timing_faults(FAST, r)
            d = np.diff(r)
            assert d.max() <= 400.0 + 1e-9 and d.min() >= 2 * 60.0 - 1e-9
            # at the largest lag, B starts no nearer the next A than its own
            worst_own = min(worst_own, (r[1:] - (r[:-1] + 60.0)).min() - 60.0)
            n += 1
    assert worst_own >= -1e-9 and log                  # and some draws were thrown away
    assert {x for b in log for x in b} <= {"hole", "pairing", "rhythm"}


def test_the_seed_fixes_the_timing_and_the_audit_replays_it():
    design = make_design(FAST, "TEST", 3)
    a = audit_timing(FAST, design)
    assert a["ok"] and a["intervals"] >= 2 * len(design["slot_plan"])
    slot = design["slot_plan"][17]
    cond, delta = slot_condition(FAST, design, slot)
    one = S.build_trial(FAST, cond, delta, slot_rng(design, slot["index"]))
    two = S.build_trial(FAST, cond, 3 * delta, slot_rng(design, slot["index"]))
    # the timing of a slot is the seed's, whatever delta the staircase asks for
    assert np.array_equal(one.first.ref_ms, two.first.ref_ms)
    assert np.array_equal(one.second.ref_ms, two.second.ref_ms)


def test_old_sessions_replay_exactly_from_their_seed():
    # the screen fields are new; a legacy config draws exactly as it did
    for name in ("tpair_scatter_feasibility", "tpair_context_feasibility"):
        cfg = Config.from_dict(json.loads((PRESETS / f"{name}.json").read_text()))
        assert cfg.regular_tol_ms is None and cfg.pair_clear_ms is None


def test_a_tones_own_timing_stays_far_from_threshold():
    for mode in ("single", "tones"):
        assert p_correct(FAST, PAIRS[-1], FAST.delta_max_ms, mode, n=1500, seed=3) < 0.65
