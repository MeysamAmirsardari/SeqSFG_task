"""Tests for the temporal-coherence model and the configuration it constrains.

The model is the hypothesis generator, so the tests that matter most are the ones that check it
against something outside this repository: the two numbers Elhilali et al. state for their
Figure 8, and the closed form the two-channel case has.
"""
import math

import numpy as np
import pytest

from tcoh import model as M
from tcoh.config import Config, ConfigError, DEFAULT, conditions, validate


# ---- the filter bank ---------------------------------------------------------
def test_seed_function_is_causal_and_decays():
    t = np.linspace(-1, 6, 2000)
    g = M.seed_function(t)
    assert np.all(g[t < 0] == 0.0)
    assert np.max(np.abs(g[t > 5])) < 1e-4 * np.max(np.abs(g))


def test_rate_filter_is_bandpass_so_a_steady_channel_says_nothing():
    """A channel that is simply ON must contribute no coherence; only co-MODULATION counts.

    Stated as the filter's DC gain relative to its total absolute weight: a filter that passed
    a constant would make two channels look coherent merely for being active at the same time,
    which is the confusion the whole model is built to avoid.

    An individual filter leaks a few percent -- g(t) is not exactly zero-mean and its Hilbert
    transform less so -- but the phase set is antipodal (h at q + pi is -h at q), so summed over
    the six phases the leakage cancels to machine precision. The half-wave rectified reading
    breaks that cancellation, which is one reason its alternating endpoint comes out at 0.89
    rather than 1.0.
    """
    for w in M.RATES_HZ:
        per_phase = [M.rate_filter(w, q, 1000.0) for q in M.PHASES]
        for h in per_phase:
            assert abs(h.sum()) < 0.06 * np.abs(h).sum()
        n = min(len(h) for h in per_phase)
        total = np.sum([h[:n] for h in per_phase], axis=0)
        assert abs(total.sum()) < 1e-9 * np.sum([np.abs(h[:n]).sum() for h in per_phase])


def test_rate_filter_is_tuned_to_its_own_rate():
    fs = 2000.0
    for w in (4.0, 8.0, 16.0):
        h = M.rate_filter(w, 0.0, fs)
        f = np.fft.rfftfreq(4096, 1 / fs)
        mag = np.abs(np.fft.rfft(h, 4096))
        assert abs(f[np.argmax(mag)] - w) < 0.35 * w


# ---- the two-channel closed form ---------------------------------------------
def test_two_channel_index_matches_the_closed_form():
    """With equal-power channels, l2/l1 = (1-r)/(1+r). Checked on a synthetic matrix."""
    for r in (0.0, 0.2, 0.5, 0.9, 1.0):
        c = np.array([[1.0, r], [r, 1.0]])
        got = M._decompose(c, 0.2)
        assert got.ratio == pytest.approx((1 - r) / (1 + r), abs=1e-9)
        assert got.correlation == pytest.approx(r, abs=1e-9)


def test_identical_channels_give_one_stream_and_silence_gives_nothing():
    fs = 1000.0
    env = M.sequence_envelope([0, 150, 300, 450], 75.0, 10.0, 800.0, fs)
    both = M.coherence_matrix(np.stack([env, env]), fs)
    assert both.ratio < 1e-6
    assert both.n_significant == 1


# ---- against the paper --------------------------------------------------------
def test_only_the_coincidence_reading_reproduces_the_published_figure_8():
    r = M.reproduce_figure8()
    hw = r["readings"]["halfwave"]
    assert abs(hw["alternating"] - M.PUBLISHED["alternating"]) < 0.10
    assert abs(hw["synchronous"] - M.PUBLISHED["synchronous"]) < 0.05
    assert hw["monotone"]
    assert r["best"] == "halfwave"
    # and the literal signed-product reading does NOT, which is why the choice is documented
    assert not r["readings"]["none"]["monotone"]
    assert r["readings"]["none"]["alternating"] < 0.5


def test_dT_is_monotone_only_at_a_half_duty_cycle():
    scan = M.duty_cycle_scan()
    assert scan[0.5]["monotone"]
    assert not scan[0.375]["monotone"]
    assert not scan[0.25]["monotone"]


def test_the_ordering_survives_every_reading_of_the_bank():
    band = M.prediction_band([0.0, 25.0, 50.0, 75.0, 100.0])
    assert band["all_monotone"]
    assert band["all_agree_on_order"]
    assert band["max_spread"] < 0.25     # the heights move; the order does not


def test_the_predicted_curve_is_not_a_straight_line_but_is_close_to_one():
    """The honest statement the report has to make, pinned down as a number."""
    p = [0.0, 25.0, 50.0, 75.0, 100.0]
    c = M.predicted_curve(p)
    y = np.array([c[x].ratio for x in p])
    lin = np.array(p) / 100.0
    assert np.corrcoef(y, lin)[0, 1] > 0.97      # so a shape test cannot separate them
    assert np.max(np.abs(y - lin)) > 0.05        # but they are not identical either


# ---- frequency relationships ----------------------------------------------------
def test_crosstalk_grows_as_the_tones_approach():
    near = M.channel_crosstalk(1000.0, 1100.0)
    far = M.channel_crosstalk(1000.0, 2378.4)
    assert near["erbs"] < far["erbs"]
    assert near["roex_attenuation_db"] < far["roex_attenuation_db"]


def test_harmonic_proximity_ignores_arithmetically_close_but_perceptually_absurd_ratios():
    h = M.harmonic_proximity(1000.0, 2378.4)
    assert h["ratio"] == "5:2"           # not 19:8, which is closer in value and means nothing
    assert h["mistuning_pct"] > 3.0
    assert M.harmonic_proximity(1000.0, 2000.0)["ratio"] == "2:1"
    assert M.harmonic_proximity(1000.0, 2000.0)["mistuning_pct"] == pytest.approx(0.0, abs=1e-9)


def test_lag_percentage_and_milliseconds_are_inverses():
    for p in (0.0, 12.5, 50.0, 100.0):
        assert M.pct_from_lag_ms(M.lag_ms_from_pct(p, 150.0), 150.0) == pytest.approx(p)
    assert M.lag_ms_from_pct(100.0, 150.0) == 75.0     # 100% is exact alternation


# ---- configuration --------------------------------------------------------------
def test_the_default_config_validates_and_derives_what_it_promises():
    d = validate(DEFAULT)
    assert d.f_b_hz == pytest.approx(1000.0 * 2 ** (15 / 12))
    assert d.erbs_apart > 3.0
    assert len(d.conditions) == len(set(c.name for c in d.conditions))
    assert set(d.model_by_condition) == set(c.name for c in d.conditions)
    assert math.isnan(d.model_by_condition["b_only"])   # one channel: not defined


def test_a_duty_cycle_the_model_cannot_order_is_refused():
    with pytest.raises(ConfigError, match="half of soa_ms"):
        validate(DEFAULT.replace(tone_ms=50.0))
    validate(DEFAULT.replace(tone_ms=50.0, allow_nonmonotone_duty=True, delta_max_ms=45.0))


def test_a_shift_that_would_collide_is_refused():
    with pytest.raises(ConfigError, match="collision|not less than"):
        validate(DEFAULT.replace(delta_max_ms=80.0))


def test_sync_reference_runs_out_of_room_before_full_alternation():
    """At duty 0.5 the final A tone meets the one before it at exactly dT = 100%.

    Zero gap is already fatal -- the two tones become one of twice the duration -- so the mode
    is legal strictly below 100%, and `validate` refuses the boundary rather than rounding it
    in the friendly direction.
    """
    from tcoh.config import sync_reference_gap_ms
    assert sync_reference_gap_ms(DEFAULT, 0.0) == pytest.approx(75.0)
    assert sync_reference_gap_ms(DEFAULT, 50.0) == pytest.approx(37.5)
    assert sync_reference_gap_ms(DEFAULT, 100.0) == pytest.approx(0.0)

    import dataclasses
    import tcoh.config as C
    cfg = DEFAULT.replace(sweep_pcts=(100.0,), scrambled_pcts=(), include_b_only=False)
    bad = dataclasses.replace(C.conditions(cfg)[0], reference="sync")
    orig = C.conditions
    C.conditions = lambda _cfg: (bad,)
    try:
        with pytest.raises(ConfigError, match="merge into a single long tone"):
            C.validate(cfg)
        ok = dataclasses.replace(bad, lag_pct=50.0)
        C.conditions = lambda _cfg: (ok,)
        C.validate(cfg)
    finally:
        C.conditions = orig


def test_tones_that_share_auditory_filters_are_refused():
    with pytest.raises(ConfigError, match="ERBs apart"):
        validate(DEFAULT.replace(df_semitones=2.0))


def test_scramble_gap_below_the_tone_duration_is_refused():
    with pytest.raises(ConfigError, match="could overlap and sum"):
        validate(DEFAULT.replace(scramble_min_gap_ms=10.0))


def test_config_round_trips_through_json():
    import json
    cfg = DEFAULT.replace(sweep_pcts=(0.0, 33.0), scrambled_pcts=(0.0,))
    back = Config.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert back == cfg
    assert back.hash() == cfg.hash()


def test_unknown_parameters_are_rejected_rather_than_ignored():
    with pytest.raises(ConfigError, match="unknown parameter"):
        Config.from_dict({**DEFAULT.to_dict(), "definitely_not_a_parameter": 1})


def test_every_shipped_preset_validates():
    import json
    from pathlib import Path
    presets = sorted(Path("tcoh/configs").glob("*.json"))
    assert presets, "no presets found"
    for p in presets:
        cfg = Config.from_dict(json.loads(p.read_text()))
        d = validate(cfg)
        assert d.conditions, p.name
