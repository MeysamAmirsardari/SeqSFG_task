"""Tests for the Temporal Overlap Pilot.

The load-bearing claims are (a) the geometry is what the condition table says, (b) a present and
an absent trial of one seed hold the same tones down to the duration of each one, and (c) the
timing schedule is identical in all seven cells so the only thing that moves is the figure.
"""
import json
import math

import numpy as np
import pytest

from seqsfg import overlap as O
from seqsfg.config import Config, validate
from seqsfg.stimulus import FIGURE, make_trial, render_interval

PRESET = "overlap_pilot.json"


@pytest.fixture(scope="module")
def preset():
    cfg, ocfg = O.load_preset(PRESET)
    return cfg, ocfg, validate(cfg)


# ---- geometry ---------------------------------------------------------------
def test_geometry_matches_the_specified_table(preset):
    cfg, ocfg, _ = preset
    want = {(20., 0.): (20., 1.00), (20., 10.): (10., 0.50), (20., 20.): (0., 0.0),
            (40., 0.): (40., 1.00), (40., 20.): (20., 0.50), (40., 30.): (10., 0.25),
            (40., 40.): (0., 0.0)}
    assert set(ocfg.cells) == set(want)
    for (t, s), (adj, frac) in want.items():
        g = O.geometry(cfg, t, s, cfg.n_components)
        assert g["adjacent_overlap_ms"] == pytest.approx(adj)
        assert g["adjacent_overlap_fraction"] == pytest.approx(frac)
        assert g["total_extent_ms"] == pytest.approx(t + 6 * s)
        assert g["common_overlap_ms"] == pytest.approx(max(0.0, t - 6 * s))
        assert g["gap_ms"] == 0.0, "at step = T the tones meet; there is no positive gap"


def test_common_overlap_is_not_adjacent_overlap(preset):
    """The distinction the report must never blur."""
    cfg, ocfg, _ = preset
    g = O.geometry(cfg, 40.0, 20.0, 7)
    assert g["adjacent_overlap_ms"] == 20.0
    assert g["common_overlap_ms"] == 0.0
    assert O.geometry(cfg, 40.0, 0.0, 7)["common_overlap_ms"] == 40.0


def test_envelope_overlap_definition(preset):
    cfg, ocfg, _ = preset
    assert O.envelope_overlap(cfg, 40.0, 0.0)[1] == pytest.approx(1.0)
    assert O.envelope_overlap(cfg, 40.0, 40.0)[1] == pytest.approx(0.0)
    assert O.envelope_overlap(cfg, 20.0, 25.0)[1] == pytest.approx(0.0)
    # the equal-ABSOLUTE-overlap pair stays equal once the ramps are counted
    a = O.envelope_overlap(cfg, 20.0, 10.0)[0]
    b = O.envelope_overlap(cfg, 40.0, 30.0)[0]
    assert a == pytest.approx(b, abs=1e-9)
    # the equal-FRACTION pair does not, because the ramp is 5 ms at both durations
    assert O.envelope_overlap(cfg, 20.0, 10.0)[1] != pytest.approx(
        O.envelope_overlap(cfg, 40.0, 20.0)[1], abs=1e-3)


# ---- configuration ----------------------------------------------------------
def test_preset_loads_validates_and_is_the_pilot(preset):
    cfg, ocfg, d = preset
    O.check(cfg, ocfg)
    assert ocfg.absent_class == "plain"
    assert cfg.n_components == 7 and cfg.n_elements == 8
    assert cfg.tone_dur_ms == 30.0            # the BACKGROUND duration, fixed across cells
    assert cfg.tone_dur_ms not in ocfg.durations, (
        "the background must not share a duration with either component, or one of the two "
        "durations would be privileged by blending into the cloud")
    assert ocfg.trials_per_cell == 10 and 2 * 10 * len(ocfg.cells) == 140


def test_roving_control_is_a_separate_preset_and_is_flagged():
    cfg, ocfg = O.load_preset("overlap_roving_control.json")
    O.check(cfg, ocfg)
    assert ocfg.absent_class == "roving"
    assert any("different question" in n for n in O.notes(cfg, ocfg))
    plain, _ = O.load_preset(PRESET)
    assert plain.hash() == cfg.hash(), "the two presets must differ only in the task section"


def test_shipped_configs_are_untouched():
    assert Config.from_dict(json.load(open("pilot_config.json"))).hash() == "7f8f210f0f5000b3"
    for f in ("asynchrony_config.json", "asynchrony_rising_config.json"):
        Config.from_dict(json.load(open(f))["config"])          # still loads and is unchanged


def test_check_refuses_timing_that_cannot_hold_the_widest_cell(preset):
    cfg, ocfg, _ = preset
    with pytest.raises(ValueError, match="per recurrence"):
        O.check(cfg.replace(iei_min_ms=300.0), ocfg)


def test_check_refuses_a_component_shorter_than_its_ramps(preset):
    cfg, ocfg, _ = preset
    with pytest.raises(ValueError, match="ramps"):
        O.check(cfg, O.OverlapConfig(cells=((8.0, 0.0), (8.0, 4.0), (8.0, 8.0))))


def test_overlapconfig_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown overlap keys"):
        O.OverlapConfig.from_dict({"absent_class": "plain", "nonsense": 1})


# ---- the stimulus -----------------------------------------------------------
@pytest.mark.parametrize("cell", list(O.DEFAULT_CELLS))
def test_present_and_absent_hold_the_same_tones(preset, cell):
    """Per (frequency, duration), not on the total: a duration cue would be a figure cue."""
    cfg, ocfg, d = preset
    t, s = cell
    for seed in (5, 61, 907):
        a = O.build_interval(cfg, ocfg, d, seed, t, s, True)
        b = O.build_interval(cfg, ocfg, d, seed, t, s, False)
        assert O.inventory(a) == O.inventory(b)
        assert a.n_tones == b.n_tones
        assert sorted(np.asarray(a.dur).tolist()) == sorted(np.asarray(b.dur).tolist())


@pytest.mark.parametrize("cell", list(O.DEFAULT_CELLS))
def test_placement_is_valid(preset, cell):
    cfg, ocfg, d = preset
    t, s = cell
    for present in (True, False):
        iv = O.build_interval(cfg, ocfg, d, 21, t, s, present)
        pl = O.placement_ok(cfg, d, iv)
        assert pl["channels_with_overlap"] == 0
        assert not pl["runs_past_end"]


@pytest.mark.parametrize("cell", list(O.DEFAULT_CELLS))
def test_the_built_stimulus_has_the_nominal_duration_and_step(preset, cell):
    cfg, ocfg, d = preset
    t, s = cell
    iv = O.build_interval(cfg, ocfg, d, 33, t, s, True)
    f = iv.kind == FIGURE
    assert np.all(np.asarray(iv.dur)[f] == cfg.ms_to_grid(t))
    assert np.all(np.asarray(iv.dur)[~f] == d.tone_dur_grid)       # background never moves
    for k in np.unique(iv.element[f]):
        on = np.sort(iv.onset[f & (iv.element == k)])
        offs = (on - on.min()) * cfg.grid_ms
        assert np.allclose(offs, np.arange(cfg.n_components) * s)
        assert (on.max() - on.min()) * cfg.grid_ms + t == pytest.approx(t + 6 * s)


def test_one_arbitrary_order_is_reused_within_a_trial(preset):
    cfg, ocfg, d = preset
    iv = O.build_interval(cfg, ocfg, d, 44, 40.0, 20.0, True)
    f = iv.kind == FIGURE
    shapes = set()
    for k in np.unique(iv.element[f]):
        m = f & (iv.element == k)
        r = np.argsort(np.argsort(iv.onset[m]))
        shapes.add(tuple(r[np.argsort(iv.channel[m])].tolist()))
    assert len(shapes) == 1


def test_a_fresh_frequency_set_and_order_across_trials(preset):
    cfg, ocfg, d = preset
    sets = {tuple(O.build_interval(cfg, ocfg, d, s, 40.0, 20.0, True).figure_set.tolist())
            for s in range(8)}
    assert len(sets) > 6


def test_absent_trials_have_no_recurring_arrangement(preset):
    cfg, ocfg, d = preset
    iv = O.build_interval(cfg, ocfg, d, 8, 40.0, 20.0, False)
    assert int((iv.kind == FIGURE).sum()) == 0
    assert iv.element_sets == []


def test_timing_is_identical_in_every_cell(preset):
    """Recurrence count, rate and shared jitter must not move with the condition."""
    cfg, ocfg, d = preset
    ref = None
    for t, s in ocfg.cells:
        iv = O.build_interval(cfg, ocfg, d, 99, t, s, True)
        assert iv.element_onsets.size == cfg.n_elements
        if ref is None:
            ref = iv.element_onsets
        assert np.array_equal(iv.element_onsets, ref), "the schedule is drawn before the cell"


def test_same_seed_reproduces_the_scene(preset):
    cfg, ocfg, d = preset
    a = O.build_interval(cfg, ocfg, d, 123, 40.0, 20.0, True)
    b = O.build_interval(cfg, ocfg, d, 123, 40.0, 20.0, True)
    assert np.array_equal(a.onset, b.onset) and np.array_equal(a.channel, b.channel)
    assert np.array_equal(np.asarray(a.dur), np.asarray(b.dur))
    assert np.allclose(a.phase, b.phase)


@pytest.mark.parametrize("cell", list(O.DEFAULT_CELLS))
def test_audio_is_not_clipped(preset, cell):
    cfg, ocfg, d = preset
    t, s = cell
    for present in (True, False):
        x = O.render(cfg, ocfg, d, O.build_interval(cfg, ocfg, d, 7, t, s, present))
        assert np.abs(x).max() < 0.99


# ---- legacy compatibility ---------------------------------------------------
def test_per_tone_duration_is_opt_in_and_agrees_with_the_legacy_path():
    """A legacy interval carries dur=None and must render exactly as it always did."""
    pilot = Config.from_dict(json.load(open("pilot_config.json")))
    d = validate(pilot)
    tr = make_trial(pilot, 4, 7.0, "rising", d=d)
    iv = tr.recurring
    assert iv.dur is None
    legacy = render_interval(pilot, iv, d)
    explicit = iv.copy()
    explicit.dur = np.full(iv.n_tones, d.tone_dur_grid, dtype=int)
    assert np.allclose(legacy, render_interval(pilot, explicit, d), atol=1e-6)


def test_durations_helper_defaults_to_the_config(preset):
    pilot = Config.from_dict(json.load(open("pilot_config.json")))
    d = validate(pilot)
    iv = make_trial(pilot, 4, 0.0, "rising", d=d).recurring
    assert np.all(iv.durations(d) == d.tone_dur_grid)


# ---- design and analysis ----------------------------------------------------
def test_design_is_balanced_and_randomised(preset):
    cfg, ocfg, d = preset
    dz = O.make_design(cfg, ocfg, "P01", 1)
    main = dz["main"]
    assert len(main) == 140
    assert sum(t["present"] for t in main) == 70
    for t, s in ocfg.cells:
        for present in (True, False):
            n = sum(1 for x in main if x["tone_ms"] == t and x["step_ms"] == s
                    and x["present"] is present)
            assert n == ocfg.trials_per_cell
    run = 1
    for j in range(1, len(main)):
        run = run + 1 if main[j]["present"] == main[j - 1]["present"] else 1
        assert run <= ocfg.max_class_run


def test_practice_covers_both_durations(preset):
    cfg, ocfg, d = preset
    pr = O.make_design(cfg, ocfg, "P01", 1)["practice"]
    assert {t["tone_ms"] for t in pr} == set(ocfg.durations)
    assert all(t["step_ms"] == 0.0 for t in pr)
    assert sum(t["present"] for t in pr) == len(pr) // 2


def test_design_is_deterministic(preset):
    cfg, ocfg, d = preset
    a = O.make_design(cfg, ocfg, "P01", 1)["design_hash"]
    assert a == O.make_design(cfg, ocfg, "P01", 1)["design_hash"]
    assert a != O.make_design(cfg, ocfg, "P02", 1)["design_hash"]


def test_no_response_is_not_a_no(preset):
    rows = [{"response": "y", "target_position": "1"}, {"response": "n", "target_position": "1"},
            {"response": "t", "target_position": "1"}, {"response": "n", "target_position": "2"},
            {"response": "t", "target_position": "2"}]
    h, ns, f, nn, miss = O._counts(rows)
    assert (h, ns, f, nn, miss) == (1, 2, 0, 1, 2)


def test_analyse_reads_a_simulated_session(tmp_path, preset):
    cfg, ocfg, d = preset
    small = O.OverlapConfig(**{**ocfg.to_dict(), "trials_per_cell": 4, "practice_trials": 4})
    sdir = O.OverlapRunner(cfg, small, tmp_path, audio=False, auto=120.0).run(code="SIM",
                                                                             session_index=1)
    text = O.analyse([sdir])
    assert "TEMPORAL OVERLAP PILOT" in text
    assert "the three comparisons chosen before the data" in text
    assert "cannot be entered as three independent predictors" in text
    for t, s in ocfg.cells:
        assert O.cell_name(t, s) in text


def test_duration_estimate_is_sane(preset):
    cfg, ocfg, d = preset
    est = O.duration_estimate(cfg, ocfg)
    assert est["n_main"] == 140 and 15 < est["minutes"] < 45


# ---- main-block feedback ----------------------------------------------------
def test_feedback_is_off_by_default(preset):
    cfg, ocfg, d = preset
    assert ocfg.feedback_main is False and ocfg.feedback_main_first == 0


def test_feedback_on_every_trial_is_recorded_per_trial(tmp_path, preset):
    import csv
    cfg, ocfg, d = preset
    fb = O.OverlapConfig(**{**ocfg.to_dict(), "trials_per_cell": 4, "practice_trials": 4,
                            "feedback_main": True})
    sdir = O.OverlapRunner(cfg, fb, tmp_path, audio=False, auto=150.0).run(code="FB", session_index=1)
    rows = [r for r in csv.DictReader(open(sdir / "trials.csv")) if r["block"] == "main"]
    assert rows and all(r["practice_round"] == "1" for r in rows)
    from seqsfg.session import read_json
    assert read_json(sdir / "session.json")["feedback_main"] is True
    text = O.analyse([sdir])
    assert "FEEDBACK WAS ON" in text and "the criterion does not" in text


def test_feedback_on_the_first_n_only(tmp_path, preset):
    import csv
    cfg, ocfg, d = preset
    fb = O.OverlapConfig(**{**ocfg.to_dict(), "trials_per_cell": 4, "practice_trials": 4,
                            "feedback_main_first": 10})
    sdir = O.OverlapRunner(cfg, fb, tmp_path, audio=False, auto=150.0).run(code="FB2", session_index=1)
    rows = [r for r in csv.DictReader(open(sdir / "trials.csv")) if r["block"] == "main"]
    got = [r["practice_round"] for r in rows]
    assert got[:10] == ["1"] * 10 and set(got[10:]) == {"0"}


def test_a_feedback_free_session_says_so(tmp_path, preset):
    cfg, ocfg, d = preset
    small = O.OverlapConfig(**{**ocfg.to_dict(), "trials_per_cell": 4, "practice_trials": 4})
    sdir = O.OverlapRunner(cfg, small, tmp_path, audio=False, auto=150.0).run(code="NF", session_index=1)
    text = O.analyse([sdir])
    assert "no feedback in the main block" in text and "FEEDBACK WAS ON" not in text


def test_notes_warn_when_feedback_is_on(preset):
    cfg, ocfg, d = preset
    on = O.OverlapConfig(**{**ocfg.to_dict(), "feedback_main": True})
    assert any("feedback is on" in n for n in O.notes(cfg, on))
    assert not any("feedback is on" in n for n in O.notes(cfg, ocfg))


def test_feedback_lengthens_the_session_estimate(preset):
    cfg, ocfg, d = preset
    on = O.OverlapConfig(**{**ocfg.to_dict(), "feedback_main": True})
    assert O.duration_estimate(cfg, on)["minutes"] > O.duration_estimate(cfg, ocfg)["minutes"]
