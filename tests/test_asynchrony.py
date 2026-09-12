"""Tests for the onset-asynchrony yes/no task.

The load-bearing claims are the matching ones: whatever the step, whatever the order, a present
interval and an absent interval must agree exactly on how many tones there are, which channels
carry them and how many each one carries. Everything else here guards a way that could quietly
stop being true.
"""
import json
import math

import numpy as np
import pytest

from seqsfg import asynchrony as A
from seqsfg.config import Config, validate
from seqsfg.stimulus import FIGURE

PRESET = "asynchrony_config.json"


@pytest.fixture(scope="module")
def preset():
    cfg, acfg = A.load_preset(PRESET)
    return cfg, acfg, validate(cfg)


# ---- configuration ---------------------------------------------------------
def test_preset_loads_and_validates(preset):
    cfg, acfg, d = preset
    A.check(cfg, acfg)
    assert cfg.tone_dur_ms == 40.0
    assert max(cfg.steps_ms) == 20.0
    assert cfg.figure_band_channels is None
    assert cfg.figure_anchor_seed is None          # a fresh figure every trial


def test_shipped_configs_are_untouched():
    """The new task must not have moved either of the configurations already in use."""
    assert Config.from_dict(json.load(open("pilot_config.json"))).hash() == "7f8f210f0f5000b3"
    ecfg = Config.from_dict(json.load(open("exposure_pilot.json"))["config"])
    assert ecfg.tone_dur_ms == 45.0 and ecfg.figure_band_channels == 13


def test_check_rejects_a_banded_pool(preset):
    cfg, acfg, _ = preset
    with pytest.raises(ValueError, match="figure_band_channels"):
        A.check(cfg.replace(figure_band_channels=13), acfg)


def test_check_rejects_a_ladder_that_would_collide(preset):
    cfg, acfg, _ = preset
    wide = cfg.replace(steps_ms=tuple(list(cfg.steps_ms) + [40.0]))
    with pytest.raises(ValueError, match="run into each other"):
        A.check(wide, acfg)


def test_check_rejects_bad_orders_and_absent(preset):
    cfg, acfg, _ = preset
    with pytest.raises(ValueError, match="unknown order"):
        A.check(cfg, A.AsyncConfig(orders=("sideways",)))
    with pytest.raises(ValueError, match="absent_class"):
        A.check(cfg, A.AsyncConfig(absent_class="nothing"))


def test_asyncconfig_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown asynchrony keys"):
        A.AsyncConfig.from_dict({"orders": ["fixed"], "nonsense": 1})


def test_asyncconfig_hash_is_stable_and_sensitive():
    a = A.AsyncConfig()
    assert a.hash() == A.AsyncConfig.from_dict(a.to_dict()).hash()
    assert a.hash() != A.AsyncConfig(trials_per_cell=a.trials_per_cell + 2).hash()


# ---- the matching that carries the inference --------------------------------
@pytest.mark.parametrize("absent", A.ABSENT_KINDS)
@pytest.mark.parametrize("order", ["fixed", "redrawn"])
def test_present_and_absent_match_exactly(preset, absent, order):
    cfg, acfg, d = preset
    for step in cfg.steps_ms:
        p = A.build_interval(cfg, d, 11, step, order, True)
        a = A.build_interval(cfg, d, 11, step, order, False, absent)
        cp = np.bincount(p.channel, minlength=d.n_channels)
        ca = np.bincount(a.channel, minlength=d.n_channels)
        assert p.n_tones == a.n_tones
        assert np.array_equal(cp, ca)                    # same count in every channel
        assert np.array_equal(cp > 0, ca > 0)            # same channels active
        assert np.all(cp[cp > 0] == cfg.tones_per_channel)
        assert np.all(ca[ca > 0] == cfg.tones_per_channel)


@pytest.mark.parametrize("absent", A.ABSENT_KINDS)
def test_no_two_tones_overlap_in_one_channel(preset, absent):
    cfg, acfg, d = preset
    D = d.tone_dur_grid
    for present in (True, False):
        iv = A.build_interval(cfg, d, 3, 20.0, "fixed", present, absent)
        for c in np.unique(iv.channel):
            o = np.sort(iv.onset[iv.channel == c])
            assert o.size < 2 or np.min(np.diff(o)) >= D


def test_absent_classes_differ_in_what_is_aligned(preset):
    cfg, acfg, d = preset
    n_fig = cfg.n_components * cfg.n_elements
    present = A.build_interval(cfg, d, 5, 12.0, "fixed", True)
    assert int((present.kind == FIGURE).sum()) == n_fig
    # roving: an aligned group at every element, on a fresh set each time
    rov = A.build_interval(cfg, d, 5, 12.0, "fixed", False, "roving")
    assert int((rov.kind == FIGURE).sum()) == n_fig
    sets = [tuple(s.tolist()) for s in rov.element_sets]
    assert len(set(sets)) == len(sets), "a roving absent interval must never repeat a set"
    # incoherent and plain: nothing is aligned anywhere
    for kind in ("incoherent", "plain"):
        iv = A.build_interval(cfg, d, 5, 12.0, "fixed", False, kind)
        assert int((iv.kind == FIGURE).sum()) == 0


def test_present_figure_uses_one_recurring_set(preset):
    cfg, acfg, d = preset
    iv = A.build_interval(cfg, d, 9, 8.0, "fixed", True)
    sets = [tuple(s.tolist()) for s in iv.element_sets]
    assert len(set(sets)) == 1
    assert sorted(sets[0]) == sorted(iv.figure_set.tolist())


# ---- the step is what it says it is ------------------------------------------
@pytest.mark.parametrize("order", ["fixed", "redrawn", "rising"])
def test_component_onsets_are_exactly_step_apart(preset, order):
    cfg, acfg, d = preset
    for step in cfg.steps_ms:
        iv = A.build_interval(cfg, d, 21, step, order, True)
        f = iv.kind == FIGURE
        for k in np.unique(iv.element[f]):
            o = np.sort(iv.onset[f & (iv.element == k)])
            offs = (o - o.min()) * cfg.grid_ms
            assert np.allclose(offs, np.arange(cfg.n_components) * step), \
                f"{order} at {step} ms: offsets {offs}"


def test_fixed_order_recurs_and_redrawn_does_not(preset):
    cfg, acfg, d = preset

    def shapes(order):
        iv = A.build_interval(cfg, d, 33, 16.0, order, True)
        f = iv.kind == FIGURE
        out = []
        for k in np.unique(iv.element[f]):
            m = f & (iv.element == k)
            rank = np.argsort(np.argsort(iv.onset[m]))          # who starts first, second, ...
            out.append(tuple(rank[np.argsort(iv.channel[m])].tolist()))
        return out

    assert len(set(shapes("fixed"))) == 1, "a fixed order must be the same at every element"
    assert len(set(shapes("redrawn"))) > 1, "a redrawn order must change between elements"


def test_at_step_zero_every_order_is_one_sound(preset):
    cfg, acfg, d = preset
    a = A.build_interval(cfg, d, 4, 0.0, "fixed", True)
    f = a.kind == FIGURE
    for k in np.unique(a.element[f]):
        o = a.onset[f & (a.element == k)]
        assert o.min() == o.max(), "at 0 ms the components must be simultaneous"
    assert A.cells(cfg, acfg)[0] == (0.0, acfg.orders[0])
    assert sum(1 for s, _ in A.cells(cfg, acfg) if s == 0.0) == 1


def test_elements_never_collide_at_the_widest_step(preset):
    cfg, acfg, d = preset
    foot = 2.0 * A.span_ms(cfg, max(cfg.steps_ms))
    for j in range(12):
        iv = A.build_interval(cfg, d, 400 + j, max(cfg.steps_ms), "fixed", True)
        gaps = np.diff(iv.element_onsets) * cfg.grid_ms
        assert gaps.min() >= foot, f"{gaps.min()} ms gap, element needs {foot} ms"


def test_interval_never_runs_past_its_end(preset):
    cfg, acfg, d = preset
    for absent in A.ABSENT_KINDS:
        for present in (True, False):
            iv = A.build_interval(cfg, d, 77, 20.0, "redrawn", present, absent)
            assert iv.onset.max() + d.tone_dur_grid <= cfg.n_grid


# ---- reproducibility ---------------------------------------------------------
def test_same_seed_gives_the_same_interval(preset):
    cfg, acfg, d = preset
    a = A.build_interval(cfg, d, 123, 12.0, "fixed", True)
    b = A.build_interval(cfg, d, 123, 12.0, "fixed", True)
    assert np.array_equal(a.onset, b.onset) and np.array_equal(a.channel, b.channel)
    assert np.allclose(a.phase, b.phase)


def test_different_seeds_give_different_intervals(preset):
    cfg, acfg, d = preset
    a = A.build_interval(cfg, d, 123, 12.0, "fixed", True)
    b = A.build_interval(cfg, d, 124, 12.0, "fixed", True)
    assert not np.array_equal(np.sort(a.figure_set), np.sort(b.figure_set))


def test_two_sides_of_one_seed_share_their_structure(preset):
    """Same seed -> the same trial seen twice; that is what the construction check relies on."""
    cfg, acfg, d = preset
    p = A.build_interval(cfg, d, 55, 8.0, "fixed", True)
    a = A.build_interval(cfg, d, 55, 8.0, "fixed", False, "roving")
    assert np.array_equal(p.element_onsets, a.element_onsets)
    assert np.array_equal(p.figure_set, a.figure_set)


# ---- design ------------------------------------------------------------------
def test_design_is_balanced_and_covers_every_cell(preset):
    cfg, acfg, d = preset
    dz = A.make_design(cfg, acfg, "P01", 1)
    main = dz["main"]
    assert len(main) == 2 * acfg.trials_per_cell * len(A.cells(cfg, acfg))
    assert sum(t["present"] for t in main) == len(main) // 2
    for (s, o) in A.cells(cfg, acfg):
        for present in (True, False):
            n = sum(1 for t in main
                    if t["step_ms"] == s and t["order"] == o and t["present"] is present)
            assert n == acfg.trials_per_cell


def test_design_respects_the_run_limits(preset):
    cfg, acfg, d = preset
    for i in range(5):
        main = A.make_design(cfg, acfg, f"P{i:02d}", 1)["main"]
        for key, lim in ((lambda t: t["present"], acfg.max_class_run),
                         (lambda t: (t["step_ms"], t["order"], t["present"]), acfg.max_cell_run)):
            run = 1
            for j in range(1, len(main)):
                run = run + 1 if key(main[j]) == key(main[j - 1]) else 1
                assert run <= lim


def test_the_answer_sequence_is_not_an_alternation(preset):
    """Capping runs raises P(the next answer differs); it must not drive it to 1."""
    cfg, acfg, d = preset
    alt = []
    for i in range(6):
        c = [t["present"] for t in A.make_design(cfg, acfg, f"Q{i:02d}", 1)["main"]]
        alt.append(np.mean([c[j] != c[j - 1] for j in range(1, len(c))]))
    assert 0.5 <= np.mean(alt) < 0.62


def test_design_is_deterministic_in_code_and_session(preset):
    cfg, acfg, d = preset
    a = A.make_design(cfg, acfg, "P01", 1)
    assert a["design_hash"] == A.make_design(cfg, acfg, "P01", 1)["design_hash"]
    assert a["design_hash"] != A.make_design(cfg, acfg, "P01", 2)["design_hash"]
    assert a["design_hash"] != A.make_design(cfg, acfg, "P02", 1)["design_hash"]


def test_duration_fits_the_session_budget(preset):
    cfg, acfg, d = preset
    assert A.duration_estimate(cfg, acfg)["minutes"] <= cfg.max_session_minutes


# ---- analysis ----------------------------------------------------------------
def test_threshold_interpolates_and_censors():
    steps = [0.0, 4.0, 8.0, 12.0]
    assert A.threshold_step(steps, [3.0, 2.0, 1.0, 0.0], 1.0) == pytest.approx(8.0)
    assert A.threshold_step(steps, [3.0, 2.0, 0.0, 0.0], 1.0) == pytest.approx(6.0)
    assert A.threshold_step(steps, [3.0, 3.0, 3.0, 3.0], 1.0) == math.inf     # beyond the range
    assert A.threshold_step(steps, [0.5, 0.4, 0.3, 0.2], 1.0) == -math.inf    # never above it


def test_threshold_ignores_a_non_monotone_wobble():
    # the first crossing is what is reported, not the last
    assert A.threshold_step([0.0, 4.0, 8.0], [2.0, 0.0, 2.0], 1.0) == pytest.approx(2.0)


def test_power_estimate_shrinks_with_trials(preset):
    cfg, acfg, d = preset
    a = A.power_estimate(cfg, acfg)
    b = A.power_estimate(cfg, A.AsyncConfig(**{**acfg.to_dict(), "trials_per_cell": 40}))
    assert b["se_cell"] < a["se_cell"]
    assert b["se_cell"] == pytest.approx(a["se_cell"] * math.sqrt(acfg.trials_per_cell / 40), rel=1e-6)


def test_analyse_reads_a_simulated_session(tmp_path, preset):
    cfg, acfg, d = preset
    from seqsfg.asynchrony import AsyncRunner
    small = A.AsyncConfig(**{**acfg.to_dict(), "trials_per_cell": 4, "practice_trials": 2})
    sdir = AsyncRunner(cfg, small, tmp_path, audio=False, auto=12.0).run(code="SIM", session_index=1)
    text = A.analyse([sdir])
    assert "ONSET ASYNCHRONY" in text
    assert "the asynchrony limit" in text
    assert "does a RECURRING temporal order help" in text
    for s in cfg.steps_ms:
        assert f"  {s:<11.4g}" in text


# ---- the audit ----------------------------------------------------------------
def test_construction_check_passes(preset):
    cfg, acfg, d = preset
    c = A.construction_check(cfg, acfg, n=8, seed=3)
    assert all(v == 0 for v in c["fails"].values()), c["fails"]
    assert c["aligned_groups"]["present"] == cfg.n_elements
    assert c["aligned_groups"]["absent"] == cfg.n_elements        # roving


def test_construction_check_sees_an_unaligned_absent(preset):
    cfg, acfg, d = preset
    inc = A.AsyncConfig(**{**acfg.to_dict(), "absent_class": "incoherent"})
    c = A.construction_check(cfg, inc, n=6, seed=3)
    assert c["aligned_groups"]["absent"] == 0
    assert c["fails"]["same_channel_counts"] == 0


def test_the_figure_and_the_other_elements_are_equally_scattered(preset):
    """If they were not, which set is the aligned one would be legible from the spectrum."""
    cfg, acfg, d = preset
    s = A.scatter_table(cfg, n=60, seed=5)
    fig, oth = s["figure"], s["other elements"]
    assert abs(fig[:, 0].mean() - oth[:, 0].mean()) < 0.4      # mean channel gap
    assert abs(fig[:, 2].mean() - oth[:, 2].mean()) < 0.3      # octave span
    assert fig[:, 0].mean() > 3.0, "unbanded elements should be spread over the pool"


def test_audit_runs_and_reports(preset):
    cfg, acfg, d = preset
    small = A.AsyncConfig(**{**acfg.to_dict(), "orders": ["fixed"]})
    res = A.run_audit(cfg.replace(steps_ms=(0.0, 12.0, 20.0)), small, n_trials=8, seed=1,
                      n_perm=200, verbose=False)
    text = A.report(res)
    assert "ONSET-ASYNCHRONY AUDIT" in text
    assert "construction, checked on the schedule" in text
    assert len(res["cells"]) == 3
