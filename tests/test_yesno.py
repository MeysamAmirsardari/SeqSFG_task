"""The single-interval task. Yes/no is stricter than forced choice: the listener answers from
one sound, so the marginal distribution of every measurable property has to match."""
import json
import pathlib

import numpy as np
import pytest
from scipy import stats

from seqsfg import yesno
from seqsfg.config import DEFAULT, validate
from seqsfg.stimulus import FIGURE


def _pilot():
    p = pathlib.Path(__file__).resolve().parent.parent / "pilot_config.json"
    if not p.exists():
        pytest.skip("pilot_config.json not present")
    from seqsfg import config
    return config.Config.from_dict(json.loads(p.read_text()))


# ---- the statistics --------------------------------------------------------
def test_ranks_match_scipy():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 5, size=(4, 30)).astype(float)      # deliberate ties
    got = yesno._ranks(x)
    want = np.array([stats.rankdata(row) for row in x])
    assert np.allclose(got, want)


def test_dprime_and_criterion_are_the_textbook_quantities():
    assert yesno.dprime_yesno(0.5, 0.5, 100, 100) == pytest.approx(0.0, abs=1e-9)
    # d' = z(H) - z(F); with the log-linear correction the answer is close, not exact
    assert yesno.dprime_yesno(0.84, 0.16, 500, 500) == pytest.approx(1.98, abs=0.05)
    assert yesno.criterion_yesno(0.5, 0.5, 100, 100) == pytest.approx(0.0, abs=1e-9)
    # a listener who says yes rarely is conservative: c > 0
    assert yesno.criterion_yesno(0.3, 0.05, 100, 100) > 0.5
    assert yesno.criterion_yesno(0.95, 0.7, 100, 100) < -0.5


def test_separation_is_calibrated_on_exchangeable_samples():
    """Two samples from the same distribution must not look separable, or the audit is worthless."""
    rng = np.random.default_rng(3)
    ps = []
    for _ in range(12):
        X = rng.normal(size=(80, 20))
        names = [f"f{i}" for i in range(20)]
        ps.append(yesno.feature_separation(names, X[:40], X[40:], n_perm=400, seed=1)["p_value"])
    assert np.mean(np.array(ps) < 0.05) <= 0.25, f"false positives too common: {ps}"
    assert np.median(ps) > 0.2


def test_separation_detects_a_shift_and_a_spread():
    rng = np.random.default_rng(4)
    names = [f"f{i}" for i in range(20)]
    A, B = rng.normal(size=(60, 20)), rng.normal(size=(60, 20))
    B[:, 3] += 1.0                                          # a mean shift in one feature
    r = yesno.feature_separation(names, A, B, n_perm=400, seed=1)
    assert r["p_value"] < 0.01 and r["names"][int(np.argmax(r["dprime"]))] == "f3"
    A2, B2 = rng.normal(size=(60, 20)), rng.normal(size=(60, 20))
    B2[:, 7] *= 3.0                                         # same mean, different spread
    r2 = yesno.feature_separation(names, A2, B2, n_perm=400, seed=1)
    assert r2["p_value"] < 0.01, "a spread difference is a criterion too and must be caught"
    assert r2["kind"][int(np.argmax(r2["dprime"]))] == "spread"


# ---- the stimulus ----------------------------------------------------------
@pytest.mark.parametrize("absent", yesno.ABSENT_KINDS)
def test_every_trial_carries_exactly_the_same_tones(absent):
    """Level cannot be a cue by construction: every interval of either class has the budget in
    every active channel, so total tone count is a constant, not a random variable."""
    cfg = _pilot()
    d = validate(cfg)
    counts = set()
    for seed in range(1, 9):
        for present in (True, False):
            iv = yesno.build_interval(cfg, d, seed, 0.0, present, absent)
            c = np.bincount(iv.channel, minlength=d.n_channels)
            assert set(np.unique(c[c > 0])) == {cfg.tones_per_channel}
            counts.add((int(c.sum()), int((c > 0).sum())))
    assert len(counts) == 1, f"tone count varies between trials: {counts}"


def test_present_and_roving_both_hold_one_bound_group_per_element():
    """The envelope can only match if BOTH classes contain the same number of aligned chords."""
    cfg = _pilot()
    d = validate(cfg)
    for seed in range(1, 6):
        for present in (True, False):
            iv = yesno.build_interval(cfg, d, seed, 0.0, present, "roving")
            fig = iv.onset[iv.kind == FIGURE]
            assert fig.size == cfg.n_elements * cfg.n_components * cfg.figure_repeats
            assert len(np.unique(fig)) == cfg.n_elements, "one coincident onset per element"


def test_only_the_present_class_repeats_its_channels():
    cfg = _pilot()
    d = validate(cfg)
    for seed in range(1, 6):
        p = yesno.build_interval(cfg, d, seed, 0.0, True, "roving")
        a = yesno.build_interval(cfg, d, seed, 0.0, False, "roving")
        assert len({tuple(s.tolist()) for s in p.element_sets}) == 1, "present must recur"
        assert len({tuple(s.tolist()) for s in a.element_sets}) > cfg.n_elements - 2


def test_scattered_and_plain_have_no_bound_group():
    cfg = _pilot()
    d = validate(cfg)
    for kind in ("scattered", "plain"):
        iv = yesno.build_interval(cfg, d, 3, 0.0, False, kind)
        assert int((iv.kind == FIGURE).sum()) == 0


# ---- the design ------------------------------------------------------------
def test_design_is_balanced_and_does_not_run_on():
    cfg = _pilot()
    dz = yesno.make_yesno_design(cfg, "P01", 1)
    main = [yesno.YesNoSpec(**t) for t in dz["main"]]
    assert len(main) == len(cfg.steps_ms) * 2 * cfg.trials_per_condition
    assert sum(s.present for s in main) == len(main) // 2
    for step in cfg.steps_ms:
        at = [s for s in main if s.step_ms == step]
        assert sum(s.present for s in at) == len(at) // 2
    run, longest = 1, 1
    for a, b in zip(main, main[1:]):
        run = run + 1 if a.present == b.present else 1
        longest = max(longest, run)
    assert longest <= 3, "a long run of one answer invites a response strategy"
    assert len({s.seed for s in main}) == len(main)
