"""The anchored figure: the same channels in every trial, without breaking the matching."""
import numpy as np
import pytest

from seqsfg import config, stimulus
from seqsfg.config import DEFAULT, validate
from seqsfg.stimulus import FIGURE, make_trial


def test_anchor_fixes_the_figure_across_trials():
    cfg = DEFAULT.replace(figure_anchor_seed=12345, anchored_fraction=1.0)
    d = validate(cfg)
    sets = {tuple(make_trial(cfg, 500 + t, 10.0, "rising", d=d).recurring.figure_set.tolist())
            for t in range(12)}
    assert len(sets) == 1, "an anchored figure must occupy the same channels every trial"


def test_no_anchor_redraws_every_trial():
    cfg = DEFAULT.replace(figure_anchor_seed=None, anchored_fraction=1.0)
    d = validate(cfg)
    sets = {tuple(make_trial(cfg, 500 + t, 10.0, "rising", d=d).recurring.figure_set.tolist())
            for t in range(12)}
    assert len(sets) >= 10, "without an anchor the figure should be redrawn each trial"


def test_anchor_leaves_the_foil_free():
    """The foil must still redraw its channels, or the contrast disappears.

    Compare the SEQUENCE of per-element sets, not their union: with 6 elements of 7 channels
    drawn from 24 the union covers almost the whole pool every trial, so it cannot discriminate.
    """
    cfg = DEFAULT.replace(figure_anchor_seed=12345)
    d = validate(cfg)
    seqs = set()
    for t in range(6):
        tr = make_trial(cfg, 700 + t, 10.0, "rising", d=d)
        seqs.add(tuple(tuple(s.tolist()) for s in tr.other.element_sets))
    assert len(seqs) == 6, "the foil must draw fresh channel sets on every trial"


@pytest.mark.parametrize("anchor", [None, 999])
def test_anchor_preserves_every_invariant(anchor):
    cfg = DEFAULT.replace(figure_anchor_seed=anchor, anchored_fraction=1.0)
    d = validate(cfg)
    for t in range(6):
        inv = stimulus.check_invariants(cfg, make_trial(cfg, 800 + t, 16.0, "rising", d=d), d)
        assert inv["same_n_tones"] and inv["same_channel_counts"]
        assert inv["budget_exact"] and inv["no_same_channel_overlap"]


def test_anchored_figure_is_reproducible_from_the_seed():
    a = DEFAULT.replace(figure_anchor_seed=4242, anchored_fraction=1.0)
    b = DEFAULT.replace(figure_anchor_seed=4242, anchored_fraction=1.0)
    c = DEFAULT.replace(figure_anchor_seed=4243, anchored_fraction=1.0)
    d = validate(a)
    Sa = make_trial(a, 1, 10.0, "rising", d=d).recurring.figure_set
    Sb = make_trial(b, 9, 10.0, "rising", d=d).recurring.figure_set
    Sc = make_trial(c, 1, 10.0, "rising", d=d).recurring.figure_set
    assert np.array_equal(Sa, Sb)
    assert not np.array_equal(Sa, Sc)


def test_figure_repeats_default_reproduces_prior_behaviour():
    assert DEFAULT.figure_repeats == 1
    d = validate(DEFAULT)
    # under matched incidence an element also holds its scattered counterpart, so the span is
    # twice the tone duration at step 0, not once.
    assert d.spans_ms[0] == 2 * DEFAULT.tone_dur_ms


def test_pilot_config_validates_and_is_audible():
    """The shipped pilot config must validate and keep the figure anchored."""
    import json, pathlib
    p = pathlib.Path(__file__).resolve().parent.parent / "pilot_config.json"
    if not p.exists():
        pytest.skip("pilot_config.json not present")
    cfg = config.Config.from_dict(json.loads(p.read_text()))
    d = validate(cfg)
    assert cfg.figure_anchor_seed is not None
    assert cfg.tone_dur_ms == 30.0, "the pilot runs 30 ms tones"
    # 30 ms tones were inaudible under the old dense pool; audibility is a masking question, so
    # assert the thing that actually matters rather than a proxy duration.
    from seqsfg import pool as pool_mod
    exc, own = pool_mod.excitation_from_pool(d.channel_freqs_hz, cfg.tone_level_db_spl,
                                             d.occupancy_per_channel)
    assert float(np.min(own - exc)) >= 6.0, "a figure tone must clear the pool's own excitation"
    assert d.est_session_minutes <= cfg.max_session_minutes
    assert d.peak_bound < 0.99


# ---- the foil subpool: superseded by matched incidence, kept working and tested ----
def _legacy():
    """The pre-matched-incidence construction, which foil_subpool_size belongs to."""
    return DEFAULT.replace(matched_incidence=False, foil_universe_size=None,
                           anchored_fraction=1.0, figure_min_spacing_channels=2,
                           steps_ms=(0.0, 4.0, 8.0, 12.0, 15.0, 18.0),
                           control_cells=(("ungrouped", 0.0), ("onechannel", 0.0)))



def test_foil_subpool_concentrates_the_foil_channels():
    """A restricted subpool makes the foil's channels recur nearly as often as the target's,
    which is what removes the single-channel periodicity residual."""
    import numpy as np
    from seqsfg.stimulus import make_trial
    wide = _legacy()
    tight = _legacy().replace(foil_subpool_size=12, max_shared_any=5, max_shared_consecutive=5)
    reuse = {}
    for label, cfg in (("wide", wide), ("tight", tight)):
        d = validate(cfg)
        tr = make_trial(cfg, 11, 10.0, "rising", d=d)
        uses = np.bincount(np.concatenate(list(tr.other.element_sets)), minlength=d.n_channels)
        reuse[label] = uses[uses > 0].mean()
    assert reuse["tight"] > reuse["wide"] * 1.5
    assert len(np.unique(np.concatenate(list(
        make_trial(tight, 11, 10.0, "rising", d=validate(tight)).other.element_sets)))) <= 12


def test_foil_still_redraws_within_the_subpool():
    """Concentrating the foil must not make it recur, or the manipulation disappears."""
    from seqsfg.stimulus import make_trial
    cfg = _legacy().replace(foil_subpool_size=12, max_shared_any=5, max_shared_consecutive=5)
    d = validate(cfg)
    tr = make_trial(cfg, 5, 10.0, "rising", d=d)
    sets = [tuple(s.tolist()) for s in tr.other.element_sets]
    assert len(set(sets)) >= cfg.n_elements - 1, "the foil must land on a different set each element"


@pytest.mark.parametrize("Q,shared,needle", [
    (7, 5, "must exceed n_components"),
    (16, 5, "exceeds the"),
    (10, 2, "share at least"),
])
def test_validator_refuses_impossible_subpools(Q, shared, needle):
    from seqsfg.config import ConfigError
    with pytest.raises(ConfigError) as e:
        validate(_legacy().replace(foil_subpool_size=Q, max_shared_any=shared,
                                   max_shared_consecutive=shared))
    assert needle in str(e.value)


def test_pilot_config_is_at_speech_rate():
    """The rate must stay >= 3 Hz: the paradigm's whole rationale is comparability with speech."""
    import json, pathlib
    p = pathlib.Path(__file__).resolve().parent.parent / "pilot_config.json"
    if not p.exists():
        pytest.skip("pilot_config.json not present")
    cfg = config.Config.from_dict(json.loads(p.read_text()))
    validate(cfg)
    assert 1000.0 / cfg.iei_max_ms >= 3.0, "element rate fell below 3 Hz"
    assert 1000.0 / cfg.iei_min_ms <= 5.0
    assert cfg.matched_incidence, "matched incidence is what makes 3-5 Hz affordable at n_components=7"
    assert cfg.n_components >= 7, "7 components is the floor the SFG literature works at"


def _pilot():
    import json, pathlib
    p = pathlib.Path(__file__).resolve().parent.parent / "pilot_config.json"
    if not p.exists():
        pytest.skip("pilot_config.json not present")
    return config.Config.from_dict(json.loads(p.read_text()))


def test_foil_elements_share_no_pitch_with_the_target_or_with_each_other():
    """'sam sam sam' vs 'bob kim she': no foil element may contain a target pitch, and consecutive
    foil elements must be disjoint. Both are guaranteed by construction, so this is exact."""
    cfg = _pilot()
    d = validate(cfg)
    for seed in range(1, 26):
        tr = stimulus.make_trial(cfg, seed, 0.0, "rising", d=d)
        S = set(tr.recurring.figure_set.tolist())
        sets = [set(x.tolist()) for x in tr.other.element_sets]
        for k, g in enumerate(sets):
            assert not (g & S), f"seed {seed} element {k}: foil element reuses a target pitch"
        for k, (a, b) in enumerate(zip(sets, sets[1:])):
            assert not (a & b), f"seed {seed} elements {k},{k+1}: consecutive foil elements overlap"


def test_matched_incidence_gives_the_two_intervals_identical_per_channel_counts():
    """Every channel carries the same number of tones in both intervals, so the long-term spectrum
    is identical by construction and cannot be what a listener uses."""
    cfg = _pilot()
    d = validate(cfg)
    for seed in range(1, 26):
        for variant in ("rising", "ungrouped"):
            tr = stimulus.make_trial(cfg, seed, 0.0, variant, d=d)
            a = np.bincount(tr.recurring.channel, minlength=d.n_channels)
            b = np.bincount(tr.other.channel, minlength=d.n_channels)
            assert np.array_equal(a, b), f"{variant} seed {seed}: per-channel counts differ"
            assert a.sum() == d.n_active_channels * cfg.tones_per_channel


def test_half_the_trials_use_the_anchored_figure():
    """anchored_fraction=0.5 splits trials between the learnable figure and a fresh one, so a
    familiar-vs-novel contrast is available within the session."""
    cfg = _pilot()
    if cfg.anchored_fraction >= 1.0:
        pytest.skip("this pilot anchors every trial")
    d = validate(cfg)
    trials = [stimulus.make_trial(cfg, s, 0.0, "rising", d=d) for s in range(1, 201)]
    anchored = [t for t in trials if t.anchored]
    assert 0.35 * len(trials) < len(anchored) < 0.65 * len(trials)
    fixed = {tuple(t.recurring.figure_set.tolist()) for t in anchored}
    assert len(fixed) == 1, "anchored trials must all use the SAME figure"
    fresh = {tuple(t.recurring.figure_set.tolist()) for t in trials if not t.anchored}
    assert len(fresh) > 0.5 * (len(trials) - len(anchored)), "unanchored trials must vary"
    assert not (fixed & fresh), "an unanchored trial happened to reuse the anchored figure"
