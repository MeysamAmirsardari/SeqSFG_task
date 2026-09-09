"""The anchored figure: the same channels in every trial, without breaking the matching."""
import numpy as np
import pytest

from seqsfg import config, stimulus
from seqsfg.config import DEFAULT, validate
from seqsfg.stimulus import FIGURE, make_trial


def test_anchor_fixes_the_figure_across_trials():
    cfg = DEFAULT.replace(figure_anchor_seed=12345)
    d = validate(cfg)
    sets = {tuple(make_trial(cfg, 500 + t, 10.0, "rising", d=d).recurring.figure_set.tolist())
            for t in range(12)}
    assert len(sets) == 1, "an anchored figure must occupy the same channels every trial"


def test_no_anchor_redraws_every_trial():
    cfg = DEFAULT.replace(figure_anchor_seed=None)
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
    cfg = DEFAULT.replace(figure_anchor_seed=anchor)
    d = validate(cfg)
    for t in range(6):
        inv = stimulus.check_invariants(cfg, make_trial(cfg, 800 + t, 16.0, "rising", d=d), d)
        assert inv["same_n_tones"] and inv["same_channel_counts"]
        assert inv["budget_exact"] and inv["no_same_channel_overlap"]


def test_anchored_figure_is_reproducible_from_the_seed():
    a = DEFAULT.replace(figure_anchor_seed=4242)
    b = DEFAULT.replace(figure_anchor_seed=4242)
    c = DEFAULT.replace(figure_anchor_seed=4243)
    d = validate(a)
    Sa = make_trial(a, 1, 10.0, "rising", d=d).recurring.figure_set
    Sb = make_trial(b, 9, 10.0, "rising", d=d).recurring.figure_set
    Sc = make_trial(c, 1, 10.0, "rising", d=d).recurring.figure_set
    assert np.array_equal(Sa, Sb)
    assert not np.array_equal(Sa, Sc)


def test_figure_repeats_default_reproduces_prior_behaviour():
    assert DEFAULT.figure_repeats == 1
    d = validate(DEFAULT)
    assert d.spans_ms[0] == DEFAULT.tone_dur_ms


def test_pilot_config_validates_and_is_audible():
    """The shipped pilot config must validate and keep the figure anchored."""
    import json, pathlib
    p = pathlib.Path(__file__).resolve().parent.parent / "pilot_config.json"
    if not p.exists():
        pytest.skip("pilot_config.json not present")
    cfg = config.Config.from_dict(json.loads(p.read_text()))
    d = validate(cfg)
    assert cfg.figure_anchor_seed is not None
    assert cfg.tone_dur_ms >= 50.0, "the pilot config exists because 30 ms tones were inaudible"
    assert d.est_session_minutes <= cfg.max_session_minutes
    assert d.peak_bound < 0.99


# ---- the foil subpool, which is what restored the speech rate ----
def test_foil_subpool_concentrates_the_foil_channels():
    """A restricted subpool makes the foil's channels recur nearly as often as the target's,
    which is what removes the single-channel periodicity residual."""
    import numpy as np
    from seqsfg.stimulus import make_trial
    wide = DEFAULT.replace(foil_subpool_size=None)
    tight = DEFAULT.replace(foil_subpool_size=12, max_shared_any=5, max_shared_consecutive=5)
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
    cfg = DEFAULT.replace(foil_subpool_size=12, max_shared_any=5, max_shared_consecutive=5)
    d = validate(cfg)
    tr = make_trial(cfg, 5, 10.0, "rising", d=d)
    sets = [tuple(s.tolist()) for s in tr.other.element_sets]
    assert len(set(sets)) >= cfg.n_elements - 1, "the foil must land on a different set each element"


@pytest.mark.parametrize("Q,shared,needle", [
    (7, 5, "must exceed n_components"),
    (20, 5, "exceeds the"),
    (10, 2, "share at least"),
])
def test_validator_refuses_impossible_subpools(Q, shared, needle):
    from seqsfg.config import ConfigError
    with pytest.raises(ConfigError) as e:
        validate(DEFAULT.replace(foil_subpool_size=Q, max_shared_any=shared,
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
    assert cfg.foil_subpool_size is not None, "the subpool is what makes 3-5 Hz affordable"
