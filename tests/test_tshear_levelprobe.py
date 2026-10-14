"""The level-probe trials leave exactly one cue intact, and the waveforms say so too."""
import numpy as np
import pytest

from tshear import levelprobe as L
from tshear.config import DEFAULT as CFG, onsets_ms

T = CFG.target_index
OTHERS = [j for j in range(CFG.n_tones) if j != T]


def _many(task, delta=4.0, n=300, seed=0):
    rng = np.random.default_rng(seed)
    return [L.make_trial(task, 25.0, delta, rng) for _ in range(n)]


def _sig_std(tr):
    return (tr.first, tr.second) if tr.target_position == 1 else (tr.second, tr.first)


def test_a_keeps_the_timbre_cue_exactly_and_scrambles_the_own_stream_cue():
    own = []
    for tr in _many("A"):
        s, n = _sig_std(tr)
        ts, os_ = L.cues(CFG, s.gains_db)
        tn, on = L.cues(CFG, n.gains_db)
        assert ts == pytest.approx(4.0) and tn == pytest.approx(0.0)
        # every earlier token is a flat profile: the rove is common to all four tones
        for g in (s.gains_db, n.gains_db):
            assert np.allclose(g[:-1] - g[:-1, [0]], 0.0)
        own.append(os_ - on)
    assert np.std(own) > 5.0                      # the destroyed cue is genuinely noisy


def test_b_keeps_the_own_stream_cue_exactly_and_scrambles_the_timbre_cue():
    tim = []
    for tr in _many("B"):
        s, n = _sig_std(tr)
        ts, os_ = L.cues(CFG, s.gains_db)
        tn, on = L.cues(CFG, n.gains_db)
        assert os_ == pytest.approx(4.0) and on == pytest.approx(0.0)
        # the target's stream is shifted as a whole; the others only move on the last token
        for g in (s.gains_db, n.gains_db):
            assert np.allclose(g[:-1, OTHERS], 0.0)
            assert np.ptp(g[:-1, T]) == pytest.approx(0.0)
        tim.append(ts - tn)
    assert np.std(tim) > 5.0


def test_without_a_rove_both_cues_survive():
    for tr in _many("none", n=20):
        s, n = _sig_std(tr)
        assert L.cues(CFG, s.gains_db) == pytest.approx((4.0, 4.0))
        assert L.cues(CFG, n.gains_db) == pytest.approx((0.0, 0.0))


def test_the_two_intervals_differ_only_in_gain():
    tr = L.make_trial("A", 45.0, 6.0, np.random.default_rng(3))
    assert np.array_equal(tr.first.onsets_ms, tr.second.onsets_ms)
    assert tr.first.is_signal != tr.second.is_signal


def test_geometry_is_the_timing_tasks_geometry():
    for s in L.STEPS_MS:
        b = np.stack([onsets_ms(CFG, s / CFG.iso_step_ms * 100, r) for r in range(CFG.n_repeats)])
        assert np.allclose(L.onsets(CFG, s), b)


def test_the_rendered_waveform_carries_the_gain_matrix():
    """Measure each tone of each repetition on the waveform and read the gain back."""
    tr = L.make_trial("B", 75.0, 6.0, np.random.default_rng(5))
    fs, env_n = CFG.sample_rate, int(round(CFG.tone_ms * CFG.sample_rate / 1000))
    for iv in tr.intervals():
        for k in range(CFG.n_tones):
            x = L.render(CFG, iv, tr.phases, only_tone=k)
            for rep in range(CFG.n_repeats):
                i = int(round(iv.onsets_ms[rep, k] * fs / 1000))
                seg = x[i + env_n // 4: i + 3 * env_n // 4]          # inside the ramps
                db = 20 * np.log10(np.sqrt(2) * np.sqrt(np.mean(seg ** 2)) / L.AMPLITUDE)
                assert db == pytest.approx(iv.gains_db[rep, k], abs=0.05)


def test_every_step_gives_the_same_interval_length():
    lens = set()
    for s in L.STEPS_MS:
        tr = L.make_trial("A", s, 4.0, np.random.default_rng(1))
        lens.add(L.render(CFG, tr.first, tr.phases).size)
    assert len(lens) == 1


@pytest.mark.parametrize("task", ["A", "B"])
def test_the_ceiling_sits_well_above_where_the_surviving_cue_works(task):
    c = L.ceiling(CFG, task, rove_db=10.0)
    assert 7.0 < c < 9.0
    # and a small rove does not protect the design: the wrong cue already works at ~4 dB
    assert L.ceiling(CFG, task, rove_db=5.0) < 4.5


def test_nothing_clips_at_the_default_amplitude():
    assert L.worst_case_peak(CFG, 10.0, 12.0) < 1.0
    rng = np.random.default_rng(9)
    peak = 0.0
    for _ in range(200):
        tr = L.make_trial("A", 0.0, 12.0, rng, rove_db=10.0)
        for iv in tr.intervals():
            peak = max(peak, float(np.abs(L.render(CFG, iv, tr.phases)).max()))
    assert peak < 1.0


def test_a_listener_who_switches_cue_with_binding_produces_the_crossover():
    """The logic of the design, checked on the real trial generator.

    A bound listener reads timbre; a segregated one reads the target's own stream. The first
    should beat type B on type A, the second the reverse. A listener who always reads the right
    cue for the trial type shows no such reversal.
    """
    kw = dict(delta_db=4.0, noise_timbre_db=2.5, noise_own_db=1.5, n=8000)
    bound = {t: L.p_correct(CFG, t, w_timbre=1.0, **kw) for t in ("A", "B")}
    apart = {t: L.p_correct(CFG, t, w_timbre=0.0, **kw) for t in ("A", "B")}
    assert bound["A"] > bound["B"] + 0.1
    assert apart["B"] > apart["A"] + 0.1
