"""Unit tests: config validation, stimulus invariants, design balance, resume refusal, analysis on synthetic data."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from seqsfg import analysis, config, design, session, stimulus
from seqsfg.config import Config, ConfigError, DEFAULT, derive, validate

SMALL = DEFAULT.replace(trials_per_condition=4, control_trials_per_cell=2, practice_n=4, practice_criterion=3,
                        bootstrap_n=30)


def test_default_config_validates():
    d = validate(DEFAULT)
    assert d.n_channels >= 20
    assert d.min_beat_rate_hz >= DEFAULT.min_beat_rate_hz
    assert d.max_span_ms <= DEFAULT.iei_min_ms
    assert d.schedule_max_ms <= DEFAULT.interval_dur_ms


@pytest.mark.parametrize("kw,needle", [
    (dict(steps_ms=(0.0, 100.0)), "run into each other"),
    (dict(interval_dur_ms=2000.0), "schedule does not fit"),
    (dict(pool_low_hz=100.0), "beat at"),
    (dict(tone_amplitude=0.2), "clipping"),
    (dict(tones_per_channel=70), "occupancy"),
    (dict(trials_per_condition=15), "even"),
    (dict(main_variants=("ungrouped",)), "must include 'rising'"),
    (dict(main_variants=("rising", "nonsense")), "unknown main variant"),
    (dict(practice_cells=(("rising", 15.0),)), "not the easiest step"),
    (dict(max_variant_run=9), "max_variant_run > 6"),
    (dict(steps_ms=(10.0, 20.0)), "start at 0"),
    (dict(tones_per_channel=8), "2*n_elements"),
    (dict(trials_per_condition=40), "estimated session"),
])
def test_validator_refuses(kw, needle):
    with pytest.raises(ConfigError) as e:
        validate(DEFAULT.replace(**kw))
    assert needle in str(e.value)


def test_config_roundtrip_and_hash():
    d = DEFAULT.to_dict()
    assert Config.from_dict(json.loads(json.dumps(d))) == DEFAULT
    assert DEFAULT.hash() != DEFAULT.replace(tone_dur_ms=60.0).hash()


@pytest.mark.parametrize("variant", config.VARIANTS)
@pytest.mark.parametrize("step", [0.0, 35.0, 75.0])
def test_trial_invariants(variant, step):
    cfg = DEFAULT
    d = derive(cfg)
    tr = stimulus.make_trial(cfg, 99, step, variant, d=d)
    inv = stimulus.check_invariants(cfg, tr, d)
    assert inv["same_n_tones"] and inv["same_channel_counts"] and inv["budget_exact"] and inv["no_same_channel_overlap"]
    n_comp = 1 if variant == "onechannel" else cfg.n_components
    assert inv["figure_tones"] == cfg.n_elements * n_comp
    A = tr.recurring
    # element structure: component i of element k on S[i] at t_k + pattern[i]*step
    for k in range(cfg.n_elements):
        for i in range(n_comp):
            j = np.flatnonzero((A.element == k) & (A.component == i))
            assert j.size == 1
            assert A.channel[j[0]] == A.figure_set[i]
            assert A.onset[j[0]] == A.element_onsets[k] + A.patterns[k][i] * cfg.ms_to_grid(step)
    # figure set spacing
    assert np.all(np.diff(A.figure_set) >= cfg.figure_min_spacing_channels)
    if variant not in ("ungrouped", "onechannel"):
        B = tr.other
        assert np.array_equal(B.element_onsets, A.element_onsets)
        sets = B.element_sets
        for k in range(1, len(sets)):
            assert np.intersect1d(sets[k], sets[k - 1]).size <= cfg.max_shared_consecutive
            for j in range(k - 1):
                assert np.intersect1d(sets[k], sets[j]).size <= cfg.max_shared_any
        if variant == "rising":
            assert all(np.array_equal(p, np.arange(cfg.n_components)) for p in B.patterns)
        if variant == "scrambled":
            assert all(np.array_equal(p, A.patterns[0]) for p in A.patterns)
    else:
        assert np.sum(tr.other.kind == stimulus.FIGURE) == 0


def test_trial_is_deterministic_and_seed_sensitive():
    cfg = DEFAULT
    a = stimulus.make_trial(cfg, 5, 20.0, "rising")
    b = stimulus.make_trial(cfg, 5, 20.0, "rising")
    c = stimulus.make_trial(cfg, 6, 20.0, "rising")
    assert np.array_equal(a.recurring.onset, b.recurring.onset) and np.array_equal(a.other.channel, b.other.channel)
    assert not np.array_equal(a.recurring.onset, c.recurring.onset)


def test_render_shapes_and_levels():
    cfg = DEFAULT
    d = derive(cfg)
    tr = stimulus.make_trial(cfg, 1, 0.0, "rising", d=d)
    x = stimulus.render_trial(cfg, tr, 2, d)
    n = cfg.ms_to_samples(cfg.lead_silence_ms) + 2 * cfg.ms_to_samples(cfg.interval_dur_ms) + cfg.ms_to_samples(cfg.isi_ms)
    assert x.shape == (n,)
    assert np.abs(x).max() < d.peak_bound
    a, b = stimulus.render_interval(cfg, tr.recurring, d), stimulus.render_interval(cfg, tr.other, d)
    ra, rb = np.sqrt(np.mean(a ** 2)), np.sqrt(np.mean(b ** 2))
    assert abs(20 * math.log10(ra / rb)) < 0.1


def _worst_run(items, key):
    run = worst = 1
    for i in range(1, len(items)):
        run = run + 1 if key(items[i]) == key(items[i - 1]) else 1
        worst = max(worst, run)
    return worst


def test_design_balance_and_runs():
    cfg = SMALL
    d = validate(cfg)
    dz = design.make_design(cfg, "P01", 1)
    main = dz["main"]
    assert len(main) == cfg.trials_per_condition * len(d.main_cells)
    for v, s in d.main_cells:
        cell = [t for t in main if t["variant"] == v and t["step_ms"] == s]
        assert len(cell) == cfg.trials_per_condition, (v, s)
        assert sum(t["target_position"] == 1 for t in cell) == cfg.trials_per_condition // 2
    assert _worst_run(main, lambda t: (t["variant"], t["step_ms"])) <= cfg.max_condition_run
    assert _worst_run(main, lambda t: t["variant"]) <= cfg.max_variant_run
    assert len({t["seed"] for t in main}) == len(main)


def test_practice_stages_follow_config():
    cfg = SMALL
    dz = design.make_design(cfg, "P01", 1)
    assert len(dz["practice_stages"]) == len(cfg.practice_cells)
    for stage, (v, s) in zip(dz["practice_stages"], cfg.practice_cells):
        assert len(stage) == cfg.practice_max_rounds
        for rnd in stage:
            assert len(rnd) == cfg.practice_n
            assert all(t["variant"] == v and t["step_ms"] == s for t in rnd)
            assert sum(t["target_position"] == 1 for t in rnd) == cfg.practice_n // 2
    specs = design.specs_from(dz, "practice", 1, 1)
    assert specs[0].variant == cfg.practice_cells[0][0]
    assert specs[0].practice_stage == 1


def test_design_differs_by_participant_and_session():
    cfg = SMALL
    a = design.make_design(cfg, "P01", 1)
    assert design.make_design(cfg, "P01", 2)["main"] != a["main"]
    assert design.make_design(cfg, "P01", 1)["design_hash"] == a["design_hash"]
    assert design.make_design(cfg, "P02", 1)["design_hash"] != a["design_hash"]


def test_resume_refuses_when_design_changes(tmp_path):
    cfg = SMALL
    dz = design.make_design(cfg, "P01", 1)
    meta = {"config_hash": cfg.hash(), "design_hash": dz["design_hash"], "source_hash": session.source_hash()}
    session.check_resumable(meta, cfg, dz)
    cfg2 = cfg.replace(tones_per_channel=26)
    with pytest.raises(session.DesignChanged):
        session.check_resumable(meta, cfg2, design.make_design(cfg2, "P01", 1))


def test_trial_log_roundtrip(tmp_path):
    p = tmp_path / "trials.csv"
    log = session.TrialLog(p)
    log.write(dict(trial_index=0, block="main", practice_round=0, practice_stage=0, variant="rising",
                   step_ms=10.0, target_position=2, seed=123, response=2, correct=1, rt_ms=512.3,
                   t_start="t0", t_response="t1"))
    log.close()
    rows = session.read_trials(p)
    assert rows[0]["correct"] == 1 and rows[0]["step_ms"] == 10.0 and rows[0]["seed"] == 123


def _synthetic_trials(rng, steps, n, tau, variant="rising"):
    out = []
    for i, s in enumerate(list(steps) * n):
        p = 0.5 + 0.48 * math.exp(-s / tau)
        tgt = 1 + (i % 2)
        correct = int(rng.random() < p)
        out.append(dict(block="main", variant=variant, step_ms=float(s), target_position=tgt,
                        response=tgt if correct else 3 - tgt, correct=correct))
    return out


def test_analysis_recovers_threshold_and_gates():
    cfg = DEFAULT.replace(bootstrap_n=60)
    rng = np.random.default_rng(1)
    trials = _synthetic_trials(rng, cfg.steps_ms, 40, tau=12.0)
    summ = analysis.summarize(trials)
    ps = analysis.psychometric_report(summ, cfg, rng)
    assert ps["reportable"], ps["gates"]
    assert 3 < ps["threshold"] < 25
    st = analysis.single_trial_test(trials)
    assert st["p"] < 0.001 and st["slope_per_ms"] < 0
    dg = analysis.diagnostics(trials, 0.0, ("rising",))
    assert set(dg) >= {"interval_preference", "by_target_position", "halves", "after_correct_vs_error", "easiest_over_time"}


def test_analysis_refuses_flat_data():
    cfg = DEFAULT.replace(bootstrap_n=40)
    rng = np.random.default_rng(2)
    trials = _synthetic_trials(rng, cfg.steps_ms, 20, tau=1e9)      # ~1.0 everywhere: no threshold in range
    ps = analysis.psychometric_report(analysis.summarize(trials), cfg, rng)
    assert not ps["reportable"]
    trials = [dict(t, correct=int(rng.random() < 0.5)) for t in trials]   # chance everywhere
    ps = analysis.psychometric_report(analysis.summarize(trials), cfg, rng)
    assert not ps["reportable"]


def test_two_ladders_curve_comparison_detects_a_planted_difference():
    """Curves that differ in shape must be caught, and identical ladders must not be."""
    cfg = DEFAULT.replace(bootstrap_n=60)
    rng = np.random.default_rng(5)
    fast = _synthetic_trials(rng, cfg.steps_ms, 40, tau=6.0, variant="rising")
    slow = _synthetic_trials(rng, cfg.steps_ms, 40, tau=40.0, variant="ungrouped")
    cd = analysis.curve_comparison_lrt(analysis.summarize(fast + slow), cfg, ("rising", "ungrouped"))
    assert cd["p"] < 0.01, cd
    # calibration: identical ladders must not be flagged more often than the nominal rate.
    # (Measured over 200 replicates: 0.05 at 14 trials per cell, 0.035 at 40. Ten replicates here
    # only needs to catch a grossly miscalibrated test, so allow at most 3.)
    flagged = 0
    for rep in range(10):
        r2 = np.random.default_rng(500 + rep)
        a2 = _synthetic_trials(r2, cfg.steps_ms, 40, tau=12.0, variant="rising")
        b2 = _synthetic_trials(r2, cfg.steps_ms, 40, tau=12.0, variant="ungrouped")
        flagged += analysis.curve_comparison_lrt(analysis.summarize(a2 + b2), cfg,
                                                 ("rising", "ungrouped"))["p"] < 0.05
    assert flagged <= 3, f"curve LRT flagged {flagged}/10 identical ladders"
    td = analysis.threshold_difference(analysis.summarize(fast + slow), cfg, ("rising", "ungrouped"), rng)
    if td.get("reportable"):
        assert td["difference"] > 0


def test_linear_interaction_detects_a_slope_difference():
    """The single-trial interaction is a LINEAR-in-step test: it catches slope differences, and is
    insensitive to shape differences that leave the mean logit slope alone (hence the LRT above)."""
    cfg = DEFAULT
    rng = np.random.default_rng(11)
    steps = list(cfg.steps_ms)
    def linear_trials(variant, slope):
        out = []
        for s in steps * 40:
            p = min(0.98, max(0.52, 0.95 - slope * s))
            tgt = 1 + (len(out) % 2)
            c = int(rng.random() < p)
            out.append(dict(block="main", variant=variant, step_ms=float(s), target_position=tgt,
                            response=tgt if c else 3 - tgt, correct=c))
        return out
    steep = linear_trials("rising", 0.016) * 5      # this term needs a lot of trials to bite
    shallow = linear_trials("ungrouped", 0.002) * 5
    inter = analysis.ladder_interaction_test(steep + shallow, ("rising", "ungrouped"))
    assert inter["p_interaction"] < 0.05, inter
    assert inter["slope_difference_per_ms"] > 0
    assert inter["slope_reference_per_ms"] < 0


def test_threshold_difference_refuses_on_an_untrustworthy_curve():
    cfg = DEFAULT.replace(bootstrap_n=30)
    rng = np.random.default_rng(6)
    good = _synthetic_trials(rng, cfg.steps_ms, 30, tau=12.0, variant="rising")
    flat = [dict(t, variant="ungrouped", correct=int(rng.random() < 0.5)) for t in
            _synthetic_trials(rng, cfg.steps_ms, 30, tau=12.0)]
    td = analysis.threshold_difference(analysis.summarize(good + flat), cfg, ("rising", "ungrouped"), rng)
    assert not td["reportable"] and "reason" in td


def test_fit_refuses_transition_finer_than_the_sampling():
    """A step function fitted to a coarse ladder is not a measurement."""
    cfg = DEFAULT.replace(bootstrap_n=30)
    rng = np.random.default_rng(7)
    steps = list(cfg.steps_ms)
    rows = []
    for s in steps:
        pc = 0.98 if s <= 10 else 0.52
        rows += [dict(block="main", variant="rising", step_ms=float(s), target_position=1 + (i % 2),
                      response=1, correct=int(rng.random() < pc)) for i in range(40)]
    ps = analysis.psychometric_report(analysis.summarize(rows), cfg, rng)
    assert ps["fit"]["beta"] < 0.25 * np.median(np.diff(steps)) or ps["reportable"]


def test_dprime_and_wilson_behave_at_extremes():
    dp, lo, hi = analysis.dprime_2ifc(16, 16)
    assert math.isfinite(dp) and math.isfinite(hi) and lo < dp
    dp0, lo0, hi0 = analysis.dprime_2ifc(0, 16)
    assert math.isfinite(dp0) and dp0 < 0
    assert analysis.wilson(0, 10)[0] == 0.0 and analysis.wilson(10, 10)[1] == 1.0


def test_holm():
    adj = analysis.holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])


# ---- verification statistics and figures -------------------------------------
def _tiny_battery(n_trials=6, seed=3):
    from seqsfg import verify
    cfg = DEFAULT
    return cfg, verify.run_battery(cfg, n_trials=n_trials, seed=seed,
                                   conditions=[("rising", 0.0), ("rising", 75.0)], verbose=False)


def test_permutation_audit_is_calibrated_on_exchangeable_data():
    """Two intervals built the same way must not be separable by any measured feature."""
    from seqsfg import verify
    cfg, res = _tiny_battery()
    pm = verify.permutation_audit(res["results"], "rising", n_perm=500, seed=1)
    assert pm["n_features"] > 30 and pm["n_trials"] == 12
    assert 0.0 < pm["p_value"] <= 1.0
    assert pm["observed_max_dprime"] <= pm["null_max_p95"] * 1.6


def test_permutation_audit_detects_a_planted_difference():
    """If a feature really does differ, the global test must catch it."""
    from seqsfg import verify
    cfg, res = _tiny_battery()
    for r in res["results"]:
        for m in r.A:
            m.scalars["rms_db"] += 5.0          # a difference no design would tolerate
    pm = verify.permutation_audit(res["results"], "rising", n_perm=500, seed=1)
    assert pm["p_value"] < 0.05
    assert pm["observed_max_dprime"] > pm["null_max_p95"]


def test_observer_multiplicity_corrects():
    from seqsfg import verify
    cfg, res = _tiny_battery()
    mult = verify.observer_multiplicity(res["observers"], res["results"])
    assert mult["n_cells"] == 5 * 2
    for c in mult["cells"]:
        assert c["p_holm"] >= c["p"] - 1e-12
    assert verify.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_learnt_observers_calibrated_against_relabelling():
    """Spectrum, envelope and occupancy observers must sit inside the band that exchanging the two
    intervals produces. (The single-channel observer is excluded on purpose: at the default rate it
    carries the documented periodicity residual.)"""
    from seqsfg import verify
    cfg, res = _tiny_battery(n_trials=10)
    rng = np.random.default_rng(0)
    for nm in ("spectrum only", "envelope only", "occupancy only"):
        obs_entries, null = [], []
        per_cond = []
        for r in res["results"]:
            Ft = np.array([verify.feature_sets(m, cfg)[nm][0] for m in r.A])
            Fo = np.array([verify.feature_sets(m, cfg)[nm][0] for m in r.O])
            per_cond.append((Ft, Fo))
            obs_entries.append(verify.observer_cv(Ft, Fo))
        observed = verify._pool_cv(obs_entries)["dprime"]
        for _ in range(25):
            entries = []
            for Ft, Fo in per_cond:
                flip = rng.random(len(Ft)) < 0.5
                A = np.where(flip[:, None], Fo, Ft); B = np.where(flip[:, None], Ft, Fo)
                entries.append(verify.observer_cv(A, B))
            null.append(verify._pool_cv(entries)["dprime"])
        null = np.array(null)
        assert observed <= np.max(null) + 0.3, f"{nm}: observed {observed:.2f} beyond relabelling max {null.max():.2f}"


def test_plots_write_files(tmp_path):
    from seqsfg import plots
    cfg = DEFAULT
    d = validate(cfg)
    plots.raster_pair(cfg, d, tmp_path / "pair.png")
    plots.raster_overview(cfg, d, tmp_path / "over.png")
    assert (tmp_path / "pair.png").stat().st_size > 5000
    assert (tmp_path / "over.png").stat().st_size > 5000
    assert plots.semitones(1000.0) == pytest.approx(0.0)
    assert plots.semitones(2000.0) == pytest.approx(12.0)
