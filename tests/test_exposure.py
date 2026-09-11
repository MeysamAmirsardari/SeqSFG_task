"""The pre / exposure / post mode. The two-interval and plain yes/no tasks must be untouched."""
import csv
import json
import pathlib

import numpy as np
import pytest

from seqsfg import exposure as X
from seqsfg import yesno
from seqsfg.config import Config, validate
from seqsfg.stimulus import FIGURE, render_interval

PRESET = pathlib.Path(__file__).resolve().parent.parent / "exposure_pilot.json"


@pytest.fixture(scope="module")
def cfge():
    if not PRESET.exists():
        pytest.skip("exposure_pilot.json not present")
    return X.load_preset(PRESET)


# ---- the experiment must not have leaked into the shared configuration -------
def test_exposure_parameters_are_not_config_fields():
    """Adding these to Config would change the hash of every existing configuration and break
    resume on sessions already recorded. They live in their own dataclass for that reason."""
    import dataclasses
    cfg_names = {f.name for f in dataclasses.fields(Config)}
    exp_names = {f.name for f in dataclasses.fields(X.ExposureConfig)}
    assert not (cfg_names & exp_names)


def test_shipped_pilot_config_still_loads_and_validates():
    p = PRESET.parent / "pilot_config.json"
    if not p.exists():
        pytest.skip("pilot_config.json not present")
    validate(Config.from_dict(json.loads(p.read_text())))


# ---- the two orders ----------------------------------------------------------
def test_orders_share_no_directed_transition_and_overlap_is_quantified():
    for n in (5, 6, 7):
        P, Q, ov = X.choose_orders(n, 20260910)
        assert sorted(P) == list(range(n)) and sorted(Q) == list(range(n))
        assert P != Q
        assert ov["shared_directed_transitions"] == 0
        assert ov["n_directed_transitions"] == n, "the wrap into the next repetition counts"
        for k in ("shared_undirected_adjacencies", "components_in_the_same_slot",
                  "kendall_tau_of_heard_sequences", "direction_changes"):
            assert k in ov, "residual similarity must be reported, not assumed away"


def test_orders_are_not_a_rising_or_falling_sweep():
    for n in (5, 7):
        P, Q, _ = X.choose_orders(n, 20260910)
        for o in (P, Q):
            assert list(o) != list(range(n)) and list(o) != list(reversed(range(n)))


def test_transitions_include_the_wrap_between_repetitions():
    t = X.directed_transitions([0, 1, 2])
    assert len(t) == 3 and (2, 0) in t


def test_orders_are_reproducible_from_the_seed():
    assert X.choose_orders(7, 5) == X.choose_orders(7, 5)
    assert X.choose_orders(7, 5)[0] != X.choose_orders(7, 6)[0]


# ---- counterbalancing --------------------------------------------------------
def test_training_assignment_is_explicit_reproducible_and_two_sided():
    e = X.ExposureConfig(trained="auto")
    assert X.assign_trained("P01", e) == X.assign_trained("P01", e)
    got = {X.assign_trained(f"S{i:03d}", e) for i in range(40)}
    assert got == {"P", "Q"}, "auto assignment must produce both, not always the same one"
    assert X.assign_trained("anything", X.ExposureConfig(trained="P")) == "P"
    assert X.assign_trained("anything", X.ExposureConfig(trained="Q")) == "Q"


# ---- design ------------------------------------------------------------------
def test_phases_are_balanced_within_phase_sequence_and_delay(cfge):
    cfg, e = cfge
    dz = X.make_design(cfg, e, "P01", 1)
    for phase in ("pre", "post"):
        rows = dz["phases"][phase]
        assert len(rows) == 2 * len(e.test_steps_ms) * 2 * e.test_trials_per_cell
        for seq in X.SEQUENCES:
            for step in e.test_steps_ms:
                cell = [r for r in rows if r["sequence"] == seq and r["step_ms"] == step]
                assert len(cell) == 2 * e.test_trials_per_cell
                assert sum(r["present"] for r in cell) == e.test_trials_per_cell


def test_exposure_phase_is_the_trained_sequence_only_and_never_at_step_zero(cfge):
    cfg, e = cfge
    dz = X.make_design(cfg, e, "P01", 1)
    rows = dz["phases"]["exposure"]
    assert len(rows) == e.exposure_trials
    assert {r["sequence"] for r in rows} == {dz["trained"]}
    assert all(r["step_ms"] != 0.0 for r in rows), "step 0 carries no order to learn"
    assert sum(r["present"] for r in rows) == round(e.exposure_trials * e.exposure_present_fraction)


def test_no_stimulus_seed_is_reused_anywhere_in_the_session(cfge):
    cfg, e = cfge
    dz = X.make_design(cfg, e, "P01", 1)
    seeds = [r["seed"] for ph in dz["phases"].values() for r in ph]
    assert len(set(seeds)) == len(seeds), "fresh acoustics everywhere; only S and the orders persist"


def test_answers_and_designations_do_not_run_on(cfge):
    cfg, e = cfge
    dz = X.make_design(cfg, e, "P01", 1)
    for phase in ("pre", "post"):
        rows = dz["phases"][phase]
        for key, limit in (("present", 3), ("sequence", 4)):
            run = longest = 1
            for a, b in zip(rows, rows[1:]):
                run = run + 1 if a[key] == b[key] else 1
                longest = max(longest, run)
            assert longest <= limit


def test_design_is_reproducible_and_participant_specific(cfge):
    cfg, e = cfge
    assert X.make_design(cfg, e, "P01", 1)["design_hash"] == X.make_design(cfg, e, "P01", 1)["design_hash"]
    assert X.make_design(cfg, e, "P01", 1)["design_hash"] != X.make_design(cfg, e, "P02", 1)["design_hash"]


# ---- sequence persistence and stimulus matching -------------------------------
def test_the_sequence_identity_is_the_same_in_every_phase(cfge):
    cfg, e = cfge
    d = validate(cfg)
    dz = X.make_design(cfg, e, "P01", 1)
    S = np.array(dz["figure_set"])
    order = tuple(dz["orders"][dz["trained"]])
    sets, orders = set(), set()
    for phase in ("pre", "exposure", "post"):
        for r in dz["phases"][phase][:4]:
            if r["sequence"] != dz["trained"] or not r["present"]:
                continue
            iv = X.build_trial(cfg, d, S, order, r["seed"], r["step_ms"], True, e.absent_class)
            sets.add(tuple(iv.figure_set.tolist()))
            orders.add(tuple(iv.patterns[0].tolist()))
    assert len(sets) == 1 and len(orders) == 1
    assert orders.pop() == order


def test_present_and_absent_are_matched_and_share_their_order(cfge):
    """Within a designation the onset order is the same on yes and no trials, so order alone
    cannot reveal the answer; and every interval carries the same tones in the same channels."""
    cfg, e = cfge
    d = validate(cfg)
    P, Q, _ = X.choose_orders(cfg.n_components, e.sequence_seed)
    S = X.figure_set_for(cfg, d, e)
    for order in (P, Q):
        counts = set()
        for seed in range(1, 7):
            for present in (True, False):
                iv = X.build_trial(cfg, d, S, order, seed, 7.0, present, e.absent_class)
                assert tuple(iv.patterns[0].tolist()) == tuple(order)
                c = np.bincount(iv.channel, minlength=d.n_channels)
                assert set(np.unique(c[c > 0])) == {cfg.tones_per_channel}
                counts.add((int(c.sum()), int((c > 0).sum())))
        assert len(counts) == 1


def test_present_uses_the_fixed_set_and_absent_never_does(cfge):
    cfg, e = cfge
    d = validate(cfg)
    P, _, _ = X.choose_orders(cfg.n_components, e.sequence_seed)
    S = X.figure_set_for(cfg, d, e)
    Sset = set(S.tolist())
    for seed in range(1, 7):
        p = X.build_trial(cfg, d, S, P, seed, 7.0, True, e.absent_class)
        a = X.build_trial(cfg, d, S, P, seed, 7.0, False, e.absent_class)
        assert {tuple(s.tolist()) for s in p.element_sets} == {tuple(sorted(Sset))}
        for g in a.element_sets:
            assert not (set(g.tolist()) & Sset), "an absent trial must not contain the target set"


def test_the_two_orders_are_the_same_sound_at_step_zero_and_differ_otherwise(cfge):
    cfg, e = cfge
    d = validate(cfg)
    P, Q, _ = X.choose_orders(cfg.n_components, e.sequence_seed)
    S = X.figure_set_for(cfg, d, e)
    xp = render_interval(cfg, X.build_trial(cfg, d, S, P, 42, 0.0, True), d)
    xq = render_interval(cfg, X.build_trial(cfg, d, S, Q, 42, 0.0, True), d)
    assert np.array_equal(xp, xq), "at zero onset separation the orders cannot differ"
    yp = render_interval(cfg, X.build_trial(cfg, d, S, P, 42, 7.0, True), d)
    yq = render_interval(cfg, X.build_trial(cfg, d, S, Q, 42, 7.0, True), d)
    assert not np.array_equal(yp, yq)


def test_build_trial_refuses_a_non_roving_absent_class(cfge):
    cfg, e = cfge
    d = validate(cfg)
    with pytest.raises(ValueError):
        X.build_trial(cfg, d, X.figure_set_for(cfg, d, e), (0, 1, 2, 3, 4, 5, 6), 1, 0.0, False, "plain")


# ---- configuration guards ------------------------------------------------------
@pytest.mark.parametrize("kw,needle", [
    (dict(absent_class="plain"), "roving"),
    (dict(exposure_steps_ms=(0.0,)), "no order information"),
    (dict(test_steps_ms=(0.0,)), "no nonzero delay"),
    (dict(test_steps_ms=(0.0, 3.5)), "validated ladder"),
    (dict(trained="R"), "must be 'P'"),
])
def test_check_refuses_combinations_that_change_the_question(cfge, kw, needle):
    cfg, e = cfge
    with pytest.raises(ValueError) as ex:
        X.check(cfg, X.ExposureConfig(**{**e.to_dict(), **kw}))
    assert needle in str(ex.value)


def test_check_refuses_a_mixture_of_anchored_and_fresh_targets(cfge):
    cfg, e = cfge
    with pytest.raises(ValueError) as ex:
        X.check(cfg.replace(anchored_fraction=0.5), e)
    assert "anchored_fraction" in str(ex.value)


# ---- scoring -------------------------------------------------------------------
def _row(phase, role, step, present, response):
    return {"phase": phase, "role": role, "step_ms": str(step),
            "present": "1" if present else "0", "response": response}


def _synthetic(hit_rates, n=20):
    """hit_rates[(phase, role)] = (hit rate, false-alarm rate) at one nonzero delay."""
    rows = []
    for (phase, role), (h, f) in hit_rates.items():
        for i in range(n):
            rows.append(_row(phase, role, 7.0, True, "y" if i < round(h * n) else "n"))
            rows.append(_row(phase, role, 7.0, False, "y" if i < round(f * n) else "n"))
    return rows


def test_difference_in_differences_recovers_a_planted_effect():
    e = X.ExposureConfig(test_steps_ms=(0.0, 7.0), n_boot=800)
    flat = {("pre", "trained"): (0.70, 0.30), ("pre", "untrained"): (0.70, 0.30),
            ("post", "trained"): (0.70, 0.30), ("post", "untrained"): (0.70, 0.30)}
    r = X.difference_in_differences(_synthetic(flat), e)
    assert abs(r["per_delay"][7.0]["D"]) < 1e-9
    lo, hi = r["per_delay"][7.0]["ci"]
    assert lo < 0 < hi

    planted = {**flat, ("post", "trained"): (0.90, 0.20)}
    r2 = X.difference_in_differences(_synthetic(planted), e)
    assert r2["per_delay"][7.0]["D"] > 0.8
    assert r2["per_delay"][7.0]["gain_untrained"] == pytest.approx(0.0, abs=1e-9)


def test_general_practice_that_helps_both_sequences_cancels():
    e = X.ExposureConfig(test_steps_ms=(0.0, 7.0), n_boot=600)
    both = {("pre", "trained"): (0.60, 0.35), ("pre", "untrained"): (0.60, 0.35),
            ("post", "trained"): (0.85, 0.20), ("post", "untrained"): (0.85, 0.20)}
    r = X.difference_in_differences(_synthetic(both), e)
    assert abs(r["per_delay"][7.0]["D"]) < 1e-9
    assert r["per_delay"][7.0]["gain_trained"] > 0.8, "both improved; D still has to be zero"


def test_ceiling_cells_do_not_produce_a_zero_width_interval():
    e = X.ExposureConfig(test_steps_ms=(0.0, 7.0), n_boot=800)
    ceil = {("pre", "trained"): (1.0, 0.0), ("pre", "untrained"): (1.0, 0.0),
            ("post", "trained"): (1.0, 0.0), ("post", "untrained"): (1.0, 0.0)}
    r = X.difference_in_differences(_synthetic(ceil), e)
    lo, hi = r["per_delay"][7.0]["ci"]
    assert np.isfinite(lo) and np.isfinite(hi) and hi - lo > 0.05


def test_aggregate_is_the_declared_rule_and_delay_results_are_kept():
    e = X.ExposureConfig(test_steps_ms=(0.0, 7.0, 14.0), n_boot=400)
    rows = _synthetic({("pre", "trained"): (0.6, 0.3), ("pre", "untrained"): (0.6, 0.3),
                       ("post", "trained"): (0.9, 0.2), ("post", "untrained"): (0.6, 0.3)})
    rows += [dict(r, step_ms="14.0") for r in rows]
    r = X.difference_in_differences(rows, e)
    assert set(r["per_delay"]) == {7.0, 14.0}
    assert r["aggregate"]["D"] == pytest.approx(np.mean([r["per_delay"][s]["D"] for s in (7.0, 14.0)]))
    assert r["aggregate_rule"] == "mean_over_nonzero_delays"


def test_empty_cells_are_reported_not_silently_dropped():
    e = X.ExposureConfig(test_steps_ms=(0.0, 7.0), n_boot=200)
    rows = _synthetic({("pre", "trained"): (0.6, 0.3), ("pre", "untrained"): (0.6, 0.3),
                       ("post", "trained"): (0.9, 0.2)})     # post/untrained missing entirely
    r = X.difference_in_differences(rows, e)
    assert r["per_delay"] == {} and r["aggregate"] is None
    assert r["not_estimable"] and r["not_estimable"][0]["step_ms"] == 7.0


def test_step_zero_check_compares_the_designations(cfge):
    rows = [_row("pre", "trained", 0.0, True, "y"), _row("pre", "trained", 0.0, False, "n"),
            _row("pre", "untrained", 0.0, True, "n"), _row("pre", "untrained", 0.0, False, "n")]
    z = X.step_zero_check(rows)
    assert z["trained"]["dprime"] > z["untrained"]["dprime"]


# ---- incomplete sessions --------------------------------------------------------
def test_analysis_survives_an_interrupted_session(tmp_path, cfge):
    cfg, e = cfge
    sdir = tmp_path / "P9" / "session_01"
    sdir.mkdir(parents=True)
    dz = X.make_design(cfg, e, "P9", 1)
    (sdir / "session.json").write_text(json.dumps(
        {"participant_code": "P9", "status": "interrupted", "phases_completed": ["practice", "pre"],
         "design": dz}))
    with open(sdir / "trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=X.LOG_FIELDS); w.writeheader()
        for r in dz["phases"]["pre"][:20]:
            w.writerow({"phase": "pre", "role": "trained", "step_ms": r["step_ms"],
                        "present": r["present"], "response": "y", "sequence": r["sequence"]})
    text = X.analyse([sdir])
    assert "INTERRUPTED" in text and "NOT ESTIMABLE" in text


def test_analysis_of_a_directory_with_no_trials_says_so(tmp_path):
    sdir = tmp_path / "P8" / "session_01"
    sdir.mkdir(parents=True)
    (sdir / "session.json").write_text(json.dumps({"participant_code": "P8", "status": "started"}))
    assert "no trials recorded" in X.analyse([sdir])
