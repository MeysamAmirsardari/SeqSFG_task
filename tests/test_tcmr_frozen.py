"""tcmr frozen-pattern design (plan C, 2026-11-14): orders, timing, the seed audit, the session."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tcmr import audit
from tcmr.cli import drive, sim_thresholds
from tcmr.config import Condition, Config, ConfigError, validate
from tcmr.procedure import SimListener
from tcmr.session import FIELDS, Experiment, Writer, order_plan
from tcmr.stimulus import build, order_code, sha256

CONFIGS = Path(__file__).resolve().parents[1] / "tcmr" / "configs"
FROZEN = validate(Config.load(CONFIGS / "tcmr_frozen.json"))
SR = FROZEN.sample_rate
NB = int(round(FROZEN.burst_ms / 1000 * SR))


def test_old_presets_and_trials_are_unchanged():
    # hashes and sounds recorded before the patterns were added (2026-11-14)
    want = {"tcmr_main": "cefe86afe3759e5a", "tcmr_ascending": "5f8d790bb48fbaf9",
            "tcmr_baseline": "8a9907a194a3e9ef"}
    for name, h in want.items():
        assert Config.load(CONFIGS / f"{name}.json").hash() == h
    main = Config.load(CONFIGS / "tcmr_main.json")
    assert sha256(build(main, main.condition_list()[0], 66.0, (9, 1, 2))[1]) == \
        "27edc7e01178b7675069a7013371a60182296a673ac9aa165355ebb41a015651"
    asc = Config.load(CONFIGS / "tcmr_ascending.json")
    assert sha256(build(asc, asc.condition_list()[2], 66.0, (9, 1, 2))[1]) == \
        "05770b9b70de68deae593c1b38beb67c8e87681f3a35c5bbe9bc25784b08e6a3"
    assert sha256(build(Config(), Condition("co", "sync", 0.0), 66.0, (9,))[1]) == \
        "684cad9e863eb459605e4c7f5d6f20d2d761d9545ae06ff4fcbbc35aca61f1d2"


def test_plan_c_preset():
    names = [c.name for c in FROZEN.condition_list()]
    assert names == ["co_sync_0", "rand_sync_0", "co_frozen_10", "rand_frozen_10", "co_frozen_20",
                     "rand_frozen_20", "co_frozen_40", "rand_frozen_40", "co_reordered_40",
                     "rand_reordered_40", "ref_sync_0"]
    assert (FROZEN.burst_ms, FROZEN.ramp_ms, FROZEN.noi_ms, FROZEN.n_precursors, FROZEN.lead_ms) == \
        (50.0, 10.0, 200.0, 5, 80.0)
    assert FROZEN.trials_per_track == 32 and FROZEN.tone_hz == 1000.0 and not FROZEN.tone_bands
    assert len(FROZEN.orders()) == 24 and all(o[FROZEN.target_band] == 0 for o in FROZEN.orders())


def test_validation_refuses_what_patterns_cannot_do():
    with pytest.raises(ConfigError):
        validate(FROZEN.replace(lead_ms=None))
    with pytest.raises(ConfigError):
        validate(FROZEN.replace(lead_ms=60.0))                    # shorter than 2 x 40 ms
    with pytest.raises(ConfigError):
        validate(FROZEN.replace(tone_bands=(2, 3)))
    with pytest.raises(ConfigError):
        validate(FROZEN.replace(conditions=(("ref", "frozen", 20.0, 1),)))


def test_frozen_trial_one_order_per_interval_all_different():
    for k in range(30):
        tr, buf = build(FROZEN, Condition("co", "frozen", 40.0), 70.0, (5, k))
        firsts = [tuple(reps[0]) for reps in tr.slots]
        assert len(set(firsts)) == 3                                    # three different orders
        for reps in tr.slots:
            assert all(r == reps[0] for r in reps)                      # frozen across repetitions
            assert reps[0][FROZEN.target_band] == 0
            assert sorted(v for b, v in enumerate(reps[0]) if b != FROZEN.target_band) == [-2, -1, 1, 2]


def test_onsets_window_and_final_burst_to_the_sample():
    starts_tone = set()
    for c in FROZEN.condition_list():
        tr, buf, stem = build(FROZEN, c, 70.0, (3, 1), stems=True)
        assert buf.shape[0] == 208320                                   # one length for every trial
        assert np.abs(stem.sum(0) - buf[:, 0]).max() < 1e-6
        sch = audit.schedule(FROZEN, tr)
        for b, starts in sch.items():
            mask = np.zeros(stem.shape[1], bool)
            for i0 in starts:
                mask[i0:i0 + NB] = True
                assert np.any(stem[b, i0:i0 + NB])
            assert not np.any(stem[b, ~mask])                           # nothing outside its bursts
        for iv in range(3):                                             # final burst synchronous
            assert len({s[iv * 6 + 5] for s in sch.values()}) == 1
        starts_tone.add(tuple(sch[FROZEN.target_band]))
        assert [round(b - a, 6) for a, b in zip(tr.interval_on_s, tr.interval_off_s)] == [1.13] * 3
    assert len(starts_tone) == 1                                        # tone band never moves


def test_reordered_changes_every_repetition_and_keeps_bands_apart():
    for k in range(40):
        tr, buf = build(FROZEN, Condition("rand", "reordered", 40.0), 70.0, (8, k))
        for reps in tr.slots:
            assert all(a != b for a, b in zip(reps, reps[1:]))
            for a, b in zip(reps, reps[1:]):
                assert min(FROZEN.burst_gap_ms(x, y, 40.0) for x, y in zip(a, b)) >= FROZEN.min_gap_ms


def test_order_plan_is_balanced_and_fixed_by_the_seed():
    plan = order_plan(FROZEN, 20261114)
    assert len(plan) == 6                                               # the six frozen tracks
    for trials in plan.values():
        assert len(trials) == 32
        assert all(len({tuple(o) for o in tri}) == 3 for tri in trials)
        cnt = {}
        for tri in trials:
            for o in tri:
                cnt[tuple(o)] = cnt.get(tuple(o), 0) + 1
        assert len(cnt) == 24 and set(cnt.values()) == {4}
    assert order_plan(FROZEN, 20261114) == plan and order_plan(FROZEN, 7) != plan
    tri = plan[2][5]                                                    # a retried trial keeps them
    a0 = build(FROZEN, Condition("co", "frozen", 10.0), 70.0, (1, 6, 2, 5, 0), orders=tri)[0]
    a1 = build(FROZEN, Condition("co", "frozen", 10.0), 70.0, (1, 6, 2, 5, 1), orders=tri)[0]
    assert [r[0] for r in a0.slots] == [r[0] for r in a1.slots] == [list(o) for o in tri]


def test_order_code_round_trip():
    tr, _ = build(FROZEN, Condition("co", "frozen", 20.0), 70.0, (2,))
    code = order_code(tr.slots)
    assert code.count("|") == 2 and "/" not in code                     # frozen: one order each
    back = [[int(v) for v in iv.split(",")] for iv in code.split("|")]
    assert back == [reps[0] for reps in tr.slots]
    tr2, _ = build(FROZEN, Condition("co", "reordered", 40.0), 70.0, (2,))
    assert order_code(tr2.slots).count("/") == 3 * 4                    # five orders per interval


def test_the_seed_audit_passes_and_catches_a_planted_fault(monkeypatch):
    small = FROZEN.replace(trials_per_track=8, conditions=(
        ("co", "sync", 0.0, 1), ("rand", "sync", 0.0, 1), ("co", "frozen", 40.0, 1),
        ("rand", "reordered", 40.0, 1), ("ref", "sync", 0.0, 1)), ratings=(("frozen", 40.0),),
        rating_reps=1)
    rep = audit.audit(small, 99, attempts=(0,))
    assert rep["ok"], rep["problems"]
    assert rep["order_balance"]["co_frozen_40"] == [1]
    import tcmr.audit as A
    real = A.schedule
    monkeypatch.setattr(A, "schedule", lambda cfg, tr: {b: [s + 48 for s in v] for b, v in real(cfg, tr).items()})
    bad = audit.audit(small, 99, attempts=(0,))                          # every burst 1 ms off
    assert not bad["ok"] and any("outside its scheduled bursts" in p for p in bad["problems"])


def test_simulated_plan_c_session_and_analysis(tmp_path):
    from tcmr.analysis import summarize
    w = Writer(tmp_path / "s", {"config": FROZEN.to_dict(), "code": "T1", "session_seed": 20261114})
    drive(Experiment(FROZEN, w, 20261114), SimListener(sim_thresholds(FROZEN)), np.random.default_rng(2))
    rows = list(csv.DictReader(open(tmp_path / "s" / "trials.csv")))
    main = [r for r in rows if r["phase"] == "main"]
    assert len(main) == 352
    assert all(r["orders"] for r in main if r["arrangement"] in ("frozen", "reordered"))
    assert not any(r["orders"] for r in main if r["arrangement"] == "sync")
    s = summarize(tmp_path / "s")
    assert set(s["cmr_db"]) >= {"sync_0", "frozen_10", "frozen_20", "frozen_40", "reordered_40"}
    assert "cmr_drop_frozen_40_db" in s["contrasts_db"]
    assert "learning_cmr_frozen_minus_reordered_40_db" in s["contrasts_db"]


def test_resume_keeps_an_older_log_header(tmp_path):
    folder = tmp_path / "old"
    folder.mkdir()
    old = [f for f in FIELDS if f != "orders"]
    with open(folder / "trials.csv", "w", newline="") as fh:
        csv.writer(fh).writerow(old)
    (folder / "session.json").write_text(json.dumps({"config": FROZEN.to_dict(), "status": "running"}))
    (folder / "trials.jsonl").write_text("")
    w = Writer(folder, {}, resume=True)
    w.trial({"phase": "main", "orders": "0,0,0,0,0", "condition": "x"}, {})
    w.close("complete")
    rows = list(csv.reader(open(folder / "trials.csv")))
    assert rows[0] == old and len(rows[1]) == len(old)                 # columns still line up
