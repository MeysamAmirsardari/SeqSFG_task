"""tcmr_exp1 (2026-11-15): the ARO study and Experiment 1 of the JASA paper.

tcmr_frozen with a 5-ms step in place of the re-ordered control, no rating block, and a practice
gate (at least 6 of the 8 practice trials right, or the session stops; tests/test_tcmr_gate.py)."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tcmr import audit
from tcmr.cli import drive, main, next_seed, seed_users, sim_thresholds
from tcmr.config import Config, validate
from tcmr.procedure import SimListener
from tcmr.session import Experiment, Writer

CONFIGS = Path(__file__).resolve().parents[1] / "tcmr" / "configs"
EXP1 = validate(Config.load(CONFIGS / "tcmr_exp1.json"))
FROZEN = Config.load(CONFIGS / "tcmr_frozen.json")


def test_exp1_is_the_pilot_with_a_5ms_step_for_the_control_and_no_ratings():
    # frozen for ARO and JASA. The hash was 25ac4a4d62854cc1 until the practice gate was added on
    # 2026-11-15, before any listener; the sounds are the same (the next test pins one).
    assert EXP1.hash() == "a5099394bbce529d"
    assert FROZEN.hash() == "1c086aee8b19081e"     # the pilot P00 ran is untouched
    assert [c.name for c in EXP1.condition_list()] == [
        "co_sync_0", "rand_sync_0", "co_frozen_5", "rand_frozen_5", "co_frozen_10",
        "rand_frozen_10", "co_frozen_20", "rand_frozen_20", "co_frozen_40", "rand_frozen_40",
        "ref_sync_0"]
    assert EXP1.ratings == () and EXP1.rating_reps == 0
    a, b = EXP1.to_dict(), FROZEN.to_dict()
    assert {k for k in a if a[k] != b[k]} == {"name", "conditions", "ratings", "rating_reps",
                                              "practice_min_correct"}
    assert EXP1.practice_min_correct == 6


def test_the_gate_changed_the_hash_of_exp1_but_not_one_sample_of_it():
    from tcmr.stimulus import build, sha256
    before = EXP1.replace(practice_min_correct=0)          # exp1 as it was checked before the gate
    assert before.hash() == "25ac4a4d62854cc1"
    for c in (EXP1.condition_list()[0], EXP1.condition_list()[9]):
        a = build(EXP1, c, 70.0, (20261201, 6, 0, 0, 0))[1]
        assert sha256(a) == sha256(build(before, c, 70.0, (20261201, 6, 0, 0, 0))[1])


def test_exp1_session_has_no_rating_block_and_the_analysis_covers_every_step(tmp_path):
    from tcmr.analysis import summarize
    w = Writer(tmp_path / "s", {"config": EXP1.to_dict(), "code": "T1", "session_seed": 5})
    drive(Experiment(EXP1, w, 5), SimListener(sim_thresholds(EXP1)), np.random.default_rng(5))
    rows = list(csv.DictReader(open(tmp_path / "s" / "trials.csv")))
    assert {r["phase"] for r in rows} == {"example", "practice", "main"}
    assert sum(r["phase"] == "main" for r in rows) == 352
    assert json.loads((tmp_path / "s" / "session.json").read_text())["status"] == "complete"
    s = summarize(tmp_path / "s")
    assert set(s["cmr_db"]) == {"sync_0", "frozen_5", "frozen_10", "frozen_20", "frozen_40"}
    assert {f"cmr_drop_frozen_{d}_db" for d in (5, 10, 20, 40)} <= set(s["contrasts_db"])
    assert not any(k.startswith("learning") for k in s["contrasts_db"])
    assert set(s["cmr_vs_band_alone_db"]) == {c.name for c in EXP1.condition_list()
                                              if c.envelope != "ref"}


def test_several_seeds_are_checked_in_parallel_and_each_is_handed_out_once(tmp_path):
    tiny = EXP1.replace(name="tiny", trials_per_track=8, conditions=(
        ("co", "sync", 0.0, 1), ("rand", "sync", 0.0, 1), ("co", "frozen", 5.0, 1),
        ("ref", "sync", 0.0, 1)))
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps(tiny.to_dict()))
    data = tmp_path / "data"
    with pytest.raises(SystemExit) as e:
        main(["check", "--config", str(path), "--seed", "11", "12", "--data", str(data)])
    assert e.value.code == 0
    for s in (11, 12):
        rep = json.loads(audit.report_path(data, tiny, s).read_text())
        assert rep["ok"] and rep["seed"] == s and rep["amplitude"] == "uncalibrated"
    assert next_seed(data, tiny, "uncalibrated") == 11
    assert next_seed(data, tiny, "calibrated") is None              # checked without calibration
    used = data / "A01" / "tcmr_session_01"
    used.mkdir(parents=True)
    (used / "session.json").write_text(json.dumps(
        {"code": "A01", "session_seed": 11, "config_hash": tiny.hash()}))
    assert seed_users(data, tiny, 11) == [("A01", used)]
    assert next_seed(data, tiny, "uncalibrated") == 12
    assert seed_users(data, EXP1, 11) == []                          # other presets keep their own
