"""Regenerate, from a session's log, exactly the stimuli the listener heard.

For model observers (the RNN, the coherence model, an ideal observer) the comparison that
matters is trial by trial: the same two intervals, the same roves, the same phases, against the
same response. Every main, catch and filler trial is rebuilt from the session seed and its slot,
and the rebuilt gains and target interval are checked against what the runner logged before
anything is written. A mismatch means the code doing the export is not the code that ran the
session, and the export stops rather than hand a model the wrong sounds.

Output: `trials.npz`, one row per trial (condition, step, type, increment, target interval,
response, both gain matrices, phases), and with --wav one stereo WAV per trial.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from tcoh.session import read_json, read_trials

from .config import Config, catch_condition, conditions
from .stimulus import Trial, build_trial, gains_from_json, render_trial, to_output


class ExportMismatch(RuntimeError):
    pass


def rebuild(cfg: Config, meta: dict, row: dict) -> Trial:
    """The trial a logged main, catch or filler row was built from."""
    from .runner import Runner
    key = [int(meta["session_seed"]), int(meta.get("seed", 0)), Runner.KEY,
           int(float(row["slot_index"]))]
    if row["phase"] == "filler":
        key.append(1)
    conds = {c.name: c for c in conditions(cfg)}
    cond = catch_condition(cfg) if row["phase"] == "catch" else conds[row["condition"]]
    return build_trial(cfg, cond, float(row["srs_db"]), np.random.default_rng(key),
                       is_catch=row["phase"] == "catch")


def export_session(sdir: Path, out: Path, wav: bool = False) -> List[Path]:
    sdir, out = Path(sdir), Path(out)
    meta = read_json(sdir / "session.json")
    cfg = Config.from_dict(meta["config"])
    rows = [r for r in read_trials(sdir / "trials.csv")
            if r.get("phase") in ("main", "catch", "filler")]
    out.mkdir(parents=True, exist_ok=True)
    keep, written = [], []
    for r in rows:
        tr = rebuild(cfg, meta, r)
        g1, g2 = gains_from_json(r["gains_1"]), gains_from_json(r["gains_2"])
        if (tr.target_position != int(float(r["target_position"]))
                or not np.allclose(tr.first.gains_db, g1, atol=1e-3, equal_nan=True)
                or not np.allclose(tr.second.gains_db, g2, atol=1e-3, equal_nan=True)):
            raise ExportMismatch(f"trial {r['trial_index']} ({r['condition']}): the rebuilt "
                                 "stimulus does not match the log, so this code did not "
                                 "generate this session")
        keep.append((r, tr))
        if wav:
            import soundfile as sf
            p = out / f"trial_{int(float(r['trial_index'])):04d}_{r['condition']}.wav"
            sf.write(p, to_output(cfg, render_trial(cfg, tr)), cfg.sample_rate)
            written.append(p)
    resp = [int(r["response"]) if str(r["response"]).strip() in ("1", "2") else 0
            for r, _ in keep]
    p = out / "trials.npz"
    np.savez_compressed(
        p, trial_index=np.array([int(float(r["trial_index"])) for r, _ in keep]),
        phase=np.array([r["phase"] for r, _ in keep]),
        condition=np.array([r["condition"] for r, _ in keep]),
        task=np.array([t.task for _, t in keep]), kind=np.array([t.kind for _, t in keep]),
        step_ms=np.array([t.step_ms for _, t in keep]),
        srs_db=np.array([t.srs_db for _, t in keep]),
        delta_db=np.array([t.delta_db for _, t in keep]),
        target_position=np.array([t.target_position for _, t in keep]),
        response=np.array(resp),
        gains=np.stack([np.stack([t.first.gains_db, t.second.gains_db]) for _, t in keep]),
        phases=np.stack([t.phases for _, t in keep]),
        captor_phases=np.stack([t.captor_phases for _, t in keep]),
        freqs_hz=np.array(cfg.freqs_hz), sample_rate=cfg.sample_rate,
        config_hash=np.array(cfg.hash()))
    return [p] + written
