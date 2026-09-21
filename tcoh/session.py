"""Participants table, session directories, provenance, per-trial logging, resume.

Self-contained on purpose: this task shares no stimulus code, no trial schema and no analysis
with the stochastic figure-ground experiments in `seqsfg`, and a shared logging layer would
couple two things that have no reason to change together.

The per-trial schema records the whole adaptive state, not just the response. A threshold is a
summary of a trajectory, and a trajectory that cannot be replayed from the log cannot be
audited: `delta_ms`, `step_index`, `reversal` and `track_trial_index` are enough to recompute
every threshold in the session from `trials.csv` alone, which is what `tcoh-analyze --replay`
does as a check on the runner.

Participant data never leaves the machine and is never committed; the repository's .gitignore
excludes the data directory, and nothing here writes outside it.
"""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .config import Config

PARTICIPANT_FIELDS = ["code", "age", "sex", "handedness", "hearing", "musical_training_years",
                      "headphones", "experimenter", "consent", "created_at"]

TRIAL_FIELDS = [
    "trial_index", "phase", "slot_index", "block", "track_id", "track_trial_index",
    "condition", "lag_pct", "a_kind", "reference", "n_precursor",
    "is_catch", "delta_ms", "delta_signed_ms", "delta_realised_ms", "direction",
    "step_index", "reversal", "at_ceiling", "at_floor",
    "target_position", "response", "correct", "rt_ms",
    "rove_db_1", "rove_db_2", "feedback", "timed_out", "t_start", "t_response",
]
# `timed_out` was added on 2026-09-21 with the response timeout. Files written before that
# date do not have the column; every reader here goes through `_f`/`_i`, which default a
# missing field, so old sessions load unchanged.


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def write_json(p: Path, obj) -> None:
    tmp = Path(p).with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, p)


# ---- participants ------------------------------------------------------------
def participants_path(data_dir: Path) -> Path:
    return Path(data_dir) / "participants.csv"


def load_participants(data_dir: Path) -> Dict[str, dict]:
    p = participants_path(data_dir)
    if not p.exists():
        return {}
    with open(p, newline="") as f:
        return {r["code"]: r for r in csv.DictReader(f)}


def upsert_participant(data_dir: Path, row: dict) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = load_participants(data_dir)
    row = {k: row.get(k, "") for k in PARTICIPANT_FIELDS}
    if not row["created_at"]:
        row["created_at"] = now_iso()
    rows[row["code"]] = row
    tmp = participants_path(data_dir).with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PARTICIPANT_FIELDS)
        w.writeheader()
        for r in rows.values():
            w.writerow(r)
    os.replace(tmp, participants_path(data_dir))


# ---- provenance --------------------------------------------------------------
def source_hash() -> str:
    pkg = Path(__file__).parent
    h = hashlib.sha256()
    for p in sorted(pkg.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def git_info() -> dict:
    root = Path(__file__).resolve().parent.parent

    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except Exception:
            return ""
    if run("rev-parse", "--is-inside-work-tree") != "true":
        return {"git": None, "commit": None, "dirty": None}
    return {"git": True, "commit": run("rev-parse", "HEAD") or None,
            "dirty": bool(run("status", "--porcelain"))}


def provenance(cfg: Config) -> dict:
    import numpy
    import scipy
    try:
        import sounddevice
        sd_ver = sounddevice.__version__
    except Exception:
        sd_ver = None
    return {"task": "tcoh", "source_hash": source_hash(), **git_info(),
            "host": socket.gethostname(), "platform": platform.platform(),
            "python": sys.version.split()[0], "numpy": numpy.__version__,
            "scipy": scipy.__version__, "sounddevice": sd_ver,
            "start_time": now_iso(), "config_hash": cfg.hash(), "config": cfg.to_dict()}


# ---- session directories -----------------------------------------------------
def session_dir(data_dir: Path, code: str, index: int) -> Path:
    return Path(data_dir) / code / f"tcoh_session_{index:02d}"


def existing_sessions(data_dir: Path, code: str) -> List[int]:
    p = Path(data_dir) / code
    if not p.exists():
        return []
    out = []
    for q in p.glob("tcoh_session_*"):
        try:
            out.append(int(q.name.rsplit("_", 1)[1]))
        except ValueError:
            pass
    return sorted(out)


def next_session_index(data_dir: Path, code: str) -> int:
    return (existing_sessions(data_dir, code) or [0])[-1] + 1


class DesignChanged(RuntimeError):
    pass


def check_resumable(meta: dict, cfg: Config, design: dict) -> None:
    if meta.get("config_hash") != cfg.hash():
        raise DesignChanged(
            f"this session was recorded under config {meta.get('config_hash')} and you are "
            f"resuming with {cfg.hash()}. Resuming would mix two experiments in one file.")
    if meta.get("design_hash") != design["design_hash"]:
        raise DesignChanged("the trial order for this participant and session no longer matches "
                            "the one recorded; refusing to resume.")


# ---- trial log ---------------------------------------------------------------
class TrialLog:
    """Append-only CSV, flushed every row, so a crashed session loses nothing."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.new = not self.path.exists()
        self.f = open(self.path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=TRIAL_FIELDS, extrasaction="ignore")
        if self.new:
            self.w.writeheader()
            self.f.flush()

    def write(self, row: dict) -> None:
        self.w.writerow({k: row.get(k, "") for k in TRIAL_FIELDS})
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass


def read_trials(path: Path) -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))
