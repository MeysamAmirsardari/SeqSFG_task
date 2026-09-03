"""Participants table, session directories, provenance, per-trial logging, resume."""
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

from . import __version__
from .config import Config

PARTICIPANT_FIELDS = ["code", "age", "sex", "handedness", "hearing", "musical_training_years",
                      "headphones", "experimenter", "consent", "created_at"]
TRIAL_FIELDS = ["trial_index", "block", "practice_round", "practice_stage", "variant", "step_ms",
                "target_position", "seed", "response", "correct", "rt_ms", "t_start", "t_response"]


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


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
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()[:16]


def git_info() -> dict:
    root = Path(__file__).resolve().parent.parent
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            return ""
    inside = run("rev-parse", "--is-inside-work-tree") == "true"
    if not inside:
        return {"git": None, "commit": None, "dirty": None, "describe": None}
    return {"git": True, "commit": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain")),
            "describe": run("describe", "--always", "--dirty") or None}


def provenance(cfg: Config) -> dict:
    import numpy, scipy
    try:
        import sounddevice
        sd_ver = sounddevice.__version__
    except Exception:
        sd_ver = None
    try:
        import matplotlib
        mpl_ver = matplotlib.__version__
    except Exception:
        mpl_ver = None
    return {
        "package_version": __version__, "source_hash": source_hash(), **git_info(),
        "host": socket.gethostname(), "platform": platform.platform(), "python": sys.version.split()[0],
        "numpy": numpy.__version__, "scipy": scipy.__version__, "sounddevice": sd_ver, "matplotlib": mpl_ver,
        "start_time": now_iso(), "config_hash": cfg.hash(), "config": cfg.to_dict(),
    }


# ---- session directories -----------------------------------------------------
def session_dir(data_dir: Path, code: str, index: int) -> Path:
    return Path(data_dir) / code / f"session_{index:02d}"


def existing_sessions(data_dir: Path, code: str) -> List[int]:
    p = Path(data_dir) / code
    if not p.exists():
        return []
    out = []
    for q in p.glob("session_*"):
        try:
            out.append(int(q.name.split("_")[1]))
        except ValueError:
            pass
    return sorted(out)


def next_session_index(data_dir: Path, code: str) -> int:
    s = existing_sessions(data_dir, code)
    return (s[-1] + 1) if s else 1


def write_json(path: Path, obj: dict) -> None:
    tmp = Path(path).with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    raise TypeError(f"not serializable: {type(o)}")


def read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---- trial log ---------------------------------------------------------------
class TrialLog:
    """Append-only CSV, flushed and fsynced per row, so an interruption costs nothing."""

    def __init__(self, path: Path):
        self.path = Path(path)
        new = not self.path.exists()
        self.f = open(self.path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=TRIAL_FIELDS)
        if new:
            self.w.writeheader(); self.f.flush(); os.fsync(self.f.fileno())

    def write(self, row: dict) -> None:
        self.w.writerow({k: row.get(k, "") for k in TRIAL_FIELDS})
        self.f.flush(); os.fsync(self.f.fileno())

    def close(self) -> None:
        self.f.close()


def read_trials(path: Path) -> List[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["trial_index"] = int(r["trial_index"]); r["practice_round"] = int(r["practice_round"] or 0)
        r["practice_stage"] = int(r.get("practice_stage") or 0)
        r["step_ms"] = float(r["step_ms"]); r["target_position"] = int(r["target_position"])
        r["seed"] = int(r["seed"]); r["response"] = int(r["response"]) if r["response"] else None
        r["correct"] = int(r["correct"]) if r["correct"] else None
        r["rt_ms"] = float(r["rt_ms"]) if r["rt_ms"] else None
    return rows


class DesignChanged(RuntimeError):
    pass


def check_resumable(session_json: dict, cfg: Config, design: dict) -> None:
    """Refuse to resume when the config or the trial list is not the one the session was started with."""
    problems = []
    if session_json.get("config_hash") != cfg.hash():
        problems.append(f"config hash {session_json.get('config_hash')} -> {cfg.hash()}")
    if session_json.get("design_hash") != design.get("design_hash"):
        problems.append(f"design hash {session_json.get('design_hash')} -> {design.get('design_hash')}")
    if session_json.get("source_hash") != source_hash():
        problems.append(f"source hash {session_json.get('source_hash')} -> {source_hash()} (code changed)")
    if problems:
        raise DesignChanged("cannot resume, the design has changed since the session started:\n  - " +
                            "\n  - ".join(problems))
