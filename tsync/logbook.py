"""Everything that happens in a session, on disk the moment it happens.

A session folder holds:

* `session.json`  -- who, what, with which code, configuration, plan, calibration, machine and
                     audio device; rewritten atomically (temp file + rename) at every milestone.
* `trials.csv`    -- one flat row per presented trial (demo, practice, catch, aborted included).
* `trials.jsonl`  -- the same trials in full: every tone's onset, level, rove, decoy flag and
                     phase; the buffer's SHA-256; audio timing; keys; system volume.
* `events.jsonl`  -- everything else, timestamped: screens, key presses outside trials, pauses,
                     warnings, volume reads, audio statistics, track summaries.
* `code.patch`    -- the uncommitted changes to the package, if there were any.

Every line is flushed and fsync'ed as it is written, so a crash or a power cut loses at most the
trial in flight, and a session can be resumed from what is on disk.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

SESSION_PREFIX = "tsync_session_"

TRIAL_FIELDS = [
    "row", "phase", "block", "block_condition", "trial_in_block", "staircase_trial", "attempt",
    "status", "abort_reason", "condition", "context_ms", "ending_shift_ms", "is_catch",
    "delta_ms", "sign", "order", "target_interval", "response", "correct", "key",
    "rt_ms", "rt_from_onset2_ms", "n_early_keys", "reminder_shown",
    "step_index", "reversal", "next_delta_ms", "at_ceiling", "at_floor",
    "isi_ms", "async1_ms", "async2_ms", "seed", "sha256", "peak",
    "onset_error_ms", "audio_late", "underflows", "callback_gap_max_ms",
    "visual_lag1_ms", "visual_lag2_ms",
    "volume", "muted", "alert_volume", "device",
    "wall_start", "wall_response", "t_session_s",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def _clean(o):
    """JSON-safe: numpy scalars and arrays to Python, NaN to None, tuples to lists."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(_clean(obj), f, indent=1, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def session_dir(data_dir: Path, code: str, index: int) -> Path:
    return Path(data_dir) / code / f"{SESSION_PREFIX}{index:02d}"


def existing_sessions(data_dir: Path, code: str) -> List[Path]:
    d = Path(data_dir) / code
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if p.is_dir() and p.name.startswith(SESSION_PREFIX))


def next_session_index(data_dir: Path, code: str) -> int:
    ix = [int(p.name[len(SESSION_PREFIX):]) for p in existing_sessions(data_dir, code)
          if p.name[len(SESSION_PREFIX):].isdigit()]
    return max(ix, default=0) + 1


class SessionWriter:
    def __init__(self, folder: Path, meta: dict, resume: bool = False):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.folder / "session.json"
        self.t0 = time.perf_counter()
        if resume:
            self.meta = json.loads(self.meta_path.read_text())
            self.meta.setdefault("resumes", []).append(now_iso())
        else:
            if self.meta_path.exists():
                raise FileExistsError(f"{self.folder} already holds a session")
            self.meta = dict(meta)
            self.meta.setdefault("status", "running")
            self.meta.setdefault("created", now_iso())
        write_json_atomic(self.meta_path, self.meta)
        new_csv = not (self.folder / "trials.csv").exists()
        self._csv_f = open(self.folder / "trials.csv", "a", newline="")
        self._csv = csv.DictWriter(self._csv_f, fieldnames=TRIAL_FIELDS, extrasaction="ignore")
        if new_csv:
            self._csv.writeheader()
            self._sync(self._csv_f)
        self._jsonl = open(self.folder / "trials.jsonl", "a")
        self._events = open(self.folder / "events.jsonl", "a")
        self.n_rows = self._count_rows()
        self.closed = False

    def _count_rows(self) -> int:
        p = self.folder / "trials.jsonl"
        if not p.exists():
            return 0
        with open(p) as f:
            return sum(1 for line in f if line.strip())

    @staticmethod
    def _sync(f) -> None:
        f.flush()
        os.fsync(f.fileno())

    def t(self) -> float:
        return time.perf_counter() - self.t0

    def event(self, event: str, /, **data) -> None:
        if self.closed:
            return
        rec = {"t": now_iso(), "t_perf": time.perf_counter(), "event": event, **data}
        self._events.write(json.dumps(_clean(rec)) + "\n")
        self._sync(self._events)

    def trial(self, row: dict, detail: dict) -> int:
        self.n_rows += 1
        row = dict(row, row=self.n_rows)
        rec = _clean(dict(detail, row=row))
        self._jsonl.write(json.dumps(rec) + "\n")
        self._sync(self._jsonl)
        flat = {k: _fmt(v) for k, v in _clean(row).items()}
        self._csv.writerow(flat)
        self._sync(self._csv_f)
        return self.n_rows

    def update_meta(self, **kv) -> None:
        self.meta.update(_clean(kv))
        write_json_atomic(self.meta_path, self.meta)

    def write_text(self, name: str, text: str) -> None:
        p = self.folder / name
        p.write_text(text)

    def close(self, status: str) -> None:
        if self.closed:
            return
        self.update_meta(status=status, ended=now_iso())
        self.closed = True
        for f in (self._csv_f, self._jsonl, self._events):
            try:
                self._sync(f)
                f.close()
            except (OSError, ValueError):
                pass


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}" if abs(v) < 1e6 else repr(v)
    if isinstance(v, (list, tuple)):
        return " ".join(str(x) for x in v)
    return v


def read_jsonl(path: Path) -> List[dict]:
    out = []
    p = Path(path)
    if not p.exists():
        return out
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break          # a line cut by a crash: everything before it is intact
    return out


def read_session(folder: Path) -> Dict[str, object]:
    folder = Path(folder)
    return {"meta": json.loads((folder / "session.json").read_text()),
            "trials": read_jsonl(folder / "trials.jsonl"),
            "events": read_jsonl(folder / "events.jsonl")}
