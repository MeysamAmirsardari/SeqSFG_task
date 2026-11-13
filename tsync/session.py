"""Opening, resuming and describing a session folder (shared by the app and the simulator)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

from .config import Config
from .design import Plan, clean_code, make_plan, participant_number
from .experiment import ResumeState
from .levels import Calibration
from .logbook import SessionWriter, next_session_index, session_dir
from .system import git_info, machine, source_hash, versions


class ResumeRefused(RuntimeError):
    pass


def open_session(cfg: Config, data_dir: Path, code: str, cal: Calibration, *,
                 simulated: bool = False, session_seed: Optional[int] = None,
                 order_row: Optional[int] = None, extra: Optional[dict] = None
                 ) -> Tuple[SessionWriter, Plan]:
    code = clean_code(code)
    index = next_session_index(data_dir, code)
    folder = session_dir(data_dir, code, index)
    plan = make_plan(cfg, participant_number(code), session_seed, order_row)
    git = git_info()
    meta = {
        "task": "tsync", "code": code, "session_index": index, "simulated": simulated,
        "config_name": cfg.name, "config_hash": cfg.hash(), "config": cfg.to_dict(),
        "plan": plan.to_dict(), "calibration": cal.to_dict(),
        "levels_calibrated": cal.calibrated,
        "source_sha256": source_hash(),
        "git": {k: v for k, v in git.items() if k != "diff"},
        "versions": versions(), "machine": machine(),
    }
    meta.update(extra or {})
    w = SessionWriter(folder, meta)
    if git.get("diff"):
        w.write_text("code.patch", git["diff"])
    w.event("session_open", folder=str(folder), code=code, index=index)
    return w, plan


def resume_session(folder: Path, cfg: Config) -> Tuple[SessionWriter, Plan, Calibration,
                                                       ResumeState]:
    folder = Path(folder)
    meta = json.loads((folder / "session.json").read_text())
    if meta.get("config_hash") != cfg.hash():
        raise ResumeRefused(f"{folder} was run with configuration {meta.get('config_hash')}, "
                            f"this one is {cfg.hash()}: a session cannot change design midway")
    if meta.get("status") == "complete":
        raise ResumeRefused(f"{folder} is already complete")
    plan = Plan.from_dict(meta["plan"])
    cal = Calibration.from_dict(meta["calibration"])
    resumes = len(meta.get("resumes", [])) + 1
    w = SessionWriter(folder, {}, resume=True)
    w.update_meta(status="running", source_sha256_resume=source_hash())
    state = ResumeState.from_folder(folder, resumes)
    w.event("session_resume", resumes=resumes, main_started=state.main_started,
            blocks_with_data=sorted(state.responses))
    return w, plan, cal, state
