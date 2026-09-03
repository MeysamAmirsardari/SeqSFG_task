"""The experiment: participant panel, calibration, practice with a criterion, main and
control blocks with self-paced breaks, per-trial logging and resume.

The condition of a trial is never displayed. Keys: 1, 2 (response), space (continue), q (quit and save).
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from .config import Config, validate
from .design import TrialSpec, make_design, specs_from
from .session import (DesignChanged, TrialLog, check_resumable, existing_sessions, load_participants,
                      next_session_index, now_iso, provenance, read_json, read_trials, session_dir,
                      upsert_participant, write_json, PARTICIPANT_FIELDS)
from .stimulus import make_trial, render_trial

INSTRUCTIONS = """
On every trial you will hear two sounds, one after the other. Each is a busy cloud of
short tones.

  In ONE of the two sounds, a small group of tones keeps coming back at the SAME pitches.
  In the OTHER, that does not happen: either the group comes back at NEW pitches every
  time, or there is no group at all. Both kinds of trial occur, mixed together.

Your job is the same either way: which sound kept coming back at the same pitches?
Press 1 or 2. Guess if you are not sure. Some trials are very hard; that is expected.
"""

PRACTICE_INTRO = {
    "ungrouped": "First practice: in one sound a group of tones arrives together again and again, and in the\n"
                 "other there is no group at all. Listen for the group.",
    "rising": "Second practice: now BOTH sounds contain a group. In one it returns at the same pitches\n"
              "every time; in the other it lands on new pitches. Listen for the one that repeats itself.",
}


class QuitRequested(Exception):
    pass


# ---- input / output helpers --------------------------------------------------
def getkey(valid: set, prompt: str = "") -> str:
    if prompt:
        print(prompt, end="", flush=True)
    while True:
        ch = _read_char()
        if ch in valid:
            print(ch if ch != " " else "")
            return ch


def _read_char() -> str:
    if not sys.stdin.isatty():
        s = sys.stdin.readline()
        if not s:
            raise QuitRequested()
        return s.strip()[:1] or " "
    try:
        import termios, tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch
    except ImportError:
        import msvcrt
        return msvcrt.getwch()


def ask(prompt: str, default: str = "", validator: Optional[Callable[[str], bool]] = None) -> str:
    while True:
        s = input(f"{prompt}{' [' + default + ']' if default else ''}: ").strip() or default
        if validator is None or validator(s):
            return s
        print("  invalid, try again")


class Audio:
    def __init__(self, sample_rate: int, device=None, enabled: bool = True):
        self.sr, self.device, self.enabled = sample_rate, device, enabled
        if enabled:
            import sounddevice as sd
            self.sd = sd
            if device is not None:
                sd.default.device = device

    def play(self, x: np.ndarray, wait_if_silent: bool = True) -> None:
        if self.enabled:
            self.sd.play(x.astype(np.float32), self.sr, blocking=True)
        elif wait_if_silent:
            time.sleep(len(x) / self.sr)


# ---- the runner ----------------------------------------------------------------
class Runner:
    def __init__(self, cfg: Config, data_dir: Path, device=None, audio: bool = True,
                 auto: Optional[float] = None, fast: bool = False):
        """auto: if not None, simulate a listener whose accuracy decays with the step (for pipeline tests)."""
        self.cfg, self.d = cfg, validate(cfg)
        self.data_dir = Path(data_dir)
        self.audio = Audio(cfg.sample_rate, device, enabled=audio and auto is None)
        self.auto, self.fast = auto, fast
        self.rng_auto = np.random.default_rng(7)

    # -- session setup ------------------------------------------------------------
    def start(self, code: Optional[str] = None, resume: bool = False, session_index: Optional[int] = None) -> Path:
        cfg = self.cfg
        if self.auto is not None:
            code = code or "AUTO"
            row = dict(code=code, age="0", sex="na", handedness="na", hearing="simulated", musical_training_years="0",
                       headphones="none", experimenter="auto", consent="yes")
        else:
            row = self.panel(code)
            code = row["code"]
        upsert_participant(self.data_dir, row)
        if resume:
            idx = session_index or (existing_sessions(self.data_dir, code) or [None])[-1]
            if idx is None:
                raise RuntimeError(f"no session to resume for {code}")
            sdir = session_dir(self.data_dir, code, idx)
            meta = read_json(sdir / "session.json")
            design = make_design(cfg, code, idx)
            check_resumable(meta, cfg, design)
            print(f"resuming {sdir} (design {design['design_hash']})")
        else:
            idx = session_index or next_session_index(self.data_dir, code)
            sdir = session_dir(self.data_dir, code, idx)
            if sdir.exists():
                raise RuntimeError(f"{sdir} exists; use --resume or choose another session index")
            sdir.mkdir(parents=True)
            design = make_design(cfg, code, idx)
            meta = {**provenance(cfg), "participant_code": code, "session_index": idx, "session_seed": design["session_seed"],
                    "design_hash": design["design_hash"], "design": design, "status": "started",
                    "calibration": None, "practice": []}
            write_json(sdir / "session.json", meta)
            print(f"new session {sdir} (seed {design['session_seed']}, design {design['design_hash']})")
        self.sdir, self.meta, self.design = sdir, meta, design
        self.log = TrialLog(sdir / "trials.csv")
        self.done = read_trials(sdir / "trials.csv")
        return sdir

    def panel(self, code: Optional[str]) -> dict:
        known = load_participants(self.data_dir)
        print("\n--- participant panel ---")
        code = code or ask("participant code", validator=lambda s: s.isalnum())
        if code in known:
            print("  existing participant:", {k: v for k, v in known[code].items() if k != "created_at"})
            if getkey({"y", "n"}, "  use this record? [y/n] ") == "y":
                row = dict(known[code])
                row["experimenter"] = ask("experimenter", row.get("experimenter", ""))
                row["headphones"] = ask("headphone model", row.get("headphones", ""))
                row["consent"] = "yes" if getkey({"y", "n"}, "  consent confirmed for this session? [y/n] ") == "y" else "no"
                if row["consent"] != "yes":
                    raise SystemExit("consent not confirmed; stopping")
                return row
        row = dict(code=code)
        row["age"] = ask("age", validator=lambda s: s.isdigit())
        row["sex"] = ask("sex (as the participant chooses to report)")
        row["handedness"] = ask("handedness", "right")
        row["hearing"] = ask("self-reported hearing (normal / other)", "normal")
        row["musical_training_years"] = ask("years of musical training", "0", validator=lambda s: s.replace(".", "").isdigit())
        row["headphones"] = ask("headphone model")
        row["experimenter"] = ask("experimenter")
        row["consent"] = "yes" if getkey({"y", "n"}, "  consent confirmed? [y/n] ") == "y" else "no"
        if row["consent"] != "yes":
            raise SystemExit("consent not confirmed; stopping")
        return row

    # -- calibration ---------------------------------------------------------------
    def calibrate(self) -> None:
        cfg = self.cfg
        if self.meta.get("calibration"):
            print("calibration already recorded for this session:", self.meta["calibration"])
            return
        if self.auto is not None:
            self.meta["calibration"] = {"measured_db_spl": cfg.tone_level_db_spl, "note": "auto", "time": now_iso()}
            write_json(self.sdir / "session.json", self.meta)
            return
        print(f"\n--- calibration ---\nA {cfg.calibration_freq_hz:.0f} Hz tone at the amplitude of ONE stimulus tone "
              f"({cfg.tone_amplitude} FS peak, {20 * math.log10(cfg.tone_amplitude / math.sqrt(2)):.1f} dB FS rms) will play for "
              f"{cfg.calibration_dur_s:.0f} s.\nSet the system so it reads {cfg.tone_level_db_spl:.0f} dB SPL at the ear "
              f"(coupler or in-ear probe). The full stimulus is about "
              f"{10 * math.log10(max(self.d.mean_simultaneous, 1)):.0f} dB above that.")
        while True:
            print("  space = play, then enter the measured level; q = skip calibration (recorded as not calibrated)")
            k = getkey({" ", "q"})
            if k == "q":
                self.meta["calibration"] = {"measured_db_spl": None, "note": "skipped", "time": now_iso()}
                break
            n = int(cfg.calibration_dur_s * cfg.sample_rate)
            t = np.arange(n) / cfg.sample_rate
            tone = cfg.tone_amplitude * np.sin(2 * np.pi * cfg.calibration_freq_hz * t)
            r = int(0.02 * cfg.sample_rate); ramp = 0.5 * (1 - np.cos(np.pi * np.arange(r) / r))
            tone[:r] *= ramp; tone[-r:] *= ramp[::-1]
            self.audio.play(tone)
            s = ask("  measured dB SPL (blank = play again)")
            if s:
                self.meta["calibration"] = {"measured_db_spl": float(s), "target_db_spl": cfg.tone_level_db_spl,
                                            "note": "reference tone", "time": now_iso()}
                break
        write_json(self.sdir / "session.json", self.meta)

    # -- one trial -------------------------------------------------------------------
    def run_trial(self, spec: TrialSpec, feedback: bool, shown_index: int, shown_total: int) -> int:
        cfg = self.cfg
        trial = make_trial(cfg, spec.seed, spec.step_ms, spec.variant, d=self.d)
        x = render_trial(cfg, trial, spec.target_position, self.d)
        print(f"\nTrial {shown_index} of {shown_total}.  Listen ...", flush=True)
        t_start = now_iso()
        if self.auto is None:
            self.audio.play(x)
            t0 = time.perf_counter()
            k = getkey({"1", "2", "q"}, "Which sound kept coming back at the same pitches? [1/2]  ")
            rt = (time.perf_counter() - t0) * 1000.0
            if k == "q":
                raise QuitRequested()
            resp = int(k)
        else:
            if not self.fast:
                time.sleep(0.01)
            if spec.variant == "ungrouped":
                acc = 0.5 + (0.5 - 0.02) * math.exp(-spec.step_ms / (2.5 * self.auto))
            elif spec.variant == "onechannel":
                acc = 0.5
            else:
                acc = 0.5 + (0.5 - 0.02) * math.exp(-spec.step_ms / self.auto)
            resp = spec.target_position if self.rng_auto.random() < acc else 3 - spec.target_position
            rt = 500.0
        correct = int(resp == spec.target_position)
        if feedback:
            print("  correct" if correct else "  wrong")
            if self.auto is None:
                time.sleep(cfg.feedback_s)
        self.log.write(dict(trial_index=spec.index, block=spec.block, practice_round=spec.practice_round,
                            practice_stage=spec.practice_stage,
                            variant=spec.variant, step_ms=spec.step_ms, target_position=spec.target_position,
                            seed=spec.seed, response=resp, correct=correct, rt_ms=round(rt, 1),
                            t_start=t_start, t_response=now_iso()))
        if self.auto is None:
            time.sleep(cfg.iti_s)
        return correct

    def pause(self, msg: str) -> None:
        if self.auto is not None:
            return
        print(f"\n{msg}\nPress space when ready to continue.")
        getkey({" "})

    # -- blocks -----------------------------------------------------------------------
    def _done_in(self, block: str) -> List[dict]:
        return [t for t in self.done if t["block"] == block]

    def _done_practice(self, stage: int, rnd: int) -> List[dict]:
        return [t for t in self.done if t["block"] == "practice"
                and t.get("practice_stage", 1) == stage and t["practice_round"] == rnd]

    def practice(self) -> bool:
        """Each stage in turn; a stage is passed by meeting the criterion within its allowed rounds."""
        cfg = self.cfg
        for si, (variant, step) in enumerate(cfg.practice_cells, start=1):
            passed = False
            for rnd in range(1, cfg.practice_max_rounds + 1):
                specs = specs_from(self.design, "practice", si, rnd)
                done = self._done_practice(si, rnd)
                if len(done) >= len(specs):
                    n_ok = sum(t["correct"] for t in done)
                    print(f"practice stage {si} round {rnd} already run ({n_ok}/{len(specs)})")
                    if n_ok >= cfg.practice_criterion:
                        passed = True
                        break
                    continue
                if si == 1 and rnd == 1 and not done:
                    print(INSTRUCTIONS)
                intro = PRACTICE_INTRO.get(variant, "")
                head = f"Practice stage {si} of {len(cfg.practice_cells)}"
                if rnd > 1:
                    head += f", attempt {rnd} of {cfg.practice_max_rounds}"
                self.pause(f"{head}: {len(specs)} trials with feedback.\n{intro}")
                n_ok = sum(t["correct"] for t in done)
                for spec in specs[len(done):]:
                    n_ok += self.run_trial(spec, True, spec.index + 1, len(specs))
                self.meta.setdefault("practice", []).append(
                    {"stage": si, "variant": variant, "step_ms": step, "round": rnd, "correct": n_ok,
                     "n": len(specs), "passed": n_ok >= cfg.practice_criterion})
                write_json(self.sdir / "session.json", self.meta)
                print(f"\npractice stage {si} ({variant} {step:g} ms) round {rnd}: {n_ok}/{len(specs)} correct "
                      f"(criterion {cfg.practice_criterion})")
                if n_ok >= cfg.practice_criterion:
                    passed = True
                    break
            if not passed:
                self.meta["practice_failed_stage"] = {"stage": si, "variant": variant, "step_ms": step}
                write_json(self.sdir / "session.json", self.meta)
                return False
        return True

    def block(self, name: str, feedback: bool) -> None:
        cfg = self.cfg
        specs = specs_from(self.design, name)
        done = self._done_in(name)
        if len(done) >= len(specs):
            print(f"{name} block already complete")
            return
        self.pause(f"{name.capitalize()} block: {len(specs)} trials" + (" with feedback." if feedback else "."))
        for spec in specs[len(done):]:
            if spec.index > 0 and spec.index % cfg.break_every == 0:
                self.pause("Take a break.")
            self.run_trial(spec, feedback, spec.index + 1, len(specs))

    def set_status(self, status: str) -> None:
        self.meta["status"] = status
        self.meta["last_update"] = now_iso()
        write_json(self.sdir / "session.json", self.meta)

    # -- whole session --------------------------------------------------------------------
    def run(self, code: Optional[str] = None, resume: bool = False, session_index: Optional[int] = None) -> Path:
        cfg = self.cfg
        sdir = self.start(code, resume, session_index)
        try:
            self.calibrate()
            if not self.practice():
                self.set_status("practice_criterion_not_met")
                failed = self.meta.get("practice_failed_stage", {})
                where = (f"stage {failed.get('stage')} ({failed.get('variant')} "
                         f"{failed.get('step_ms', 0):g} ms)") if failed else "practice"
                print(f"\nPractice criterion ({cfg.practice_criterion}/{cfg.practice_n}) not met at {where} "
                      f"after {cfg.practice_max_rounds} rounds. The session stops here and is recorded as such.")
                return sdir
            self.set_status("main")
            self.block("main", cfg.feedback_main)
            self.set_status("control")
            self.block("control", cfg.feedback_control)
            self.set_status("complete")
            print("\nSession complete. Thank you.")
        except QuitRequested:
            self.set_status("interrupted")
            print("\nstopped; every answered trial is saved. Resume with --resume.")
        finally:
            self.log.close()
        return sdir
