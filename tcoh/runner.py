"""Running a session: panel, calibration, familiarisation, practice to criterion, interleaved
adaptive tracks, breaks, per-trial logging, resume.

Keys: 1 and 2 to respond, space to continue, q to quit and save.

What the listener is asked
---------------------------
"Two sounds, one after the other. They are the same except that in ONE of them the very last
high tone is slightly out of place. Which one?" That is the whole task, and it is the same
sentence in every condition -- the instruction never mentions streams, grouping, or the low
tone, because a listener told to attend to the relationship between the two tones would be
performing a different experiment from one told to listen for an irregularity, and the whole
point is to find out which one they do spontaneously.

Feedback is ON by default, which is the opposite of the choice made for the yes/no tasks in
this repository, and for a reason. Those measure d' and a criterion, and feedback drives the
criterion to wherever the feedback wants it. Two-interval forced choice has no criterion to
distort: the measure is a threshold, feedback keeps a listener calibrated and on task through
a long session, and it is standard for adaptive 2AFC. It is still recorded per trial.

Catch trials
------------
Interleaved at `catch_rate`, always at `catch_delta_ms`, and they never update the track that
hosts them -- a free correct answer at 45 ms would drag the track down and the threshold with
it. They are the only measure of whether the listener was still awake, and `analyse` refuses
to report a session that fails them.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import Condition, Config, validate
from .design import make_design
from .observer import SimulatedListener
from .session import (DesignChanged, TrialLog, check_resumable, existing_sessions,
                      load_participants, next_session_index, now_iso, provenance, read_json,
                      read_trials, session_dir, upsert_participant, write_json)
from .audiolevel import (amplitude_for, describe_output, drift, scene_db_spl,
                         system_output)
from .stimulus import build_trial, render_trial, to_output
from .track import Track

INSTRUCTIONS = """
On every trial you will hear TWO sounds, one after the other, with a short gap between them.

Each sound is a low tone and a high tone, repeating together in a steady pattern.

The two sounds are the same except for one thing: in ONE of them, the very LAST high tone is
slightly out of place in time. It may be a little early or a little late.

  Your job: which sound had the last high tone out of place -- the first or the second?

Press 1 or 2. Guess if you are not sure; you often will be. Some trials are very easy and some
are close to impossible, and that is how it is supposed to be.
"""

FAMILIARISE = """
--- familiarisation ---
First, two examples with the shift made very large, so you know what to listen for.
You will be told the answer each time.
"""


class QuitRequested(Exception):
    pass


# ---- terminal i/o ------------------------------------------------------------
def _read_char() -> str:
    if not sys.stdin.isatty():
        s = sys.stdin.readline()
        if not s:
            raise QuitRequested()
        return s.strip()[:1] or " "
    try:
        import termios
        import tty
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


def getkey(valid: set, prompt: str = "") -> str:
    if prompt:
        print(prompt, end="", flush=True)
    while True:
        ch = _read_char()
        if ch in valid:
            print(ch if ch != " " else "")
            return ch


def ask(prompt: str, default: str = "", validator=None) -> str:
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


# ---- the runner ---------------------------------------------------------------
class Runner:
    RT_REFERENCE = "end of the second interval (audio playback returns)"

    def __init__(self, cfg: Config, data_dir: Path, device=None, audio: bool = True,
                 auto: Optional[str] = None, fast: bool = False, seed: int = 0):
        """`auto` names a mode from tcoh.observer and runs the session with no human at all."""
        self.cfg, self.d = cfg, validate(cfg)
        self.data_dir = Path(data_dir)
        self.auto = auto
        self.audio = Audio(cfg.sample_rate, device, enabled=audio and auto is None)
        self.fast = fast
        self.sim = SimulatedListener(cfg, mode=auto, seed=seed or 20260913) if auto else None
        self.cond = {c.name: c for c in self.d.conditions}
        self.catch_cond = next((c for c in self.d.conditions
                                if c.a_kind == "coherent" and cfg.catch_at_pct is not None
                                and abs(c.lag_pct - cfg.catch_at_pct) < 1e-9), None)
        self.seed = int(seed)
        # replaced in start() by a stream keyed on the session. Seeding the per-trial draws --
        # which interval holds the target, which way the tone moves, the starting phases -- from
        # a constant gave every participant the identical sequence of target positions. Balanced,
        # and no use to a listener, but any accidental structure would then be shared by everyone
        # rather than averaging out across the sample.
        self.rng = np.random.default_rng([self.seed, 0x7C00])

    # -- setup -------------------------------------------------------------------
    def start(self, code: Optional[str] = None, resume: bool = False,
              session_index: Optional[int] = None) -> Path:
        cfg = self.cfg
        if self.auto:
            code = code or "AUTO"
            row = dict(code=code, age="0", sex="na", handedness="na", hearing="simulated",
                       musical_training_years="0", headphones="none", experimenter="auto",
                       consent="yes")
        else:
            row = self.panel(code)
            code = row["code"]
        upsert_participant(self.data_dir, row)

        if resume:
            idx = session_index or (existing_sessions(self.data_dir, code) or [None])[-1]
            if idx is None:
                raise RuntimeError(f"no tcoh session to resume for {code}")
            sdir = session_dir(self.data_dir, code, idx)
            meta = read_json(sdir / "session.json")
            design = make_design(cfg, code, idx)
            check_resumable(meta, cfg, design)
            print(f"resuming {sdir} (design {design['design_hash']})")
        else:
            idx = session_index or next_session_index(self.data_dir, code)
            sdir = session_dir(self.data_dir, code, idx)
            if sdir.exists():
                raise RuntimeError(f"{sdir} exists; use --resume or another session index")
            sdir.mkdir(parents=True)
            design = make_design(cfg, code, idx)
            meta = {**provenance(cfg), "participant_code": code, "session_index": idx,
                    "session_seed": design["session_seed"], "design_hash": design["design_hash"],
                    "design": design, "status": "started", "calibration": None, "practice": [],
                    "rt_reference": self.RT_REFERENCE, "feedback": cfg.feedback,
                    "auto": self.auto}
            write_json(sdir / "session.json", meta)
            print(f"new session {sdir} (seed {design['session_seed']}, design {design['design_hash']})")

        self.sdir, self.meta, self.design = sdir, meta, design
        self.rng = np.random.default_rng([design["session_seed"], self.seed, 0x7C00])
        self.log = TrialLog(sdir / "trials.csv")
        self.done = read_trials(sdir / "trials.csv")
        self.tracks: Dict[int, Track] = {}
        self.trial_index = len(self.done)
        self.done_slots = self._restore()
        if self.done_slots:
            live = sum(1 for t in self.tracks.values() if not t.finished)
            print(f"  restored {len(self.done_slots)} completed trials: "
                  f"{len(self.tracks) - live}/{len(self.tracks)} started tracks already finished, "
                  f"{live} mid-flight")
        return sdir

    def _restore(self) -> set:
        """Rebuild every staircase from the trial log, and return the slots already done.

        Without this, resuming re-created each Track at its starting delta and replayed the
        whole session from the top, appending a second copy of every trial. A session of this
        length has to be splittable across sittings, so resume has to actually resume: the
        logged responses are fed back through the same rule, in the order they were collected,
        which leaves each track at exactly the delta and reversal count it had when the listener
        stopped. Catch trials occupy a slot but never touched a staircase, so they are skipped
        here for the same reason they were skipped then.
        """
        rows = [r for r in self.done if r.get("phase") in ("main", "catch")]
        if not rows:
            return set()
        by_id = {t["track_id"]: t for t in self.design["tracks"]}
        done_slots = set()
        for r in sorted(rows, key=lambda r: int(float(r["slot_index"] or 0))):
            done_slots.add(int(float(r["slot_index"] or 0)))
            if r.get("phase") == "catch":
                continue
            tid = int(float(r["track_id"]))
            spec = by_id.get(tid)
            if spec is None:
                continue
            t = self.tracks.setdefault(tid, Track(self.cfg, spec["condition"], spec["seed"]))
            if not t.finished:
                t.update(bool(int(float(r["correct"]))))
        return done_slots

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
        row["musical_training_years"] = ask("years of musical training", "0",
                                            validator=lambda s: s.replace(".", "").isdigit())
        row["headphones"] = ask("headphone model")
        row["experimenter"] = ask("experimenter")
        row["consent"] = "yes" if getkey({"y", "n"}, "  consent confirmed? [y/n] ") == "y" else "no"
        if row["consent"] != "yes":
            raise SystemExit("consent not confirmed; stopping")
        return row

    def calibrate(self) -> None:
        cfg = self.cfg
        if self.meta.get("calibration"):
            print("calibration already recorded:", self.meta["calibration"])
            self._check_output_drift()
            return
        if self.auto:
            self.meta["calibration"] = {"measured_db_spl": cfg.tone_level_db_spl, "note": "auto",
                                        "time": now_iso(), "system": system_output()}
            write_json(self.sdir / "session.json", self.meta)
            return
        rms_db = 20 * math.log10(cfg.tone_amplitude / math.sqrt(2))
        print(f"\n--- calibration ---\nA {cfg.f_a_hz:.0f} Hz tone at the amplitude of ONE stimulus "
              f"tone ({cfg.tone_amplitude} FS peak, {rms_db:.1f} dB FS rms) will play for 5 s.\n"
              f"Target is {cfg.tone_level_db_spl:.0f} dB SPL at the ear for ONE tone; the two "
              f"together come to {scene_db_spl(cfg.tone_level_db_spl):.0f} dB SPL.")
        if cfg.monaural:
            print("  NOTE: this session is monaural -- LEFT earpiece only.")
        print("  " + describe_output())
        print("  Set the system volume now and DO NOT TOUCH IT AGAIN. It is recorded with the\n"
              "  measurement, and the session will warn you if it changes.")
        while True:
            print("  space = play, then enter the measured level; q = skip (recorded as not calibrated)")
            k = getkey({" ", "q"})
            if k == "q":
                self.meta["calibration"] = {"measured_db_spl": None, "note": "skipped",
                                            "time": now_iso(), "system": system_output()}
                print("  skipped. The level at the ear is now unknown, and every level this session "
                      "reports is nominal.")
                break
            sysnow = system_output()
            if sysnow.get("muted"):
                print("  output is MUTED -- unmute before measuring.")
                continue
            t = np.arange(int(5 * cfg.sample_rate)) / cfg.sample_rate
            x = cfg.tone_amplitude * np.sin(2 * np.pi * cfg.f_a_hz * t)
            self.audio.play(to_output(cfg, x))
            s = ask("measured dB SPL (blank to replay)")
            if s:
                measured = float(s)
                want = amplitude_for(cfg.tone_level_db_spl, measured, cfg.tone_amplitude)
                off = measured - cfg.tone_level_db_spl
                self.meta["calibration"] = {
                    "measured_db_spl": measured, "note": "measured", "time": now_iso(),
                    "system": sysnow, "tone_amplitude": cfg.tone_amplitude,
                    "target_db_spl": cfg.tone_level_db_spl, "offset_db": off,
                    "suggested_tone_amplitude": want}
                print(f"  measured {measured:.1f} dB SPL, {off:+.1f} dB from target.")
                if abs(off) > 2.0:
                    print(f"  To hit {cfg.tone_level_db_spl:.0f} dB exactly, rerun with "
                          f"--set tone_amplitude={want:.4f} (digital scaling is linear, so this "
                          f"is exact). Changing the system volume instead would work too, but it "
                          f"is an undocumented taper and you would have to re-measure.")
                break
        write_json(self.sdir / "session.json", self.meta)

    def _check_output_drift(self) -> None:
        """Has the system volume or output device moved since the level was measured?

        A calibration is a statement about the whole chain. Recording only the SPL pins one end
        of it; if the volume slider moves afterwards the number in session.json is quietly
        false and nothing in the data would ever show it.
        """
        cal = self.meta.get("calibration") or {}
        d = drift(cal.get("system"))
        if not d["known"] or not d["changed"]:
            return
        print("\n  *** THE OUTPUT HAS CHANGED SINCE CALIBRATION ***")
        for n in d["notes"]:
            print("   -", n)
        print("   The recorded level no longer describes what the listener is hearing.")
        if self.auto:
            return
        print("   space = carry on anyway (it is recorded), q = stop and re-calibrate")
        if getkey({" ", "q"}) == "q":
            raise SystemExit("stopped so the level can be re-measured")
        self.meta.setdefault("output_drift", []).append({"time": now_iso(), "notes": d["notes"],
                                                         "now": d["now"]})
        write_json(self.sdir / "session.json", self.meta)

    # -- one trial -----------------------------------------------------------------
    def _present(self, cond: Condition, delta_ms: float, is_catch: bool,
                 feedback: bool) -> dict:
        cfg = self.cfg
        tr = build_trial(cfg, cond, delta_ms, self.rng, is_catch=is_catch)
        t0 = now_iso()
        if self.auto:
            correct = self.sim.respond(cond, abs(tr.delta_realised_ms))
            resp = tr.target_position if correct else 3 - tr.target_position
            rt = float("nan")
            t_start = time.time()
        else:
            x = render_trial(cfg, tr, self.d)
            t_start = time.time()
            self.audio.play(to_output(cfg, x))
            t_play = time.time()
            k = getkey({"1", "2", "q"}, "  1 or 2? ")
            if k == "q":
                raise QuitRequested()
            rt = (time.time() - t_play) * 1000.0
            resp = int(k)
            correct = resp == tr.target_position
        if feedback and not self.auto:
            print("   correct" if correct else f"   no -- it was {tr.target_position}")
        return {"trial": tr, "response": resp, "correct": bool(correct), "rt_ms": rt,
                "t_start": t0, "t_response": now_iso(), "wall_start": t_start}

    def _row(self, phase: str, cond: Condition, out: dict, slot: Optional[dict] = None,
             track: Optional[Track] = None, track_trial=None, feedback: bool = False) -> dict:
        tr = out["trial"]
        row = {"trial_index": self.trial_index, "phase": phase,
               "slot_index": "" if slot is None else slot["index"],
               "block": "" if slot is None else slot["block_index"],
               "track_id": "" if slot is None else slot["track_id"],
               "track_trial_index": "" if track_trial is None else track_trial,
               "condition": cond.name, "lag_pct": cond.lag_pct, "a_kind": cond.a_kind,
               "reference": cond.reference,
               "n_precursor": self.cfg.n_precursor if cond.n_precursor is None else cond.n_precursor,
               "is_catch": int(tr.is_catch), "delta_ms": f"{tr.delta_ms:.6f}",
               "delta_signed_ms": f"{tr.delta_signed_ms:.6f}",
               "delta_realised_ms": f"{tr.delta_realised_ms:.6f}",
               "direction": 1 if tr.delta_signed_ms > 0 else -1,
               "step_index": "" if track is None else track.step_index,
               "reversal": "" if track is None or not track.trials else int(track.trials[-1].reversal),
               "at_ceiling": "" if track is None or not track.trials else int(track.trials[-1].at_ceiling),
               "at_floor": "" if track is None or not track.trials else int(track.trials[-1].at_floor),
               "target_position": tr.target_position, "response": out["response"],
               "correct": int(out["correct"]),
               "rt_ms": "" if out["rt_ms"] != out["rt_ms"] else f"{out['rt_ms']:.1f}",
               "rove_db_1": f"{tr.first.level_db:.3f}", "rove_db_2": f"{tr.second.level_db:.3f}",
               "feedback": int(feedback), "t_start": out["t_start"], "t_response": out["t_response"]}
        self.trial_index += 1
        return row

    # -- blocks ---------------------------------------------------------------------
    def familiarise(self) -> None:
        if not self.cfg.familiarise or self.auto:
            return
        cfg = self.cfg
        print(FAMILIARISE)
        easy = self.cond.get("coh_0") or self.d.conditions[0]
        for i in range(2):
            input("  press enter to hear an example")
            tr = build_trial(cfg, easy, cfg.practice_delta_ms, self.rng)
            print(f"   ... the LAST high tone is out of place in sound {tr.target_position}")
            self.audio.play(to_output(cfg, render_trial(cfg, tr, self.d)))
            print(f"   that was sound {tr.target_position}.")

    def practice(self) -> bool:
        """Easy trials until the listener reaches criterion, or gives up. Returns pass/fail."""
        cfg = self.cfg
        if self.auto:
            return True
        cond = self.cond.get("coh_0") or self.d.conditions[0]
        for attempt in range(1, 4):
            print(f"\n--- practice round {attempt} ({cfg.practice_trials} easy trials) ---")
            n_ok = 0
            for i in range(cfg.practice_trials):
                out = self._present(cond, cfg.practice_delta_ms, False, True)
                n_ok += out["correct"]
                self.log.write(self._row("practice", cond, out, feedback=True))
            p = n_ok / cfg.practice_trials
            self.meta.setdefault("practice", []).append({"round": attempt, "p_correct": p,
                                                         "criterion": cfg.practice_criterion})
            write_json(self.sdir / "session.json", self.meta)
            print(f"  {n_ok}/{cfg.practice_trials} correct ({p:.0%}); criterion is "
                  f"{cfg.practice_criterion:.0%}")
            if p >= cfg.practice_criterion:
                return True
            if attempt < 3:
                print("  let's try that again -- listen for the very last high tone.")
        print("  practice criterion not reached after three rounds. The session will still run, "
              "and the analysis will say so.")
        return False

    def main_block(self) -> None:
        cfg, design = self.cfg, self.design
        slots = design["slot_plan"]
        by_id = {t["track_id"]: t for t in design["tracks"]}
        from .design import duration_estimate
        est = duration_estimate(cfg)
        print(f"\n--- main block: {len(design['tracks'])} tracks, about {est['n_trials']} trials, "
              f"about {est['main_minutes']:.0f} minutes ---")
        self._check_output_drift()
        print(INSTRUCTIONS)
        if not self.auto:
            print("  " + describe_output())
            getkey({" "}, "press space to begin ")

        seen_blocks = set()
        done_slots = getattr(self, "done_slots", set())
        for slot in slots:
            if slot["index"] in done_slots:
                seen_blocks.add(slot["block_index"])
                continue
            tid = slot["track_id"]
            tr_spec = by_id[tid]
            if tid not in self.tracks:
                self.tracks[tid] = Track(cfg, tr_spec["condition"], tr_spec["seed"])
            track = self.tracks[tid]
            if track.finished:
                continue
            cond = self.cond[tr_spec["condition"]]

            if slot["block_index"] not in seen_blocks:
                seen_blocks.add(slot["block_index"])
                if len(seen_blocks) > 1 and (len(seen_blocks) - 1) % max(cfg.break_every, 1) == 0:
                    self._offer_break(len(seen_blocks), design["n_blocks"])

            is_catch = bool(slot["is_catch"])
            delta = cfg.catch_delta_ms if is_catch else track.delta
            probe = cond
            if is_catch and cfg.catch_at_pct is not None:
                # the probe is the SAME easy stimulus wherever it lands, so that a miss means a
                # lapse rather than a hard condition. It is still logged against the track whose
                # slot it occupied, and still does not update it.
                probe = self.catch_cond
            n_before = len(track.trials)
            try:
                out = self._present(probe, delta, is_catch, cfg.feedback)
            except QuitRequested:
                self._finish("quit")
                return
            # a catch trial is a probe, not part of the staircase
            if not is_catch:
                track.update(out["correct"])
            self.log.write(self._row("catch" if is_catch else "main", probe, out, slot=slot,
                                     track=None if is_catch else track,
                                     track_trial=None if is_catch else n_before,
                                     feedback=cfg.feedback))
        self._finish("complete")

    def _offer_break(self, block_no: int, n_blocks: int) -> None:
        if self.auto:
            return
        print(f"\n  --- break ({block_no - 1} of {n_blocks} blocks done). "
              f"Rest as long as you like. ---")
        getkey({" "}, "  press space to carry on ")

    def _finish(self, status: str) -> None:
        self.meta["status"] = status
        self.meta["end_time"] = now_iso()
        self.meta["tracks"] = {str(tid): t.audit() for tid, t in self.tracks.items()}
        write_json(self.sdir / "session.json", self.meta)
        self.log.close()
        n_done = sum(1 for t in self.tracks.values() if t.stop_reason == "converged")
        print(f"\nsession {status}: {n_done}/{len(self.tracks)} tracks converged. "
              f"Saved to {self.sdir}")

    def run(self, code=None, resume=False, session_index=None) -> Path:
        sdir = self.start(code, resume, session_index)
        self.calibrate()
        if not resume:
            self.familiarise()
            self.practice()
        self.main_block()
        return sdir
