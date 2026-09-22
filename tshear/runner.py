"""Running a session. The terminal I/O, audio, level calibration and trial log are reused from
`tcoh` -- they are task-agnostic and there is nothing to gain by writing them twice."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from tcoh.audiolevel import describe_output, drift, scene_db_spl, system_output
from tcoh.runner import (Audio, QuitRequested, ResponseTimeout, ask, getkey)
from tcoh.session import (DesignChanged, TrialLog, check_resumable, existing_sessions,
                          load_participants, next_session_index, now_iso, read_json,
                          read_trials, session_dir, upsert_participant, write_json)
from tcoh.track import Track

from .config import Condition, Config, validate
from .design import duration_estimate, make_design
from .stimulus import build_trial, render_trial, to_output

FIELDS = ["trial_index", "phase", "slot_index", "block", "track_id", "track_trial_index",
          "condition", "step_pct", "kind", "is_catch", "delta_ms", "delta_signed_ms",
          "delta_realised_ms", "direction", "crosses_neighbour", "step_index", "reversal",
          "at_ceiling", "at_floor", "target_position", "response", "correct", "rt_ms",
          "rove_db_1", "rove_db_2", "feedback", "timed_out", "t_start", "t_response"]

INSTRUCTIONS = """
On every trial you will hear TWO sounds, one after the other, with a short gap between them.

Each sound is a short pattern of four tones that repeats five times, always the same way.

The two sounds are the same except for one thing: in ONE of them, on the LAST repetition only,
one of the tones is slightly out of place in time. It may be a little early or a little late.

  Your job: which sound had the tone out of place -- the first or the second?

Press 1 or 2. Guess if you are not sure; you often will be. Some trials are very easy and some
are close to impossible, and that is how it is supposed to be.
"""


class Runner:
    RT_REFERENCE = "end of the second interval (audio playback returns)"
    MAX_CONSECUTIVE_TIMEOUTS = 3

    def __init__(self, cfg: Config, data_dir: Path, device=None, audio: bool = True,
                 auto: Optional[str] = None, seed: int = 0):
        self.cfg, self.d = cfg, validate(cfg)
        self.data_dir = Path(data_dir)
        self.auto = auto
        self.audio = Audio(cfg.sample_rate, device, enabled=audio and auto is None)
        self.seed = int(seed)
        self.cond = {c.name: c for c in self.d.conditions}
        self.catch_cond = next((c for c in self.d.conditions
                                if c.kind == "figure" and cfg.catch_at_pct is not None
                                and abs(c.step_pct - cfg.catch_at_pct) < 1e-9), None)
        self.rng = np.random.default_rng([self.seed, 0x5E])

    # -- setup ----------------------------------------------------------------
    def start(self, code=None, resume=False, session_index=None) -> Path:
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
            idx = session_index or (existing_sessions(self.data_dir, code, prefix="tshear")
                                    or [None])[-1]
            if idx is None:
                raise RuntimeError(f"no tshear session to resume for {code}")
            sdir = session_dir(self.data_dir, code, idx, prefix="tshear")
            meta = read_json(sdir / "session.json")
            design = make_design(cfg, code, idx)
            check_resumable(meta, cfg, design)
            print(f"resuming {sdir} (design {design['design_hash']})")
        else:
            idx = session_index or next_session_index(self.data_dir, code, prefix="tshear")
            sdir = session_dir(self.data_dir, code, idx, prefix="tshear")
            if sdir.exists():
                raise RuntimeError(f"{sdir} exists; use --resume or another session index")
            sdir.mkdir(parents=True)
            design = make_design(cfg, code, idx)
            meta = {"task": "tshear", "config": cfg.to_dict(), "config_hash": cfg.hash(),
                    "participant_code": code, "session_index": idx,
                    "session_seed": design["session_seed"],
                    "design_hash": design["design_hash"], "design": design,
                    "status": "started", "calibration": None, "practice": [],
                    "rt_reference": self.RT_REFERENCE, "feedback": cfg.feedback,
                    "auto": self.auto, "start_time": now_iso()}
            write_json(sdir / "session.json", meta)
            print(f"new session {sdir} (seed {design['session_seed']}, "
                  f"design {design['design_hash']})")

        self.sdir, self.meta, self.design = sdir, meta, design
        self.session_seed = int(design["session_seed"])
        self.log = TrialLog(sdir / "trials.csv", fields=FIELDS)
        self.done = read_trials(sdir / "trials.csv")
        self.tracks: Dict[int, Track] = {}
        self.trial_index = len(self.done)
        self.done_slots = self._restore()
        if self.done_slots:
            live = sum(1 for t in self.tracks.values() if not t.finished)
            print(f"  restored {len(self.done_slots)} completed trials: "
                  f"{len(self.tracks) - live}/{len(self.tracks)} started tracks finished, "
                  f"{live} mid-flight")
        return sdir

    def _restore(self) -> set:
        """Rebuild every staircase from the trial log, and return the slots already done."""
        rows = [r for r in self.done if r.get("phase") in ("main", "catch")]
        if not rows:
            return set()
        by_id = {t["track_id"]: t for t in self.design["tracks"]}
        done = set()
        for r in sorted(rows, key=lambda r: int(float(r["slot_index"] or 0))):
            done.add(int(float(r["slot_index"] or 0)))
            if r.get("phase") == "catch" or str(r.get("timed_out", "")).strip() in ("1", "True"):
                continue                       # neither ever touched a staircase
            tid = int(float(r["track_id"]))
            spec = by_id.get(tid)
            if spec is None:
                continue
            t = self.tracks.setdefault(tid, Track(self.cfg, spec["condition"], spec["seed"]))
            if not t.finished:
                t.update(bool(int(float(r["correct"]))))
        return done

    def panel(self, code=None) -> dict:
        known = load_participants(self.data_dir)
        print("\n--- participant panel ---")
        code = code or ask("participant code", validator=lambda s: s.isalnum())
        if code in known:
            print("  existing participant:", {k: v for k, v in known[code].items()
                                              if k != "created_at"})
            if getkey({"y", "n"}, "  use this record? [y/n] ") == "y":
                row = dict(known[code])
                row["experimenter"] = ask("experimenter", row.get("experimenter", ""))
                row["headphones"] = ask("headphone model", row.get("headphones", ""))
                row["consent"] = "yes" if getkey({"y", "n"},
                                                 "  consent confirmed? [y/n] ") == "y" else "no"
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

    # -- level ----------------------------------------------------------------
    def calibrate(self) -> None:
        import math
        cfg = self.cfg
        if self.meta.get("calibration"):
            print("calibration already recorded:", self.meta["calibration"])
            self._check_drift()
            return
        if self.auto:
            self.meta["calibration"] = {"measured_db_spl": cfg.tone_level_db_spl,
                                        "note": "auto", "time": now_iso(),
                                        "system": system_output()}
            write_json(self.sdir / "session.json", self.meta)
            return
        rms_db = 20 * math.log10(cfg.tone_amplitude / math.sqrt(2))
        print(f"\n--- calibration ---\nA {cfg.target_freq_hz:.0f} Hz tone at the amplitude of ONE "
              f"stimulus tone ({cfg.tone_amplitude:g} FS peak, {rms_db:.1f} dB FS rms).\n"
              f"Target is {cfg.tone_level_db_spl:.0f} dB SPL at the ear for ONE tone; at the "
              f"smallest step the four overlap and the figure reaches "
              f"{scene_db_spl(cfg.tone_level_db_spl, cfg.n_tones):.0f} dB SPL.")
        print("  " + describe_output())
        print("  Set the system volume now and DO NOT TOUCH IT AGAIN.")
        while True:
            print("  space = play, then enter the measured level; q = skip")
            k = getkey({" ", "q"})
            if k == "q":
                self.meta["calibration"] = {"measured_db_spl": None, "note": "skipped",
                                            "time": now_iso(), "system": system_output()}
                print("  skipped -- every level this session reports is nominal.")
                break
            if system_output().get("muted"):
                print("  output is MUTED -- unmute before measuring.")
                continue
            t = np.arange(int(5.0 * cfg.sample_rate)) / cfg.sample_rate
            self.audio.play(to_output(cfg, cfg.tone_amplitude
                                      * np.sin(2 * np.pi * cfg.target_freq_hz * t)))
            got = ask("measured dB SPL (blank to play again)")
            if not got:
                continue
            try:
                measured = float(got)
            except ValueError:
                print("  not a number")
                continue
            from tcoh.audiolevel import amplitude_for
            want = amplitude_for(cfg.tone_level_db_spl, measured, cfg.tone_amplitude)
            self.meta["calibration"] = {"measured_db_spl": measured, "note": "measured",
                                        "time": now_iso(), "system": system_output(),
                                        "tone_amplitude": cfg.tone_amplitude,
                                        "target_db_spl": cfg.tone_level_db_spl,
                                        "offset_db": cfg.tone_level_db_spl - measured,
                                        "suggested_tone_amplitude": want}
            print(f"  measured {measured:.1f} dB SPL. For {cfg.tone_level_db_spl:.0f} dB SPL set "
                  f"tone_amplitude={want:.4f} (currently {cfg.tone_amplitude:g}).")
            if scene_db_spl(measured, cfg.n_tones) > 80:
                print("  WARNING: the four tones together exceed 80 dB SPL at this setting.")
            break
        write_json(self.sdir / "session.json", self.meta)

    def _check_drift(self) -> None:
        cal = self.meta.get("calibration") or {}
        dd = drift(cal.get("system"))
        if not dd["known"] or not dd["changed"]:
            return
        print("\n  *** THE OUTPUT HAS CHANGED SINCE CALIBRATION ***")
        for n in dd["notes"]:
            print("   -", n)
        if self.auto:
            return
        print("   space = carry on anyway (it is recorded), q = stop and re-calibrate")
        if getkey({" ", "q"}) == "q":
            raise SystemExit("stopped so the level can be re-measured")
        self.meta.setdefault("output_drift", []).append({"time": now_iso(), "notes": dd["notes"]})
        write_json(self.sdir / "session.json", self.meta)

    # -- one trial ------------------------------------------------------------
    def _slot_rng(self, slot_index: int) -> np.random.Generator:
        """Keyed on the slot, so resuming reproduces exactly what an uninterrupted run gives."""
        return np.random.default_rng([self.session_seed, self.seed, 0x5E, int(slot_index)])

    def _present(self, cond: Condition, delta_ms: float, is_catch: bool, feedback: bool,
                 progress: str = "", rng=None) -> dict:
        cfg = self.cfg
        tr = build_trial(cfg, cond, delta_ms, rng if rng is not None else self.rng,
                         is_catch=is_catch)
        t0 = now_iso()
        if self.auto:
            correct = self._simulate(cond, abs(tr.delta_realised_ms))
            resp = tr.target_position if correct else 3 - tr.target_position
            rt = float("nan")
        else:
            x = render_trial(cfg, tr)
            self.audio.play(to_output(cfg, x))
            t_play = time.time()
            try:
                k = getkey({"1", "2", "q"}, f"  {progress}1 or 2? ",
                           timeout_s=cfg.response_timeout_s or None)
            except ResponseTimeout:
                print(f"\n   no response within {cfg.response_timeout_s:g} s -- not scored")
                return {"trial": tr, "response": "", "correct": False, "rt_ms": float("nan"),
                        "timed_out": True, "t_start": t0, "t_response": now_iso()}
            if k == "q":
                raise QuitRequested()
            rt = (time.time() - t_play) * 1000.0
            resp = int(k)
            correct = resp == tr.target_position
        if feedback and not self.auto:
            print("   correct" if correct else f"   no -- it was {tr.target_position}")
        return {"trial": tr, "response": resp, "correct": bool(correct), "rt_ms": rt,
                "timed_out": False, "t_start": t0, "t_response": now_iso()}

    def _simulate(self, cond: Condition, delta_ms: float) -> bool:
        """A listener with no ears, for exercising the pipeline.

        `figure`  -- threshold rises with the step, as the coherence account predicts.
        `within`  -- the null: the listener uses only the target's own rhythm, so the
                     threshold is the same at every step and the curve is FLAT.
        """
        from tcoh.psychometric import PF
        if self.auto == "within":
            thr = 9.0
        else:
            from .model import effective_objects
            v = (0.0 if cond.kind == "single"
                 else effective_objects(self.cfg, cond.step_pct)["normalised"])
            thr = 2.5 * (9.0 / 2.5) ** v if cond.kind == "figure" else 9.0
        p = PF(threshold_ms=thr, sigma=0.55).p_correct(max(delta_ms, 1e-6))
        return bool(self.rng.random() < float(p))

    def _row(self, phase, cond, out, slot=None, track=None, track_trial=None,
             feedback=False) -> dict:
        tr = out["trial"]
        row = {"trial_index": self.trial_index, "phase": phase,
               "slot_index": "" if slot is None else slot["index"],
               "block": "" if slot is None else slot["block_index"],
               "track_id": "" if slot is None else slot["track_id"],
               "track_trial_index": "" if track_trial is None else track_trial,
               "condition": cond.name, "step_pct": cond.step_pct, "kind": cond.kind,
               "is_catch": int(tr.is_catch), "delta_ms": f"{tr.delta_ms:.6f}",
               "delta_signed_ms": f"{tr.delta_signed_ms:.6f}",
               "delta_realised_ms": f"{tr.delta_realised_ms:.6f}",
               "direction": 1 if tr.delta_signed_ms > 0 else -1,
               "crosses_neighbour": int(tr.crosses_neighbour),
               "step_index": "" if track is None else track.step_index,
               "reversal": "" if track is None or not track.trials
               else int(track.trials[-1].reversal),
               "at_ceiling": "" if track is None or not track.trials
               else int(track.trials[-1].at_ceiling),
               "at_floor": "" if track is None or not track.trials
               else int(track.trials[-1].at_floor),
               "target_position": tr.target_position, "response": out["response"],
               "correct": int(out["correct"]),
               "rt_ms": "" if out["rt_ms"] != out["rt_ms"] else f"{out['rt_ms']:.1f}",
               "rove_db_1": f"{tr.first.level_db:.3f}", "rove_db_2": f"{tr.second.level_db:.3f}",
               "feedback": int(feedback), "timed_out": int(bool(out.get("timed_out"))),
               "t_start": out["t_start"], "t_response": out["t_response"]}
        self.trial_index += 1
        return row

    # -- phases ---------------------------------------------------------------
    def familiarise(self) -> None:
        if self.auto or not self.cfg.familiarise:
            return
        easy = next(c for c in self.d.conditions if c.kind == "figure")
        print("\n--- familiarisation ---\nTwo examples with the shift made very large. "
              "You will be told the answer.")
        for _ in range(2):
            getkey({" "}, "  press space to listen ")
            out = self._present(easy, self.cfg.practice_delta_ms, False, False)
            print(f"   that was interval {out['trial'].target_position}")

    def practice(self) -> None:
        cfg = self.cfg
        if self.auto:
            return
        easy = next(c for c in self.d.conditions if c.kind == "figure")
        for rnd in range(1, 4):
            print(f"\n--- practice round {rnd} ({cfg.practice_trials} easy trials) ---")
            n_ok = 0
            for _ in range(cfg.practice_trials):
                out = self._present(easy, cfg.practice_delta_ms, False, True)
                n_ok += int(out["correct"])
                self.log.write(self._row("practice", easy, out, feedback=True))
            p = n_ok / cfg.practice_trials
            self.meta.setdefault("practice", []).append(p)
            write_json(self.sdir / "session.json", self.meta)
            print(f"  {n_ok}/{cfg.practice_trials} correct ({p:.0%}); "
                  f"criterion is {cfg.practice_criterion:.0%}")
            if p >= cfg.practice_criterion:
                return
            print("  let's try that again -- listen for the tone that is out of place.")
        print("  practice criterion not reached after three rounds. The session will still run, "
              "and the analysis will say so.")

    def main_block(self) -> None:
        cfg, design = self.cfg, self.design
        slots = design["slot_plan"]
        by_id = {t["track_id"]: t for t in design["tracks"]}
        est = duration_estimate(cfg)
        print(f"\n--- main block: {len(design['tracks'])} tracks, about {est['n_trials']} "
              f"trials, about {est['main_minutes']:.0f} minutes ---")
        self._check_drift()
        print(INSTRUCTIONS)
        if not self.auto:
            print("  " + describe_output())
            getkey({" "}, "press space to begin ")

        seen_blocks, done_slots = set(), getattr(self, "done_slots", set())
        n_est = max(int(est["n_trials"]), 1)
        n_done = len(done_slots)
        since_break = n_timeouts = consecutive = 0
        for slot in slots:
            if slot["index"] in done_slots:
                seen_blocks.add(slot["block_index"])
                continue
            tid = slot["track_id"]
            spec = by_id[tid]
            if tid not in self.tracks:
                self.tracks[tid] = Track(cfg, spec["condition"], spec["seed"])
            track = self.tracks[tid]
            if track.finished:
                continue
            cond = self.cond[spec["condition"]]

            if slot["block_index"] not in seen_blocks:
                seen_blocks.add(slot["block_index"])
                if len(seen_blocks) > 1 and (len(seen_blocks) - 1) % max(cfg.break_every, 1) == 0:
                    self._offer("break", len(seen_blocks), design["n_blocks"])
                    since_break = 0
            elif cfg.break_every_trials and since_break >= cfg.break_every_trials:
                self._offer("rest", n_done, n_est)
                since_break = 0

            is_catch = bool(slot["is_catch"])
            delta = cfg.catch_delta_ms if is_catch else track.delta
            probe = self.catch_cond if (is_catch and self.catch_cond) else cond
            n_before = len(track.trials)
            n_done += 1
            pct = min(100.0 * n_done / n_est, 99.0)
            try:
                out = self._present(probe, delta, is_catch, cfg.feedback,
                                    progress=f"[{n_done}/~{n_est}  {pct:.0f}%]  ",
                                    rng=self._slot_rng(slot["index"]))
                since_break += 1
            except QuitRequested:
                self._finish("quit", n_timeouts)
                return
            if out.get("timed_out"):
                n_timeouts += 1
                consecutive += 1
                if consecutive >= self.MAX_CONSECUTIVE_TIMEOUTS:
                    print(f"\n  {consecutive} trials in a row with no response. Stopping.")
                    self.log.write(self._row("main", probe, out, slot=slot,
                                             feedback=cfg.feedback))
                    self._finish("abandoned", n_timeouts)
                    return
            else:
                consecutive = 0
            scored = not is_catch and not out.get("timed_out")
            if scored:
                track.update(out["correct"])
            self.log.write(self._row("catch" if is_catch else "main", probe, out, slot=slot,
                                     track=track if scored else None,
                                     track_trial=n_before if scored else None,
                                     feedback=cfg.feedback))
        self._finish("complete", n_timeouts)

    def _offer(self, what: str, a: int, b: int) -> None:
        if self.auto:
            return
        if what == "break":
            print(f"\n  --- break ({a - 1} of {b} blocks done). Rest as long as you like. ---")
        else:
            print(f"\n  --- rest ({a} trials done of about {b}, "
                  f"{min(100.0 * a / max(b, 1), 99.0):.0f}%). Rest as long as you like. ---")
        getkey({" "}, "  press space to carry on ")

    def _finish(self, status: str, n_timeouts: int = 0) -> None:
        self.meta["status"] = status
        self.meta["end_time"] = now_iso()
        self.meta["n_timeouts"] = n_timeouts
        self.meta["tracks"] = {str(t): tr.audit() for t, tr in self.tracks.items()}
        write_json(self.sdir / "session.json", self.meta)
        self.log.close()
        n_done = sum(1 for t in self.tracks.values() if t.finished)
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
