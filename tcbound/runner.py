"""Running a session: panel, calibration, familiarisation, practice, interleaved tracks, rests,
fillers, catch trials, timeouts, per-trial logging of every gain, resume.

Keys: 1 and 2 to respond, space to continue, q to quit and save. Terminal I/O, audio and the
session-file helpers are reused from `tcoh`; they are task-agnostic.

What the listener is told
-------------------------
That one tone, the target, is louder on the last repetition in one of the two sounds, and that
the loudness of everything also changes at random. Not louder than WHAT: in type A the change
is only visible against the other tones of the probe, in type B only against the target's own
past, and naming either reference would tell the listener which cue to use. The instruction is
the same for every trial and the types are randomly interleaved, so the cue a listener uses is
the one their percept offers, which is the quantity of interest.

Feedback is on, as in every adaptive 2AFC task in this repository: two-interval forced choice
has no criterion for feedback to move, and it keeps a listener calibrated through a long block.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from tcoh.audiolevel import amplitude_for, describe_output, drift, system_output
from tcoh.runner import Audio, QuitRequested, ResponseTimeout, ask, getkey
from tcoh.session import (TrialLog, check_resumable, existing_sessions, load_participants,
                          next_session_index, now_iso, read_json, read_trials, session_dir,
                          upsert_participant, write_json)

from .config import Condition, Config, catch_condition, conditions, srs_db, validate
from .design import make_design
from .observer import SimulatedListener, cues
from .stimulus import (Trial, build_trial, gains_json, render_interval, render_trial,
                       to_output)
from .track import Track, simulate

PREFIX = "tcbound"
FIELDS = ["trial_index", "phase", "slot_index", "track_id", "track_trial_index", "condition",
          "task", "kind", "step_ms", "is_catch", "srs_db", "delta_db", "step_index",
          "reversal", "at_ceiling", "at_floor", "target_position", "response", "correct",
          "rt_ms", "gains_1", "gains_2", "latent_1", "latent_2", "timbre_1", "timbre_2",
          "own_1", "own_2", "n_active_tracks", "filler_for", "feedback", "timed_out",
          "t_start", "t_response"]

INSTRUCTIONS = """
On every trial you will hear TWO sounds, one after the other.

Each sound is a short pattern of four tones that repeats five times. One of the four tones is
the TARGET tone you have just heard on its own.

In ONE of the two sounds, the target tone is LOUDER on the LAST repetition.

The loudness of the tones also changes at random, in both sounds. That is there to make the
task harder. A change in loudness by itself is not the answer.

  Your job: in which sound was the target tone louder on the last repetition,
  the first or the second?

Press 1 or 2. Guess if you are not sure; you often will be. Some trials are easy and some are
close to impossible, and that is how it is supposed to be.
"""


class Runner:
    RT_REFERENCE = "end of the second interval (audio playback returns)"
    MAX_CONSECUTIVE_TIMEOUTS = 3
    KEY = 0xB0

    def __init__(self, cfg: Config, data_dir: Optional[Path], device=None, audio: bool = True,
                 auto: Optional[str] = None, seed: int = 0, dry: bool = False):
        self.cfg, self.d = cfg, validate(cfg)
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.auto, self.dry, self.seed = auto, dry, int(seed)
        self.audio = Audio(cfg.sample_rate, device, enabled=audio and auto is None and not dry)
        self.cond: Dict[str, Condition] = {c.name: c for c in self.d.conditions}
        self.catch = catch_condition(cfg)
        self.sim = SimulatedListener(cfg, mode=auto, seed=seed + 7) if auto else None
        self.rows: List[dict] = []
        self.history: List[tuple] = []          # (condition, task) of every main-block trial

    # -- setup ----------------------------------------------------------------
    def start_dry(self, code: str = "SIM") -> None:
        """Everything `start` does except the participant panel and the files."""
        self.design = make_design(self.cfg, code, 1)
        self.session_seed = int(self.design["session_seed"])
        self.meta = {"calibration": None}
        self.log = None
        self.tracks: Dict[int, Track] = {}
        self.trial_index = 0
        self.done_slots = set()

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
            idx = session_index or (existing_sessions(self.data_dir, code, PREFIX) or [None])[-1]
            if idx is None:
                raise RuntimeError(f"no {PREFIX} session to resume for {code}")
            sdir = session_dir(self.data_dir, code, idx, PREFIX)
            meta = read_json(sdir / "session.json")
            design = make_design(cfg, code, idx)
            check_resumable(meta, cfg, design)
            print(f"resuming {sdir} (design {design['design_hash']})")
        else:
            idx = session_index or next_session_index(self.data_dir, code, PREFIX)
            sdir = session_dir(self.data_dir, code, idx, PREFIX)
            if sdir.exists():
                raise RuntimeError(f"{sdir} exists; use --resume or another session index")
            sdir.mkdir(parents=True)
            design = make_design(cfg, code, idx)
            from tcoh.session import git_info
            meta = {"task": PREFIX, "config": cfg.to_dict(), "config_hash": cfg.hash(),
                    "participant_code": code, "session_index": idx,
                    "session_seed": design["session_seed"],
                    "design_hash": design["design_hash"], "design": design,
                    "status": "started", "calibration": None, "practice": [],
                    "rt_reference": self.RT_REFERENCE, "feedback": cfg.feedback,
                    "auto": self.auto, "seed": self.seed, "git": git_info(),
                    "start_time": now_iso()}
            write_json(sdir / "session.json", meta)
            print(f"new session {sdir} (seed {design['session_seed']}, "
                  f"design {design['design_hash']})")
        self.sdir, self.meta, self.design = sdir, meta, design
        self.session_seed = int(design["session_seed"])
        self.log = TrialLog(sdir / "trials.csv", fields=FIELDS)
        self.done = read_trials(sdir / "trials.csv")
        self.tracks = {}
        self.trial_index = len(self.done)
        self.done_slots = self._restore()
        if self.done_slots:
            live = sum(1 for t in self.tracks.values() if not t.finished)
            print(f"  restored {len(self.done_slots)} completed slots: "
                  f"{len(self.tracks) - live}/{len(self.tracks)} started tracks finished, "
                  f"{live} mid-flight")
        return sdir

    def _restore(self) -> set:
        """Rebuild every staircase, and the recent order, from the trial log."""
        by_id = {t["track_id"]: t for t in self.design["tracks"]}
        done = set()
        rows = [r for r in self.done if r.get("phase") in ("main", "catch", "filler")]
        rows.sort(key=lambda r: int(float(r["trial_index"])))
        for r in rows:
            if r.get("phase") != "filler" and r.get("slot_index") not in ("", None):
                done.add(int(float(r["slot_index"])))
            self.history.append((r["condition"], r["task"]))
            if r.get("phase") != "main" or str(r.get("timed_out", "")).strip() in ("1", "True"):
                continue
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
        row["hearing"] = ask("hearing (audiogram <= 20 dB HL / self-report normal / other)",
                             "self-report normal")
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
        if self.auto or self.dry:
            self.meta["calibration"] = {"measured_db_spl": cfg.tone_level_db_spl,
                                        "note": "auto", "time": now_iso(),
                                        "system": system_output() if not self.dry else None}
            if not self.dry:
                write_json(self.sdir / "session.json", self.meta)
            return
        rms_db = 20 * math.log10(cfg.tone_amplitude / math.sqrt(2))
        print(f"\n--- calibration ---\nA {cfg.target_freq_hz:.0f} Hz tone at the amplitude of ONE "
              f"stimulus tone at its nominal level ({cfg.tone_amplitude:g} FS peak, "
              f"{rms_db:.1f} dB FS rms).\nTarget: {cfg.tone_level_db_spl:.0f} dB SPL at the ear. "
              f"The roves put single tones between {cfg.tone_level_db_spl - cfg.rove_db:.0f} and "
              f"{cfg.tone_level_db_spl + cfg.rove_db + cfg.delta_max_db:.0f} dB SPL.")
        print("  " + describe_output())
        print("  Set the system volume now and DO NOT TOUCH IT AGAIN.")
        while True:
            print("  space = play, then enter the measured level; q = skip")
            k = getkey({" ", "q"})
            if k == "q":
                self.meta["calibration"] = {"measured_db_spl": None, "note": "skipped",
                                            "time": now_iso(), "system": system_output()}
                print("  skipped: every level this session reports is nominal.")
                break
            if system_output().get("muted"):
                print("  output is MUTED; unmute before measuring.")
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
            want = amplitude_for(cfg.tone_level_db_spl, measured, cfg.tone_amplitude)
            self.meta["calibration"] = {"measured_db_spl": measured, "note": "measured",
                                        "time": now_iso(), "system": system_output(),
                                        "tone_amplitude": cfg.tone_amplitude,
                                        "target_db_spl": cfg.tone_level_db_spl,
                                        "offset_db": cfg.tone_level_db_spl - measured,
                                        "suggested_tone_amplitude": want}
            print(f"  measured {measured:.1f} dB SPL. For {cfg.tone_level_db_spl:.0f} dB SPL set "
                  f"tone_amplitude={want:.4f} (currently {cfg.tone_amplitude:g}).")
            if abs(measured - cfg.tone_level_db_spl) > 3:
                peak = validate(cfg.replace(tone_amplitude=min(want, 0.2))).peak["worst"] \
                    if want < 0.2 else float("inf")
                if peak >= 0.98:
                    print("  That amplitude would clip at the top of the rove. Raise the system "
                          "volume instead, then measure again.")
                else:
                    print("  More than 3 dB out: stop, rerun with --set tone_amplitude=..., and "
                          "measure again.")
            break
        write_json(self.sdir / "session.json", self.meta)

    def _check_drift(self) -> None:
        if self.dry or self.auto:
            return
        cal = self.meta.get("calibration") or {}
        dd = drift(cal.get("system"))
        if not dd["known"] or not dd["changed"]:
            return
        print("\n  *** THE OUTPUT HAS CHANGED SINCE CALIBRATION ***")
        for n in dd["notes"]:
            print("   -", n)
        print("   space = carry on anyway (it is recorded), q = stop and re-calibrate")
        if getkey({" ", "q"}) == "q":
            raise SystemExit("stopped so the level can be re-measured")
        self.meta.setdefault("output_drift", []).append({"time": now_iso(), "notes": dd["notes"]})
        write_json(self.sdir / "session.json", self.meta)

    # -- one trial ------------------------------------------------------------
    def _rng(self, *key) -> np.random.Generator:
        """Keyed on the slot (and on whether this is a filler), so a resumed session presents
        exactly what an uninterrupted one would have."""
        return np.random.default_rng([self.session_seed, self.seed, self.KEY,
                                      *[int(k) for k in key]])

    def _present(self, cond: Condition, srs: float, rng: np.random.Generator,
                 is_catch: bool = False, feedback: bool = True, progress: str = "",
                 delta_override_db: Optional[float] = None,
                 listener_rng: Optional[np.random.Generator] = None) -> dict:
        cfg = self.cfg
        tr = build_trial(cfg, cond, srs, rng, is_catch=is_catch,
                         delta_override_db=delta_override_db)
        t0 = now_iso()
        if self.auto or self.dry:
            resp = self.sim.respond(tr, cond, listener_rng)
            correct = resp == tr.target_position
            rt = float("nan")
        else:
            self.audio.play(to_output(cfg, render_trial(cfg, tr)))
            t_play = time.time()
            try:
                k = getkey({"1", "2", "q"}, f"  {progress}1 or 2? ",
                           timeout_s=cfg.response_timeout_s or None)
            except ResponseTimeout:
                print(f"\n   no response within {cfg.response_timeout_s:g} s; not scored")
                return {"trial": tr, "response": "", "correct": False, "rt_ms": float("nan"),
                        "timed_out": True, "t_start": t0, "t_response": now_iso()}
            if k == "q":
                raise QuitRequested()
            rt = (time.time() - t_play) * 1000.0
            resp = int(k)
            correct = resp == tr.target_position
        if feedback and not (self.auto or self.dry):
            print("   correct" if correct else f"   no: it was {tr.target_position}")
        return {"trial": tr, "response": resp, "correct": bool(correct), "rt_ms": rt,
                "timed_out": False, "t_start": t0, "t_response": now_iso()}

    def _row(self, phase: str, cond: Condition, out: dict, slot=None, track=None,
             track_trial=None, feedback=False, n_active="") -> dict:
        tr: Trial = out["trial"]
        c1 = cues(self.cfg, tr.first.gains_db)
        c2 = cues(self.cfg, tr.second.gains_db)

        def num(x):
            return "" if x is None or not np.isfinite(x) else f"{float(x):.4f}"
        last = track.trials[-1] if track is not None and track.trials else None
        row = {"trial_index": self.trial_index, "phase": phase,
               "slot_index": "" if slot is None else slot["index"],
               "track_id": "" if slot is None else slot["track_id"],
               "track_trial_index": "" if track_trial is None else track_trial,
               "condition": cond.name, "task": cond.task, "kind": cond.kind,
               "step_ms": f"{cond.step_ms:g}", "is_catch": int(tr.is_catch),
               "srs_db": f"{tr.srs_db:.4f}", "delta_db": f"{tr.delta_db:.4f}",
               "step_index": "" if last is None else last.step_index,
               "reversal": "" if last is None else int(last.reversal),
               "at_ceiling": "" if last is None else int(last.at_ceiling),
               "at_floor": "" if last is None else int(last.at_floor),
               "target_position": tr.target_position, "response": out["response"],
               "correct": int(out["correct"]),
               "rt_ms": "" if out["rt_ms"] != out["rt_ms"] else f"{out['rt_ms']:.1f}",
               "gains_1": gains_json(tr.first.gains_db), "gains_2": gains_json(tr.second.gains_db),
               "latent_1": json.dumps([round(x, 4) for x in tr.first.latent]),
               "latent_2": json.dumps([round(x, 4) for x in tr.second.latent]),
               "timbre_1": num(c1["timbre"]), "timbre_2": num(c2["timbre"]),
               "own_1": num(c1["own"]), "own_2": num(c2["own"]),
               "n_active_tracks": n_active,
               "feedback": int(feedback), "timed_out": int(bool(out.get("timed_out"))),
               "t_start": out["t_start"], "t_response": out["t_response"]}
        self.trial_index += 1
        return row

    def _write(self, row: dict) -> None:
        self.rows.append(row)
        if self.log is not None:
            self.log.write(row)

    # -- familiarisation and practice --------------------------------------------
    def _play_interval(self, cond: Condition, srs: float = -30.0, signal: bool = False,
                       rng=None) -> None:
        tr = build_trial(self.cfg, cond, srs, rng or np.random.default_rng(), target_position=1)
        iv = tr.first if signal else tr.second
        self.audio.play(to_output(self.cfg, render_interval(self.cfg, iv, tr.phases,
                                                            tr.captor_phases, tr.total_ms)))

    def familiarise(self) -> None:
        if self.auto or self.dry or not self.cfg.familiarise:
            return
        cfg = self.cfg
        rng = self._rng(9000)
        alone = Condition("fam_alone", "none", 0.0, "alone")
        fig0 = Condition("fam_0", "none", 0.0, "figure")
        figmax = Condition("fam_max", "none", max(cfg.steps_ms), "figure")
        print("\n--- familiarisation ---\nFirst, the TARGET tone on its own.")
        for _ in range(2):
            getkey({" "}, "  press space to listen ")
            self._play_interval(alone, rng=rng)
        print("\nNow the whole pattern, twice in its tightest form and twice spread out. The "
              "target tone is in it every time. Try to follow it.")
        for c in (fig0, fig0, figmax, figmax):
            getkey({" "}, "  press space to listen ")
            self._play_interval(c, rng=rng)
        print("\nNow two complete trials with no random loudness changes. In one of the two "
              "sounds the target is louder on the last repetition. You will be told which.")
        for c in (fig0, figmax):
            getkey({" "}, "  press space to listen ")
            out = self._present(c, srs_db(cfg.catch_delta_db), rng, feedback=False)
            print(f"   it was sound {out['trial'].target_position}")
        print("\nIn the real trials the loudness of the tones also changes at random. Two "
              "examples, with the answer.")
        for name in ("A_0", f"B_{max(cfg.steps_ms):g}".replace(".", "p")):
            c = self.cond.get(name)
            if c is None:
                continue
            getkey({" "}, "  press space to listen ")
            out = self._present(c, cfg.practice_srs_db, rng, feedback=False)
            print(f"   it was sound {out['trial'].target_position}")

    def practice(self) -> None:
        cfg = self.cfg
        if self.auto or self.dry:
            return
        names = list(self.cond)
        for rnd in range(1, cfg.practice_rounds + 1):
            print(f"\n--- practice round {rnd} ({cfg.practice_trials} trials, answers shown) ---")
            rng = self._rng(8000 + rnd)
            n_ok = 0
            for i in range(cfg.practice_trials):
                c = self.cond[names[int(rng.integers(len(names)))]]
                out = self._present(c, cfg.practice_srs_db, rng, feedback=True)
                n_ok += int(out["correct"])
                self._write(self._row("practice", c, out, feedback=True))
            p = n_ok / cfg.practice_trials
            self.meta.setdefault("practice", []).append(p)
            write_json(self.sdir / "session.json", self.meta)
            print(f"  {n_ok}/{cfg.practice_trials} correct ({p:.0%}); "
                  f"criterion is {cfg.practice_criterion:.0%}")
            if p >= cfg.practice_criterion:
                return
            print("  Once more. Listen for the target tone on the last repetition.")
        print("  Practice criterion not reached. The session still runs, and the analysis "
              "will say so.")

    # -- the main block -----------------------------------------------------------
    def expected_trials(self) -> int:
        sim = simulate(self.cfg, thresholds_srs=(-10.0, -4.0, 2.0), betas=(1.0,), n_runs=40,
                       seed=3)
        per = sim["median_trials"]
        return int(round(self.d.n_tracks * per * (1.0 + self.cfg.catch_rate)))

    def _needs_filler(self, cond: Condition) -> Optional[str]:
        cfg = self.cfg
        if not cfg.fillers:
            return None
        h = self.history
        n_c, n_t = cfg.max_same_condition_run, cfg.max_same_task_run
        if len(h) >= n_c and all(c == cond.name for c, _ in h[-n_c:]):
            return "condition"
        if len(h) >= n_t and all(t == cond.task for _, t in h[-n_t:]):
            return "task"
        return None

    def _filler(self, cond: Condition, why: str, slot: dict) -> None:
        """One unscored trial of another condition, at that condition's current level.

        The choice and the trial come from two separate keyed generators, so the trial can be
        rebuilt from the logged condition alone (`export`).
        """
        pool = [c for c in self.cond.values()
                if (c.task != cond.task if why == "task" else c.name != cond.name)]
        pick = pool[int(self._rng(slot["index"], 2).integers(len(pool)))]
        tid = next(t["track_id"] for t in self.design["tracks"] if t["condition"] == pick.name)
        tr = self.tracks.get(tid)
        srs = tr.srs if tr is not None else self.cfg.srs_start_db
        out = self._present(pick, srs, self._rng(slot["index"], 1), feedback=self.cfg.feedback,
                            listener_rng=self._rng(slot["index"], 4))
        self.sound_s += out["trial"].total_ms * 2 / 1000.0 + self.cfg.isi_ms / 1000.0
        row = self._row("filler", pick, out, slot=slot, feedback=self.cfg.feedback,
                        n_active=self._n_active())
        row["track_id"] = ""
        row["filler_for"] = f"{cond.name}:{why}"
        self._write(row)
        self.history.append((pick.name, pick.task))
        self.n_filler += 1

    def _n_active(self) -> int:
        by_id = {t["track_id"]: t for t in self.design["tracks"]}
        return sum(1 for tid in by_id if tid not in self.tracks or not self.tracks[tid].finished)

    def main_block(self) -> None:
        cfg, design = self.cfg, self.design
        by_id = {t["track_id"]: t for t in design["tracks"]}
        n_est = self.expected_trials() if not self.dry else 1
        self.n_filler = self.n_catch = self.n_rests = self.n_main = 0
        self.sound_s = 0.0
        if not (self.auto or self.dry):
            print(f"\n--- main block: {len(design['tracks'])} conditions interleaved, about "
                  f"{n_est} trials ---")
            self._check_drift()
            print(INSTRUCTIONS)
            print("  " + describe_output())
            getkey({" "}, "press space to begin ")
        n_done = len(self.done_slots)
        since_rest = n_timeouts = consecutive = 0
        for slot in design["slot_plan"]:
            if slot["index"] in self.done_slots:
                continue
            tid = slot["track_id"]
            spec = by_id[tid]
            track = self.tracks.setdefault(tid, Track(cfg, spec["condition"], spec["seed"]))
            if track.finished:
                continue
            if cfg.break_every_trials and since_rest >= cfg.break_every_trials:
                self._offer_rest(n_done, n_est)
                since_rest = 0
            cond = self.cond[spec["condition"]]
            is_catch = bool(slot["is_catch"])
            try:
                if not is_catch:
                    why = self._needs_filler(cond)
                    if why:
                        self._filler(cond, why, slot)
                        since_rest += 1
                        n_done += 1
                n_done += 1
                pct = min(100.0 * n_done / max(n_est, 1), 99.0)
                prog = f"[{n_done}/~{n_est}  {pct:.0f}%]  "
                rng = self._rng(slot["index"])
                n_active = self._n_active()
                lrng = self._rng(slot["index"], 3)
                if is_catch:
                    out = self._present(self.catch, srs_db(cfg.catch_delta_db), rng,
                                        is_catch=True, feedback=cfg.feedback, progress=prog,
                                        listener_rng=lrng)
                else:
                    out = self._present(cond, track.srs, rng, feedback=cfg.feedback,
                                        progress=prog, listener_rng=lrng)
                since_rest += 1
            except QuitRequested:
                self._finish("quit", n_timeouts)
                return
            self.sound_s += out["trial"].total_ms * 2 / 1000.0 + cfg.isi_ms / 1000.0
            probe = self.catch if is_catch else cond
            self.history.append((probe.name, probe.task))
            if out.get("timed_out"):
                n_timeouts += 1
                consecutive += 1
                self._write(self._row("catch" if is_catch else "main", probe, out, slot=slot,
                                      feedback=cfg.feedback, n_active=n_active))
                if consecutive >= self.MAX_CONSECUTIVE_TIMEOUTS:
                    print(f"\n  {consecutive} trials in a row with no response. Stopping.")
                    self._finish("abandoned", n_timeouts)
                    return
                continue
            consecutive = 0
            if is_catch:
                self.n_catch += 1
                self._write(self._row("catch", probe, out, slot=slot, feedback=cfg.feedback,
                                      n_active=n_active))
                continue
            n_before = len(track.trials)
            track.update(out["correct"])
            self.n_main += 1
            self._write(self._row("main", cond, out, slot=slot, track=track,
                                  track_trial=n_before, feedback=cfg.feedback,
                                  n_active=n_active))
            if all(t.finished for t in self.tracks.values()) and \
                    len(self.tracks) == len(by_id):
                break
        self._finish("complete", n_timeouts)

    def _offer_rest(self, a: int, b: int) -> None:
        self.n_rests += 1
        if self.auto or self.dry:
            return
        print(f"\n  --- rest ({a} trials done of about {b}, "
              f"{min(100.0 * a / max(b, 1), 99.0):.0f}%). Rest as long as you like. ---")
        getkey({" "}, "  press space to carry on ")

    def _finish(self, status: str, n_timeouts: int = 0) -> None:
        self.status = status
        if self.dry:
            return
        self.meta["status"] = status
        self.meta["end_time"] = now_iso()
        self.meta["n_timeouts"] = n_timeouts
        self.meta["n_fillers"] = getattr(self, "n_filler", 0)
        self.meta["tracks"] = {str(t): tr.audit() for t, tr in self.tracks.items()}
        write_json(self.sdir / "session.json", self.meta)
        if self.log is not None:
            self.log.close()
        n_done = sum(1 for t in self.tracks.values() if t.finished)
        print(f"\nsession {status}: {n_done}/{len(self.tracks)} tracks finished. "
              f"Saved to {self.sdir}")

    def summary(self) -> dict:
        lens = [len(t.trials) for t in self.tracks.values()]
        n = self.n_main + self.n_catch + self.n_filler
        return {"n_trials": n, "n_main": self.n_main, "n_catch": self.n_catch,
                "n_filler": self.n_filler, "n_rests": self.n_rests, "sound_s": self.sound_s,
                "trials_per_track_median": float(np.median(lens)) if lens else 0.0,
                "status": getattr(self, "status", None),
                "thresholds": {t.condition: t.threshold_db() for t in self.tracks.values()}}

    def run(self, code=None, resume=False, session_index=None) -> Path:
        sdir = self.start(code, resume, session_index)
        self.calibrate()
        if not resume:
            self.familiarise()
            self.practice()
        self.main_block()
        return sdir
