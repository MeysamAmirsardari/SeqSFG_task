"""Which track runs when, and how long the session takes.

Every track is interleaved with every other for the whole session. The plan is a sequence of
CYCLES, each a fresh random permutation of all tracks, so every condition gets one trial per
cycle and no condition is ever more than one cycle ahead of another. A finished track's slots
are skipped. Catch trials are a random 6% of the slots and present the catch stimulus instead of
the host track's trial, which they never update.

Why the order must stay unpredictable here
-------------------------------------------
The inference is that the listener used whatever cue their percept affords, without knowing in
advance which cue the trial will reward. If the end of a session were a run of one condition,
or of one trial type, a listener could switch strategy for it. So the runner adds a FILLER (an
unscored trial of another condition, at that track's current level) whenever the next trial
would make a run longer than `max_same_condition_run` of one condition or `max_same_task_run` of
one type. Fillers are logged and enter the decision-weight analysis; they never touch a track.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List

import numpy as np

from .config import Config, catch_condition, validate

# Measured on the timing tasks across completed sessions, pauses of 30 s or more excluded: the
# interval between trial starts ran 0.55 s beyond the duration of the sound.
PER_TRIAL_OVERHEAD_S = 0.55
REST_MIN = 0.75
FAMILIARISE_MIN = 2.0
INSTRUCTIONS_MIN = 1.0


def session_seed(cfg: Config, code: str, session_index: int) -> int:
    h = hashlib.sha256(f"tcbound|{cfg.hash()}|{code}|{session_index}".encode()).hexdigest()
    return int(h[:12], 16)


def make_design(cfg: Config, code: str, session_index: int) -> dict:
    d = validate(cfg)
    names = [c.name for c in d.conditions]
    seed = session_seed(cfg, code, session_index)
    rng = np.random.default_rng(seed)
    tracks, tid = [], 0
    for rnd in range(cfg.tracks_per_condition):
        for name in names:
            tracks.append({"track_id": tid, "condition": name, "round_index": rnd,
                           "seed": int(rng.integers(0, 2 ** 31))})
            tid += 1
    slots, prev = [], None
    for cyc in range(cfg.max_trials_per_track):
        order = list(range(len(tracks)))
        for _ in range(100):
            rng.shuffle(order)
            if prev is None or tracks[order[0]]["condition"] != prev:
                break
        for i in order:
            slots.append({"index": len(slots), "track_id": tracks[i]["track_id"],
                          "condition": tracks[i]["condition"], "cycle": cyc,
                          "is_catch": False})
        prev = tracks[order[-1]]["condition"]
    n_catch = int(round(len(slots) * cfg.catch_rate))
    for i in rng.choice(len(slots), size=n_catch, replace=False):
        slots[int(i)]["is_catch"] = True
    payload = {"session_seed": seed, "config_hash": cfg.hash(), "conditions": names,
               "catch_condition": catch_condition(cfg).name, "tracks": tracks,
               "slot_plan": slots}
    payload["design_hash"] = hashlib.sha256(
        repr([(s["index"], s["track_id"], s["is_catch"]) for s in slots]).encode()
    ).hexdigest()[:16]
    return payload


def audit_design(cfg: Config, design: dict) -> dict:
    """Balance of the PLAN. What actually ran is audited from the log by `analysis`."""
    slots, names = design["slot_plan"], design["conditions"]
    first = [s for s in slots if s["cycle"] < 30]          # the part a session actually uses
    n = max(len(first), 1)
    pos = {c: float(np.mean([s["index"] / n for s in first if s["condition"] == c]))
           for c in names}
    longest, cur, prev = 0, 0, None
    for s in slots:
        cur = cur + 1 if s["condition"] == prev else 1
        prev = s["condition"]
        longest = max(longest, cur)
    per = {c: sum(1 for t in design["tracks"] if t["condition"] == c) for c in names}
    return {"n_tracks": len(design["tracks"]), "n_planned_slots": len(slots),
            "tracks_per_condition": per, "balanced": len(set(per.values())) == 1,
            "mean_serial_position": pos,
            "serial_position_spread": max(pos.values()) - min(pos.values()),
            "longest_condition_run_in_plan": longest,
            "catch_rate_planned": float(np.mean([s["is_catch"] for s in slots]))}


def simulate_session(cfg: Config, mode: str = "coherence", seed: int = 0) -> dict:
    """Run the real design, runner logic and trial generator against a simulated listener.

    No audio and no files. Returns trial counts by phase and the duration they imply, which is
    what the duration estimate is made of: the median of several of these, not an assumption
    about how long a track takes.
    """
    from .runner import Runner
    r = Runner(cfg, data_dir=None, auto=mode, seed=seed, dry=True)
    r.start_dry(code=f"SIM{seed}")
    r.main_block()
    return r.summary()


def duration_estimate(cfg: Config, n_sim: int = 8, mode: str = "coherence") -> dict:
    sims = [simulate_session(cfg, mode=mode, seed=i + 1) for i in range(n_sim)]
    sound = np.array([s["sound_s"] for s in sims])
    n_tr = np.array([s["n_trials"] for s in sims])
    main_min = (sound + n_tr * PER_TRIAL_OVERHEAD_S) / 60.0
    n_rests = np.array([s["n_rests"] for s in sims])
    mean_trial_s = float(np.sum(sound) / np.sum(n_tr)) + PER_TRIAL_OVERHEAD_S
    practice_min = cfg.practice_trials * mean_trial_s / 60.0
    fam = FAMILIARISE_MIN if cfg.familiarise else 0.0
    total = main_min + n_rests * REST_MIN + practice_min + fam + INSTRUCTIONS_MIN
    worst_practice = cfg.practice_rounds * practice_min
    return {"n_sim": n_sim, "mode": mode,
            "n_trials_median": float(np.median(n_tr)),
            "n_trials_range": (int(n_tr.min()), int(n_tr.max())),
            "n_catch_median": float(np.median([s["n_catch"] for s in sims])),
            "n_filler_median": float(np.median([s["n_filler"] for s in sims])),
            "trials_per_track_median": float(np.median([s["trials_per_track_median"]
                                                         for s in sims])),
            "main_minutes": float(np.median(main_min)),
            "rest_minutes": float(np.median(n_rests) * REST_MIN),
            "practice_minutes": practice_min, "familiarise_minutes": fam,
            "instructions_minutes": INSTRUCTIONS_MIN,
            "total_minutes": float(np.median(total)),
            "p90_total_minutes": float(np.percentile(total, 90)),
            "worst_practice_total_minutes": float(np.percentile(total, 90))
            + worst_practice - practice_min,
            "mean_trial_s": mean_trial_s}
