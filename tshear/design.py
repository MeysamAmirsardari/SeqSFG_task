"""Which track runs when, and how long the session takes.

Every condition contributes exactly one track per round, so each gets one track in each
1/tracks_per_condition of the session by construction. Within a round the order is chosen
greedily to even out serial position rather than left to chance, because the quantity of
interest is a RATIO of thresholds across conditions and a slow drift over the session would
otherwise land unevenly on them.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import Config, validate

# Measured on the two-tone task across two completed sessions, excluding every pause of 30 s or
# more: the interval between trial starts ran 0.55 s beyond the duration of the sound itself.
PER_TRIAL_OVERHEAD_S = 0.55
BLOCK_BREAK_MIN = 1.5
REST_MIN = 0.75


def session_seed(cfg: Config, code: str, session_index: int) -> int:
    h = hashlib.sha256(f"{cfg.hash()}|{code}|{session_index}".encode()).hexdigest()
    return int(h[:12], 16)


def make_design(cfg: Config, code: str, session_index: int) -> dict:
    d = validate(cfg)
    names = [c.name for c in d.conditions]
    per_block = cfg.tracks_per_block or len(names)
    seed = session_seed(cfg, code, session_index)
    rng = np.random.default_rng(seed)

    tracks, tid = [], 0
    for rnd in range(cfg.tracks_per_condition):
        order = list(names)
        rng.shuffle(order)
        for i, name in enumerate(order):
            tracks.append({"track_id": tid, "condition": name, "round_index": rnd,
                           "block_index": (tid // per_block),
                           "seed": int(rng.integers(0, 2 ** 31))})
            tid += 1

    # interleave the tracks of a block, honouring the run-length limit
    slots, idx = [], 0
    by_block: Dict[int, List[dict]] = {}
    for t in tracks:
        by_block.setdefault(t["block_index"], []).append(t)
    for b in sorted(by_block):
        members = by_block[b]
        pool = {t["track_id"]: cfg.max_trials_per_track for t in members}
        last, run = None, 0
        while any(v > 0 for v in pool.values()):
            avail = [t for t in members if pool[t["track_id"]] > 0]
            if run >= cfg.max_same_condition_run:
                other = [t for t in avail if t["condition"] != last]
                if other:
                    avail = other
            pick = max(avail, key=lambda t: (pool[t["track_id"]], -t["track_id"]))
            pool[pick["track_id"]] -= 1
            run = run + 1 if pick["condition"] == last else 1
            last = pick["condition"]
            slots.append({"index": idx, "track_id": pick["track_id"],
                          "condition": pick["condition"], "block_index": b,
                          "is_catch": False})
            idx += 1

    n_catch = int(round(len(slots) * cfg.catch_rate))
    if n_catch:
        for i in sorted(rng.choice(len(slots), size=min(n_catch, len(slots)), replace=False)):
            slots[int(i)]["is_catch"] = True

    payload = {"session_seed": seed, "config_hash": cfg.hash(), "conditions": names,
               "tracks": tracks, "n_blocks": len(by_block), "tracks_per_block": per_block,
               "slot_plan": slots}
    payload["design_hash"] = hashlib.sha256(
        repr([(s["index"], s["track_id"], s["is_catch"]) for s in slots]).encode()
    ).hexdigest()[:16]
    return payload


def expected_track_trials(cfg: Config) -> float:
    from tcoh.track import simulate
    return float(simulate(cfg, thresholds_ms=(3.0, 8.0, 14.0), sigmas=(0.6,),
                          n_runs=120, seed=1)["median_trials"])


def duration_estimate(cfg: Config) -> dict:
    d = validate(cfg)
    per_track = expected_track_trials(cfg)
    n_trials = d.n_tracks * per_track * (1.0 + cfg.catch_rate)
    sound_s = d.trial_ms / 1000.0
    per_trial_s = sound_s + PER_TRIAL_OVERHEAD_S
    minutes = n_trials * per_trial_s / 60.0
    n_blocks = make_design(cfg, "DURATION", 1)["n_blocks"]
    n_breaks = max(0, (n_blocks - 1) // max(cfg.break_every, 1))
    n_rests = (max(0, int(n_trials // cfg.break_every_trials) - n_breaks)
               if cfg.break_every_trials else 0)
    breaks = n_breaks * BLOCK_BREAK_MIN + n_rests * REST_MIN
    practice = cfg.practice_trials * (sound_s + 2.5) / 60.0
    setup = 7.0
    return {"n_tracks": d.n_tracks, "n_blocks": n_blocks, "n_breaks": n_breaks,
            "n_rests": n_rests, "median_trials_per_track": per_track,
            "n_trials": int(round(n_trials)), "sound_per_trial_s": sound_s,
            "main_minutes": minutes, "practice_minutes": practice, "break_minutes": breaks,
            "setup_minutes": setup,
            "total_minutes": minutes + practice + breaks + setup,
            "worst_case_minutes": (d.n_tracks * cfg.max_trials_per_track
                                   * (1.0 + cfg.catch_rate) * per_trial_s / 60.0
                                   + practice + breaks + setup)}


def audit_design(cfg: Config, design: dict) -> dict:
    slots, tracks = design["slot_plan"], design["tracks"]
    names = design["conditions"]
    n = max(len(slots), 1)
    pos: Dict[str, List[float]] = {c: [] for c in names}
    rounds: Dict[str, List[int]] = {c: [] for c in names}
    for s in slots:
        pos[s["condition"]].append(s["index"] / n)
    for t in tracks:
        rounds[t["condition"]].append(t["round_index"])
    mean_pos = {c: float(np.mean(v)) for c, v in pos.items() if v}
    mean_round = {c: float(np.mean(v)) for c, v in rounds.items() if v}
    longest, cur, prev = 0, 0, None
    for s in slots:
        cur = cur + 1 if s["condition"] == prev else 1
        prev = s["condition"]
        longest = max(longest, cur)
    per_cond = {c: sum(1 for t in tracks if t["condition"] == c) for c in names}
    return {"n_tracks": len(tracks), "n_planned_slots": len(slots),
            "tracks_per_condition": per_cond,
            "balanced_track_count": len(set(per_cond.values())) == 1,
            "mean_serial_position": mean_pos,
            "serial_position_spread": (max(mean_pos.values()) - min(mean_pos.values()))
            if mean_pos else 0.0,
            "round_balance_spread": (max(mean_round.values()) - min(mean_round.values()))
            if mean_round else 0.0,
            "longest_condition_run": longest,
            "max_same_condition_run": cfg.max_same_condition_run,
            "run_constraint_respected": longest <= cfg.max_same_condition_run,
            "catch_rate_planned": float(np.mean([s["is_catch"] for s in slots])) if slots else 0.0,
            "catch_rate_configured": cfg.catch_rate}
