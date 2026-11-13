"""Which track runs when, and how long the session takes. Same scheme as tshear.

Every condition contributes exactly one track per round, so each gets one track in each
1/tracks_per_condition of the session by construction. Within a round the order is chosen
greedily to even out serial position rather than left to chance, because the quantity of
interest is a RATIO of thresholds across conditions and a slow drift over the session would
otherwise land unevenly on them.

With two directions each round is split into one block per direction, and the order of the two
reverses every round: in, out | out, in | ... (ABBA). Over an even number of rounds both
directions sit at the same mean position in the session. Which direction opens is set by the
participant code and the session index, so it alternates from one session to the next and is
balanced across people rather than left to the seed.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import Config, conditions, validate

# Measured across three completed sessions (tcoh x2, tshear x1), excluding every pause of 30 s or
# more: the interval between trial starts ran 0.51-0.57 s beyond the sound itself.
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
    dirs = cfg.directions
    first_dir = None
    if len(dirs) == 1:
        for rnd in range(cfg.tracks_per_condition):
            order = list(names)
            rng.shuffle(order)
            for i, name in enumerate(order):
                tracks.append({"track_id": tid, "condition": name, "round_index": rnd,
                               "block_index": (tid // per_block),
                               "seed": int(rng.integers(0, 2 ** 31))})
                tid += 1
    else:
        flip = (int(hashlib.sha256(code.encode()).hexdigest(), 16) + int(session_index)) % 2
        opening = dirs[::-1] if flip else dirs
        first_dir = opening[0]
        of = {c.name: c.direction for c in d.conditions}
        block = 0
        for rnd in range(cfg.tracks_per_condition):
            for direction in (opening if rnd % 2 == 0 else opening[::-1]):
                order = [n for n in names if of[n] == direction]
                rng.shuffle(order)
                for name in order:
                    tracks.append({"track_id": tid, "condition": name, "round_index": rnd,
                                   "block_index": block, "direction": direction,
                                   "seed": int(rng.integers(0, 2 ** 31))})
                    tid += 1
                block += 1
        per_block = len(names) // len(dirs)

    # interleave the tracks of a block, honouring the run-length limit. Catch trials and probes
    # take slots from a track's allotment, so with probes on the allotment grows to match:
    # a track can still use all max_trials_per_track scored trials.
    per_track = cfg.max_trials_per_track
    if cfg.probe_rate:
        per_track = int(np.ceil(per_track * (1.0 + cfg.catch_rate + cfg.probe_rate)))
    slots, idx = [], 0
    by_block: Dict[int, List[dict]] = {}
    for t in tracks:
        by_block.setdefault(t["block_index"], []).append(t)
    for b in sorted(by_block):
        members = by_block[b]
        pool = {t["track_id"]: per_track for t in members}
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
    if cfg.probe_rate:
        # decoupled probes: never on a catch slot, spread at random like the catches
        free = [i for i, sl in enumerate(slots) if not sl["is_catch"]]
        for sl in slots:
            sl["is_probe"] = False
        n_probe = int(round(len(slots) * cfg.probe_rate))
        for i in rng.choice(free, size=min(n_probe, len(free)), replace=False):
            slots[int(i)]["is_probe"] = True

    payload = {"session_seed": seed, "config_hash": cfg.hash(), "conditions": names,
               "tracks": tracks, "n_blocks": len(by_block), "tracks_per_block": per_block,
               "slot_plan": slots}
    if first_dir is not None:
        payload["direction_order"] = [by_block[b][0]["direction"] for b in sorted(by_block)]
    key = [(s["index"], s["track_id"], s["is_catch"]) + ((s["is_probe"],) if cfg.probe_rate else ())
           for s in slots]
    payload["design_hash"] = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
    return payload


def expected_track_trials(cfg: Config) -> float:
    from tcoh.track import simulate
    return float(simulate(cfg, thresholds_ms=(3.0, 12.0, 40.0), sigmas=(0.6,),
                          n_runs=120, seed=1)["median_trials"])


def duration_estimate(cfg: Config) -> dict:
    d = validate(cfg)
    per_track = expected_track_trials(cfg)
    n_trials = d.n_tracks * per_track * (1.0 + cfg.catch_rate + cfg.probe_rate)
    sound_s = cfg.trial_ms() / 1000.0
    per_trial_s = sound_s + PER_TRIAL_OVERHEAD_S
    minutes = n_trials * per_trial_s / 60.0
    n_blocks = make_design(cfg, "DURATION", 1)["n_blocks"]
    n_breaks = max(0, (n_blocks - 1) // max(cfg.break_every, 1))
    n_rests = (max(0, int(n_trials // cfg.break_every_trials) - n_breaks)
               if cfg.break_every_trials else 0)
    breaks = n_breaks * BLOCK_BREAK_MIN + n_rests * REST_MIN
    practice = len(cfg.directions) * cfg.practice_trials * (sound_s + 2.5) / 60.0
    setup = 7.0
    return {"n_tracks": d.n_tracks, "n_blocks": n_blocks, "n_breaks": n_breaks,
            "n_rests": n_rests, "median_trials_per_track": per_track,
            "n_trials": int(round(n_trials)), "sound_per_trial_s": sound_s,
            "main_minutes": minutes, "practice_minutes": practice, "break_minutes": breaks,
            "setup_minutes": setup,
            "total_minutes": minutes + practice + breaks + setup,
            "worst_case_minutes": (d.n_tracks * cfg.max_trials_per_track
                                   * (1.0 + cfg.catch_rate + cfg.probe_rate) * per_trial_s / 60.0
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
    # the comparisons that matter are across lags WITHIN a direction; direction itself is blocked
    of = {c.name: c.direction for c in conditions(cfg)}
    by_dir: Dict[str, List[float]] = {}
    for c, v in mean_pos.items():
        by_dir.setdefault(of.get(c, "in"), []).append(v)
    within = max((max(v) - min(v) for v in by_dir.values()), default=0.0)
    return {"n_tracks": len(tracks), "n_planned_slots": len(slots),
            "tracks_per_condition": per_cond,
            "balanced_track_count": len(set(per_cond.values())) == 1,
            "mean_serial_position": mean_pos,
            "serial_position_spread": (max(mean_pos.values()) - min(mean_pos.values()))
            if mean_pos else 0.0,
            "serial_position_spread_within_direction": within,
            "direction_mean_position": {d: float(np.mean(v)) for d, v in by_dir.items()},
            "round_balance_spread": (max(mean_round.values()) - min(mean_round.values()))
            if mean_round else 0.0,
            "longest_condition_run": longest,
            "max_same_condition_run": cfg.max_same_condition_run,
            "run_constraint_respected": longest <= cfg.max_same_condition_run,
            "catch_rate_planned": float(np.mean([s["is_catch"] for s in slots])) if slots else 0.0,
            "probe_rate_planned": float(np.mean([s.get("is_probe", False) for s in slots]))
            if slots else 0.0,
            "catch_rate_configured": cfg.catch_rate}


# ---------------------------------------------------------------------------- timing audit
def slot_rng(design: dict, slot_index: int, seed: int = 0) -> np.random.Generator:
    """The generator the runner gives slot `slot_index` (Runner._slot_rng): the timing of every
    trial in a session is fixed by its seed before anyone listens."""
    return np.random.default_rng([int(design["session_seed"]), int(seed), 0x5E, int(slot_index)])


def slot_condition(cfg: Config, design: dict, slot: dict):
    """Which condition a slot plays, as the runner decides it: a probe, a catch trial, or the
    track's own condition."""
    from .config import probe_conditions
    conds = {c.name: c for c in conditions(cfg)}
    cond = conds[{t["track_id"]: t for t in design["tracks"]}[slot["track_id"]]["condition"]]
    probes = {c.direction: c for c in probe_conditions(cfg)}
    if slot.get("is_probe") and cond.direction in probes:
        return probes[cond.direction], cfg.delta_max_ms
    if slot.get("is_catch"):
        catch = [c for c in reversed(conditions(cfg)) if c.kind == "pair" and c.stationary
                 and c.direction == cond.direction and cfg.catch_at_lag_ms is not None
                 and abs(c.lag_ms - cfg.catch_at_lag_ms) < 1e-9]
        return (catch[0] if catch else cond), cfg.catch_delta_ms
    return cond, cfg.delta_start_ms


def audit_timing(cfg: Config, design: dict, seed: int = 0) -> dict:
    """Every interval of every planned slot, drawn from the session seed exactly as the runner
    will draw it: how many draws the screen threw away and why, and what was kept.

    The reference timing of a slot does not depend on delta, so this is the session's timing,
    not an estimate of it. Slots a finished track never reaches are drawn too; they are part of
    the plan.
    """
    from .config import scatter_geometry, timing_faults
    from .stimulus import build_trial
    if cfg.timing != "scatter":
        return {"ok": True, "applies": False}
    g = scatter_geometry(cfg)["min_ioi_ms"]
    log: List[str] = []
    gaps, worst, kept_bad = [], [], 0
    n_iv = 0
    for slot in design["slot_plan"]:
        cond, delta = slot_condition(cfg, design, slot)
        tr = build_trial(cfg, cond, delta, slot_rng(design, slot["index"], seed),
                         timing_log=log)
        for iv in (tr.first, tr.second):
            refs = [iv.ref_ms] + ([iv.ref_b_ms] if iv.ref_b_ms is not None else [])
            for r in refs:
                n_iv += 1
                d = np.diff(r)
                gaps.append(d)
                worst.append(d.min())
                kept_bad += bool(timing_faults(cfg, r)) or bool((d < g - 1e-6).any())
    allg = np.concatenate(gaps)
    by = {k: sum(k in b for b in log) for k in sorted({x for b in log for x in b})}
    return {"ok": kept_bad == 0, "applies": True, "intervals": n_iv,
            "draws_rejected": len(log),
            "rejected_by_rule": by,
            "share_redrawn": len(log) / (len(log) + n_iv) if log else 0.0,
            "kept_with_a_fault": kept_bad,
            "gap_mean_ms": float(allg.mean()), "gap_sd_ms": float(allg.std()),
            "gap_min_ms": float(allg.min()), "gap_max_ms": float(allg.max()),
            "rate_hz": 1000.0 / float(allg.mean())}


def describe_timing_audit(a: dict) -> str:
    if not a.get("applies"):
        return "  timing     grid timing: nothing to screen"
    rules = ", ".join(f"{k} {v}" for k, v in a["rejected_by_rule"].items()) or "none"
    return (f"  timing     {a['intervals']} intervals drawn from the seed; "
            f"{100 * a['share_redrawn']:.0f}% of draws thrown away and redrawn (by rule: {rules}); "
            f"{'every kept interval passes' if a['ok'] else str(a['kept_with_a_fault']) + ' KEPT WITH A FAULT'}\n"
            f"             kept gaps {a['gap_mean_ms']:.0f} +- {a['gap_sd_ms']:.0f} ms "
            f"({a['gap_min_ms']:.0f}-{a['gap_max_ms']:.0f}), {a['rate_hz']:.2f} Hz")
