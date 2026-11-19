"""Thresholds, CMR and the shear contrasts for one session; a table across sessions.

CMR is threshold(random) minus threshold(comodulated). Random baselines exist at sync and at
ascending 20 ms; zigzag 20 shares the ascending-20 baseline (same set of onset offsets), and
ascending 10 uses the mean of the two. The key contrasts do not need a baseline at all:
co_ascending_20 - co_sync (does shear cost release?) and co_zigzag_20 - co_ascending_20 (does
order matter?), both in dB of tone level (positive = harder).

Patterns (frozen, reordered; see config.py): each has its own random baseline at the same step.
Reported as well: the drop in CMR from sync to each step (positive = release lost) and, where both
exist at a step, frozen minus re-ordered CMR (positive = the repeating pattern kept more release),
with the same contrast on the comodulated thresholds alone.

The top of the level range (2026-11-15). A track cannot go above max_db, so a listener whose
threshold is near or above it is not measured like the others. Two things are reported:

    at_ceiling        conditions with a track on which the rule asked for more than max_db
                      CEILING_CALLS times or more (procedure.py): the threshold is at the top of
                      the range. The fitted value is kept.
    censored_at_max   conditions whose fitted threshold is above max_db, the highest level ever
                      played. Such a value is an extrapolation along the shared slope, so the
                      threshold is entered as max_db (a lower bound); the fitted value is kept
                      beside it. CMR and every contrast that uses it are then bounds, not estimates.

Both are flagged. Also flagged: a session that is not complete, a practice gate that was not
passed, and a fit that did not converge.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from tsync.logbook import read_jsonl

from .config import Condition, Config
from .procedure import CEILING_CALLS, Track, fit


def summarize(folder: Path) -> dict:
    folder = Path(folder)
    meta = json.loads((folder / "session.json").read_text())
    cfg = Config.from_dict(meta["config"])
    rows = [r["row"] for r in read_jsonl(folder / "trials.jsonl")]
    main = [r for r in rows if r["phase"] == "main" and r["status"] == "ok"]
    data: Dict[str, tuple] = {}
    for r in main:
        L, C = data.setdefault(r["condition"], ([], []))
        L.append(r["level_db"])
        C.append(bool(r["correct"]))
    f = fit(data, lapse=cfg.lapse)
    th = f["thresholds_db"]
    censored = {k: v for k, v in th.items() if v > cfg.max_db}      # the fitted values, kept
    for k in censored:                                             # above every level played:
        th[k] = float(cfg.max_db)                                  # a lower bound from here on
    # with the tone's band randomised, a threshold per band as well (fewer trials each)
    by_band: Dict[str, float] = {}
    if len({r.get("tone_hz") for r in main}) > 1:
        bd: Dict[str, tuple] = {}
        for r in main:
            L, C = bd.setdefault(f"{r['condition']}@{r['tone_hz']:g}", ([], []))
            L.append(r["level_db"])
            C.append(bool(r["correct"]))
        by_band = fit(bd, lapse=cfg.lapse)["thresholds_db"]
    rev, calls, at_max = {}, {}, {}
    for tid in sorted({int(r["track"]) for r in main}):
        rs = [r for r in main if int(r["track"]) == tid]
        c = next(c for c in cfg.condition_list() if c.name == rs[0]["condition"])
        t = Track.replay(cfg, c, tid, [r["correct"] for r in rs])
        rev.setdefault(c.name, []).append(t.reversal_threshold())
        calls.setdefault(c.name, []).append(t.calls_above_max)
        at_max.setdefault(c.name, []).append(sum(1 for r in rs if r["level_db"] >= cfg.max_db))
    at_ceiling = {k: max(v) for k, v in calls.items() if max(v) >= CEILING_CALLS}

    def g(k):
        return th.get(k)

    def d(a, b):
        return None if g(a) is None or g(b) is None else g(a) - g(b)

    # random-envelope baseline at any step: measured where available, otherwise linear in dT
    # between the measured random conditions of the same arrangement (sync counts as dT = 0)
    def rand_at(arr, dt):
        pts = sorted({(0.0, th["rand_sync_0"])} if "rand_sync_0" in th else set())
        pts = sorted(set(pts) | {(c.dt_ms, th[c.name]) for c in cfg.condition_list()
                                 if c.envelope == "rand" and c.arrangement == arr and c.name in th})
        if not pts:
            return None
        xs, ys = zip(*pts)
        return float(np.interp(dt, xs, ys))

    cmr, contrasts = {}, {}
    co_sync = g("co_sync_0")
    for c in cfg.condition_list():
        if c.envelope != "co" or c.name not in th:
            continue
        key = f"{c.arrangement}_{c.dt_ms:g}"
        r = rand_at("sync" if c.dt_ms == 0 else c.arrangement, c.dt_ms)
        cmr[key] = None if r is None else r - th[c.name]
        if c.dt_ms > 0 and co_sync is not None:
            contrasts[f"shear_cost_{key}_db"] = th[c.name] - co_sync
    for c in cfg.condition_list():
        if c.envelope == "co" and c.arrangement == "zigzag" and c.name in th:
            a = g(f"co_ascending_{c.dt_ms:g}")
            if a is not None:
                contrasts[f"order_cost_zigzag_vs_ascending_{c.dt_ms:g}_db"] = th[c.name] - a
    for key, v in list(cmr.items()):
        if key != "sync_0" and v is not None and cmr.get("sync_0") is not None:
            contrasts[f"cmr_drop_{key}_db"] = cmr["sync_0"] - v
    for c in cfg.condition_list():
        if c.envelope == "co" and c.arrangement == "reordered":
            k_f, k_r = f"frozen_{c.dt_ms:g}", f"reordered_{c.dt_ms:g}"
            if cmr.get(k_f) is not None and cmr.get(k_r) is not None:
                contrasts[f"learning_cmr_frozen_minus_reordered_{c.dt_ms:g}_db"] = cmr[k_f] - cmr[k_r]
            if g(f"co_frozen_{c.dt_ms:g}") is not None and g(c.name) is not None:
                contrasts[f"learning_co_reordered_minus_frozen_{c.dt_ms:g}_db"] = \
                    th[c.name] - th[f"co_frozen_{c.dt_ms:g}"]
    for c in cfg.condition_list():
        if c.envelope == "rand" and c.dt_ms > 0 and c.name in th and "rand_sync_0" in th:
            contrasts[f"random_baseline_shift_{c.arrangement}_{c.dt_ms:g}_db"] = \
                th[c.name] - th["rand_sync_0"]
    cmr.setdefault("sync_0", None)
    # the standard CMR reference: the tone's band alone (no flanking bands)
    cmr_ref = {}
    if "ref_sync_0" in th:
        for c in cfg.condition_list():
            if c.envelope in ("co", "rand") and c.name in th:
                cmr_ref[c.name] = th["ref_sync_0"] - th[c.name]
    ratings: Dict[str, List[int]] = {}
    for r in rows:
        if r["phase"] == "rating" and r["status"] == "ok":
            ratings.setdefault(f"{r['arrangement']}_{r['dt_ms']:g}", []).append(r["rating"])
    prac = [r for r in rows if r["phase"] == "practice" and r["status"] == "ok"]
    n_right = sum(bool(r["correct"]) for r in prac)
    flags = []
    planned = cfg.n_tracks * cfg.trials_per_track
    if meta.get("status") != "complete":
        flags.append(f"the session is not complete (status {meta.get('status')}): {len(main)} of "
                     f"{planned} main trials")
    need = int(cfg.practice_min_correct)
    gate = None if need == 0 else bool(len(prac) >= len(cfg.practice_levels_db) and n_right >= need)
    if gate is False:
        flags.append(f"practice gate not passed: {n_right} of {len(prac)} correct ({need} needed)")
    for k, n in at_ceiling.items():
        flags.append(f"{k} is at ceiling: the rule asked for more than {cfg.max_db:g} dB on {n} "
                     f"trials ({max(at_max[k])} trials were played at {cfg.max_db:g} dB)")
    for k, v in censored.items():
        flags.append(f"{k}: the fitted threshold ({v:.1f} dB) is above the highest level played; it "
                     f"is entered as {cfg.max_db:g} dB, a lower bound, and what is computed from it "
                     f"is a bound too")
    if f.get("converged") is False:
        flags.append("the threshold fit did not converge")
    if cmr["sync_0"] is not None and cmr["sync_0"] < 3:
        flags.append(f"CMR at sync is {cmr['sync_0']:.1f} dB (< 3 dB): little across-band release")
    vols = sorted({r["volume"] for r in rows if r["status"] == "ok" and r["volume"] is not None})
    if len(vols) > 1:
        flags.append(f"system volume varied: {vols}")
    under = sum(int(r["underflows"] or 0) for r in rows if r["status"] == "ok")
    if under:
        flags.append(f"{under} audio underflows in valid trials")
    if meta["config"].get("burst_level", "equalised") != "natural":
        flags.append("bursts were equalised in level (the default before 2026-11-14): thresholds "
                     "and CMR are not comparable with natural-level sessions")
    rep = {}
    for r in rows:
        if r["status"] != "ok":
            rep[r["reason"]] = rep.get(r["reason"], 0) + 1
    out = {"code": meta.get("code"), "status": meta.get("status"), "thresholds_db": th,
           "slope_sigma_db": f["sigma_db"], "reversal_thresholds_db": rev, "cmr_db": cmr, "cmr_vs_band_alone_db": cmr_ref, "thresholds_by_band_db": by_band,
           "contrasts_db": contrasts,
           "ratings_mean": {k: float(np.mean(v)) for k, v in ratings.items()},
           "practice_correct": f"{n_right}/{len(prac)}", "practice_gate_passed": gate,
           "main_trials": [len(main), planned], "max_db": cfg.max_db,
           "calls_above_max": calls, "at_ceiling": sorted(at_ceiling),
           "censored_at_max": {k: float(v) for k, v in censored.items()},
           "repeated": rep, "volumes": vols, "flags": flags,
           "duration_min": rows[-1]["t_s"] / 60 if rows else None}
    (folder / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    with open(folder / "thresholds.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["code", "condition", "threshold_db", "reversal_thresholds_db"])
        for k, v in th.items():
            w.writerow([out["code"], k, v, " ".join(f"{x:.1f}" for x in rev.get(k, []) if x)])
    return out


def describe(s: dict) -> str:
    f = lambda v, p=1: "--" if v is None else f"{v:+.{p}f}" if isinstance(v, float) else str(v)
    lines = [f"{s['code']}  status {s['status']}  slope {f(s['slope_sigma_db'])} dB",
             "  thresholds (dB SPL, 70.7%):"]
    cens, ceil_ = s.get("censored_at_max") or {}, s.get("at_ceiling") or []
    for k, v in s["thresholds_db"].items():
        mark = (f"   AT LEAST this (fit {cens[k]:.1f})" if k in cens else
                "   at ceiling" if k in ceil_ else "")
        lines.append(f"    {k:<20} {v:6.1f}   reversals {s['reversal_thresholds_db'].get(k)}{mark}")
    if s.get("cmr_vs_band_alone_db"):
        lines.append("  release vs the tone's band alone (dB, + = easier than alone): " + ", ".join(
            f"{k} {f(v)}" for k, v in s["cmr_vs_band_alone_db"].items()))
    if s.get("thresholds_by_band_db"):
        lines.append("  by tone band (dB SPL): " + ", ".join(
            f"{k} {v:.1f}" for k, v in s["thresholds_by_band_db"].items()))
    lines.append("  CMR (dB): " + ", ".join(f"{k} {f(v)}" for k, v in s["cmr_db"].items()))
    lines.append("  contrasts (dB, + = harder): " +
                 ", ".join(f"{k} {f(v)}" for k, v in s["contrasts_db"].items()))
    if s["ratings_mean"]:
        lines.append("  ratings (1 one sound .. 5 separate): " +
                     ", ".join(f"{k} {v:.1f}" for k, v in s["ratings_mean"].items()))
    lines.append(f"  practice {s['practice_correct']}; repeated {s['repeated'] or 'none'}; "
                 f"volumes {s['volumes']}")
    if s["duration_min"]:
        lines.append(f"  duration {s['duration_min']:.1f} min")
    lines += [f"  ! {x}" for x in s["flags"]]
    return "\n".join(lines)
