"""Thresholds, CMR and the shear contrasts for one session; a table across sessions.

CMR is threshold(random) minus threshold(comodulated). Random baselines exist at sync and at
ascending 20 ms; zigzag 20 shares the ascending-20 baseline (same set of onset offsets), and
ascending 10 uses the mean of the two. The key contrasts do not need a baseline at all:
co_ascending_20 - co_sync (does shear cost release?) and co_zigzag_20 - co_ascending_20 (does
order matter?), both in dB of tone level (positive = harder).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from tsync.logbook import read_jsonl

from .config import Condition, Config
from .procedure import Track, fit


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
    # with the tone's band randomised, a threshold per band as well (fewer trials each)
    by_band: Dict[str, float] = {}
    if len({r.get("tone_hz") for r in main}) > 1:
        bd: Dict[str, tuple] = {}
        for r in main:
            L, C = bd.setdefault(f"{r['condition']}@{r['tone_hz']:g}", ([], []))
            L.append(r["level_db"])
            C.append(bool(r["correct"]))
        by_band = fit(bd, lapse=cfg.lapse)["thresholds_db"]
    rev = {}
    for tid in sorted({int(r["track"]) for r in main}):
        rs = [r for r in main if int(r["track"]) == tid]
        c = next(c for c in cfg.condition_list() if c.name == rs[0]["condition"])
        t = Track.replay(cfg, c, tid, [r["correct"] for r in rs])
        rev.setdefault(c.name, []).append(t.reversal_threshold())

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
    flags = []
    if cmr["sync_0"] is not None and cmr["sync_0"] < 3:
        flags.append(f"CMR at sync is {cmr['sync_0']:.1f} dB (< 3 dB): little across-band release")
    vols = sorted({r["volume"] for r in rows if r["status"] == "ok" and r["volume"] is not None})
    if len(vols) > 1:
        flags.append(f"system volume varied: {vols}")
    under = sum(int(r["underflows"] or 0) for r in rows if r["status"] == "ok")
    if under:
        flags.append(f"{under} audio underflows in valid trials")
    rep = {}
    for r in rows:
        if r["status"] != "ok":
            rep[r["reason"]] = rep.get(r["reason"], 0) + 1
    out = {"code": meta.get("code"), "status": meta.get("status"), "thresholds_db": th,
           "slope_sigma_db": f["sigma_db"], "reversal_thresholds_db": rev, "cmr_db": cmr, "cmr_vs_band_alone_db": cmr_ref, "thresholds_by_band_db": by_band,
           "contrasts_db": contrasts,
           "ratings_mean": {k: float(np.mean(v)) for k, v in ratings.items()},
           "practice_correct": f"{sum(bool(r['correct']) for r in prac)}/{len(prac)}",
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
    for k, v in s["thresholds_db"].items():
        lines.append(f"    {k:<20} {v:6.1f}   reversals {s['reversal_thresholds_db'].get(k)}")
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
