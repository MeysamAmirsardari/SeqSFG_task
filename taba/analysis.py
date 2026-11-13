"""From key presses to percept statistics. Pure functions, so they are tested without a listener.

A report is a list of (time_s, percept) with time from the sequence's first sound (DAC time).
The listener presses a key only when the percept CHANGES, so the percept is undefined before the
first press and held between presses.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SEP, TOG = "separate", "together"


def clean_reports(events: Sequence[Tuple[float, str]], duration_s: float) -> List[Tuple[float, str]]:
    """Sorted, within the sequence, repeated presses of the same key collapsed."""
    out: List[Tuple[float, str]] = []
    for t, p in sorted((float(t), p) for t, p in events if 0.0 <= float(t) <= duration_s):
        if not out or out[-1][1] != p:
            out.append((t, p))
    return out


def trial_stats(events: Sequence[Tuple[float, str]], duration_s: float,
                late_from_s: Optional[float] = None) -> dict:
    ev = clean_reports(events, duration_s)
    if not ev:
        return {"n_reports": 0, "first_percept": None, "first_latency_s": None,
                "prop_separate": None, "prop_separate_late": None, "switches": 0,
                "switch_rate_per_min": None, "mean_phase_s": {SEP: None, TOG: None}}
    bounds = [t for t, _ in ev] + [duration_s]
    phases = [(ev[i][1], bounds[i], bounds[i + 1]) for i in range(len(ev))]
    covered = duration_s - ev[0][0]
    sep = sum(b - a for p, a, b in phases if p == SEP)
    late_from = duration_s / 3.0 if late_from_s is None else late_from_s
    late_sep = sum(max(0.0, b - max(a, late_from)) for p, a, b in phases if p == SEP)
    late_cov = sum(max(0.0, b - max(a, late_from)) for _, a, b in phases)
    complete = phases[1:-1]          # the first starts at an unknown time, the last is cut off
    mean = {}
    for p in (SEP, TOG):
        d = [b - a for q, a, b in complete if q == p]
        mean[p] = float(np.mean(d)) if d else None
    sw = len(ev) - 1
    return {"n_reports": len(ev), "first_percept": ev[0][1], "first_latency_s": ev[0][0],
            "prop_separate": sep / covered if covered > 0 else None,
            "prop_separate_late": late_sep / late_cov if late_cov > 0 else None,
            "switches": sw, "switch_rate_per_min": sw / covered * 60.0 if covered > 0 else None,
            "mean_phase_s": mean}


def summarize(folder: Path) -> dict:
    folder = Path(folder)
    meta = json.loads((folder / "session.json").read_text())
    rows = [json.loads(l) for l in open(folder / "trials.jsonl") if l.strip()]
    main = [r for r in rows if r["phase"] == "main" and r["status"] == "ok"]
    per: Dict[str, dict] = {}
    for r in main:
        per.setdefault(r["condition"], []).append(r["stats"])
    out = {}
    for c, ss in per.items():
        def m(key):
            v = [s[key] for s in ss if s.get(key) is not None]
            return float(np.mean(v)) if v else None
        firsts = [s["first_percept"] for s in ss if s["first_percept"]]
        out[c] = {"n": len(ss), "prop_separate": m("prop_separate"),
                  "prop_separate_late": m("prop_separate_late"),
                  "first_separate": (sum(f == SEP for f in firsts) / len(firsts)) if firsts else None,
                  "first_latency_s": m("first_latency_s"),
                  "switch_rate_per_min": m("switch_rate_per_min")}
    res = {"code": meta.get("code"), "status": meta.get("status"), "conditions": out,
           "silent_trials": sum(1 for r in main if r["stats"]["n_reports"] == 0)}
    (folder / "summary.json").write_text(json.dumps(res, indent=1))
    with open(folder / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "n", "prop_separate", "prop_separate_late", "first_separate",
                    "first_latency_s", "switch_rate_per_min"])
        for c, v in out.items():
            w.writerow([c] + [v[k] for k in ("n", "prop_separate", "prop_separate_late",
                                             "first_separate", "first_latency_s",
                                             "switch_rate_per_min")])
    return res


def describe(res: dict) -> str:
    lines = [f"{res['code']}  status {res['status']}",
             f"  {'condition':<13}{'n':>3}{'P(sep)':>9}{'late':>7}{'first sep':>11}"
             f"{'1st RT s':>10}{'switch/min':>12}"]
    f = lambda v, p=2: "--" if v is None else f"{v:.{p}f}"
    for c, v in res["conditions"].items():
        lines.append(f"  {c:<13}{v['n']:>3}{f(v['prop_separate']):>9}"
                     f"{f(v['prop_separate_late']):>7}{f(v['first_separate']):>11}"
                     f"{f(v['first_latency_s'], 1):>10}{f(v['switch_rate_per_min'], 1):>12}")
    if res["silent_trials"]:
        lines.append(f"  ! {res['silent_trials']} sequences with no report at all")
    return "\n".join(lines)
