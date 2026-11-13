"""From a session folder to thresholds, with every check a reviewer could ask for.

Per listener
------------
* threshold per track: the geometric mean of the last six reversals (the published rule),
  recomputed here by replaying the logged responses and checked against what was logged live;
* threshold per condition from a maximum-likelihood psychometric fit to every staircase trial
  of that condition (cumulative Gaussian on log delta, guess 0.5, lapse 0.02), with a parametric
  bootstrap 95% interval: uses all the data, not just six reversals;
* data quality: catch-trial accuracy, practice score, repeated trials and why, audio underflows,
  late starts, onset timing error, visual lag, the volume on every trial, interval bias, the effect of
  the ending's order (B first / A first), response times;
* the grouping index g(D) = (log th(D100) - log th(D)) / (log th(D100) - log th(D0)): 1 at D0,
  0 at D100, normalised the same way as Micheyl et al.'s d' curve is normalised for the figure;
* the control: log2 th(D0_late) - log2 th(D0), the cost of displacing the ending alone.

Across listeners: `group` stacks the summaries and gives means of log thresholds with standard
errors and the individual curves.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import optimize, stats

from .config import Config
from .logbook import read_session
from .staircase import Staircase

LAPSE = 0.02


# ----------------------------------------------------------------------------------------------
# psychometric fit
# ----------------------------------------------------------------------------------------------
def _p(delta, mu, sigma, lapse=LAPSE):
    return 0.5 + (0.5 - lapse) * stats.norm.cdf((np.log(delta) - mu) / sigma)


def fit_pf(delta: Sequence[float], correct: Sequence[bool], target_p: float,
           n_boot: int = 300, seed: int = 0) -> dict:
    d = np.asarray(delta, float)
    c = np.asarray(correct, bool)
    if len(d) < 10:
        return {"threshold_ms": None, "n": int(len(d))}

    def nll(x, d=d, c=c):
        mu, ls = x
        p = np.clip(_p(d, mu, math.exp(ls)), 1e-9, 1 - 1e-9)
        return -np.sum(np.where(c, np.log(p), np.log(1 - p)))

    def solve(d_, c_):
        x0 = [float(np.log(np.median(d_))), math.log(0.6)]
        r = optimize.minimize(nll, x0, args=(d_, c_), method="L-BFGS-B", bounds=bounds)
        return r.x

    # the slope is confined to what a track can estimate (0.15-2 ln units) and the midpoint to
    # within a factor e of the tested range, so a bootstrap replicate cannot run off to a bound
    bounds = [(math.log(d.min()) - 1.0, math.log(d.max()) + 1.0), (math.log(0.15), math.log(2.0))]
    z = stats.norm.ppf((target_p - 0.5) / (0.5 - LAPSE))
    mu, ls = solve(d, c)
    th = float(math.exp(mu + math.exp(ls) * z))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sim = rng.random(len(d)) < _p(d, mu, math.exp(ls))
        m2, l2 = solve(d, sim)
        boots.append(math.exp(m2 + math.exp(l2) * z))
    ci = [float(x) for x in np.percentile(boots, [2.5, 97.5])] if boots else None
    return {"threshold_ms": th, "ci95_ms": ci, "sigma": float(math.exp(ls)),
            "n": int(len(d)), "prop_correct": float(c.mean())}


# ----------------------------------------------------------------------------------------------
# one session
# ----------------------------------------------------------------------------------------------
def summarize(folder: Path, n_boot: int = 300) -> dict:
    folder = Path(folder)
    s = read_session(folder)
    meta = s["meta"]
    cfg = Config.from_dict(meta["config"])
    rows = [t["row"] for t in s["trials"]]
    ok = [r for r in rows if r["status"] == "ok"]
    main = [r for r in ok if r["phase"] == "main"]
    stair = [r for r in main if not r["is_catch"]]
    catch = [r for r in main if r["is_catch"]]
    flags: List[str] = []

    tracks = []
    for b in sorted({int(r["block"]) for r in stair}):
        rs = [r for r in stair if int(r["block"]) == b]
        st = Staircase.replay(cfg, [bool(r["correct"]) for r in rs])
        logged = [float(r["delta_ms"]) for r in rs]
        replayed = [x.delta_ms for x in st.steps]
        consistent = np.allclose(logged, replayed, rtol=0, atol=1e-9)
        if not consistent:
            flags.append(f"block {b}: logged deltas differ from the replayed staircase")
        a = st.audit()
        a.update(block=b, condition=rs[0]["condition"], consistent=bool(consistent))
        if not a["converged"]:
            flags.append(f"block {b} ({a['condition']}): track did not converge")
        if a["ceiling_in_threshold"]:
            flags.append(f"block {b} ({a['condition']}): ceiling trials inside the threshold")
        if a["floor_in_threshold"]:
            flags.append(f"block {b} ({a['condition']}): floor trials inside the threshold")
        tracks.append(a)

    conds = [c.name for c in cfg.conditions()]
    per = {}
    for name in conds:
        ths = [t["threshold_ms"] for t in tracks if t["condition"] == name and t["threshold_ms"]]
        rs = [r for r in stair if r["condition"] == name]
        fit = fit_pf([r["delta_ms"] for r in rs], [bool(r["correct"]) for r in rs], cfg.target_p,
                     n_boot=n_boot) if rs else {"threshold_ms": None, "n": 0}
        cs = [r for r in catch if r["condition"] == name]
        per[name] = {
            "context_ms": cfg.condition(name).context_ms,
            "ending_shift_ms": cfg.condition(name).ending_shift_ms,
            "threshold_ms": float(np.exp(np.mean(np.log(ths)))) if ths else None,
            "track_thresholds_ms": ths, "fit": fit,
            "catch_correct": int(sum(bool(r["correct"]) for r in cs)), "catch_n": len(cs),
            "rt_median_ms": float(np.median([r["rt_ms"] for r in rs if r["rt_ms"] is not None]))
            if rs else None,
        }

    def lt(n):
        v = per.get(n, {}).get("threshold_ms")
        return math.log(v) if v else None

    g = {}
    top, bot = lt("D100"), lt("D0")
    for name in conds:
        x = lt(name)
        if None not in (x, top, bot) and top != bot:
            g[name] = (top - x) / (top - bot)
    late = None
    if lt("D0_late") is not None and bot is not None:
        late = (lt("D0_late") - bot) / math.log(2.0)

    n_catch = len(catch)
    catch_acc = float(np.mean([bool(r["correct"]) for r in catch])) if catch else None
    easy = [r for r in catch if r["condition"] in ("D0", "D0_late")]
    easy_acc = float(np.mean([bool(r["correct"]) for r in easy])) if easy else None
    if catch_acc is not None and catch_acc < 0.75:
        flags.append(f"catch-trial accuracy {catch_acc:.0%} is below 75%")
    aborted: Dict[str, int] = {}
    for r in rows:
        if r["status"] != "ok":
            aborted[r["abort_reason"] or "?"] = aborted.get(r["abort_reason"] or "?", 0) + 1
    lat = [r["onset_error_ms"] for r in ok if r.get("onset_error_ms") is not None]
    under = sum(int(r.get("underflows") or 0) for r in ok)
    vols = sorted({r["volume"] for r in ok if r.get("volume") is not None})
    if len(vols) > 1:
        flags.append(f"system volume varied across valid trials: {vols}")
    if under:
        flags.append(f"{under} audio underflows in trials marked valid")
    resp1 = [r["response"] == 1 for r in stair]
    by_sign = {}
    for sgn, lab in ((1, "B first"), (-1, "A first")):
        rs = [r for r in stair if int(r["sign"]) == sgn]
        by_sign[lab] = {"n": len(rs), "prop_correct": float(np.mean([bool(r["correct"]) for r in rs]))
                        if rs else None}
    practice = [r for r in ok if r["phase"] == "practice"]
    if meta.get("practice_passed") is False:
        flags.append("practice criterion was not reached")
    if not meta.get("levels_calibrated", False):
        flags.append("levels are nominal (uncalibrated session)")

    return {
        "folder": str(folder), "code": meta.get("code"), "session_index": meta.get("session_index"),
        "status": meta.get("status"), "simulated": meta.get("simulated", False),
        "config_hash": meta.get("config_hash"), "order": meta.get("plan", {}).get("blocks"),
        "conditions": per, "tracks": tracks, "grouping_index": g,
        "late_control_octaves": late,
        "catch": {"n": n_catch, "accuracy": catch_acc, "accuracy_easy_conditions": easy_acc},
        "practice": {"n": len(practice), "accuracy": float(np.mean(
            [bool(r["correct"]) for r in practice])) if practice else None,
            "passed": meta.get("practice_passed")},
        "repeated_trials": aborted,
        "audio": {"onset_error_ms_median": float(np.median(lat)) if lat else None,
                  "onset_error_ms_range": [float(min(lat)), float(max(lat))] if lat else None,
                  "underflows_in_valid_trials": under},
        "volumes_seen": vols,
        "interval_bias_p_first": float(np.mean(resp1)) if resp1 else None,
        "by_order": by_sign,
        "n_trials": {"all_rows": len(rows), "valid": len(ok), "staircase": len(stair),
                     "catch": n_catch},
        "duration_min": (ok[-1]["t_session_s"] / 60.0) if ok else None,
        "flags": flags,
    }


def write_summary(folder: Path, n_boot: int = 300) -> dict:
    folder = Path(folder)
    summ = summarize(folder, n_boot=n_boot)
    (folder / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    with open(folder / "thresholds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "condition", "context_ms", "ending_shift_ms", "threshold_ms",
                    "fit_threshold_ms", "fit_ci_lo", "fit_ci_hi", "grouping_index", "catch_correct",
                    "catch_n"])
        for name, c in summ["conditions"].items():
            fit = c["fit"]
            ci = fit.get("ci95_ms") or [None, None]
            w.writerow([summ["code"], name, c["context_ms"], c["ending_shift_ms"],
                        c["threshold_ms"], fit.get("threshold_ms"), ci[0], ci[1],
                        summ["grouping_index"].get(name), c["catch_correct"], c["catch_n"]])
    return summ


def describe(summ: dict) -> str:
    out = [f"{summ['code']} session {summ['session_index']}  status: {summ['status']}"
           f"{'  (SIMULATED)' if summ['simulated'] else ''}"]
    out.append(f"  {'condition':<9} {'threshold':>10} {'fit [95% CI]':>22} {'g':>6} {'catch':>6}")
    for name, c in summ["conditions"].items():
        th = f"{c['threshold_ms']:.2f}" if c["threshold_ms"] else "--"
        fit = c["fit"]
        ft = "--"
        if fit.get("threshold_ms"):
            ci = fit.get("ci95_ms")
            ft = f"{fit['threshold_ms']:.2f}" + (f" [{ci[0]:.1f}, {ci[1]:.1f}]" if ci else "")
        g = summ["grouping_index"].get(name)
        out.append(f"  {name:<9} {th:>10} {ft:>22} {'' if g is None else f'{g:.2f}':>6} "
                   f"{c['catch_correct']}/{c['catch_n']:<4}")
    if summ["late_control_octaves"] is not None:
        out.append(f"  displaced ending alone (D0_late vs D0): "
                   f"{summ['late_control_octaves']:+.2f} octaves")
    cat = summ["catch"]
    if cat["accuracy"] is not None:
        out.append(f"  catch trials: {cat['accuracy']:.0%} of {cat['n']}")
    if summ["repeated_trials"]:
        out.append(f"  repeated trials: {summ['repeated_trials']}")
    a = summ["audio"]
    if a["onset_error_ms_median"] is not None:
        out.append(f"  audio: onset error {a['onset_error_ms_median']:+.3f} ms median, "
                   f"{a['underflows_in_valid_trials']} underflows; volumes {summ['volumes_seen']}")
    if summ["duration_min"] and not summ["simulated"]:
        out.append(f"  duration {summ['duration_min']:.1f} min")
    for f in summ["flags"]:
        out.append(f"  ! {f}")
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------
# across listeners
# ----------------------------------------------------------------------------------------------
def group(folders: Sequence[Path], out_dir: Optional[Path] = None) -> dict:
    sums = [summarize(Path(f), n_boot=0) for f in folders]
    names = list(sums[0]["conditions"]) if sums else []
    table = {}
    for name in names:
        logs = [math.log(s["conditions"][name]["threshold_ms"]) for s in sums
                if s["conditions"][name]["threshold_ms"]]
        gi = [s["grouping_index"][name] for s in sums if name in s["grouping_index"]]
        table[name] = {
            "n": len(logs),
            "geomean_ms": float(math.exp(np.mean(logs))) if logs else None,
            "log_sd": float(np.std(logs, ddof=1)) if len(logs) > 1 else None,
            "log_se": float(np.std(logs, ddof=1) / math.sqrt(len(logs))) if len(logs) > 1 else None,
            "grouping_index_mean": float(np.mean(gi)) if gi else None,
            "grouping_index_se": float(np.std(gi, ddof=1) / math.sqrt(len(gi)))
            if len(gi) > 1 else None,
        }
    res = {"n_listeners": len(sums), "codes": [s["code"] for s in sums], "conditions": table,
           "individual": {s["code"]: {n: s["conditions"][n]["threshold_ms"] for n in names}
                          for s in sums},
           "flags": {s["code"]: s["flags"] for s in sums if s["flags"]}}
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "group_summary.json").write_text(json.dumps(res, indent=1, default=float))
        with open(out_dir / "group_thresholds.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["code"] + names)
            for code, d in res["individual"].items():
                w.writerow([code] + [d[n] for n in names])
    return res
