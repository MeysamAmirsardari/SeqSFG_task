"""From trials to thresholds, to a binding index, to decision weights, to a bound.

The measures, in the order they are built
-----------------------------------------
threshold   per track, the mean SRS (dB) at the last four reversals at the 2 dB step. A track
            whose averaged reversals touch the cap (SRS +9.5 dB, dL 12 dB) is CEILING-LIMITED:
            the listener could not do it with any increment the stimulus allows, so the number
            is a lower bound. It enters means at the cap and is always flagged.

BI(s)       the binding index at step s: threshold(B, s) - threshold(A, s), in dB. Positive when
            the timbre comparison (A) is easier than following the target (B): the target is
            bound into the figure. The primary test (H1) is that BI falls as the shear grows. A
            listener for whom shear changes nothing has a flat BI, whatever its level.

w(s)        the timbre weight: a probit of each response on the difference between the two
            intervals in each cue, pooled over A and B trials at one step, and
            w = b_timbre / (b_timbre + b_own). 1 means the listener read only the timbre of the
            probe, 0 only the target's own stream. The roves make both cues vary at random on
            every trial, so this is estimated from every scored trial, not from two thresholds.

s50         THE BOUND: the shear at which w has fallen half-way from w(0) to w(max), the same
            definition the model's half-way point uses. Estimated on the group-mean curve with a
            bootstrap over listeners. A second estimate fits the model's own curve stretched in
            time, w(s) = a + b * rho(s / alpha): alpha = 1 means binding decays exactly as the
            model says, 2 that it persists to twice the shear.

Why the bound comes from the weights and not from BI
----------------------------------------------------
BI saturates. Once a listener stops using the cue a trial type preserves, its threshold
climbs to the shortcut ceiling and stays there, however much further the binding state moves.
In two simulated groups whose binding follows the model exactly (true half-way 18.2 ms), the
half-way point of BI landed at 23 and 30 ms and that of w at 17 and 18 ms. BI is the robust,
conventional demonstration that the shear changes cue use; w is the measurement of how far.

What is never done
-------------------
No curve is smoothed or forced monotone. No interval is computed from one or two tracks: spread
is across LISTENERS, n = the number of people. The model is drawn on its own axis, labelled as a
comparison, and its predictions are fixed before the data (`model.predictions`).
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, stats

from tcoh.session import read_json, read_trials

from .config import Condition, Config, conditions, delta_db, validate
from .observer import ceilings
from .track import Track

MIN_SWING_DB = 2.0
MIN_SWING_W = 0.15


def _f(r, k, default=np.nan):
    try:
        return float(r.get(k, ""))
    except (TypeError, ValueError):
        return default


def _i(r, k, default=-1):
    try:
        return int(float(r.get(k, "")))
    except (TypeError, ValueError):
        return default


def _usable(r) -> bool:
    return not _i(r, "timed_out", 0)


def load_session(d: Path) -> Tuple[List[dict], dict, Config]:
    d = Path(d)
    meta = read_json(d / "session.json")
    cfg = Config.from_dict(meta["config"])
    rows = read_trials(d / "trials.csv")
    for r in rows:
        r["_session"] = str(d)
        r["_code"] = meta.get("participant_code", d.parent.name)
    return rows, meta, cfg


# ----------------------------------------------------------------------------
# tracks and thresholds
# ----------------------------------------------------------------------------
def replay_tracks(rows: Sequence[dict], cfg: Config) -> Dict[int, dict]:
    """Recompute every threshold from trials.csv alone, and check the log against the rule."""
    by_track: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        if r.get("phase") == "main" and _usable(r) and _i(r, "track_id") >= 0:
            by_track[_i(r, "track_id")].append(r)
    out = {}
    for tid, rs in by_track.items():
        rs.sort(key=lambda r: _i(r, "track_trial_index"))
        t = Track(cfg, rs[0]["condition"])
        for r in rs:
            if t.finished:
                break
            t.update(bool(_i(r, "correct", 0)))
        a = t.audit()
        logged = [_f(r, "srs_db") for r in rs][: len(t.trials)]
        a["logged_levels_match"] = bool(np.allclose([x.srs_db for x in t.trials], logged,
                                                    rtol=0.0, atol=1e-3))
        a["track_id"] = tid
        out[tid] = a
    return out


@dataclass
class CondResult:
    name: str
    task: str
    kind: str
    step_ms: float
    tracks: List[dict]
    threshold_srs: Optional[float]           # mean over converged tracks, cap-imputed
    n_converged: int
    n_ceiling_limited: int
    ceiling_srs: float                       # lowest shortcut, ideal observer
    n_trials: int
    prop_correct: float

    @property
    def threshold_delta(self) -> Optional[float]:
        return None if self.threshold_srs is None else delta_db(self.threshold_srs)

    @property
    def near_ceiling(self) -> bool:
        return (self.threshold_srs is not None and np.isfinite(self.ceiling_srs)
                and self.threshold_srs > self.ceiling_srs - 2.0)


def thresholds(rows: Sequence[dict], cfg: Config) -> Dict[str, CondResult]:
    conds = {c.name: c for c in conditions(cfg)}
    rep = replay_tracks(rows, cfg)
    by_cond_tracks: Dict[str, List[dict]] = defaultdict(list)
    for a in rep.values():
        by_cond_tracks[a["condition"]].append(a)
    by_cond_rows: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        if r.get("phase") == "main" and _usable(r):
            by_cond_rows[r["condition"]].append(r)
    out = {}
    for name, c in conds.items():
        tr = by_cond_tracks.get(name, [])
        vals, n_ceil = [], 0
        for a in tr:
            if not a["converged"]:
                continue
            if a["ceiling_limited"]:
                n_ceil += 1
                vals.append(cfg.srs_max_db)
            else:
                vals.append(a["threshold_srs_db"])
        rs = by_cond_rows.get(name, [])
        yy = np.array([_i(r, "correct", 0) for r in rs], dtype=float)
        out[name] = CondResult(
            name=name, task=c.task, kind=c.kind, step_ms=c.step_ms, tracks=tr,
            threshold_srs=float(np.mean(vals)) if vals else None, n_converged=len(vals),
            n_ceiling_limited=n_ceil, ceiling_srs=ceilings(cfg, c)["ceiling_srs_db"],
            n_trials=len(rs), prop_correct=float(yy.mean()) if yy.size else float("nan"))
    return out


# ----------------------------------------------------------------------------
# the binding index and the bound
# ----------------------------------------------------------------------------
def first_half_crossing(steps: Sequence[float], nbi: Sequence[float]) -> Optional[float]:
    """The shear at which nBI first falls to 0.5, linearly interpolated. None if never."""
    s, v = np.asarray(steps, float), np.asarray(nbi, float)
    for i in range(1, len(s)):
        if np.isfinite(v[i - 1]) and np.isfinite(v[i]) and v[i - 1] > 0.5 >= v[i]:
            return float(np.interp(0.5, [v[i], v[i - 1]], [s[i], s[i - 1]]))
    return None


def binding_curve(th: Dict[str, float], cfg: Config) -> dict:
    """BI, nBI and s50 from a {condition: threshold_srs} map (one session or one listener)."""
    steps = sorted(cfg.steps_ms)
    tag = lambda t, s: f"{t}_{s:g}".replace(".", "p")

    def val(name):
        v = th.get(name)
        return np.nan if v is None else float(v)
    bi = np.array([val(tag("B", s)) - val(tag("A", s)) for s in steps], dtype=float)
    swing = bi[0] - bi[-1] if np.isfinite(bi[0]) and np.isfinite(bi[-1]) else np.nan
    if np.isfinite(swing) and swing >= MIN_SWING_DB:
        nbi = (bi - bi[-1]) / swing
        s50 = first_half_crossing(steps, nbi)
    else:
        nbi, s50 = np.full_like(bi, np.nan), None
    cap = None
    if th.get("A_cap") is not None and th.get("B_cap") is not None and np.isfinite(bi[0]):
        bic = th["B_cap"] - th["A_cap"]
        cap = {"bi_capture": float(bic), "delta_bi": float(bic - bi[0])}
    alone = None
    if th.get("B_alone") is not None:
        alone = {"b_alone": float(th["B_alone"]),
                 "interference": [float(th[tag("B", s)] - th["B_alone"])
                                  if th.get(tag("B", s)) is not None else np.nan
                                  for s in steps]}
    return {"steps_ms": steps, "bi": bi.tolist(), "swing_db": float(swing),
            "nbi": nbi.tolist(), "s50_bi_ms": s50, "capture": cap, "alone": alone}


def scaled_model_fit(steps: Sequence[float], y: Sequence[float], cfg: Config,
                     variant: str = "halfwave") -> dict:
    """y(s) = a + b * rho(s / alpha): the model's curve, stretched in time by alpha.

    alpha > 1: binding persists to larger shears than the model predicts. Least squares on a
    log grid of alpha, with a and b solved exactly at each alpha. The model's curve is defined
    up to 90 ms of shear, so alpha below about 0.7 reads it past its end, where it is held
    constant; a fit there is flagged rather than trusted.
    """
    from .model import curve, half_way_ms
    s = np.asarray(steps, float)
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return {"ok": False, "reason": "fewer than three steps with a value"}
    alphas = np.exp(np.linspace(np.log(0.25), np.log(6.0), 81))
    best = None
    for a_ in alphas:
        X = np.c_[np.ones(ok.sum()), curve(cfg, s[ok] / a_, variant)]
        coef, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
        sse = float(np.sum((y[ok] - X @ coef) ** 2))
        if best is None or sse < best[0]:
            best = (sse, a_, coef)
    sse, a_, coef = best
    return {"ok": True, "variant": variant, "alpha": float(a_), "a": float(coef[0]),
            "b": float(coef[1]), "sse": sse,
            "at_grid_edge": bool(a_ in (alphas[0], alphas[-1])) or a_ < 0.7,
            "s50_ms": float(a_ * half_way_ms(cfg, variant))}


def weight_curve(weights: dict, cfg: Config) -> dict:
    """w(s), its half-way point, and where it crosses 0.5, from `decision_weights` output."""
    steps = sorted(cfg.steps_ms)
    w = np.array([weights.get(f"{x:g}", {}).get("w_timbre", np.nan) for x in steps],
                 dtype=float)
    cap = weights.get("capture", {}).get("w_timbre", np.nan)
    swing = w[0] - w[-1] if np.isfinite(w[0]) and np.isfinite(w[-1]) else np.nan
    s50 = None
    if np.isfinite(swing) and swing >= MIN_SWING_W:
        s50 = first_half_crossing(steps, (w - w[-1]) / swing)
    cross = first_half_crossing(steps, w) if np.isfinite(w[0]) and w[0] > 0.5 else None
    return {"steps_ms": steps, "w": w.tolist(), "swing": float(swing), "s50_ms": s50,
            "crosses_half_ms": cross, "w_capture": float(cap),
            "capture_change": float(cap - w[0]) if np.isfinite(cap) and np.isfinite(w[0])
            else float("nan")}


# ----------------------------------------------------------------------------
# decision weights
# ----------------------------------------------------------------------------
def probit(X: np.ndarray, y: np.ndarray) -> dict:
    """Maximum-likelihood probit with standard errors from the observed information."""
    from scipy.stats import norm

    def nll(b):
        z = X @ b
        p = np.clip(norm.cdf(z), 1e-9, 1 - 1e-9)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    res = optimize.minimize(nll, np.zeros(X.shape[1]), method="BFGS")
    try:
        se = np.sqrt(np.clip(np.diag(res.hess_inv), 0, None))
    except Exception:
        se = np.full(X.shape[1], np.nan)
    return {"coef": res.x.tolist(), "se": se.tolist(), "converged": bool(res.success),
            "n": int(len(y))}


def decision_weights(rows: Sequence[dict], cfg: Config, by: str = "step") -> dict:
    """Weights on the two cues, from every scored trial that has both.

    Regressors are differences between interval 1 and interval 2, in dB, so the two weights are
    in the same units and their ratio means something. Catch trials (both cues equal, nothing
    to separate) and the target-alone control (no timbre) are left out; fillers are in.
    """
    groups: Dict[str, List[Tuple[float, float, int]]] = defaultdict(list)
    for r in rows:
        if r.get("phase") not in ("main", "filler") or not _usable(r):
            continue
        if r.get("kind") == "alone" or r.get("task") == "none":
            continue
        t1, t2, o1, o2 = (_f(r, k) for k in ("timbre_1", "timbre_2", "own_1", "own_2"))
        if not all(np.isfinite([t1, t2, o1, o2])):
            continue
        resp = str(r.get("response", "")).strip()
        if resp not in ("1", "2"):
            continue
        key = (f"capture" if r.get("kind") == "capture" else f"{_f(r, 'step_ms'):g}") \
            if by == "step" else r.get("condition")
        groups[key].append((t1 - t2, o1 - o2, 1 if resp == "1" else 0))
    out = {}
    for k, v in groups.items():
        a = np.array(v, dtype=float)
        if len(a) < 20:
            out[k] = {"n": len(a), "ok": False}
            continue
        X = np.c_[np.ones(len(a)), a[:, 0], a[:, 1]]
        p = probit(X, a[:, 2])
        bt, bo = p["coef"][1], p["coef"][2]
        w = bt / (bt + bo) if (bt + bo) > 0 else float("nan")
        out[k] = {"n": len(a), "ok": True, "b_timbre": bt, "b_own": bo,
                  "se_timbre": p["se"][1], "se_own": p["se"][2], "bias": p["coef"][0],
                  "w_timbre": float(w)}
    return out


# ----------------------------------------------------------------------------
# one session
# ----------------------------------------------------------------------------
def diagnostics(rows: Sequence[dict], meta: dict, cfg: Config,
                res: Dict[str, CondResult]) -> dict:
    main = [r for r in rows if r.get("phase") == "main" and _usable(r)]
    catch = [r for r in rows if r.get("phase") == "catch" and _usable(r)]
    filler = [r for r in rows if r.get("phase") == "filler"]
    timeouts = [r for r in rows if r.get("phase") in ("main", "catch") and not _usable(r)]
    catch_p = float(np.mean([_i(r, "correct", 0) for r in catch])) if catch else float("nan")
    resp1 = float(np.mean([str(r.get("response")).strip() == "1" for r in main])) \
        if main else float("nan")
    pos1 = float(np.mean([_i(r, "target_position") == 1 for r in main])) if main else float("nan")
    rts = np.array([_f(r, "rt_ms") for r in main])
    rts = rts[np.isfinite(rts)]
    tail = [r for r in main if _i(r, "n_active_tracks", 99) <= 2]
    rep = [t for c in res.values() for t in c.tracks]
    cal = meta.get("calibration") or {}
    practice = meta.get("practice") or []
    catch_ok = bool(catch) and (1 - catch_p) <= cfg.max_catch_miss_rate
    status = meta.get("status")
    usable = catch_ok and status == "complete"
    reasons = []
    if not catch_ok:
        reasons.append(f"catch miss rate {1 - catch_p:.0%} exceeds "
                       f"{cfg.max_catch_miss_rate:.0%}" if catch else "no catch trials")
    if status != "complete":
        reasons.append(f"status {status}")
    return {"n_main": len(main), "n_catch": len(catch), "n_filler": len(filler),
            "n_timeouts": len(timeouts), "catch_correct": catch_p, "catch_ok": catch_ok,
            "prop_respond_1": resp1, "prop_target_1": pos1,
            "rt_median_ms": float(np.median(rts)) if rts.size else float("nan"),
            "trials_with_two_or_fewer_tracks_active": len(tail),
            "n_tracks": len(rep), "n_converged": sum(t["converged"] for t in rep),
            "n_ceiling_limited": sum(bool(t.get("ceiling_limited")) for t in rep),
            "log_replays_cleanly": all(t["logged_levels_match"] for t in rep),
            "measured_db_spl": cal.get("measured_db_spl"), "calibration_note": cal.get("note"),
            "practice": practice,
            "practice_ok": bool(practice) and max(practice) >= cfg.practice_criterion,
            "status": status, "usable": usable, "unusable_because": reasons}


def analyse_session(d: Path) -> dict:
    rows, meta, cfg = load_session(d)
    res = thresholds(rows, cfg)
    th = {k: v.threshold_srs for k, v in res.items()}
    weights = decision_weights(rows, cfg)
    wc = weight_curve(weights, cfg)
    return {"dir": str(d), "code": meta.get("participant_code"),
            "session_index": meta.get("session_index"), "config_hash": cfg.hash(),
            "cfg": cfg, "results": res, "thresholds": th, "curve": binding_curve(th, cfg),
            "weights": weights, "wcurve": wc,
            "fit": {v: scaled_model_fit(wc["steps_ms"], wc["w"], cfg, v)
                    for v in ("halfwave", "magnitude")},
            "diagnostics": diagnostics(rows, meta, cfg, res), "rows": rows}


def _fmt_ms(x) -> str:
    return "undefined" if x is None or not np.isfinite(x) else f"{x:.1f} ms"


def format_session(a: dict) -> str:
    cfg, res, c, dg, wc = a["cfg"], a["results"], a["curve"], a["diagnostics"], a["wcurve"]
    L = [f"tcbound session {a['dir']}  ({a['code']}, session {a['session_index']}, "
         f"config {a['config_hash']})", ""]
    L.append(f"  {'condition':<9} {'thr SRS':>8} {'dL dB':>7} {'ceiling':>8} {'trials':>7} "
             f"{'%corr':>6}  flags")
    for name, r in res.items():
        thr = "-" if r.threshold_srs is None else f"{r.threshold_srs:+.1f}"
        dl = "-" if r.threshold_delta is None else f"{r.threshold_delta:.1f}"
        flags = []
        if r.n_converged < len(r.tracks) or not r.tracks:
            flags.append("not converged" if r.tracks else "not run")
        if r.n_ceiling_limited:
            flags.append("CEILING-LIMITED (lower bound)")
        elif r.near_ceiling:
            flags.append("within 2 dB of the shortcut ceiling")
        L.append(f"  {name:<9} {thr:>8} {dl:>7} {r.ceiling_srs:+8.1f} {r.n_trials:7d} "
                 f"{100 * r.prop_correct:5.0f}%  {', '.join(flags)}")
    L += ["", "  by step: BI = threshold(B) - threshold(A) in dB (positive = bound); "
          "w = timbre weight (1 = bound)"]
    for x, b, w in zip(c["steps_ms"], c["bi"], wc["w"]):
        L.append(f"    step {x:5g} ms   BI {b:+6.1f}   w {w:5.2f}")
    if c["capture"] or np.isfinite(wc["w_capture"]):
        bi_c = f"{c['capture']['bi_capture']:+.1f} (change {c['capture']['delta_bi']:+.1f})" \
            if c["capture"] else "-"
        L.append(f"    capture        BI {bi_c}   w {wc['w_capture']:.2f} "
                 f"(change {wc['capture_change']:+.2f})")
    if c["alone"]:
        L.append(f"    target alone: {c['alone']['b_alone']:+.1f} dB SRS; B minus alone by "
                 "step: " + ", ".join(f"{x:+.1f}" for x in c["alone"]["interference"]))
    L.append(f"    bound from w: s50 {_fmt_ms(wc['s50_ms'])}; w crosses 0.5 at "
             f"{_fmt_ms(wc['crosses_half_ms'])}")
    for v, f in a["fit"].items():
        if f.get("ok"):
            L.append(f"    model-shaped fit to w ({v}): alpha {f['alpha']:.2f}"
                     f"{' (outside the range the fit can trust)' if f['at_grid_edge'] else ''}"
                     f", s50 {f['s50_ms']:.1f} ms")
    L.append("    one session is one track per condition: these are descriptions, and the "
             "tests are run across listeners")
    L += ["", "  decision weights (probit on interval differences)"]
    for k in sorted(a["weights"], key=lambda k: (k == "capture",
                                                  float(k) if k != "capture" else 0.0)):
        w = a["weights"][k]
        if w.get("ok"):
            L.append(f"    {k:>8}: n {w['n']:3d}  b_timbre {w['b_timbre']:+.3f} "
                     f"(se {w['se_timbre']:.3f})  b_own {w['b_own']:+.3f} "
                     f"(se {w['se_own']:.3f})  w {w['w_timbre']:.2f}")
    L += ["", "  diagnostics",
          f"    status {dg['status']}; usable {dg['usable']}"
          + (f" ({'; '.join(dg['unusable_because'])})" if dg['unusable_because'] else ""),
          f"    catch {dg['n_catch']} trials, {100 * dg['catch_correct']:.0f}% correct "
          f"(criterion: miss rate <= {100 * cfg.max_catch_miss_rate:.0f}%)",
          f"    {dg['n_main']} scored, {dg['n_filler']} fillers, {dg['n_timeouts']} timeouts; "
          f"{dg['n_converged']}/{dg['n_tracks']} tracks converged, "
          f"{dg['n_ceiling_limited']} ceiling-limited",
          f"    responded '1' on {100 * dg['prop_respond_1']:.0f}% (target was in 1 on "
          f"{100 * dg['prop_target_1']:.0f}%); median RT {dg['rt_median_ms']:.0f} ms",
          f"    trials run with two or fewer tracks left: "
          f"{dg['trials_with_two_or_fewer_tracks_active']}",
          f"    log replays cleanly: {dg['log_replays_cleanly']}",
          f"    level: {dg['measured_db_spl']} dB SPL measured ({dg['calibration_note']}); "
          f"practice {', '.join(f'{p:.0%}' for p in dg['practice']) or 'none'}"]
    return "\n".join(L)


# ----------------------------------------------------------------------------
# the group: one row per listener, all statistics across listeners
# ----------------------------------------------------------------------------
def _ttest_1s(x, alternative: str) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return {"n": len(x), "ok": False}
    t = stats.ttest_1samp(x, 0.0, alternative=alternative)
    w = stats.wilcoxon(x, alternative=alternative) if len(x) >= 6 else None
    sd = x.std(ddof=1)
    return {"n": len(x), "ok": True, "mean": float(x.mean()), "sd": float(sd),
            "t": float(t.statistic), "p": float(t.pvalue),
            "dz": float(x.mean() / sd) if sd > 0 else float("inf"),
            "wilcoxon_p": None if w is None else float(w.pvalue)}


def _slope(y: Sequence[float]) -> float:
    y = np.asarray(y, float)
    k = np.arange(len(y), dtype=float)
    ok = np.isfinite(y)
    return float(np.polyfit(k[ok], y[ok], 1)[0]) if ok.sum() >= 3 else float("nan")


def group(dirs: Sequence[Path], n_boot: int = 4000, seed: int = 11) -> dict:
    sessions = [analyse_session(Path(d)) for d in dirs]
    hashes = {s["config_hash"] for s in sessions}
    if len(hashes) > 1:
        raise ValueError(f"sessions were run under different configurations {sorted(hashes)}; "
                         "they are separate experiments and are not pooled")
    cfg = sessions[0]["cfg"]
    excluded = [s for s in sessions if not s["diagnostics"]["usable"]]
    by_code: Dict[str, List[dict]] = defaultdict(list)
    for s in sessions:
        if s["diagnostics"]["usable"]:
            by_code[s["code"]].append(s)
    names = [c.name for c in conditions(cfg)]
    steps = sorted(cfg.steps_ms)
    listeners = {}
    for code, ss in sorted(by_code.items()):
        th = {}
        for n in names:
            v = [s["thresholds"][n] for s in ss if s["thresholds"].get(n) is not None]
            th[n] = float(np.mean(v)) if v else None
        rows = [r for s in ss for r in s["rows"]]
        wts = decision_weights(rows, cfg)
        wc = weight_curve(wts, cfg)
        listeners[code] = {"n_sessions": len(ss), "thresholds": th,
                           "curve": binding_curve(th, cfg), "weights": wts, "wcurve": wc,
                           "fit": scaled_model_fit(steps, wc["w"], cfg, "halfwave")}
    BI = np.array([L["curve"]["bi"] for L in listeners.values()], dtype=float)
    W = np.array([L["wcurve"]["w"] for L in listeners.values()], dtype=float)
    n = len(listeners)
    if n == 0:
        raise ValueError("no usable session")
    mean_w, mean_bi = np.nanmean(W, axis=0), np.nanmean(BI, axis=0)

    def s50_of(w):
        sw = w[0] - w[-1]
        if not np.isfinite(sw) or sw < MIN_SWING_W:
            return None
        return first_half_crossing(steps, (w - w[-1]) / sw)

    s50 = s50_of(mean_w)
    rng = np.random.default_rng(seed)
    boots, undefined = [], 0
    if n >= 3:
        for _ in range(n_boot):
            v = s50_of(np.nanmean(W[rng.integers(0, n, n)], axis=0))
            if v is None:
                undefined += 1
            else:
                boots.append(v)
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) \
        if boots and undefined <= 0.05 * n_boot else None
    fits = {v: scaled_model_fit(steps, mean_w, cfg, v) for v in ("halfwave", "magnitude")}
    from .model import predictions
    pred = predictions(cfg)
    rho = np.array(pred["variants"]["halfwave"]["curve"])
    cap_pred = None
    if np.isfinite(mean_w).all():
        b1, b0 = np.polyfit(rho, mean_w, 1)
        r_cap = pred["variants"]["halfwave"]["capture_rho"]
        w_cap = np.array([L["wcurve"]["w_capture"] for L in listeners.values()], dtype=float)
        cap_pred = {"rho_capture": r_cap, "predicted_w": float(b0 + b1 * r_cap),
                    "observed_w": float(np.nanmean(w_cap)) if np.isfinite(w_cap).any()
                    else None,
                    "observed_sem": float(np.nanstd(w_cap, ddof=1)
                                          / np.sqrt(np.isfinite(w_cap).sum()))
                    if np.isfinite(w_cap).sum() >= 2 else None}
    rel = None
    pairs = [(ss[0], ss[1]) for ss in by_code.values() if len(ss) >= 2]
    if len(pairs) >= 3:
        sw = np.array([(a["curve"]["swing_db"], b["curve"]["swing_db"]) for a, b in pairs])
        ws = np.array([(a["wcurve"]["swing"], b["wcurve"]["swing"]) for a, b in pairs])
        ok_b, ok_w = np.isfinite(sw).all(1), np.isfinite(ws).all(1)
        rel = {"n": len(pairs),
               "bi_swing_r": float(np.corrcoef(sw[ok_b].T)[0, 1]) if ok_b.sum() >= 3 else None,
               "w_swing_r": float(np.corrcoef(ws[ok_w].T)[0, 1]) if ok_w.sum() >= 3 else None,
               "bi_swing_mean_abs_diff_db": float(np.mean(np.abs(sw[ok_b, 0] - sw[ok_b, 1])))
               if ok_b.any() else None}
    tag_max = f"B_{max(steps):g}".replace(".", "p")
    ca = ceilings(cfg, Condition("A_0", "A", 0.0))["ceiling_srs_db"]
    cb = ceilings(cfg, Condition("B", "B", 0.0))["ceiling_srs_db"]
    checks = {code: {"A0_below_ceiling": L["thresholds"].get("A_0") is not None
                     and L["thresholds"]["A_0"] < ca - cfg.near_ceiling_db,
                     "Bmax_below_ceiling": L["thresholds"].get(tag_max) is not None
                     and L["thresholds"][tag_max] < cb - cfg.near_ceiling_db}
              for code, L in listeners.items()}
    dcap_bi = [L["curve"]["capture"]["delta_bi"] if L["curve"]["capture"] else np.nan
               for L in listeners.values()]
    return {"cfg": cfg, "n_sessions": len(sessions),
            "excluded": [{"dir": s["dir"], "code": s["code"],
                          "because": s["diagnostics"]["unusable_because"]} for s in excluded],
            "listeners": listeners, "steps_ms": steps,
            "mean_bi": mean_bi.tolist(), "mean_w": mean_w.tolist(),
            "sem_bi": (np.nanstd(BI, axis=0, ddof=1) / np.sqrt(np.isfinite(BI).sum(0))).tolist()
            if n >= 2 else None,
            "sem_w": (np.nanstd(W, axis=0, ddof=1) / np.sqrt(np.isfinite(W).sum(0))).tolist()
            if n >= 2 else None,
            "s50_ms": s50, "s50_ci": ci, "s50_boot_undefined": undefined, "fits": fits,
            "H1": _ttest_1s([_slope(b) for b in BI], "less"),
            "H1_swing": _ttest_1s(BI[:, 0] - BI[:, -1], "greater"),
            "H2": _ttest_1s([_slope(w) for w in W], "less"),
            "H3": _ttest_1s([L["wcurve"]["capture_change"] for L in listeners.values()], "less"),
            "H3_bi": _ttest_1s(dcap_bi, "less"),
            "H4_checks": checks, "capture_prediction": cap_pred, "reliability": rel,
            "model": pred}


def format_group(g: dict) -> str:
    cfg = g["cfg"]
    L = [f"tcbound group summary  |  config {cfg.hash()}  |  {len(g['listeners'])} listeners, "
         f"{g['n_sessions']} sessions", ""]
    if g["excluded"]:
        L.append("  excluded sessions (rules fixed in the protocol):")
        for e in g["excluded"]:
            L.append(f"    {e['code']}  {e['dir']}: {'; '.join(e['because'])}")
        L.append("")
    steps = ", ".join(f"{x:g}" for x in g["steps_ms"])
    L.append(f"  per listener, by step ({steps} ms)")
    for code, Lr in g["listeners"].items():
        f = Lr["fit"]
        al = f"{f['alpha']:.2f}" if f.get("ok") else "-"
        L.append(f"    {code:<6} BI " + " ".join(f"{b:+6.1f}" for b in Lr["curve"]["bi"])
                 + "   w " + " ".join(f"{w:4.2f}" for w in Lr["wcurve"]["w"])
                 + f"   s50 {_fmt_ms(Lr['wcurve']['s50_ms']):>9}   alpha {al}"
                 f"   ({Lr['n_sessions']} session{'s' if Lr['n_sessions'] != 1 else ''})")
    L += ["", "  group mean BI " + " ".join(f"{b:+.1f}" for b in g["mean_bi"]),
          "  group mean w  " + " ".join(f"{w:.2f}" for w in g["mean_w"])]
    if g["s50_ms"] is not None:
        ci = (f", 95% CI over listeners {g['s50_ci'][0]:.1f} to {g['s50_ci'][1]:.1f} ms"
              if g["s50_ci"] else ", no CI: the bound is undefined in more than 5% of "
              "bootstrap samples")
        L.append(f"  THE BOUND: w falls half-way at s50 = {g['s50_ms']:.1f} ms{ci}")
    else:
        L.append("  THE BOUND: undefined; the group-mean timbre weight does not fall by "
                 f"{MIN_SWING_W} between step 0 and the largest step")
    for v, f in g["fits"].items():
        if f.get("ok"):
            L.append(f"  model-shaped fit to w, {v}: alpha {f['alpha']:.2f}, s50 "
                     f"{f['s50_ms']:.1f} ms, SSE {f['sse']:.4f}"
                     + (" (outside the range the fit can trust)" if f["at_grid_edge"] else ""))

    def t(name, h, label):
        if not h.get("ok"):
            return f"  {name}: not enough listeners ({h.get('n', 0)})"
        return (f"  {name}: {label} mean {h['mean']:+.3f} (sd {h['sd']:.3f}), "
                f"t({h['n'] - 1}) = {h['t']:.2f}, one-sided p = {h['p']:.2g}, dz = {h['dz']:.2f}"
                + (f"; Wilcoxon p = {h['wilcoxon_p']:.2g}" if h.get("wilcoxon_p") is not None
                   else ""))
    L += ["", t("H1 primary", g["H1"], "slope of BI per step,"),
          t("H1 swing  ", g["H1_swing"], "BI(0) - BI(max),"),
          t("H2        ", g["H2"], "slope of w per step,"),
          t("H3 capture", g["H3"], "w(capture) - w(0),"),
          t("H3 (BI)   ", g["H3_bi"], "BI(capture) - BI(0),")]
    ok = sum(c["A0_below_ceiling"] and c["Bmax_below_ceiling"] for c in g["H4_checks"].values())
    L.append(f"  H4 manipulation check: {ok}/{len(g['H4_checks'])} listeners have A at step 0 and "
             f"B at the largest step more than {cfg.near_ceiling_db:g} dB below their shortcut "
             "ceilings")
    cp = g["capture_prediction"]
    if cp:
        obs = "-" if cp["observed_w"] is None else (
            f"{cp['observed_w']:.2f}" + (f" (sem {cp['observed_sem']:.2f})"
                                         if cp["observed_sem"] is not None else ""))
        L.append(f"  capture, out of sample: rho_T {cp['rho_capture']:.2f}; the w-to-rho line of "
                 f"the shear conditions predicts w = {cp['predicted_w']:.2f}; observed {obs}")
    if g["reliability"]:
        r = g["reliability"]
        fmt = lambda x: "-" if x is None else f"{x:.2f}"
        L.append(f"  repeatability ({r['n']} listeners with two sessions): swing of BI r = "
                 f"{fmt(r['bi_swing_r'])}, swing of w r = {fmt(r['w_swing_r'])}")
    pv = g["model"]["variants"]
    L += ["", "  model predictions, fixed before the data (rho_T by step; half-way point)"]
    for v in ("halfwave", "magnitude"):
        L.append(f"    {v:<9} " + " ".join(f"{x:.2f}" for x in pv[v]["curve"])
                 + f"   half-way {pv[v]['half_way_ms']:.1f} ms; capture rho "
                 f"{pv[v]['capture_rho']:.2f}")
    return "\n".join(L)
