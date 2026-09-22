"""From trials to thresholds to a verdict, with the limits of each step stated.

What this measures
-------------------
One threshold per condition: the jitter at which the listener is 79.4% correct. The axis is the
shear step. The quantity of interest is the SHAPE of threshold against step, and in particular
where it sits relative to the within-channel control.

What the shape can and cannot settle
-------------------------------------
  rising with step
      The listener is using the figure. A within-channel strategy cannot produce this, because
      every channel is isochronous at `rate_hz` whatever the shear -- that is a property of the
      stimulus, not an assumption. This is the load-bearing inference of the design.

  flat across step
      The listener is using the target's own rhythm and ignoring the figure. Not a null result
      about binding; a positive result about strategy, and the `single` control says what that
      strategy's limit is.

  step 0 well below the `single` control
      Four synchronous tones give better temporal resolution on one of them than that tone
      alone does. That is the synchrony advantage, and it is the one contrast the design is
      built to see.

  step 100% better than step 75%
      NOT a coherence effect. The model puts those two at 0.901 and 0.947 -- the same figure --
      and 100% is the only step whose combined onset train is isochronous. An improvement
      there is the rhythm cue, and `rhythm_test` reports it separately for that reason.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from tcoh.psychometric import Fit, fit as pf_fit, target_proportion
from tcoh.session import read_json, read_trials
from tcoh.track import Track

from .config import Condition, Config, validate


def load(dirs: Sequence[Path]) -> Tuple[List[dict], List[dict]]:
    rows, metas = [], []
    for d in dirs:
        d = Path(d)
        meta = read_json(d / "session.json")
        metas.append(meta)
        for r in read_trials(d / "trials.csv"):
            r["_session"] = str(d)
            r["_code"] = meta.get("participant_code", d.parent.name)
            rows.append(r)
    return rows, metas


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


@dataclass
class CondResult:
    name: str
    step_pct: float
    kind: str
    track_thresholds: List[float]
    censored_thresholds: List[float]
    censor_side: str
    n_tracks_attempted: int
    geomean_ms: Optional[float]
    fit: Optional[Fit]
    n_trials: int
    prop_correct: float
    crossing_fraction: float

    @property
    def censored(self) -> bool:
        return bool(self.censored_thresholds)


def replay_tracks(rows: Sequence[dict], cfg: Config) -> Dict[int, dict]:
    """Recompute every threshold from trials.csv alone, without trusting session.json."""
    by_track: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        if r.get("phase") != "main" or not _usable(r):
            continue
        tid = _i(r, "track_id")
        if tid >= 0:
            by_track[tid].append(r)
    out = {}
    for tid, rs in by_track.items():
        rs.sort(key=lambda r: _i(r, "track_trial_index"))
        t = Track(cfg, rs[0]["condition"], 0)
        for r in rs:
            if t.finished:
                break
            t.update(bool(_i(r, "correct", 0)))
        a = t.audit()
        a["logged_deltas_match"] = bool(np.allclose(
            [tt.delta_ms for tt in t.trials][: len(rs)],
            [_f(r, "delta_ms") for r in rs][: len(t.trials)], rtol=0.0, atol=1e-5))
        out[tid] = a
    return out


def thresholds(rows: Sequence[dict], cfg: Config) -> Dict[str, CondResult]:
    d = validate(cfg)
    conds = {c.name: c for c in d.conditions}
    rep = replay_tracks(rows, cfg)
    usable: Dict[str, List[float]] = defaultdict(list)
    censored: Dict[str, List[float]] = defaultdict(list)
    sides: Dict[str, str] = {}
    attempted: Dict[str, int] = defaultdict(int)
    for a in rep.values():
        attempted[a["condition"]] += 1
        if not a["converged"]:
            continue
        # A reversal recorded AT a clamp is not a reversal: the rule asked for a value the
        # stimulus cannot provide. Such a track bounds the threshold, it does not measure it.
        if a["ceiling_inside_threshold"]:
            censored[a["condition"]].append(a["threshold_ms"]); sides[a["condition"]] = "ceiling"
        elif a["floor_inside_threshold"]:
            censored[a["condition"]].append(a["threshold_ms"]); sides[a["condition"]] = "floor"
        else:
            usable[a["condition"]].append(a["threshold_ms"])

    by_cond: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        if r.get("phase") in ("main", "catch") and _usable(r):
            by_cond[r["condition"]].append(r)

    out = {}
    for name, c in conds.items():
        tr = by_cond.get(name, [])
        dd = np.array([abs(_f(r, "delta_realised_ms")) for r in tr])
        yy = np.array([_i(r, "correct", 0) for r in tr], dtype=bool)
        f = pf_fit(dd, yy, lapse_max=cfg.lapse_max) if dd.size >= 8 else None
        u = usable.get(name, [])
        out[name] = CondResult(
            name=name, step_pct=c.step_pct, kind=c.kind,
            track_thresholds=u, censored_thresholds=censored.get(name, []),
            censor_side=sides.get(name, ""), n_tracks_attempted=attempted.get(name, 0),
            geomean_ms=float(np.exp(np.mean(np.log(u)))) if u else None,
            fit=f, n_trials=len(tr),
            prop_correct=float(yy.mean()) if yy.size else float("nan"),
            crossing_fraction=float(np.mean([_i(r, "crosses_neighbour", 0) for r in tr]))
            if tr else 0.0)
    return out


def trend_test(res: Dict[str, CondResult], n_perm: int = 10000, seed: int = 7) -> dict:
    """Does threshold rise with the step? Spearman over TRACKS, permuted for a calibrated p."""
    pts = [(r.step_pct, t) for r in res.values() if r.kind == "figure"
           for t in r.track_thresholds]
    if len({p for p, _ in pts}) < 3:
        return {"ok": False, "reason": "fewer than three step levels with usable tracks"}
    x = np.array([p for p, _ in pts]); y = np.log(np.array([t for _, t in pts]))
    rho = float(stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    null = np.array([stats.spearmanr(x, rng.permutation(y)).statistic for _ in range(n_perm)])
    return {"ok": True, "rho": rho, "p_one_sided": float((null >= rho).mean()),
            "n_tracks": len(pts), "n_levels": len({p for p, _ in pts})}


def synchrony_advantage(res: Dict[str, CondResult]) -> dict:
    """How much better is the synchronous figure than the target tone alone?

    The one contrast the design is built for, and the only one that does not depend on reading
    a shape off a handful of points.
    """
    zero = next((r for r in res.values() if r.kind == "figure" and r.step_pct == 0.0), None)
    solo = next((r for r in res.values() if r.kind == "single"), None)
    if not zero or not zero.geomean_ms:
        return {"ok": False, "reason": "no usable dT=0 condition"}
    if not solo or not solo.geomean_ms:
        return {"ok": False, "reason": "the single-tone control was not measured or not usable"}
    return {"ok": True, "figure_ms": zero.geomean_ms, "single_ms": solo.geomean_ms,
            "ratio": solo.geomean_ms / zero.geomean_ms,
            "log_ratio": math.log(solo.geomean_ms / zero.geomean_ms)}


def rhythm_test(res: Dict[str, CondResult], cfg: Config) -> dict:
    """Is the isochronous step better than the one just below it?

    The model puts the top two steps at nearly the same index, so it predicts no difference.
    An improvement at the isochronous step is therefore evidence for the rhythmic cue and not
    for coherence, and is reported on its own rather than folded into the trend.
    """
    from .config import is_isochronous
    iso = [r for r in res.values() if r.kind == "figure"
           and is_isochronous(cfg, r.step_pct) and r.step_pct > 0]
    if not iso:
        return {"ok": False, "reason": "no isochronous step in this design"}
    top = iso[0]
    below = [r for r in res.values() if r.kind == "figure" and 0 < r.step_pct < top.step_pct]
    if not below or not top.geomean_ms:
        return {"ok": False, "reason": "nothing to compare the isochronous step against"}
    nearest = max(below, key=lambda r: r.step_pct)
    if not nearest.geomean_ms:
        return {"ok": False, "reason": f"{nearest.name} has no usable threshold"}
    return {"ok": True, "iso_pct": top.step_pct, "iso_ms": top.geomean_ms,
            "below_pct": nearest.step_pct, "below_ms": nearest.geomean_ms,
            "ratio": nearest.geomean_ms / top.geomean_ms,
            "better_at_iso": bool(top.geomean_ms < nearest.geomean_ms)}


def diagnostics(rows, metas, cfg: Config, res) -> dict:
    main = [r for r in rows if r.get("phase") == "main" and _usable(r)]
    catch = [r for r in rows if r.get("phase") == "catch" and _usable(r)]
    prac = [r for r in rows if r.get("phase") == "practice"]
    catch_p = float(np.mean([_i(r, "correct", 0) for r in catch])) if catch else float("nan")
    n1 = sum(1 for r in main if str(r.get("response", "")).strip() == "1")
    pos1 = sum(1 for r in main if _i(r, "target_position") == 1)
    n_cens = sum(len(r.censored_thresholds) for r in res.values())
    n_conv = sum(len(r.track_thresholds) for r in res.values()) + n_cens
    lvl = None
    for m in metas:
        c = m.get("calibration") or {}
        if c.get("measured_db_spl") is not None:
            lvl = float(c["measured_db_spl"])
    half = len(main) // 2
    return {
        "n_main": len(main), "n_catch": len(catch), "n_practice": len(prac),
        "n_timeouts": sum(1 for r in rows if _i(r, "timed_out", 0)),
        "catch_p_correct": catch_p,
        "catch_ok": bool(catch and (1.0 - catch_p) <= cfg.max_catch_miss_rate),
        "practice_rounds": [p for m in metas for p in m.get("practice", [])],
        "press_1_rate": n1 / max(len(main), 1),
        "press_1_p": float(stats.binomtest(n1, max(len(main), 1), 0.5).pvalue),
        "target_position_p": float(stats.binomtest(pos1, max(len(main), 1), 0.5).pvalue),
        "bias_drift_rho": float(stats.spearmanr(
            np.arange(len(main)),
            [str(r.get("response", "")).strip() == "1" for r in main]).statistic)
        if len(main) > 10 else float("nan"),
        "bias_drift_p": float(stats.spearmanr(
            np.arange(len(main)),
            [str(r.get("response", "")).strip() == "1" for r in main]).pvalue)
        if len(main) > 10 else float("nan"),
        "first_half_correct": float(np.mean([_i(r, "correct", 0) for r in main[:half]]))
        if half else float("nan"),
        "second_half_correct": float(np.mean([_i(r, "correct", 0) for r in main[half:]]))
        if half else float("nan"),
        "crossing_fraction": float(np.mean([_i(r, "crosses_neighbour", 0) for r in main]))
        if main else 0.0,
        "tracks_converged": n_conv, "tracks_censored": n_cens,
        "tracks_attempted": sum(r.n_tracks_attempted for r in res.values()),
        "level_db_spl": lvl,
        "replay_ok": all(a["logged_deltas_match"] for a in replay_tracks(rows, cfg).values()),
    }


def analyse(dirs: Sequence[Path], cfg: Optional[Config] = None) -> str:
    rows, metas = load([Path(d) for d in dirs])
    if cfg is None:
        cfg = Config.from_dict(metas[0]["config"])
    d = validate(cfg)
    res = thresholds(rows, cfg)
    diag = diagnostics(rows, metas, cfg, res)
    L: List[str] = []
    A = L.append
    A("=" * 78)
    A(f"sheared four-tone figure  |  {len(metas)} session(s), "
      f"participant(s) {', '.join(sorted({m.get('participant_code', '?') for m in metas}))}")
    A(f"config {cfg.hash()}   {cfg.n_tones} tones at {cfg.rate_hz:g} Hz, "
      f"{cfg.tone_ms:g} ms, target {cfg.target_freq_hz:.0f} Hz")
    A(f"threshold = {100 * target_proportion(cfg.n_down):.1f}% correct point, geometric mean "
      f"of the last {cfg.n_final_reversals} reversals")
    A("")
    A("-- session validity " + "-" * 57)
    A(f"  main {diag['n_main']}   catch {diag['n_catch']}   practice {diag['n_practice']}"
      + (f"   timed out {diag['n_timeouts']}" if diag["n_timeouts"] else ""))
    if diag["n_catch"]:
        A(f"  catch at {cfg.catch_delta_ms:g} ms: {diag['catch_p_correct']:.1%} correct "
          f"(limit {cfg.max_catch_miss_rate:.0%} misses) -> "
          f"{'OK' if diag['catch_ok'] else 'FAILED: every threshold below is unsafe'}")
    A(f"  tracks converged {diag['tracks_converged']}/{diag['tracks_attempted']}"
      + (f", {diag['tracks_censored']} boundary-limited" if diag["tracks_censored"] else ""))
    A(f"  every threshold recomputes from the trial log: {diag['replay_ok']}")
    A(f"  pressed '1' on {100 * diag['press_1_rate']:.1f}% (p={diag['press_1_p']:.3f}); "
      f"drift across the session rho={diag['bias_drift_rho']:+.3f} (p={diag['bias_drift_p']:.3f})")
    A(f"  first half {100 * diag['first_half_correct']:.1f}% correct, "
      f"second half {100 * diag['second_half_correct']:.1f}%")
    A(f"  jitter exceeded the step on {100 * diag['crossing_fraction']:.1f}% of trials "
      "(the target moved past a neighbour rather than away from it)")
    if diag["level_db_spl"] is None:
        A("  WARNING: level was NOT measured; every level quoted here is nominal.")
    else:
        A(f"  level: {diag['level_db_spl']:.1f} dB SPL for one tone")
    A("")
    A("-- thresholds " + "-" * 63)
    A(f"  {'condition':<10} {'step%':>7} {'tracks':>7} {'staircase':>11} {'fit':>9} "
      f"{'%corr':>7} {'model':>7}")
    for c in d.conditions:
        r = res[c.name]
        g = f"{r.geomean_ms:9.2f} ms" if r.geomean_ms else (
            f"  >{np.exp(np.mean(np.log(r.censored_thresholds))):6.1f}*" if r.censored
            else "        - ")
        fit = f"{r.fit.threshold_ms:7.2f}" if r.fit else "      -"
        mv = d.model_index[c.name]
        A(f"  {c.name:<10} {('-' if c.kind == 'single' else f'{c.step_pct:g}'):>7} "
          f"{len(r.track_thresholds)}/{r.n_tracks_attempted:<5} {g:>11} {fit:>9} "
          f"{100 * r.prop_correct:6.1f}% "
          + ("      -" if mv != mv else f"{1 + mv * (cfg.n_tones - 1):6.2f}"))
    if any(r.censored for r in res.values()):
        A("  * boundary-limited: a bound, not a threshold; excluded from every summary below")
    A("")
    A("-- what the shape says " + "-" * 54)
    t = trend_test(res)
    if t["ok"]:
        A(f"  threshold rises with the step: Spearman rho = {t['rho']:+.3f} over "
          f"{t['n_tracks']} tracks at {t['n_levels']} steps, permutation p = "
          f"{t['p_one_sided']:.4f} (one sided)")
        A("    A within-channel strategy CANNOT produce this: every channel is isochronous at")
        A(f"    {cfg.rate_hz:g} Hz whatever the shear, so the target's own rhythm is the same")
        A("    reference at every step. A rise means the figure was used.")
    else:
        A(f"  trend not computed: {t['reason']}")
    s = synchrony_advantage(res)
    if s["ok"]:
        A(f"  synchrony advantage: {s['single_ms']:.2f} ms for the target tone ALONE against "
          f"{s['figure_ms']:.2f} ms")
        A(f"    inside the synchronous figure -- a factor of {s['ratio']:.2f}"
          + ("  (the figure helped)" if s["ratio"] > 1 else "  (the figure did NOT help)"))
    else:
        A(f"  synchrony advantage not computed: {s['reason']}")
    rt = rhythm_test(res, cfg)
    if rt["ok"]:
        A(f"  isochronous step {rt['iso_pct']:g}% gives {rt['iso_ms']:.2f} ms against "
          f"{rt['below_ms']:.2f} ms at {rt['below_pct']:g}% "
          f"({rt['ratio']:.2f}x)")
        A("    The model puts these two steps at almost the same index, so it predicts no")
        A("    difference. Better at the isochronous step is the RHYTHM cue, not coherence.")
    A("")
    A("-- what this does not establish " + "-" * 45)
    A("  * a rise with the step is consistent with temporal coherence, and also with any")
    A("    account in which a more spread-out pattern is simply harder. The single-tone")
    A("    control bounds the second of those; it does not eliminate it.")
    A("  * one listener per session. Nothing here generalises.")
    A("  * the model index is plotted beside the data, never on it. Its curve is close to")
    A("    linear in the step over this range, so agreement is not mechanistic evidence.")
    A("=" * 78)
    return "\n".join(L)
