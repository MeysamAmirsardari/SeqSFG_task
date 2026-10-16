"""From trials to thresholds to a verdict, with the limits of each step stated.

What the measurements can settle
---------------------------------
  every pair condition below B-alone
      The listener used A. B-alone measures exactly what is available from the high tone on its
      own, with the same jitter and the same displacement; a pair condition that beats it can
      only have done so through the A-B relation. This is the load-bearing comparison.

  threshold rising with the lag
      Using A gets harder as A and B separate in time. Consistent with temporal coherence, and
      also with any account in which a longer interval is harder to judge -- a Weber account of
      the A-B gap predicts a rise too. The rise alone does not choose between them.

  the strategy check
      Every trial's jitter is logged, so the analysis can reconstruct how irregular the B channel
      looked ON ITS OWN in each interval, and ask whether the listener's answers followed that.
      A listener using A should not care, once the displacement itself is accounted for; a
      listener using B alone should care a great deal. B-alone trials are the positive control:
      there the answers MUST follow it, and if they do not, the check has no power.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, stats
from scipy.special import expit

from tcoh.psychometric import Fit, fit as pf_fit, target_proportion
from tcoh.session import read_json, read_trials
from tcoh.track import Track

from .config import Config, validate


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
    lag_ms: float
    kind: str
    track_thresholds: List[float]
    censored_thresholds: List[float]
    censor_side: str
    n_tracks_attempted: int
    geomean_ms: Optional[float]
    fit: Optional[Fit]
    n_trials: int
    prop_correct: float
    pairing_flip_fraction: float

    @property
    def censored(self) -> bool:
        return bool(self.censored_thresholds)


def replay_tracks(rows: Sequence[dict], cfg: Config) -> Dict[int, dict]:
    by: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        if r.get("phase") == "main" and _usable(r) and _i(r, "track_id") >= 0:
            by[_i(r, "track_id")].append(r)
    out = {}
    for tid, rs in by.items():
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
    usable, cens = defaultdict(list), defaultdict(list)
    side, att = {}, defaultdict(int)
    for a in rep.values():
        att[a["condition"]] += 1
        if not a["converged"]:
            continue
        # a reversal recorded at a clamp is not a reversal: such a track bounds the threshold
        if a["ceiling_inside_threshold"]:
            cens[a["condition"]].append(a["threshold_ms"]); side[a["condition"]] = "ceiling"
        elif a["floor_inside_threshold"]:
            cens[a["condition"]].append(a["threshold_ms"]); side[a["condition"]] = "floor"
        else:
            usable[a["condition"]].append(a["threshold_ms"])
    by = defaultdict(list)
    for r in rows:
        if r.get("phase") in ("main", "catch") and _usable(r):
            by[r["condition"]].append(r)
    out = {}
    for name, c in conds.items():
        tr = [r for r in by.get(name, []) if r.get("phase") == "main"]
        dd = np.array([abs(_f(r, "delta_realised_ms")) for r in tr])
        yy = np.array([_i(r, "correct", 0) for r in tr], dtype=bool)
        f = pf_fit(dd, yy, lapse_max=cfg.lapse_max) if dd.size >= 8 else None
        u = usable.get(name, [])
        out[name] = CondResult(
            name, c.lag_ms, c.kind, u, cens.get(name, []), side.get(name, ""), att.get(name, 0),
            float(np.exp(np.mean(np.log(u)))) if u else None, f, len(tr),
            float(yy.mean()) if yy.size else float("nan"),
            float(np.mean([1 - _i(r, "pairing_preserved", 1) for r in tr])) if tr else 0.0)
    return out


def trend_test(res: Dict[str, CondResult], n_perm: int = 10000, seed: int = 7) -> dict:
    """Does threshold rise with the lag? Spearman over tracks, permuted for a calibrated p."""
    pts = [(r.lag_ms, t) for r in res.values() if r.kind == "pair" for t in r.track_thresholds]
    if len({p for p, _ in pts}) < 3:
        return {"ok": False, "reason": "fewer than three lags with usable tracks"}
    x = np.array([p for p, _ in pts]); y = np.log([t for _, t in pts])
    rho = float(stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    null = np.array([stats.spearmanr(x, rng.permutation(y)).statistic for _ in range(n_perm)])
    return {"ok": True, "rho": rho, "p_one_sided": float((null >= rho).mean()),
            "n_tracks": len(pts), "n_levels": len({p for p, _ in pts})}


def what_a_adds(res: Dict[str, CondResult]) -> dict:
    """B-alone threshold divided by each lag's: how much the low tone helped, lag by lag."""
    solo = res.get("b_alone")
    if not solo or not (solo.geomean_ms or solo.censored):
        return {"ok": False, "reason": "B-alone was not measured or has no threshold"}
    # A censored B-alone is a LOWER bound on that threshold, which is still enough for the one
    # inference this comparison carries: a pair condition below the bound beat B alone. The
    # ratios are then lower bounds too, and are reported as such rather than as infinities.
    bound = (float(np.exp(np.mean(np.log(solo.censored_thresholds))))
             if solo.censored and not solo.geomean_ms else None)
    ref = solo.geomean_ms or bound
    out = {}
    for r in sorted(res.values(), key=lambda r: r.lag_ms):
        if r.kind == "pair" and r.geomean_ms:
            out[r.lag_ms] = ref / r.geomean_ms
    return {"ok": True, "b_alone_ms": solo.geomean_ms, "b_alone_bound_ms": bound,
            "is_lower_bound": bound is not None, "ratio_by_lag": out}


# ---------------------------------------------------------------------------- strategy
def _onsets(cfg: Config, jitter: np.ndarray, target_pair: Optional[int], realised: float):
    b = cfg.nominal_onsets() + jitter
    if target_pair is not None:
        b = b.copy()
        b[target_pair] += realised
    return b


def _earliness(cfg: Config, b: np.ndarray) -> float:
    """How early the most-early candidate B looks against a grid fitted to the safe pairs."""
    k = np.arange(cfg.n_pairs)
    safe = [i for i in range(cfg.n_pairs) if i not in cfg.target_pairs]
    phase = np.median(b[safe] - k[safe] * cfg.slot_ms)
    res = b - (k * cfg.slot_ms + phase)
    return float(-min(res[i] for i in cfg.target_pairs))


def single_channel_evidence(cfg: Config, r: dict) -> Optional[float]:
    """For one trial: how much better the B channel ALONE made the target interval look (ms).

    Positive means a B-only listener would have leaned towards the right answer on this trial.
    Rebuilt from the logged jitter, so it needs nothing the runner did not record.
    """
    try:
        j1 = np.array([float(v) for v in r["jitter_1"].split(";")])
        j2 = np.array([float(v) for v in r["jitter_2"].split(";")])
    except (KeyError, ValueError, AttributeError):
        return None
    tp, pair = _i(r, "target_position"), _i(r, "target_pair")
    rd = _f(r, "delta_realised_ms")
    if tp not in (1, 2) or pair < 0 or not np.isfinite(rd):
        return None
    js = (j1, j2)
    tgt = _onsets(cfg, js[tp - 1], pair, rd)
    std = _onsets(cfg, js[2 - tp], None, 0.0)
    return _earliness(cfg, tgt) - _earliness(cfg, std)


def _logit(X, y):
    def nll(b):
        z = X @ b
        return float(np.sum(np.logaddexp(0, z) - y * z))
    def grad(b):
        return X.T @ (expit(X @ b) - y)
    # expit, not 1/(1+exp(-z)): a strong positive control drives z far enough that the naive
    # form overflows. The answer happens to survive that; the gradient near it need not.
    b = optimize.minimize(nll, np.zeros(X.shape[1]), jac=grad, method="BFGS").x
    p = expit(X @ b)
    H = X.T @ (X * (p * (1 - p))[:, None])
    try:
        se = np.sqrt(np.diag(np.linalg.inv(H)))
    except np.linalg.LinAlgError:
        se = np.full(X.shape[1], np.nan)
    return b, se


def strategy_check(rows: Sequence[dict], cfg: Config) -> dict:
    """Do the answers follow the B channel's own irregularity, beyond the displacement itself?

    Logistic regression of correct on log(delta) and the single-channel evidence, fitted
    separately to the pair conditions and to B-alone. The coefficient on the evidence is the
    quantity: near zero means the B channel's irregularity did not drive the answers.
    """
    out = {}
    for label, keep in (("pair conditions", lambda r: r.get("kind") == "pair"),
                        ("B alone", lambda r: r.get("kind") == "b_alone")):
        X, y = [], []
        for r in rows:
            if r.get("phase") != "main" or not _usable(r) or not keep(r):
                continue
            e = single_channel_evidence(cfg, r)
            dl = abs(_f(r, "delta_realised_ms"))
            if e is None or not np.isfinite(dl) or dl <= 0:
                continue
            X.append((math.log(dl), e)); y.append(_i(r, "correct", 0))
        if len(y) < 30:
            out[label] = {"ok": False, "n": len(y)}
            continue
        X = np.array(X); y = np.array(y, float)
        sd = X[:, 1].std() or 1.0
        Z = np.column_stack([np.ones(len(y)), X[:, 0] - X[:, 0].mean(), X[:, 1] / sd])
        b, se = _logit(Z, y)
        coef, s = float(b[2]), float(se[2])
        out[label] = {"ok": True, "n": len(y), "coef_per_sd": coef, "se": s,
                      "z": coef / s if s > 0 else float("nan"),
                      "p_two_sided": float(2 * stats.norm.sf(abs(coef / s))) if s > 0 else float("nan"),
                      "evidence_sd_ms": float(sd)}
    return out


def position_check(rows: Sequence[dict]) -> dict:
    """Proportion correct with the target in pair N against pair N-1, within each condition.

    The staircase never sees the position, so the two are matched in delta by construction and
    a raw comparison is fair. Permutation within condition gives the p-value.
    """
    by = defaultdict(list)
    for r in rows:
        if r.get("phase") == "main" and _usable(r):
            by[r["condition"]].append((_i(r, "target_pair"), _i(r, "correct", 0)))
    pairs = sorted({p for v in by.values() for p, _ in v if p >= 0})
    if len(pairs) != 2:
        return {"ok": False}
    late, early = pairs[1], pairs[0]
    def diff(groups):
        a = [c for v in groups for p, c in v if p == late]
        b = [c for v in groups for p, c in v if p == early]
        return (np.mean(a) if a else np.nan) - (np.mean(b) if b else np.nan), len(a), len(b)
    obs, na, nb = diff(by.values())
    rng = np.random.default_rng(3); null = []
    for _ in range(4000):
        perm = []
        for v in by.values():
            ps = [p for p, _ in v]; cs = [c for _, c in v]
            perm.append(list(zip(rng.permutation(ps), cs)))
        null.append(diff(perm)[0])
    null = np.array(null)
    return {"ok": True, "last_minus_second_last": float(obs), "n_last": na, "n_second_last": nb,
            "p_two_sided": float((np.abs(null) >= abs(obs)).mean())}


def diagnostics(rows, metas, cfg: Config, res) -> dict:
    main = [r for r in rows if r.get("phase") == "main" and _usable(r)]
    catch = [r for r in rows if r.get("phase") == "catch" and _usable(r)]
    catch_p = float(np.mean([_i(r, "correct", 0) for r in catch])) if catch else float("nan")
    resp1 = [str(r.get("response", "")).strip() == "1" for r in main]
    n1 = sum(resp1)
    lvl = next((float(c["measured_db_spl"]) for m in metas
                for c in [m.get("calibration") or {}] if c.get("measured_db_spl") is not None), None)
    half = len(main) // 2
    dur = []
    import datetime as dt
    for m in metas:
        try:
            dur.append((dt.datetime.fromisoformat(m["end_time"])
                        - dt.datetime.fromisoformat(m["start_time"])).total_seconds() / 60)
        except (KeyError, ValueError, TypeError):
            pass
    n_cens = sum(len(r.censored_thresholds) for r in res.values())
    return {
        "n_main": len(main), "n_catch": len(catch),
        "n_practice": sum(1 for r in rows if r.get("phase") == "practice"),
        "n_timeouts": sum(1 for r in rows if _i(r, "timed_out", 0)),
        "catch_p_correct": catch_p,
        "catch_ok": bool(catch and (1 - catch_p) <= cfg.max_catch_miss_rate),
        "press_1_rate": n1 / max(len(main), 1),
        "press_1_p": float(stats.binomtest(n1, max(len(main), 1), .5).pvalue),
        "drift_rho": float(stats.spearmanr(np.arange(len(main)), resp1).statistic)
        if len(main) > 10 else float("nan"),
        "drift_p": float(stats.spearmanr(np.arange(len(main)), resp1).pvalue)
        if len(main) > 10 else float("nan"),
        "first_half": float(np.mean([_i(r, "correct", 0) for r in main[:half]])) if half else np.nan,
        "second_half": float(np.mean([_i(r, "correct", 0) for r in main[half:]])) if half else np.nan,
        "pairing_flip_fraction": float(np.mean([1 - _i(r, "pairing_preserved", 1)
                                                for r in main if r.get("kind") == "pair"]))
        if main else 0.0,
        "tracks_converged": sum(len(r.track_thresholds) for r in res.values()) + n_cens,
        "tracks_censored": n_cens,
        "tracks_attempted": sum(r.n_tracks_attempted for r in res.values()),
        "replay_ok": all(a["logged_deltas_match"] for a in replay_tracks(rows, cfg).values()),
        "level_db_spl": lvl, "duration_min": sum(dur) if dur else None,
    }


def analyse(dirs: Sequence[Path], cfg: Optional[Config] = None) -> str:
    rows, metas = load([Path(d) for d in dirs])
    cfg = cfg or Config.from_dict(metas[0]["config"])
    d = validate(cfg)
    res = thresholds(rows, cfg)
    g = diagnostics(rows, metas, cfg, res)
    L: List[str] = []
    A = L.append
    A("=" * 78)
    A(f"jittered A-B pairs  |  {len(metas)} session(s), participant(s) "
      f"{', '.join(sorted({m.get('participant_code', '?') for m in metas}))}")
    A(f"config {cfg.hash()}   A {cfg.f_a_hz:.0f} / B {cfg.f_b_hz:.0f} Hz, {cfg.tone_ms:g} ms, "
      f"{cfg.n_pairs} pairs at {cfg.rate_hz:g} Hz, jitter +-{cfg.pair_jitter_ms:g} ms")
    A(f"threshold = {100 * target_proportion(cfg.n_down):.1f}% correct, geometric mean of the "
      f"last {cfg.n_final_reversals} reversals")
    A("")
    A("-- session validity " + "-" * 57)
    A(f"  main {g['n_main']}   catch {g['n_catch']}   practice {g['n_practice']}"
      + (f"   timed out {g['n_timeouts']}" if g["n_timeouts"] else "")
      + (f"   {g['duration_min']:.0f} min" if g["duration_min"] else ""))
    if g["n_catch"]:
        A(f"  catch at {cfg.catch_delta_ms:g} ms: {g['catch_p_correct']:.1%} correct -> "
          f"{'OK' if g['catch_ok'] else 'FAILED: every threshold below is unsafe'}")
    A(f"  tracks converged {g['tracks_converged']}/{g['tracks_attempted']}"
      + (f", {g['tracks_censored']} boundary-limited" if g["tracks_censored"] else "")
      + f"; every threshold recomputes from the log: {g['replay_ok']}")
    A(f"  pressed '1' on {100 * g['press_1_rate']:.1f}% (p={g['press_1_p']:.3f}); drift "
      f"rho={g['drift_rho']:+.3f} (p={g['drift_p']:.3f}); halves {100 * g['first_half']:.1f}% / "
      f"{100 * g['second_half']:.1f}% correct")
    A(f"  the early B sat nearer the previous pair's A on {100 * g['pairing_flip_fraction']:.1f}% "
      "of pair trials")
    A("  WARNING: level was NOT measured; levels are nominal." if g["level_db_spl"] is None
      else f"  level: {g['level_db_spl']:.1f} dB SPL for one tone")
    A("")
    A("-- thresholds " + "-" * 63)
    A(f"  {'condition':<9} {'lag':>5} {'tracks':>7} {'staircase':>11} {'fit':>8} {'%corr':>7} "
      f"{'model':>7}")
    for c in d.conditions:
        r = res[c.name]
        g_ = (f"{r.geomean_ms:8.2f} ms" if r.geomean_ms else
              f"  >{np.exp(np.mean(np.log(r.censored_thresholds))):5.1f}*" if r.censored else "       -  ")
        fit = f"{r.fit.threshold_ms:6.2f}" if r.fit else "     -"
        mv = d.model_index[c.name]
        A(f"  {c.name:<9} {('-' if c.kind == 'b_alone' else f'{c.lag_ms:g}'):>5} "
          f"{len(r.track_thresholds)}/{r.n_tracks_attempted:<5} {g_:>11} {fit:>8} "
          f"{100 * r.prop_correct:6.1f}% {('-' if mv != mv else f'{mv:.3f}'):>7}")
    if any(r.censored for r in res.values()):
        A("  * boundary-limited: a bound, not a threshold; excluded from every summary below")
    A("")
    A("-- did the listener use A? " + "-" * 50)
    w = what_a_adds(res)
    if w["ok"]:
        ge = "\u2265" if w["is_lower_bound"] else ""
        ref = (f"{w['b_alone_ms']:.1f} ms" if w["b_alone_ms"] else
               f"at least {w['b_alone_bound_ms']:.0f} ms (boundary-limited, so a lower bound)")
        A(f"  B alone: {ref}. Each lag's threshold against it:")
        for lag, ratio in w["ratio_by_lag"].items():
            A(f"    lag {lag:5g} ms   {ge}{ratio:5.2f}x better than B alone"
              + ("   <-- NO BETTER: A was not being used here"
                 if ratio < 1.2 and not w["is_lower_bound"] else ""))
    else:
        A(f"  not computed: {w['reason']}")
    sc = strategy_check(rows, cfg)
    for label, v in sc.items():
        if v.get("ok"):
            A(f"  {label:16} answers vs the B channel's own irregularity: "
              f"{v['coef_per_sd']:+.2f} per SD (z={v['z']:+.2f}, p={v['p_two_sided']:.3f}, "
              f"n={v['n']})")
    if all(v.get("ok") for v in sc.values()):
        A("    B alone is the positive control: it SHOULD track the irregularity. If it does and")
        A("    the pair conditions do not, the pair answers were not coming from B on its own.")
    A("")
    A("-- shape " + "-" * 68)
    t = trend_test(res)
    if t["ok"]:
        A(f"  threshold rises with the lag: rho = {t['rho']:+.3f} over {t['n_tracks']} tracks at "
          f"{t['n_levels']} lags, permutation p = {t['p_one_sided']:.4f} (one sided)")
        A("    Consistent with temporal coherence, and also with a Weber account of the A-B gap.")
    pc = position_check(rows)
    if pc.get("ok"):
        A(f"  target in the last pair vs the one before: {100 * pc['last_minus_second_last']:+.1f} "
          f"points correct (p={pc['p_two_sided']:.3f}; n={pc['n_last']} / {pc['n_second_last']})")
    A("")
    A("-- what this does not establish " + "-" * 45)
    A("  * one listener. Nothing here generalises.")
    A("  * a rise with the lag does not by itself separate coherence from interval timing;")
    A("    the B-alone comparison establishes that A was used, not how.")
    A("=" * 78)
    return "\n".join(L)
