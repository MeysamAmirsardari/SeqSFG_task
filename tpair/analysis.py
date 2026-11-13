"""From trials to thresholds to a verdict, with the limits of each step stated.

What the measurements can settle
---------------------------------
  the decoupled probes (scatter timing)
      A and B scattered independently and moved exactly as in a target token, at the ceiling.
      Each tone on its own carries what it carries in the real trials; the relation is gone. If
      listeners sit at chance there while converging well below the ceiling in the token
      conditions, those thresholds came through the A-B relation. This is the load-bearing
      comparison for every session run on scatter timing.

  every pair condition below B-alone (sessions with a B-alone control)
      The listener used A. B-alone measures what is available from the high tone on its own,
      with the same timing and the same displacement; a pair condition that beats it can only
      have done so through the A-B relation.

  threshold rising with the lag
      Using A gets harder as A and B separate in time. Consistent with temporal coherence, and
      also with any account in which a longer interval is harder to judge -- a Weber account of
      the A-B gap predicts a rise too. The rise alone does not choose between them.

  the strategy check
      Every tone's onset is logged (the grid jitter, for older sessions), so the analysis can
      reconstruct how displaced A and B each looked ON THEIR OWN in each interval, and ask
      whether the answers followed that. A listener using the relation should not care, once
      delta itself is accounted for; a listener judging one tone's timing should care a great
      deal.

  in against out
      With both directions, every comparison above is made within a direction, each against its
      own B alone. The out/in ratio at each lag is then descriptive: an "out" shift at large lags
      also opens a silent gap between A and B (between-channel gap detection), a cue "in" never
      offers, so a difference there is not by itself about grouping.
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

from .config import REL_WORD, WORD, Config, validate
from .observer import tone_evidence


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
    passes_a_onset_fraction: float = 0.0
    direction: str = "in"
    opens_gap_fraction: float = 0.0
    context_lag_ms: Optional[float] = None

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
        t = Track(cfg.track_cfg(rs[0].get("kind", "pair")), rs[0]["condition"], 0)
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
        # A condition where one track hit the ceiling has no geometric mean, even if another
        # converged: a listener near chance there can still walk a track down by luck (about one
        # track in 80 in simulation), and keeping only that track would report the luck. Such a
        # condition is reported as a bound over all its tracks.
        mixed = bool(u) and side.get(name) == "ceiling"
        out[name] = CondResult(
            name, c.lag_ms, c.kind, u, cens.get(name, []), side.get(name, ""), att.get(name, 0),
            float(np.exp(np.mean(np.log(u)))) if u and not mixed else None, f, len(tr),
            float(yy.mean()) if yy.size else float("nan"),
            float(np.mean([1 - _i(r, "pairing_preserved", 1) for r in tr])) if tr else 0.0,
            float(np.mean([_i(r, "passes_a_onset", 0) for r in tr])) if tr else 0.0,
            c.direction,
            float(np.mean([_i(r, "opens_gap", 0) for r in tr])) if tr else 0.0,
            c.context_lag_ms)
    return out


def directions(res: Dict[str, CondResult]) -> List[str]:
    return list(dict.fromkeys(r.direction for r in res.values()))


def _only(res: Dict[str, CondResult], direction: Optional[str]) -> Dict[str, CondResult]:
    direction = direction or directions(res)[0]
    return {k: r for k, r in res.items() if r.direction == direction}


def trend_test(res: Dict[str, CondResult], n_perm: int = 10000, seed: int = 7,
               direction: Optional[str] = None) -> dict:
    """Does threshold rise with the lag? Spearman over tracks, permuted for a calibrated p.
    Within one direction (the first, unless named)."""
    res = _only(res, direction)
    pts = [(r.lag_ms, t) for r in res.values() if r.kind == "pair" for t in r.track_thresholds]
    if len({p for p, _ in pts}) < 3:
        return {"ok": False, "reason": "fewer than three lags with usable tracks"}
    x = np.array([p for p, _ in pts]); y = np.log([t for _, t in pts])
    rho = float(stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    null = np.array([stats.spearmanr(x, rng.permutation(y)).statistic for _ in range(n_perm)])
    return {"ok": True, "rho": rho, "p_one_sided": float((null >= rho).mean()),
            "n_tracks": len(pts), "n_levels": len({p for p, _ in pts})}


def what_a_adds(res: Dict[str, CondResult], direction: Optional[str] = None) -> dict:
    """B-alone threshold divided by each lag's: how much the low tone helped, lag by lag.
    Within one direction (the first, unless named), against that direction's own B alone."""
    res = _only(res, direction)
    solo = next((r for r in res.values() if r.kind == "b_alone"), None)
    if not solo or not (solo.geomean_ms or solo.censored):
        return {"ok": False, "reason": "B-alone was not measured or has no threshold"}
    # A censored B-alone is a LOWER bound on that threshold, which is still enough for the one
    # inference this comparison carries: a pair condition below the bound beat B alone. The
    # ratios are then lower bounds too, and are reported as such rather than as infinities.
    bound = (float(np.exp(np.mean(np.log(solo.censored_thresholds))))
             if solo.censored and not solo.geomean_ms else None)
    ref = solo.geomean_ms or bound
    # A lag whose tracks hit the ceiling is the opposite case. Dropping those tracks would keep
    # only the lucky ones and make a listener who ignored A look as if A helped; so they are
    # pooled with the usable tracks as values the threshold is at least, and the ratio becomes
    # an UPPER bound on what A added there.
    out, upper = {}, set()
    for r in sorted(res.values(), key=lambda r: r.lag_ms):
        if r.kind != "pair":
            continue
        ceil = r.censored_thresholds if r.censor_side == "ceiling" else []
        vals = list(r.track_thresholds) + list(ceil)
        if not vals:
            continue
        out[r.lag_ms] = ref / float(np.exp(np.mean(np.log(vals))))
        if ceil:
            upper.add(r.lag_ms)
    return {"ok": True, "b_alone_ms": solo.geomean_ms, "b_alone_bound_ms": bound,
            "is_lower_bound": bound is not None, "ratio_by_lag": out,
            "upper_bound_lags": sorted(upper)}


def _pooled(r: CondResult) -> Tuple[Optional[float], str]:
    """Usable tracks with ceiling-limited ones as values the threshold is at least.
    The mark says what the number is: '' a threshold, '>=' a lower bound."""
    ceil = r.censored_thresholds if r.censor_side == "ceiling" else []
    vals = list(r.track_thresholds) + list(ceil)
    if not vals:
        return None, ""
    return float(np.exp(np.mean(np.log(vals)))), ("\u2265" if ceil else "")


def in_vs_out(res: Dict[str, CondResult]) -> dict:
    """out / in threshold at each lag, and for B alone. Above 1: late shifts were harder."""
    if not {"in", "out"} <= set(directions(res)):
        return {"ok": False, "reason": "needs both directions"}
    key = lambda r: ("b_alone" if r.kind == "b_alone" else r.lag_ms)
    i = {key(r): r for r in res.values() if r.direction == "in"}
    o = {key(r): r for r in res.values() if r.direction == "out"}
    rows = []
    for k in [k for k in i if k in o]:
        (ti, mi), (to, mo) = _pooled(i[k]), _pooled(o[k])
        rows.append({"level": k, "in_ms": ti, "in_mark": mi, "out_ms": to, "out_mark": mo,
                     "ratio": (to / ti) if (ti and to) else None,
                     "bounded": bool(mi or mo),
                     "out_gap_fraction": o[k].opens_gap_fraction})
    return {"ok": True, "rows": rows}


# ---------------------------------------------------------------------------- strategy
def _onsets(cfg: Config, jitter: np.ndarray, target_pair: Optional[int], realised: float):
    b = cfg.nominal_onsets() + jitter
    if target_pair is not None:
        b = b.copy()
        b[target_pair] += realised
    return b


def _displacement(cfg: Config, b: np.ndarray, sign: float = -1.0) -> float:
    """How far the most-displaced candidate B looks, in the shift's own direction (early for
    sign -1, late for +1), against a grid fitted to the safe pairs."""
    k = np.arange(cfg.n_pairs)
    safe = [i for i in range(cfg.n_pairs) if i not in cfg.target_pairs]
    phase = np.median(b[safe] - k[safe] * cfg.slot_ms)
    res = b - (k * cfg.slot_ms + phase)
    return float(max(sign * res[i] for i in cfg.target_pairs))


def _parse(r, k) -> Optional[np.ndarray]:
    v = str(r.get(k, "") or "").strip()
    if not v:
        return None
    try:
        return np.array([float(x) for x in v.split(";")])
    except ValueError:
        return None


def channel_evidence(cfg: Config, r: dict) -> Optional[Dict[str, float]]:
    """For one trial: how much more displaced each tone looked ON ITS OWN in the target interval
    than in the standard, in the direction it moved (ms), per channel that moved.

    Positive means a listener judging that tone's own timing would have leaned towards the right
    answer. Rebuilt from the logged onsets; for grid sessions logged before onsets were, from the
    jitter (B only, since A never moved there).
    """
    tp = _i(r, "target_position")
    if tp not in (1, 2):
        return None
    on = {c: (_parse(r, f"{c}_onsets_1"), _parse(r, f"{c}_onsets_2")) for c in ("a", "b")}
    if on["b"][0] is None:
        e = single_channel_evidence(cfg, r)
        return None if e is None else {"B": e}
    out = {}
    for c, key in (("a", "shift_a_ms"), ("b", "shift_b_ms")):
        move = _f(r, key, 0.0)
        x_t, x_s = on[c][tp - 1], on[c][2 - tp]
        if not move or x_t is None or x_s is None or not np.isfinite(x_t).all():
            continue
        d = 1.0 if move > 0 else -1.0
        out[c.upper()] = tone_evidence(cfg, x_t, d) - tone_evidence(cfg, x_s, d)
    return out or None


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
    sign = 1.0 if rd > 0 else -1.0
    tgt = _onsets(cfg, js[tp - 1], pair, rd)
    std = _onsets(cfg, js[2 - tp], None, 0.0)
    return _displacement(cfg, tgt, sign) - _displacement(cfg, std, sign)


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
    """Do the answers follow either tone's own timing, beyond the change itself?

    Logistic regression of correct on log(delta) and, for each tone that moves, how displaced it
    looked on its own (`channel_evidence`), fitted separately to the token conditions of each
    direction, to B alone and to the decoupled probes (where delta is fixed and drops out). The
    coefficients on the evidence are the quantity: near zero means neither tone's own timing
    drove the answers. The top-level numbers are those of the tone with the larger z.
    """
    out = {}
    dirs = cfg.directions
    groups = []
    for d in dirs:
        sfx = f", {d}" if len(dirs) > 1 else ""
        same = (lambda r, d=d: len(dirs) == 1 or r.get("direction") == d)
        groups += [(f"pair conditions{sfx}", "main",
                    lambda r, s=same: r.get("kind") == "pair" and s(r)),
                   (f"B alone{sfx}", "main", lambda r, s=same: r.get("kind") == "b_alone" and s(r))]
    groups.append(("decoupled probes", "probe", lambda r: True))
    for label, phase, keep in groups:
        X, E, y = [], [], []
        for r in rows:
            if r.get("phase") != phase or not _usable(r) or not keep(r):
                continue
            e = channel_evidence(cfg, r)
            dl = abs(_f(r, "delta_realised_ms"))
            if e is None or not np.isfinite(dl) or dl <= 0:
                continue
            X.append(math.log(dl)); E.append(e); y.append(_i(r, "correct", 0))
        chans = sorted({c for e in E for c in e})
        rows_ok = [i for i, e in enumerate(E) if all(c in e for c in chans)]
        if len(rows_ok) < 30 or not chans:
            out[label] = {"ok": False, "n": len(rows_ok)}
            continue
        y_ = np.array([y[i] for i in rows_ok], float)
        cols = [np.ones(len(rows_ok))]
        x = np.array([X[i] for i in rows_ok])
        if x.std() > 1e-9:
            cols.append(x - x.mean())
        sds = {}
        for c in chans:
            v = np.array([E[i][c] for i in rows_ok])
            sds[c] = float(v.std() or 1.0)
            cols.append(v / sds[c])
        b, se = _logit(np.column_stack(cols), y_)
        per = {}
        for j, c in enumerate(chans):
            k = len(cols) - len(chans) + j
            coef, s_ = float(b[k]), float(se[k])
            per[c] = {"coef_per_sd": coef, "se": s_, "z": coef / s_ if s_ > 0 else float("nan"),
                      "p_two_sided": float(2 * stats.norm.sf(abs(coef / s_))) if s_ > 0 else float("nan"),
                      "evidence_sd_ms": sds[c]}
        top = max(per, key=lambda c: per[c]["z"] if per[c]["z"] == per[c]["z"] else -np.inf)
        out[label] = {"ok": True, "n": len(rows_ok), "channels": per, "tone": top, **per[top]}
    return out


def probe_summary(rows: Sequence[dict], cfg: Config) -> dict:
    """Proportion correct on the decoupled probes, overall and per direction.

    `p_below` tests whether performance sits below the 79.4% the staircases converge on: if it
    does, a listener with only the tones' own timing could not have produced those thresholds
    even at the ceiling.
    """
    pr = [r for r in rows if r.get("phase") == "probe" and _usable(r)]
    out = {}
    groups = {"all": pr}
    for d in cfg.directions:
        if len(cfg.directions) > 1:
            groups[d] = [r for r in pr if r.get("direction") == d]
    for label, rs in groups.items():
        n = len(rs)
        if not n:
            continue
        k = int(sum(_i(r, "correct", 0) for r in rs))
        bt = stats.binomtest(k, n, 0.5)
        ci = bt.proportion_ci(confidence_level=0.95, method="wilson")
        out[label] = {"n": n, "k": k, "p": k / n, "ci": (float(ci.low), float(ci.high)),
                      "p_vs_chance": float(bt.pvalue),
                      "p_below": float(stats.binomtest(k, n, target_proportion(cfg.n_down),
                                                       alternative="less").pvalue),
                      "delta_ms": float(np.median([_f(r, "delta_ms") for r in rs]))}
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
        "n_probe": sum(1 for r in rows if r.get("phase") == "probe" and _usable(r)),
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


def _context_section(A, rows, cfg: Config, res) -> None:
    from . import context as X
    word = REL_WORD if cfg.shift == "split" else WORD
    A("")
    A("-- does the context carry over? " + "-" * 45)
    for L, e in X.effects(res, cfg).items():
        if not e["ok"]:
            A(f"  test {cfg.lag_pct(L):g}%: {e['reason']}")
            continue
        lo, hi = e["context_low"], e["context_high"]
        A(f"  test dT {cfg.lag_pct(L):g}% ({L:g} ms); contexts {cfg.lag_pct(lo):g}% (bound side) and "
          f"{cfg.lag_pct(hi):g}% (segregated side)")
        A(f"    {'':<12} {'ctx ' + format(cfg.lag_pct(lo), 'g') + '%':>10} "
          f"{'stationary':>11} {'ctx ' + format(cfg.lag_pct(hi), 'g') + '%':>10}   ln hi/lo")
        for d, v in e["per_direction"].items():
            T = v["T"]
            A(f"    {d + ' (' + word[d] + ')':<12} {T[lo]:8.1f} ms {T[L]:8.1f} ms {T[hi]:8.1f} ms   "
              f"{v['ln_hi_lo']:+.2f}")
        A(f"  P (persistence > 0 > contrast) = {e['P']:+.2f}"
          + (f" +- {e['se_P']:.2f}" if e["se_P"] else ""))
        if e["D"] is not None:
            A(f"  D (decoy: the switch read as a change, flips with direction) = {e['D']:+.2f}")
        A(f"  S (both mismatched contexts against stationary) = {e['S']:+.2f}")
        A(f"  -> {X.verdict(e)}")
        if not e["both_directions"]:
            A("     One direction only: P cannot be told from the decoy. Run both.")
    dec = X.decay(rows, cfg)
    if any(v.get("ok") for v in dec.values()):
        A("  within the trial, accuracy against the time since the switch (logistic, per SD):")
        A("    " + "   ".join(f"ctx {cfg.lag_pct(C):g}%: {v['coef_per_sd']:+.2f} (z={v['z']:+.1f})"
                            for C, v in dec.items() if v.get("ok")))
        A("    Under persistence the bound context helps less, and the segregated one hurts less,")
        A("    the longer ago it ended. Exploratory.")
    co = X.carry_over(rows)
    if co.get("ok"):
        A(f"  the previous trial's context: {co['coef_per_sd']:+.2f} per SD (z={co['z']:+.1f}, "
          f"n={co['n']}). Prior-stimulus effects in streaming outlast a trial; this checks them.")


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
    timing = (f"scattered, onsets >= {d.geometry['min_ioi_ms']:.1f} ms apart, {cfg.rate_hz:g} Hz "
              f"on average; lag split between A and B" if cfg.timing == "scatter" else
              f"at {cfg.rate_hz:g} Hz, jitter +-{cfg.pair_jitter_ms:g} ms")
    A(f"config {cfg.hash()}   A {cfg.f_a_hz:.0f} / B {cfg.f_b_hz:.0f} Hz, {cfg.tone_ms:g} ms, "
      f"{cfg.n_pairs} tokens {timing}")
    A(f"threshold = {100 * target_proportion(cfg.n_down):.1f}% correct, geometric mean of the "
      f"last {cfg.n_final_reversals} reversals")
    A("")
    A("-- session validity " + "-" * 57)
    A(f"  main {g['n_main']}   catch {g['n_catch']}"
      + (f"   probes {g['n_probe']}" if g["n_probe"] else "") + f"   practice {g['n_practice']}"
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

    A("  WARNING: level was NOT measured; levels are nominal." if g["level_db_spl"] is None
      else f"  level: {g['level_db_spl']:.1f} dB SPL for one tone")
    A("")
    A("-- thresholds " + "-" * 63)
    disagree: List[str] = []
    A(f"  {'condition':<20} {'dT':>5} {'tracks':>7} {'staircase':>11} {'fit':>8} {'%corr':>7} "
      f"{'model':>7} {'B past A':>9} {'gap':>5}")
    for c in d.conditions:
        r = res[c.name]
        bound = list(r.track_thresholds) + list(r.censored_thresholds)
        g_ = (f"{r.geomean_ms:8.2f} ms" if r.geomean_ms else
              f"  >{np.exp(np.mean(np.log(bound))):5.1f}*" if r.censored else "       -  ")
        fit = f"{r.fit.threshold_ms:6.2f}" if r.fit else "     -"
        if (r.geomean_ms and r.fit and np.isfinite(r.fit.threshold_ms) and r.fit.threshold_ms > 0
                and abs(math.log(r.fit.threshold_ms / r.geomean_ms)) > math.log(2.0)):
            fit += "!"
            disagree.append(c.name)
        mv = d.model_index[c.name]
        pair = c.kind == "pair"
        A(f"  {c.name:<20} {(f'{cfg.lag_pct(c.lag_ms):g}%' if pair else '-'):>5} "
          f"{len(r.track_thresholds)}/{r.n_tracks_attempted:<5} {g_:>11} {fit:>8} "
          f"{100 * r.prop_correct:6.1f}% {('-' if mv != mv else f'{mv:.3f}'):>7} "
          f"{(f'{100 * r.passes_a_onset_fraction:.0f}%' if pair and c.direction != 'out' else '-'):>9} "
          f"{(f'{100 * r.opens_gap_fraction:.0f}%' if pair and c.direction != 'in' else '-'):>5}")
    if any(r.censored for r in res.values()):
        A("  * boundary-limited: a bound over all of the condition's tracks, not a threshold;")
        A("    excluded from every summary below")
    if disagree:
        A(f"  ! the staircase and a psychometric fit to the same trials differ by more than 2x "
          f"({', '.join(disagree)}).")
        A("    A staircase can walk down on a lucky run; trust neither number there until the")
        A("    condition is run again.")
    if cfg.can_move_early:
        A("  B past A: changes that put B at or before its own A's onset")
    if cfg.can_move_late:
        A("  gap: changes that left a silent gap between A's offset and B's onset")
    A("")
    A("-- did the listener use the A-B relation? " + "-" * 36)
    multi = len(directions(res)) > 1
    word = REL_WORD if cfg.shift == "split" else WORD
    ps = probe_summary(rows, cfg)
    if ps:
        A(f"  decoupled probes: A and B scattered independently, moved as a target token at "
          f"{ps['all']['delta_ms']:g} ms. Each tone's own timing is all there is.")
        for label, v in ps.items():
            A(f"    {('all' if label == 'all' else label):>4}: {v['k']}/{v['n']} = "
              f"{100 * v['p']:.0f}% correct (95% CI {100 * v['ci'][0]:.0f}-{100 * v['ci'][1]:.0f}%; "
              f"vs chance p={v['p_vs_chance']:.3f}; below "
              f"{100 * target_proportion(cfg.n_down):.1f}% p={v['p_below']:.3f})")
        conv = [(r.name, t) for r in res.values() if r.kind == "pair" for t in r.track_thresholds]
        under = [t for _, t in conv if t < 0.9 * cfg.delta_max_ms]
        n_pair = sum(r.n_tracks_attempted for r in res.values() if r.kind == "pair")
        if under:
            A(f"    {len(under)}/{n_pair} token tracks converged below {0.9 * cfg.delta_max_ms:.0f} ms, "
              "where the probes say the tones alone are no")
            A("    help: those thresholds came through the relation. Probes well above (or below)")
            A("    chance would mean a route around it, and thresholds near them would be unsafe.")
        else:
            A(f"    no token track converged below {0.9 * cfg.delta_max_ms:.0f} ms: nothing here "
              "shows the relation was used.")
    for direction in (directions(res) if any(r.kind == "b_alone" for r in res.values()) else []):
        w = what_a_adds(res, direction)
        if multi:
            A(f"  [{direction}: B {word[direction]}]")
        if not w["ok"]:
            A(f"  not computed: {w['reason']}")
            continue
        ge = "\u2265" if w["is_lower_bound"] else ""
        ref = (f"{w['b_alone_ms']:.1f} ms" if w["b_alone_ms"] else
               f"at least {w['b_alone_bound_ms']:.0f} ms (boundary-limited, so a lower bound)")
        A(f"  B alone: {ref}. Each lag's threshold against it:")
        for lag, ratio in w["ratio_by_lag"].items():
            up = lag in w["upper_bound_lags"]
            if up and w["is_lower_bound"]:
                A(f"    dT {cfg.lag_pct(lag):5g}%   not determinable: both this lag and B alone "
                  "hit the ceiling")
                continue
            mark = "\u2264" if up else ge
            A(f"    dT {cfg.lag_pct(lag):5g}%   {mark}{ratio:5.2f}x better than B alone"
              + ("   (tracks at the ceiling: at most this)" if up else "")
              + ("   <-- NO BETTER: A was not being used here"
                 if ratio < 1.2 and (up or not w["is_lower_bound"]) else ""))
    sc = strategy_check(rows, cfg)
    if any(v.get("ok") for v in sc.values()):
        A("  answers against each tone's own timing (logistic, per SD of that evidence):")
    for label, v in sc.items():
        if v.get("ok"):
            A(f"    {label:<22} " + "   ".join(
                f"{c}: {w['coef_per_sd']:+.2f} (z={w['z']:+.2f}, p={w['p_two_sided']:.3f})"
                for c, w in sorted(v["channels"].items())) + f"   n={v['n']}")
    if any(v.get("ok") for v in sc.values()):
        A("    Near zero: the answers did not follow either tone's own timing. A clearly positive")
        A("    coefficient in the token conditions means a listener judging one tone against its")
        A("    neighbours, the route the design exists to close.")
    A("")
    A("-- shape " + "-" * 68)
    for direction in directions(res):
        t = trend_test(res, direction=direction)
        if t["ok"]:
            A(f"  {(direction + ': ') if multi else ''}threshold rises with the lag: "
              f"rho = {t['rho']:+.3f} over {t['n_tracks']} tracks at {t['n_levels']} lags, "
              f"permutation p = {t['p_one_sided']:.4f} (one sided)")
    A("    Consistent with temporal coherence, and also with a Weber account of the A-B gap.")
    if cfg.n_context:
        _context_section(A, rows, cfg, res)
    io = in_vs_out(res) if not cfg.n_context else {"ok": False}
    if io["ok"]:
        A("")
        A("-- in against out " + "-" * 59)
        A(f"  {'':<9} {'in (' + word['in'] + ')':>12} {'out (' + word['out'] + ')':>12} "
          f"{'out/in':>8} {'out gap':>8}")
        for row in io["rows"]:
            lvl = "B alone" if row["level"] == "b_alone" else f"dT {cfg.lag_pct(row['level']):g}%"
            ms = lambda v, m: f"{m}{v:.1f} ms" if v else "-"
            ratio = f"{row['ratio']:.2f}" if row["ratio"] else "-"
            gap = "-" if row["level"] == "b_alone" else f"{100 * row['out_gap_fraction']:.0f}%"
            A(f"  {lvl:<9} {ms(row['in_ms'], row['in_mark']):>12} "
              f"{ms(row['out_ms'], row['out_mark']):>12} {ratio:>8}{'*' if row['bounded'] else ' '} "
              f"{gap:>7}")
        if any(r["bounded"] for r in io["rows"]):
            A("  * includes ceiling-limited tracks pooled as lower bounds")
        A("  Descriptive. At large lags a later B also opens a gap between A and B, a cue the")
        A("  other direction never offers, and listeners are known to time a high tone after a")
        A("  low one better than the reverse.")
    pc = position_check(rows)
    if pc.get("ok"):
        A(f"  target in the last pair vs the one before: {100 * pc['last_minus_second_last']:+.1f} "
          f"points correct (p={pc['p_two_sided']:.3f}; n={pc['n_last']} / {pc['n_second_last']})")
    A("")
    A("-- what this does not establish " + "-" * 45)
    A("  * one listener. Nothing here generalises.")
    if multi:
        A("  * in and out ran in separate blocks; a difference between them carries any change")
        A("    over the session unless the blocks were balanced (ABBA needs two rounds).")
    A("  * a rise with the lag does not by itself separate coherence from interval timing;")
    A("    the probes (or B alone) establish that the relation was used, not how.")
    A("=" * 78)
    return "\n".join(L)
