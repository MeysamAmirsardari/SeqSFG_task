"""Analysis: proportion correct with honest intervals, d', a psychometric fit that refuses
when it should, corrected tests, a single-trial test, session diagnostics, and figures."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, stats

from .config import Config
from .session import read_json, read_trials, write_json


# ---- basic statistics --------------------------------------------------------
def wilson(k: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo, hi = max(0.0, c - h), min(1.0, c + h)
    return (0.0 if k <= 0 else lo, 1.0 if k >= n else hi)


def dprime_2ifc(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float, float]:
    """d' = sqrt(2) z(pc); pc clipped to [1/(2n), 1-1/(2n)] so 0 and 1 stay finite. CI from Wilson."""
    def dp(p):
        p = min(max(p, 0.5 / n), 1 - 0.5 / n)
        return math.sqrt(2.0) * stats.norm.ppf(p)
    lo, hi = wilson(k, n, alpha)
    return dp(k / n), dp(lo), dp(hi)


def holm(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def fisher_2x2(k1: int, n1: int, k2: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return 1.0
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


# ---- cue profile measured by the verification battery ----
def cue_profile_from_battery(path: Path, alpha: float = 0.05) -> dict:
    """Per (variant, step), the largest blind ideal-observer d' that survives multiple-comparison
    correction over that ladder's whole observer grid.

    Read from a `seqsfg verify --json` run of the SAME configuration. A cell listed here can be
    done, in part, without binding anything, so the analysis marks it rather than letting the
    reader take it at face value. Cells whose raw interval excludes zero but which do not survive
    correction are NOT listed: on a grid of 30 cells a few of those are expected by chance.
    """
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        b = json.load(f)
    best: Dict[str, dict] = {}
    for lv, blk in (b.get("by_ladder") or {}).items():
        for c in ((blk.get("multiplicity") or {}).get("cells") or []):
            if c.get("p_holm", 1.0) >= alpha or c["dprime"] <= 0:
                continue
            key = c["condition"]
            if c["dprime"] > best.get(key, {"dprime": 0.0})["dprime"]:
                best[key] = {"dprime": float(c["dprime"]), "observer": c["observer"],
                             "p_holm": float(c["p_holm"])}
    return best


def default_battery_path() -> Path:
    return Path(__file__).resolve().parent.parent / "verification" / "battery.json"


# ---- summaries ---------------------------------------------------------------
def summarize(trials: List[dict], alpha: float = 0.05) -> List[dict]:
    cells: Dict[Tuple[str, float], List[int]] = {}
    for t in trials:
        if t["correct"] is None:
            continue
        cells.setdefault((t["variant"], t["step_ms"]), []).append(t["correct"])
    out = []
    for (v, s), c in sorted(cells.items()):
        n, k = len(c), int(sum(c))
        lo, hi = wilson(k, n, alpha)
        dp, dlo, dhi = dprime_2ifc(k, n, alpha)
        out.append(dict(variant=v, step_ms=s, n=n, k=k, pc=k / n, pc_ci=(lo, hi), dprime=dp, dprime_ci=(dlo, dhi),
                        p_chance=float(stats.binomtest(k, n, 0.5).pvalue)))
    return out


# ---- psychometric function ---------------------------------------------------
def pf(s, a, b, lam):
    return 0.5 + (0.5 - lam) / (1.0 + np.exp((np.asarray(s, float) - a) / b))


def threshold_of(a, b, lam, thr_pc):
    arg = (0.5 - lam) / (thr_pc - 0.5) - 1.0
    return float(a + b * math.log(arg)) if arg > 0 else float("nan")


def fit_psychometric(steps, ks, ns, thr_pc: float, x0=None) -> dict:
    steps, ks, ns = np.asarray(steps, float), np.asarray(ks, float), np.asarray(ns, float)
    rng_ = max(steps.max() - steps.min(), 1.0)

    def nll(theta):
        p = np.clip(pf(steps, *theta), 1e-6, 1 - 1e-6)
        return -np.sum(ks * np.log(p) + (ns - ks) * np.log(1 - p))

    bounds = [(-rng_, steps.max() + 2 * rng_), (0.5, 4 * rng_ + 1), (0.0, 0.2)]
    starts = [list(x0)] if x0 is not None else [[a0, b0, 0.02] for a0 in np.linspace(steps.min(), steps.max(), 5)
                                                 for b0 in (rng_ / 10, rng_ / 3, rng_)]
    best = None
    for s0 in starts:
        r = optimize.minimize(nll, x0=s0, bounds=bounds, method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    a, b, lam = best.x
    return dict(alpha=float(a), beta=float(b), lapse=float(lam), nll=float(best.fun), converged=bool(best.success),
                threshold=threshold_of(a, b, lam, thr_pc))


def bootstrap_threshold(steps, ks, ns, thr_pc, n_boot, rng, x0=None) -> np.ndarray:
    """Non-parametric: resample trials within each condition, refit (warm-started), take the threshold."""
    out = np.full(n_boot, np.nan)
    ks = np.asarray(ks); ns = np.asarray(ns)
    for i in range(n_boot):
        kb = rng.binomial(ns, np.where(ns > 0, ks / np.maximum(ns, 1), 0.5))
        out[i] = fit_psychometric(steps, kb, ns, thr_pc, x0=x0)["threshold"]
    return out


def psychometric_report(summary_main: List[dict], cfg: Config, rng: np.random.Generator,
                        variant: str = "rising") -> dict:
    rows = [r for r in summary_main if r["variant"] == variant]
    rows.sort(key=lambda r: r["step_ms"])
    if len(rows) < 3:
        return dict(variant=variant, reportable=False, gates={"at least three steps measured": False},
                    fit={}, threshold=float("nan"), threshold_ci=(float("nan"),) * 2,
                    tested_range=(float("nan"),) * 2, steps=[], ks=[], ns=[],
                    bootstrap_defined_fraction=0.0)
    steps = [r["step_ms"] for r in rows]; ks = [r["k"] for r in rows]; ns = [r["n"] for r in rows]
    fit = fit_psychometric(steps, ks, ns, cfg.threshold_pc)
    boots = bootstrap_threshold(steps, ks, ns, cfg.threshold_pc, cfg.bootstrap_n, rng,
                                x0=[fit["alpha"], fit["beta"], fit["lapse"]])
    defined = boots[np.isfinite(boots)]
    ci = (float(np.percentile(defined, 2.5)), float(np.percentile(defined, 97.5))) if defined.size else (float("nan"),) * 2
    easiest, others = rows[0], rows[1:]
    tested = (min(steps), max(steps))
    thr = fit["threshold"]
    spacing = float(np.median(np.diff(sorted(steps)))) if len(steps) > 1 else float("inf")
    gates = {
        "transition width resolvable by the step spacing": bool(fit["beta"] >= 0.25 * spacing),
        "easiest condition above chance": easiest["p_chance"] < cfg.alpha,
        "performance peaks at the easiest condition": all(easiest["pc"] >= o["pc"] for o in others),
        "threshold defined and inside the tested range": bool(np.isfinite(thr) and tested[0] <= thr <= tested[1]),
        "bootstrap CI narrower than the tested range": bool(np.isfinite(ci[0]) and (ci[1] - ci[0]) <= (tested[1] - tested[0])),
        "bootstrap threshold defined in >= 75% of resamples": defined.size >= 0.75 * cfg.bootstrap_n,
        "fit converged": fit["converged"],
    }
    return dict(variant=variant, fit=fit, threshold=thr, threshold_ci=ci,
                bootstrap_defined_fraction=defined.size / max(cfg.bootstrap_n, 1),
                tested_range=tested, gates=gates, reportable=all(gates.values()), steps=steps, ks=ks, ns=ns)


# ---- tests ---------------------------------------------------------------------
def condition_tests(summary_main: List[dict], alpha: float, variant: str = "rising") -> dict:
    rows = sorted([r for r in summary_main if r["variant"] == variant], key=lambda r: r["step_ms"])
    if not rows:
        return dict(variant=variant, steps=[], p_chance=[], p_chance_holm=[], p_vs_easiest=[],
                    p_vs_easiest_holm=[], alpha=alpha)
    p_chance = [r["p_chance"] for r in rows]
    easiest = rows[0]
    p_vs_easiest = [fisher_2x2(r["k"], r["n"], easiest["k"], easiest["n"]) for r in rows[1:]]
    return dict(variant=variant, steps=[r["step_ms"] for r in rows], p_chance=p_chance,
                p_chance_holm=holm(p_chance), p_vs_easiest=p_vs_easiest,
                p_vs_easiest_holm=holm(p_vs_easiest) if p_vs_easiest else [], alpha=alpha)


def _logistic_fit(X: np.ndarray, y: np.ndarray, iters: int = 50) -> Tuple[np.ndarray, float, np.ndarray]:
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p)
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w = w + step
        if np.max(np.abs(step)) < 1e-8:
            break
    p = np.clip(1 / (1 + np.exp(-X @ w)), 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    cov = np.linalg.inv((X * (p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(X.shape[1]))
    return w, ll, cov


def single_trial_test(trials_main: List[dict], variant: Optional[str] = None) -> dict:
    """Does the delay matter at all? Logistic regression of correct on step, every trial, LRT."""
    t = [x for x in trials_main if x["correct"] is not None and (variant is None or x["variant"] == variant)]
    y = np.array([x["correct"] for x in t], float)
    s = np.array([x["step_ms"] for x in t], float)
    if len(t) < 10 or np.ptp(s) == 0:
        return dict(variant=variant, n=len(t), p=float("nan"), slope_per_ms=float("nan"), slope_se=float("nan"))
    X1 = np.column_stack([np.ones_like(s), s])
    w1, ll1, cov1 = _logistic_fit(X1, y)
    w0, ll0, _ = _logistic_fit(np.ones((len(s), 1)), y)
    lrt = 2 * (ll1 - ll0)
    return dict(variant=variant, n=len(t), lrt=float(lrt), p=float(stats.chi2.sf(lrt, 1)),
                slope_per_ms=float(w1[1]), slope_se=float(math.sqrt(cov1[1, 1])))


def ladder_interaction_test(trials_main: List[dict], variants: Sequence[str]) -> dict:
    """Single-trial description of the two ladders: is there an overall advantage, and do the
    logit-linear slopes differ?

    Nested logistic models on every main trial, likelihood-ratio tested:
        step  -> + ladder (does one foil give an overall advantage?)
              -> + step x ladder (do the two curves differ in LINEAR logit slope?)

    SECONDARY AND LOW-POWERED. Simulated at this design's trial counts, the interaction term
    rejects at only 0.03 when the two curves differ in shape but share a mean slope, and at 0.42
    when they differ in slope outright. `curve_comparison_lrt` is the primary comparison; a null
    interaction here is not evidence that the curves agree.
    """
    if len(variants) < 2:
        return {}
    v0, v1 = variants[0], variants[1]
    t = [x for x in trials_main if x["correct"] is not None and x["variant"] in (v0, v1)]
    if len(t) < 20:
        return {}
    y = np.array([x["correct"] for x in t], float)
    s = np.array([x["step_ms"] for x in t], float)
    g = np.array([1.0 if x["variant"] == v1 else 0.0 for x in t])
    if np.ptp(s) == 0 or np.ptp(g) == 0:
        return {}
    sc = (s - s.mean()) / max(s.std(), 1e-9)
    X_s = np.column_stack([np.ones_like(sc), sc])
    X_sg = np.column_stack([np.ones_like(sc), sc, g])
    X_full = np.column_stack([np.ones_like(sc), sc, g, sc * g])
    _, ll_s, _ = _logistic_fit(X_s, y)
    _, ll_sg, _ = _logistic_fit(X_sg, y)
    w, ll_f, cov = _logistic_fit(X_full, y)
    lrt_g, lrt_i = 2 * (ll_sg - ll_s), 2 * (ll_f - ll_sg)
    scale = max(s.std(), 1e-9)
    return dict(reference=v0, other=v1, n=len(t),
                p_ladder=float(stats.chi2.sf(lrt_g, 1)), lrt_ladder=float(lrt_g),
                p_interaction=float(stats.chi2.sf(lrt_i, 1)), lrt_interaction=float(lrt_i),
                slope_reference_per_ms=float(w[1] / scale),
                slope_difference_per_ms=float(w[3] / scale),
                slope_difference_se=float(math.sqrt(cov[3, 3]) / scale))


def curve_comparison_lrt(summary_main: List[dict], cfg: Config, variants: Sequence[str]) -> dict:
    """Do the two ladders need different psychometric functions at all?

    Likelihood-ratio test of one shared curve against one curve per ladder (3 extra parameters).
    This is the PRIMARY comparison. Unlike the linear step x ladder interaction it is sensitive to
    differences in SHAPE: two curves can share a mean logit slope over the tested range and still
    have very different thresholds, which is exactly the case this experiment is looking for.

    Calibration, simulated at this design's trial counts (200 replicates): under identical ladders
    it rejects at 0.05 (14 trials per cell) and 0.035 (40 per cell); against a real difference in
    shape it rejects at 0.87 and 1.00 respectively.
    """
    if len(variants) < 2:
        return {}
    per = {}
    for v in variants[:2]:
        rows = sorted([r for r in summary_main if r["variant"] == v], key=lambda r: r["step_ms"])
        if len(rows) < 3:
            return {}
        per[v] = ([r["step_ms"] for r in rows], [r["k"] for r in rows], [r["n"] for r in rows])
    v0, v1 = variants[0], variants[1]
    ll_sep = -sum(fit_psychometric(*per[v], cfg.threshold_pc)["nll"] for v in (v0, v1))
    steps = per[v0][0] + per[v1][0]
    ks = per[v0][1] + per[v1][1]
    ns = per[v0][2] + per[v1][2]
    ll_shared = -fit_psychometric(steps, ks, ns, cfg.threshold_pc)["nll"]
    lrt = 2.0 * (ll_sep - ll_shared)
    return dict(reference=v0, other=v1, lrt=float(lrt), df=3,
                p=float(stats.chi2.sf(max(lrt, 0.0), 3)))


def threshold_difference(summary_main: List[dict], cfg: Config, variants: Sequence[str],
                         rng: np.random.Generator) -> dict:
    """Difference in threshold between two ladders, with a bootstrap CI.

    Refuses when either curve's own gates fail: a difference between two numbers, one of which
    the fit declined to report, is not a result.
    """
    if len(variants) < 2:
        return {}
    reports = {v: psychometric_report(summary_main, cfg, rng, v) for v in variants[:2]}
    v0, v1 = variants[0], variants[1]
    if not (reports[v0]["reportable"] and reports[v1]["reportable"]):
        return dict(reference=v0, other=v1, reportable=False,
                    reason=f"{v0} fit reportable: {reports[v0]['reportable']}; "
                           f"{v1} fit reportable: {reports[v1]['reportable']}")
    obs = reports[v1]["threshold"] - reports[v0]["threshold"]
    diffs = np.full(cfg.bootstrap_n, np.nan)
    for i in range(cfg.bootstrap_n):
        th = {}
        for v in (v0, v1):
            r = reports[v]
            ns = np.asarray(r["ns"]); ks = np.asarray(r["ks"])
            kb = rng.binomial(ns, np.where(ns > 0, ks / np.maximum(ns, 1), 0.5))
            th[v] = fit_psychometric(r["steps"], kb, ns, cfg.threshold_pc,
                                     x0=[r["fit"]["alpha"], r["fit"]["beta"], r["fit"]["lapse"]])["threshold"]
        diffs[i] = th[v1] - th[v0]
    ok = diffs[np.isfinite(diffs)]
    if ok.size < 0.75 * cfg.bootstrap_n:
        return dict(reference=v0, other=v1, reportable=False, difference=float(obs),
                    reason=f"bootstrap defined in only {ok.size}/{cfg.bootstrap_n} resamples")
    ci = (float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5)))
    return dict(reference=v0, other=v1, reportable=True, difference=float(obs), ci=ci,
                threshold_reference=reports[v0]["threshold"], threshold_other=reports[v1]["threshold"],
                excludes_zero=bool(ci[0] > 0 or ci[1] < 0))


# ---- diagnostics -----------------------------------------------------------------
def diagnostics(trials_main: List[dict], easiest_step: float, variants: Sequence[str] = ("rising",)) -> dict:
    t = [x for x in trials_main if x["correct"] is not None]
    if not t:
        return {}
    resp = np.array([x["response"] for x in t]); corr = np.array([x["correct"] for x in t])
    tgt = np.array([x["target_position"] for x in t]); step = np.array([x["step_ms"] for x in t])
    n = len(t)
    out = {}
    n1 = int(np.sum(resp == 1))
    out["interval_preference"] = dict(n_resp_1=n1, n_resp_2=n - n1, p=float(stats.binomtest(n1, n, 0.5).pvalue))
    k1, m1 = int(corr[tgt == 1].sum()), int((tgt == 1).sum()); k2, m2 = int(corr[tgt == 2].sum()), int((tgt == 2).sum())
    out["by_target_position"] = dict(pc_target_1=k1 / max(m1, 1), pc_target_2=k2 / max(m2, 1), p=fisher_2x2(k1, m1, k2, m2))
    h = n // 2
    out["halves"] = dict(pc_first=float(corr[:h].mean()), pc_second=float(corr[h:].mean()),
                         p=fisher_2x2(int(corr[:h].sum()), h, int(corr[h:].sum()), n - h))
    prev = corr[:-1]; nxt = corr[1:]
    ka, na = int(nxt[prev == 1].sum()), int((prev == 1).sum()); ke, ne = int(nxt[prev == 0].sum()), int((prev == 0).sum())
    out["after_correct_vs_error"] = dict(pc_after_correct=ka / max(na, 1), pc_after_error=ke / max(ne, 1), n_after_error=ne,
                                         p=fisher_2x2(ka, na, ke, ne))
    var = np.array([x["variant"] for x in t])
    out["easiest_by_ladder"] = {}
    for v in variants:
        ev = corr[(step == easiest_step) & (var == v)]
        if ev.size >= 4:
            h2 = ev.size // 2
            out["easiest_by_ladder"][v] = dict(n=int(ev.size), pc=float(ev.mean()),
                                               pc_first=float(ev[:h2].mean()), pc_second=float(ev[h2:].mean()),
                                               p_chance=float(stats.binomtest(int(ev.sum()), ev.size, 0.5).pvalue))
    e = corr[step == easiest_step]
    if e.size >= 4:
        he = e.size // 2
        order = np.arange(e.size, dtype=float)
        X = np.column_stack([np.ones_like(order), (order - order.mean()) / max(order.std(), 1)])
        _, ll1, _ = _logistic_fit(X, e.astype(float)); _, ll0, _ = _logistic_fit(X[:, :1], e.astype(float))
        out["easiest_over_time"] = dict(n=int(e.size), pc_first=float(e[:he].mean()), pc_second=float(e[he:].mean()),
                                        p_halves=fisher_2x2(int(e[:he].sum()), he, int(e[he:].sum()), e.size - he),
                                        p_trend=float(stats.chi2.sf(2 * (ll1 - ll0), 1)),
                                        second_half_above_chance_p=float(stats.binomtest(int(e[he:].sum()), e.size - he, 0.5).pvalue))
    return out


# ---- whole session -------------------------------------------------------------
def analyze_sessions(session_dirs: Sequence[Path], out_dir: Optional[Path] = None,
                     make_figures: bool = True, battery_json: Optional[Path] = None) -> dict:
    session_dirs = [Path(p) for p in session_dirs]
    metas = [read_json(p / "session.json") for p in session_dirs]
    hashes = {m["config_hash"] for m in metas}
    if len(hashes) > 1:
        raise RuntimeError(f"refusing to pool sessions with different configs: {sorted(hashes)}")
    cfg = Config.from_dict(metas[0]["config"])
    trials = []
    for p, m in zip(session_dirs, metas):
        for t in read_trials(p / "trials.csv"):
            t["session"] = str(p); trials.append(t)
    main = [t for t in trials if t["block"] == "main"]
    ctrl = [t for t in trials if t["block"] == "control"]
    prac = [t for t in trials if t["block"] == "practice"]
    rng = np.random.default_rng(0)
    variants = [v for v in cfg.main_variants]
    battery = Path(battery_json) if battery_json else default_battery_path()
    cue = {}
    if battery.exists():
        try:
            with open(battery) as f:
                same = json.load(f).get("config_hash") == cfg.hash()
            cue = cue_profile_from_battery(battery) if same else {}
            cue_source = str(battery) if same else f"{battery} (different config, ignored)"
        except Exception as exc:
            cue_source = f"{battery} (unreadable: {exc})"
    else:
        cue_source = f"{battery} (not found; run 'seqsfg verify --json' to measure the cue profile)"
    res = {"sessions": [str(p) for p in session_dirs], "config_hash": cfg.hash(), "n_trials": len(trials),
           "status": [m.get("status") for m in metas], "main_variants": variants,
           "threshold_pc": cfg.threshold_pc, "cue_profile": cue, "cue_source": cue_source,
           "practice": summarize(prac, cfg.alpha), "main": summarize(main, cfg.alpha),
           "control": summarize(ctrl, cfg.alpha)}
    if res["main"]:
        res["curves"] = {}
        for v in variants:
            if not any(r["variant"] == v for r in res["main"]):
                continue
            res["curves"][v] = {
                "psychometric": psychometric_report(res["main"], cfg, rng, v),
                "tests": condition_tests(res["main"], cfg.alpha, v),
                "single_trial": single_trial_test(main, v),
            }
        measured = [v for v in variants if v in res["curves"]]
        res["comparison"] = {
            "interaction": ladder_interaction_test(main, measured),
            "curves_differ": curve_comparison_lrt(res["main"], cfg, measured),
            "threshold_difference": threshold_difference(res["main"], cfg, measured, rng),
        }
        res["single_trial"] = single_trial_test(main)
        res["diagnostics"] = diagnostics(main, float(cfg.steps_ms[0]), measured)
    out_dir = Path(out_dir) if out_dir else session_dirs[0] / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "results.json", res)
    if make_figures and res["main"]:
        from . import figures
        figures.psychometric(res, cfg, out_dir / "psychometric.png")
        figures.timecourse(main, cfg, out_dir / "timecourse.png")
        if res["control"]:
            figures.controls(res, cfg, out_dir / "controls.png")
    res["out_dir"] = str(out_dir)
    return res


def summary_text(res: dict) -> str:
    L = [f"sessions: {', '.join(res['sessions'])}   status: {res['status']}   trials: {res['n_trials']}"]
    for block in ("practice", "main", "control"):
        if not res.get(block):
            continue
        L.append(f"\n{block}:")
        L.append("   variant     step   n    k    pc   [95% CI]        d'   [95% CI]      p(chance)")
        cue = res.get("cue_profile") or {}
        for r in res[block]:
            key = f"{r['variant']}:{r['step_ms']:g}"
            mark = ""
            if key in cue:
                mark = f"   <-- non-binding route available here: {cue[key]['observer']} d'={cue[key]['dprime']:+.2f}"
            L.append(f"   {r['variant']:10s} {r['step_ms']:5.0f}  {r['n']:3d}  {r['k']:3d}  {r['pc']:.2f} "
                     f"[{r['pc_ci'][0]:.2f},{r['pc_ci'][1]:.2f}]  {r['dprime']:+.2f} "
                     f"[{r['dprime_ci'][0]:+.2f},{r['dprime_ci'][1]:+.2f}]   {r['p_chance']:.3f}{mark}")
        if block == "main":
            L.append(f"   (cue profile from {res.get('cue_source', 'not loaded')})")
    thr_pc = res.get("threshold_pc", 0.75)
    for v, cur in (res.get("curves") or {}).items():
        ps = cur["psychometric"]
        L.append(f"\n=== ladder '{v}' ===")
        if ps.get("fit"):
            L.append(f"   fit: alpha={ps['fit']['alpha']:.1f} ms  beta={ps['fit']['beta']:.1f} ms  "
                     f"lapse={ps['fit']['lapse']:.3f}")
        for g, ok in ps["gates"].items():
            L.append(f"   [{'ok' if ok else 'FAIL'}] {g}")
        if ps["reportable"]:
            L.append(f"   threshold (pc={thr_pc:.2f}): {ps['threshold']:.1f} ms, 95% bootstrap CI "
                     f"[{ps['threshold_ci'][0]:.1f}, {ps['threshold_ci'][1]:.1f}] ms")
        else:
            L.append(f"   threshold NOT REPORTED (fit not trustworthy); treat any number from it as noise.")
        t = cur["tests"]
        L.append("   tests (Holm-corrected within this ladder):")
        for i, s in enumerate(t["steps"]):
            vs = f"  vs easiest p={t['p_vs_easiest_holm'][i - 1]:.3f}" if i > 0 else ""
            L.append(f"     step {s:5.0f}: vs chance p={t['p_chance_holm'][i]:.3f}{vs}")
        st = cur["single_trial"]
        L.append(f"   single-trial slope: {st['slope_per_ms']:+.4f}/ms (se {st['slope_se']:.4f}), "
                 f"LRT p={st['p']:.4f}, n={st['n']}")
    cmp_ = res.get("comparison") or {}
    inter = cmp_.get("interaction") or {}
    cd = cmp_.get("curves_differ") or {}
    if inter or cd:
        ref = (inter or cd).get("reference"); oth = (inter or cd).get("other")
        L.append(f"\n=== comparing the ladders: '{oth}' against '{ref}' ===")
    if cd:
        L.append(f"   PRIMARY  do the two ladders need different psychometric functions?")
        L.append(f"            LRT p={cd['p']:.4f} (chi2={cd['lrt']:.1f}, {cd['df']} df)")
    if inter:
        L.append(f"   overall advantage of one foil: LRT p={inter['p_ladder']:.4f}")
        L.append(f"   secondary, LOW POWER: step x ladder interaction on the linear logit scale, "
                 f"p={inter['p_interaction']:.4f}")
        L.append(f"            (slope difference {inter['slope_difference_per_ms']:+.4f}/ms, se "
                 f"{inter['slope_difference_se']:.4f}; reference slope "
                 f"{inter['slope_reference_per_ms']:+.4f}/ms over {inter['n']} trials)")
        if cd and cd["p"] < 0.05 <= inter["p_interaction"]:
            L.append(f"            a null here alongside a significant LRT above is expected: this term "
                     f"rejects at only 0.03 when curves differ in shape rather than slope.")
    td = cmp_.get("threshold_difference") or {}
    if td:
        if td.get("reportable"):
            L.append(f"   threshold difference ({td['other']} - {td['reference']}): {td['difference']:+.1f} ms, "
                     f"95% CI [{td['ci'][0]:+.1f}, {td['ci'][1]:+.1f}] "
                     f"({'excludes' if td['excludes_zero'] else 'includes'} zero)")
        else:
            L.append(f"   threshold difference NOT REPORTED: {td.get('reason', '')}")
    dg = res.get("diagnostics") or {}
    if dg:
        L.append("\ndiagnostics:")
        ip = dg["interval_preference"]
        L.append(f"   responses '1' vs '2': {ip['n_resp_1']} vs {ip['n_resp_2']} (p={ip['p']:.3f})")
        bt = dg["by_target_position"]
        L.append(f"   pc when target is interval 1 vs 2: {bt['pc_target_1']:.2f} vs {bt['pc_target_2']:.2f} "
                 f"(p={bt['p']:.3f})")
        hv = dg["halves"]
        L.append(f"   first half vs second half: {hv['pc_first']:.2f} vs {hv['pc_second']:.2f} (p={hv['p']:.3f})")
        ae = dg["after_correct_vs_error"]
        L.append(f"   after a correct vs after an error: {ae['pc_after_correct']:.2f} vs "
                 f"{ae['pc_after_error']:.2f} (n after error {ae['n_after_error']}, p={ae['p']:.3f})")
        for v, e in (dg.get("easiest_by_ladder") or {}).items():
            L.append(f"   easiest step of '{v}': pc {e['pc']:.2f} (n={e['n']}), first vs second half "
                     f"{e['pc_first']:.2f} vs {e['pc_second']:.2f}, vs chance p={e['p_chance']:.3f}")
    L.append(f"\nwritten to {res.get('out_dir')}")
    return "\n".join(L)
