"""From trials to thresholds to a verdict, with the limits of each step stated rather than hidden.

The measured quantity
----------------------
Each track yields one threshold: the delta at which the listener is 79.4% correct. The
experiment's quantity of interest is not a threshold but a NORMALISED one,

    kappa(dT) = [log theta(dT) - log theta(0%)] / [log theta(B only) - log theta(0%)]

which is 0 when the listener does as well as when the two tones are exactly synchronous, and 1
when they do no better than with the low tone switched off entirely. That is the behavioural
counterpart of the model's lambda2/lambda1, which is also 0 for one stream and 1 for two, and
the two can be plotted on the same axis without any free parameter beyond the two endpoints
that define the scale -- both of which are measured in the same session, by the same listener,
with the same procedure.

kappa is a ratio, so it needs the denominator to exist. `coherence_index` refuses to report one
when the bootstrap interval on log theta(B only) - log theta(0%) includes zero: with no
dynamic range there is nothing to normalise and any kappa would be noise divided by noise.

What each test can and cannot separate
---------------------------------------
This is the part that decides whether the experiment is worth running, so it is written down
here and repeated in the report.

  monotone trend across dT
      Establishes that grouping changes with onset asynchrony. Does NOT distinguish the
      temporal-coherence account from the alternatives, all of which also predict a rise.

  observed kappa against the model curve, versus against a straight line
      Nearly worthless, and reported as such. The model's predicted curve correlates with a
      straight line in dT at better than r = 0.99 over the levels this design uses
      (`predictor_collinearity` prints the number). No realistic amount of data separates them.
      A study that claimed to have confirmed the model's SHAPE from five points would be
      overreaching, and this one does not.

  coherent versus scrambled, at matched dT
      The load-bearing test. The scrambled control holds the final A-B interval, the tone count,
      the levels and the frequencies, and destroys only the temporal coherence of the A
      sequence. An interval-discrimination account and a local acoustic-overlap account both
      predict NO difference at any dT, because both depend only on the final pair, which is
      identical. The coherence account predicts a large difference at dT = 0 that shrinks to
      nothing at dT = 100%. This is an interaction, and interactions are what the design is for.

  five precursors versus thirteen
      Separates the coherence account from any account local to the final pair. Streaming builds
      up over seconds; a local cue does not care what came before. Only runs if the config asks
      for it.

  the endpoints against Elhilali et al. (2009)
      A replication check, not a test of anything new: dT = 0% should land near her 2-4 ms and
      B-only near her 10-20 ms. If they do not, the disagreement is with a published result and
      has to be explained before the sweep between them means anything.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from .config import Condition, Config, validate
from .psychometric import Fit, fit as pf_fit, target_proportion
from .session import read_json, read_trials

ELHILALI_SYNC_MS = (2.0, 4.0)      # her Figure 2, filled squares, across three frequency separations
ELHILALI_CEILING_MS = (10.0, 20.0)  # her crosses and her 30/70 ms controls


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
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
    v = r.get(k, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(r, k, default=-1):
    try:
        return int(float(r.get(k, "")))
    except (TypeError, ValueError):
        return default


# ----------------------------------------------------------------------------
# thresholds
# ----------------------------------------------------------------------------
@dataclass
class CondResult:
    name: str
    lag_pct: float
    a_kind: str
    n_precursor: int
    track_thresholds: List[float]          # usable only; censored ones are held separately
    n_tracks_attempted: int
    geomean_ms: Optional[float]
    fit: Optional[Fit]
    n_trials: int
    prop_correct: float
    ceiling_trials: int
    floor_trials: int
    censored_thresholds: List[float] = None    # tracks whose averaged reversals hit a clamp
    censor_side: str = ""                      # "ceiling" | "floor" | ""

    @property
    def censored(self) -> bool:
        return bool(self.censored_thresholds)

    @property
    def bound_ms(self) -> Optional[float]:
        """The lower (or upper) bound a censored condition supports, if any."""
        if not self.censored_thresholds:
            return None
        return float(np.exp(np.mean(np.log(self.censored_thresholds))))

    @property
    def log_thr(self) -> Optional[float]:
        return None if not self.geomean_ms else math.log(self.geomean_ms)


def replay_tracks(rows: Sequence[dict], cfg: Config) -> Dict[int, dict]:
    """Recompute every threshold from trials.csv alone, without trusting session.json.

    The runner writes its own summary; this recomputes the same numbers from the per-trial log
    by re-running the staircase rule over the logged responses. If the two disagree, one of
    them is wrong and the report says so instead of picking a favourite.
    """
    from .track import Track
    by_track: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        if r.get("phase") != "main":
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
            [_f(r, "delta_ms") for r in rs][: len(t.trials)], rtol=1e-6, atol=1e-9))
        out[tid] = a
    return out


def thresholds(rows: Sequence[dict], cfg: Config, use: str = "replay") -> Dict[str, CondResult]:
    """Per-condition thresholds, by both routes: staircase reversals and a fit to every trial."""
    d = validate(cfg)
    conds = {c.name: c for c in d.conditions}
    replay = replay_tracks(rows, cfg)

    # A "reversal" recorded at the delta clamp is not a reversal: the rule called for a step the
    # track could not take, so the value is the clamp rather than a measurement. Averaging those
    # in produces a number that looks like a threshold and is really a statement about
    # delta_max_ms. The pre-registration says such tracks are reported separately and kept out of
    # kappa; this is where that happens.
    by_cond_tracks: Dict[str, List[float]] = defaultdict(list)
    censored: Dict[str, List[float]] = defaultdict(list)
    sides: Dict[str, str] = {}
    attempted: Dict[str, int] = defaultdict(int)
    for tid, a in replay.items():
        attempted[a["condition"]] += 1
        if a["threshold_ms"] is None:
            continue
        used = a.get("reversals_used_ms") or []
        hi = sum(1 for v in used if v >= cfg.delta_max_ms - 1e-9)
        lo = sum(1 for v in used if v <= cfg.delta_min_ms + 1e-9)
        if hi or lo:
            censored[a["condition"]].append(a["threshold_ms"])
            sides[a["condition"]] = "ceiling" if hi >= lo else "floor"
        else:
            by_cond_tracks[a["condition"]].append(a["threshold_ms"])

    by_cond_trials: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        if r.get("phase") in ("main", "catch"):
            by_cond_trials[r["condition"]].append(r)

    out = {}
    for name, c in conds.items():
        tr = by_cond_trials.get(name, [])
        deltas = [abs(_f(r, "delta_realised_ms")) for r in tr]
        correct = [bool(_i(r, "correct", 0)) for r in tr]
        ok = [i for i, x in enumerate(deltas) if np.isfinite(x) and x > 0]
        f = pf_fit([deltas[i] for i in ok], [correct[i] for i in ok],
                   lapse_max=cfg.lapse_max, n_boot=0) if len(ok) >= 8 else None
        thr = by_cond_tracks.get(name, [])
        gm = float(np.exp(np.mean(np.log(thr)))) if thr else None
        out[name] = CondResult(
            name, c.lag_pct, c.a_kind,
            cfg.n_precursor if c.n_precursor is None else c.n_precursor,
            sorted(thr), attempted.get(name, 0), gm, f, len(tr),
            float(np.mean(correct)) if correct else float("nan"),
            sum(_i(r, "at_ceiling", 0) == 1 for r in tr),
            sum(_i(r, "at_floor", 0) == 1 for r in tr),
            sorted(censored.get(name, [])), sides.get(name, ""))
    return out


# ----------------------------------------------------------------------------
# the index
# ----------------------------------------------------------------------------
def _boot_logmean(values: Sequence[float], rng: np.random.Generator, n: int) -> np.ndarray:
    v = np.log(np.asarray(values, dtype=float))
    if v.size == 0:
        return np.full(n, np.nan)
    idx = rng.integers(0, v.size, size=(n, v.size))
    return v[idx].mean(axis=1)


def coherence_index(res: Dict[str, CondResult], cfg: Config, n_boot: int = 4000,
                    seed: int = 11, a_kind: str = "coherent") -> dict:
    """kappa at each dT, with a bootstrap over tracks, and a refusal when the scale is degenerate.

    `a_kind` chooses which curve to express; the SCALE is always the same one -- the coherent
    dT = 0% condition and b_only. Normalising a control by its own dT = 0% threshold would put
    it on a different axis from the coherent curve while appearing to share one, so the two
    could not be plotted together or subtracted, which is the entire purpose of computing a
    control curve at all.
    """
    rng = np.random.default_rng(seed)
    floor = res.get(_name_for(res, 0.0, "coherent", cfg.n_precursor))
    ceil = res.get("b_only")
    if floor is None or ceil is None or not floor.track_thresholds or not ceil.track_thresholds:
        return {"ok": False, "reason": "need both a coherent dT=0 condition and b_only with "
                                       "converged tracks"}

    bf = _boot_logmean(floor.track_thresholds, rng, n_boot)
    bc = _boot_logmean(ceil.track_thresholds, rng, n_boot)
    span = bc - bf
    span_ci = (float(np.percentile(span, 2.5)), float(np.percentile(span, 97.5)))
    span = np.where(np.abs(span) < 1e-12, np.nan, span)   # a resample with no range says nothing
    if span_ci[0] <= 0:
        return {"ok": False, "reason": "the dynamic range is not distinguishable from zero "
                                       f"(log ceiling - log floor 95% CI {span_ci[0]:+.2f} to "
                                       f"{span_ci[1]:+.2f}); kappa would be noise over noise",
                "span_ci": span_ci,
                "span": float(np.mean(span))}

    levels, kap, ci, raw = [], [], [], []
    for name, r in sorted(res.items(), key=lambda kv: kv[1].lag_pct):
        if r.a_kind != a_kind or not r.track_thresholds:
            continue
        if r.n_precursor != cfg.n_precursor:
            continue
        b = _boot_logmean(r.track_thresholds, rng, n_boot)
        k = (b - bf) / span
        levels.append(r.lag_pct)
        kap.append(float(np.nanmean(k)))
        ci.append((float(np.nanpercentile(k, 2.5)), float(np.nanpercentile(k, 97.5))))
        raw.append(k)
    return {"ok": True, "a_kind": a_kind, "pcts": levels, "kappa": kap, "ci": ci,
            "boot": np.array(raw), "span": float(np.nanmean(span)), "span_ci": span_ci,
            "floor_ms": floor.geomean_ms, "ceiling_ms": ceil.geomean_ms,
            "scale": "coherent dT=0% to b_only"}


def _name_for(res: Dict[str, CondResult], pct: float, a_kind: str,
              n_precursor: Optional[int] = None) -> str:
    for n, r in res.items():
        if r.a_kind == a_kind and abs(r.lag_pct - pct) < 1e-9:
            if n_precursor is None or r.n_precursor == n_precursor:
                return n
    return ""


# ----------------------------------------------------------------------------
# the tests
# ----------------------------------------------------------------------------
def trend_test(res: Dict[str, CondResult], a_kind: str = "coherent",
               n_perm: int = 20000, seed: int = 13) -> dict:
    """Is log threshold ordered in dT? Permutation test on Spearman's rho over TRACKS.

    Tracks, not condition means, because tracks are what was randomised: the design assigns
    each track a condition and a position, and permuting condition labels across tracks is the
    null that the design's own randomisation creates.
    """
    xs, ys = [], []
    for r in res.values():
        if r.a_kind != a_kind:
            continue
        for t in r.track_thresholds:
            xs.append(r.lag_pct)
            ys.append(math.log(t))
    if len(set(xs)) < 3:
        return {"ok": False, "reason": "fewer than three dT levels with converged tracks"}
    x, y = np.array(xs), np.array(ys)
    rho = stats.spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    null = np.array([stats.spearmanr(rng.permutation(x), y).statistic for _ in range(n_perm)])
    p = float((np.sum(null >= rho) + 1) / (n_perm + 1))     # one-sided: the prediction is a RISE
    return {"ok": True, "rho": float(rho), "p_one_sided": p, "n_tracks": len(x),
            "n_levels": len(set(xs)), "null_sd": float(np.std(null))}


def control_kinds(res: Dict[str, CondResult]) -> List[str]:
    """Which matched controls this session actually contains, in reporting order."""
    have = {r.a_kind for r in res.values() if r.track_thresholds}
    return [k for k in ("scrambled", "pair_only", "nopartner") if k in have]


def interaction_test(res: Dict[str, CondResult], cfg: Config, n_boot: int = 4000,
                     seed: int = 17, n_perm: int = 20000,
                     control_kind: str = "scrambled") -> dict:
    """Coherent minus scrambled, at each dT they share, and whether that difference shrinks.

    The prediction that separates this experiment's hypothesis from its rivals: large at
    dT = 0, near zero at dT = 100%. A pedestal or local-overlap account predicts zero at every
    dT, so the test is on the SLOPE of the difference, not just on its average.

    The p-value comes from a PERMUTATION of the coherent/scrambled labels within each dT level,
    not from the bootstrap. That distinction is load-bearing: with three tracks per cell the
    percentile bootstrap cannot represent its own tails, and a first version of this test that
    used it fired at 10-23% under a simulated null instead of 5% (`tcoh.analysis.power`
    measures this, which is how it was caught). Permuting the labels is exactly the null the
    design randomises over -- the two cells hold the same number of tracks, run in the same
    session, by the same listener -- so the test is calibrated by construction. The bootstrap
    is kept, but only for the confidence intervals on the effect size.
    """
    rng = np.random.default_rng(seed)
    shared = []
    for r in res.values():
        # the build-up conditions carry the same dT as the sweep but a different precursor
        # count, and including them here would enter the same dT twice and inflate the level
        # count. They are H3's business, not H2's.
        if (r.a_kind != "coherent" or not r.track_thresholds
                or r.n_precursor != cfg.n_precursor):
            continue
        j = res.get(_name_for(res, r.lag_pct, control_kind))
        if j is not None and j.track_thresholds:
            shared.append((r.lag_pct, r, j))
    if len(shared) < 2:
        return {"ok": False, "control_kind": control_kind,
                "reason": f"need {control_kind} controls at two or more dT levels"}
    shared.sort(key=lambda t: t[0])

    pcts = np.array([p for p, _, _ in shared], dtype=float)
    diffs, ci, boots = [], [], []
    for _, r, j in shared:
        b = _boot_logmean(j.track_thresholds, rng, n_boot) - _boot_logmean(r.track_thresholds, rng, n_boot)
        diffs.append(float(np.mean(b)))
        ci.append((float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))))
        boots.append(b)
    B = np.array(boots)                              # (levels, n_boot), scrambled minus coherent
    xc = pcts - pcts.mean()
    denom = float((xc ** 2).sum())
    slope = (xc[:, None] * B).sum(0) / denom
    slope_ci = (float(np.percentile(slope, 2.5)), float(np.percentile(slope, 97.5)))

    # permutation null: within each dT level, which tracks were coherent and which scrambled
    obs_per_level = [(np.log(r.track_thresholds), np.log(j.track_thresholds))
                     for _, r, j in shared]
    obs_slope = float(sum(xc[i] * (obs_per_level[i][1].mean() - obs_per_level[i][0].mean())
                          for i in range(len(xc))) / denom)
    prng = np.random.default_rng(seed + 1)
    null = np.empty(n_perm)
    pooled = [np.concatenate(o) for o in obs_per_level]
    n_coh = [len(o[0]) for o in obs_per_level]
    for b in range(n_perm):
        acc = 0.0
        for i, pool in enumerate(pooled):
            q = prng.permutation(pool)
            acc += xc[i] * (q[n_coh[i]:].mean() - q[:n_coh[i]].mean())
        null[b] = acc / denom
    p_perm = float((np.sum(null <= obs_slope) + 1) / (n_perm + 1))   # one sided: predicted NEGATIVE

    return {"ok": True, "control_kind": control_kind, "n_levels": len(shared),
            "pcts": pcts.tolist(), "diff_log": diffs, "diff_ci": ci,
            "slope_per_pct": obs_slope, "slope_ci": slope_ci,
            "p_slope_negative": p_perm, "p_method": "permutation of condition labels within dT",
            "n_perm": n_perm, "boot_p_slope_negative": float(np.mean(slope >= 0)),
            "mean_diff_log": float(np.mean(B)),
            "p_mean_positive": float(np.mean(B.mean(0) <= 0))}


def buildup_test(res: Dict[str, CondResult], cfg: Config, n_boot: int = 4000, seed: int = 19) -> dict:
    """Does a longer precursor sharpen the dT effect? Local accounts say no."""
    rng = np.random.default_rng(seed)
    pairs = []
    for r in res.values():
        if r.a_kind != "coherent" or r.n_precursor != cfg.buildup_n_precursor:
            continue
        base = next((q for q in res.values() if q.a_kind == "coherent"
                     and abs(q.lag_pct - r.lag_pct) < 1e-9 and q.n_precursor == cfg.n_precursor), None)
        if base is not None and base.track_thresholds and r.track_thresholds:
            pairs.append((r.lag_pct, base, r))
    if not pairs:
        return {"ok": False, "reason": "no build-up conditions in this design"}

    # The ceiling for each precursor length. Without both, a build-up effect cannot be told
    # apart from "a longer B rhythm is easier to judge", which predicts the same direction.
    short_ceiling = res.get("b_only")
    long_ceiling = res.get("b_only_long")
    have_ceilings = (short_ceiling is not None and long_ceiling is not None
                     and short_ceiling.track_thresholds and long_ceiling.track_thresholds)
    ceil_gain = None
    if have_ceilings:
        cg = (_boot_logmean(long_ceiling.track_thresholds, rng, n_boot)
              - _boot_logmean(short_ceiling.track_thresholds, rng, n_boot))
        ceil_gain = {"diff_log": float(np.mean(cg)),
                     "ci": (float(np.percentile(cg, 2.5)), float(np.percentile(cg, 97.5)))}

    out = []
    for pct, base, long in sorted(pairs, key=lambda t: t[0]):
        b = (_boot_logmean(long.track_thresholds, rng, n_boot)
             - _boot_logmean(base.track_thresholds, rng, n_boot))
        row = {"pct": pct, "diff_log": float(np.mean(b)),
               "ci": (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))}
        if have_ceilings:
            inter = b - cg
            row["interaction_log"] = float(np.mean(inter))
            row["interaction_ci"] = (float(np.percentile(inter, 2.5)),
                                     float(np.percentile(inter, 97.5)))
            row["p_interaction_negative"] = float(np.mean(inter >= 0))
        out.append(row)
    return {"ok": True, "levels": out, "ceiling_gain": ceil_gain,
            "confounded": not have_ceilings,
            "n_precursor_short": cfg.n_precursor, "n_precursor_long": cfg.buildup_n_precursor}


def predictor_collinearity(cfg: Config) -> dict:
    """How different are the candidate predictors over the dT levels this design uses?

    Printed before any model comparison, because the honest answer is usually "barely", and a
    comparison between predictors that correlate at 0.99 over five points is not evidence.
    """
    d = validate(cfg)
    pcts = sorted({c.lag_pct for c in d.conditions if c.a_kind == "coherent"})
    if len(pcts) < 3:
        return {"ok": False, "reason": "fewer than three dT levels"}
    p = {"coherence": np.array([d.model_curve[x] for x in pcts]),
         "linear_in_dT": np.array(pcts) / 100.0,
         "overlap_lost": np.array(pcts) / 100.0}
    keys = ["coherence", "linear_in_dT"]
    r = float(np.corrcoef(p[keys[0]], p[keys[1]])[0, 1])
    # the biggest vertical gap after each is rescaled to span [0, 1]
    def norm(v):
        return (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else v * 0
    gap = float(np.max(np.abs(norm(p["coherence"]) - norm(p["linear_in_dT"]))))
    return {"ok": True, "pcts": pcts, "predictors": {k: v.tolist() for k, v in p.items()},
            "r_coherence_vs_linear": r, "max_normalised_gap": gap,
            "separable": bool(r < 0.98 and gap > 0.1)}


def compare_predictors(idx: dict, cfg: Config) -> dict:
    """Least-squares fit of observed kappa to each predictor, with the caveat attached.

    Reported for completeness, never as the headline. See `predictor_collinearity`.
    """
    if not idx.get("ok"):
        return {"ok": False, "reason": idx.get("reason", "no index")}
    d = validate(cfg)
    pcts = np.array(idx["pcts"], dtype=float)
    y = np.array(idx["kappa"], dtype=float)
    if y.size < 3:
        return {"ok": False, "reason": "fewer than three dT levels"}
    cands = {"coherence_model": np.array([d.model_curve.get(p, np.nan) for p in pcts]),
             "linear_in_dT": pcts / 100.0}
    out = {}
    for k, x in cands.items():
        if not np.all(np.isfinite(x)):
            continue
        A = np.stack([np.ones_like(x), x], axis=1)
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ beta
        sse = float(resid @ resid)
        n = y.size
        out[k] = {"intercept": float(beta[0]), "slope": float(beta[1]), "sse": sse,
                  "aicc": float(n * math.log(max(sse, 1e-12) / n) + 2 * 3 + (2 * 3 * 4) / max(n - 4, 1)),
                  "r2": float(1 - sse / max(float(((y - y.mean()) ** 2).sum()), 1e-12))}
    if not out:
        return {"ok": False, "reason": "no finite predictor"}
    best = min(out, key=lambda k: out[k]["aicc"])
    others = [v["aicc"] for k, v in out.items() if k != best]
    return {"ok": True, "fits": out, "best": best,
            "delta_aicc": float(min(others) - out[best]["aicc"]) if others else 0.0,
            "collinearity": predictor_collinearity(cfg)}


def endpoint_check(res: Dict[str, CondResult]) -> dict:
    """Do dT = 0% and B-only land where Elhilali et al. put them?"""
    out = {}
    sync = res.get(_name_for(res, 0.0, "coherent"))
    ceil = res.get("b_only")
    for label, r, ref in (("dT=0%", sync, ELHILALI_SYNC_MS), ("b_only", ceil, ELHILALI_CEILING_MS)):
        if r is None or r.geomean_ms is None:
            out[label] = {"ok": False}
            continue
        out[label] = {"ok": True, "ms": r.geomean_ms, "published_range_ms": ref,
                      "inside": bool(ref[0] <= r.geomean_ms <= ref[1]),
                      "factor_off": float(r.geomean_ms / ref[1] if r.geomean_ms > ref[1]
                                          else (ref[0] / r.geomean_ms if r.geomean_ms < ref[0] else 1.0))}
    return out


# ----------------------------------------------------------------------------
# diagnostics
# ----------------------------------------------------------------------------
def diagnostics(rows: Sequence[dict], metas: Sequence[dict], cfg: Config,
                res: Dict[str, CondResult]) -> dict:
    main = [r for r in rows if r.get("phase") == "main"]
    catch = [r for r in rows if r.get("phase") == "catch"]
    prac = [r for r in rows if r.get("phase") == "practice"]

    catch_p = float(np.mean([_i(r, "correct", 0) for r in catch])) if catch else float("nan")
    fwd = [_i(r, "correct", 0) for r in main if _f(r, "direction") > 0]
    bwd = [_i(r, "correct", 0) for r in main if _f(r, "direction") < 0]
    # matched on delta: compare the median delta at which each direction was run, too
    dir_test = None
    if len(fwd) > 20 and len(bwd) > 20:
        dir_test = {"p_forward": float(np.mean(fwd)), "p_backward": float(np.mean(bwd)),
                    "n_forward": len(fwd), "n_backward": len(bwd),
                    "median_delta_forward": float(np.median([abs(_f(r, "delta_realised_ms"))
                                                             for r in main if _f(r, "direction") > 0])),
                    "median_delta_backward": float(np.median([abs(_f(r, "delta_realised_ms"))
                                                              for r in main if _f(r, "direction") < 0])),
                    "chi2_p": float(stats.chi2_contingency(
                        [[sum(fwd), len(fwd) - sum(fwd)], [sum(bwd), len(bwd) - sum(bwd)]])[1])}

    # which interval held the target -- should be 50/50 and uninformative
    pos = [_i(r, "target_position") for r in main]
    pos_p = float(stats.binomtest(sum(1 for p in pos if p == 1), len(pos), 0.5).pvalue) if pos else float("nan")
    resp = [_i(r, "response") for r in main]
    bias_p = float(stats.binomtest(sum(1 for p in resp if p == 1), len(resp), 0.5).pvalue) if resp else float("nan")

    # does the level rove carry information about the answer?
    rove_gap, rove_auc = [], float("nan")
    for r in main:
        t = _i(r, "target_position")
        if t in (1, 2):
            rove_gap.append(_f(r, f"rove_db_{t}") - _f(r, f"rove_db_{3 - t}"))
    if len(rove_gap) > 20:
        rove_auc = float(np.mean(np.asarray(rove_gap) > 0))

    # drift: first half versus second half of the main trials, at matched delta
    drift = None
    if len(main) > 60:
        half = len(main) // 2
        a = [(abs(_f(r, "delta_realised_ms")), _i(r, "correct", 0)) for r in main[:half]]
        b = [(abs(_f(r, "delta_realised_ms")), _i(r, "correct", 0)) for r in main[half:]]
        drift = {"p_first_half": float(np.mean([c for _, c in a])),
                 "p_second_half": float(np.mean([c for _, c in b])),
                 "median_delta_first": float(np.median([d for d, _ in a])),
                 "median_delta_second": float(np.median([d for d, _ in b]))}

    slopes = {n: r.fit.sigma for n, r in res.items() if r.fit is not None and r.fit.reliable}
    unreliable = [n for n, r in res.items() if r.fit is not None and not r.fit.reliable]
    # a censored track DID converge -- its reversals just sat on a clamp -- so it counts as
    # converged here and is reported separately as censored. Conflating the two made the
    # convergence rate look like a problem with the staircase when the problem is delta_max_ms.
    n_cens = sum(len(r.censored_thresholds or []) for r in res.values())
    n_conv = sum(len(r.track_thresholds) for r in res.values()) + n_cens
    n_att = sum(r.n_tracks_attempted for r in res.values())
    return {
        "n_main": len(main), "n_catch": len(catch), "n_practice": len(prac),
        "catch_p_correct": catch_p, "catch_miss_rate": 1.0 - catch_p if catch else float("nan"),
        "catch_ok": bool(catch and (1.0 - catch_p) <= cfg.max_catch_miss_rate),
        "practice_rounds": [p for m in metas for p in m.get("practice", [])],
        "direction": dir_test,
        "target_position_balance_p": pos_p, "response_bias_p": bias_p,
        "rove_favours_target": rove_auc,
        "drift": drift,
        "tracks_converged": n_conv, "tracks_attempted": n_att, "tracks_censored": n_cens,
        "convergence_rate": n_conv / n_att if n_att else float("nan"),
        "ceiling_trials": sum(r.ceiling_trials for r in res.values()),
        "floor_trials": sum(r.floor_trials for r in res.values()),
        "fitted_slopes": slopes, "unreliable_fits": unreliable,
        "slope_range": (min(slopes.values()), max(slopes.values())) if slopes else None,
        "feedback": [m.get("feedback") for m in metas],
        "auto": [m.get("auto") for m in metas],
        "calibrated": [bool((m.get("calibration") or {}).get("measured_db_spl")) for m in metas],
    }


# ----------------------------------------------------------------------------
# the report
# ----------------------------------------------------------------------------
def analyse(dirs: Sequence[Path], cfg: Optional[Config] = None, n_boot: Optional[int] = None) -> str:
    rows, metas = load(dirs)
    if not rows:
        return "no trials found"
    if cfg is None:
        cfg = Config.from_dict(metas[0]["config"])
    nb = n_boot if n_boot is not None else cfg.n_boot
    d = validate(cfg)
    res = thresholds(rows, cfg)
    diag = diagnostics(rows, metas, cfg, res)
    idx = coherence_index(res, cfg, n_boot=nb)
    jdx = coherence_index(res, cfg, n_boot=nb, a_kind="scrambled")
    trend = trend_test(res)
    kinds = control_kinds(res) or ["scrambled"]
    inters = {k: interaction_test(res, cfg, n_boot=nb, control_kind=k) for k in kinds}
    inter = next((v for v in inters.values() if v.get("ok")), inters[kinds[0]])
    build = buildup_test(res, cfg, n_boot=nb)
    cmp_ = compare_predictors(idx, cfg)
    ends = endpoint_check(res)

    L: List[str] = []
    A = L.append
    codes = sorted({r["_code"] for r in rows})
    sim = [a for a in diag["auto"] if a]
    A("=" * 78)
    if sim:
        A("*** SIMULATED DATA (observer mode: " + ", ".join(sorted(set(sim))) + ") ***")
        A("*** Nothing below describes a listener. It describes a generative model. ***")
        A("=" * 78)
    A(f"two-tone coherence  |  {len(dirs)} session(s), participant(s) {', '.join(codes)}")
    A(f"config {cfg.hash()}   A {cfg.f_a_hz:.0f} Hz / B {d.f_b_hz:.0f} Hz ({cfg.df_semitones:g} st)   "
      f"{cfg.tone_ms:g} ms tones, {cfg.soa_ms:g} ms SOA, {cfg.n_precursor} precursors")
    A(f"threshold = {100 * target_proportion(cfg.n_down):.1f}% correct point, "
      f"geometric mean of the last {cfg.n_final_reversals} reversals")
    A("")

    # ---- is this session usable at all ----------------------------------------
    A("-- session validity " + "-" * 57)
    A(f"  main trials {diag['n_main']}   catch {diag['n_catch']}   practice {diag['n_practice']}")
    if diag["n_catch"]:
        A(f"  catch trials at {cfg.catch_delta_ms:g} ms: {diag['catch_p_correct']:.1%} correct "
          f"(miss rate {diag['catch_miss_rate']:.1%}, limit {cfg.max_catch_miss_rate:.0%})"
          f"  -> {'OK' if diag['catch_ok'] else 'FAILED: treat every threshold below as unsafe'}")
        if cfg.catch_at_pct is None:
            A("    CAVEAT: these probes were built from whichever condition hosted them, so their")
            A("    difficulty tracked that condition -- many times threshold in an easy one, barely")
            A("    above it in a hard one. A miss therefore cannot be read as a lapse, and the")
            A("    verdict above is not a measure of attention either way. The verdict stands as")
            A("    recorded; the remedy is to re-run with catch_at_pct set, not to reinterpret it.")
    else:
        A("  no catch trials: nothing here can tell you whether the listener stayed awake.")
    A(f"  tracks converged {diag['tracks_converged']}/{diag['tracks_attempted']} "
      f"({diag['convergence_rate']:.0%})"
      + (f", of which {diag['tracks_censored']} are censored by the delta ceiling"
         if diag["tracks_censored"] else ""))
    if diag["ceiling_trials"] or diag["floor_trials"]:
        A(f"  trials pinned at the delta ceiling {diag['ceiling_trials']}, at the floor "
          f"{diag['floor_trials']}  (a floor-pinned track gives an upper bound, not a threshold)")
    for p in diag["practice_rounds"]:
        A(f"  practice round {p['round']}: {p['p_correct']:.0%} (criterion {p['criterion']:.0%})")
    A(f"  target interval balance p={diag['target_position_balance_p']:.2f}; "
      f"response bias p={diag['response_bias_p']:.2f}")
    if np.isfinite(diag["rove_favours_target"]):
        A(f"  level rove favoured the target interval on {diag['rove_favours_target']:.1%} of trials "
          f"(50% means it carried no information)")
    if diag["direction"]:
        dd = diag["direction"]
        A(f"  early vs late shift: {dd['p_backward']:.1%} vs {dd['p_forward']:.1%} correct "
          f"(chi2 p={dd['chi2_p']:.2f}; median delta {dd['median_delta_backward']:.1f} vs "
          f"{dd['median_delta_forward']:.1f} ms)")
    if diag["drift"]:
        dr = diag["drift"]
        A(f"  first half {dr['p_first_half']:.1%} correct at {dr['median_delta_first']:.1f} ms, "
          f"second half {dr['p_second_half']:.1%} at {dr['median_delta_second']:.1f} ms")
    if not any(diag["calibrated"]):
        A("  WARNING: no measured calibration recorded for any session.")
    if any(f is not None and not f for f in diag["feedback"]):
        A("  feedback was OFF in at least one session.")
    A("")

    # ---- thresholds -------------------------------------------------------------
    A("-- thresholds " + "-" * 63)
    A("  condition      dT%   A-kind     tracks   staircase    fit (all trials)   slope   %corr")
    censored_names = []
    for name, r in sorted(res.items(), key=lambda kv: (kv[1].a_kind, kv[1].lag_pct)):
        st = f"{r.geomean_ms:7.2f} ms" if r.geomean_ms else (
            f">={r.bound_ms:6.1f} ms" if r.censored else "     --   ")
        if r.censored:
            censored_names.append(name)
        ft = f"{r.fit.threshold_ms:7.2f} ms" if r.fit else "     --   "
        sl = (f"{r.fit.sigma:5.2f}" if r.fit and r.fit.reliable else ("  ~  " if r.fit else "   --"))
        A(f"  {name:<13} {r.lag_pct:5.1f}  {r.a_kind:<10} "
          f"{len(r.track_thresholds)}/{r.n_tracks_attempted}    {st}    {ft}   {sl}   "
          f"{r.prop_correct:5.1%}" + ("   CENSORED" if r.censored else ""))
    if res:
        both = [(r.geomean_ms, r.fit.threshold_ms) for r in res.values()
                if r.geomean_ms and r.fit and np.isfinite(r.fit.threshold_ms)]
        if len(both) >= 3:
            lr = np.log([a for a, _ in both]); lf = np.log([b for _, b in both])
            A(f"  the two routes agree to within a factor of {math.exp(np.max(np.abs(lr - lf))):.2f} "
              f"(r={np.corrcoef(lr, lf)[0, 1]:.3f} on log thresholds)")
    if censored_names:
        A("")
        A(f"  CENSORED: {', '.join(censored_names)}. Some of the reversals averaged into those")
        A(f"  thresholds sit exactly on the delta {res[censored_names[0]].censor_side} "
          f"({cfg.delta_max_ms:g} ms). The staircase")
        A("  called for a step it could not take, so the value recorded is the clamp, not a")
        A("  measurement. Those conditions are reported as bounds, are kept out of kappa and out")
        A("  of the tests, and the fix is a wider delta range rather than a longer session.")
    if diag["unreliable_fits"]:
        A(f"  slope not estimable for {len(diag['unreliable_fits'])} condition(s) "
          f"(marked ~): an adaptive track piles its trials at threshold, which pins the")
        A("    threshold down and leaves the slope barely constrained. The threshold from those "
          "fits is still usable.")
    if diag["slope_range"]:
        A(f"  fitted slopes span {diag['slope_range'][0]:.2f} to {diag['slope_range'][1]:.2f} log units"
          + ("  -- similar, so the staircase's small downward bias cancels in the ratios below"
             if diag['slope_range'][1] / max(diag['slope_range'][0], 1e-9) < 2
             else "  -- NOT similar; the staircase bias differs by condition and the ratios below "
                  "inherit that. Prefer the fitted thresholds."))
    A("")

    # ---- replication anchor -----------------------------------------------------
    A("-- against Elhilali et al. (2009), Figure 2 " + "-" * 33)
    for label, e in ends.items():
        if not e.get("ok"):
            A(f"  {label}: not measured")
            continue
        A(f"  {label}: {e['ms']:.2f} ms   published {e['published_range_ms'][0]:g}-"
          f"{e['published_range_ms'][1]:g} ms   "
          + ("consistent" if e["inside"] else f"OUTSIDE, by a factor of {e['factor_off']:.2f}"))
    A("")

    # ---- the curve ---------------------------------------------------------------
    A("-- kappa: how far from 'one object' towards 'no low tone at all' " + "-" * 12)
    if not idx.get("ok"):
        A(f"  not computed: {idx['reason']}")
    else:
        A(f"  scale: dT=0% {idx['floor_ms']:.2f} ms  ->  B-only {idx['ceiling_ms']:.2f} ms   "
          f"(log span {idx['span']:.2f}, 95% CI {idx['span_ci'][0]:.2f} to {idx['span_ci'][1]:.2f})")
        A("    dT%    kappa observed (95% CI)     model l2/l1")
        for p, k, (lo, hi) in zip(idx["pcts"], idx["kappa"], idx["ci"]):
            A(f"   {p:5.1f}    {k:5.2f}  ({lo:5.2f}, {hi:5.2f})        {d.model_curve.get(p, float('nan')):.3f}")
        if jdx.get("ok"):
            A(f"  scrambled control, on the SAME scale ({idx['scale']}):")
            for p, k, (lo, hi) in zip(jdx["pcts"], jdx["kappa"], jdx["ci"]):
                A(f"   {p:5.1f}    {k:5.2f}  ({lo:5.2f}, {hi:5.2f})")
    A("")

    # ---- the tests ----------------------------------------------------------------
    A("-- tests " + "-" * 68)
    A("  H1  does threshold rise with dT?   [establishes the effect; separates nothing]")
    if trend.get("ok"):
        A(f"      Spearman rho = {trend['rho']:+.3f} over {trend['n_tracks']} tracks at "
          f"{trend['n_levels']} levels, permutation p = {trend['p_one_sided']:.4f} (one sided)")
    else:
        A(f"      not run: {trend['reason']}")

    A("")
    A("  H2  coherent vs scrambled at matched dT   [THE test: separates coherence from")
    A("      interval discrimination and from local acoustic overlap, both of which")
    A("      predict no difference at any dT because the final pair is identical]")
    A("      The coherence account predicts a NEGATIVE slope: a coherent low-tone sequence")
    A("      helps a great deal at synchrony and not at all at alternation.")
    any_ok = False
    for k, t in inters.items():
        A("")
        if not t.get("ok"):
            A(f"      [{k}] not run: {t['reason']}")
            continue
        any_ok = True
        A(f"      [{k} control, {t['n_levels']} matched dT levels]")
        for p, dl, (lo, hi) in zip(t["pcts"], t["diff_log"], t["diff_ci"]):
            A(f"        dT={p:5.1f}%   control - coherent = {dl:+.2f} log units "
              f"(x{math.exp(dl):.2f}), 95% CI {lo:+.2f} to {hi:+.2f}")
        A(f"        slope {t['slope_per_pct'] * 100:+.3f} log units per 100% dT, 95% CI "
          f"{t['slope_ci'][0] * 100:+.3f} to {t['slope_ci'][1] * 100:+.3f}; permutation "
          f"p = {t['p_slope_negative']:.4f}")
        A("        -> " + ("consistent with coherence" if t["p_slope_negative"] < 0.05
                           else "NOT resolved by this control"))
    if any_ok and len(inters) > 1:
        agree = {k: t["p_slope_negative"] < 0.05 for k, t in inters.items() if t.get("ok")}
        A("")
        A(f"      the controls {'AGREE' if len(set(agree.values())) == 1 else 'DISAGREE'}"
          f" ({', '.join(f'{k}: ' + ('yes' if v else 'no') for k, v in agree.items())}). They fail "
          "in different")
        A("      directions -- scrambled matches tone count and energy but is not perfectly "
          "decoherent; pair_only")
        A("      has no A sequence at all but holds fewer tones -- so the argument rests on their "
          "agreeing.")

    A("")
    A("  H3  five precursors vs thirteen   [separates coherence from anything local]")
    if build.get("ok"):
        if build["confounded"]:
            A("      WARNING: this design has no b_only condition at the long precursor length, so")
            A("      the raw difference below cannot be told apart from 'a longer B rhythm is easier")
            A("      to judge' -- which predicts the same direction. Add one before interpreting it.")
        elif build["ceiling_gain"]:
            g = build["ceiling_gain"]
            A(f"      with no low tone at all, the longer precursor moved threshold by "
              f"{g['diff_log']:+.2f} log units")
            A(f"      (95% CI {g['ci'][0]:+.2f} to {g['ci'][1]:+.2f}) -- the part that is about the B "
              "rhythm and not about binding.")
        for e in build["levels"]:
            A(f"      dT={e['pct']:5.1f}%   long - short = {e['diff_log']:+.2f} log units, "
              f"95% CI {e['ci'][0]:+.2f} to {e['ci'][1]:+.2f}"
              + ("" if "interaction_log" in e else "   [confounded, see above]"))
            if "interaction_log" in e:
                A(f"                  over and above the ceiling's own gain: "
                  f"{e['interaction_log']:+.2f} log units, 95% CI "
                  f"{e['interaction_ci'][0]:+.2f} to {e['interaction_ci'][1]:+.2f}, "
                  f"p = {e['p_interaction_negative']:.3f}")
    else:
        A(f"      not run: {build['reason']}")

    A("")
    A("  H4  observed shape vs the model curve   [reported, but see the caveat]")
    if cmp_.get("ok"):
        col = cmp_["collinearity"]
        for k, v in cmp_["fits"].items():
            A(f"      {k:<18} r2 = {v['r2']:.3f}   AICc = {v['aicc']:+.1f}")
        A(f"      best: {cmp_['best']} by dAICc = {cmp_['delta_aicc']:.1f}")
        if col.get("ok"):
            A(f"      CAVEAT: over these dT levels the model curve and a straight line correlate at "
              f"r = {col['r_coherence_vs_linear']:.4f}")
            A(f"      and differ by at most {col['max_normalised_gap']:.2f} once both are rescaled to "
              f"[0,1]. They are")
            A(f"      {'' if col['separable'] else 'NOT '}separable by this design. "
              + ("" if col["separable"] else "Do not report the winner as evidence for either."))
    else:
        A(f"      not run: {cmp_['reason']}")
    A("")
    A("-- what this session does not establish " + "-" * 37)
    A("  * a monotone rise is predicted by every account on the table; only H2 and H3 choose")
    A("    between them.")
    A("  * kappa is normalised by two conditions measured in the same session, so a bad floor")
    A("    or a bad ceiling moves the whole curve. Both are printed above in milliseconds.")
    if len(codes) == 1:
        A("  * one listener. Nothing here generalises; it is a threshold sweep for this person.")
    A("=" * 78)
    return "\n".join(L)


# ----------------------------------------------------------------------------
# power
# ----------------------------------------------------------------------------
def power(cfg: Config, modes: Sequence[str] = ("coherence", "pedestal", "null"),
          n_sessions: int = 40, floor_ms: float = 3.0, ceiling_ms: float = 15.0,
          sigma: float = 0.6, seed: int = 23, n_boot: int = 1500) -> dict:
    """Run the whole pipeline on simulated listeners and count how often each test is right.

    This is the only honest way to state what the experiment can detect, because the quantity
    of interest is a ratio of thresholds from an adaptive procedure with its own bias and
    spread, and no closed form covers that. Three generative truths are run:

      coherence  the hypothesis. H1 and H2 should both fire.
      pedestal   thresholds rise with the judged interval, coherent and scrambled alike. H1
                 SHOULD still fire -- which is the point: H1 is not diagnostic -- and H2
                 should not.
      null       nothing depends on dT. Neither should fire; the rate at which they do is the
                 false positive rate.
    """
    from .observer import SimulatedListener
    from .track import Track

    d = validate(cfg)
    out = {}
    for mode in modes:
        h1, h2, h2_wrong, kappas = 0, 0, 0, []
        for s in range(n_sessions):
            sim = SimulatedListener(cfg, mode=mode, floor_ms=floor_ms, ceiling_ms=ceiling_ms,
                                    sigma=sigma, seed=seed + 1000 * s)
            res: Dict[str, CondResult] = {}
            for c in d.conditions:
                thr = []
                for k in range(cfg.tracks_per_condition):
                    t = Track(cfg, c.name, 0)
                    while not t.finished:
                        t.update(sim.respond(c, t.delta))
                    v = t.threshold_ms()
                    if v:
                        thr.append(v)
                res[c.name] = CondResult(c.name, c.lag_pct, c.a_kind,
                                         cfg.n_precursor if c.n_precursor is None else c.n_precursor,
                                         sorted(thr), cfg.tracks_per_condition,
                                         float(np.exp(np.mean(np.log(thr)))) if thr else None,
                                         None, 0, float("nan"), 0, 0)
            t1 = trend_test(res, n_perm=2000, seed=seed + s)
            if t1.get("ok") and t1["p_one_sided"] < 0.05:
                h1 += 1
            t2 = interaction_test(res, cfg, n_boot=n_boot, seed=seed + s, n_perm=2000)
            if t2.get("ok"):
                if t2["p_slope_negative"] < 0.05:
                    h2 += 1
                if 1.0 - t2["p_slope_negative"] < 0.05:
                    h2_wrong += 1
            i = coherence_index(res, cfg, n_boot=n_boot, seed=seed + s)
            if i.get("ok"):
                kappas.append(i["kappa"])
        out[mode] = {"n_sessions": n_sessions,
                     "H1_rate": h1 / n_sessions, "H2_rate": h2 / n_sessions,
                     "H2_wrong_sign_rate": h2_wrong / n_sessions,
                     "mean_kappa": (np.mean(kappas, axis=0).tolist() if kappas else None),
                     "sd_kappa": (np.std(kappas, axis=0).tolist() if len(kappas) > 1 else None)}
    return {"modes": out, "tracks_per_condition": cfg.tracks_per_condition,
            "floor_ms": floor_ms, "ceiling_ms": ceiling_ms, "sigma": sigma,
            "note": "H1 firing under 'pedestal' is correct behaviour, not a false positive: "
                    "a monotone rise is what that rival predicts too. H2 is the diagnostic test."}
