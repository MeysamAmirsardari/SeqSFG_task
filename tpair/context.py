"""Does binding persist? The context-to-test design, its predictions, and how to read it.

The question
------------
Is the binding of A and B computed moment by moment from the timing in front of the listener,
or is it a state that outlasts the timing that built it? Each interval starts with a CONTEXT of
`n_context` tokens at one lag, then switches to the TEST lag for the remaining tokens, and the
target is a change of the lag in one of the last test tokens. The test part is physically the
same in every condition; only its history differs. Ferret A1 shows the neural version: responses
after a synchronous context are enhanced, after an alternating one suppressed, fading within
about 0.8 s (Lu et al. 2017, Nat Commun). Human streaming shows the opposite sign for frequency
separation: a large-separation context reduces later streaming (Snyder et al. 2008). Nobody has
measured it for the A-B timing relation, objectively, with the tones' own timing taken away.

Why the numbers are what they are (L01's sessions)
--------------------------------------------------
* With the tones' own timing removed (scatter timing, split change), the relation was usable at
  0% (1.5 / 2.4 ms) and at 25% "in" (7.3 ms), and at nothing from 50% upwards: every track there
  hit the ceiling. The test therefore sits at 25% (15 ms), on the steep part of the curve,
  where a shift of the binding state should move the threshold most.
* The two contexts sit at the same distance either side of it, 0 and 30 ms. A context that is
  simply further away is a bigger change at the switch, and any account in which a bigger
  change costs more would otherwise mimic persistence.
* "Out" at 25% hit a 55 ms ceiling; the fit put it near 33 ms. Here the largest lag is 30 ms,
  which buys room: the ceiling is 70 ms.

What each account predicts
--------------------------
With contexts L- < test < L+ (0, 15, 30 ms) and T the threshold:

  persistence   T(L-) < T(stationary) < T(L+), in both directions: the bound context helps
  contrast      the reverse, in both directions: the context repels (adaptation)
  reference     T(L-) and T(L+) both above T(stationary): a listener who does not separate the
                context from the test takes a polluted reference, whatever its sign
  decoy         the switch is itself a change of the lag, in BOTH intervals. Into the test from
                L+ it looks like an "in" target; from L- like an "out" one. So it costs L+ in "in"
                blocks and L- in "out" blocks: the sign flips with the direction.

Hence three numbers, all in log units of threshold:

  P  = mean over directions of ln T(L+)/T(L-)            persistence > 0 > contrast
  D  = half the difference of ln T(L+)/T(L-), in - out   the decoy; nothing to do with binding
  S  = mean over directions and both contexts of ln T(L)/T(stationary)   the reference cost

Only P speaks to binding, and only because both directions are run: a single direction cannot
separate persistence from the decoy.
"""
from __future__ import annotations

import math
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import Config, conditions


# ---------------------------------------------------------------------------- the model
@lru_cache(maxsize=256)
def context_index(cfg: Config, context_lag: float, test_lag: float, window_ms: float = 600.0,
                  n_seq: int = 30, seed: int = 0) -> float:
    """lambda2/lambda1 of the coherence model over the `window_ms` that end with the last token.

    The model filters causally, so the context reaches the window only through the filters' own
    memory (the slowest is 2 Hz). Whatever context effect it shows is the one temporal coherence
    predicts with no persistence added.
    """
    from tcoh.model import coherence_matrix
    from .model import _env
    from .stimulus import reference_onsets
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_seq):
        t = reference_onsets(cfg, rng) + 200.0
        lags = np.full(cfg.n_pairs, float(test_lag))
        lags[:cfg.n_context] = context_lag
        end = int(t[-1] + max(lags[-1], 0.0) + cfg.tone_ms)
        e = np.stack([_env(t, end, cfg.tone_ms, cfg.ramp_ms),
                      _env(t + lags, end, cfg.tone_ms, cfg.ramp_ms)])
        vals.append(coherence_matrix(e, 1000.0, trim_s=max(end - window_ms, 0.0) / 1000.0).ratio)
    return float(np.mean(vals))


def model_table(cfg: Config, window_ms: float = 600.0) -> Dict[float, Dict[float, float]]:
    """{test lag: {context lag: index}}."""
    return {L: {C: context_index(cfg, C, L, window_ms) for C in cfg.context_lags_ms}
            for L in cfg.lags_ms}


# ---------------------------------------------------------------------------- reading it
def _value(r):
    """A condition's threshold for these indices: the geometric mean, or the pooled bound when a
    track hit the ceiling (then the index built on it is itself a bound)."""
    vals = list(r.track_thresholds) + list(r.censored_thresholds)
    if not vals:
        return None, False
    return float(np.exp(np.mean(np.log(vals)))), bool(r.censored_thresholds)


def _track_sd(res) -> Optional[float]:
    """Pooled within-condition SD of log track thresholds, from conditions with two or more."""
    ss, df = 0.0, 0
    for r in res.values():
        t = np.log(r.track_thresholds) if len(r.track_thresholds) > 1 else None
        if t is not None:
            ss += float(np.sum((t - t.mean()) ** 2)); df += len(t) - 1
    return math.sqrt(ss / df) if df else None


def effects(res, cfg: Config) -> dict:
    """P, D and S for every test lag, with what can be said of each."""
    out = {}
    sd = _track_sd(res)
    for L in cfg.lags_ms:
        lo = [C for C in cfg.context_lags_ms if C < L - 1e-9]
        hi = [C for C in cfg.context_lags_ms if C > L + 1e-9]
        if not lo or not hi:
            out[L] = {"ok": False, "reason": "needs a context either side of the test lag"}
            continue
        C_lo, C_hi = max(lo), min(hi)              # the nearest pair either side
        per_dir, bounded, n_used = {}, False, []
        for d in cfg.directions:
            pick = {c.context_lag_ms: res.get(c.name) for c in conditions(cfg)
                    if c.direction == d and abs(c.lag_ms - L) < 1e-9}
            vals = {}
            for C, r in pick.items():
                if r is None:
                    continue
                v, b = _value(r)
                vals[C] = v; bounded |= b and C in (C_lo, C_hi, L)
                n_used.append(max(len(r.track_thresholds), 1))
            if any(vals.get(C) is None for C in (C_lo, C_hi, L)):
                continue
            per_dir[d] = {"T": vals, "ln_hi_lo": math.log(vals[C_hi] / vals[C_lo]),
                          "ln_lo_s": math.log(vals[C_lo] / vals[L]),
                          "ln_hi_s": math.log(vals[C_hi] / vals[L])}
        if not per_dir:
            out[L] = {"ok": False, "reason": "no direction has all three conditions"}
            continue
        P = float(np.mean([v["ln_hi_lo"] for v in per_dir.values()]))
        S = float(np.mean([(v["ln_lo_s"] + v["ln_hi_s"]) / 2 for v in per_dir.values()]))
        D = (float((per_dir["in"]["ln_hi_lo"] - per_dir["out"]["ln_hi_lo"]) / 2)
             if {"in", "out"} <= set(per_dir) else None)
        n = min(n_used) if n_used else 1
        se_P = sd * math.sqrt(2.0 / n) / math.sqrt(len(per_dir)) if sd else None
        out[L] = {"ok": True, "context_low": C_lo, "context_high": C_hi, "per_direction": per_dir,
                  "P": P, "D": D, "S": S, "se_P": se_P, "track_sd_log": sd,
                  "bounded": bounded, "both_directions": D is not None}
    return out


def verdict(e: dict) -> str:
    """One line on what P says, honest about how much it can say."""
    if not e.get("ok"):
        return e.get("reason", "")
    P, se = e["P"], e["se_P"]
    sign = ("persistence: the bound context helped" if P > 0
            else "contrast: the bound context hurt")
    if e["bounded"]:
        return f"P = {P:+.2f} ({sign}), but built on ceiling-limited conditions: a bound, not a size"
    if se is None:
        return (f"P = {P:+.2f} ({sign}); one track per condition, so no error bar: a direction, "
                "not a result")
    z = P / se if se else float("nan")
    if abs(z) < 2:
        return (f"P = {P:+.2f} +- {se:.2f} (z = {z:+.1f}): no reliable context effect in one "
                "listener; it counts towards the group")
    return f"P = {P:+.2f} +- {se:.2f} (z = {z:+.1f}): {sign}"


# ---------------------------------------------------------------------------- trial level
def _onsets(r, which):
    v = str(r.get(which, "") or "").strip()
    return np.array([float(x) for x in v.split(";")]) if v else None


def time_since_switch(cfg: Config, r: dict) -> Optional[float]:
    """ms from the first test token to the target, in the target interval."""
    tp, k = int(float(r.get("target_position") or 0)), int(float(r.get("target_pair") or -1))
    a, b = _onsets(r, f"a_onsets_{tp}"), _onsets(r, f"b_onsets_{tp}")
    if a is None or b is None or k < cfg.n_context:
        return None
    first = min(a[cfg.n_context], b[cfg.n_context])
    return float(min(a[k], b[k]) - first)


def decay(rows: Sequence[dict], cfg: Config) -> dict:
    """Does a context effect fade within the trial? Logistic regression of correct on log delta
    and the time since the switch, per context, over the main trials. A context that helps should
    help less the longer ago it ended (a negative time coefficient for L-, positive for L+, under
    persistence). Exploratory: one listener, a few hundred trials."""
    from .analysis import _logit
    out = {}
    by = defaultdict(list)
    for r in rows:
        if r.get("phase") != "main" or str(r.get("timed_out", "0")).strip() in ("1", "True"):
            continue
        c = r.get("context_lag_ms", "")
        t = time_since_switch(cfg, r)
        d = abs(float(r.get("delta_realised_ms") or "nan"))
        if c == "" or t is None or not np.isfinite(d) or d <= 0:
            continue
        by[float(c)].append((math.log(d), t, int(float(r.get("correct") or 0))))
    for C, v in sorted(by.items()):
        if len(v) < 40:
            out[C] = {"ok": False, "n": len(v)}
            continue
        X = np.array(v, float)
        Z = np.column_stack([np.ones(len(X)), X[:, 0] - X[:, 0].mean(),
                             (X[:, 1] - X[:, 1].mean()) / (X[:, 1].std() or 1.0)])
        b, se = _logit(Z, X[:, 2])
        out[C] = {"ok": True, "n": len(X), "coef_per_sd": float(b[2]), "z": float(b[2] / se[2])
                  if se[2] > 0 else float("nan"), "time_sd_ms": float(X[:, 1].std())}
    return out


def carry_over(rows: Sequence[dict]) -> dict:
    """Does the PREVIOUS trial's context matter? Prior-stimulus effects in streaming last seconds
    (Snyder et al. 2008), longer than a trial. Logistic regression of correct on log delta, the
    current context and the previous trial's context lag (within a block)."""
    from .analysis import _logit
    main = [r for r in rows if r.get("phase") in ("main", "catch", "probe")]
    main.sort(key=lambda r: int(float(r.get("trial_index") or 0)))
    X, y = [], []
    prev, prev_block = None, None
    ctxs = sorted({float(r["context_lag_ms"]) for r in main if r.get("context_lag_ms", "") != ""})
    for r in main:
        c = r.get("context_lag_ms", "")
        if (r.get("phase") == "main" and c != "" and prev is not None
                and r.get("block") == prev_block and str(r.get("timed_out", "0")) not in ("1", "True")):
            d = abs(float(r.get("delta_realised_ms") or "nan"))
            if np.isfinite(d) and d > 0:
                X.append([math.log(d)] + [float(float(c) == k) for k in ctxs[1:]] + [prev])
                y.append(int(float(r.get("correct") or 0)))
        prev = float(c) if c != "" else prev
        prev_block = r.get("block")
    if len(y) < 60:
        return {"ok": False, "n": len(y)}
    X = np.array(X, float); y = np.array(y, float)
    cols = [np.ones(len(y)), X[:, 0] - X[:, 0].mean()] + [X[:, i] for i in range(1, X.shape[1] - 1)]
    cols.append((X[:, -1] - X[:, -1].mean()) / (X[:, -1].std() or 1.0))
    b, se = _logit(np.column_stack(cols), y)
    return {"ok": True, "n": len(y), "coef_per_sd": float(b[-1]),
            "z": float(b[-1] / se[-1]) if se[-1] > 0 else float("nan")}


# ---------------------------------------------------------------------------- predictions
def predictions(cfg: Config, modes=("pair", "change", "persist", "contrast"), n: int = 1200,
                seed: int = 0) -> Dict[str, dict]:
    """Every model listener's threshold in every condition, and the P, D, S it implies."""
    from .observer import threshold
    out = {}
    for m in modes:
        th = {c.name: threshold(cfg, c, m, n=n, seed=seed) for c in conditions(cfg)}
        idx = {}
        for L in cfg.lags_ms:
            lo = max(C for C in cfg.context_lags_ms if C < L)
            hi = min(C for C in cfg.context_lags_ms if C > L)
            per = {}
            for d in cfg.directions:
                g = {c.context_lag_ms: th[c.name] for c in conditions(cfg)
                     if c.direction == d and abs(c.lag_ms - L) < 1e-9}
                if all(np.isfinite(g[k]) for k in (lo, hi, L)):
                    per[d] = (math.log(g[hi] / g[lo]), math.log(g[lo] / g[L]), math.log(g[hi] / g[L]))
            if per:
                idx[L] = {"P": float(np.mean([v[0] for v in per.values()])),
                          "S": float(np.mean([(v[1] + v[2]) / 2 for v in per.values()])),
                          "D": (per["in"][0] - per["out"][0]) / 2 if {"in", "out"} <= set(per) else None}
        out[m] = {"thresholds": th, "indices": idx}
    return out


def power(effect_P: float, track_sd_log: float = 0.30, tracks: Sequence[int] = (1, 2, 3, 4),
          directions: int = 2, listeners: Sequence[int] = (1, 6, 12)) -> List[dict]:
    """How often P would come out with z >= 2, for an effect of this size, by tracks per
    condition and number of listeners (each listener's P averaged, n listeners)."""
    rows = []
    for k in tracks:
        se1 = track_sd_log * math.sqrt(2.0 / k) / math.sqrt(directions)
        for n in listeners:
            se = se1 / math.sqrt(n)
            from scipy.stats import norm
            rows.append({"tracks": k, "listeners": n, "se": se,
                         "power": float(norm.sf(2.0 - effect_P / se))})
    return rows
