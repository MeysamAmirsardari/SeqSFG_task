"""Verification battery and ideal observers.

Builds many fresh trials per condition, renders both intervals, measures each
property on the audio (with exact schedule counterparts where the property is a
count), and prints one table per question:

* between the two intervals of a trial (must match at every condition)
* across conditions (must match except for the step itself)
* ideal observers with access to ONE property each (must be at chance)
* an oracle that knows the element schedule (expected to succeed; shows what the
  jitter is protecting)
* the level/masking diagnostic behind the equal-amplitude decision
"""
from __future__ import annotations

import json
import math
import time
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from . import measure as M
from . import pool as pool_mod
from .config import Config, Derived, describe, validate
from .stimulus import BACKGROUND, FIGURE, Interval, Trial, check_invariants, make_trial, render_interval

FRAME_MS = 4.0     # broadband envelope frame
WIN_MS = 40.0      # channel demodulation window
HOP_MS = 1.0


@dataclass
class IntervalMeasures:
    scalars: Dict[str, float]
    spectrum_db: np.ndarray
    occupancy_audio: np.ndarray
    env_feats: Dict[str, float]
    sc_sched: Dict[str, np.ndarray]
    sc_audio: Dict[str, np.ndarray]
    locked_env: np.ndarray
    oracle: Dict[str, float]


def _element_windows(cfg: Config, iv: Interval, span_grid: int) -> List[Tuple[int, int]]:
    return [(int(t), int(t) + span_grid) for t in iv.element_onsets]


def measure_interval(cfg: Config, d: Derived, iv: Interval, x: np.ndarray, S: np.ndarray,
                     span_ms: float, ref: float) -> IntervalMeasures:
    sr, P, D, T = cfg.sample_rate, d.n_channels, d.tone_dur_grid, cfg.interval_dur_ms
    n_grid = cfg.n_grid
    span_grid = cfg.ms_to_grid(span_ms)
    el_ms = iv.element_onsets * cfg.grid_ms
    sc: Dict[str, float] = {}

    # ---- exact, from the schedule ------------------------------------------
    sc["n_tones"] = float(iv.n_tones)
    cnt = M.count_trace(iv.onset, n_grid, D)
    sc["count_mean"], sc["count_sd"] = float(cnt.mean()), float(cnt.std())
    sc["count_min"], sc["count_max"] = float(cnt.min()), float(cnt.max())
    on = M.channel_on_matrix(iv, P, n_grid, D)
    onS = on[S]
    sc["occ_S_mean"] = float(onS.mean())
    sc["occ_S_union"] = float(np.any(onS, axis=0).mean())
    sc["S_max_simul"] = float(onS.sum(axis=0).max())
    simul, sync, elcount = [], [], []
    for k, (a, b) in enumerate(_element_windows(cfg, iv, span_grid)):
        chans = iv.element_sets[k] if iv.element_sets else S
        simul.append(on[chans, a:b].sum(axis=0).max())
        # components starting in the same grid instant: onsets of those channels inside the window
        sel = np.isin(iv.channel, chans) & (iv.onset >= a) & (iv.onset < b)
        sync.append(np.bincount(iv.onset[sel] - a, minlength=1).max() if sel.any() else 0)
        elcount.append(cnt[a:b].mean())
    sc["el_comp_simul"] = float(np.mean(simul))
    sc["el_comp_sync"] = float(np.mean(sync))
    sc["el_count_mean"] = float(np.mean(elcount))
    sc["el_count_sd"] = float(np.std(elcount))
    onsets_sched = [iv.onset[iv.channel == c] * cfg.grid_ms for c in range(P)]
    sc_sched = M.single_channel_stats(onsets_sched, cfg)

    # ---- from the audio ------------------------------------------------------
    sc["rms_db"] = float(20 * np.log10(max(np.sqrt(np.mean(x.astype(float) ** 2)), M.EPS)))
    sc["peak"] = float(np.abs(x).max())
    env = M.frame_rms(x, sr, FRAME_MS)
    ef = M.envelope_features(env, FRAME_MS, cfg.iei_min_ms, cfg.iei_max_ms)
    sc["env_mod_depth"] = ef["mod_depth"]
    sc["env_ac_peak_iei"] = ef["ac_peak_iei"]
    locked = M.element_locked_envelope(env, FRAME_MS, el_ms, span_ms)
    sc["locked_peak_db"] = float(locked.max() - locked.min())
    per_el = M.per_element_rms_db(x, sr, el_ms, span_ms)
    sc["el_rms_mean_db"], sc["el_rms_sd_db"] = float(per_el.mean()), float(per_el.std())
    cenv = M.channel_envelopes(x, sr, d.channel_freqs_hz, WIN_MS, HOP_MS)
    spec = M.channel_power_db(cenv)
    sc["spec_peakedness"] = M.peakedness(spec, cfg.n_components)
    sc["spec_sd"] = float(spec.std())
    states = M.on_states(cenv, ref)
    occ = states.mean(axis=1)
    sc["occ_mean"] = float(occ.mean())
    sc["occ_peakedness"] = M.peakedness(occ, cfg.n_components)
    acount = states.sum(axis=0)
    sc["count_audio_mean"], sc["count_audio_max"] = float(acount.mean()), float(acount.max())
    sc["occ_S_audio"] = float(states[S].mean())
    onsets_audio = [o * HOP_MS for o in M.onsets_from_states(states)]
    sc_audio = M.single_channel_stats(onsets_audio, cfg)
    lvl = np.array([M.db(np.mean(cenv[c, states[c]] ** 2)) if states[c].any() else -120.0 for c in range(P)])
    sc_audio["level_db"] = lvl
    sc_audio["spec_db"] = spec

    # ---- oracle: knows the element windows -----------------------------------
    per_chan_all = np.ones(P, dtype=bool)
    for (a, b) in _element_windows(cfg, iv, span_grid):
        per_chan_all &= np.array([np.any((iv.onset[iv.channel == c] >= a) & (iv.onset[iv.channel == c] < b))
                                  for c in range(P)])
    oracle = {"channels_active_in_every_element_window": float(per_chan_all.sum())}
    return IntervalMeasures(sc, spec, occ, ef, sc_sched, sc_audio, locked, oracle)


# ----------------------------------------------------------------------------
# observers
# ----------------------------------------------------------------------------
def dprime_from_pc(pc: float, n: int) -> float:
    pc = min(max(pc, 0.5 / n), 1 - 0.5 / n)
    return float(math.sqrt(2.0) * stats.norm.ppf(pc))


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


def observer_from_stat(st_target: np.ndarray, st_other: np.ndarray) -> dict:
    """Fixed a-priori rule: pick the interval with the LARGER statistic."""
    n = st_target.size
    if n == 0 or not np.all(np.isfinite(st_target)):
        return dict(n=n, pc=float("nan"), dprime=float("nan"), ci=(float("nan"), float("nan")), p=float("nan"))
    wins = float(np.sum(st_target > st_other) + 0.5 * np.sum(st_target == st_other))
    pc = wins / n
    lo, hi = wilson(wins, n)
    p = stats.binomtest(int(round(wins)), n, 0.5).pvalue
    return dict(n=n, pc=pc, dprime=dprime_from_pc(pc, n), ci=(dprime_from_pc(lo, n), dprime_from_pc(hi, n)), p=p)


def _fit_ridge_logistic(Z: np.ndarray, lam: float, iters: int = 25) -> np.ndarray:
    """No-intercept logistic regression on difference vectors (label +1 by construction), ridge lam."""
    n, p = Z.shape
    w = np.zeros(p)
    I = np.eye(p)
    for _ in range(iters):
        s = Z @ w
        sig = 1.0 / (1.0 + np.exp(-s))
        g = -Z.T @ (1.0 - sig) + lam * w
        H = (Z * (sig * (1 - sig))[:, None]).T @ Z + lam * I
        step = np.linalg.solve(H, g)
        w = w - step
        if np.max(np.abs(step)) < 1e-6:
            break
    return w


def observer_cv(F_target: np.ndarray, F_other: np.ndarray, lam: float = 2.0) -> dict:
    """Leave-one-out linear observer on the feature difference; learns any linear cue that exists."""
    Dm = np.asarray(F_target, float) - np.asarray(F_other, float)
    Dm = Dm[:, np.isfinite(Dm).all(axis=0)]
    n = Dm.shape[0]
    if n < 4 or Dm.shape[1] == 0:
        return dict(n=n, pc=0.5, dprime=0.0, ci=(0.0, 0.0), p=1.0)
    wins = 0.0
    for i in range(n):
        tr = np.delete(Dm, i, axis=0)
        scale = np.sqrt(np.mean(tr ** 2, axis=0)) + 1e-12
        w = _fit_ridge_logistic(tr / scale, lam)
        s = float(w @ (Dm[i] / scale))
        wins += 1.0 if s > 0 else (0.5 if s == 0 else 0.0)
    pc = wins / n
    lo, hi = wilson(wins, n)
    p = stats.binomtest(int(round(wins)), n, 0.5).pvalue
    return dict(n=n, pc=pc, dprime=dprime_from_pc(pc, n), ci=(dprime_from_pc(lo, n), dprime_from_pc(hi, n)), p=p)


def scalar_features(m: IntervalMeasures) -> Dict[str, float]:
    """Every named scalar an observer could use (no schedule knowledge), for the per-feature audit."""
    out = {}
    for k in ("spec_peakedness", "spec_sd", "occ_mean", "occ_peakedness", "rms_db", "peak", "count_audio_mean",
              "count_audio_max", "env_mod_depth", "env_ac_peak_iei"):
        out[k] = m.scalars[k]
    for k, val in m.env_feats.items():
        out["env:" + k] = val
    for k in ("ioi_cv", "ioi_sd", "frac_ioi_in_iei", "pairs_iei_norm", "ioi_min", "ioi_max", "ioi_mean"):
        vv = m.sc_sched[k]
        out[f"ch:{k}:max"], out[f"ch:{k}:min"], out[f"ch:{k}:mean"], out[f"ch:{k}:sd"] = vv.max(), vv.min(), vv.mean(), vv.std()
    for k in ("level_db", "count", "ioi_cv", "pairs_iei_norm"):
        vv = m.sc_audio[k]
        out[f"ch_audio:{k}:max"], out[f"ch_audio:{k}:min"], out[f"ch_audio:{k}:sd"] = vv.max(), vv.min(), vv.std()
    out["spec:max"] = float(m.spectrum_db.max())
    out["occ:max"] = float(m.occupancy_audio.max())
    return out


def feature_matrix(results: List["ConditionResult"], only_variant: Optional[str] = "rising"):
    """(names, D) where D[trial, feature] = feature(A) - feature(B), pooled over one variant's conditions."""
    A, O = [], []
    for r in results:
        if only_variant and r.variant != only_variant:
            continue
        A += [scalar_features(m) for m in r.A]
        O += [scalar_features(m) for m in r.O]
    if not A:
        return [], np.zeros((0, 0))
    names = list(A[0].keys())
    D = np.array([[a[k] - o[k] for k in names] for a, o in zip(A, O)], dtype=float)
    return names, D


def _pc_from_signs(D: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Proportion correct of the 'pick the larger' rule per feature, with trial labels flipped by s."""
    X = D * s[:, None]
    return (np.sum(X > 0, axis=0) + 0.5 * np.sum(X == 0, axis=0)) / D.shape[0]


def feature_audit(results: List["ConditionResult"], only_variant: Optional[str] = "rising") -> List[dict]:
    """Fixed-rule d' of every scalar feature (pick the larger), pooled over the conditions of one variant."""
    names, D = feature_matrix(results, only_variant)
    if not names:
        return []
    n = D.shape[0]
    pc = _pc_from_signs(D, np.ones(n))
    rows = []
    for j, k in enumerate(names):
        wins = pc[j] * n
        lo, hi = wilson(wins, n)
        rows.append(dict(feature=k, dprime=dprime_from_pc(pc[j], n), p=float(stats.binomtest(int(round(wins)), n, 0.5).pvalue),
                         n=n, ci=(dprime_from_pc(lo, n), dprime_from_pc(hi, n)),
                         mean_diff=float(np.mean(D[:, j])),
                         sd_diff=float(np.std(D[:, j], ddof=1)) if n > 1 else 0.0))
    rows.sort(key=lambda r: -abs(r["dprime"]))
    return rows


def permutation_audit(results: List["ConditionResult"], only_variant: Optional[str] = "rising",
                      n_perm: int = 20000, seed: int = 11) -> dict:
    """Global test: is the LARGEST |d'| over all features bigger than relabelling can produce?

    Exchanging the two intervals of a trial is exactly the null hypothesis "the intervals differ in
    nothing an observer can measure". Flipping that label for a random subset of trials therefore gives
    the null distribution of the whole audit, and the largest |d'| over features is corrected for having
    looked at all of them at once.
    """
    names, D = feature_matrix(results, only_variant)
    if not names:
        return {}
    n, p = D.shape
    obs = np.abs(np.array([dprime_from_pc(x, n) for x in _pc_from_signs(D, np.ones(n))]))
    rng = np.random.default_rng(seed)
    null_max = np.empty(n_perm)
    for i in range(n_perm):
        s = rng.choice([-1.0, 1.0], size=n)
        pc = _pc_from_signs(D, s)
        null_max[i] = np.max(np.abs(np.sqrt(2.0) * stats.norm.ppf(np.clip(pc, 0.5 / n, 1 - 0.5 / n))))
    obs_max = float(np.max(obs))
    return dict(n_trials=n, n_features=p, n_perm=n_perm, variant=only_variant,
                observed_max_dprime=obs_max, worst_feature=names[int(np.argmax(obs))],
                p_value=float((np.sum(null_max >= obs_max) + 1) / (n_perm + 1)),
                null_max_median=float(np.median(null_max)), null_max_p95=float(np.percentile(null_max, 95)),
                single_feature_95=float(dprime_from_pc(0.5 + 1.96 * math.sqrt(0.25 / n), n)),
                null_max=null_max, observed=obs, names=names)


def feature_sets(m: IntervalMeasures, cfg: Config) -> Dict[str, Tuple[np.ndarray, float]]:
    """Per observer: (feature vector for the learnt observer, scalar for the fixed rule)."""
    env_keys = sorted(m.env_feats)
    env_vec = np.array([m.env_feats[k] for k in env_keys])
    occ_sorted = np.sort(m.occupancy_audio)[::-1]
    sc_keys = ["ioi_cv", "ioi_sd", "frac_ioi_in_iei", "pairs_iei_norm", "ioi_min", "ioi_max", "ioi_mean"]
    sc_vec = []
    for k in sc_keys:
        v = m.sc_sched[k]
        sc_vec += [v.max(), v.min(), v.mean(), v.std()]
    for k in ("level_db", "count", "ioi_cv", "pairs_iei_norm"):
        v = m.sc_audio[k]
        sc_vec += [v.max(), v.min(), v.std()]
    sc_vec = np.array(sc_vec)
    spec_vec = np.concatenate([m.spectrum_db, [m.scalars["spec_peakedness"], m.scalars["spec_sd"], m.spectrum_db.max()]])
    occ_vec = np.concatenate([occ_sorted, [m.scalars["occ_peakedness"], m.occupancy_audio.std()]])
    out = {
        "spectrum only": (spec_vec, m.scalars["spec_peakedness"]),
        "envelope only": (env_vec, m.env_feats["mod_depth"]),
        "occupancy only": (occ_vec, m.scalars["occ_peakedness"]),
        "single-channel stats": (sc_vec, float(m.sc_sched["pairs_iei_norm"].mean())),
    }
    out["all of the above"] = (np.concatenate([v[0] for v in out.values()]), float("nan"))
    out["oracle: knows element windows"] = (np.array([m.oracle["channels_active_in_every_element_window"]]),
                                            m.oracle["channels_active_in_every_element_window"])
    return out


# ----------------------------------------------------------------------------
# the battery
# ----------------------------------------------------------------------------
ROWS_BETWEEN = [
    ("total tones", "n_tones", "{:.0f}"),
    ("tones sounding: mean", "count_mean", "{:.2f}"),
    ("tones sounding: sd", "count_sd", "{:.2f}"),
    ("tones sounding: min", "count_min", "{:.0f}"),
    ("tones sounding: max", "count_max", "{:.0f}"),
    ("tones sounding (audio): mean", "count_audio_mean", "{:.2f}"),
    ("tones sounding (audio): max", "count_audio_max", "{:.1f}"),
    ("long-term RMS (dB FS)", "rms_db", "{:.2f}"),
    ("peak amplitude", "peak", "{:.2f}"),
    ("spectrum peakedness (dB)", "spec_peakedness", "{:.2f}"),
    ("spectrum sd across channels (dB)", "spec_sd", "{:.2f}"),
    ("occupancy, all channels (audio)", "occ_mean", "{:.3f}"),
    ("occupancy peakedness (audio)", "occ_peakedness", "{:.3f}"),
    ("occupancy of S: per channel", "occ_S_mean", "{:.3f}"),
    ("occupancy of S: per channel (audio)", "occ_S_audio", "{:.3f}"),
    ("occupancy of S: union", "occ_S_union", "{:.3f}"),
    ("S channels on simultaneously: max", "S_max_simul", "{:.2f}"),
    ("figure comps sounding together (per element max)", "el_comp_simul", "{:.2f}"),
    ("figure comps starting together (per element max)", "el_comp_sync", "{:.2f}"),
    ("tones sounding inside element windows", "el_count_mean", "{:.2f}"),
    ("envelope modulation depth", "env_mod_depth", "{:.3f}"),
    ("envelope autocorr peak at IEI lags", "env_ac_peak_iei", "{:.3f}"),
    ("element-locked envelope: peak-to-trough (dB)", "locked_peak_db", "{:.2f}"),
    ("element RMS: mean (dB FS)", "el_rms_mean_db", "{:.2f}"),
    ("element RMS: sd over elements (dB)", "el_rms_sd_db", "{:.2f}"),
]


@dataclass
class ConditionResult:
    variant: str
    step_ms: float
    span_ms: float
    A: List[IntervalMeasures]
    O: List[IntervalMeasures]
    invariants: List[dict]
    iei_ms: List[np.ndarray]
    locked_diff: List[np.ndarray]
    spec_absdiff_mean: List[float]
    spec_absdiff_max: List[float]


def run_condition(cfg: Config, d: Derived, variant: str, step_ms: float, n_trials: int, seed: int,
                  ref: float, progress=None) -> ConditionResult:
    n_comp = 1 if variant == "onechannel" else cfg.n_components
    span_ms = (n_comp - 1) * step_ms + cfg.tone_dur_ms
    res = ConditionResult(variant, step_ms, span_ms, [], [], [], [], [], [], [])
    for j in range(n_trials):
        tseed = int(np.random.default_rng([seed, zlib.crc32(variant.encode()) & 0xFFFF, int(step_ms * 1000), j]).integers(2 ** 31 - 1))
        tr = make_trial(cfg, tseed, step_ms, variant, d=d)
        res.invariants.append(check_invariants(cfg, tr, d))
        S = tr.recurring.figure_set
        xa = render_interval(cfg, tr.recurring, d)
        xo = render_interval(cfg, tr.other, d)
        ma = measure_interval(cfg, d, tr.recurring, xa, S, span_ms, ref)
        mo = measure_interval(cfg, d, tr.other, xo, S, span_ms, ref)
        res.A.append(ma); res.O.append(mo)
        res.iei_ms.append(np.diff(tr.recurring.element_onsets) * cfg.grid_ms)
        res.locked_diff.append(ma.locked_env - mo.locked_env)
        res.spec_absdiff_mean.append(float(np.mean(np.abs(ma.spectrum_db - mo.spectrum_db))))
        res.spec_absdiff_max.append(float(np.max(np.abs(ma.spectrum_db - mo.spectrum_db))))
        if progress:
            progress(j + 1, n_trials)
    return res


def run_battery(cfg: Config, n_trials: int = 40, seed: int = 2026,
                conditions: Optional[Sequence[Tuple[str, float]]] = None, verbose: bool = True) -> dict:
    d = validate(cfg)
    ref = M.single_tone_reference(cfg, d, WIN_MS, HOP_MS)
    if conditions is None:
        conditions = list(d.main_cells) + list(cfg.control_cells)
    results: List[ConditionResult] = []
    t0 = time.time()
    for (variant, step) in conditions:
        if verbose:
            print(f"  building {n_trials} trials: {variant} step={step:g} ms ...", end="", flush=True)
        t1 = time.time()
        results.append(run_condition(cfg, d, variant, float(step), n_trials, seed, ref))
        if verbose:
            print(f" {time.time() - t1:.0f}s")
    out = {"cfg": cfg, "d": d, "results": results, "n_trials": n_trials, "seed": seed,
           "elapsed_s": time.time() - t0}
    out["observers"] = compute_observers(cfg, results)
    ladders = [lv for lv in cfg.main_variants if any(r.variant == lv for r in results)]
    out["ladders"] = ladders
    out["audit"] = feature_audit(results, "rising")
    out["permutation"] = permutation_audit(results, "rising")
    out["multiplicity"] = observer_multiplicity(out["observers"], results)
    out["pooled_main"] = pooled_main_observers(cfg, results, out["observers"])
    out["by_ladder"] = {
        lv: {"pooled": pooled_main_observers(cfg, results, out["observers"], lv),
             "permutation": permutation_audit(results, lv),
             "multiplicity": observer_multiplicity(out["observers"], results, lv)}
        for lv in ladders
    }
    return out


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


def observer_multiplicity(res_observers: dict, results: List["ConditionResult"],
                          variant: str = "rising") -> dict:
    """The observer grid has many cells. Correct the blind ones for having looked at all of them."""
    cells = []
    for nm, per in res_observers.items():
        if nm.startswith("oracle"):
            continue
        for r in results:
            if r.variant != variant:
                continue
            k = f"{r.variant}:{r.step_ms:g}"
            cells.append(dict(observer=nm, condition=k, dprime=per[k]["cv"]["dprime"], p=per[k]["cv"]["p"],
                              rule_dprime=per[k]["rule"]["dprime"], rule_p=per[k]["rule"]["p"]))
    if not cells:
        return {}
    adj = holm([c["p"] for c in cells])
    for c, a in zip(cells, adj):
        c["p_holm"] = a
    worst = max(cells, key=lambda c: abs(c["dprime"]))
    return dict(variant=variant, n_cells=len(cells), cells=cells, worst=worst,
                any_survive=[c for c in cells if c["p_holm"] < 0.05])


def _pool_cv(entries: List[dict]) -> dict:
    """Pool held-out decisions of several within-condition cross-validations."""
    wins = sum(e["pc"] * e["n"] for e in entries)
    n = sum(e["n"] for e in entries)
    if n == 0:
        return dict(n=0, pc=float("nan"), dprime=float("nan"), ci=(float("nan"), float("nan")), p=float("nan"))
    pc = wins / n
    lo, hi = wilson(wins, n)
    return dict(n=n, pc=pc, dprime=dprime_from_pc(pc, n), ci=(dprime_from_pc(lo, n), dprime_from_pc(hi, n)),
                p=float(stats.binomtest(int(round(wins)), n, 0.5).pvalue))


def compute_observers(cfg: Config, results: List["ConditionResult"]) -> dict:
    """Per condition: fixed rule and within-condition leave-one-out. 'pooled' = every condition's
    held-out decisions pooled (fixed rule pooled directly)."""
    names = list(feature_sets(results[0].A[0], cfg).keys())
    out: Dict[str, Dict[str, dict]] = {nm: {} for nm in names}
    pooled_st: Dict[str, Tuple[list, list]] = {nm: ([], []) for nm in names}
    for r in results:
        key = f"{r.variant}:{r.step_ms:g}"
        for nm in names:
            Ft, Fo, st, so = [], [], [], []
            for ma, mo in zip(r.A, r.O):
                fa, fo = feature_sets(ma, cfg)[nm], feature_sets(mo, cfg)[nm]
                Ft.append(fa[0]); Fo.append(fo[0]); st.append(fa[1]); so.append(fo[1])
            Ft, Fo, st, so = np.array(Ft), np.array(Fo), np.array(st), np.array(so)
            out[nm][key] = {"rule": observer_from_stat(st, so), "cv": observer_cv(Ft, Fo)}
            pooled_st[nm][0].extend(st); pooled_st[nm][1].extend(so)
    for nm in names:
        st, so = (np.array(v) for v in pooled_st[nm])
        out[nm]["pooled"] = {"rule": observer_from_stat(st, so),
                             "cv": _pool_cv([out[nm][k]["cv"] for k in out[nm] if k != "pooled"])}
    return out


def pooled_main_observers(cfg: Config, results: List["ConditionResult"], observers: Optional[dict] = None,
                          variant: str = "rising") -> dict:
    """Each observer over one ladder: leave-one-out WITHIN each condition, held-out decisions pooled,
    Holm over the blind observers. (One rule fitted across conditions with different feature variances
    finds a direction that anti-generalizes; the per-condition fits are calibrated.)"""
    observers = observers if observers is not None else compute_observers(cfg, results)
    out = {}
    for nm, per in observers.items():
        entries = [per[f"{r.variant}:{r.step_ms:g}"]["cv"] for r in results if r.variant == variant]
        if entries:
            out[nm] = _pool_cv(entries)
    blind = [nm for nm in out if not nm.startswith("oracle")]
    for nm, a in zip(blind, holm([out[nm]["p"] for nm in blind])):
        out[nm]["p_holm"] = a
    return out


# ----------------------------------------------------------------------------
# printing
# ----------------------------------------------------------------------------
def _cell(fmt: str, a: float, o: float) -> str:
    return f"{fmt.format(a)}/{fmt.format(o)}"


def _table(header: List[str], rows: List[List[str]], label_w: int = 48, col_w: int = 15) -> str:
    lines = [" " * label_w + "".join(h.rjust(col_w) for h in header)]
    for r in rows:
        lines.append(r[0][:label_w].ljust(label_w) + "".join(c.rjust(col_w) for c in r[1:]))
    return "\n".join(lines)


def format_report(res: dict) -> str:
    cfg: Config = res["cfg"]; d: Derived = res["d"]
    results: List[ConditionResult] = res["results"]
    heads = [f"{r.variant[:5]} {r.step_ms:g}" for r in results]
    L = []
    L.append("=" * 100)
    L.append(f"SeqSFG verification battery: {res['n_trials']} fresh trials per condition, seed {res['seed']}, "
             f"{res['elapsed_s']:.0f} s")
    L.append(f"config {cfg.hash()}; {d.n_channels} channels; cells = interval A (recurring) / interval B (the other)")
    L.append("=" * 100)
    L.append(describe(cfg))

    # invariants
    L.append("\n[1] construction invariants (fraction of trials passing)")
    rows = []
    for key in ("same_n_tones", "same_channel_counts", "budget_exact", "no_same_channel_overlap"):
        rows.append([key] + [f"{np.mean([inv[key] for inv in r.invariants]):.2f}" for r in results])
    rows.append(["shared onset times (fraction)"] + [f"{np.mean([inv['shared_onset_fraction'] for inv in r.invariants]):.3f}" for r in results])
    rows.append(["rebuilds per trial (mean)"] + [f"{np.mean([inv['n_rebuilds'] for inv in r.invariants]):.2f}" for r in results])
    L.append(_table(heads, rows))

    # between intervals
    L.append("\n[2] between the two intervals of a trial: mean over trials, A / B")
    rows = []
    for label, key, fmt in ROWS_BETWEEN:
        rows.append([label] + [_cell(fmt, np.mean([m.scalars[key] for m in r.A]), np.mean([m.scalars[key] for m in r.O])) for r in results])
    L.append(_table(heads, rows))

    L.append("\n[3] paired difference A-B: mean +/- SE over trials  (* = |mean| > 2 SE)")
    rows = []
    for label, key, fmt in ROWS_BETWEEN:
        cells = []
        for r in results:
            diff = np.array([ma.scalars[key] - mo.scalars[key] for ma, mo in zip(r.A, r.O)])
            se = diff.std(ddof=1) / math.sqrt(len(diff)) if len(diff) > 1 else 0.0
            flag = "*" if (se > 0 and abs(diff.mean()) > 2 * se) or (se == 0 and abs(diff.mean()) > 1e-9) else " "
            cells.append(f"{diff.mean():+.3g}±{se:.2g}{flag}")
        rows.append([label] + cells)
    extra = [
        ("spectrum |A-B| per channel: mean (dB)", lambda r: np.mean(r.spec_absdiff_mean)),
        ("spectrum |A-B| per channel: max (dB)", lambda r: np.mean(r.spec_absdiff_max)),
        ("element-locked envelope A-B: rms (dB)", lambda r: np.mean([np.sqrt(np.mean(x ** 2)) for x in r.locked_diff])),
        ("element-locked envelope A-B: max |.| (dB)", lambda r: np.mean([np.max(np.abs(x)) for x in r.locked_diff])),
    ]
    for label, fn in extra:
        rows.append([label] + [f"{fn(r):.3f}" for r in results])
    L.append(_table(heads, rows))

    # across conditions
    L.append("\n[4] across conditions (interval A; B identical by construction where marked =)")
    rows = []
    rows.append(["elements per interval"] + [f"{cfg.n_elements}" for r in results])
    rows.append(["element span (ms)"] + [f"{r.span_ms:.0f}" for r in results])
    for label, fn in (("IEI mean (ms)", np.mean), ("IEI sd (ms)", np.std), ("IEI min (ms)", np.min), ("IEI max (ms)", np.max)):
        rows.append([label] + [f"{fn(np.concatenate(r.iei_ms)):.0f}" for r in results])
    rows.append(["tones per channel (min..max) ="] + [f"{cfg.tones_per_channel}..{cfg.tones_per_channel}" for r in results])
    rows.append(["tones sounding: mean ="] + [f"{np.mean([m.scalars['count_mean'] for m in r.A]):.2f}" for r in results])
    rows.append(["tones sounding: sd"] + [f"{np.mean([m.scalars['count_sd'] for m in r.A]):.2f}" for r in results])
    rows.append(["channel level spread max-min (dB, audio)"] + [f"{np.mean([m.spectrum_db.max() - m.spectrum_db.min() for m in r.A]):.2f}" for r in results])
    rows.append(["channel level sd (dB, audio)"] + [f"{np.mean([m.spectrum_db.std() for m in r.A]):.2f}" for r in results])
    rows.append(["single-channel IOI cv (mean over ch)"] + [f"{np.mean([m.sc_sched['ioi_cv'].mean() for m in r.A]):.3f}" for r in results])
    rows.append(["long-term RMS (dB FS)"] + [f"{np.mean([m.scalars['rms_db'] for m in r.A]):.2f}" for r in results])
    L.append(_table(heads, rows))

    # observers
    obs = res["observers"]
    keys = [f"{r.variant}:{r.step_ms:g}" for r in results] + ["pooled"]
    L.append("\n[5] ideal observers, learnt linear rule (leave-one-out within condition), d' [95% CI]  -- must be at chance")
    rows = []
    for nm, per in obs.items():
        rows.append([nm] + [f"{per[k]['cv']['dprime']:+.2f} [{per[k]['cv']['ci'][0]:+.1f},{per[k]['cv']['ci'][1]:+.1f}]" for k in keys])
    L.append(_table(heads + ["pooled"], rows, col_w=17))
    L.append("    Leave-one-out on null data is biased BELOW chance; a negative d' means no learnable cue. Only a")
    L.append("    positive d' whose CI excludes 0 counts against the design.")
    by_ladder = res.get("by_ladder") or {}
    if by_ladder:
        L.append("\n[5b] the primary claim, PER LADDER: each observer over that ladder's conditions")
        L.append("     (leave-one-out within each condition, held-out decisions pooled; Holm over the blind")
        L.append("     observers). The 'rising' ladder must be clean. Any other ladder's cues are reported here,")
        L.append("     not hidden: they bound what that ladder's psychometric function can be read to mean.")
        for lv, blk in by_ladder.items():
            pooled = blk["pooled"]
            n = next((r["n"] for r in pooled.values()), 0)
            L.append(f"\n     --- ladder '{lv}' ({n} trials) ---")
            for nm, r_ in pooled.items():
                if nm.startswith("oracle"):
                    tag = "  (oracle: expected to succeed)"
                else:
                    flag = "  <-- NOT AT CHANCE" if r_.get("p_holm", 1.0) < 0.05 else ""
                    tag = f"  Holm p = {r_.get('p_holm', float('nan')):.3f}{flag}"
                L.append(f"     {nm:32s} d' = {r_['dprime']:+.3f}  [{r_['ci'][0]:+.2f}, {r_['ci'][1]:+.2f}]{tag}")
            pm_l = blk.get("permutation") or {}
            if pm_l:
                L.append(f"     global permutation over {pm_l['n_features']} features: largest |d'| "
                         f"{pm_l['observed_max_dprime']:.3f} ({pm_l['worst_feature']}), "
                         f"relabelling 95th pct {pm_l['null_max_p95']:.3f}, p = {pm_l['p_value']:.3f}")
            ml = blk.get("multiplicity") or {}
            if ml:
                w = ml["worst"]
                L.append(f"     worst of {ml['n_cells']} observer x condition cells: {w['observer']} at "
                         f"{w['condition']}, d' = {w['dprime']:+.2f}, Holm p = {w['p_holm']:.3f}; "
                         f"{len(ml['any_survive'])} survive correction")

        # the step-by-step cue profile of every non-primary ladder
        others = [lv for lv in by_ladder if lv != "rising"]
        if others:
            obs_all = res["observers"]
            L.append("\n[5d] where a non-primary ladder's cue lives, step by step (learnt observer d')")
            steps = sorted({r.step_ms for r in results if r.variant == "rising"})
            head = [f"{s:g} ms" for s in steps]
            rows = []
            for lv in ["rising"] + others:
                for nm in ("envelope only", "spectrum only", "occupancy only"):
                    cells = []
                    for s in steps:
                        k = f"{lv}:{s:g}"
                        cells.append(f"{obs_all[nm][k]['cv']['dprime']:+.2f}" if k in obs_all[nm] else "-")
                    rows.append([f"{lv}: {nm}"] + cells)
            L.append(_table(head, rows, label_w=34, col_w=9))
            L.append("     A figure that is present in one interval and absent in the other IS a level event, so")
            L.append("     'grouped vs absent' cannot be envelope-matched even in principle: synchronous onsets are")
            L.append("     an envelope transient. The cue shrinks as the components spread. Read that ladder's")
            L.append("     psychometric function against this row, not against chance.")

    L.append("\n[6] ideal observers, fixed a-priori rule (taller spectrum peaks / deeper envelope modulation / peakier")
    L.append("    occupancy / more same-channel onset pairs at element-rate lags, averaged over channels), d' (p vs chance)")
    rows = []
    for nm, per in obs.items():
        cells = []
        for k in keys:
            r_ = per[k]["rule"]
            cells.append("-" if not np.isfinite(r_["dprime"]) else f"{r_['dprime']:+.2f} (p={r_['p']:.2f})")
        rows.append([nm] + cells)
    L.append(_table(heads + ["pooled"], rows, col_w=17))
    L.append("    The oracle row is expected to succeed: it is given the element windows, which the listener is not.")
    audit = res.get("audit", [])
    if audit:
        L.append("\n[6b] feature audit: fixed-rule d' of EVERY scalar feature, pooled over the 'rising' conditions "
                 f"(n={audit[0]['n']}), largest first")
        L.append("     feature                                d'      p        mean A-B     sd of A-B")
        for r in audit[:12]:
            L.append(f"     {r['feature']:36s} {r['dprime']:+.2f}   {r['p']:.3f}   {r['mean_diff']:+.4g}   {r['sd_diff']:.3g}")
    pm = res.get("permutation") or {}
    if pm:
        L.append(f"\n[6c] global permutation test over all {pm['n_features']} features at once "
                 f"({pm['n_perm']} relabellings of the {pm['n_trials']} 'rising' trials)")
        L.append(f"     largest |d'| observed          {pm['observed_max_dprime']:.3f}  ({pm['worst_feature']})")
        L.append(f"     largest |d'| under relabelling  median {pm['null_max_median']:.3f}, 95th percentile "
                 f"{pm['null_max_p95']:.3f}")
        L.append(f"     p (any feature separates the intervals) = {pm['p_value']:.3f}")
        L.append(f"     for reference, a SINGLE pre-chosen feature would clear |d'| = {pm['single_feature_95']:.3f} "
                 f"5% of the time.")

    # level / masking diagnostic
    L.append("\n[7] what limits audibility (per channel): equal amplitude is right when masking >> absolute threshold")
    f = d.channel_freqs_hz
    exc, own = pool_mod.excitation_from_pool(f, cfg.tone_level_db_spl, d.occupancy_per_channel)
    thr = pool_mod.abs_threshold_db_spl(f)
    aw = pool_mod.a_weighting_db(f)
    L.append("   ch   freq(Hz)  abs.thr(dB SPL)  pool excitation(dB)  tone-above-thr  tone-above-excitation  A-weight(dB)")
    for i in range(len(f)):
        L.append(f"   {i:2d}  {f[i]:8.0f}  {thr[i]:14.1f}  {exc[i]:18.1f}  {own[i]-thr[i]:14.1f}  {own[i]-exc[i]:20.1f}  {aw[i]:10.1f}")
    L.append(f"   masking exceeds the absolute threshold by {np.min(exc - thr):.0f}..{np.max(exc - thr):.0f} dB in every channel;")
    L.append(f"   an A-weighting style equal-loudness correction would spread the levels WITHIN one element by up to "
             f"{np.max(aw) - np.min(aw):.1f} dB (worst-case pair), which is why no weighting is applied.")
    L.append("=" * 100)
    return "\n".join(L)


def to_json(res: dict) -> dict:
    out = {"config_hash": res["cfg"].hash(), "n_trials": res["n_trials"], "seed": res["seed"],
           "conditions": [], "observers": {}}
    for r in res["results"]:
        c = {"variant": r.variant, "step_ms": r.step_ms, "span_ms": r.span_ms, "A": {}, "B": {}, "diff": {}}
        for label, key, fmt in ROWS_BETWEEN:
            a = np.array([m.scalars[key] for m in r.A]); o = np.array([m.scalars[key] for m in r.O])
            c["A"][key] = float(a.mean()); c["B"][key] = float(o.mean())
            c["diff"][key] = {"mean": float((a - o).mean()), "se": float((a - o).std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0}
        c["iei_ms"] = {"mean": float(np.mean(np.concatenate(r.iei_ms))), "sd": float(np.std(np.concatenate(r.iei_ms)))}
        out["conditions"].append(c)
    pm = res.get("permutation") or {}
    if pm:
        out["permutation"] = {k: v for k, v in pm.items() if k not in ("null_max", "observed", "names")}
    out["audit"] = [{k: (list(r[k]) if k == "ci" else r[k]) for k in r} for r in res.get("audit", [])]
    mult = res.get("multiplicity") or {}
    if mult:
        out["multiplicity"] = {"n_cells": mult["n_cells"], "worst": mult["worst"],
                               "n_surviving": len(mult["any_survive"]), "cells": mult["cells"]}
    out["ladders"] = res.get("ladders", [])
    out["by_ladder"] = {}
    for lv, blk in (res.get("by_ladder") or {}).items():
        pm_l = blk.get("permutation") or {}
        out["by_ladder"][lv] = {
            "pooled": {k: {"dprime": r["dprime"], "ci": list(r["ci"]), "p": r["p"], "n": r["n"],
                           "p_holm": r.get("p_holm")} for k, r in blk["pooled"].items()},
            "permutation": {k: val for k, val in pm_l.items() if k not in ("null_max", "observed", "names")},
            "multiplicity": {"n_cells": blk["multiplicity"]["n_cells"], "worst": blk["multiplicity"]["worst"],
                             "n_surviving": len(blk["multiplicity"]["any_survive"]),
                             "cells": blk["multiplicity"]["cells"]} if blk.get("multiplicity") else {},
        }
    out["pooled_main"] = {k: {"dprime": r["dprime"], "ci": list(r["ci"]), "p": r["p"], "n": r["n"],
                              "p_holm": r.get("p_holm")} for k, r in (res.get("pooled_main") or {}).items()}
    for nm, per in res["observers"].items():
        out["observers"][nm] = {k: {"cv_dprime": v["cv"]["dprime"], "cv_ci": list(v["cv"]["ci"]), "cv_p": v["cv"]["p"],
                                    "rule_dprime": v["rule"]["dprime"], "rule_p": v["rule"]["p"], "n": v["cv"]["n"]}
                                for k, v in per.items()}
    return out
