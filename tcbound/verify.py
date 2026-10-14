"""Everything that can be checked without a listener, checked.

Grouped by what a reviewer would ask:

  stimulus   Do the two types keep exactly the cue they claim, and nothing else usable? Are the
             repetitions before the probe the same in both? Does the waveform contain the gains
             the log says it does, at the onsets the geometry says, without clipping?
  periphery  Is the target its own component on the cochlea at the worst-case levels?
  procedure  Does the staircase measure what it claims, is the order balanced, does the session
             fit in 40 minutes?
  inference  Does the analysis recover a bound it was given, and stay flat when there is none?
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy import stats

from .config import Condition, Config, conditions, delta_db, onsets, captor_onsets, validate
from .observer import SURVIVING, ceilings, cues, draw_gains
from .stimulus import (build_trial, draw_latent, gains_from_latent, render_interval,
                       render_trial, tone_envelope)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str

    def line(self) -> str:
        return f"  [{'PASS' if self.passed else 'FAIL'}] {self.name}\n         {self.detail}"


# ----------------------------------------------------------------------------
def stimulus_checks(cfg: Config, n: int = 4000, seed: int = 1) -> List[Check]:
    out: List[Check] = []
    conds = {c.name: c for c in conditions(cfg)}
    rng = np.random.default_rng(seed)
    t, oth = cfg.target_index, cfg.others

    # 1. the surviving cue is exact, for every condition
    bad = []
    for c in conds.values():
        for delta in (1.0, 6.0):
            gs = draw_gains(cfg, c, delta, True, 300, rng)
            gn = draw_gains(cfg, c, delta, False, 300, rng)
            cue = SURVIVING[c.task]
            d = cues(cfg, gs)[cue] - cues(cfg, gn)[cue]
            if not np.allclose(d, delta, atol=1e-9):
                bad.append(f"{c.name} {cue} at {delta:g} dB: spread {np.ptp(d):.3g}")
    out.append(Check("the cue each type keeps is exact: signal minus standard = the increment",
                     not bad, "; ".join(bad) or
                     f"{len(conds)} conditions x 2 increments x 300 draws, error < 1e-9 dB"))

    # 2. every other single cue needs a large increment
    worst = []
    for task, kind in sorted({(c.task, c.kind) for c in conds.values()}):
        ce = ceilings(cfg, Condition("x", task, 0.0, kind))
        worst.append((f"{task}/{kind}", ce["lowest_shortcut"], ce["ceiling_db"], ce["exact"]))
    low = min(w[2] for w in worst)
    exp = {"A": {"timbre", "nearest"}, "B": {"own"}}
    unexpected = [w for w in worst if not set(w[3]) >= exp[w[0][0]]
                  or (w[0] != "B/alone" and set(w[3]) - exp[w[0][0]])]
    out.append(Check("no shortcut: every cue other than the kept one needs >= 8 dB",
                     low >= 8.0 and not unexpected,
                     ", ".join(f"{a}: {b} {c:.1f} dB" for a, b, c, _ in worst)
                     + (f"; unexpected exact cues {unexpected}" if unexpected else "")))

    # 3. the repetitions before the probe are drawn identically in A and B
    a = np.stack([gains_from_latent(cfg, "A", "figure", draw_latent(cfg, rng), 0.0, False)[0]
                  for _ in range(n)])
    b = np.stack([gains_from_latent(cfg, "B", "figure", draw_latent(cfg, rng), 0.0, False)[0]
                  for _ in range(n)])
    ks_pre = stats.ks_2samp(a[:, 0, t] - a[:, 0, oth[0]], b[:, 0, t] - b[:, 0, oth[0]]).pvalue
    same_code = np.allclose(a[:, :-1, oth], a[:, :1, oth]) and np.allclose(b[:, :-1, oth],
                                                                          b[:, :1, oth])
    # the probe: the others' jump, and the target's jump relative to them
    jo_a = a[:, -1, oth].mean(1) - a[:, -2, oth].mean(1)
    jo_b = b[:, -1, oth].mean(1) - b[:, -2, oth].mean(1)
    jr_a = (a[:, -1, t] - a[:, -2, t]) - jo_a
    jr_b = (b[:, -1, t] - b[:, -2, t]) - jo_b
    p1, p2 = stats.ks_2samp(jo_a, jo_b).pvalue, stats.ks_2samp(jr_a, jr_b).pvalue
    out.append(Check("A and B are indistinguishable before the probe, and alike at it",
                     ks_pre > 0.01 and same_code and p1 > 0.01 and p2 > 0.01,
                     f"before the probe: one code path, target offset KS p = {ks_pre:.2f}; "
                     f"on the probe: jump of the others KS p = {p1:.2f}, jump of the target "
                     f"relative to them KS p = {p2:.2f} ({n} intervals each)"))

    # 4. the two intervals share timing and phases, and differ only in gain
    bad = []
    for c in conds.values():
        tr = build_trial(cfg, c, 0.0, np.random.default_rng(3))
        if not np.array_equal(tr.first.onsets_ms, tr.second.onsets_ms, equal_nan=True):
            bad.append(f"{c.name}: onsets differ")
        if tr.first.is_signal == tr.second.is_signal:
            bad.append(f"{c.name}: both or neither interval is the signal")
        if tr.first.captor_onsets_ms is not None and not np.array_equal(
                tr.first.captor_onsets_ms, tr.second.captor_onsets_ms, equal_nan=True):
            bad.append(f"{c.name}: captor onsets differ")
    out.append(Check("both intervals of a trial share every onset and phase", not bad,
                     "; ".join(bad) or f"{len(conds)} conditions"))

    # 5. the waveform carries the logged gains, tone by tone
    worst_err = 0.0
    fs, env_n = cfg.sample_rate, int(round(cfg.tone_ms * cfg.sample_rate / 1000))
    for name in ("A_16", "B_64", "A_cap", "B_alone"):
        if name not in conds:
            continue
        tr = build_trial(cfg, conds[name], 2.0, np.random.default_rng(5))
        for iv in tr.intervals():
            for k in range(cfg.n_tones):
                x = render_interval(cfg, iv, tr.phases, tr.captor_phases, tr.total_ms,
                                    only_tone=k, captors=False)
                for r in range(cfg.n_repeats):
                    if not np.isfinite(iv.onsets_ms[r, k]):
                        continue
                    i = int(round(iv.onsets_ms[r, k] * fs / 1000))
                    seg = x[i + env_n // 4: i + 3 * env_n // 4]
                    db = 20 * np.log10(np.sqrt(2 * np.mean(seg ** 2)) / cfg.tone_amplitude)
                    worst_err = max(worst_err, abs(db - iv.gains_db[r, k]))
    out.append(Check("the rendered waveform reads back the gain matrix", worst_err < 0.05,
                     f"every tone of every repetition, four conditions, both intervals: largest "
                     f"error {worst_err:.4f} dB"))

    # 6. onsets on the waveform, and nothing overlaps its own channel
    errs, overlap = [], []
    for c in conds.values():
        o = onsets(cfg, c)
        co = captor_onsets(cfg, c)
        for k in range(cfg.n_tones):
            ts = sorted([v for v in o[:, k] if np.isfinite(v)]
                        + ([v for v in co[:, k] if np.isfinite(v)] if co is not None else []))
            gaps = np.diff(ts) - cfg.tone_ms
            if len(gaps) and gaps.min() < 40.0:
                overlap.append(f"{c.name} tone {k + 1}: {gaps.min():.0f} ms")
        tr = build_trial(cfg, c, 0.0, np.random.default_rng(2))
        x = render_interval(cfg, tr.first, tr.phases, tr.captor_phases, tr.total_ms,
                            only_tone=cfg.target_index)
        # the signal is exactly zero between tones; a tone starts one sample before its first
        # non-zero sample, because the raised-cosine ramp is zero at its first sample
        idx = np.flatnonzero(x != 0.0)
        starts = idx[np.r_[True, np.diff(idx) > fs // 1000]] - 1
        want = [v for v in o[:, cfg.target_index] if np.isfinite(v)]
        got = starts * 1000.0 / fs
        if len(got) == len(want):
            errs.append(float(np.max(np.abs(got - np.asarray(want)))))
        else:
            errs.append(float("inf"))
    out.append(Check("target onsets on the waveform match the geometry; no channel overlaps "
                     "itself", max(errs) < 0.1 and not overlap,
                     f"largest onset error {max(errs):.3f} ms over {len(conds)} conditions; "
                     f"closest same-channel gap {'; '.join(overlap) or 'at least 40 ms'}"))

    # 7. headroom
    d = validate(cfg)
    peak = 0.0
    rng2 = np.random.default_rng(9)
    for _ in range(150):
        for c in (conds["A_0"], conds["B_0"]):
            tr = build_trial(cfg, c, cfg.srs_max_db, rng2)
            peak = max(peak, float(np.abs(render_trial(cfg, tr)).max()))
    out.append(Check("nothing clips", d.peak["worst"] < 0.98 and peak <= d.peak["worst"] + 1e-9,
                     f"worst case {d.peak['worst']:.2f} FS (all four in phase at the top of the "
                     f"rove and the track); largest of 300 rendered trials at the cap "
                     f"{peak:.2f} FS"))

    # 8. ramps: no energy splatter beyond the tone's own band
    env = tone_envelope(cfg)
    f = np.fft.rfftfreq(8192, 1 / cfg.sample_rate)
    s = np.abs(np.fft.rfft(env * np.sin(2 * np.pi * cfg.target_freq_hz
                                        * np.arange(env.size) / cfg.sample_rate), 8192))
    s_db = 20 * np.log10(s / s.max() + 1e-12)
    near = [k for k in cfg.others]
    leak = max(s_db[np.argmin(np.abs(f - cfg.freqs_hz[k]))] for k in near)
    out.append(Check("the gated target puts no energy at the other tones' frequencies",
                     leak < -60, f"spectrum of one gated target tone at the nearest other "
                     f"frequency: {leak:.0f} dB re its peak"))
    return out


# ----------------------------------------------------------------------------
def periphery_checks(cfg: Config) -> List[Check]:
    """Level-dependent roex excitation at the target's place, worst case.

    Glasberg and Moore (1990) ERBs; the lower skirt p_l widens with level as in Moore and
    Glasberg (1987), p_l = p - 0.35 (p / p_1k) (L - 51). The worst case is the target at the
    bottom of its rove and every other tone at the top of its own, which happens on the probe of
    type B: target at nominal - rove, others at nominal + rove + jitter.
    """
    def erb(fc):
        return 24.7 * (4.37 * fc / 1000.0 + 1.0)
    fc = cfg.target_freq_hz
    p = 4.0 * fc / erb(fc)
    p1k = 4.0 * 1000.0 / erb(1000.0)
    lvl_t = cfg.tone_level_db_spl - cfg.rove_db - cfg.interval_rove_db
    lvl_o = cfg.tone_level_db_spl + cfg.rove_db + cfg.flanker_jitter_db + cfg.interval_rove_db
    rows, worst = [], float("inf")
    for k in cfg.others:
        f = cfg.freqs_hz[k]
        g = abs(f - fc) / fc
        pk = max(p - 0.35 * (p / p1k) * (lvl_o - 51.0), 1.0) if f < fc else p
        w = (1 + pk * g) * math.exp(-pk * g)
        exc = lvl_o + 10 * math.log10(max(w, 1e-30))
        margin = lvl_t - exc
        worst = min(worst, margin)
        rows.append(f"{f:.0f} Hz at {lvl_o:.0f} dB SPL -> {exc:.0f} dB at the target's place")
    return [Check("the target dominates its own place on the cochlea at the worst-case levels",
                  worst >= 15.0, f"target at {lvl_t:.0f} dB SPL; " + "; ".join(rows)
                  + f"; smallest margin {worst:.0f} dB")]


# ----------------------------------------------------------------------------
def procedure_checks(cfg: Config, quick: bool = False) -> List[Check]:
    from .design import audit_design, duration_estimate, make_design
    from .track import Track, simulate
    out: List[Check] = []

    # the staircase is the multiplicative tcoh rule on dA/A
    from types import SimpleNamespace
    from tcoh.track import Track as TTrack
    ns = SimpleNamespace(delta_start_ms=10 ** (cfg.srs_start_db / 20),
                         delta_min_ms=10 ** (cfg.srs_min_db / 20),
                         delta_max_ms=10 ** (cfg.srs_max_db / 20), n_down=cfg.n_down,
                         step_factors=tuple(10 ** (s / 20) for s in cfg.step_sizes_db),
                         reversals_per_step=cfg.reversals_per_step,
                         n_final_reversals=cfg.n_final_reversals,
                         max_trials_per_track=cfg.max_trials_per_track)
    rng = np.random.default_rng(0)
    mism = 0
    for _ in range(200):
        a, b = Track(cfg, "x"), TTrack(ns, "x")
        while not a.finished:
            ok = bool(rng.random() < 0.8)
            a.update(ok)
            b.update(ok)
        ra = [x for x, _ in a.reversals]
        rb = [20 * math.log10(x) for x, _ in b.reversals]
        if len(ra) != len(rb) or not np.allclose(ra, rb, atol=1e-9) or \
                a.finished != b.finished:
            mism += 1
    out.append(Check("the SRS staircase is the tcoh rule on dA/A, reversal for reversal",
                     mism == 0, f"200 random response sequences, {mism} mismatches"))

    sim = simulate(cfg, n_runs=120 if quick else 400)
    out.append(Check("the staircase recovers a known threshold",
                     sim["worst_abs_bias_db"] < 2.0 and sim["max_fail_rate"] < 0.05,
                     f"thresholds -12..+4 dB SRS, slopes 0.7-1.5: worst bias "
                     f"{sim['worst_abs_bias_db']:.2f} dB, median spread {sim['median_sd_db']:.2f} "
                     f"dB, median {sim['median_trials']:.0f} trials (90th pct "
                     f"{sim['p90_trials']:.0f}), unconverged {100 * sim['max_fail_rate']:.1f}%"))

    a = audit_design(cfg, make_design(cfg, "VERIFY", 1))
    out.append(Check("every condition is interleaved evenly over the session",
                     a["balanced"] and a["serial_position_spread"] < 0.05
                     and a["longest_condition_run_in_plan"] <= cfg.max_same_condition_run,
                     f"{a['n_tracks']} tracks; serial-position spread "
                     f"{a['serial_position_spread']:.3f}; longest planned run "
                     f"{a['longest_condition_run_in_plan']}; catch {a['catch_rate_planned']:.1%}"))

    e = duration_estimate(cfg, n_sim=4 if quick else 10)
    out.append(Check("the session fits in 40 minutes",
                     e["p90_total_minutes"] <= 40.0,
                     f"median {e['total_minutes']:.1f} min, 90th percentile "
                     f"{e['p90_total_minutes']:.1f} min over {e['n_sim']} simulated sessions "
                     f"({e['n_trials_median']:.0f} trials incl. {e['n_catch_median']:.0f} catch "
                     f"and {e['n_filler_median']:.0f} fillers); "
                     f"{e['worst_practice_total_minutes']:.1f} min if practice needs three "
                     "rounds"))
    return out


# ----------------------------------------------------------------------------
def inference_checks(cfg: Config, quick: bool = False) -> List[Check]:
    """Simulated groups through the real runner and the real analysis."""
    import tempfile
    from pathlib import Path
    from .analysis import group
    from .runner import Runner
    from .model import half_way_ms
    out: List[Check] = []
    n_list = 6 if quick else 10
    res = {}
    for mode in ("coherence", "null"):
        with tempfile.TemporaryDirectory() as tmp:
            dirs = []
            for i in range(n_list):
                code = f"V{i:02d}"
                r = Runner(cfg, Path(tmp), auto=mode, seed=100 * i + 1)
                import contextlib, io
                with contextlib.redirect_stdout(io.StringIO()):
                    dirs.append(r.run(code=code))
            res[mode] = group(dirs, n_boot=400)
    g = res["coherence"]
    truth = half_way_ms(cfg)
    ok_b = g["s50_ms"] is not None and abs(g["s50_ms"] - truth) < 6.0
    out.append(Check("a listener whose binding follows the model gives the model's bound",
                     g["H1"]["ok"] and g["H1"]["p"] < 0.01 and g["H2"]["p"] < 0.01 and ok_b,
                     f"{n_list} simulated listeners, one session each: H1 p = {g['H1']['p']:.1g}, "
                     f"H2 p = {g['H2']['p']:.1g}, bound {g['s50_ms'] if g['s50_ms'] is None else round(g['s50_ms'], 1)} ms "
                     f"against a true {truth:.1f} ms; alpha {g['fits']['halfwave']['alpha']:.2f}"))
    cap = g["H3"]
    out.append(Check("captors at synchrony lower the timbre weight, as the model says",
                     cap["ok"] and cap["p"] < 0.01,
                     f"w(capture) - w(0) = {cap['mean']:+.2f}, p = {cap['p']:.1g}; predicted "
                     f"w {g['capture_prediction']['predicted_w']:.2f}, observed "
                     f"{g['capture_prediction']['observed_w']:.2f}"))
    n = res["null"]
    out.append(Check("a listener for whom shear changes nothing gives no effect",
                     n["H1"]["p"] > 0.01 and n["H2"]["p"] > 0.01,
                     f"H1 p = {n['H1']['p']:.2f}, H2 p = {n['H2']['p']:.2f}, bound "
                     f"{'undefined' if n['s50_ms'] is None else round(n['s50_ms'], 1)}"))
    return out


def model_checks(cfg: Config) -> List[Check]:
    from .model import predictions
    p = predictions(cfg)
    out = []
    hw, mg = p["variants"]["halfwave"], p["variants"]["magnitude"]
    mono = all(np.all(np.diff(v["curve"]) < 0) for v in (hw, mg))
    out.append(Check("rho_T is 1 at step 0 and falls monotonically, under both readings",
                     abs(hw["curve"][0] - 1) < 1e-6 and mono,
                     "half-wave " + " ".join(f"{x:.2f}" for x in hw["curve"])
                     + "; magnitude " + " ".join(f"{x:.2f}" for x in mg["curve"])))
    sep = abs(mg["half_way_ms"] - hw["half_way_ms"])
    out.append(Check("the two readings predict bounds the steps can tell apart",
                     sep > 2 * (sorted(cfg.steps_ms)[2] - sorted(cfg.steps_ms)[1]) / 2,
                     f"half-way {hw['half_way_ms']:.1f} ms against {mg['half_way_ms']:.1f} ms"))
    out.append(Check("the capture control lowers rho_T without asynchrony",
                     hw["capture_rho"] < 0.9,
                     f"rho_T {hw['capture_rho']:.2f}, the value a shear of "
                     f"{hw['capture_equivalent_ms']:.1f} ms gives (half-wave reading)"))
    return out


def run_battery(cfg: Config, quick: bool = False) -> Dict[str, List[Check]]:
    return {"stimulus": stimulus_checks(cfg), "periphery": periphery_checks(cfg),
            "model": model_checks(cfg), "procedure": procedure_checks(cfg, quick),
            "inference": inference_checks(cfg, quick)}


def format_report(cfg: Config, battery: Dict[str, List[Check]]) -> str:
    n = sum(len(v) for v in battery.values())
    ok = sum(c.passed for v in battery.values() for c in v)
    L = [f"tcbound verification  |  config {cfg.hash()}  |  {ok}/{n} checks pass", ""]
    for section, checks in battery.items():
        L.append(f"{section}")
        L += [c.line() for c in checks]
        L.append("")
    return "\n".join(L)
