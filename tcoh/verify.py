"""The verification battery: everything that can be checked without a listener, checked.

The battery is organised around the claims the experiment makes, not around the code that
implements them, so that a reader who distrusts a claim can find the number that supports it.

The headline claim, and why it is unusually strong here
--------------------------------------------------------
In every condition:

  * the A channel is IDENTICAL between the standard and the target interval -- A never moves --
    so A on its own carries no information at all about which interval is which;
  * the B channel is IDENTICAL across conditions -- same grid, same frequency, same level, same
    ramps -- so whatever information B carries on its own is the same at every dT.

Put together: the information available in either channel ALONE is the same in every condition,
and the only thing that varies with dT is the RELATION between them. A single-channel account
of any dT effect is therefore ruled out by construction rather than by argument. `invariants`
checks both halves by rendering the channels separately and comparing samples, not summaries.

What the battery covers
------------------------
  invariants          the four properties `tcoh.stimulus` claims, sample by sample
  peripheral_audit    a gammatone bank across the spectrum: does shifting B change anything in
                      any auditory filter other than B's own?
  model_checks        reproduce the published Figure 8 values; confirm dT is monotone at this
                      duty cycle; confirm the jittered control really does destroy coherence
  procedure_checks    the staircase recovers a known threshold; the design is balanced; the
                      tests are calibrated
  level_check         peak amplitude, headroom, and what the two tones together come to at the ear

`run_battery` runs all of it and `format_report` prints it with a PASS/FAIL per claim.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Condition, Config, validate
from .stimulus import (Interval, build_trial, render_interval, render_trial, tone_envelope)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    value: object = None

    def line(self) -> str:
        return f"  [{'PASS' if self.passed else 'FAIL'}] {self.name}\n         {self.detail}"


def _render_channel(cfg: Config, iv: Interval, d, which: str,
                    phases=None) -> np.ndarray:
    """Render one channel of an interval on its own, so the two can be compared separately."""
    from .config import interval_ms
    total = interval_ms(cfg, iv.b_onsets_ms.size - 1)
    stripped = Interval(iv.a_onsets_ms if which == "a" else np.zeros(0),
                        iv.b_onsets_ms if which == "b" else np.zeros(0),
                        iv.delta_ms, 0.0, iv.a_kind, iv.lag_pct)
    return render_interval(cfg, stripped, d, phases, total_ms=total)


# ----------------------------------------------------------------------------
def invariants(cfg: Config, delta_ms: float = 8.0, seed: int = 4) -> List[Check]:
    d = validate(cfg)
    conds = d.conditions
    out: List[Check] = []

    # -- 1. within a trial, only the last B tone moves ---------------------------
    worst = 0.0
    bad = []
    for c in conds:
        tr = build_trial(cfg, c, delta_ms, np.random.default_rng(seed), target_position=1,
                         direction=+1)
        tgt, std = tr.first, tr.second
        if not np.array_equal(std.a_onsets_ms, tgt.a_onsets_ms):
            bad.append(f"{c.name}: A onsets differ")
        if not np.array_equal(std.b_onsets_ms[:-1], tgt.b_onsets_ms[:-1]):
            bad.append(f"{c.name}: B precursors differ")
        worst = max(worst, abs((tgt.b_onsets_ms[-1] - std.b_onsets_ms[-1]) - delta_ms))
    out.append(Check("only the final B tone differs between the two intervals", not bad,
                     "; ".join(bad) if bad else
                     f"checked {len(conds)} conditions: A onsets equal, B precursors equal, the "
                     f"final B tone displaced by exactly delta (largest error "
                     f"{worst:.2e} ms)"))

    # -- 2. A alone carries no information ---------------------------------------
    diffs = []
    for c in conds:
        if c.a_kind == "absent":
            continue
        tr = build_trial(cfg, c, delta_ms, np.random.default_rng(seed), target_position=1)
        a1 = _render_channel(cfg, tr.first, d, "a", tr.phases)
        a2 = _render_channel(cfg, tr.second, d, "a", tr.phases)
        diffs.append(float(np.max(np.abs(a1 - a2))))
    out.append(Check("the A channel is bit-identical between standard and target",
                     bool(diffs) and max(diffs) == 0.0,
                     f"max |difference| over {len(diffs)} conditions = {max(diffs) if diffs else float('nan'):.3e} "
                     "(so no observer listening only to the low tone can do the task at all)"))

    # -- 3. B is identical ACROSS conditions of the same length ----------------------
    #
    # A build-up condition deliberately has more precursors, so its B sequence is longer and
    # cannot be identical to the others -- and that is worth stating rather than passing over,
    # because it means the build-up comparison is NOT protected by this invariant and needs its
    # own ceiling (which `conditions` adds, and `analysis.buildup_test` scores against).
    refs, diffs, groups = {}, [], set()
    for c in conds:
        n = cfg.n_precursor if c.n_precursor is None else c.n_precursor
        groups.add(n)
        tr = build_trial(cfg, c, delta_ms, np.random.default_rng(seed), target_position=1,
                         direction=+1)
        pair = (_render_channel(cfg, tr.second, d, "b", tr.phases),
                _render_channel(cfg, tr.first, d, "b", tr.phases))
        if n not in refs:
            refs[n] = pair
        else:
            diffs.append(max(float(np.max(np.abs(pair[0] - refs[n][0]))),
                             float(np.max(np.abs(pair[1] - refs[n][1])))))
    extra = ("" if len(groups) == 1 else
             f"  Precursor lengths {sorted(groups)} are compared separately: a build-up condition "
             "has a longer B sequence by design, so its comparison rests on its own ceiling "
             "(b_only_long) rather than on this invariant.")
    out.append(Check("the B channel is bit-identical across conditions of the same length",
                     bool(diffs) and max(diffs) == 0.0,
                     f"max |difference| over {len(diffs)} comparisons = "
                     f"{max(diffs) if diffs else float('nan'):.3e} (so the cue available from the "
                     "high tone alone cannot vary with dT)." + extra))

    # -- 4. energy and duration -----------------------------------------------------
    lens, es = set(), []
    for c in conds:
        tr = build_trial(cfg, c, delta_ms, np.random.default_rng(seed), target_position=1)
        for iv in (tr.first, tr.second):
            x = render_interval(cfg, Interval(iv.a_onsets_ms, iv.b_onsets_ms, iv.delta_ms, 0.0,
                                              iv.a_kind, iv.lag_pct), d, tr.phases)
            lens.add((c.name, x.size))
            es.append(float(np.sum(x ** 2)))
    tr = build_trial(cfg, conds[0], delta_ms, np.random.default_rng(seed), target_position=1)
    e1 = np.sum(render_interval(cfg, Interval(tr.first.a_onsets_ms, tr.first.b_onsets_ms,
                                              tr.first.delta_ms, 0.0, tr.first.a_kind,
                                              tr.first.lag_pct), d, tr.phases) ** 2)
    e2 = np.sum(render_interval(cfg, Interval(tr.second.a_onsets_ms, tr.second.b_onsets_ms,
                                              tr.second.delta_ms, 0.0, tr.second.a_kind,
                                              tr.second.lag_pct), d, tr.phases) ** 2)
    rel = abs(e1 - e2) / max(e1, e2)
    from .config import interval_ms as _ivms
    n_lengths = len({(cfg.n_precursor if c.n_precursor is None else c.n_precursor) for c in conds})
    per_cond = {}
    for name, n in lens:
        per_cond.setdefault(name, set()).add(n)
    sizes = sorted({n for _, n in lens})
    out.append(Check("standard and target always last exactly as long as each other",
                     all(len(v) == 1 for v in per_cond.values()) and len(sizes) == n_lengths,
                     f"{len(sizes)} distinct interval length(s) across {len(conds)} conditions, for "
                     f"{n_lengths} distinct precursor count(s): {sizes} samples. Duration is "
                     "constant WITHIN a trial, which is what stops it being a cue; it differs "
                     "between conditions of different length, which it must, and which the "
                     "listener cannot use to answer a within-trial question."))
    out.append(Check("shifting the tone does not change the energy", rel < 1e-6,
                     f"standard vs target total energy differs by {100 * rel:.2e}%, i.e. by "
                     f"{10 * math.log10(1 + rel):.1e} dB. The residual is the starting phase of the "
                     "displaced tone landing on a different sample, not a level difference any "
                     "listener or meter could resolve. The level rove, which is a deliberate "
                     "randomisation, is excluded here."))

    # -- 5. no collisions within a channel ------------------------------------------
    worst_gap = np.inf
    coll = []
    for c in conds:
        for dd in (-cfg.delta_max_ms, 0.0, cfg.delta_max_ms):
            tr = build_trial(cfg, c, abs(dd) or cfg.delta_min_ms,
                             np.random.default_rng(seed), target_position=1,
                             direction=-1 if dd < 0 else +1)
            for iv in (tr.first, tr.second):
                for ons in (iv.a_onsets_ms, iv.b_onsets_ms):
                    if ons.size < 2:
                        continue
                    gaps = np.diff(np.sort(ons)) - cfg.tone_ms
                    worst_gap = min(worst_gap, float(gaps.min()))
                    if gaps.min() < -1e-9:
                        coll.append(f"{c.name} at delta={dd:+g}")
    out.append(Check("two tones in one channel never overlap", not coll,
                     "; ".join(sorted(set(coll))) if coll else
                     f"smallest silent gap within a channel across every condition and the "
                     f"extremes of delta: {worst_gap:.2f} ms"))

    # -- 6. the shift survives sample quantisation ------------------------------------
    errs = []
    for dd in np.geomspace(cfg.delta_min_ms, cfg.delta_max_ms, 40):
        tr = build_trial(cfg, conds[0], dd, np.random.default_rng(seed), target_position=1,
                         direction=+1)
        errs.append(abs(abs(tr.delta_realised_ms) - dd))
    q = 1000.0 / cfg.sample_rate
    out.append(Check("the realised shift matches the requested one", max(errs) <= q,
                     f"largest deviation {max(errs) * 1000:.1f} us over 40 values from "
                     f"{cfg.delta_min_ms:g} to {cfg.delta_max_ms:g} ms; the sample grid is "
                     f"{q * 1000:.1f} us. The analysis uses the realised value."))

    # -- 7. the level rove says nothing --------------------------------------------
    rng = np.random.default_rng(99)
    gaps = []
    for _ in range(4000):
        tr = build_trial(cfg, conds[0], 8.0, rng)
        t = tr.target_position
        gaps.append((tr.first.level_db if t == 1 else tr.second.level_db)
                    - (tr.second.level_db if t == 1 else tr.first.level_db))
    frac = float(np.mean(np.asarray(gaps) > 0))
    out.append(Check("the level rove carries no information about the answer",
                     abs(frac - 0.5) < 0.03,
                     f"the target interval was the louder one on {frac:.1%} of 4000 trials "
                     f"(mean gap {np.mean(gaps):+.3f} dB); 50% is uninformative"))
    return out


# ----------------------------------------------------------------------------
def _gammatone(fc: float, fs: float, n: int = 2048, order: int = 4) -> np.ndarray:
    """A 4th-order gammatone impulse response at fc, Glasberg & Moore bandwidth."""
    erb = 24.7 * (4.37 * fc / 1000.0 + 1.0)
    b = 1.019 * erb
    t = np.arange(n) / fs
    h = t ** (order - 1) * np.exp(-2 * np.pi * b * t) * np.cos(2 * np.pi * fc * t)
    return h / np.sqrt(np.sum(h ** 2))


def peripheral_audit(cfg: Config, delta_ms: float = 8.0, n_filters: int = 28,
                     seed: int = 4) -> List[Check]:
    """Does displacing B change anything in any auditory filter that is not B's own?

    The two-channel reduction the model prediction rests on assumes the tones do not share
    filters. This tests it on the actual waveform: pass the standard and target intervals
    through a gammatone bank spanning the spectrum, and report the largest envelope change in
    each filter. If a filter near A moves when B moves, there is a peripheral cue the design
    does not control and the reduction is wrong.
    """
    d = validate(cfg)
    fs = cfg.sample_rate
    lo, hi = cfg.f_a_hz / 2.0, d.f_b_hz * 2.0
    fcs = np.geomspace(lo, hi, n_filters)
    c = next(x for x in d.conditions if x.a_kind == "coherent")
    tr = build_trial(cfg, c, delta_ms, np.random.default_rng(seed), target_position=1, direction=+1)
    x1 = render_interval(cfg, Interval(tr.first.a_onsets_ms, tr.first.b_onsets_ms,
                                       tr.first.delta_ms, 0.0, c.a_kind, c.lag_pct), d, tr.phases)
    x2 = render_interval(cfg, Interval(tr.second.a_onsets_ms, tr.second.b_onsets_ms,
                                       tr.second.delta_ms, 0.0, c.a_kind, c.lag_pct), d, tr.phases)

    from scipy.signal import fftconvolve, hilbert
    rel = []
    for fc in fcs:
        h = _gammatone(fc, fs)
        e1 = np.abs(hilbert(fftconvolve(x1, h)[: x1.size]))
        e2 = np.abs(hilbert(fftconvolve(x2, h)[: x2.size]))
        ref = max(float(e1.max()), 1e-12)
        rel.append(float(np.max(np.abs(e1 - e2)) / ref))
    rel = np.asarray(rel)

    def erbn(f):
        return 21.4 * math.log10(4.37 * f / 1000.0 + 1.0)

    # How far from B does the shift still move anything? A gammatone centred near B must of
    # course respond -- that IS the signal. The question is whether the disturbance reaches A.
    e_fcs = np.array([erbn(f) for f in fcs])
    e_b, e_a = erbn(d.f_b_hz), erbn(cfg.f_a_hz)
    moved = rel > 0.01
    reach = float(np.max(np.abs(e_fcs[moved] - e_b))) if moved.any() else 0.0
    sep = abs(e_b - e_a)
    near_a = np.abs(e_fcs - e_a) <= 1.0
    worst_a = float(rel[near_a].max()) if near_a.any() else float("nan")
    mid = np.argmin(np.abs(e_fcs - (e_a + e_b) / 2.0))
    return [
        Check("displacing B leaves A's auditory filters alone", worst_a < 0.01,
              f"largest envelope change within one ERB of {cfg.f_a_hz:.0f} Hz: "
              f"{100 * worst_a:.3f}% of that filter's peak; at the midpoint between the tones "
              f"({fcs[mid]:.0f} Hz) it is {100 * rel[mid]:.3f}%"),
        Check("the disturbance stays inside B's own auditory region", reach < sep,
              f"filters more than {reach:.1f} ERB from B are unmoved (change under 1%); A is "
              f"{sep:.1f} ERB away, so the disturbance falls short of it by {sep - reach:.1f} ERB. "
              f"Bank of {n_filters} gammatones from {lo:.0f} to {hi:.0f} Hz. This is what makes "
              "the one-channel-per-tone reduction the model prediction rests on safe here."),
    ]


# ----------------------------------------------------------------------------
def model_checks(cfg: Config) -> List[Check]:
    from .model import (coherence_matrix, duty_cycle_scan, PUBLISHED, prediction_band,
                        reproduce_figure8, sequence_envelope)
    d = validate(cfg)
    out: List[Check] = []

    f8 = reproduce_figure8()
    hw = f8["readings"]["halfwave"]
    ok = (abs(hw["alternating"] - PUBLISHED["alternating"]) < 0.1
          and abs(hw["synchronous"] - PUBLISHED["synchronous"]) < 0.05 and hw["monotone"])
    out.append(Check("the model implementation reproduces the published Figure 8", ok,
                     f"alternating {hw['alternating']:.3f} (paper {PUBLISHED['alternating']}), "
                     f"synchronous {hw['synchronous']:.3f} (paper {PUBLISHED['synchronous']}), "
                     f"monotone {hw['monotone']}. The literal signed-product reading gives "
                     f"{f8['readings']['none']['alternating']:.3f} and is not monotone; see the "
                     "note in tcoh/model.py."))

    scan = duty_cycle_scan(soa_ms=cfg.soa_ms, n_tones=cfg.n_tones)
    here = scan.get(round(cfg.duty, 6)) or duty_cycle_scan((cfg.duty,), soa_ms=cfg.soa_ms,
                                                           n_tones=cfg.n_tones)[cfg.duty]
    out.append(Check("dT is a monotone axis at this duty cycle", here["monotone"],
                     f"duty {cfg.duty:.3f}: predicted index is "
                     f"{'monotone in dT' if here['monotone'] else 'NOT monotone, peaking at dT='
                        + format(here['argmax_pct'], '.0f') + '%'}. "
                     + "  ".join(f"{k:.3f}:{'mono' if v['monotone'] else 'non-mono'}"
                                 for k, v in scan.items())))

    pcts = sorted({c.lag_pct for c in d.conditions if c.a_kind == "coherent"})
    band = prediction_band(pcts, tone_ms=cfg.tone_ms, soa_ms=cfg.soa_ms, n_tones=cfg.n_tones)
    out.append(Check("the prediction survives every defensible reading of the filter bank",
                     band["all_monotone"] and band["all_agree_on_order"],
                     f"{len(band['curves'])} variants: all monotone {band['all_monotone']}, all "
                     f"agreeing on the ordering of dT levels {band['all_agree_on_order']}, "
                     f"largest disagreement in height {band['max_spread']:.2f}. The ORDER is "
                     "predicted robustly; the heights are not, which is why the shape test is "
                     "reported with a caveat."))

    # Do the controls predict the interaction H2 is built to detect?
    #
    # The requirement is NOT that a control's own coherence be flat -- it need not be, and for
    # the scrambled one it is not. The requirement is that the model's predicted control-minus-
    # coherent difference be positive where the tones are synchronous and FALL as dT grows,
    # because that declining difference is the entire content of H2. A control for which the
    # model predicts no decline would be one that cannot decide anything, however well matched
    # it is on paper.
    mbc = d.model_by_condition
    coh = {c.lag_pct: mbc[c.name] for c in d.conditions if c.a_kind == "coherent"}
    for kind in ("scrambled", "pair_only"):
        ctl = {c.lag_pct: mbc[c.name] for c in d.conditions if c.a_kind == kind}
        shared = sorted(p for p in ctl if p in coh)
        if len(shared) < 2:
            continue
        diff = np.array([ctl[p] - coh[p] for p in shared])
        slope = float(np.polyfit(np.array(shared, dtype=float), diff, 1)[0]) * 100.0
        out.append(Check(f"the {kind} control predicts a falling difference (this is what H2 tests)",
                         bool(diff[0] > 0.15 and slope < -0.1),
                         "model index, control minus coherent: "
                         + ", ".join(f"dT={p:g}%: {x:+.2f}" for p, x in zip(shared, diff))
                         + f"; slope {slope:+.2f} per 100% dT. Positive at synchrony and falling is "
                           "the signature. A pedestal or local-overlap account predicts zero at "
                           "every dT -- a different shape, which is what makes them separable."))

    out.append(Check("the two tones do not share auditory filters", d.erbs_apart >= 3.0,
                     f"{d.erbs_apart:.1f} ERBs apart ({cfg.df_semitones:g} semitones); "
                     f"cross-channel leak {'<' if d.roex_leak_is_floor else ''}"
                     f"-{d.roex_attenuation_db:.0f} dB"))
    out.append(Check("the two tones are not in a simple frequency ratio",
                     d.harmonic["mistuning_pct"] >= 3.0,
                     f"f_b/f_a = {d.harmonic['frequency_ratio']:.4f}, nearest ratio with terms up "
                     f"to {d.harmonic['max_term']} is {d.harmonic['ratio']}, "
                     f"{d.harmonic['mistuning_pct']:.1f}% away. Closer than about 3% and "
                     "harmonicity would fuse the pair for reasons this design does not control."))
    return out


# ----------------------------------------------------------------------------
def procedure_checks(cfg: Config, quick: bool = True) -> List[Check]:
    from .analysis import power
    from .design import audit_design, make_design
    from .track import simulate
    out: List[Check] = []

    sim = simulate(cfg, n_runs=120 if quick else 400, seed=5)
    out.append(Check("the staircase recovers a threshold it was given",
                     sim["worst_abs_bias_pct"] < 20.0 and sim["max_fail_rate"] < 0.05,
                     f"worst bias {sim['worst_abs_bias_pct']:+.1f}% across thresholds from 2 to "
                     f"24 ms and slopes from 0.35 to 0.9; single-track spread sd(log) = "
                     f"{sim['median_sd_log']:.2f}; median {sim['median_trials']:.0f} trials; "
                     f"non-convergence {sim['max_fail_rate']:.1%}. The bias is a slight "
                     "UNDERestimate at every threshold, so it largely cancels in the log ratios "
                     "kappa is built from."))

    des = make_design(cfg, "VERIFY", 1)
    a = audit_design(cfg, des)
    out.append(Check("no condition is confounded with time in the session",
                     a["round_balance_spread"] < 1e-9 and a["serial_position_spread"] < 0.10
                     and a["balanced_track_count"],
                     f"every condition contributes exactly one track per round (mean round index "
                     f"varies by {a['round_balance_spread']:.1e}), so each gets one track in each "
                     f"{1 / cfg.tracks_per_condition:.0%} of the session by construction. Within "
                     f"that, mean serial position still varies by "
                     f"{a['serial_position_spread']:.3f} of the session across "
                     f"{len(a['tracks_per_condition'])} conditions -- a residue of blocks that do "
                     f"not divide evenly. A slow drift across the session therefore cannot line up "
                     f"with dT; `analysis.diagnostics` measures the drift itself as well."))
    out.append(Check("no condition repeats more than the configured run length",
                     a["run_constraint_respected"],
                     f"longest planned run {a['longest_condition_run']}, limit "
                     f"{a['max_same_condition_run']}; catch trials planned at "
                     f"{a['catch_rate_planned']:.1%} against {a['catch_rate_configured']:.1%}"))

    if not quick:
        p = power(cfg, n_sessions=60, n_boot=400, seed=7)
        m = p["modes"]
        out.append(Check("the diagnostic test is calibrated and has power",
                         m["coherence"]["H2_rate"] >= 0.8 and m["pedestal"]["H2_rate"] <= 0.10
                         and m["null"]["H2_rate"] <= 0.10,
                         f"H2 fires on {m['coherence']['H2_rate']:.0%} of simulated sessions when "
                         f"coherence is true, {m['pedestal']['H2_rate']:.0%} when the pedestal "
                         f"rival is true, {m['null']['H2_rate']:.0%} under the null. H1 fires "
                         f"{m['pedestal']['H1_rate']:.0%} under the pedestal rival, which is why "
                         "H1 alone settles nothing."))
    return out


def level_check(cfg: Config) -> List[Check]:
    d = validate(cfg)
    c = d.conditions[0]
    tr = build_trial(cfg, c, cfg.delta_max_ms, np.random.default_rng(1))
    x = render_trial(cfg, tr, d)
    peak = float(np.max(np.abs(x)))
    one_tone = cfg.tone_level_db_spl
    both = one_tone + 10 * math.log10(2)
    headroom_db = 20 * math.log10(1.0 / max(peak, 1e-12))
    return [
        Check("the waveform cannot clip", peak < 0.95,
              f"peak {peak:.3f} FS with the rove at its maximum ({headroom_db:.1f} dB of headroom)"),
        Check("the level at the ear is comfortable and safe", 45.0 <= both <= 80.0,
              f"one tone at {one_tone:.0f} dB SPL as calibrated; both together at most "
              f"{both:.1f} dB SPL. A whole session is around "
              f"{d.est_minutes:.0f} minutes, which at this level is far below any exposure limit."),
        Check("both tones sit in a comfortable frequency range",
              120.0 <= cfg.f_a_hz and d.f_b_hz <= 6000.0,
              f"A {cfg.f_a_hz:.0f} Hz, B {d.f_b_hz:.0f} Hz -- both well inside the range where "
              "hearing is most sensitive and neither is shrill"),
    ]


# ----------------------------------------------------------------------------
def run_battery(cfg: Config, quick: bool = True) -> Dict[str, List[Check]]:
    return {"stimulus invariants": invariants(cfg),
            "peripheral audit": peripheral_audit(cfg),
            "model": model_checks(cfg),
            "procedure": procedure_checks(cfg, quick=quick),
            "level and safety": level_check(cfg)}


def format_report(cfg: Config, battery: Dict[str, List[Check]]) -> str:
    d = validate(cfg)
    L = ["=" * 78, f"tcoh verification battery   |   config {cfg.hash()}",
         f"A {cfg.f_a_hz:.0f} Hz / B {d.f_b_hz:.0f} Hz, {cfg.tone_ms:g} ms tones, "
         f"{cfg.soa_ms:g} ms SOA, {cfg.n_precursor} precursors, {len(d.conditions)} conditions",
         "=" * 78]
    n_fail = 0
    for section, checks in battery.items():
        L.append("")
        L.append(f"-- {section} " + "-" * max(2, 72 - len(section)))
        for c in checks:
            L.append(c.line())
            n_fail += not c.passed
    L.append("")
    L.append("=" * 78)
    total = sum(len(v) for v in battery.values())
    L.append(f"{total - n_fail}/{total} checks passed"
             + ("" if n_fail == 0 else f"   -- {n_fail} FAILED, see above"))
    L.append("=" * 78)
    return "\n".join(L)
