"""Everything that can be checked without a listener, checked.

The headline claim
-------------------
In every condition the two intervals of a trial are identical except for the position of ONE
tone in ONE repetition, and every channel other than the target's is bit-identical between
them. So no observer listening to anything but the target tone can do the task at all, and the
only thing that varies with the step is the RELATION between the target and the rest.

Separately, and this is what the design rests on: the target's OWN channel is isochronous at
`rate_hz` at every step. `within_channel_reference_is_invariant` checks that on the rendered
waveform, because it is the reason a flat curve means 'the listener ignored the figure' rather
than 'the manipulation did nothing'.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .config import Condition, Config, combined_gaps_ms, is_isochronous, validate
from .stimulus import (Interval, build_trial, figure_onsets, render_interval, render_trial,
                       tone_envelope)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str

    def line(self) -> str:
        return f"  [{'PASS' if self.passed else 'FAIL'}] {self.name}\n         {self.detail}"


def invariants(cfg: Config, delta_ms: float = 8.0, seed: int = 4) -> List[Check]:
    d = validate(cfg)
    conds = d.conditions
    out: List[Check] = []
    k = cfg.target_index

    # 1. only the target tone of the last repetition moves
    bad, worst = [], 0.0
    for c in conds:
        tr = build_trial(cfg, c, delta_ms, np.random.default_rng(seed), target_position=1,
                         direction=+1)
        a, b = tr.first.onsets_ms, tr.second.onsets_ms
        diff = np.nan_to_num(a - b, nan=0.0)
        moved = np.argwhere(np.abs(diff) > 1e-12)
        if len(moved) != 1 or tuple(moved[0]) != (cfg.n_repeats - 1, k):
            bad.append(f"{c.name}: {len(moved)} onsets differ at {moved.tolist()}")
        worst = max(worst, abs(diff[cfg.n_repeats - 1, k] - delta_ms))
    out.append(Check("only the target tone of the last repetition differs between the intervals",
                     not bad,
                     "; ".join(bad) if bad else
                     f"checked {len(conds)} conditions: exactly one onset differs, by delta "
                     f"(largest error {worst:.2e} ms)"))

    # 2. every OTHER channel is bit-identical between standard and target
    diffs = []
    for c in conds:
        tr = build_trial(cfg, c, delta_ms, np.random.default_rng(seed), target_position=1)
        for j in range(cfg.n_tones):
            if j == k:
                continue
            # zero the level rove for the comparison: it is a deliberate, logged randomisation
            # that differs between the intervals by design, and leaving it in would make this
            # check fail on the rove rather than test what it claims to test
            a = Interval(tr.first.onsets_ms, tr.first.delta_ms, 0.0, tr.first.step_pct,
                         tr.first.kind)
            b = Interval(tr.second.onsets_ms, tr.second.delta_ms, 0.0, tr.second.step_pct,
                         tr.second.kind)
            x1 = render_interval(cfg, a, tr.phases, only_tone=j)
            x2 = render_interval(cfg, b, tr.phases, only_tone=j)
            diffs.append(float(np.max(np.abs(x1 - x2))))
    out.append(Check("every channel except the target's is bit-identical between the intervals",
                     bool(diffs) and max(diffs) == 0.0,
                     f"max |difference| over {len(diffs)} channel comparisons = "
                     f"{max(diffs) if diffs else float('nan'):.3e} -- an observer listening to "
                     "anything but the target tone cannot do this task"))

    # 3. the target's own channel is isochronous at rate_hz at EVERY step
    bad = []
    for c in conds:
        o = figure_onsets(cfg, c)[:, k]
        gaps = np.diff(o)
        if not np.allclose(gaps, cfg.period_ms):
            bad.append(f"{c.name}: gaps {np.round(gaps, 3).tolist()}")
    out.append(Check("the target's own channel is isochronous at every step", not bad,
                     "; ".join(bad) if bad else
                     f"every condition: the target repeats every {cfg.period_ms:.2f} ms exactly. "
                     "The within-channel reference is therefore the same at every step, which is "
                     "why a flat curve means the figure was ignored rather than that the "
                     "manipulation did nothing."))

    # 4. duration and energy
    lens, rel = set(), []
    for c in conds:
        tr = build_trial(cfg, c, delta_ms, np.random.default_rng(seed), target_position=1)
        e = []
        for iv in (tr.first, tr.second):
            x = render_interval(cfg, Interval(iv.onsets_ms, iv.delta_ms, 0.0, iv.step_pct,
                                              iv.kind), tr.phases)
            lens.add(x.size)
            e.append(float(np.sum(x ** 2)))
        rel.append(abs(e[0] - e[1]) / max(e))
    out.append(Check("standard and target last the same time and carry the same energy",
                     len(lens) == 1 and max(rel) < 1e-3,
                     f"{len(lens)} distinct interval length(s); largest energy difference "
                     f"{100 * max(rel):.4f}%, which is the displaced tone's starting phase "
                     "landing on a different sample, not a level difference"))

    # 5. the realised shift is the one that was rendered
    q = 1000.0 / cfg.sample_rate
    worst = 0.0
    for dd in (cfg.delta_min_ms, 1.0, 5.0, 20.0, cfg.delta_max_ms):
        for c in conds:
            tr = build_trial(cfg, c, dd, np.random.default_rng(seed), direction=+1)
            worst = max(worst, abs(abs(tr.delta_realised_ms) - dd))
    out.append(Check("the realised shift matches the request", worst <= q,
                     f"largest deviation {worst * 1000:.1f} us; one sample is {q * 1000:.1f} us. "
                     "The analysis uses the realised value."))

    # 6. the level rove carries no information about the answer
    rng = np.random.default_rng(seed)
    louder = []
    for _ in range(2000):
        tr = build_trial(cfg, conds[0], delta_ms, rng)
        lv = (tr.first.level_db, tr.second.level_db)
        louder.append(int(np.argmax(lv)) + 1 == tr.target_position)
    p = float(np.mean(louder))
    out.append(Check("the level rove says nothing about which interval moved",
                     abs(p - 0.5) < 0.05,
                     f"the target was the louder interval on {100 * p:.1f}% of 2000 trials"))
    return out


def model_checks(cfg: Config) -> List[Check]:
    from .model import effective_objects, prediction_band, two_channel_check
    d = validate(cfg)
    out: List[Check] = []

    tc = two_channel_check()
    out.append(Check("the index used here reduces to the published one for two channels",
                     tc["monotone"] and abs(tc["at_0"]) < 1e-9 and abs(tc["at_1"] - 1.0) < 1e-9,
                     f"for N=2 the normalised participation ratio is monotone in "
                     f"lambda2/lambda1, running {tc['at_0']:.3f} to {tc['at_1']:.3f}. It is a "
                     "generalisation of Elhilali's reading, not a different quantity."))

    pcts = [c.step_pct for c in d.conditions if c.kind == "figure"]
    vals = [effective_objects(cfg, p)["normalised"] for p in pcts]
    mono = all(b - a >= -1e-9 for a, b in zip(vals, vals[1:]))
    l2l1 = [effective_objects(cfg, p)["lambda2_over_lambda1"] for p in pcts]
    l2mono = all(b - a >= -1e-9 for a, b in zip(l2l1, l2l1[1:]))
    out.append(Check("the step is a monotone axis for the index the design uses", mono,
                     f"effective objects {', '.join(f'{1 + v * (cfg.n_tones - 1):.2f}' for v in vals)} "
                     f"across steps {', '.join(f'{p:g}%' for p in pcts)}. "
                     + ("lambda2/lambda1 over the same steps is "
                        f"{', '.join(f'{v:.3f}' for v in l2l1)}, which is "
                        + ("also monotone." if l2mono else "NOT monotone -- which is why the "
                           "design does not use it."))))

    band = prediction_band(cfg, pcts)
    out.append(Check("the prediction survives every reading of the filter bank",
                     band["all_monotone"],
                     f"{band['n_variants']} variants: largest step down {band['max_decrease']:.4f}, "
                     f"largest disagreement in height {band['max_spread']:.3f}"))

    iso = [p for p in pcts if is_isochronous(cfg, p) and p > 0]
    if iso:
        below = max((p for p in pcts if 0 < p < iso[0]), default=None)
        gap = (abs(effective_objects(cfg, iso[0])["normalised"]
                   - effective_objects(cfg, below)["normalised"]) if below is not None else None)
        out.append(Check("the isochronous step has a neighbour the model cannot tell it from",
                         below is not None and gap is not None and gap < 0.1,
                         f"step {iso[0]:g}% is isochronous; {below:g}% is the nearest step below "
                         f"it and the model separates them by only {gap:.3f}. Any behavioural "
                         "difference between the two is therefore the rhythm cue, not coherence -- "
                         "which is the comparison `rhythm_test` reports."
                         if below is not None else
                         "no step below the isochronous one, so the rhythm cue cannot be "
                         "separated from coherence"))
    return out


def procedure_checks(cfg: Config) -> List[Check]:
    from tcoh.track import simulate
    from .design import audit_design, make_design
    out: List[Check] = []
    sim = simulate(cfg, thresholds_ms=(2.0, 6.0, 12.0), sigmas=(0.5, 0.7), n_runs=200, seed=2)
    bias = float(sim["worst_abs_bias_log"])
    out.append(Check("the staircase recovers a threshold it was given",
                     bias < 0.25 and sim["max_fail_rate"] < 0.05,
                     f"worst bias {sim['worst_abs_bias_pct']:+.1f}% across thresholds from 2 to "
                     f"12 ms and slopes from 0.5 to 0.7; single-track spread sd(log) = "
                     f"{sim['median_sd_log']:.2f} (a factor of "
                     f"{math.exp(1.96 * sim['median_sd_log']) ** 2:.2f} on a 95% interval, which "
                     f"is why one track per condition is a feasibility run and not a "
                     f"measurement); median {sim['median_trials']:.0f} trials; non-convergence "
                     f"{100 * sim['max_fail_rate']:.1f}%"))
    a = audit_design(cfg, make_design(cfg, "VERIFY", 1))
    out.append(Check("no condition is confounded with time in the session",
                     a["round_balance_spread"] < 1e-9 and a["serial_position_spread"] < 0.15,
                     f"every condition contributes one track per round (spread "
                     f"{a['round_balance_spread']:.1e}); within that, mean serial position "
                     f"varies by {a['serial_position_spread']:.3f} of the session"
                     if cfg.tracks_per_condition > 1 else
                     "With one track per condition there is a single round, so each condition "
                     "is heard once at a fixed point in the session. No ordering can fix that; "
                     "only a second track can. Expected for a feasibility preset."))
    out.append(Check("no condition repeats more than the configured run length",
                     a["run_constraint_respected"],
                     f"longest planned run {a['longest_condition_run']}, limit "
                     f"{a['max_same_condition_run']}; catch planned at "
                     f"{100 * a['catch_rate_planned']:.1f}% against {100 * cfg.catch_rate:.0f}%"))
    return out


def level_checks(cfg: Config) -> List[Check]:
    from tcoh.audiolevel import scene_db_spl
    d = validate(cfg)
    rng = np.random.default_rng(1)
    peak = 0.0
    for c in d.conditions:
        tr = build_trial(cfg, c, cfg.delta_max_ms, rng)
        peak = max(peak, float(np.abs(render_trial(cfg, tr)).max()))
    peak *= 10 ** (cfg.level_rove_db / 20.0)
    scene = scene_db_spl(cfg.tone_level_db_spl, cfg.n_tones)
    out = [Check("the waveform cannot clip", peak < 1.0,
                 f"peak {peak:.3f} FS with the rove at its maximum "
                 f"({-20 * math.log10(max(peak, 1e-9)):.1f} dB of headroom)"),
           Check("the level at the ear is comfortable and safe", scene <= 80.0,
                 f"one tone at {cfg.tone_level_db_spl:.0f} dB SPL as calibrated; at the smallest "
                 f"step all {cfg.n_tones} overlap, reaching {scene:.0f} dB SPL"),
           Check("every tone sits in a comfortable frequency range",
                 min(cfg.freqs_hz) >= 120.0 and max(cfg.freqs_hz) <= 8000.0,
                 " + ".join(f"{f:.0f}" for f in sorted(cfg.freqs_hz))
                 + " Hz -- all well inside the range where hearing is sensitive")]
    return out


def run_battery(cfg: Config) -> Dict[str, List[Check]]:
    return {"stimulus invariants": invariants(cfg), "model": model_checks(cfg),
            "procedure": procedure_checks(cfg), "level and safety": level_checks(cfg)}


def format_report(cfg: Config, battery: Dict[str, List[Check]]) -> str:
    d = validate(cfg)
    L = ["=" * 78, f"tshear verification battery   |   config {cfg.hash()}",
         " + ".join(f"{f:.0f}" for f in sorted(cfg.freqs_hz))
         + f" Hz, {cfg.tone_ms:g} ms tones at {cfg.rate_hz:g} Hz, "
           f"{cfg.n_repeats} repetitions, {len(d.conditions)} conditions", "=" * 78]
    n_fail = 0
    for section, checks in battery.items():
        L.append(f"\n-- {section} " + "-" * max(4, 74 - len(section)))
        for c in checks:
            L.append(c.line())
            n_fail += not c.passed
    total = sum(len(v) for v in battery.values())
    L += ["", "=" * 78,
          f"{total - n_fail}/{total} checks passed"
          + (f"   -- {n_fail} FAILED, see above" if n_fail else ""), "=" * 78]
    return "\n".join(L)
