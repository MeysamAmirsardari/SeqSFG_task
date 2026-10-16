"""Everything that can be checked without a listener, checked.

Organised around the claims the design makes, so a reader who distrusts one can find the number
that supports it. The two that carry the argument are `a_channel_is_uninformative` and
`the jitter defeats a B-only listener`: together they say that the only route to the answer
below about 40 ms runs through the relation between A and B.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy import stats

from .config import FUSION_MARGIN_MS, Config, conditions, validate
from .stimulus import Interval, build_trial, render_interval, render_trial


@dataclass
class Check:
    name: str
    passed: bool
    detail: str

    def line(self) -> str:
        return f"  [{'PASS' if self.passed else 'FAIL'}] {self.name}\n         {self.detail}"


def _cond(cfg, lag=None, kind="pair"):
    return next(c for c in conditions(cfg) if c.kind == kind and (lag is None or c.lag_ms == lag))


def stimulus_checks(cfg: Config, n: int = 3000, seed: int = 4) -> List[Check]:
    out = []
    rng = np.random.default_rng(seed)
    conds = conditions(cfg)
    pair_conds = [c for c in conds if c.kind == "pair"]

    # 1. lag exact everywhere except the displaced B
    worst, bad = 0.0, []
    for c in pair_conds:
        for _ in range(200):
            tr = build_trial(cfg, c, float(rng.uniform(1, cfg.delta_max_ms)), rng)
            for iv in (tr.first, tr.second):
                lag = iv.a_onsets_ms - iv.b_onsets_ms
                keep = [k for k in range(cfg.n_pairs) if k != iv.target_pair]
                worst = max(worst, float(np.abs(lag[keep] - c.lag_ms).max()))
            t = tr.target
            got = t.a_onsets_ms[tr.target_pair] - t.b_onsets_ms[tr.target_pair]
            if abs(got - (c.lag_ms - tr.delta_signed_ms)) > 1e-9:
                bad.append(c.name)
    out.append(Check("the lag is exact in every pair except the displaced one", not bad and worst < 1e-9,
                     f"{len(pair_conds)} lags x 200 trials: largest error {worst:.1e} ms; in the "
                     "displaced pair the lag is exactly lag + delta"))

    # 2. jitter independent between intervals
    c0 = pair_conds[0]
    j1, j2 = [], []
    for _ in range(n):
        tr = build_trial(cfg, c0, 10.0, rng)
        j1.append(tr.first.jitter_ms); j2.append(tr.second.jitter_ms)
    j1, j2 = np.array(j1), np.array(j2)
    r = float(np.max(np.abs([np.corrcoef(j1[:, k], j2[:, k])[0, 1] for k in range(cfg.n_pairs)])))
    J = cfg.pair_jitter_ms
    span_ok = (j1.min() >= -J - 1e-9) and (j1.max() <= J + 1e-9)
    ks = stats.kstest((j1.ravel() + J) / (2 * J), "uniform").pvalue
    out.append(Check("the jitter is independent between the two intervals", r < 0.06 and span_ok,
                     f"largest |r| between the intervals' jitter at any pair = {r:.3f} over {n} "
                     f"trials; values span [{j1.min():.1f}, {j1.max():.1f}] ms against +-{J:g}; "
                     f"uniform (KS p = {ks:.2f})"))

    # 3. A carries no information
    a_tgt, a_std, moved = [], [], 0
    for _ in range(n):
        tr = build_trial(cfg, c0, 30.0, rng)
        nom = cfg.nominal_onsets() + c0.lag_ms
        a_tgt.append(tr.target.a_onsets_ms - nom); a_std.append(tr.standard.a_onsets_ms - nom)
        moved += abs((tr.target.a_onsets_ms[tr.target_pair] - tr.target.b_onsets_ms[tr.target_pair])
                     - (c0.lag_ms + 30.0)) > 1e-9
    p_ks = min(stats.ks_2samp(np.array(a_tgt)[:, k], np.array(a_std)[:, k]).pvalue
               for k in range(cfg.n_pairs))
    out.append(Check("the A channel carries no information about which interval moved",
                     p_ks > 0.001 and moved == 0,
                     f"A onsets in target and standard intervals have the same distribution at "
                     f"every pair (smallest KS p = {p_ks:.3f} over {n} trials); no A tone ever moves "
                     "relative to its own pair. A listener attending only to A is at chance."))

    # 4. which tone moves, where, and which way
    pos = []
    for _ in range(n):
        tr = build_trial(cfg, c0, 20.0, rng)
        pos.append(tr.target_pair)
        assert tr.delta_signed_ms < 0 or cfg.delta_direction != "backward"
    share = {p: pos.count(p) / n for p in cfg.target_pairs}
    out.append(Check("only the B tone of pair N or N-1 moves, and only early",
                     all(abs(v - 1 / len(share)) < 0.03 for v in share.values()),
                     "target pair share " + ", ".join(
                         f"{'N' if p == cfg.n_pairs - 1 else 'N-1'}: {100 * v:.1f}%"
                         for p, v in sorted(share.items(), reverse=True))
                     + f"; direction {cfg.delta_direction} on every trial"))

    # 5. duration fixed, nothing overlaps its own channel, at the extremes
    lens, gaps = set(), []
    for c in conds:
        for _ in range(150):
            tr = build_trial(cfg, c, cfg.delta_max_ms, rng)
            lens.add(render_trial(cfg, tr).size)
            for iv in (tr.first, tr.second):
                for ons in (iv.b_onsets_ms, iv.a_onsets_ms):
                    o = np.sort(ons[np.isfinite(ons)])
                    if o.size > 1:
                        gaps.append(float(np.diff(o).min() - cfg.tone_ms))
    worst_gap = min(gaps)
    out.append(Check("duration is fixed and no tone ever sums with one of its own frequency",
                     len(lens) == 1 and worst_gap >= FUSION_MARGIN_MS,
                     f"{len(lens)} trial length(s); smallest silent gap between two tones of one "
                     f"frequency, with delta at its {cfg.delta_max_ms:g} ms ceiling: "
                     f"{worst_gap:.1f} ms (validated worst case {validate(cfg).same_channel_gap_ms:.1f})"))

    # 6. realised shift
    q = 1000.0 / cfg.sample_rate
    worst = max(abs(abs(build_trial(cfg, c0, d, rng).delta_realised_ms) - d)
                for d in (cfg.delta_min_ms, 1.0, 7.3, 33.0, cfg.delta_max_ms))
    out.append(Check("the realised shift matches the request", worst <= q,
                     f"largest deviation {1000 * worst:.1f} us; one sample is {1000 * q:.1f} us"))

    # 7. level rove and energy
    louder, rel = [], []
    for _ in range(1500):
        tr = build_trial(cfg, c0, 20.0, rng)
        lv = (tr.first.level_db, tr.second.level_db)
        louder.append(int(np.argmax(lv)) + 1 == tr.target_position)
    for c in pair_conds[:1] + pair_conds[-1:]:
        for _ in range(40):
            tr = build_trial(cfg, c, 30.0, rng)
            e = [float(np.sum(render_interval(cfg, iv, ph, level=False) ** 2))
                 for iv, ph in ((tr.first, tr.phases[0]), (tr.second, tr.phases[1]))]
            rel.append(abs(e[0] - e[1]) / max(e))
    out.append(Check("neither level nor energy says which interval moved",
                     abs(np.mean(louder) - 0.5) < 0.05 and max(rel) < 0.01,
                     f"target louder on {100 * np.mean(louder):.1f}% of trials; with the rove "
                     f"removed, the two intervals' energy differs by at most {100 * max(rel):.3f}%"))
    return out


def strategy_checks(cfg: Config, n: int = 1200) -> List[Check]:
    from .observer import threshold
    out = []
    pair_conds = [c for c in conditions(cfg) if c.kind == "pair"]
    single = {c.lag_ms: threshold(cfg, c, "single", n=n) for c in pair_conds}
    pair = {c.lag_ms: threshold(cfg, c, "pair", n=n) for c in pair_conds}
    s_vals = list(single.values())
    flat = max(s_vals) / min(s_vals) < 1.25
    out.append(Check("a B-only listener cannot see the lag at all",
                     flat,
                     "single-channel 79.4% point " + ", ".join(f"{v:.0f}" for v in s_vals)
                     + " ms across lags " + ", ".join(f"{k:g}" for k in single)
                     + " -- flat, because B's own distribution is the same whatever the lag"))
    worst_pair = max(pair.values())
    floor = min(s_vals)
    out.append(Check("the jitter defeats a B-only listener", floor > 1.3 * worst_pair,
                     f"a B-only observer needs {floor:.0f} ms; an observer using the A-B lag needs "
                     f"{min(pair.values()):.1f}-{worst_pair:.1f} ms across lags. Below ~{floor:.0f} ms "
                     "the answer has to come through the relation between the tones."))
    return out


def model_checks(cfg: Config) -> List[Check]:
    from .model import prediction_band
    lags = [c.lag_ms for c in conditions(cfg) if c.kind == "pair"]
    band = prediction_band(cfg, lags, n_seq=15)
    return [Check("the prediction is monotone in the lag under every reading of the filter bank",
                  band["all_monotone"],
                  f"{len(band['curves'])} variants: largest step down {band['max_decrease']:.4f}; "
                  f"mid curve " + ", ".join(f"{v:.2f}" for v in band["mid"]) + "; largest "
                  f"disagreement in height {band['max_spread']:.2f}")]


def procedure_checks(cfg: Config) -> List[Check]:
    from tcoh.track import simulate
    from .design import audit_design, make_design
    sim = simulate(cfg, thresholds_ms=(2.0, 10.0, 40.0), sigmas=(0.5, 0.7), n_runs=200, seed=2)
    a = audit_design(cfg, make_design(cfg, "VERIFY", 1))
    return [
        Check("the staircase recovers a threshold it was given",
              sim["worst_abs_bias_log"] < 0.25 and sim["max_fail_rate"] < 0.05,
              f"worst bias {sim['worst_abs_bias_pct']:+.1f}% from 2 to 40 ms; single-track "
              f"sd(log) {sim['median_sd_log']:.2f}; median {sim['median_trials']:.0f} trials"),
        Check("no condition is confounded with time in the session",
              a["round_balance_spread"] < 1e-9 and a["serial_position_spread"] < 0.15,
              f"serial-position spread {a['serial_position_spread']:.3f} of the session"
              if cfg.tracks_per_condition > 1 else
              "one track per condition: each condition is heard once. Expected for a feasibility "
              "preset; only a second track can fix it."),
        Check("no condition repeats more than the configured run length",
              a["run_constraint_respected"],
              f"longest run {a['longest_condition_run']}, limit {a['max_same_condition_run']}; "
              f"catch planned at {100 * a['catch_rate_planned']:.1f}%"),
    ]


def level_checks(cfg: Config) -> List[Check]:
    from tcoh.audiolevel import scene_db_spl
    rng = np.random.default_rng(1)
    peak = max(float(np.abs(render_trial(cfg, build_trial(cfg, c, 30.0, rng))).max())
               for c in conditions(cfg) for _ in range(20))
    peak *= 10 ** (cfg.level_rove_db / 20.0)
    ra = validate(cfg).ratio
    return [Check("the waveform cannot clip", peak < 1.0,
                  f"peak {peak:.3f} FS with the rove at its maximum "
                  f"({-20 * math.log10(peak):.1f} dB of headroom)"),
            Check("the level is comfortable and safe",
                  scene_db_spl(cfg.tone_level_db_spl, 2) <= 80,
                  f"one tone {cfg.tone_level_db_spl:.0f} dB SPL; overlapping at lag 0, "
                  f"{scene_db_spl(cfg.tone_level_db_spl, 2):.0f} dB SPL"),
            Check("the tones are more than an octave apart and in no simple ratio",
                  ra["octaves"] > 1.0 and ra["mistuning_pct"] >= 3.0,
                  f"{cfg.f_a_hz:.0f} / {cfg.f_b_hz:.0f} Hz: {ra['octaves']:.2f} octaves, "
                  f"{ra['erbs']:.1f} ERB, {ra['mistuning_pct']:.1f}% from {ra['nearest']}, the "
                  f"nearest ratio with terms up to 8")]


def run_battery(cfg: Config) -> Dict[str, List[Check]]:
    return {"stimulus": stimulus_checks(cfg), "what a listener can use": strategy_checks(cfg),
            "model": model_checks(cfg), "procedure": procedure_checks(cfg),
            "level and safety": level_checks(cfg)}


def format_report(cfg: Config, battery: Dict[str, List[Check]]) -> str:
    L = ["=" * 78, f"tpair verification battery   |   config {cfg.hash()}",
         f"A {cfg.f_a_hz:.0f} / B {cfg.f_b_hz:.0f} Hz, {cfg.tone_ms:g} ms tones, {cfg.n_pairs} "
         f"pairs at {cfg.rate_hz:g} Hz, jitter +-{cfg.pair_jitter_ms:g} ms", "=" * 78]
    nf = 0
    for sec, checks in battery.items():
        L.append(f"\n-- {sec} " + "-" * max(4, 74 - len(sec)))
        for c in checks:
            L.append(c.line()); nf += not c.passed
    tot = sum(len(v) for v in battery.values())
    L += ["", "=" * 78, f"{tot - nf}/{tot} checks passed" + (f"   -- {nf} FAILED" if nf else ""),
          "=" * 78]
    return "\n".join(L)
