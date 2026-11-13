"""Everything that can be checked without a listener, checked.

Organised around the claims the design makes, so a reader who distrusts one can find the number
that supports it. For scatter timing the claims that carry the argument are in "what a listener
can use": a listener judging either tone's own timing never reaches threshold, an ideal observer
of the timing's one weak point does little better, and the relational listener converges at
every lag. Together they say the only route to the answer runs through the A-B relation. Grid
sessions (the original design) are checked against the claims that design made.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy import stats

from .config import (FUSION_MARGIN_MS, SIGN, Config, conditions, probe_conditions,
                     realised_timing, scatter_geometry, shifts, validate)
from .stimulus import (build_trial, between_token_silence, render_interval, render_trial,
                       token_gaps, token_lags)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str

    def line(self) -> str:
        return f"  [{'PASS' if self.passed else 'FAIL'}] {self.name}\n         {self.detail}"


def _all_conds(cfg: Config):
    return list(conditions(cfg)) + list(probe_conditions(cfg))


def stimulus_checks(cfg: Config, n: int = 3000, seed: int = 4) -> List[Check]:
    out = []
    rng = np.random.default_rng(seed)
    conds = conditions(cfg)
    pair_conds = [c for c in conds if c.kind == "pair"]
    scatter = cfg.timing == "scatter"

    # 1. lag exact everywhere except the target token (context tokens at the context lag)
    worst, bad = 0.0, []
    for c in pair_conds:
        want_lags = np.full(cfg.n_pairs, c.lag_ms)
        if c.context_lag_ms is not None:
            want_lags[:cfg.n_context] = c.context_lag_ms
        for _ in range(200):
            tr = build_trial(cfg, c, float(rng.uniform(1, cfg.delta_max_ms)), rng)
            for iv in (tr.first, tr.second):
                lag = token_lags(cfg, iv)
                keep = [k for k in range(cfg.n_pairs) if k != iv.target_pair]
                worst = max(worst, float(np.abs(lag[keep] - want_lags[keep]).max()))
            got = token_lags(cfg, tr.target)[tr.target_pair]
            want = c.lag_ms + (tr.delta_signed_ms if cfg.a_leads else -tr.delta_signed_ms)
            if abs(got - want) > 1e-9:
                bad.append(c.name)
    out.append(Check("the lag is exact in every token except the target",
                     not bad and worst < 1e-9,
                     f"{len(pair_conds)} token conditions x 200 trials: largest error {worst:.1e} ms; "
                     "in the target token the lag is exactly lag - delta (in) or lag + delta (out)"))

    # 2. a gap inside a token only where the design puts one: never when B comes earlier, and
    #    when it comes later exactly when lag + delta > tone_ms, with the trial saying so
    worst_in, bad_flag, worst_err, n_gap = -np.inf, 0, 0.0, 0
    for c in pair_conds:
        for d in (cfg.delta_min_ms, cfg.delta_max_ms / 2, cfg.delta_max_ms):
            for _ in range(100):
                tr = build_trial(cfg, c, d, rng)
                for iv in (tr.first, tr.second):
                    gap = token_gaps(cfg, iv)
                    keep = [k for k in range(cfg.n_pairs) if k != iv.target_pair]
                    worst_in = max(worst_in, float(gap[keep].max()))
                    if iv.target_pair is not None:
                        g = gap[iv.target_pair]
                        if tr.delta_signed_ms < 0:
                            worst_in = max(worst_in, float(g))
                        else:
                            worst_err = max(worst_err, abs(g - (c.lag_ms + tr.delta_ms - cfg.tone_ms)))
                        bad_flag += (g > 1e-9) != tr.opens_gap
                        n_gap += tr.opens_gap
    out.append(Check("a gap opens inside a token only where the design puts one",
                     worst_in <= 1e-9 and bad_flag == 0 and worst_err < 1e-9,
                     f"every untouched token, and every target where B comes earlier: the later tone "
                     f"starts at most {worst_in:+.1f} ms after the earlier one ends (0 = abutting). "
                     + (f"Where B comes later: gap = lag + delta - {cfg.tone_ms:g} to {worst_err:.0e} "
                        f"ms, flagged correctly on every trial ({n_gap} with a gap)"
                        if cfg.can_move_late else "No condition moves B later.")))

    # 3. tokens keep their distance, and no tone sums with one of its own frequency
    lens, same, between = set(), [], []
    for c in _all_conds(cfg):
        for _ in range(150):
            tr = build_trial(cfg, c, cfg.ceiling_for(c.kind), rng)
            lens.add(render_trial(cfg, tr).size)
            for iv in (tr.first, tr.second):
                for ons in (iv.b_onsets_ms, iv.a_onsets_ms):
                    o = np.sort(ons[np.isfinite(ons)])
                    if o.size > 1:
                        same.append(float(np.diff(o).min() - cfg.tone_ms))
                if c.kind == "pair":
                    between.append(float(between_token_silence(cfg, iv).min()))
    worst_same = min(same)
    floor = (cfg.token_gap_ms if scatter else 0.0) if cfg.clearance == "token" else -np.inf
    out.append(Check("duration is fixed, tokens keep apart, and no tone sums with its own frequency",
                     len(lens) == 1 and worst_same >= FUSION_MARGIN_MS
                     and (min(between) >= floor - 1e-6),
                     f"{len(lens)} trial length(s); with every change at its ceiling, the least "
                     f"silence between two tones of one frequency is {worst_same:.1f} ms "
                     f"(validated worst case {validate(cfg).same_channel_gap_ms:.1f}); between two "
                     f"tokens {min(between):.1f} ms"
                     + (f" (required {cfg.token_gap_ms:g})" if scatter and cfg.clearance == "token"
                        else " (tokens may interleave: clearance = channel)" if scatter else "")))

    # 4. the two intervals' timing is independent
    c0 = pair_conds[0]
    r1, r2 = [], []
    for _ in range(n):
        tr = build_trial(cfg, c0, 10.0, rng)
        r1.append(tr.first.ref_ms); r2.append(tr.second.ref_ms)
    r1, r2 = np.array(r1), np.array(r2)
    r = float(np.max(np.abs([np.corrcoef(r1[:, k], r2[:, k])[0, 1] for k in range(cfg.n_pairs)])))
    out.append(Check("the two intervals' timing is drawn independently", r < 0.06,
                     f"largest |r| between the intervals' onsets at any token = {r:.3f} over {n} trials"))

    if scatter:
        # 5. no rhythm to learn: gaps exchangeable, wide, the shortest also the commonest
        geo, rt = scatter_geometry(cfg), realised_timing(cfg)
        ioi = np.diff(r1, axis=1)
        means = ioi.mean(0)
        hist, edges = np.histogram(ioi.ravel(), bins=12)
        ok = (np.ptp(means) / means.mean() < 0.06 and int(np.argmax(hist)) == 0
              and rt["rate_hz"] >= cfg.rate_hz * 0.99)
        out.append(Check("the timing has no rhythm to learn", ok,
                         f"gaps between tokens {rt['mean_ioi_ms']:.0f} +- {rt['sd_ioi_ms']:.0f} ms "
                         f"({rt['min_ioi_ms']:.0f}-{rt['max_ioi_ms']:.0f}), {rt['rate_hz']:.2f} Hz on "
                         f"average; the same at every position in the sequence (means "
                         f"{means.min():.0f}-{means.max():.0f} ms); the commonest gap is the "
                         f"shortest, so a short gap is never a sign of anything"))

        # 6. the one weak point: a move that crosses the minimum gap
        hits = 0
        m = 4000
        g = geo["min_ioi_ms"]
        for i in range(m):
            c = pair_conds[i % len(pair_conds)]
            tr = build_trial(cfg, c, cfg.delta_max_ms, rng)
            tgt = tr.target
            hits += any((np.diff(x) < g - 1e-6).any() for x in (tgt.a_onsets_ms, tgt.b_onsets_ms))
        p = hits / m
        out.append(Check("only a move across the minimum gap could ever give a tone away",
                         0.5 + p / 2 < 0.70,
                         f"at the {cfg.delta_max_ms:g} ms ceiling, {100 * p:.1f}% of target intervals "
                         f"put two tones of one channel closer than {g:.1f} ms. An observer with "
                         f"perfect timing who knows that minimum to the millisecond can use those, "
                         f"and only those: at most {100 * (0.5 + p / 2):.0f}% correct, against the "
                         f"79.4% the staircase needs"))
    else:
        # 5'. grid, B moving alone: A carries no information
        a_tgt, a_std, moved = [], [], 0
        for _ in range(n):
            tr = build_trial(cfg, c0, 30.0, rng)
            nom = cfg.nominal_onsets() + (0.0 if cfg.a_leads else c0.lag_ms)
            a_tgt.append(tr.target.a_onsets_ms - nom); a_std.append(tr.standard.a_onsets_ms - nom)
            moved += bool(np.any(np.abs(tr.target.a_onsets_ms - nom - tr.target.jitter_ms) > 1e-9))
        p_ks = min(stats.ks_2samp(np.array(a_tgt)[:, k], np.array(a_std)[:, k]).pvalue
                   for k in range(cfg.n_pairs))
        out.append(Check("the A channel carries no information about which interval moved",
                         p_ks > 0.001 and (moved == 0 or cfg.shift == "split"),
                         f"A onsets in target and standard intervals have the same distribution at "
                         f"every pair (smallest KS p = {p_ks:.3f} over {n} trials)"))

    # 7. which tones move, where, and which way
    pos, wrong = [], 0
    for i in range(n):
        c = pair_conds[i % len(pair_conds)]
        tr = build_trial(cfg, c, 20.0, rng)
        pos.append(tr.target_pair)
        if c.direction in SIGN:
            want_a, want_b = shifts(cfg, SIGN[c.direction] * 20.0)
            wrong += (abs(tr.target.a_shift_ms - want_a) > 1e-9) or (abs(tr.target.b_shift_ms - want_b) > 1e-9)
    share = {p: pos.count(p) / n for p in cfg.target_pairs}
    how = ("A moves delta/2 and B delta/2 the other way" if cfg.shift == "split"
           else "B moves delta, A stays")
    out.append(Check("only the target token changes, and only the way its condition says",
                     all(abs(v - 1 / len(share)) < 0.03 for v in share.values()) and not wrong,
                     "target share " + ", ".join(
                         f"{'N' if p == cfg.n_pairs - 1 else 'N-1'}: {100 * v:.1f}%"
                         for p, v in sorted(share.items(), reverse=True))
                     + f"; {how}; {wrong} trials moved otherwise"))

    # 8. realised change
    q = 1000.0 / cfg.sample_rate
    worst = max(abs(abs(build_trial(cfg, c0, d, rng).delta_realised_ms) - d)
                for d in (cfg.delta_min_ms, 1.0, 7.3, 33.0, cfg.delta_max_ms))
    out.append(Check("the realised change matches the request", worst <= 2 * q,
                     f"largest deviation {1000 * worst:.1f} us; one sample is {1000 * q:.1f} us "
                     "(two tones are rounded when both move)"))

    # 9. level rove and energy
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
    out.append(Check("neither level nor energy says which interval changed",
                     abs(np.mean(louder) - 0.5) < 0.05 and max(rel) < 0.01,
                     f"target louder on {100 * np.mean(louder):.1f}% of trials; with the roves "
                     f"removed, the two intervals' energy differs by at most {100 * max(rel):.3f}%"
                     + (f"; every token also roved +-{cfg.token_rove_db:g} dB" if cfg.token_rove_db
                        else "")))

    # 10. the probes keep each tone's own timing and lose the relation
    probes = probe_conditions(cfg)
    if probes:
        pr, pa = probes[0], pair_conds[0]
        ia_p, ia_t, lag_sd = [], [], []
        for _ in range(1500):
            tp = build_trial(cfg, pr, cfg.delta_max_ms, rng)
            tt = build_trial(cfg, pa, cfg.delta_max_ms, rng)
            ia_p.append(np.diff(tp.standard.a_onsets_ms)); ia_t.append(np.diff(tt.standard.a_onsets_ms))
            lag_sd.append(np.std(tp.standard.b_onsets_ms - tp.standard.a_onsets_ms))
        p_ks = stats.ks_2samp(np.ravel(ia_p), np.ravel(ia_t)).pvalue
        out.append(Check("decoupled probes keep each tone's own timing and lose the relation",
                         p_ks > 0.001 and np.median(lag_sd) > 2 * cfg.delta_max_ms,
                         f"a channel's gaps in a probe and in a token condition: same distribution "
                         f"(KS p = {p_ks:.2f}); the 'lag' between a probe's A and B wanders by "
                         f"{np.median(lag_sd):.0f} ms within an interval, against 0 in every token"))
    return out


def strategy_checks(cfg: Config, n: int = 1200) -> List[Check]:
    from .observer import p_correct, threshold
    out = []
    for direction in cfg.directions:
        tag = f" ({direction})" if len(cfg.directions) > 1 else ""
        pair_conds = [c for c in conditions(cfg) if c.kind == "pair" and c.direction == direction]
        pair = {(c.lag_ms if c.context_lag_ms is None else f"{c.context_lag_ms:g}->{c.lag_ms:g}"):
                threshold(cfg, c, "pair", n=n) for c in pair_conds}
        if cfg.timing == "grid":
            single = {c.lag_ms: threshold(cfg, c, "single", n=n) for c in pair_conds}
            s_vals = list(single.values())
            out.append(Check(f"a B-only listener cannot see the lag at all{tag}",
                             max(s_vals) / min(s_vals) < 1.25,
                             "single-channel 79.4% point " + ", ".join(f"{v:.0f}" for v in s_vals)
                             + " ms across lags " + ", ".join(f"{k:g}" for k in single)
                             + " -- flat, because B's own distribution is the same whatever the lag"))
            worst_pair, floor = max(pair.values()), min(s_vals)
            out.append(Check(f"the jitter defeats a B-only listener{tag}", floor > 1.3 * worst_pair,
                             f"a B-only observer needs {floor:.0f} ms; an observer using the A-B lag "
                             f"needs {min(pair.values()):.1f}-{worst_pair:.1f} ms across lags."))
            continue
        ends = [pair_conds[0], pair_conds[-1]]
        pc = {m: max(p_correct(cfg, c, cfg.delta_max_ms, m, 2 * n, 7) for c in ends)
              for m in ("single", "tones", "boundary")}
        out.append(Check(f"a listener judging the tones' own timing never reaches threshold{tag}",
                         max(pc["single"], pc["tones"]) < 0.62,
                         f"at the {cfg.delta_max_ms:g} ms ceiling, judging B against its own "
                         f"neighbours: {100 * pc['single']:.0f}% correct; watching A and B each for "
                         f"a hiccup: {100 * pc['tones']:.0f}%; the staircase needs 79.4%"))
        out.append(Check(f"an ideal observer of the timing's one weak point does little better{tag}",
                         pc["boundary"] < 0.70,
                         f"perfect timing and the minimum gap known to the millisecond: "
                         f"{100 * pc['boundary']:.0f}% at the ceiling. No listener has either"))
        fin = [v for v in pair.values() if np.isfinite(v)]
        out.append(Check(f"a listener using the A-B lag converges at every lag{tag}",
                         len(fin) == len(pair) and max(fin) < 0.6 * cfg.delta_max_ms,
                         "79.4% point " + ", ".join(f"{v:.1f}" for v in pair.values())
                         + " ms across " + ", ".join(f"{k}" for k in pair)
                         + f"; the ceiling is {cfg.delta_max_ms:g}"))
    probes = probe_conditions(cfg)
    if probes:
        # far from 50% either way is information: a listener could flip a below-chance rule
        res = {(m, p.direction): p_correct(cfg, p, cfg.delta_max_ms, m, 2 * n, 9)
               for m in ("single", "tones", "pair") for p in probes}
        out.append(Check("on decoupled probes every listener is near chance",
                         max(abs(v - 0.5) for v in res.values()) < 0.12,
                         ", ".join(f"{m} {d} {100 * v:.0f}%" for (m, d), v in res.items())
                         + f" at {cfg.delta_max_ms:g} ms. A person far from 50% here has a route "
                         "around the relation the design does not know about"))
    return out


def context_checks(cfg: Config, n: int = 1200) -> List[Check]:
    """The context design separates the accounts it claims to separate, on the real stimulus."""
    from .context import model_table, predictions
    from .observer import CONTEXT_TAU_MS
    pr = predictions(cfg, n=n)
    out = []
    for L in cfg.lags_ms:
        idx = {m: v["indices"].get(L) for m, v in pr.items()}
        if any(v is None for v in idx.values()):
            out.append(Check("every model listener reaches threshold in every context", False,
                             "a listener sat at the ceiling, so its signature cannot be read"))
            continue
        f = lambda i: f"P {i['P']:+.2f}, D {i['D']:+.2f}, S {i['S']:+.2f}"
        out.append(Check("a listener with no memory of the context shows no context effect",
                         max(abs(idx['pair'][k]) for k in "PDS") < 0.12,
                         f"the plain relational listener: {f(idx['pair'])}. Any effect a person shows "
                         "is about the person, not the stimulus"))
        out.append(Check("a listener fooled by the switch shows it as D, not as P",
                         abs(idx['change']['P']) < 0.25 and idx['change']['D'] > 0.3,
                         f"listening for any change of the lag: {f(idx['change'])}. With one direction "
                         "only, its D would have read as persistence"))
        out.append(Check("persistence and contrast come out as P of opposite sign, D near zero",
                         idx['persist']['P'] > 0.2 and idx['contrast']['P'] < -0.2
                         and max(abs(idx['persist']['D']), abs(idx['contrast']['D'])) < 0.12,
                         f"persistence ({CONTEXT_TAU_MS:g} ms decay): {f(idx['persist'])}; contrast: "
                         f"{f(idx['contrast'])}"))
        mt = model_table(cfg)[L]
        out.append(Check("temporal coherence itself predicts almost no carry-over",
                         max(mt.values()) - min(mt.values()) < 0.05,
                         "l2/l1 over the 600 ms ending at the last token, by context: "
                         + ", ".join(f"{cfg.lag_pct(C):g}% {v:.3f}" for C, v in mt.items())
                         + ". A context effect in a listener is memory the model does not have"))
    return out


def model_checks(cfg: Config) -> List[Check]:
    from .model import prediction_band
    lags = list(dict.fromkeys(c.lag_ms for c in conditions(cfg) if c.kind == "pair"))
    band = prediction_band(cfg, lags, n_seq=15)
    return [Check("the prediction is monotone in the lag under every reading of the filter bank",
                  band["all_monotone"],
                  f"{len(band['curves'])} variants: largest step down {band['max_decrease']:.4f}; "
                  f"mid curve " + ", ".join(f"{v:.2f}" for v in band["mid"]) + "; largest "
                  f"disagreement in height {band['max_spread']:.2f}")]


def procedure_checks(cfg: Config) -> List[Check]:
    from tcoh.track import simulate
    from .design import audit_design, audit_timing, describe_timing_audit, make_design
    sim = simulate(cfg, thresholds_ms=(2.0, 10.0, 40.0), sigmas=(0.5, 0.7), n_runs=200, seed=2)
    a = audit_design(cfg, make_design(cfg, "VERIFY", 1))
    timing = []
    if cfg.timing == "scatter":
        t = audit_timing(cfg, make_design(cfg, "VERIFY", 1))
        timing = [Check("every interval a session plays is drawn from its seed and passes the screen",
                        t["ok"], describe_timing_audit(t).strip().replace("\n            ", ""))]
    return timing + [
        Check("the staircase recovers a threshold it was given",
              sim["worst_abs_bias_log"] < 0.25 and sim["max_fail_rate"] < 0.05,
              f"worst bias {sim['worst_abs_bias_pct']:+.1f}% from 2 to 40 ms; single-track "
              f"sd(log) {sim['median_sd_log']:.2f}; median {sim['median_trials']:.0f} trials"),
        Check("no lag is confounded with time in the session",
              a["round_balance_spread"] < 1e-9 and a["serial_position_spread_within_direction"] < 0.15,
              (f"serial-position spread across lags {a['serial_position_spread_within_direction']:.3f}"
               " of the session" if cfg.tracks_per_condition > 1 else
               "one track per condition: each lag is heard once, interleaved with the others "
               f"(spread {a['serial_position_spread_within_direction']:.3f})")
              + _direction_note(a)),
        Check("no condition repeats more than the configured run length",
              a["run_constraint_respected"],
              f"longest run {a['longest_condition_run']}, limit {a['max_same_condition_run']}; "
              f"catch planned at {100 * a['catch_rate_planned']:.1f}%"
              + (f", probes at {100 * a['probe_rate_planned']:.1f}%" if cfg.probe_rate else "")),
    ]


def _direction_note(a: dict) -> str:
    pos = a.get("direction_mean_position", {})
    if len(pos) < 2:
        return ""
    txt = "; directions at mean position " + ", ".join(f"{d} {v:.2f}" for d, v in pos.items())
    if max(pos.values()) - min(pos.values()) < 0.05:
        return txt + " (ABBA balances them)"
    return (txt + ". One block each: within this session direction goes with time. The opening "
            "direction alternates across sessions, so a pair of sessions balances it")


def level_checks(cfg: Config) -> List[Check]:
    from tcoh.audiolevel import scene_db_spl
    rng = np.random.default_rng(1)
    peak = 0.0
    for c in _all_conds(cfg):
        for _ in range(20):
            tr = build_trial(cfg, c, 30.0, rng)
            for iv, ph in ((tr.first, tr.phases[0]), (tr.second, tr.phases[1])):
                peak = max(peak, float(np.abs(render_interval(cfg, iv, ph, level=False)).max()))
    rove = cfg.level_rove_db + cfg.token_rove_db
    peak *= 10 ** (rove / 20.0)
    ra = validate(cfg).ratio
    return [Check("the waveform cannot clip", peak < 1.0,
                  f"peak {peak:.3f} FS with every rove at its maximum (+{rove:g} dB; "
                  f"{-20 * math.log10(peak):.1f} dB of headroom)"),
            Check("the level is comfortable and safe",
                  scene_db_spl(cfg.tone_level_db_spl + rove, 2) <= 80,
                  f"one tone {cfg.tone_level_db_spl:.0f} dB SPL; overlapping at lag 0 with every "
                  f"rove at its top, {scene_db_spl(cfg.tone_level_db_spl + rove, 2):.0f} dB SPL"),
            Check("the tones are more than an octave apart and in no simple ratio",
                  ra["octaves"] > 1.0 and ra["mistuning_pct"] >= 3.0,
                  f"{cfg.f_a_hz:.0f} / {cfg.f_b_hz:.0f} Hz: {ra['octaves']:.2f} octaves, "
                  f"{ra['erbs']:.1f} ERB, {ra['mistuning_pct']:.1f}% from {ra['nearest']}, the "
                  f"nearest ratio with terms up to 8")]


def run_battery(cfg: Config) -> Dict[str, List[Check]]:
    out = {"stimulus": stimulus_checks(cfg), "what a listener can use": strategy_checks(cfg)}
    if cfg.n_context:
        out["what the context design can tell apart"] = context_checks(cfg)
    else:
        out["model"] = model_checks(cfg)
    out.update({"procedure": procedure_checks(cfg), "level and safety": level_checks(cfg)})
    return out


def format_report(cfg: Config, battery: Dict[str, List[Check]]) -> str:
    timing = (f"{cfg.n_pairs} tokens scattered at {cfg.rate_hz:g} Hz on average, change split "
              f"between A and B" if cfg.timing == "scatter" else
              f"{cfg.n_pairs} pairs at {cfg.rate_hz:g} Hz, jitter +-{cfg.pair_jitter_ms:g} ms")
    L = ["=" * 78, f"tpair verification battery   |   config {cfg.hash()}",
         f"A {cfg.f_a_hz:.0f} / B {cfg.f_b_hz:.0f} Hz, {cfg.tone_ms:g} ms tones, {timing}",
         "=" * 78]
    nf = 0
    for sec, checks in battery.items():
        L.append(f"\n-- {sec} " + "-" * max(4, 74 - len(sec)))
        for c in checks:
            L.append(c.line()); nf += not c.passed
    tot = sum(len(v) for v in battery.values())
    L += ["", "=" * 78, f"{tot - nf}/{tot} checks passed" + (f"   -- {nf} FAILED" if nf else ""),
          "=" * 78]
    return "\n".join(L)
