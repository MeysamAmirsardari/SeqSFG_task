"""Every trial a session will play, drawn from its seed and checked before anyone listens.

A session's sounds are fixed by its seed except for the tone level, which the tracks choose. So
every planned trial (the examples, the practice, every trial of every track and its first repeat
after a pause, the ratings) is drawn here exactly as the session will draw it, and checked:

  timing     each band's own signal is non-zero only inside its scheduled bursts, to the sample:
             precursor j of band b at lead + j x NOI + slot x dT, the final burst at lead + n x NOI
             in every band; the tone band at the same samples in every trial; one trial length
  silence    at least min_gap_ms between two bursts of one band, the final burst included
  orders     frozen: one order for every precursor of an interval, the tone band at 0, three
             different orders in a trial, the orders the session plan assigns, each of the 24
             equally often per track; reordered: a valid order in every repetition, tone band at 0
  noise      comodulated: every band's j-th burst is the same noise (identical baseband signal);
             random: independent; the noise is identical at the lowest and the highest tone level
  tone       only in the signal interval and only in the tone band's final burst, at the level
             asked; none in the rating sequences
  level      no peak above 0.95 at the lowest or the highest tone level, so no trial is ever
             redrawn and every listener hears the noise checked here (the peak over all levels is
             largest at one of the two ends: the maximum of |noise + a x tone| is convex in a)
  repeat     the same seed gives the same buffer (SHA-256); the per-band signals add up to it
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import ARRANGEMENTS, FLANK_SLOTS, PATTERNS, Condition, Config, validate
from .procedure import make_order, make_tracks
from .stimulus import build, sha256

PRE_S = 0.05                     # silence before the first interval (stimulus.build)


def planned(cfg: Config, seed: int, attempts=(0, 1)) -> List[dict]:
    """Every trial the session script can play, in the script's own terms (session.Experiment)."""
    from .session import order_plan
    plan = order_plan(cfg, seed)
    conds = cfg.condition_list()
    co0 = next(c for c in conds if c.envelope == "co" and c.dt_ms == 0)
    r0 = next(c for c in conds if c.envelope == "rand" and c.dt_ms == 0)
    out = []
    for i, (c, iv) in enumerate(((co0, 2), (r0, 3))):
        out.append(dict(phase="example", cond=c, levels=(80.0,), seed=(seed, 1, i, 0), n=3,
                        signal=iv, orders=None))
    for i, lvl in enumerate(cfg.practice_levels_db):
        out.append(dict(phase="practice", cond=co0 if i % 2 == 0 else r0, levels=(lvl,),
                        seed=(seed, 2, i, 0), n=3, signal=None, orders=None))
    for t in make_tracks(cfg):
        for k in range(cfg.trials_per_track):
            for a in attempts:
                out.append(dict(phase="main", cond=t.cond, levels=(cfg.min_db, cfg.max_db),
                                seed=(seed, 6, t.id, k, a), n=3, signal=None,
                                orders=plan[t.id][k] if t.id in plan else None,
                                track=t.id, k=k, attempt=a))
    items = [(a, d) for a, d in cfg.ratings for _ in range(cfg.rating_reps)]
    rng = np.random.default_rng([seed, 77])
    for j, idx in enumerate(rng.permutation(len(items))):
        a, d = items[idx]
        out.append(dict(phase="rating", cond=Condition("co", a, float(d)), levels=(None,),
                        seed=(seed, 8, j, 0), n=1, signal=None, orders=None))
    return out


def schedule(cfg: Config, tr) -> Dict[int, List[int]]:
    """Burst start samples by band, derived from the definition and the trial's logged slots."""
    sr = cfg.sample_rate
    fixed = tr.arrangement not in PATTERNS
    if cfg.lead_ms is not None:
        lead = cfg.lead_ms / 1000
    else:
        off = cfg.offsets_ms(tr.arrangement, tr.dt_ms)
        lead = -min(off) / 1000 if min(off) < 0 else 0.0
    span = lead + cfg.n_precursors * cfg.noi_ms / 1000 + cfg.burst_ms / 1000
    bands = [tr.tone_band] if tr.envelope == "ref" else list(range(len(cfg.bands_hz)))
    out: Dict[int, List[int]] = {b: [] for b in bands}
    for iv in range(tr.n_intervals):
        t0 = PRE_S + iv * (span + cfg.isi_ms / 1000) + lead
        for j in range(cfg.n_precursors + 1):
            for b in bands:
                if j == cfg.n_precursors:
                    on = t0 + cfg.n_precursors * cfg.noi_ms / 1000
                else:
                    sl = ARRANGEMENTS[tr.arrangement][b] if fixed else tr.slots[iv][j][b]
                    on = t0 + j * cfg.noi_ms / 1000 + sl * tr.dt_ms / 1000
                out[b].append(int(round(on * sr)))
    return out


def check_trial(cfg: Config, item: dict, amplitude=None, tol_comod: float = 0.02) -> dict:
    """Render one planned trial at each of its levels and check it. Returns the measurements and
    a list of problems (empty = fine)."""
    from scipy.signal import hilbert
    sr, nb = cfg.sample_rate, int(round(cfg.burst_ms / 1000 * cfg.sample_rate))
    amp = amplitude or (lambda f, db: cfg.uncal_peak_60db * 10 ** ((db - 60.0) / 20))
    probs: List[str] = []
    renders = []
    for lvl in item["levels"]:
        tr, buf, stem = build(cfg, item["cond"], lvl, item["seed"], phase=item["phase"],
                              n_intervals=item["n"], signal_interval=item["signal"],
                              amplitude=amplitude, orders=item["orders"], stems=True)
        renders.append((lvl, tr, buf, stem))
        if tuple(tr.seed) != tuple(item["seed"]):
            probs.append(f"redrawn at level {lvl} (peak over 0.95)")
        if float(np.abs(buf).max()) > 0.95:
            probs.append(f"peak {np.abs(buf).max():.2f} at level {lvl}")
        if not all(np.array_equal(buf[:, 0], buf[:, c]) for c in range(buf.shape[1])):
            probs.append("channels differ")
        if np.abs(stem.sum(0) - buf[:, 0]).max() > 1e-6:
            probs.append("the per-band signals do not add up to the buffer")
    lvl, tr, buf, stem = renders[-1]
    # the noise must not depend on the tone level
    if len(renders) > 1 and not np.array_equal(renders[0][3][:-1], renders[-1][3][:-1]):
        probs.append("the noise differs between tone levels")
    # timing: every band sounds exactly inside its scheduled bursts and nowhere else
    sch = schedule(cfg, tr)
    n_bands = len(cfg.bands_hz)
    for b in range(n_bands):
        if b not in sch:
            if np.any(stem[b]):
                probs.append(f"band {b} sounds in band-alone")
            continue
        mask = np.zeros(stem.shape[1], bool)
        for i0 in sch[b]:
            mask[i0:i0 + nb] = True
            if not np.any(stem[b, i0:i0 + nb]):
                probs.append(f"band {b}: no burst at sample {i0}")
        if np.any(stem[b, ~mask]):
            probs.append(f"band {b}: sound outside its scheduled bursts")
        starts = sorted(sch[b])
        gaps = [(s2 - (s1 + nb)) / sr * 1000 for s1, s2 in zip(starts, starts[1:])]
        per = cfg.n_precursors + 1
        within = [g for k, g in enumerate(gaps) if (k + 1) % per != 0]   # not across intervals
        if within and min(within) < cfg.min_gap_ms - 1e-6:
            probs.append(f"band {b}: only {min(within):.1f} ms between two of its bursts")
    # the final burst: synchronous in every band, in every interval
    per = cfg.n_precursors + 1
    for iv in range(tr.n_intervals):
        finals = {sch[b][iv * per + per - 1] for b in sch}
        if len(finals) != 1 or int(round(tr.target_on_s[iv] * sr)) not in finals:
            probs.append(f"interval {iv + 1}: final burst not synchronous")
    # orders
    if tr.arrangement in PATTERNS:
        flank = sorted(FLANK_SLOTS)
        for iv, reps in enumerate(tr.slots):
            for r in reps:
                if r[tr.tone_band] != 0 or sorted(v for b, v in enumerate(r) if b != tr.tone_band) != flank:
                    probs.append(f"interval {iv + 1}: invalid order {r}")
            if tr.arrangement == "frozen" and any(r != reps[0] for r in reps):
                probs.append(f"interval {iv + 1}: frozen order changes between repetitions")
            if tr.arrangement == "reordered" and any(a == b for a, b in zip(reps, reps[1:])):
                probs.append(f"interval {iv + 1}: re-ordered repetition repeats the previous order")
        if tr.arrangement == "frozen":
            firsts = [tuple(reps[0]) for reps in tr.slots]
            if len(set(firsts)) != len(firsts):
                probs.append("two intervals of a trial share an order")
            if item["orders"] is not None and [list(x) for x in firsts] != [list(o) for o in item["orders"]]:
                probs.append("orders differ from the session plan")
    elif tr.slots:
        probs.append("a fixed arrangement logged pattern slots")
    # noise: comodulated bursts are the same noise in every band, random ones are not
    t = np.arange(nb) / sr
    diffs = []
    if tr.envelope in ("co", "rand"):
        for iv in range(tr.n_intervals):
            for j in range(per):
                base = []
                for b in range(n_bands):
                    i0 = sch[b][iv * per + j]
                    seg = stem[b, i0:i0 + nb]
                    y = hilbert(seg) * np.exp(-2j * np.pi * cfg.bands_hz[b] * t)
                    base.append(y / np.sqrt(np.mean(seg ** 2)))
                ref = base[tr.tone_band]
                diffs.append(max(np.abs(y - ref).max() / np.abs(ref).max() for y in base))
        worst = max(diffs)
        if tr.envelope == "co" and worst > tol_comod:
            probs.append(f"comodulated bands differ by {worst:.3f} of the burst")
        if tr.envelope == "rand" and np.median(diffs) < 0.5:
            probs.append("random bands look comodulated")
    # the tone: only in the signal interval's tone-band final burst, at the level asked
    tone = stem[-1]
    if lvl is None or tr.signal_interval == 0:
        if np.any(tone):
            probs.append("a tone in a sequence that should have none")
    else:
        iv = tr.signal_interval - 1
        i0 = int(round(tr.target_on_s[iv] * sr))
        inside = np.zeros_like(tone, bool)
        inside[i0:i0 + nb] = True
        if np.any(tone[~inside]):
            probs.append("tone outside the signal interval's final burst")
        want = amp(tr.tone_hz, lvl)
        if abs(np.abs(tone).max() / want - 1) > 0.01:
            probs.append(f"tone peak {np.abs(tone).max():.4f}, expected {want:.4f}")
        if abs(tr.tone_hz - cfg.bands_hz[tr.tone_band]) > 1e-9:
            probs.append("the tone is not at the tone band's centre")
    # repeat: the same seed gives the same sound
    again = build(cfg, item["cond"], lvl, item["seed"], phase=item["phase"], n_intervals=item["n"],
                  signal_interval=item["signal"], amplitude=amplitude, orders=item["orders"])
    if sha256(again[1]) != sha256(buf):
        probs.append("not reproducible from its seed")
    return {"problems": probs, "n_samples": buf.shape[0], "peak": float(np.abs(buf).max()),
            "peaks": [float(np.abs(r[2]).max()) for r in renders],
            "tone_band_starts": tuple(sch[tr.tone_band]), "slots": tr.slots,
            "signal_interval": tr.signal_interval, "comod_worst": max(diffs) if diffs else None,
            "window_s": [round(b - a, 6) for a, b in zip(tr.interval_on_s, tr.interval_off_s)]}


def audit(cfg: Config, seed: int, amplitude=None, attempts=(0, 1), progress=None) -> dict:
    """Check every planned trial of the session with this seed. See the module docstring."""
    validate(cfg)
    items = planned(cfg, seed, attempts)
    results, problems = [], []
    for n, it in enumerate(items):
        r = check_trial(cfg, it, amplitude)
        results.append((it, r))
        for p in r["problems"]:
            problems.append(f"{it['phase']} {it['cond'].name} seed {'-'.join(map(str, it['seed']))}: {p}")
        if progress and (n + 1) % 50 == 0:
            progress(n + 1, len(items))
    main = [(it, r) for it, r in results if it["phase"] == "main"]
    # one trial length, one interval window, one tone-band schedule in every main trial
    lengths = {r["n_samples"] for _, r in main}
    windows = {tuple(r["window_s"]) for _, r in main}
    tone_sched = {r["tone_band_starts"] for _, r in main}
    if len(lengths) != 1:
        problems.append(f"main trials have {len(lengths)} different lengths")
    if cfg.lead_ms is not None and len(windows) != 1:
        problems.append("interval windows differ between conditions")
    if cfg.lead_ms is not None and len(tone_sched) != 1:
        problems.append("the tone band's bursts move between conditions")
    # orders over each frozen track: each of the 24 equally often
    balance = {}
    for t in make_tracks(cfg):
        if t.cond.arrangement != "frozen":
            continue
        cnt = Counter(tuple(reps[0]) for it, r in main if it["track"] == t.id and it["attempt"] == 0
                      for reps in r["slots"])
        balance[t.cond.name] = sorted(set(cnt.values()))
        if len(cnt) != len(cfg.orders()) or len(set(cnt.values())) != 1:
            problems.append(f"{t.cond.name}: orders not balanced ({dict(Counter(cnt.values()))})")
    # the interleaved order of the tracks
    order = make_order(cfg, seed)
    counts = Counter(order)
    if set(counts.values()) != {cfg.trials_per_track} or any(a == b for a, b in zip(order, order[1:])):
        problems.append("track order: unequal counts or a track twice in a row")
    # signal interval: about a third each in every track
    sig = Counter((it["cond"].name, r["signal_interval"]) for it, r in main if it["attempt"] == 0)
    by_phase = Counter(it["phase"] for it, _ in results)
    reordered = [r for it, r in main if it["cond"].arrangement == "reordered"]
    repeats = 0
    if reordered:
        pairs = [(a, b) for r in reordered for reps in r["slots"] for a, b in zip(reps, reps[1:])]
        repeats = sum(a == b for a, b in pairs) / max(len(pairs), 1)
    return {
        "ok": not problems, "seed": int(seed), "config": cfg.name, "config_hash": cfg.hash(),
        "amplitude": "calibrated" if amplitude else "uncalibrated",
        "trials_checked": dict(by_phase), "attempts": list(attempts),
        "problems": problems[:200], "n_problems": len(problems),
        "trial_samples": sorted(lengths), "interval_window_s": sorted(windows)[0] if windows else None,
        "peak_max": max(max(r["peaks"]) for _, r in results),
        "comod_worst": max((r["comod_worst"] for it, r in results
                            if it["cond"].envelope == "co" and r["comod_worst"] is not None), default=None),
        "order_balance": balance, "reordered_repeat_share": repeats,
        "signal_interval_counts": {f"{k[0]}@{k[1]}": v for k, v in sorted(sig.items())},
    }


def describe(rep: dict) -> str:
    L = [f"seed {rep['seed']}  {rep['config']} ({rep['config_hash']}), {rep['amplitude']} levels",
         "  trials checked: " + ", ".join(f"{k} {v}" for k, v in rep["trials_checked"].items())
         + f"  (main trials at attempts {rep['attempts']}, each at the lowest and highest tone level)",
         f"  one trial length: {rep['trial_samples']} samples; interval window {rep['interval_window_s']} s",
         f"  tallest peak {rep['peak_max']:.3f} (limit 0.95, so no trial is redrawn)",
         f"  comodulated bands: largest difference between bands {rep['comod_worst']:.2e} of a burst"
         if rep["comod_worst"] is not None else "  no comodulated trials",
         "  frozen orders per track: " + (", ".join(f"{k} each order x{v}" for k, v in rep["order_balance"].items())
                                         or "none"),
         f"  re-ordered: the same order twice in a row in {100 * rep['reordered_repeat_share']:.1f}% of repetitions"]
    if rep["ok"]:
        L.append("  ALL CHECKS PASSED")
    else:
        L.append(f"  {rep['n_problems']} PROBLEM(S):")
        L += [f"    - {p}" for p in rep["problems"][:30]]
    return "\n".join(L)


def report_path(data: Path, cfg: Config, seed: int) -> Path:
    return Path(data) / "checks" / f"tcmr_{cfg.hash()}_{int(seed)}.json"


def save(rep: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep, indent=1, default=float))
    return path
