"""Every trial a session will play, drawn from its seed and checked before anyone listens.

Nothing in a tinc session depends on the listener's answers, so the whole session is known from
its seed: the examples, the practice, every trial of every block (and its first repeat after a
pause). Each is rendered and checked against the definition, from the sound itself:

  timing   the B and A signals are non-zero only inside their tones, to the sample: B tone k at
           k x SOA, A tone k at k x SOA + D, every tone 100 ms
  levels   every tone's level measured from its samples matches its intended level (0.01 dB);
           common: each A tone has its B partner's rove, decoys raise both tones of a pair;
           independent: B tones fixed, A roved alone, decoys on A; roves within +-rove_db;
           one or two decoys
  change   a change trial has exactly one B tone raised by target_db relative to the rule, never
           among the first target_from; a no-change trial has none; half of every block changes
  level    no peak above max_peak
  repeat   the same seed gives the same buffer (SHA-256); one buffer length for all trials
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np

from .config import Config, validate
from .session import block_plan
from .stimulus import build, sha256


def planned(cfg: Config, seed: int, attempts=(0, 1)) -> List[dict]:
    out = []
    for i, (d, p) in enumerate(cfg.examples):
        out.append(dict(phase="example", delay=float(d), present=bool(p), seed=(seed, 1, i, 0), block=-1))
    rng = np.random.default_rng([seed, 2])
    j = 0
    for item in cfg.practice:
        d, n = item[0], item[1]
        tdb = float(item[2]) if len(item) == 3 else None
        for p in rng.permutation([1] * (n // 2) + [0] * (n - n // 2)):
            out.append(dict(phase="practice", delay=float(d), present=bool(p), seed=(seed, 2, j, 0),
                            block=-1, target_db=tdb))
            j += 1
    for blk in block_plan(cfg, seed):
        for k, p in enumerate(blk["present"]):
            for a in attempts:
                out.append(dict(phase="main", delay=blk["delay_ms"], present=bool(p),
                                seed=(seed, 6, blk["block"], k, a), block=blk["block"], k=k, attempt=a))
    return out


def check_trial(cfg: Config, it: dict, amplitude=None) -> dict:
    sr = cfg.sample_rate
    m = int(round(cfg.tone_ms / 1000 * sr))
    amp = amplitude or (lambda f, db: cfg.uncal_peak_60db * 10 ** ((db - 60.0) / 20))
    probs = []
    tdb = it.get("target_db")
    want_change = cfg.target_db if tdb is None else tdb
    tr, buf, stem = build(cfg, it["delay"], it["present"], it["seed"], phase=it["phase"],
                          block=it["block"], amplitude=amplitude, stems=True, target_db=tdb)
    if np.abs(stem.sum(0) - buf[:, 0]).max() > 1e-6:
        probs.append("the B and A signals do not add up to the buffer")
    if float(np.abs(buf).max()) > cfg.max_peak:
        probs.append(f"peak {np.abs(buf).max():.3f}")
    pre = cfg.pre_ms / 1000
    env_ms = None
    meas = {"A": [], "B": []}
    for row, stream, f, off in ((0, "B", cfg.f_b_hz, 0.0), (1, "A", cfg.f_a_hz, it["delay"] / 1000)):
        mask = np.zeros(stem.shape[1], bool)
        for k in range(cfg.n_tones):
            i0 = int(round((pre + k * cfg.soa_ms / 1000 + off) * sr))
            seg = stem[row, i0:i0 + m]
            mask[i0:i0 + m] = True
            if not np.any(seg):
                probs.append(f"{stream} tone {k} missing at sample {i0}")
                meas[stream].append(float("nan"))
                continue
            tone = next(t for t in tr.tones if t.stream == stream and t.index == k)
            ref = amp(f, tone.level_db)
            t = np.arange(m) / sr
            dur, ramp = cfg.tone_ms / 1000, cfg.ramp_ms / 1000
            env = np.ones(m)
            on, of = t < ramp, t > dur - ramp
            env[on] = np.sin(0.5 * math.pi * t[on] / ramp) ** 2
            env[of] = np.sin(0.5 * math.pi * (dur - t[of]) / ramp) ** 2
            want = ref * env * np.sin(2 * math.pi * f * t + tone.phase_rad)
            got_db = 20 * math.log10(np.sqrt(np.mean(seg ** 2)) / np.sqrt(np.mean(want ** 2)))
            if abs(got_db) > 0.01 or np.abs(seg - want).max() > 1e-9:
                probs.append(f"{stream} tone {k}: off by {got_db:+.3f} dB from {tone.level_db:.2f} dB")
            meas[stream].append(tone.level_db + got_db)
        if np.any(stem[row, ~mask]):
            probs.append(f"{stream}: sound outside its tones")
    a, b = np.array(meas["A"]), np.array(meas["B"])
    L = cfg.level_db
    dec = np.zeros(cfg.n_tones, bool)
    dec[tr.decoys] = True
    if len(tr.decoys) not in cfg.decoy_counts:
        probs.append(f"{len(tr.decoys)} decoys")
    rove_a = a - L - cfg.decoy_db * dec
    if np.any(np.abs(rove_a) > cfg.rove_db + 1e-6):
        probs.append("an A rove outside the range")
    if cfg.levels == "common":
        extra = b - a                                   # 0 everywhere but the change
    else:
        extra = b - L                                   # B fixed at L but the change
    raised = [k for k in range(cfg.n_tones) if abs(extra[k]) > 1e-6]
    if it["present"]:
        if len(raised) != 1 or abs(extra[raised[0]] - want_change) > 1e-6 or raised[0] < cfg.target_from:
            probs.append(f"change trial: B tones raised {raised}")
    elif raised:
        probs.append(f"no-change trial: B tones raised {raised}")
    again = build(cfg, it["delay"], it["present"], it["seed"], phase=it["phase"], block=it["block"],
                  amplitude=amplitude, target_db=tdb)[1]
    if sha256(again) != sha256(buf):
        probs.append("not reproducible from its seed")
    return {"problems": probs, "n_samples": buf.shape[0], "peak": float(np.abs(buf).max()),
            "target_index": tr.target_index, "n_decoys": len(tr.decoys)}


def audit(cfg: Config, seed: int, amplitude=None, attempts=(0, 1), progress=None) -> dict:
    validate(cfg)
    items = planned(cfg, seed, attempts)
    results, problems = [], []
    for n, it in enumerate(items):
        r = check_trial(cfg, it, amplitude)
        results.append((it, r))
        problems += [f"{it['phase']} D{it['delay']:g} seed {'-'.join(map(str, it['seed']))}: {p}"
                     for p in r["problems"]]
        if progress and (n + 1) % 50 == 0:
            progress(n + 1, len(items))
    lengths = {r["n_samples"] for _, r in results}
    if len(lengths) != 1:
        problems.append(f"{len(lengths)} different buffer lengths")
    blocks = block_plan(cfg, seed)
    for blk in blocks:
        if sum(blk["present"]) * 2 != len(blk["present"]):
            problems.append(f"block {blk['block']}: {sum(blk['present'])} of {len(blk['present'])} change")
    per_delay = Counter(b["delay_ms"] for b in blocks)
    if set(per_delay.values()) != {cfg.blocks_per_delay} or set(per_delay) != set(map(float, cfg.delays_ms)):
        problems.append(f"blocks per delay: {dict(per_delay)}")
    order = [b["delay_ms"] for b in blocks]
    pos = {d: np.mean([i for i, x in enumerate(order) if x == d]) for d in per_delay}
    main = [r for it, r in results if it["phase"] == "main" and it["attempt"] == 0 and it["present"]]
    tpos = Counter(r["target_index"] for r in main)
    return {"ok": not problems, "seed": int(seed), "config": cfg.name, "config_hash": cfg.hash(),
            "levels": cfg.levels, "amplitude": "calibrated" if amplitude else "uncalibrated",
            "trials_checked": dict(Counter(it["phase"] for it, _ in results)),
            "attempts": list(attempts), "problems": problems[:200], "n_problems": len(problems),
            "buffer_samples": sorted(lengths), "peak_max": max(r["peak"] for _, r in results),
            "block_order_ms": order, "mean_block_position": {f"{k:g}": float(v) for k, v in pos.items()},
            "target_positions": {str(k): v for k, v in sorted(tpos.items())}}


def describe(rep: dict) -> str:
    L = [f"seed {rep['seed']}  {rep['config']} ({rep['config_hash']}), levels {rep['levels']}, "
         f"{rep['amplitude']} amplitudes",
         "  trials checked: " + ", ".join(f"{k} {v}" for k, v in rep["trials_checked"].items())
         + f"  (main trials at attempts {rep['attempts']})",
         f"  one buffer length: {rep['buffer_samples']} samples; tallest peak {rep['peak_max']:.3f}",
         "  block order (delay, ms): " + " ".join(f"{d:g}" for d in rep["block_order_ms"]),
         "  mean block position per delay: " + ", ".join(f"{k} ms {v:.1f}" for k, v in rep["mean_block_position"].items()),
         "  change positions (B tone index): " + ", ".join(f"{k}:{v}" for k, v in rep["target_positions"].items())]
    L.append("  ALL CHECKS PASSED" if rep["ok"] else f"  {rep['n_problems']} PROBLEM(S):")
    L += [f"    - {p}" for p in rep["problems"][:30]]
    return "\n".join(L)


def report_path(data: Path, cfg: Config, seed: int) -> Path:
    return Path(data) / "checks" / f"tinc_{cfg.hash()}_{int(seed)}.json"


def save(rep: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep, indent=1, default=float))
    return path
