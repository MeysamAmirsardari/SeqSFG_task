"""d' per delay, drawn like Micheyl et al. (2013), Fig. 7, so the two can sit side by side.

Point estimates: d' = z(H) - z(F) and c = -(z(H) + z(F)) / 2 (equal-variance Gaussian model, as
theirs), with the log-linear correction (0.5 added to each count, 1 to each total).

Posteriors, for the figure: their boxes are the 75% posterior interval of d', the whiskers the 95%
interval and the thick line the posterior mode (MAP), from a hierarchical Bayesian probit model.
Here, for each listener and delay, the hit and false-alarm rates get flat Beta(1, 1) priors and
d' = z(H) - z(F) is sampled from their posteriors; the group posterior is the average of the
listeners' samples, draw by draw (the mean of these listeners, without their model's shrinkage
towards the group). Same quantities, simpler model.

Their group values (`MICHEYL_EXP3_FIG7`) are read off their Fig. 7 by eye, to about 0.1; the two
ends agree with their text (mean d' 1.48 at 0 ms and 2.43 at 100 ms).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats

from tsync.logbook import read_jsonl

from .config import Config

MICHEYL_EXP3 = {0.0: 1.48, 100.0: 2.43}     # group mean d' in their text (doi 10.1037/a0032241)
# delay: (mode, 75% interval, 95% interval), group, read off their Fig. 7 (to about 0.1)
MICHEYL_EXP3_FIG7 = {0.0: (1.5, (1.3, 1.85), (0.85, 2.15)),
                     35.0: (1.9, (1.75, 2.1), (1.45, 2.4)),
                     50.0: (2.2, (2.05, 2.4), (1.75, 2.65)),
                     71.0: (2.4, (2.25, 2.6), (1.95, 2.9)),
                     100.0: (2.5, (2.3, 2.8), (1.85, 3.25))}
N_DRAWS = 20000


def sdt(hits: int, n_signal: int, fas: int, n_noise: int) -> Dict[str, float]:
    h = (hits + 0.5) / (n_signal + 1.0)
    f = (fas + 0.5) / (n_noise + 1.0)
    zh, zf = stats.norm.ppf(h), stats.norm.ppf(f)
    return {"hit_rate": hits / max(n_signal, 1), "fa_rate": fas / max(n_noise, 1),
            "d_prime": float(zh - zf), "c": float(-(zh + zf) / 2),
            "hits": hits, "false_alarms": fas, "n_signal": n_signal, "n_noise": n_noise}


def dprime_draws(hits: int, n_signal: int, fas: int, n_noise: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = rng.beta(hits + 1, n_signal - hits + 1, N_DRAWS)
    f = rng.beta(fas + 1, n_noise - fas + 1, N_DRAWS)
    return stats.norm.ppf(h) - stats.norm.ppf(f)


def quantiles(x: np.ndarray) -> Dict[str, float]:
    """The posterior mode (MAP, from a kernel density estimate), and the 75% and 95% intervals."""
    grid = np.linspace(np.percentile(x, 0.5), np.percentile(x, 99.5), 600)
    mode = float(grid[np.argmax(stats.gaussian_kde(x)(grid))])
    q = np.percentile(x, [2.5, 12.5, 87.5, 97.5])
    return {"mode": mode, "lo95": float(q[0]), "lo75": float(q[1]), "hi75": float(q[2]),
            "hi95": float(q[3])}


def summarize(folder: Path) -> dict:
    folder = Path(folder)
    meta = json.loads((folder / "session.json").read_text())
    cfg = Config.from_dict(meta["config"])
    rows = [r["row"] for r in read_jsonl(folder / "trials.jsonl")]
    main = [r for r in rows if r["phase"] == "main" and r["status"] == "ok"]
    out_d, per_block = {}, {}
    for i, d in enumerate(cfg.delays_ms):
        rs = [r for r in main if abs(r["delay_ms"] - d) < 1e-9]
        sig = [r for r in rs if r["target_present"]]
        noi = [r for r in rs if not r["target_present"]]
        v = sdt(sum(bool(r["said_yes"]) for r in sig), len(sig),
                sum(bool(r["said_yes"]) for r in noi), len(noi))
        if v["n_signal"] and v["n_noise"]:
            v["posterior"] = quantiles(dprime_draws(v["hits"], v["n_signal"], v["false_alarms"],
                                                    v["n_noise"], seed=i))
        out_d[f"{d:g}"] = v
    for b in sorted({r["block"] for r in main}):
        rs = [r for r in main if r["block"] == b]
        sig = [r for r in rs if r["target_present"]]
        noi = [r for r in rs if not r["target_present"]]
        per_block[str(b)] = dict(sdt(sum(bool(r["said_yes"]) for r in sig), len(sig),
                                     sum(bool(r["said_yes"]) for r in noi), len(noi)),
                                 delay_ms=rs[0]["delay_ms"])
    xs = np.array([float(d) for d in out_d])
    ys = np.array([v["d_prime"] for v in out_d.values()])
    slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) > 1 else None
    prac = [r for r in rows if r["phase"] == "practice" and r["status"] == "ok"]
    flags = []
    vols = sorted({r["volume"] for r in rows if r["status"] == "ok" and r["volume"] is not None})
    if len(vols) > 1:
        flags.append(f"system volume varied: {vols}")
    under = sum(int(r["underflows"] or 0) for r in rows if r["status"] == "ok")
    if under:
        flags.append(f"{under} audio underflows in valid trials")
    rep = {}
    for r in rows:
        if r["status"] != "ok":
            rep[r["reason"]] = rep.get(r["reason"], 0) + 1
    out = {"code": meta.get("code"), "status": meta.get("status"), "levels": cfg.levels,
           "by_delay": out_d, "by_block": per_block,
           "d_prime_slope_per_100ms": None if slope is None else slope * 100,
           "practice_correct": f"{sum(bool(r['correct']) for r in prac)}/{len(prac)}",
           "repeated": rep, "volumes": vols, "flags": flags,
           "duration_min": rows[-1]["t_s"] / 60 if rows else None}
    (folder / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    with open(folder / "dprime.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["code", "levels", "delay_ms", "d_prime", "c", "hit_rate", "fa_rate",
                    "n_signal", "n_noise", "map", "lo75", "hi75", "lo95", "hi95"])
        for d, v in out_d.items():
            p = v.get("posterior", {})
            w.writerow([out["code"], cfg.levels, d, v["d_prime"], v["c"], v["hit_rate"],
                        v["fa_rate"], v["n_signal"], v["n_noise"], p.get("mode"), p.get("lo75"),
                        p.get("hi75"), p.get("lo95"), p.get("hi95")])
    return out


def describe(s: dict) -> str:
    L = [f"{s['code']}  status {s['status']}  levels {s['levels']}",
         "  delay    d'   (MAP, 95% interval)      c     hits   false alarms   trials"]
    for d, v in s["by_delay"].items():
        p = v.get("posterior")
        pp = f"({p['mode']:4.2f}, {p['lo95']:4.2f} to {p['hi95']:4.2f})" if p else ""
        L.append(f"  {float(d):5g}   {v['d_prime']:5.2f}  {pp:<24} {v['c']:+5.2f}   {v['hit_rate']:4.0%}   "
                 f"{v['fa_rate']:4.0%}           {v['n_signal'] + v['n_noise']}")
    if s["d_prime_slope_per_100ms"] is not None:
        L.append(f"  d' change per 100 ms of delay (linear fit): {s['d_prime_slope_per_100ms']:+.2f}"
                 f"   (Micheyl et al. 2013, Exp. 3: about +0.95, from 1.48 to 2.43)")
    L.append(f"  practice {s['practice_correct']}; repeated {s['repeated'] or 'none'}; "
             f"volumes {s['volumes']}")
    if s["duration_min"]:
        L.append(f"  duration {s['duration_min']:.1f} min")
    L += [f"  ! {x}" for x in s["flags"]]
    return "\n".join(L)


# ---------------------------------------------------------------------------------- figures
def _panel(ax, delays: Sequence[float], boxes: Dict[float, tuple], label: str, ylim=(-0.35, 5.45)):
    """Their Fig. 7 style: white box = 75% interval, thick line = mode, dashed whiskers with
    caps = 95% interval; the delays evenly spaced; a dashed line at d' = 0."""
    for i, d in enumerate(delays):
        if d not in boxes:
            continue
        mode, (l75, h75), (l95, h95) = boxes[d]
        w = 0.6
        ax.add_patch(__import__("matplotlib").patches.Rectangle((i - w / 2, l75), w, h75 - l75,
                                                               fill=False, lw=1.0, ec="k"))
        ax.plot([i - w / 2, i + w / 2], [mode, mode], color="k", lw=3.2, solid_capstyle="butt")
        for a, b in ((h75, h95), (l75, l95)):
            ax.plot([i, i], [a, b], color="k", lw=0.9, ls=(0, (3, 2)))
            ax.plot([i - w / 4, i + w / 4], [b, b], color="k", lw=0.9)
    ax.axhline(0, color="k", lw=0.8, ls=(0, (4, 3)))
    ax.set_xlim(-0.6, len(delays) - 0.4)
    ax.set_ylim(*ylim)
    ax.set_xticks(range(len(delays)))
    ax.set_xticklabels([f"{d:g}" for d in delays])
    ax.set_yticks(np.arange(0.0, 5.01, 1.0))
    ax.set_yticklabels([f"{v:.1f}" for v in np.arange(0.0, 5.01, 1.0)])
    ax.text(0.03, 0.95, label, transform=ax.transAxes, va="top", fontsize=10)


def _boxes(s: dict) -> Dict[float, tuple]:
    out = {}
    for d, v in s["by_delay"].items():
        p = v.get("posterior")
        if p:
            out[float(d)] = (p["mode"], (p["lo75"], p["hi75"]), (p["lo95"], p["hi95"]))
    return out


def figure(folder: Path, s: dict, out: Path = None) -> Path:
    """One session: their group (left, read off their Fig. 7) beside this listener (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(folder)
    out = out or folder / "summary.png"
    delays = [float(d) for d in s["by_delay"]]
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    _panel(axs[0], delays, MICHEYL_EXP3_FIG7, "Micheyl et al. 2013, Exp. 3: group")
    ours = "A and B roved together" if s["levels"] == "common" else "A roved alone (their rule)"
    _panel(axs[1], delays, _boxes(s), f"{s['code']}: {ours}")
    axs[0].set_ylabel("d'")
    for ax in axs:
        ax.set_xlabel("Asynchrony (ms)")
    fig.text(0.01, 0.005, "Boxes 75% and whiskers 95% posterior intervals, thick line the mode. "
             "Left: values read off their Fig. 7 (about 0.1). Right: flat-prior binomial model.",
             fontsize=6.5, color="0.35")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def group_figure(folders: Sequence[Path], out: Path) -> Path:
    """Several sessions in their Fig. 7 layout: one panel per listener, then the group, two
    columns, d' from 0 to 5."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sums = [summarize(Path(f)) for f in folders]
    delays = [float(d) for d in sums[0]["by_delay"]]
    draws: Dict[float, List[np.ndarray]] = {d: [] for d in delays}
    for s in sums:
        for i, (d, v) in enumerate(s["by_delay"].items()):
            draws[float(d)].append(dprime_draws(v["hits"], v["n_signal"], v["false_alarms"],
                                                v["n_noise"], seed=i))
    grp = {}
    for d, xs in draws.items():
        q = quantiles(np.mean(xs, axis=0))
        grp[d] = (q["mode"], (q["lo75"], q["hi75"]), (q["lo95"], q["hi95"]))
    panels = [(s["code"], _boxes(s)) for s in sums] + [("Group", grp)]
    n = len(panels)
    rows = (n + 1) // 2
    fig, axs = plt.subplots(rows, 2, figsize=(7.2, 2.15 * rows), sharex=True, sharey=True,
                            squeeze=False)
    for k, ax in enumerate(axs.flat):
        if k < n:
            _panel(ax, delays, panels[k][1], panels[k][0])
        else:
            ax.axis("off")
        if k % 2 == 0:
            ax.set_ylabel("d'")
        else:                                           # their figure labels the right side too
            ax.yaxis.tick_right()
            ax.tick_params(labelright=True)
    for ax in axs[-1]:
        ax.set_xlabel("Asynchrony (ms)")
    fig.subplots_adjust(hspace=0, wspace=0, left=0.11, right=0.89, top=0.98,
                        bottom=0.55 / (2.15 * rows) + 0.03)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["listener", "delay_ms", "map", "lo75", "hi75", "lo95", "hi95"])
        for name, boxes in panels:
            for d, (m, (l75, h75), (l95, h95)) in boxes.items():
                w.writerow([name, d, m, l75, h75, l95, h95])
    return out
