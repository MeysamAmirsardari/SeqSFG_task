"""Figures: one per session (every track, the curve) and one per group (the curve beside Micheyl's)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .logbook import read_session  # noqa: E402

COLORS = {"D0": "#1b4f9c", "D35": "#2f7fc1", "D50": "#3aa0a0", "D71": "#d08a2e", "D100": "#c4452f",
          "D0_late": "#7a5bbf"}
# Micheyl et al. (2013) Exp. 3, group posterior-mode d'. 0 and 100 ms are given in their text;
# 35, 50 and 71 ms are read from their Figure 7 by eye (about +-0.05).
MICHEYL_EXP3 = {0: 1.48, 35: 1.85, 50: 2.10, 71: 2.35, 100: 2.43}


def _logy(ax) -> None:
    """Log axis with plain numbers (1, 2, 5, 10, ...) instead of 2x10^0."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def session_figure(folder: Path, summ: dict, out: Optional[Path] = None) -> Path:
    folder = Path(folder)
    s = read_session(folder)
    rows = [t["row"] for t in s["trials"] if t["row"]["phase"] == "main"
            and t["row"]["status"] == "ok"]
    blocks = sorted({int(r["block"]) for r in rows})
    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(2, max(4, (len(blocks) + 1) // 2 + 2), wspace=0.35, hspace=0.45)
    ncol = (len(blocks) + 1) // 2
    for i, b in enumerate(blocks):
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        rs = [r for r in rows if int(r["block"]) == b]
        name = rs[0]["condition"]
        st = [r for r in rs if not r["is_catch"]]
        x = np.arange(len(st))
        y = [r["delta_ms"] for r in st]
        ax.plot(x, y, "-", color=COLORS.get(name, "k"), lw=1.2)
        ok = np.array([bool(r["correct"]) for r in st])
        ax.scatter(x[ok], np.array(y)[ok], s=12, color=COLORS.get(name, "k"), zorder=3)
        ax.scatter(x[~ok], np.array(y)[~ok], s=14, facecolor="white",
                   edgecolor=COLORS.get(name, "k"), zorder=3)
        rev = [j for j, r in enumerate(st) if r["reversal"]]
        ax.scatter(rev, [y[j] for j in rev], s=40, facecolor="none", edgecolor="0.3", lw=0.8)
        th = next((t["threshold_ms"] for t in summ["tracks"] if t["block"] == b), None)
        if th:
            ax.axhline(th, color="0.4", ls="--", lw=0.8)
        _logy(ax)
        ax.set_title(f"part {b + 1}: {name}" + (f"  {th:.1f} ms" if th else ""), fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("trial", fontsize=8)
        if i % ncol == 0:
            ax.set_ylabel("delta (ms)", fontsize=8)
    ax = fig.add_subplot(gs[:, ncol:])
    conds = summ["conditions"]
    main = [n for n in conds if conds[n]["ending_shift_ms"] == 0]
    xs = [conds[n]["context_ms"] for n in main]
    ys = [conds[n]["threshold_ms"] or np.nan for n in main]
    ax.plot(xs, ys, "o-", color="#1b4f9c", lw=2, label="threshold (reversals)")
    for n in main:
        f = conds[n]["fit"]
        if f.get("threshold_ms") and f.get("ci95_ms"):
            ax.errorbar(conds[n]["context_ms"] + 1.5, f["threshold_ms"],
                        yerr=[[f["threshold_ms"] - f["ci95_ms"][0]], [f["ci95_ms"][1] - f["threshold_ms"]]],
                        fmt="s", ms=4, color="0.45", capsize=2, lw=1)
    if "D0_late" in conds and conds["D0_late"]["threshold_ms"]:
        ax.plot([-8], [conds["D0_late"]["threshold_ms"]], "D", color=COLORS["D0_late"],
                label="D0_late (control)")
    _logy(ax)
    ax.set_xlabel("context: A lags B by D (ms)")
    ax.set_ylabel("threshold (ms, 79.4% correct)")
    ax.set_xticks([0, 35, 50, 71, 100])
    ax.set_title(f"{summ['code']}: grouping curve" + ("  (simulated)" if summ["simulated"] else ""))
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(alpha=0.25, which="both")
    out = Path(out) if out else folder / "summary.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def group_figure(res: dict, out: Path) -> Path:
    conds = res["conditions"]
    main = [n for n in conds if n != "D0_late"]
    xs = [int(n[1:]) for n in main]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for code, d in res["individual"].items():
        a1.plot(xs, [d[n] or np.nan for n in main], "-", color="0.75", lw=0.9)
    gm = [conds[n]["geomean_ms"] or np.nan for n in main]
    se = [conds[n]["log_se"] or 0.0 for n in main]
    a1.errorbar(xs, gm, yerr=[np.array(gm) * (1 - np.exp(-np.array(se))),
                              np.array(gm) * (np.exp(np.array(se)) - 1)],
                fmt="o-", color="#1b4f9c", lw=2, capsize=3, label=f"geometric mean, n={res['n_listeners']}")
    if conds.get("D0_late", {}).get("geomean_ms"):
        a1.plot([-8], [conds["D0_late"]["geomean_ms"]], "D", color=COLORS["D0_late"],
                label="D0_late (control)")
    _logy(a1)
    a1.set_xticks([0, 35, 50, 71, 100])
    a1.set_xlabel("context: A lags B by D (ms)")
    a1.set_ylabel("asynchrony threshold (ms)")
    a1.legend(frameon=False, fontsize=8)
    a1.set_title("Grouping task: thresholds")
    g = [conds[n]["grouping_index_mean"] for n in main]
    gse = [conds[n]["grouping_index_se"] or 0.0 for n in main]
    a2.errorbar(xs, g, yerr=gse, fmt="o-", color="#1b4f9c", lw=2, capsize=3,
                label="grouping helps: this study (sync detection)")
    mx = sorted(MICHEYL_EXP3)
    m = np.array([MICHEYL_EXP3[k] for k in mx])
    a2.plot(mx, (m - m[0]) / (m[-1] - m[0]), "s--", color="#c4452f", lw=2,
            label="segregation helps: Micheyl et al. 2013, Exp. 3 (d')")
    a2.set_xticks([0, 35, 50, 71, 100])
    a2.set_xlabel("A lags B by D (ms)")
    a2.set_ylabel("performance, scaled 0 to 1 within each task")
    tops = [x + e for x, e in zip(g, gse) if x is not None]
    a2.set_ylim(-0.12, max([1.12] + [t + 0.06 for t in tops]))
    a2.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
    a2.set_title("Same sequences, opposite tasks")
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return Path(out)


def explained_figure(out: Path) -> Path:
    """The design in one figure: the endings at every context, a rendered ending, one trial."""
    from .config import Config
    from .levels import uncalibrated
    from .stimulus import build_trial, render, sequence_onsets_ms
    from .verify import measure_ending_onsets

    cfg = Config()
    fig = plt.figure(figsize=(13, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.38, wspace=0.22)
    ax = fig.add_subplot(gs[0, :])
    y = 0.0
    yt, yl = [], []
    for c in cfg.conditions():
        for label, d in (("standard", 0.0), ("target", 20.0)):
            b, a = sequence_onsets_ms(cfg, c, d, 1)
            for k in range(cfg.n_tones - 3, cfg.n_tones):
                hl = k == cfg.n_tones - 1
                ax.add_patch(plt.Rectangle((b[k], y + 0.42), cfg.tone_ms, 0.32,
                                           color="#2f6fd6", alpha=1 if hl else 0.45, lw=0))
                ax.add_patch(plt.Rectangle((a[k], y + 0.02), cfg.tone_ms, 0.32,
                                           color="#1aa391", alpha=1 if hl else 0.45, lw=0))
            yt.append(y + 0.38)
            yl.append(f"{c.name}  {label}")
            y -= 1.0
        y -= 0.35
    ax.axvline((cfg.n_tones - 1) * cfg.soa_ms, color="0.6", ls=":", lw=1)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=8)
    ax.set_xlim((cfg.n_tones - 3) * cfg.soa_ms - 20, (cfg.n_tones - 1) * cfg.soa_ms + 230)
    ax.set_ylim(y + 0.6, 1.2)
    ax.set_xlabel("time from the first tone of the sequence (ms); dotted: B's slot for the ending")
    ax.set_title("A: last two context pairs and the ending (blue B 1000 Hz, green A 420 Hz); "
                 "target shown with delta = 20 ms, B first", fontsize=10, loc="left")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    cal = uncalibrated(cfg)
    tr = build_trial(cfg, cfg.condition("D50"), 8.0, (3, 1, 4), target_interval=2, sign=1)
    buf, _ = render(cfg, tr, cal.amplitude)
    mb, ma = measure_ending_onsets(cfg, buf, tr)
    ax2 = fig.add_subplot(gs[1, 0])
    t0 = tr.ending_b_s[1] - 0.012
    i0, i1 = int(t0 * cfg.sample_rate), int((t0 + 0.06) * cfg.sample_rate)
    tt = (np.arange(i0, i1) / cfg.sample_rate - t0) * 1000.0
    ax2.plot(tt, buf[i0:i1, 0], color="0.25", lw=0.6)
    for v, col_, lab in ((mb[1], "#2f6fd6", "B onset"), (ma[1], "#1aa391", "A onset")):
        ax2.axvline((v - t0) * 1000.0 - 10.0, color=col_, lw=1.2, label=lab + " (measured)")
    ax2.set_xlabel("ms")
    ax2.set_yticks([])
    ax2.legend(fontsize=8, frameon=False, loc="upper left")
    ax2.set_title(f"B: the rendered target ending, D50, delta = 8 ms; measured "
                  f"{((ma[1] - mb[1]) - (ma[0] - mb[0])) * 1000:.3f} ms", fontsize=10, loc="left")

    ax3 = fig.add_subplot(gs[1, 1])
    segs = [("ready", 0, cfg.ready_ms / 1000, "0.8"),
            ("sequence 1", None, None, "#2f6fd6"), ("gap 450-550", None, None, "0.85"),
            ("sequence 2", None, None, "#2f6fd6"), ("answer", None, None, "#e0a33a"),
            ("feedback", None, None, "#35a96e"), ("blank", None, None, "0.8")]
    seq = tr.interval_off_s[0] - tr.interval_on_s[0]
    durs = [cfg.ready_ms / 1000, seq, tr.isi_ms / 1000, seq, 0.7, cfg.feedback_ms / 1000,
            cfg.blank_ms / 1000]
    x = 0.0
    for i, ((lab, _, _, col_), dur) in enumerate(zip(segs, durs)):
        ax3.barh(0, dur, left=x, color=col_, height=0.5)
        ax3.text(x + dur / 2, 0.32 + (0.22 if i % 2 else 0.0), lab, ha="center", va="bottom",
                 fontsize=7.5)
        x += dur
    ax3.set_xlim(0, x)
    ax3.set_ylim(-0.5, 1.0)
    ax3.set_yticks([])
    ax3.set_xlabel("seconds (answer time is the listener's)")
    ax3.set_title(f"C: one trial, about {x:.1f} s", fontsize=10, loc="left")
    for s in ("top", "right", "left"):
        ax3.spines[s].set_visible(False)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return Path(out)
