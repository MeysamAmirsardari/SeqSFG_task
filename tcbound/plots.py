"""Figures. The shear axis runs from a chord (0 ms) on the left to an arpeggio on the right.

Colours are fixed across every figure: A (timbre kept) blue, B (own stream kept) orange, the
model grey, the target red. The model always has its own axis and its own label.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .config import Condition, Config, captor_onsets, conditions, onsets, srs_db, validate

C_A, C_B, C_MODEL, C_TARGET, C_OTHER = "#2c6fbb", "#e07b26", "#8a8a8a", "#c0392b", "#5d6d7e"


def _mpl():
    import matplotlib
    backend = matplotlib.get_backend().lower()
    if not any(k in backend for k in ("inline", "ipympl", "widget", "nbagg")):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlesize": 10,
                         "axes.titleweight": "bold", "axes.titlelocation": "left"})
    return plt


def schematic(cfg: Config, path: Optional[Path] = None):
    """The whole design on one page: the shear, the two rove types, the model, the ceilings."""
    plt = _mpl()
    from matplotlib.patches import Rectangle
    from .model import curve, predictions
    from .observer import CUE_LABELS, ceilings
    fig = plt.figure(figsize=(11, 8.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1.0, 1.0], hspace=0.55, wspace=0.28)

    # (a) one repetition at every step, and the capture arrangement
    ax = fig.add_subplot(gs[0, :])
    rows = [Condition(f"s{s:g}", "A", s) for s in cfg.steps_ms]
    rows.append(Condition("cap", "A", 0.0, "capture"))
    for i, c in enumerate(rows):
        y0 = (len(rows) - 1 - i) * 5.0
        o = onsets(cfg, c)[:2] - cfg.lead_ms
        co = captor_onsets(cfg, c)
        for rep in range(2):
            for k in range(cfg.n_tones):
                col = C_TARGET if k == cfg.target_index else C_OTHER
                ax.add_patch(Rectangle((o[rep, k], y0 + k), cfg.tone_ms, 0.8, color=col, lw=0))
            if co is not None and rep == 0:
                for k in cfg.others:
                    ax.add_patch(Rectangle((co[0, k] - cfg.lead_ms, y0 + k), cfg.tone_ms, 0.8,
                                           lw=0, alpha=0.4, fc=C_OTHER))
        lab = (f"{c.step_ms:g} ms" if c.kind == "figure"
               else "0 ms + captors")
        ax.text(-12, y0 + 1.9, lab, ha="right", va="center", fontsize=8.5)
    ax.set_xlim(-70, 2 * cfg.period_ms + 10)
    ax.set_ylim(-1, 5.0 * len(rows))
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("time within the sequence (ms); two of the five repetitions shown")
    ax.set_title("a  The shear: tone k starts k x step after tone 1. Target (2241 Hz) in red")
    ax.text(cfg.period_ms * 1.5 + 30, 1.5,
            "captors (pale): the other three tones\nalone, half-way between repetitions",
            fontsize=8, color=C_OTHER, va="center")

    # (b) the rove scheme: one example interval of each type
    rng = np.random.default_rng(4)
    from .stimulus import draw_latent, gains_from_latent
    for j, task in enumerate(("A", "B")):
        ax = fig.add_subplot(gs[1, j])
        lat = (0.0, 7.0, -5.0, 6.0, *([0.0] * (cfg.n_tones - 1)))
        g, _ = gains_from_latent(cfg, task, "figure", lat, 6.0, True)
        R = cfg.n_repeats
        x = np.arange(R)
        for k in range(cfg.n_tones):
            col = C_TARGET if k == cfg.target_index else C_OTHER
            off = (k - 1.5) * 0.17
            ax.bar(x + off, g[:, k] + 15, bottom=-15, width=0.15, color=col,
                   alpha=0.95 if k == cfg.target_index else 0.6)
        t = cfg.target_index
        col = C_A if task == "A" else C_B
        if task == "A":
            ax.annotate("", xy=(R - 1 + (t - 1.5) * 0.17, g[-1, t] + 0.3),
                        xytext=(R - 1 + 0.5, g[-1, cfg.others[0]] + 0.3),
                        arrowprops=dict(arrowstyle="<->", color=col, lw=1.6))
            msg = ("on the probe all four jump to one new level:\nthe target re the others "
                   "AT THAT MOMENT is exact,\nits own past is no guide")
        else:
            ax.annotate("", xy=(R - 1 + (t - 1.5) * 0.17, g[-1, t] + 0.3),
                        xytext=(R - 2 + (t - 1.5) * 0.17, g[-2, t] + 0.3),
                        arrowprops=dict(arrowstyle="<->", color=col, lw=1.6))
            msg = ("on the probe the other three jump and the target holds:\nthe target re "
                   "ITS OWN past is exact,\nthe chord's timbre is no guide")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{i + 1}" for i in range(R - 1)] + ["probe"])
        ax.set_ylim(-15, 27)
        ax.set_yticks([-10, 0, 10])
        ax.set_ylabel("level re nominal (dB)")
        ax.set_title(f"{'b' if task == 'A' else 'c'}  Type {task}: a signal interval, "
                     "increment 6 dB", color=col)
        ax.text(0.01, 0.98, msg, transform=ax.transAxes, fontsize=7.8, va="top")
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)

    # (d) the model's prediction
    ax = fig.add_subplot(gs[2, 0])
    p = predictions(cfg)
    grid = np.linspace(0, max(cfg.steps_ms), 129)
    for v, ls in (("halfwave", "-"), ("magnitude", "--")):
        ax.plot(grid, curve(cfg, grid, v), ls, color=C_MODEL, lw=1.6,
                label=f"{v} (half-way {p['variants'][v]['half_way_ms']:.0f} ms)")
        ax.plot(cfg.steps_ms, p["variants"][v]["curve"], "o", color=C_MODEL, ms=4)
    rc = p["variants"]["halfwave"]["capture_rho"]
    ax.axhline(rc, color=C_MODEL, lw=0.8, ls=":")
    ax.text(max(cfg.steps_ms) * 0.42, rc - 0.025, "captors at step 0 (half-wave)", ha="left",
            va="top", fontsize=7.5, color=C_MODEL)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("shear step (ms)")
    ax.set_ylabel("target-to-figure coherence")
    ax.set_title("d  Model: two readings, two bounds")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")

    # (e) ceilings
    ax = fig.add_subplot(gs[2, 1])
    cues = list(CUE_LABELS)
    ya = ceilings(cfg, Condition("A", "A", 0.0))["per_cue_db"]
    yb = ceilings(cfg, Condition("B", "B", 0.0))["per_cue_db"]
    y = np.arange(len(cues))
    for off, d, col, lab in ((-0.2, ya, C_A, "type A"), (0.2, yb, C_B, "type B")):
        vals = [min(d[c], 20) if np.isfinite(d[c]) else 20 for c in cues]
        ax.barh(y + off, vals, height=0.38, color=col, label=lab)
        for yy, v in zip(y + off, vals):
            if v == 0:
                ax.text(0.3, yy, "exact", va="center", fontsize=7, color=col)
    ax.set_yticks(y)
    ax.set_yticklabels([CUE_LABELS[c] for c in cues], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 21)
    ax.set_xlabel("increment an ideal observer of that cue alone needs (dB)")
    ax.set_title("e  Every shortcut needs more than 8 dB")
    ax.legend(frameon=False, fontsize=7.5, loc="center right")
    fig.suptitle("tcbound: how far can the figure be sheared before the target stops binding?",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def session_figure(sdir: Path, out: Path) -> Path:
    plt = _mpl()
    from .analysis import analyse_session
    from .model import curve
    a = analyse_session(sdir)
    cfg, res, c, wc = a["cfg"], a["results"], a["curve"], a["wcurve"]
    steps = c["steps_ms"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axs[0]
    for task, col in (("A", C_A), ("B", C_B)):
        y = [res[f"{task}_{s:g}".replace(".", "p")].threshold_srs for s in steps]
        y = [np.nan if v is None else v for v in y]
        ax.plot(steps, y, "o-", color=col, label=f"type {task}")
        lim = [res[f"{task}_{s:g}".replace(".", "p")].n_ceiling_limited for s in steps]
        for s, v, l in zip(steps, y, lim):
            if l:
                ax.plot([s], [v], marker="^", color=col, ms=9, mfc="white")
        ce = res[f"{task}_0"].ceiling_srs
        ax.axhline(ce, color=col, lw=0.8, ls=":")
    ax.axhline(cfg.srs_max_db, color="k", lw=0.6, ls="--")
    ax.text(steps[-1], cfg.srs_max_db + 0.3, "cap", ha="right", fontsize=7.5)
    xc = steps[-1] + 10
    for name, col, dx in (("A_cap", C_A, 0), ("B_cap", C_B, 0), ("B_alone", C_B, 8)):
        r = res.get(name)
        if r is not None and r.threshold_srs is not None:
            ax.plot([xc + dx], [r.threshold_srs], "s" if "cap" in name else "D", color=col,
                    mfc="white" if name == "B_alone" else col)
    ax.set_xticks(list(steps) + [xc, xc + 8])
    ax.set_xticklabels([f"{s:g}" for s in steps] + ["cap", "alone"])
    ax.set_xlabel("shear step (ms)")
    ax.set_ylabel("threshold, SRS (dB)")
    ax.set_title("Thresholds (dotted: shortcut ceilings)")
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1]
    ax.plot(steps, c["bi"], "o-", color="k")
    if c["capture"]:
        ax.plot([steps[0]], [c["capture"]["bi_capture"]], "s", color="k", mfc="white",
                label="captors")
        ax.legend(frameon=False, fontsize=8)
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xlabel("shear step (ms)")
    ax.set_ylabel("BI = B - A (dB)")
    ax.set_title("Binding index")
    ax = axs[2]
    ax.plot(steps, wc["w"], "o-", color="k", label="listener")
    if np.isfinite(wc["w_capture"]):
        ax.plot([steps[0]], [wc["w_capture"]], "s", color="k", mfc="white", label="captors")
    ax.set_ylim(-0.1, 1.15)
    ax.set_xlabel("shear step (ms)")
    ax.set_ylabel("timbre weight w")
    ax.set_title("Decision weights")
    ax2 = ax.twinx()
    grid = np.linspace(0, max(steps), 129)
    ax2.plot(grid, curve(cfg, grid, "halfwave"), color=C_MODEL, lw=1, ls="--")
    ax2.set_ylim(-0.1, 1.15)
    ax2.set_ylabel("model coherence (comparison only)", color=C_MODEL)
    ax2.tick_params(axis="y", colors=C_MODEL)
    ax2.spines["right"].set_visible(True)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(f"{a['code']}  session {a['session_index']}", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"tcbound_{a['code']}_session_{a['session_index']:02d}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def group_figure(g: dict, out: Path) -> Path:
    plt = _mpl()
    from .model import curve
    cfg, steps = g["cfg"], g["steps_ms"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    grid = np.linspace(0, max(steps), 129)
    ax = axs[0]
    for code, L in g["listeners"].items():
        ax.plot(steps, L["curve"]["bi"], "-", color="k", lw=0.7, alpha=0.35)
    ax.errorbar(steps, g["mean_bi"], yerr=g["sem_bi"], fmt="o-", color="k", lw=1.8, capsize=3)
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xlabel("shear step (ms)")
    ax.set_ylabel("BI = B - A (dB)")
    ax.set_title(f"Binding index, {len(g['listeners'])} listeners")
    ax = axs[1]
    for code, L in g["listeners"].items():
        ax.plot(steps, L["wcurve"]["w"], "-", color="k", lw=0.7, alpha=0.35)
    ax.errorbar(steps, g["mean_w"], yerr=g["sem_w"], fmt="o-", color="k", lw=1.8, capsize=3)
    if g["s50_ms"] is not None:
        ax.axvline(g["s50_ms"], color=C_TARGET, lw=1)
        if g["s50_ci"]:
            ax.axvspan(*g["s50_ci"], color=C_TARGET, alpha=0.12, lw=0)
        ax.text(g["s50_ms"] + 1, 1.08, f"bound {g['s50_ms']:.0f} ms", color=C_TARGET,
                fontsize=8)
    ax.set_ylim(-0.1, 1.15)
    ax.set_xlabel("shear step (ms)")
    ax.set_ylabel("timbre weight w")
    ax.set_title("Decision weights and the bound")
    ax = axs[2]
    for v, ls in (("halfwave", "-"), ("magnitude", "--")):
        f = g["fits"][v]
        if not f.get("ok"):
            continue
        rho = curve(cfg, grid / f["alpha"], v)
        ax.plot(grid, f["a"] + f["b"] * rho, ls, color=C_MODEL,
                label=f"{v}, alpha {f['alpha']:.2f}")
    ax.plot(steps, g["mean_w"], "o", color="k", label="group mean")
    ax.set_ylim(-0.1, 1.15)
    ax.set_xlabel("shear step (ms)")
    ax.set_ylabel("timbre weight w")
    ax.set_title("The model's curve, stretched to fit")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "tcbound_group.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def write_all(cfg: Config, out: Path, dirs: Sequence[Path] = ()) -> List[Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "tcbound_design.png"]
    schematic(cfg, paths[0])
    for d in dirs:
        paths.append(session_figure(Path(d), out))
    return paths
