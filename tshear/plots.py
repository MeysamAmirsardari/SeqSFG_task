"""Figures. The axis runs 0% (a chord) on the left to 100% (an even arpeggio) on the right."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .config import Config, combined_gaps_ms, is_isochronous, onsets_ms, validate


def _mpl():
    import matplotlib
    backend = matplotlib.get_backend().lower()
    if not any(k in backend for k in ("inline", "ipympl", "widget", "nbagg")):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def schematic(cfg: Config, path: Optional[Path] = None):
    """What the shear axis is, and where the target sits in it."""
    plt = _mpl()
    from matplotlib.patches import Rectangle
    d = validate(cfg)
    pcts = [c.step_pct for c in d.conditions if c.kind == "figure"]
    fig, axs = plt.subplots(len(pcts), 1, figsize=(12, 1.05 * len(pcts) + 1.6), sharex=True)
    ylab = [f"{f:.0f}" for f in cfg.freqs_hz]
    for ax, p in zip(np.atleast_1d(axs), pcts):
        for rep in range(3):
            o = onsets_ms(cfg, p, rep) - cfg.lead_ms
            for k, on in enumerate(o):
                col = "#c0392b" if k == cfg.target_index else "#5d6d7e"
                ax.add_patch(Rectangle((on, k - 0.32), cfg.tone_ms, 0.64, color=col,
                                       alpha=0.95 if k == cfg.target_index else 0.8))
        for rep in range(3):
            for on in onsets_ms(cfg, p, rep) - cfg.lead_ms:
                ax.plot([on, on], [-1.15, -0.75], color="k", lw=1.2)
        g = combined_gaps_ms(cfg, p)
        tag = ("all four simultaneous" if p == 0 else
               ("ISOCHRONOUS at %.0f Hz" % (cfg.n_tones * cfg.rate_hz)
                if is_isochronous(cfg, p) else
                f"gaps {'/'.join(f'{v:.0f}' for v in g)} ms"))
        ax.text(3 * cfg.period_ms + 25, 1.5, f"step {p:g}%  ({cfg.step_ms(p):.1f} ms)   {tag}",
                fontsize=8.5, va="center",
                color="#c0392b" if is_isochronous(cfg, p) and p > 0 else "#333",
                fontweight="bold" if is_isochronous(cfg, p) and p > 0 else "normal")
        ax.set_xlim(-40, 3 * cfg.period_ms + 430)
        ax.set_ylim(-1.35, cfg.n_tones - 0.4)
        ax.set_yticks(range(cfg.n_tones)); ax.set_yticklabels(ylab, fontsize=7)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.set_ylabel(f"{p:g}%", rotation=0, ha="right", va="center", fontsize=9)
    np.atleast_1d(axs)[-1].set_xlabel("time (ms)")
    np.atleast_1d(axs)[0].set_title(
        f"the shear axis  —  four tones repeating at {cfg.rate_hz:g} Hz, three repetitions "
        f"shown\nred = the target ({cfg.target_freq_hz:.0f} Hz), ticks below = every onset of "
        "any tone", fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def prediction(cfg: Config, path: Optional[Path] = None):
    """The model's index against the step, beside what the two listening strategies predict."""
    plt = _mpl()
    from .model import effective_objects, prediction_band
    fine = list(np.arange(0, 101, 5.0))
    band = prediction_band(cfg, fine)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    a = ax[0]
    a.fill_between(band["pcts"], [1 + v * 3 for v in band["lo"]],
                   [1 + v * 3 for v in band["hi"]], color="#5d6d7e", alpha=.2, lw=0)
    a.plot(band["pcts"], [1 + v * 3 for v in band["mid"]], color="#5d6d7e", lw=2.5)
    for c in validate(cfg).conditions:
        if c.kind != "figure":
            continue
        v = effective_objects(cfg, c.step_pct)["normalised"]
        a.plot(c.step_pct, 1 + v * 3, "o", color="#2c3e50", ms=9, zorder=3)
    a.set_xticks([c.step_pct for c in validate(cfg).conditions if c.kind == "figure"])
    a.set_ylim(0.8, cfg.n_tones + 0.2); a.grid(alpha=.25)
    a.set_xlabel("shear step (% of the isochronous step)")
    a.set_ylabel("effective number of objects")
    a.set_title("what the model sees", fontsize=10)
    a.axhline(1, color="grey", ls=":", lw=1); a.axhline(cfg.n_tones, color="grey", ls=":", lw=1)
    a.text(50, 1.06, "one object", fontsize=8, color="grey")
    a.text(3, cfg.n_tones - 0.16, f"{cfg.n_tones} objects", fontsize=8, color="grey")

    a = ax[1]
    pcts = np.array(fine)
    idx = np.array([effective_objects(cfg, p)["normalised"] for p in fine])
    floor, plateau = 2.5, 9.0
    figure_only = floor * (plateau / floor * 2.2) ** idx
    a.plot(pcts, figure_only, ":", color="#1a5276", lw=1.8,
           label="the figure reference on its own")
    a.axhline(plateau, color="#c0392b", lw=1.8, ls=":",
              label="the target's own rhythm on its own")
    # A listener uses whichever reference is better, so what is MEASURED is the minimum of the
    # two: the figure wins while the tones are close to synchronous and the within-channel
    # rhythm takes over once the figure has sheared apart. The plateau is therefore a ceiling
    # the design can measure directly, which is what the single-tone control is for.
    a.plot(pcts, np.minimum(figure_only, plateau), "-", color="#117864", lw=3,
           label="what a listener would actually produce", zorder=4)
    a.set_xticks([c.step_pct for c in validate(cfg).conditions if c.kind == "figure"])
    a.set_yscale("log"); a.set_ylim(1.5, 20); a.grid(alpha=.25)
    import matplotlib
    a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_yticks([2, 3, 5, 8, 12, 20]); a.set_yticklabels(["2", "3", "5", "8", "12", "20"])
    a.set_xlabel("shear step (%)"); a.set_ylabel("jitter threshold (ms)")
    a.set_title("the two strategies make different curves", fontsize=10)
    a.legend(frameon=False, fontsize=8.5, loc="lower right")
    a.text(2, 16, "illustrative levels; the SHAPES are what the design tests.\nA flat curve is "
                  "not a null result -- it is a positive\nresult about which reference the "
                  "listener used.", fontsize=8, color="#444", va="top")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150); plt.close(fig)
    return path


def curve(cfg: Config, res, path: Optional[Path] = None, title: Optional[str] = None):
    """The measured curve: every track, their geometric mean, and the single-tone control."""
    plt = _mpl()
    import matplotlib
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    fig_conds = sorted([r for r in res.values() if r.kind == "figure"], key=lambda r: r.step_pct)
    for r in fig_conds:
        for t in r.track_thresholds:
            ax.plot(r.step_pct, t, "o", mfc="none", mec="#2471a3", ms=8, zorder=2)
        for t in r.censored_thresholds:
            ax.plot(r.step_pct, t, "v", color="#c0392b", ms=9, zorder=2)
    xs = [r.step_pct for r in fig_conds if r.geomean_ms]
    ys = [r.geomean_ms for r in fig_conds if r.geomean_ms]
    ax.plot(xs, ys, "o-", color="k", lw=2.2, ms=10, zorder=3, label="geometric mean")
    solo = next((r for r in res.values() if r.kind == "single" and r.geomean_ms), None)
    if solo:
        ax.axhline(solo.geomean_ms, color="#c0392b", ls="--", lw=2)
        ax.text(2, solo.geomean_ms * 1.06, f"target tone alone: {solo.geomean_ms:.2f} ms",
                fontsize=8.5, color="#c0392b")
    ax.set_xticks([r.step_pct for r in fig_conds])
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([1, 2, 3, 5, 8, 12, 20, 30])
    ax.set_yticklabels(["1", "2", "3", "5", "8", "12", "20", "30"])
    ax.grid(alpha=.25)
    ax.set_xlabel("shear step (% of the isochronous step)     0 = chord, 100 = even arpeggio")
    ax.set_ylabel("jitter detection threshold (ms)")
    ax.set_title(title or "jitter threshold against shear", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150); plt.close(fig)
    return path


def explained(cfg: Config, path: Optional[Path] = None):
    """One figure that explains the whole design: the tones, a trial, the jitter, the model,
    the two predicted shapes, and the two references a listener can use."""
    plt = _mpl()
    import matplotlib
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import NullFormatter
    import numpy as np
    from .stimulus import build_trial
    from .model import effective_objects
    d = validate(cfg)
    K = cfg.target_index
    CT, CO = "#c0392b", "#5d6d7e"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    
    fig = plt.figure(figsize=(15.5, 10.2))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 0.95, 1.05], hspace=.72, wspace=.55,
                          left=.055, right=.985, top=.935, bottom=.075)
    
    def panel(ax, letter, text):
        ax.set_title(f"{letter}   {text}", fontsize=10.5, loc="left", pad=8)
    
    # ================================================== A: the four tones
    ax = fig.add_subplot(gs[0, 0:2])
    for i, f in enumerate(cfg.freqs_hz):
        c = CT if i == K else CO
        ax.plot([f, f], [0, 1], color=c, lw=4, solid_capstyle="butt")
        ax.plot(f, 1, "s" if i == K else "o", color=c, ms=9)
        ax.annotate(f"{f:.0f}", (f, 1.10), ha="center", fontsize=8.5, color=c,
                    fontweight="bold" if i == K else "normal")
    ax.set_xscale("log"); ax.set_xlim(480, 5800); ax.set_ylim(0, 1.5)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([700, 1200, 2200, 4000]); ax.set_xticklabels(["700", "1200", "2200", "4000"])
    ax.set_yticks([]); ax.set_xlabel("frequency (Hz)")
    panel(ax, "A", "four tones; red is the target")
    ax.text(0, -.25, "4.4 ERB apart, inharmonic, no common fundamental.\nThe target is the 3rd of four: interior, same every trial.",
            transform=ax.transAxes, va="top", fontsize=8, color="#444")
    
    # ================================================== B: one trial
    ax = fig.add_subplot(gs[0, 2:6])
    cond = next(c for c in d.conditions if c.name == "step_50")
    tr = build_trial(cfg, cond, 35.0, np.random.default_rng(1), target_position=1, direction=+1)
    IV = cfg.interval_ms()
    for off, (name, iv) in zip((0.0, IV + cfg.isi_ms),
                               (("interval 1   TARGET", tr.first),
                                ("interval 2   standard", tr.second))):
        for rep in range(cfg.n_repeats):
            for k in range(cfg.n_tones):
                ax.add_patch(Rectangle((off + iv.onsets_ms[rep, k], k - .34), cfg.tone_ms, .68,
                                       color=CT if k == K else CO, alpha=.95 if k == K else .72))
        ax.text(off + 25, cfg.n_tones - .15, name, fontsize=9.5,
                fontweight="bold" if "TARGET" in name else "normal")
    nom, shifted = tr.second.onsets_ms[-1, K], tr.first.onsets_ms[-1, K]
    ax.add_patch(Rectangle((nom, K - .48), cfg.tone_ms, .96, fill=False, ec="k", lw=1.3, ls=":"))
    ax.set_xlim(-60, 2 * IV + cfg.isi_ms + 60); ax.set_ylim(-.7, cfg.n_tones + .45)
    ax.set_yticks(range(cfg.n_tones))
    ax.set_yticklabels([f"{f:.0f}" for f in cfg.freqs_hz], fontsize=8)
    ax.set_xlabel("time (ms)"); ax.set_ylabel("Hz", fontsize=8)
    panel(ax, "B", "one trial at step 50%  —  the intervals differ in ONE tone of the LAST "
                   "repetition, and nothing else")
    
    # ================================================== C: the jitter, magnified
    ax = fig.add_subplot(gs[1, 0:2])
    for k in range(cfg.n_tones):
        ax.add_patch(Rectangle((tr.first.onsets_ms[-1, k], k - .34), cfg.tone_ms, .68,
                               color=CT if k == K else CO, alpha=.95 if k == K else .72))
    ax.add_patch(Rectangle((nom, K - .48), cfg.tone_ms, .96, fill=False, ec="k", lw=1.6, ls=":"))
    yy = K + .78
    ax.annotate("", (nom, yy), (shifted, yy), arrowprops=dict(arrowstyle="<->", lw=1.9, color="k"))
    ax.text((nom + shifted) / 2, yy + .14, f"δ = {abs(tr.delta_signed_ms):.0f} ms", ha="center",
            fontsize=10, fontweight="bold")
    lo = float(tr.first.onsets_ms[-1].min()) - 50
    ax.set_xlim(lo, lo + 380); ax.set_ylim(-.7, cfg.n_tones + .45)
    ax.set_yticks([]); ax.set_xlabel("time (ms)")
    ax.spines["left"].set_visible(False)
    panel(ax, "C", "the last repetition, magnified")
    ax.text(0, -.34, "dotted = where the standard puts it.\nThe staircase tracks δ; direction is "
            "random.", transform=ax.transAxes, va="top", fontsize=8, color="#444")
    
    # ================================================== D: model index
    ax = fig.add_subplot(gs[1, 2:4])
    fine = np.arange(0, 101, 5.0)
    idx = np.array([effective_objects(cfg, p)["normalised"] for p in fine])
    l2 = np.array([effective_objects(cfg, p)["lambda2_over_lambda1"] for p in fine])
    ax.plot(fine, 1 + idx * 3, "-", color="#2c3e50", lw=2.6, label="effective objects (used)")
    ax.plot(fine, 1 + l2 * 3, ":", color="#b7950b", lw=2,
            label="$\\lambda_2/\\lambda_1$ rescaled (not used)")
    for p in cfg.step_pcts:
        ax.plot(p, 1 + effective_objects(cfg, p)["normalised"] * 3, "o", color="#2c3e50", ms=8,
                zorder=3)
    ax.axhline(1, color="grey", ls=":", lw=1); ax.axhline(4, color="grey", ls=":", lw=1)
    ax.set_xticks([0, 15, 30, 50, 75, 100]); ax.set_ylim(0.75, 4.35); ax.grid(alpha=.22)
    ax.set_xlabel("shear step (%)"); ax.set_ylabel("effective number of objects")
    panel(ax, "D", "what the model sees")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.text(0, -.34, "1.00 objects at step 0, 3.84 at 100%.\n$\\lambda_2/\\lambda_1$ is NOT monotone "
            "here — which is\nwhy the design does not use it.",
            transform=ax.transAxes, va="top", fontsize=8, color="#444")
    
    # ================================================== E: the two predictions
    ax = fig.add_subplot(gs[1, 4:6])
    floor, plateau = 2.5, 9.0
    fig_only = floor * (plateau / floor * 2.2) ** idx
    ax.plot(fine, fig_only, ":", color="#1a5276", lw=1.8, label="figure reference alone")
    ax.axhline(plateau, color=CT, lw=1.8, ls=":", label="own-rhythm reference alone")
    ax.plot(fine, np.minimum(fig_only, plateau), "-", color="#117864", lw=3.2, zorder=4,
            label="what a listener produces")
    ax.set_xticks([0, 15, 30, 50, 75, 100]); ax.set_yscale("log"); ax.set_ylim(1.6, 22)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([2, 3, 5, 8, 12, 20]); ax.set_yticklabels(["2", "3", "5", "8", "12", "20"])
    ax.grid(alpha=.22); ax.set_xlabel("shear step (%)"); ax.set_ylabel("jitter threshold (ms)")
    panel(ax, "E", "the two strategies differ in SHAPE")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.text(0, -.34, "Illustrative levels; the shapes are the claim.\nA flat curve is a positive "
            "result about strategy,\nnot a null result about binding.",
            transform=ax.transAxes, va="top", fontsize=8, color="#444")
    
    # ================================================== F: the two references
    for col, p in enumerate((0.0, 50.0, 100.0)):
        ax = fig.add_subplot(gs[2, 2 * col:2 * col + 2])
        for rep in range(3):
            o = onsets_ms(cfg, p, rep) - cfg.lead_ms
            ax.add_patch(Rectangle((o[K], 1.16), cfg.tone_ms, .46, color=CT))
            for k in range(cfg.n_tones):
                if k == K:
                    continue
                row = [j for j in range(cfg.n_tones) if j != K].index(k)
                ax.add_patch(Rectangle((o[k], row * .30), cfg.tone_ms, .24, color=CO, alpha=.78))
        o0 = onsets_ms(cfg, p, 0)[K] - cfg.lead_ms
        for rep in range(2):
            x0 = o0 + rep * cfg.period_ms + cfg.tone_ms / 2
            ax.annotate("", (x0, 1.78), (x0 + cfg.period_ms, 1.78),
                        arrowprops=dict(arrowstyle="<->", lw=1.2, color=CT))
        ax.text(o0 + cfg.period_ms, 1.90, f"{cfg.period_ms:.0f} ms", fontsize=8.5, ha="center",
                color=CT, fontweight="bold")
        ax.axhline(1.06, color="#ccc", lw=1)
        ax.set_xlim(-300 if col == 0 else -40, 3 * cfg.period_ms + 30)
        ax.set_ylim(-.16, 2.20)
        ax.set_yticks([]); ax.set_xlabel("time →", fontsize=8); ax.set_xticks([])
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(False)
        ax.set_title(f"step {p:g}%", fontsize=10, pad=4)
        if col == 0:
            ax.text(-285, 1.39, "the target's\nown channel", ha="left", va="center",
                    fontsize=8.5, color=CT, fontweight="bold")
            ax.text(-285, .34, "the figure\nit sits in", ha="left", va="center",
                    fontsize=8.5, color=CO, fontweight="bold")
            ax.text(-300, 2.42, "F   the two references, and why a flat curve is interpretable",
                    fontsize=10.5, fontweight="bold", ha="left", va="bottom")
    fig.text(.52, .028,
             "ABOVE THE LINE: the target repeats every 333 ms at every step — identical in all "
             "three panels. That reference does not know the step exists,\n"
             "so a listener using only it gives a FLAT curve.    BELOW THE LINE: the figure spreads "
             "apart. A listener using it gives a RISING curve.",
             ha="center", va="bottom", fontsize=9.2, color="#222")
    if path:
        fig.savefig(path, dpi=150); plt.close(fig)
    return path


def write_all(cfg: Config, out_dir: Path, dirs: Sequence[Path] = ()) -> List[Path]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    made = [explained(cfg, out_dir / "tshear_explained.png"),
            schematic(cfg, out_dir / "tshear_schematic.png"),
            prediction(cfg, out_dir / "tshear_prediction.png")]
    if dirs:
        from .analysis import load, thresholds
        rows, metas = load(list(dirs))
        made.append(curve(cfg, thresholds(rows, cfg), out_dir / "tshear_curve.png"))
    return [p for p in made if p]
