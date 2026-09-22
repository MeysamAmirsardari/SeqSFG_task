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


def write_all(cfg: Config, out_dir: Path, dirs: Sequence[Path] = ()) -> List[Path]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    made = [schematic(cfg, out_dir / "tshear_schematic.png"),
            prediction(cfg, out_dir / "tshear_prediction.png")]
    if dirs:
        from .analysis import load, thresholds
        rows, metas = load(list(dirs))
        made.append(curve(cfg, thresholds(rows, cfg), out_dir / "tshear_curve.png"))
    return [p for p in made if p]
