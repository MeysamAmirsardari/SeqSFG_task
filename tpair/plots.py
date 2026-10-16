"""Figures for the jittered-pair task."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .config import Config, conditions

CB, CA, CT = "#c0392b", "#5d6d7e", "#117864"


def _mpl():
    import matplotlib
    backend = matplotlib.get_backend().lower()
    if not any(k in backend for k in ("inline", "ipympl", "widget", "nbagg")):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _draw_interval(ax, cfg, iv, y_b=1.0, y_a=0.0, h=.62, x0=0.0, mark=True):
    from matplotlib.patches import Rectangle
    for k in range(cfg.n_pairs):
        ax.add_patch(Rectangle((x0 + iv.b_onsets_ms[k], y_b - h / 2), cfg.tone_ms, h, color=CB))
        if np.isfinite(iv.a_onsets_ms[k]):
            ax.add_patch(Rectangle((x0 + iv.a_onsets_ms[k], y_a - h / 2), cfg.tone_ms, h,
                                   color=CA, alpha=.8))
    if mark and iv.target_pair is not None:
        t = iv.target_pair
        nominal = iv.b_onsets_ms[t] - iv.delta_ms
        ax.add_patch(Rectangle((x0 + nominal, y_b - h / 2 - .06), cfg.tone_ms, h + .12,
                               fill=False, ec="k", lw=1.1, ls=":"))
        ax.annotate("", (x0 + iv.b_onsets_ms[t], y_b + h / 2 + .16),
                    (x0 + nominal, y_b + h / 2 + .16),
                    arrowprops=dict(arrowstyle="->", lw=1.1, color="k"))


def explained(cfg: Config, path: Optional[Path] = None, seed: int = 3):
    """The design in one figure."""
    plt = _mpl()
    from matplotlib.ticker import NullFormatter
    from .stimulus import build_trial
    from .model import lag_index
    from .observer import threshold
    fig = plt.figure(figsize=(14.5, 9.4))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.05, 1.0], hspace=.75, wspace=.6,
                          left=.055, right=.985, top=.94, bottom=.07)
    def title(ax, s):
        ax.set_title(s, fontsize=10.5, loc="left", pad=8)

    # A: one trial
    ax = fig.add_subplot(gs[0, :])
    c50 = next(c for c in conditions(cfg) if c.kind == "pair" and c.lag_ms == 50.0)
    tr = build_trial(cfg, c50, 45.0, np.random.default_rng(seed), target_position=1)
    off = cfg.interval_ms + cfg.isi_ms
    _draw_interval(ax, cfg, tr.first)
    _draw_interval(ax, cfg, tr.second, x0=off)
    ax.text(20, 1.62, "interval 1   TARGET", fontsize=9.5, fontweight="bold")
    ax.text(off + 20, 1.62, "interval 2   standard", fontsize=9.5)
    ax.set_xlim(-40, 2 * cfg.interval_ms + cfg.isi_ms + 40); ax.set_ylim(-.6, 1.9)
    ax.set_yticks([0, 1]); ax.set_yticklabels([f"A {cfg.f_a_hz:.0f}", f"B {cfg.f_b_hz:.0f}"])
    ax.set_xlabel("time (ms)")
    title(ax, f"A   one trial at lag 50 ms: {cfg.n_pairs} pairs at {cfg.rate_hz:g} Hz, each "
              f"pair moved as a unit, the jitter drawn afresh for each interval")
    ax.text(.995, -.30, "dotted = where that B tone would have been; arrow = it came early",
            transform=ax.transAxes, ha="right", fontsize=8, color="#444")

    # B: B on its own is irregular, the lag is not
    rng = np.random.default_rng(11)
    ax = fig.add_subplot(gs[1, 0:3])
    for row in range(5):
        tr2 = build_trial(cfg, c50, 1.0, rng)
        iv = tr2.standard
        for k in range(cfg.n_pairs):
            ax.plot([iv.b_onsets_ms[k]] * 2, [row - .3, row + .3], color=CB, lw=2.2)
    for k in range(cfg.n_pairs):
        ax.axvline(cfg.nominal_onsets()[k], color="#bbb", lw=.8, ls=":")
    ax.set_yticks(range(5)); ax.set_yticklabels([f"#{i + 1}" for i in range(5)], fontsize=8)
    ax.set_xlim(-10, cfg.interval_ms - 200); ax.set_xlabel("B onsets (ms)")
    ax.set_ylabel("five intervals")
    title(ax, f"B   B on its own: every onset wanders by ±{cfg.pair_jitter_ms:g} ms")
    ax.text(0, -.36, "Dotted lines are the 3 Hz grid. One early tone among these is hard to spot:\n"
            "a listener using B alone needs about 43 ms.",
            transform=ax.transAxes, va="top", fontsize=8, color="#444")

    ax = fig.add_subplot(gs[1, 3:6])
    rng = np.random.default_rng(11)
    for row in range(5):
        iv = build_trial(cfg, c50, 1.0, rng).standard
        lag = iv.a_onsets_ms - iv.b_onsets_ms
        ax.plot(range(1, cfg.n_pairs + 1), lag + row * .0, "o-", color=CT, ms=5, lw=1.2, alpha=.6)
    ax.set_ylim(0, 100); ax.set_xticks(range(1, cfg.n_pairs + 1))
    ax.set_xlabel("pair"); ax.set_ylabel("A onset − B onset (ms)")
    title(ax, "C   the A-B lag in the same five intervals: exact")
    ax.text(0, -.36, "Every pair, every interval: 50 ms. The relation between the tones is the\n"
            "only consistent structure the stimulus has.", transform=ax.transAxes, va="top",
            fontsize=8, color="#444")

    # D: model
    lags = np.array([0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 100], float)
    ax = fig.add_subplot(gs[2, 0:3])
    ax.plot(lags, [lag_index(cfg, L, n_seq=15) for L in lags], "-", color="#2c3e50", lw=2.4)
    for c in conditions(cfg):
        if c.kind == "pair":
            ax.plot(c.lag_ms, lag_index(cfg, c.lag_ms, n_seq=15), "o", color="#2c3e50", ms=7)
    ax.set_xticks([c.lag_ms for c in conditions(cfg) if c.kind == "pair"])
    ax.set_ylim(-.03, .9); ax.grid(alpha=.22)
    ax.set_xlabel("A-B lag (ms)"); ax.set_ylabel("λ₂/λ₁")
    title(ax, "D   what the coherence model predicts")

    # E: two model listeners
    ax = fig.add_subplot(gs[2, 3:6])
    pc = [c for c in conditions(cfg) if c.kind == "pair"]
    pair = [threshold(cfg, c, "pair", n=500) for c in pc]
    single = [threshold(cfg, c, "single", n=500) for c in pc]
    x = [c.lag_ms for c in pc]
    ax.plot(x, pair, "o-", color=CT, lw=2.4, ms=7, label="uses the A-B lag")
    ax.plot(x, single, "s--", color=CB, lw=2, ms=6, label="uses B alone")
    ax.set_yscale("log"); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([2, 5, 10, 20, 50]); ax.set_yticklabels(["2", "5", "10", "20", "50"])
    ax.set_ylim(1.5, 90); ax.set_xticks(x); ax.grid(alpha=.22)
    ax.set_xlabel("A-B lag (ms)"); ax.set_ylabel("threshold (ms)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    title(ax, "E   two model listeners, deciding from the real onsets")
    if path:
        fig.savefig(path, dpi=150); plt.close(fig)
    return path


def curve(cfg: Config, res, path: Optional[Path] = None, title: Optional[str] = None):
    plt = _mpl()
    from matplotlib.ticker import NullFormatter
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    pc = sorted([r for r in res.values() if r.kind == "pair"], key=lambda r: r.lag_ms)
    for r in pc:
        for t in r.track_thresholds:
            ax.plot(r.lag_ms, t, "o", mfc="none", mec="#2471a3", ms=8)
        for t in r.censored_thresholds:
            ax.plot(r.lag_ms, t, "v", color=CB, ms=9)
    xs = [r.lag_ms for r in pc if r.geomean_ms]
    ax.plot(xs, [r.geomean_ms for r in pc if r.geomean_ms], "o-", color="k", lw=2.2, ms=10,
            label="geometric mean")
    solo = res.get("b_alone")
    if solo and solo.geomean_ms:
        ax.axhline(solo.geomean_ms, color=CB, ls="--", lw=2)
        ax.text(50, solo.geomean_ms * 1.08, f"B alone: {solo.geomean_ms:.1f} ms", color=CB,
                ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks([r.lag_ms for r in pc]); ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100]); ax.set_yticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.grid(alpha=.25); ax.set_xlabel("A-B lag (ms)      0 = synchronous, 100 = abutting")
    ax.set_ylabel("threshold (ms)"); ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title(title or "threshold against A-B lag", fontsize=11)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150); plt.close(fig)
    return path


def write_all(cfg: Config, out_dir: Path, dirs: Sequence[Path] = ()) -> List[Path]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    made = [explained(cfg, out_dir / "tpair_explained.png")]
    if dirs:
        from .analysis import load, thresholds
        rows, _ = load(list(dirs))
        made.append(curve(cfg, thresholds(rows, cfg), out_dir / "tpair_curve.png"))
    return [p for p in made if p]
