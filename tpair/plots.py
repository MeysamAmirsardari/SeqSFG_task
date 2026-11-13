"""Figures for the jittered-pair task."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .config import Config, conditions

CB, CA, CT = "#c0392b", "#5d6d7e", "#117864"
DIR_STYLE = {"in": dict(color="#1f4e79", marker="o", ls="-"),
             "out": dict(color="#b9770e", marker="D", ls="-"),
             "random": dict(color="#1f4e79", marker="o", ls="-")}


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
        moves = [(iv.b_onsets_ms[t], iv.b_shift_ms, y_b, +1)]
        if np.isfinite(iv.a_onsets_ms[t]):
            moves.append((iv.a_onsets_ms[t], iv.a_shift_ms, y_a, -1))
        for on, shift, y, side in moves:
            if not shift:
                continue
            was = on - shift
            ax.add_patch(Rectangle((x0 + was, y - h / 2 - .06), cfg.tone_ms, h + .12,
                                   fill=False, ec="k", lw=1.1, ls=":"))
            yy = y + side * (h / 2 + .16)
            ax.annotate("", (x0 + on + cfg.tone_ms / 2, yy), (x0 + was + cfg.tone_ms / 2, yy),
                        arrowprops=dict(arrowstyle="->", lw=1.1, color="k"))


def _pct_axis(ax, cfg, levels):
    """Elhilali's Fig. 8 convention: dT runs from 100% on the left to 0% on the right."""
    ax.set_xlim(108, -8)
    ax.set_xticks([cfg.lag_pct(L) for L in levels])
    ax.set_xticklabels([f"{cfg.lag_pct(L):g}%" for L in levels])
    ax.set_xlabel(f"\u0394T, B onset after A onset   (100% = {cfg.tone_ms:g} ms, B starts at A's offset)")


def explained(cfg: Config, path: Optional[Path] = None, seed: int = 3):
    """The design in one figure."""
    plt = _mpl()
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import NullFormatter
    from .stimulus import build_trial, token_lags
    from .model import lag_index
    from .observer import p_correct, threshold
    from .config import scatter_geometry
    scatter = cfg.timing == "scatter"
    dirs = cfg.directions
    allpc = [c for c in conditions(cfg) if c.kind == "pair"]
    pc = [c for c in allpc if c.direction == dirs[0]]
    levels = [c.lag_ms for c in pc]
    mid = pc[len(pc) // 2]
    fig = plt.figure(figsize=(14.5, 9.8))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.05, 1.0], hspace=.8, wspace=.65,
                          left=.055, right=.985, top=.94, bottom=.07)
    def title(ax, s):
        ax.set_title(s, fontsize=10.5, loc="left", pad=8)

    # A: one trial
    ax = fig.add_subplot(gs[0, :])
    tr = build_trial(cfg, mid, 30.0, np.random.default_rng(seed), target_position=1)
    ilen = cfg.interval_len_ms
    off = ilen + cfg.isi_ms
    _draw_interval(ax, cfg, tr.first)
    _draw_interval(ax, cfg, tr.second, x0=off)
    ax.text(20, 1.72, "interval 1   TARGET", fontsize=9.5, fontweight="bold")
    ax.text(off + 20, 1.72, "interval 2   standard", fontsize=9.5)
    ax.set_xlim(-40, 2 * ilen + cfg.isi_ms + 40); ax.set_ylim(-.75, 2.0)
    ax.set_yticks([0, 1]); ax.set_yticklabels([f"A {cfg.f_a_hz:.0f}", f"B {cfg.f_b_hz:.0f}"])
    ax.set_xlabel("time (ms)")
    title(ax, f"A   one trial at \u0394T = {cfg.lag_pct(mid.lag_ms):g}%: {cfg.n_pairs} AB tokens "
              + ("scattered at random, a new draw for every interval" if scatter else
                 f"at {cfg.rate_hz:g} Hz on average, each moved as a unit, the jitter drawn afresh "
                 "for each interval"))
    longer = tr.delta_signed_ms > 0
    moved = (f"A and B each moved 15 ms, opposite ways: the lag is 30 ms "
             f"{'longer' if longer else 'shorter'}"
             if cfg.shift == "split" else f"B moved 30 ms {'later' if longer else 'earlier'}")
    ax.text(.995, -.30, f"dotted = where the target's tones would have been; {moved}",
            transform=ax.transAxes, ha="right", fontsize=8, color="#444")

    # B: the five tokens
    ax = fig.add_subplot(gs[1, 0:2])
    h = .34
    for i, L in enumerate(levels[::-1]):
        y = i
        ax.add_patch(Rectangle((0, y - h - .02), cfg.tone_ms, h, color=CA, alpha=.85))
        ax.add_patch(Rectangle((L, y + .02), cfg.tone_ms, h, color=CB))
        ax.text(-6, y, f"{cfg.lag_pct(L):g}%", ha="right", va="center", fontsize=9)
    if len(dirs) > 1:
        y = len(levels) - 1 + .02 + h + .2
        for sign, d in ((-1, "in"), (1, "out")):
            L0 = levels[0]
            ax.annotate("", (L0 + cfg.tone_ms / 2 + sign * 22, y), (L0 + cfg.tone_ms / 2, y),
                        arrowprops=dict(arrowstyle="->", lw=1.3, color=DIR_STYLE[d]["color"]))
            ax.text(L0 + cfg.tone_ms / 2 + sign * 26, y, d, ha="right" if sign < 0 else "left",
                    va="center", fontsize=8.5, color=DIR_STYLE[d]["color"], fontweight="bold")
    ax.set_xlim(-30, 2 * cfg.tone_ms + 8); ax.set_ylim(-.6, len(levels) - .4 + (.45 if len(dirs) > 1 else 0))
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("ms from A onset")
    title(ax, "B   the five tokens: A first, B trails")
    ax.text(0, -.36, ("From synchrony to B starting at A's offset. In the target token B\n"
                      "comes earlier (in) or later (out) relative to its own A.")
            if len(dirs) > 1 else
            "From synchrony to B starting at A's offset. The tones always\n"
            "overlap or touch; there is never a gap inside a token.",
            transform=ax.transAxes, va="top", fontsize=8, color="#444")

    # C: each tone on its own
    rng = np.random.default_rng(11)
    ax = fig.add_subplot(gs[1, 2:4])
    for row in range(5):
        iv = build_trial(cfg, mid, 1.0, rng).standard
        for k in range(cfg.n_pairs):
            ax.plot([iv.b_onsets_ms[k]] * 2, [row - .3, row + .3], color=CB, lw=2.2)
    if not scatter:
        for k in range(cfg.n_pairs):
            ax.axvline(cfg.nominal_onsets()[k] + mid.lag_ms, color="#bbb", lw=.8, ls=":")
    ax.set_yticks(range(5)); ax.set_yticklabels([f"#{i + 1}" for i in range(5)], fontsize=8)
    ax.set_xlim(-10, cfg.interval_len_ms); ax.set_xlabel("B onsets (ms)")
    ax.set_ylabel("five intervals")
    if scatter:
        g = scatter_geometry(cfg)["min_ioi_ms"]
        if cfg.pair_clear_ms is not None:      # the screen raises the least gap that is kept
            g = max(g, 2 * max(cfg.lags_ms) + cfg.pair_clear_ms)
        pc_t = max(p_correct(cfg, c, cfg.delta_max_ms, "tones", 800, 3) for c in pc[:1])
        title(ax, "C   each tone on its own: no rhythm to learn")
        ax.text(0, -.36, f"Placed uniformly at random, at least {g:.0f} ms apart. Judging A or B\n"
                f"against its neighbours: {100 * pc_t:.0f}% correct at the {cfg.delta_max_ms:g} ms ceiling.",
                transform=ax.transAxes, va="top", fontsize=8, color="#444")
    else:
        single = threshold(cfg, pc[0], "single", n=500)
        title(ax, f"C   B on its own wanders by ±{cfg.pair_jitter_ms:g} ms")
        ax.text(0, -.36, "Dotted lines are the 3 Hz grid. One early B among these is hard to\n"
                f"spot: a listener using B alone needs about {single:.0f} ms.",
                transform=ax.transAxes, va="top", fontsize=8, color="#444")

    # D: the lag inside the token is exact
    ax = fig.add_subplot(gs[1, 4:6])
    rng = np.random.default_rng(11)
    for row in range(5):
        iv = build_trial(cfg, mid, 1.0, rng).standard
        ax.plot(range(1, cfg.n_pairs + 1), token_lags(cfg, iv), "o-", color=CT, ms=5, lw=1.2,
                alpha=.6)
    ax.set_ylim(0, cfg.tone_ms); ax.set_xticks(range(1, cfg.n_pairs + 1))
    ax.set_xlabel("token"); ax.set_ylabel("B onset − A onset (ms)")
    title(ax, "D   the A-B lag in the same intervals: exact")
    ax.text(0, -.36, f"Every token, every interval: {mid.lag_ms:g} ms. The relation between\n"
            "the tones is the only regular thing in the sound.", transform=ax.transAxes,
            va="top", fontsize=8, color="#444")

    # E: model
    grid = np.linspace(0, cfg.tone_ms, 13)
    ax = fig.add_subplot(gs[2, 0:3])
    ax.plot([cfg.lag_pct(L) for L in grid], [lag_index(cfg, L, n_seq=15) for L in grid], "-",
            color="#2c3e50", lw=2.4)
    for L in levels:
        ax.plot(cfg.lag_pct(L), lag_index(cfg, L, n_seq=15), "o", color="#2c3e50", ms=7)
    _pct_axis(ax, cfg, levels)
    ax.set_ylim(-.03, .85); ax.grid(alpha=.22); ax.set_ylabel("λ₂/λ₁")
    title(ax, "E   what the coherence model predicts, with this timing")

    # F: two model listeners
    ax = fig.add_subplot(gs[2, 3:6])
    x = [cfg.lag_pct(L) for L in levels]
    for d in dirs:
        cs = [c for c in allpc if c.direction == d]
        tag = f", {d}" if len(dirs) > 1 else ""
        st = DIR_STYLE[d] if len(dirs) > 1 else dict(color=CT, marker="o", ls="-")
        ax.plot(x, [threshold(cfg, c, "pair", n=500) for c in cs], lw=2.4, ms=7,
                label=f"uses the A-B lag{tag}", **st)
    if scatter:
        ax.axhspan(cfg.delta_max_ms, 200, color=CB, alpha=.08, lw=0)
        ax.axhline(cfg.delta_max_ms, color=CB, ls="--", lw=1.6)
        ax.text(50, cfg.delta_max_ms * 1.25, "judging either tone's own timing: never reaches "
                "threshold below the ceiling", color=CB, ha="center", fontsize=8.2,
                fontweight="bold")
        ax.set_ylim(1.5, 160)
    else:
        sing = [threshold(cfg, c, "single", n=500) for c in pc]
        ax.plot(x, sing, "s--", color=CB, lw=2, ms=6, label="uses B alone")
        ax.set_ylim(1.5, 90)
    ax.set_yscale("log"); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([2, 5, 10, 20, 50, 100]); ax.set_yticklabels(["2", "5", "10", "20", "50", "100"])
    _pct_axis(ax, cfg, levels); ax.grid(alpha=.22)
    ax.set_ylabel("threshold (ms)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    title(ax, "F   model listeners, deciding from the real onsets")
    if path:
        fig.savefig(path, dpi=150); plt.close(fig)
    return path


def curve(cfg: Config, res, path: Optional[Path] = None, title: Optional[str] = None):
    plt = _mpl()
    from matplotlib.ticker import NullFormatter
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    P = cfg.lag_pct
    dirs = list(dict.fromkeys(r.direction for r in res.values()))
    multi = len(dirs) > 1
    for i, d in enumerate(dirs):
        st = DIR_STYLE[d] if multi else dict(color="k", marker="o", ls="-")
        nudge = (i - (len(dirs) - 1) / 2) * 2.5          # keep the two directions' dots apart
        pc = sorted([r for r in res.values() if r.kind == "pair" and r.direction == d],
                    key=lambda r: r.lag_ms)
        for r in pc:
            for t in r.track_thresholds:
                ax.plot(P(r.lag_ms) + nudge, t, st["marker"], mfc="none",
                        mec=st["color"] if multi else "#2471a3", ms=8)
            for t in r.censored_thresholds:
                ax.plot(P(r.lag_ms) + nudge, t, "^" if r.censor_side == "ceiling" else "v",
                        color=st["color"] if multi else CB, ms=9)
        xs = [P(r.lag_ms) + nudge for r in pc if r.geomean_ms]
        ax.plot(xs, [r.geomean_ms for r in pc if r.geomean_ms], lw=2.2, ms=10,
                label=f"{d}: B {'earlier' if d == 'in' else 'later'}" if multi else "geometric mean",
                **st)
        solo = next((r for r in res.values() if r.kind == "b_alone" and r.direction == d), None)
        if solo and solo.geomean_ms:
            c = st["color"] if multi else CB
            ax.axhline(solo.geomean_ms, color=c, ls="--", lw=1.8)
            ax.text(50, solo.geomean_ms * 1.08,
                    f"B alone{f' ({d})' if multi else ''}: {solo.geomean_ms:.1f} ms", color=c,
                    ha="center", fontsize=8.5, fontweight="bold")
    if cfg.timing == "scatter":
        ax.axhline(cfg.delta_max_ms, color="#888", ls=":", lw=1.2)
        ax.text(50, cfg.delta_max_ms * 1.06, "ceiling", color="#888", ha="center", fontsize=8)
    _pct_axis(ax, cfg, sorted({r.lag_ms for r in res.values() if r.kind == "pair"}))
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100]); ax.set_yticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.grid(alpha=.25)
    ax.set_ylabel("threshold (ms)"); ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title(title or "threshold against \u0394T", fontsize=11)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150); plt.close(fig)
    return path


def write_all(cfg: Config, out_dir: Path, dirs: Sequence[Path] = ()) -> List[Path]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.n_context:
        made = [context_explained(cfg, out_dir / "tpair_context_explained.png")]
    else:
        made = [explained(cfg, out_dir / "tpair_explained.png")]
    if dirs:
        from .analysis import load, thresholds
        rows, _ = load(list(dirs))
        res = thresholds(rows, cfg)
        made.append(context_curve(cfg, res, out_dir / "tpair_context_curve.png") if cfg.n_context
                    else curve(cfg, res, out_dir / "tpair_curve.png"))
    return [p for p in made if p]


# ---------------------------------------------------------------------------- context mode
CTX_COLORS = ("#1f4e79", "#7f8c8d", "#b9770e")     # bound side, stationary, segregated side


def context_explained(cfg: Config, path: Optional[Path] = None, seed: int = 5, n: int = 700):
    """The context design in one figure: a trial, the three histories, what each account
    predicts, and what the coherence model predicts."""
    plt = _mpl()
    from matplotlib.patches import Rectangle
    from .context import context_index, predictions
    from .stimulus import build_trial
    conds = [c for c in conditions(cfg) if c.direction == cfg.directions[0]]
    ctxs = list(cfg.context_lags_ms)
    L = cfg.lags_ms[0]
    fig = plt.figure(figsize=(14.5, 9.6))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.05], hspace=.75, wspace=.45,
                          left=.055, right=.985, top=.94, bottom=.07)
    def title(ax, s):
        ax.set_title(s, fontsize=10.5, loc="left", pad=8)

    # A: one trial, context shaded
    ax = fig.add_subplot(gs[0, :])
    c0 = conds[0]
    tr = build_trial(cfg, c0, 20.0, np.random.default_rng(seed), target_position=1)
    ilen = cfg.interval_len_ms
    off = ilen + cfg.isi_ms
    for x0, iv in ((0.0, tr.first), (off, tr.second)):
        a, b = iv.a_onsets_ms, iv.b_onsets_ms
        end_ctx = max(a[cfg.n_context - 1], b[cfg.n_context - 1]) + cfg.tone_ms
        ax.add_patch(Rectangle((x0, -.5), end_ctx + 20, 2.0, color="#eaf0f6", lw=0, zorder=0))
        ax.text(x0 + 10, 1.62, "context", fontsize=8.5, color=CTX_COLORS[0])
        _draw_interval(ax, cfg, iv, x0=x0)
    ax.text(20, 1.95, "interval 1   TARGET", fontsize=9.5, fontweight="bold")
    ax.text(off + 20, 1.95, "interval 2   standard", fontsize=9.5)
    ax.set_xlim(-40, 2 * ilen + cfg.isi_ms + 40); ax.set_ylim(-.75, 2.2)
    ax.set_yticks([0, 1]); ax.set_yticklabels([f"A {cfg.f_a_hz:.0f}", f"B {cfg.f_b_hz:.0f}"])
    ax.set_xlabel("time (ms)")
    title(ax, f"A   one trial: {cfg.n_context} context tokens at ΔT {cfg.lag_pct(c0.context_lag_ms):g}%, "
              f"then {cfg.n_pairs - cfg.n_context} test tokens at {cfg.lag_pct(L):g}%; both intervals "
              "share the context, the change is in a test token")

    # B: the three histories
    ax = fig.add_subplot(gs[1, 0:2])
    k = np.arange(1, cfg.n_pairs + 1)
    for C, col in zip(ctxs, CTX_COLORS):
        lag = np.where(k <= cfg.n_context, C, L)
        ax.step(k, cfg.lag_pct(lag), where="mid", color=col, lw=2.4,
                label=f"context {cfg.lag_pct(C):g}%" + ("  (stationary)" if abs(C - L) < 1e-9 else ""))
    for t in cfg.target_pairs:
        ax.axvline(t + 1, color="#c0392b", lw=1, ls=":")
    ax.text(cfg.target_pairs[-1] + 1.05, cfg.lag_pct(max(ctxs)) * 1.02, "target here", color="#c0392b",
            fontsize=8)
    ax.axvspan(.5, cfg.n_context + .5, color="#eaf0f6", lw=0, zorder=0)
    ax.set_xticks(k); ax.set_xlabel("token"); ax.set_ylabel("ΔT (%)")
    ax.set_ylim(-8, cfg.lag_pct(max(ctxs)) + 18)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=3)
    title(ax, "B   three histories, one test: the test tokens are identical in all three")

    # C: the coherence model
    ax = fig.add_subplot(gs[1, 2:4])
    for w, ls in ((400, ":"), (600, "-"), (1000, "--")):
        ax.plot([cfg.lag_pct(C) for C in ctxs], [context_index(cfg, C, L, w) for C in ctxs], ls,
                color="#2c3e50", marker="o", lw=2, label=f"window {w} ms")
    ax.set_xticks([cfg.lag_pct(C) for C in ctxs])
    ax.set_xlabel("context ΔT (%)"); ax.set_ylabel("λ₂/λ₁ at the target")
    ax.legend(frameon=False, fontsize=8.5)
    title(ax, "C   temporal coherence: flat unless its window reaches back into the context")

    # D: what each account predicts
    pr = predictions(cfg, n=n)
    names = {"pair": "no memory", "change": "fooled by the switch (decoy)",
             "persist": "persistence", "contrast": "contrast"}
    for j, (m, lab) in enumerate(names.items()):
        ax = fig.add_subplot(gs[2, j])
        th = pr[m]["thresholds"]
        for d, mk in zip(cfg.directions, ("o", "D")):
            ys = [th[c.name] for c in conditions(cfg) if c.direction == d]
            ax.plot([cfg.lag_pct(C) for C in ctxs], ys, marker=mk, lw=2,
                    color=DIR_STYLE[d]["color"], label=d)
        idx = pr[m]["indices"].get(L, {})
        ax.set_xticks([cfg.lag_pct(C) for C in ctxs]); ax.set_xlabel("context ΔT (%)")
        ax.set_ylim(0, 25)
        if j == 0:
            ax.set_ylabel("threshold (ms)"); ax.legend(frameon=False, fontsize=8.5)
        ax.text(.03, .95, f"P {idx.get('P', np.nan):+.2f}  D {idx.get('D', np.nan):+.2f}",
                transform=ax.transAxes, fontsize=8.5, va="top")
        title(ax, ("D   " if j == 0 else "") + lab)
    if path:
        fig.savefig(path, dpi=150); plt.close(fig)
    return path


def context_curve(cfg: Config, res, path: Optional[Path] = None, title: Optional[str] = None):
    """Thresholds by context, one line per direction, tracks as open markers."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ctxs = list(cfg.context_lags_ms)
    for i, d in enumerate(cfg.directions):
        st = DIR_STYLE[d]
        nudge = (i - (len(cfg.directions) - 1) / 2) * 2.0
        xs, ys = [], []
        for C in ctxs:
            r = next((r for r in res.values() if r.direction == d and r.context_lag_ms == C), None)
            if r is None:
                continue
            x = cfg.lag_pct(C) + nudge
            for t in r.track_thresholds:
                ax.plot(x, t, st["marker"], mfc="none", mec=st["color"], ms=8)
            for t in r.censored_thresholds:
                ax.plot(x, t, "^", color=st["color"], ms=9)
            vals = list(r.track_thresholds) + list(r.censored_thresholds)
            if vals:
                xs.append(x); ys.append(float(np.exp(np.mean(np.log(vals)))))
        ax.plot(xs, ys, color=st["color"], marker=st["marker"], lw=2.2, ms=9,
                label=f"{d}: B {'earlier' if d == 'in' else 'later'}")
    ax.axvline(cfg.lag_pct(cfg.lags_ms[0]), color="#bbb", lw=1, ls=":")
    ax.set_xticks([cfg.lag_pct(C) for C in ctxs])
    ax.set_xticklabels([f"{cfg.lag_pct(C):g}%" + ("\nstationary" if abs(C - cfg.lags_ms[0]) < 1e-9
                                                   else "") for C in ctxs])
    ax.set_xlabel(f"context ΔT (test at {cfg.lag_pct(cfg.lags_ms[0]):g}%)")
    ax.set_ylabel("threshold (ms)"); ax.set_yscale("log"); ax.grid(alpha=.25)
    from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
    lo, hi = ax.get_ylim()
    nice = [v for v in (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70) if lo <= v <= hi] or [lo, hi]
    ax.yaxis.set_major_locator(FixedLocator(nice))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(title or "threshold against the context", fontsize=11)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150); plt.close(fig)
    return path
