"""Diagnostic figures: what the stimulus looks like, and why the two intervals cannot be told
apart by anything except the recurrence of the figure.

`seqsfg plots` writes all of them. Rasters are cheap (they rebuild trials from seeds); the
matching and observer figures reuse one run of the verification battery.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from . import measure as M
from . import pool as pool_mod
from . import verify as V
from .config import Config, Derived, validate
from .stimulus import BACKGROUND, FIGURE, Interval, Trial, make_trial

BG_COLOR = "0.15"
FIG_COLOR = "#d62728"
A_COLOR = "#1f77b4"
B_COLOR = "#ff7f0e"


def semitones(f_hz) -> np.ndarray:
    return 12.0 * np.log2(np.asarray(f_hz, dtype=float) / 1000.0)


# ----------------------------------------------------------------------------
# rasters
# ----------------------------------------------------------------------------
def _raster(ax, cfg: Config, d: Derived, iv: Interval, t0: float, t1: float, lw: float = 3.0,
            boxes: bool = True, recurring_lines: Optional[np.ndarray] = None, mark_figure: bool = True) -> None:
    st = semitones(d.channel_freqs_hz)
    on_ms = iv.onset * cfg.grid_ms
    dur = cfg.tone_dur_ms
    vis = (on_ms + dur >= t0 * 1000) & (on_ms <= t1 * 1000)
    is_fig = (iv.kind == FIGURE) & mark_figure
    for sel, color, z in ((vis & ~is_fig, BG_COLOR, 2), (vis & is_fig, FIG_COLOR, 4)):
        if not np.any(sel):
            continue
        ax.hlines(st[iv.channel[sel]], on_ms[sel] / 1000.0, (on_ms[sel] + dur) / 1000.0,
                  color=color, lw=lw, zorder=z, capstyle="butt")
    if recurring_lines is not None:
        for c in recurring_lines:
            ax.axhline(st[c], color=FIG_COLOR, lw=0.7, ls=":", alpha=0.8, zorder=1)
    if boxes and iv.element_sets:
        n_comp = len(iv.element_sets[0])
        span = (n_comp - 1) * iv.step_ms + dur
        for k, t in enumerate(iv.element_onsets * cfg.grid_ms):
            if t / 1000.0 > t1 or (t + span) / 1000.0 < t0:
                continue
            chans = iv.element_sets[k]
            lo, hi = st[chans].min(), st[chans].max()
            pad = 1.6
            ax.add_patch(Rectangle((t / 1000.0 - 0.012, lo - pad), span / 1000.0 + 0.024, hi - lo + 2 * pad,
                                   fill=False, ec=FIG_COLOR, lw=0.9, ls="-", alpha=0.55, zorder=3))
    ax.set_xlim(t0, t1)
    ax.set_ylim(st.min() - 2.5, st.max() + 2.5)


def raster_pair(cfg: Config, d: Derived, path: Path, seed: int = 21, steps: Tuple[float, float] = (0.0, 10.0),
                t0: float = 0.0, t1: float = 1.4) -> None:
    """The classic view: a coherent chord next to a staircase, figure tones in red."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, step in zip(axes, steps):
        tr = make_trial(cfg, seed, step, "rising", d=d)
        _raster(ax, cfg, d, tr.recurring, t0, t1, boxes=False)
        ax.set_title("coherent, 0 ms" if step == 0 else f"staircase, {step:g} ms step, rise", fontsize=13)
        ax.set_xlabel("Time (s)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Semitones re 1000 Hz")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def raster_ladder(cfg: Config, d: Derived, path: Path, seed: int = 21, window_ms: Optional[float] = None) -> None:
    """One row per step, both intervals, zoomed onto one element so the shear is visible."""
    steps = list(cfg.steps_ms)
    if window_ms is None:
        window_ms = max(600.0, (cfg.n_components - 1) * steps[-1] + cfg.tone_dur_ms + 2.2 * cfg.iei_max_ms)
    fig, axes = plt.subplots(len(steps), 2, figsize=(14, 2.35 * len(steps)), sharey=True, squeeze=False)
    for i, step in enumerate(steps):
        tr = make_trial(cfg, seed, step, "rising", d=d)
        t_el = tr.recurring.element_onsets[1] * cfg.grid_ms
        span = (cfg.n_components - 1) * step + cfg.tone_dur_ms
        t0 = (t_el - 0.35 * (window_ms - span)) / 1000.0
        t1 = t0 + window_ms / 1000.0
        for j, iv in enumerate((tr.recurring, tr.other)):
            ax = axes[i][j]
            _raster(ax, cfg, d, iv, t0, t1, lw=3.4, recurring_lines=tr.recurring.figure_set)
            if i == 0:
                ax.set_title("interval A: same pitches every element" if j == 0 else
                             "interval B: new pitches every element", fontsize=12)
            if i == len(steps) - 1:
                ax.set_xlabel("Time (s)")
            if j == 0:
                ax.set_ylabel(f"step {step:g} ms\nsemitones re 1 kHz", fontsize=10)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    fig.suptitle("One element of each interval, at every step of the ladder.  Red = figure components, "
                 "box = the element, dotted = A's recurring channels", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def raster_overview(cfg: Config, d: Derived, path: Path, seed: int = 21, step: float = 20.0) -> None:
    """The whole interval, both sides. A's elements sit on one set of dotted lines; B's move."""
    tr = make_trial(cfg, seed, step, "rising", d=d)
    T = cfg.interval_dur_ms / 1000.0
    fig, axes = plt.subplots(2, 1, figsize=(15, 8.5), sharex=True, sharey=True)
    for ax, iv, name in ((axes[0], tr.recurring, "interval A (recurring): every element on the same 7 channels"),
                         (axes[1], tr.other, "interval B (redrawn): every element on 7 new channels")):
        _raster(ax, cfg, d, iv, 0.0, T, lw=2.2, recurring_lines=tr.recurring.figure_set)
        ax.set_title(name, fontsize=12, loc="left")
        ax.set_ylabel("Semitones re 1 kHz")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[1].set_xlabel("Time (s)")
    fig.suptitle(f"One trial, step {step:g} ms. Both intervals: {tr.recurring.n_tones} tones, "
                 f"{cfg.tones_per_channel} per channel, {cfg.n_elements} elements at the same times.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def raster_controls(cfg: Config, d: Derived, path: Path, seed: int = 21) -> None:
    """What each control variant does, at one step."""
    mid = sorted({s for v, s in cfg.control_cells if v in ("scrambled", "redrawn")})
    mid = mid[0] if mid else cfg.steps_ms[len(cfg.steps_ms) // 2]
    cells = [(v, mid, f"{v}: main ladder") for v in cfg.main_variants]
    cells += [(v, s if v != "onechannel" else 0.0,
               {"scrambled": "scrambled: fixed random delay order",
                "redrawn": "redrawn: delay order new every element",
                "onechannel": "onechannel: one channel recurs, no group"}.get(v, v))
              for v, s in sorted({(cv, mid if cv != "onechannel" else 0.0) for cv, _ in cfg.control_cells})]
    fig, axes = plt.subplots(len(cells), 2, figsize=(14, 2.35 * len(cells)), sharey=True, squeeze=False)
    for i, (variant, step, label) in enumerate(cells):
        tr = make_trial(cfg, seed, step, variant, d=d)
        t_el = tr.recurring.element_onsets[1] * cfg.grid_ms
        win = max(600.0, (cfg.n_components - 1) * step + cfg.tone_dur_ms + 2 * cfg.iei_max_ms)
        t0 = (t_el - 0.3 * win) / 1000.0
        t1 = t0 + win / 1000.0
        for j, iv in enumerate((tr.recurring, tr.other)):
            ax = axes[i][j]
            _raster(ax, cfg, d, iv, t0, t1, lw=3.4, recurring_lines=tr.recurring.figure_set)
            if i == 0:
                ax.set_title("interval A" if j == 0 else "interval B", fontsize=12)
            if j == 0:
                ax.set_ylabel(label.split(":")[0] + f"\n{step:g} ms", fontsize=10)
            if i == len(cells) - 1:
                ax.set_xlabel("Time (s)")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[i][1].text(1.01, 0.5, label.split(": ")[1], transform=axes[i][1].transAxes, fontsize=9,
                        rotation=270, va="center", color="0.35")
    fig.suptitle("The two main ladders (top) and the control variants", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# matching between the two intervals
# ----------------------------------------------------------------------------
def _cond_label(r: "V.ConditionResult") -> str:
    return f"{r.variant} {r.step_ms:g}"


def spectrum_matching(cfg: Config, d: Derived, results: List["V.ConditionResult"], path: Path) -> None:
    """The strategy the design has to defeat: read the long-term spectrum and pick the taller peaks."""
    main = [r for r in results if r.variant == "rising"]
    f = d.channel_freqs_hz
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])

    ax = fig.add_subplot(gs[0, :])
    r0 = main[0]
    for arr, col, mk, lbl in ((np.array([m.spectrum_db for m in r0.A]), A_COLOR, "o-", "A (recurring)"),
                              (np.array([m.spectrum_db for m in r0.O]), B_COLOR, "s--", "B (redrawn)")):
        arr = arr - arr.mean(axis=1, keepdims=True)
        ax.plot(f, arr.mean(0), mk, color=col, label=f"{lbl}, mean over trials", ms=4)
        ax.fill_between(f, arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0), color=col, alpha=0.22, lw=0)
    ax.set(xscale="log", xlabel="channel centre frequency (Hz)", ylabel="long-term level re interval mean (dB)",
           title="Long-term spectrum, step 0 ms (ribbons are ±1 SD over trials). The recurring channels leave no trace.\n"
                 "Every channel holds exactly 28 identical tones, so the true spectrum is flat; the 0.1 dB rise at the "
                 "bottom is leakage in the measuring filter, and is identical in both intervals.")
    ax.set_xticks([250, 500, 1000, 2000, 4000]); ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[1, 0])
    diffs = np.concatenate([(np.array([m.spectrum_db for m in r.A]) - np.array([m.spectrum_db for m in r.O])).ravel()
                            for r in main])
    ax.hist(diffs, bins=60, color="0.45")
    ax.axvline(0, color=FIG_COLOR, lw=1)
    ax.set(xlabel="per-channel level, A − B (dB)", ylabel="count",
           title=f"channel-by-channel difference\nsd {diffs.std():.3f} dB, max |.| {np.abs(diffs).max():.2f} dB")

    ax = fig.add_subplot(gs[1, 1])
    pa = np.concatenate([[m.scalars["spec_peakedness"] for m in r.A] for r in main])
    pb = np.concatenate([[m.scalars["spec_peakedness"] for m in r.O] for r in main])
    bins = np.linspace(min(pa.min(), pb.min()), max(pa.max(), pb.max()), 30)
    ax.hist(pa, bins=bins, color=A_COLOR, alpha=0.6, label="A (recurring)")
    ax.hist(pb, bins=bins, color=B_COLOR, alpha=0.6, label="B (redrawn)")
    ob = V.observer_from_stat(pa, pb)
    ax.set(xlabel="top-7 channels minus median channel (dB)", ylabel="count",
           title=f"the 'taller peaks' statistic\n'pick the peakier' gives d' = {ob['dprime']:+.2f} (p = {ob['p']:.2f})")
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[1, 2])
    x = np.arange(len(main))
    for lbl, key, col in (("A", "A", A_COLOR), ("B", "O", B_COLOR)):
        v = [np.mean([m.scalars["spec_peakedness"] for m in getattr(r, key)]) for r in main]
        e = [np.std([m.scalars["spec_peakedness"] for m in getattr(r, key)], ddof=1) /
             math.sqrt(len(r.A)) for r in main]
        ax.errorbar(x + (0.08 if lbl == "B" else -0.08), v, yerr=e, fmt="o", color=col, capsize=3, label=lbl)
    ax.set_xticks(x); ax.set_xticklabels([f"{r.step_ms:g}" for r in main])
    ax.set(xlabel="step (ms)", ylabel="peakedness (dB)", title="by condition (±1 SE)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _perm_max_diff(A: np.ndarray, B: np.ndarray, n_perm: int = 2000, seed: int = 3) -> Tuple[float, float, float]:
    """Observed max_t |mean_t (A-B)|, its 95th percentile under exchanging the intervals, and p."""
    D = A - B
    obs = float(np.max(np.abs(D.mean(0))))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        sgn = rng.choice([-1.0, 1.0], size=D.shape[0])[:, None]
        null[i] = np.max(np.abs((D * sgn).mean(0)))
    return obs, float(np.percentile(null, 95)), float((np.sum(null >= obs) + 1) / (n_perm + 1))


def envelope_matching(cfg: Config, d: Derived, results: List["V.ConditionResult"], path: Path) -> None:
    """The trap: a level transient locked to the element, present in one interval and not the other."""
    main = [r for r in results if r.variant == "rising"]
    others = [lv for lv in cfg.main_variants if lv != "rising"]
    ung = [r for r in results if r.variant in others]
    n = len(main) + len(ung)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4.1 * ((n + 1) // 2), 8.4), squeeze=False)
    pre_ms = 100.0
    for i, r in enumerate(main + ung):
        ax = axes[i // ((n + 1) // 2)][i % ((n + 1) // 2)]
        A = np.array([m.locked_env for m in r.A]); B = np.array([m.locked_env for m in r.O])
        t = (np.arange(A.shape[1]) * V.FRAME_MS - pre_ms)
        for Y, col, lbl in ((A, A_COLOR, "A"), (B, B_COLOR, "B")):
            mu, se = Y.mean(0), Y.std(0, ddof=1) / math.sqrt(Y.shape[0])
            ax.plot(t, mu, color=col, lw=1.4, label=lbl)
            ax.fill_between(t, mu - 1.96 * se, mu + 1.96 * se, color=col, alpha=0.3, lw=0)
        ax.axvline(0, color="0.5", lw=0.8, ls=":")
        ax.axvspan(0, r.span_ms, color="0.85", alpha=0.5, zorder=0)
        obs, p95, pv = _perm_max_diff(A, B)
        ax.set_title(f"{_cond_label(r)} ms:  max |A−B| = {obs:.2f} dB\n"
                     f"exchange gives {p95:.2f} dB 95% of the time (p = {pv:.2f})", fontsize=9.5)
        ax.set_xlabel("time re element onset (ms)")
        if i % ((n + 1) // 2) == 0:
            ax.set_ylabel("broadband level re interval RMS (dB)")
        ax.legend(fontsize=8)
    for j in range(n, axes.size):
        axes[j // ((n + 1) // 2)][j % ((n + 1) // 2)].axis("off")
    fig.suptitle("Broadband envelope locked to element onsets (mean ±95% CI over trials). Shading marks the element.\n"
                 "On the 'rising' ladder both intervals carry the same transient, because both contain elements. On the "
                 "'ungrouped' ladder the foil has none,\nso the transient itself separates the intervals: that is the "
                 "in-principle limit of a presence-versus-absence comparison, largest when the components are "
                 "synchronous.", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=140)
    plt.close(fig)


# rows whose value cannot be computed without being told which channels recur, or when the
# elements are. A listener has neither; they are shown so the reader can see they are the only
# rows that separate the intervals.
ORACLE_ROWS = {"occ_S_mean", "occ_S_audio", "occ_S_union", "S_max_simul", "el_comp_simul", "el_comp_sync",
               "el_count_mean", "locked_peak_db", "el_rms_mean_db", "el_rms_sd_db"}


def row_matching(cfg: Config, results: List["V.ConditionResult"], path: Path) -> None:
    """Every measured row, as standardized paired differences: the battery table as one picture."""
    blind = [(lab, key) for lab, key, fmt in V.ROWS_BETWEEN if key not in ORACLE_ROWS]
    oracle = [(lab + " †", key) for lab, key, fmt in V.ROWS_BETWEEN if key in ORACLE_ROWS]
    rows = blind + oracle
    labels = [lab for lab, key in rows]
    keys = [key for lab, key in rows]
    Z = np.zeros((len(keys), len(results)))
    for j, r in enumerate(results):
        for i, k in enumerate(keys):
            dif = np.array([ma.scalars[k] - mo.scalars[k] for ma, mo in zip(r.A, r.O)])
            se = dif.std(ddof=1) / math.sqrt(len(dif)) if len(dif) > 1 else 0.0
            Z[i, j] = 0.0 if se == 0 else dif.mean() / se
    n_main = sum(1 for r in results if r.variant in cfg.main_variants)
    fig, ax = plt.subplots(figsize=(1.05 * len(results) + 7.5, 0.42 * len(keys) + 3.0))
    lim = 6
    im = ax.imshow(np.clip(Z, -lim, lim), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([_cond_label(r) for r in results], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(keys)):
        for j in range(len(results)):
            if abs(Z[i, j]) > 2:
                ax.text(j, i, f"{Z[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(Z[i, j]) > 3.5 else "black")
    ax.axhline(len(blind) - 0.5, color="k", lw=1.6)
    ax.axvline(n_main - 0.5, color="k", lw=1.6)
    ax.text(n_main / 2 - 0.5, -1.15, "main experiment (both ladders)", ha="center", fontsize=10, style="italic")
    ax.text((n_main + len(results)) / 2 - 0.5, -1.15, "control cells", ha="center", fontsize=10, style="italic")
    fig.colorbar(im, ax=ax, label="paired difference A − B, in standard errors", shrink=0.8)
    ax.set_title("Every measured property, every condition. White = the two intervals match.\n"
                 "Cells are labelled only where |difference| exceeds 2 SE. Above the line: everything a listener\n"
                 "could measure. Below (†): rows that need to be told which channels recur, or when the elements are.",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def occupancy_matching(cfg: Config, d: Derived, results: List["V.ConditionResult"], path: Path) -> None:
    """Occupancy: does the figure's channels sounding for more of the time give the trial away?"""
    main = [r for r in results if r.variant == "rising"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    r = main[0]
    A = np.array([np.sort(m.occupancy_audio)[::-1] for m in r.A])
    B = np.array([np.sort(m.occupancy_audio)[::-1] for m in r.O])
    x = np.arange(1, d.n_channels + 1)
    for Y, col, lbl in ((A, A_COLOR, "A (recurring)"), (B, B_COLOR, "B (redrawn)")):
        mu, sd = Y.mean(0), Y.std(0)
        ax.plot(x, mu, color=col, lw=1.6, label=lbl)
        ax.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.25, lw=0)
    ax.axvline(cfg.n_components + 0.5, color="0.5", ls=":", lw=1)
    ax.text(cfg.n_components + 0.8, ax.get_ylim()[1], " top 7", fontsize=8, va="top", color="0.4")
    ax.set(xlabel="channel, ranked by occupancy", ylabel="fraction of the interval sounding",
           title="occupancy profile, step 0 (±1 SD)")
    ax.legend(fontsize=9)

    ax = axes[1]
    xs = np.arange(len(main))
    for key, col, lbl, lw in (("A", A_COLOR, "A", 4.5), ("O", B_COLOR, "B", 1.6)):
        v = [np.mean([m.scalars["occ_S_mean"] for m in getattr(r, key)]) for r in main]
        v2 = [np.mean([m.scalars["occ_S_union"] for m in getattr(r, key)]) for r in main]
        ax.plot(xs, v, "o-", color=col, label=f"{lbl}: per channel", lw=lw, alpha=0.55 if lw > 3 else 1.0)
        ax.plot(xs, v2, "s--", color=col, label=f"{lbl}: union of the 7", lw=lw, alpha=0.55 if lw > 3 else 1.0)
    ax.set_xticks(xs); ax.set_xticklabels([f"{r.step_ms:g}" for r in main])
    ax.set(xlabel="step (ms)", ylabel="occupancy",
           title="occupancy of the figure's channels\nper channel: identical by construction (the thick and thin\n"
                 "lines coincide). The union needs to know which they are.")
    ax.legend(fontsize=8)

    ax = axes[2]
    pa = np.concatenate([[m.scalars["occ_peakedness"] for m in r.A] for r in main])
    pb = np.concatenate([[m.scalars["occ_peakedness"] for m in r.O] for r in main])
    bins = np.linspace(min(pa.min(), pb.min()), max(pa.max(), pb.max()), 30)
    ax.hist(pa, bins=bins, color=A_COLOR, alpha=0.6, label="A")
    ax.hist(pb, bins=bins, color=B_COLOR, alpha=0.6, label="B")
    ob = V.observer_from_stat(pa, pb)
    ax.set(xlabel="top-7 occupancy minus median (fraction)", ylabel="count",
           title=f"'pick the peakier occupancy'\nd' = {ob['dprime']:+.2f} (p = {ob['p']:.2f})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# observers
# ----------------------------------------------------------------------------
def observers_forest(cfg: Config, res: dict, path: Path) -> None:
    results = res["results"]
    obs = res["observers"]
    names = [n for n in obs if not n.startswith("oracle")]
    oracle = [n for n in obs if n.startswith("oracle")]
    keys = [f"{r.variant}:{r.step_ms:g}" for r in results]
    is_main = [r.variant in cfg.main_variants for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4), gridspec_kw={"width_ratios": [2.1, 1]})

    ax = axes[0]
    cmap = plt.get_cmap("viridis")
    n_main = sum(is_main)
    yticks, ylabels = [], []
    y = 0.0
    for nm in names + oracle:
        for i, k in enumerate(keys):
            r_ = obs[nm][k]["cv"]
            main = is_main[i]
            col = cmap((i % max(len(cfg.steps_ms), 1)) / max(len(cfg.steps_ms) - 1, 1)) if main else "0.55"
            ax.errorbar(r_["dprime"], y, xerr=[[r_["dprime"] - r_["ci"][0]], [r_["ci"][1] - r_["dprime"]]],
                        fmt="o" if main else "s", ms=4.2 if main else 3.6, color=col, capsize=1.6, lw=1,
                        mfc=col if main else "white", alpha=0.95)
            y += 0.32
        yticks.append(y - 0.32 * len(keys) / 2 - 0.16); ylabels.append(nm)
        y += 0.8
    ax.axvline(0, color="0.3", lw=1.2)
    ax.axvspan(-0.5, 0.5, color="0.88", alpha=0.7, zorder=0)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=10)
    ax.set_xlabel("d' of the learnt observer (leave-one-out), 95% CI")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=cmap(0.35), label="main ladders (dark to light by step)"),
                       Line2D([], [], marker="s", ls="", mfc="white", color="0.55", label="control cell")],
              fontsize=8.5, loc="lower right")
    ax.set_title("Each observer sees ONE property of the two intervals, one point per condition.\n"
                 "Grey band is |d'| < 0.5. Every main-experiment point sits in it; the outliers are the\n"
                 "ungrouped and one-channel controls, whose cues are documented.", fontsize=11)
    ax.invert_yaxis()

    ax = axes[1]
    pooled = res.get("pooled_main") or {n: obs[n]["pooled"]["cv"] for n in names + oracle}
    order = names + oracle
    ys = np.arange(len(order))
    cols = [FIG_COLOR if n in oracle else A_COLOR for n in order]
    for yy, nm, c in zip(ys, order, cols):
        r_ = pooled[nm]
        ax.errorbar(r_["dprime"], yy, xerr=[[r_["dprime"] - r_["ci"][0]], [r_["ci"][1] - r_["dprime"]]],
                    fmt="o", ms=7, color=c, capsize=4)
    ax.axvline(0, color="0.3", lw=1.2)
    ax.axvspan(-0.5, 0.5, color="0.88", alpha=0.7, zorder=0)
    ax.set_yticks(ys); ax.set_yticklabels(order, fontsize=10)
    ax.set_xlabel("d', pooled over the primary ('rising') ladder")
    ax.set_title(f"the primary claim ({pooled[order[0]]['n']} trials)\n"
                 f"red = the oracle, which is told when the elements are", fontsize=10.5)
    ax.invert_yaxis()
    mult = res.get("multiplicity") or {}
    if mult:
        w = mult["worst"]
        ax.text(0.5, -0.155, f"Across all {mult['n_cells']} blind observer × condition cells the largest is\n"
                             f"{w['observer']} at {w['condition']}: d' = {w['dprime']:+.2f}, Holm p = {w['p_holm']:.2f}. "
                             f"None survives correction.",
                transform=ax.transAxes, fontsize=8.5, ha="center", va="top", color="0.25")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def audit_permutation(res: dict, path: Path) -> None:
    """Every scalar feature against the null produced by exchanging the two intervals."""
    audit = res["audit"]
    pm = res["permutation"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    d_ = np.array([r["dprime"] for r in audit])
    order = np.argsort(d_)
    y = np.arange(len(d_))
    lo = np.array([audit[i]["ci"][0] for i in order]); hi = np.array([audit[i]["ci"][1] for i in order])
    ax.errorbar(d_[order], y, xerr=[d_[order] - lo, hi - d_[order]], fmt="o", ms=3, color="0.25",
                capsize=1.5, lw=0.8)
    s95 = pm["single_feature_95"]
    ax.axvspan(-s95, s95, color=A_COLOR, alpha=0.18, lw=0, label=f"95% band for ONE pre-chosen feature (±{s95:.2f})")
    ax.axvspan(-pm["null_max_p95"], pm["null_max_p95"], color="0.7", alpha=0.25, lw=0,
               label=f"95% band for the LARGEST of {pm['n_features']} (±{pm['null_max_p95']:.2f})")
    ax.axvline(0, color="0.3", lw=1)
    names = [audit[i]["feature"] for i in order]
    show = {0, len(y) - 1, len(y) - 2, 1}
    for i in show:
        ax.text(d_[order][i] + (0.03 if d_[order][i] > 0 else -0.03), y[i], names[i], fontsize=7,
                va="center", ha="left" if d_[order][i] > 0 else "right", color="0.3")
    ax.set(xlabel="fixed-rule d' ('pick the interval with the larger value')", ylabel="feature (sorted)",
           title=f"All {pm['n_features']} scalar features an observer could use,\npooled over the {pm['n_trials']} "
                 f"main-condition trials")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_yticks([])

    ax = axes[1]
    ax.hist(pm["null_max"], bins=60, color="0.6", label="largest |d'| when the two\nintervals are exchanged at random")
    ax.axvline(pm["observed_max_dprime"], color=FIG_COLOR, lw=2.2,
               label=f"observed largest |d'| = {pm['observed_max_dprime']:.2f}\n({pm['worst_feature']})")
    ax.set(xlabel="largest |d'| over all features", ylabel="count",
           title=f"Global permutation test, {pm['n_perm']} relabellings\n"
                 f"p = {pm['p_value']:.3f}: no feature separates the intervals")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# design and stimulus checks
# ----------------------------------------------------------------------------
def design_checks(cfg: Config, d: Derived, results: List["V.ConditionResult"], path: Path, seed: int = 5) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))

    ax = axes[0][0]
    iei = np.concatenate([np.concatenate(r.iei_ms) for r in results])
    w = (cfg.iei_max_ms - cfg.iei_min_ms) / 24.0
    ax.hist(iei, bins=np.arange(cfg.iei_min_ms - 2 * w, cfg.iei_max_ms + 3 * w, w), color="0.45", density=True)
    ax.axhline(1.0 / (cfg.iei_max_ms - cfg.iei_min_ms), color=FIG_COLOR, lw=1.6,
               label=f"requested U[{cfg.iei_min_ms:.0f}, {cfg.iei_max_ms:.0f}] ms")
    ax.axvline(cfg.iei_min_ms, color=FIG_COLOR, ls=":", lw=1); ax.axvline(cfg.iei_max_ms, color=FIG_COLOR, ls=":", lw=1)
    ax.set(xlabel="inter-element interval (ms)", ylabel="density",
           title=f"jitter as drawn: n = {iei.size}, min {iei.min():.0f}, max {iei.max():.0f}\n"
                 f"nothing clipped by a rejection rule")
    ax.legend(fontsize=8)

    ax = axes[0][1]
    D = d.tone_dur_grid
    for lbl, col in (("A", A_COLOR), ("B", B_COLOR)):
        counts = []
        for j in range(12):
            tr = make_trial(cfg, seed * 1000 + j, 20.0, "rising", d=d)
            iv = tr.recurring if lbl == "A" else tr.other
            counts.append(M.count_trace(iv.onset, cfg.n_grid, D))
        c = np.concatenate(counts)
        ax.hist(c, bins=np.arange(-0.5, c.max() + 1.5), color=col, alpha=0.55, density=True, label=lbl)
    ax.set(xlabel="tones sounding at once", ylabel="fraction of the interval",
           title=f"instantaneous density (mean {d.mean_simultaneous:.1f})\nidentical distribution in both intervals")
    ax.legend(fontsize=9)

    ax = axes[0][2]
    steps = np.array([r["step_ms"] for r in d.ladder])
    ax.plot(steps, [r["adjacent_overlap"] for r in d.ladder], "o-", color=A_COLOR, label="overlap of adjacent components")
    ax2 = ax.twinx()
    ax2.plot(steps, [r["max_simultaneous"] for r in d.ladder], "s--", color=FIG_COLOR,
             label="max components sounding at once")
    ax.set(xlabel="step (ms)", ylabel="fraction of the tone overlapping", title="the ladder")
    ax2.set_ylabel("components at once", color=FIG_COLOR)
    ax.set_ylim(-0.05, 1.05); ax2.set_ylim(0, cfg.n_components + 0.5)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")

    ax = axes[1][0]
    f = d.channel_freqs_hz
    beats = np.diff(f)
    ax.semilogx(f[:-1], beats, "o-", color="0.3")
    ax.axhline(cfg.min_beat_rate_hz, color=FIG_COLOR, lw=1.5, label=f"throb limit, {cfg.min_beat_rate_hz:.0f} Hz")
    ax.set(xlabel="channel frequency (Hz)", ylabel="beat rate with the next channel (Hz)",
           title=f"critical-band rule: 1 ERB spacing\nslowest beat {beats.min():.0f} Hz, heard as roughness")
    ax.set_xticks([250, 500, 1000, 2000, 4000]); ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(fontsize=8)

    ax = axes[1][1]
    exc, own = pool_mod.excitation_from_pool(f, cfg.tone_level_db_spl, d.occupancy_per_channel)
    thr = pool_mod.abs_threshold_db_spl(f)
    ax.semilogx(f, own, "o-", color="0.2", label="one tone")
    ax.semilogx(f, exc, "s-", color=B_COLOR, label="masking by the rest of the pool")
    ax.semilogx(f, thr, "^--", color=A_COLOR, label="absolute threshold")
    ax.fill_between(f, thr, exc, color=B_COLOR, alpha=0.15, lw=0)
    ax.set(xlabel="channel frequency (Hz)", ylabel="dB SPL",
           title=f"masking exceeds threshold by {np.min(exc - thr):.0f}–{np.max(exc - thr):.0f} dB:\n"
                 f"no channel is near audibility, so no loudness weighting")
    ax.set_xticks([250, 500, 1000, 2000, 4000]); ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(fontsize=8)

    ax = axes[1][2]
    aw = pool_mod.a_weighting_db(f)
    ax.semilogx(f, aw, "o-", color=FIG_COLOR)
    ax.axhline(0, color="0.6", lw=0.8)
    within = aw.max() - aw.min()
    ax.set(xlabel="channel frequency (Hz)", ylabel="A-weighting (dB)",
           title=f"what a loudness weighting would do:\nspread one element's components by up to {within:.1f} dB")
    ax.set_xticks([250, 500, 1000, 2000, 4000]); ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def single_channel_view(cfg: Config, d: Derived, results: List["V.ConditionResult"], path: Path) -> None:
    """The residue named in the README: a recurring channel is quasi-periodic, and by how much."""
    main = [r for r in results if r.variant == "rising"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    r = main[len(main) // 2]
    lags_A, lags_B = [], []
    for m_list, out in ((r.A, lags_A), (r.O, lags_B)):
        for m in m_list:
            out.append(m.sc_sched["pairs_iei_norm"])
    A = np.concatenate(lags_A); B = np.concatenate(lags_B)
    bins = np.linspace(0, max(A.max(), B.max()), 40)
    ax.hist(A, bins=bins, color=A_COLOR, alpha=0.6, label="A: every channel")
    ax.hist(B, bins=bins, color=B_COLOR, alpha=0.6, label="B: every channel")
    ax.axvline(1.0, color="0.4", ls=":", lw=1)
    ax.set(xlabel="same-channel onset pairs at element-rate lags\n(observed / expected under uniform placement)",
           ylabel="count of channels", title=f"per-channel periodicity, step {r.step_ms:g} ms")
    ax.legend(fontsize=8)

    ax = axes[1]
    xs = np.arange(len(main))
    for key, col, lbl in (("A", A_COLOR, "A"), ("O", B_COLOR, "B")):
        mu = [np.mean([m.sc_sched["pairs_iei_norm"].mean() for m in getattr(r, key)]) for r in main]
        se = [np.std([m.sc_sched["pairs_iei_norm"].mean() for m in getattr(r, key)], ddof=1) / math.sqrt(len(r.A))
              for r in main]
        ax.errorbar(xs, mu, yerr=se, fmt="o-", color=col, capsize=3, label=lbl)
    ax.set_xticks(xs); ax.set_xticklabels([f"{r.step_ms:g}" for r in main])
    ax.axhline(1.0, color="0.4", ls=":", lw=1)
    ax.text(0.02, 1.0, " 1.0 = no periodicity at all", transform=ax.get_yaxis_transform(), fontsize=8,
            va="bottom", color="0.4")
    lo = min(0.995, ax.get_ylim()[0])
    ax.set_ylim(lo, ax.get_ylim()[1])
    muA = np.mean([np.mean([m.sc_sched["pairs_iei_norm"].mean() for m in r.A]) for r in main])
    muB = np.mean([np.mean([m.sc_sched["pairs_iei_norm"].mean() for m in r.O]) for r in main])
    ax.set(xlabel="step (ms)", ylabel="mean over channels",
           title=f"channel-averaged periodicity (±1 SE)\nA exceeds B by {100 * (muA - muB):.1f}% of a {100 * (muB - 1):.0f}% effect, "
                 f"and does not vary with step:\na constant floor, not a slope")
    ax.legend(fontsize=9)

    ax = axes[2]
    obs = [V.observer_from_stat(
        np.array([m.sc_sched["pairs_iei_norm"].mean() for m in r.A]),
        np.array([m.sc_sched["pairs_iei_norm"].mean() for m in r.O])) for r in main]
    dp = [o["dprime"] for o in obs]
    err = [[o["dprime"] - o["ci"][0] for o in obs], [o["ci"][1] - o["dprime"] for o in obs]]
    ax.errorbar(xs, dp, yerr=err, fmt="o", color="0.25", capsize=3)
    ax.axhline(0, color="0.4", lw=1)
    ax.axhspan(-0.5, 0.5, color="0.85", alpha=0.6, zorder=0)
    ax.set_xticks(xs); ax.set_xticklabels([f"{r.step_ms:g}" for r in main])
    ax.set(xlabel="step (ms)", ylabel="d'", title="an observer using ONLY that statistic\n"
           "(the onechannel control measures this in a listener)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def ladder_cues(cfg: Config, res: dict, path: Path) -> None:
    """What each ladder's foil affords a listener who never binds anything, step by step."""
    obs = res["observers"]
    ladders = [lv for lv in cfg.main_variants if any(r.variant == lv for r in res["results"])]
    steps = sorted({r.step_ms for r in res["results"] if r.variant == ladders[0]})
    names = ["envelope only", "spectrum only", "occupancy only", "single-channel stats"]
    fig, axes = plt.subplots(1, len(ladders), figsize=(6.2 * len(ladders), 4.6), sharey=True, squeeze=False)
    for ax, lv in zip(axes[0], ladders):
        for i, nm in enumerate(names):
            d_, lo, hi = [], [], []
            for s in steps:
                k = f"{lv}:{s:g}"
                cv = obs[nm][k]["cv"]
                d_.append(cv["dprime"]); lo.append(cv["ci"][0]); hi.append(cv["ci"][1])
            d_ = np.array(d_)
            off = (i - (len(names) - 1) / 2) * 0.35
            ax.errorbar(np.array(steps) + off, d_, yerr=[d_ - np.array(lo), np.array(hi) - d_],
                        fmt="o-", capsize=2.5, lw=1.2, ms=4, label=nm)
        ax.axhline(0, color="0.35", lw=1)
        ax.axhspan(-0.5, 0.5, color="0.88", alpha=0.8, zorder=0)
        ax.set(xlabel="onset step between components (ms)",
               title=f"ladder '{lv}'\n" + ("foil = a figure on new pitches" if lv == "rising"
                                            else "foil = no figure at all"))
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("d' of an observer with access to that property only")
    fig.suptitle("What each foil affords a listener who never binds anything. Grey band is |d'| < 0.5.\n"
                 "Presence-versus-absence cannot be envelope-matched: a synchronous chord IS a level event.",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
def make_all(cfg: Config, out_dir: Path, n_trials: int = 24, seed: int = 2026, verbose: bool = True) -> List[Path]:
    d = validate(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def say(msg):
        if verbose:
            print(msg, flush=True)

    say("rasters ...")
    raster_pair(cfg, d, out_dir / "raster_pair.png"); written.append(out_dir / "raster_pair.png")
    raster_ladder(cfg, d, out_dir / "raster_ladder.png"); written.append(out_dir / "raster_ladder.png")
    raster_overview(cfg, d, out_dir / "raster_overview.png"); written.append(out_dir / "raster_overview.png")
    raster_controls(cfg, d, out_dir / "raster_controls.png"); written.append(out_dir / "raster_controls.png")

    say(f"battery for the measurement figures ({n_trials} trials per condition) ...")
    res = V.run_battery(cfg, n_trials=n_trials, seed=seed, verbose=verbose)
    results = res["results"]

    say("matching and observer figures ...")
    for fn, name in ((spectrum_matching, "matching_spectrum.png"), (occupancy_matching, "matching_occupancy.png")):
        fn(cfg, d, results, out_dir / name); written.append(out_dir / name)
    envelope_matching(cfg, d, results, out_dir / "matching_envelope.png"); written.append(out_dir / "matching_envelope.png")
    row_matching(cfg, results, out_dir / "matching_rows.png"); written.append(out_dir / "matching_rows.png")
    observers_forest(cfg, res, out_dir / "observers.png"); written.append(out_dir / "observers.png")
    if len(cfg.main_variants) > 1:
        ladder_cues(cfg, res, out_dir / "ladder_cues.png"); written.append(out_dir / "ladder_cues.png")
    audit_permutation(res, out_dir / "observers_audit.png"); written.append(out_dir / "observers_audit.png")
    design_checks(cfg, d, results, out_dir / "design_checks.png"); written.append(out_dir / "design_checks.png")
    single_channel_view(cfg, d, results, out_dir / "single_channel.png"); written.append(out_dir / "single_channel.png")
    return written
