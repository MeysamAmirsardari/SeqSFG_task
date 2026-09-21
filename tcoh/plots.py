"""Figures. Every one is a diagnostic first and a presentation graphic second.

The set is deliberately small, because each answers a question someone will ask:

    trial           what is the listener actually asked, on one trial?
    pilot_curve     the descriptive pilot's one output: threshold against onset lag
    schematic       what does the stimulus look like at each dT?
    envelopes       what does the coherence model see, and why does the index move?
    prediction      what is being predicted, with the spread across defensible readings
    tracks          did the staircases converge, or did they wander?
    curve           the result: kappa against dT, with the model on the same axis
    controls        coherent against scrambled -- the interaction H2 tests

Nothing here computes anything the report does not also compute. A figure that disagrees with
the text would mean one of them is wrong, so they share the same functions.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import Config, validate
from .stimulus import a_onsets, b_onsets, scramble_onsets, scramble_window


def _mpl():
    """matplotlib.pyplot, with a file-writing backend only when there is nothing to display on.

    Calling matplotlib.use("Agg") unconditionally -- which this function did at first -- forces
    the backend even inside a notebook, where the inline backend is already set up. Figures then
    go nowhere: plt.show() silently does nothing and every cell comes out blank. So an inline or
    widget backend that is already in place is left alone.
    """
    import matplotlib
    backend = matplotlib.get_backend().lower()
    if not any(k in backend for k in ("inline", "ipympl", "widget", "nbagg")):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def schematic(cfg: Config, pcts: Optional[Sequence[float]] = None, path: Optional[Path] = None,
              delta_ms: float = 30.0):
    """Onset diagram of the standard and target intervals at each dT. The shift is exaggerated."""
    plt = _mpl()
    d = validate(cfg)
    pcts = list(pcts) if pcts is not None else sorted({c.lag_pct for c in d.conditions
                                                       if c.a_kind == "coherent"})
    fig, axes = plt.subplots(len(pcts), 1, figsize=(9, 1.35 * len(pcts) + 1.0), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, pcts):
        b = b_onsets(cfg)
        a = b + cfg.lag_ms(p)
        for on in a:
            ax.add_patch(plt.Rectangle((on, 0.05), cfg.tone_ms, 0.35, fc="#4477aa", ec="none"))
        for i, on in enumerate(b):
            last = i == b.size - 1
            ax.add_patch(plt.Rectangle((on, 0.55), cfg.tone_ms, 0.35,
                                       fc="#cc6677" if last else "#88aacc", ec="none"))
            if last:
                ax.add_patch(plt.Rectangle((on + delta_ms, 0.55), cfg.tone_ms, 0.35,
                                           fc="none", ec="#cc6677", ls="--", lw=1.4))
                ax.annotate("", (on + delta_ms, 1.0), (on, 1.0),
                            arrowprops=dict(arrowstyle="->", color="#cc6677", lw=1.2))
                ax.text(on + delta_ms / 2, 1.08, "delta", ha="center", va="bottom",
                        fontsize=7, color="#cc6677")
        # patches do not autoscale; without an explicit x limit the axes stay at 0-1 and the
        # figure comes out blank, which is exactly how it first came out
        ax.set_xlim(0, float(b[-1]) + delta_ms + cfg.tone_ms + 40.0)
        ax.set_ylim(0, 1.3)
        ax.set_yticks([0.22, 0.72])
        ax.set_yticklabels([f"A {cfg.f_a_hz:.0f}", f"B {d.f_b_hz:.0f}"], fontsize=7)
        ax.set_ylabel(f"dT={p:g}%", rotation=0, ha="right", va="center", fontsize=8)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    axes[-1].set_xlabel("time within one interval (ms)")
    fig.suptitle("Two-tone sequences at each onset asynchrony.\n"
                 "Only the last B tone (outlined) moves; A never moves; B's grid is the same in "
                 "every row.", fontsize=9)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.86, bottom=0.09, hspace=0.45)
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def trial(cfg: Config, lag_pct: float = 50.0, delta_ms: float = 40.0,
          target_position: int = 2, direction: int = -1, path: Optional[Path] = None):
    """One whole trial, plus a zoom on the ending where the one difference lives.

    The `schematic` figure shows what changes across conditions; this shows what the listener
    is actually asked on a single trial, which is the question people ask first.

    The zoom is not decoration. At threshold delta is a few milliseconds against a trial nearly
    three seconds long -- about one part in a thousand -- so a whole-trial view cannot show the
    manipulation at all, and a first version of this figure duly showed a red arrow two pixels
    wide. The top row is for the structure; the bottom row is for the thing being detected.

    `direction` defaults to -1, an EARLY shift, for drawing rather than for science: a late
    shift at an intermediate dT slides the last B tone onto the last A tone, so the interval
    that is supposed to look wrong comes out looking more aligned than the other one. In the
    experiment the direction is randomised per trial and both are logged.
    """
    delta_ms = abs(delta_ms)
    signed = delta_ms * (1 if direction >= 0 else -1)
    plt = _mpl()
    d = validate(cfg)
    b = b_onsets(cfg)
    a = b + cfg.lag_ms(lag_pct)
    iv_ms = float(b[-1] + cfg.tone_ms + cfg.tail_ms)
    gap = cfg.isi_ms

    fig = plt.figure(figsize=(11, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.62, wspace=0.10)
    top = fig.add_subplot(gs[0, :])

    def draw(ax, off, shift, tone_w=None):
        for on in a:
            ax.add_patch(plt.Rectangle((off + on, 0.08), cfg.tone_ms, 0.34, fc="#4477aa", ec="none"))
        for i, on in enumerate(b):
            last = i == b.size - 1
            x = off + on + (shift if last else 0.0)
            ax.add_patch(plt.Rectangle((x, 0.56), cfg.tone_ms, 0.34,
                                       fc="#cc6677" if last else "#88aacc", ec="none"))
            if last and shift:
                ax.add_patch(plt.Rectangle((off + on, 0.56), cfg.tone_ms, 0.34, fc="none",
                                           ec="#cc3311", ls=(0, (3, 2)), lw=1.4))

    # ---- whole trial ----------------------------------------------------------
    for which in (1, 2):
        off = 0.0 if which == 1 else iv_ms + gap
        shift = signed if which == target_position else 0.0
        draw(top, off, shift)
        top.text(off + iv_ms / 2, 1.30,
                 f"interval {which}" + ("   \u2190 the odd one out" if shift else ""),
                 ha="center", va="center", fontsize=10.5,
                 color="#cc3311" if shift else "#333333",
                 fontweight="bold" if shift else "normal")
        top.add_patch(plt.Rectangle((off, 0.02), iv_ms, 0.94, fc="none", ec="#bbbbbb", lw=0.8))
    top.annotate("", (iv_ms + gap, 0.49), (iv_ms, 0.49),
                 arrowprops=dict(arrowstyle="<->", color="#888888", lw=1))
    top.text(iv_ms + gap / 2, 0.43, f"{gap:g} ms", ha="center", va="top", fontsize=8.5,
             color="#666666")
    top.set_xlim(-60, 2 * iv_ms + gap + 60)
    top.set_ylim(0, 1.5)
    top.set_yticks([0.25, 0.73])
    top.set_yticklabels([f"A  {cfg.f_a_hz:.0f} Hz", f"B  {d.f_b_hz:.0f} Hz"], fontsize=9)
    top.set_xlabel("time (ms)", fontsize=9)
    top.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        top.spines[sp].set_visible(False)

    # ---- the ending of each interval, magnified ---------------------------------
    win_lo = float(b[-2]) - 30.0 - delta_ms
    win_hi = float(b[-1]) + cfg.tone_ms + delta_ms + 40.0
    for col, which in enumerate((1, 2)):
        ax = fig.add_subplot(gs[1, col])
        shift = signed if which == target_position else 0.0
        draw(ax, 0.0, shift)
        ax.set_xlim(win_lo, win_hi)
        ax.set_ylim(0, 1.45)
        ax.set_yticks([0.25, 0.73])
        ax.set_yticklabels([f"A", f"B"] if col == 0 else ["", ""], fontsize=9)
        ax.tick_params(axis="y", length=0, labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        if shift:
            y = 1.10
            ax.annotate("", (float(b[-1]) + shift, y), (float(b[-1]), y),
                        arrowprops=dict(arrowstyle="->", color="#cc3311", lw=2))
            ax.text(float(b[-1]) + shift / 2, y + 0.05,
                    f"$\\delta$ = {delta_ms:g} ms {'early' if shift < 0 else 'late'}",
                    ha="center", va="bottom", fontsize=9.5, color="#cc3311")
            ax.text(float(b[-1]) + cfg.tone_ms / 2, 0.50, "where it should have been",
                    ha="center", va="center", fontsize=7.5, color="#cc3311")
            ax.set_title("interval 2, magnified  \u2014  the last B tone has moved",
                         fontsize=9.5, color="#cc3311")
        else:
            ax.text(float(b[-1]) + cfg.tone_ms / 2, 1.10, "in step", ha="center", va="bottom",
                    fontsize=9, color="#666666")
            ax.set_title("interval 1, magnified  \u2014  nothing has moved", fontsize=9.5,
                         color="#333333")
        ax.set_xlabel("time (ms)", fontsize=8)

    fig.suptitle("One trial.  The two sounds are identical except that in ONE of them the last "
                 "high tone is out of place.\n"
                 f"Which one?   Drawn at $\\Delta$T = {lag_pct:g}% and with $\\delta$ "
                 "exaggerated; at threshold it is a few milliseconds.", fontsize=10.5)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.84, bottom=0.09)
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def envelopes(cfg: Config, pcts: Optional[Sequence[float]] = None, path: Optional[Path] = None):
    """What the coherence model is handed, and the index it returns, at each dT."""
    plt = _mpl()
    from .model import coherence_matrix, sequence_envelope
    d = validate(cfg)
    pcts = list(pcts) if pcts is not None else sorted({c.lag_pct for c in d.conditions
                                                       if c.a_kind == "coherent"})
    fs = 1000.0
    fig, axes = plt.subplots(len(pcts), 1, figsize=(9, 1.1 * len(pcts) + 0.8), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, pcts):
        b = b_onsets(cfg)
        a = b + cfg.lag_ms(p)
        ea = sequence_envelope(a, cfg.tone_ms, cfg.ramp_ms, d.interval_ms, fs)
        eb = sequence_envelope(b, cfg.tone_ms, cfg.ramp_ms, d.interval_ms, fs)
        t = np.arange(ea.size)
        ax.fill_between(t, 0, ea, color="#4477aa", alpha=0.75, lw=0)
        ax.fill_between(t, 0, -eb, color="#cc6677", alpha=0.75, lw=0)
        r = coherence_matrix(np.stack([ea, eb]), fs)
        ax.text(0.995, 0.5, f"l2/l1 = {r.ratio:.3f}", transform=ax.transAxes,
                ha="right", va="center", fontsize=8)
        ax.set_ylabel(f"{p:g}%", rotation=0, ha="right", va="center", fontsize=8)
        ax.set_yticks([])
        ax.set_xlim(0, min(d.interval_ms, 1400))
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    axes[-1].set_xlabel("time (ms)")
    fig.suptitle("Channel envelopes (A up, B down) and the model's segregation index.\n"
                 "The index is near 0 when the channels rise and fall together and near 1 when "
                 "they do not.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def prediction(cfg: Config, path: Optional[Path] = None):
    """The pre-registered curve, with the band across readings the paper leaves open."""
    plt = _mpl()
    from .model import prediction_band
    d = validate(cfg)
    pcts = sorted({c.lag_pct for c in d.conditions if c.a_kind == "coherent"})
    band = prediction_band(pcts, tone_ms=cfg.tone_ms, soa_ms=cfg.soa_ms, n_tones=cfg.n_tones)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.fill_between(pcts, band["lo"], band["hi"], color="#4477aa", alpha=0.2,
                    label="across 8 readings of the filter bank")
    ax.plot(pcts, [d.model_curve[p] for p in pcts], "o-", color="#4477aa", lw=2,
            label="coherent sequence")
    for kind, style, col in (("scrambled", "s--", "#ddaa33"), ("pair_only", "^--", "#117733")):
        ctl = [(c.lag_pct, d.model_by_condition[c.name]) for c in d.conditions if c.a_kind == kind]
        if ctl:
            ax.plot([p for p, _ in ctl], [v for _, v in ctl], style, color=col, label=kind)
    ax.plot([0, 100], [0, 1], ":", color="#888888", lw=1, label="a straight line, for scale")
    ax.set_xlabel("onset asynchrony dT (% of half a period)")
    ax.set_ylabel("predicted segregation  $\\lambda_2/\\lambda_1$")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("The prediction, before any data.\nThe ORDER is robust; the heights are not.",
                 fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def tracks(cfg: Config, rows: Sequence[dict], path: Optional[Path] = None, max_panels: int = 12):
    """Every staircase, so a wandering one is visible rather than averaged away."""
    plt = _mpl()
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        # timed-out trials never entered the staircase, so drawing them would show a step the
        # rule did not take
        if r.get("phase") == "main" and not str(r.get("timed_out", "")).strip() in ("1", "True"):
            by[(r["condition"], r.get("track_id"))].append(r)
    keys = sorted(by, key=lambda k: (k[0], str(k[1])))[:max_panels]
    n = max(len(keys), 1)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.1 * nrow), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, k in zip(axes, keys):
        rs = sorted(by[k], key=lambda r: int(float(r.get("track_trial_index") or 0)))
        dv = [float(r["delta_ms"]) for r in rs]
        ok = [int(float(r["correct"])) for r in rs]
        ax.plot(dv, "-", color="#999999", lw=1)
        ax.plot([i for i, c in enumerate(ok) if c], [v for v, c in zip(dv, ok) if c], "o",
                ms=3, color="#117733")
        ax.plot([i for i, c in enumerate(ok) if not c], [v for v, c in zip(dv, ok) if not c], "x",
                ms=4, color="#cc3311")
        ax.set_yscale("log")
        ax.set_title(f"{k[0]}  track {k[1]}", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes[len(keys):]:
        ax.axis("off")
    fig.supxlabel("trial within track", fontsize=8)
    fig.supylabel("delta (ms)", fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def curve(cfg: Config, idx: dict, control_idx: Optional[dict] = None,
          path: Optional[Path] = None, simulated: bool = False):
    """The result: observed kappa and the model, on one axis, with the control line."""
    plt = _mpl()
    d = validate(cfg)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    if idx.get("ok"):
        p = np.array(idx["pcts"], dtype=float)
        k = np.array(idx["kappa"], dtype=float)
        lo = np.array([c[0] for c in idx["ci"]])
        hi = np.array([c[1] for c in idx["ci"]])
        ax.errorbar(p, k, yerr=[k - lo, hi - k], fmt="o-", color="#cc3311", lw=2, capsize=3,
                    label="observed $\\kappa$")
        ax.plot(p, [d.model_curve.get(x, np.nan) for x in p], "s--", color="#4477aa",
                label="model $\\lambda_2/\\lambda_1$")
    if control_idx and control_idx.get("ok"):
        p = np.array(control_idx["pcts"], dtype=float)
        k = np.array(control_idx["kappa"], dtype=float)
        lo = np.array([c[0] for c in control_idx["ci"]])
        hi = np.array([c[1] for c in control_idx["ci"]])
        ax.errorbar(p, k, yerr=[k - lo, hi - k], fmt="^-", color="#ddaa33", lw=1.5, capsize=3,
                    label=f"{control_idx.get('a_kind', 'control')} control")
    ax.axhline(0, color="#888888", lw=0.8, ls=":")
    ax.axhline(1, color="#888888", lw=0.8, ls=":")
    ax.text(2, 0.02, "as good as exact synchrony", fontsize=7, color="#555555", va="bottom")
    ax.text(2, 0.98, "no better than with the low tone off", fontsize=7, color="#555555", va="top")
    ax.set_xlabel("onset asynchrony dT (% of half a period)")
    ax.set_ylabel("$\\kappa$   (0 = one object, 1 = two)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    title = "Behavioural segregation against onset asynchrony"
    if simulated:
        title = "SIMULATED DATA -- " + title
    ax.set_title(title, fontsize=9, color="#cc3311" if simulated else "black")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def pilot_curve(cfg: Config, res, path: Optional[Path] = None, title: Optional[str] = None,
                simulated: bool = False):
    """The descriptive pilot's one figure: threshold against onset lag.

    Axis runs 100% on the LEFT to 0% on the RIGHT, matching Figure 8B of Elhilali et al. (2009),
    so the two can be laid side by side. That is the only relationship claimed between them.
    What is plotted here is a detection threshold in milliseconds, measured behaviourally; it is
    not lambda2/lambda1, it is not a rescaling of it, and no normalisation against a B-only
    ceiling is computed, because this preset does not measure one.

    Individual tracks are drawn as well as their geometric mean, because with two tracks per
    condition the mean of two numbers is not a summary anyone should read without seeing both.
    Tracks that did not converge, or whose averaged reversals sat on the delta clamp, are shown
    in their own marker and take no part in the mean.
    """
    plt = _mpl()
    d = validate(cfg)
    pcts = sorted({c.lag_pct for c in d.conditions if c.a_kind == "coherent"})
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    gm_x, gm_y, any_cens, n_missing = [], [], False, 0
    for p in pcts:
        r = next((v for v in res.values()
                  if v.a_kind == "coherent" and abs(v.lag_pct - p) < 1e-9
                  and v.n_precursor == cfg.n_precursor), None)
        if r is None:
            continue
        good = list(r.track_thresholds)
        cens = list(r.censored_thresholds or [])
        n_missing += max(0, r.n_tracks_attempted - len(good) - len(cens))
        if good:
            # nudged apart in x so two tracks with similar thresholds stay two visible points,
            # and so neither hides under the mean marker drawn on top of them
            off = np.linspace(-2.2, 2.2, len(good)) if len(good) > 1 else np.zeros(1)
            ax.plot(p + off, good, "o", ms=6, mfc="white", mec="#4477aa", mew=1.6, zorder=6)
            g = float(np.exp(np.mean(np.log(good))))
            gm_x.append(p)
            gm_y.append(g)
        for v in cens:
            any_cens = True
            ax.plot([p], [v], "v", ms=9, color="#cc3311", zorder=4)
            ax.annotate("", (p, v * 1.45), (p, v * 1.05),
                        arrowprops=dict(arrowstyle="->", color="#cc3311", lw=1.5))
    if gm_x:
        ax.plot(gm_x, gm_y, "-", color="#333333", lw=2, zorder=4)
        ax.plot(gm_x, gm_y, "o", ms=11, color="#333333", zorder=5,
                label="geometric mean of usable tracks")
    ax.plot([], [], "o", ms=6, mfc="white", mec="#4477aa", mew=1.6, label="individual track")
    if any_cens:
        ax.plot([], [], "v", ms=9, color="#cc3311",
                label=f"hit the {cfg.delta_max_ms:g} ms ceiling — a bound, not a threshold")
    if cfg.delta_max_ms:
        ax.axhline(cfg.delta_max_ms, color="#cc3311", ls="--", lw=1.2)
        ax.text(-10, cfg.delta_max_ms * 1.05, f"ceiling {cfg.delta_max_ms:g} ms", fontsize=8,
                color="#cc3311", va="bottom", ha="right")

    ax.set_xlim(112, -12)                # 100% alternation on the LEFT, 0% synchrony on the RIGHT
    ax.set_xticks(pcts)
    ax.set_yscale("log")
    ax.minorticks_off()
    lo = min([v for v in gm_y] or [2.0]) * 0.6
    ticks = [t for t in (1, 2, 3, 5, 8, 12, 20, 30, 50, 80) if lo <= t <= cfg.delta_max_ms * 1.9]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])
    ax.set_ylim(lo, cfg.delta_max_ms * 1.55)
    ax.set_xlabel("onset lag ΔT  (% of half the repetition period)\n"
                  "100% = alternating                                        0% = synchronous",
                  fontsize=9)
    ax.set_ylabel("final-B displacement threshold (ms)")
    sec = ax.secondary_xaxis("top")
    sec.set_xticks(pcts)
    sec.set_xticklabels([f"{cfg.lag_ms(p):.0f}" for p in pcts], fontsize=8)
    sec.set_xlabel("the same axis in milliseconds of lag", fontsize=8.5)

    t = title or "Displacement threshold against onset asynchrony"
    if simulated:
        t = "SIMULATED DATA — " + t
    if n_missing:
        t += f"\n{n_missing} track(s) did not converge and are not shown"
    ax.set_title(t, fontsize=10, color="#cc3311" if simulated else "black")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def controls(cfg: Config, inter: dict, path: Optional[Path] = None, simulated: bool = False):
    """The interaction H2 tests, with the prediction it is being tested against."""
    plt = _mpl()
    d = validate(cfg)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    if inter.get("ok"):
        p = np.array(inter["pcts"], dtype=float)
        v = np.array(inter["diff_log"], dtype=float)
        lo = np.array([c[0] for c in inter["diff_ci"]])
        hi = np.array([c[1] for c in inter["diff_ci"]])
        ax.errorbar(p, v, yerr=[v - lo, hi - v], fmt="o-", color="#cc3311", lw=2, capsize=3,
                    label="observed (log threshold: control minus coherent)")
    coh = {c.lag_pct: d.model_by_condition[c.name] for c in d.conditions if c.a_kind == "coherent"}
    ctl = {c.lag_pct: d.model_by_condition[c.name] for c in d.conditions if c.a_kind == "scrambled"}
    sh = sorted(set(coh) & set(ctl))
    if len(sh) >= 2:
        ax2 = ax.twinx()
        ax2.plot(sh, [ctl[x] - coh[x] for x in sh], "s--", color="#4477aa",
                 label="predicted (model index difference)")
        ax2.set_ylabel("model index difference\n(different scale: compare shape, not height)",
                       color="#4477aa", fontsize=8)
        ax2.tick_params(axis="y", colors="#4477aa")
        ax2.spines["top"].set_visible(False)
        ax2.legend(fontsize=8, frameon=False, loc="upper right")
    ax.axhline(0, color="#888888", lw=1)
    if inter.get("ok"):
        ax.annotate(f"observed slope {inter['slope_per_pct'] * 100:+.2f} log units per 100% dT\n"
                    f"permutation p = {inter['p_slope_negative']:.4f} (one sided, predicted negative)",
                    (0.03, 0.04), xycoords="axes fraction", fontsize=8, color="#cc3311",
                    va="bottom")
    ax.set_xlabel("onset asynchrony dT (% of half a period)")
    ax.set_ylabel("log threshold difference")
    ax.legend(fontsize=8, frameon=False, loc="center left")
    title = ("A coherent low-tone sequence helps only when the tones are synchronous.\n"
             "A pedestal or local-overlap account predicts a flat line at zero.")
    if simulated:
        title = "SIMULATED DATA -- " + title
    ax.set_title(title, fontsize=9, color="#cc3311" if simulated else "black")
    ax.spines["top"].set_visible(False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def write_all(cfg: Config, outdir: Path, rows: Optional[Sequence[dict]] = None,
              idx: Optional[dict] = None, control_idx: Optional[dict] = None,
              inter: Optional[dict] = None, simulated: bool = False) -> List[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    made = []
    for name, fn in (("trial", lambda p: trial(cfg, path=p)),
                     ("schematic", lambda p: schematic(cfg, path=p)),
                     ("envelopes", lambda p: envelopes(cfg, path=p)),
                     ("prediction", lambda p: prediction(cfg, path=p))):
        p = outdir / f"tcoh_{name}.png"
        fn(p)
        made.append(p)
    if rows:
        p = outdir / "tcoh_tracks.png"
        tracks(cfg, rows, path=p)
        made.append(p)
    if idx:
        p = outdir / "tcoh_curve.png"
        curve(cfg, idx, control_idx, path=p, simulated=simulated)
        made.append(p)
    if inter:
        p = outdir / "tcoh_controls.png"
        controls(cfg, inter, path=p, simulated=simulated)
        made.append(p)
    return made
