"""Figures. Every one is a diagnostic first and a presentation graphic second.

The set is deliberately small, because each answers a question someone will ask:

    schematic       what does the stimulus actually look like at each dT?
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
        if r.get("phase") == "main":
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
    for name, fn in (("schematic", lambda p: schematic(cfg, path=p)),
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
