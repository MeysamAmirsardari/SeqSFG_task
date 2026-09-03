"""Figures for the analysis (matplotlib, Agg backend)."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .analysis import pf
from .config import Config


VARIANT_COLOR = {"rising": "#1f77b4", "ungrouped": "#2ca02c", "scrambled": "#9467bd", "redrawn": "#8c564b",
                 "onechannel": "#7f7f7f"}
VARIANT_LABEL = {"rising": "rising: foil is a figure on new pitches (recurrence)",
                 "ungrouped": "ungrouped: foil has no figure at all (presence)",
                 "scrambled": "scrambled", "redrawn": "redrawn", "onechannel": "one channel"}


def psychometric(res: dict, cfg: Config, path: Path) -> None:
    curves = res.get("curves") or {}
    variants = [v for v in res.get("main_variants", ["rising"]) if v in curves]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    thr_pc = res.get("threshold_pc", 0.75)
    for v in variants:
        ps = curves[v]["psychometric"]
        rows = sorted([r for r in res["main"] if r["variant"] == v], key=lambda r: r["step_ms"])
        if not rows:
            continue
        col = VARIANT_COLOR.get(v, "k")
        s = np.array([r["step_ms"] for r in rows]); pc = np.array([r["pc"] for r in rows])
        lo = np.array([r["pc_ci"][0] for r in rows]); hi = np.array([r["pc_ci"][1] for r in rows])
        ax[0].errorbar(s, pc, yerr=[pc - lo, hi - pc], fmt="o", color=col, capsize=3,
                       label=VARIANT_LABEL.get(v, v))
        if ps.get("fit"):
            xs = np.linspace(s.min(), s.max(), 200)
            fpar = ps["fit"]
            ax[0].plot(xs, pf(xs, fpar["alpha"], fpar["beta"], fpar["lapse"]),
                       "-" if ps["reportable"] else "--", color=col,
                       alpha=1.0 if ps["reportable"] else 0.45)
        if ps.get("reportable"):
            ax[0].axvspan(ps["threshold_ci"][0], ps["threshold_ci"][1], color=col, alpha=0.12, lw=0)
            ax[0].axvline(ps["threshold"], color=col, alpha=0.6, lw=1)
        d = np.array([r["dprime"] for r in rows])
        dlo = np.array([r["dprime_ci"][0] for r in rows]); dhi = np.array([r["dprime_ci"][1] for r in rows])
        ax[1].errorbar(s, d, yerr=[d - dlo, dhi - d], fmt="s", color=col, capsize=3, label=v)
    ax[0].axhline(0.5, color="0.7", ls=":")
    ax[0].axhline(thr_pc, color="0.85", ls="--", lw=0.8)
    ax[0].set(xlabel="onset step between components (ms)", ylabel="proportion correct", ylim=(0.3, 1.02),
              title="one psychometric function per foil")
    ax[0].legend(loc="lower left", fontsize=7.5)
    ax[1].axhline(0, color="0.7", ls=":")
    ax[1].set(xlabel="onset step between components (ms)", ylabel="d'", title="sensitivity")
    ax[1].legend(fontsize=8)
    cmp_ = (res.get("comparison") or {})
    inter = cmp_.get("interaction") or {}
    td = cmp_.get("threshold_difference") or {}
    bits = []
    if inter:
        bits.append(f"step x ladder interaction p = {inter['p_interaction']:.3f}")
    if td.get("reportable"):
        bits.append(f"threshold difference {td['difference']:+.1f} ms "
                    f"[{td['ci'][0]:+.1f}, {td['ci'][1]:+.1f}]")
    elif td:
        bits.append("threshold difference not reportable")
    if bits:
        fig.text(0.5, 0.005, "   |   ".join(bits), ha="center", fontsize=9, color="0.25")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def timecourse(main: List[dict], cfg: Config, path: Path, window: int = 6) -> None:
    """Accuracy over the session, one panel per ladder, one line per condition."""
    t = [x for x in main if x["correct"] is not None]
    variants = [v for v in cfg.main_variants if any(x["variant"] == v for x in t)]
    steps = sorted({x["step_ms"] for x in t})
    fig, axes = plt.subplots(len(variants), 1, figsize=(11, 3.2 * len(variants)), sharex=True, sharey=True,
                             squeeze=False)
    cmap = plt.get_cmap("viridis")
    for ax, v in zip(axes[:, 0], variants):
        sel = [j for j, x in enumerate(t) if x["variant"] == v]
        for i, s in enumerate(steps):
            idx = np.array([j for j in sel if t[j]["step_ms"] == s])
            if idx.size == 0:
                continue
            c = np.array([t[j]["correct"] for j in idx], float)
            col = cmap(i / max(len(steps) - 1, 1))
            ax.plot(idx + 1, np.cumsum(c) / np.arange(1, c.size + 1), "-", color=col, lw=2,
                    label=f"{s:g} ms")
            if c.size >= window:
                run = np.convolve(c, np.ones(window) / window, mode="valid")
                ax.plot(idx[window - 1:] + 1, run, ":", color=col, lw=1, alpha=0.8)
        ax.axhline(0.5, color="0.7", ls=":")
        ax.set(ylabel="cumulative pc", ylim=(0, 1.02),
               title=f"ladder '{v}' (dotted: running window of {window})")
        ax.legend(fontsize=7.5, ncol=len(steps), loc="lower right")
    axes[-1, 0].set_xlabel("trial number in the main block (both ladders interleaved)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def controls(res: dict, cfg: Config, path: Path) -> None:
    """Control cells beside the main-block cell at the same step, on the primary ladder."""
    rows = res["control"]
    if not rows:
        return
    main = {(r["variant"], r["step_ms"]): r for r in res["main"]}
    ref_variant = cfg.main_variants[0]
    labels, pcs, los, his, ref = [], [], [], [], []
    for r in rows:
        labels.append(f"{r['variant']}\n{r['step_ms']:g} ms")
        pcs.append(r["pc"]); los.append(r["pc_ci"][0]); his.append(r["pc_ci"][1])
        m = main.get((ref_variant, r["step_ms"]))
        ref.append(m["pc"] if m else np.nan)
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(1.5 * len(rows) + 3, 4.2))
    ax.bar(x - 0.2, pcs, 0.4, yerr=[np.array(pcs) - np.array(los), np.array(his) - np.array(pcs)],
           capsize=3, label="control cell")
    ax.bar(x + 0.2, ref, 0.4, color="0.6", label=f"main block, '{ref_variant}' at the same step")
    ax.axhline(0.5, color="0.7", ls=":")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set(ylabel="proportion correct", ylim=(0, 1.02), title="control block")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
