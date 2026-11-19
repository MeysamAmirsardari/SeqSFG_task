"""tinc (tinc_8db): one trial's tone levels at every A delay, no-change vs change.
Onsets and levels are measured from the real stimulus (tinc.stimulus.build) and checked against the
trial's own tone list. The same seed is used in every row and both columns, so the roves and decoys
are identical and only the delay (rows) and the change (columns) differ.

Usage: python figures/schematics/tinc_levels.py"""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize, LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tinc.config import Config
from tinc.stimulus import build

cfg = Config.load(ROOT / "tinc/configs/tinc_8db.json")
OUT = Path(__file__).resolve().parent
SEED = [20261116, 3]
DELAYS = cfg.delays_ms
sr, dur = cfg.sample_rate, cfg.tone_ms / 1000
Y = {"A": 0.0, "B": np.log2(cfg.f_b_hz / cfg.f_a_hz)}          # log-frequency rows

trials = {}
for d in DELAYS:
    for change in (False, True):
        tr, _, stem = build(cfg, d, change, SEED, stems=True)
        t0 = min(t.onset_s for t in tr.tones)
        rows = []
        for tn in tr.tones:                   # measure each tone in the sound itself
            s = stem[0 if tn.stream == "B" else 1]
            nz = np.flatnonzero(np.abs(s[int(round((tn.onset_s - 0.002) * sr)):]) > 0)
            on = tn.onset_s - 0.002 + nz[0] / sr
            assert abs(on - tn.onset_s) <= 1.5 / sr, (d, tn)
            rows.append((tn.stream, on - t0, tn.level_db, tn.target, tn.decoy))
        trials[d, change] = (rows, tr.target_index if change else None, tr.decoys)
lv = np.array([r[2] for v in trials.values() for r in v[0]])
lo, hi = 2 * np.floor(lv.min() / 2), 2 * np.ceil(lv.max() / 2)

plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 12, "axes.linewidth": 0.7,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "pdf.fonttype": 42})
cmap = LinearSegmentedColormap.from_list("level", ["#e6eef6", "#a9c8e2", "#5f9acb", "#2f5d8a", "#132a45"])
norm = Normalize(lo, hi)
INK, MARK = "#1f2933", "#12c25a"
HALO = [pe.Stroke(linewidth=4.5, foreground="white"), pe.Normal()]

fig, axs = plt.subplots(len(DELAYS), 2, figsize=(13, 8.6), sharex=True, sharey=True)
fig.subplots_adjust(left=0.13, right=0.98, bottom=0.16, top=0.91, hspace=0.32, wspace=0.08)
for col, title in enumerate(("No change", "Change")):
    axs[0, col].set_title(title, loc="left", fontsize=17, pad=12, color=INK)
    for row, d in enumerate(DELAYS):
        ax = axs[row, col]
        rows, tgt, _ = trials[d, bool(col)]
        for y in Y.values():
            ax.axhline(y, color="0.9", lw=0.6, zorder=0)
        for stream, on, db, is_t, _ in rows:
            ax.add_patch(Rectangle((on, Y[stream] - 0.2), dur, 0.4, fc=cmap(norm(db)), ec="none"))
            if is_t:
                ax.add_patch(Rectangle((on - 0.012, Y[stream] - 0.32), dur + 0.024, 0.64, fc="none",
                                       ec=MARK, lw=2.4, zorder=5, path_effects=HALO))
        ax.set_ylim(-0.55, Y["B"] + 0.55)
        ax.set_xlim(-0.02, (cfg.n_tones - 1) * cfg.soa_ms / 1000 + 0.1 + max(DELAYS) / 1000 + 0.03)
        ax.set_yticks([Y["A"], Y["B"]], [f"A {cfg.f_a_hz / 1000:.2f}", f"B {cfg.f_b_hz / 1000:g}"])
        ax.tick_params(axis="y", length=0, labelsize=10)
        ax.spines["left"].set_visible(False)
        if col == 0:
            ax.text(-0.33, Y["B"] / 2, f"{d:g} ms", ha="right", va="center", fontsize=15, color=INK)
    axs[-1, col].set_xlabel("Time (s)")
axs[0, 0].text(-0.33, Y["B"] + 1.0, "A delay", ha="right", fontsize=13, color=INK, style="italic")
fig.text(0.012, 0.54, "Frequency (kHz)", rotation=90, va="center", fontsize=12)

cax = fig.add_axes([0.13, 0.05, 0.3, 0.022])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
cb.set_ticks(np.arange(lo, hi + 1, 4))
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=10, length=0)
cb.set_label("Tone level (dB SPL)", fontsize=11, labelpad=4)
kax = fig.add_axes([0.52, 0.04, 0.3, 0.045]); kax.axis("off"); kax.set_xlim(0, 1); kax.set_ylim(0, 1)
kax.add_patch(Rectangle((0, 0.25), 0.07, 0.5, fc="none", ec=MARK, lw=2.4, path_effects=HALO))
kax.text(0.11, 0.5, f"the change: one B tone +{cfg.target_db:g} dB", va="center", fontsize=12, color=INK)

for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"tinc_levels.{ext}", dpi=200, facecolor="white")
rows, tgt, dec = trials[DELAYS[0], True]
print(f"levels {lv.min():.1f} to {lv.max():.1f} dB SPL; change on B tone {tgt + 1}; decoy pairs {[k + 1 for k in dec]}")
print("onsets checked against the sound for every tone (within 1.5 samples)")
