"""tcmr_exp1: the burst levels in one interval, at every onset step, comodulated vs independent.
Levels are measured from the real stimulus (tcmr.stimulus.build); the same noise tokens are reused
across rows so only the timing changes. Based on figures/tcmr/task_visuals/repetition_levels.py.

Usage: python figures/schematics/repetition_levels.py"""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize, LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tcmr.config import Config, Condition
from tcmr.stimulus import build

cfg = Config.load(ROOT / "tcmr/configs/tcmr_exp1.json")
OUT = Path(__file__).resolve().parent
STEPS = [0, 5, 10, 20, 40]
ORDER = [-1, 2, 0, -2, 1]                   # one frozen order; the 1-kHz band keeps 0
NB, B = cfg.n_precursors + 1, cfg.burst_ms / 1000

levels = []
for env in ("co", "rand"):
    tr, _, stem = build(cfg, Condition(env, "frozen", 40), 65, [20261116, 25], n_intervals=1,
                        signal_interval=1, orders=[ORDER], stems=True)
    rms = np.zeros((NB, 5))
    for j in range(NB):
        for b in range(5):
            on = tr.target_on_s[0] - cfg.n_precursors * cfg.noi_ms / 1000 + j * cfg.noi_ms / 1000 \
                + (ORDER[b] * 0.04 if j < cfg.n_precursors else 0)
            k = round(on * cfg.sample_rate)
            rms[j, b] = np.sqrt(np.mean(stem[b, k:k + int(B * cfg.sample_rate)] ** 2))
    levels.append(rms)
lev = 20 * np.log10(np.array(levels) / np.max(levels))
lo = 4 * np.floor(lev.min() / 4)

plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 12, "axes.linewidth": 0.7,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "pdf.fonttype": 42})
cmap = LinearSegmentedColormap.from_list(
    "level", ["#e6eef6", "#a9c8e2", "#5f9acb", "#2f5d8a", "#132a45"])
norm = Normalize(lo, 0)
INK, TONE = "#1f2933", "#12c25a"
import matplotlib.patheffects as pe
HALO = [pe.Stroke(linewidth=5.5, foreground="white"), pe.Normal()]

fig, axs = plt.subplots(len(STEPS), 2, figsize=(13, 10.5), sharex=True, sharey=True)
fig.subplots_adjust(left=0.13, right=0.98, bottom=0.14, top=0.92, hspace=0.28, wspace=0.08)
for col, title in enumerate(("Comodulated", "Independent")):
    axs[0, col].set_title(title, loc="left", fontsize=17, pad=12, color=INK)
    for row, s in enumerate(STEPS):
        ax = axs[row, col]
        for j in range(NB):
            for b in range(5):
                on = 0.08 + j * cfg.noi_ms / 1000 + (ORDER[b] * s / 1000 if j < cfg.n_precursors else 0)
                ax.add_patch(Rectangle((on, b - 0.17), B, 0.34, fc=cmap(norm(lev[col, j, b])),
                                       ec="none"))
        for b in range(5):                        # a faint line per band: separate narrow channels
            ax.axhline(b, color="0.9", lw=0.6, zorder=0)
        t1 = 0.08 + cfg.n_precursors * cfg.noi_ms / 1000
        ax.plot([t1 - 0.004, t1 + B + 0.004], [2, 2], color=TONE, lw=3, solid_capstyle="butt", zorder=5, path_effects=HALO)
        ax.set_ylim(-0.6, 4.6)
        ax.set_xlim(0, 1.16)
        ax.set_yticks(range(5), ["0.25", "0.5", "1", "2", "4"])
        ax.tick_params(axis="y", length=0, labelsize=9.5)
        ax.spines["left"].set_visible(False)
        ax.set_xticks([0, 0.4, 0.8, 1.1])
        if col == 0:
            ax.text(-0.09, 2, f"{s} ms", ha="right", va="center", fontsize=15, color=INK)
    axs[-1, col].set_xlabel("Time (s)")
axs[0, 0].text(-0.09, 5.3, "ΔT", ha="right", fontsize=15, color=INK, style="italic")
kax = fig.add_axes([0.52, 0.04, 0.2, 0.04]); kax.axis("off"); kax.set_xlim(0, 1); kax.set_ylim(0, 1)
kax.plot([0, 0.12], [0.5, 0.5], color=TONE, lw=3, solid_capstyle="butt", path_effects=HALO)
kax.text(0.17, 0.5, "the 1-kHz tone", va="center", fontsize=12, color=INK)
fig.text(0.012, 0.53, "Band (kHz)", rotation=90, va="center", fontsize=12)

cax = fig.add_axes([0.13, 0.045, 0.3, 0.022])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
cb.set_ticks(np.arange(lo, 1, 4))
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=10, length=0)
cb.set_label("Burst level (dB re loudest)", fontsize=11, labelpad=4)

for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"repetition_levels.{ext}", dpi=200, facecolor="white")
print("level range", round(lev.min(), 1), "to 0 dB")
