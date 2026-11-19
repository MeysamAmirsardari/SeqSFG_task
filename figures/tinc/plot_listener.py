"""Standalone, publication-style figures for one tinc session: d' by delay, and hits / false alarms.
Usage: python figures/tinc/plot_listener.py data/P00/tinc_session_03"""
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tinc.analysis import dprime_draws

folder = Path(sys.argv[1])
rows = list(csv.DictReader(open(folder / "dprime.csv")))
code = rows[0]["code"]
d = np.array([float(r["delay_ms"]) for r in rows])
x = np.arange(len(d))
f = lambda k: np.array([float(r[k]) for r in rows])

plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 9, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
INK, ACC, SOFT = "#1f2933", "#2f5d8a", "#c9d6e3"

# 1. d' by delay
fig, ax = plt.subplots(figsize=(3.5, 3.0))
for i, r in enumerate(rows):
    dr = dprime_draws(int(round(float(r["hit_rate"]) * int(r["n_signal"]))), int(r["n_signal"]),
                      int(round(float(r["fa_rate"]) * int(r["n_noise"]))), int(r["n_noise"]), seed=i)
    v = ax.violinplot(dr, [i], widths=0.7, showextrema=False, points=200)
    for b in v["bodies"]:
        b.set_facecolor(SOFT); b.set_edgecolor("none"); b.set_alpha(0.55)
lo95, hi95, lo75, hi75, mode = f("lo95"), f("hi95"), f("lo75"), f("hi75"), f("map")
ax.vlines(x, lo95, hi95, color=ACC, lw=1.0)
ax.vlines(x, lo75, hi75, color=ACC, lw=4.5)
ax.plot(x, mode, "o", ms=6.5, mfc="white", mec=ACC, mew=1.6, zorder=4)
sl, ic, *_ = stats.linregress(d, f("d_prime"))
ax.plot(x, ic + sl * d, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=1)
ax.axhline(0, color="0.55", lw=0.7)
ax.set_xticks(x, [f"{v:g}" for v in d])
ax.set_xlim(-0.6, len(d) - 0.4)
ax.set_ylim(-0.5, 2.5)
ax.set_xlabel("A–B onset asynchrony (ms)")
ax.set_ylabel("Sensitivity, d′")
ax.set_title(f"{code}: A and B roved together", loc="left", fontsize=9.5, color=INK)
ax.text(0.98, 0.97, f"slope {sl * 100:+.2f} d′ per 100 ms".replace("-", "\u2212"), transform=ax.transAxes,
        ha="right", va="top", fontsize=8, color=INK)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(folder / f"dprime_{code}.{ext}", dpi=300)

# 2. hits and false alarms
fig, ax = plt.subplots(figsize=(3.5, 3.0))
for k, lab, c, m in (("hit_rate", "Hits", ACC, "o"), ("fa_rate", "False alarms", "#8a96a3", "s")):
    p, n = f(k), f("n_signal" if k == "hit_rate" else "n_noise")
    lo, hi = stats.beta.ppf(0.025, p * n + 1, n - p * n + 1), stats.beta.ppf(0.975, p * n + 1, n - p * n + 1)
    ax.vlines(x, lo, hi, color=c, lw=1.0)
    ax.plot(x, p, "-", color=c, lw=1.2)
    ax.plot(x, p, m, ms=6, mfc="white", mec=c, mew=1.6, label=lab, zorder=4)
ax.set_xticks(x, [f"{v:g}" for v in d])
ax.set_xlim(-0.6, len(d) - 0.4)
ax.set_ylim(0, 1)
ax.set_xlabel("A–B onset asynchrony (ms)")
ax.set_ylabel("Proportion “yes”")
ax.set_title(f"{code}: hits and false alarms", loc="left", fontsize=9.5, color=INK)
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(folder / f"hits_fa_{code}.{ext}", dpi=300)
print("ok")

# 3. d' by delay, with Micheyl et al. (2013, Exp. 3) group on the same axes
from tinc.analysis import MICHEYL_EXP3_FIG7 as MH
GREY = "#8a96a3"
fig, ax = plt.subplots(figsize=(3.8, 3.1))
dx = 0.13
md = [MH[float(v)] for v in d]
mx = x - dx
ax.vlines(mx, [m[2][0] for m in md], [m[2][1] for m in md], color=GREY, lw=1.0)
ax.vlines(mx, [m[1][0] for m in md], [m[1][1] for m in md], color=GREY, lw=4.5)
ax.plot(mx, [m[0] for m in md], "-", color=GREY, lw=1.2, zorder=3)
ax.plot(mx, [m[0] for m in md], "s", ms=6, mfc="white", mec=GREY, mew=1.6, zorder=4,
        label="Micheyl et al. (2013), A roved alone")
px = x + dx
ax.vlines(px, lo95, hi95, color=ACC, lw=1.0)
ax.vlines(px, lo75, hi75, color=ACC, lw=4.5)
ax.plot(px, mode, "-", color=ACC, lw=1.2, zorder=3)
ax.plot(px, mode, "o", ms=6.5, mfc="white", mec=ACC, mew=1.6, zorder=4,
        label=f"{code}, A and B roved together")
ax.axhline(0, color="0.55", lw=0.7)
ax.set_xticks(x, [f"{v:g}" for v in d])
ax.set_xlim(-0.6, len(d) - 0.4)
ax.set_ylim(-0.5, 3.5)
ax.set_xlabel("A–B onset asynchrony (ms)")
ax.set_ylabel("Sensitivity, d′")
ax.legend(frameon=False, fontsize=7.5, loc="upper left", handlelength=1.2)
fig.text(0.01, 0.005, "Micheyl: group, read off their Fig. 7 (±0.1). Bars: 75% and 95% posterior intervals.",
         fontsize=6, color="0.45")
fig.tight_layout(rect=(0, 0.03, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(folder / f"dprime_vs_micheyl_{code}.{ext}", dpi=300)
