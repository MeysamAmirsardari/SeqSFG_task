"""P00's results for both tasks, in the style of figures/tinc/plot_listener.py:
violins of the uncertainty, 75% (thick) and 95% (thin) intervals, the estimate as an open circle,
and a dashed straight-line fit against the actual asynchrony.
  left:  tcmr, CMR by onset step (sessions pooled; figures/tcmr/plot_pooled.py's fit and bootstrap)
  right: tinc, d' by A delay (flat-prior posterior), with Micheyl et al. (2013) Exp. 3 in grey

Usage: python figures/schematics/plot_results.py
       (run figures/tcmr/plot_pooled.py first; writes data/P00/results_P00.{png,pdf})"""
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "figures/tcmr"))
from tcmr.procedure import fit
from plot_pooled import load, merge
from tinc.analysis import MICHEYL_EXP3_FIG7, dprime_draws

TCMR = [ROOT / "data/P00/tcmr_session_05", ROOT / "data/DEMO/tcmr_session_02"]
BOOT = ROOT / "data/P00/tcmr_pooled/P00_bootstrap.npz"
TINC = ROOT / "data/P00/tinc_session_03/dprime.csv"
OUT = ROOT / "data/P00/results_P00"

# tcmr: pooled fit and its bootstrap
S = [load(f) for f in TCMR]
th = fit(merge([s[2] for s in S]), lapse=S[0][1].lapse)["thresholds_db"]
z = np.load(BOOT)
draw = {str(n): z["draws"][:, i] for i, n in enumerate(z["names"])}
steps = np.array([0, 5, 10, 20, 40])
nm = lambda e, s: f"{e}_sync_0" if s == 0 else f"{e}_frozen_{s}"
cmr = np.array([th[nm("rand", s)] - th[nm("co", s)] for s in steps])
cmr_d = [draw[nm("rand", s)] - draw[nm("co", s)] for s in steps]
drop_d = cmr_d[0] - cmr_d[-1]

# tinc: d' posteriors
rows = list(csv.DictReader(open(TINC)))
dl = np.array([float(r["delay_ms"]) for r in rows])
dp = np.array([float(r["map"]) for r in rows])
dp_ml = np.array([float(r["d_prime"]) for r in rows])      # the trend uses the ML values, as plot_listener.py
dp_d = [dprime_draws(int(round(float(r["hit_rate"]) * int(r["n_signal"]))), int(r["n_signal"]),
                     int(round(float(r["fa_rate"]) * int(r["n_noise"]))), int(r["n_noise"]), seed=i)
        for i, r in enumerate(rows)]

plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 9, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
INK, ACC, SOFT, GREY = "#1f2933", "#2f5d8a", "#c9d6e3", "#9aa5b1"


def panel(ax, x, point, draws, col=ACC, soft=SOFT, mk="o", ms=6.5, violins=True, label=None):
    if violins:
        for xi, d in zip(x, draws):
            v = ax.violinplot(d[np.isfinite(d)], [xi], widths=0.7, showextrema=False, points=200)
            for b in v["bodies"]:
                b.set_facecolor(soft); b.set_edgecolor("none"); b.set_alpha(0.55)
    q = np.array([np.nanpercentile(d, [2.5, 12.5, 87.5, 97.5]) for d in draws])
    ax.vlines(x, q[:, 0], q[:, 3], color=col, lw=1.0)
    ax.vlines(x, q[:, 1], q[:, 2], color=col, lw=4.5)
    ax.plot(x, point, mk, ms=ms, mfc="white", mec=col, mew=1.6, zorder=4, label=label)


def trend(ax, x, t, y):
    sl, ic, *_ = stats.linregress(t, y)
    ax.plot(x, ic + sl * t, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=1)
    return sl


fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1))
fig.subplots_adjust(left=0.085, right=0.985, bottom=0.17, top=0.88, wspace=0.32)

x = np.arange(len(steps))
panel(a, x, cmr, cmr_d)
trend(a, x, steps, cmr)
a.axhline(0, color="0.55", lw=0.7)
a.set_xticks(x, [f"{s}" for s in steps]); a.set_xlim(-0.6, len(x) - 0.4)
a.set_ylim(-6, 15)
a.set_xlabel("Onset step between bands, ΔT (ms)")
a.set_ylabel("Masking release, CMR (dB)")
a.set_title("tcmr: P00, two sessions pooled", loc="left", fontsize=9.5, color=INK)
lo, hi = np.percentile(drop_d, [2.5, 97.5])
a.text(0.98, 0.97, f"drop from 0 to 40 ms: {cmr[0] - cmr[-1]:.1f} dB\n95%: {lo:.1f} to {hi:.1f}".replace("-", "−"),
       transform=a.transAxes, ha="right", va="top", fontsize=8, color=INK)

x = np.arange(len(dl))
mic = np.array([MICHEYL_EXP3_FIG7[d][0] for d in dl])
for xi, d in zip(x + 0.28, dl):                       # their 75% and 95% intervals, set beside
    _, q75, q95 = MICHEYL_EXP3_FIG7[d]
    b.vlines(xi, *q95, color=GREY, lw=0.9); b.vlines(xi, *q75, color=GREY, lw=3.5)
b.plot(x + 0.28, mic, "s", ms=5, mfc="white", mec=GREY, mew=1.4, zorder=4, label="Micheyl et al. (2013)")
panel(b, x, dp, dp_d, label="P00")
sl = trend(b, x, dl, dp_ml)
b.axhline(0, color="0.55", lw=0.7)
b.set_xticks(x, [f"{d:g}" for d in dl]); b.set_xlim(-0.6, len(x) - 0.2)
b.set_ylim(-0.5, 3.5)
b.set_xlabel("A–B onset asynchrony (ms)")
b.set_ylabel("Sensitivity, d′")
b.set_title("tinc: P00, A and B roved together", loc="left", fontsize=9.5, color=INK)
b.text(0.02, 0.97, f"slope {sl * 100:+.2f} d′ per 100 ms".replace("-", "−"), transform=b.transAxes,
       ha="left", va="top", fontsize=8, color=INK)
b.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, 0.9), handlelength=1.0, markerscale=0.9)

for ext in ("png", "pdf"):
    fig.savefig(OUT.with_suffix(f".{ext}"), dpi=300, facecolor="white")
print("CMR", np.round(cmr, 1), "drop", round(cmr[0] - cmr[-1], 1), " d'", np.round(dp, 2), "slope/100ms", round(sl * 100, 2))
