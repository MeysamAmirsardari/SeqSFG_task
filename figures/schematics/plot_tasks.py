"""Simple schematics of the two tasks: tcmr (tcmr_exp1) and tinc (tinc_8db).
Drawn from the preset numbers; the noise envelopes and roves are illustrative draws.

Usage: python figures/schematics/plot_tasks.py   (writes tcmr_task.* and tinc_task.* beside this file)"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 8.5, "axes.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
INK, ACC, GREY, LIGHT, GREEN = "#1f2933", "#2f5d8a", "#8a96a3", "#d5dbe1", "#4f8a6b"


def trap(ax, t0, dur, ramp, y, h, **kw):
    ax.add_patch(Polygon([(t0, y - h / 2), (t0 + ramp, y + h / 2), (t0 + dur - ramp, y + h / 2),
                          (t0 + dur, y - h / 2)], closed=True, **kw))


def label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="right")


def box(fig, x, y, w, h, text, fc, ec, fs=7.5, color=INK):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008",
                                      transform=fig.transFigure, fc=fc, ec=ec, lw=0.8))
    fig.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color)


# ------------------------------------------------------------------------------------------- tcmr
def tcmr():
    c = json.loads((ROOT / "tcmr/configs/tcmr_exp1.json").read_text())
    bands, nb, noi, b, r, lead = c["bands_hz"], c["n_precursors"], c["noi_ms"], c["burst_ms"], c["ramp_ms"], c["lead_ms"]
    order = [+2, -1, 0, -2, +1]                      # one of the 24 frozen orders; 1 kHz keeps 0
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.62, wspace=0.32,
                          left=0.08, right=0.98, top=0.89, bottom=0.13)

    for k, dt in enumerate((0, 10, 40)):
        ax = fig.add_subplot(gs[0, k])
        for i, f in enumerate(bands):
            y = i
            for j in range(nb + 1):
                on = lead + j * noi + (order[i] * dt if j < nb else 0)
                final = j == nb
                trap(ax, on, b, r, y, 0.62, fc=INK if final else GREY, ec="none", alpha=1 if final else 0.85)
        trap(ax, lead + nb * noi, b, r, 2, 0.28, fc=ACC, ec="none", zorder=5)          # the tone
        if k == 0:
            ax.annotate("tone", (lead + nb * noi, 2), (lead + nb * noi - 170, 2.45), fontsize=7.5,
                        color=ACC, arrowprops=dict(arrowstyle="-", color=ACC, lw=0.8))
        ax.set_yticks(range(5), [f"{f / 1000:g}" for f in bands])
        ax.set_ylim(-0.7, 4.7)
        ax.set_xlim(0, lead + nb * noi + b + 30)
        ax.set_xticks([0, 500, 1000])
        ax.set_xlabel("Time (ms)")
        if k == 0:
            ax.set_ylabel("Band centre (kHz)")
        ax.set_title(f"ΔT = {dt} ms" + ("  (synchronous)" if dt == 0 else ""), fontsize=8.5, loc="left")
        if k == 0:
            label(ax, "a")
    fig.text(0.08, 0.975, "tcmr: is a 1-kHz tone in noise easier to hear when the noise bands are grouped?",
             fontsize=9.5, color=INK, fontweight="bold")

    # b: comodulated vs random envelopes of one burst
    rng = np.random.default_rng(3)
    t = np.linspace(0, b, 400)
    ramp = np.clip(np.minimum(t, b - t) / r, 0, 1)

    def env():
        x = np.convolve(rng.standard_normal(1400), np.hanning(120), "same")[500:900]
        return np.abs(x + 1j * np.convolve(rng.standard_normal(1400), np.hanning(120), "same")[500:900])
    ax = fig.add_subplot(gs[1, 0])
    shared = env()
    for side, (name, envs) in enumerate((("Comodulated", [shared] * 5), ("Random", [env() for _ in range(5)]))):
        for i in range(5):
            e = envs[i] * ramp
            e = e / e.max() * 0.8
            x0 = side * (b + 25)
            ax.fill_between(x0 + t, i, i + e, color=ACC if side == 0 else GREY, lw=0)
        ax.text(side * (b + 25) + b / 2, 5.15, name, ha="center", fontsize=7.8)
    ax.set_yticks(np.arange(5) + 0.3, [f"{f / 1000:g}" for f in bands])
    ax.set_ylim(-0.2, 5.6)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("Band (kHz)")
    ax.set_title("Envelopes of one burst", fontsize=8.5, loc="left")
    label(ax, "b")

    # c: the trial
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.set_title("One trial (3 intervals)", fontsize=8.5, loc="left")
    label(ax, "c")
    for i in range(3):
        x0 = i * 1.35
        ax.add_patch(FancyBboxPatch((x0, 0.35), 1.1, 0.5, boxstyle="round,pad=0.02", fc="white",
                                    ec=ACC if i == 1 else GREY, lw=1.6 if i == 1 else 0.9))
        ax.text(x0 + 0.55, 0.6, f"{i + 1}", ha="center", va="center", fontsize=11,
                color=ACC if i == 1 else INK)
        ax.text(x0 + 0.55, 0.2, "noise + tone" if i == 1 else "noise", ha="center", fontsize=7,
                color=ACC if i == 1 else GREY)
    ax.text(1.95, 1.08, "“Which one had the beep?”  1 / 2 / 3", ha="center", fontsize=7.8)
    ax.text(1.95, -0.2, "2-down 1-up on tone level\ngives the threshold at 70.7% correct", ha="center",
            va="top", fontsize=7.2, color=INK)
    ax.set_xlim(-0.1, 4.0)
    ax.set_ylim(-0.7, 1.3)

    # d: the design
    ax = fig.add_subplot(gs[1, 2])
    ax.set_title("11 interleaved tracks × 32 trials", fontsize=8.5, loc="left")
    label(ax, "d")
    steps = [0, 5, 10, 20, 40]
    rows = [("Co", ACC), ("Rand", GREY), ("Alone", INK)]
    for i, (name, col) in enumerate(rows):
        for j, s in enumerate(steps):
            if name == "Alone" and j:
                continue
            ax.add_patch(plt.Rectangle((j + 0.1, 2 - i + 0.12), 0.8, 0.76, fc=col, ec="none",
                                       alpha=1 if name != "Rand" else 0.75))
    ax.set_xlim(0, 5); ax.set_ylim(0, 3)
    ax.set_xticks(np.arange(5) + 0.5, [f"{s}" for s in steps])
    ax.set_yticks([2.5, 1.5, 0.5], [r[0] for r in rows])
    ax.text(1.2, 0.5, "the 1-kHz band alone", va="center", fontsize=7, color=INK)
    ax.set_xlabel("ΔT (ms)")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.text(0.98, 0.015, "CMR = threshold(random) − threshold(comodulated), at each ΔT.  "
             "Bands 20 Hz wide, 60 dB SPL; bursts 50 ms every 200 ms (5 Hz); the final burst is always synchronous.",
             ha="right", fontsize=6.3, color="0.4")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"tcmr_task.{ext}", dpi=300)


# ------------------------------------------------------------------------------------------- tinc
def tinc():
    c = json.loads((ROOT / "tinc/configs/tinc_8db.json").read_text())
    n, soa, dur, ramp = c["n_tones"], c["soa_ms"], c["tone_ms"], c["ramp_ms"]
    rove, dec, tgt = c["rove_db"], c["decoy_db"], c["target_db"]
    fA = c["f_b_hz"] * 2 ** (-c["df_semitones"] / 12)
    rng = np.random.default_rng(11)
    rv = rng.uniform(-rove, rove, n)
    decoys, target = [2, 8], 6
    LA = 60 + rv + np.isin(np.arange(n), decoys) * dec
    LB = LA + (np.arange(n) == target) * tgt

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.62, wspace=0.35,
                          left=0.11, right=0.98, top=0.89, bottom=0.13)
    fig.text(0.08, 0.975, "tinc: did one B tone get louder?  A and B rise and fall together, so comparing B with its A helps",
             fontsize=9.5, color=INK, fontweight="bold")

    for k, d in enumerate((0, 50, 100)):
        ax = fig.add_subplot(gs[0, k])
        for i in range(6):
            trap(ax, i * soa, dur, ramp, 1, 0.5, fc=INK, ec="none")
            trap(ax, i * soa + d, dur, ramp, 0, 0.5, fc=GREY, ec="none")
        ax.set_yticks([0, 1], [f"A {fA:.0f}", f"B {c['f_b_hz']:.0f}"])
        ax.set_ylim(-0.6, 1.6)
        ax.set_xlim(-20, 6 * soa + 40)
        ax.set_xlabel("Time (ms)")
        if k == 0:
            ax.set_ylabel("Frequency (Hz)")
            label(ax, "a")
        ax.set_title(f"A delay {d} ms" + ("  (synchronous)" if d == 0 else "  (alternating)" if d == 100 else ""),
                     fontsize=8.5, loc="left")

    ax = fig.add_subplot(gs[1, :2])
    label(ax, "b")
    x = np.arange(1, n + 1)
    ax.plot(x - 0.12, LA, "s-", color=GREY, ms=5, lw=1, label="A tones (420 Hz)")
    ax.plot(x + 0.12, LB, "o-", color=INK, ms=5.5, lw=1.2, mfc="white", mew=1.4, label="B tones (1000 Hz)")
    for dd in decoys:
        ax.axvspan(dd + 1 - 0.35, dd + 1 + 0.35, color=GREEN, alpha=0.15, lw=0)
        ax.text(dd + 1, 78.2, "decoy\n(pair +8)", ha="center", fontsize=6.8, color=GREEN)
    ax.axvspan(target + 1 - 0.35, target + 1 + 0.35, color=ACC, alpha=0.18, lw=0)
    ax.text(target + 1, 78.2, "change\n(B alone +8)", ha="center", fontsize=6.8, color=ACC)
    ax.set_xticks(x)
    ax.set_xlabel("Tone pair")
    ax.set_ylabel("Level (dB SPL)")
    ax.set_ylim(45, 83)
    ax.set_title(f"Levels in one 'change' trial: each pair roved together (±{rove:g} dB)", fontsize=8.5, loc="left")
    ax.legend(frameon=False, fontsize=7.2, loc="lower center", ncol=2)

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    label(ax, "c")
    ax.set_title("Task and design", fontsize=8.5, loc="left")
    lines = ["Yes/no: “Did a B tone get louder?”",
             "Change on half the trials; feedback",
             "",
             "5 delays: 0, 35, 50, 71, 100 ms",
             "one delay per block; 2 blocks × 52",
             "gives d′ per delay",
             "",
             "Grouping helps here, so d′ should",
             "FALL with delay: the mirror image",
             "of Micheyl et al. (2013), Exp. 3."]
    for i, s in enumerate(lines):
        ax.text(0, 0.95 - i * 0.1, s, fontsize=7.6, color=INK, va="top",
                fontweight="bold" if i in (0, 5) else "normal")
    fig.text(0.98, 0.012, f"Tones 100 ms with 20-ms ramps, one per stream every 200 ms (5 Hz), 12 per stream; "
             f"B 1000 Hz, A 15 semitones below; base level 60 dB SPL.", ha="right", fontsize=6.3, color="0.4")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"tinc_task.{ext}", dpi=300)


if __name__ == "__main__":
    tcmr()
    tinc()
    print(OUT)
