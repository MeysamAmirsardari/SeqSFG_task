"""Standalone figures for one tcmr session, in the style of figures/tinc/plot_listener.py:
CMR by onset step, and the thresholds behind it.

Points are the session's own shared-slope fit (as `tcmr analyze`). Intervals come from a
parametric bootstrap: every track's staircase is re-run with the session's own rules against the
fitted psychometric functions and the whole set is refitted, so they respect the adaptive
sampling. Draws are cached in the session folder (bootstrap_<code>.npz).

Usage: python figures/tcmr/plot_listener.py data/P00/tcmr_session_05 [--reps 2000]"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tcmr.config import Config
from tcmr.procedure import GUESS, Track, _p, fit
from tsync.logbook import read_jsonl


def one_rep(args):
    cfg, th, sigma, rep = args
    rng = np.random.default_rng([20261115, rep])
    z = stats.norm.ppf((0.707 - GUESS) / (1 - GUESS - cfg.lapse))
    data = {}
    for c in cfg.condition_list():
        mu = th[c.name] - sigma * z
        for _ in range(c.tracks):
            t = Track(cfg, c, 0)
            L, C = data.setdefault(c.name, ([], []))
            while not t.finished:
                ok = rng.random() < _p(t.level, mu, sigma, cfg.lapse)
                L.append(t.level); C.append(bool(ok))
                t.update(ok)
    return fit(data, lapse=cfg.lapse)["thresholds_db"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args()
    folder = a.folder
    meta = json.loads((folder / "session.json").read_text())
    cfg, code = Config.from_dict(meta["config"]), meta["code"]
    cal = bool(meta.get("levels_calibrated"))
    rows = [r["row"] for r in read_jsonl(folder / "trials.jsonl")]
    data = {}
    for r in rows:
        if r["phase"] == "main" and r["status"] == "ok":
            L, C = data.setdefault(r["condition"], ([], []))
            L.append(r["level_db"]); C.append(bool(r["correct"]))
    f0 = fit(data, lapse=cfg.lapse)
    th, sigma = f0["thresholds_db"], f0["sigma_db"]
    names = [c.name for c in cfg.condition_list()]

    cache = folder / f"bootstrap_{code}.npz"
    key = json.dumps([th, sigma, a.reps], sort_keys=True)
    if cache.exists() and str(np.load(cache)["key"]) == key:
        B = np.load(cache)["draws"]
    else:
        with Pool() as pool:
            out = pool.map(one_rep, [(cfg, th, sigma, i) for i in range(a.reps)], chunksize=20)
        B = np.array([[o.get(n, np.nan) for n in names] for o in out])
        np.savez(cache, draws=B, names=np.array(names), key=key)
    draw = {n: B[:, i] for i, n in enumerate(names)}

    steps = sorted({c.dt_ms for c in cfg.condition_list() if c.arrangement == "frozen"})
    steps = [0.0] + steps
    nm = lambda env, dt: f"{env}_sync_0" if dt == 0 else f"{env}_frozen_{dt:g}"
    ctrl = sorted({c.dt_ms for c in cfg.condition_list() if c.arrangement == "reordered"})
    x = np.arange(len(steps), dtype=float)
    xc = len(steps) - 1 + 1.4                               # the re-ordered control, set apart
    ticks = list(x) + ([xc] if ctrl else [])
    labels = [f"{s:g}" for s in steps] + ([f"{ctrl[0]:g}\nre-ordered"] if ctrl else [])

    plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 9, "axes.linewidth": 0.8,
                         "xtick.major.width": 0.8, "ytick.major.width": 0.8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    INK, ACC, GREY = "#1f2933", "#2f5d8a", "#8a96a3"
    q = lambda v: np.nanpercentile(v, [2.5, 12.5, 87.5, 97.5])
    note = (f"Bars: 75% and 95% intervals ({a.reps} re-run staircases).\nOne "
            f"{cfg.trials_per_track}-trial track per condition; levels "
            f"{'calibrated' if cal else 'uncalibrated (relative dB)'}.")

    def bars(ax, xs, point, dr, col, marker, ms=6.5, line=True, label=None):
        Q = np.array([q(v) for v in dr])
        ax.vlines(xs, Q[:, 0], Q[:, 3], color=col, lw=1.0)
        ax.vlines(xs, Q[:, 1], Q[:, 2], color=col, lw=4.5)
        if line:
            ax.plot(xs, point, "-", color=col, lw=1.2, zorder=3)
        ax.plot(xs, point, marker, ms=ms, mfc="white", mec=col, mew=1.6, zorder=4, label=label)
        return Q

    def frame(ax):
        ax.set_xticks(ticks, labels)
        ax.set_xlim(-0.6, (xc if ctrl else x[-1]) + 0.6)
        ax.set_xlabel("Onset step between bands, ΔT (ms)")
        if ctrl:
            ax.axvline((x[-1] + xc) / 2, color="0.8", lw=0.7, ls=":")

    # 1. CMR = random - comodulated, by step
    fig, ax = plt.subplots(figsize=(3.8, 3.1))
    cm = [th[nm("rand", s)] - th[nm("co", s)] for s in steps]
    cmd = [draw[nm("rand", s)] - draw[nm("co", s)] for s in steps]
    Q = bars(ax, x, cm, cmd, ACC, "o", label="Frozen pattern")
    lo, hi = Q[:, 0].min(), Q[:, 3].max()
    if ctrl:
        c0 = th[f"rand_reordered_{ctrl[0]:g}"] - th[f"co_reordered_{ctrl[0]:g}"]
        cd = draw[f"rand_reordered_{ctrl[0]:g}"] - draw[f"co_reordered_{ctrl[0]:g}"]
        Qc = bars(ax, [xc], [c0], [cd], GREY, "s", ms=6, line=False, label="Re-ordered (control)")
        lo, hi = min(lo, Qc[0, 0]), max(hi, Qc[0, 3])
    ax.axhline(0, color="0.55", lw=0.7)
    frame(ax)
    ax.set_ylim(min(0, lo) - 1, hi + 1)
    ax.set_ylabel("CMR (dB): random − comodulated")
    ax.set_title(f"{code}: masking release by onset step", loc="left", fontsize=9.5, color=INK)
    if ctrl:
        ax.legend(frameon=False, fontsize=7.5, loc="upper right", handlelength=1.2)
    fig.text(0.01, 0.005, note, fontsize=5.6, color="0.45")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"cmr_{code}.{ext}", dpi=300)

    # 2. the thresholds behind it, with the tone's band alone as the reference
    fig, ax = plt.subplots(figsize=(3.8, 3.1))
    dx = 0.13
    if "ref_sync_0" in th:
        rq = q(draw["ref_sync_0"])
        ax.axhspan(rq[1], rq[2], color="0.92", lw=0, zorder=0)
        ax.axhline(th["ref_sync_0"], color=INK, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.text(-0.55, th["ref_sync_0"] + 0.25, "tone's band alone", fontsize=7.5, color=INK,
                va="bottom")
    for env, col, mk, lab, off in (("rand", GREY, "s", "Random envelopes", dx),
                                   ("co", ACC, "o", "Comodulated", -dx)):
        bars(ax, x + off, [th[nm(env, s)] for s in steps], [draw[nm(env, s)] for s in steps],
             col, mk, ms=6 if mk == "s" else 6.5, label=lab)
        if ctrl:
            k = f"{env}_reordered_{ctrl[0]:g}"
            bars(ax, [xc + off], [th[k]], [draw[k]], col, mk, ms=6 if mk == "s" else 6.5, line=False)
    frame(ax)
    ax.set_ylabel(f"Tone threshold ({'dB SPL' if cal else 'dB'}, 70.7% correct)")
    ax.set_title(f"{code}: thresholds", loc="left", fontsize=9.5, color=INK)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], frameon=False, fontsize=7.5, loc="upper right", handlelength=1.2)
    fig.text(0.01, 0.005, note, fontsize=5.6, color="0.45")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"thresholds_{code}.{ext}", dpi=300)

    for s in steps:
        v = draw[nm("rand", s)] - draw[nm("co", s)]
        print(f"CMR {s:>4g} ms {th[nm('rand', s)] - th[nm('co', s)]:+5.1f}  95% {q(v)[0]:+.1f} to {q(v)[3]:+.1f}")
    if ctrl:
        print(f"CMR {ctrl[0]:g} re-ordered {c0:+5.1f}  95% {q(cd)[0]:+.1f} to {q(cd)[3]:+.1f}")
    d = (draw[nm("rand", 0)] - draw[nm("co", 0)]) - (draw[nm("rand", steps[-1])] - draw[nm("co", steps[-1])])
    print(f"CMR drop 0 -> {steps[-1]:g} ms: {cm[0] - cm[-1]:+.1f}  95% {q(d)[0]:+.1f} to {q(d)[3]:+.1f}")
    for k in names:
        print(f"{k:<20} {th[k]:6.1f}  95% {q(draw[k])[0]:.1f} to {q(draw[k])[3]:.1f}")
