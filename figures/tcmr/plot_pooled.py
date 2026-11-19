"""One listener's tcmr sessions pooled: every main trial of the given sessions in one shared-slope
fit, conditions matched by name (co_sync_0, rand_frozen_10, ...). A condition present in only one
session keeps that session's trials alone.

Intervals: parametric bootstrap. Every track of every session is re-run with its own session's
rules against the pooled psychometric functions, and the whole pooled set is refitted.

Pooling assumes the listener's thresholds did not move between sessions. CMR is a difference
within a session's interleaved tracks, so an overall level shift between days mostly cancels in it;
the per-session CMR is drawn beside the pooled one so a real change would be visible.

Usage: python figures/tcmr/plot_pooled.py data/P00/tcmr_session_05 data/DEMO/tcmr_session_02 \
           --out figures/tcmr/pooled_P00 [--reps 1000]"""
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


def load(folder):
    meta = json.loads((folder / "session.json").read_text())
    rows = [r["row"] for r in read_jsonl(folder / "trials.jsonl")]
    data = {}
    for r in rows:
        if r["phase"] == "main" and r["status"] == "ok":
            L, C = data.setdefault(r["condition"], ([], []))
            L.append(r["level_db"]); C.append(bool(r["correct"]))
    return meta, Config.from_dict(meta["config"]), data


def merge(sets):
    out = {}
    for d in sets:
        for k, (L, C) in d.items():
            a, b = out.setdefault(k, ([], []))
            a.extend(L); b.extend(C)
    return out


def one_rep(args):
    cfgs, th, sigma, rep = args
    rng = np.random.default_rng([20261115, 7, rep])
    sets = []
    for cfg in cfgs:
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
        sets.append(data)
    return fit(merge(sets), lapse=cfgs[0].lapse)["thresholds_db"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True, help="output stem, e.g. figures/tcmr/pooled_P00")
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()
    S = [load(f) for f in a.folders]
    cfgs = [s[1] for s in S]
    pooled = merge([s[2] for s in S])
    f0 = fit(pooled, lapse=cfgs[0].lapse)
    th, sigma = f0["thresholds_db"], f0["sigma_db"]
    per = [fit(s[2], lapse=s[1].lapse)["thresholds_db"] for s in S]
    names = sorted(th)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    cache = a.out.with_name(a.out.name + "_bootstrap.npz")
    key = json.dumps([[str(f) for f in a.folders], th, sigma, a.reps], sort_keys=True)
    if cache.exists() and str(np.load(cache)["key"]) == key:
        B = np.load(cache)["draws"]
    else:
        with Pool() as pool:
            out = pool.map(one_rep, [(cfgs, th, sigma, i) for i in range(a.reps)], chunksize=10)
        B = np.array([[o.get(n, np.nan) for n in names] for o in out])
        np.savez(cache, draws=B, names=np.array(names), key=key)
    draw = {n: B[:, i] for i, n in enumerate(names)}

    nm = lambda env, dt: f"{env}_sync_0" if dt == 0 else f"{env}_frozen_{dt:g}"
    steps = [0.0] + sorted({float(k.split("_")[-1]) for k in th if k.startswith("co_frozen_")})
    x = np.arange(len(steps), dtype=float)
    ctrl = sorted({float(k.split("_")[-1]) for k in th if k.startswith("co_reordered_")})
    xc = x[-1] + 1.4
    q = lambda v: np.nanpercentile(v, [2.5, 12.5, 87.5, 97.5])
    cmr = lambda T, s: (T[nm("rand", s)] - T[nm("co", s)]) if nm("co", s) in T and nm("rand", s) in T else np.nan

    plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 9, "axes.linewidth": 0.8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    INK, ACC, GREY = "#1f2933", "#2f5d8a", "#8a96a3"
    SESS = ["#7fa7c9", "#5e8c6a", "#a08cb8", "#c9a227"]
    fig, ax = plt.subplots(figsize=(4.2, 3.3))
    for j, (s, T) in enumerate(zip(S, per)):
        lab = f"{s[0]['code']} {Path(a.folders[j]).name} ({s[1].name})"
        ys = [cmr(T, st) for st in steps]
        ax.plot(x + (j - (len(S) - 1) / 2) * 0.12, ys, "-", color=SESS[j], lw=0.9, alpha=0.9)
        ax.plot(x + (j - (len(S) - 1) / 2) * 0.12, ys, "o", ms=3.8, color=SESS[j], label=lab)
        for c in ctrl:
            v = T.get(f"rand_reordered_{c:g}", np.nan) - T.get(f"co_reordered_{c:g}", np.nan)
            ax.plot([xc + (j - (len(S) - 1) / 2) * 0.12], [v], "s", ms=3.6, color=SESS[j])
    cm = [cmr(th, s) for s in steps]
    Q = np.array([q(draw[nm("rand", s)] - draw[nm("co", s)]) for s in steps])
    ax.vlines(x, Q[:, 0], Q[:, 3], color=INK, lw=1.0)
    ax.vlines(x, Q[:, 1], Q[:, 2], color=INK, lw=4.5)
    ax.plot(x, cm, "-", color=INK, lw=1.4, zorder=3)
    ax.plot(x, cm, "o", ms=7, mfc="white", mec=INK, mew=1.7, zorder=4, label="Pooled (95% and 75%)")
    ticks, labels = list(x), [f"{s:g}" for s in steps]
    if ctrl:
        k = f"{ctrl[0]:g}"
        dc = draw[f"rand_reordered_{k}"] - draw[f"co_reordered_{k}"]
        qc = q(dc)
        ax.vlines([xc], qc[0], qc[3], color=GREY, lw=1.0); ax.vlines([xc], qc[1], qc[2], color=GREY, lw=4.5)
        ax.plot([xc], [th[f"rand_reordered_{k}"] - th[f"co_reordered_{k}"]], "s", ms=6.5, mfc="white",
                mec=GREY, mew=1.6, zorder=4)
        ticks.append(xc); labels.append(f"{k}\nre-ordered")
        ax.axvline((x[-1] + xc) / 2, color="0.8", lw=0.7, ls=":")
    ax.axhline(0, color="0.55", lw=0.7)
    ax.set_xticks(ticks, labels)
    ax.set_xlabel("Onset step between bands, ΔT (ms)")
    ax.set_ylabel("CMR (dB): random − comodulated")
    ax.set_title(f"{S[0][0]['code']}: sessions pooled", loc="left", fontsize=9.5, color=INK)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right", handlelength=1.2)
    fig.text(0.01, 0.005, f"Pooled shared-slope fit over {len(S)} sessions; bars from {a.reps} re-run "
             f"staircases.\nSmall points: each session alone. Levels uncalibrated.", fontsize=5.6, color="0.45")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(a.out.with_name(a.out.name + f"_cmr.{ext}"), dpi=300)

    print(f"pooled slope {sigma:.1f} dB")
    print(f"{'step':>6} {'pooled CMR':>11} {'95% interval':>16}  " + "  ".join(f"{s[0]['code']}/{Path(f).name}" for s, f in zip(S, a.folders)))
    for i, s in enumerate(steps):
        print(f"{s:>6g} {cm[i]:+11.1f} {Q[i,0]:+7.1f} to {Q[i,3]:+.1f}  " + "  ".join(f"{cmr(T, s):+12.1f}" for T in per))
    d = (draw[nm("rand", 0)] - draw[nm("co", 0)]) - (draw[nm("rand", steps[-1])] - draw[nm("co", steps[-1])])
    print(f"drop 0 -> {steps[-1]:g} ms: {cm[0] - cm[-1]:+.1f}  95% {q(d)[0]:+.1f} to {q(d)[3]:+.1f}")
    for k in names:
        print(f"  {k:<20} {th[k]:6.1f}  95% {q(draw[k])[0]:.1f} to {q(draw[k])[3]:.1f}   " + "  ".join(f"{T.get(k, np.nan):6.1f}" for T in per))
