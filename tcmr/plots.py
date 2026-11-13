"""One-page figure for a tcmr session: tracks, thresholds, psychometric data, CMR, volume, ratings."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from tsync.logbook import read_jsonl  # noqa: E402

from .config import Config, Condition  # noqa: E402
from .procedure import GUESS, Track  # noqa: E402

def _style(cfg):
    """Colour and label for every condition: blues for comodulated ascending (darker = larger
    step), orange for zigzag, greys for random envelopes."""
    col, lab, order = {}, {}, []
    conds = cfg.condition_list()
    asc = sorted({c.dt_ms for c in conds if c.envelope == "co" and c.arrangement == "ascending"})
    cmap = matplotlib.colormaps["viridis"]
    for c in sorted(conds, key=lambda c: (c.envelope == "ref", c.envelope != "co", c.arrangement != "sync",
                                          c.arrangement, c.dt_ms)):
        short = {"sync": "sync", "ascending": "asc", "zigzag": "zig"}[c.arrangement]
        lab[c.name] = f"{c.envelope} {short}" + (f" {c.dt_ms:g}" if c.dt_ms else "")
        if c.envelope == "ref":
            col[c.name] = "#7a5bbf"
            lab[c.name] = "band alone"
        elif c.envelope == "rand":
            col[c.name] = "#9aa4b2" if c.arrangement == "sync" else "#c9b28a"
        elif c.arrangement == "sync":
            col[c.name] = "#1b4f9c"
        elif c.arrangement == "zigzag":
            col[c.name] = "#d0682e"
        else:
            i = asc.index(c.dt_ms)
            col[c.name] = matplotlib.colors.to_hex(cmap(0.25 + 0.6 * i / max(1, len(asc) - 1)))
        order.append(c.name)
    return col, lab, order


def session_figure(folder: Path, summ: dict, out: Path = None) -> Path:
    folder = Path(folder)
    cfg = Config.from_dict(json.loads((folder / "session.json").read_text())["config"])
    rows = [r["row"] for r in read_jsonl(folder / "trials.jsonl")]
    main = [r for r in rows if r["phase"] == "main" and r["status"] == "ok"]
    th = summ["thresholds_db"]
    COL, LAB, ORDER = _style(cfg)
    conds = [c for c in ORDER if c in th]
    tracks = sorted({int(r["track"]) for r in main})
    nrow = (min(len(tracks), 8) + 3) // 4
    fig = plt.figure(figsize=(15, 3.4 * (nrow + 1)))
    gs = fig.add_gridspec(nrow + 1, 4, hspace=0.55, wspace=0.35)

    # A: every track, level by trial
    for i, tid in enumerate(tracks):
        ax = fig.add_subplot(gs[i // 4, i % 4]) if i < 8 else None
        if ax is None:
            break
        rs = [r for r in main if int(r["track"]) == tid]
        c = rs[0]["condition"]
        L = np.array([r["level_db"] for r in rs])
        C = np.array([bool(r["correct"]) for r in rs])
        x = np.arange(len(L))
        ax.plot(x, L, "-", color=COL.get(c, "k"), lw=1)
        ax.scatter(x[C], L[C], s=12, color=COL.get(c, "k"), zorder=3)
        ax.scatter(x[~C], L[~C], s=16, facecolor="white", edgecolor=COL.get(c, "k"), zorder=3)
        rev = [j for j, r in enumerate(rs) if r["reversal"]]
        ax.scatter(rev, L[rev], s=45, facecolor="none", edgecolor="0.3", lw=0.8)
        if c in th:
            ax.axhline(th[c], color="0.35", ls="--", lw=0.9)
        ax.set_title(f"track {tid}: {LAB.get(c, c)}", fontsize=9)
        ax.set_ylim(min(45, L.min() - 3), max(80, L.max() + 3))
        ax.tick_params(labelsize=7)
        if i % 4 == 0:
            ax.set_ylabel("tone level (dB SPL)", fontsize=8)
        ax.set_xlabel("trial in track", fontsize=7)

    # B: thresholds
    ax = fig.add_subplot(gs[nrow, 0])
    for k, c in enumerate(conds):
        ax.plot(k, th[c], "o", ms=9, color=COL[c])
        revs = [v for v in summ["reversal_thresholds_db"].get(c, []) if v]
        ax.plot([k + 0.18] * len(revs), revs, "x", color="0.4", ms=6)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([LAB[c] for c in conds], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("threshold (dB SPL, 70.7%)", fontsize=8)
    ax.set_title("Thresholds (o fit, x reversal means)", fontsize=9)
    ax.grid(alpha=0.25)

    # C: psychometric data and the shared-slope fit
    ax = fig.add_subplot(gs[nrow, 1])
    s = summ["slope_sigma_db"] or 3.0
    z = stats.norm.ppf((0.707 - GUESS) / (1 - GUESS - cfg.lapse))
    xs = np.linspace(45, 80, 200)
    for c in conds:
        rs = [r for r in main if r["condition"] == c]
        L = np.array([r["level_db"] for r in rs])
        C = np.array([bool(r["correct"]) for r in rs])
        for lv in np.unique(np.round(L / 4) * 4):
            m = np.round(L / 4) * 4 == lv
            if m.sum() >= 3:
                ax.scatter(lv, C[m].mean(), s=6 + 5 * m.sum(), color=COL[c], alpha=0.55)
        mu = th[c] - s * z
        ax.plot(xs, GUESS + (1 - GUESS - cfg.lapse) * stats.norm.cdf((xs - mu) / s), color=COL[c],
                lw=1.5, label=LAB[c])
    ax.axhline(0.707, color="0.6", ls=":", lw=0.8)
    ax.set_ylim(0.2, 1.02)
    ax.set_xlabel("tone level (dB SPL, 4-dB bins)", fontsize=8)
    ax.set_ylabel("proportion correct", fontsize=8)
    ax.set_title("Psychometric data and fit", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")

    # D: masking release against the step (ascending), zigzag and sync as markers
    ax = fig.add_subplot(gs[nrow, 2])
    cmr = summ["cmr_db"]
    pts = sorted((float(k.split("_")[1]), v) for k, v in cmr.items()
                 if v is not None and k.split("_")[0] in ("ascending", "sync"))
    if pts:
        ax.plot([x for x, _ in pts], [y for _, y in pts], "o-", color="#2f6f9f", lw=2,
                label="sync / ascending")
    zz = [(float(k.split("_")[1]), v) for k, v in cmr.items() if k.startswith("zigzag") and v is not None]
    if zz:
        ax.plot([x for x, _ in zz], [y for _, y in zz], "s", color="#d0682e", ms=8, label="zigzag")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhspan(4, 7, color="0.85", alpha=0.6, zorder=0)
    ax.set_xlabel("step per row dT (ms); largest shift = 2 dT", fontsize=8)
    ax.set_ylabel("CMR (dB) = random - comodulated", fontsize=8)
    ax.set_title("Masking release vs shear (grey: typical at sync)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    # E: volume over the session, and ratings
    ax = fig.add_subplot(gs[nrow, 3])
    ok = [r for r in rows if r["status"] == "ok" and r["volume"] is not None]
    ax.plot([r["row"] for r in ok], [r["volume"] for r in ok], ".", ms=3, color="0.4")
    ax.set_xlabel("trial (in order, all phases)", fontsize=8)
    ax.set_ylabel("system volume", fontsize=8)
    rat = summ.get("ratings_mean", {})
    if rat:
        txt = "ratings (1 one sound, 5 separate)\n" + "\n".join(f"{k}: {v:.1f}" for k, v in rat.items())
        ax.text(0.98, 0.04, txt, transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
                bbox=dict(facecolor="white", edgecolor="0.8"))
    ax.set_title("Volume during the session", fontsize=9)
    fig.suptitle(f"{summ['code']}  tcmr session  (status {summ['status']}; "
                 f"slope {summ['slope_sigma_db']:.1f} dB)", fontsize=12)
    out = Path(out) if out else folder / "summary.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
