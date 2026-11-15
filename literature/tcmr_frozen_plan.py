"""tcmr round 1, frozen-pattern design (decisions of 2026-11-14): timing, listeners, and the figure.

    .venv/bin/python literature/tcmr_frozen_plan.py        # prints the tables, writes the figure

Decisions taken on 2026-11-14 (see tcmr_redesign_2026-11-14.md, section 13):
  - the pattern is a staircase of offsets with step s, and the bands are shuffled over it; the
    shuffled order is frozen across the 5 repetitions of an interval. The noise is never frozen:
    every burst is new noise at its natural level;
  - a new order for each interval (the user, later the same day): with one order per trial a tone
    in interval 3 would follow 15 repetitions of the pattern and one in interval 1 only 5, so how
    much could be learned would depend on where the tone is. The three intervals of a trial get
    three different orders, drawn from the 24 without replacement;
  - 50-ms bursts (so at the largest step, 40 ms, neighbouring bands overlap by 10 ms);
  - the re-ordered control only if there is time.
Kept from the proposal: 5 Hz, 5 precursors, a synchronous final burst carrying the tone, the
tone band at offset 0, 3I-3AFC 2-down 1-up 8/4/2 dB.

The precision numbers come from tcmr's own staircase and threshold fit (tcmr.procedure), with
psychometric slopes taken from the two P00 pilot sessions; the timing comes from the P00 logs.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------------- design
BANDS = (250, 500, 1000, 2000, 4000)
TONE = 2                                  # index of the tone band in the example (1 kHz)
BURST, RAMP, NOI, N_PRE = 50.0, 10.0, 200.0, 5
STEPS = (0.0, 10.0, 20.0, 40.0)
ISI = 400.0
LEAD = 2 * max(STEPS)                     # the window opens with the earliest band at 40 ms steps
INTERVAL = LEAD + N_PRE * NOI + BURST     # 1130 ms in every condition: the tone band's bursts
TRIAL_SOUND = 3 * INTERVAL + 2 * ISI      # always sit at 80, 280, ... and 1080 ms in the window
TRIALS_PER_TRACK = 32

# seconds per trial beyond the sound: box lights 0.4 + response + feedback 0.5 + blank 0.6.
# P00 (practised, 0.3-s responses) measured 1.9 s in both pilot sessions; naive listeners are
# planned at 0.8-s responses, as tcmr's own session simulator assumes.
OVERHEAD_NAIVE, OVERHEAD_P00 = 2.3, 1.9

PLANS = {
    "A": {"label": "A  core: steps 0, 10, 20, 40 ms + band alone",
          "tracks": [("co", s) for s in STEPS] + [("rand", s) for s in STEPS] + [("alone", 0.0)]},
    "B": {"label": "B  control in place of the 10-ms step",
          "tracks": [("co", s) for s in (0.0, 20.0, 40.0)] + [("rand", s) for s in (0.0, 20.0, 40.0)]
          + [("alone", 0.0), ("co-reordered", 40.0), ("rand-reordered", 40.0)]},
    "C": {"label": "C  core + control",
          "tracks": [("co", s) for s in STEPS] + [("rand", s) for s in STEPS] + [("alone", 0.0),
                                                                             ("co-reordered", 40.0),
                                                                             ("rand-reordered", 40.0)]},
}

# minutes outside the main block (the app's screens as the pilots ran them, plus the visit)
OTHER = [("setup", "consent, headphones, level check, volume locked", 5.0),
         ("questions", "questionnaire", 1.0),
         ("instructions", "instructions and two labelled examples", 1.5),
         ("practice", "8 practice trials, easy levels, step 0", None),     # computed
         ("breaks", "3 breaks of about 1 min (pilot: 0.3-2.3 min)", 3.0),
         ("ratings", "6 sequences rated 1-5: one sound or separate layers", 1.0),
         ("debrief", "debrief", 1.0)]


def trial_s(overhead=OVERHEAD_NAIVE) -> float:
    return TRIAL_SOUND / 1000 + overhead


def session_minutes(plan: str, overhead=OVERHEAD_NAIVE) -> dict:
    n = len(PLANS[plan]["tracks"]) * TRIALS_PER_TRACK
    main = n * trial_s(overhead) / 60
    parts = {}
    for key, _, m in OTHER:
        parts[key] = m if m is not None else 8 * trial_s(overhead) / 60
    parts["main"] = main
    return {"tracks": len(PLANS[plan]["tracks"]), "trials": n, "parts": parts,
            "total": sum(parts.values())}


# ---------------------------------------------------------------------------------- precision
def contrast_sd(sigma_psy, lapse, n_sessions=300, seed=7) -> dict:
    """SD of one track's threshold and of a 4-track contrast, CMR(0) - CMR(40), simulated with
    tcmr's staircase and its shared-slope maximum-likelihood fit, one track per condition."""
    from tcmr.config import Condition, Config
    from tcmr.procedure import SimListener, Track, fit
    cfg = Config(trials_per_track=TRIALS_PER_TRACK)
    true = {"co_sync_0": 56.0, "rand_sync_0": 60.0, "co_sync_10": 56.5, "rand_sync_10": 60.0,
            "co_sync_20": 57.5, "rand_sync_20": 60.0, "co_sync_40": 59.0, "rand_sync_40": 60.0,
            "ref_sync_0": 60.0}
    rng = np.random.default_rng(seed)
    errs, cons = [], []
    for _ in range(n_sessions):
        lis = SimListener(true, sigma_db=sigma_psy, lapse=lapse)
        trials = {}
        for name in true:
            e, a, dt = name.split("_")
            t = Track(cfg, Condition(e, a, float(dt)), 0)
            L, Cc = [], []
            while not t.finished:
                lvl = t.level
                ok = rng.random() < lis.p(name, lvl)
                t.update(ok)
                L.append(lvl)
                Cc.append(ok)
            trials[name] = (L, Cc)
        th = fit(trials, lapse=cfg.lapse)["thresholds_db"]
        errs += [th[k] - true[k] for k in true]
        est = (th["rand_sync_0"] - th["co_sync_0"]) - (th["rand_sync_40"] - th["co_sync_40"])
        cons.append(est - 3.0)
    return {"track_sd": float(np.std(errs)), "contrast_sd": float(np.std(cons))}


def power(delta, sd, n, alpha=0.05) -> float:
    """Paired (one-sample) t-test on the per-listener contrast."""
    df, nc = n - 1, delta / sd * math.sqrt(n)
    tc = stats.t.ppf(1 - alpha / 2, df)
    p = stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)
    if not math.isfinite(p):                      # scipy's noncentral t fails far in the tail,
        p = stats.norm.sf(tc - nc) + stats.norm.cdf(-tc - nc)   # where power is ~1 anyway
    return float(min(p, 1.0))


def n_needed(delta, sd, target=0.8) -> int:
    return next(n for n in range(3, 1000) if power(delta, sd, n) >= target)


def mde(sd, n, target=0.8) -> float:
    hi = 12 * sd / math.sqrt(n)
    return float(optimize.brentq(lambda d: power(d, sd, n) - target, 0.01, hi))


# ---------------------------------------------------------------------------------- figure
COL = {250: "#24477a", 500: "#2f8f83", 1000: "#151515", 2000: "#6a4fa3", 4000: "#4a95d1"}
TONE_COL = "#d1352b"


def flank_order(rng) -> dict:
    """Band index -> slot (-2..2): the tone band keeps 0, the four flankers take -2, -1, +1, +2
    in a random order."""
    flank = [b for b in range(5) if b != TONE]
    slots = rng.permutation([-2, -1, 1, 2])
    return {TONE: 0, **{b: int(s) for b, s in zip(flank, slots)}}


def all_orders() -> list:
    """The 24 orders: the tone band at 0, the four flankers over -2, -1, +1, +2."""
    from itertools import permutations
    flank = [b for b in range(5) if b != TONE]
    return [{TONE: 0, **dict(zip(flank, perm))} for perm in permutations((-2, -1, 1, 2))]


def trial_orders(rng, k=3) -> list:
    """Three different orders for the three intervals of a trial."""
    pool = all_orders()
    return [pool[i] for i in rng.permutation(len(pool))[:k]]


def reordered(rng, n=N_PRE) -> list:
    """A new order on every repetition; no band may go from +2 to -2 between repetitions (at
    40-ms steps its two bursts would be 40 ms apart and overlap)."""
    out = [flank_order(rng)]
    while len(out) < n:
        o = flank_order(rng)
        if all(not (out[-1][b] == 2 and o[b] == -2) for b in range(5)):
            out.append(o)
    return out


def burst_levels(rng, comodulated: bool, n=N_PRE + 1) -> np.ndarray:
    """Natural burst level (rms amplitude re long-term rms) of 50-ms bursts of 20-Hz noise,
    shape (n bursts, 5 bands): shared across bands when comodulated."""
    fs, K = 1000.0, np.arange(-10, 11)
    t = (np.arange(int(BURST)) + 0.5) / fs
    w = np.ones_like(t)
    on, off = t < RAMP / 1000, t > (BURST - RAMP) / 1000
    w[on] = np.sin(0.5 * np.pi * t[on] / (RAMP / 1000)) ** 2
    w[off] = np.sin(0.5 * np.pi * (BURST / 1000 - t[off]) / (RAMP / 1000)) ** 2
    E = np.exp(2j * np.pi * np.outer(K, t))

    def one():
        z = (rng.normal(size=K.size) + 1j * rng.normal(size=K.size)) @ E
        return math.sqrt(np.sum(np.abs(z) ** 2 * w ** 2) / np.sum(w ** 2) / (2 * K.size))
    L = np.empty((n, 5))
    for j in range(n):
        if comodulated:
            L[j] = one()
        else:
            L[j] = [one() for _ in range(5)]
    return L


def draw_burst(ax, t0, y, h, color, alpha=1.0, z=2):
    t = np.linspace(0, BURST, 50)
    env = np.ones_like(t)
    on, off = t < RAMP, t > BURST - RAMP
    env[on] = np.sin(0.5 * np.pi * t[on] / RAMP) ** 2
    env[off] = np.sin(0.5 * np.pi * (BURST - t[off]) / RAMP) ** 2
    ax.fill(np.r_[t0 + t, t0 + t[::-1]], np.r_[y + h * env / 2, y - h * env[::-1] / 2],
            color=color, lw=0, alpha=alpha, zorder=z)


def draw_interval(ax, x0, step, orders, levels, tone=False, bands=range(5), scale=0.55):
    """One interval: precursors at each band's slot x step, then the synchronous final burst."""
    for j in range(N_PRE + 1):
        o = orders[min(j, len(orders) - 1)]
        for b in bands:
            on = x0 + LEAD + j * NOI + (o[b] * step if j < N_PRE else 0.0)
            h = float(np.clip(scale * levels[j, b], 0.18, 0.92))
            draw_burst(ax, on, b, h, COL[BANDS[b]])
    if tone:
        on = x0 + LEAD + N_PRE * NOI
        draw_burst(ax, on, TONE, 0.95, TONE_COL, alpha=0.85, z=3)


def band_axis(ax, labels=True, tone_label=True):
    ax.set_ylim(-0.7, 4.7)
    ax.set_yticks(range(5))
    if labels:
        ax.set_yticklabels([f"{f / 1000:g} kHz" + ("  tone band" if (b == TONE and tone_label) else "")
                            for b, f in enumerate(BANDS)], fontsize=8.5)
        for lab, b in zip(ax.get_yticklabels(), range(5)):
            lab.set_color(COL[BANDS[b]])
            if b == TONE:
                lab.set_fontweight("bold")
    else:
        ax.set_yticklabels([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)


def cycle_axis(ax, step, order, title, labels=False):
    """One 200-ms repetition: each band's burst at slot x step."""
    for b in range(5):
        draw_burst(ax, order[b] * step, b, 0.62, COL[BANDS[b]])
    ax.axvline(0, color="#999", lw=0.6, ls=":", zorder=1)
    ax.set_xlim(-95, 145)
    ax.set_xticks([-80, 0, 80])
    ax.tick_params(axis="x", labelsize=7.5, length=2.5, pad=1.5)
    band_axis(ax, labels=labels, tone_label=False)
    ax.set_title(title, fontsize=9, pad=3, linespacing=1.15)


def order_text(order) -> str:
    by_time = sorted(range(5), key=lambda b: order[b])
    return "  ".join(f"[{BANDS[b] / 1000:g}]" if b == TONE else f"{BANDS[b] / 1000:g}" for b in by_time)


def figure(path: Path, plans: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    rng = np.random.default_rng(14)
    fig = plt.figure(figsize=(17, 15))
    gs = GridSpec(4, 6, figure=fig, height_ratios=[3.0, 1.9, 1.9, 1.55], hspace=0.95, wspace=0.42,
                  left=0.085, right=0.985, top=0.925, bottom=0.045)

    def header(cell, head, sub=None, dy=0.0):
        bb = cell.get_position(fig)
        two = sub is not None and "\n" in sub
        fig.text(bb.x0 - 0.004, bb.y1 + (0.057 if two else 0.047) + dy, head, fontsize=11.5,
                 fontweight="bold", va="bottom")
        if sub:
            fig.text(bb.x0 - 0.004, bb.y1 + 0.030 + dy, sub, fontsize=8.6, color="#333", va="bottom",
                     linespacing=1.25)

    orders = trial_orders(rng)
    while any(o[0] == -2 and o[4] == 2 for o in orders):   # keep the example visibly unordered
        orders = trial_orders(rng)
    order = orders[0]
    step = 40.0

    # ---- A: one trial ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    target = 2
    for iv in range(3):
        x0 = iv * (INTERVAL + ISI)
        ax.axvspan(x0, x0 + INTERVAL, color="#f3f5f8", zorder=0)
        draw_interval(ax, x0, step, [orders[iv]], burst_levels(rng, True), tone=(iv + 1 == target))
        ax.text(x0 + INTERVAL / 2, 5.72, f"interval {iv + 1}" + ("  (the tone)" if iv + 1 == target else "")
                + f":  order {order_text(orders[iv])} kHz", ha="center", fontsize=10,
                color=TONE_COL if iv + 1 == target else "#222",
                fontweight="bold" if iv + 1 == target else "normal")
    band_axis(ax)
    ax.set_ylim(-0.75, 6.15)
    ax.set_xlim(-30, TRIAL_SOUND + 30)
    ax.set_xlabel("time in the trial (ms)", fontsize=9)
    # interval 1: what repeats, the period, the final burst
    x0, fin = 0.0, LEAD + N_PRE * NOI
    ax.annotate("", xy=(LEAD + 1 * NOI, 4.62), xytext=(LEAD + 2 * NOI, 4.62),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=0.9))
    ax.text(LEAD + 1.5 * NOI, 4.72, "200 ms", ha="center", va="bottom", fontsize=8.5)
    ax.plot([0, LEAD + 4 * NOI + 130], [5.02, 5.02], color="#666", lw=0.8)
    ax.text((LEAD + 4 * NOI + 130) / 2, 5.07, "5 precursors: the interval's order in every repetition",
            ha="center", va="bottom", fontsize=8.5, color="#333")
    ax.annotate("final burst: all bands\ntogether; carries the\ntone in one interval",
                xy=(fin + 52, 2.0), xytext=(fin + 95, 2.0), fontsize=8.5, va="center", color="#333",
                arrowprops=dict(arrowstyle="-", color="#777", lw=0.7))
    header(gs[0, :], "A   One trial at step 40 ms, comodulated: a new order in each interval, kept in all 5 "
           "repetitions of that interval",
           "Orders list the bands earliest first; [ ] = the tone band, always at 0. The noise is never frozen: "
           "every burst is new noise at its natural level.", dy=-0.012)

    # ---- B: staircase, then shuffled ------------------------------------------------------
    sub = gs[1, 0:2].subgridspec(1, 2, wspace=0.12)
    a1 = fig.add_subplot(sub[0, 0])
    cycle_axis(a1, step, {b: b - 2 for b in range(5)}, "1. a staircase, step s", labels=True)
    a2 = fig.add_subplot(sub[0, 1])
    cycle_axis(a2, step, order, "2. the bands shuffled over it")
    for a in (a1, a2):
        a.set_xlabel("ms re the tone band", fontsize=8)
    header(gs[1, 0:2], "B   How a pattern is made (one repetition, s = 40 ms)",
           "offsets -2s, -s, 0, +s, +2s; the tone band keeps 0 and the four\nothers are shuffled: 24 orders, one per interval")

    # ---- C: four step sizes ---------------------------------------------------------------
    sub = gs[1, 2:4].subgridspec(1, 4, wspace=0.10)
    for i, s_ in enumerate(STEPS):
        a = fig.add_subplot(sub[0, i])
        ov = max(0.0, (BURST - s_) / BURST)
        cycle_axis(a, s_, order, f"step {s_:g} ms\noverlap {ov:.0%}")
    header(gs[1, 2:4], "C   The four steps, same order",
           "neighbouring bands overlap by (50 - s) / 50; at 40 ms each band\noverlaps its two neighbours by 10 ms")

    # ---- D: a new order on every trial ------------------------------------------------------
    sub = gs[1, 4:6].subgridspec(1, 4, wspace=0.10)
    nxt = trial_orders(rng)[0]
    for i, (o, name) in enumerate(zip(orders + [nxt], ["interval 1", "interval 2", "interval 3",
                                                       f"next trial\ninterval 1"])):
        a = fig.add_subplot(sub[0, i])
        cycle_axis(a, step, o, name)
    header(gs[1, 4:6], "D   A new order in every interval",
           "the trial in A: three intervals, three different orders, drawn\nfrom the 24 without replacement "
           "(each order 4 times per track)")

    # ---- E: comodulated vs random -----------------------------------------------------------
    sub = gs[2, 0:2].subgridspec(1, 2, wspace=0.10)
    a1 = fig.add_subplot(sub[0, 0])
    a2 = fig.add_subplot(sub[0, 1])
    for a, co, name in ((a1, True, "comodulated"), (a2, False, "random")):
        draw_interval(a, 0.0, step, [order], burst_levels(rng, co))
        band_axis(a, labels=(a is a1), tone_label=False)
        a.set_xlim(-20, INTERVAL + 20)
        a.set_xticks([0, 500, 1000])
        a.tick_params(axis="x", labelsize=7.5)
        a.set_title(name, fontsize=9, pad=3)
        a.set_xlabel("ms", fontsize=8)
    header(gs[2, 0:2], "E   Comodulated and random (one interval each)",
           "comodulated: the j-th burst of every band is the same noise, so the\nlevels move together; random: "
           "independent noise per band. Same timing.")

    # ---- F: re-ordered control ----------------------------------------------------------------
    a = fig.add_subplot(gs[2, 2:4])
    draw_interval(a, 0.0, step, reordered(rng), burst_levels(rng, True))
    band_axis(a, labels=False)
    a.set_xlim(-20, INTERVAL + 20)
    a.tick_params(axis="x", labelsize=7.5)
    a.set_xlabel("ms", fontsize=8)
    header(gs[2, 2:4], "F   Control, if there is time: re-ordered every repetition",
           "the same five offsets, a new order in every repetition: nothing to learn.\n"
           "No band jumps from +80 to -80 (its two bursts would overlap).")

    # ---- G: band alone --------------------------------------------------------------------
    a = fig.add_subplot(gs[2, 4:6])
    draw_interval(a, 0.0, 0.0, [order], burst_levels(rng, True), tone=True, bands=[TONE])
    band_axis(a, labels=False)
    a.set_xlim(-20, INTERVAL + 20)
    a.tick_params(axis="x", labelsize=7.5)
    a.set_xlabel("ms", fontsize=8)
    header(gs[2, 4:6], "G   Band alone (drawn with the tone)",
           "only the 1-kHz tone band: the masking one band gives.\n"
           "CMR = threshold(random) - threshold(comodulated), at each step")

    # ---- H: the visit ---------------------------------------------------------------------
    ax = fig.add_subplot(gs[3, :])
    seg_col = {"setup": "#cdd2da", "questions": "#aab3c0", "instructions": "#aab3c0",
               "practice": "#7f8da1", "main": "#3d6fa6", "break": "#e6eaf0", "ratings": "#7f8da1",
               "debrief": "#cdd2da"}
    order_segs = ["setup", "questions", "instructions", "practice", "main", "ratings", "debrief"]
    for p, y in {"A": 2.0, "B": 1.0, "C": 0.0}.items():
        m = plans[p]
        x = 0.0
        for key in order_segs:
            if key == "main":
                chunk, brk = m["parts"]["main"] / 4, m["parts"]["breaks"] / 3
                for k in range(4):
                    ax.barh(y, chunk, left=x, height=0.42, color=seg_col["main"], edgecolor="white", lw=0.8)
                    x += chunk
                    if k < 3:
                        ax.barh(y, brk, left=x, height=0.42, color=seg_col["break"], edgecolor="white", lw=0.8)
                        x += brk
            else:
                ax.barh(y, m["parts"][key], left=x, height=0.42, color=seg_col[key], edgecolor="white", lw=0.8)
                x += m["parts"][key]
        ax.text(max(x, 45.0) + 0.45, y, f"{m['total']:.0f} min", va="center", fontsize=10.5,
                fontweight="bold", color="#b02a20" if m["total"] > 46 else "#1b5e20")
        ax.text(0.0, y + 0.27, f"{PLANS[p]['label']}:  {m['tracks']} tracks x {TRIALS_PER_TRACK} trials = "
                f"{m['trials']} trials, {m['parts']['main']:.0f} min of trials", fontsize=9, va="bottom")
    ax.axvline(45, color="#b02a20", lw=1.0, ls="--")
    ax.text(45.0, -0.47, "45 min", color="#b02a20", fontsize=8.5, va="bottom", ha="center",
            bbox=dict(fc="white", ec="none", pad=1.0))
    ax.set_xlim(0, 56)
    ax.set_ylim(-0.5, 2.62)
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("minutes from arrival; one trial = 6.5 s (4.2 s of sound + box, answer, feedback)", fontsize=8.5)
    legend = [Patch(color=seg_col[k], label=lab) for k, lab in
              (("setup", "setup, debrief"), ("questions", "questions, instructions"),
               ("practice", "practice, ratings"), ("main", "main trials"), ("break", "breaks"))]
    ax.legend(handles=legend, ncol=5, fontsize=8.3, frameon=False, loc="lower right",
              bbox_to_anchor=(1.0, 1.0))
    header(gs[3, :], "H   The visit", dy=-0.012)
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------------- main
def main():
    print("design: 5 x 20-Hz bands 0.25-4 kHz, 50-ms bursts (10-ms ramps), one per band every 200 ms, "
          "5 precursors + synchronous final burst")
    print(f"  interval {INTERVAL:.0f} ms (the same at every step), ISI {ISI:.0f} ms, trial sound "
          f"{TRIAL_SOUND / 1000:.2f} s; one trial {trial_s():.2f} s naive, {trial_s(OVERHEAD_P00):.2f} s as P00")
    for s in STEPS:
        print(f"  step {s:4.0f} ms: offsets {[int(k * s) for k in (-2, -1, 0, 1, 2)]} ms, neighbour overlap "
              f"{max(0, BURST - s) / BURST:.0%}, last gap before the final burst "
              f"{NOI - BURST - 2 * s:.0f} to {NOI - BURST + 2 * s:.0f} ms")
    plans = {p: session_minutes(p) for p in PLANS}
    print("\nsessions (naive pace; as P00 in brackets):")
    for p, m in plans.items():
        fast = session_minutes(p, OVERHEAD_P00)
        print(f"  {PLANS[p]['label']:<48} {m['tracks']:2d} tracks, {m['trials']} trials: main "
              f"{m['parts']['main']:.1f} min, visit {m['total']:.1f} min ({fast['total']:.1f})")

    print("\nprecision (tcmr staircase + fit, 32 trials, one track per condition):")
    cases = {"practised (P00 slope 2.2 dB, 2% lapses)": (2.15, 0.02),
             "naive, middle (slope 3.0 dB, 3% lapses)": (3.0, 0.03),
             "naive, poor (slope 3.8 dB, 5% lapses)": (3.76, 0.05)}
    sds = {}
    for name, (sg, lp) in cases.items():
        r = contrast_sd(sg, lp)
        sds[name] = r
        print(f"  {name:<40} one track {r['track_sd']:.2f} dB; CMR(0) - CMR(40) {r['contrast_sd']:.2f} dB")
    tau = 1.5
    tot = {k: math.hypot(v["contrast_sd"], tau) for k, v in sds.items()}
    print(f"  plus listeners differing in the true drop (sd {tau} dB): total sd "
          + ", ".join(f"{v:.2f}" for v in tot.values()) + " dB")
    print("\nlisteners for a fall in CMR from step 0 to step 40 (two-sided alpha 0.05), "
          "80% power [90%], per case above:")
    for d in (2.0, 2.5, 3.0, 4.0):
        print(f"  drop {d:3.1f} dB: " + "   ".join(f"{n_needed(d, v):3d} [{n_needed(d, v, 0.9):3d}]"
                                              for v in tot.values()))
    for n in (12, 16, 20, 24):
        print(f"  N = {n}: smallest drop found with 80% power "
              + ", ".join(f"{mde(v, n):.1f}" for v in tot.values()) + " dB")
    out = ROOT / "literature" / "tcmr_frozen_paradigm.png"
    figure(out, plans)
    print(f"\nfigure: {out}")


if __name__ == "__main__":
    main()
