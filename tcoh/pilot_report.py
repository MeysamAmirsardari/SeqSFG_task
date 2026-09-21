"""Per-listener curves and a group overview for the six-listener repeatability pilot.

What this reports, and what it refuses to
------------------------------------------
The pilot asks four questions and this module answers exactly those: can listeners do the
task, does the synchrony advantage recur across people, how repeatable is an individual
threshold, and is the behaviour near full alternation worth a targeted follow-up.

Four things are therefore deliberately absent.

*No kappa.* kappa is normalised by a B-only ceiling. This preset does not measure one, so the
denominator does not exist and any number put in its place would be invented.

*No lambda2/lambda1 label on the behavioural axis.* What is plotted is a displacement
threshold in milliseconds. The model's index is a different quantity in different units; the
two may be shown side by side, and `model_overlay` does that, but the overlay is a visual
comparison and is labelled as one.

*No bootstrap interval from two tracks.* Two numbers do not support a confidence interval that
anyone should read as precision. Repeatability is reported as the log ratio of the two tracks,
which is what two numbers can actually say, and the group summary uses the spread ACROSS
LISTENERS, with n = the number of listeners, never the number of tracks.

*No monotone fitting.* The curve is plotted as measured. If it is not monotone, that is the
result.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .analysis import _f, _i, diagnostics, load, replay_tracks, thresholds
from .config import Config, validate
from .session import read_json


@dataclass
class ListenerResult:
    code: str
    sessions: List[str]
    cfg_hash: str
    lags: List[float]
    tracks_ms: Dict[float, List[float]]        # usable track estimates, by lag
    censored_ms: Dict[float, List[float]]      # boundary-limited: reported, never averaged
    geomean_ms: Dict[float, Optional[float]]
    log_ratio: Dict[float, Optional[float]]    # |log(t2/t1)|, the repeatability of one lag
    n_tracks_attempted: int
    n_converged: int
    n_censored: int
    catch_correct: int
    catch_total: int
    n_timeouts: int
    press_1_rate: float
    press_1_p: float
    duration_min: Optional[float]
    level_db_spl: Optional[float]
    usable: bool
    notes: List[str] = field(default_factory=list)

    @property
    def synchrony_advantage(self) -> Optional[float]:
        """log(worst lag / 0%) -- the one contrast this pilot is powered to see."""
        base = self.geomean_ms.get(0.0)
        others = [v for k, v in self.geomean_ms.items() if k > 0 and v]
        if not base or not others:
            return None
        return math.log(max(others) / base)


def _duration_min(meta: dict) -> Optional[float]:
    try:
        a = dt.datetime.fromisoformat(meta["start_time"])
        b = dt.datetime.fromisoformat(meta["end_time"])
    except (KeyError, TypeError, ValueError):
        return None
    return (b - a).total_seconds() / 60.0


def listener(dirs: Sequence[Path], cfg: Optional[Config] = None) -> ListenerResult:
    """One listener's curve. `dirs` are that listener's session directories."""
    from scipy import stats
    dirs = [Path(d) for d in dirs]
    rows, metas = load(dirs)
    if cfg is None:
        cfg = Config.from_dict(metas[0]["config"])
    res = thresholds(rows, cfg)
    diag = diagnostics(rows, metas, cfg, res)
    main = [r for r in rows if r.get("phase") == "main" and not _i(r, "timed_out", 0)]
    n1 = sum(1 for r in main if str(r.get("response", "")).strip() == "1")

    lags = sorted({c.lag_pct for c in validate(cfg).conditions if c.a_kind == "coherent"})
    tracks, cens, geo, ratio = {}, {}, {}, {}
    for p in lags:
        r = next((v for v in res.values() if v.lag_pct == p and v.a_kind == "coherent"), None)
        tracks[p] = list(r.track_thresholds) if r else []
        cens[p] = list(r.censored_thresholds or []) if r else []
        geo[p] = r.geomean_ms if r else None
        ratio[p] = (abs(math.log(max(tracks[p]) / min(tracks[p])))
                    if len(tracks[p]) == 2 else None)

    notes = []
    if any(cens.values()):
        notes.append("boundary-limited track(s) at "
                     + ", ".join(f"{p:g}%" for p, v in cens.items() if v)
                     + " -- excluded from the geometric mean and shown as bounds")
    if diag["n_timeouts"]:
        notes.append(f"{diag['n_timeouts']} trial(s) timed out")
    lvl = None
    for m in metas:
        c = m.get("calibration") or {}
        if c.get("measured_db_spl") is not None:
            lvl = float(c["measured_db_spl"])
    if lvl is None:
        notes.append("level was not measured; every level is nominal")
    if not diag["catch_ok"]:
        notes.append("CATCH FAILED -- session not usable under the protocol")

    catch_n = diag["n_catch"]
    catch_ok = int(round(diag["catch_p_correct"] * catch_n)) if catch_n else 0
    dur = [_duration_min(m) for m in metas]
    return ListenerResult(
        code=metas[0].get("participant_code", dirs[0].parent.name),
        sessions=[str(d) for d in dirs], cfg_hash=cfg.hash(), lags=lags,
        tracks_ms=tracks, censored_ms=cens, geomean_ms=geo, log_ratio=ratio,
        n_tracks_attempted=diag["tracks_attempted"], n_converged=diag["tracks_converged"],
        n_censored=diag["tracks_censored"], catch_correct=catch_ok, catch_total=catch_n,
        n_timeouts=diag["n_timeouts"], press_1_rate=n1 / max(len(main), 1),
        press_1_p=float(stats.binomtest(n1, max(len(main), 1), 0.5).pvalue),
        duration_min=sum(x for x in dur if x) if any(dur) else None,
        level_db_spl=lvl, usable=bool(diag["catch_ok"]), notes=notes)


def group(results: Sequence[ListenerResult]) -> dict:
    """Descriptive summary ACROSS LISTENERS. n is the number of people, never of tracks."""
    usable = [r for r in results if r.usable]
    lags = sorted({p for r in usable for p in r.lags})
    per_lag = {}
    for p in lags:
        vals = [r.geomean_ms[p] for r in usable if r.geomean_ms.get(p)]
        per_lag[p] = {
            "n_listeners": len(vals),
            "geomean_ms": float(np.exp(np.mean(np.log(vals)))) if vals else None,
            # spread ACROSS listeners, in log units; with six people this is a description,
            # not an inference
            "sd_log": float(np.std(np.log(vals), ddof=1)) if len(vals) > 1 else None,
            "min_ms": float(min(vals)) if vals else None,
            "max_ms": float(max(vals)) if vals else None,
        }
    adv = [r.synchrony_advantage for r in usable if r.synchrony_advantage is not None]
    rep = [v for r in usable for v in r.log_ratio.values() if v is not None]
    return {
        "n_listeners_run": len(results), "n_usable": len(usable),
        "excluded": [r.code for r in results if not r.usable],
        "per_lag": per_lag,
        "synchrony_advantage_log": adv,
        "synchrony_advantage_n_positive": int(sum(1 for v in adv if v > 0)),
        "repeatability_median_log_ratio": float(np.median(rep)) if rep else None,
        "repeatability_median_factor": float(np.exp(np.median(rep))) if rep else None,
        "n_track_pairs": len(rep),
    }


def model_overlay(lags: Sequence[float], cfg: Config) -> dict:
    """The model's index at these lags, for VISUAL comparison only.

    Returned separately from every behavioural quantity, and the caller is expected to keep it
    on its own axis. Agreement here is not evidence that the mechanism is the one the model
    describes: the model's curve correlates with a straight line in dT at better than r = 0.99
    over this range, so no set of six points can tell the two apart.
    """
    from .model import prediction_band
    b = prediction_band(tuple(lags), tone_ms=cfg.tone_ms, soa_ms=cfg.soa_ms,
                        n_tones=cfg.n_tones)
    return {"pcts": b["pcts"], "mid": b["mid"], "lo": b["lo"], "hi": b["hi"],
            "caveat": "visual comparison only; not a mechanistic validation"}


def format_listener(r: ListenerResult) -> str:
    L = [f"listener {r.code}   config {r.cfg_hash}   {len(r.sessions)} session(s)"]
    dur = f"{r.duration_min:.1f} min" if r.duration_min else "duration unknown"
    lvl = (f"{r.level_db_spl:.0f} dB SPL/tone" if r.level_db_spl is not None
           else "level NOT measured")
    L.append(f"  {dur}   {lvl}   catch {r.catch_correct}/{r.catch_total}   "
             f"tracks {r.n_converged}/{r.n_tracks_attempted} converged"
             + (f", {r.n_censored} boundary-limited" if r.n_censored else "")
             + (f", {r.n_timeouts} timed out" if r.n_timeouts else ""))
    L.append(f"  pressed '1' on {100 * r.press_1_rate:.1f}% of trials (p={r.press_1_p:.3f})")
    L.append("")
    L.append("   dT     track 1   track 2   geomean   repeatability")
    for p in r.lags:
        t = r.tracks_ms.get(p, [])
        c = r.censored_ms.get(p, [])
        cells = [f"{v:7.2f}" for v in sorted(t)] + [f"{v:6.1f}*" for v in sorted(c)]
        cells += ["      -"] * (2 - len(cells))
        g = r.geomean_ms.get(p)
        rr = r.log_ratio.get(p)
        L.append(f"  {p:5.4g}%  {cells[0]}   {cells[1]}   "
                 + (f"{g:7.2f}" if g else "      -")
                 + ("   " + (f"{math.exp(rr):.2f}x" if rr is not None else "n/a")))
    if any(r.censored_ms.values()):
        L.append("  * boundary-limited: a bound, not a threshold; excluded from the geomean")
    for n in r.notes:
        L.append(f"  note: {n}")
    return "\n".join(L)


def format_group(results: Sequence[ListenerResult], g: dict) -> str:
    L = ["=" * 78, "GROUP OVERVIEW", "=" * 78,
         f"  {g['n_usable']} of {g['n_listeners_run']} listeners usable"
         + (f"; excluded: {', '.join(g['excluded'])}" if g["excluded"] else "")]
    L.append("")
    L.append("  per lag, across listeners (n is people, not tracks)")
    L.append("    dT       n   geomean   range              sd(log)")
    for p, s in g["per_lag"].items():
        if not s["geomean_ms"]:
            continue
        L.append(f"   {p:5.4g}%   {s['n_listeners']}   {s['geomean_ms']:7.2f}   "
                 f"{s['min_ms']:6.2f} - {s['max_ms']:6.2f} ms   "
                 + (f"{s['sd_log']:.3f}" if s["sd_log"] is not None else "n/a"))
    L.append("")
    adv = g["synchrony_advantage_log"]
    if adv:
        L.append(f"  synchrony advantage (worst lag / 0%): positive in "
                 f"{g['synchrony_advantage_n_positive']}/{len(adv)} listeners, "
                 f"median {math.exp(float(np.median(adv))):.2f}x")
    if g["repeatability_median_factor"]:
        L.append(f"  repeatability: median |log ratio| between the two tracks of a lag = "
                 f"{g['repeatability_median_factor']:.2f}x over {g['n_track_pairs']} pairs")
    L.append("")
    L.append("  This is a descriptive summary of six people. It is not an estimate of a")
    L.append("  population mean, the curve has not been fitted or smoothed, and no kappa or")
    L.append("  lambda2/lambda1 is computed -- B-only was not measured in this preset.")
    L.append("=" * 78)
    return "\n".join(L)


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
def figures(results: Sequence[ListenerResult], cfg: Config, out_dir: Path,
            overlay: bool = False) -> List[Path]:
    """Two figures: every listener's curve, and the group overview.

    The dT axis runs 100% on the LEFT to 0% on the RIGHT throughout, matching the intended
    figure. Thresholds are on a log axis because the quantity is a ratio and the staircase
    estimates it in log units.
    """
    from .plots import _mpl
    plt = _mpl()
    import matplotlib
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    usable = [r for r in results if r.usable]
    g = group(results)
    lags = sorted({p for r in usable for p in r.lags})
    made = []

    def _ax(a):
        a.set_xticks(lags)
        a.set_xticklabels([f"{p:g}" for p in lags])
        a.invert_xaxis()                     # 100% left, 0% right
        a.set_yscale("log")
        a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.set_yticks([1, 2, 3, 5, 8, 12, 20, 30, 50])
        a.set_yticklabels(["1", "2", "3", "5", "8", "12", "20", "30", "50"])
        a.set_ylim(0.8, 60)
        a.grid(alpha=.25)
        a.set_xlabel("onset lag ΔT (%)      100 = alternating, 0 = synchronous")
        a.set_ylabel("final-B displacement threshold (ms)")

    # -- per listener ---------------------------------------------------------
    n = max(len(usable), 1)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.7 * nrow), squeeze=False)
    for ax, r in zip(axes.ravel(), usable):
        for p in r.lags:
            for v in r.tracks_ms.get(p, []):
                ax.plot(p, v, "o", mfc="none", mec="#2471a3", ms=8, zorder=2)
            for v in r.censored_ms.get(p, []):
                ax.plot(p, v, "v", color="#c0392b", ms=9, zorder=2)
        xs = [p for p in r.lags if r.geomean_ms.get(p)]
        ax.plot(xs, [r.geomean_ms[p] for p in xs], "o-", color="k", lw=2, ms=9, zorder=3)
        _ax(ax)
        ax.set_title(f"{r.code}", fontsize=11)
    for ax in axes.ravel()[len(usable):]:
        ax.axis("off")
    h = [plt.Line2D([], [], color="k", marker="o", lw=2, label="geometric mean of the two tracks"),
         plt.Line2D([], [], color="#2471a3", marker="o", mfc="none", lw=0, label="individual track"),
         plt.Line2D([], [], color="#c0392b", marker="v", lw=0, label="boundary-limited (a bound)")]
    fig.legend(handles=h, frameon=False, fontsize=9, loc="lower center", ncol=3)
    fig.suptitle("each listener's threshold curve, as measured", fontsize=12)
    fig.tight_layout(rect=[0, .05, 1, .96])
    p1 = out_dir / "pilot_listeners.png"
    fig.savefig(p1, dpi=150); plt.close(fig); made.append(p1)

    # -- group ----------------------------------------------------------------
    fig, ax = plt.subplots(1, 2 if overlay else 1, figsize=(12 if overlay else 6.6, 4.8),
                           squeeze=False)
    a = ax[0][0]
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, r in enumerate(usable):
        xs = [p for p in r.lags if r.geomean_ms.get(p)]
        a.plot(xs, [r.geomean_ms[p] for p in xs], "-", color=cmap[i % 10], lw=1.3,
               alpha=.75, marker="o", ms=5, label=r.code)
    xs = [p for p, s in g["per_lag"].items() if s["geomean_ms"]]
    a.plot(xs, [g["per_lag"][p]["geomean_ms"] for p in xs], "o-", color="k", lw=3, ms=11,
           zorder=5, label=f"group ({g['n_usable']} listeners)")
    _ax(a)
    a.set_title("individual listeners and the group geometric mean", fontsize=11)
    a.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")

    if overlay:
        b = ax[0][1]
        mo = model_overlay(lags, cfg)
        b.plot(mo["pcts"], mo["mid"], "-", color="#5d6d7e", lw=2.5)
        b.fill_between(mo["pcts"], mo["lo"], mo["hi"], color="#5d6d7e", alpha=.2, lw=0)
        b.set_xticks(lags); b.set_xticklabels([f"{p:g}" for p in lags]); b.invert_xaxis()
        b.grid(alpha=.25)
        b.set_xlabel("onset lag ΔT (%)")
        b.set_ylabel("model segregation index  λ₂/λ₁")
        b.set_title("the model, on its own axis and its own scale", fontsize=11)
        b.text(.5, -.30, "Shown beside the data, not on it. This is a visual comparison;\n"
                         "the model's curve is within r = 0.99 of a straight line over this\n"
                         "range, so six points cannot tell the two apart.",
               transform=b.transAxes, ha="center", va="top", fontsize=8, color="#555")
    fig.tight_layout(rect=[0, .08 if overlay else 0, 1, 1])
    p2 = out_dir / "pilot_group.png"
    fig.savefig(p2, dpi=150); plt.close(fig); made.append(p2)
    return made


def report(dirs_by_listener: Dict[str, Sequence[Path]], cfg: Optional[Config] = None,
           out_dir: Optional[Path] = None, overlay: bool = False) -> str:
    """The whole pilot: one block per listener, then the group overview."""
    results = [listener(d, cfg) for d in dirs_by_listener.values()]
    g = group(results)
    parts = [format_listener(r) for r in results]
    text = "\n\n".join(parts) + "\n\n" + format_group(results, g)
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "pilot_report.txt").write_text(text)
        if results:
            base = cfg or Config.from_dict(read_json(Path(results[0].sessions[0])
                                                     / "session.json")["config"])
            figures(results, base, out_dir, overlay=overlay)
    return text
