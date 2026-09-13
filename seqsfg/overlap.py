"""Temporal Overlap Pilot: does an asynchronous figure depend on how LONG its components
overlap, on what FRACTION of them overlaps, or only on how far apart their onsets are?

Two tones can overlap by 50% and share very different amounts of time. A 20 ms tone stepped by
10 ms and a 40 ms tone stepped by 20 ms both overlap by half; the first shares 10 ms and the
second 20. Whether that matters is the question, not an assumption, and nothing in the analysis
here is built to find it.

The seven cells cross two component durations with several onset steps:

    T (ms)   step (ms)   adjacent overlap   fraction
      20         0            20 ms          100%
      20        10            10 ms           50%
      20        20             0 ms            0%
      40         0            40 ms          100%
      40        20            20 ms           50%
      40        30            10 ms           25%
      40        40             0 ms            0%

and three comparisons were chosen before any data:

    equal fraction     (20,10) against (40,20)
    equal absolute     (20,10) against (40,30)
    equal separation   (20,20) against (40,20)

with a synchronous and a zero-overlap reference at each duration.

Three things this file is careful about, because each is easy to get wrong.

*Adjacent overlap is not common overlap.* max(0, T - step) is what two CONSECUTIVE components
share. What all seven share is max(0, T - (N-1)*step), which is zero in every cell except the
two synchronous ones. Both are logged, separately, and the report says so. At step = T the tones
meet; there is no positive silent gap.

*The background must not move when the figure's duration does.* Changing cfg.tone_dur_ms would
change the cloud too, and the contrast would no longer be about the figure. So the candidate
components carry their own duration (Interval.dur) while the background keeps cfg.tone_dur_ms.

*An absent trial holds the same tones.* Same channels, same durations, same amplitudes, same
counts -- only the arrangement is randomised. Otherwise "longer tones" would mean "figure
present" and the task would be a duration-detection task wearing a costume.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import zlib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from . import measure as M
from .config import Config, Derived, validate
from .stimulus import (BACKGROUND, FIGURE, Interval, PlacementError, render_interval,
                       sample_figure_set, tone_envelope)
from .yesno import criterion_yesno, dprime_yesno, feature_separation, learnt_observer

ABSENT_KINDS = ("plain", "roving")
# plain  : the primary contrast for this pilot. An absent trial has no recurring arrangement at
#          all -- the candidate tones are the same tones, scattered at random over the scene.
#          The question is then literally "was a pattern there?".
# roving  : the control kept from the asynchrony task. Both classes carry organised groups and
#          only the recurrence of one frequency set differs. It answers a DIFFERENT question --
#          recurring frequency identity in the presence of other organised groups -- and is
#          analysed separately. It is never substituted for 'plain'.

DEFAULT_CELLS = ((20.0, 0.0), (20.0, 10.0), (20.0, 20.0),
                 (40.0, 0.0), (40.0, 20.0), (40.0, 30.0), (40.0, 40.0))
PLANNED = (("equal overlap fraction", (20.0, 10.0), (40.0, 20.0)),
           ("equal absolute overlap", (20.0, 10.0), (40.0, 30.0)),
           ("equal onset separation", (20.0, 20.0), (40.0, 20.0)))


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------
def geometry(cfg: Config, tone_ms: float, step_ms: float, n: int) -> dict:
    """Every overlap quantity this pilot distinguishes, for one cell. All in ms unless named."""
    adj = max(0.0, tone_ms - step_ms)
    common = max(0.0, tone_ms - (n - 1) * step_ms)
    env_adj, env_frac = envelope_overlap(cfg, tone_ms, step_ms)
    return dict(tone_ms=float(tone_ms), step_ms=float(step_ms), n=int(n),
                adjacent_overlap_ms=adj,
                adjacent_overlap_fraction=adj / tone_ms if tone_ms else 0.0,
                total_extent_ms=tone_ms + (n - 1) * step_ms,
                common_overlap_ms=common,
                common_overlap_fraction=common / tone_ms if tone_ms else 0.0,
                gap_ms=max(0.0, step_ms - tone_ms),
                envelope_overlap_ms=env_adj,
                envelope_overlap_fraction=env_frac,
                duty_ms=tone_ms - cfg.ramp_ms,      # full-amplitude time, ramps are half-weight
                ramp_ms=cfg.ramp_ms)


def envelope_overlap(cfg: Config, tone_ms: float, step_ms: float) -> Tuple[float, float]:
    """Overlap weighted by the raised-cosine envelopes, for ADJACENT components.

    Definition: with e(t) the gated amplitude envelope of one component,

        envelope overlap (ms)       = (1/sr) * INTEGRAL e(t) e(t - step) dt  * 1000
        envelope overlap (fraction) = that integral divided by INTEGRAL e(t)^2 dt

    So the fraction is 1 at step 0 and 0 once the tones no longer touch. It differs from the
    geometric overlap because the ramps are 5 ms whatever the duration: a 20 ms component is
    half ramp and a 40 ms one a quarter, so the same geometric overlap is worth less envelope
    when it falls on the ramps. Both numbers are reported; neither is a substitute for the
    other.
    """
    e = tone_envelope(cfg, tone_ms)
    lag = cfg.ms_to_samples(step_ms)
    den = float(np.sum(e * e))
    num = float(np.sum(e[lag:] * e[:e.size - lag])) if 0 <= lag < e.size else 0.0
    return num / cfg.sample_rate * 1000.0, (num / den if den else 0.0)


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class OverlapConfig:
    """This pilot's own parameters. Not Config fields: adding those would move the hash of
    every configuration already run and break resume on sessions already collected."""
    cells: Tuple[Tuple[float, float], ...] = DEFAULT_CELLS
    absent_class: str = "plain"
    trials_per_cell: int = 10          # present; the same number of absent trials
    figure_amplitude: Optional[float] = None       # None -> cfg.tone_amplitude, i.e. the same
    jitter_ms: float = 120.0           # shared per-recurrence onset jitter, FIXED across cells
    practice_trials: int = 16          # split over the two synchronous cells, with feedback
    feedback_main: bool = False
    # Off by default. Feedback in a yes/no main block is not neutral: it drives the listener
    # towards the criterion that maximises accuracy, so c stops being a free parameter you are
    # measuring and becomes one the procedure imposed, and it lets performance drift over the
    # session in a way that is confounded with trial order. d' survives it better than c does.
    # It is a defensible choice for a feasibility run, where keeping the listener calibrated
    # matters more than a clean criterion -- but the analysis has to say it was on, and it does.
    feedback_main_first: int = 0
    # A middle course: feedback on the first N main trials only, then silence. Calibrates the
    # listener at the start without touching the rest of the block. Ignored when feedback_main
    # is True, which feeds back on every trial.
    response_timeout_s: float = 4.0    # measured from the end of the sound; recorded as a miss
    max_class_run: int = 4
    max_cell_run: int = 2
    n_boot: int = 4000

    def __post_init__(self):
        # to_dict() renders cells as lists, and every override path rebuilds the object from it,
        # so normalise here rather than making each caller remember.
        object.__setattr__(self, "cells", tuple((float(a), float(b)) for a, b in self.cells))

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cells"] = [list(c) for c in self.cells]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "OverlapConfig":
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown overlap keys: {sorted(unknown)}")
        kw = dict(d)
        if "cells" in kw:
            kw["cells"] = tuple((float(a), float(b)) for a, b in kw["cells"])
        return cls(**kw)

    @property
    def durations(self) -> Tuple[float, ...]:
        return tuple(sorted({t for t, _ in self.cells}))

    def reference(self, tone_ms: float) -> Optional[Tuple[float, float]]:
        """The synchronous cell at this duration, which the exploratory contrasts use."""
        for t, s in self.cells:
            if t == tone_ms and s == 0.0:
                return (t, s)
        return None


def load_preset(path) -> Tuple[Config, OverlapConfig]:
    raw = {k: v for k, v in json.loads(Path(path).read_text()).items() if not k.startswith("_")}
    if "config" not in raw or "overlap" not in raw:
        raise ValueError(f"{path}: needs both a 'config' and an 'overlap' section")
    return Config.from_dict(raw["config"]), OverlapConfig.from_dict(raw["overlap"])


def widest_footprint_ms(cfg: Config, ocfg: OverlapConfig) -> float:
    """The room one recurrence needs in the cell that needs most: jitter plus the widest extent."""
    n = cfg.n_components
    return ocfg.jitter_ms + max(t + (n - 1) * s for t, s in ocfg.cells)


def check(cfg: Config, ocfg: OverlapConfig) -> None:
    """Refuse anything that would make the seven cells not comparable."""
    errs: List[str] = []
    n, K = cfg.n_components, cfg.n_elements
    if ocfg.absent_class not in ABSENT_KINDS:
        errs.append(f"absent_class={ocfg.absent_class!r}; choose from {ABSENT_KINDS}")
    if not ocfg.cells:
        errs.append("cells must name at least one (tone_ms, step_ms) pair")
    if len(set(ocfg.cells)) != len(ocfg.cells):
        errs.append("cells must be unique")
    for t, s in ocfg.cells:
        if t < 2 * cfg.ramp_ms:
            errs.append(f"a {t:g} ms component cannot carry two {cfg.ramp_ms:g} ms ramps")
        for name, v in (("tone", t), ("step", s)):
            if abs(v / cfg.grid_ms - round(v / cfg.grid_ms)) > 1e-6:
                errs.append(f"{name} {v:g} ms is not a multiple of grid_ms={cfg.grid_ms:g}")
        if s < 0:
            errs.append("step must be >= 0")
    # one timing schedule for every cell, sized for the widest one
    foot = widest_footprint_ms(cfg, ocfg)
    if foot > cfg.iei_min_ms:
        errs.append(f"the widest cell needs {foot:.0f} ms per recurrence ({ocfg.jitter_ms:.0f} ms "
                    f"of shared jitter plus a {foot - ocfg.jitter_ms:.0f} ms extent) but "
                    f"iei_min_ms is {cfg.iei_min_ms:.0f}, so recurrences would run into each "
                    f"other in that cell and not in the others. Raise iei_min_ms to >= {foot:.0f}")
    need = cfg.lead_max_ms + (K - 1) * cfg.iei_max_ms + foot + cfg.tail_min_ms
    if need > cfg.interval_dur_ms:
        errs.append(f"the schedule does not fit: {need:.0f} ms needed, interval_dur_ms is "
                    f"{cfg.interval_dur_ms:.0f}. Raise it to >= {need:.0f}")
    if cfg.tones_per_channel <= K:
        errs.append(f"tones_per_channel={cfg.tones_per_channel} leaves no background on a "
                    f"channel that spends {K} of its budget on the figure")
    if cfg.figure_band_channels is not None:
        errs.append("figure_band_channels must be null here: a banded element has a register, "
                    "and this pilot varies duration, not register")
    if ocfg.trials_per_cell < 4:
        errs.append("trials_per_cell < 4 gives a d' no interval can be put around")
    if ocfg.practice_trials % (2 * len(ocfg.durations)) and ocfg.practice_trials:
        errs.append(f"practice_trials should divide evenly into {2 * len(ocfg.durations)} "
                    f"(present and absent at each duration)")
    if ocfg.figure_amplitude is not None and not (0 < ocfg.figure_amplitude < 1):
        errs.append("figure_amplitude must be in (0, 1) or null")
    if n < 2 or K < 2:
        errs.append("n_components and n_elements must both be >= 2")
    if errs:
        raise ValueError("overlap configuration is not usable:\n  - " + "\n  - ".join(errs))


def notes(cfg: Config, ocfg: OverlapConfig) -> List[str]:
    out = [
        f"the same component amplitude is used at both durations, so a {max(ocfg.durations):g} ms "
        f"component carries about {10 * math.log10(max(ocfg.durations) / min(ocfg.durations)):.1f} dB "
        f"more energy than a {min(ocfg.durations):g} ms one. The synchronous cell at each duration "
        f"is what a duration effect would show up in; it does not remove the confound.",
        "adjacent overlap is not common overlap: all seven components share "
        f"{geometry(cfg, 40.0, 20.0, cfg.n_components)['common_overlap_ms']:.0f} ms in the "
        "(40, 20) cell, not 20 ms. Only the synchronous cells have positive common overlap.",
    ]
    if ocfg.feedback_main or ocfg.feedback_main_first:
        where = "every main trial" if ocfg.feedback_main else \
            f"the first {ocfg.feedback_main_first} main trials"
        out.append(f"feedback is on for {where}. d' tolerates that; the criterion does not -- c "
                   f"becomes a number the procedure imposed rather than one the listener chose. "
                   f"Read c as a description of what happened, not as a free parameter, and read "
                   f"the half-split in section [4] for drift.")
    if ocfg.absent_class == "roving":
        out.append("absent_class='roving' answers a different question from this pilot's: whether "
                   "a frequency set RECURS among other organised groups, not whether a pattern "
                   "was there. Analyse it separately and do not pool it with the plain cells.")
    return out


# ----------------------------------------------------------------------------
# stimulus
# ----------------------------------------------------------------------------
def _free_mask(on: np.ndarray, dur: np.ndarray, n_grid: int, new_dur: int) -> np.ndarray:
    """Where a tone of `new_dur` can start without overlapping anything already in the channel.

    Two tones [t1, t1+u1) and [t, t+new_dur) overlap when t1 - new_dur < t < t1 + u1, so each
    existing tone blocks that half-open range. Uniform-duration placement is the special case.
    """
    ok = np.ones(max(n_grid - new_dur + 1, 0), dtype=bool)
    for t, u in zip(np.asarray(on, int), np.asarray(dur, int)):
        ok[max(0, int(t) - new_dur + 1):min(ok.size, int(t) + int(u))] = False
    return ok


def _place(rng: np.random.Generator, on: List[int], dur: List[int], n_grid: int,
           new_dur: int, count: int) -> List[int]:
    """`count` onsets at free positions, chosen uniformly among those still free (sequential RSA).

    Raises PlacementError rather than returning a shorter list: a trial that could not be laid
    out is rejected and reseeded, never silently thinned.
    """
    out: List[int] = []
    for _ in range(count):
        free = np.flatnonzero(_free_mask(on, dur, n_grid, new_dur))
        if free.size == 0:
            raise PlacementError(f"no free position left for a {new_dur}-grid tone")
        t = int(free[rng.integers(free.size)])
        out.append(t); on.append(t); dur.append(new_dur)
    return out


def _schedule(rng: np.random.Generator, cfg: Config) -> np.ndarray:
    lead = rng.integers(cfg.ms_to_grid(cfg.lead_min_ms), cfg.ms_to_grid(cfg.lead_max_ms) + 1)
    ieis = rng.integers(cfg.ms_to_grid(cfg.iei_min_ms), cfg.ms_to_grid(cfg.iei_max_ms) + 1,
                        size=cfg.n_elements - 1)
    return np.concatenate([[lead], lead + np.cumsum(ieis)]).astype(int)


def build_interval(cfg: Config, ocfg: OverlapConfig, d: Derived, seed: int, tone_ms: float,
                   step_ms: float, present: bool, absent: Optional[str] = None,
                   max_rebuilds: int = 60) -> Interval:
    """One trial of the pilot.

    The structure a present and an absent trial of the same seed share -- which channels carry
    candidate tones, how many, how long they are, where the recurrences sit, the shared jitter --
    comes from a stream seeded on (seed, tone, step) alone. Only the arrangement of the candidate
    tones, and the background drawn around it, come from a second stream. So two sides built from
    one seed have identical tone inventories by construction, and the audit checks it per
    frequency AND per duration rather than on the total.
    """
    absent = absent or ocfg.absent_class
    P, N, K = d.n_channels, cfg.n_components, cfg.n_elements
    T = cfg.ms_to_grid(tone_ms)
    B = d.tone_dur_grid                      # background duration, fixed across every cell
    step = cfg.ms_to_grid(step_ms)
    jit = max(1, cfg.ms_to_grid(ocfg.jitter_ms))
    n_grid, Mbud = cfg.n_grid, cfg.tones_per_channel
    last: Optional[Exception] = None
    for attempt in range(max_rebuilds):
        # The shared structure is drawn WITHOUT the cell in the seed, so one seed gives the same
        # figure set, the same order, the same recurrence schedule and the same jitter in all
        # seven cells, and the only thing a cell changes is the components' duration and step.
        # That is what makes "the timing is identical across cells" a fact to check rather than a
        # distributional claim to argue about.
        rs = np.random.default_rng([int(seed), attempt, 0x0FA1])
        try:
            S = sample_figure_set(rs, P, N, cfg.figure_min_spacing_channels)
            order = rs.permutation(N)          # one arbitrary order, reused by every recurrence
            t_el = _schedule(rs, cfg)
            a = rs.integers(0, jit, size=K)
            rng = np.random.default_rng([int(seed), attempt, int(tone_ms), int(step_ms),
                                         int(present), 0x0FFE])
            if present or absent == "roving":
                if present:
                    sets = [S] * K
                    orders = [order] * K
                else:
                    # roving: a group at every recurrence, on a fresh set of channels and a fresh
                    # internal order each time, so nothing recurs. Its per-channel inventory does
                    # NOT match the present trial's -- see the audit and the README.
                    sets, orders = [], []
                    for k in range(K):
                        avoid = sets[-1] if sets else np.zeros(0, int)
                        cand = np.array([c for c in range(P) if c not in set(avoid.tolist())], int)
                        sets.append(np.sort(rng.choice(cand, size=N, replace=False)))
                        orders.append(rng.permutation(N))
                c_on, c_ch = [], []
                for k in range(K):
                    for i in range(N):
                        c_on.append(int(t_el[k] + a[k] + orders[k][i] * step))
                        c_ch.append(int(sets[k][i]))
                c_on = np.array(c_on, int); c_ch = np.array(c_ch, int)
                if c_on.max() + T > n_grid:
                    raise PlacementError("a recurrence runs past the end of the scene")
                el = np.repeat(np.arange(K), N)
                comp = np.tile(np.arange(N), K)
            else:
                # plain: the same candidate tones, the same channels, the same durations and
                # counts -- placed at random free times, so no arrangement recurs.
                c_on, c_ch = [], []
                for i in range(N):
                    c_on += _place(rng, [], [], n_grid, T, K); c_ch += [int(S[i])] * K
                c_on = np.array(c_on, int); c_ch = np.array(c_ch, int)
                el = np.full(c_on.size, -1); comp = np.full(c_on.size, -1)
                sets, orders = [], []

            # background: every channel to the same budget, around whatever it already holds
            b_on, b_ch = [], []
            for c in range(P):
                m = c_ch == c
                on = [int(t) for t in c_on[m]]; du = [T] * int(m.sum())
                need = Mbud - int(m.sum())
                if need < 0:
                    raise PlacementError(f"channel {c} holds more candidate tones than its budget")
                b_on += _place(rng, on, du, n_grid, B, need); b_ch += [c] * need
            b_on = np.array(b_on, int); b_ch = np.array(b_ch, int)
            onset = np.concatenate([c_on, b_on]); chan = np.concatenate([c_ch, b_ch])
            dur = np.concatenate([np.full(c_on.size, T, int), np.full(b_on.size, B, int)])
            kind = np.concatenate([np.full(c_on.size, FIGURE if present else BACKGROUND),
                                   np.full(b_on.size, BACKGROUND)])
            if int((onset + dur).max()) > n_grid:
                raise PlacementError("a tone runs past the end of the scene")
            return Interval(
                role="present" if present else absent, variant=f"T{tone_ms:g}_d{step_ms:g}",
                step_ms=float(step_ms), onset=onset, channel=chan,
                phase=rng.uniform(0.0, 2.0 * math.pi, size=onset.size), kind=kind,
                element=np.concatenate([el, np.full(b_on.size, -1)]),
                component=np.concatenate([comp, np.full(b_on.size, -1)]),
                element_onsets=t_el, element_sets=[np.asarray(s).copy() for s in sets],
                patterns=[np.asarray(o).copy() for o in orders], figure_set=S, dur=dur)
        except PlacementError as e:
            last = e
            continue
    raise PlacementError(f"overlap seed={seed} T={tone_ms} step={step_ms} present={present}: "
                         f"{max_rebuilds} rebuilds rejected ({last})")


def render(cfg: Config, ocfg: OverlapConfig, d: Derived, iv: Interval) -> np.ndarray:
    """The scene as audio, with the lead silence the runner plays before it."""
    amp = None
    if ocfg.figure_amplitude is not None:
        amp = np.where(iv.dur == cfg.ms_to_grid(cfg.tone_dur_ms) , cfg.tone_amplitude,
                       ocfg.figure_amplitude)
    lead = np.zeros(cfg.ms_to_samples(cfg.lead_silence_ms), dtype=np.float32)
    return np.concatenate([lead, render_interval(cfg, iv, d, amplitude=amp)])


def inventory(iv: Interval) -> Dict[Tuple[int, int], int]:
    """How many tones of each (channel, duration) the scene holds. The matching unit."""
    out: Dict[Tuple[int, int], int] = {}
    for c, u in zip(iv.channel.tolist(), np.asarray(iv.dur).tolist()):
        out[(int(c), int(u))] = out.get((int(c), int(u)), 0) + 1
    return out


def placement_ok(cfg: Config, d: Derived, iv: Interval) -> dict:
    """Explicit placement checks: nothing overlaps inside a channel, nothing is clipped by the end."""
    dur = iv.durations(d)
    bad = 0
    for c in np.unique(iv.channel):
        m = iv.channel == c
        o = np.argsort(iv.onset[m]); on = iv.onset[m][o]; du = dur[m][o]
        if on.size > 1 and np.any(on[1:] < on[:-1] + du[:-1]):
            bad += 1
    return dict(channels_with_overlap=int(bad),
                runs_past_end=bool(int((iv.onset + dur).max()) > cfg.n_grid),
                last_offset_ms=float((iv.onset + dur).max() * cfg.grid_ms))


# ----------------------------------------------------------------------------
# acoustic cue audit
# ----------------------------------------------------------------------------
def _count_trace(onset: np.ndarray, dur: np.ndarray, n_grid: int) -> np.ndarray:
    diff = np.zeros(n_grid + 1, dtype=int)
    np.add.at(diff, onset, 1)
    np.add.at(diff, np.minimum(onset + dur, n_grid), -1)
    return np.cumsum(diff)[:n_grid]


def features(cfg: Config, ocfg: OverlapConfig, d: Derived, iv: Interval, x: np.ndarray,
             ref: float) -> Dict[str, float]:
    """Level, envelope modulation, spectral energy and single-channel timing, from one scene.

    Duration-aware throughout: the shared measure_interval assumes one tone length for its
    schedule statistics, which is exactly the assumption this pilot breaks.
    """
    sr = cfg.sample_rate
    dur = iv.durations(d)
    out: Dict[str, float] = {}
    out["rms_db"] = float(20 * np.log10(max(np.sqrt(np.mean(x.astype(float) ** 2)), M.EPS)))
    out["peak"] = float(np.abs(x).max())
    env = M.frame_rms(x, sr, 2.0)
    for k, v in M.envelope_features(env, 2.0, cfg.iei_min_ms, cfg.iei_max_ms).items():
        out["env:" + k] = float(v)
    cnt = _count_trace(iv.onset, dur, cfg.n_grid)
    out["count_mean"], out["count_sd"] = float(cnt.mean()), float(cnt.std())
    out["count_max"] = float(cnt.max())
    out["on_time_ms"] = float(dur.sum() * cfg.grid_ms)          # total tone-seconds in the scene
    cenv = M.channel_envelopes(x, sr, d.channel_freqs_hz, 40.0, 1.0)
    spec = M.channel_power_db(cenv)
    out["spec_peakedness"] = M.peakedness(spec, cfg.n_components)
    out["spec_sd"] = float(spec.std()); out["spec_max"] = float(spec.max())
    st = M.on_states(cenv, ref)
    occ = st.mean(axis=1)
    out["occ_mean"], out["occ_peakedness"] = float(occ.mean()), M.peakedness(occ, cfg.n_components)
    ac = st.sum(axis=0)
    out["count_audio_mean"], out["count_audio_max"] = float(ac.mean()), float(ac.max())
    sc = M.single_channel_stats([iv.onset[iv.channel == c] * cfg.grid_ms for c in range(d.n_channels)], cfg)
    for k in ("ioi_cv", "ioi_sd", "ioi_min", "ioi_max", "frac_ioi_in_iei", "pairs_iei_norm"):
        v = sc[k]
        out[f"ch:{k}:max"], out[f"ch:{k}:min"] = float(v.max()), float(v.min())
        out[f"ch:{k}:mean"], out[f"ch:{k}:sd"] = float(v.mean()), float(v.std())
    return out


@dataclass
class CellAudit:
    tone_ms: float
    step_ms: float
    names: List[str]
    Fp: np.ndarray
    Fa: np.ndarray
    sep: dict
    learnt: dict
    inventory_match: int
    n_pairs: int
    placement_rejects: int


def run_audit(cfg: Config, ocfg: OverlapConfig, n_trials: int = 40, seed: int = 606,
              n_perm: int = 20000, verbose: bool = True) -> dict:
    """Independent seeds for the two classes: a listener meets independent scenes, and a paired
    audit would understate exactly the between-trial variance a yes/no criterion feeds on.

    The construction check is separate and DOES pair them, because there the point is that a
    present and an absent trial of one seed hold the same tones."""
    check(cfg, ocfg)
    d = validate(cfg)
    ref = M.single_tone_reference(cfg, d, 40.0, 1.0)
    tag = zlib.crc32(ocfg.absent_class.encode()) & 0xFFFF
    cells: List[CellAudit] = []
    t0 = time.time()
    for (T, st) in ocfg.cells:
        if verbose:
            print(f"  T={T:g} step={st:g}: {n_trials} present + {n_trials} absent ...",
                  end="", flush=True)
        t1 = time.time()
        rows = {True: [], False: []}
        for present in (True, False):
            for j in range(n_trials):
                s = int(np.random.default_rng([seed, tag, int(T), int(st), int(present), j])
                        .integers(2 ** 31 - 1))
                iv = build_interval(cfg, ocfg, d, s, T, st, present)
                rows[present].append(features(cfg, ocfg, d, iv, render_interval(cfg, iv, d), ref))
        # paired construction check, separate seeds again
        match = rej = 0
        for j in range(min(n_trials, 20)):
            s = int(np.random.default_rng([seed, 0xC0, int(T), int(st), j]).integers(2 ** 31 - 1))
            try:
                a = build_interval(cfg, ocfg, d, s, T, st, True)
                b = build_interval(cfg, ocfg, d, s, T, st, False)
            except PlacementError:
                rej += 1
                continue
            match += int(inventory(a) == inventory(b))
        names = list(rows[True][0].keys())
        Fp = np.array([[r[k] for k in names] for r in rows[True]], float)
        Fa = np.array([[r[k] for k in names] for r in rows[False]], float)
        cells.append(CellAudit(float(T), float(st), names, Fp, Fa,
                               feature_separation(names, Fp, Fa, n_perm, seed + 1),
                               learnt_observer(Fp, Fa), match, min(n_trials, 20), rej))
        if verbose:
            print(f" {time.time() - t1:.0f}s")
    names = cells[0].names
    Fp = np.vstack([c.Fp for c in cells]); Fa = np.vstack([c.Fa for c in cells])
    hits = sum(c.learnt["hit"] * c.Fp.shape[0] for c in cells)
    fas = sum(c.learnt["fa"] * c.Fa.shape[0] for c in cells)
    np_, na_ = Fp.shape[0], Fa.shape[0]
    return dict(cfg=cfg, ocfg=ocfg, d=d, n_trials=n_trials, seed=seed, cells=cells, names=names,
                Fp=Fp, Fa=Fa, pooled=feature_separation(names, Fp, Fa, n_perm, seed + 2),
                pooled_learnt=dict(hit=hits / np_, fa=fas / na_, n=np_ + na_,
                                   dprime=dprime_yesno(hits / np_, fas / na_, np_, na_),
                                   pc=(hits + (na_ - fas)) / (np_ + na_)),
                elapsed_s=time.time() - t0)


HEADLINE = [("rms_db", "long-term RMS (dB)"), ("peak", "peak amplitude"),
            ("on_time_ms", "total tone-time in the scene (ms)"),
            ("count_mean", "tones sounding, mean"), ("count_max", "tones sounding, max"),
            ("env:mod_depth", "envelope modulation depth"),
            ("env:mod_0.5_3Hz", "modulation power 0.5-3 Hz (dB)"),
            ("env:mod_3_10Hz", "modulation power 3-10 Hz (dB)"),
            ("env:n_bursts_3sd", "envelope peaks above 3 SD"),
            ("env:burst_height_max", "tallest burst (SD)"),
            ("env:ac_peak_iei", "envelope autocorrelation at recurrence lags"),
            ("spec_peakedness", "spectral peakedness (dB)"),
            ("ch:ioi_cv:max", "most irregular channel (IOI CV)"),
            ("ch:pairs_iei_norm:max", "strongest single-channel periodicity")]


def report(res: dict) -> str:
    cfg, ocfg, d = res["cfg"], res["ocfg"], res["d"]
    W = 100
    L = ["=" * W, f"TEMPORAL OVERLAP PILOT -- acoustic cue audit   absent class "
         f"'{ocfg.absent_class}'   {res['n_trials']} present + {res['n_trials']} absent per cell",
         "=" * W,
         "A yes/no listener answers from ONE scene, so any property whose distribution differs",
         "between the classes -- in mean OR in spread -- is a usable criterion whether or not",
         "anyone hears a pattern. This section describes alternative explanations; it does not",
         "recommend replacing the task.", ""]

    L.append("[1] the seven cells, and what 'overlap' means in each")
    L.append(f"    {'T':>4}{'step':>6}{'adjacent':>10}{'fraction':>10}{'common(7)':>11}"
             f"{'extent':>8}{'gap':>6}{'envelope':>10}{'env frac':>10}")
    for (T, st) in ocfg.cells:
        g = geometry(cfg, T, st, cfg.n_components)
        L.append(f"    {T:>4.0f}{st:>6.0f}{g['adjacent_overlap_ms']:>9.0f}m"
                 f"{g['adjacent_overlap_fraction']:>10.2f}{g['common_overlap_ms']:>10.0f}m"
                 f"{g['total_extent_ms']:>8.0f}{g['gap_ms']:>6.0f}"
                 f"{g['envelope_overlap_ms']:>9.2f}m{g['envelope_overlap_fraction']:>10.3f}")
    L.append("    'adjacent' is what two CONSECUTIVE components share; 'common(7)' is what all")
    L.append("    seven share, and it is zero everywhere except the synchronous cells. 'envelope'")
    L.append("    weights the overlap by the raised-cosine gates (5 ms ramps at both durations),")
    L.append("    so the same geometric overlap is worth less when it falls on a ramp.")
    L.append("")

    L.append("[2] construction: do a present and an absent trial of one seed hold the same tones?")
    L.append(f"    {'cell':>12}{'inventory match':>18}{'placement rejects':>20}")
    for c in res["cells"]:
        L.append(f"    T{c.tone_ms:g} d{c.step_ms:g}".rjust(16)
                 + f"{c.inventory_match}/{c.n_pairs}".rjust(14) + f"{c.placement_rejects:>20}")
    L.append("    matched per (frequency, duration), not on the total count.")
    if ocfg.absent_class == "roving":
        L.append("    NOTE: 'roving' puts its groups on fresh channels, so per-frequency matching")
        L.append("    does NOT hold for it. The fully matched roving contrast is the asynchrony")
        L.append("    task (asynchrony_rising_config.json), not this pilot.")
    L.append("")

    idx = {n: i for i, n in enumerate(res["names"])}
    sep = res["pooled"]
    dp = {n: v for n, v in zip(sep["names"], sep["dprime"])}
    L.append("[3] the cues a listener could use instead, pooled over the seven cells")
    L.append(f"    {'feature':<40}{'present':>18}{'absent':>18}{'yes/no d':>10}")
    for key, lab in HEADLINE:
        if key not in idx:
            continue
        p, a = res["Fp"][:, idx[key]], res["Fa"][:, idx[key]]
        L.append(f"    {lab:<40}{p.mean():>10.3f}+-{p.std():<6.3f}{a.mean():>10.3f}"
                 f"+-{a.std():<6.3f}{dp.get(key, 0.0):>+10.2f}")
    L.append("")

    order = np.argsort(-sep["dprime"])
    L.append(f"[4] every one of the {sep['n_features']} features, largest first, and whether ANY")
    L.append(f"    separates the classes (max-statistic permutation over class labels)")
    for i in order[:10]:
        L.append(f"    {sep['names'][i]:<40}{sep['dprime'][i]:>8.2f}{sep['auc'][i]:>8.3f}"
                 f"{sep['kind'][i]:>14}")
    L.append(f"    largest observed {sep['observed_max']:.3f}   95th percentile under "
             f"relabelling {sep['null_q95']:.3f}   p = {sep['p_value']:.3f}")
    L.append("")

    pl = res["pooled_learnt"]
    L.append("[5] an observer that has LEARNT the best linear criterion over every feature")
    L.append(f"    (leave-one-out, fitted within each cell, held-out decisions pooled)")
    L.append(f"    hit {pl['hit']:.3f}   false alarm {pl['fa']:.3f}   d' = {pl['dprime']:+.3f}   "
             f"{pl['pc'] * 100:.1f}% correct")
    L.append("")

    n_ = res["n_trials"]
    sd0 = float(np.sqrt(2 * 0.25 / n_) / stats.norm.pdf(0.0))
    L.append("[6] per cell")
    L.append(f"    a null cell d' has SD {sd0:.2f} at {n_}+{n_} trials; the largest of "
             f"{len(res['cells'])} cells sits near "
             f"{stats.norm.ppf(1 - 1 / (2 * len(res['cells']))) * sd0:+.2f} with nothing wrong.")
    L.append(f"    {'cell':>12}{'permutation p':>16}{'largest d':>12}{'learnt d':>11}{'hit/fa':>14}")
    for c in res["cells"]:
        L.append(f"    T{c.tone_ms:g} d{c.step_ms:g}".rjust(16)
                 + f"{c.sep['p_value']:>12.3f}{c.sep['observed_max']:>12.2f}"
                 f"{c.learnt['dprime']:>+11.2f}{c.learnt['hit']:>7.2f}/{c.learnt['fa']:<6.2f}")
    L.append("")
    for n in notes(cfg, ocfg):
        L.append(f"    note: {n}")
    L.append(f"built in {res['elapsed_s']:.0f}s; config {cfg.hash()}; overlap {ocfg.hash()}")
    L.append("=" * W)
    return "\n".join(L)


# ----------------------------------------------------------------------------
# design
# ----------------------------------------------------------------------------
def cell_name(tone_ms: float, step_ms: float) -> str:
    return f"T{tone_ms:g}_d{step_ms:g}"


def parse_cell(name: str) -> Tuple[float, float]:
    t, s = name.split("_d")
    return float(t[1:]), float(s)


def duration_estimate(cfg: Config, ocfg: OverlapConfig) -> dict:
    n_main = 2 * ocfg.trials_per_cell * len(ocfg.cells)
    trial_s = ((cfg.interval_dur_ms + cfg.lead_silence_ms) / 1000.0
               + min(ocfg.response_timeout_s, cfg.response_allowance_s) + cfg.iti_s)
    n_fb = n_main if ocfg.feedback_main else min(max(ocfg.feedback_main_first, 0), n_main)
    main_extra = n_fb * cfg.feedback_s
    prac_s = ocfg.practice_trials * (trial_s + cfg.feedback_s)
    n_breaks = max(0, n_main // max(cfg.break_every, 1) - 1) + 1
    total = (cfg.setup_minutes * 60.0 + n_main * trial_s + main_extra + prac_s
             + n_breaks * cfg.break_s)
    return dict(n_cells=len(ocfg.cells), n_main=n_main, n_practice=ocfg.practice_trials,
                trial_s=trial_s, n_breaks=n_breaks, minutes=total / 60.0)


@dataclass
class OverlapSpec:
    index: int
    block: str
    tone_ms: float
    step_ms: float
    present: bool
    seed: int


def make_design(cfg: Config, ocfg: OverlapConfig, code: str, session_index: int) -> dict:
    """Conditions randomised and classes balanced within every cell."""
    from .asynchrony import _shuffle
    from .design import session_seed
    check(cfg, ocfg)
    seed = session_seed(code, session_index)
    rng = np.random.default_rng(seed)
    items = [(t, s, p) for (t, s) in ocfg.cells for p in (True, False)
             for _ in range(ocfg.trials_per_cell)]
    items = _shuffle(rng, items, [(lambda x: x[2], ocfg.max_class_run),
                                  (lambda x: x, ocfg.max_cell_run)])
    main = [OverlapSpec(i, "main", float(t), float(s), bool(p), int(rng.integers(1, 2 ** 31 - 1)))
            for i, (t, s, p) in enumerate(items)]
    # practice: the synchronous cell at EVERY duration, so both are demonstrated
    refs = [ocfg.reference(t) or (t, 0.0) for t in ocfg.durations]
    per = max(ocfg.practice_trials // (2 * len(refs)), 1)
    pit = [(t, s, p) for (t, s) in refs for p in (True, False) for _ in range(per)]
    pit = _shuffle(rng, pit, [(lambda x: x[2], ocfg.max_class_run)])
    practice = [OverlapSpec(i, "practice", float(t), float(s), bool(p),
                            int(rng.integers(1, 2 ** 31 - 1))) for i, (t, s, p) in enumerate(pit)]
    dz = {"task": "overlap", "participant_code": code, "session_index": int(session_index),
          "session_seed": seed, "config_hash": cfg.hash(), "overlap_hash": ocfg.hash(),
          "absent": ocfg.absent_class, "cells": [list(c) for c in ocfg.cells],
          "practice": [vars(t) for t in practice], "main": [vars(t) for t in main]}
    dz["design_hash"] = hashlib.sha256(json.dumps(
        {k: dz[k] for k in ("session_seed", "config_hash", "overlap_hash", "absent",
                            "practice", "main")},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    return dz


# ----------------------------------------------------------------------------
# the session
# ----------------------------------------------------------------------------
QUESTION = "  Did you hear a recurring sound pattern in the background? [y/n]  "
BRIEF = ("Each trial is one sound: a cloud of short tones, about five seconds long.\n\n"
         "On half of the trials a small group of tones recurs inside it -- the same few\n"
         "pitches, coming back again and again through the scene. On the other half the\n"
         "same tones are there but scattered, with no recurring pattern.\n\n"
         "Answer 'y' if you heard a recurring pattern and 'n' if you did not.\n"
         "The tones are shorter on some trials than others; that is part of the design.")


def _getkey_timeout(valid: set, prompt: str, timeout_s: Optional[float]) -> Optional[str]:
    """A keypress, or None if `timeout_s` elapses first. None is a miss, not a 'no'."""
    import select
    import sys
    if prompt:
        print(prompt, end="", flush=True)
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        return (line.strip()[:1] or None) if line else None
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        deadline = None if timeout_s is None else time.time() + timeout_s
        while True:
            rem = None if deadline is None else deadline - time.time()
            if rem is not None and rem <= 0:
                return None
            r, _, _ = select.select([sys.stdin], [], [], rem)
            if not r:
                return None
            ch = sys.stdin.read(1)
            if ch in valid:
                return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class OverlapRunner:
    """Single-interval yes/no over the seven cells.

    Response time is measured from the moment the sound STOPS, because audio.play blocks: a
    clock started before it would carry the whole 4.9 s scene inside every RT. The reference is
    recorded in session.json so the number is interpretable later.
    """
    RT_REFERENCE = "stimulus offset (audio playback returns)"

    def __init__(self, cfg: Config, ocfg: OverlapConfig, data_dir, device=None,
                 audio: bool = True, auto: Optional[float] = None):
        from .runner import Audio
        check(cfg, ocfg)
        self.cfg, self.ocfg, self.d = cfg, ocfg, validate(cfg)
        self.data_dir = Path(data_dir)
        self.audio = Audio(cfg.sample_rate, device, enabled=audio and auto is None)
        self.auto = auto
        self.rng_auto = np.random.default_rng(23)

    def pause(self, msg: str) -> None:
        from .runner import getkey
        print("\n" + msg)
        if self.auto is None:
            getkey({" "}, "  press space to go on  ")

    def _answer(self, spec: OverlapSpec) -> Optional[str]:
        if self.auto is not None:
            g = geometry(self.cfg, spec.tone_ms, spec.step_ms, self.cfg.n_components)
            dp = 2.2 * float(np.exp(-g["total_extent_ms"] / max(self.auto, 1e-6)))
            x = self.rng_auto.normal(dp if spec.present else 0.0, 1.0)
            return "y" if x > 0.5 else "n"
        return _getkey_timeout({"y", "n", "q"}, QUESTION, self.ocfg.response_timeout_s)

    def _trial(self, spec: OverlapSpec, feedback: bool, i: int, n: int) -> Optional[bool]:
        from .runner import QuitRequested
        from .session import now_iso
        iv = build_interval(self.cfg, self.ocfg, self.d, spec.seed, spec.tone_ms, spec.step_ms,
                            spec.present)
        x = render(self.cfg, self.ocfg, self.d, iv)
        print(f"  trial {i}/{n} ...", end="", flush=True)
        if self.auto is None:
            self.audio.play(x)
        t_off = now_iso()
        t0 = time.time()                      # RT reference: the sound has finished
        k = self._answer(spec)
        rt = (time.time() - t0) * 1000.0
        if k == "q":
            raise QuitRequested()
        timeout = k is None
        correct = None if timeout else int((k == "y") == spec.present)
        if timeout:
            print("   no response")
        elif feedback:
            print("   correct" if correct else
                  f"   wrong ({'a pattern recurred' if spec.present else 'nothing recurred'})")
        else:
            print("   ok")
        # practice_round is otherwise unused here, so it carries whether THIS trial had feedback;
        # that keeps the CSV schema identical to every other task's and still records it per trial
        self.log.write(dict(trial_index=spec.index, block=spec.block,
                            practice_round=1 if feedback else 0,
                            practice_stage=0, variant=cell_name(spec.tone_ms, spec.step_ms),
                            step_ms=spec.step_ms,
                            target_position=1 if spec.present else 2, seed=spec.seed,
                            response="t" if timeout else k,
                            correct="" if timeout else correct, rt_ms=round(rt, 1),
                            t_start=t_off, t_response=now_iso()))
        return None if timeout else bool(correct)

    def run(self, code: Optional[str] = None, session_index: Optional[int] = None):
        from .runner import QuitRequested, Runner
        from .session import (TrialLog, next_session_index, provenance, session_dir,
                              upsert_participant, write_json)
        cfg, ocfg = self.cfg, self.ocfg
        if self.auto is not None:
            code = code or "AUTO"
            row = dict(code=code, age="0", sex="na", handedness="na", hearing="simulated",
                       musical_training_years="0", headphones="none", experimenter="auto",
                       consent="yes")
        else:
            row = Runner.panel(self, code)
            code = row["code"]
        upsert_participant(self.data_dir, row)
        idx = session_index or next_session_index(self.data_dir, code)
        sdir = session_dir(self.data_dir, code, idx)
        if sdir.exists():
            raise RuntimeError(f"{sdir} exists; choose another session index")
        sdir.mkdir(parents=True)
        design = make_design(cfg, ocfg, code, idx)
        meta = {**provenance(cfg), "task": "overlap", "overlap": ocfg.to_dict(),
                "overlap_hash": ocfg.hash(), "absent": ocfg.absent_class,
                "rt_reference": self.RT_REFERENCE,
                "response_timeout_s": ocfg.response_timeout_s,
                "feedback_main": bool(ocfg.feedback_main),
                "feedback_main_first": int(ocfg.feedback_main_first),
                "geometry": [geometry(cfg, t, s, cfg.n_components) for t, s in ocfg.cells],
                "participant_code": code, "session_index": idx,
                "session_seed": design["session_seed"], "design_hash": design["design_hash"],
                "design": design, "status": "started"}
        write_json(sdir / "session.json", meta)
        print(f"new overlap session {sdir} (absent='{ocfg.absent_class}', "
              f"design {design['design_hash']})")
        self.sdir, self.meta = sdir, meta
        Runner.calibrate(self)
        self.log = TrialLog(sdir / "trials.csv")
        try:
            self.pause("Practice, with feedback.\n\n" + BRIEF)
            pr = [OverlapSpec(**t) for t in design["practice"]]
            for i, s in enumerate(pr, 1):
                self._trial(s, True, i, len(pr))
            mn = [OverlapSpec(**t) for t in design["main"]]
            self.pause(f"Main block: {len(mn)} trials, no feedback. Same question each time.\n"
                       f"If you do not answer within {ocfg.response_timeout_s:.0f} s the trial is "
                       f"recorded as no response and moves on.")
            n_fb = len(mn) if ocfg.feedback_main else max(0, int(ocfg.feedback_main_first))
            if n_fb:
                print(f"\n  feedback is on for {'every trial' if n_fb >= len(mn) else f'the first {n_fb} trials'}"
                      f" of the main block; this is recorded in session.json and the analysis says so.")
            for i, s in enumerate(mn, 1):
                if i > 1 and (i - 1) % cfg.break_every == 0:
                    self.pause("Take a break.")
                self._trial(s, i <= n_fb, i, len(mn))
            meta["status"] = "complete"
        except QuitRequested:
            meta["status"] = "interrupted"
            print("\nstopped.")
        write_json(sdir / "session.json", meta)
        self.log.close()
        print(analyse([sdir]))
        return sdir


# ----------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------
def _counts(rows: Sequence[dict]) -> Tuple[int, int, int, int, int]:
    """(hits, n_present_answered, false alarms, n_absent_answered, no-responses)."""
    ans = [r for r in rows if r["response"] in ("y", "n")]
    miss = len(rows) - len(ans)
    sig = [r for r in ans if str(r["target_position"]) == "1"]
    noi = [r for r in ans if str(r["target_position"]) == "2"]
    return (sum(1 for r in sig if r["response"] == "y"), len(sig),
            sum(1 for r in noi if r["response"] == "y"), len(noi), miss)


def _dp(c) -> float:
    h, ns, f, nn, _ = c
    return dprime_yesno(h / ns, f / nn, ns, nn) if ns and nn else float("nan")


def _boot(counts: Dict[str, tuple], n_boot: int, seed: int = 11) -> List[Dict[str, tuple]]:
    """One joint resample per replicate, so contrasts inherit the right correlation."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        rep = {}
        for k, (h, ns, f, nn, mi) in counts.items():
            rep[k] = (int(rng.binomial(ns, h / ns)) if ns else 0, ns,
                      int(rng.binomial(nn, f / nn)) if nn else 0, nn, mi)
        out.append(rep)
    return out


def _ci(vals: np.ndarray) -> str:
    v = vals[np.isfinite(vals)]
    return f"[{np.quantile(v, .025):+.2f}, {np.quantile(v, .975):+.2f}]" if v.size > 20 else ""


def analyse(sessions: Sequence, n_boot: Optional[int] = None) -> str:
    import csv as _csv
    from .session import read_json
    rows: List[dict] = []
    metas: List[dict] = []
    for sdir in sessions:
        p = Path(sdir)
        with open(p / "trials.csv", newline="") as f:
            rows += [r for r in _csv.DictReader(f) if r["block"] == "main"]
        if (p / "session.json").exists():
            metas.append(read_json(p / "session.json"))
    L = ["", "=" * 96, "TEMPORAL OVERLAP PILOT", "=" * 96]
    if not rows:
        return "\n".join(L + ["no main-block trials recorded."])
    meta = metas[0] if metas else {}
    ocfg = OverlapConfig.from_dict(meta["overlap"]) if "overlap" in meta else OverlapConfig()
    cfg = Config.from_dict(meta["config"]) if "config" in meta else None
    n_boot = ocfg.n_boot if n_boot is None else n_boot
    cells = [(t, s) for t, s in ocfg.cells
             if any(r["variant"] == cell_name(t, s) for r in rows)]
    counts = {cell_name(t, s): _counts([r for r in rows if r["variant"] == cell_name(t, s)])
              for t, s in cells}
    tot = _counts(rows)
    L.append(f"\n{len(rows)} main trials, {len(metas)} session(s), absent class "
             f"'{meta.get('absent', '?')}'")
    L.append(f"  hits {tot[0]}/{tot[1]}   false alarms {tot[2]}/{tot[3]}   "
             f"no response {tot[4]}   d' = {_dp(tot):+.2f}")
    L.append(f"  response time measured from {meta.get('rt_reference', 'an unrecorded reference')}")
    L.append("  no-responses are counted separately and are NOT folded into 'no'.")
    fb_all, fb_n = bool(meta.get("feedback_main")), int(meta.get("feedback_main_first") or 0)
    n_fb_seen = sum(1 for r in rows if str(r.get("practice_round")) == "1")
    if fb_all or fb_n or n_fb_seen:
        L.append(f"  FEEDBACK WAS ON for {n_fb_seen} of {len(rows)} main trials. d' tolerates that;")
        L.append("  the criterion does not -- c below describes the criterion the feedback drove the")
        L.append("  listener to, not one they chose. Read the half-split in [4] for drift, and do not")
        L.append("  pool these sessions with feedback-free ones without saying so.")
    else:
        L.append("  no feedback in the main block.")

    boots = _boot(counts, n_boot)
    L.append("\n[1] per cell")
    L.append(f"  {'cell':>10}{'adj ov':>8}{'frac':>7}{'hits':>9}{'false al':>11}{'acc':>7}"
             f"{'d':>8}{'95% CI':>18}{'c':>7}{'miss':>6}")
    for t, s in cells:
        k = cell_name(t, s); c = counts[k]
        g = geometry(cfg, t, s, cfg.n_components) if cfg else dict(adjacent_overlap_ms=max(0, t - s),
                                                                   adjacent_overlap_fraction=max(0, t - s) / t)
        bs = np.array([_dp(b[k]) for b in boots])
        acc = (c[0] + (c[3] - c[2])) / max(c[1] + c[3], 1)
        cr = criterion_yesno(c[0] / c[1], c[2] / c[3], c[1], c[3]) if c[1] and c[3] else float("nan")
        L.append(f"  {k:>10}{g['adjacent_overlap_ms']:>7.0f}m{g['adjacent_overlap_fraction']:>7.2f}"
                 f"{c[0]:>5}/{c[1]:<3}{c[2]:>7}/{c[3]:<3}{acc:>7.2f}{_dp(c):>+8.2f}"
                 f"{_ci(bs):>18}{cr:>+7.2f}{c[4]:>6}")
    L.append("  d' uses the log-linear correction (Hautus 1995), so 0 and 1 stay finite;")
    L.append("  intervals are a joint bootstrap over every cell's counts.")

    L.append("\n[2] the three comparisons chosen before the data")
    for label, a, b in PLANNED:
        ka, kb = cell_name(*a), cell_name(*b)
        if ka not in counts or kb not in counts:
            continue
        obs = _dp(counts[ka]) - _dp(counts[kb])
        bs = np.array([_dp(x[ka]) - _dp(x[kb]) for x in boots])
        p2 = 2 * min(np.mean(bs <= 0), np.mean(bs >= 0))
        L.append(f"  {label:<24}{ka} - {kb}   d' difference {obs:+.2f}  {_ci(bs)}  p = {p2:.3f}")
    L.append("  A difference near zero is not evidence of equivalence: with this sample the")
    L.append("  interval is wide, and an interval that contains zero also contains effects")
    L.append("  worth having. Read the interval, not the p.")

    L.append("\n[3] each asynchronous cell against the synchronous one at its OWN duration")
    L.append("  (exploratory: a baseline-adjusted difference does not isolate overlap, because")
    L.append("   the reference differs from the cell in step, extent and common overlap at once)")
    for t in sorted({t for t, _ in cells}):
        ref = ocfg.reference(t)
        if ref is None or cell_name(*ref) not in counts:
            continue
        kr = cell_name(*ref)
        for tt, s in [c for c in cells if c[0] == t and c[1] != 0.0]:
            k = cell_name(tt, s)
            bs = np.array([_dp(x[k]) - _dp(x[kr]) for x in boots])
            L.append(f"  {k:>10} - {kr:<10}{_dp(counts[k]) - _dp(counts[kr]):>+8.2f}  {_ci(bs)}")

    L.append("\n[4] how to read this, and how not to")
    L.append("  * step, component duration and overlap are related by overlap = max(0, T - step),")
    L.append("    so they cannot be entered as three independent predictors of anything. Any")
    L.append("    model here must fix one and vary the others.")
    L.append("  * this is a feasibility sample. It is not powered to separate an absolute-overlap")
    L.append("    account from a proportional one, and the ladder is not a threshold: nothing")
    L.append("    here should be fitted with a monotonic psychometric function and read off.")
    L.append("  * the longer component carries more energy at equal amplitude. The synchronous")
    L.append("    cell at each duration is where that would show up; it does not remove it.")
    half = len(rows) // 2
    for name, rs in (("first half", rows[:half]), ("second half", rows[half:])):
        c2 = _counts(rs)
        if c2[1] and c2[3]:
            L.append(f"  {name:<12} d' = {_dp(c2):+.2f}   "
                     f"c = {criterion_yesno(c2[0] / c2[1], c2[2] / c2[3], c2[1], c2[3]):+.2f}")
    rts = [float(r["rt_ms"]) for r in rows if r["rt_ms"] and r["response"] in ("y", "n")]
    if rts:
        L.append(f"  median response time {np.median(rts):.0f} ms after the sound ended")
    L.append("=" * 96)
    return "\n".join(L)


def condition_table(cfg: Config, ocfg: OverlapConfig) -> str:
    L = [f"{'cell':>10}{'T':>5}{'step':>6}{'adjacent':>10}{'fraction':>10}{'common(7)':>11}"
         f"{'extent':>8}{'gap':>6}{'env ms':>9}{'env frac':>10}"]
    for t, s in ocfg.cells:
        g = geometry(cfg, t, s, cfg.n_components)
        L.append(f"{cell_name(t, s):>10}{t:>5.0f}{s:>6.0f}{g['adjacent_overlap_ms']:>9.0f}m"
                 f"{g['adjacent_overlap_fraction']:>10.2f}{g['common_overlap_ms']:>10.0f}m"
                 f"{g['total_extent_ms']:>8.0f}{g['gap_ms']:>6.0f}"
                 f"{g['envelope_overlap_ms']:>9.2f}{g['envelope_overlap_fraction']:>10.3f}")
    return "\n".join(L)
