"""Every parameter of the experiment, and the validator that refuses impossible ones.

Nothing numerical is decided anywhere else. Other modules ask for
``cfg.<name>`` or for a value in ``Derived`` (which is computed *from* the
config, never chosen independently).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field, fields
from typing import Any, Tuple

import numpy as np

from . import pool as _pool


class ConfigError(ValueError):
    """Raised by :func:`validate` with a message that says what to change."""


VARIANTS = ("rising", "scrambled", "redrawn", "ungrouped", "onechannel")
# rising     : main task. Element = rising staircase of N components, step apart.
#              Interval A recurs on one channel set; interval B redraws it each element.
# scrambled  : same asynchronies, fixed random order of the delays (same order in both
#              intervals, same order for every element). A recurs, B redraws channels.
# redrawn    : same channels recur in A, but the delay order is redrawn every element
#              (in both intervals). Tests whether the recurring *pattern* matters.
# ungrouped  : A as in rising; the comparison interval has the same channels at the
#              same rate but its figure tones are scattered in time, never grouped.
#              Asks "which binds" rather than "which recurs". Its onset envelope
#              differs between intervals by construction; see README.
# onechannel : ONE channel recurs at the element times (no grouping possible) against a
#              plain background. Measures the single-channel periodicity cue on its own.


@dataclass(frozen=True)
class Config:
    # ---- audio ---------------------------------------------------------------
    sample_rate: int = 48000
    grid_ms: float = 1.0                 # onset-time resolution of the schedule

    # ---- tone pool -----------------------------------------------------------
    pool_low_hz: float = 250.0           # lowest channel
    pool_high_hz: float = 6000.0         # no channel above this
    pool_spacing_erb: float = 1.0        # channel spacing in ERB units (critical-band rule)
    min_beat_rate_hz: float = 40.0       # adjacent channels must beat faster than this ("throb" rule)

    # ---- tones ---------------------------------------------------------------
    tone_dur_ms: float = 30.0
    ramp_ms: float = 5.0                 # raised-cosine onset and offset ramps
    tone_amplitude: float = 0.031        # linear peak amplitude of one tone, all channels equal

    # ---- background ----------------------------------------------------------
    tones_per_channel: int = 31          # fixed budget per channel per interval (figure tones included)
    interval_dur_ms: float = 2250.0

    # ---- figure --------------------------------------------------------------
    n_components: int = 7
    figure_min_spacing_channels: int = 2 # components of one element at least this many channels apart
    steps_ms: Tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0, 28.0)
    main_variants: Tuple[str, ...] = ("rising", "ungrouped")   # one psychometric function each
    n_elements: int = 6
    iei_min_ms: float = 200.0            # inter-element onset interval, drawn uniformly: 3-5 Hz
    iei_max_ms: float = 333.0
    lead_min_ms: float = 150.0           # first element onset, drawn uniformly
    lead_max_ms: float = 250.0
    tail_min_ms: float = 100.0           # guaranteed background after the last element ends
    max_shared_consecutive: int = 1      # redrawn sets: channels in common with the previous set
    max_shared_any: int = 2              # redrawn sets: channels in common with any earlier set

    # ---- trial ---------------------------------------------------------------
    isi_ms: float = 400.0
    lead_silence_ms: float = 50.0

    # ---- experiment ----------------------------------------------------------
    trials_per_condition: int = 14       # main block, per (variant, step) cell; must be even
    max_condition_run: int = 2           # consecutive trials sharing a (variant, step) cell
    max_variant_run: int = 4             # consecutive trials sharing a ladder
    practice_cells: Tuple[Tuple[str, float], ...] = (("ungrouped", 0.0), ("rising", 0.0))
    practice_n: int = 12
    practice_criterion: int = 10         # correct out of practice_n required to proceed
    practice_max_rounds: int = 2         # attempts allowed per practice stage
    break_every: int = 40
    feedback_main: bool = True
    feedback_control: bool = True
    control_cells: Tuple[Tuple[str, float], ...] = (
        ("scrambled", 15.0), ("scrambled", 28.0),
        ("redrawn", 15.0), ("redrawn", 28.0),
        ("onechannel", 0.0),
    )
    control_trials_per_cell: int = 8     # must be even
    max_session_minutes: float = 40.0
    response_allowance_s: float = 1.2    # for the duration estimate only
    feedback_s: float = 0.4
    iti_s: float = 0.5
    break_s: float = 45.0
    setup_minutes: float = 5.0

    # ---- calibration ---------------------------------------------------------
    tone_level_db_spl: float = 60.0      # intended SPL of ONE tone at the eardrum
    calibration_freq_hz: float = 1000.0
    calibration_dur_s: float = 5.0

    # ---- analysis ------------------------------------------------------------
    bootstrap_n: int = 1000
    threshold_pc: float = 0.75
    alpha: float = 0.05

    # ------------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["steps_ms"] = list(self.steps_ms)
        d["control_cells"] = [list(c) for c in self.control_cells]
        d["practice_cells"] = [list(c) for c in self.practice_cells]
        d["main_variants"] = list(self.main_variants)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ConfigError(f"unknown config keys: {sorted(unknown)}")
        kw = dict(d)
        if "steps_ms" in kw:
            kw["steps_ms"] = tuple(float(s) for s in kw["steps_ms"])
        if "control_cells" in kw:
            kw["control_cells"] = tuple((str(v), float(s)) for v, s in kw["control_cells"])
        if "practice_cells" in kw:
            kw["practice_cells"] = tuple((str(v), float(s)) for v, s in kw["practice_cells"])
        if "main_variants" in kw:
            kw["main_variants"] = tuple(str(v) for v in kw["main_variants"])
        return cls(**kw)

    def replace(self, **kw) -> "Config":
        return dataclasses.replace(self, **kw)

    def hash(self) -> str:
        """Stable hash of every parameter; part of the design hash used by resume."""
        s = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    # convenience -------------------------------------------------------------
    @property
    def n_grid(self) -> int:
        return int(round(self.interval_dur_ms / self.grid_ms))

    def ms_to_grid(self, ms: float) -> int:
        g = ms / self.grid_ms
        if abs(g - round(g)) > 1e-6:
            raise ConfigError(f"{ms} ms is not a multiple of grid_ms={self.grid_ms}")
        return int(round(g))

    def ms_to_samples(self, ms: float) -> int:
        return int(round(ms * self.sample_rate / 1000.0))


@dataclass(frozen=True)
class Derived:
    """Quantities computed from the config. Read them here; never re-derive elsewhere."""
    channel_freqs_hz: np.ndarray        # ascending
    n_channels: int
    tone_dur_grid: int
    spans_ms: Tuple[float, ...]         # element span per step, (N-1)*step + D
    max_span_ms: float
    schedule_max_ms: float              # worst-case end of the last element + tail
    occupancy_per_channel: float        # M*D/T
    mean_simultaneous: float            # P*M*D/T
    max_simultaneous_bound: int         # used for the clipping bound
    peak_bound: float                   # tone_amplitude * max_simultaneous_bound
    min_beat_rate_hz: float             # ERB width (Hz) * spacing at the lowest channel
    n_valid_figure_sets: int
    ladder: Tuple[dict, ...]            # per step: overlap fraction, max simultaneous components
    n_main_trials: int
    n_practice_trials: int
    n_control_trials: int
    main_cells: Tuple[Tuple[str, float], ...]
    trial_dur_s: float
    est_session_minutes: float


def _n_valid_sets(P: int, N: int, spacing: int) -> int:
    # sets of N channels out of P with consecutive gaps >= spacing
    m = P - (N - 1) * (spacing - 1)
    if m < N:
        return 0
    return math.comb(m, N)


def derive(cfg: Config) -> Derived:
    freqs = _pool.make_pool(cfg.pool_low_hz, cfg.pool_high_hz, cfg.pool_spacing_erb)
    P = len(freqs)
    N, D, T, M, K = cfg.n_components, cfg.tone_dur_ms, cfg.interval_dur_ms, cfg.tones_per_channel, cfg.n_elements
    spans = tuple((N - 1) * s + D for s in cfg.steps_ms)
    control_spans = tuple((N - 1) * s + D for _, s in tuple(cfg.control_cells) + tuple(cfg.practice_cells))
    max_span = max(spans + control_spans) if (spans or control_spans) else D
    sched_max = cfg.lead_max_ms + (K - 1) * cfg.iei_max_ms + max_span + cfg.tail_min_ms
    occ = M * D / T
    mean_sim = P * occ
    # background count is ~binomial(P, occ) at any instant; add the whole element on top
    max_sim_bound = int(min(P, math.ceil(mean_sim + 4.0 * math.sqrt(max(mean_sim, 1.0)))) + N)
    peak_bound = cfg.tone_amplitude * max_sim_bound
    erb_low = _pool.erb_width_hz(freqs[0]) if P else float("nan")
    ladder = []
    for s in cfg.steps_ms:
        overlap = max(0.0, 1.0 - s / D) if s > 0 else 1.0
        max_simul = N if s == 0 else min(N, int(math.ceil(D / s)) if D % s else int(D // s))
        ladder.append(dict(step_ms=s, adjacent_overlap=overlap, max_simultaneous=max_simul,
                           span_ms=(N - 1) * s + D, gap_ms=max(0.0, s - D)))
    main_cells = tuple((v, float(s)) for v in cfg.main_variants for s in cfg.steps_ms)
    n_main = cfg.trials_per_condition * len(main_cells)
    n_prac = cfg.practice_n * len(cfg.practice_cells)
    n_ctrl = cfg.control_trials_per_cell * len(cfg.control_cells)
    trial_dur = (2 * T + cfg.isi_ms + cfg.lead_silence_ms) / 1000.0 + cfg.response_allowance_s + cfg.feedback_s + cfg.iti_s
    n_trials = n_prac + n_main + n_ctrl
    n_breaks = (n_main + n_ctrl) // max(cfg.break_every, 1) + 2   # + block transitions
    est = cfg.setup_minutes + (n_trials * trial_dur + n_breaks * cfg.break_s) / 60.0
    return Derived(
        channel_freqs_hz=freqs, n_channels=P, tone_dur_grid=int(round(D / cfg.grid_ms)),
        spans_ms=spans, max_span_ms=max_span, schedule_max_ms=sched_max,
        occupancy_per_channel=occ, mean_simultaneous=mean_sim,
        max_simultaneous_bound=max_sim_bound, peak_bound=peak_bound,
        min_beat_rate_hz=erb_low * cfg.pool_spacing_erb,
        n_valid_figure_sets=_n_valid_sets(P, N, cfg.figure_min_spacing_channels),
        ladder=tuple(ladder), n_main_trials=n_main, n_practice_trials=n_prac, n_control_trials=n_ctrl,
        main_cells=main_cells,
        trial_dur_s=trial_dur, est_session_minutes=est,
    )


def validate(cfg: Config) -> Derived:
    """Return Derived, or raise ConfigError saying what to change."""
    errs = []
    d = derive(cfg)
    N, D, T, M, K = cfg.n_components, cfg.tone_dur_ms, cfg.interval_dur_ms, cfg.tones_per_channel, cfg.n_elements

    def grid_ok(ms, name):
        g = ms / cfg.grid_ms
        if abs(g - round(g)) > 1e-6:
            errs.append(f"{name}={ms} ms is not a multiple of grid_ms={cfg.grid_ms} ms")

    if cfg.sample_rate < 8000:
        errs.append("sample_rate must be >= 8000")
    if cfg.grid_ms <= 0:
        errs.append("grid_ms must be > 0")
    for name in ("tone_dur_ms", "interval_dur_ms", "iei_min_ms", "iei_max_ms", "lead_min_ms",
                 "lead_max_ms", "tail_min_ms", "isi_ms"):
        grid_ok(getattr(cfg, name), name)
    for s in cfg.steps_ms:
        grid_ok(s, "steps_ms entry")
    for _, s in cfg.control_cells:
        grid_ok(s, "control_cells step")

    # pool
    if d.n_channels < 2:
        errs.append("pool has fewer than 2 channels: widen pool_low_hz..pool_high_hz or reduce pool_spacing_erb")
    if d.min_beat_rate_hz < cfg.min_beat_rate_hz:
        errs.append(f"adjacent channels at the bottom of the pool beat at {d.min_beat_rate_hz:.1f} Hz "
                    f"< min_beat_rate_hz={cfg.min_beat_rate_hz}: raise pool_low_hz or pool_spacing_erb "
                    f"(this is the 'repeated beep')")
    if cfg.pool_high_hz * 2.2 > cfg.sample_rate:
        errs.append("pool_high_hz too close to Nyquist: raise sample_rate or lower pool_high_hz")
    if cfg.figure_min_spacing_channels < 1:
        errs.append("figure_min_spacing_channels must be >= 1")
    if d.n_valid_figure_sets < 50:
        errs.append(f"only {d.n_valid_figure_sets} valid figure sets: reduce n_components or "
                    f"figure_min_spacing_channels, or widen the pool")

    # tones
    if cfg.tone_dur_ms < 2 * cfg.ramp_ms:
        errs.append("tone_dur_ms must be at least 2*ramp_ms")
    if not (0 < cfg.tone_amplitude < 1):
        errs.append("tone_amplitude must be in (0,1)")
    if d.peak_bound > 0.99:
        errs.append(f"clipping risk: tone_amplitude*{d.max_simultaneous_bound} simultaneous tones = "
                    f"{d.peak_bound:.2f} > 0.99: lower tone_amplitude to <= {0.99 / d.max_simultaneous_bound:.3f} "
                    f"or reduce tones_per_channel")

    # background / budget
    if M < 2 * K:
        errs.append(f"tones_per_channel={M} < 2*n_elements={2 * K}: the recurring channels need {K} figure "
                    f"tones and at least as many background tones to swap against; raise tones_per_channel")
    if d.occupancy_per_channel > 0.6:
        errs.append(f"per-channel occupancy {d.occupancy_per_channel:.2f} > 0.6: tones cannot be packed "
                    f"without overlap; reduce tones_per_channel or tone_dur_ms, or raise interval_dur_ms")

    # figure schedule
    if N < 2:
        errs.append("n_components must be >= 2")
    if K < 2:
        errs.append("n_elements must be >= 2")
    if list(cfg.steps_ms) != sorted(set(cfg.steps_ms)):
        errs.append("steps_ms must be strictly increasing and unique")
    if cfg.steps_ms and cfg.steps_ms[0] != 0.0:
        errs.append("steps_ms must start at 0 (the synchronous, easiest condition anchors practice and the fit)")
    if cfg.iei_min_ms > cfg.iei_max_ms or cfg.lead_min_ms > cfg.lead_max_ms:
        errs.append("iei_min_ms<=iei_max_ms and lead_min_ms<=lead_max_ms required")
    if d.max_span_ms > cfg.iei_min_ms:
        s_max = (cfg.iei_min_ms - D) / (N - 1)
        errs.append(f"widest element ({d.max_span_ms:.0f} ms) is longer than iei_min_ms={cfg.iei_min_ms:.0f}: "
                    f"elements would run into each other. Raise iei_min_ms to >= {d.max_span_ms:.0f} or cap the "
                    f"largest step at {s_max:.0f} ms (or reduce n_components/tone_dur_ms)")
    if d.schedule_max_ms > T:
        errs.append(f"schedule does not fit: lead_max + (K-1)*iei_max + widest span + tail = "
                    f"{d.schedule_max_ms:.0f} ms > interval_dur_ms={T:.0f}. Raise interval_dur_ms to >= "
                    f"{d.schedule_max_ms:.0f} (the jitter is never clipped by a rejection rule)")
    if cfg.max_shared_consecutive > cfg.max_shared_any:
        errs.append("max_shared_consecutive must be <= max_shared_any")
    if cfg.max_shared_any < 0:
        errs.append("max_shared_any must be >= 0")
    # pairwise-overlap feasibility (union bound)
    if K * N - math.comb(K, 2) * cfg.max_shared_any > d.n_channels:
        errs.append(f"{K} redrawn sets of {N} channels with at most {cfg.max_shared_any} shared pairwise need "
                    f"more than {d.n_channels} channels; raise max_shared_any")

    # experiment
    if cfg.trials_per_condition % 2:
        errs.append("trials_per_condition must be even (target interval is balanced within condition)")
    if not cfg.main_variants:
        errs.append("main_variants must name at least one ladder")
    for v in cfg.main_variants:
        if v not in VARIANTS:
            errs.append(f"unknown main variant {v!r}; choose from {VARIANTS}")
    if "rising" not in cfg.main_variants:
        errs.append("main_variants must include 'rising': it is the spectrally matched ladder that carries "
                    "the inference, and the battery's clean-ladder checks are defined on it")
    if len(set(cfg.main_variants)) != len(cfg.main_variants):
        errs.append("main_variants must be unique")
    if not cfg.practice_cells:
        errs.append("practice_cells must name at least one stage")
    for v, s in cfg.practice_cells:
        if v not in VARIANTS:
            errs.append(f"unknown practice variant {v!r}; choose from {VARIANTS}")
        if v in cfg.main_variants and s != cfg.steps_ms[0]:
            errs.append(f"practice stage ({v}, {s}) is not the easiest step of its ladder ({cfg.steps_ms[0]}): "
                        f"practice must be run where the task is easiest")
    if cfg.max_variant_run < 1:
        errs.append("max_variant_run must be >= 1")
    if len(cfg.main_variants) > 1 and cfg.max_variant_run > 6:
        errs.append("max_variant_run > 6 lets one ladder run long enough for the listener to notice which "
                    "foil is in play; lower it")
    for v, s in tuple(cfg.control_cells) + tuple(cfg.practice_cells):
        if (v, float(s)) in tuple((mv, float(ms)) for mv in cfg.main_variants for ms in cfg.steps_ms) and \
                (v, float(s)) in [(cv, float(cs)) for cv, cs in cfg.control_cells]:
            errs.append(f"control cell ({v}, {s:g}) duplicates a main-block cell; the main block already "
                        f"measures it, so remove it from control_cells")
    if cfg.control_trials_per_cell % 2:
        errs.append("control_trials_per_cell must be even")
    for v, s in cfg.control_cells:
        if v not in VARIANTS:
            errs.append(f"unknown control variant {v!r}; choose from {VARIANTS}")
        if s < 0:
            errs.append("control step must be >= 0")
    if cfg.practice_criterion > cfg.practice_n:
        errs.append("practice_criterion cannot exceed practice_n")
    if cfg.max_condition_run < 1:
        errs.append("max_condition_run must be >= 1")
    if d.est_session_minutes > cfg.max_session_minutes:
        errs.append(f"estimated session {d.est_session_minutes:.1f} min > max_session_minutes="
                    f"{cfg.max_session_minutes}: reduce trials_per_condition/control_trials_per_cell or "
                    f"interval_dur_ms")
    if cfg.threshold_pc <= 0.5 or cfg.threshold_pc >= 1:
        errs.append("threshold_pc must be in (0.5, 1)")

    if errs:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(errs))
    return d


DEFAULT = Config()


def describe(cfg: Config) -> str:
    d = validate(cfg)
    f = d.channel_freqs_hz
    lines = [
        f"config hash          {cfg.hash()}",
        f"channels             {d.n_channels} at {cfg.pool_spacing_erb} ERB spacing, "
        f"{f[0]:.0f}..{f[-1]:.0f} Hz; adjacent beat rate >= {d.min_beat_rate_hz:.0f} Hz",
        f"tone                 {cfg.tone_dur_ms:.0f} ms, {cfg.ramp_ms:.0f} ms ramps, amplitude {cfg.tone_amplitude}",
        f"budget               {cfg.tones_per_channel} tones/channel/interval -> occupancy "
        f"{d.occupancy_per_channel:.2f}, mean simultaneous {d.mean_simultaneous:.1f}",
        f"interval             {cfg.interval_dur_ms:.0f} ms, {cfg.n_elements} elements, IEI "
        f"U[{cfg.iei_min_ms:.0f},{cfg.iei_max_ms:.0f}] ms, lead U[{cfg.lead_min_ms:.0f},{cfg.lead_max_ms:.0f}]",
        f"schedule worst case  {d.schedule_max_ms:.0f} ms of {cfg.interval_dur_ms:.0f}",
        f"figure               {cfg.n_components} components, >= {cfg.figure_min_spacing_channels} channels apart, "
        f"{d.n_valid_figure_sets} valid sets",
        f"timing floor         {cfg.grid_ms} ms grid; ramps {cfg.ramp_ms} ms bound the perceptual floor",
        f"clipping bound       {d.peak_bound:.2f} of full scale",
        "ladder (step ms, adjacent overlap, max simultaneous components, span ms, gap ms):",
    ]
    for r in d.ladder:
        lines.append(f"   {r['step_ms']:6.1f}   {r['adjacent_overlap']:.2f}   {r['max_simultaneous']}   "
                     f"{r['span_ms']:.0f}   {r['gap_ms']:.0f}")
    lines.append(f"ladders              {len(cfg.main_variants)}: " +
                 ", ".join(f"{v} x {len(cfg.steps_ms)} steps x {cfg.trials_per_condition} trials"
                           for v in cfg.main_variants))
    lines.append(f"practice stages      " + ", ".join(f"{v} {s:g} ms" for v, s in cfg.practice_cells) +
                 f" ({cfg.practice_n} trials each, criterion {cfg.practice_criterion})")
    lines.append(f"trials               practice {d.n_practice_trials}, main {d.n_main_trials}, "
                 f"control {d.n_control_trials}")
    lines.append(f"estimated session    {d.est_session_minutes:.1f} min (limit {cfg.max_session_minutes})")
    return "\n".join(lines)
