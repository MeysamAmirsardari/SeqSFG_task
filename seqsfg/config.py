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
from typing import Any, Optional, Tuple

import numpy as np

from . import pool as _pool


class ConfigError(ValueError):
    """Raised by :func:`validate` with a message that says what to change."""


VARIANTS = ("rising", "scrambled", "redrawn", "ungrouped", "onechannel", "scattered")
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
# scattered  : the LOAD-BEARING control for a dense stream. The target's channels recur exactly
#              as in 'rising', but each component is placed at a random time inside the element
#              window, so the components never group. The foil redraws channels and is scattered
#              the same way. The ONLY thing separating the intervals is channel recurrence, with
#              binding removed. A listener above chance here is using single-channel periodicity
#              rather than grouping, which bounds how much of the main result binding explains.


@dataclass(frozen=True)
class Config:
    # ---- audio ---------------------------------------------------------------
    sample_rate: int = 48000
    grid_ms: float = 1.0                 # onset-time resolution of the schedule

    # ---- tone pool -----------------------------------------------------------
    pool_low_hz: float = 200.0           # lowest channel
    pool_high_hz: float = 10000.0        # no channel above this
    pool_spacing_erb: float = 1.0        # channel spacing in ERB units (critical-band rule)
    min_beat_rate_hz: float = 40.0       # adjacent channels must beat faster than this ("throb" rule)

    # ---- tones ---------------------------------------------------------------
    tone_dur_ms: float = 45.0
    ramp_ms: float = 5.0                 # raised-cosine onset and offset ramps
    tone_amplitude: float = 0.028        # linear peak amplitude of one tone, all channels equal

    # ---- background ----------------------------------------------------------
    tones_per_channel: int = 20          # fixed budget per channel per interval (figure tones included)
    interval_dur_ms: float = 4000.0

    # ---- figure --------------------------------------------------------------
    n_components: int = 7
    figure_repeats: int = 1              # tone-slots each component occupies: the element's DURATION.
                                         # 1 = a single pip (isolated blip); >1 = a sustained figure, as in
                                         # the published stimulus where the figure spans consecutive chords.
    figure_min_spacing_channels: int = 1 # components of one element at least this many channels apart
    figure_anchor_seed: Optional[int] = 20260909
    # None  -> the figure occupies FRESH channels on every trial (nothing can be learned across
    #          trials; only the K repetitions inside one trial are available).
    # int   -> the figure occupies the SAME channels on every trial, so a listener accumulates
    #          K x n_trials exposures to one pattern. Required for any claim about implicit
    #          learning. The foil interval still redraws its channels every element, so the
    #          within-trial comparison is unchanged and the per-channel budget still matches.
    steps_ms: Tuple[float, ...] = (0.0, 4.0, 7.0, 11.0, 14.0, 17.0)
    main_variants: Tuple[str, ...] = ("rising", "redrawn")   # one psychometric function each
    n_elements: int = 9
    iei_min_ms: float = 300.0            # inter-element onset interval, drawn uniformly: 3-5 Hz
    iei_max_ms: float = 333.0
    lead_min_ms: float = 350.0           # first element onset, drawn uniformly
    lead_max_ms: float = 600.0
    tail_min_ms: float = 350.0           # guaranteed background after the last element ends
    matched_incidence: bool = True
    # True -> both intervals carry BOTH the target's channels and the foil's channels at every
    #         element; only which of the two is time-aligned differs. Per-channel counts, channel
    #         recurrence and element-rate structure are then identical by construction, so the
    #         foil's elements can be made maximally unlike each other without reopening the
    #         single-channel periodicity cue that foil_subpool_size was invented to close.
    foil_universe_size: Optional[int] = 23
    # size of the channel universe the foil elements are drawn from, disjoint from the figure set.
    # Larger -> more dissimilar consecutive foil elements. Defaults to n_elements*n_components.
    anchored_fraction: float = 0.5
    # fraction of trials that use the anchored (learnable) figure set; the rest draw a fresh one.
    # 0.5 gives a within-session contrast between a familiar figure and a novel one.
    foil_subpool_size: Optional[int] = None
    # None -> the foil draws its element channels from the WHOLE pool, so its channels are each
    #         used ~K*N/P times while the target's are used K times. That asymmetry is the
    #         single-channel periodicity residual.
    # int  -> the foil draws from a restricted subpool of this many channels, so its channels
    #         recur ~K*N/Q times and the periodicity of the two intervals is far closer. The
    #         foil's figure still lands on a DIFFERENT set every element, so the manipulation
    #         (does the set recur?) is preserved.
    max_shared_consecutive: int = 5      # redrawn sets: channels in common with the previous set
    max_shared_any: int = 5              # redrawn sets: channels in common with any earlier set

    # ---- trial ---------------------------------------------------------------
    isi_ms: float = 400.0
    lead_silence_ms: float = 50.0

    # ---- experiment ----------------------------------------------------------
    trials_per_condition: int = 10       # main block, per (variant, step) cell; must be even
    max_condition_run: int = 2           # consecutive trials sharing a (variant, step) cell
    max_variant_run: int = 4             # consecutive trials sharing a ladder
    practice_cells: Tuple[Tuple[str, float], ...] = (("ungrouped", 0.0), ("rising", 0.0))
    practice_n: int = 12
    practice_criterion: int = 10         # correct out of practice_n required to proceed
    practice_max_rounds: int = 2         # attempts allowed per practice stage
    break_every: int = 40
    feedback_main: bool = True
    feedback_control: bool = True
    control_cells: Tuple[Tuple[str, float], ...] = (("ungrouped", 0.0), ("onechannel", 0.0))
    # 'scattered' used to sit here -- a figure whose components recur on the same channels but
    # never line up. Matched incidence builds that control into every trial: the interval without
    # the recurring group still carries the target's channels at every element, scattered.
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
    n_active_channels: int               # channels carrying tones in a trial
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
    R = cfg.figure_repeats
    # Only the figure set plus the foil universe carry tones; a wide pool costs nothing.
    U = cfg.foil_universe_size or (K * N)
    P_act = min(P, N + U) if cfg.matched_incidence else P
    # A matched-incidence element also holds the scattered counterpart tones, which are spread
    # over one element-duration, so its footprint is at least 2*R*D whatever the step is.
    # matched incidence: the element holds the aligned group AND a scattered counterpart of the
    # same extent, one after the other in the worst case, so its footprint is twice the span.
    mult = 2.0 if cfg.matched_incidence else 1.0
    spans = tuple(mult * ((N - 1) * s + R * D) for s in cfg.steps_ms)
    # 'scattered' spreads its components over one element-duration instead of using the step,
    # so its element is 2*R*D wide regardless of the step. Budget for that.
    control_spans = tuple((2 * R * D if v == "scattered" else mult * ((N - 1) * s + R * D))
                          for v, s in tuple(cfg.control_cells) + tuple(cfg.practice_cells))
    max_span = max(spans + control_spans) if (spans or control_spans) else D
    sched_max = cfg.lead_max_ms + (K - 1) * cfg.iei_max_ms + max_span + cfg.tail_min_ms
    occ = M * D / T
    mean_sim = P_act * occ
    # background count is ~binomial(P, occ) at any instant; add the whole element on top
    # a matched-incidence element window holds the aligned set AND its scattered counterpart
    max_sim_bound = int(min(P_act, math.ceil(mean_sim + 4.0 * math.sqrt(max(mean_sim, 1.0))))
                        + (2 * N if cfg.matched_incidence else N))
    peak_bound = cfg.tone_amplitude * max_sim_bound
    erb_low = _pool.erb_width_hz(freqs[0]) if P else float("nan")
    ladder = []
    for s in cfg.steps_ms:
        overlap = max(0.0, 1.0 - s / (R * D)) if s > 0 else 1.0
        max_simul = N if s == 0 else min(N, int(math.ceil(R * D / s)) if (R * D) % s else int(R * D // s))
        ladder.append(dict(step_ms=s, adjacent_overlap=overlap, max_simultaneous=max_simul,
                           span_ms=(N - 1) * s + R * D, gap_ms=max(0.0, s - R * D)))
    main_cells = tuple((v, float(s)) for v in cfg.main_variants for s in cfg.steps_ms)
    n_main = cfg.trials_per_condition * len(main_cells)
    n_prac = cfg.practice_n * len(cfg.practice_cells)
    n_ctrl = cfg.control_trials_per_cell * len(cfg.control_cells)
    trial_dur = (2 * T + cfg.isi_ms + cfg.lead_silence_ms) / 1000.0 + cfg.response_allowance_s + cfg.feedback_s + cfg.iti_s
    n_trials = n_prac + n_main + n_ctrl
    n_breaks = (n_main + n_ctrl) // max(cfg.break_every, 1) + 2   # + block transitions
    est = cfg.setup_minutes + (n_trials * trial_dur + n_breaks * cfg.break_s) / 60.0
    return Derived(
        channel_freqs_hz=freqs, n_channels=P, n_active_channels=P_act, tone_dur_grid=int(round(D / cfg.grid_ms)),
        spans_ms=spans, max_span_ms=max_span, schedule_max_ms=sched_max,
        occupancy_per_channel=occ, mean_simultaneous=mean_sim,
        max_simultaneous_bound=max_sim_bound, peak_bound=peak_bound,
        min_beat_rate_hz=erb_low * cfg.pool_spacing_erb * (cfg.figure_min_spacing_channels
                                                          if cfg.matched_incidence else 1),
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
    if cfg.figure_anchor_seed is not None and not isinstance(cfg.figure_anchor_seed, int):
        errs.append("figure_anchor_seed must be an int or None")
    if cfg.figure_repeats < 1:
        errs.append("figure_repeats must be >= 1")
    need = int(math.ceil(K * cfg.figure_repeats * 1.25)) + 2
    if M < need:
        errs.append(f"tones_per_channel={M} is too small: a recurring channel spends "
                    f"{K * cfg.figure_repeats} tones on the figure and needs background left over to "
                    f"place around them; raise tones_per_channel to >= {need} or lower "
                    f"n_elements/figure_repeats")
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
    if not (0.0 <= cfg.anchored_fraction <= 1.0):
        errs.append("anchored_fraction must be in [0, 1]")
    if cfg.anchored_fraction < 1.0 and cfg.figure_anchor_seed is None:
        errs.append("anchored_fraction < 1 needs figure_anchor_seed set: there is nothing to anchor to")
    if cfg.matched_incidence:
        gap = cfg.figure_min_spacing_channels
        U = cfg.foil_universe_size or (K * N)
        if U < N:
            errs.append(f"foil_universe_size={U} is smaller than n_components={N}: a foil element "
                        f"cannot be drawn")
        elif U < 2 * N:
            errs.append(f"foil_universe_size={U} < 2*n_components={2 * N}: consecutive foil elements "
                        f"cannot be disjoint, which is the point of matched_incidence; raise it")
        if (N + U) * gap > d.n_channels:
            errs.append(f"matched_incidence needs {(N + U) * gap} pool channels to lay out "
                        f"{N}+{U} channels {gap} apart, but the pool has {d.n_channels}; widen "
                        f"pool_low_hz..pool_high_hz, reduce pool_spacing_erb, or lower "
                        f"foil_universe_size")
        if cfg.foil_subpool_size is not None:
            errs.append("foil_subpool_size and matched_incidence are two different answers to the "
                        "same problem; set foil_subpool_size to null when matched_incidence is on")
        reuse = int(math.ceil(K * N / max(U, 1)))
        need_m = int(math.ceil(max(K, reuse) * cfg.figure_repeats * 1.25)) + 2
        if M < need_m:
            errs.append(f"tones_per_channel={M} is too small for matched_incidence: every active "
                        f"channel carries element tones in BOTH intervals (up to "
                        f"{max(K, reuse) * cfg.figure_repeats}); raise it to >= {need_m}")
    if cfg.foil_subpool_size is not None:
        Q = cfg.foil_subpool_size
        n_spaced = (d.n_channels + cfg.figure_min_spacing_channels - 1) // cfg.figure_min_spacing_channels
        if Q < N + 1:
            errs.append(f"foil_subpool_size={Q} must exceed n_components={N}, or the foil cannot draw "
                        f"a different set each element")
        elif Q > n_spaced:
            errs.append(f"foil_subpool_size={Q} exceeds the {n_spaced} channels available at "
                        f"figure_min_spacing_channels={cfg.figure_min_spacing_channels}; lower it")
        else:
            # two N-subsets of a Q-subpool must share at least 2N-Q channels
            forced = max(0, 2 * N - Q)
            if cfg.max_shared_any < forced:
                errs.append(f"with foil_subpool_size={Q} and n_components={N}, any two foil sets must "
                            f"share at least {forced} channels; raise max_shared_any to >= {forced}")
            if cfg.max_shared_consecutive < forced:
                errs.append(f"with foil_subpool_size={Q} and n_components={N}, CONSECUTIVE foil sets must "
                            f"also share at least {forced} channels; raise max_shared_consecutive to "
                            f">= {forced} (it is currently {cfg.max_shared_consecutive})")
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
    lines.append(f"figure anchoring     " + ("FIXED across trials (seed "
                 f"{cfg.figure_anchor_seed}): learning is possible" if cfg.figure_anchor_seed is not None
                 else "fresh every trial: cross-trial learning is IMPOSSIBLE"))
    lines.append(f"ladders              {len(cfg.main_variants)}: " +
                 ", ".join(f"{v} x {len(cfg.steps_ms)} steps x {cfg.trials_per_condition} trials"
                           for v in cfg.main_variants))
    lines.append(f"practice stages      " + ", ".join(f"{v} {s:g} ms" for v, s in cfg.practice_cells) +
                 f" ({cfg.practice_n} trials each, criterion {cfg.practice_criterion})")
    lines.append(f"trials               practice {d.n_practice_trials}, main {d.n_main_trials}, "
                 f"control {d.n_control_trials}")
    lines.append(f"estimated session    {d.est_session_minutes:.1f} min (limit {cfg.max_session_minutes})")
    return "\n".join(lines)
