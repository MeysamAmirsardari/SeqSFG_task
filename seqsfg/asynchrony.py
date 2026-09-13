"""Onset asynchrony: how far apart can component onsets be and still bind into one figure?

The two-interval task asked which of two sounds carried a recurring group. This one asks a
single-interval question -- was a figure there at all -- and walks the components of the figure
apart in time until the answer stops being available. Two things are varied:

    step      how far consecutive components are separated, 0 to 20 ms
    order     whether the within-element timing pattern RECURS (fixed) or is redrawn every
              element. The channels recur either way; only the temporal order changes.

Why this file exists instead of a change to yesno.py: the absent class is different. In
yesno.py the listener is asked whether one figure kept coming back, so the absent interval
carries a figure that never repeats. Here the listener is asked whether a figure was there,
so the absent interval must sound like a cloud -- and the channels every element lands on are
drawn from the WHOLE pool rather than from a contiguous band, so an element has no register
and a run of them does not stream.

What is identical between present and absent, by construction and not on average:

    * the set of channels that carry any tone at all
    * the number of tones in every channel (the per-channel budget, exactly)
    * the number of tones sounding at any instant, in distribution
    * the element schedule: the same onsets, the same jittered rate
    * under the default absent class, the number of time-aligned groups per element

and the only thing that differs is WHICH set of channels is the aligned one, and whether it is
the same set every element. That is the manipulation.

Three absent classes are implemented and the audit measures all three, because the choice is a
scientific claim rather than a detail:

    roving      every element carries an aligned group, on a fresh scattered channel set each
                time, so nothing recurs. The figure's own channels are present and scattered.
                Fully matched; the default.
    incoherent  no aligned group anywhere: the figure's channels and the others are all
                scattered inside their element windows. The purest cloud, and the closest to
                what the question literally asks -- but synchrony IS an envelope event, so the
                audit will say how much of a level cue that leaves at the bottom of the ladder.
    plain       a flat cloud with no element structure at all. This is the classic detection
                contrast. It is here so the report can say, with a number, how detectable it is
                without hearing anything group.

Run ``seqsfg asynchrony-verify`` before running a listener, and read sections [3] and [5].
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
from .stimulus import FIGURE, Interval, make_trial, render_interval
from .verify import HOP_MS, WIN_MS, measure_interval, scalar_features
from .yesno import (auc_dprime, criterion_yesno, dprime_yesno, feature_separation,
                    learnt_observer, _plain_like)

ORDERS = ("fixed", "redrawn", "rising")
# fixed   : one random permutation of the delays, drawn per trial and used by EVERY element.
#           The group's shape recurs; that is the "recurring temporal order" condition.
# redrawn : a fresh permutation every element. The channels still recur, the shape does not.
# rising  : the delays ascend with frequency, so the element is a sweep. Available, but not in
#           the default design: a sweep adds a frequency-time trajectory, which is a second
#           grouping cue on top of the one being measured.
ABSENT_KINDS = ("roving", "incoherent", "plain")

_VARIANT = {"fixed": "scrambled", "redrawn": "redrawn", "rising": "rising"}


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class AsyncConfig:
    """This task's own parameters.

    Deliberately NOT fields of Config: adding fields there would change the hash of every
    configuration already run, and the two shipped pilots would stop resuming.
    """
    orders: Tuple[str, ...] = ("fixed", "redrawn")
    absent_class: str = "roving"
    trials_per_cell: int = 10          # present trials per (step, order); absent gets the same
    practice_trials: int = 12
    practice_step_ms: float = 0.0
    feedback_main: bool = False        # a yes/no criterion is the listener's to set; no nudging
    feedback_practice: bool = True
    max_class_run: int = 4             # consecutive trials with the same answer. Capping runs at
    # all raises P(the next answer differs) above 0.5; 4 costs 0.54, 3 costs 0.57. There is no
    # feedback in the main block, so the listener cannot track the true sequence either way.
    max_cell_run: int = 2              # consecutive trials from the same (step, order, class)
    threshold_dprime: float = 1.0      # the d' the asynchrony limit is read off at
    n_boot: int = 4000

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["orders"] = list(self.orders)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AsyncConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown asynchrony keys: {sorted(unknown)}")
        kw = dict(d)
        if "orders" in kw:
            kw["orders"] = tuple(str(o) for o in kw["orders"])
        return cls(**kw)


def load_preset(path) -> Tuple[Config, AsyncConfig]:
    """A preset is one JSON file with a `config` section and an `asynchrony` section.

    Top-level keys beginning with an underscore are ignored, so the file can carry its own
    explanation next to its values.
    """
    raw = {k: v for k, v in json.loads(Path(path).read_text()).items() if not k.startswith("_")}
    if "config" not in raw or "asynchrony" not in raw:
        raise ValueError(f"{path}: needs both a 'config' and an 'asynchrony' section")
    return Config.from_dict(raw["config"]), AsyncConfig.from_dict(raw["asynchrony"])


def check(cfg: Config, acfg: AsyncConfig) -> None:
    """Refuse combinations that would quietly change what is being measured."""
    errs: List[str] = []
    if not cfg.matched_incidence:
        errs.append("matched_incidence must be true: both the 'roving' and the 'incoherent' "
                    "absent classes are built out of it, and without it the per-channel tone "
                    "counts stop matching between the classes")
    if cfg.figure_band_channels is not None:
        errs.append(f"figure_band_channels={cfg.figure_band_channels} confines every element to a "
                    f"contiguous band, which gives it a register; a run of banded elements streams, "
                    f"and that is what made the absent interval sound like a figure. Set it to null "
                    f"so elements are drawn from the whole pool")
    if acfg.absent_class not in ABSENT_KINDS:
        errs.append(f"absent_class={acfg.absent_class!r}; choose from {ABSENT_KINDS}")
    if not acfg.orders:
        errs.append("orders must name at least one condition")
    for o in acfg.orders:
        if o not in ORDERS:
            errs.append(f"unknown order {o!r}; choose from {ORDERS}")
    if len(set(acfg.orders)) != len(acfg.orders):
        errs.append("orders must be unique")
    if len(cfg.steps_ms) < 3:
        errs.append("a ladder of fewer than three steps cannot show where detection fails")
    if cfg.steps_ms and cfg.steps_ms[0] != 0.0:
        errs.append("steps_ms must start at 0: the synchronous rung is the reference every other "
                    "rung is read against")
    if acfg.practice_step_ms not in cfg.steps_ms:
        errs.append(f"practice_step_ms={acfg.practice_step_ms} is not on the ladder {list(cfg.steps_ms)}")
    if acfg.trials_per_cell < 4:
        errs.append("trials_per_cell < 4 gives a d' no bootstrap can say anything about")
    if acfg.max_class_run < 1 or acfg.max_cell_run < 1:
        errs.append("max_class_run and max_cell_run must be >= 1")
    if acfg.threshold_dprime <= 0:
        errs.append("threshold_dprime must be > 0")
    # the footprint rule, stated here too because it is the one that bites when the step grows
    N, D = cfg.n_components, cfg.tone_dur_ms
    foot = 2.0 * ((N - 1) * max(cfg.steps_ms) + cfg.figure_repeats * D)
    if foot > cfg.iei_min_ms:
        errs.append(f"the widest element needs {foot:.0f} ms of room (the aligned group plus its "
                    f"scattered counterpart, each jittered over one span) but iei_min_ms is "
                    f"{cfg.iei_min_ms:.0f}: elements would run into each other. Raise iei_min_ms "
                    f"to >= {foot:.0f}, or cap the ladder at "
                    f"{(cfg.iei_min_ms / 2.0 - D) / (N - 1):.0f} ms")
    if errs:
        raise ValueError("asynchrony configuration is not usable:\n  - " + "\n  - ".join(errs))


def notes(cfg: Config, acfg: AsyncConfig) -> List[str]:
    """Things worth knowing about a configuration that are choices, not errors."""
    out = []
    if cfg.figure_anchor_seed is not None:
        out.append(f"figure_anchor_seed={cfg.figure_anchor_seed}: {cfg.anchored_fraction:.0%} of "
                   f"trials reuse ONE figure, so a listener can learn it across the session and "
                   f"the ladder measures learning as well as binding. For a limit, set it to null.")
    if "rising" in acfg.orders and len(acfg.orders) > 1:
        out.append("the 'rising' order makes each element a frequency sweep, which is a grouping "
                   "cue of its own, so its ladder is not comparable rung for rung with an order "
                   "that has no trajectory; the rising-minus-fixed difference is what that "
                   "trajectory is worth.")
    elif acfg.orders == ("rising",):
        out.append("every element is a frequency sweep, so the limit this measures is the limit "
                   "for a figure that has a trajectory as well as a shared onset pattern. It is "
                   "the most generous case, and it is the right one to report as long as it is "
                   "described that way. A 'fixed' ladder bounds it from below.")
    if acfg.absent_class != "roving":
        out.append(f"absent_class={acfg.absent_class!r} is not the matched one; read section [3] "
                   f"of the audit before running a listener on it.")
    if acfg.feedback_main:
        out.append("feedback in the main block moves the criterion during the run; d' survives "
                   "that but c does not, and the half-split diagnostic will show it.")
    return out


# ----------------------------------------------------------------------------
# stimulus
# ----------------------------------------------------------------------------
_ALIGNED = {"present": "S", "roving": "foil", "incoherent": "none"}


def _build(cfg: Config, d: Derived, seed: int, step_ms: float, order: str, aligned: str,
           max_rebuilds: int = 50) -> Interval:
    """One interval of an asynchrony trial. `aligned` is 'S', 'foil' or 'none'.

    The structure a trial shares across its three possible sides -- the figure's channels, the
    universe the other elements are drawn from, those elements' channels, the element schedule
    and the delay pattern -- is drawn from a stream seeded on (seed, step, order) alone. Which
    set is time-aligned, the scatter offsets, the phases and the background fill come from a
    second stream that also knows `aligned`, so two sides built from the same seed are the same
    trial seen two ways while two sides built from different seeds are independent sounds.

    That distinction matters: a listener meets independent sounds, so the audit uses different
    seeds for the two classes, and only the construction check pairs them.
    """
    from .stimulus import (PlacementError, build_matched, sample_figure_set, sample_foil_sets_dissimilar,
                           sample_foil_universe, sample_patterns, sample_schedule)
    P, N, K = d.n_channels, cfg.n_components, cfg.n_elements
    variant = _VARIANT[order]
    oi = ORDERS.index(order)
    ai = ("S", "foil", "none").index(aligned)
    last = None
    for attempt in range(max_rebuilds):
        try:
            rs = np.random.default_rng([int(seed), attempt, oi, 0xA5F6])
            S = sample_figure_set(rs, P, N, cfg.figure_min_spacing_channels)
            patterns = sample_patterns(rs, cfg, variant)
            universe = sample_foil_universe(rs, cfg, P, S, cfg.foil_universe_size or (K * N))
            foil_sets = sample_foil_sets_dissimilar(rs, cfg, universe, K)
            t_el = sample_schedule(rs, cfg)
            active = np.unique(np.concatenate([S, universe] + [np.asarray(g) for g in foil_sets]))
            rng = np.random.default_rng([int(seed), attempt, oi, ai, 0x5A6F])
            role = {"S": "recurring", "foil": "redrawn", "none": "ungrouped"}[aligned]
            return build_matched(role, rng, cfg, d, step_ms, variant, t_el, S, foil_sets,
                                 patterns, active, aligned=aligned)
        except PlacementError as e:                                   # pragma: no cover - rare
            last = e
            continue
    raise RuntimeError(f"asynchrony seed={seed} step={step_ms} order={order} aligned={aligned}: "
                       f"{max_rebuilds} rebuilds failed ({last})")


def build_interval(cfg: Config, d: Derived, seed: int, step_ms: float, order: str, present: bool,
                   absent: str = "roving") -> Interval:
    """One interval. `present` selects the class, `order` the temporal pattern, `absent` the foil.

    At step 0 every order is the same sound: the delays are all multiplied by zero. The design
    collapses that rung to one cell rather than pretending otherwise.
    """
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}")
    if absent not in ABSENT_KINDS:
        raise ValueError(f"absent must be one of {ABSENT_KINDS}")
    if present:
        return _build(cfg, d, seed, step_ms, order, "S")
    if absent == "plain":
        # no element structure at all: the same channels at the same budget, placed at random
        iv = _build(cfg, d, seed, step_ms, order, "S")
        return _plain_like(cfg, d, iv, np.random.default_rng([int(seed), 0xC10D]))
    return _build(cfg, d, seed, step_ms, order, _ALIGNED[absent])


def render(cfg: Config, d: Derived, iv: Interval) -> np.ndarray:
    lead = np.zeros(cfg.ms_to_samples(cfg.lead_silence_ms), dtype=np.float32)
    return np.concatenate([lead, render_interval(cfg, iv, d)])


def span_ms(cfg: Config, step_ms: float) -> float:
    return (cfg.n_components - 1) * step_ms + cfg.figure_repeats * cfg.tone_dur_ms


# ----------------------------------------------------------------------------
# design
# ----------------------------------------------------------------------------
def cells(cfg: Config, acfg: AsyncConfig) -> Tuple[Tuple[float, str], ...]:
    """(step, order) cells. Step 0 appears once: at zero separation there is no order."""
    out: List[Tuple[float, str]] = [(0.0, acfg.orders[0])]
    for s in cfg.steps_ms[1:]:
        out.extend((float(s), o) for o in acfg.orders)
    return tuple(out)


def order_of(cfg: Config, acfg: AsyncConfig, step: float, order: str) -> str:
    """Which cell a (step, order) pair is measured in. Step 0 is shared by every order."""
    return acfg.orders[0] if float(step) == 0.0 else order


def duration_estimate(cfg: Config, acfg: AsyncConfig) -> dict:
    n_cells = len(cells(cfg, acfg))
    n_main = 2 * acfg.trials_per_cell * n_cells
    trial_s = ((cfg.interval_dur_ms + cfg.lead_silence_ms) / 1000.0 + cfg.response_allowance_s
               + cfg.iti_s)
    main_s = n_main * (trial_s + (cfg.feedback_s if acfg.feedback_main else 0.0))
    prac_s = acfg.practice_trials * (trial_s + cfg.feedback_s)
    n_breaks = max(0, n_main // max(cfg.break_every, 1) - 1) + 1
    total = cfg.setup_minutes * 60.0 + main_s + prac_s + n_breaks * cfg.break_s
    return dict(n_cells=n_cells, n_main=n_main, n_practice=acfg.practice_trials,
                trial_s=trial_s, n_breaks=n_breaks, minutes=total / 60.0)


def _trailing_run(out: List, key, value) -> int:
    n = 0
    for it in reversed(out):
        if key(it) != value:
            break
        n += 1
    return n


def _shuffle(rng: np.random.Generator, items: List, keys: Sequence[Tuple], tries: int = 400) -> List:
    """Random order under every (key, max_run) limit, built forwards instead of by rejection.

    design.constrained_shuffle draws whole orders and rejects the ones that violate a limit.
    That is uniform over the admissible set and is the right thing for a hundred-trial block,
    but the chance a random order of 220 trials contains no run of four is about one in a
    million, so it never finds one. Here each trial is drawn uniformly from the items that are
    still legal, and the whole order is restarted if that ever runs out.

    Uniform over ITEMS and not over classes, deliberately. Preferring the class with the most
    trials left keeps the greedy from cornering itself, but it also makes 'yes' and 'no' strictly
    alternate whenever the two are equally numerous -- which they are here, by design. A listener
    who noticed that would score perfectly without hearing anything. Drawing an item uniformly
    already favours the larger class in proportion to its size, which is enough.
    """
    for _ in range(tries):
        pool = list(items)
        rng.shuffle(pool)
        out: List = []
        while pool:
            ok = [i for i, it in enumerate(pool)
                  if all(_trailing_run(out, k, k(it)) < m for k, m in keys)]
            if not ok:
                break
            out.append(pool.pop(ok[int(rng.integers(len(ok)))]))
        if not pool:
            return out
    raise RuntimeError("could not order the trials under max_class_run/max_cell_run; raise one "
                       "of them or change trials_per_cell")


def power_estimate(cfg: Config, acfg: AsyncConfig, at_dprime: float = 1.0) -> dict:
    """What one session can and cannot resolve, before anyone is run.

    For an unbiased yes/no cell of n present and n absent trials at true d',

        var(d') = p_h(1-p_h)/(n phi(z_h)^2) + p_f(1-p_f)/(n phi(z_f)^2)

    which is the delta-method variance of z(H) - z(F). Everything else follows from it: the
    contrast between two orders at one step costs a factor sqrt(2), averaging it over the
    nonzero steps buys back sqrt(number of them), and listeners divide by sqrt(their number).
    """
    n = acfg.trials_per_cell
    zh, zf = at_dprime / 2.0, -at_dprime / 2.0
    ph, pf = stats.norm.cdf(zh), stats.norm.cdf(zf)
    var = (ph * (1 - ph) / (n * stats.norm.pdf(zh) ** 2)
           + pf * (1 - pf) / (n * stats.norm.pdf(zf) ** 2))
    se_cell = float(np.sqrt(var))
    n_nz = max(len([s for s in cfg.steps_ms if s != 0.0]), 1)
    se_contrast = float(se_cell * np.sqrt(2.0) / np.sqrt(n_nz)) if len(acfg.orders) > 1 else float("nan")
    # the smallest effect an 80%-power two-sided test at 0.05 would catch
    mde = 2.802 * se_contrast
    return dict(at_dprime=at_dprime, n_per_cell=n, se_cell=se_cell, se_contrast=se_contrast,
                mde_contrast=mde,
                listeners_for=lambda target: int(math.ceil((2.802 * se_contrast / target) ** 2)))


@dataclass
class AsyncSpec:
    index: int
    block: str
    step_ms: float
    order: str
    present: bool
    seed: int


def make_design(cfg: Config, acfg: AsyncConfig, code: str, session_index: int) -> dict:
    """Balanced present/absent in every cell, shuffled so neither the answer nor the cell runs on.

    In yes/no the listener sets their own criterion, so a run of identical correct answers is
    far more damaging than in forced choice: it invites a response strategy that survives into
    the trials after it.
    """
    from .design import session_seed
    check(cfg, acfg)
    seed = session_seed(code, session_index)
    rng = np.random.default_rng(seed)
    items = [(s, o, p) for (s, o) in cells(cfg, acfg) for p in (True, False)
             for _ in range(acfg.trials_per_cell)]
    items = _shuffle(rng, items, [(lambda t: t[2], acfg.max_class_run),
                                  (lambda t: t, acfg.max_cell_run)])
    main = [AsyncSpec(i, "main", float(s), o, bool(p), int(rng.integers(1, 2 ** 31 - 1)))
            for i, (s, o, p) in enumerate(items)]
    po = acfg.orders[0]
    pit = [(float(acfg.practice_step_ms), po, bool(p)) for p in (True, False)] * \
          (acfg.practice_trials // 2)
    pit = _shuffle(rng, pit, [(lambda t: t[2], acfg.max_class_run)])
    practice = [AsyncSpec(i, "practice", s, o, p, int(rng.integers(1, 2 ** 31 - 1)))
                for i, (s, o, p) in enumerate(pit)]
    dz = {"task": "asynchrony", "participant_code": code, "session_index": int(session_index),
          "session_seed": seed, "config_hash": cfg.hash(), "asynchrony_hash": acfg.hash(),
          "absent": acfg.absent_class, "cells": [list(c) for c in cells(cfg, acfg)],
          "practice": [vars(t) for t in practice], "main": [vars(t) for t in main]}
    dz["design_hash"] = hashlib.sha256(
        json.dumps({k: dz[k] for k in ("session_seed", "config_hash", "asynchrony_hash",
                                       "absent", "practice", "main")},
                   sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    return dz


# ----------------------------------------------------------------------------
# the session
# ----------------------------------------------------------------------------
class AsyncRunner:
    """Single-interval session: was a figure there, yes or no."""

    def __init__(self, cfg: Config, acfg: AsyncConfig, data_dir, device=None, audio: bool = True,
                 auto: Optional[float] = None):
        from .runner import Audio
        check(cfg, acfg)
        self.cfg, self.acfg, self.d = cfg, acfg, validate(cfg)
        self.data_dir = Path(data_dir)
        self.audio = Audio(cfg.sample_rate, device, enabled=audio and auto is None)
        self.auto = auto
        self.rng_auto = np.random.default_rng(17)

    def pause(self, msg: str) -> None:
        from .runner import getkey
        print("\n" + msg)
        if self.auto is None:
            getkey({" "}, "  press space to go on  ")

    def _answer(self, spec: AsyncSpec) -> str:
        from .runner import getkey
        if self.auto is not None:
            # a simulated listener: sensitivity falls with the step, and a redrawn order costs
            # a fixed amount. Only ever used to exercise the pipeline end to end.
            dp = 2.4 * float(np.exp(-spec.step_ms / self.auto)) - (0.4 if spec.order == "redrawn" else 0.0)
            x = self.rng_auto.normal(max(dp, 0.0) if spec.present else 0.0, 1.0)
            return "y" if x > 0.55 else "n"
        return getkey({"y", "n", "q"}, "  was a figure there? [y/n]  ")

    def _trial(self, spec: AsyncSpec, feedback: bool, i: int, n: int) -> bool:
        import time as _t
        from .runner import QuitRequested
        from .session import now_iso
        iv = build_interval(self.cfg, self.d, spec.seed, spec.step_ms, spec.order,
                            spec.present, self.acfg.absent_class)
        x = render(self.cfg, self.d, iv)
        print(f"  trial {i}/{n} ...", end="", flush=True)
        t0 = _t.time()
        if self.auto is None:
            self.audio.play(x)
        t_play = now_iso()
        k = self._answer(spec)
        if k == "q":
            raise QuitRequested()
        rt = (_t.time() - t0) * 1000.0
        correct = int((k == "y") == spec.present)
        if feedback:
            print("   correct" if correct else
                  f"   wrong ({'there was one' if spec.present else 'there was none'})")
        else:
            print("   ok")
        self.log.write(dict(trial_index=spec.index, block=spec.block, practice_round=0,
                            practice_stage=0, variant=f"async:{spec.order}", step_ms=spec.step_ms,
                            target_position=1 if spec.present else 2, seed=spec.seed,
                            response=k, correct=correct, rt_ms=round(rt, 1),
                            t_start=t_play, t_response=now_iso()))
        return bool(correct)

    def run(self, code: Optional[str] = None, session_index: Optional[int] = None):
        from .runner import QuitRequested, Runner
        from .session import (TrialLog, next_session_index, provenance, session_dir,
                              upsert_participant, write_json)
        cfg, acfg = self.cfg, self.acfg
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
        design = make_design(cfg, acfg, code, idx)
        meta = {**provenance(cfg), "task": "asynchrony", "asynchrony": acfg.to_dict(),
                "asynchrony_hash": acfg.hash(), "absent": acfg.absent_class,
                "participant_code": code, "session_index": idx,
                "session_seed": design["session_seed"], "design_hash": design["design_hash"],
                "design": design, "status": "started"}
        write_json(sdir / "session.json", meta)
        print(f"new asynchrony session {sdir} (absent='{acfg.absent_class}', "
              f"design {design['design_hash']})")
        self.log = TrialLog(sdir / "trials.csv")
        try:
            self.pause("Practice, with feedback. You will hear ONE sound each time.\n"
                       "Some of them contain a figure: a handful of tones that keep coming back\n"
                       "together, on the same pitches, over and over. The rest are just cloud.\n"
                       "Answer 'y' if a figure was there, 'n' if it was not.")
            pr = [AsyncSpec(**t) for t in design["practice"]]
            for i, s in enumerate(pr, 1):
                self._trial(s, acfg.feedback_practice, i, len(pr))
            mn = [AsyncSpec(**t) for t in design["main"]]
            self.pause(f"Main block: {len(mn)} trials, no feedback. The figure gets harder to\n"
                       "hear as the block goes on -- its tones stop starting together. Answer\n"
                       "what you hear; guessing 'no' when you are unsure is normal and fine.")
            for i, s in enumerate(mn, 1):
                if i > 1 and (i - 1) % cfg.break_every == 0:
                    self.pause("Take a break.")
                self._trial(s, acfg.feedback_main, i, len(mn))
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
def _counts(rows: Sequence[dict]) -> Tuple[int, int, int, int]:
    """(hits, n_present, false alarms, n_absent). target_position 1 means the figure was there."""
    sig = [r for r in rows if str(r["target_position"]) == "1"]
    noi = [r for r in rows if str(r["target_position"]) == "2"]
    return (sum(1 for r in sig if r["response"] == "y"), len(sig),
            sum(1 for r in noi if r["response"] == "y"), len(noi))


def _dp(c: Tuple[int, int, int, int]) -> float:
    h, ns, f, nn = c
    if not ns or not nn:
        return float("nan")
    return dprime_yesno(h / ns, f / nn, ns, nn)


def threshold_step(steps: Sequence[float], dprimes: Sequence[float], crit: float) -> float:
    """Where the ladder crosses `crit`, by linear interpolation between the rungs it crosses.

    Returns +inf when d' never falls to crit inside the measured range (the limit is beyond the
    widest step tested) and -inf when it is already below crit at step 0 (nothing was detected
    even when the components were simultaneous, so there is no limit to report).
    """
    s = np.asarray(steps, float)
    y = np.asarray(dprimes, float)
    ok = np.isfinite(y)
    s, y = s[ok], y[ok]
    if s.size < 2 or y[0] < crit:
        return -np.inf
    for i in range(1, s.size):
        if y[i] < crit <= y[i - 1]:
            frac = (y[i - 1] - crit) / max(y[i - 1] - y[i], 1e-12)
            return float(s[i - 1] + frac * (s[i] - s[i - 1]))
    return np.inf


def _boot_cells(cell_counts: Dict[Tuple[float, str], Tuple[int, int, int, int]], n_boot: int,
                seed: int = 7) -> List[Dict[Tuple[float, str], Tuple[int, int, int, int]]]:
    """One joint resample per replicate: every cell's hits and false alarms redrawn together.

    Resampling cells independently would understate the correlation a threshold and an order
    contrast both inherit from sharing the step-0 cell.
    """
    rng = np.random.default_rng(seed)
    out = []
    keys = list(cell_counts)
    for _ in range(n_boot):
        rep = {}
        for k in keys:
            h, ns, f, nn = cell_counts[k]
            rep[k] = (int(rng.binomial(ns, h / ns)) if ns else 0, ns,
                      int(rng.binomial(nn, f / nn)) if nn else 0, nn)
        out.append(rep)
    return out


def _ladder(counts: Dict[Tuple[float, str], Tuple[int, int, int, int]], steps: Sequence[float],
            order: str, first_order: str) -> Tuple[List[float], List[float]]:
    ss, dd = [], []
    for s in steps:
        k = (float(s), first_order if float(s) == 0.0 else order)
        if k in counts:
            ss.append(float(s)); dd.append(_dp(counts[k]))
    return ss, dd


def analyse(sessions: Sequence, crit: Optional[float] = None, n_boot: Optional[int] = None) -> str:
    """d' per (step, order), the asynchrony limit per order, and the order contrast."""
    from .session import read_json
    rows: List[dict] = []
    metas: List[dict] = []
    for sdir in sessions:
        p = Path(sdir)
        import csv as _csv
        with open(p / "trials.csv", newline="") as f:
            rr = [r for r in _csv.DictReader(f) if r["block"] == "main"]
        for r in rr:
            r["_session"] = p.name
        rows += rr
        if (p / "session.json").exists():
            metas.append(read_json(p / "session.json"))
    L = ["", "=" * 92, "ONSET ASYNCHRONY  (single interval: was a figure there?)", "=" * 92]
    if not rows:
        return "\n".join(L + ["no main-block trials recorded."])
    acfg = AsyncConfig.from_dict(metas[0]["asynchrony"]) if metas and "asynchrony" in metas[0] \
        else AsyncConfig()
    crit = acfg.threshold_dprime if crit is None else crit
    n_boot = acfg.n_boot if n_boot is None else n_boot
    steps = sorted({float(r["step_ms"]) for r in rows})
    orders = [o for o in acfg.orders]
    seen = {r["variant"].split(":", 1)[-1] for r in rows}
    orders = [o for o in orders if o in seen] or sorted(seen)
    first = orders[0]

    counts: Dict[Tuple[float, str], Tuple[int, int, int, int]] = {}
    for s in steps:
        for o in orders:
            key = (s, first if s == 0.0 else o)
            if key in counts:
                continue
            rs = [r for r in rows if float(r["step_ms"]) == s
                  and r["variant"].split(":", 1)[-1] == key[1]]
            if rs:
                counts[key] = _counts(rs)

    h, ns, f, nn = (sum(c[0] for c in counts.values()), sum(c[1] for c in counts.values()),
                    sum(c[2] for c in counts.values()), sum(c[3] for c in counts.values()))
    L.append(f"\n{len(rows)} main trials from {len({r['_session'] for r in rows})} session(s): "
             f"{ns} present, {nn} absent   absent class '{metas[0].get('absent', '?') if metas else '?'}'")
    L.append(f"  overall  hits {h}/{ns} = {h / ns:.3f}   false alarms {f}/{nn} = {f / nn:.3f}   "
             f"d' = {_dp((h, ns, f, nn)):+.2f}   c = {criterion_yesno(h / ns, f / nn, ns, nn):+.2f}")

    L.append("\n[1] the ladder")
    L.append(f"  {'step (ms)':<11}" + "".join(f"{o + ' d':>22}" for o in orders))
    boots = _boot_cells(counts, n_boot)
    for s in steps:
        line = f"  {s:<11.4g}"
        for o in orders:
            k = (s, first if s == 0.0 else o)
            if k not in counts:
                line += f"{'--':>22}"
                continue
            dv = _dp(counts[k])
            bs = np.array([_dp(b[k]) for b in boots])
            line += f"{dv:>+10.2f} [{np.quantile(bs, .025):+.2f},{np.quantile(bs, .975):+.2f}]"
        if s == 0.0 and len(orders) > 1:
            line += "   (one cell: at 0 ms every order is the same sound)"
        L.append(line)

    L.append(f"\n[2] the asynchrony limit: the step at which d' falls to {crit:.1f}")
    for o in orders:
        ss, dd = _ladder(counts, steps, o, first)
        t = threshold_step(ss, dd, crit)
        bt = np.array([threshold_step(*_ladder(b, steps, o, first), crit) for b in boots])
        fin = bt[np.isfinite(bt)]
        cens = 100.0 * np.mean(~np.isfinite(bt))
        word = (f"{t:.1f} ms" if np.isfinite(t) else
                (f"beyond {max(steps):g} ms" if t > 0 else f"not reached at 0 ms"))
        ci = (f"  [{np.quantile(fin, .025):.1f}, {np.quantile(fin, .975):.1f}] ms"
              if fin.size > 20 else "")
        L.append(f"  {o:<10} {word}{ci}" + (f"   ({cens:.0f}% of resamples fall outside the "
                                            f"measured range)" if cens > 2 else ""))

    if len(orders) > 1:
        L.append("\n[3] does a RECURRING temporal order help?  mean over the nonzero steps")
        base, alt = orders[0], orders[1]
        nz = [s for s in steps if s != 0.0]

        def contrast(c):
            v = [_dp(c[(s, base)]) - _dp(c[(s, alt)]) for s in nz
                 if (s, base) in c and (s, alt) in c]
            return float(np.mean(v)) if v else float("nan")
        obs = contrast(counts)
        bs = np.array([contrast(b) for b in boots])
        lo, hi = np.quantile(bs, .025), np.quantile(bs, .975)
        L.append(f"  d'({base}) - d'({alt}) = {obs:+.2f}  [{lo:+.2f}, {hi:+.2f}]   "
                 f"p(two-sided) = {2 * min(np.mean(bs <= 0), np.mean(bs >= 0)):.3f}")
        L.append(f"  positive means the figure was easier to hear when its within-element timing")
        L.append(f"  pattern repeated from element to element.")
        for s in nz:
            if (s, base) in counts and (s, alt) in counts:
                L.append(f"     {s:>5.4g} ms   {_dp(counts[(s, base)]) - _dp(counts[(s, alt)]):+.2f}")

    L.append("\n[4] bias, and whether it moved")
    yes = sum(1 for r in rows if r["response"] == "y")
    L.append(f"  said 'yes' on {yes}/{len(rows)} ({yes / len(rows):.2f}); "
             f"{'balanced' if abs(yes / len(rows) - 0.5) < 0.12 else 'LOPSIDED -- read c'}")
    half = len(rows) // 2
    for name, rs in (("first half", rows[:half]), ("second half", rows[half:])):
        c2 = _counts(rs)
        if c2[1] and c2[3]:
            L.append(f"  {name:<12} d' = {_dp(c2):+.2f}   "
                     f"c = {criterion_yesno(c2[0] / c2[1], c2[2] / c2[3], c2[1], c2[3]):+.2f}")
    L.append("  a criterion that moves by more than about 0.3 between halves is the commonest")
    L.append("  yes/no artefact; if it does, read the ladder by half as well as pooled.")
    rts = [float(r["rt_ms"]) for r in rows if r["rt_ms"]]
    if rts:
        L.append(f"  response time median {np.median(rts):.0f} ms")
    L.append("=" * 92)
    return "\n".join(L)


# ----------------------------------------------------------------------------
# the audit
# ----------------------------------------------------------------------------
def construction_check(cfg: Config, acfg: AsyncConfig, n: int = 40, seed: int = 909) -> dict:
    """Everything that must be EXACTLY equal between the classes, checked on the schedule.

    Both sides are built from the same seed here, so a failure is a construction bug and not a
    sampling fluctuation. The statistical audit below uses independent seeds instead, because
    independent sounds are what a listener meets.
    """
    d = validate(cfg)
    P, D = d.n_channels, d.tone_dur_grid
    keys = ("same_n_tones", "same_channel_counts", "same_active_channels", "budget_exact",
            "no_same_channel_overlap", "same_element_schedule", "elements_never_collide")
    fails = {k: 0 for k in keys}
    counts = {"present": [], "absent": []}
    aligned_groups = {"present": [], "absent": []}
    for j in range(n):
        step = float(cfg.steps_ms[j % len(cfg.steps_ms)])
        order = acfg.orders[j % len(acfg.orders)]
        s = int(np.random.default_rng([seed, j]).integers(2 ** 31 - 1))
        A_ = build_interval(cfg, d, s, step, order, True)
        B_ = build_interval(cfg, d, s, step, order, False, acfg.absent_class)
        ca = np.bincount(A_.channel, minlength=P); cb = np.bincount(B_.channel, minlength=P)
        fails["same_n_tones"] += A_.n_tones != B_.n_tones
        fails["same_channel_counts"] += not np.array_equal(ca, cb)
        fails["same_active_channels"] += not np.array_equal(ca > 0, cb > 0)
        fails["budget_exact"] += not (np.all(ca[ca > 0] == cfg.tones_per_channel)
                                      and np.all(cb[cb > 0] == cfg.tones_per_channel))
        for iv in (A_, B_):
            for c in range(P):
                o = np.sort(iv.onset[iv.channel == c])
                if o.size > 1 and np.min(np.diff(o)) < D:
                    fails["no_same_channel_overlap"] += 1
                    break
        if acfg.absent_class != "plain":
            fails["same_element_schedule"] += not np.array_equal(A_.element_onsets, B_.element_onsets)
        # an element must finish before the next one starts: the aligned group plus its
        # scattered counterpart, each jittered over one span
        foot = 2.0 * span_ms(cfg, step)
        if foot > np.min(np.diff(A_.element_onsets)) * cfg.grid_ms:
            fails["elements_never_collide"] += 1
        for tag, iv in (("present", A_), ("absent", B_)):
            cnt = M.count_trace(iv.onset, cfg.n_grid, D)
            counts[tag].append((cnt.mean(), cnt.std(), cnt.max()))
            aligned_groups[tag].append(int((iv.kind == FIGURE).sum()) // max(cfg.n_components, 1))
    return dict(n=n, fails=fails,
                counts={k: np.array(v) for k, v in counts.items()},
                aligned_groups={k: float(np.mean(v)) for k, v in aligned_groups.items()})


def scatter_table(cfg: Config, n: int = 300, seed: int = 31) -> dict:
    """How spread out an element's channels are: the thing a band destroys and a pool restores.

    A run of banded elements streams because each one has a register. Elements drawn from the
    whole pool span most of the spectrum and have none, so they do not group into a melody --
    which is what the absent interval has to avoid.
    """
    from .stimulus import (PlacementError, sample_figure_set, sample_foil_sets_dissimilar,
                           sample_foil_universe)
    d = validate(cfg)
    P, f = d.n_channels, d.channel_freqs_hz
    rows = {"figure": [], "other elements": []}
    for t in range(n):
        for att in range(20):
            try:
                rng = np.random.default_rng([seed, t, att])
                S = sample_figure_set(rng, P, cfg.n_components, cfg.figure_min_spacing_channels)
                uni = sample_foil_universe(rng, cfg, P, S, cfg.foil_universe_size or
                                           (cfg.n_elements * cfg.n_components))
                fs = sample_foil_sets_dissimilar(rng, cfg, uni, cfg.n_elements)
                break
            except PlacementError:                                     # pragma: no cover - rare
                continue
        else:                                                          # pragma: no cover - rare
            continue
        for tag, sets in (("figure", [S]), ("other elements", fs)):
            for s in np.asarray(sets):
                s = np.sort(np.asarray(s))
                rows[tag].append((np.diff(s).mean(), np.diff(s).min(),
                                  math.log2(f[s[-1]] / f[s[0]]),
                                  max(int(np.sum((s >= c) & (s < c + 4))) for c in range(P))))
    return {k: np.array(v) for k, v in rows.items()}


@dataclass
class CellResult:
    step_ms: float
    order: str
    names: List[str]
    Fp: np.ndarray
    Fa: np.ndarray
    sep: dict
    learnt: dict


def run_audit(cfg: Config, acfg: AsyncConfig, n_trials: int = 40, seed: int = 4242,
              absent: Optional[str] = None, n_perm: int = 20000, verbose: bool = True) -> dict:
    """Build independent present and absent intervals in every cell and ask what separates them.

    Independent, not paired: a yes/no listener hears one sound and compares it with a memory of
    what these sounds are usually like, so the audit must carry the same between-trial variance
    they do. Anything whose distribution differs -- in mean OR in spread -- is a criterion.
    """
    check(cfg, acfg)
    d = validate(cfg)
    absent = absent or acfg.absent_class
    ref = M.single_tone_reference(cfg, d, WIN_MS, HOP_MS)
    tag = zlib.crc32(absent.encode()) & 0xFFFF
    out: List[CellResult] = []
    t0 = time.time()
    for (step, order) in cells(cfg, acfg):
        if verbose:
            print(f"  {n_trials} present + {n_trials} absent at step {step:g} ms, "
                  f"{order} order ...", end="", flush=True)
        t1 = time.time()
        rows = {True: [], False: []}
        for present in (True, False):
            for j in range(n_trials):
                s = int(np.random.default_rng([seed, tag, int(step * 1000),
                                               ORDERS.index(order), int(present), j])
                        .integers(2 ** 31 - 1))
                iv = build_interval(cfg, d, s, float(step), order, present, absent)
                x = render_interval(cfg, iv, d)
                m = measure_interval(cfg, d, iv, x, iv.figure_set, span_ms(cfg, step), ref)
                rows[present].append(scalar_features(m))
        names = list(rows[True][0].keys())
        Fp = np.array([[r[k] for k in names] for r in rows[True]], float)
        Fa = np.array([[r[k] for k in names] for r in rows[False]], float)
        out.append(CellResult(float(step), order, names, Fp, Fa,
                              feature_separation(names, Fp, Fa, n_perm, seed + 1),
                              learnt_observer(Fp, Fa)))
        if verbose:
            print(f" {time.time() - t1:.0f}s")
    names = out[0].names
    Fp = np.vstack([c.Fp for c in out]); Fa = np.vstack([c.Fa for c in out])
    hits = sum(c.learnt["hit"] * c.Fp.shape[0] for c in out)
    fas = sum(c.learnt["fa"] * c.Fa.shape[0] for c in out)
    np_, na_ = Fp.shape[0], Fa.shape[0]
    return dict(cfg=cfg, acfg=acfg, d=d, absent=absent, n_trials=n_trials, seed=seed, cells=out,
                names=names, Fp=Fp, Fa=Fa,
                pooled=feature_separation(names, Fp, Fa, n_perm, seed + 2),
                pooled_learnt=dict(n=np_ + na_, hit=hits / np_, fa=fas / na_,
                                   dprime=dprime_yesno(hits / np_, fas / na_, np_, na_),
                                   pc=(hits + (na_ - fas)) / (np_ + na_)),
                construction=construction_check(cfg, acfg.__class__(**{**acfg.to_dict(),
                                                                      "absent_class": absent,
                                                                      "orders": list(acfg.orders)})),
                scatter=scatter_table(cfg), elapsed_s=time.time() - t0)


HEADLINE = [("rms_db", "long-term RMS (dB)"), ("peak", "peak amplitude"),
            ("env:level_db", "envelope level (dB)"), ("env:crest", "crest factor"),
            ("env:n_bursts_3sd", "envelope peaks above 3 SD"),
            ("env:n_bursts_5sd", "envelope peaks above 5 SD"),
            ("env:burst_height_max", "tallest burst (SD)"),
            ("env:kurtosis", "envelope kurtosis"),
            ("env:mod_depth", "envelope modulation depth"),
            ("env:mod_3_10Hz", "modulation power, element-rate band (dB)"),
            ("env:ac_peak_iei", "envelope autocorrelation at element lags"),
            ("spec_peakedness", "spectral peakedness (dB)"),
            ("ch:ioi_cv:max", "most irregular channel (IOI CV)"),
            ("ch:ioi_sd:mean", "channel IOI spread, mean (ms)"),
            ("count_audio_mean", "tones sounding, mean")]


def report(res: dict) -> str:
    cfg, acfg = res["cfg"], res["acfg"]
    W = 100
    L = ["=" * W,
         f"ONSET-ASYNCHRONY AUDIT   absent class '{res['absent']}'   "
         f"{res['n_trials']} present + {res['n_trials']} absent per cell",
         "=" * W,
         "A yes/no listener answers from ONE sound, so any property whose distribution differs",
         "between the classes -- in mean OR in spread -- is a usable criterion, whether or not",
         "anyone hears a figure. Everything below is measured on single intervals.", ""]

    c = res["construction"]
    L.append(f"[0] construction, checked on the schedule over {c['n']} matched pairs")
    for k, v in c["fails"].items():
        L.append(f"    {k:<36}{'ok' if v == 0 else f'FAILED on {v} of ' + str(c['n'])}")
    for tag in ("present", "absent"):
        a = c["counts"][tag]
        L.append(f"    tones sounding at once, {tag:<8} mean {a[:, 0].mean():5.2f}   "
                 f"sd {a[:, 1].mean():5.2f}   max {a[:, 2].mean():5.1f}")
    L.append(f"    time-aligned groups per interval   present {c['aligned_groups']['present']:.0f}"
             f"   absent {c['aligned_groups']['absent']:.0f}")
    L.append("")

    s = res["scatter"]
    L.append("[1] how scattered an element's channels are (a band gives an element a register,")
    L.append("    and a run of elements with registers streams -- which is a figure)")
    L.append(f"    {'':<18}{'mean gap (ch)':>15}{'span (oct)':>13}{'most in 4 ch':>15}"
             f"{'adjacent pair':>15}")
    for tag, a in s.items():
        L.append(f"    {tag:<18}{a[:, 0].mean():>15.2f}{a[:, 2].mean():>13.2f}"
                 f"{a[:, 3].mean():>15.2f}{np.mean(a[:, 1] == 1) * 100:>14.0f}%")
    L.append("    the figure and the other elements must match here, or which set is the aligned")
    L.append("    one is legible from the spectrum before anything is heard.")
    L.append("")

    idx = {n: i for i, n in enumerate(res["names"])}
    sep = res["pooled"]
    dp = {n: v for n, v in zip(sep["names"], sep["dprime"])}
    L.append("[2] the cues a listener would try first, pooled over the ladder")
    L.append(f"    {'feature':<42}{'present':>18}{'absent':>18}{'yes/no d':>10}")
    for key, label in HEADLINE:
        if key not in idx:
            continue
        p, a = res["Fp"][:, idx[key]], res["Fa"][:, idx[key]]
        L.append(f"    {label:<42}{p.mean():>10.3f}+-{p.std():<6.3f}{a.mean():>10.3f}"
                 f"+-{a.std():<6.3f}{dp.get(key, 0.0):>+10.2f}")
    L.append("")

    order = np.argsort(-sep["dprime"])
    L.append(f"[3] every one of the {sep['n_features']} features, largest yes/no d' first")
    L.append(f"    {'feature':<42}{'d':>8}{'AUC':>8}{'what differs':>14}")
    for i in order[:12]:
        L.append(f"    {sep['names'][i]:<42}{sep['dprime'][i]:>8.2f}{sep['auc'][i]:>8.3f}"
                 f"{sep['kind'][i]:>14}")
    L.append("")

    L.append("[4] can ANY of them separate the classes?  max-statistic permutation over labels")
    L.append(f"    largest d' observed           {sep['observed_max']:.3f}  "
             f"({sep['names'][int(order[0])]})")
    L.append(f"    largest under relabelling     95th percentile {sep['null_q95']:.3f}")
    L.append(f"    p (any feature separates)     {sep['p_value']:.3f}")
    L.append(f"    -> {'no measured property separates the classes' if sep['p_value'] >= 0.05 else 'THE CLASSES ARE SEPARABLE WITHOUT HEARING A FIGURE'}")
    L.append("")

    pl = res["pooled_learnt"]
    L.append("[5] an ideal observer that has LEARNT the best linear criterion over every")
    L.append("    feature (leave-one-out, fitted within each cell, held-out decisions pooled)")
    L.append(f"    hit {pl['hit']:.3f}   false alarm {pl['fa']:.3f}   d' = {pl['dprime']:+.3f}   "
             f"{pl['pc'] * 100:.1f}% correct")
    L.append("    This is the ceiling for someone who never hears a group and has only learnt")
    L.append("    what these sounds are usually like. It must sit at chance.")
    L.append("")

    L.append("[6] per cell")
    # a null d' from n present + n absent trials has a sampling SD of about
    # sqrt(2 p(1-p)/n) / phi(0) at p = 0.5, and the LARGEST of that many draws is what the eye
    # lands on. Without the band below, a cell reading +0.6 looks like a leak when it is noise.
    n_ = res["n_trials"]
    sd0 = float(np.sqrt(2 * 0.25 / n_) / stats.norm.pdf(0.0))
    kmax = float(stats.norm.ppf(1 - 1.0 / (2 * max(len(res["cells"]), 2)))) * sd0
    L.append(f"    a cell's learnt d' has a null SD of {sd0:.2f} at {n_}+{n_} trials, so the "
             f"largest of {len(res['cells'])} cells")
    L.append(f"    sits near {kmax:+.2f} with nothing wrong. Read a cell only if it is well past "
             f"that AND the same")
    L.append(f"    cell moves the same way at another seed.")
    L.append(f"    {'step (ms)':<11}{'order':<10}{'permutation p':>15}{'largest d':>12}"
             f"{'learnt d':>11}{'learnt hit/fa':>16}")
    for cr in res["cells"]:
        L.append(f"    {cr.step_ms:<11.4g}{cr.order:<10}{cr.sep['p_value']:>15.3f}"
                 f"{cr.sep['observed_max']:>12.2f}{cr.learnt['dprime']:>+11.2f}"
                 f"{cr.learnt['hit']:>8.2f}/{cr.learnt['fa']:<7.2f}")
    L.append("")
    for n_ in notes(cfg, acfg):
        L.append(f"    note: {n_}")
    L.append(f"built in {res['elapsed_s']:.0f}s; config {cfg.hash()}; asynchrony {acfg.hash()}")
    L.append("=" * W)
    return "\n".join(L)
