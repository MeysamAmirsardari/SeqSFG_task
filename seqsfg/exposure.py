"""Pre-exposure / exposure / post-exposure: does learning one temporal order help extract it?

The hypothesis is that repeated exposure to a particular temporal sequence establishes
predictive relationships among its frequency components, and that those relationships later
help pull the sequence out of a stochastic background. The behavioural estimand is a
difference in differences, per participant:

    D_i = (d'_post,trained - d'_pre,trained) - (d'_post,untrained - d'_pre,untrained)

A positive D_i would say that exposure helped *this sequence* beyond whatever general practice
the session produced. It would say nothing about neural pre-activation, binding circuitry,
implicit learning, or whether attention was involved; the exposure phase here is the same
attended yes/no task, so it is practice with a particular sequence, and that is how it should
be described.

Three phases, on top of the single-interval yes/no task of `seqsfg.yesno`:

    pre       yes/no detection of P-designated and Q-designated trials
    exposure  extra yes/no trials of the trained designation only
    post      pre repeated, with freshly generated acoustics

P and Q are two fixed onset orders over the SAME frequency set, so the only thing that
distinguishes them is *when* each component starts. One is trained, the other is the
comparison; which one is counterbalanced across participants and recorded explicitly.

What this file does NOT change: `Config` gains no fields, so every existing configuration
hashes exactly as before and existing sessions still resume. The parameters of this experiment
live in `ExposureConfig` and travel in their own section of a preset file.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
import zlib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Config, Derived, validate
from .stimulus import Interval, make_trial, render_interval
from . import yesno

PHASES = ("pre", "exposure", "post")
SEQUENCES = ("P", "Q")


# ----------------------------------------------------------------------------
# configuration (deliberately NOT part of Config; see the module docstring)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ExposureConfig:
    # --- sequences -----------------------------------------------------------
    sequence_seed: int = 20260910      # picks the pair (P, Q); same pair for every participant
    figure_set_seed: int = 20260909    # picks the frequency set both sequences use
    trained: str = "auto"              # "P" | "Q" | "auto" (counterbalanced by participant code)
    # --- what is tested ------------------------------------------------------
    test_steps_ms: Tuple[float, ...] = (0.0, 7.0, 14.0)
    # step 0 is the shared reference: at zero onset separation P and Q are the SAME sound, so
    # it measures general practice and acts as a manipulation check. D is never computed there.
    test_trials_per_cell: int = 8      # per phase x sequence x delay x {present, absent}
    # --- exposure ------------------------------------------------------------
    exposure_trials: int = 60          # total yes/no trials in the exposure phase
    exposure_steps_ms: Optional[Tuple[float, ...]] = None   # None -> the nonzero test steps
    exposure_present_fraction: float = 0.5
    # --- task ----------------------------------------------------------------
    absent_class: str = "roving"
    feedback_test: bool = False        # the same in BOTH pre/post sequence conditions, always
    feedback_exposure: bool = True
    practice_trials: int = 12
    # breaks come from Config.break_every: one source of truth for a shared setting
    # --- analysis ------------------------------------------------------------
    aggregate: str = "mean_over_nonzero_delays"   # fixed in advance; delay-specific kept too
    n_boot: int = 4000

    def hash(self) -> str:
        s = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["test_steps_ms"] = list(self.test_steps_ms)
        if self.exposure_steps_ms is not None:
            d["exposure_steps_ms"] = list(self.exposure_steps_ms)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ExposureConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown exposure keys: {sorted(unknown)}")
        kw = dict(d)
        if "test_steps_ms" in kw:
            kw["test_steps_ms"] = tuple(float(s) for s in kw["test_steps_ms"])
        if kw.get("exposure_steps_ms") is not None:
            kw["exposure_steps_ms"] = tuple(float(s) for s in kw["exposure_steps_ms"])
        return cls(**kw)

    @property
    def nonzero_steps(self) -> Tuple[float, ...]:
        return tuple(s for s in self.test_steps_ms if s != 0.0)

    @property
    def exposure_step_list(self) -> Tuple[float, ...]:
        return self.exposure_steps_ms if self.exposure_steps_ms is not None else self.nonzero_steps


def load_preset(path) -> Tuple[Config, ExposureConfig]:
    """A preset is one JSON file with a `config` section and an `exposure` section.

    Top-level keys beginning with an underscore are ignored, so a preset can carry its own
    explanation next to its values instead of in a separate document.
    """
    raw = {k: v for k, v in json.loads(Path(path).read_text()).items() if not k.startswith("_")}
    if "config" not in raw or "exposure" not in raw:
        raise ValueError(f"{path}: an exposure preset needs both a 'config' and an 'exposure' section")
    return Config.from_dict(raw["config"]), ExposureConfig.from_dict(raw["exposure"])


def check(cfg: Config, ecfg: ExposureConfig) -> None:
    """Refuse combinations that would quietly change what the experiment measures."""
    errs = []
    if not cfg.matched_incidence:
        errs.append("matched_incidence must be True: the roving absent class is built from it")
    if ecfg.absent_class != "roving":
        errs.append(f"absent_class={ecfg.absent_class!r}: only 'roving' is envelope-matched; the "
                    f"audit rejects 'plain' and 'scattered' (README section 9)")
    if cfg.anchored_fraction != 1.0:
        errs.append(f"anchored_fraction={cfg.anchored_fraction}: this experiment needs ONE figure "
                    f"set held constant across phases, not a per-trial mixture of anchored and "
                    f"fresh targets. Set it to 1.0 so the configuration states what happens.")
    if any(s not in cfg.steps_ms for s in ecfg.test_steps_ms):
        errs.append(f"test_steps_ms {list(ecfg.test_steps_ms)} must be drawn from the validated "
                    f"ladder {list(cfg.steps_ms)}, which is what the span budget was checked for")
    if any(s not in cfg.steps_ms for s in ecfg.exposure_step_list):
        errs.append(f"exposure_steps_ms {list(ecfg.exposure_step_list)} must be drawn from "
                    f"{list(cfg.steps_ms)}")
    if 0.0 in ecfg.exposure_step_list:
        errs.append("exposure at step 0 carries no order information at all: the components are "
                    "simultaneous, so there is no sequence to learn")
    if not ecfg.nonzero_steps:
        errs.append("test_steps_ms contains no nonzero delay, so D is not defined anywhere")
    if ecfg.trained not in ("P", "Q", "auto"):
        errs.append("trained must be 'P', 'Q' or 'auto'")
    if not (0.0 < ecfg.exposure_present_fraction < 1.0):
        errs.append("exposure_present_fraction must be strictly between 0 and 1")
    if cfg.n_components < 3:
        errs.append("two orders over fewer than three components are not meaningfully different")
    if errs:
        raise ValueError("invalid exposure configuration:\n  - " + "\n  - ".join(errs))


# ----------------------------------------------------------------------------
# the two orders
# ----------------------------------------------------------------------------
def heard_order(order: Sequence[int]) -> List[int]:
    """`order[i]` is the onset slot of component i, so the sequence heard is its inverse."""
    return [int(i) for i in np.argsort(np.asarray(order, dtype=int))]


def directed_transitions(order: Sequence[int]) -> List[Tuple[int, int]]:
    """The component-to-component steps heard in one element, INCLUDING the wrap into the next.

    The element repeats every inter-element interval, so the last component of one repetition is
    followed by the first component of the next; that transition is part of what is on offer to
    learn and is counted here. With n components this is a directed cycle of n arcs.
    """
    s = heard_order(order)
    n = len(s)
    return [(s[j], s[(j + 1) % n]) for j in range(n)]


def _direction_changes(order: Sequence[int]) -> int:
    """How often the heard contour reverses direction; components are indexed by frequency."""
    s = heard_order(order)
    d = np.sign(np.diff(s))
    return int(np.sum(d[1:] != d[:-1]))


def order_overlap(p: Sequence[int], q: Sequence[int]) -> dict:
    """Everything the two orders still share. Different permutations are NOT independent."""
    tp, tq = set(directed_transitions(p)), set(directed_transitions(q))
    up = {frozenset(t) for t in tp}
    uq = {frozenset(t) for t in tq}
    p_, q_ = np.asarray(p), np.asarray(q)
    n = p_.size
    sp = np.array(heard_order(p)); sq = np.array(heard_order(q))
    return {
        "shared_directed_transitions": len(tp & tq),
        "n_directed_transitions": len(tp),
        "shared_undirected_adjacencies": len(up & uq),
        "components_in_the_same_slot": int(np.sum(p_ == q_)),
        "spearman_between_orders": float(np.corrcoef(p_, q_)[0, 1]),
        "direction_changes": (_direction_changes(p), _direction_changes(q)),
        "mean_abs_slot_shift": float(np.mean(np.abs(p_ - q_))),
        "kendall_tau_of_heard_sequences": float(
            (np.sum(np.sign(np.subtract.outer(sp, sp)) * np.sign(np.subtract.outer(sq, sq)))
             / (n * (n - 1)))),
    }


def choose_orders(n_components: int, seed: int) -> Tuple[Tuple[int, ...], Tuple[int, ...], dict]:
    """Two orders sharing as few directed transitions as the pool of permutations allows.

    Search is exhaustive over the second order given the first, which is cheap at these sizes.
    The strictly ascending and strictly descending orders are excluded from both: a monotone
    contour is a frequency sweep, perceptually a different kind of object from an arbitrary
    order, and making one of the pair a sweep would stop them being exchangeable. Among the
    orders that share fewest transitions the pair is chosen to match on the number of contour
    direction changes, so neither is obviously the more jagged, and then to leave as few
    components as possible starting in the same slot in both, and finally to put the rank
    correlation of the two heard sequences near zero -- a pair that is strongly anti-correlated
    is one order and its rough reverse, which is a gross contour difference and not the
    arbitrary-versus-arbitrary comparison this experiment wants.
    """
    rng = np.random.default_rng([int(seed), 0x5E9])
    ident = tuple(range(n_components))
    rev = tuple(reversed(ident))
    perms = [p for p in itertools.permutations(range(n_components)) if p not in (ident, rev)]
    p = perms[int(rng.integers(len(perms)))]
    tp = set(directed_transitions(p))
    pa = np.asarray(p)
    sp = np.array(heard_order(p))
    n = len(p)

    def tau(q):
        sq = np.array(heard_order(q))
        return float(np.sum(np.sign(np.subtract.outer(sp, sp)) * np.sign(np.subtract.outer(sq, sq)))
                     / (n * (n - 1)))

    scored = []
    for q in perms:
        if q == p:
            continue
        scored.append((len(tp & set(directed_transitions(q))),                 # the main criterion
                       abs(_direction_changes(q) - _direction_changes(p)),     # comparable contours
                       int(np.sum(np.asarray(q) == pa)),                       # leftover slot overlap
                       round(abs(tau(q)), 6),                                  # neither aligned nor reversed
                       q))
    best = min(x[:4] for x in scored)
    tier = [x[4] for x in scored if x[:4] == best]
    q = tier[int(rng.integers(len(tier)))]
    return tuple(p), tuple(q), order_overlap(p, q)


# ----------------------------------------------------------------------------
# the stimulus, and what "trained-designated absent" means
# ----------------------------------------------------------------------------
def figure_set_for(cfg: Config, d: Derived, ecfg: ExposureConfig) -> np.ndarray:
    """The one frequency set both sequences use, for the whole session.

    Drawn the same way the anchored figure is drawn, so setting `figure_set_seed` equal to
    `figure_anchor_seed` gives literally the figure the two-interval task has been using.
    """
    from .stimulus import sample_banded_figure_set, sample_figure_set
    rng = np.random.default_rng([int(ecfg.figure_set_seed), 0xF16])
    if cfg.figure_band_channels:
        return sample_banded_figure_set(rng, cfg, d.n_channels, cfg.figure_band_channels)
    return sample_figure_set(rng, d.n_channels, cfg.n_components, cfg.figure_min_spacing_channels)


def build_trial(cfg: Config, d: Derived, S: np.ndarray, order: Sequence[int], seed: int,
                step_ms: float, present: bool, absent_class: str = "roving") -> Interval:
    """One yes/no interval of a *designated* trial.

    A trial's designation is which order governs whatever is time-aligned, and the designation
    is the same on present and absent trials:

        P-designated, present  -> the fixed set S, its components starting in order P, in every
                                  element. The sequence recurs; the answer is yes.
        P-designated, absent   -> a fresh band of channels each element, its components starting
                                  in order P. A group is there and it has P's timing, but the
                                  frequencies never come back; the answer is no.

    Q-designated is the same with Q. The consequence is the point: within a designation the
    onset order is identical on yes and no trials, so **order carries no information about the
    answer** and cannot be used as a shortcut. What differs between yes and no is only whether
    the frequency set returns, which is what detection means here. Each designation therefore
    has its own false-alarm reference, measured under its own order.
    """
    if absent_class != "roving":
        raise ValueError("only the roving absent class is envelope-matched; see README section 9")
    tr = make_trial(cfg, int(seed), float(step_ms), "rising", d=d, order=order, figure_set=S)
    return tr.recurring if present else tr.other


def assign_trained(code: str, ecfg: ExposureConfig) -> str:
    """Which sequence gets the exposure phase. 'auto' counterbalances across participants."""
    if ecfg.trained in SEQUENCES:
        return ecfg.trained
    return SEQUENCES[zlib.crc32(code.strip().upper().encode()) % 2]


# ----------------------------------------------------------------------------
# the design
# ----------------------------------------------------------------------------
@dataclass
class ExpSpec:
    phase: str
    index: int
    sequence: str          # "P" or "Q"
    present: bool
    step_ms: float
    seed: int
    feedback: bool

    def as_row(self) -> dict:
        return dict(phase=self.phase, index=self.index, sequence=self.sequence,
                    present=int(self.present), step_ms=self.step_ms, seed=self.seed,
                    feedback=int(self.feedback))


def _shuffle(rng, items, max_same_answer=3, max_same_sequence=4):
    from .design import constrained_shuffle
    return constrained_shuffle(rng, list(items),
                               [(lambda t: t[1], max_same_answer),        # the yes/no answer
                                (lambda t: t[0], max_same_sequence)])     # the designation


def make_design(cfg: Config, ecfg: ExposureConfig, code: str, session_index: int) -> dict:
    """Three phases. Balanced present/absent within phase, sequence and delay; order randomised.

    Every trial gets its own stimulus seed and no seed is reused anywhere in the session, so the
    background and the element timing are freshly generated everywhere. Only the frequency set
    and the two onset orders persist -- which is the point, and the reason a post-test gain
    cannot be recognition of a remembered waveform.
    """
    check(cfg, ecfg)
    from .design import session_seed
    seed0 = session_seed(code, session_index)
    rng = np.random.default_rng(seed0)
    trained = assign_trained(code, ecfg)
    P, Q, overlap = choose_orders(cfg.n_components, ecfg.sequence_seed)
    S = figure_set_for(cfg, validate(cfg), ecfg)

    used: set = set()

    def fresh():
        while True:
            s = int(rng.integers(1, 2 ** 31 - 1))
            if s not in used:
                used.add(s)
                return s

    def test_phase(name):
        cells = [(sq, pr, st) for sq in SEQUENCES for st in ecfg.test_steps_ms for pr in (True, False)]
        items = [c for c in cells for _ in range(ecfg.test_trials_per_cell)]
        items = _shuffle(rng, items)
        return [ExpSpec(name, i, sq, pr, float(st), fresh(), ecfg.feedback_test)
                for i, (sq, pr, st) in enumerate(items)]

    pre = test_phase("pre")
    steps = list(ecfg.exposure_step_list)
    n_pres = int(round(ecfg.exposure_trials * ecfg.exposure_present_fraction))
    flags = [True] * n_pres + [False] * (ecfg.exposure_trials - n_pres)
    expo_items = [(trained, fl, steps[i % len(steps)]) for i, fl in enumerate(flags)]
    expo_items = _shuffle(rng, expo_items, max_same_sequence=len(expo_items))
    exposure = [ExpSpec("exposure", i, sq, pr, float(st), fresh(), ecfg.feedback_exposure)
                for i, (sq, pr, st) in enumerate(expo_items)]
    post = test_phase("post")

    practice = []
    if ecfg.practice_trials:
        pitems = [(SEQUENCES[i % 2], i % 2 == 0, float(ecfg.test_steps_ms[0]))
                  for i in range(ecfg.practice_trials)]
        pitems = _shuffle(rng, pitems)
        practice = [ExpSpec("practice", i, sq, pr, st, fresh(), True)
                    for i, (sq, pr, st) in enumerate(pitems)]

    dz = {
        "task": "exposure", "session_seed": seed0, "participant_code": code,
        "session_index": session_index, "trained": trained, "untrained": _other(trained),
        "orders": {"P": list(P), "Q": list(Q)},
        "heard_order": {"P": heard_order(P), "Q": heard_order(Q)},
        "order_overlap": overlap, "figure_set": [int(c) for c in S],
        "config_hash": cfg.hash(), "exposure_hash": ecfg.hash(),
        "exposure_config": ecfg.to_dict(),
        "phases": {"practice": [t.as_row() for t in practice], "pre": [t.as_row() for t in pre],
                   "exposure": [t.as_row() for t in exposure], "post": [t.as_row() for t in post]},
    }
    dz["design_hash"] = hashlib.sha256(
        json.dumps({k: dz[k] for k in ("orders", "figure_set", "trained", "phases")},
                   sort_keys=True).encode()).hexdigest()[:16]
    return dz


def _other(seq: str) -> str:
    return "Q" if seq == "P" else "P"


def specs_of(design: dict, phase: str) -> List[ExpSpec]:
    return [ExpSpec(phase=phase, index=r["index"], sequence=r["sequence"], present=bool(r["present"]),
                    step_ms=float(r["step_ms"]), seed=int(r["seed"]), feedback=bool(r["feedback"]))
            for r in design["phases"][phase]]


def duration_estimate(cfg: Config, ecfg: ExposureConfig) -> dict:
    """Wall-clock estimate, from the same per-trial budget the two-interval runner uses."""
    n_test = 2 * len(ecfg.test_steps_ms) * 2 * ecfg.test_trials_per_cell
    n = ecfg.practice_trials + 2 * n_test + ecfg.exposure_trials
    one = (cfg.interval_dur_ms + cfg.lead_silence_ms) / 1000.0 + cfg.response_allowance_s + cfg.iti_s
    fb = cfg.feedback_s
    secs = (ecfg.practice_trials * (one + fb) + 2 * n_test * (one + (fb if ecfg.feedback_test else 0))
            + ecfg.exposure_trials * (one + (fb if ecfg.feedback_exposure else 0)))
    breaks = n // max(cfg.break_every, 1) + 4
    return {"trials": n, "per_phase_test": n_test, "exposure": ecfg.exposure_trials,
            "practice": ecfg.practice_trials, "minutes": (secs + breaks * cfg.break_s) / 60.0,
            "trial_seconds": one}


# ----------------------------------------------------------------------------
# running a session
# ----------------------------------------------------------------------------
LOG_FIELDS = ["participant", "session", "phase", "trial_index", "sequence", "role", "trained",
              "order", "heard_order", "present", "absent_class", "step_ms", "n_elements",
              "figure_repeats", "stimulus_seed", "feedback", "response", "correct", "rt_ms",
              "t_start", "t_response", "cum_seq_present", "cum_seq_any",
              "config_hash", "exposure_hash", "design_hash", "source_hash"]


class _Log:
    def __init__(self, path: Path):
        new = not path.exists()
        self.f = open(path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=LOG_FIELDS)
        if new:
            self.w.writeheader(); self.f.flush(); os.fsync(self.f.fileno())

    def write(self, row: dict):
        self.w.writerow({k: row.get(k, "") for k in LOG_FIELDS})
        self.f.flush(); os.fsync(self.f.fileno())

    def close(self):
        self.f.close()


class ExposureRunner:
    """Three-phase yes/no session. The two-interval Runner and the yes/no Runner are untouched."""

    def __init__(self, cfg: Config, ecfg: ExposureConfig, data_dir, device=None,
                 audio: bool = True, auto: Optional[float] = None, fast: bool = False,
                 exposure_gain: float = 0.0):
        from .runner import Audio
        check(cfg, ecfg)
        self.cfg, self.ecfg, self.d = cfg, ecfg, validate(cfg)
        self.data_dir = Path(data_dir)
        self.audio = Audio(cfg.sample_rate, device, enabled=audio and auto is None)
        self.auto, self.fast = auto, fast
        if exposure_gain != 0.0:
            raise ValueError("the exposure phase uses the configured background and target levels; "
                             "attenuating the background or boosting the target would make it a "
                             "different stimulus from the one being tested")
        self.rng_auto = np.random.default_rng(23)

    def _answer(self, spec: ExpSpec, trained_seq: str) -> str:
        from .runner import getkey
        if self.auto is not None:
            # A simulated listener: sensitivity falls with the onset step, and the TRAINED
            # sequence gains a little after its exposure phase. This exists to exercise the
            # pipeline and the analysis, and is not a model of anything.
            dp = 2.0 * float(np.exp(-spec.step_ms / self.auto))
            if spec.phase == "post" and spec.sequence == trained_seq and spec.step_ms > 0:
                dp += 0.6
            x = self.rng_auto.normal(dp if spec.present else 0.0, 1.0)
            return "y" if x > 0.5 else "n"
        return getkey({"y", "n", "q"}, "  was the figure there? [y/n]  ")

    def run(self, code: Optional[str] = None, session_index: Optional[int] = None) -> Path:
        import time as _t
        from .runner import QuitRequested, Runner, getkey
        from .session import (next_session_index, now_iso, provenance, session_dir, source_hash,
                              upsert_participant, write_json)
        cfg, ecfg = self.cfg, self.ecfg
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
        design = make_design(cfg, ecfg, code, idx)
        S = np.array(design["figure_set"], dtype=int)
        orders = {k: tuple(v) for k, v in design["orders"].items()}
        trained = design["trained"]
        meta = {**provenance(cfg), "task": "exposure", "participant_code": code,
                "session_index": idx, "design_hash": design["design_hash"], "design": design,
                "status": "started", "phases_completed": []}
        write_json(sdir / "session.json", meta)
        est = duration_estimate(cfg, ecfg)
        print(f"\nexposure session {sdir}")
        print(f"  trained sequence: {trained} (heard order {design['heard_order'][trained]})")
        print(f"  comparison      : {design['untrained']} (heard order {design['heard_order'][design['untrained']]})")
        print(f"  {est['trials']} trials, about {est['minutes']:.0f} minutes")
        log = _Log(sdir / "trials.csv")
        cum = {s: {"present": 0, "any": 0} for s in SEQUENCES}
        try:
            for phase, blurb in (("practice", "Practice, with feedback."),
                                 ("pre", "Pre-test. One sound per trial, no feedback."),
                                 ("exposure", "Listening block. Same task."),
                                 ("post", "Post-test. Same as the pre-test.")):
                specs = specs_of(design, phase)
                if not specs:
                    continue
                self._pause(f"{blurb}  {len(specs)} trials.")
                for i, spec in enumerate(specs, 1):
                    if i > 1 and (i - 1) % cfg.break_every == 0:
                        self._pause("Take a break.")
                    iv = build_trial(cfg, self.d, S, orders[spec.sequence], spec.seed,
                                     spec.step_ms, spec.present, ecfg.absent_class)
                    x = yesno.render(cfg, self.d, iv)
                    print(f"  {phase} {i}/{len(specs)} ...", end="", flush=True)
                    t0 = _t.time()
                    if self.auto is None:
                        self.audio.play(x)
                    k = self._answer(spec, trained)
                    if k == "q":
                        raise QuitRequested()
                    correct = int((k == "y") == spec.present)
                    if spec.feedback:
                        print("   correct" if correct else "   wrong")
                    else:
                        print("")
                    cum[spec.sequence]["any"] += 1
                    cum[spec.sequence]["present"] += int(spec.present)
                    log.write(dict(
                        participant=code, session=idx, phase=phase, trial_index=spec.index,
                        sequence=spec.sequence,
                        role="trained" if spec.sequence == trained else "untrained",
                        trained=trained, order="".join(map(str, orders[spec.sequence])),
                        heard_order="".join(map(str, heard_order(orders[spec.sequence]))),
                        present=int(spec.present), absent_class=ecfg.absent_class,
                        step_ms=spec.step_ms, n_elements=cfg.n_elements,
                        figure_repeats=cfg.figure_repeats, stimulus_seed=spec.seed,
                        feedback=int(spec.feedback), response=k, correct=correct,
                        rt_ms=round((_t.time() - t0) * 1000.0, 1), t_start=now_iso(),
                        t_response=now_iso(),
                        cum_seq_present=cum[spec.sequence]["present"],
                        cum_seq_any=cum[spec.sequence]["any"],
                        config_hash=cfg.hash(), exposure_hash=ecfg.hash(),
                        design_hash=design["design_hash"], source_hash=source_hash()))
                meta["phases_completed"].append(phase)
                write_json(sdir / "session.json", meta)
            meta["status"] = "complete"
        except QuitRequested:
            meta["status"] = "interrupted"
            print("\nstopped. The phases finished so far are recorded and analysable.")
        meta["cumulative_presentations"] = cum
        write_json(sdir / "session.json", meta)
        log.close()
        print(analyse([sdir]))
        return sdir

    def _pause(self, msg: str):
        from .runner import getkey
        print("\n" + msg)
        if self.auto is None:
            getkey({" "}, "  press space to go on  ")


# ----------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------
def read_session(sdir) -> Tuple[List[dict], dict]:
    sdir = Path(sdir)
    rows = list(csv.DictReader(open(sdir / "trials.csv"))) if (sdir / "trials.csv").exists() else []
    meta = json.loads((sdir / "session.json").read_text()) if (sdir / "session.json").exists() else {}
    return rows, meta


def _counts(rows: Sequence[dict]) -> Tuple[int, int, int, int]:
    """(hits, n_signal, false alarms, n_noise)."""
    sig = [r for r in rows if r["present"] == "1"]
    noi = [r for r in rows if r["present"] == "0"]
    return (sum(1 for r in sig if r["response"] == "y"), len(sig),
            sum(1 for r in noi if r["response"] == "y"), len(noi))


def _corrected(x: int, n: int) -> float:
    """Log-linear correction (Hautus 1995): (x + 0.5) / (n + 1).

    Applied to EVERY cell, not only the extreme ones, so cells are treated identically and a
    hit rate of 1.0 does not become an infinite d'. It is also what the bootstrap resamples
    from, which is what stops a ceiling cell producing a zero-width interval.
    """
    return (x + 0.5) / (n + 1.0) if n > 0 else float("nan")


def cell_table(rows: Sequence[dict]) -> List[dict]:
    """One row per phase x role x delay, with counts, rates, d', c and a bootstrap interval."""
    out = []
    keys = sorted({(r["phase"], r["role"], float(r["step_ms"])) for r in rows
                   if r["phase"] in ("pre", "post")}, key=lambda t: (t[0] != "pre", t[1], t[2]))
    for phase, role, step in keys:
        sel = [r for r in rows if r["phase"] == phase and r["role"] == role
               and float(r["step_ms"]) == step]
        h, ns, f, nn = _counts(sel)
        rec = dict(phase=phase, role=role, step_ms=step, hits=h, n_signal=ns,
                   false_alarms=f, n_noise=nn, n=len(sel))
        if ns and nn:
            rec["hit_rate"], rec["fa_rate"] = h / ns, f / nn
            rec["dprime"] = yesno.dprime_yesno(h / ns, f / nn, ns, nn)
            rec["criterion"] = yesno.criterion_yesno(h / ns, f / nn, ns, nn)
            lo, hi = _boot_cell(h, ns, f, nn)
            rec["ci"] = (lo, hi)
        else:
            rec.update(hit_rate=float("nan"), fa_rate=float("nan"), dprime=float("nan"),
                       criterion=float("nan"), ci=(float("nan"), float("nan")))
        out.append(rec)
    return out


def _boot_cell(h, ns, f, nn, n_boot=4000, seed=17):
    rng = np.random.default_rng(seed)
    ph, pf = _corrected(h, ns), _corrected(f, nn)
    hb, fb = rng.binomial(ns, ph, n_boot), rng.binomial(nn, pf, n_boot)
    dd = np.array([yesno.dprime_yesno(a / ns, b / nn, ns, nn) for a, b in zip(hb, fb)])
    return float(np.quantile(dd, 0.025)), float(np.quantile(dd, 0.975))


def difference_in_differences(rows: Sequence[dict], ecfg: ExposureConfig,
                              n_boot: Optional[int] = None, seed: int = 29) -> dict:
    """D at each nonzero delay, and the pre-declared aggregate, from ONE joint resampling.

    The eight counts that enter D at a delay -- hits and false alarms, in each of pre and post,
    for each of trained and untrained -- are resampled together on every bootstrap draw, and D
    is recomputed from the resampled counts. That is the interval for the contrast. Subtracting
    the endpoints of four separately computed d' intervals is not, and would be both wrong and
    conservative in an uncontrolled way.
    """
    n_boot = n_boot or ecfg.n_boot
    rng = np.random.default_rng(seed)
    out = {"per_delay": {}, "aggregate": None, "aggregate_rule": ecfg.aggregate,
           "n_boot": n_boot, "not_estimable": []}
    draws = {}
    for step in ecfg.nonzero_steps:
        cells = {}
        ok = True
        for phase in ("pre", "post"):
            for role in ("trained", "untrained"):
                sel = [r for r in rows if r["phase"] == phase and r["role"] == role
                       and float(r["step_ms"]) == step]
                h, ns, f, nn = _counts(sel)
                cells[(phase, role)] = (h, ns, f, nn)
                if ns == 0 or nn == 0:
                    ok = False
        if not ok:
            out["not_estimable"].append({"step_ms": step, "cells": {f"{p}/{r}": cells[(p, r)]
                                                                   for p, r in cells}})
            continue
        point = {k: yesno.dprime_yesno(h / ns, f / nn, ns, nn) if ns and nn else float("nan")
                 for k, (h, ns, f, nn) in cells.items()}
        D = ((point[("post", "trained")] - point[("pre", "trained")])
             - (point[("post", "untrained")] - point[("pre", "untrained")]))
        boot = np.empty(n_boot)
        for b in range(n_boot):
            dp = {}
            for k, (h, ns, f, nn) in cells.items():
                hb = rng.binomial(ns, _corrected(h, ns))
                fb = rng.binomial(nn, _corrected(f, nn))
                dp[k] = yesno.dprime_yesno(hb / ns, fb / nn, ns, nn)
            boot[b] = ((dp[("post", "trained")] - dp[("pre", "trained")])
                       - (dp[("post", "untrained")] - dp[("pre", "untrained")]))
        draws[step] = boot
        out["per_delay"][step] = {
            "D": D, "ci": (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))),
            "p_two_sided": float(2 * min((boot <= 0).mean(), (boot >= 0).mean())),
            "cells": {f"{p}/{r}": {"hits": cells[(p, r)][0], "n_signal": cells[(p, r)][1],
                                   "false_alarms": cells[(p, r)][2], "n_noise": cells[(p, r)][3],
                                   "dprime": point[(p, r)]} for p, r in cells},
            "gain_trained": point[("post", "trained")] - point[("pre", "trained")],
            "gain_untrained": point[("post", "untrained")] - point[("pre", "untrained")]}
    if draws and ecfg.aggregate == "mean_over_nonzero_delays":
        agg = np.mean(np.vstack([draws[s] for s in sorted(draws)]), axis=0)
        point = float(np.mean([out["per_delay"][s]["D"] for s in sorted(draws)]))
        out["aggregate"] = {"D": point,
                            "ci": (float(np.quantile(agg, 0.025)), float(np.quantile(agg, 0.975))),
                            "p_two_sided": float(2 * min((agg <= 0).mean(), (agg >= 0).mean())),
                            "delays": sorted(draws)}
    return out


def step_zero_check(rows: Sequence[dict]) -> Optional[dict]:
    """At step 0 the two orders are the SAME sound, so trained and untrained must not differ.

    This is a manipulation check on the whole apparatus, not a result. If it separates, either
    the labelling is wrong or something other than the order differs between the designations.
    """
    sel = [r for r in rows if float(r["step_ms"]) == 0.0 and r["phase"] in ("pre", "post")]
    if not sel:
        return None
    out = {}
    for role in ("trained", "untrained"):
        h, ns, f, nn = _counts([r for r in sel if r["role"] == role])
        out[role] = dict(hits=h, n_signal=ns, false_alarms=f, n_noise=nn,
                         dprime=yesno.dprime_yesno(h / ns, f / nn, ns, nn) if ns and nn else float("nan"))
    out["difference"] = out["trained"]["dprime"] - out["untrained"]["dprime"]
    return out


def analyse(sdirs: Sequence, ecfg: Optional[ExposureConfig] = None) -> str:
    """Per-session tables and contrasts, plus a participant-level summary when there are several."""
    L: List[str] = ["", "=" * 92, "EXPOSURE EXPERIMENT", "=" * 92]
    per_participant: Dict[str, Dict[float, float]] = {}
    aggregates: Dict[str, float] = {}
    assignment: Dict[str, str] = {}
    for sdir in sdirs:
        rows, meta = read_session(sdir)
        design = meta.get("design", {})
        e = ecfg or (ExposureConfig.from_dict(design["exposure_config"])
                     if "exposure_config" in design else ExposureConfig())
        code = meta.get("participant_code", str(sdir))
        L.append(f"\n--- {sdir}")
        if not rows:
            L.append("    no trials recorded."); continue
        done = meta.get("phases_completed", [])
        status = meta.get("status", "unknown")
        L.append(f"    participant {code}   status {status}   phases completed: "
                 f"{', '.join(done) if done else 'none'}")
        if status != "complete":
            L.append("    INTERRUPTED. Cells below are whatever was recorded; a contrast needs "
                     "both pre and post, and is skipped where either is missing.")
        assignment[code] = design.get("trained", "?")
        L.append(f"    trained {design.get('trained','?')}  "
                 f"heard order {design.get('heard_order',{}).get(design.get('trained','P'),'?')}"
                 f"   |  comparison {design.get('untrained','?')}  "
                 f"heard order {design.get('heard_order',{}).get(design.get('untrained','Q'),'?')}")
        ov = design.get("order_overlap", {})
        if ov:
            L.append(f"    order overlap: {ov['shared_directed_transitions']}/"
                     f"{ov['n_directed_transitions']} shared directed transitions, "
                     f"{ov['shared_undirected_adjacencies']} shared adjacencies, "
                     f"{ov['components_in_the_same_slot']} components in the same slot, "
                     f"tau {ov['kendall_tau_of_heard_sequences']:+.2f}")
        cum = meta.get("cumulative_presentations")
        if cum:
            L.append("    cumulative presentations in this session (BOTH sequences are presented "
                     "during testing; only one gets the exposure phase):")
            for s in SEQUENCES:
                L.append(f"        {s}: {cum[s]['any']} trials, {cum[s]['present']} of them with "
                         f"the figure actually present")

        L.append("\n    phase  role        delay    hits      false alarms      d'          95% CI"
                 "            c")
        for c in cell_table(rows):
            if c["n_signal"] and c["n_noise"]:
                L.append(f"    {c['phase']:<6} {c['role']:<11} {c['step_ms']:>4.4g}  "
                         f"{c['hits']:>3}/{c['n_signal']:<4} {c['false_alarms']:>8}/{c['n_noise']:<5} "
                         f"{c['dprime']:>+8.2f}   [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]   "
                         f"{c['criterion']:>+6.2f}")
            else:
                L.append(f"    {c['phase']:<6} {c['role']:<11} {c['step_ms']:>4.4g}  "
                         f"EMPTY CELL ({c['n_signal']} present, {c['n_noise']} absent trials)")

        z = step_zero_check(rows)
        if z and not np.isnan(z["difference"]):
            L.append(f"\n    manipulation check at step 0, where P and Q are the same sound: "
                     f"d' trained {z['trained']['dprime']:+.2f} vs untrained "
                     f"{z['untrained']['dprime']:+.2f}, difference {z['difference']:+.2f}")
            L.append("    a large difference here is a fault, not a finding: nothing distinguishes "
                     "the designations at zero onset separation.")

        did = difference_in_differences(rows, e)
        L.append("\n    D = (post-pre | trained) - (post-pre | untrained), at each nonzero delay")
        L.append("    resampled jointly over all eight counts; NOT by subtracting separate intervals")
        for step in sorted(did["per_delay"]):
            r = did["per_delay"][step]
            L.append(f"      {step:>5.4g} ms   D = {r['D']:+.2f}  [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]"
                     f"   p = {r['p_two_sided']:.3f}    (trained gain {r['gain_trained']:+.2f}, "
                     f"comparison gain {r['gain_untrained']:+.2f})")
            per_participant.setdefault(code, {})[step] = r["D"]
        for ne in did["not_estimable"]:
            L.append(f"      {ne['step_ms']:>5.4g} ms   NOT ESTIMABLE (an empty cell): {ne['cells']}")
        if did["aggregate"]:
            a = did["aggregate"]
            L.append(f"      aggregate ({did['aggregate_rule']}, fixed before running): "
                     f"D = {a['D']:+.2f}  [{a['ci'][0]:+.2f}, {a['ci'][1]:+.2f}]  p = {a['p_two_sided']:.3f}")
            aggregates[code] = a["D"]

    if len(assignment) > 1:
        n_p = sum(1 for v in assignment.values() if v == "P")
        n_q = sum(1 for v in assignment.values() if v == "Q")
        L.append("\n" + "-" * 92)
        L.append(f"training assignment across the sessions analysed: {n_p} trained on P, "
                 f"{n_q} trained on Q")
        if abs(n_p - n_q) > 1:
            L.append("    NOT BALANCED. `trained: auto` is a deterministic function of the "
                     "participant code, which is reproducible but only balances in expectation.")
            L.append("    For a real sample, assign explicitly from a counterbalancing list "
                     "(--exposure-set trained='\"P\"') and record it per participant.")
            L.append("    Any property of one order rather than the other is confounded with "
                     "training while the assignment is lopsided.")
    if len(aggregates) > 1:
        from scipy import stats as _st
        vals = np.array(list(aggregates.values()))
        n = vals.size
        se = vals.std(ddof=1) / np.sqrt(n)
        tcrit = _st.t.ppf(0.975, n - 1)
        t, p = _st.ttest_1samp(vals, 0.0)
        L.append("\n" + "-" * 92)
        L.append(f"participant-level summary over {n} participants (one aggregate D each)")
        L.append(f"    mean D = {vals.mean():+.3f}   SD {vals.std(ddof=1):.3f}   "
                 f"95% CI [{vals.mean() - tcrit * se:+.3f}, {vals.mean() + tcrit * se:+.3f}]   "
                 f"t({n - 1}) = {t:.2f}, p = {p:.3f}")
        L.append("    the participant is the unit of analysis here; the within-session intervals "
                 "above describe one listener and do not generalise.")
        if n < 6:
            L.append(f"    with {n} participants this is descriptive. Do not read the p value as a test.")
    elif len(aggregates) == 1:
        L.append("\n    one session: the interval above is within-participant. A group claim needs "
                 "several participants with counterbalanced training assignment.")
    L.append("=" * 92)
    return "\n".join(L)


# ----------------------------------------------------------------------------
# acoustic audit of THIS mode's stimulus distributions
# ----------------------------------------------------------------------------
def _features(cfg, d, ecfg, S, order, step, present, n, seed0, ref):
    from . import verify as V
    rows = []
    for j in range(n):
        s = int(np.random.default_rng([seed0, int(step * 1000), int(present), j]).integers(2 ** 31 - 1))
        iv = build_trial(cfg, d, S, order, s, step, present, ecfg.absent_class)
        x = render_interval(cfg, iv, d)
        span = (cfg.n_components - 1) * step + cfg.figure_repeats * cfg.tone_dur_ms
        rows.append(V.scalar_features(V.measure_interval(cfg, d, iv, x, iv.figure_set, span, ref)))
    names = list(rows[0].keys())
    return names, np.array([[r[k] for k in names] for r in rows], float)


def audit(cfg: Config, ecfg: ExposureConfig, n_trials: int = 50, seed: int = 808,
          n_perm: int = 8000, verbose: bool = True, n_seeds: int = 2) -> dict:
    """Three questions this mode has to answer for itself.

    The yes/no audit in README section 9 does not carry over: it drew a fresh figure set on half
    its trials, and this mode holds one set fixed for a whole session. A fixed target is exactly
    the situation in which an idiosyncrasy of one particular channel set could become a cue.

    1. present vs absent, within each designation and delay -- the detection task itself;
    2. P vs Q, within present and within absent -- if any feature separates the designations,
       then a trained-versus-untrained d' difference could be acoustic rather than learned;
    3. at step 0, P and Q are built identically, so (2) there is a null check on the audit.

    Question 1 is run at `n_seeds` independent samples, because one battery is one draw and a
    single clean p-value proves less than it looks like.
    """
    from . import measure as M, verify as V
    check(cfg, ecfg)
    d = validate(cfg)
    ref = M.single_tone_reference(cfg, d, V.WIN_MS, V.HOP_MS)
    P, Q, overlap = choose_orders(cfg.n_components, ecfg.sequence_seed)
    S = figure_set_for(cfg, d, ecfg)
    orders = {"P": P, "Q": Q}
    detect, contrast, cache = {}, {}, {}
    for step in ecfg.test_steps_ms:
        for key, order in orders.items():
            ps, first = [], None
            for r in range(n_seeds):
                s0 = seed + 1000 * r + zlib.crc32(key.encode()) % 1000
                names, Fp = _features(cfg, d, ecfg, S, order, step, True, n_trials, s0, ref)
                _, Fa = _features(cfg, d, ecfg, S, order, step, False, n_trials, s0 + 1, ref)
                sep = yesno.feature_separation(names, Fp, Fa, n_perm, seed + 3)
                ps.append(sep["p_value"])
                if first is None:
                    first = (names, Fp, Fa, sep)
            names, Fp, Fa, sep = first
            cache[(step, key)] = (names, Fp, Fa)
            detect[(step, key)] = {"separation": sep, "p_values": ps,
                                   "learnt": yesno.learnt_observer(Fp, Fa)}
            if verbose:
                print(f"  step {step:g} ms, {key}: present-vs-absent p = "
                      + ", ".join(f"{x:.3f}" for x in ps))
        names = cache[(step, "P")][0]
        for cls, i in (("present", 1), ("absent", 2)):
            A, B = cache[(step, "P")][i], cache[(step, "Q")][i]
            contrast[(step, cls)] = {
                "separation": yesno.feature_separation(names, A, B, n_perm, seed + 4),
                "learnt": yesno.learnt_observer(A, B)}
        if verbose:
            print(f"  step {step:g} ms, P-vs-Q: present p="
                  f"{contrast[(step, 'present')]['separation']['p_value']:.3f}, absent p="
                  f"{contrast[(step, 'absent')]['separation']['p_value']:.3f}")
    return {"cfg": cfg, "ecfg": ecfg, "n_trials": n_trials, "detect": detect,
            "contrast": contrast, "orders": orders, "overlap": overlap,
            "figure_set": [int(c) for c in S]}


def audit_report(res: dict) -> str:
    ecfg = res["ecfg"]
    L = ["=" * 96, "EXPOSURE MODE: ACOUSTIC AUDIT", "=" * 96,
         f"{res['n_trials']} present and {res['n_trials']} absent trials per designation per delay.",
         f"figure set (channels): {res['figure_set']}",
         f"P onset order {list(res['orders']['P'])}  heard {heard_order(res['orders']['P'])}",
         f"Q onset order {list(res['orders']['Q'])}  heard {heard_order(res['orders']['Q'])}", ""]
    ov = res["overlap"]
    L.append("[0] what the two orders still share")
    for k, v in ov.items():
        L.append(f"      {k:<36} {v}")
    L.append("    Zero shared directed transitions does not make them independent; the rows above")
    L.append("    are the residual similarity, reported rather than waved away.")
    L.append("")
    L.append("[1] the detection task: present vs absent, within a designation")
    L.append("    (this is what a hit rate and its own false-alarm rate are measured against)")
    from .verify import holm
    keys1 = sorted(res["detect"])
    raw1 = [min(res["detect"][k]["p_values"]) for k in keys1]
    adj1 = holm(raw1)
    L.append(f"    {'delay':>7} {'seq':>4}   permutation p, independent samples      "
             f"{'Holm':>7} {'largest d':>10} {'learnt d':>10} {'pc':>7}")
    for k, a in zip(keys1, adj1):
        step, key = k
        r = res["detect"][k]; s, l = r["separation"], r["learnt"]
        ps = "  ".join(f"{x:.4f}" for x in r["p_values"])
        L.append(f"    {step:>7.4g} {key:>4}   {ps:<36} {a:>7.3f} {s['observed_max']:>10.2f}"
                 f" {l['dprime']:>+10.2f} {l['pc'] * 100:>6.1f}%")
    L.append("    The Holm column corrects across the rows of this table. Each permutation p is")
    L.append("    already corrected across the 65 features; it is not corrected across delays,")
    L.append("    designations and repeat samples, and at twelve tests roughly one in two audits")
    L.append("    throws a p below 0.05 with nothing wrong. Read the Holm column.")
    L.append("")
    L.append("[2] P vs Q: can anything tell the designations apart, holding present/absent fixed?")
    L.append("    If it can, a trained-minus-untrained difference need not be learning at all.")
    keys2 = sorted(res["contrast"])
    adj2 = holm([res["contrast"][k]["separation"]["p_value"] for k in keys2])
    L.append(f"    {'delay':>7} {'class':>9} {'permutation p':>15} {'Holm':>8} {'largest d':>11}"
             f" {'learnt d':>10} {'pc':>7}")
    for k, a in zip(keys2, adj2):
        step, cls = k
        r = res["contrast"][k]; s, l = r["separation"], r["learnt"]
        flag = "" if a >= 0.05 else "   <-- SEPARABLE"
        L.append(f"    {step:>7.4g} {cls:>9} {s['p_value']:>15.3f} {a:>8.3f} {s['observed_max']:>11.2f}"
                 f" {l['dprime']:>+10.2f} {l['pc'] * 100:>6.1f}%{flag}")
    if 0.0 in ecfg.test_steps_ms:
        L.append("")
        L.append("")
        L.append("    The step-0 rows of table [2] are a NULL CHECK. At zero onset separation the two")
        L.append("    designations are the same construction drawn with different seeds, so those two")
        L.append("    rows are two independent samples from one distribution and their p values are")
        L.append("    the audit's own false-positive behaviour, measured rather than assumed. Read")
        L.append("    every other row against them.")
    L.append("")
    L.append("Chance performance here is not proof that no confound exists. It says that 65 named")
    L.append("acoustic measures, and the best linear combination of them a leave-one-out classifier")
    L.append("could find at this sample size, did not separate the classes.")
    L.append("=" * 96)
    return "\n".join(L)
