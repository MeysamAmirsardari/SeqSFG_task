"""Single-interval yes/no detection: was a figure there or not?

The two-interval task only has to make interval A and interval B indistinguishable to
everything except binding. Yes/no is stricter in a way that is easy to underestimate: the
listener hears ONE sound and answers from memory of what these sounds are usually like, so
any property whose *distribution* differs between the classes is usable. A present trial that
is on average 0.2 dB louder, or that has one more envelope burst, is a criterion. Nothing here
may differ in mean OR in spread.

That has a consequence worth stating before any code:

    A yes/no task contrasting "coherent onsets present" with "coherent onsets absent" CANNOT
    be envelope-matched, because synchrony IS an envelope event. Seven tones starting together
    concentrate the same energy into a shorter window than seven tones spread over the element,
    and no arrangement of the same tones avoids it.

So the absent class has to be chosen, not assumed, and three are implemented:

* ``roving``     - elements are present and bound, but land on a fresh band every element, so
                   nothing recurs. Both classes then contain exactly one aligned chord per
                   element and the envelope matches by construction. "Is a figure there?"
                   becomes "did one figure keep coming back?", which is the single-interval
                   form of the two-interval task. This is the default and the only one that
                   survives its own audit.
* ``scattered``  - the same channels at the same element times, components scattered inside the
                   element so nothing binds. Recurrence without binding. Envelope partly
                   matched: the chord is gone, so its transient is gone with it.
* ``plain``      - a plain cloud with no element structure at all. This is the classic SFG
                   detection task, and the audit says plainly how detectable it is without
                   hearing anything group.

Run ``seqsfg yesno-verify`` before running a listener, and read section [3].
"""
from __future__ import annotations

import time
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from . import measure as M
from .config import Config, Derived, validate
from .stimulus import (Interval, _assemble, _fill_background, make_trial, render_interval)
from .verify import (HOP_MS, WIN_MS, _fit_ridge_logistic, measure_interval, scalar_features,
                     wilson)

ABSENT_KINDS = ("roving", "scattered", "plain")


# ----------------------------------------------------------------------------
# stimulus
# ----------------------------------------------------------------------------
def _plain_like(cfg: Config, d: Derived, iv: Interval, rng: np.random.Generator) -> Interval:
    """A plain cloud on the same active channels, at the same per-channel budget, no elements."""
    active = np.unique(iv.channel)
    empty = np.zeros(0, dtype=int)
    b_onset, b_chan = _fill_background(rng, cfg, d, empty, empty, active=active)
    proto = Interval("ungrouped", iv.variant, iv.step_ms, empty, empty, np.zeros(0), empty, empty,
                     empty, iv.element_onsets.copy(), [], [], iv.figure_set.copy())
    return _assemble("ungrouped", proto, empty, empty, empty, empty, b_onset, b_chan, rng, [], [])


def build_interval(cfg: Config, d: Derived, seed: int, step_ms: float, present: bool,
                   absent: str = "roving") -> Interval:
    """One interval of a yes/no trial. `present` selects the class; `absent` selects its foil."""
    if absent not in ABSENT_KINDS:
        raise ValueError(f"absent must be one of {ABSENT_KINDS}")
    if present:
        return make_trial(cfg, seed, step_ms, "rising", d=d).recurring
    if absent == "roving":
        return make_trial(cfg, seed, step_ms, "rising", d=d).other
    if absent == "scattered":
        return make_trial(cfg, seed, step_ms, "ungrouped", d=d).other
    tr = make_trial(cfg, seed, step_ms, "rising", d=d)
    return _plain_like(cfg, d, tr.recurring, np.random.default_rng([int(seed), 0xC10D]))


def render(cfg: Config, d: Derived, iv: Interval) -> np.ndarray:
    """The interval as audio, with the lead silence the runner plays before it."""
    lead = np.zeros(cfg.ms_to_samples(cfg.lead_silence_ms), dtype=np.float32)
    return np.concatenate([lead, render_interval(cfg, iv, d)])


# ----------------------------------------------------------------------------
# statistics for a single-interval decision
# ----------------------------------------------------------------------------
def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks along the last axis (ties averaged), for a rank-sum AUC."""
    order = np.argsort(x, axis=-1, kind="stable")
    r = np.empty_like(x, dtype=float)
    idx = np.arange(x.shape[-1], dtype=float)
    np.put_along_axis(r, order, idx + 1.0, axis=-1)
    # average ties
    xs = np.take_along_axis(x, order, axis=-1)
    for i in range(x.shape[0]):
        j = 0
        row = xs[i]
        while j < row.size:
            k = j
            while k + 1 < row.size and row[k + 1] == row[j]:
                k += 1
            if k > j:
                r[i, order[i, j:k + 1]] = (j + k + 2) / 2.0
            j = k + 1
    return r


def auc_dprime(auc: np.ndarray) -> np.ndarray:
    """d' of the best criterion on a feature, from its area under the ROC (equal-variance)."""
    a = np.clip(auc, 1e-6, 1 - 1e-6)
    return np.sqrt(2.0) * stats.norm.ppf(a)


def _auc_from_ranks(R: np.ndarray, n_p: int, n_a: int) -> np.ndarray:
    return (R - n_p * (n_p + 1) / 2.0) / (n_p * n_a)


def feature_separation(names: Sequence[str], Fp: np.ndarray, Fa: np.ndarray,
                       n_perm: int = 20000, seed: int = 11) -> dict:
    """How well each single-interval feature separates present from absent, and whether ANY does.

    Two statistics per feature, because a yes/no listener can key on either: a shift of the
    distribution (rank-sum AUC on the value) and a change in its spread (rank-sum AUC on the
    absolute deviation from the pooled median). The reported d' is the larger of the two.

    The multiplicity correction is a max-statistic permutation over class labels: features are
    heavily correlated, so testing 66 of them separately at 0.05 would reject constantly.
    Ranks do not change when labels are permuted, so the whole null is one matrix product.
    """
    n_p, n_a = Fp.shape[0], Fa.shape[0]
    X = np.vstack([Fp, Fa]).T                                  # (features, trials)
    ok = np.isfinite(X).all(axis=1) & (np.nanstd(X, axis=1) > 0)
    X, names = X[ok], [n for n, k in zip(names, ok) if k]
    if X.shape[0] == 0:
        return dict(names=[], dprime=np.zeros(0), auc=np.zeros(0), p_value=1.0, n_features=0,
                    n_present=n_p, n_absent=n_a, null_q95=0.0, observed_max=0.0, kind=[])
    med = np.median(X, axis=1, keepdims=True)
    R_shift = _ranks(X)
    R_spread = _ranks(np.abs(X - med))
    rng = np.random.default_rng(seed)
    sel = np.zeros((X.shape[1], n_perm + 1), dtype=float)
    sel[:n_p, 0] = 1.0
    for b in range(1, n_perm + 1):
        sel[rng.permutation(X.shape[1])[:n_p], b] = 1.0
    best = None
    for R in (R_shift, R_spread):
        dp = np.abs(auc_dprime(_auc_from_ranks(R @ sel, n_p, n_a)))
        best = dp if best is None else np.maximum(best, dp)
    obs, null = best[:, 0], best[:, 1:].max(axis=0)
    which = np.array(["spread" if np.abs(auc_dprime(_auc_from_ranks(R_spread[i] @ sel[:, 0], n_p, n_a)))
                      >= np.abs(auc_dprime(_auc_from_ranks(R_shift[i] @ sel[:, 0], n_p, n_a)))
                      else "shift" for i in range(X.shape[0])])
    auc_shift = _auc_from_ranks(R_shift @ sel[:, 0], n_p, n_a)
    return dict(names=names, dprime=obs, auc=auc_shift, kind=which,
                p_value=float((1 + np.sum(null >= obs.max())) / (n_perm + 1)),
                null_q95=float(np.quantile(null, 0.95)), observed_max=float(obs.max()),
                n_features=len(names), n_present=n_p, n_absent=n_a)


def learnt_observer(Fp: np.ndarray, Fa: np.ndarray, lam: float = 2.0) -> dict:
    """Leave-one-out ridge logistic on the whole feature vector, scored as a yes/no observer.

    Features are centred and scaled on the TRAINING fold only, so the observer is given its best
    shot without the held-out trial leaking into its own standardisation.

    Held-out scores are thresholded at zero and turned into hits and false alarms, so the
    number reported is the d' of an ideal observer that has learnt the best linear criterion --
    the thing a listener could in principle learn over a session.
    """
    X = np.vstack([Fp, Fa])
    y = np.concatenate([np.ones(len(Fp)), -np.ones(len(Fa))])
    keep = np.isfinite(X).all(axis=0) & (X.std(axis=0) > 0)
    X = X[:, keep]
    n = X.shape[0]
    if n < 8 or X.shape[1] == 0:
        return dict(n=n, hit=0.5, fa=0.5, dprime=0.0, ci=(0.0, 0.0), p=1.0, pc=0.5)
    score = np.empty(n)
    for i in range(n):
        tr = np.delete(np.arange(n), i)
        mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-12
        Xt = np.hstack([(X[tr] - mu) / sd, np.ones((n - 1, 1))])
        Xi = np.concatenate([(X[i] - mu) / sd, [1.0]])
        w = _fit_ridge_logistic(Xt * y[tr][:, None], lam)
        score[i] = float(w @ Xi)
    hits = float(np.mean(score[y > 0] > 0))
    fas = float(np.mean(score[y < 0] > 0))
    correct = float(np.sum(score[y > 0] > 0) + np.sum(score[y < 0] <= 0))
    lo, hi = wilson(correct, n)
    return dict(n=n, hit=hits, fa=fas, dprime=dprime_yesno(hits, fas, len(Fp), len(Fa)),
                pc=correct / n, ci=(dprime_from_pc_yesno(lo), dprime_from_pc_yesno(hi)),
                p=stats.binomtest(int(correct), n, 0.5).pvalue)


def dprime_yesno(hit: float, fa: float, n_signal: int, n_noise: int) -> float:
    """d' = z(H) - z(F), log-linear corrected so 0 and 1 are finite (Hautus 1995)."""
    h = (hit * n_signal + 0.5) / (n_signal + 1.0)
    f = (fa * n_noise + 0.5) / (n_noise + 1.0)
    return float(stats.norm.ppf(h) - stats.norm.ppf(f))


def criterion_yesno(hit: float, fa: float, n_signal: int, n_noise: int) -> float:
    """c = -(z(H) + z(F))/2. Zero is unbiased; positive means a conservative 'no' bias."""
    h = (hit * n_signal + 0.5) / (n_signal + 1.0)
    f = (fa * n_noise + 0.5) / (n_noise + 1.0)
    return float(-0.5 * (stats.norm.ppf(h) + stats.norm.ppf(f)))


def dprime_from_pc_yesno(pc: float) -> float:
    """d' for an unbiased yes/no observer at proportion correct pc."""
    p = min(max(pc, 1e-6), 1 - 1e-6)
    return float(2.0 * stats.norm.ppf(p))


# ----------------------------------------------------------------------------
# the audit
# ----------------------------------------------------------------------------
HEADLINE = [("rms_db", "long-term RMS (dB)"), ("peak", "peak amplitude"),
            ("env:level_db", "envelope level (dB)"), ("env:crest", "crest factor"),
            ("env:n_bursts_3sd", "envelope peaks above 3 SD"),
            ("env:n_bursts_5sd", "envelope peaks above 5 SD"),
            ("env:burst_height_mean", "mean burst height (SD)"),
            ("env:burst_height_max", "tallest burst (SD)"),
            ("env:kurtosis", "envelope kurtosis"),
            ("env:mod_depth", "envelope modulation depth"),
            ("env:mod_3_10Hz", "modulation power, element rate band (dB)"),
            ("env:ac_peak_iei", "envelope autocorrelation at element lags"),
            ("spec_peakedness", "spectral peakedness (dB)"),
            ("count_audio_mean", "tones sounding, mean")]


@dataclass
class StepResult:
    step_ms: float
    names: List[str]
    Fp: np.ndarray
    Fa: np.ndarray
    sep: dict
    learnt: dict


def run_audit(cfg: Config, n_trials: int = 60, seed: int = 4242,
              steps: Optional[Sequence[float]] = None, absent: str = "roving",
              n_perm: int = 20000, verbose: bool = True) -> dict:
    """Build independent present and absent trials and ask whether anything separates them.

    Present and absent trials are built from DIFFERENT seeds, never as two halves of one pair,
    because that is what a listener meets: single sounds drawn independently. A paired audit
    would understate exactly the between-trial variability a yes/no criterion feeds on.
    """
    d = validate(cfg)
    ref = M.single_tone_reference(cfg, d, WIN_MS, HOP_MS)
    steps = list(cfg.steps_ms) if steps is None else list(steps)
    tag = zlib.crc32(absent.encode()) & 0xFFFF
    out_steps: List[StepResult] = []
    t0 = time.time()
    for step in steps:
        if verbose:
            print(f"  building {n_trials} present + {n_trials} absent at step {step:g} ms ...",
                  end="", flush=True)
        t1 = time.time()
        rows = {True: [], False: []}
        for present in (True, False):
            for j in range(n_trials):
                s = int(np.random.default_rng(
                    [seed, tag, int(step * 1000), int(present), j]).integers(2 ** 31 - 1))
                iv = build_interval(cfg, d, s, float(step), present, absent)
                x = render_interval(cfg, iv, d)
                span = (cfg.n_components - 1) * step + cfg.figure_repeats * cfg.tone_dur_ms
                m = measure_interval(cfg, d, iv, x, iv.figure_set, span, ref)
                rows[present].append(scalar_features(m))
        names = list(rows[True][0].keys())
        Fp = np.array([[r[k] for k in names] for r in rows[True]], float)
        Fa = np.array([[r[k] for k in names] for r in rows[False]], float)
        out_steps.append(StepResult(float(step), names, Fp, Fa,
                                    feature_separation(names, Fp, Fa, n_perm, seed + 1),
                                    learnt_observer(Fp, Fa)))
        if verbose:
            print(f" {time.time() - t1:.0f}s")
    names = out_steps[0].names
    Fp = np.vstack([s.Fp for s in out_steps])
    Fa = np.vstack([s.Fa for s in out_steps])
    pooled_sep = feature_separation(names, Fp, Fa, n_perm, seed + 2)
    # the pooled learnt observer is fitted WITHIN each step and its held-out decisions pooled,
    # because feature variances differ across steps and a single fit would learn the step
    hits = sum(s.learnt["hit"] * s.Fp.shape[0] for s in out_steps)
    fas = sum(s.learnt["fa"] * s.Fa.shape[0] for s in out_steps)
    np_, na_ = Fp.shape[0], Fa.shape[0]
    pooled_learnt = dict(n=np_ + na_, hit=hits / np_, fa=fas / na_,
                         dprime=dprime_yesno(hits / np_, fas / na_, np_, na_),
                         pc=(hits + (na_ - fas)) / (np_ + na_))
    return dict(cfg=cfg, d=d, absent=absent, n_trials=n_trials, seed=seed, steps=out_steps,
                names=names, Fp=Fp, Fa=Fa, pooled=pooled_sep, pooled_learnt=pooled_learnt,
                elapsed_s=time.time() - t0)


def report(res: dict) -> str:
    cfg, d = res["cfg"], res["d"]
    L: List[str] = []
    W = 100
    L.append("=" * W)
    L.append(f"YES/NO SINGLE-INTERVAL AUDIT   absent class = '{res['absent']}'   "
             f"{res['n_trials']} present + {res['n_trials']} absent per step")
    L.append("=" * W)
    L.append("A yes/no listener answers from ONE sound, so any property whose distribution differs")
    L.append("between the classes -- in mean OR in spread -- is a usable criterion. Everything below")
    L.append("is measured on single intervals, never on pairs.")
    L.append("")

    L.append("[1] the cues a listener would try first, pooled over the ladder")
    L.append(f"    {'feature':<40}{'present':>18}{'absent':>18}{'yes/no d':>10}")
    idx = {n: i for i, n in enumerate(res["names"])}
    sep = res["pooled"]
    dp = {n: v for n, v in zip(sep["names"], sep["dprime"])}
    for key, label in HEADLINE:
        if key not in idx:
            continue
        p, a = res["Fp"][:, idx[key]], res["Fa"][:, idx[key]]
        L.append(f"    {label:<40}{p.mean():>10.3f}+-{p.std():<6.3f}{a.mean():>10.3f}+-{a.std():<6.3f}"
                 f"{dp.get(key, 0.0):>+10.2f}")
    L.append("    mean +- SD of the single-trial distribution. Both must match: a shift is a")
    L.append("    criterion, and so is a difference in spread.")
    L.append("")

    L.append(f"[2] every one of the {sep['n_features']} features, largest yes/no d' first")
    L.append(f"    {'feature':<40}{'d':>8}{'AUC':>8}{'what differs':>14}")
    order = np.argsort(-sep["dprime"])
    for i in order[:12]:
        L.append(f"    {sep['names'][i]:<40}{sep['dprime'][i]:>8.2f}{sep['auc'][i]:>8.3f}"
                 f"{sep['kind'][i]:>14}")
    L.append("")

    L.append("[3] can ANY of them separate the classes?  max-statistic permutation over class labels")
    L.append(f"    largest d' observed           {sep['observed_max']:.3f}  ({sep['names'][int(order[0])]})")
    L.append(f"    largest under relabelling     95th percentile {sep['null_q95']:.3f}")
    L.append(f"    p (any feature separates)     {sep['p_value']:.3f}")
    ok = sep["p_value"] >= 0.05
    L.append(f"    -> {'no measured property separates the classes' if ok else 'THE CLASSES ARE SEPARABLE WITHOUT HEARING A FIGURE'}")
    L.append("")

    L.append("[4] an ideal observer that has LEARNT the best linear criterion (leave-one-out,")
    L.append("    fitted within each step, held-out decisions pooled)")
    pl = res["pooled_learnt"]
    L.append(f"    hit rate {pl['hit']:.3f}   false-alarm rate {pl['fa']:.3f}   "
             f"d' = {pl['dprime']:+.3f}   {pl['pc'] * 100:.1f}% correct")
    L.append("    This is the ceiling for a listener who never hears a group and only learns what")
    L.append("    these sounds are usually like. It must be at chance.")
    L.append("")

    L.append("[5] per step")
    L.append(f"    {'step (ms)':<12}{'permutation p':>15}{'largest d':>12}{'learnt d':>11}{'learnt hit/fa':>16}")
    for s in res["steps"]:
        L.append(f"    {s.step_ms:<12.4g}{s.sep['p_value']:>15.3f}{s.sep['observed_max']:>12.2f}"
                 f"{s.learnt['dprime']:>+11.2f}{s.learnt['hit']:>8.2f}/{s.learnt['fa']:<7.2f}")
    L.append("")
    L.append(f"built in {res['elapsed_s']:.0f}s; config hash {cfg.hash()}")
    L.append("=" * W)
    return "\n".join(L)


# ----------------------------------------------------------------------------
# the session
# ----------------------------------------------------------------------------
@dataclass
class YesNoSpec:
    index: int
    block: str
    step_ms: float
    present: bool
    seed: int


def make_yesno_design(cfg: Config, code: str, session_index: int, absent: str = "roving") -> dict:
    """Balanced present/absent at every step, shuffled so neither the class nor the step runs on.

    A run of identical answers invites a response strategy, and in yes/no -- where the listener
    sets their own criterion -- that is far more damaging than in forced choice. At most three
    consecutive trials share a class and at most two share a (step, class) cell.
    """
    from .design import constrained_shuffle, session_seed
    seed = session_seed(code, session_index)
    rng = np.random.default_rng(seed)
    cells = [(float(s), bool(p)) for s in cfg.steps_ms for p in (True, False)]
    items = [c for c in cells for _ in range(cfg.trials_per_condition)]
    items = constrained_shuffle(rng, items, [(lambda t: t[1], 3), (lambda t: t, 2)])
    main = [YesNoSpec(i, "main", s, p, int(rng.integers(1, 2 ** 31 - 1)))
            for i, (s, p) in enumerate(items)]
    ptrials = [(float(cfg.steps_ms[0]), bool(p)) for p in (True, False)] * (cfg.practice_n // 2)
    ptrials = constrained_shuffle(rng, ptrials, [(lambda t: t[1], 3)])
    practice = [YesNoSpec(i, "practice", s, p, int(rng.integers(1, 2 ** 31 - 1)))
                for i, (s, p) in enumerate(ptrials)]
    dz = {"task": "yesno", "absent": absent, "session_seed": seed,
          "practice": [vars(t) for t in practice], "main": [vars(t) for t in main]}
    import hashlib as _h
    dz["design_hash"] = _h.sha256(repr(sorted(map(str, dz["main"]))).encode()).hexdigest()[:16]
    return dz


class YesNoRunner:
    """Single-interval yes/no session. Separate from the two-interval Runner on purpose."""

    def __init__(self, cfg: Config, data_dir, device=None, audio: bool = True,
                 absent: str = "roving", auto: Optional[float] = None, fast: bool = False):
        from pathlib import Path
        from .runner import Audio
        self.cfg, self.d = cfg, validate(cfg)
        self.absent = absent
        self.data_dir = Path(data_dir)
        self.audio = Audio(cfg.sample_rate, device, enabled=audio and auto is None)
        self.auto, self.fast = auto, fast
        self.rng_auto = np.random.default_rng(11)

    # -- plumbing -----------------------------------------------------------------
    def pause(self, msg: str) -> None:
        from .runner import getkey
        print("\n" + msg)
        if self.auto is None:
            getkey({" "}, "  press space to go on  ")

    def _play(self, spec: YesNoSpec):
        iv = build_interval(self.cfg, self.d, spec.seed, spec.step_ms, spec.present, self.absent)
        return render(self.cfg, self.d, iv)

    def _answer(self, spec: YesNoSpec) -> str:
        from .runner import getkey
        if self.auto is not None:
            # a simulated listener whose sensitivity decays with the step, with a mild 'no' bias
            dp = 2.2 * float(np.exp(-spec.step_ms / self.auto))
            x = self.rng_auto.normal(dp if spec.present else 0.0, 1.0)
            return "y" if x > 0.6 else "n"
        return getkey({"y", "n", "q"}, "  was the figure there? [y/n]  ")

    def _trial(self, spec: YesNoSpec, feedback: bool, i: int, n: int) -> bool:
        import time as _t
        from .runner import QuitRequested
        from .session import now_iso
        print(f"  trial {i}/{n} ...", end="", flush=True)
        x = self._play(spec)
        t0 = _t.time()
        if self.auto is None:
            self.audio.play(x)
        t_play = now_iso()
        k = self._answer(spec)
        if k == "q":
            raise QuitRequested()
        rt = (_t.time() - t0) * 1000.0
        correct = int((k == "y") == spec.present)
        print("   correct" if correct else f"   wrong ({'it was there' if spec.present else 'it was not there'})")
        self.log.write(dict(trial_index=spec.index, block=spec.block, practice_round=0,
                            practice_stage=0, variant="yesno", step_ms=spec.step_ms,
                            target_position=1 if spec.present else 2, seed=spec.seed,
                            response=k, correct=correct, rt_ms=round(rt, 1),
                            t_start=t_play, t_response=now_iso()))
        return bool(correct)

    def run(self, code: Optional[str] = None, session_index: Optional[int] = None):
        from pathlib import Path
        from .runner import QuitRequested, Runner
        from .session import (TrialLog, next_session_index, provenance, session_dir,
                              upsert_participant, write_json)
        cfg = self.cfg
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
        design = make_yesno_design(cfg, code, idx, self.absent)
        meta = {**provenance(cfg), "task": "yesno", "absent": self.absent,
                "participant_code": code, "session_index": idx,
                "session_seed": design["session_seed"], "design_hash": design["design_hash"],
                "design": design, "status": "started"}
        write_json(sdir / "session.json", meta)
        print(f"new yes/no session {sdir} (absent='{self.absent}', design {design['design_hash']})")
        self.log = TrialLog(sdir / "trials.csv")
        try:
            self.pause("Practice, with feedback. You will hear ONE sound each time.\n"
                       "Answer 'y' if a figure kept coming back at the same pitches, 'n' if not.")
            pr = [YesNoSpec(**t) for t in design["practice"]]
            for i, s in enumerate(pr, 1):
                self._trial(s, True, i, len(pr))
            mn = [YesNoSpec(**t) for t in design["main"]]
            self.pause(f"Main block: {len(mn)} trials. Same question, no feedback.")
            for i, s in enumerate(mn, 1):
                if i > 1 and (i - 1) % cfg.break_every == 0:
                    self.pause("Take a break.")
                self._trial(s, False, i, len(mn))
            meta["status"] = "complete"
        except QuitRequested:
            meta["status"] = "interrupted"
            print("\nstopped.")
        write_json(sdir / "session.json", meta)
        self.log.close()
        print(analyse(sdir))
        return sdir


# ----------------------------------------------------------------------------
# analysis of a run session
# ----------------------------------------------------------------------------
def _boot_dprime(h: int, ns: int, f: int, nn: int, n_boot: int = 2000, seed: int = 5) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    hb = rng.binomial(ns, max(min(h / ns, 1.0), 0.0), n_boot) / ns if ns else np.zeros(n_boot)
    fb = rng.binomial(nn, max(min(f / nn, 1.0), 0.0), n_boot) / nn if nn else np.zeros(n_boot)
    dd = np.array([dprime_yesno(a, b, ns, nn) for a, b in zip(hb, fb)])
    return float(np.quantile(dd, 0.025)), float(np.quantile(dd, 0.975))


def analyse(sdir) -> str:
    """d', criterion and the bias diagnostics that matter when the listener sets the criterion."""
    import csv
    from pathlib import Path
    rows = list(csv.DictReader(open(Path(sdir) / "trials.csv")))
    main = [r for r in rows if r["block"] == "main"]
    L = ["", "=" * 78, f"YES/NO SESSION  {Path(sdir)}", "=" * 78]
    if not main:
        return "\n".join(L + ["no main-block trials recorded."])

    def split(rs):
        sig = [r for r in rs if r["target_position"] == "1"]
        noi = [r for r in rs if r["target_position"] == "2"]
        h = sum(1 for r in sig if r["response"] == "y")
        f = sum(1 for r in noi if r["response"] == "y")
        return h, len(sig), f, len(noi)

    h, ns, f, nn = split(main)
    dp = dprime_yesno(h / ns, f / nn, ns, nn)
    lo, hi = _boot_dprime(h, ns, f, nn)
    c = criterion_yesno(h / ns, f / nn, ns, nn)
    L.append(f"\n{len(main)} main trials: {ns} present, {nn} absent")
    L.append(f"  hit rate  {h}/{ns} = {h / ns:.3f}     false-alarm rate {f}/{nn} = {f / nn:.3f}")
    L.append(f"  d' = {dp:+.2f}  [{lo:+.2f}, {hi:+.2f}]       criterion c = {c:+.2f} "
             f"({'unbiased' if abs(c) < 0.25 else 'conservative' if c > 0 else 'liberal'})")
    L.append(f"  percent correct {(h + (nn - f)) / (ns + nn) * 100:.1f}%")
    L.append("  d' is the measure; percent correct is not, because the listener chooses c.")

    L.append("\nper step")
    L.append(f"  {'step (ms)':<11}{'hits':>10}{'false alarms':>15}{'d':>9}{'95% CI':>18}{'c':>8}")
    for s in sorted({float(r["step_ms"]) for r in main}):
        rs = [r for r in main if float(r["step_ms"]) == s]
        h2, ns2, f2, nn2 = split(rs)
        if not ns2 or not nn2:
            continue
        d2 = dprime_yesno(h2 / ns2, f2 / nn2, ns2, nn2)
        l2, u2 = _boot_dprime(h2, ns2, f2, nn2)
        L.append(f"  {s:<11.4g}{h2:>4}/{ns2:<5}{f2:>8}/{nn2:<6}{d2:>+9.2f}"
                 f"  [{l2:+.2f}, {u2:+.2f}]{criterion_yesno(h2 / ns2, f2 / nn2, ns2, nn2):>+8.2f}")

    L.append("\ndiagnostics")
    yes = sum(1 for r in main if r["response"] == "y")
    L.append(f"  said 'yes' on {yes}/{len(main)} trials ({yes / len(main):.2f}); "
             f"{'balanced' if abs(yes / len(main) - 0.5) < 0.12 else 'LOPSIDED -- read c above'}")
    half = len(main) // 2
    for name, rs in (("first half", main[:half]), ("second half", main[half:])):
        h3, ns3, f3, nn3 = split(rs)
        if ns3 and nn3:
            L.append(f"  {name:<12} d' = {dprime_yesno(h3 / ns3, f3 / nn3, ns3, nn3):+.2f}   "
                     f"c = {criterion_yesno(h3 / ns3, f3 / nn3, ns3, nn3):+.2f}")
    L.append("  a criterion that drifts between halves is the commonest yes/no artefact; if c moves")
    L.append("  by more than about 0.3, treat the pooled d' with suspicion and analyse by half.")
    rts = [float(r["rt_ms"]) for r in main if r["rt_ms"]]
    if rts:
        L.append(f"  response time median {np.median(rts):.0f} ms "
                 f"(a median under ~200 ms after the sound means the answer preceded the evidence)")
    L.append("=" * 78)
    return "\n".join(L)
