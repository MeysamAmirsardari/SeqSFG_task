"""Progressive training: learn what the figure sounds like before being tested on it.

The first pilot failed partly because the listener had no way to learn the target. This
walks the background up in stages: the figure alone, then buried a little deeper each time,
with feedback, advancing only when a criterion is met and dropping back on failure. It is a
teaching tool, not a measurement -- nothing it records is analysed.
"""
from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import Config, Derived, validate
from .stimulus import BACKGROUND, FIGURE, Interval, make_trial, render_interval


def _subset(iv: Interval, keep: int) -> Interval:
    m = iv.kind == keep
    out = iv.copy()
    for a in ("onset", "channel", "phase", "kind", "element", "component"):
        setattr(out, a, getattr(iv, a)[m])
    return out


def render_with_background_gain(cfg: Config, d: Derived, iv: Interval, gain_db: float) -> np.ndarray:
    """The interval with its background attenuated. gain_db = 0 is the real stimulus."""
    fig = render_interval(cfg, _subset(iv, FIGURE), d)
    bg = render_interval(cfg, _subset(iv, BACKGROUND), d)
    return fig + bg * (10.0 ** (gain_db / 20.0))


def make_pair(cfg: Config, d: Derived, seed: int, step_ms: float, variant: str,
              gain_db: float, target_position: int) -> np.ndarray:
    """A two-interval trial with the background attenuated in BOTH intervals."""
    tr = make_trial(cfg, seed, step_ms, variant, d=d)
    a = render_with_background_gain(cfg, d, tr.recurring, gain_db)
    b = render_with_background_gain(cfg, d, tr.other, gain_db)
    first, second = (a, b) if target_position == 1 else (b, a)
    lead = np.zeros(cfg.ms_to_samples(cfg.lead_silence_ms), dtype=np.float32)
    isi = np.zeros(cfg.ms_to_samples(cfg.isi_ms), dtype=np.float32)
    return np.concatenate([lead, first, isi, second])


DEFAULT_LEVELS = (-30.0, -24.0, -18.0, -12.0, -6.0, 0.0)

STAGE_INTRO = {
    "ungrouped": "one sound has the group, the other has NOTHING.",
    "rising": "BOTH sounds have a group. Which one repeats on the SAME pitches?",
    "scrambled": "BOTH sounds have a group, in a fixed order. Which repeats on the same pitches?",
    "redrawn": "BOTH sounds have a group whose order changes. Which repeats on the same pitches?",
    "scattered": "the group never lines up. Which sound's pitches keep coming back?",
    "onechannel": "one PITCH keeps coming back, with no group at all. Which sound has it?",
}


def _demo_stage(cfg: Config, d: Derived, audio, variant: str, step: float,
                seed: int, pause=None, reps: int = 2) -> None:
    """Both streams with the noise stripped out, labelled, so the contrast is unmistakable.

    The background ladder makes the figure audible; it does not teach you WHICH difference to
    listen for. For the discrimination stages that is the whole difficulty, so hear it clean first.
    """
    tr = make_trial(cfg, seed, step, variant, d=d)
    tgt = render_interval(cfg, _subset(tr.recurring, FIGURE), d)
    foil_iv = _subset(tr.other, FIGURE)
    silent = foil_iv.n_tones == 0
    foil = render_interval(cfg, foil_iv, d)
    if pause:
        pause("Press space to hear both streams with the noise removed, "
              f"{reps} times each. [space=go, s=skip, q=quit]")
    for _ in range(reps):
        print("   [1] TARGET -- the same pitches, coming back over and over")
        audio.play(tgt)
        print("   [2] OTHER  -- " + ("no group at all (silence here)" if silent
                                     else "a new set of pitches every time"))
        audio.play(foil)
    print("   That difference is what every trial in this stage asks about.")


def run_training(cfg: Config, audio, stages: Optional[Sequence[Tuple[str, float]]] = None,
                 levels_db: Sequence[float] = DEFAULT_LEVELS, per_level: int = 5,
                 criterion: int = 4, max_rounds: int = 3, seed: int = 20260909,
                 getkey=None, pause=None, start_level: int = 0) -> bool:
    """Walk up the levels for each stage. Returns True if the real stimulus (0 dB) was cleared.

    's' skips the current level and moves on; 'q' quits. Both work at a prompt or mid-level.
    """
    if stages is None:
        stages = tuple(cfg.practice_cells)
    cfg_anchored_fraction = cfg.anchored_fraction
    # Training always uses the anchored figure, even when the session itself only anchors a
    # fraction of its trials: you cannot learn a target that changes while you are learning it.
    cfg = replace(cfg, anchored_fraction=1.0)
    d = validate(cfg)
    rng = np.random.default_rng(seed)
    S = make_trial(cfg, 1, 0.0, "rising", d=d).recurring.figure_set
    freqs = np.round(d.channel_freqs_hz[S]).astype(int).tolist()

    print("\n" + "=" * 68)
    print("TRAINING.  Nothing here is recorded as data -- this is for your ears.")
    print("=" * 68)
    if cfg.figure_anchor_seed is not None:
        print(f"\nThe figure is the SAME in every trial here. Its {len(S)} pitches, in Hz:")
        print(f"   {freqs}")
        print("Listen to it on its own first, until you can hum the shape.")
        if pause:
            pause("Press space to hear the figure alone, three times.")
        fig_only = render_interval(cfg, _subset(
            make_trial(cfg, 1, 0.0, "rising", d=d).recurring, FIGURE), d)
        for _ in range(3):
            audio.play(fig_only)
        print("That shape is what you are hunting for.")
        if cfg_anchored_fraction < 1.0:
            print(f"In the real session only {cfg_anchored_fraction:.0%} of trials use THIS figure; the rest\n"
                  "use a different one drawn fresh. The task is the same either way: which sound\n"
                  "keeps coming back on the same pitches.")

    cleared_all = True
    for si, (variant, step) in enumerate(stages, start=1):
        label = STAGE_INTRO.get(variant, variant)
        print(f"\n{'-'*68}\nSTAGE {si} of {len(stages)} ('{variant}', {step:g} ms): {label}"
              f"\n{'-'*68}")
        _demo_stage(cfg, d, audio, variant, step, int(rng.integers(1, 2**31 - 1)), pause)
        li = min(max(start_level, 0), len(levels_db) - 1)
        rounds = 0
        passed_real = False
        while li < len(levels_db):
            gain = levels_db[li]
            tag = "the real stimulus" if gain == 0 else f"background {gain:+.0f} dB"
            if pause:
                k = pause(f"\nLevel {li+1} of {len(levels_db)}: {tag}. {per_level} trials. "
                          f"[space=go, s=skip this level, q=quit]")
                if k == "q":
                    print("  training stopped.")
                    return False
                if k == "s":
                    print("  skipped.")
                    li += 1
                    rounds = 0
                    continue
            n_ok = 0
            skipped = False
            for t in range(per_level):
                tgt = int(rng.integers(1, 3))
                x = make_pair(cfg, d, int(rng.integers(1, 2**31 - 1)), step, variant, gain, tgt)
                print(f"  trial {t+1}/{per_level} ...", flush=True)
                audio.play(x)
                k = getkey({"1", "2", "s", "q"},
                           "  which one? [1/2, s=skip level, q=quit]  ") if getkey else "q"
                if k == "q":
                    print("  training stopped.")
                    return False
                if k == "s":
                    print("  skipped.")
                    skipped = True
                    break
                ok = int(k) == tgt
                n_ok += ok
                print("   correct" if ok else f"   wrong - it was {tgt}")
            if skipped:
                li += 1
                rounds = 0
                continue
            print(f"  => {n_ok}/{per_level} at {tag}")
            if n_ok >= criterion:
                passed_real = passed_real or gain == 0.0
                li += 1
                rounds = 0
                if li < len(levels_db):
                    print("  good. Making it harder.")
            else:
                rounds += 1
                if rounds >= max_rounds:
                    print(f"  stalling at {tag}. Stopping this stage here.")
                    cleared_all = False
                    break
                if li > 0:
                    li -= 1
                    print("  let's back off one level and rebuild it.")
                else:
                    print("  trying this level again.")
        else:
            if passed_real:
                print(f"  STAGE {si} CLEARED at the real stimulus level.")
            else:
                print(f"  STAGE {si} ended, but the real level was skipped rather than passed.")
                cleared_all = False
    print("\n" + "=" * 68)
    print("Training done." + ("  You cleared the real level -- you are ready for the session."
                              if cleared_all else
                              "  You did not clear the real level; the task may still be too hard."))
    print("=" * 68 + "\n")
    return cleared_all
