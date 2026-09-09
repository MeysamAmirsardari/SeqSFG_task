"""Progressive training: learn what the figure sounds like before being tested on it.

The first pilot failed partly because the listener had no way to learn the target. This
walks the background up in stages: the figure alone, then buried a little deeper each time,
with feedback, advancing only when a criterion is met and dropping back on failure. It is a
teaching tool, not a measurement -- nothing it records is analysed.
"""
from __future__ import annotations

import math
import time
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


def run_training(cfg: Config, audio, stages: Sequence[Tuple[str, float]] = (("ungrouped", 0.0),
                                                                            ("rising", 0.0)),
                 levels_db: Sequence[float] = DEFAULT_LEVELS, per_level: int = 5,
                 criterion: int = 4, max_rounds: int = 3, seed: int = 20260909,
                 getkey=None, pause=None) -> bool:
    """Walk up the levels for each stage. Returns True if the real stimulus (0 dB) was cleared."""
    d = validate(cfg)
    rng = np.random.default_rng(seed)
    S = make_trial(cfg, 1, 0.0, "rising", d=d).recurring.figure_set
    freqs = np.round(d.channel_freqs_hz[S]).astype(int).tolist()

    print("\n" + "=" * 68)
    print("TRAINING.  Nothing here is recorded as data -- this is for your ears.")
    print("=" * 68)
    if cfg.figure_anchor_seed is not None:
        print(f"\nThe figure is the SAME in every trial. Its {len(S)} pitches, in Hz:")
        print(f"   {freqs}")
        print("Listen to it on its own first, until you can hum the shape.")
        if pause:
            pause("Press space to hear the figure alone, three times.")
        fig_only = render_interval(cfg, _subset(
            make_trial(cfg, 1, 0.0, "rising", d=d).recurring, FIGURE), d)
        for _ in range(3):
            audio.play(fig_only)
        print("That shape is what you are hunting for in every trial from now on.")

    cleared_all = True
    for si, (variant, step) in enumerate(stages, start=1):
        head = ("STAGE %d: one sound has the group, the other has NOTHING." % si
                if variant == "ungrouped" else
                "STAGE %d: BOTH sounds have a group. Which one repeats on the SAME pitches?" % si)
        print(f"\n{'-'*68}\n{head}\n{'-'*68}")
        li = 0
        rounds = 0
        while li < len(levels_db):
            gain = levels_db[li]
            tag = "the real stimulus" if gain == 0 else f"background {gain:+.0f} dB"
            if pause:
                pause(f"\nLevel {li+1} of {len(levels_db)}: {tag}. {per_level} trials. Press space.")
            n_ok = 0
            for t in range(per_level):
                tgt = int(rng.integers(1, 3))
                x = make_pair(cfg, d, int(rng.integers(1, 2**31 - 1)), step, variant, gain, tgt)
                print(f"  trial {t+1}/{per_level} ...", flush=True)
                audio.play(x)
                k = getkey({"1", "2", "q"}, "  which one? [1/2, q=quit]  ") if getkey else "q"
                if k == "q":
                    print("  training stopped.")
                    return False
                ok = int(k) == tgt
                n_ok += ok
                print("   correct" if ok else f"   wrong - it was {tgt}")
            print(f"  => {n_ok}/{per_level} at {tag}")
            if n_ok >= criterion:
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
            print(f"  STAGE {si} CLEARED at the real stimulus level.")
    print("\n" + "=" * 68)
    print("Training done." + ("  You cleared the real level -- you are ready for the session."
                              if cleared_all else
                              "  You did not clear the real level; the task may still be too hard."))
    print("=" * 68 + "\n")
    return cleared_all
