"""Listeners with no ears: what each listening strategy can get out of the stimulus.

They decide from the levels a trial actually has, with internal noise on every tone's level
(`noise_db`), and test the STIMULUS, not people. Each computes one number per trial and says
"yes" when it is large; sensitivity is d' = sqrt(2) z(AUC), free of any criterion.

    b_alone   follows the B tones only: the B tone (from target_from on) that stands out most
              above the median B level. The segregated listener's strategy.
    pair      compares every B tone with its own A partner (B_k - A_k): the largest difference
              against the median difference. The grouped listener's strategy.
    nearest   like pair, but compares each B with the A tone NEAREST in time. Below D = 100 ms
              that is its own A; at D = 100 ms (alternation) the A before and the A after are
              equally near and it picks one at random: the pairing a timing-only rule can recover.
    fused     hears each pair as one sound and follows its overall level.

With levels = "common" only pair-type listening works; with "independent" (Micheyl et al. 2013)
only b_alone does. The stimulus carries the same information at every delay; any effect of the
delay on people is about how they hear the sequence.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats

from .config import Config
from .stimulus import build

MODES = ("b_alone", "pair", "nearest", "fused")


def statistic(cfg: Config, tr, mode: str, rng: np.random.Generator, noise_db: float = 1.0) -> float:
    a = np.array([t.level_db for t in tr.tones if t.stream == "A"]) + rng.normal(0, noise_db, cfg.n_tones)
    b = np.array([t.level_db for t in tr.tones if t.stream == "B"]) + rng.normal(0, noise_db, cfg.n_tones)
    k0 = cfg.target_from
    if mode == "b_alone":
        x = b
    elif mode == "pair":
        x = b - a
    elif mode == "nearest":
        partner = np.arange(cfg.n_tones)
        if tr.delay_ms > cfg.soa_ms / 2 - 1e-9:          # the earlier A is as near or nearer
            tie = abs(tr.delay_ms - cfg.soa_ms / 2) < 1e-9
            earlier = rng.random(cfg.n_tones) < 0.5 if tie else np.ones(cfg.n_tones, bool)
            partner = np.where(earlier & (partner > 0), partner - 1, partner)
        x = b - a[partner]
    elif mode == "fused":
        x = 10 * np.log10(10 ** (a / 10) + 10 ** (b / 10))
    else:
        raise ValueError(mode)
    return float(np.max(x[k0:]) - np.median(x))


def d_prime(cfg: Config, delay_ms: float, mode: str, n: int = 2000, seed: int = 0,
            noise_db: float = 1.0) -> float:
    rng = np.random.default_rng(seed)
    yes, no = [], []
    for i in range(n):
        for present, out in ((True, yes), (False, no)):
            tr, _ = build(cfg, delay_ms, present, (seed, i, int(present)))
            out.append(statistic(cfg, tr, mode, rng, noise_db))
    auc = stats.mannwhitneyu(yes, no).statistic / (len(yes) * len(no))
    auc = min(max(auc, 1e-4), 1 - 1e-4)
    return float(math.sqrt(2) * stats.norm.ppf(auc))


class SimListener:
    """For simulated sessions: hears through one strategy, says yes above a criterion.

    `grouping` is for illustration only: on each trial it compares B with A (pair) with a
    probability that falls linearly from 1 at 0 ms to 0 at the largest delay, and otherwise
    follows B alone. It shows what the figure would look like if grouping fell with the delay;
    it is not a prediction."""

    def __init__(self, cfg: Config, mode: str = "pair", noise_db: float = 1.0, criterion: float = 2.5,
                 seed: int = 0):
        self.cfg, self.mode, self.noise, self.crit = cfg, mode, noise_db, criterion
        self.rng = np.random.default_rng(seed)

    def respond(self, tr) -> int:
        mode = self.mode
        if mode == "grouping":
            p_pair = 1.0 - tr.delay_ms / max(self.cfg.delays_ms)
            mode = "pair" if self.rng.random() < p_pair else "b_alone"
        return 2 if statistic(self.cfg, tr, mode, self.rng, self.noise) > self.crit else 1
