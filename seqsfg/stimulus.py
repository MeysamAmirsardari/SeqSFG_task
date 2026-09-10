"""Stimulus construction and rendering.

An *interval* is a list of tones (onset on the timing grid, channel, phase, and
labels saying whether the tone belongs to a figure element). Three kinds of
interval are built:

* ``recurring`` (A): K elements on ONE channel set S. Element k, component i is
  on channel S[i] and starts at t_k + pattern_k[i] * step.
* ``redrawn`` (B): built FROM A by swapping channel labels. Each figure tone of A
  is moved to a fresh channel set S_k, and in exchange a background tone from that
  channel takes over the figure tone's old channel. Every onset time of A is an
  onset time of B, and every channel has the same number of tones in both.
* ``ungrouped`` (C): built FROM A by moving each figure tone to a random free time
  inside its own channel. Same channels, same counts, no elements.

Per-channel budgets are fixed (cfg.tones_per_channel), tones in one channel never
overlap, and every tone has the same amplitude and duration, so the long-term
spectrum is fixed by construction in every interval and the only thing that can
differ is *when* channels sound.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import numpy as np

from .config import Config, Derived, derive, VARIANTS

BACKGROUND, FIGURE = 0, 1


class PlacementError(RuntimeError):
    """A channel budget could not be placed without overlap (should be rare; caller reseeds)."""


@dataclass
class Interval:
    role: str                       # 'recurring' | 'redrawn' | 'ungrouped'
    variant: str                    # rising | scrambled | redrawn | ungrouped (trial variant)
    step_ms: float
    onset: np.ndarray               # int grid units, shape (n_tones,)
    channel: np.ndarray             # int channel index
    phase: np.ndarray               # float radians
    kind: np.ndarray                # BACKGROUND / FIGURE
    element: np.ndarray             # element index or -1
    component: np.ndarray           # component index (rank in frequency within the element) or -1
    element_onsets: np.ndarray      # int grid units, shape (K,)  (A's schedule; shared by B and C)
    element_sets: List[np.ndarray]  # channel set of each element (empty list for 'ungrouped')
    patterns: List[np.ndarray]      # delay order per element: component i starts at pattern[i]*step
    figure_set: np.ndarray          # A's recurring set S (kept on B and C for oracle measurements)

    @property
    def n_tones(self) -> int:
        return int(self.onset.size)

    def copy(self) -> "Interval":
        return Interval(self.role, self.variant, self.step_ms, self.onset.copy(), self.channel.copy(),
                        self.phase.copy(), self.kind.copy(), self.element.copy(), self.component.copy(),
                        self.element_onsets.copy(), [s.copy() for s in self.element_sets],
                        [p.copy() for p in self.patterns], self.figure_set.copy())


@dataclass
class Trial:
    seed: int
    variant: str
    step_ms: float
    recurring: Interval             # A, the target ("kept coming back at the same pitches")
    other: Interval                 # B (redrawn) or C (ungrouped)
    n_rebuilds: int = 0             # how many reseeds it took to satisfy every constraint
    anchored: bool = True           # did this trial use the fixed, learnable figure set?


# ----------------------------------------------------------------------------
# sampling helpers
# ----------------------------------------------------------------------------
def sample_figure_set(rng: np.random.Generator, n_channels: int, n: int, spacing: int) -> np.ndarray:
    """Uniform over sorted n-subsets of range(n_channels) whose consecutive gaps are >= spacing."""
    m = n_channels - (n - 1) * (spacing - 1)
    if m < n:
        raise ValueError("no valid figure set")
    y = np.sort(rng.choice(m, size=n, replace=False))
    return y + np.arange(n) * (spacing - 1)


def sample_spaced_subpool(rng: np.random.Generator, n_channels: int, size: int, spacing: int) -> np.ndarray:
    """A subpool of `size` channels, itself spaced, so any subset of it satisfies the spacing rule."""
    return sample_figure_set(rng, n_channels, size, spacing)


def sample_redrawn_sets(rng: np.random.Generator, cfg: Config, n_channels: int, k: int,
                        max_tries: int = 20000) -> List[np.ndarray]:
    """k channel sets with |S_k & S_{k-1}| <= max_shared_consecutive and |S_k & S_j| <= max_shared_any.

    With cfg.foil_subpool_size set, all k sets are drawn from one restricted, already-spaced
    subpool, so the foil's channels recur nearly as often as the target's and the single-channel
    periodicity of the two intervals is matched.
    """
    subpool = (sample_spaced_subpool(rng, n_channels, cfg.foil_subpool_size, cfg.figure_min_spacing_channels)
               if cfg.foil_subpool_size is not None else None)
    sets: List[np.ndarray] = []
    tries = 0
    while len(sets) < k:
        tries += 1
        if tries > max_tries:
            raise PlacementError("could not draw redrawn channel sets under the sharing constraints")
        if subpool is None:
            cand = sample_figure_set(rng, n_channels, cfg.n_components, cfg.figure_min_spacing_channels)
        else:
            cand = np.sort(rng.choice(subpool, size=cfg.n_components, replace=False))
        ok = True
        for j, s in enumerate(sets):
            shared = np.intersect1d(cand, s).size
            lim = cfg.max_shared_consecutive if j == len(sets) - 1 else cfg.max_shared_any
            if shared > lim:
                ok = False
                break
        if ok:
            sets.append(cand)
    return sets


def sample_foil_universe(rng: np.random.Generator, cfg: Config, n_channels: int,
                        S: np.ndarray, size: int, max_tries: int = 400) -> np.ndarray:
    """`size` channels, mutually >= spacing apart and >= spacing from every channel of S.

    Every foil element is drawn from this universe, so no foil element can contain a channel of
    the target figure: "there is no sam in the second interval" is true by construction, not on
    average.
    """
    gap = cfg.figure_min_spacing_channels
    cand = np.array([c for c in range(n_channels)
                     if np.all(np.abs(S - c) >= gap)], dtype=int)
    if cand.size < size:
        raise PlacementError(f"only {cand.size} channels lie clear of the figure set; "
                             f"foil_universe_size={size} needs more pool")
    for _ in range(max_tries):
        # Walk the candidates in ascending order from a random start (wrapping), taking every
        # channel that clears the ones already taken. Ascending packing is near-maximal, so this
        # succeeds even when the universe needs almost every spaced slot the pool has; a random
        # permutation would not. The random start is what varies the lattice between trials.
        start = int(rng.integers(cand.size))
        order = np.concatenate([cand[start:], cand[:start]])
        taken: List[int] = []
        for c in order:
            if all(abs(int(c) - t) >= gap for t in taken):
                taken.append(int(c))
        if len(taken) >= size:
            keep = np.sort(rng.choice(len(taken), size=size, replace=False))
            return np.sort(np.array(taken, dtype=int)[keep])
    raise PlacementError(f"could not lay out {size} spaced foil-universe channels")


def sample_foil_sets_dissimilar(rng: np.random.Generator, cfg: Config, universe: np.ndarray,
                                k: int) -> List[np.ndarray]:
    """k N-subsets of `universe`, each as unlike its predecessor as the universe allows.

    Random draws are not enough: with N components out of U channels two consecutive draws share
    N^2/U by chance, and at N=7 that felt like repetition even though nothing repeated. Here each
    element is chosen greedily -- heavily penalising any channel used by the previous element,
    lightly penalising channels used earlier -- so consecutive elements are disjoint whenever the
    universe is big enough to allow it.
    """
    N, U = cfg.n_components, universe.size
    uses = np.zeros(U, dtype=float)                  # how many elements have used this channel
    last = np.full(U, -1000.0)                       # element index of its last use
    sets: List[np.ndarray] = []
    for e in range(k):
        # k*N slots over U channels forces some reuse whenever U < k*N, and no schedule can avoid
        # it: pairwise-disjoint elements would need k*n_components channels, far more ERBs than
        # hearing has. So rule the previous element out outright, then balance usage and break
        # ties at RANDOM. Deterministic tie-breaks (longest-since-used, decaying recency) cycle
        # through the universe and pile the whole unavoidable overlap onto one lag -- which
        # sounds like exactly the repetition the foil is supposed to lack.
        score = 100.0 * (last == e - 1) + 10.0 * uses + rng.uniform(0.0, 5.0, size=U)
        pick = np.argsort(score, kind="stable")[:N]
        sets.append(np.sort(universe[pick]))
        uses[pick] += 1.0
        last[pick] = e
    return sets


def sample_schedule(rng: np.random.Generator, cfg: Config) -> np.ndarray:
    """Element onsets on the grid: lead ~ U[lead_min, lead_max], IEIs ~ U[iei_min, iei_max].

    The config validator guarantees the worst case fits, so nothing is ever clipped
    or rejected here; the realized distribution is what was drawn.
    """
    g = cfg.grid_ms
    lead = rng.integers(cfg.ms_to_grid(cfg.lead_min_ms), cfg.ms_to_grid(cfg.lead_max_ms) + 1)
    ieis = rng.integers(cfg.ms_to_grid(cfg.iei_min_ms), cfg.ms_to_grid(cfg.iei_max_ms) + 1,
                        size=cfg.n_elements - 1)
    return np.concatenate([[lead], lead + np.cumsum(ieis)]).astype(int)


def sample_patterns(rng: np.random.Generator, cfg: Config, variant: str) -> List[np.ndarray]:
    n, k = cfg.n_components, cfg.n_elements
    ident = np.arange(n)
    if variant in ("rising", "ungrouped", "onechannel", "scattered"):
        return [ident.copy() for _ in range(k)]
    if variant == "scrambled":
        p = rng.permutation(n)
        return [p.copy() for _ in range(k)]
    if variant == "redrawn":
        return [rng.permutation(n) for _ in range(k)]
    raise ValueError(variant)


# ----------------------------------------------------------------------------
# per-channel placement without overlap
# ----------------------------------------------------------------------------
def _blocked_mask(existing: np.ndarray, n_onsets: int, dur: int) -> np.ndarray:
    """Boolean mask over candidate onset positions 0..n_onsets-1 that would overlap `existing`."""
    blocked = np.zeros(n_onsets, dtype=bool)
    for t in existing:
        lo, hi = max(0, t - dur + 1), min(n_onsets, t + dur)
        blocked[lo:hi] = True
    return blocked


def place_free(rng: np.random.Generator, existing: np.ndarray, n_onsets: int, dur: int, count: int) -> np.ndarray:
    """Place `count` onsets uniformly at random among positions not overlapping anything (sequential RSA)."""
    blocked = _blocked_mask(existing, n_onsets, dur)
    out = np.empty(count, dtype=int)
    for i in range(count):
        free = np.flatnonzero(~blocked)
        if free.size == 0:
            raise PlacementError("no free position left in channel")
        t = int(free[rng.integers(free.size)])
        out[i] = t
        blocked[max(0, t - dur + 1):min(n_onsets, t + dur)] = True
    return out


def _conflicts(onsets_in_channel: np.ndarray, t: int, dur: int) -> bool:
    return bool(np.any(np.abs(onsets_in_channel - t) < dur))


# ----------------------------------------------------------------------------
# interval builders
# ----------------------------------------------------------------------------
def build_recurring(rng: np.random.Generator, cfg: Config, d: Derived, step_ms: float, variant: str) -> Interval:
    """Interval A: K elements on one channel set S, background filling every channel's budget."""
    P, N, K = d.n_channels, cfg.n_components, cfg.n_elements
    D = d.tone_dur_grid
    step = cfg.ms_to_grid(step_ms)
    # An anchored figure sits on the SAME channels in every trial, so it can be learned.
    set_rng = (np.random.default_rng([int(cfg.figure_anchor_seed), 0xF16])
               if cfg.figure_anchor_seed is not None else rng)
    if variant == "onechannel":
        N = 1
        S = np.array([int(set_rng.integers(P))])
        patterns = [np.zeros(1, dtype=int) for _ in range(K)]
    else:
        S = sample_figure_set(set_rng, P, N, cfg.figure_min_spacing_channels)
        patterns = sample_patterns(rng, cfg, variant)
    t_el = sample_schedule(rng, cfg)
    R = cfg.figure_repeats
    if variant == "scattered":
        # Same channels, same recurrence, same element windows -- but the components are placed
        # at random times inside the window so they never form a coherent onset. Recurrence
        # without binding.
        # Spread over one element-duration, so the components never coincide even at step 0
        # where the 'rising' element is a chord. The span is then 2*R*D, which the validator
        # budgets for (see scattered_span_ms in config.derive).
        window = max(1, R * D)
        # One offset per (element, component) -- the R repeats stay back to back, as in every
        # other variant. Drawing per repeat would let a component collide with itself.
        offs = {(k, i): int(rng.integers(0, window)) for k in range(K) for i in range(N)}
        f_onset = np.array([t_el[k] + offs[(k, i)] + r * D
                            for k in range(K) for i in range(N) for r in range(R)], dtype=int)
    else:
        # Each component occupies R consecutive tone-slots, so the figure SUSTAINS for R*tone_dur
        # rather than being a single pip. Repeats are back-to-back, never overlapping.
        f_onset = np.array([t_el[k] + patterns[k][i] * step + r * D
                            for k in range(K) for i in range(N) for r in range(R)], dtype=int)
    f_chan = np.array([S[i] for k in range(K) for i in range(N) for r in range(R)], dtype=int)
    f_elem = np.array([k for k in range(K) for i in range(N) for r in range(R)], dtype=int)
    f_comp = np.array([i for k in range(K) for i in range(N) for r in range(R)], dtype=int)
    if f_onset.max() + D > cfg.n_grid:
        raise PlacementError("element runs past the end of the interval (validator should prevent this)")
    b_onset, b_chan = _fill_background(rng, cfg, d, f_onset, f_chan)
    proto = Interval("recurring", variant, step_ms, np.zeros(0, int), np.zeros(0, int), np.zeros(0), np.zeros(0, int),
                     np.zeros(0, int), np.zeros(0, int), t_el, [], [], S)
    return _assemble("recurring", proto, f_onset, f_chan, f_elem, f_comp, b_onset, b_chan, rng,
                     [S.copy() for _ in range(K)], patterns)


def _fill_background(rng: np.random.Generator, cfg: Config, d: Derived, fig_onset: np.ndarray,
                     fig_chan: np.ndarray, active: Optional[np.ndarray] = None):
    """Per channel, place (budget - figure count) background tones uniformly among free positions.

    Only `active` channels are filled; the rest are silent in BOTH intervals, so the active set is
    a property of the trial and never of the interval. This is what lets the pool be wide enough
    for genuinely dissimilar foil elements without the background getting any denser: density is
    len(active) * tones_per_channel, not pool size * tones_per_channel.
    """
    P, M, D = d.n_channels, cfg.tones_per_channel, d.tone_dur_grid
    n_onsets = cfg.n_grid - D + 1
    onset, chan = [], []
    for c in (range(P) if active is None else [int(x) for x in active]):
        fixed = np.sort(fig_onset[fig_chan == c])
        if fixed.size > 1 and np.min(np.diff(fixed)) < D:
            raise PlacementError("figure tones overlap within a channel (validator should prevent this)")
        need = M - fixed.size
        if need < 0:
            raise PlacementError(f"channel {c} has more figure tones than its budget")
        bg = place_free(rng, fixed, n_onsets, D, need)
        onset.extend(int(t) for t in bg); chan.extend([c] * need)
    return np.array(onset, dtype=int), np.array(chan, dtype=int)


def _assemble(role, A_like: Interval, fig_onset, fig_chan, fig_elem, fig_comp, bg_onset, bg_chan, rng,
              element_sets, patterns) -> Interval:
    n_f, n_b = fig_onset.size, bg_onset.size
    return Interval(
        role=role, variant=A_like.variant, step_ms=A_like.step_ms,
        onset=np.concatenate([fig_onset, bg_onset]), channel=np.concatenate([fig_chan, bg_chan]),
        phase=rng.uniform(0.0, 2.0 * math.pi, size=n_f + n_b),
        kind=np.concatenate([np.full(n_f, FIGURE), np.full(n_b, BACKGROUND)]),
        element=np.concatenate([fig_elem, np.full(n_b, -1)]), component=np.concatenate([fig_comp, np.full(n_b, -1)]),
        element_onsets=A_like.element_onsets.copy(), element_sets=element_sets, patterns=patterns,
        figure_set=A_like.figure_set.copy(),
    )


def build_redrawn(rng: np.random.Generator, cfg: Config, d: Derived, A: Interval) -> Interval:
    """Interval B by the SAME procedure as A, sharing A's element schedule and delay patterns,
    with the channel set redrawn for every element. Nothing else is inherited from A."""
    P, K = d.n_channels, cfg.n_elements
    sets = sample_redrawn_sets(rng, cfg, P, K)
    fig = np.flatnonzero(A.kind == FIGURE)
    f_onset, f_elem, f_comp = A.onset[fig], A.element[fig], A.component[fig]
    f_chan = np.array([sets[k][i] for k, i in zip(f_elem, f_comp)], dtype=int)
    if f_onset.max() + d.tone_dur_grid > cfg.n_grid:
        raise PlacementError("element runs past the end of the interval")
    b_onset, b_chan = _fill_background(rng, cfg, d, f_onset, f_chan)
    return _assemble("redrawn", A, f_onset, f_chan, f_elem, f_comp, b_onset, b_chan, rng,
                     [s.copy() for s in sets], [p.copy() for p in A.patterns])


def build_ungrouped(rng: np.random.Generator, cfg: Config, d: Derived, A: Interval) -> Interval:
    """Interval C: a plain background, every channel at its budget, no elements at all.
    Same channels at the same rate as A (the budget fixes both); never grouped."""
    empty = np.zeros(0, dtype=int)
    b_onset, b_chan = _fill_background(rng, cfg, d, empty, empty)
    return _assemble("ungrouped", A, empty, empty, empty, empty, b_onset, b_chan, rng, [], [])


# ----------------------------------------------------------------------------
# matched-incidence pair (the two intervals differ ONLY in what is time-aligned)
# ----------------------------------------------------------------------------
def build_matched(role: str, rng: np.random.Generator, cfg: Config, d: Derived, step_ms: float,
                  variant: str, t_el: np.ndarray, S: np.ndarray, foil_sets: List[np.ndarray],
                  patterns: List[np.ndarray], active: np.ndarray, aligned: str) -> Interval:
    """One interval of a matched pair. `aligned` is 'S', 'foil' or 'none'.

    Every element k puts one tone on each channel of S AND one on each channel of foil_sets[k],
    in EVERY interval. What differs is which of the two is time-aligned into a group: the aligned
    set starts at t_k + pattern[i]*step, the other is scattered inside the element window and so
    never binds. Both intervals therefore have identical per-channel tone counts, identical
    channel recurrence, and identical element-rate structure -- the classic single-channel
    periodicity cue is matched by construction rather than by tuning a subpool.
    """
    N, K, D, R = cfg.n_components, cfg.n_elements, d.tone_dur_grid, cfg.figure_repeats
    step = cfg.ms_to_grid(step_ms)
    # The scattered counterpart spans exactly what the aligned group spans, so extent is
    # matched at every rung and the ONLY thing the step changes is whether the relative
    # timing is consistent from element to element. A fixed window would make the group
    # wider than its scattered counterpart at the top of the ladder -- the target would be
    # the more smeared of the two, which inverts what the ladder is supposed to measure.
    window = max(1, (N - 1) * step + R * D)
    S_by_element = [S] * K
    if aligned == "S":
        grp, scat_sets = S_by_element, (foil_sets,)
    elif aligned == "foil":
        grp, scat_sets = foil_sets, (S_by_element,)
    elif aligned == "none":
        grp, scat_sets = None, (S_by_element, foil_sets)
    else:
        raise ValueError(f"aligned must be 'S', 'foil' or 'none', not {aligned!r}")

    # The aligned group gets ONE jitter per element, shared by all of its components so they stay
    # perfectly coincident, drawn from the same distribution as the scattered offsets. Without it
    # an aligned channel's inter-onset intervals would be exactly the IEI while a scattered
    # channel's carry an extra +-window of jitter, and "the channel with the least variable IOI"
    # would separate the intervals without anyone having to hear a group.
    a_off = rng.integers(0, window, size=K)
    f_onset: List[int] = []; f_chan: List[int] = []; f_elem: List[int] = []; f_comp: List[int] = []
    if grp is not None:
        for k in range(K):
            for i in range(len(grp[k])):
                for r in range(R):
                    f_onset.append(int(t_el[k] + a_off[k] + patterns[k][i] * step + r * D))
                    f_chan.append(int(grp[k][i])); f_elem.append(k); f_comp.append(i)
    # scattered element tones: one random offset per (element, component), repeats stay adjacent
    s_onset: List[int] = []; s_chan: List[int] = []
    for sets in scat_sets:
        for k in range(K):
            for i in range(len(sets[k])):
                off = int(rng.integers(0, window))
                for r in range(R):
                    s_onset.append(int(t_el[k] + off + r * D)); s_chan.append(int(sets[k][i]))

    fig_onset = np.array(f_onset, dtype=int); fig_chan = np.array(f_chan, dtype=int)
    sc_onset = np.array(s_onset, dtype=int); sc_chan = np.array(s_chan, dtype=int)
    fixed_onset = np.concatenate([fig_onset, sc_onset]) if sc_onset.size or fig_onset.size else fig_onset
    fixed_chan = np.concatenate([fig_chan, sc_chan]) if sc_chan.size or fig_chan.size else fig_chan
    if fixed_onset.size and fixed_onset.max() + D > cfg.n_grid:
        raise PlacementError("element runs past the end of the interval (validator should prevent this)")
    b_onset, b_chan = _fill_background(rng, cfg, d, fixed_onset, fixed_chan, active=active)
    # the scattered element tones are background: they are not a group and are not labelled one
    b_onset = np.concatenate([sc_onset, b_onset]); b_chan = np.concatenate([sc_chan, b_chan])
    proto = Interval(role, variant, step_ms, np.zeros(0, int), np.zeros(0, int), np.zeros(0),
                     np.zeros(0, int), np.zeros(0, int), np.zeros(0, int), t_el, [], [], S)
    el_sets = ([] if grp is None else [np.asarray(g).copy() for g in grp])
    return _assemble(role, proto, fig_onset, fig_chan,
                     np.array(f_elem, dtype=int), np.array(f_comp, dtype=int),
                     b_onset, b_chan, rng, el_sets,
                     [] if grp is None else [p.copy() for p in patterns])


def _anchored_for(cfg: Config, seed: int) -> bool:
    """Whether this trial uses the anchored (learnable) figure set. Deterministic in the seed."""
    if cfg.figure_anchor_seed is None:
        return False
    if cfg.anchored_fraction >= 1.0:
        return True
    if cfg.anchored_fraction <= 0.0:
        return False
    return (zlib.crc32(f"anchor{int(seed)}".encode()) % 10000) < int(round(cfg.anchored_fraction * 10000))


def make_matched_trial(cfg: Config, d: Derived, rng: np.random.Generator, seed: int,
                       step_ms: float, variant: str) -> "Trial":
    P, N, K = d.n_channels, cfg.n_components, cfg.n_elements
    anchored = _anchored_for(cfg, seed)
    set_rng = np.random.default_rng([int(cfg.figure_anchor_seed), 0xF16]) if anchored else rng
    if variant == "onechannel":
        S = np.array([int(set_rng.integers(P))])
        patterns = [np.zeros(1, dtype=int) for _ in range(K)]
    else:
        S = sample_figure_set(set_rng, P, N, cfg.figure_min_spacing_channels)
        patterns = sample_patterns(rng, cfg, variant)
    U = cfg.foil_universe_size or (K * N)
    universe = sample_foil_universe(rng, cfg, P, S, U)
    foil_sets = sample_foil_sets_dissimilar(rng, cfg, universe, K)
    if variant == "onechannel":
        foil_sets = [g[:1].copy() for g in foil_sets]
    active = np.sort(np.concatenate([S, universe]))
    t_el = sample_schedule(rng, cfg)
    A = build_matched("recurring", rng, cfg, d, step_ms, variant, t_el, S, foil_sets, patterns,
                      active, aligned="S")
    B = build_matched("redrawn" if variant not in ("ungrouped", "onechannel") else "ungrouped",
                      rng, cfg, d, step_ms, variant, t_el, S, foil_sets, patterns, active,
                      aligned="none" if variant in ("ungrouped", "onechannel") else "foil")
    return Trial(seed=seed, variant=variant, step_ms=step_ms, recurring=A, other=B,
                 n_rebuilds=0, anchored=anchored)


def make_trial(cfg: Config, seed: int, step_ms: float, variant: str, max_rebuilds: int = 50,
               d: Optional[Derived] = None) -> Trial:
    """Deterministic in (cfg, seed, step_ms, variant). Reseeds on a placement failure and counts it."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    d = d or derive(cfg)
    for attempt in range(max_rebuilds):
        rng = np.random.default_rng([int(seed), attempt, 0xA5F6])
        try:
            if cfg.matched_incidence and variant != "scattered":
                tr = make_matched_trial(cfg, d, rng, seed, step_ms, variant)
                return replace(tr, n_rebuilds=attempt)
            A = build_recurring(rng, cfg, d, step_ms, variant)
            other = (build_ungrouped(rng, cfg, d, A) if variant in ("ungrouped", "onechannel")
                     else build_redrawn(rng, cfg, d, A))   # 'scattered' uses redrawn: same scatter, new channels
            return Trial(seed=seed, variant=variant, step_ms=step_ms, recurring=A, other=other, n_rebuilds=attempt)
        except PlacementError:
            continue
    raise PlacementError(f"trial seed={seed} step={step_ms} variant={variant}: {max_rebuilds} rebuilds failed")


# ----------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------
def tone_envelope(cfg: Config) -> np.ndarray:
    n = cfg.ms_to_samples(cfg.tone_dur_ms)
    r = cfg.ms_to_samples(cfg.ramp_ms)
    env = np.ones(n)
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(r) / r))
    env[:r] = ramp
    env[n - r:] = ramp[::-1]
    return env


def render_interval(cfg: Config, iv: Interval, d: Optional[Derived] = None) -> np.ndarray:
    d = d or derive(cfg)
    sr = cfg.sample_rate
    n_total = cfg.ms_to_samples(cfg.interval_dur_ms)
    env = tone_envelope(cfg)
    n_tone = env.size
    t = np.arange(n_tone) / sr
    x = np.zeros(n_total, dtype=np.float64)
    freqs = d.channel_freqs_hz
    samples_per_grid = cfg.grid_ms * sr / 1000.0
    for j in range(iv.n_tones):
        start = int(round(iv.onset[j] * samples_per_grid))
        f = freqs[iv.channel[j]]
        x[start:start + n_tone] += cfg.tone_amplitude * env * np.sin(2.0 * np.pi * f * t + iv.phase[j])
    return x.astype(np.float32)


def render_trial(cfg: Config, trial: Trial, target_position: int, d: Optional[Derived] = None) -> np.ndarray:
    """Whole trial as one buffer: lead silence, interval 1, ISI, interval 2. target_position in {1,2}."""
    d = d or derive(cfg)
    a = render_interval(cfg, trial.recurring, d)
    b = render_interval(cfg, trial.other, d)
    first, second = (a, b) if target_position == 1 else (b, a)
    lead = np.zeros(cfg.ms_to_samples(cfg.lead_silence_ms), dtype=np.float32)
    isi = np.zeros(cfg.ms_to_samples(cfg.isi_ms), dtype=np.float32)
    return np.concatenate([lead, first, isi, second])


def check_invariants(cfg: Config, trial: Trial, d: Optional[Derived] = None) -> dict:
    """Construction invariants, checked on the schedule. The battery measures the audio separately."""
    d = d or derive(cfg)
    A, O = trial.recurring, trial.other
    D = d.tone_dur_grid
    out = {}
    out["same_n_tones"] = A.n_tones == O.n_tones
    ca_ = np.bincount(A.onset, minlength=cfg.n_grid); co_ = np.bincount(O.onset, minlength=cfg.n_grid)
    out["shared_onset_fraction"] = float(np.minimum(ca_, co_).sum() / A.n_tones)
    ca = np.bincount(A.channel, minlength=d.n_channels)
    co = np.bincount(O.channel, minlength=d.n_channels)
    out["same_channel_counts"] = bool(np.array_equal(ca, co))
    # Every channel the trial uses carries exactly the budget; channels it does not use are
    # silent in BOTH intervals, so the active set is a property of the trial and never a cue.
    out["budget_exact"] = bool(np.all(ca[ca > 0] == cfg.tones_per_channel))
    out["same_active_channels"] = bool(np.array_equal(ca > 0, co > 0))

    def no_overlap(iv):
        for c in range(d.n_channels):
            o = np.sort(iv.onset[iv.channel == c])
            if o.size > 1 and np.min(np.diff(o)) < D:
                return False
        return True
    out["no_same_channel_overlap"] = no_overlap(A) and no_overlap(O)
    out["figure_tones"] = int(np.sum(A.kind == FIGURE))
    out["n_rebuilds"] = trial.n_rebuilds
    return out
