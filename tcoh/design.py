"""The order of the session: which track a trial belongs to, and where catch trials go.

Three things have to be true of the order, and each is checked by `audit_design` rather than
argued for here.

*No condition may be confounded with time.* Thresholds drift over an evening -- listeners learn
and then tire -- so a condition that happens to be run late would differ from one run early for
reasons that have nothing to do with dT. The design runs in ROUNDS: every condition contributes
exactly one track to each round, in an order redrawn per round. Each condition therefore has one
track early, one in the middle and one late, and the mean serial position of every condition is
the same to within the spread of a shuffle.

*The listener must not be able to tell which condition a trial is from and adopt a strategy for
it.* Within a round the tracks are run INTERLEAVED in small groups, so consecutive trials come
from different conditions and the delta jumps around. `max_same_condition_run` bounds how many
trials in a row may come from one track.

*A threshold has to be trusted to be a threshold.* Catch trials -- a supra-threshold shift that
anyone paying attention will get -- are sprinkled through the session at `catch_rate`. They do
NOT update the track that hosts them: a free correct answer at 45 ms would drag the track down
and the threshold with it. They are scored separately, and a listener who misses more than
`max_catch_miss_rate` of them has their session flagged.

Everything here is a deterministic function of (config, participant code, session index), so
the same participant resuming the same session gets the same order, and the design can be
printed, audited and archived before anyone hears anything.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Condition, Config, validate


@dataclass(frozen=True)
class TrackSpec:
    track_id: int
    condition: str
    round_index: int
    block_index: int
    seed: int


@dataclass(frozen=True)
class Slot:
    """One scheduled trial: which track it advances, and whether it is a catch probe."""
    index: int
    track_id: int
    condition: str
    block_index: int
    is_catch: bool


def session_seed(cfg: Config, code: str, session_index: int) -> int:
    h = hashlib.sha256(f"{cfg.hash()}|{code}|{session_index}".encode()).hexdigest()
    return int(h[:12], 16)


def choose_block_size(n_conditions: int, want: int = 3) -> int:
    """A block size that divides the conditions evenly, as near `want` as it can manage.

    An uneven split leaves one short block per round, and whichever condition lands in it every
    round is heard systematically later than the others -- which is a confound with fatigue and
    learning, the very thing the round structure exists to prevent. With a small design the
    answer is simply to interleave everything.
    """
    if n_conditions <= 6:
        return n_conditions
    exact = [b for b in range(3, 7) if n_conditions % b == 0]
    if exact:
        return min(exact, key=lambda b: (abs(b - want), -b))
    return min(range(3, 7), key=lambda b: (n_conditions % b, abs(b - want)))


def make_design(cfg: Config, code: str, session_index: int,
                tracks_per_block: Optional[int] = None) -> dict:
    d = validate(cfg)
    if tracks_per_block is None:
        tracks_per_block = cfg.tracks_per_block or choose_block_size(len(d.conditions))
    seed = session_seed(cfg, code, session_index)
    rng = np.random.default_rng(seed)
    names = [c.name for c in d.conditions]

    # Each condition contributes one track per round, and the ORDER within a round is chosen to
    # even out serial position across rounds rather than left to chance. Independent shuffles
    # leave each condition's mean position to luck -- measured at 0.21 of the session across
    # conditions, which confounds condition with learning and fatigue when the quantity of
    # interest is a ratio of thresholds. A rotation fixes it only when the conditions divide
    # evenly into blocks, which they need not; so the order is built greedily: whichever
    # condition has been latest so far goes earliest next, with a small random tiebreak so the
    # sequence is not deterministic. `audit_design` reports what was achieved.
    # What actually determines when a condition is heard is which BLOCK its track lands in --
    # the tracks inside a block are interleaved, so they all start at about the same moment.
    # Balancing therefore has to be done on block index, not on position in a list. Each round
    # the conditions are sorted by how late they have been so far, latest first, and dealt into
    # blocks; a small random tiebreak keeps the order from being deterministic.
    n_blocks = int(np.ceil(len(names) / tracks_per_block))
    score = {n: 0.0 for n in names}
    tracks: List[TrackSpec] = []
    tid = 0
    block = 0
    for r in range(cfg.tracks_per_condition):
        tie = {n: float(rng.random()) * 1e-6 for n in names}
        order = sorted(names, key=lambda n: -(score[n] + tie[n]))
        groups = _partition(order, n_blocks)
        for gi, group in enumerate(groups):
            for name in group:
                score[name] += gi / max(n_blocks - 1, 1)
            for name in rng.permutation(group):
                tracks.append(TrackSpec(tid, str(name), r, block, int(rng.integers(0, 2 ** 31))))
                tid += 1
            block += 1

    slots = _schedule(cfg, tracks, rng)
    return {"session_seed": seed, "config_hash": cfg.hash(),
            "conditions": [c.name for c in d.conditions],
            "tracks": [t.__dict__ for t in tracks],
            "n_blocks": block, "tracks_per_block": tracks_per_block,
            "slot_plan": [s.__dict__ for s in slots],
            "design_hash": _design_hash(tracks, slots)}


def _partition(items: Sequence[str], n_blocks: int) -> List[List[str]]:
    """Split into `n_blocks` groups of as near equal size as possible.

    Never leaves a block of one: a single-track block has nothing to interleave with, so the
    run-length constraint cannot be satisfied inside it and `_bounded_shuffle` would fail. With
    16 conditions in blocks of 3 the naive split is 3,3,3,3,3,1; this gives 3,3,3,3,2,2.
    """
    n = len(items)
    base, extra = divmod(n, n_blocks)
    out, i = [], 0
    for b in range(n_blocks):
        k = base + (1 if b < extra else 0)
        out.append(list(items[i:i + k]))
        i += k
    return out


def _schedule(cfg: Config, tracks: Sequence[TrackSpec], rng: np.random.Generator) -> List[Slot]:
    """Interleave the tracks of each block, then lay the blocks end to end.

    The plan is an upper bound on trial count: a track finishes when it converges, and the
    runner simply skips slots belonging to tracks that are already done. Planning a generous
    number of slots rather than generating the order on the fly keeps the order a property of
    the design -- printable, hashable, archived before the session -- instead of something that
    depends on how the listener happened to perform.
    """
    per_track = cfg.max_trials_per_track
    slots: List[Slot] = []
    idx = 0
    tail: Optional[str] = None       # the condition the previous block ended on
    tail_run = 0
    for b in sorted({t.block_index for t in tracks}):
        members = [t for t in tracks if t.block_index == b]
        pool: List[int] = []
        for t in members:
            pool += [t.track_id] * per_track
        by_id = {t.track_id: t for t in members}
        # a block holding a single track cannot interleave with anything, so the run
        # constraint does not apply inside it
        order = (pool if len(members) < 2
                 else _bounded_shuffle(pool, cfg.max_same_condition_run, rng,
                                       start_run=(tail, tail_run),
                                       label=lambda i: by_id[i].condition))
        for tid in order:
            is_catch = bool(rng.random() < cfg.catch_rate)
            slots.append(Slot(idx, tid, by_id[tid].condition, b, is_catch))
            idx += 1
        for s in reversed(slots):
            if s.condition != slots[-1].condition:
                break
        tail = slots[-1].condition
        tail_run = 0
        for s in reversed(slots):
            if s.condition != tail:
                break
            tail_run += 1
    return slots


def _bounded_shuffle(items: Sequence[int], max_run: int, rng: np.random.Generator,
                     max_attempts: int = 40, start_run=(None, 0), label=None) -> List[int]:
    """Shuffle so that no CONDITION repeats more than `max_run` times in a row.

    Constructive and forward-only: at each step the next item is drawn uniformly from those
    still legal. Uniform over remaining ITEMS, not over remaining values, because preferring
    whichever value has most left makes the sequence over-alternate and therefore predictable --
    the same mistake this project already made once in `seqsfg.asynchrony` and fixed there.

    `start_run` carries the run the PREVIOUS block ended on, so that a run cannot be created
    across a block boundary -- which it silently was until the design audit counted runs of 4
    under a limit of 2. `label` maps an item to the thing the constraint is about: two tracks
    of the same condition are the same thing for this purpose even though they are different
    tracks.
    """
    key = label or (lambda v: v)
    for _ in range(max_attempts):
        remaining = list(items)
        out: List[int] = []
        run_val, run_len = start_run
        ok = True
        while remaining:
            legal = [i for i, v in enumerate(remaining)
                     if not (key(v) == run_val and run_len >= max_run)]
            if not legal:
                ok = False
                break
            j = legal[int(rng.integers(len(legal)))]
            v = remaining.pop(j)
            run_len = run_len + 1 if key(v) == run_val else 1
            run_val = key(v)
            out.append(v)
        if ok:
            return out
    raise RuntimeError(f"could not order {len(items)} trials with max run {max_run}")


def _design_hash(tracks, slots) -> str:
    blob = json.dumps({"t": [t.__dict__ for t in tracks], "s": [s.__dict__ for s in slots]},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------
def audit_design(cfg: Config, design: dict) -> dict:
    """Check the three properties the docstring claims, as numbers a report can print."""
    tracks = design["tracks"]
    slots = design["slot_plan"]
    names = design["conditions"]

    # serial position of each condition's tracks, as a fraction through the session
    first_slot: Dict[int, int] = {}
    for s in slots:
        first_slot.setdefault(s["track_id"], s["index"])
    # Where each condition's TRIALS sit in the session, as a fraction of it -- not where its
    # first trial falls. The earlier version measured the first slot and divided by the last
    # track's first slot, which reported a spread near 1.0 for a design where every track was
    # interleaved through the whole session and every condition in fact averaged dead centre.
    # This is the number the surrounding comment always claimed to be about: if a condition's
    # trials cluster early and another's cluster late, a drift over the session can masquerade
    # as an effect of dT.
    n = max(len(slots), 1)
    pos: Dict[str, List[float]] = {c: [] for c in names}
    rounds: Dict[str, List[int]] = {c: [] for c in names}
    for s in slots:
        pos[s["condition"]].append(s["index"] / n)
    for t in tracks:
        rounds[t["condition"]].append(t["round_index"])
    mean_pos = {c: float(np.mean(v)) for c, v in pos.items() if v}
    mean_round = {c: float(np.mean(v)) for c, v in rounds.items() if v}

    # longest run of one condition in the planned order
    longest, cur, prev = 0, 0, None
    for s in slots:
        cur = cur + 1 if s["condition"] == prev else 1
        prev = s["condition"]
        longest = max(longest, cur)

    per_cond = {c: sum(1 for t in tracks if t["condition"] == c) for c in names}
    return {
        "n_tracks": len(tracks), "n_planned_slots": len(slots),
        "tracks_per_condition": per_cond,
        "balanced_track_count": len(set(per_cond.values())) == 1,
        "mean_serial_position": mean_pos,
        "serial_position_spread": float(max(mean_pos.values()) - min(mean_pos.values())) if mean_pos else 0.0,
        # Each condition contributes exactly one track per round, so the coarse balance -- one
        # track in each third of the session -- is exact by construction, and the residual
        # serial-position spread above is WITHIN-round only. That is the number to read when
        # asking whether a slow drift over the session could masquerade as a dT effect.
        "round_balance_spread": float(max(mean_round.values()) - min(mean_round.values())) if mean_round else 0.0,
        "longest_condition_run": longest,
        "max_same_condition_run": cfg.max_same_condition_run,
        "run_constraint_respected": longest <= cfg.max_same_condition_run,
        "catch_rate_planned": float(np.mean([s["is_catch"] for s in slots])) if slots else 0.0,
        "catch_rate_configured": cfg.catch_rate,
    }


PER_TRIAL_OVERHEAD_S = 0.55
BLOCK_BREAK_MIN = 1.5
REST_MIN = 0.75
# Measured, not guessed. Across P01's two completed sessions the mean interval between trial
# starts, excluding every pause of 30 s or more, ran 0.56 s and 0.51 s beyond the duration of
# the sound itself -- response, feedback and the next render. The constant here was 1.6 s,
# which put about eight minutes of imaginary time into a twelve-track design and would have
# had us shorten a session that already fits.


def duration_estimate(cfg: Config) -> dict:
    """How long this will really take, using the simulated track length rather than a guess."""
    from .track import simulate
    d = validate(cfg)
    sim = simulate(cfg, thresholds_ms=(3.0, 10.0, 18.0), sigmas=(0.6,), n_runs=120, seed=1)
    per_track = sim["median_trials"]
    n_trials = d.n_tracks * per_track * (1.0 + cfg.catch_rate)
    sound_s = d.trial_ms / 1000.0
    per_trial_s = sound_s + PER_TRIAL_OVERHEAD_S
    minutes = n_trials * per_trial_s / 60.0
    # Breaks are offered between BLOCKS, and -- since break_every_trials exists -- also within
    # one. Counting only the block breaks understated a design that interleaves every condition
    # at once, because that design has a single block and therefore no boundaries at all.
    n_blocks = make_design(cfg, "DURATION", 1)["n_blocks"]
    n_breaks = max(0, (n_blocks - 1) // max(cfg.break_every, 1))
    n_rests = 0
    if cfg.break_every_trials:
        n_rests = max(0, int(n_trials // cfg.break_every_trials) - n_breaks)
    breaks = n_breaks * BLOCK_BREAK_MIN + n_rests * REST_MIN
    practice = cfg.practice_trials * (sound_s + 2.5) / 60.0
    setup = 7.0     # participant panel, level calibration, instructions and familiarisation
    total = minutes + practice + breaks + setup
    worst = (d.n_tracks * cfg.max_trials_per_track * (1.0 + cfg.catch_rate) * per_trial_s / 60.0
             + practice + breaks + setup)
    return {"n_tracks": d.n_tracks, "n_blocks": n_blocks, "n_breaks": n_breaks,
            "median_trials_per_track": per_track,
            "n_trials": int(round(n_trials)), "sound_per_trial_s": sound_s,
            "n_rests": n_rests,
            "main_minutes": minutes, "practice_minutes": practice, "break_minutes": breaks,
            "setup_minutes": setup, "total_minutes": total,
            "worst_case_minutes": worst,
            "max_trials": int(d.n_tracks * cfg.max_trials_per_track * (1.0 + cfg.catch_rate)),
            "sessions_at_50_min": max(1, int(np.ceil(total / 50.0)))}
