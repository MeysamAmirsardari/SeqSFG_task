# tpair: jitter detection in irregular A-B pairs

Six pairs of tones fill a 2-second interval at an average of 3 Hz. In every pair the low tone A
starts a fixed `lag` after the high tone B, from 0 ms (synchronous) to 100 ms (the tones abut).
Each pair is jittered as a whole, so the lag inside every pair is exact while neither tone on its
own keeps a regular rhythm. In one of the two intervals, the B tone of the last or second-to-last
pair comes early by δ. The listener says which interval, and δ is tracked to threshold at each lag.

```bash
python -m tpair verify
python -m tpair run --config tpair/configs/tpair_feasibility.json --data data --code L01
python -m tpair analyze data/L01/tpair_session_01
```

## What this fixes

Both earlier tasks left the target's own rhythm intact. In the two-tone task B ran on a fixed grid;
in the sheared figure every channel repeated isochronously. Either way a listener could judge the
displaced tone against its own previous repetitions and ignore the other tone completely, which is
a strategy blind to the very thing the task manipulates.

Here that reference is randomised away. Each pair moves by U(±40 ms) as a unit, so B on its own
wanders by ±40 ms while the A-B lag stays exact. The only consistent structure in the stimulus is
the relation between the two tones.

Two model observers, deciding from the actual onsets of each trial, show what that buys:

| lag | listener using B alone | listener using the A-B lag |
|---|---|---|
| 0 ms | 43 ms | 2.4 ms |
| 25 ms | 43 ms | 9.0 ms |
| 50 ms | 43 ms | 14.6 ms |
| 100 ms | 43 ms | 27.6 ms |

The B-only listener is flat, because B's own distribution is the same whatever the lag. It cannot
see the manipulation at all. Below about 43 ms, the answer has to come through A.

## Design decisions

**The jitter is drawn independently for each interval.** Sharing it would make the two intervals
identical apart from the target, and B could then be compared across intervals directly, which
puts back exactly the cue the jitter removes. This is the logic of a level rove, applied to time.

**B moves, and only early.** Early shifts grow the lag from L to L + δ. Late shifts would shrink
it, and at δ = L the target B would land exactly on its own A, producing a synchronous chord the
standard never contains: a different cue rather than a bigger one. The two-tone pilot was
restricted to early shifts for the same reason.

**The target is in pair N or N-1, 50/50.** The listener cannot simply wait for the final tone, and
the final pair's special status (nothing follows it) is diluted rather than built into every trial.
Performance by position is reported.

**±40 ms of jitter.** Enough to put a B-only listener at 43 ms, about 1.6 times above the A-B
listener's worst lag. More jitter defends harder but erodes the pair structure at long lags.

**A 120 ms ceiling.** The B-alone control sits near 41-46 ms by design, and at a 100 ms ceiling
its upward reversals kept landing on the clamp: 3 of 4 simulated tracks came back censored. The
geometry allows up to 128 ms before two tones of one frequency come within 25 ms; 120 keeps 33.
At 120, no simulated B-alone track was censored.

**700 and 2262 Hz.** 20.3 semitones, 1.69 octaves, 9.2 ERB apart. The ratio sits 7.7% from both
3:1 and 7:2, which makes it the point above an octave furthest from every ratio with terms up to 8.

## The controls

**B alone.** The same jitter and displacement with no A. It measures what a listener ignoring A
could do, so every lag is read against it. A lag that does no better than B alone is a lag at
which A was not used.

**The strategy check.** Every trial's jitter is logged, so the analysis rebuilds how irregular B
looked *on its own* in each interval and asks whether the answers followed it, once δ is
accounted for. A listener using A should not care; one using B alone should care a lot. B-alone
trials are the positive control: there the answers must follow it, or the check has no power.

Against simulated listeners, over twelve sessions each:

| | trend fires | worst lag vs B alone | strategy check flags |
|---|---|---|---|
| uses the A-B lag | 100% | 1.59× better | 8% |
| uses B alone | 0% | 0.75×, no better | 100% |

## Conditions

| lag | model λ₂/λ₁ |
|---|---|
| 0 ms | 0.000 |
| 10 ms | 0.047 |
| 25 ms | 0.173 |
| 50 ms | 0.399 |
| 75 ms | 0.635 |
| 100 ms | 0.804 |
| B alone | |

Dense near synchrony, where every behavioural result in this repository has changed fastest. The
jitter barely moves the prediction (0.794 at 100 ms without it, 0.804 with it), because it shifts
A and B together and leaves their relation where the lag puts it.

## Session length

| preset | tracks | trials | total |
|---|---|---|---|
| `tpair_feasibility` | 7 | 303 | 37 min |
| `tpair_curve` | 14 | 607 | 66 min, two sittings |

A trial is 2.0 + 0.5 + 2.0 = 4.5 s of sound. Per-trial overhead is 0.55 s, measured across three
completed sessions. The feasibility preset has one track per condition, and a single track's 95%
interval spans about 3.6 times, so it answers whether the task works, not what the curve is.

## Known limits

* **A rise with lag does not by itself say how A was used.** Temporal coherence predicts it; so
  does any account in which a longer A-B gap is simply harder to time. The B-alone comparison
  establishes that A was used, not why it stops helping.
* **At 100 ms the tones abut,** so B's offset coincides with A's onset, and an early target opens
  a silent gap between them. Gap detection across frequency channels is poor, so this is a weak
  cue, but it exists at no other lag.
* **At long lags with a large δ and an unlucky jitter draw,** the early B can sit nearer the
  previous pair's A than its own. Inherent to moving a tone away from its partner; every trial
  records it and the analysis reports the fraction.
* **The defence has a margin, not a guarantee.** If a real listener's A-B threshold at 100 ms
  approaches 43 ms, the B-only route becomes competitive there. The strategy check is the
  safeguard, and it reports per session.
* **The model's ordering is robust; its heights are not.** All five readings of the filter bank
  agree the index rises with lag, and disagree on how much by up to 0.69.
* **Four pairs of build-up before the earliest target,** about 1.3 s. Streaming builds over
  seconds.
