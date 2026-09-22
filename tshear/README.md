# tshear — jitter detection in a sheared four-tone figure

Four pure tones repeat together at 3 Hz. Within each repetition their onsets are **sheared**:
tone *k* starts *k × step* after tone 0, so the figure runs from a chord (step 0) to an even
arpeggio. On the last repetition one tone is displaced in time by δ, and the listener says
which of two intervals contained the displacement. The threshold δ is measured at each step.

```bash
python -m tshear design
python -m tshear verify
python -m tshear run --config tshear/configs/tshear_feasibility.json --data data --code L01
python -m tshear analyze data/L01/tshear_session_01
```

## Why this is cleaner than SFG

Stochastic figure-ground re-randomises the figure on every trial, so "how coherent was the
figure" is a distributional property you argue for rather than a number you set. Here the
figure is one deterministic parameter. At step 0 the model sees exactly **1.000** effective
objects; at step 100% it sees **3.839**. The axis is bounded and interpretable at both ends
with no free parameter.

| step | ms | model objects | combined onset train |
|---|---|---|---|
| 0% | 0.0 | 1.00 | all four simultaneous |
| 15% | 12.5 | 1.69 | 12/12/12/296 ms |
| 30% | 25.0 | 2.59 | 25/25/25/258 ms |
| 50% | 41.7 | 3.41 | 42/42/42/208 ms |
| 75% | 62.5 | 3.70 | 62/62/62/146 ms |
| 100% | 83.3 | 3.84 | **isochronous at 12 Hz** |

The steps are spaced by the **model**, not uniformly in milliseconds. The index moves 1.00 →
3.41 over the first half of the axis and 3.41 → 3.84 over the second; stepping uniformly would
spend half the conditions where nothing is being distinguished.

## The null hypothesis is built into the stimulus

A listener can judge the displaced tone against two things:

* **its own previous repetitions**, isochronous at 3 Hz in that one channel;
* **the other three tones** of the same repetition — the figure.

The first reference **does not depend on the step at all**. Every channel is isochronous
whatever the shear; the step changes only the relative phase of one channel against another,
and `verify` checks that on the rendered onsets. So

> a listener who ignores the figure produces a **flat** curve, necessarily.

A rise with the step cannot be produced by within-channel timing. That is the load-bearing
inference, and it is a property of the geometry rather than a control condition bolted on.

Measured against simulated listeners: a figure listener gives ρ = +0.83 and the trend test
fires on **100%** of sessions; a within-channel listener gives ρ = −0.04 and it fires on **8%**,
against a nominal 5%.

The `single` condition — the target tone alone, same rate, same jitter — measures the
within-channel limit directly, so the plateau is a number rather than an inference.

## λ₂/λ₁ does not generalise; the participation ratio does

Elhilali et al. read segregation off a two-channel coherence matrix as the ratio of its two
eigenvalues. With four channels that ratio only asks whether there is a *second* object, and
over these steps it is **not monotone** — it rises to 0.654 at 70% shear, falls to 0.615 at
80%, and rises again. An axis built on a non-monotone index cannot interpret an ordered result.

This package uses the participation ratio of the eigenvalue spectrum,

```
PR = (Σ λᵢ)² / Σ λᵢ²
```

the effective number of objects: 1 when one eigenvalue carries everything, N when all N are
equal. Normalised as (PR−1)/(N−1) it runs 0 to 1 like the published index, and for N = 2 it is
a monotone function of λ₂/λ₁ — so it is a **generalisation of the published reading, not a
different quantity**. `verify` checks that reduction rather than asserting it.

## Which tone carries the jitter, and why it is fixed

**Always the 3rd of four counting up (2241 Hz), on every trial.** Three reasons, and the second
is the one that decides it.

**It is interior.** Displacing the lowest or highest tone changes when the figure begins or
ends — a cue about the figure as a whole rather than about one component of it. An interior
tone has a neighbour on each side in both frequency and time.

**A random target would put an attention confound on the manipulation.** The number of
perceptual events per repetition grows with the step: one at step 0, four at step 100%. A
listener who must monitor all four channels is therefore monitoring more things at large steps
than at small ones, and that cost rises *in the same direction as the prediction*. With a fixed
target the listener monitors one channel at every step, so monitoring load is constant and
cannot masquerade as a binding effect. The residual worry — that a fixed target lets the
listener use within-channel timing — is not a confound here, because that strategy is
step-invariant by construction and so cannot generate the effect.

**Uncertainty costs precision the design cannot spare.** The whole range between the synchrony
floor (~2 ms) and the within-channel plateau (~9 ms) is a factor of about five. Four-way
stimulus uncertainty would inflate every threshold and compress that range.

The direction of the jitter *is* randomised per trial, so there is no "the figure got longer"
cue, and the geometry is symmetric for an interior tone.

## The isochrony trap, and the comparison that catches it

At exactly 100% the combined onset train is perfectly regular at 12 Hz. Deviation-from-
isochrony detection on an 83 ms interval runs a few milliseconds, which is *better* than the
within-channel plateau — so performance may well improve at 100% for a reason that has nothing
to do with binding. This is the same trap that produced an unexplained dip at ΔT = 100% in the
two-tone task.

The design catches it rather than avoiding it. The model puts 75% and 100% at 0.901 and 0.947
— the same figure — so it predicts no difference between them. Any behavioural difference there
is the rhythm cue. `analysis.rhythm_test` reports that pair on its own and never folds it into
the trend.

## Session length

| preset | conditions | tracks | trials | total |
|---|---|---|---|---|
| `tshear_feasibility` | 7 | 7 | 307 | **38 min** |
| `tshear_curve` | 7 | 14 | 614 | **69 min** (two sittings) |

At 3 Hz with five repetitions a trial carries 4.7 s of sound, and 2I-2AFC plays it twice. The
lever, if session time binds, is a single-interval design — at the cost of a criterion the
2AFC does not have, which matters because a criterion shift across steps would look exactly
like a threshold change.

A single track's own 95% interval spans a factor of **3.65**, so the feasibility preset answers
"is the task doable and roughly where do thresholds land", not "what is the curve".

## Known limits

* A rise with the step is consistent with temporal coherence, and also with any account in
  which a more spread-out pattern is simply harder to judge. The single-tone control bounds
  that alternative; it does not eliminate it.
* The figure's own level falls as it shears apart — four overlapping tones at step 0, one at a
  time at step 100%, a 6 dB range. Both intervals of a trial share a step, so it is not a cue
  about which interval moved, but the conditions are not equal-loudness and the analysis says
  so.
* A jitter larger than the step moves the target *past* a neighbour rather than away from it.
  At small steps the staircase visits that regime on its way down. It is inherent to displacing
  a tone out of a chord, not a fault; every trial records whether it happened and the analysis
  reports the fraction.
* Five repetitions give 1.3 s of build-up before the target. Streaming builds over seconds, so
  the number of repetitions is itself a candidate manipulation and not a settled choice.
