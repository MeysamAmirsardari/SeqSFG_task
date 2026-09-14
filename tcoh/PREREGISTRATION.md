# Two-tone coherence: onset asynchrony and the binding of two tones

**Pre-registration.** Written before any listener has been run. Everything below is fixed
unless a dated entry in *Deviations* says otherwise.

---

## 1. The question

Elhilali, Ma, Micheyl, Oxenham & Shamma (2009, *Neuron* 61:317–329) argue that two sound
elements are bound into one perceptual stream when their channels are temporally **coherent**,
and split into two when they are not — and that tonotopic separation alone predicts neither.
Their Figure 8B makes this quantitative for the simplest possible scene: two tones, whose onset
asynchrony ΔT is swept from exact synchrony to exact alternation. The model's segregation index
λ₂/λ₁ — the ratio of the second to the first singular value of the channel coherence matrix —
rises monotonically from 0.01 to 0.93 across that sweep.

That curve has never been measured behaviourally. Their own psychophysics (Figure 2) compares
only two states, synchronous and not, and does so through a tempo difference rather than a
fixed lag.

**We ask whether the behavioural counterpart of λ₂/λ₁ traces the same course, and — the part
that decides whether the answer means anything — whether it does so for the reason the model
gives rather than for three duller reasons that predict a similar rise.**

## 2. The measurement

We use Elhilali et al.'s asynchrony-detection task, generalised so that the precursor onset
asynchrony is a parametric variable.

Two isochronous pure-tone sequences, A (low) and B (high), six tones each, SOA 150 ms. B's grid
is the reference; A's is the same grid displaced by a lag expressed as **ΔT%**, a percentage of
half the period, so 0% is exact synchrony and 100% exact alternation. On each trial the
listener hears two such sequences separated by 500 ms. They are identical except that in one of
them — chosen at random — **the last B tone is displaced by ±δ**. The listener says which. An
adaptive staircase tracks δ.

The logic, which is Elhilali et al.'s: if the two tones are heard as one object, a displacement
of one of them changes that object's internal timing and is detectable at a few milliseconds.
If they are heard as two streams, the listener has only the B rhythm to go on, and thresholds
are an order of magnitude worse. **How far along that range a listener sits is a behavioural
measure of whether the tones are bound.**

### The normalised quantity

    κ(ΔT) = [log θ(ΔT) − log θ(0%)] / [log θ(B only) − log θ(0%)]

θ(0%) is the threshold with the tones exactly synchronous — one object — and θ(B only) the
threshold with the low tone switched off entirely. Both are measured in the same session, by
the same listener, with the same staircase. κ is therefore 0 for "bound" and 1 for "no better
than with no low tone at all", and is on the same scale as λ₂/λ₁ with no free parameter.

### Why the tone fills exactly half the period

Because otherwise ΔT is not an ordered axis. At any other duty cycle the model's own index is
**not monotone in ΔT** — it peaks near 75%, where the channels interdigitate most thoroughly,
and falls again at full alternation (`tcoh.model.duty_cycle_scan`; the configuration validator
refuses other duty cycles). A monotone behavioural result under a non-monotone prediction would
confirm nothing.

## 3. Conditions

| condition | A sequence | what it is for |
|---|---|---|
| `coh_0 … coh_100` | isochronous at lag ΔT | the sweep, ΔT ∈ {0, 25, 50, 75, 100}% |
| `scr_0 … scr_100` | A precursors at random times, final A tone unmoved | control 1, at every ΔT of the sweep |
| `b_only` | absent | the ceiling |
| `par_*` *(full config)* | only the final A tone | control 2 |
| `bld_*` *(full config)* | 13 precursors instead of 5 | build-up |
| `b_only_long` *(with build-up)* | absent, 13 precursors | the ceiling for the build-up comparison |

**Three invariants make the comparison a test rather than a demonstration**, and each is
checked numerically by `tcoh.verify`, sample by sample, not argued for:

1. Within a trial, the two intervals are **bit-identical except for the position of one tone**.
   Same frequencies, same count, same ramps, same starting phases, same A onsets, same duration.
2. **A never moves**, so the A channel alone carries *no* information about which interval is
   which — an observer listening only to the low tone is at chance by construction.
3. **B's grid is identical in every condition of the same length**, so whatever information the
   B channel carries on its own is the same at every ΔT. (The build-up conditions are the one
   exception, by design — see H3, which is scored against its own ceiling for exactly that
   reason.)

Together: the information available in either channel *alone* is the same in every condition.
Only the **relation** between them varies with ΔT. Any single-channel account of a ΔT effect is
excluded by construction.

## 4. Hypotheses, and what each can actually settle

### H1 — threshold rises with ΔT
*Prediction:* Spearman ρ > 0 between ΔT and log θ across coherent tracks; one-sided
permutation p < 0.05.
**H1 is not diagnostic and will not be reported as though it were.** Every account on the table
predicts a rise. Measured: H1 fires on 100% of simulated sessions when the hypothesis is true
*and on 100% when the leading rival is true.*

### H2 — coherent versus control, at matched ΔT — **the decisive test**
*Prediction:* the difference in log threshold between a control and the coherent condition is
**positive at ΔT = 0% and falls as ΔT grows**. Test: slope of that difference against ΔT,
one-sided permutation of the condition labels within each ΔT level, p < 0.05.

This is what the design exists for. The two rivals both predict a **flat line at zero**:

- **Interval discrimination (Weber).** In the yoked reference the A–B interval the listener
  judges grows with ΔT, and discriminating a change in a longer interval is harder for reasons
  that have nothing to do with streaming. But the controls hold that interval *exactly* — the
  final A tone is where the coherent condition puts it — so this account predicts no difference
  at any ΔT.
- **Local acoustic overlap.** The final pair's overlap is likewise identical in control and
  coherent conditions, so this account also predicts no difference at any ΔT.

The coherence account predicts a large difference at synchrony that vanishes at alternation.
Different shapes, so the data can choose. The model's predicted difference for the scrambled
control, computed for the stimuli actually used, runs **+0.32, +0.20, +0.00, −0.19, −0.27**
across ΔT = 0 … 100% (`python -m tcoh model`).

### H3 — five precursors versus thirteen *(full configuration only)*
*Prediction:* the ΔT effect is **larger** with the longer precursor. Streaming builds up over
seconds; a cue local to the final pair does not care what came before. This separates coherence
from local accounts by a route independent of H2.

**H3 carries a confound that H2 does not, and it is handled rather than noted.** Lengthening
the precursor lengthens the *B* sequence too, and a longer rhythm is easier to judge whether or
not anything streams — an advantage pointing in the same direction as the prediction. The
design therefore carries a **second ceiling**, `b_only_long`: the B-only condition at the long
precursor length. H3 is scored as an interaction —

    (θ_long − θ_short) for the coherent condition   minus   (θ_long − θ_short) for B only

— so the part of any gain that is about having more rhythm to listen to is subtracted rather
than claimed. `analysis.buildup_test` refuses to report the raw difference without saying it is
confounded when that second ceiling is missing.

### H4 — the shape of κ against the model curve
Reported, and **explicitly not treated as evidence.** Over the ΔT levels this design uses, the
model's predicted curve correlates with a straight line in ΔT at **r = 0.988** and differs from
it by at most 0.15 once both are rescaled to [0,1] (`tcoh.analysis.predictor_collinearity`). No
realistic amount of data separates them. The report prints the comparison with that caveat
attached and draws no conclusion from the winner.

### Replication anchor
ΔT = 0% should land near Elhilali et al.'s 2–4 ms and `b_only` near their 10–20 ms. A
disagreement with a published result has to be explained before the sweep between them means
anything.

## 5. Procedure

- **Task:** 2I-2AFC. Feedback **on** — the measure is a threshold and 2AFC has no criterion for
  feedback to distort, so it only keeps a listener calibrated through a long session. Recorded
  per trial regardless.
- **Staircase:** 3-down 1-up (converging on 79.4% correct), multiplicative steps ×4 → ×2 → ×√2,
  changing after the 1st and then 2 further reversals, stopping at the 6th reversal at ×√2.
  Threshold = geometric mean of the last six reversals. This is Elhilali et al.'s rule.
  *One discrepancy in their Methods is noted rather than silently resolved:* the same paragraph
  calls the rule "three-down one-up … 79.4%" and describes stepping down after **two**
  consecutive correct responses, which converges on 70.7%. We follow the stated target.
- **δ:** starts at 20 ms, floor 0.25 ms, ceiling 45 ms. Direction (early/late) randomised per
  trial and logged; the report tests for an asymmetry.
- **Order:** tracks interleaved; every condition contributes exactly one track per round, so
  each gets one track in each third of the session; no condition runs more than twice in a row.
- **Catch trials:** 6% of trials at 45 ms. They do **not** update the staircase. A session
  missing more than 15% of them is flagged and its thresholds are not reported as valid.
- **Practice:** to a 75% criterion at 45 ms, up to three rounds; failure is recorded, not hidden.
- **Level:** 65 dB SPL per tone, calibrated; both together at most 68 dB SPL. Levels roved ±3 dB
  per interval (verified uninformative: the target was the louder interval on 50.4% of 4000
  trials).
- **Tones:** A 1000 Hz, B 2378 Hz (15 semitones, 1.25 octaves — Elhilali et al.'s largest
  separation). 7.0 ERB apart; nearest low-order frequency ratio 5:2, 4.9% away, so harmonicity
  is not a fusion cue here.

## 6. Exclusion and stopping

Fixed in advance:

- A **session** is excluded if catch-trial misses exceed 15%, or if fewer than 80% of its
  tracks converge.
- A **track** contributes no threshold unless it reached six reversals at the final step size.
  A track whose averaged reversals include trials pinned at the δ ceiling or floor is reported
  separately and excluded from κ.
- κ is **not computed at all** if the bootstrap interval on log θ(B only) − log θ(0%) includes
  zero: with no dynamic range there is nothing to normalise.
- Sample size is fixed in advance by the configuration; no listener is added or dropped after
  looking at the tests.

## 7. Power (measured, not assumed)

By simulating the whole pipeline — staircase, design, analysis — against listeners whose
thresholds are generated by the hypothesis and by each rival (`python -m tcoh power`,
100 simulated sessions per truth, floor 3 ms, ceiling 15 ms, slope 0.6):

| generating truth | H1 fires | H2 fires |
|---|---|---|
| coherence (the hypothesis) | 100% | **87%** |
| pedestal (the Weber rival) | 100% | 4% |
| null (nothing depends on ΔT) | 6% | 3% |

Read the middle row: **the rival produces H1 on every single simulated session.** That is the
whole reason H2 exists.

A shortened design — a control at three ΔT levels instead of five, two tracks per condition —
retains 99% power for H1 and drops to **9%** for H2. It is a screen, not a test, and the
configuration validator says so on every run.

These numbers are for one listener's within-session comparison. They do not license
generalisation to a population; that needs several listeners, and the report says so whenever
only one has been run.

## 8. What this design cannot establish

- **It cannot confirm the model's shape.** See H4.
- **κ is normalised by two conditions measured in the same session**, so a bad floor or ceiling
  moves the whole curve. Both are reported in milliseconds alongside it.
- **Onset lag and acoustic overlap are perfectly confounded at a 50% duty cycle** — ΔT% is
  exactly 100 × (1 − overlap fraction). Breaking that confound needs a duty-cycle manipulation,
  which is a separate experiment; `allow_nonmonotone_duty` exists to build it.
- **The controls are imperfect in different directions.** The scrambled control matches tone
  count and energy but its own coherence is not flat across ΔT (the model puts it at 0.32 → 0.60).
  The pair-only control has no A sequence at all by construction but holds five fewer tones.
  Neither alone is decisive; the argument rests on their agreeing.
- **One listener generalises to one listener.**

## 9. Deviations

*(none yet — append dated entries here)*
