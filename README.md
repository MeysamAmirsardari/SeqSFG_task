# SeqSFG: does an auditory figure survive being sheared in time?

[![Open the playground in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MeysamAmirsardari/SeqSFG_task/blob/main/notebooks/SeqSFG_playground.ipynb)

**Start here if you are skeptical.** `notebooks/SeqSFG_playground.ipynb` is a runnable
playground written to be attacked: it lets you hear the figure, take the 2IFC task yourself,
listen to each interval reduced to *only* its long-term spectrum and *only* its amplitude
envelope (the two cues that would break the design, if either were the cue), re-run the
62-feature permutation test on a fresh random draw, and **plant a confound on purpose and
watch the verification battery catch it**. No installation: it clones this repository and runs.

A two-interval forced-choice experiment on the stochastic figure-ground stimulus in which
the figure's components are pulled apart in onset time. The independent variable is
`step`, the onset delay between successive components of one figure element. At
`step = 0` the element is a chord; as `step` grows it becomes a rising staircase.

The experiment measures **two psychometric functions over the same ladder, interleaved
trial by trial in the same block**, differing only in what the foil interval contains:

| ladder | foil interval | what falling performance means |
|---|---|---|
| `rising` | a figure of the same size on **new pitches every element** | the listener stopped detecting **recurrence**: binding plus pattern memory |
| `ungrouped` | **no figure at all**, the same channels at the same rate | the listener stopped detecting **presence**: binding alone |

If recurrence collapses at an asynchrony where presence survives, statistical learning
needs a bound object and not merely the components. If both fall together, learning
follows binding. The comparison between the two curves is the result; the `rising` ladder
is the one that is spectrally and temporally matched, and it carries the inference.

Everything in this repository is signal processing, scheduling and statistics: pure tones
rendered to a numpy array, a keyboard, a CSV per session. There is no clinical content.

```
pip install -e .            # numpy, scipy, matplotlib, sounddevice, soundfile
seqsfg config               # print the validated configuration, the ladder, the session estimate
seqsfg verify               # the verification battery and the ideal observers (about 1 minute)
seqsfg plots                # the diagnostic figures, into verification/figures
seqsfg demo --step 20       # write one trial to demo.wav (add --split for the two intervals)
seqsfg calibrate            # loop the reference tone for level calibration
seqsfg run --data data      # run a session (panel, calibration, practice, main, control)
seqsfg run --resume --code P01
seqsfg analyze data/P01/session_01 [data/P01/session_02 ...]
pytest tests                # 35 tests
```

The last battery report is in `verification/battery_report.txt` (`battery.json` alongside), and the
figures it refers to are in `verification/figures/`.

---

## 1. The design in one paragraph

Both intervals of a trial contain the same 744 tones, 31 in every channel of a 24-channel
pool, arriving over 2.25 s. In the target interval, six figure *elements* of seven
components each arrive at 3 to 5 Hz, always on the same seven channels. What the other
interval holds is the manipulation: on the `rising` ladder it holds six elements too, each
on seven freshly drawn channels, so the listener must hear which set *returns*; on the
`ungrouped` ladder it holds no elements at all, so the listener need only hear that
something groups. Figure tones are never added to the background, they are scheduled out
of each channel's fixed budget, so every channel carries the same number of tones in every
interval and the long-term spectrum is flat by construction. That is what makes
grouped-versus-grouped spectrally matchable, and it is why the only thing distinguishing
the `rising` target is the *conjunction* of channels and relative timing, which is what
binding means.

## 2. Decisions that were genuinely difficult, and what was traded

**Grouped on both sides, not the ungrouped fallback.** The task asks the listener to
compare two intervals that both contain bound elements. The obvious construction (add the
figure's tones on top of a random background) fails immediately: the recurring interval
piles five extra tones into each of seven channels and the long-term spectrum reads it off
without any binding. The fix is a fixed per-channel budget (31 tones per channel per
2.25 s interval). A recurring channel then has 6 figure tones and 25 background tones; a
non-recurring channel has 31 background tones; the long-term spectrum is flat in every
interval by construction, and the battery measures it as flat to 0.02 dB per channel. The
grouped-versus-grouped comparison is therefore used as the main experiment. The ungrouped
comparison is kept as a control cell, with its known envelope cue reported below.

**Yoking was tried and abandoned.** The first implementation built the redrawn interval
by swapping channel labels between figure tones and background tones, so that both
intervals shared the identical multiset of onset times. An exact swap is infeasible at
33% channel occupancy (the figure tone's onset time is blocked in the target channel in
about one case in five, and 35 swaps must all succeed). A minimal-edit version that reused
the background and only repaired collisions *worked* and looked yoked (95% of onsets
shared), but the battery caught it: the redrawn interval inherited the holes that the
recurring interval's background had left around its figure tones *and* acquired new holes
around its own, so its element windows were sparser by 1.3 tones at `step = 0` and its
element RMS lower by 0.5 dB. That is an envelope cue that grows exactly as the step
shrinks. Both intervals are now built by the same procedure from scratch, sharing only the
element schedule; roughly 18% of onset times coincide by chance. The trade is
trial-to-trial nuisance variance (each interval has its own background realisation) for a
guarantee that holds in distribution and is measured rather than argued.

**Two ladders, interleaved, not two blocks.** The `ungrouped` comparison was originally
two control cells. It is now a full psychometric function over the same ladder, because
"does statistical learning need a bound object?" is a question about how two curves differ,
and two points cannot answer it. Interleaving rather than blocking costs nothing and buys
a great deal: both curves are measured by the same ears in the same state, so fatigue,
criterion drift and level differences cannot masquerade as a difference between foils.
The price is trials per point. Two curves inside one 40-minute session means 14 trials per
cell rather than 26, and a 95% interval about ±0.25 wide on each point. That is thin for a
single session and the design is built for pooling; the alternative, blocking the ladders
into separate sessions, would have bought precision by giving up the within-session
control that makes the comparison interpretable.

**One question, two foils.** Interleaving only works if the listener is not switching
tasks. The instruction is "which sound kept coming back at the same pitches?", and that is
the correct question on both ladders: when the foil contains no figure, nothing in it comes
back. Practice therefore runs in two stages, `ungrouped` first (a chord against a plain
background, which demonstrates the target percept in its clearest form) and then `rising`
(which teaches that both intervals can contain a group and only one of them repeats). A
listener who only ever saw the easy foil would learn to listen for "a louder moment", which
is exactly the cue §4 says that foil affords.

**A 3 to 5 Hz element rate, and what it cost.** The elements repeat at 3 to 5 Hz
(inter-element interval drawn uniformly from 200 to 333 ms), with 30 ms tones. Two things
follow. First, the widest element must fit inside the shortest interval, so with seven
components the step is capped at 28 ms and the ladder cannot reach full non-overlap (that
would need 30 ms, or 210 ms intervals, or six components). Second, and less obviously, a
channel that recurs at 4 Hz is a rhythm, and the single-channel periodicity residual that
was inside the noise at 1.5 Hz became a real cue at this rate. The battery was the only
reason it was noticed. Measured on 30 fresh trials per condition, the channel-averaged
periodicity statistic ("pick the interval whose channels return more often at element-rate
lags") gave:

| configuration | recurrences per channel | tones sounding | periodicity d' | permutation p |
|---|---|---|---|---|
| 1.5 Hz, 50 ms tones (the earlier design) | 5 | 7.8 | +0.25 | 0.17 |
| 3–5 Hz, 8 elements, occupancy 0.33 | 8 | 7.9 | +0.57 | < 0.001 |
| 2.5–4 Hz, 8 elements, occupancy 0.33 | 8 | 7.8 | +0.61 | < 0.001 |
| 3–5 Hz, 8 elements, occupancy 0.41 | 8 | 9.9 | +0.43 | 0.06 |
| 3–5 Hz, 8 elements, occupancy 0.50 | 8 | 11.9 | +0.28 | 0.43 |
| 3–5 Hz, 6 elements, occupancy 0.33 | 6 | 8.0 | +0.41 | 0.06 |
| 3–5 Hz, 5 elements, occupancy 0.33 | 5 | 8.0 | +0.32 | 0.05 |
| **3–5 Hz, 6 elements, occupancy 0.41 (adopted)** | **6** | **9.9** | **+0.28** | **0.30** |
| 3–5 Hz, 8 elements, redrawn sets allowed to share more | 8 | 7.9 | +0.45 to +0.49 | 0.01 to 0.04 |

The residual is set by the number of recurrences per channel and diluted by background
density; the rate itself is irrelevant (2.5–4 Hz and 3–5 Hz give the same value). Letting
the redrawn interval share more channels between elements does not help, because the
statistic counts consecutive returns and those are exactly what "new pitches every time"
forbids. The adopted configuration trades two recurrences and a slightly denser cloud
(9.9 tones sounding instead of 7.9) for a residual that is back inside the permutation
null, and shortens the interval to 2.25 s, which buys 26 trials per condition inside the
40 minute session.

**A coarse pool, on purpose.** A pool fine enough to be dense (1/24 octave, as in the
published stimulus) puts several channels inside one critical band, and at the bottom of
the pool two of them beat slowly enough to be heard as a throb. The pool here has one
channel per ERB, 24 channels from 250 to 5459 Hz, so adjacent channels beat at 52 Hz or
faster (roughness, not throb); tones in one channel never overlap (a refractory rule), so
a channel never beats with itself. Density is bought with tones per channel, not with
channels. The validator refuses a pool whose lowest adjacent pair beats below 40 Hz.

**Equal amplitude, no loudness weighting.** Table [7] of the battery computes, per
channel, the excitation produced by the rest of the pool (roex filters, Glasberg & Moore
ERBs) against the absolute threshold (Terhardt). Masking exceeds absolute threshold by
33 to 53 dB in every channel, so masking and not audibility limits every channel, and
every tone stands about 12 dB above the pool's excitation in its own filter. An
equal-loudness style correction (A-weighting is shown as the concrete example) would spread
the levels *within one element* by up to 9.9 dB, which is the last thing components meant
to bind by common onset should have.

**Tone duration and the sweep, together.** With N = 7 components of 30 ms, adjacent
components stop overlapping at `step = 30 ms`. The default ladder `0, 5, 10, 15, 20, 28 ms`
is close to even in adjacent-component overlap (100, 83, 67, 50, 33, 7%); the maximum
number of components sounding at once runs 7, 6, 3, 2, 2, 2 and the element span runs 30
to 198 ms. The top of the ladder is set by the rate (see above), not by choice. The ladder
is a configuration entry and should be re-centred after piloting; the analysis refuses to
report a threshold that its own data do not bracket.

**Element rate versus widest element.** Elements must not run into each other, so the
minimum inter-element interval (200 ms) must exceed the widest element (198 ms at
`step = 28`). With six elements, a jittered interval of U[200, 333] ms, a lead of
U[150, 250] ms and a guaranteed 100 ms tail, the worst case is 2213 ms inside a 2250 ms
interval. The validator computes this and refuses anything that does not fit; nothing is
ever clipped or rejected after being drawn, and the battery reports the realised
inter-element interval distribution.

**The timing floor.** Onsets live on a 1 ms grid at 48 kHz, so the finest step is 1 ms
and background density is decoupled from the grid. The perceptual floor is the 5 ms
raised-cosine ramp: two onsets less than a ramp apart are not two onsets. The smallest
non-zero step in the default ladder is one ramp length.

**The learnt observers had to be calibrated too.** A first version pooled all trials of all
conditions into one ridge-logistic observer and reported leave-one-out d'. Under random
exchange of the two intervals that statistic is centred on zero, yet on the real data two
of the blind observers came out significantly *below* chance, which no cue can produce.
The reason was pooling: one linear rule fitted across conditions whose feature variances
differ finds a direction that anti-generalises to held-out trials. Within each condition
every one of those observers was at chance. The observers are now fitted within each
condition and only their held-out decisions are pooled; under relabelling that statistic
is centred on zero with a standard deviation of about 0.14, and the report gives a
Holm-corrected p over the five blind observers.

## 3. What is controlled, and how it was measured

`seqsfg verify` builds 40 fresh trials per condition (six main steps and the seven control
cells), renders both intervals, and measures the rows below on the audio, with exact
schedule counterparts where the quantity is a count. `seqsfg plots` draws the same
comparisons; §5 lists the figures. Table [3] of the report gives the
paired difference with its standard error for every row and condition. Numbers below are
from `verification/battery_report.txt`.

Between the two intervals of a trial, at every step:

| property | how measured | result |
|---|---|---|
| total number of tones | schedule | 744 / 744, exact |
| tones sounding at any instant (mean, sd, min, max) | schedule, 1 ms grid; and demodulated audio | mean 9.92 exact; sd, min, max differences within noise |
| long-term RMS | audio | −24.24 dB FS both; differences < 0.003 dB |
| long-term spectrum, band by band | complex demodulation at each channel frequency, 40 ms Hann | mean per-channel |A−B| 0.02 dB; peakedness (top-7 minus median) matched |
| occupancy of every channel | audio on-states | identical to three decimals |
| occupancy of the figure's channels | schedule and audio, per channel | identical by construction |
| figure components sounding simultaneously | schedule, per element | 7.00 / 7.00 at step 0; matched at every step to within 0.2 |
| figure components starting in the same instant | schedule | 7 / 7 at step 0; 1.0 to 1.4 otherwise, matched |
| tones inside element windows | schedule | 14.1 / 14.0 at step 0, 10.2 / 10.3 at 28 ms; matched at every step |
| broadband envelope: modulation depth, IEI-lag autocorrelation | 4 ms RMS frames | matched |
| element-locked envelope, averaged over elements | linear average of 4 ms frames | peak-to-trough 4.6 dB at step 0 in both; A−B smaller than exchanging the intervals produces |
| element-to-element loudness variation | RMS per element window | matched |

Across conditions: six elements per interval; inter-element interval mean 264 to 271 ms,
sd 38 to 40 ms, min 200, max 333; 31 tones per channel in every channel; 9.9 tones
sounding on average; channel level spread under 0.2 dB; RMS identical to 0.01 dB.

### The ideal observers

Each observer sees one property of the two intervals and nothing else. Two versions are
run: a fixed a-priori rule (pick the interval with the taller spectral peaks / deeper
envelope modulation / peakier occupancy / more same-channel onset pairs at element-rate
lags) and a learnt rule (ridge logistic regression on the feature difference,
leave-one-out). The primary claim is each observer pooled over the 240 trials of the six
main conditions.

| observer | learnt d', main conditions (within-condition LOO, decisions pooled) | Holm p |
|---|---|---|
| spectrum only | −0.06 [−0.28, +0.16] | 1.00 |
| envelope only | +0.06 [−0.16, +0.28] | 1.00 |
| occupancy only | 0.00 [−0.22, +0.22] | 1.00 |
| single-channel statistics | 0.00 [−0.22, +0.22] | 1.00 |
| all of the above at once | −0.16 [−0.39, +0.06] | 0.88 |
| oracle that is told the element windows | +0.82 [+0.58, +1.05] | < 0.001 |

Under random exchange of the two intervals this statistic is centred on zero with a
standard deviation of about 0.14, so only a positive d' beyond about +0.25 counts against
the design. The oracle row is the point of the jitter: an observer who knows *when* the
elements are can count which channels return in every window, and does so easily at small
steps (d' = 3.17 at step 0, 2.17 at 5 ms, and about 0.3 from 10 ms on). The listener is
not told, and cannot infer it from any of the properties above.

Two further checks guard against reading a grid of numbers optimistically.

**Multiplicity over the observer grid.** Across the 30 blind observer × main-condition
cells, the largest is the combined observer at 10 ms with d' = −0.74 (uncorrected
p = 0.017, Holm p = 0.50); it is negative, and nothing survives correction.

**A global permutation test over every feature at once.** Exchanging the two intervals of
a trial is exactly the null hypothesis "the intervals differ in nothing an observer can
measure". Flipping that label for a random subset of trials therefore gives the null
distribution of the whole audit, corrected for having looked at all 62 scalar features.

| | |
|---|---|
| largest \|d'\| observed, over 62 features | 0.335 (channel-averaged periodicity, the residual named in §4) |
| median largest \|d'\| under relabelling | 0.275 |
| 95th percentile under relabelling | 0.374 |
| p, any feature separates the intervals | 0.14 |

A test of this form has real power: planting a 5 dB level difference in the battery's own
measurements drives it below p = 0.05, which is checked as a unit test, and at the
rejected 8-element configuration it returned p < 0.001 on the same statistic.

## 4. What is not controlled, and cannot be

This is the section to read first.

**Single-channel periodicity at the element rate.** A channel that recurs in six
elements carries six quasi-periodic onsets at about 4 Hz; in the redrawn interval no
channel does. This is not a flaw of the implementation, it *is* recurrence, seen one
channel at a time, and no construction that keeps "new pitches every time" can remove it.
It is the one property of the two intervals that the battery cannot drive to zero, and
§2 shows how it was traded down.

Its size at the adopted configuration. The channel-averaged count of same-channel onset
pairs at element-rate lags is 0.5% higher in the recurring interval, on an effect that is
itself about 4% (`single_channel.png`). As a fixed rule that gives d' = +0.34, uncorrected
p = 0.005, which is the largest of the 62 features audited and inside the band the largest
of 62 reaches by chance (permutation p = 0.14). The learnt single-channel observer is at
chance (d' = 0.00). The statistic does not vary with `step`, so it can only add a constant
floor to the psychometric function; it cannot shape it, and a threshold estimated from the
fall-off is unaffected by a constant. It is measured empirically in a listener by the
**onechannel** control cell (one channel recurring at the element times against a plain
background), where it is the only cue available. At 3 to 5 Hz that cell matters more than
it did at 1.5 Hz: a single channel pulsing at 4 Hz is the kind of regularity listeners are
known to extract from noise, and the cell's result bounds how much of the main task can be
done that way.

**Oracle-only differences.** Rows that need to know S differ and must: the union
occupancy of S (a chord's seven tones overlap in time, so at step 0 the union is 0.04
lower in the recurring interval); the maximum number of S channels sounding at once
(7 versus about 6 at step 0); and, by 0.1 to 0.2 tones, the chance coincidences among an
element's own channels, because a recurring channel holds 25 background tones where a
redrawn one holds about 30. None of these are visible to an observer that does not know S.

**Momentary silences.** With 9.9 tones sounding on average, the instantaneous count
reaches zero briefly in about one interval in eight (row "tones sounding: min"). It does
so equally in both intervals.

**The `ungrouped` ladder carries a non-binding route at its smallest steps, and this
cannot be designed away.** A figure that is present in one interval and absent in the other
*is* a level event: synchronous onsets are an envelope transient. Any comparison of
"grouped" against "not grouped at all" therefore differs in broadband envelope, and no
construction can match it, because matching the envelope means giving the foil synchronous
onsets, which is giving it a group. The trichotomy is real: a foil is either grouped
(and then it is the `rising` foil), or ungrouped (and then its envelope differs).

What the battery measures, per step, for an observer with access to the broadband envelope
and nothing else:

| step | 0 ms | 5 ms | 10 ms | 15 ms | 20 ms | 28 ms |
|---|---|---|---|---|---|---|
| `rising` ladder, envelope observer d' | 0.00 | +0.74 | −0.45 | +0.18 | −0.36 | +0.27 |
| `ungrouped` ladder, envelope observer d' | **+2.04** | **+0.64** | +0.18 | +0.09 | −0.27 | −0.09 |

So the cue is confined to the two smallest steps and is gone from 10 ms onward, because by
then the seven components are spread over 90 ms or more and no longer make a transient.
Three consequences, all of which the software enforces rather than merely stating:

1. The `ungrouped` curve's own high end is inflated: at step 0 a listener could be right
   most of the time without binding anything. Its fall from 0 to 10 ms therefore partly
   tracks a vanishing envelope cue rather than a failure to bind.
2. The informative part of that curve is 10 to 28 ms, where it is clean.
3. `seqsfg analyze` reads the battery's measured cue profile and marks exactly those cells
   in its own output, naming the observer and its d', so the flag travels with the data
   instead of living in this file.

The `rising` ladder has no such route at any step, which is why it carries the inference
and why the validator refuses a configuration that drops it.

**The feedback question.** Trial-by-trial feedback does not bias 2IFC, but it teaches
whatever cue works. It is on by default because the observers above are at chance; if the
configuration is changed, run the battery again before running a listener.

**Shearing in time is two manipulations at once.** Delaying the components also makes the
element longer (30 to 198 ms across the ladder) and reduces how many components sound at
once (7 down to 2). "The components stopped binding" and "the element became a longer,
slower object with fewer simultaneous parts" are the same manipulation in this design,
and this experiment cannot tell them apart. The control that separates them is a 2 × 2
crossing onset arrangement with component duration:

1. synchronous onsets, 50 ms components (the reference chord);
2. sheared onsets, 50 ms components (this experiment);
3. synchronous onsets, components lengthened to the sheared element's span: the same
   long, slow object with fully coherent onsets. If (3) stays high while (2) falls, length
   is not the cause;
4. sheared onsets, each component held until the end of the element (offsets
   synchronous): the same staggered onsets, but the components end up sounding together.
   If (4) recovers relative to (2), co-activation and not onset coherence is what binds.

Cells (3) and (4) change tone durations, so the per-channel budget must then be expressed
in occupied time rather than tone count, and the battery must be rerun on them. They are
proposed, not implemented.

**The ladder stops short of non-overlap.** At 5 Hz the widest element that fits is 198 ms,
so the largest step is 28 ms and adjacent components still overlap by 7%. If the
psychometric function has not reached floor by 28 ms the experiment cannot say where it
does; the fix is six components, or a rate ceiling of 4.8 Hz, both configuration entries.

**The single-channel and duration issues are the residue; everything else that could be
named was matched and measured.** Rows in the report marked `*` outside the oracle rows
are at the 2 to 3 standard-error level among some 300 comparisons and change sign between
seeds.

## 4b. What a reviewer will push on, in order

1. **Is a budget-matched figure audible at all?** In the published stimulus the figure's
   channels gain long-term energy; here they do not, by design, so the figure is carried by
   coherence alone. Nothing in the battery can answer this; the practice criterion answers
   it per listener. If listeners cannot pass practice at step 0, the design has no dynamic
   range and the budget rule is too strict for this density. Pilot before anything else.
2. **Recurrence versus binding.** Both curves are now measured over the same ladder, so
   the comparison is a curve comparison rather than an anchor point, and the primary test
   is whether the two need different psychometric functions at all. What a reviewer will
   press on instead is the asymmetry in what the two curves can be trusted to mean: the
   `rising` curve is clean at every step, the `ungrouped` curve is clean only from 10 ms
   (§4). Expect to be asked to base any claim about the interplay on the 10 to 28 ms range,
   or to add the envelope-observer curve to the figure as a reference.
3. **Length and simultaneity are confounded with asynchrony** (above). The 2 × 2 is
   proposed, not implemented, and a reviewer will ask for at least cell (3).
4. **The element rate is in the range of rhythmic entrainment.** At 3 to 5 Hz with 25%
   jitter, temporal expectation can direct attention to the elements. Because both
   intervals share the schedule this is not a cue for the judgment, but it is a mechanism
   the discussion has to own: the recurring channels are sampled at predictable times.
5. **The ladder is capped at 28 ms by the rate**, and the single-channel residual is
   bounded rather than eliminated. Both are stated with numbers; a reviewer can disagree
   with the trade but not discover it.
6. **Group-level inference is not implemented.** The analysis is per listener (with
   pooling of a listener's sessions). Thresholds across listeners, or a mixed-effects
   logistic model of correctness on step and ladder, are needed for a paper and are a
   modest addition.
7. **Fourteen trials per cell in one session.** Enough to see a large curve difference, not
   enough to bound a small one. Plan on two sessions per listener; the analysis pools them
   automatically and the seeding guarantees the second session is a fresh stimulus set.

## 5. The figures

`seqsfg plots` writes these to `verification/figures/`. The first four rebuild trials from
their seeds; the rest reuse one run of the battery.

**What the stimulus is**

* `raster_pair.png` — a coherent chord beside a 10 ms staircase, figure components in red,
  frequency in semitones re 1 kHz. The conventional view of this stimulus.
* `raster_ladder.png` — one element of both intervals at every step of the ladder, zoomed
  so the shear is visible, with the element boxed and interval A's recurring channels
  marked. Chord at the top, nearly sequential at the bottom.
* `raster_overview.png` — a whole trial, both intervals. Interval A's five elements all sit
  on the same dotted lines; interval B's five elements each sit somewhere new. This is the
  design in one picture: same number of tones, same element times, same density.
* `raster_controls.png` — what each control variant does to the stimulus.

**Why the two intervals cannot be told apart**

* `matching_rows.png` — every measured property against every condition, as paired
  differences in standard errors. The main-experiment block is white. Rows below the line
  are the ones that need to be told which channels recur; they and the ungrouped control
  are the only places colour appears.
* `matching_spectrum.png` — the long-term spectrum of both intervals with the strategy the
  design has to defeat: the distribution of the "taller peaks" statistic, which overlaps
  completely (d' = −0.07).
* `matching_envelope.png` — the broadband envelope locked to element onsets. Both intervals
  carry the same transient because both contain elements; each panel reports the observed
  maximum difference beside the value that exchanging the intervals produces 95% of the
  time. The ungrouped panels are where that breaks, by design.
* `matching_occupancy.png` — occupancy per channel, identical by construction, and the
  blind occupancy statistic, at chance.

**Why no observer succeeds**

* `observers.png` — every observer at every condition, with control cells marked open so
  the outliers are identifiable as the documented controls, and the pooled main-condition
  result beside it.
* `observers_audit.png` — all 62 scalar features with the band a single pre-chosen feature
  would clear by chance and the band the largest of 62 would clear, plus the permutation
  null with the observed maximum marked.
* `single_channel.png` — the residual cue named below, measured three ways.
* `ladder_cues.png` — what each ladder's foil affords a listener who never binds anything,
  step by step, with the four blind observers side by side. This is the figure that shows
  where the `ungrouped` curve can and cannot be taken at face value.

**Whether the parameters are sane**

* `design_checks.png` — the jitter as actually drawn against what was requested, the
  instantaneous density in both intervals, the ladder, the adjacent-channel beat rate
  against the throb limit, masking against absolute threshold, and what an equal-loudness
  weighting would do to one element.

## 6. The experiment

* **Two-interval forced choice**, target interval balanced within every cell. The two
  ladders are **interleaved in one block**, shuffled under two constraints at once: at most
  2 consecutive trials share a (ladder, step) cell and at most 4 share a ladder, so the
  listener cannot infer from recent history which foil is coming. The condition is never
  displayed.
* **One question for both ladders.** The instruction is "which sound kept coming back at
  the same pitches?", and it is the correct question on both: when the foil has no figure,
  nothing in it comes back. The listener never switches task.
* **Practice runs in two stages**, each with feedback and a criterion of 10 of 12, up to
  two attempts each. Stage 1 is `ungrouped` at step 0 (a chord against a plain background,
  the clearest demonstration of the target percept); stage 2 is `rising` at step 0, which
  teaches that both intervals can contain a group and only one repeats. A session that
  fails either stage stops and is recorded as `practice_criterion_not_met`, naming the
  stage that failed.
* **Main block**: 2 ladders × 6 steps × 14 trials = 168. **Control block**: 5 cells × 8 = 40
  (`scrambled` and `redrawn` at 15 and 28 ms, `onechannel` at 0 ms). Self-paced breaks
  every 40 trials and between blocks. Estimated session 37.5 minutes; the validator refuses
  a configuration that exceeds 40.
* **Fourteen trials per cell is the cost of two curves.** One session gives a 95% interval
  about ±0.25 wide on each point. The design is built for pooling: sessions of one
  participant are pooled by `seqsfg analyze` when their configuration hashes agree, and two
  sessions bring each point to 28 trials.
* **Calibration**: a 1 kHz tone at the amplitude of one stimulus tone plays until the
  experimenter enters the measured level; the intended level is 60 dB SPL per tone (the
  cloud is about 10 dB above that). The entered value is stored with the session.
* **Participant panel**: code, age, sex, handedness, self-reported hearing, musical
  training, headphone model, experimenter, consent confirmation. One row per person in
  `data/participants.csv`; a session cannot start without consent confirmed.
* **Seeding**: the session seed is a hash of (participant code, session index); every
  trial has its own recorded seed; `seqsfg.stimulus.make_trial(cfg, seed, step, variant)`
  rebuilds any stimulus exactly. The same person run twice gets different orders and
  different stimuli.
* **Logging**: every trial is appended to `trials.csv` and fsynced as it is answered.
  `--resume` continues from the next trial and refuses if the configuration, the trial
  list, or the package source has changed since the session started.
* **Provenance** in `session.json`: package version, a hash of the package source, git
  commit and dirty flag when the tree is a git repository, host, platform, Python and
  library versions, start time, the full configuration and its hash, the whole design.

Control variants: `scrambled` = the same asynchronies in a fixed random order (same order
for both intervals and every element); `redrawn` = the same channels recur but the delay
order is redrawn every element (in both intervals); `onechannel` = one channel recurring
against a plain background, which isolates the single-channel periodicity residual of §4.

## 7. The analysis

`seqsfg analyze <session dir> [...]` (several sessions of one participant are pooled if
their configuration hashes agree) writes `analysis/results.json` and three figures.

Per ladder:

* Proportion correct with Wilson intervals (finite at 0 and 1); d' = √2·z(pc) with pc
  clipped to [1/2n, 1 − 1/2n].
* A decreasing logistic psychometric function with the lower asymptote fixed at 0.5 and a
  lapse parameter, fitted by maximum likelihood; threshold at pc = 0.75; a bootstrap CI.
  The threshold is **reported only if** the easiest condition is above chance, performance
  peaks at the easiest condition, the threshold lies inside the tested range, the CI is
  narrower than the tested range, the bootstrap defines a threshold in ≥ 75% of resamples,
  the fit converged, **and the fitted transition width is at least a quarter of the step
  spacing** (a curve that switches inside 0.5 ms cannot be measured by a ladder sampled
  every 5 ms). Otherwise the number is withheld and named as noise.
* Exact binomial test against chance per condition and Fisher's exact test of each
  condition against that ladder's easiest, both Holm-corrected within the ladder.
* A single-trial logistic of correctness on step.

Between the ladders:

* **Primary: do the two ladders need different psychometric functions at all?** A
  likelihood-ratio test of one shared curve against one curve per ladder, 3 df. Simulated
  at this design's trial counts over 200 replicates, it rejects at 0.05 (14 trials per
  cell) and 0.035 (40 per cell) when the ladders are identical, and at 0.87 and 1.00 when
  they genuinely differ in shape.
* **Threshold difference** with a bootstrap CI, which refuses to report when either
  curve's own gates failed. A difference between two numbers, one of which the fit declined
  to report, is not a result.
* **Secondary, low-powered: a step × ladder interaction** on the linear logit scale, from
  every trial. It rejects at only 0.03 when the curves differ in shape but share a mean
  slope, and 0.42 when they differ in slope outright, so a null here is not evidence that
  the curves agree. The summary says so in place when the primary test disagrees with it.

Diagnostics and honesty checks: interval preference, accuracy by target position, first
versus second half, after a correct versus after an error, and whether each ladder's
easiest condition stayed easy. Every main-block cell is annotated with the **measured
non-binding route**, if any: `seqsfg analyze` reads `verification/battery.json` for the
same configuration and marks each cell where a blind ideal observer beats chance after
correction over that ladder's whole observer grid, printing which observer and how large.

Figures: `psychometric.png` (both curves with their fits, thresholds only where
trustworthy, and the comparison in the footer), `timecourse.png` (one panel per ladder),
`controls.png` (control cells beside the primary ladder at the same step).

A simulated listener (`seqsfg run --auto 12 --fast`) exercises the whole pipeline
end to end, including both curves and the comparison; this is a pipeline test, not data.

## 8. Layout

```
seqsfg/config.py     every parameter, Derived quantities, the validator, describe()
seqsfg/pool.py       ERB pool, absolute threshold, roex excitation, A-weighting (diagnostics only)
seqsfg/stimulus.py   schedule construction (recurring / redrawn / ungrouped), rendering, invariants
seqsfg/measure.py    audio measurements: envelopes, demodulation, occupancy, single-channel statistics
seqsfg/verify.py     the battery, the observers, the feature audit, the report
seqsfg/design.py     trial lists, balancing, run-length constraint, seeding, design hash
seqsfg/session.py    participants table, provenance, trial log, resume check
seqsfg/runner.py     the experiment
seqsfg/analysis.py   statistics; seqsfg/figures.py  result plots
seqsfg/plots.py      diagnostic figures: rasters, matching, observers, design checks
seqsfg/cli.py        seqsfg config | verify | plots | demo | calibrate | run | analyze | participants
notebooks/           SeqSFG_playground.ipynb: the Colab playground (audio, self-test, live tests)
tests/               50 tests: validator refusals, stimulus invariants, two-ladder balance and ordering,
                     practice stages, resume, analysis gates, and positive controls that the permutation
                     test catches a planted difference and the curve comparison catches a planted
                     difference in shape while staying calibrated on identical ladders
verification/        the battery report and JSON, and figures/ for the default configuration
demo/                example trials as WAV
```

Parameters are overridden with `--set key=value` (JSON values) or `--config file.json`,
and every command validates the result first.
