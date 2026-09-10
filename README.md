# SeqSFG: does an auditory figure survive being sheared in time?

[![Open the playground in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MeysamAmirsardari/SeqSFG_task/blob/main/notebooks/SeqSFG_playground.ipynb)

**Runnable notebooks.** [`SeqSFG_playground.ipynb`](https://colab.research.google.com/github/MeysamAmirsardari/SeqSFG_task/blob/main/notebooks/SeqSFG_playground.ipynb)
is the skeptic's notebook for the two-interval task (below).
[`SeqSFG_yesno_playground.ipynb`](https://colab.research.google.com/github/MeysamAmirsardari/SeqSFG_task/blob/main/notebooks/SeqSFG_yesno_playground.ipynb)
is the same treatment for the single-interval yes/no task of section 9: it plays the naive
figure-versus-cloud version, shows the envelope cue that makes it 90% solvable without hearing a
figure, runs the audit live on all three absent classes, lets you take the task yourself, and
plants a 0.1 dB cue to check the audit catches it.
[`SeqSFG_strategy.ipynb`](https://colab.research.google.com/github/MeysamAmirsardari/SeqSFG_task/blob/main/notebooks/SeqSFG_strategy.ipynb)
is the strategy notebook: the crossover hypothesis made quantitative, the ordered-versus-reshuffled
contrast played aloud, a demonstration that the current design forbids the cross-trial learning the
hypothesis needs, a **working rate-STDP model prototype**, a power analysis for the order x asynchrony
interaction, and falsification conditions.

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

Both intervals of a trial contain the same 780 tones, 26 in every channel of a 30-channel
pool spaced 1 ERB apart from 200 to 9486 Hz, arriving over 4.0 s. Tones are 45 ms with 5 ms
ramps. Nine figure *elements* of seven components each arrive at 3.0 to 3.3 Hz, so the target
is a 2.5 s continuous stream at a speech-like rate rather than a few isolated events.

**Each element is confined to a band of 13 pool channels**, about two octaves wide, so it has
a register. That is not decoration. Components drawn from the whole pool span 4.7 of the
pool's 5.6 octaves, so two elements on completely disjoint channels still cover the same range
and sound alike: interleaved combs, not different pitches. Banding cuts an element's own
spread to 2.2 octaves and lets consecutive foil elements land 1.5 octaves apart, with the foil
sitting 2.1 octaves from the target's register -- against 0.69 and 0.57 octaves unbanded.

The two intervals are built by a **matched-incidence** construction. Every element puts one
tone on each of the target's seven channels AND one on each of that element's seven foil
channels, in *both* intervals. What differs is which of the two is time-aligned: the aligned
set starts together at the element onset and binds into a group, the other is scattered across
the same span and never binds. In the target interval the aligned set is the same seven
channels every time ("sam, sam, sam"); in the other it is a fresh band each time, none of them
containing a single channel of the target set ("bob, kim, she", with no sam anywhere).

Three properties follow by construction rather than by tuning. Per-channel tone counts are
identical, so the long-term spectrum is identical. Every channel that recurs in one interval
recurs just as often in the other, so single-channel periodicity at the element rate is
matched channel by channel. And the scattered counterpart spans exactly what the aligned
group spans at every rung of the ladder, so the step changes only whether the relative timing
is *consistent* from element to element -- never how wide the element is. The aligned group
also takes one shared jitter per element from the same distribution as the scattered offsets,
so being aligned does not by itself make a channel's inter-onset intervals less variable.
What is left as the only difference is the *conjunction* of channels and relative timing,
which is what binding means.

Half the trials use one fixed figure, the same in every session, so it can be learned; the
other half draw a fresh figure each trial. The contrast between them is a within-session
measure of whether a learned regularity helps.

## 2. Decisions that were genuinely difficult, and what was traded

**Grouped on both sides, not the ungrouped fallback.** The task asks the listener to
compare two intervals that both contain bound elements. The obvious construction (add the
figure's tones on top of a random background) fails immediately: the recurring interval
piles extra tones into each of seven channels and the long-term spectrum reads it off
without any binding. The fix is a fixed per-channel budget (26 tones per channel per
4.0 s interval), out of which figure tones are scheduled rather than added. A recurring
channel then has 9 element tones and 17 background tones; every channel the trial uses carries
26 either way; the long-term spectrum is identical in the two intervals by construction, and
the battery measures per-channel counts as exactly equal. The grouped-versus-grouped
comparison is therefore used as the main experiment. The ungrouped comparison is kept as a
second ladder, with its known envelope cue reported below.

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

**A 3 Hz element rate, and what it costs.** Elements repeat at 3.0 to 3.3 Hz
(inter-element interval drawn uniformly from 300 to 333 ms), which is the point of the
paradigm: a stream at a speech-like rate, not a few isolated events. The cost is the top of
the ladder. Elements must not run into each other, and a matched-incidence element holds the
aligned group *and* a scattered counterpart of the same extent, so its footprint is twice the
span. With seven components of 45 ms that caps the step at 17 ms, an adjacent-component overlap of
0.62. Tone duration and ladder reach trade directly against each other at a fixed rate,
because the footprint is twice `(N-1)*step + D`: 30 ms tones reach overlap 0.33, 45 ms reach
0.62, 60 ms reach 0.75. Forty-five is what this pilot runs, so the ladder spans 1.00 down to
0.62 and no further, and the whole informative range is 0 to 17 ms. If a listener's
psychometric function has not fallen by then the levers are shorter tones or the background,
not a wider step.

What the rate used to cost, and no longer does, was single-channel periodicity: a channel
recurring at 3 Hz is a rhythm, and in the target interval seven channels had it while in the
foil none did. Two constructions were built and measured against that. `foil_subpool_size`
restricted the foil to a small subpool so its channels recurred nearly as often; it worked
statistically and destroyed the task, because with only twelve usable channels two
"different" foil elements then shared four of seven pitches and sounded like repetition.
Matched incidence removes the asymmetry instead of trading against it: the target's channels
recur in *both* intervals, bound in one and scattered in the other. The subpool code is still
there and still tested, but nothing ships using it.

**A coarse pool, on purpose.** A pool fine enough to be dense (1/24 octave, as in the
published stimulus) puts several channels inside one critical band, and at the bottom of the
pool two of them beat slowly enough to be heard as a throb. The pool here has one channel per
ERB, 30 channels from 200 to 9486 Hz, so adjacent channels beat at 46 Hz or faster
(roughness, not throb); tones in one channel never overlap (a refractory rule), so a channel
never beats with itself. Density is bought with tones per channel, not with channels. The
validator refuses a pool whose lowest adjacent pair beats below 40 Hz.

Thirty channels is not a free parameter. Seven elements of seven components need a foil
universe large enough that consecutive elements can be disjoint and the target's pitches
excluded, which is 7 + 23; one channel per ERB over the usable range is 30. Pairwise-disjoint
foil elements would need 49 channels and hearing has about 30 ERBs to spend, which is why
some pitch reuse at longer lags is arithmetic rather than a choice (see section 4).

**Elements are banded, because disjoint channels are not different pitches.** The first
version drew each element's seven components from the whole pool. Their channel sets were
exactly disjoint -- zero shared channels between consecutive foil elements, zero with the
target set -- and they still sounded the same, which is what the first listener reported. The
measurement says why: an element drawn from the whole pool spans 4.73 of the pool's 5.57
octaves, while consecutive elements' centroids differ by 0.69 octaves, 15% of an element's own
spread. Two interleaved combs covering the same five octaves have the same register and the
same timbre; membership is not something the ear compares across a 300 ms gap.

Confining an element to 13 contiguous channels fixes it: spread falls to 2.22 octaves,
consecutive foil elements land 1.49 octaves apart, and the foil sits 2.05 octaves from the
target's register. The jump is now 0.67 of an element's own width rather than 0.15.

What it costs, precisely. Narrower is better for register -- a 9-channel band gives a 1.54
octave element and 2.30 octaves of separation -- but narrower bands crowd the components into
fewer critical bands and the residual of section 4 becomes visible: at band 9 the global
permutation test rejected in one seed of four and sat below 0.10 in two more. Thirteen is the
widest band that still buys most of the register separation and the narrowest that stays clean
across seven seeds. The pool affords only about three non-overlapping registers for a
seven-component element, so the foil reuses registers at longer lags; the sampler spends that
reuse as far back as it can and never repeats a channel set (`max_shared_any`). Setting
`figure_band_channels` to null restores the unbanded version, which is cleaner still and which
no listener could do.

**Equal amplitude, no loudness weighting.** Table [7] of the battery computes, per channel,
the excitation produced by the rest of the pool (roex filters, Glasberg & Moore ERBs) against
the absolute threshold (Terhardt). Masking exceeds absolute threshold by 29 to 51 dB in every
channel, so masking and not audibility limits every channel, and every tone stands 13.6 to
17.5 dB above the pool's excitation in its own filter. That headroom is why short tones work
here: the earlier configuration that made them inaudible was denser, not shorter. An
equal-loudness style correction (A-weighting is shown as the concrete example) would spread
the levels *within one element* by up to 12.1 dB, which is the last thing components meant to
bind by common onset should have.

**Tone duration and the sweep, together.** With seven components of 45 ms, adjacent
components stop overlapping at `step = 45 ms`. The ladder `0, 4, 7, 11, 14, 17 ms` runs
adjacent-component overlap 1.00, 0.91, 0.84, 0.76, 0.69, 0.62; the maximum number of
components sounding at once runs 7, 7, 7, 5, 4, 3 and the element span runs 45 to 147 ms. The
top of the ladder is set by the rate (see above), not by choice. The ladder is a configuration
entry and should be re-centred after piloting; the analysis refuses to report a threshold that
its own data do not bracket.

**Element rate versus widest element.** Elements must not run into each other, so the minimum
inter-element interval (300 ms) must exceed the widest element footprint (294 ms at
`step = 17`, which is twice the 147 ms span because the element carries its scattered
counterpart too). With nine elements, a jittered interval of U[300, 333] ms, a lead of
U[350, 600] ms and a guaranteed 350 ms tail, the worst case is 3908 ms inside a 4000 ms
interval. The validator computes this and refuses anything that does not fit; nothing is ever
clipped or rejected after being drawn, and the battery reports the realised inter-element
interval distribution.

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
| total number of tones | schedule | 780 / 780, exact, every condition |
| tones sounding at any instant (mean, min, max) | schedule, 1 ms grid; and demodulated audio | mean and min exact; max within 0.35 +/- 0.24 |
| long-term RMS | audio | differences under 0.005 dB |
| long-term spectrum, band by band | complex demodulation at each channel frequency, 40 ms Hann | mean per-channel abs(A-B) 0.034 to 0.036 dB, worst channel 0.21 dB; peakedness matched |
| per-channel tone counts | schedule | identical channel by channel, exact, by construction |
| occupancy of every channel | audio on-states | identical to four decimals |
| occupancy of the figure's channels | schedule and audio, per channel | identical by construction |
| figure components sounding simultaneously | schedule, per element | 7.00 / 7.00 at step 0; matched to 0.29 at every step |
| figure components starting in the same instant | schedule | 7 / 7 at step 0, exact; matched to 0.03 elsewhere |
| tones inside element windows | schedule | matched at every step, largest difference 0.15 +/- 0.06 |
| broadband envelope: modulation depth, IEI-lag autocorrelation | 2 ms RMS frames | matched |
| envelope bursts: count above 3 SD and 5 SD, mean and max height | 2 ms RMS frames, no schedule | matched; see section 4 |
| element-locked envelope, averaged over elements | linear average of RMS frames | peak-to-trough matched at every step, largest difference 0.19 +/- 0.12 dB |
| element-to-element loudness variation | RMS per element window | matched |

Every row above is for the `rising` ladder, which carries the inference. The `ungrouped`
ladder does not match on envelope and cannot; that is section 4.

Across conditions: nine elements per interval; inter-element interval mean 316 to 317 ms,
sd 10 ms, min 300, max 333; element span 45 to 147 ms across the ladder; 26 tones in every
channel the trial uses; 8.8 tones sounding on average; RMS identical to 0.005 dB.

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

**Synchrony shows up in across-channel dispersion, and can only be buried.** The target's
seven channels are aligned, so they share their onset times exactly and their per-channel
timing statistics are perfectly correlated. The foil's seven counterpart channels are
scattered independently, so theirs are not. Any statistic that takes a **max or an SD across
channels** is therefore sensitive to it, even though every channel-averaged statistic matches.

This is irreducible in principle. To match the dispersion, the scattered channels would have
to be correlated too -- each one's onset would have to be a common per-element jitter plus a
constant per channel -- and a set of channels with a fixed relative timing pattern repeated
every element is precisely a bound figure. Removing the statistic means removing the
manipulation.

What can be done is to put it under the noise. Each channel's timing statistics are estimated
from that channel's own tones, so the background tones sharing the channel act as independent
noise on the estimate, and the leak's visibility tracks how many of them there are. It has
surfaced three times and been buried three times by the same knob:

| tone duration | elements banded | tones per channel | result on the `rising` ladder |
|---|---|---|---|
| 30 ms | no | 11 | learnt single-channel observer d' = +0.50 (Holm p = 0.000), 1 cell surviving, permutation p = 0.002 |
| 30 ms | no | 16 | clean; permutation p = 0.197 |
| 45 ms | no | 16 | permutation p = 0.001, largest feature `ch_audio:pairs_iei_norm:sd` at 0.497 |
| 45 ms | no | 20 | clean; permutation p = 0.298 |
| 45 ms | band 13 | 20 | one draw in seven at permutation p = 0.001 -- a rejection, not noise |
| **45 ms** | **band 13** | **26 (shipped)** | **clean; see the seed sweep below** |

Longer tones and banded elements both raise the number of background tones needed. Longer
tones raise per-channel occupancy, which leaves background placement less independent of the
element tones. Banding does something different and more interesting: making the foil's
elements audibly unlike *each other* necessarily makes them physically unlike each other, and
how much a trial's elements differ among themselves is measurable without binding anything.
The perceptual gain and the statistical residual are the same quantity seen twice.

**One battery is one draw, so here are seven.** The audit is itself a random sample, and a
single clean report proves less than it appears to. Running the `rising` ladder at seven
independent seeds, 240 trials each:

| | seeds |
|---|---|
| shipped (band 13, 26 tones/channel) | 0.658, 0.403, 0.966, 0.289, 0.096, 0.744, 0.621 |
| the same at 20 tones/channel | 0.658, 0.206, 0.143, 0.203, **0.001**, 0.616, 0.694 |
| unbanded, 20 tones/channel | 0.281, 0.855, 0.202, 0.623 |

The shipped row has no rejection and no obvious left shift; the 20-tone row does. Three of the
seven shipped draws showed one learnt-observer flag apiece, on a *different* feature each time
and with inconsistent sign -- the signature of the learnt observers' own false-positive rate
(they are calibrated at SD 0.14 on null data, so |d'| = 0.33 is 2.4 SD, and five observers are
run per battery), not of a stable cue. The properly multiplicity-corrected test is the global
permutation, and it does not reject at any seed.

State it honestly: the mechanism has not been removed, it has been driven below what a
240-trial audit can detect. A much larger audit would find it again. The size that matters is
the one a listener could exploit in a session of this length, and at this configuration that
is bounded by the audit above.

**Envelope bursts specifically.** A synchronous onset of seven tones is a level event, so the
first thing to check is whether the two intervals differ in how often the envelope spikes or
how hard. They do not. Measured over 80 trials at step 0, the rung where the group is a
perfect chord and bursts are most likely to differ:

| envelope statistic | target | foil | d' |
|---|---|---|---|
| peaks above 3 SD | 34.95 | 35.44 | -0.04 |
| peaks above 5 SD | 1.66 | 1.70 | -0.02 |
| mean peak height (SD units) | 3.744 | 3.747 | -0.02 |
| max peak height (SD units) | 5.30 | 5.46 | -0.17 |
| crest factor | 2.250 | 2.295 | -0.23 |
| kurtosis | 4.41 | 4.46 | -0.12 |
| level | -26.426 dB | -26.424 dB | -0.31 |

The reason is structural: *both* intervals contain exactly one aligned set of seven and one
scattered set of seven per element, so a burst in one has a burst opposite it in the other.
All four burst statistics are part of the audited feature set, not a separate check, and the
learnt envelope observer runs between -0.27 and +0.64 across the `rising` ladder and -0.54 to
+0.36 across `redrawn`.

**Oracle-only differences.** Rows that need to know S differ and must: the union
occupancy of S (a chord's seven tones overlap in time, so at step 0 the union is 0.04
lower in the recurring interval); the maximum number of S channels sounding at once
(7 versus about 6 at step 0); and, by 0.1 to 0.2 tones, the chance coincidences among an
element's own channels, because a recurring channel holds 25 background tones where a
redrawn one holds about 30. None of these are visible to an observer that does not know S.

**Momentary silences.** With 8.8 tones sounding on average the cloud is moderately dense, and the
instantaneous count reaches zero briefly in most intervals (row "tones sounding: min", which
the battery reports as exactly equal between the two intervals of every trial). It does so
equally in both, so it is texture rather than a cue -- but it is texture, and a listener will
hear the background as a scatter of pips rather than a wash.

**The `ungrouped` comparison carries a non-binding route at every step, and this cannot be
designed away.** A figure that is present in one interval and absent in the other
*is* a level event: synchronous onsets are an envelope transient. Any comparison of
"grouped" against "not grouped at all" therefore differs in broadband envelope, and no
construction can match it, because matching the envelope means giving the foil synchronous
onsets, which is giving it a group. The trichotomy is real: a foil is either grouped
(and then it is the `rising` foil), or ungrouped (and then its envelope differs).

What the battery measures, per step, for an observer with access to the broadband envelope
and nothing else:

| step | 0 ms | 4 ms | 7 ms | 11 ms | 14 ms | 17 ms |
|---|---|---|---|---|---|---|
| `rising` ladder, envelope observer d' | +0.36 | -0.09 | +0.64 | -0.18 | -0.27 | +0.18 |
| `redrawn` ladder, envelope observer d' | +0.18 | +0.00 | +0.36 | +0.00 | +0.36 | +0.09 |
| `ungrouped` ladder, envelope observer d' | **+2.33** | **+1.47** | **+0.36** | **+0.95** | **+1.19** | **+1.63** |

**This changed with matched incidence, and it changed for the worse.** In the earlier
construction the ungrouped cue faded once the components spread, because the target's chord
stopped being a transient; it was confined to the two smallest steps. Under matched incidence
interval A holds an aligned group at *every* rung while the ungrouped foil holds none, so the
cue does not fade: the element-locked envelope, its modulation depth, its autocorrelation at
element-rate lags and the per-element RMS all separate the intervals at every step. A
psychometric function measured on that ladder would be an envelope-detection curve wearing a
figure-detection label at all six points, not just at two.

So `ungrouped` is no longer a main ladder. It is kept where its envelope cue is the *point*
rather than a contaminant: practice stage 1, where a listener has to hear that something
groups at all, and one control cell at step 0. The second psychometric function is `redrawn`
-- the figure returns on the same pitches but in a fresh delay order every element -- which
asks whether a consistent *order* buys anything beyond a consistent *pitch set*, and which
the battery finds clean at every step (envelope observer between -0.54 and +0.36, global
permutation over all 66 features p = 0.622, no observer x condition cell surviving
correction, no blind observer above chance).

The `rising` ladder has no envelope route at any step, which is why it carries the inference
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

**The ladder stops well short of non-overlap.** At 3 Hz with 45 ms tones the widest element
footprint that fits is 294 ms, so the largest step is 17 ms and adjacent components still
overlap by 62%. This is the sharpest limit in the design: the whole ladder lives in the upper
third of the overlap range, and if the psychometric function has not fallen by 17 ms the
experiment cannot say where it does. Tone duration is the lever, and it is a direct trade --
30 ms tones reach overlap 0.33, 45 ms reach 0.62, 60 ms reach 0.75. The others are fewer
components or a rate floor below 3 Hz, and each gives up something the paradigm was built to
keep.

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
   (§4). Expect to be asked to base any claim about the interplay on the 7 to 17 ms range,
   or to add the envelope-observer curve to the figure as a reference.
3. **Length and simultaneity are confounded with asynchrony** (above). The 2 × 2 is
   proposed, not implemented, and a reviewer will ask for at least cell (3).
4. **The element rate is in the range of rhythmic entrainment.** At 3.0 to 3.3 Hz with
   11% jitter, temporal expectation can direct attention to the elements. Because both
   intervals share the schedule this is not a cue for the judgment, but it is a mechanism
   the discussion has to own: the recurring channels are sampled at predictable times.
5. **The ladder is capped at 17 ms by the rate and the 45 ms tone**, so it never leaves the
   upper third of the overlap range, and the synchrony residual is driven below the audit's
   noise floor rather than eliminated. Both are stated with numbers; a reviewer can disagree
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
  the same pitches?", and it is the correct question on both: on `rising` the foil's group
  lands on fresh pitches every element, on `redrawn` it does too and the target's order also
  changes, and in practice, when the foil has no group at all, nothing in it comes back. The
  listener never switches task.
* **Practice runs in two stages**, each with feedback and a criterion of 10 of 12, up to
  two attempts each. Stage 1 is `ungrouped` at step 0 (a chord against a background with no
  group in it, the clearest demonstration of the target percept); stage 2 is `rising` at
  step 0, which teaches that both intervals can contain a group and only one repeats. A session that
  fails either stage stops and is recorded as `practice_criterion_not_met`, naming the
  stage that failed.
* **Main block**: 2 ladders (`rising`, `redrawn`) × 6 steps × 10 trials = 120.
  **Control block**: 2 cells × 8 = 16 (`ungrouped` at 0 ms, `onechannel` at 0 ms).
  Self-paced breaks every 40 trials and between blocks. Estimated session 36.9 minutes; the
  validator refuses a configuration that exceeds 40.
* **Half of every cell uses the anchored figure**, the other half a figure drawn fresh for
  that trial. The split is deterministic in the trial seed, recorded per trial, and is the
  within-session measure of whether a learned regularity helps.
* **Ten trials per cell is the cost of two curves.** One session gives a 95% interval about
  ±0.29 wide on each point, and half that many again on each side of the anchored split. The
  design is built for pooling: sessions of one participant are pooled by `seqsfg analyze`
  when their configuration hashes agree, and three sessions bring each point to 30 trials.
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
notebooks/           SeqSFG_playground.ipynb  the skeptic's playground (audio, self-test, live tests)
                     SeqSFG_strategy.ipynb    where the project can go (model prototype, power, plan)
tests/               50 tests: validator refusals, stimulus invariants, two-ladder balance and ordering,
                     practice stages, resume, analysis gates, and positive controls that the permutation
                     test catches a planted difference and the curve comparison catches a planted
                     difference in shape while staying calibrated on identical ladders
verification/        the battery report and JSON, and figures/ for the default configuration
demo/                example trials as WAV
```

Parameters are overridden with `--set key=value` (JSON values) or `--config file.json`,
and every command validates the result first.

## 9. The single-interval yes/no task

A second task, in `seqsfg/yesno.py`, sharing the same stimulus machinery and leaving the
two-interval experiment untouched. The listener hears **one** sound and answers whether a
figure was there.

**Yes/no is a stricter matching problem than forced choice, and the difference is easy to
underestimate.** In 2IFC only the two intervals of a trial have to be indistinguishable, and
anything common to both cancels. In yes/no the listener answers from memory of what these
sounds are usually like, so any property whose *distribution* differs between the classes is a
criterion -- a mean shift of 0.2 dB, or one extra envelope burst on average, or the same mean
with a wider spread. The audit therefore tests both, and the test statistic is a rank-sum AUC
converted to the d' of the best criterion on that feature.

**The absent class has to be chosen, not assumed.** A yes/no task contrasting "coherent onsets
present" with "coherent onsets absent" cannot be envelope-matched, because synchrony *is* an
envelope event: seven tones starting together concentrate the same energy into a shorter window
than seven tones spread across the element, and no arrangement of the same tones avoids it.
Three absent classes are implemented and all three were measured, 60 present and 60 absent
trials at each of six steps:

| absent class | what it is | permutation p | largest single feature | learnt observer |
|---|---|---|---|---|
| **`roving`** (default) | elements bound, but on a fresh band every element, so nothing recurs | **0.180** | 0.23 (`ch:frac_ioi_in_iei:sd`) | **d' = -0.04, 49.3% correct** |
| `scattered` | same channels at the same element times, components scattered so nothing binds | 0.000 | 0.42 (`env:ac_peak_iei`) | d' = +0.55, 60.8% |
| `plain` | a plain cloud, no element structure at all -- the classic SFG detection task | 0.000 | 1.79 (`env:ac_peak_iei`) | **d' = +2.60, 90.4%** |

Only `roving` survives. Against a plain cloud an ideal observer that never hears a group gets
**90% of the trials right from the envelope alone**, and at step 0 it gets 100%: the giveaway is
envelope autocorrelation at element lags and modulation power in the 3-10 Hz band, both of which
are just "a chord arrives every 316 ms". Against `scattered` the same observer still gets 61%.
Both are fatal, and neither is fixable, because the cue is the manipulation.

`roving` works because both classes contain exactly one bound chord per element -- the question
becomes "did **one** figure keep coming back?", the single-interval form of the two-interval
task. Pooled over the ladder, the cues a listener would reach for first:

| | present | absent | yes/no d' |
|---|---|---|---|
| long-term RMS (dB) | -25.930 +- 0.007 | -25.930 +- 0.007 | +0.07 |
| peak amplitude | 0.258 +- 0.020 | 0.261 +- 0.022 | +0.10 |
| envelope peaks above 3 SD | 4.472 +- 4.977 | 4.317 +- 4.389 | +0.10 |
| envelope peaks above 5 SD | 0.019 +- 0.138 | 0.028 +- 0.195 | +0.01 |
| tallest burst (SD units) | 3.602 +- 0.581 | 3.604 +- 0.585 | +0.05 |
| crest factor | 1.659 +- 0.110 | 1.658 +- 0.113 | +0.02 |
| envelope kurtosis | 4.474 +- 0.695 | 4.436 +- 0.659 | +0.05 |
| modulation power, element-rate band (dB) | 20.660 +- 1.538 | 20.622 +- 1.587 | +0.04 |

Level is not merely matched, it is *constant*: every interval of either class carries exactly
`tones_per_channel` tones in every one of the 30 active channels, so total tone count is not a
random variable at all. That is checked by a test, not by inspection.

Two things the analysis insists on. **d', not percent correct**, because the listener chooses
the criterion; percent correct confounds sensitivity with bias. And **the criterion is reported
and watched**: `analyse` prints c overall and per step, the proportion of "yes" responses, and c
in the first versus the second half, because a drifting criterion is the commonest yes/no
artefact and it inflates or deflates the pooled d' silently.

```
seqsfg yesno-verify --config pilot_config.json --absent all     # audit all three classes
seqsfg yesno-run    --config pilot_config.json --code P01       # run a session
seqsfg yesno-analyze data/P01/session_01
```

Reports for all three classes are in `verification/yesno_*_report.txt`, and
[`SeqSFG_yesno_playground.ipynb`](https://colab.research.google.com/github/MeysamAmirsardari/SeqSFG_task/blob/main/notebooks/SeqSFG_yesno_playground.ipynb)
walks through the whole argument with audio.
