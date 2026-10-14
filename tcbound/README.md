# tcbound: how far can a figure be sheared before its parts stop binding?

Four pure tones repeat together at 3 Hz. Their onsets are sheared: tone *k* starts *k × step*
after tone 1, so the figure runs from a chord (step 0) to an arpeggio (64 ms). One tone, the
target, gets louder on the last repetition in one of two intervals. Two trial types, randomly
interleaved, make that judgment depend on opposite perceptual organisations. Across the shear
steps, the result is the point at which temporal coherence stops binding the target into the
figure.

```bash
python -m tcbound design                     # conditions, levels, ceilings, duration
python -m tcbound verify                     # every check that needs no listener (19 checks)
python -m tcbound model                      # predictions, fixed before any data
python -m tcbound demo --out demo/           # one WAV per condition
python -m tcbound run --data data_tcbound    # a session (about 36 min)
python -m tcbound analyze data_tcbound/L01/tcbound_session_01 --out out/
python -m tcbound group data_tcbound/L*/tcbound_session_* --out group_out/
python -m tcbound export data_tcbound/L01/tcbound_session_01 --out stimuli/   # for the RNN
```

The protocol for the main study, with hypotheses, sample size and exclusion rules fixed in
advance, is [PROTOCOL_2026-10-14.md](PROTOCOL_2026-10-14.md).

## The question

Temporal coherence (Elhilali et al. 2009; Shamma et al. 2011) says that channels whose
responses rise and fall together are bound into one object. A chord is maximally coherent.
Shearing it in time lowers the coherence between the channels without changing any channel on
its own: every tone still repeats at 3 Hz with the same duration and level. So the shear is a
one-parameter manipulation of exactly the quantity the theory is about. The question is where
along it the target stops being part of the figure, and whether that bound is where the model
puts it.

The jitter task (`tshear`) cannot answer this cleanly. Its probe is a timing judgment, and timing
is what the shear manipulates, so detecting asynchrony from synchrony and discriminating it on a
pedestal of asynchrony are different tasks at different steps. Here the probe is a level
increment, which is the same physical event at every step.

## The stimulus

| | |
|---|---|
| tones | 683, 1229, 2241, 3985 Hz: inharmonic, at least 4.4 ERB apart, no common fundamental |
| target | tone 3 (2241 Hz), the same on every trial; an interior tone, so it has a neighbour on each side in frequency and in time |
| timing | 60 ms tones, 10 ms raised-cosine ramps, 3 Hz, 5 repetitions; the 5th is the probe |
| shear | 0, 8, 16, 32, 64 ms between adjacent onsets (doubling) |
| level | 55 dB SPL per tone nominal; roves put single tones between 43 and 79 dB SPL |
| intervals | 2I-2AFC, 400 ms apart, same timing and phases in both |

The steps double because the plausible bound runs from about 10 to 45 ms. At 64 ms adjacent
tones no longer overlap and the combined onset train (64/64/64/141 ms) is far from the
isochronous 83.3 ms, the rhythm trap that produced artefacts in the timing tasks.

## The probe, and why it measures binding

Per interval, with *c* = 0 by default and every other quantity uniform over ±12 dB:

| | repetitions 1 to 4 | probe, type A | probe, type B |
|---|---|---|---|
| the other three tones | c | c + v | c + w |
| the target | c + d | c + v (+Δ) | c + d (+Δ) |

| cue | type A | type B |
|---|---|---|
| target re the others on the probe (timbre) | **Δ exactly** | Δ + d − w (noise) |
| target re its own earlier tones (own stream) | Δ + v − d (noise) | **Δ exactly** |

**A** resets the probe to a flat profile. The target is louder than the others at that moment,
which a listener can read if the four tones are one object (profile analysis; Green 1988).
**B** lets the others jump while the target holds. The target is louder than its own past, which
a listener can read if the target is a stream of its own.

Three properties hold by construction and are checked numerically by `verify`:

1. **Repetitions 1 to 4 are drawn by the same code in A and B.** The organisation that builds up
   before the probe is the same in both, and the listener cannot know the type in advance.
2. **At the probe, the jump of the other tones has the same distribution in A and B** (*v* and
   *w*), **and so does the jump of the target relative to them** (−*d* and −*w*). The types differ
   only in which comparison stays exact.
3. **No shortcut works.** For each type, an ideal observer of any single other cue needs at least
   8.5 dB of increment to reach 79%: target level across intervals, target against the others'
   earlier tones, change of timbre into the probe, loudness of the probe, loudness change, the
   two nearest neighbours.

The instruction says the target is louder on the last repetition. It does not say louder than
what. Naming either reference would tell the listener which cue to use.

### What changed from the level-probe notebook

`tshear.levelprobe` roved type A on every repetition and left the others flat in type B until the
probe. Its B trials kept a third cue nearly intact: the change of timbre into the probe varied
only by the mean of three independent flanker jumps, and an ideal observer of that cue reached
79% at **4.0 dB**. That is half the intended ceiling, and it is a cue a *fused* listener uses, so
B did not force segregation. The scheme above closes it (8.5 dB) and adds properties 1 and 2,
which the old scheme lacked: its A trials fluctuated on every repetition and its B trials did not.

## Measures

**Threshold.** 3-down 1-up on SRS = 20 log10(ΔA/A) (Levitt 1971). SRS starts at +4 dB
(ΔL 8.3 dB) and moves in 4 dB steps to the first reversal, then 2 dB. The threshold is the mean
of the next 4 reversals. Simulated against slopes from shallow to steep, this takes a median of
26 trials. Spread is 3 dB and bias under 2 dB. A level increment has a shallow psychometric
function in SRS (d′ doubles per 6 dB), so that slope, not the rule, sets the precision. A
maximum-likelihood fit to all trials is no better. A track whose averaged reversals touch the cap
(ΔL 12 dB) is ceiling-limited and enters means at the cap, flagged.

**Binding index**, BI(s) = threshold(B, s) − threshold(A, s). Positive when the target is bound.
The conventional, robust demonstration that shear changes which comparison the listener can make.

**Timbre weight**, w(s). Both cues vary at random on every trial (the roves see to that), so
each response can be regressed on the difference between the intervals in each cue. That is a
probit pooled over A and B trials at one step, with w = b_timbre / (b_timbre + b_own)
(molecular psychophysics; Berg 1989; Dai & Micheyl 2010). w = 1 means the listener read only the
timbre, w = 0 only the target's own stream.

**The bound**, s50: the shear at which w has fallen half-way from w(0) to w(64). This is the
model's own definition of its half-way point, so the two are directly comparable. A second
estimate fits the model's curve stretched in time, w(s) = a + b·ρ(s/α). α = 1 means binding
decays as the model says, α = 2 that it persists to twice the shear.

The bound comes from w and not from BI because BI saturates. Once a listener stops using the cue a
type preserves, the threshold climbs to the shortcut ceiling and stays there. Two simulated groups
were run whose binding follows the model exactly, so the true half-way point is 18.2 ms. In those
groups the half-way point of BI landed at 23 and 30 ms. The half-way point of w landed at 17 and
18 ms, each inside its bootstrap interval over listeners.

## Controls

| control | what it rules out |
|---|---|
| **Complementary types A and B** | a general difficulty that grows with shear: it would raise both thresholds, and BI and w would not move |
| **Captors at step 0** (A_cap, B_cap): the other three tones alone, half-way between repetitions | that the effect is about asynchrony rather than coherence. Onsets stay synchronous, but the others now repeat at 6 Hz and the target at 3 Hz. The model puts ρ_T at 0.71, as for a 13 ms shear, so the captor condition has a predicted position on the shear curve that nothing was fitted to |
| **Target alone** (B_alone) | how close B gets to having no figure at all: the segregation asymptote |
| **Shortcut ceilings** | that a listener did the task without the comparison the type preserves |
| **Catch trials**, 6%, no rove, ΔL 8 dB at step 0 | inattention. More than 15% missed makes the session unusable |
| **Interleaving and fillers** | strategy switching. All 13 tracks run in random cycles to the end, and an unscored filler breaks any run of more than 2 trials of one condition or 3 of one type |
| **Practice to criterion**, familiarisation with the target alone | that early thresholds measure learning what to listen for |
| **Calibration and drift check** | an unknown or changing level: the system volume and device are recorded and checked |
| **Peripheral audit** | that the flankers mask the target: at the worst-case levels the target leads every flanker at its own place by 18 dB |

## Predictions

Target-to-figure coherence ρ_T, the mean normalised coherence between the target channel and
each other channel (`tcoh.model`, rate filters 2 to 32 Hz):

| reading | 0 | 8 | 16 | 32 | 64 ms | half-way | captors |
|---|---|---|---|---|---|---|---|
| half-wave coincidence (validated against Elhilali et al. 2009) | 1.00 | 0.85 | 0.64 | 0.37 | 0.17 | **18 ms** | 0.71 |
| magnitude | 1.00 | 0.98 | 0.94 | 0.85 | 0.65 | **35 ms** | 0.81 |

The two readings differ in what they make coherence depend on. Half-wave rectification lets
the fast rate filters dominate, and they lose the 60 ms pulses once these stop overlapping. The
magnitude reading weights the slow 3 Hz repetition, which a 64 ms shift barely dephases. The
step range is chosen so that the two predict visibly different curves. The measured bound is
therefore a test of the model, not only a number.

## Hypotheses

Stated in full, with tests and decision rules, in the protocol.

- **H1 (primary):** BI falls with shear. Per-listener slope of BI over the five steps, one-sided
  t-test across listeners.
- **H2:** w falls with shear.
- **H3:** captors at synchrony lower w (and BI) relative to step 0.
- **H4 (manipulation check):** A at step 0 and B at 64 ms are more than 2 dB below their shortcut
  ceilings, so the preserved cues were in use.
- **Estimate:** the bound s50 with a CI over listeners; α under each reading of the model.

## What is new

- **A bound, not a contrast.** Earlier work compares synchronous with grossly asynchronous
  sounds. Hill and Bailey (1997), replicating Green and Dai (1992), raised profile-analysis
  thresholds 6 to 16 dB with a single 320 ms onset asynchrony. Micheyl et al. (2013) found
  intensity changes in a target tone easier to detect with asynchronous than with synchronous
  random-intensity distractors. Darwin and Ciocca (1992) removed a component from a pitch with
  300 ms. This design sweeps the shear in a repeating figure and locates where binding gives way.
- **A double dissociation inside one task.** A needs binding and B needs segregation, with the
  same stimulus statistics up to the probe and one instruction. Shear has to move them in
  opposite directions, which a general difficulty cannot do.
- **A trial-level binding measure.** The roves make every trial a perturbation experiment, so w
  is estimated from every response. It does not saturate the way thresholds do, and it is what
  locates the bound.
- **A model test with a held-out condition.** Two readings of the coherence model predict
  bounds of 18 and 35 ms. The captor condition is predicted from the shear conditions without
  being fitted.
- **Stimuli exportable trial by trial** for model observers (`export`), so an RNN can be run on
  exactly what each listener heard and compared response by response.

## Session

| | |
|---|---|
| conditions | 13 tracks: A and B at 5 steps, A_cap, B_cap, B_alone; plus catch trials |
| trials | about 398 (349 to 429 over 10 simulated sessions), about 25 per track, 23 of them catch and 19 fillers |
| time | **36 min** median, 39 min at the 90th percentile (28 main, 4.5 rests, 2 familiarisation, 1 practice, 1 instructions); 41 min at the 90th percentile if practice needs all three rounds |
| per listener | two sessions on different days. Thresholds are averaged; the second session gives repeatability |

`configs/tcbound_main.json` is the frozen preset (hash `5010d4525779b21a`).
`configs/tcbound_screen.json` (A and B at 0 and 64 ms, 12 min) is for piloting the apparatus and
the experimenters' own training. It is not part of the protocol for participants.

## Checks

`python -m tcbound verify` runs 19 checks. All pass.

- The preserved cue is exact.
- Every other single cue needs at least 8.5 dB.
- A and B are indistinguishable before the probe (KS tests).
- The waveform reads back the logged gains to 0.014 dB and the onsets to the sample.
- Nothing clips (worst case 0.83 FS).
- The target dominates its cochlear place by 18 dB.
- The staircase is the multiplicative `tcoh` rule on ΔA/A, reversal for reversal.
- The session fits in 40 minutes.

Simulated listeners also run through the real runner and the real analysis:

- One whose binding follows the model gives the model's bound (18 ms).
- One for whom shear changes nothing gives no effect and an undefined bound.

The test suite (`tests/test_tcbound.py`, 26 tests) includes one where an interrupted and resumed
session is identical, trial for trial, to an uninterrupted one. In another, `export` rebuilds
every logged trial exactly.

## Known limits

- **The premise is perceptual, not informational.** A listener who could read every tone's
  level separately and compare across streams at will could do both types. The design rests on
  the evidence that across-component level comparison needs grouping (Green & Dai 1992; Hill &
  Bailey 1997) and that following one component needs segregation (Micheyl et al. 2013). The
  ceilings rule out the shortcuts that need neither, which is what a stimulus can rule out.
- **Captors may act by adaptation or inhibition as well as by grouping.** Roberts and Holmes
  (2006) attribute a captor effect on vowel timbre to broadband central inhibition. The captor
  condition shows that w responds to a coherence manipulation without asynchrony and lands where
  the model says. It does not by itself prove the mechanism is grouping.
- **The figure's overall level falls as it shears apart:** four overlapping tones at step 0, one
  at a time at 64 ms. Both intervals share a step, so this is not a cue to the answer. Neither
  type's surviving cue depends on overall level.
- **w assumes a listener who adds the two cues.** A listener who switches between them from trial
  to trial gives w as the mixing proportion, which is still the quantity of interest. A listener
  who uses a cue outside the two lowers both weights together, and their ratio is unaffected.
- **Five repetitions are 1.3 s of build-up.** Coherence-based grouping may keep strengthening
  over seconds, so the bound measured here is the bound at this exposure.

## References

- Berg, B. G. (1989). Observer efficiency and weights in a multiple observation task. *JASA* 86, 1743–1746.
- Dai, H., & Micheyl, C. (2010). Psychophysical reverse correlation with multiple response alternatives. *JEP:HPP* 36, 976–993. https://doi.org/10.1037/a0017171
- Darwin, C. J., & Ciocca, V. (1992). Grouping in pitch perception: effects of onset asynchrony and ear of presentation of a mistuned component. *JASA* 91, 3381–3390. https://doi.org/10.1121/1.402828
- Elhilali, M., Ma, L., Micheyl, C., Oxenham, A. J., & Shamma, S. A. (2009). Temporal coherence in the perceptual organization and cortical representation of auditory scenes. *Neuron* 61, 317–329.
- Glasberg, B. R., & Moore, B. C. J. (1990). Derivation of auditory filter shapes from notched-noise data. *Hear. Res.* 47, 103–138.
- Green, D. M. (1988). *Profile Analysis: Auditory Intensity Discrimination.* Oxford University Press.
- Green, D. M., & Dai, H. (1992). Temporal relations in profile comparisons. In Cazals, Demany & Horner (Eds.), *Auditory Physiology and Perception*. Pergamon.
- Hill, N. I., & Bailey, P. J. (1997). Profile analysis with an asynchronous target: evidence for auditory grouping. *JASA* 102, 477–481. https://doi.org/10.1121/1.419720
- Levitt, H. (1971). Transformed up-down methods in psychoacoustics. *JASA* 49, 467–477.
- Micheyl, C., Hanson, C., Demany, L., Shamma, S., & Oxenham, A. J. (2013). Auditory stream segregation for alternating and synchronous tones. *JEP:HPP* 39, 1568–1580. https://doi.org/10.1037/a0032241
- Micheyl, C., Hunter, C., & Oxenham, A. J. (2010). Auditory stream segregation and the perception of across-frequency synchrony. *JEP:HPP* 36, 1029–1039. https://doi.org/10.1037/a0017601
- Moore, B. C. J., & Glasberg, B. R. (1987). Formulae describing frequency selectivity as a function of frequency and level. *Hear. Res.* 28, 209–225.
- Oxenham, A. J., & Dau, T. (2001). Modulation detection interference: effects of concurrent and sequential streaming. *JASA* 110, 402–408. https://doi.org/10.1121/1.1373443
- Roberts, B., & Holmes, S. D. (2006). Asynchrony and the grouping of vowel components: captor tones revisited. *JASA* 119, 2905–2918. https://doi.org/10.1121/1.2190164
- Shamma, S. A., Elhilali, M., & Micheyl, C. (2011). Temporal coherence and attention in auditory scene analysis. *Trends Neurosci.* 34, 114–123.
