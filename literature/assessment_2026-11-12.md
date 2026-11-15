# Independent assessment and recommendation (12 Nov 2026)

## Verdict

**Substantially revise.** Keep the question and the title. Stop treating timing thresholds (tcoh, tpair, tsync) as measures of grouping, and do not make tcmr the main study. Run one detection experiment: a repeating multi-tone figure in a tone cloud, with the onset lags inside the figure arranged in three ways at matched sizes. It tests a prediction of the published coherence model that I confirmed by simulation: grouping should depend on how large the lags are and never on how they are arranged.

## 1. What we actually know

I read the trial logs directly. Numbers below are recomputed from them.

| Task | Human data | What the data show | Status |
|---|---|---|---|
| seqsfg (5 variants) | 5 complete sessions, each a different variant; 7 more stopped at practice | A synchronous figure was detected 61–82% of the time. In the figure-vs-plain-cloud session (P01/09), d′ was 1.9 at 0 ms and between −0.2 and 0.5 at 4–16 ms per step, with 50% false alarms and 20 signal trials per cell | Baseline too weak to carry a function |
| tcoh | 2 complete sessions, different parameters | 1–4 ms at synchrony; 8–26 ms at lags of 19–100 ms; not monotonic | One listener |
| tshear | 1 session | 1.9 ms at synchrony; 16–20 ms at every sheared step, the same as the tone alone (22.7 ms) | One listener |
| tpair | 5 sessions across 4 design versions | 1.5–2.8 ms at synchrony; 6–46 ms elsewhere; the same condition differs 3–5× between sessions | No design run twice |
| tsync | 1 session, no thresholds | Nothing | |
| tcmr | 2 sessions on Bluetooth earbuds, uncalibrated | Masking release at synchrony was 0.2 dB, then 3.5 dB. Sheared conditions were not worse than synchronous | Not interpretable |
| tcbound, taba, trmr, tpair fast | None | | Untested proposals |
| Network model (separate repo) | Simulations only, not rerun by me | Learning slows with shear: criterion after 104 s at 5 ms per step, 241 s at 25 ms | Timescale far longer than a trial |

**What this establishes**

- **There is one listener.** Every code in `data/participants.csv` has the same age, sex and experimenter, and P00 appears to be the same person. No naive listener has completed any task in this repository. If PILOT01 and PILOT03 were other people, they never passed practice.
- **One finding repeats, within that listener.** Timing thresholds are 1–4 ms when the tones are physically synchronous, in every session of three tasks. They are several times worse at any lag of 12 ms or more.

**What it cannot establish**

- **Grouping.** In tcoh, tshear, tpair and tsync the listener judges the same quantity that is being manipulated. Detecting an asynchrony against synchrony and telling two lags apart are different judgements, and the jump from 2 ms to 15–20 ms may be that difference alone. tshear's sheared thresholds equal the tone-alone threshold. That fits "binding is lost at the smallest step" and equally "a lag is harder to judge than a coincidence".
- **Weber's law in tcoh.** The summary "threshold ≈ 0.4 × lag" holds only up to about 56 ms (ratios 0.40–0.48). At 75–100 ms the ratio falls to 0.14–0.23, as the sequence approaches even alternation and a rhythm cue appears. The curve reflects the geometry of the timing reference.
- **Any SFG function.** The matched-incidence design removed the within-channel repetition cue that makes classic SFG easy. What remained, alignment across channels at 3 Hz in a dense cloud, was barely detectable even at synchrony.
- **Human feasibility.** The verification batteries test each stimulus against model listeners. They are sound, but none of them shows that a person can do the task.

## 2. The problem worth solving

The project has been mixing four levels, and has used "persistence" in two senses.

| Level | What it is here | Where it got mixed up |
|---|---|---|
| Phenomenon | Several components heard as one object | Inferred from timing acuity |
| Manipulation | Onset asynchrony | Four different quantities under one name: the lag between two tones, the step between neighbours, the total spread, and the lag as a share of the tone or the period |
| Task | Timing threshold, detection, masking release | In the timing tasks the outcome is the manipulated quantity |
| Mechanism | Zero-lag coherence, a learned lag-tolerant association (the network model), or a fixed onset window | A rising curve was read as support for coherence, though all three predict it |
| "Persistence" | Tolerance of asynchrony (the title's sense) | Also used for carry-over in time (tpair context mode, CMR precursors) |

**The question.** When the components of a repeating figure no longer start together, does the benefit of temporal coherence depend only on the size of the onset lags? Or does it also depend on their arrangement: whether the lags repeat identically on every repetition, and whether they run in order across frequency?

- **Observable:** proportion correct in detecting the figure.
- **Latent process:** binding of the components into one object.
- **Link between them:** the coherence benefit, defined as performance with a synchronous figure minus performance on a floor condition whose components repeat without any shared timing.

Call the time from the first to the last component's onset within one repetition the **spread**.

| Account | Ordered vs scrambled (same lags) | Fixed vs redrawn each repetition | Benefit at 80 ms spread |
|---|---|---|---|
| Zero-lag coherence (published model) | Equal | Equal | None |
| Learned, lag-tolerant association | Equal | Fixed better | Present for fixed |
| Binding chained through frequency neighbours, or glide detection | Ordered better | Equal | Present for ordered |
| Fixed onset window of about 30 ms | Equal | Equal | None, and gone by 40 ms |

## 3. Approaches compared

| | A. Detect a sheared figure in a cloud | B. tcmr (latest) | C. Timing thresholds | D. Subjective report alone |
|---|---|---|---|---|
| Interpretable | Yes: the judged attribute is presence, not timing | Yes, indirectly | No | Partly: criterion-dependent |
| Naive listeners in one hour | Likely, by the literature; unproven here | Unlikely without training | Feasible | Certain |
| Reliability with 8 listeners (simulated) | 84% power for a 12-point difference | 34–42% power for 1.5 dB | Poor in the logs | Good |
| Contribution | First asynchrony manipulation inside an SFG figure that I could find | Increment on Christiansen & Oxenham (2014) | Covered by Micheyl et al. (2013) | Weak alone |

**Why A over B, the strongest alternative.** tcmr asks a good question, but its vehicle is wrong for this window.
- A 36-trial track has a standard deviation of 1.4–1.9 dB (simulated with tcmr's own rule).
- The entire masking release in the pilot was 0.2 and 3.5 dB, so an order effect would be a fraction of that.
- The repo's own literature notes say naive listeners show little across-channel release without training (Goldman et al. 2012; I did not verify this one).
- Detection of SFG figures is reported to need minimal training.

**Literature status**

- **Established** (I checked the full texts, according to PubMed):
  - Synchronous sequences give 2–4 ms asynchrony thresholds against 10–20 ms for asynchronous ones: Elhilali et al. 2009, [doi:10.1016/j.neuron.2008.12.005](https://doi.org/10.1016/j.neuron.2008.12.005).
  - With two tones and lags of 0–100 ms, "an asynchrony of 35 ms or less" was enough to break binding for some listeners, with large individual differences: Micheyl et al. 2013, [doi:10.1037/a0032241](https://doi.org/10.1037/a0032241).
  - SFG figures are detected at about 90% hits "with minimal training", though one experiment kept only listeners above d′ = 1.5: Teki et al. 2013, [doi:10.7554/eLife.00699](https://doi.org/10.7554/eLife.00699).
- **Second-hand** (the full text would not load; taken from the repo's stored extraction): Christiansen & Oxenham 2014, [doi:10.1121/1.4872300](https://doi.org/10.1121/1.4872300). A 40–60 ms offset of one band reduced masking release and 20 ms did not.
- **Unverified novelty.** A PubMed search for SFG with asynchrony, jitter or coherence returned two papers (PMIDs 27325682 and 25948273). To my knowledge neither varies timing inside the figure, but I did not re-read them. One narrow search is not proof of originality.

## 4. Recommended study

**What the participant hears and does.** Two bursts of a random tone cloud, about 1.1 s each. One contains a figure: 6 tones at fixed frequencies, each repeating 8 times, with 50-ms tones every 125 ms. The listener presses 1 or 2 for the burst with the figure, with feedback. Cloud tones have random onset times, so the cloud has no synchrony of its own.

**Manipulated variables.** The arrangement of the figure's onset lags and the spread. The 5 lags are always evenly spaced across the spread.

| Condition | Lags | Spread |
|---|---|---|
| Synchronous | none | 0 |
| Ordered | ascending with frequency, same every repetition | 40, 80 ms |
| Scrambled | same lags, random frequency order, same every repetition | 40, 80 ms |
| Redrawn | same lags, reassigned on every repetition | 40, 80 ms |
| Floor | redrawn across the whole period | 125 ms |

That is 8 conditions × 60 trials = 480 trials, about 32 minutes of testing. A 3-minute block follows in which the figure is played alone and rated "one sound or several?".

**Measured outcome.** Proportion correct per condition, converted to d′.

**Planned contrasts**
1. **Ordered vs scrambled**, pooled over both spreads. These two have identical rhythm within each channel and an identical summed envelope, so neither cue can explain a difference.
2. **Fixed (ordered and scrambled) vs redrawn**, pooled over both spreads.
3. **Descriptive:** the share of the coherence benefit left at each spread, against the model's curve.

**Analysis.** A mixed logistic model with listener as a random effect, on arrangement × spread. Benefit left is (d′ − d′ floor) / (d′ sync − d′ floor), per listener.

**Role of the model.** Run the published coherence analysis on the real target and foil waveforms. Fit one internal-noise value per listener on the synchronous and floor conditions only. The other six conditions are then predicted with nothing left to adjust. My simulation on envelopes alone (`assessment_2026-11-12_sim.py`):
- Benefit left is about 72% at 20 ms, 37% at 40 ms, 13% at 60 ms and 0% at 80 ms.
- It is exactly equal for ordered and scrambled, and within a few points for redrawn.

This is a prediction from simulation, not evidence about people. The network model does not yet predict anything within a trial, so it motivates the follow-up and is not tested here.

**Main alternative that survives**
- If ordered beats scrambled, listeners may be detecting a repeating glide. That would show the zero-lag model is insufficient without showing that coherence chains through neighbours.
- If fixed beats redrawn, each component's own regular rhythm could be responsible. A consistent lag between channels and a common period are the same thing here. The follow-up control gives each component a slightly different rate, so each stays regular while the relations drift.

**Pilot evidence needed before committing the 8 listeners.** Three listeners, at least two naive, 30 minutes each, wired headphones at a fixed volume. The criteria use only the synchronous and floor conditions:
1. At least two of three reach 85% on synchronous figures within 10 minutes of practice.
2. Synchronous minus floor is at least 15 points.
3. Split-half proportions differ by less than 10 points on average.
4. The session fits in 55 minutes.

Two adjustments are allowed, cloud density and then the number of repetitions. After that the configuration is frozen. Do not look at the arrangement contrasts in the pilot.

**Strongest remaining weakness.** The baseline is unproven. This repository's SFG attempts failed at synchrony, and I am relying on published reports that classic-style figures are easy. If the floor sits close to the synchronous condition, there is no coherence effect to study.

## 5. Publication assessment

**What it adds.** Earlier work offset one component or band against the rest and found a limit near 20–40 ms. None of it asked whether that limit applies to neighbouring components or to the total spread, or whether a repeated pattern of lags is treated differently from a changing one.

**Informative outcomes**
- All arrangements equal, and the benefit gone by 80 ms: supports the zero-lag model and puts a number on the tolerance. The title still fits; the answer is that the effect does not persist.
- Fixed better than redrawn: consistency matters as well as synchrony.
- Ordered better than scrambled: frequency order matters.

**Uninformative outcomes.** Ceiling or floor compression, or a synchronous-minus-floor gap under 15 points.

**Claims that would remain unjustified.** Anything about plasticity or learning, pre-attentive processing, carry-over in time, or individual differences from 8 listeners.

**Smallest coherent JASA study.** Eight listeners in one visit is pilot-scale: power is poor below about 12 points and nothing is replicated. The smallest paper is:
1. This experiment replicated in 16–20 new listeners with the analysis fixed in advance, adding 20- and 60-ms spreads.
2. A second experiment with the unequal-rate control and two repetition rates, to separate absolute time from phase.
3. The model comparison across both.

## Two-week plan

| Days | Work |
|---|---|
| 1–2 | Build it as a preset of the existing seqsfg generator, whose variant list already includes rising, scrambled and redrawn (I did not audit those code paths). Add the constant-stimulus two-interval block. Run the coherence model and simple energy and envelope observers on the real waveforms. |
| 3–4 | Pilot three listeners and apply the four criteria. |
| 5 | Freeze the configuration; write a dated protocol with the contrasts and exclusion rule. |
| 6–12 | Eight listeners, one visit each. |
| 13–14 | Analysis and abstract. |

## Fallback if the pilot fails

Run the same eight figure conditions without the cloud. On each trial the listener hears one figure and reports "one sound or several?" (30 trials per condition, about 12 minutes). It needs no training and gives perceived organisation against spread and arrangement, to set beside the model's curve at ARO. It is criterion-dependent and would not carry a JASA paper without an objective companion.

## Checks behind this document

- Trial logs under `data/` (accuracy by condition, thresholds, session status).
- Full texts retrieved through PubMed for the three papers marked as checked.
- `literature/assessment_2026-11-12_sim.py`: the coherence analysis by arrangement, the precision of a tcmr track, and power for the detection design. The power figures assume a listener-to-listener standard deviation of 6 points (detection) and 1.5 dB (tcmr).
