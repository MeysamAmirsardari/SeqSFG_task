# tcmr redesign notes, 2026-11-14

Status: decisions taken and plan C built on 2026-11-14 (section 13): preset tcmr_frozen, seed 20261114 checked.

## 1. Why the long-burst version had to change

- Reference frame (Christiansen and Oxenham 2014, JASA 135:3520): five 20-Hz noise bands at 0.25, 0.5, 1, 2, 4 kHz; bursts of 187.5 ms every 250 ms; 4 precursors; a synchronous final burst that carries the tone; CMR = threshold(random) minus threshold(comodulated).
- With 187.5 ms bursts at 250 ms the silent gap is 62.5 ms. Any shift is at most a third of a burst, and all bands overlap heavily at every shift. The shift is a small perturbation of a mostly synchronous sound.
- Requirements from the PI: keep the CMR frame and stay comparable to the reference; rate above 4.5 Hz; bursts of 60 ms or less, final burst the same length as the precursors; first round uses per-band asynchrony, not the ascending staircase; one visit of about 45 minutes; ARO podium and a JASA paper.

## 2. Finding: equalising every burst removes the cue

- `band_bursts` in `tcmr/stimulus.py` scaled every burst of every band to unit rms. This removes the natural burst-to-burst level fluctuation of narrowband noise, in the random and band-alone conditions too.
- Consequence 1: in the random condition all bands had exactly the same energy in every burst, so an across-band level comparison worked there as well. Random was not fully random.
- Consequence 2: the masker energy in the tone band was constant across intervals, so the plain energy cue in the tone band was cleaner than in natural noise. Random and band-alone thresholds come out lower, and CMR smaller.
- Consequence 3: for short bursts this is fatal. A 20-Hz band barely changes within 40 to 60 ms, so the burst level is the only thing that can be comodulated. Equalised short bursts are fixed-level tone pips in both conditions.
- Model listener (section 4): about 2 dB easier band-alone threshold with equalised bursts at 187.5 ms, and the level-based release drops from about 3 dB to about 0.3 dB. This probably explains part of the pilot result (3.5 dB with volume held, against 6.1 dB published). Training and AirPods remain the other suspects.
- Pilot sessions recorded before the fix (data/P00/tcmr_session_02 and _04) used equalised bursts. They are not comparable with anything recorded after it.
- Not yet confirmed: how the reference paper scaled its bursts. Natural noise is the usual practice. Read their methods before finalising.

## 3. Why noise and not tones

- Noise is not part of the question, but it is the measure. CMR only exists if the masker fluctuates and the other bands either fluctuate with it or do not.
- With fixed-level tones there is nothing to comodulate. The precursors would also give a level reference inside the tone band, so the task could be done within one band and would stop measuring grouping.
- The tone version that works is profile analysis with a roving level (Green 1988; Hill and Bailey 1997 showed that onset asynchrony disrupts it). That is a different paradigm with a different literature.
- A 40 to 60 ms burst of 20-Hz noise is physically close to a tone pip with a naturally random level and phase. Short natural-level noise gives the tone-like timing while the stimulus definition stays the reference's. In the short-burst limit CMR and roving-level profile analysis are the same across-band comparison.
- Keep 20-Hz bands. Wider bands fluctuate faster within a burst but give less release, and 100 Hz exceeds the auditory filter at 250 Hz.

## 4. Model check (literature/tcmr_short_burst_model.py)

Complex-baseband 20-Hz Gaussian noise, raised-cosine ramps, tone at band centre with random phase, 3 intervals, 70.7% correct, 12000 trials per point. "Alone" listener: picks the interval with the most energy in the tone band (1 dB internal noise). "Compare" listener: picks the interval where the tone band is highest relative to the flanking bands (internal noise 1, 2 or 3 dB). Thresholds are tone level re long-term band level (dB). The model uses level only and ignores listening in the dips within a long burst, so sizes are lower bounds for long bursts. The pattern is the result.

| burst (ms) | noise | masker level SD (dB) | alone | compare 1 dB | compare 2 dB | compare 3 dB | release at 1 dB |
|---|---|---|---|---|---|---|---|
| 187.5 | natural | 2.28 | -0.3 | -3.1 | -0.4 | 1.7 | 2.8 |
| 187.5 | equalised | 0.45 | -2.4 | -2.6 | 0.2 | 2.2 | 0.3 |
| 100 | natural | 2.97 | 1.0 | -2.4 | 0.0 | 2.0 | 3.4 |
| 100 | equalised | 0.40 | -1.2 | -1.4 | 1.0 | 2.8 | 0.2 |
| 60 | natural | 3.72 | 2.1 | -1.5 | 0.5 | 2.2 | 3.6 |
| 60 | equalised | 0.60 | 0.7 | 0.1 | 2.0 | 3.5 | 0.6 |
| 50 | natural | 3.95 | 2.4 | -1.0 | 0.8 | 2.5 | 3.4 |
| 50 | equalised | 0.67 | 0.9 | 0.5 | 2.1 | 3.6 | 0.3 |
| 40 | natural | 4.35 | 2.5 | -1.2 | 0.6 | 2.2 | 3.7 |
| 40 | equalised | 0.74 | 1.5 | 1.3 | 2.7 | 4.0 | 0.2 |
| 30 | natural | 4.71 | 2.7 | -1.1 | 0.6 | 2.2 | 3.8 |
| 30 | equalised | 0.81 | 1.8 | 1.5 | 2.7 | 3.9 | 0.3 |

Reading: with natural noise the level-based release does not shrink when bursts get shorter (2.8 dB at 187.5 ms, 3.7 dB at 40 ms), because the masker level varies more from burst to burst and the flanking bands tell the listener what it was. With equalised bursts there is almost none at any duration. How much real listeners get at 40 to 50 ms is an empirical question (pilot, section 9).

## 5. Design v1 (superseded by v2): fresh jitter on every burst

- Every band jittered independently on every precursor, uniform plus or minus J, final burst synchronous. 50 ms bursts, 5 Hz, 5 precursors. J limited to 65 ms by within-band overlap: J_max = (200 - burst - 20)/2.
- Statistics for 50 ms bursts (two bands, independent jitter):

| J (ms) | mean asynchrony (ms) | band pairs with no overlap | mean overlap | envelope correlation |
|---|---|---|---|---|
| 0 | 0 | 0% | 1.00 | 1.00 |
| 10 | 6.7 | 0% | 0.87 | 0.85 |
| 20 | 13.3 | 0% | 0.73 | 0.65 |
| 30 | 20.1 | 3% | 0.60 | 0.45 |
| 40 | 26.7 | 14% | 0.49 | 0.34 |
| 50 | 33.3 | 25% | 0.42 | 0.24 |
| 65 | 43.4 | 38% | 0.34 | 0.15 |

- Controls planned for v1: coherent jitter (same jitter in all bands: irregular but synchronous, envelope correlation 1.0), random envelope at every timing, band alone.
- Why superseded: the PI pointed out that the pattern must stay the same across repetitions so that it can be learned. In v1 nothing repeats.

## 6. Design v2 (current proposal): a frozen pattern, "rapid formation of a syllable"

Idea from the PI (2026-11-14): within a stream the order of the elements and their offsets are frozen, so every repetition is the same spectro-temporal pattern and can be learned. The pattern changes from trial to trial, so only short-term formation is measured, like the rapid formation of a syllable.

What this changes, and my recommendations:

1. Frozen offsets make every band perfectly regular at 5 Hz. Only the phase between bands differs. The rate requirement (> 4.5 Hz) holds exactly, and the irregular-rhythm confound of v1 disappears, so the coherent-jitter control is no longer needed.
2. A repeating pattern wraps around. Two bands can be at most half a cycle apart (100 ms at 5 Hz). Five bands are maximally spread when they are a fifth of a cycle apart: a step of 40 ms. Larger steps bring bands back together.
3. Use a random order over a fixed set of offsets instead of free random offsets. The tone band sits at offset 0; the four flanking bands take -2, -1, +1, +2 steps in a random order, drawn once per trial. Every trial then has the same asynchronies (neighbours in time are always one step apart, no pair is ever accidentally synchronous), and only the order varies. There are 24 orders; the order is logged on every trial, so ascending-like against scrambled orders can be compared for free. The ascending staircase and the zigzag are two of the 24.
4. Burst length 40 ms (10 ms ramps), final burst included. At the 40 ms step the five bands then tile the cycle with no overlap at all, which answers the original objection. Steps 0, 10, 20, 40 ms give overlap between neighbours of 100%, 75%, 50%, 0%.
5. Tone band at offset 0, so the tone band is regular right through the final burst and its timing is identical in every condition (same logic as the reference, where only the flanking bands move). The last precursor of the latest band ends 80 ms before the final burst at the 40 ms step.
6. Freeze the timing, not the noise. Every burst still gets new noise with its natural level. If the noise were frozen too, the masker in the last repetition would be known from the earlier ones and the tone could be detected within one band.
7. Same pattern in the three intervals of a trial, new pattern on every trial. The intervals then differ only in the tone. Bonus: the tone's interval position (1, 2, 3) corresponds to 5, 10, 15 repetitions heard, a free look at build-up.
8. A control that tests the learning claim directly: the same offsets, but re-ordered on every repetition (nothing to learn). If the frozen pattern keeps more release than the re-ordered one, repetition binds the pattern. Needs one constraint at the 40 ms step: a band may not jump from the last slot to the first (bursts would touch). Note it makes each band irregular, so if it differs from frozen, the coherent-jitter control from v1 is the follow-up.
9. The final burst stays synchronous, as in the reference. The sound that carries the tone is then physically identical in every condition.

Proposed parameters (reference in brackets): bands 5 x 20 Hz at 0.25 to 4 kHz, 60 dB SPL (same); burst 40 ms with 10 ms ramps (187.5 ms, 20 ms); 5 Hz (4 Hz); 5 precursors = 1 s (4 = 1 s); final burst synchronous with the tone (same); natural burst level; 3 intervals, 2-down-1-up, 8/4/2 dB (same); tone at 1 or 2 kHz at random (1 kHz).

Proposed session: comodulated and random at frozen steps 0, 10, 20, 40 ms (8 tracks) + comodulated and random with re-ordered repetitions at the largest step (2) + band alone (1) = 11 tracks of 32 trials, about 46 minutes with instructions and practice. To shorten: drop the 10 ms step (9 tracks of 36).

## 7. What the theories predict

- Zero-lag temporal coherence (Shamma, Elhilali, Micheyl): binding needs synchrony. Release falls with the step, and frozen equals re-ordered.
- Binding by repetition / pattern learning (McDermott, Wrobleski and Oxenham 2011, sources recovered from embedded repetition; Agus, Thorpe and Pressnitzer 2010, rapid formation of auditory memories for noise; Barascud et al. 2016, repeating tone patterns detected within about one cycle): a frozen pattern becomes one object, so it keeps or regains release relative to the re-ordered control.
- Simultaneous grouping dominates: the synchronous final burst regroups the bands whatever came before, and release does not depend on the step. The reference found a reduction with fixed offsets up to 60 ms, so an effect is expected.
- References in this section are from memory. Verify before citing.

## 8. Ideas for later rounds

- Number of repetitions (1, 2, 3, 5, 8) at the largest step: the time course of formation. Stream build-up predicts release falling with repetitions; pattern learning predicts it recovering. This is the variable that actually measures "rapid".
- Test inside the pattern: make the final repetition asynchronous too and ask whether the tone band is compared with the others within the learned pattern. More direct, but the across-band cue is then not simultaneous, so a drop could be sensory rather than grouping. Only with the frozen against re-ordered contrast.
- Fixed-order staircase and zigzag as their own conditions, if the order analysis from round 1 suggests order matters.
- Bridge condition: the exact reference stimulus (187.5 ms, 4 Hz, synchronous, comodulated and random) in the same listeners, to tie the short-burst release to the published one.

## 9. Risks and the pilot

- Unknown: release at 40 to 50 ms in naive listeners. Pilot first, 2 to 3 people, wired headphones, volume held: synchronous only, comodulated, random, band alone, two tracks each, about 15 minutes. Go if release is at least 3 to 4 dB. If not, lengthen to 60 or 80 ms before changing anything else (at 80 ms the largest non-overlapping pattern needs fewer bands or a slower rate, so this is a real trade).
- The final synchronous burst may regroup the bands (see section 7).
- Precision: one track of 32 to 36 trials gives roughly plus or minus 2 dB per listener per threshold. Plan 14 to 16 listeners.
- 24 orders recur across the session (about 8 times each), so long-term learning across trials is small but not zero. Avoid immediate repeats; the order is in the log.

## 10. Open decisions for the PI

1. Random order over fixed offsets (recommended) or free random offsets per band.
2. Burst 40 ms (recommended: zero overlap at the 40 ms step) or 50 ms.
3. Same pattern in the three intervals of a trial (recommended) or a new one per interval.
4. Include the re-ordered control in round 1 (recommended).
5. Tone band at the temporal centre of the pattern (recommended).

## 11. What implementing v2 needs (not done)

- New arrangements in `tcmr/config.py`: random order frozen within a trial, and re-ordered on every repetition; offsets drawn per trial after the tone band, stored in the trial record.
- Gap rules for periodic patterns in `validate`; preset `tcmr_frozen.json` (burst 40, ramp 10, onset to onset 200, 5 precursors, conditions above, ratings at steps 0, 20, 40).
- Analysis and plots: release against step for the new arrangement, frozen against re-ordered contrast, order analysis, percent correct by interval position.
- Tests, README, a pilot preset (synchronous only), and the listening demo with frozen patterns.

## 12. Status of the code (2026-11-14, evening)

Done, not committed:
- `tcmr/config.py`: new field `burst_level`, "natural" by default, "equalised" keeps the old behaviour. The presets need no change.
- `tcmr/stimulus.py`: bursts are scaled by the long-term rms, so a burst keeps its natural level. The random draws are unchanged, so the same seed gives the same noise, only not rescaled.
- Natural noise has no hard ceiling. If a trial's noise happens to exceed full scale it is drawn again and the trial carries the new seed, so the log still regenerates what was played. A session can no longer stop on a rare tall peak.
- `tcmr verify`: the headroom check now looks at 6 trials per condition without redraws, and a new check reports the burst-level variation.
- `--resume` continues a session with the burst level it started with. `tcmr analyze` flags sessions recorded with equalised bursts.
- Two new tests in `tests/test_tcmr.py` (natural level statistics; redraw on a tall peak). 11 of 11 tcmr tests pass.

Not done: everything in section 11 (the frozen-pattern design), the pilot, and checking the reference's methods for how bursts were scaled.

## 13. Decisions (2026-11-14, evening) and the round-1 plan

Figure: `literature/tcmr_frozen_paradigm.png`. Numbers: `literature/tcmr_frozen_plan.py` (prints every table below).

Decisions:

1. **The pattern is a staircase, shuffled.** Offsets are -2s, -s, 0, +s, +2s. The tone band keeps 0 and the four flanking bands take the other four offsets in a random order (24 orders). The noise is never frozen: every burst is new noise at its natural level.
2. **Frozen within an interval, new for each interval.** An interval's 5 precursor repetitions all use its order. The three intervals of a trial get three different orders, drawn from the 24 without replacement, so each order is used 4 times per 32-trial track.
   - Why (the user's point): with one order for the whole trial, a tone in interval 3 would follow 15 repetitions of the pattern and a tone in interval 1 only 5. How much could be learned would then depend on where the tone is. Learning helps at 40 ms and does nothing at step 0, so this would shrink the drop with step.
   - It also keeps the re-ordered control honest: with one sequence per trial, its 1-s sequence of orders would repeat across the three intervals and could be learned.
   - Cost: the free look at build-up by interval position is gone; it moves to the repetitions round in section 8. The three intervals differ in their order, but the order says nothing about where the tone is.
   - This replaces recommendation 3 in section 10.
3. **50-ms bursts with 10-ms ramps.** Neighbouring bands overlap by (50 - s)/50: 100, 80, 60 and 20% at s = 0, 10, 20 and 40 ms. At 40 ms every band overlaps its two neighbours by 10 ms; the last band also overlaps the first band of the next repetition.
4. **The re-ordered control only if there is time.** See plans B and C.
5. **Kept from section 6:**
   - 5 Hz, 5 precursors, a synchronous final burst carrying the tone, the tone band at offset 0, natural burst level;
   - 3I-3AFC, 2-down 1-up, steps 8/4/2 dB, 32 trials per track, tracks interleaved round by round.
   - The tone stays at 1 kHz, as in the reference and the P00 pilots (section 6 had proposed 1 or 2 kHz at random).

**Timing.**
- One interval: an 80-ms lead, 5 x 200 ms, then the 50-ms final burst = 1130 ms at every step. The tone band's bursts sit at 80, 280, ... and 1080 ms in every condition.
- One trial: 3 intervals + 2 x 400 ms = 4.19 s of sound. With the box, the answer, feedback and the blank, a trial takes 6.5 s at a naive pace; P00 measured 6.1 s in the pilots.
- Last silent gap before the final burst, by step: 150 ms (0), 130-170 (10), 110-190 (20), 70-230 (40).

| plan | tracks | trials | trial minutes | visit, naive (P00 pace) |
|---|---|---|---|---|
| A core: comodulated + random at 0, 10, 20, 40 ms, + band alone | 9 | 288 | 31.2 | 44.5 (42.5) min |
| B the control in place of the 10-ms step | 9 | 288 | 31.2 | 44.5 (42.5) min |
| C core + control (comodulated + random, re-ordered, 40 ms) | 11 | 352 | 38.1 | 51.4 (49.0) min |

Visit = setup 5 + questionnaire 1 + instructions and 2 examples 1.5 + practice 0.9 (8 trials, easy levels, step 0) + trials + 3 breaks of about 1 min (the pilots' breaks ran 0.3-2.3 min) + ratings 1 (6 sequences) + debrief 1.

**Listeners.** The quantity is the per-listener drop in CMR from step 0 to step 40 ms: four tracks, one each. Its measurement SD comes from tcmr's own staircase and shared-slope fit, simulated at 32 trials:

| case | psychometric slope, lapses | one track | drop in CMR |
|---|---|---|---|
| practised | 2.2 dB (P00, session 02), 2% | 1.2 dB | 2.4 dB |
| naive, middle | 3.0 dB, 3% | 1.5 dB | 3.1 dB |
| naive, poor | 3.8 dB (P00, session 04), 5% | 2.1 dB | 4.5 dB |

P00's repeated tracks differed by 0.4 to 2.1 dB (single-track SD 0.9 dB), in line with the practised case. Listeners are assumed to differ in their true drop with an SD of 1.5 dB. The test is a paired t-test, two-sided alpha 0.05.

| drop in CMR, 0 to 40 ms | practised: N for 80% [90%] | naive, middle | naive, poor |
|---|---|---|---|
| 2.0 dB | 18 [23] | 26 [34] | 47 [62] |
| 2.5 dB | 13 [16] | 18 [23] | 31 [41] |
| 3.0 dB | 10 [12] | 13 [17] | 22 [29] |
| 4.0 dB | 7 [8] | 9 [11] | 14 [17] |

With N = 20, the smallest drop found with 80% power is 1.9 dB (practised), 2.3 dB (middle) and 3.1 dB (poor). The control's contrast (frozen minus re-ordered at 40 ms) has the same SD, so it has the same detectable size. Its true size is limited by how much release the frozen pattern still has at 40 ms.

**Recommendation.**
- **Sample size:** 20 completers, recruiting 24. Christiansen and Oxenham's Exp. 3 lost CMR entirely at 60 ms, and the pilot CMR at synchrony was 3.5 dB, so a full loss is a drop of about 3-4 dB. That drop is covered even in the poor case.
- **Exclusions:** do not exclude listeners on CMR measured in the same tracks. Selecting large CMRs at 0 ms inflates the drop by regression to the mean. Exclude only for an incomplete session, failed practice, or audio faults.
- **Plan:** A for round 1 if the visit must stay at 45 min. B if the PI values the learning test over the shape of the curve between 0 and 20 ms. C if a 50-minute visit is acceptable. The control powers only a large learning effect at N = 20. Its stronger form is the repetitions round in section 8 (1, 2, 3, 5 and 8 repetitions, frozen vs re-ordered).
- **Pilot first, as in section 9:** 50-ms bursts at synchrony only (comodulated, random, band alone, 2 tracks each), about 15 min. Go if CMR at synchrony is at least 3 dB.

**Built (2026-11-14, late evening).** Plan C is implemented as `tcmr/configs/tcmr_frozen.json`. The user chose it as the first pilot, with comodulated and random tracks; it replaces the sync-only pilot recommended above.
- **New arrangements:** `frozen` and `reordered`. The session balances the orders over each track, and every trial's orders are logged.
- **Seed check:** `python -m tcmr check` replays and checks every trial a seeded session will play. Seed 20261114 passed:
  - 720 trials (examples, practice, the 352 main trials and their first repeats, the ratings);
  - tallest peak 0.55, every frozen order exactly 4 times per track;
  - a simulated session with that seed regenerated all 368 logged trials bit for bit.
- **Old presets:** they keep their hashes and their sounds, byte for byte.
- **Still open:** how the reference scaled its bursts, and an analysis by order.
