# Final recommendation: a grouping-helps measure to mirror Micheyl et al. (2013) Exp. 3

## 1. Bottom line

Stop using tsync, and staircases in general, as the headline measure. Run **trmr**: rhythmic masking release with pure tones, where low A "captor" tones lag their high B "distractor" partners by D. The listener only has to say whether a hidden B rhythm was steady or galloping. That judgement has no pedestal and no Weber's law. The B waveform is bit-identical at every D, so any strategy that uses one stream alone has to give the same score at every D, and a fall in d′ with D can only come from A-B binding. Use constant stimuli: 30 trials per condition, 12 naive listeners, about 48 minutes in one visit. About 70% of the code can come from `/Users/eminent/Res/SeqSFG/tsync/`. On the frank question: tsync's logic is defensible (Christiansen & Oxenham 2014 published the same "synchronous test after offset context" design in JASA), but it has a pedestal that grows with D, and running a staircase on a mixture of percepts is exactly why its thresholds came out noisy or at ceiling.

## 2. Is it bistable?

**Plausibly yes, but here it shows up as a mixture of percepts across short trials, not as switching within a trial. Nobody has yet measured switching as a function of A-B onset delay, so that measurement would be a new result.**

What supports it:
- Auditory streaming is genuinely bistable: percepts alternate spontaneously, with dynamics and volitional control like visual plaids (Pressnitzer & Hupé 2006; Moore & Gockel 2012).
- Micheyl 2013 Exp. 1: at 15 semitones, synchronous A/B tones were reported as "two streams" about 76% of the time (alternating tones about 97%). Individual curves differed a lot, and one listener (L7) reversed.
- Micheyl 2013 Exp. 3: listeners L6 and L8 jumped abruptly between D = 0 and 35 ms. A step like that is what a mixture of two discrete states predicts; a graded binding mechanism would give a ramp. The group mean can look graded only because it averages these steps (Denham & Winkler 2006).
- Objective performance follows the momentary percept, over and above stimulus parameters (Billig & Carlyon 2016: t(11) = 4.15, d = 1.80).

Limits on the claim:
- Dominance phases last several seconds (mean 7.7 s at 5 semitones in Rankin et al. 2015).
- The first percept is biased toward grouping and lasts longer than later ones (Hupé & Pressnitzer 2012).
- So a fresh 2.4-4 s trial mostly samples one state, the first percept. Each condition is then a mixture across trials: Pc(D) = π(D)·Pc_grouped + (1 − π(D))·Pc_floor.

What this means for the measure:
1. **Staircases are the wrong tool.** A mixture gives a psychometric function that is shallow and capped below 100%, so tracks wander or pin at ceiling. This is a plausible (not proven) reason why tpair and tsync looked the way they did. Use fixed levels (constant stimuli) instead.
2. **Make π(D) the quantity you report.** Fit each listener's data with the mixture model, which gives D50 (the delay at which half the trials are grouped) and a steepness σ. A step-like individual curve under a graded group mean is the bistability signature.
3. **Keep trials short, fresh and the same length at every D.** That holds build-up constant. It also puts every trial in the grouping-biased first percept, which gives a grouping-helps task its widest range.
4. **Add a 3-minute subjective block** (one or two streams, Micheyl Exp. 1 protocol) to give P(two streams | D) in the same listeners.
5. Long continuous reporting as a function of D is the obvious follow-up for JASA or EEG. It does not fit this visit (see section 5).

## 3. Recommended paradigm: trmr

### How the proposals scored (1 = poor, 5 = strong)

| Criterion | RMR-D (P1) | PTDD (P2) | trmr (P3) | **Final (P3 chassis + P1 grid and capture control)** |
|---|---|---|---|---|
| Validity of the grouping inference | 4 | 4 | 4 | **5** |
| Free of Weber/pedestal and single-stream confounds | 5 | 3 | 4 | **5** |
| Handles bistability | 4 | 5 | 3 | **4** |
| Effect and reliability, ~12 naive listeners, one visit | 3 | 2 | 5 | **4** |
| Comparability with Micheyl Exp. 3 | 4 | 3 | 5 | **5** |
| Deadline risk (higher = safer) | 3 | 2 | 4 | **4** |

Why the scores came out this way:
- **PTDD** loses on confounds and feasibility. Its across-D d′ curve still carries the step from a synchronous to an asynchronous reference (Zera & Green 1993b). Its mediation test is underpowered. It asks naive listeners to do a dual task, and it moves to 9 semitones.
- **trmr (P3)** has the best economy. But its distractors are placed at random times off the grid, which opens a B-only "find the periodic subset" cue. That cue is flat across D, but it raises the floor.
- **RMR-D (P1)** keeps every tone on a 200-ms grid and adds a capture control that is specific to grouping. Both are adopted below.

### Stimulus

- **Tones** (as Micheyl 2013):
  - B = 1000 Hz; A = 420.45 Hz (15 semitones below).
  - 100 ms each, including 20-ms raised-cosine ramps; random starting phase; 48 kHz (tsync default); diotic over the MacBook headphone output.
  - Nominal 60 dB SPL. **Every** tone, A and B, is roved independently by ±4 dB (uniform), so "louder event" is not a reliable cue for a dyad.
- **Grid:** slots every 200 ms (5 Hz, Micheyl's SOA). All B tones start exactly on a slot onset. Never move distractors off the grid.
- **Trial:** 19 slot onsets (0 to 3.6 s), about 3.7 s of sound. Sound starts 0-200 ms (uniform jitter) after a ready cue.
- **Target rhythm** (B tones with no captor), 7 tones in each case, same first and last slot:
  - STEADY: slots {0, 3, 6, 9, 12, 15, 18}, inter-onset interval 600 ms.
  - GALLOP: inter-onset intervals alternate 400 and 800 ms. The phase is random: either slots {0, 2, 6, 8, 12, 14, 18} or {0, 4, 6, 10, 12, 16, 18}.
- **Distractors:** each of the 12 non-target slots holds a B tone with probability p. Start at p = 0.5, which gives about 6 distractors. p is the only difficulty knob, tuned between 0.4 and 0.7 in the pilot.
- **Captors:** every distractor, and only distractors, gets one A tone starting D ms after it. A lags B, matching the tcoh/tsync code (`tsync/config.py`). Confirm the sign against the Micheyl 2013 Methods before freezing the design.
- **Main conditions (8), interleaved:**
  1. D = 0 ms
  2. D = 20 ms
  3. D = 35 ms
  4. D = 50 ms
  5. D = 100 ms (A starts at B offset, i.e. pure alternation)
  6. No captor: B only, the classic RMR floor (Turgeon 2005 reported Pc 0.53-0.56)
  7. D = 0 with **on-frequency fillers**: 420.45-Hz tones at every slot midpoint (+100 ms), roved like A. They pull the captors into their own sequential A stream.
  8. D = 0 with **off-frequency fillers**: identical, but at 250.0 Hz (9 semitones below A, as in the Micheyl, Hunter & Oxenham 2010 off-frequency captors).

  The four D values 0, 35, 50 and 100 are shared with Micheyl. D = 71 is dropped because it was not credibly different from its neighbours. D = 20 samples the 10-40 ms zone where fusion is expected to collapse.
- **Yoking:** use 30 frozen seeds (15 steady, 15 gallop). A seed fixes the rhythm, the gallop phase, the distractor slots and all roves. Each seed is played once in every condition, spread across blocks with a Latin square. Within a seed, the B waveform is therefore sample-identical in conditions 1 to 6, and the A waveform differs only by a rigid time shift. Add a unit test that hashes the B buffer.
- **Clean catches:** 12 trials with the target rhythm only (no distractors, no A), drawn from fresh seeds. They are used for exclusion only.

### Task and dependent variable

- One interval per trial, two alternatives: "Was the hidden beat STEADY (←) or GALLOPING (→)?" Feedback on every trial (0.4 s), at least 1.5 s of silence after it.
- The instructions never mention streams or grouping, and they are the same in every condition (intention shifts organisation: Billig, Davis & Carlyon 2018).
- **Dependent variable:** d′ = z(H) − z(F), with "gallop" as the signal and a log-linear correction. This puts it on the same axis as Micheyl's d′. Also report Pc.

### Built-in controls

- **No pedestal:** rhythm identity is judged from intervals of 400, 600 and 800 ms, and none of them depends on D.
- **No single-stream route:** an A-only or B-only strategy has to give the same score at every D, because B is identical and A is only shifted. Such a strategy can raise the floor but cannot create a slope. The floor is measured directly (condition 6, and D = 100).
- **Grouping, not mere coincidence detection:** in conditions 7 and 8 the D = 0 dyads are physically identical; only the sequential context changes. Grouping predicts Pc(on-frequency) < Pc(off-frequency) ≈ Pc(D = 0). A loudness, masking or coincidence account predicts no difference. This follows the logic of Bregman & Pinker (1978) and Micheyl, Hunter & Oxenham (2010).
- **Overlap versus asynchrony:** temporal overlap falls linearly with D (100 − D ms). Grouping predicts a steep fall by about 20-35 ms. Report the shape index (Pc0 − Pc35)/(Pc0 − Pc100); an energy or overlap account predicts about 0.35.
- **Carry-over:** the previous trial's condition is a covariate (Snyder et al. 2008).
- **Learning and memorisation:** block is a covariate, and a half-session × D interaction is pre-registered. If listeners memorise the frozen seeds, the floor rises; it cannot reverse the slope.
- **Exclusion:** pre-registered on training and clean catches (fewer than 10 of 12 correct). It is **never** based on Pc at D = 0, because selecting on D = 0 would inflate the D = 0 versus D = 100 difference.

### Session plan, about 48 minutes

| Minutes | Step |
|---|---|
| 0-4 | Consent; questionnaire (years of musical and percussion training); headphone and fixed-volume check (tsync calibration and device-match guard) |
| 4-11 | Familiarisation: 4 steady and 4 gallop, clean and labelled. Then 12 clean identification trials with feedback (criterion 10/12, at most 2 rounds). Then 12 trials at D = 0 plus 4 no-captor trials ("this one is hard, guess"). Then 8 mixed practice trials. |
| 11-43 | Main: 6 blocks × 42 trials. Each block has 8 conditions × 5 trials plus 2 clean catches, randomised. About 7.0 s per trial, so about 4.9 min per block, with 30-s breaks. Total 240 test trials (30 per condition) plus 12 catches. |
| 43-46 | Subjective block, Micheyl Exp. 1 protocol: 12 A + 12 B tones at 5 Hz (2.4 s), D = 0, 35 or 100 at 15 semitones, 6 repetitions each, plus 4 catch trials (synchronous at 1 semitone, where one stream is expected). Report "one stream or two" at the end of each sequence; listeners are told the percept may change. |
| 46-48 | Debrief and strategy questions |

**N:** recruit 14-16 naive listeners and aim for at least 12 who complete. At about 55 minutes per slot, two listeners a day fit easily into the two-week window.

### Analysis

1. **Confirmatory:** fit Micheyl's hierarchical Bayesian probit (for direct comparability) and a frequentist GLMM:
   `correct ~ condition + block + prevCond + (1 + D_linear | listener)`.
   Planned one-sided contrasts, mirroring Micheyl's two credible contrasts with the opposite sign: d′(0) > d′(50) and d′(0) > d′(100), Holm-corrected. Also a monotone trend over D.
2. **Grouping specificity:** Pc(on-frequency fillers) < Pc(off-frequency fillers), one-sided.
3. **Per-listener mixture fit:** Pc(D) = P_floor + (P_max − P_floor)·[1 − Φ((D − D50)/σ)].
   - P_floor is fixed from that listener's no-captor trials, and P_max has a lapse bound.
   - Report D50, σ and π(D).
   - Small σ in individuals under a graded group curve is the evidence for a mixture.
4. **Subjective versus objective:** correlate each listener's P(one stream | D) with π(D), and their subjective boundary with their objective D50.
5. **ARO figure:** the trmr d′ falling with D, plotted beside Micheyl's published d′ rising from 1.48 to 2.43, on the same D axis with the same tones. Individual curves go in an inset.

### Expected effect and power

- **Predictions:**
  - Pc(D = 0) ≈ 0.85 (d′ ≈ 2.1).
  - Pc(D = 35) ≈ 0.68.
  - Pc(D = 100) ≈ floor ≈ 0.55-0.60.
  - Δd′ from 0 to 100 ms ≈ −1.5, against Micheyl's +0.95 (MAP +1.10, 95% CR [0.03, 2.05], n = 7).
- **Individual listeners:** with 30 trials per condition, the SE of Pc(0) − Pc(100) is about 0.11, so most individual curves can be read on their own.
- **Group:** in the pessimistic case (effect 0.12, SD 0.12, dz = 1.0), power is about 0.89 at n = 12 and about 0.97 at n = 16 (two-sided α = .05).
- **Capture contrast:** its size is untested. Treat it as a directional test; 30 trials per condition at n = 12 detects a Pc difference of about 0.12 or more.

## 4. Why this is better than tcoh and tsync

- **tcoh** judges D against D + δ, so a threshold that grows with D is ambiguous between a pedestal effect and grouping. The fact that it fits about 0.4·D may not be pure Weber: for high-leading isolated pairs, Wojtczak et al. (2012) found thresholds flat over 0-40 ms. But the ambiguity cannot be removed without a separate control. trmr's judgement does not involve D at all.
- **tsync:**
  - Its re-synchronised ending moves each stream's last tone D/2 off its own rhythm: 50 ms at D = 100, which is 25% of the 200-ms IOI and about 3-4 times the 6-8% single-stream anisochrony threshold (Hirsh et al. 1990). So a strategy that uses one stream alone also predicts thresholds rising with D.
  - The D0_late control (a 50-ms displacement in the same direction for both tones) does not match that geometry.
  - A reset at the boundary is possible but not established after alternating context (Haywood & Roberts 2013). Either way, the design cannot answer the question.
  - It runs 2I staircases on a mixture of percepts and costs about 40 minutes for 6 noisy thresholds.
- **trmr:**
  - No pedestal; B is bit-identical across D; no anomalous ending.
  - Short single-interval trials sample the first percept.
  - It yields 8 fixed-level d′ values plus an organisation curve π(D) in about 32 minutes of testing.
  - It is the same stimulus family and the same D axis as Micheyl, with the opposite predicted sign.
- **tpair:** whole-pair jitter needs room in the timing, which rules out 5 Hz. trmr needs no jitter.

## 5. Runner-up and when to prefer it

**Runner-up: PTDD (percept-tracked deviant detection).** These are 20-s Micheyl-type sequences with a continuous one/two-stream report plus fixed-size B-displacement deviants (Billig & Carlyon 2016 Exp. 2: 12 listeners, 48 sequences, about 30 min). Prefer it:
- when bistability becomes the headline question, for the JASA second experiment or a second visit;
- for an EEG session, where percept-locked responses are possible (Gutschalk et al. 2005; Billig et al. 2018);
- if the trmr go/no-go fails at both 15 and 9 semitones.

Read its inference from the comparison of integrated and segregated stretches at a fixed D, not from the raw d′(D) curve, which still contains the step from a synchronous to an asynchronous reference. Jitter the pair-to-pair interval so that a B displacement is not also a cue within the B stream, which segregation would help (Micheyl & Oxenham 2010).

**Cheap salvage for existing data (1 day of code):** run tcoh again in the same listener with an isolated-pair, no-context pedestal control at D = 0, 38 and 75 ms. If thresholds after context exceed the isolated-pair thresholds, tcoh becomes a supplementary figure.

**Later, for JASA:** run Micheyl's increment task (D = 0, 35, 100; about 15 min) in the same trmr listeners. That gives crossing functions within each listener, following the matched-task template of Micheyl & Oxenham (2010).

## 6. Risks and a 2-3 day pilot

**Risks**

1. **The D = 0 anchor may be too weak.** Turgeon's near-ceiling 0.99-1.00 used interleaved odd and even harmonics in one spectral region. Pure tones 15 semitones apart fuse less (Micheyl Exp. 1: about 76% "two streams" when synchronous). Fix: lower p first. If D = 0 still fails, move to 9 semitones (A = 594.6 Hz), which breaks exact comparability with Micheyl.
2. **The floor may be too high** (rhythm audible through the distractors). Raise p toward 0.7, never move distractors off the grid, and check the A-only and B-only ideal observers.
3. **The fall may be complete by 20 ms.** That is a valid result in itself. If the pilot's D50 is under 15 ms, add D = 10 and drop the off-frequency filler condition.
4. **The fall may extend beyond 100 ms**, since humans fuse partly overlapping tones (Neilans & Dent 2015). D = 100 (no overlap) is still the anchor, and the curve shape is analysed.
5. **Feedback may teach a flat strategy, such as lag-matching**, which raises the floor. Pre-registered block × D test. If drift appears, add test blocks without feedback.
6. **Fillers may act through distraction or similarity rather than capture.** The off-frequency filler condition controls for this, although off-frequency captors were not perfectly neutral in Micheyl, Hunter & Oxenham 2010 Exp. 1.
7. **Uncalibrated MacBook output.** Use nominal levels with fixed volume, and tsync's calibrate command and device guard.

**Pilot**

- **Day 1 (build and simulate):**
  - Generator on top of tsync's `stimulus.py` (envelope and render), `audio.py`, `levels.py`, `app.py` (two keys), `logbook.py`, `sim.py` and `verify.py`.
  - Tests: the B-buffer hash is identical across D; no A onset falls within 100 ms of a non-partner B onset.
  - Ideal-observer simulation on 10,000 trials: (a) B-only template matching, (b) A-only gap pattern, (c) a cross-stream "B without an A partner" observer.
  - Choose p so that (a) and (b) are at most about 0.75. (c) should be near 1.0 at every D, which shows the information is present at all D, so any fall is perceptual.
  - Listen through every condition.
- **Day 2 (listeners):** 3 listeners, at least 1 naive, about 25 min each. Conditions D = 0, 35, 100, no captor, and on- and off-frequency fillers, 24 trials each.
  - Go/no-go: group Pc(D = 0) ≥ 0.80; Pc(no captor) ≤ 0.65; Pc(0) − Pc(100) ≥ 0.15.
  - Changes allowed, in this order: p, at most 2 iterations; then the 9-semitone fallback.
- **Day 3 (freeze):** confirm the sign of D against Micheyl's Methods. Run a power simulation with the pilot's P_max, P_floor and between-listener SD. Freeze the config hash and pre-register the contrasts, the exclusion rule and the mixture model.

## 7. Key references (verified)

1. Micheyl C, Hanson C, Demany L, Shamma S, Oxenham AJ (2013). Auditory stream segregation for alternating and synchronous tones. *JEP:HPP* 39:1568-1580. doi:10.1037/a0032241
2. Turgeon M, Bregman AS, Ahad PA (2002). Rhythmic masking release: contribution of cues for perceptual organization to the cross-spectral fusion of concurrent narrow-band noises. *JASA* 111:1819-1831. doi:10.1121/1.1453450
3. Turgeon M, Bregman AS, Roberts B (2005). Rhythmic masking release: effects of asynchrony, temporal overlap, harmonic relations, and source separation on cross-spectral grouping. *JEP:HPP* 31:939-953. doi:10.1037/0096-1523.31.5.939
4. Christiansen SK, Oxenham AJ (2014). Assessing the effects of temporal coherence on auditory stream formation through comodulation masking release. *JASA* 135:3520-3529. doi:10.1121/1.4872300
5. Micheyl C, Hunter C, Oxenham AJ (2010). Auditory stream segregation and the perception of across-frequency synchrony. *JEP:HPP* 36:1029-1039. doi:10.1037/a0017601
6. Micheyl C, Oxenham AJ (2010). Objective and subjective psychophysical measures of auditory stream integration and segregation. *JARO* 11:709-724. doi:10.1007/s10162-010-0227-2
7. Elhilali M, Ma L, Micheyl C, Oxenham AJ, Shamma SA (2009). Temporal coherence in the perceptual organization and cortical representation of auditory scenes. *Neuron* 61:317-329. doi:10.1016/j.neuron.2008.12.005
8. Shamma SA, Elhilali M, Micheyl C (2011). Temporal coherence and attention in auditory scene analysis. *Trends Neurosci* 34:114-123. doi:10.1016/j.tins.2010.11.002
9. Pressnitzer D, Hupé JM (2006). Temporal dynamics of auditory and visual bistability reveal common principles of perceptual organization. *Curr Biol* 16:1351-1357. doi:10.1016/j.cub.2006.05.054
10. Hupé JM, Pressnitzer D (2012). The initial phase of auditory and visual scene analysis. *Phil Trans R Soc B* 367:942-953. doi:10.1098/rstb.2011.0368
11. Billig AJ, Carlyon RP (2016). Automaticity and primacy of auditory streaming: concurrent subjective and objective measures. *JEP:HPP* 42:339-353. doi:10.1037/xhp0000146
12. Rankin J, Sussman E, Rinzel J (2015). Neuromechanistic model of auditory bistability. *PLoS Comput Biol* 11:e1004555. doi:10.1371/journal.pcbi.1004555
13. Denham SL, Winkler I (2006). The role of predictive models in the formation of auditory streams. *J Physiol Paris* 100:154-170. doi:10.1016/j.jphysparis.2006.09.012
14. Moore BCJ, Gockel HE (2012). Properties of auditory stream formation. *Phil Trans R Soc B* 367:919-931. doi:10.1098/rstb.2011.0355
15. Snyder JS, Carter OL, Lee SK, Hannon EE, Alain C (2008). Effects of context on auditory stream segregation. *JEP:HPP* 34:1007-1016. doi:10.1037/0096-1523.34.4.1007
16. Haywood NR, Roberts B (2013). Build-up of auditory stream segregation induced by tone sequences of constant or alternating frequency and the resetting effects of single deviants. *JEP:HPP* 39:1652-1666. doi:10.1037/a0032562
17. Roberts B, Glasberg BR, Moore BCJ (2008). Effects of the build-up and resetting of auditory stream segregation on temporal discrimination. *JEP:HPP* 34:992-1006. doi:10.1037/0096-1523.34.4.992
18. Bregman AS, Pinker S (1978). Auditory streaming and the building of timbre. *Can J Psychol* 32:19-31. doi:10.1037/h0081664
19. Wojtczak M, Beim JA, Micheyl C, Oxenham AJ (2012). Perception of across-frequency asynchrony and the role of cochlear delays. *JASA* 131:363-377. doi:10.1121/1.3665995
20. Wojtczak M, Mehta AH, Oxenham AJ (2017). Rhythm judgments reveal a frequency asymmetry in the perception and neural coding of sound synchrony. *PNAS* 114:1201-1206. doi:10.1073/pnas.1615669114
21. Hirsh IJ, Monahan CB, Grant KW, Singh PG (1990). Studies in auditory timing: 1. Simple patterns. *Percept Psychophys* 47:215-226. doi:10.3758/bf03204997
22. Zera J, Green DM (1993). Detecting temporal asynchrony with asynchronous standards. *JASA* 93:1571-1579. doi:10.1121/1.406816
23. Farkas D, Denham SL, Bendixen A, Winkler I (2016). Assessing the validity of subjective reports in the auditory streaming paradigm. *JASA* 139:1762. doi:10.1121/1.4945720
24. Billig AJ, Davis MH, Carlyon RP (2018). Neural decoding of bistable sounds reveals an effect of intention on perceptual organization. *J Neurosci* 38:2844-2853. doi:10.1523/JNEUROSCI.3022-17.2018
25. Darwin CJ, Ciocca V (1992). Grouping in pitch perception: effects of onset asynchrony and ear of presentation of a mistuned component. *JASA* 91:3381-3390. doi:10.1121/1.402828
26. Neilans EG, Dent ML (2015). Temporal coherence for pure tones in budgerigars and humans. *J Comp Psychol* 129:52-61. doi:10.1037/a0038368
27. Gutschalk A, Micheyl C, Melcher JR, Rupp A, Scherg M, Oxenham AJ (2005). Neuromagnetic correlates of streaming in human auditory cortex. *J Neurosci* 25:5382-5388. doi:10.1523/JNEUROSCI.0347-05.2005

**Do not cite:**
- The Turgeon 2002 burst and rhythm parameters (48 ms; 384/768 ms). They are unverified for that paper.
- The intermediate Pc values from Turgeon 2005, which appear only in its figures.
- "Thompson et al. 2011 Exp. 3". The 96-trial, about 25-minute design is their Exp. 2.

When citing Micheyl, Hunter & Oxenham (2010), state the ideal-observer result in the right direction: order thresholds were 1.84 times larger than detection thresholds, although they were predicted to be about 2.91 times smaller.