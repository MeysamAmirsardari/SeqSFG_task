# tcmr: CMR after precursors sheared across frequency

Does an ordered temporal shear of the precursors break across-frequency grouping less than an
unordered one? The measure is comodulation masking release (CMR): a 1-kHz tone is easier to hear
in a 20-Hz noise band when four remote bands share its envelope, but only if all five bands are
heard as one stream (Christiansen & Oxenham 2014, JASA 135:3520, doi 10.1121/1.4872300).

**Stimulus (as Christiansen & Oxenham 2014, Exp. 1).**
- Five 20-Hz noise bands at 0.25, 0.5, 1, 2 and 4 kHz, 60 dB each on the nominal scale (their measured levels are reported; see `audiometry/README.md`).
- 187.5-ms bursts with 20-ms ramps, one every 250 ms.
- Four precursor bursts, then a final burst that is always synchronous across bands. In one of
  three intervals it carries the 1-kHz tone.

**Shear.** Each band's precursors are offset by k·ΔT, bands listed low to high:

| arrangement | offsets (×ΔT) | |
|---|---|---|
| sync | 0 0 0 0 0 | baseline |
| ascending | −2 −1 0 +1 +2 | ordered sweep, low bands first |
| zigzag | +2 −1 0 +1 −2 | same 1-kHz-to-flank lag at each frequency distance, not monotonic |

The 1-kHz band never moves. Each band's last gap before the final burst absorbs its offset; the
smallest gap at ΔT = 20 ms is 22.5 ms.

**Conditions (8 interleaved tracks × 36 trials).**

| track | envelopes | arrangement | ΔT | tracks |
|---|---|---|---|---|
| baseline | comodulated | sync | 0 | 1 |
| CMR reference | random | sync | 0 | 1 |
| break-point step | comodulated | ascending | 10 ms | 1 |
| key contrast | comodulated | ascending | 20 ms | 2 |
| key contrast | comodulated | zigzag | 20 ms | 2 |
| CMR reference | random | ascending | 20 ms | 1 |

**Task.** Three intervals: which one had the beep? Answer with 1/2/3 or ←/↓/→, with feedback.
- **Tracks:** 2-down 1-up on tone level, steps 8 → 4 → 2 dB (Christiansen & Oxenham's rule),
  interleaved round by round.
- **Thresholds:** a maximum-likelihood fit to every trial, with one slope per listener; reversal
  means are reported too.
- **Ratings:** at the end, 10 precursor-only sequences rated 1 (one sound) to 5 (separate layers).

**Session.** About 36 minutes: questionnaire, two labelled examples, 8 practice trials, 288 main
trials with three breaks, then the ratings. `python -m tcmr design` prints the simulated length.

```bash
python -m tcmr verify              # comodulation, gaps, zigzag matching, headroom, reproducibility
python -m tcmr demo                # one-interval examples as WAV (data_tcmr/demo)
python -m tcmr calibrate           # sine at each band centre, sound level meter + coupler
python -m tcmr run --code P01      # a session (full screen; --windowed, --uncalibrated for pilots)
python -m tcmr run --code P01 --resume data/P01/tcmr_session_01
python -m tcmr analyze data/P01/tcmr_session_01
```

**Output.** `analyze` reports:
- thresholds (dB on the nominal scale, at 70.7%);
- CMR for sync, ascending 10 and 20, and zigzag 20;
- the contrasts in dB, where positive means harder:
  - **shear cost:** ascending vs sync;
  - **order cost:** zigzag vs ascending at 20 ms;
  - **random-baseline shift:** whether the shear moves the random-envelope threshold;
- the mean ratings, plus flags for CMR at sync below 3 dB, volume changes and audio dropouts.

Every trial is in `trials.csv` and `trials.jsonl`, with seed, buffer SHA-256, onset error,
volume and keys.

**Reused from tsync:** the audio engine (one stream, sample-scheduled onsets, dropout detection),
the volume guard, calibration storage, the questionnaire and the screen style.

## Plan C: the frozen-pattern pilot (2026-11-14)

Preset `tcmr_frozen` (design notes: `literature/tcmr_redesign_2026-11-14.md`, section 13; figure:
`literature/tcmr_frozen_paradigm.png`).

- **Bands and bursts.** The same five 20-Hz bands at 60 dB (nominal). Bursts are 50 ms with 10-ms ramps, one per band every 200 ms (5 Hz). Each interval has 5 precursors, then a synchronous final burst that carries the 1-kHz tone. Every burst is new noise at its natural level.
- **The pattern** (`frozen`). The offsets are a staircase (-2, -1, 0, +1, +2) x step with the bands shuffled over it: the tone band keeps 0 and the other four take one of 24 orders.
  - The order is kept for all precursors of an interval and changes between the three intervals of a trial, so the amount heard never depends on where the tone is.
  - Each track uses each of the 24 orders 4 times.
- **The control** (`reordered`). The same offsets, but a different order in every repetition, with at least 25 ms of silence between two bursts of one band.
- **One window for every trial.** Each interval opens 80 ms before the tone band's first burst, so the tone band sits at the same samples in every condition and every trial has the same length (3 x 1.13 s + 2 x 0.4 s).
- **Tracks.** Comodulated and random at sync and at frozen 10, 20 and 40 ms, comodulated and random re-ordered at 40 ms, and band alone: 11 interleaved tracks x 32 trials = 352 trials.
- **Length.** About 40 minutes of trials and 50 minutes in all.
- **Analysis.** `analyze` reports CMR at every step, the drop from sync, and frozen minus re-ordered CMR at 40 ms. The orders of every trial are in the `orders` column of `trials.csv`.

**Seeds are checked before anyone listens.** `check` draws every trial a session with that seed will play and checks each one:
- what is drawn: the examples, the practice, all 352 main trials and their first repeat after a pause, and the ratings;
- what is checked: each band's timing to the sample, the silences, the orders and their balance, comodulation, the tone's place and level, headroom at the lowest and highest tone level (so no trial is ever redrawn), and reproducibility.

`run --seed` refuses a seed whose check is missing or failed.

```bash
python -m tcmr check --config tcmr/configs/tcmr_frozen.json --seed 20261114
python -m tcmr run   --config tcmr/configs/tcmr_frozen.json --code P00 --seed 20261114 --uncalibrated --volume 56
```

## Experiment 1: the ARO study and the JASA paper (`tcmr_exp1`, 2026-11-15)

Preset `tcmr_exp1` (hash `a5099394bbce529d`). It is plan C with three changes, and is frozen: change it only through a new preset.
- The hash was `25ac4a4d62854cc1` until the practice gate was added on 2026-11-15, before any listener. The sounds are the same, sample for sample; the DEMO run of that day is stored under the old hash.

| | `tcmr_frozen` (P00 pilot) | `tcmr_exp1` |
|---|---|---|
| steps (frozen pattern) | 0, 10, 20, 40 ms | **0, 5, 10, 20, 40 ms** |
| re-ordered control at 40 ms | yes | **no** (moves to Experiment 2, the repetitions study) |
| rating block | 6 sequences | **none** (as a listener, P00 found the ratings meaningless) |
| practice gate | none | **at least 6 of the 8 practice trials right**, or the session stops there |
| everything else | | the same: 5 x 20-Hz bands at 60 dB (nominal), 50-ms bursts every 200 ms, 5 precursors, synchronous final burst with the 1-kHz tone, natural burst level, 3 intervals, 2-down 1-up 8/4/2 dB, 32 trials per track |

- **Why the 5-ms step:** the question is how far coherence effects persist. 0 to 10 ms is where P00's release fell and where models with different integration windows disagree. Neighbouring bursts overlap by 100, 90, 80, 60 and 20% at the five steps.
- **Tracks:** comodulated and random at each step, plus the tone's band alone: 11 interleaved tracks x 32 trials = 352 trials. The session lasts about 41 minutes (simulated); P00 took 38.7 minutes for the same length.

**Practice gate.** The session stops after the 8 practice trials if fewer than 6 were right (`practice_min_correct`).
- The screen only thanks the listener ("The practice is over"). The session is stored with status `practice_failed`, and `run` prints the count when it ends.
- A session that stopped at the gate, or before the practice was finished, cannot be continued with `--resume`. The gate has to be passed in the session itself.
- Chance of passing by guessing: 2%. A listener like P00 (random-band threshold near 67 dB) passes more than 99 times in 100.
- The four random-band practice trials go down to 72 dB, so the gate is not only a test of understanding. With a random-band threshold of 74 dB a listener passes 96% of the time, with 76 dB 87%, with 78 dB 67%. If a pilot's random threshold is above about 74 dB, raise the practice levels in a new preset before recording.

**The top of the level range.** A track starts at 75 dB and cannot go above 85 dB.
- When the rule asks for more than 85 dB the track stays at 85 dB and the call is counted. A track with 3 or more such calls is reported as **at ceiling**. In simulated tracks that happens in under 5% of tracks with a true threshold of 80 dB or lower, in about half at 84 dB and in about 90% at 86 dB or higher.
- The fit stays within about 2 dB up to a true threshold of 86 dB and loses precision above it. So a fitted threshold above 85 dB is **entered as 85 dB, a lower bound**, with the fitted value kept beside it; CMR and contrasts that use it are bounds.
- `analyze` flags both, and the session figure marks the track.

**Seeds.** 20261201 to 20261226 have passed `check` under the present hash (26 seeds: 2 pilots and 24 listeners).
- `run --seed next` takes the lowest checked seed that no session of this preset has used yet.
- A seed another listener already used is refused. A session stopped at the gate has used its seed too.
- A calibrated run needs calibrated checks: after calibrating, check the seeds again with `--calibrated`, which takes about 2 minutes for all 26.
- A calibrated check belongs to the calibration it was made with. After calibrating again, check again; `run` refuses a check made with other levels.

**Since 2026-11-17 the listeners are recorded under `audiometry/README.md` instead:** no ear simulator yet, so every session runs `--uncalibrated` at the rig's fixed volume (`audiometry setup`), the chain is checked on a jig every session day (`audiometry daily`), each listener is screened first (`audiometry screen`), and an IEC 60318-1 ear simulator measures the same chain later, so every session's levels can be reported in dB SPL (`audiometry report`; thresholds relative to the 1-kHz band need no calibration). `calibrate` and calibrated checks are not used for Exp 1; the rule below is kept for reference.

**Calibration (before the first listener whose data will count).**
- Use one pair of wired headphones for every session; the device is always called "External Headphones".
- Put the headphones on an ear coupler with a sound level meter and run `calibrate` at the system volume every session will use. Meter on C or Z weighting, slow. Not A: it reads 8.6 dB low at 250 Hz.
- `calibrate` plays the 1-kHz tone first, then the other four bands, each for a minute at amplitude 0.05. Type the reading; Enter plays the tone again; a mistyped reading is asked again.
- At 1 kHz the tone should read at least 68 dB SPL. Then 60 dB SPL is a sine peak of at most 0.02, the headroom the seeds were checked at. Aim for 72 dB or more: that also leaves room to raise the 85-dB maximum to 90 dB if the pilots need it. Below about 63 dB `calibrate` stops and saves nothing.
- Nothing is saved if the system volume changed during the measurement. Every calibration is also kept as a dated file beside `tcmr_latest.json`.

```bash
python -m tcmr calibrate --config tcmr/configs/tcmr_exp1.json
python -m tcmr check     --config tcmr/configs/tcmr_exp1.json --seed {20261201..20261226} --calibrated
python -m tcmr run       --config tcmr/configs/tcmr_exp1.json --code L01 --seed next --volume <calibrated volume>
python -m tcmr analyze   data/L01/tcmr_session_01
python figures/tcmr/plot_listener.py data/L01/tcmr_session_01
```

For an uncalibrated pilot, add `--uncalibrated` to `run`; the seeds are already checked for that.

**What code ran.** Every new session stores a hash of the tcmr code and presets (`source_tcmr`) and whether the tcmr folder had uncommitted changes (`tcmr_dirty`, with the diff in `code.patch`). `run` warns when it has: commit before a listener whose data will count.

**Continuing a session** (`--resume`) is refused unless the levels are what the session started with: calibrated or not, the same calibration, the same system volume. The trial that was interrupted is drawn anew, as after a pause.

**Planned analysis (a draft: fix it before the first listener).**
- **Primary outcome:** per listener, CMR (random minus comodulated threshold) at synchrony minus CMR at the 40-ms step. Test it across listeners, two-sided, alpha 0.05.
- **Secondary outcomes:**
  - CMR at each step, which gives the shape of the curve;
  - release relative to the tone's band alone, for the comodulated and the random thresholds separately (benefit and interference).
- **Pilots:** the first two seeds. Go if both show at least 3 dB of CMR at synchrony; if only one does, run a third. Report the pilots separately.
  - One 32-trial track gives a threshold to about ±1.5 dB (sd), so CMR from one session is uncertain by about ±2 dB. P00's own two runs gave 10.2 and 2.2 dB at synchrony. Read the 3-dB rule with that in mind.
  - No track of a pilot may be at ceiling. If one is, raise the level range in a new preset before recording.
- **Thresholds above the range:** a fitted threshold above 85 dB is entered as 85 dB (see above). Run the primary test twice, with these values and without the listeners who have one in the four tracks of the primary outcome, and report both.
- **Exclusions:** an incomplete session, fewer than 6 of 8 practice trials correct (the session stops itself), or audio dropouts in valid trials. Never exclude on CMR measured in the same tracks. P00 is reported separately.
- **Eligibility is measured, not declared.** The only eligibility criterion is the hearing screen (`audiometry/README.md`); the only performance gate is the practice block above. There is no exclusion on any diagnosis, condition or demographic characteristic, self-reported or documented. In particular, a self-reported neurodevelopmental or auditory-processing condition (ASD, ADHD, APD, dyslexia) is **never** grounds for exclusion. Two reasons: the measure is a level difference at the ear through one fixed chain, which the audiometric screen already bounds; and no principled line separates such a condition from musicianship, bilingualism or anxiety, each of which has its own auditory-attention literature and none of which is screened here.
- **Self-reported conditions are recorded and reported.** Every listener is asked the same intake question, in the same words, before the screen, with "prefer not to say" available; nothing is recorded from an unsolicited disclosure that is not asked of everyone. The answers are stored with the consent records, not in `data/`, and are reported as participant characteristics. Where a listener's self-report is relevant to a primary outcome, run the primary test twice, with and without them, and report both — as for a threshold above the range. Fix this before seeing any listener's d' or CMR.
