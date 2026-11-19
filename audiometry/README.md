# audiometry: the rig, its level checks and the hearing screen

This is the protocol for recording listeners before an ear simulator is available (decided 2026-11-17, revised after three independent reviews).

- **Now:** the sounds play uncalibrated through one fixed playback chain. A jig and the Voltcraft SL-100 show every day that the chain has not changed.
- **Later:** an IEC 60318-1 ear simulator measures the same chain once, and the levels of every session are then reported in dB SPL.

CMR, its drop and tinc's d′ are differences between levels played through the same chain, so they do not depend on the absolute level. The absolute levels are measured and reported, not set.

- **Bands:** the five bands have equal digital rms. Their levels at the ear follow the headphone's response, so the flanking bands will differ from the 1-kHz band by a few dB.
- **In the paper:**
  - Report each band's measured long-term level.
  - Report thresholds relative to the 1-kHz band; these need no calibration at all.
  - Never write "presented at 60 dB SPL".

## The rule: one chain, never changed

- **Hardware:** this laptop and one Sennheiser HD 280 Pro.
  - Write a label on the pair, for example `HD280Pro-1`.
  - The 3.5-mm plug goes straight into the headphone jack, with no adapter.
  - Plug it in before the daily check and leave it in until the last session of the day. The Mac re-detects the headphones at every plug-in, so after any re-plug the daily check is run again.
- **Playback:** every session (tcmr, tinc, the screen) runs at the rig's system volume with `--uncalibrated`. Keep this for every Exp 1 listener, even after the ear-simulator day.
- **Software:** no macOS update and no audio, EQ or "enhancement" apps for any Exp 1 listener, before or after the ear-simulator day: turn automatic macOS updates off. If an update is ever forced, run `python -m audiometry levels --frequencies all` before the next listener and compare it with the baseline.
- **System settings** (yours to set once):
  - Sound effects play through the laptop's own speakers.
  - Bluetooth off.
  - Do Not Disturb on during sessions.
- **Strongly recommended purchase:** a sound calibrator for the SL-100: 94 dB at 1 kHz, IEC 60942, fitting a 1/2-inch microphone.
  - The SL-100's manual gives it "Class 3" (±2%), 31.5 Hz to 8 kHz. Without a calibrator, a drift of the meter cannot be told apart from a drift of the chain.
  - The daily check logs its reading. Never touch the SL-100's trim pot.

## Once, before the first listener

1. **Build the jig.**
   - **Block:** use a rigid block 145 mm wide (the ISO 389-8 cup separation). Hold the meter's microphone flush in one face: remove the foam ball and seal the gap with fresh putty.
   - **Fit:** the headphones clamp the block with their own headband.
     - Mark the headband slider position.
     - Mark each cup's outline and its rotation on the block.
   - **Why:** the force is then the headphone's own, the same every time.
2. **Choose the volume.** Repeat with different N until the 1-kHz check tone reads about 68 dB on the jig:

   ```bash
   python -m audiometry levels --trial --volume N --cups left --reseats 1
   ```

   The check tone is 7.96 dB above the program's nominal 60 dB, so the target is 67.96 exactly. The 1-kHz band then sits near 60 dB on the jig. The ear simulator gives the true value later.
3. **Fix the rig.** Running it again with the same values changes nothing; a different volume or pair needs `--new-rig`.

   ```bash
   python -m audiometry setup --headphones HD280Pro-1 --volume N
   ```

4. **Booth noise.** BSA 2018 4.2 sets 35 dB(A). The SL-100 reads only broadband, down to 30 dB, so report "<30 dB(A)" if it reads below its range. The one-third-octave check against BSA Appendix C comes on the ear-simulator day.

   ```bash
   python -m audiometry ambient
   ```

5. **Baseline.** This measures both cups at every frequency used (0.25, 0.42, 0.5, 1, 2, 4, 8 kHz), with 3 seatings each and C weighting (corrected for C's −0.8 dB at 4 kHz and −3.0 dB at 8 kHz).
   - It is the baseline the daily checks are held to.
   - It is also the screen's provisional scale.

   ```bash
   python -m audiometry levels --frequencies all
   ```

## Every session day

Before the first listener, run the daily check:

```bash
python -m audiometry daily
```

- **Stage A checks** (BSA 2018 11.1): a checklist, including that this is the labelled pair, the cable, the key and the room.
- **Logged values:** the calibrator reading (if you have one), the temperature and the humidity.
- **The jig at 1 kHz, both cups,** compared with the baseline:
  - up to 0.5 dB: fine;
  - 0.5–1 dB: warning;
  - over 1 dB: fail.
- **Listening checks you do yourself:**
  - the screen frequencies at 15 dB HL (just audible, no hum or hiss);
  - the same at 60 dB HL (clean);
  - a 1-kHz tone while you move the cable (no crackle).

If the check fails, no listeners that day: the screen will not start unless the last daily check that day passed. A failure that cannot be undone starts a new chain, and that chain needs its own ear-simulator measurement. Readings from failed checks are logged as failed and never become a baseline.

For each listener:

1. **Consent.** It must cover the screen and permission to contact them later for a hearing test on a calibrated audiometer.
2. **The screen.** It first asks the BSA 5.1 questions: loud noise in the last 24 hours, tinnitus, a cold or ear pain, the ear examination (or "not done"), and clearing glasses and hair from under the cups. `--better-ear` is the listener's own account. It takes about 10 minutes (up to about 15 when an ear is repeated).

   ```bash
   python -m audiometry screen --code S01 --better-ear right
   ```

3. **The tasks**, at the rig's volume:

   ```bash
   python -m tcmr run --config tcmr/configs/tcmr_exp1.json --code S01 --seed next --uncalibrated --volume N
   ```

Use new codes (`S01`, `S02`, ...): `DEMO`, `L01`, `P00`, `P01`, `PILOT01` and `PILOT03` already exist in `data/`, and the screen asks before adding to an existing folder.

Once a week:
- Measure the jig at every frequency:

  ```bash
  python -m audiometry levels --frequencies all
  ```

- Screen yourself as `SELF` (BSA check 12, "an audiogram on a known subject"). `report` flags any change of 10 dB or more.

## Ear-simulator day (borrowed IEC 60318-1 ear simulator and class 1 meter)

1. **Set up:**
   - **Meter:** check it with its calibrator, and record the serials and certificate dates of the simulator, adapter, meter and calibrator.
   - **Room:** 21 to 25 °C. Record the temperature, humidity and static pressure.
   - **Adapter:** use the circumaural flat-plate adapter of IEC 60318-2:1998 Figure 1, as in ISO 389-8, unless the HD 280 Pro RETSPL paper (below) specifies otherwise. Use what that paper used.
   - **Force:** measure the HD 280 Pro's own headband force at the ISO 389-8 geometry, with the cups 145 mm apart and the headband 130 mm high (a luggage scale on one cup is enough). Apply that force on the simulator, not the HDA 200's 10 N.
2. **The jig first**, to tie this day to every earlier session:

   ```bash
   python -m audiometry levels --frequencies all
   ```

3. **The check tones**, with at least 5 refits per cup and Z weighting:

   ```bash
   python -m audiometry levels --coupler ear-simulator --frequencies all --meter "<simulator, adapter, meter, calibrator, serials>"
   ```

4. **Linearity** (BSA Stage B, check 17): run 1 kHz at 85, 75, 65, 55, 45, 35 and 25 nominal.

   ```bash
   python -m audiometry levels --coupler ear-simulator --frequencies 1000 --nominal 85
   ```

5. **Harmonic distortion** (Stage B, check 18): measure with the lab's analyser while a tone plays at 85 nominal (as in step 4) and at the screen's highest level. The bands are an octave apart, so second-order products of the flanking bands land exactly on 1 kHz (2 × 500, 2000 − 1000). Show they are at least 40 dB down.
6. **The real noise bands.** Read the meter's Leq over at least 30 s for each band, and for the sum of the five:

   ```bash
   python -m audiometry levels --coupler ear-simulator --signal band --frequencies 250 500 1000 2000 4000
   python -m audiometry levels --coupler ear-simulator --signal bands-sum
   ```

7. **Booth noise** in one-third octaves with the class 1 meter, against BSA 2018 Appendix C Table 1. Add 20 dB to its limits for a 20-dB HL criterion.
8. **The jig again, then the calibrator check again.**
9. **The conversion:**

   ```bash
   python -m audiometry report --csv data/rig/screens.csv
   ```

Measure the chain on the ear simulator again at the end of Exp 1.

How the conversion works, for each cup and frequency:

- offset = the ear simulator's unweighted reading of the check tone − 67.96
- level in dB SPL = nominal level + offset
- hearing level = nominal level + offset − RETSPL

`report` prints:
- tcmr thresholds relative to the 1-kHz band (the primary measure);
- tcmr thresholds in dB SPL per cup, and their mean;
- the band levels;
- the linearity and the measured noise-band levels against the tone offsets;
- every screen re-scored.

Each screen and each tcmr session is scored with the chain (pair and volume) it was recorded on, never with a later one, so a new chain after `setup --new-rig` needs its own ear-simulator measurement.

Suggested wording for the paper:

> Bands had equal digital rms. On an IEC 60318-1 ear simulator (adapter, force), their long-term levels were a to e dB SPL (cups within x dB). Thresholds are given relative to the 1-kHz band. The 1-kHz output, checked daily on a jig, varied by at most y dB.

## Reference thresholds for the HD 280 Pro

- **The values exist.** Smull, Madsen and Margolis (2019, Ear Hear 40:177-183, doi 10.1097/AUD.0000000000000585) determined RETSPLs for the HD 280 Pro.
  - Get the paper and enter its table, and the coupler, adapter and force it states, in `standards.RETSPL_HD280PRO`. `report` then scores with them.
  - Measure on the ear-simulator day with that same coupler.
- **Until then:**
  - The screen presents on, and `report` scores with, the HDA 200 values of ISO 389-8:2004 Table 1.
  - Four of the paper's values, as transcribed by a third party (0.5–4 kHz), are 2.6 to 7.3 dB lower than the HDA 200's (4.2, 4.1, 2.6 and 7.3 dB at 0.5, 1, 2 and 4 kHz). So the HDA 200 scale makes hearing look that much better than it is: a lenient screen.
  - `report` prints this difference.

## The screen

| Item | Value | Source |
|---|---|---|
| Before the test | Questions on loud noise (24 h), tinnitus, cold, ear examination; cups cleared | BSA 2018 5.1 |
| Instructions | Read aloud; press and hold while the tone is heard, release when it stops; ignore tinnitus; each ear separately; then ask whether it is clear | BSA 6.1, ASHA 2005 |
| Order | Better ear first; 1, 2, 4, 8, 0.5, 0.25 kHz; then 1 kHz again in the first ear | BSA 6.4 |
| Familiarisation | 1 kHz at 40 dB HL; if missed, up in 10-dB steps; press and release must match the tone's onset and offset; twice wrong: re-instruct | BSA 6.6 |
| Search | Down 10 dB after a response, up 5 dB after a miss | BSA 6.7, ASHA 2005 |
| Threshold | Lowest level with responses on at least half of its ascending presentations, at least two responses (2 of 2, 3 or 4) | BSA 6.7. ASHA's minimum is 2 of 3; ISO 8253-1:1989 asks 3 of up to 5 ascents (2 of 3 in its shortened method) |
| Next frequency | Starts 30 dB above the adjacent threshold | BSA 6.7 |
| 1-kHz retest | Within 5 dB: keep the lower. More: re-instruct and repeat that ear once, compare the passes, and if it still differs keep the lower and flag it. Retest the second ear only after such a variation | BSA 6.4, ASHA 2005 |
| Tones | Continuous pure tones of 1 to 2 s, always played in full; not pulsed (BSA 6.8 advises against pulsed tones) | BSA 6.5, 6.8; ASHA |
| Intervals | Random, from the longer of the tones before and after to 3.5 s, and at least 1 s after the key is released | BSA 6.5, ASHA; the 1 s is ours |
| Ramps | 50-ms raised cosine; −20 to −1 dB in 29 ms, no overshoot | Ours (see below) |
| Range | −10 to 60 dB HL (presentation scale) | Ours |
| Response window | A press from 150 ms after onset to 1 s after offset | Ours; comparable to published automated audiometry |
| Anticipations | A press before the tone starts aborts the tone; a press in its first 150 ms voids it; either counts as a false response, and the next tone waits 2 s | Ours |
| Catch trials | Silent presentations (15%, never two in a row) | Ours, as in automated audiometry (e.g. AMTAS) |
| False responses | 3 since the last instructions: re-instruct. 2 or more within one threshold: re-instruct and repeat that frequency once. 2 or more again: that threshold is "unreliable" | Ours; ASHA 2005 on re-instruction |
| Criterion | ≤ 20 dB HL at every octave from 0.25 to 8 kHz in both ears; no response at 60 dB HL is a fail; an unreliable threshold makes the screen incomplete | Ours (the usual research criterion) |

- **Ear-change cues:** the window never changes while a tone plays.
- **Volume:** the system volume is read twice a second, and a tone is scored only after a reading taken after it ended. A changed volume pauses the screen and voids the tone.
- **Held key:** a key held for 10 s with no tone pauses the screen.
- **Saved files:** every tone and catch trial (with the presses and releases during it), every false response (with its press time), void, pause and re-instruction goes to `data/<code>/screen_NN/events.jsonl`. `screen.json` holds the thresholds as provisional hearing levels and as nominal levels, with each frequency's false responses.

### Who is included (decided before any data; the PI may change this before the first listener)

1. **Primary:** a calibrated audiometer screen at 20 dB HL, 0.25 to 8 kHz, at a recontact.
2. **Listeners who cannot return:** they are included only if their rig screen, re-scored with the ear-simulator offsets and the HD 280 Pro reference values, passes. The main results are then reported both with and without them.
3. **Never on CMR.**
4. **Agreement:** once 10 listeners have both screens, report how the rig screen agrees with the audiometer.

On the day, run every listener who reports normal hearing. If someone's thresholds are clearly worse than everyone else's, tell them it is not a diagnosis and suggest a clinic.

## What is not standard, said plainly

- **Scope.** BSA 2018 excludes screening and self-recording audiometry (3.2), and ASHA 2005 covers manual audiometry. This is an automated screen that follows their manual procedure. The automatic scoring, catch trials and false-response rules stand in for the tester's judgement.
- **Reference thresholds:** see above. The HD 280 Pro's own values must be entered from Smull et al. (2019).
- **Ear examination:** BSA 5.1 asks for an ear examination before testing. If no one qualified can do it, the screen records "not done".
- **Response key:** BSA 6.2 asks for an inaudible response. The space bar makes a faint click, which the closed cups attenuate.
- **Releases** are logged but are used only in the familiarisation check, not to score thresholds.
- **The jig** is not a standard coupler. Its readings are only a stability check and a provisional scale.
- **The SL-100** is a class 3 meter by its own manual. It is used only for those relative checks and the broadband booth noise, never for reported levels.
- **Tone switching.** IEC 60645-1 clause 8.6.3 (rise and fall times) could not be read. The ramps are measured instead (29 ms, in `tests/test_audiometry.py`); secondary sources give 20 to 50 ms.
- **No masking.**
  - **Why it isn't needed:** BSA 8.1 gives 40 to 80 dB interaural attenuation for circumaural earphones (assumed here, not measured for the HD 280 Pro). A response heard through the other ear therefore needs at least 30 dB HL, so an ear above 20 dB HL cannot pass by cross-hearing.
  - **Booth noise:** thresholds at or below 0 dB HL may be limited by the booth noise. The 20-dB criterion is not affected.

## Files

- `standards.py`: every number, with its clause.
- `rig.py`: the level scale, the C-weighting correction and the logs (`data/rig/levels.jsonl`, `daily.jsonl`, `ambient.jsonl`, `rig.json`).
- `tones.py`: the tones and the tcmr-scaled noise bands.
- `procedure.py`: the BSA search and test order, with no sound or screen.
- `screen_app.py`: the booth window.
- `levels.py`: setup, levels, daily, ambient.
- `report.py`: the conversion.
- Tests: `tests/test_audiometry.py`. They never play sound or change the volume. Four of them run the screen headless: an ideal listener, a random presser, a press before the onset, and a volume change.
