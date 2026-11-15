# tsync protocol: recording, exclusions, analysis

Fixed before the first listener. Change it only with a dated note at the end of this file.

## Sample

- **Listeners.** 12 complete listeners, adults with no known hearing problem, naive to the
  hypothesis.
- **Order.** The 6-row Williams square is used twice. A listener's row is (participant number −
  1) mod 6, so P01 and P07 share an order, P02 and P08, and so on.
- **Replacements.** An excluded listener is replaced by the next code with the same row, so the
  counterbalancing stays complete.

## Before a recording day

1. Close every other app. Turn on Focus / Do Not Disturb. Turn Bluetooth off, so no wireless
   headphones can take over the output.
2. Sound settings:
   - Plug the headphones into the Mac's headphone jack.
   - Set the system volume to the calibrated value.
   - Set "Play sound effects through" to the MacBook speakers (or set the alert volume to 0).
3. Calibrate:
   - Run `python -m tsync calibrate` with a sound level meter and an ear coupler (Z or C
     weighting).
   - Do this on the first day, then weekly, and after any change of headphones, cable or system
     volume.
4. Run `python -m tsync check-audio`. It must report 0 underflows.

## Each listener

1. Consent. If you run audiometry, record the thresholds (250–8000 Hz, both ears) separately
   under the participant code.
2. Run `python -m tsync run --code PXX` and go through the experimenter checklist.
3. Fit the headphones (L on the left) and explain the keys: ←/1, →/2, P to pause.
4. Leave the booth and close the door. The app guides the listener through everything,
   including the questionnaire, the examples and the practice.
5. If the app shows "Please call the experimenter", fix what it says and press Space. The
   trial is played from the start.
6. When the session ends, `analyze` prints the thresholds and every flag. Check the flags before
   the listener leaves.

If anything crashes, rerun `python -m tsync run --code PXX --resume`. The session continues
where it stopped, with three warm-up trials.

## Exclusions (fixed in advance)

**A session is excluded if:**
- the practice criterion (8/10) was not reached in 3 rounds (the experimenter may let the
  session continue for the listener's sake, but it is not analysed);
- catch-trial accuracy is below 75% (fewer than 9 of 12);
- more than one track failed to converge;
- the system volume changed during valid trials, or any valid trial had an audio underflow
  (the app prevents both; the check confirms it).

**A track is handled as follows:**
- If it did not converge within 80 trials, it gives no threshold. The condition's fitted
  threshold is still reported, marked.
- If ceiling trials fall inside the six averaged reversals, it is reported as a lower bound
  (≥) and analysed as censored.

## Analysis (fixed in advance)

- **Dependent variable.** Log threshold per listener and condition, from the published
  reversal rule. The psychometric-fit threshold is reported alongside as a check.
- **Primary test.** The effect of the context delay D (0, 35, 50, 71, 100 ms) on log
  threshold, in a linear mixed model with a random intercept per listener:
  - D as a categorical factor (omnibus test);
  - a planned linear trend across D;
  - a planned contrast D0 vs D100.
- **Control.** D0_late vs D0, as a paired difference in octaves with its 95% CI. "The
  displaced ending alone does not raise thresholds" is claimed only if that interval lies
  within ±0.5 octave.
- **Comparison with Micheyl et al. (2013).** Descriptive. Both curves are scaled from 0 to 1
  between their own D0 and D100 points and plotted on one axis. No test is made across the two
  studies.
- **Also reported:**
  - response-order effect (B first vs A first);
  - interval bias;
  - catch accuracy by condition;
  - response times;
  - onset latency and visual-cue lag distributions;
  - every repeated trial and its reason.
