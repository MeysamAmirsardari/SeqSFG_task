# tcmr: CMR after precursors sheared across frequency

Does an ordered temporal shear of the precursors break across-frequency grouping less than an
unordered one? The measure is comodulation masking release (CMR): a 1-kHz tone is easier to hear
in a 20-Hz noise band when four remote bands share its envelope, but only if all five bands are
heard as one stream (Christiansen & Oxenham 2014, JASA 135:3520, doi 10.1121/1.4872300).

**Stimulus (as Christiansen & Oxenham 2014, Exp. 1).**
- Five 20-Hz noise bands at 0.25, 0.5, 1, 2 and 4 kHz, 60 dB SPL each.
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
- thresholds (dB SPL at 70.7%);
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

- **Bands and bursts.** The same five 20-Hz bands at 60 dB SPL. Bursts are 50 ms with 10-ms ramps, one per band every 200 ms (5 Hz). Each interval has 5 precursors, then a synchronous final burst that carries the 1-kHz tone. Every burst is new noise at its natural level.
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
