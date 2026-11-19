# tinc: a level change in A-B tone sequences, where grouping helps

The mirror image of Micheyl, Hanson, Demany, Shamma & Oxenham (2013), "Auditory stream segregation
for alternating and synchronous tones", *JEP:HPP* 39:1568 (doi 10.1037/a0032241), Experiment 3.

**Their task.** Listeners say whether one B tone (1000 Hz) got 4 dB louder while ignoring A tones
(420.45 Hz) whose level is roved ±4 dB, with one or two decoy A tones 4 dB up. Ignoring A is
easier once A and B segregate, so their d′ rose with the A–B onset delay: about 1.48 at 0 ms and
2.43 at 100 ms (group).

**This task.** Every number of theirs is kept, and one thing changes: A and B are roved together,
pair by pair, and the decoys raise both tones of a pair. A and B then rise and fall together, and
only the change breaks that, by raising one B tone alone.
- Following B alone is now the losing strategy: B is as unpredictable as their A was.
- Comparing each B with its A wins, and that comparison is what grouping supports.
- So d′ should fall with the delay, the opposite of their curve.

| | theirs (`tinc_micheyl`) | this task (`tinc_main`) |
|---|---|---|
| tones | B 1000 Hz, A 420.45 Hz (15 semitones), 100 ms, 20-ms ramps | same |
| rhythm | one tone per stream every 200 ms (5 Hz), 12 per stream | same |
| delays (A after B) | 0, 35, 50, 71, 100 ms | same |
| B tones | 60 dB (nominal); one (never among the first four) +4 dB on half the trials | the same change, on top of the pair's rove |
| A tones | roved ±4 dB alone; 1–2 a further +4 dB | roved WITH their B; 1–2 pairs a further +4 dB |
| task | yes/no, feedback, d′ per delay | same |
| blocks | one delay per block, 51–52 trials | one delay per block, 52 trials, 2 per delay (104 per delay) |

**What each listening strategy can do.** These are model listeners with 1 dB of noise on every tone (`python -m tinc verify`):

| strategy | this task | theirs |
|---|---|---|
| compare each B with its own A | **d′ 1.7** | 0.5 |
| follow B alone | 0.5 | **2.9** |
| overall loudness of each pair | 0.2 | 0.3 |

The sound carries the same information at every delay. Any effect of the delay is about how the
sequence is heard.

**One subtlety at 100 ms.** At pure alternation, the A before a B and the A after it are equally
near. A listener who pairs tones by timing alone can no longer tell which A belongs to which B
(model d′ 0.5 at 100 ms, 1.6 at 71 ms). That is grouping too, but the 0–71 ms part of the curve
is free of it.

**The session.**
- Questionnaire, then four labelled examples.
- 20 practice trials: 10 at 0 ms and 10 at 100 ms, with feedback.
- 10 blocks of 52. The first five follow an order drawn from the seed and the second five reverse it, so every delay sits at the same mean position.
- About 45–50 minutes.

**Seeds are checked before anyone listens.** `check` draws every trial a session with that seed will play and checks it from the sound: the timing to the sample, every tone's level (0.01 dB), the rove rule, the decoys, the place and size of the change, the balance of each block, headroom, and reproducibility. `run --seed` refuses a seed whose check is missing or failed.

```bash
python -m tinc verify
python -m tinc check --seed 20261115
python -m tinc run   --code P00 --seed 20261115 --uncalibrated --volume 56
python -m tinc analyze data/P00/tinc_session_01
python -m tinc group data/P*/tinc_session_01 --out group_tinc/fig7_style.png
```

**The figures are drawn like their Fig. 7** (their Experiment 3, posterior quantiles of d′ per listener and for the group):
- **Drawing conventions:** the box is the 75% posterior interval, the whiskers the 95% interval, the thick line the mode; the asynchronies are evenly spaced and d′ runs from 0 to 5.
- **`analyze`:** puts their group (left) beside the listener (right).
- **`group`:** draws one panel per listener and a Group panel in their two-column layout, and writes the numbers to a CSV.
- **How the posteriors are made:** their posteriors come from a hierarchical Bayesian probit model. Here hit and false-alarm rates get flat Beta priors, and the group is the draw-by-draw mean of the listeners: the same quantities from a simpler model, without their shrinkage.
- **Their group values** are read off their Fig. 7 by eye (to about 0.1). The two ends agree with their text (1.48 and 2.43).

**Notes.**
- **Calibration.** Since 2026-11-17 listeners are recorded under `audiometry/README.md`: `--uncalibrated` at the rig's fixed volume, a daily jig check, and an ear-simulator measurement later (it includes 420 Hz and 1 kHz) that gives every session's levels in dB SPL. Calibrated runs (`python -m tsync calibrate`, then `check --calibrated` and `run` without `--uncalibrated`) are not used for these recordings.
- **Replication.** `tinc_micheyl` is their condition in the same code, to measure their curve in the same listeners.
- **Why not 6 Hz.** At 6 Hz the 100-ms tones leave 67 ms between tones of a stream. Their 71 and 100 ms delays would then run each A into the next B, and 100 ms would no longer be alternation. Keeping their curve comparable needs their 5 Hz.
