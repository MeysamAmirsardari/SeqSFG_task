"""Every procedural number of the package, with the clause it comes from.

Sources, read on 2026-11-17:
  BSA 2018   British Society of Audiology, Recommended Procedure: Pure-tone air-conduction and
             bone-conduction threshold audiometry with and without masking (August 2018).
             Sections 4.2, 5.1, 6.1-6.8, 8.1, 11.1-11.2, Appendix C.
  ASHA 2005  American Speech-Language-Hearing Association, Guidelines for Manual Pure-Tone
             Threshold Audiometry (GL2005-00014).
  ISO 389-8  ISO 389-8:2004, RETSPLs for pure tones and circumaural earphones, Table 1, clause 4.
  Smull 2019 Smull, Madsen & Margolis (2019), Evaluation of two circumaural earphones for
             audiometry, Ear Hear 40:177-183, doi 10.1097/AUD.0000000000000585: RETSPLs for the
             Sennheiser HD 280 Pro.

Numbers that are our own choices, not a standard's, are marked OURS with the reason. The screen
follows BSA 2018, a procedure for manual clinical audiometry; an automated screen is outside the
scope of BSA (3.2) and of ASHA 2005, so the automatic parts are ours (README, "not standard").
"""
from __future__ import annotations

from typing import Dict

# ---- reference equivalent threshold sound pressure levels (dB re 20 uPa) -----------------------
# ISO 389-8:2004 Table 1: Sennheiser HDA 200 on an IEC 60318-1 ear simulator with the
# IEC 60318-2:1998 Figure 1 adapter (headband force 10.0 +- 1.0 N, clause 4).
RETSPL_HDA200: Dict[int, float] = {
    125: 30.5, 160: 26.0, 200: 22.0, 250: 18.0, 315: 15.5, 400: 13.5, 500: 11.0, 630: 8.0,
    750: 6.0, 800: 6.0, 1000: 5.5, 1250: 6.0, 1500: 5.5, 1600: 5.5, 2000: 4.5, 2500: 3.0,
    3000: 2.5, 3150: 4.0, 4000: 9.5, 5000: 14.0, 6000: 17.0, 6300: 17.5, 8000: 17.5,
}
ISO389_8_HEADBAND_FORCE_N = 10.0          # the HDA 200's own headband force (clause 4)
ISO389_8_SEPARATION_MM = 145.0            # cups this far apart when that force is measured
ISO389_8_HEADBAND_HEIGHT_MM = 130.0
ISO389_8_CALIBRATION_TEMP_C = (21.0, 25.0)

# The HD 280 Pro's own RETSPLs (Smull 2019). TO FILL from the paper's table, together with the
# coupler, adapter and force it states; the report then scores hearing levels with them. Until
# then the report scores with the HDA 200 values above and shows the difference.
RETSPL_HD280PRO: Dict[int, float] = {}
# Four of the paper's values as transcribed by a third party (UW AHSLAB Hearing Screener,
# js/perceptualCalibration.js, "IEC 60318 coupler with flat-plate adapter"). Not checked against
# the paper; used only to show the size of the HDA 200 approximation.
RETSPL_HD280PRO_SECONDARY: Dict[int, float] = {500: 6.8, 1000: 1.4, 2000: 1.9, 4000: 2.2}

# ---- BSA 2018 5.1: questions before the test ---------------------------------------------------
PRE_TEST_QUESTIONS = (
    ("noise_24h", "Loud noise in the last 24 hours (had to shout or raise the voice to talk at 1 m)? [y/n]"),
    ("tinnitus", "Tinnitus (ringing, hissing) now? [none/left/right/both]"),
    ("cold_or_ear_pain", "A cold, an ear infection or ear pain now? [y/n]"),
    ("ear_exam", "Ear examination (otoscopy): who did it and what was found, or 'not done'"),
    ("cleared", "Glasses, earrings, headwear and hair out from under the cups? [y/n]"),
)

# ---- BSA 2018 6.4 test order ------------------------------------------------------------------
TEST_ORDER_HZ = (1000, 2000, 4000, 8000, 500, 250)
RETEST_HZ = 1000
RETEST_TOLERANCE_DB = 5.0          # BSA 6.4; ASHA 2005 also accepts the lower if it still differs
PASS_DIFF_FLAG_DB = 5.0            # OURS: a repeated ear differing from its first pass by more

# ---- BSA 2018 6.5 and ASHA 2005: timing ---------------------------------------------------------
TONE_S = (1.0, 2.0)                # BSA "varied between 1 and 3 seconds", ASHA "1 to 2 seconds"
# Interval: BSA "varied between 1 second and at least 3 seconds"; ASHA "not shorter than the test
# tone". Drawn from [the longer of the tones before and after it, 3.5 s].
INTERVAL_MAX_S = 3.5
KEY_QUIET_S = 1.0                  # OURS: no tone sooner than this after the key is released

# ---- BSA 2018 6.6 familiarisation --------------------------------------------------------------
FAMILIARISATION_HZ = 1000
FAMILIARISATION_DB_HL = 40.0       # "e.g. at 40 dB HL for a normally hearing subject"
FAMILIARISATION_STEP_DB = 10.0     # no response: up in 10-dB steps until a response
FAMILIARISATION_TRIES = 2          # not consistent with onset and offset: repeat, then re-instruct

# ---- BSA 2018 6.7 method for finding threshold ---------------------------------------------------
STEP_DOWN_DB = 10.0
STEP_UP_DB = 5.0
# BSA 6.7: the lowest level with responses on "two out of two, three or four (i.e. 50 % or more)"
# ascending presentations, "with a minimum of two responses". (ASHA 2005 states the minimum as two
# of three; ISO 8253-1:1989 asks three of up to five ascents, or two of three in its shortened
# method. The criterion here is BSA's.)
MIN_RESPONSES = 2
MIN_FRACTION = 0.5
NEXT_START_ABOVE_DB = 30.0         # next frequency: "30 dB above the adjacent threshold"

# ---- OURS: limits of a research screen -----------------------------------------------------------
FLOOR_DB_HL = -10.0                # the usual bottom of an audiometer's dial
CAP_DB_HL = 60.0                   # a 20-dB HL screen never needs more; BSA's familiarisation
                                   # ceiling is 80 dB HL and its high-level check is 60 dB HL
MAX_PRESENTATIONS = 30             # per threshold; past this the threshold is "unreliable"
PASS_DB_HL = 20.0                  # the usual research criterion, octaves 250-8000 Hz, both ears

# ---- OURS: automatic scoring --------------------------------------------------------------------
RESPONSE_FROM_S = 0.15             # a press counts from 150 ms after onset (faster = anticipation)
RESPONSE_UNTIL_AFTER_OFFSET_S = 1.0  # ... to 1 s after offset
QUIET_AFTER_FALSE_S = 2.0          # after a false response the next tone waits this long
RELEASE_WINDOW_S = (-0.25, 1.25)   # familiarisation: release within this of the offset (BSA 6.6)
FALSE_REINSTRUCT = 3               # false responses since the last instructions: re-instruct
# Catch trials (silent presentations, as in automated audiometry, e.g. AMTAS) and false responses
# per threshold: 2 or more and the frequency is repeated once after re-instruction; 2 or more again
# and its threshold is marked unreliable.
CATCH_PROB = 0.15
FALSE_REPEAT_AT = 2
HELD_KEY_PAUSE_S = 10.0            # a key held this long with no tone: pause for the experimenter

# ---- BSA 2018 4.2, Appendix C, 8.1 -------------------------------------------------------------
AMBIENT_MAX_DBA = 35.0             # "should not exceed 35 dB (A)"
INTERAURAL_ATTENUATION_DB = (40.0, 80.0)  # BSA 8.1, circumaural earphones (assumed, not measured)

# ---- BSA 2018 11.1 Stage A (daily, checks 1-8; weekly 9-12), adapted to one laptop and one pair ----
LOW_CHECK_DB_HL = 15.0             # check 3: "just audible tones (e.g. 10 dB HL or 15 dB HL)"
HIGH_CHECK_DB_HL = 60.0            # check 5: "high-level listening check ... 60 dB HL for a-c"
CABLE_CHECK_DB_HL = 40.0           # check 6: move the cable and plug while a tone plays
DAILY_CHECKLIST = (
    ("pair", "This is the labelled pair {headphones}, and no other headphones are plugged in (check 2)"),
    ("cushions", "Earcups, cushions, headband, cable and plug clean and undamaged (check 1)"),
    ("headband", "Headband tension and swivels normal; nothing loose (check 11)"),
    ("plug", "3.5-mm plug straight in the laptop's jack, no adapter; it stays plugged in until the "
             "last session today (the Mac re-detects the headphones at every plug-in)"),
    ("power", "Laptop on mains power; Bluetooth off; Do Not Disturb on"),
    ("alerts", "System sound effects play through the laptop's own speakers, not the headphones"),
    ("key", "The response key (space bar) works and is quiet (check 8)"),
    ("room", "Booth door closed, no fan or other noise source running"),
)

# ---- OURS: stability of the playback chain ---------------------------------------------------------
DRIFT_WARN_DB = 0.5                # jig mean this far from the first one: look at the seating
DRIFT_LIMIT_DB = 1.0               # this far: stop; no listeners until it is explained
RESEAT_RANGE_DB = 1.0              # readings of one cup spread wider than this: reseat again
WEEKLY_FULL_JIG_DAYS = 7           # the jig at every frequency at least this often
