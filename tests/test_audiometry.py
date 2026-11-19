"""audiometry (2026-11-17): the rig's level scale, the tones and noise bands, the BSA 2018 threshold
procedure, the test order, the logs, and the screen app run end to end with simulated listeners
(no sound card, no system volume)."""
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from audiometry import procedure as P
from audiometry import rig as R
from audiometry import standards as S
from audiometry import tones as T

ROOT = Path(__file__).resolve().parents[1]


# ---- the standards' numbers ---------------------------------------------------------------------------
def test_iso_389_8_table_values():
    # ISO 389-8:2004 Table 1, Sennheiser HDA 200 (checked against the standard's page 2)
    want = {250: 18.0, 500: 11.0, 1000: 5.5, 2000: 4.5, 3000: 2.5, 4000: 9.5, 6000: 17.0, 8000: 17.5}
    for f, v in want.items():
        assert S.RETSPL_HDA200[f] == v
    assert len(S.RETSPL_HDA200) == 23


def test_bsa_numbers():
    assert S.TEST_ORDER_HZ == (1000, 2000, 4000, 8000, 500, 250)
    assert (S.STEP_DOWN_DB, S.STEP_UP_DB, S.MIN_RESPONSES, S.MIN_FRACTION) == (10, 5, 2, 0.5)
    assert S.FAMILIARISATION_DB_HL == 40 and S.NEXT_START_ABOVE_DB == 30
    assert S.TONE_S[0] >= 1.0 and S.TONE_S[1] <= 2.0          # inside BSA 1-3 s and ASHA 1-2 s
    assert S.INTERVAL_MAX_S >= 3.0                            # BSA: up to at least 3 s


def test_scoring_table_waits_for_the_paper():
    assert R.scoring_table() == "hda200"                      # until RETSPL_HD280PRO is filled
    assert set(S.RETSPL_HD280PRO_SECONDARY) == {500, 1000, 2000, 4000}


# ---- the level scale matches tcmr and tinc --------------------------------------------------------------
def test_scale_matches_the_tasks():
    from tcmr.config import Config as TC
    from tinc.config import Config as IC
    tc = TC.load(ROOT / "tcmr/configs/tcmr_exp1.json")
    ic = IC.load(ROOT / "tinc/configs/tinc_8db.json")
    assert tc.uncal_peak_60db == R.PEAK_AT_60 == ic.uncal_peak_60db
    for f in tc.bands_hz:
        assert R.fkey(f) in R.ALL_HZ and R.fkey(f) in R.TCMR_BANDS_HZ
    assert R.fkey(ic.f_a_hz) in R.ALL_HZ and R.fkey(ic.f_b_hz) in R.ALL_HZ
    from tcmr.cli import TEST_AMP
    assert TEST_AMP == R.TEST_AMP
    assert R.TEST_NOMINAL_DB == pytest.approx(67.9588, abs=1e-4)


def test_tcmr_band_power_is_the_sine_power():
    # a "60 dB" tcmr band (natural burst level) has the long-term power of a 0.02-peak sine
    from tcmr.config import Config as TC
    from tcmr.stimulus import band_bursts
    tc = TC.load(ROOT / "tcmr/configs/tcmr_exp1.json")
    rng = np.random.default_rng(5)
    p = np.mean([np.mean(np.array(band_bursts(tc, rng, False)) ** 2, axis=1) for _ in range(1500)],
                axis=0)
    assert np.all(np.abs(10 * np.log10(p)) < 0.25)            # unit long-term power per band
    x = T.band_noise([1000.0], 30.0, 60.0, "left", np.random.default_rng(1))[:, 0].astype(float)
    mid = x[48000:-48000]
    assert 10 * np.log10(np.mean(mid ** 2) / (0.02 ** 2 / 2)) == pytest.approx(0.0, abs=0.3)
    spec = np.abs(np.fft.rfft(mid)) ** 2
    fr = np.fft.rfftfreq(len(mid), 1 / 48000)
    assert np.sum(spec[(fr > 985) & (fr < 1015)]) / np.sum(spec) > 0.99


def test_level_arithmetic():
    assert R.peak_for(60) == pytest.approx(0.02)
    assert R.nominal_for_peak(R.peak_for(37.3)) == pytest.approx(37.3)
    off = R.offset_from_reading(70.0)                         # check tone read 70.0 dB (Z, 1 kHz)
    assert off == pytest.approx(70.0 - 67.9588, abs=1e-4)
    n = R.nominal_from_hl(20.0, 1000, off)
    assert n + off == pytest.approx(25.5)                     # 20 dB HL = 25.5 dB SPL at 1 kHz
    assert R.hl_from_nominal(n, 1000, off) == pytest.approx(20.0)
    with pytest.raises(KeyError):
        R.retspl(420.45)


def test_c_weighting_correction():
    assert R.weighting_db(1000, "C") == pytest.approx(0.0, abs=0.01)
    assert R.weighting_db(250, "C") == pytest.approx(0.0, abs=0.05)
    assert R.weighting_db(4000, "C") == pytest.approx(-0.83, abs=0.05)
    assert R.weighting_db(8000, "C") == pytest.approx(-3.05, abs=0.05)
    assert R.weighting_db(8000, "Z") == 0.0
    # an 8-kHz tone reading 70.0 on C weighting is 73.05 unweighted
    assert R.offset_from_reading(70.0, 8000, "C") == pytest.approx(73.05 - R.TEST_NOMINAL_DB, abs=0.05)
    with pytest.raises(ValueError):
        R.weighting_db(1000, "A")


def test_parse_reading():
    assert R.parse_reading("68.4") == 68.4
    assert R.parse_reading("۶۸٫۴") == 68.4
    assert R.parse_reading("68,4") == 68.4
    assert R.parse_reading("abc") is None and R.parse_reading("200") is None


# ---- tones -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("f", [250.0, 1000.0, 8000.0])
def test_tone(f):
    x = T.tone(f, 1.0, 0.1, "right")
    assert x.shape == (48000, 2) and x.dtype == np.float32
    assert np.all(x[:, 0] == 0)                                # the other cup is silent
    y = x[:, 1].astype(float)
    mid = y[4800:-4800]                                        # the steady part
    assert np.sqrt(np.mean(mid ** 2)) == pytest.approx(0.1 / np.sqrt(2), rel=1e-3)   # rms = peak/sqrt2
    assert np.max(np.abs(y)) <= 0.1 + 1e-6
    spec = np.abs(np.fft.rfft(mid * np.hanning(len(mid))))
    fpk = np.argmax(spec) * 48000 / len(mid)
    assert abs(fpk - f) < 2.0
    assert abs(y[0]) < 1e-4 and abs(y[-1]) < 1e-4             # starts and ends at zero


def test_ramps_click_free_and_in_range():
    rise, fall = T.ramp_times_ms()
    assert rise == pytest.approx(29.0, abs=1.0) and fall == pytest.approx(29.0, abs=1.0)
    assert 20.0 <= rise <= 50.0 and 20.0 <= fall <= 50.0


def test_tone_refuses_loud_or_wrong():
    with pytest.raises(ValueError):
        T.tone(1000, 1.0, 0.6, "left")
    with pytest.raises(ValueError):
        T.tone(1000, 1.0, 0.1, "both")


# ---- the threshold search ---------------------------------------------------------------------------
def run(search, hears, limit=200):
    seen = []
    while not search.done and limit:
        lv = search.current().level
        assert search.floor <= lv <= search.cap
        seen.append(lv)
        search.record(hears(lv))
        limit -= 1
    return seen


@pytest.mark.parametrize("true", [-5, 0, 5, 10, 15, 20, 35, 55])
def test_deterministic_listener_found_exactly(true):
    s = P.ThresholdSearch(40)
    run(s, lambda lv: lv >= true)
    assert s.status == "found" and s.threshold == true


def test_familiarisation_goes_up_in_10s_then_down_10_up_5():
    s = P.ThresholdSearch(40)
    seen = run(s, lambda lv: lv >= 52)
    assert seen[:3] == [40, 50, 60]                            # missed 40 and 50: +10 each
    assert seen[3] == 50                                       # heard 60: down 10
    assert seen[4] == 55                                       # missed 50: up 5
    assert s.threshold == 55


def test_floor_and_cap():
    s = P.ThresholdSearch(40)
    seen = run(s, lambda lv: True)
    assert s.status == "at_floor" and s.threshold == S.FLOOR_DB_HL and min(seen) == S.FLOOR_DB_HL
    s = P.ThresholdSearch(40)
    seen = run(s, lambda lv: False)
    assert s.status == "no_response_at_cap" and s.threshold is None and max(seen) == S.CAP_DB_HL


def test_only_ascending_responses_count():
    s = P.ThresholdSearch(40)
    run(s, lambda lv: lv >= 20)
    asc = s.summary()["ascending"]
    assert asc["20"][0] == 2 and asc["20"][1] == 2
    assert all(st[1] == 0 for st in s.summary()["steps"][:3])  # 40, 30, 20 descending


def test_two_of_three_ascending():
    s = P.ThresholdSearch(40)
    asc15 = {"n": 0}

    def hears(lv):
        if lv >= 20:
            return True
        if lv == 15 and s.phase == "ascend":
            asc15["n"] += 1
            return asc15["n"] >= 2
        return False
    run(s, hears)
    assert s.threshold == 15 and s.asc_resp[15] == 2 and s.asc_pres[15] == 3


def test_criterion_is_half_with_two_responses():
    s = P.ThresholdSearch(40)
    for (r, n), ok in {(2, 2): True, (2, 3): True, (2, 4): True, (3, 6): True, (2, 5): False,
                       (1, 1): False, (1, 2): False}.items():
        s.asc_resp, s.asc_pres = {15.0: r}, {15.0: n}
        assert s.meets(15.0) is ok


def test_stochastic_listener_is_unbiased():
    rng = np.random.default_rng(3)
    est = []
    for _ in range(400):
        true = rng.uniform(-5, 30)

        def hears(lv, t=true):
            return rng.random() < 1 / (1 + math.exp(-(lv - t) / 1.5))
        s = P.ThresholdSearch(40)
        run(s, hears)
        assert s.status in ("found", "at_floor", "unreliable")
        if s.threshold is not None:
            est.append(s.threshold - true)
    est = np.array(est)
    assert len(est) > 380
    assert abs(np.median(est)) <= 3.0
    assert np.percentile(np.abs(est), 95) <= 8.0


# ---- the test order ---------------------------------------------------------------------------------
def drive(plan, thresholds):
    order = []
    while True:
        t = plan.next()
        if t is None:
            break
        order.append((t.ear, t.freq, t.kind, t.ear_pass, t.start, t.familiarise))
        s = plan.search_for(t)
        th = thresholds(t)
        run(s, lambda lv: lv >= th)
        plan.finish(t, s)
    return order


def test_order_and_starts():
    plan = P.ScreenPlan("left")
    order = drive(plan, lambda t: 10 if t.ear == "left" else 5)
    left = [o for o in order if o[0] == "left"]
    right = [o for o in order if o[0] == "right"]
    assert [o[1] for o in left] == [1000, 2000, 4000, 8000, 500, 250, 1000]
    assert left[-1][2] == "retest" and left[0][5] is True       # familiarisation first
    assert [o[1] for o in right] == [1000, 2000, 4000, 8000, 500, 250]   # no retest
    assert left[0][4] == 40 and left[1][4] == 40               # 10 + 30
    assert right[0][4] == 40 and right[1][4] == 35             # 5 + 30
    fin = plan.final()
    assert all(v == 10 for v in fin["left"].values()) and all(v == 5 for v in fin["right"].values())
    assert plan.events == ["other_ear"] and plan.flags == []


def test_retest_within_5_keeps_lower():
    calls = {"n": 0}

    def th(t):
        if t.ear == "left" and t.freq == 1000:
            calls["n"] += 1
            return 15 if calls["n"] == 1 else 10
        return 10
    plan = P.ScreenPlan("left")
    drive(plan, th)
    assert plan.final()["left"][1000.0] == 10
    assert plan.ears[0].retest_diff_db == 5


def test_retest_beyond_5_repeats_the_ear_once_and_compares_passes():
    calls = {"n": 0}

    def th(t):
        if t.ear == "left" and t.freq == 1000:
            calls["n"] += 1
            return {1: 20, 2: 5, 3: 10, 4: 10}[calls["n"]]
        if t.ear == "left" and t.freq == 4000:
            return 30 if t.ear_pass == 1 else 15                 # passes differ by 15 dB
        return 10
    plan = P.ScreenPlan("left")
    order = drive(plan, th)
    assert plan.events[0] == "reinstruct"
    passes = [o for o in order if o[0] == "left" and o[3] == 2]
    assert len(passes) == 7 and passes[0][5] is True            # repeated, re-familiarised
    right = [o for o in order if o[0] == "right"]
    assert right[-1][2] == "retest"                             # variation seen: retest 2nd ear
    assert plan.final()["left"][1000.0] == 10
    assert any("re-instructed" in f for f in plan.flags)
    assert any("4000 Hz: the two passes differ by 15" in f for f in plan.flags)


def test_retest_still_beyond_5_keeps_lower_and_flags():
    calls = {"n": 0}

    def th(t):
        if t.ear == "left" and t.freq == 1000:
            calls["n"] += 1
            return {1: 20, 2: 5, 3: 25, 4: 10}[calls["n"]]
        return 10
    plan = P.ScreenPlan("left")
    drive(plan, th)
    assert plan.final()["left"][1000.0] == 10
    assert any("still differed" in f for f in plan.flags)


def test_false_response_repeat_then_unreliable():
    plan = P.ScreenPlan("right")
    t = plan.next()
    s = plan.search_for(t)
    run(s, lambda lv: lv >= 10)
    plan.repeat(t, s, {"false_responses": 2})
    assert plan.events == ["reinstruct"]
    t2 = plan.next()
    assert (t2.ear, t2.freq, t2.retry, t2.familiarise) == ("right", 1000.0, 1, False)
    with pytest.raises(RuntimeError):
        plan.repeat(t2, s)
    s2 = plan.search_for(t2)
    run(s2, lambda lv: lv >= 10)
    plan.finish(t2, s2, {"false_responses": 3}, unreliable="3 false responses again")
    rec = plan.current_ear.tests[1000.0]
    assert rec["status"] == "unreliable_false_responses" and rec["threshold_db_hl"] is None
    assert rec["threshold_measured_db_hl"] == 10 and plan.current_ear.attempts


# ---- scoring presses --------------------------------------------------------------------------------
def test_response_window_and_consistency():
    on, off = 10.0, 11.5
    assert P.heard(on, off, [10.1]) == (False, None)            # anticipation
    assert P.heard(on, off, [10.4])[0] and P.heard(on, off, [12.4])[0]
    assert not P.heard(on, off, [12.6])[0]
    assert P.consistent(on, off, 10.4, 11.7)
    assert not P.consistent(on, off, 10.4, None)
    assert not P.consistent(on, off, 11.6, 11.9)                # pressed after the tone ended
    assert not P.consistent(on, off, 10.4, 13.0)                # held far too long


# ---- logs, rig, offsets ---------------------------------------------------------------------------------
class FakeOutput:
    def __init__(self, volume, say=print):
        self.device = {"name": "External Headphones", "index": 0}
        self.volume = {"output_volume": volume}
        self.played = []
        self.unchanged = True

    def volume_unchanged(self):
        return self.unchanged

    def play_buffer(self, buf):
        assert np.max(np.abs(buf)) <= R.MAX_PEAK
        self.played.append(buf.shape)

    def play(self, f, cup, seconds, peak=R.TEST_AMP):
        assert peak <= R.MAX_PEAK
        self.played.append((f, cup, seconds, peak))

    def playing(self):
        return True

    def stop(self):
        pass

    def wait(self, timeout_s=10.0):
        pass

    def close(self):
        pass


def scripted(answers):
    it = iter(answers)
    return lambda q: next(it)


def args(**kw):
    base = dict(coupler="jig", frequencies=["daily"], cups=["left", "right"], reseats=3,
                trial=False, volume=None, headphones=None, weighting="C", meter="SL-100",
                operator="me", note="", new_rig=False, no_audio=True, csv=None, signal="tone",
                nominal=None)
    base.update(kw)
    return SimpleNamespace(**base)


ENV = ["", ""]                    # temperature and humidity prompts, skipped


@pytest.fixture
def lab(tmp_path, monkeypatch):
    from audiometry import levels as L
    monkeypatch.setattr(L, "Output", FakeOutput)
    monkeypatch.setattr(L, "LISTEN_GAP_S", 0.0)
    monkeypatch.setattr(R, "environment", lambda: {"test": True})
    return tmp_path


def setup_rig(lab):
    from audiometry import levels as L
    assert L.cmd_setup(args(headphones="HD280Pro-1", volume=38), say=lambda s: None, root=lab) == 0


def test_setup_refuses_a_silent_change_and_keeps_its_date(lab):
    from audiometry import levels as L
    setup_rig(lab)
    since = R.Rig.load(lab).since
    time.sleep(1.1)
    assert L.cmd_setup(args(headphones="HD280Pro-1", volume=38), say=lambda s: None, root=lab) == 0
    assert R.Rig.load(lab).since == since                     # same values: nothing rewritten
    assert L.cmd_setup(args(headphones="HD280Pro-1", volume=40), say=lambda s: None, root=lab) == 2
    assert R.Rig.load(lab).volume == 38


def test_levels_baseline_drift_and_trials(lab):
    from audiometry import levels as L
    setup_rig(lab)
    assert L.cmd_levels(args(trial=True, volume=30, cups=["left"], reseats=1),
                        ask=scripted(ENV + ["60"]), say=lambda s: None, root=lab) == 0
    assert L.cmd_levels(args(), ask=scripted(ENV + ["68.0", "68.2", "67.9", "68.1", "68.0", "68.2"]),
                        say=lambda s: None, root=lab) == 0
    recs = R.level_records(lab, headphones="HD280Pro-1", volume=38)
    assert len(recs) == 2 and not any(r["trial"] for r in recs)
    assert R.baseline(recs, "left", 1000)["mean_db"] == pytest.approx(68.033, abs=1e-3)
    assert L.cmd_levels(args(), ask=scripted(ENV + ["68.5"] * 6), say=lambda s: None, root=lab) == 0
    assert L.cmd_levels(args(), ask=scripted(ENV + ["69.6"] * 6), say=lambda s: None, root=lab) == 1
    recs = R.level_records(lab, headphones="HD280Pro-1", volume=38)
    assert [r["passed"] for r in recs] == [True, True, True, True, False, False]


def test_failed_checks_never_become_the_scale(lab):
    from audiometry import levels as L
    setup_rig(lab)
    n = len(R.ALL_HZ) * 2
    L.cmd_levels(args(frequencies=["all"], reseats=1), ask=scripted(ENV + ["68"] * n),
                 say=lambda s: None, root=lab)
    # a mis-seated 250-Hz check reads 60: it fails, is saved, and changes nothing
    assert L.cmd_levels(args(frequencies=["250"], reseats=1), ask=scripted(ENV + ["60", "68"]),
                        say=lambda s: None, root=lab) == 1
    recs = R.level_records(lab, headphones="HD280Pro-1", volume=38)
    first = R.offsets_for(recs, "jig", R.SCREEN_HZ, which="first")
    latest = R.offsets_for(recs, "jig", R.SCREEN_HZ, which="latest")
    assert first["left"][250.0] == pytest.approx(latest["left"][250.0])
    assert first["left"][250.0] == pytest.approx(68 - R.weighting_db(250, "C") - R.TEST_NOMINAL_DB, abs=1e-3)


def test_linearity_and_band_records_are_not_offsets(lab):
    from audiometry import levels as L
    setup_rig(lab)
    assert L.cmd_levels(args(coupler="ear-simulator", weighting="Z", nominal=85.0, reseats=1),
                        ask=scripted(ENV + ["87", "87"]), say=lambda s: None, root=lab) == 0
    assert L.cmd_levels(args(coupler="ear-simulator", weighting="Z", signal="band",
                             frequencies=["1000"], reseats=1),
                        ask=scripted(ENV + ["62", "62"]), say=lambda s: None, root=lab) == 0
    recs = R.level_records(lab)
    assert len(recs) == 4 and not any(R.usable(r) for r in recs)
    assert R.offsets_for(recs, "ear-simulator", R.DAILY_HZ) is None
    assert L.cmd_levels(args(coupler="ear-simulator", nominal=90.0, reseats=1),
                        ask=scripted(ENV), say=lambda s: None, root=lab) == 2   # above 85 nominal


def test_wide_spread_asks_for_more_readings(lab):
    from audiometry import levels as L
    setup_rig(lab)
    ans = ENV + ["67.0", "68.5", "68.0", "y", "68.1", "68.0", "n"] + ["68"] * 3
    assert L.cmd_levels(args(), ask=scripted(ans), say=lambda s: None, root=lab) == 0
    left = [r for r in R.level_records(lab) if r["cup"] == "left"][0]
    assert len(left["readings_db"]) == 5


def test_ear_simulator_offsets_and_report(lab):
    from audiometry import levels as L
    from audiometry.cli import thresholds_block
    from audiometry.report import final_hl, verdict
    setup_rig(lab)
    n = len(R.ALL_HZ) * 2
    assert L.cmd_levels(args(frequencies=["all"], reseats=1), ask=scripted(ENV + ["68"] * n),
                        say=lambda s: None, root=lab) == 0
    assert L.cmd_levels(args(frequencies=["all"], reseats=1, coupler="ear-simulator"),
                        ask=scripted(ENV + ["70"] * n), say=lambda s: None, root=lab) == 0
    recs = R.level_records(lab, headphones="HD280Pro-1", volume=38)
    sim = R.offsets_for(recs, "ear-simulator", R.ALL_HZ)
    jig = R.offsets_for(recs, "jig", R.SCREEN_HZ, which="first")
    assert sim["left"][1000.0] == pytest.approx(70 - R.TEST_NOMINAL_DB, abs=1e-3)
    plan = {"final_db_hl": {e: {f"{f:g}": 10.0 for f in R.SCREEN_HZ} for e in R.CUPS},
            "final_status": {e: {f"{f:g}": "found" for f in R.SCREEN_HZ} for e in R.CUPS}}
    screen = {"thresholds": thresholds_block(plan, jig)}
    hl = final_hl(screen, sim)
    assert all(v == pytest.approx(12.0) for d in hl.values() for v in d.values())
    assert verdict(hl, screen) == "pass"
    plan["final_db_hl"]["left"]["4000"] = 25.0
    s2 = {"thresholds": thresholds_block(plan, jig)}
    assert verdict(final_hl(s2, None), s2) == "fail"
    plan["final_db_hl"]["left"]["4000"] = None
    plan["final_status"]["left"]["4000"] = "no_response_at_cap"
    s3 = {"thresholds": thresholds_block(plan, jig)}
    assert verdict(final_hl(s3, None), s3) == "fail"
    plan["final_status"]["left"]["4000"] = "unreliable"
    s4 = {"thresholds": thresholds_block(plan, jig)}
    assert verdict(final_hl(s4, None), s4) == "incomplete"


DAILY_OK = (["y"] * len(S.DAILY_CHECKLIST) + ["", "", ""] + ["68"] * 6
            + ["", "y", "", "y"] + ["", "y", "", "y"] + ["", "y"])


def test_daily_needs_the_baseline_first_and_logs_failures(lab):
    from audiometry import levels as L
    from audiometry.cli import daily_ok_today
    setup_rig(lab)
    assert L.cmd_daily(args(), ask=scripted([]), say=lambda s: None, root=lab) == 2
    n = len(R.ALL_HZ) * 2
    L.cmd_levels(args(frequencies=["all"], reseats=1), ask=scripted(ENV + ["68"] * n),
                 say=lambda s: None, root=lab)
    ans = ["y", "n"] + ["y"] * (len(S.DAILY_CHECKLIST) - 2)
    assert L.cmd_daily(args(), ask=scripted(ans), say=lambda s: None, root=lab) == 1
    assert L.cmd_daily(args(), ask=scripted(DAILY_OK), say=lambda s: None, root=lab) == 0
    rig = R.Rig.load(lab)
    assert daily_ok_today(rig, lab)
    # a later failing check today blocks the screen again
    bad = list(DAILY_OK)
    bad[len(S.DAILY_CHECKLIST) + 3:len(S.DAILY_CHECKLIST) + 9] = ["70"] * 6
    assert L.cmd_daily(args(), ask=scripted(bad), say=lambda s: None, root=lab) == 1
    assert not daily_ok_today(rig, lab)
    days = R.read_jsonl(R.rig_dir(lab) / "daily.jsonl")
    assert [d["ok"] for d in days] == [False, True, False]


def test_daily_refuses_a_baseline_out_of_range(lab):
    from audiometry import levels as L
    setup_rig(lab)
    n = len(R.ALL_HZ) * 2
    L.cmd_levels(args(frequencies=["all"], reseats=1), ask=scripted(ENV + ["58"] * n),
                 say=lambda s: None, root=lab)            # 10 dB quiet: 60 dB HL needs > 85 nominal
    assert L.cmd_daily(args(), ask=scripted([]), say=lambda s: None, root=lab) == 2


def test_range_folder_code_and_pretest(lab):
    from audiometry.cli import CODE_RE, RESERVED, _screen_folder, pre_test
    ok = {c: {R.fkey(f): 0.0 for f in R.SCREEN_HZ} for c in R.CUPS}
    assert R.range_problems(ok) == []
    quiet = {c: {R.fkey(f): -20.0 for f in R.SCREEN_HZ} for c in R.CUPS}
    assert R.range_problems(quiet)
    assert _screen_folder("S01", lab).name == "screen_01"
    (lab / "S01" / "screen_01").mkdir(parents=True)
    assert _screen_folder("S01", lab).name == "screen_02"
    assert CODE_RE.fullmatch("S01") and not CODE_RE.fullmatch("S01\n") and "rig" in RESERVED
    yes = ["n", "none", "n", "not done", "y", "y"]
    assert pre_test(scripted(yes), say=lambda s: None)["ear_exam"] == "not done"
    assert pre_test(scripted(["n", "none", "n", "not done", "n"]), say=lambda s: None) is None


# ---- the screen app, end to end, with simulated listeners -----------------------------------------------
class FakePB:
    def __init__(self, t_on, n, sr):
        self.t_on, self.end = t_on, t_on + n / sr
        self.late = self.aborted = False

    @property
    def finished(self):
        return self.aborted or time.perf_counter() > self.end

    def onset_perf(self):
        return self.t_on

    def report(self):
        return {"onset_error_ms": 0.0, "underflows": 0}


class FakeEngine:
    """Keeps the real engine's rule: one playback at a time."""

    def __init__(self):
        self.played, self.aborts, self.last, self.alive = [], 0, None, True

    def play_at(self, buf, t):
        assert np.max(np.abs(buf)) <= R.MAX_PEAK
        if self.last is not None and not self.last.finished:
            raise RuntimeError("a trial is already playing")
        self.played.append((t, buf.shape[0], float(np.max(np.abs(buf[:, 0]))),
                            float(np.max(np.abs(buf[:, 1])))))
        self.last = FakePB(t, buf.shape[0], R.SAMPLE_RATE)
        return self.last

    def abort(self, pb):
        pb.aborted = True
        self.aborts += 1


def fast(monkeypatch):
    from audiometry import screen_app as A
    for k, v in {"TONE_S": (0.06, 0.09), "INTERVAL_MAX_S": 0.12, "RESPONSE_FROM_S": 0.01,
                 "RESPONSE_UNTIL_AFTER_OFFSET_S": 0.04, "QUIET_AFTER_FALSE_S": 0.1,
                 "KEY_QUIET_S": 0.02, "RELEASE_WINDOW_S": (-0.25, 0.1)}.items():
        monkeypatch.setattr(S, k, v)
    for k, v in {"LEAD_S": 0.02, "SETTLE_S": 0.05, "EVAL_MARGIN_S": 0.02, "GUARD_TICK_MS": 5,
                 "READ_EVERY_S": 0.01}.items():
        monkeypatch.setattr(A, k, v)
    monkeypatch.setattr(A, "tone", lambda f, d, pk, c: T.tone(f, d, pk, c, ramp_s=0.01))
    return A


def drive_app(A, ctx, eng, listener, volume=lambda: 38):
    from PySide6 import QtCore

    def hook(s):
        def tick():
            now = time.perf_counter()
            if s.state in ("message", "end"):
                s.on_key("return", True, now)
                s.on_key("return", False, now)
                return
            listener(s, now, QtCore)
        tm = QtCore.QTimer(s)
        tm.timeout.connect(tick)
        tm.start(5)

    return A.run_screen(eng, ctx, lambda: {"readable": True, "muted": False,
                                           "output_volume": volume()},
                        fullscreen=False, hook=hook)


def ideal(true):
    def listener(s, now, QtCore):
        p = s.pres
        if p is None or p.get("sim"):
            return
        p["sim"] = True
        if not p["catch"] and p["level_db_hl"] >= true[s.test.ear]:
            on, off = p["t_on"], p["t_on"] + p["dur"]
            QtCore.QTimer.singleShot(max(0, int((on + 0.02 - now) * 1000)),
                                     lambda: s.on_key("space", True, time.perf_counter()))
            QtCore.QTimer.singleShot(max(0, int((off + 0.01 - now) * 1000)),
                                     lambda: s.on_key("space", False, time.perf_counter()))
    return listener


def test_screen_app_end_to_end(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    A = fast(monkeypatch)
    offsets = {c: {R.fkey(f): 1.0 for f in R.SCREEN_HZ} for c in R.CUPS}
    events, saved = [], {}
    ctx = A.Context("SIM", "right", offsets, {"output_volume": 38}, 7, events.append,
                    lambda d: saved.update(d))
    eng = FakeEngine()
    status = drive_app(A, ctx, eng, ideal({"left": 5.0, "right": 15.0}))
    assert status == "complete"
    fin = saved["plan"]["final_db_hl"]
    assert all(v == 15.0 for v in fin["right"].values())
    assert all(v == 5.0 for v in fin["left"].values())
    assert saved["n_false_responses"] == 0 and saved["n_catch_trials"] > 0
    for _, _, l, r in eng.played:                              # one cup only
        assert (l == 0) != (r == 0)
    tones = [e for e in events if e["event"] == "tone"]
    assert max(e["level_db_hl"] for e in tones) <= S.CAP_DB_HL
    assert any(e["event"] == "familiarisation" and e["ok"] for e in events)
    assert [e for e in events if e["event"] == "test"][0]["ear"] == "right"
    assert not any(e["event"] == "catch" and e["responded"] for e in events)


def test_screen_app_random_presser_is_caught(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    A = fast(monkeypatch)
    monkeypatch.setattr(S, "CATCH_PROB", 0.3)
    offsets = {c: {R.fkey(f): 1.0 for f in R.SCREEN_HZ} for c in R.CUPS}
    events, saved = [], {}
    ctx = A.Context("SIM", "left", offsets, {"output_volume": 38}, 3, events.append,
                    lambda d: saved.update(d))
    rng = np.random.default_rng(11)
    state = {"down": False, "t0": time.perf_counter()}

    def presser(s, now, QtCore):                    # about one press per 0.5 s, hears nothing
        if not state["down"] and rng.random() < 0.01:
            state["down"] = True
            s.on_key("space", True, now)
        elif state["down"] and rng.random() < 0.3:
            state["down"] = False
            s.on_key("space", False, now)
        if now - state["t0"] > 25 and s.state in ("run", "pause"):
            s.on_key("escape", True, now)
            s.on_key("q", True, now)
    status = drive_app(A, ctx, FakeEngine(), presser)
    assert status in ("complete", "stopped")
    assert saved["n_false_responses"] > 0
    assert any(e["event"] == "reinstruct" for e in events)
    flags = " ".join(saved["plan"]["flags"])
    assert "false responses" in flags


def test_press_before_onset_aborts_the_tone(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    A = fast(monkeypatch)
    monkeypatch.setattr(A, "LEAD_S", 0.15)
    offsets = {c: {R.fkey(f): 1.0 for f in R.SCREEN_HZ} for c in R.CUPS}
    events, saved = [], {}
    ctx = A.Context("SIM", "left", offsets, {"output_volume": 38}, 5, events.append,
                    lambda d: saved.update(d))
    eng = FakeEngine()
    done = {"n": 0}

    def listener(s, now, QtCore):
        p = s.pres
        if p is not None and not p.get("sim") and now < p["t_on"] - 0.05 and done["n"] < 1:
            p["sim"] = True
            done["n"] += 1
            s.on_key("space", True, now)
            s.on_key("space", False, now + 0.01)
        elif done["n"] >= 1:
            s.on_key("escape", True, now)
            s.on_key("q", True, now)
    status = drive_app(A, ctx, eng, listener)
    assert status == "stopped"
    assert eng.aborts >= 1
    assert any(e["event"] == "void" and e["why"] == "press before onset" for e in events)
    assert any(e["event"] == "false_response" and e["during"] == "before onset" for e in events)


def test_screen_app_volume_change_pauses(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    A = fast(monkeypatch)
    offsets = {c: {R.fkey(f): 1.0 for f in R.SCREEN_HZ} for c in R.CUPS}
    events, saved = [], {}
    ctx = A.Context("SIM", "left", offsets, {"output_volume": 38}, 1, events.append,
                    lambda d: saved.update(d))
    vol = {"v": 38}

    def listener(s, now, QtCore):
        if s.state == "run" and s.n_tones >= 2:
            vol["v"] = 50                                      # someone touches the volume
        elif s.state == "pause":
            s.on_key("q", True, now)
    status = drive_app(A, ctx, FakeEngine(), listener, volume=lambda: vol["v"])
    assert status == "stopped"
    assert any(e["event"] == "volume_changed" for e in events)
    assert saved["status"] == "stopped"
    scored = [e for e in events if e["event"] == "tone"]
    assert len(scored) <= 2                                    # nothing scored after the change


# ---- after the final review (2026-11-18) ------------------------------------------------------------------
def test_drift_compares_weighting_corrected_levels(lab):
    from audiometry import levels as L
    setup_rig(lab)
    n = len(R.ALL_HZ) * 2
    L.cmd_levels(args(frequencies=["all"], reseats=1), ask=scripted(ENV + ["68"] * n),
                 say=lambda s: None, root=lab)
    # the same 8-kHz output read on Z is 3.05 dB higher than on C: not a drift
    z = 68 + 3.05
    assert L.cmd_levels(args(frequencies=["8000"], reseats=1, weighting="Z"),
                        ask=scripted(ENV + [f"{z:.2f}"] * 2), say=lambda s: None, root=lab) == 0
    # a real 3-dB drop read on Z fails
    assert L.cmd_levels(args(frequencies=["8000"], reseats=1, weighting="Z"),
                        ask=scripted(ENV + [f"{z - 3:.2f}"] * 2), say=lambda s: None, root=lab) == 1


def _rig_with_sim(lab, label, volume, sim_reading, since):
    from audiometry import levels as L
    rig = R.Rig(label, volume, "External Headphones", since)
    if R.Rig.load(lab) is not None:
        old = R.Rig.load(lab)
        R.write_json_atomic(R.rig_dir(lab) / f"rig_until_{old.since[:10]}.json", old.__dict__)
    rig.save(lab)
    n = len(R.ALL_HZ) * 2
    L.cmd_levels(args(frequencies=["all"], reseats=1), ask=scripted(ENV + ["68"] * n),
                 say=lambda s: None, root=lab)
    L.cmd_levels(args(frequencies=["all"], reseats=1, coupler="ear-simulator", weighting="Z"),
                 ask=scripted(ENV + [str(sim_reading)] * n), say=lambda s: None, root=lab)
    return rig


def test_report_scores_each_screen_with_its_own_chain(lab):
    from audiometry.cli import thresholds_block
    from audiometry.report import cmd_report
    r1 = _rig_with_sim(lab, "HD280Pro-1", 38, 68.0, "2026-11-18T09:00:00-05:00")
    recs = R.level_records(lab, headphones="HD280Pro-1", volume=38)
    jig = R.offsets_for(recs, "jig", R.SCREEN_HZ, which="first")
    plan = {"final_db_hl": {e: {f"{f:g}": 10.0 for f in R.SCREEN_HZ} for e in R.CUPS},
            "final_status": {e: {f"{f:g}": "found" for f in R.SCREEN_HZ} for e in R.CUPS}}
    d = lab / "S01" / "screen_01"
    d.mkdir(parents=True)
    R.write_json_atomic(d / "screen.json", {"code": "S01", "status": "complete", "rig": r1.__dict__,
                                            "thresholds": thresholds_block(plan, jig)})
    _rig_with_sim(lab, "HD280Pro-2", 40, 80.0, "2026-11-25T09:00:00-05:00")   # a much louder chain
    out = []
    cmd_report(SimpleNamespace(csv=str(lab / "x.csv")), say=out.append, root=lab)
    text = "\n".join(out)
    assert "S01/screen_01 (HD280Pro-1 at 38)" in text and "PASS" in text
    import csv as _csv
    rows = list(_csv.DictReader(open(lab / "x.csv")))
    # chain 1: the jig read 68 on C, the simulator 68 on Z, so HL = 10 + C(f); chain 2 would add 12 dB
    for r in rows:
        want = 10.0 + R.weighting_db(float(r["freq_hz"]), "C")
        assert float(r["db_hl"]) == pytest.approx(want, abs=0.06), r


def test_provisional_rescore_uses_the_scoring_table(monkeypatch):
    from audiometry.cli import thresholds_block
    from audiometry.report import final_hl
    offs = {c: {R.fkey(f): 0.0 for f in R.SCREEN_HZ} for c in R.CUPS}
    plan = {"final_db_hl": {e: {f"{f:g}": 18.0 for f in R.SCREEN_HZ} for e in R.CUPS}}
    screen = {"thresholds": thresholds_block(plan, offs)}
    assert final_hl(screen, None, "hda200")["left"][500.0] == pytest.approx(18.0)
    full = {250: 18.0, 500: 6.8, 1000: 1.4, 2000: 1.9, 4000: 2.2, 8000: 17.5}   # test values only
    monkeypatch.setattr(S, "RETSPL_HD280PRO", full)
    assert R.scoring_table() == "hd280pro"
    assert final_hl(screen, None)["left"][500.0] == pytest.approx(18.0 + 11.0 - 6.8)


def test_report_survives_a_band_check_off_the_list(lab):
    from audiometry import levels as L
    from audiometry.report import cmd_report
    _rig_with_sim(lab, "HD280Pro-1", 38, 70.0, "2026-11-18T09:00:00-05:00")
    L.cmd_levels(args(coupler="ear-simulator", weighting="Z", signal="band", frequencies=["3000"],
                      reseats=1), ask=scripted(ENV + ["62", "62"]), say=lambda s: None, root=lab)
    out = []
    assert cmd_report(SimpleNamespace(csv=None), say=out.append, root=lab) == 0
    assert any("no check-tone offset" in x for x in out)


def test_screen_cli_refuses_an_existing_code_and_runs_with_fakes(lab, monkeypatch):
    from audiometry import cli as C
    from audiometry import levels as L
    setup_rig(lab)
    n = len(R.ALL_HZ) * 2
    L.cmd_levels(args(frequencies=["all"], reseats=1), ask=scripted(ENV + ["68"] * n),
                 say=lambda s: None, root=lab)
    L.cmd_daily(args(), ask=scripted(DAILY_OK), say=lambda s: None, root=lab)
    import tsync.audio, tsync.system
    from audiometry import screen_app
    monkeypatch.setattr(tsync.audio, "find_output", lambda m: {"name": "External Headphones", "index": 0})
    monkeypatch.setattr(tsync.audio, "AudioEngine", lambda *a, **k: SimpleNamespace(
        start=lambda: None, stop=lambda: None))
    monkeypatch.setattr(tsync.system, "set_output_volume",
                        lambda v: {"readable": True, "muted": False, "output_volume": v})
    monkeypatch.setattr(tsync.system, "KeepAwake", lambda: SimpleNamespace(start=lambda: None,
                                                                          stop=lambda: None))
    seen = {}

    def fake_run(engine, ctx, reader, fullscreen=True, hook=None):
        seen["ctx"] = ctx
        ctx.save({"status": "complete", "plan": {}})
        return "complete"
    monkeypatch.setattr(screen_app, "run_screen", fake_run)
    (lab / "L01" / "tpair_session_01").mkdir(parents=True)
    a = SimpleNamespace(code="L01", better_ear="same", windowed=True)
    assert C.cmd_screen(a, root=lab, ask=scripted(["n"])) == 2           # not the same person
    pre = ["n", "none", "n", "not done", "y", "y"]
    assert C.cmd_screen(SimpleNamespace(code="S01", better_ear="same", windowed=True), root=lab,
                        ask=scripted(pre)) == 0
    doc = json.loads((lab / "S01" / "screen_01" / "screen.json").read_text())
    assert doc["first_ear"] == "right" and doc["pre_test"]["ear_exam"] == "not done"
    assert doc["daily_check_today"] is True and seen["ctx"].offsets["left"][1000.0] == pytest.approx(
        68 - R.TEST_NOMINAL_DB, abs=1e-3)


def test_app_error_at_start_ends_cleanly(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    A = fast(monkeypatch)
    offsets = {c: {R.fkey(f): 1.0 for f in R.SCREEN_HZ} for c in R.CUPS}

    def broken_log(ev):
        raise OSError("disk full")
    ctx = A.Context("SIM", "left", offsets, {"output_volume": 38}, 1, broken_log, lambda d: None)
    status = drive_app(A, ctx, FakeEngine(), lambda s, now, QtCore: None)
    assert status == "error"


def test_app_pauses_after_three_audio_failures(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    A = fast(monkeypatch)
    offsets = {c: {R.fkey(f): 1.0 for f in R.SCREEN_HZ} for c in R.CUPS}
    events = []
    ctx = A.Context("SIM", "left", offsets, {"output_volume": 38}, 2, events.append, lambda d: None)

    class LateEngine(FakeEngine):
        def play_at(self, buf, t):
            pb = super().play_at(buf, t)
            pb.late = True
            return pb

    def listener(s, now, QtCore):
        if s.state == "pause":
            s.on_key("q", True, now)
    status = drive_app(A, ctx, LateEngine(), listener)
    assert status == "stopped"
    assert sum(e["event"] == "void" and e.get("why") == "audio" for e in events) == 3
    assert any(e["event"] == "pause" and "not playing correctly" in e["why"] for e in events)
