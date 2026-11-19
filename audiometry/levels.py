"""Terminal commands for the rig: setup, levels (jig or ear simulator), daily, ambient.

Every reading is typed by the person holding the meter and appended to data/rig/*.jsonl, failed
checks included (marked passed: false, and never used as a baseline or an offset). Nothing is
saved if the system volume changed during a measurement.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from . import rig as R
from . import standards as S
from .tones import band_noise, tone

Ask = Callable[[str], str]
Say = Callable[[str], None]
LISTEN_GAP_S = 0.8                         # silence between the listening-check tones
SIGNAL_S = 60.0                            # each check signal plays this long
JIG_TEXT = (
    "Jig: a rigid block 145 mm wide (the ISO 389-8 cup separation) with the meter's microphone "
    "flush in one face (foam ball off, gap sealed with fresh putty). The headphones clamp the block "
    "with their own headband at the marked slider position, the cup centred on its outline and at "
    "its marked rotation. Meter: C weighting, slow, Lo range. Never touch the trim pot.")
SIM_TEXT = (
    "Ear simulator: IEC 60318-1 with the circumaural (flat-plate) adapter, the cup at the HD 280 "
    "Pro's own measured headband force (not the HDA 200's 10 N), room 21-25 °C. Class 1 meter, "
    "Z weighting, checked with its calibrator before and after. Refit the cup between readings. "
    "Write the simulator, adapter, meter and calibrator with serials in --meter.")


# ---- the audio side ---------------------------------------------------------------------------------
class Output:
    """The headphone output at a fixed system volume, with the shared audio engine."""

    def __init__(self, volume: int, say: Say = print):
        from tsync.audio import AudioEngine, find_output
        from tsync.system import set_output_volume
        self.say = say
        self.device = find_output(R.DEVICE_MATCH)
        back = set_output_volume(int(volume))
        if not back.get("readable") or back.get("output_volume") != int(volume) or back.get("muted"):
            raise SystemExit(f"✗ the system volume reads {back.get('output_volume')} "
                             f"(muted {back.get('muted')}) after setting {volume}; nothing measured")
        self.volume = back
        self.engine = AudioEngine(self.device["index"], R.SAMPLE_RATE, R.CHANNELS, R.BLOCKSIZE,
                                  R.LATENCY)
        self.engine.start()
        self.pb = None

    def volume_unchanged(self) -> bool:
        from tsync.system import volume_settings
        v = volume_settings()
        return (bool(v.get("readable")) and not v.get("muted")
                and v.get("output_volume") == self.volume.get("output_volume"))

    def play_buffer(self, buf) -> None:
        self.stop()
        self.pb = self.engine.play_at(buf, time.perf_counter() + 0.15)

    def play(self, freq: float, cup: str, seconds: float, peak: float = R.TEST_AMP) -> None:
        self.play_buffer(tone(freq, seconds, peak, cup))

    def playing(self) -> bool:
        return self.pb is not None and not self.pb.finished

    def stop(self) -> None:
        if self.pb is not None and not self.pb.finished:
            self.engine.abort(self.pb)
            t_end = time.perf_counter() + 1.0
            while not self.pb.finished and time.perf_counter() < t_end:
                time.sleep(0.01)
        self.pb = None

    def wait(self, timeout_s: float = 10.0) -> None:
        t_end = time.perf_counter() + timeout_s
        while self.pb is not None and not self.pb.finished and time.perf_counter() < t_end:
            time.sleep(0.01)

    def close(self) -> None:
        self.stop()
        self.engine.stop()


def signal_buffer(signal: str, freq: float, cup: str, nominal: float, seconds: float = SIGNAL_S,
                  seed: int = 0):
    if signal == "tone":
        return tone(freq, seconds, R.peak_for(nominal), cup)
    rng = np.random.default_rng(seed)
    freqs = R.TCMR_BANDS_HZ if signal == "bands-sum" else (freq,)
    return band_noise(freqs, seconds, 60.0, cup, rng)


# ---- measuring ---------------------------------------------------------------------------------------
def measure_one(out: Output, buf, label: str, reseats: int, ask: Ask, say: Say) -> List[float]:
    say(f"\n{label}. The signal plays for {SIGNAL_S:g} s in this cup only; Enter on its own plays "
        f"it again.")
    out.play_buffer(buf)
    got: List[float] = []
    limit = 2 * reseats
    while len(got) < reseats:
        k = len(got) + 1
        what = "Seat the cup" if k == 1 else "Lift the cup off and seat it again"
        txt = ask(f"  {what} ({k} of {reseats}); when the meter settles, type the reading in dB: ")
        if not txt.strip():
            out.play_buffer(buf)
            continue
        if not out.playing():
            say("  the signal had stopped, so that reading is not taken: Enter plays it again")
            continue
        v = R.parse_reading(txt)
        if v is None:
            say("  not a reading between 20 and 130 dB; type it again")
            continue
        got.append(v)
        if len(got) == reseats and max(got) - min(got) > S.RESEAT_RANGE_DB and reseats < limit:
            say(f"  ⚠ the readings spread {max(got) - min(got):.1f} dB (more than "
                f"{S.RESEAT_RANGE_DB:g}): check the seal and the seating, then add readings")
            if ask("  add readings? [y/n] ").strip().lower().startswith("y"):
                reseats = min(limit, reseats + 2)
    out.stop()
    return got


def measure(out: Output, freqs: Sequence[float], cups: Sequence[str], reseats: int, coupler: str,
            fields: dict, ask: Ask, say: Say, signal: str = "tone",
            nominal: float = R.TEST_NOMINAL_DB) -> Optional[List[R.LevelRecord]]:
    recs: List[R.LevelRecord] = []
    for cup in cups:
        for i, f in enumerate(freqs):
            buf = signal_buffer(signal, f, cup, nominal, seed=i)
            what = {"tone": f"{f:g}-Hz tone at {nominal:.2f} nominal dB",
                    "band": f"{f:g}-Hz noise band at 60 nominal dB (read Leq)",
                    "bands-sum": "the five tcmr bands at 60 nominal dB each (read Leq)"}[signal]
            got = measure_one(out, buf, f"{cup} cup: {what}", reseats, ask, say)
            if not out.volume_unchanged():
                say("✗ the system volume changed or was muted during the measurement: "
                    "nothing is saved")
                return None
            recs.append(R.LevelRecord(coupler=coupler, cup=cup,
                                      freq_hz=0.0 if signal == "bands-sum" else R.fkey(f),
                                      readings_db=got, signal=signal,
                                      test_nominal_db=nominal if signal == "tone" else 60.0,
                                      **fields))
    return recs


def _freqs(spec: Sequence[str]) -> List[float]:
    if list(spec) == ["daily"]:
        return list(R.DAILY_HZ)
    if list(spec) == ["all"]:
        return list(R.ALL_HZ)
    return [float(x) for x in spec]


def _need_rig(root=None) -> R.Rig:
    rig = R.Rig.load(root)
    if rig is None:
        raise SystemExit("✗ no rig yet: choose the volume with `levels --trial`, then run "
                         "`python -m audiometry setup --headphones LABEL --volume N`")
    return rig


def _optional_number(ask: Ask, q: str, lo: float, hi: float) -> Optional[float]:
    while True:
        t = ask(q).strip()
        if not t:
            return None
        v = R.parse_reading(t, lo, hi)
        if v is not None:
            return v


def _judge(recs: List[R.LevelRecord], previous: List[dict], say: Say) -> bool:
    """Jig check: each record against the first jig record of its cup and frequency. Marks every
    record passed or not; True if all passed."""
    ok = True
    say(f"\n{'cup':6}{'Hz':>9}{'mean':>8}{'spread':>8}{'offset':>8}{'vs first':>10}")
    for r in recs:
        b = R.baseline(previous, r.cup, r.freq_hz)
        d = r.as_dict()
        line = f"{r.cup:6}{r.freq_hz:>9g}{d['mean_db']:>8.1f}{d['range_db']:>8.1f}{d['offset_db']:>+8.1f}"
        if b is None:
            r.passed = True
            line += f"{'(first)':>10}"
        else:
            dv = R.unweighted(r.as_dict()) - R.unweighted(b)     # C and Z readings compare
            r.passed = abs(dv) <= S.DRIFT_LIMIT_DB
            mark = "✓" if abs(dv) <= S.DRIFT_WARN_DB else ("⚠" if r.passed else "✗")
            line += f"{dv:>+9.1f} {mark}"
        ok = ok and bool(r.passed)
        say(line)
    return ok


def _save(recs: List[R.LevelRecord], root) -> None:
    env = R.environment()
    for r in recs:
        d = r.as_dict()
        d["environment"] = env
        R.append_jsonl(R.levels_log(root), d)


# ---- commands -------------------------------------------------------------------------------------------
def cmd_setup(a, ask: Ask = input, say: Say = print, root=None) -> int:
    old = R.Rig.load(root)
    if old is not None and old.headphones == a.headphones and old.volume == a.volume:
        say(f"✓ the rig is already {old.headphones} at volume {old.volume} (since {old.since}); "
            f"nothing changed")
        return 0
    if old is not None:
        if not a.new_rig:
            say(f"✗ the rig is already {old.headphones} at volume {old.volume} (since "
                f"{old.since}). Changing it splits the data: only with --new-rig, and only before "
                f"the first listener or with a new ear-simulator measurement.")
            return 2
        stamp = old.since.replace(":", "").replace("-", "")[:15] or "old"
        R.write_json_atomic(R.rig_dir(root) / f"rig_until_{stamp}.json", old.__dict__)
    if not a.no_audio:
        from tsync.audio import find_output
        from tsync.system import set_output_volume
        dev = find_output(R.DEVICE_MATCH)
        back = set_output_volume(a.volume)
        if back.get("output_volume") != a.volume:
            say(f"✗ macOS reads the volume back as {back.get('output_volume')}, not "
                f"{a.volume}: pick a volume that reads back exactly")
            return 2
        device = dev["name"]
    else:
        device = "External Headphones"
    rig = R.Rig(a.headphones, int(a.volume), device, R.now_iso(), a.note or "")
    rig.save(root)
    say(f"✓ rig: {rig.headphones} on '{rig.device}' at system volume {rig.volume}. Every "
        f"session (tcmr, tinc, the screen) runs at --volume {rig.volume} --uncalibrated.")
    return 0


def cmd_levels(a, ask: Ask = input, say: Say = print, root=None) -> int:
    coupler = a.coupler
    reseats = a.reseats or (5 if coupler == "ear-simulator" else 3)
    weighting = a.weighting or ("Z" if coupler == "ear-simulator" else "C")
    signal = getattr(a, "signal", "tone") or "tone"
    nominal = R.TEST_NOMINAL_DB if getattr(a, "nominal", None) is None else float(a.nominal)
    if signal == "tone" and (nominal > R.NOMINAL_MAX_DB or R.peak_for(nominal) > R.MAX_PEAK):
        say(f"✗ --nominal {nominal:g} is above the {R.NOMINAL_MAX_DB:g}-dB limit")
        return 2
    freqs = [0.0] if signal == "bands-sum" else _freqs(a.frequencies)
    cups = list(a.cups)
    if a.trial:
        if a.volume is None:
            say("✗ --trial needs --volume N (the volume to try)")
            return 2
        fields = dict(volume=int(a.volume), headphones=a.headphones or "?", device="?", trial=True)
    else:
        rig = _need_rig(root)
        if a.volume is not None and a.volume != rig.volume:
            say(f"✗ the rig's volume is {rig.volume}; --volume {a.volume} would measure a "
                f"different chain")
            return 2
        fields = dict(volume=rig.volume, headphones=rig.headphones, device=rig.device)
    say(JIG_TEXT if coupler == "jig" else SIM_TEXT)
    fields.update(weighting=weighting, meter=a.meter or "", operator=a.operator or "",
                  note=a.note or "")
    fields["temperature_c"] = _optional_number(ask, "Room temperature in °C (Enter to skip): ",
                                               5, 40)
    fields["humidity_pct"] = _optional_number(ask, "Relative humidity in % (Enter to skip): ", 1, 100)
    out = Output(fields["volume"], say)
    try:
        if fields["device"] == "?":
            fields["device"] = out.device["name"]
        if out.device["name"] != fields["device"]:
            say(f"✗ the output is '{out.device['name']}', the rig's is '{fields['device']}'")
            return 2
        previous = R.level_records(root, headphones=fields["headphones"], volume=fields["volume"])
        recs = measure(out, freqs, cups, reseats, coupler, fields, ask, say, signal, nominal)
    finally:
        out.close()
    if recs is None:
        return 1
    standard = signal == "tone" and abs(nominal - R.TEST_NOMINAL_DB) < 0.01
    ok = True
    if coupler == "jig" and standard and not a.trial:
        ok = _judge(recs, previous, say)
    else:
        for r in recs:
            d = r.as_dict()
            say(f"  {r.cup:5} {r.freq_hz:>8g} Hz: mean {d['mean_db']:.1f} dB, spread "
                f"{d['range_db']:.1f}" + ("" if d["offset_db"] is None else
                                          f", offset {d['offset_db']:+.2f} dB"))
    _save(recs, root)
    if a.trial:
        r1 = [r for r in recs if r.freq_hz == 1000.0]
        if r1:
            say(f"\nAim: the 1-kHz check tone at {R.TEST_NOMINAL_DB:.1f} dB on the jig. This try "
                f"read {r1[0].mean_db:.1f} dB at volume {a.volume}.")
        return 0
    if not ok:
        say(f"\n✗ drift above {S.DRIFT_LIMIT_DB:g} dB (saved, marked failed). Reseat and measure "
            f"again; if it stays, find what changed (volume, cable, cushion, jig) before any "
            f"listener. A change that cannot be undone starts a new chain: measure it on the ear "
            f"simulator again.")
        return 1
    say("\n✓ saved")
    return 0


def cmd_ambient(a, ask: Ask = input, say: Say = print, root=None) -> int:
    say("Booth closed, everything running as in a session, the meter where the listener's head "
        "will be, slow. Below the meter's range (its Lo range starts at 30 dB) type <30.")

    def read(w):
        while True:
            t = ask(f"  {w}-weighted reading (dB): ").strip()
            if t.replace(" ", "").lower() in ("<30", "lo"):
                return "<30"
            v = R.parse_reading(t, 20.0, 120.0)
            if v is not None:
                return v
            say("  not a reading; again")

    rec = {"time": R.now_iso(), "dBA": read("A"), "dBC": read("C"), "meter": a.meter or "",
           "operator": a.operator or "", "note": a.note or ""}
    A = rec["dBA"]
    rec["ok"] = A == "<30" or float(A) <= S.AMBIENT_MAX_DBA
    R.append_jsonl(R.rig_dir(root) / "ambient.jsonl", rec)
    say(("✓" if rec["ok"] else "✗") + f" {A} dB(A) (BSA 2018 4.2: at most "
        f"{S.AMBIENT_MAX_DBA:g} dB(A)); saved")
    return 0 if rec["ok"] else 1


def _yes(ask: Ask, q: str) -> bool:
    while True:
        t = ask(q + " [y/n] ").strip().lower()
        if t[:1] in ("y", "n"):
            return t[:1] == "y"


def listening_check(out: Output, offsets: Dict[str, Dict[float, float]], level_hl: float,
                    ask: Ask, say: Say, question: str) -> Dict[str, bool]:
    res = {}
    for cup in R.CUPS:
        ask(f"  Press Enter, then listen: {len(R.SCREEN_HZ)} tones in the {cup} cup at "
            f"{level_hl:g} dB HL.")
        for f in S.TEST_ORDER_HZ:
            nominal = R.nominal_from_hl(level_hl, f, offsets[cup][R.fkey(f)])
            out.play(float(f), cup, 1.0, R.peak_for(nominal))
            out.wait()
            time.sleep(LISTEN_GAP_S)
        res[cup] = _yes(ask, f"  {question} ({cup} cup)")
    return res


def cmd_daily(a, ask: Ask = input, say: Say = print, root=None) -> int:
    rig = _need_rig(root)
    recs_all = R.level_records(root, headphones=rig.headphones, volume=rig.volume)
    offsets = R.offsets_for(recs_all, "jig", R.SCREEN_HZ, which="first")
    if offsets is None:
        say("✗ no jig baseline at every screening frequency yet: run "
            "`python -m audiometry levels --frequencies all` once first")
        return 2
    bad = R.range_problems(offsets, max(S.HIGH_CHECK_DB_HL, S.CAP_DB_HL))
    if bad:
        say("✗ the baseline puts the listening check out of range:\n  " + "\n  ".join(bad))
        return 2
    full = [r for r in recs_all if r.get("coupler") == "jig" and R.usable(r)
            and R.fkey(r["freq_hz"]) == R.fkey(250.0)]
    if full:
        last = R.parse_time(full[-1]["time"])
        if last is not None and (datetime.now().astimezone() - last).days >= S.WEEKLY_FULL_JIG_DAYS:
            say("⚠ the jig has not been read at every frequency for a week: run "
                "`python -m audiometry levels --frequencies all` today as well")
    say("Daily check (BSA 2018 11.1, Stage A, plus the level on the jig).")
    answers = {k: _yes(ask, "  " + text.format(headphones=rig.headphones) + "?")
               for k, text in S.DAILY_CHECKLIST}
    record = {"time": R.now_iso(), "checklist": answers, "operator": a.operator or "",
              "rig": rig.__dict__}
    if not all(answers.values()):
        say("✗ fix the items answered 'n' first; nothing else is checked")
        R.append_jsonl(R.rig_dir(root) / "daily.jsonl", {**record, "ok": False})
        return 1
    cal = _optional_number(
        ask, "Sound calibrator on the SL-100 (94 dB, 1 kHz): the reading, or Enter if none: ", 80, 110)
    record["calibrator_reading_db"] = cal
    if cal is not None:
        first = [d["calibrator_reading_db"] for d in R.read_jsonl(R.rig_dir(root) / "daily.jsonl")
                 if d.get("calibrator_reading_db") is not None]
        if first:
            record["calibrator_change_db"] = round(cal - first[0], 2)
            if abs(cal - first[0]) > S.DRIFT_WARN_DB:
                say(f"⚠ the meter reads the calibrator {cal - first[0]:+.1f} dB away from its first "
                    f"reading: the meter itself has drifted; the jig readings today carry that error")
    temp = _optional_number(ask, "Room temperature in °C (Enter to skip): ", 5, 40)
    hum = _optional_number(ask, "Relative humidity in % (Enter to skip): ", 1, 100)
    record.update(temperature_c=temp, humidity_pct=hum)
    say(JIG_TEXT)
    fields = dict(volume=rig.volume, headphones=rig.headphones, device=rig.device, weighting="C",
                  meter=a.meter or "", operator=a.operator or "", note="daily",
                  temperature_c=temp, humidity_pct=hum)
    out = Output(rig.volume, say)
    ok = False
    try:
        if out.device["name"] != rig.device:
            say(f"✗ the output is '{out.device['name']}', the rig's is '{rig.device}'")
            return 2
        recs = measure(out, R.DAILY_HZ, R.CUPS, a.reseats, "jig", fields, ask, say)
        if recs is None:
            record["why"] = "the volume changed during the jig reading"
            return 1
        level_ok = _judge(recs, recs_all, say)
        _save(recs, root)                          # kept even if a later step fails
        record["drift_db"] = {r.cup: round(R.unweighted(r.as_dict())
                                           - R.unweighted(R.baseline(recs_all, r.cup, r.freq_hz)), 2)
                              for r in recs if R.baseline(recs_all, r.cup, r.freq_hz)}
        say("\nNow wear the headphones yourself (left cup on the left ear).")
        low = listening_check(out, offsets, S.LOW_CHECK_DB_HL, ask, say,
                              "Heard every tone, each just audible, with no hum, hiss or fan noise")
        high = listening_check(out, offsets, S.HIGH_CHECK_DB_HL, ask, say,
                               "Every tone clean: no distortion, crackle or click")
        nominal = R.nominal_from_hl(S.CABLE_CHECK_DB_HL, 1000.0, offsets["left"][1000.0])
        ask(f"  Press Enter: a 1-kHz tone plays for 10 s in both cups in turn; gently move the "
            f"cable along its length and at the plug.")
        for cup in R.CUPS:
            out.play(1000.0, cup, 10.0, R.peak_for(min(nominal, R.NOMINAL_MAX_DB)))
            out.wait(15.0)
        cable = _yes(ask, "  The tone stayed steady: no crackle or dropout")
        same_volume = out.volume_unchanged()
        ok = level_ok and all(low.values()) and all(high.values()) and cable and same_volume
        record.update(low_check=low, high_check=high, cable_check=cable,
                      volume_unchanged=same_volume, jig_ok=level_ok)
    finally:
        out.close()
        R.append_jsonl(R.rig_dir(root) / "daily.jsonl", {**record, "ok": ok})
    if ok:
        say("✓ daily check passed and saved")
        return 0
    say("✗ daily check failed (saved as failed): do not test listeners until it passes")
    return 1
