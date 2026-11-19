"""python -m audiometry {setup, levels, ambient, daily, screen, report}. See audiometry/README.md."""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Callable, Optional

from . import rig as R
from . import standards as S

CODE_RE = re.compile(r"[A-Za-z0-9_-]{1,24}")
RESERVED = {"rig", "checks"}


def _screen_folder(code: str, root=None) -> Path:
    base = Path(root or R.DATA) / code
    n = [int(m.group(1)) for p in base.glob("screen_*") if (m := re.fullmatch(r"screen_(\d+)", p.name))]
    return base / f"screen_{(max(n) + 1) if n else 1:02d}"


def thresholds_block(plan_summary: dict, offsets) -> dict:
    """{ear: {freq: {db_hl_provisional, nominal_db, status, ...}}}. The nominal value is what the
    ear-simulator offsets later turn into the reported hearing level."""
    out = {}
    statuses = plan_summary.get("final_status") or {}
    for ear, d in (plan_summary.get("final_db_hl") or {}).items():
        out[ear] = {}
        for f, hl in d.items():
            fv = float(f)
            off = offsets[ear][R.fkey(fv)]
            nominal = None if hl is None else round(R.nominal_from_hl(hl, fv, off), 2)
            out[ear][f] = {"db_hl_provisional": hl, "nominal_db": nominal,
                           "status": (statuses.get(ear) or {}).get(f),
                           "jig_offset_db": off, "presentation_retspl_db": R.retspl(fv)}
    return out


def daily_ok_today(rig: R.Rig, root=None) -> bool:
    """The last daily check today, for this rig, passed."""
    today = date.today().isoformat()
    days = [d for d in R.read_jsonl(R.rig_dir(root) / "daily.jsonl")
            if str(d.get("time", ""))[:10] == today
            and (d.get("rig") or {}).get("headphones") == rig.headphones
            and (d.get("rig") or {}).get("volume") == rig.volume]
    return bool(days) and bool(days[-1].get("ok"))


def pre_test(ask: Callable[[str], str] = input, say=print) -> Optional[dict]:
    """BSA 2018 5.1: the questions the tester asks before the test. None = do not test today."""
    say("Before the screen (BSA 2018 5.1). Ask the listener:")
    out = {}
    for key, q in S.PRE_TEST_QUESTIONS:
        out[key] = ask(f"  {q} ").strip()
    if out["noise_24h"].lower().startswith("y"):
        out["noise_24h_details"] = ask("  What noise, how long, how long ago? ").strip()
        say("  ⚠ BSA 5.1: recent loud noise can cause a temporary loss; re-test another day "
            "if the results may have been affected.")
    if out["cold_or_ear_pain"].lower().startswith("y"):
        say("  ⚠ a cold or ear problem can raise thresholds.")
    if not out["cleared"].lower().startswith("y"):
        say("  ✗ clear glasses, earrings, headwear and hair from under the cups first")
        return None
    go = ask("  Go on with the screen today? [y/n] ").strip().lower()
    return out if go.startswith("y") else None


def cmd_screen(a, root=None, ask: Callable[[str], str] = input) -> int:
    from tsync.audio import AudioEngine, find_output
    from tsync.system import KeepAwake, set_output_volume, volume_settings
    from .levels import _need_rig
    from .screen_app import Context, run_screen

    if not CODE_RE.fullmatch(a.code) or a.code.lower() in RESERVED:
        print("✗ --code: 1-24 letters, digits, - or _ (and not 'rig' or 'checks')")
        return 2
    rig = _need_rig(root)
    recs = R.level_records(root, headphones=rig.headphones, volume=rig.volume)
    offsets = R.offsets_for(recs, "jig", R.SCREEN_HZ, which="first")
    if offsets is None:
        print("✗ no jig baseline at every screening frequency: run "
              "`python -m audiometry levels --frequencies all` first")
        return 2
    bad = R.range_problems(offsets)
    if bad:
        print("✗ the screen's range does not fit this rig:\n  " + "\n  ".join(bad))
        return 2
    daily_ok = daily_ok_today(rig, root)
    if not daily_ok:
        print("✗ the last daily check today did not pass (or there is none): run "
              "`python -m audiometry daily` first")
        return 2
    other = [p.name for p in (Path(root or R.DATA) / a.code).glob("*")
             if not p.name.startswith("screen_")]
    if other and not ask(f"data/{a.code} already holds other sessions ({', '.join(sorted(other)[:4])}"
                         f"{'...' if len(other) > 4 else ''}). Is this the same person? [y/n] "
                         ).strip().lower().startswith("y"):
        print("pick a new code for a new listener")
        return 2
    answers = pre_test(ask)
    if answers is None:
        print("not screened")
        return 1
    first = a.better_ear if a.better_ear in ("left", "right") else "right"
    dev = find_output(R.DEVICE_MATCH)
    if dev["name"] != rig.device:
        print(f"✗ the output is '{dev['name']}', the rig's is '{rig.device}'")
        return 2
    back = set_output_volume(rig.volume)
    if not back.get("readable") or back.get("muted") or back.get("output_volume") != rig.volume:
        print(f"✗ the volume reads {back.get('output_volume')} (muted {back.get('muted')}), "
              f"the rig's is {rig.volume}")
        return 2
    folder = _screen_folder(a.code, root)
    folder.mkdir(parents=True, exist_ok=False)
    seed = int.from_bytes(os.urandom(4), "little")
    used = {f"{c} {f:g}": r["time"] for (c, f), r in R.pick_records(recs, "jig", "first").items()}
    header = {
        "code": a.code, "task": "audiometry screen", "created": R.now_iso(), "seed": seed,
        "better_ear_reported": a.better_ear, "first_ear": first, "pre_test": answers,
        "daily_check_today": daily_ok, "rig": rig.__dict__, "audio_device": dev["name"],
        "volume_at_start": back,
        "offsets_provisional": {c: {f"{f:g}": v for f, v in d.items()} for c, d in offsets.items()},
        "offsets_from_jig_records": used,
        "presentation_scale": "provisional hearing level: jig baseline offsets and the HDA 200 "
                              "RETSPLs of ISO 389-8:2004; re-scored by `report`",
        "procedure": "BSA 2018 6.1-6.7 (10-down 5-up, 2 of 2-4 ascending), order of 6.4; automatic "
                     "scoring, catch trials and false-response rules are ours (README)",
        "environment": R.environment(),
    }

    def log(ev):
        R.append_jsonl(folder / "events.jsonl", ev)

    def save(state):
        doc = dict(header)
        doc.update(state)
        doc["updated"] = R.now_iso()
        doc["thresholds"] = thresholds_block(state.get("plan") or {}, offsets)
        R.write_json_atomic(folder / "screen.json", doc)

    save({"status": "started"})
    engine = AudioEngine(dev["index"], R.SAMPLE_RATE, R.CHANNELS, R.BLOCKSIZE, R.LATENCY)
    engine.start()
    awake = KeepAwake()
    awake.start()
    ctx = Context(a.code, first, offsets, back, seed, log, save)
    try:
        status = run_screen(engine, ctx, volume_settings, fullscreen=not a.windowed)
    finally:
        awake.stop()
        engine.stop()
    print(f"{status}: {folder}")
    return 0 if status == "complete" else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m audiometry", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="fix the rig: one headphone label, one system volume")
    s.add_argument("--headphones", required=True, help="the label written on the pair")
    s.add_argument("--volume", type=int, required=True)
    s.add_argument("--note", default="")
    s.add_argument("--new-rig", action="store_true", help="replace an existing rig (splits the data)")
    s.add_argument("--no-audio", action="store_true", help=argparse.SUPPRESS)

    s = sub.add_parser("levels", help="readings on the jig or the ear simulator")
    s.add_argument("--coupler", choices=R.COUPLERS, default="jig")
    s.add_argument("--signal", choices=R.SIGNALS, default="tone",
                   help="tone (the check tone), band (a tcmr noise band at 60 nominal), bands-sum")
    s.add_argument("--frequencies", nargs="+", default=["daily"],
                   help="daily (1 kHz), all (every frequency used), or numbers in Hz")
    s.add_argument("--nominal", type=float, default=None,
                   help="tone level on the nominal scale (linearity); default the check tone, 67.96")
    s.add_argument("--cups", nargs="+", choices=R.CUPS, default=list(R.CUPS))
    s.add_argument("--reseats", type=int, default=None,
                   help="seatings per cup (default 3 on the jig, 5 on the ear simulator)")
    s.add_argument("--trial", action="store_true", help="choosing the volume: never a baseline")
    s.add_argument("--volume", type=int, default=None)
    s.add_argument("--headphones", default=None)
    s.add_argument("--weighting", choices=("C", "Z"), default=None,
                   help="default C on the jig (the SL-100), Z on the ear simulator")
    s.add_argument("--meter", default="")
    s.add_argument("--operator", default="")
    s.add_argument("--note", default="")

    s = sub.add_parser("ambient", help="booth noise with the sound level meter")
    s.add_argument("--meter", default="")
    s.add_argument("--operator", default="")
    s.add_argument("--note", default="")

    s = sub.add_parser("daily", help="BSA Stage A checks and the jig level, before any listener")
    s.add_argument("--operator", default="")
    s.add_argument("--meter", default="")
    s.add_argument("--reseats", type=int, default=3)

    s = sub.add_parser("screen", help="the hearing screen (full screen, space bar)")
    s.add_argument("--code", required=True)
    s.add_argument("--better-ear", required=True, choices=("left", "right", "same"),
                   help="by the listener's own account; 'same' starts with the right ear")
    s.add_argument("--windowed", action="store_true")

    s = sub.add_parser("report", help="rig history, screens, tcmr thresholds in dB SPL")
    s.add_argument("--csv", default=None)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if getattr(a, "reseats", None) is not None and a.reseats < 1:
        print("✗ --reseats must be at least 1")
        return 2
    from . import levels as L
    from .report import cmd_report
    fn = {"setup": L.cmd_setup, "levels": L.cmd_levels, "ambient": L.cmd_ambient,
          "daily": L.cmd_daily, "screen": cmd_screen, "report": cmd_report}[a.cmd]
    return fn(a)


if __name__ == "__main__":
    sys.exit(main())
