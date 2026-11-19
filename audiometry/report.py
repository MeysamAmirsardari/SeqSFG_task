"""What the rig logs and the screens say, and the conversion of the sessions once the ear-simulator
offsets exist. Every screen and every tcmr session is scored with the chain (pair + volume) that was
in use when it was recorded, never with a later one.

  python -m audiometry report            # everything
  python -m audiometry report --csv out.csv
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

from . import procedure as P
from . import rig as R
from . import standards as S

TCMR_BAND_NOMINAL = 60.0      # tcmr_exp1: every band at 60 nominal, the tone at 1 kHz


def screen_files(root=None) -> List[Path]:
    base = Path(root or R.DATA)
    return sorted(base.glob("*/screen_*/screen.json"))


def chain_sim(rig_like: dict, root=None) -> Optional[Dict[str, Dict[float, float]]]:
    """Ear-simulator offsets of one chain (its headphone label and volume), or None."""
    recs = R.level_records(root, headphones=rig_like.get("headphones"), volume=rig_like.get("volume"))
    return R.offsets_for(recs, "ear-simulator", R.ALL_HZ)


def final_hl(screen: dict, sim: Optional[Dict[str, Dict[float, float]]],
             table: Optional[str] = None) -> Dict[str, Dict[float, Optional[float]]]:
    """Hearing levels from the stored nominal thresholds, with the scoring reference table: with the
    chain's ear-simulator offsets if given, else with the jig offsets the screen stored
    (provisional)."""
    table = table or R.scoring_table()
    out: Dict[str, Dict[float, Optional[float]]] = {}
    for ear, d in screen.get("thresholds", {}).items():
        out[ear] = {}
        for f, v in d.items():
            fv = float(f)
            if v.get("nominal_db") is None:
                out[ear][fv] = None
            elif sim is not None:
                out[ear][fv] = round(R.hl_from_nominal(v["nominal_db"], fv, sim[ear][R.fkey(fv)],
                                                       table), 1)
            elif v.get("jig_offset_db") is not None:
                out[ear][fv] = round(R.hl_from_nominal(v["nominal_db"], fv, v["jig_offset_db"],
                                                       table), 1)
            else:
                out[ear][fv] = v.get("db_hl_provisional")
    return out


def verdict(hl: Dict[str, Dict[float, Optional[float]]], screen: Optional[dict] = None) -> str:
    """pass: every octave 250-8000 Hz in both ears at or below the criterion. fail: any above it,
    or no response at the top of the range. incomplete: anything missing or unreliable."""
    statuses = {e: {float(f): (v or {}).get("status") for f, v in d.items()}
                for e, d in ((screen or {}).get("thresholds") or {}).items()}
    worse = missing = False
    for ear in R.CUPS:
        for f in R.SCREEN_HZ:
            v = hl.get(ear, {}).get(f)
            st = statuses.get(ear, {}).get(f)
            if v is None:
                if st in P.FAIL_STATUSES:
                    worse = True
                else:
                    missing = True
            elif v > S.PASS_DB_HL:
                worse = True
    if worse:
        return "fail"
    return "incomplete" if missing else "pass"


def _sessions(root, pattern: str) -> List[tuple]:
    base = Path(root or R.DATA)
    out = []
    for p in sorted(base.glob(pattern)):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append((p.parent, s, R.parse_time(s.get("created"))))
    return out


def session_checks(history: List[R.Rig], root=None) -> List[str]:
    """tcmr and tinc sessions recorded under a rig that do not match that rig."""
    base = Path(root or R.DATA)
    out = []
    for folder, s, created in _sessions(root, "*/*_session_*/session.json"):
        rig = R.rig_at(history, created)
        if rig is None:
            continue
        vol = (s.get("volume_at_start") or {}).get("output_volume")
        dev = s.get("audio_device")
        dev_name = dev.get("name") if isinstance(dev, dict) else dev
        problems = []
        if vol != rig.volume:
            problems.append(f"volume {vol} (rig {rig.volume})")
        if s.get("levels_calibrated") or s.get("calibration"):
            problems.append("calibrated levels (the rig plan uses --uncalibrated)")
        if dev_name and rig.device not in str(dev_name):
            problems.append(f"device {dev_name}")
        if problems:
            out.append(f"{folder.relative_to(base)}: " + "; ".join(problems))
    return out


def _sd(xs: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _rig_section(rig: R.Rig, root, say) -> Optional[Dict[str, Dict[float, float]]]:
    say(f"\nRig {rig.headphones} at system volume {rig.volume} on '{rig.device}', since {rig.since}")
    recs = R.level_records(root, headphones=rig.headphones, volume=rig.volume)
    for cup in R.CUPS:
        js = [r for r in recs if r["coupler"] == "jig" and R.usable(r) and r["cup"] == cup
              and R.fkey(r["freq_hz"]) == 1000.0]
        failed = [r for r in recs if r["coupler"] == "jig" and r.get("passed") is False and r["cup"] == cup]
        if js:
            lv = [R.unweighted(r) for r in js]
            dev = [x - lv[0] for x in lv]
            sd = _sd(lv)
            say(f"  Jig, {cup} cup, 1 kHz: {len(js)} passing checks, largest drift "
                f"{max(abs(d) for d in dev):.1f} dB, SD {'-' if sd is None else f'{sd:.2f}'} dB; "
                f"{sum(abs(d) > S.DRIFT_WARN_DB for d in dev)} above {S.DRIFT_WARN_DB:g} dB; "
                f"{len(failed)} failed checks")
    daily = [d for d in R.read_jsonl(R.rig_dir(root) / "daily.jsonl")
             if (d.get("rig") or {}).get("headphones") == rig.headphones
             and (d.get("rig") or {}).get("volume") == rig.volume]
    if daily:
        cal = [d["calibrator_reading_db"] for d in daily if d.get("calibrator_reading_db") is not None]
        say(f"  Daily checks: {sum(1 for d in daily if d.get('ok'))} passed, "
            f"{sum(1 for d in daily if not d.get('ok'))} failed; last {daily[-1]['time'][:16]}"
            + (f"; calibrator on the SL-100 {min(cal):.1f} to {max(cal):.1f} dB over {len(cal)} days"
               if cal else "; no calibrator readings"))
    sim = R.offsets_for(recs, "ear-simulator", R.ALL_HZ)
    if sim is None:
        say("  Ear simulator: not measured yet for this chain; its levels are provisional (jig).")
        return None
    say("  Ear-simulator offsets (dB; level in dB SPL = nominal + offset):")
    for f in R.ALL_HZ:
        l, r = sim["left"][R.fkey(f)], sim["right"][R.fkey(f)]
        say(f"    {f:>8g} Hz: left {l:+.2f}, right {r:+.2f}, mean {(l + r) / 2:+.2f} -> nominal 60 "
            f"dB = {60 + (l + r) / 2:.1f} dB SPL (long-term level)")
    for r in recs:
        if r["coupler"] != "ear-simulator" or R.usable(r) or r.get("trial"):
            continue
        c, f = r["cup"], float(r["freq_hz"])
        if r.get("signal", "tone") == "tone":
            ref = sim[c].get(R.fkey(f))
            if ref is None:
                say(f"    linearity {c} {f:g} Hz: no check-tone offset at this frequency")
                continue
            off = R.offset_from_reading(float(r["mean_db"]), f, r.get("weighting", "Z"),
                                        float(r["test_nominal_db"]))
            say(f"    linearity {c} {f:g} Hz at {float(r['test_nominal_db']):.1f} nominal: "
                f"{off - ref:+.2f} dB from the check tone")
        else:
            if r["signal"] == "band":
                ref = sim[c].get(R.fkey(f))
                if ref is None:
                    say(f"    band {c} {f:g} Hz: no check-tone offset at this frequency")
                    continue
                pred = 60.0 + ref
                label = f"{f:g}-Hz band"
            else:
                pred = 10 * math.log10(sum(10 ** ((60.0 + sim[c][R.fkey(b)]) / 10)
                                           for b in R.TCMR_BANDS_HZ))
                label = "five-band sum"
            meas = float(r["mean_db"]) - (0.0 if r.get("weighting") == "Z"
                                          else R.weighting_db(f or 1000.0, "C"))
            say(f"    {label}, {c} cup: Leq {meas - pred:+.2f} dB from what the tone offsets predict")
    return sim


def cmd_report(a, say=print, root=None) -> int:
    history = R.rig_history(root)
    if not history:
        say("No rig set up yet (python -m audiometry setup).")
        return 1
    base = Path(root or R.DATA)
    amb = R.read_jsonl(R.rig_dir(root) / "ambient.jsonl")
    if amb:
        x = amb[-1]
        say(f"Booth noise ({x['time'][:10]}): {x['dBA']} dB(A), {x['dBC']} dB(C) "
            f"{'ok' if x.get('ok') else 'ABOVE 35 dB(A)'} (SL-100, broadband only)")
    sims = {}
    for rig in history:
        sims[(rig.headphones, rig.volume)] = _rig_section(rig, root, say)
    table = R.scoring_table()
    ref_name = "HD 280 Pro (Smull et al. 2019)" if table == "hd280pro" else "HDA 200 (ISO 389-8)"
    say(f"\nHearing levels use the {ref_name} reference values"
        + ("." if table == "hd280pro" else
           "; the HD 280 Pro values are not entered yet (standards.RETSPL_HD280PRO)."))
    if table == "hda200":
        diffs = [S.RETSPL_HDA200[f] - S.RETSPL_HD280PRO_SECONDARY[f] for f in S.RETSPL_HD280PRO_SECONDARY]
        say(f"  With the four HD 280 Pro values transcribed for 0.5-4 kHz, hearing levels would be "
            f"{min(diffs):.1f} to {max(diffs):.1f} dB higher (worse) than shown.")
    rows, self_runs = [], []
    for p in screen_files(root):
        s = json.loads(p.read_text(encoding="utf-8"))
        chain = s.get("rig") or {}
        sim = sims.get((chain.get("headphones"), chain.get("volume")))
        hl = final_hl(s, sim, table)
        v = verdict(hl, s)
        scale = "ear simulator" if sim else "jig, provisional"
        say(f"\n{p.parent.relative_to(base)} ({chain.get('headphones')} at {chain.get('volume')}): "
            f"{s.get('status')}; hearing levels ({scale}, {table}): {v.upper()}")
        for ear in R.CUPS:
            if ear in hl:
                say(f"  {ear:5} " + "  ".join(f"{f:g}:{'-' if x is None else f'{x:g}'}"
                                            for f, x in sorted(hl[ear].items())))
        for fl in (s.get("plan") or {}).get("flags", []):
            say(f"  flag: {fl}")
        say(f"  tones {s.get('n_tones')}, catch trials {s.get('n_catch_trials')}, false responses "
            f"{s.get('n_false_responses')}, voided {s.get('n_void')}; pre-test: {s.get('pre_test')}")
        if str(s.get("code", "")).upper() == "SELF":
            self_runs.append((p.parent.name, hl))
        for ear in R.CUPS:
            for f, x in sorted(hl.get(ear, {}).items()):
                rows.append({"session": str(p.parent.relative_to(base)), "code": s.get("code"),
                             "headphones": chain.get("headphones"), "volume": chain.get("volume"),
                             "ear": ear, "freq_hz": f, "db_hl": x,
                             "status": ((s.get("thresholds") or {}).get(ear) or {}).get(f"{f:g}", {}).get("status"),
                             "scale": scale, "reference": table, "verdict": v})
    for (n0, a0), (n1, a1) in zip(self_runs, self_runs[1:]):
        for ear in R.CUPS:
            for f in R.SCREEN_HZ:
                x, y = a0.get(ear, {}).get(f), a1.get(ear, {}).get(f)
                if x is not None and y is not None and abs(y - x) >= 10:
                    say(f"⚠ SELF {n0} -> {n1}: {ear} {f:g} Hz changed by {y - x:+g} dB "
                        f"(BSA check 12)")
    tc = [(d, s, t) for d, s, t in _sessions(root, "*/tcmr_session_*/session.json")
          if R.rig_at(history, t) is not None]
    if tc:
        say("\ntcmr thresholds: relative to the 1-kHz band (needs no calibration); and in dB SPL per "
            "cup where the chain has ear-simulator offsets")
        for d, s, t in tc:
            rig = R.rig_at(history, t)
            sim = sims.get((rig.headphones, rig.volume))
            sp = d / "summary.json"
            if not sp.exists():
                say(f"  {d.relative_to(base)}: not analysed yet")
                continue
            th = json.loads(sp.read_text(encoding="utf-8")).get("thresholds_db") or {}
            say(f"  {d.relative_to(base)} ({rig.headphones} at {rig.volume}):")
            for cond, x in th.items():
                if x is None:
                    continue
                line = f"    {cond:22} {x - TCMR_BAND_NOMINAL:+6.1f} dB re the band"
                if sim:
                    l, r = x + sim["left"][1000.0], x + sim["right"][1000.0]
                    line += f"   {l:5.1f} / {r:5.1f} dB SPL (left / right), mean {(l + r) / 2:.1f}"
                say(line)
    probs = session_checks(history, root)
    if probs:
        say("\nSessions that do not match the rig they were recorded on:")
        for x in probs:
            say("  " + x)
    if getattr(a, "csv", None) and rows:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        say(f"\nwrote {a.csv}")
    return 0
