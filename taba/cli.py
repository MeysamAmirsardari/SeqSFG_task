"""python -m taba <command>: design | demo | verify | run | analyze."""
from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path

import numpy as np

from .config import Config, validate
from .stimulus import build, render, sha256, sound_end

PRESET = Path(__file__).resolve().parent / "configs" / "taba_main.json"


def _cfg(p):
    return validate(Config.load(p or PRESET))


def _amp(cfg, cal=None):
    if cal is not None and cal.calibrated:
        return lambda f, db: cal.amplitude(f, db) if abs(f - cfg.f_off_hz) > 0.5 else \
            cal.amplitude(cfg.f_a_hz, db)
    return lambda f, db: cfg.amplitude * 10 ** ((db - cfg.level_db) / 20)


def make_plan(cfg: Config, seed: int):
    rng = np.random.default_rng([seed, 7])
    main = [c for c in cfg.conditions() if c.flank == "same"] * cfg.reps + \
           [c for c in cfg.conditions() if c.flank != "same"] * cfg.control_reps
    for _ in range(1000):                    # shuffle until no condition repeats back to back
        order = [main[i] for i in rng.permutation(len(main))]
        if all(a.name != b.name for a, b in zip(order, order[1:])):
            break
    else:
        raise ValueError("no order without back-to-back repeats exists for this design")
    practice = [f"D{min(cfg.d_ms):g}", f"D{max(cfg.d_ms):g}"]

    def entry(phase, c, secs, sd):
        return {"phase": phase, "condition": c.name, "d_ms": c.d_ms, "flank": c.flank,
                "seconds": secs, "seed": sd, "sound_s": sound_end(build(cfg, c, secs, sd))}
    plan = [entry("practice", cfg.condition(n), cfg.practice_s, [seed, 0, i])
            for i, n in enumerate(practice)]
    plan += [entry("main", c, cfg.trial_s, [seed, 1, i]) for i, c in enumerate(order)]
    return plan


def cmd_design(a):
    cfg = _cfg(a.config)
    plan = make_plan(cfg, 1)
    print(f"{cfg.name} ({cfg.hash()}): A {cfg.f_a_hz:.2f} Hz, B {cfg.f_b_hz:g} Hz "
          f"({cfg.df_semitones:g} st), off-frequency flanks {cfg.f_off_hz:.2f} Hz; tones "
          f"{cfg.tone_ms:g} ms ({cfg.ramp_ms:g}-ms ramps), slot {cfg.soa_ms:g} ms, triplet "
          f"{cfg.period_ms:g} ms")
    for c in cfg.conditions():
        print(f"  {c.name:<13} B {c.d_ms:+g} ms after the middle A, flanks: {c.flank}")
    secs = sum(t["seconds"] + cfg.rest_s + 1 for t in plan)
    print(f"{len(plan) - 2} sequences of {cfg.trial_s:g} s + 2 practice: about "
          f"{secs / 60 + 2:.0f} min with instructions")


def cmd_demo(a):
    import soundfile as sf
    cfg = _cfg(a.config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for c in cfg.conditions():
        x = render(cfg, build(cfg, c, a.seconds, [2026]), _amp(cfg))
        p = out / f"{c.name}.wav"
        sf.write(p, x, cfg.sample_rate, subtype="PCM_24")
        print(p)


def cmd_verify(a):
    cfg = _cfg(a.config)
    ok = True
    for c in cfg.conditions():
        tones = build(cfg, c, 10.0, [1])
        mids = {t.triplet: t.onset_s for t in tones if t.role == "mid_a"}
        bs = {t.triplet: t.onset_s for t in tones if t.role == "b"}
        lag = {round((bs[k] - mids[k]) * 1000, 9) for k in mids}
        flanks = [t for t in tones if t.role.startswith("flank")]
        good = lag == {c.d_ms} and (len(flanks) == 0) == (c.flank == "none")
        x1 = render(cfg, tones, _amp(cfg))
        x2 = render(cfg, build(cfg, c, 10.0, [1]), _amp(cfg))
        good &= sha256(x1) == sha256(x2) and float(np.abs(x1).max()) < 0.95
        ok &= good
        print(f"  {'pass' if good else 'FAIL'}  {c.name}: B lag {sorted(lag)} ms, "
              f"{len(flanks)} flanking tones, peak {np.abs(x1).max():.3f}, reproducible")
    sys.exit(0 if ok else 1)


def cmd_run(a):
    from tsync.audio import AudioEngine, find_output
    from tsync.design import clean_code
    from tsync.levels import load_latest
    from tsync.logbook import now_iso, write_json_atomic
    from tsync.system import KeepAwake, git_info, machine, source_hash, versions, volume_settings
    from . import app

    cfg = _cfg(a.config)
    from tsync.audio import AudioError
    code = clean_code(a.code)
    try:
        dev = find_output(a.device_match or cfg.device_match)
    except AudioError as e:
        print(f"\u2717 {e}\n  For a quick listen on another output (flagged as a pilot in the data):"
              f"\n  python -m taba run --code {code} --uncalibrated --windowed --device-match Speakers")
        sys.exit(2)
    if "headphones" not in dev["name"].lower():
        print(f"! playing through '{dev['name']}', not headphones: pilot listening only")
    eng = AudioEngine(dev["index"], cfg.sample_rate, cfg.channels, cfg.blocksize, cfg.latency)
    info = eng.start()
    vol = volume_settings()
    cal = None if a.uncalibrated else load_latest(Path(a.data))
    problem = None
    if cal is None and not a.uncalibrated:
        problem = "no calibration found: run `python -m tsync calibrate` or pass --uncalibrated"
    elif cal is not None and (cal.device != dev["name"] or cal.system_volume != vol["output_volume"]):
        problem = (f"the calibration was made on '{cal.device}' at volume {cal.system_volume}; "
                   f"now '{dev['name']}' at volume {vol['output_volume']}")
    elif vol.get("muted"):
        problem = "the output is muted"
    if problem:
        print(f"\u2717 {problem}")
        eng.stop()
        sys.exit(2)
    root = Path(a.data) / code
    n = 1 + max([int(p.name.rsplit("_", 1)[1]) for p in root.glob("taba_session_*")], default=0)
    folder = root / f"taba_session_{n:02d}"
    folder.mkdir(parents=True)
    seed = secrets.randbits(62)
    plan = make_plan(cfg, seed)
    g = git_info()
    meta = {"task": "taba", "code": code, "session_index": n, "status": "running",
            "created": now_iso(), "config": cfg.to_dict(), "config_hash": cfg.hash(),
            "session_seed": seed, "plan": plan, "audio_device": dev, "audio_stream": info,
            "volume_at_start": vol, "calibration": cal.to_dict() if cal else None,
            "levels": "calibrated (off-frequency flanks use the A-tone calibration)" if cal
            else "nominal", "pilot_output": "headphones" not in dev["name"].lower(),
            "git": {k: v for k, v in g.items() if k != "diff"}, "source_tsync": source_hash(),
            "versions": versions(), "machine": machine()}
    if g.get("diff"):
        (folder / "code.patch").write_text(g["diff"])
    trials = open(folder / "trials.jsonl", "a")

    def save():
        write_json_atomic(folder / "session.json", meta)

    def trial(rec):
        trials.write(json.dumps(rec, default=float) + "\n")
        trials.flush()

    save()
    amp = _amp(cfg, cal)

    def rend(tr):
        x = render(cfg, build(cfg, cfg.condition(tr["condition"]), tr["seconds"], tr["seed"]), amp)
        tr["sha256"] = sha256(x)
        return x

    awake = KeepAwake()
    awake.start()
    try:
        why = app.run(cfg, plan, eng, {"meta": meta, "save": save, "trial": trial}, rend,
                      vol.get("output_volume") if vol.get("readable") else None,
                      fullscreen=not a.windowed)
    finally:
        awake.stop()
        eng.stop()
        trials.close()
        if meta["status"] == "running":
            meta["status"] = "interrupted"
        meta["ended"] = now_iso()
        save()
    print(f"{why}: {folder}")
    from .analysis import describe, summarize
    print(describe(summarize(folder)))


def cmd_analyze(a):
    from .analysis import describe, summarize
    for f in a.sessions:
        print(describe(summarize(Path(f))))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m taba", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def p(name, fn):
        sp = sub.add_parser(name)
        sp.add_argument("--config", default=None)
        sp.set_defaults(fn=fn)
        return sp

    p("design", cmd_design)
    sp = p("demo", cmd_demo)
    sp.add_argument("--out", default="data_taba/demo")
    sp.add_argument("--seconds", type=float, default=12.0)
    p("verify", cmd_verify)
    sp = p("run", cmd_run)
    sp.add_argument("--code", required=True)
    sp.add_argument("--data", default="data")
    sp.add_argument("--uncalibrated", action="store_true")
    sp.add_argument("--windowed", action="store_true")
    sp.add_argument("--device-match", default=None)
    sp = p("analyze", cmd_analyze)
    sp.add_argument("sessions", nargs="+")
    a = ap.parse_args(argv)
    a.fn(a)
