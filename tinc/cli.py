"""python -m tinc <command>: design | verify | check | demo | run | simulate | analyze.

    design             the stimulus, the blocks, the expected session length
    verify             every check that needs no listener (stimulus and listening strategies)
    check --seed N     every trial a session with seed N will play, drawn and checked;
                       `run --seed N` refuses a seed that has not passed
    demo               example sequences as WAV files
    run --code P01     a session (full screen)
    simulate           a session played by a model listener
    analyze FOLDER     d' and c per delay, and a figure beside Micheyl et al. (2013)
    group FOLDERS      every listener and the group in the layout of their Fig. 7
"""
from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from pathlib import Path

import numpy as np

from .config import Config, validate
from .stimulus import build

PRESET = Path(__file__).resolve().parent / "configs" / "tinc_main.json"


def _cfg(p):
    return validate(Config.load(p or PRESET))


def drive(exp, listener) -> float:
    """Run the script headlessly with a model listener; returns simulated seconds."""
    from tcmr.session import Play, Result, Screen
    cfg, t = exp.cfg, 0.0
    gen = exp.script()
    act = next(gen)
    try:
        while True:
            if isinstance(act, Screen):
                if act.kind == "form":
                    res = {"key": "return", "values": {f["name"]: "sim" for f in act.fields}}
                    t += 60
                else:
                    res = {"key": act.keys[0]}
                    t += 6 + act.min_s
            else:
                tr = act.trial
                sound = tr.n_samples / tr.sample_rate
                if act.mode == "yesno":
                    res = Result(response=listener.respond(tr), rt_ms=800.0)
                    t += cfg.ready_ms / 1000 + sound + 0.8 + (cfg.feedback_ms + cfg.blank_ms) / 1000
                else:
                    res = Result()
                    t += cfg.ready_ms / 1000 + sound + 1.6
            act = gen.send(res)
    except StopIteration:
        pass
    return t


def new_session(cfg, data, code, meta_extra=None, seed=None):
    from tsync.design import clean_code
    from tsync.system import git_info, machine, source_hash, versions
    from .session import FIELDS, Writer
    code = clean_code(code)
    root = Path(data) / code
    n = 1 + max([int(p.name.rsplit("_", 1)[1]) for p in root.glob("tinc_session_*")], default=0)
    seed = secrets.randbits(62) if seed is None else seed
    g = git_info()
    meta = {"task": "tinc", "code": code, "session_index": n, "config": cfg.to_dict(),
            "config_hash": cfg.hash(), "session_seed": seed,
            "git": {k: v for k, v in g.items() if k != "diff"}, "source_tsync": source_hash(),
            "versions": versions(), "machine": machine(), **(meta_extra or {})}
    w = Writer(root / f"tinc_session_{n:02d}", meta, fields=FIELDS)
    if g.get("diff"):
        (w.folder / "code.patch").write_text(g["diff"])
    return w, seed


# ----------------------------------------------------------------------------------------------
def cmd_design(a):
    import tempfile
    from .observer import SimListener
    from .session import Experiment, FIELDS, Writer, block_plan
    cfg = _cfg(a.config)
    print(f"{cfg.name} ({cfg.hash()}): B {cfg.f_b_hz:g} Hz, A {cfg.f_a_hz:.2f} Hz "
          f"({cfg.df_semitones:g} semitones below); {cfg.n_tones} tones per stream, "
          f"{cfg.tone_ms:g} ms with {cfg.ramp_ms:g}-ms ramps, one every {cfg.soa_ms:g} ms "
          f"({1000 / cfg.soa_ms:g} Hz per stream)")
    rule = ("A and B roved together per pair (+-{r:g} dB); 1-2 pairs {s:g} dB louder (both tones)"
            if cfg.levels == "common" else
            "B fixed; A roved alone (+-{r:g} dB); 1-2 A tones {s:g} dB louder (Micheyl et al. 2013)")
    print(f"  levels {cfg.level_db:g} dB SPL; " + rule.format(r=cfg.rove_db, s=cfg.decoy_db))
    print(f"  change: one B tone (index {cfg.target_from}-{cfg.n_tones - 1}) {cfg.target_db:g} dB louder, "
          f"half the trials")
    print(f"  delays (A after B): {', '.join(f'{d:g}' for d in cfg.delays_ms)} ms; "
          f"sequence {cfg.sequence_ms(0) / 1000:.2f}-{cfg.sequence_ms(max(cfg.delays_ms)) / 1000:.2f} s")
    print(f"  blocks: {len(cfg.delays_ms) * cfg.blocks_per_delay} x {cfg.trials_per_block} trials "
          f"= {cfg.n_trials} trials ({cfg.blocks_per_delay * cfg.trials_per_block} per delay); "
          f"example order: " + " ".join(f"{b['delay_ms']:g}" for b in block_plan(cfg, 1)))
    with tempfile.TemporaryDirectory() as td:
        ts = []
        for i in range(3):
            w = Writer(Path(td) / f"s{i}", {"config": cfg.to_dict()}, fields=FIELDS)
            ts.append(drive(Experiment(cfg, w, 1000 + i), SimListener(cfg, seed=i)))
    print(f"  session about {np.median(ts) / 60:.0f} min in the app (simulated, 0.8-s answers; "
          f"add setup and breaks)")


def cmd_verify(a):
    from .observer import d_prime
    from .stimulus import sha256
    cfg = _cfg(a.config)
    ok = True

    def check(name, cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'pass' if cond else 'FAIL'}  {name}: {msg}")

    tr, buf = build(cfg, 50.0, True, (1, 2, 3))
    b_on = [round(t.onset_s * 1000, 6) for t in tr.tones if t.stream == "B"]
    a_on = [round(t.onset_s * 1000, 6) for t in tr.tones if t.stream == "A"]
    check("timing", b_on == [cfg.pre_ms + k * cfg.soa_ms for k in range(cfg.n_tones)]
          and a_on == [cfg.pre_ms + k * cfg.soa_ms + 50.0 for k in range(cfg.n_tones)],
          f"B every {cfg.soa_ms:g} ms, A {50:g} ms after its B (example at D = 50)")
    t1, b1 = build(cfg, 0.0, False, (9,))
    t2, b2 = build(cfg, 0.0, False, (9,))
    check("reproducible", sha256(b1) == sha256(b2), "same seed, same buffer")
    peak = max(float(np.abs(build(cfg, d, p, (5, k))[1]).max())
               for d in cfg.delays_ms for p in (0, 1) for k in range(20))
    check("headroom (uncalibrated)", peak < cfg.max_peak, f"tallest peak in 200 trials {peak:.3f}")
    dp = {m: d_prime(cfg, 0.0, m, n=1200, seed=2) for m in ("b_alone", "pair", "fused")}
    if cfg.levels == "common":
        check("only comparing B with its A works", dp["pair"] > 1.2 and dp["b_alone"] < 0.7
              and dp["fused"] < 0.7,
              f"model d' (1-dB listening noise): pair {dp['pair']:.2f}, B alone {dp['b_alone']:.2f}, "
              f"overall loudness {dp['fused']:.2f}")
    else:
        check("only following B alone works", dp["b_alone"] > 2.0 and dp["pair"] < 0.8
              and dp["fused"] < 0.8,
              f"model d' (1-dB listening noise): B alone {dp['b_alone']:.2f}, pair {dp['pair']:.2f}, "
              f"overall loudness {dp['fused']:.2f}")
    flat = [d_prime(cfg, d, "pair" if cfg.levels == "common" else "b_alone", n=800, seed=4)
            for d in (0.0, max(cfg.delays_ms))]
    check("the delay changes nothing in the sound", abs(flat[0] - flat[1]) < 0.25,
          f"best-strategy d' {flat[0]:.2f} at 0 ms, {flat[1]:.2f} at {max(cfg.delays_ms):g} ms")
    sys.exit(0 if ok else 1)


def cmd_check(a):
    from . import audit
    cal = None
    if a.calibrated:
        from tsync.levels import load_latest
        cal = load_latest(Path(a.data))
        if cal is None or not cal.calibrated:
            print("✗ no calibration: run `python -m tsync calibrate` (same two frequencies)")
            sys.exit(2)
    cfg = _cfg(a.config)
    amp = (lambda f, db: cal.amplitude(f, db)) if cal else None
    t0 = time.time()
    rep = audit.audit(cfg, a.seed, amp, progress=lambda i, n: print(f"  {i}/{n} trials", end="\r"))
    rep["seconds"] = round(time.time() - t0, 1)
    path = audit.save(rep, audit.report_path(a.data, cfg, a.seed))
    print(audit.describe(rep))
    print(f"  report: {path}  ({rep['seconds']:.0f} s)")
    sys.exit(0 if rep["ok"] else 1)


def cmd_demo(a):
    import soundfile as sf
    cfg = _cfg(a.config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for d in cfg.delays_ms:
        for p in (1, 0):
            tr, buf = build(cfg, d, bool(p), (2026, int(d), p))
            f = out / f"{cfg.levels}_D{d:g}_{'change' if p else 'nochange'}.wav"
            sf.write(f, buf, cfg.sample_rate, subtype="PCM_24")
            print(f, f"(change on B tone {tr.target_index})" if p else "")


def cmd_run(a):
    from tsync.audio import AudioEngine, AudioError, find_output
    from tsync.logbook import read_jsonl
    from tsync.system import KeepAwake, set_output_volume, volume_settings
    from .analysis import describe, figure, summarize
    from .app import run_app
    from .session import Experiment, FIELDS, Writer
    cfg = _cfg(a.config)
    match = "AirPods" if a.airpods else (a.device_match or cfg.device_match)
    try:
        dev = find_output(match)
    except AudioError as e:
        print(f"✗ {e}\n  Pass --device-match with part of the output's name.")
        sys.exit(2)
    name = dev["name"].lower()
    bluetooth = any(k in name for k in ("airpods", "bluetooth", "beats", "buds"))
    vol = volume_settings()
    if bluetooth:
        print("! Bluetooth output: compression and adaptive EQ change the sound, and levels cannot "
              "be calibrated (pilot data). Turn noise control and Spatial Audio OFF.")
    cal = None
    if not (a.uncalibrated or bluetooth):
        from tsync.levels import load_latest
        cal = load_latest(Path(a.data))
        if cal is None or not cal.calibrated or cal.device != dev["name"] \
                or cal.system_volume != vol.get("output_volume"):
            print("✗ no matching calibration (device and system volume must match): run "
                  "`python -m tsync calibrate` (it measures 1000 and 420.45 Hz), or pass "
                  "--uncalibrated for a pilot")
            sys.exit(2)
    if a.volume is not None:
        vol = set_output_volume(a.volume)
        if abs((vol.get("output_volume") or -99) - a.volume) > 1:
            print(f"✗ could not set the system volume to {a.volume}")
            sys.exit(2)
        print(f"volume set to {vol['output_volume']} and held there")
    elif vol.get("muted"):
        print("✗ the output is muted")
        sys.exit(2)
    amp = (lambda f, db: cal.amplitude(f, db)) if cal else None
    eng = AudioEngine(dev["index"], cfg.sample_rate, cfg.channels,
                      1024 if bluetooth else cfg.blocksize, "high" if bluetooth else cfg.latency)
    try:
        info = eng.start()
    except AudioError as e:
        print(f"✗ {e}")
        sys.exit(2)
    extra = {"audio_device": dev, "audio_stream": info, "volume_at_start": vol,
             "calibration": cal.to_dict() if cal else None, "levels_calibrated": cal is not None,
             "bluetooth_output": bluetooth, "volume_locked": True}
    if a.resume:
        folder = Path(a.resume)
        w = Writer(folder, {}, resume=True, fields=FIELDS)
        cfg = validate(Config.from_dict(w.meta["config"]))
        rows = [r["row"] for r in read_jsonl(folder / "trials.jsonl")]
        exp = Experiment(cfg, w, w.meta["session_seed"], amp, resume_rows=rows)
    else:
        if a.seed is not None:
            from . import audit
            rp = audit.report_path(a.data, cfg, a.seed)
            rep = json.loads(rp.read_text()) if rp.exists() else None
            want = "calibrated" if cal else "uncalibrated"
            if rep is None or not rep.get("ok") or rep.get("amplitude") != want:
                why = ("no check found" if rep is None else "the check failed" if not rep.get("ok")
                       else f"the check used {rep.get('amplitude')} amplitudes, this run is {want}")
                print(f"✗ seed {a.seed}: {why}. Run first:\n  python -m tinc check --config "
                      f"{a.config or PRESET} --seed {a.seed}" + (" --calibrated" if cal else ""))
                sys.exit(2)
            extra["seed_check"] = {"report": str(rp), "ok": True, "trials_checked": rep["trials_checked"]}
            print(f"seed {a.seed}: every planned trial was checked ({rp.name})")
        w, seed = new_session(cfg, a.data, a.code, extra, seed=a.seed)
        exp = Experiment(cfg, w, seed, amp)
    print(f"session: {w.folder}  (output {dev['name']})")
    locked = vol.get("output_volume") if vol.get("readable") else None
    awake = KeepAwake()
    awake.start()
    try:
        why = run_app(exp, eng, locked, fullscreen=not a.windowed,
                      set_volume=set_output_volume if a.volume is not None else None)
    finally:
        awake.stop()
        eng.stop()
        if w.meta.get("status") == "running":
            w.close("interrupted")
    print(f"ended: {why}")
    s = summarize(w.folder)
    print(describe(s))
    print(figure(w.folder, s))


def cmd_simulate(a):
    from .analysis import describe, figure, summarize
    from .observer import SimListener
    from .session import Experiment
    cfg = _cfg(a.config)
    w, seed = new_session(cfg, a.data, a.code, {"simulated": True, "listener": a.listener}, seed=a.seed)
    t = drive(Experiment(cfg, w, seed), SimListener(cfg, a.listener, seed=a.seed))
    print(f"simulated {t / 60:.1f} min -> {w.folder}")
    s = summarize(w.folder)
    print(describe(s))
    print(figure(w.folder, s))


def cmd_analyze(a):
    from .analysis import describe, figure, summarize
    for f in a.sessions:
        s = summarize(Path(f))
        print(describe(s))
        print(figure(Path(f), s))


def cmd_group(a):
    from .analysis import group_figure
    out = group_figure([Path(f) for f in a.sessions], Path(a.out))
    print(out)
    print(out.with_suffix(".csv"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tinc", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def p(name, fn):
        sp = sub.add_parser(name)
        sp.add_argument("--config", default=None)
        sp.set_defaults(fn=fn)
        return sp

    p("design", cmd_design)
    p("verify", cmd_verify)
    sp = p("check", cmd_check)
    sp.add_argument("--seed", type=int, required=True)
    sp.add_argument("--data", default="data")
    sp.add_argument("--calibrated", action="store_true")
    sp = p("demo", cmd_demo)
    sp.add_argument("--out", default="data_tinc/demo")
    sp = p("run", cmd_run)
    sp.add_argument("--code", required=True)
    sp.add_argument("--data", default="data")
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument("--device-match", default=None)
    sp.add_argument("--uncalibrated", action="store_true")
    sp.add_argument("--airpods", action="store_true")
    sp.add_argument("--volume", type=int, default=None)
    sp.add_argument("--windowed", action="store_true")
    sp.add_argument("--resume", default=None)
    sp = p("simulate", cmd_simulate)
    sp.add_argument("--data", default="data_tinc/sim")
    sp.add_argument("--code", default="SIM01")
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--listener", default="pair",
                    choices=("pair", "b_alone", "nearest", "fused", "grouping"))
    sp = p("analyze", cmd_analyze)
    sp.add_argument("sessions", nargs="+")
    sp = p("group", cmd_group)
    sp.add_argument("sessions", nargs="+")
    sp.add_argument("--out", default="group_tinc/fig7_style.png")
    a = ap.parse_args(argv)
    a.fn(a)
