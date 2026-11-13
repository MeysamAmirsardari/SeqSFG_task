"""python -m tcmr <command>: design | verify | demo | calibrate | run | simulate | analyze."""
from __future__ import annotations

import argparse
import json
import math
import secrets
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from .config import Condition, Config, validate
from .stimulus import band_envelopes, build

PRESET = Path(__file__).resolve().parent / "configs" / "tcmr_main.json"


def _cfg(p):
    return validate(Config.load(p or PRESET))


# ----------------------------------------------------------------------------------------------
def drive(exp, listener, rng):
    """Run the script headlessly; returns simulated seconds."""
    from .session import Play, Result, Screen
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
                sound = tr.interval_off_s[-1] - tr.interval_on_s[0]
                if act.mode == "respond":
                    ok = rng.random() < listener.p(tr.condition, tr.level_db)
                    resp = tr.signal_interval if ok else int(rng.choice(
                        [i for i in (1, 2, 3) if i != tr.signal_interval]))
                    res = Result(response=resp, rt_ms=800.0)
                    t += cfg.ready_ms / 1000 + sound + 0.8 + (cfg.feedback_ms + cfg.blank_ms) / 1000
                elif act.mode == "rate":
                    res = Result(response=int(rng.integers(1, 6)), rt_ms=1500.0)
                    t += cfg.ready_ms / 1000 + sound + 1.5 + cfg.blank_ms / 1000
                else:
                    res = Result()
                    t += cfg.ready_ms / 1000 + sound + 1.6
            act = gen.send(res)
    except StopIteration:
        pass
    return t


DEFAULT_SIM = {"co_sync_0": 52.0, "rand_sync_0": 60.0, "co_ascending_10": 54.0,
               "co_ascending_20": 57.0, "co_zigzag_20": 58.5, "rand_ascending_20": 60.0}


def sim_thresholds(cfg) -> dict:
    """A plausible listener for any design: 8 dB of release at sync, shrinking with the shear."""
    out = dict(DEFAULT_SIM)
    for c in cfg.condition_list():
        if c.name not in out:
            out[c.name] = 60.0 if c.envelope == "rand" else min(60.0, 52.0 + 0.25 * c.dt_ms
                                                                  + (1.5 if c.arrangement == "zigzag" else 0))
    return out


def new_session(cfg, data, code, meta_extra=None, seed=None):
    from tsync.design import clean_code
    from tsync.system import git_info, machine, source_hash, versions
    from .session import Writer
    code = clean_code(code)
    root = Path(data) / code
    n = 1 + max([int(p.name.rsplit("_", 1)[1]) for p in root.glob("tcmr_session_*")], default=0)
    seed = secrets.randbits(62) if seed is None else seed
    g = git_info()
    meta = {"task": "tcmr", "code": code, "session_index": n, "config": cfg.to_dict(),
            "config_hash": cfg.hash(), "session_seed": seed, "git": {k: v for k, v in g.items()
                                                                     if k != "diff"},
            "source_tsync": source_hash(), "versions": versions(), "machine": machine(),
            **(meta_extra or {})}
    w = Writer(root / f"tcmr_session_{n:02d}", meta)
    if g.get("diff"):
        (w.folder / "code.patch").write_text(g["diff"])
    return w, seed


# ----------------------------------------------------------------------------------------------
def cmd_design(a):
    from .procedure import SimListener
    from .session import Experiment
    cfg = _cfg(a.config)
    print(f"{cfg.name} ({cfg.hash()}): bands {', '.join(f'{f:g}' for f in cfg.bands_hz)} Hz, "
          f"{cfg.bandwidth_hz:g} Hz wide, {cfg.band_db:g} dB SPL each; bursts {cfg.burst_ms:g} ms "
          f"every {cfg.noi_ms:g} ms; {cfg.n_precursors} precursors; tone {cfg.tone_hz:g} Hz")
    for c in cfg.condition_list():
        off = cfg.offsets_ms(c.arrangement, c.dt_ms)
        gap = min(cfg.noi_ms - o - cfg.burst_ms for o in off)
        print(f"  {c.name:<20} x{c.tracks}  offsets {off} ms, smallest gap before the final "
              f"burst {gap:g} ms")
    with tempfile.TemporaryDirectory() as td:
        from .session import Writer
        ts = []
        for i in range(5):
            w = Writer(Path(td) / f"s{i}", {"config": cfg.to_dict()})
            ts.append(drive(Experiment(cfg, w, 1000 + i), SimListener(sim_thresholds(cfg)),
                            np.random.default_rng(i)))
    print(f"{cfg.n_tracks} tracks x {cfg.trials_per_track} trials = "
          f"{cfg.n_tracks * cfg.trials_per_track} main trials; session about "
          f"{np.median(ts) / 60:.0f} min (simulated, including instructions and ratings)")


def cmd_verify(a):
    cfg = _cfg(a.config)
    ok = True

    def check(name, cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'pass' if cond else 'FAIL'}  {name}: {msg}")

    co = Condition("co", "ascending", 20.0)
    tr, buf = build(cfg, co, 70.0, (1, 2, 3), signal_interval=2)
    e = band_envelopes(cfg, buf, tr.target_on_s[0] + 0.03, 0.12)
    r = np.corrcoef(e)
    check("comodulation", np.min(r) > 0.9,
          f"final-burst envelope correlation across bands min {np.min(r):.3f}")
    rd = Condition("rand", "sync", 0.0)
    tr2, buf2 = build(cfg, rd, 70.0, (1, 2, 4), signal_interval=2)
    e2 = band_envelopes(cfg, buf2, tr2.target_on_s[0] + 0.03, 0.12)
    off_diag = np.abs(np.corrcoef(e2)[np.triu_indices(5, 1)])
    check("random envelopes", np.mean(off_diag) < 0.6,
          f"mean |correlation| {np.mean(off_diag):.2f} (independent noise)")
    gaps = {c.name: min(cfg.noi_ms - o - cfg.burst_ms for o in cfg.offsets_ms(c.arrangement, c.dt_ms))
            for c in cfg.condition_list()}
    check("final burst separated", min(gaps.values()) >= cfg.min_gap_ms,
          f"smallest silence before the final burst {min(gaps.values()):g} ms")
    t1, b1 = build(cfg, co, 66.0, (9,))
    t2, b2 = build(cfg, co, 66.0, (9,))
    check("reproducible", np.array_equal(b1, b2) and t1.signal_interval == t2.signal_interval,
          "same seed, same buffer")
    from .config import ARRANGEMENTS
    asc, zig = ARRANGEMENTS["ascending"], ARRANGEMENTS["zigzag"]
    tgt = cfg.target_band
    lag = lambda arr: sorted((abs(i - tgt), abs(arr[i] - arr[tgt])) for i in range(5) if i != tgt)
    check("zigzag matches ascending", lag(asc) == lag(zig) and sorted(asc) == sorted(zig),
          "same onsets and the same 1-kHz-to-flank lag at each frequency distance")
    peak = max(float(np.abs(build(cfg, c, cfg.max_db, (5, k))[1]).max())
               for k, c in enumerate(cfg.condition_list()))
    check("headroom (uncalibrated)", peak < 0.95, f"peak at the maximum tone level {peak:.2f}")
    sys.exit(0 if ok else 1)


def cmd_demo(a):
    import soundfile as sf
    cfg = _cfg(a.config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = [("co", "sync", 0), ("rand", "sync", 0), ("co", "ascending", 10),
             ("co", "ascending", 20), ("co", "zigzag", 20)]
    for e, arr, dt in items:
        c = Condition(e, arr, float(dt))
        for lvl in (None, a.level):
            tr, buf = build(cfg, c, lvl, (2026, int(dt)), n_intervals=1,
                            signal_interval=None if lvl is None else 1)
            p = out / f"{c.name}_{'notone' if lvl is None else f'tone{lvl:g}dB'}.wav"
            sf.write(p, buf, cfg.sample_rate, subtype="PCM_24")
            print(p)


def cmd_calibrate(a):
    from datetime import datetime
    from tsync.audio import AudioEngine, find_output
    from tsync.levels import Calibration
    from tsync.stimulus import envelope
    from tsync.system import volume_settings
    cfg = _cfg(a.config)
    dev = find_output(a.device_match or cfg.device_match)
    eng = AudioEngine(dev["index"], cfg.sample_rate, cfg.channels, cfg.blocksize, cfg.latency)
    eng.start()
    vol = volume_settings()
    print(f"{dev['name']}, system volume {vol['output_volume']}. Sound level meter on an ear "
          f"coupler, Z or C weighting, slow.")
    info = {k: input(f"{k}: ").strip() for k in ("headphones", "meter", "coupler", "operator")}
    amps, meas, test = {}, [], 0.05
    try:
        for f in cfg.bands_hz:
            n = 30 * cfg.sample_rate
            t = np.arange(n) / cfg.sample_rate
            x = (test * envelope(t, 30.0, 0.05) * np.sin(2 * np.pi * f * t)).astype(np.float32)
            pb = eng.play_at(np.repeat(x[:, None], cfg.channels, 1), time.perf_counter() + 0.1)
            db = float(input(f"{f:g} Hz playing at amplitude {test}: level on the meter (dB SPL): "))
            eng.abort(pb)
            time.sleep(0.1)
            amps[f"{f:.6f}"] = test * 10 ** ((60.0 - db) / 20)
            meas.append({"freq_hz": f, "amplitude": test, "db_spl": db})
    finally:
        eng.stop()
    cal = Calibration(True, 60.0, amps, dev["name"], vol["output_volume"], info["headphones"],
                      info["meter"], info["coupler"], "Z/C",
                      datetime.now().astimezone().isoformat(timespec="seconds"), info["operator"],
                      "tcmr: sines at the band centres", meas)
    d = Path(a.data) / "calibration"
    d.mkdir(parents=True, exist_ok=True)
    (d / "tcmr_latest.json").write_text(json.dumps(cal.to_dict(), indent=1))
    print(f"saved {d / 'tcmr_latest.json'}")


def _load_cal(data):
    from tsync.levels import Calibration
    p = Path(data) / "calibration" / "tcmr_latest.json"
    return Calibration.from_dict(json.loads(p.read_text())) if p.exists() else None


def cmd_run(a):
    from tsync.audio import AudioEngine, AudioError, find_output
    from tsync.system import KeepAwake, volume_settings
    from .analysis import describe, summarize
    from .app import run_app
    from .session import Experiment, Writer
    from tsync.logbook import read_jsonl
    cfg = _cfg(a.config)
    match = "AirPods" if a.airpods else (a.device_match or cfg.device_match)
    try:
        dev = find_output(match)
    except AudioError as e:
        print(f"\u2717 {e}\n  Pilot listening on the speakers: python -m tcmr run --code {a.code} "
              f"--uncalibrated --windowed --device-match Speakers")
        sys.exit(2)
    name = dev["name"].lower()
    bluetooth = any(k in name for k in ("airpods", "bluetooth", "beats", "buds"))
    wired = "headphones" in name and not bluetooth
    vol = volume_settings()
    notes = []
    if bluetooth:
        notes.append("Bluetooth output: compression, Apple's adaptive EQ and noise control change "
                     "the sound, and there is extra delay. Turn noise control OFF (not Transparency, "
                     "not Adaptive) and Spatial Audio OFF. Levels cannot be calibrated: pilot data.")
    if a.allow_volume_change:
        notes.append("Volume is not locked: it is logged on every trial; levels are nominal.")
    for n_ in notes:
        print("! " + n_)
    uncal = a.uncalibrated or bluetooth or a.allow_volume_change
    cal = None if uncal else _load_cal(a.data)
    if not uncal and (cal is None or cal.device != dev["name"]
                      or cal.system_volume != vol["output_volume"]):
        print("\u2717 no matching calibration (device and system volume must match): run "
              "`python -m tcmr calibrate`, or pass --uncalibrated for a pilot")
        sys.exit(2)
    if vol.get("muted") and a.volume is None:
        print("\u2717 the output is muted")
        sys.exit(2)
    if a.volume is not None:                      # the person asked the program to hold this volume
        from tsync.system import set_output_volume
        vol = set_output_volume(a.volume)
        if abs((vol.get("output_volume") or -99) - a.volume) > 1:
            print(f"\u2717 could not set the system volume to {a.volume} (it reads "
                  f"{vol.get('output_volume')})")
            sys.exit(2)
        print(f"volume set to {vol['output_volume']} and held there: any change is put back "
              f"before the next trial and logged")
        if bluetooth:
            print("! AirPods can change the volume by themselves: in Bluetooth settings for the "
                  "AirPods, turn OFF Personalized Volume and Conversation Awareness")
    amp = (lambda f, db: cal.amplitude(f, db)) if cal else None
    eng = AudioEngine(dev["index"], cfg.sample_rate, cfg.channels,
                      1024 if bluetooth else cfg.blocksize, "high" if bluetooth else cfg.latency)
    try:
        info = eng.start()
    except AudioError as e:
        print(f"\u2717 {e}")
        sys.exit(2)
    extra = {"audio_device": dev, "audio_stream": info, "volume_at_start": vol,
             "calibration": cal.to_dict() if cal else None, "levels_calibrated": cal is not None,
             "pilot_output": not wired, "bluetooth_output": bluetooth,
             "volume_locked": not a.allow_volume_change, "visual_offset_ms": a.visual_offset_ms,
             "output_notes": notes}
    if a.resume:
        folder = Path(a.resume)
        w = Writer(folder, {}, resume=True)
        rows = [r["row"] for r in read_jsonl(folder / "trials.jsonl")]
        exp = Experiment(cfg, w, w.meta["session_seed"], amp, resume_rows=rows)
    else:
        w, seed = new_session(cfg, a.data, a.code, extra)
        exp = Experiment(cfg, w, seed, amp)
    print(f"session: {w.folder}  (output {dev['name']}, stream latency "
          f"{info['latency_s'] * 1000:.0f} ms)")
    locked = None if a.allow_volume_change or not vol.get("readable") else vol["output_volume"]
    awake = KeepAwake()
    awake.start()
    try:
        from tsync.system import set_output_volume
        why = run_app(exp, eng, locked, fullscreen=not a.windowed,
                      visual_offset_s=a.visual_offset_ms / 1000.0,
                      set_volume=set_output_volume if a.volume is not None else None)
    finally:
        awake.stop()
        eng.stop()
        if w.meta.get("status") == "running":
            w.close("interrupted")
    print(f"ended: {why}")
    print(describe(summarize(w.folder)))


def cmd_simulate(a):
    from .analysis import describe, summarize
    from .procedure import SimListener
    from .session import Experiment
    cfg = _cfg(a.config)
    w, seed = new_session(cfg, a.data, a.code, {"simulated": True}, seed=a.seed)
    t = drive(Experiment(cfg, w, seed), SimListener(sim_thresholds(cfg)), np.random.default_rng(a.seed))
    print(f"simulated {t / 60:.1f} min -> {w.folder}")
    print(describe(summarize(w.folder)))


def cmd_analyze(a):
    from .analysis import describe, summarize
    from .plots import session_figure
    for f in a.sessions:
        s = summarize(Path(f))
        print(describe(s))
        print(session_figure(Path(f), s))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tcmr", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def p(name, fn):
        sp = sub.add_parser(name)
        sp.add_argument("--config", default=None)
        sp.set_defaults(fn=fn)
        return sp

    p("design", cmd_design)
    p("verify", cmd_verify)
    sp = p("demo", cmd_demo)
    sp.add_argument("--out", default="data_tcmr/demo")
    sp.add_argument("--level", type=float, default=66.0)
    for name, fn in (("calibrate", cmd_calibrate), ("run", cmd_run)):
        sp = p(name, fn)
        sp.add_argument("--data", default="data")
        sp.add_argument("--device-match", default=None)
        if name == "run":
            sp.add_argument("--code", required=True)
            sp.add_argument("--uncalibrated", action="store_true")
            sp.add_argument("--windowed", action="store_true")
            sp.add_argument("--resume", default=None, help="session folder to continue")
            sp.add_argument("--airpods", action="store_true",
                            help="play through AirPods (pilot only: uncalibrated)")
            sp.add_argument("--allow-volume-change", action="store_true",
                            help="do not lock the system volume (it is still logged)")
            sp.add_argument("--volume", type=int, default=None,
                            help="set the system volume (0-100) at the start and hold it there")
            sp.add_argument("--visual-offset-ms", type=float, default=0.0,
                            help="delay the box lighting (Bluetooth delay), ms")
    sp = p("simulate", cmd_simulate)
    sp.add_argument("--data", default="data_tcmr/sim")
    sp.add_argument("--code", default="SIM01")
    sp.add_argument("--seed", type=int, default=1)
    sp = p("analyze", cmd_analyze)
    sp.add_argument("sessions", nargs="+")
    a = ap.parse_args(argv)
    a.fn(a)
