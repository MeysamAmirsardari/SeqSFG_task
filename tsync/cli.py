"""python -m tsync <command>.  Run `python -m tsync -h` for the list."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .config import Config, validate

PRESET = Path(__file__).resolve().parent / "configs" / "tsync_main.json"
OK, BAD, WARN = "✓", "✗", "!"


def _cfg(path) -> Config:
    return validate(Config.load(path or PRESET))


def _ask(prompt: str, default: str = "") -> str:
    s = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
    return s or default


def _yes(prompt: str) -> bool:
    return _ask(prompt + " (y/n)").lower().startswith("y")


# ----------------------------------------------------------------------------------------------
def cmd_design(a):
    from .sim import estimate_duration
    from .stimulus import ending_onsets_ms, nominal_end_ms
    cfg = _cfg(a.config)
    print(f"{cfg.name}  hash {cfg.hash()}")
    print(f"tones: B {cfg.f_b_hz:g} Hz, A {cfg.f_a_hz:.2f} Hz ({cfg.df_semitones:g} st below), "
          f"{cfg.tone_ms:g} ms with {cfg.ramp_ms:g}-ms ramps, one every {cfg.soa_ms:g} ms per "
          f"stream, {cfg.n_tones} per stream")
    print(f"levels: B {cfg.level_b_db:g} dB SPL; A {cfg.level_a_db:g} +-{cfg.a_rove_db:g} dB, "
          f"{'/'.join(map(str, cfg.a_decoy_counts))} decoys +{cfg.a_decoy_db:g} dB")
    print(f"\n  {'condition':<9} {'A lags B':>9} {'ending':>10} {'sequence':>9}")
    for c in cfg.conditions():
        b, _ = ending_onsets_ms(cfg, c, 0, 1)
        off = b - (cfg.n_tones - 1) * cfg.soa_ms
        print(f"  {c.name:<9} {c.context_ms:>7g} ms {off:>+8g} ms {nominal_end_ms(cfg, c):>6g} ms")
    print(f"\ntrial: ready {cfg.ready_ms} ms, sequence, silence {cfg.isi_ms[0]:g}-{cfg.isi_ms[1]:g} "
          f"ms, sequence, answer, feedback {cfg.feedback_ms} ms, blank {cfg.blank_ms} ms")
    print(f"track: start {cfg.delta_start_ms:g} ms, {cfg.n_down}-down 1-up "
          f"({cfg.target_p:.1%}), steps {' / '.join(f'{f:.3g}' for f in cfg.step_factors)}, "
          f"{cfg.n_final_reversals} final reversals, range {cfg.delta_min_ms:g}-"
          f"{cfg.delta_max_ms:g} ms; {cfg.catch_per_track} catch trials per track at "
          f"{cfg.catch_delta_ms:g} ms")
    d = estimate_duration(cfg, n=a.n)
    print(f"\nduration over {d['n']} simulated listeners: median {d['median_min']:.1f} min, "
          f"90th percentile {d['p90_min']:.1f}, longest {d['max_min']:.1f}; "
          f"median {d['median_trials']:.0f} trials")


def cmd_verify(a):
    from .verify import run_checks
    sys.exit(0 if run_checks(_cfg(a.config)) else 1)


def cmd_preview(a):
    from .preview import render_previews
    for p in render_previews(Path(a.out)):
        print(p)


def cmd_demo(a):
    import soundfile as sf
    from .levels import uncalibrated
    from .stimulus import build_trial, render
    cfg = _cfg(a.config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cal = uncalibrated(cfg, 0.1)
    lines = []
    for c in cfg.conditions():
        for i in range(a.n):
            tr = build_trial(cfg, c, a.delta, (2026, i, int(c.context_ms), int(c.ending_shift_ms)))
            buf, _ = render(cfg, tr, cal.amplitude)
            name = f"{c.name}_{i + 1}_target{tr.target_interval}.wav"
            sf.write(out / name, buf, cfg.sample_rate, subtype="PCM_24")
            lines.append(f"{name}: {c.label}; ending out of sync in sequence "
                         f"{tr.target_interval} by {a.delta:g} ms "
                         f"({'B' if tr.sign > 0 else 'A'} first)")
    (out / "README.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ----------------------------------------------------------------------------------------------
def _engine(cfg, match):
    from .audio import AudioEngine, find_output
    dev = find_output(match or cfg.device_match)
    eng = AudioEngine(dev["index"], cfg.sample_rate, cfg.channels, cfg.blocksize, cfg.latency)
    info = eng.start()
    return eng, dev, info


def cmd_check_audio(a):
    from .levels import uncalibrated
    from .stimulus import build_trial, render
    cfg = _cfg(a.config)
    eng, dev, info = _engine(cfg, a.device_match)
    print(f"{OK} {dev['name']} ({dev['hostapi']}), stream latency {info['latency_s'] * 1000:.1f} ms,"
          f" block {cfg.blocksize} frames")
    cal = uncalibrated(cfg, 0.02 if a.audible else 0.0)
    errs, lats, under, gaps = [], [], 0, []
    try:
        for k in range(a.n):
            tr = build_trial(cfg, cfg.conditions()[k % 6], 20.0, (31, k))
            buf, _ = render(cfg, tr, cal.amplitude) if a.audible else \
                (np.zeros((tr.n_samples, cfg.channels), np.float32), None)
            t = time.perf_counter() + cfg.schedule_lead_ms / 1000.0
            pb = eng.play_at(buf, t)
            while not pb.finished:
                time.sleep(0.005)
            r = pb.report()
            errs.append((pb.onset_perf() - t) * 1000.0)
            lats.append(r["onset_error_ms"])
            under += r["underflows"]
            gaps.append(r["callback_gap_max_ms"])
            print(f"  trial {k + 1:>2}: onset error {errs[-1]:+.3f} ms, underflows "
                  f"{r['underflows']}, longest callback gap {gaps[-1]:.2f} ms", flush=True)
    finally:
        eng.stop()
    print(f"\n{OK if under == 0 else BAD} {a.n} trials: onset error {np.min(errs):+.3f} to "
          f"{np.max(errs):+.3f} ms; underflows {under}; longest callback gap {max(gaps):.2f} ms "
          f"(block = {cfg.blocksize / cfg.sample_rate * 1000:.2f} ms)")
    sys.exit(0 if under == 0 else 1)


def cmd_calibrate(a):
    from datetime import datetime
    from .levels import Calibration, amplitude_for, save_calibration, worst_case_peak
    from .stimulus import envelope
    from .system import volume_settings
    cfg = _cfg(a.config)
    eng, dev, _ = _engine(cfg, a.device_match)
    vol = volume_settings()
    print(f"output: {dev['name']}; system volume {vol['output_volume']}, muted {vol['muted']}")
    print("Use a sound level meter on an ear coupler / artificial ear, Z or C weighting, slow.")
    info = {k: _ask(k.capitalize()) for k in ("headphones", "meter", "coupler", "operator")}
    weighting = _ask("Weighting (Z or C)", "Z")
    test_amp = float(_ask("Test amplitude", "0.05"))
    meas, amps = [], {}
    sr = cfg.sample_rate
    try:
        for f in (cfg.f_b_hz, cfg.f_a_hz):
            n = int(30 * sr)
            t = np.arange(n) / sr
            x = (test_amp * envelope(t, 30.0, 0.05) * np.sin(2 * np.pi * f * t)).astype(np.float32)
            pb = eng.play_at(np.repeat(x[:, None], cfg.channels, 1), time.perf_counter() + 0.1)
            got = float(_ask(f"Playing {f:.2f} Hz at amplitude {test_amp:g} for 30 s. "
                             f"Level on the meter (dB SPL)"))
            eng.abort(pb)
            time.sleep(0.1)
            amps[f"{f:.6f}"] = amplitude_for(cfg.level_b_db, got, test_amp)
            meas.append({"freq_hz": f, "amplitude": test_amp, "measured_db_spl": got})
            print(f"  -> {cfg.level_b_db:g} dB SPL needs amplitude {amps[f'{f:.6f}']:.5f}")
    finally:
        eng.stop()
    cal = Calibration(True, cfg.level_b_db, amps, dev["name"], vol["output_volume"],
                      info["headphones"], info["meter"], info["coupler"], weighting,
                      datetime.now().astimezone().isoformat(timespec="seconds"),
                      info["operator"], "", meas)
    pk = worst_case_peak(cfg, cal)
    print(f"worst-case peak (B + loudest A) = {pk:.3f}")
    if pk > cfg.max_peak:
        print(f"{BAD} above {cfg.max_peak}: raise the system volume and calibrate again")
        sys.exit(1)
    p = save_calibration(Path(a.data), cal)
    print(f"{OK} saved {p}")


# ----------------------------------------------------------------------------------------------
def cmd_run(a):
    from .design import clean_code
    from .experiment import Experiment
    from .levels import load_latest, uncalibrated, worst_case_peak
    from .logbook import existing_sessions
    from .session import open_session, resume_session
    from .system import KeepAwake, disk_free_gb, git_info, volume_settings

    cfg = _cfg(a.config)
    data = Path(a.data)
    code = clean_code(a.code)
    print(f"{OK} configuration {cfg.name} ({cfg.hash()})")
    resume_dir = None
    for p in reversed(existing_sessions(data, code)):
        st = json.loads((p / "session.json").read_text()).get("status")
        if st not in ("complete",) and (a.resume or _yes(f"{p.name} for {code} is '{st}'. Resume it")):
            resume_dir = p
        break
    try:
        eng, dev, ainfo = _engine(cfg, a.device_match)
    except Exception as e:
        print(f"{BAD} {e}")
        sys.exit(2)
    print(f"{OK} output {dev['name']}: {ainfo['samplerate']:g} Hz, block {cfg.blocksize}, "
          f"latency {ainfo['latency_s'] * 1000:.1f} ms")
    vol = volume_settings()
    problems = []
    if vol["readable"]:
        print(f"{OK if not vol['muted'] else BAD} system volume {vol['output_volume']}, muted "
              f"{vol['muted']}, alert volume {vol['alert_volume']}")
        if vol["muted"]:
            problems.append("output is muted")
        if vol["alert_volume"]:
            print(f"{WARN} alert sounds can play through the headphones: set System Settings > "
                  "Sound > Play sound effects through: MacBook speakers, or alert volume to 0")
    else:
        print(f"{WARN} the system volume cannot be read: it will not be locked")
    cal = load_latest(data)
    if a.uncalibrated:
        cal = uncalibrated(cfg)
        print(f"{WARN} uncalibrated session: levels will be reported as nominal")
    elif cal is None:
        problems.append("no calibration: run `python -m tsync calibrate` or pass --uncalibrated")
    else:
        age = cal.age_days()
        bits = [f"{cal.date[:10]} ({age:.0f} days)", cal.device, f"volume {cal.system_volume}"]
        good = cal.device == dev["name"] and cal.system_volume == vol["output_volume"]
        print(f"{OK if good else BAD} calibration {', '.join(bits)}")
        if not good:
            problems.append("the device or system volume differs from the calibration")
        if age is not None and age > 30:
            print(f"{WARN} calibration is more than 30 days old")
        if worst_case_peak(cfg, cal) > cfg.max_peak:
            problems.append("the calibrated levels would clip")
    g = git_info()
    print(f"{OK if not g['dirty'] else WARN} code {str(g['commit'])[:10]}"
          f"{' with uncommitted changes (saved as code.patch)' if g['dirty'] else ''}")
    free = disk_free_gb(data)
    print(f"{OK if free > 1 else BAD} disk {free:.0f} GB free")
    if free < 1:
        problems.append("less than 1 GB free")
    if problems:
        eng.stop()
        for p in problems:
            print(f"{BAD} {p}")
        sys.exit(2)
    checklist = {}
    if not a.yes:
        print("\nExperimenter checklist:")
        for k, q in (("focus", "Focus / Do Not Disturb is on"),
                     ("bluetooth", "Bluetooth audio devices are off"),
                     ("headphones", "Headphones are on the right way (L on left)"),
                     ("booth", "The booth door is closed and the phone is silent")):
            checklist[k] = _yes("  " + q)
        if not all(checklist.values()) and not _yes("Not all checks passed. Start anyway"):
            eng.stop()
            sys.exit(2)
    operator = a.operator or ("" if a.yes else _ask("Experimenter initials"))
    extra = {"operator": operator, "checklist": checklist, "audio_device": dev,
             "audio_stream": ainfo, "system_volume_at_start": vol,
             "locked_volume": vol["output_volume"] if vol["readable"] else None}
    if resume_dir is not None:
        w, plan, cal, state = resume_session(resume_dir, cfg)
        w.update_meta(**{f"resume_{len(w.meta.get('resumes', []))}": extra})
    else:
        w, plan = open_session(cfg, data, code, cal, order_row=a.order_row, extra=extra)
        state = None
    print(f"\n{code}: {w.folder}\norder: {' > '.join(plan.blocks)}\n")
    from .app import Guard, run_app
    exp = Experiment(cfg, cal, plan, w, resume=state)
    awake = KeepAwake()
    awake.start()
    try:
        why = run_app(exp, eng, w, Guard(extra["locked_volume"], dev["name"]),
                      fullscreen=not a.windowed, screen_index=a.screen)
    finally:
        awake.stop()
        eng.stop()
        if w.meta.get("status") == "running":
            w.close("interrupted")
    print(f"session ended: {why}; status {w.meta.get('status')}")
    _analyze(w.folder)


def _analyze(folder: Path, plot: bool = True):
    from .analysis import describe, write_summary
    try:
        summ = write_summary(folder)
    except Exception as e:
        print(f"{WARN} analysis failed: {e}")
        return
    print(describe(summ))
    if plot:
        from .plots import session_figure
        print(session_figure(folder, summ))


def cmd_analyze(a):
    for f in a.sessions:
        _analyze(Path(f), not a.no_plot)


def cmd_group(a):
    from .analysis import group
    from .plots import group_figure
    out = Path(a.out)
    res = group([Path(p) for p in a.sessions], out)
    for name, c in res["conditions"].items():
        print(f"  {name:<9} n={c['n']:<3} {c['geomean_ms'] or float('nan'):7.2f} ms  "
              f"g={c['grouping_index_mean'] if c['grouping_index_mean'] is not None else float('nan'):.2f}")
    print(group_figure(res, out / "group.png"))


def cmd_simulate(a):
    from .experiment import Experiment
    from .levels import uncalibrated
    from .session import open_session
    from .sim import ModelListener, drive
    cfg = _cfg(a.config)
    rng = np.random.default_rng(a.seed)
    for i in range(a.n):
        code = f"{a.code}{i + 1:02d}" if a.n > 1 else a.code
        w, plan = open_session(cfg, Path(a.data), code, uncalibrated(cfg), simulated=True,
                               session_seed=int(rng.integers(2 ** 62)))
        c = drive(Experiment(cfg, uncalibrated(cfg), plan, w), ModelListener(), rng)
        print(f"{code}: {c.trials} trials, {c.t / 60:.1f} simulated minutes -> {w.folder}")
        _analyze(w.folder, plot=not a.no_plot)


def cmd_audit(a):
    from .levels import Calibration
    from .logbook import read_session
    from .stimulus import build_trial, buffer_sha256, render
    for f in a.sessions:
        s = read_session(Path(f))
        cfg = Config.from_dict(s["meta"]["config"])
        cal = Calibration.from_dict(s["meta"]["calibration"])
        bad = 0
        for t in s["trials"]:
            r, tr = t["row"], t["trial"]
            cond = cfg.condition(tr["condition"])
            x = build_trial(cfg, cond, tr["delta_ms"], tr["seed"], phase=tr["phase"],
                            is_catch=tr["is_catch"], target_interval=tr["target_interval"],
                            sign=tr["sign"])
            if buffer_sha256(render(cfg, x, cal.amplitude)[0]) != r["sha256"]:
                bad += 1
        print(f"{OK if not bad else BAD} {f}: {len(s['trials'])} trials regenerated, "
              f"{bad} mismatches")
        if bad:
            sys.exit(1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tsync", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def p(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", default=None, help="configuration JSON (default: the preset)")
        sp.set_defaults(fn=fn)
        return sp

    sp = p("design", cmd_design, "print the design and the expected session length")
    sp.add_argument("--n", type=int, default=24)
    p("verify", cmd_verify, "run every check that needs no listener")
    sp = p("preview", cmd_preview, "render every screen to PNG")
    sp.add_argument("--out", default="data_tsync/preview")
    sp = p("demo", cmd_demo, "write example trials as WAV files")
    sp.add_argument("--out", default="data_tsync/demo")
    sp.add_argument("--delta", type=float, default=30.0)
    sp.add_argument("--n", type=int, default=1)
    for name, fn, h in (("check-audio", cmd_check_audio, "test the audio path (silent by default)"),
                        ("calibrate", cmd_calibrate, "measure levels with a sound level meter")):
        sp = p(name, fn, h)
        sp.add_argument("--device-match", default=None)
        sp.add_argument("--data", default="data")
        sp.add_argument("--n", type=int, default=20)
        sp.add_argument("--audible", action="store_true")
    sp = p("run", cmd_run, "run a session")
    sp.add_argument("--code", required=True)
    sp.add_argument("--data", default="data")
    sp.add_argument("--resume", action="store_true")
    sp.add_argument("--uncalibrated", action="store_true")
    sp.add_argument("--windowed", action="store_true")
    sp.add_argument("--screen", type=int, default=0)
    sp.add_argument("--device-match", default=None)
    sp.add_argument("--order-row", type=int, default=None)
    sp.add_argument("--operator", default="")
    sp.add_argument("--yes", action="store_true", help="skip the experimenter checklist")
    sp = p("simulate", cmd_simulate, "run simulated listeners through the whole session")
    sp.add_argument("--code", default="SIM")
    sp.add_argument("--data", default="data_sim")
    sp.add_argument("--n", type=int, default=1)
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--no-plot", action="store_true")
    sp = p("analyze", cmd_analyze, "thresholds and checks for session folders")
    sp.add_argument("sessions", nargs="+")
    sp.add_argument("--no-plot", action="store_true")
    sp = p("group", cmd_group, "group thresholds and the figure")
    sp.add_argument("sessions", nargs="+")
    sp.add_argument("--out", default="data_tsync/group")
    sp = p("audit", cmd_audit, "regenerate every logged trial and compare SHA-256")
    sp.add_argument("sessions", nargs="+")
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
