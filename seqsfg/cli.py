"""Command line: config | verify | demo | calibrate | run | analyze | participants."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config, ConfigError, DEFAULT, describe, validate


def load_config(args) -> Config:
    cfg = DEFAULT
    if getattr(args, "config", None):
        with open(args.config) as f:
            cfg = Config.from_dict(json.load(f))
    for kv in getattr(args, "set", None) or []:
        k, _, v = kv.partition("=")
        if not hasattr(cfg, k):
            raise SystemExit(f"unknown parameter {k}")
        try:
            val = json.loads(v)
        except json.JSONDecodeError:
            val = v
        if isinstance(val, list):
            val = tuple(tuple(x) if isinstance(x, list) else x for x in val)
        cfg = cfg.replace(**{k: val})
    return cfg


def add_common(p):
    p.add_argument("--config", help="JSON file with config overrides")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override one parameter (JSON value)")


def cmd_config(args):
    cfg = load_config(args)
    if args.json:
        print(json.dumps(cfg.to_dict(), indent=1))
    else:
        print(describe(cfg))


def cmd_verify(args):
    from .verify import format_report, run_battery, to_json
    from .session import write_json
    cfg = load_config(args)
    validate(cfg)
    conds = None
    if args.main_only:
        conds = [("rising", s) for s in cfg.steps_ms]
    n = 12 if args.quick else args.trials
    print(f"verification battery: {n} trials per condition")
    res = run_battery(cfg, n_trials=n, seed=args.seed, conditions=conds)
    print(format_report(res))
    if args.json:
        write_json(Path(args.json), to_json(res))
        print(f"json written to {args.json}")


def cmd_demo(args):
    import numpy as np
    import soundfile as sf
    from .stimulus import make_trial, render_trial, render_interval
    cfg = load_config(args)
    d = validate(cfg)
    tr = make_trial(cfg, args.seed, args.step, args.variant, d=d)
    x = render_trial(cfg, tr, args.target, d)
    sf.write(args.out, x, cfg.sample_rate)
    S = tr.recurring.figure_set
    print(f"wrote {args.out}: variant={args.variant} step={args.step} ms, recurring interval is #{args.target}")
    print(f"  recurring channels S = {S.tolist()} -> {np.round(d.channel_freqs_hz[S]).astype(int).tolist()} Hz")
    print(f"  element onsets (ms): {(tr.recurring.element_onsets * cfg.grid_ms).tolist()}")
    if args.variant != "ungrouped":
        print("  redrawn sets:", [s.tolist() for s in tr.other.element_sets])
    if args.split:
        base = Path(args.out)
        sf.write(base.with_name(base.stem + "_recurring.wav"), render_interval(cfg, tr.recurring, d), cfg.sample_rate)
        sf.write(base.with_name(base.stem + "_other.wav"), render_interval(cfg, tr.other, d), cfg.sample_rate)


def cmd_plots(args):
    from .plots import make_all
    cfg = load_config(args)
    validate(cfg)
    written = make_all(cfg, Path(args.out), n_trials=args.trials, seed=args.seed)
    print("\n".join(str(p) for p in written))


def cmd_calibrate(args):
    import math
    import numpy as np
    import sounddevice as sd
    cfg = load_config(args)
    validate(cfg)
    if args.device is not None:
        sd.default.device = args.device
    n = int(cfg.calibration_dur_s * cfg.sample_rate)
    t = np.arange(n) / cfg.sample_rate
    tone = cfg.tone_amplitude * np.sin(2 * np.pi * cfg.calibration_freq_hz * t)
    r = int(0.02 * cfg.sample_rate); ramp = 0.5 * (1 - np.cos(np.pi * np.arange(r) / r))
    tone[:r] *= ramp; tone[-r:] *= ramp[::-1]
    print(f"{cfg.calibration_freq_hz:.0f} Hz at one-tone amplitude ({20 * math.log10(cfg.tone_amplitude / math.sqrt(2)):.1f} dB FS rms); "
          f"target {cfg.tone_level_db_spl:.0f} dB SPL. Ctrl-C to stop.")
    try:
        while True:
            sd.play(tone.astype(np.float32), cfg.sample_rate, blocking=True)
    except KeyboardInterrupt:
        pass


def cmd_run(args):
    from .runner import Runner
    from .analysis import analyze_sessions, summary_text
    cfg = load_config(args)
    validate(cfg)
    r = Runner(cfg, Path(args.data), device=args.device, audio=not args.no_audio, auto=args.auto, fast=args.fast)
    sdir = r.run(code=args.code, resume=args.resume, session_index=args.session)
    if (sdir / "trials.csv").exists():
        try:
            res = analyze_sessions([sdir])
            print("\n" + summary_text(res))
        except Exception as e:  # analysis must never lose a session
            print(f"(analysis skipped: {e})")


def cmd_analyze(args):
    from .analysis import analyze_sessions, summary_text
    res = analyze_sessions([Path(p) for p in args.sessions], out_dir=Path(args.out) if args.out else None,
                           battery_json=Path(args.battery) if args.battery else None)
    print(summary_text(res))


def cmd_participants(args):
    from .session import load_participants
    for code, row in load_participants(Path(args.data)).items():
        print(code, {k: v for k, v in row.items() if k != "code"})


def main(argv=None):
    p = argparse.ArgumentParser(prog="seqsfg", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("config", help="print the validated configuration and derived quantities"); add_common(q)
    q.add_argument("--json", action="store_true"); q.set_defaults(fn=cmd_config)

    q = sub.add_parser("verify", help="build fresh trials, measure everything, run the ideal observers"); add_common(q)
    q.add_argument("--trials", type=int, default=40); q.add_argument("--quick", action="store_true")
    q.add_argument("--seed", type=int, default=2026); q.add_argument("--main-only", action="store_true")
    q.add_argument("--json", help="write results to this JSON file"); q.set_defaults(fn=cmd_verify)

    q = sub.add_parser("demo", help="write one trial to a WAV file"); add_common(q)
    q.add_argument("--out", default="demo.wav"); q.add_argument("--step", type=float, default=0.0)
    q.add_argument("--variant", default="rising"); q.add_argument("--target", type=int, default=1, choices=(1, 2))
    q.add_argument("--seed", type=int, default=1); q.add_argument("--split", action="store_true", help="also write the two intervals separately")
    q.set_defaults(fn=cmd_demo)

    q = sub.add_parser("plots", help="write the diagnostic figures (rasters, matching, observers)"); add_common(q)
    q.add_argument("--out", default="verification/figures"); q.add_argument("--trials", type=int, default=24)
    q.add_argument("--seed", type=int, default=2026); q.set_defaults(fn=cmd_plots)

    q = sub.add_parser("calibrate", help="loop the reference tone for level calibration"); add_common(q)
    q.add_argument("--device"); q.set_defaults(fn=cmd_calibrate)

    q = sub.add_parser("run", help="run a session"); add_common(q)
    q.add_argument("--data", default="data"); q.add_argument("--code", help="participant code (else asked)")
    q.add_argument("--session", type=int, help="session index (else next, or last for --resume)")
    q.add_argument("--resume", action="store_true"); q.add_argument("--device")
    q.add_argument("--no-audio", action="store_true", help="dry run without sound (timings kept)")
    q.add_argument("--auto", type=float, default=None, metavar="TAU_MS",
                   help="simulated listener whose accuracy decays with step (time constant TAU_MS); pipeline test only")
    q.add_argument("--fast", action="store_true", help="with --auto: no waiting at all")
    q.set_defaults(fn=cmd_run)

    q = sub.add_parser("analyze", help="analyze one or more session directories (pooled if several)")
    q.add_argument("sessions", nargs="+"); q.add_argument("--out")
    q.add_argument("--battery", help="battery.json from 'seqsfg verify --json' for the cue profile "
                                     "(default: verification/battery.json)")
    q.set_defaults(fn=cmd_analyze)

    q = sub.add_parser("participants", help="list the participants table"); q.add_argument("--data", default="data")
    q.set_defaults(fn=cmd_participants)

    args = p.parse_args(argv)
    try:
        args.fn(args)
    except ConfigError as e:
        print(e, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
