"""Command line for the sheared four-tone figure task.

    python -m tshear design                     the conditions, the geometry, the duration
    python -m tshear verify                     every check that does not need a listener
    python -m tshear model                      the prediction, and why lambda2/lambda1 is not it
    python -m tshear demo --out demo/           one WAV per condition
    python -m tshear calibrate                  loop the reference tone
    python -m tshear run --data data            run a session
    python -m tshear analyze data/P01/tshear_session_01
    python -m tshear plots --out verification/  the figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DEFAULT, Config, ConfigError, describe, validate


def load_config(args) -> Config:
    cfg = DEFAULT
    if getattr(args, "config", None):
        cfg = Config.from_dict(json.loads(Path(args.config).read_text()))
    for kv in getattr(args, "set", None) or []:
        k, _, v = kv.partition("=")
        if not hasattr(cfg, k):
            raise SystemExit(f"unknown parameter {k}")
        try:
            val = json.loads(v)
        except json.JSONDecodeError:
            val = v
        cfg = cfg.replace(**{k: tuple(val) if isinstance(val, list) else val})
    return cfg


def add_common(p):
    p.add_argument("--config")
    p.add_argument("--set", action="append", metavar="KEY=VALUE")


def cmd_design(args):
    cfg = load_config(args)
    from .design import audit_design, duration_estimate, make_design
    print(describe(cfg))
    e = duration_estimate(cfg)
    print(f"\n  duration   {e['n_trials']} trials, {e['median_trials_per_track']:.0f} per track, "
          f"{e['total_minutes']:.0f} min total ({e['main_minutes']:.0f} main + "
          f"{e['practice_minutes']:.0f} practice + {e['break_minutes']:.0f} breaks + "
          f"{e['setup_minutes']:.0f} setup)")
    print(f"             worst case {e['worst_case_minutes']:.0f} min")
    a = audit_design(cfg, make_design(cfg, args.code, 1))
    print(f"\n  order      {a['n_tracks']} tracks in {make_design(cfg, args.code, 1)['n_blocks']} "
          f"block(s); serial-position spread {a['serial_position_spread']:.3f}, "
          f"longest run {a['longest_condition_run']}")


def cmd_verify(args):
    cfg = load_config(args)
    from .verify import format_report, run_battery
    battery = run_battery(cfg)
    text = format_report(cfg, battery)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    sys.exit(1 if any(not c.passed for v in battery.values() for c in v) else 0)


def cmd_model(args):
    cfg = load_config(args)
    from .model import effective_objects, prediction_band, two_channel_check
    d = validate(cfg)
    tc = two_channel_check()
    print("lambda2/lambda1 is a TWO-channel reading and does not generalise.")
    print(f"  for N=2 the normalised participation ratio is monotone in it "
          f"({tc['at_0']:.3f} to {tc['at_1']:.3f}), so what follows is a generalisation of the")
    print("  published index rather than a different quantity.\n")
    print(f"  {'step%':>7} {'step ms':>9} {'l2/l1':>8} {'objects':>9} {'normalised':>11}")
    for c in d.conditions:
        if c.kind != "figure":
            continue
        v = effective_objects(cfg, c.step_pct)
        print(f"  {c.step_pct:7.4g} {cfg.step_ms(c.step_pct):9.2f} "
              f"{v['lambda2_over_lambda1']:8.3f} {v['participation_ratio']:9.2f} "
              f"{v['normalised']:11.3f}")
    band = prediction_band(cfg)
    print(f"\n  across {band['n_variants']} readings of the filter bank: monotone "
          f"{band['all_monotone']}, largest step down {band['max_decrease']:.4f}, "
          f"largest disagreement in height {band['max_spread']:.3f}")


def cmd_demo(args):
    import numpy as np
    import soundfile as sf
    cfg = load_config(args)
    d = validate(cfg)
    from .stimulus import build_trial, render_trial, to_output
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    direction = {"random": +1, "forward": +1, "backward": -1}[cfg.delta_direction]
    for c in d.conditions:
        tr = build_trial(cfg, c, args.delta, rng, target_position=1, direction=direction)
        p = out / f"tshear_{c.name}_delta{args.delta:g}ms.wav"
        sf.write(p, to_output(cfg, render_trial(cfg, tr)), cfg.sample_rate)
        print(f"  {p}   target interval 1, shift {tr.delta_signed_ms:+.1f} ms"
              + ("  (jitter exceeds the step)" if tr.crosses_neighbour else ""))


def cmd_calibrate(args):
    from .runner import Runner
    cfg = load_config(args)
    r = Runner(cfg, Path(args.data), device=args.device)
    r.sdir = Path(args.data)
    r.meta = {"calibration": None}
    r.calibrate()


def cmd_run(args):
    from .runner import Runner
    cfg = load_config(args)
    Runner(cfg, Path(args.data), device=args.device, audio=not args.no_audio,
           auto=args.auto).run(code=args.code, resume=args.resume, session_index=args.session)


def cmd_analyze(args):
    from .analysis import analyse
    cfg = load_config(args) if (args.config or args.set) else None
    text = analyse([Path(p) for p in args.dirs], cfg=cfg)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)


def cmd_plots(args):
    cfg = load_config(args)
    from .plots import write_all
    for p in write_all(cfg, Path(args.out), dirs=[Path(x) for x in args.dirs or []]):
        print(f"  {p}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tshear", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("design"); add_common(q); q.add_argument("--code", default="EXAMPLE")
    q.set_defaults(func=cmd_design)
    q = sub.add_parser("verify"); add_common(q); q.add_argument("--out")
    q.set_defaults(func=cmd_verify)
    q = sub.add_parser("model"); add_common(q); q.set_defaults(func=cmd_model)
    q = sub.add_parser("demo"); add_common(q)
    q.add_argument("--out", default="demo"); q.add_argument("--delta", type=float, default=25.0)
    q.add_argument("--seed", type=int, default=0); q.set_defaults(func=cmd_demo)
    q = sub.add_parser("calibrate"); add_common(q)
    q.add_argument("--data", default="data"); q.add_argument("--device")
    q.set_defaults(func=cmd_calibrate)
    q = sub.add_parser("run"); add_common(q)
    q.add_argument("--data", default="data"); q.add_argument("--code")
    q.add_argument("--device"); q.add_argument("--no-audio", action="store_true")
    q.add_argument("--resume", action="store_true"); q.add_argument("--session", type=int)
    q.add_argument("--auto", choices=["figure", "within"])
    q.set_defaults(func=cmd_run)
    q = sub.add_parser("analyze"); add_common(q)
    q.add_argument("dirs", nargs="+"); q.add_argument("--out")
    q.set_defaults(func=cmd_analyze)
    q = sub.add_parser("plots"); add_common(q)
    q.add_argument("--out", default="verification"); q.add_argument("dirs", nargs="*")
    q.set_defaults(func=cmd_plots)

    args = ap.parse_args(argv)
    try:
        args.func(args)
    except ConfigError as e:
        raise SystemExit(f"configuration error: {e}")


if __name__ == "__main__":
    main()
