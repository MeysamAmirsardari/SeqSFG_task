"""Command line for the jittered A-B pair task.

    python -m tpair design                      the conditions, the geometry, the duration
    python -m tpair verify                      every check that does not need a listener
    python -m tpair model                       the prediction, and what each listener could do
    python -m tpair demo --out demo/            one WAV per condition
    python -m tpair calibrate                   loop the reference tone
    python -m tpair run --data data             run a session
    python -m tpair analyze data/L01/tpair_session_01
    python -m tpair plots --out verification/   the figures
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
    from .model import lag_index, prediction_band
    from .observer import threshold
    from .config import conditions
    print("lambda2/lambda1 for jittered pairs, and what two model listeners can reach")
    print(f"  {'condition':<10} {'lag':>5} {'l2/l1':>7} {'B only':>9} {'A-B lag':>9}")
    for c in conditions(cfg):
        idx = "-" if c.kind == "b_alone" else f"{lag_index(cfg, c.lag_ms):.3f}"
        print(f"  {c.name:<10} {('-' if c.kind == 'b_alone' else f'{c.lag_ms:g}'):>5} {idx:>7} "
              f"{threshold(cfg, c, 'single', n=800):7.1f}ms {threshold(cfg, c, 'pair', n=800):7.1f}ms")
    lags = [c.lag_ms for c in conditions(cfg) if c.kind == "pair"]
    band = prediction_band(cfg, lags, n_seq=15)
    print(f"\n  across {len(band['curves'])} readings of the filter bank: monotone "
          f"{band['all_monotone']}, largest disagreement in height {band['max_spread']:.2f}")


def cmd_demo(args):
    import numpy as np
    import soundfile as sf
    cfg = load_config(args)
    from .config import conditions
    from .stimulus import build_trial, render_trial, to_output
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for c in conditions(cfg):
        tr = build_trial(cfg, c, args.delta, rng, target_position=1)
        p = out / f"tpair_{c.name}_delta{args.delta:g}ms.wav"
        sf.write(p, to_output(cfg, render_trial(cfg, tr)), cfg.sample_rate)
        print(f"  {p}   target interval 1, pair "
              f"{'N' if tr.target_pair == cfg.n_pairs - 1 else 'N-1'}, B {abs(tr.delta_signed_ms):g} ms early")


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
    ap = argparse.ArgumentParser(prog="tpair", description=__doc__,
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
    q.add_argument("--auto", choices=["pair", "single"])
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
