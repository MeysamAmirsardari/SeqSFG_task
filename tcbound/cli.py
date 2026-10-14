"""Command line for the binding-bound task.

    python -m tcbound design                   conditions, levels, ceilings, duration
    python -m tcbound verify                   every check that does not need a listener
    python -m tcbound model                    the predictions, fixed before any data
    python -m tcbound demo --out demo/         one WAV per condition
    python -m tcbound calibrate                play the reference tone and record the level
    python -m tcbound run --data data          run a session
    python -m tcbound analyze data/L01/tcbound_session_01
    python -m tcbound group data/L*/tcbound_session_* --out group_out/
    python -m tcbound plots --out verification/ [session dirs]
    python -m tcbound export data/L01/tcbound_session_01 --out stimuli/
    python -m tcbound power --gain 0.5 --listeners 12 --studies 20
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
    from .observer import CUE_LABELS, ceilings
    print(describe(cfg))
    print("\n  shortcut ceilings: the increment (dL, dB) at which an ideal observer of ONE cue "
          "reaches 79.4%")
    seen = set()
    for c in validate(cfg).conditions:
        if (c.task, c.kind) in seen:
            continue
        seen.add((c.task, c.kind))
        ce = ceilings(cfg, c)
        print(f"    {c.task} {c.kind:<8} exact: {', '.join(ce['exact'])}; lowest shortcut "
              f"{CUE_LABELS[ce['lowest_shortcut']]} at {ce['ceiling_db']:.1f} dB "
              f"(SRS {ce['ceiling_srs_db']:+.1f})")
    e = duration_estimate(cfg, n_sim=args.n_sim)
    print(f"\n  duration   {e['n_trials_median']:.0f} trials ({e['n_trials_range'][0]}-"
          f"{e['n_trials_range'][1]} over {e['n_sim']} simulated sessions), "
          f"{e['trials_per_track_median']:.0f} per track, {e['n_catch_median']:.0f} catch, "
          f"{e['n_filler_median']:.0f} fillers")
    print(f"             {e['total_minutes']:.0f} min median, {e['p90_total_minutes']:.0f} min "
          f"90th percentile: {e['main_minutes']:.0f} main + {e['rest_minutes']:.1f} rests + "
          f"{e['practice_minutes']:.1f} practice + {e['familiarise_minutes']:.0f} "
          f"familiarisation + {e['instructions_minutes']:.0f} instructions")
    print(f"             with three practice rounds: "
          f"{e['worst_practice_total_minutes']:.0f} min at the 90th percentile")
    a = audit_design(cfg, make_design(cfg, args.code, 1))
    print(f"\n  order      {a['n_tracks']} tracks, cycles of random permutations; serial-"
          f"position spread {a['serial_position_spread']:.3f}; catch {a['catch_rate_planned']:.1%}")


def cmd_verify(args):
    cfg = load_config(args)
    from .verify import format_report, run_battery
    battery = run_battery(cfg, quick=args.quick)
    text = format_report(cfg, battery)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    sys.exit(1 if any(not c.passed for v in battery.values() for c in v) else 0)


def cmd_model(args):
    cfg = load_config(args)
    from .model import predictions
    p = predictions(cfg)
    print("target-to-figure coherence rho_T, by step (ms): "
          + ", ".join(f"{s:g}" for s in cfg.steps_ms))
    for v, d in p["variants"].items():
        print(f"  {v:<10} " + " ".join(f"{x:.3f}" for x in d["curve"])
              + f"   half-way {d['half_way_ms']:.1f} ms   capture rho {d['capture_rho']:.3f} "
              f"(= {d['capture_equivalent_ms']:.1f} ms of shear)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(p, indent=1))


def cmd_demo(args):
    import numpy as np
    import soundfile as sf
    cfg = load_config(args)
    from .config import srs_db
    from .stimulus import build_trial, render_trial, to_output
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for c in validate(cfg).conditions:
        tr = build_trial(cfg, c, srs_db(args.delta), rng, target_position=1)
        p = out / f"tcbound_{c.name}_dL{args.delta:g}dB.wav"
        sf.write(p, to_output(cfg, render_trial(cfg, tr)), cfg.sample_rate)
        print(f"  {p}   target in interval 1, increment {tr.delta_db:.1f} dB")


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
           auto=args.auto, seed=args.seed).run(code=args.code, resume=args.resume,
                                               session_index=args.session)


def cmd_analyze(args):
    from .analysis import analyse_session, format_session
    texts = [format_session(analyse_session(Path(d))) for d in args.dirs]
    text = "\n\n".join(texts)
    print(text)
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "session_report.txt").write_text(text)
        from .plots import session_figure
        for d in args.dirs:
            p = session_figure(Path(d), Path(args.out))
            print(f"  {p}")


def cmd_group(args):
    from .analysis import format_group, group
    g = group([Path(d) for d in args.dirs])
    text = format_group(g)
    print(text)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "group_report.txt").write_text(text)
        from .plots import group_figure
        print(f"  {group_figure(g, out)}")


def cmd_plots(args):
    cfg = load_config(args)
    from .plots import write_all
    for p in write_all(cfg, Path(args.out), dirs=[Path(x) for x in args.dirs or []]):
        print(f"  {p}")


def cmd_export(args):
    from .export import export_session
    for p in export_session(Path(args.dir), Path(args.out), wav=args.wav):
        print(f"  {p}")


def cmd_power(args):
    cfg = load_config(args)
    from .power import run
    r = run(cfg, n_studies=args.studies, n_listeners=args.listeners, gain=args.gain)
    p = r["power"]
    print(f"\n  {r['n_listeners']} listeners, two sessions each, gain {r['gain']:g}, "
          f"{r['n_studies']} studies")
    print(f"  power: H1 {p['H1']:.0%}  H2 {p['H2']:.0%}  H3 {p['H3']:.0%}  H3b {p['H3b']:.0%}")
    print(f"  bound: median {r['bound_median_ms']:.1f} ms (IQR {r['bound_iqr_ms'][0]:.1f} to "
          f"{r['bound_iqr_ms'][1]:.1f}); alpha median {r['alpha_median']:.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tcbound", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("design"); add_common(q); q.add_argument("--code", default="EXAMPLE")
    q.add_argument("--n-sim", type=int, default=8); q.set_defaults(func=cmd_design)
    q = sub.add_parser("verify"); add_common(q); q.add_argument("--out")
    q.add_argument("--quick", action="store_true"); q.set_defaults(func=cmd_verify)
    q = sub.add_parser("model"); add_common(q); q.add_argument("--out")
    q.set_defaults(func=cmd_model)
    q = sub.add_parser("demo"); add_common(q)
    q.add_argument("--out", default="demo"); q.add_argument("--delta", type=float, default=8.0)
    q.add_argument("--seed", type=int, default=0); q.set_defaults(func=cmd_demo)
    q = sub.add_parser("calibrate"); add_common(q)
    q.add_argument("--data", default="data"); q.add_argument("--device")
    q.set_defaults(func=cmd_calibrate)
    q = sub.add_parser("run"); add_common(q)
    q.add_argument("--data", default="data"); q.add_argument("--code")
    q.add_argument("--device"); q.add_argument("--no-audio", action="store_true")
    q.add_argument("--resume", action="store_true"); q.add_argument("--session", type=int)
    q.add_argument("--auto", choices=["coherence", "late", "null", "within", "fused"])
    q.add_argument("--seed", type=int, default=0)
    q.set_defaults(func=cmd_run)
    q = sub.add_parser("analyze"); add_common(q)
    q.add_argument("dirs", nargs="+"); q.add_argument("--out")
    q.set_defaults(func=cmd_analyze)
    q = sub.add_parser("group"); q.add_argument("dirs", nargs="+"); q.add_argument("--out")
    q.set_defaults(func=cmd_group)
    q = sub.add_parser("plots"); add_common(q)
    q.add_argument("--out", default="verification"); q.add_argument("dirs", nargs="*")
    q.set_defaults(func=cmd_plots)
    q = sub.add_parser("export"); q.add_argument("dir"); q.add_argument("--out", required=True)
    q.add_argument("--wav", action="store_true"); q.set_defaults(func=cmd_export)
    q = sub.add_parser("power"); add_common(q)
    q.add_argument("--gain", type=float, default=0.5)
    q.add_argument("--listeners", type=int, default=12)
    q.add_argument("--studies", type=int, default=20)
    q.set_defaults(func=cmd_power)
    args = ap.parse_args(argv)
    try:
        args.func(args)
    except ConfigError as e:
        raise SystemExit(f"configuration error: {e}")


if __name__ == "__main__":
    main()
