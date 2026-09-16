"""Command line for the two-tone coherence task.

    python -m tcoh design                      the conditions, the geometry, the duration
    python -m tcoh verify                      every check that does not need a listener
    python -m tcoh model                       the prediction, and the check against the paper
    python -m tcoh demo --out demo/            write WAV files, one per condition
    python -m tcoh calibrate                   loop the reference tone
    python -m tcoh run --data data             run a session
    python -m tcoh analyze data/P01/tcoh_session_01
    python -m tcoh plots --out verification/
    python -m tcoh power                       what this design can and cannot detect
    python -m tcoh simulate --mode coherence   a dataset with no listener, for the pipeline
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
            val = tuple(val)
        cfg = cfg.replace(**{k: val})
    return cfg


def add_common(p):
    p.add_argument("--config", help="JSON file with parameter overrides")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override one parameter (JSON value); repeatable")


# ----------------------------------------------------------------------------
def cmd_design(args):
    cfg = load_config(args)
    from .design import audit_design, duration_estimate, make_design
    print(describe(cfg))
    e = duration_estimate(cfg)
    print(f"\n  duration   {e['n_trials']} trials, {e['median_trials_per_track']:.0f} per track, "
          f"{e['total_minutes']:.0f} min total "
          f"({e['main_minutes']:.0f} main + {e['practice_minutes']:.0f} practice + "
          f"{e['break_minutes']:.0f} breaks)")
    print(f"             at 50 minutes a sitting that is {e['sessions_at_50_min']} session(s)")
    a = audit_design(cfg, make_design(cfg, args.code, 1))
    print(f"\n  order      {a['n_tracks']} tracks; every condition once per round, so one track in "
          f"each {1 / cfg.tracks_per_condition:.0%} of the session")
    print(f"             within-round serial-position spread {a['serial_position_spread']:.3f}; "
          f"longest run of one condition {a['longest_condition_run']} "
          f"(limit {a['max_same_condition_run']})")
    if args.json:
        print(json.dumps({"config": cfg.to_dict(), "duration": e, "order": a}, indent=1,
                         default=str))


def cmd_verify(args):
    cfg = load_config(args)
    from .verify import format_report, run_battery
    battery = run_battery(cfg, quick=not args.slow)
    text = format_report(cfg, battery)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"\nwritten to {args.out}")
    n_fail = sum(1 for v in battery.values() for c in v if not c.passed)
    sys.exit(1 if n_fail else 0)


def cmd_model(args):
    cfg = load_config(args)
    from .model import duty_cycle_scan, prediction_band, reproduce_figure8, PUBLISHED
    d = validate(cfg)
    r = reproduce_figure8()
    print("check against the published Figure 8 (300 / 952 Hz, 75 ms tones):")
    print(f"   {'reading':<12} {'dT=100%':>9} {'dT=0%':>8}  monotone")
    for k, v in r["readings"].items():
        print(f"   {k:<12} {v['alternating']:9.3f} {v['synchronous']:8.4f}  {v['monotone']}")
    print(f"   {'published':<12} {PUBLISHED['alternating']:9.3f} {PUBLISHED['synchronous']:8.4f}")
    print(f"   -> this package uses '{r['best']}'; see the note in tcoh/model.py\n")

    print("is dT a monotone axis at each duty cycle?")
    for duty, v in duty_cycle_scan(soa_ms=cfg.soa_ms, n_tones=cfg.n_tones).items():
        print(f"   tone/soa = {duty:5.3f}   {'monotone' if v['monotone'] else 'NOT monotone'}"
              + ("" if v["monotone"] else f", peaking at dT={v['argmax_pct']:.0f}%"))
    print(f"   -> this config uses {cfg.duty:.3f}\n")

    pcts = sorted({c.lag_pct for c in d.conditions if c.a_kind == "coherent"})
    band = prediction_band(pcts, tone_ms=cfg.tone_ms, soa_ms=cfg.soa_ms, n_tones=cfg.n_tones)
    print("the prediction for THIS stimulus:")
    print("   dT%   index   range across 8 readings")
    for i, p in enumerate(pcts):
        print(f"   {p:5.1f}  {band['mid'][i]:.3f}   {band['lo'][i]:.3f} - {band['hi'][i]:.3f}")
    print(f"   all monotone: {band['all_monotone']}   all agreeing on the order: "
          f"{band['all_agree_on_order']}")
    print("\nthe model index for every condition as built:")
    for c in d.conditions:
        v = d.model_by_condition[c.name]
        print(f"   {c.name:<13} {c.a_kind:<11} " + ("  -" if v != v else f"{v:.3f}"))


def cmd_demo(args):
    import numpy as np
    import soundfile as sf
    cfg = load_config(args)
    d = validate(cfg)
    from .stimulus import build_trial, render_trial, to_output
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    made = []
    for c in d.conditions:
        tr = build_trial(cfg, c, args.delta, rng, target_position=1, direction=+1)
        x = to_output(cfg, render_trial(cfg, tr, d))
        p = out / f"tcoh_{c.name}_delta{args.delta:g}ms.wav"
        sf.write(p, x, cfg.sample_rate)
        made.append((p, tr))
    for p, tr in made:
        print(f"  {p}   target interval {tr.target_position}, shift "
              f"{tr.delta_signed_ms:+.2f} ms ({'late' if tr.delta_signed_ms > 0 else 'early'})")
    print(f"\n{len(made)} files. In every one the FIRST interval is the target: its last high "
          f"tone is {args.delta:g} ms late.")


def cmd_calibrate(args):
    """Loop the reference tone, read the machine's output state, and do the arithmetic."""
    import math
    import numpy as np
    cfg = load_config(args)
    from .audiolevel import amplitude_for, describe_output, scene_db_spl, system_output
    from .runner import Audio, ask, getkey
    from .stimulus import to_output
    d = validate(cfg)

    rms_db = 20 * math.log10(cfg.tone_amplitude / math.sqrt(2))
    print(f"\n--- level calibration ---")
    print(f"  reference   {cfg.f_a_hz:.0f} Hz at one stimulus tone's amplitude "
          f"({cfg.tone_amplitude:g} FS peak, {rms_db:.1f} dB FS rms)")
    print(f"  target      {cfg.tone_level_db_spl:.0f} dB SPL for ONE tone; the two together come "
          f"to {scene_db_spl(cfg.tone_level_db_spl):.0f} dB SPL")
    print(f"  right now   {describe_output()}")
    if cfg.monaural:
        print("  NOTE: this configuration is monaural -- LEFT earpiece only.")
    print("\n  Put the headphones on a coupler or an in-ear probe, or hold a phone SPL app at the")
    print("  earpiece. Set the system volume, then leave it alone: it is what the measurement is")
    print("  about, and `tcoh run` records it and warns if it moves.")

    a = Audio(cfg.sample_rate, args.device)
    t = np.arange(int(args.seconds * cfg.sample_rate)) / cfg.sample_rate
    x = cfg.tone_amplitude * np.sin(2 * np.pi * cfg.f_a_hz * t)
    while True:
        print(f"\n  space = play {args.seconds:g} s, m = enter a measurement, q = quit")
        k = getkey({" ", "m", "q"})
        if k == "q":
            return
        if k == " ":
            s_ = system_output()
            if s_.get("muted"):
                print("  output is MUTED.")
                continue
            print(f"  playing... ({describe_output(s_)})")
            a.play(to_output(cfg, x))
            continue
        v = ask("  measured dB SPL for ONE tone")
        try:
            measured = float(v)
        except ValueError:
            print("  not a number")
            continue
        want = amplitude_for(cfg.tone_level_db_spl, measured, cfg.tone_amplitude)
        off = measured - cfg.tone_level_db_spl
        sysnow = system_output()
        print(f"\n  measured    {measured:.1f} dB SPL  ({off:+.1f} dB from target)")
        print(f"  scene       {scene_db_spl(measured):.1f} dB SPL with both tones sounding")
        print(f"  at          {describe_output(sysnow)}")
        if abs(off) <= 1.0:
            print("  -> within 1 dB. Leave the volume where it is and run the session.")
        else:
            print(f"  -> to hit {cfg.tone_level_db_spl:.0f} dB exactly, keep the volume where it is "
                  f"and set\n       tone_amplitude = {want:.4f}   "
                  f"({20 * math.log10(want / cfg.tone_amplitude):+.1f} dB)")
            print(f"     python -m tcoh run --config <preset> --set tone_amplitude={want:.4f} ...")
            print("     Digital scaling is linear, so that is exact. Moving the system volume "
                  "would\n     also work, but it is an undocumented taper and you would have to "
                  "measure again.")
        if scene_db_spl(measured) > 80:
            print(f"  WARNING: {scene_db_spl(measured):.0f} dB SPL for a session of this length is "
                  "louder than this experiment needs. Turn it down.")


def cmd_run(args):
    cfg = load_config(args)
    from .runner import Runner
    Runner(cfg, Path(args.data), device=args.device, audio=not args.no_audio,
           auto=args.auto).run(code=args.code, resume=args.resume,
                               session_index=args.session)


def cmd_analyze(args):
    cfg = load_config(args) if args.config or args.set else None
    from .analysis import analyse
    text = analyse([Path(p) for p in args.dirs], cfg=cfg, n_boot=args.boot)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"\nwritten to {args.out}")


def cmd_plots(args):
    cfg = load_config(args)
    from .analysis import coherence_index, interaction_test, load, thresholds
    from .plots import write_all
    rows = idx = ctl = inter = None
    simulated = False
    if args.dirs:
        rows, metas = load([Path(p) for p in args.dirs])
        if metas and metas[0].get("config") and not (args.config or args.set):
            cfg = Config.from_dict(metas[0]["config"])
        simulated = any(m.get("auto") for m in metas)
        res = thresholds(rows, cfg)
        if cfg.include_b_only:
            idx = coherence_index(res, cfg, n_boot=args.boot)
            ctl = coherence_index(res, cfg, n_boot=args.boot, a_kind="scrambled")
            inter = interaction_test(res, cfg, n_boot=args.boot)
        else:
            # a descriptive preset has no ceiling to normalise against, so there is no kappa to
            # plot and the output is the threshold curve itself
            from .plots import pilot_curve
            out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
            pilot_curve(cfg, res, path=out / "tcoh_pilot_curve.png", simulated=simulated)
            print(" ", out / "tcoh_pilot_curve.png")
    for p in write_all(cfg, Path(args.out), rows, idx, ctl, inter, simulated=simulated):
        print(" ", p)


def cmd_power(args):
    cfg = load_config(args)
    from .analysis import power
    from .design import duration_estimate
    e = duration_estimate(cfg)
    r = power(cfg, n_sessions=args.sessions, n_boot=args.boot, floor_ms=args.floor,
              ceiling_ms=args.ceiling, sigma=args.sigma)
    print(f"{args.sessions} simulated sessions per generative truth; a session is "
          f"{e['n_trials']} trials, about {e['total_minutes']:.0f} min")
    print(f"floor {args.floor:g} ms, ceiling {args.ceiling:g} ms, slope {args.sigma:g}, "
          f"{cfg.tracks_per_condition} tracks per condition\n")
    print(f"  {'generative truth':<12} {'H1 fires':>10} {'H2 fires':>10}   what that should be")
    want = {"coherence": "both high", "pedestal": "H1 high, H2 at 5%", "null": "both at 5%"}
    for m, v in r["modes"].items():
        print(f"  {m:<12} {v['H1_rate']:10.0%} {v['H2_rate']:10.0%}   {want.get(m, '')}")
    print("\n" + r["note"])


def cmd_simulate(args):
    cfg = load_config(args)
    from .runner import Runner
    r = Runner(cfg, Path(args.data), audio=False, auto=args.mode, seed=args.seed)
    sdir = r.run(code=args.code)
    print(f"\nSIMULATED data written to {sdir}. Every report built from it is labelled as such.")


# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(prog="tcoh", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("design", help="the conditions, the geometry and the duration")
    add_common(q); q.add_argument("--code", default="P01"); q.add_argument("--json", action="store_true")
    q.set_defaults(func=cmd_design)

    q = sub.add_parser("verify", help="every check that does not need a listener")
    add_common(q); q.add_argument("--out"); q.add_argument("--slow", action="store_true",
                                                           help="include the power simulation")
    q.set_defaults(func=cmd_verify)

    q = sub.add_parser("model", help="the prediction, and the check against the published figure")
    add_common(q); q.set_defaults(func=cmd_model)

    q = sub.add_parser("demo", help="write one WAV per condition")
    add_common(q); q.add_argument("--out", default="demo"); q.add_argument("--delta", type=float, default=25.0)
    q.add_argument("--seed", type=int, default=1); q.set_defaults(func=cmd_demo)

    q = sub.add_parser("calibrate", help="loop the reference tone, read the output level, do the sums")
    add_common(q); q.add_argument("--device")
    q.add_argument("--seconds", type=float, default=5.0, help="length of each reference burst")
    q.set_defaults(func=cmd_calibrate)

    q = sub.add_parser("run", help="run a session")
    add_common(q); q.add_argument("--data", default="data"); q.add_argument("--code")
    q.add_argument("--device"); q.add_argument("--no-audio", action="store_true")
    q.add_argument("--resume", action="store_true"); q.add_argument("--session", type=int)
    q.add_argument("--auto", choices=("coherence", "pedestal", "overlap", "null"),
                   help="no listener: simulate one (the report says so)")
    q.set_defaults(func=cmd_run)

    q = sub.add_parser("analyze", help="thresholds, the coherence index, the tests")
    add_common(q); q.add_argument("dirs", nargs="+"); q.add_argument("--out")
    q.add_argument("--boot", type=int, default=4000); q.set_defaults(func=cmd_analyze)

    q = sub.add_parser("plots", help="write the figures")
    add_common(q); q.add_argument("dirs", nargs="*"); q.add_argument("--out", default="verification")
    q.add_argument("--boot", type=int, default=2000); q.set_defaults(func=cmd_plots)

    q = sub.add_parser("power", help="what this design can and cannot detect")
    add_common(q); q.add_argument("--sessions", type=int, default=60)
    q.add_argument("--boot", type=int, default=400); q.add_argument("--floor", type=float, default=3.0)
    q.add_argument("--ceiling", type=float, default=15.0); q.add_argument("--sigma", type=float, default=0.6)
    q.set_defaults(func=cmd_power)

    q = sub.add_parser("simulate", help="write a dataset with no listener, for the pipeline")
    add_common(q); q.add_argument("--data", default="data")
    q.add_argument("--mode", default="coherence", choices=("coherence", "pedestal", "overlap", "null"))
    q.add_argument("--code", default="SIM01"); q.add_argument("--seed", type=int, default=1)
    q.set_defaults(func=cmd_simulate)

    args = ap.parse_args(argv)
    try:
        args.func(args)
    except ConfigError as e:
        raise SystemExit(f"configuration error: {e}")


if __name__ == "__main__":
    main()
