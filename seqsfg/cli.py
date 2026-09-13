"""Command line: config | verify | demo | calibrate | run | analyze | participants."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import VARIANTS, Config, ConfigError, DEFAULT, describe, validate


def apply_sets(cfg: Config, args) -> Config:
    """Apply every --set KEY=VALUE to a Config.

    Split out of load_config because the preset-based commands build their Config from the
    preset file instead, and used to drop --set on the floor without saying so -- so
    `--preset x.json --set tones_per_channel=64` quietly ran at the preset's density.
    """
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


def load_config(args) -> Config:
    cfg = DEFAULT
    if getattr(args, "config", None):
        with open(args.config) as f:
            cfg = Config.from_dict(json.load(f))
    return apply_sets(cfg, args)


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


def cmd_train(args):
    from .runner import Audio, QuitRequested, getkey
    from .train import run_training
    cfg = load_config(args)
    validate(cfg)
    audio = Audio(cfg.sample_rate, args.device, enabled=not args.no_audio)

    def pause(msg):
        print(msg)
        return getkey({" ", "s", "q"})

    stages = None
    if args.stages:
        easiest = float(cfg.steps_ms[0])
        stages = tuple((v.strip(), easiest) for v in args.stages.split(",") if v.strip())
        for v, _ in stages:
            if v not in VARIANTS:
                raise SystemExit(f"unknown stage {v!r}; choose from {', '.join(VARIANTS)}")
    try:
        run_training(cfg, audio, stages=stages, per_level=args.per_level,
                     criterion=args.criterion, start_level=args.start_level,
                     getkey=getkey, pause=pause)
    except (QuitRequested, KeyboardInterrupt):
        print("\ntraining stopped. Nothing here is recorded, so just run it again when you want.")


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


def cmd_yesno_verify(args):
    from . import yesno
    cfg = load_config(args)
    validate(cfg)
    kinds = [k.strip() for k in args.absent.split(",")] if args.absent != "all" else list(yesno.ABSENT_KINDS)
    for k in kinds:
        if k not in yesno.ABSENT_KINDS:
            raise SystemExit(f"unknown absent class {k!r}; choose from {', '.join(yesno.ABSENT_KINDS)}")
        res = yesno.run_audit(cfg, n_trials=args.trials, seed=args.seed, absent=k,
                              n_perm=args.perm, verbose=not args.quiet)
        text = yesno.report(res)
        print(text)
        if args.out:
            from pathlib import Path
            path = Path(args.out) if len(kinds) == 1 else Path(args.out).with_name(
                Path(args.out).stem + f"_{k}" + Path(args.out).suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text + "\n")
            print(f"written to {path}")


def cmd_yesno_run(args):
    from . import yesno
    cfg = load_config(args)
    validate(cfg)
    r = yesno.YesNoRunner(cfg, args.data, args.device, audio=not args.no_audio,
                          absent=args.absent, auto=args.auto, fast=args.fast)
    r.run(code=args.code, session_index=args.session)


def cmd_yesno_analyze(args):
    from . import yesno
    for s in args.sessions:
        print(yesno.analyse(s))



# ---- onset asynchrony -------------------------------------------------------
def _async_cfgs(args):
    from .asynchrony import AsyncConfig, check, load_preset
    if getattr(args, "preset", None):
        cfg, acfg = load_preset(args.preset)
        cfg = apply_sets(cfg, args)
    else:
        cfg, acfg = load_config(args), AsyncConfig()
    for kv in getattr(args, "async_set", None) or []:
        k, _, v = kv.partition("=")
        if not hasattr(acfg, k):
            raise SystemExit(f"unknown asynchrony parameter {k}")
        try:
            val = json.loads(v)
        except json.JSONDecodeError:
            val = v
        if isinstance(val, list):
            val = tuple(val)
        acfg = AsyncConfig(**{**acfg.to_dict(), k: val})
    validate(cfg)
    check(cfg, acfg)
    return cfg, acfg


def cmd_async_design(args):
    from .asynchrony import cells, duration_estimate, make_design, notes
    cfg, acfg = _async_cfgs(args)
    est = duration_estimate(cfg, acfg)
    dz = make_design(cfg, acfg, args.code or "P01", args.session or 1)
    print(f"participant {dz['participant_code']}  session {dz['session_index']}  "
          f"design {dz['design_hash']}")
    print(f"absent class         {acfg.absent_class}")
    print(f"ladder               " + ", ".join(f"{s:g}" for s in cfg.steps_ms) + " ms"
          f"   ({cfg.tone_dur_ms:g} ms tones, {cfg.n_components} components)")
    print(f"orders               " + ", ".join(acfg.orders) +
          "   (step 0 is one cell: at 0 ms every order is the same sound)")
    print(f"cells                {est['n_cells']} x {acfg.trials_per_cell} present "
          f"+ {acfg.trials_per_cell} absent = {est['n_main']} main trials")
    print(f"practice             {est['n_practice']} trials at {acfg.practice_step_ms:g} ms")
    print(f"one trial            {est['trial_s']:.2f} s")
    print(f"estimated session    {est['minutes']:.0f} min "
          f"({est['n_breaks']} breaks of {cfg.break_s:.0f} s)")
    from .asynchrony import power_estimate
    pw = power_estimate(cfg, acfg)
    print(f"\nwhat one session resolves (at a true d' of {pw['at_dprime']:.1f}):")
    print(f"  one cell's d'        SE {pw['se_cell']:.2f}")
    if pw['se_contrast'] == pw['se_contrast']:
        print(f"  the order contrast   SE {pw['se_contrast']:.2f}; one session catches an effect of "
              f"{pw['mde_contrast']:.2f} or larger at 80% power")
        for tgt in (0.5, 0.3):
            print(f"                       {pw['listeners_for'](tgt):>2d} listeners to catch {tgt:.1f}")
    print("  the ladder is the measurement; the limit and the contrast want several listeners.")
    for n in notes(cfg, acfg):
        print(f"note: {n}")
    if args.json:
        from .session import write_json
        write_json(Path(args.json), dz)
        print(f"design written to {args.json}")


def cmd_async_verify(args):
    from . import asynchrony as A
    cfg, acfg = _async_cfgs(args)
    kinds = list(A.ABSENT_KINDS) if args.absent == "all" else [k.strip() for k in args.absent.split(",")]
    for k in kinds:
        if k not in A.ABSENT_KINDS:
            raise SystemExit(f"unknown absent class {k!r}; choose from {', '.join(A.ABSENT_KINDS)}")
        res = A.run_audit(cfg, acfg, n_trials=args.trials, seed=args.seed, absent=k,
                          n_perm=args.perm, verbose=not args.quiet)
        text = A.report(res)
        print(text)
        if args.out:
            path = Path(args.out) if len(kinds) == 1 else Path(args.out).with_name(
                Path(args.out).stem + f"_{k}" + Path(args.out).suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text + "\n")
            print(f"written to {path}")


def cmd_async_run(args):
    from .asynchrony import AsyncRunner
    cfg, acfg = _async_cfgs(args)
    AsyncRunner(cfg, acfg, args.data, args.device, audio=not args.no_audio,
                auto=args.auto).run(code=args.code, session_index=args.session)


def cmd_async_analyze(args):
    from .asynchrony import analyse
    print(analyse(args.sessions))


def cmd_async_demo(args):
    import numpy as np
    import soundfile as sf
    from .asynchrony import build_interval, render, span_ms
    cfg, acfg = _async_cfgs(args)
    d = validate(cfg)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    steps = [float(s) for s in (args.steps.split(",") if args.steps else cfg.steps_ms)]
    for step in steps:
        for present in (True, False):
            iv = build_interval(cfg, d, args.seed, step, args.order, present,
                                acfg.absent_class, acfg.absent_order)
            name = out / f"step{step:g}_{'figure' if present else 'no-figure'}.wav"
            sf.write(name, render(cfg, d, iv), cfg.sample_rate)
            if present:
                S = iv.figure_set
                print(f"{step:5g} ms  channels {S.tolist()} -> "
                      f"{np.round(d.channel_freqs_hz[S]).astype(int).tolist()} Hz, "
                      f"element spans {span_ms(cfg, step):.0f} ms")
    print(f"wrote {2 * len(steps)} files to {out}  (order '{args.order}', "
          f"absent '{acfg.absent_class}', same seed throughout so the figure is the same set)")


# ---- temporal overlap pilot -------------------------------------------------
def _overlap_cfgs(args):
    from .overlap import OverlapConfig, check, load_preset
    if getattr(args, "preset", None):
        cfg, ocfg = load_preset(args.preset)
        cfg = apply_sets(cfg, args)
    else:
        cfg, ocfg = load_config(args), OverlapConfig()
    for kv in getattr(args, "overlap_set", None) or []:
        k, _, v = kv.partition("=")
        if not hasattr(ocfg, k):
            raise SystemExit(f"unknown overlap parameter {k}")
        try:
            val = json.loads(v)
        except json.JSONDecodeError:
            val = v
        if isinstance(val, list):
            val = tuple(tuple(x) if isinstance(x, list) else x for x in val)
        ocfg = OverlapConfig(**{**ocfg.to_dict(), k: val})
    validate(cfg)
    check(cfg, ocfg)
    return cfg, ocfg


def cmd_overlap_design(args):
    from .overlap import condition_table, duration_estimate, make_design, notes, widest_footprint_ms
    cfg, ocfg = _overlap_cfgs(args)
    dz = make_design(cfg, ocfg, args.code or "P01", args.session or 1)
    est = duration_estimate(cfg, ocfg)
    print(f"participant {dz['participant_code']}  session {dz['session_index']}  "
          f"design {dz['design_hash']}")
    print(f"absent class         {ocfg.absent_class}")
    print(f"components           {cfg.n_components}, {cfg.n_elements} recurrences, "
          f"one arbitrary order per trial reused by every recurrence")
    print(f"background           {cfg.tone_dur_ms:g} ms tones, {cfg.tones_per_channel} per "
          f"channel, amplitude {cfg.tone_amplitude} -- fixed in every cell")
    print(f"timing               recurrence every {cfg.iei_min_ms:.0f}-{cfg.iei_max_ms:.0f} ms, "
          f"{ocfg.jitter_ms:.0f} ms shared jitter, scene {cfg.interval_dur_ms:.0f} ms")
    print(f"                     widest cell needs {widest_footprint_ms(cfg, ocfg):.0f} ms per "
          f"recurrence; the same schedule is used for all {est['n_cells']}")
    print(f"trials               {est['n_cells']} cells x {ocfg.trials_per_cell} present "
          f"+ {ocfg.trials_per_cell} absent = {est['n_main']}, plus {est['n_practice']} practice")
    print(f"estimated session    {est['minutes']:.0f} min")
    print()
    print(condition_table(cfg, ocfg))
    for n in notes(cfg, ocfg):
        print(f"\nnote: {n}")
    if args.json:
        from .session import write_json
        write_json(Path(args.json), dz)
        print(f"\ndesign written to {args.json}")


def cmd_overlap_verify(args):
    from . import overlap as O
    cfg, ocfg = _overlap_cfgs(args)
    res = O.run_audit(cfg, ocfg, n_trials=args.trials, seed=args.seed, n_perm=args.perm,
                      verbose=not args.quiet)
    text = O.report(res)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"written to {args.out}")


def cmd_overlap_run(args):
    from .overlap import OverlapRunner
    cfg, ocfg = _overlap_cfgs(args)
    OverlapRunner(cfg, ocfg, args.data, args.device, audio=not args.no_audio,
                  auto=args.auto).run(code=args.code, session_index=args.session)


def cmd_overlap_analyze(args):
    from .overlap import analyse
    print(analyse(args.sessions))


def cmd_overlap_demo(args):
    import soundfile as sf
    from .overlap import build_interval, cell_name, geometry, render
    cfg, ocfg = _overlap_cfgs(args)
    d = validate(cfg)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for (T, s) in ocfg.cells:
        for present in (True, False):
            iv = build_interval(cfg, ocfg, d, args.seed, T, s, present)
            name = out / f"{cell_name(T, s)}_{'pattern' if present else 'no-pattern'}.wav"
            sf.write(name, render(cfg, ocfg, d, iv), cfg.sample_rate)
        g = geometry(cfg, T, s, cfg.n_components)
        print(f"  {cell_name(T, s):>10}  adjacent overlap {g['adjacent_overlap_ms']:.0f} ms "
              f"({g['adjacent_overlap_fraction']:.0%}), extent {g['total_extent_ms']:.0f} ms, "
              f"envelope overlap {g['envelope_overlap_ms']:.2f} ms")
    print(f"wrote {2 * len(ocfg.cells)} full-mixture files to {out} "
          f"(absent '{ocfg.absent_class}', one seed throughout)")


def _exposure_cfgs(args):
    from .exposure import ExposureConfig, load_preset
    if getattr(args, "preset", None):
        cfg, ecfg = load_preset(args.preset)
        cfg = apply_sets(cfg, args)
    else:
        cfg, ecfg = load_config(args), ExposureConfig()
    for kv in getattr(args, "exposure_set", None) or []:
        k, _, v = kv.partition("=")
        if not hasattr(ecfg, k):
            raise SystemExit(f"unknown exposure parameter {k}")
        try:
            val = json.loads(v)
        except json.JSONDecodeError:
            val = v
        if isinstance(val, list):
            val = tuple(val)
        ecfg = ExposureConfig(**{**ecfg.to_dict(), k: val})
    validate(cfg)
    return cfg, ecfg


def cmd_exposure_design(args):
    from .exposure import check, duration_estimate, heard_order, make_design
    cfg, ecfg = _exposure_cfgs(args)
    check(cfg, ecfg)
    dz = make_design(cfg, ecfg, args.code or "P01", args.session or 1)
    est = duration_estimate(cfg, ecfg)
    print(f"participant {dz['participant_code']}  session {dz['session_index']}")
    print(f"  trained    {dz['trained']}  onset order {dz['orders'][dz['trained']]} "
          f"heard {dz['heard_order'][dz['trained']]}")
    print(f"  comparison {dz['untrained']}  onset order {dz['orders'][dz['untrained']]} "
          f"heard {dz['heard_order'][dz['untrained']]}")
    print("  order overlap:")
    for k, v in dz["order_overlap"].items():
        print(f"      {k:<36} {v}")
    print(f"  figure set (channels): {dz['figure_set']}")
    for ph in ("practice", "pre", "exposure", "post"):
        rows = dz["phases"][ph]
        print(f"  {ph:<9} {len(rows):>4} trials", end="")
        if rows:
            pres = sum(r["present"] for r in rows)
            seqs = {s: sum(1 for r in rows if r["sequence"] == s) for s in ("P", "Q")}
            print(f"   present {pres}/{len(rows)}   P {seqs['P']}  Q {seqs['Q']}")
        else:
            print("")
    print(f"  estimated session {est['minutes']:.0f} min ({est['trials']} trials at "
          f"{est['trial_seconds']:.1f} s plus breaks)")
    print(f"  config hash {cfg.hash()}   exposure hash {ecfg.hash()}   design hash {dz['design_hash']}")


def cmd_exposure_verify(args):
    from .exposure import audit, audit_report
    cfg, ecfg = _exposure_cfgs(args)
    res = audit(cfg, ecfg, n_trials=args.trials, seed=args.seed, n_perm=args.perm,
                verbose=not args.quiet)
    text = audit_report(res)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"written to {args.out}")


def cmd_exposure_run(args):
    from .exposure import ExposureRunner
    cfg, ecfg = _exposure_cfgs(args)
    ExposureRunner(cfg, ecfg, args.data, device=args.device, audio=not args.no_audio,
                   auto=args.auto, fast=args.fast).run(code=args.code, session_index=args.session)


def cmd_exposure_analyze(args):
    from .exposure import analyse
    print(analyse([Path(p) for p in args.sessions]))


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

    q = sub.add_parser("train", help="progressive training: learn what the figure sounds like"); add_common(q)
    q.add_argument("--device"); q.add_argument("--no-audio", action="store_true")
    q.add_argument("--per-level", type=int, default=5, help="trials at each background level")
    q.add_argument("--criterion", type=int, default=4, help="correct needed to move a level harder")
    q.add_argument("--stages", help="comma-separated variants to train on, e.g. 'rising' for the "
                                    "repeated-vs-different-pitch task only (default: the config's "
                                    "practice stages)")
    q.add_argument("--start-level", type=int, default=0,
                   help="skip straight to this background level (0 = easiest)")
    q.set_defaults(fn=cmd_train)

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

    q = sub.add_parser("yesno-verify", help="single-interval yes/no: can anything separate present from absent?")
    add_common(q)
    q.add_argument("--absent", default="roving",
                   help="absent class: roving | scattered | plain | all (default: roving)")
    q.add_argument("--trials", type=int, default=60, help="present AND absent trials per step")
    q.add_argument("--seed", type=int, default=4242)
    q.add_argument("--perm", type=int, default=20000)
    q.add_argument("--quiet", action="store_true")
    q.add_argument("--out", help="write the report here")
    q.set_defaults(fn=cmd_yesno_verify)

    q = sub.add_parser("yesno-run", help="run a single-interval yes/no session"); add_common(q)
    q.add_argument("--data", default="data"); q.add_argument("--code")
    q.add_argument("--session", type=int); q.add_argument("--device")
    q.add_argument("--no-audio", action="store_true")
    q.add_argument("--absent", default="roving", help="absent class (default: roving)")
    q.add_argument("--auto", type=float, default=None, metavar="TAU_MS",
                   help="simulated listener; pipeline test only")
    q.add_argument("--fast", action="store_true")
    q.set_defaults(fn=cmd_yesno_run)

    q = sub.add_parser("yesno-analyze", help="d', criterion and bias diagnostics for a yes/no session")
    q.add_argument("sessions", nargs="+"); q.set_defaults(fn=cmd_yesno_analyze)

    def add_exposure(q, with_preset=True):
        add_common(q)
        if with_preset:
            q.add_argument("--preset", default="exposure_pilot.json",
                           help="two-section preset file (config + exposure)")
        q.add_argument("--exposure-set", action="append", metavar="KEY=VALUE",
                       help="override one exposure parameter")

    q = sub.add_parser("exposure-design", help="print the three-phase design and its duration")
    add_exposure(q); q.add_argument("--code"); q.add_argument("--session", type=int)
    q.set_defaults(fn=cmd_exposure_design)

    q = sub.add_parser("exposure-verify", help="audit this mode's stimulus distributions")
    add_exposure(q)
    q.add_argument("--trials", type=int, default=50); q.add_argument("--seed", type=int, default=808)
    q.add_argument("--perm", type=int, default=8000); q.add_argument("--quiet", action="store_true")
    q.add_argument("--out"); q.set_defaults(fn=cmd_exposure_verify)

    q = sub.add_parser("exposure-run", help="run a pre / exposure / post session")
    add_exposure(q)
    q.add_argument("--data", default="data"); q.add_argument("--code")
    q.add_argument("--session", type=int); q.add_argument("--device")
    q.add_argument("--no-audio", action="store_true")
    q.add_argument("--auto", type=float, default=None, metavar="TAU_MS",
                   help="simulated listener; pipeline test only")
    q.add_argument("--fast", action="store_true"); q.set_defaults(fn=cmd_exposure_run)

    q = sub.add_parser("exposure-analyze", help="cell table, D per delay, participant summary")
    q.add_argument("sessions", nargs="+"); q.set_defaults(fn=cmd_exposure_analyze)

    def add_async(q):
        add_common(q)
        q.add_argument("--preset", default="asynchrony_config.json",
                       help="two-section preset file (config + asynchrony)")
        q.add_argument("--async-set", action="append", metavar="KEY=VALUE",
                       help="override one asynchrony parameter")

    q = sub.add_parser("asynchrony-design", help="print the asynchrony design and its duration")
    add_async(q); q.add_argument("--code"); q.add_argument("--session", type=int)
    q.add_argument("--json"); q.set_defaults(fn=cmd_async_design)

    q = sub.add_parser("asynchrony-verify",
                       help="can anything separate figure from no-figure without hearing it?")
    add_async(q)
    q.add_argument("--trials", type=int, default=40); q.add_argument("--seed", type=int, default=4242)
    q.add_argument("--perm", type=int, default=20000)
    q.add_argument("--absent", default="roving",
                   help="roving | incoherent | plain | all (comma-separated)")
    q.add_argument("--quiet", action="store_true"); q.add_argument("--out")
    q.set_defaults(fn=cmd_async_verify)

    q = sub.add_parser("asynchrony-run", help="run a single-interval onset-asynchrony session")
    add_async(q)
    q.add_argument("--data", default="data"); q.add_argument("--code")
    q.add_argument("--session", type=int); q.add_argument("--device")
    q.add_argument("--no-audio", action="store_true")
    q.add_argument("--auto", type=float, default=None, metavar="TAU_MS",
                   help="simulated listener; pipeline test only")
    q.set_defaults(fn=cmd_async_run)

    q = sub.add_parser("asynchrony-analyze", help="the ladder, the limit, and the order contrast")
    q.add_argument("sessions", nargs="+"); q.set_defaults(fn=cmd_async_analyze)

    q = sub.add_parser("asynchrony-demo", help="write one figure and one no-figure WAV per step")
    add_async(q)
    q.add_argument("--out", default="demo/asynchrony"); q.add_argument("--seed", type=int, default=7)
    q.add_argument("--order", default="fixed"); q.add_argument("--steps",
                   help="comma-separated, default the whole ladder")
    q.set_defaults(fn=cmd_async_demo)

    def add_overlap(q):
        add_common(q)
        q.add_argument("--preset", default="overlap_pilot.json",
                       help="two-section preset file (config + overlap)")
        q.add_argument("--overlap-set", action="append", metavar="KEY=VALUE",
                       help="override one overlap parameter")

    q = sub.add_parser("overlap-design", help="the seven cells, their geometry and the duration")
    add_overlap(q); q.add_argument("--code"); q.add_argument("--session", type=int)
    q.add_argument("--json"); q.set_defaults(fn=cmd_overlap_design)

    q = sub.add_parser("overlap-verify", help="matching, placement and the acoustic cue audit")
    add_overlap(q)
    q.add_argument("--trials", type=int, default=40); q.add_argument("--seed", type=int, default=606)
    q.add_argument("--perm", type=int, default=20000); q.add_argument("--quiet", action="store_true")
    q.add_argument("--out"); q.set_defaults(fn=cmd_overlap_verify)

    q = sub.add_parser("overlap-run", help="run a temporal overlap pilot session")
    add_overlap(q)
    q.add_argument("--data", default="data"); q.add_argument("--code")
    q.add_argument("--session", type=int); q.add_argument("--device")
    q.add_argument("--no-audio", action="store_true")
    q.add_argument("--auto", type=float, default=None, metavar="TAU_MS",
                   help="simulated listener; pipeline test only")
    q.set_defaults(fn=cmd_overlap_run)

    q = sub.add_parser("overlap-analyze", help="per-cell d', the three planned contrasts, caveats")
    q.add_argument("sessions", nargs="+"); q.set_defaults(fn=cmd_overlap_analyze)

    q = sub.add_parser("overlap-demo", help="full-mixture present and absent audio for every cell")
    add_overlap(q)
    q.add_argument("--out", default="demo/overlap"); q.add_argument("--seed", type=int, default=12)
    q.set_defaults(fn=cmd_overlap_demo)

    args = p.parse_args(argv)
    try:
        args.fn(args)
    except ConfigError as e:
        print(e, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
