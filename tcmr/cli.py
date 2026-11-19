"""python -m tcmr <command>: design | verify | check | demo | calibrate | run | simulate | analyze.

    check --seed N     every trial a session with seed N will play, drawn and checked (tcmr.audit);
                       `run --seed N` then refuses to start unless that check passed. Several
                       seeds (--seed N1 N2 ...) are checked in parallel.
    run --seed next    the lowest checked seed that no session of this configuration has used;
                       a seed another listener already used is refused (--reuse-seed overrides)
    calibrate          one tone per band on the meter. A calibrated seed check is tied to the
                       calibration it was made with: after calibrating again, check again.

A session whose listener does not pass the practice gate (practice_min_correct) stops after the
practice; `run` says so when it ends, and such a session cannot be resumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import secrets
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from .config import PATTERNS, Condition, Config, validate
from .stimulus import band_envelopes, build

PRESET = Path(__file__).resolve().parent / "configs" / "tcmr_main.json"
TEST_AMP = 0.05                  # peak of the calibration tones
NOISE_PEAK_PER_AMP = 2.8         # tallest noise peak / sum of the bands' 60-dB amplitudes, from the
                                 # uncalibrated checks of 27 exp1 seeds (0.28 above the tone's 0.36)
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


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


def source_state():
    """The tcmr code that is about to run: a hash over its .py and preset files, what git says
    about the tcmr folder, and its uncommitted diff. tsync's git_info looks at tsync alone, so a
    session used to be recorded as clean while tcmr had uncommitted changes."""
    from tsync.system import REPO, _run
    pkg = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for f in sorted(list(pkg.rglob("*.py")) + list(pkg.rglob("*.json"))):
        h.update(f.relative_to(pkg).as_posix().encode())
        h.update(f.read_bytes())
    status = _run(["git", "-C", str(REPO), "status", "--porcelain", "--", str(pkg)]) or ""
    diff = _run(["git", "-C", str(REPO), "diff", "HEAD", "--", str(pkg)]) or ""
    return {"source_tcmr": h.hexdigest(), "tcmr_dirty": bool(status),
            "tcmr_status": status}, diff


def new_session(cfg, data, code, meta_extra=None, seed=None):
    from tsync.design import clean_code
    from tsync.system import git_info, machine, source_hash, versions
    from .session import Writer
    code = clean_code(code)
    root = Path(data) / code
    n = 1 + max([int(p.name.rsplit("_", 1)[1]) for p in root.glob("tcmr_session_*")], default=0)
    seed = secrets.randbits(62) if seed is None else seed
    g = git_info()
    own, own_diff = source_state()
    meta = {"task": "tcmr", "code": code, "session_index": n, "config": cfg.to_dict(),
            "config_hash": cfg.hash(), "session_seed": seed, "git": {k: v for k, v in g.items()
                                                                     if k != "diff"},
            "source_tsync": source_hash(), **own, "versions": versions(), "machine": machine(),
            **(meta_extra or {})}
    w = Writer(root / f"tcmr_session_{n:02d}", meta)
    if g.get("diff") or own_diff:
        (w.folder / "code.patch").write_text((g.get("diff") or "") + own_diff)
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
        how = {"frozen": "shuffled, one order per interval", "reordered": "shuffled every repetition"}
        if c.arrangement in PATTERNS:
            print(f"  {c.name:<20} x{c.tracks}  offsets {sorted(off)} ms ({how[c.arrangement]}); "
                  f"smallest gap before the final burst {gap:g} ms")
        else:
            print(f"  {c.name:<20} x{c.tracks}  offsets {off} ms, smallest gap before the final "
                  f"burst {gap:g} ms")
    lv = ", ".join(f"{v:g}" for v in cfg.practice_levels_db)
    gate = (f"; gate: at least {cfg.practice_min_correct} correct, otherwise the session stops there"
            if cfg.practice_min_correct else "; no gate")
    print(f"practice: {len(cfg.practice_levels_db)} trials at {lv} dB{gate}")
    print(f"tone level: starts at {cfg.start_db:g} dB, between {cfg.min_db:g} and {cfg.max_db:g} dB")
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

    # the final burst's level in each band, measured in the rendered sound over 40 trials: the
    # same across comodulated bands, independent across random ones (any burst length)
    def final_levels(env, arr, dt):
        out = []
        for k in range(40):
            tr, buf = build(cfg, Condition(env, arr, dt), None, (1, 2, k), n_intervals=1)
            t0 = tr.target_on_s[0] + cfg.ramp_ms / 1000
            e = band_envelopes(cfg, buf, t0, (cfg.burst_ms - 2 * cfg.ramp_ms) / 1000)
            out.append(20 * np.log10(e.mean(axis=1)))
        return np.array(out)
    step = max([c.dt_ms for c in cfg.condition_list()] + [0.0])
    arr = next((c.arrangement for c in cfg.condition_list() if c.dt_ms == step), "sync")
    co = Condition("co", arr, step)
    r = np.corrcoef(final_levels("co", arr, step).T)
    check("comodulation", np.min(r) > 0.95,
          f"final-burst level across bands over 40 trials ({arr} {step:g}): correlation min {np.min(r):.3f}")
    off_diag = np.abs(np.corrcoef(final_levels("rand", "sync", 0.0).T)[np.triu_indices(5, 1)])
    check("random envelopes", np.mean(off_diag) < 0.4,
          f"final-burst level across bands over 40 trials: mean |correlation| {np.mean(off_diag):.2f} "
          f"(independent noise)")
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
    n_head = 6 * len(cfg.condition_list())
    try:                                   # no redraws here: a tall peak must show up as a failure
        peak = max(float(np.abs(build(cfg, c, cfg.max_db, (5, k, j), _redraw=99)[1]).max())
                   for k, c in enumerate(cfg.condition_list()) for j in range(6))
    except RuntimeError as e:
        peak = float(str(e).split()[1].rstrip(":"))
    check("headroom (uncalibrated)", peak < 0.95,
          f"tallest peak in {n_head} trials at the maximum tone level {peak:.2f}")
    from .stimulus import band_bursts
    rng = np.random.default_rng(11)
    lv = np.array([[10 * np.log10(np.mean(x ** 2)) for x in band_bursts(cfg, rng, True)]
                   for _ in range(300)])
    sd = float(lv[:, cfg.target_band].std())
    spread = float(np.abs(lv - lv[:, [cfg.target_band]]).max())
    if cfg.burst_level == "natural":
        check("natural burst level", 1.0 < sd < 6.5 and spread < 0.5,
              f"a burst's level varies by {sd:.1f} dB sd, the same in all comodulated bands "
              f"(within {spread:.2f} dB)")
    else:
        check("equalised burst level", sd < 0.01,
              "every burst has exactly the band level (legacy: removes the level cue)")
    sys.exit(0 if ok else 1)


def cal_id(cal):
    """A short name for a calibration's levels (device, system volume, every band's amplitude);
    None without a calibration. A seed check made with calibrated levels carries it, and a
    calibrated run asks for a check with the same one: peaks depend on the amplitudes."""
    if cal is None:
        return None
    key = [cal.device, cal.system_volume, cal.ref_db,
           sorted((str(k), round(float(v), 9)) for k, v in cal.amp_at_ref.items())]
    return hashlib.sha256(json.dumps(key).encode()).hexdigest()[:12]


def _check_seed(cfg, seed, data, cal, progress=None):
    from . import audit
    amp = (lambda f, db: cal.amplitude(f, db)) if cal else None
    t0 = time.time()
    rep = audit.audit(cfg, seed, amp, progress=progress)
    rep["seconds"] = round(time.time() - t0, 1)
    rep["calibration"] = cal_id(cal)
    return rep, audit.save(rep, audit.report_path(data, cfg, seed))


def _check_worker(job):
    """One seed in a worker process (check with several seeds)."""
    cfg_dict, seed, data, calibrated = job
    cal = _load_cal(data) if calibrated else None
    return _check_seed(validate(Config.from_dict(cfg_dict)), seed, data, cal)


def cmd_check(a):
    from . import audit
    cfg = _cfg(a.config)
    cal = _load_cal(a.data) if a.calibrated else None
    if a.calibrated and cal is None:
        print("\u2717 no calibration in", Path(a.data) / "calibration")
        sys.exit(2)
    seeds = list(dict.fromkeys(a.seed))
    if len(seeds) == 1:
        rep, path = _check_seed(cfg, seeds[0], a.data, cal,
                                progress=lambda i, n: print(f"  {i}/{n} trials", end="\r"))
        print(audit.describe(rep))
        print(f"  report: {path}  ({rep['seconds']:.0f} s)")
        sys.exit(0 if rep["ok"] else 1)
    # several seeds: one worker process each, as many at a time as there are cores
    import os
    from concurrent.futures import ProcessPoolExecutor
    print(f"checking {len(seeds)} seeds of {cfg.name} ({cfg.hash()}), "
          f"{'calibrated' if cal else 'uncalibrated'} levels ...")
    jobs = [(cfg.to_dict(), s, a.data, bool(cal)) for s in seeds]
    passed = 0
    with ProcessPoolExecutor(max_workers=max(1, min(len(seeds), (os.cpu_count() or 2) - 1))) as ex:
        for rep, path in ex.map(_check_worker, jobs):
            passed += bool(rep["ok"])
            if rep["ok"]:
                print(f"  seed {rep['seed']}: passed ({sum(rep['trials_checked'].values())} trials, "
                      f"tallest peak {rep['peak_max']:.2f}, {rep['seconds']:.0f} s)")
            else:
                print(audit.describe(rep))
    print(f"{passed} of {len(seeds)} seeds passed; reports in {Path(a.data) / 'checks'}")
    sys.exit(0 if passed == len(seeds) else 1)


def seed_users(data, cfg, seed) -> list:
    """Sessions of this configuration that used this seed: [(code, folder)]."""
    out = []
    for p in sorted(Path(data).glob("*/tcmr_session_*/session.json")):
        try:
            m = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if m.get("session_seed") == int(seed) and m.get("config_hash") == cfg.hash() \
                and not m.get("simulated"):
            out.append((m.get("code"), p.parent))
    return out


def next_seed(data, cfg, levels: str, calibration=None):
    """The lowest seed whose check passed with these levels ('calibrated' or 'uncalibrated') and
    that no session of this configuration has used yet; None if there is none. For calibrated
    levels the check must have been made with this calibration (`cal_id`)."""
    from . import audit
    for p in sorted(audit.report_path(data, cfg, 0).parent.glob(f"tcmr_{cfg.hash()}_*.json"),
                    key=lambda p: int(p.stem.rsplit("_", 1)[1])):
        try:
            rep = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if levels == "calibrated" and rep.get("calibration") != calibration:
            continue
        if rep.get("ok") and rep.get("amplitude") == levels and not seed_users(data, cfg, rep["seed"]):
            return int(rep["seed"])
    return None


def cmd_demo(a):
    import soundfile as sf
    cfg = _cfg(a.config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = [("co", "sync", 0), ("rand", "sync", 0), ("co", "ascending", 10),
             ("co", "ascending", 20), ("co", "zigzag", 20)]
    if any(c.arrangement in PATTERNS for c in cfg.condition_list()):
        items = [(c.envelope, c.arrangement, c.dt_ms) for c in cfg.condition_list()]
    for e, arr, dt in items:
        c = Condition(e, arr, float(dt))
        for lvl in (None, a.level):
            tr, buf = build(cfg, c, lvl, (2026, int(dt)), n_intervals=1,
                            signal_interval=None if lvl is None else 1)
            p = out / f"{c.name}_{'notone' if lvl is None else f'tone{lvl:g}dB'}.wav"
            sf.write(p, buf, cfg.sample_rate, subtype="PCM_24")
            print(p)


def parse_level(text: str):
    """A meter reading typed at the keyboard, in dB SPL. Persian and Arabic digits and a decimal
    comma are read too. None if it is not a level between 30 and 120 dB SPL."""
    t = text.strip().translate(_DIGITS).replace("\u066b", ".").replace(",", ".")
    try:
        db = float(t)
    except ValueError:
        return None
    return db if 30.0 <= db <= 120.0 else None


def calibration_notes(cfg, readings: dict, test: float = TEST_AMP) -> list:
    """What a set of readings (band centre in Hz -> dB SPL at amplitude `test`) means for the
    levels: [(mark, text)], mark "!" for something to act on before running a listener."""
    a60 = {f: test * 10 ** ((60.0 - db) / 20) for f, db in readings.items()}
    out = []
    tones = [cfg.bands_hz[b] for b in (cfg.tone_bands or (cfg.target_band,))]
    if all(f in a60 for f in tones) and len(a60) == len(cfg.bands_hz):
        tone = max(a60[f] for f in tones) * 10 ** ((cfg.max_db - 60.0) / 20)
        peak = tone + NOISE_PEAK_PER_AMP * sum(a60.values()) * 10 ** ((cfg.band_db - 60.0) / 20)
        out.append(("!" if peak > 0.95 else " ",
                    f"the {cfg.max_db:g}-dB tone peaks at {tone:.2f}; with the noise the tallest "
                    f"peak will be about {peak:.2f} (limit 0.95)" +
                    (": too little headroom, raise the system volume and calibrate again"
                     if peak > 0.95 else "")))
    for f in tones:
        if f in a60 and a60[f] > cfg.uncal_peak_60db * 1.0005:
            need = 60.0 + 20 * math.log10(test / cfg.uncal_peak_60db)
            out.append((" ", f"{f:g} Hz read {readings[f]:g} dB SPL, below the {need:.0f} dB at "
                             f"which the seeds were checked without a calibration: less headroom "
                             f"than those checks had; the calibrated check decides"))
    if len(readings) > 1 and max(readings.values()) - min(readings.values()) > 15.0:
        out.append(("!", f"the bands differ by {max(readings.values()) - min(readings.values()):.0f} "
                         f"dB: look at the seal on the coupler and at the weighting (A weighting "
                         f"reads 8.6 dB low at 250 Hz)"))
    return out


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
    print(f"{dev['name']}, system volume {vol['output_volume']}.\n"
          f"Sound level meter on an ear coupler, C or Z weighting (not A), slow. Each band's tone "
          f"plays for a minute; Enter plays it again.")
    info = {k: input(f"{k}: ").strip() for k in ("headphones", "meter", "coupler", "operator")}
    tone_s, readings, now = 60.0, {}, {"pb": None}
    t = np.arange(int(tone_s * cfg.sample_rate)) / cfg.sample_rate
    gate = TEST_AMP * envelope(t, tone_s, 0.05)

    def stop():
        pb = now["pb"]
        if pb is not None and not pb.finished:
            eng.abort(pb)
            t_end = time.perf_counter() + 1.0
            while not pb.finished and time.perf_counter() < t_end:
                time.sleep(0.01)
        now["pb"] = None

    def play(f):
        stop()
        x = (gate * np.sin(2 * np.pi * f * t)).astype(np.float32)
        now["pb"] = eng.play_at(np.repeat(x[:, None], cfg.channels, 1), time.perf_counter() + 0.1)

    # the tone's own band first: it shows at once whether this volume is high enough
    first = [cfg.bands_hz[b] for b in (cfg.tone_bands or (cfg.target_band,))]
    order = first + [f for f in cfg.bands_hz if f not in first]
    saved = False
    try:
        for f in order:
            play(f)
            while True:
                txt = input(f"{f:g} Hz at amplitude {TEST_AMP}: reading on the meter (dB SPL): ")
                if not txt.strip():
                    play(f)
                    continue
                db = parse_level(txt)
                if db is not None:
                    break
                print("  that is not a level between 30 and 120 dB SPL: type the reading again")
            readings[f] = db
            a60 = TEST_AMP * 10 ** ((60.0 - db) / 20)
            print(f"  60 dB SPL at {f:g} Hz = amplitude {a60:.4f}")
            if f in first and a60 * 10 ** ((cfg.max_db - 60.0) / 20) > 0.6:
                stop()                         # the tone alone would use most of the range
                print(f"\u2717 at this volume the {cfg.max_db:g}-dB tone would peak at "
                      f"{a60 * 10 ** ((cfg.max_db - 60.0) / 20):.2f} before any noise. Raise the "
                      f"system volume (now {vol['output_volume']}) and run calibrate again. "
                      f"Nothing was saved.")
                sys.exit(2)
        stop()
        after = volume_settings()
        if after.get("output_volume") != vol.get("output_volume") or after.get("muted"):
            print(f"\u2717 the system volume changed during the calibration "
                  f"({vol.get('output_volume')} to {after.get('output_volume')}, muted "
                  f"{after.get('muted')}). Nothing was saved: run calibrate again.")
            sys.exit(2)
        amps = {f"{f:.6f}": TEST_AMP * 10 ** ((60.0 - readings[f]) / 20) for f in cfg.bands_hz}
        meas = [{"freq_hz": f, "amplitude": TEST_AMP, "db_spl": readings[f]} for f in order]
        stamp = datetime.now().astimezone()
        cal = Calibration(True, 60.0, amps, dev["name"], vol["output_volume"], info["headphones"],
                          info["meter"], info["coupler"], "Z/C", stamp.isoformat(timespec="seconds"),
                          info["operator"], "tcmr: sines at the band centres", meas)
        d = Path(a.data) / "calibration"
        d.mkdir(parents=True, exist_ok=True)
        text = json.dumps(cal.to_dict(), indent=1)
        (d / f"tcmr_calibration_{stamp.strftime('%Y%m%d-%H%M%S')}.json").write_text(text)
        (d / "tcmr_latest.json").write_text(text)
        saved = True
    finally:
        stop()
        eng.stop()
        if not saved:
            print("no calibration was saved")
    print(f"saved {d / 'tcmr_latest.json'} (calibration {cal_id(cal)}, system volume "
          f"{vol['output_volume']})")
    for mark, note in calibration_notes(cfg, readings):
        print(f"{mark} {note}")
    cfg_arg = f" --config {a.config}" if a.config else ""
    print(f"next: check the seeds with these levels, then run every session at this volume:\n"
          f"  python -m tcmr check{cfg_arg} --seed N1 N2 ... --calibrated\n"
          f"  python -m tcmr run{cfg_arg} --code CODE --seed next --volume {vol['output_volume']}")


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
    old = None                             # the session to continue: its own record, read first
    if a.resume:
        from .session import resume_blocked
        folder = Path(a.resume)
        try:
            old = json.loads((folder / "session.json").read_text())
            old_rows = [r["row"] for r in read_jsonl(folder / "trials.jsonl")]
        except (OSError, ValueError) as e:
            print(f"\u2717 {folder} is not a session that can be continued ({e})")
            sys.exit(2)
        stored = dict(old.get("config") or cfg.to_dict())
        stored.setdefault("burst_level", "equalised")
        why = resume_blocked(validate(Config.from_dict(stored)), old, old_rows)
        if why:
            print(f"\u2717 {why}")
            sys.exit(2)
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
    run_volume = a.volume if a.volume is not None else vol["output_volume"]   # --volume sets it below
    if not uncal and (cal is None or cal.device != dev["name"]
                      or cal.system_volume != run_volume):
        print("\u2717 no matching calibration (device and system volume must match): run "
              "`python -m tcmr calibrate`, or pass --uncalibrated for a pilot")
        sys.exit(2)
    if old is not None:                    # a session is continued with the levels it started with
        v0 = (old.get("volume_at_start") or {}).get("output_volume")
        again = (f"--volume {v0}" if v0 is not None else "") + \
            ("" if old.get("levels_calibrated") else " --uncalibrated")
        if bool(old.get("levels_calibrated")) != (cal is not None):
            print(f"\u2717 this session started with "
                  f"{'calibrated' if old.get('levels_calibrated') else 'uncalibrated'} levels: "
                  f"continue it the same way ({again.strip()})")
            sys.exit(2)
        if cal is not None and old.get("calibration") and \
                cal_id(cal) != cal_id(type(cal).from_dict(old["calibration"])):
            print("\u2717 the calibration changed since this session started: its levels would "
                  "change in the middle. Start a new session instead.")
            sys.exit(2)
        if old.get("volume_locked") and v0 is not None and run_volume != v0 \
                and not a.allow_volume_change:
            print(f"\u2717 this session started at system volume {v0}, now it would be "
                  f"{run_volume}: continue it with {again.strip()}")
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
        w = Writer(folder, {}, resume=True)
        stored = dict(w.meta.get("config") or {})
        if stored:                         # a session keeps the stimulus it started with
            stored.setdefault("burst_level", "equalised")
            was = validate(Config.from_dict(stored))
            if was != cfg:
                print(f"! continuing with the configuration this session started with "
                      f"({stored.get('name')}, {stored.get('burst_level')} bursts)")
            cfg = was
        rows = [r["row"] for r in read_jsonl(folder / "trials.jsonl")]
        exp = Experiment(cfg, w, w.meta["session_seed"], amp, resume_rows=rows)
    else:
        if a.seed is not None:             # a seed whose every trial was checked beforehand
            from tsync.design import clean_code
            from . import audit
            want = "calibrated" if cal else "uncalibrated"
            if str(a.seed).lower() == "next":
                a.seed = next_seed(a.data, cfg, want, cal_id(cal))
                if a.seed is None:
                    print(f"✗ no seed of {cfg.name} is both checked ({want} levels) and unused. "
                          f"Check more first:\n  python -m tcmr check --config {a.config or PRESET} "
                          f"--seed N1 N2 ..." + (" --calibrated" if cal else ""))
                    sys.exit(2)
                print(f"seed {a.seed}: the lowest checked seed no one has used yet")
            else:
                try:
                    a.seed = int(a.seed)
                except ValueError:
                    print("✗ --seed takes a number or 'next'")
                    sys.exit(2)
            others = [(c, f) for c, f in seed_users(a.data, cfg, a.seed) if c != clean_code(a.code)]
            if others and not a.reuse_seed:      # every listener hears their own trials
                nxt = next_seed(a.data, cfg, want, cal_id(cal))
                print(f"✗ seed {a.seed} was already used by {others[0][0]} ({others[0][1]}). "
                      f"Every listener gets their own seed" +
                      (f": the next unused checked seed is {nxt} (or pass --seed next)." if nxt
                       else "; check more seeds first."))
                sys.exit(2)
            rp = audit.report_path(a.data, cfg, a.seed)
            rep = json.loads(rp.read_text()) if rp.exists() else None
            stale = bool(rep) and cal is not None and rep.get("calibration") != cal_id(cal)
            if rep is None or not rep.get("ok") or rep.get("amplitude") != want or stale:
                why = ("no check found" if rep is None else "the check failed" if not rep.get("ok")
                       else f"the check was made with {rep.get('amplitude')} levels, this run is {want}"
                       if rep.get("amplitude") != want
                       else "the check was made with another calibration than the one in use now")
                print(f"\u2717 seed {a.seed}: {why}. Run first:\n  python -m tcmr check --config "
                      f"{a.config or PRESET} --seed {a.seed}" + (" --calibrated" if cal else ""))
                sys.exit(2)
            extra["seed_check"] = {"report": str(rp), "ok": True, "trials_checked": rep["trials_checked"]}
            print(f"seed {a.seed}: every planned trial was checked ({rp.name})")
        w, seed = new_session(cfg, a.data, a.code, extra, seed=a.seed)
        exp = Experiment(cfg, w, seed, amp)
        if w.meta.get("tcmr_dirty"):
            print("! the tcmr folder has uncommitted changes. This session records their hash "
                  "(and the diff of tracked files), but commit before a listener whose data will count.")
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
    gate = w.meta.get("practice") or {}
    if gate and not gate.get("passed", True):
        print(f"\u2717 practice gate: {gate.get('correct')} of {gate.get('trials')} correct, "
              f"{gate.get('needed')} needed. The session stopped before the main part and "
              f"cannot be continued.")
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
        if not s.get("thresholds_db"):              # e.g. quit before the main part
            print("  no main trials: no figure")
            continue
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
    sp = p("check", cmd_check)
    sp.add_argument("--seed", type=int, nargs="+", required=True,
                    help="one or more seeds (several are checked in parallel)")
    sp.add_argument("--data", default="data")
    sp.add_argument("--calibrated", action="store_true",
                    help="use the calibration in DATA/calibration (as a calibrated run would)")
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
            sp.add_argument("--seed", default=None,
                            help="session seed; must have passed `python -m tcmr check`. 'next': "
                                 "the lowest checked seed that no session has used yet")
            sp.add_argument("--reuse-seed", action="store_true",
                            help="allow a seed another listener already used")
    sp = p("simulate", cmd_simulate)
    sp.add_argument("--data", default="data_tcmr/sim")
    sp.add_argument("--code", default="SIM01")
    sp.add_argument("--seed", type=int, default=1)
    sp = p("analyze", cmd_analyze)
    sp.add_argument("sessions", nargs="+")
    a = ap.parse_args(argv)
    a.fn(a)
