"""Checks that need no listener. `python -m tsync verify` runs them all and says what failed.

The most important one measures the asynchrony from the rendered waveform, not from the numbers
the stimulus code was given: each stream is isolated with a zero-phase band-pass filter, the
Hilbert envelope of the ending tone is taken, and its half-amplitude crossing is located. The
difference between the target and the standard interval must equal the requested delta.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
from scipy import signal

from .config import Config, ConfigError, validate
from .design import balance_report, make_plan, williams_square
from .levels import uncalibrated, worst_case_peak
from .stimulus import build_trial, buffer_sha256, ending_onsets_ms, render, sequence_onsets_ms

PRESET = Path(__file__).resolve().parent / "configs" / "tsync_main.json"
PRESET_HASH = "164b8acdaf27c3b3"


def measure_ending_onsets(cfg: Config, buf: np.ndarray, trial) -> Tuple[List[float], List[float]]:
    """Half-amplitude onset times (s) of the ending's B and A tones in each interval."""
    sr = cfg.sample_rate
    x = buf[:, 0].astype(np.float64)
    out_b, out_a = [], []
    for f, out in ((cfg.f_b_hz, out_b), (cfg.f_a_hz, out_a)):
        sos = signal.butter(4, [f / 1.25, f * 1.25], btype="bandpass", fs=sr, output="sos")
        env = np.abs(signal.hilbert(signal.sosfiltfilt(sos, x)))
        for iv in (0, 1):
            nominal = trial.ending_b_s[iv] if f == cfg.f_b_hz else trial.ending_a_s[iv]
            # search from just before the onset: the previous tone of the stream ended at
            # least min_gap_ms earlier, so its (ramped, filtered) envelope is gone by then
            i0, i1 = int((nominal - 0.006) * sr), int((nominal + 0.020) * sr)
            plateau = np.median(env[int((nominal + 0.030) * sr):int((nominal + 0.070) * sr)])
            seg = env[i0:i1] / plateau
            up = np.flatnonzero((seg[1:] >= 0.5) & (seg[:-1] < 0.5))
            if up.size == 0:
                raise AssertionError("no onset found in the waveform")
            k = int(up[0]) + 1
            frac = (0.5 - seg[k - 1]) / (seg[k] - seg[k - 1])
            out.append((i0 + k - 1 + frac) / sr)
    return out_b, out_a


def _checks(cfg: Config) -> List[Tuple[str, Callable[[], str]]]:
    cal = uncalibrated(cfg)
    conds = cfg.conditions()

    def preset():
        p = Config.load(PRESET)
        assert p == Config(), "the preset differs from the defaults in config.py"
        assert p.hash() == PRESET_HASH, f"preset hash {p.hash()} != frozen {PRESET_HASH}"
        return f"preset {PRESET.name} = defaults, hash {p.hash()}"

    def valid():
        validate(cfg)
        try:
            validate(cfg.replace(delta_max_ms=200.0))
        except ConfigError:
            return "validates; refuses a ceiling that would make a stream's tones touch"
        raise AssertionError("an impossible ceiling was accepted")

    def geometry():
        for c in conds:
            for sign in (1, -1):
                b0, a0 = ending_onsets_ms(cfg, c, 0.0, sign)
                assert abs(b0 - a0) < 1e-9, "standard ending is not synchronous"
                for d in (0.25, 5.0, 60.0):
                    b, a = ending_onsets_ms(cfg, c, d, sign)
                    assert abs((a - b) - sign * d) < 1e-9, "asynchrony != sign * delta"
                    assert abs((b0 - b) - sign * d / 2) < 1e-9, "B does not carry delta/2"
                    assert abs((a - a0) - sign * d / 2) < 1e-9, "A does not carry delta/2"
            bs, as_ = sequence_onsets_ms(cfg, c, 30.0, 1)
            bt, at = sequence_onsets_ms(cfg, c, 0.0, 1)
            assert bs[:-1] == bt[:-1] and as_[:-1] == at[:-1], "the context changed with delta"
            assert all(abs(a - b - c.context_ms) < 1e-9 for b, a in zip(bs[:-1], as_[:-1]))
        e = {c.name: ending_onsets_ms(cfg, c, 0, 1)[0] - (cfg.n_tones - 1) * cfg.soa_ms
             for c in conds}
        return ("ending synchronous in every standard; target asynchrony = +-delta split evenly; "
                f"context untouched; ending offsets from B's slot (ms): "
                + ", ".join(f"{k} {v:+g}" for k, v in e.items()))

    def waveform():
        worst = 0.0
        for name in ("D0", "D50", "D100", "D0_late"):
            c = cfg.condition(name)
            for sign in (1, -1):
                for d in (0.5, 2.0, 8.0, 30.0, 60.0):
                    tr = build_trial(cfg, c, d, (7, int(d * 10), sign + 2), target_interval=2,
                                     sign=sign)
                    buf, _ = render(cfg, tr, cal.amplitude)
                    mb, ma = measure_ending_onsets(cfg, buf, tr)
                    got = ((ma[1] - mb[1]) - (ma[0] - mb[0])) * 1000.0
                    worst = max(worst, abs(got - sign * d))
        assert worst < 0.02, f"measured asynchrony is off by up to {worst:.4f} ms"
        return f"asynchrony measured from the waveform matches delta to within {worst * 1000:.2f} us"

    def levels():
        rng = np.random.default_rng(3)
        roves, decoys, ti, sg = [], [], [], []
        for k in range(1500):
            tr = build_trial(cfg, conds[k % len(conds)], 10.0, (11, k))
            for iv in (1, 2):
                a = [t for t in tr.tones if t.stream == "A" and t.interval == iv]
                roves += [t.rove_db for t in a]
                decoys.append(sum(t.decoy for t in a))
                assert all(abs(t.level_db - (cfg.level_a_db + t.rove_db + (cfg.a_decoy_db if t.decoy
                                                                            else 0))) < 1e-9 for t in a)
                assert all(t.level_db == cfg.level_b_db for t in tr.tones if t.stream == "B")
            ti.append(tr.target_interval)
            sg.append(tr.sign)
        roves = np.array(roves)
        assert roves.min() >= -cfg.a_rove_db and roves.max() <= cfg.a_rove_db
        assert set(decoys) == set(cfg.a_decoy_counts)
        p1, pb = np.mean(np.array(ti) == 1), np.mean(np.array(sg) == 1)
        assert abs(p1 - 0.5) < 0.04 and abs(pb - 0.5) < 0.04
        return (f"A roved {roves.min():+.2f} to {roves.max():+.2f} dB, decoys {sorted(set(decoys))} "
                f"per sequence; target first {p1:.1%}, B first {pb:.1%} of 1500 trials")

    def timing():
        gaps, isis = [], []
        for k in range(400):
            c = conds[k % len(conds)]
            tr = build_trial(cfg, c, cfg.delta_max_ms, (13, k))
            gaps.append((tr.interval_on_s[1] - tr.interval_off_s[0]) * 1000.0)
            isis.append(tr.isi_ms)
            for iv in (1, 2):
                for s in ("A", "B"):
                    ts = sorted((t for t in tr.tones if t.interval == iv and t.stream == s),
                                key=lambda t: t.onset_s)
                    g = min(b.onset_s - (a.onset_s + a.dur_s) for a, b in zip(ts, ts[1:]))
                    assert g * 1000.0 >= cfg.min_gap_ms - 1e-6, "tones of one stream too close"
        assert min(isis) >= cfg.isi_ms[0] and max(isis) <= cfg.isi_ms[1]
        inside = cfg.soa_ms - cfg.tone_ms
        assert min(gaps) > 3 * inside, "the two sequences are not clearly separated"
        return (f"silence between the two sequences {min(gaps):.0f} to {max(gaps):.0f} ms (vs "
                f"{inside:g} ms inside a sequence); no stream ever closer than {cfg.min_gap_ms:g} ms")

    def headroom():
        pk = worst_case_peak(cfg, cal)
        worst = 0.0
        for k in range(200):
            tr = build_trial(cfg, conds[k % len(conds)], 0.3, (17, k))
            buf, _ = render(cfg, tr, cal.amplitude)
            worst = max(worst, float(np.max(np.abs(buf))))
        assert worst <= pk + 1e-6 and pk < cfg.max_peak
        return f"worst rendered peak {worst:.3f}, bound {pk:.3f}, refusal above {cfg.max_peak}"

    def repro():
        c = cfg.condition("D50")
        a = buffer_sha256(render(cfg, build_trial(cfg, c, 12.0, (5, 6, 7)), cal.amplitude)[0])
        b = buffer_sha256(render(cfg, build_trial(cfg, c, 12.0, (5, 6, 7)), cal.amplitude)[0])
        d = buffer_sha256(render(cfg, build_trial(cfg, c, 12.0, (5, 6, 8)), cal.amplitude)[0])
        assert a == b and a != d
        return "same seed, same SHA-256; different seed, different buffer"

    def order():
        rows = williams_square(len(conds))
        rep = balance_report(rows)
        assert all(rep.values()), rep
        p1, p2 = make_plan(cfg, 1, 5), make_plan(cfg, 7, 5)
        assert p1.blocks == p2.blocks
        return f"Williams square, {len(rows)} orders: positions and carry-over balanced"

    def staircase():
        from .sim import ModelListener
        from .staircase import Staircase
        rng = np.random.default_rng(1)
        err, lens = [], []
        for th in (3.0, 10.0, 30.0):
            lst = ModelListener({"x": th})
            for _ in range(150):
                s = Staircase(cfg)
                while not s.finished:
                    s.update(rng.random() < lst.p_correct("x", s.delta))
                if s.threshold_ms():
                    err.append(math.log(s.threshold_ms() / th))
                    lens.append(s.n_trials)
        bias, sd = float(np.mean(err)), float(np.std(err))
        assert abs(bias) < 0.12, f"bias {bias:.3f} log units"
        return (f"recovers known thresholds: bias {bias:+.3f} ln units, single-track SD {sd:.2f}, "
                f"{np.median(lens):.0f} trials per track (median)")

    def session():
        from .analysis import summarize
        from .experiment import Experiment
        from .session import open_session
        from .sim import ModelListener, drive
        with tempfile.TemporaryDirectory() as td:
            w, plan = open_session(cfg, Path(td), "V01", cal, simulated=True, session_seed=9)
            lst = ModelListener(p_abort=0.03)
            drive(Experiment(cfg, cal, plan, w), lst, np.random.default_rng(2))
            folder = w.folder
            n_csv = sum(1 for _ in open(folder / "trials.csv")) - 1
            n_js = sum(1 for _ in open(folder / "trials.jsonl"))
            meta = json.loads((folder / "session.json").read_text())
            summ = summarize(folder, n_boot=0)
        assert n_csv == n_js and meta["status"] == "complete"
        assert all(t["converged"] and t["consistent"] for t in summ["tracks"])
        rep = summ["repeated_trials"].get("pause", 0)
        return (f"simulated session complete: {n_js} trials logged identically in CSV and JSONL, "
                f"{rep} paused trials repeated, all tracks converged and replay exactly")

    def duration():
        from .sim import estimate_duration
        d = estimate_duration(cfg, n=24)
        assert d["p90_min"] < 45.0, d
        return (f"session length over 24 simulated listeners: median {d['median_min']:.1f} min, "
                f"90th percentile {d['p90_min']:.1f}, longest {d['max_min']:.1f}")

    return [("preset", preset), ("configuration", valid), ("geometry", geometry),
            ("waveform asynchrony", waveform), ("levels and randomisation", levels),
            ("timing", timing), ("headroom", headroom), ("reproducibility", repro),
            ("order", order), ("staircase", staircase), ("whole session", session),
            ("duration", duration)]


def run_checks(cfg: Config = None, verbose: bool = True) -> bool:
    cfg = cfg or Config()
    ok = True
    for name, fn in _checks(cfg):
        try:
            msg = fn()
            line = f"  pass  {name}: {msg}"
        except Exception as e:     # report every failure, then fail
            ok = False
            line = f"  FAIL  {name}: {type(e).__name__}: {e}"
        if verbose:
            print(line, flush=True)
    return ok
