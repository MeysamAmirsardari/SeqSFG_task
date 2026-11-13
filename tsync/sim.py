"""A simulated listener and a headless driver: the whole session without sound or screen.

`drive` runs the same `Experiment.script()` the app runs, answering every Screen and Play the
way a listener with known thresholds would, and keeps a clock of how long each step would take
in the booth. It is used to test the script and the logging end to end, to check that the
analysis recovers the thresholds it was given, and to estimate how long a session lasts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy import stats

from .config import Config
from .experiment import Experiment, Play, PlayResult, Screen, ScreenResult, SessionEnded
from .logbook import now_iso

DEFAULT_THRESHOLDS = {"D0": 4.0, "D35": 7.0, "D50": 11.0, "D71": 17.0, "D100": 24.0,
                      "D0_late": 4.5}

# seconds a listener spends on each kind of screen (reading, deciding), excluding min_s
SCREEN_S = {"message": 7.0, "form": 70.0, "rest": 12.0, "alert": 15.0, "end": 0.0}


@dataclass
class ModelListener:
    thresholds_ms: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    sigma: float = 0.5           # psychometric slope, natural-log units of delta
    lapse: float = 0.02
    target_p: float = 0.5 ** (1.0 / 3.0)
    rt_median_s: float = 0.7
    p_abort: float = 0.0         # chance that a trial is paused midway (exercises the repeat path)

    def p_correct(self, cond: str, delta_ms: float) -> float:
        if delta_ms <= 0:
            return 0.5
        th = self.thresholds_ms[cond]
        f = (self.target_p - 0.5) / (0.5 - self.lapse)
        mu = math.log(th) - self.sigma * stats.norm.ppf(f)
        return 0.5 + (0.5 - self.lapse) * stats.norm.cdf((math.log(delta_ms) - mu) / self.sigma)


@dataclass
class Clock:
    t: float = 0.0
    trials: int = 0
    screens: int = 0


def answer(screen: Screen) -> ScreenResult:
    if screen.kind == "form":
        vals = {}
        for f in screen.fields:
            if f.get("optional"):
                vals[f["name"]] = ""
            elif f["type"] == "int":
                vals[f["name"]] = int(f.get("min", 0)) + 5
            elif f["type"] == "choice":
                vals[f["name"]] = f["options"][0]
            else:
                vals[f["name"]] = "simulated"
        return ScreenResult("return", vals, SCREEN_S["form"])
    pref = {"check_q": "y", "practice_fail": "c"}
    key = pref.get(screen.name)
    if key is None:
        key = "space" if "space" in screen.keys else screen.keys[0]
    return ScreenResult(key, {}, SCREEN_S.get(screen.kind, 5.0) + screen.min_s)


def play_seconds(cfg: Config, play: Play, rt_s: float) -> float:
    tr = play.trial
    sound = tr.interval_off_s[1] + cfg.post_ms / 1000.0
    if play.mode == "listen":
        return cfg.ready_ms / 1000.0 + sound + 0.6
    fb = cfg.feedback_ms / 1000.0 if play.feedback else 0.0
    return cfg.ready_ms / 1000.0 + sound + rt_s + fb + cfg.blank_ms / 1000.0


def respond(cfg: Config, play: Play, listener: ModelListener, rng: np.random.Generator
            ) -> PlayResult:
    tr = play.trial
    fake_audio = {"onset_error_ms": 0.0, "late": False, "underflows": 0,
                  "callback_gap_max_ms": 0.0, "simulated": True}
    sysinfo = {"output_volume": 50, "muted": False, "alert_volume": 0, "device": "simulated"}
    if listener.p_abort and rng.random() < listener.p_abort:
        return PlayResult("aborted", "pause", audio=fake_audio, system=sysinfo,
                          wall_start=now_iso(), wall_response=now_iso())
    if play.mode == "listen":
        return PlayResult("ok", audio=fake_audio, system=sysinfo, wall_start=now_iso())
    p = listener.p_correct(tr.condition, tr.delta_ms)
    correct = rng.random() < p
    resp = tr.target_interval if correct else 3 - tr.target_interval
    rt = float(rng.lognormal(math.log(listener.rt_median_s), 0.35))
    return PlayResult("ok", "", resp, "left" if resp == 1 else "right", rt * 1000.0,
                      rt * 1000.0 + (tr.interval_off_s[1] - tr.seq_start_s[1]) * 1000.0,
                      audio=fake_audio, system=sysinfo, wall_start=now_iso(),
                      wall_response=now_iso())


class SimulatedCrash(RuntimeError):
    pass


def drive(exp: Experiment, listener: ModelListener, rng: np.random.Generator,
          crash_after_trials: Optional[int] = None) -> Clock:
    """Run the script to the end (or to a simulated crash), returning the simulated clock."""
    clock = Clock()
    gen = exp.script()
    try:
        action = next(gen)
        while True:
            if isinstance(action, Screen):
                res = answer(action)
                clock.t += res.shown_s
                clock.screens += 1
            elif isinstance(action, Play):
                if crash_after_trials is not None and clock.trials >= crash_after_trials:
                    gen.close()
                    raise SimulatedCrash(f"crash after {clock.trials} trials")
                res = respond(exp.cfg, action, listener, rng)
                clock.t += play_seconds(exp.cfg, action, (res.rt_ms or 0.0) / 1000.0)
                clock.trials += 1
            else:
                raise TypeError(f"unexpected action {action!r}")
            action = gen.send(res)
    except StopIteration:
        pass
    except SessionEnded:
        pass
    return clock


def estimate_duration(cfg: Config, n: int = 40, seed: int = 7, listener=None) -> dict:
    """Session length (minutes) over `n` simulated listeners, without writing anything."""
    import tempfile
    from pathlib import Path

    from .levels import uncalibrated
    from .session import open_session

    rng = np.random.default_rng(seed)
    mins, trials = [], []
    with tempfile.TemporaryDirectory() as td:
        for i in range(n):
            w, plan = open_session(cfg, Path(td), f"E{i + 1:02d}", uncalibrated(cfg),
                                   simulated=True, session_seed=int(rng.integers(2 ** 62)))
            lst = listener or ModelListener(
                {k: v * float(np.exp(rng.normal(0, 0.3))) for k, v in DEFAULT_THRESHOLDS.items()})
            exp = Experiment(cfg, uncalibrated(cfg), plan, w)
            c = drive(exp, lst, rng)
            mins.append(c.t / 60.0)
            trials.append(c.trials)
    mins, trials = np.array(mins), np.array(trials)
    return {"n": n, "median_min": float(np.median(mins)), "p90_min": float(np.percentile(mins, 90)),
            "max_min": float(mins.max()), "median_trials": float(np.median(trials)),
            "max_trials": int(trials.max())}
