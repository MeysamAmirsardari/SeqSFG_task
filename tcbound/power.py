"""Power of the planned study, by simulating whole studies through the real runner and analysis.

Each simulated listener's binding is

    b(s) = 0.5 + gain * (rho(s / alpha) - 0.5)

gain = 1 is the model's full swing, 0.5 half of it. alpha, the time scale, is drawn per listener
log-uniformly from 0.7 to 1.5. Internal noise is drawn per listener from 2.5 to 4.0 dB on the
timbre and from 1.5 to 3.0 dB on the own stream. Each study runs `n_listeners` listeners, two
sessions each, and the protocol's tests are read off `analysis.group`.

The listener adds the two cues, which is what the decision weights assume, so the power for H2
and H3 is an upper limit: real listeners add variance this one does not have.

    python -m tcbound power --gain 0.5 --listeners 12 --studies 20
"""
from __future__ import annotations

import contextlib
import io
import tempfile
from pathlib import Path

import numpy as np

from .config import Condition, Config
from .observer import SimulatedListener


class ScaledListener(SimulatedListener):
    gain: float = 1.0
    alpha: float = 1.0

    def binding(self, cond: Condition) -> float:
        from .model import curve, target_coherence
        if cond.kind == "alone":
            return 0.0
        if cond.kind == "capture":
            rho = target_coherence(self.cfg, cond)
        else:
            rho = float(curve(self.cfg, [cond.step_ms / self.alpha])[0])
        return float(np.clip(0.5 + self.gain * (rho - 0.5), 0.0, 1.0))


def study(cfg: Config, seed: int, n_listeners: int = 12, gain: float = 0.5,
          n_sessions: int = 2, n_boot: int = 300) -> dict:
    from .analysis import group
    from .runner import Runner
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory() as tmp:
        dirs = []
        for i in range(n_listeners):
            alpha = float(np.exp(rng.uniform(np.log(0.7), np.log(1.5))))
            nt, no = rng.uniform(2.5, 4.0), rng.uniform(1.5, 3.0)
            for s in range(1, n_sessions + 1):
                r = Runner(cfg, Path(tmp), auto="coherence", seed=seed * 1000 + i * 10 + s)
                sim = ScaledListener(cfg, "coherence", noise_timbre_db=nt, noise_own_db=no)
                sim.gain, sim.alpha = gain, alpha
                r.sim = sim
                with contextlib.redirect_stdout(io.StringIO()):
                    dirs.append(r.run(code=f"P{i:02d}", session_index=s))
        return group(dirs, n_boot=n_boot)


def run(cfg: Config, n_studies: int = 20, n_listeners: int = 12, gain: float = 0.5,
        alpha_level: float = 0.05, progress: bool = True) -> dict:
    rows = []
    for k in range(n_studies):
        g = study(cfg, k + 1, n_listeners, gain)
        rows.append((g["H1"]["p"], g["H2"]["p"], g["H3"]["p"], g["H3_bi"]["p"],
                     np.nan if g["s50_ms"] is None else g["s50_ms"],
                     g["fits"]["halfwave"]["alpha"]))
        if progress:
            print(f"  study {k + 1}/{n_studies}: p = " + ", ".join(f"{x:.3g}" for x in rows[-1][:4])
                  + f"; bound {rows[-1][4]:.1f} ms", flush=True)
    r = np.array(rows, dtype=float)
    hit = (r[:, :4] < alpha_level).mean(0)
    return {"n_studies": n_studies, "n_listeners": n_listeners, "gain": gain,
            "power": {"H1": hit[0], "H2": hit[1], "H3": hit[2], "H3b": hit[3]},
            "bound_median_ms": float(np.nanmedian(r[:, 4])),
            "bound_iqr_ms": tuple(float(x) for x in np.nanpercentile(r[:, 4], [25, 75])),
            "alpha_median": float(np.nanmedian(r[:, 5]))}
