"""From dB SPL to a linear amplitude: the calibration.

A calibration is a statement about a whole chain (digital amplitude, system volume, output
device, headphones), so it records all of them, and a session refuses to run if the device or the
system volume it finds is not the one the calibration was made with.

Each tone frequency is calibrated separately (1000 Hz and 420.45 Hz here) because no headphone
is flat: one number for both would put the A tones at an unknown level relative to B. Levels are
measured with a sound level meter and an ear coupler, Z (flat) or C weighting; A weighting is
-3 dB at 420 Hz and would bias the low tone.

Without a calibration the session can still run for piloting (`--uncalibrated`): every level is
then reported as nominal and the session is flagged.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import Config

FREQ_TOL_HZ = 0.5


@dataclass
class Calibration:
    calibrated: bool
    ref_db: float                          # the level the amplitudes below produce
    amp_at_ref: Dict[str, float]           # frequency (Hz, as text) -> linear peak amplitude
    device: str = ""
    system_volume: Optional[int] = None
    headphones: str = ""
    meter: str = ""
    coupler: str = ""
    weighting: str = ""
    date: str = ""
    operator: str = ""
    notes: str = ""
    measurements: List[dict] = field(default_factory=list)

    def amplitude(self, freq_hz: float, level_db: float) -> float:
        a_ref = self._amp_ref(freq_hz)
        return a_ref * 10.0 ** ((level_db - self.ref_db) / 20.0)

    def _amp_ref(self, f: float) -> float:
        pts = sorted((float(k), v) for k, v in self.amp_at_ref.items())
        for fk, v in pts:
            if abs(fk - f) <= FREQ_TOL_HZ:
                return v
        if not self.calibrated:
            return pts[0][1]
        raise KeyError(f"no calibration for {f:.2f} Hz; calibrated: {[p[0] for p in pts]}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        return cls(**d)

    def age_days(self, now: Optional[datetime] = None) -> Optional[float]:
        if not self.date:
            return None
        then = datetime.fromisoformat(self.date)
        now = now or datetime.now(then.tzinfo)
        return (now - then).total_seconds() / 86400.0


def uncalibrated(cfg: Config, amplitude: float = 0.05) -> Calibration:
    """Nominal levels only: `amplitude` for each tone at its nominal level."""
    return Calibration(False, cfg.level_b_db,
                       {f"{cfg.f_b_hz:.6f}": amplitude, f"{cfg.f_a_hz:.6f}": amplitude},
                       notes="uncalibrated: levels are nominal")


def amplitude_for(ref_db: float, measured_db: float, at_amplitude: float) -> float:
    """The amplitude that gives `ref_db`, from a reading of `measured_db` at `at_amplitude`."""
    return at_amplitude * 10.0 ** ((ref_db - measured_db) / 20.0)


def worst_case_peak(cfg: Config, cal: Calibration) -> float:
    """Largest possible sum of one B and one A tone (A at nominal + rove + decoy)."""
    a_b = cal.amplitude(cfg.f_b_hz, cfg.level_b_db)
    a_a = cal.amplitude(cfg.f_a_hz, cfg.level_a_db + cfg.a_rove_db + cfg.a_decoy_db)
    return a_b + a_a


def calibration_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "calibration"


def save_calibration(data_dir: Path, cal: Calibration) -> Path:
    d = calibration_dir(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(cal.date).strftime("%Y%m%d-%H%M%S")
    p = d / f"tsync_calibration_{stamp}.json"
    text = json.dumps(cal.to_dict(), indent=1)
    p.write_text(text)
    (d / "tsync_latest.json").write_text(text)
    return p


def load_latest(data_dir: Path) -> Optional[Calibration]:
    p = calibration_dir(data_dir) / "tsync_latest.json"
    if not p.exists():
        return None
    return Calibration.from_dict(json.loads(p.read_text()))


def db_of(amplitude: float, cal: Calibration, freq_hz: float) -> float:
    """Inverse of `Calibration.amplitude`: the level an amplitude produces at a frequency."""
    return cal.ref_db + 20.0 * math.log10(amplitude / cal._amp_ref(freq_hz))
