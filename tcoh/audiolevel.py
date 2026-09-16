"""What the machine is doing with the sound, and how to tie it to a measured level.

A calibration is a statement about a whole chain: digital amplitude -> system volume -> output
device -> transducer -> ear. Recording only the measured SPL pins down one end of that and
leaves the rest free, so a session calibrated at one system volume and run at another is wrong
in a way nothing in the data would show. This module records the parts that can be read
automatically and gives the arithmetic for the part that cannot.

What is readable, and what is not
----------------------------------
The system volume, the mute state and the output device name can be read on macOS. The level at
the ear cannot: that needs a meter, and the relationship between a volume setting and an SPL
depends on the headphones, the amplifier and the fit. So the tool reads what it can, asks the
experimenter for the one number it cannot, and then refuses to let the readable part drift
silently afterwards.

The one piece of exact arithmetic
----------------------------------
Digital scaling IS linear, so once a level has been measured at a known amplitude the amplitude
needed for any other level follows exactly:

    amplitude_needed = amplitude_measured * 10 ** ((target_db - measured_db) / 20)

That is `amplitude_for`. It is the reason it is better to leave the system volume alone and
change `tone_amplitude` instead: the volume control is an undocumented, quantised, possibly
non-linear taper, while this is a multiplication.
"""
from __future__ import annotations

import math
import platform
import subprocess
from typing import Optional


def _osascript(script: str) -> Optional[str]:
    try:
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


def system_output() -> dict:
    """System volume, mute state and output device, as far as the platform will say.

    Every field may be None: this is a convenience for keeping a session honest, never a
    precondition for running one.
    """
    out = {"platform": platform.system(), "volume": None, "muted": None, "device": None,
           "readable": False}
    if out["platform"] == "Darwin":
        v = _osascript("output volume of (get volume settings)")
        m = _osascript("output muted of (get volume settings)")
        try:
            out["volume"] = int(v) if v is not None else None
        except ValueError:
            pass
        if m in ("true", "false"):
            out["muted"] = m == "true"
        out["readable"] = out["volume"] is not None
    try:
        import sounddevice as sd
        out["device"] = str(sd.query_devices(kind="output")["name"])
    except Exception:
        pass
    return out


def describe_output(s: Optional[dict] = None) -> str:
    s = s or system_output()
    if not s["readable"] and not s["device"]:
        return "output level and device could not be read on this platform"
    bits = []
    if s["device"]:
        bits.append(f"device {s['device']!r}")
    if s["volume"] is not None:
        bits.append(f"system volume {s['volume']}/100")
    if s["muted"]:
        bits.append("MUTED")
    return ", ".join(bits)


def amplitude_for(target_db_spl: float, measured_db_spl: float, at_amplitude: float) -> float:
    """The digital amplitude that puts the tone at `target_db_spl`, given one measurement.

    Exact, because digital scaling is linear. Preferable to nudging the system volume, which is
    a quantised taper nobody has documented.
    """
    return float(at_amplitude * 10.0 ** ((target_db_spl - measured_db_spl) / 20.0))


def scene_db_spl(tone_db_spl: float, n_tones: int = 2) -> float:
    """Two equal, incoherent tones together are 3 dB above one of them."""
    return float(tone_db_spl + 10.0 * math.log10(max(n_tones, 1)))


def drift(calibrated: Optional[dict], now: Optional[dict] = None) -> dict:
    """Has anything readable changed since the calibration? Returns what, if so."""
    now = now or system_output()
    if not calibrated:
        return {"known": False, "changed": False, "notes": ["no calibration recorded"]}
    notes, changed = [], False
    for key, label in (("volume", "system volume"), ("device", "output device")):
        was, is_ = calibrated.get(key), now.get(key)
        if was is not None and is_ is not None and was != is_:
            changed = True
            notes.append(f"{label} was {was} at calibration and is {is_} now")
    if now.get("muted"):
        changed = True
        notes.append("output is MUTED")
    return {"known": True, "changed": changed, "notes": notes, "then": calibrated, "now": now}
