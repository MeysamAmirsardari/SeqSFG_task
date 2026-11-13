"""What the machine says about itself: volume, versions, code state. Nothing here changes a setting.

The system volume, mute state and alert volume are read with AppleScript (`get volume
settings`), which takes ~50-100 ms, so the app reads them in a worker thread during the silent
gap before every trial and logs them with it. The session locks the volume it started with and
pauses if it ever reads a different one.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

PKG_DIR = Path(__file__).resolve().parent
REPO = PKG_DIR.parent


def _run(args, timeout=5.0) -> Optional[str]:
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def volume_settings() -> Dict[str, object]:
    """{'output_volume': 0-100, 'muted': bool, 'alert_volume': 0-100, 'readable': bool}."""
    out: Dict[str, object] = {"output_volume": None, "muted": None, "alert_volume": None,
                              "input_volume": None, "readable": False}
    if platform.system() != "Darwin":
        return out
    s = _run(["osascript", "-e", "get volume settings"], timeout=3.0)
    if not s:
        return out
    for key, name in (("output volume", "output_volume"), ("input volume", "input_volume"),
                      ("alert volume", "alert_volume")):
        m = re.search(key + r":(\d+|missing value)", s)
        if m and m.group(1).isdigit():
            out[name] = int(m.group(1))
    m = re.search(r"output muted:(true|false|missing value)", s)
    if m and m.group(1) in ("true", "false"):
        out["muted"] = m.group(1) == "true"
    out["readable"] = out["output_volume"] is not None
    return out


def git_info() -> dict:
    head = _run(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    branch = _run(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "-C", str(REPO), "status", "--porcelain", "--", "tsync"])
    diff = _run(["git", "-C", str(REPO), "diff", "HEAD", "--", "tsync"]) or ""
    untracked = _run(["git", "-C", str(REPO), "ls-files", "--others", "--exclude-standard",
                      "--", "tsync"]) or ""
    return {"commit": head, "branch": branch, "dirty": bool(status),
            "status": status or "", "diff": diff, "untracked": untracked.splitlines()}


def source_hash() -> str:
    """SHA-256 over every .py file of the package: the code that actually ran."""
    h = hashlib.sha256()
    for p in sorted(PKG_DIR.rglob("*.py")):
        h.update(p.relative_to(PKG_DIR).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for mod in ("numpy", "scipy", "sounddevice", "soundfile", "matplotlib", "PySide6"):
        try:
            out[mod] = importlib.import_module(mod).__version__
        except Exception:
            out[mod] = None
    try:
        import sounddevice as sd
        out["portaudio"] = sd.get_portaudio_version()[1]
    except Exception:
        out["portaudio"] = None
    return out


def machine() -> dict:
    out = {"platform": platform.platform(), "machine": platform.machine(),
           "macos": platform.mac_ver()[0] or None, "node": platform.node()}
    if platform.system() == "Darwin":
        out["model"] = _run(["sysctl", "-n", "hw.model"])
        out["cpu"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    return out


def disk_free_gb(path: Path) -> float:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


class KeepAwake:
    """Hold off display and system sleep for as long as this process lives (macOS caffeinate)."""

    def __init__(self):
        self.proc = None

    def start(self) -> bool:
        if platform.system() != "Darwin" or shutil.which("caffeinate") is None:
            return False
        self.proc = subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
        self.proc = None


def set_output_volume(volume: int, unmute: bool = True) -> dict:
    """Set the system output volume (0-100) and, optionally, unmute; returns the read-back.

    Used only when the person running a session asks for a fixed volume (`--volume`); macOS
    quantises the setting, so the read-back may differ by 1.
    """
    if platform.system() != "Darwin":
        return volume_settings()
    v = max(0, min(100, int(volume)))
    script = f"set volume output volume {v}" + (" without output muted" if unmute else "")
    _run(["osascript", "-e", script], timeout=3.0)
    return volume_settings()
