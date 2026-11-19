"""The playback rig (this laptop, one HD 280 Pro, one fixed system volume) and its level scale.

Level scale. tcmr and tinc play uncalibrated sounds on one digital scale: a sine of peak 0.02
stands for 60 dB (`uncal_peak_60db`), and a noise band at "60 dB" has the long-term rms of that
sine. Every level in this package is on that scale ("nominal dB"). The check tone has peak 0.05,
so its nominal level is 60 + 20 log10(0.05 / 0.02) = 67.96 dB.

With the volume fixed, a reading L of a tone of nominal level N at frequency f (one cup, one
coupler, meter weighting W) gives that cup's offset at f:  offset = (L - W(f)) - N.  Then for any
sound of that cup at f:
    sound pressure level = nominal + offset
    hearing level        = nominal + offset - RETSPL(f)
Offsets from the jig are provisional (the jig is not an ear simulator); offsets from the IEC 60318-1
ear simulator are the ones reported. Readings are logged and never edited: data/rig/*.jsonl.
"""
from __future__ import annotations

import json
import math
import os
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from . import standards as S

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("AUDIOMETRY_DATA", REPO / "data"))

PEAK_AT_60 = 0.02                         # tcmr / tinc `uncal_peak_60db`
TEST_AMP = 0.05                           # the check tone's peak (as tcmr calibrate)
TEST_NOMINAL_DB = 60.0 + 20.0 * math.log10(TEST_AMP / PEAK_AT_60)    # 67.96
MAX_PEAK = 0.5                            # nothing this package plays may go above this
NOMINAL_MAX_DB = 85.0                     # the loudest nominal level tcmr may play

DEVICE_MATCH = "Headphones"               # macOS calls the wired jack "External Headphones"
SAMPLE_RATE, CHANNELS, BLOCKSIZE, LATENCY = 48000, 2, 256, "low"
CUPS = ("left", "right")                  # channel 0 = left cup, channel 1 = right cup
COUPLERS = ("jig", "ear-simulator")
SIGNALS = ("tone", "band", "bands-sum")

# Every frequency a listener hears: tcmr's five bands, tinc's A tone (15 semitones below 1 kHz)
# and the screen's octaves 250-8000 Hz. Keys are rounded to 0.01 Hz.
TINC_A_HZ = 1000.0 * 2 ** (-15 / 12)
SCREEN_HZ = (250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
TCMR_BANDS_HZ = (250.0, 500.0, 1000.0, 2000.0, 4000.0)
ALL_HZ = tuple(sorted({*SCREEN_HZ, round(TINC_A_HZ, 2)}))
DAILY_HZ = (1000.0,)


def fkey(f: float) -> float:
    return round(float(f), 2)


# ---- level arithmetic ----------------------------------------------------------------------------
def peak_for(nominal_db: float) -> float:
    return PEAK_AT_60 * 10.0 ** ((nominal_db - 60.0) / 20.0)


def nominal_for_peak(peak: float) -> float:
    return 60.0 + 20.0 * math.log10(peak / PEAK_AT_60)


def weighting_db(freq: float, weighting: str) -> float:
    """The meter's frequency weighting at a pure tone's frequency (IEC 61672-1 analytic C; Z = 0)."""
    if weighting == "Z":
        return 0.0
    if weighting == "C":
        f2 = float(freq) ** 2
        r = (12194.0 ** 2 * f2) / ((f2 + 20.6 ** 2) * (f2 + 12194.0 ** 2))
        return 20.0 * math.log10(r) + 0.062
    raise ValueError("weighting must be C or Z (A is not used: it reads 8.6 dB low at 250 Hz)")


def offset_from_reading(reading_db: float, freq: float = 1000.0, weighting: str = "Z",
                        nominal_db: float = TEST_NOMINAL_DB) -> float:
    return reading_db - weighting_db(freq, weighting) - nominal_db


def retspl(freq: float, table: str = "hda200") -> float:
    t = {"hda200": S.RETSPL_HDA200, "hd280pro": S.RETSPL_HD280PRO,
         "hd280pro_secondary": S.RETSPL_HD280PRO_SECONDARY}[table]
    k = int(round(freq))
    if abs(k - freq) > 0.01 or k not in t:
        raise KeyError(f"no {table} reference value at {freq:g} Hz")
    return t[k]


def scoring_table() -> str:
    """HD 280 Pro values once the paper's table is entered for every screen frequency."""
    return "hd280pro" if all(int(f) in S.RETSPL_HD280PRO for f in SCREEN_HZ) else "hda200"


def nominal_from_hl(hl: float, freq: float, offset: float, table: str = "hda200") -> float:
    return hl + retspl(freq, table) - offset


def hl_from_nominal(nominal: float, freq: float, offset: float, table: str = "hda200") -> float:
    return nominal + offset - retspl(freq, table)


def range_problems(offsets: Dict[str, Dict[float, float]], top_hl: float = S.CAP_DB_HL) -> List[str]:
    """Frequencies where top_hl (presentation scale) would need more than the playable range."""
    bad = []
    for cup, d in offsets.items():
        for f in SCREEN_HZ:
            top = nominal_from_hl(top_hl, f, d[fkey(f)])
            if top > NOMINAL_MAX_DB + 1e-9 or peak_for(top) > MAX_PEAK:
                bad.append(f"{cup} {f:g} Hz: {top_hl:g} dB HL would need {top:.1f} nominal dB "
                           f"(the limit is {NOMINAL_MAX_DB:g}); is the jig reading right?")
    return bad


# ---- logs ----------------------------------------------------------------------------------------
def rig_dir(root: Optional[Path] = None) -> Path:
    return Path(root or DATA) / "rig"


def append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> List[dict]:
    if not Path(path).exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_time(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


# ---- the rig profile: what every session must match --------------------------------------------------
@dataclass
class Rig:
    headphones: str                       # a label written on the pair, e.g. "HD280Pro-1"
    volume: int                           # the macOS output volume every session uses
    device: str = "External Headphones"
    since: str = ""
    note: str = ""

    @staticmethod
    def path(root: Optional[Path] = None) -> Path:
        return rig_dir(root) / "rig.json"

    @classmethod
    def load(cls, root: Optional[Path] = None) -> Optional["Rig"]:
        p = cls.path(root)
        if not p.exists():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        return cls(**{k: d[k] for k in ("headphones", "volume", "device", "since", "note") if k in d})

    def save(self, root: Optional[Path] = None) -> None:
        write_json_atomic(self.path(root), asdict(self))


# ---- level records -----------------------------------------------------------------------------------
@dataclass
class LevelRecord:
    coupler: str                          # jig | ear-simulator
    cup: str                              # left | right
    freq_hz: float                        # 0 for the five-band sum
    readings_db: List[float]
    volume: int
    headphones: str
    device: str
    weighting: str = "C"
    signal: str = "tone"                  # tone | band | bands-sum
    test_nominal_db: float = TEST_NOMINAL_DB
    meter: str = ""
    operator: str = ""
    trial: bool = False                   # volume search: never a baseline, never an offset
    passed: Optional[bool] = None         # False: a check that failed (never a baseline or offset)
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    note: str = ""
    time: str = field(default_factory=now_iso)
    test_amp: float = 0.0

    def __post_init__(self):
        if not self.test_amp:
            self.test_amp = peak_for(self.test_nominal_db)

    @property
    def mean_db(self) -> float:
        return sum(self.readings_db) / len(self.readings_db)

    def offset_db(self) -> Optional[float]:
        if self.signal != "tone":
            return None
        return offset_from_reading(self.mean_db, self.freq_hz, self.weighting, self.test_nominal_db)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["mean_db"] = round(self.mean_db, 3)
        d["range_db"] = round(max(self.readings_db) - min(self.readings_db), 3)
        if len(self.readings_db) > 1:
            m = self.mean_db
            d["sd_db"] = round(math.sqrt(sum((x - m) ** 2 for x in self.readings_db)
                                         / (len(self.readings_db) - 1)), 3)
        off = self.offset_db()
        d["offset_db"] = None if off is None else round(off, 3)
        return d


def levels_log(root: Optional[Path] = None) -> Path:
    return rig_dir(root) / "levels.jsonl"


def usable(r: dict) -> bool:
    """A record that may define a baseline or an offset: a check tone at the standard level, not a
    trial and not a failed check."""
    return (not r.get("trial") and r.get("passed") is not False and r.get("signal", "tone") == "tone"
            and abs(float(r.get("test_nominal_db", TEST_NOMINAL_DB)) - TEST_NOMINAL_DB) < 0.01)


def level_records(root: Optional[Path] = None, coupler: Optional[str] = None,
                  headphones: Optional[str] = None, volume: Optional[int] = None,
                  include_trials: bool = False) -> List[dict]:
    out = []
    for r in read_jsonl(levels_log(root)):
        if r.get("trial") and not include_trials:
            continue
        if coupler and r.get("coupler") != coupler:
            continue
        if headphones is not None and r.get("headphones") != headphones:
            continue
        if volume is not None and int(r.get("volume", -1)) != int(volume):
            continue
        out.append(r)
    return out


def unweighted(r: dict) -> float:
    """A tone record's mean reading with the meter's weighting taken out (C and Z compare)."""
    return float(r["mean_db"]) - weighting_db(float(r["freq_hz"]) or 1000.0, r.get("weighting", "C"))


def rig_history(root: Optional[Path] = None) -> List[Rig]:
    """Every rig this lab has used, oldest first: archived ones (rig_until_*.json) and the current."""
    out = []
    for p in sorted(rig_dir(root).glob("rig_until_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        out.append(Rig(**{k: d[k] for k in ("headphones", "volume", "device", "since", "note") if k in d}))
    cur = Rig.load(root)
    if cur is not None:
        out.append(cur)
    return sorted(out, key=lambda r: parse_time(r.since) or datetime.min.astimezone())


def rig_at(history: List[Rig], when: Optional[datetime]) -> Optional[Rig]:
    """The rig in use at a time: the last one set up before it."""
    if when is None:
        return None
    use = None
    for r in history:
        t = parse_time(r.since)
        if t is not None and t <= when:
            use = r
    return use


def baseline(records: Iterable[dict], cup: str, freq: float) -> Optional[dict]:
    """The first usable jig record of this cup and frequency: what every later check is held to."""
    for r in records:
        if (r.get("coupler") == "jig" and usable(r) and r.get("cup") == cup
                and fkey(r["freq_hz"]) == fkey(freq)):
            return r
    return None


def pick_records(records: Iterable[dict], coupler: str,
                 which: str = "latest") -> Dict[Tuple[str, float], dict]:
    """(cup, frequency) -> the first or the latest usable record of that coupler."""
    out: Dict[Tuple[str, float], dict] = {}
    for r in records:
        if r.get("coupler") == coupler and usable(r):
            k = (r["cup"], fkey(r["freq_hz"]))
            if which == "latest" or k not in out:
                out[k] = r
    return out


def offsets_for(records: Iterable[dict], coupler: str, freqs: Iterable[float],
                cups: Iterable[str] = CUPS, which: str = "latest") -> Optional[Dict[str, Dict[float, float]]]:
    """{cup: {freq: offset}} if every cup and frequency has a usable record of this coupler, else
    None. The jig's provisional scale uses the FIRST records (the baseline the daily checks hold
    the rig to); the ear simulator's uses the latest."""
    lat = pick_records(records, coupler, which)
    out: Dict[str, Dict[float, float]] = {}
    for c in cups:
        out[c] = {}
        for f in freqs:
            r = lat.get((c, fkey(f)))
            if r is None or r.get("offset_db") is None:
                return None
            out[c][fkey(f)] = float(r["offset_db"])
    return out


# ---- typed readings --------------------------------------------------------------------------------
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def parse_reading(text: str, lo: float = 20.0, hi: float = 130.0) -> Optional[float]:
    """A number typed at the keyboard; Persian and Arabic digits and a decimal comma are read too.
    None if it is not a number between lo and hi."""
    t = text.strip().translate(_DIGITS).replace("٫", ".").replace(",", ".")
    try:
        v = float(t)
    except ValueError:
        return None
    return v if math.isfinite(v) and lo <= v <= hi else None


def environment() -> dict:
    """What the record should say about the computer and the code."""
    out = {"python": platform.python_version(), "macos": platform.mac_ver()[0] or None}
    try:
        from tsync.system import machine, versions
        m = machine()
        out.update({"model": m.get("model"), "platform": m.get("platform")})
        out["versions"] = versions()
    except Exception:
        pass
    try:
        import subprocess
        r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=3)
        out["git_commit"] = r.stdout.strip() or None
        r = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "audiometry"],
                           capture_output=True, text=True, timeout=3)
        out["audiometry_dirty"] = bool(r.stdout.strip())
    except Exception:
        pass
    out["audiometry_source"] = source_hash()
    return out


def source_hash() -> str:
    import hashlib
    h = hashlib.sha256()
    pkg = Path(__file__).resolve().parent
    for p in sorted(pkg.rglob("*.py")):
        h.update(p.relative_to(pkg).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]
