"""The audio engine: one output stream, open for the whole session, onsets scheduled to the sample.

Why not `sounddevice.play()` per trial: it opens and closes a stream every time, so the delay
from call to sound varies from trial to trial and nothing tells you what it was. Here a single
PortAudio stream runs from start to end, writing silence when idle. A trial is handed to the
callback with the exact frame at which it must begin; the callback copies samples (nothing else
happens in the real-time thread) and stamps the DAC time at which the first and last samples
leave the converter. Those stamps, mapped onto the same clock as the key presses, give every
trial's true onset time, the response time from the actual end of the sound, and the lag of
the visual cues.

Reliability, not just latency: the callback records every PortAudio status flag (output
underflow means a gap was played) with the frame it happened at, and a trial whose span saw one,
or which could not start on time, is reported as such so the script can run it again.

Inside a trial nothing the driver does can move a tone: both sequences are one buffer, so the
asynchrony and the gap between them are fixed by the samples.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


class AudioError(RuntimeError):
    pass


def list_outputs() -> List[dict]:
    import sounddevice as sd
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0:
            api = sd.query_hostapis(d["hostapi"])["name"]
            out.append({"index": i, "name": d["name"], "hostapi": api,
                        "channels": d["max_output_channels"],
                        "default_samplerate": d["default_samplerate"],
                        "low_latency_s": d["default_low_output_latency"]})
    return out


def find_output(match: str) -> dict:
    """The output device whose name contains `match`; never silently another one."""
    outs = list_outputs()
    hits = [d for d in outs if match.lower() in d["name"].lower()]
    if not hits:
        names = ", ".join(f"'{d['name']}'" for d in outs) or "none"
        raise AudioError(f"no output device matching '{match}'. Plug the headphones into the "
                         f"Mac's headphone jack. Available outputs: {names}")
    return hits[0]


@dataclass
class Playback:
    buf: np.ndarray
    start_frame: int
    sr: int
    n: int = 0
    started: bool = False
    finished: bool = False
    late: bool = False
    aborted: bool = False
    abort_frame: Optional[int] = None
    abort_seen: bool = False
    onset_dac: Optional[float] = None       # stream clock, s
    end_dac: Optional[float] = None
    flags: List[Tuple[int, str]] = field(default_factory=list)
    cb_gaps_ms: List[float] = field(default_factory=list)
    t_scheduled_perf: float = 0.0
    clock_offset: float = 0.0               # perf_counter - stream time, when scheduled

    def __post_init__(self):
        self.n = int(self.buf.shape[0])

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.n

    def onset_perf(self) -> Optional[float]:
        return None if self.onset_dac is None else self.onset_dac + self.clock_offset

    def report(self) -> dict:
        lat = None
        if self.onset_dac is not None:
            lat = (self.onset_dac + self.clock_offset - self.t_scheduled_perf) * 1000.0
        under = sum(1 for _, s in self.flags if "underflow" in s)
        return {"start_frame": self.start_frame, "n_frames": self.n,
                "onset_dac_s": self.onset_dac, "end_dac_s": self.end_dac,
                "onset_error_ms": lat, "late": self.late, "aborted": self.aborted,
                "underflows": under, "flags": [f"{f}:{s}" for f, s in self.flags],
                "callback_gap_max_ms": max(self.cb_gaps_ms) if self.cb_gaps_ms else None,
                "clock_offset_s": self.clock_offset}


class AudioEngine:
    FADE_S = 0.010        # an abort fades out over this, so a pause never clicks

    def __init__(self, device: Optional[int], samplerate: int = 48000, channels: int = 2,
                 blocksize: int = 256, latency="low"):
        self.device, self.sr, self.channels = device, int(samplerate), int(channels)
        self.blocksize, self.latency = int(blocksize), latency
        self._stream = None
        self._pb: Optional[Playback] = None
        self._frame = 0
        self._last: Optional[Tuple[int, float, float]] = None   # (frame, dac time, perf)
        self.n_callbacks = 0
        self.flags: List[Tuple[int, str]] = []
        self.error: Optional[str] = None
        self._fade = max(1, int(self.FADE_S * self.sr))

    # -- lifecycle ------------------------------------------------------------------------------
    def start(self, timeout_s: float = 2.0) -> dict:
        import sounddevice as sd
        try:
            sd.check_output_settings(device=self.device, channels=self.channels,
                                     dtype="float32", samplerate=self.sr)
            self._stream = sd.OutputStream(device=self.device, samplerate=self.sr,
                                           channels=self.channels, dtype="float32",
                                           blocksize=self.blocksize, latency=self.latency,
                                           callback=self._callback,
                                           finished_callback=self._finished)
            self._stream.start()
        except Exception as e:      # PortAudioError and friends
            raise AudioError(f"could not open the audio output: {e}") from e
        t_end = time.perf_counter() + timeout_s
        while self.n_callbacks < 4:
            if time.perf_counter() > t_end:
                raise AudioError("the audio stream started but its callback never ran")
            time.sleep(0.005)
        return self.describe()

    def describe(self) -> dict:
        s = self._stream
        return {"device": self.device, "samplerate": s.samplerate if s else None,
                "blocksize": self.blocksize, "latency_s": s.latency if s else None,
                "dtype": "float32", "channels": self.channels}

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        self._stream = None

    def _finished(self) -> None:
        if self.error is None:
            self.error = "the audio stream stopped"

    @property
    def alive(self) -> bool:
        return self._stream is not None and self._stream.active and self.error is None

    # -- the real-time callback: copy, stamp, nothing else ----------------------------------------
    def _callback(self, outdata, frames, time_info, status) -> None:
        t_perf = time.perf_counter()
        f0 = self._frame
        dac = time_info.outputBufferDacTime or (time_info.currentTime + (self._stream.latency
                                                                         if self._stream else 0))
        if status:
            self.flags.append((f0, str(status)))
        outdata.fill(0.0)
        pb = self._pb
        if pb is not None and not pb.finished:
            if self._last is not None:
                pb.cb_gaps_ms.append((t_perf - self._last[2]) * 1000.0)
            if status:
                pb.flags.append((f0, str(status)))
            s, e = pb.start_frame, pb.end_frame
            if pb.abort_frame is not None:
                if not pb.abort_seen:
                    pb.abort_seen = True
                    if pb.abort_frame < f0:           # requested for a block already played
                        pb.abort_frame = f0
                e = min(e, pb.abort_frame + self._fade)
            if not pb.started and f0 > s:
                pb.late = pb.finished = True          # its first sample is already gone
            else:
                a, b = max(f0, s), min(f0 + frames, e)
                if a < b:
                    seg = pb.buf[a - s:b - s]
                    if pb.abort_frame is not None:
                        idx = np.arange(a, b)
                        g = np.clip(1.0 - (idx - pb.abort_frame) / self._fade, 0.0, 1.0)
                        seg = seg * g[:, None].astype(np.float32)
                    outdata[a - f0:b - f0] = seg
                    if not pb.started:
                        pb.started = True
                        pb.onset_dac = dac + (s - f0) / self.sr
                if f0 + frames >= e and (pb.started or pb.abort_frame is not None):
                    pb.end_dac = dac + (e - f0) / self.sr
                    pb.aborted = pb.abort_frame is not None and e < pb.end_frame
                    pb.finished = True
        self._last = (f0, dac, t_perf)
        self._frame = f0 + frames
        self.n_callbacks += 1

    # -- scheduling ------------------------------------------------------------------------------
    def clock_offset(self) -> float:
        """perf_counter() minus the stream clock, from the tightest of a few paired readings."""
        best, off = None, 0.0
        for _ in range(5):
            a = time.perf_counter()
            st = self._stream.time
            b = time.perf_counter()
            if best is None or b - a < best:
                best, off = b - a, (a + b) / 2.0 - st
        return off

    def frame_at(self, stream_t: float) -> int:
        f0, dac, _ = self._last
        return int(round(f0 + (stream_t - dac) * self.sr))

    def play_at(self, buf: np.ndarray, t_perf: float) -> Playback:
        """Schedule `buf` so its first sample leaves the DAC at perf_counter time `t_perf`."""
        if not self.alive:
            raise AudioError(self.error or "the audio stream is not running")
        if self._pb is not None and not self._pb.finished:
            raise AudioError("a trial is already playing")
        buf = np.ascontiguousarray(buf, dtype=np.float32)
        if buf.ndim != 2 or buf.shape[1] != self.channels:
            raise ValueError(f"buffer must be (n, {self.channels})")
        off = self.clock_offset()
        start = self.frame_at(t_perf - off)
        pb = Playback(buf, start, self.sr, t_scheduled_perf=t_perf, clock_offset=off)
        self._pb = pb
        return pb

    def frame_to_perf(self, pb: Playback, frame: int) -> float:
        f0, dac, _ = self._last
        return dac + (frame - f0) / self.sr + pb.clock_offset

    def abort(self, pb: Playback) -> None:
        if pb.finished:
            return
        if not pb.started:
            pb.abort_frame = pb.start_frame
            pb.aborted = pb.finished = True
            return
        f0, _, _ = self._last
        pb.abort_frame = f0 + self.blocksize

    def silence(self) -> None:
        pb = self._pb
        if pb is not None and not pb.finished:
            self.abort(pb)
