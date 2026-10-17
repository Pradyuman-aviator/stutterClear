"""A live StutterClear session for the app: mic -> detector -> splicer -> output device.

poll() hands the UI everything new since the last call: input and output level envelopes
(100 points per second), cuts, detector events, delay and counters.
"""
import ctypes
import json
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

from .audio import save_wav
from .config import HOP, SAMPLE_RATE as SR, preset
from .pipeline import Pipeline
from .segments import CutConfig

# Off: nothing is cut. Gentle: the "balanced" cut rules, 250 ms delay. Strong: lower thresholds
# (removes the most stutters, risks more words), 600 ms delay.
STRONG_CUTS = CutConfig(thresholds=(0.3, 0.25, 0.4, 0.5, 0.3), off_ratio=0.5, merge_gap=5, snap=0,
                        refine_long_s=1e9, prolong_keep_ms=100.0)
MODES = {"off": ("verbatim", None, 0), "gentle": ("trim", "gentle", 200), "strong": ("clean", "strong", 500)}
DEFAULT_STUDENT = Path(__file__).resolve().parents[1] / "runs" / "kaggle_student" / "student" / "student.pt"
PLAYBACK_MAX_S = 600  # the app keeps the last 10 minutes in memory so you can play it back


class Recorder:
    """Audio blocks kept in memory, oldest dropped past a cap. Appends come from audio threads."""

    def __init__(self, max_s: float = PLAYBACK_MAX_S):
        self.blocks: deque = deque()
        self.n = 0
        self.cap = int(max_s * SR)

    def add(self, x: np.ndarray):
        self.blocks.append(x)
        self.n += len(x)
        while self.n > self.cap and len(self.blocks) > 1:
            self.n -= len(self.blocks.popleft())

    def audio(self) -> np.ndarray:
        blocks = list(self.blocks)
        return np.concatenate(blocks) if blocks else np.zeros(0, np.float32)


def student_eval(path: Path = DEFAULT_STUDENT) -> dict | None:
    """Measured quality of a student checkpoint, from scripts/eval_live.py saved next to it as
    live_eval.json: stutters removed and words lost in Gentle mode on held-out speakers."""
    p = Path(path).with_name("live_eval.json")
    if not p.exists():
        return None
    for r in json.loads(p.read_text()):
        if r["detector"] == "student" and r["mode"] == "gentle":
            return {"removed_pct": r["stutter_removed_pct"], "words_lost_pct": r["words_lost_pct"]}
    return None


def wasapi_index() -> int:
    for i, h in enumerate(sd.query_hostapis()):
        if h["name"] == "Windows WASAPI":
            return i
    return sd.default.hostapi


def devices() -> dict:
    idx = wasapi_index()
    out = {"inputs": [], "outputs": [], "default_input": sd.query_hostapis(idx)["default_input_device"]}
    for i, d in enumerate(sd.query_devices()):
        if d["hostapi"] != idx:
            continue
        if d["max_input_channels"] > 0:
            out["inputs"].append({"index": i, "name": d["name"]})
        if d["max_output_channels"] > 0:
            out["outputs"].append({"index": i, "name": d["name"]})
    return out


def make_detector(kind: str, mode: str, student_path=None):
    preset_name, cuts, lookahead_ms = MODES[mode]
    path = Path(student_path) if student_path else DEFAULT_STUDENT
    if kind == "student" and path.exists() and cuts:
        from .student_runtime import LIVE_CUTS, StudentDetector
        return StudentDetector(path, lookahead_ms, path.with_name("student_test_report.json"),
                               cut_cfg=LIVE_CUTS if cuts == "gentle" else STRONG_CUTS), "student"
    return None, "rule-based"  # Pipeline falls back to the heuristic detector


class LiveSession:
    def __init__(self, mode="gentle", in_device=None, out_device=None, detector="student", student_path=None,
                 save=False):
        self.mode = mode
        preset_name, _, _ = MODES[mode]
        det, self.detector_name = make_detector(detector, mode, student_path)
        self.pipe = Pipeline(preset(preset_name), detector=det)
        self.in_device = in_device if in_device is not None else devices()["default_input"]
        self.out_device = out_device
        self.save = save
        self.q: queue.Queue = queue.Queue()
        # Kept in memory for the play buttons, dropped with the session; written to disk only
        # when the user switched saving on.
        self.raw, self.out = Recorder(), Recorder()
        self.env_in, self.env_out = [], []  # 10 ms peak envelopes
        self.detect_ms = []
        self.running = False
        self.started = None
        self._sent = {"env_in": 0, "env_out": 0, "cuts": 0, "events": 0}

    # ---- audio callbacks and worker -------------------------------------------------------

    def _on_input(self, indata, frames, t, status):
        x = indata[:, 0].copy()
        self.pipe.write(x)
        self.raw.add(x)
        self.env_in.append(float(np.abs(x).max()) if len(x) else 0.0)
        self.q.put(x)

    def _emit(self, y):
        self.env_out.append(float(np.abs(y).max()) if len(y) else 0.0)
        self.out.add(y.copy())

    def _on_output(self, outdata, frames, t, status):
        y = self.pipe.read(frames)
        outdata[:, 0] = y
        self._emit(y)

    def _worker(self):
        while self.running or not self.q.empty():
            try:
                x = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            self.pipe.detect(x)
            self.detect_ms.append((time.perf_counter() - t0) * 1000)
            if self.out_device is None:
                self._emit(self.pipe.read(len(x)))  # no device: the input clock drives the output

    # ---- control ----------------------------------------------------------------------------

    def start(self):
        if sys.platform == "win32":
            # WASAPI streams fail to open from a thread without COM (web servers call us from
            # worker threads). The result is ignored: S_FALSE / changed-mode mean COM is already up.
            ctypes.windll.ole32.CoInitializeEx(None, 0)
        settings = sd.WasapiSettings(auto_convert=True)
        self.running, self.started = True, time.time()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.streams = [sd.InputStream(device=self.in_device, samplerate=SR, channels=1, blocksize=HOP,
                                       dtype="float32", latency="low", extra_settings=settings,
                                       callback=self._on_input)]
        if self.out_device is not None:
            self.streams.append(sd.OutputStream(device=self.out_device, samplerate=SR, channels=1, blocksize=HOP,
                                                dtype="float32", latency="low", extra_settings=settings,
                                                callback=self._on_output))
        for s in self.streams:
            s.start()

    def stop(self) -> dict:
        for s in getattr(self, "streams", []):
            s.stop()
            s.close()
        self.running = False
        self.thread.join(timeout=2)
        summary = self.pipe.summary()
        if self.save and self.raw.n:  # files are written only when the user switched saving on
            folder = Path(__file__).resolve().parents[1] / "sessions" / datetime.now().strftime("%Y%m%d-%H%M%S")
            save_wav(folder / "raw.wav", self.raw.audio())
            save_wav(folder / "out.wav", self.out.audio() if self.out.n else np.zeros(1))
            summary["saved_to"] = str(folder)
        return summary

    def recording(self) -> dict:
        """What you said and what the listener heard, for playback after the session."""
        return {"original": self.raw.audio(), "clean": self.out.audio()}

    # ---- what the UI shows ------------------------------------------------------------------

    def poll(self) -> dict:
        st = self.pipe.splicer.stats
        spans = st.removed_spans
        events = getattr(self.pipe.detector, "events", [])
        msg = {
            "t": round(time.time() - self.started, 2) if self.started else 0.0,
            "lag_ms": round(self.pipe.splicer.lag_ms()),
            "level_db": round(20 * np.log10(max(self.env_in[-1], 1e-9)), 1) if self.env_in else -120.0,
            "removed_s": round(st.removed / SR, 2),
            "n_cuts": len(spans),
            "detector": self.detector_name,
            "detect_ms": round(float(np.mean(self.detect_ms[-50:])), 2) if self.detect_ms else 0.0,
            # Index of the first envelope point below, so a page that joins mid-session lines up
            # its waveform with the cut times (both count from the start of the session).
            "in_start": self._sent["env_in"],
            "out_start": self._sent["env_out"],
            "env_in": self.env_in[self._sent["env_in"]:],
            "env_out": self.env_out[self._sent["env_out"]:],
            "cuts": [{"start": s / SR, "end": e / SR, "kind": k} for s, e, k in spans[self._sent["cuts"]:]],
            "events": [{"kind": ev.kind, "start": ev.start / SR, "end": ev.end / SR}
                       for ev in events[self._sent["events"]:] if ev.kind in ("block", "pause")],
        }
        self._sent = {"env_in": self._sent["env_in"] + len(msg["env_in"]),
                      "env_out": self._sent["env_out"] + len(msg["env_out"]),
                      "cuts": len(spans), "events": len(events)}
        return msg
