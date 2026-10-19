"""The big teacher (WavLM-Large) as a live detector, and as the shared model for offline cleaning.

Live, it runs on sliding windows: CONTEXT_S of past audio plus `lookahead_ms` ahead, re-run every
HOP_MS, deciding only the frames that have just received their lookahead. Measured on 7 held-out
UCLASS speakers (scripts/teacher_lookahead.py): with 200 ms of lookahead it keeps nearly all of its
offline quality (53.8% of stutters removed, 4.6% of words lost, vs 58.4% / 5.8% offline; the rule-
based detector removes 7.4%), and one 5 s window takes 60 ms on a laptop RTX 4050. Needs a GPU.
"""
import json
import os
import threading
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE
from .detector import DetectorEvent
from .segments import BALANCED_CUTS, CutConfig
from .student_runtime import CLASSES, REMOVABLE, StudentDetector

# STUTTERCLEAR_TEACHER points at another checkpoint (to evaluate or switch to a new teacher).
TEACHER_CKPT = Path(os.environ.get("STUTTERCLEAR_TEACHER")
                    or Path(__file__).resolve().parents[1] / "runs" / "kaggle_teacher" / "teacher" / "best.pt")
# Recordings (voice notes, demo clips) use their own teacher: v2 measured better there (more
# stutters confirmed, fewer normal words flagged), while v1 stays best live. Falls back to the live
# teacher; STUTTERCLEAR_RECORDINGS_TEACHER overrides.
_V2 = Path(__file__).resolve().parents[1] / "runs" / "kaggle_teacher_v2" / "teacher_v2" / "best.pt"
RECORDINGS_CKPT = Path(os.environ.get("STUTTERCLEAR_RECORDINGS_TEACHER") or (_V2 if _V2.exists() else TEACHER_CKPT))
FRAME = 320
CONTEXT_S = 4.0
HOP_MS = 160

_lock = threading.Lock()
_model = {}


def get_teacher(path: Path = TEACHER_CKPT, device: str | None = None):
    """(model, device), each checkpoint loaded once per device: live and recordings may use
    different teachers. Without a device: the GPU if there is one."""
    with _lock:
        key = (str(Path(path).resolve()), device)
        if key not in _model:
            import torch
            from .teacher import FrameTeacher
            dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model = FrameTeacher(ckpt["backbone"])
            model.load_state_dict(ckpt["state_dict"])
            _model[key] = (model.to(dev).eval(), dev)
        return _model[key]


def release_gpu():
    """Free the teachers on the GPU for another model that needs the room (CrisperWhisper on a 6 GB
    card); the next use loads them again. Only call it when no live session is running."""
    with _lock:
        for key in [k for k, (_, dev) in _model.items() if dev.type == "cuda"]:
            del _model[key]
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


def recordings_device() -> str | None:
    """Where the recordings teacher runs. None = the live teacher's own model (same checkpoint).
    A second teacher goes on the GPU only if it has room for both plus CrisperWhisper (10 GB+);
    otherwise on the CPU: a voice note is not live, so a slower one is fine."""
    if os.environ.get("STUTTERCLEAR_RECORDINGS_DEVICE"):
        return os.environ["STUTTERCLEAR_RECORDINGS_DEVICE"]
    if Path(RECORDINGS_CKPT).resolve() == Path(TEACHER_CKPT).resolve():
        return None
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 10e9:
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def calibration(path: Path = TEACHER_CKPT) -> dict:
    """Settings tuned for this checkpoint on validation speakers (scripts/calibrate_teacher.py), from
    live_cuts.json next to it: live cut settings per mode and the recordings' check threshold. Empty
    when there is none, and the defaults apply."""
    f = Path(path).with_name("live_cuts.json")
    try:
        return json.loads(f.read_text()) if f.exists() else {}
    except (OSError, ValueError):
        return {}


def live_capable(path: Path = TEACHER_CKPT) -> bool:
    """A checkpoint and a CUDA GPU: on CPU a window takes far longer than the 160 ms it has."""
    if not Path(path).exists():
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def window_probs(x: np.ndarray) -> np.ndarray:
    """Teacher probabilities [20 ms frames, 5 classes] for one window of audio."""
    import torch
    model, dev = get_teacher()
    with torch.no_grad(), torch.autocast(dev.type, enabled=dev.type == "cuda"):
        return torch.sigmoid(model(torch.from_numpy(np.ascontiguousarray(x, np.float32))[None].to(dev)).float())[0].cpu().numpy()


class TeacherDetector(StudentDetector):
    """Same online cut logic as the student detector (hysteresis, merging, block reports); the
    probabilities come from the teacher on sliding windows instead of a causal network."""

    def __init__(self, lookahead_ms: float = 200.0, cut_cfg: CutConfig = BALANCED_CUTS, hop_ms: float = HOP_MS,
                 context_s: float = CONTEXT_S, cut_interjections: bool = False):
        get_teacher()  # load now, not on the first word
        window_probs(np.zeros(SAMPLE_RATE, np.float32))  # and run once: the first GPU call is slow
        self.lookahead_ms = lookahead_ms
        self.lookahead_n = int(lookahead_ms * SAMPLE_RATE / 1000)
        self.hop_frames = max(1, int(hop_ms * SAMPLE_RATE / 1000 / FRAME))
        self.win = int((context_s + lookahead_ms / 1000) * SAMPLE_RATE)
        self.cfg = cut_cfg
        self.thresholds = np.array(cut_cfg.thresholds, dtype=float)
        removable = REMOVABLE + (("interjection",) if cut_interjections else ())
        self.cut_idx = [CLASSES.index(c) for c in removable]
        self.thresholds[self.cut_idx] = np.maximum(self.thresholds[self.cut_idx], self.MIN_CUT_THRESHOLD)
        self.close_after = max(2, cut_cfg.merge_gap)
        self.refine_frames = cut_cfg.refine_long_s * SAMPLE_RATE / FRAME
        self.keep = int(round(cut_cfg.prolong_keep_ms * SAMPLE_RATE / 1000 / FRAME))
        self.tail = int(round(cut_cfg.prolong_tail_ms * SAMPLE_RATE / 1000 / FRAME))
        self.cooldown = False
        self.block_idx = CLASSES.index("block")
        self.events: list[DetectorEvent] = []
        self.next_id = 0
        self.cut = None
        self.block = None
        self.frames_seen = 0
        self.buf = np.zeros(0, np.float32)  # the last window's worth of audio
        self.total = 0  # samples received
        self.decided = 0  # frames decided so far

    def decided_samples(self) -> int:
        return self.decided * FRAME

    def process(self, x: np.ndarray):
        x = np.asarray(x, np.float32).ravel()
        self.buf = np.concatenate([self.buf, x])[-(self.win + FRAME * 4):]
        self.total += len(x)
        ready = max(0, (self.total - self.lookahead_n - 400) // FRAME + 1)  # frames with full lookahead
        updates = []
        if ready - self.decided < self.hop_frames:
            return updates
        start = max(0, self.total - self.win) // FRAME * FRAME  # windows start on a frame boundary
        window = self.buf[len(self.buf) - (self.total - start):]
        if len(window) < 4000:
            return updates
        probs = window_probs(window)
        w0 = start // FRAME
        self.frames_seen = (self.total - 560) // FRAME  # so decision times are logged as "now"
        for f in range(self.decided, ready):
            j = f - w0
            p = probs[j] if 0 <= j < len(probs) else np.zeros(len(CLASSES), np.float32)
            self._cut_frame(f, p, updates)
            self._block_frame(f, p)
        self.decided = ready
        return updates
