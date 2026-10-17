"""Clean my voice note: a recording made in the browser (or an uploaded file the browser decoded),
cleaned offline with the same engines as the demo: the teacher finds the stutters and the verbatim
transcript (CrisperWhisper) puts every cut on a word boundary.

Notes live in memory only, for an hour at most, and are never written to disk. One note is processed
at a time, because the teacher and CrisperWhisper together fill most of a laptop GPU.
"""
import io
import threading
import time
import uuid
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from ..config import SAMPLE_RATE as SR
from ..silver import label_tokens
from ..teacher_runtime import TEACHER_CKPT as TEACHER  # one checkpoint for live and notes

MAX_S = 180  # longest note accepted
KEEP_S = 3600  # notes are forgotten after an hour
MAX_WAITING = 4  # notes waiting for or on the GPU; more are turned away for a minute
MAX_KEPT = 40  # finished notes held in memory; the oldest are forgotten first
FRAME = 320

_notes: dict = {}
_gpu = threading.Lock()
_models: dict = {}


class NoteError(ValueError):
    pass


class NoteBusy(RuntimeError):
    pass


def available() -> bool:
    """Voice notes need the teacher checkpoint (CrisperWhisper downloads itself on first use)."""
    return TEACHER.exists()


def decode(data: bytes) -> np.ndarray:
    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as e:
        raise NoteError("That file could not be read as audio.") from e
    x = x.mean(axis=1)
    if sr != SR:
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    if len(x) < SR:
        raise NoteError("That recording is shorter than a second.")
    if len(x) > MAX_S * SR:
        raise NoteError(f"Voice notes can be up to {MAX_S // 60} minutes long.")
    return x


def _teacher():
    from ..teacher_runtime import get_teacher
    return get_teacher(TEACHER)  # shared with the live detector: the GPU holds it once


def _crisper():
    if "crisper" not in _models:
        from crisperwhisper import CrisperWhisperModel
        _models["crisper"] = CrisperWhisperModel("large", backend="transformers")
    return _models["crisper"]


def teacher_probs(x: np.ndarray) -> np.ndarray:
    """Teacher probabilities [20 ms frames, 5 classes], in 30 s chunks with overlap."""
    import torch
    model, dev = _teacher()
    n, hop, ov = len(x), 30 * SR, 2 * SR
    total = np.zeros(((n - 400) // FRAME + 1, 5), np.float32)
    weight = np.zeros(len(total), np.float32)
    for start in range(0, n, hop - ov):
        seg = x[start:start + hop]
        if len(seg) < 4000:
            break
        with torch.no_grad(), torch.autocast(dev.type, enabled=dev.type == "cuda"):
            p = torch.sigmoid(model(torch.from_numpy(seg)[None].to(dev)).float())[0].cpu().numpy()
        f0 = start // FRAME
        k = min(len(p), len(total) - f0)
        total[f0:f0 + k] += p[:k]
        weight[f0:f0 + k] += 1
        if start + hop >= n:
            break
    return total / np.maximum(weight, 1)[:, None]


def _process(nid: str):
    note = _notes[nid]
    try:
        with _gpu:
            note["status"] = "listening"  # the teacher looks for stutters
            note["probs"] = teacher_probs(note["x"])
            note["status"] = "transcribing"  # CrisperWhisper writes down every word, stutters included
            r = _crisper().transcribe(note["x"], sr=SR, language="en", word_timestamps=True)
            note["toks"] = label_tokens([{"w": t.word, "s": float(t.start), "e": float(t.end)} for t in r.words])
        note["status"] = "done"
    except Exception as e:  # report it on the page instead of failing silently
        note["status"], note["error"] = "error", f"Cleaning failed: {e}"


def submit(data: bytes) -> str:
    expire()
    if sum(n["status"] not in ("done", "error") for n in _notes.values()) >= MAX_WAITING:
        raise NoteBusy("Several voice notes are being cleaned right now. Try again in a minute.")
    x = decode(data)
    finished = [k for k, v in _notes.items() if v["status"] in ("done", "error")]  # oldest first
    for nid in finished[:max(0, len(_notes) + 1 - MAX_KEPT)]:
        _notes.pop(nid, None)
    nid = "note-" + uuid.uuid4().hex[:12]
    _notes[nid] = {"status": "queued", "x": x, "created": time.time(), "error": None}
    threading.Thread(target=_process, args=(nid,), daemon=True).start()
    return nid


def status(nid: str) -> dict | None:
    note = _notes.get(nid)
    if note is None:
        return None
    return {"id": nid, "status": note["status"], "error": note["error"],
            "duration_s": round(len(note["x"]) / SR, 2)}


def ready(nid: str) -> dict | None:
    note = _notes.get(nid)
    return note if note and note["status"] == "done" else None


def expire():
    now = time.time()
    for nid in [k for k, v in _notes.items() if now - v["created"] > KEEP_S]:
        _notes.pop(nid, None)
