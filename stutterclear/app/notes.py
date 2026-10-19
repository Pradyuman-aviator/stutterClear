"""Clean my voice note: a recording made in the browser (or an uploaded file the browser decoded),
cleaned offline with the same engines as the demo: the teacher finds the stutters and the verbatim
transcript (CrisperWhisper) puts every cut on a word boundary.

Notes live in memory only, for an hour at most, and are never written to disk. One note is processed
at a time, because the teacher and CrisperWhisper together fill most of a laptop GPU.
"""
import io
import os
import threading
import time
import uuid
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from ..config import SAMPLE_RATE as SR
from ..silver import label_tokens
from ..teacher_runtime import RECORDINGS_CKPT as TEACHER  # the recordings teacher (may differ from live)

MAX_S = 180  # longest note accepted
KEEP_S = 3600  # notes are forgotten after an hour
MAX_WAITING = 4  # notes waiting for or on the GPU; more are turned away for a minute
MAX_KEPT = 40  # finished notes held in memory; the oldest are forgotten first
FRAME = 320

_notes: dict = {}
_gpu = threading.Lock()
_models: dict = {}
# Set once the server's warm-up (live teacher, demo renders) is over; until then a note waiting
# for the GPU would load CrisperWhisper next to the live teacher. Nothing to wait for by default.
warm = threading.Event()
warm.set()


class NoteError(ValueError):
    pass


class NoteBusy(RuntimeError):
    pass


def available() -> bool:
    """Voice notes need the teacher checkpoint (CrisperWhisper downloads itself on first use).
    STUTTERCLEAR_NOTES=0 turns them off, for a server without a GPU (they would take minutes)."""
    return os.environ.get("STUTTERCLEAR_NOTES") != "0" and TEACHER.exists()


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
    from ..teacher_runtime import get_teacher, recordings_device
    return get_teacher(TEACHER, recordings_device())  # shares the live model when it is the same checkpoint


def _crisper(device: str = "auto"):
    key = f"crisper-{device}"
    if key not in _models:
        from crisperwhisper import CrisperWhisperModel
        cpu = {} if device == "auto" else {"device": device, "compute_type": "float32"}
        _models[key] = _lean(CrisperWhisperModel("large", backend="transformers", **cpu))
    return _models[key]


def _lean(cw):
    """CrisperWhisper's transformers backend asks generate() for attention weights (its word timing
    reads the cross-attention), and Whisper then also keeps every encoder layer's self-attention:
    32 maps of 1500 x 1500, 2.9 GB that a 6 GB card does not have (Windows spills it into system
    memory and runs ten times slower). Running the encoder first, without them, gives the same
    words and times at 4.0 GB peak instead of 6.8: a 31 s note transcribes in 10 s."""
    model = getattr(getattr(cw, "_engine", None), "model", None)
    if model is None or not hasattr(model, "get_encoder"):
        return cw
    generate = model.generate

    def lean_generate(input_features=None, **kw):
        if kw.get("output_attentions") and "encoder_outputs" not in kw and input_features is not None:
            import torch
            with torch.no_grad():
                kw["encoder_outputs"] = model.get_encoder()(input_features, output_attentions=False, return_dict=True)
            return generate(None, **kw)  # Whisper takes the features or the encoder's output, not both
        return generate(input_features, **kw)

    model.generate = lean_generate
    return cw


NEED_GB = 4.5  # free GPU memory CrisperWhisper large needs to run there (4.0 GB peak, measured)


def live_running() -> bool:
    """Whether a live session holds the live teacher on the GPU (the server replaces this)."""
    return False


def _gpu_free_gb() -> float | None:
    """Free GPU memory, or None without a GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.mem_get_info()[0] / 1e9
    except ImportError:
        return None


def _gpu_ready() -> bool:
    """Whether CrisperWhisper can use the GPU. A 6 GB card cannot hold it next to the live teacher,
    and Windows does not fail then: it spills into system memory and runs many times slower. So the
    live teacher makes room when no session needs it (the next session loads it again); during a
    session the note goes to the CPU."""
    free = _gpu_free_gb()
    if free is None or "crisper-auto" in _models or free >= NEED_GB:
        return True
    if live_running():
        return False
    from ..teacher_runtime import release_gpu
    release_gpu()
    return (_gpu_free_gb() or 0.0) >= NEED_GB


def release_gpu():
    """Give the GPU back to a live session: drop CrisperWhisper's GPU copy, unless a note is being
    transcribed right now."""
    if not _gpu.acquire(blocking=False):
        return
    try:
        if _models.pop("crisper-auto", None) is not None:
            import torch
            torch.cuda.empty_cache()
    finally:
        _gpu.release()


def _transcribe(note: dict):
    """CrisperWhisper on the GPU when it fits, else on the CPU (slower, but the note gets cleaned).
    Running out of memory anyway also falls back to the CPU."""
    if _gpu_ready():
        try:
            return _crisper().transcribe(note["x"], sr=SR, language="en", word_timestamps=True)
        except Exception as e:
            if "out of memory" not in str(e).lower():
                raise
        import torch
        torch.cuda.empty_cache()
    note["slow"] = True  # the page says it will take longer
    return _crisper("cpu").transcribe(note["x"], sr=SR, language="en", word_timestamps=True)


def teacher_probs(x: np.ndarray, model=None, dev=None) -> np.ndarray:
    """Teacher probabilities [20 ms frames, 5 classes], in 30 s chunks with overlap. The app's teacher
    unless a model is given (scripts/calibrate_teacher.py compares checkpoints)."""
    import torch
    if model is None:
        model, dev = _teacher()
    elif dev is None:
        dev = next(model.parameters()).device
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
        warm.wait(timeout=300)
        with _gpu:
            note["status"] = "listening"  # the teacher looks for stutters
            note["probs"] = teacher_probs(note["x"])
            note["status"] = "transcribing"  # CrisperWhisper writes down every word, stutters included
            r = _transcribe(note)
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
    return {"id": nid, "status": note["status"], "error": note["error"], "slow": note.get("slow", False),
            "duration_s": round(len(note["x"]) / SR, 2)}


def ready(nid: str) -> dict | None:
    note = _notes.get(nid)
    return note if note and note["status"] == "done" else None


def expire():
    now = time.time()
    for nid in [k for k, v in _notes.items() if now - v["created"] > KEEP_S]:
        _notes.pop(nid, None)
