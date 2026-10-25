"""Clean my voice note: a recording made in the browser (or an uploaded file the browser decoded),
cleaned offline with the same engines as the demo: the teacher finds the stutters and the verbatim
transcript (CrisperWhisper) puts every cut on a word boundary.

Notes live in memory only, for an hour at most, and are never written to disk. One note is processed
at a time, because the teacher and CrisperWhisper together fill most of a laptop GPU.

A server without the recordings teacher (a CPU server in the cloud) sends the two GPU steps, the
teacher's probabilities and CrisperWhisper's words, to a GPU worker it wakes for them
(stutterclear/app/gpu.py); the cut decisions, the meaning guard and the cleaned audio stay here.
"""
import io
import os
import struct
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
# A public server: the page always uploads 16 kHz mono 16-bit WAV (web/src/lib/wav.ts), and that is all
# it reads, without an audio library, at most MAX_S of it. A personal computer takes any format
# soundfile reads (8-48 kHz, one or two channels).
HOSTED = os.environ.get("STUTTERCLEAR_HOSTED") == "1"
MAX_BYTES = 44 + MAX_S * SR * 2 + 4096 if HOSTED else 40_000_000  # an upload's size (a little room for more header)
RATES = (8000, 48000)
# Finished notes held in memory; the oldest are forgotten first (fewer on a small cloud server)
MAX_KEPT = 20 if HOSTED else 40
ABANDON_S = 600  # a note still waiting for a GPU worker is dropped once its page stopped asking
FRAME = 320
UNREADABLE = "That file could not be read as audio."

_notes: dict = {}
_lock = threading.Lock()  # a note's admission and its place in _notes, against concurrent uploads
_gpu = threading.Lock()
_models: dict = {}
# Set once the server's warm-up (live teacher, demo renders) is over; until then a note waiting
# for the GPU would load CrisperWhisper next to the live teacher. Nothing to wait for by default.
warm = threading.Event()
warm.set()


class NoteError(ValueError):
    pass


class NoteBusy(RuntimeError):
    """Turned away for now (429): the queue is full, or this visitor already has one being cleaned."""


class NoteUnavailable(RuntimeError):
    """No GPU can take a note now (503): the GPU worker rests or failed; the reason says which."""


def remote() -> bool:
    """Whether notes go to a GPU worker elsewhere: one is set up, and the recordings teacher is not
    on this machine."""
    from . import gpu
    return gpu.broker.configured() and not TEACHER.exists()


def available() -> bool:
    """Voice notes need the teacher checkpoint (CrisperWhisper downloads itself on first use), or a
    GPU worker to send them to. STUTTERCLEAR_NOTES=0 turns them off, for a server without a GPU
    (they would take minutes)."""
    return os.environ.get("STUTTERCLEAR_NOTES") != "0" and (TEACHER.exists() or remote())


def decode(data: bytes) -> np.ndarray:
    """An upload as 16 kHz mono float32, never more than MAX_S of it (a few hundred bytes can declare
    hours of audio, or a sample rate that would resample to gigabytes)."""
    x = pcm16_wav(data)
    if x is None:
        if HOSTED:
            raise NoteError(UNREADABLE)
        x = _any_format(data)
    return check_length(x)


def pcm16_wav(data: bytes) -> np.ndarray | None:
    """The samples of a 16 kHz mono 16-bit PCM WAV, read directly; None if data is anything else."""
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    pos, fmt = 12, None
    while pos + 8 <= len(data):
        kind, size = data[pos:pos + 4], int.from_bytes(data[pos + 4:pos + 8], "little")
        body = pos + 8
        if kind == b"fmt " and size >= 16 and body + 16 <= len(data):
            fmt = struct.unpack_from("<HHIIHH", data, body)  # format, channels, rate, bytes/s, block, bits
        elif kind == b"data":
            if fmt is None or (fmt[0], fmt[1], fmt[2], fmt[5]) != (1, 1, SR, 16):
                return None
            n = min(size, len(data) - body) // 2  # (a writer that could not fill in the size: to the end)
            return np.frombuffer(data, "<i2", count=n, offset=body).astype(np.float32) / 32768
        pos = body + size + (size & 1)
    return None


def _any_format(data: bytes) -> np.ndarray:
    """Any format soundfile reads (a personal computer only), within the limits before it is read."""
    try:
        with sf.SoundFile(io.BytesIO(data)) as f:
            sr, channels = f.samplerate, f.channels
            if not RATES[0] <= sr <= RATES[1] or not 1 <= channels <= 2:
                raise NoteError(f"{UNREADABLE[:-1]}: 8 to 48 kHz, one or two channels.")
            x = f.read(frames=MAX_S * sr + 1, dtype="float32", always_2d=True)  # at most MAX_S (and a sample)
    except NoteError:
        raise
    except Exception as e:
        raise NoteError(UNREADABLE) from e
    if len(x) > MAX_S * sr:
        raise NoteError(f"Recordings can be up to {MAX_S // 60} minutes long.")
    x = x.mean(axis=1)
    if sr != SR:
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def check_length(x: np.ndarray) -> np.ndarray:
    if len(x) < SR:
        raise NoteError("That recording is shorter than a second.")
    if len(x) > MAX_S * SR:
        raise NoteError(f"Recordings can be up to {MAX_S // 60} minutes long.")
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
        if remote():
            _analyse_remote(note)
        else:
            warm.wait(timeout=300)
            with _gpu:
                note["status"] = "listening"  # the teacher looks for stutters
                note["probs"] = teacher_probs(note["x"])
                note["status"] = "transcribing"  # CrisperWhisper writes down every word, stutters included
                r = _transcribe(note)
                note["toks"] = label_tokens([{"w": t.word, "s": float(t.start), "e": float(t.end)} for t in r.words])
        try:  # the journal (if the person started one) must never fail a note
            from .. import journal
            journal.record(journal.from_note(note["toks"], len(note["x"]) / SR))
        except Exception:
            pass
        note["status"] = "done"
    except Exception as e:  # report it on the page instead of failing silently
        from .gpu import GpuError  # the GPU worker's own sentences are written for the page already
        note["status"], note["error"] = "error", str(e) if isinstance(e, GpuError) else f"Cleaning failed: {e}"


def _analyse_remote(note: dict):
    """The teacher's probabilities and CrisperWhisper's words from a GPU worker elsewhere (it is woken
    if it sleeps); the note waits its turn in the worker's queue."""
    from . import gpu

    def on_update(stage: str, ahead: int | None):
        note["status"], note["ahead"] = stage, ahead

    probs, words, slow = gpu.broker.analyse(note["x"], on_update, abandoned=lambda: time.time() - note["seen"] > ABANDON_S,
                                            client=note.pop("client", ""))  # (the note keeps no address)
    note["probs"], note["toks"], note["slow"], note["ahead"] = probs, label_tokens(words), slow, None


def submit(data: bytes, client: str = "") -> str:
    """A recording as uploaded (decode() says which formats). client: who sent it (their address),
    for a GPU worker's fair share; kept only while the note is processed."""
    room(client)  # turned away before anything is decoded
    return _admit(decode(data), client)


def submit_audio(x: np.ndarray, client: str = "") -> str:
    """Audio already at 16 kHz mono (a video's soundtrack)."""
    return _admit(check_length(np.asarray(x, np.float32)), client)


def room(client: str = ""):
    """Whether a recording from this visitor could be taken now (the queue, the GPU worker, their fair
    share), before it is read: raises NoteBusy or NoteUnavailable with the reason."""
    _make_room(client, 0.0, dry_run=True)


def _admit(x: np.ndarray, client: str) -> str:
    with _lock:  # two uploads at once never both take the last place
        _make_room(client, len(x) / SR)
        return _add(x, client)


def _make_room(client: str = "", seconds: float = 0.0, dry_run: bool = False):
    expire()
    if sum(n["status"] not in ("done", "error") for n in list(_notes.values())) >= MAX_WAITING:
        raise NoteBusy("Several voice notes are being cleaned right now. Try again in a minute.")
    if remote():  # the GPU worker must be able to take it, and each visitor gets a fair share
        from . import gpu
        reason = gpu.broker.accepting(client)
        if reason:
            raise NoteUnavailable(reason)
        reason = gpu.broker.admit(client, seconds, dry_run=dry_run)
        if reason:
            raise NoteBusy(reason)


def _add(x: np.ndarray, client: str = "") -> str:
    finished = [k for k, v in list(_notes.items()) if v["status"] in ("done", "error")]  # oldest first
    for nid in finished[:max(0, len(_notes) + 1 - MAX_KEPT)]:
        _notes.pop(nid, None)
    nid = "note-" + uuid.uuid4().hex[:12]
    now = time.time()
    # renders: the cleaned versions and captions made from it (demo.py), forgotten with the note
    _notes[nid] = {"status": "queued", "x": x, "created": now, "error": None, "seen": now, "client": client,
                   "renders": {}, "lock": threading.Lock()}
    threading.Thread(target=_process, args=(nid,), daemon=True).start()
    return nid


def _get(nid: str) -> dict | None:
    """The note, unless its hour is over: then it is forgotten now, not at the next upload."""
    note = _notes.get(nid)
    if note is not None and time.time() - note["created"] > KEEP_S:
        _notes.pop(nid, None)
        return None
    return note


def touch(nid: str):
    """The page asked about this note: it still wants it (a note nobody asks about any more is not
    cleaned once it has waited ABANDON_S for a GPU worker)."""
    note = _get(nid)
    if note is not None:
        note["seen"] = time.time()


def status(nid: str) -> dict | None:
    note = _get(nid)
    if note is None:
        return None
    return {"id": nid, "status": note["status"], "error": note["error"], "slow": note.get("slow", False),
            "duration_s": round(len(note["x"]) / SR, 2), "ahead": note.get("ahead")}


def ready(nid: str) -> dict | None:
    note = _get(nid)
    return note if note and note["status"] == "done" else None


def expire():
    """Forget notes past their hour, with everything made from them (the server also calls this every
    minute, so a quiet server keeps nothing longer)."""
    now = time.time()
    for nid, note in list(_notes.items()):  # (uploads add notes meanwhile)
        if now - note["created"] > KEEP_S:
            _notes.pop(nid, None)
