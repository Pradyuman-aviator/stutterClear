"""Voice notes: a full GPU must not lose the note, nor slow it to a crawl."""
import time
from types import SimpleNamespace

import numpy as np

from stutterclear import teacher_runtime
from stutterclear.app import notes


class Model:
    def __init__(self, device, calls, oom=False):
        self.device, self.calls, self.oom = device, calls, oom

    def transcribe(self, x, **k):
        self.calls.append(self.device)
        if self.oom and self.device == "auto":
            raise RuntimeError("CUDA error: out of memory")
        words = [SimpleNamespace(word="hi", start=0.1, end=0.3), SimpleNamespace(word="hi", start=0.4, end=0.6)]
        return SimpleNamespace(words=words)


def run(monkeypatch, free, live=False, oom=False):
    """Process one note with a stand-in CrisperWhisper; `free` is GPU memory (GB) before and after
    the live teacher is released. Returns (devices used, note, whether the teacher was released)."""
    calls, released, room = [], [], list(free)
    monkeypatch.setattr(notes, "_crisper", lambda device="auto": Model(device, calls, oom))
    monkeypatch.setattr(notes, "_models", {})
    monkeypatch.setattr(notes, "teacher_probs", lambda x: np.zeros((50, 5), np.float32))
    monkeypatch.setattr(notes, "_gpu_free_gb", lambda: room[0])
    monkeypatch.setattr(notes, "live_running", lambda: live)
    monkeypatch.setattr(teacher_runtime, "release_gpu", lambda: (released.append(1), room.pop(0) if len(room) > 1 else None))
    monkeypatch.setattr(notes, "_notes", {"note-x": {"status": "queued", "x": np.zeros(16000, np.float32),
                                                     "created": time.time(), "error": None}})
    notes._process("note-x")
    return calls, notes._notes["note-x"], bool(released)


def test_a_full_gpu_falls_back_to_the_cpu(monkeypatch):
    calls, note, _ = run(monkeypatch, free=[8.0], oom=True)
    assert note["status"] == "done" and calls == ["auto", "cpu"] and notes.status("note-x")["slow"]
    assert [t["label"] for t in note["toks"]] == ["repeat", "fluent"]


def test_an_idle_live_teacher_makes_room(monkeypatch):
    calls, note, released = run(monkeypatch, free=[1.5, 6.0])
    assert released and calls == ["auto"] and not note.get("slow")


def test_during_a_live_session_the_note_goes_to_the_cpu(monkeypatch):
    calls, note, released = run(monkeypatch, free=[1.5, 6.0], live=True)
    assert not released and calls == ["cpu"] and note["slow"] and note["status"] == "done"


def test_a_live_session_takes_the_gpu_back_unless_a_note_is_running(monkeypatch):
    monkeypatch.setattr(notes, "_models", {"crisper-auto": object()})
    with notes._gpu:  # a note is being transcribed
        notes.release_gpu()
    assert "crisper-auto" in notes._models
    notes.release_gpu()
    assert "crisper-auto" not in notes._models


def test_other_errors_are_reported(monkeypatch):
    class Broken:
        def transcribe(self, x, **k):
            raise ValueError("bad audio")

    monkeypatch.setattr(notes, "_crisper", lambda device="auto": Broken())
    monkeypatch.setattr(notes, "_gpu_free_gb", lambda: 8.0)
    monkeypatch.setattr(notes, "teacher_probs", lambda x: np.zeros((50, 5), np.float32))
    monkeypatch.setattr(notes, "_notes", {"note-y": {"status": "queued", "x": np.zeros(16000, np.float32),
                                                     "created": time.time(), "error": None}})
    notes._process("note-y")
    assert notes._notes["note-y"]["status"] == "error" and "bad audio" in notes._notes["note-y"]["error"]


def test_lean_crisper_runs_the_encoder_without_attention_maps():
    seen = {}

    class Encoder:
        def __call__(self, feats, output_attentions, return_dict):
            seen["encoder_attentions"] = output_attentions
            return "encoded"

    class Model:
        def get_encoder(self):
            return Encoder()

        def generate(self, input_features=None, **kw):
            seen["features"], seen["kw"] = input_features, kw
            return "out"

    cw = SimpleNamespace(_engine=SimpleNamespace(model=Model()))
    notes._lean(cw)
    gen = cw._engine.model.generate
    assert gen("feats", output_attentions=True, max_new_tokens=5) == "out"
    assert seen["encoder_attentions"] is False and seen["features"] is None
    assert seen["kw"] == {"output_attentions": True, "max_new_tokens": 5, "encoder_outputs": "encoded"}
    gen("feats", max_new_tokens=5)  # no attention asked for: untouched
    assert seen["features"] == "feats" and "encoder_outputs" not in seen["kw"]


# ---- what an upload may hold ----------------------------------------------------------------------

def wav_bytes(x: np.ndarray, sr: int = 16000, channels: int = 1, extra: bytes = b"") -> bytes:
    """A 16-bit PCM WAV, with any chunks before the samples (`extra`), written by hand."""
    import struct
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    fmt = struct.pack("<HHIIHH", 1, channels, sr, sr * 2 * channels, 2 * channels, 16)
    body = b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt + extra + b"data" + struct.pack("<I", len(pcm)) + pcm
    return b"RIFF" + struct.pack("<I", len(body)) + body


def test_the_pages_wav_is_read_directly():
    x = np.random.default_rng(0).uniform(-0.5, 0.5, 32000).astype(np.float32)
    y = notes.pcm16_wav(wav_bytes(x, extra=b"LIST" + (5).to_bytes(4, "little") + b"abcde\0"))  # (odd chunk padded)
    assert y is not None and len(y) == 32000 and np.allclose(y, x, atol=1e-4)
    assert notes.pcm16_wav(wav_bytes(x, sr=44100)) is None and notes.pcm16_wav(wav_bytes(x, channels=2)) is None
    assert notes.pcm16_wav(b"not audio at all, but long enough to look at its first bytes") is None


def test_a_public_server_reads_nothing_but_the_pages_wav(monkeypatch):
    monkeypatch.setattr(notes, "HOSTED", True)
    x = np.zeros(32000, np.float32)
    assert len(notes.decode(wav_bytes(x))) == 32000
    for other in (wav_bytes(x, sr=8000), wav_bytes(x, channels=2), b"fLaC" + b"\0" * 100):
        try:
            notes.decode(other)
        except notes.NoteError as e:
            assert str(e) == notes.UNREADABLE  # no audio library ever parses what a visitor sent
        else:
            raise AssertionError("decoded")


def flac(seconds: float, sr: int, channels: int) -> bytes:
    """Silence as FLAC, written a second at a time (it compresses to almost nothing)."""
    import io
    import soundfile as sf
    buf = io.BytesIO()
    with sf.SoundFile(buf, "w", sr, channels, format="FLAC") as f:
        for _ in range(int(seconds)):
            f.write(np.zeros((sr, channels), np.float32))
    return buf.getvalue()


def test_a_file_that_claims_hours_of_audio_is_never_read_whole():
    for data, said in ((flac(1, 48000, 8), "8 to 48 kHz, one or two channels"),  # 8 channels: refused unread
                       (wav_bytes(np.zeros(20000, np.float32), sr=1), "8 to 48 kHz"),  # 1 Hz: 16000 times longer
                       (wav_bytes(np.zeros(20000, np.float32), sr=192000), "8 to 48 kHz"),
                       (flac(notes.MAX_S + 60, 44100, 1), "up to 3 minutes")):  # only MAX_S of it is ever read
        t = time.perf_counter()
        try:
            notes.decode(data)
        except notes.NoteError as e:
            assert said in str(e), str(e)
        else:
            raise AssertionError("decoded")
        assert time.perf_counter() - t < 3


def test_a_note_past_its_hour_is_gone_even_if_nothing_else_happens(monkeypatch):
    now = time.time()
    monkeypatch.setattr(notes, "_notes", {"note-old": {"status": "done", "x": np.zeros(16000, np.float32),
                                                       "created": now - notes.KEEP_S - 1, "error": None, "seen": now},
                                          "note-new": {"status": "done", "x": np.zeros(16000, np.float32),
                                                       "created": now, "error": None, "seen": now}})
    assert notes.ready("note-old") is None and notes.status("note-old") is None and "note-old" not in notes._notes
    assert notes.ready("note-new") is not None
