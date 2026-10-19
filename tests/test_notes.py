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
