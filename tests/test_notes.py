"""Voice notes: a full GPU must not lose the note."""
import time
from types import SimpleNamespace

import numpy as np

from stutterclear.app import notes


def test_a_full_gpu_falls_back_to_the_cpu(monkeypatch):
    calls = []

    class Model:
        def __init__(self, device):
            self.device = device

        def transcribe(self, x, **k):
            calls.append(self.device)
            if self.device == "auto":
                raise RuntimeError("CUDA error: out of memory")
            words = [SimpleNamespace(word="hi", start=0.1, end=0.3), SimpleNamespace(word="hi", start=0.4, end=0.6)]
            return SimpleNamespace(words=words)

    monkeypatch.setattr(notes, "_crisper", lambda device="auto": Model(device))
    monkeypatch.setattr(notes, "teacher_probs", lambda x: np.zeros((50, 5), np.float32))
    monkeypatch.setattr(notes, "_notes", {"note-x": {"status": "queued", "x": np.zeros(16000, np.float32),
                                                     "created": time.time(), "error": None}})
    notes._process("note-x")
    note = notes._notes["note-x"]
    assert note["status"] == "done" and calls == ["auto", "cpu"] and notes.status("note-x")["slow"]
    assert [t["label"] for t in note["toks"]] == ["repeat", "fluent"]


def test_other_errors_are_reported(monkeypatch):
    class Broken:
        def transcribe(self, x, **k):
            raise ValueError("bad audio")

    monkeypatch.setattr(notes, "_crisper", lambda device="auto": Broken())
    monkeypatch.setattr(notes, "teacher_probs", lambda x: np.zeros((50, 5), np.float32))
    monkeypatch.setattr(notes, "_notes", {"note-y": {"status": "queued", "x": np.zeros(16000, np.float32),
                                                     "created": time.time(), "error": None}})
    notes._process("note-y")
    assert notes._notes["note-y"]["status"] == "error" and "bad audio" in notes._notes["note-y"]["error"]
