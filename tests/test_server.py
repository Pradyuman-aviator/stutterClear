"""API smoke tests. The client is used without its context manager, so the server's startup work
(loading models, rendering demo samples) does not run here."""
import io

import numpy as np
from fastapi.testclient import TestClient
from scipy.io import wavfile

from stutterclear import nemotron
from stutterclear.app import server

client = TestClient(server.app)


def test_info_reports_features():
    r = client.get("/api/info")
    assert r.status_code == 200
    d = r.json()
    for key in ("notes", "captions", "nemotron", "search", "detectors", "recommended"):
        assert key in d
    assert {x["kind"] for x in d["detectors"]} == {"teacher", "student", "heuristic"}


def test_nemotron_features_say_how_to_enable_them(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    for url, body in (("/api/rescue", {"said": "I want"}), ("/api/context", {"situation": "interview"}),
                      ("/api/insights", {"kinds": {}, "duration_s": 1, "removed_s": 0})):
        r = client.post(url, json=body)
        assert r.status_code == 503 and "NEBIUS_API_KEY" in r.json()["detail"]


def test_nemotron_features_with_a_stand_in_model(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(nemotron, "chat", lambda msgs, **k: '["Kaggle", "keyboard", "kind"]'
                        if "stuck" in msgs[0]["content"] else "- One.\n- Two.\n- Three.")
    assert client.post("/api/rescue", json={"said": "I trained it on", "context": ""}).json()["suggestions"] == \
        ["Kaggle", "keyboard", "kind"]
    assert client.post("/api/insights", json={"kinds": {"word_rep": 2}, "duration_s": 30, "removed_s": 1.0}).json()[
        "points"] == ["One.", "Two.", "Three."]
    assert "card" in client.post("/api/context", json={"situation": "Meeting with Dr. Mehta"}).json()


def test_voice_notes_reject_what_is_not_audio():
    if not server.notes.available():
        return  # needs the teacher checkpoint
    assert client.post("/api/notes", content=b"not audio").status_code == 415
    buf = io.BytesIO()
    wavfile.write(buf, 16000, np.zeros(1000, np.int16))  # 1/16 s
    r = client.post("/api/notes", content=buf.getvalue())
    assert r.status_code == 415 and "shorter than a second" in r.json()["detail"]


def test_playback_and_unknown_ids():
    assert client.get("/api/playback").status_code == 200
    assert client.get("/api/notes/note-nope").status_code == 404
    assert client.get("/api/demo/note-nope?engine=natural").status_code == 404
    assert client.get("/api/demo/no-such-sample").status_code == 404
