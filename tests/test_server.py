"""API smoke tests. The client is used without its context manager, so the server's startup work
(loading models, rendering demo samples) does not run here."""
import io
import time

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


def test_browser_mic_session_end_to_end():
    """The hosted live path: settings, audio frames, 'stop' -> summary and listen-back."""
    rng = np.random.default_rng(0)
    with client.websocket_connect("/ws/mic") as ws:
        ws.send_json({"mode": "gentle", "detector": "heuristic"})
        assert ws.receive_json()["ok"] is True
        for _ in range(100):  # 2 s of quiet noise in 20 ms frames
            ws.send_bytes((rng.standard_normal(320) * 0.01).astype(np.float32).tobytes())
        ws.send_text("stop")
        end = ws.receive_json()
    assert end["summary"]["mode_delay_ms"] > 0
    assert end["playback"]["available"] and end["playback"]["original_s"] > 1.5
    assert server.state["session"] is None  # one session at a time: the slot is free again


def test_browser_mic_sessions_end_themselves(monkeypatch):
    """A public demo has one live slot: a session without audio, or past the limit, gives it back."""
    monkeypatch.setattr(server, "MIC_IDLE_S", 0.3)
    with client.websocket_connect("/ws/mic") as ws:
        ws.send_json({"mode": "gentle", "detector": "heuristic"})
        ws.receive_json()
        end = ws.receive_json()  # sends nothing: the server ends it
    assert "No audio" in end["ended"] and server.state["session"] is None

    monkeypatch.setattr(server, "LIVE_LIMIT_S", 0.5)
    with client.websocket_connect("/ws/mic") as ws:
        ws.send_json({"mode": "gentle", "detector": "heuristic"})
        assert ws.receive_json()["limit_s"] == 0.5
        frame = np.zeros(320, np.float32).tobytes()
        for _ in range(100):  # 2 s of audio, sent faster than real time: the clock is the server's
            ws.send_bytes(frame)
            time.sleep(0.01)
        end = ws.receive_json()
    assert "minutes" in end["ended"] and end["playback"]["available"]
    assert server.state["session"] is None


def test_voice_notes_queue_is_bounded(monkeypatch):
    """A public server: a burst of uploads is turned away, not queued for ever."""
    monkeypatch.setattr(server.notes, "available", lambda: True)
    monkeypatch.setattr(server.notes, "_process", lambda nid: None)  # stays "queued", as if the GPU were busy
    monkeypatch.setattr(server.notes, "_notes", {})
    buf = io.BytesIO()
    wavfile.write(buf, 16000, (np.random.default_rng(0).standard_normal(32000) * 1000).astype(np.int16))
    codes = [client.post("/api/notes", content=buf.getvalue()).status_code for _ in range(server.notes.MAX_WAITING + 1)]
    assert codes == [200] * server.notes.MAX_WAITING + [429]


def test_hosted_nemotron_calls_are_capped(monkeypatch):
    """Visitors spend the owner's Nebius credits: past the per-minute budget they are asked to wait."""
    monkeypatch.setenv("NEBIUS_API_KEY", "test")
    monkeypatch.setattr(nemotron, "chat", lambda msgs, **k: '["one", "two", "three"]')
    monkeypatch.setattr(server, "NEMOTRON_PER_MIN", 2)
    monkeypatch.setattr(server, "_nemotron_calls", server.deque())
    codes = [client.post("/api/rescue", json={"said": "I am"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_hosted_sessions_are_private_to_their_page(monkeypatch):
    """On a public server, another visitor never gets a session's captions, waveforms or audio."""
    monkeypatch.setattr(server, "HOSTED", True)
    with client.websocket_connect("/ws/mic") as ws:
        ws.send_json({"mode": "gentle", "detector": "heuristic", "page": "alice"})
        ws.receive_json()
        assert server.sees_session("alice") and not server.sees_session("bob") and not server.sees_session("")
        for _ in range(100):
            ws.send_bytes(np.zeros(320, np.float32).tobytes())
        assert client.post("/api/stop").status_code == 403  # nobody else can end it
        ws.send_text("stop")
        assert ws.receive_json()["playback"]["available"]
    assert client.get("/api/playback?page=alice").json()["available"]
    assert not client.get("/api/playback?page=bob").json()["available"]
    assert not client.get("/api/playback").json()["available"]
    assert client.get("/api/playback/clean.wav?page=bob").status_code == 404
    assert client.get("/api/playback/clean.wav?page=alice").status_code == 200
