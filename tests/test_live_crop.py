"""Live, cropped: a call hears a short pause where each stutter was cut (it cannot play what was not
said yet); recordings, the demo clips and the listen-back after a session, close that time up, and
keep the call's own version as "heard"."""
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from stutterclear import live
from stutterclear.app import demo, server
from stutterclear.config import SAMPLE_RATE as SR
from stutterclear.detector import HeuristicDetector

client = TestClient(server.app)


def test_the_live_demo_closes_up_what_a_call_hears_as_pauses(monkeypatch):
    monkeypatch.setattr(demo, "_live_detector", lambda: (HeuristicDetector(), "heuristic"))
    r = demo._render("read-1", "live")  # a synthetic clip in git, with seconds of stutters
    tl = r["timeline"]
    assert tl["cuts"] and len(r["clean"]) < len(r["original"]) - SR
    assert len(r["heard"]) >= len(r["original"])  # a call cannot hear it sooner than it was said
    assert tl["clean_s"] == round(len(r["clean"]) / SR, 2) and tl["heard_s"] == round(len(r["heard"]) / SR, 2)
    assert len(tl["env_heard"]) == len(r["heard"]) // 160


def test_the_calls_version_is_served_for_live_only(monkeypatch):
    x = np.zeros(SR, np.float32)
    monkeypatch.setattr(demo, "available", lambda: [{"id": "clip", "engines": ["natural", "live"]}])
    monkeypatch.setattr(demo, "render", lambda sid, engine: {"timeline": {}, "original": x, "clean": x,
                                                             **({"heard": x} if engine == "live" else {})})
    assert client.get("/api/demo/clip/heard.wav?engine=live").status_code == 200
    assert client.get("/api/demo/clip/heard.wav?engine=natural").status_code == 404
    assert client.get("/api/demo/clip/other.wav?engine=live").status_code == 404


def test_listen_back_closes_up_the_cuts_after_the_oldest_audio_was_dropped():
    x = (np.random.default_rng(0).standard_normal(6 * SR) * 0.1).astype(np.float32)
    raw = live.Recorder(max_s=4)  # keeps the last 4 s, as a long session keeps its last 10 minutes
    for i in range(0, len(x), 1600):
        raw.add(x[i:i + 1600])
    heard = live.Recorder()
    heard.add(x)
    stats = SimpleNamespace(removed_spans=[(SR // 2, SR, "word_rep"), (3 * SR, 4 * SR, "word_rep")])
    session = SimpleNamespace(raw=raw, out=heard, pipe=SimpleNamespace(splicer=SimpleNamespace(write_pos=len(x), stats=stats)))
    rec = live.LiveSession.recording(session)
    assert len(rec["original"]) == 4 * SR and len(rec["heard"]) == 6 * SR
    # the first cut was in the dropped audio; the second, a second long, is closed up
    assert abs(len(rec["clean"]) - 3 * SR) < 0.2 * SR


def test_listen_back_serves_the_calls_version(monkeypatch):
    x = np.zeros(SR, np.float32)
    monkeypatch.setitem(server.state, "last", {"original": x, "clean": x[: SR // 2], "heard": x})
    monkeypatch.setitem(server.state, "last_at", time.time())
    monkeypatch.setattr(server, "HOSTED", False)
    assert client.get("/api/playback").json() == {"available": True, "original_s": 1.0, "clean_s": 0.5, "heard_s": 1.0}
    assert client.get("/api/playback/heard.wav").status_code == 200


def test_a_recorded_live_run_replays_exactly():
    import json

    from stutterclear.audio import load_wav
    from stutterclear.pipeline import stream
    x = load_wav(Path(demo.CLIPS) / "read-1.wav")
    rec = json.loads(json.dumps(demo.record_live(x, HeuristicDetector(), "heuristic")))  # as stored
    a, ya = stream(x, "trim", HeuristicDetector())
    b, yb = stream(x, "trim", demo.ReplayDetector(rec))
    assert a.splicer.stats.removed_spans and a.splicer.stats.removed_spans == b.splicer.stats.removed_spans
    assert np.array_equal(ya, yb) and len(b.detector.events) == len(a.detector.events)


def test_the_demo_plays_the_recorded_gpu_decisions_when_it_has_them(monkeypatch):
    rec = demo.record_live(demo._excerpt("read-1"), HeuristicDetector(), "heuristic")
    monkeypatch.setattr(demo, "_recorded_live", lambda s: {**rec, "detector": "teacher"})
    monkeypatch.setattr(demo, "_live_detector", lambda: (_ for _ in ()).throw(AssertionError("woke a detector")))
    tl = demo._render("read-1", "live")["timeline"]
    assert tl["label"] == "Live · Best (GPU), Gentle" and tl["cuts"]
