"""Voice notes, Dictate and videos on a CPU server, through a GPU worker elsewhere: the notes path with
a stand-in broker (the broker itself is tests/test_gpu_broker.py), uploads in parts, a cleaned video's
ranges, ffmpeg's limits, and what protects a public server (same-origin requests, the egress budget)."""
import io
import os
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path

import anyio
import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.io import wavfile
from starlette.websockets import WebSocketDisconnect

from stutterclear import captions
from stutterclear.app import demo, gpu, notes, server, video

WORDS = [{"w": "I", "s": 0.10, "e": 0.25}, {"w": "I", "s": 0.40, "e": 0.55}, {"w": "want", "s": 0.70, "e": 1.00},
         {"w": "to", "s": 1.05, "e": 1.20}, {"w": "go.", "s": 1.30, "e": 1.70}]


class StandIn:
    """A GPU broker that is set up and answers at once (or when told to): the remote path without a
    worker. accepting/busy/error: what it says instead."""

    def __init__(self):
        self.refuse = self.busy = self.error = None
        self.go = threading.Event()
        self.go.set()
        self.admitted = []

    def configured(self):
        return True

    def live_offered(self):
        return False

    def health(self):
        return "ready"

    def info(self):
        return {"where": "Kaggle", "typical_s": [120, 240], "idle_s": 600, "live": False}

    def accepting(self, client=""):
        return self.refuse

    def admit(self, client, seconds, dry_run=False):
        if not dry_run and not self.busy:
            self.admitted.append((client, round(seconds, 1)))
        return self.busy

    def analyse(self, x, on_update, abandoned=lambda: False, client=""):
        on_update("queued", 1)
        self.go.wait(5)
        on_update("listening", None)
        on_update("transcribing", None)
        if self.error:
            raise gpu.GpuError(self.error)
        probs = np.zeros(((len(x) - 400) // 320 + 1, 5), np.float32)
        probs[5:12, 1] = 0.95  # the first "I": a repeated word
        return probs, [dict(w) for w in WORDS], False


@pytest.fixture
def remote(monkeypatch, tmp_path):
    """A CPU server: the recordings teacher is not here, and a GPU worker is set up."""
    monkeypatch.setattr(notes, "TEACHER", tmp_path / "no-teacher.pt")
    monkeypatch.setattr(notes, "_notes", {})
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.delenv("STUTTERCLEAR_NOTES", raising=False)
    b = StandIn()
    monkeypatch.setattr(gpu, "broker", b)
    return b


def wav(seconds: float = 2.0) -> bytes:
    buf = io.BytesIO()
    wavfile.write(buf, 16000, (np.random.default_rng(0).standard_normal(int(16000 * seconds)) * 3000).astype(np.int16))
    return buf.getvalue()


def until(fn, timeout: float = 10.0):
    end = time.monotonic() + timeout
    while not (v := fn()):
        assert time.monotonic() < end, "timed out"
        time.sleep(0.01)
    return v


def finished(c: TestClient, nid: str) -> dict:
    return until(lambda: (s := c.get(f"/api/notes/{nid}").json())["status"] in ("done", "error") and s)


def tiny_video(path: Path, seconds: int = 3) -> Path:
    video._run(["-f", "lavfi", "-i", f"testsrc=size=160x120:rate=10:duration={seconds}", "-f", "lavfi", "-i",
                f"sine=frequency=440:duration={seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                "-shortest", str(path)])
    return path


# ---- voice notes ----------------------------------------------------------------------------------

def test_a_voice_note_is_cleaned_through_the_gpu_worker(remote):
    remote.go.clear()
    c = TestClient(server.app)
    assert notes.remote() and notes.available()
    nid = c.post("/api/notes?page=" + "a" * 32, content=wav()).json()["id"]
    until(lambda: c.get(f"/api/notes/{nid}").json()["ahead"] == 1)  # waiting its turn on the GPU
    assert c.get(f"/api/notes/{nid}").json()["status"] == "queued"
    remote.go.set()
    s = finished(c, nid)
    assert (s["status"], s["ahead"], s["error"], s["duration_s"]) == ("done", None, None, 2.0)
    assert remote.admitted == [("testclient", 2.0)]
    note = notes.ready(nid)
    assert [t["label"] for t in note["toks"]][:2] == ["repeat", "fluent"]
    tl = c.get(f"/api/demo/{nid}?engine=natural&captions=0").json()
    assert tl["captions"] is None and [x["word"] for x in tl["cuts"]] == ["I"]  # cut on the CPU, here
    r = c.get(f"/api/demo/{nid}/clean.wav?engine=natural")
    assert r.status_code == 200 and r.headers["cache-control"] == "private, no-store"
    assert "natural" in note["renders"]
    note["created"] -= notes.KEEP_S + 1
    notes.expire()  # forgotten with everything made from it
    assert notes.status(nid) is None and c.get(f"/api/demo/{nid}?engine=natural").status_code == 404


def test_the_remote_path_does_not_wait_for_the_warm_up(remote):
    notes.warm.clear()  # the live teacher's warm-up has nothing to do with a GPU elsewhere
    try:
        nid = notes.submit(wav(), "203.0.113.5")
        until(lambda: notes.status(nid)["status"] == "done", timeout=3)
    finally:
        notes.warm.set()


def test_the_gpu_workers_refusals_reach_the_page(remote):
    c = TestClient(server.app)
    remote.refuse = "This week's free GPU hours are used up, so the GPU rests until Mon 18 Jan, 08:00 UTC."
    r = c.post("/api/notes", content=wav())
    assert r.status_code == 503 and r.json()["detail"] == remote.refuse
    if video.available():
        assert c.post("/api/video/upload").status_code == 503  # before anything is sent
    remote.refuse, remote.busy = None, gpu.CLIENT_BUSY
    r = c.post("/api/notes", content=wav())
    assert r.status_code == 429 and r.json()["detail"] == gpu.CLIENT_BUSY
    if video.available():
        assert c.post("/api/video/upload").status_code == 429
    remote.busy, remote.error = None, gpu.STOPPED_TWICE
    s = finished(c, c.post("/api/notes", content=wav()).json()["id"])
    assert s["status"] == "error" and s["error"] == gpu.STOPPED_TWICE  # the broker's own sentence, as it is


def test_with_the_teacher_here_notes_stay_on_this_computer(remote, monkeypatch, tmp_path):
    (tmp_path / "teacher.pt").write_bytes(b"")
    monkeypatch.setattr(notes, "TEACHER", tmp_path / "teacher.pt")
    assert not notes.remote()
    seen = []
    monkeypatch.setattr(notes, "teacher_probs", lambda x: seen.append("teacher") or np.zeros((99, 5), np.float32))
    monkeypatch.setattr(notes, "_transcribe", lambda note: seen.append("asr") or type("R", (), {"words": []})())
    nid = notes.submit(wav())
    until(lambda: notes.status(nid)["status"] == "done")
    assert seen == ["teacher", "asr"] and remote.admitted == []


def test_info_lists_what_the_gpu_worker_does(remote):
    g = TestClient(server.app).get("/api/info").json()["gpu"]
    assert g["features"][:2] == ["notes", "dictate"] and ("video" in g["features"]) == video.available()
    assert "live" not in g["features"]


def test_note_renders_are_bounded(remote, monkeypatch):
    monkeypatch.setattr(demo, "NOTE_RENDERS", 2)
    monkeypatch.setattr(demo, "_note_renders", demo.deque())
    c = TestClient(server.app)
    ids = [c.post("/api/notes", content=wav()).json()["id"] for _ in range(2)]
    for nid in ids:
        finished(c, nid)
    c.get(f"/api/demo/{ids[0]}?engine=natural&captions=0")
    c.get(f"/api/demo/{ids[1]}?engine=natural&captions=0")
    c.get(f"/api/demo/{ids[1]}?engine=thorough&captions=0")  # the oldest render is let go
    assert list(notes._notes[ids[0]]["renders"]) == [] and set(notes._notes[ids[1]]["renders"]) == {"natural", "thorough"}
    assert c.get(f"/api/demo/{ids[0]}?engine=natural&captions=0").status_code == 200  # made again when asked


def test_captions_are_made_once_and_kept_with_their_render(remote, monkeypatch):
    calls = []
    monkeypatch.setattr(captions, "available", lambda: True)
    monkeypatch.setattr(captions, "caption_audio", lambda x: calls.append(len(x)) or [{"w": "i", "t": 0.1}])
    c = TestClient(server.app)
    nid = c.post("/api/notes", content=wav()).json()["id"]
    finished(c, nid)
    first = c.get(f"/api/demo/{nid}?engine=natural").json()["captions"]
    assert first == {"original": [{"w": "i", "t": 0.1}], "clean": [{"w": "i", "t": 0.1}]} and len(calls) == 2
    assert c.get(f"/api/demo/{nid}?engine=natural").json()["captions"] == first and len(calls) == 2
    c.get(f"/api/demo/{nid}?engine=thorough")
    assert len(calls) == 3  # the original's captions are the same for every engine
    assert c.get(f"/api/demo/{nid}?engine=thorough&captions=0").json()["captions"] is None  # Dictate
    assert ("original", nid) in notes._notes[nid]["renders"]  # so they go with the note


def test_captions_take_one_cpu_thread():
    import inspect
    assert inspect.signature(captions.recognizer).parameters["threads"].default == 1  # same text, a tenth of a core


def test_the_recordings_calibration_can_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("STUTTERCLEAR_CALIBRATION_JSON", '{"confirm": 0.42, "gentle": {}}')
    assert demo._env_calibration()["confirm"] == 0.42
    for bad in ("not json", "[1, 2]", ""):
        monkeypatch.setenv("STUTTERCLEAR_CALIBRATION_JSON", bad)
        assert demo._env_calibration() == {}


# ---- videos -----------------------------------------------------------------------------------------

def test_upload_rules(monkeypatch):
    monkeypatch.setattr(video, "PART_BYTES", 10)
    monkeypatch.setattr(video, "MAX_BYTES", 25)
    monkeypatch.setattr(video, "_uploads", {})
    up = video.begin_upload()["upload"]
    assert video.add_part(up, 0, b"x" * 10) == 10
    for i, part, code in ((2, b"y", 409), (1, b"x" * 11, 400), (1, b"", 400)):
        with pytest.raises(video.UploadError) as e:
            video.add_part(up, i, part)
        assert e.value.code == code
    assert video.add_part(up, 0, b"x" * 10) == 10  # sent again (its answer was lost): taken once
    assert video.add_part(up, 1, b"x" * 10) == 20
    with pytest.raises(video.UploadError) as e:
        video.add_part(up, 2, b"x" * 6)
    assert e.value.code == 413
    other = video.begin_upload()["upload"]
    with pytest.raises(notes.NoteBusy):  # two uploads at a time
        video.begin_upload()
    folder = video._uploads[other]["dir"]
    video._uploads[other]["at"] -= video.UPLOAD_IDLE_S + 1  # nothing came for ten minutes
    video.expire()
    assert not folder.exists()
    with pytest.raises(video.UploadError) as e:
        video.add_part(other, 0, b"x")
    assert e.value.code == 404
    with pytest.raises(video.UploadError) as e:
        video.submit_upload("up-nope")
    assert e.value.code == 404
    shutil.rmtree(video._uploads.pop(up)["dir"], ignore_errors=True)


def test_one_upload_per_visitor_and_none_held_for_ever(monkeypatch):
    monkeypatch.setattr(video, "_uploads", {})
    first = video.begin_upload("203.0.113.5")["upload"]
    folder = video._uploads[first]["dir"]
    again = video.begin_upload("203.0.113.5")["upload"]  # the same visitor again (a retry): it replaces the first
    assert set(video._uploads) == {again} and not folder.exists()
    other = video.begin_upload("203.0.113.6")["upload"]
    with pytest.raises(notes.NoteBusy):
        video.begin_upload("203.0.113.7")
    video.add_part(other, 0, b"x" * 10)
    video._uploads[other]["at"] -= video.UPLOAD_IDLE_S - 5
    video.add_part(other, 0, b"x" * 10)  # the last part again: taken once, and it buys no more time
    video.expire()
    assert other in video._uploads
    video._uploads[other]["at"] -= 10
    video.expire()
    assert other not in video._uploads
    video._uploads[again]["began"] -= video.UPLOAD_MAX_S + 1  # still sending, but begun half an hour ago
    video._uploads[again]["at"] = time.time()
    video.expire()
    assert video._uploads == {}


def test_an_upload_the_page_gives_up_on_frees_its_place(monkeypatch):
    if not video.available():
        pytest.skip("needs the video feature")
    monkeypatch.setattr(video, "_uploads", {})
    c = TestClient(server.app)
    up = c.post("/api/video/upload").json()["upload"]
    assert c.delete(f"/api/video/upload/{up}").json() == {"ok": True} and video._uploads == {}
    read = []
    real = server._body
    monkeypatch.setattr(server, "_body", lambda *a: read.append(a) or real(*a))
    r = c.put(f"/api/video/upload/{up}?i=0", content=b"x" * 1000)
    assert r.status_code == 404 and not read  # turned away before the part was read
    up = c.post("/api/video/upload").json()["upload"]
    assert c.put(f"/api/video/upload/{up}?i=3", content=b"x").status_code == 409 and not read
    c.delete(f"/api/video/upload/{up}")


def test_a_long_videos_sound_is_never_read_whole(tmp_path):
    if not video.ffmpeg():
        pytest.skip("needs ffmpeg")
    src = tmp_path / "long.mkv"  # 200 s of sound in a few kilobytes
    video._run(["-f", "lavfi", "-i", "testsrc=size=32x32:rate=1:duration=200", "-f", "lavfi", "-i",
                "anullsrc=r=8000:cl=mono", "-t", "200", "-c:v", "libx264", "-c:a", "aac", "-b:a", "8k", str(src)])
    with pytest.raises(video.VideoError, match="up to 3 minutes"):
        video.soundtrack(src)
    with pytest.raises(video.VideoError, match="more than expected"):  # and nothing ffmpeg writes is held beyond a cap
        video._run(["-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "5", "-f", "f32le", "-"], max_out=1000)


def test_a_video_past_its_hour_is_gone_even_if_nothing_else_happens(monkeypatch):
    old = {"status": "done", "created": time.time() - video.KEEP_S - 1, "error": None, "engine": "natural",
           "out": b"x", "cuts": [], "text": None}
    monkeypatch.setattr(video, "_jobs", {"video-old": old})
    assert video.result("video-old") is None and video.status("video-old") is None and video._jobs == {}


def test_only_video_extensions_are_kept():
    assert [video._suffix(n) for n in ("talk.MOV", ".webm", "mkv", "a.b.avi", "clip.m4v")] == \
        [".mov", ".webm", ".mkv", ".avi", ".m4v"]
    assert {video._suffix(n) for n in ("list.m3u8", "", "x.txt", "../../etc/passwd")} == {".mp4"}


def test_ffmpeg_reads_only_video_files(tmp_path):
    if not video.ffmpeg():
        pytest.skip("needs ffmpeg")
    src = tiny_video(tmp_path / "in.mp4", 1)
    playlist = tmp_path / "playlist.mp4"  # a playlist named like a video would make ffmpeg read other files
    playlist.write_text(f"#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\n{src.as_posix()}\n#EXT-X-ENDLIST\n")
    with pytest.raises(video.VideoError, match="MP4, MOV, WebM, MKV and AVI files work"):
        video.soundtrack(playlist)
    assert len(video.soundtrack(src)) > 15000


def test_a_video_uploaded_in_parts_is_cleaned_through_the_gpu_worker(remote, monkeypatch, tmp_path):
    if not video.ffmpeg():
        pytest.skip("needs ffmpeg")
    data = tiny_video(tmp_path / "in.mp4").read_bytes()
    monkeypatch.setattr(video, "PART_BYTES", 8192)
    monkeypatch.setattr(video, "_jobs", {})
    monkeypatch.setattr(video, "_uploads", {})
    c = TestClient(server.app)
    up = c.post("/api/video/upload").json()
    assert up["part_bytes"] == 8192 and up["max_bytes"] == video.MAX_BYTES
    parts = [data[i:i + 8192] for i in range(0, len(data), 8192)]
    for i, part in enumerate(parts):
        assert c.put(f"/api/video/upload/{up['upload']}?i={i}", content=part).status_code == 200
    last = c.put(f"/api/video/upload/{up['upload']}?i={len(parts) - 1}", content=parts[-1])
    assert last.json() == {"received": len(data)}  # a resent last part is taken once
    assert c.put(f"/api/video/upload/{up['upload']}?i={len(parts) + 3}", content=b"x").status_code == 409
    assert c.put(f"/api/video/upload/{up['upload']}?i={len(parts)}", content=b"x" * 9000).status_code == 413
    folder = video._uploads[up["upload"]]["dir"]
    jid = c.post(f"/api/video?upload={up['upload']}&engine=natural&ext=.MP4").json()["id"]
    s = until(lambda: (s := c.get(f"/api/video/{jid}").json())["status"] in ("done", "error") and s, timeout=60)
    assert s["status"] == "done", s["error"]
    assert [x["word"] for x in s["cuts"]] == ["I"]  # cut where the worker's probabilities said, on the CPU here
    assert s["clean_s"] == pytest.approx(3 - sum(x["end"] - x["start"] for x in s["cuts"]), abs=0.05)
    assert remote.admitted == [("testclient", 3.0)]
    out = c.get(f"/api/video/{jid}/clean.mp4")
    assert out.headers["accept-ranges"] == "bytes" and out.content[4:8] == b"ftyp"
    r = c.get(f"/api/video/{jid}/clean.mp4", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and r.content == out.content[:100]
    assert r.headers["content-range"] == f"bytes 0-99/{len(out.content)}"
    assert not folder.exists()  # nothing of it is left on disk


def test_a_videos_status_says_where_its_sound_is(remote, monkeypatch):
    now = time.time()
    monkeypatch.setattr(video, "_jobs", {"video-x": {"status": "cleaning", "note": "note-x", "created": now,
                                                     "error": None, "engine": "natural", "out": None, "client": ""}})
    notes._notes["note-x"] = {"status": "queued", "ahead": 2, "x": np.zeros(16000, np.float32), "created": now,
                              "error": None, "seen": 0.0}
    s = video.status("video-x")
    assert (s["note"], s["ahead"]) == ("queued", 2)
    assert notes._notes["note-x"]["seen"] > 0  # the page still waits for it: never dropped as abandoned


def test_a_cleaned_video_is_served_in_ranges_and_streamed(monkeypatch):
    big = bytes(range(256)) * (20 * 2**20 // 256 + 7)  # just over 20 MiB
    job = {"status": "done", "created": time.time(), "error": None, "engine": "natural", "client": ""}
    monkeypatch.setattr(video, "_jobs", {"video-big": {**job, "out": big}, "video-small": {**job, "out": b"x" * 1000}})
    c = TestClient(server.app)
    url = "/api/video/video-big/clean.mp4"
    r = c.get(url, headers={"Range": "bytes=0-"})  # what Safari asks for first
    assert r.status_code == 206 and len(r.content) == 16 * 2**20 and r.content == big[:16 * 2**20]
    assert r.headers["content-range"] == f"bytes 0-{16 * 2**20 - 1}/{len(big)}"
    assert c.get(url, headers={"Range": "bytes=-100"}).content == big[-100:]
    assert c.get(url, headers={"Range": "bytes=100-199"}).content == big[100:200]
    r = c.get(url, headers={"Range": f"bytes={len(big)}-"})
    assert r.status_code == 416 and r.headers["content-range"] == f"bytes */{len(big)}"
    r = c.get(url)  # the whole file, streamed in chunks (no 32 MiB limit then)
    assert r.status_code == 200 and r.content == big and "content-length" not in r.headers
    r = c.get("/api/video/video-small/clean.mp4?download=1")
    assert r.content == b"x" * 1000 and r.headers["content-length"] == "1000" and "attachment" in r.headers["content-disposition"]


def test_old_video_folders_are_swept_at_startup():
    old = Path(tempfile.mkdtemp(prefix="stutterclear-video-"))
    new = Path(tempfile.mkdtemp(prefix="stutterclear-video-"))
    os.utime(old, (time.time() - 7200, time.time() - 7200))
    video.sweep()
    assert not old.exists() and new.exists()
    new.rmdir()


# ---- what protects a public server --------------------------------------------------------------

def test_other_websites_cannot_post_or_open_sockets(remote):
    c = TestClient(server.app)
    evil = {"Origin": "https://evil.example"}
    r = c.post("/api/notes", content=wav(), headers=evil)
    assert r.status_code == 403 and r.json()["detail"] == server.FOREIGN
    assert c.post("/api/notes", content=wav(), headers={"Origin": "null"}).status_code == 403
    assert c.post("/api/notes", content=wav(), headers={"Origin": "http://testserver"}).status_code == 200
    assert c.get("/api/info", headers=evil).status_code == 200  # reading is fine
    for path in ("/ws/mic", "/ws"):
        with pytest.raises(WebSocketDisconnect):
            c.websocket_connect(path, headers=evil).__enter__()


def test_a_public_server_caps_what_one_visitor_downloads(monkeypatch):
    monkeypatch.setattr(server, "HOSTED", True)
    monkeypatch.setattr(server, "TRUST_PROXY", 1)
    monkeypatch.setattr(server, "EGRESS_CLIENT", 50_000)
    monkeypatch.setattr(server, "_egress", {})
    monkeypatch.setattr(server, "_egress_day", [0, 0])
    page = "a" * 32
    for k, v in {"last": {"original": np.zeros(16000, np.float32), "clean": np.zeros(16000, np.float32)},
                 "last_owner": page, "last_at": time.time()}.items():
        monkeypatch.setitem(server.state, k, v)
    c = TestClient(server.app)
    them = {"X-Forwarded-For": "spoofed, 198.51.100.7"}  # the proxy's own entry, last, is the visitor
    wav = f"/api/playback/clean.wav?page={page}"
    r = c.get(wav, headers=them)
    assert r.status_code == 200 and "content-encoding" not in r.headers  # (audio is never gzipped)
    assert [c.get(wav, headers=them).status_code for _ in range(2)] == [200, 429]  # 32 KB each
    assert c.get("/api/info", headers=them).status_code == 200  # the page and its small calls always answer
    assert c.get("/", headers=them).status_code == 200
    assert c.get(wav, headers={"X-Forwarded-For": "198.51.100.8"}).status_code == 200  # others are fine
    assert c.get("/api/health").json()["ok"] is True
    monkeypatch.setattr(server, "EGRESS_TOTAL", 60_000)  # everyone's share of the hour is used up
    assert c.get(wav, headers={"X-Forwarded-For": "198.51.100.9"}).status_code == 429
    h = c.get("/api/health")
    assert h.status_code == 200 and h.json()["ok"] is False and h.json()["reason"] == "downloads"  # front door: elsewhere
    monkeypatch.setattr(server, "EGRESS_TOTAL", 10**12)
    monkeypatch.setattr(server, "EGRESS_DAY", 60_000)  # or today's
    assert c.get(wav, headers={"X-Forwarded-For": "198.51.100.10"}).status_code == 429
    assert server.client_ip({"headers": [(b"x-forwarded-for", b"2001:db8::1, 2001:db8:0:0:ffff::2")],
                             "client": ("127.0.0.1", 1)}) == "2001:db8::/64"  # IPv6: per /64
    # no proxy entry: the socket's own address, IPv4 (a real visitor; the test client's is "testclient") or mapped
    assert server.client_ip({"headers": [], "client": ("203.0.113.5", 1234)}) == "203.0.113.5"
    assert server.client_ip({"headers": [], "client": ("::ffff:203.0.113.5", 1234)}) == "203.0.113.5"
    monkeypatch.setattr(server, "TRUST_PROXY", 2)  # behind two proxies (a load balancer in front of Cloud Run)
    assert server.client_ip({"headers": [(b"x-forwarded-for", b"spoofed, 198.51.100.7, 10.0.0.1")],
                             "client": ("127.0.0.1", 1)}) == "198.51.100.7"


def test_the_proxy_log_shows_every_count_of_forwarded_entries_once(monkeypatch, capsys):
    monkeypatch.setattr(server, "TRUST_PROXY", 1)
    monkeypatch.setattr(server, "_proxy_seen", set())
    for header in (b"198.51.100.7", b"198.51.100.8", b"spoofed, 198.51.100.7"):
        server.client_ip({"headers": [(b"x-forwarded-for", header)], "client": ("127.0.0.1", 1)})
    out = capsys.readouterr().out
    assert out.count("proxy: a request with 1 ") == 1 and out.count("proxy: a request with 2 ") == 1
    assert "198.51" not in out  # counts, never addresses


def test_egress_minutes_older_than_an_hour_are_forgotten(monkeypatch):
    monkeypatch.setattr(server, "_egress", {"": {100: 5, 159: 7}, "x": {90: 3}})
    monkeypatch.setattr(server, "_egress_swept", [0])
    server._sweep_egress(160)
    assert server._egress == {"": {159: 7}}


def test_the_page_is_compressed_and_its_assets_kept_by_browsers():
    c = TestClient(server.app)
    page = c.get("/", headers={"Accept-Encoding": "gzip"})
    assert page.status_code == 200 and page.headers["cache-control"] == "no-cache"  # a new deploy shows at once
    script = re.search(r'src="/(assets/[^"]+\.js)"', page.text)[1]
    r = c.get("/" + script, headers={"Accept-Encoding": "gzip"})
    assert r.headers["cache-control"] == "public, max-age=31536000, immutable" and r.headers["content-encoding"] == "gzip"


def test_request_bodies_are_bounded(remote, monkeypatch):
    c = TestClient(server.app)
    r = c.post("/api/insights", content=b"{" + b" " * (server.JSON_MAX + 1) + b"}",
               headers={"Content-Type": "application/json"})
    assert r.status_code == 413  # refused by its length, before anything is read
    assert c.post("/api/feedback/summary", json={"entries": [{"verdict": "accept"}] * 3000}).status_code == 200

    def chunks(n):  # no length up front: counted as it arrives
        for _ in range(n):
            yield b" " * 65536
    assert c.post("/api/insights", content=chunks(8), headers={"Content-Type": "application/json"}).status_code == 413
    assert c.post("/api/notes", content=b"\0" * (notes.MAX_BYTES + 1)).status_code == 413
    monkeypatch.setattr(server, "_big", {"": server.BIG_BODIES})  # as many large uploads arriving as allowed
    r = c.post("/api/notes", content=wav(10.0))  # a large body (its length says so): waits for room
    assert r.status_code == 503
    monkeypatch.setattr(server, "_big", {"testclient": 2})
    assert c.post("/api/notes", content=wav(10.0)).status_code == 429  # two at a time from one visitor
    monkeypatch.setattr(server, "_big", {})
    assert c.post("/api/notes", content=wav(10.0)).status_code == 200 and server._big == {}  # (its place given back)


def receive(ws, timeout: float) -> dict:
    """The next message on a test client's WebSocket (TimeoutError rather than a hang)."""
    async def get():
        with anyio.fail_after(timeout):
            return await ws._send_rx.receive()
    msg = ws.portal.call(get)
    ws._raise_on_close(msg)
    return msg


def test_a_public_server_rests_page_sockets_left_open(monkeypatch):
    monkeypatch.setattr(server, "HOSTED", True)
    monkeypatch.setattr(server, "PAGE_MAX_S", 0.2)
    c = TestClient(server.app)
    with c.websocket_connect("/ws?page=" + "d" * 32) as ws:
        with pytest.raises(WebSocketDisconnect) as e:
            receive(ws, 5)
    assert (e.value.code, e.value.reason) == (4000, "rest")  # an active page reconnects at once
    monkeypatch.setitem(server.state, "session", object())  # this page's own live session is running
    monkeypatch.setitem(server.state, "owner", "d" * 32)
    with c.websocket_connect("/ws?page=" + "d" * 32) as ws:
        with pytest.raises(TimeoutError):
            receive(ws, 0.6)  # never cut off mid-session


def test_idle_pages_get_a_keepalive_not_a_flood(monkeypatch):
    monkeypatch.setattr(server, "warm_demo", server.notes.warm.set)  # no models here
    monkeypatch.setattr(server, "KEEPALIVE_S", 0.4)
    with TestClient(server.app) as c:  # the startup runs the broadcast
        with c.websocket_connect("/ws?page=" + "e" * 32) as ws:
            assert receive(ws, 5)["text"] == '{"idle":true}'  # at once: no session
            n, t0 = 0, time.monotonic()
            with pytest.raises(TimeoutError):
                while True:
                    receive(ws, 1.5 - (time.monotonic() - t0))
                    n += 1
    assert 2 <= n <= 5  # every 0.4 s, not 33 a second


def test_page_and_mic_sockets_per_visitor_are_capped_on_a_public_server(monkeypatch):
    monkeypatch.setattr(server, "HOSTED", True)
    monkeypatch.setattr(server, "_sockets", {})
    c = TestClient(server.app)
    pages = [c.websocket_connect("/ws?page=" + f"{i:032x}") for i in range(server.SOCKETS_PER_CLIENT)]
    for p in pages:
        p.__enter__()
    for path in ("/ws?page=" + "f" * 32, "/ws/mic"):
        with c.websocket_connect(path) as extra, pytest.raises(WebSocketDisconnect) as e:
            receive(extra, 5)
        assert (e.value.code, e.value.reason) == (server.CLOSE_TOO_MANY, server.TOO_MANY_PAGES)  # said, not an outage
    pages[0].__exit__(None, None, None)
    until(lambda: server._sockets.get("testclient", 0) < server.SOCKETS_PER_CLIENT)
    with c.websocket_connect("/ws?page=" + "e" * 32):
        pass
    for p in pages[1:]:
        p.__exit__(None, None, None)
    with pytest.raises(WebSocketDisconnect):  # page ids are what the page makes: 32 hex digits
        c.websocket_connect("/ws?page=not-a-page").__enter__()


def test_one_visitors_live_time_is_rationed_on_a_public_server(monkeypatch):
    monkeypatch.setattr(server, "HOSTED", True)
    monkeypatch.setattr(server, "LIVE_PER_HOUR_S", 0.3)
    monkeypatch.setattr(server, "_live_used", {})
    c = TestClient(server.app)
    with c.websocket_connect("/ws/mic") as ws:
        ws.send_json({"mode": "gentle", "detector": "heuristic"})
        assert ws.receive_json()["ok"]
        time.sleep(0.4)
        ws.send_text("stop")
        ws.receive_json()
    with c.websocket_connect("/ws/mic") as ws, pytest.raises(WebSocketDisconnect) as e:
        receive(ws, 5)  # the one live slot is shared: they had their turn this hour
    assert (e.value.code, e.value.reason) == (4409, server.LIVE_SPENT) and server.state["session"] is None
    monkeypatch.setitem(server._live_used, "testclient", server.deque([(time.monotonic() - 3601, 0.4)]))
    with c.websocket_connect("/ws/mic") as ws:  # an hour later
        ws.send_json({"mode": "gentle", "detector": "heuristic"})
        assert ws.receive_json()["ok"]
        ws.send_text("stop")
        ws.receive_json()


def test_after_a_session_someone_else_waited_for_they_go_first(monkeypatch):
    monkeypatch.setattr(server, "HOSTED", True)
    monkeypatch.setattr(server, "LIVE_PER_HOUR_S", 1200.0)
    monkeypatch.setattr(server, "TRUST_PROXY", 1)
    for name, value in (("_live_used", {}), ("_live_rest", {}), ("_live_turn", {"client": None, "turned_away": 0.0})):
        monkeypatch.setattr(server, name, value)
    c = TestClient(server.app)
    alice, bob = ({"X-Forwarded-For": f"198.51.100.{i}"} for i in (1, 2))
    settings = {"mode": "gentle", "detector": "heuristic"}
    with c.websocket_connect("/ws/mic", headers=alice) as ws:
        ws.send_json(settings)
        assert ws.receive_json()["ok"]
        with c.websocket_connect("/ws/mic", headers=bob) as other, pytest.raises(WebSocketDisconnect) as e:
            receive(other, 5)
        assert e.value.code == 4409 and e.value.reason == server.BUSY_HOSTED  # bob waits...
        ws.send_text("stop")
        ws.receive_json()
    with c.websocket_connect("/ws/mic", headers=alice) as ws, pytest.raises(WebSocketDisconnect) as e:
        receive(ws, 5)
    assert (e.value.code, e.value.reason) == (4409, server.LIVE_YIELD)  # ...so alice does not start again at once
    with c.websocket_connect("/ws/mic", headers=bob) as ws:  # it is bob's turn
        ws.send_json(settings)
        assert ws.receive_json()["ok"]
        ws.send_text("stop")
        ws.receive_json()
