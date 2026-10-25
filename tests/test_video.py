"""Clean my video: what is kept, the ffmpeg graph, and a real cut on a tiny generated video."""
import time

import pytest
from fastapi.testclient import TestClient

from stutterclear.app import server, video


def test_keep_segments_skip_cuts_and_slivers():
    cuts = [{"start": 2.0, "end": 2.5}, {"start": 0.5, "end": 1.0}, {"start": 1.02, "end": 1.4}, {"start": 2.4, "end": 3.0}]
    assert video.keep_segments(cuts, 4.0) == [(0.0, 0.5), (1.4, 2.0), (3.0, 4.0)]  # 1.0-1.02 is a sliver
    assert video.keep_segments([], 3.0) == [(0.0, 3.0)]
    assert video.keep_segments([{"start": 0.0, "end": 3.0}], 3.0) == []


def test_filter_graph_trims_fades_and_joins():
    g = video.filter_graph([(0.0, 0.5), (1.4, 2.0)])
    assert "[0:v]trim=start=0.0:end=0.5,setpts=PTS-STARTPTS[v0]" in g
    assert "[0:a]atrim=start=1.4:end=2.0,asetpts=PTS-STARTPTS,afade=t=in:d=0.01,afade=t=out:st=0.59:d=0.01[a1]" in g
    assert g.endswith("[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]")


def test_a_video_is_cut_where_the_sound_was(tmp_path, monkeypatch):
    if not video.ffmpeg():
        pytest.skip("needs ffmpeg")
    src = tmp_path / "in.mp4"  # 3 s of picture and a tone
    video._run(["-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=3", "-f", "lavfi", "-i",
                "sine=frequency=440:duration=3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(src)])
    monkeypatch.setattr(video.notes, "available", lambda: True)
    monkeypatch.setattr(video.notes, "submit_audio", lambda x, client="": "note-v")
    monkeypatch.setattr(video.notes, "status", lambda nid: {"status": "done", "slow": False})
    monkeypatch.setattr(video.demo, "render", lambda nid, engine: {"timeline": {"cuts": [{"start": 1.0, "end": 1.6, "kind": "word_rep"}],
                                                                               "text": None}})
    client = TestClient(server.app)
    jid = client.post("/api/video?engine=natural&name=in.mp4", content=src.read_bytes()).json()["id"]
    for _ in range(300):
        s = client.get(f"/api/video/{jid}").json()
        if s["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert s["status"] == "done", s["error"]
    assert s["clean_s"] == pytest.approx(2.4, abs=0.05) and s["cuts"][0]["kind"] == "word_rep"
    out = tmp_path / "out.mp4"
    out.write_bytes(client.get(f"/api/video/{jid}/clean.mp4").content)
    import imageio_ffmpeg
    frames, secs = imageio_ffmpeg.count_frames_and_secs(str(out))
    assert secs == pytest.approx(2.4, abs=0.15) and frames == pytest.approx(24, abs=2)
    assert "attachment" in client.get(f"/api/video/{jid}/clean.mp4?download=1").headers["content-disposition"]
    assert client.get("/api/video/video-nope").status_code == 404
