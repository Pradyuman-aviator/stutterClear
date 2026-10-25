"""Demo render cache: samples stay rendered; a failed Nemotron check is retried later."""
from stutterclear.app import demo


def test_samples_stay_cached_and_a_failed_guard_is_retried(monkeypatch):
    calls = []

    def fake(sid, engine):
        calls.append(sid)
        return {"timeline": {"guard": {"error": "timeout"} if sid == "down" else None}}

    monkeypatch.setattr(demo, "_render", fake)
    monkeypatch.setattr(demo, "_samples", {})
    demo.render("ok", "natural")
    demo.render("ok", "natural")
    assert calls == ["ok"]  # rendered once, then kept

    clock = [1000.0]
    monkeypatch.setattr(demo.time, "time", lambda: clock[0])
    demo.render("down", "natural")
    demo.render("down", "natural")
    assert calls.count("down") == 1  # not hammering Nemotron on every click
    clock[0] += demo.GUARD_RETRY_S + 1
    demo.render("down", "natural")
    assert calls.count("down") == 2  # asked again later


CUTS = [{"kind": "word_rep", "start": 0.5, "end": 0.92, "word": "I I"},
        {"kind": "interjection", "start": 3661.25, "end": 3661.6}]  # past an hour; a live cut has no words


def test_timestamps_as_csv_and_srt():
    assert demo.cuts_csv(CUTS).splitlines() == ["start_s,end_s,kind,words", "0.500,0.920,word_rep,I I",
                                                "3661.250,3661.600,interjection,"]
    assert demo.cuts_srt(CUTS) == ("1\n00:00:00,500 --> 00:00:00,920\n[removed: repeated word] I I\n\n"
                                   "2\n01:01:01,250 --> 01:01:01,600\n[removed: filler]\n")


def test_timestamps_endpoint(monkeypatch):
    from fastapi.testclient import TestClient
    from stutterclear.app import server
    monkeypatch.setattr(server.notes, "ready", lambda sid: {} if sid == "note-abc" else None)
    monkeypatch.setattr(demo, "render", lambda sid, engine: {"timeline": {"cuts": CUTS}})
    client = TestClient(server.app)
    r = client.get("/api/demo/note-abc/cuts.csv?engine=thorough")
    assert r.status_code == 200 and r.text.startswith("start_s,end_s")
    assert 'filename="note-abc-thorough-cuts.csv"' in r.headers["content-disposition"]
    assert "-->" in client.get("/api/demo/note-abc/cuts.srt").text
    assert client.get("/api/demo/note-abc/cuts.txt").status_code == 404
    assert client.get("/api/demo/note-gone/cuts.csv").status_code == 404


def test_public_servers_leave_uclass_clips_out(tmp_path, monkeypatch):
    (tmp_path / "u.wav").write_bytes(b"")
    (tmp_path / "s.wav").write_bytes(b"")
    monkeypatch.setattr(demo, "DATA", tmp_path)
    monkeypatch.setattr(demo, "CLIPS", tmp_path)
    monkeypatch.setattr(demo, "SAMPLES", {
        "u": {"id": "u", "title": "U", "who": "x", "file": "u.wav", "start": 0, "end": 1, "source": "uclass"},
        "s": {"id": "s", "title": "S", "who": "y", "file": "s.wav", "start": 0, "end": 1, "source": "synthetic"}})
    ids = lambda: [s["id"] for s in demo.available()]  # noqa: E731
    monkeypatch.delenv("STUTTERCLEAR_UCLASS_DEMOS", raising=False)
    monkeypatch.delenv("STUTTERCLEAR_HOSTED", raising=False)
    assert ids() == ["u"]  # the real recording replaces the synthetic stand-in
    monkeypatch.setenv("STUTTERCLEAR_HOSTED", "1")
    assert ids() == ["s"]
    assert "LibriSpeech" in demo.credit(demo.available()) and "UCLASS" not in demo.credit(demo.available())
    monkeypatch.setenv("STUTTERCLEAR_UCLASS_DEMOS", "1")  # the Cloud Run demo
    assert ids() == ["u"]
    assert "UCLASS" in demo.credit(demo.available()) and "LibriSpeech" not in demo.credit(demo.available())
    (tmp_path / "u.wav").unlink()  # an image without the recording still has something to play
    assert ids() == ["s"]


def test_transcript_marks_removed_words_and_cleans_the_text():
    from stutterclear.silver import label_tokens, word_cuts
    said = ["So", "I", "I", "w-", "[UM]", "want", "to", "go.", "it", "is", "fine"]
    toks = label_tokens([{"w": w, "s": 0.5 * i, "e": 0.5 * i + 0.4} for i, w in enumerate(said)])
    t = demo.transcript(toks, word_cuts(toks))  # Natural: the repeat and the broken-off start go
    assert [w["w"] for w in t["words"] if w["removed"]] == ["I", "w-"]
    assert t["words"][4] == {"w": "um", "removed": False, "label": "filler"}
    assert t["said"] == "So I I w- um want to go. It is fine"
    assert t["clean"] == "So I want to go. It is fine"  # text never keeps a filler
