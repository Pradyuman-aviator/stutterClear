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
