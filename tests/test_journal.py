"""Fluency journal: opt-in, local only, and the numbers it shows."""
import json
from datetime import date, datetime

from fastapi.testclient import TestClient

from stutterclear import journal
from stutterclear.app import server
from stutterclear.silver import label_tokens

client = TestClient(server.app)


def at(day: str, seconds: float, kinds: dict, words=()):
    return {"t": datetime.fromisoformat(f"{day}T12:00:00").timestamp(), "seconds": seconds, "kinds": kinds,
            "words": list(words), "source": "note"}


def test_entries_from_a_note_and_a_session():
    toks = label_tokens([{"w": w, "s": i, "e": i + 0.5} for i, w in enumerate("I I b- [UM] want the the tea".split())])
    e = journal.from_note(toks, 30.0)
    assert e["kinds"] == {"word_rep": 2, "sound_rep": 1, "interjection": 1} and e["words"] == ["i", "b", "the"]
    s = journal.from_session({"cuts": [{"kind": "word_rep"}, {"kind": "sound_rep"}], "events": [{"kind": "block"}, {"kind": "pause"}]}, 60)
    assert s["kinds"] == {"word_rep": 1, "sound_rep": 1, "block": 1} and s["source"] == "live"


def test_summary_rates_by_day_and_week():
    entries = [at("2026-10-19", 120, {"word_rep": 4}, ["the", "the", "think"]),
               at("2026-10-18", 60, {"sound_rep": 3}, ["b"]),
               at("2026-10-10", 60, {"word_rep": 6}, ["the"])]
    s = journal.summary(entries, days=3, today=date(2026, 10, 19))
    assert [d["date"] for d in s["days"]] == ["2026-10-17", "2026-10-18", "2026-10-19"]
    assert [d["per_min"] for d in s["days"]] == [None, 3.0, 2.0]
    assert s["this_week"] == round(7 / 3, 2) and s["last_week"] == 6.0  # fewer stutters a minute this week
    assert s["words"][0] == ["the", 3] or s["words"][0] == ("the", 3)
    assert dict(s["onsets"])["th"] == 4 and s["kinds"] == {"sound_rep": 3, "word_rep": 10}


def test_opt_in_and_local_only(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "LOG", tmp_path / "journal.jsonl")
    monkeypatch.delenv("STUTTERCLEAR_HOSTED", raising=False)
    entry = {"source": "note", "seconds": 20.0, "kinds": {"word_rep": 1}, "words": ["i"]}
    assert not journal.record(entry)  # not started: nothing is written
    journal.start()
    assert journal.record(entry) and not journal.record({**entry, "seconds": 1.0})  # too short to count
    assert len(journal.read()) == 1
    monkeypatch.setenv("STUTTERCLEAR_HOSTED", "1")
    assert not journal.enabled() and not journal.record(entry)  # a public server keeps none
    monkeypatch.delenv("STUTTERCLEAR_HOSTED")
    journal.forget()
    assert not journal.enabled() and journal.read() == []


def test_journal_endpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "LOG", tmp_path / "journal.jsonl")
    monkeypatch.delenv("STUTTERCLEAR_HOSTED", raising=False)
    assert client.get("/api/journal").json()["enabled"] is False
    assert client.post("/api/journal/start").json()["enabled"] is True
    (tmp_path / "journal.jsonl").write_text(json.dumps(at(date.today().isoformat(), 60, {"word_rep": 2}, ["so"])) + "\n")
    s = client.get("/api/journal").json()
    assert s["entries"] == 1 and s["days"][-1]["per_min"] == 2.0
    assert client.delete("/api/journal").json()["ok"] and client.get("/api/journal").json()["enabled"] is False
    monkeypatch.setattr(server, "HOSTED", True)
    assert client.post("/api/journal/start").status_code == 403
