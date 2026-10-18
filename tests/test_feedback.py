"""Human feedback loop: the update rule, its limits, where it applies, and where it is stored."""
import numpy as np
import torch
from fastapi.testclient import TestClient

from stutterclear import feedback
from stutterclear.app import server
from stutterclear.live import make_detector
from stutterclear.silver import label_tokens, word_cuts
from stutterclear.student import CausalStudent

client = TestClient(server.app)


def cut(kind, verdict):
    return {"source": "note", "id": "note-1", "verdict": verdict,
            "cut": {"start": 1.0, "end": 1.4, "kind": kind, "words": "I"}}


def missed(kind):
    return {"source": "note", "id": "note-1", "missed_at_s": 2.0, "kind": kind}


def test_rejected_cuts_raise_and_missed_stutters_lower_within_limits():
    m = feedback.multipliers([cut("word_rep", "reject"), cut("word_rep", "accept"), missed("sound_rep"),
                              missed(None)])
    assert m == {"sound_rep": 0.95, "word_rep": 1.05, "prolongation": 1.0, "interjection": 1.0}
    m = feedback.multipliers([cut("interjection", "reject")] * 40 + [missed("prolongation")] * 40)
    assert m["interjection"] == feedback.HIGH and m["prolongation"] == feedback.LOW


def test_auto_mode_is_offered_only_after_enough_agreement():
    assert not feedback.agreement([cut("word_rep", "accept")] * 49)["auto_ready"]  # too few yet
    assert feedback.agreement([cut("word_rep", "accept")] * 48 + [missed("word_rep")] * 2)["auto_ready"]  # 96%
    assert not feedback.agreement([cut("word_rep", "accept")] * 47 + [cut("word_rep", "reject")] * 3)["auto_ready"]
    old_mistakes = [cut("word_rep", "reject")] * 10
    assert feedback.agreement(old_mistakes + [cut("word_rep", "accept")] * 50)["auto_ready"]  # only the last 50 count


def test_clean_rejects_what_is_not_feedback():
    assert feedback.clean({"verdict": "accept", "cut": {"start": 1, "end": 2, "kind": "block"}}) is None  # never cut
    assert feedback.clean({"verdict": "maybe", "cut": {"start": 1, "end": 2, "kind": "word_rep"}}) is None
    assert feedback.clean({"verdict": "reject", "cut": {"kind": "word_rep"}}) is None
    assert feedback.clean({"missed_at_s": "soon"}) is None
    assert feedback.clean({"missed_at_s": 3, "kind": "nonsense"})["kind"] is None


def test_log_on_this_computer_and_none_on_a_public_server(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "LOG", tmp_path / "feedback.jsonl")
    monkeypatch.delenv("STUTTERCLEAR_HOSTED", raising=False)
    feedback.record(feedback.clean(cut("word_rep", "reject")))
    with (tmp_path / "feedback.jsonl").open("a") as f:
        f.write("not json\n")  # a damaged line is skipped
    assert feedback.current()["word_rep"] == 1.05
    monkeypatch.setenv("STUTTERCLEAR_HOSTED", "1")
    assert feedback.current() == feedback.NEUTRAL  # a public server never applies the owner's log
    feedback.forget()
    assert feedback.read() == []


def test_page_adjustments_are_clamped():
    assert feedback.from_page({"word_rep": 9, "sound_rep": "x", "block": 0.1, "evil": 2}) == \
        {**feedback.NEUTRAL, "word_rep": feedback.HIGH}
    assert feedback.from_page("nope") == feedback.NEUTRAL


def test_thresholds_and_voice_note_check_follow_the_multipliers():
    adj = {**feedback.NEUTRAL, "word_rep": 1.5, "interjection": 0.5}
    t = feedback.scale_thresholds([0.4, 0.8, 0.5, 0.5, 0.4], adj)
    assert np.allclose(t, [0.4, 0.99, 0.5, 0.5, 0.2])  # capped below 1; blocks untouched
    assert feedback.confirm(0.3, feedback.NEUTRAL) == 0.3
    assert np.allclose(feedback.confirm(0.3, adj), [0.3, 0.45, 0.3, 0.15])

    toks = label_tokens([{"w": w, "s": 0.5 * i, "e": 0.5 * i + 0.4} for i, w in enumerate("I I want tea".split())])
    probs = np.zeros((100, 4))
    probs[:, 1] = 0.35  # the teacher hears a word repeat, a little
    assert word_cuts(toks, probs, confirm=0.3)  # cut by default
    assert not word_cuts(toks, probs, confirm=feedback.confirm(0.3, adj))  # this person undid such cuts


def test_missed_stutter_kind_comes_from_the_teacher():
    probs = np.zeros((200, 5))
    probs[95:105, 2] = 0.7  # a prolongation around 2 s
    probs[150:160, 1] = 0.9  # a word repeat later, outside the window
    assert feedback.kind_at(probs, 2.0) == "prolongation"
    assert feedback.kind_at(probs, 99.0) is None


def test_live_detector_thresholds_move(tmp_path):
    model = CausalStudent(hidden=16, conv=16, layers=1)
    path = tmp_path / "student.pt"
    torch.save({"state_dict": model.state_dict(), "lookaheads": model.lookaheads, "arch": model.arch}, path)
    base, _ = make_detector("student", "gentle", path)
    det, name = make_detector("student", "gentle", path, {**feedback.NEUTRAL, "word_rep": 1.2, "sound_rep": 0.5})
    assert name == "student"
    assert det.thresholds[1] == min(0.99, base.thresholds[1] * 1.2)
    assert det.thresholds[0] == max(det.MIN_CUT_THRESHOLD, base.thresholds[0] * 0.5)  # never below the floor


def test_feedback_endpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "LOG", tmp_path / "feedback.jsonl")
    monkeypatch.delenv("STUTTERCLEAR_HOSTED", raising=False)
    probs = np.zeros((200, 5))
    probs[95:105, 4] = 0.8  # a filler at 2 s
    monkeypatch.setattr(server.notes, "ready", lambda nid: {"probs": probs} if nid == "note-1" else None)

    r = client.post("/api/feedback", json=cut("word_rep", "reject")).json()
    assert r["stored"] and r["multipliers"]["word_rep"] == 1.05
    r = client.post("/api/feedback", json={"id": "note-1", "missed_at_s": 2.0}).json()
    assert r["entry"]["kind"] == "interjection" and r["multipliers"]["interjection"] == 0.95
    assert client.get("/api/feedback").json()["entries"] == 2
    assert client.post("/api/feedback", json={"verdict": "accept"}).status_code == 400
    assert client.delete("/api/feedback").json()["ok"] and client.get("/api/feedback").json()["entries"] == 0

    monkeypatch.setattr(server, "HOSTED", True)  # a public server: handed back, never written
    r = client.post("/api/feedback", json=cut("sound_rep", "reject")).json()
    assert r == {"entry": feedback.clean(cut("sound_rep", "reject")), "stored": False}
    assert not (tmp_path / "feedback.jsonl").exists()
    s = client.post("/api/feedback/summary", json={"entries": [r["entry"], {"junk": 1}]}).json()
    assert s["multipliers"]["sound_rep"] == 1.05 and s["entries"] == 1


def test_hosted_live_session_uses_the_pages_adjustments(monkeypatch):
    seen = {}
    real = server.LiveSession

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)
    monkeypatch.setattr(server, "LiveSession", spy)
    monkeypatch.setattr(server, "HOSTED", True)
    with client.websocket_connect("/ws/mic") as ws:
        ws.send_json({"mode": "gentle", "detector": "heuristic", "page": "p", "adjust": {"word_rep": 7}})
        ws.receive_json()
        ws.send_text("stop")
        ws.receive_json()
    assert seen["adjust"] == {**feedback.NEUTRAL, "word_rep": feedback.HIGH}
