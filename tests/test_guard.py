"""Meaning guard with a stand-in Nemotron: keeps grammatical repeats, deletes stutters."""
from stutterclear import guard, nemotron
from stutterclear.silver import label_tokens

WORDS = "I I want to say that that is true and she had had enough".split()


def toks():
    return label_tokens([{"w": w, "s": i * 0.3, "e": i * 0.3 + 0.2} for i, w in enumerate(WORDS)])


def test_marks_only_candidates():
    t = toks()
    cands = {i for c in guard.word_cuts(t) for i in c["tokens"]}
    assert [WORDS[i] for i in sorted(cands)] == ["I", "that", "had"]  # first copy of each repeat
    assert guard.marked_transcript(t, cands).startswith("[0:I] I want")


def test_keeps_what_nemotron_says_was_meant(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test")
    monkeypatch.setattr(nemotron, "chat", lambda *a, **k: (
        'Looking at it. {"keep": [{"id": 5, "reason": "grammatical: that that"}, '
        '{"id": 11, "reason": "grammatical: had had"}, {"id": 99, "reason": "not a candidate"}]}'))
    cuts, report = guard.guarded_cuts(toks())
    removed = [WORDS[i] for c in cuts for i in c["tokens"]]
    assert removed == ["I"]  # the stutter goes; both grammatical repeats stay
    assert report["checked"] == 3 and [k["word"] for k in report["kept"]] == ["that", "had"]


def test_without_a_key_or_an_answer_the_cuts_stand(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    cuts, report = guard.guarded_cuts(toks())
    assert report is None and len(cuts) == 3

    def down(*a, **k):
        raise nemotron.NemotronError("timeout")

    monkeypatch.setenv("NEBIUS_API_KEY", "test")
    monkeypatch.setattr(nemotron, "chat", down)
    cuts, report = guard.guarded_cuts(toks())
    assert len(cuts) == 3 and "timeout" in report["error"]
