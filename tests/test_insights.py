"""Session insights with a stand-in Nemotron."""
from stutterclear import insights, nemotron


def test_prompt_carries_counts_words_and_recent_transcript():
    m = insights.messages({"word_rep": 4, "interjection": 0, "sound_rep": 2}, 61.0, 3.4, ["I", "b-"],
                          " ".join(f"w{i}" for i in range(400)))
    user = m[1]["content"]
    assert "4 word repeats, 2 broken-off starts" in user and "fillers" not in user
    assert "Words the stutters were on: I, b-" in user
    assert "w399" in user and "w100 " not in user


def test_parse_keeps_three_bullets(monkeypatch):
    monkeypatch.setattr(nemotron, "chat", lambda *a, **k: (
        "Here you go:\n- Long stretches flowed easily.\n- Words starting with b were hardest.\n"
        "- Try a short pause before 'budget'.\n- extra"))
    out = insights.summarize({"sound_rep": 3}, 40, 1.2)
    assert out == ["Long stretches flowed easily.", "Words starting with b were hardest.", "Try a short pause before 'budget'."]
