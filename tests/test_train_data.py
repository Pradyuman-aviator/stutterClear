"""Training data plumbing that only runs on Kaggle's full data: checked here on small stand-ins."""
import json

from stutterclear.silver import frame_targets
from stutterclear.train_data import load_silver, silver_words


def test_silver_records_give_word_lists(tmp_path):
    """Teacher v2's first Kaggle run crashed here: the whole record went where its words belong."""
    words = [{"w": "I", "s": 0.1, "e": 0.3}, {"w": "I", "s": 0.4, "e": 0.6}, {"w": "want", "s": 0.7, "e": 1.0}]
    p = tmp_path / "words_0.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"id": "train-00000-of-00007.parquet:2", "source": "sep", "words": words},
        {"id": "train-00000-of-00007.parquet:3", "source": "sep", "words": []},
    ]))
    silver = load_silver([p])
    assert silver_words(silver, "train-00000-of-00007.parquet:2") == words
    assert silver_words(silver, "train-00000-of-00007.parquet:3") is None  # nothing heard: no labels
    assert silver_words(silver, "train-00000-of-00007.parquet:9") is None
    assert silver_words(None, "anything") is None
    lab, cmask, fmask = frame_targets(silver_words(silver, "train-00000-of-00007.parquet:2"), 149)
    assert lab.shape[0] == 149 and lab.sum() > 0  # the repeated "I" is labelled
