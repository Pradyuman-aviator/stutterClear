"""Fluency journal: progress over time, kept only on the person's own computer and only after they
start it. Each live session and voice note adds one entry: how long, how many stutters of each kind,
and the words they were on (never audio). The summary shows stutters per minute by day, this week
against the last, and the hardest words and first sounds.

A public server keeps no journal. STUTTERCLEAR_JOURNAL moves the log (default runs/journal/).
"""
import json
import os
import re
import time
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

LOG = Path(os.environ.get("STUTTERCLEAR_JOURNAL")
           or Path(__file__).resolve().parents[1] / "runs" / "journal" / "journal.jsonl")
TOKEN_KIND = {"repeat": "word_rep", "cutoff": "sound_rep", "filler": "interjection"}
KINDS = ("sound_rep", "word_rep", "prolongation", "block", "interjection")
DIGRAPHS = ("th", "sh", "ch", "wh", "ph")


def stored_here() -> bool:
    return os.environ.get("STUTTERCLEAR_HOSTED") != "1"


def enabled() -> bool:
    """The journal runs once the person started it (the log exists) on their own computer."""
    return stored_here() and LOG.exists()


def start():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.touch()


def forget():
    LOG.unlink(missing_ok=True)


def _word(w: str) -> str:
    return re.sub(r"[^a-z']", "", w.lower())


def from_note(toks: list, seconds: float) -> dict:
    """A voice note: the stutters as the verbatim transcript shows them (repeats, broken-off starts,
    fillers), whatever was cut."""
    kinds = Counter(TOKEN_KIND[t["label"]] for t in toks if t["label"] in TOKEN_KIND)
    words = [_word(t["w"]) for t in toks if t["label"] in ("repeat", "cutoff")]
    return {"source": "note", "seconds": round(seconds, 1), "kinds": dict(kinds), "words": [w for w in words if w]}


def from_session(summary: dict, seconds: float) -> dict:
    """A live session: what the detector removed, and the blocks it reported (no words: live audio is
    not transcribed word by word)."""
    kinds = Counter(c["kind"] for c in summary.get("cuts", []))
    kinds.update(e["kind"] for e in summary.get("events", []) if e.get("kind") == "block")
    return {"source": "live", "seconds": round(seconds, 1), "kinds": dict(kinds), "words": []}


def record(entry: dict) -> bool:
    """Append an entry if the journal is on; long-enough stretches of speech only."""
    if not enabled() or entry["seconds"] < 3:
        return False
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({**entry, "t": round(time.time())}) + "\n")
    return True


def read() -> list:
    if not LOG.exists():
        return []
    out = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
            if isinstance(e, dict) and "t" in e and "seconds" in e:
                out.append(e)
        except json.JSONDecodeError:
            continue
    return out


def onset(word: str) -> str:
    w = word.strip("'-")
    return w[:2] if w[:2] in DIGRAPHS else w[:1]


def _rate(entries: list) -> float | None:
    minutes = sum(e["seconds"] for e in entries) / 60
    return round(sum(sum(e["kinds"].values()) for e in entries) / minutes, 2) if minutes >= 0.25 else None


def summary(entries: list, days: int = 14, today: date | None = None) -> dict:
    today = today or date.today()
    day_of = lambda e: datetime.fromtimestamp(e["t"]).date()  # noqa: E731
    by_day = []
    for k in range(days - 1, -1, -1):
        d = today - timedelta(days=k)
        es = [e for e in entries if day_of(e) == d]
        by_day.append({"date": d.isoformat(), "minutes": round(sum(e["seconds"] for e in es) / 60, 1),
                       "stutters": sum(sum(e["kinds"].values()) for e in es), "per_min": _rate(es)})
    week = [e for e in entries if (today - day_of(e)).days < 7]
    before = [e for e in entries if 7 <= (today - day_of(e)).days < 14]
    kinds = Counter()
    words = Counter()
    for e in entries:
        kinds.update(e["kinds"])
        words.update(e["words"])
    onsets = Counter(onset(w) for w in words.elements() if w)
    return {"enabled": enabled(), "stored": stored_here(), "entries": len(entries),
            "minutes": round(sum(e["seconds"] for e in entries) / 60, 1),
            "per_min": _rate(entries), "this_week": _rate(week), "last_week": _rate(before),
            "days": by_day, "kinds": {k: kinds[k] for k in KINDS if kinds[k]},
            "words": words.most_common(8), "onsets": onsets.most_common(5)}
