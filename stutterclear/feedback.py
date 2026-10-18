"""Human feedback loop: the person reviews what was cut from their own speech (keep a cut, undo it,
or tap a stutter that was missed) and StutterClear adapts to them. Each kind of stutter has a
multiplier on its thresholds: an undone cut makes that kind a little more careful (+5%), a missed
stutter a little less (-5%), within 0.5-1.5. They apply to live detection (teacher and student) and
to the acoustic check on voice notes. When nearly every recent decision was right (95% of the last
50), the page may offer to stop asking (auto mode).

On a personal computer the log is runs/feedback/feedback.jsonl (STUTTERCLEAR_FEEDBACK overrides
it). A public server stores nothing: the page keeps its own log in the browser and sends the
multipliers with a live session.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

from .synth import CLASSES

KINDS = ("sound_rep", "word_rep", "prolongation", "interjection")  # what can be cut; blocks never are
STEP = 0.05
LOW, HIGH = 0.5, 1.5
AGREE_N, AGREE_MIN = 50, 0.95
LOG = Path(os.environ.get("STUTTERCLEAR_FEEDBACK")
           or Path(__file__).resolve().parents[1] / "runs" / "feedback" / "feedback.jsonl")
NEUTRAL = {k: 1.0 for k in KINDS}


def stored_here() -> bool:
    """Feedback is kept on this computer unless it serves the public."""
    return os.environ.get("STUTTERCLEAR_HOSTED") != "1"


def clean(entry: dict) -> dict | None:
    """A valid log entry, or None. {source, id, cut: {start, end, kind, words}, verdict: accept |
    reject} for a reviewed cut; {source, id, missed_at_s, kind} for a missed stutter (kind may be
    None when nobody could tell which kind it was)."""
    base = {"source": str(entry.get("source", ""))[:16], "id": str(entry.get("id", ""))[:40]}
    if entry.get("verdict") in ("accept", "reject"):
        cut = entry.get("cut") or {}
        if cut.get("kind") not in KINDS:
            return None
        try:
            span = {"start": round(float(cut["start"]), 3), "end": round(float(cut["end"]), 3)}
        except (KeyError, TypeError, ValueError):
            return None
        return {**base, "verdict": entry["verdict"],
                "cut": {**span, "kind": cut["kind"], "words": str(cut.get("words") or "")[:80]}}
    if entry.get("missed_at_s") is not None:
        try:
            t = round(float(entry["missed_at_s"]), 2)
        except (TypeError, ValueError):
            return None
        kind = entry.get("kind")
        return {**base, "missed_at_s": t, "kind": kind if kind in KINDS else None}
    return None


def _kind(e: dict):
    return (e.get("cut") or {}).get("kind") if "verdict" in e else e.get("kind")


def multipliers(entries: list) -> dict:
    m = dict(NEUTRAL)
    for e in entries:
        k = _kind(e)
        if k not in m:
            continue
        if e.get("verdict") == "reject":
            m[k] = min(HIGH, m[k] * (1 + STEP))
        elif "missed_at_s" in e:
            m[k] = max(LOW, m[k] * (1 - STEP))
    return {k: round(v, 4) for k, v in m.items()}


def agreement(entries: list) -> dict:
    """How often the recent decisions were right: a kept cut agrees, an undone cut or a missed
    stutter does not."""
    recent = [e for e in entries if "verdict" in e or "missed_at_s" in e][-AGREE_N:]
    rate = sum(e.get("verdict") == "accept" for e in recent) / len(recent) if recent else None
    return {"recent": len(recent), "agreement": None if rate is None else round(rate, 3),
            "auto_ready": len(recent) >= AGREE_N and rate >= AGREE_MIN}


def summary(entries: list) -> dict:
    return {"multipliers": multipliers(entries), **agreement(entries), "entries": len(entries)}


def read() -> list:
    if not LOG.exists():
        return []
    out = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        try:
            e = clean(json.loads(line))
        except (json.JSONDecodeError, AttributeError):
            continue
        if e:
            out.append(e)
    return out


def record(entry: dict) -> dict:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({**entry, "t": round(time.time())}) + "\n")
    return summary(read())


def forget():
    LOG.unlink(missing_ok=True)


def current() -> dict:
    """This person's multipliers (neutral on a public server, which keeps no log)."""
    return multipliers(read()) if stored_here() else dict(NEUTRAL)


def from_page(adjust) -> dict:
    """Multipliers a page sent (a public server's visitor keeps them in the browser): known kinds
    only, clamped, anything else ignored."""
    out = dict(NEUTRAL)
    for k, v in (adjust or {}).items() if isinstance(adjust, dict) else ():
        if k in out:
            try:
                out[k] = float(np.clip(float(v), LOW, HIGH))
            except (TypeError, ValueError):
                pass
    return out


def scale_thresholds(thresholds, adjust: dict) -> np.ndarray:
    """Live thresholds (one per class, in CLASSES order) times the multipliers."""
    t = np.array(thresholds, dtype=float)
    for i, c in enumerate(CLASSES):
        if c in adjust:
            t[i] = min(0.99, t[i] * adjust[c])
    return t


def confirm(base: float, adjust: dict):
    """The voice-note acoustic check per kind (columns sound_rep, word_rep, prolongation,
    interjection, as word_cuts gets them); the plain threshold when nothing has moved."""
    if all(adjust.get(k, 1.0) == 1.0 for k in KINDS):
        return base
    return np.minimum(0.99, np.array([base * adjust.get(k, 1.0) for k in KINDS]))


def kind_at(probs: np.ndarray, t: float, frame_s: float = 0.02, around_s: float = 0.3):
    """Which kind of stutter the teacher heard most around a tapped moment (probs: [frames, 5])."""
    a, b = max(0, int((t - around_s) / frame_s)), min(len(probs), int((t + around_s) / frame_s) + 1)
    if b <= a:
        return None
    cols = [CLASSES.index(k) for k in KINDS]
    return KINDS[int(np.argmax(probs[a:b][:, cols].max(0)))]
