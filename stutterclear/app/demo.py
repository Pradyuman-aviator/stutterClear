"""'Try a real recording': short excerpts of UCLASS recordings from adult speakers, cleaned by one of
three engines, served as a timeline the page replays in sync with the audio (no mic needed).

- natural: the offline teacher finds the stutters, the verbatim transcript (CrisperWhisper) places
  every cut on word boundaries, so no real word is ever sliced; fillers are kept
- thorough: the same, and fillers (um, uh) are removed too
- live: the real-time pipeline exactly as a call gets it (delay buffer, online detector, splicer)

Model-edge cutting sliced real words in 30-43% of cuts (a word that loses its first or last sound is
what a listener hears as a jerk); word-aligned cutting slices none (scripts/check notes in commit).

The audio is read from the local UCLASS download (data/external) and never committed: UCLASS is
free for research and teaching with credit (CREDIT below), not for redistribution as a product.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..audio import load_wav
from ..config import HOP, SAMPLE_RATE as SR
from ..pipeline import stream
from ..segments import NATURAL_JOIN, render_edited
from ..guard import guarded_cuts
from ..silver import label_tokens

DATA = Path(__file__).resolve().parents[2] / "data" / "external"
SAMPLES = {s["id"]: s for s in json.loads(Path(__file__).with_name("demo_samples.json").read_text())}
CREDIT = ("Recordings: UCLASS, University College London (Howell, Davis & Bartrip, 2009). "
          "Data collection was supported by the Wellcome Trust.")
TEACHER_FRAME = 320  # 20 ms frames in the cached teacher probabilities


def _wav(sample: dict) -> Path:
    return DATA / sample["file"]


def engines(sample: dict) -> list:
    w = _wav(sample)
    offline = w.with_suffix(".teacher.npy").exists() and w.with_suffix(".crisper.json").exists()
    return (["natural", "thorough"] if offline else []) + ["live"]


def _tokens(sample: dict) -> list:
    """The excerpt's verbatim words, with times relative to the excerpt."""
    words = json.loads(_wav(sample).with_suffix(".crisper.json").read_text())
    a, b = sample["start"], sample["end"]
    inside = [dict(w, s=w["s"] - a, e=w["e"] - a) for w in words if w["s"] >= a and w["e"] <= b]
    return label_tokens(inside)


def available() -> list:
    return [{"id": s["id"], "title": s["title"], "who": s["who"], "duration_s": round(s["end"] - s["start"], 2),
             "engines": engines(s)} for s in SAMPLES.values() if _wav(s).exists()]


def _envelope(x: np.ndarray) -> list:
    n = len(x) // HOP
    return np.round(np.abs(x[: n * HOP]).reshape(n, HOP).max(1), 4).tolist()


@lru_cache(maxsize=8)
def _excerpt(sid: str) -> np.ndarray:
    s = SAMPLES[sid]
    x = load_wav(_wav(s))
    return x[int(s["start"] * SR): int(s["end"] * SR)]


def _live_detector():
    """The detector the page recommends (best measured quality this machine can run)."""
    from ..live import make_detector, recommended
    return make_detector(recommended(), "gentle")


def _source(sid: str):
    """Audio, teacher probabilities and labelled words of a demo sample or a cleaned voice note."""
    if sid.startswith("note-"):
        from . import notes
        note = notes.ready(sid)
        if note is None:
            raise KeyError(sid)
        return note["x"], note["probs"], note["toks"]
    s = SAMPLES[sid]
    x = _excerpt(sid)
    probs = np.load(_wav(s).with_suffix(".teacher.npy"))
    f0 = int(round(s["start"] * SR / TEACHER_FRAME))
    return x, probs[f0: f0 + len(x) // TEACHER_FRAME], _tokens(s)


def _captions(x: np.ndarray, y: np.ndarray) -> dict | None:
    """Word-timed captions of both tracks (NVIDIA streaming FastConformer), if the model is here."""
    from .. import captions
    if not captions.available():
        return None
    return {"original": captions.caption_audio(x), "clean": captions.caption_audio(y)}


@lru_cache(maxsize=16)
def render(sid: str, engine: str) -> dict:
    """Timeline for the page plus the original and cleaned audio of one sample or voice note."""
    if engine in ("natural", "thorough"):
        x, probs, toks = _source(sid)
        stutter = probs[:, [0, 1, 2, 4]]  # sound_rep, word_rep, prolongation, interjection
        thorough = engine == "thorough"
        cuts, guard = guarded_cuts(toks, stutter, cut_fillers=thorough)  # Nemotron checks each cut
        label = f"{'Thorough' if thorough else 'Natural'} · offline model"
        y = render_edited(x, cuts, NATURAL_JOIN)
        cuts = [{"kind": c["kind"], "start": c["start"], "end": c["end"],
                 "word": " ".join(toks[i]["w"] for i in c["tokens"])} for c in cuts]
        events, meta = [], {"detector": "teacher", "label": label, "lag_ms": 0, "detect_ms": 0.0, "guard": guard}
    elif engine == "live":
        x = _source(sid)[0] if sid.startswith("note-") else _excerpt(sid)
        from ..live import delay_overrides
        detector, name = _live_detector()
        pipe, y = stream(x, "trim", detector, **delay_overrides(name, "gentle"))  # Gentle mode's preset
        cuts = [{"kind": k, "start": round(a / SR, 3), "end": round(b / SR, 3)}
                for a, b, k in pipe.splicer.stats.removed_spans]
        events = [{"kind": e.kind, "start": round(e.start / SR, 3), "end": round(e.end / SR, 3)}
                  for e in getattr(pipe.detector, "events", []) if e.kind in ("block", "pause")]
        summary = pipe.summary()
        from ..live import LABELS
        meta = {"detector": name, "label": f"Live · {LABELS.get(name, 'rule-based')}, Gentle",
                "lag_ms": summary["lag_ms"]["p50"], "detect_ms": 0.0, "guard": None}
    else:
        raise KeyError(engine)
    timeline = {
        "id": sid, "engine": engine, **meta,
        "original_s": round(len(x) / SR, 2), "clean_s": round(len(y) / SR, 2),
        "removed_s": round(sum(c["end"] - c["start"] for c in cuts), 2),
        "env_in": _envelope(x), "env_out": _envelope(y), "cuts": cuts, "events": events,
        "captions": _captions(x, y),
    }
    return {"timeline": timeline, "original": x, "clean": y}
