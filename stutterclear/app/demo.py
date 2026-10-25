"""'Try a real recording': short excerpts of UCLASS recordings from adult speakers, cleaned by one of
three engines, served as a timeline the page replays in sync with the audio (no mic needed).

- natural: the offline teacher finds the stutters, the verbatim transcript (CrisperWhisper) places
  every cut on word boundaries, so no real word is ever sliced; fillers are kept
- thorough: the same, and ordinary disfluencies go too: fillers (um, uh), repeated phrases, and long
  silent pauses (a block or a hesitation) shrink to under half a second
- live: the real-time pipeline exactly as a call gets it (delay buffer, online detector, splicer)

Model-edge cutting sliced real words in 30-43% of cuts (a word that loses its first or last sound is
what a listener hears as a jerk); word-aligned cutting slices none (scripts/check notes in commit).

The audio is read from the local UCLASS download (data/external) and never committed: UCLASS is
free for research and teaching with credit (CREDIT below), not for redistribution as a product, so a
public server shows only synthetic and consented clips (data/demo_clips) unless told otherwise (the
Cloud Run demo is: its private image carries the three recordings, deploy/cloudrun/uclass.sha256).
The read-aloud clips with synthetic stutters stand in for real recordings: they show only where no
real recording can.
"""
import csv
import io
import json
import os
import re
import threading
import time
from collections import deque
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..audio import load_wav
from ..config import HOP, SAMPLE_RATE as SR
from ..pipeline import stream
from ..segments import NATURAL_JOIN, render_edited
from ..guard import guarded_cuts
from ..silver import label_tokens, tighten_pauses

DATA = Path(__file__).resolve().parents[2] / "data" / "external"
CLIPS = Path(__file__).resolve().parents[2] / "data" / "demo_clips"  # synthetic and consented clips
SAMPLES = {s["id"]: s for s in json.loads(Path(__file__).with_name("demo_samples.json").read_text())}
CREDIT = ("Recordings: UCLASS, University College London (Howell, Davis & Bartrip, 2009). "
          "Data collection was supported by the Wellcome Trust.")
SOURCE_CREDIT = {"uclass": CREDIT,
                 "synthetic": ("Read aloud: LibriSpeech (Panayotov et al., 2015; CC BY 4.0), with synthetic stutters "
                               "inserted by StutterClear."),
                 "consented": "Real recordings shared with consent for this demo."}
TEACHER_FRAME = 320  # 20 ms frames in the cached teacher probabilities


def uclass_enabled() -> bool:
    """UCLASS clips play on a personal computer; a public server leaves them out unless
    STUTTERCLEAR_UCLASS_DEMOS=1 (private judging). =0 hides them anywhere."""
    v = os.environ.get("STUTTERCLEAR_UCLASS_DEMOS")
    if v in ("0", "1"):
        return v == "1"
    return os.environ.get("STUTTERCLEAR_HOSTED") != "1"


def _source_of(sample: dict) -> str:
    return sample.get("source", "uclass")


def _wav(sample: dict) -> Path:
    return (DATA if _source_of(sample) == "uclass" else CLIPS) / sample["file"]


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
    shown = [s for s in SAMPLES.values() if _wav(s).exists() and (_source_of(s) != "uclass" or uclass_enabled())]
    if any(_source_of(s) != "synthetic" for s in shown):  # real recordings replace the stand-ins
        shown = [s for s in shown if _source_of(s) != "synthetic"]
    return [{"id": s["id"], "title": s["title"], "who": s["who"], "duration_s": round(s["end"] - s["start"], 2),
             "engines": engines(s), "source": _source_of(s)} for s in shown]


def credit(samples: list) -> str:
    """Credit lines for the kinds of clip on show."""
    return " ".join(SOURCE_CREDIT[k] for k in SOURCE_CREDIT if any(s["source"] == k for s in samples))


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
    from . import notes
    notes.release_gpu()  # the live teacher needs the GPU back from a finished voice note
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


def _note(sid: str) -> dict:
    from . import notes
    note = notes.ready(sid)
    if note is None:
        raise KeyError(sid)
    return note


_originals: dict = {}  # sample id -> captions of its original audio (the same for every engine)


def captions(sid: str, engine: str) -> dict | None:
    """Word-timed captions of a render's original and cleaned audio (NVIDIA streaming
    FastConformer), if the model is here. Kept with the render they describe: a sample's for good,
    a voice note's for as long as the note."""
    from .. import captions as asr
    if not asr.available():
        return None
    r = render(sid, engine)
    if "captions" not in r:
        home = _note(sid)["renders"] if sid.startswith("note-") else _originals
        if ("original", sid) not in home:
            home[("original", sid)] = asr.caption_audio(r["original"])
        r["captions"] = {"original": home[("original", sid)], "clean": asr.caption_audio(r["clean"])}
    return r["captions"]


_samples: dict = {}  # (sample, engine) -> (time, guard failed, render)
GUARD_RETRY_S = 300.0
NOTE_RENDERS = 12  # renders voice notes keep at once (each holds the cleaned audio); the oldest go first
_note_renders: deque = deque()  # (note, engine), oldest first
_note_locks = threading.Lock()


def render(sid: str, engine: str) -> dict:
    """Timeline for the page plus the original and cleaned audio of one sample or voice note.

    Demo samples are rendered once and kept, so visitors' voice notes never push them out of the
    cache (a re-render waits on Nemotron). If Nemotron did not answer, the guard is asked again
    after GUARD_RETRY_S. A voice note keeps its renders itself, so they are forgotten with it."""
    if sid.startswith("note-"):
        note = _note(sid)
        with _note_locks:
            lock = note.setdefault("lock", threading.Lock())
        with lock:  # one render of a note at a time: a second click waits for the first
            renders = note.setdefault("renders", {})
            if engine not in renders:
                renders[engine] = _render(sid, engine)
                _keep_note_render(sid, engine)
            return renders[engine]
    hit = _samples.get((sid, engine))
    if hit is None or (hit[1] and time.time() - hit[0] > GUARD_RETRY_S):
        r = _render(sid, engine)
        _samples[(sid, engine)] = (time.time(), bool((r["timeline"]["guard"] or {}).get("error")), r)
        return r
    return hit[2]


def _keep_note_render(sid: str, engine: str):
    """Past NOTE_RENDERS, the oldest render of a voice note is let go (asked for again, it is made
    again), so cleaned audio never piles up on a small server."""
    from . import notes
    with _note_locks:
        _note_renders.append((sid, engine))
        while len(_note_renders) > NOTE_RENDERS:
            old, old_engine = _note_renders.popleft()
            if (old, old_engine) != (sid, engine):
                notes._notes.get(old, {}).get("renders", {}).pop(old_engine, None)


def _env_calibration() -> dict:
    """The recordings teacher's calibration (its live_cuts.json) from STUTTERCLEAR_CALIBRATION_JSON,
    for a server that has no runs/ folder; empty when unset or unreadable."""
    try:
        c = json.loads(os.environ.get("STUTTERCLEAR_CALIBRATION_JSON") or "{}")
    except ValueError:
        return {}
    return c if isinstance(c, dict) else {}


def _render(sid: str, engine: str) -> dict:
    if engine in ("natural", "thorough"):
        x, probs, toks = _source(sid)
        stutter = probs[:, [0, 1, 2, 4]]  # sound_rep, word_rep, prolongation, interjection
        thorough = engine == "thorough"
        from .. import feedback
        from ..teacher_runtime import RECORDINGS_CKPT, calibration
        confirm = (calibration(RECORDINGS_CKPT) or _env_calibration()).get("confirm", 0.3)
        if sid.startswith("note-"):  # the person's own voice: their feedback applies
            confirm = feedback.confirm(confirm, feedback.current())
        cuts, guard = guarded_cuts(toks, stutter, cut_fillers=thorough, cut_phrases=thorough,  # Nemotron checks each cut
                                   confirm=confirm)
        label = f"{'Thorough' if thorough else 'Natural'} · offline model"
        text = transcript(toks, cuts)
        if thorough:  # long silent blocks and hesitations shrink too: a recording can give that time back
            cuts = sorted(cuts + tighten_pauses(toks, cuts), key=lambda c: c["start"])
        y = render_edited(x, cuts, NATURAL_JOIN)
        cuts = [{"kind": c["kind"], "start": c["start"], "end": c["end"],
                 "word": " ".join(toks[i]["w"] for i in c["tokens"])} for c in cuts]
        events, meta = [], {"detector": "teacher", "label": label, "lag_ms": 0, "detect_ms": 0.0, "guard": guard,
                            "text": text}
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
                "lag_ms": summary["lag_ms"]["p50"], "detect_ms": 0.0, "guard": None, "text": None}
    else:
        raise KeyError(engine)
    timeline = {
        "id": sid, "engine": engine, **meta,
        "original_s": round(len(x) / SR, 2), "clean_s": round(len(y) / SR, 2),
        "removed_s": round(sum(c["end"] - c["start"] for c in cuts), 2),
        "env_in": _envelope(x), "env_out": _envelope(y), "cuts": cuts, "events": events,
        "captions": None,  # captions(): only when the page asks (Dictate never shows them)
    }
    return {"timeline": timeline, "original": x, "clean": y}


SPOKEN_FILLER = {"[UM]": "um", "[UH]": "uh"}


def _tidy(text: str) -> str:
    """Spacing before punctuation, and a capital at the start of each sentence."""
    text = re.sub(r"\s+([,.!?;:])", r"\1", " ".join(text.split())).strip(" ,;:")
    text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return text


def transcript(toks: list, cuts: list) -> dict:
    """What was said, word by word, with what the cuts removed, and the cleaned text. The text also
    leaves out fillers and broken-off words a cut kept (text readers never want "um" or "b-"):
    dictation copies it as it is."""
    gone = {i for c in cuts for i in c["tokens"]}
    words = [{"w": SPOKEN_FILLER.get(t["w"].upper(), t["w"]), "removed": i in gone, "label": t["label"]}
             for i, t in enumerate(toks)]
    keep = [w["w"] for w in words if not w["removed"] and w["label"] not in ("filler", "cutoff")]
    return {"words": words, "clean": _tidy(" ".join(keep)), "said": _tidy(" ".join(w["w"] for w in words))}


KIND_NAMES = {"sound_rep": "broken-off start", "word_rep": "repeated word", "prolongation": "stretched sound",
              "interjection": "filler", "block": "block"}


def cuts_csv(cuts: list) -> str:
    """What was removed, one row per cut, on the original recording's clock."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["start_s", "end_s", "kind", "words"])
    for c in cuts:
        w.writerow([f"{c['start']:.3f}", f"{c['end']:.3f}", c["kind"], c.get("word", "")])
    return buf.getvalue()


def _srt_time(t: float) -> str:
    ms = max(0, round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def cuts_srt(cuts: list) -> str:
    """The cuts as subtitles over the original recording (an editor or a player shows each one)."""
    blocks = []
    for i, c in enumerate(cuts, 1):
        text = f"[removed: {KIND_NAMES.get(c['kind'], c['kind'])}]" + (f" {c['word']}" if c.get("word") else "")
        blocks.append(f"{i}\n{_srt_time(c['start'])} --> {_srt_time(c['end'])}\n{text}\n")
    return "\n".join(blocks)
