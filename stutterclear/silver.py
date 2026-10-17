"""Silver labels: frame labels for real stuttered speech from verbatim word timings (CrisperWhisper).

Tokens of a verbatim transcript are labelled:
- filler  ([UM], [UH], ...)                    -> interjection
- cutoff  (a word ending in "-", e.g. "l-")    -> sound_rep
- repeat  (every copy of a repeated word or 2-3 word phrase except the last) -> word_rep;
            the tokens of one copy share "copy" (the index of its first token)
- fluent  (everything else, which must survive)
A stutter token's label runs on through the silence after it, up to the next word, so a cut
covers the gap too (as the synthetic labels do). Prolongations and blocks are not marked in a
verbatim transcript, so their frames are masked out and learned from other data.
"""
import re

import numpy as np

from .synth import CLASSES, FRAME, SR

KIND_TO_CLASS = {"filler": "interjection", "cutoff": "sound_rep", "repeat": "word_rep"}
KNOWN = [CLASSES.index(c) for c in KIND_TO_CLASS.values()]
MAX_GAP_S = 0.3
# Verbatim-ASR word boundaries on stuttered speech are off by ~100 ms on average (CrisperWhisper 2.0
# paper, FluencyBank), so frames just outside each labelled event are not trained on.
EDGE_TOLERANCE_FRAMES = 5  # 100 ms


def norm(w: str) -> str:
    return re.sub(r"[^a-z'\-\[\]]", "", w.lower())


def label_tokens(words: list) -> list:
    toks = [dict(w, n=norm(w["w"])) for w in words]
    for t in toks:
        t["label"] = "filler" if t["n"].startswith("[") else "cutoff" if t["n"].endswith("-") else "fluent"
    content = [i for i, t in enumerate(toks) if t["label"] == "fluent"]
    # Repeated words and short phrases, ignoring fillers and cut-offs between copies.
    for n in (3, 2, 1):
        k = 0
        while k + 2 * n <= len(content):
            a = [toks[i]["n"] for i in content[k:k + n]]
            b = [toks[i]["n"] for i in content[k + n:k + 2 * n]]
            if a == b and all(toks[i]["label"] == "fluent" for i in content[k:k + n]):
                for i in content[k:k + n]:
                    toks[i]["label"] = "repeat"
                    toks[i]["copy"] = content[k]
            k += 1
    return toks


def intended_text(model, x: np.ndarray) -> str:
    """CrisperWhisper's intended (clean) transcript. Its path without word timestamps often returns
    nothing for clips under 30 s, and padding them with silence makes it invent text ("I don't
    know. I don't know. ..."), so this asks for word timestamps, which takes the path that works."""
    return model.transcribe(np.asarray(x, np.float32), sr=SR, language="en", mode="intended",
                            word_timestamps=True).text


def word_cuts(toks: list, probs=None, cut_fillers: bool = False, confirm: float = 0.3,
              frame_s: float = 0.02, keep: frozenset = frozenset(), cut_phrases: bool = False) -> list:
    """Cuts placed on word boundaries from a verbatim transcript, for recordings (not live).

    Removes repeated and cut-off words (and fillers if asked), each from its start to the start of
    the next word, so no real word is ever sliced: model frame edges are about 100 ms off, a whole
    syllable, and a word that loses its first or last sound is what a listener hears as a jerk.
    With teacher probabilities, a candidate is cut only if the model also hears a stutter there
    (so a deliberate "very very" survives). A repeated phrase ("when he was when he was") is one
    candidate, cut or kept whole, never left half-cut as "when when he was". `cut_phrases` cuts
    repeated 2-3 word phrases without the acoustic check: like fillers they are ordinary
    disfluencies, rarely meant, and the model hears them less well than repeated sounds. `keep`
    lists token indices never to cut (the meaning guard's verdicts; one kept word keeps its whole
    phrase). Each cut lists the token indices it removes.
    """
    kinds = {"repeat", "cutoff"} | ({"filler"} if cut_fillers else set())
    groups = {}
    for i, t in enumerate(toks):
        if t["label"] in kinds:
            groups.setdefault(t.get("copy", i), []).append(i)

    def heard(ids):
        a = max(0, int((toks[ids[0]]["s"] - 0.1) / frame_s))
        b = min(len(probs), int((toks[ids[-1]]["e"] + 0.1) / frame_s) + 1)
        return b > a and float(probs[a:b].max()) >= confirm

    chosen = sorted(i for ids in groups.values() if not keep.intersection(ids)
                    and (probs is None or (cut_phrases and len(ids) > 1) or heard(ids)) for i in ids)
    cuts = []
    for i in chosen:
        t = toks[i]
        end = t["e"]
        if i + 1 < len(toks) and 0 <= toks[i + 1]["s"] - end <= 0.3:
            end = toks[i + 1]["s"]
        kind = {"repeat": "word_rep", "cutoff": "sound_rep", "filler": "interjection"}[t["label"]]
        if cuts and t["s"] - cuts[-1]["end"] <= 0.02:
            cuts[-1]["end"] = end
            cuts[-1]["tokens"].append(i)
        else:
            cuts.append({"kind": kind, "start": round(t["s"], 3), "end": round(end, 3), "tokens": [i]})
    return cuts


def _overlap(s, e, cuts) -> float:
    return sum(max(0.0, min(e, c["end"]) - max(s, c["start"])) for c in cuts) / max(1e-3, e - s)


def score_cuts(toks: list, cuts: list, removed_at: float = 0.5) -> dict:
    """Score a cut list against labelled tokens, without caring which copy of a repeat is kept.

    A repeat chain ("is is is is is") is the repeat tokens plus the final copy they repeat. It
    counts as resolved when exactly one copy survives; it counts as a lost word when none does.
    """
    cut = [_overlap(t["s"], t["e"], cuts) >= removed_at for t in toks]
    final_of = {}
    for i, t in enumerate(toks):
        if t["label"] != "repeat":
            continue
        for j in range(i + 1, min(len(toks), i + 12)):
            if toks[j]["n"] == t["n"] and toks[j]["label"] == "fluent":
                final_of[i] = j
                break
    chains = {}
    for i, j in final_of.items():
        chains.setdefault(j, []).append(i)
    out = {}
    for kind in ("filler", "cutoff"):
        idx = [i for i, t in enumerate(toks) if t["label"] == kind]
        out[f"{kind}_removed_pct"] = round(100 * sum(cut[i] for i in idx) / max(1, len(idx)), 1)
    extra = removed = lost_chains = 0
    for j, reps in chains.items():
        members = reps + [j]
        kept = sum(not cut[m] for m in members)
        extra += len(members) - 1
        removed += len(members) - 1 if kept == 0 else len(members) - kept
        lost_chains += kept == 0
    out["repeat_copies_removed_pct"] = round(100 * removed / max(1, extra), 1)
    in_chain = set(chains) | set(final_of)
    fluent = [i for i, t in enumerate(toks) if t["label"] == "fluent"]
    lost = sum(cut[i] for i in fluent if i not in in_chain) + lost_chains
    n_stutter = sum(t["label"] in ("filler", "cutoff") for t in toks) + extra
    n_removed = sum(cut[i] for i, t in enumerate(toks) if t["label"] in ("filler", "cutoff")) + removed
    out["stutter_removed_pct"] = round(100 * n_removed / max(1, n_stutter), 1)
    out["words_lost_pct"] = round(100 * lost / max(1, len(fluent)), 1)
    out["n_stutter"], out["n_fluent"], out["n_removed"], out["n_lost"] = n_stutter, len(fluent), n_removed, lost
    out["lost_examples"] = [f"{toks[i]['w']}@{toks[i]['s']:.1f}" for i in fluent if cut[i] and i not in in_chain][:8]
    return out


def frame_targets(words: list, n_frames: int):
    """Returns (labels [n_frames, classes], class_mask [classes], frame_mask [n_frames])."""
    lab = np.zeros((n_frames, len(CLASSES)), np.float32)
    frame_mask = np.ones(n_frames, np.float32)
    tol = EDGE_TOLERANCE_FRAMES
    toks = label_tokens(words)
    for i, t in enumerate(toks):
        if t["label"] == "fluent":
            continue
        end = t["e"]
        if i + 1 < len(toks) and 0 <= toks[i + 1]["s"] - end <= MAX_GAP_S:
            end = toks[i + 1]["s"]
        a, b = max(0, int(t["s"] * SR) // FRAME), min(n_frames, -(-int(end * SR) // FRAME))
        lab[a:b, CLASSES.index(KIND_TO_CLASS[t["label"]])] = 1.0
        frame_mask[max(0, a - tol):a] = 0.0
        frame_mask[b:b + tol] = 0.0
    frame_mask[lab[:, KNOWN].max(1) > 0] = 1.0  # positives of a neighbouring event still count
    class_mask = np.zeros(len(CLASSES), np.float32)
    class_mask[KNOWN] = 1.0
    return lab, class_mask, frame_mask
