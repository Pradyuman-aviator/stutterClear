"""Silver labels: frame labels for real stuttered speech from verbatim word timings (CrisperWhisper).

Tokens of a verbatim transcript are labelled:
- filler  ([UM], [UH], ...)                    -> interjection
- cutoff  (a word ending in "-", e.g. "l-")    -> sound_rep
- repeat  (every copy of a repeated word or 2-3 word phrase except the last) -> word_rep
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
            k += 1
    return toks


def frame_targets(words: list, n_frames: int):
    """Returns (labels [n_frames, classes], class_mask [classes]) for one recording or clip."""
    lab = np.zeros((n_frames, len(CLASSES)), np.float32)
    toks = label_tokens(words)
    for i, t in enumerate(toks):
        if t["label"] == "fluent":
            continue
        end = t["e"]
        if i + 1 < len(toks) and 0 <= toks[i + 1]["s"] - end <= MAX_GAP_S:
            end = toks[i + 1]["s"]
        a, b = int(t["s"] * SR) // FRAME, -(-int(end * SR) // FRAME)
        lab[max(0, a):min(n_frames, b), CLASSES.index(KIND_TO_CLASS[t["label"]])] = 1.0
    mask = np.zeros(len(CLASSES), np.float32)
    mask[KNOWN] = 1.0
    return lab, mask
