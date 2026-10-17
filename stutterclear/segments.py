"""Turn per-frame stutter probabilities into cut segments.

Probabilities only cross the threshold in the core of a stutter, so naive thresholding leaves
the onset and tail of each repeat in the output ("LA- LAST"). Three steps fix that:
- hysteresis: a segment opens at the class threshold and extends while the score stays above
  OFF_RATIO of it;
- merging: cut segments closer than MERGE_GAP frames become one ("FOR uh FOR" goes together);
- snapping: each edge moves (within SNAP frames) to the quietest nearby frame, normally the
  gap between words, so no word is cut in half.
"""
import numpy as np

from .synth import CLASSES, FRAME

OFF_RATIO = 0.5
MERGE_GAP = 10  # 200 ms
SNAP = 8  # 160 ms
MIN_FRAMES = 3  # 60 ms


def frame_energy_db(x: np.ndarray, n_frames: int) -> np.ndarray:
    e = np.full(n_frames, -120.0)
    for f in range(n_frames):
        seg = x[f * FRAME:(f + 1) * FRAME]
        if len(seg):
            e[f] = 10 * np.log10(np.mean(seg.astype(np.float64) ** 2) + 1e-12)
    return e


def raw_segments(probs: np.ndarray, thresholds, kinds) -> list:
    """Hysteresis segments per class: [start_frame, end_frame) with class name and peak."""
    out = []
    for name in kinds:
        c = CLASSES.index(name)
        on, off = thresholds[c], thresholds[c] * OFF_RATIO
        p = probs[:, c]
        f = 0
        while f < len(p):
            if p[f] < on:
                f += 1
                continue
            a = f
            while a > 0 and p[a - 1] >= off:
                a -= 1
            b = f
            while b < len(p) and p[b] >= off:
                b += 1
            out.append([a, b, name, float(p[a:b].max())])
            f = b
    return sorted(out, key=lambda s: s[0])


def merge(segs: list, gap: int = MERGE_GAP) -> list:
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= gap:
            m = merged[-1]
            if s[3] > m[3]:
                m[2] = s[2]  # keep the most confident class as the label
            m[1], m[3] = max(m[1], s[1]), max(m[3], s[3])
        else:
            merged.append(list(s))
    return merged


def snap(segs: list, energy_db: np.ndarray, width: int = SNAP) -> list:
    n = len(energy_db)
    for s in segs:
        lo = max(0, s[0] - width)
        s[0] = lo + int(np.argmin(energy_db[lo:s[0] + 1]))
        hi = min(n, s[1] + width)
        s[1] = s[1] + int(np.argmin(energy_db[s[1]:hi + 1])) if s[1] < n else s[1]
    return segs


def cut_segments(probs, thresholds, x, kinds=("sound_rep", "word_rep", "prolongation")) -> list:
    """Segments to remove, in seconds, from frame probabilities and the audio they describe."""
    energy = frame_energy_db(x, len(probs))
    segs = snap(merge(raw_segments(probs, thresholds, kinds)), energy)
    sr = 16000
    return [{"kind": k, "start": round(a * FRAME / sr, 3), "end": round(b * FRAME / sr, 3), "peak": round(pk, 3)}
            for a, b, k, pk in segs if b - a >= MIN_FRAMES]
