"""Turn per-frame stutter probabilities into cut segments, and render cuts offline.

Probabilities only cross the threshold in the core of a stutter, so naive thresholding leaves
the onset and tail of each repeat in the output ("LA- LAST"). The cut rules:
- hysteresis: a segment opens at the class threshold and extends while the score stays above
  off_ratio of it;
- merging: cut segments closer than merge_gap frames become one ("FOR uh FOR" goes together);
- snapping: each edge moves (within snap frames) to the quietest nearby frame, normally the gap
  between words, so no word is cut in half;
- keep the last copy: a repetition cut that runs into a pause has swallowed the final copy of
  the word, so it is pulled back to the last speech burst before that pause;
- a long cut (over refine_long_s) usually spans a dense stutter region with real words in
  between, so it is re-segmented strictly (no hysteresis) to keep only the stutter cores;
- a prolongation is shortened, not deleted: the first prolong_keep_ms and the last
  prolong_tail_ms of the held sound stay, so the word keeps its onset;
- a repetition or prolongation cut longer than max_cut_s is more likely a detector error than a
  stutter and is dropped.
"""
from dataclasses import dataclass

import numpy as np

from .synth import CLASSES, FRAME, SR


@dataclass(frozen=True)
class CutConfig:
    thresholds: tuple = (0.4, 0.35, 0.5, 0.5, 0.4)
    kinds: tuple = ("sound_rep", "word_rep", "prolongation", "interjection")
    off_ratio: float = 0.5
    merge_gap: int = 10  # frames (200 ms)
    snap: int = 8  # frames (160 ms)
    min_frames: int = 3  # 60 ms
    keep_last_copy: bool = False  # tested: never on the removal / words-lost Pareto front
    refine_long_s: float = 1e9
    prolong_keep_ms: float = 0.0
    prolong_tail_ms: float = 40.0
    max_cut_s: float = 1e9


OFF_RATIO = CutConfig.off_ratio
MERGE_GAP = CutConfig.merge_gap
SNAP = CutConfig.snap
MIN_FRAMES = CutConfig.min_frames
REPEATS = ("sound_rep", "word_rep")


def frame_energy_db(x: np.ndarray, n_frames: int) -> np.ndarray:
    n = min(n_frames, len(x) // FRAME)
    e = np.full(n_frames, -120.0)
    if n:
        fr = x[: n * FRAME].astype(np.float64).reshape(n, FRAME)
        e[:n] = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-12)
    return e


def raw_segments(probs: np.ndarray, thresholds, kinds, off_ratio: float = OFF_RATIO) -> list:
    """Hysteresis segments per class: [start_frame, end_frame, class name, peak]."""
    out = []
    for name in kinds:
        c = CLASSES.index(name)
        on, off = thresholds[c], thresholds[c] * off_ratio
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


def keep_last_copy(segs: list, energy_db: np.ndarray, pause_frames: int = 12) -> list:
    """A repetition cut must end just before the copy that is kept. If what follows the cut is a
    pause (no speech in the next pause_frames), the kept copy was swallowed: end the cut at the
    start of the last speech burst inside it instead."""
    speech = energy_db > max(np.percentile(energy_db, 20) + 12.0, -60.0)
    for s in segs:
        if s[2] not in REPEATS:
            continue
        a, b = s[0], s[1]
        if speech[b:b + pause_frames].any():
            continue
        inside = np.flatnonzero(speech[a:b])
        if len(inside) == 0:
            continue
        last = a + inside[-1]
        start = last
        while start > a and speech[start - 1]:
            start -= 1
        s[1] = start  # cut up to the onset of the final burst
    return segs


def cut_segments(probs, thresholds_or_cfg, x, kinds=None) -> list:
    """Segments to remove, in seconds. Accepts a CutConfig, or (thresholds, kinds) for the
    original v2 behaviour (no last-copy rule, no length cap)."""
    if isinstance(thresholds_or_cfg, CutConfig):
        cfg = thresholds_or_cfg
    else:
        cfg = CutConfig(thresholds=tuple(thresholds_or_cfg), kinds=tuple(kinds or CutConfig.kinds),
                        keep_last_copy=False, max_cut_s=1e9)
    energy = frame_energy_db(x, len(probs))
    segs = raw_segments(probs, cfg.thresholds, cfg.kinds, cfg.off_ratio)
    if cfg.merge_gap:
        segs = merge(segs, cfg.merge_gap)
    if cfg.snap:
        segs = snap(segs, energy, cfg.snap)
    if cfg.keep_last_copy:
        segs = keep_last_copy(segs, energy)
    refined = []
    for a, b, k, pk in segs:
        if k != "interjection" and (b - a) * FRAME / SR > cfg.refine_long_s:
            refined += [[a + sa, a + sb, sk, spk]
                        for sa, sb, sk, spk in raw_segments(probs[a:b], cfg.thresholds, cfg.kinds, 1.0)]
        else:
            refined.append([a, b, k, pk])
    keep = int(round(cfg.prolong_keep_ms * SR / 1000 / FRAME))
    tail = int(round(cfg.prolong_tail_ms * SR / 1000 / FRAME))
    out = []
    for a, b, k, pk in refined:
        if k == "prolongation" and keep:
            a, b = a + keep, b - tail
        dur = (b - a) * FRAME / SR
        if b - a < cfg.min_frames or (k != "interjection" and dur > cfg.max_cut_s):
            continue
        out.append({"kind": k, "start": round(a * FRAME / SR, 3), "end": round(b * FRAME / SR, 3),
                    "peak": round(pk, 3)})
    return out


# ---- offline rendering -------------------------------------------------------------------------

@dataclass(frozen=True)
class JoinConfig:
    xfade_ms: float = 10.0
    align_ms: float = 0.0  # WSOLA-style search for the best-matching join point (0 = off)
    pause_ms: float = 0.0  # room tone inserted where a stutter was removed
    level_match: bool = False


def room_tone(x: np.ndarray) -> np.ndarray:
    fr = x[: len(x) // 160 * 160].reshape(-1, 160)
    e = np.mean(fr ** 2, axis=1)
    quiet = fr[e <= np.percentile(e, 15)]
    return quiet.ravel() if len(quiet) else np.zeros(160, np.float32)


def _best_shift(tail, x, right_start, search, limit):
    """WSOLA-style: the shift of right_start (within +-search samples) whose first samples best
    match the output tail they are crossfaded with, so the join stays in phase."""
    n = len(tail)
    best, best_d = -np.inf, 0
    for d in range(-search, search + 1):
        r = right_start + d
        if r < 0 or r + n > min(len(x), limit):
            continue
        cand = x[r:r + n]
        c = float(np.dot(tail, cand) / (np.linalg.norm(tail) * np.linalg.norm(cand) + 1e-9))
        if c > best:
            best, best_d = c, d
    return best_d


def render_edited(x: np.ndarray, cuts: list, cfg: JoinConfig = JoinConfig()) -> np.ndarray:
    """Delete the cut spans (like editing a recording) with crossfaded, optionally aligned joins."""
    x = np.asarray(x, np.float32)
    xf = max(2, int(cfg.xfade_ms * SR / 1000))
    search = int(cfg.align_ms * SR / 1000)
    tone = room_tone(x)
    rng = np.random.default_rng(0)
    out, pos = [], 0
    for c in sorted(cuts, key=lambda c: c["start"]):
        a, b = int(c["start"] * SR), int(c["end"] * SR)
        if a <= pos:
            pos = max(pos, b)
            continue
        out.append(("keep", pos, a))
        pos = b
    out.append(("keep", pos, len(x)))

    y = x[out[0][1]:out[0][2]].copy()
    ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32)
    for _, s, e in out[1:]:
        if e <= s:
            continue
        if cfg.pause_ms > 0:
            n = int(cfg.pause_ms * SR / 1000)
            start = int(rng.integers(0, max(1, len(tone) - n)))
            pause = np.resize(tone[start:], n).astype(np.float32)
            y = _xfade(y, pause, ramp)
            seg = x[s:e].copy()
        else:
            d = _best_shift(y[-xf:], x, s, search, e) if search and len(y) >= xf else 0
            seg = x[s + d:e].copy()
        if cfg.level_match and len(y) > 800 and len(seg) > 800:
            g = np.sqrt(np.mean(y[-800:] ** 2) + 1e-9) / np.sqrt(np.mean(seg[:800] ** 2) + 1e-9)
            g = float(np.clip(g, 0.7, 1.4))
            seg[:800] *= np.linspace(g, 1.0, 800, dtype=np.float32)
        y = _xfade(y, seg, ramp)
    return y


def _xfade(y, seg, ramp):
    n = min(len(ramp), len(y), len(seg))
    if n < 2:
        return np.concatenate([y, seg])
    r = ramp[np.linspace(0, len(ramp) - 1, n).astype(int)]
    # equal-power crossfade keeps loudness steady through the join
    fade_in, fade_out = np.sin(r * np.pi / 2), np.cos(r * np.pi / 2)
    return np.concatenate([y[:-n], y[-n:] * fade_out + seg[:n] * fade_in, seg[n:]])
