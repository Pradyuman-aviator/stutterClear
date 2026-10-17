"""How jerky are the joins? Scores cut placement and join strategies on held-out recordings.

    python scripts/join_quality.py --dir data/external/uclass_eval

A join glues the audio before a cut to the audio after it. If both sides are speech and they sound
different (pitch, vowel, loudness), the listener hears a jump. For every join this measures the
spectral distance (log-mel, dB RMS over 40 bands) between the 30 ms either side, and calls it
audible when it exceeds the 95th percentile of the natural 30 ms-to-30 ms change in that same
recording. Joins next to silence, or masked by an inserted pause, cannot jump. Word loss and
stutter removal (silver labels) are reported too, so a smoother strategy cannot hide a worse cut.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stutterclear.audio import load_wav  # noqa: E402
from stutterclear.features import mel_filterbank  # noqa: E402
from stutterclear.segments import BALANCED_CUTS, NATURAL_CUTS, cut_segments, smooth_cuts  # noqa: E402
from stutterclear.silver import label_tokens, score_cuts  # noqa: E402

SR, HOP, WIN = 16000, 160, 400
SIDE = 3  # 10 ms frames on each side of a join (30 ms)
SPEECH_DB = -42.0  # frame energy above this counts as speech
MEL = mel_filterbank(SR, 512, 40, 60.0, 7600.0)


def logmel(x: np.ndarray) -> tuple:
    n = 1 + max(0, (len(x) - WIN) // HOP)
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    fr = x[np.minimum(idx, len(x) - 1)] * np.hanning(WIN)
    spec = np.abs(np.fft.rfft(fr, 512)) ** 2
    mel = 10 * np.log10(spec @ MEL.T + 1e-10)
    energy = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-12)
    return mel, energy


def side_distance(mel, a, b):
    """Distance between the 30 ms ending at frame a and the 30 ms starting at frame b."""
    left, right = mel[max(0, a - SIDE):a], mel[b:b + SIDE]
    if not len(left) or not len(right):
        return 0.0
    return float(np.sqrt(np.mean((left.mean(0) - right.mean(0)) ** 2)))


def natural_p95(mel, energy):
    d = [side_distance(mel, t, t) for t in range(SIDE, len(mel) - SIDE, 2)
         if energy[t - SIDE:t + SIDE].min() > SPEECH_DB]
    return float(np.percentile(d, 95)) if d else 99.0


def joins(cuts, mel, energy, p95, pause_ms):
    out = {"joins": 0, "speech_joins": 0, "audible": 0}
    for c in cuts:
        a, b = int(c["start"] * SR / HOP), int(c["end"] * SR / HOP)
        if a <= 0 or b >= len(mel) - 1:
            continue  # at the very start or end: nothing to glue
        out["joins"] += 1
        speech = energy[max(0, a - SIDE):a].min(initial=0) > SPEECH_DB and energy[b:b + SIDE].min(initial=0) > SPEECH_DB
        if not speech:
            continue
        out["speech_joins"] += 1
        if not pause_ms and side_distance(mel, a, b) > p95:
            out["audible"] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/external/uclass_eval")
    args = ap.parse_args()
    root = Path(args.dir)
    rows = []
    for w in sorted(p for p in root.rglob("*.wav") if "_edited" not in p.parts):
        x = load_wav(w)
        mel, energy = logmel(x)
        rows.append({"x": x, "probs": np.load(w.with_suffix(".teacher.npy")), "mel": mel, "energy": energy,
                     "p95": natural_p95(mel, energy),
                     "toks": label_tokens(json.loads(w.with_suffix(".crisper.json").read_text()))})
    strategies = {
        "balanced, as cut": dict(cuts=BALANCED_CUTS, merge_ms=0, snap_ms=0, pause_ms=0),
        "balanced, merge islands": dict(cuts=BALANCED_CUTS, merge_ms=150, snap_ms=0, pause_ms=0),
        "balanced, merge + snap": dict(cuts=BALANCED_CUTS, merge_ms=150, snap_ms=50, pause_ms=0),
        "natural (first edits)": dict(cuts=NATURAL_CUTS, merge_ms=0, snap_ms=0, pause_ms=0),
        "natural, merge islands": dict(cuts=NATURAL_CUTS, merge_ms=150, snap_ms=0, pause_ms=0),
    }
    print(f"{'strategy':26s} {'joins':>6} {'speech':>7} {'audible':>8} {'removed':>8} {'lost':>6}")
    for name, s in strategies.items():
        tot = {"joins": 0, "speech_joins": 0, "audible": 0}
        sc = {"n_stutter": 0, "n_removed": 0, "n_fluent": 0, "n_lost": 0}
        for r in rows:
            cuts = cut_segments(r["probs"], s["cuts"], r["x"])
            cuts = smooth_cuts(cuts, r["x"], merge_ms=s["merge_ms"], snap_ms=s["snap_ms"])
            j = joins(cuts, r["mel"], r["energy"], r["p95"], s["pause_ms"])
            for k in tot:
                tot[k] += j[k]
            q = score_cuts(r["toks"], cuts)
            for k in sc:
                sc[k] += q[k]
        removed = 100 * sc["n_removed"] / max(1, sc["n_stutter"])
        lost = 100 * sc["n_lost"] / max(1, sc["n_fluent"])
        print(f"{name:26s} {tot['joins']:6d} {tot['speech_joins']:7d} {tot['audible']:8d} {removed:7.1f}% {lost:5.2f}%")


if __name__ == "__main__":
    main()
