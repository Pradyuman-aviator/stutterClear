"""Can the big teacher run live? Each held-out recording is streamed through it on sliding windows
(CONTEXT s of past plus L s of lookahead, a decision every HOP s, keeping only the frames decided at
that moment), then cut and scored against silver labels. Full-context (offline) probabilities are
the ceiling. Also times one window, which must finish well inside HOP to keep up live.

    python scripts/teacher_lookahead.py --ckpt runs/kaggle_teacher/teacher/best.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_teacher import load_teacher  # noqa: E402
from stutterclear.audio import load_wav  # noqa: E402
from stutterclear.segments import BALANCED_CUTS, cut_segments  # noqa: E402
from stutterclear.silver import label_tokens, score_cuts  # noqa: E402

SR, FRAME = 16000, 320


@torch.no_grad()
def run_batch(model, segs, dev):
    with torch.autocast(dev.type, enabled=dev.type == "cuda"):
        return torch.sigmoid(model(torch.from_numpy(np.stack(segs)).to(dev)).float()).cpu().numpy()


def streamed_probs(model, x, dev, lookahead, context, hop, batch=16):
    """Frame probabilities as a live system would have them: each frame decided with only
    `lookahead` seconds of audio after it."""
    n_frames = (len(x) - 400) // FRAME + 1
    out = np.zeros((n_frames, 5), np.float32)
    win = int((context + lookahead) * SR)
    jobs = []  # (window start sample, first frame to keep, last frame to keep)
    t = hop
    while (t - hop) * SR < len(x):
        end = int(min(len(x), (t + lookahead) * SR))
        start = max(0, end - win)
        jobs.append((start, end, int(round((t - hop) * SR / FRAME)), int(round(t * SR / FRAME))))
        t += hop
    full = [j for j in jobs if j[1] - j[0] == win]
    edge = [j for j in jobs if j[1] - j[0] != win]
    for group, size in ((full, batch), (edge, 1)):
        for i in range(0, len(group), size):
            chunk = group[i:i + size]
            segs = [x[a:b] for a, b, _, _ in chunk]
            if any(len(s) < 4000 for s in segs):
                continue
            probs = run_batch(model, segs, dev)
            for (a, b, f0, f1), p in zip(chunk, probs):
                w0 = a // FRAME
                for f in range(f0, min(f1, n_frames)):
                    j = f - w0
                    if 0 <= j < len(p):
                        out[f] = p[j]
    return out


def chopped(toks, cuts):
    n = 0
    for c in cuts:
        for t in toks:
            if t["label"] == "fluent":
                ov = max(0.0, min(t["e"], c["end"]) - max(t["s"], c["start"])) / max(1e-3, t["e"] - t["s"])
                n += ov >= 0.3
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/kaggle_teacher/teacher/best.pt")
    ap.add_argument("--dir", default="data/external/uclass_eval")
    ap.add_argument("--lookaheads", default="0.2,0.5,1.0")
    ap.add_argument("--context", type=float, default=4.0)
    ap.add_argument("--hop", type=float, default=0.2)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_teacher(args.ckpt, dev)
    rows = []
    for w in sorted(p for p in Path(args.dir).rglob("*.wav") if "_edited" not in p.parts):
        rows.append((load_wav(w), np.load(w.with_suffix(".teacher.npy")),
                     label_tokens(json.loads(w.with_suffix(".crisper.json").read_text()))))
    audio_s = sum(len(x) for x, _, _ in rows) / SR

    # Live cost: one window at a time, as a live loop would run it.
    seg = rows[0][0][: int((args.context + 1.0) * SR)]
    for _ in range(3):
        run_batch(model, [seg], dev)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        run_batch(model, [seg], dev)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    per_window_ms = (time.perf_counter() - t0) / 20 * 1000
    print(f"{len(rows)} recordings, {audio_s / 60:.1f} min; one {args.context + 1.0:.0f} s window takes "
          f"{per_window_ms:.0f} ms on {dev.type} (a decision is due every {args.hop * 1000:.0f} ms)")

    def score(probs_of):
        tot = dict(n_stutter=0, n_removed=0, n_fluent=0, n_lost=0)
        chop = 0
        for (x, full, toks), probs in zip(rows, probs_of):
            cuts = cut_segments(probs, BALANCED_CUTS, x)
            q = score_cuts(toks, cuts)
            for k in tot:
                tot[k] += q[k]
            chop += chopped(toks, cuts)
        return (100 * tot["n_removed"] / max(1, tot["n_stutter"]), 100 * tot["n_lost"] / max(1, tot["n_fluent"]), chop)

    print(f"{'lookahead':>10} {'removed':>8} {'lost':>7} {'chopped words':>14}")
    r = score([full for _, full, _ in rows])
    print(f"{'offline':>10} {r[0]:7.1f}% {r[1]:6.2f}% {r[2]:14d}")
    for la in [float(v) for v in args.lookaheads.split(",")]:
        t0 = time.perf_counter()
        probs = [streamed_probs(model, x, dev, la, args.context, args.hop) for x, _, _ in rows]
        r = score(probs)
        print(f"{la * 1000:8.0f}ms {r[0]:7.1f}% {r[1]:6.2f}% {r[2]:14d}   ({time.perf_counter() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
