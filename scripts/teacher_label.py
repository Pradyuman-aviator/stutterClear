"""Label a recording with the teacher: per-frame probabilities, stutter segments and a plot.

    python scripts/teacher_label.py sessions/<id>/raw.wav --ckpt runs/kaggle_teacher/teacher/best.pt

Writes <wav>.teacher.json (segments, thresholds used) and <wav>.teacher.png. The segments are a
starting point for hand labelling: correcting them is much faster than labelling from scratch.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_teacher import load_teacher  # noqa: E402
from stutterclear.audio import load_wav  # noqa: E402
from stutterclear.synth import CLASSES, FRAME, SR  # noqa: E402

COLORS = ("tab:red", "tab:orange", "tab:green", "tab:purple", "tab:blue")


@torch.no_grad()
def frame_probs(model, x, dev, chunk_s=30.0, overlap_s=2.0):
    """Teacher probabilities [frames, classes] for a long recording, in overlapping chunks."""
    n, hop, ov = len(x), int(chunk_s * SR), int(overlap_s * SR)
    total = np.zeros(((n - 400) // FRAME + 1, len(CLASSES)), np.float32)
    weight = np.zeros(len(total), np.float32)
    for start in range(0, n, hop - ov):
        seg = x[start:start + hop]
        if len(seg) < 4000:
            break
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            p = torch.sigmoid(model(torch.from_numpy(seg)[None].to(dev)).float())[0].cpu().numpy()
        f0 = start // FRAME
        k = min(len(p), len(total) - f0)
        total[f0:f0 + k] += p[:k]
        weight[f0:f0 + k] += 1
        if start + hop >= n:
            break
    return total / np.maximum(weight, 1)[:, None]


def segments(probs, thresholds, min_frames=3):
    out = []
    for c, name in enumerate(CLASSES):
        on = probs[:, c] >= thresholds[c]
        f = 0
        while f < len(on):
            if on[f]:
                g = f
                while g < len(on) and on[g]:
                    g += 1
                if g - f >= min_frames:
                    out.append({"kind": name, "start": round(f * FRAME / SR, 2), "end": round(g * FRAME / SR, 2),
                                "peak": round(float(probs[f:g, c].max()), 3)})
                f = g
            else:
                f += 1
    return sorted(out, key=lambda s: s["start"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--ckpt", default="runs/kaggle_teacher/teacher/best.pt")
    ap.add_argument("--thresholds", help="thresholds.json from eval_teacher (default 0.5)")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_teacher(args.ckpt, dev)
    x = load_wav(args.wav)
    probs = frame_probs(model, x, dev)
    th = json.loads(Path(args.thresholds).read_text()) if args.thresholds else [0.5] * len(CLASSES)
    segs = segments(probs, th)
    np.save(Path(args.wav).with_suffix(".teacher.npy"), probs)

    wav = Path(args.wav)
    wav.with_suffix(".teacher.json").write_text(json.dumps(
        {"checkpoint_step": ckpt.get("step"), "thresholds": th, "segments": segs}, indent=1))
    fig, ax = plt.subplots(2, 1, figsize=(max(12, len(x) / SR / 4), 5), sharex=True,
                           gridspec_kw={"height_ratios": [2, 2]})
    ax[0].specgram(x, NFFT=512, Fs=SR, noverlap=352, cmap="magma", vmin=-120)
    ax[0].set_ylim(0, 8000)
    t = np.arange(len(probs)) * FRAME / SR
    for c, name in enumerate(CLASSES):
        ax[1].plot(t, probs[:, c], color=COLORS[c], label=name, lw=1)
    ax[1].axhline(0.5, color="gray", lw=0.5, ls="--")
    ax[1].legend(loc="upper right", fontsize=7, ncol=5)
    ax[1].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(wav.with_suffix(".teacher.png"), dpi=80)
    for s in segs:
        print(f"{s['start']:7.2f}-{s['end']:6.2f}s  {s['kind']:<13} peak {s['peak']}")


if __name__ == "__main__":
    main()
