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
from stutterclear.audio import load_wav, save_wav  # noqa: E402
from stutterclear.segments import cut_segments  # noqa: E402
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
    ap.add_argument("--apply", action="store_true",
                    help="also write <wav>.teacher_clean.wav: teacher segments cut out by the splicer")
    ap.add_argument("--cut-interjections", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="reuse saved <wav>.teacher.npy instead of running the model")
    args = ap.parse_args()

    x = load_wav(args.wav)
    cache = Path(args.wav).with_suffix(".teacher.npy")
    if args.reuse and cache.exists():
        probs, ckpt = np.load(cache), json.loads(Path(args.wav).with_suffix(".teacher.json").read_text())
        ckpt = {"step": ckpt.get("checkpoint_step")}
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, ckpt = load_teacher(args.ckpt, dev)
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
    if args.apply:
        kinds = ("sound_rep", "word_rep", "prolongation") + (("interjection",) if args.cut_interjections else ())
        cuts = cut_segments(probs, th, x, kinds)
        save_wav(wav.with_suffix(".teacher_live.wav"), apply_cuts(x, cuts))
        y = splice_offline(x, cuts)
        save_wav(wav.with_suffix(".teacher_edited.wav"), y)
        print(f"cut {len(cuts)} segments, {sum(s['end'] - s['start'] for s in cuts):.1f} s of stutter.\n"
              f"  live (real time, stutters become short silences): {wav.with_suffix('.teacher_live.wav')}\n"
              f"  edited (stutters deleted, {len(x) / SR:.1f} s -> {len(y) / SR:.1f} s): "
              f"{wav.with_suffix('.teacher_edited.wav')}")


def splice_offline(x, cuts, xfade=80):
    """Delete the segments outright with short crossfades, like editing a recording."""
    keep, pos = [], 0
    for s in sorted(cuts, key=lambda s: s["start"]):
        a, b = int(s["start"] * SR), int(s["end"] * SR)
        if a > pos:
            keep.append(x[pos:a])
        pos = max(pos, b)
    keep.append(x[pos:])
    out = keep[0].copy()
    ramp = np.linspace(0.0, 1.0, xfade, dtype=np.float32)
    for seg in keep[1:]:
        n = min(xfade, len(out), len(seg))
        if n < 2:
            out = np.concatenate([out, seg])
            continue
        r = ramp[np.linspace(0, xfade - 1, n).astype(int)]
        out = np.concatenate([out[:-n], out[-n:] * r[::-1] + seg[:n] * r, seg[n:]])
    return out


def apply_cuts(x, cuts):
    """Offline preview: every segment registered with the splicer up front, then streamed through."""
    from stutterclear.config import HOP, preset
    from stutterclear.splicer import Splicer

    sp = Splicer(SR, preset("clean", target_delay_ms=200.0, min_delay_ms=0.0))
    for i, s in enumerate(cuts):
        sp.set_cut(i, int(s["start"] * SR), int(s["end"] * SR), s["kind"])
    out = []
    for src in (x, np.zeros(SR, np.float32)):
        for i in range(0, len(src), HOP):
            sp.write(src[i:i + HOP])
            out.append(sp.read(len(src[i:i + HOP])))
    y = np.concatenate(out)
    nz = np.flatnonzero(np.abs(y) > 1e-4)
    return y[nz[0]:nz[-1] + 1] if len(nz) else y


if __name__ == "__main__":
    main()
