"""Zoomed plots of teacher predictions, from the .teacher.npy / .teacher.json written by teacher_label.py.

    python scripts/plot_teacher.py data/external/uclass/M_0065_20y1m_1.wav --page-s 20

Writes <wav stem>_plots/page_NN.png: spectrogram, per-class probabilities, shaded segments.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stutterclear.audio import load_wav  # noqa: E402
from stutterclear.synth import CLASSES, FRAME, SR  # noqa: E402

COLORS = dict(zip(CLASSES, ("tab:red", "tab:orange", "tab:green", "tab:purple", "tab:blue")))
CUT = {"sound_rep", "word_rep", "prolongation", "interjection"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--page-s", type=float, default=20.0)
    args = ap.parse_args()
    wav = Path(args.wav)
    x = load_wav(wav)
    probs = np.load(wav.with_suffix(".teacher.npy"))
    meta = json.loads(wav.with_suffix(".teacher.json").read_text())
    segs, th = meta["segments"], meta["thresholds"]
    out = wav.parent / f"{wav.stem}_plots"
    out.mkdir(exist_ok=True)
    t = np.arange(len(probs)) * FRAME / SR
    pages = int(np.ceil(len(x) / SR / args.page_s))
    for p in range(pages):
        a, b = p * args.page_s, min((p + 1) * args.page_s, len(x) / SR)
        fig, ax = plt.subplots(2, 1, figsize=(16, 6), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
        seg = x[int(a * SR):int(b * SR)]
        ax[0].specgram(seg, NFFT=512, Fs=SR, noverlap=352, cmap="magma", vmin=-120, xextent=(a, b))
        ax[0].set_ylim(0, 6000)
        ax[0].set_ylabel("Hz")
        in_page = [s for s in segs if s["end"] > a and s["start"] < b]
        for s in in_page:
            for axis in ax:
                axis.axvspan(s["start"], s["end"], color=COLORS[s["kind"]],
                             alpha=0.25 if s["kind"] in CUT else 0.12, lw=0)
            ax[0].text((s["start"] + s["end"]) / 2, 5700, s["kind"].replace("_", " "), ha="center", va="top",
                       fontsize=7, color="white", rotation=90 if s["end"] - s["start"] < 0.6 else 0)
        m = (t >= a) & (t <= b)
        for c, name in enumerate(CLASSES):
            ax[1].plot(t[m], probs[m, c], color=COLORS[name], lw=1.2, label=name)
        ax[1].axhline(th[0], color="gray", lw=0.6, ls="--")
        ax[1].set_ylim(0, 1.02)
        ax[1].set_ylabel("teacher probability")
        ax[1].set_xlabel("seconds")
        ax[1].legend(loc="upper right", fontsize=8, ncol=5)
        ax[0].set_title(f"{wav.name}  {a:.0f}-{b:.0f} s   shaded = detected (strong = cut, light = reported only)",
                        fontsize=10)
        ax[1].set_xlim(a, b)
        fig.tight_layout()
        fig.savefig(out / f"page_{p:02d}.png", dpi=85)
        plt.close(fig)
    print(f"{pages} pages -> {out}")


if __name__ == "__main__":
    main()
