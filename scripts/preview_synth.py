"""Generate a few synthetic stutters to listen to and inspect.

    python scripts/preview_synth.py --n 8

Writes demo/synth/NN.wav (stuttered), NN.orig.wav (fluent source), NN.png (spectrogram with
labels) and index.json (what was inserted where).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stutterclear.audio import save_wav  # noqa: E402
from stutterclear.synth import CLASSES, FRAME, SR, Utterance, augment, frame_labels, synthesize  # noqa: E402

COLORS = ("tab:red", "tab:orange", "tab:green", "tab:purple", "tab:blue")


def plot(path, y, lab, title):
    fig, ax = plt.subplots(2, 1, figsize=(12, 4), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].specgram(y, NFFT=512, Fs=SR, noverlap=352, cmap="magma", vmin=-120)
    ax[0].set_ylim(0, 8000)
    ax[0].set_title(title, fontsize=9)
    t = np.arange(len(lab)) * FRAME / SR
    for c, name in enumerate(CLASSES):
        ax[1].fill_between(t, c, c + 0.8 * lab[:, c], step="post", color=COLORS[c], label=name)
    ax[1].set_yticks(np.arange(len(CLASSES)) + 0.4, CLASSES, fontsize=8)
    ax[1].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--augment", action="store_true")
    args = ap.parse_args()

    files = sorted((ROOT / "data" / "librispeech-alignments" / "data").glob("dev_clean-*.parquet"))
    table = pq.read_table(files[0], columns=["id", "audio", "words", "phonemes"])
    rng = np.random.default_rng(args.seed)
    out_dir = ROOT / "demo" / "synth"
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    rows = rng.choice(table.num_rows, size=args.n * 3, replace=False)
    made = 0
    for r in rows:
        row = table.slice(int(r), 1).to_pylist()[0]
        utt = Utterance.from_row(row)
        y, spans, events = synthesize(utt, rng)
        if not events:
            continue
        if args.augment:
            y = augment(y, rng)
        lab = frame_labels(spans, len(y))
        name = f"{made:02d}"
        save_wav(out_dir / f"{name}.wav", y)
        save_wav(out_dir / f"{name}.orig.wav", utt.audio)
        title = " | ".join(f"{e['kind']}:{e['word']}" for e in events)
        plot(out_dir / f"{name}.png", y, lab, f"{row['id']}  {title}")
        index.append({"file": f"{name}.wav", "id": row["id"], "events": events,
                      "spans_s": [(round(s / SR, 2), round(e / SR, 2), CLASSES[c]) for s, e, c in spans]})
        made += 1
        if made == args.n:
            break
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    counts = {c: sum(e["kind"] == c for i in index for e in i["events"]) for c in CLASSES}
    print(f"wrote {made} examples to {out_dir}  events: {counts}")


if __name__ == "__main__":
    main()
