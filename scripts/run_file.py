"""Run a WAV file through the same streaming pipeline used live, 10 ms at a time.

    python scripts/run_file.py demo/stuttered.wav --mode clean
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stutterclear.audio import load_wav, save_wav  # noqa: E402
from stutterclear.config import HOP, PRESETS, SAMPLE_RATE as SR, preset  # noqa: E402
from stutterclear.pipeline import Pipeline  # noqa: E402


def run(x: np.ndarray, mode: str, **overrides):
    pipe = Pipeline(preset(mode, **overrides))
    out = []
    tail = np.zeros(int(SR * (pipe.cfg.target_delay_ms / 1000 + 1.0)), np.float32)
    for src in (x, tail):
        for i in range(0, len(src), HOP):
            block = src[i:i + HOP]
            pipe.write(block)
            pipe.detect(block)
            out.append(pipe.read(len(block)))
    return pipe, np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--mode", choices=list(PRESETS), default="clean")
    ap.add_argument("--labels", help="labels json (defaults to <wav>.labels.json if present)")
    ap.add_argument("--out", help="output wav (defaults to <wav>.<mode>.wav)")
    args = ap.parse_args()

    wav = Path(args.wav)
    x = load_wav(wav)
    pipe, y = run(x, args.mode)

    labels_path = Path(args.labels) if args.labels else wav.with_suffix(".labels.json")
    labels = json.loads(labels_path.read_text())["remove"] if labels_path.exists() else None
    summary = pipe.summary(labels)
    summary["input_s"] = round(len(x) / SR, 2)

    out_path = Path(args.out) if args.out else wav.with_name(f"{wav.stem}.{args.mode}.wav")
    save_wav(out_path, y)
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
