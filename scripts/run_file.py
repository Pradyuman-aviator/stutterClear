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
from stutterclear.config import PRESETS, SAMPLE_RATE as SR  # noqa: E402
from stutterclear.pipeline import stream  # noqa: E402

# Student head per mode: detection lookahead must fit inside the mode's delay.
STUDENT_LOOKAHEAD_MS = {"verbatim": 0, "trim": 200, "clean": 500}


def make_detector(args):
    if args.detector != "student":
        return None
    from stutterclear.student_runtime import StudentDetector
    report = Path(args.student).with_name("student_test_report.json")
    lookahead = args.lookahead_ms if args.lookahead_ms is not None else STUDENT_LOOKAHEAD_MS[args.mode]
    det = StudentDetector(args.student, lookahead, report)
    print(f"student detector: {det.lookahead_ms} ms lookahead, thresholds {np.round(det.thresholds, 2).tolist()}")
    return det


def add_detector_args(ap):
    ap.add_argument("--detector", choices=["heuristic", "student"], default="heuristic")
    ap.add_argument("--student", default="runs/student/student.pt", help="trained student checkpoint")
    ap.add_argument("--lookahead-ms", type=float, help="student head (default: by mode)")


run = stream  # kept for scripts that import it from here


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--mode", choices=list(PRESETS), default="clean")
    ap.add_argument("--labels", help="labels json (defaults to <wav>.labels.json if present)")
    ap.add_argument("--out", help="output wav (defaults to <wav>.<mode>.wav)")
    add_detector_args(ap)
    args = ap.parse_args()

    wav = Path(args.wav)
    x = load_wav(wav)
    pipe, y = run(x, args.mode, make_detector(args))

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
