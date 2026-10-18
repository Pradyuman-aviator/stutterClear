"""Check the NVIDIA key and Say it (Magpie TTS) before a demo.

    python scripts/check_tts.py [--voice Magpie-Multilingual.EN-US.Aria] [--out say_it.wav]

Needs NVIDIA_API_KEY in the environment (build.nvidia.com, free). Sends only the made-up phrase
below and writes what came back to a WAV so you can listen to it.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stutterclear import tts  # noqa: E402
from stutterclear.audio import save_wav  # noqa: E402
from stutterclear.config import SAMPLE_RATE as SR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default=tts.VOICE)
    ap.add_argument("--text", default="Kaggle notebook")
    ap.add_argument("--out", default="say_it.wav")
    args = ap.parse_args()
    if not tts.available():
        sys.exit("Set NVIDIA_API_KEY (setx NVIDIA_API_KEY \"nvapi-...\", then open a new terminal).")
    print(f"Magpie TTS on {tts.SERVER}, voice {args.voice}")
    t0 = time.perf_counter()
    try:
        y = tts.synthesize(args.text, args.voice)
    except tts.TTSError as e:
        sys.exit(f"  FAIL  {e}")
    save_wav(Path(args.out), y)
    print(f"  ok    {len(y) / SR:.2f} s of audio in {time.perf_counter() - t0:.1f} s -> {args.out}")


if __name__ == "__main__":
    main()
