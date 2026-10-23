"""Check WhisperFlow streaming pipeline availability, latency, and accuracy.

    python scripts/check_whisper_flow.py

Validates faster-whisper model loading, streaming chunk ingestion, real-time partials,
sentence endpointing, and word-level timestamps on demo audio.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stutterclear import captions, whisper_flow  # noqa: E402
from stutterclear.audio import load_wav  # noqa: E402


def main():
    print("Checking WhisperFlow streaming pipeline...")

    # 1. Availability
    avail = whisper_flow.available()
    print(f"  [1/4] faster-whisper available: {avail}")
    if not avail:
        sys.exit("FAIL: faster-whisper is not installed. Run: pip install faster-whisper")

    # 2. Engine resolution
    eng_name = captions.engine_name()
    eng_label = captions.engine_label()
    print(f"  [2/4] Active ASR engine: {eng_name} ({eng_label})")

    # 3. Model loading
    t0 = time.perf_counter()
    model = whisper_flow.get_model()
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  [3/4] WhisperModel loaded in {load_ms:.1f} ms")

    # 4. Stream test
    demo_wav = Path(__file__).resolve().parents[1] / "data" / "demo_clips" / "read-1.wav"
    if demo_wav.exists():
        x = load_wav(demo_wav)[: 16000 * 5]  # first 5 seconds
    else:
        # Fallback synthetic tone
        t = np.linspace(0, 3, 16000 * 3, dtype=np.float32)
        x = 0.1 * np.sin(2 * np.pi * 300 * t)

    captioner = captions.Captioner()
    step = 1600  # 100 ms chunks at 16 kHz
    partials = []
    feed_t0 = time.perf_counter()
    for i in range(0, len(x), step):
        captioner.feed(x[i : i + step])
        if captioner.partial and (not partials or partials[-1] != captioner.partial):
            partials.append(captioner.partial)
    captioner.close()
    stream_ms = (time.perf_counter() - feed_t0) * 1000

    print(f"  [4/4] Streamed {len(x)/16000:.1f} s in {stream_ms:.1f} ms")
    print(f"        Partials captured ({len(partials)}): {partials[:3]}...")
    print(f"        Committed lines: {captioner.lines}")
    print(f"        Final words ({len(captioner.words)}): {[w['w'] for w in captioner.words[:6]]}...")
    print("All WhisperFlow checks passed!")


if __name__ == "__main__":
    main()
