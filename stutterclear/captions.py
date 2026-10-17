"""Live captions of what the listener hears, with NVIDIA's streaming FastConformer (NeMo
stt_en_fastconformer_hybrid_large_streaming, 480 ms chunks, int8 ONNX via sherpa-onnx, runs on CPU).

Model files go in models/asr/ (scripts/download_asr.py); without them captions are simply off.
"""
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "asr" / \
    "sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-480ms-int8"


def available(model_dir: Path = MODEL_DIR) -> bool:
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return False
    return (Path(model_dir) / "encoder.int8.onnx").exists()


def recognizer(model_dir: Path = MODEL_DIR, threads: int = 2):
    import sherpa_onnx
    d = Path(model_dir)
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
        joiner=str(d / "joiner.int8.onnx"), tokens=str(d / "tokens.txt"),
        num_threads=threads, sample_rate=SAMPLE_RATE, feature_dim=80, decoding_method="greedy_search",
        enable_endpoint_detection=True, rule1_min_trailing_silence=1.2, rule2_min_trailing_silence=0.6,
        rule3_min_utterance_length=15.0,
    )


class Captioner:
    """Feed audio, read lines: finished lines (ended at a pause) plus the line being spoken."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.rec = recognizer(model_dir)
        self.stream = self.rec.create_stream()
        self.lines: list[str] = []
        self.partial = ""
        self.t = 0.0  # seconds of audio fed
        self.words: list[dict] = []  # finished words with their start times (s)
        self._line_t0 = 0.0

    def feed(self, x: np.ndarray) -> None:
        self.stream.accept_waveform(SAMPLE_RATE, np.asarray(x, np.float32))
        self.t += len(x) / SAMPLE_RATE
        while self.rec.is_ready(self.stream):
            self.rec.decode_stream(self.stream)
        self.partial = self.rec.get_result(self.stream).strip().lower()
        if self.rec.is_endpoint(self.stream):
            self._finish()

    def _finish(self):
        r = self.rec.get_result_all(self.stream)
        text = r.text.strip().lower()
        if text:
            self.lines.append(text)
            self.words += _words(r.tokens, r.timestamps, self._line_t0)
        self.rec.reset(self.stream)
        self.partial = ""
        self._line_t0 = self.t

    def close(self) -> None:
        tail = np.zeros(int(0.8 * SAMPLE_RATE), np.float32)  # flush the last chunk
        self.stream.accept_waveform(SAMPLE_RATE, tail)
        self.stream.input_finished()
        while self.rec.is_ready(self.stream):
            self.rec.decode_stream(self.stream)
        self._finish()


def _words(tokens, times, t0) -> list:
    """BPE tokens (a leading space or '▁' starts a word) with times -> words with start times."""
    out = []
    for tok, t in zip(tokens, times):
        if tok[:1] in (" ", "▁") or not out:
            out.append({"w": tok.lstrip(" ▁").lower(), "t": round(t0 + float(t), 2)})
        else:
            out[-1]["w"] += tok.lower()
    return [w for w in out if w["w"]]


def caption_audio(x: np.ndarray) -> list:
    """Words with start times for a whole recording (used for demo samples and voice notes)."""
    c = Captioner()
    for i in range(0, len(x), SAMPLE_RATE // 10):
        c.feed(x[i:i + SAMPLE_RATE // 10])
    c.close()
    return c.words
