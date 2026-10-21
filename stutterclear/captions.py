"""Live captions of what the listener hears.

Supports two engines:
1. WhisperFlow (default): Streaming Whisper (faster-whisper, CTranslate2, int8 on CPU/CUDA),
   providing high-accuracy real-time transcription and word-level timestamps without manual setup.
2. NVIDIA streaming FastConformer: NeMo transducer via sherpa-onnx (int8 ONNX on CPU) in models/asr/.

Engine selection can be forced with STUTTERCLEAR_ASR_ENGINE ("whisperflow" or "fastconformer").
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

from .config import SAMPLE_RATE
from . import whisper_flow

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "asr" / \
    "sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-480ms-int8"


def fastconformer_available(model_dir: Path = MODEL_DIR) -> bool:
    """Whether sherpa-onnx and the FastConformer ONNX checkpoint exist."""
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return False
    return (Path(model_dir) / "encoder.int8.onnx").exists()


def engine_name() -> str:
    """The active ASR engine: 'whisperflow', 'fastconformer', or '' if none available."""
    forced = os.environ.get("STUTTERCLEAR_ASR_ENGINE", "").lower().strip()
    if forced:
        if forced in ("off", "none", "0"):
            return ""
        if forced == "whisperflow":
            return "whisperflow" if whisper_flow.available() else ""
        if forced == "fastconformer":
            return "fastconformer" if fastconformer_available() else ""
        return ""

    # Auto priority: WhisperFlow first, then FastConformer
    if whisper_flow.available():
        return "whisperflow"
    if fastconformer_available():
        return "fastconformer"
    return ""


def engine_label() -> str:
    """User-facing label for the active captions engine."""
    eng = engine_name()
    if eng == "whisperflow":
        m = os.environ.get("STUTTERCLEAR_WHISPER_MODEL", "tiny.en")
        return f"WhisperFlow ({m})"
    if eng == "fastconformer":
        return "NVIDIA FastConformer"
    return "Captions off"


def available(model_dir: Path = MODEL_DIR) -> bool:
    """Whether any captioning engine is available."""
    return bool(engine_name())


def recognizer(model_dir: Path = MODEL_DIR, threads: int = 2):
    """FastConformer sherpa-onnx recognizer."""
    import sherpa_onnx
    d = Path(model_dir)
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
        joiner=str(d / "joiner.int8.onnx"), tokens=str(d / "tokens.txt"),
        num_threads=threads, sample_rate=SAMPLE_RATE, feature_dim=80, decoding_method="greedy_search",
        enable_endpoint_detection=True, rule1_min_trailing_silence=1.2, rule2_min_trailing_silence=0.6,
        rule3_min_utterance_length=15.0,
    )


class FastConformerCaptioner:
    """FastConformer captioner feeding audio chunks to sherpa-onnx."""

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


def Captioner(model_dir: Path = MODEL_DIR):
    """Instantiate the active streaming Captioner (WhisperFlow or FastConformer)."""
    eng = engine_name()
    if eng == "whisperflow":
        return whisper_flow.WhisperFlowCaptioner()
    if eng == "fastconformer":
        return FastConformerCaptioner(model_dir)
    raise RuntimeError("No captioning engine is available.")


def caption_audio(x: np.ndarray) -> List[Dict[str, Any]]:
    """Words with start times for a whole recording (used for demo samples and voice notes)."""
    eng = engine_name()
    if eng == "whisperflow":
        return whisper_flow.caption_audio(x)
    if eng == "fastconformer":
        c = FastConformerCaptioner()
        for i in range(0, len(x), SAMPLE_RATE // 10):
            c.feed(x[i:i + SAMPLE_RATE // 10])
        c.close()
        return c.words
    return []
