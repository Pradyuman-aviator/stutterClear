"""Real-time streaming Whisper ("WhisperFlow") pipeline for live captions and transcription.

Uses faster-whisper (CTranslate2) with int8 quantization on CPU or CUDA.
Processes incoming 16 kHz audio chunks, providing low-latency streaming partials
as words are spoken and finalizing lines at natural speech pauses.
"""
import os
import threading
from typing import Optional, List, Dict, Any

import numpy as np

from .config import SAMPLE_RATE

# Configurable defaults via environment variables
DEFAULT_MODEL_SIZE = os.environ.get("STUTTERCLEAR_WHISPER_MODEL", "tiny.en")
DEFAULT_DEVICE = os.environ.get("STUTTERCLEAR_WHISPER_DEVICE", "cpu")
DEFAULT_COMPUTE_TYPE = os.environ.get("STUTTERCLEAR_WHISPER_COMPUTE", "int8")

# Energy thresholds for voice activity detection (RMS on 16 kHz float32)
SPEECH_RMS_THRESHOLD = 0.008  # ~ -42 dB
SILENCE_RMS_THRESHOLD = 0.012  # ~ -38 dB
MIN_SPEECH_DURATION_S = 0.3  # minimum audio before attempting partial decode
SILENCE_WINDOW_S = 0.6  # trailing silence required to trigger sentence boundary
MAX_UTTERANCE_DURATION_S = 12.0  # max length before committing a line

_model_lock = threading.Lock()
_cached_models: Dict[str, Any] = {}


def available() -> bool:
    """Whether faster-whisper is installed and can be imported."""
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def get_model(
    model_size: Optional[str] = None,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
):
    """Get or load a cached WhisperModel instance."""
    if not available():
        raise RuntimeError("faster-whisper is not installed. Run: pip install faster-whisper")
    from faster_whisper import WhisperModel

    m_size = model_size or DEFAULT_MODEL_SIZE
    dev = device or DEFAULT_DEVICE
    comp = compute_type or DEFAULT_COMPUTE_TYPE
    key = f"{m_size}_{dev}_{comp}"

    with _model_lock:
        if key not in _cached_models:
            _cached_models[key] = WhisperModel(m_size, device=dev, compute_type=comp)
        return _cached_models[key]


class WhisperFlowCaptioner:
    """Streaming Whisper captioner matching StutterClear's Captioner interface.

    Accepts 16 kHz mono float32 audio chunks, exposes `lines`, `partial`, `words`, and `t`.
    """

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        model=None,
    ):
        self.model = model or get_model(model_size, device, compute_type)
        self.lines: List[str] = []
        self.partial: str = ""
        self.words: List[Dict[str, Any]] = []
        self.t: float = 0.0  # total seconds of audio fed
        self._line_t0: float = 0.0  # start time of the active utterance

        self._buffer: np.ndarray = np.zeros(0, dtype=np.float32)
        self._new_samples: int = 0
        self._had_speech: bool = False
        self._lock = threading.Lock()

    def feed(self, x: np.ndarray) -> None:
        """Feed an audio chunk (16 kHz mono float32) into the streaming pipeline."""
        if x is None or len(x) == 0:
            return
        chunk = np.asarray(x, dtype=np.float32)
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=-1)

        chunk_len_s = len(chunk) / SAMPLE_RATE
        with self._lock:
            self._buffer = np.concatenate([self._buffer, chunk])
            self.t += chunk_len_s
            self._new_samples += len(chunk)

            buf_len = len(self._buffer)
            silence_samples = int(SILENCE_WINDOW_S * SAMPLE_RATE)

            # Check if current utterance has finished (trailing silence or max duration)
            if buf_len >= int(1.2 * SAMPLE_RATE) and self._had_speech:
                trailing = self._buffer[-silence_samples:]
                trailing_rms = np.sqrt(np.mean(trailing ** 2) + 1e-12)
                buf_s = buf_len / SAMPLE_RATE

                if trailing_rms < SILENCE_RMS_THRESHOLD or buf_s >= MAX_UTTERANCE_DURATION_S:
                    self._finish_utterance()
                    return

            # Periodic update for active speaking partial (every ~350 ms)
            if self._new_samples >= int(0.35 * SAMPLE_RATE):
                rms = np.sqrt(np.mean(self._buffer ** 2) + 1e-12)
                if rms >= SPEECH_RMS_THRESHOLD:
                    self._had_speech = True
                    try:
                        segments, _ = self.model.transcribe(
                            self._buffer,
                            beam_size=1,
                            language="en",
                            condition_on_previous_text=False,
                            temperature=0.0,
                            vad_filter=True,
                            word_timestamps=False,
                        )
                        text = " ".join(s.text.strip() for s in segments if s.text.strip()).lower()
                        self.partial = text
                    except Exception:
                        pass
                    self._new_samples = 0
                elif not self._had_speech:
                    # Keep rolling buffer small during silence to avoid memory accumulation
                    max_silent_keep = int(0.2 * SAMPLE_RATE)
                    if len(self._buffer) > max_silent_keep:
                        self._buffer = self._buffer[-max_silent_keep:]
                    self._new_samples = 0

    def _finish_utterance(self) -> None:
        """Decode the finished utterance with word timestamps, commit lines and words, and reset buffer."""
        if len(self._buffer) < int(MIN_SPEECH_DURATION_S * SAMPLE_RATE) or not self._had_speech:
            self.partial = ""
            self._buffer = np.zeros(0, dtype=np.float32)
            self._new_samples = 0
            self._had_speech = False
            self._line_t0 = self.t
            return

        try:
            segments, _ = self.model.transcribe(
                self._buffer,
                beam_size=1,
                language="en",
                condition_on_previous_text=False,
                temperature=0.0,
                vad_filter=True,
                word_timestamps=True,
            )
            parts = []
            new_words = []
            for s in segments:
                t = s.text.strip().lower()
                if t:
                    parts.append(t)
                    if s.words:
                        for w in s.words:
                            cw = w.word.strip().lower()
                            if cw:
                                new_words.append({
                                    "w": cw,
                                    "t": round(self._line_t0 + float(w.start), 2),
                                })
            full_line = " ".join(parts)
            if full_line:
                self.lines.append(full_line)
                self.words.extend(new_words)
        except Exception:
            pass
        finally:
            self.partial = ""
            self._line_t0 = self.t
            self._buffer = np.zeros(0, dtype=np.float32)
            self._new_samples = 0
            self._had_speech = False

    def close(self) -> None:
        """Flush the remaining audio buffer and finalize."""
        with self._lock:
            if len(self._buffer) >= int(MIN_SPEECH_DURATION_S * SAMPLE_RATE) and self._had_speech:
                self._finish_utterance()
            self.partial = ""
            self._buffer = np.zeros(0, dtype=np.float32)


def caption_audio(x: np.ndarray, model=None) -> List[Dict[str, Any]]:
    """Transcribe a full audio array and return word start timestamps.

    Returns:
        List of dicts: [{"w": word, "t": start_seconds}]
    """
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return []
    if x.ndim > 1:
        x = x.mean(axis=-1)

    m = model or get_model()
    try:
        segments, _ = m.transcribe(
            x,
            beam_size=1,
            language="en",
            condition_on_previous_text=False,
            temperature=0.0,
            vad_filter=True,
            word_timestamps=True,
        )
        words = []
        for s in segments:
            if s.words:
                for w in s.words:
                    cw = w.word.strip().lower()
                    if cw:
                        words.append({"w": cw, "t": round(float(w.start), 2)})
        return words
    except Exception:
        return []
