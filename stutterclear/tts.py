"""Say it: NVIDIA Magpie TTS speaks a word or short phrase the user picked (On stage, word rescue).

Nothing is ever spoken without the user's click, and the voice is always labelled as synthetic.
NVIDIA_API_KEY comes from the environment (build.nvidia.com, free); MAGPIE_VOICE, MAGPIE_SERVER and
MAGPIE_FUNCTION_ID override the defaults. Only the text is sent.

The hosted model is served over gRPC (NVIDIA Cloud Functions). This calls its Synthesize method with
plain grpcio and writes the two small Riva messages by hand: nvidia-riva-client would pull in an
older protobuf than the rest of the stack uses.
"""
import os
from functools import lru_cache

import numpy as np

from .config import SAMPLE_RATE as SR

SERVER = os.environ.get("MAGPIE_SERVER", "grpc.nvcf.nvidia.com:443")
FUNCTION_ID = os.environ.get("MAGPIE_FUNCTION_ID", "877104f7-e885-42b9-8de8-f6e4c6303969")
VOICE = os.environ.get("MAGPIE_VOICE", "Magpie-Multilingual.EN-US.Aria")
METHOD = "/nvidia.riva.tts.RivaSpeechSynthesis/Synthesize"
LINEAR_PCM = 1  # riva_audio.proto AudioEncoding
MAX_CHARS = 200
LABEL = "Synthetic voice: NVIDIA Magpie TTS"


class TTSError(RuntimeError):
    pass


def available() -> bool:
    if not os.environ.get("NVIDIA_API_KEY"):
        return False
    try:
        import grpc  # noqa: F401
    except ImportError:
        return False
    return True


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _text(num: int, s: str) -> bytes:
    b = s.encode()
    return _varint(num << 3 | 2) + _varint(len(b)) + b


def request(text: str, voice: str = VOICE, language: str = "en-US", sr: int = SR) -> bytes:
    """A SynthesizeSpeechRequest (riva_tts.proto): text 1, language_code 2, encoding 3,
    sample_rate_hz 4, voice_name 5."""
    return (_text(1, text) + _text(2, language) + _varint(3 << 3) + _varint(LINEAR_PCM)
            + _varint(4 << 3) + _varint(sr) + _text(5, voice))


def audio_field(msg: bytes) -> bytes:
    """Field 1 (audio, bytes) of a SynthesizeSpeechResponse; other fields are skipped."""
    i = 0

    def varint():
        nonlocal i
        n = shift = 0
        while True:
            b = msg[i]
            i += 1
            n |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                return n

    while i < len(msg):
        key = varint()
        num, wire = key >> 3, key & 7
        if wire == 0:
            varint()
        elif wire == 1:
            i += 8
        elif wire == 5:
            i += 4
        elif wire == 2:
            n = varint()
            if num == 1:
                return msg[i:i + n]
            i += n
        else:
            raise TTSError(f"Magpie sent a reply this app cannot read (wire type {wire}).")
    return b""


_channel = None


def _call(req: bytes, timeout: float) -> bytes:
    global _channel
    import grpc
    if _channel is None:
        _channel = grpc.secure_channel(SERVER, grpc.ssl_channel_credentials())
    rpc = _channel.unary_unary(METHOD)  # no serializers: the messages are bytes already
    meta = (("function-id", FUNCTION_ID), ("authorization", f"Bearer {os.environ.get('NVIDIA_API_KEY', '')}"))
    return rpc(req, metadata=meta, timeout=timeout)


@lru_cache(maxsize=64)  # the same word is often asked for twice (a retry, a rehearsal)
def _pcm(text: str, voice: str) -> bytes:
    try:
        reply = _call(request(text, voice), timeout=15.0)
    except Exception as e:
        raise TTSError(f"Magpie TTS did not answer: {e}") from e
    pcm = audio_field(reply)
    if not pcm:
        raise TTSError("Magpie TTS returned no audio.")
    return pcm


def synthesize(text: str, voice: str = VOICE) -> np.ndarray:
    """Mono float32 audio at the app's sample rate."""
    text = " ".join(text.split())[:MAX_CHARS]
    if not text:
        raise TTSError("Nothing to say.")
    if not os.environ.get("NVIDIA_API_KEY"):
        raise TTSError("Set NVIDIA_API_KEY where the app runs to use Say it.")
    pcm = _pcm(text, voice or VOICE)
    return np.frombuffer(pcm[:len(pcm) // 2 * 2], np.int16).astype(np.float32) / 32768.0
