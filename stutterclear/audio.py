import io
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from .config import SAMPLE_RATE


def load_wav(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    file_sr, x = wavfile.read(path)
    if x.dtype.kind == "i":
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if file_sr != sr:
        g = gcd(file_sr, sr)
        x = resample_poly(x, sr // g, file_sr // g).astype(np.float32)
    return x


def save_wav(path: str | Path, x: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sr, (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16))


def wav_bytes(x: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    """16-bit WAV in memory, for playing a recording back without writing it to disk."""
    buf = io.BytesIO()
    wavfile.write(buf, sr, (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16))
    return buf.getvalue()
