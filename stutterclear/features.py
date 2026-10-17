"""Streaming log-mel / MFCC features at a 10 ms hop."""
import numpy as np

from .config import HOP, SAMPLE_RATE


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    edges = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(bins)))
    for m in range(n_mels):
        lo, c, hi = edges[m], edges[m + 1], edges[m + 2]
        fb[m] = np.clip(np.minimum((bins - lo) / (c - lo), (hi - bins) / (hi - c)), 0.0, None)
    return fb


def dct_matrix(n_out: int, n_in: int) -> np.ndarray:
    k = np.arange(n_out)[:, None]
    n = np.arange(n_in)[None, :]
    return np.cos(np.pi * k * (2 * n + 1) / (2 * n_in)) * np.sqrt(2.0 / n_in)


class FeatureStream:
    """Turns an audio stream into frames; frame t covers samples [t*HOP, t*HOP + WIN)."""

    WIN = 400  # 25 ms
    N_FFT = 512

    def __init__(self, sr: int = SAMPLE_RATE, n_mels: int = 40, n_mfcc: int = 13):
        self.window = np.hanning(self.WIN).astype(np.float32)
        self.fb = mel_filterbank(sr, self.N_FFT, n_mels, 60.0, 7600.0).astype(np.float32)
        self.dct = dct_matrix(n_mfcc, n_mels).astype(np.float32)
        self.pending = np.zeros(0, np.float32)
        self.next_frame = 0

    def push(self, x: np.ndarray):
        """Returns (first_frame_index, energy_db[F], logmel_db[F, n_mels], mfcc[F, n_mfcc])."""
        self.pending = np.concatenate([self.pending, np.asarray(x, np.float32).ravel()])
        n = (len(self.pending) - self.WIN) // HOP + 1 if len(self.pending) >= self.WIN else 0
        first = self.next_frame
        if n <= 0:
            return first, np.zeros(0), np.zeros((0, self.fb.shape[0])), np.zeros((0, self.dct.shape[0]))
        idx = np.arange(self.WIN)[None, :] + HOP * np.arange(n)[:, None]
        frames = self.pending[idx]
        self.pending = self.pending[n * HOP:]
        self.next_frame += n
        energy_db = 10.0 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10)
        power = np.abs(np.fft.rfft(frames * self.window, self.N_FFT)) ** 2
        logmel = 10.0 * np.log10(power @ self.fb.T + 1e-10)
        mfcc = logmel @ self.dct.T
        return first, energy_db, logmel, mfcc
