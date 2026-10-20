"""Student: a small causal stutter detector that runs live.

Input is the same 80-band log-mel that features.FeatureStream computes at runtime (10 ms hop).
A causal stride-2 conv brings it to the teacher's 20 ms frames, a one-directional GRU keeps
state, and one output head per lookahead L predicts the labels of frame j - L at frame j.
The head choice is the latency dial: L frames of 20 ms is the detection delay.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .features import FeatureStream, mel_filterbank
from .synth import CLASSES, SR

N_MELS = 80
LOOKAHEADS = (0, 5, 10, 15, 25, 40)  # in 20 ms frames: 0, 100, 200, 300, 500, 800 ms


class LogMel(nn.Module):
    """Torch twin of FeatureStream's log-mel, so training and live inference see identical features."""

    def __init__(self, n_mels: int = N_MELS):
        super().__init__()
        self.win, self.hop, self.n_fft = FeatureStream.WIN, 160, FeatureStream.N_FFT
        self.register_buffer("window", torch.hann_window(self.win, periodic=False))
        self.register_buffer("fb", torch.tensor(mel_filterbank(SR, self.n_fft, n_mels, 60.0, 7600.0),
                                                dtype=torch.float32))

    def forward(self, wave: torch.Tensor) -> torch.Tensor:  # [B, N] -> [B, F, n_mels]
        # Always fp32: power spectra overflow fp16 under autocast.
        with torch.autocast(wave.device.type, enabled=False):
            frames = wave.float().unfold(-1, self.win, self.hop) * self.window
            power = torch.fft.rfft(frames, self.n_fft).abs() ** 2
            return 10.0 * torch.log10(power @ self.fb.T + 1e-10)


class CausalConv(nn.Module):
    def __init__(self, cin, cout, k, stride=1):
        super().__init__()
        self.pad = k - stride  # output j sees inputs up to index stride*j + stride - 1
        self.conv = nn.Conv1d(cin, cout, k, stride)

    def forward(self, x):  # [B, C, T]
        return self.conv(F.pad(x, (self.pad, 0)))


class CausalStudent(nn.Module):
    def __init__(self, hidden: int = 160, lookaheads=LOOKAHEADS, conv: int = 128, layers: int = 2):
        super().__init__()
        self.lookaheads = tuple(lookaheads)
        self.arch = {"hidden": hidden, "conv": conv, "layers": layers}
        self.logmel = LogMel()
        self.norm = nn.LayerNorm(N_MELS)  # per frame, so still causal
        self.conv1 = CausalConv(N_MELS, conv, 5, stride=2)
        self.conv2 = CausalConv(conv, hidden, 3)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True)
        self.heads = nn.ModuleList(nn.Linear(hidden, len(CLASSES)) for _ in self.lookaheads)

    def trunk(self, mel: torch.Tensor, h=None):  # mel [B, F, n_mels] -> [B, F//2, hidden]
        x = self.norm((mel + 40.0) / 20.0).transpose(1, 2)
        x = F.gelu(self.conv2(F.gelu(self.conv1(x)))).transpose(1, 2)
        return self.gru(x, h)

    def forward(self, wave: torch.Tensor) -> list:
        """Returns one [B, T, classes] logit tensor per lookahead head; row j of head L is frame j - L."""
        if wave.ndim == 1:
            wave = wave.unsqueeze(0)
        x, _ = self.trunk(self.logmel(wave))
        return [head(x) for head in self.heads]


def shifted(logits: torch.Tensor, target: torch.Tensor, lookahead: int):
    """Aligns a head's output row j with target frame j - lookahead."""
    t = min(logits.shape[1], target.shape[1])
    return logits[:, lookahead:t], target[:, :t - lookahead]


def distill_loss(heads_logits, lookaheads, soft, hard=None, mask=None, hard_weight: float = 0.5):
    total = 0.0
    for logits, la in zip(heads_logits, lookaheads):
        lg, sf = shifted(logits, soft, la)
        m = torch.ones(sf.shape[:2], device=sf.device) if mask is None else shifted(logits, mask[..., None], la)[1][..., 0]
        loss = F.binary_cross_entropy_with_logits(lg, sf, reduction="none").mean(-1)
        if hard is not None:
            _, hd = shifted(logits, hard, la)
            loss = (1 - hard_weight) * loss + hard_weight * F.binary_cross_entropy_with_logits(
                lg, hd, reduction="none").mean(-1)
        total = total + (loss * m).sum() / m.sum().clamp(min=1.0)
    return total / len(lookaheads)

