"""Teacher: a self-supervised speech encoder with a frame-level stutter head.

Outputs per-frame logits (20 ms frames) for CLASSES. It is non-causal and large; it only
labels data and teaches the small causal student that runs live.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModel

from .synth import CLASSES


class FrameTeacher(nn.Module):
    def __init__(self, backbone: str = "microsoft/wavlm-base-plus", hidden: int = 256):
        super().__init__()
        self.backbone_name = backbone
        self.ssl = AutoModel.from_pretrained(backbone)
        self.ssl.freeze_feature_encoder()
        self.normalize = bool(getattr(AutoFeatureExtractor.from_pretrained(backbone), "do_normalize", False))
        n_layers = self.ssl.config.num_hidden_layers + 1
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))
        self.proj = nn.Linear(self.ssl.config.hidden_size, hidden)
        self.rnn = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.out = nn.Linear(2 * hidden, len(CLASSES))

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            wave = (wave - wave.mean(dim=1, keepdim=True)) / (wave.std(dim=1, keepdim=True) + 1e-5)
        hidden = self.ssl(wave, output_hidden_states=True).hidden_states
        w = torch.softmax(self.layer_weights, dim=0)
        x = sum(wi * h for wi, h in zip(w, hidden))
        x, _ = self.rnn(F.gelu(self.proj(x)))
        return self.out(x)  # [batch, frames, classes]

    def param_groups(self, lr_backbone: float, lr_head: float):
        backbone = [p for p in self.ssl.parameters() if p.requires_grad]
        head = [p for n, p in self.named_parameters() if not n.startswith("ssl.")]
        return [{"params": backbone, "lr": lr_backbone}, {"params": head, "lr": lr_head}]


def frame_loss(logits, labels, mask, pos_weight: float = 3.0):
    t = min(logits.shape[1], labels.shape[1])
    logits, labels, mask = logits[:, :t], labels[:, :t], mask[:, :t]
    pw = torch.full((logits.shape[-1],), pos_weight, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw, reduction="none").mean(-1)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def clip_logits(frame_logits, r: float = 5.0):
    """Log-sum-exp pooling: a smooth max over frames, for clip-level (weak) labels."""
    t = frame_logits.shape[1]
    return (torch.logsumexp(r * frame_logits, dim=1) - torch.log(torch.tensor(float(t)))) / r


def clip_loss(frame_logits, clip_labels):
    return F.binary_cross_entropy_with_logits(clip_logits(frame_logits), clip_labels)
