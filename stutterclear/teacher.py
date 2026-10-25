"""Teacher: a self-supervised speech encoder with a frame-level stutter head.

Outputs per-frame logits (20 ms frames) for CLASSES. It is non-causal and large; it only
labels data and teaches the small causal student that runs live.
"""
import os
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModel

from .synth import CLASSES

# The last encoder layer's raw output, kept per thread: the app runs one model from several threads
# at once (a live session, a demo render, a voice note), and a call must never read another's.
_last = threading.local()


def _keep_last(mod, args, out):  # returns None, so the layer's own output is untouched
    _last.out = out[0] if isinstance(out, tuple) else out


class FrameTeacher(nn.Module):
    def __init__(self, backbone: str = "microsoft/wavlm-base-plus", hidden: int = 256):
        super().__init__()
        self.backbone_name = backbone
        # STUTTERCLEAR_BACKBONE_REVISION pins the backbone's files to one tested revision (the GPU
        # worker sets it, so a new upload of the backbone cannot change what it runs); unset: the latest.
        revision = os.environ.get("STUTTERCLEAR_BACKBONE_REVISION")
        pin = {"revision": revision} if revision else {}
        # LayerDrop skips whole transformer layers at random while training; under DDP every
        # parameter must take part in every step, and a fixed depth is cleaner for distillation.
        # Set at load time too: some transformers versions copy it when the model is built.
        self.ssl = AutoModel.from_pretrained(backbone, layerdrop=0.0, **pin)
        self.ssl.freeze_feature_encoder()
        self.ssl.config.layerdrop = 0.0
        self.normalize = bool(getattr(AutoFeatureExtractor.from_pretrained(backbone, **pin), "do_normalize", False))
        n_layers = self.ssl.config.num_hidden_layers + 1
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))
        self.proj = nn.Linear(self.ssl.config.hidden_size, hidden)
        self.rnn = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.out = nn.Linear(2 * hidden, len(CLASSES))
        # One hook for the model's life: adding and removing one per call let concurrent calls read
        # each other's output (a crash on different lengths, mixed audio on equal ones).
        self.ssl.encoder.layers[-1].register_forward_hook(_keep_last)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            wave = (wave - wave.mean(dim=1, keepdim=True)) / (wave.std(dim=1, keepdim=True) + 1e-5)
        # The last hidden state must be the last layer's raw output: that is what current transformers
        # (where the models were trained, on Kaggle) returns, while older versions return it after
        # the final layer norm, a 160x smaller tensor. Taken from a hook, it is the same either way.
        _last.out = None
        hidden = list(self.ssl(wave, output_hidden_states=True).hidden_states)
        hidden[-1] = _last.out
        w = torch.softmax(self.layer_weights, dim=0)
        # strict: a transformers version returning a different number of layers must fail loudly,
        # not silently drop a layer weight
        x = sum(wi * h for wi, h in zip(w, hidden, strict=True))
        x, _ = self.rnn(F.gelu(self.proj(x)))
        return self.out(x)  # [batch, frames, classes]

    def param_groups(self, lr_backbone: float, lr_head: float):
        backbone = [p for p in self.ssl.parameters() if p.requires_grad]
        head = [p for n, p in self.named_parameters() if not n.startswith("ssl.")]
        return [{"params": backbone, "lr": lr_backbone}, {"params": head, "lr": lr_head}]


def frame_loss(logits, labels, mask, pos_weight: float = 3.0, class_mask=None):
    """class_mask [B, C]: which classes each example's labels actually cover (silver labels from a
    transcript say nothing about prolongations or blocks, so those must not be trained as absent)."""
    t = min(logits.shape[1], labels.shape[1])
    logits, labels, mask = logits[:, :t].float(), labels[:, :t], mask[:, :t]
    pw = torch.full((logits.shape[-1],), pos_weight, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw, reduction="none")
    cm = torch.ones_like(loss[:, 0]) if class_mask is None else class_mask.to(loss.dtype)
    per_frame = (loss * cm[:, None, :]).sum(-1) / cm.sum(-1, keepdim=True).clamp(min=1.0)
    w = mask * (cm.sum(-1, keepdim=True) > 0)
    return (per_frame * w).sum() / w.sum().clamp(min=1.0)


def clip_logits(frame_logits, r: float = 5.0):
    """Log-sum-exp pooling: a smooth max over frames, for clip-level (weak) labels."""
    t = frame_logits.shape[1]
    return (torch.logsumexp(r * frame_logits, dim=1) - torch.log(torch.tensor(float(t)))) / r


def clip_loss(frame_logits, clip_labels):
    return F.binary_cross_entropy_with_logits(clip_logits(frame_logits), clip_labels)
