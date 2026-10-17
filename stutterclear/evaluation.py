"""Shared evaluation: clip-level predictions on SEP-28k, threshold tuning, F1 and AUC."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from .synth import CLASSES

MAJORITY = 2 / 3 - 1e-6  # at least 2 of 3 annotators


def clip_scores(frame_logits: torch.Tensor, r: float = 5.0) -> torch.Tensor:
    """Log-sum-exp pooled clip probability from frame logits [B, T, C]."""
    t = frame_logits.shape[1]
    return torch.sigmoid((torch.logsumexp(r * frame_logits, 1) - np.log(t)) / r)


@torch.no_grad()
def predict_clips(fn, dataset, dev, batch: int = 16):
    """fn(wave) -> list of frame logits (one per head). Returns ([heads][N, C] probs, [N, C] votes/3)."""
    probs, targets = None, []
    for b in DataLoader(dataset, batch_size=batch, num_workers=0):
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            outs = fn(b["wave"].to(dev))
        p = [clip_scores(o.float()).cpu().numpy() for o in outs]
        probs = [[x] for x in p] if probs is None else [acc + [x] for acc, x in zip(probs, p)]
        targets.append(b["clip_labels"].numpy())
    return [np.concatenate(p) for p in probs], np.concatenate(targets)


def f1_score(pred: np.ndarray, truth: np.ndarray) -> float:
    tp = (pred & truth).sum()
    return float(2 * tp / max(1, 2 * tp + (pred & ~truth).sum() + (~pred & truth).sum()))


def tune_thresholds(probs: np.ndarray, votes: np.ndarray) -> list:
    truth = votes >= MAJORITY
    grid = np.arange(0.02, 0.99, 0.01)
    return [float(grid[np.argmax([f1_score(probs[:, k] > g, truth[:, k]) for g in grid])])
            for k in range(probs.shape[1])]


def report(probs: np.ndarray, votes: np.ndarray, thresholds=None) -> dict:
    truth = votes >= MAJORITY
    th = thresholds or [0.5] * probs.shape[1]
    out = {"f1": {c: round(f1_score(probs[:, k] > th[k], truth[:, k]), 3) for k, c in enumerate(CLASSES)}}
    out["macro_f1"] = round(float(np.mean(list(out["f1"].values()))), 3)
    try:
        from sklearn.metrics import roc_auc_score
        out["auc"] = {c: round(float(roc_auc_score(truth[:, k], probs[:, k])), 3)
                      for k, c in enumerate(CLASSES) if 0 < truth[:, k].sum() < len(truth)}
    except ImportError:
        pass
    out["positives"] = {c: int(truth[:, k].sum()) for k, c in enumerate(CLASSES)}
    out["n_clips"] = int(len(truth))
    return out
