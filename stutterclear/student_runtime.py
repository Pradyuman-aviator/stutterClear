"""Live use of the trained student.

StudentStream runs the causal student one 10 ms feature frame at a time, carrying conv buffers
and GRU state, and matches the whole-clip forward pass exactly.
StudentDetector turns its per-frame probabilities into cut updates for the splicer, with the
same process() interface as HeuristicDetector.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import SAMPLE_RATE
from .detector import CutUpdate, DetectorEvent
from .features import FeatureStream
from .segments import BALANCED_CUTS, CutConfig
from .student import N_MELS, CausalStudent
from .synth import CLASSES, FRAME

REMOVABLE = ("sound_rep", "word_rep", "prolongation")


def load_student(path) -> CausalStudent:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = CausalStudent(lookaheads=ckpt["lookaheads"], **ckpt.get("arch", {}))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


class StudentStream:
    def __init__(self, model: CausalStudent):
        self.m = model
        self.features = FeatureStream(n_mels=N_MELS)
        k1, k2 = model.conv1.conv.kernel_size[0], model.conv2.conv.kernel_size[0]
        self.buf1 = torch.zeros(1, N_MELS, k1)  # zeros = the causal left padding
        self.buf2 = torch.zeros(1, model.conv1.conv.out_channels, k2)
        self.h = None
        self.n_feat = 0
        self.n_out = 0

    @torch.no_grad()
    def push(self, x: np.ndarray):
        """Returns (index of first new output frame, [n, heads, classes] probabilities)."""
        first = self.n_out
        _, _, logmel, _ = self.features.push(x)
        rows = []
        for row in logmel:
            v = self.m.norm(torch.from_numpy((row.astype(np.float32) + 40.0) / 20.0))
            self.buf1 = torch.cat([self.buf1[..., 1:], v.view(1, -1, 1)], dim=-1)
            if self.n_feat % 2 == 1:  # stride 2: one output per two 10 ms frames
                c1 = F.gelu(self.m.conv1.conv(self.buf1))
                self.buf2 = torch.cat([self.buf2[..., 1:], c1], dim=-1)
                c2 = F.gelu(self.m.conv2.conv(self.buf2))
                y, self.h = self.m.gru(c2.transpose(1, 2), self.h)
                rows.append(torch.stack([torch.sigmoid(hd(y[0, 0])) for hd in self.m.heads]))
                self.n_out += 1
            self.n_feat += 1
        probs = torch.stack(rows).numpy() if rows else np.zeros((0, len(self.m.heads), len(CLASSES)))
        return first, probs


# Live cut rules, from the offline search on 7 held-out UCLASS speakers ("balanced" preset).
LIVE_CUTS = BALANCED_CUTS


class StudentDetector:
    """Online version of segments.cut_segments: hysteresis, merging of nearby cuts, strict mode
    inside long cuts, and prolongations shortened rather than deleted."""

    MAX_FRAMES = 100  # a "stutter" longer than 2 s is more likely a detector error: cancel it
    MIN_CUT_THRESHOLD = 0.2  # tuned thresholds can be tiny for rare classes; never cut on that

    def __init__(self, ckpt_path, lookahead_ms: float, report_path=None, cut_interjections: bool = False,
                 cut_cfg: CutConfig = LIVE_CUTS):
        self.model = load_student(ckpt_path)
        las = list(self.model.lookaheads)
        self.head = int(np.argmin([abs(la * 20 - lookahead_ms) for la in las]))
        self.lookahead = las[self.head]
        self.lookahead_ms = self.lookahead * 20
        self.cfg = cut_cfg
        self.thresholds = np.array(cut_cfg.thresholds, dtype=float)
        if report_path and Path(report_path).exists():
            rep = json.loads(Path(report_path).read_text()).get(f"{self.lookahead_ms}ms")
            if rep:
                self.thresholds = np.array(rep["thresholds"], dtype=float)
        removable = REMOVABLE + (("interjection",) if cut_interjections else ())
        self.cut_idx = [CLASSES.index(c) for c in removable]
        self.thresholds[self.cut_idx] = np.maximum(self.thresholds[self.cut_idx], self.MIN_CUT_THRESHOLD)
        self.close_after = max(2, cut_cfg.merge_gap)
        self.refine_frames = cut_cfg.refine_long_s * SAMPLE_RATE / FRAME
        self.keep = int(round(cut_cfg.prolong_keep_ms * SAMPLE_RATE / 1000 / FRAME))
        self.tail = int(round(cut_cfg.prolong_tail_ms * SAMPLE_RATE / 1000 / FRAME))
        self.cooldown = False  # after a cancelled over-long cut, wait for the score to drop
        self.block_idx = CLASSES.index("block")
        self.stream = StudentStream(self.model)
        self.events: list[DetectorEvent] = []
        self.next_id = 0
        self.cut = None  # open cut: dict(id, start, first, last, strict_last, below, kind, peak)
        self.block = None
        self.frames_seen = 0

    def decided_samples(self) -> int:
        """Audio up to here has been judged (frames lag the input by the lookahead)."""
        return max(0, self.frames_seen - self.lookahead + 1) * FRAME

    def process(self, x: np.ndarray) -> list[CutUpdate]:
        first, probs = self.stream.push(x)
        updates = []
        for i, p in enumerate(probs[:, self.head]):
            j = first + i
            f = j - self.lookahead  # this row describes label frame f
            self.frames_seen = j
            if f >= 0:
                self._cut_frame(f, p, updates)
                self._block_frame(f, p)
        return updates

    def _cut_frame(self, f, p, updates):
        score = p[self.cut_idx] / self.thresholds[self.cut_idx]
        best = int(np.argmax(score))
        kind = CLASSES[self.cut_idx[best]]
        if self.cut is None:
            if self.cooldown:
                self.cooldown = score[best] >= self.cfg.off_ratio
                return
            if score[best] >= 1.0:
                self.next_id += 1
                start = f + (self.keep if kind == "prolongation" else 0)  # keep the onset of a held sound
                self.cut = {"id": self.next_id, "start": start, "first": f, "last": f, "strict_last": f,
                            "below": 0, "kind": kind, "peak": float(p[self.cut_idx[best]])}
                updates.append(CutUpdate("set", self.next_id, start * FRAME, None, kind))
            return
        c = self.cut
        if f - c["first"] >= self.MAX_FRAMES:
            updates.append(CutUpdate("cancel", c["id"]))
            self.cut, self.cooldown = None, True
            return
        long_cut = f - c["first"] > self.refine_frames
        if score[best] >= 1.0:
            c["strict_last"] = f
        # Inside a long cut only confident frames extend it, so real words between stutters survive.
        if score[best] >= (1.0 if long_cut else self.cfg.off_ratio):
            c["last"], c["below"] = f, 0
            c["peak"] = max(c["peak"], float(p[self.cut_idx[best]]))
            return
        c["below"] += 1
        if c["below"] >= self.close_after:
            last = c["strict_last"] if long_cut else c["last"]
            end_frame = last + 1 - (self.tail if c["kind"] == "prolongation" else 0)
            if end_frame - c["start"] >= self.cfg.min_frames:
                updates.append(CutUpdate("set", c["id"], c["start"] * FRAME, end_frame * FRAME, c["kind"]))
                self._log(c["kind"], c["start"] * FRAME, end_frame * FRAME, {"peak": round(c["peak"], 3)})
            else:
                updates.append(CutUpdate("cancel", c["id"]))
            self.cut = None

    def _block_frame(self, f, p):
        """Blocks are reported, never cut: the time a block takes cannot be given back."""
        on = p[self.block_idx] >= self.thresholds[self.block_idx]
        if on and self.block is None:
            self.block = [f, f]
        elif on:
            self.block[1] = f
        elif self.block is not None:
            if self.block[1] - self.block[0] + 1 >= self.cfg.min_frames:
                self._log("block", self.block[0] * FRAME, (self.block[1] + 1) * FRAME, {})
            self.block = None

    def _log(self, kind, start, end, info):
        decided = self.frames_seen * FRAME + 560  # audio the student had heard when it decided
        self.events.append(DetectorEvent(kind, start, end, decided, info))
