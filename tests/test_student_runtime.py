"""StudentDetector runs on a tiny untrained student, so these tests need no downloaded checkpoint."""
import numpy as np
import torch

from stutterclear.student import CausalStudent
from stutterclear.student_runtime import LIVE_CUTS, StudentDetector


def detector(tmp_path):
    model = CausalStudent(hidden=16, conv=16, layers=1)
    path = tmp_path / "student.pt"
    torch.save({"state_dict": model.state_dict(), "lookaheads": model.lookaheads, "arch": model.arch}, path)
    return StudentDetector(path, 200, cut_cfg=LIVE_CUTS)


def test_block_is_reported_not_cut(tmp_path):
    det = detector(tmp_path)
    on = np.zeros(5)
    on[det.block_idx] = 0.99
    for f in range(10):
        det._block_frame(f, on)
    det._block_frame(10, np.zeros(5))
    assert [(e.kind, e.start, e.end) for e in det.events] == [("block", 0, 10 * 320)]


def test_short_block_is_ignored(tmp_path):
    det = detector(tmp_path)
    on = np.zeros(5)
    on[det.block_idx] = 0.99
    det._block_frame(0, on)
    det._block_frame(1, np.zeros(5))
    assert det.events == []


def test_streams_audio_without_errors(tmp_path):
    det = detector(tmp_path)
    x = (np.random.default_rng(0).standard_normal(16000 * 2) * 0.1).astype(np.float32)
    for i in range(0, len(x), 160):
        list(det.process(x[i:i + 160]))
