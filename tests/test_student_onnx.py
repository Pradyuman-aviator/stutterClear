"""Unit tests for student model ONNX export, INT8 quantization, and ONNXStudent runtime."""
import numpy as np
import pytest
import torch

from stutterclear.student import CausalStudent, ONNXStudent, export_onnx, N_MELS
from stutterclear.student_runtime import StudentDetector, StudentStream, load_student, LIVE_CUTS
from stutterclear.synth import CLASSES


@pytest.fixture
def tiny_student_checkpoint(tmp_path):
    torch.manual_seed(42)
    model = CausalStudent(hidden=16, conv=16, layers=1)
    path = tmp_path / "student.pt"
    torch.save(
        {"state_dict": model.state_dict(), "lookaheads": model.lookaheads, "arch": model.arch},
        path,
    )
    return path, model


def test_export_onnx_and_int8(tiny_student_checkpoint, tmp_path):
    pt_path, model = tiny_student_checkpoint
    out_dir = tmp_path / "onnx_export"
    paths = export_onnx(model, out_dir / "student.onnx", quantize_int8=True, export_step=True)

    assert paths["sequence_fp32"].exists()
    assert paths["sequence_fp32"].stat().st_size > 0
    assert paths["sequence_int8"].exists()
    assert paths["sequence_int8"].stat().st_size > 0
    assert paths["step_fp32"].exists()
    assert paths["step_int8"].exists()
    assert paths["meta"].exists()


def test_onnx_student_parity(tiny_student_checkpoint, tmp_path):
    pt_path, model = tiny_student_checkpoint
    paths = export_onnx(model, tmp_path / "model.onnx", quantize_int8=True, export_step=True)

    onnx_fp32 = ONNXStudent(paths["sequence_fp32"])
    onnx_int8 = ONNXStudent(paths["sequence_int8"])

    dummy_mel = torch.randn(1, 40, N_MELS)

    # PyTorch baseline
    with torch.no_grad():
        x, _ = model.trunk(dummy_mel)
        pt_logits = torch.stack([hd(x) for hd in model.heads], dim=2).numpy()

    # ONNX FP32
    fp32_logits = onnx_fp32.forward_mel(dummy_mel)
    max_diff_fp32 = np.max(np.abs(pt_logits - fp32_logits))
    assert max_diff_fp32 < 1e-4

    # ONNX INT8
    int8_logits = onnx_int8.forward_mel(dummy_mel)
    max_diff_int8 = np.max(np.abs(pt_logits - int8_logits))
    assert max_diff_int8 < 0.15


def test_onnx_student_stream(tiny_student_checkpoint):
    pt_path, _ = tiny_student_checkpoint
    onnx_model = load_student(pt_path, use_onnx=True, quantize_int8=True)
    assert isinstance(onnx_model, ONNXStudent)

    stream = StudentStream(onnx_model)
    assert stream.is_onnx is True

    x = (np.random.default_rng(0).standard_normal(16000) * 0.1).astype(np.float32)
    total_frames = 0
    for i in range(0, len(x), 320):  # 20 ms chunks = 2 mel frames each
        first, probs = stream.push(x[i : i + 320])
        total_frames += len(probs)
        if len(probs) > 0:
            assert probs.shape[1] == len(onnx_model.lookaheads)
            assert probs.shape[2] == len(CLASSES)
            assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    assert total_frames > 0


def test_student_detector_with_onnx(tiny_student_checkpoint):
    pt_path, _ = tiny_student_checkpoint
    det = StudentDetector(pt_path, lookahead_ms=200, cut_cfg=LIVE_CUTS, use_onnx=True)
    assert isinstance(det.model, ONNXStudent)

    x = (np.random.default_rng(1).standard_normal(16000) * 0.1).astype(np.float32)
    updates_count = 0
    for i in range(0, len(x), 160):
        updates = list(det.process(x[i : i + 160]))
        updates_count += len(updates)

    assert det.decided_samples() > 0
