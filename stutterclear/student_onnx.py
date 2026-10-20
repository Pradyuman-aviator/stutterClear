"""ONNX export, INT8 dynamic quantization, and ONNX Runtime inference for CausalStudent."""
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .student import CausalStudent, LogMel, N_MELS, LOOKAHEADS
from .synth import CLASSES, SR

logger = logging.getLogger(__name__)


class StudentMelWrapper(nn.Module):
    """Wraps CausalStudent to process log-mel spectrogram features [B, F, N_MELS]."""

    def __init__(self, student: CausalStudent):
        super().__init__()
        self.student = student

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x, _ = self.student.trunk(mel)
        # Returns stacked logits [B, T, num_heads, classes]
        return torch.stack([head(x) for head in self.student.heads], dim=2)


class StudentStepWrapper(nn.Module):
    """Wraps a single 20 ms (2-mel-frame) streaming step for CausalStudent."""

    def __init__(self, student: CausalStudent):
        super().__init__()
        self.norm = student.norm
        self.conv1 = student.conv1.conv
        self.conv2 = student.conv2.conv
        self.gru = student.gru
        self.heads = student.heads

    def forward(
        self,
        mel_pair: torch.Tensor,
        buf1: torch.Tensor,
        buf2: torch.Tensor,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # mel_pair: [2, 80]
        v0 = self.norm((mel_pair[0:1] + 40.0) / 20.0).unsqueeze(-1)
        v1 = self.norm((mel_pair[1:2] + 40.0) / 20.0).unsqueeze(-1)
        b1_mid = torch.cat([buf1[:, :, 1:], v0], dim=-1)
        b1_full = torch.cat([b1_mid[:, :, 1:], v1], dim=-1)
        c1 = F.gelu(self.conv1(b1_full))
        b2_full = torch.cat([buf2[:, :, 1:], c1], dim=-1)
        c2 = F.gelu(self.conv2(b2_full))
        y, new_h = self.gru(c2.transpose(1, 2), h)
        logits = torch.stack([hd(y[0, 0]) for hd in self.heads], dim=0)
        probs = torch.sigmoid(logits)
        return probs, b1_full, b2_full, new_h


def export_onnx(
    model: CausalStudent,
    output_path: str | Path,
    quantize_int8: bool = True,
    export_step: bool = True,
    opset_version: int = 14,
) -> dict[str, Path]:
    """Exports a CausalStudent model to ONNX with optional dynamic INT8 quantization.

    Returns dict mapping:
      'sequence_fp32': Path to sequence model (mel -> logits [B, T, heads, classes])
      'sequence_int8': Path to dynamic INT8 quantized sequence model
      'step_fp32': Path to streaming step model (if export_step)
      'step_int8': Path to dynamic INT8 streaming step model (if export_step)
      'meta': Path to JSON metadata
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.stem.replace(".quant", "")

    model.eval()

    # 1. Export sequence model
    seq_fp32 = output_path.parent / f"{stem}.onnx"
    wrapper = StudentMelWrapper(model)
    dummy_mel = torch.zeros(1, 100, N_MELS, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy_mel,
        str(seq_fp32),
        input_names=["mel"],
        output_names=["logits"],
        dynamic_axes={"mel": {0: "batch", 1: "frames"}, "logits": {0: "batch", 1: "frames"}},
        opset_version=opset_version,
        dynamo=False,
    )

    # 2. Metadata (required for runtime to avoid guessing model dimensions)
    meta_path = output_path.parent / f"{stem}.json"
    meta = {
        "arch": model.arch,
        "lookaheads": list(model.lookaheads),
        "classes": list(CLASSES),
        "n_mels": N_MELS,
        "sample_rate": SR,
        "k1": int(model.conv1.conv.kernel_size[0]),
        "k2": int(model.conv2.conv.kernel_size[0]),
        "conv_channels": int(model.conv1.conv.out_channels),
        "hidden": int(model.arch.get("hidden", 160)),
        "layers": int(model.arch.get("layers", 2)),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    result = {"sequence_fp32": seq_fp32, "meta": meta_path}

    # 3. Export step model if requested
    if export_step:
        step_fp32 = output_path.parent / f"{stem}_step.onnx"
        step_wrapper = StudentStepWrapper(model)
        dummy_mel_pair = torch.zeros(2, N_MELS, dtype=torch.float32)
        dummy_b1 = torch.zeros(1, N_MELS, meta["k1"], dtype=torch.float32)
        dummy_b2 = torch.zeros(1, meta["conv_channels"], meta["k2"], dtype=torch.float32)
        dummy_h = torch.zeros(meta["layers"], 1, meta["hidden"], dtype=torch.float32)

        torch.onnx.export(
            step_wrapper,
            (dummy_mel_pair, dummy_b1, dummy_b2, dummy_h),
            str(step_fp32),
            input_names=["mel_pair", "buf1", "buf2", "h"],
            output_names=["probs", "new_buf1", "new_buf2", "new_h"],
            opset_version=opset_version,
            dynamo=False,
        )
        result["step_fp32"] = step_fp32

    # 4. Dynamic INT8 quantization
    if quantize_int8:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            seq_int8 = output_path.parent / f"{stem}.quant.onnx"
            quantize_dynamic(
                str(seq_fp32),
                str(seq_int8),
                weight_type=QuantType.QInt8,
                op_types_to_quantize=["MatMul", "Gather"],
            )
            result["sequence_int8"] = seq_int8

            if export_step and "step_fp32" in result:
                step_int8 = output_path.parent / f"{stem}_step.quant.onnx"
                quantize_dynamic(
                    str(result["step_fp32"]),
                    str(step_int8),
                    weight_type=QuantType.QInt8,
                    op_types_to_quantize=["MatMul", "Gather"],
                )
                result["step_int8"] = step_int8
        except Exception as e:
            raise RuntimeError(f"Dynamic INT8 quantization failed: {e}") from e

    return result


def find_exported_student(
    path: str | Path, prefer_int8: bool = True, require_step: bool = False
) -> Optional[Tuple[Path, Optional[Path], Path]]:
    """Locates pre-exported ONNX files for a given student model path without writing to disk.

    Returns (seq_model_path, step_model_path_or_None, meta_path) or None if files don't exist.
    If require_step is True, returns None unless a step model file also exists.
    """
    path = Path(path)
    parent = path.parent
    stem = path.stem.replace(".quant", "").replace("_step", "")

    meta_path = parent / f"{stem}.json"
    if not meta_path.exists():
        return None

    # Check sequence model
    quant_seq = parent / f"{stem}.quant.onnx"
    fp32_seq = parent / f"{stem}.onnx"

    seq_path = None
    if prefer_int8 and quant_seq.exists():
        seq_path = quant_seq
    elif fp32_seq.exists():
        seq_path = fp32_seq
    elif quant_seq.exists():
        seq_path = quant_seq

    if seq_path is None:
        return None

    # Check step model (optional for sequence evaluation, required for live streaming)
    is_quant = ".quant" in seq_path.name
    step_path = None
    if is_quant:
        cand = parent / f"{stem}_step.quant.onnx"
        if cand.exists():
            step_path = cand
    if step_path is None:
        cand = parent / f"{stem}_step.onnx"
        if cand.exists():
            step_path = cand

    if require_step and step_path is None:
        return None

    return seq_path, step_path, meta_path


def onnx_available_for_path(
    path: str | Path, prefer_int8: bool = True, require_step: bool = True
) -> bool:
    """Returns True if onnxruntime is installed and exported ONNX model files exist for path.

    When require_step=True (the default for live streaming sessions), also requires
    the streaming step model file.
    """
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False

    return find_exported_student(path, prefer_int8=prefer_int8, require_step=require_step) is not None


class ONNXStudent:
    """Drop-in inference engine for CausalStudent powered by ONNX Runtime."""

    def __init__(
        self,
        model_path: str | Path,
        lookaheads: tuple[int, ...] = LOOKAHEADS,
        arch: dict | None = None,
        step_model_path: Optional[str | Path] = None,
        meta_path: Optional[str | Path] = None,
    ):
        import onnxruntime as ort

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX student model not found: {self.model_path}")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )

        stem = self.model_path.stem.replace(".quant", "").replace("_step", "")
        resolved_meta = Path(meta_path) if meta_path else self.model_path.parent / f"{stem}.json"

        if resolved_meta.exists():
            try:
                meta = json.loads(resolved_meta.read_text(encoding="utf-8"))
            except Exception as e:
                raise ValueError(f"Failed to read metadata from {resolved_meta}: {e}") from e

            self.lookaheads = tuple(meta["lookaheads"])
            self.arch = meta["arch"]
            self.k1 = meta["k1"]
            self.k2 = meta["k2"]
            self.conv_channels = meta["conv_channels"]
            self.hidden = meta["hidden"]
            self.layers = meta["layers"]
        elif arch is not None:
            self.lookaheads = tuple(lookaheads)
            self.arch = arch
            self.k1 = 5
            self.k2 = 3
            self.conv_channels = arch.get("conv", 128)
            self.hidden = arch.get("hidden", 160)
            self.layers = arch.get("layers", 2)
        else:
            raise FileNotFoundError(
                f"Metadata file '{resolved_meta}' not found for ONNX model. "
                "Models must be exported using scripts/export_student_onnx.py to preserve architecture dimensions."
            )

        # Load step session for live streaming
        self.step_path = None
        self.step_session = None

        if step_model_path is not None:
            resolved_step = Path(step_model_path)
            if resolved_step.exists():
                self.step_path = resolved_step
        else:
            is_quant = ".quant" in self.model_path.name
            if is_quant:
                cand = self.model_path.parent / f"{stem}_step.quant.onnx"
                if cand.exists():
                    self.step_path = cand
            if self.step_path is None:
                cand = self.model_path.parent / f"{stem}_step.onnx"
                if cand.exists():
                    self.step_path = cand

        if self.step_path is not None:
            self.step_session = ort.InferenceSession(
                str(self.step_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )

        self.logmel = LogMel()

    @classmethod
    def from_checkpoint(cls, pt_path: str | Path, quantize_int8: bool = True) -> "ONNXStudent":
        """Loads pre-exported ONNX models for a PyTorch checkpoint. Never writes to disk."""
        found = find_exported_student(pt_path, prefer_int8=quantize_int8)
        if found is None:
            raise FileNotFoundError(
                f"No pre-exported ONNX models found for {pt_path}. "
                "Export them first using scripts/export_student_onnx.py."
            )
        seq_path, step_path, meta_path = found
        return cls(seq_path, step_model_path=step_path, meta_path=meta_path)

    def step(self, mel_pair: np.ndarray, buf1: np.ndarray, buf2: np.ndarray, h: np.ndarray):
        """Runs one 20 ms streaming step via ONNX Runtime."""
        if self.step_session is None:
            stem = self.model_path.stem.replace(".quant", "")
            raise RuntimeError(
                f"ONNXStudent has no step session loaded ({stem}_step.onnx missing). "
                "Live streaming requires the step model. Export using scripts/export_student_onnx.py."
            )
        out = self.step_session.run(
            None,
            {
                "mel_pair": mel_pair,
                "buf1": buf1,
                "buf2": buf2,
                "h": h,
            },
        )
        return out[0], out[1], out[2], out[3]

    def forward_mel(self, mel: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(mel, torch.Tensor):
            mel = mel.detach().cpu().numpy()
        mel = np.asarray(mel, dtype=np.float32)
        if mel.ndim == 2:
            mel = mel[np.newaxis, ...]
        return self.session.run(None, {"mel": mel})[0]

    def forward(self, wave: torch.Tensor | np.ndarray) -> list[torch.Tensor]:
        if isinstance(wave, np.ndarray):
            wave = torch.from_numpy(wave)
        if wave.ndim == 1:
            wave = wave.unsqueeze(0)
        with torch.no_grad():
            mel = self.logmel(wave)
        logits_np = self.forward_mel(mel)  # [B, T, heads, classes]
        logits_t = torch.from_numpy(logits_np)
        return [logits_t[:, :, i, :] for i in range(len(self.lookaheads))]

    def predict_probs(self, wave_or_mel: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(wave_or_mel, torch.Tensor):
            if wave_or_mel.ndim == 2 and wave_or_mel.shape[-1] == N_MELS:
                logits = self.forward_mel(wave_or_mel)
            else:
                with torch.no_grad():
                    mel = self.logmel(wave_or_mel)
                logits = self.forward_mel(mel)
        elif isinstance(wave_or_mel, np.ndarray):
            if wave_or_mel.ndim >= 2 and wave_or_mel.shape[-1] == N_MELS:
                logits = self.forward_mel(wave_or_mel)
            else:
                wave_t = torch.from_numpy(wave_or_mel)
                if wave_t.ndim == 1:
                    wave_t = wave_t.unsqueeze(0)
                with torch.no_grad():
                    mel = self.logmel(wave_t)
                logits = self.forward_mel(mel)
        else:
            raise TypeError(f"Unsupported input type: {type(wave_or_mel)}")
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
