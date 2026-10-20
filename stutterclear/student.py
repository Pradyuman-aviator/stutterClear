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
        x, _ = self.trunk(self.logmel(wave))
        return [head(x) for head in self.heads]


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

    def forward(self, mel_pair: torch.Tensor, buf1: torch.Tensor, buf2: torch.Tensor, h: torch.Tensor):
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
    output_path,
    quantize_int8: bool = True,
    export_step: bool = True,
    opset_version: int = 14,
):
    """Exports a CausalStudent model to ONNX with optional dynamic INT8 quantization.

    Returns dict mapping:
      'sequence_fp32': Path to sequence model (mel -> logits [B, T, heads, classes])
      'sequence_int8': Path to dynamic INT8 quantized sequence model
      'step_fp32': Path to streaming step model (if export_step)
      'step_int8': Path to dynamic INT8 streaming step model (if export_step)
      'meta': Path to JSON metadata
    """
    import json
    from pathlib import Path

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

    # 2. Metadata
    meta_path = output_path.parent / f"{stem}.json"
    meta = {
        "arch": model.arch,
        "lookaheads": list(model.lookaheads),
        "classes": list(CLASSES),
        "n_mels": N_MELS,
        "sample_rate": SR,
        "k1": model.conv1.conv.kernel_size[0],
        "k2": model.conv2.conv.kernel_size[0],
        "conv_channels": model.conv1.conv.out_channels,
        "hidden": model.arch.get("hidden", 160),
        "layers": model.arch.get("layers", 2),
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
        except Exception:
            pass

    return result


class ONNXStudent:
    """Drop-in inference engine for CausalStudent powered by ONNX Runtime with INT8 quantization."""

    def __init__(
        self,
        model_path,
        lookaheads: tuple = LOOKAHEADS,
        arch: dict | None = None,
        use_step_session: bool = True,
    ):
        import json
        from pathlib import Path
        import onnxruntime as ort

        self.model_path = Path(model_path)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )

        stem = self.model_path.stem.replace(".quant", "")
        meta_path = self.model_path.parent / f"{stem}.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                self.lookaheads = tuple(meta.get("lookaheads", lookaheads))
                self.arch = meta.get("arch", arch or {})
                self.k1 = meta.get("k1", 5)
                self.k2 = meta.get("k2", 3)
                self.conv_channels = meta.get("conv_channels", self.arch.get("conv", 128))
                self.hidden = meta.get("hidden", self.arch.get("hidden", 160))
                self.layers = meta.get("layers", self.arch.get("layers", 2))
            except Exception:
                self.lookaheads = tuple(lookaheads)
                self.arch = arch or {}
                self.k1, self.k2, self.conv_channels = 5, 3, 128
                self.hidden, self.layers = 160, 2
        else:
            self.lookaheads = tuple(lookaheads)
            self.arch = arch or {}
            self.k1, self.k2, self.conv_channels = 5, 3, 128
            self.hidden, self.layers = 160, 2

        self.step_session = None
        if use_step_session:
            is_quant = ".quant" in self.model_path.name
            step_cand = (
                self.model_path.parent / f"{stem}_step.quant.onnx"
                if is_quant
                else self.model_path.parent / f"{stem}_step.onnx"
            )
            if not step_cand.exists() and is_quant:
                step_cand = self.model_path.parent / f"{stem}_step.onnx"
            if step_cand.exists():
                try:
                    self.step_session = ort.InferenceSession(
                        str(step_cand), sess_options=opts, providers=["CPUExecutionProvider"]
                    )
                except Exception:
                    pass

        self.logmel = LogMel()

    @classmethod
    def from_checkpoint(cls, pt_path, quantize_int8: bool = True):
        from pathlib import Path

        pt_path = Path(pt_path)
        stem = pt_path.stem
        quant_path = pt_path.parent / f"{stem}.quant.onnx"
        fp32_path = pt_path.parent / f"{stem}.onnx"

        if quantize_int8 and quant_path.exists():
            return cls(quant_path)
        if fp32_path.exists():
            return cls(fp32_path)

        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
        model = CausalStudent(lookaheads=ckpt["lookaheads"], **ckpt.get("arch", {}))
        model.load_state_dict(ckpt["state_dict"])
        paths = export_onnx(model, fp32_path, quantize_int8=quantize_int8, export_step=True)
        chosen = paths.get("sequence_int8") if quantize_int8 and "sequence_int8" in paths else paths["sequence_fp32"]
        return cls(chosen, lookaheads=model.lookaheads, arch=model.arch)

    def step(self, mel_pair: np.ndarray, buf1: np.ndarray, buf2: np.ndarray, h: np.ndarray):
        """Runs one 20 ms streaming step via ONNX Runtime."""
        if self.step_session is None:
            raise RuntimeError("Step session not available in ONNXStudent")
        out = self.step_session.run(None, {
            "mel_pair": mel_pair,
            "buf1": buf1,
            "buf2": buf2,
            "h": h,
        })
        return out[0], out[1], out[2], out[3]

    def forward_mel(self, mel):
        if isinstance(mel, torch.Tensor):
            mel = mel.detach().cpu().numpy()
        mel = np.asarray(mel, dtype=np.float32)
        if mel.ndim == 2:
            mel = mel[np.newaxis, ...]
        return self.session.run(None, {"mel": mel})[0]

    def forward(self, wave) -> list:
        if isinstance(wave, np.ndarray):
            wave = torch.from_numpy(wave)
        if wave.ndim == 1:
            wave = wave.unsqueeze(0)
        with torch.no_grad():
            mel = self.logmel(wave)
        logits_np = self.forward_mel(mel)  # [B, T, heads, classes]
        logits_t = torch.from_numpy(logits_np)
        return [logits_t[:, :, i, :] for i in range(len(self.lookaheads))]

    def predict_probs(self, wave_or_mel) -> np.ndarray:
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

