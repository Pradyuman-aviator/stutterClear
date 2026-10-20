"""Export CausalStudent model checkpoints to ONNX with INT8 dynamic quantization."""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stutterclear.student import CausalStudent, N_MELS
from stutterclear.student_onnx import ONNXStudent, export_onnx


def benchmark(fn, name: str, n_iters: int = 200) -> float:
    for _ in range(10):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    avg_ms = (time.perf_counter() - t0) / n_iters * 1000
    print(f"  {name:<24}: {avg_ms:6.3f} ms")
    return avg_ms


def main():
    parser = argparse.ArgumentParser(description="Export student detector to ONNX / INT8")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--out", type=str, default=None, help="Output onnx path (defaults to beside checkpoint)")
    parser.add_argument("--no-int8", action="store_true", help="Skip INT8 dynamic quantization")
    parser.add_argument("--benchmark", action="store_true", default=True, help="Run latency benchmark")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_path = Path(args.out) if args.out else ckpt_path.with_suffix(".onnx")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = CausalStudent(lookaheads=ckpt["lookaheads"], **ckpt.get("arch", {}))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print(f"Exporting to ONNX: {out_path}")
    paths = export_onnx(model, out_path, quantize_int8=not args.no_int8, export_step=True)
    print("\nExported artifacts:")
    for key, path in paths.items():
        size_kb = os.path.getsize(path) / 1024
        print(f"  {key:<16}: {path.name} ({size_kb:.1f} KB)")

    if "sequence_int8" in paths and "sequence_fp32" in paths:
        fp32_size = os.path.getsize(paths["sequence_fp32"])
        int8_size = os.path.getsize(paths["sequence_int8"])
        reduction = (1 - int8_size / fp32_size) * 100
        print(f"\nSize reduction: {reduction:.1f}%")

    if args.benchmark:
        print("\n1. Sequence benchmark on CPU (100 frames = 1 s of audio):")
        dummy_mel = np.random.randn(1, 100, N_MELS).astype(np.float32)
        dummy_mel_t = torch.from_numpy(dummy_mel)

        def run_torch_seq():
            with torch.no_grad():
                x, _ = model.trunk(dummy_mel_t)
                _ = [head(x) for head in model.heads]

        t_pt_seq = benchmark(run_torch_seq, "PyTorch CPU")

        onnx_fp32 = ONNXStudent(paths["sequence_fp32"])
        t_fp32_seq = benchmark(lambda: onnx_fp32.forward_mel(dummy_mel), "ONNX FP32")

        if "sequence_int8" in paths:
            onnx_int8 = ONNXStudent(paths["sequence_int8"])
            t_int8_seq = benchmark(lambda: onnx_int8.forward_mel(dummy_mel), "ONNX INT8")
            print(f"  Speedup vs PyTorch    : {t_pt_seq / t_int8_seq:.2f}x")
        else:
            print(f"  Speedup vs PyTorch    : {t_pt_seq / t_fp32_seq:.2f}x")

        if "step_fp32" in paths:
            print("\n2. Live streaming 20 ms step benchmark on CPU (2 mel frames):")
            k1 = model.conv1.conv.kernel_size[0]
            k2 = model.conv2.conv.kernel_size[0]
            conv_ch = model.conv1.conv.out_channels
            hidden = model.arch.get("hidden", 160)
            layers = model.arch.get("layers", 2)

            pt_buf1 = torch.zeros(1, N_MELS, k1)
            pt_buf2 = torch.zeros(1, conv_ch, k2)
            pt_h = None
            pt_mel_pair = torch.randn(2, N_MELS)

            def run_torch_step():
                nonlocal pt_buf1, pt_buf2, pt_h
                with torch.no_grad():
                    v0 = model.norm((pt_mel_pair[0:1] + 40.0) / 20.0)
                    pt_buf1 = torch.cat([pt_buf1[..., 1:], v0.view(1, -1, 1)], dim=-1)
                    v1 = model.norm((pt_mel_pair[1:2] + 40.0) / 20.0)
                    pt_buf1 = torch.cat([pt_buf1[..., 1:], v1.view(1, -1, 1)], dim=-1)
                    c1 = F.gelu(model.conv1.conv(pt_buf1))
                    pt_buf2 = torch.cat([pt_buf2[..., 1:], c1], dim=-1)
                    c2 = F.gelu(model.conv2.conv(pt_buf2))
                    y, pt_h = model.gru(c2.transpose(1, 2), pt_h)
                    _ = [torch.sigmoid(hd(y[0, 0])) for hd in model.heads]

            t_pt_step = benchmark(run_torch_step, "PyTorch 20ms step")

            onnx_mel_pair = pt_mel_pair.numpy()
            b1 = np.zeros((1, N_MELS, k1), dtype=np.float32)
            b2 = np.zeros((1, conv_ch, k2), dtype=np.float32)
            h_state = np.zeros((layers, 1, hidden), dtype=np.float32)

            def run_onnx_step_fp32():
                nonlocal b1, b2, h_state
                _, b1, b2, h_state = onnx_fp32.step(onnx_mel_pair, b1, b2, h_state)

            t_fp32_step = benchmark(run_onnx_step_fp32, "ONNX FP32 20ms step")

            if "step_int8" in paths:
                b1_int8 = np.zeros((1, N_MELS, k1), dtype=np.float32)
                b2_int8 = np.zeros((1, conv_ch, k2), dtype=np.float32)
                h_int8 = np.zeros((layers, 1, hidden), dtype=np.float32)

                def run_onnx_step_int8():
                    nonlocal b1_int8, b2_int8, h_int8
                    _, b1_int8, b2_int8, h_int8 = onnx_int8.step(onnx_mel_pair, b1_int8, b2_int8, h_int8)

                t_int8_step = benchmark(run_onnx_step_int8, "ONNX INT8 20ms step")
                print(f"  Live Step Speedup     : {t_pt_step / t_int8_step:.2f}x")
            else:
                print(f"  Live Step Speedup     : {t_pt_step / t_fp32_step:.2f}x")


if __name__ == "__main__":
    main()
