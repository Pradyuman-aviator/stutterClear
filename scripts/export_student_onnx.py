import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stutterclear.student import CausalStudent, ONNXStudent, export_onnx, N_MELS


def benchmark(fn, name: str, n_iters: int = 50) -> float:
    # Warmup
    for _ in range(5):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    avg_ms = (time.perf_counter() - t0) / n_iters * 1000
    print(f"  {name:<22}: {avg_ms:6.2f} ms")
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
        print("\nBenchmarking latency on CPU (100 frames = 1 s of audio):")
        dummy_mel = np.random.randn(1, 100, N_MELS).astype(np.float32)
        dummy_mel_t = torch.from_numpy(dummy_mel)

        def run_torch():
            with torch.no_grad():
                x, _ = model.trunk(dummy_mel_t)
                _ = [head(x) for head in model.heads]

        t_pt = benchmark(run_torch, "PyTorch CPU")

        onnx_fp32 = ONNXStudent(paths["sequence_fp32"], use_step_session=False)
        t_fp32 = benchmark(lambda: onnx_fp32.forward_mel(dummy_mel), "ONNX FP32")

        if "sequence_int8" in paths:
            onnx_int8 = ONNXStudent(paths["sequence_int8"], use_step_session=False)
            t_int8 = benchmark(lambda: onnx_int8.forward_mel(dummy_mel), "ONNX INT8")
            print(f"  Speedup vs PyTorch    : {t_pt / t_int8:.2f}x")
        else:
            print(f"  Speedup vs PyTorch    : {t_pt / t_fp32:.2f}x")


if __name__ == "__main__":
    main()
