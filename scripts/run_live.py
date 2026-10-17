"""Live laptop-mic session.

    python scripts/run_live.py --mode clean                 # no playback; saves raw + corrected audio
    python scripts/run_live.py --mode clean --out-device 5  # play corrected audio (headphones / virtual cable)
    python scripts/run_live.py --list-devices

Each session is saved to sessions/<timestamp>/ (raw.wav, out.wav, summary.json) so the
raw and corrected audio can be compared afterwards. Nothing leaves this machine.
"""
import argparse
import json
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stutterclear.audio import save_wav  # noqa: E402
from stutterclear.config import HOP, PRESETS, SAMPLE_RATE as SR, preset  # noqa: E402
from stutterclear.pipeline import Pipeline  # noqa: E402
from run_file import add_detector_args, make_detector  # noqa: E402


def wasapi():
    for i, h in enumerate(sd.query_hostapis()):
        if h["name"] == "Windows WASAPI":
            return i, h
    raise SystemExit("WASAPI not found")


def list_devices():
    idx, _ = wasapi()
    for i, d in enumerate(sd.query_devices()):
        if d["hostapi"] == idx:
            print(f"{i:3d}  in={d['max_input_channels']} out={d['max_output_channels']}  {d['name']}")


def find_device(spec: str | None, kind: str) -> int | None:
    """Device by WASAPI index or by name substring; indices shift when Bluetooth devices connect."""
    if spec is None or spec.isdigit():
        return None if spec is None else int(spec)
    idx, _ = wasapi()
    channels = "max_input_channels" if kind == "input" else "max_output_channels"
    for i, d in enumerate(sd.query_devices()):
        if d["hostapi"] == idx and d[channels] > 0 and spec.lower() in d["name"].lower():
            return i
    raise SystemExit(f"No WASAPI {kind} device matching '{spec}'. Try --list-devices.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(PRESETS), default="clean")
    ap.add_argument("--in-device", help='index or name, e.g. "Microphone Array" (default: Windows default mic)')
    ap.add_argument("--out-device", help="index or name; omit to record only (no playback)")
    ap.add_argument("--seconds", type=float, help="stop automatically after this long")
    ap.add_argument("--list-devices", action="store_true")
    add_detector_args(ap)
    args = ap.parse_args()
    if args.list_devices:
        return list_devices()

    _, host = wasapi()
    in_dev = find_device(args.in_device, "input")
    in_dev = in_dev if in_dev is not None else host["default_input_device"]
    args.out_device = find_device(args.out_device, "output")
    settings = sd.WasapiSettings(auto_convert=True)
    pipe = Pipeline(preset(args.mode), detector=make_detector(args))
    raw, out, detect_ms = [], [], []
    q: queue.Queue = queue.Queue()
    running = True

    if args.out_device is not None:
        print("Playback on. Use headphones or a virtual cable: laptop speakers feed back into the mic, "
              "and hearing your own delayed voice can change how you speak.")

    def on_input(indata, frames, t, status):
        x = indata[:, 0].copy()
        pipe.write(x)
        raw.append(x)
        q.put(x)

    def on_output(outdata, frames, t, status):
        y = pipe.read(frames)
        outdata[:, 0] = y
        out.append(y)

    def worker():
        while running or not q.empty():
            try:
                x = q.get(timeout=0.2)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            pipe.detect(x)
            detect_ms.append((time.perf_counter() - t0) * 1000)
            if args.out_device is None:
                out.append(pipe.read(len(x)))  # no device: the input clock drives the output

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    streams = [sd.InputStream(device=in_dev, samplerate=SR, channels=1, blocksize=HOP, dtype="float32",
                              latency="low", extra_settings=settings, callback=on_input)]
    if args.out_device is not None:
        streams.append(sd.OutputStream(device=args.out_device, samplerate=SR, channels=1, blocksize=HOP,
                                       dtype="float32", latency="low", extra_settings=settings,
                                       callback=on_output))

    print(f"Listening on: {sd.query_devices(in_dev)['name']}  mode={args.mode}  (Ctrl+C to stop)")
    start = time.time()
    try:
        for s in streams:
            s.start()
        while args.seconds is None or time.time() - start < args.seconds:
            time.sleep(0.5)
            level = 20 * np.log10(np.sqrt(np.mean(raw[-1] ** 2)) + 1e-9) if raw else -180
            n_cuts = len(pipe.splicer.stats.removed_spans)
            print(f"\r  {time.time() - start:6.1f}s  mic {level:6.1f} dBFS  lag {pipe.splicer.lag_ms():5.0f} ms  "
                  f"cuts {n_cuts}   ", end="", flush=True)
            if time.time() - start > 2 and raw and max(np.abs(r).max() for r in raw[-100:]) == 0.0:
                print("\n  Mic is delivering pure silence. Check Settings > Privacy & security > Microphone "
                      "('Let desktop apps access your microphone') and that the mic is not muted.")
                break
    except KeyboardInterrupt:
        pass
    finally:
        for s in streams:
            s.stop()
            s.close()
        running = False
        th.join(timeout=2)

    session = ROOT / "sessions" / datetime.now().strftime("%Y%m%d-%H%M%S")
    save_wav(session / "raw.wav", np.concatenate(raw) if raw else np.zeros(1))
    save_wav(session / "out.wav", np.concatenate(out) if out else np.zeros(1))
    summary = pipe.summary()
    if detect_ms:
        summary["detector_ms_per_10ms_block"] = {"p50": round(float(np.percentile(detect_ms, 50)), 3),
                                                 "p95": round(float(np.percentile(detect_ms, 95)), 3)}
    (session / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {session}")
    print(json.dumps({k: summary[k] for k in ("lag_ms", "removed_ms", "detector_ms_per_10ms_block") if k in summary}))


if __name__ == "__main__":
    main()
