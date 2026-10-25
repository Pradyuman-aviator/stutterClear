"""Record the GPU model's live decisions on each demo clip, so a server whose GPU sleeps still plays
what the big model does live: the demo's Live button replays them through the same splicer
(stutterclear/app/demo.py, ReplayDetector).

    python scripts/record_live_demo.py    # every demo clip whose audio is here; needs this computer's GPU

Writes <clip>.live.json next to each clip's audio: timings only, no audio. The synthetic clips' go in
git (data/demo_clips); the UCLASS clips' stay with their audio in data/external and go up with it
(list them in deploy/cloudrun/uclass.sha256). Record again after changing the live teacher or its cut
settings.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stutterclear.app import demo  # noqa: E402
from stutterclear.live import make_detector  # noqa: E402
from stutterclear.teacher_runtime import live_capable  # noqa: E402


def main():
    if not live_capable():
        sys.exit("The live teacher runs on this computer's GPU, and there is none here.")
    for sid, s in demo.SAMPLES.items():
        if not demo._wav(s).exists():
            continue
        det, name = make_detector("teacher", "gentle")
        rec = demo.record_live(demo._excerpt(sid), det, name)
        rec.update(start=s["start"], end=s["end"])
        path = demo._wav(s).with_suffix(demo.LIVE_REC)
        path.write_text(json.dumps(rec, separators=(",", ":")) + "\n")
        cuts = {u[1] for _, _, ups in rec["steps"] for u in ups if u[0] == "set" and u[3] is not None}
        print(f"{sid}: {len(cuts)} cuts decided in {s['end'] - s['start']:.1f} s -> {path.name} ({path.stat().st_size} bytes)",
              flush=True)


if __name__ == "__main__":
    main()
