"""Bundle what the hosted server needs and git does not hold, into one archive to copy to the VM:
the teacher checkpoints (live, and recordings if different) with their calibration, the measured
live quality, the demo recordings with their caches, and the captions model. See docs/DEPLOY.md.

    python scripts/pack_assets.py            # writes stutterclear_assets.tar
"""
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stutterclear.captions import MODEL_DIR  # noqa: E402
from stutterclear.teacher_runtime import RECORDINGS_CKPT, TEACHER_CKPT  # noqa: E402


def files() -> list:
    out = [ROOT / "runs" / "eval" / "live_quality.json"]
    for ckpt in dict.fromkeys([TEACHER_CKPT, RECORDINGS_CKPT]):  # live and recordings teachers, once each
        out.append(ckpt)
        if ckpt.with_name("live_cuts.json").exists():
            out.append(ckpt.with_name("live_cuts.json"))
    for s in json.loads((ROOT / "stutterclear" / "app" / "demo_samples.json").read_text()):
        wav = ROOT / "data" / "external" / s["file"]
        out += [wav, wav.with_suffix(".teacher.npy"), wav.with_suffix(".crisper.json")]
    out += sorted(p for p in MODEL_DIR.iterdir() if p.is_file())
    return out


if __name__ == "__main__":
    missing = [p for p in files() if not p.exists()]
    if missing:
        sys.exit("missing:\n  " + "\n  ".join(str(p) for p in missing))
    dest = ROOT / "stutterclear_assets.tar"
    with tarfile.open(dest, "w") as tar:
        for p in files():
            tar.add(p, arcname=p.relative_to(ROOT).as_posix())
    print(f"{dest} ({dest.stat().st_size / 1e6:.0f} MB, {len(files())} files)")
