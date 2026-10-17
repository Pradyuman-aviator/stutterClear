"""Download the captions model: NVIDIA's streaming FastConformer (NeMo), 480 ms chunks, int8 ONNX
for sherpa-onnx (106 MB), into models/asr/.

    python scripts/download_asr.py
"""
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stutterclear.captions import MODEL_DIR  # noqa: E402

URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{}.tar.bz2"

if __name__ == "__main__":
    if (MODEL_DIR / "encoder.int8.onnx").exists():
        sys.exit(f"already there: {MODEL_DIR}")
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    archive = MODEL_DIR.parent / f"{MODEL_DIR.name}.tar.bz2"
    print("downloading", URL.format(MODEL_DIR.name))
    urllib.request.urlretrieve(URL.format(MODEL_DIR.name), archive)
    with tarfile.open(archive) as tf:
        tf.extractall(MODEL_DIR.parent, filter="data")  # plain files under models/asr only
    archive.unlink()
    print("ready:", MODEL_DIR)
