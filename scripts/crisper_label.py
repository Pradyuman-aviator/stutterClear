"""Verbatim word timings for real stuttered speech, with CrisperWhisper 2.0 (research licence).

    python scripts/crisper_label.py --uclass data/external/uclass_all --sep-dir data/sep28k-fluencybank \
        --out runs/silver --shard 0 --nshard 2

Each shard writes runs/silver/words_<shard>.jsonl, one line per recording or clip:
    {"id", "source": "uclass"|"sep", "speaker", "words": [{"w", "s", "e"}]}
and resumes where it stopped. Run one shard per GPU (CUDA_VISIBLE_DEVICES) in parallel.
"""
import argparse
import glob
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stutterclear.audio import load_wav  # noqa: E402
from stutterclear.train_data import speaker_group  # noqa: E402


def uclass_speaker(name: str) -> str:
    return "uclass/" + "_".join(Path(name).stem.split("_")[:2])  # e.g. M_0065_20y1m_1 -> M_0065


def record_name(source: str, file: str, args) -> str:
    """UCLASS: path under the UCLASS root ("monologue/M_0065_20y1m_1.wav"), because the same file
    name can be a different recording in another task folder. SEP: the parquet file name."""
    return Path(file).relative_to(args.uclass).as_posix() if source == "uclass" else Path(file).name


def tasks(args) -> list:
    out = []
    if args.uclass:
        for p in sorted(Path(args.uclass).rglob("*.wav")):
            out.append(("uclass", str(p), None, uclass_speaker(p.name)))
    if args.sep_dir:
        for f in sorted(glob.glob(str(Path(args.sep_dir) / "**" / "*.parquet"), recursive=True)):
            meta = pq.read_table(f, columns=["Show", "EpId"]).to_pydict()
            for i, (show, ep) in enumerate(zip(meta["Show"], meta["EpId"])):
                out.append(("sep", f, i, speaker_group(show, ep)))
        if args.max_sep:
            uc = [t for t in out if t[0] == "uclass"]
            sep = [t for t in out if t[0] == "sep"]
            step = max(1, len(sep) // args.max_sep)
            out = uc + sep[::step][: args.max_sep]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uclass")
    ap.add_argument("--sep-dir")
    ap.add_argument("--max-sep", type=int, help="label an evenly spaced subset of SEP clips")
    ap.add_argument("--out", default="runs/silver")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--model", default="large")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    from crisperwhisper import CrisperWhisperModel
    model = CrisperWhisperModel(args.model, backend=args.backend)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"words_{args.shard}.jsonl"
    done = {json.loads(line)["id"] for line in open(path)} if path.exists() else set()
    mine = tasks(args)[args.shard::args.nshard][: args.limit]
    todo = [t for t in mine if f"{record_name(t[0], t[1], args)}:{t[2]}" not in done]
    print(f"shard {args.shard}/{args.nshard}: {len(mine)} items, {len(todo)} to do", flush=True)

    tables, t0, audio_s = {}, time.time(), 0.0
    with open(path, "a") as f:
        for n, (source, file, row, speaker) in enumerate(todo, 1):
            if source == "uclass":
                x = load_wav(file)
            else:
                if file not in tables:
                    tables = {file: pq.read_table(file, columns=["audio"]).column("audio")}
                x, sr = sf.read(io.BytesIO(tables[file][row]["bytes"].as_py()), dtype="float32")
                x = x.mean(axis=1) if x.ndim > 1 else x
            audio_s += len(x) / 16000
            try:
                r = model.transcribe(np.ascontiguousarray(x, np.float32), sr=16000, language="en",
                                     word_timestamps=True)
                words = [{"w": w.word, "s": round(float(w.start), 3), "e": round(float(w.end), 3)} for w in r.words]
            except Exception as e:  # one bad clip must not stop a multi-hour job
                print(f"failed {file}:{row}: {e}", flush=True)
                continue
            name = record_name(source, file, args)
            f.write(json.dumps({"id": f"{name}:{row}", "source": source, "file": name,
                                "row": row, "speaker": speaker, "duration": round(len(x) / 16000, 2),
                                "words": words}) + "\n")
            f.flush()
            if n % 50 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"shard {args.shard}: {n}/{len(todo)}  {audio_s / 60:.0f} min audio in {el / 60:.1f} min "
                      f"({audio_s / max(el, 1e-6):.1f}x real time)", flush=True)


if __name__ == "__main__":
    main()
