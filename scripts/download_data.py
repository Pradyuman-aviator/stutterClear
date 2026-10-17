"""Download training data into data/ (git-ignored).

    python scripts/download_data.py            # dev-clean + train-clean-100 + SEP-28k/FluencyBank (~9.7 GB)
    python scripts/download_data.py --small    # dev-clean only (~0.35 GB), enough to try the generator

- gilkeyio/librispeech-alignments: LibriSpeech audio with Montreal Forced Aligner word and
  phone timings (CC BY 4.0). Clean speech that stutters are synthesised into.
- saeedzou/sep28k-fluencybank-stutter-dataset: real stuttered 3 s clips with clip-level
  labels. SEP-28k labels are CC BY-NC 4.0: research use only.
"""
import argparse
from pathlib import Path

import urllib.request

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1] / "data"
SEP_E_URL = ("https://raw.githubusercontent.com/th-nuernberg/ml-stuttering-events-dataset-extended/"
             "main/SEP-28k-Extended_clips.csv")


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--root", help="download folder (default: <repo>/data)")
    ap.add_argument("--sep-only", action="store_true")
    args = ap.parse_args()
    ROOT = Path(args.root) if args.root else ROOT

    if not args.sep_only:
        libri = ["data/dev_clean-*.parquet"] + ([] if args.small else ["data/train_clean_100-*.parquet"])
        snapshot_download("gilkeyio/librispeech-alignments", repo_type="dataset", allow_patterns=libri,
                          local_dir=ROOT / "librispeech-alignments")
        print("librispeech-alignments done", flush=True)
    if not args.small:
        snapshot_download("saeedzou/sep28k-fluencybank-stutter-dataset", repo_type="dataset",
                          allow_patterns=["data/*.parquet", "README.md"], local_dir=ROOT / "sep28k-fluencybank")
        # Official speaker-exclusive SEP-28k-E partition (Bayerl et al. 2022, CC BY-NC 4.0).
        urllib.request.urlretrieve(SEP_E_URL, ROOT / "sep28k-fluencybank" / "SEP-28k-Extended_clips.csv")
        print("sep28k-fluencybank done (with SEP-28k-E partition)", flush=True)


if __name__ == "__main__":
    main()
