"""Download UCLASS (University College London Archive of Stuttered Speech) WAV files.

    python scripts/download_uclass.py --root data/external        # all ~456 recordings
    python scripts/download_uclass.py --root data/external --limit 3

Free for research and teaching; acknowledge UCLASS and the Wellcome Trust (Howell, Davis &
Bartrip 2009, JSLHR 52, 556-569). Files are saved as <task>/<name>: the same file name can be a
different recording in another task folder (e.g. a monologue and a reading).
"""
import argparse
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://www.uclass.psychol.ucl.ac.uk/"
FOLDERS = {
    "release1": "Audio/wav/",
    "conversation": "Release2/Conversation/AudioOnly/wav/",
    "reading": "Release2/Reading/AudioOnly/wav/",
    "monologue": "Release2/Monologue/AudioOnly/wav/",
}


def listing(folder: str) -> list:
    html = urllib.request.urlopen(BASE + folder, timeout=60).read().decode("utf-8", "ignore")
    names = re.findall(r'href="(?:[^"]*/)?([^"/]+\.wav)"', html, flags=re.I)
    return sorted(set(names))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/external")
    ap.add_argument("--limit", type=int, help="only this many files (for testing)")
    args = ap.parse_args()
    out = Path(args.root) / "uclass_all"
    jobs = {}
    for task, folder in FOLDERS.items():
        for name in listing(folder):
            jobs[(task, name)] = BASE + folder + name
    items = sorted(jobs.items())[: args.limit] if args.limit else sorted(jobs.items())

    def fetch(item):
        (task, name), url = item
        dest = out / task / name
        if dest.exists() and dest.stat().st_size > 0:
            return 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        try:
            urllib.request.urlretrieve(url, tmp)
        except OSError as e:  # the index lists a few files the server no longer has (404)
            print(f"skipped {task}/{name}: {e}", flush=True)
            tmp.unlink(missing_ok=True)
            return 0
        tmp.replace(dest)
        return dest.stat().st_size

    with ThreadPoolExecutor(8) as pool:
        total = sum(pool.map(fetch, items))
    have = sum(1 for _ in out.rglob("*.wav"))
    print(f"{have} of {len(items)} recordings in {out} ({total / 1e9:.2f} GB downloaded now)")


if __name__ == "__main__":
    main()
