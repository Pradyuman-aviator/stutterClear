"""Fallback only: put the two teacher checkpoints in a PRIVATE Kaggle dataset, for the day the GPU worker
kernel cannot mount the training kernels' outputs. Normally it reads them from there and nothing has to
be uploaded (docs/DEPLOY.md, "GPU on demand").

    python scripts/upload_weights_kaggle.py --dataset OWNER/stutterclear-weights --dry-run
    python scripts/upload_weights_kaggle.py --dataset OWNER/stutterclear-weights [--notes "..."]

Then set STUTTERCLEAR_KAGGLE_WEIGHTS=OWNER/stutterclear-weights on the server. The checkpoints are
research-licensed and must never be public: the dataset is created private, a new version keeps it so,
there is no option to change that, and the script checks afterwards. They are staged under the names
the worker's launcher looks for, in a temporary folder outside the repo (as hard links where the disk
allows, so nothing is copied). Kaggle credentials come from the usual places (KAGGLE_API_TOKEN, or
KAGGLE_USERNAME and KAGGLE_KEY, or ~/.kaggle) and are never printed; --dry-run does not need them.
"""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stutterclear.gpu_worker.kaggle import LIVE_FILE, RECORDINGS_FILE  # noqa: E402

REF_RE = re.compile(r"[\w-]+/[a-z0-9][a-z0-9-]{5,49}")


def checkpoints() -> list[tuple[Path, str]]:
    """(the checkpoint, its name in the dataset): the live teacher and the recordings teacher, the ones
    the app itself would use (STUTTERCLEAR_TEACHER and STUTTERCLEAR_RECORDINGS_TEACHER apply)."""
    from stutterclear.teacher_runtime import RECORDINGS_CKPT, TEACHER_CKPT
    return [(Path(TEACHER_CKPT), LIVE_FILE), (Path(RECORDINGS_CKPT), RECORDINGS_FILE)]


def stage(files: list[tuple[Path, str]], folder: str):
    for src, name in files:
        dst = os.path.join(folder, name)
        try:
            os.link(src, dst)  # the same disk: nothing is copied
        except OSError:
            shutil.copy2(src, dst)


def is_private(api, ref: str) -> bool | None:
    """Whether the dataset is private; None while Kaggle does not show it (a new one takes a moment)."""
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
    req = ApiGetDatasetRequest()
    req.owner_slug, req.dataset_slug = ref.split("/")
    try:
        with api.build_kaggle_client() as k:
            return bool(k.datasets.dataset_api_client.get_dataset(req).is_private)
    except Exception as e:
        if getattr(getattr(e, "response", None), "status_code", None) == 404:
            return None
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Upload the teacher checkpoints to a PRIVATE Kaggle dataset (fallback).")
    ap.add_argument("--dataset", required=True, help="OWNER/slug of the dataset (created private if it does not exist)")
    ap.add_argument("--notes", default="StutterClear teachers (private)", help="the new version's notes")
    ap.add_argument("--dry-run", action="store_true", help="list the files and their sizes; upload nothing")
    args = ap.parse_args(argv)
    if not REF_RE.fullmatch(args.dataset):
        print("--dataset is OWNER/slug, the slug 6 to 50 lowercase letters, digits and dashes")
        return 2
    files = checkpoints()
    missing = [str(src) for src, _ in files if not src.is_file()]
    if missing:
        print("missing:\n  " + "\n  ".join(missing))
        return 1
    for src, name in files:
        print(f"{name:20} {src.stat().st_size / 1e6:8.1f} MB  from {src}")
    if args.dry_run:
        print("dry run: nothing uploaded")
        return 0
    settings = f"https://www.kaggle.com/datasets/{args.dataset}/settings"
    from kaggle.api.kaggle_api_extended import KaggleApi  # importing kaggle signs in: only now
    api = KaggleApi()
    api.authenticate()
    before = is_private(api, args.dataset)
    if before is False:
        print(f"THE DATASET IS PUBLIC: make it private now at {settings} (nothing was uploaded)")
        return 1
    with tempfile.TemporaryDirectory(prefix="stutterclear-weights-") as tmp:
        folder = os.path.normpath(tmp)  # the Kaggle client mishandles some Windows paths otherwise
        if ROOT in Path(folder).resolve().parents:
            print("the temporary folder is inside the repo; set TMP or TEMP elsewhere")
            return 1
        stage(files, folder)
        meta = {"title": "StutterClear weights (private)", "id": args.dataset, "licenses": [{"name": "other"}],
                "description": "Research-licensed StutterClear teacher checkpoints. Private: never share or publish."}
        Path(folder, "dataset-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        if before is None:
            r = api.dataset_create_new(folder, public=False, quiet=False, convert_to_csv=False, dir_mode="skip")
        else:
            r = api.dataset_create_version(folder, args.notes, quiet=False, convert_to_csv=False, dir_mode="skip")
    if getattr(r, "error", None):
        print(f"Kaggle refused the upload: {r.error}")
        return 1
    after = None
    for _ in range(24):  # a new dataset takes a moment to appear
        after = is_private(api, args.dataset)
        if after is not None:
            break
        time.sleep(5)
    if after is False:
        print(f"THE DATASET IS PUBLIC: make it private now at {settings}")
        return 1
    if after is None:
        print(f"Uploaded, but Kaggle does not show the dataset yet: check it is private at {settings}")
        return 1
    print(f"Uploaded privately: {args.dataset}. Set STUTTERCLEAR_KAGGLE_WEIGHTS={args.dataset} on the server.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
