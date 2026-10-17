"""Tune per-class thresholds on SEP-28k validation speakers, then report once on test speakers.

    python scripts/eval_teacher.py --ckpt runs/teacher/best.pt --sep-dir data/sep28k-fluencybank --out runs/teacher
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stutterclear.evaluation import predict_clips, report, tune_thresholds  # noqa: E402
from stutterclear.teacher import FrameTeacher  # noqa: E402
from stutterclear.train_data import Sep28kClips  # noqa: E402


def load_teacher(path, dev):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = FrameTeacher(ckpt["backbone"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(dev).eval(), ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sep-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_teacher(args.ckpt, dev)
    files = sorted(glob.glob(str(Path(args.sep_dir) / "**" / "*.parquet"), recursive=True))

    val = Sep28kClips(files, "val", augment_p=0.0)
    test = Sep28kClips(files, "test", augment_p=0.0)
    (pv,), yv = predict_clips(lambda w: [model(w)], val, dev)
    th = tune_thresholds(pv, yv)
    (pt,), yt = predict_clips(lambda w: [model(w)], test, dev)
    res = {"checkpoint_step": ckpt.get("step"), "thresholds": th,
           "val_tuned": report(pv, yv, th),
           "test_at_0.5": report(pt, yt), "test_tuned": report(pt, yt, th),
           "val_groups": val.groups, "test_groups": test.groups}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "thresholds.json").write_text(json.dumps(th))
    (out / "teacher_test_report.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("checkpoint_step", "thresholds", "test_at_0.5", "test_tuned")}, indent=1))


if __name__ == "__main__":
    main()
