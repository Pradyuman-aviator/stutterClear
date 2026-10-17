"""Distil the teacher into the small causal student (one head per lookahead).

    python scripts/train_student.py --teacher-ckpt runs/teacher/best.pt \
        --libri-train "data/librispeech-alignments/data/train_clean_100-*.parquet" \
        --libri-val "data/librispeech-alignments/data/dev_clean-*.parquet" --sep-dir data/sep28k-fluencybank

On two GPUs the frozen teacher runs on the second one.
"""
import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_teacher import load_teacher  # noqa: E402
from stutterclear.evaluation import clip_scores, predict_clips, report, tune_thresholds  # noqa: E402
from stutterclear.student import CausalStudent, distill_loss, shifted  # noqa: E402
from stutterclear.synth import CLASSES  # noqa: E402
from stutterclear.train_data import Sep28kClips, SynthStutterStream, UclassCrops, fixed_synth_set  # noqa: E402
from train_teacher import cycle  # noqa: E402


@torch.no_grad()
def synth_f1(student, val, dev):
    """Frame F1 against the exact synthetic labels, for the 4 synthesised classes, per lookahead head."""
    student.eval()
    stats = np.zeros((len(student.lookaheads), 4, 3))
    for i in range(0, len(val), 16):
        b = val[i:i + 16]
        outs = student(torch.stack([x["wave"] for x in b]).to(dev))
        lab = torch.stack([x["labels"] for x in b]).to(dev)
        mask = torch.stack([x["mask"] for x in b]).to(dev)
        for h, (lg, la) in enumerate(zip(outs, student.lookaheads)):
            lg, y = shifted(lg, lab, la)
            _, m = shifted(outs[h], mask[..., None], la)
            p, y, m = (torch.sigmoid(lg) > 0.5)[..., :4], (y > 0.5)[..., :4], (m > 0.5)
            stats[h] += torch.stack([(p & y & m).sum((0, 1)), (p & ~y & m).sum((0, 1)),
                                     (~p & y & m).sum((0, 1))], -1).cpu().numpy()
    student.train()
    f1 = 2 * stats[..., 0] / np.maximum(1, 2 * stats[..., 0] + stats[..., 1] + stats[..., 2])
    return {f"{la * 20}ms": {c: round(float(v), 3) for c, v in zip(CLASSES[:4], row)}
            for la, row in zip(student.lookaheads, f1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-ckpt", required=True)
    ap.add_argument("--libri-train", required=True)
    ap.add_argument("--libri-val", required=True)
    ap.add_argument("--sep-dir")
    ap.add_argument("--uclass-dir", help="unlabelled UCLASS recordings: distilled from the teacher (train speakers only)")
    ap.add_argument("--uclass-batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--sep-batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--clip-weight", type=float, default=0.3)
    ap.add_argument("--crop-s", type=float, default=6.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--val-n", type=int, default=400)
    ap.add_argument("--max-hours", type=float, default=10.0)
    ap.add_argument("--out", default="runs/student")
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tdev = torch.device("cuda:1") if torch.cuda.device_count() > 1 else dev
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    teacher, _ = load_teacher(args.teacher_ckpt, tdev)
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = CausalStudent().to(dev)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 500) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))))

    @torch.no_grad()
    def teach(wave):
        with torch.autocast(tdev.type, enabled=tdev.type == "cuda"):
            return torch.sigmoid(teacher(wave.to(tdev)).float()).to(dev)

    train_files, val_files = sorted(glob.glob(args.libri_train)), sorted(glob.glob(args.libri_val))
    synth_it = cycle(DataLoader(SynthStutterStream(train_files, args.crop_s, seed=1), batch_size=args.batch,
                                num_workers=args.workers, persistent_workers=args.workers > 0))
    val_synth = fixed_synth_set(val_files, args.val_n, args.crop_s)
    sep_it = val_sep = test_sep = None
    uc_it = None
    if args.uclass_dir:
        uc_it = cycle(DataLoader(UclassCrops(args.uclass_dir, "train", args.crop_s, seed=2), batch_size=args.uclass_batch,
                                 num_workers=args.workers, persistent_workers=args.workers > 0))
    if args.sep_dir:
        files = sorted(glob.glob(str(Path(args.sep_dir) / "**" / "*.parquet"), recursive=True))
        sep_it = cycle(DataLoader(Sep28kClips(files, "train"), batch_size=args.sep_batch, shuffle=True,
                                  drop_last=True, num_workers=args.workers, persistent_workers=args.workers > 0))
        val_sep, test_sep = Sep28kClips(files, "val", augment_p=0.0), Sep28kClips(files, "test", augment_p=0.0)

    (out / "config.json").write_text(json.dumps({**vars(args), "lookaheads": student.lookaheads,
                                                 "classes": CLASSES}, indent=2))
    log, best, t0 = open(out / "metrics.jsonl", "a"), -1.0, time.time()
    for step in range(1, args.steps + 1):
        b = next(synth_it)
        wave = b["wave"].to(dev)
        loss = distill_loss(student(wave), student.lookaheads, teach(wave), b["labels"].to(dev), b["mask"].to(dev))
        if sep_it is not None:
            c = next(sep_it)
            wave = c["wave"].to(dev)
            outs = student(wave)
            loss = loss + distill_loss(outs, student.lookaheads, teach(wave))
            clip_y = c["clip_labels"].to(dev)
            loss = loss + args.clip_weight * sum(
                torch.nn.functional.binary_cross_entropy(clip_scores(o).clamp(1e-6, 1 - 1e-6), clip_y)
                for o in outs) / len(outs)
        if uc_it is not None:  # real stuttered speech, labelled only by the teacher
            wave = next(uc_it)["wave"].to(dev)
            loss = loss + distill_loss(student(wave), student.lookaheads, teach(wave))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if step % 50 == 0 or step == 1:
            el = time.time() - t0
            print(f"step {step} loss {loss.item():.4f}  {step / el:.2f} it/s  {el / 60:.1f} min", flush=True)
        last = step == args.steps or (time.time() - t0) / 3600 > args.max_hours
        if step % args.eval_every == 0 or last:
            m = {"step": step, "synth_f1": synth_f1(student, val_synth, dev)}
            if val_sep is not None:
                student.eval()
                probs, votes = predict_clips(lambda w: student(w), val_sep, dev)
                student.train()
                m["sep_val_auc"] = {f"{la * 20}ms": report(p, votes).get("auc", {})
                                    for la, p in zip(student.lookaheads, probs)}
                score = float(np.mean([np.mean(list(v.values())) for v in m["sep_val_auc"].values()]))
            else:
                score = float(np.mean([np.mean(list(v.values())) for v in m["synth_f1"].values()]))
            m.update(score=round(score, 4), minutes=round((time.time() - t0) / 60, 1))
            print(json.dumps(m), flush=True)
            log.write(json.dumps(m) + "\n")
            log.flush()
            if score > best:
                best = score
                torch.save({"state_dict": student.state_dict(), "lookaheads": student.lookaheads,
                            "classes": CLASSES, "step": step, "metrics": m}, out / "student.pt")
        if last:
            break

    # Final: thresholds tuned per head on validation speakers, reported once on test speakers.
    if val_sep is not None:
        student.load_state_dict(torch.load(out / "student.pt", weights_only=False)["state_dict"])
        student.eval()
        pv, yv = predict_clips(lambda w: student(w), val_sep, dev)
        pt, yt = predict_clips(lambda w: student(w), test_sep, dev)
        final = {}
        for la, a, b in zip(student.lookaheads, pv, pt):
            th = tune_thresholds(a, yv)
            final[f"{la * 20}ms"] = {"thresholds": th, "test": report(b, yt, th)}
        (out / "student_test_report.json").write_text(json.dumps(final, indent=2))
        print(json.dumps({k: {"macro_f1": v["test"]["macro_f1"], "auc": v["test"].get("auc")}
                          for k, v in final.items()}, indent=1))
    print(f"done. best score {best:.4f}; {out / 'student.pt'}")


if __name__ == "__main__":
    main()
