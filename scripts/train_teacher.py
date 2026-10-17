"""Train the frame-level stutter teacher.

Local smoke test (RTX 4050, dev-clean only):
    python scripts/train_teacher.py --libri-train "data/librispeech-alignments/data/dev_clean-*.parquet" \
        --libri-val "data/librispeech-alignments/data/dev_clean-*.parquet" --steps 30 --batch 2 --eval-every 30 --val-n 8

Kaggle (full run): see kaggle/teacher/run.py.
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
from stutterclear.synth import CLASSES  # noqa: E402
from stutterclear.teacher import FrameTeacher, clip_logits, clip_loss, frame_loss  # noqa: E402
from stutterclear.train_data import Sep28kClips, SynthStutterStream, fixed_synth_set  # noqa: E402

SYNTH_CLASSES = 4  # interjection never appears in synthetic data


def cycle(loader):
    while True:
        yield from loader


def f1(tp, fp, fn):
    return 2 * tp / max(1, 2 * tp + fp + fn)


@torch.no_grad()
def evaluate(model, val_synth, val_sep, dev, batch):
    model.eval()
    res = {}
    tp = np.zeros(len(CLASSES)); fp = np.zeros(len(CLASSES)); fn = np.zeros(len(CLASSES))
    cut = np.zeros(3)  # any removable class (sound/word repetition, prolongation): what the splicer acts on
    for i in range(0, len(val_synth), batch):
        b = val_synth[i:i + batch]
        wave = torch.stack([x["wave"] for x in b]).to(dev)
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            prob = torch.sigmoid(model(wave).float()).cpu().numpy()
        lab = torch.stack([x["labels"] for x in b]).numpy()
        mask = torch.stack([x["mask"] for x in b]).numpy()
        t = min(prob.shape[1], lab.shape[1])
        p, y, m = prob[:, :t] > 0.5, lab[:, :t] > 0.5, mask[:, :t, None] > 0.5
        tp += (p & y & m).sum((0, 1)); fp += (p & ~y & m).sum((0, 1)); fn += (~p & y & m).sum((0, 1))
        pc, yc, mc = p[..., :3].any(-1), y[..., :3].any(-1), m[..., 0]
        cut += [(pc & yc & mc).sum(), (pc & ~yc & mc).sum(), (~pc & yc & mc).sum()]
    res["synth_f1"] = {c: round(f1(tp[k], fp[k], fn[k]), 3) for k, c in enumerate(CLASSES[:SYNTH_CLASSES])}
    res["synth_macro_f1"] = round(float(np.mean(list(res["synth_f1"].values()))), 3)
    res["synth_cut_f1"] = round(f1(*cut), 3)

    if val_sep is not None:
        probs, targets = [], []
        loader = DataLoader(val_sep, batch_size=batch * 2, num_workers=0)
        for b in loader:
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                probs.append(torch.sigmoid(clip_logits(model(b["wave"].to(dev)).float())).cpu().numpy())
            targets.append(b["clip_labels"].numpy())
        P, Y = np.concatenate(probs), np.concatenate(targets) >= 2 / 3 - 1e-6  # majority of 3 annotators
        res["sep_f1"] = {c: round(f1(((P[:, k] > 0.5) & Y[:, k]).sum(), ((P[:, k] > 0.5) & ~Y[:, k]).sum(),
                                      ((P[:, k] <= 0.5) & Y[:, k]).sum()), 3) for k, c in enumerate(CLASSES)}
        res["sep_macro_f1"] = round(float(np.mean(list(res["sep_f1"].values()))), 3)
        try:
            from sklearn.metrics import roc_auc_score
            res["sep_auc"] = {c: round(float(roc_auc_score(Y[:, k], P[:, k])), 3)
                              for k, c in enumerate(CLASSES) if 0 < Y[:, k].sum() < len(Y)}
        except ImportError:
            pass
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libri-train", required=True, help="glob of librispeech-alignments parquet files")
    ap.add_argument("--libri-val", required=True)
    ap.add_argument("--sep-dir", help="folder with sep28k-fluencybank parquet files")
    ap.add_argument("--backbone", default="microsoft/wavlm-base-plus")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--sep-batch", type=int, default=8)
    ap.add_argument("--sep-weight", type=float, default=1.0)
    ap.add_argument("--crop-s", type=float, default=6.0)
    ap.add_argument("--lr-backbone", type=float, default=3e-5)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--freeze-steps", type=int, default=500, help="train only the head at first")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--val-n", type=int, default=400)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--out", default="runs/teacher")
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_files, val_files = sorted(glob.glob(args.libri_train)), sorted(glob.glob(args.libri_val))
    assert train_files and val_files, "no parquet files matched"

    model = FrameTeacher(args.backbone).to(dev)
    if args.grad_ckpt:
        model.ssl.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.param_groups(args.lr_backbone, args.lr_head), weight_decay=0.01)

    def schedule(step, frozen_until=0):
        if step < frozen_until:
            return 0.0
        s = step - frozen_until
        if s < args.warmup:
            return (s + 1) / args.warmup
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, args.steps - frozen_until))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, [lambda s: schedule(s, args.freeze_steps), schedule])
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")

    synth_it = cycle(DataLoader(SynthStutterStream(train_files, args.crop_s, seed=0), batch_size=args.batch,
                                num_workers=args.workers, persistent_workers=args.workers > 0))
    val_synth = fixed_synth_set(val_files, args.val_n, args.crop_s)
    sep_it, val_sep = None, None
    if args.sep_dir:
        sep_files = sorted(glob.glob(str(Path(args.sep_dir) / "**" / "*.parquet"), recursive=True))
        sep_train = Sep28kClips(sep_files, "train")
        val_sep = Sep28kClips(sep_files, "val", augment_p=0.0)
        print(f"SEP-28k/FluencyBank: {len(sep_train)} train clips ({len(sep_train.groups)} speaker groups), "
              f"{len(val_sep)} val clips ({len(val_sep.groups)} groups)", flush=True)
        sep_it = cycle(DataLoader(sep_train, batch_size=args.sep_batch, shuffle=True, drop_last=True,
                                  num_workers=args.workers, persistent_workers=args.workers > 0))

    (out / "config.json").write_text(json.dumps({**vars(args), "classes": CLASSES}, indent=2))
    best, t0, log = -1.0, time.time(), open(out / "metrics.jsonl", "a")
    model.train()
    for step in range(1, args.steps + 1):
        b = next(synth_it)
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            loss_f = frame_loss(model(b["wave"].to(dev)), b["labels"].to(dev), b["mask"].to(dev))
            loss = loss_f
            if sep_it is not None:
                c = next(sep_it)
                loss_c = clip_loss(model(c["wave"].to(dev)).float(), c["clip_labels"].to(dev))
                loss = loss + args.sep_weight * loss_c
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()

        if step % 25 == 0 or step == 1:
            el = time.time() - t0
            msg = f"step {step} loss {loss.item():.4f} (frame {loss_f.item():.4f}"
            msg += f", clip {loss_c.item():.4f})" if sep_it is not None else ")"
            print(f"{msg}  {step / el:.2f} it/s  {el / 60:.1f} min", flush=True)
        last = step == args.steps or (time.time() - t0) / 3600 > args.max_hours
        if step % args.eval_every == 0 or last:
            m = evaluate(model, val_synth, val_sep, dev, args.batch)
            score = m["synth_macro_f1"] if "sep_macro_f1" not in m else (m["synth_macro_f1"] + m["sep_macro_f1"]) / 2
            m.update(step=step, score=round(score, 4), minutes=round((time.time() - t0) / 60, 1))
            print(json.dumps(m), flush=True)
            log.write(json.dumps(m) + "\n")
            log.flush()
            if score > best:
                best = score
                torch.save({"state_dict": model.state_dict(), "backbone": args.backbone, "classes": CLASSES,
                            "step": step, "metrics": m}, out / "best.pt")
        if last:
            break
    print(f"done. best score {best:.4f}; checkpoint {out / 'best.pt'}")


if __name__ == "__main__":
    main()
