"""Train the frame-level stutter teacher. Runs on one GPU, or on several with torchrun (DDP).

Each step adds up gradients from up to three real/synthetic sources:
- synthetic stutters on LibriSpeech (exact frame labels, all classes);
- SEP-28k / FluencyBank 3 s clips (clip labels; plus silver frame labels where available);
- UCLASS recordings (silver frame labels for fillers, cut-offs and repeats).
The best checkpoint is chosen on real data only.

Local smoke test:
    python scripts/train_teacher.py --libri-train "data/librispeech-alignments/data/dev_clean-*.parquet" \
        --libri-val "data/librispeech-alignments/data/dev_clean-*.parquet" --steps 30 --batch 2 --val-n 8
Two GPUs:
    torchrun --nproc_per_node=2 scripts/train_teacher.py ...
"""
import argparse
import datetime
import glob
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stutterclear.silver import KNOWN  # noqa: E402
from stutterclear.synth import CLASSES  # noqa: E402
from stutterclear.teacher import FrameTeacher, clip_logits, clip_loss, frame_loss  # noqa: E402
from stutterclear.train_data import (Sep28kClips, SynthStutterStream, UclassSilver,  # noqa: E402
                                     fixed_synth_set, load_silver)


def cycle(loader):
    while True:
        yield from loader


def f1(tp, fp, fn):
    return 2 * tp / max(1, 2 * tp + fp + fn)


@torch.no_grad()
def frame_f1(model, batches, dev, classes):
    """Frame F1 per class against frame labels, plus 'cut' F1 over the removable classes."""
    tp = np.zeros(len(CLASSES)); fp = np.zeros(len(CLASSES)); fn = np.zeros(len(CLASSES))
    cut = np.zeros(3)
    removable = [CLASSES.index(c) for c in ("sound_rep", "word_rep", "prolongation", "interjection") if
                 CLASSES.index(c) in classes]
    for b in batches:
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            prob = torch.sigmoid(model(b["wave"].to(dev)).float()).cpu().numpy()
        lab, mask = b["labels"].numpy(), b["mask"].numpy()
        t = min(prob.shape[1], lab.shape[1])
        p, y, m = prob[:, :t] > 0.5, lab[:, :t] > 0.5, mask[:, :t, None] > 0.5
        tp += (p & y & m).sum((0, 1)); fp += (p & ~y & m).sum((0, 1)); fn += (~p & y & m).sum((0, 1))
        pc, yc, mc = p[..., removable].any(-1), y[..., removable].any(-1), m[..., 0]
        cut += [(pc & yc & mc).sum(), (pc & ~yc & mc).sum(), (~pc & yc & mc).sum()]
    per = {CLASSES[k]: round(f1(tp[k], fp[k], fn[k]), 3) for k in classes}
    return per, round(f1(*cut), 3)


@torch.no_grad()
def evaluate(model, val_synth, val_sep, val_uclass, dev, batch):
    model.eval()
    res = {}
    synth_batches = [{k: torch.stack([x[k] for x in val_synth[i:i + batch]]) for k in ("wave", "labels", "mask")}
                     for i in range(0, len(val_synth), batch)]
    res["synth_f1"], res["synth_cut_f1"] = frame_f1(model, synth_batches, dev, range(len(CLASSES)))
    if val_uclass is not None:
        res["uclass_f1"], res["uclass_cut_f1"] = frame_f1(
            model, DataLoader(val_uclass, batch_size=batch, num_workers=0), dev, KNOWN)
    if val_sep is not None:
        probs, targets = [], []
        for b in DataLoader(val_sep, batch_size=batch * 2, num_workers=0):
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                probs.append(torch.sigmoid(clip_logits(model(b["wave"].to(dev)).float())).cpu().numpy())
            targets.append(b["clip_labels"].numpy())
        P, Y = np.concatenate(probs), np.concatenate(targets) >= 2 / 3 - 1e-6  # majority of 3 annotators
        res["sep_f1"] = {c: round(f1(((P[:, k] > 0.5) & Y[:, k]).sum(), ((P[:, k] > 0.5) & ~Y[:, k]).sum(),
                                      ((P[:, k] <= 0.5) & Y[:, k]).sum()), 3) for k, c in enumerate(CLASSES)}
        try:
            from sklearn.metrics import roc_auc_score
            res["sep_auc"] = {c: round(float(roc_auc_score(Y[:, k], P[:, k])), 3)
                              for k, c in enumerate(CLASSES) if 0 < Y[:, k].sum() < len(Y)}
        except ImportError:
            pass
    model.train()
    return res


def real_score(m: dict) -> float:
    """Checkpoint selection on real speech only (falls back to synthetic when there is none)."""
    parts = []
    if "sep_auc" in m:
        parts.append(float(np.mean(list(m["sep_auc"].values()))))
    if "uclass_cut_f1" in m:
        parts.append(m["uclass_cut_f1"])
    return float(np.mean(parts)) if parts else m["synth_cut_f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libri-train", required=True, help="glob of librispeech-alignments parquet files")
    ap.add_argument("--libri-val", required=True)
    ap.add_argument("--sep-dir", help="folder with sep28k-fluencybank parquet files")
    ap.add_argument("--silver", nargs="*", default=[], help="words_*.jsonl from crisper_label.py")
    ap.add_argument("--uclass-dir", help="folder with UCLASS wavs (needs --silver)")
    ap.add_argument("--init-ckpt", help="start from an earlier teacher checkpoint")
    ap.add_argument("--backbone", default="microsoft/wavlm-base-plus")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--sep-batch", type=int, default=6)
    ap.add_argument("--uclass-batch", type=int, default=6)
    ap.add_argument("--sep-weight", type=float, default=1.0)
    ap.add_argument("--uclass-weight", type=float, default=1.0)
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

    ddp = "LOCAL_RANK" in os.environ
    rank, world = (int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])) if ddp else (0, 1)
    if ddp:
        local = int(os.environ["LOCAL_RANK"])
        nccl = torch.cuda.is_available() and dist.is_nccl_available()  # NCCL on Linux; gloo elsewhere
        dist.init_process_group("nccl" if nccl else "gloo", timeout=datetime.timedelta(minutes=60))
        dev = torch.device(f"cuda:{local}") if torch.cuda.is_available() else torch.device("cpu")
        if dev.type == "cuda":
            torch.cuda.set_device(dev)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_proc = rank == 0
    log = print if main_proc else (lambda *a, **k: None)
    torch.manual_seed(rank)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_files, val_files = sorted(glob.glob(args.libri_train)), sorted(glob.glob(args.libri_val))
    assert train_files and val_files, "no parquet files matched"

    model = FrameTeacher(args.backbone)
    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        assert ckpt["backbone"] == args.backbone, f"checkpoint backbone is {ckpt['backbone']}"
        model.load_state_dict(ckpt["state_dict"])
        log(f"initialised from {args.init_ckpt} (step {ckpt.get('step')})")
    model.to(dev)
    if args.grad_ckpt:
        model.ssl.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW(model.param_groups(args.lr_backbone, args.lr_head), weight_decay=0.01)
    net = model
    if ddp:
        net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index] if dev.type == "cuda" else None)

    def schedule(step, frozen_until=0):
        if step < frozen_until:
            return 0.0
        s = step - frozen_until
        if s < args.warmup:
            return (s + 1) / args.warmup
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, args.steps - frozen_until))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, [lambda s: schedule(s, args.freeze_steps), schedule])
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")

    def loader(ds, bs, **kw):
        return cycle(DataLoader(ds, batch_size=bs, num_workers=args.workers, drop_last=True,
                                persistent_workers=args.workers > 0, **kw))

    sources = [("synth", loader(SynthStutterStream(train_files, args.crop_s, seed=1000 * rank), args.batch), 1.0)]
    silver = load_silver(args.silver) if args.silver else {}
    val_sep = val_uclass = None
    if args.sep_dir:
        sep_files = sorted(glob.glob(str(Path(args.sep_dir) / "**" / "*.parquet"), recursive=True))
        sep_train = Sep28kClips(sep_files, "train", seed=rank, silver=silver)
        val_sep = Sep28kClips(sep_files, "val", augment_p=0.0)
        val_sep = val_sep if len(val_sep) else None
        log(f"SEP-28k/FluencyBank: {len(sep_train)} train clips ({len(sep_train.groups)} speaker groups, "
            f"{sep_train.n_silver} with silver frame labels), {len(val_sep) if val_sep else 0} val clips")
        sources.append(("sep", loader(sep_train, args.sep_batch, shuffle=True), args.sep_weight))
    if args.uclass_dir and silver:
        uc_train = UclassSilver(silver, args.uclass_dir, "train", args.crop_s, seed=rank)
        val_uclass = UclassSilver(silver, args.uclass_dir, "val", args.crop_s, fixed=True)
        val_uclass = val_uclass if len(val_uclass.audio) else None
        log(f"UCLASS: {len(uc_train.audio)} train recordings ({len(uc_train.speakers)} speakers, "
            f"{uc_train.lengths.sum() / 16000 / 3600:.1f} h), "
            f"{len(val_uclass.audio) if val_uclass else 0} val recordings")
        if len(uc_train.audio):
            sources.append(("uclass", loader(uc_train, args.uclass_batch), args.uclass_weight))
    val_synth = fixed_synth_set(val_files, args.val_n, args.crop_s) if main_proc else []

    if main_proc:
        (out / "config.json").write_text(json.dumps({**vars(args), "classes": CLASSES, "world": world}, indent=2))
    best, t0 = -1.0, time.time()
    log_f = open(out / "metrics.jsonl", "a") if main_proc else None
    net.train()
    for step in range(1, args.steps + 1):
        losses = {}
        for i, (name, it, weight) in enumerate(sources):
            b = next(it)
            last_source = i == len(sources) - 1
            # Gradients from all sources add up locally; DDP syncs once, on the last backward.
            with net.no_sync() if ddp and not last_source else nullcontext():
                with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    logits = net(b["wave"].to(dev))
                    loss = frame_loss(logits, b["labels"].to(dev), b["mask"].to(dev),
                                      class_mask=b["class_mask"].to(dev))
                    if "clip_labels" in b:
                        loss = loss + clip_loss(logits.float(), b["clip_labels"].to(dev))
                scaler.scale(weight * loss).backward()
            losses[name] = loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()

        if step % 25 == 0 or step == 1:
            el = time.time() - t0
            log(f"step {step} " + " ".join(f"{k} {v:.4f}" for k, v in losses.items())
                + f"  {step / el:.2f} it/s  {el / 60:.1f} min", flush=True)
        stop = torch.tensor([step == args.steps or (time.time() - t0) / 3600 > args.max_hours], device=dev)
        if ddp:
            dist.broadcast(stop, 0)  # every process stops on the same step
        if step % args.eval_every == 0 or stop.item():
            if main_proc:
                m = evaluate(model, val_synth, val_sep, val_uclass, dev, args.batch)
                score = real_score(m)
                m.update(step=step, score=round(score, 4), minutes=round((time.time() - t0) / 60, 1))
                print(json.dumps(m), flush=True)
                log_f.write(json.dumps(m) + "\n")
                log_f.flush()
                if score > best:
                    best = score
                    torch.save({"state_dict": model.state_dict(), "backbone": args.backbone, "classes": CLASSES,
                                "step": step, "metrics": m}, out / "best.pt")
            if ddp:
                dist.barrier()
        if stop.item():
            break
    log(f"done. best score {best:.4f}; checkpoint {out / 'best.pt'}")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
