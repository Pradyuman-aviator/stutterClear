"""Measure stutter cutting on held-out recordings, and search for the best cut and join settings.

    python scripts/eval_splice.py prepare --dir data/external/uclass_eval
    python scripts/eval_splice.py grid    --dir data/external/uclass_eval
    python scripts/eval_splice.py render  --dir data/external/uclass_eval --cut v3 --join plain [--full]

prepare: caches teacher frame probabilities (<wav>.teacher.npy) and CrisperWhisper verbatim words
         (<wav>.crisper.json) and intended text (<wav>.intended.txt) for every original recording.
grid:    scores cut settings with silver labels (stutter removed vs words lost), pooled over all
         recordings; copy-agnostic, so keeping a different copy of a repeat is not an error.
render:  writes edited audio to <dir>/_edited/<cut>+<join>/ and scores naturalness (UTMOS) of
         original vs edited; --full also re-transcribes the edited audio with CrisperWhisper to
         count audible stutters left and the change in meaning (intended-transcript WER).
"""
import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stutterclear.audio import load_wav, save_wav  # noqa: E402
from stutterclear.segments import CutConfig, JoinConfig, cut_segments, render_edited  # noqa: E402
from stutterclear.silver import intended_text, label_tokens, score_cuts  # noqa: E402

SR = 16000
CUTS = {
    "v2": CutConfig(thresholds=(0.5,) * 5, keep_last_copy=False, max_cut_s=1e9),
    "v3": CutConfig(keep_last_copy=False, max_cut_s=1e9),
    "v3+last": CutConfig(keep_last_copy=True, max_cut_s=2.0),
}
JOINS = {
    "plain": JoinConfig(xfade_ms=5.0),
    "eqpow": JoinConfig(xfade_ms=10.0),
    "aligned": JoinConfig(xfade_ms=10.0, align_ms=10.0, level_match=True),
    "pause60": JoinConfig(xfade_ms=10.0, pause_ms=60.0, level_match=True),
    "pause120": JoinConfig(xfade_ms=10.0, pause_ms=120.0, level_match=True),
}


def recordings(root: Path) -> list:
    return sorted(p for p in root.rglob("*.wav") if "_edited" not in p.parts)


@torch.no_grad()
def student_probs(student_ckpt: str, wavs: list):
    from stutterclear.student_runtime import load_student
    model = load_student(student_ckpt)
    for w in wavs:
        x = load_wav(w)
        for la, logits in zip(model.lookaheads, model(torch.from_numpy(x)[None])):
            p = torch.sigmoid(logits[0]).numpy()
            aligned = np.zeros_like(p)
            aligned[:len(p) - la] = p[la:]  # head row j describes frame j - lookahead
            np.save(w.with_suffix(f".student_{la * 20}ms.npy"), aligned)
        print("student:", w.name, flush=True)


def prepare(root: Path, ckpt: str, student: str = None):
    wavs = recordings(root)
    if student:
        student_probs(student, wavs)
    todo = [w for w in wavs if not w.with_suffix(".teacher.npy").exists()]
    if todo:
        sys.path.insert(0, str(Path(__file__).parent))
        from eval_teacher import load_teacher
        from teacher_label import frame_probs
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = load_teacher(ckpt, dev)
        for w in todo:
            np.save(w.with_suffix(".teacher.npy"), frame_probs(model, load_wav(w), dev))
            print("teacher:", w.name, flush=True)
        del model
        torch.cuda.empty_cache()
    todo = [w for w in wavs if not w.with_suffix(".crisper.json").exists() or not w.with_suffix(".intended.txt").exists()]
    if todo:
        from crisperwhisper import CrisperWhisperModel
        cw = CrisperWhisperModel("large", backend="transformers")
        for w in todo:
            x = load_wav(w)
            r = cw.transcribe(x, sr=SR, language="en", word_timestamps=True)
            w.with_suffix(".crisper.json").write_text(json.dumps(
                [{"w": t.word, "s": float(t.start), "e": float(t.end)} for t in r.words]))
            w.with_suffix(".intended.txt").write_text(intended_text(cw, x))
            print("crisper:", w.name, flush=True)


def load_set(root: Path, probs: str = "teacher") -> list:
    out = []
    for w in recordings(root):
        out.append({"wav": w, "x": load_wav(w), "probs": np.load(w.with_suffix(f".{probs}.npy")),
                    "toks": label_tokens(json.loads(w.with_suffix(".crisper.json").read_text()))})
    return out


def pooled(rows: list, cfg: CutConfig) -> dict:
    tot = {"n_stutter": 0, "n_removed": 0, "n_fluent": 0, "n_lost": 0}
    per = []
    for r in rows:
        s = score_cuts(r["toks"], cut_segments(r["probs"], cfg, r["x"]))
        for k in tot:
            tot[k] += s[k]
        per.append((r["wav"].name, s["stutter_removed_pct"], s["words_lost_pct"]))
    return {"removed": round(100 * tot["n_removed"] / max(1, tot["n_stutter"]), 1),
            "lost": round(100 * tot["n_lost"] / max(1, tot["n_fluent"]), 2), "per": per}


def grid(root: Path, probs: str = "teacher"):
    rows = load_set(root, probs)
    for name, cfg in CUTS.items():
        p = pooled(rows, cfg)
        print(f"{name:8s} removed {p['removed']:5.1f}%  words lost {p['lost']:5.2f}%")
    thresholds = [(0.5,) * 5, (0.4, 0.35, 0.5, 0.5, 0.4), (0.35, 0.3, 0.45, 0.5, 0.35),
                  (0.3, 0.25, 0.4, 0.5, 0.3), (0.45, 0.4, 0.6, 0.5, 0.45)]
    if probs != "teacher":  # students are less confident: their useful thresholds sit lower
        thresholds += [(0.25,) * 5, (0.2,) * 5, (0.15,) * 5, (0.1,) * 5]
    space = {
        "thresholds": thresholds,
        "off_ratio": [0.75, 0.5, 0.35],
        "merge_gap": [0, 5, 10],
        "snap": [0, 4],
        "refine_long_s": [0.8, 1.2, 1e9],
        "prolong_keep_ms": [0.0, 100.0, 150.0],
    }
    results = []
    for vals in itertools.product(*space.values()):
        cfg = CutConfig(**dict(zip(space.keys(), vals)))
        p = pooled(rows, cfg)
        results.append((p["removed"], p["lost"], cfg))
    results.sort(key=lambda r: (r[1], -r[0]))
    print(f"\n{len(results)} settings. Pareto front (most removal for each level of words lost):")
    best = -1
    front = []
    for removed, lost, cfg in results:
        if removed > best:
            best = removed
            front.append((removed, lost, cfg))
            print(f"  removed {removed:5.1f}%  lost {lost:5.2f}%  thr={cfg.thresholds} off={cfg.off_ratio} "
                  f"merge={cfg.merge_gap} snap={cfg.snap} last={cfg.keep_last_copy} max={cfg.max_cut_s}")
    (root / f"_grid_{probs}.json").write_text(json.dumps(
        [{"removed": r, "lost": l, "cfg": c.__dict__} for r, l, c in front], indent=1, default=list))


def utmos_model():
    return torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()


@torch.no_grad()
def utmos(model, x, chunk_s=10.0):
    n = int(chunk_s * SR)
    dev = next(model.parameters()).device
    scores = [float(model(torch.from_numpy(x[i:i + n].astype(np.float32))[None].to(dev), SR))
              for i in range(0, len(x), n) if len(x[i:i + n]) >= 3 * SR]
    return float(np.mean(scores)) if scores else float("nan")


def words(t: str) -> list:
    return re.sub(r"[^a-z0-9' ]", " ", t.lower()).split()


def wer(ref: list, hyp: list) -> float:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
            prev, d[j] = d[j], cur
    return d[len(hyp)] / max(1, len(ref))


def oracle_cuts(toks: list) -> list:
    """Cuts taken from the verbatim transcript itself: a stand-in for a near-perfect detector,
    to show how far better detection could take the output."""
    cuts = []
    for i, t in enumerate(toks):
        if t["label"] == "fluent":
            continue
        end = t["e"]
        if i + 1 < len(toks) and 0 <= toks[i + 1]["s"] - end <= 0.3:
            end = toks[i + 1]["s"]
        cuts.append({"kind": t["label"], "start": t["s"], "end": end})
    return cuts


def render(root: Path, cut_name: str, join_name: str, full: bool, probs: str = "teacher"):
    cfg = None if cut_name == "oracle" else CUTS[cut_name] if cut_name in CUTS else CutConfig(**json.loads(cut_name))
    jcfg = JOINS[join_name]
    tag = cut_name if cut_name in CUTS or cut_name == "oracle" else "balanced" if '"refine_long_s":0.8' in cut_name else "custom"
    out_dir = root / "_edited" / f"{probs}+{tag}+{join_name}"
    rows = load_set(root, probs)
    mos = utmos_model().to("cuda" if torch.cuda.is_available() else "cpu")
    dev = next(mos.parameters()).device
    cw = None
    if full:
        from crisperwhisper import CrisperWhisperModel
        cw = CrisperWhisperModel("large", backend="transformers")
    table = []
    for r in rows:
        cuts = oracle_cuts(r["toks"]) if cfg is None else cut_segments(r["probs"], cfg, r["x"])
        y = render_edited(r["x"], cuts, jcfg)
        dest = out_dir / r["wav"].relative_to(root)
        save_wav(dest, y)
        row = {"file": r["wav"].name, "cuts": len(cuts), "orig_s": round(len(r["x"]) / SR, 1),
               "edited_s": round(len(y) / SR, 1),
               "utmos_orig": round(utmos(mos.to(dev), r["x"]), 3), "utmos_edit": round(utmos(mos, y), 3)}
        if cw is not None:
            v = cw.transcribe(y, sr=SR, language="en", word_timestamps=True)
            toks = label_tokens([{"w": t.word, "s": float(t.start), "e": float(t.end)} for t in v.words])
            left = sum(t["label"] != "fluent" for t in toks)
            orig = sum(t["label"] != "fluent" for t in r["toks"])
            intended = intended_text(cw, y)
            row.update(stutter_tokens_orig=orig, stutter_tokens_left=left,
                       audible_removed_pct=round(100 * (1 - left / max(1, orig)), 1),
                       meaning_wer=round(wer(words(r["wav"].with_suffix(".intended.txt").read_text()), words(intended)), 3))
        table.append(row)
        print(json.dumps(row), flush=True)
    summary = {"cut": cut_name, "join": join_name,
               "utmos_orig": round(float(np.mean([t["utmos_orig"] for t in table])), 3),
               "utmos_edit": round(float(np.mean([t["utmos_edit"] for t in table])), 3)}
    if full:
        summary["audible_removed_pct"] = round(100 * (1 - sum(t["stutter_tokens_left"] for t in table)
                                                      / max(1, sum(t["stutter_tokens_orig"] for t in table))), 1)
        summary["meaning_wer"] = round(float(np.mean([t["meaning_wer"] for t in table])), 3)
    (out_dir / "summary.json").write_text(json.dumps({"summary": summary, "per_file": table}, indent=1))
    print(json.dumps(summary))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "grid", "render"])
    ap.add_argument("--dir", default="data/external/uclass_eval")
    ap.add_argument("--ckpt", default="runs/kaggle_teacher/teacher/best.pt")
    ap.add_argument("--cut", default="v3+last", help=f"{list(CUTS)} or a CutConfig JSON")
    ap.add_argument("--join", default="aligned", choices=list(JOINS))
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--student", help="student checkpoint: prepare also caches its per-lookahead probabilities")
    ap.add_argument("--probs", default="teacher", help='"teacher" or e.g. "student_500ms"')
    args = ap.parse_args()
    root = Path(args.dir)
    if args.cmd == "prepare":
        prepare(root, args.ckpt, args.student)
    elif args.cmd == "grid":
        grid(root, args.probs)
    else:
        render(root, args.cut, args.join, args.full, args.probs)


if __name__ == "__main__":
    main()
