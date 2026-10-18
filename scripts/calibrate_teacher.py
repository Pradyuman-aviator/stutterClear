"""Calibrate a teacher checkpoint on validation speakers, so it can be compared fairly with another.

A new teacher can be more (or less) confident than the one the cut settings were tuned for; with the
old thresholds it then over- or under-cuts. This tunes, on the UCLASS validation speakers only (never
the 7 test speakers in data/external/uclass_eval):

- live Gentle and Strong: per-class thresholds, off_ratio and merge_gap of the online cut logic,
  maximising stutters removed minus 3 x fluent words lost on sliding-window probabilities;
- recordings: the acoustic-check threshold of word_cuts, set so the new teacher flags fluent words
  exactly as often as the reference teacher does at 0.3 (same strictness on normal speech).

The result is written next to the checkpoint as live_cuts.json, which the app reads when present.

    python scripts/calibrate_teacher.py --ckpt runs/kaggle_teacher_v2/teacher_v2/best.pt --download
"""
import argparse
import dataclasses
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from download_uclass import BASE, FOLDERS  # noqa: E402
from eval_teacher import load_teacher  # noqa: E402
from teacher_lookahead import streamed_probs  # noqa: E402
from stutterclear.audio import load_wav  # noqa: E402
from stutterclear.live import STRONG_CUTS, TEACHER_LOOKAHEAD_MS  # noqa: E402
from stutterclear.segments import BALANCED_CUTS  # noqa: E402
from stutterclear.silver import label_tokens, score_cuts  # noqa: E402
from stutterclear.student_runtime import CLASSES, REMOVABLE, StudentDetector  # noqa: E402
from stutterclear.teacher_runtime import HOP_MS  # noqa: E402
from stutterclear.train_data import uclass_speaker, uclass_split  # noqa: E402

SR, FRAME = 16000, 320
REFERENCE = ROOT / "runs" / "kaggle_teacher" / "teacher" / "best.pt"
SILVER = sorted((ROOT / "runs" / "kaggle_silver" / "silver").glob("words_*.jsonl"))
GATE = [0, 1, 2, 4]  # the classes word_cuts checks (sound_rep, word_rep, prolongation, interjection)


def download_val(root: Path):
    """The UCLASS recordings of validation speakers, with their verbatim word timings."""
    recs = []
    for p in SILVER:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r["source"] == "uclass" and uclass_split(r.get("speaker") or uclass_speaker(r["file"])) == "val":
                recs.append(r)

    def get(r):
        task, name = r["file"].split("/", 1)
        dest = root / task / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            try:
                dest.write_bytes(urllib.request.urlopen(BASE + FOLDERS[task] + name, timeout=120).read())
            except Exception as e:  # one missing file must not stop the rest
                return f"skipped {r['file']}: {e}"
        dest.with_suffix(".crisper.json").write_text(json.dumps(r["words"]))
        return None

    with ThreadPoolExecutor(6) as ex:
        errors = [e for e in ex.map(get, recs) if e]
    print(f"{len(recs) - len(errors)} validation recordings in {root}", *errors, sep="\n")


class OnlineCuts(StudentDetector):
    """The live cut logic (hysteresis, merging, refinement) run over precomputed probabilities."""

    def __init__(self, cfg, cut_interjections):
        self.cfg = cfg
        self.thresholds = np.array(cfg.thresholds, float)
        removable = REMOVABLE + (("interjection",) if cut_interjections else ())
        self.cut_idx = [CLASSES.index(c) for c in removable]
        self.thresholds[self.cut_idx] = np.maximum(self.thresholds[self.cut_idx], self.MIN_CUT_THRESHOLD)
        self.close_after = max(2, cfg.merge_gap)
        self.refine_frames = cfg.refine_long_s * SR / FRAME
        self.keep = int(round(cfg.prolong_keep_ms * SR / 1000 / FRAME))
        self.tail = int(round(cfg.prolong_tail_ms * SR / 1000 / FRAME))
        self.block_idx = CLASSES.index("block")
        self.events, self.next_id, self.frames_seen = [], 0, 0

    def run(self, probs):
        self.cut, self.cooldown = None, False
        done = {}
        for f, p in enumerate(probs):
            updates = []
            self._cut_frame(f, p, updates)
            for u in updates:
                if u.op == "cancel":
                    done.pop(u.cut_id, None)
                elif u.end is not None:
                    done[u.cut_id] = u
        return [{"start": u.start / SR, "end": u.end / SR} for u in done.values()]


def score(rows, cfg, cut_interjections):
    tot = dict(n_stutter=0, n_removed=0, n_fluent=0, n_lost=0)
    for probs, toks in rows:
        q = score_cuts(toks, OnlineCuts(cfg, cut_interjections).run(probs))
        for k in tot:
            tot[k] += q[k]
    removed = 100 * tot["n_removed"] / max(1, tot["n_stutter"])
    lost = 100 * tot["n_lost"] / max(1, tot["n_fluent"])
    return removed - 3 * lost, removed, lost


def tune(rows, base, cut_interjections, starts):
    """Coordinate search: one setting at a time, a few passes, from each starting point."""
    classes = [CLASSES.index(c) for c in REMOVABLE + (("interjection",) if cut_interjections else ())]
    best = None
    for start in starts:
        cfg = dataclasses.replace(base, thresholds=tuple(start))
        cur = score(rows, cfg, cut_interjections)
        for _ in range(3):
            improved = False
            for c in classes:
                for v in np.round(np.arange(0.2, 0.86, 0.05), 2):
                    th = list(cfg.thresholds)
                    th[c] = float(v)
                    cand = dataclasses.replace(cfg, thresholds=tuple(th))
                    s = score(rows, cand, cut_interjections)
                    if s[0] > cur[0] + 1e-9:
                        cfg, cur, improved = cand, s, True
            for field, values in (("off_ratio", (0.25, 0.35, 0.5, 0.65)), ("merge_gap", (3, 5, 8))):
                for v in values:
                    cand = dataclasses.replace(cfg, **{field: v})
                    s = score(rows, cand, cut_interjections)
                    if s[0] > cur[0] + 1e-9:
                        cfg, cur, improved = cand, s, True
            if not improved:
                break
        print(f"  from {[round(t, 2) for t in start]}: score {cur[0]:.1f} (removed {cur[1]:.1f}%, lost {cur[2]:.2f}%)",
              flush=True)
        if best is None or cur[0] > best[1][0]:
            best = (cfg, cur)
    return best


def fluent_alarm_rate(rows_offline, confirm):
    """Share of fluent words the acoustic check would flag (max probability near the word >= confirm)."""
    flagged = total = 0
    for probs, toks in rows_offline:
        p = probs[:, GATE]
        for t in toks:
            if t["label"] != "fluent":
                continue
            a = max(0, int((t["s"] - 0.1) / 0.02))
            b = min(len(p), int((t["e"] + 0.1) / 0.02) + 1)
            if b > a:
                total += 1
                flagged += float(p[a:b].max()) >= confirm
    return flagged / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--reference", default=str(REFERENCE), help="teacher whose recording-check strictness to match")
    ap.add_argument("--val-dir", default="data/external/uclass_val")
    ap.add_argument("--cache", default="runs/eval/calibration_cache")
    ap.add_argument("--download", action="store_true", help="fetch the validation recordings first")
    args = ap.parse_args()

    val, cache = Path(args.val_dir), Path(args.cache)
    if args.download:
        download_val(val)
    wavs = sorted(p for p in val.rglob("*.wav") if p.with_suffix(".crisper.json").exists())
    assert wavs, f"no validation recordings in {val} (run with --download)"
    test = {p.stem for p in (ROOT / "data" / "external" / "uclass_eval").rglob("*.wav")}
    assert not test & {w.stem for w in wavs}, "a test recording is in the validation set"
    toks = {w: label_tokens(json.loads(w.with_suffix(".crisper.json").read_text())) for w in wavs}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from stutterclear.app.notes import teacher_probs

    def cached(tag, ckpt, fn):
        out = {}
        model = None
        for w in wavs:
            # UCLASS reuses file names across task folders for different recordings: key by both
            f = cache / Path(ckpt).parent.name / tag / f"{w.parent.name}__{w.stem}.npy"
            if not f.exists():
                if model is None:
                    model, _ = load_teacher(ckpt, dev)
                f.parent.mkdir(parents=True, exist_ok=True)
                np.save(f, fn(model, load_wav(w)))
            out[w] = np.load(f)
        return out

    print(f"{len(wavs)} validation recordings ({sum(len(t) for t in toks.values())} words)")
    result = {"checkpoint": str(Path(args.ckpt).as_posix()), "calibrated_on": f"{len(wavs)} UCLASS validation recordings"}
    for mode, base, cut_fillers in (("gentle", BALANCED_CUTS, False), ("strong", STRONG_CUTS, True)):
        la = TEACHER_LOOKAHEAD_MS[mode] / 1000
        probs = cached(f"stream{int(la * 1000)}", args.ckpt,
                       lambda m, x: streamed_probs(m, x, dev, la, 4.0, HOP_MS / 1000))
        rows = [(probs[w], toks[w]) for w in wavs]
        before = score(rows, base, cut_fillers)
        print(f"{mode}: with the current settings: score {before[0]:.1f} (removed {before[1]:.1f}%, lost {before[2]:.2f}%)")
        # Starting points: the current settings, and the SEP-28k-E validation tuning of v2. Only the
        # classes that are cut are tuned; the block threshold (which triggers word rescue, never a
        # cut) stays as it is.
        sep = [0.36, 0.64, 0.41, 0.25, 0.65]
        sep[CLASSES.index("block")] = base.thresholds[CLASSES.index("block")]
        starts = [list(base.thresholds), sep]
        cfg, (s, removed, lost) = tune(rows, base, cut_fillers, starts)
        result[mode] = {"thresholds": [round(float(t), 2) for t in cfg.thresholds], "off_ratio": cfg.off_ratio,
                        "merge_gap": cfg.merge_gap,
                        "validation": {"removed_pct": round(removed, 1), "words_lost_pct": round(lost, 2),
                                       "before": {"removed_pct": round(before[1], 1), "words_lost_pct": round(before[2], 2)}}}
        print(f"{mode}: calibrated {result[mode]}")

    offline_new = cached("offline", args.ckpt, lambda m, x: teacher_probs(x, m, dev))
    offline_ref = cached("offline", args.reference, lambda m, x: teacher_probs(x, m, dev))
    target = fluent_alarm_rate([(offline_ref[w], toks[w]) for w in wavs], 0.3)
    rows_new = [(offline_new[w], toks[w]) for w in wavs]
    grid = np.round(np.arange(0.3, 0.96, 0.01), 2)
    confirm = float(min(grid, key=lambda c: abs(fluent_alarm_rate(rows_new, c) - target)))
    result["confirm"] = confirm
    result["confirm_note"] = (f"flags {100 * fluent_alarm_rate(rows_new, confirm):.1f}% of fluent words, as the "
                              f"reference does at 0.3 ({100 * target:.1f}%)")
    print("recordings check:", confirm, result["confirm_note"])

    out = Path(args.ckpt).with_name("live_cuts.json")
    out.write_text(json.dumps(result, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
