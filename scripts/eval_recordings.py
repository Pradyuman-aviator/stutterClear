"""Score the recording engines (voice notes, "Try a real recording") on held-out recordings: the
offline teacher plus word-aligned cuts from the verbatim transcript, against silver labels.

    python scripts/eval_recordings.py --dir data/external/uclass_eval --out runs/eval/recordings.json

The cuts come from the same transcript as the labels, so "lost" here only catches cuts that spill
into a neighbouring word; whether a repeat was meant is the meaning guard's job (Nemotron), which
this script does not call. Needs the <wav>.crisper.json files that `eval_splice.py prepare` writes.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stutterclear.audio import load_wav  # noqa: E402
from stutterclear.silver import label_tokens, score_cuts, word_cuts  # noqa: E402

RECIPES = {  # name -> word_cuts options; the app's engines are "natural" and "thorough"
    "natural": {},
    "natural + fillers": {"cut_fillers": True},
    "thorough": {"cut_fillers": True, "cut_phrases": True},
    "transcript only, no acoustic check": {"cut_fillers": True, "no_probs": True},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/external/uclass_eval")
    ap.add_argument("--cache", help="keep teacher probabilities here between runs")
    ap.add_argument("--out", help="write results as JSON here")
    args = ap.parse_args()

    from stutterclear.app.notes import teacher_probs
    from stutterclear.teacher_runtime import RECORDINGS_CKPT, calibration
    confirm = calibration(RECORDINGS_CKPT).get("confirm", 0.3)  # the checkpoint's calibrated check, if any
    print(f"recordings teacher {RECORDINGS_CKPT}, acoustic check threshold {confirm}")
    root = Path(args.dir)
    wavs = sorted(p for p in root.rglob("*.wav") if "_edited" not in p.parts)
    rows = []
    for w in wavs:
        toks = label_tokens(json.loads(w.with_suffix(".crisper.json").read_text()))
        # keyed by folder too: UCLASS reuses file names across task folders for different recordings
        cached = Path(args.cache) / f"{w.parent.name}__{w.stem}.npy" if args.cache else None
        if cached and cached.exists():
            probs = np.load(cached)
        else:
            probs = teacher_probs(load_wav(w))
            if cached:
                cached.parent.mkdir(parents=True, exist_ok=True)
                np.save(cached, probs)
        rows.append((toks, probs[:, [0, 1, 2, 4]]))  # sound_rep, word_rep, prolongation, interjection
    print(f"{len(rows)} recordings")

    results = []
    for name, opts in RECIPES.items():
        opts = dict(opts)
        no_probs = opts.pop("no_probs", False)
        tot = {"n_stutter": 0, "n_removed": 0, "n_fluent": 0, "n_lost": 0}
        for toks, probs in rows:
            s = score_cuts(toks, word_cuts(toks, None if no_probs else probs, confirm=confirm, **opts))
            for k in tot:
                tot[k] += s[k]
        r = {"recipe": name,
             "stutter_removed_pct": round(100 * tot["n_removed"] / max(1, tot["n_stutter"]), 1),
             "words_lost_pct": round(100 * tot["n_lost"] / max(1, tot["n_fluent"]), 2), **tot}
        results.append(r)
        print(f"{name:36s} removed {r['stutter_removed_pct']:5.1f}%  words lost {r['words_lost_pct']:5.2f}%")
    if args.out:
        Path(args.out).write_text(json.dumps({"results": results}, indent=1))


if __name__ == "__main__":
    main()
