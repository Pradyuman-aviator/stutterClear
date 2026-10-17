"""Score the live app end to end: each detector and mode streamed 10 ms at a time through the same
pipeline the app runs (delay buffer, online detector, splicer), on held-out recordings, against
silver labels from CrisperWhisper.

    python scripts/eval_live.py --dir data/external/uclass_eval
    python scripts/eval_live.py --dir data/external/uclass_eval --student runs/kaggle_student/student/student.pt

Needs the <wav>.crisper.json files that `eval_splice.py prepare` writes.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_file import run  # noqa: E402
from stutterclear.audio import load_wav  # noqa: E402
from stutterclear.config import SAMPLE_RATE as SR  # noqa: E402
from stutterclear.live import DEFAULT_STUDENT, MODES, make_detector  # noqa: E402
from stutterclear.silver import label_tokens, score_cuts  # noqa: E402

KEYS = ("n_stutter", "n_removed", "n_fluent", "n_lost")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/external/uclass_eval")
    ap.add_argument("--student", default=str(DEFAULT_STUDENT))
    ap.add_argument("--detectors", default="heuristic,student")
    ap.add_argument("--modes", default="gentle,strong")
    ap.add_argument("--out", help="write results as JSON here")
    args = ap.parse_args()

    root = Path(args.dir)
    wavs = sorted(p for p in root.rglob("*.wav") if "_edited" not in p.parts)
    rows = [(w, load_wav(w), label_tokens(json.loads(w.with_suffix(".crisper.json").read_text()))) for w in wavs]
    audio_s = sum(len(x) for _, x, _ in rows) / SR
    print(f"{len(rows)} recordings, {audio_s / 60:.1f} min")

    results = []
    for kind in args.detectors.split(","):
        for mode in args.modes.split(","):
            tot = dict.fromkeys(KEYS, 0)
            removed_s = 0.0
            t0 = time.perf_counter()
            name = None
            for w, x, toks in rows:
                det, name = make_detector(kind, mode, args.student)
                if kind == "student" and name != "student":
                    break  # no checkpoint: make_detector fell back to rule-based
                pipe, _ = run(x, MODES[mode][0], det)
                spans = pipe.splicer.stats.removed_spans
                s = score_cuts(toks, [{"start": a / SR, "end": b / SR} for a, b, _ in spans])
                for k in KEYS:
                    tot[k] += s[k]
                removed_s += sum(b - a for a, b, _ in spans) / SR
            else:
                r = {
                    "detector": name, "mode": mode,
                    "stutter_removed_pct": round(100 * tot["n_removed"] / max(1, tot["n_stutter"]), 1),
                    "words_lost_pct": round(100 * tot["n_lost"] / max(1, tot["n_fluent"]), 2),
                    "removed_s": round(removed_s, 1),
                    "realtime_factor": round((time.perf_counter() - t0) / audio_s, 3),
                    **tot,
                }
                results.append(r)
                print(f"{name:10s} {mode:7s} removed {r['stutter_removed_pct']:5.1f}%  "
                      f"words lost {r['words_lost_pct']:5.2f}%  cut {r['removed_s']:6.1f} s  "
                      f"rtf {r['realtime_factor']:.3f}", flush=True)
                continue
            print(f"{kind}: no checkpoint at {args.student}, skipped")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
