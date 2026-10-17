"""Score a cut list against an automatic verbatim transcript (CrisperWhisper 2.0, research licence).

    python scripts/score_cuts.py data/external/uclass/M_0065_20y1m_1.16k.wav --cuts <wav>.cuts.json

The verbatim transcript (cached as <wav>.crisper.json) marks every filler ([UM], [UH]), every
cut-off word ("l-") and every repeated word. Tokens are labelled from it:
- filler, cutoff:     always stutter
- repeat:             every copy of a repeated word or 2-3 word phrase except the last
- fluent:             everything else, which must survive
Then each token's overlap with the cut list gives stutter removal and words lost; a repeat
chain counts as fixed when one copy survives, whichever copy that is.
These are silver labels: an automatic stand-in until hand labels exist.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stutterclear.silver import label_tokens, score_cuts  # noqa: E402


def transcribe(wav: Path) -> list:
    cache = wav.with_suffix(".crisper.json")
    if cache.exists():
        return json.loads(cache.read_text())
    from crisperwhisper import CrisperWhisperModel
    model = CrisperWhisperModel("large", backend="transformers")
    r = model.transcribe(str(wav), language="en", word_timestamps=True)
    words = [{"w": w.word, "s": float(w.start), "e": float(w.end)} for w in r.words]
    cache.write_text(json.dumps(words))
    return words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="the ORIGINAL recording the cuts refer to")
    ap.add_argument("--cuts", nargs="+", required=True, help="cut list json(s): [{start, end, kind}]")
    args = ap.parse_args()
    toks = label_tokens(transcribe(Path(args.wav)))
    counts = {k: sum(t["label"] == k for t in toks) for k in ("filler", "cutoff", "repeat", "fluent")}
    print(f"silver labels from verbatim transcript: {counts}")
    for path in args.cuts:
        cuts = json.loads(Path(path).read_text())
        print(f"\n{Path(path).name}: {len(cuts)} cuts, {sum(c['end'] - c['start'] for c in cuts):.1f} s")
        print(json.dumps(score_cuts(toks, cuts), indent=1))


if __name__ == "__main__":
    main()
