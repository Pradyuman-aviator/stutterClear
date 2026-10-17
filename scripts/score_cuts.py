"""Score a cut list against an automatic verbatim transcript (CrisperWhisper 2.0, research licence).

    python scripts/score_cuts.py data/external/uclass/M_0065_20y1m_1.16k.wav --cuts <wav>.cuts.json

The verbatim transcript (cached as <wav>.crisper.json) marks every filler ([UM], [UH]), every
cut-off word ("l-") and every repeated word. Tokens are labelled from it:
- filler, cutoff:     always stutter
- repeat:             every copy of a repeated word or 2-3 word phrase except the last
- fluent:             everything else, which must survive
Then each token's overlap with the cut list gives stutter removal and wrongly cut speech.
These are silver labels: an automatic stand-in until hand labels exist.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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


def norm(w: str) -> str:
    return re.sub(r"[^a-z'\-\[\]]", "", w.lower())


def label_tokens(words: list) -> list:
    toks = [dict(w, n=norm(w["w"])) for w in words]
    for t in toks:
        t["label"] = ("filler" if t["n"].startswith("[") else
                      "cutoff" if t["n"].endswith("-") else "fluent")
    content = [i for i, t in enumerate(toks) if t["label"] == "fluent"]
    # Repeated words and short phrases, ignoring fillers and cut-offs between copies:
    # every copy but the last is a repetition.
    for n in (3, 2, 1):
        k = 0
        while k + 2 * n <= len(content):
            a = [toks[i]["n"] for i in content[k:k + n]]
            b = [toks[i]["n"] for i in content[k + n:k + 2 * n]]
            if a == b and all(toks[i]["label"] == "fluent" for i in content[k:k + n]):
                for i in content[k:k + n]:
                    toks[i]["label"] = "repeat"
            k += 1
    return toks


def overlap(s, e, cuts):
    return sum(max(0.0, min(e, c["end"]) - max(s, c["start"])) for c in cuts) / max(1e-3, e - s)


def score(toks: list, cuts: list) -> dict:
    out = {}
    for kind in ("filler", "cutoff", "repeat"):
        ts = [t for t in toks if t["label"] == kind]
        if ts:
            removed = [overlap(t["s"], t["e"], cuts) for t in ts]
            out[kind] = {"n": len(ts), "removed_pct": round(100 * sum(r >= 0.5 for r in removed) / len(ts))}
    stut = [t for t in toks if t["label"] != "fluent"]
    out["all_stutter_removed_pct"] = round(100 * sum(overlap(t["s"], t["e"], cuts) >= 0.5 for t in stut)
                                           / max(1, len(stut)))
    fluent = [t for t in toks if t["label"] == "fluent"]
    hit = [t for t in fluent if overlap(t["s"], t["e"], cuts) >= 0.3]
    out["fluent_words"] = len(fluent)
    out["fluent_words_cut_pct"] = round(100 * len(hit) / max(1, len(fluent)), 1)
    out["wrongly_cut_examples"] = [f"{t['w']}@{t['s']:.1f}" for t in hit[:12]]
    return out


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
        print(json.dumps(score(toks, cuts), indent=1))


if __name__ == "__main__":
    main()
