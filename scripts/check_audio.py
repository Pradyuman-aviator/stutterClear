"""Check cleaned audio the way a listening test would: what was said, what was lost, and whether it
sounds broken. Compares an original recording with its cleaned version.

    python scripts/check_audio.py --demo                 # the app's demo samples, every engine
    python scripts/check_audio.py original.wav out.wav   # any pair, e.g. a saved session

Reports per version:
- transcript checks (CrisperWhisper verbatim): stutter tokens left, real words kept, and the
  change in meaning (WER of the intended transcript against the original's)
- artefacts: dropouts (a gap inside speech, < 40 ms), digital silence (exact zeros where the
  room had noise: sounds like a gate opening and closing), clicks (sample jumps far above the
  local level)
- naturalness: UTMOS (higher is better)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stutterclear.audio import load_wav  # noqa: E402
from stutterclear.silver import label_tokens  # noqa: E402

SR = 16000


def frames_db(x, hop=80):  # 5 ms frames
    n = len(x) // hop
    return 10 * np.log10(np.mean(x[: n * hop].astype(np.float64).reshape(n, hop) ** 2, axis=1) + 1e-12)


def artefacts(y: np.ndarray) -> dict:
    e = frames_db(y)
    # Dropout: a dip of at least 25 dB below both neighbours, lasting at most 40 ms, inside speech.
    dropouts, i = 0, 2
    while i < len(e) - 10:
        if e[i] < e[i - 2] - 25 and e[i - 2] > -40:
            j = i
            while j < len(e) and e[j] < e[i - 2] - 25:
                j += 1
            if j - i <= 8 and j + 2 < len(e) and e[j + 1] > -40:
                dropouts += 1
            i = j + 1
        else:
            i += 1
    zeros = float(np.mean(e < -95))  # exact silence: a real mic never gets there
    d = np.abs(np.diff(y.astype(np.float64)))
    local = np.convolve(d, np.ones(160) / 160, mode="same") + 1e-6
    clicks = int(np.sum((d > 12 * local) & (d > 0.05)))
    return {"dropouts": dropouts, "digital_silence_pct": round(100 * zeros, 1), "clicks": clicks}


class Ears:
    """CrisperWhisper (verbatim and intended) and UTMOS, loaded once."""

    def __init__(self):
        import torch
        from crisperwhisper import CrisperWhisperModel
        from eval_splice import utmos, utmos_model
        self.cw = CrisperWhisperModel("large", backend="transformers")
        self.mos = utmos_model().to("cuda" if torch.cuda.is_available() else "cpu")
        self._utmos = utmos

    def listen(self, x: np.ndarray) -> dict:
        r = self.cw.transcribe(x, sr=SR, language="en", word_timestamps=True)
        toks = label_tokens([{"w": t.word, "s": float(t.start), "e": float(t.end)} for t in r.words])
        intended = self.cw.transcribe(x, sr=SR, language="en", mode="intended").text
        return {"verbatim": " ".join(t["w"] for t in toks), "intended": intended,
                "stutter_tokens": sum(t["label"] != "fluent" for t in toks),
                "fluent_words": sum(t["label"] == "fluent" for t in toks),
                "utmos": round(self._utmos(self.mos, x), 3)}


def words(t: str) -> list:
    return [w for w in "".join(c.lower() if c.isalnum() or c in " '" else " " for c in t).split() if w]


def wer(ref: list, hyp: list) -> float:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)] / max(1, len(ref))


def compare(ears: Ears, name: str, original: np.ndarray, versions: dict) -> list:
    base = ears.listen(original)
    rows = [{"sample": name, "version": "original", **artefacts(original), **base}]
    for vname, y in versions.items():
        h = ears.listen(y)
        rows.append({"sample": name, "version": vname, **artefacts(y), **h,
                     "stutters_removed_pct": round(100 * (1 - h["stutter_tokens"] / max(1, base["stutter_tokens"])), 1),
                     "meaning_wer": round(wer(words(base["intended"]), words(h["intended"])), 3)})
    return rows


def demo_versions(sid: str) -> tuple:
    """Original, plus the cleaned audio as the app produced it before and after the smooth joins."""
    from stutterclear.app import demo
    from stutterclear.segments import BALANCED_CUTS, JoinConfig, cut_segments, render_edited
    x = demo._excerpt(sid)
    s = demo.SAMPLES[sid]
    probs = np.load(demo._wav(s).with_suffix(".teacher.npy"))
    f0 = int(round(s["start"] * SR / demo.TEACHER_FRAME))
    probs = probs[f0: f0 + len(x) // demo.TEACHER_FRAME]
    old = render_edited(x, cut_segments(probs, BALANCED_CUTS, x), JoinConfig(xfade_ms=10.0))
    return x, {"thorough, old joins": old, "thorough, smooth": demo.render(sid, "thorough")["clean"],
               "natural": demo.render(sid, "natural")["clean"], "live": demo.render(sid, "live")["clean"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="original.wav cleaned.wav")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default="runs/eval/audio_check.json")
    args = ap.parse_args()
    ears = Ears()
    rows = []
    if args.demo:
        from stutterclear.app import demo
        for s in demo.available():
            x, versions = demo_versions(s["id"])
            rows += compare(ears, s["id"], x, versions)
    if args.files:
        rows += compare(ears, Path(args.files[0]).parent.name, load_wav(args.files[0]),
                        {"cleaned": load_wav(args.files[1])})
    print(f"\n{'sample':13s} {'version':20s} {'removed':>8} {'meaning':>8} {'words':>6} {'drop':>5} {'zeros%':>7} {'click':>6} {'utmos':>6}")
    for r in rows:
        print(f"{r['sample']:13s} {r['version']:20s} {str(r.get('stutters_removed_pct', '')):>8} "
              f"{str(r.get('meaning_wer', '')):>8} {r['fluent_words']:6d} {r['dropouts']:5d} "
              f"{r['digital_silence_pct']:7.1f} {r['clicks']:6d} {r['utmos']:6.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=1))
    print(f"\ntranscripts in {args.out}")


if __name__ == "__main__":
    main()
