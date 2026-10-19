"""Demo clips that are safe to publish: real read speech from LibriSpeech (CC BY 4.0) with stutters
inserted by StutterClear's generator, labelled as synthetic wherever they show. A public server
plays these instead of the UCLASS clips (research licence).

    python scripts/make_synthetic_demo.py [--n 3] [--seconds 28] [--seed 7]

Writes data/demo_clips/<id>.wav with the caches the offline engines read (<id>.crisper.json, the
verbatim words; <id>.teacher.npy, the recordings teacher's probabilities) and <id>.labels.json
(what was inserted where), and adds the clips to stutterclear/app/demo_samples.json. Uses the GPU
for CrisperWhisper: on a 6 GB card stop the app first. The speech is LibriSpeech dev-clean, the
generator's validation set, which no model was trained on.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stutterclear.audio import save_wav  # noqa: E402
from stutterclear.synth import CLASSES, SR, SynthConfig, Utterance, _room_tone, synthesize  # noqa: E402

LIBRI = ROOT / "data" / "librispeech-alignments" / "data"
OUT = ROOT / "data" / "demo_clips"
SAMPLES = ROOT / "stutterclear" / "app" / "demo_samples.json"
# Every utterance stutters (a demo shows what the engines do), a little more often than in training.
DEMO = SynthConfig(p_fluent_utterance=0.0, p_sound_rep=0.12, p_word_rep=0.08, p_prolongation=0.06, p_block=0.03,
                   p_pause=0.06, p_rapid_rep=0.05, p_phrase_rep=0.05, p_filler=0.07, p_cluster=0.03)


def chapters():
    """dev-clean utterances grouped by speaker and chapter, in reading order, with the speaker's sex."""
    rows = defaultdict(list)
    for f in sorted(LIBRI.glob("dev_clean-*.parquet")):
        for r in pq.read_table(f, columns=["id", "sex"]).to_pylist():
            spk, chap, _ = r["id"].split("-")
            rows[(spk, chap)].append((r["id"], r["sex"]))
    return {k: sorted(v) for k, v in rows.items()}


def load(ids: set) -> dict:
    """The full rows (audio, word and phone alignments) of these utterances."""
    import pyarrow as pa
    import pyarrow.compute as pc
    out = {}
    for f in sorted(LIBRI.glob("dev_clean-*.parquet")):
        t = pq.read_table(f)
        for r in t.filter(pc.is_in(t["id"], value_set=pa.array(sorted(ids)))).to_pylist():
            out[r["id"]] = r
    return out


def make_clip(rows: list, seconds: float, rng) -> tuple:
    """Consecutive utterances of one reader, each with inserted stutters, joined by short pauses of
    the recording's own room tone, up to `seconds`. Returns (audio, labels)."""
    parts, labels, pos = [], [], 0
    for row in rows:
        utt = Utterance.from_row(row)
        y, spans, _ = synthesize(utt, rng, DEMO)
        tone = _room_tone(utt.audio)
        pause = np.resize(tone, int(rng.uniform(0.35, 0.6) * SR)) if len(tone) else np.zeros(int(0.4 * SR), np.float32)
        labels += [{"start": round((pos + a) / SR, 3), "end": round((pos + b) / SR, 3), "kind": CLASSES[c]} for a, b, c in spans]
        parts += [y, pause]
        pos += len(y) + len(pause)
        if pos >= seconds * SR:
            break
    x = np.concatenate(parts).astype(np.float32)
    return x * (0.9 / max(1e-6, float(np.abs(x).max()))), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=28.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # Readers with enough consecutive text, alternating women and men.
    groups = [(k, v) for k, v in chapters().items() if len(v) >= 12]
    order = rng.permutation(len(groups))
    picked, want = [], ["F", "M"]
    for i in order:
        (spk, chap), utts = groups[i]
        if utts[0][1] == want[len(picked) % 2] and spk not in {p[0][0] for p in picked}:
            picked.append(((spk, chap), utts))
        if len(picked) == args.n:
            break
    rows = load({uid for _, utts in picked for uid, _ in utts[:12]})

    from stutterclear.app import notes  # CrisperWhisper (lean, GPU if it fits) and the recordings teacher
    OUT.mkdir(parents=True, exist_ok=True)
    samples = [s for s in json.loads(SAMPLES.read_text()) if s.get("source") != "synthetic"]
    for k, ((spk, chap), utts) in enumerate(picked, 1):
        clip = f"read-{k}"
        x, labels = make_clip([rows[uid] for uid, _ in utts[:12]], args.seconds, rng)
        save_wav(OUT / f"{clip}.wav", x)
        (OUT / f"{clip}.labels.json").write_text(json.dumps({"librispeech": [uid for uid, _ in utts[:12]],
                                                             "inserted": labels}, indent=1))
        r = notes._crisper().transcribe(x, sr=SR, language="en", word_timestamps=True)
        words = [{"w": t.word, "s": round(float(t.start), 3), "e": round(float(t.end), 3)} for t in r.words]
        (OUT / f"{clip}.crisper.json").write_text(json.dumps(words))
        np.save(OUT / f"{clip}.teacher.npy", notes.teacher_probs(x))
        sex = "woman" if utts[0][1] == "F" else "man"
        samples.append({"id": clip, "title": "Read aloud", "who": f"{sex}, synthetic stutters", "file": f"{clip}.wav",
                        "start": 0.0, "end": round(len(x) / SR, 2), "source": "synthetic"})
        print(f"{clip}: LibriSpeech reader {spk} ({sex}), {len(x) / SR:.1f} s, {len(labels)} inserted stutters, "
              f"{len(words)} words heard", flush=True)
    SAMPLES.write_text(json.dumps(samples, indent=2) + "\n")
    print(f"{len(picked)} clips in {OUT}; demo_samples.json updated")


if __name__ == "__main__":
    main()
