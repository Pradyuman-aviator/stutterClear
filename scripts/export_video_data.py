"""Everything the demo film shows, from the real pipeline, and its narration from the storyboard.

    python scripts/export_video_data.py            # writes video/public/generated/
    python scripts/make_video_sound.py             # music and sound effects (once)
    cd video && npm run render                      # renders out/stutterclear.mp4

The clip is Alex's voice in the film: a real recording of a 20-year-old man who stutters, from
UCLASS (research and teaching use; credit UCLASS and the Wellcome Trust; it is read from the local
download in data/external and never committed). It is cleaned by the Thorough recordings engine,
exactly as the app does: the teacher's probabilities, CrisperWhisper's verbatim words, cuts between
words, the meaning guard (NVIDIA Nemotron), long silent pauses shortened. Word rescue and the meaning
guard scenes show Nemotron's real answers. Narration: NVIDIA Magpie TTS with NVIDIA_API_KEY,
otherwise a placeholder Windows voice. Needs NEBIUS_API_KEY and the GPU-free cached probabilities.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stutterclear import guard, nemotron, rescue, tts  # noqa: E402
from stutterclear.audio import load_wav, save_wav  # noqa: E402
from stutterclear.config import HOP, SAMPLE_RATE as SR  # noqa: E402
from stutterclear.segments import NATURAL_JOIN, render_edited  # noqa: E402
from stutterclear.silver import label_tokens, score_cuts, tighten_pauses  # noqa: E402
from stutterclear.teacher_runtime import RECORDINGS_CKPT, calibration  # noqa: E402

OUT = ROOT / "video" / "public" / "generated"
STORYBOARD = ROOT / "video" / "storyboard.json"
CLIP = ROOT / "data" / "external" / "uclass_eval" / "monologue" / "M_0065_20y1m_1.wav"
WINDOW = (48.85, 68.30)  # "It is because they they they have it ... in Roman l- l- l- letters"
CREDIT = ("Alex is fictional. His voice is a real recording of a 20-year-old man who stutters, from UCLASS, "
          "University College London (Howell, Davis & Bartrip, 2009); data collection supported by the Wellcome Trust.")
CLASSES = ["sound repetition", "word repetition", "prolongation", "block", "filler"]
NARRATION_SR = 44100
# The narrator speaks with another Magpie voice than Say it (the app's Aria), so the two never mix up.
NARRATOR_VOICE = os.environ.get("MAGPIE_NARRATOR", "Magpie-Multilingual.EN-US.Mia")
# The meaning guard on a sentence in Alex's story (not one of the examples in its prompt), and word
# rescue where the real recording gets stuck ("in Roman l- l- l- letters"): his app teaches
# travellers to read Japanese train signs (kanji, and romaji: Japanese in Roman letters).
GUARD_SENTENCE = "I I know that that sign means the exit, and it is very very easy to miss."  # judged alike 5 times in 5
RESCUE_SAID = "It is because they have it in their Chinese writing there, and some and sometimes in Roman"
RESCUE_CONTEXT = "Demo day pitch for my app: it teaches travellers to read Japanese train signs."
SAY_WORD = "letters"  # the suggestion Alex taps


def guard_demo(text: str, runs: int = 3) -> dict:
    """The meaning guard's real exchange with Nemotron for a typed sentence: what is sent (text,
    candidates marked) and the JSON that comes back, with each word's verdict. Nemotron does not
    always give the same verdict twice, so it is asked `runs` times and a real reply carrying the
    most common verdict is shown (with the tally)."""
    words = text.split()
    toks = label_tokens([{"w": guard.TYPED_FILLERS.get(w.lower().strip(",.!?;:"), w), "s": 0.4 * i, "e": 0.4 * i + 0.3}
                         for i, w in enumerate(words)])
    candidates = {i for i, t in enumerate(toks) if t["label"] != "fluent"}
    request = guard.marked_transcript(toks, candidates)
    replies = []
    for _ in range(runs):
        reply = nemotron.chat([{"role": "system", "content": guard.SYSTEM}, {"role": "user", "content": request}],
                              max_tokens=4000, temperature=0.0, timeout=90.0, think=True)
        replies.append((tuple(sorted(guard._verdict(reply, candidates))), reply))
    verdicts = [v for v, _ in replies]
    best = max(set(verdicts), key=verdicts.count)
    reply = next(r for v, r in replies if v == best)
    keep = guard._verdict(reply, candidates)
    return {"model": nemotron.MODEL, "request": request, "reply": json.dumps(nemotron.last_json(reply, "{")),
            "agreed": f"{verdicts.count(best)} of {runs}",
            "words": [{"w": w, "candidate": i in candidates, "kept": i in keep, "reason": keep.get(i, "")}
                      for i, w in enumerate(words)]}


CLIP_SPEECH_DB = -21.0  # the narration's speech level: the real voice must be heard as clearly


def speech_db(x: np.ndarray) -> float:
    """Loudness of the speech in a clip (RMS of its 50 ms frames above -45 dBFS)."""
    n = int(0.05 * SR)
    r = np.sqrt((x[: len(x) // n * n].reshape(-1, n) ** 2).mean(1))
    return float(20 * np.log10(np.sqrt((r[r > 10 ** (-45 / 20)] ** 2).mean())))


def level(x: np.ndarray, y: np.ndarray) -> tuple:
    """The same gain for both versions of the clip (so they compare fairly), up to the narration's
    level and without clipping."""
    g = min(10 ** ((CLIP_SPEECH_DB - speech_db(x)) / 20), 0.89 / max(np.abs(x).max(), np.abs(y).max()))
    return x * g, y * g


def envelope(x: np.ndarray) -> list:
    n = len(x) // HOP
    return np.round(np.abs(x[: n * HOP]).reshape(n, HOP).max(1), 4).tolist()


def clip() -> dict:
    a, b = WINDOW
    x = load_wav(CLIP)[int(a * SR):int(b * SR)]
    words = [dict(w, s=round(w["s"] - a, 3), e=round(w["e"] - a, 3))
             for w in json.loads(CLIP.with_suffix(".crisper.json").read_text()) if w["s"] >= a and w["e"] <= b]
    toks = label_tokens(words)
    f0 = int(round(a * SR / 320))  # 20 ms frames, sliced as the app slices them
    probs = np.load(CLIP.with_suffix(".teacher.npy"))[f0: f0 + len(x) // 320]
    confirm = calibration(RECORDINGS_CKPT).get("confirm", 0.3)
    cuts, report = guard.guarded_cuts(toks, probs[:, [0, 1, 2, 4]], cut_fillers=True, cut_phrases=True, confirm=confirm)
    pauses = tighten_pauses(toks, cuts)
    allc = sorted(cuts + pauses, key=lambda c: c["start"])
    y = render_edited(x, allc, NATURAL_JOIN)
    x, y = level(x, y)
    save_wav(OUT / "before.wav", x)
    save_wav(OUT / "after.wav", y)
    from stutterclear.app.demo import transcript
    text = transcript(toks, cuts)
    score = score_cuts(toks, allc)

    def after_time(t):  # a time on the original clock, on the cleaned clock (joins rescale the rest)
        removed = sum(max(0.0, min(t, c["end"]) - c["start"]) for c in allc)
        return (t - removed) * len(y) / (len(x) - sum(c["end"] - c["start"] for c in allc) * SR)

    after_words = [{"w": w["w"], "s": round(after_time(t["s"]), 3), "e": round(after_time(t["e"]), 3)}
                   for w, t in zip(text["words"], toks) if not w["removed"]]
    return {"before_s": round(len(x) / SR, 2), "after_s": round(len(y) / SR, 2),
            "env_before": envelope(x), "env_after": envelope(y),
            "cuts": [{"start": c["start"], "end": c["end"], "kind": c["kind"],
                      "words": " ".join(toks[i]["w"] for i in c["tokens"])} for c in allc],
            "words": [dict(w, s=t["s"], e=t["e"]) for w, t in zip(text["words"], toks)],
            "after_words": after_words,
            "clean": text["clean"], "said": text["said"], "guard": report,
            "probs": np.round(probs, 3).tolist(), "classes": CLASSES, "frame_ms": 20,
            "removed_pct": score["stutter_removed_pct"], "lost_pct": score["words_lost_pct"],
            "credit": CREDIT}


def numbers() -> dict:
    live = {(r["detector"], r["mode"]): r for r in json.loads((ROOT / "runs/eval/live_quality.json").read_text())["results"]}
    rec = {r["recipe"]: r for r in json.loads((ROOT / "runs/eval/recordings.json").read_text())["results"]}
    return {"live_strong_removed": live[("teacher", "strong")]["stutter_removed_pct"],
            "live_strong_lost": live[("teacher", "strong")]["words_lost_pct"],
            "live_gentle_removed": live[("teacher", "gentle")]["stutter_removed_pct"],
            "live_gentle_lost": live[("teacher", "gentle")]["words_lost_pct"],
            "rec_thorough_removed": rec["thorough"]["stutter_removed_pct"],
            "rec_thorough_lost": rec["thorough"]["words_lost_pct"], "speakers": 7, "f1_sep28k": 0.618}


def say_magpie(text: str, path: Path, voice: str = tts.VOICE):
    reply = tts._call(tts.request(text, voice, sr=NARRATION_SR), timeout=60.0)
    pcm = tts.audio_field(reply)
    save_wav(path, np.frombuffer(pcm[: len(pcm) // 2 * 2], np.int16).astype(np.float32) / 32768.0, NARRATION_SR)


def trim(path: Path, pad: float = 0.06, floor_db: float = -40.0) -> float:
    """Cut the silence a voice engine leaves before and after a line and bring its speech to the
    clip's level (CLIP_SPEECH_DB), so narration, Say it and the real voice sound equally loud;
    returns the new length (s)."""
    from scipy.io import wavfile
    sr, y = wavfile.read(path)
    y = y.astype(np.float32) / (32768.0 if y.dtype == np.int16 else 1.0)
    a = np.abs(y)
    loud = np.flatnonzero(a > a.max() * 10 ** (floor_db / 20))
    if len(loud):
        y = y[max(0, loud[0] - int(pad * sr)): loud[-1] + int(pad * sr)]
    n = int(0.05 * sr)
    r = np.sqrt((y[: len(y) // n * n].reshape(-1, n) ** 2).mean(1))
    speech = r[r > np.abs(y).max() * 10 ** (-30 / 20)]  # frames within 30 dB of the line's peak
    if len(speech):
        g = min(10 ** ((CLIP_SPEECH_DB - 20 * np.log10(np.sqrt((speech ** 2).mean()))) / 20), 0.89 / np.abs(y).max())
        y = y * g
    wavfile.write(path, sr, (np.clip(y, -1, 1) * 32767).astype(np.int16))
    return round(len(y) / sr, 2)


def say_windows(text: str, path: Path, voice: str = "Microsoft Zira Desktop", rate: int = 1):
    """Placeholder until NVIDIA_API_KEY is set: a Windows speech voice (rate -10..10)."""
    work = Path(tempfile.mkdtemp())
    (work / "say.txt").write_text(text, encoding="utf-8")
    (work / "say.ps1").write_text(
        "Add-Type -AssemblyName System.Speech\n"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
        f"$s.SelectVoice('{voice}')\n$s.Rate = {rate}\n"
        f"$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo({NARRATION_SR}, "
        "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)\n"
        f"$s.SetOutputToWaveFile('{path}', $f)\n$s.Speak([IO.File]::ReadAllText('{work / 'say.txt'}'))\n$s.Dispose()\n",
        encoding="utf-8")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(work / "say.ps1")], check=True)


def narration(board: dict) -> tuple:
    """One file per narrator line of the storyboard, named <scene>-<n>, trimmed of the engine's silence."""
    magpie = tts.available()
    (OUT / "narration").mkdir(parents=True, exist_ok=True)
    for old in (OUT / "narration").glob("*.wav"):  # lines of an earlier storyboard
        old.unlink()
    lines = []
    for scene in board["scenes"]:
        k = 0
        for step in scene["steps"]:
            if "line" not in step:
                continue
            key = f"{scene['id']}-{k}"
            k += 1
            path = OUT / "narration" / f"{key}.wav"
            text = step.get("say", step["line"])  # say: how to pronounce it
            if magpie:
                say_magpie(text, path, NARRATOR_VOICE)
            else:
                say_windows(text, path)
            lines.append({"id": key, "text": step["line"], "file": f"generated/narration/{key}.wav", "seconds": trim(path)})
    label = "Narrator: NVIDIA Magpie TTS (synthetic voice)" if magpie else "Narrator: placeholder synthetic voice (Windows)"
    print(f"  {len(lines)} lines, {sum(n['seconds'] for n in lines):.1f} s of narration ({label})", flush=True)
    return lines, label


def say_it(word: str) -> dict:
    """The rescued word as Say it speaks it: NVIDIA Magpie, as in the app, or (without a key) a
    labelled placeholder in another voice than the narrator's."""
    path = OUT / "say.wav"
    if tts.available():
        say_magpie(word, path)
        label = "Say it · NVIDIA Magpie TTS · synthetic voice"
    else:
        say_windows(word, path, "Microsoft David Desktop", rate=0)
        label = "Say it · placeholder synthetic voice (the app uses NVIDIA Magpie TTS)"
    return {"word": word, "file": "generated/say.wav", "seconds": trim(path), "label": label}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not nemotron.available():
        sys.exit("Set NEBIUS_API_KEY: the word rescue and meaning guard scenes show Nemotron's real answers.")
    board = json.loads(STORYBOARD.read_text(encoding="utf-8"))
    print("clip…", flush=True)
    data = {"clip": clip()}
    c = data["clip"]
    print(f"  {c['before_s']} s -> {c['after_s']} s, {len(c['cuts'])} cuts, removed {c['removed_pct']}%, lost {c['lost_pct']}%\n"
          f"  clean: {c['clean']}", flush=True)
    print("Nemotron: meaning guard and word rescue…", flush=True)
    data["guard"] = guard_demo(GUARD_SENTENCE)
    for _ in range(3):  # the film shows Alex tapping SAY_WORD: only with a real answer that offers it
        suggestions = rescue.suggest(RESCUE_SAID, RESCUE_CONTEXT, "l")
        if SAY_WORD in suggestions:
            break
    else:
        sys.exit(f"Nemotron did not suggest {SAY_WORD!r} in three tries ({suggestions}): change the rescue scene.")
    data["rescue"] = {"said": RESCUE_SAID, "onset": "l", "context": RESCUE_CONTEXT, "suggestions": suggestions}
    print("  guard:", data["guard"]["request"], "->", data["guard"]["reply"], "| rescue:", data["rescue"]["suggestions"], flush=True)
    data["numbers"] = numbers()
    data["model"] = nemotron.MODEL
    print("narration…", flush=True)
    data["narration"], data["voice_label"] = narration(board)
    data["say"] = say_it(SAY_WORD)
    data["board"] = board
    tmp = OUT / "data.json.tmp"  # replaced in one step: previews may be reading it
    tmp.write_text(json.dumps(data))
    os.replace(tmp, OUT / "data.json")
    print(f"wrote {OUT / 'data.json'}")


if __name__ == "__main__":
    main()
