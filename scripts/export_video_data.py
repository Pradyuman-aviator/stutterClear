"""Everything the demo video shows, from the real pipeline: a before/after clip with its real cuts and
transcript, the meaning guard's and word rescue's real Nemotron answers, the measured numbers, and
the narration (NVIDIA Magpie TTS with NVIDIA_API_KEY; otherwise a placeholder Windows voice).

    python scripts/export_video_data.py            # writes video/public/generated/
    cd video && npm run render                      # then render the video

Needs NEBIUS_API_KEY for the Nemotron scenes. The clip is LibriSpeech read speech (CC BY 4.0) with
synthetic stutters, labelled as such in the video.
"""
import json
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
from stutterclear.silver import label_tokens  # noqa: E402

OUT = ROOT / "video" / "public" / "generated"
CLIP, CLIP_END_S = ROOT / "data" / "demo_clips" / "read-3.wav", 13.4  # "...out of the the the mist"
NARRATION_SR = 44100

NARRATION = [
    ("hook", "About one in a hundred people stutter. On a call, every repeat is something the listener has to wait through."),
    ("intro", "StutterClear is a voice filter for people who stutter. You speak the way you speak. "
              "The listener hears your own voice, without the repeats."),
    ("before", "Here is read speech, with stutters."),
    ("after", "And here is what the listener hears."),
    ("how", "A WavLM model, trained on real stuttered speech, listens about half a second behind you, on the GPU. "
            "Each cut lands between words, and the join is filled with your own room tone."),
    ("guard", "But not every repeat is a stutter. NVIDIA Nemotron, on Nebius Token Factory, reads each cut in its "
              "sentence. That that, is grammar. Very very, is emphasis. So both stay."),
    ("calls", "In Zoom, Meet or Teams, it works as a virtual microphone, with live captions from NVIDIA's streaming "
              "FastConformer."),
    ("rescue", "When you are stuck on a word, Nemotron suggests what you mean, from your context. Tap one, and a "
               "synthetic voice says it, only when you choose."),
    ("app", "It also cleans voice notes and videos, turns speech into clean text, and learns from your feedback. "
            "Nothing is saved unless you choose."),
    ("numbers", "On speakers it never trained on, it removes sixty five percent of stutters live, and ninety one "
                "percent in recordings, cutting only between words."),
    ("end", "StutterClear. Your real voice, on your terms."),
]


def envelope(x: np.ndarray) -> list:
    n = len(x) // HOP
    return np.round(np.abs(x[: n * HOP]).reshape(n, HOP).max(1), 4).tolist()


def clip() -> dict:
    """The before/after: the Thorough engine (repeats, broken-off starts, fillers, repeated phrases)
    with the meaning guard, exactly as a recording is cleaned in the app."""
    x = load_wav(CLIP)[: int(CLIP_END_S * SR)]
    words = [w for w in json.loads(CLIP.with_suffix(".crisper.json").read_text()) if w["e"] <= CLIP_END_S]
    toks = label_tokens(words)
    probs = np.load(CLIP.with_suffix(".teacher.npy"))[: len(x) // 320, [0, 1, 2, 4]]
    cuts, report = guard.guarded_cuts(toks, probs, cut_fillers=True, cut_phrases=True)
    y = render_edited(x, cuts, NATURAL_JOIN)
    save_wav(OUT / "before.wav", x)
    save_wav(OUT / "after.wav", y)
    from stutterclear.app.demo import transcript
    text = transcript(toks, cuts)
    return {"before_s": round(len(x) / SR, 2), "after_s": round(len(y) / SR, 2),
            "env_before": envelope(x), "env_after": envelope(y),
            "cuts": [{"start": c["start"], "end": c["end"], "kind": c["kind"]} for c in cuts],
            "words": [dict(w, s=t["s"], e=t["e"]) for w, t in zip(text["words"], toks)],
            "clean": text["clean"], "guard": report,
            "credit": "Read speech from LibriSpeech (CC BY 4.0) with synthetic stutters"}


def numbers() -> dict:
    live = {(r["detector"], r["mode"]): r for r in json.loads((ROOT / "runs/eval/live_quality.json").read_text())["results"]}
    rec = {r["recipe"]: r for r in json.loads((ROOT / "runs/eval/recordings.json").read_text())["results"]}
    return {"live_strong_removed": live[("teacher", "strong")]["stutter_removed_pct"],
            "live_strong_lost": live[("teacher", "strong")]["words_lost_pct"],
            "live_gentle_removed": live[("teacher", "gentle")]["stutter_removed_pct"],
            "live_gentle_lost": live[("teacher", "gentle")]["words_lost_pct"],
            "rec_thorough_removed": rec["thorough"]["stutter_removed_pct"],
            "rec_thorough_lost": rec["thorough"]["words_lost_pct"], "speakers": 7}


def say_magpie(text: str, path: Path):
    reply = tts._call(tts.request(text, tts.VOICE, sr=NARRATION_SR), timeout=60.0)
    pcm = tts.audio_field(reply)
    y = np.frombuffer(pcm[: len(pcm) // 2 * 2], np.int16).astype(np.float32) / 32768.0
    save_wav(path, y, NARRATION_SR)


def say_windows(text: str, path: Path):
    """Placeholder until NVIDIA_API_KEY is set: the Windows speech voice."""
    script = Path(tempfile.mkdtemp()) / "say.ps1"
    script.write_text(
        "Add-Type -AssemblyName System.Speech\n"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
        "$s.SelectVoice('Microsoft Zira Desktop')\n$s.Rate = 0\n"
        f"$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo({NARRATION_SR}, "
        "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)\n"
        f"$s.SetOutputToWaveFile('{path}', $f)\n$s.Speak([IO.File]::ReadAllText('{script.with_suffix('.txt')}'))\n"
        "$s.Dispose()\n", encoding="utf-8")
    script.with_suffix(".txt").write_text(text, encoding="utf-8")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)], check=True)


def narration() -> tuple:
    from scipy.io import wavfile
    magpie = tts.available()
    (OUT / "narration").mkdir(parents=True, exist_ok=True)
    lines = []
    for key, text in NARRATION:
        path = OUT / "narration" / f"{key}.wav"
        (say_magpie if magpie else say_windows)(text, path)
        sr, y = wavfile.read(path)
        lines.append({"id": key, "text": text, "file": f"generated/narration/{key}.wav", "seconds": round(len(y) / sr, 2)})
        print(f"  {key}: {lines[-1]['seconds']} s", flush=True)
    label = ("Narration: NVIDIA Magpie TTS (synthetic voice)" if magpie
             else "Narration: placeholder synthetic voice (Windows)")
    return lines, label


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not nemotron.available():
        sys.exit("Set NEBIUS_API_KEY: the meaning guard and word rescue scenes show Nemotron's real answers.")
    print("clip…", flush=True)
    data = {"clip": clip()}
    print(f"  {data['clip']['before_s']} s -> {data['clip']['after_s']} s, {len(data['clip']['cuts'])} cuts; "
          f"guard kept {[k['word'] for k in (data['clip']['guard'] or {}).get('kept', [])]}", flush=True)
    print("meaning guard and word rescue (Nemotron)…", flush=True)
    data["guard"] = guard.check_text("I I I think that that plan is very very good")
    context = "Demo day at the Nebius and NVIDIA Global AI Hackathon. Our project: StutterClear."
    said = "Thank you all for coming. Today I want to show you what we built for the"
    data["rescue"] = {"said": said, "onset": "h", "context": context,
                      "suggestions": rescue.suggest(said, context, "h")}
    print("  guard:", [(w["w"], w["reason"]) for w in data["guard"]["words"] if w["kept"]],
          "| rescue:", data["rescue"]["suggestions"], flush=True)
    data["numbers"] = numbers()
    data["model"] = nemotron.MODEL
    print("narration…", flush=True)
    data["narration"], data["voice_label"] = narration()
    (OUT / "data.json").write_text(json.dumps(data, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
