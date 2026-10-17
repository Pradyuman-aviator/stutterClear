"""Build a stuttered test sentence with exact labels, using the Windows built-in voice.

Writes demo/stuttered.wav, demo/fluent.wav and demo/stuttered.labels.json.
Labels mark the audio that should be removed (seconds).
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stutterclear.audio import load_wav, save_wav  # noqa: E402
from stutterclear.config import SAMPLE_RATE as SR  # noqa: E402

WORDS = ["My", "name", "is", "Sam", "and", "I", "want", "the", "ball"]


def synth_words(words, out_dir: Path) -> dict:
    lines = ["Add-Type -AssemblyName System.Speech",
             "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer",
             "$s.Rate = 0",
             "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
             "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)"]
    for w in words:
        lines += [f'$s.SetOutputToWaveFile("{out_dir / (w + ".wav")}", $f)', f'$s.Speak("{w}")']
    lines.append("$s.SetOutputToNull()")
    script = out_dir / "synth.ps1"
    script.write_text("\n".join(lines), encoding="utf-8")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)], check=True)
    return {w: trim(load_wav(out_dir / f"{w}.wav")) for w in words}


def trim(x: np.ndarray, thresh_db: float = -40.0) -> np.ndarray:
    frames = x[: len(x) // 160 * 160].reshape(-1, 160)
    db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10)
    on = np.flatnonzero(db > db.max() + thresh_db)
    return x[on[0] * 160: (on[-1] + 1) * 160]


def stretch_noise(seg: np.ndarray, dur: float, rng) -> np.ndarray:
    """Granular extension of a fricative: overlap-added grains from random offsets."""
    grain, hop = 480, 240
    win = np.hanning(grain)
    out = np.zeros(int(dur * SR) + grain)
    for p in range(0, len(out) - grain, hop):
        o = rng.integers(0, len(seg) - grain)
        out[p:p + grain] += seg[o:o + grain] * win
    return out[: int(dur * SR)].astype(np.float32)


def main():
    rng = np.random.default_rng(0)
    out_dir = Path(__file__).resolve().parents[1] / "demo"
    out_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        w = synth_words(WORDS, Path(tmp))

    gap = lambda s: np.zeros(int(s * SR), np.float32)
    parts, fluent, labels = [], [], []
    t = 0.0

    def add(x, remove_kind=None, to_fluent=True):
        nonlocal t
        parts.append(x)
        if remove_kind:
            labels.append({"kind": remove_kind, "start": round(t, 3), "end": round(t + len(x) / SR, 3)})
        if to_fluent:
            fluent.append(x)
        t += len(x) / SR

    add(gap(0.6))
    add(w["My"]); add(gap(0.12))
    add(gap(1.2), "block", to_fluent=False)  # silent block before "name": reported, never cut
    add(w["name"]); add(gap(0.1)); add(w["is"]); add(gap(0.1))

    sam = w["Sam"]
    fric = sam[: int(0.09 * SR)]
    add(fric)
    add(stretch_noise(fric, 0.8, rng), "prolongation", to_fluent=False)  # Sssssssam
    add(sam[len(fric):]); add(gap(0.15))

    add(w["and"]); add(gap(0.12))
    rep = np.concatenate([w["I"], gap(0.12), w["I"], gap(0.12)])  # I- I- I want
    add(rep, "word_repetition", to_fluent=False)
    add(w["I"]); add(gap(0.08)); add(w["want"]); add(gap(0.08)); add(w["the"]); add(gap(0.1))
    b = w["ball"][: int(0.09 * SR)]
    add(np.concatenate([b, gap(0.15), b, gap(0.15)]), "sound_repetition", to_fluent=False)  # b- b- ball
    add(w["ball"]); add(gap(0.8))

    save_wav(out_dir / "stuttered.wav", np.concatenate(parts) * 0.9)
    save_wav(out_dir / "fluent.wav", np.concatenate(fluent) * 0.9)
    removable = [l for l in labels if l["kind"] != "block"]
    (out_dir / "stuttered.labels.json").write_text(json.dumps(
        {"remove": removable, "blocks": [l for l in labels if l["kind"] == "block"]}, indent=2))
    print(f"wrote {out_dir}/stuttered.wav ({t:.2f}s) with {len(removable)} removable stutters + 1 block")


if __name__ == "__main__":
    main()
