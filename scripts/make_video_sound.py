"""The demo film's music bed and sound effects, synthesised here so nothing needs a licence.

    python scripts/make_video_sound.py      # writes video/public/generated/sound/*.wav

music.wav: a slow ambient pad (Am7 - Fmaj7 - Cmaj7 - G6, 8 s per chord) with a soft sub and a sparse
plucked arpeggio, through a long reverb; the film ducks it under speech. whoosh.wav (scene
transitions), snap.wav (a stutter cut out), chime.wav (the meaning guard keeps a word), pop.wav (a
suggestion appears).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stutterclear.audio import save_wav  # noqa: E402

SR = 44100
OUT = ROOT / "video" / "public" / "generated" / "sound"
rng = np.random.default_rng(7)


def note(name: str) -> float:
    names = {"C": -9, "C#": -8, "D": -7, "D#": -6, "E": -5, "F": -4, "F#": -3, "G": -2, "G#": -1, "A": 0, "A#": 1, "B": 2}
    return 440.0 * 2 ** ((names[name[:-1]] + 12 * (int(name[-1]) - 4)) / 12)


def reverb(x: np.ndarray, seconds: float = 2.6, mix: float = 0.35) -> np.ndarray:
    n = int(seconds * SR)
    ir = rng.standard_normal(n) * np.exp(-np.linspace(0, 7, n))
    ir = sosfilt(butter(2, 5000, "low", fs=SR, output="sos"), ir)
    wet = fftconvolve(x, ir)[: len(x)]
    wet /= np.abs(wet).max() + 1e-9
    return (1 - mix) * x / (np.abs(x).max() + 1e-9) + mix * wet


def pad(freqs, seconds: float) -> np.ndarray:
    """Soft detuned voices with slow attack and release."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for f in freqs:
        for det in (-0.12, 0.0, 0.13):  # a gentle chorus
            ff = f * 2 ** (det / 12)
            phase = rng.uniform(0, 2 * np.pi)
            for h, a in ((1, 1.0), (2, 0.32), (3, 0.12), (4, 0.05)):
                y += a * np.sin(2 * np.pi * ff * h * t + phase * h)
    env = np.minimum(1.0, t / 2.2) * np.minimum(1.0, (seconds - t) / 2.6)
    return y * np.clip(env, 0, 1) * (1 + 0.08 * np.sin(2 * np.pi * 0.11 * t))


def pluck(f: float, seconds: float = 2.4) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n) / SR
    y = sum(a * np.sin(2 * np.pi * f * h * t) for h, a in ((1, 1.0), (2, 0.4), (3, 0.15)))
    return y * np.exp(-t * 2.6) * np.minimum(1.0, t / 0.004)


def music(seconds: float = 200.0) -> np.ndarray:
    chords = [["A2", "E3", "G3", "C4", "E4"], ["F2", "C3", "E3", "A3", "C4"], ["C3", "G3", "B3", "E4"], ["G2", "D3", "E3", "B3", "D4"]]
    step = 8.0
    n = int(seconds * SR)
    y = np.zeros(n + int(4 * SR))
    for k in range(int(seconds // step) + 1):
        ch = chords[k % len(chords)]
        seg = pad([note(c) for c in ch], step + 3.0)
        a = int(k * step * SR)
        y[a:a + len(seg)] += seg[: len(y) - a]
        sub = pad([note(ch[0]) / 2], step + 3.0) * 0.6
        y[a:a + len(sub)] += sub[: len(y) - a]
        for j, c in enumerate(ch[1:] + ch[1:2]):  # a sparse arpeggio, two notes a second
            if (k + j) % 3 == 2:
                continue
            p = pluck(note(c) * 2) * 0.35
            b = a + int((0.5 + j * 1.5) * SR)
            if b < len(y):
                y[b:b + len(p)] += p[: len(y) - b]
    y = sosfilt(butter(2, 3800, "low", fs=SR, output="sos"), y[:n])
    y = reverb(y, 3.2, 0.45)
    fade = np.minimum(1.0, np.arange(n) / (2 * SR)) * np.minimum(1.0, (n - np.arange(n)) / (4 * SR))
    return (0.8 * y / (np.abs(y).max() + 1e-9) * fade).astype(np.float32)


def whoosh() -> np.ndarray:
    n = int(0.9 * SR)
    t = np.arange(n) / n
    noise = rng.standard_normal(n)
    out = np.zeros(n)
    for i in range(0, n, 512):  # a band-pass that sweeps up then down
        f = 300 + 3200 * np.sin(np.pi * t[i]) ** 2
        sos = butter(2, [f * 0.7, f * 1.3], "band", fs=SR, output="sos")
        out[i:i + 512] = sosfilt(sos, noise[i:i + 512])
    return reverb(out * np.sin(np.pi * t) ** 1.5, 1.2, 0.3).astype(np.float32) * 0.8


def snap() -> np.ndarray:
    n = int(0.25 * SR)
    t = np.arange(n) / SR
    click = rng.standard_normal(n) * np.exp(-t * 90)
    tone = np.sin(2 * np.pi * 1800 * t) * np.exp(-t * 40) * 0.5
    return reverb(sosfilt(butter(2, 900, "high", fs=SR, output="sos"), click) + tone, 0.6, 0.2).astype(np.float32) * 0.7


def chime() -> np.ndarray:
    n = int(1.6 * SR)
    y = pluck(note("E5"), 1.6) + 0.7 * np.pad(pluck(note("B5"), 1.5), (int(0.09 * SR), n))[:n]
    return reverb(y, 1.8, 0.4).astype(np.float32) * 0.7


def pop() -> np.ndarray:
    n = int(0.18 * SR)
    t = np.arange(n) / SR
    f = 600 + 900 * np.exp(-t * 30)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 28)
    return reverb(y, 0.5, 0.2).astype(np.float32) * 0.6


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, y in (("music", music()), ("whoosh", whoosh()), ("snap", snap()), ("chime", chime()), ("pop", pop())):
        save_wav(OUT / f"{name}.wav", y, SR)
        print(f"{name}: {len(y) / SR:.1f} s")


if __name__ == "__main__":
    main()
