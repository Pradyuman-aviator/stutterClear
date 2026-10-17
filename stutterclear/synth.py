"""Synthesise stutters into fluent, forced-aligned speech, with exact frame labels.

Each label marks audio that should be removed (the splicer's cut regions):
- sound_rep:    extra copies of a word's onset ("b- b- ball"): the copies and gaps
- word_rep:     extra copies of a whole word ("I I I want"): the copies and gaps
- prolongation: the stretched part of a sustained sound ("sssssnake")
- block:        a stuck onset fragment plus the silence after it ("t........test")

Plain pauses are also inserted, unlabelled, so silence alone is never evidence of a block.
Copies are perturbed (speed, gain, length) so repetitions are never exact duplicates.
"""
import io
import re
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly, butter, sosfilt

SR = 16000
FRAME = 320  # 20 ms, the frame rate of WavLM / Whisper encoders
# interjection ("um", "uh"): v2 synthesises fillers from the speaker's own schwa and nasal sounds.
CLASSES = ("sound_rep", "word_rep", "prolongation", "block", "interjection")

VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
FRICATIVES = {"S", "Z", "SH", "ZH", "F", "V", "TH", "DH", "HH"}
SONORANTS = {"M", "N", "NG", "L", "R", "W", "Y"}
PLOSIVES = {"P", "B", "T", "D", "K", "G", "CH", "JH"}
FUNCTION_WORDS = {"i", "a", "the", "and", "to", "my", "it", "is", "in", "of", "we", "he", "she", "you", "that",
                  "but", "so", "what", "this", "be", "was", "for", "on", "with", "me", "can", "do", "if", "they"}


@dataclass
class Utterance:
    audio: np.ndarray
    words: list  # (word, start_s, end_s)
    phones: list  # (phone, start_s, end_s)

    @classmethod
    def from_row(cls, row: dict) -> "Utterance":
        audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        assert sr == SR
        words = [(w["word"], w["start"], w["end"]) for w in row["words"]]
        phones = [(re.sub(r"\d", "", p["phoneme"]), p["start"], p["end"]) for p in row["phonemes"]]
        return cls(audio, words, phones)


@dataclass
class SynthConfig:
    p_fluent_utterance: float = 0.2
    p_sound_rep: float = 0.10
    p_word_rep: float = 0.06
    p_prolongation: float = 0.06
    p_block: float = 0.04
    p_pause: float = 0.08
    # v2: patterns seen in real speech that v1 never produced
    p_rapid_rep: float = 0.05  # "m-m-m-my": 3-5 onsets with little or no gap
    p_phrase_rep: float = 0.04  # "in roman | in roman letters", sometimes abandoned midway
    p_filler: float = 0.06  # "uh" / "um" built from the speaker's own schwa and nasal
    p_cluster: float = 0.03  # "th- [uh] the the": onsets, a filler and a word copy together
    max_event_fraction: float = 0.35


def synthesize(utt: Utterance, rng: np.random.Generator, cfg: SynthConfig = SynthConfig()):
    """Returns (audio, spans, events). spans: [(start, end, class_index)] in output samples."""
    x = utt.audio
    tone = _room_tone(x)
    out, spans, events = [], [], []
    pos = 0

    def emit(seg, cls=None):
        nonlocal pos
        seg = np.asarray(seg, np.float32)
        if cls is not None and len(seg):
            spans.append((pos, pos + len(seg), CLASSES.index(cls)))
        out.append(seg)
        pos += len(seg)

    def gap(sec):
        n = int(sec * SR)
        start = int(rng.integers(0, max(1, len(tone) - n))) if len(tone) > n else 0
        return np.resize(tone[start:], n) if len(tone) else np.zeros(n, np.float32)

    words = [(w, int(s * SR), int(e * SR)) for w, s, e in utt.words if e > s]
    phones = [(p, int(s * SR), int(e * SR)) for p, s, e in utt.phones if e > s]
    fluent = rng.random() < cfg.p_fluent_utterance
    budget = max(1, int(cfg.max_event_fraction * len(words)))
    schwas = [x[s:e] for p, s, e in phones if p in ("AH", "ER", "UH") and e - s >= 0.04 * SR]
    nasals = [x[s:e] for p, s, e in phones if p in ("M", "N") and e - s >= 0.03 * SR]
    cursor = 0

    def filler():
        if not schwas:
            return None
        y = _sustain(schwas[int(rng.integers(len(schwas)))], rng.uniform(0.15, 0.5), rng)
        if nasals and rng.random() < 0.5:  # "um" rather than "uh"
            y = np.concatenate([y, _sustain(nasals[int(rng.integers(len(nasals)))], rng.uniform(0.08, 0.25), rng)])
        return y

    def onsets(wph, n, gap_max):
        onset = _onset(x, wph, rng)
        if onset is None:
            return False
        for _ in range(n):
            emit(_vary(onset, rng), "sound_rep")
            emit(gap(rng.uniform(0.0, gap_max)), "sound_rep")
        return True

    for wi, (w, ws, we) in enumerate(words):
        if ws < cursor:
            ws = cursor
        emit(x[cursor:ws])
        cursor = ws
        wph = [p for p in phones if p[1] >= ws - 160 and p[2] <= we + 160 and p[0] != "spn"]
        kind = None
        if not fluent and budget > 0 and w != "<unk>" and wph:
            kind = _choose(rng, cfg, w, ws, we, wi, len(words))
        word = x[ws:we]

        if kind == "pause":
            emit(gap(rng.uniform(0.2, 0.9)))
        elif kind == "sound_rep":
            onset = _onset(x, wph, rng)
            if onset is not None:
                n = int(rng.choice([1, 2, 3, 4], p=[0.45, 0.3, 0.15, 0.1]))
                for _ in range(n):
                    emit(_vary(onset, rng), "sound_rep")
                    emit(gap(rng.uniform(0.04, 0.25)), "sound_rep")
                events.append({"kind": kind, "word": w, "iterations": n + 1})
                budget -= 1
        elif kind == "word_rep" and len(word) <= 0.6 * SR:
            n = int(rng.choice([1, 2, 3, 4], p=[0.45, 0.3, 0.15, 0.1]))
            for _ in range(n):
                emit(_vary(word, rng), "word_rep")
                emit(gap(rng.uniform(0.0, 0.3)), "word_rep")
            events.append({"kind": kind, "word": w, "iterations": n + 1})
            budget -= 1
        elif kind == "rapid_rep":
            n = int(rng.integers(3, 6))
            if onsets(wph, n, 0.06):
                events.append({"kind": kind, "word": w, "iterations": n + 1})
                budget -= 1
        elif kind == "filler":
            f = filler()
            if f is not None:
                emit(gap(rng.uniform(0.05, 0.2)))
                emit(f, "interjection")
                emit(gap(rng.uniform(0.05, 0.3)), "interjection")
                events.append({"kind": kind, "word": w})
                budget -= 1
        elif kind == "phrase_rep":
            k = int(rng.integers(2, 4))
            phrase = x[ws:words[min(wi + k, len(words)) - 1][2]]
            if len(phrase) <= 1.5 * SR:
                reps = int(rng.choice([1, 2], p=[0.75, 0.25]))
                for r in range(reps):
                    copy = _vary(phrase, rng)
                    if r == reps - 1 and rng.random() < 0.4:  # restart: the last copy is abandoned
                        copy = _fade(copy[: int(len(copy) * rng.uniform(0.35, 0.85))].copy(), 64)
                    emit(copy, "word_rep")
                    emit(gap(rng.uniform(0.05, 0.35)), "word_rep")
                events.append({"kind": kind, "word": w, "words": k, "iterations": reps + 1})
                budget -= 1
        elif kind == "cluster":
            parts = ["onsets", "filler"] + (["word"] if len(word) <= 0.6 * SR else [])
            rng.shuffle(parts)
            for part in parts:
                if part == "onsets":
                    onsets(wph, int(rng.integers(1, 4)), 0.1)
                elif part == "filler" and (f := filler()) is not None:
                    emit(f, "interjection")
                    emit(gap(rng.uniform(0.03, 0.2)), "interjection")
                elif part == "word":
                    emit(_vary(word, rng), "word_rep")
                    emit(gap(rng.uniform(0.0, 0.2)), "word_rep")
            events.append({"kind": kind, "word": w, "parts": parts})
            budget -= 1
        elif kind == "prolongation":
            target = _prolongable(wph, rng)
            if target is not None:
                _, ps, pe = target
                mid = (ps + pe) // 2
                emit(x[ws:mid])
                emit(_sustain(x[ps:pe], rng.uniform(0.25, 1.5), rng), "prolongation")
                word = x[mid:we]
                events.append({"kind": kind, "word": w, "phone": target[0]})
                budget -= 1
        elif kind == "block":
            if rng.random() < 0.6:
                frag = x[ws:ws + int(rng.uniform(0.02, 0.06) * SR)]
                emit(_fade(frag.copy(), 48), "block")
            emit(gap(rng.uniform(0.25, 1.8)), "block")
            events.append({"kind": kind, "word": w})
            budget -= 1

        emit(word)
        cursor = we
    emit(x[cursor:])
    return np.concatenate(out), spans, events


def frame_labels(spans, n_samples: int) -> np.ndarray:
    """Multi-hot [frames, classes]; a frame gets a class if at least half of it is covered."""
    n_frames = -(-n_samples // FRAME)
    lab = np.zeros((n_frames, len(CLASSES)), np.float32)
    for s, e, c in spans:
        for f in range(s // FRAME, min(n_frames, -(-e // FRAME))):
            if min(e, (f + 1) * FRAME) - max(s, f * FRAME) >= FRAME // 2:
                lab[f, c] = 1.0
    return lab


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Length-preserving acoustic augmentation, so frame labels stay valid."""
    x = x * 10 ** (rng.uniform(-12, 6) / 20)
    if rng.random() < 0.3:  # room reverb with a synthetic exponentially decaying impulse response
        rt60 = rng.uniform(0.1, 0.6)
        t = np.arange(int(rt60 * SR)) / SR
        rir = rng.normal(size=len(t)) * np.exp(-6.9 * t / rt60)
        rir[0] = 1.0
        x = fftconvolve(x, rir / np.sqrt(np.sum(rir ** 2)))[: len(x)]
    if rng.random() < 0.2:  # narrowband mics, phone calls
        x = sosfilt(butter(6, rng.uniform(3400, 7000), "low", fs=SR, output="sos"), x)
    if rng.random() < 0.7:  # background noise, white-to-brown
        noise = np.cumsum(rng.normal(size=len(x))) if rng.random() < 0.3 else rng.normal(size=len(x))
        noise = noise - np.convolve(noise, np.ones(400) / 400, "same") if rng.random() < 0.3 else noise
        snr = rng.uniform(5, 40)
        p_sig = np.mean(x ** 2) + 1e-10
        x = x + noise * np.sqrt(p_sig / (np.mean(noise ** 2) + 1e-10) / 10 ** (snr / 10))
    if rng.random() < 0.15:  # OS noise gates that output exact zeros in quiet frames (seen on our laptop)
        fr = x[: len(x) // 160 * 160].reshape(-1, 160)
        quiet = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-12) < np.percentile(
            10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-12), rng.uniform(10, 40))
        fr[quiet] = 0.0
    peak = np.abs(x).max()
    return (x / peak * 0.95 if peak > 0.95 else x).astype(np.float32)


# ---- helpers ------------------------------------------------------------------

def _choose(rng, cfg, w, ws, we, wi, n_words):
    p = {"sound_rep": cfg.p_sound_rep,
         "word_rep": cfg.p_word_rep * (3.0 if w in FUNCTION_WORDS else 0.5),
         "prolongation": cfg.p_prolongation,
         "block": cfg.p_block if wi > 0 else 0.0,
         "pause": cfg.p_pause if wi > 0 else 0.0,
         "rapid_rep": cfg.p_rapid_rep,
         "phrase_rep": cfg.p_phrase_rep if wi + 2 < n_words else 0.0,
         "filler": cfg.p_filler,
         "cluster": cfg.p_cluster}
    r = rng.random()
    for k, v in p.items():
        if r < v:
            return k
        r -= v
    return None


def _onset(x, wph, rng):
    """First phone (plus the start of the next vowel for plosives) of a word."""
    ph, s, e = wph[0]
    if ph in PLOSIVES and len(wph) > 1 and wph[1][0] in VOWELS:
        e = min(wph[1][2], wph[1][1] + int(rng.uniform(0.03, 0.09) * SR))
    elif ph in VOWELS:
        e = min(e, s + int(rng.uniform(0.06, 0.14) * SR))
    elif len(wph) > 1 and wph[1][0] in VOWELS and rng.random() < 0.5:
        e = min(wph[1][2], wph[1][1] + int(0.04 * SR))
    if not 0.03 * SR <= e - s <= 0.3 * SR:
        return None
    return x[s:e]


def _prolongable(wph, rng):
    ok = [p for p in wph if p[0] in FRICATIVES | SONORANTS | VOWELS and p[2] - p[1] >= 0.03 * SR]
    if not ok:
        return None
    return ok[0] if ok[0] is wph[0] and rng.random() < 0.7 else ok[int(rng.integers(len(ok)))]


def _vary(seg, rng):
    """A perturbed copy: speed 0.9-1.1, gain +-3 dB, tail trimmed by up to 20 %."""
    up, down = [(9, 10), (19, 20), (1, 1), (21, 20), (11, 10)][int(rng.integers(5))]
    y = resample_poly(seg, up, down) if up != down else seg.copy()
    y = y[: max(160, int(len(y) * rng.uniform(0.8, 1.0)))]
    return _fade(y * 10 ** (rng.uniform(-3, 3) / 20), 64)


def _fade(y, n):
    n = min(n, len(y) // 2)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n)
        y[:n] *= ramp
        y[-n:] *= ramp[::-1]
    return y


def _sustain(seg, dur, rng):
    """Stretch a phone to `dur` seconds: period-aligned looping if voiced, grain shuffling if not."""
    n = int(dur * SR)
    period = _pitch_period(seg)
    if period is None or len(seg) < 3 * period:
        grain, hop = 480, 240
        if len(seg) <= grain:
            seg = np.resize(seg, grain + 1)
        win = np.hanning(grain)
        out = np.zeros(n + grain)
        for p in range(0, n, hop):
            o = int(rng.integers(0, len(seg) - grain))
            out[p:p + grain] += seg[o:o + grain] * win
        y = out[:n]
    else:
        k = max(1, int(0.03 * SR / period))
        length = k * period
        start = (len(seg) - length) // 2
        chunk = seg[start:start + length]
        xf = min(period, length // 2)
        y = chunk.copy()
        ramp = np.linspace(0.0, 1.0, xf)
        while len(y) < n:
            y = np.concatenate([y[:-xf], y[-xf:] * ramp[::-1] + chunk[:xf] * ramp, chunk[xf:]])
        y = y[:n]
        t = np.arange(n) / SR
        y = y * 10 ** (0.08 * np.sin(2 * np.pi * rng.uniform(3, 6) * t + rng.uniform(0, 6.28)) / 20)
    rms_in = np.sqrt(np.mean(seg ** 2)) + 1e-8
    y = y * rms_in / (np.sqrt(np.mean(y ** 2)) + 1e-8)
    return _fade(y.astype(np.float32), 80)


def _pitch_period(seg):
    if len(seg) < 400:
        return None
    s = seg - seg.mean()
    ac = np.correlate(s, s, "full")[len(s) - 1:]
    if ac[0] <= 0:
        return None
    lo, hi = SR // 400, SR // 70
    lag = lo + int(np.argmax(ac[lo:hi]))
    return lag if ac[lag] / ac[0] > 0.5 else None


def _room_tone(x):
    fr = x[: len(x) // 160 * 160].reshape(-1, 160)
    e = np.mean(fr ** 2, axis=1)
    quiet = fr[e <= np.percentile(e, 15)]
    return quiet.ravel() if len(quiet) else np.zeros(160, np.float32)
