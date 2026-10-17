"""Training data.

SynthStutterStream: LibriSpeech with synthesised stutters -> frame labels (exact).
Sep28kClips: real SEP-28k / FluencyBank 3 s clips -> clip labels (annotator votes / 3).
Both are split by speaker group, never by clip, so validation measures unseen speakers.
"""
import io
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .synth import CLASSES, FRAME, SR, SynthConfig, Utterance, augment, frame_labels, synthesize

SEP_COLUMNS = ("SoundRep", "WordRep", "Prolongation", "Block", "Interjection")  # same order as CLASSES
assert len(SEP_COLUMNS) == len(CLASSES)


class SynthStutterStream(IterableDataset):
    def __init__(self, files, crop_s: float = 6.0, seed: int = 0, augment_p: float = 0.8,
                 cfg: SynthConfig = SynthConfig()):
        self.files = [str(f) for f in files]
        self.n = int(crop_s * SR) // FRAME * FRAME
        self.seed, self.augment_p, self.cfg = seed, augment_p, cfg

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng(self.seed + 7919 * wid)
        files = self.files[wid::nw] or self.files
        while True:
            for f in rng.permutation(files):
                pf = pq.ParquetFile(f)
                for g in rng.permutation(pf.num_row_groups):
                    rows = pf.read_row_group(int(g), columns=["audio", "words", "phonemes"]).to_pylist()
                    for i in rng.permutation(len(rows)):
                        yield self.example(rows[i], rng)

    def example(self, row, rng):
        y, spans, _ = synthesize(Utterance.from_row(row), rng, self.cfg)
        if rng.random() < self.augment_p:
            y = augment(y, rng)
        lab = frame_labels(spans, len(y))
        y, lab, mask = _crop(y, lab, spans, self.n, rng)
        return {"wave": torch.from_numpy(y), "labels": torch.from_numpy(lab), "mask": torch.from_numpy(mask)}


def fixed_synth_set(files, n: int, crop_s: float = 6.0, seed: int = 1234) -> list:
    """A deterministic validation set (no augmentation) from held-out utterances."""
    stream = SynthStutterStream(files, crop_s=crop_s, seed=seed, augment_p=0.0)
    it = iter(stream)
    return [next(it) for _ in range(n)]


def _crop(y, lab, spans, n, rng):
    frames = n // FRAME
    if len(y) >= n:
        if spans and rng.random() < 0.7:  # most crops contain a stutter
            s, e, _ = spans[int(rng.integers(len(spans)))]
            lo, hi = max(0, e - n), min(s, len(y) - n)
            start = int(rng.integers(lo, hi + 1)) if hi >= lo else int(rng.integers(0, len(y) - n + 1))
        else:
            start = int(rng.integers(0, len(y) - n + 1))
        start = start // FRAME * FRAME
        y = y[start:start + n]
        lab = lab[start // FRAME:start // FRAME + frames]
    out_y = np.zeros(n, np.float32)
    out_y[:len(y)] = y
    out_lab = np.zeros((frames, len(CLASSES)), np.float32)
    out_lab[:len(lab)] = lab[:frames]
    mask = np.zeros(frames, np.float32)
    mask[:min(frames, len(y) // FRAME)] = 1.0
    return out_y, out_lab, mask


def speaker_group(show: str, ep_id: int) -> str:
    # SEP-28k: one podcast is roughly one set of speakers. FluencyBank: one episode per participant.
    return f"FluencyBank/{ep_id}" if show == "FluencyBank" else show


def split_of(group: str) -> str:
    h = zlib.crc32(group.encode()) % 100
    return "val" if h < 15 else "test" if h < 30 else "train"


class Sep28kClips(Dataset):
    def __init__(self, files, split: str = "train", augment_p: float = 0.5, seed: int = 0):
        cols = ["audio", "Show", "EpId", *SEP_COLUMNS, "PoorAudioQuality", "Music", "NoSpeech"]
        # Audio stays as one Arrow array per file (not Python objects, so forked DataLoader workers
        # share it; not one merged array, because the full column exceeds Arrow's 2 GB offset limit).
        self.audio, self.index, labels, groups = [], [], [], set()
        for f in files:
            t = pq.read_table(f, columns=cols)
            d = t.drop_columns(["audio"]).to_pydict()
            keep = [i for i in range(t.num_rows)
                    if d["PoorAudioQuality"][i] < 2 and d["Music"][i] < 2 and d["NoSpeech"][i] < 2
                    and split_of(speaker_group(d["Show"][i], d["EpId"][i])) == split]
            if not keep:  # this file has no speakers in this split
                continue
            self.audio.append(t.column("audio").take(pa.array(keep, pa.int64())).combine_chunks())
            self.index += [(len(self.audio) - 1, j) for j in range(len(keep))]
            labels += [[d[c][i] / 3.0 for c in SEP_COLUMNS] for i in keep]
            groups |= {speaker_group(d["Show"][i], d["EpId"][i]) for i in keep}
        self.labels = np.array(labels, np.float32).reshape(-1, len(SEP_COLUMNS))
        self.groups = sorted(groups)
        self.augment_p, self.rng = augment_p, np.random.default_rng(seed)
        self.n = 3 * SR

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        k, j = self.index[i]
        y, sr = sf.read(io.BytesIO(self.audio[k][j]["bytes"].as_py()), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = np.pad(y[:self.n], (0, max(0, self.n - len(y))))
        if self.augment_p and self.rng.random() < self.augment_p:
            y = augment(y, self.rng)
        return {"wave": torch.from_numpy(y.astype(np.float32)), "clip_labels": torch.from_numpy(self.labels[i])}
