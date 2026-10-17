"""Training data.

SynthStutterStream: LibriSpeech with synthesised stutters -> frame labels (exact).
Sep28kClips: real SEP-28k / FluencyBank 3 s clips -> clip labels (annotator votes / 3).
Both are split by speaker group, never by clip, so validation measures unseen speakers.
"""
import io
import json
import zlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .audio import load_wav
from .silver import frame_targets
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
        return {"wave": torch.from_numpy(y), "labels": torch.from_numpy(lab), "mask": torch.from_numpy(mask),
                "class_mask": torch.ones(len(CLASSES))}


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


SEP_E_FILE = "SEP-28k-Extended_clips.csv"  # Bayerl et al. 2022, speaker-exclusive SEP-28k partition


def load_sep28k_e(folder) -> dict:
    """{(show, ep, clip): train|val|test} from the official SEP-28k-E partition, if present.
    Published results use this split, so SEP-28k numbers stay comparable."""
    hits = sorted(Path(folder).rglob(SEP_E_FILE))
    if not hits:
        return {}
    import csv
    names = {"train": "train", "dev": "val", "test": "test"}
    with open(hits[0], encoding="utf-8") as f:
        return {(r["Show"], int(r["EpId"]), int(r["ClipId"])): names[r["SEP28k-E"]] for r in csv.DictReader(f)}


class Sep28kClips(Dataset):
    """Real 3 s clips with clip labels; clips that have verbatim word timings (`silver`, keyed
    "<parquet name>:<row>") also get frame labels for the classes a transcript can mark."""

    def __init__(self, files, split: str = "train", augment_p: float = 0.5, seed: int = 0, silver=None):
        cols = ["audio", "Show", "EpId", "ClipId", *SEP_COLUMNS, "PoorAudioQuality", "Music", "NoSpeech"]
        # SEP-28k clips follow the official SEP-28k-E partition when its CSV sits next to the data;
        # FluencyBank (no official partition) and anything unmatched are split by speaker group.
        official = load_sep28k_e(Path(files[0]).parent.parent) if files else {}
        self.split_source = "SEP-28k-E" if official else "speaker hash"

        def split_for(d, i):
            show = d["Show"][i]
            if show != "FluencyBank":
                part = official.get((show, d["EpId"][i], d["ClipId"][i]))
                if part:
                    return part
            return split_of(speaker_group(show, d["EpId"][i]))

        # Audio stays as one Arrow array per file (not Python objects, so forked DataLoader workers
        # share it; not one merged array, because the full column exceeds Arrow's 2 GB offset limit).
        self.audio, self.index, labels, groups, self.silver_words = [], [], [], set(), []
        for f in files:
            t = pq.read_table(f, columns=cols)
            d = t.drop_columns(["audio"]).to_pydict()
            keep = [i for i in range(t.num_rows)
                    if d["PoorAudioQuality"][i] < 2 and d["Music"][i] < 2 and d["NoSpeech"][i] < 2
                    and split_for(d, i) == split]
            if not keep:  # this file has no speakers in this split
                continue
            self.audio.append(t.column("audio").take(pa.array(keep, pa.int64())).combine_chunks())
            self.index += [(len(self.audio) - 1, j) for j in range(len(keep))]
            labels += [[d[c][i] / 3.0 for c in SEP_COLUMNS] for i in keep]
            groups |= {speaker_group(d["Show"][i], d["EpId"][i]) for i in keep}
            name = Path(f).name
            self.silver_words += [(silver or {}).get(f"{name}:{i}") for i in keep]
        self.labels = np.array(labels, np.float32).reshape(-1, len(SEP_COLUMNS))
        self.groups = sorted(groups)
        self.augment_p, self.rng = augment_p, np.random.default_rng(seed)
        self.n = 3 * SR
        self.frames = (self.n - 400) // FRAME + 1
        self.n_silver = sum(w is not None for w in self.silver_words)

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
        words = self.silver_words[i]
        if words is not None:
            lab, cmask, fmask = frame_targets(words, self.frames)
        else:
            lab, cmask = np.zeros((self.frames, len(CLASSES)), np.float32), np.zeros(len(CLASSES), np.float32)
            fmask = np.ones(self.frames, np.float32)
        return {"wave": torch.from_numpy(y.astype(np.float32)), "clip_labels": torch.from_numpy(self.labels[i]),
                "labels": torch.from_numpy(lab), "mask": torch.from_numpy(fmask),
                "class_mask": torch.from_numpy(cmask)}


def load_silver(paths) -> dict:
    """Verbatim word timings written by scripts/crisper_label.py, keyed by record id."""
    out = {}
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            out[r["id"]] = r
    return out


HELD_OUT_SPEAKERS = {"uclass/M_0065": "test"}  # the speaker we listen to stays unseen


def uclass_split(speaker: str) -> str:
    return HELD_OUT_SPEAKERS.get(speaker) or split_of(speaker)


class UclassSilver(Dataset):
    """Long UCLASS recordings with frame labels from verbatim word timings.

    Training draws random crops, weighted by recording length; validation (fixed=True) tiles
    every recording into consecutive crops so each frame is scored once.
    """

    def __init__(self, records: dict, wav_root, split: str, crop_s: float = 6.0, augment_p: float = 0.5,
                 seed: int = 0, fixed: bool = False):
        paths = {p.name: p for p in Path(wav_root).rglob("*.wav")}
        recs = [r for r in records.values() if r["source"] == "uclass" and r["file"] in paths
                and uclass_split(r["speaker"]) == split]
        self.n = int(crop_s * SR) // FRAME * FRAME
        self.audio, self.targets, self.edges, self.class_mask = [], [], [], None
        for r in recs:
            x = load_wav(paths[r["file"]])
            lab, self.class_mask, fmask = frame_targets(r["words"], len(x) // FRAME)
            self.audio.append((np.clip(x, -1, 1) * 32767).astype(np.int16))  # half the memory of float32
            self.targets.append(lab)
            self.edges.append(fmask)
        self.speakers = sorted({r["speaker"] for r in recs})
        self.lengths = np.array([len(a) for a in self.audio], np.float64)
        self.fixed, self.augment_p, self.rng = fixed, augment_p, np.random.default_rng(seed)
        self.windows = [(k, s) for k, a in enumerate(self.audio) for s in range(0, len(a) - self.n + 1, self.n)]
        if self.class_mask is None:
            self.class_mask = np.zeros(len(CLASSES), np.float32)

    def __len__(self):
        return len(self.windows) if self.fixed else max(1, int(self.lengths.sum() // self.n))

    def __getitem__(self, i):
        if self.fixed:
            k, s = self.windows[i]
        else:
            k = int(self.rng.choice(len(self.audio), p=self.lengths / self.lengths.sum()))
            s = int(self.rng.integers(0, max(1, len(self.audio[k]) - self.n))) // FRAME * FRAME
        y = self.audio[k][s:s + self.n].astype(np.float32) / 32767
        frames = self.n // FRAME
        lab = np.zeros((frames, len(CLASSES)), np.float32)
        part = self.targets[k][s // FRAME:s // FRAME + frames]
        lab[:len(part)] = part
        mask = np.zeros(frames, np.float32)
        edge = self.edges[k][s // FRAME:s // FRAME + frames]
        mask[:len(edge)] = edge
        mask[min(frames, len(y) // FRAME):] = 0.0
        y = np.pad(y, (0, self.n - len(y)))
        if not self.fixed and self.rng.random() < self.augment_p:
            y = augment(y, self.rng)
        return {"wave": torch.from_numpy(y), "labels": torch.from_numpy(lab), "mask": torch.from_numpy(mask),
                "class_mask": torch.from_numpy(self.class_mask)}
