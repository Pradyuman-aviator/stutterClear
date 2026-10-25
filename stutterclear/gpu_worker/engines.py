"""What a GPU worker computes, and on which device. Only the steps that need a GPU run here: for a
recording, the recordings teacher's probabilities and CrisperWhisper's words; for a live stream
(P2), the live teacher's probabilities for one window. The cut logic, the meaning guard, captions
and every render stay on the app server.

torch, transformers and CrisperWhisper are imported inside the functions that use them, so the
worker's --help and its --fake engines (tests, and end-to-end runs on a machine without a GPU)
start without them.
"""
import logging
import os
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np

from .. import gpu_protocol as gp

# What a worker downloads, at the revisions the team's laptop tested, so a new upload under the same
# name can never change what a worker runs. The checkpoints themselves never come from a download:
# Kaggle mounts them privately (kaggle.py), the laptop has them in runs/.
CRISPER_REPO = "nyralabs/CrisperWhisper2.0_large"
CRISPER_REVISION = "f4334f6e8193f2691212d49b20fa12d370e13896"
WAVLM_REVISION = "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"  # microsoft/wavlm-large, both teachers' backbone
TRIES = 3  # downloads (and the loads that download) are tried this often...
WAITS_S = (5.0, 15.0)  # ...this long apart
WARM_WINDOW_S = 4.5  # the longest live window: 4 s of context and 500 ms ahead (strong)
WORD_REP = 1  # student_runtime.CLASSES.index("word_rep"), without importing torch for it

NO_GPU = "This machine has no GPU."
MISSING = "The GPU machine could not find its model files."
NO_DOWNLOAD = "The GPU could not download its models."
NO_LOAD = "The GPU could not load its models."


def log(message: str):
    """One line of the worker's log. Times and counts only: never a token, a word or the audio."""
    print(time.strftime("%H:%M:%S"), message, flush=True)


class ModelsError(RuntimeError):
    """The models could not be loaded. str() is the plain sentence CAPS carries to the server."""


class Plan(NamedTuple):
    """Where each engine runs: a torch device, or None (see plan_devices)."""
    live: str | None
    recordings: str | None


def plan_devices(count: int, mem_gb: float) -> Plan | None:
    """Two GPUs (Kaggle's T4 x2): the live teacher on the first, the recordings teacher and
    CrisperWhisper on the second, so a recording never slows a live stream. One GPU of 14 GB or
    more: everything on it. A smaller one (the team's laptop): the live teacher where the app puts
    it, and recordings exactly as the app runs them on that laptop today (both None: notes.py places
    the teacher and runs CrisperWhisper on the GPU when it fits, else on the CPU). No GPU: None."""
    if count < 1:
        return None
    if count >= 2:
        return Plan("cuda:0", "cuda:1")
    if mem_gb >= 14:
        return Plan("cuda:0", "cuda:0")
    return Plan(None, None)


def gpus() -> list[dict]:
    """The CUDA GPUs this process sees, as HELLO lists them: [{name, mem_gb}]."""
    import torch
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        out.append({"name": gp.printable(p.name, 40), "mem_gb": round(p.total_memory / 1e9, 1)})
    return out


def memory_gb() -> list[float]:
    """GPU memory this process holds, per GPU (HEARTBEAT). Never starts CUDA just to ask."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_initialized():
        return []
    return [round(torch.cuda.memory_reserved(i) / 1e9, 2) for i in range(torch.cuda.device_count())]


def quiet():
    """CrisperWhisper logs the token ids of the loops it repairs at INFO, and Kaggle keeps every
    line a run prints: these libraries say only warnings. Also pins the teachers' backbone download
    (teacher.py reads the revision from the environment)."""
    os.environ.setdefault("STUTTERCLEAR_BACKBONE_REVISION", WAVLM_REVISION)
    for name in ("crisperwhisper", "transformers", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)


def tries(what: str, fn: Callable, n: int = TRIES):
    """fn(), tried up to n times: a fresh machine's first downloads fail now and then."""
    for i in range(n):
        try:
            return fn()
        except Exception as e:
            if i == n - 1:
                raise
            wait = WAITS_S[min(i, len(WAITS_S) - 1)]
            log(f"{what} failed ({type(e).__name__}); trying again in {wait:.0f} s")
            time.sleep(wait)


def fetch(repo: str, revision: str) -> str:
    """A model's files at one exact revision, downloaded once into the Hugging Face cache."""
    from huggingface_hub import snapshot_download
    return tries(f"downloading {repo}", lambda: snapshot_download(
        repo, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.txt"]))


class Background:
    """fn() on a daemon thread (so it never holds up the process's exit); result() waits for it."""

    def __init__(self, fn: Callable):
        self._out, self._err = None, None
        self._thread = threading.Thread(target=self._run, args=(fn,), daemon=True)
        self._thread.start()

    def _run(self, fn):
        try:
            self._out = fn()
        except BaseException as e:
            self._err = e

    def result(self):
        self._thread.join()
        if self._err is not None:
            raise self._err
        return self._out


def load_error(e: BaseException) -> str:
    """The plain sentence for a load that failed."""
    if isinstance(e, ModelsError):
        return str(e)
    text = f"{type(e).__name__} {e}".lower()
    if "out of memory" in text:
        return "The GPU ran out of memory loading its models."
    if isinstance(e, OSError) or any(w in text for w in ("connection", "timed out", "huggingface", "http")):
        return NO_DOWNLOAD
    return NO_LOAD


def fatal(e: BaseException) -> bool:
    """A CUDA error that leaves the process's GPU context unusable (unlike running out of memory):
    only a fresh process can carry on."""
    text = str(e)
    return isinstance(e, RuntimeError) and "out of memory" not in text.lower() and (
        "CUDA error" in text or "device-side assert" in text or "illegal memory access" in text)


def _name(path) -> str:
    """A checkpoint as CAPS names it: its folder and file, never where it is mounted."""
    p = Path(path)
    return f"{p.parent.name}/{p.name}"


def _on(dev):
    """Make `dev` the current CUDA device for this thread: a library that allocates on "cuda" then
    uses the engine's GPU, not the first one."""
    import torch
    return torch.cuda.device(dev) if dev is not None and dev.type == "cuda" else nullcontext()


def _noise(n: int) -> np.ndarray:
    return (np.random.default_rng(0).standard_normal(n) * 0.01).astype(np.float32)


class NotesEngine:
    """A recording's teacher probabilities and CrisperWhisper's words."""

    def __init__(self, device: str | None, live_active: Callable[[], bool] = lambda: False,
                 preload_asr: bool = True):
        self.device = device  # None: the laptop's shared GPU, used as notes.py uses it
        self.live_active = live_active
        self.preload_asr = preload_asr  # shared GPU only: load CrisperWhisper now (no live teacher to make room for)
        self.name = None
        self.model = self.dev = self.asr = None

    def load(self):
        quiet()
        from .. import teacher_runtime as tr
        from ..app import notes
        ckpt = Path(tr.RECORDINGS_CKPT)
        if not ckpt.is_file():
            raise ModelsError(MISSING)
        self.name = _name(ckpt)
        if self.device is None:
            notes.live_running = self.live_active  # a live stream keeps the GPU; a recording then uses the CPU
            tries("loading the recordings teacher", notes._teacher)
            if self.preload_asr:
                tries("loading CrisperWhisper", notes._crisper)
            quiet()
            return
        asr = Background(lambda: fetch(CRISPER_REPO, CRISPER_REVISION))  # downloads while the teacher loads
        self.model, self.dev = tries("loading the recordings teacher", lambda: tr.get_teacher(ckpt, self.device))
        folder = asr.result()
        from crisperwhisper import CrisperWhisperModel
        where = ({"device": "cpu", "compute_type": "float32"} if self.dev.type == "cpu"
                 else {"device": "cuda", "device_index": self.dev.index or 0})
        with _on(self.dev):
            self.asr = notes._lean(CrisperWhisperModel(folder, backend="transformers", **where))
            quiet()
            x = _noise(gp.SR)  # once through both, so the first visitor does not wait for CUDA's first calls
            notes.teacher_probs(x, self.model, self.dev)
            self.asr.transcribe(x, sr=gp.SR, language="en", word_timestamps=True)

    def analyse(self, x: np.ndarray, progress: Callable[[str], None], cancelled: Callable[[], bool],
                language: str = "en") -> tuple | None:
        """(probabilities [frames, 5], words [{w, s, e}], slow, timing) for one recording, or None if
        it was cancelled between the two steps."""
        from ..app import notes
        t0 = time.perf_counter()
        progress("listening")
        if self.device is None:  # the laptop: as notes._process runs it, holding its GPU lock
            with notes._gpu:
                probs = notes.teacher_probs(x)
                t1 = time.perf_counter()
                if cancelled():
                    return None
                progress("transcribing")
                note = {"x": x}
                # the GPU when CrisperWhisper fits there (an idle live teacher makes room), else the CPU
                r = notes._transcribe(note)
            slow = bool(note.get("slow"))
        else:
            import torch
            try:
                with _on(self.dev):
                    probs = notes.teacher_probs(x, self.model, self.dev)
                    t1 = time.perf_counter()
                    if cancelled():
                        return None
                    progress("transcribing")
                    r = self.asr.transcribe(x, sr=gp.SR, language=language, word_timestamps=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()  # the server sends the recording again
                raise
            slow = self.dev.type == "cpu"
        t2 = time.perf_counter()
        words = [{"w": t.word, "s": round(float(t.start), 3), "e": round(float(t.end), 3)} for t in r.words]
        return probs, words, slow, {"teacher_s": round(t1 - t0, 2), "asr_s": round(t2 - t1, 2)}


class LiveEngine:
    """The live teacher's probabilities for one window (P2)."""

    def __init__(self, device: str | None):
        self.device = device  # None: the app's own live teacher device (the laptop shares it with notes)
        self.name = None
        self.window_ms = None  # p90 of a warm window, for CAPS

    def load(self):
        quiet()
        from .. import teacher_runtime as tr
        ckpt = Path(tr.TEACHER_CKPT)
        if not ckpt.is_file():
            raise ModelsError(MISSING)
        self.name = _name(ckpt)
        tries("loading the live teacher", lambda: tr.get_teacher(ckpt, self.device))
        x = _noise(int(WARM_WINDOW_S * gp.SR))
        self.window_probs(x)  # the first GPU call is slow
        times = []
        for _ in range(5):
            t = time.perf_counter()
            self.window_probs(x)
            times.append((time.perf_counter() - t) * 1000)
        self.window_ms = round(float(np.percentile(times, 90)), 1)

    def start(self):
        """Before a stream's first window. On the laptop's shared GPU, CrisperWhisper makes room for the
        live teacher, as it does for a live session in the app (not while it transcribes)."""
        if self.device is None:
            from ..app import notes
            notes.release_gpu()

    def window_probs(self, x: np.ndarray, w0: int = 0) -> np.ndarray:
        """Probabilities [frames, 5] for one window, as teacher_runtime.window_probs on this engine's
        device (w0, the window's first frame, matters only to the stand-in). The model is looked up
        each time: on the laptop a recording may have released it, and it loads again."""
        import torch
        from .. import teacher_runtime as tr
        model, dev = tr.get_teacher(tr.TEACHER_CKPT, self.device)
        with torch.no_grad(), torch.autocast(dev.type, enabled=dev.type == "cuda"), _on(dev):
            wave = torch.from_numpy(np.ascontiguousarray(x, np.float32))[None].to(dev)
            return torch.sigmoid(model(wave).float())[0].cpu().numpy()


class FakeNotesEngine:
    """A stand-in without torch (--fake), for tests and end-to-end runs of the server without a GPU:
    a word repetition at 1.0-1.2 s, and a word every half second."""
    name = "fake"

    def load(self):
        pass

    def analyse(self, x, progress, cancelled, language: str = "en"):
        progress("listening")
        probs = np.full((gp.frames_for(len(x)), gp.N_CLASSES), 0.02, np.float32)
        probs[50:60, WORD_REP] = 0.9
        if cancelled():
            return None
        progress("transcribing")
        words = [{"w": f"w{i}", "s": round(i * 0.5, 3), "e": round(i * 0.5 + 0.4, 3)}
                 for i in range(int(len(x) / gp.SR / 0.5))]
        return probs, words, False, {"teacher_s": 0.0, "asr_s": 0.0}


class FakeLiveEngine:
    """The live stand-in (--fake): a word repetition 1.0-1.2 s into every 5 s of the stream."""
    name = "fake"
    window_ms = 5.0

    def load(self):
        pass

    def start(self):
        pass

    def window_probs(self, x, w0: int = 0):
        n = gp.frames_for(len(x))
        p = np.full((n, gp.N_CLASSES), 0.02, np.float32)
        f = np.arange(w0, w0 + n) % 250
        p[(f >= 50) & (f < 60), WORD_REP] = 0.9
        return p


def make(plan: Plan, *, notes: bool, live: bool, fake: bool = False, live_active: Callable[[], bool] = lambda: False,
         live_device: str | None = None, notes_device: str | None = None) -> tuple:
    """(notes engine, live engine) for this worker, None for one it does not run."""
    if fake:
        return (FakeNotesEngine() if notes else None), (FakeLiveEngine() if live else None)
    rec = notes_device or plan.recordings
    return ((NotesEngine(rec, live_active, preload_asr=not live) if notes else None),
            (LiveEngine(live_device or plan.live) if live else None))
