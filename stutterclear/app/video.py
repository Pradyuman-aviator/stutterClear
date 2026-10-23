"""Clean my video: a talk, a post, a recorded answer goes in; the same video comes out with the
stutters cut from picture and sound together, plus the cut list.

The soundtrack goes through the voice-note engines (the teacher finds the stutters, the verbatim
transcript puts every cut on a word boundary), then ffmpeg keeps the parts between the cuts, with a
10 ms fade at each join so no click is heard. The upload is written to a temporary folder only while
ffmpeg works on it and deleted after; the result is kept in memory for an hour, like a voice note.
ffmpeg comes from imageio-ffmpeg (or the system's ffmpeg).
"""
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from ..config import SAMPLE_RATE as SR
from . import demo, notes

MAX_BYTES = 300_000_000
MIN_KEEP_S = 0.05  # shorter pieces between two cuts are dropped with them
FADE_S = 0.01
KEEP_S = 3600
_jobs: dict = {}
_work = threading.Lock()  # one video is cut at a time: encoding fills the CPU


class VideoError(ValueError):
    pass


def ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def available() -> bool:
    return notes.available() and ffmpeg() is not None


def keep_segments(cuts: list, duration: float) -> list:
    """The parts of the video to keep: everything outside the cuts (sorted, merged), without slivers."""
    keep, pos = [], 0.0
    for c in sorted(cuts, key=lambda c: c["start"]):
        if c["start"] - pos >= MIN_KEEP_S:
            keep.append((round(pos, 3), round(c["start"], 3)))
        pos = max(pos, c["end"])
    if duration - pos >= MIN_KEEP_S:
        keep.append((round(pos, 3), round(duration, 3)))
    return keep


def filter_graph(keep: list, audio: bool = True) -> str:
    """ffmpeg filter_complex: each kept part trimmed from the picture and the sound, faded in and out
    by FADE_S at the joins, then all joined."""
    parts = []
    for i, (a, b) in enumerate(keep):
        parts.append(f"[0:v]trim=start={a}:end={b},setpts=PTS-STARTPTS[v{i}]")
        if audio:
            out = max(0.0, round(b - a - FADE_S, 3))
            parts.append(f"[0:a]atrim=start={a}:end={b},asetpts=PTS-STARTPTS,"
                         f"afade=t=in:d={FADE_S},afade=t=out:st={out}:d={FADE_S}[a{i}]")
    streams = "".join(f"[v{i}]" + (f"[a{i}]" if audio else "") for i in range(len(keep)))
    parts.append(f"{streams}concat=n={len(keep)}:v=1:a={1 if audio else 0}[v]" + ("[a]" if audio else ""))
    return ";".join(parts)


def _run(args: list, timeout: float = 900) -> subprocess.CompletedProcess:
    r = subprocess.run([ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", *args],
                       capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise VideoError(f"ffmpeg failed: {r.stderr.decode(errors='replace').strip()[-300:]}")
    return r


def soundtrack(path: Path) -> np.ndarray:
    """The video's sound as 16 kHz mono float32."""
    r = _run(["-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], timeout=300)
    x = np.frombuffer(r.stdout, np.float32)
    if not len(x):
        raise VideoError("That video has no sound to clean.")
    return x


def submit(data: bytes, name: str = "video.mp4", engine: str = "natural") -> str:
    if engine not in ("natural", "thorough"):
        raise VideoError("Pick the Natural or Thorough engine.")
    expire()
    if sum(j["status"] not in ("done", "error") for j in _jobs.values()) >= 2:
        raise notes.NoteBusy("A video is being cleaned right now. Try again in a few minutes.")
    jid = "video-" + uuid.uuid4().hex[:12]
    suffix = Path(name).suffix.lower() if Path(name).suffix else ".mp4"
    _jobs[jid] = {"status": "queued", "created": time.time(), "error": None, "engine": engine, "out": None}
    threading.Thread(target=_process, args=(jid, data, suffix), daemon=True).start()
    return jid


def _process(jid: str, data: bytes, suffix: str):
    job = _jobs[jid]
    tmp = Path(tempfile.mkdtemp(prefix="stutterclear-video-"))
    try:
        src = tmp / f"in{suffix}"
        src.write_bytes(data)
        del data
        job["status"] = "sound"  # reading the soundtrack
        x = soundtrack(src)
        nid = notes.submit_audio(x)
        job["note"] = nid
        job["status"] = "cleaning"  # the voice-note engines find the stutters
        while (s := notes.status(nid)) and s["status"] not in ("done", "error"):
            time.sleep(0.5)
        if s is None or s["status"] == "error":
            raise VideoError((s or {}).get("error") or "Cleaning the sound failed.")
        tl = demo.render(nid, job["engine"])["timeline"]
        duration = len(x) / SR
        keep = keep_segments(tl["cuts"], duration)
        if not keep:
            raise VideoError("Nothing would be left of that video.")
        job["status"] = "cutting"  # ffmpeg cuts picture and sound
        out = tmp / "clean.mp4"
        with _work:
            # passthrough: the kept frames at their own times. Some ffmpeg builds otherwise re-time the
            # joined picture to 25 fps, duplicating frames (or dropping them from a 30 fps phone video).
            _run(["-i", str(src), "-filter_complex", filter_graph(keep), "-map", "[v]", "-map", "[a]", "-fps_mode", "passthrough",
                  "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
                  "-movflags", "+faststart", str(out)])
        job.update(out=out.read_bytes(), cuts=tl["cuts"], text=tl.get("text"), duration_s=round(duration, 2),
                   clean_s=round(sum(b - a for a, b in keep), 2), status="done")
    except subprocess.TimeoutExpired:
        job["status"], job["error"] = "error", "That video took too long to cut."
    except Exception as e:  # report it on the page instead of failing silently
        job["status"], job["error"] = "error", str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # the upload never stays on disk


def status(jid: str) -> dict | None:
    job = _jobs.get(jid)
    if job is None:
        return None
    out = {k: job.get(k) for k in ("status", "error", "engine", "duration_s", "clean_s", "cuts", "text")}
    return {"id": jid, **out, "slow": bool(job.get("note") and (notes.status(job["note"]) or {}).get("slow"))}


def result(jid: str) -> bytes | None:
    job = _jobs.get(jid)
    return job["out"] if job and job["status"] == "done" else None


def expire():
    now = time.time()
    for jid in [k for k, v in _jobs.items() if now - v["created"] > KEEP_S]:
        _jobs.pop(jid, None)
