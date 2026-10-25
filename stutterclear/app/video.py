"""Clean my video: a talk, a post, a recorded answer goes in; the same video comes out with the
stutters cut from picture and sound together, plus the cut list.

The soundtrack goes through the voice-note engines (the teacher finds the stutters, the verbatim
transcript puts every cut on a word boundary), then ffmpeg keeps the parts between the cuts, with a
10 ms fade at each join so no click is heard. The upload is written to a temporary folder only while
ffmpeg works on it and deleted after; the result is kept in memory for an hour, like a voice note.
ffmpeg comes from imageio-ffmpeg (or the system's ffmpeg).

A video arrives in one request, or in parts of PART_BYTES (Cloud Run takes at most 32 MiB a
request), one upload per visitor at a time. ffmpeg reads a visitor's file with only the video
demuxers allowed (a playlist or another format could make it read other files or fetch URLs), sees
nothing of the server's environment but PATH, runs below the live sessions' priority and, on Linux,
within MEMORY_LIMIT; at most MAX_S and a second of the sound is ever read into this server.
"""
import os
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

MAX_BYTES = int(os.environ.get("STUTTERCLEAR_VIDEO_MAX_MB", 300)) * 10**6
PART_BYTES = 16 * 1024 * 1024
SUFFIXES = (".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi")
INPUT = ["-format_whitelist", "mov,matroska,avi", "-protocol_whitelist", "file,pipe"]  # before every visitor -i
MEMORY_LIMIT = 2 * 1024**3  # bytes of address space; a 1080p phone video cuts in well under 1 GB
MAX_SIDE = 1920  # larger pictures are scaled down to fit (the cut must not take minutes)
MIN_KEEP_S = 0.05  # shorter pieces between two cuts are dropped with them
FADE_S = 0.01
KEEP_S = 3600
MAX_RESULTS = 4  # cleaned videos held in memory; the oldest is forgotten first
MAX_UPLOADS = 2  # uploads in progress at once (one per visitor)
UPLOAD_IDLE_S = 600  # an upload is dropped this long after its last new part...
UPLOAD_MAX_S = 1800  # ...and this long after it began (150 MB at 1 Mbit/s take 20 minutes)
_jobs: dict = {}
_uploads: dict = {}
_lock = threading.Lock()  # _jobs and _uploads, from request threads and job threads
_work = threading.Lock()  # one video is cut at a time: encoding fills the CPU


class VideoError(ValueError):
    pass


class UploadError(VideoError):
    """A part that cannot be taken; code is the HTTP status to answer with."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


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


def _limit(pid: int):
    """Below the live sessions' priority, and on Linux at most MEMORY_LIMIT of memory. Set right after
    the start, not in the child before it (preexec_fn can deadlock a threaded server)."""
    try:
        import resource
        resource.prlimit(pid, resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))
        os.setpriority(os.PRIO_PROCESS, pid, 10)
    except (ImportError, AttributeError, OSError):  # Windows (priority set at the start), or it already ended
        pass


def _run(args: list, timeout: float = 900, max_out: int = 1 << 20) -> subprocess.CompletedProcess:
    """ffmpeg with these arguments. Its output is read as it comes, at most max_out bytes (more is a
    VideoError), and of what it says only the end is kept (a damaged file can make it say a lot)."""
    env = {"PATH": os.environ.get("PATH", "")}  # never the server's keys
    if os.name == "nt":
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
    p = subprocess.Popen([ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", *args], stdin=subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                         creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
    _limit(p.pid)
    err = bytearray()

    def said():
        for chunk in iter(lambda: p.stderr.read(4096), b""):
            err[:] = (err + chunk)[-2048:]

    listener = threading.Thread(target=said, daemon=True)
    listener.start()
    expired = threading.Event()
    timer = threading.Timer(timeout, lambda: (expired.set(), p.kill()))
    timer.start()
    out, too_much = bytearray(), False
    try:
        for chunk in iter(lambda: p.stdout.read(1 << 16), b""):
            if len(out) + len(chunk) > max_out:
                too_much = True
                p.kill()
                break
            out += chunk
        p.wait()
    finally:
        timer.cancel()
        if p.poll() is None:
            p.kill()
            p.wait()
        listener.join(5)
        p.stdout.close()
        p.stderr.close()
    if expired.is_set():
        raise subprocess.TimeoutExpired(p.args, timeout)
    if too_much:
        raise VideoError("ffmpeg wrote more than expected.")
    if p.returncode != 0:
        raise VideoError(f"ffmpeg failed: {err.decode(errors='replace').strip()[-300:]}")
    return subprocess.CompletedProcess(p.args, p.returncode, bytes(out), bytes(err))


def soundtrack(path: Path) -> np.ndarray:
    """The video's sound as 16 kHz mono float32: at most MAX_S and a second of it is read, however long
    the file is (a megabyte of sound can hold hours). A video a moment over MAX_S keeps its first MAX_S."""
    limit = notes.MAX_S + 1
    try:
        r = _run([*INPUT, "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-t", str(limit), "-f", "f32le", "-"],
                 timeout=300, max_out=(limit + 1) * SR * 4)
    except VideoError:  # damaged, or a format not read here (ffmpeg says the same for both)
        raise VideoError("That video could not be read. MP4, MOV, WebM, MKV and AVI files work.") from None
    x = np.frombuffer(r.stdout, np.float32)
    if not len(x):
        raise VideoError("That video has no sound to clean.")
    if len(x) >= limit * SR - SR // 100:  # ffmpeg stopped at the limit: it is longer
        raise VideoError(f"Videos can be up to {notes.MAX_S // 60} minutes long. Trim it first.")
    return x[:notes.MAX_S * SR]


def _suffix(name_or_ext: str) -> str:
    """The file's lowercase extension when it is a video format read here, else .mp4 (ffmpeg looks at
    the content anyway; no file name is kept)."""
    s = str(name_or_ext).strip().lower()
    s = "." + s.rsplit(".", 1)[-1] if "." in s else "." + s
    return s if s in SUFFIXES else ".mp4"


def _check(engine: str):
    if engine not in ("natural", "thorough"):
        raise VideoError("Pick the Natural or Thorough engine.")


def _room():
    """Two videos in progress at most (under _lock)."""
    _expire()
    if sum(j["status"] not in ("done", "error") for j in _jobs.values()) >= 2:
        raise notes.NoteBusy("A video is being cleaned right now. Try again in a few minutes.")


def room():
    """Whether a video could be cleaned now, before it is read: raises NoteBusy if not."""
    with _lock:
        _room()


def _start(src: Path, engine: str, client: str) -> str:
    """A job for the video at src, in its own temporary folder, which the job deletes (under _lock)."""
    jid = "video-" + uuid.uuid4().hex[:12]
    _jobs[jid] = {"status": "queued", "created": time.time(), "error": None, "engine": engine, "out": None,
                  "client": client}
    threading.Thread(target=_process, args=(jid, src), daemon=True).start()
    return jid


def submit(data: bytes, name: str = "video.mp4", engine: str = "natural", client: str = "") -> str:
    """A whole video in one request. client: who sent it, for a GPU worker's fair share."""
    _check(engine)
    with _lock:
        _room()
    folder = Path(tempfile.mkdtemp(prefix="stutterclear-video-"))
    src = folder / f"in{_suffix(name)}"
    src.write_bytes(data)
    with _lock:
        return _start(src, engine, client)


def begin_upload(client: str = "") -> dict:
    """Start a video upload in parts (PUT them in order with add_part, then submit_upload). A visitor
    has one upload at a time: a new one replaces theirs (a retry, or another video chosen). client:
    their address, kept with the upload only."""
    with _lock:
        _expire()
        for uid in [k for k, v in _uploads.items() if client and v["client"] == client]:
            _drop_upload(uid)
        if len(_uploads) >= MAX_UPLOADS:
            raise notes.NoteBusy("Other videos are being uploaded right now. Try again in a few minutes.")
        uid = "up-" + uuid.uuid4().hex[:16]
        now = time.time()
        _uploads[uid] = {"dir": Path(tempfile.mkdtemp(prefix="stutterclear-video-")), "next": 0, "size": 0,
                         "last": None, "at": now, "began": now, "client": client}
    return {"upload": uid, "part_bytes": PART_BYTES, "max_bytes": MAX_BYTES}


def check_part(upload: str, i: int):
    """Before part i is read: the upload exists and expects it (or it is the last part again)."""
    with _lock:
        _expire()
        u = _uploads.get(upload)
        if u is None:
            raise UploadError(404, "That upload expired. Please choose the video again.")
        if i not in (u["next"], u["next"] - 1):
            raise UploadError(409, "Parts of the video arrived out of order. Please try again.")


def cancel_upload(upload: str):
    """The page gave up on an upload (a part that failed, or no room at the end): its place is free."""
    with _lock:
        _drop_upload(upload)


def add_part(upload: str, i: int, data: bytes) -> int:
    """Part i of an upload, in order; returns the bytes received so far. The last part sent again (its
    answer was lost on the way) is taken once."""
    with _lock:  # held while writing, so parts of one upload are appended strictly in order
        _expire()
        u = _uploads.get(upload)
        if u is None:
            raise UploadError(404, "That upload expired. Please choose the video again.")
        if i == u["next"] - 1 and len(data) == u["last"]:
            return u["size"]  # (and no more time for the upload: only new parts keep it going)
        if i != u["next"]:
            raise UploadError(409, "Parts of the video arrived out of order. Please try again.")
        if not data or len(data) > PART_BYTES:
            raise UploadError(400, f"Each part must hold 1 byte to {PART_BYTES // 2**20} MiB.")
        if u["size"] + len(data) > MAX_BYTES:
            raise UploadError(413, f"That video is too large ({MAX_BYTES // 10**6} MB at most).")
        with open(u["dir"] / "in.part", "ab") as f:
            f.write(data)
        u.update(next=i + 1, size=u["size"] + len(data), last=len(data), at=time.time())
        return u["size"]


def submit_upload(upload: str, ext: str = ".mp4", engine: str = "natural", client: str = "") -> str:
    """Clean an uploaded video. ext: its file's extension (the name itself is not kept)."""
    _check(engine)
    with _lock:
        _room()  # busy: the upload stays, so the page can try again in a while
        u = _uploads.get(upload)
        if u is None:
            raise UploadError(404, "That upload expired. Please choose the video again.")
        if not u["size"]:
            raise UploadError(400, "Nothing of that video arrived. Please try again.")
        del _uploads[upload]  # the job owns its folder now
        src = u["dir"] / f"in{_suffix(ext)}"
        (u["dir"] / "in.part").rename(src)
        return _start(src, engine, client)


def _process(jid: str, src: Path):
    job = _jobs[jid]
    try:
        job["status"] = "sound"  # reading the soundtrack
        x = soundtrack(src)
        nid = notes.submit_audio(x, client=job.pop("client", ""))  # (a GPU worker's fair share; not kept)
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
        out = src.parent / "clean.mp4"
        fit = (f"[v]scale=w='min({MAX_SIDE},iw)':h='min({MAX_SIDE},ih)':force_original_aspect_ratio=decrease"
               f":force_divisible_by=2[fit]")
        with _work:
            # passthrough: the kept frames at their own times. Some ffmpeg builds otherwise re-time the
            # joined picture to 25 fps, duplicating frames (or dropping them from a 30 fps phone video).
            _run([*INPUT, "-i", str(src), "-filter_complex", filter_graph(keep) + ";" + fit, "-map", "[fit]",
                  "-map", "[a]", "-fps_mode", "passthrough", "-threads", "2",
                  "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
                  "-movflags", "+faststart", str(out)])
        job.update(out=out.read_bytes(), cuts=tl["cuts"], text=tl.get("text"), duration_s=round(duration, 2),
                   clean_s=round(sum(b - a for a, b in keep), 2), status="done")
        _keep_results()
    except subprocess.TimeoutExpired:
        job["status"], job["error"] = "error", "That video took too long to cut."
    except Exception as e:  # report it on the page instead of failing silently
        job["status"], job["error"] = "error", str(e)
    finally:
        shutil.rmtree(src.parent, ignore_errors=True)  # the upload never stays on disk


def _keep_results():
    """Forget the oldest cleaned videos past MAX_RESULTS (they are big)."""
    with _lock:
        done = [k for k, v in _jobs.items() if v["status"] == "done"]  # oldest first
        for jid in done[:max(0, len(done) - MAX_RESULTS)]:
            _jobs.pop(jid, None)


def _get(jid: str) -> dict | None:
    """The job, unless its hour is over: then it is forgotten now, not at the next video (under _lock)."""
    job = _jobs.get(jid)
    if job is not None and time.time() - job["created"] > KEEP_S:
        _jobs.pop(jid, None)
        return None
    return job


def status(jid: str) -> dict | None:
    with _lock:
        job = _get(jid)
    if job is None:
        return None
    out = {k: job.get(k) for k in ("status", "error", "engine", "duration_s", "clean_s", "cuts", "text")}
    note = None
    if job.get("note"):
        notes.touch(job["note"])  # the page still waits for it
        note = notes.status(job["note"])
    cleaning = job["status"] == "cleaning" and note is not None
    return {"id": jid, **out, "slow": bool(note and note.get("slow")),
            "note": note["status"] if cleaning else None, "ahead": note.get("ahead") if cleaning else None}


def result(jid: str) -> bytes | None:
    with _lock:
        job = _get(jid)
    return job["out"] if job and job["status"] == "done" else None


def _drop_upload(uid: str):
    """Under _lock."""
    u = _uploads.pop(uid, None)
    if u is not None:
        shutil.rmtree(u["dir"], ignore_errors=True)


def _expire():
    """Forget videos past KEEP_S, and uploads idle past UPLOAD_IDLE_S or begun UPLOAD_MAX_S ago (with
    their folders). Under _lock."""
    now = time.time()
    for jid in [k for k, v in _jobs.items() if now - v["created"] > KEEP_S]:
        _jobs.pop(jid, None)
    for uid in [k for k, v in _uploads.items() if now - v["at"] > UPLOAD_IDLE_S or now - v["began"] > UPLOAD_MAX_S]:
        _drop_upload(uid)


def expire():
    """The server calls this every minute: a quiet server keeps nothing past its time either."""
    with _lock:
        _expire()


def sweep():
    """At startup: delete video folders an earlier run left behind (a crash mid-cut), if older than
    an hour (a younger one may belong to another server running here)."""
    now = time.time()
    for d in Path(tempfile.gettempdir()).glob("stutterclear-video-*"):
        try:
            if now - d.stat().st_mtime > 3600:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass
