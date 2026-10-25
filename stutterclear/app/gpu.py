"""GPU on demand: the app server's side of the GPU worker (stutterclear/gpu_worker/).

This server may have no GPU (a CPU instance in the cloud). When a visitor's voice note, Dictate or
video needs one, the broker wakes a GPU worker: the team's laptop if it stands by, else a machine
the launcher starts (a private Kaggle notebook, through the Kaggle API). The worker connects out to
/ws/gpu, so nothing has to be opened where it runs, takes the recordings one at a time and sends back
the teacher's probabilities and CrisperWhisper's words; the cut decisions, the meaning guard,
captions and ffmpeg stay here. It is told to sleep again once nobody has used it for a while. With
STUTTERCLEAR_GPU_LIVE=1 it also runs the live teacher for "Best (GPU)" sessions
(stutterclear/remote_detector.py).

Wakes are rationed: one at a time, a few launches an hour and a day (per visitor too), and never past
our share of Kaggle's weekly GPU quota, which Kaggle itself keeps (so it survives a restart). All
other state lives in this process's memory (one Cloud Run instance), behind one lock that no
launcher call (network) and no frame encoding ever runs under; every send goes through the worker
connection's own event loop. The wire is stutterclear/gpu_protocol.py. Without
STUTTERCLEAR_GPU_SECRET none of this is on and the app behaves as before.
"""
import asyncio
import math
import os
import queue
import subprocess
import threading
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

import numpy as np
from starlette.websockets import WebSocketDisconnect

from .. import gpu_protocol as gp

ROOT = Path(__file__).resolve().parents[2]

# What visitors read, as the page shows it
NOT_SET_UP = "No GPU is set up on this server."
BROKEN = "The GPU is not set up correctly on this server."
NOT_CONNECTED = "The team's GPU is not connected right now."
LAPTOP_SILENT = "The team's computer did not answer."
LOAD_TIMEOUT = "The GPU took too long to load."
LOAD_FAILED = "The GPU could not load its models."
STOPPED_EARLY = "The GPU machine stopped before it was ready."
KEEPS_FAILING = "The GPU keeps failing."
START_FAILED = "The GPU could not be started."
STOPPED_TWICE = "The GPU stopped twice while cleaning this recording. Please try again later."
CANNOT_CLEAN = "The GPU could not clean this recording."
NO_NOTES = "The GPU that is awake cannot clean recordings right now."
ABANDONED = "This recording waited without the page open, so it was dropped."
TOO_LATE = "The GPU did not come up in time. Please try again."
CLIENT_BUSY = "You already have a recording being cleaned. Wait for it to finish."
CLIENT_AUDIO = "That is a lot of audio for one hour; try again later."
CLIENT_AUDIO_DAY = "That is a lot of audio for one day; try again tomorrow."
WAKES_HOUR = "The GPU has been woken many times this hour. Try again in a short while."
WAKES_DAY = "The GPU has been woken many times today. Try again in {m} minutes."
CLIENT_LIVE = "You have used Best (GPU) several times this hour. Try again later, or pick Rule-based."

BULK_PACE_S = 0.05  # while a live stream is open, one recording part per this long at most
RESERVE_S = 120.0  # a visitor's admitted recording holds their turn until it is queued
LAPTOP = "the team's laptop"
RECORDING = ("voice-note", "dictate", "video")  # wakes for a recording that is on its way


def _log(msg: str):
    # One write per line, so lines from the event loop and the tick thread never interleave. (uvicorn
    # logs at warning level and nothing configures logging: print is what reaches the server's log.)
    print(f"gpu: {msg}\n", end="", flush=True)


def _minutes(s: float) -> int:
    return max(1, math.ceil(s / 60))


@cache
def server_commit() -> str:
    """The code this server runs: STUTTERCLEAR_COMMIT (the Cloud Run image has it), else the checkout's
    HEAD, else "dev". A launched worker fetches exactly this commit."""
    c = (os.environ.get("STUTTERCLEAR_COMMIT") or "").strip()
    if c:
        return c
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and gp.SHA_RE.fullmatch(r.stdout.strip()):
            return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "dev"


def version() -> str:
    """The commit's first 12 hex digits, or "dev", as /api/health and /api/info show it."""
    c = server_commit()
    return c[:12] if gp.SHA_RE.fullmatch(c) else "dev"


class GpuError(RuntimeError):
    """A recording the GPU worker did not clean; str() is a plain sentence for the page."""


class GpuUnavailable(GpuError):
    """No GPU can take a recording now; str() says why."""


class WakeRefused(Exception):
    """A wake that is not allowed now: code is the HTTP status, str() the plain sentence."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def _env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        _log(f"{name} is not a number; using {default}")
        return default


@dataclass
class Settings:
    """The broker's timers and limits; from_env() reads the ones that can be set (docs/DEPLOY.md)."""
    idle_exit_s: float = field(default_factory=lambda: gp.IDLE_EXIT_S)  # sleep this long after real work...
    idle_unused_s: float = field(default_factory=lambda: gp.IDLE_UNUSED_S)  # ...or after a wake no work came to
    wake_timeout_s: float = 720.0  # a wake gives up when no worker said hello by then
    load_timeout_s: float = 360.0  # hello to models loaded
    note_wait_s: float = 1200.0  # the longest a recording waits for a worker (wake + load fit inside)
    reconnect_grace_s: float = 90.0  # a worker that dropped may come back this soon...
    reconnect_look_s: float = 5.0  # ...unless its machine is seen gone this long after the drop
    laptop_seen_s: float = 45.0  # a laptop that polled this recently stands by
    laptop_wait_s: float = 40.0  # a wake waits this long for it before launching
    quota_cache_s: float = 600.0
    launch_poll_s: float = 10.0
    stop_grace_s: float = 60.0  # a worker told to stop that has not said bye by then is closed
    live_start_s: float = 5.0  # a live link nobody started by then is closed...
    live_silent_s: float = 15.0  # ...and one that carried no audio for this long
    week_hours: float = 20.0  # our cap of Kaggle's weekly GPU hours...
    reserve_hours: float = 2.0  # ...always leaving this much of them unused...
    day_hours: float | None = None  # ...and at most this many in one UTC day (None: a quarter of week_hours)
    max_wake_s: float = 5400.0  # a worker takes no new work this long after it was ready: the next work wakes
    # a fresh one, which counts against the limits below (so steady use cannot keep one machine up for hours)
    wakes_per_hour: int = 4  # launches
    wakes_per_day: int = 10
    client_wakes_per_day: int = 3
    laptop_wakes_per_hour: int = 12  # a standby laptop's wakes cost no GPU hours, but each holds this server
    laptop_wakes_per_day: int = 60  # up (and billed) for a while...
    client_laptop_wakes_per_hour: int = 4
    button_unused_s: float = 60.0  # ...so one woken by the Wake button alone sleeps this soon if no work comes
    client_audio_s_per_hour: float = 600.0
    client_audio_s_per_day: float = 1800.0
    client_live_per_hour: int = 3
    typical_s: tuple = (120, 240)  # how long a wake usually takes, as the page says it
    live: bool = False  # offer Best (GPU) live through the worker
    hosted: bool = False
    roles: tuple | None = None  # token roles that may connect; None: the launcher's role and "laptop"
    poll_s: float = 1.0  # how often a waiting recording reports its place

    @classmethod
    def from_env(cls) -> "Settings":
        typical = (os.environ.get("STUTTERCLEAR_GPU_TYPICAL_S") or "").split(",")
        try:
            typical_s = (int(typical[0]), int(typical[1])) if len(typical) == 2 else (120, 240)
        except ValueError:
            typical_s = (120, 240)
        roles = tuple(r.strip() for r in (os.environ.get("STUTTERCLEAR_GPU_ROLES") or "").split(",") if r.strip())
        idle = _env("STUTTERCLEAR_GPU_IDLE_S", gp.IDLE_EXIT_S)
        week = _env("STUTTERCLEAR_GPU_WEEK_HOURS", 20.0)
        return cls(idle_exit_s=idle, idle_unused_s=min(gp.IDLE_UNUSED_S, idle),  # an unused wake never stays longer
                   wake_timeout_s=_env("STUTTERCLEAR_GPU_WAKE_TIMEOUT_S", 720.0),
                   week_hours=week, reserve_hours=_env("STUTTERCLEAR_GPU_RESERVE_HOURS", 2.0),
                   day_hours=_env("STUTTERCLEAR_GPU_DAY_HOURS", week / 4),
                   max_wake_s=_env("STUTTERCLEAR_GPU_MAX_WAKE_S", 5400.0),
                   wakes_per_hour=int(_env("STUTTERCLEAR_GPU_WAKES_PER_HOUR", 4)),
                   wakes_per_day=int(_env("STUTTERCLEAR_GPU_WAKES_PER_DAY", 10)),
                   client_wakes_per_day=int(_env("STUTTERCLEAR_GPU_CLIENT_WAKES_PER_DAY", 3)),
                   laptop_wakes_per_hour=int(_env("STUTTERCLEAR_GPU_LAPTOP_WAKES_PER_HOUR", 12)),
                   client_audio_s_per_hour=_env("STUTTERCLEAR_GPU_CLIENT_AUDIO_S_PER_HOUR", 600.0),
                   client_audio_s_per_day=_env("STUTTERCLEAR_GPU_CLIENT_AUDIO_S_PER_DAY", 1800.0),
                   client_live_per_hour=int(_env("STUTTERCLEAR_GPU_CLIENT_LIVE_PER_HOUR", 3)),
                   typical_s=typical_s, live=os.environ.get("STUTTERCLEAR_GPU_LIVE") == "1",
                   hosted=os.environ.get("STUTTERCLEAR_HOSTED") == "1", roles=roles or None)


@dataclass(eq=False)
class Job:
    """One recording for the GPU: queued, then sent to a worker, running, done or failed."""
    id: str
    x: np.ndarray | None  # dropped once done: the note keeps the only copy
    samples: int
    client: str
    created: float
    state: str = "queued"
    sends: int = 0  # a job is sent at most twice
    seq: int = 0  # bumped whenever it is sent or taken back, so its parts still queued stop
    stage: str | None = None
    conn: "Conn | None" = None
    sent_at: float = 0.0
    result: tuple | None = None
    error: str | None = None
    charge: list | None = None  # its [time, seconds] in the visitor's audio count: given back if never sent
    done: threading.Event = field(default_factory=threading.Event)


@dataclass(eq=False)
class Wake:
    """A wake in progress, until a worker says hello."""
    id: str
    started: float
    reason: str
    client: str
    step: str = "asking"  # laptop | asking | queued | starting
    token: str | None = None
    laptop_until: float | None = None
    version: int | None = None  # the run the launcher started or found
    launched: bool = False  # the launcher started or found one: how it is doing is polled
    adopted: bool = False
    fresh: bool = False  # this wake started a new run: only its own worker (or the laptop) may connect
    repushed: bool = False
    next_poll: float = 0.0
    poll_every: float = 10.0


class Conn:
    """One worker's WebSocket. Its lanes are used only on its event loop; send() gets there from any
    thread. Frames go out control first, then live, then bulk (a recording's parts)."""

    def __init__(self, socket, loop, token: gp.Token, hello: dict, now: float):
        self.socket, self.loop = socket, loop
        self.role, self.wake, self.worker = token.role, token.wake, hello["worker"]
        self.where = gp.printable(hello["where"], gp.MAX_WHERE) or (LAPTOP if token.role == "laptop" else "a GPU")
        self.commit, self.deadline = hello["commit"], hello["deadline"]
        self.hello_at = self.started = now  # started: when its wake began (elapsed time on the page)
        self.caps: dict = {}
        self.ready_at: float | None = None
        self.used = False  # real work came since it was ready
        self.unused_s: float | None = None  # how soon it sleeps if no work comes at all (the broker's default if None)
        self.last_work: float | None = None
        self.stopping: float | None = None  # when it was told to stop...
        self.stop_reason = "idle"  # ...and why
        self.bye: str | None = None
        self.version: int | None = None  # its Kaggle run, when this server launched it
        self.rtts: deque = deque(maxlen=12)  # round trips of the last pings, ms
        self.rtt_logged = False
        self.window_ms: float | None = None  # the worker's p90 GPU time per live window
        self.live: LiveLink | None = None
        self.heard = loop.time()
        self.closed = False
        self.close_code: int | None = None
        self.reader: asyncio.Task | None = None
        self.control: deque = deque()
        self.live_q: deque = deque()
        self.bulk: deque = deque()  # iterators of frames: a job's header and parts
        self.bulk_at = 0.0
        self.wakeup = asyncio.Event()

    def send(self, lane: str, item) -> bool:
        """Queue a frame (or a job's frames) on a lane, from any thread."""
        if self.closed:
            return False
        try:
            self.loop.call_soon_threadsafe(self._put, lane, item)
            return True
        except RuntimeError:  # its event loop is gone, and the connection with it
            return False

    def _put(self, lane: str, item):
        if not self.closed:
            getattr(self, lane).append(item)
            self.wakeup.set()

    def abort(self, code: int):
        """Close it from any thread (a worker that does not leave when told)."""
        def stop():
            self.close_code = self.close_code or code
            if self.reader is not None:
                self.reader.cancel()
        with suppress(RuntimeError):
            self.loop.call_soon_threadsafe(stop)

    def _next(self, now: float):
        if self.control:
            return self.control.popleft()
        if self.live_q:
            return self.live_q.popleft()
        while self.bulk and now >= self.bulk_at:
            frame = next(self.bulk[0], None)
            if frame is None:
                self.bulk.popleft()
                continue
            if self.live is not None:  # a live stream is open: never queue seconds of a recording ahead of it
                self.bulk_at = now + BULK_PACE_S
            return frame
        return None

    async def write(self):
        loop = asyncio.get_running_loop()
        while True:
            frame = self._next(loop.time())
            if frame is None:
                self.wakeup.clear()
                wait = max(0.0, self.bulk_at - loop.time()) if self.bulk else None
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self.wakeup.wait(), wait)
                continue
            await self.socket.send_bytes(frame)


class LiveLink:
    """A live stream on the worker for one RemoteTeacherDetector (stutterclear/remote_detector.py).
    Never blocks: send() encodes in the caller's thread and hops onto the connection's event loop.
    Replies arrive on `replies`: ("probs", seq, f0, f1, rows, ms), or ("closed", "gone" | "error")
    when the stream ends without the detector asking."""

    def __init__(self, broker: "Broker", conn: Conn, stream: str, extra_delay_ms: int, now: float):
        self.broker, self.conn, self.stream = broker, conn, stream
        self.extra_delay_ms = extra_delay_ms
        self.replies: queue.Queue = queue.Queue()
        self.ready = threading.Event()  # the worker's model is warm (LIVE_READY), or the link closed
        self.ready_at: float | None = None
        self.opened = now
        self.started = False
        self.last_audio: float | None = None
        self.closed = False

    def start(self, lookahead_ms: float, window: int, hop_frames: int, mode: str) -> bool:
        if self.closed:
            return False
        frame = gp.encode({"t": gp.LIVE_OPEN, "stream": self.stream, "mode": mode, "lookahead_ms": lookahead_ms,
                           "window": int(window), "hop_frames": int(hop_frames)})
        self.started = self.conn.send("control", frame)
        return self.started

    def send(self, seq: int, start: int, end: int, f0: int, f1: int, audio: np.ndarray) -> bool:
        """Ask for frames [f0, f1) of the window [start, end); audio: its samples not sent before.
        False when the link is closed or the worker is too far behind."""
        if self.closed or len(self.conn.live_q) >= gp.LIVE_QUEUE_MAX:
            return False
        try:
            frame = gp.encode({"t": gp.LIVE_AUDIO, "stream": self.stream, "seq": int(seq), "start": int(start),
                               "end": int(end), "f0": int(f0), "f1": int(f1)}, gp.pcm16(audio))
        except gp.ProtocolError:
            return False
        self.last_audio = self.broker.clock()
        return self.conn.send("live_q", frame)

    def close(self):
        """Idempotent; the worker drops the stream."""
        self.broker._close_link(self, None)


class _Close(Exception):
    def __init__(self, code: int | None, reason: str = ""):
        super().__init__(reason)
        self.code, self.reason = code, reason


class Broker:
    def __init__(self, secret: str | None, launcher=None, settings: Settings | None = None, clock=time.time,
                 commit: str | None = None, autostart: bool = True, public_url: str | None = None,
                 broken: str | None = None):
        """secret: STUTTERCLEAR_GPU_SECRET (None: the feature is off). launcher: starts workers
        (gp.Launcher), or None: only a standby laptop can serve. public_url: this server's URL, which
        launched workers dial and their tokens are signed for. broken: why a launcher that was asked
        for could not be built. autostart=False: no threads at all; the caller drives tick() (tests)."""
        self.secret = secret if secret and len(secret) >= gp.MIN_SECRET else None
        self.launcher = launcher
        self.s = settings or Settings()
        self.clock = clock
        self.commit = commit if commit is not None else server_commit()
        self.autostart = autostart
        self.public_url = public_url
        self.roles = frozenset(self.s.roles or ((launcher.role,) if launcher else ()) + ("laptop",))
        self._lock = threading.RLock()
        self._launcher_lock = threading.Lock()  # one launcher call at a time (they keep state)
        self._started = False
        self._active: Conn | None = None
        self._others: list[Conn] = []  # told to stop, or draining, while they finish their work
        self._wake: Wake | None = None
        self._reconnect_until: float | None = None
        self._reconnect_check = math.inf  # when to look at a dropped worker's machine next
        self._reconnect_checked = False  # its machine was seen running once: the wait was extended
        self._last_worker: Conn | None = None  # the worker that dropped, while it may come back
        self._resting_until: float | None = None
        self._resting_why = "week"  # week | day: whose share of Kaggle's hours is used up
        self._unavailable: tuple | None = (broken, None) if broken else None  # (reason, retry at or None)
        self._failures = 0  # launches failed in a row (backoff)
        self._crashes: deque = deque(maxlen=4)
        self._queue: deque[Job] = deque()
        self._inflight: dict[str, Job] = {}
        self._jobs = 0
        self._streams = 0
        self._ended: deque = deque(maxlen=100)  # wake ids that said bye or were told to stop here
        self._retired: set = set()  # Kaggle versions told to stop or seen leave: never adopted again
        self._versions: dict[str, int] = {}  # wake id -> the run launched or found for it (the last 100)
        self._standby_at: float | None = None
        self._quota: gp.Quota | None = None  # Kaggle's last reading...
        self._quota_at = -math.inf  # ...taken then
        self._quota_tried = -math.inf  # the last time it was asked for
        self._quota_busy = False
        self._day_base: tuple | None = None  # (UTC day, Kaggle's hours used at its first reading that day)
        self._polling = False
        self._launches: deque = deque()  # launch times (the last day or so)
        self._client_launches: dict[str, deque] = {}
        self._laptop_wakes: deque = deque()  # the standby laptop's wakes (the last day)
        self._client_laptop_wakes: dict[str, deque] = {}
        self._client_audio: dict[str, deque] = {}  # [time, seconds] (the last day)
        self._client_live: dict[str, deque] = {}
        self._reserved: dict[str, deque] = {}  # admitted recordings not queued yet: their [time, seconds]
        self._seen = ("off", None, clock())  # the last state, step and since when
        self._refused = [0, -math.inf]  # refused worker connections since the last log line, and when
        self._pruned_at = -math.inf

    @classmethod
    def from_env(cls) -> "Broker":
        """The broker this server runs, from its environment (docs/DEPLOY.md lists every variable)."""
        secret = gp.read_secret("STUTTERCLEAR_GPU_SECRET")
        if secret is None:
            return cls(None)
        if len(secret) < gp.MIN_SECRET:
            _log("off: STUTTERCLEAR_GPU_SECRET is shorter than 32 characters")
            return cls(None)
        _log(f"worker secret key id {gp.key_id(secret)}")
        settings = Settings.from_env()
        name = (os.environ.get("STUTTERCLEAR_GPU_LAUNCHER") or "").strip().lower()
        url = os.environ.get("STUTTERCLEAR_PUBLIC_URL") or ""
        launcher, broken = None, None
        if name:
            try:
                if name not in ("kaggle", "local"):
                    raise ValueError("STUTTERCLEAR_GPU_LAUNCHER must be kaggle or local")
                try:
                    url = gp.server_url(url)
                except ValueError:
                    raise ValueError("STUTTERCLEAR_PUBLIC_URL must be this server's https:// URL") from None
                if name == "kaggle":
                    from ..gpu_worker.kaggle import KaggleLauncher
                    launcher = KaggleLauncher.from_env()
                else:
                    from ..gpu_worker.local import LocalLauncher
                    launcher = LocalLauncher.from_env()
            except Exception as e:  # configured, but no launches: one line naming what is missing (never a value)
                said = f": {gp.printable(e, 160)}" if isinstance(e, (ValueError, ImportError)) else ""
                _log(f"the {name} launcher is not set up ({type(e).__name__}{said})")
                broken = BROKEN
                settings.roles = settings.roles or ((name, "laptop") if name in ("kaggle", "local") else ("laptop",))
        return cls(secret, launcher, settings, public_url=url or None, broken=broken)

    # ---- public ---------------------------------------------------------------------------------

    def configured(self) -> bool:
        return self.secret is not None

    def live_offered(self) -> bool:
        return self.configured() and self.s.live

    def start(self) -> None:
        """The tick thread (once; the server's startup calls it) and a first look at the quota."""
        with self._lock:
            if self._started or not self.configured() or not self.autostart:
                return
            self._started = True
        threading.Thread(target=self._ticking, daemon=True, name="gpu-tick").start()
        if self.launcher is not None:
            self._spawn(self._refresh_quota)

    def _ticking(self):
        while True:
            time.sleep(1.0)
            self.tick()

    def status(self, client: str = "") -> dict:
        with self._lock:
            return self._status(self.clock(), client)

    def info(self) -> dict | None:
        """What /api/info says about the GPU (static: it never changes while a page is open)."""
        if not self.configured():
            return None
        return {"where": self.launcher.where if self.launcher is not None else LAPTOP,
                "typical_s": list(self.s.typical_s), "live": self.live_offered()}

    def health(self) -> str | None:
        if not self.configured():
            return None
        with self._lock:
            return self._state(self.clock())[0]

    def wake(self, reason: str, client: str = "", job: bool = False) -> dict:
        """Wake a worker for `reason` (one of gp.REASONS). A no-op while one is awake or waking.
        Raises WakeRefused with the plain reason when it is not allowed now. job: queued work needs it
        (then a worker that is about to stop does not count as awake)."""
        if reason not in gp.REASONS:
            raise ValueError("unknown reason")
        if not self.configured():
            raise WakeRefused(404, NOT_SET_UP)
        self._autostart()
        with self._lock:
            now = self.clock()
            if self._wake is not None or not self._needs_worker(now, job):
                return self._status(now, client)  # single flight: awake, waking or coming back
            if self._laptop_standing_by(now):  # free and seconds away: always first
                if not job:  # (recordings already waiting were rationed when they were admitted)
                    cap = self._laptop_cap(now, client)
                    if cap is not None:
                        raise WakeRefused(429, cap)
                    self._laptop_wakes.append(now)
                    self._client_laptop_wakes.setdefault(client, deque()).append(now)
                self._wake = Wake(gp.new_wake("laptop-"), now, reason, client, step="laptop",
                                  laptop_until=now + self.s.laptop_wait_s)
                _log(f"waking the team's laptop ({reason})")
                return self._status(now, client)
            refusal = self._refusal(now, client, per_client=not job)
            if refusal is not None:
                raise WakeRefused(*refusal)
            launch = self._begin_launch(now, reason, client)
            status = self._status(now, client)
        self._spawn(*launch)
        return status

    def accepting(self, client: str = "") -> str | None:
        """None when a recording can be queued now (a worker is awake or waking, a laptop stands by, or
        a wake would be allowed), else the plain reason."""
        if not self.configured():
            return NOT_SET_UP
        with self._lock:
            return self._accepting(self.clock(), client)

    def admit(self, client: str, seconds: float, dry_run: bool = False) -> str | None:
        """A visitor's fair share: one recording at a time, and so much audio an hour and a day. None
        when this one may go (and, unless dry_run, it is counted), else the plain reason."""
        with self._lock:
            now = self.clock()
            if self._client_busy(client, now):
                return CLIENT_BUSY
            spent = self._recent(self._client_audio.setdefault(client, deque()), now, 86400, key=lambda e: e[0])
            if sum(s for t, s in spent if now - t < 3600) + seconds > self.s.client_audio_s_per_hour:
                return CLIENT_AUDIO
            if sum(s for _, s in spent) + seconds > self.s.client_audio_s_per_day:
                return CLIENT_AUDIO_DAY
            if not dry_run:
                entry = [now, seconds]
                spent.append(entry)
                self._reserved.setdefault(client, deque()).append(entry)
        return None

    def admit_live(self, client: str) -> str | None:
        """Best (GPU) live sessions per visitor and hour. Only sessions that will really use the GPU
        count (one that starts while it sleeps runs rule-based)."""
        with self._lock:
            now = self.clock()
            if self._live_refusal(now) is not None:
                return None
            used = self._recent(self._client_live.setdefault(client, deque()), now, 3600)
            if len(used) >= self.s.client_live_per_hour:
                return CLIENT_LIVE
            used.append(now)
        return None

    def standby(self, token: str | None, host: str) -> bool | None:
        """GET /api/gpu/wanted from a standby laptop: whether it should connect now; None to refuse."""
        if not self.configured() or "laptop" not in self.roles:
            return None
        t = gp.check_token(self.secret, token, host, self.clock())
        if t is None or t.role != "laptop":
            return None
        with self._lock:
            self._standby_at = self.clock()
            last = self._last_worker
            ended = (self._active is None and self._reconnect_until is not None and last is not None
                     and (last.role == "laptop" or bool(self._queue)))
            if ended:
                # Waiting for a dropped worker is pointless now: it was this laptop, back to standing by
                # after it lost its connection, or recordings are waiting and the laptop can take them at
                # once (a Kaggle worker that comes back later is turned away, its run never adopted).
                self._reconnect_until, self._last_worker = None, None
                if last.version is not None:
                    self._retired.add(last.version)
                _log("the team's laptop is standing by: not waiting for the worker that dropped")
        if ended:
            self._kick()  # recordings waiting: the laptop is woken for them now
        with self._lock:
            return self._needs_worker(self.clock(), True) and (self._wake is not None or bool(self._queue))

    def analyse(self, x: np.ndarray, on_update, abandoned=lambda: False, client: str = "") -> tuple:
        """The teacher's probabilities [frames, 5], CrisperWhisper's words [{w, s, e}] and whether they
        were written slowly, for one recording, from a GPU worker (woken if needed). Blocks its caller
        (a note's own thread); on_update(stage, ahead) reports its place: ("queued", n), then
        "listening" and "transcribing". abandoned(): the page stopped asking, so a recording still
        waiting for a worker is dropped. Raises GpuUnavailable or GpuError with a plain sentence."""
        if not self.configured():
            raise GpuUnavailable(NOT_SET_UP)
        x = np.asarray(x, np.float32).ravel()
        if not 0 < len(x) <= gp.MAX_JOB_S * gp.SR:
            raise GpuError(f"Recordings can be up to {gp.MAX_JOB_S // 60} minutes long.")
        with self._lock:
            now = self.clock()
            charge = self._take_reservation(client)
            reason = self._accepting(now, client)
            if reason is not None:
                self._refund(client, charge)
                raise GpuUnavailable(reason)
            self._jobs += 1
            job = Job(f"job-{self._jobs}", x, len(x), client, now, charge=charge)
            self._queue.append(job)
        try:
            self.wake("job", client, job=True)
        except WakeRefused as e:
            self._withdraw(job, str(e))
        self._dispatch()
        last = None
        while not job.done.is_set():
            with self._lock:
                now = self.clock()
                if job.state == "queued":
                    update = ("queued", list(self._queue).index(job) + len(self._inflight))
                    waiting = not self._worker_ready()
                else:
                    update, waiting = (job.stage or "listening", None), False
            if job.done.is_set():
                break
            if update != last:
                on_update(*update)
                last = update
            if update[0] == "queued" and waiting and abandoned():
                self._withdraw(job, ABANDONED)
            elif update[0] == "queued" and now - job.created > self.s.note_wait_s:
                self._withdraw(job, TOO_LATE)
            job.done.wait(self.s.poll_s)
        if job.error is not None:
            raise GpuError(job.error)
        return job.result

    def open_live(self, mode: str) -> "LiveLink | None":
        """A live stream on the ready worker for one session, or None (asleep, busy with another
        session, about to stop, or too far away for live use)."""
        if not self.live_offered() or mode not in gp.MODES:
            return None
        with self._lock:
            now = self.clock()
            if self._live_refusal(now) is not None:
                return None
            c = self._active
            self._streams += 1
            link = LiveLink(self, c, f"live-{self._streams}", self._extra_delay(c), now)
            c.live, c.used, c.last_work = link, True, now
            return link

    # ---- the worker's connection ----------------------------------------------------------------

    async def serve(self, socket) -> None:
        """The whole /ws/gpu handler: a worker whose token is not good for this server is refused
        before the socket is accepted (HTTP 403); then HELLO, admission, and the connection's life."""
        token = None
        if self.configured():
            token = gp.check_token(self.secret, socket.headers.get(gp.TOKEN_HEADER), socket.headers.get("host", ""),
                                   self.clock())
        refusal = self._handshake_refusal(token)
        if refusal is not None:
            self._log_refused(refusal)
            await socket.close(code=gp.CLOSE_BAD)
            return
        self._autostart()
        await socket.accept()
        loop = asyncio.get_running_loop()
        try:
            hello = self._hello(await asyncio.wait_for(socket.receive(), gp.HELLO_S), token)
            conn = Conn(socket, loop, token, hello, self.clock())
            self._admit(conn)
        except (asyncio.TimeoutError, gp.ProtocolError) as e:
            await self._close(socket, gp.CLOSE_BAD, "no hello" if isinstance(e, asyncio.TimeoutError) else "bad hello")
            return
        except _Close as c:
            if c.code is not None:
                await self._close(socket, c.code, c.reason)
            return
        tasks = []
        try:
            await socket.send_bytes(gp.encode({
                "t": gp.WELCOME, "proto": gp.PROTO, "commit": self.commit, "idle_exit_s": self.s.idle_exit_s,
                "idle_unused_s": self._unused_s(conn), "rotate_after_s": gp.ROTATE_AFTER_S, "now": self.clock()}))
            conn.reader = asyncio.create_task(self._read(conn))
            tasks = [conn.reader, asyncio.create_task(conn.write()), asyncio.create_task(self._ping(conn))]
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except (OSError, RuntimeError, WebSocketDisconnect):
            pass  # the socket broke under us: as a drop
        finally:
            for t in tasks:
                e = t.exception() if t.done() and not t.cancelled() else None
                if e is not None and not isinstance(e, (OSError, RuntimeError, WebSocketDisconnect)):
                    _log(f"the worker connection failed ({type(e).__name__})")
                    conn.close_code = conn.close_code or 1011
                t.cancel()
            self._gone(conn)  # before any await
            if not asyncio.current_task().cancelling():  # a handler being cancelled must not wait for anything
                if tasks:  # (wait, not gather: gather would raise a task's own cancellation in place of ours)
                    await asyncio.wait(tasks)
                if conn.close_code is not None:
                    await self._close(socket, conn.close_code, "")

    @staticmethod
    async def _close(socket, code: int, reason: str):
        with suppress(Exception):
            await asyncio.wait_for(socket.close(code=code, reason=reason), 5)

    def _handshake_refusal(self, token: gp.Token | None) -> str | None:
        if not self.configured():
            return "no GPU feature here"
        if token is None:
            return "a bad or expired token, or one for another server"
        if token.role not in self.roles:
            return f"the {token.role} role is not allowed here"
        with self._lock:
            if token.wake in self._ended:
                return "a worker that already said goodbye"
        return None

    def _log_refused(self, reason: str):
        with self._lock:
            now = time.monotonic()
            self._refused[0] += 1
            if now - self._refused[1] < 60:
                return
            n, self._refused = self._refused[0], [0, now]
        _log(f"refused a worker connection ({reason})" + (f"; {n - 1} more in the last minute" if n > 1 else ""))

    def _hello(self, msg: dict, token: gp.Token) -> dict:
        if msg["type"] == "websocket.disconnect":
            raise _Close(None)
        if msg.get("bytes") is None:
            raise _Close(gp.CLOSE_BAD, "binary frames only")
        h, _ = gp.decode(msg["bytes"], expect=gp.FROM_WORKER)
        if h["t"] != gp.HELLO:
            raise _Close(gp.CLOSE_BAD, "hello first")
        if h["proto"] != gp.PROTO:
            _log(f"refused a worker speaking protocol {h['proto']} (this server speaks {gp.PROTO})")
            raise _Close(gp.CLOSE_VERSION, "another protocol version")
        if h["wake"] != token.wake or h["role"] != token.role:
            raise _Close(gp.CLOSE_BAD, "not what its token says")
        if h["commit"] != self.commit and gp.SHA_RE.fullmatch(h["commit"]) and gp.SHA_RE.fullmatch(self.commit):
            if token.role != "laptop":  # a launched worker must run exactly the server's code
                _log(f"refused a {token.role} worker on commit {h['commit'][:12]} (this server: {self.commit[:12]})")
                raise _Close(gp.CLOSE_VERSION, "another code version")
            _log(f"the laptop runs commit {h['commit'][:12]} (this server: {self.commit[:12]})")
        return h

    def _admit(self, conn: Conn):
        """One active worker. Raises _Close(CLOSE_BUSY) when this one may not serve now."""
        with self._lock:
            now = self.clock()
            w, a = self._wake, self._active
            if a is not None and (a.worker, a.wake, a.role) == (conn.worker, conn.wake, conn.role):
                self._replace(a, conn, now)
                return
            busy = None
            if w is not None and w.fresh and conn.wake != w.id and conn.role != "laptop":
                busy = "a newer GPU machine is starting"
            elif a is not None and not self._retiring(a, now):
                busy = "another worker is active"
            if busy:
                _log(f"turned a {conn.role} worker away ({busy})")
                raise _Close(gp.CLOSE_BUSY, busy)
            if a is not None:
                self._others.append(a)
            conn.version = self._versions.get(conn.wake)  # its run, when a wake here launched or found it
            last = self._last_worker
            if w is not None:
                conn.started = w.started
                conn.unused_s = self._unused_limit(w.reason, conn.role)
                if w.version is not None:
                    if conn.wake == w.id:
                        conn.version = w.version
                    else:  # the laptop came first: that run will be turned away, never adopt it again
                        self._retired.add(w.version)
            elif self._reconnect_until is not None and last is not None and last.worker == conn.worker:
                # the same worker, back: its idle time and its run go on
                for attr in ("used", "last_work", "ready_at", "unused_s", "version"):
                    setattr(conn, attr, getattr(last, attr))
                last = None
            if last is not None and last.version is not None:  # another worker serves now: never adopt that run
                self._retired.add(last.version)
            self._active, self._wake, self._reconnect_until, self._unavailable = conn, None, None, None
            self._last_worker = None
            _log(f"{conn.role} worker connected ({conn.where})")

    def _replace(self, old: Conn, new: Conn, now: float):
        """The same worker back on a fresh socket before its old one was seen dead (a network stall
        neither side heard end): the fresh socket takes over at once, with the old one's run, idle time
        and wake, and whatever the old one was doing goes back to the queue. (Under the lock.)"""
        old.closed = True
        for attr in ("used", "last_work", "ready_at", "unused_s", "version"):
            setattr(new, attr, getattr(old, attr))
        self._active = new
        if old.live is not None:
            self._close_link(old.live, "gone")
        lost = sorted((j for j in self._inflight.values() if j.conn is old), key=lambda j: j.created)
        for job in reversed(lost):
            self._lose(job, STOPPED_TWICE)
        if old.stopping is not None:  # it never heard that it should stop: say it again
            self._stop(new, old.stop_reason, now)
        old.abort(1001)
        _log(f"the {new.role} worker came back on a new connection")

    async def _read(self, conn: Conn):
        loop = asyncio.get_running_loop()
        try:
            while conn.bye is None:
                msg = await conn.socket.receive()
                if msg["type"] == "websocket.disconnect":
                    return
                if msg.get("bytes") is None:
                    raise gp.ProtocolError("a text frame")
                h, payload = gp.decode(msg["bytes"], expect=gp.FROM_WORKER)
                conn.heard = loop.time()
                self._on_message(conn, h, payload, loop)
            conn.close_code = gp.CLOSE_SHUTDOWN  # it said bye
        except gp.ProtocolError as e:
            _log(f"closed a worker connection: {gp.printable(e, 120)}")
            conn.close_code = gp.CLOSE_BAD

    async def _ping(self, conn: Conn):
        """A ping every HEARTBEAT_S; whether anything came is looked at every second, so a dead socket is
        let go about as soon as the worker gives up on it (and the worker's next socket is not turned away)."""
        loop = asyncio.get_running_loop()
        n, ping_at = 0, loop.time() + gp.HEARTBEAT_S
        while loop.time() - conn.heard <= gp.DEAD_AFTER_S:
            await asyncio.sleep(min(1.0, gp.HEARTBEAT_S / 4))
            if loop.time() < ping_at:
                continue
            n += 1
            ping_at = loop.time() + gp.HEARTBEAT_S
            conn._put("control", gp.encode({"t": gp.PING, "n": n, "ts": round(loop.time(), 4)}))
        _log(f"nothing from the worker for {gp.DEAD_AFTER_S:.0f} s; dropping its connection")
        conn.close_code = 1011

    def _on_message(self, conn: Conn, h: dict, payload: bytes, loop):
        t = h["t"]
        if t == gp.PONG:
            conn.rtts.append(max(0.0, (loop.time() - h["ts"]) * 1000))
            if len(conn.rtts) >= 6 and not conn.rtt_logged:
                conn.rtt_logged = True
                _log(f"round trip to the worker: p90 {self._rtt_p90(conn):.0f} ms")
        elif t == gp.HEARTBEAT:
            if h["window_ms"] is not None:
                conn.window_ms = h["window_ms"]
        elif t == gp.CAPS:
            self._on_caps(conn, h)
        elif t == gp.PROGRESS:
            with self._lock:
                job = self._inflight.get(h["job"])
                if job is not None and job.conn is conn:
                    job.state, job.stage = "running", h["stage"]
        elif t == gp.RESULT:
            self._on_result(conn, h, payload)
        elif t == gp.ERROR:
            self._on_error(conn, h)
        elif t == gp.BYE:
            conn.bye = h["reason"]
            _log(f"the worker said goodbye ({h['reason']})")
        elif t in (gp.LIVE_READY, gp.LIVE_PROBS, gp.LIVE_ERROR):
            self._on_live(conn, h, payload)
        elif t == gp.HELLO:
            raise gp.ProtocolError("a second hello")

    def _on_caps(self, conn: Conn, h: dict):
        with self._lock:
            now = self.clock()
            conn.caps = h
            if h["window_ms"] is not None:
                conn.window_ms = h["window_ms"]
            if h["error"]:
                if conn.stopping is None:
                    _log(f"the worker could not load its models: {gp.printable(h['error'], 160)}")
                    self._stop(conn, "error", now)
                    self._give_up(LOAD_FAILED, None, now, retry_after=3600)
                    if "no gpu" in h["error"].lower():
                        self._quota_at = self._quota_tried = -math.inf  # Kaggle may have run out: read it again
                return
            if (h["notes"] or h["live"]) and conn.ready_at is None:
                conn.ready_at = now
                self._failures = 0
                _log(f"ready after {now - conn.started:.0f} s ({conn.where}"
                     + (f", Kaggle v{conn.version}" if conn.version is not None else "") + ")")
            if conn is self._active and self._no_notes(conn, now):
                self._fail_queued(NO_NOTES)  # never leave a recording waiting for what will not come
        self._dispatch()

    def _on_result(self, conn: Conn, h: dict, payload: bytes):
        with self._lock:
            job = self._inflight.get(h["job"])
            if job is None or job.conn is not conn:
                return  # unknown, cancelled or already done: ignored
        try:
            if h["frames"] != gp.frames_for(job.samples):
                raise gp.ProtocolError("the wrong number of frames")
            probs = gp.probs_from(payload, h["frames"])
            words = gp.check_words(h["words"], job.samples / gp.SR)
        except gp.ProtocolError as e:
            _log(f"{job.id}: a broken result ({gp.printable(e, 80)})")
            self._retry(job, conn, CANNOT_CLEAN)
            return
        with self._lock:
            if self._inflight.get(job.id) is not job:
                return
            now = self.clock()
            del self._inflight[job.id]
            job.result, job.state, job.x = (probs, words, bool(h["slow"])), "done", None
            conn.used, conn.last_work = True, now
            _log(f"{job.id}: {job.samples / gp.SR:.1f} s of audio in {now - job.sent_at:.1f} s"
                 + (" (words on the CPU)" if h["slow"] else ""))
        job.done.set()
        self._dispatch()

    def _on_error(self, conn: Conn, h: dict):
        with self._lock:
            job = self._inflight.get(h["job"])
            if job is None or job.conn is not conn:
                return
        _log(f"{job.id}: the worker could not clean it ({gp.printable(h['message'], 120)})")
        if h["retry"]:
            self._retry(job, conn, CANNOT_CLEAN)
            return
        with self._lock:
            if self._inflight.get(job.id) is job:
                self._fail(job, CANNOT_CLEAN)
        self._dispatch()

    def _on_live(self, conn: Conn, h: dict, payload: bytes):
        link = conn.live
        if link is None or link.closed or h["stream"] != link.stream:
            return  # a stream that already ended
        if h["t"] == gp.LIVE_READY:  # its model is warm: the session may start
            link.ready_at = self.clock()
            link.ready.set()
        elif h["t"] == gp.LIVE_PROBS:
            rows = gp.probs_from(payload, h["f1"] - h["f0"])
            link.replies.put(("probs", h["seq"], h["f0"], h["f1"], rows, h["ms"]))
        else:
            _log(f"the worker ended a live stream: {gp.printable(h['message'], 120)}")
            self._close_link(link, "error")

    def _gone(self, conn: Conn):
        """Its socket closed: a goodbye (or one after being told to stop) ends it; anything else may be
        a drop it comes back from. Its unfinished recordings go back to the queue."""
        with self._lock:
            now = self.clock()
            conn.closed = True
            for lane in (conn.control, conn.live_q, conn.bulk):  # (a recording's parts hold its audio)
                lane.clear()
            if conn.live is not None:
                self._close_link(conn.live, "gone")
            lost = sorted((j for j in self._inflight.values() if j.conn is conn), key=lambda j: j.created)
            normal = conn.bye is not None or conn.stopping is not None
            if normal:
                self._ended.append(conn.wake)
                if conn.version is not None:
                    self._retired.add(conn.version)
                if conn.role == "laptop" and conn.bye == "idle":
                    self._standby_at = now  # it stands by again at once (its first poll comes within seconds)
            if conn is self._active:
                self._active = None
                if normal:
                    _log("the worker left")
                else:
                    self._reconnect_until = now + self.s.reconnect_grace_s
                    self._reconnect_check = now + min(self.s.reconnect_look_s, self.s.reconnect_grace_s)
                    self._reconnect_checked = False
                    self._last_worker = conn
                    _log(f"the worker dropped; waiting {self.s.reconnect_grace_s:.0f} s for it to come back")
            elif conn in self._others:
                self._others.remove(conn)
            for job in reversed(lost):
                self._lose(job, STOPPED_TWICE)
        self._dispatch()
        if normal:
            self._kick()

    # ---- the job queue --------------------------------------------------------------------------

    @staticmethod
    def _job_timeout(job: Job) -> float:
        return 120 + 1.5 * job.samples / gp.SR

    def _dispatch(self):
        """Send the next recording to the active worker if it can take one now."""
        with self._lock:
            now = self.clock()
            c = self._active
            if (c is None or not c.caps.get("notes") or c.closed or not self._queue or self._retiring(c, now)
                    or any(j.conn is c for j in self._inflight.values())):
                return
            job = self._queue[0]
            if c.deadline is not None and now + self._job_timeout(job) >= c.deadline - 60:
                return  # it could not finish before its machine's time limit
            self._queue.popleft()
            job.state, job.conn, job.sent_at, job.stage = "sent", c, now, None
            job.sends += 1
            job.seq += 1
            self._inflight[job.id] = job
            c.last_work = now
            frames = self._job_frames(job, job.x, job.seq)
        if not c.send("bulk", frames):
            with self._lock:
                if self._inflight.get(job.id) is job:
                    self._lose(job, STOPPED_TWICE)

    def _job_frames(self, job: Job, x: np.ndarray, seq: int):
        """A job's header and its parts, encoded one at a time as the connection sends them."""
        if job.seq != seq:  # taken back before its turn to go out: not even the header
            return
        yield gp.encode({"t": gp.JOB, "job": job.id, "kind": "note", "samples": job.samples,
                         "parts": gp.n_parts(job.samples), "enc": "pcm16", "sr": gp.SR, "language": "en"})
        for i, part in enumerate(gp.job_parts(x)):
            if job.seq != seq:  # taken back meanwhile (a timeout): the rest is not wanted
                return
            yield gp.encode({"t": gp.JOB_AUDIO, "job": job.id, "i": i}, part)

    def _lose(self, job: Job, text: str):
        """A sent job came back unfinished (under the lock): to the head of the queue, once."""
        self._inflight.pop(job.id, None)
        if job.sends >= 2:
            self._fail(job, text)
            return
        job.state, job.conn, job.stage = "queued", None, None
        job.seq += 1
        self._queue.appendleft(job)

    def _retry(self, job: Job, conn: Conn, text: str):
        with self._lock:
            if self._inflight.get(job.id) is not job or job.conn is not conn:
                return
            self._lose(job, text)
        self._dispatch()

    def _fail(self, job: Job, text: str):
        """Under the lock. A recording that never reached a GPU does not count against its visitor."""
        self._inflight.pop(job.id, None)
        with suppress(ValueError):
            self._queue.remove(job)
        if job.sends == 0:
            self._refund(job.client, job.charge)
        job.state, job.error, job.x, job.charge = "failed", text, None, None
        job.done.set()

    def _refund(self, client: str, charge: list | None):
        if charge is not None:
            with suppress(ValueError):
                self._client_audio.get(client, deque()).remove(charge)

    def _fail_queued(self, text: str):
        """Every recording still waiting for a worker gets this sentence (under the lock)."""
        for job in list(self._queue):
            self._fail(job, text)

    def _withdraw(self, job: Job, text: str):
        with self._lock:
            if job.state == "queued":
                self._fail(job, text)

    def _client_busy(self, client: str, now: float) -> bool:
        if self._recent(self._reserved.get(client, deque()), now, RESERVE_S, key=lambda e: e[0]):
            return True
        return any(j.client == client for j in (*self._queue, *self._inflight.values()))

    def _take_reservation(self, client: str) -> list | None:
        """The visitor's admitted recording, now queued: its turn is no longer held; returns its charge."""
        r = self._reserved.get(client)
        return r.popleft() if r else None

    def _worker_ready(self) -> bool:
        c = self._active
        return c is not None and bool(c.caps.get("notes")) and c.stopping is None

    # ---- waking ---------------------------------------------------------------------------------

    def _autostart(self):
        if self.autostart and not self._started:
            self.start()

    def _spawn(self, fn, *args):
        """Run a launcher call (network) off the lock: in a thread, or inline without autostart (tests)."""
        def run():
            try:
                fn(*args)
            except Exception as e:  # logged; the tick's timeouts still end anything left hanging
                _log(f"{fn.__name__} failed ({type(e).__name__})")
        if self.autostart:
            threading.Thread(target=run, daemon=True).start()
        else:
            run()

    def _retiring(self, c: Conn, now: float) -> bool:
        """Takes no new work: told to stop, near its machine's time limit, up for long enough (the next
        work wakes a fresh one, which counts against the limits), or past our GPU budget."""
        return c.stopping is not None or self._draining(c, now) or self._aged(c, now) or self._budget_over(c)

    def _aged(self, c: Conn, now: float) -> bool:
        return c.ready_at is not None and now - c.ready_at > self.s.max_wake_s

    def _unused_limit(self, reason: str, role: str) -> float:
        """How soon a worker woken for `reason` sleeps if no work comes at all."""
        if reason in RECORDING:  # Record was pressed, or a file chosen: up to three minutes and an upload away
            return min(self.s.idle_exit_s, max(self.s.idle_unused_s, gp.MAX_JOB_S + 120))
        if reason == "button" and role == "laptop":  # it comes back within seconds when it is wanted
            return min(self.s.idle_unused_s, self.s.button_unused_s)
        return self.s.idle_unused_s

    def _unused_s(self, c: Conn) -> float:
        return c.unused_s if c.unused_s is not None else self.s.idle_unused_s

    def _idle_limit(self, c: Conn) -> float:
        return self.s.idle_exit_s if c.used else self._unused_s(c)

    def _needs_worker(self, now: float, job: bool) -> bool:
        """No worker is awake or coming back (one that takes no new work counts as gone for queued work)."""
        c = self._active
        if c is not None:
            return job and self._retiring(c, now)
        return self._reconnect_until is None or now >= self._reconnect_until

    def _no_notes(self, c: Conn | None, now: float) -> bool:
        """An awake worker that has loaded all it will and has no recordings engine (a live-only one)."""
        return (c is not None and c.ready_at is not None and not c.caps.get("notes") and not c.caps.get("loading")
                and not self._retiring(c, now))

    def _accepting(self, now: float, client: str) -> str | None:
        if self._no_notes(self._active, now):
            return NO_NOTES
        if self._wake is not None or not self._needs_worker(now, True) or self._laptop_standing_by(now):
            return None
        refusal = self._refusal(now, client)
        return refusal[1] if refusal else None

    def _laptop_standing_by(self, now: float) -> bool:
        """The team's laptop can be woken in seconds: it polled lately, or it is the worker on its way
        back to standing by (told to stop, or up for long enough)."""
        if "laptop" not in self.roles:
            return False
        c = self._active
        if c is not None and c.role == "laptop" and self._retiring(c, now):
            return True
        return self._standby_at is not None and now - self._standby_at <= self.s.laptop_seen_s

    def _refusal(self, now: float, client: str, per_client: bool = True) -> tuple | None:
        """Why a launch may not happen now, as (HTTP status, sentence), or None (under the lock).
        per_client=False: for recordings already admitted, only what holds for everyone."""
        if self._resting_until is not None and now < self._resting_until:
            return 409, self._resting_text(now)
        if self._unavailable is not None and (self._unavailable[1] is None or now < self._unavailable[1]):
            reason, retry_at = self._unavailable
            return 409, reason + (f" Try again in {_minutes(retry_at - now)} min." if retry_at else "")
        if self.launcher is None:
            return 409, NOT_CONNECTED
        if self.public_url is None:
            return 409, BROKEN
        cap = self._cap_refusal(now, client, per_client)
        if cap is not None:
            return 429, cap
        if self._over(self._cached_quota(now), now):
            return 409, self._resting_text(now)
        return None

    def _cap_refusal(self, now: float, client: str, per_client: bool = True) -> str | None:
        launches = self._recent(self._launches, now, 2 * 86400)
        if sum(now - t < 3600 for t in launches) >= self.s.wakes_per_hour:
            return WAKES_HOUR
        day = int(now // 86400)  # the UTC day
        if sum(int(t // 86400) == day for t in launches) >= self.s.wakes_per_day:
            return WAKES_DAY.format(m=_minutes((day + 1) * 86400 - now))
        mine = self._recent(self._client_launches.get(client, deque()), now, 86400)
        if per_client and len(mine) >= self.s.client_wakes_per_day:
            return WAKES_DAY.format(m=_minutes(mine[0] + 86400 - now))
        return None

    def _laptop_cap(self, now: float, client: str) -> str | None:
        """Wakes of the standby laptop that bring no work yet (a Wake press, Record, a file chosen)."""
        wakes = self._recent(self._laptop_wakes, now, 86400)
        if sum(now - t < 3600 for t in wakes) >= self.s.laptop_wakes_per_hour:
            return WAKES_HOUR
        if len(wakes) >= self.s.laptop_wakes_per_day:
            return WAKES_DAY.format(m=_minutes(wakes[0] + 86400 - now))
        if len(self._recent(self._client_laptop_wakes.get(client, deque()), now, 3600)) >= \
                self.s.client_laptop_wakes_per_hour:
            return WAKES_HOUR
        return None

    def _begin_launch(self, now: float, reason: str, client: str) -> tuple:
        """A launch of a new worker (under the lock); returns the call for _spawn."""
        for c in (self._active, *self._others):  # a run on its way out is never what this wake finds
            if c is not None and c.version is not None and self._retiring(c, now):
                self._retired.add(c.version)
        wake = Wake(gp.new_wake(), now, reason, client, poll_every=self.s.launch_poll_s)
        wake.token = gp.make_token(self.secret, self.launcher.role, wake.id, int(now + self.launcher.token_ttl_s),
                                   self.public_url)
        self._wake = wake
        _log(f"waking a GPU ({reason})")
        return self._launch, wake.id

    def _launch(self, wake_id: str):
        """In its own thread: Kaggle's quota first, then the launcher starts a worker (or finds one)."""
        self._refresh_quota()
        with self._lock:
            w = self._wake
            if w is None or w.id != wake_id:
                return
            now = self.clock()
            if self._over(self._cached_quota(now), now):
                self._wake = None
                self._fail_queued(self._resting_text(now))
                return
            token, reason, retired = w.token, w.reason, frozenset(self._retired)
        try:
            with self._launcher_lock:
                launch = self.launcher.launch(wake_id, token, reason, retired)
        except Exception as e:  # a bug or a surprise from the API: this wake fails, with a retry soon
            _log(f"the launcher failed ({type(e).__name__})")  # (its text might hold a URL, or worse)
            launch = gp.Launch("failed", START_FAILED, retry_s=60)
        self._launched(wake_id, launch)

    def _launched(self, wake_id: str, launch: gp.Launch):
        with self._lock:
            now = self.clock()
            w = self._wake
            started = launch.state in ("queued", "running")
            if started and not launch.adopted:  # only a run started here counts against the caps
                self._launches.append(now)
                if w is not None and w.id == wake_id:
                    self._client_launches.setdefault(w.client, deque()).append(now)
            if started and launch.version is not None:
                self._remember(wake_id, launch.version)
            if w is None or w.id != wake_id:  # a worker came first (the laptop), or the wake ended
                if self._active is not None and self._active.wake == wake_id:
                    self._active.version = launch.version
                elif launch.version is not None:
                    self._retired.add(launch.version)
                return
            if started:
                w.step = "queued" if launch.state == "queued" else "starting"
                w.version, w.adopted, w.fresh, w.launched = launch.version, launch.adopted, not launch.adopted, True
                w.next_poll = now + w.poll_every
                _log(f"{'found' if launch.adopted else 'started'} {launch.ref or 'a worker'} ({launch.state})")
            elif launch.state == "quota":
                self._rest(launch.until, "week", now)
                self._wake = None
                self._fail_queued(self._resting_text(now))
            else:
                self._give_up(launch.reason if launch.state == "failed" and launch.reason else STOPPED_EARLY,
                              launch.retry_s if launch.state == "failed" else 60, now)

    def _poll(self, wake_id: str):
        """In its own thread: how the run of a wake in progress is doing."""
        launch, again = None, False
        try:
            with self._launcher_lock:
                launch = self.launcher.poll()
        except Exception as e:  # a network error, or Kaggle asking us to slow down: poll less often
            _log(f"could not ask how the GPU machine is doing ({type(e).__name__})")
        with self._lock:
            self._polling = False
            now = self.clock()
            w = self._wake
            if w is None or w.id != wake_id:
                return
            if launch is None:
                w.poll_every = min(60.0, w.poll_every * 2)
                w.next_poll = now + w.poll_every
                return
            w.poll_every = self.s.launch_poll_s
            w.next_poll = now + w.poll_every
            if launch.state in ("queued", "running"):
                w.step = "queued" if launch.state == "queued" else "starting"
                if launch.version is not None and launch.version != w.version:  # the run it follows now
                    w.version = launch.version
                    self._remember(w.id, launch.version)
            elif launch.state == "quota":
                self._rest(launch.until, "week", now)
                self._wake = None
                self._fail_queued(self._resting_text(now))
            elif w.adopted and not w.repushed:  # the run it found was on its way out: start a fresh one, once
                if w.version is not None:
                    self._retired.add(w.version)
                w.step, w.version, w.adopted, w.repushed, w.launched = "asking", None, False, True, False
                _log("the run it found has ended; starting a fresh one")
                again = True
            else:
                self._give_up(launch.reason if launch.state == "failed" and launch.reason else STOPPED_EARLY,
                              launch.retry_s if launch.state == "failed" else 60, now)
        if again:
            self._spawn(self._launch, wake_id)

    def _remember(self, wake_id: str, version: int):
        """Which run a wake launched or found, so its worker is known by its run when it connects (or
        comes back): a run told to stop is then never adopted again."""
        self._versions[wake_id] = version
        while len(self._versions) > 100:
            self._versions.pop(next(iter(self._versions)))

    def _give_up(self, reason: str, retry_s: float | None, now: float, retry_after: float | None = None):
        """A wake failed (under the lock): unavailable for a while, and the recordings waiting are told."""
        if retry_after is None:
            retry_after = 3600.0 if retry_s is None else (60.0, 120.0, 300.0)[min(self._failures, 2)]
            self._failures += 1
        self._unavailable = (reason, now + retry_after)
        self._wake = None
        self._fail_queued(reason)
        _log(f"unavailable for {retry_after / 60:.0f} min: {reason}")

    def _kick(self):
        """Queued work and no worker coming for it: wake one (a refusal fails that work)."""
        with self._lock:
            if not self._queue or self._wake is not None or not self._needs_worker(self.clock(), True):
                return
            client = self._queue[0].client
        try:
            self.wake("job", client, job=True)
        except WakeRefused as e:
            with self._lock:
                self._fail_queued(str(e))

    # ---- the quota ------------------------------------------------------------------------------

    def _refresh_quota(self):
        """Kaggle's own count of our GPU hours this week (in a thread; never for a page request)."""
        with self._lock:
            if self._quota_busy:
                return
            self._quota_busy, self._quota_tried = True, self.clock()
        q = None
        try:
            with self._launcher_lock:
                q = self.launcher.quota()
        except Exception as e:  # unknown: launches go ahead within the launch caps
            _log(f"could not read the GPU quota ({type(e).__name__})")
        with self._lock:
            self._quota_busy = False
            if q is not None:
                now = self.clock()
                self._quota, self._quota_at = q, now
                day = int(now // 86400)
                if self._day_base is None or self._day_base[0] != day or q.used_h < self._day_base[1]:
                    self._day_base = (day, q.used_h)  # the day's first reading (or the week started again)

    def _cached_quota(self, now: float) -> gp.Quota | None:
        """The last reading while it is recent; an old one says nothing (the week may have reset)."""
        return self._quota if now - self._quota_at <= self.s.quota_cache_s else None

    def _cap_h(self, q: gp.Quota) -> float:
        return min(self.s.week_hours, q.total_h - self.s.reserve_hours)

    def _day_cap_h(self) -> float:
        return self.s.day_hours if self.s.day_hours is not None else self.s.week_hours / 4

    def _today_h(self, q: gp.Quota, now: float) -> float:
        """Kaggle's GPU hours used since its first reading this UTC day."""
        b = self._day_base
        return max(0.0, q.used_h - b[1]) if b is not None and b[0] == int(now // 86400) else 0.0

    def _spent(self, q: gp.Quota | None, now: float) -> tuple | None:
        """(until when, "week" | "day") if our share of Kaggle's GPU hours is used up, this week's or
        today's (so a few visitors cannot use up the week in one day), else None."""
        if q is None:
            return None
        if q.used_h + q.reserved_h >= self._cap_h(q):
            return q.refresh_at, "week"
        if self._today_h(q, now) >= self._day_cap_h():
            return (int(now // 86400) + 1) * 86400.0, "day"
        return None

    def _over(self, q: gp.Quota | None, now: float) -> bool:
        """The same; if so, the GPU rests from now on (under the lock)."""
        spent = self._spent(q, now)
        if spent is not None:
            self._rest(*spent, now)
        return spent is not None

    def _budget_over(self, c: Conn | None) -> bool:
        """A Kaggle worker past our share of the quota takes no new work (the laptop costs nothing)."""
        now = self.clock()
        return c is not None and c.role == "kaggle" and self._spent(self._cached_quota(now), now) is not None

    def _rest(self, until: float | None, why: str, now: float):
        self._resting_until = until if until is not None and until > now else now + 6 * 3600
        self._resting_why = why
        _log(f"the {'weekly' if why == 'week' else 'daily'} GPU hours are used up; resting until {self._until_iso()}")

    def _until_iso(self) -> str | None:
        if self._resting_until is None:
            return None
        return datetime.fromtimestamp(self._resting_until, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _resting_text(self, now: float) -> str:
        until = self._resting_until if self._resting_until and self._resting_until > now else (
            self._quota.refresh_at if self._quota and self._quota.refresh_at else now + 6 * 3600)
        when = datetime.fromtimestamp(until, timezone.utc).strftime("%a %d %b, %H:%M UTC")
        if self._resting_why == "day":
            return f"Today's share of the free GPU hours is used up, so the GPU rests until {when}."
        return f"This week's free GPU hours are used up, so the GPU rests until {when}."

    # ---- the clock ------------------------------------------------------------------------------

    def tick(self, now: float | None = None) -> None:
        """Every second: timeouts, idle sleep, polls and refreshes. Never raises."""
        try:
            self._tick(self.clock() if now is None else now)
        except Exception as e:
            _log(f"tick failed ({type(e).__name__})")
            with self._lock:
                self._give_up(START_FAILED, 120, self.clock())

    def _tick(self, now: float):
        todo, aborts = [], []
        with self._lock:
            w, c = self._wake, self._active
            if w is not None and w.step == "laptop" and now >= w.laptop_until:
                self._wake = None  # the laptop did not come: launch instead, if allowed
                self._standby_at = None  # and it no longer counts as standing by, until it asks again
                refusal = self._refusal(now, w.client, per_client=not self._queue)
                if refusal is None:
                    todo.append(self._begin_launch(now, w.reason, w.client))
                elif self.launcher is None:
                    self._give_up(LAPTOP_SILENT, None, now, retry_after=300)
                else:
                    self._fail_queued(refusal[1])
            elif w is not None and now - w.started > self.s.wake_timeout_s:
                minutes = _minutes(self.s.wake_timeout_s)
                self._give_up(f"Kaggle had no free GPU for {minutes} minutes."
                              if self.launcher is not None and self.launcher.name == "kaggle"
                              else f"The GPU did not start within {minutes} minutes.", None, now,
                              retry_after=300)  # a worker that turns up late is still welcome
            elif w is not None and w.launched and now >= w.next_poll and not self._polling:
                self._polling = True
                todo.append((self._poll, w.id))
            if c is not None:
                if c.ready_at is None and c.stopping is None and now - c.hello_at > self.s.load_timeout_s:
                    self._stop(c, "error", now)
                    self._give_up(LOAD_TIMEOUT, None, now, retry_after=300)
                self._check_live(c, now)
                if self._should_sleep(c, now):
                    spent = self._spent(self._cached_quota(now), now) if c.role == "kaggle" else None
                    self._stop(c, "budget" if spent else "idle", now)
                    if spent:
                        self._rest(*spent, now)
            for conn in [x for x in (c, *self._others) if x is not None]:
                self._check_jobs(conn, now)
                if conn.stopping is not None and now - conn.stopping > self.s.stop_grace_s:
                    aborts.append(conn)
            if self._reconnect_until is not None:
                last = self._last_worker
                machine = self.launcher is not None and last is not None and last.role == self.launcher.role
                if now >= self._reconnect_until and (self._reconnect_checked or not machine):
                    self._reconnect_until = None
                    self._crashed(now)
                elif machine and now >= self._reconnect_check and not self._polling:
                    self._reconnect_check = now + 30  # past the grace: a look at its machine every 30 s
                    self._polling = True
                    todo.append((self._check_run,))
            if (self.launcher is not None and not self._quota_busy and now - self._quota_tried > 60
                    and now - self._quota_at > self.s.quota_cache_s
                    and (self._wake is not None or (c is not None and c.role == self.launcher.role))):
                todo.append((self._refresh_quota,))  # keep the reading fresh while the GPU is up
            self._prune(now)
            state, step = self._state(now)
            if (state, step) != self._seen[:2]:
                self._seen = (state, step, now)
        for conn in aborts:
            conn.abort(gp.CLOSE_SHUTDOWN)
        for call in todo:
            self._spawn(*call)
        self._dispatch()
        self._kick()

    def _check_jobs(self, c: Conn, now: float):
        """A recording a worker has had for too long is taken back (it counts as one try)."""
        for job in [j for j in self._inflight.values() if j.conn is c]:
            if now - job.sent_at > self._job_timeout(job):
                _log(f"{job.id}: no result after {now - job.sent_at:.0f} s; taking it back")
                c.send("control", gp.encode({"t": gp.CANCEL, "job": job.id}))
                self._lose(job, CANNOT_CLEAN)

    def _check_live(self, c: Conn, now: float):
        """The broker owns a live link's life: one never started, or gone silent, is closed."""
        link = c.live
        if link is None:
            return
        since = link.last_audio or link.ready_at  # its last audio, or when its model was warm
        if not link.started and now - link.opened > self.s.live_start_s:
            self._close_link(link, "gone")
        elif since is None and link.started and now - link.opened > gp.LIVE_READY_S + self.s.live_start_s:
            self._close_link(link, "gone")  # never ready (its session went without it long ago)
        elif since is not None and now - since > self.s.live_silent_s:
            self._close_link(link, "gone")

    def _should_sleep(self, c: Conn, now: float) -> bool:
        if c.stopping is not None or c.ready_at is None or c.live is not None \
                or any(j.conn is c for j in self._inflight.values()):
            return False
        if self._budget_over(c) or self._aged(c, now):
            return True  # its work in flight is done; queued work goes to a fresh worker, or is told why
        if self._queue:
            return False
        idle_since = max(c.ready_at, c.last_work or 0.0)
        return now - idle_since >= self._idle_limit(c)

    def _stop(self, c: Conn, reason: str, now: float):
        """Tell a worker to leave (under the lock); its run is never adopted again."""
        if c.stopping is not None:
            return
        c.stopping, c.stop_reason = now, reason
        if c.version is not None:
            self._retired.add(c.version)
        idle = now - max(c.ready_at or c.hello_at, c.last_work or 0.0)
        _log(f"telling the worker to stop ({reason}" + (f", idle {idle:.0f} s)" if reason == "idle" else ")"))
        c.send("control", gp.encode({"t": gp.SHUTDOWN, "reason": reason}))

    def _check_run(self):
        """In its own thread: a worker dropped; is its machine still up? Seen gone (a crash it did not
        survive), the wait ends at once; still running past the grace, it gets five more minutes."""
        launch = None
        try:
            with self._launcher_lock:
                launch = self.launcher.poll()
        except Exception as e:
            _log(f"could not ask how the GPU machine is doing ({type(e).__name__})")
        with self._lock:
            self._polling = False
            if self._reconnect_until is None:
                return  # it came back meanwhile
            now = self.clock()
            if launch is None or launch.state in ("queued", "running"):
                if now < self._reconnect_until and not self._reconnect_checked:
                    self._reconnect_check = self._reconnect_until  # the first look: again when the grace ends
                    return
                if launch is not None and not self._reconnect_checked:
                    self._reconnect_until = now + 300  # its machine still runs: five more minutes at most
                    _log("its machine is still running; waiting up to five more minutes")
                self._reconnect_checked = True  # (no answer: the deadline decides)
                return
            # (the launcher's own sentence: an exit code, or Kaggle's failure text already cleaned of anything secret)
            _log(f"its machine has stopped; not waiting for it" + (f" ({launch.reason})" if launch.reason else ""))
            self._reconnect_until = None
            self._crashed(now)
        self._kick()

    def _crashed(self, now: float):
        """A worker dropped and did not come back (under the lock). The team's laptop is not counted as
        failing: a laptop that lost its network stands by again by itself."""
        last, self._last_worker = self._last_worker, None
        if last is not None and last.role == "laptop":
            _log("the team's laptop did not come back")
            return
        self._crashes.append(now)
        if sum(now - t < 1800 for t in self._crashes) >= 2:
            self._give_up(KEEPS_FAILING, None, now, retry_after=1800)
        else:
            _log("the worker did not come back")

    def _prune(self, now: float):
        """Forget per-visitor counts nobody needs any more (once a minute)."""
        if now - self._pruned_at < 60:
            return
        self._pruned_at = now
        first = lambda e: e[0]  # noqa: E731
        for d, window, key in ((self._client_launches, 86400, None), (self._client_laptop_wakes, 3600, None),
                               (self._client_live, 3600, None), (self._client_audio, 86400, first),
                               (self._reserved, RESERVE_S, first)):
            for k in [k for k, v in d.items() if not self._recent(v, now, window, key or (lambda t: t))]:
                del d[k]

    @staticmethod
    def _recent(d: deque, now: float, window: float, key=lambda t: t) -> deque:
        while d and now - key(d[0]) >= window:
            d.popleft()
        return d

    # ---- live -----------------------------------------------------------------------------------

    def _draining(self, c: Conn, now: float) -> bool:
        return c.deadline is not None and now > c.deadline - gp.DRAIN_S

    @staticmethod
    def _rtt_p90(c: Conn) -> float:
        r = sorted(c.rtts)
        return r[min(len(r) - 1, int(0.9 * len(r)))] if r else 250.0  # before any ping: a typical guess

    def _extra_delay(self, c: Conn) -> int | None:
        window = c.window_ms if c.window_ms is not None else gp.LOCAL_WINDOW_MS
        limit = gp.LIVE_MAX_EXTRA_HOSTED_MS if self.s.hosted else gp.LIVE_MAX_EXTRA_MS
        return gp.live_extra_delay_ms(self._rtt_p90(c), window, limit)

    def _live_refusal(self, now: float) -> str | None:
        """Why a live stream cannot open now, or None (under the lock)."""
        if not self.live_offered():
            return "Best (GPU) live is not offered on this server."
        c = self._active
        if c is None and self._reconnect_until is not None and now < self._reconnect_until:
            return "The GPU is reconnecting."
        if c is None or not c.caps.get("live"):
            return "The GPU is still waking." if self._wake is not None or c is not None else "The GPU is asleep."
        if self._retiring(c, now):
            return "The GPU is about to stop."
        if c.live is not None:
            return "Someone else is using Best (GPU) right now."
        if self._extra_delay(c) is None:
            return "The GPU is too far away for live use right now."
        return None

    def _close_link(self, link: LiveLink, reason: str | None):
        """End a live stream (idempotent). reason: why, for the detector, when it did not ask."""
        with self._lock:
            if link.closed:
                return
            link.closed = True
            link.ready.set()  # (a session still waiting for it goes on without it at once)
            c = link.conn
            if c.live is link:
                c.live = None
                c.last_work = self.clock()
            if link.started:
                c.send("control", gp.encode({"t": gp.LIVE_CLOSE, "stream": link.stream}))
        if reason is not None:
            link.replies.put(("closed", reason))
        self._dispatch()

    # ---- what the page sees ---------------------------------------------------------------------

    def _state(self, now: float) -> tuple[str, str | None]:
        c = self._active
        if c is not None:
            if c.caps.get("notes") or c.caps.get("live"):
                return ("busy" if c.live is not None or self._inflight else "ready"), None
            return "connecting", "loading"
        if self._reconnect_until is not None and now < self._reconnect_until:
            return "connecting", "reconnecting"
        if self._wake is not None:
            return "waking", self._wake.step
        if self._laptop_standing_by(now):  # it serves whatever Kaggle's state: free, and seconds away
            return "off", None
        if self._resting_until is not None and now < self._resting_until:
            return "resting", self._resting_why
        if self._unavailable is not None and (self._unavailable[1] is None or now < self._unavailable[1]):
            return "unavailable", None
        if self.launcher is None and not self._laptop_standing_by(now):
            return "unavailable", None
        return "off", None

    def _status(self, now: float, client: str = "") -> dict:
        state, step = self._state(now)
        if (state, step) != self._seen[:2]:
            self._seen = (state, step, now)
        c, w = self._active, self._wake
        elapsed = None
        if w is not None:
            elapsed = now - w.started
        elif state == "connecting":
            elapsed = now - (c.started if c is not None else self._seen[2])
        laptop = self._laptop_standing_by(now)
        if c is not None:
            where = c.where
        elif (w is not None and w.step == "laptop") or (w is None and laptop):
            where = LAPTOP
        elif state == "connecting" and self._last_worker is not None:
            where = self._last_worker.where
        elif self.launcher is not None:
            where = self.launcher.where
        else:
            where = None
        reason = retry_s = None
        if state == "resting":
            reason = self._resting_text(now)
        elif state == "unavailable":
            if self._unavailable is not None and (self._unavailable[1] is None or now < self._unavailable[1]):
                reason = self._unavailable[0]
                retry_s = round(self._unavailable[1] - now) if self._unavailable[1] is not None else None
            else:
                reason = NOT_CONNECTED
        sleeps = None
        if state == "ready" and c.ready_at is not None and c.stopping is None:
            sleeps = max(0, round(max(c.ready_at, c.last_work or 0.0) + self._idle_limit(c) - now))
        # whether a Wake press would start something now
        can_wake = state == "off" and (self._laptop_cap(now, client) is None if laptop
                                       else self._refusal(now, client) is None)
        live_reason = self._live_refusal(now)
        delay = None
        if live_reason is None:
            from ..live import delay_overrides
            extra = self._extra_delay(c)
            delay = {m: delay_overrides("teacher", m, extra)["target_delay_ms"] for m in gp.MODES}
        q = self._quota
        return {
            "state": state, "step": step, "now": round(now, 1), "since": round(self._seen[2], 1),
            "elapsed_s": round(elapsed) if elapsed is not None else None, "typical_s": list(self.s.typical_s),
            "where": where, "ahead": len(self._queue) + len(self._inflight),
            "notes": bool(c is not None and c.caps.get("notes")), "standby": laptop,
            "live": {"available": live_reason is None, "delay_ms": delay,
                     "reason": live_reason if self.live_offered() else None},
            "sleeps_in_s": sleeps, "until": self._until_iso() if state == "resting" else None, "reason": reason,
            "retry_s": retry_s, "can_wake": can_wake,
            "week": {"used_h": round(q.used_h + q.reserved_h, 1), "cap_h": round(self._cap_h(q), 1)} if q else None,
        }


broker = Broker.from_env()
