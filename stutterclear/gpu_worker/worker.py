"""The GPU worker: it connects out to the app server's /ws/gpu (stutterclear/app/gpu.py), says who it
is, loads its engines, and does the GPU work the server sends (recordings, and in P2 live windows)
until the server says goodbye, it has had nothing to do for too long, or its machine's time is
nearly up. The wire is stutterclear/gpu_protocol.py.

On Kaggle (and from the local test launcher) it connects at once and leaves for good when done. On
the team's laptop it loads its models first (the live teacher only if the server uses a worker for
live) and stands by without a socket, asking the server every STANDBY_POLL_S whether the GPU is
wanted, and stands by again after each wake.

One asyncio loop serves the socket; GPU work never runs on it. Recordings run on one job thread, a
live stream on a thread of its own, and models load on loader threads; each hands its results back
with call_soon_threadsafe. Nothing is kept across connections: a lost connection drops the recording
and the stream (the server sends them again), and nothing a visitor said is ever written down.
"""
import asyncio
import http.client
import json
import os
import queue
import random
import secrets
import ssl
import subprocess
import threading
import time
import traceback
import urllib.parse
from collections import deque
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from .. import gpu_protocol as gp
from . import engines
from .engines import log

OK, FAILED, USAGE, REFUSED, VERSION, NO_MODELS = 0, 1, 2, 3, 4, 5  # exit codes
IDLE_MARGIN_S = 60.0  # the worker's own idle timer trails the server's (which says SHUTDOWN first)
BYE_WAIT_S = 5.0  # after BYE, the server has this long to close the socket
BUSY_AFTER_LOSS_S = 60.0  # "another worker is active" this soon after a lost connection: most likely the
# server still holds this worker's own old socket (it was lost without a close), so try again
POLL_TIMEOUT_S = 10.0  # one standby poll
MAX_BAD = 3  # connections in a row closed as malformed before the worker gives up
MAX_WINDOW = 30 * gp.SR  # the longest live window served (they are 4-5 s)
MIN_WINDOW = 4000  # shorter audio is decided as "no stutter", as TeacherDetector does
USER_AGENT = "stutterclear-gpu-worker/1"
ROOT = Path(__file__).resolve().parents[2]
BYE_FOR = {"idle": "idle", "budget": "shutdown", "error": "error"}  # SHUTDOWN reason -> BYE reason


class NoRedirects(connect):
    """websockets' connect(), except that a redirect is never followed: websockets would send the
    token's header on to wherever the redirect points."""

    def process_redirect(self, exc: Exception) -> Exception:
        return exc


@dataclass
class Options:
    server: str  # the app server's base URL, checked by gp.server_url
    role: str  # kaggle | laptop | local: what the token says
    token: Callable[[], str]  # a token for the next connection (the laptop mints a fresh one per wake)
    commit: str  # the code this worker runs
    where: str = ""  # shown on the page; from the role and the GPUs when empty
    idle_exit_s: float | None = None  # until the server's WELCOME says (gp.IDLE_EXIT_S when None)
    deadline: float | None = None  # epoch s: this machine's time limit (Kaggle)
    first: str = "notes"  # on one GPU, the engine loaded first
    notes: bool = True
    live: bool = True
    fake: bool = False  # torch-free stand-in engines (tests)
    live_device: str | None = None
    notes_device: str | None = None
    delay_ms: int = 0  # fault injection: wait before each live reply and result
    crash_after: int = 0  # fault injection: die instead of returning the Nth recording (or Nth live window)
    key_id: str | None = None  # the GPU secret's key id, for the refusal message (laptop)


def code_commit() -> str:
    """The code this worker runs: STUTTERCLEAR_COMMIT (set on Kaggle), else its checkout's HEAD."""
    sha = (os.environ.get("STUTTERCLEAR_COMMIT") or "").strip()
    if not sha:
        try:
            sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                 timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            sha = ""
    return gp.printable(sha, 64) or "dev"


def where_label(role: str, gpus: list[dict]) -> str:
    """Where a worker runs, as the page shows it: "Kaggle (2× Tesla T4)", "the team's laptop"."""
    if role != "kaggle":
        return {"laptop": "the team's laptop", "local": "this computer"}[role]
    names = [g["name"] for g in gpus]
    if not names:
        return "Kaggle"
    label = f"{len(names)}× {names[0]}" if len(names) > 1 and len(set(names)) == 1 else " + ".join(names)
    return gp.printable(f"Kaggle ({label})", gp.MAX_WHERE)


def _thread(fn: Callable) -> asyncio.Future:
    """fn() on a daemon thread, as an awaitable. Unlike asyncio.to_thread it never holds up the
    process's exit (a standby poll or a model load in progress when Ctrl+C comes)."""
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def settle(ok, value):
        if not future.done():
            (future.set_result if ok else future.set_exception)(value)

    def run():
        try:
            out, ok = fn(), True
        except BaseException as e:
            out, ok = e, False
        try:
            loop.call_soon_threadsafe(settle, ok, out)
        except RuntimeError:  # the loop is gone: nobody is waiting any more
            pass

    threading.Thread(target=run, daemon=True).start()
    return future


class Job:
    """One recording: received part by part, then cleaned on the job thread."""

    def __init__(self, h: dict, gen: int):
        self.id, self.samples, self.parts = h["job"], h["samples"], h["parts"]
        self.language = h["language"] if h["language"].isalpha() and len(h["language"]) <= 3 else "en"
        self.gen = gen  # the connection it came on: its result goes back on that one only
        self.audio = bytearray(2 * self.samples)
        self.received = 0  # bytes
        self.next = 0  # the part expected next
        self.running = False
        self.cancelled = False


class LiveStream:
    """One live stream (P2). Its LIVE_AUDIO messages queue here and one thread answers them; a backlog
    is answered with one window, the newest, whose rows cover every frame the backlog asked for."""

    def __init__(self, h: dict, gen: int, engine, delay_s: float, call: Callable):
        self.id, self.gen, self.engine, self.delay_s, self.call = h["stream"], gen, engine, delay_s, call
        self.keep = h["window"] + 4 * gp.FRAME
        self.q = queue.Queue()
        self.buf = np.zeros(0, np.float32)  # the last `keep` samples received
        self.total = 0  # samples received
        self.last_audio = time.monotonic()
        self.windows = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def stop(self):
        self.q.put(None)

    def _append(self, h: dict, pcm: bytes):
        x = gp.from_pcm16(pcm)
        if h["end"] - len(x) == self.total:
            self.buf = np.concatenate([self.buf, x])[-self.keep:]
        else:  # a gap (or a resend): start again from this audio
            self.buf = x[-self.keep:]
        self.total = h["end"]

    def _answer(self, batch: list) -> tuple[bytes, float]:
        """LIVE_PROBS for every frame the batch asked for, from the newest window."""
        h = batch[-1][0]
        held = self.total - len(self.buf)  # the first sample still held
        start = max(h["start"], -(-held // gp.FRAME) * gp.FRAME)  # on a frame boundary, within what is held
        audio = self.buf[start - held:] if start < self.total else self.buf[:0]
        w0 = start // gp.FRAME
        t = time.perf_counter()
        if len(audio) >= MIN_WINDOW:
            probs = np.clip(np.nan_to_num(self.engine.window_probs(audio, w0), nan=0.0), 0.0, 1.0)
        else:
            probs = np.zeros((0, gp.N_CLASSES), np.float32)
        ms = (time.perf_counter() - t) * 1000
        f0, f1 = min(b[0]["f0"] for b in batch), max(b[0]["f1"] for b in batch)
        rows = gp.window_rows(probs, w0, f0, f1)
        header = {"t": gp.LIVE_PROBS, "stream": self.id, "seq": h["seq"], "f0": f0, "f1": f1, "ms": round(ms, 1)}
        return gp.encode(header, gp.probs_bytes(rows)), ms

    def _run(self):
        try:
            self.engine.start()
            # Warm before saying ready: on a shared GPU a recording may have let the live teacher go to
            # make room, and loading it again must not eat into the session's first answers.
            self.engine.window_probs(np.zeros(MIN_WINDOW, np.float32))
            self.call("ready", self)
            while True:
                batch = [self.q.get()]
                while batch[-1] is not None:
                    try:
                        batch.append(self.q.get_nowait())
                    except queue.Empty:
                        break
                if batch[-1] is None:
                    return
                for h, pcm in batch:
                    self._append(h, pcm)
                frame, ms = self._answer(batch)
                self.windows += 1
                if self.delay_s:
                    time.sleep(self.delay_s)
                self.call("reply", self, frame, ms)
        except Exception as e:
            self.call("failed", self, e)


class Worker:
    def __init__(self, opts: Options):
        self.o = opts
        self.id = secrets.token_hex(4)  # this process, as HELLO names it
        self.loop: asyncio.AbstractEventLoop | None = None
        self.gpus: list[dict] = []
        self.where = opts.where
        self.notes = self.live = None  # the engines
        self.ready: set[str] = set()  # "notes", "live" once loaded
        self.loading: set[str] = set()
        self.loaded: asyncio.Future | None = None  # done when nothing is loading any more
        self.error: str | None = None  # the CAPS error: no GPU, or the models did not load
        self.window_ms: float | None = None  # the live teacher's warm p90
        self.recent_ms = deque(maxlen=50)  # recent live windows
        self.ready_at: float | None = None  # monotonic: the idle clock starts once something is ready
        self.last_work = 0.0
        self.used = False  # any work since ready (then the longer idle limit applies)
        self.jobs_done = self.windows_done = 0
        self.job: Job | None = None
        self.stream: LiveStream | None = None
        self.jobs = queue.Queue()
        self.token: str | None = None
        self.outage: str | None = None
        self.lost_at: float | None = None  # monotonic: when a connection the server had welcomed was lost
        # the current connection
        self.ws = None
        self.gen = 0
        self.out: asyncio.Queue | None = None
        self.welcomed = False
        self.end: str | None = None  # how it ends, when the worker (or a message) decided it
        self.closing_at: float | None = None
        self.connected_at = self.last_heard = 0.0
        self.idle_exit_s = self.idle_unused_s = self.rotate_after_s = None

    # ---- the whole run ------------------------------------------------------------------------

    async def run(self) -> int:
        """Serve until done; returns the exit code."""
        self.loop = asyncio.get_running_loop()
        loaded = self.loaded = self.loop.create_future()
        threading.Thread(target=self._job_thread, daemon=True).start()
        plan = engines.Plan(None, None)
        if not self.o.fake:
            try:
                self.gpus = await _thread(engines.gpus)
            except Exception as e:  # no torch, or a broken driver: the same as no GPU
                log(f"could not look for GPUs ({type(e).__name__})")
            plan = engines.plan_devices(len(self.gpus), self.gpus[0]["mem_gb"] if self.gpus else 0.0)
        self.where = self.where or where_label(self.o.role, self.gpus)
        log(f"worker {self.id} on {self.where}: " + (", ".join(f"{g['name']} ({g['mem_gb']} GB)" for g in self.gpus)
                                                    or ("stand-in engines" if self.o.fake else "no GPU")))
        laptop = self.o.role == "laptop"
        live = self.o.live
        if live and laptop and self.o.notes and plan is not None and await _thread(self._live_offered) is False:
            log("the server does not use a GPU worker for live cleaning: the live teacher stays unloaded")
            live = False
        if plan is None:
            self._failed_to_load(engines.NO_GPU)
        else:
            # live_active: on a shared GPU a recording never makes room by unloading a live teacher that is
            # serving a stream, or still loading (it uses the CPU instead)
            self.notes, self.live = engines.make(
                plan, notes=self.o.notes, live=live, fake=self.o.fake,
                live_active=lambda: self.stream is not None or "live" in self.loading,
                live_device=self.o.live_device, notes_device=self.o.notes_device)
            self._load()
        if laptop:
            await loaded
            if self.error:
                log(f"{self.error} The worker stops.")
                return NO_MODELS
        try:
            return await self._serve(laptop)
        except Exception:
            log("unexpected error:\n" + traceback.format_exc())
            return FAILED

    async def _serve(self, laptop: bool) -> int:
        failures = bad = 0
        standby = laptop
        while True:
            if standby:
                code = await self._standby()
                if code is not None:
                    return code
                self.token, standby, failures = self.o.token(), False, 0  # a fresh wake for this connection
                # A wake is a fresh start: the idle clock begins now, not when it last worked before it
                # stood by (or it would say goodbye at once), and the server counts its use per wake.
                self.last_work, self.used = time.monotonic(), False
            elif self.token is None:
                self.token = self.o.token()
            code = self._time_to_go()
            if code is not None:
                return code
            end = await self._session()
            if self.welcomed:
                failures = 0
                if end == "lost":
                    self.lost_at = time.monotonic()
            if end == "busy" and not laptop and self.lost_at is not None \
                    and time.monotonic() - self.lost_at < BUSY_AFTER_LOSS_S:
                log("the server still holds this worker's last connection; trying again")
                end = "lost"
            bad = bad + 1 if end == "bad" else 0
            if end == "refused":
                return REFUSED
            if end == "version":
                return VERSION
            if end == "fatal":
                return FAILED
            if end in ("done", "busy"):
                if end == "busy":
                    log("another GPU worker is serving this server" + ("; standing by" if laptop else "; leaving"))
                if self.error:
                    return NO_MODELS
                if not laptop:
                    return OK
                standby = True
                continue
            if bad >= MAX_BAD:
                log(f"the server closed {MAX_BAD} connections in a row as malformed; the worker stops")
                return FAILED
            if end == "rotate":
                continue
            if laptop and failures >= len(gp.RECONNECT_S):  # the quick retries are over: poll instead
                standby = True
                continue
            wait = gp.RECONNECT_S[min(failures, len(gp.RECONNECT_S) - 1)] * random.uniform(0.8, 1.2)
            failures += 1
            await asyncio.sleep(wait)

    def _time_to_go(self) -> int | None:
        """Between connections (Kaggle, local): leave once past the deadline's drain, or idle past the
        limit, so a worker whose server is gone for good does not burn GPU hours waiting for it."""
        if self.o.role == "laptop":
            return None
        if self.o.deadline is not None and time.time() >= self.o.deadline - gp.DRAIN_S:
            log("this machine's time is nearly up; the worker leaves")
        elif self.ready_at is not None and time.monotonic() - self.last_work >= self._idle_limit() + IDLE_MARGIN_S:
            log("no work and no server for too long; the worker leaves")
        else:
            return None
        return NO_MODELS if self.error else OK

    # ---- loading the engines ------------------------------------------------------------------

    def _load(self):
        """Load the engines on loader threads: side by side when they have GPUs of their own, else the
        --first one first. A CAPS goes out after each one (once connected)."""
        kinds = [k for k in (self.o.first, "notes" if self.o.first == "live" else "live") if getattr(self, k)]
        self.loading = set(kinds)
        if not kinds:
            self._failed_to_load(engines.NO_LOAD)
            return
        devices = [getattr(getattr(self, k), "device", None) for k in kinds]
        apart = len(kinds) == 2 and None not in devices and devices[0] != devices[1]
        for group in ([[k] for k in kinds] if apart else [kinds]):
            threading.Thread(target=self._load_thread, args=(group,), daemon=True).start()

    def _load_thread(self, kinds: list[str]):
        for kind in kinds:
            if self.error:  # the worker is leaving: nothing more to load
                return
            t = time.perf_counter()
            try:
                getattr(self, kind).load()
                err = None
            except Exception as e:
                log(f"loading the {kind} engine failed: {type(e).__name__}: {gp.printable(e, 160)}")
                err = engines.load_error(e)
            self._call(self._loaded, kind, err, time.perf_counter() - t)

    def _loaded(self, kind: str, err: str | None, seconds: float):
        self.loading.discard(kind)
        if self.error:  # already leaving: the recordings engine failed
            return
        if err is not None:
            if kind == "notes" or self.notes is None:  # recordings are what most features need
                return self._failed_to_load(err)
            self.live = None  # the recordings carry on without live
        else:
            self.ready.add(kind)
            if kind == "live":
                self.window_ms = self.live.window_ms
            log(f"{kind} engine ready in {seconds:.0f} s"
                + (f" (a live window takes {self.window_ms} ms)" if kind == "live" else ""))
            if self.ready_at is None:
                self.ready_at = self.last_work = time.monotonic()
        if self.welcomed:
            self._caps()
        if not self.loading and not self.loaded.done():
            self.loaded.set_result(None)

    def _failed_to_load(self, err: str):
        self.error = self.error or err
        self.loading.clear()
        if self.ready_at is None:
            self.ready_at = self.last_work = time.monotonic()  # the idle clock bounds how long it tries to say so
        if not self.loaded.done():
            self.loaded.set_result(None)
        if self.welcomed:
            self._caps()
            self._leave("error")

    # ---- standing by (the laptop) -------------------------------------------------------------

    async def _standby(self) -> int | None:
        """No socket: ask the server whether the GPU is wanted every STANDBY_POLL_S. None once it is
        (connect now), else an exit code."""
        log(f"standing by for {self.o.server}")
        while True:
            answer = await _thread(self._wanted)
            if answer == "wanted":
                self._outage(None)
                return None
            if answer == "refused":
                return REFUSED
            self._outage(None if answer == "idle" else answer)
            await asyncio.sleep(gp.STANDBY_POLL_S)

    def _get(self, path: str, headers: dict, limit: int) -> tuple[int, bytes]:
        """GET a path on the server: (status, body). http.client never follows a redirect, so a token
        in the headers goes nowhere else. Raises OSError or HTTPException when it cannot be asked."""
        u = urllib.parse.urlsplit(self.o.server)
        if u.scheme == "https":
            conn = http.client.HTTPSConnection(u.hostname, u.port, timeout=POLL_TIMEOUT_S,
                                               context=ssl.create_default_context())
        else:
            conn = http.client.HTTPConnection(u.hostname, u.port, timeout=POLL_TIMEOUT_S)
        try:
            conn.request("GET", path, headers={"User-Agent": USER_AGENT, "Cache-Control": "no-store", **headers})
            r = conn.getresponse()
            return r.status, r.read(limit)
        finally:
            conn.close()

    def _wanted(self) -> str:
        """One standby poll: "wanted", "idle", "refused", or what went wrong."""
        try:
            status, body = self._get(gp.WANTED_PATH, {gp.TOKEN_HEADER: self.o.token()}, 4096)
        except (OSError, http.client.HTTPException) as e:
            return type(e).__name__
        if 300 <= status < 400 or status in (401, 403, 404):
            log(self._refusal(status))
            return "refused"
        if status != 200:
            return f"HTTP {status}"
        try:
            return "wanted" if json.loads(body).get("wanted") is True else "idle"
        except (ValueError, AttributeError):
            return "an answer that is not JSON"

    def _live_offered(self) -> bool | None:
        """Whether the server uses a GPU worker for live cleaning (its public /api/info, gpu.live), or
        None when it cannot say. The team's laptop asks once, so it never holds the live teacher on its
        small GPU for a server that will not use it (CrisperWhisper then stays loaded for recordings)."""
        try:
            status, body = self._get("/api/info", {}, 1 << 20)
            info = json.loads(body) if status == 200 else None
            return bool((info.get("gpu") or {}).get("live")) if isinstance(info, dict) else None
        except (OSError, http.client.HTTPException, ValueError, AttributeError):
            return None

    def _refusal(self, status: int) -> str:
        if 300 <= status < 400:
            return (f"the server answered with a redirect (HTTP {status}), which a worker never follows (it would "
                    "carry the token elsewhere): use the server's own address")
        secret = f"the secret (key id {self.o.key_id})" if self.o.key_id else "the token"
        return f"the server refused this worker (HTTP {status}): check the URL, {secret} and that the service is public"

    def _outage(self, kind: str | None):
        """One log line when the server stops answering, one when it answers again."""
        if kind and not self.outage:
            log(f"cannot reach the server ({kind}); trying again")
        elif kind is None and self.outage:
            log("the server answers again")
        self.outage = kind

    # ---- one connection -----------------------------------------------------------------------

    async def _session(self) -> str:
        """One connection, from the handshake to its close. Returns how it ended: done (goodbyes said),
        busy (another worker serves), rotate, lost (try again), bad (malformed), refused, version."""
        self.gen += 1
        self.welcomed, self.end, self.closing_at = False, None, None
        try:
            ws = await NoRedirects(gp.ws_url(self.o.server), additional_headers={gp.TOKEN_HEADER: self.token},
                                   max_size=gp.MAX_FRAME, ping_interval=20, ping_timeout=20, open_timeout=30,
                                   user_agent_header=USER_AGENT, compression=None)
        except InvalidStatus as e:
            status = e.response.status_code
            if status in (408, 429) or status >= 500:  # starting, or busy: try again
                self._outage(f"HTTP {status}")
                return "lost"
            log(self._refusal(status))
            return "refused"
        except Exception as e:  # the network, a timeout, something that is not a WebSocket server
            self._outage(type(e).__name__)
            return "lost"
        self._outage(None)
        self.ws, self.out = ws, asyncio.Queue()
        self.connected_at = self.last_heard = time.monotonic()
        self._send({"t": gp.HELLO, "proto": gp.PROTO, "worker": self.id, "wake": gp.peek_token(self.token).wake,
                    "role": self.o.role, "where": self.where, "commit": self.o.commit, "gpus": self.gpus,
                    "deadline": self.o.deadline, "loading": bool(self.loading)})
        tasks = [asyncio.create_task(c) for c in (self._reader(ws), self._writer(ws), self._timer(ws))]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in done:  # a bug in a handler must show, not pass for a lost connection
                e = t.exception()
                if e is not None and not isinstance(e, ConnectionClosed):
                    log("unexpected error:\n" + "".join(traceback.format_exception(e)))
                    self.end = "fatal"
        except asyncio.CancelledError:  # Ctrl+C: goodbye first, if the server knows this worker
            if self.welcomed and self.end != "done":
                await self._bye_now(ws, "shutdown")
            raise
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await asyncio.wait_for(ws.close(), BYE_WAIT_S)
            except Exception:
                pass
            self._drop()
            self.ws, self.out = None, None
        if self.end:
            return self.end
        code = ws.close_code
        return {gp.CLOSE_VERSION: "version", gp.CLOSE_BUSY: "busy", gp.CLOSE_SHUTDOWN: "done",
                gp.CLOSE_BAD: "bad"}.get(code, "lost")

    async def _bye_now(self, ws, reason: str):
        try:
            await asyncio.wait_for(ws.send(gp.encode({"t": gp.BYE, "reason": reason})), 2)
            await asyncio.wait_for(ws.close(1000, "bye"), 2)
        except Exception:
            pass

    async def _reader(self, ws):
        try:
            async for message in ws:
                self.last_heard = time.monotonic()
                if isinstance(message, str):
                    return await self._malformed(ws, "a text message")
                try:
                    h, payload = gp.decode(message, expect=gp.FROM_SERVER)
                except gp.ProtocolError as e:
                    return await self._malformed(ws, str(e))
                self._handle(h, payload)
                if self.end in ("version",):
                    await ws.close()
                    return
        except ConnectionClosed:
            pass

    async def _malformed(self, ws, why: str):
        log(f"a malformed message from the server ({gp.printable(why, 80)}); reconnecting")
        self.end = "bad"
        await ws.close(gp.CLOSE_BAD, "malformed message")

    async def _writer(self, ws):
        while True:
            await ws.send(await self.out.get())

    async def _timer(self, ws):
        """Heartbeats, and what time decides: a silent server, idleness, the deadline, rotation."""
        beat = 0.0
        while True:
            await asyncio.sleep(min(0.5, gp.HEARTBEAT_S / 4))
            now = time.monotonic()
            if now - self.last_heard > gp.DEAD_AFTER_S:
                log(f"nothing from the server for {gp.DEAD_AFTER_S:.0f} s; reconnecting")
                self.end = self.end or "lost"
                ws.transport.abort()
                return
            if self.closing_at is not None:
                if now >= self.closing_at:  # BYE said, and the server did not close: close it here
                    await ws.close(1000, "bye")
                    return
                continue
            if not self.welcomed:
                continue
            if now >= beat:
                self._heartbeat()
                beat = now + gp.HEARTBEAT_S
            s = self.stream
            if s is not None and now - s.last_audio > gp.LIVE_IDLE_S:
                self._send({"t": gp.LIVE_ERROR, "stream": s.id,
                            "message": f"No audio for {gp.LIVE_IDLE_S:.0f} s, so the live stream was closed."})
                self._end_stream("dropped: no audio")
            reason = self._why_leave(now)
            if reason:
                log({"idle": "nothing to do for a while; saying goodbye",
                     "deadline": "this machine's time is nearly up; saying goodbye"}[reason])
                self._leave(reason)
            elif self._idle() and now - self.connected_at >= (self.rotate_after_s or gp.ROTATE_AFTER_S):
                log("refreshing the connection before the server's time limit for it")
                self.end = "rotate"
                await ws.close(gp.CLOSE_ROTATE, "rotate")
                return

    def _why_leave(self, now: float) -> str | None:
        if self.o.deadline is not None:
            wall = time.time()
            if wall >= self.o.deadline or (wall >= self.o.deadline - gp.DRAIN_S and self._idle()):
                return "deadline"
        if self.ready_at is not None and self._idle() and now - self.last_work >= self._idle_limit() + IDLE_MARGIN_S:
            return "idle"
        return None

    def _idle(self) -> bool:
        return self.job is None and self.stream is None

    def _idle_limit(self) -> float:
        exit_s = self.idle_exit_s or self.o.idle_exit_s or gp.IDLE_EXIT_S
        unused_s = self.idle_unused_s or min(gp.IDLE_UNUSED_S, exit_s)
        return exit_s if self.used else unused_s

    def _leave(self, reason: str):
        """Say goodbye; the server then closes the socket (or the timer does, after BYE_WAIT_S)."""
        if self.end == "done":
            return
        self._drop()
        self._send({"t": gp.BYE, "reason": reason})
        self.end, self.closing_at = "done", time.monotonic() + BYE_WAIT_S

    def _send(self, h: dict, payload: bytes = b""):
        if self.out is not None:
            self.out.put_nowait(gp.encode(h, payload))

    def _call(self, fn: Callable, *args):
        """Run fn(*args) on the event loop, from another thread."""
        try:
            self.loop.call_soon_threadsafe(fn, *args)
        except RuntimeError:  # the loop has closed: the process is exiting
            pass

    def _caps(self):
        models = {"live": getattr(self.live, "name", None), "recordings": getattr(self.notes, "name", None)}
        self._send({"t": gp.CAPS, "notes": "notes" in self.ready, "live": "live" in self.ready,
                    "loading": bool(self.loading), "window_ms": self.window_ms, "models": models, "error": self.error})

    def _heartbeat(self):
        ms = float(np.percentile(self.recent_ms, 90)) if self.recent_ms else self.window_ms
        self._send({"t": gp.HEARTBEAT, "jobs": [self.job.id] if self.job else [], "live": self.stream is not None,
                    "window_ms": None if ms is None else round(ms, 1), "gpu_mem_gb": engines.memory_gb()})

    def _drop(self):
        """The connection's work ends with it: the server sends it again."""
        if self.job is not None:
            self.job.cancelled = True
            self.job = None
            self.last_work = time.monotonic()
        if self.stream is not None:
            self._end_stream("closed: disconnected")

    # ---- messages -----------------------------------------------------------------------------

    def _handle(self, h: dict, payload: bytes):
        t = h["t"]
        if self.end == "done":  # goodbye said: nothing more
            return
        if t == gp.PING:
            self._send({"t": gp.PONG, "n": h["n"], "ts": h["ts"]})
        elif t == gp.WELCOME:
            self._welcome(h)
        elif t == gp.JOB:
            self._job(h)
        elif t == gp.JOB_AUDIO:
            self._job_audio(h, payload)
        elif t == gp.CANCEL:
            self._cancel(h["job"])
        elif t == gp.SHUTDOWN:
            log(f"the server says goodbye ({h['reason']})")
            self._leave(BYE_FOR[h["reason"]])
        elif t == gp.LIVE_OPEN:
            self._live_open(h)
        elif t == gp.LIVE_AUDIO:
            self._live_audio(h, payload)
        elif t == gp.LIVE_CLOSE:
            if self.stream is not None and self.stream.id == h["stream"]:
                self._end_stream("closed")

    def _welcome(self, h: dict):
        if h["proto"] != gp.PROTO:
            log(f"the server speaks protocol {h['proto']} and this worker {gp.PROTO}: update one of them")
            self.end = "version"
            return
        self.idle_exit_s, self.idle_unused_s, self.rotate_after_s = (
            max(0.01, float(h[k])) for k in ("idle_exit_s", "idle_unused_s", "rotate_after_s"))
        if h["commit"] != self.o.commit:
            log(f"the server runs code {gp.printable(h['commit'], 12)}, this worker {self.o.commit[:12]}")
        self.welcomed = True
        log(f"connected to {self.o.server}" + (" (still loading the models)" if self.loading else ""))
        self._caps()
        if self.error:
            self._leave("error")

    # A recording

    def _job(self, h: dict):
        j = self.job
        if "notes" not in self.ready or (j is not None and j.running):
            why = "The models are still loading." if "notes" not in self.ready else "One recording at a time."
            return self._send({"t": gp.ERROR, "job": h["job"], "message": why, "retry": True})
        if j is not None:  # a recording that never arrived whole
            self._job_failed(j, "The recording arrived incomplete.", retry=True)
        self.job, self.used = Job(h, self.gen), True

    def _job_audio(self, h: dict, payload: bytes):
        j = self.job
        if j is None or j.id != h["job"] or j.running:  # a part of a recording dropped or cancelled
            return
        if h["i"] != j.next or j.received + len(payload) > len(j.audio):
            return self._job_failed(j, "The recording arrived out of order or too long.", retry=True)
        j.audio[j.received:j.received + len(payload)] = payload
        j.received += len(payload)
        j.next += 1
        if j.next < j.parts:
            return
        if j.received != len(j.audio):
            return self._job_failed(j, "The recording arrived incomplete.", retry=True)
        j.running = True
        self.jobs.put(j)

    def _cancel(self, job_id: str):
        j = self.job
        if j is not None and j.id == job_id:
            j.cancelled, self.job, self.last_work = True, None, time.monotonic()
            log(f"{job_id}: cancelled")

    def _job_failed(self, j: Job, message: str, retry: bool):
        if self.job is j:
            self.job = None
        self.last_work = time.monotonic()
        if j.gen == self.gen:
            self._send({"t": gp.ERROR, "job": j.id, "message": gp.printable(message, gp.MAX_TEXT), "retry": retry})

    def _job_thread(self):
        """Recordings, one at a time, off the event loop."""
        while True:
            j = self.jobs.get()
            if j.cancelled:
                continue
            t = time.perf_counter()
            try:
                out, err = self.notes.analyse(gp.from_pcm16(j.audio), partial(self._call, self._progress, j),
                                              lambda j=j: j.cancelled, j.language), None
            except Exception as e:
                out, err = None, e
            j.audio = None
            if self.o.delay_ms:
                time.sleep(self.o.delay_ms / 1000)
            self._call(self._job_done, j, out, err, time.perf_counter() - t)

    def _progress(self, j: Job, stage: str):
        if j is self.job and not j.cancelled and j.gen == self.gen:
            self._send({"t": gp.PROGRESS, "job": j.id, "stage": stage})

    def _job_done(self, j: Job, out, err: Exception | None, seconds: float):
        if self.job is j:
            self.job = None
        self.last_work = time.monotonic()
        if err is not None and engines.fatal(err):
            log(f"{j.id}: {type(err).__name__}: the GPU is in a bad state; the worker stops so a fresh one can start")
            os._exit(FAILED)
        if j.cancelled or j.gen != self.gen or out is None and err is None:
            log(f"{j.id}: result dropped (cancelled, or the connection it came on is gone)")
            return
        self.jobs_done += 1
        if self.o.crash_after and self.jobs_done >= self.o.crash_after:
            log(f"{j.id}: crashing on purpose (--crash-after {self.o.crash_after})")
            os._exit(FAILED)
        if err is not None:
            oom = "out of memory" in str(err).lower()
            retry = oom or isinstance(err, RuntimeError)
            log(f"{j.id}: failed after {seconds:.1f} s ({type(err).__name__})")
            return self._job_failed(j, f"{type(err).__name__}: {err}", retry)
        probs, words, slow, timing = out
        frames = gp.frames_for(j.samples)
        probs = np.asarray(probs, np.float32)
        if probs.shape != (frames, gp.N_CLASSES):
            return self._job_failed(j, f"The teacher gave {probs.shape} probabilities for {frames} frames.", False)
        probs = np.clip(np.nan_to_num(probs, nan=0.0), 0.0, 1.0)  # half precision can round past 1
        try:
            words = gp.check_words([{**w, "w": w["w"][:100]} if isinstance(w, dict) and isinstance(w.get("w"), str)
                                    else w for w in words], j.samples / gp.SR)
            frame = gp.encode({"t": gp.RESULT, "job": j.id, "frames": frames, "words": words, "slow": bool(slow),
                               "timing": timing}, gp.probs_bytes(probs))
        except gp.ProtocolError as e:  # the server would refuse it: say so instead of sending it again
            return self._job_failed(j, f"The result could not be sent: {e}", retry=False)
        self.out.put_nowait(frame)
        log(f"{j.id}: {j.samples / gp.SR:.1f} s of audio in {seconds:.1f} s (teacher "
            f"{timing.get('teacher_s', 0):.1f} s, ASR {timing.get('asr_s', 0):.1f} s{', on the CPU' if slow else ''})")

    # A live stream (P2)

    def _live_open(self, h: dict):
        sid = h["stream"]
        if self.stream is not None:
            problem = "One live stream at a time."
        elif "live" not in self.ready:
            problem = "The live teacher is not loaded on this worker."
        elif not MIN_WINDOW <= h["window"] <= MAX_WINDOW:
            problem = "This worker does not serve that window size."
        else:
            problem = None
        if problem:
            return self._send({"t": gp.LIVE_ERROR, "stream": sid, "message": problem})
        self.used = True
        self.stream = LiveStream(h, self.gen, self.live, self.o.delay_ms / 1000, self._from_stream)
        self.stream.thread.start()  # it says LIVE_READY once its model is warm
        log(f"{sid}: live stream open ({h['mode']}, {h['lookahead_ms']:.0f} ms ahead)")

    def _live_audio(self, h: dict, payload: bytes):
        s = self.stream
        if s is not None and s.id == h["stream"]:
            s.last_audio = time.monotonic()
            s.q.put((h, payload))

    def _from_stream(self, what: str, s: LiveStream, *args):
        self._call(self._stream_event, what, s, *args)

    def _stream_event(self, what: str, s: LiveStream, *args):
        if s is not self.stream or s.gen != self.gen:
            return
        if what == "ready":
            self._send({"t": gp.LIVE_READY, "stream": s.id})
            return
        if what == "reply":
            frame, ms = args
            self.last_work = time.monotonic()
            self.recent_ms.append(ms)
            self.windows_done += 1
            if self.o.crash_after and not self.jobs_done and self.windows_done >= self.o.crash_after:
                log(f"{s.id}: crashing on purpose (--crash-after {self.o.crash_after})")
                os._exit(FAILED)
            self.out.put_nowait(frame)
            return
        (err,) = args
        if engines.fatal(err):
            log(f"{s.id}: {type(err).__name__}: the GPU is in a bad state; the worker stops so a fresh one can start")
            os._exit(FAILED)
        log(f"{s.id}: failed ({type(err).__name__})")
        message = gp.printable(f"{type(err).__name__}: {err}", gp.MAX_TEXT)
        self._send({"t": gp.LIVE_ERROR, "stream": s.id, "message": message})
        self._end_stream("failed")

    def _end_stream(self, why: str):
        s, self.stream = self.stream, None
        s.stop()
        self.last_work = time.monotonic()
        log(f"{s.id}: live stream {why} after {s.windows} windows")
