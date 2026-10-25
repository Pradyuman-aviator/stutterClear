"""StutterClear app server: the Fluency Mirror UI on http://127.0.0.1:8765.

    python scripts/run_app.py

The UI is a Vite + React build in static/ (source in web/: `npm run build` there refreshes it).
"""
import asyncio
import ipaddress
import json
import os
import re
import threading
import time
from collections import deque
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.middleware.gzip import GZipMiddleware

from .. import feedback, gpu_protocol as gp, journal, live, nemotron
from ..audio import wav_bytes
from ..config import SAMPLE_RATE as SR
from ..live import DEFAULT_STUDENT, LiveSession, detectors, devices, recommended, student_eval
from . import demo, gpu, notes, video

STATIC = Path(__file__).parent / "static"
# A public deployment: visitors bring their own mic through the browser; the server never opens its
# own audio devices and never writes recordings to disk.
HOSTED = os.environ.get("STUTTERCLEAR_HOSTED") == "1"
UPDATE_S = 0.03  # ~30 updates a second: the orb reacts within a frame or two
KEEPALIVE_S = 15.0  # a page with no session gets one message this often, so its socket stays open
# One live session at a time, so on a public deployment a tab left open must not hold it for ever.
LIVE_LIMIT_S = 300.0 if HOSTED else None
MIC_IDLE_S = 15.0  # no audio for this long: the page froze or the laptop went to sleep
SETTINGS_S = 10.0  # a browser mic sends its settings this soon after connecting
FRAME_MAX = 32_000  # bytes in one browser mic message: half a second of 16 kHz float32
AHEAD_S = 3.0  # browser audio may run this far ahead of the clock (a burst after a network stall)
PAGE_MAX_S = 1800.0  # a public server closes a page's socket after this long; an active page reconnects
SOCKETS_PER_CLIENT = 10  # page and mic sockets one visitor (one network) may hold on a public server
# Live time one visitor may use in an hour on a public server, where the one live slot is shared, and
# how long they wait after a session someone else was turned away from (that one goes next)
LIVE_PER_HOUR_S = 1200.0 if HOSTED else None
LIVE_TURN_S = 30.0
BUSY_HOSTED = "Someone else is live right now (turns last up to 5 minutes). Try again shortly, or play a real recording."
LIVE_SPENT = "You have had live cleaning for 20 minutes this hour. Try again later, or play a real recording."
LIVE_YIELD = "Someone else was waiting for live cleaning, so it is their turn. Try again in half a minute."
TOO_MANY_PAGES = "Too many StutterClear pages are open from your network. Close one and this page reconnects."
CLOSE_TOO_MANY = 4429  # the page says why, instead of "can't reach the server"
FOREIGN = "Requests from other websites are not accepted."
# Visitors' Nemotron calls spend the owner's Nebius credits: a public demo caps them per minute.
NEMOTRON_PER_MIN = 60 if HOSTED else None
_nemotron_calls: deque = deque()
TTS_PER_MIN = 30 if HOSTED else None  # Say it spends the owner's NVIDIA credits the same way
_tts_calls: deque = deque()


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return default


# Behind proxies (Cloud Run's front end, Caddy) the visitor's address is in X-Forwarded-For: each proxy
# appends the address it saw, so behind N of them the visitor is the N-th entry from the end
# (STUTTERCLEAR_TRUST_PROXY=N; 1 for Cloud Run or Caddy alone). 0 or unset: the socket's own address.
TRUST_PROXY = _int_env("STUTTERCLEAR_TRUST_PROXY", 0)
# The front door (a static page elsewhere) may read /api/health from these origins.
CORS_ORIGINS = {o.strip() for o in (os.environ.get("STUTTERCLEAR_CORS_ORIGINS")
                                    or "https://pradyuman-aviator.github.io").split(",") if o.strip()}
# Bandwidth out of the cloud is the one cost a visitor could run up on purpose (public servers only):
# bytes an hour per visitor and for everyone, and a day for everyone
EGRESS_CLIENT = float(os.environ.get("STUTTERCLEAR_EGRESS_MB_PER_CLIENT") or 250) * 1e6
EGRESS_TOTAL = float(os.environ.get("STUTTERCLEAR_EGRESS_MB") or 1000) * 1e6
EGRESS_DAY = float(os.environ.get("STUTTERCLEAR_EGRESS_MB_PER_DAY") or 8000) * 1e6
# What the budget refuses once spent: the audio and video downloads, never the page or its small calls
MEDIA = re.compile(r"/api/(demo/[^/]+/(original|clean)\.wav|playback/(original|clean)\.wav|video/[^/]+/clean\.mp4|tts)")
DOWNLOADS = "Too much downloading from this server for now. Try again in a while."
# Request bodies: what each route takes at most (the rest are small JSON calls), how many large ones
# may arrive at once (two per visitor), and how long one may stall
JSON_MAX = 256 * 1024
FEEDBACK_MAX = 1024 * 1024  # a browser's own review log: up to 2000 decisions
BIG_BODIES = 4
BODY_STALL_S = 30.0
NO_STORE = {"Cache-Control": "no-store"}
PRIVATE = {"Cache-Control": "private, no-store"}  # a visitor's own words: never kept by a browser or a proxy
# "last" holds the finished session's audio, in memory only, for the play buttons, for an hour at
# most. Starting a new session drops it.
# On a public server a session belongs to the page that started it ("owner", a random id the page
# sends): only that page gets its live updates (waveforms, captions) and its listen-back.
# "claim": a browser mic is setting up a session (its slot is taken before anything can interleave).
state = {"session": None, "last": None, "last_at": None, "owner": None, "last_owner": None, "claim": False}
LAST_KEEP_S = 3600
notes.live_running = lambda: state["session"] is not None  # a voice note leaves the live teacher alone
# A GPU worker elsewhere can run the live teacher here (stutterclear/app/gpu.py, P2)
live.remote_offered = lambda: gpu.broker.live_offered()
live.remote_live = lambda mode: gpu.broker.open_live(mode)
clients: dict[WebSocket, str] = {}  # open pages -> their page id
_sent: dict[WebSocket, tuple] = {}  # open pages -> (their last message was a tick, when it went)
_sockets: dict[str, int] = {}  # page and mic sockets per visitor
IDLE = {"idle": True}
PAGE_RE = re.compile(r"[0-9a-f]{32}")


def sees_session(page: str) -> bool:
    """Whether a page gets the live session's updates: every page on this computer; on a public
    server only the page that started it."""
    return not (HOSTED and state["session"] and (not page or page != state["owner"]))


_proxy_seen: set = set()  # how many X-Forwarded-For entries requests had (each count logged once)


def client_ip(scope) -> str:
    """Who is asking, for the per-visitor limits. Behind proxies (STUTTERCLEAR_TRUST_PROXY=N) the N-th
    X-Forwarded-For entry from the end: each proxy appends the address it saw, and anything before
    is whatever the client sent. IPv6 addresses count per /64 (one home or phone). Never logged."""
    ip = ""
    if TRUST_PROXY:
        forwarded = [v for k, v in scope.get("headers") or [] if k == b"x-forwarded-for"]
        entries = [e.strip() for e in b",".join(forwarded).decode("latin-1").split(",") if e.strip()]
        if len(entries) not in _proxy_seen and len(_proxy_seen) < 8:  # a count, never an address
            _proxy_seen.add(len(entries))
            print(f"proxy: a request with {len(entries)} X-Forwarded-For entries; the visitor is entry "
                  f"{TRUST_PROXY} from the end", flush=True)
        ip = entries[-TRUST_PROXY] if len(entries) >= TRUST_PROXY else ""
    if not ip:
        ip = (scope.get("client") or ("",))[0] or ""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return ip[:64]
    if addr.version == 6:
        if addr.ipv4_mapped is not None:  # ::ffff:1.2.3.4 is 1.2.3.4
            return str(addr.ipv4_mapped)
        return str(ipaddress.ip_network(f"{addr}/64", strict=False))
    return str(addr)


def _same_site(origin: bytes, host: bytes) -> bool:
    try:
        return gp.audience(origin.decode("latin-1")) == gp.audience(host.decode("latin-1"))
    except ValueError:  # "null" (a sandboxed frame), garbage
        return False


class SameOrigin:
    """Requests that change something (any method but GET, HEAD and OPTIONS) and every WebSocket must
    come from this site's own pages: a page on another site could otherwise make a visitor's browser
    wake the GPU, upload recordings or take the live slot. No Origin header (curl, the GPU worker)
    passes; a browser always sends one with these."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket" or (scope["type"] == "http" and scope["method"] not in ("GET", "HEAD", "OPTIONS")):
            headers = dict(scope["headers"])
            origin = headers.get(b"origin")
            if origin is not None and not _same_site(origin, headers.get(b"host", b"")):
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008, "reason": ""})  # before accept: HTTP 403
                else:
                    await JSONResponse({"detail": FOREIGN}, 403)(scope, receive, send)
                return
        await self.app(scope, receive, send)


_egress: dict[str, dict[int, int]] = {}  # visitor ("" = everyone) -> minute -> response bytes (the last hour)
_egress_day = [0, 0]  # the UTC day, and everyone's response bytes in it


class EgressBudget:
    """On a public server, response bytes per visitor and in total, by the minute. Past a budget (the
    visitor's this hour; everyone's this hour or today) the audio and video downloads answer 429 until
    it drains; the page, its scripts and the small JSON calls always answer, so a visitor sees why.
    /api/health and WebSockets are not counted."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if not HOSTED or scope["type"] != "http" or scope["path"] == "/api/health":
            await self.app(scope, receive, send)
            return
        who = client_ip(scope)
        minute = int(time.time() // 60)
        _sweep_egress(minute)
        if MEDIA.fullmatch(scope["path"]) and (_last_hour(who, minute) > EGRESS_CLIENT or egress_spent(minute)):
            await JSONResponse({"detail": DOWNLOADS}, 429)(scope, receive, send)
            return

        async def counted(message):
            if message["type"] == "http.response.body":
                n = len(message.get("body", b""))
                for k in (who, ""):
                    b = _egress.setdefault(k, {})
                    b[minute] = b.get(minute, 0) + n
                if _egress_day[0] != minute // 1440:
                    _egress_day[:] = [minute // 1440, 0]
                _egress_day[1] += n
            await send(message)
        await self.app(scope, receive, counted)


def _last_hour(key: str, minute: int) -> int:
    return sum(n for m, n in _egress.get(key, {}).items() if m > minute - 60)


def egress_spent(minute: int) -> bool:
    """Everyone's budget is used up (this hour's, or today's): the media downloads wait."""
    return _last_hour("", minute) > EGRESS_TOTAL or (_egress_day[0] == minute // 1440 and _egress_day[1] > EGRESS_DAY)


_egress_swept = [0]


def _sweep_egress(minute: int):
    """Once a minute: minutes older than an hour, and visitors with none left, are forgotten."""
    if _egress_swept[0] == minute:
        return
    _egress_swept[0] = minute
    for k in list(_egress):
        b = _egress[k]
        for m in [m for m in b if m <= minute - 60]:
            del b[m]
        if not b:
            del _egress[k]


def _content_length(scope) -> int | None:
    for k, v in scope.get("headers") or []:
        if k == b"content-length":
            try:
                return int(v)
            except ValueError:
                return None
    return None


def _body_limit(method: str, path: str) -> int:
    if method == "POST" and path == "/api/notes":
        return notes.MAX_BYTES
    if method == "PUT" and path.startswith("/api/video/upload/"):
        return video.PART_BYTES
    if method == "POST" and path == "/api/video":
        return video.PART_BYTES if HOSTED else video.MAX_BYTES  # (the page always sends parts)
    if path == "/api/feedback/summary":
        return FEEDBACK_MAX
    return JSON_MAX


_big: dict[str, int] = {}  # large request bodies being received: "" = everyone, else per visitor


class BodyLimit:
    """Request bodies: never more than their route takes (413, by Content-Length or as they arrive),
    never stalled for long (408), and only BIG_BODIES large ones at once (503), two per visitor (429).
    The outermost layer, so nothing reads a body before these checks."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] in ("GET", "HEAD", "OPTIONS"):
            await self.app(scope, receive, send)
            return
        limit, length = _body_limit(scope["method"], scope["path"]), _content_length(scope)
        if length is not None and length > limit:
            await JSONResponse({"detail": "That is too large for this server."}, 413)(scope, receive, send)
            return
        who = client_ip(scope) if limit > JSON_MAX and (length is None or length > JSON_MAX) else None
        if who is not None:
            refusal = ((503, "The server is receiving other uploads right now. Try again in a moment.")
                       if _big.get("", 0) >= BIG_BODIES else
                       (429, "Two uploads at a time, please.") if _big.get(who, 0) >= 2 else None)
            if refusal:
                await JSONResponse({"detail": refusal[1]}, refusal[0])(scope, receive, send)
                return
            for k in ("", who):
                _big[k] = _big.get(k, 0) + 1
        seen, whole = 0, False

        async def bounded():
            nonlocal seen, whole
            if whole:  # the body is in: from here on only a disconnect comes (a streamed answer waits for it)
                return await receive()
            try:
                message = await asyncio.wait_for(receive(), BODY_STALL_S)
            except asyncio.TimeoutError:
                raise HTTPException(408, "The upload stalled. Please try again.") from None
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > limit:
                    raise HTTPException(413, "That is too large for this server.")
                whole = not message.get("more_body", False)
            return message
        try:
            await self.app(scope, bounded, send)
        finally:
            if who is not None:
                for k in ("", who):
                    _big[k] -= 1
                    if not _big[k]:
                        del _big[k]


class TextGZip:
    """gzip for the page and its JSON (the UI's scripts shrink to about a third); never for audio,
    video or their ranges, which gain nothing and must keep their exact bytes."""

    def __init__(self, app):
        self.app = app
        self.gzip = GZipMiddleware(app, minimum_size=1024, compresslevel=6)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and not MEDIA.fullmatch(scope["path"]):
            await self.gzip(scope, receive, send)
        else:
            await self.app(scope, receive, send)


class UI(StaticFiles):
    """The built page: its assets are named by their content, so caches keep them a year; the page
    itself is checked again each time (a new deploy shows at once)."""

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        assets = path.replace(os.sep, "/").startswith("assets/")  # (the path comes with the system's separator)
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable" if assets else "no-cache"
        return response


def _claim_socket(client: str) -> bool:
    """One more page or mic socket for this visitor, unless a public server's cap is reached."""
    if HOSTED and _sockets.get(client, 0) >= SOCKETS_PER_CLIENT:
        return False
    _sockets[client] = _sockets.get(client, 0) + 1
    return True


def _release_socket(client: str):
    n = _sockets.get(client, 1) - 1
    if n > 0:
        _sockets[client] = n
    else:
        _sockets.pop(client, None)


def _forget_last():
    """The finished session's audio is kept for an hour at most."""
    if state["last"] is not None and state["last_at"] is not None and time.time() - state["last_at"] > LAST_KEEP_S:
        state["last"] = None


def _forget_old():
    """Once a minute: voice notes, videos and abandoned uploads past their time are forgotten even on a
    server nobody uses meanwhile (in a worker thread: deleting an upload's folder takes a moment)."""
    for expire in (notes.expire, video.expire):
        try:
            expire()
        except Exception as e:  # never stop the broadcast; the next minute tries again
            print(f"forgetting old recordings failed ({type(e).__name__})", flush=True)


async def broadcast():
    """Poll the session once per tick and send the same message to every open page.

    poll() hands out what is new since its last call, so polling per connection would split the
    audio between tabs (or an OBS browser source and the main window). A page that is not watching a
    session gets IDLE once when its session ends, and then only every KEEPALIVE_S.
    """
    loop, swept = asyncio.get_running_loop(), time.monotonic()
    while True:
        await asyncio.sleep(UPDATE_S)
        _forget_last()
        if time.monotonic() - swept > 60:
            swept = time.monotonic()
            loop.run_in_executor(None, _forget_old)
            for client in list(_live_used):
                _live_spent(client, swept)  # (visitors gone for an hour are forgotten)
            for client in [c for c, until in _live_rest.items() if until <= swept]:
                del _live_rest[client]
        if not clients:
            continue
        s = state["session"]
        msg = s.poll() if s else None
        now = time.monotonic()
        for socket, page in list(clients.items()):
            out = msg if msg is not None and sees_session(page) else None
            was_tick, at = _sent.get(socket, (True, 0.0))
            if out is None:
                if not was_tick and now - at < KEEPALIVE_S:
                    continue
                out = IDLE
            try:
                await socket.send_json(out)
                _sent[socket] = (out is not IDLE, now)
            except Exception:  # closed mid-send; its handler cleans up
                clients.pop(socket, None)


def warm_demo():
    """Load the big model and render every demo sample once in the background, so the first live
    session and the first click on a sample start at once. Nemotron's model is checked first:
    the meaning guard runs while the samples render. Never wakes a GPU worker."""
    try:
        from .. import captions, context_agent, tts  # noqa: F401  (imported once here, not on a visitor's click)
        if nemotron.available():
            print(nemotron.resolve(), flush=True)
        from ..teacher_runtime import get_teacher, live_capable, window_probs
        if live_capable():
            get_teacher()
            window_probs(np.zeros(16000, np.float32))
        samples = demo.available()
        _warm_each(samples, demo.render)  # every render first: the page needs those before captions
    finally:
        notes.warm.set()  # voice notes may take the GPU now (the captions below run on the CPU)
    _warm_each(samples, demo.captions)


def _warm_each(samples: list, fn):
    for s in samples:
        for engine in s["engines"]:
            try:
                fn(s["id"], engine)
            except Exception:  # a broken sample must not take the server down; it errors on click
                pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    task = asyncio.create_task(broadcast())
    notes.warm.clear()  # voice notes wait: the warm-up uses the live teacher on the GPU
    threading.Thread(target=warm_demo, daemon=True).start()
    gpu.broker.start()
    video.sweep()  # video folders a crash left behind
    yield
    # No word to a GPU worker here: it reconnects to the next instance, which adopts it.
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


app = FastAPI(title="StutterClear", lifespan=lifespan)
app.add_middleware(TextGZip)  # (the last added is the outermost: requests pass BodyLimit first)
app.add_middleware(EgressBudget)  # counts what actually goes out, compressed
app.add_middleware(SameOrigin)
app.add_middleware(BodyLimit)


class StartRequest(BaseModel):
    mode: str = "gentle"
    in_device: int | None = None
    out_device: int | None = None
    # Names win over indices: Windows renumbers devices whenever one (say, Bluetooth earbuds)
    # connects or drops, so an index the page remembered can point at nothing, or the wrong mic.
    in_name: str | None = None
    out_name: str | None = None
    detector: str = "student"
    save: bool = False
    sensitivity: float = live.SENSITIVITY  # 0-100: how readily a stutter is cut (live.py)


class SensitivityRequest(BaseModel):
    sensitivity: float


def resolve(kind: str, index: int | None, name: str | None, available: list[dict]) -> int | None:
    if name:
        for d in available:
            if d["name"] == name:
                return d["index"]
        raise HTTPException(409, f"{name} is not connected any more. Pick a {kind} again.")
    if index is not None and all(d["index"] != index for d in available):
        raise HTTPException(409, f"That {kind} is not available any more. Pick one again.")
    return index


@app.get("/api/devices")
def list_devices():
    return devices()


@app.get("/api/health")
def health(request: Request):
    """For the front door (a static page elsewhere): is this server up, which code, is the live slot
    free. From memory only; CORS for the listed origins only."""
    headers = {**NO_STORE, "Vary": "Origin"}
    origin = request.headers.get("origin")
    if origin in CORS_ORIGINS:
        headers["Access-Control-Allow-Origin"] = origin
    out = {"ok": True, "service": "stutterclear", "version": gpu.version(), "hosted": HOSTED,
           "gpu": gpu.broker.health(), "live_free": state["session"] is None and not state["claim"]}
    if HOSTED and egress_spent(int(time.time() // 60)):  # the recordings would not play: send visitors elsewhere
        out.update(ok=False, reason="downloads")
    return JSONResponse(out, headers=headers)


@app.get("/api/info")
def info():
    # The AI detector needs its downloaded checkpoint; its measured quality (if evaluated) lets the
    # page recommend it only when it is actually better.
    from .. import captions, context_agent, tts
    out = {"student": DEFAULT_STUDENT.exists(), "student_eval": student_eval(), "notes": notes.available(),
           "captions": captions.available(), "rescue": nemotron.available(), "nemotron": nemotron.available(),
           "search": context_agent.search_available(), "model": nemotron.MODEL, "hosted": HOSTED,
           "tts": tts.available(), "tts_voice": tts.VOICE, "tts_label": tts.LABEL, "interview": nemotron.available(),
           "feedback": "browser" if HOSTED else "server", "uclass_demos": demo.uclass_enabled(),
           "video": video.available(), "video_max_mb": video.MAX_BYTES // 10**6, "version": gpu.version(),
           "detectors": detectors(), "recommended": recommended()}
    g = gpu.broker.info()  # only when a GPU worker is set up; static, like the rest
    if g is not None:
        remote = notes.remote() and notes.available()  # the features that go through the worker
        g["features"] = ((["notes", "dictate"] if remote else []) + (["video"] if remote and out["video"] else [])
                         + (["live"] if g["live"] else []))
        out["gpu"] = g
    return out


@app.get("/api/gpu")
def gpu_status(request: Request):
    """The GPU worker's state for the page (from memory: never a call to Kaggle)."""
    if not gpu.broker.configured():
        raise HTTPException(404, gpu.NOT_SET_UP)
    return JSONResponse(gpu.broker.status(client_ip(request.scope)), headers=NO_STORE)


class WakeRequest(BaseModel):
    reason: Literal["voice-note", "dictate", "video", "live", "button"]


@app.post("/api/gpu/wake")
async def gpu_wake(req: WakeRequest, request: Request):
    """Wake the GPU worker: only on a visitor's explicit, checked action (the page never wakes it on
    its own). A no-op while it is awake or waking."""
    try:
        return JSONResponse(gpu.broker.wake(req.reason, client_ip(request.scope)), headers=NO_STORE)
    except gpu.WakeRefused as e:
        raise HTTPException(e.code, str(e)) from e


@app.get("/api/gpu/wanted")
def gpu_wanted(request: Request):
    """The team's laptop, standing by, asks whether it should connect (its token in the header)."""
    wanted = gpu.broker.standby(request.headers.get(gp.TOKEN_HEADER), request.headers.get("host", ""))
    if wanted is None:
        raise HTTPException(403, "Only this server's standby GPU worker may ask.")
    return JSONResponse({"wanted": wanted}, headers=NO_STORE)


@app.websocket("/ws/gpu")
async def gpu_worker(socket: WebSocket):
    """A GPU worker connects out to here (stutterclear/gpu_worker/); see stutterclear/app/gpu.py."""
    await gpu.broker.serve(socket)


class ContextRequest(BaseModel):
    situation: str  # where the person is about to speak
    current: str = ""  # what the context card says already


def _budget(calls: deque, per_min: int | None, busy: str):
    """Count one visitor call against a per-minute cap, or turn it away for a moment."""
    if not per_min:
        return
    now = time.monotonic()
    while calls and now - calls[0] > 60:
        calls.popleft()
    if len(calls) >= per_min:
        raise HTTPException(429, busy)
    calls.append(now)


def nemotron_budget():
    _budget(_nemotron_calls, NEMOTRON_PER_MIN, "Nemotron is busy with other visitors; try again in a moment.")


@app.post("/api/context")
def prepare_context(req: ContextRequest):
    """Nemotron (with Tavily search when a key is set) writes a context card for the situation."""
    from .. import context_agent
    if not nemotron.available():
        raise HTTPException(503, "The context agent needs NEBIUS_API_KEY set where the app runs.")
    nemotron_budget()
    try:
        return context_agent.fill(req.situation, req.current)
    except nemotron.NemotronError as e:
        raise HTTPException(502, str(e)) from e


class InsightsRequest(BaseModel):
    kinds: dict[str, int]  # stutters removed, by kind
    duration_s: float
    removed_s: float
    words: list[str] = []  # words the stutters were on
    transcript: str = ""


@app.post("/api/insights")
def session_insights(req: InsightsRequest):
    """Three short, supportive points about a session or voice note, from Nemotron (text only)."""
    from .. import insights
    if not nemotron.available():
        raise HTTPException(503, "Insights need NEBIUS_API_KEY set where the app runs.")
    nemotron_budget()
    try:
        return {"points": insights.summarize(req.kinds, req.duration_s, req.removed_s, req.words, req.transcript)}
    except nemotron.NemotronError as e:
        raise HTTPException(502, str(e)) from e


class GuardRequest(BaseModel):
    text: str


@app.post("/api/guard")
def guard_try(req: GuardRequest):
    """The meaning guard on typed text: which repeats Nemotron would cut and which it keeps."""
    from .. import guard
    if not nemotron.available():
        raise HTTPException(503, "The meaning guard needs NEBIUS_API_KEY set where the app runs.")
    if not req.text.split():
        raise HTTPException(400, "Type a sentence first.")
    nemotron_budget()
    try:
        return guard.check_text(req.text)
    except nemotron.NemotronError as e:
        raise HTTPException(502, str(e)) from e


class RescueRequest(BaseModel):
    said: str  # recent caption text
    context: str = ""  # the context card
    onset: str = ""  # the sound the person is stuck on, if known


@app.post("/api/rescue")
def word_rescue(req: RescueRequest):
    """Nemotron's guesses for the word the person is stuck on (opt-in; sends text, never audio)."""
    from .. import rescue
    if not rescue.available():
        raise HTTPException(503, "Word rescue needs NEBIUS_API_KEY set where the app runs.")
    nemotron_budget()
    try:
        return {"suggestions": rescue.suggest(req.said, req.context, req.onset), "model": nemotron.MODEL}
    except Exception as e:
        raise HTTPException(502, f"Nemotron did not answer: {e}") from e


class TTSRequest(BaseModel):
    text: str  # the word or short phrase the user clicked
    voice: str = ""  # a Magpie voice name; empty for the default


@app.post("/api/tts")
def say_it(req: TTSRequest):
    """Say it: NVIDIA Magpie TTS speaks what the user clicked, only ever on that click. The reply is
    a WAV; its X-Voice-Label header is the label the page shows (the voice is synthetic)."""
    from .. import tts
    if not tts.available():
        raise HTTPException(503, "Say it needs NVIDIA_API_KEY set where the app runs.")
    if not req.text.split():
        raise HTTPException(400, "Pick a word first.")
    if len(req.text) > tts.MAX_CHARS or (req.voice and not re.fullmatch(r"[\w.\-]{1,80}", req.voice)):
        raise HTTPException(400, f"Say it takes a word or a short phrase (up to {tts.MAX_CHARS} characters).")
    _budget(_tts_calls, TTS_PER_MIN, "Say it is busy with other visitors; try again in a moment.")
    try:
        y = tts.synthesize(req.text, req.voice or tts.VOICE)
    except tts.TTSError as e:
        raise HTTPException(502, str(e)) from e
    return Response(wav_bytes(y), media_type="audio/wav", headers={"Cache-Control": "no-store", "X-Voice-Label": tts.LABEL})


class InterviewRequest(BaseModel):
    role: str
    company: str = ""
    posting: str = ""  # the job post, optional
    card: str = ""  # the context card from /api/interview/start
    history: list[dict] = []  # [{question, answer}], answers as the captions wrote them


def _interview(fn):
    if not nemotron.available():
        raise HTTPException(503, "The interview partner needs NEBIUS_API_KEY set where the app runs.")
    nemotron_budget()
    try:
        return fn()
    except nemotron.NemotronError as e:
        raise HTTPException(502, str(e)) from e


@app.post("/api/interview/start")
def interview_start(req: InterviewRequest):
    """Practice: research the role into a context card, then the first question."""
    from .. import interview
    if not req.role.strip():
        raise HTTPException(400, "Say which role the interview is for.")
    return _interview(lambda: interview.start(req.role, req.company, req.posting))


@app.post("/api/interview/next")
def interview_next(req: InterviewRequest):
    """The next question, following up on the answers so far."""
    from .. import interview
    return _interview(lambda: {"question": interview.next_question(req.role, req.company, req.card, req.history)})


@app.post("/api/interview/summary")
def interview_summary(req: InterviewRequest):
    """Three practice points about the answers (content only, never how they were spoken)."""
    from .. import interview
    if not req.history:
        raise HTTPException(400, "Answer a question first.")
    return _interview(lambda: {"points": interview.summary(req.role, req.company, req.history)})


class FeedbackRequest(BaseModel):
    source: str = "note"  # note | live
    id: str = ""  # the voice note (or session) the cut belongs to
    cut: dict | None = None  # {start, end, kind, words}
    verdict: str | None = None  # accept (the cut was right) | reject (it should not have been cut)
    missed_at_s: float | None = None  # a stutter that was left in, tapped at this time


class FeedbackLog(BaseModel):
    entries: list[dict] = []


@app.post("/api/feedback")
def give_feedback(req: FeedbackRequest):
    """One review decision. The server works out which kind a missed stutter was (from the voice
    note's teacher probabilities); a personal computer logs it, a public server hands it back for
    the page to keep in the browser."""
    entry = req.model_dump()
    if req.missed_at_s is not None and req.id.startswith("note-"):
        note = notes.ready(req.id)
        if note is not None:
            entry["kind"] = feedback.kind_at(note["probs"], req.missed_at_s)
    entry = feedback.clean(entry)
    if entry is None:
        raise HTTPException(400, "Feedback is a reviewed cut (accept or reject) or a missed stutter's time.")
    if HOSTED or not feedback.stored_here():
        return {"entry": entry, "stored": False}
    return {"entry": entry, "stored": True, **feedback.record(entry)}


@app.get("/api/feedback")
def feedback_summary():
    """Multipliers per kind, agreement over the recent decisions, whether auto mode can be offered."""
    entries = [] if HOSTED else feedback.read()
    return {"stored": not HOSTED, **feedback.summary(entries)}


@app.post("/api/feedback/summary")
def feedback_summary_of(req: FeedbackLog):
    """The same summary for a log the page keeps itself (public servers store none)."""
    return feedback.summary([e for e in (feedback.clean(x) for x in req.entries[-2000:]) if e])


@app.delete("/api/feedback")
def forget_feedback():
    """Forget every review decision on this computer; StutterClear goes back to its defaults."""
    if not HOSTED:
        feedback.forget()
    return {"ok": True}


class SayRequest(BaseModel):
    text: str
    voice: str = ""
    page: str = ""  # on a public server, only the page that owns the session


@app.post("/api/say")
def say_in_call(req: SayRequest):
    """Say it in the call: NVIDIA Magpie TTS speaks the clicked word into the live session's output
    (what the listener hears, e.g. through a virtual cable), only ever on that click."""
    from .. import tts
    s = state["session"]
    if s is None:
        raise HTTPException(409, "Start a live session first.")
    if HOSTED and req.page != state["owner"]:
        raise HTTPException(403, "Only the page that started the session can speak into it.")
    if s.out_device is None:
        raise HTTPException(409, "Nothing is sending your voice to a call here, so Say it plays in this browser instead.")
    if not tts.available():
        raise HTTPException(503, "Say it needs NVIDIA_API_KEY set where the app runs.")
    if not req.text.split() or len(req.text) > tts.MAX_CHARS:
        raise HTTPException(400, f"Say it takes a word or a short phrase (up to {tts.MAX_CHARS} characters).")
    _budget(_tts_calls, TTS_PER_MIN, "Say it is busy with other visitors; try again in a moment.")
    try:
        y = tts.synthesize(req.text, req.voice or tts.VOICE)
    except tts.TTSError as e:
        raise HTTPException(502, str(e)) from e
    s.say(y)
    return {"ok": True, "seconds": round(len(y) / SR, 2), "label": tts.LABEL}


@app.get("/api/journal")
def journal_summary():
    """Progress over time: stutters per minute by day, this week against the last, hardest words."""
    return journal.summary([] if HOSTED else journal.read())


@app.post("/api/journal/start")
def journal_start():
    """Start keeping a journal on this computer (counts and words, never audio)."""
    if HOSTED:
        raise HTTPException(403, "The journal is kept only on your own computer; a public server keeps none.")
    journal.start()
    return journal.summary(journal.read())


@app.delete("/api/journal")
def journal_forget():
    """Forget the journal and stop keeping one."""
    if not HOSTED:
        journal.forget()
    return {"ok": True}


@app.post("/api/start")
def start(req: StartRequest):
    if HOSTED:
        raise HTTPException(403, "On the hosted demo, live mode uses your browser's microphone.")
    if state["claim"]:
        raise HTTPException(409, "A browser is starting a live session here. Try again in a moment.")
    if state["session"] is not None:
        state["session"].stop()
        state["session"] = None
    state["last"] = None
    devs = devices()
    in_device = resolve("microphone", req.in_device, req.in_name, devs["inputs"])
    out_device = resolve("output", req.out_device, req.out_name, devs["outputs"])
    notes.release_gpu()  # the live teacher gets the GPU back from a finished voice note
    s = LiveSession(req.mode, in_device, out_device, req.detector, save=req.save, adjust=feedback.current(),
                    sensitivity=req.sensitivity)
    try:
        s.start()
    except Exception as e:  # device busy or gone between listing and opening
        with suppress(Exception):
            s.stop()
        raise HTTPException(409, f"Could not open the audio device: {e}") from e
    state["session"] = s
    return {"ok": True, "detector": s.detector_name, "mode": req.mode}


@app.post("/api/sensitivity")
def set_sensitivity(req: SensitivityRequest):
    """The Live panel's sensitivity slider moved while a session runs on this computer (a browser-mic
    session changes it over its own connection)."""
    if HOSTED:
        raise HTTPException(403, "On the hosted demo, the page that started a session changes it over its own connection.")
    s = state["session"]
    return {"ok": True, "sensitivity": s.set_sensitivity(req.sensitivity) if s is not None else None}


@app.post("/api/stop")
def stop():
    if HOSTED:  # browser sessions end over their own connection; nobody else may end them
        raise HTTPException(403, "On the hosted demo, a session stops from the page that started it.")
    s, state["session"] = state["session"], None
    if s is None:
        return {"ok": True, "summary": None}
    summary = s.stop()
    state["last"], state["last_at"] = s.recording(), time.time()
    journal.record(journal.from_session(summary, s.raw.n / SR))  # only if the person started a journal
    return {"ok": True, "summary": summary, "playback": playback_info()}


def _last(page: str):
    """The finished session's audio, if this page may hear it (and it is not an hour old)."""
    _forget_last()
    return None if HOSTED and (not page or page != state["last_owner"]) else state["last"]


@app.get("/api/playback")
def playback_info(page: str = ""):
    last = _last(page)
    if not last or not len(last["original"]):
        return {"available": False}
    return {"available": True, **{f"{k}_s": round(len(v) / SR, 2) for k, v in last.items()}}


@app.get("/api/playback/{which}.wav")
def playback(which: str, page: str = ""):
    last = _last(page)
    if which not in ("original", "clean", "heard") or not last:
        raise HTTPException(404, "Nothing to play yet: record a session first.")
    return Response(wav_bytes(last[which]), media_type="audio/wav", headers={"Cache-Control": "no-store"})


def _teacher_remote() -> bool:
    """Whether a "Best (GPU)" session here would run on a GPU worker elsewhere."""
    from ..teacher_runtime import live_capable
    return not live_capable() and gpu.broker.live_offered()


@app.websocket("/ws/mic")
async def browser_mic(socket: WebSocket):
    """Live from the browser's mic: the first message is the settings (JSON), then 16 kHz mono
    float32 audio as binary messages, then the text "stop" (or just closing). The session shows up
    on /ws like any other; at the end the server replies with the summary and keeps the recording
    for the play buttons. One live session at a time; the server ends it itself after
    LIVE_LIMIT_S or MIC_IDLE_S without audio, and says why ("ended")."""
    client = client_ip(socket.scope)
    if not _claim_socket(client):
        await _too_many(socket)
        return
    try:
        await _browser_mic(socket, client)
    finally:
        _release_socket(client)


async def _too_many(socket: WebSocket):
    """Over the per-visitor socket cap: accepted only to say why (a refusal before accept reads as an outage)."""
    await socket.accept()
    await _close_quietly(socket, CLOSE_TOO_MANY, TOO_MANY_PAGES)


_claim_lock = threading.Lock()
_live_used: dict[str, deque] = {}  # visitor -> (when, seconds) of their live sessions in the last hour
_live_turn = {"client": None, "turned_away": 0.0}  # who holds the slot; when someone else was last turned away
_live_rest: dict[str, float] = {}  # visitor -> when they may start again (someone else goes first)


def _live_spent(client: str, now: float) -> float:
    used = _live_used.get(client)
    while used and now - used[0][0] > 3600:
        used.popleft()
    if not used:
        _live_used.pop(client, None)
        return 0.0
    return sum(s for _, s in used)


async def _browser_mic(socket: WebSocket, client: str):
    now = time.monotonic()
    if LIVE_PER_HOUR_S and (_live_spent(client, now) >= LIVE_PER_HOUR_S or _live_rest.get(client, 0.0) > now):
        await socket.accept()  # the one live slot is shared: every visitor gets a turn
        await _close_quietly(socket, 4409, LIVE_SPENT if _live_spent(client, now) >= LIVE_PER_HOUR_S else LIVE_YIELD)
        return
    with _claim_lock:  # before any await: of two pages starting at once, exactly one gets the slot
        taken = state["session"] is not None or state["claim"]
        if not taken:
            state["claim"] = True
            _live_turn["client"] = client
        elif client != _live_turn["client"]:
            _live_turn["turned_away"] = now
    if taken:
        await socket.accept()
        await socket.close(code=4409, reason=BUSY_HOSTED if HOSTED else "A live session is already running here.")
        return
    try:
        await _live_session(socket, client)
    finally:
        if LIVE_PER_HOUR_S:  # the time it held the slot, settings included
            end = time.monotonic()
            _live_used.setdefault(client, deque()).append((end, end - now))
            if _live_turn["turned_away"] > now:  # someone else waited meanwhile: they go next
                _live_rest[client] = end + LIVE_TURN_S


async def _live_session(socket: WebSocket, client: str):
    try:
        s, page = await _open_browser_session(socket, client)
    finally:
        state["claim"] = False
    if s is None:
        return
    loop = asyncio.get_running_loop()
    t0, ended, close, samples = loop.time(), None, None, 0
    try:
        while True:
            try:
                msg = await asyncio.wait_for(socket.receive(), timeout=MIC_IDLE_S)
            except asyncio.TimeoutError:
                ended = "No audio arrived for a while, so the session ended."
                break
            if msg["type"] == "websocket.disconnect" or msg.get("text") == "stop":
                break
            if LIVE_LIMIT_S and loop.time() - t0 > LIVE_LIMIT_S:
                ended = f"Live sessions here last up to {LIVE_LIMIT_S / 60:.0f} minutes, so others can try it too."
                break
            data = msg.get("bytes")
            if not data:
                _retune(s, msg.get("text"))
                continue
            if len(data) > FRAME_MAX:
                ended = "Audio frames must be at most half a second."
                close = 1009
                break
            samples += len(data) // 4
            if samples > (loop.time() - t0 + AHEAD_S) * SR:
                ended = "Audio arrived faster than real time."
                close = 1008
                break
            x = np.frombuffer(data[:len(data) // 4 * 4], np.float32)
            await loop.run_in_executor(None, s.feed, x)
    except WebSocketDisconnect:
        pass
    finally:
        summary = None
        if state["session"] is s:
            state["session"] = None
            summary = await loop.run_in_executor(None, s.stop)
            state["last"], state["last_owner"], state["last_at"] = s.recording(), page, time.time()
            if not HOSTED:
                journal.record(journal.from_session(summary, s.raw.n / SR))
        try:  # the page may already be gone
            await socket.send_json({"summary": summary, "playback": playback_info(page), "ended": ended})
            await socket.close(code=close or 1000, reason=ended if close else "")
        except Exception:
            pass


def _retune(s: LiveSession, text: str | None):
    """A settings change from the page mid-session, {"sensitivity": 0-100}; anything else is ignored.
    No reply: the page reads the next message as the session's end."""
    if not text or len(text) > 200:
        return
    try:
        cfg = json.loads(text)
    except ValueError:
        return
    if isinstance(cfg, dict) and "sensitivity" in cfg:
        s.set_sensitivity(cfg["sensitivity"])


async def _open_browser_session(socket: WebSocket, client: str):
    """Accept, read the settings, build the session. (session, page), or (None, None) once the page
    has been told why not."""
    await socket.accept()
    try:
        msg = await asyncio.wait_for(socket.receive(), SETTINGS_S)
        cfg = json.loads(msg.get("text") or "")
        if not isinstance(cfg, dict) or cfg.get("mode", "gentle") not in live.MODES:
            raise ValueError
    except asyncio.TimeoutError:
        await _close_quietly(socket, 1008, "No settings arrived.")
        return None, None
    except (ValueError, TypeError, KeyError):
        await _close_quietly(socket, 1008, "The settings could not be read.")
        return None, None
    page = str(cfg.get("page", ""))[:64]
    detector = str(cfg.get("detector", "heuristic"))
    if detector == "teacher" and _teacher_remote():
        reason = gpu.broker.admit_live(client)
        if reason:
            with suppress(Exception):
                await socket.send_json({"ok": False, "error": reason})
            await _close_quietly(socket, 1000, "")
            return None, None
    # The person's feedback: this computer's log, or on a public server what the page kept for them
    adjust = feedback.from_page(cfg.get("adjust")) if HOSTED else feedback.current()
    notes.release_gpu()
    loop = asyncio.get_running_loop()
    s = None
    try:  # may load a model: off the event loop. A detector on a GPU worker is closed if this fails.
        s = await loop.run_in_executor(None, lambda: LiveSession(cfg.get("mode", "gentle"), detector=detector,
                                                                 source="browser", adjust=adjust,
                                                                 sensitivity=cfg.get("sensitivity", live.SENSITIVITY)))
        s.start()
        await socket.send_json({"ok": True, "detector": s.detector_name, "limit_s": LIVE_LIMIT_S,
                                "delay_ms": round(s.pipe.cfg.target_delay_ms)})
    except Exception as e:
        print(f"live: could not start a browser session ({type(e).__name__})", flush=True)
        if s is not None:
            with suppress(Exception):
                await loop.run_in_executor(None, s.stop)
        await _close_quietly(socket, 1011, "Could not start the session.")
        return None, None
    state["session"], state["last"], state["owner"] = s, None, page
    return s, page


async def _close_quietly(socket: WebSocket, code: int, reason: str):
    with suppress(Exception):
        await socket.close(code=code, reason=reason)


@app.websocket("/ws")
async def ws(socket: WebSocket, page: str = ""):
    client = client_ip(socket.scope)
    if page and not PAGE_RE.fullmatch(page):
        await socket.close(code=1008)  # before accept: HTTP 403
        return
    if not _claim_socket(client):
        await _too_many(socket)
        return
    try:
        await socket.accept()
        clients[socket] = page
        loop = asyncio.get_running_loop()
        until = loop.time() + PAGE_MAX_S
        while True:  # the page never sends anything; this waits for it to go away
            try:
                msg = await asyncio.wait_for(socket.receive(), until - loop.time() if HOSTED else None)
            except asyncio.TimeoutError:  # a public server: a page left open must not hold the server up
                if state["session"] is not None and page and page == state["owner"]:
                    until = loop.time() + 60  # not in the middle of its own session
                    continue
                await socket.close(code=4000, reason="rest")
                break
            if msg["type"] == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    finally:
        clients.pop(socket, None)
        _sent.pop(socket, None)
        _release_socket(client)


@app.get("/api/demo")
def demo_list():
    samples = demo.available()
    return {"samples": samples, "credit": demo.credit(samples)}


def _demo(sid: str, engine: str) -> dict:
    if sid.startswith("note-"):  # a cleaned voice note plays through the same player
        if notes.ready(sid) is None or engine not in ("natural", "thorough", "live"):
            raise HTTPException(404, "No such voice note (they are kept for an hour).")
        return demo.render(sid, engine)
    s = next((s for s in demo.available() if s["id"] == sid), None)
    if s is None or engine not in s["engines"]:
        raise HTTPException(404, "No such sample.")
    return demo.render(sid, engine)


async def _body(request: Request, limit: int, too_large: str) -> bytearray:
    """The request body, refused (413) as soon as it passes `limit` bytes instead of after it all
    arrived in memory, and held once (its size is known ahead: no second copy to join its chunks)."""
    try:
        size = int(request.headers.get("content-length") or -1)
    except ValueError:
        size = -1
    if size > limit:
        raise HTTPException(413, too_large)
    buf, n = bytearray(max(size, 0)), 0
    async for chunk in request.stream():
        end = n + len(chunk)
        if end > limit:
            raise HTTPException(413, too_large)
        buf[n:end] = chunk  # in place within the size it said (past it, the buffer grows)
        n = end
    if n < len(buf):
        raise HTTPException(400, "The upload was cut short. Please try again.")
    return buf


def _gpu_client(request: Request) -> str:
    """The visitor's address when their recording goes to a GPU worker (its fair share needs it);
    otherwise nothing is kept."""
    return client_ip(request.scope) if notes.remote() else ""


def _gpu_room(client: str):
    """Before an upload: the GPU worker can take it now (503), and this visitor may send one (429)."""
    if not notes.remote():
        return
    reason = gpu.broker.accepting(client)
    if reason:
        raise HTTPException(503, reason)
    reason = gpu.broker.admit(client, 0, dry_run=True)
    if reason:
        raise HTTPException(429, reason)


@app.post("/api/notes")
async def create_note(request: Request):
    """Body: a WAV file (the page decodes recordings and uploads to WAV in the browser)."""
    if not notes.available():
        raise HTTPException(503, "Voice notes need the teacher model, which is not downloaded.")
    client = _gpu_client(request)
    try:
        notes.room(client)  # turned away before the upload is read
        data = await _body(request, notes.MAX_BYTES, f"Recordings can be up to {notes.MAX_S // 60} minutes long.")
        # decoding stays off the event loop (it also carries the GPU worker's socket)
        return {"id": await run_in_threadpool(notes.submit, data, client)}
    except notes.NoteUnavailable as e:
        raise HTTPException(503, str(e)) from e
    except notes.NoteBusy as e:
        raise HTTPException(429, str(e)) from e
    except notes.NoteError as e:
        raise HTTPException(415, str(e)) from e


def _video_ready():
    if not video.available():
        raise HTTPException(503, "Cleaning a video needs the offline model and ffmpeg (pip install imageio-ffmpeg).")


@app.post("/api/video/upload")
def video_upload(request: Request):
    """Start uploading a video in parts (PUT /api/video/upload/{upload}?i=0, 1, ...), for servers that
    take at most 32 MiB a request. The GPU worker's room is checked before anything is sent."""
    _video_ready()
    _gpu_room(_gpu_client(request))
    try:
        return video.begin_upload(client_ip(request.scope))
    except notes.NoteBusy as e:
        raise HTTPException(429, str(e)) from e


@app.put("/api/video/upload/{upload}")
async def video_upload_part(upload: str, i: int, request: Request):
    try:
        await run_in_threadpool(video.check_part, upload, i)  # before the part is read
        data = await _body(request, video.PART_BYTES, "That part is too large.")
        return {"received": await run_in_threadpool(video.add_part, upload, i, data)}
    except video.UploadError as e:
        raise HTTPException(e.code, str(e)) from e


@app.delete("/api/video/upload/{upload}")
def video_upload_cancel(upload: str):
    """The page gave up on an upload: its place is free for the next visitor at once."""
    video.cancel_upload(upload)
    return {"ok": True}


@app.post("/api/video")
async def create_video(request: Request, engine: str = "natural", name: str = "video.mp4", upload: str = "",
                       ext: str = ".mp4"):
    """A video to clean: its uploaded parts (upload=, ext=), or the whole file as the body (name=). Its
    stutters are cut from picture and sound together (the voice-note engines on its soundtrack,
    ffmpeg for the cuts)."""
    _video_ready()
    client = _gpu_client(request)
    try:
        if upload:
            jid = await run_in_threadpool(video.submit_upload, upload, ext, engine, client)
        else:
            _gpu_room(client)
            await run_in_threadpool(video.room)  # before the video is read
            data = await _body(request, _body_limit("POST", "/api/video"),
                               f"That video is too large ({video.MAX_BYTES // 10**6} MB at most).")
            jid = await run_in_threadpool(video.submit, data, name, engine, client)
        return {"id": jid}
    except video.UploadError as e:
        raise HTTPException(e.code, str(e)) from e
    except notes.NoteBusy as e:
        raise HTTPException(429, str(e)) from e
    except video.VideoError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/video/{jid}")
def video_status(jid: str):
    s = video.status(jid)
    if s is None:
        raise HTTPException(404, "No such video (they are kept for an hour).")
    return JSONResponse(s, headers=PRIVATE)  # its transcript


RANGE_MAX = 16 * 1024 * 1024  # bytes in one partial answer (Cloud Run: 32 MiB a response unless chunked)
_RANGE = re.compile(r"bytes=(\d*)-(\d*)")


@app.get("/api/video/{jid}/clean.mp4")
def video_result(jid: str, request: Request, download: bool = False):
    """The cleaned video. A Range request gets at most RANGE_MAX bytes (the player asks for the rest);
    a whole-file request is streamed in chunks."""
    data = video.result(jid)
    if data is None:
        raise HTTPException(404, "That video is not ready (or was forgotten after an hour).")
    n = len(data)
    headers = {"Cache-Control": "no-store", "Accept-Ranges": "bytes"}
    if download:
        headers["Content-Disposition"] = 'attachment; filename="stutterclear-video.mp4"'
    m = _RANGE.fullmatch((request.headers.get("range") or "").strip())
    if m and (m[1] or m[2]):  # one range (several, or anything else, get the whole file)
        start, end = (int(m[1]), int(m[2]) if m[2] else n - 1) if m[1] else (max(0, n - int(m[2])), n - 1)
        if start >= n or end < start:
            raise HTTPException(416, "That part of the video does not exist.", headers={"Content-Range": f"bytes */{n}"})
        end = min(end, n - 1, start + RANGE_MAX - 1)
        headers["Content-Range"] = f"bytes {start}-{end}/{n}"
        return Response(memoryview(data)[start:end + 1], status_code=206, media_type="video/mp4", headers=headers)
    if n <= RANGE_MAX:
        return Response(data, media_type="video/mp4", headers=headers)
    return StreamingResponse((data[i:i + 2**20] for i in range(0, n, 2**20)), media_type="video/mp4", headers=headers)


@app.get("/api/notes/{nid}")
def note_status(nid: str):
    notes.touch(nid)  # the page still wants it
    s = notes.status(nid)
    if s is None:
        raise HTTPException(404, "No such voice note (they are kept for an hour).")
    return JSONResponse(s, headers=NO_STORE)


@app.get("/api/demo/{sid}")
def demo_timeline(sid: str, engine: str = "natural", captions: bool = True):
    timeline = _demo(sid, engine)["timeline"]
    if captions:  # word-timed captions of both tracks (Dictate never shows them: captions=0)
        timeline = {**timeline, "captions": demo.captions(sid, engine)}
    return JSONResponse(timeline, headers=PRIVATE) if sid.startswith("note-") else timeline


@app.get("/api/demo/{sid}/cuts.{fmt}")
def demo_cuts(sid: str, fmt: str, engine: str = "natural"):
    """Timestamps of everything removed, as a table (csv) or subtitles (srt), on the original
    recording's clock, for editing or checking a recording elsewhere."""
    if fmt not in ("csv", "srt"):
        raise HTTPException(404, "Timestamps come as cuts.csv or cuts.srt.")
    cuts = _demo(sid, engine)["timeline"]["cuts"]
    body = demo.cuts_csv(cuts) if fmt == "csv" else demo.cuts_srt(cuts)
    return Response(body, media_type="text/csv" if fmt == "csv" else "application/x-subrip",
                    headers={"Content-Disposition": f'attachment; filename="{sid}-{engine}-cuts.{fmt}"',
                             **(PRIVATE if sid.startswith("note-") else {})})


@app.get("/api/demo/{sid}/{which}.wav")
def demo_audio(sid: str, which: str, engine: str = "natural"):
    if which not in ("original", "clean", "heard"):
        raise HTTPException(404, "No such track.")
    r = _demo(sid, engine)
    if which not in r:  # "heard" (a call's version, with its pauses) is the live engine's only
        raise HTTPException(404, "No such track.")
    # A visitor's own voice is never kept by a browser or a proxy; the sample clips may be
    private = sid.startswith("note-")
    return Response(wav_bytes(r[which]), media_type="audio/wav",
                    headers=PRIVATE if private else {"Cache-Control": "public, max-age=3600"})


# Last, so the API and WebSocket routes above take precedence over the built UI files.
app.mount("/", UI(directory=STATIC, html=True), name="ui")
