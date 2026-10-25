"""The wire contract between the app server (stutterclear/app/gpu.py) and a GPU worker
(stutterclear/gpu_worker/): the messages and their fields, the frame codec, how audio and teacher
probabilities travel, worker tokens, the launcher interface, and the limits and timings both sides
share.

The worker connects out to wss://<server>/ws/gpu with its token in the X-StutterClear-Worker header,
so nothing has to be opened where it runs (a Kaggle notebook, the team's laptop). Every message, in
both directions, is one binary frame: a 4-byte big-endian length N, N bytes of UTF-8 JSON (the
header, whose "t" names the message), then the payload: 16 kHz mono 16-bit PCM audio, float32
teacher probabilities, or nothing. decode() checks everything a receiver relies on (a known type,
its fields and their types, the payload's size), so a malformed message from either side is a
ProtocolError and never reaches the GPU or the cut logic. encode() checks the same, so a bug is
caught on the side that made it.

Standard library and numpy only, and importing it has no side effects: the server imports it on
every start, the worker before torch. Read the timings as gpu_protocol.NAME when they are used,
never copied at import, so a test can shorten them for both sides at once.
"""
import base64
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import struct
import urllib.parse
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np

PROTO = 1  # HELLO and WELCOME keep their fields in every version, so a mismatch is always recognised
PATH = "/ws/gpu"  # the worker's WebSocket on the app server
WANTED_PATH = "/api/gpu/wanted"  # a standby laptop asks here whether the GPU is wanted
TOKEN_HEADER = "X-StutterClear-Worker"  # the token travels only in this header: servers log URLs

SR = 16000
FRAME = 320  # teacher frames are 20 ms
N_CLASSES = 5  # sound_rep, word_rep, prolongation, block, interjection (student_runtime.CLASSES order)

# Sizes
MAX_FRAME = 4 * 1024 * 1024  # the largest message either side sends or accepts
MAX_HEADER = 256 * 1024
CHUNK_BYTES = 64 * 1024  # a recording goes up in parts this big, so a live window never waits long behind it
MAX_JOB_S = 180  # the longest recording a job carries (notes.MAX_S)
MAX_WORDS = 3000  # far more words than 3 minutes of speech hold
MAX_TEXT = 200  # any text field: an error message, a label
MAX_WHERE = 40  # where the worker runs, as the page shows it

# Timings, in seconds
HEARTBEAT_S = 5.0  # worker heartbeats and server pings
DEAD_AFTER_S = 20.0  # nothing heard for this long: the connection is dead
HELLO_S = 10.0  # a worker says hello this soon after connecting, or is closed
IDLE_EXIT_S = 600.0  # a woken worker sleeps again this long after its last real work...
IDLE_UNUSED_S = 240.0  # ...or this long after it was ready, if no work came at all
DRAIN_S = 600.0  # no new work this close to a worker's deadline (its machine's hard time limit)
ROTATE_AFTER_S = 3000.0  # Cloud Run ends WebSockets at 60 min: an idle worker reconnects at 50
RECONNECT_S = (1.0, 2.0, 4.0, 8.0, 15.0)  # waits after a lost connection (each +-20%), then 15 s
STANDBY_POLL_S = 15.0  # a standby laptop asks whether it is wanted this often
MAX_TOKEN_S = 48 * 3600  # no token is valid for longer than this, however it was signed

# Live: the teacher runs on the worker, the cut logic on the server (stutterclear/remote_detector.py)
LIVE_HOP_MS = 160
LIVE_QUEUE_MAX = 32  # live windows waiting to be sent (about 5 s): more means the worker is too slow
LIVE_DEADLINE_SLACK_MS = 250  # a reply this much later than the delay allows counts as missed
LIVE_FIRST_REPLY_S = 5.0  # no first reply this soon after opening: not usable for live
LIVE_READY_S = 15.0  # a stream's model warm (LIVE_READY) this soon after LIVE_OPEN, or the session goes without it
# (on a shared GPU a recording may have let the live teacher go, and it loads again from disk first)
LIVE_IDLE_S = 30.0  # the worker drops a live stream that sent no audio for this long
LIVE_MAX_EXTRA_MS = 600  # a worker further away than this is not offered for live use...
LIVE_MAX_EXTRA_HOSTED_MS = 1500  # ...on a public server more is fine: the cleaned voice plays after the session
LIVE_MISS_RUN_MS = 1000  # fall back to the rule-based detector after this long of missed replies in a row...
LIVE_MISS_SHARE = 0.25  # ...or once this share of the last LIVE_MISS_WINDOW replies missed
LIVE_MISS_WINDOW = 50
LOCAL_WINDOW_MS = 60  # GPU time per window that the local teacher's delay already allows (live.delay_overrides)

ROLES = ("kaggle", "laptop", "local")  # who runs a worker, as its token says
REASONS = ("voice-note", "dictate", "video", "live", "button", "job")  # why the GPU was woken
MODES = ("gentle", "strong")
STAGES = ("listening", "transcribing")
SHUTDOWN_REASONS = ("idle", "budget", "error")
BYE_REASONS = ("idle", "deadline", "shutdown", "error")
LAUNCH_STATES = ("queued", "running", "done", "failed", "quota")

# WebSocket close codes. A bad, expired or used token, or a server without the GPU feature, is
# refused before the socket is accepted: the worker sees HTTP 403 and exits without retrying.
CLOSE_ROTATE = 1000  # with the reason "rotate": the worker reconnects at once
CLOSE_BAD = 1008  # a malformed message: reconnect with backoff
CLOSE_RESTART = 1012  # the server is restarting: reconnect (the next instance adopts the worker)
CLOSE_SHUTDOWN = 4000  # the server said shutdown and the worker said bye
CLOSE_BUSY = 4409  # another worker is active: a Kaggle or local worker exits, a laptop goes back to standby
CLOSE_VERSION = 4426  # a different protocol, or (Kaggle, local) different code from the server's: exit

# Message types. Server -> worker:
WELCOME, PING, JOB, JOB_AUDIO, CANCEL, SHUTDOWN = "welcome", "ping", "job", "job_audio", "cancel", "shutdown"
LIVE_OPEN, LIVE_AUDIO, LIVE_CLOSE = "live_open", "live_audio", "live_close"
# Worker -> server:
HELLO, CAPS, HEARTBEAT, PONG, PROGRESS = "hello", "caps", "heartbeat", "pong", "progress"
RESULT, ERROR, BYE = "result", "error", "bye"
LIVE_READY, LIVE_PROBS, LIVE_ERROR = "live_ready", "live_probs", "live_error"

# Field kinds: every number in this protocol is non-negative; "?" also allows null.
INT, NUM, STR, BOOL, LIST, DICT, OPT_NUM, OPT_STR = "int", "num", "str", "bool", "list", "dict", "num?", "str?"

FIELDS: dict[str, dict[str, str]] = {
    # proto: PROTO; commit: the server's code; idle_*: when to sleep again; now: the server's clock (epoch s)
    WELCOME: {"proto": INT, "commit": STR, "idle_exit_s": NUM, "idle_unused_s": NUM, "rotate_after_s": NUM,
              "now": NUM},
    PING: {"n": INT, "ts": NUM},  # ts: the server's monotonic clock, echoed in PONG (round-trip time)
    # One recording: `parts` JOB_AUDIO messages follow, in order, with its pcm16 audio
    JOB: {"job": STR, "kind": STR, "samples": INT, "parts": INT, "enc": STR, "sr": INT, "language": STR},
    JOB_AUDIO: {"job": STR, "i": INT},  # payload: part i, at most CHUNK_BYTES of pcm16
    CANCEL: {"job": STR},  # drop the job and any result
    SHUTDOWN: {"reason": STR},  # say bye and leave
    # A live stream: window = context + lookahead in samples; hop_frames = frames decided per request
    LIVE_OPEN: {"stream": STR, "mode": STR, "lookahead_ms": NUM, "window": INT, "hop_frames": INT},
    # Decide frames [f0, f1) from the window of samples [start, end); payload: pcm16 of samples
    # [end - n, end), the ones not sent before (n = payload bytes / 2)
    LIVE_AUDIO: {"stream": STR, "seq": INT, "start": INT, "end": INT, "f0": INT, "f1": INT},
    LIVE_CLOSE: {"stream": STR},
    # worker: 8 hex per process; wake and role: its token's; where: a label for the page; deadline:
    # when its machine stops (epoch s; null on the laptop); gpus: [{name, mem_gb}]
    HELLO: {"proto": INT, "worker": STR, "wake": STR, "role": STR, "where": STR, "commit": STR, "gpus": LIST,
            "deadline": OPT_NUM, "loading": BOOL},
    # Sent again as engines become ready. window_ms: p90 of a warm live window; models: {live,
    # recordings} checkpoint names; error: plain text when loading failed
    CAPS: {"notes": BOOL, "live": BOOL, "loading": BOOL, "window_ms": OPT_NUM, "models": DICT, "error": OPT_STR},
    HEARTBEAT: {"jobs": LIST, "live": BOOL, "window_ms": OPT_NUM, "gpu_mem_gb": LIST},
    PONG: {"n": INT, "ts": NUM},
    PROGRESS: {"job": STR, "stage": STR},
    # payload: float32 probabilities [frames, N_CLASSES]; words: [{w, s, e}] (check_words); slow:
    # the words were written on the CPU; timing: {teacher_s, asr_s}
    RESULT: {"job": STR, "frames": INT, "words": LIST, "slow": BOOL, "timing": DICT},
    ERROR: {"job": STR, "message": STR, "retry": BOOL},  # retry: worth sending the job again
    BYE: {"reason": STR},
    LIVE_READY: {"stream": STR},
    LIVE_PROBS: {"stream": STR, "seq": INT, "f0": INT, "f1": INT, "ms": NUM},  # payload: float32 [f1 - f0, N_CLASSES]
    LIVE_ERROR: {"stream": STR, "message": STR},
}
FROM_SERVER = frozenset({WELCOME, PING, JOB, JOB_AUDIO, CANCEL, SHUTDOWN, LIVE_OPEN, LIVE_AUDIO, LIVE_CLOSE})
FROM_WORKER = frozenset(FIELDS) - FROM_SERVER
CHOICES = {(HELLO, "role"): ROLES, (JOB, "kind"): ("note",), (JOB, "enc"): ("pcm16",), (JOB, "sr"): (SR,),
           (LIVE_OPEN, "mode"): MODES, (PROGRESS, "stage"): STAGES, (SHUTDOWN, "reason"): SHUTDOWN_REASONS,
           (BYE, "reason"): BYE_REASONS}


class ProtocolError(ValueError):
    """A message that breaks this contract: the receiver closes the connection (CLOSE_BAD)."""


# ---- the frame codec ----------------------------------------------------------------------------

_BIG = 2 ** 53  # numbers stay exact as floats; NaN and infinity fail the comparison too


def _is(kind: str, v) -> bool:
    if v is None:
        return kind in (OPT_NUM, OPT_STR)
    if kind == INT:
        return type(v) is int and 0 <= v <= _BIG  # type(): not bool, which is an int to Python
    if kind in (NUM, OPT_NUM):
        return type(v) in (int, float) and 0 <= v <= _BIG
    if kind in (STR, OPT_STR):
        return type(v) is str and len(v) <= MAX_TEXT
    return type(v) is {BOOL: bool, LIST: list, DICT: dict}[kind]


def _payload_ok(t: str, h: dict, n: int) -> bool:
    if t == JOB_AUDIO:
        return 0 < n <= CHUNK_BYTES and n % 2 == 0
    if t == LIVE_AUDIO:
        return 0 < n and n % 2 == 0 and n // 2 <= h["end"] - h["start"]
    if t == RESULT:
        return n == h["frames"] * N_CLASSES * 4
    if t == LIVE_PROBS:
        return n == (h["f1"] - h["f0"]) * N_CLASSES * 4
    return n == 0


def _check(h, n: int):
    if type(h) is not dict:
        raise ProtocolError("the header is not a JSON object")
    t = h.get("t")
    if type(t) is not str or t not in FIELDS:  # a list or an object here must not reach a dict lookup
        raise ProtocolError(f"unknown message type {printable(t, 40)!r}")
    for name, kind in FIELDS[t].items():
        if name not in h:
            raise ProtocolError(f"{t}: no {name}")
        if not _is(kind, h[name]):
            raise ProtocolError(f"{t}: bad {name}")
        if (t, name) in CHOICES and h[name] not in CHOICES[(t, name)]:
            raise ProtocolError(f"{t}: unknown {name}")
    if t == JOB and not (0 < h["samples"] <= MAX_JOB_S * SR and h["parts"] == n_parts(h["samples"])):
        raise ProtocolError("job: bad samples or parts")
    if (t in (LIVE_AUDIO, LIVE_PROBS) and h["f0"] > h["f1"]) or (t == LIVE_AUDIO and h["start"] > h["end"]):
        raise ProtocolError(f"{t}: bad range")
    if t == RESULT and (h["frames"] > frames_for(MAX_JOB_S * SR) or len(h["words"]) > MAX_WORDS):
        raise ProtocolError("result: too many frames or words")
    if not _payload_ok(t, h, n):
        raise ProtocolError(f"{t}: wrong payload size")


def encode(header: dict, payload: bytes = b"") -> bytes:
    """One message as a frame. Raises ProtocolError for anything decode() would refuse."""
    _check(header, len(payload))
    try:
        h = json.dumps(header, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    except (TypeError, ValueError) as e:  # numpy numbers, NaN
        raise ProtocolError(f"the header is not plain JSON: {e}") from None
    if len(h) > MAX_HEADER:
        raise ProtocolError("header too large")
    if 4 + len(h) + len(payload) > MAX_FRAME:
        raise ProtocolError("frame too large")
    return b"".join((struct.pack(">I", len(h)), h, payload))


def _no_constant(name: str):
    raise ValueError(f"{name} is not a number here")


def decode(frame: bytes | bytearray | memoryview, expect: frozenset | None = None) -> tuple[dict, bytes]:
    """(header, payload) of one frame, fully checked. `expect`: the types this side may receive
    (FROM_WORKER on the server, FROM_SERVER on the worker)."""
    if len(frame) > MAX_FRAME:
        raise ProtocolError("frame too large")
    if len(frame) < 4:
        raise ProtocolError("frame too short")
    (n,) = struct.unpack_from(">I", frame)
    if n > MAX_HEADER or 4 + n > len(frame):
        raise ProtocolError("bad header length")
    view = memoryview(frame)
    try:
        header = json.loads(bytes(view[4:4 + n]).decode("utf-8"), parse_constant=_no_constant)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise ProtocolError("the header is not JSON") from None
    payload = bytes(view[4 + n:])
    t = header.get("t") if type(header) is dict else None
    if expect is not None and type(t) is str and t in FIELDS and t not in expect:
        raise ProtocolError(f"{t} is not sent this way")
    _check(header, len(payload))
    return header, payload


# ---- audio and probabilities --------------------------------------------------------------------

def pcm16(x: np.ndarray) -> bytes:
    """Float audio in [-1, 1] as 16-bit little-endian PCM: clipped, and non-finite samples become
    silence. Audio decoded from a 16-bit WAV (exactly k/32768) goes through unchanged."""
    x = np.nan_to_num(np.asarray(x, np.float32).ravel(), nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(np.round(x * 32768.0), -32768, 32767).astype("<i2").tobytes()


def from_pcm16(b: bytes) -> np.ndarray:
    """pcm16 bytes back to float32 audio (a new, writable array)."""
    if len(b) % 2:
        raise ProtocolError("an odd number of audio bytes")
    return np.frombuffer(b, "<i2").astype(np.float32) / 32768.0


def n_parts(samples: int, size: int = CHUNK_BYTES) -> int:
    """How many JOB_AUDIO parts a recording of this many samples takes."""
    return -(-2 * samples // size)


def job_parts(x: np.ndarray, size: int = CHUNK_BYTES) -> Iterator[bytes]:
    """A recording as pcm16 parts of `size` bytes (the last one shorter), encoded one at a time, so a
    3-minute note is never copied whole."""
    if size <= 0 or size % 2:
        raise ValueError("the part size must be even")
    step = size // 2
    for i in range(0, len(x), step):
        yield pcm16(x[i:i + step])


def frames_for(samples: int) -> int:
    """Teacher frames in a recording of this many samples, as notes.teacher_probs counts them."""
    return max(0, (samples - 400) // FRAME + 1)


def probs_bytes(p: np.ndarray) -> bytes:
    """Teacher probabilities [frames, N_CLASSES] as float32 little-endian bytes."""
    p = np.asarray(p)
    if p.ndim != 2 or p.shape[1] != N_CLASSES:
        raise ProtocolError(f"probabilities must be [frames, {N_CLASSES}]")
    return np.ascontiguousarray(p, "<f4").tobytes()


def probs_from(b: bytes, frames: int) -> np.ndarray:
    """Probabilities back from bytes: a writable float32 [frames, N_CLASSES], every value in [0, 1]."""
    if len(b) != frames * N_CLASSES * 4:
        raise ProtocolError("probabilities of the wrong size")
    p = np.frombuffer(b, "<f4").reshape(frames, N_CLASSES).astype(np.float32)
    if p.size and not (np.isfinite(p).all() and p.min() >= 0.0 and p.max() <= 1.0):
        raise ProtocolError("probabilities outside [0, 1]")
    return p


def window_rows(probs: np.ndarray, w0: int, f0: int, f1: int) -> np.ndarray:
    """The rows for frames [f0, f1) of a window's probabilities whose first row is frame w0: frames
    outside the window get zeros ("no stutter"), as TeacherDetector decides them."""
    out = np.zeros((max(0, f1 - f0), N_CLASSES), np.float32)
    lo, hi = max(f0, w0), min(f1, w0 + len(probs))
    if hi > lo:
        out[lo - f0:hi - f0] = probs[lo - w0:hi - w0]
    return out


def check_words(words, duration_s: float) -> list[dict]:
    """A result's words, [{"w": text, "s": start s, "e": end s}], as CrisperWhisper gave them. Only
    what is broken is refused (wrong types, non-finite or impossible times, too many words); the
    words themselves pass unchanged, as they do on the local path."""
    if type(words) is not list or len(words) > MAX_WORDS:
        raise ProtocolError("words: not a list, or too many")
    out = []
    for w in words:
        if type(w) is not dict or type(w.get("w")) is not str or len(w["w"]) > 100:
            raise ProtocolError("words: a bad word")
        s, e = w.get("s"), w.get("e")
        if not all(type(v) in (int, float) and -1.0 <= v <= duration_s + 5.0 for v in (s, e)):  # NaN fails too
            raise ProtocolError("words: a bad time")
        out.append({"w": w["w"], "s": float(s), "e": float(e)})
    return out


def live_extra_delay_ms(rtt_ms: float, window_ms: float, max_ms: float = LIVE_MAX_EXTRA_MS) -> int | None:
    """Delay a remote teacher adds over the local one: the round trip, plus GPU time beyond what the
    local delay allows, plus a margin. None when that is too far for live use (over max_ms)."""
    raw = rtt_ms + max(0.0, window_ms - LOCAL_WINDOW_MS) + 40.0
    if not math.isfinite(raw) or raw > max_ms:
        return None
    return int(round(max(100.0, raw)))


def first_engine(reason: str) -> str:
    """Which engine a woken worker loads first: the live teacher for a live session, else the
    recordings engines. A fixed word, never the visitor's text, goes on the worker's command line."""
    return "live" if reason == "live" else "notes"


def printable(text, limit: int = MAX_TEXT) -> str:
    """Text from the other side, safe for a log line or the page: printable characters only, spaces
    collapsed, at most `limit` characters."""
    s = "".join(c if c.isprintable() else " " for c in str(text))
    return " ".join(s.split())[:limit].rstrip()


# ---- worker tokens ------------------------------------------------------------------------------
# sc1.<role>.<wake>.<expires>.<sig>: sig = base64url(HMAC-SHA256(secret, "sc1|aud|role|wake|expires")).
# The wake id is random, so every wake has its own token; `aud` (the server's host and port) is
# signed but not sent: a token signed for one server is no good on another that shares the secret.
# A signed token, unlike a random one kept in memory, lets a restarted server adopt the worker that
# is already running instead of paying for a second one.

MIN_SECRET = 32
WAKE_RE = re.compile(r"[0-9a-z][0-9a-z-]{3,39}")
SHA_RE = re.compile(r"[0-9a-f]{40}")  # a full git commit: what a Kaggle worker runs
_SIG_RE = re.compile(r"[A-Za-z0-9_-]{43}")
_EXPIRES_RE = re.compile(r"[1-9][0-9]{0,11}")


@dataclass(frozen=True)
class Token:
    role: str
    wake: str
    expires: int  # epoch s


def new_wake(prefix: str = "") -> str:
    """A fresh wake id (64 random bits), e.g. "laptop-" + 16 hex digits."""
    return prefix + secrets.token_hex(8)


def _sig(secret: str, aud: str, role: str, wake: str, expires: int) -> str:
    mac = hmac.new(secret.encode(), f"sc1|{aud}|{role}|{wake}|{expires}".encode(), hashlib.sha256)
    return base64.urlsafe_b64encode(mac.digest()).rstrip(b"=").decode()


def make_token(secret: str, role: str, wake: str, expires: int, server: str) -> str:
    """A worker token for one server (its URL or host; audience() normalizes it). ValueError on a
    secret shorter than MIN_SECRET characters, an unknown role, a bad wake id, expiry or server."""
    if type(secret) is not str or len(secret) < MIN_SECRET:
        raise ValueError(f"the GPU secret must be at least {MIN_SECRET} characters")
    if role not in ROLES:
        raise ValueError(f"the role must be one of {', '.join(ROLES)}")
    if type(wake) is not str or not WAKE_RE.fullmatch(wake):
        raise ValueError("a wake id is 4-40 lowercase letters, digits and dashes")
    if type(expires) is not int or not _EXPIRES_RE.fullmatch(str(expires)):
        raise ValueError("expires is a whole number of seconds since 1970")
    aud = audience(server)
    return f"sc1.{role}.{wake}.{expires}.{_sig(secret, aud, role, wake, expires)}"


def peek_token(token) -> Token | None:
    """What a token claims, NOT verified (a worker reading its own role and wake), or None."""
    if type(token) is not str or len(token) > 200:
        return None
    parts = token.split(".")
    if len(parts) != 5 or parts[0] != "sc1":
        return None
    _, role, wake, expires, sig = parts
    if role not in ROLES or not WAKE_RE.fullmatch(wake) or not _EXPIRES_RE.fullmatch(expires) \
            or not _SIG_RE.fullmatch(sig):
        return None
    return Token(role, wake, int(expires))


def check_token(secret: str, token, host: str, now: float) -> Token | None:
    """The token's claims if it was signed with this secret for this server (`host`: the Host header
    the worker's request came with) and is still valid by the server's clock (now), else None.
    Never raises: anything malformed is refused."""
    try:
        t = peek_token(token)
        if t is None or type(secret) is not str or len(secret) < MIN_SECRET:
            return None
        if not now < t.expires <= now + MAX_TOKEN_S:
            return None
        good = _sig(secret, audience(host), t.role, t.wake, t.expires)
        return t if hmac.compare_digest(good.encode(), token.rsplit(".", 1)[1].encode()) else None
    except Exception:
        return None


def key_id(secret: str) -> str:
    """A short fingerprint of the GPU secret, safe to print: the server and the team's laptop both
    show it, so a secret that differs between them shows at a glance. The secret itself never does."""
    return hmac.new(secret.encode(), b"sc-key-id", hashlib.sha256).hexdigest()[:8]


_tidied: set = set()


def read_secret(name: str, environ=None) -> str | None:
    """A secret from the environment, without the spaces, line ending and byte-order mark a Windows
    shell can add when it is stored (PowerShell pipes "\\ufeff...\\r\\n"). None when unset or empty.
    The variable's NAME is printed once if anything had to be removed; the value never is."""
    raw = (os.environ if environ is None else environ).get(name)
    if raw is None:
        return None
    value = raw.strip().lstrip("﻿").strip()
    if value != raw and name not in _tidied:
        _tidied.add(name)
        print(f"{name}: removed a byte-order mark or spaces around the value", flush=True)
    return value or None


# ---- where the server is ------------------------------------------------------------------------

LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


def server_url(server: str) -> str:
    """The app server's base URL, checked: https, or plain http only to this computer (exactly
    localhost, so localhost.example.com does not count). No path, query or credentials."""
    p = urllib.parse.urlsplit(str(server).strip())
    try:
        port = p.port
    except ValueError:
        raise ValueError("That server URL has a bad port.") from None
    host = p.hostname
    if p.scheme not in ("https", "http") or not host:
        raise ValueError("The server must be an https:// URL, like https://stutterclear-123.asia-south1.run.app")
    if p.username is not None or p.password is not None or p.query or p.fragment or p.path not in ("", "/"):
        raise ValueError("The server URL must be just https://<host>, with nothing after it.")
    if p.scheme == "http" and host not in LOCAL_HOSTS:
        raise ValueError("Plain http:// is only for this computer (localhost); use https://.")
    host = f"[{host}]" if ":" in host else host
    return f"{p.scheme}://{host}" + (f":{port}" if port is not None else "")


def ws_url(server: str) -> str:
    """The worker's WebSocket URL on that server: wss://<host>/ws/gpu (ws:// for localhost)."""
    scheme, rest = server_url(server).split("://", 1)
    return ("wss" if scheme == "https" else "ws") + "://" + rest + PATH


def audience(server_or_host: str) -> str:
    """Who a token is for: the server's host (lowercase) and port, the same whether it comes from the
    URL a worker dials or the Host header the server receives. Default ports are left out."""
    s = str(server_or_host).strip()
    p = urllib.parse.urlsplit(s if "://" in s else "//" + s)
    try:
        port = p.port
    except ValueError:
        raise ValueError("bad port") from None
    host = (p.hostname or "").rstrip(".")
    if not host or p.username is not None:
        raise ValueError("no host")
    host = f"[{host}]" if ":" in host else host
    return host if port in (None, 80, 443) else f"{host}:{port}"


# ---- the launcher interface (stutterclear/gpu_worker/kaggle.py, local.py) -----------------------

@dataclass
class Launch:
    """What a launcher knows about the GPU machine it started, or found already running."""
    state: str  # one of LAUNCH_STATES
    reason: str | None = None  # plain English for visitors (failed, quota): never a key or a URL
    until: float | None = None  # quota: when GPU time is back (epoch s), if known
    ref: str | None = None  # the run, for the server log: "owner/slug v13", "pid 4242"
    version: int | None = None  # the Kaggle kernel version of that run
    adopted: bool = False  # an earlier run was still queued or running: nothing new was started
    retry_s: float | None = None  # failed: when trying again may help (None: an operator must fix it)


@dataclass
class Quota:
    """Kaggle's weekly GPU quota, in hours. Kaggle keeps it, so it survives a server restart."""
    used_h: float
    total_h: float
    reserved_h: float = 0.0  # held by sessions running now
    refresh_at: float | None = None  # when the weekly window resets (epoch s)

    @property
    def left_h(self) -> float:
        return self.total_h - self.used_h - self.reserved_h


class Launcher(Protocol):
    """Starts a GPU worker somewhere and reports on it. Every call may block on the network; the
    broker never calls one while holding its lock."""
    name: str  # "kaggle" | "local"
    role: str  # the token role its workers use
    where: str  # shown before the worker says where it is: "Kaggle", "this computer"
    token_ttl_s: float  # how long a wake's token stays valid (Kaggle: its time limit + 1 h of queue)

    def launch(self, wake: str, token: str, reason: str, retired: frozenset = frozenset()) -> Launch:
        """Start a worker for this wake, or adopt a run already queued or running, unless its
        version is in `retired` (runs this server told to stop, or saw leave)."""
        ...

    def poll(self) -> Launch:
        """The state of the run launched or adopted last."""
        ...

    def quota(self) -> Quota | None:
        """GPU time used and left, or None when unknown."""
        ...
