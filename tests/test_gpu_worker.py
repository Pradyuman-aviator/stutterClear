"""The GPU worker (stutterclear/gpu_worker/): what it sends and when, against a tiny stand-in for the
app server on 127.0.0.1 (port 0) that speaks the wire contract; the Kaggle launcher against a stand-in
Kaggle API (no kagglesdk client is ever built, no credentials are ever read); the bootstrap it pushes;
the local launcher; the weights upload script. No GPU, network or keys are needed."""
import asyncio
import base64
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from stutterclear import gpu_protocol as gp
from stutterclear.gpu_worker import engines, kaggle, local
from stutterclear.gpu_worker import worker as wk
from stutterclear.gpu_worker.__main__ import main

ROOT = Path(__file__).resolve().parents[1]
SECRET = "s" * 40
COMMIT = "c" * 40
HF_HOME = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")  # before HOME is moved


@pytest.fixture(autouse=True)
def no_real_secrets(monkeypatch):
    """No test can use a real worker token or secret (conftest.py keeps Kaggle's credentials out of reach)."""
    for name in ("STUTTERCLEAR_GPU_TOKEN", "STUTTERCLEAR_GPU_SECRET", "STUTTERCLEAR_COMMIT"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fast(monkeypatch):
    """The contract's timers, shortened (both sides read them when they use them)."""
    monkeypatch.setattr(gp, "HEARTBEAT_S", 0.05)
    monkeypatch.setattr(gp, "RECONNECT_S", (0.01, 0.02, 0.03))
    monkeypatch.setattr(gp, "STANDBY_POLL_S", 0.02)
    monkeypatch.setattr(wk, "IDLE_MARGIN_S", 0.0)
    monkeypatch.setattr(wk, "BYE_WAIT_S", 1.0)


# ---- a stand-in app server ----------------------------------------------------------------------

class Conn:
    """The server's side of one worker connection, as a test script drives it."""

    def __init__(self, ws):
        self.ws, self.request, self.seen = ws, ws.request, []

    async def send(self, t: str, payload: bytes = b"", **fields):
        await self.ws.send(gp.encode({"t": t, **fields}, payload))

    async def recv(self, *types: str, timeout: float = 5.0) -> tuple[dict, bytes]:
        """The next message of one of these types; heartbeats and the like in between are kept in seen."""
        while True:
            h, payload = gp.decode(await asyncio.wait_for(self.ws.recv(), timeout), expect=gp.FROM_WORKER)
            self.seen.append(h)
            if not types or h["t"] in types:
                return h, payload

    async def welcome(self, **change):
        fields = {"proto": gp.PROTO, "commit": COMMIT, "idle_exit_s": 60, "idle_unused_s": 60, "rotate_after_s": 3000,
                  "now": time.time()} | change
        await self.send(gp.WELCOME, **fields)

    async def hello_welcome(self, **change) -> dict:
        hello, _ = await self.recv(gp.HELLO)
        await self.welcome(**change)
        return hello

    async def ready(self, **change) -> tuple[dict, dict]:
        """HELLO, WELCOME, then the CAPS that says the engines are loaded."""
        hello = await self.hello_welcome(**change)
        while True:
            caps, _ = await self.recv(gp.CAPS)
            if not caps["loading"]:
                return hello, caps

    async def goodbye(self, reason: str = "idle") -> dict:
        await self.send(gp.SHUTDOWN, reason=reason)
        bye, _ = await self.recv(gp.BYE)
        await self.ws.close(gp.CLOSE_SHUTDOWN)
        return bye

    async def job(self, x: np.ndarray, job: str = "job-1", parts=None):
        """A recording, as the server sends it; `parts`: (i, payload) pairs to send instead."""
        await self.send(gp.JOB, job=job, kind="note", samples=len(x), parts=gp.n_parts(len(x)), enc="pcm16", sr=gp.SR,
                        language="en")
        for i, part in enumerate(gp.job_parts(x)) if parts is None else parts:
            await self.send(gp.JOB_AUDIO, part, job=job, i=i)

    async def closed(self, timeout: float = 5.0):
        await asyncio.wait_for(self.ws.wait_closed(), timeout)
        return self.ws.close_code, self.ws.close_reason


class FakeServer:
    """/ws/gpu runs the next script for each connection (none left: 4409); /api/gpu/wanted answers from
    `wanted` ("wanted", "idle" or an HTTP status; the last answer repeats). A handshake is answered with
    the next of `statuses` (None: accepted), and once they are used up with `status` (a 30x carries
    `location`)."""

    def __init__(self, *scripts, wanted=("idle",), statuses=(), status=None, location=None, ping_s=None,
                 info=None):
        self.scripts, self.wanted, self.statuses = list(scripts), list(wanted), list(statuses)
        self.status, self.location = status, location
        self.ping_s = ping_s  # ping like the real server (scripts that wait longer than DEAD_AFTER_S)
        self.info = {"gpu": {"live": True}} if info is None else info  # what /api/info says
        self.handshakes, self.polls, self.conns, self.errors = [], [], [], []
        self.url = None

    def process_request(self, connection, request):
        if request.path == "/api/info":
            return connection.respond(200, json.dumps(self.info))
        if request.path == gp.WANTED_PATH:
            self.polls.append(request.headers.get(gp.TOKEN_HEADER))
            answer = self.wanted.pop(0) if len(self.wanted) > 1 else self.wanted[0]
            if isinstance(answer, int):
                return connection.respond(answer, "no\n")
            return connection.respond(200, json.dumps({"wanted": answer == "wanted"}))
        self.handshakes.append((request.path, request.headers.get(gp.TOKEN_HEADER), request.headers.get("User-Agent")))
        status = self.statuses.pop(0) if self.statuses else self.status
        if status:
            response = connection.respond(status, "not here\n")
            if self.location:
                response.headers["Location"] = self.location
            return response
        return None

    async def handler(self, ws):
        conn = Conn(ws)
        self.conns.append(conn)
        if not self.scripts:
            await ws.close(gp.CLOSE_BUSY, "another worker is active")
            return
        pinger = asyncio.create_task(self._ping(conn)) if self.ping_s else None
        try:
            await self.scripts.pop(0)(conn)
        except Exception as e:  # reported by run(): an assertion in a script fails the test
            self.errors.append(e)
            await ws.close(gp.CLOSE_BUSY)
        finally:
            if pinger:
                pinger.cancel()

    async def _ping(self, conn: Conn):
        n = 0
        while True:
            await asyncio.sleep(self.ping_s)
            n += 1
            await conn.send(gp.PING, n=n, ts=time.monotonic())

    async def run(self, role: str = "kaggle", timeout: float = 15.0, token=None, **opts) -> int:
        async with serve(self.handler, "127.0.0.1", 0, process_request=self.process_request, max_size=gp.MAX_FRAME,
                         compression=None) as s:
            self.url = f"http://127.0.0.1:{s.sockets[0].getsockname()[1]}"
            if token is None:
                fixed = gp.make_token(SECRET, role, gp.new_wake(), int(time.time()) + 3600, self.url)
                token = (lambda: gp.make_token(SECRET, role, gp.new_wake(f"{role}-"), int(time.time()) + 3600, self.url)
                         ) if role == "laptop" else (lambda: fixed)
            options = wk.Options(server=self.url, role=role, token=token, commit=COMMIT, fake=True)
            for k, v in opts.items():
                setattr(options, k, v)
            self.worker = wk.Worker(options)
            code = await asyncio.wait_for(self.worker.run(), timeout)
        if self.errors:
            raise self.errors[0]
        return code

    async def cli(self, *args: str, role: str = "local", timeout: float = 30.0,
                  env: dict | None = None) -> tuple[int, str]:
        """`python -m stutterclear.gpu_worker --server <this server> ...` as its own process: (exit code, output)."""
        async with serve(self.handler, "127.0.0.1", 0, process_request=self.process_request, max_size=gp.MAX_FRAME,
                         compression=None) as s:
            self.url = f"http://127.0.0.1:{s.sockets[0].getsockname()[1]}"
            token = gp.make_token(SECRET, role, gp.new_wake(), int(time.time()) + 3600, self.url)
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "stutterclear.gpu_worker", "--server", self.url, "--where", role, *args, cwd=ROOT,
                env={**os.environ, **(env or {}), "STUTTERCLEAR_GPU_TOKEN": token, "PYTHONUNBUFFERED": "1"},
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            try:
                out, _ = await asyncio.wait_for(proc.communicate(), timeout)
            except TimeoutError:
                proc.kill()
                out, _ = await proc.communicate()
                raise AssertionError("the worker did not finish:\n" + out.decode(errors="replace")) from None
        if self.errors:
            raise AssertionError(out.decode(errors="replace")) from self.errors[0]
        return proc.returncode, out.decode(errors="replace")


def noise(seconds: float, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-0.5, 0.5, int(seconds * gp.SR)).astype(np.float32)


# ---- the command line ---------------------------------------------------------------------------

def test_help_needs_no_torch():
    code = ("import runpy, sys; sys.argv = ['gpu_worker', '--help']\n"
            "try:\n    runpy.run_module('stutterclear.gpu_worker', run_name='__main__')\n"
            "except SystemExit as e:\n    assert e.code == 0\n"
            "print(sorted(m for m in ('torch', 'transformers', 'crisperwhisper', 'kagglesdk') if m in sys.modules))")
    t = time.time()
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip().endswith("[]") and "--server" in out.stdout and time.time() - t < 10


def test_settings_that_are_wrong_exit_2_without_showing_a_token(monkeypatch, capsys):
    https = "https://stutterclear-123.asia-south1.run.app"
    assert main(["--server", "http://example.com"]) == 2  # plain http only to this computer
    assert main(["--server", https, "--where", "kaggle"]) == 2
    assert "No worker token" in capsys.readouterr().err
    monkeypatch.setenv("STUTTERCLEAR_GPU_TOKEN", "Bearer something-else")
    assert main(["--server", https]) == 2 and "not a StutterClear worker token" in capsys.readouterr().err
    token = gp.make_token(SECRET, "kaggle", "abcd1234", int(time.time()) + 3600, https)
    monkeypatch.setenv("STUTTERCLEAR_GPU_TOKEN", token)
    assert main(["--server", https, "--where", "laptop"]) == 2  # a Kaggle token on the laptop
    assert main(["--server", https, "--no-live", "--no-notes"]) == 2
    assert main(["--server", https, "--notes-device", "gpu1"]) == 2
    assert main(["--server", https, "--deadline", str(time.time() + 60)]) == 0  # too close to its time limit
    out = capsys.readouterr()
    assert token not in out.out + out.err and SECRET not in out.out + out.err


def test_the_laptop_shows_its_key_id_never_its_secret(monkeypatch, capsys, fast):
    server = FakeServer(wanted=(403,))

    async def go():
        async with serve(server.handler, "127.0.0.1", 0, process_request=server.process_request) as s:
            host = f"127.0.0.1:{s.sockets[0].getsockname()[1]}"
            monkeypatch.setenv("STUTTERCLEAR_GPU_SECRET", "\ufeff" + SECRET + "\r\n")  # as PowerShell stores it
            return host, await asyncio.to_thread(main, ["--server", f"http://{host}", "--where", "laptop", "--fake"])

    host, code = asyncio.run(go())
    out = capsys.readouterr().out
    assert code == wk.REFUSED and not server.handshakes
    assert f"key id {gp.key_id(SECRET)}" in out and SECRET not in out and "refused this worker (HTTP 403)" in out
    assert gp.check_token(SECRET, server.polls[0], host, time.time()).role == "laptop"  # signed for this server


# ---- connecting ---------------------------------------------------------------------------------

def test_hello_carries_the_tokens_claims_and_the_token_only_travels_in_the_header(fast):
    seen = {}

    async def script(c: Conn):
        seen["hello"], seen["caps"] = await c.ready()
        seen["bye"] = await c.goodbye("idle")

    server = FakeServer(script)
    token = gp.make_token(SECRET, "kaggle", "9c1e5a7b2d3f4e60", int(time.time()) + 3600, "http://127.0.0.1")
    assert asyncio.run(server.run(token=lambda: token, deadline=time.time() + 3 * 3600)) == wk.OK
    (path, header, agent), = server.handshakes
    assert path == gp.PATH and header == token and token not in path and agent.startswith("stutterclear-gpu-worker")
    hello = seen["hello"]
    assert (hello["role"], hello["wake"], hello["proto"], hello["commit"]) == ("kaggle", "9c1e5a7b2d3f4e60", gp.PROTO,
                                                                              COMMIT)
    assert len(hello["worker"]) == 8 and hello["where"] == "Kaggle" and hello["deadline"] > time.time()
    assert seen["caps"] | {"window_ms": None} == {"t": gp.CAPS, "notes": True, "live": True, "loading": False,
                                                 "window_ms": None, "models": {"live": "fake", "recordings": "fake"},
                                                 "error": None}
    assert seen["bye"]["reason"] == "idle"


@pytest.mark.parametrize("status,code", [(403, wk.REFUSED), (404, wk.REFUSED), (401, wk.REFUSED)])
def test_a_refused_handshake_is_not_retried(status, code, fast):
    server = FakeServer(status=status)
    assert asyncio.run(server.run()) == code and len(server.handshakes) == 1


def test_a_redirect_is_never_followed(fast):
    other = FakeServer()

    async def go():
        async with serve(other.handler, "127.0.0.1", 0, process_request=other.process_request) as s:
            target = f"ws://127.0.0.1:{s.sockets[0].getsockname()[1]}/ws/gpu"
            return await FakeServer(status=302, location=target).run()

    assert asyncio.run(go()) == wk.REFUSED and not other.handshakes  # the token never went there


def test_a_busy_or_starting_server_is_tried_again(fast):
    async def script(c: Conn):
        await c.ready()
        await c.goodbye()

    server = FakeServer(script, statuses=[503, 429, None])  # Cloud Run starting, then busy, then up
    assert asyncio.run(server.run()) == wk.OK and len(server.handshakes) == 3


@pytest.mark.parametrize("close,role,code", [(gp.CLOSE_VERSION, "kaggle", wk.VERSION), (gp.CLOSE_BUSY, "kaggle", wk.OK),
                                             (gp.CLOSE_BUSY, "local", wk.OK)])
def test_close_codes(close, role, code, fast):
    async def script(c: Conn):
        await c.recv(gp.HELLO)
        await c.ws.close(close)

    assert asyncio.run(FakeServer(script).run(role=role)) == code


def test_a_lost_connection_is_reconnected_with_the_same_token(fast):
    async def restart(c: Conn):  # the server restarts (uvicorn closes every socket with 1012)
        await c.ready()
        await c.ws.close(gp.CLOSE_RESTART)

    async def drop(c: Conn):  # the network drops without a close frame
        await c.recv(gp.HELLO)
        c.ws.transport.abort()

    async def adopted(c: Conn):
        await c.ready()
        await c.goodbye()

    server = FakeServer(restart, drop, adopted)
    assert asyncio.run(server.run()) == wk.OK
    assert len(server.handshakes) == 3 and len({h[1] for h in server.handshakes}) == 1


def test_three_malformed_closes_in_a_row_stop_the_worker(fast):
    async def bad(c: Conn):
        await c.recv(gp.HELLO)
        await c.ws.close(gp.CLOSE_BAD)

    server = FakeServer(bad, bad, bad, bad)
    assert asyncio.run(server.run()) == wk.FAILED and len(server.handshakes) == 3


def test_a_malformed_message_closes_the_connection_and_the_worker_comes_back(fast):
    async def text(c: Conn):
        await c.hello_welcome()
        await c.ws.send("not a frame")
        assert (await c.closed())[0] == gp.CLOSE_BAD

    async def nonsense(c: Conn):
        await c.hello_welcome()
        await c.ws.send(b"\x00\x00\x00\x02{}")  # no type
        assert (await c.closed())[0] == gp.CLOSE_BAD

    async def fine(c: Conn):
        await c.ready()
        await c.goodbye()

    assert asyncio.run(FakeServer(text, nonsense, fine).run()) == wk.OK


def test_a_silent_server_is_given_up_on(monkeypatch, fast):
    monkeypatch.setattr(gp, "DEAD_AFTER_S", 0.3)

    async def silent(c: Conn):
        await c.recv(gp.HELLO)
        await c.closed(timeout=10)  # never welcomes, never pings: the worker gives up

    async def fine(c: Conn):
        await c.ready()
        await c.goodbye()

    t = time.time()
    assert asyncio.run(FakeServer(silent, fine).run()) == wk.OK and time.time() - t < 5


def test_pings_are_answered(fast):
    async def script(c: Conn):
        await c.ready()
        await c.send(gp.PING, n=7, ts=123.5)
        pong, _ = await c.recv(gp.PONG)
        assert (pong["n"], pong["ts"]) == (7, 123.5)
        beat, _ = await c.recv(gp.HEARTBEAT)
        assert beat["jobs"] == [] and beat["live"] is False and isinstance(beat["gpu_mem_gb"], list)
        await c.goodbye()

    assert asyncio.run(FakeServer(script).run()) == wk.OK


# ---- recordings ---------------------------------------------------------------------------------

def test_a_recording_goes_up_in_parts_and_its_result_comes_back(fast):
    x = noise(7.3)
    got = {}

    async def script(c: Conn):
        await c.ready()
        for job in ("job-1", "job-2"):
            await c.job(x, job)
            stages = [(await c.recv(gp.PROGRESS))[0]["stage"] for _ in range(2)]
            assert stages == ["listening", "transcribing"]
            got[job] = await c.recv(gp.RESULT)
        await c.goodbye()

    assert asyncio.run(FakeServer(script).run()) == wk.OK
    h, payload = got["job-2"]
    frames = gp.frames_for(len(x))
    assert h["job"] == "job-2" and h["frames"] == frames and h["slow"] is False
    assert set(h["timing"]) == {"teacher_s", "asr_s"}
    probs = gp.probs_from(payload, frames)
    assert probs[55, engines.WORD_REP] == pytest.approx(0.9) and probs[10, engines.WORD_REP] == pytest.approx(0.02)
    assert gp.check_words(h["words"], len(x) / gp.SR) == h["words"] and len(h["words"]) == 14


@pytest.mark.parametrize("parts,why", [
    ("swapped", "out of order"), ("short", "incomplete"), ("long", "too long")])
def test_a_recording_that_arrives_wrong_is_an_error_worth_retrying(parts, why, fast):
    x = noise(5)
    good = list(enumerate(gp.job_parts(x)))
    wrong = {"swapped": [good[1], good[0], *good[2:]], "short": [*good[:-1], (good[-1][0], good[-1][1][:-2])],
             "long": [*good[:-1], (good[-1][0], good[-1][1] + b"\x00\x00")]}[parts]

    async def script(c: Conn):
        await c.ready()
        await c.job(x, "job-1", parts=wrong)
        error, _ = await c.recv(gp.ERROR)
        assert error["job"] == "job-1" and error["retry"] is True and why in error["message"]
        await c.job(x, "job-2")  # the worker carries on
        result, _ = await c.recv(gp.RESULT)
        assert result["job"] == "job-2"
        await c.goodbye()

    assert asyncio.run(FakeServer(script).run()) == wk.OK


def test_a_cancelled_recording_sends_no_result(fast):
    async def script(c: Conn):
        await c.ready()
        await c.job(noise(3), "job-1")
        await c.recv(gp.PROGRESS)
        await c.send(gp.CANCEL, job="job-1")
        await c.job(noise(2), "job-2")
        result, _ = await c.recv(gp.RESULT, gp.ERROR)
        assert result["t"] == gp.RESULT and result["job"] == "job-2"
        await c.goodbye()
        assert not any(h["t"] == gp.RESULT and h["job"] == "job-1" for h in c.seen)

    assert asyncio.run(FakeServer(script).run(delay_ms=300)) == wk.OK


def test_a_second_recording_while_one_runs_is_turned_back(fast):
    async def script(c: Conn):
        await c.ready()
        await c.job(noise(3), "job-1")
        await c.recv(gp.PROGRESS)
        await c.job(noise(2), "job-2")
        error, _ = await c.recv(gp.ERROR)
        assert error["job"] == "job-2" and error["retry"] is True
        result, _ = await c.recv(gp.RESULT)
        assert result["job"] == "job-1"
        await c.goodbye()

    assert asyncio.run(FakeServer(script).run(delay_ms=300)) == wk.OK


@pytest.mark.parametrize("out,retry,why", [
    ("too-many-words", False, "could not be sent"),
    ("wrong-frames", False, "frames"),
    (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), True, "out of memory"),
    (ValueError("a bad recording"), False, "a bad recording"),
])
def test_what_goes_wrong_on_the_gpu_is_reported_and_marked_retry_or_not(out, retry, why, monkeypatch, fast):
    def analyse(self, x, progress, cancelled, language="en"):
        progress("listening")
        if isinstance(out, Exception):
            raise out
        n = gp.frames_for(len(x)) + (1 if out == "wrong-frames" else 0)
        words = [{"w": "uh", "s": 0.1, "e": 0.2}] * (gp.MAX_WORDS + 1 if out == "too-many-words" else 1)
        return np.zeros((n, gp.N_CLASSES), np.float32), words, False, {}

    monkeypatch.setattr(engines.FakeNotesEngine, "analyse", analyse)

    async def script(c: Conn):
        await c.ready()
        await c.job(noise(2))
        error, _ = await c.recv(gp.ERROR)
        assert error["retry"] is retry and why in error["message"]
        await c.goodbye()

    assert asyncio.run(FakeServer(script).run()) == wk.OK


def test_crash_after_kills_the_process_mid_recording():
    """--crash-after 1: the first recording's result never comes; the process dies with code 1."""
    async def script(c: Conn):
        await c.ready()
        await c.job(noise(2))
        with pytest.raises(ConnectionClosed):  # the connection drops before any result
            await c.recv(gp.RESULT, timeout=20)
        assert not any(h["t"] == gp.RESULT for h in c.seen)

    code, out = asyncio.run(FakeServer(script).cli("--fake", "--crash-after", "1"))
    assert code == wk.FAILED and "crashing on purpose" in out
    assert "sc1." not in out  # the token is never printed


@pytest.mark.skipif(os.environ.get("STUTTERCLEAR_E2E_GPU") != "1",
                    reason="the real models on this computer's GPU: set STUTTERCLEAR_E2E_GPU=1")
def test_end_to_end_on_this_gpu():
    """The real worker (no stand-ins) with the checkpoints in runs/: a live stream of a demo clip, then the
    same clip as a recording (on a small GPU the two make room for each other, as in the app), then a
    goodbye. The recording's probabilities and words match the clip's caches."""
    import soundfile as sf
    clip = ROOT / "data" / "demo_clips" / "read-1.wav"
    x, _ = sf.read(clip, dtype="float32")
    cached, words = np.load(clip.with_suffix(".teacher.npy")), json.loads(clip.with_suffix(".crisper.json").read_text())
    got = {}

    async def script(c: Conn):
        await c.hello_welcome()
        caps = {"loading": True}
        while caps["loading"]:
            caps, _ = await c.recv(gp.CAPS, timeout=600)
        assert caps["notes"] and caps["live"] and caps["error"] is None and caps["window_ms"] > 0
        got["caps"] = caps
        await c.send(gp.LIVE_OPEN, stream="live-1", mode="gentle", lookahead_ms=200, window=WINDOW, hop_frames=8)
        await c.recv(gp.LIVE_READY)
        sent, ms = 0, []
        for seq in range(25):  # what RemoteTeacherDetector asks for, one hop (160 ms) at a time
            end = WINDOW + seq * 2560
            f1 = (end - 3200 - 400) // gp.FRAME + 1
            h, pcm = live_audio(x, seq, (end - WINDOW) // gp.FRAME * gp.FRAME, end, f1 - 8 if seq else 0, f1, sent)
            await c.send(gp.LIVE_AUDIO, pcm, **h)
            sent = end
            reply, payload = await c.recv(gp.LIVE_PROBS, timeout=60)
            ms.append(reply["ms"])
            rows = gp.probs_from(payload, reply["f1"] - reply["f0"])
            assert reply["f1"] == f1 and rows.shape[0] == f1 - reply["f0"]
        got["live_ms"] = sorted(ms)[len(ms) // 2]
        await c.send(gp.LIVE_CLOSE, stream="live-1")
        t = time.time()
        await c.job(x, "job-1")
        got["result"] = await c.recv(gp.RESULT, timeout=900)
        got["note_s"] = time.time() - t
        await c.goodbye()

    code, out = asyncio.run(FakeServer(script, ping_s=5).cli("--first", "live", timeout=1500, env={"HF_HOME": HF_HOME}))
    assert code == wk.OK, out
    h, payload = got["result"]
    probs = gp.probs_from(payload, h["frames"])
    assert h["frames"] == len(cached) == gp.frames_for(len(x))
    diff = np.abs(probs - cached)
    assert diff.mean() < 0.02 and np.percentile(diff, 99) < 0.2, (diff.mean(), diff.max())
    assert abs(len(h["words"]) - len(words)) <= 0.15 * len(words)
    print(f"\nlive window median {got['live_ms']} ms; a {len(x) / gp.SR:.0f} s recording in {got['note_s']:.1f} s "
          f"(teacher {h['timing']['teacher_s']} s, ASR {h['timing']['asr_s']} s, slow={h['slow']}); "
          f"probabilities vs the cache: mean |diff| {diff.mean():.4f}, max {diff.max():.3f}; "
          f"{len(h['words'])} words (cache {len(words)})\n{out}")


# ---- sleeping, leaving, standing by -------------------------------------------------------------

def test_an_unused_worker_sleeps_sooner_than_a_used_one(fast):
    times = {}

    async def unused(c: Conn):
        await c.ready(idle_unused_s=0.3, idle_exit_s=5)
        t = time.monotonic()
        bye, _ = await c.recv(gp.BYE)
        times["unused"] = time.monotonic() - t
        assert bye["reason"] == "idle"

    async def used(c: Conn):
        await c.ready(idle_unused_s=0.3, idle_exit_s=1.0)
        await c.job(noise(1.5))
        await c.recv(gp.RESULT)
        t = time.monotonic()
        bye, _ = await c.recv(gp.BYE)
        times["used"] = time.monotonic() - t
        assert bye["reason"] == "idle"

    assert asyncio.run(FakeServer(unused).run()) == wk.OK
    assert asyncio.run(FakeServer(used).run()) == wk.OK
    assert 0.2 < times["unused"] < 0.9 and 0.8 < times["used"] < 2.0


def test_near_its_deadline_a_worker_says_goodbye(monkeypatch, fast):
    monkeypatch.setattr(gp, "DRAIN_S", 0.5)

    async def script(c: Conn):
        hello, _ = await c.ready()
        assert hello["deadline"] is not None
        bye, _ = await c.recv(gp.BYE, timeout=3)
        assert bye["reason"] == "deadline"

    t = time.time()
    assert asyncio.run(FakeServer(script).run(deadline=time.time() + 0.9)) == wk.OK and time.time() - t < 3


def test_an_idle_connection_is_rotated_before_the_servers_time_limit(fast):
    async def rotated(c: Conn):
        await c.ready(rotate_after_s=0.2)
        assert await c.closed() == (gp.CLOSE_ROTATE, "rotate")

    async def fresh(c: Conn):
        await c.ready()
        await c.goodbye()

    server = FakeServer(rotated, fresh)
    assert asyncio.run(server.run()) == wk.OK and len(server.handshakes) == 2


def test_the_worker_leaves_when_the_server_is_gone_for_good(fast):
    async def once(c: Conn):
        await c.ready(idle_unused_s=0.3)
        c.ws.transport.abort()  # and from now on the server answers 503

    server = FakeServer(once, statuses=[None], status=503)
    t = time.time()
    assert asyncio.run(server.run()) == wk.OK and time.time() - t < 5 and len(server.handshakes) > 2


def test_ctrl_c_says_goodbye_first(fast):
    said = {}

    async def script(c: Conn):
        await c.ready()
        said["ready"].set()
        said["bye"] = (await c.recv(gp.BYE))[0]["reason"]

    async def go():
        said["ready"] = asyncio.Event()
        task = asyncio.create_task(FakeServer(script).run())
        await said["ready"].wait()
        task.cancel()  # what Ctrl+C does to asyncio.run's task
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    assert said["bye"] == "shutdown"


def test_the_laptop_stands_by_until_it_is_wanted_and_after_each_wake(fast):
    async def wake(c: Conn):
        await c.ready()
        await c.goodbye("idle")

    server = FakeServer(wake, wanted=("idle", "idle", "wanted", "idle", 403))
    assert asyncio.run(server.run(role="laptop")) == wk.REFUSED
    assert len(server.polls) == 5 and len(server.handshakes) == 1
    claims = [gp.peek_token(t) for t in server.polls]
    assert all(c.role == "laptop" for c in claims) and len({c.wake for c in claims}) == 5  # a fresh token each time
    assert gp.peek_token(server.handshakes[0][1]).wake not in {c.wake for c in claims}


@pytest.mark.parametrize("info,live", [({"gpu": {"live": False}}, False), ({}, False), ({"gpu": {"live": True}}, True),
                                       ("not json", True)])
def test_the_laptop_loads_the_live_teacher_only_for_a_server_that_uses_it(info, live, capsys, fast):
    said = {}

    async def wake(c: Conn):
        said["caps"] = (await c.ready())[1]
        await c.goodbye()

    server = FakeServer(wake, wanted=("wanted", 403), info=info)
    assert asyncio.run(server.run(role="laptop")) == wk.REFUSED
    assert said["caps"]["live"] is live and (said["caps"]["models"]["live"] is None) is not live
    assert ("the live teacher stays unloaded" in capsys.readouterr().out) is not live


def test_a_busy_server_sends_the_laptop_back_to_standing_by(fast):
    async def busy(c: Conn):
        await c.recv(gp.HELLO)
        await c.ws.close(gp.CLOSE_BUSY)

    server = FakeServer(busy, wanted=("wanted", "idle", 403))
    assert asyncio.run(server.run(role="laptop")) == wk.REFUSED and len(server.handshakes) == 1
    assert len(server.polls) == 3


def test_a_laptop_woken_after_a_long_standby_does_not_leave_at_once(fast):
    said = {}

    async def wake(c: Conn):
        await c.ready(idle_unused_s=0.5, idle_exit_s=5)
        t = time.monotonic()
        await c.recv(gp.BYE)
        said["after"] = time.monotonic() - t
        await c.ws.close(gp.CLOSE_SHUTDOWN)

    server = FakeServer(wake, wanted=("idle",) * 60 + ("wanted", 403))  # stood by far longer than 0.5 s
    assert asyncio.run(server.run(role="laptop")) == wk.REFUSED
    assert said["after"] > 0.4  # its idle clock starts at the wake, not when it loaded its models


def test_another_worker_active_right_after_a_lost_connection_is_tried_again(fast):
    async def lost(c: Conn):
        await c.ready()
        c.ws.transport.abort()  # lost without a close: the server may hold this socket a few seconds more

    async def busy(c: Conn):
        await c.recv(gp.HELLO)
        await c.ws.close(gp.CLOSE_BUSY)  # so the fresh one is turned away as "another worker"

    async def fine(c: Conn):
        await c.ready()
        await c.goodbye()

    server = FakeServer(lost, busy, fine)
    assert asyncio.run(server.run()) == wk.OK and len(server.handshakes) == 3  # it came back, not left for good


def test_a_machine_without_a_gpu_says_so_and_exits_5(monkeypatch, fast):
    monkeypatch.setattr(engines, "gpus", lambda: [])
    said = {}

    async def script(c: Conn):
        await c.hello_welcome()
        said["caps"], _ = await c.recv(gp.CAPS)
        said["bye"], _ = await c.recv(gp.BYE)
        await c.ws.close(gp.CLOSE_SHUTDOWN)

    assert asyncio.run(FakeServer(script).run(fake=False)) == wk.NO_MODELS
    assert said["caps"]["error"] == engines.NO_GPU and not said["caps"]["notes"] and said["bye"]["reason"] == "error"


def test_missing_weights_are_reported_as_such(monkeypatch, tmp_path, fast):
    from stutterclear import teacher_runtime
    monkeypatch.setattr(engines, "gpus", lambda: [{"name": "Tesla T4", "mem_gb": 15.0}] * 2)
    monkeypatch.setattr(teacher_runtime, "RECORDINGS_CKPT", tmp_path / "missing" / "best.pt")
    said = {}

    async def script(c: Conn):
        hello = await c.hello_welcome()
        assert hello["where"] == "Kaggle (2× Tesla T4)"
        while "error" not in said:
            caps, _ = await c.recv(gp.CAPS)
            if caps["error"]:
                said["error"] = caps["error"]
        await c.recv(gp.BYE)

    assert asyncio.run(FakeServer(script).run(fake=False, live=False)) == wk.NO_MODELS
    assert said["error"] == engines.MISSING


# ---- where things run ---------------------------------------------------------------------------

def test_plan_devices():
    assert engines.plan_devices(2, 15.0) == engines.Plan("cuda:0", "cuda:1")  # Kaggle's T4 x2
    assert engines.plan_devices(1, 16.0) == engines.Plan("cuda:0", "cuda:0")  # one big GPU
    assert engines.plan_devices(1, 6.4) == engines.Plan(None, None)  # the laptop: as the app runs it
    assert engines.plan_devices(0, 0.0) is None


@pytest.mark.parametrize("slow", [False, True])
def test_on_the_laptops_shared_gpu_a_recording_runs_as_the_app_runs_it(slow, monkeypatch):
    """notes.py's own path: its GPU lock held, a live stream keeping the GPU (live_running), the CPU when
    CrisperWhisper does not fit (slow)."""
    from stutterclear.app import notes
    seen = {}

    def transcribe(note):
        seen["locked"], seen["live"] = notes._gpu.locked(), notes.live_running()
        note["slow"] = slow
        return SimpleNamespace(words=[SimpleNamespace(word="hi", start=0.1, end=0.31234)])

    monkeypatch.setattr(notes, "teacher_probs", lambda x: np.zeros((gp.frames_for(len(x)), gp.N_CLASSES), np.float32))
    monkeypatch.setattr(notes, "_transcribe", transcribe)
    monkeypatch.setattr(notes, "live_running", lambda: True)  # what NotesEngine.load sets: the worker's stream
    stages = []
    probs, words, was_slow, timing = engines.NotesEngine(None).analyse(noise(2), stages.append, lambda: False)
    assert seen == {"locked": True, "live": True} and stages == ["listening", "transcribing"] and was_slow is slow
    assert words == [{"w": "hi", "s": 0.1, "e": 0.312}] and probs.shape == (gp.frames_for(32000), gp.N_CLASSES)
    assert set(timing) == {"teacher_s", "asr_s"}
    assert engines.NotesEngine(None).analyse(noise(2), stages.append, lambda: True) is None  # cancelled in between


def test_what_a_load_or_a_gpu_failure_means():
    assert engines.load_error(engines.ModelsError(engines.MISSING)) == engines.MISSING
    assert engines.load_error(OSError("We couldn't connect to 'https://huggingface.co'")) == engines.NO_DOWNLOAD
    assert engines.load_error(RuntimeError("CUDA out of memory")) == "The GPU ran out of memory loading its models."
    assert engines.load_error(KeyError("state_dict")) == engines.NO_LOAD
    assert engines.fatal(RuntimeError("CUDA error: an illegal memory access was encountered"))
    assert not engines.fatal(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert not engines.fatal(ValueError("CUDA error"))


def test_where_a_worker_says_it_runs():
    t4 = {"name": "Tesla T4", "mem_gb": 15.0}
    assert wk.where_label("kaggle", [t4, t4]) == "Kaggle (2× Tesla T4)"
    assert wk.where_label("kaggle", [t4]) == "Kaggle (Tesla T4)" and wk.where_label("kaggle", []) == "Kaggle"
    assert wk.where_label("laptop", [t4]) == "the team's laptop" and wk.where_label("local", []) == "this computer"
    assert len(wk.where_label("kaggle", [{"name": "x" * 60, "mem_gb": 1}] * 3)) <= gp.MAX_WHERE


# ---- live streams (P2) --------------------------------------------------------------------------

WINDOW = 67200  # 4 s of context and 200 ms ahead


def live_audio(x: np.ndarray, seq: int, start: int, end: int, f0: int, f1: int, sent: int) -> tuple[dict, bytes]:
    """A LIVE_AUDIO as the server sends it: the samples [sent, end) not sent before."""
    return ({"stream": "live-1", "seq": seq, "start": start, "end": end, "f0": f0, "f1": f1},
            gp.pcm16(x[max(sent, start):end]))


def test_a_live_stream_gets_rows_for_the_frames_it_asks_for(fast):
    x = noise(6)
    fake = engines.FakeLiveEngine()

    async def script(c: Conn):
        _, caps = await c.ready()
        assert caps["live"] and caps["window_ms"] == fake.window_ms
        await c.send(gp.LIVE_OPEN, stream="live-1", mode="gentle", lookahead_ms=200, window=WINDOW, hop_frames=8)
        assert (await c.recv(gp.LIVE_READY, gp.LIVE_ERROR))[0]["t"] == gp.LIVE_READY
        await c.send(gp.LIVE_OPEN, stream="live-2", mode="gentle", lookahead_ms=200, window=WINDOW, hop_frames=8)
        error, _ = await c.recv(gp.LIVE_ERROR)
        assert error["stream"] == "live-2" and "one live stream" in error["message"].lower()
        sent = 0
        for seq, (start, end, f0, f1) in enumerate([(0, WINDOW, 0, 200), (2560, WINDOW + 2560, 200, 208)]):
            h, pcm = live_audio(x, seq, start, end, f0, f1, sent)
            await c.send(gp.LIVE_AUDIO, pcm, **h)
            sent = end
            reply, payload = await c.recv(gp.LIVE_PROBS)
            assert (reply["stream"], reply["seq"], reply["f0"], reply["f1"]) == ("live-1", seq, f0, f1)
            rows = gp.probs_from(payload, f1 - f0)
            w0 = start // gp.FRAME
            assert np.array_equal(rows, gp.window_rows(fake.window_probs(x[start:end], w0), w0, f0, f1))
        assert np.allclose(rows[:, engines.WORD_REP], 0.02)
        beat, _ = await c.recv(gp.HEARTBEAT)
        assert beat["live"] is True and beat["window_ms"] is not None
        await c.send(gp.LIVE_CLOSE, stream="live-1")
        h, pcm = live_audio(x, 2, 5120, WINDOW + 5120, 208, 216, sent)
        await c.send(gp.LIVE_AUDIO, pcm, **h)  # after the close: nothing comes back
        await c.send(gp.PING, n=1, ts=1.0)
        await c.recv(gp.PONG)
        await c.goodbye()
        assert sum(h["t"] == gp.LIVE_PROBS for h in c.seen) == 2

    assert asyncio.run(FakeServer(script).run()) == wk.OK


def test_a_live_stream_is_ready_only_once_its_model_is_warm(monkeypatch, fast):
    warmed = []
    real = engines.FakeLiveEngine.window_probs

    def slow_first(self, x, w0=0):  # the first window loads the model again (a recording made room)
        if not warmed:
            time.sleep(0.3)
        warmed.append(len(x))
        return real(self, x, w0)

    monkeypatch.setattr(engines.FakeLiveEngine, "window_probs", slow_first)
    said = {}

    async def script(c: Conn):
        await c.ready()
        t = time.monotonic()
        await c.send(gp.LIVE_OPEN, stream="live-1", mode="gentle", lookahead_ms=200, window=WINDOW, hop_frames=8)
        await c.recv(gp.LIVE_READY)
        said["after"] = time.monotonic() - t
        await c.goodbye()

    assert asyncio.run(FakeServer(script).run()) == wk.OK
    assert said["after"] >= 0.25 and warmed[0] == wk.MIN_WINDOW  # warmed before it said ready


def test_a_live_backlog_is_answered_once_and_a_gap_starts_again():
    replies, ready = [], []
    x = noise(8)
    s = wk.LiveStream({"stream": "live-1", "window": WINDOW}, 1, engines.FakeLiveEngine(), 0.0,
                      lambda what, stream, *args: (ready if what == "ready" else replies).append((what, *args)))
    sent = 0
    for seq in range(3):  # three windows queued before the thread looks: one answer covers them all
        end = WINDOW + seq * 2560
        h, pcm = live_audio(x, seq, end - WINDOW, end, 0 if seq == 0 else 200 + 8 * (seq - 1), 200 + 8 * seq, sent)
        s.q.put((h, pcm))
        sent = end
    s.thread.start()
    deadline = time.time() + 5
    while not replies and time.time() < deadline:
        time.sleep(0.01)
    what, frame, ms = replies[0]
    h, payload = gp.decode(frame)
    assert what == "reply" and (h["seq"], h["f0"], h["f1"]) == (2, 0, 216) and len(replies) == 1
    assert ready == [("ready",)]  # (said once its model was warm, before any answer)
    assert s.total == sent and len(s.buf) <= WINDOW + 4 * gp.FRAME
    end = sent + 10 * gp.SR  # a gap (the server skipped audio): start again from what this message holds
    s.q.put(({"stream": "live-1", "seq": 3, "start": end - WINDOW, "end": end, "f0": 900, "f1": 908},
             gp.pcm16(noise(WINDOW / gp.SR, seed=2))))
    while len(replies) < 2 and time.time() < deadline:
        time.sleep(0.01)
    h, payload = gp.decode(replies[1][1])
    assert s.total == end and (h["f0"], h["f1"]) == (900, 908) and len(payload) == 8 * gp.N_CLASSES * 4
    s.stop()
    s.thread.join(2)
    assert not s.thread.is_alive()


def test_a_silent_live_stream_is_dropped(monkeypatch, fast):
    monkeypatch.setattr(gp, "LIVE_IDLE_S", 0.3)

    async def script(c: Conn):
        await c.ready()
        await c.send(gp.LIVE_OPEN, stream="live-1", mode="strong", lookahead_ms=500, window=72000, hop_frames=8)
        await c.recv(gp.LIVE_READY)
        error, _ = await c.recv(gp.LIVE_ERROR, timeout=3)
        assert error["stream"] == "live-1" and "No audio" in error["message"]
        await c.goodbye()

    assert asyncio.run(FakeServer(script).run()) == wk.OK


def test_without_the_live_teacher_a_stream_is_refused(fast):
    async def script(c: Conn):
        _, caps = await c.ready()
        assert caps["live"] is False and caps["models"]["live"] is None
        await c.send(gp.LIVE_OPEN, stream="live-1", mode="gentle", lookahead_ms=200, window=WINDOW, hop_frames=8)
        error, _ = await c.recv(gp.LIVE_ERROR)
        assert "not loaded" in error["message"]
        await c.goodbye()

    assert asyncio.run(FakeServer(script).run(live=False)) == wk.OK


# ---- the Kaggle launcher ------------------------------------------------------------------------

OWNER = "pradyumansharmaaa"
WORKER_KERNEL = f"{OWNER}/stutterclear-gpu-worker"
V1, V2 = f"{OWNER}/stutterclear-teacher", f"{OWNER}/stutterclear-teacher-v2"
PUBLIC_URL = "https://stutterclear-123.asia-south1.run.app"


def http_error(status: int, message: str = "error"):
    import requests
    return requests.HTTPError(message, response=SimpleNamespace(status_code=status))


class FakeKaggle:
    """A stand-in for kagglesdk's KaggleClient (one object plays every service): kernels with their
    latest version and run status, datasets, the GPU quota. `fail` holds exceptions for a method's
    next calls; `lose_answer` pushes make their version and then lose the answer."""

    def __init__(self, worker=None, status="COMPLETE", failure=""):
        self.meta = {V1: SimpleNamespace(is_private=True, current_version_number=1),
                     V2: SimpleNamespace(is_private=True, current_version_number=1)}
        if worker is not None:
            self.meta[WORKER_KERNEL] = SimpleNamespace(is_private=True, current_version_number=worker)
        self.status, self.failure = status, failure
        self.saves, self.calls, self.fail, self.datasets_private = [], [], {}, {}
        self.lose_answer, self.save_response = 0, None
        self.kernels = SimpleNamespace(kernels_api_client=self)
        self.datasets = SimpleNamespace(dataset_api_client=self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _call(self, name: str):
        self.calls.append(name)
        if self.fail.get(name):
            raise self.fail[name].pop(0)

    def get_kernel(self, req):
        self._call("get_kernel")
        ref = f"{req.user_name}/{req.kernel_slug}"
        if ref not in self.meta:  # what Kaggle really answers for a kernel that does not exist (not 404)
            raise http_error(403, "Permission 'kernels.get' was denied")
        return SimpleNamespace(metadata=self.meta[ref])

    def get_kernel_session_status(self, req):
        self._call("status")
        assert f"{req.user_name}/{req.kernel_slug}" == WORKER_KERNEL
        return SimpleNamespace(status=SimpleNamespace(name=self.status), failure_message=self.failure)

    def save_kernel(self, req):
        self._call("save_kernel")
        self.saves.append(req)
        if self.save_response is not None:
            return self.save_response
        meta = self.meta.setdefault(WORKER_KERNEL, SimpleNamespace(is_private=True, current_version_number=0))
        meta.current_version_number += 1
        self.status = "QUEUED"
        if self.lose_answer:
            self.lose_answer -= 1
            raise ConnectionError("the answer was lost")
        return SimpleNamespace(error=None, version_number=meta.current_version_number, url="",
                               invalid_kernel_sources=[], invalid_dataset_sources=[])

    def get_accelerator_quota_statistics(self, req):
        self._call("quota")
        quota = SimpleNamespace(time_used=timedelta(hours=12.5), time_reserved=timedelta(hours=1),
                                total_time_allowed=timedelta(hours=30))
        return SimpleNamespace(gpu_quota=quota, quota_refresh_time=datetime(2026, 10, 31, 0, 0))

    def get_dataset(self, req):
        self._call("get_dataset")
        ref = f"{req.owner_slug}/{req.dataset_slug}"
        if ref not in self.datasets_private:
            raise http_error(404)
        return SimpleNamespace(is_private=self.datasets_private[ref])


def launcher(fake: FakeKaggle, **change) -> kaggle.KaggleLauncher:
    settings = {"server": PUBLIC_URL, "commit": COMMIT, "kernel": WORKER_KERNEL, "weight_kernels": (V1, V2),
                "sleep": lambda s: None} | change
    return kaggle.KaggleLauncher(lambda: fake, **settings)


def wake_token(wake: str = "9c1e5a7b2d3f4e60") -> str:
    return gp.make_token(SECRET, "kaggle", wake, int(time.time()) + 4 * 3600, PUBLIC_URL)


def pushed_config(request) -> dict:
    blob = request.text.split('base64.b64decode("', 1)[1].split('"', 1)[0]
    return json.loads(base64.b64decode(blob))


def test_from_env_builds_a_client_that_is_never_verbose(monkeypatch):
    import kagglesdk
    built = []
    monkeypatch.setattr(kagglesdk, "KaggleClient", lambda **k: built.append(k))
    env = {"KAGGLE_USERNAME": OWNER, "KAGGLE_KEY": "k" * 32 + "\r\n", "STUTTERCLEAR_COMMIT": COMMIT,
           "STUTTERCLEAR_PUBLIC_URL": PUBLIC_URL + "/"}
    launch = kaggle.KaggleLauncher.from_env(env)
    launch.client()
    assert built == [{"verbose": False, "username": OWNER, "password": "k" * 32}]  # the key as stored, cleaned
    launch = kaggle.KaggleLauncher.from_env(env | {"KAGGLE_API_TOKEN": "KGAT_" + "t" * 32})
    launch.client()
    assert built[-1] == {"verbose": False, "api_token": "KGAT_" + "t" * 32}  # a named token is preferred
    assert (launch.kernel, launch.weight_kernels, launch.server) == (WORKER_KERNEL, (V1, V2), PUBLIC_URL)
    assert (launch.accelerator, launch.timeout_s, launch.pin) == ("NvidiaTeslaT4", 10800, "original")
    assert (launch.name, launch.role, launch.where, launch.token_ttl_s) == ("kaggle", "kaggle", "Kaggle", 14400)


@pytest.mark.parametrize("change,named", [
    ({"KAGGLE_KEY": None}, "KAGGLE_API_TOKEN"),
    ({"STUTTERCLEAR_COMMIT": "main"}, "STUTTERCLEAR_COMMIT"),
    ({"STUTTERCLEAR_COMMIT": None}, "STUTTERCLEAR_COMMIT"),
    ({"STUTTERCLEAR_REPO_URL": "https://gitlab.com/x/y.git"}, "STUTTERCLEAR_REPO_URL"),
    ({"STUTTERCLEAR_PUBLIC_URL": "http://stutterclear.example.com"}, "STUTTERCLEAR_PUBLIC_URL"),
    ({"STUTTERCLEAR_PUBLIC_URL": "http://127.0.0.1:8801"}, "STUTTERCLEAR_PUBLIC_URL"),  # Kaggle cannot reach it
    ({"STUTTERCLEAR_PUBLIC_URL": None}, "STUTTERCLEAR_PUBLIC_URL"),
    ({"STUTTERCLEAR_KAGGLE_KERNEL": "no-owner"}, "STUTTERCLEAR_KAGGLE_KERNEL"),
    ({"STUTTERCLEAR_KAGGLE_WEIGHT_KERNELS": V1}, "STUTTERCLEAR_KAGGLE_WEIGHT_KERNELS"),
    ({"STUTTERCLEAR_KAGGLE_TIMEOUT_S": "99999"}, "STUTTERCLEAR_KAGGLE_TIMEOUT_S"),
    ({"STUTTERCLEAR_KAGGLE_PIN": "newest"}, "STUTTERCLEAR_KAGGLE_PIN"),
])
def test_from_env_refuses_what_is_missing_or_unpinned(change, named):
    env = {"KAGGLE_USERNAME": OWNER, "KAGGLE_KEY": "secret-key-value", "STUTTERCLEAR_COMMIT": COMMIT,
           "STUTTERCLEAR_PUBLIC_URL": PUBLIC_URL} | change
    with pytest.raises(kaggle.LauncherError) as e:
        kaggle.KaggleLauncher.from_env({k: v for k, v in env.items() if v is not None})
    assert named in str(e.value) and "secret-key-value" not in str(e.value)


def test_a_push_is_private_on_t4s_with_the_weights_and_this_wakes_settings():
    fake = FakeKaggle()
    fake.datasets_private[f"{OWNER}/stutterclear-weights"] = True
    image = "gcr.io/kaggle-gpu-images/python@sha256:" + "a" * 64
    launch = launcher(fake, docker_image=image, live=True, weights_dataset=f"{OWNER}/stutterclear-weights")
    token = wake_token()
    assert launch.launch("9c1e5a7b2d3f4e60", token, "voice-note") == gp.Launch("queued", ref=f"{WORKER_KERNEL} v1",
                                                                              version=1)
    (r,) = fake.saves
    assert (r.slug, r.new_title, r.language, r.kernel_type) == (WORKER_KERNEL, "stutterclear-gpu-worker", "python",
                                                                "script")
    assert r.is_private and r.enable_gpu and r.enable_internet and not r.enable_tpu
    assert (r.machine_shape, r.session_timeout_seconds, r.docker_image_pinning_type, r.docker_image) == (
        "NvidiaTeslaT4", 10800, "original", image)
    assert r.kernel_data_sources == [V1, V2] and r.dataset_data_sources == [f"{OWNER}/stutterclear-weights"]
    config = pushed_config(r)
    assert config | {"weights": None} == {"server": PUBLIC_URL, "wake": "9c1e5a7b2d3f4e60", "token": token,
                                          "commit": COMMIT, "repo": kaggle.REPO, "idle_exit_s": gp.IDLE_EXIT_S,
                                          "timeout_s": 10800, "first": "notes", "live": True, "weights": None}
    weights = config["weights"]
    assert weights["live"][0] == f"/kaggle/input/notebooks/{OWNER}/stutterclear-teacher/teacher/best.pt"
    assert weights["recordings"][0] == f"/kaggle/input/notebooks/{OWNER}/stutterclear-teacher-v2/teacher_v2/best.pt"
    assert weights["live"][-1] == f"/kaggle/input/stutterclear-weights/{kaggle.LIVE_FILE}"
    assert token not in r.text  # only inside the encoded settings
    compile(r.text, "run.py", "exec")


def test_the_first_engine_is_a_fixed_word_never_the_visitors_text():
    fake = FakeKaggle()
    launcher(fake).launch("abcd1234", wake_token("abcd1234"), "live")
    fake.status = "COMPLETE"
    launcher(fake).launch("abcd1235", wake_token("abcd1235"), "--first=x; rm -rf /")
    assert [pushed_config(r)["first"] for r in fake.saves] == ["live", "notes"]


@pytest.mark.parametrize("status,state", [("QUEUED", "queued"), ("NEW_SCRIPT", "queued"), ("SOMETHING_NEW", "queued"),
                                          ("RUNNING", "running"), ("COMPLETE", "done"), ("ERROR", "failed"),
                                          ("CANCEL_REQUESTED", "failed"), ("CANCEL_ACKNOWLEDGED", "failed")])
def test_run_statuses(status, state):
    assert kaggle.kaggle_state(status) == state
    fake = FakeKaggle(worker=13, status=status, failure=f"CUDA error: {wake_token()} at /tmp/stutterclear/code/x.py")
    got = launcher(fake).poll()
    assert (got.state, got.version, got.ref) == (state, 13, f"{WORKER_KERNEL} v13")
    if state == "failed":
        assert got.reason.startswith("The GPU machine stopped: CUDA error") and got.retry_s == 60
        assert "sc1." not in got.reason and "/tmp" not in got.reason


def test_poll_before_any_run_and_when_kaggle_does_not_answer():
    assert launcher(FakeKaggle()).poll().state == "done"
    fake = FakeKaggle(worker=3)
    fake.fail["get_kernel"] = [http_error(429)]
    with pytest.raises(kaggle.KaggleError):
        launcher(fake).poll()


def test_just_after_a_push_a_finished_status_is_not_believed():
    clock = [1000.0]
    fake = FakeKaggle(worker=4)
    launch = launcher(fake, clock=lambda: clock[0])
    assert launch.launch("abcd1234", wake_token("abcd1234"), "button").version == 5
    fake.status = "COMPLETE"  # still the previous run's
    assert launch.poll().state == "queued"
    clock[0] += kaggle.SETTLE_S + 1
    assert launch.poll().state == "done"


def test_a_queued_or_running_run_is_adopted_unless_it_was_retired():
    fake = FakeKaggle(worker=7, status="RUNNING")
    got = launcher(fake).launch("abcd1234", wake_token("abcd1234"), "video")
    assert got == gp.Launch("running", ref=f"{WORKER_KERNEL} v7", version=7, adopted=True) and not fake.saves
    got = launcher(fake).launch("abcd1234", wake_token("abcd1234"), "video", retired=frozenset({7}))
    assert (got.state, got.version, got.adopted, len(fake.saves)) == ("queued", 8, False, 1)


def test_a_push_whose_answer_was_lost_is_adopted_not_pushed_again():
    fake = FakeKaggle(worker=2)
    fake.lose_answer = 1
    got = launcher(fake).launch("abcd1234", wake_token("abcd1234"), "dictate")
    assert got.adopted and got.version == 3 and len(fake.saves) == 1


def test_a_kaggle_that_does_not_answer_is_tried_four_times():
    fake = FakeKaggle()
    fake.fail["save_kernel"] = [http_error(503), http_error(429), ConnectionError("reset"), http_error(502)]
    waits = []
    got = launcher(fake, sleep=waits.append).launch("abcd1234", wake_token("abcd1234"), "button")
    assert (got.state, got.retry_s, got.reason) == ("failed", 60, kaggle.NO_ANSWER)
    assert waits == list(kaggle.RETRY_WAITS_S) and fake.calls.count("save_kernel") == 4


def test_a_network_error_is_never_taken_for_the_gpu_quota():
    import requests
    lost = requests.ConnectionError("HTTPSConnectionPool(host='www.kaggle.com', port=443): Max retries exceeded "
                                    "with url: /api/i/kernels.KernelsApiService/SaveKernel (Caused by ...)")
    fake = FakeKaggle()
    fake.fail["save_kernel"] = [lost] * 4
    got = launcher(fake).launch("abcd1234", wake_token("abcd1234"), "button")
    assert (got.state, got.retry_s, got.reason) == ("failed", 60, kaggle.NO_ANSWER)  # tried four times, not "resting"
    assert fake.calls.count("save_kernel") == 4 and "quota" not in fake.calls
    fake = FakeKaggle()
    fake.fail["get_kernel"] = [lost]  # during the privacy check
    assert launcher(fake).launch("abcd1234", wake_token("abcd1234"), "button").state == "failed"
    fake = FakeKaggle()
    fake.fail["save_kernel"] = [http_error(429, "API rate quota exceeded")] * 4  # slow down: not the GPU quota
    assert launcher(fake).launch("abcd1234", wake_token("abcd1234"), "button").reason == kaggle.NO_ANSWER


def test_a_hung_call_times_out():
    fake = FakeKaggle()
    fake.get_kernel = lambda req: time.sleep(3)
    t = time.time()
    got = launcher(fake, call_timeout_s=0.1).launch("abcd1234", wake_token("abcd1234"), "button")
    assert (got.state, got.retry_s) == ("failed", 60) and time.time() - t < 2


@pytest.mark.parametrize("problem,state,retry_s,reason", [
    ("auth", "failed", None, kaggle.REFUSED),
    ("quota", "quota", None, kaggle.QUOTA),
    ("quota-http", "quota", None, kaggle.QUOTA),
    ("sources", "failed", None, kaggle.NO_FILES),
    ("other", "failed", 300, "Kaggle could not start the GPU machine."),
])
def test_push_failures_are_told_apart(problem, state, retry_s, reason):
    fake = FakeKaggle()
    ok = {"error": None, "version_number": None, "url": "", "invalid_kernel_sources": [], "invalid_dataset_sources": []}
    if problem == "auth":
        fake.fail["save_kernel"] = [http_error(401, "Unauthenticated")]
    elif problem == "quota-http":
        fake.fail["save_kernel"] = [http_error(400, "You have exceeded your weekly GPU quota")]
    else:
        fake.save_response = SimpleNamespace(**ok | {
            "quota": {"error": "Maximum weekly GPU quota of 30.00 hours reached"},
            "sources": {"invalid_kernel_sources": [V2]}, "other": {"error": "Something unusual"}}[problem])
    got = launcher(fake).launch("abcd1234", wake_token("abcd1234"), "button")
    assert (got.state, got.retry_s, got.reason) == (state, retry_s, reason)
    if state == "quota":
        assert got.until == datetime(2026, 10, 31, tzinfo=timezone.utc).timestamp()


@pytest.mark.parametrize("problem,logged", [("worker-public", "worker kernel"),
                                            ("v1-public", "stutterclear-teacher is public"),
                                            ("v2-missing", "was not found"), ("dataset-public", "dataset")])
def test_nothing_is_pushed_unless_every_kernel_is_private(problem, logged, capsys):
    fake = FakeKaggle(worker=1)
    fake.meta[WORKER_KERNEL].is_private = problem != "worker-public"
    fake.meta[V1].is_private = problem != "v1-public"
    if problem == "v2-missing":
        del fake.meta[V2]
    fake.datasets_private[f"{OWNER}/stutterclear-weights"] = problem != "dataset-public"
    launch = launcher(fake, weights_dataset=f"{OWNER}/stutterclear-weights")
    assert launch.launch("abcd1234", wake_token("abcd1234"), "button") == gp.Launch("failed", kaggle.SETUP)
    assert not fake.saves and logged in capsys.readouterr().out


def test_the_privacy_check_runs_once_per_process():
    fake = FakeKaggle()
    launch = launcher(fake)
    launch.launch("abcd1234", wake_token("abcd1234"), "button")
    fake.status = "COMPLETE"
    launch.launch("abcd1235", wake_token("abcd1235"), "button")
    assert len(fake.saves) == 2 and fake.calls.count("get_kernel") == 3 + 1 + 1  # the check, then one look per launch


def test_the_quota_is_read_in_hours():
    q = launcher(FakeKaggle()).quota()
    assert (q.used_h, q.total_h, q.reserved_h, q.left_h) == (12.5, 30.0, 1.0, 16.5)
    assert q.refresh_at == datetime(2026, 10, 31, tzinfo=timezone.utc).timestamp()
    fake = FakeKaggle()
    fake.fail["quota"] = [http_error(500)]
    assert launcher(fake).quota() is None


def test_kaggles_text_is_cleaned_before_anyone_sees_it():
    token = wake_token()
    text = (f"Error {token} in https://www.kaggle.com/code/x?token=abc at /kaggle/input/a/b/best.pt "
            f"and C:\\Users\\ultra\\x.py: {'QUJD' * 30}\x1b[31m done")
    clean = kaggle.sanitize(text)
    assert token not in clean and "kaggle.com" not in clean and "/kaggle/input" not in clean and "QUJDQUJD" not in clean
    assert "Users" not in clean and "\x1b" not in clean and clean.startswith("Error") and clean.endswith("done")
    assert kaggle.stopped("") == "The GPU machine stopped."


# ---- the bootstrap the launcher pushes ----------------------------------------------------------

def bootstrap_config(folder: Path, **change) -> dict:
    """One wake's settings, with the weights at exact paths under `folder` (the first live path missing)."""
    (folder / "v1" / "teacher").mkdir(parents=True)
    (folder / "v1" / "teacher" / "best.pt").write_bytes(b"x")
    (folder / "v2" / "teacher_v2").mkdir(parents=True)
    (folder / "v2" / "teacher_v2" / "best.pt").write_bytes(b"x")
    return {"server": PUBLIC_URL, "wake": "9c1e5a7b2d3f4e60", "token": wake_token(), "commit": COMMIT,
            "repo": kaggle.REPO, "idle_exit_s": 600.0, "timeout_s": 10800, "first": "notes", "live": False,
            "weights": {"live": [str(folder / "nope.pt"), str(folder / "v1" / "teacher" / "best.pt")],
                        "recordings": [str(folder / "v2" / "teacher_v2" / "best.pt")]}} | change


def load_bootstrap(folder: Path, monkeypatch, config: dict | None = None):
    """The rendered run.py (the raw template without a config) as a module, its /tmp and /kaggle/working
    moved under `folder`."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "run.py"
    raw = (kaggle.TEMPLATE / "run.py").read_text(encoding="utf-8")
    path.write_text(kaggle.render_kernel(config) if config else raw, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("bootstrap_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "TMP", str(folder / "tmp"))
    monkeypatch.setattr(module, "WORK", str(folder / "working"))
    monkeypatch.setattr(module.time, "sleep", lambda s: None)
    monkeypatch.chdir(folder)  # the bootstrap changes directory: this puts it back afterwards
    return module


def record_commands(monkeypatch, codes: dict | None = None) -> list:
    """subprocess.run replaced: every call recorded; git and the worker exit with the next of their codes."""
    calls, codes = [], {"git": [], "worker": [0]} | (codes or {})

    def run(args, **kw):
        calls.append((list(args), kw))
        kind = "worker" if "stutterclear.gpu_worker" in args else "git" if args[0] == "git" else None
        return subprocess.CompletedProcess(args, codes[kind].pop(0) if kind and codes[kind] else 0)

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_the_kernel_template_is_python_before_and_after_rendering(tmp_path):
    raw = (kaggle.TEMPLATE / "run.py").read_text(encoding="utf-8")
    compile(raw, "run.py", "exec")
    assert raw.count("__CONFIG__") == 1 and raw.count(kaggle.PLACEHOLDER) == 1
    compile(kaggle.render_kernel(bootstrap_config(tmp_path)), "run.py", "exec")
    meta = kaggle.template_metadata()
    assert meta["is_private"] is True and meta["enable_internet"] is True and meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["kernel_sources"] == [V1, V2] and meta["code_file"] == "run.py" and meta["id"] == WORKER_KERNEL


def test_the_bootstrap_runs_the_worker_with_the_token_in_its_environment_only(tmp_path, monkeypatch, capsys):
    config = bootstrap_config(tmp_path / "in")
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    monkeypatch.setenv("KAGGLE_DATA_PROXY_TOKEN", "kaggles-own-session-token")
    boot = load_bootstrap(tmp_path / "run", monkeypatch, config)
    working = Path(boot.WORK)
    (working / "folder").mkdir(parents=True)
    (working / "left-behind.txt").write_text("x")
    calls = record_commands(monkeypatch)
    with pytest.raises(SystemExit) as e:
        boot.main()
    assert e.value.code == 0
    git = [a for a, _ in calls if a[0] == "git"]
    assert ["git", "-C", f"{boot.TMP}/code", "fetch", "-q", "--depth", "1", "origin", COMMIT] in git
    assert not any("main" in a or "master" in a for a in git)  # the exact commit, never a branch
    pip, pip_kw = next((a, k) for a, k in calls if "pip" in a)
    assert "--only-binary=:all:" in pip and "--no-deps" in pip and pip[-1].endswith("gpu_worker/requirements.lock")
    assert "--require-hashes" in pip  # only the exact files hashed in the lock
    assert "STUTTERCLEAR_GPU_TOKEN" not in pip_kw["env"]  # the token only where it is needed
    args, kw = next((a, k) for a, k in calls if "stutterclear.gpu_worker" in a)
    assert config["token"] not in " ".join(args) and "--no-live" in args and "--first=notes" in args
    assert args[args.index("--where") + 1] == "kaggle" and "--idle-exit=600" in args and kw["cwd"] == f"{boot.TMP}/code"
    deadline = int(next(a for a in args if a.startswith("--deadline=")).split("=")[1])
    assert time.time() + 10800 - 400 < deadline < time.time() + 10800
    env = kw["env"]
    assert env["STUTTERCLEAR_GPU_TOKEN"] == config["token"] and env["STUTTERCLEAR_COMMIT"] == COMMIT
    assert env["STUTTERCLEAR_TEACHER"] == config["weights"]["live"][1]  # the first exact path that exists
    assert env["STUTTERCLEAR_RECORDINGS_TEACHER"] == config["weights"]["recordings"][0]
    assert not any(k.startswith("KAGGLE_") for k in env) and env["HF_HOME"].startswith(boot.TMP)
    assert os.listdir(working) == []  # nothing is left as the notebook's output
    out = capsys.readouterr().out
    assert config["token"] not in out and "sc1." not in out and "weights: live ok / recordings ok" in out


def test_missing_weights_are_left_for_the_worker_to_report(tmp_path, monkeypatch):
    config = bootstrap_config(tmp_path / "in")
    config["weights"] = {"live": [str(tmp_path / "x.pt")], "recordings": [str(tmp_path / "y.pt")]}
    boot = load_bootstrap(tmp_path / "run", monkeypatch, config)
    calls = record_commands(monkeypatch, {"worker": [5]})  # the worker: "could not find its model files"
    with pytest.raises(SystemExit) as e:
        boot.main()
    env = next(k["env"] for a, k in calls if "stutterclear.gpu_worker" in a)
    assert e.value.code == 1 and env["STUTTERCLEAR_RECORDINGS_TEACHER"] == env["STUTTERCLEAR_TEACHER"] == "missing"
    assert sum("stutterclear.gpu_worker" in a for a, _ in calls) == 1  # a fresh process would not find them either


def test_the_bootstrap_never_falls_back_to_other_code(tmp_path, monkeypatch):
    boot = load_bootstrap(tmp_path / "a", monkeypatch, bootstrap_config(tmp_path / "in"))
    calls = record_commands(monkeypatch, {"git": [0, 0, 128] * 3})  # the commit cannot be fetched, three times
    with pytest.raises(SystemExit) as e:
        boot.main()
    assert e.value.code == 1 and not any("stutterclear.gpu_worker" in a or "pip" in a for a, _ in calls)
    assert sum(a[3] == "fetch" for a, _ in calls if a[0] == "git") == 3
    unpinned = load_bootstrap(tmp_path / "b", monkeypatch, bootstrap_config(tmp_path / "in2", commit="main"))
    calls = record_commands(monkeypatch)
    with pytest.raises(SystemExit) as e:
        unpinned.main()
    assert e.value.code == 1 and not any(a[0] == "git" for a, _ in calls)


@pytest.mark.parametrize("codes,runs,rc", [([1, -9, 0], 3, 0), ([1, 1, 1, 1], 3, 1), ([3], 1, 1), ([0], 1, 0)])
def test_a_crashed_worker_gets_a_fresh_process_twice(codes, runs, rc, tmp_path, monkeypatch):
    boot = load_bootstrap(tmp_path / "run", monkeypatch, bootstrap_config(tmp_path / "in"))
    calls = record_commands(monkeypatch, {"worker": list(codes)})
    with pytest.raises(SystemExit) as e:
        boot.main()
    assert sum("stutterclear.gpu_worker" in a for a, _ in calls) == runs and e.value.code == rc


def test_unreadable_settings_stop_the_bootstrap_without_printing_them(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as e:
        load_bootstrap(tmp_path, monkeypatch)  # the template as it is, never pushed by the server
    assert e.value.code == 1 and "settings are unreadable" in capsys.readouterr().out


# ---- the local launcher -------------------------------------------------------------------------

def test_the_local_launcher_hands_the_worker_the_gpu_and_its_token_but_no_secrets(monkeypatch):
    for name, value in {"CUDA_VISIBLE_DEVICES": "-1", "STUTTERCLEAR_RECORDINGS_TEACHER": "none",
                        "STUTTERCLEAR_GPU_SECRET": SECRET, "NEBIUS_API_KEY": "n", "KAGGLE_KEY": "k",
                        "STUTTERCLEAR_HOSTED": "1"}.items():
        monkeypatch.setenv(name, value)
    started = []

    class Proc:
        pid, rc = 4242, None

        def poll(self):
            return self.rc

    monkeypatch.setattr(local.subprocess, "Popen",
                        lambda args, env, cwd, **apart: started.append((args, env, cwd, apart)) or Proc())
    launch = local.LocalLauncher.from_env({"STUTTERCLEAR_PUBLIC_URL": "http://127.0.0.1:8801",
                                           "STUTTERCLEAR_GPU_LOCAL_ARGS": "--fake --delay-ms 1200"})
    token = gp.make_token(SECRET, "local", "abcd1234", int(time.time()) + 3600, "http://127.0.0.1:8801")
    assert launch.launch("abcd1234", token, "voice-note") == gp.Launch("running", ref="pid 4242")
    (args, env, cwd, apart), = started
    assert token not in " ".join(args) and env["STUTTERCLEAR_GPU_TOKEN"] == token and cwd == ROOT
    # in a process group of its own: Ctrl+C on the server (a restart) leaves it running, as on Kaggle
    assert apart == ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                     else {"start_new_session": True})
    assert args[-5:] == ["--first=notes", "--fake", "--delay-ms", "1200", "--no-live"]
    for gone in ("CUDA_VISIBLE_DEVICES", "STUTTERCLEAR_RECORDINGS_TEACHER", "STUTTERCLEAR_GPU_SECRET", "NEBIUS_API_KEY",
                 "KAGGLE_KEY"):
        assert gone not in env
    assert env["STUTTERCLEAR_HOSTED"] == "1"
    assert launch.launch("abcd1234", token, "voice-note").adopted and len(started) == 1  # still running
    assert launch.poll().state == "running"
    launch.proc.rc = 0
    assert launch.poll().state == "done"
    launch.proc.rc = 1
    assert launch.poll() == gp.Launch("failed", "The worker exited with code 1.", ref="pid 4242", retry_s=60)
    assert launch.quota() is None and (launch.name, launch.role, launch.token_ttl_s) == ("local", "local", 86400.0)


def test_the_local_launcher_can_pretend_the_quota_is_nearly_used_up():
    q = local.LocalLauncher("http://127.0.0.1:8801", quota_left_h=1.0).quota()
    refresh = datetime.fromtimestamp(q.refresh_at, timezone.utc)
    assert (q.used_h, q.total_h, q.left_h) == (29.0, 30.0, 1.0)
    assert refresh.weekday() == 5 and refresh.hour == 0 and 0 < q.refresh_at - time.time() <= 7 * 86400


# ---- the weights upload script (the fallback) ---------------------------------------------------

def upload_script() -> dict:
    import runpy
    return runpy.run_path(str(ROOT / "scripts" / "upload_weights_kaggle.py"))


def test_the_upload_dry_run_lists_the_files_without_kaggle(tmp_path, monkeypatch, capsys):
    from stutterclear import teacher_runtime
    v1, v2 = tmp_path / "v1.pt", tmp_path / "v2.pt"
    v1.write_bytes(b"x" * 2_000_000)
    v2.write_bytes(b"x" * 3_000_000)
    monkeypatch.setattr(teacher_runtime, "TEACHER_CKPT", v1)
    monkeypatch.setattr(teacher_runtime, "RECORDINGS_CKPT", v2)
    script = upload_script()
    before = set(sys.modules)
    assert script["main"](["--dataset", f"{OWNER}/stutterclear-weights", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert kaggle.LIVE_FILE in out and kaggle.RECORDINGS_FILE in out and "2.0 MB" in out and "3.0 MB" in out
    assert not any(m == "kaggle" or m.startswith("kaggle.") for m in set(sys.modules) - before)
    assert script["main"](["--dataset", "no-slash", "--dry-run"]) == 2
    v2.unlink()
    assert script["main"](["--dataset", f"{OWNER}/stutterclear-weights", "--dry-run"]) == 1


def test_staging_links_the_checkpoints_under_the_names_the_launcher_expects(tmp_path):
    src = tmp_path / "best.pt"
    src.write_bytes(b"weights")
    folder = tmp_path / "stage"
    folder.mkdir()
    upload_script()["stage"]([(src, kaggle.LIVE_FILE), (src, kaggle.RECORDINGS_FILE)], str(folder))
    assert sorted(os.listdir(folder)) == sorted([kaggle.LIVE_FILE, kaggle.RECORDINGS_FILE])
    assert (folder / kaggle.RECORDINGS_FILE).read_bytes() == b"weights" and src.read_bytes() == b"weights"
