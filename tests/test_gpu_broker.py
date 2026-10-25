"""The GPU broker (stutterclear/app/gpu.py): waking, the recordings queue, the worker's connection and
who may serve, the GPU's budget and the HTTP API. A fake launcher, a fake clock and a fake worker that
speaks the wire protocol through the test client; no GPU, network, Kaggle or keys (conftest.py keeps
Kaggle's credentials out of reach)."""
import threading
import time
from types import SimpleNamespace

import anyio
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from stutterclear import gpu_protocol as gp
from stutterclear.app import gpu, server

SECRET = "k" * 40
COMMIT = "c" * 40
T0 = 1_800_000_000.0  # 08:00 UTC: the day's launch cap is far from midnight
WORDS = [{"w": "I", "s": 0.1, "e": 0.2}, {"w": "I", "s": 0.3, "e": 0.4}, {"w": "want", "s": 0.5, "e": 0.8}]


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self) -> float:
        return self.t


class FakeLauncher:
    name, role, where, token_ttl_s = "kaggle", "kaggle", "Kaggle", 4 * 3600.0

    def __init__(self):
        self.calls = []  # (wake, token, reason, retired)
        self.result = gp.Launch("queued", ref="owner/worker v1", version=1)
        self.state = gp.Launch("running", ref="owner/worker v1", version=1)
        self.quota_left = None  # hours of 30, or None: unknown
        self.polls = 0

    def launch(self, wake, token, reason, retired=frozenset()):
        self.calls.append((wake, token, reason, retired))
        return self.result

    def poll(self):
        self.polls += 1
        return self.state

    def quota(self):
        if self.quota_left is None:
            return None
        return gp.Quota(used_h=30.0 - self.quota_left, total_h=30.0, refresh_at=T0 + 3 * 86400)


def make(clock, launcher, **settings) -> gpu.Broker:
    s = {"poll_s": 0.01, "wakes_per_hour": 100, "wakes_per_day": 100, "client_wakes_per_day": 100} | settings
    return gpu.Broker(SECRET, launcher, gpu.Settings(**s), clock=clock, commit=COMMIT, autostart=False,
                      public_url="http://testserver")


@pytest.fixture
def env(monkeypatch):
    """A broker on a fake clock with a fake launcher, serving as the app's broker."""
    clock, launcher = Clock(), FakeLauncher()
    b = make(clock, launcher)
    monkeypatch.setattr(gpu, "broker", b)
    return SimpleNamespace(b=b, clock=clock, launcher=launcher, client=TestClient(server.app))


def token(role="kaggle", wake="9c1e5a7b2d3f4e60", secret=SECRET, server_url="http://testserver", ttl=3600.0):
    return gp.make_token(secret, role, wake, int(T0 + ttl), server_url)


class Worker:
    """A GPU worker on the test client's WebSocket, speaking the wire protocol."""

    def __init__(self, client, tok: str, worker: str = "0badcafe"):
        self.token, self.claims, self.worker = tok, gp.peek_token(tok), worker
        self.session = client.websocket_connect(gp.PATH, headers={gp.TOKEN_HEADER: tok})
        self.ws = self.session.__enter__()

    def send(self, t, payload=b"", **fields):
        self.ws.send_bytes(gp.encode({"t": t, **fields}, payload))

    def recv(self, timeout: float = 5.0) -> tuple[dict, bytes]:
        """The next message but pings; WebSocketDisconnect once the server closed (a test that
        waits for nothing fails instead of hanging)."""
        async def get():
            with anyio.fail_after(timeout):
                return await self.ws._send_rx.receive()
        while True:
            msg = self.ws.portal.call(get)
            self.ws._raise_on_close(msg)
            h, payload = gp.decode(msg["bytes"], expect=gp.FROM_SERVER)
            if h["t"] != gp.PING:
                return h, payload

    def hello(self, commit=COMMIT, proto=gp.PROTO, deadline=None, where="Kaggle (2× Tesla T4)") -> dict:
        self.send(gp.HELLO, proto=proto, worker=self.worker, wake=self.claims.wake, role=self.claims.role,
                  where=where, commit=commit, gpus=[], deadline=deadline, loading=True)
        h, _ = self.recv()
        assert h["t"] == gp.WELCOME and h["commit"] == COMMIT
        return h

    def caps(self, notes=True, live=False, error=None, window_ms=None):
        self.send(gp.CAPS, notes=notes, live=live, loading=False, window_ms=window_ms, models={}, error=error)

    def job(self) -> tuple[dict, np.ndarray]:
        h, _ = self.recv()
        assert h["t"] == gp.JOB, h
        parts = [self.recv() for _ in range(h["parts"])]
        assert [p[0]["i"] for p in parts] == list(range(h["parts"])) and {p[0]["job"] for p in parts} == {h["job"]}
        return h, gp.from_pcm16(b"".join(p[1] for p in parts))

    def result(self, h: dict, frames: int | None = None, words=WORDS, slow=False):
        n = gp.frames_for(h["samples"]) if frames is None else frames
        probs = np.zeros((n, gp.N_CLASSES), np.float32)
        probs[10:20, 1] = 0.9
        self.send(gp.RESULT, gp.probs_bytes(probs), job=h["job"], frames=n, words=words, slow=slow, timing={})

    def bye(self, reason="idle") -> int:
        """Say goodbye; the code the server closes with."""
        self.send(gp.BYE, reason=reason)
        with pytest.raises(WebSocketDisconnect) as e:
            self.recv()
        return e.value.code

    def close(self):
        self.session.__exit__(None, None, None)


def ready_worker(env, tok=None, **hello) -> Worker:
    w = Worker(env.client, tok or token())
    w.hello(**hello)
    w.caps()
    wait_for(lambda: env.b.status()["state"] == "ready")
    return w


def wait_for(cond, timeout: float = 5.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.005)


def audio(seconds: float = 2.0, seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(int(seconds * gp.SR)) * 0.1).astype(np.float32)


def analysing(b: gpu.Broker, x: np.ndarray, client: str = "1.2.3.4", abandoned=lambda: False) -> SimpleNamespace:
    """broker.analyse() in its own thread, as a note's thread runs it."""
    out = SimpleNamespace(updates=[], result=None, error=None)

    def run():
        try:
            out.result = b.analyse(x, lambda stage, ahead: out.updates.append((stage, ahead)), abandoned, client)
        except gpu.GpuError as e:
            out.error = str(e)
    out.thread = threading.Thread(target=run, daemon=True)
    out.thread.start()
    return out


def done(job: SimpleNamespace) -> SimpleNamespace:
    job.thread.join(5)
    assert not job.thread.is_alive()
    return job


# ---- off, and what never wakes it ---------------------------------------------------------------

def test_without_a_secret_the_app_is_as_before(monkeypatch):
    monkeypatch.setattr(gpu, "broker", gpu.Broker(None))
    c = TestClient(server.app)
    assert "gpu" not in c.get("/api/info").json()
    assert c.get("/api/gpu").status_code == 404
    assert c.post("/api/gpu/wake", json={"reason": "button"}).status_code == 404
    assert c.get(gp.WANTED_PATH, headers={gp.TOKEN_HEADER: token(role="laptop", wake="laptop-1")}).status_code == 403
    assert c.get("/api/health").json()["gpu"] is None
    with pytest.raises(WebSocketDisconnect):  # refused before it is accepted: HTTP 403
        Worker(c, token())
    assert gpu.broker.accepting() == gpu.NOT_SET_UP and not gpu.broker.live_offered()


def test_the_broker_from_the_environment(monkeypatch, capsys):
    for k in ("STUTTERCLEAR_GPU_SECRET", "STUTTERCLEAR_GPU_LAUNCHER", "STUTTERCLEAR_PUBLIC_URL", "STUTTERCLEAR_GPU_ROLES"):
        monkeypatch.delenv(k, raising=False)
    assert not gpu.Broker.from_env().configured()
    monkeypatch.setenv("STUTTERCLEAR_GPU_SECRET", "too short")
    assert not gpu.Broker.from_env().configured() and "shorter than 32 characters" in capsys.readouterr().out
    monkeypatch.setenv("STUTTERCLEAR_GPU_SECRET", "﻿" + SECRET + "\r\n")  # as PowerShell stores it
    b = gpu.Broker.from_env()  # no launcher: only a standby laptop can serve
    assert b.configured() and b.launcher is None and b.roles == {"laptop"}
    assert b.status()["state"] == "unavailable" and b.status()["reason"] == gpu.NOT_CONNECTED
    monkeypatch.setenv("STUTTERCLEAR_GPU_LAUNCHER", "kaggle")  # without its credentials: on, but broken
    b = gpu.Broker.from_env()
    assert b.launcher is None and b.roles == {"kaggle", "laptop"} and b.status()["reason"] == gpu.BROKEN
    with pytest.raises(gpu.WakeRefused) as e:
        b.wake("button", "1.1.1.1")
    assert e.value.code == 409 and str(e.value) == gpu.BROKEN
    out = capsys.readouterr().out
    assert f"worker secret key id {gp.key_id(SECRET)}" in out and "kaggle launcher is not set up" in out
    assert SECRET not in out


def test_the_local_launcher_from_the_environment(monkeypatch):
    pytest.importorskip("stutterclear.gpu_worker.local")
    monkeypatch.setenv("STUTTERCLEAR_GPU_SECRET", SECRET)
    monkeypatch.setenv("STUTTERCLEAR_GPU_LAUNCHER", "local")
    monkeypatch.delenv("STUTTERCLEAR_GPU_ROLES", raising=False)
    monkeypatch.delenv("STUTTERCLEAR_PUBLIC_URL", raising=False)
    assert gpu.Broker.from_env().status()["reason"] == gpu.BROKEN  # it needs to know where it is
    monkeypatch.setenv("STUTTERCLEAR_PUBLIC_URL", "http://127.0.0.1:8801")
    b = gpu.Broker.from_env()
    assert b.launcher.name == "local" and b.roles == {"local", "laptop"} and b.status()["state"] == "off"
    assert b.public_url == "http://127.0.0.1:8801"


def test_looking_never_wakes_it(env):
    for _ in range(3):
        for path in ("/api/info", "/api/gpu", "/api/health", "/"):
            env.client.get(path)
    assert env.launcher.calls == [] and env.b.status()["state"] == "off"


def test_two_visitors_waking_at_once_start_one_machine(env):
    replies = []
    threads = [threading.Thread(target=lambda i=i: replies.append(env.b.wake("button", f"10.0.0.{i}")))
               for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(env.launcher.calls) == 1 and {r["state"] for r in replies} == {"waking"}
    wake, tok, reason, retired = env.launcher.calls[0]
    assert gp.check_token(SECRET, tok, "testserver", T0) == gp.Token("kaggle", wake, int(T0 + 4 * 3600))
    assert reason == "button" and retired == frozenset()


def test_launches_are_rationed(env):
    b = make(env.clock, env.launcher, wakes_per_hour=4, wakes_per_day=10, client_wakes_per_day=3)
    env.launcher.state = gp.Launch("failed", "The GPU machine stopped.", retry_s=60)  # each run ends at its first look

    def wake(client, after=400.0):
        env.clock.t += after  # past the last failure's backoff
        s = b.wake("button", client)
        env.clock.t += 11
        b.tick()
        return s

    for _ in range(3):
        wake("1.1.1.1")
    with pytest.raises(gpu.WakeRefused) as e:  # one visitor: 3 a day
        wake("1.1.1.1")
    assert e.value.code == 429 and "many times today" in str(e.value)
    wake("2.2.2.2")  # the 4th launch this hour
    with pytest.raises(gpu.WakeRefused) as e:
        wake("3.3.3.3")
    assert e.value.code == 429 and str(e.value) == gpu.WAKES_HOUR
    env.clock.t += 3600  # a new hour, the same day
    for i in range(6):  # 10 a day in all
        wake(f"4.4.4.{i}", after=960)
    with pytest.raises(gpu.WakeRefused) as e:
        wake("5.5.5.5", after=960)
    assert e.value.code == 429 and "many times today" in str(e.value)
    assert len(env.launcher.calls) == 10


def test_only_runs_started_here_count_against_the_caps(env):
    b = make(env.clock, env.launcher, wakes_per_hour=2, client_wakes_per_day=1)
    env.launcher.result = gp.Launch("failed", "Kaggle did not answer.", retry_s=60)  # nothing was started
    for _ in range(3):
        env.clock.t += 400
        b.wake("button", "1.1.1.1")
    env.launcher.result = gp.Launch("running", ref="owner/worker v7", version=7, adopted=True)  # found, not started
    env.clock.t += 400
    b.wake("button", "1.1.1.1")
    assert len(env.launcher.calls) == 4 and not b._launches and not b._client_launches.get("1.1.1.1")


def test_a_visitors_daily_limit_never_fails_other_visitors_recordings(env):
    env.b.s.client_wakes_per_day = 1
    env.launcher.state = gp.Launch("done", version=1)  # once its worker is gone, its machine is too
    first = analysing(env.b, audio(seed=1), "1.1.1.1")
    wait_for(lambda: env.launcher.calls)  # this visitor's one start today
    second = analysing(env.b, audio(seed=2), "2.2.2.2")
    w = Worker(env.client, env.launcher.calls[0][1])
    w.hello()
    w.caps()
    w.job()
    w.close()  # it crashed with the first recording
    wait_for(lambda: env.b.status()["step"] == "reconnecting")
    env.clock.t += 6
    env.b.tick()  # its machine is gone: the recordings waiting get a fresh worker, whoever woke the last one
    assert len(env.launcher.calls) == 2 and env.b.status()["state"] == "waking"
    assert first.error is None and second.error is None
    with env.b._lock:
        env.b._fail_queued("the end of the test")
    done(first), done(second)


def test_each_visitor_gets_a_fair_share(env):
    b = env.b
    assert b.admit("1.1.1.1", 30) is None
    assert b.admit("1.1.1.1", 5) == gpu.CLIENT_BUSY  # one recording at a time
    assert b.admit("2.2.2.2", 601) == gpu.CLIENT_AUDIO  # ten minutes of audio an hour
    assert b.admit("2.2.2.2", 300, dry_run=True) is None and b.admit("2.2.2.2", 300, dry_run=True) is None
    assert b.admit("2.2.2.2", 400) is None
    env.clock.t += gpu.RESERVE_S + 1  # its turn is no longer held (it never got queued)
    assert b.admit("2.2.2.2", 300) == gpu.CLIENT_AUDIO
    env.clock.t += 3600
    assert b.admit("2.2.2.2", 300) is None and b.admit("1.1.1.1", 30) is None


# ---- a recording through the worker ---------------------------------------------------------------

def test_a_recording_goes_to_the_worker_and_back(env):
    x = audio(3.0)
    job = analysing(env.b, x)
    wait_for(lambda: env.launcher.calls)  # the job woke it
    assert env.b.status()["state"] == "waking" and env.b.status()["ahead"] == 1
    wake, tok, reason, _ = env.launcher.calls[0]
    assert reason == "job"
    w = Worker(env.client, tok)
    w.hello()
    assert env.b.status()["state"] == "connecting" and env.b.status()["step"] == "loading"
    w.caps()
    h, sent = w.job()
    assert h["samples"] == len(x) and h["parts"] == gp.n_parts(len(x)) and h["language"] == "en"
    assert np.array_equal(sent, gp.from_pcm16(gp.pcm16(x)))  # what a 16-bit WAV holds goes through exactly
    assert env.b.status()["state"] == "busy"
    w.send(gp.PROGRESS, job=h["job"], stage="listening")
    wait_for(lambda: ("listening", None) in job.updates)
    w.send(gp.PROGRESS, job=h["job"], stage="transcribing")
    wait_for(lambda: ("transcribing", None) in job.updates)
    w.result(h, slow=True)
    probs, words, slow = done(job).result
    assert probs.shape == (gp.frames_for(len(x)), gp.N_CLASSES) and probs[15, 1] == pytest.approx(0.9)
    assert words == WORDS and slow is True and job.updates[0] == ("queued", 0)
    s = env.b.status()
    assert s["state"] == "ready" and s["where"] == "Kaggle (2× Tesla T4)" and s["sleeps_in_s"] == 600


def test_recordings_wait_their_turn(env):
    w = ready_worker(env)
    first, second = analysing(env.b, audio(seed=1), "1.1.1.1"), analysing(env.b, audio(seed=2), "2.2.2.2")
    h1, _ = w.job()
    wait_for(lambda: ("queued", 1) in second.updates)  # one ahead of it
    w.result(h1)
    h2, _ = w.job()
    w.result(h2)
    assert done(first).result and done(second).result and h1["job"] != h2["job"]


def test_a_dropped_worker_gets_the_recording_again_once(env):
    w = ready_worker(env)
    job = analysing(env.b, audio())
    h, _ = w.job()
    w.close()  # dropped mid-recording, without a goodbye
    wait_for(lambda: env.b.status()["step"] == "reconnecting")
    w = Worker(env.client, w.token)  # back within the grace
    w.hello()
    w.caps()
    h2, _ = w.job()
    assert h2["job"] == h["job"]
    w.result(h2)
    assert done(job).result is not None
    job = analysing(env.b, audio(seed=3))
    w.job()
    w.close()
    wait_for(lambda: env.b.status()["step"] == "reconnecting")
    w = Worker(env.client, w.token)
    w.hello()
    w.caps()
    w.job()
    w.close()  # lost a second time
    assert done(job).error == gpu.STOPPED_TWICE


def test_the_same_worker_back_before_its_old_socket_is_seen_dead_takes_over(env):
    w = ready_worker(env)
    job = analysing(env.b, audio())
    h, _ = w.job()
    again = Worker(env.client, w.token)  # the same worker on a fresh socket: a stall neither side heard end
    again.hello()  # welcome: not "another worker is active"
    with pytest.raises(WebSocketDisconnect) as e:
        w.recv()
    assert e.value.code == 1001  # its old socket is let go
    again.caps()
    h2, _ = again.job()  # the recording it had goes again
    assert h2["job"] == h["job"]
    again.result(h2)
    assert done(job).result is not None and env.b.status()["state"] == "ready"
    assert env.b.status()["sleeps_in_s"] == 600 and not env.b._crashes
    again.close()


def test_a_recording_taken_back_before_its_turn_sends_nothing(env):
    job = gpu.Job("job-1", audio(), 32000, "", T0)
    frames = env.b._job_frames(job, job.x, job.seq)
    job.seq += 1  # taken back (a timeout) while the connection was still busy with something else
    assert list(frames) == []


def test_a_recording_that_never_reached_a_gpu_is_not_counted(env):
    env.launcher.result = gp.Launch("failed", "Kaggle did not answer.", retry_s=60)
    for _ in range(4):
        assert env.b.admit("1.1.1.1", 180) is None  # four times three minutes: more than an hour's share
        job = analysing(env.b, audio(), "1.1.1.1")
        assert done(job).error == "Kaggle did not answer."
        env.clock.t += 400


def test_a_visitors_audio_is_counted_per_day_too(env):
    b = env.b
    for _ in range(3):
        assert b.admit("1.1.1.1", 600) is None
        b._take_reservation("1.1.1.1")  # (queued)
        env.clock.t += 3600
    assert b.admit("1.1.1.1", 1) == gpu.CLIENT_AUDIO_DAY and b.admit("2.2.2.2", 1) is None
    env.clock.t += 86400 - 3 * 3600 + 1
    assert b.admit("1.1.1.1", 600) is None


def test_a_broken_result_is_asked_for_again_once(env):
    w = ready_worker(env)
    job = analysing(env.b, audio())
    h, _ = w.job()
    w.result(h, frames=gp.frames_for(h["samples"]) - 1)  # the wrong number of frames
    h2, _ = w.job()
    assert h2["job"] == h["job"]
    w.result(h2, words=[{"w": "x", "s": 0.0, "e": 9999.0}])  # a word long after the recording ended
    assert done(job).error == gpu.CANNOT_CLEAN


def test_worker_errors(env):
    w = ready_worker(env)
    job = analysing(env.b, audio())
    h, _ = w.job()
    w.send(gp.ERROR, job=h["job"], message="CUDA out of memory", retry=True)
    h2, _ = w.job()  # worth another try
    w.send(gp.ERROR, job=h2["job"], message="CUDA out of memory", retry=True)
    assert done(job).error == gpu.CANNOT_CLEAN
    job = analysing(env.b, audio())
    h, _ = w.job()
    w.send(gp.ERROR, job=h["job"], message="bad audio", retry=False)
    assert done(job).error == gpu.CANNOT_CLEAN
    w.send(gp.RESULT, gp.probs_bytes(np.zeros((1, 5))), job="job-nope", frames=1, words=[], slow=False, timing={})
    assert env.b.status()["state"] == "ready"  # a result for no job is ignored


def test_a_worker_without_the_recordings_engine_says_so(env):
    w = Worker(env.client, token())
    w.hello()
    job = analysing(env.b, audio())  # it waits: the worker is still loading
    wait_for(lambda: job.updates)
    w.caps(notes=False, live=True)  # all it will load
    assert done(job).error == gpu.NO_NOTES and env.b.accepting("1.1.1.1") == gpu.NO_NOTES
    w.close()


def test_a_recording_the_worker_sits_on_is_taken_back(env):
    w = ready_worker(env)
    job = analysing(env.b, audio(2.0))
    h, _ = w.job()
    env.clock.t += 120 + 1.5 * 2 + 1
    env.b.tick()
    cancel, _ = w.recv()
    assert cancel == {"t": gp.CANCEL, "job": h["job"]}
    h2, _ = w.job()  # sent again
    w.result(h2)
    assert done(job).result is not None


def test_malformed_messages_close_the_connection_and_requeue_the_work(env):
    w = ready_worker(env)
    job = analysing(env.b, audio())
    w.job()
    w.ws.send_text("hello?")  # every message is a binary frame
    with pytest.raises(WebSocketDisconnect) as e:
        w.recv()
    assert e.value.code == gp.CLOSE_BAD
    wait_for(lambda: env.b.status()["step"] == "reconnecting")
    w = Worker(env.client, w.token)
    w.hello()
    w.caps()
    h, _ = w.job()
    w.ws.send_bytes(b"\x00\x00\x00\x05{nope")
    with pytest.raises(WebSocketDisconnect) as e:
        w.recv()
    assert e.value.code == gp.CLOSE_BAD and done(job).error == gpu.STOPPED_TWICE


def test_a_recording_nobody_waits_for_is_dropped(env):
    gone = [False]
    job = analysing(env.b, audio(), abandoned=lambda: gone[0])
    wait_for(lambda: job.updates)
    gone[0] = True  # the page stopped asking while it waited for a worker
    assert done(job).error == gpu.ABANDONED
    job = analysing(env.b, audio(), abandoned=lambda: False)
    wait_for(lambda: job.updates)
    env.clock.t += gpu.Settings().note_wait_s + 1
    assert done(job).error == gpu.TOO_LATE


# ---- waking and failing ---------------------------------------------------------------------------

def test_a_wake_shows_its_steps(env):
    assert env.b.wake("voice-note", "1.1.1.1")["step"] == "asking"  # the answer comes before the launcher's
    s = env.b.status()
    assert (s["state"], s["step"], s["where"], s["elapsed_s"]) == ("waking", "queued", "Kaggle", 0)
    env.clock.t += 15
    env.b.tick()  # polls the run: Kaggle started it
    s = env.b.status()
    assert (s["step"], s["elapsed_s"], s["can_wake"]) == ("starting", 15, False) and env.launcher.polls == 1
    assert env.b.wake("button", "2.2.2.2")["step"] == "starting" and len(env.launcher.calls) == 1  # a no-op


def test_a_run_already_going_is_adopted(env):
    env.launcher.result = gp.Launch("running", ref="owner/worker v7", version=7, adopted=True)
    env.b.wake("button", "1.1.1.1")
    assert env.b.status()["step"] == "starting"
    w = Worker(env.client, token(wake="0dd0ld0001"))  # its worker has an older wake's token
    w.hello()


def test_a_fresh_run_admits_only_its_own_worker(env):
    env.b.wake("button", "1.1.1.1")
    old = Worker(env.client, token(wake="0dd0ld0001"))  # an older run's worker, still around
    old.send(gp.HELLO, proto=1, worker="feedface", wake="0dd0ld0001", role="kaggle", where="Kaggle", commit=COMMIT,
             gpus=[], deadline=None, loading=True)
    with pytest.raises(WebSocketDisconnect) as e:
        old.recv()
    assert e.value.code == gp.CLOSE_BUSY
    laptop = Worker(env.client, token(role="laptop", wake="laptop-0001"))  # the laptop is always welcome
    laptop.hello(where="the team's laptop")
    assert 1 in env.b._retired  # that run will be turned away: never adopted again


def test_an_adopted_run_that_ends_is_replaced_once(env):
    env.launcher.result = gp.Launch("running", ref="owner/worker v7", version=7, adopted=True)
    env.b.wake("button", "1.1.1.1")
    env.launcher.state = gp.Launch("done", ref="owner/worker v7", version=7)  # it was on its way out
    env.launcher.result = gp.Launch("queued", ref="owner/worker v8", version=8)
    env.clock.t += 11
    env.b.tick()
    assert len(env.launcher.calls) == 2 and env.launcher.calls[1][3] == {7}  # a fresh push, never v7 again
    assert env.b.status()["step"] == "queued"
    env.launcher.state = gp.Launch("failed", "The GPU machine stopped: out of memory.", version=8, retry_s=60)
    env.clock.t += 11
    env.b.tick()  # our own fresh run failed: no third push
    s = env.b.status()
    assert len(env.launcher.calls) == 2 and s["state"] == "unavailable" and s["reason"].startswith("The GPU machine")


def test_a_job_after_the_worker_left_starts_a_fresh_run(env):
    env.b.wake("button", "1.1.1.1")
    w = Worker(env.client, env.launcher.calls[0][1])
    w.hello()
    w.caps()
    wait_for(lambda: env.b.status()["state"] == "ready")
    env.clock.t += gpu.Settings().idle_unused_s + 1  # nobody used it
    env.b.tick()
    shutdown, _ = w.recv()
    assert shutdown == {"t": gp.SHUTDOWN, "reason": "idle"}
    assert w.bye() == gp.CLOSE_SHUTDOWN
    assert env.b.status()["state"] == "off"
    with pytest.raises(WebSocketDisconnect):  # that wake ended here: its token is no good any more
        Worker(env.client, w.token)
    env.launcher.result = gp.Launch("queued", ref="owner/worker v2", version=2)
    job = analysing(env.b, audio())
    wait_for(lambda: len(env.launcher.calls) == 2)
    assert env.launcher.calls[1][3] == {1}  # the run told to stop is never adopted
    w = Worker(env.client, env.launcher.calls[1][1])
    w.hello()
    w.caps()
    h, _ = w.job()
    w.result(h)
    assert done(job).result is not None


def test_a_worker_that_dies_before_it_says_hello_is_seen_at_the_first_look(env):
    env.launcher.result = gp.Launch("running", ref="pid 4242")  # the local launcher: no version number
    env.launcher.state = gp.Launch("failed", "The worker exited with code 1.", ref="pid 4242", retry_s=60)
    job = analysing(env.b, audio())
    wait_for(lambda: env.launcher.calls)
    env.clock.t += 11
    env.b.tick()  # not twelve minutes later
    assert done(job).error == "The worker exited with code 1." and env.launcher.polls == 1


def test_a_run_told_to_stop_is_never_adopted_even_after_its_worker_reconnected(env):
    env.b.wake("button", "1.1.1.1")
    w = Worker(env.client, env.launcher.calls[0][1])
    w.hello()
    w.caps()
    wait_for(lambda: env.b.status()["state"] == "ready")
    w.close()  # every idle worker takes a fresh socket at 50 minutes (rotation)
    w = Worker(env.client, w.token)
    w.hello()
    w.caps()
    wait_for(lambda: env.b.status()["state"] == "ready")
    env.clock.t += gpu.Settings().idle_unused_s + 1
    env.b.tick()
    assert w.recv()[0] == {"t": gp.SHUTDOWN, "reason": "idle"}
    w.bye()
    assert 1 in env.b._retired


def test_a_wake_for_work_a_draining_worker_cannot_take_never_finds_that_run(env):
    env.b.wake("button", "1.1.1.1")
    w = Worker(env.client, env.launcher.calls[0][1])
    w.hello(deadline=T0 + 700)  # its machine stops in 700 s
    w.caps()
    env.clock.t += 101  # within DRAIN_S of that
    job = analysing(env.b, audio())
    wait_for(lambda: len(env.launcher.calls) == 2)
    assert 1 in env.launcher.calls[1][3]  # a fresh run, never the one on its way out
    with env.b._lock:
        env.b._fail_queued("the end of the test")
    done(job)
    w.close()


def test_a_used_worker_sleeps_ten_minutes_after_its_last_work(env):
    w = ready_worker(env)
    job = analysing(env.b, audio())
    h, _ = w.job()
    w.result(h)
    done(job)
    env.clock.t += 599
    env.b.tick()
    assert env.b.status()["sleeps_in_s"] == 1
    env.clock.t += 2
    env.b.tick()
    assert w.recv()[0] == {"t": gp.SHUTDOWN, "reason": "idle"}


def test_kaggle_with_no_free_gpu(env):
    job = analysing(env.b, audio())
    wait_for(lambda: env.launcher.calls)
    env.launcher.state = gp.Launch("queued", version=1)
    env.clock.t += 721
    env.b.tick()
    assert done(job).error == "Kaggle had no free GPU for 12 minutes."
    s = env.b.status()
    assert s["state"] == "unavailable" and s["retry_s"] == 300 and not s["can_wake"]
    w = Worker(env.client, env.launcher.calls[0][1])  # it turned up late after all: welcome
    w.hello()
    w.caps()
    wait_for(lambda: env.b.status()["state"] == "ready")


@pytest.mark.parametrize("launch,retry_s", [
    (gp.Launch("failed", "Kaggle refused our request.", retry_s=None), 3600),  # an operator must fix it
    (gp.Launch("failed", "Kaggle did not answer.", retry_s=60), 60),  # worth trying again soon
])
def test_failed_launches(env, launch, retry_s):
    env.launcher.result = launch
    job = analysing(env.b, audio())
    assert done(job).error == launch.reason
    s = env.b.status()
    assert s["state"] == "unavailable" and s["reason"] == launch.reason and s["retry_s"] == retry_s
    r = env.client.post("/api/gpu/wake", json={"reason": "button"})
    assert r.status_code == 409 and r.json()["detail"].startswith(launch.reason) and "Try again in" in r.json()["detail"]


def test_the_backoff_grows_with_failures_in_a_row(env):
    env.launcher.result = gp.Launch("failed", "Kaggle did not answer.", retry_s=60)
    waits = []
    for _ in range(4):
        env.b.wake("button", "1.1.1.1")
        waits.append(env.b.status()["retry_s"])
        env.clock.t += waits[-1]
    assert waits == [60, 120, 300, 300]


def test_a_worker_that_cannot_load_its_models(env):
    w = Worker(env.client, token())
    w.hello()
    env.clock.t += 361
    env.b.tick()
    assert w.recv()[0] == {"t": gp.SHUTDOWN, "reason": "error"}
    assert w.bye("shutdown") == gp.CLOSE_SHUTDOWN
    assert env.b.status()["reason"] == gpu.LOAD_TIMEOUT
    env.clock.t += 301
    w = Worker(env.client, token(wake="5ec0ndwake"))
    w.hello()
    w.caps(notes=False, error="This machine has no GPU.")
    assert w.recv()[0] == {"t": gp.SHUTDOWN, "reason": "error"}
    w.bye("error")
    s = env.b.status()
    assert s["state"] == "unavailable" and s["reason"] == gpu.LOAD_FAILED and s["retry_s"] == 3600


def test_a_worker_that_does_not_come_back(env):
    env.launcher.state = gp.Launch("done", version=1)
    w = ready_worker(env)
    w.close()
    assert env.b.status()["step"] == "reconnecting"
    env.clock.t += 91
    env.b.tick()  # its machine is gone too
    assert env.b.status()["state"] == "off" and env.launcher.polls == 1
    w = ready_worker(env, token(wake="5ec0ndwake"))
    w.close()
    env.clock.t += 91
    env.b.tick()  # a second time within half an hour
    assert env.b.status()["reason"] == gpu.KEEPS_FAILING


def test_a_dropped_worker_whose_machine_still_runs_gets_more_time(env):
    w = ready_worker(env)
    w.close()
    env.clock.t += 91
    env.b.tick()
    assert env.b.status()["step"] == "reconnecting"  # still running: up to five more minutes
    env.clock.t += 290
    env.b.tick()
    assert env.launcher.polls == 2  # looked at again meanwhile
    w = Worker(env.client, w.token)
    w.hello()
    w.caps()
    wait_for(lambda: env.b.status()["state"] == "ready")


def test_a_machine_that_ends_during_the_extra_wait_is_given_up_at_once(env):
    w = ready_worker(env)
    w.close()
    env.clock.t += 91
    env.b.tick()  # running: five more minutes
    env.launcher.state = gp.Launch("failed", "The GPU machine stopped: out of memory.", version=1, retry_s=60)
    env.clock.t += 31
    env.b.tick()  # the next look finds it gone
    assert env.b.status()["state"] == "off" and env.launcher.polls == 2


def test_a_worker_whose_machine_died_with_it_is_not_waited_for(env):
    env.launcher.state = gp.Launch("failed", "The worker exited with code 1.", version=1, retry_s=60)
    w = ready_worker(env)
    job = analysing(env.b, audio())
    w.job()
    w.close()  # it crashed mid-recording, and its machine (a process) is gone too
    wait_for(lambda: env.b.status()["step"] == "reconnecting")
    env.clock.t += 6
    env.b.tick()  # the first look, seconds after the drop: no 90 s wait, a fresh worker for the recording
    assert env.launcher.polls == 1 and len(env.launcher.calls) == 1
    w = Worker(env.client, env.launcher.calls[0][1], worker="0ddba11")
    w.hello()
    w.caps()
    h, _ = w.job()
    w.result(h)
    assert done(job).result is not None
    w.close()


def test_the_first_look_at_a_running_machine_waits_out_the_grace(env):
    w = ready_worker(env)
    w.close()
    wait_for(lambda: env.b.status()["step"] == "reconnecting")
    env.clock.t += 6
    env.b.tick()  # running: its worker may come back (Kaggle's bootstrap restarts a crashed one)
    assert env.b.status()["step"] == "reconnecting" and env.launcher.polls == 1
    env.clock.t += 30
    env.b.tick()  # no new look before the grace is over
    assert env.launcher.polls == 1
    env.clock.t += 60
    env.b.tick()  # the grace is over and it still runs: five more minutes
    assert env.b.status()["step"] == "reconnecting" and env.launcher.polls == 2
    env.clock.t += 200
    env.b.tick()
    assert env.b.status()["step"] == "reconnecting"


def test_a_worker_back_from_a_drop_keeps_its_idle_time(env):
    w = ready_worker(env)
    job = analysing(env.b, audio())
    h, _ = w.job()
    w.result(h)
    done(job)
    env.clock.t += 500
    w.close()  # as a rotation: a fresh socket, the same worker
    w = Worker(env.client, w.token)
    w.hello()
    w.caps()
    wait_for(lambda: env.b.status()["state"] == "ready")
    assert env.b.status()["sleeps_in_s"] == 100


def test_a_worker_near_its_time_limit_takes_no_new_work(env):
    w = ready_worker(env, deadline=T0 + 700)  # its machine stops in 700 s
    env.clock.t += 101  # within DRAIN_S of it
    job = analysing(env.b, audio())
    wait_for(lambda: env.launcher.calls)  # a fresh machine for the waiting work
    w2 = Worker(env.client, env.launcher.calls[0][1], worker="0ddba11")
    w2.hello()  # welcome: the old one takes no work
    w2.caps()
    h, _ = w2.job()
    w2.result(h)
    assert done(job).result is not None
    w.close()


def test_a_restarted_server_adopts_the_running_worker(env):
    w = ready_worker(env, token(wake="ab0de0f1"))  # no wake here: it was started by the instance before
    assert env.launcher.calls == [] and env.b.status()["where"] == "Kaggle (2× Tesla T4)"
    w.close()


def test_the_server_stopping_says_nothing_to_the_worker(env, monkeypatch):
    monkeypatch.setattr(server, "warm_demo", server.notes.warm.set)  # no models; voice notes may go at once
    stops = []
    monkeypatch.setattr(env.b, "_stop", lambda *a: stops.append(a))
    with TestClient(server.app) as c:  # the startup and shutdown run
        w = Worker(c, token())
        w.hello()
        w.caps()
        w.close()
    assert stops == []  # the worker reconnects to the next instance, which adopts it


# ---- who may connect ------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    "nonsense", token(secret="t" * 40), token(server_url="https://elsewhere.example"),
    token(ttl=-10), token(role="local"),
], ids=["junk", "other-secret", "other-server", "expired", "role-not-allowed"])
def test_bad_tokens_are_refused_before_accepting(env, bad):
    with pytest.raises(WebSocketDisconnect) as e:
        Worker(env.client, bad)
    assert e.value.code == 1008


def test_a_second_worker_is_turned_away(env):
    w = ready_worker(env)
    other = Worker(env.client, token(wake="0th3rwake"), worker="f00dbabe")
    other.send(gp.HELLO, proto=1, worker="f00dbabe", wake="0th3rwake", role="kaggle", where="Kaggle", commit=COMMIT,
               gpus=[], deadline=None, loading=True)
    with pytest.raises(WebSocketDisconnect) as e:
        other.recv()
    assert e.value.code == gp.CLOSE_BUSY and env.b.status()["where"] == "Kaggle (2× Tesla T4)"
    w.close()


@pytest.mark.parametrize("role,change,code", [
    ("kaggle", {"proto": 2}, gp.CLOSE_VERSION),
    ("kaggle", {"commit": "d" * 40}, gp.CLOSE_VERSION),  # a launched worker must run the server's code
    ("kaggle", {"role": "laptop"}, gp.CLOSE_BAD),  # not what its token says
    ("laptop", {"commit": "d" * 40}, None),  # the laptop may differ
    ("kaggle", {"commit": "dev"}, None),
])
def test_hello_is_checked(env, role, change, code):
    tok = token(role=role, wake="laptop-1234" if role == "laptop" else "9c1e5a7b2d3f4e60")
    w = Worker(env.client, tok)
    hello = {"proto": 1, "worker": "0badcafe", "wake": gp.peek_token(tok).wake, "role": role, "where": "x",
             "commit": COMMIT, "gpus": [], "deadline": None, "loading": True} | change
    w.send(gp.HELLO, **hello)
    if code is None:
        assert w.recv()[0]["t"] == gp.WELCOME
        w.close()
        return
    with pytest.raises(WebSocketDisconnect) as e:
        w.recv()
    assert e.value.code == code


def test_a_hello_must_come_first_and_soon(env, monkeypatch):
    w = Worker(env.client, token())
    w.send(gp.CAPS, notes=True, live=False, loading=False, window_ms=None, models={}, error=None)
    with pytest.raises(WebSocketDisconnect) as e:
        w.recv()
    assert e.value.code == gp.CLOSE_BAD
    monkeypatch.setattr(gp, "HELLO_S", 0.05)
    w = Worker(env.client, token())
    with pytest.raises(WebSocketDisconnect) as e:
        w.recv()
    assert e.value.code == gp.CLOSE_BAD


# ---- the team's laptop -----------------------------------------------------------------------------

def test_a_standby_laptop_is_woken_first(env):
    lt = token(role="laptop", wake="laptop-aaaa1111")
    r = env.client.get(gp.WANTED_PATH, headers={gp.TOKEN_HEADER: lt})
    assert r.json() == {"wanted": False} and r.headers["cache-control"] == "no-store"
    assert env.b.status()["standby"] is True
    s = env.b.wake("voice-note", "1.1.1.1")
    assert (s["state"], s["step"], s["where"]) == ("waking", "laptop", "the team's laptop")
    assert env.client.get(gp.WANTED_PATH, headers={gp.TOKEN_HEADER: lt}).json() == {"wanted": True}
    w = Worker(env.client, lt)
    w.hello(where="the team's laptop")
    w.caps()
    wait_for(lambda: env.b.status()["state"] == "ready")
    assert env.launcher.calls == []  # free, and no launch counted
    assert env.client.get(gp.WANTED_PATH, headers={gp.TOKEN_HEADER: lt}).json() == {"wanted": False}
    assert env.client.get(gp.WANTED_PATH, headers={gp.TOKEN_HEADER: token()}).status_code == 403  # not a laptop
    assert env.client.get(gp.WANTED_PATH).status_code == 403
    w.close()


def test_a_laptop_that_does_not_come_is_replaced_by_a_launch(env):
    env.client.get(gp.WANTED_PATH, headers={gp.TOKEN_HEADER: token(role="laptop", wake="laptop-aaaa1111")})
    job = analysing(env.b, audio())
    wait_for(lambda: env.b.status()["step"] == "laptop")
    env.clock.t += 41
    env.b.tick()
    assert len(env.launcher.calls) == 1 and env.b.status()["step"] == "queued"
    w = Worker(env.client, env.launcher.calls[0][1])
    w.hello()
    w.caps()
    h, _ = w.job()
    w.result(h)
    assert done(job).result is not None


def test_a_laptop_only_server(monkeypatch):
    clock = Clock()
    b = gpu.Broker(SECRET, None, gpu.Settings(poll_s=0.01), clock=clock, commit=COMMIT, autostart=False)
    monkeypatch.setattr(gpu, "broker", b)
    c = TestClient(server.app)
    assert b.status()["state"] == "unavailable" and b.accepting() == gpu.NOT_CONNECTED
    c.get(gp.WANTED_PATH, headers={gp.TOKEN_HEADER: token(role="laptop", wake="laptop-aaaa1111")})
    assert b.status()["state"] == "off" and b.status()["can_wake"] and b.accepting() is None
    b.wake("button", "1.1.1.1")
    clock.t += 41
    b.tick()  # it did not come, and nothing else can
    s = b.status()
    assert s["state"] == "unavailable" and s["reason"] == gpu.LAPTOP_SILENT and s["retry_s"] == 300


def laptop_polls(env, wake="laptop-aaaa1111") -> dict:
    return env.client.get(gp.WANTED_PATH, headers={gp.TOKEN_HEADER: token(role="laptop", wake=wake)}).json()


def test_laptop_wakes_without_work_are_rationed(env):
    b = env.b
    b.s.client_laptop_wakes_per_hour, b.s.laptop_wakes_per_hour = 2, 3

    def wake(client):
        laptop_polls(env)
        b._wake = None  # (the last wake is over)
        return b.wake("button", client)

    wake("1.1.1.1")
    wake("1.1.1.1")
    with pytest.raises(gpu.WakeRefused) as e:  # each one holds this server up a while: a few an hour each
        wake("1.1.1.1")
    assert e.value.code == 429 and str(e.value) == gpu.WAKES_HOUR and not b.status("1.1.1.1")["can_wake"]
    wake("2.2.2.2")
    with pytest.raises(gpu.WakeRefused):  # and a few more in all
        wake("3.3.3.3")
    b._wake = None
    job = analysing(b, audio(), "1.1.1.1")  # a recording to clean always wakes it
    wait_for(lambda: b.status()["step"] == "laptop")
    assert env.launcher.calls == []
    with b._lock:
        b._fail_queued("the end of the test")
    done(job)


@pytest.mark.parametrize("role,reason,unused_s", [("laptop", "button", 60), ("kaggle", "button", 240),
                                                  ("kaggle", "voice-note", 300), ("laptop", "video", 300)])
def test_how_soon_an_unused_worker_sleeps_depends_on_what_woke_it(env, role, reason, unused_s):
    if role == "laptop":
        laptop_polls(env)
    env.b.wake(reason, "1.1.1.1")
    w = Worker(env.client, token(role="laptop", wake="laptop-bbbb2222") if role == "laptop" else env.launcher.calls[0][1])
    assert w.hello()["idle_unused_s"] == unused_s  # (the worker's own idle timer goes by it too)
    w.caps()
    wait_for(lambda: env.b.status()["state"] == "ready")
    assert env.b.status()["sleeps_in_s"] == unused_s  # a Wake press alone lets go of this server soon; a
    env.clock.t += unused_s + 1  # recording on its way (Record pressed, a file chosen) keeps it longer
    env.b.tick()
    assert w.recv()[0] == {"t": gp.SHUTDOWN, "reason": "idle"}
    w.close()


def test_a_laptop_told_to_sleep_stands_by_at_once(env):
    laptop_polls(env)
    env.b.wake("voice-note", "1.1.1.1")
    w = Worker(env.client, token(role="laptop", wake="laptop-bbbb2222"))
    w.hello(where="the team's laptop")
    w.caps()
    wait_for(lambda: env.b.status()["state"] == "ready")
    env.clock.t += 301
    env.b.tick()
    assert w.recv()[0] == {"t": gp.SHUTDOWN, "reason": "idle"}
    assert w.bye() == gp.CLOSE_SHUTDOWN
    env.clock.t += 5  # a recording before its next poll: still the laptop, never a Kaggle start
    assert env.b.status()["standby"]
    job = analysing(env.b, audio())
    wait_for(lambda: env.b.status()["step"] == "laptop")
    assert env.launcher.calls == []
    with env.b._lock:
        env.b._fail_queued("the end of the test")
    done(job)


def test_a_laptop_that_lost_its_network_is_used_again_once_it_stands_by(env):
    for attempt in range(2):  # twice in half an hour: a laptop's network never makes the GPU "keep failing"
        laptop_polls(env)
        job = analysing(env.b, audio(seed=attempt), f"1.1.1.{attempt}")
        wait_for(lambda: env.b.status()["step"] == "laptop")
        w = Worker(env.client, token(role="laptop", wake=f"laptop-{attempt}aaa"))
        w.hello(where="the team's laptop")
        w.caps()
        h, _ = w.job()
        w.close()  # its Wi-Fi dropped mid-recording; it is back to standing by before the 90 s are over
        wait_for(lambda: env.b.status()["step"] == "reconnecting")
        assert laptop_polls(env) == {"wanted": True} and env.b.status()["step"] == "laptop"
        w = Worker(env.client, token(role="laptop", wake=f"laptop-{attempt}bbb"))
        w.hello(where="the team's laptop")
        w.caps()
        h2, _ = w.job()
        assert h2["job"] == h["job"]
        w.result(h2)
        assert done(job).result is not None
        w.close()
        env.clock.t += 600
    env.clock.t += 91
    env.b.tick()
    assert env.b.status()["state"] != "unavailable" and not env.b._crashes


def test_a_laptop_standing_by_is_used_while_kaggle_is_unavailable(env):
    env.launcher.result = gp.Launch("failed", "Kaggle refused our request.", retry_s=None)
    env.b.wake("button", "1.1.1.1")
    assert env.b.status()["state"] == "unavailable"  # for an hour: an operator must fix it
    laptop_polls(env)
    s = env.b.status("1.1.1.1")
    assert (s["state"], s["standby"], s["can_wake"]) == ("off", True, True)
    assert env.b.wake("button", "1.1.1.1")["step"] == "laptop" and len(env.launcher.calls) == 1


# ---- the weekly budget ------------------------------------------------------------------------------

def test_kaggles_quota_makes_it_rest(env):
    env.launcher.quota_left = 9.0  # 21 h used: past our 20 h cap
    job = analysing(env.b, audio())
    assert "rests until" in done(job).error and env.launcher.calls == []  # checked before any push
    s = env.b.status()
    assert s["state"] == "resting" and s["until"] == "2027-01-18T08:00:00Z" and s["week"] == {"used_h": 21.0, "cap_h": 20.0}
    r = env.client.post("/api/gpu/wake", json={"reason": "button"})
    assert r.status_code == 409 and "free GPU hours are used up" in r.json()["detail"]
    assert "rests until" in env.b.accepting("1.1.1.1")
    env.clock.t += 3 * 86400  # the weekly window resets
    env.launcher.quota_left = 30.0
    assert env.b.status()["state"] == "off"


def test_a_day_uses_at_most_its_share_of_the_week(env):
    env.b.s.day_hours = 5.0
    env.launcher.quota_left = 28.0  # 2 h used at the day's first reading
    env.b._refresh_quota()
    env.launcher.quota_left = 22.5  # 5.5 h more since: today's share is used up, the week's is not
    env.b._quota_at = -gpu.math.inf
    env.b._refresh_quota()
    with pytest.raises(gpu.WakeRefused) as e:
        env.b.wake("button", "1.1.1.1")
    assert e.value.code == 409 and str(e.value).startswith("Today's share of the free GPU hours is used up")
    s = env.b.status()
    assert (s["state"], s["step"], s["until"]) == ("resting", "day", "2027-01-16T00:00:00Z")  # the next UTC day
    env.clock.t += 16 * 3600
    assert env.b.status()["state"] == "off"


def test_a_worker_up_for_long_takes_no_new_work(env):
    env.b.s.max_wake_s = 1000
    w = ready_worker(env)
    env.clock.t += 500
    job = analysing(env.b, audio())
    h, _ = w.job()
    w.result(h)
    done(job)
    env.clock.t += 501  # used 501 s ago, but up for 1001 s: steady use cannot keep one machine up for hours
    env.b.tick()
    assert w.recv()[0] == {"t": gp.SHUTDOWN, "reason": "idle"}
    assert w.bye() == gp.CLOSE_SHUTDOWN
    job = analysing(env.b, audio())
    wait_for(lambda: env.launcher.calls)  # the next work wakes a fresh one, counted like any other
    assert len(env.b._launches) == 1
    with env.b._lock:
        env.b._fail_queued("the end of the test")
    done(job)


def test_a_quota_error_from_the_push_makes_it_rest(env):
    env.launcher.result = gp.Launch("quota", "This week's GPU hours are used up.", until=T0 + 86400)
    env.b.wake("button", "1.1.1.1")
    assert env.b.status()["state"] == "resting" and env.b.status()["until"] == "2027-01-16T08:00:00Z"


def test_past_the_budget_while_awake_it_finishes_then_rests(env, monkeypatch):
    env.b = make(env.clock, env.launcher, quota_cache_s=60)
    monkeypatch.setattr(gpu, "broker", env.b)
    env.launcher.quota_left = 12.0
    env.b.wake("button", "1.1.1.1")
    w = Worker(env.client, env.launcher.calls[0][1])
    w.hello()
    w.caps()
    job = analysing(env.b, audio())
    h, _ = w.job()
    env.launcher.quota_left = 9.0  # Kaggle's count passes our cap while it works
    env.clock.t += 61
    env.b.tick()  # reads the quota again
    w.result(h)
    assert done(job).result is not None  # the work in flight finishes
    env.b.tick()
    assert w.recv()[0] == {"t": gp.SHUTDOWN, "reason": "budget"}
    w.bye("shutdown")
    assert env.b.status()["state"] == "resting"


def test_the_laptop_serves_while_kaggle_rests(env):
    env.launcher.quota_left = 1.0
    env.b.wake("button", "1.1.1.1")
    assert env.b.status()["state"] == "resting"
    lt = token(role="laptop", wake="laptop-aaaa1111")
    env.client.get(gp.WANTED_PATH, headers={gp.TOKEN_HEADER: lt})
    assert env.b.accepting("1.1.1.1") is None  # it costs nothing
    s = env.b.status("1.1.1.1")  # so the page lets visitors record: it wakes in seconds
    assert (s["state"], s["standby"], s["can_wake"], s["reason"], s["where"]) == ("off", True, True, None, gpu.LAPTOP)
    job = analysing(env.b, audio())
    wait_for(lambda: env.b.status()["step"] == "laptop")
    w = Worker(env.client, lt)
    w.hello(where="the team's laptop")
    w.caps()
    h, _ = w.job()
    w.result(h)
    assert done(job).result is not None


# ---- Best (GPU) live through the worker -------------------------------------------------------------

@pytest.fixture
def live_env(env, monkeypatch):
    env.b = make(env.clock, env.launcher, live=True)
    monkeypatch.setattr(gpu, "broker", env.b)
    return env


def live_worker(env, window_ms=40.0) -> Worker:
    w = Worker(env.client, token())
    w.hello()
    w.caps(live=True, window_ms=window_ms)
    wait_for(lambda: env.b.status()["state"] == "ready")
    return w


def test_best_gpu_live_through_the_worker(live_env):
    env = live_env
    w = live_worker(env)
    s = env.b.status()["live"]
    assert s["available"] and s["delay_ms"] == {"gentle": 830, "strong": 1130}  # a 250 ms trip until pings say
    link = env.b.open_live("gentle")
    assert link.extra_delay_ms == 290 and env.b.status()["state"] == "busy"
    assert env.b.open_live("gentle") is None  # one stream at a time
    assert env.b.status()["live"]["reason"] == "Someone else is using Best (GPU) right now."
    assert link.start(200, 67200, 8, "gentle")
    assert w.recv()[0] == {"t": gp.LIVE_OPEN, "stream": link.stream, "mode": "gentle", "lookahead_ms": 200,
                           "window": 67200, "hop_frames": 8}
    sound = np.linspace(-0.5, 0.5, 4000, dtype=np.float32)
    assert link.send(0, 0, 4000, 0, 2, sound)
    h, payload = w.recv()
    assert (h["t"], h["seq"], h["f0"], h["f1"]) == (gp.LIVE_AUDIO, 0, 0, 2)
    assert np.allclose(gp.from_pcm16(payload), sound, atol=1e-4)
    rows = np.full((2, gp.N_CLASSES), 0.5, np.float32)
    w.send(gp.LIVE_PROBS, gp.probs_bytes(rows), stream=link.stream, seq=0, f0=0, f1=2, ms=12.5)
    kind, seq, f0, f1, got, ms = link.replies.get(timeout=5)
    assert (kind, seq, f0, f1, ms) == ("probs", 0, 0, 2, 12.5) and np.array_equal(got, rows)
    link.close()
    link.close()  # idempotent
    assert w.recv()[0] == {"t": gp.LIVE_CLOSE, "stream": link.stream}
    assert env.b.status()["live"]["available"] and not link.send(1, 0, 4000, 2, 4, sound)


def test_a_live_link_is_ready_once_the_workers_model_is_warm(live_env):
    env = live_env
    w = live_worker(env)
    link = env.b.open_live("gentle")
    link.start(200, 67200, 8, "gentle")
    w.recv()
    assert not link.ready.is_set()
    env.clock.t += 14  # warming (a teacher loaded again from disk): not silent yet
    env.b.tick()
    assert not link.closed
    w.send(gp.LIVE_READY, stream=link.stream)
    assert link.ready.wait(5) and link.ready_at == env.clock.t
    env.clock.t += 16  # ready, and no audio since: closed as silent
    env.b.tick()
    assert link.replies.get_nowait() == ("closed", "gone")
    waiting = env.b.open_live("gentle")
    waiting.start(200, 67200, 8, "gentle")
    w.recv()
    waiting.close()  # a session still waiting for it is told at once
    assert waiting.ready.is_set() and waiting.closed


def test_the_broker_closes_live_links_nobody_uses(live_env):
    env = live_env
    w = live_worker(env)
    never_started = env.b.open_live("gentle")  # its session failed to build
    env.clock.t += 6
    env.b.tick()
    assert never_started.replies.get_nowait() == ("closed", "gone") and env.b.status()["live"]["available"]
    silent = env.b.open_live("gentle")
    silent.start(200, 67200, 8, "gentle")
    w.recv()
    silent.send(0, 0, 4000, 0, 2, np.zeros(4000, np.float32))
    w.recv()
    env.clock.t += 16  # no audio since
    env.b.tick()
    assert silent.replies.get_nowait() == ("closed", "gone")
    assert w.recv()[0] == {"t": gp.LIVE_CLOSE, "stream": silent.stream}


def test_a_worker_that_drops_ends_its_live_stream(live_env):
    env = live_env
    w = live_worker(env)
    link = env.b.open_live("strong")
    link.start(500, 72000, 8, "strong")
    w.close()
    assert link.replies.get(timeout=5) == ("closed", "gone") and not link.send(0, 0, 4000, 0, 2, np.zeros(4000))


def test_a_worker_too_far_away_is_not_offered_for_live(live_env):
    env = live_env
    live_worker(env, window_ms=400.0)  # 250 ms trip + 340 ms of GPU beyond what the local delay allows
    s = env.b.status()["live"]
    assert not s["available"] and s["reason"] == "The GPU is too far away for live use right now."
    assert env.b.open_live("gentle") is None
    env.b.s.hosted = True  # a public server can take more: the voice is heard after the session
    assert env.b.open_live("gentle").extra_delay_ms == 630


def test_best_gpu_sessions_are_counted_per_visitor(live_env):
    env = live_env
    assert env.b.admit_live("1.1.1.1") is None and not env.b._client_live  # asleep: it runs rule-based, uncounted
    live_worker(env)
    for _ in range(3):
        assert env.b.admit_live("1.1.1.1") is None
    assert env.b.admit_live("1.1.1.1") == gpu.CLIENT_LIVE and env.b.admit_live("2.2.2.2") is None
    env.clock.t += 3601
    assert env.b.admit_live("1.1.1.1") is None


def test_live_windows_never_wait_behind_a_recording():
    loop = SimpleNamespace(time=lambda: 0.0)
    c = gpu.Conn(None, loop, gp.Token("kaggle", "abcd1234", 0),
                 {"worker": "w", "where": "x", "commit": "", "deadline": None}, 0.0)
    c.control.append(b"control")
    c.live_q.append(b"live")
    c.bulk.append(iter([b"part-1", b"part-2", b"part-3"]))
    assert [c._next(0.0) for _ in range(3)] == [b"control", b"live", b"part-1"]
    c.live = object()  # a live stream is open: one recording part per BULK_PACE_S
    assert c._next(0.0) == b"part-2" and c._next(0.01) is None
    c.live_q.append(b"live-2")
    assert c._next(0.01) == b"live-2" and c._next(gpu.BULK_PACE_S) == b"part-3" and c._next(1.0) is None


# ---- the HTTP API ---------------------------------------------------------------------------------

STATUS_KEYS = {"state", "step", "now", "since", "elapsed_s", "typical_s", "where", "ahead", "notes", "standby", "live",
               "sleeps_in_s", "until", "reason", "retry_s", "can_wake", "week"}


def test_the_gpu_api(env):
    c = env.client
    r = c.get("/api/gpu")
    s = r.json()
    assert r.headers["cache-control"] == "no-store" and set(s) == STATUS_KEYS
    assert (s["state"], s["can_wake"], s["typical_s"], s["live"]) == (
        "off", True, [120, 240], {"available": False, "delay_ms": None, "reason": None})
    assert c.post("/api/gpu/wake", json={"reason": "job"}).status_code == 422  # not a page's reason
    assert c.post("/api/gpu/wake", json={}).status_code == 422
    assert c.post("/api/gpu/wake", json={"reason": "button"}, headers={"Origin": "https://evil.example"}).status_code == 403
    r = c.post("/api/gpu/wake", json={"reason": "button", "page": "f" * 32}, headers={"Origin": "http://testserver"})
    assert r.status_code == 200 and r.json()["state"] == "waking" and set(r.json()) == STATUS_KEYS
    assert len(env.launcher.calls) == 1
    gi = c.get("/api/info").json()
    assert gi["gpu"]["where"] == "Kaggle" and gi["gpu"]["live"] is False and "idle_s" not in gi["gpu"]
    assert gi["version"] == gpu.version() and gi["video_max_mb"] == server.video.MAX_BYTES // 10**6
    teacher = next(d for d in gi["detectors"] if d["kind"] == "teacher")
    assert teacher["on_demand"] is False and teacher["delay_ms"] == {"gentle": 540, "strong": 840}


def test_health_answers_the_front_door(env):
    front = "https://pradyuman-aviator.github.io"
    r = env.client.get("/api/health", headers={"Origin": front})
    assert r.json() == {"ok": True, "service": "stutterclear", "version": gpu.version(), "hosted": server.HOSTED,
                        "gpu": "off", "live_free": True}
    assert r.headers["access-control-allow-origin"] == front and r.headers["cache-control"] == "no-store"
    assert "access-control-allow-origin" not in env.client.get("/api/health", headers={"Origin": "https://evil.example"}).headers
    assert gpu.version() == "dev" or len(gpu.version()) == 12


def test_logs_never_show_secrets_tokens_or_addresses(env, capsys):
    job = analysing(env.b, audio(), client="203.0.113.77")
    wait_for(lambda: env.launcher.calls)
    tok = env.launcher.calls[0][1]
    w = Worker(env.client, tok)
    w.hello()
    w.caps()
    h, _ = w.job()
    w.result(h)
    done(job)
    with pytest.raises(WebSocketDisconnect):
        Worker(env.client, token(secret="t" * 40))
    out = capsys.readouterr().out
    assert "gpu: " in out
    for secret in (SECRET, tok, tok.rsplit(".", 1)[1], "203.0.113.77", "note-"):
        assert secret not in out
