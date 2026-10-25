"""Best (GPU) live with the teacher on a GPU worker elsewhere (stutterclear/remote_detector.py): it cuts
exactly as the local teacher does, never stalls the voice and never raises, and hands over to the
rule-based detector when the worker drops or lags; and what keeps a browser-mic session safe (one at
a time, settings in time, frame and rate limits, a detector that fails)."""
import queue
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from stutterclear import captions, gpu_protocol as gp, live, teacher_runtime
from stutterclear.app import server
from stutterclear.config import HOP, SAMPLE_RATE as SR
from stutterclear.detector import CutUpdate
from stutterclear.remote_detector import NOTES, RemoteTeacherDetector
from stutterclear.segments import BALANCED_CUTS
from stutterclear.student_runtime import CLASSES
from stutterclear.teacher_runtime import FRAME, TeacherDetector


def stand_in_teacher(w: np.ndarray) -> np.ndarray:
    """A teacher stand-in: a repeated word wherever the audio is loud, per 20 ms frame of the window."""
    n = (len(w) - 400) // FRAME + 1
    frames = np.lib.stride_tricks.sliding_window_view(w, 400)[::FRAME][:n]
    p = np.full((n, len(CLASSES)), 0.02, np.float32)
    p[:, CLASSES.index("word_rep")] = np.where(np.sqrt((frames ** 2).mean(1)) > 0.2, 0.95, 0.02)
    return p


@pytest.fixture
def teacher(monkeypatch):
    """The local teacher without a model: window probabilities from the stand-in."""
    monkeypatch.setattr(teacher_runtime, "get_teacher", lambda *a, **k: (None, None))
    monkeypatch.setattr(teacher_runtime, "window_probs", stand_in_teacher)
    monkeypatch.setattr(captions, "available", lambda: False)


def boom(*a, **k):
    raise RuntimeError("boom")


def bursts(seconds: float = 20.0, seed: int = 0) -> np.ndarray:
    """Quiet room noise with a loud 300 ms burst every 2 s."""
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(int(seconds * SR)) * 0.01).astype(np.float32)
    for t in np.arange(1.0, seconds - 0.5, 2.0):
        a = int(t * SR)
        x[a:a + int(0.3 * SR)] = (rng.standard_normal(int(0.3 * SR)) * 0.5).astype(np.float32)
    return x


class EchoLink:
    """A worker that answers every window at once, from exactly the audio it was sent (it keeps
    what came before, as the real worker's ring does)."""

    def __init__(self, teach=stand_in_teacher, extra_delay_ms=300):
        self.teach, self.extra_delay_ms = teach, extra_delay_ms
        self.replies: queue.Queue = queue.Queue()
        self.audio = np.zeros(0, np.float32)
        self.started, self.closed, self.sent = None, False, []

    def start(self, lookahead_ms, window, hop_frames, mode):
        self.started = (lookahead_ms, window, hop_frames, mode)
        return True

    def send(self, seq, start, end, f0, f1, audio):
        assert end - len(audio) == len(self.audio)  # only the samples it has not got yet
        self.audio = np.concatenate([self.audio, audio])
        self.sent.append((seq, start, end, f0, f1))
        if self.teach is not None:
            rows = gp.window_rows(self.teach(self.audio[start:end]), start // FRAME, f0, f1)
            self.replies.put(("probs", seq, f0, f1, rows, 5.0))
        return True

    def close(self):
        self.closed = True


def feed(det, x: np.ndarray, clock=None) -> list:
    """The audio 10 ms at a time, as a session feeds it (advancing a fake clock in real time)."""
    out = []
    for i in range(0, len(x), HOP):
        if clock is not None:
            clock[0] += HOP / SR
        out += det.process(x[i:i + HOP])
    return out


@pytest.mark.parametrize("lookahead,cuts,interjections", [(200, "gentle", False), (500, "strong", True)])
def test_settings_match_the_teacher(teacher, lookahead, cuts, interjections):
    cfg = live.teacher_cut_config(cuts)
    local = TeacherDetector(lookahead, cut_cfg=cfg, cut_interjections=interjections)
    remote = RemoteTeacherDetector(EchoLink(), lookahead, cut_cfg=cfg, cut_interjections=interjections)
    for name, value in vars(local).items():
        mine = getattr(remote, name)
        assert (np.array_equal(value, mine) if isinstance(value, np.ndarray) else value == mine), name


def test_the_remote_teacher_cuts_exactly_like_the_local_one(teacher):
    x = bursts()
    local = TeacherDetector(200, cut_cfg=BALANCED_CUTS)
    link = EchoLink()
    remote = RemoteTeacherDetector(link, 200, cut_cfg=BALANCED_CUTS, clock=lambda: 0.0)
    assert link.started == (200, remote.win, remote.hop_frames, "gentle")
    expected, got = feed(local, x), feed(remote, x)
    got += remote.process(np.zeros(0, np.float32))  # its last answer lands one process() later
    assert got == expected and sum(u.op == "set" and u.end is not None for u in expected) == 10  # one per burst
    assert remote.decided_samples() == local.decided_samples()
    assert [(e.kind, e.start, e.end) for e in remote.events] == [(e.kind, e.start, e.end) for e in local.events]
    assert (remote.name, remote.note) == ("teacher", None) and remote.detect_ms() is not None


def test_a_worker_that_never_answers_never_stalls_the_voice(teacher):
    clock = [0.0]
    det = RemoteTeacherDetector(EchoLink(teach=None), 200, clock=lambda: clock[0])
    x = bursts(8.0)
    behind, decided = [], []
    for i in range(0, len(x), HOP):
        clock[0] += HOP / SR
        det.process(x[i:i + HOP])
        decided.append(det.decided_samples())
        if det.note is None and i > SR:
            behind.append(det.total - det.decided_samples())
    assert decided == sorted(decided)  # never backwards
    # decided as "no stutter" at each deadline: never further behind than the delay allows
    assert max(behind) <= (200 + 160 + 300 + gp.LIVE_DEADLINE_SLACK_MS + 50) / 1000 * SR
    assert det.note == NOTES["slow"] and det.decided_samples() == det.total  # no answer at all within 5 s


class WarmingLink(EchoLink):
    """A worker whose model takes `warm_s` to be ready (None: never), as after a recording made room."""

    def __init__(self, warm_s):
        super().__init__()
        self.ready, self.warm_s = threading.Event(), warm_s

    def start(self, *a):
        if self.warm_s is not None:
            threading.Timer(self.warm_s, self.ready.set).start()
        return super().start(*a)


def test_the_session_waits_for_the_workers_model_as_for_a_local_one(teacher, monkeypatch):
    t = time.monotonic()
    det = RemoteTeacherDetector(WarmingLink(0.3), 200)
    assert time.monotonic() - t >= 0.25 and det.note is None  # set up once the model is warm...
    feed(det, bursts(3.0))
    assert det.note is None and det.decided > 0  # ...so its first answers are in time
    monkeypatch.setattr(gp, "LIVE_READY_S", 0.2)
    link = WarmingLink(None)  # never ready: the session starts without it, on the rule-based detector
    det = RemoteTeacherDetector(link, 200)
    det.process(np.zeros(HOP, np.float32))
    assert det.note == NOTES["slow"] and link.closed and det.decided_samples() == det.total


def test_an_answer_for_frames_nobody_asked_for_is_dropped(teacher):
    link = EchoLink()
    det = RemoteTeacherDetector(link, 200, clock=lambda: 0.0)
    feed(det, bursts(3.0))
    far = 10 ** 12  # a valid-looking reply (a worker's token leaked, say) that would decide frames for ever
    link.replies.put(("probs", 999, far - 1, far, np.zeros((1, len(CLASSES)), np.float32), 1.0))
    t = time.perf_counter()
    det.process(np.zeros(HOP, np.float32))
    assert time.perf_counter() - t < 0.5 and det.decided <= det.requested < 10 ** 6 and det.note is None


class SlowLink(EchoLink):
    """A worker whose answers arrive `late` seconds after each window, on the test's clock."""

    def __init__(self, clock):
        super().__init__()
        self.clock, self.late, self.held = clock, 0.0, []

    def send(self, *a):
        ok = super().send(*a)
        self.held.append((self.clock[0] + self.late, self.replies.get_nowait()))
        return ok

    def deliver(self):
        while self.held and self.held[0][0] <= self.clock[0]:
            self.replies.put(self.held.pop(0)[1])


def test_late_answers_hand_over_to_the_rule_based_detector(teacher):
    clock = [0.0]
    link = SlowLink(clock)
    det = RemoteTeacherDetector(link, 200, clock=lambda: clock[0])
    x = bursts(12.0)
    for i in range(0, len(x), HOP):
        if i == 2 * SR:
            assert det.note is None and det.decided > 0  # answered in time so far
            link.late, switched = 1.0, clock[0]  # from now on every answer comes a second late
        clock[0] += HOP / SR
        link.deliver()
        det.process(x[i:i + HOP])
        if det.note:
            break
    assert det.note == NOTES["slow"] and link.closed
    assert clock[0] - switched < 2.0  # about a second of missed answers in a row, after the delay


def test_a_worker_that_drops_mid_cut_hands_over_cleanly(teacher):
    link = EchoLink()
    det = RemoteTeacherDetector(link, 200, clock=lambda: 0.0)
    x = bursts(10.0)
    i = 0
    while det.cut is None:  # until the teacher follows a stutter
        det.process(x[i:i + HOP])
        i += HOP
    open_id, events = det.cut["id"], list(det.events)
    link.replies = queue.Queue()  # it drops with answers still on the way
    link.replies.put(("closed", "gone"))
    first = det.process(x[i:i + HOP])
    assert first[0] == CutUpdate("cancel", open_id)  # play it after all: nothing half-cut
    assert (det.name, det.note, link.closed) == ("rule-based", NOTES["gone"], True)
    tone = (0.3 * np.sin(2 * np.pi * 180 * np.arange(SR) / SR)).astype(np.float32)  # a held sound
    later = feed(det, np.concatenate([x[i + HOP:i + 2 * SR], tone, x[:SR]]))
    assert later and all(u.cut_id >= RemoteTeacherDetector.SHADOW_IDS for u in later)
    assert det.events[:len(events)] == events and det.decided_samples() == det.total
    assert det.detect_ms() is None


@pytest.mark.parametrize("fault,note", [("raises", "gone"), ("refuses", "slow"), ("closed", "gone"), ("start", "gone")])
def test_a_refused_window_hands_over_and_nothing_raises(teacher, fault, note):
    link = EchoLink()
    if fault == "start":
        link.start = lambda *a: False
    else:
        def send(*a):
            if fault == "raises":
                raise RuntimeError("the socket went away")
            link.closed = fault == "closed"
            return False
        link.send = send
    det = RemoteTeacherDetector(link, 200, clock=lambda: 0.0)
    feed(det, bursts(3.0))
    assert det.name == "rule-based" and det.note == NOTES[note]


def test_the_remote_teachers_delay_includes_the_trip():
    assert live.delay_overrides("teacher", "gentle", 300) == {"min_delay_ms": 760, "target_delay_ms": 840}
    assert live.delay_overrides("teacher", "gentle") == {"min_delay_ms": 460, "target_delay_ms": 540}
    assert live.delay_overrides("heuristic", "strong", 300) == {}


def test_make_detector_uses_a_gpu_elsewhere_only_when_one_is_ready(teacher, monkeypatch):
    monkeypatch.setattr(teacher_runtime, "live_capable", lambda *a: False)
    monkeypatch.setattr(live, "remote_live", lambda mode: None)
    assert live.make_detector("teacher", "gentle") == (None, "rule-based")
    link = EchoLink()
    monkeypatch.setattr(live, "remote_live", lambda mode: link)
    plain, _ = live.make_detector("teacher", "strong")
    det, name = live.make_detector("teacher", "strong", adjust={"word_rep": 2.0})
    assert name == "teacher" and isinstance(det, RemoteTeacherDetector) and link.started == (500, det.win, 8, "strong")
    w = CLASSES.index("word_rep")
    assert det.thresholds[w] != plain.thresholds[w]  # the person's feedback applies
    broken = EchoLink()
    monkeypatch.setattr(live, "remote_live", lambda mode: broken)
    monkeypatch.setattr("stutterclear.feedback.scale_thresholds", boom)
    with pytest.raises(RuntimeError):
        live.make_detector("teacher", "gentle", adjust={"word_rep": 2.0})
    assert broken.closed  # a failure after the link opened frees it


def test_the_detector_list_offers_the_teacher_on_demand(teacher, monkeypatch):
    monkeypatch.setattr(teacher_runtime, "live_capable", lambda *a: False)
    monkeypatch.setattr(live, "remote_offered", lambda: True)
    t = next(d for d in live.detectors() if d["kind"] == "teacher")
    assert t["available"] and t["on_demand"] and t["delay_ms"] == {"gentle": 840, "strong": 1140}
    assert live.recommended() != "teacher"  # never one that may be asleep
    monkeypatch.setattr(live, "remote_offered", lambda: False)
    t = next(d for d in live.detectors() if d["kind"] == "teacher")
    assert not t["available"] and not t["on_demand"]


def test_a_session_shows_the_switch_to_the_page(teacher, monkeypatch):
    link = EchoLink()
    monkeypatch.setattr(teacher_runtime, "live_capable", lambda *a: False)
    monkeypatch.setattr(live, "remote_live", lambda mode: link)
    s = live.LiveSession("gentle", detector="teacher", source="browser")
    assert s.detector_name == "teacher" and s.pipe.cfg.target_delay_ms == 840  # the trip is in the delay
    s.start()
    try:
        s.feed(bursts(2.0))
        until(lambda: s.q.empty() and link.sent)
        tick = s.poll()
        assert (tick["detector"], tick["detector_note"]) == ("teacher", None)
        link.replies.put(("closed", "gone"))
        s.feed(bursts(0.5))
        until(lambda: s.poll()["detector"] == "rule-based")
        assert s.poll()["detector_note"] == NOTES["gone"]
    finally:
        s.stop()
    assert link.closed


def test_a_detector_that_raises_never_stops_the_voice(monkeypatch):
    class Broken:
        events: list = []

        def process(self, x):
            raise RuntimeError("boom")

        def decided_samples(self):
            return 0
    monkeypatch.setattr(captions, "available", lambda: False)
    monkeypatch.setattr(live, "make_detector", lambda *a, **k: (Broken(), "student"))
    s = live.LiveSession("gentle", detector="student", source="browser")
    s.start()
    s.feed(bursts(2.0))
    until(lambda: s.out.n >= 1.5 * SR)  # the voice keeps flowing, uncut
    s.stop()
    assert s.detect_failed and s.poll()["detector_note"] == "The detector stopped, so your voice now goes through uncut."


def test_session_envelopes_are_bounded(monkeypatch):
    monkeypatch.setattr(captions, "available", lambda: False)
    monkeypatch.setattr(live, "PLAYBACK_MAX_S", 2)  # keep 200 points (and drop them 1000 at a time)
    s = live.LiveSession("gentle", detector="heuristic", source="browser")
    s.start()
    s.feed(np.zeros(20 * SR, np.float32))  # 2000 points
    until(lambda: s.q.empty() and s._env_off["env_in"] + len(s.env_in) == 2000)
    tick = s.poll()
    s.stop()
    assert len(s.env_in) <= 1200 and tick["in_start"] == s._env_off["env_in"] == 1000
    assert tick["in_start"] + len(tick["env_in"]) == 2000  # the page's indices still count from the start


def until(cond, timeout: float = 10.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.01)


# ---- the browser mic socket -------------------------------------------------------------------------

def test_two_pages_starting_at_once_get_one_session(monkeypatch):
    monkeypatch.setattr(captions, "available", lambda: False)
    baseline = threading.active_count()
    c = TestClient(server.app)
    results, answered = [], threading.Event()

    def start():
        try:
            with c.websocket_connect("/ws/mic") as ws:
                ws.send_json({"mode": "gentle", "detector": "heuristic"})
                results.append(ws.receive_json())
                answered.wait(10)  # keep the session until the other page has its answer
                ws.send_text("stop")
                ws.receive_json()
        except WebSocketDisconnect as e:
            results.append(e.code)
            answered.set()
    threads = [threading.Thread(target=start) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(15)
    assert sorted(results, key=str) == [4409, {"ok": True, "detector": "rule-based", "limit_s": None, "delay_ms": 250}]
    assert server.state["session"] is None and not server.state["claim"]
    until(lambda: threading.active_count() <= baseline)  # no session thread left behind


def test_a_mic_socket_must_send_its_settings_soon(monkeypatch):
    monkeypatch.setattr(server, "SETTINGS_S", 0.1)
    c = TestClient(server.app)
    with c.websocket_connect("/ws/mic") as ws:
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
    assert e.value.code == 1008 and not server.state["claim"]
    with c.websocket_connect("/ws/mic") as ws:
        ws.send_text("not json")
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
    assert e.value.code == 1008 and server.state["session"] is None


@pytest.mark.parametrize("frames,code,ended", [
    ([np.zeros(8001, np.float32)], 1009, "Audio frames must be at most half a second."),  # 32,004 bytes
    ([np.zeros(8000, np.float32)] * 8, 1008, "Audio arrived faster than real time."),  # 4 s at once
])
def test_mic_frames_too_big_or_too_fast_end_the_session(monkeypatch, frames, code, ended):
    monkeypatch.setattr(captions, "available", lambda: False)
    c = TestClient(server.app)
    with c.websocket_connect("/ws/mic") as ws:
        ws.send_json({"mode": "gentle", "detector": "heuristic"})
        assert ws.receive_json()["ok"] is True
        for f in frames:
            ws.send_bytes(f.tobytes())
        end = ws.receive_json()
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
    assert end["ended"] == ended and e.value.code == code and server.state["session"] is None


def test_a_session_that_cannot_start_says_so(monkeypatch):
    monkeypatch.setattr(server, "LiveSession", boom)
    c = TestClient(server.app)
    with c.websocket_connect("/ws/mic") as ws:
        ws.send_json({"mode": "gentle", "detector": "heuristic"})
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
    assert e.value.code == 1011 and e.value.reason == "Could not start the session."
    assert server.state["session"] is None and not server.state["claim"]


def test_best_gpu_sessions_are_rationed_per_visitor(monkeypatch):
    monkeypatch.setattr(server, "_teacher_remote", lambda: True)
    monkeypatch.setattr(server.gpu.broker, "admit_live", lambda client: "You have used Best (GPU) several times.")
    c = TestClient(server.app)
    with c.websocket_connect("/ws/mic") as ws:
        ws.send_json({"mode": "gentle", "detector": "teacher"})
        assert ws.receive_json() == {"ok": False, "error": "You have used Best (GPU) several times."}
    assert server.state["session"] is None and not server.state["claim"]
