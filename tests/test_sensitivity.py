"""Live sensitivity (the Live panel's slider): 50 is each mode as measured, higher cuts more readily,
and a running session follows the slider without restarting."""
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from stutterclear import live
from stutterclear.app import server
from stutterclear.audio import load_wav
from stutterclear.config import preset
from stutterclear.detector import HeuristicDetector
from stutterclear.pipeline import Pipeline, stream
from stutterclear.remote_detector import RemoteTeacherDetector
from stutterclear.student_runtime import StudentDetector

client = TestClient(server.app)
CLIP = Path(__file__).resolve().parents[1] / "data" / "demo_clips" / "read-1.wav"


def test_slider_values_are_clamped_and_50_changes_nothing():
    assert [live.clamp_sensitivity(v) for v in (None, "x", float("nan"), float("inf"), -5, 150, "70")] == \
        [50, 50, 50, 50, 0, 100, 70]
    assert live.sensitivity_scale(50) == 1.0
    assert live.sensitivity_scale(0) == pytest.approx(live.SENSITIVITY_RANGE[0])
    assert live.sensitivity_scale(100) == pytest.approx(live.SENSITIVITY_RANGE[1])
    scales = [live.sensitivity_scale(v) for v in range(0, 101, 5)]
    assert all(a > b for a, b in zip(scales, scales[1:]))  # higher: lower thresholds, more cuts


def test_model_thresholds_scale_from_where_they_started():
    det = StudentDetector.__new__(StudentDetector)  # the cut rule only: no model needed
    det.thresholds = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    det.cut_idx = [0, 1, 2]
    det.set_sensitivity(0.5)
    det.set_sensitivity(0.5)  # moving the slider back and forth never compounds
    assert det.thresholds.tolist() == pytest.approx([0.15, 0.2, 0.25, 0.6, 0.7])  # blocks are never cut: untouched
    assert det.MAX_FRAMES == 141  # eager cuts run longer before they count as a detector error
    det.set_sensitivity(10.0)
    assert det.thresholds[:3].tolist() == [0.99, 0.99, 0.99] and det.MAX_FRAMES == StudentDetector.MAX_FRAMES
    det.set_sensitivity(0.1)  # the floor comes down with the slider, never to nothing
    assert det.thresholds[:3].tolist() == pytest.approx([0.03, 0.04, 0.05])


def test_the_splicer_lets_eager_cuts_run_longer_and_never_shorter():
    pipe = Pipeline(preset("clean"), detector=HeuristicDetector())
    base = pipe.splicer.max_open
    pipe.set_sensitivity(0.25)
    assert pipe.splicer.max_open == 2 * base and pipe.detector.REP_THRESHOLD > HeuristicDetector.REP_THRESHOLD
    pipe.set_sensitivity(2.0)
    assert pipe.splicer.max_open == base


def test_rule_based_detector_loosens_its_tests_for_one_session_only():
    a, b = HeuristicDetector(), HeuristicDetector()
    a.set_sensitivity(0.25)
    assert a.REP_THRESHOLD > HeuristicDetector.REP_THRESHOLD
    assert a.PRO_STATIONARY_DB > HeuristicDetector.PRO_STATIONARY_DB and a.PRO_MIN < HeuristicDetector.PRO_MIN
    assert b.REP_THRESHOLD == HeuristicDetector.REP_THRESHOLD  # another session is not touched
    a.set_sensitivity(1.0)
    assert (a.REP_THRESHOLD, a.PRO_STATIONARY_DB, a.PRO_MIN) == \
        (HeuristicDetector.REP_THRESHOLD, HeuristicDetector.PRO_STATIONARY_DB, HeuristicDetector.PRO_MIN)


def test_the_gpu_detectors_stand_in_follows_too():
    det = RemoteTeacherDetector.__new__(RemoteTeacherDetector)
    det.thresholds, det.cut_idx, det.shadow = np.array([0.4, 0.4, 0.4, 0.4, 0.4]), [0, 1, 2], HeuristicDetector()
    det.set_sensitivity(0.5)
    assert det.thresholds[0] == pytest.approx(0.2) and det.shadow.REP_THRESHOLD > HeuristicDetector.REP_THRESHOLD


def test_higher_sensitivity_cuts_more_of_a_real_clip():
    x = load_wav(CLIP)
    removed = []
    for v in (0, 50, 100):
        det = HeuristicDetector()
        det.set_sensitivity(live.sensitivity_scale(v))
        pipe, _ = stream(x, "trim", det)
        removed.append(pipe.splicer.stats.removed)
    assert removed[0] <= removed[1] < removed[2]


def test_browser_session_starts_with_the_slider_and_follows_it(monkeypatch):
    seen = []
    real = live.LiveSession.set_sensitivity

    def spy(self, v):
        seen.append(real(self, v))
        return seen[-1]
    monkeypatch.setattr(live.LiveSession, "set_sensitivity", spy)
    with client.websocket_connect("/ws/mic") as ws:
        ws.send_json({"mode": "gentle", "detector": "heuristic", "sensitivity": 80})
        assert ws.receive_json()["ok"] is True
        ws.send_text('{"sensitivity": 20}')  # moved while talking: no reply, the session just follows
        ws.send_text("not json")
        ws.send_text('{"other": 1}')
        ws.send_bytes(np.zeros(320, np.float32).tobytes())
        ws.send_text("stop")
        end = ws.receive_json()
    assert seen == [80, 20] and end["summary"]["sensitivity"] == 20


def test_slider_for_a_session_on_this_computer(monkeypatch):
    monkeypatch.setitem(server.state, "session", None)
    assert client.post("/api/sensitivity", json={"sensitivity": 70}).json() == {"ok": True, "sensitivity": None}

    class Session:
        def set_sensitivity(self, v):
            return live.clamp_sensitivity(v)
    monkeypatch.setitem(server.state, "session", Session())
    assert client.post("/api/sensitivity", json={"sensitivity": 170}).json()["sensitivity"] == 100
    monkeypatch.setattr(server, "HOSTED", True)  # a public server: only the page's own connection changes it
    assert client.post("/api/sensitivity", json={"sensitivity": 70}).status_code == 403
