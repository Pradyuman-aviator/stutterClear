"""Say it: the hand-written Riva messages, and the endpoint with a stand-in for NVIDIA's server."""
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.io import wavfile

from stutterclear import tts
from stutterclear.app import server

client = TestClient(server.app)


def test_request_bytes_match_the_riva_fields():
    req = tts.request("hi", "V", sr=16000)
    assert req == (b"\x0a\x02hi" + b"\x12\x05en-US" + b"\x18\x01" + b"\x20\x80\x7d" + b"\x2a\x01V")


def test_messages_agree_with_protobuf():
    """Encode and decode with the real protobuf library, from a descriptor built like riva_tts.proto."""
    pytest.importorskip("google.protobuf")
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    T = descriptor_pb2.FieldDescriptorProto
    f = descriptor_pb2.FileDescriptorProto(name="riva_tts_test.proto", package="t", syntax="proto3")
    req = f.message_type.add(name="Req")
    for name, num, typ in (("text", 1, T.TYPE_STRING), ("language_code", 2, T.TYPE_STRING),
                           ("encoding", 3, T.TYPE_INT32), ("sample_rate_hz", 4, T.TYPE_INT32),
                           ("voice_name", 5, T.TYPE_STRING)):
        req.field.add(name=name, number=num, type=typ, label=T.LABEL_OPTIONAL)
    resp = f.message_type.add(name="Resp")
    for name, num, typ in (("audio", 1, T.TYPE_BYTES), ("meta", 2, T.TYPE_BYTES), ("id", 100, T.TYPE_STRING)):
        resp.field.add(name=name, number=num, type=typ, label=T.LABEL_OPTIONAL)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(f)
    Req = message_factory.GetMessageClass(pool.FindMessageTypeByName("t.Req"))
    Resp = message_factory.GetMessageClass(pool.FindMessageTypeByName("t.Resp"))

    m = Req.FromString(tts.request("Kaggle notebook", "Magpie-Multilingual.EN-US.Aria", sr=22050))
    assert (m.text, m.language_code, m.encoding, m.sample_rate_hz, m.voice_name) == \
        ("Kaggle notebook", "en-US", tts.LINEAR_PCM, 22050, "Magpie-Multilingual.EN-US.Aria")
    pcm = np.arange(-300, 300, dtype=np.int16).tobytes()
    reply = Resp(meta=b"\x0a\x03abc", id="r1", audio=pcm).SerializeToString()
    assert tts.audio_field(reply) == pcm


def _fake_reply(n=1600):
    pcm = (np.sin(np.arange(n) / 5) * 8000).astype(np.int16).tobytes()
    return b"\x12\x02\x08\x01" + b"\x0a" + tts._varint(len(pcm)) + pcm  # meta first, then audio


def test_say_it_endpoint(monkeypatch):
    tts._pcm.cache_clear()
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    r = client.post("/api/tts", json={"text": "Kaggle"})
    assert r.status_code == 503 and "NVIDIA_API_KEY" in r.json()["detail"]

    monkeypatch.setenv("NVIDIA_API_KEY", "test")
    sent = []
    monkeypatch.setattr(tts, "_call", lambda req, timeout: sent.append(req) or _fake_reply())
    r = client.post("/api/tts", json={"text": "  Kaggle  "})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    assert "synthetic" in r.headers["x-voice-label"].lower()
    sr, y = wavfile.read(io.BytesIO(r.content))
    assert sr == 16000 and len(y) == 1600
    assert b"Kaggle" in sent[0] and b"  " not in sent[0][:12]  # whitespace tidied before sending
    client.post("/api/tts", json={"text": "Kaggle"})
    assert len(sent) == 1  # the same word again comes from the cache

    assert client.post("/api/tts", json={"text": "   "}).status_code == 400
    assert client.post("/api/tts", json={"text": "x" * 201}).status_code == 400
    assert client.post("/api/tts", json={"text": "hi", "voice": "a b/c"}).status_code == 400


def test_say_it_reports_a_failed_call(monkeypatch):
    tts._pcm.cache_clear()
    monkeypatch.setenv("NVIDIA_API_KEY", "test")

    def down(req, timeout):
        raise OSError("unreachable")
    monkeypatch.setattr(tts, "_call", down)
    r = client.post("/api/tts", json={"text": "hello"})
    assert r.status_code == 502 and "did not answer" in r.json()["detail"]


def test_hosted_say_it_is_capped(monkeypatch):
    tts._pcm.cache_clear()
    monkeypatch.setenv("NVIDIA_API_KEY", "test")
    monkeypatch.setattr(tts, "_call", lambda req, timeout: _fake_reply())
    monkeypatch.setattr(server, "TTS_PER_MIN", 2)
    monkeypatch.setattr(server, "_tts_calls", server.deque())
    codes = [client.post("/api/tts", json={"text": w}).status_code for w in ("one", "two", "three")]
    assert codes == [200, 200, 429]
