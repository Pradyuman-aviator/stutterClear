"""The GPU worker's wire contract: what one side encodes the other decodes exactly, anything malformed
is a ProtocolError (never another exception), audio and probabilities survive the trip, and tokens
can be neither forged nor reused on another server."""
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from stutterclear import gpu_protocol as gp

SECRET = "s" * 40
HOST = "stutterclear-123.asia-south1.run.app"
NOW = 1_800_000_000


def probs(n: int) -> np.ndarray:
    return np.random.default_rng(n).random((n, gp.N_CLASSES), dtype=np.float32)


# One valid message of every type, with its payload: also a reference for the units.
SAMPLES = {
    gp.WELCOME: ({"proto": 1, "commit": "a" * 40, "idle_exit_s": 600, "idle_unused_s": 240.0,
                  "rotate_after_s": 3000, "now": NOW + 0.5}, b""),
    gp.PING: ({"n": 3, "ts": 812.25}, b""),
    gp.JOB: ({"job": "job-7", "kind": "note", "samples": 2_880_000, "parts": 88, "enc": "pcm16", "sr": 16000,
              "language": "en"}, b""),
    gp.JOB_AUDIO: ({"job": "job-7", "i": 0}, b"\x01\x00" * (gp.CHUNK_BYTES // 2)),
    gp.CANCEL: ({"job": "job-7"}, b""),
    gp.SHUTDOWN: ({"reason": "idle"}, b""),
    gp.LIVE_OPEN: ({"stream": "live-3", "mode": "gentle", "lookahead_ms": 200, "window": 67200, "hop_frames": 8},
                   b""),
    gp.LIVE_AUDIO: ({"stream": "live-3", "seq": 41, "start": 38400, "end": 105600, "f0": 306, "f1": 314},
                    b"\x00\x00" * 2560),
    gp.LIVE_CLOSE: ({"stream": "live-3"}, b""),
    gp.HELLO: ({"proto": 1, "worker": "3fa2c1d0", "wake": "9c1e5a7b2d3f4e60", "role": "kaggle",
                "where": "Kaggle (2× Tesla T4)", "commit": "b" * 40,
                "gpus": [{"name": "Tesla T4", "mem_gb": 15.0}] * 2, "deadline": NOW + 10_000, "loading": True}, b""),
    gp.CAPS: ({"notes": True, "live": False, "loading": True, "window_ms": None,
               "models": {"live": None, "recordings": "teacher_v2/best.pt"}, "error": None}, b""),
    gp.HEARTBEAT: ({"jobs": ["job-7"], "live": False, "window_ms": 41.5, "gpu_mem_gb": [3.7, 1.9]}, b""),
    gp.PONG: ({"n": 3, "ts": 812.25}, b""),
    gp.PROGRESS: ({"job": "job-7", "stage": "transcribing"}, b""),
    gp.RESULT: ({"job": "job-7", "frames": 99, "words": [{"w": "I", "s": 0.31, "e": 0.42}], "slow": False,
                 "timing": {"teacher_s": 0.4, "asr_s": 1.2}}, gp.probs_bytes(probs(99))),
    gp.ERROR: ({"job": "job-7", "message": "CUDA out of memory", "retry": True}, b""),
    gp.BYE: ({"reason": "idle"}, b""),
    gp.LIVE_READY: ({"stream": "live-3"}, b""),
    gp.LIVE_PROBS: ({"stream": "live-3", "seq": 41, "f0": 306, "f1": 314, "ms": 38.0}, gp.probs_bytes(probs(8))),
    gp.LIVE_ERROR: ({"stream": "live-3", "message": "one live stream at a time"}, b""),
}


def frame(t: str, payload: bytes | None = None, **change) -> bytes:
    """A sample message of type t with some fields changed (None deletes one), encoded by hand so
    that decode() alone has to catch what is wrong."""
    header, body = SAMPLES[t]
    header = {"t": t, **header, **change}
    header = {k: v for k, v in header.items() if v is not None or k in gp.FIELDS[t] and k not in change}
    return raw(json.dumps(header).encode()) + (body if payload is None else payload)


def refused(data) -> bool:
    try:
        gp.decode(data)
    except gp.ProtocolError:
        return True
    return False


# ---- codec --------------------------------------------------------------------------------------

def test_every_message_type_round_trips():
    assert set(SAMPLES) == set(gp.FIELDS) == gp.FROM_SERVER | gp.FROM_WORKER
    for t, (header, payload) in SAMPLES.items():
        data = gp.encode({"t": t, **header}, payload)
        assert gp.decode(data) == ({"t": t, **header}, payload)
        side = gp.FROM_SERVER if t in gp.FROM_SERVER else gp.FROM_WORKER
        assert gp.decode(data, expect=side)[0]["t"] == t
        with pytest.raises(gp.ProtocolError):  # a worker never sends a job, nor the server a result
            gp.decode(data, expect=(gp.FROM_SERVER | gp.FROM_WORKER) - side)


def test_extra_fields_pass_for_forward_compatibility():
    header, _ = SAMPLES[gp.PING]
    assert gp.decode(gp.encode({"t": gp.PING, **header, "later": [1, 2]}))[0]["later"] == [1, 2]


def raw(header: bytes) -> bytes:
    return struct.pack(">I", len(header)) + header


@pytest.mark.parametrize("data", [
    b"", b"\x00\x00\x00", struct.pack(">I", 10) + b"{}",  # too short; length past the end
    raw(b"x" * (gp.MAX_HEADER + 1)),  # header too large
    raw(b"[]"), raw(b"null"), raw(b"{x}"), raw(b"\xff\xfe"),  # not an object, not JSON, not UTF-8
    raw(b"[" * 100_000 + b"]" * 100_000),  # nested too deep for the parser
    raw(b"{}"), raw(b'{"t":"nope"}'),  # no type, unknown type
    raw(b'{"t":["ping"]}'), raw(b'{"t":{"a":1}}'),  # a type that is not text (must not reach a dict lookup)
], ids=["empty", "short", "past-end", "huge-header", "list", "null", "not-json", "not-utf8", "deep", "no-type",
        "unknown-type", "type-list", "type-object"])
def test_malformed_frames_are_protocol_errors(data):
    assert refused(data)


def test_a_frame_over_the_limit_is_refused_before_parsing():
    assert refused(b"\x00" * (gp.MAX_FRAME + 1))


@pytest.mark.parametrize("t,change", [
    (gp.PING, {"n": None}),  # missing
    (gp.PING, {"n": -1}), (gp.PING, {"n": 1.5}), (gp.PING, {"n": True}), (gp.PING, {"n": "1"}),
    (gp.PING, {"n": 2 ** 60}), (gp.PING, {"ts": 10 ** 400}),  # too big (never an OverflowError)
    (gp.CAPS, {"notes": 1}), (gp.CAPS, {"models": []}), (gp.HEARTBEAT, {"jobs": {}}),
    (gp.HELLO, {"deadline": "soon"}), (gp.HELLO, {"where": "x" * (gp.MAX_TEXT + 1)}),
    (gp.HELLO, {"role": "admin"}), (gp.PROGRESS, {"stage": "thinking"}), (gp.LIVE_OPEN, {"mode": "loud"}),
    (gp.SHUTDOWN, {"reason": "deploy"}), (gp.BYE, {"reason": "bored"}), (gp.JOB, {"sr": 8000}),
    (gp.JOB, {"kind": "video"}), (gp.JOB, {"enc": "f32"}),
    (gp.JOB, {"samples": gp.MAX_JOB_S * gp.SR + 1, "parts": gp.n_parts(gp.MAX_JOB_S * gp.SR + 1)}),
    (gp.JOB, {"parts": 87}), (gp.JOB, {"samples": 0, "parts": 0}),
    (gp.LIVE_AUDIO, {"f0": 315}), (gp.LIVE_AUDIO, {"start": 105601}), (gp.LIVE_PROBS, {"f1": 300}),
    (gp.RESULT, {"words": [{"w": "x", "s": 0, "e": 1}] * (gp.MAX_WORDS + 1)}),
])
def test_bad_fields_are_protocol_errors(t, change):
    assert refused(frame(t, **change))


def test_numbers_that_json_allows_but_the_protocol_does_not():
    for bad in (b"NaN", b"Infinity", b"-Infinity", b"1e999"):
        assert refused(raw(b'{"t":"ping","n":1,"ts":' + bad + b"}"))


@pytest.mark.parametrize("t,payload", [
    (gp.PING, b"\x00\x00"),  # no payload expected
    (gp.JOB_AUDIO, b""), (gp.JOB_AUDIO, b"\x00" * 3), (gp.JOB_AUDIO, b"\x00" * (gp.CHUNK_BYTES + 2)),
    (gp.LIVE_AUDIO, b""), (gp.LIVE_AUDIO, b"\x00" * 5),
    (gp.LIVE_AUDIO, b"\x00\x00" * (105600 - 38400 + 1)),  # more audio than the window holds
    (gp.RESULT, gp.probs_bytes(probs(98))), (gp.LIVE_PROBS, gp.probs_bytes(probs(9))),
], ids=["ping-payload", "part-empty", "part-odd", "part-too-big", "live-empty", "live-odd", "live-past-window",
        "result-rows", "live-probs-rows"])
def test_payload_sizes_are_checked(t, payload):
    assert refused(frame(t, payload))


def test_encode_refuses_what_decode_would():
    with pytest.raises(gp.ProtocolError):
        gp.encode({"t": gp.PING, "n": np.int64(1), "ts": 0.0})  # a numpy number is not JSON
    with pytest.raises(gp.ProtocolError):
        gp.encode({"t": gp.PING, "n": 1, "ts": float("nan")})
    with pytest.raises(gp.ProtocolError):
        gp.encode({"t": gp.PING, "n": 1, "ts": 0.0, "note": float("nan")})  # NaN anywhere in the header
    with pytest.raises(gp.ProtocolError):
        gp.encode({"t": gp.BYE, "reason": "idle"}, b"\x00\x00")
    big_words = [{"w": "x" * 99, "s": 0.0, "e": 1.0}] * gp.MAX_WORDS
    with pytest.raises(gp.ProtocolError, match="header too large"):
        gp.encode({"t": gp.RESULT, **{**SAMPLES[gp.RESULT][0], "words": big_words}}, SAMPLES[gp.RESULT][1])
    end = gp.MAX_FRAME  # a live window that holds more audio than one frame may carry
    with pytest.raises(gp.ProtocolError, match="frame too large"):
        gp.encode({"t": gp.LIVE_AUDIO, "stream": "s", "seq": 0, "start": 0, "end": end, "f0": 0, "f1": 8},
                  b"\x00\x00" * end)


def test_the_largest_note_fits():
    """180 s of audio in 64 KiB parts, and its result with 3000 words, stay within the limits."""
    x = np.zeros(gp.MAX_JOB_S * gp.SR, np.float32)
    parts = [gp.encode({"t": gp.JOB_AUDIO, "job": "job-1", "i": i}, p) for i, p in enumerate(gp.job_parts(x))]
    assert len(parts) == gp.n_parts(len(x)) == 88 and max(map(len, parts)) <= gp.MAX_FRAME
    n = gp.frames_for(len(x))
    words = [{"w": "stutterclear", "s": 123.456, "e": 123.789}] * gp.MAX_WORDS
    header = {"t": gp.RESULT, "job": "job-1", "frames": n, "words": words, "slow": False, "timing": {}}
    h, payload = gp.decode(gp.encode(header, gp.probs_bytes(np.zeros((n, gp.N_CLASSES)))))
    assert h["frames"] == 8999 and len(payload) == 8999 * 20


# ---- audio and probabilities --------------------------------------------------------------------

def test_pcm16_is_lossless_for_16_bit_audio():
    k = np.arange(-32768, 32768, dtype=np.int16)
    x = k.astype(np.float32) / 32768  # what soundfile gives for a 16-bit WAV
    b = gp.pcm16(x)
    assert b == k.astype("<i2").tobytes()
    assert np.array_equal(gp.from_pcm16(b), x)


def test_pcm16_clips_and_silences_what_is_not_audio():
    b = gp.pcm16(np.array([1.0, 2.0, -1.0, -3.0, np.nan, np.inf, -np.inf, 1 / 32768], np.float32))
    assert np.frombuffer(b, "<i2").tolist() == [32767, 32767, -32768, -32768, 0, 32767, -32768, 1]
    assert b[-2:] == b"\x01\x00"  # little-endian
    with pytest.raises(gp.ProtocolError):
        gp.from_pcm16(b"\x00\x00\x00")


def test_job_parts_cover_the_recording_in_even_parts():
    x = np.random.default_rng(0).uniform(-1, 1, 100_001).astype(np.float32)
    parts = list(gp.job_parts(x))
    assert len(parts) == gp.n_parts(len(x)) and all(len(p) <= gp.CHUNK_BYTES and len(p) % 2 == 0 for p in parts)
    assert b"".join(parts) == gp.pcm16(x)
    assert len(list(gp.job_parts(x, size=1000))) == gp.n_parts(len(x), 1000)
    with pytest.raises(ValueError):
        next(gp.job_parts(x, size=1001))


def test_probabilities_round_trip_and_are_checked():
    p = probs(37)
    back = gp.probs_from(gp.probs_bytes(p), 37)
    assert np.array_equal(back, p) and back.flags.writeable and back.dtype == np.float32
    assert gp.probs_from(b"", 0).shape == (0, gp.N_CLASSES)
    with pytest.raises(gp.ProtocolError):
        gp.probs_bytes(np.zeros((4, 3)))
    with pytest.raises(gp.ProtocolError):
        gp.probs_from(gp.probs_bytes(p), 36)
    for bad in (np.nan, -0.1, 1.5):  # logits instead of probabilities, or garbage
        q = p.copy()
        q[5, 2] = bad
        with pytest.raises(gp.ProtocolError):
            gp.probs_from(gp.probs_bytes(q), 37)


def test_frames_for_matches_the_teachers_count():
    import torch
    from stutterclear.app import notes

    def model(wave):  # a stand-in teacher: one row per 20 ms frame, as FrameTeacher gives
        return torch.zeros(1, (wave.shape[1] - 400) // gp.FRAME + 1, gp.N_CLASSES)

    for n in (16_000, 16_399, 16_400, 31 * 16_000 + 123, 61 * 16_000):
        x = np.zeros(n, np.float32)
        assert len(notes.teacher_probs(x, model=model, dev=torch.device("cpu"))) == gp.frames_for(n)
    assert gp.frames_for(399) == gp.frames_for(0) == 0 and gp.frames_for(400) == 1


def test_window_rows_give_zeros_outside_the_window():
    p = probs(10)  # frames 100..109
    assert np.array_equal(gp.window_rows(p, 100, 102, 105), p[2:5])
    rows = gp.window_rows(p, 100, 98, 103)
    assert np.array_equal(rows[:2], np.zeros((2, 5))) and np.array_equal(rows[2:], p[:3])
    assert not gp.window_rows(p, 100, 120, 124).any() and gp.window_rows(p, 100, 104, 104).shape == (0, 5)


def test_words_are_checked_but_kept_as_given():
    words = [{"w": " I", "s": 0.31, "e": 0.42}, {"w": "", "s": 1, "e": 0.9}]  # odd, but what CrisperWhisper may give
    assert gp.check_words(words, 2.0) == [{"w": " I", "s": 0.31, "e": 0.42}, {"w": "", "s": 1.0, "e": 0.9}]
    for bad in ({"w": 1, "s": 0, "e": 1}, {"w": "a", "s": "0", "e": 1}, {"w": "a", "s": 0, "e": float("nan")},
                {"w": "a", "s": 0, "e": True}, {"w": "a", "s": 0, "e": 60.0}, {"w": "a", "s": -2, "e": 1},
                {"w": "a" * 101, "s": 0, "e": 1}, ["a", 0, 1], {"w": "a", "s": 0, "e": 10 ** 400}):
        with pytest.raises(gp.ProtocolError):
            gp.check_words([bad], 2.0)
    with pytest.raises(gp.ProtocolError):
        gp.check_words({"w": "a"}, 2.0)


def test_live_extra_delay():
    assert gp.live_extra_delay_ms(20, 30) == 100  # a near worker still gets the minimum
    assert gp.live_extra_delay_ms(250, 90) == 320
    assert gp.live_extra_delay_ms(600, 50) is None  # too far for live on a personal computer...
    assert gp.live_extra_delay_ms(600, 50, gp.LIVE_MAX_EXTRA_HOSTED_MS) == 640  # ...fine on a public server
    assert gp.live_extra_delay_ms(float("nan"), 50) is None


def test_small_helpers():
    assert gp.first_engine("live") == "live" and gp.first_engine("voice-note") == "notes"
    assert gp.first_engine("--first --fake") == "notes"
    assert gp.printable("Kaggle\x1b[31m (T4)\n\tx") == "Kaggle [31m (T4) x" and gp.printable(None) == "None"
    assert gp.printable("Kaggle (2× Tesla T4) and more", gp.MAX_WHERE // 2) == "Kaggle (2× Tesla T4)"
    q = gp.Quota(used_h=12.5, total_h=30.0, reserved_h=1.0)
    assert q.left_h == 16.5 and gp.Launch("queued").version is None


# ---- tokens -------------------------------------------------------------------------------------

def token(**k) -> str:
    args = {"secret": SECRET, "role": "kaggle", "wake": "9c1e5a7b2d3f4e60", "expires": NOW + 3600,
            "server": "https://" + HOST} | k
    return gp.make_token(args["secret"], args["role"], args["wake"], args["expires"], args["server"])


def test_a_token_checks_out_on_its_own_server_only():
    t = token()
    assert gp.check_token(SECRET, t, HOST, NOW) == gp.Token("kaggle", "9c1e5a7b2d3f4e60", NOW + 3600)
    assert gp.check_token(SECRET, t, HOST.upper() + ":443", NOW)  # the Host header may carry the port
    assert gp.check_token(SECRET, t, "stutterclear-demo.centralindia.cloudapp.azure.com", NOW) is None
    assert gp.check_token("t" * 40, t, HOST, NOW) is None  # another secret
    assert gp.peek_token(t) == gp.Token("kaggle", "9c1e5a7b2d3f4e60", NOW + 3600)  # the worker reads its own


def test_tampered_tokens_are_refused():
    t = token()
    _, role, wake, expires, sig = t.split(".")
    other = sig[:-1] + ("B" if sig[-1] == "A" else "A")
    for forged in (f"sc1.laptop.{wake}.{expires}.{sig}", f"sc1.{role}.9c1e5a7b2d3f4e61.{expires}.{sig}",
                   f"sc1.{role}.{wake}.{int(expires) + 1}.{sig}", f"sc1.{role}.{wake}.{expires}.{other}",
                   f"sc2.{role}.{wake}.{expires}.{sig}", f"sc1.{role}.{wake}.0{expires}.{sig}", t + "."):
        assert gp.check_token(SECRET, forged, HOST, NOW) is None, forged


def test_expiry_is_judged_by_the_servers_clock():
    t = token(expires=NOW + 60)
    assert gp.check_token(SECRET, t, HOST, NOW + 59)
    assert gp.check_token(SECRET, t, HOST, NOW + 60) is None
    far = token(expires=NOW + gp.MAX_TOKEN_S + 1)  # signed correctly, but valid for too long
    assert gp.check_token(SECRET, far, HOST, NOW) is None


@pytest.mark.parametrize("junk", [None, "", 42, b"sc1.kaggle", "sc1....", "a" * 10_000, "sc1.kaggle.x.1.y",
                                  "sc1.local.abcd.1.x", ["sc1"], "Bearer sc1.kaggle.9c1e5a7b2d3f4e60.1.x"],
                         ids=["none", "empty", "int", "bytes", "dots", "long", "short-wake", "bad-sig", "list", "bearer"])
def test_check_token_never_raises(junk):
    assert gp.check_token(SECRET, junk, HOST, NOW) is None
    assert gp.peek_token(junk) is None
    assert gp.check_token(SECRET, token(), junk, NOW) is None  # nor on a bad Host header


def test_making_a_token_refuses_bad_inputs():
    for k in ({"secret": "short"}, {"role": "admin"}, {"wake": "UPPER"}, {"wake": "abc"}, {"wake": "a" * 41},
              {"expires": 0}, {"expires": 1.5}, {"expires": "1800000000"}, {"server": ""}):
        with pytest.raises(ValueError):
            token(**k)
    assert gp.check_token("short", token(), HOST, NOW) is None


def test_wake_ids_are_fresh():
    ids = {gp.new_wake() for _ in range(100)} | {gp.new_wake("laptop-")}
    assert len(ids) == 101 and all(gp.WAKE_RE.fullmatch(w) for w in ids)


def test_key_id_identifies_a_secret_without_showing_it():
    k = gp.key_id(SECRET)
    assert len(k) == 8 and int(k, 16) >= 0 and k == gp.key_id(SECRET) and k != gp.key_id("t" * 40)
    assert k not in SECRET


def test_read_secret_removes_what_powershell_adds(capsys, monkeypatch):
    monkeypatch.setattr(gp, "_tidied", set())
    env = {"STUTTERCLEAR_GPU_SECRET": "﻿" + SECRET + "\r\n", "CLEAN": SECRET, "EMPTY": " \r\n"}
    assert gp.read_secret("STUTTERCLEAR_GPU_SECRET", env) == SECRET
    assert gp.read_secret("STUTTERCLEAR_GPU_SECRET", env) == SECRET
    out = capsys.readouterr().out
    assert out.count("STUTTERCLEAR_GPU_SECRET") == 1 and SECRET not in out  # the name once, never the value
    assert gp.read_secret("CLEAN", env) == SECRET and capsys.readouterr().out == ""
    assert gp.read_secret("EMPTY", env) is None and gp.read_secret("MISSING", env) is None


# ---- where the server is ------------------------------------------------------------------------

def test_server_urls():
    assert gp.server_url(f"https://{HOST}/") == f"https://{HOST}"
    assert gp.ws_url(f"https://{HOST}") == f"wss://{HOST}/ws/gpu"
    assert gp.ws_url("http://127.0.0.1:8801") == "ws://127.0.0.1:8801/ws/gpu"
    assert gp.ws_url("http://[::1]:8801") == "ws://[::1]:8801/ws/gpu"
    for bad in ("http://example.com", "http://localhost.evil.com:8801", "ftp://x", f"https://{HOST}/app",
                f"https://user:pw@{HOST}", f"https://{HOST}?token=x", f"https://{HOST}:99999", "", "https://"):
        with pytest.raises(ValueError):
            gp.server_url(bad)


def test_audience_is_the_same_from_the_url_and_the_host_header():
    assert gp.audience(f"https://{HOST.upper()}") == gp.audience(HOST + ":443") == gp.audience(HOST + ".") == HOST
    assert gp.audience("http://127.0.0.1:8801") == gp.audience("127.0.0.1:8801") == "127.0.0.1:8801"
    assert gp.audience("http://[::1]:8801") == gp.audience("[::1]:8801") == "[::1]:8801"
    for bad in ("", ":8801", "host:x", "user@host"):
        with pytest.raises(ValueError):
            gp.audience(bad)


def test_the_module_needs_only_the_standard_library_and_numpy():
    root = Path(__file__).resolve().parents[1]
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import stutterclear.gpu_protocol; "
            "heavy = {'torch', 'scipy', 'soundfile', 'fastapi', 'starlette', 'websockets', 'kagglesdk', 'kaggle'}; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in heavy))")
    out = subprocess.run([sys.executable, "-c", code, str(root)], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
