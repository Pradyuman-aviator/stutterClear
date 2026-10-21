import numpy as np
import pytest

from stutterclear import captions, whisper_flow


def test_whisper_flow_available():
    assert whisper_flow.available() is True


def test_whisper_flow_engine_selection(monkeypatch):
    monkeypatch.delenv("STUTTERCLEAR_ASR_ENGINE", raising=False)
    assert captions.engine_name() == "whisperflow"
    assert "WhisperFlow" in captions.engine_label()
    assert captions.available() is True


def test_whisper_flow_force_engine_not_available(monkeypatch):
    monkeypatch.setenv("STUTTERCLEAR_ASR_ENGINE", "nonexistent")
    assert captions.engine_name() == ""
    assert captions.available() is False


def test_whisper_flow_captioner_streaming():
    model = whisper_flow.get_model()
    cap = whisper_flow.WhisperFlowCaptioner(model=model)

    # 1. Feed pure silence
    silence = np.zeros(16000, dtype=np.float32)
    cap.feed(silence)
    assert cap.partial == ""
    assert cap.lines == []

    # 2. Feed simulated speech-like tone followed by silence
    t = np.linspace(0, 1.0, 16000, dtype=np.float32)
    tone = 0.1 * np.sin(2 * np.pi * 300 * t)
    for i in range(0, len(tone), 1600):
        cap.feed(tone[i : i + 1600])

    # Feed silence to trigger pause endpointing
    for _ in range(10):
        cap.feed(np.zeros(1600, dtype=np.float32))

    cap.close()
    assert cap.t >= 2.0


def test_whisper_flow_caption_audio():
    # Empty audio returns empty word list
    assert whisper_flow.caption_audio(np.zeros(0, dtype=np.float32)) == []

    # Synthetic tone returns word list (or empty list if no speech tokens recognized)
    t = np.linspace(0, 2.0, 32000, dtype=np.float32)
    tone = 0.1 * np.sin(2 * np.pi * 440 * t)
    res = whisper_flow.caption_audio(tone)
    assert isinstance(res, list)
    for w in res:
        assert "w" in w and "t" in w


def test_captions_wrapper_delegation(monkeypatch):
    monkeypatch.setenv("STUTTERCLEAR_ASR_ENGINE", "whisperflow")
    cap = captions.Captioner()
    assert isinstance(cap, whisper_flow.WhisperFlowCaptioner)

    # Feed small chunk and close
    cap.feed(np.zeros(1600, dtype=np.float32))
    cap.close()
    assert cap.lines == []
