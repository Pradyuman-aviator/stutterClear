"""Context agent with stand-ins for Tavily and Nemotron."""
import io
import json

from stutterclear import context_agent, nemotron


class Reply(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_card_uses_search_results_when_tavily_is_set(monkeypatch):
    seen = {}
    monkeypatch.setenv("TAVILY_API_KEY", "tv")
    monkeypatch.setattr(context_agent.urllib.request, "urlopen", lambda req, timeout: Reply(json.dumps(
        {"results": [{"title": "NVIDIA Riva", "url": "https://nvidia.com/riva", "content": "Speech AI SDK"}]}).encode()))

    def fake_chat(msgs, **k):
        seen["user"] = msgs[1]["content"]
        return "Interview at NVIDIA.\nWords: Riva, speech AI, StutterClear"

    monkeypatch.setattr(nemotron, "chat", fake_chat)
    out = context_agent.fill("Interview with NVIDIA speech team", "My name is Sam.")
    assert out["searched"] and out["sources"] == ["https://nvidia.com/riva"]
    assert "Riva" in seen["user"] and "My name is Sam." in seen["user"]
    assert out["card"].startswith("Interview at NVIDIA.")


def test_card_without_search(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(nemotron, "chat", lambda msgs, **k: "Doctor visit.\nWords: Dr. Mehta, results")
    out = context_agent.fill("Seeing Dr. Mehta about my results")
    assert not out["searched"] and out["sources"] == [] and "Dr. Mehta" in out["card"]
