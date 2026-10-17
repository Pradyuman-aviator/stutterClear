"""Word rescue: prompt building and reply parsing, and the API call with a stand-in server reply."""
import io
import json

from stutterclear import rescue


def test_parse_takes_the_last_json_array_and_cleans_it():
    reply = 'Thinking... maybe ["x"]? Final answer: ["Kaggle", "Kaggle notebook", "kaggle", "Kubernetes cluster setup"]'
    assert rescue.parse(reply) == ["Kaggle", "Kaggle notebook", "Kubernetes cluster setup"]


def test_parse_ignores_reasoning_and_falls_back_to_lines():
    assert rescue.parse("<think>[\"no\"]</think>\n1. Mehta\n2. meeting\n3. Monday") == ["Mehta", "meeting", "Monday"]


def test_messages_keep_only_recent_words_and_include_the_onset():
    m = rescue.messages(" ".join(f"w{i}" for i in range(200)), "My name is Sam.", onset="b")
    user = m[1]["content"]
    assert "w199" in user and "w100" not in user
    assert "Sam" in user and 'stuck on: "b"' in user


def test_suggest_calls_the_api(monkeypatch):
    sent = {}

    class Reply(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout):
        sent["url"], sent["auth"], sent["body"] = req.full_url, req.headers["Authorization"], json.loads(req.data)
        return Reply(json.dumps({"choices": [{"message": {"content": '["StutterClear", "stutter", "study"]'}}]}).encode())

    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.setattr(rescue.nemotron.urllib.request, "urlopen", fake_urlopen)
    assert rescue.suggest("I am building", "Project: StutterClear", "st") == ["StutterClear", "stutter", "study"]
    assert sent["url"].endswith("/chat/completions") and sent["auth"] == "Bearer test-key"
    assert sent["body"]["model"] == rescue.MODEL


def test_last_json_takes_the_outermost_object():
    from stutterclear.nemotron import last_json
    assert last_json('ok {"keep": [{"id": 5}, {"id": 11}]} done') == {"keep": [{"id": 5}, {"id": 11}]}
    assert last_json('first ["a"] then ["b", "c"]', "[") == ["b", "c"]
    assert last_json("no json here") is None
