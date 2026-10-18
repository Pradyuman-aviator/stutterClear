"""Practice interview with a stand-in Nemotron."""
from fastapi.testclient import TestClient

from stutterclear import interview, nemotron
from stutterclear.app import server

client = TestClient(server.app)


def test_prompt_carries_the_role_card_and_recent_answers():
    history = [{"question": f"Q{i}?", "answer": f"answer {i}"} for i in range(8)]
    user = interview.messages("ML engineer", "Nebius", "Words: Token Factory, H100", history)[1]["content"]
    assert "Role: ML engineer" in user and "Company: Nebius" in user and "Token Factory" in user
    assert "Q7?" in user and "answer 2" in user and "Q1?" not in user  # only the last KEEP pairs
    assert "question 9 of" in user
    first = interview.messages("Nurse", "", "", [])[1]["content"]
    assert "first question" in first and "(not given)" in first


def test_parse_question_keeps_only_the_question():
    assert interview.parse_question("Sure!\n**Question 2:** What drew you to Nebius?") == "What drew you to Nebius?"
    assert interview.parse_question('Interviewer: "Tell me about a hard bug you fixed?"') == \
        "Tell me about a hard bug you fixed?"
    assert interview.parse_question("Walk me through your last project.") == "Walk me through your last project."


def test_interview_endpoints(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    def chat(msgs, **k):
        system = msgs[0]["content"]
        if "context card" in system and "prepare" in system:
            return "Interview at Nebius.\nWords: Token Factory, GPUs"
        if "coach" in system:
            return "- Clear example.\n- Add numbers.\n- Prepare: why Nebius?"
        return "Question: Why do you want to work at Nebius?"
    monkeypatch.setattr(nemotron, "chat", chat)

    r = client.post("/api/interview/start", json={"role": "ML engineer", "company": "Nebius"}).json()
    assert r["question"] == "Why do you want to work at Nebius?" and "Token Factory" in r["card"]
    assert r["questions"] == interview.QUESTIONS
    history = [{"question": r["question"], "answer": "I like GPUs"}]
    nxt = client.post("/api/interview/next", json={"role": "ML engineer", "card": r["card"], "history": history})
    assert nxt.json()["question"].endswith("?")
    pts = client.post("/api/interview/summary", json={"role": "ML engineer", "history": history}).json()["points"]
    assert len(pts) == 3

    assert client.post("/api/interview/start", json={"role": " "}).status_code == 400
    assert client.post("/api/interview/summary", json={"role": "x"}).status_code == 400
    monkeypatch.delenv("NEBIUS_API_KEY")
    r = client.post("/api/interview/next", json={"role": "x"})
    assert r.status_code == 503 and "NEBIUS_API_KEY" in r.json()["detail"]
