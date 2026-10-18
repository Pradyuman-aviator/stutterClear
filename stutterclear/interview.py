"""Practice: a mock interview with NVIDIA Nemotron. The person names a role and a company (and may
paste the job post); the context agent researches it into a context card (which word rescue then
draws on), and Nemotron asks realistic questions one at a time, following up on the answers. After
the last question it writes three short practice points.

The interviewer never comments on how the person speaks. Text only (the questions, and the answers
as transcribed by the captions); the audio stays on the server, in memory. Opt-in.
"""
import re

from . import context_agent, nemotron

QUESTIONS = 5  # a practice round
KEEP = 6  # question/answer pairs the interviewer sees

SYSTEM = (
    "You are a friendly, realistic interviewer running a practice interview. Ask exactly one "
    "question, under 40 words. Fit it to the role and company and the facts in the context card. "
    "If their last answer invites a follow-up, ask it; otherwise move to a new topic (motivation, "
    "experience, a technical or role-specific question, a situation they handled, their questions "
    "for you). Never comment on how they speak, their fluency or stuttering; never give feedback "
    "during the interview. Reply with only the question."
)

SUMMARY = (
    "You coach a person who stutters after a practice interview, from the questions and their "
    "transcribed answers (transcripts may contain recognition errors: ignore wording slips). "
    "Exactly 3 bullets, each under 30 words: one answer that worked well and why, one answer to "
    "strengthen (what to add), and one question to prepare for next time. About content only: "
    "never mention fluency, stuttering or how they speak. Plain text, each bullet starting with '- '."
)


def _history(history: list) -> str:
    lines = []
    for h in history[-KEEP:]:
        lines.append(f"Interviewer: {h.get('question', '').strip()}")
        answer = " ".join(str(h.get("answer", "")).split()[-150:])
        lines.append(f"Candidate: {answer or '(no transcript)'}")
    return "\n".join(lines)


def messages(role: str, company: str, card: str, history: list) -> list:
    user = f"Role: {role.strip()}\nCompany: {company.strip() or '(not given)'}"
    if card.strip():
        user += f"\nContext card: {card.strip()}"
    user += ("\nThe interview so far:\n" + _history(history) if history else "\nThis is the first question.")
    user += f"\nAsk question {len(history) + 1} of {QUESTIONS}."
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_question(text: str) -> str:
    """The question alone: no label, quotes or talk around it."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    asks = [ln for ln in lines if ln.endswith("?")]
    q = asks[-1] if asks else (lines[-1] if lines else "")
    q = re.sub(r"^(\*\*)?(question( \d+)?|interviewer)\s*[:.-]\s*(\*\*)?", "", q, flags=re.I)
    return q.strip().strip("\"'*").strip()


def next_question(role: str, company: str, card: str = "", history: list | None = None) -> str:
    q = parse_question(nemotron.chat(messages(role, company, card, history or []), max_tokens=300,
                                     temperature=0.6, timeout=20.0))
    if not q:
        raise nemotron.NemotronError("Nemotron did not ask a question.")
    return q


def start(role: str, company: str, posting: str = "") -> dict:
    """Research the role (context card), then the first question."""
    situation = f"Job interview for {role.strip()} at {company.strip() or 'a company'}"
    if posting.strip():
        situation += f". Job post: {' '.join(posting.split()[:300])}"
    found = context_agent.fill(situation)
    return {**found, "question": next_question(role, company, found["card"]), "questions": QUESTIONS}


def summary(role: str, company: str, history: list) -> list:
    user = f"Role: {role.strip()} at {company.strip() or '(not given)'}\n" + _history(history)
    text = nemotron.chat([{"role": "system", "content": SUMMARY}, {"role": "user", "content": user}],
                         max_tokens=500, temperature=0.4, timeout=30.0)
    from .insights import parse
    return parse(text)
