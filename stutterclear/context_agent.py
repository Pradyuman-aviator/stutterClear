"""Context agent: you say where you are about to speak, and NVIDIA Nemotron fills the context card with
the names, terms and phrases you are likely to need, which word rescue then draws on. With a Tavily
key it first searches the web for the situation (an interviewer's company, a conference, a topic),
so the card holds real names and terms rather than guesses. Text only; opt-in.

TAVILY_API_KEY is optional and, like NEBIUS_API_KEY, comes from the environment.
"""
import json
import os
import urllib.request

from . import nemotron

TAVILY_URL = "https://api.tavily.com/search"

SYSTEM = (
    "You prepare a context card for a person who stutters, before they speak. The card helps an "
    "assistant guess the word they are stuck on, so it lists the words they will most likely need: "
    "names of people, organisations, products and places, technical terms, and a few key phrases. "
    "Use only what the person wrote and what the search results say; never invent personal facts. "
    "Format: one short line naming the situation, then 'Words: ' followed by 10 to 25 comma-"
    "separated items, most important first. Plain text, under 90 words."
)


def search_available() -> bool:
    return bool(os.environ.get("TAVILY_API_KEY"))


def search(query: str, n: int = 5) -> list:
    """Tavily results as {title, url, content}, or [] without a key."""
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return []
    body = json.dumps({"api_key": key, "query": query[:300], "max_results": n, "search_depth": "basic"}).encode()
    req = urllib.request.Request(TAVILY_URL, data=body, method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=12.0) as r:
            results = json.load(r).get("results", [])
    except Exception:
        return []  # the card still gets made from what the person wrote
    return [{"title": x.get("title", ""), "url": x.get("url", ""), "content": (x.get("content") or "")[:400]}
            for x in results]


def messages(situation: str, current: str, found: list) -> list:
    user = f"Where they are about to speak: {situation.strip()}"
    if current.strip():
        user += f"\nWhat their card says already: {current.strip()}"
    if found:
        user += "\nSearch results:\n" + "\n".join(f"- {f['title']}: {f['content']}" for f in found)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def fill(situation: str, current: str = "") -> dict:
    found = search(situation)
    card = nemotron.chat(messages(situation, current, found), max_tokens=500, timeout=30.0)
    return {"card": card.strip(), "sources": [f["url"] for f in found if f["url"]], "searched": bool(found)}
