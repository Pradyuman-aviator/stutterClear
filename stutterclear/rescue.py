"""Blocked-word rescue with NVIDIA Nemotron on Nebius Token Factory.

When you are stuck on a word, suggest the few words you most likely mean, from what you have said
so far and your context card. You choose; nothing is ever said for you. Opt-in: it sends recent
text (never audio) to Nebius. Needs NEBIUS_API_KEY in the environment; NEBIUS_MODEL and
NEBIUS_BASE_URL override the defaults.
"""
import json
import os
import re
import urllib.request

BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.us-central1.nebius.com/v1/")
MODEL = os.environ.get("NEBIUS_MODEL", "nvidia/nemotron-3-super-120b-a12b")
TIMEOUT_S = 8.0

SYSTEM = (
    "You help a person who stutters during a live conversation. They are stuck on the next word "
    "(a speech block). From what they have said so far and the facts they gave about themselves, "
    "guess the next word or short phrase they most likely intend. Prefer names and terms from their "
    "facts when they fit. If the first sound they are stuck on is given, every guess must start with "
    "that sound. Reply with only a JSON array of 3 different guesses, most likely first, each at most "
    "3 words, no explanations."
)


def available() -> bool:
    return bool(os.environ.get("NEBIUS_API_KEY"))


def messages(said: str, context: str, onset: str = "") -> list:
    said = " ".join(said.split()[-60:])  # the last few sentences are enough, and keep it private
    user = f"Facts they gave: {context.strip() or '(none)'}\nWhat they have said so far: \"{said}\""
    if onset:
        user += f"\nThe sound they are stuck on: \"{onset}\""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse(text: str) -> list:
    """The model's reply -> up to 3 short, distinct suggestions. Tolerates prose around the JSON
    and models that think out loud before answering."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    found = re.findall(r"\[[^\[\]]*\]", text)
    items = []
    for chunk in reversed(found):  # the answer is usually the last array
        try:
            items = [str(v) for v in json.loads(chunk)]
            break
        except json.JSONDecodeError:
            continue
    if not items:
        items = [line.strip(" -*0123456789.\"'") for line in text.splitlines()]
    out = []
    for v in items:
        v = " ".join(v.strip().strip("\"'.,!?").split()[:3])
        if v and v.lower() not in {o.lower() for o in out}:
            out.append(v)
    return out[:3]


def suggest(said: str, context: str = "", onset: str = "") -> list:
    key = os.environ.get("NEBIUS_API_KEY")
    if not key:
        raise RuntimeError("Set NEBIUS_API_KEY to use word rescue.")
    body = json.dumps({"model": MODEL, "messages": messages(said, context, onset),
                       "temperature": 0.2, "max_tokens": 400}).encode()
    req = urllib.request.Request(BASE_URL.rstrip("/") + "/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        reply = json.load(r)
    return parse(reply["choices"][0]["message"]["content"] or "")
