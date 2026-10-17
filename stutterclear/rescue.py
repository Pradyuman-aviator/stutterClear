"""Blocked-word rescue with NVIDIA Nemotron on Nebius Token Factory.

When you are stuck on a word, suggest the few words you most likely mean, from what you have said
so far and your context card. You choose; nothing is ever said for you. Opt-in: it sends recent
text (never audio) to Nebius.
"""
import re

from . import nemotron
from .nemotron import available  # noqa: F401  (re-exported for the server)

SYSTEM = (
    "You help a person who stutters during a live conversation. They are stuck on the next word "
    "(a speech block). From what they have said so far and the facts they gave about themselves, "
    "guess the next word or short phrase they most likely intend. Prefer names and terms from their "
    "facts when they fit. If the first sound they are stuck on is given, every guess must start with "
    "that sound. Reply with only a JSON array of 3 different guesses, most likely first, each at most "
    "3 words, no explanations."
)


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
    found = nemotron.last_json(text, "[")
    items = [str(v) for v in found] if isinstance(found, list) else []
    if not items:
        items = [line.strip(" -*0123456789.\"'") for line in text.splitlines()]
    out = []
    for v in items:
        v = " ".join(v.strip().strip("\"'.,!?").split()[:3])
        if v and v.lower() not in {o.lower() for o in out}:
            out.append(v)
    return out[:3]


def suggest(said: str, context: str = "", onset: str = "") -> list:
    return parse(nemotron.chat(messages(said, context, onset), max_tokens=400, timeout=8.0))
