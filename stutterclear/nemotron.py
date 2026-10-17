"""NVIDIA Nemotron on Nebius Token Factory (OpenAI-compatible chat API), shared by every feature
that reasons about language: word rescue, the meaning guard, the context agent, session insights.

NEBIUS_API_KEY comes from the environment (never from code or chat); NEBIUS_MODEL and
NEBIUS_BASE_URL override the defaults. Only text is ever sent, never audio.
"""
import json
import os
import re
import urllib.request

BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.us-central1.nebius.com/v1/")
MODEL = os.environ.get("NEBIUS_MODEL", "nvidia/nemotron-3-super-120b-a12b")


class NemotronError(RuntimeError):
    pass


def available() -> bool:
    return bool(os.environ.get("NEBIUS_API_KEY"))


def chat(messages: list, max_tokens: int = 400, temperature: float = 0.2, timeout: float = 20.0) -> str:
    key = os.environ.get("NEBIUS_API_KEY")
    if not key:
        raise NemotronError("Set NEBIUS_API_KEY where the app runs to use Nemotron.")
    body = json.dumps({"model": MODEL, "messages": messages, "temperature": temperature,
                       "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(BASE_URL.rstrip("/") + "/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            reply = json.load(r)
    except Exception as e:
        raise NemotronError(f"Nemotron did not answer: {e}") from e
    text = reply["choices"][0]["message"].get("content") or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # drop visible reasoning


def last_json(text: str, opener: str = "{"):
    """The last complete JSON object (or array, opener='[') in a reply, taking the outermost one when
    they nest; models sometimes talk around their answer."""
    closer = "}" if opener == "{" else "]"
    best = None  # (end, -start, value)
    for i, ch in enumerate(text):
        if ch != opener:
            continue
        depth = 0
        for j in range(i, len(text)):
            depth += text[j] == opener
            depth -= text[j] == closer
            if depth == 0:
                try:
                    value = json.loads(text[i:j + 1])
                except json.JSONDecodeError:
                    break
                if best is None or (j, -i) > best[:2]:
                    best = (j, -i, value)
                break
    return best[2] if best else None
