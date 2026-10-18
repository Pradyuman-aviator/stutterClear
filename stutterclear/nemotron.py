"""NVIDIA Nemotron on Nebius Token Factory (OpenAI-compatible chat API), shared by every feature
that reasons about language: word rescue, the meaning guard, the context agent, session insights.

NEBIUS_API_KEY comes from the environment (never from code or chat); NEBIUS_MODEL and
NEBIUS_BASE_URL override the defaults. Only text is ever sent, never audio.
"""
import json
import os
import re
import urllib.error
import urllib.request

BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
MODEL = os.environ.get("NEBIUS_MODEL", "nvidia/nemotron-3-super-120b-a12b")


class NemotronError(RuntimeError):
    pass


def available() -> bool:
    return bool(os.environ.get("NEBIUS_API_KEY"))


def models(timeout: float = 10.0) -> list:
    """The model ids this endpoint serves."""
    req = urllib.request.Request(BASE_URL.rstrip("/") + "/models",
                                 headers={"Authorization": f"Bearer {os.environ.get('NEBIUS_API_KEY', '')}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return [m["id"] for m in json.load(r).get("data", [])]


def resolve() -> str:
    """Make sure MODEL is served here; if not (ids change), switch to the closest Nemotron: Super,
    then Ultra, then Nano. Returns a line for the log."""
    global MODEL
    try:
        ids = models()
    except Exception as e:
        return f"Nemotron: could not list models ({e}); using {MODEL}"
    if MODEL in ids:
        return f"Nemotron: {MODEL}"
    rank = {"super": 3, "ultra": 2, "nano": 1}
    found = sorted((i for i in ids if "nemotron" in i.lower()),
                   key=lambda i: (max([v for k, v in rank.items() if k in i.lower()], default=0), i))
    if not found:
        return f"Nemotron: {MODEL} is not listed and no Nemotron model is; check NEBIUS_BASE_URL"
    old, MODEL = MODEL, found[-1]
    return f"Nemotron: {old} is not served here; using {MODEL}"


def _post(body: dict, key: str, timeout: float) -> dict:
    req = urllib.request.Request(BASE_URL.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat(messages: list, max_tokens: int = 400, temperature: float = 0.2, timeout: float = 20.0,
         think: bool = False) -> str:
    """One reply. Nemotron 3 reasons before answering unless told not to; quick features turn that
    off (think=False) so the token budget goes to the answer. If the server does not take the
    switch, retry once without it and with room for the reasoning."""
    key = os.environ.get("NEBIUS_API_KEY")
    if not key:
        raise NemotronError("Set NEBIUS_API_KEY where the app runs to use Nemotron.")
    body = {"model": MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": think}}
    try:
        try:
            reply = _post(body, key, timeout)
        except urllib.error.HTTPError as e:
            if e.code not in (400, 422):
                raise
            body.pop("chat_template_kwargs")
            body["max_tokens"] = max_tokens + 2048
            reply = _post(body, key, timeout + 20)
    except Exception as e:
        raise NemotronError(f"Nemotron did not answer: {e}") from e
    msg = reply["choices"][0]["message"]
    text = re.sub(r"<think>.*?</think>", "", msg.get("content") or "", flags=re.S).strip()
    if not text:
        raise NemotronError("Nemotron spent its whole reply reasoning; no answer came back.")
    return text


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
