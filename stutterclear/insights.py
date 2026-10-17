"""Private session insights: after a session or a voice note, NVIDIA Nemotron turns the numbers (kinds
of stutters, how many, which words, how long) and the transcript into three short, supportive,
practical points. It never diagnoses, judges or promises a cure. Text only; opt-in.
"""
from . import nemotron

SYSTEM = (
    "You write a short, warm, practical note for a person who stutters about one stretch of their "
    "own speech, from counts and a transcript. Exactly 3 bullets, each under 25 words: something "
    "that went smoothly, which sounds or words were hardest (name them), and one concrete thing to "
    "try or prepare next time (pausing, pacing, rehearsing a hard word). Stuttering is a normal way "
    "of speaking: never diagnose, never promise a cure, never judge or use the word 'fix'. Plain "
    "text, each bullet starting with '- '."
)

KIND_WORDS = {"word_rep": "word repeats", "sound_rep": "broken-off starts", "prolongation": "stretched sounds",
              "interjection": "fillers", "block": "blocks", "repetition": "repeats"}


def messages(kinds: dict, duration_s: float, removed_s: float, words: list, transcript: str) -> list:
    counts = ", ".join(f"{n} {KIND_WORDS.get(k, k)}" for k, n in sorted(kinds.items(), key=lambda kv: -kv[1]) if n)
    user = (f"Length: {duration_s:.0f} s. Stutters StutterClear removed: {counts or 'none'} "
            f"({removed_s:.1f} s in total).")
    if words:
        user += f"\nWords the stutters were on: {', '.join(words[:30])}"
    if transcript.strip():
        user += f"\nTranscript: \"{' '.join(transcript.split()[-250:])}\""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse(text: str) -> list:
    bullets = [line.strip()[1:].strip() for line in text.splitlines() if line.strip().startswith(("-", "•", "*"))]
    if not bullets:
        bullets = [s.strip() for s in text.split("\n") if s.strip()]
    return [b for b in bullets if b][:3]


def summarize(kinds: dict, duration_s: float, removed_s: float, words: list | None = None, transcript: str = "") -> list:
    return parse(nemotron.chat(messages(kinds, duration_s, removed_s, words or [], transcript),
                               max_tokens=500, temperature=0.4, timeout=30.0))
