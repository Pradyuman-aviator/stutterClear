"""Meaning guard: NVIDIA Nemotron reviews every proposed cut in a recording before it is applied.

A transcript alone cannot tell a stutter ("I I I want") from speech that repeats a word on purpose:
grammar ("I know that that is true", "she had had enough", "what it is is"), emphasis ("very very
good", "no no no"), or a word whose removal would change the meaning. The acoustic model cannot
either. Nemotron reads each candidate in its sentence and keeps those the person meant to say.
Without a key, or if Nemotron does not answer, the cuts are applied as the transcript and the
acoustic model agreed, exactly as before.
"""
from . import nemotron
from .silver import word_cuts

SYSTEM = (
    "You are a careful speech editor helping a person who stutters. You get a verbatim transcript "
    "of their speech. Words marked like [12:word] are candidates for deletion because they look "
    "like stuttering: a repeated word, a broken-off start of a word (ending in '-'), or a filler "
    "such as [UH] or [UM]. For each candidate decide:\n"
    "- DELETE when it is stuttering: an involuntary repetition, a false start or a filler, so the "
    "sentence says what the person meant without it.\n"
    "- KEEP when the person said it on purpose: grammar that needs the repeated word ('that that', "
    "'had had', 'is is' in 'what it is is'), deliberate emphasis ('very very', 'no no no'), or when "
    "deleting it would change or lose meaning.\n"
    "Most candidates are real stutters. Reply with only JSON in this form, listing the candidates to "
    'KEEP (an empty list if every candidate should be deleted): {"keep": [{"id": 12, "reason": '
    '"grammatical: had had"}]}'
)


def marked_transcript(toks: list, candidates: set) -> str:
    return " ".join(f"[{i}:{t['w']}]" if i in candidates else t["w"] for i, t in enumerate(toks))


def review(toks: list, candidates: set) -> dict:
    """Nemotron's verdicts: {token index: reason} for the candidates to keep."""
    reply = nemotron.chat([{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": marked_transcript(toks, candidates)}],
                          max_tokens=4000, timeout=90.0, think=True)  # a judgement call: let it reason
    data = nemotron.last_json(reply, "{")
    keep = {}
    for item in (data or {}).get("keep", []) if isinstance(data, dict) else []:
        idx = item.get("id") if isinstance(item, dict) else item
        reason = str(item.get("reason", "")) if isinstance(item, dict) else ""
        if isinstance(idx, int) and idx in candidates:
            keep[idx] = reason[:120]
    return keep


def guarded_cuts(toks: list, probs=None, cut_fillers: bool = False, cut_phrases: bool = False) -> tuple:
    """Word-aligned cuts, checked by Nemotron. Returns (cuts, report); report is None without a key."""
    cuts = word_cuts(toks, probs, cut_fillers=cut_fillers, cut_phrases=cut_phrases)
    candidates = {i for c in cuts for i in c["tokens"]}
    if not candidates or not nemotron.available():
        return cuts, None
    try:
        keep = review(toks, candidates)
    except nemotron.NemotronError as e:
        return cuts, {"model": nemotron.MODEL, "checked": len(candidates), "kept": [], "error": str(e)}
    kept = [{"word": toks[i]["w"], "context": " ".join(t["w"] for t in toks[max(0, i - 3):i + 4]), "reason": r}
            for i, r in sorted(keep.items())]
    return (word_cuts(toks, probs, cut_fillers=cut_fillers, keep=frozenset(keep), cut_phrases=cut_phrases),
            {"model": nemotron.MODEL, "checked": len(candidates), "kept": kept})
