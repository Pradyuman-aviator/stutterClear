"""Check the Nebius key and every Nemotron feature in one go, before a demo.

    python scripts/check_nemotron.py

Needs NEBIUS_API_KEY in the environment (TAVILY_API_KEY optional). Sends only the short made-up
texts below. The meaning-guard case has a known answer: "that that" is grammar and must be kept,
"I I I" is a stutter and must go.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stutterclear import context_agent, guard, insights, nemotron, rescue  # noqa: E402
from stutterclear.silver import label_tokens  # noqa: E402


def step(name, fn):
    t0 = time.perf_counter()
    try:
        out = fn()
        print(f"  ok    {name} ({time.perf_counter() - t0:.1f} s): {out}")
        return True
    except Exception as e:
        print(f"  FAIL  {name} ({time.perf_counter() - t0:.1f} s): {e}")
        return False


def guard_case():
    said = "I I I want to say that I know that that is true"
    toks = label_tokens([{"w": w, "s": 0.4 * i, "e": 0.4 * i + 0.3} for i, w in enumerate(said.split())])
    candidates = {i for i, t in enumerate(toks) if t["label"] == "repeat"}
    keep = guard.review(toks, candidates)
    kept = {toks[i]["w"]: r for i, r in keep.items()}
    if set(kept) != {"that"}:
        raise RuntimeError(f"kept {kept}, expected to keep only 'that' (grammar) and cut the repeated 'I'")
    return f"kept {kept}, cut the repeated 'I': right"


def main():
    if not nemotron.available():
        sys.exit("NEBIUS_API_KEY is not set here. Windows: setx NEBIUS_API_KEY \"...\" and open a new terminal.")
    print(nemotron.resolve())
    print(f"endpoint {nemotron.BASE_URL}")
    results = [
        step("chat", lambda: nemotron.chat([{"role": "user", "content": "Reply with just: ready"}], max_tokens=20)),
        step("word rescue", lambda: rescue.suggest("Tomorrow I have an interview at", "Company: NVIDIA", "N")),
        step("meaning guard", guard_case),
        step("insights", lambda: insights.summarize({"word_rep": 4, "sound_rep": 2}, 60, 3.2)),
        step("context agent" + (" + Tavily" if context_agent.search_available() else ""),
             lambda: context_agent.fill("Job interview with the speech AI team at NVIDIA")["card"][:200]),
    ]
    print("all good" if all(results) else "some features failed: see above")


if __name__ == "__main__":
    main()
