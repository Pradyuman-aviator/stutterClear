# StutterClear

**Your real voice, on your terms.** StutterClear removes stutters (repeated words, broken-off
starts, and optionally fillers) from your speech, in **your own voice**: live in calls, or from
recordings and voice notes. Nothing is re-synthesised; the listener hears you, minus the repeats.

It is an assistive tool, not a treatment, and it never speaks for you. Stuttering is part of how many
people talk; StutterClear gives you a choice about what a listener hears.

```
                 ┌──────────── stutter detector (big model on GPU, or small model / rules) ──┐
mic ──► delay buffer (0.25-0.85 s) ──► splicer: skip the stutter, join with room tone ─────────┴──► call / headphones
```

## Three ways to use it

| | What it is | How well it works today (7 speakers it never trained on) |
|---|---|---|
| **Live** | Your mic (or your browser's, on the hosted demo), cleaned in real time, sent to headphones or a virtual mic (Zoom, Meet). | Big model on a laptop GPU: **32% of stutters removed in Gentle** (fillers kept, 3.2% of words lost) and **65% in Strong** (fillers too, 5.6% lost), about 0.5-0.85 s behind you. Without a GPU the rule-based detector removes ~8%. |
| **Clean a voice note** | Record or upload a message in the browser and get it back cleaned. | Cuts land **between words only**: about 49% of stutters removed (75% if fillers go too) with **no real word cut**. |
| **Try a real recording** | Three real stuttered recordings (UCLASS, adult speakers) with A/B playback. | Same engines as above, side by side. |

The app also shows a **Fluency Mirror**: your voice and what the listener hears, scrolling live,
with every removed stutter marked and labelled, and **live captions** of what the listener hears
(NVIDIA FastConformer, streaming, on the CPU).

### NVIDIA Nemotron on Nebius Token Factory

Nemotron does the parts that need an understanding of language (text only, never audio; each one
opt-in or shown on demand):

- **Meaning guard.** Before a recording is cleaned, Nemotron reviews every proposed cut in its
  sentence and keeps repeats the person meant: grammar ("I know that that is true", "she had had
  enough"), emphasis ("very very good"). Neither the transcript nor the acoustic model can tell.
- **Word rescue.** Stuck on a word (a block)? Nemotron suggests the few words you most likely mean,
  from what you have said and your context card. You choose; nothing is said for you.
- **Context agent.** Say where you will speak; Nemotron (with Tavily web search, if set up) fills
  the context card with the names and terms you will need.
- **Insights.** Three short, supportive notes after a session or recording. No diagnosis, no cure
  claims.

Set `NEBIUS_API_KEY` (and optionally `TAVILY_API_KEY`) in the environment where the app runs.

## How it works

1. **Detection.** A teacher model (WavLM-Large with a small BiGRU head, 5 classes: sound repetition,
   word repetition, prolongation, block, interjection) is trained on SEP-28k / FluencyBank plus
   synthetic stutters spliced into LibriSpeech, plus silver labels from verbatim transcripts
   (CrisperWhisper) of real stuttered speech.
2. **Live.** The teacher runs on sliding windows (4 s of past audio plus a short lookahead, a
   decision every 160 ms); a causal student model and a rule-based detector are the fallbacks for
   machines without a GPU. The splicer plays your voice a fraction of a second late, skips what was
   marked, never plays audio the detector has not judged yet, rebuilds the delay in natural pauses,
   and fills any gap with your own room tone (never digital silence).
3. **Recordings.** The teacher finds the stutters, the verbatim transcript places every cut on a
   word boundary, and Nemotron's meaning guard reviews each cut, removing only repeated and
   broken-off words the person did not mean. Model frame edges are about 100 ms
   off, a whole syllable: cutting on them sliced a real word in 30-43% of cuts, which is exactly
   what sounds like a jerk; word-aligned cutting slices none.

## Measured results

All on 7 held-out UCLASS speakers (23 min), scored against silver labels from CrisperWhisper.
"Removed" = stutter tokens removed; "lost" = fluent words cut by more than half.

**Live, end to end** (`scripts/eval_live.py`: each recording streamed 10 ms at a time through the
real pipeline: delay buffer, online detector, splicer):

| detector | Gentle (fillers kept): removed / lost | Strong (fillers removed too): removed / lost |
|---|---|---|
| rule-based | 7.6% / 1.25% | 10.2% / 1.45% |
| causal student v1 | 19.8% / 8.37% | 19.5% / 8.33% |
| teacher on GPU (sliding windows) | 31.7% / 3.18% | 65.2% / 5.64% |

**How much lookahead the big model needs** (`scripts/teacher_lookahead.py`: probabilities from
sliding windows, cut with the offline recipe, fillers included):

| lookahead | removed | lost |
|---|---|---|
| whole recording (offline) | 58.4% | 5.76% |
| 200 ms | 53.8% | 4.63% |
| 500 ms | 60.1% | 5.11% |
| 1 s | 66.3% | 4.95% |

One 5 s window takes 60 ms on a laptop RTX 4050, so it keeps up live with room to spare.

**Recordings** (`stutterclear/silver.py: word_cuts`):

| recipe | real words sliced | real words removed | stutters removed |
|---|---|---|---|
| model edges, natural settings | 98 | 183 | 49.9% |
| word-aligned | 0 | 0 | 48.7% |
| word-aligned + fillers | 0 | 0 | 75.4% |

**Causal student, v1** (0.3M parameters): too small. SEP-28k macro F1 0.38 vs the teacher's 0.65,
and live it removes 19.8% of stutters while losing 8.4% of words, so the app does not use it. A
5.2M-parameter student v2 is training. The app recommends whichever detector measures best
(removed minus 3 x lost); `scripts/eval_live.py` produces those numbers.

## Run it

Windows 10/11, Python 3.11+, a mic. A CUDA GPU enables the big model live and voice notes.

```
pip install -r requirements.txt
python scripts/run_app.py          # opens http://127.0.0.1:8765 (or the next free port)
python -m pytest -q
```

Optional: `python scripts/download_asr.py` for live captions (NVIDIA FastConformer, 106 MB), and
`NEBIUS_API_KEY` for the Nemotron features. The hosted demo setup (Nebius AI Cloud GPU VM, Docker,
HTTPS) is in `docs/DEPLOY.md`; the hackathon submission kit is in `docs/SUBMISSION.md`.

Use headphones for live playback (laptop speakers feed back into the mic). For calls, install a
virtual audio cable (e.g. VB-CABLE), pick "CABLE Input" as the output in Setup, and "CABLE Output"
as the microphone in Zoom / Meet.

The UI is a Vite + React app in `stutterclear/app/web`; its build is committed in
`stutterclear/app/static`, so Node is only needed to change it (`npm install && npm run build`).
Model checkpoints are not in the repository; the Kaggle jobs in `kaggle/` train them.

## Privacy

- Audio is processed on your computer. Nothing is recorded unless you switch saving on.
- The last session and voice notes are kept in memory for playback (voice notes for at most an
  hour) and are never written to disk. The hosted demo never stores recordings.
- Nemotron features send text only (recent captions, the context card, counts), never audio, and
  only when you use them.
- The context card stays in your browser.

## Data and licences

The code is MIT licensed (see `LICENSE`). Data and models keep their own terms:

- **UCLASS**, University College London Archive of Stuttered Speech (Howell, Davis & Bartrip,
  2009). Data collection was supported by the Wellcome Trust. Free for research and teaching; the
  demo uses short excerpts from adult speakers, read from a local download and not redistributed.
- **SEP-28k / FluencyBank**: CC BY-NC 4.0 (non-commercial).
- **CrisperWhisper**: non-commercial licence; used to create silver labels and to place cuts.
- **LibriSpeech**: CC BY 4.0. **WavLM**: MIT.
- UI components from ElevenLabs UI and shadcn/ui (MIT).

Commercial use would need a commercial UCLASS licence and models trained only on commercially
licensed or own consented recordings.

## Repository map

- `stutterclear/`: splicer, detectors (rules, student, teacher), cut and join logic, live session
- `stutterclear/app/`: FastAPI server, demo samples, voice notes, React UI
- `scripts/`: training, evaluation (`eval_live.py`, `eval_splice.py`, `teacher_lookahead.py`,
  `join_quality.py`, `check_audio.py`), data downloads
- `kaggle/`: GPU training jobs
- `reports/`, `research_notes/`, `docs/`: literature review, novelty check, learning guide
