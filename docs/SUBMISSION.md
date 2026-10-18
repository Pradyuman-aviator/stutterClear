# StutterClear: Devpost submission kit

Everything for the Nebius x NVIDIA Global AI Hackathon submission: the Devpost text, how the project
meets each judging criterion, the demo video script, and testing steps for judges. Numbers are from
`README.md` (7 speakers the models never trained on); update them if a new model is evaluated.

Gallery image: `docs/img/app.jpg` (a real recording playing, with the Fluency Mirror). Add one of
the guard line and one of live captions once Nemotron and the hosted demo are up.

## Name and tagline

**StutterClear**: your real voice, without the repeats.

Elevator pitch (Devpost, 200 characters): Real-time and recorded stutter removal in your own voice.
NVIDIA Nemotron on Nebius checks every cut, prepares your words, and helps when you're stuck.

## Devpost text

### Inspiration

About 1% of adults stutter, around 70 million people. Video calls, interviews and voice messages
are where it costs the most: people switch words, avoid calls, or skip speaking up. Existing tools
either retrain speech over months or replace your voice with a synthetic one. We wanted the
listener to hear *you*, just without the repeats, and to keep every word you meant.

### What it does

- **Live**: speak into your mic; StutterClear plays your own voice about half a second later with
  repeated words and broken-off starts removed, to headphones or into Zoom/Meet via a virtual
  cable. A Fluency Mirror shows what you said, what was removed, and what the listener hears;
  NVIDIA FastConformer captions show what the listener hears, live.
- **Voice notes**: record or upload a message; get it back cleaned, cutting only between words.
- **Word rescue**: when you get stuck on a word, Nemotron suggests the few words you most likely
  mean, from what you've said and your context card. You choose; nothing is said for you.
- **Context agent**: say where you'll speak; Nemotron (with Tavily search) prepares the names and
  terms you'll need.
- **Meaning guard**: before a recording is cleaned, Nemotron reviews every proposed cut and keeps
  repeats that were meant ("that that is true", "had had", "very very"). Anyone can try it on a
  typed sentence under "How StutterClear works".
- **Insights**: three short, supportive notes after a session. No diagnosis, no cure claims.

### How we built it

- **Detection**: a WavLM-Large teacher with a BiGRU head, 5 stutter classes, trained on Kaggle
  GPUs with SEP-28k / FluencyBank, synthetic stutters spliced into LibriSpeech, and silver labels
  we derived from CrisperWhisper verbatim transcripts of real stuttered speech.
- **Live**: the teacher runs on sliding windows (4 s past + 200 ms lookahead, a decision every
  160 ms; 60 ms per window on a laptop RTX 4050). A splicer delays your voice, skips what was
  marked, never plays audio before it has been judged, rebuilds the delay in natural pauses, and
  fills gaps with your own room tone.
- **Recordings**: the teacher finds stutters; the verbatim transcript places every cut on a word
  boundary; Nemotron's meaning guard reviews each cut.
- **NVIDIA Nemotron on Nebius Token Factory** in four roles: meaning guard, word rescue, context
  agent, insights. **NVIDIA FastConformer** (NeMo streaming, via sherpa-onnx) for live captions.
  **Nebius AI Cloud** hosts the demo on a GPU VM (Docker + Caddy).
- **App**: FastAPI + WebSockets; React UI with a WebGL "liquid" background, the ElevenLabs orb,
  live waveform mirror, A/B playback.

### Challenges we ran into

- **Jerky edits.** Our first cuts sounded jerky. Measuring every join showed why: the model's
  frame edges are ~100 ms off (a syllable), so 43% of cuts sliced into a real word. Placing cuts
  on word boundaries from the transcript removed every sliced word.
- **Live vs. lookahead.** A small causal student was too weak (it lost 8% of words); running the
  big teacher on short sliding windows kept nearly all of its offline quality.
- **A stutter slipping through.** An end-to-end test showed the GPU model's warm-up let undecided
  audio play; the splicer now never plays audio the detector has not judged.

### Accomplishments we're proud of

On 7 speakers the models never trained on, through the real live pipeline:

| | stutters removed | words lost |
|---|---|---|
| Live, rule-based baseline | 7.6% | 1.25% |
| Live, teacher on GPU, Gentle (keeps fillers) | 33.1% | 3.10% |
| Live, teacher on GPU, Strong (fillers too) | 65.2% | 5.68% |
| Recordings, word-aligned: Natural / Thorough (fillers, repeated phrases) | 50.1% / 80.7% | 0 words sliced |

On the standard speaker-exclusive SEP-28k-E test set, teacher v2 reaches macro F1 0.618 (AUC 0.98
for word repetitions, 0.96 for fillers).

### What we learned

Listening beats metrics: the version that scored best on paper sounded worse, and the reason was
measurable once we looked at the joins. And a stutter is not a repeated word: only a language model
can tell "I I I want" from "that that is true".

### What's next

User testing with people who stutter (and their speech therapists), a smaller live model for
laptops without a GPU, Zoom/Meet one-click setup, and more languages.

### Built with

NVIDIA Nemotron, Nebius Token Factory, Nebius AI Cloud, NVIDIA NeMo FastConformer, sherpa-onnx,
WavLM, PyTorch, CrisperWhisper, FastAPI, React, Vite, Tailwind, three.js, Tavily, Kaggle.

## How it meets the criteria

| Criterion | Where it shows |
|---|---|
| Technological implementation | Nemotron in four roles in the core flows; Token Factory API; hosted on Nebius AI Cloud GPU; measured live pipeline with a GPU model on sliding windows; NVIDIA streaming ASR |
| Design | One coherent app: live, voice notes, real-recording demo, captions, A/B playback, insights; privacy defaults (nothing stored; a live session reaches only its own tab); works on phones; plain-language errors (mic blocked, busy, someone else's turn) |
| Potential impact | 70M people; calls and voice messages; keeps the speaker's own voice; honest numbers on unseen speakers |
| Quality of the idea | Nemotron as a meaning guard for speech editing (an LLM deciding which repeats were meant, try it on any sentence); word rescue for blocks |

## Demo video (under 3 minutes)

| Time | Shot | Say |
|---|---|---|
| 0:00-0:15 | Face to camera, or title card | "About 70 million people stutter. On calls, the repeats are what people hear first." |
| 0:15-0:40 | App home; tap a real recording; play Original, then switch to Clean voice | "This is a real person who stutters. Here's the original... and here's the same recording, his own voice, without the repeats." |
| 0:40-1:05 | Point at the meaning guard line and the mirror's red cuts; then "Try the meaning guard" with "I I I think that that plan is very very good" | "Every cut was checked by NVIDIA Nemotron on Nebius. A repeat isn't always a stutter: here it cuts 'I I I' but keeps 'that that' and 'very very', because they were meant." |
| 1:05-1:40 | Live: Start, speak a sentence with a stutter; captions appear; Stop; play Clean voice | "Live, it runs half a second behind me, on the GPU. Captions are NVIDIA FastConformer." |
| 1:40-2:05 | Word rescue: get stuck, chips appear | "When I get stuck, Nemotron suggests the word I mean. I choose; it never speaks for me." |
| 2:05-2:25 | Context agent: type a situation, Prepare | "Before an interview, it prepares the names and terms I'll need." |
| 2:25-2:45 | Voice note: record, cleaned, Insights | "Voice notes too, with a few supportive notes afterwards." |
| 2:45-3:00 | Numbers slide / README table | "Measured on speakers it never saw. Open source, running on Nebius." |

Use headphones while recording the live part (laptop speakers feed back into the mic).

## Testing instructions for judges

1. Open the hosted link (Chrome or Edge; allow the microphone).
2. No mic needed: under Start, pick a real recording, play it, switch Original / Clean voice and
   Natural / Thorough / Live; click "Insights · Nemotron".
3. Live: Setup shows "This browser's mic"; press Start, speak for 20 seconds, press Stop, then play
   the Clean voice. One person can be live at a time, for up to 5 minutes. Turn on Word rescue on the Context card first to see suggestions.
4. Voice note: "Clean a voice note" on the right; record or upload, then play it back.
5. Meaning guard: under "How StutterClear works", press Check on the example sentence (or type
   your own); Nemotron marks the repeats it would cut and the ones it keeps, with reasons.
