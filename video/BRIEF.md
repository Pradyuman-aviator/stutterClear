# The StutterClear film: production notes

A 3-minute film for the Nebius × NVIDIA Global AI Hackathon, rendered with Remotion (React,
1920×1080, 30 fps) from real pipeline output. Its script is `storyboard.json` (13 scenes); its data
is `public/generated/data.json`, written by `scripts/export_video_data.py`. Each scene is one file,
`src/scenes/<id>.tsx`, built from the kit in `src/kit`.

```
python scripts/export_video_data.py     # data, narration, the clips (needs NEBIUS_API_KEY; NVIDIA_API_KEY for Magpie)
python scripts/make_video_sound.py      # music and sound effects (once)
cd video
node scripts/scaffold.mjs               # scene files, preview entries and the registry from storyboard.json
node scripts/look.mjs <scene>           # out/look/<scene>/sheet.png: nine moments of a scene
node scripts/look.mjs <scene> 30 90     # full frames f30.png, f90.png
node scripts/typecheck.mjs scenes/<scene>.tsx
npm run render                          # out/stutterclear.mp4
```

## The story

Alex (fictional) built an app that teaches travellers to read Japanese train signs. He stutters;
it is how he speaks. Tonight he records his demo-day pitch: his real voice (a real recording, see
below) travels through StutterClear's recordings engine. Tomorrow, live on the call, the live engine
holds his voice half a second, NeMo captions it, Nemotron rescues a word and Magpie says it. The
judges hear what he meant, on his terms. Both versions say what he meant; which one you hear is up
to him.

## Facts (anything not here, don't state)

**Product.** StutterClear removes stutters from a person's own voice, live in calls and on
recordings. "Your real voice, on your terms." It is optional; stuttering is a normal way of speaking.
Open source: github.com/Pradyuman-aviator/stutterClear; run it with `python scripts/run_app.py`
(opens http://127.0.0.1:8765).

**The real clip.** A real recording of a 20-year-old man who stutters, from UCLASS (University
College London Archive of Stuttered Speech; Howell, Davis & Bartrip, 2009; data collection supported
by the Wellcome Trust). In the film the voice is Alex's; the film credits UCLASS and the Wellcome
Trust on screen while it plays and says Alex is fictional. The speaker describes reading train signs
in Japan. The window used (19.45 s), verbatim: "It is because they they they have it in their Chinese
writ- writ- writing there um and and and some and sometimes in in Roman l- l- l- letters" (Chinese
writing = kanji, Roman letters = romaji).
Cleaned (Thorough, real output, 8.25 s): "It is because they have it in their Chinese writing there
and some and sometimes in Roman letters". 18 cuts: 9 stutter cuts (repeated words, broken-off
starts, a filler) and 9 long silent pauses shortened to under half a second. Every stutter in this
take removed, no word lost. The meaning guard checked the 11 words marked for cutting in this take and
kept none: all were real stutters.

**Live (in a call).** Microphone, 16 kHz mono → a delay buffer that holds the voice back (Gentle about
0.5 s, Strong about 0.85 s) → the WavLM-Large teacher (316M parameters; self-supervised speech encoder;
a learned mix of all 25 hidden layers, then a 2-layer bidirectional GRU; 5 classes per 20 ms frame:
sound repetition, word repetition, prolongation, block, filler) on the laptop's NVIDIA GeForce RTX
4050 GPU with CUDA, on sliding windows: 4 s of past audio plus 200 ms lookahead (Gentle) or 500 ms
(Strong), re-run every 160 ms; one window takes about 60 ms → per-frame probabilities → hysteresis
cut logic (a cut opens when a probability crosses its threshold and closes when it falls; close cuts
merge; blocks are reported, never cut) → the splicer skips the stuttered audio before it is played,
crossfades the join and fills gaps with the room's own background sound (never digital silence) →
a virtual audio cable (e.g. VB-CABLE: StutterClear sends to "CABLE Input"; Zoom, Google Meet or
Microsoft Teams use "CABLE Output" as the microphone). Delay budget: lookahead (200 or 500 ms) + hop
(160 ms) + 100 ms for GPU time (about 60 ms) and margin, plus 80 ms headroom (0.54 s / 0.84 s).
Live captions of what the listener hears: NVIDIA NeMo streaming FastConformer
(stt_en_fastconformer_hybrid_large_streaming, 480 ms chunks, int8 ONNX via sherpa-onnx, on the CPU).

**Recordings** (voice notes, videos, dictation). The recordings teacher (WavLM, v2) scores every
20 ms frame; CrisperWhisper (a verbatim speech recogniser) writes every word with its timing,
stutters included; cuts are placed only between words, so no real word is ever clipped (repeated
words, broken-off starts; Thorough also fillers, repeated phrases, and long pauses shortened);
NVIDIA Nemotron reviews each proposed cut in its sentence, as text only, and keeps repeats that were
meant (grammar like "that that", emphasis like "very very"); the audio is re-joined with room tone.

**NVIDIA.**
1. NVIDIA Nemotron 3 Super (`nvidia/nemotron-3-super-120b-a12b`): 120B total parameters, 12B active
   per token; hybrid Mamba-2 + mixture-of-experts (LatentMoE) + attention; multi-token prediction;
   context up to 1M tokens; served on Nebius Token Factory through an OpenAI-compatible API. Used for
   the meaning guard, word rescue, the context card, session insights and the practice interview.
   It only ever receives text, never audio.
   - Real meaning-guard exchange (`d.guard`) on a typed test sentence: request
     `[0:I] I know [3:that] that sign means the exit, and it is [12:very] very easy to miss.` → the
     reply in `d.guard.reply` (keeps "that that" as grammar and "very very" as emphasis; cuts the
     extra "I"). Nemotron does not give the same verdict every time: this one was the most common in
     `d.guard.agreed` runs (5 of 5 in a separate test).
   - Real word rescue (`d.rescue`): the caption so far, the stuck sound "l", the context card "Demo
     day pitch for my app: it teaches travellers to read Japanese train signs." → three suggestions
     (`d.rescue.suggestions`). Alex taps "letters".
2. NVIDIA NeMo streaming FastConformer: live captions of what the listener hears, on the CPU.
3. NVIDIA Magpie TTS (build.nvidia.com): "Say it": a synthetic voice speaks the suggested word only
   when the user taps it; it can be mixed into the call. The film plays the tapped word from
   `d.say` and shows `d.say.label` (Magpie with an NVIDIA key, otherwise a labelled placeholder).
4. NVIDIA GPUs with CUDA: the live WavLM detector runs on the laptop's GeForce RTX 4050; the models
   were trained on NVIDIA GPUs (Kaggle).
The narrator is a synthetic voice (`d.voice_label`, shown by the film).

**Measured** on 7 speakers the models never trained on (UCLASS): live Strong removes 65.2% of
stutters with 5.68% of words lost; live Gentle 33.1% with 3.1% lost; recordings (Thorough) 90.9% with
0% lost. Teacher v2 macro F1 0.618 on SEP-28k-E.

**The app** (real screenshots, 1680×1050, `staticFile("generated/screens/<name>.png")`): tabs Calls,
Recordings, Dictate, Practice, Journal (and On stage when an NVIDIA key is set).
- `calls.png`: "Say it your way. The listener hears you, without the repeats."; Start listening (or
  Space); Off | Gentle | Strong ("Removes clear repeats. About 0.5 s behind you."); Setup:
  Microphone, "Send the cleaned voice to", Detector Best (GPU) | AI model | Rule-based; Fluency
  Mirror; Context card; Word rescue · Nemotron ("Sends your last sentences and this card (never
  audio) to NVIDIA Nemotron on Nebius"). The context card's placeholder text ("My name is Sam…")
  must not be shown large: crop it out.
- `recordings.png`: Clean a voice note; the Monologue · man, 20 demo cleaned with Thorough (Original /
  Clean voice, Natural / Thorough / Live, Download clean, Timestamps, Subtitles; "Meaning guard ·
  Nemotron checked 23 words marked for cutting; every one was a real stutter."; struck-through
  transcript); Clean my video (Natural / Thorough, Choose a video).
- `dictate.png`: speak or upload, get the text without the stutters. Nothing is saved.
- `practice.png`: a practice interview about a role; it never comments on how you speak.
- `journal.png`: opt-in fluency journal: only numbers and words, never audio, only on this computer.

**Privacy.** Audio is processed on the user's computer; nothing is saved unless the user switches
saving on; only text goes to the cloud (Nemotron; Magpie gets the word to say).

## Honesty rules

- Alex is fictional; his voice is a real recording, credited whenever it plays (the film draws the
  credit). Never draw a face or a person: Alex is the Avatar, others are tiles with initials.
- Never imply a cure, a fix, or that stuttering is a flaw. StutterClear is optional, on the speaker's
  terms. No red or "error" styling on his real speech: every spoken word is shown the same.
- Nemotron gets text, never audio. Synthetic voices are labelled.
- Numbers only from the facts above or `d` (e.g. `d.numbers`), never invented. Pause cuts are "long
  pauses shortened", never counted as stutters. The live engine's scores are drawn as schematic (the
  real per-frame scores shown are the recording's, `d.clip.probs`).

## What the film draws (scenes must not)

- The backdrop (navy aurora, grain, vignette): scenes are transparent; never fill the background.
- Narration subtitles at the bottom centre (y ≈ 930–1040): keep everything important above y = 900.
- While the real clip plays (`before`, `after`): the credit card at top-left (x 56–820, y 48–175).
- The NVIDIA tracker at top-right (x ≈ 1150–1864, y 44–84) in scenes that light it, and the
  narrator note at top-right in the first scene: keep that strip clear.
- Narration, the clips, music (ducked under speech), transitions (20 frames overlap the start and
  the end of every scene) and a whoosh on each transition.

## One film, one look

- Type: Instrument Serif (`SERIF`) for story statements and big moments (64–150 px, tight leading,
  letter-spacing −1); Inter (`SANS`) for UI and explanation (26–48 px, weight 500–700); JetBrains Mono
  (`MONO`) for technical labels, model names, numbers on diagrams and wire formats (18–24 px). Noto
  Sans JP (`JP`) only for 東京 on the station sign. Eyebrow labels in MONO caps above titles.
- Colour meanings never change: sky/teal = Alex's voice; rose = what is cut; green = what is kept, a
  good result; violet = Nemotron, text going to the cloud; `C.nv` (#76b900) = an NVIDIA GPU or model;
  amber = struggle (a block, a stuck word). Text: `C.text`, `C.dim`, `C.faint` (nothing that must be
  read fainter than `C.dim`).
- Layout: 96 px safe margins (except deliberate bleeds); one clear focal point; generous space; a
  grid. Legibility: key words ≥ 56 px, body ≥ 26 px, labels ≥ 20 px, nothing below 18 px.
- Motion: every element enters with a spring or eased move within ~12 frames of its cue, siblings
  staggered 3–5 frames; exits ~12 frames (fade and lift). A slow camera on every scene (`Camera`: a
  gentle push-in, zoom 1 → ~1.05, or a move across a wide stage; drift 4–6). Light sweeps on key
  reveals; glows behind focal objects; a few particles (≤ 30) in the scene's colour; parallax.
- Diagrams: blocks joined by FlowPaths that draw themselves; packets show what travels (audio frames
  to the GPU model, words to Nemotron). Label every block: what it is, which model, where it runs.
- Numbers count up when they land, with their condition ("on 7 unseen speakers").
- Pacing: a new beat at least every ~2 s while narration runs; hold a clean frame ~1 s when a key
  message lands. Visuals land with the words that name them (± 6 frames).
- Sound effects, sparingly: snap = a stutter cut leaves, chime = the guard keeps a word or a result
  lands, pop = a suggestion appears. Never over the real clip or the Say it word.
- Performance: under ~2,500 elements per frame; deterministic (no `Math.random`, `Date`); avoid
  full-frame CSS blur except brief moments.

## Timing

`scene.frames` is the scene's length. `lineCue(scene, k)` gives `{at, frames}` (scene frames) for
narration line k; `clipCue(scene, "before" | "after" | "say")` for the audio, or undefined. Tie every
beat to the cue it illustrates (offsets inside a line are fine: e.g. a word ~40% into a 6 s line is
at `cue.at + 0.4 * cue.frames`); never hard-code seconds for something the narration names, because
the narration is re-generated with another voice later and every duration changes.

## The kit (`import { … } from "../kit"`; read `src/kit/*.tsx` for every prop)

Motion
- `springAt(f, at, config?)` → spring progress 0..1 from frame `at` (`SOFT`, `SNAPPY`, `BOUNCY`); pure.
- `ramp(f, a, b, easing?)` → 0..1 eased between frames (`ease`, `easeOut`, `easeIn`); `clamp`.
- `<Rise at out? y? blur?>`, `<Pop at out? from?>`, `<Beat at out? fade?>` (an absolute group that
  fades in and out).
- `<KineticText text at per? by="word"|"char" effect="rise"|"blur"|"drop"|"scale" highlight style>`.
- `<Typewriter text at cps? caret? hideCaretAfter? style>`, `<Counter to from? at dur? decimals?
  prefix? suffix? style>`.
- `<Camera shots={[{at, x?, y?, zoom?, rotate?, dur?}]} drift? seed?>` over a 1920×1080 stage: each
  shot eases the view so (x, y) is centred at `zoom`; `cameraAt(f, shots)`.

Atmosphere: `<Particles count color seed x y w h size rise opacity>`, `<GlowOrb x y r color opacity>`,
`<LightSweep at dur color angle>` (inside a relative, overflow-hidden parent).

Voice and model
- `<Wave env seconds width height color cuts cutTint collapse={[0..1 per cut]} progress dimAfter
  reveal barSeconds align glow>`: bars; cuts tint (`cutColor`) and shrink out with `collapse`.
  `waveX(t, seconds, width, cuts, collapse, align)` pins labels; `envAt(env, t0, t1?)` → loudness.
- `<Ribbon env seconds width height color color2 strands progress calm speed>`: a neon voice line.
- `<ProbLanes probs classes width laneHeight gap from to progress threshold emphasize labels
  labelWidth playhead>`, `<FrameCells probs from count cell gap perRow progress threshold>`.

System design
- `<Diagram>` (an SVG over the stage) with `<FlowPath d draw flow color width packets period
  kind="dot"|"frame"|"word" words dashed label labelAt labelOffset>`.
- `<Block x y w h title model where icon accent nv appear active>`, `<NvTag label?>`, `<Tag>`.
- `<Icon name size color stroke>`: mic, wave, chip, cloud, shield, speaker, captions, text, video,
  sparkles, scissors, user, lock, laptop, message, check, x, zap, headphones, clock, layers, arrow,
  heart, globe.

Chrome: `<Card glow? style>`, `<Pill>`, `<Eyebrow>`, `<Title size>`, `<Logo size>`, `<Avatar size
initial level struggle color>`, `<CallWindow title width height live>` with `<Tile name speaking>`,
`<ScreenFrame src width tiltX tiltY url>` (to show part of a screenshot, put an `<Img>` in an
overflow-hidden box and scale or translate it).

Sound: `<Sfx name="snap"|"chime"|"pop"|"whoosh" at volume>`.

Theme: `import { C, CLASS_COLORS, SANS, SERIF, MONO, JP, T } from "../theme"`. Timing: `import {
lineCue, clipCue, sec, FPS } from "../data"`. Remotion: `AbsoluteFill`, `useCurrentFrame`,
`interpolate`, `interpolateColors`, `spring`, `Easing`, `random` (deterministic), `Img`,
`staticFile`, `Sequence`; `@remotion/paths` (`evolvePath`, `getPointAtLength`, `getLength`);
`@remotion/noise` (`noise2D`). `src/scenes/kit.tsx` is a gallery that uses most of the kit.
