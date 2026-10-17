# Splice Real Speech First, Synthesize Only Blocks

StutterClear can feel fast, but not "ultra-low latency" in the telephone sense. The reason is that it rewrites speech rather than replying to it. A voice agent can start a reply 160–200 ms after the user stops ([Moshi](https://arxiv.org/abs/2410.00037)). StutterClear has to hear each word long enough to know it is not a stutter, and on speaker-independent data even large models detect blocks at only about **0.35 F1** ([Sušac et al. 2026](https://arxiv.org/abs/2606.20338)). No pipeline can deliver a fluent word before the user says it without guessing the word, and real-time prediction of intended words has never been validated. On the evidence in the notes, a cascade that transcribes, corrects and re-synthesizes speech adds roughly **0.55–0.75 s at best and 1.2–1.6 s typically after each phrase**. Hosted speech-to-speech APIs take **1.4–2.2 s** and come with preset voices. Calling US-hosted clouds from India adds another **0.35–0.5 s**. Only one route comes near the ITU-T G.114 "acceptable" band of 400 ms or less one-way: keep the user's real audio and cut stutter segments out of it in a short lookahead buffer. That costs an estimated **0.15–0.4 s** for partial trimming and **0.35–1.0 s** for full cleaning, with cloned TTS used only for blocked or badly distorted words. The design also keeps the user's voice and Indian-English accent by construction and reduces the need for voice cloning. Nobody occupies this space yet: no shipped product or paper streams stutter removal in the user's own voice, and the closest research system, STEAMROLLER, takes **3.8 s per segment** ([STEAMROLLER](https://arxiv.org/html/2601.10223v1)). Whatever the pipeline, corrected audio should go only to the far end by default. Hearing your own voice 50–200 ms late falls in the delayed-auditory-feedback band, which measurably changes how people speak. The research-risky features are telling blocks from pauses, correcting without changing meaning, and cloning that keeps the accent. Most of the rest is careful engineering.

## "Ultra-low latency" for a rewrite means hundreds of milliseconds, not tens

### Replies start in 200 ms; rewrites trail the speaker like an interpreter

The ~200 ms figures common in voice-AI marketing measure how fast a model starts a **reply**. Moshi, presented as the first real-time full-duplex speech model, reports **160 ms theoretical and 200 ms practical** response latency ([Moshi](https://arxiv.org/abs/2410.00037)). Qwen3-Omni claims a 234 ms theoretical first packet ([Qwen3-Omni](https://arxiv.org/abs/2509.17765)). A reply can open with any plausible words, because its content does not depend on what the user says next.

StutterClear does the opposite. It **rewrites** speech the user is still producing, so its latency is a lag behind the speaker, the measure used for simultaneous interpreters. The closest published template is Kyutai's Hibiki, a streaming speech-to-speech translator that keeps the speaker's voice. It trails the source by **2.3–2.9 s end offset and 3.4–5.0 s LAAL** ([Hibiki](https://arxiv.org/html/2502.03382v2)). Hosted realtime APIs were measured independently in July 2026 on a small 20-question sample. They took **1.4 s (Gemini 3.1 Live) to 2.2 s (GPT-Realtime-2)** from end of turn to first audio ([Plain Concepts](https://www.plainconcepts.com/?p=121434)).

Removing disfluency is easier than translating, because it is mostly in-order deletion. Google's streaming disfluency tagger waits for only **0.21 tokens** of following context on average, and at most about one token on its hardest cases ([Chen et al. 2022](https://arxiv.org/abs/2205.00620)). But streaming ASR takes 300–500 ms to emit each word, and TTS takes 200–350 ms to start. With those added, the research notes estimate a **1.0–2.0 s ear-voice lag** for any pipeline that re-synthesizes text. That figure is an estimate, because no system that re-voices stuttered speech in the same language has published a measurement.

### The stutter itself sets the minimum lookahead

Three facts set a floor that more GPU spending cannot lower.

First, a repetition cannot be labelled until its second iteration has been heard. Full removal therefore needs a lookahead of at least one repeated unit plus the detector's decision time. The notes budget **300–800 ms of confirmation latency per repetition**, which is an estimate rather than a measurement.

Second, nobody can reliably tell a silent block from a thinking pause. Block F1 is about **0.35 on the speaker-exclusive SEP-28k split**, even with 300M-parameter Whisper or WavLM encoders ([Sušac et al. 2026](https://arxiv.org/abs/2606.20338)). Annotators agree on blocks at only κ=0.25 ([Lea et al. 2021](https://arxiv.org/abs/2102.12394)), and detectors routinely classify blocks as pauses ([Ghosh et al. 2025](https://arxiv.org/abs/2508.02255)).

Third, predicting a stutter before it happens works poorly. The best continuous forecaster catches 43.9% of events with a mean **196 ms lead time, but raises 27.6 false warnings per minute**. Within a single speaker, its AUROC for blocks is about 0.54 ([Kumar, Daliri & Gupta 2026](https://arxiv.org/abs/2610.19235)).

What follows from these facts matters more than any of the numbers. **A block or prolongation uses up real time that no pipeline can give back, because the user has not yet said the next word.** Removal only changes what fills that time: a stutter sound or a clean silence. Only predicting the intended word could shorten it. The evidence for real-time completion of intended words is a 2015 n-gram system with 87% next-word accuracy in constrained tests ([NITK](https://idr.nitk.ac.in/handle/123456789/32330)) and a patent ([US 2019/0043490](https://patents.justia.com/patent/20190043490)). Neither has a peer-reviewed evaluation in open conversation. StutterClear should therefore define latency as **delay added on top of the user's own speaking time**, and report it separately for fluent stretches and for stutter events.

### Perceptual benchmarks: 150 ms is transparent, 200 ms is a normal gap

ITU-T G.114 treats one-way delay under **150 ms** as "essentially transparent", 150–400 ms as acceptable with increasing degradation, and over 400 ms as unacceptable for network planning ([ITU-T G.114](https://www.cs.columbia.edu/%7Eandreaf/new/documents/other/T-REC-G.114-200305.pdf)). Human turn-taking is tighter still. Across ten languages, the most common gap between turns is 0–200 ms and the mean is **+208 ms**, ranging from +7 ms in Japanese to +469 ms in Danish ([Stivers et al. 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2705608/)).

These numbers bound what the listener on the far end will feel. Adding about 0.3 s before the user's turns keeps gaps within the range people already accept, since Danish conversation averages nearly half a second. Adding 1.5 s does not, and the other person will start talking into the silence. That comes on top of whatever one-way delay the meeting app itself adds.

For people who stutter, the largest delay StutterClear controls is endpointing (deciding that the speaker has finished), not compute. Apple's general-population endpointer **cut off 23.8% of utterances** from people who stutter, against a 3% design target. Fixing it with longer silence thresholds cost **1.2–2.7 s of extra median delay** ([Lea et al. 2023](https://arxiv.org/abs/2302.09044)).

### Playing corrected speech into the user's ears is an uncontrolled DAF experiment

Any mode that plays the corrected voice into the user's own headphones lands in the delayed-auditory-feedback (DAF) band. The research points both ways:
- In lab studies, **50 ms DAF and other altered feedback cut stuttering frequency by roughly 60–75%** ([Stuart, Kalinowski & Rastatter 1997](https://pubs.asha.org/doi/full/10.1044/jslhr.4005.1130)).
- Fluent speakers are disrupted most near **200 ms**, where they produce two to three times more dysfluencies than at 0–50 ms ([Stuart et al. 2002](https://sonicfield.org/library/effect-of-delayed-auditory-feedback-on-normal-speakers-at-two-speech-rates)).
- Hearing-aid research finds that delays to one's own voice become disturbing above about **15–20 ms** ([Stone & Moore, via Canadian Audiologist](https://canadianaudiologist.ca/moore-feature-09/?output=pdf)).
- Hearing a fluent version in unison is stronger still. Choral speech using the speaker's own recording or TTS cut stuttering from about 10% to under 3% of syllables in a simulated presentation ([Demirel 2025, BNDU Oxford](https://www.bndu.ox.ac.uk/papers/novel-use-choral-speech-significantly-reduces-stuttering-simulated-presentation-setting)).
- The effects fade in daily life. A six-month naturalistic SpeechEasy trial found no significant change in stuttering frequency, despite positive self-reports ([Pollard et al. 2009, summarized by the Stuttering Foundation](https://stutteringhelp.org/node/551)).

StutterClear's output is fluent, in the user's own voice, and 0.1–2 s late. No study has tested that combination. It could help some users and disrupt others, and it would contaminate any fluency measurement. The safe defaults are these:
- send **corrected audio to the far end only**;
- make self-monitoring opt-in, with a delay the user chooses;
- allow raw sidetone (hearing your own unprocessed voice) only over wired connections, since Bluetooth alone adds 50–75 ms ([qb DAF listing](https://apps.apple.com/app/id1565802634)).

In person, the listener hears both the raw and the corrected voice. A consecutive mode that repeats each phrase after it ends fits that situation better. Google's Project Relate shipped this pattern as "Repeat" ([Tom's Guide](https://www.tomsguide.com/uk/news/watch-google-unveils-new-ai-app-to-help-people-with-speech-impairments)).

## Three candidate pipelines add between 0.15 and 2.1 seconds

The table compares three candidate pipelines stage by stage:
- **A** is a local-first cascade in phrase mode.
- **B** is the same cascade built on US-hosted cloud APIs and used from India.
- **C** is the recommended real-voice cut-and-splice path for calls.

Figures are milliseconds unless marked. Each figure carries an evidence tag:
- **[M]** is a measurement from a paper or third-party benchmark, including Pipecat's runs of other companies' models.
- **[V]** is a vendor claim.
- **[E]** is an estimate from the notes' arithmetic or mine.

Nothing in the totals row has been measured end to end, because no published system does this task.

| Stage | A. Local-first cascade (phrase mode) | B. Cloud cascade, India client to US-hosted APIs | C. Real-voice cut-and-splice (call mode) | Evidence basis |
|---|---|---|---|---|
| Mic capture buffer | 10–40 | 10–40 | 10–40 | [V] 10 ms WASAPI period ([JUCE forum](https://forum.juce.com/t/windows-audio-low-latency-mode/42085)); ~20 ms Android round trip ([Oboe guide](https://developer.android.com/games/sdk/oboe/low-latency-audio)); 40 ms macOS in Pipecat's budget ([Pipecat guide](https://voiceaiandvoiceagents.com/)) |
| Echo cancellation | 0–10 | 0–10 | 0–10 | [E] no primary figure for WebRTC AEC3 |
| Noise suppression | 10–40 | 10–40 | 10–40, on output and ASR paths only | [M] RNNoise 20 ms window, 10 ms hop ([RNNoise](https://ar5iv.labs.arxiv.org/html/1709.08243)); DeepFilterNet2 40 ms algorithmic ([DeepFilterNet2](https://ar5iv.labs.arxiv.org/html/2205.05474)); [V] Krisp 15 ms |
| VAD frame | 10–32 | 10–32 | 10–32 | [V] TEN VAD 10/16 ms hops ([TEN VAD](https://github.com/TEN-framework/ten-vad)); Silero handles a 30 ms chunk in under 1 ms ([Silero](https://github.com/snakers4/silero-vad)) |
| Stutter detection | Runs in parallel, gates the endpointer; ~0 on the critical path | Runs in parallel; ~0 | **On the critical path: 100–250 decision ("trim"); 300–800 repetition confirmation ("clean")** | [M] 1.34 ms per window including mel features for a 616K-parameter CNN ([Kozak 2026](https://arxiv.org/abs/2604.27279)); [E] decision and confirmation delays |
| End-of-phrase decision | 200–500 (1,670–3,100 if a silence timeout is tuned for people who stutter) | same | Off the output path; used only to hold the turn | [V] VAD 200 ms + Smart Turn 12–60 ms ([Smart Turn v3](https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/)); [M] Apple's medians for people who stutter ([Lea et al. 2023](https://arxiv.org/abs/2302.09044)) |
| Uplink | 0 | ~175–250 | 0 | [V] India-to-US round trip ~350 ms, Opus codec 21–26.5 ms, 40 ms jitter buffer ([Pipecat guide](https://voiceaiandvoiceagents.com/)); [E] split per direction |
| ASR to usable final | 50–300 | 20–140 after end of speech | Off the path (captions, dashboard, patch proposals) | [M] Nemotron 160–560 ms chunks ([Nemotron card](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)); Moonshine v2 148–258 ms on an M3 CPU ([Moonshine v2](https://arxiv.org/html/2602.12241v1)); cloud latency from end of speech ([Artificial Analysis](https://artificialanalysis.ai/articles/new-streaming-speech-to-text-benchmark-aa-wer-streaming)); live words 114–496 ms p50 ([Speko](https://benchmarks.speko.ai/)); [E] finalization window |
| Correction (tagger + LLM) | 50–150 best, 300–600 typical (tagger alone: milliseconds) | 536–739 median; 1,100–1,800 P95 | None on the path | [E] tagger and typical range; [V] 0.5B model on CPU 50–160 ms ([LiveKit](https://docs.livekit.io/agents/build/turns/turn-detector/)); 0.6 ms per prompt token for a 3B phone model ([Apple](https://machinelearning.apple.com/research/introducing-apple-foundation-models)); [M] hosted time to first token, run by Pipecat ([Pipecat guide](https://voiceaiandvoiceagents.com/)) |
| TTS first audio (cloned voice) | 100–350 | 75–270 P50, plus 50–200 connection setup | 0 (real audio); ~300–600 only when patching a word | [V/paper claim] Qwen3-TTS 97–101 ms ([Qwen3-TTS](https://arxiv.org/abs/2601.15621v1)); Kyutai 350 ms at 32 streams ([Kyutai TTS](https://kyutai.org/next/tts)); [M] Coval P50 ([Gradium/Coval](https://gradium.ai/content/tts-latency-benchmark-2026)); [V] ElevenLabs time to first byte from India 150–200 ms ([ElevenLabs](https://elevenlabs.io/docs/eleven-api/guides/how-to/best-practices/latency-optimization.md)); [E] patch path |
| Downlink | 0 | ~175–250 | 0 | as uplink |
| Playout + virtual mic | 10–110 | 10–110 | 10–110 | [V] playout 10–40 ms; BlackHole 0 ms ([BlackHole](https://github.com/ExistentialAudio/BlackHole)); VB-CABLE up to ~70 ms ([VB-Audio](https://forum.vb-audio.com/viewtopic.php?p=1594)) |
| **Added delay** | **0.55–0.75 s best and 1.2–1.6 s typical after the phrase ends; 1.0–2.0 s ear-voice lag in streaming mode** | **~1.5–2.1 s typical; ~3 s or more at P95** | **~0.15–0.4 s for trim; ~0.35–1.0 s for clean; drops after each cut** | [E] notes' arithmetic for A; my arithmetic from the rows above for B and C |
| Meeting app to remote ear | +100–400 | +100–400 | +100–400 | G.114 bands; outside StutterClear's control |

**Reading A (local-first cascade).** Pipecat publishes an end-to-end budget for a cloud cascade: 1,293 ms typical, and about 500 ms when all models run in one place ([Pipecat guide](https://voiceaiandvoiceagents.com/)). Those two figures bracket the notes' typical and best columns, which is a useful sanity check. For people who stutter, the largest term StutterClear can control is the end-of-phrase decision. An endpointer that understands stuttering keeps the turn open only while a block or repetition is detected. That holds the typical case near 0.3–0.5 s instead of Apple's 1.7–3.1 s silence-timeout medians. The second-largest term is LLM plus network. Small local models that output edits rather than rewritten text remove 0.5–1.5 s compared with a frontier API on another continent.

**Reading B (cloud from India).** India pays a structural penalty:
- Pipecat puts the round trip from India to US-hosted APIs at about **350 ms**, against about 250 ms from Europe ([Pipecat guide](https://voiceaiandvoiceagents.com/)).
- ElevenLabs publishes a **150–200 ms** time to first byte for South Asia, versus 100–150 ms for North America, Europe and South-East Asia. It serves from the USA, the Netherlands and Singapore, and offers India data residency only to enterprise customers ([ElevenLabs latency docs](https://elevenlabs.io/docs/eleven-api/guides/how-to/best-practices/latency-optimization.md)).
- Hosted-LLM P95 tails of 1.1–1.8 s ([Pipecat guide](https://voiceaiandvoiceagents.com/)) are what users feel as stalls.

Vendor figures need discounting. TTS vendors claim 25–100 ms, but those numbers usually cover model inference only and run 2–5x below independent end-to-end measurements. Coval itself excludes 50–200 ms of connection setup ([Gradium/Coval](https://gradium.ai/content/tts-latency-benchmark-2026)). The two independent ASR benchmarks also disagree sharply. ElevenLabs Scribe v2's time to first token is 0.13 s on Artificial Analysis but 2.02 s on Speko. Cartesia Ink-2's WER is 3.6% on one and 12.6% on the other ([Artificial Analysis](https://artificialanalysis.ai/articles/new-streaming-speech-to-text-benchmark-aa-wer-streaming); [Speko](https://benchmarks.speko.ai/)).

Stability matters as much as speed, because audio that has been spoken cannot be taken back. Speko measures how often partial transcripts get rewritten: **0% for Cartesia Ink-2, 13% for Soniox and 40% for Deepgram Flux** ([Speko](https://benchmarks.speko.ai/)). Every rewrite forces the corrector and TTS to redo work.

**Reading C (cut-and-splice).** Compute is negligible here. Kozak's on-device CNN uses 0.54% of a 250 ms budget per window ([Kozak 2026](https://arxiv.org/abs/2604.27279)). The minimum delay therefore comes from lookahead, not from hardware. Because the output is shorter than the input, every cut drains the buffer and pulls the lag back toward the I/O floor. These figures are the notes' inference. No one has measured a streaming splice system.

**Two architectures outside the table.** Hosted speech-to-speech models used as the rewrite engine would land around **1.4–2.5 s after the phrase ends**, before any voice-conversion stage. A self-hosted end-to-end rewrite model has a 160–200 ms frame-level floor but would realistically lag **1–3 s**, as Hibiki does ([Hibiki](https://arxiv.org/html/2502.03382v2)).

## Four architectures, ranked by what they do to the user's voice

### Option 1: a local-first cascade synthesizes the user's words again in a cloned voice

This is the textbook pipeline, run on the device or in the user's region. Its stages are:
- WebRTC AEC3-class echo cancellation and light noise suppression;
- a stutter detector that reads the raw mic signal;
- a streaming ASR that keeps disfluencies verbatim;
- a deletion-only disfluency tagger, followed by a small LLM that emits edits;
- a streaming TTS with a cloned voice.

For ASR, NVIDIA's Nemotron Speech Streaming 0.6B in int8 ONNX reaches **8.01% average WER at about seven times real time on CPU** ([Microsoft CoreAI evaluation](https://arxiv.org/html/2604.14493v1)). Moonshine v2 Small responds in **148 ms at 7.84% WER on an M3** ([Moonshine v2](https://arxiv.org/html/2602.12241v1)). For TTS, two Apache-2.0 models fit. Qwen3-TTS claims a 97 ms first packet and cloning from 3 s of audio ([Qwen3-TTS](https://arxiv.org/abs/2601.15621v1)). CosyVoice 3 claims about 150 ms with bidirectional streaming ([CosyVoice README](https://gitserver.onethingai.com/ai-models/CosyVoice2-0.5B)). A cheaper variant puts a per-user "voice skin" on a fast stock TTS. LLVC, an any-to-one converter trained for one target speaker, runs in under **20 ms on a consumer CPU** ([LLVC](https://arxiv.org/abs/2311.00873v1)).

This option can fix anything the ASR can transcribe, including misheard words and abandoned restarts. It keeps data on the device, and its latency does not depend on India's network. Its weaknesses are significant:
- typical delay is 1.2–1.6 s;
- the voice is synthetic: STEAMROLLER's zero-shot StyleTTS2 clone scored only **3.4/5 on speaker similarity**, and 7 of 23 users said it lacked warmth or did not sound like them ([STEAMROLLER](https://arxiv.org/html/2601.10223v1));
- every substitution risks changing the meaning;
- nobody has measured these models' speed on the laptops and phones StutterClear targets.

### Option 2: a cloud cascade is the fastest to prototype and the slowest to run from India

This option assembles best-of-breed APIs. The notes' ASR shortlist:
- Deepgram Flux, for the fastest partials: about 20 ms after end of speech on Artificial Analysis, and 190 ms p50 live words on Speko;
- ElevenLabs Scribe v2 Realtime, which is verbatim by default and gives per-word log-probabilities ([ElevenLabs API reference](https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime));
- Soniox, at about $0.12 per hour with few rewrites;
- Cartesia Ink-2, with no rewrites.

For TTS, Inworld TTS-2 Flash measured **75 ms P50** on Coval, and ElevenLabs Flash v2.5 measured 185 ms with a stock voice. ElevenLabs places instant clones in the same fast tier and professional clones in a slower one ([ElevenLabs latency docs](https://elevenlabs.io/docs/eleven-api/guides/how-to/best-practices/latency-optimization.md)). A hosted LLM does the correction.

Every vendor's endpointer must be overridden so that StutterClear's stutter detector owns the end-of-turn decision. On Flux that means `eot_threshold=1.0` with `ForceEndTurn` and a raised `eot_timeout_ms` ([Deepgram Flux configuration](https://developers.deepgram.com/docs/flux/configuration)). The equivalents are manual finalize on Soniox and manual commit on ElevenLabs.

Cost is not the constraint: streaming ASR costs **$0.12–1.02 per audio hour**. The constraints are India's round trip, P95 tails, and handing voice data to processors under India's Digital Personal Data Protection (DPDP) Act. Use this option as the harness that picks engines, not as the core of the product.

Hosted speech-to-speech models are a tempting shortcut for a "repeat what I said, fluently" demo, but they fall short in three ways:
- They take **1.4–2.2 s** ([Plain Concepts](https://www.plainconcepts.com/?p=121434)).
- They ship with preset voices. OpenAI's custom-voice program covers its Realtime API, but only for eligible customers via sales, and it is incompatible with Zero Data Retention ([OpenAI custom voices](https://developers.openai.com/api/docs/guides/custom-voices)).
- They are tuned to answer the user, not repeat them.

### Option 3: cut-and-splice keeps the real voice and pays only for lookahead

This option makes the user's real voice the default output and treats synthesis as the exception. It works like this:
1. The raw mic stream feeds a small causal stutter detector.
2. A delayed copy of the same audio flows toward the output through a short lookahead buffer.
3. When the detector confirms a segment to remove, that segment is cut and the joins are crossfaded before the audio reaches the speaker or virtual mic. Removable segments are a repeated unit, an over-long prolongation, a gasp or tense-voice segment, or block silence beyond a natural pause length.
4. Each cut shortens the output, which drains the buffer and pulls the lag back toward the I/O floor.
5. The system then rebuilds lookahead by stretching the next natural pause. This step is an untested design choice.

The correction-intensity control maps directly onto lookahead:
- **Trim** holds only the detector's decision delay and removes every iteration after the first, so "b-b-b-ball" becomes "b-ball". Estimated cost: **0.15–0.4 s**.
- **Clean** holds one repetition unit of lookahead, so the first iteration can be removed too. Estimated cost: **0.35–1.0 s**.
- **Repair** escalates to synthesis.

Each piece has a precedent, though nobody has combined them into a streaming product:
- Amazon's Detect-and-Pass flags stutter on 100 ms frames and drops those frames before an RNN-T recognizer, cutting WER by **12–71% relative** ([Shonibare et al. 2022](https://arxiv.org/abs/2202.05396)).
- Descript and Cleanvoice remove fillers and stutters from recordings offline, and Descript offers an "avoid harsh cuts" safeguard ([Descript](https://help.descript.com/hc/en-us/articles/10164806394509-Remove-filler-words); [Cleanvoice](https://cleanvoice.ai/stutter-remover/)).
- FluentSpeech drives offline diffusion editing from an automatic stutter localizer with 80.5% accuracy ([FluentSpeech](https://ar5iv.labs.arxiv.org/html/2305.13612)).
- Whispp claims under 120 ms end-to-end, on-device, for one-to-one voice reconstruction ([Whispp](https://www.whispp.com/)). That is a vendor claim, but it suggests acoustic mapping is fast when no lookahead is needed.

The advantages are large:
- Voice, accent, room acoustics and prosody survive exactly. This sidesteps both STEAMROLLER's 3.4/5 similarity problem and the risk of Indian-English accent drift.
- The default path creates no voice clone.
- The default path cannot change meaning by substitution, because it only deletes. That is the same deletion-only principle that makes LLM disfluency removal safe ([DRES](https://arxiv.org/abs/2509.20321)).

The weaknesses are just as real:
- It cannot fix misheard words or word choice.
- A false cut deletes a word the user meant. Apple's text-level refinement wrongly deleted intentional repeats such as "so so happy" ([Lea et al. 2023](https://arxiv.org/abs/2302.09044)).
- Nobody has evaluated how natural real-time joins sound.
- Prolongation and block detection are mediocre, so thresholds must be conservative and set per user.
- Listeners still hear silence through long blocks.

Repair escalation comes in two kinds, with very different risk:
- **Re-voicing** re-renders, in the cloned voice, a word the user *did* say that came out heavily distorted. The content comes from the user's own recognized word, so the risk to meaning is low. Estimated cost: 0.3–0.6 s after the word ends.
- **Prediction** synthesizes a blocked word the user has *not yet* said, which literally puts words in the user's mouth. It should be opt-in and limited to high-confidence, profile-backed cases or a confirm gesture. Otherwise the system should simply hold the turn and show a "speaking…" indicator.

### Option 4: an end-to-end streaming rewrite model is the research horizon

Hibiki offers a recipe that would carry over to stuttering:
- build parallel training pairs;
- derive per-word delays by "contextual alignment";
- constrain output to lag the source by at least 2 s;
- then, as Hibiki-Zero does, use GRPO reinforcement learning to push the lag down without word-level alignments ([Hibiki](https://arxiv.org/html/2502.03382v2); [Hibiki-Zero](https://arxiv.org/html/2602.11072)).

The distilled 1.7B Hibiki-M runs faster than real time on an iPhone 16 Pro and transfers the speaker's voice with a similarity of 0.41–0.48. The base architecture's floor is 160 ms theoretical and 200 ms practical on an L4 GPU ([Moshi GitHub](https://github.com/kyutai-labs/moshi)), but the notes put realistic rewrite lag at 1–3 s.

Three things block this option today:
- Hibiki and Hibiki-Zero weights are non-commercial, while Moshi's CC-BY 4.0 weights can be used commercially.
- No parallel corpus of disfluent and fluent speech in the same voice exists.
- Targets generated by TTS fail. The only end-to-end stutter-to-fluent models, StutterZero and StutterFormer, are offline, use a Griffin-Lim vocoder, and lose the speaker's identity because their fluent targets came from MeloTTS ([StutterZero/StutterFormer](https://arxiv.org/html/2510.18938v1)).

This is a track for v2 or v3, not a launch architecture.

### Recommended starting design

Build Option 3 as the core of call mode. Run Option 1's text path alongside it to drive captions, the stutter-aware endpointer, the dashboard and repair escalations. Ship Option 1's full re-voicing as a consecutive "phrase" mode for in-person use. Keep Option 2 as a test harness, and host in-region if any cloud component reaches production.

The components:
- **Detector:** distill a Whisper-encoder attention-MIL teacher into a causal student. The teacher reaches frame F1 0.70 on the CASA gold set ([Sušac et al. 2026](https://arxiv.org/abs/2606.20338)). The student should be close to the 616K-parameter size of Kozak's on-device model. Calibrate thresholds per user, and feed the detector raw mic audio, never audio that has passed through denoising or VAD.
- **ASR:** a streaming model that leans verbatim, with keyterm biasing from the user's profile and per-user LoRA adaptation.
- **Correction:** a streaming tagger with a learned wait head (BERT_SV-class, 3.1M parameters, streaming F1 0.83 ([Chen et al. 2022](https://arxiv.org/abs/2205.00620))), in front of a 0.5–3B instruct model. That model may only output keep/delete masks, choices among the recognizer's top hypotheses, or substitutions drawn from the profile.
- **Orchestration:** Pipecat-style two-lane frame queues and generation IDs, so a restart flushes stale audio immediately ([Pipecat interruptions](https://docs.pipecat.ai/pipecat/fundamentals/interruptions)).

### Where the literature leaves room for something new

Five gaps in the literature are open for StutterClear to fill:
1. **The real-voice streaming splice with an intensity dial.** No paper or product describes one.
2. **A learned endpointer for people who stutter.** The notes found nothing beyond Apple's threshold tuning. Smart Turn v3 ships open weights, data and training code at 8 MB and about 12 ms on CPU ([Smart Turn v3](https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/)). Fine-tuning it on stuttered audio with labelled blocks is therefore cheap, as is fusing it with the detector's block probability and LiveKit-style per-user pause statistics ([LiveKit turn detector](https://docs.livekit.io/agents/build/turns/turn-detector/)).
3. **LocalAgreement-style commit policies applied to correction output** rather than ASR output. The notes found this unstudied.
4. **The splice system's own consented output as training targets for Option 4.** Real-voice fluent audio cleaned by cutting would supply the natural targets, in the same voice and accent, that StutterZero lacked.
5. **Clone prompts built only from fluent segments the detector has certified.** Zero-shot cloners copy accent along with timbre ([UT Dallas](https://arxiv.org/pdf/2603.05977)), so they plausibly copy stutter artifacts too.

The last two are hypotheses, but cheap ones to test.

## Blocks, meaning and accent carry the research risk; most other features are plumbing

The table maps each Phase 1 feature to its best-supported technique and flags its risk:
- **Engineering:** known techniques with published numbers.
- **Moderate:** techniques exist but have not been measured on stuttered or Indian-English speech.
- **Research-risky:** the literature has no validated solution.

| # | Feature | What the evidence supports | Risk |
|---|---|---|---|
| 1 | Real-time mic input | 10 ms WASAPI periods ([JUCE forum](https://forum.juce.com/t/windows-audio-low-latency-mode/42085)). ~20 ms Android round trip with Oboe best practice, versus ~205 ms with low-latency mode off ([Oboe guide](https://developer.android.com/games/sdk/oboe/low-latency-audio)). Call mode needs a virtual mic: BlackHole adds 0 ms but is GPL-3.0 ([BlackHole](https://github.com/ExistentialAudio/BlackHole)). | Engineering |
| 2 | Stutter detection | Transcript-free frame-level models lead: attention-MIL scores frame F1 0.70 on CASA ([Sušac et al. 2026](https://arxiv.org/abs/2606.20338)); StutterCut scores time-F1 69.3 on FluencyBank++ ([Ghosh et al. 2025](https://arxiv.org/abs/2508.02255)). Clip-level F1 on speaker-exclusive data: interjection 0.82, word repetition 0.74–0.80, sound repetition 0.53, prolongation 0.47–0.50, block 0.35. The best models run offline and have 300M+ parameters. | Research-risky for blocks and prolongations; moderate for repetitions |
| 3 | Streaming ASR | Autoregressive models drift toward intended text; CTC models stay verbatim ([Toyin et al. 2026](https://arxiv.org/abs/2606.31112)). Whisper hallucinated on 22.6% (v2) and 6.9% (v3) of clips with sound repetitions ([Sridhar & Wu 2025](https://www.isca-archive.org/interspeech_2025/sridhar25_interspeech.pdf)). Per-user LoRA cut Whisper-small WER from 16–18% to 9.4–9.8% ([Mujtaba & Mahapatra 2025](https://arxiv.org/abs/2506.00853)). No 2026 streaming engine has been benchmarked on stuttered speech. | Moderate |
| 4 | Context-aware correction | Unconstrained LLM correction nearly doubled TORGO WER, from 10.83% to 20.01%; confidence prompting held it at 10.58% ([Hernandez et al. 2025](https://arxiv.org/abs/2509.25048)). Reasoning models over-delete ([DRES](https://arxiv.org/abs/2509.20321)). No benchmark measures meaning preservation on stuttered transcripts. | Research-risky |
| 5 | Personalized profile | Keyterm and context biasing exist in Soniox, ElevenLabs, Deepgram and AssemblyAI ([Soniox docs](https://soniox.com/docs/stt/rt/real-time-transcription)). CLAS reported up to ~68% relative WER gain over shallow fusion ([Pundak et al. 2018](https://arxiv.org/abs/1808.02480)). Use the profile as a closed candidate set for substitutions. A per-user memory file of the Voice Memory kind is auditable ([Voice Memory](https://arxiv.org/abs/2607.26410)). | Engineering |
| 6 | Conversation history | Context helps ranking, but history that contains errors propagates them ([contextual exposure bias](https://arxiv.org/abs/2603.24034)). Admit only committed turns the user has confirmed. | Moderate |
| 7 | Natural TTS | Coval P50: Inworld TTS-2 Flash 75 ms, ElevenLabs Flash v2.5 185 ms, Cartesia Sonic 3.5 269 ms ([Gradium/Coval](https://gradium.ai/content/tts-latency-benchmark-2026)). Send phrase-sized text chunks so the buffer does not stall ([ElevenLabs](https://elevenlabs.io/docs/eleven-api/guides/how-to/best-practices/latency-optimization.md)). | Engineering |
| 8 | Voice personalization with consent | Instant clones from 3–30 s of audio are standard. ElevenLabs professional clones are slower than instant clones on Flash. STEAMROLLER's clone scored 3.4/5 on similarity. No independent benchmark of Indian-English accent drift exists. Splicing keeps the accent by construction. | Moderate for cloning; research-risky for accent fidelity |
| 9 | Low-latency streaming | Pass raw PCM between local stages. Each cloud hop costs Opus 21–26.5 ms + a 40 ms jitter buffer + round-trip time ([Pipecat guide](https://voiceaiandvoiceagents.com/)). Prefer WebRTC to WebSocket on client devices ([OpenAI Realtime guide](https://developers.openai.com/api/docs/guides/realtime-conversations)). | Engineering |
| 10 | Pause vs block handling | Stock endpointers cut off 23.8% of utterances from people who stutter; timeouts long enough to tolerate blocks add 1.2–2.7 s ([Lea et al. 2023](https://arxiv.org/abs/2302.09044)). No validated acoustic classifier separates blocks from pauses. | Research-risky |
| 11 | Confidence-based correction | Gating on word-level confidence kept harmful edits on high-confidence utterances to 1.1–5.0% ([Hernandez et al. 2025](https://arxiv.org/abs/2509.25048)). Abstaining per utterance left no dataset worse than the recognizer's top hypothesis ([Voice Memory](https://arxiv.org/abs/2607.26410)). The tagger's wait head lets it abstain per token. | Moderate |
| 12 | Interruption handling, no stale audio | Cancel the LLM and TTS, drain unplayed audio, and commit only text that was actually played ([Pipecat interruptions](https://docs.pipecat.ai/pipecat/fundamentals/interruptions)). The difference here: the trigger is the same speaker restarting, not "user started speaking". | Engineering; the stutter-aware trigger is moderate |
| 13 | Noise handling | RNNoise uses ~1.3% of one x86 core ([RNNoise](https://ar5iv.labs.arxiv.org/html/1709.08243)). But a chain of denoising, VAD and Opus cut block F1 from 0.638 to 0.465 ([Pattanayak et al. 2026](https://arxiv.org/abs/2609.32285)). | Engineering, with a known trap |
| 14 | User controls | Apple recommends opt-in endpointer profiles and opt-in disfluency removal. Flux and AssemblyAI accept configuration changes mid-stream ([Deepgram Flux](https://developers.deepgram.com/docs/flux/configuration)). Correction intensity maps onto lookahead. | Engineering plus UX research |
| 15 | Privacy and data control | Spoken, scripted consent plus speaker verification ([OpenAI custom voices](https://developers.openai.com/api/docs/guides/custom-voices)). Voice profiles stay on the device. DPDP requires erasure when consent is withdrawn ([DPDP Rule 8](https://dpdpa.com/dpdparules/rule8.html)). | Engineering plus legal |
| 16 | Evaluation dashboard | No prior stutter-repair paper reports objective speaker similarity, UTMOS or mouth-to-ear latency ([STEAMROLLER](https://arxiv.org/html/2601.10223v1)). | Engineering; the metrics themselves are new |

### Telling a block from a pause

This is the hardest unsolved piece, and it affects latency, correctness and dignity at once. What the evidence offers:
- Practitioners who stutter rely on strain, breathing and tempo changes: "a small pause where someone's voice is strained is a block, but a long pause where someone is thinking and their voice sounds fine is fluent speech" ([Sridhar & Wu 2025](https://www.isca-archive.org/interspeech_2025/sridhar25_interspeech.pdf)).
- Kozak cites clinical literature that tension in the larynx builds hundreds of milliseconds before the audible silence. Masking the last second of input lowered his block AUC by 0.021 ([Kozak 2026](https://arxiv.org/abs/2604.27279)).
- Yet no peer-reviewed classifier separates silent blocks from fluent pauses acoustically. The notes infer that off-the-shelf turn detectors, trained on general-population speech, will often decide the speaker has finished when they are in a block.

The practical design treats end of phrase as a fused decision combining five signals:
- VAD silence;
- a Smart Turn-style completeness score fine-tuned on stuttered audio;
- the detector's "block in progress" probability;
- per-user pause statistics;
- a push-to-talk fallback, which Apple recommends ([Lea et al. 2023](https://arxiv.org/abs/2302.09044)).

Front-end processing is the trap. In a simulated chain, WebRTC VAD alone removed **78% of clean silence** and cost 12 points of block F1. When the detector was tested on podcast shows held out from training, block recall fell from 0.718 to 0.383 ([Pattanayak et al. 2026](https://arxiv.org/abs/2609.32285)). The detector must tap the mic before any processing.

### ASR on stuttered speech

Every engine on the shortlist is unproven on stuttered speech. All the API numbers for stuttered speech date from March 2024, when mean WER was **28.1% on stuttered speech versus 10.5% on fluent speech** ([Mujtaba et al. 2024](https://arxiv.org/abs/2405.06150)). Severe speakers saw **49.2% WER** on consumer ASR ([Lea et al. 2023](https://arxiv.org/abs/2302.09044)). Whisper hallucinates on long pauses and has produced degrading onomatopoeia such as "oink oink oink" on sound repetitions ([Sridhar & Wu 2025](https://www.isca-archive.org/interspeech_2025/sridhar25_interspeech.pdf)). That rules it out as the primary engine for a system whose users pause by definition.

The notes recommend running two transcript streams:
- a verbatim stream, which aligns with detector events and drives deletion;
- an intended stream, which feeds correction and TTS.

This also matches stated preferences: people who stutter prefer intended transcripts for communication, and speech-language pathologists prefer verbatim ones ([Toyin et al. 2026 scoping review](https://arxiv.org/abs/2604.20535)).

### Correction that preserves meaning

The notes split correction into three operations with very different risk:
- **Deletion** of fragments, repetitions and abandoned restarts cannot hallucinate if the output is a mask.
- **Substitution** is allowed only for low-confidence words, and only toward the recognizer's alternative hypotheses (N-best) or profile candidates.
- **Insertion** of a blocked word is a proposal that needs high confidence or user confirmation.

Unconstrained generative correction breaks correct tokens in up to **64% of its edits** in one domain. A frozen corrector that can abstain per utterance cut that to 35% ([Voice Memory](https://arxiv.org/abs/2607.26410)). DRES recommends small models that tend to under-delete for use on the device, and warns against reasoning models. It also finds that chunking context to about four sentences stabilizes behaviour ([DRES](https://arxiv.org/abs/2509.20321)).

### Voice and accent for Indian users

Zero-shot cloners "often inherit both the accent and timbre" of the reference ([UT Dallas](https://arxiv.org/pdf/2603.05977)). Several vendors address accent explicitly, though nobody has independently evaluated their claims:
- Smallest AI documents that a South Indian English clone keeps its accent by design ([Smallest AI](https://docs.smallest.ai/models/documentation/best-practices/voice-cloning-best-practices.md)).
- Azure Personal Voice "matches your natural speaking style, intonation, and accent" ([Microsoft Learn](https://learn.microsoft.com/azure/ai-services/speech-service/personal-voice-overview)).
- Google's Chirp 3 Instant Custom Voice supports en-IN but is allow-list only ([Google Cloud](https://docs.cloud.google.com/text-to-speech/docs/chirp3-instant-custom-voice)).
- The Indic open models Dhee-Indic-F5, Svara-TTS and Gnani's Vachana list Indian English ([Dhee-Indic-F5](https://huggingface.co/dheeyantra/dhee-indic-f5); [Svara-TTS](https://huggingface.co/kenpath/svara-tts-voiceclone-beta); [Gnani Vachana](https://www.businesstoday.in/amp/technology/story/gnaniai-launches-zero-shot-voice-cloning-tts-model-for-12-indic-languages-516896-2026-02-19)).

In a FAccT 2025 study, people hearing their own clones felt their accent had been "put through a filter" ([FAccT 2025 voice-clone study](https://arxiv.org/html/2504.09346v1)). Data is thin too. Boli, the only Indian stuttering corpus, has **28 speakers who actually stuttered**, about 2.5–2.8 hours across six languages, and only 70 labelled blocks ([Batra et al. 2025](https://arxiv.org/abs/2501.15877)). Together, these points are the strongest single argument for the splice-first design for Indian users.

### Traps in the engineering features

The features marked "engineering" still hide traps:
- **Echo leakage.** In call mode StutterClear captures the raw mic before Zoom's own echo cancellation. Remote voices from laptop speakers will leak into ASR unless StutterClear runs its own AEC with a loopback reference. Headphones should be the default.
- **Restarts versus repetitions.** "I-I-I" must not count as a restart, so the interrupt trigger needs a minimum-words or stutter-aware filter ([Pipecat interruptions](https://docs.pipecat.ai/pipecat/fundamentals/interruptions)).
- **Blocks misread as stops.** When that happens, playback must resume, which LiveKit ships as `resume_false_interruption` ([LiveKit turns](https://docs.livekit.io/agents/build/turns/)).

## Consent, community trust and a dual-reference dashboard

### The community wants accommodation, not a cure

The organized stuttering community increasingly treats stuttering as a valid way of speaking. Tichenor, Constantino and Yaruss argue that the binary of fluent versus stuttered speech itself fuels stigma ([Tichenor et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9135003)). The FAccT 2026 paper "Cripping AI" asks speech systems to "shift goal from promoting fluency toward supporting the speaker's comfort and dignity". It notes that many people who stutter want verbatim captioning as an option, and quotes Constantino: "our pauses, hesitations, and silences carry semantic weight" ([Tang et al. 2026](https://arxiv.org/abs/2605.02080)). Marketing a fix provokes organized backlash. Samsung's Impulse ad drew an open letter with **99 organizational and 828 individual signatories**, and Samsung re-edited the ad ([SPACE](https://www.spacetostutter.org/impulse)).

Yet people who stutter clearly want technology that works for them. In Apple's survey, **41% said voice assistants cut them off always or often** ([Lea et al. 2023](https://arxiv.org/abs/2302.09044)). STEAMROLLER's 23 users who stutter rated overall satisfaction **4.30/5** and meaning preservation 4.13, but usability (SUS) only 66.9. They valued seeing the original and repaired output side by side ([STEAMROLLER](https://arxiv.org/html/2601.10223v1)). Constantino and colleagues describe "passing as fluent" through communication technology as active resistance rather than repression ([Constantino et al. 2017](https://digitalcommons.memphis.edu/facpubs/10375)). That supports positioning StutterClear as an accommodation the speaker chooses.

The product consequences:
- Never auto-enable.
- Offer a per-call toggle and an intensity dial that includes verbatim passthrough.
- Provide side-by-side review.
- Never market StutterClear as a cure.
- Include people who stutter in design and evaluation, and pay them for the emotional labour involved. An annotator who stutters relabelled **more than 25%** of sampled SEP-28k clips ([Sridhar & Wu 2025](https://www.isca-archive.org/interspeech_2025/sridhar25_interspeech.pdf)).

Whispp is a warning about positioning. Its 2024 materials described a reimbursed consumer app; its 2026 site defers stuttering use until "medical validation" ([Whispp](https://www.whispp.com/); [GSMA case study](https://gsma.com/get-involved/gsma-foundry/wp-content/uploads/2024/09/Giving-More-People-a-Voice.pdf)).

### Consent, deletion and law: copy the strictest vendor practice

The 2026 industry norm for cloning a real voice is a spoken, scripted consent statement by the same speaker, checked against the voice sample:
- OpenAI offers its consent phrase in 16 languages, including Hindi, and restricts custom voices to adults ([OpenAI custom voices](https://developers.openai.com/api/docs/guides/custom-voices)).
- Google requires a fixed script recorded in the same environment as the sample ([Google Cloud](https://docs.cloud.google.com/text-to-speech/docs/chirp3-instant-custom-voice)).
- Azure verifies that the consenting speaker matches the voice audio ([Microsoft Learn](https://learn.microsoft.com/azure/ai-services/speech-service/personal-voice-overview)).
- Apple trains Personal Voice on the device, and iOS 26 needs only 10 phrases and under a minute ([9to5Mac](https://9to5mac.com/2026/03/19/ios-26-made-one-of-iphones-wildest-most-unique-features-a-lot-better/)).

StutterClear should:
- enroll only the user's own voice, with a randomized liveness phrase;
- store an embedding or adapter, not raw audio, in the platform keystore;
- offer one-tap deletion that propagates to any vendor-side clone;
- keep a hashed consent log;
- let the voice drive only the user's own live speech, never exported TTS.

The last point matters because Tennessee's ELVIS Act reaches distributors of tools whose primary purpose is replicating a voice without authorization ([Phelps Dunbar](https://phelps.com/insights/the-elvis-act-protects-artists-voices-but-creates-risks-for-media-companies.html)). Any synthetic patch should carry an AudioSeal or PerTh watermark. Treat that as compliance and forensics rather than security, because neural-codec and voice-conversion attacks strip watermarks ([SoK on audio watermarking](https://arxiv.org/html/2503.19176v2)).

The splice-first design brings a regulatory side benefit, which is my inference. Its default path processes the voice but synthesizes nothing, which narrows both what needs consent and what needs disclosure. That matters because listeners judge a clone to be the real person about 80% of the time ([Sci. Reports 2025](https://www.nature.com/articles/s41598-025-94170-3)).

Three legal regimes apply:
- **EU.** The AI Act's Article 50 requires machine-readable marking of synthetic audio from **2 Aug 2026** ([Cooley](https://www.cooley.com/news/insight/2026/2026-08-03-eu-ai-act-transparency-obligations-take-effect-2-august-2026)). Sources disagree on whether a 2 Dec 2026 transition covers systems already on the market.
- **US.** The federal NO FAKES Act was still not law in October 2026 ([Daily Journal](https://www.dailyjournal.com/article/395055-no-fakes-act-dead-for-now-as-cruz-objects-to-unanimous-consent-request)).
- **India.** The DPDP Rules were notified in November 2025 and phase in ([Hogan Lovells](https://www.hoganlovells.com/en/publications/indias-digital-personal-data-protection-act-2023-brought-into-force-)). Consent Manager registration follows after about 12 months, around November 2026. Most obligations apply from about **13 May 2027**: notice, security safeguards, breach reporting, Rule 8 erasure and data-principal rights. Rule 8 requires erasure, including by processors, when consent is withdrawn or the purpose ends ([DPDP Rule 8](https://dpdpa.com/dpdparules/rule8.html)). The notes did not verify whether DPDP treats voiceprints specially.

Vendor terms interact with these laws. OpenAI's custom voices are incompatible with Zero Data Retention, and ElevenLabs offers India data residency only at enterprise tier ([ElevenLabs](https://elevenlabs.io/docs/eleven-api/guides/how-to/best-practices/latency-optimization.md)).

Licensing traps run through the stack:
- SEP-28k is CC BY-NC 4.0, and the podcast owners keep copyright in the audio ([Apple dataset repo](https://github.com/apple/ml-stuttering-events-dataset)). Whether a commercial detector may be trained on it is legally unclear.
- Hibiki and Hibiki-Zero weights are non-commercial.
- Among TTS models, F5-TTS weights, XTTS-v2 and Fish S2 are non-commercial or need paid licences.
- Seed-VC is GPL-3.0, and its repository was archived in November 2025 ([Seed-VC](https://github.com/Plachtaa/seed-vc)).
- BlackHole is GPL-3.0.
- Kyutai has not released the voice-embedding model its TTS needs to clone arbitrary voices ([Kyutai Unmute](https://kyutai.org/2025/05/22/unmute.html)).

### The dashboard should measure what no prior system has reported

Prior work evaluates fragments of the chain, never the whole. STEAMROLLER's metric set is the broadest. No stutter-repair paper reports objective speaker-embedding similarity, UTMOS (an automatic naturalness score), time to first audio, or NASA-TLX workload. Two references run through the table:
- **Intended reference:** what the user meant to say, with stutters removed.
- **Verbatim reference:** exactly what was said, stutters included.

| Dashboard panel | Metrics | Why and precedent |
|---|---|---|
| Transcription accuracy | WER against the intended reference (primary) and the verbatim reference (diagnostic), averaged **per speaker**; hallucination rate during blocks and sound repetitions; cut-off rate; share of users under 10% WER | Model rankings flip between the two references ([Toyin et al. 2026](https://arxiv.org/abs/2606.31112)). Per-speaker averaging exposes the severe tail: Boli's WER is 4.55% pooled versus 21.27% averaged per speaker ([Batra et al. 2025](https://arxiv.org/abs/2501.15877)). Apple used thresholded WER and truncation rate ([Lea et al. 2023](https://arxiv.org/abs/2302.09044)) |
| Detection | Frame-level F1 and time-F1 at IoU>0.5 per class; false cuts per minute; confusion between blocks and pauses | StutterCut and MIL ([Ghosh et al. 2025](https://arxiv.org/abs/2508.02255)) |
| Correction quality | Deletion precision, recall and F1 (E-scores), and Z-scores per disfluency category; over-correction rate split by deletion, substitution and insertion; help and harm by confidence band; how error falls as the corrector abstains more; meaning preservation rated by the speaker plus an NLI or LLM judge, with BERTScore secondary | ([Z-scores](https://arxiv.org/abs/2509.20319)); ([Hernandez et al. 2025](https://arxiv.org/abs/2509.25048)). DRES warns that metrics tolerant of paraphrase misjudge deletion-only fidelity ([DRES](https://arxiv.org/abs/2509.20321)) |
| End-to-end latency | Per-stage timestamps; end of phrase to first audio; continuous ear-voice lag (LAAL-style); P50 and P95; edits that reach the TTS after audio is already spoken (target 0) | Streaming Parrotron reported 320 ms "total delay" after end of speech on a Pixel 4 ([Streaming Parrotron](https://ar5iv.labs.arxiv.org/html/2210.13761)); Hibiki reports LAAL; the edit-overhead metric comes from [Chen et al. 2022](https://arxiv.org/abs/2205.00620) |
| Voice | UTMOS or naturalness MOS; ECAPA or WavLM speaker cosine; similarity rated by the speaker; accent-ID consistency for Indian English; ASR WER of the output | FluentSpeech, STEAMROLLER |
| User satisfaction | SUS, NPS, NASA-TLX; OASES (a stuttering-impact questionnaire) before and after over weeks; far-end-only versus self-monitoring arms; far-end listener ratings | ([OASES](https://www.pearsonclinical.in/solutions/overall-assessment-of-the-speakers-experience-of-stuttering-oases)). Pre-register both objective and self-report outcomes, because SpeechEasy users reported benefit without measurable change ([Stuttering Foundation](https://stutteringhelp.org/node/551)) |

## Twelve unknowns only our own measurements can settle

The notes kept hitting the same wall: nobody has measured this exact task end to end, so most of StutterClear's design rests on estimates. The unknowns below would change the architecture. For each, the first experiment is chosen to answer it early.

| Unknown | Why it matters | First measurement |
|---|---|---|
| Which streaming ASR handles stuttered Indian English, including Hindi-English code-switching? | No 2026 engine has been benchmarked on stuttered speech; API evidence dates from 2024 | Build a consented Indian-English dataset of people who stutter, with both verbatim and intended references, and add Boli. Score the shortlist on both references, partial-transcript rewrites, and hallucinations during blocks |
| How does a causal on-device detector's accuracy trade off against lookahead? | It sets the trim and clean latencies in Option 3; no causal frame-level detector has been evaluated on CASA or FluencyBank++ | Distill an MIL teacher into a causal student and plot time-F1 against lookahead from 100 to 800 ms |
| Can blocks be told from pauses in real time? | Drives the endpointer, block cutting, and whether the remote party talks over the user | Fuse strain before the silence, F0 cut-off, syntactic position and whether the language model sees the sentence as unfinished. Measure false end-of-turn during labelled blocks |
| How natural do real-time splices sound, and how often do false cuts delete intended words? | No literature covers streaming splicing | Listener MOS on spliced versus raw audio; false cuts per minute; speaker rating of "said what I meant" |
| How fast is cloned-voice TTS to first audio, and how faithfully does it keep an Indian accent? | Published first-audio tables use stock voices; no accent-drift benchmark exists | A bake-off on identical hardware with phrase-sized text chunks; accent-ID plus speaker cosine plus speaker ratings |
| How fast are local LLMs to first token on short prompts, on target laptops and phones? | Decides whether the repair path stays local | llama-bench prompt-processing speed for 0.5–3B instruct models |
| How does hearing one's own corrected voice affect fluency? | DAF and choral effects; no study exists | Crossover study: far-end only versus self-monitoring at chosen delays, measuring percent syllables stuttered (%SS) and comfort ratings |
| Do far-end listeners accept the lag and the re-voiced speech? | Normal turn gaps are about 200 ms; listeners spot AI voices only about 60% of the time | Listener study at lags of 0.3, 1.0 and 1.5 s, with and without disclosure |
| What do Zoom, Meet and Teams do to virtual-mic input? | Their noise suppression may smear splices or synthetic speech | Loopback recordings through each app's processing modes |
| Can blocked words be predicted well enough to speak them? | Prediction is the only way to shorten a block, and no validated system exists | Offline replay first: top-1 accuracy against abstention rate, conditioned on profile and history, before any live use |
| How much calibration data does each user need? | ASR LoRA gains saturate after 25–50% of 60 prompts ([Mujtaba & Mahapatra 2025](https://arxiv.org/abs/2506.00853)); curves for the detector and endpointer are unknown | Learning curves for threshold calibration from enrollment and early-use data |
| How do DPDP and medical-device rules classify the product? | How the law treats voiceprints is unverified; Whispp deferred its stuttering claims | Legal review before any health claim or cloud voice storage |

## Conclusion

StutterClear's latency question turns out to be less about speed than about what fills the time a stutter takes. Every architecture inherits that time from the speaker. A cascade adds recognition, rewriting and synthesis on top of it. A splice adds only the lookahead needed to decide what to drop. That reverses the usual assumption that personalization costs latency: the user's own unprocessed voice is the fastest, most accent-faithful and lowest-risk output available. Synthesis earns its latency only where the real audio cannot be saved.

The hardest problems are therefore not about compute. They are perceptual and social. Do real-time cuts sound natural? Can an endpointer respect a block without stalling the conversation? Will listeners and users accept a voice running a few hundred milliseconds behind? Will people who stutter experience the tool as accommodation rather than erasure?

The build order follows from this. First, build the measurement rig and a consented Indian-English dataset with both references, because no published number answers the questions that decide the architecture. Next, build the stutter-aware endpointer and the "trim" splice, the two pieces with the clearest path to under 400 ms and no cloning exposure. Cloned-voice repair and block-word prediction come after that, both opt-in and gated on confidence. The end-to-end rewrite model is a later bet, and its most valuable input may be the clean speech, in the user's own voice, that the splice system itself produces.
