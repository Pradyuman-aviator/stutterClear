# Streaming TTS, Zero-Shot Voice Cloning, Real-Time Voice Conversion, and Consent/Safety for StutterClear (research date: 2026-10-17)

Notes for the report writer:
- All numbers carry a source. "Vendor claim" means the company that sells the product published it. "Independent" means a third-party benchmark. Coval (benchmarks.coval.ai/tts) is the main independent TTS latency board. I could not load its live page, so the Coval figures below come from vendor blogs that quote Coval snapshots. Treat them as snapshots, not the current board.
- Coval changed its metric on 3 June 2026. Since then it reports "perceived TTFA" (round trip plus any leading silence in the stream). Figures from before that date are round-trip only and cannot be compared with later ones ([Gradium summary of Coval](https://gradium.ai/content/tts-latency-benchmark-2026)).
- The Artificial Analysis (AA) TTS arena ranks voice quality by Elo from blind votes. Its leaderboard page does not measure latency ([AA leaderboard](https://artificialanalysis.ai/text-to-speech/leaderboard)).

---

## Q1. Cloud streaming TTS: TTFB/TTFA, streaming text input, instant voice cloning, price

### Takeaway
The independently measured fastest cloud TTS in Sept 2026 are Inworld TTS-2 Flash (about 75 ms P50 perceived TTFA on Coval), ElevenLabs Flash v2.5 (about 185 ms) and Inworld TTS-2 (about 170 ms). Newer quality leaders sit in the 200–450 ms range: ElevenLabs Eleven v4 Turbo (Coval mean about 204 ms) and Cartesia Sonic 3.6 (about 440 ms P50, with high variance). Vendor latency claims (25–100 ms) are usually model-inference-only and run 2–5x below independent end-to-end figures. ElevenLabs, Cartesia, Inworld, Hume, Google (allow-listed), Azure (limited access) and OpenAI (eligible customers only) offer instant cloning from about 3–30 s of audio. Deepgram Aura-2 offers no cloning. PlayHT no longer exists.

### Cited Findings

**Independent latency: Coval perceived TTFA, 1-day window read 8 Sept 2026, 480 samples per model** (as quoted by [Gradium's benchmark page](https://gradium.ai/content/tts-latency-benchmark-2026). Gradium is a vendor, but the data is Coval's.)

| Model | P50 (ms) | P95 (ms) | IQR (ms) |
|---|---|---|---|
| Fluxions vui | 51 | 96 | 39 |
| Inworld TTS-2 Flash | 75 | 111 | 17 |
| Palabra TTS v1 | 103 | 144 | 17 |
| Inworld TTS-2 | 170 | 230 | 38 |
| ElevenLabs Flash v2.5 | 185 | 225 | 25 |
| Gradium TTS | 214 | 346 | 31 |
| Soniox TTS RT v2 | 255 | 311 | 72 |
| Rime Mist v3 | 256 | 295 | 21 |
| Cartesia Sonic 3.5 | 269 | 346 | 46 |
| Deepgram Aura-2 | 290 | 522 | 231 |
| Fish Audio S2.1 Pro (hosted) | 293 | 586 | 67 |
| ElevenLabs v3 Conversational | 320 | 393 | 49 |
| Cartesia Sonic 3.6 | 440 | 696 | 171 |
| OpenAI gpt-4o-mini-tts | n/a | 3,927 | n/a |

All rows above come from [Gradium/Coval](https://gradium.ai/content/tts-latency-benchmark-2026). Coval excludes TLS and WebSocket setup, which that page puts at about 50–200 ms, and measures English only.
- A separate Coval read on 28 Aug 2026 had Cartesia Sonic 3.6 at 454 ms (IQR 165) and Inworld Realtime TTS-2 at 166 ms. The day-to-day variance is real ([Gradium/Coval](https://gradium.ai/content/tts-latency-benchmark-2026)).
- Older pre-June-2026 round-trip-only Coval captures (May 2026): Cartesia Sonic-3 about 188 ms P50 (IQR 100), ElevenLabs Turbo v2.5 about 264 ms, Flash v2.5 about 288 ms, Deepgram Aura-2 about 313 ms. Rime Mist-v3 and Arcana were flagged for high latency variance ([Gradium, May 2026](https://gradium.ai/content/tts-latency-benchmark-2026); [Gradium blog](https://gradium.ai/blog/coval-tts-benchmarks-may-2026)).

**ElevenLabs**
- Flash models are about 75 ms model inference only. End-to-end latency depends on location and endpoint ([ElevenLabs latency docs](https://elevenlabs.io/docs/eleven-api/guides/how-to/best-practices/latency-optimization.md)).
- Vendor-published WebSocket TTFB for Flash by region: North America, Europe and South-East Asia 100–150 ms. **South Asia (India) and North-East Asia 150–200 ms.** Serving regions are the USA, the Netherlands and Singapore. Enterprise customers can get dedicated EU and **India data-residency** environments ([ElevenLabs latency docs](https://elevenlabs.io/docs/eleven-api/guides/how-to/best-practices/latency-optimization.md)).
- WebSocket is bidirectional and recommended when text arrives in real time (for example LLM output). `auto_mode` handles generation triggers. Without it, a slow text stream can stall audio until the chunk schedule fills (example: 125-char schedule, only 50 chars received) ([ElevenLabs latency docs](https://elevenlabs.io/docs/eleven-api/guides/how-to/best-practices/latency-optimization.md)).
- Voice type affects latency. Default, synthetic and **Instant Voice Clone (IVC)** voices are in the fastest group. **Professional Voice Clones (PVC) are slower**, and "optimization of PVC latency for Flash v2.5 is in progress" ([ElevenLabs latency docs](https://elevenlabs.io/docs/eleven-api/guides/how-to/best-practices/latency-optimization.md)).
- Eleven v4 and v4 Turbo launched 28 Sept 2026. Vendor claims: about 150 ms median "time to first speech" over WebSocket with network latency removed, and about 100 ms median inference ([Unite.AI](https://unite.ai/elevenlabs-launches-eleven-v4-with-low-latency-turbo-variant); [ProPakistani](https://propakistani.pk/2026/09/29/elevenlabs-unveils-its-fastest-most-expressive-voice-ai-models-yet/amp/)). Independent: Coval TTFA mean of 0.204 s for Eleven v4 Turbo over 568 samples ([Gradually.ai listing](https://www.gradually.ai/en/ai-models/eleven-v4-turbo/)).
- AA arena (Oct 2026): Eleven v4 Turbo is #1 at Elo 1330 and $40 per 1M chars. Eleven v4 is #2 at 1329 and $80 per 1M. v3 Conversational is 1210 at $50. Eleven v3 is 1178 at $100 ([AA leaderboard](https://artificialanalysis.ai/text-to-speech/leaderboard)).
- Flash v2.5 costs about $50 per 1M chars per third-party sources citing ElevenLabs pricing (half the credits of standard models). One competitor page instead says about $100 per 1M at the standard API rate ([LLM Reference](https://www.llmreference.com/model/eleven-flash-v2-5/elevenlabs); [Inworld comparison](https://inworld.ai/resources/best-voice-ai-tts-apis-for-real-time-voice-agents-2026-benchmarks)). This conflicts, so verify on the live pricing page.
- A competitor (Deepgram) cites an independent 255 ms median TTFB for ElevenLabs including network ([Deepgram blog](https://deepgram.com/learn/is-elevenlabs-real-time-what-developers-need-to-know)).
- Professional cloning reportedly needs 30+ min of audio, per a competitor's page ([Inworld comparison](https://inworld.ai/resources/best-voice-ai-tts-apis-for-real-time-voice-agents-2026-benchmarks)).

**Cartesia**
- Sonic-3.6: vendor claims under 90 ms to first audio, instant cloning from about 10 s of reference audio, and 44 languages. It is closed with no weights. Release is dated 27 Aug 2026 and described as beta ([MarkTechPost, 18 Aug 2026](https://www.marktechpost.com/2026/08/18/cartesia-ships-sonic-3-6-a-streaming-tts-model-that-now-leads-both-artificial-analysis-speech-arenas/); [Essa Mamdani guide](https://essamamdani.com/blog/cartesia-sonic-3-6-real-time-voice-agents-guide-2026)). The MarkTechPost article date (18 Aug) conflicts with the 27 Aug release date.
- AA Elo: Sonic 3.6 is 1285 and Sonic 3.5 is 1193, both $49 per 1M chars ([AA leaderboard](https://artificialanalysis.ai/text-to-speech/leaderboard)).
- Earlier vendor claims: Sonic-3 about 90 ms TTFA and Sonic Turbo about 40 ms. A 3-second clip gives a "usable" instant clone ([eesel.ai](https://www.eesel.ai/blog/cartesia-sonic-3-pricing)). Vapi measured Sonic 3 at a 166 ms median including network ([via Essa Mamdani guide](https://essamamdani.com/blog/cartesia-sonic-3-6-real-time-voice-agents-guide-2026)).
- Plans (third-party, unverified): Free 20k credits; Pro $5/mo with 100k credits and IVC; Startup $49/mo with 1.25M credits and PVC; Scale $299/mo with 8M credits. 1 credit per character, PVC 1.5 credits per character ([eesel.ai](https://www.eesel.ai/blog/cartesia-sonic-3-pricing); [CloudTalk](https://www.cloudtalk.io/cartesia-pricing/)). Sources disagree on which plan includes IVC.
- WebSocket multiplexing supports dozens of concurrent generations ([Inworld comparison](https://inworld.ai/resources/best-voice-ai-tts-apis-for-real-time-voice-agents-2026-benchmarks)).

**Inworld**
- Realtime TTS-2 (May 2026): vendor claims under 100 ms P99 TTFB, measured server-side. TTS-2 Flash: under 25 ms P99, also server-side. Instant clone from 5–15 s. Professional clone from 10+ min (preview). WebSocket supported ([Inworld](https://inworld.ai/resources/best-voice-ai-tts-apis-for-real-time-voice-agents-2026-benchmarks)). The page is internally inconsistent: it calls the under-100 ms figure both "server-side" and "end-to-end."
- AA: Realtime TTS-2 is Elo 1262 at $20.8 per 1M. TTS-2 Flash is 1220 at $10.4 per 1M ([AA leaderboard](https://artificialanalysis.ai/text-to-speech/leaderboard)).

**OpenAI**
- gpt-4o-mini-tts (March 2025) is steered by instructions and has a set of about a dozen preset voices ([TechCrunch](https://techcrunch.com/2025/03/20/openai-upgrades-its-transcription-and-voice-generating-ai-models/)). Pricing per third parties: $0.60 per 1M text input tokens plus $12 per 1M audio output tokens ([LLM Reference](https://www.llmreference.com/model/gpt-4o-mini-tts)). Coval measured a P95 of 3,927 ms, which makes it unsuitable for ultra-low latency ([Gradium/Coval](https://gradium.ai/content/tts-latency-benchmark-2026)).
- **Custom voices from recordings** are limited to "eligible customers" and enabled per organization via sales. Requirements:
  - A consent recording plus a 10–30 s sample from the same speaker, with at least 5 s of actual speech.
  - Usable with `gpt-4o-mini-tts`, the Realtime API (`gpt-realtime-2`), Chat Completions audio and GPT-Live.
  - Limit of 20 recording-based voices per organization.
  - Not compatible with Zero Data Retention.
  - `gpt-live-1` supports custom voices with English accents specified in session instructions.

  Source: [OpenAI custom voices docs](https://developers.openai.com/api/docs/guides/custom-voices).

**Deepgram Aura-2**
- $0.030 per 1K chars ($30 per 1M), Growth tier $0.027. **No voice cloning**: a fixed catalogue of 40+ voices ([Smallest.ai blog](https://smallest.ai/blog/top-fastest-text-to-speech-apis-in-2026); [rywalker.com](https://rywalker.com/research/deepgram-aura)). Vendor claims sub-200 ms ([BusinessWire, Apr 2025](https://www.businesswire.com/news/home/20250415446781/en)). Coval: 290 ms P50 with a very wide IQR of 231 ms ([Gradium/Coval](https://gradium.ai/content/tts-latency-benchmark-2026)).

**Hume Octave 2**
- Launched Oct 2025, preview. Vendor claims about 100 ms excluding network, versus about 200 ms for Octave 1. Clone from as little as 15 s ([AIWiki Hume Octave 2](https://www.aiwiki.ai/wiki/hume_octave_2); [Inworld comparison](https://inworld.ai/resources/best-voice-ai-tts-apis-for-real-time-voice-agents-2026-benchmarks)). Price $50–150 per 1M chars per a competitor's page ([Speechify alternatives page](https://speechify.ai/alternatives/hume)). No independent latency figure was found.

**Rime**
- Vendor docs cite about 70 ms on-prem latency ([Rime docs](https://docs.rime.ai/docs/models)). Coval: Mist v3 256 ms P50 ([Gradium/Coval](https://gradium.ai/content/tts-latency-benchmark-2026)). A competitor says Rime does not offer cloning ([Inworld comparison](https://inworld.ai/resources/best-voice-ai-tts-apis-for-real-time-voice-agents-2026-benchmarks)). This is unverified against Rime's own docs.

**PlayHT / PlayAI: defunct**
- Meta acqui-hired the PlayAI team in July 2025. The public API went offline 26 July 2025 and the service ended 31 Dec 2025. Voice clones were reportedly deleted with no migration path ([AIWiki PlayHT](https://www.aiwiki.ai/wiki/play_ht); [Murf transition guide](https://murf.ai/blog/play-ai-transition-guide); [Trillet](https://trillet.ai/blogs/playai-alternatives-2026)).

**Google and Azure**
- Google Chirp 3 Instant Custom Voice:
  - Reference audio of up to 10 s.
  - A scripted consent statement is required: "I am the owner of this voice and I consent to Google using this voice to create a synthetic voice model."
  - **en-IN is a supported locale.** Streaming synthesis is supported.
  - **Allow-list only** (contact sales).
  - Voice-cloning keys are stored client-side and sent with each request.

  Source: [Google Cloud docs](https://docs.cloud.google.com/text-to-speech/docs/chirp3-instant-custom-voice). AA lists Gemini 3.8 Flash TTS at Elo 1275 and $16.5 per 1M ([AA leaderboard](https://artificialanalysis.ai/text-to-speech/leaderboard)). Google WaveNet/Neural2 latency is 200–250 ms per a 2025 third-party benchmark ([cited by Inworld](https://inworld.ai/resources/best-voice-ai-tts-apis-for-real-time-voice-agents-2026-benchmarks)).
- Azure Personal Voice:
  - A recorded verbal consent statement is required. It also verifies that the speaker matches the voice audio.
  - The output is a speaker profile ID.
  - Zero-shot synthesis "matches your natural speaking style, intonation, and accent."
  - API access is limited and needs an application.

  Source: [Microsoft Learn](https://learn.microsoft.com/azure/ai-services/speech-service/personal-voice-overview). Azure Neural is about $15 per 1M chars ([Inworld comparison](https://inworld.ai/resources/best-voice-ai-tts-apis-for-real-time-voice-agents-2026-benchmarks)).

**Other 2025–2026 entrants** (AA arena, Oct 2026)
- Alibaba Qwen-Audio-3.1-TTS-Plus: Elo 1296, $19.3 per 1M.
- Speechify Simba 3.2: Elo 1244, $6.6 per 1M.
- Soniox TTS RT v2: Elo 1179, $14.2 per 1M.
- VUI Labs Luna TTS: Elo 1215.
- StepFun StepAudio 2.5: Elo 1214, $85 per 1M.

All from the [AA leaderboard](https://artificialanalysis.ai/text-to-speech/leaderboard). On Coval, Fluxions vui (51 ms) and Palabra (103 ms) are the fastest measured ([Gradium/Coval](https://gradium.ai/content/tts-latency-benchmark-2026)). ElevenLabs compared v4 Turbo against xAI TTS, which suggests xAI has a TTS API ([Unite.AI](https://unite.ai/elevenlabs-launches-eleven-v4-with-low-latency-turbo-variant)).

### Inferences
- For StutterClear's TTS leg, independent numbers put realistic cloud TTFA at about 75–300 ms before network setup. Vendor "40–100 ms" figures should not be used for budgeting.
- From India, add the ElevenLabs-documented 150–200 ms South Asia TTFB tier, or use enterprise India residency.
- Inworld TTS-2 Flash and ElevenLabs Flash v2.5 are the best independently verified latency-plus-cloning options. Eleven v4 Turbo is the best quality-plus-latency compromise if the Coval mean of about 204 ms holds.
- Cartesia's vendor claims are not borne out by Coval for Sonic 3.6 (440 ms P50, high variance). Sonic 3.5 (269 ms) is faster.
- Use IVC, not PVC, on ElevenLabs if latency is paramount, since PVC is slower on Flash.
- A word-by-word text stream fits poorly with chunk-schedule buffering. StutterClear should either enable `auto_mode` or equivalent flush controls, or send phrase-sized chunks. Otherwise the TTS will wait for more characters.
- Avoid OpenAI gpt-4o-mini-tts for the real-time path, and do not build on PlayHT.

### Gaps
- I could not load the live Coval board (benchmarks.coval.ai/tts) or the AA "controlled voice" (cloning) leaderboard, so cloned-voice quality rankings are missing.
- I found no independent TTFA for Hume Octave 2, Google Chirp 3 HD or Azure in 2026.
- I found no verified ElevenLabs IVC minimum sample length or current per-plan pricing from ElevenLabs' own pricing page.
- I did not verify that Cartesia supports incremental text input ("continuations") from its docs.

---

## Q2. Open-source / local TTS: first-packet latency, RTF, zero-shot cloning, license

### Takeaway
The open models that genuinely accept streaming text input are Kyutai TTS (DSM), CosyVoice 2/3 (bidirectional), Qwen3-TTS, VibeVoice-Realtime and Voxtral TTS (streaming listed). Their first-packet figures range from about 97 ms (Qwen3-TTS, paper claim) and 150 ms (CosyVoice 2) to about 300 ms (VibeVoice-Realtime). The commercially safe zero-shot cloners are Qwen3-TTS (Apache-2.0, 3 s reference), CosyVoice 3 (Apache-2.0) and Chatterbox Turbo (MIT, about 5 s, PerTh-watermarked). F5-TTS weights, XTTS-v2, Fish Speech and Fish S2 are non-commercial or need a paid license. Kokoro and Piper are fast but cannot clone.

### Cited Findings

| Model | Streaming text in? | First packet / RTF (hardware) | Zero-shot cloning | License | Source |
|---|---|---|---|---|---|
| **Kyutai TTS 1.6B** (Jul 2025, DSM) | Yes. Text can be piped in as an LLM generates it | 350 ms latency while serving 32 concurrent streams on one L40S, faster than real time | 10 s voice sample, but **voice-embedding model not released**; only a repository of provided voices (Expresso, VCTK). kyutai.org/tts also says you can clone from any sample when running locally, which is unclear and conflicting | Code MIT/Apache; weights reported CC-BY-4.0 (secondary) | [Kyutai TTS page](https://kyutai.org/next/tts); [Kyutai blog Jul 2025](https://kyutai.org/2025/07/03/tts-unmute-open-source.html); [DSM GitHub](https://github.com/kyutai-labs/delayed-streams-modeling); [learnopoly (secondary)](https://learnopoly.com/kyutai-releases-tt-of-text-in-2b-parameter-streaming-with-a-latency-of-220-ms-and-2-5-million-hours-of-training/) |
| **Kyutai Pocket TTS** 100M (Jan 13 2026) | n/a | About 200 ms to first audio, about 6x real time on a MacBook Air M4 using 2 CPU cores (secondary review) | About 5 s | Code MIT; per-voice licenses vary | [dev.to review](https://dev.to/andrew-ooo/pocket-tts-review-kyutais-100m-cpu-voice-cloner-2026-40lb) |
| **CosyVoice 2** (0.5B) | Yes, bidirectional streaming | First packet "as low as 150 ms" | Yes | Apache-2.0 (per ocdevel for v3) | [CosyVoice2 README mirror](https://gitserver.onethingai.com/ai-models/CosyVoice2-0.5B) |
| **Fun-CosyVoice3-0.5B-2512** | Yes (streaming 150 ms listed) | 150 ms streaming, about 4 GB VRAM | Yes. Speaker similarity SIM: 78.0 test-zh / 71.8 test-en / 75.8 test-hard, vs CosyVoice2 75.7/65.9/72.4 and Seed-TTS 79.6/76.2/77.6 | Apache-2.0 | [mirror README table](https://gitserver.onethingai.com/ai-models/CosyVoice2-0.5B); [ocdevel](https://ocdevel.com/blog/20250720-tts) |
| **Qwen3-TTS** 0.6B/1.7B (arXiv Jan 22 2026) | Yes | First packet 97 ms (0.6B) / 101 ms (1.7B) with the 12 Hz tokenizer (paper claim) | Yes, about 3 s; 10 languages | Apache-2.0 (models and tokenizers) | [arXiv 2601.15621](https://arxiv.org/abs/2601.15621v1); [ocdevel](https://ocdevel.com/blog/20250720-tts) |
| **VibeVoice-Realtime-0.5B** (Microsoft) | Yes. Incrementally encodes incoming text chunks, so it can start speaking from an LLM's first tokens | About 300 ms first audible on the HF card, about 200 ms in the GitHub doc (hardware-dependent) | Single-speaker; English only, 9 other languages experimental | MIT (research-oriented) | [HF model card](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B); [GitHub doc](https://raw.githubusercontent.com/microsoft/VibeVoice/refs/heads/main/docs/vibevoice-realtime-0.5b.md) |
| **Chatterbox Turbo** 350M (Resemble) | Not stated (chunked output) | Vendor 75 ms. The Decoder reports under 150 ms first audio. ocdevel reports about 470 ms first chunk at about 2x RT on about 6 GB. Conflicting | Yes, about 5 s (7–10 s per ocdevel); English-focused. The multilingual Chatterbox covers 23 languages | **MIT, commercial OK**; every output carries the **PerTh watermark** | [Resemble](https://www.resemble.ai/learn/models/chatterbox-turbo); [The Decoder](https://the-decoder.com/resemble-ai-drops-chatterbox-turbo-an-open-source-text-to-speech-model-that-clones-voices-in-five-seconds/); [ocdevel](https://ocdevel.com/blog/20250720-tts) |
| **Voxtral TTS** (Mistral) | Streaming listed | n/a | Yes, about 3 s. AA Elo 1087 | "Open-weight", specific license unstated | [ocdevel](https://ocdevel.com/blog/20250720-tts); [AA](https://artificialanalysis.ai/text-to-speech/leaderboard) |
| **F5-TTS** (~0.3B) | No (non-autoregressive flow matching) | RTF 0.15 at 16 NFE (paper). "Fast F5" RTF 0.030 on an RTX 3090 at 7 steps | Yes | **Conflict:** CC-BY-NC-4.0 (non-commercial) per GIGAGPU vs MIT per ocdevel. Likely code MIT, weights NC | [F5-TTS paper](https://arxiv.org/pdf/2410.06885); [Luna-TTS report](https://arxiv.org/pdf/2608.11593); [GIGAGPU](https://gigagpu.com/best-tts-models-2026/); [ocdevel](https://ocdevel.com/blog/20250720-tts) |
| **XTTS-v2** (Coqui) | n/a | No verified figure | Yes | CPML (Coqui Public Model License) | [GIGAGPU](https://gigagpu.com/best-tts-models-2026/) |
| **Fish Speech 1.5 / Fish Audio S2 (4B)** | n/a | S2 RTF 0.195 on an H200 with SGLang-Omni. Fish 1.5 RTF about 0.15–0.25 on an A100 | Yes, 10–30 s | Fish 1.5: CC-BY-NC-SA-4.0. S2: research license, **commercial use needs a paid license** | [Luna-TTS report](https://arxiv.org/pdf/2608.11593); [Spheron](https://www.spheron.network/blog/deploy-open-source-tts-gpu-cloud-2026/); [ocdevel](https://ocdevel.com/blog/20250720-tts) |
| **Spark-TTS** | n/a | RTF 0.136 on an L20 at concurrency 1 | Yes | Not verified | [Luna-TTS report](https://arxiv.org/pdf/2608.11593) |
| **Kokoro 82M** | n/a | RTF about 0.04–0.06 on an RTX 4090 (community); "about 200x real time on 4090"; 5x on a 32-core CPU | **No**, 54 preset voices | Apache-2.0. AA Elo 1061, $0.7 per 1M hosted | [Spheron](https://www.spheron.network/blog/deploy-open-source-tts-gpu-cloud-2026/); [ocdevel](https://ocdevel.com/blog/20250720-tts); [AA](https://artificialanalysis.ai/text-to-speech/leaderboard) |
| **Piper** | n/a | Real time on CPU / Raspberry Pi | **No** | MIT (per ocdevel) | [ocdevel](https://ocdevel.com/blog/20250720-tts) |
| **Sesame CSM-1B** | n/a | No figure | Context-conditioned | Apache-2.0 (secondary; "research and educational use" intent) | [OpenRouter listing](https://openrouter.ai/models/sesame/csm-1b) |
| **Zonos** | n/a | No figure | Yes | Apache-2.0 (secondary) | [freeapihub](https://freeapihub.com/ai-models/zonos) |
| **StyleTTS 2** | n/a | n/a | Yes | n/a. AA Elo 894 | [AA](https://artificialanalysis.ai/text-to-speech/leaderboard) |

- AA lists 17 open-weights models out of 97. The top open-weights entries are Fish Audio Drama 3 Preview (1244), Breeze TTS 2 (1222), Fish Audio S2 Pro (1116), Step Audio EditX (1097) and Voxtral TTS (1087) ([AA leaderboard](https://artificialanalysis.ai/text-to-speech/leaderboard)).
- Seed-TTS-eval style SIM numbers (above) are the most comparable public speaker-similarity metric. Only CosyVoice 3 versus CosyVoice 2 versus Seed-TTS was retrieved ([mirror README](https://gitserver.onethingai.com/ai-models/CosyVoice2-0.5B)).

### Inferences
- **On-device / self-hosted shortlist for StutterClear:**
  1. Qwen3-TTS-0.6B: Apache, about 97 ms first packet claimed, 3 s cloning, streaming.
  2. CosyVoice 3 0.5B: Apache, about 150 ms, bidirectional streaming, measured SIM.
  3. Chatterbox Turbo: MIT, built-in watermark, but first-chunk claims conflict.
  4. Kyutai Pocket TTS for CPU-only.
- Kyutai TTS 1.6B has the cleanest streaming-text design. However, arbitrary-voice cloning appears restricted by Kyutai's choice not to release the voice-embedding model, which is a blocker for "user's own voice" unless that has changed.
- None of the RTF figures are on the same hardware. A same-GPU bake-off (for example an RTX 4090 or laptop 4060), measuring first-packet latency with word-by-word text input, is needed before committing.
- Non-commercial licenses (F5-TTS weights, XTTS-v2 CPML, Fish S2 research) exclude these models from a commercial StutterClear unless the team licenses them.

### Gaps
- No verified latency or license found for Orpheus, Higgs Audio v2, Dia, MaskGCT, Llasa, IndexTTS2 or NeuTTS. Check their HF cards.
- No same-hardware first-packet benchmark found across open models.
- The CosyVoice 3 paper's own latency table was not retrieved, so the 150 ms figure is a README/aggregator claim.
- Kyutai TTS weights license (CC-BY-4.0) comes only from a secondary source.
- I did not confirm whether VibeVoice-Realtime allows arbitrary-voice zero-shot cloning.

---

## Q3. Accent preservation (for example Indian English) in cloned output

### Takeaway
Zero-shot cloners generally inherit the reference speaker's accent along with timbre. This is documented as an entanglement problem in the research literature and as intended behavior by at least one vendor. However, I found **no independent study** measuring accent drift toward American English for ElevenLabs, Cartesia or open models. Vendors that explicitly support or document accent handling include Azure Personal Voice ("matches ... accent"), Google Chirp 3 (en-IN locale), OpenAI gpt-live-1 (accent set via instructions), Smallest AI (accent retained by design), and Indic-specific open models (IndicF5 / Dhee-Indic-F5, Svara-TTS, Gnani Vachana).

### Cited Findings
- A UT Dallas paper finds zero-shot TTS "can generate speech that captures both the voice timbre and accent of a reference speaker," that disentangling them is hard, and that output "often inherits both the accent and timbre." It proposes inference-time activation steering to *reduce* accent ([arXiv 2603.05977](https://arxiv.org/pdf/2603.05977); [overview](https://awesomepapers.io/speech-audio/papers/2603.05977)).
- Smallest AI docs say a clone from a South Indian English speaker keeps that accent even when generating Hindi or Tamil, by design. For accent-neutral output they suggest a native-speaker reference ([Smallest AI docs](https://docs.smallest.ai/models/documentation/best-practices/voice-cloning-best-practices.md)).
- Azure Personal Voice zero-shot "matches your natural speaking style, intonation, and accent" ([Microsoft Learn](https://learn.microsoft.com/azure/ai-services/speech-service/personal-voice-overview)).
- Google Chirp 3 Instant Custom Voice supports en-IN plus many Indic locales (bn, gu, hi, kn, ml, mr, ta, te-IN). Keys created with en-US can synthesize some other locales. The docs say nothing about accent fidelity ([Google Cloud docs](https://docs.cloud.google.com/text-to-speech/docs/chirp3-instant-custom-voice)).
- OpenAI: "`gpt-live-1` supports custom voices with English accents. Specify the accent in session instructions" ([OpenAI docs](https://developers.openai.com/api/docs/guides/custom-voices)).
- Indic open models (all vendor or model-card claims, not independent evaluations):
  - Dhee-Indic-F5 (fine-tune of AI4Bharat IndicF5) targets Indian-accented English as a first-class target ([HF](https://huggingface.co/dheeyantra/dhee-indic-f5)).
  - Svara-TTS voiceclone-beta lists Indian English, with about 10 s reference ([HF](https://huggingface.co/kenpath/svara-tts-voiceclone-beta)).
  - Gnani.ai's Vachana TTS lists Indian English with cloning from under 10 s (Feb 2026) ([Business Today](https://www.businesstoday.in/amp/technology/story/gnaniai-launches-zero-shot-voice-cloning-tts-model-for-12-indic-languages-516896-2026-02-19)).
- ElevenLabs markets Indian-accent voices ([ElevenLabs Indian accent page](https://elevenlabs.io/text-to-speech/indian-accent)). A Resemble guide covers building Indian-English clones ([Resemble](https://www.resemble.ai/resources/how-to-create-an-indian-english-ai-voice-clone)). Both are marketing pages, not evaluations.

### Inferences
- For a person who stutters, the cloned voice should come from **fluent** reference segments. If the reference contains blocks or repetitions, zero-shot cloners may reproduce prosodic artifacts the same way they reproduce accent. This follows from the timbre/accent entanglement finding, applied by analogy, and has not been tested.
- Practical plan: run a small accent-preservation test with Indian English speakers. Use an accent-ID classifier plus speaker-similarity (for example a WavLM-based SIM) and listener ratings across ElevenLabs IVC, Cartesia, Inworld, Qwen3-TTS, CosyVoice 3 and IndicF5-based models. Keep the reference at least 10 s.
- The speech-to-speech route (Q4/Q5) keeps the accent by construction, because the user's real audio is reused. This is a strong argument for it if accent fidelity is a top user requirement.

### Gaps
- I found no published independent benchmark of accent drift (Indian English toward US English) for ElevenLabs, Cartesia, Inworld or OpenAI clones.
- The AA controlled-voice (cloning) leaderboard was not retrieved, and it may be English-default anyway.

---

## Q4. Real-time voice conversion (keeping the user's real voice)

### Takeaway
Published algorithmic or end-to-end latencies:
- LLVC: under 20 ms (CPU, 16 kHz). Zero-VC: 20 ms algorithmic.
- RT-VC: about 61 ms on an M3 CPU. StreamVC (Google): about 71 ms. SynthVC: about 77 ms.
- StreamVoice: about 124 ms. X-VC: 240 ms.
- Seed-VC real-time mode: about 300 ms algorithmic plus about 100 ms device (430 ms measured on an RTX 3060 Laptop GPU).
- Beatrice v2: one user reports about 38 ms round trip.

RVC has no authoritative ms figure; it depends on chunk and buffer settings. For StutterClear, voice conversion is attractive *only* for converting TTS output back into the user's own voice (TTS-to-VC "voice skin"). It is not useful for removing stutters, because VC preserves content and timing.

### Cited Findings
- **LLVC (Koe AI):** "latency of under 20ms at a bitrate of 16kHz and runs nearly 2.8x faster than real-time on a consumer CPU." The authors claim the lowest resource use and latency of any open-source VC model at the time (2023) ([arXiv 2311.00873](https://arxiv.org/abs/2311.00873v1)). Code: github.com/KoeAI/LLVC. LLVC is any-to-one, trained per target speaker; that is from the paper's framing and was not explicitly re-verified here.
- **StreamVC (Google, 2024):** 70.8 ms end-to-end as cited in the RT-VC paper. Designed for mobile ([StreamVC arXiv 2401.03078](https://arxiv.org/pdf/2401.03078); [RT-VC arXiv 2506.10289](https://arxiv.org/pdf/2506.10289)).
- **RT-VC (articulatory coding, 2025):** 61.4 ms on an Apple M3 CPU. Comparisons cited alongside it: SynthVC 77.1 ms, StreamVoice 124.3 ms ([RT-VC arXiv 2506.10289](https://arxiv.org/pdf/2506.10289); [StreamVoice arXiv 2401.11053](https://arxiv.org/pdf/2401.11053)). The table values came via search extraction because the PDF tables could not be parsed directly.
- **X-VC (2026, codec-space zero-shot streaming):** 240 ms model-induced latency. It reports Seed-VC real-time factors of 0.069–0.161 ([arXiv 2604.12456](https://arxiv.org/pdf/2604.12456)).
- **Zero-VC (2026):** 20 ms algorithmic latency (excluding inference), RTF 0.063 on a Xeon CPU, zero lookahead ([arXiv 2606.20218](https://arxiv.org/pdf/2606.20218)).
- **Seed-VC:**
  - Algorithmic delay ≈ Block Time × 2 + right context, about 300 ms algorithmic plus about 100 ms device.
  - Benchmark on an RTX 3060 Laptop GPU: **430 ms latency, 150 ms inference per chunk** (tiny 25M model, 0.18 s block, 10 diffusion steps).
  - Zero-shot from a 1–30 s reference.
  - **GPL-3.0**; repo **archived Nov 21 2025**.

  Source: [Seed-VC GitHub](https://github.com/Plachtaa/seed-vc).
- **Beatrice v2 (VCClient):** one user reports about 38 ms round trip with rmvpe_onnx. Results vary by environment, so this is not a benchmark ([Booth listing](https://booth.pm/en/items/4905799)).
- **RVC realtime:** latency depends on chunk length and ASIO/WASAPI buffer settings. No ms figure is given ([RVC Realtime GUI](https://huggingface.co/niel-blue/RVC-Realtime-GUI)).
- **Older baseline:** CycleVAE + multiband WaveRNN runs at RTF 0.87–0.95 on a single CPU core, with a 10 ms frame shift and 27.5 ms window ([arXiv 2105.09858](https://arxiv.org/pdf/2105.09858)).
- ElevenLabs offers a streaming **Voice Changer** API, cloud speech-to-speech ([ElevenLabs latency docs](https://elevenlabs.io/docs/eleven-api/guides/how-to/best-practices/latency-optimization.md)).

### Inferences
- **Hybrid design option:** fast generic TTS (for example Kokoro or Inworld Flash) followed by a low-latency any-to-one VC trained on the user's voice (LLVC-style, under 20 ms, or StreamVC-style, about 70 ms). This could beat zero-shot TTS cloning on both latency and speaker similarity, and it keeps cloning on-device.
- Seed-VC's about 430 ms total is too slow for "ultra-low latency." Its GPL-3.0 license and archived status are further negatives.
- VC alone does not remove stutters. It must be combined with segment excision or TTS.

### Gaps
- I found no head-to-head same-hardware comparison of RVC, LLVC, StreamVC, Seed-VC, kNN-VC and Beatrice.
- I found no kNN-VC latency figure (it is generally not streaming; unverified).
- I found no speaker-similarity numbers for real-time VC on atypical or dysfluent speech.
- StreamVC weights are not confirmed public.

---

## Q5. Speech editing (insert or replace words inside the user's real recording)

### Takeaway
Editing models can keep the user's real voice, accent and room acoustics around an edit. The academic stutter-specific editor is FluentSpeech (2023), and newer general editors include CosyEdit (2026), Step-Audio-EditX and Ming-UniAudio. However, published editors are offline or utterance-level and need bidirectional context, so they are not suited to sub-300 ms streaming. VoiceCraft was anecdotally about 90 s for a 3 s clip. End-to-end stutter-to-fluent waveform models (StutterZero/StutterFormer, Oct 2025) are an emerging alternative.

### Cited Findings
- **VoiceCraft:** a neural codec LM doing autoregressive token infilling conditioned on bidirectional context. At inference it diffs the original and target transcripts and forced-aligns to locate the codec spans to mask ([arXiv 2403.16973](https://arxiv.org/pdf/2403.16973)). An informal student project found it ">90 seconds for inference on a 3 second clip" and switched to FluentSpeech, aiming for about 0.5 s per 3 s of audio ([Erdős Institute project PDF](https://erdosinstitute.org/_files/ugd/d4971e_cf02b0072c9c445f854bcb1c41c35ed6.pdf)).
- **FluentSpeech (2023):** stutter-oriented automatic speech editing using a context-aware diffusion model. A stutter predictor localizes stutter regions, so edits do not have to be marked by hand. It released the SASE time-aligned stutter dataset ([arXiv 2305.13612](https://arxiv.org/pdf/2305.13612)).
- **CosyEdit (arXiv 2601.05329, Jan 2026):** end-to-end editing built from zero-shot TTS. Replacement-edit WER: CosyEdit 5.84%, Ming-UniAudio 9.65%, Step-Audio-EditX 11.41%. No latency is reported ([arXiv 2601.05329](https://arxiv.org/html/2601.05329v1)).
- **Step-Audio-EditX:** an open LLM-based audio model for expressive, paralinguistic editing plus zero-shot TTS ([CosyEdit paper context](https://arxiv.org/pdf/2601.05329)). It is open-weights on AA at Elo 1097 ([AA](https://artificialanalysis.ai/text-to-speech/leaderboard)).
- **StutterZero / StutterFormer (arXiv 2510.18938, Oct 2025):** waveform-to-waveform stuttered-to-fluent models that also predict transcripts, trained on synthetic pairs. StutterFormer shows a 28% WER decrease and 34% BERTScore improvement versus Whisper-Medium ([arXiv 2510.18938](https://arxiv.org/abs/2510.18938v1)).
- **STEAMROLLER (arXiv 2601.10223):** a pipeline of ASR, agentic transcript correction, then StyleTTS voice cloning to "preserve the speaker's authentic voice." This is essentially StutterClear's text path ([arXiv 2601.10223](https://arxiv.org/pdf/2601.10223)).

### Inferences
- **Lowest latency, highest fidelity:** detect and cut repetitions, prolongations and blocks, then crossfade the user's real audio. This needs no generation, so added latency is only the detector lookahead. It keeps voice and accent perfectly.
- Generative editing (FluentSpeech-style infill) can be reserved for smoothing joins or for blocks where cutting leaves unnatural gaps, and run in near-real-time "delayed playback" mode.
- Full word replacement, where the corrected text differs from what was said, needs TTS or editing. Here, a fast streaming zero-shot TTS (Q2) with a fluent-segment voice prompt is the practical route.

### Gaps
- I found no measured latency for FluentSpeech, CosyEdit, Step-Audio-EditX or Ming-UniAudio.
- I found no streaming or causal speech-editing model.
- The StutterZero latency and license were not checked.

---

## Q6. Consent, privacy and safety: verification practices, watermarking, legal landscape, storage and deletion

### Takeaway
The industry norm in 2026 for cloning a *real* person's voice is a **spoken, scripted consent statement by the same speaker**, matched against the sample: OpenAI, Google and Azure all do this. Add access gating (allow-lists or limited access) and blocked "no-go" public-figure voices (ElevenLabs).

EU AI Act Art. 50 machine-readable marking of synthetic audio applies from 2 Aug 2026 (with a reported transition to 2 Dec 2026 for systems already on the market). The US federal NO FAKES Act is still not law as of Oct 2026 (passed Senate Judiciary on 18 June 2026, stalled). Tennessee's ELVIS Act has applied since July 2024 and reaches tool providers. India's DPDP Rules (notified Nov 2025) phase in through about May 2027 and require erasure on consent withdrawal.

Audio watermarks (AudioSeal, PerTh) help with compliance but are removable by neural-codec or VC attacks.

### Cited Findings

**Vendor consent / verification practices**
- **OpenAI:**
  - The speaker must read an exact consent phrase: "I am the owner of this voice and I consent to OpenAI using this voice to create a synthetic voice model." Phrases exist in 16 languages, including Hindi.
  - The voice in the sample must match the consent recording.
  - Adult voices only. Disclosure that speech is AI-generated is required, and provenance metadata and watermarks must be preserved.
  - Not compatible with Zero Data Retention.

  Source: [OpenAI custom voices](https://developers.openai.com/api/docs/guides/custom-voices).
- **Google Chirp 3 Instant Custom Voice:** a scripted consent statement (custom scripts not allowed), recorded in the same environment as the reference. Allow-listed access ([Google Cloud docs](https://docs.cloud.google.com/text-to-speech/docs/chirp3-instant-custom-voice)).
- **Azure Personal Voice:** a verbal consent statement is required and is used to verify the speaker is the same person. Limited-access application; "no unapproved use case allowed" ([Microsoft Learn](https://learn.microsoft.com/azure/ai-services/speech-service/personal-voice-overview)).
- **ElevenLabs:**
  - "No-go voices" block cloning of prominent public figures ([ElevenLabs no-go voices](https://elevenlabs.io/docs/help-center/legal/what-are-no-go-voices.md)).
  - Only human-verified PVCs can be shared in the Voice Library ([ElevenLabs help](https://elevenlabs.io/docs/help-center/product/voices/voice-library/can-i-share-an-ai-generated-voice-in-the-voice-library)).
  - Evading guardrails "including voice verification (e.g., CAPTCHA)" can lead to a permanent ban (same help page).
  - A failed PVC verification brings a 24-hour wait ([ElevenLabs PVC help](https://elevenlabs.io/docs/it/help-center/product/voices/voice-cloning/what-can-i-do-if-i-failed-to-verify-my-professional-voice-clone-pvc.md)).

  The detailed mechanics of the voice CAPTCHA were not found.
- **Apple Personal Voice (on-device):**
  - 2023 launch: about 15 min of randomized prompts, trained on-device "to keep the user's information private," paired with Live Speech ([CNN via KTVZ](https://ktvz.com/money/cnn-business-consumer/2023/05/16/your-iphone-will-soon-be-able-to-replicate-your-voice-after-15-minutes-of-training/)).
  - iOS 26: **10 phrases, under a minute**, improved with on-device Apple Intelligence ([9to5Mac, Mar 2026](https://9to5mac.com/2026/03/19/ios-26-made-one-of-iphones-wildest-most-unique-features-a-lot-better/); [GSMSpecs](https://www.gsmspecs.com/news/ios-26-features-ai-personal-voice-updates)).
  - Third-party AAC apps can request to speak with Personal Voice but cannot capture its audio ([SpokenAAC](https://spokenaac.com/aac-apps-with-personal-voice/)).
  - Languages: en-US, es-MX, zh-CN ([GSMSpecs](https://www.gsmspecs.com/news/ios-26-features-ai-personal-voice-updates)).
- **Open models:** Chatterbox Turbo watermarks every output with PerTh, which is designed to survive compression. Resemble says removing it is against intended use ([Resemble](https://www.resemble.ai/learn/models/chatterbox-turbo)).

**Watermarking**
- **AudioSeal:** sample-level (localized) detection with a global detection probability plus a decoded message. Trained with EnCodec and distortion augmentations ([SoK arXiv 2503.19176](https://arxiv.org/html/2503.19176v2)).
- **Robustness:** neural compression (EnCodec, DAC) "poses the greatest challenge" even for systems trained with it. Polarity inversion, time-stretch and reverb break some schemes. The SoK paper introduces 8 black-box removal attacks ([SoK arXiv 2503.19176](https://arxiv.org/html/2503.19176v2)). AudioMarkBench benchmarks both removal and forgery ([arXiv 2406.06979](https://arxiv.org/html/2406.06979v2)). Overwriting attacks and self-voice-conversion attacks can strip or replace watermarks ([arXiv 2509.05835](https://arxiv.org/html/2509.05835v1)).

**EU AI Act**
- Art. 50 transparency obligations took effect **2 Aug 2026** ([Cooley, 3 Aug 2026](https://www.cooley.com/news/insight/2026/2026-08-03-eu-ai-act-transparency-obligations-take-effect-2-august-2026)). Providers of systems generating synthetic audio must apply machine-readable marking and detectability, and deployers must disclose deepfakes.
- A final Code of Practice on marking and labelling was reportedly published 10 June 2026. It recommends combining signed metadata and watermarking, since no single technique suffices.
- One source says the Digital Omnibus gives systems already on the market a transition to **2 Dec 2026**.

  Sources for these points: [Mishcon de Reya](https://www.mishcon.com/news/ai-act-transparency-obligations-code-of-practice-and-draft-guidelines); [Bristows](https://inquisitiveminds.bristows.com/post/102nfaz/the-european-commissions-code-of-practice-on-transparency-of-ai-generated-conten); [Dudkowiak](https://www.dudkowiak.com/blog/ai-act-article-50-ai-content-labeling-and-code-of-practice-from-august-2-2026/). These came from search summaries. The sources disagree on the dates and transition details, so verify against the Official Journal and Commission FAQ. A headline reports penalties of up to 3% of turnover ([PPC Land](https://ppc.land/eu-ai-content-rules-force-publishers-to-label-or-risk-3-of-turnover/)).

**US**
- **NO FAKES Act:** reintroduced 20 May 2026 as S.4591 / H.R.8915. The 2026 version adds a DMCA-style counter-notice and preempts state digital-replica laws with exceptions ([Manatt](https://www.manatt.com/insights/newsletters/client-alert/congress-reintroduces-the-no-fakes-act-what-s-new-in-the-2026-bill)). It passed Senate Judiciary on 18 June 2026 ([Sen. Blackburn](https://www.blackburn.senate.gov/index.php/2026/6/blackburn-coons-bipartisan-bill-to-protect-individuals-and-creators-from-deepfakes-passes-senate-judiciary-committee)). It is stalled "until at least November" after Sen. Cruz objected to unanimous consent ([Daily Journal, 7 Oct 2026](https://www.dailyjournal.com/article/395055-no-fakes-act-dead-for-now-as-cruz-objects-to-unanimous-consent-request)). **Not law as of 17 Oct 2026.**
- **Tennessee ELVIS Act:** effective 1 July 2024. It extends publicity rights to AI voice replicas and creates liability for distributing tools whose primary purpose is producing an individual's voice without authorization ([Phelps Dunbar](https://phelps.com/insights/the-elvis-act-protects-artists-voices-but-creates-risks-for-media-companies.html); [internetcases.com](https://internetcases.com/tag/ai/page/3/)). Academics call it "written incredibly broadly" and note First Amendment questions about the tool provision ([Bloomberg Law](https://news.bloomberglaw.com/esg/elvis-inspires-first-state-ai-protections-for-musicians-voices)).
- **Other states:**
  - California AB 2602 (digital replica contract terms) effective 1 Jan 2025.
  - New York amended Civil Rights Law §50 et seq. (Apr 2024).
  - Illinois amended its Right of Publicity Act (HB 4875, HB 4762) to cover digital voices and likenesses.

  Source: [WashU Law AI policy table](https://law.washu.edu/ai-policy-and-regulation-resources/intellectual-property-publicity).

**India: DPDP Act 2023 and DPDP Rules 2025**
- The Rules were notified 13/14 Nov 2025 and phase in. Board provisions apply immediately. Consent Manager registration follows at about 12 months (about 13 Nov 2026). The bulk of obligations (notice, security safeguards, breach notice, Rule 8 erasure, Data Principal rights) apply at 18 months (about 13 May 2027) ([Hogan Lovells](https://www.hoganlovells.com/en/publications/indias-digital-personal-data-protection-act-2023-brought-into-force-); [Mondaq](https://webiis10.mondaq.com/india/data-protection/1706878/indias-digital-personal-data-protection-act-phased-rollout-and-key-compliance-milestones)). Sources conflict on exact dates (12, 13 or 14 May 2027).
- Erasure: personal data must be erased, and processors instructed to erase it, on **consent withdrawal** or when the purpose is no longer served, whichever is earlier. Third Schedule 3-year inactivity limits apply to large e-commerce, gaming and social platforms, with **48-hour notice before erasure** ([dpdpa.com Rule 8](https://dpdpa.com/dpdparules/rule8.html); [DSCI FAQ](https://www.dsci.in/files/content/documents/2025/faq-on-data-retention-and-data-erasure-v2.pdf); [KPMG](https://assets.kpmg.com/content/dam/kpmgsites/in/pdf/2025/11/dpdp-rules-2025-guidance-to-dpdp-act-implementation.pdf)).
- Consent Managers must keep consent records for at least 7 years ([search summary of DPDP sources](https://www.seclore.com/fundamentals/dpdp-rules-2025-compliance-guide/)).
- Breach reporting within 72 hours is cited as a core obligation ([TCSA roadmap](https://www.tcsa.in/resources/dpdp-rules-2025-implementation-roadmap)).

### Inferences
**Best-practice recommendations for StutterClear.** These are synthesized from the vendor practices and laws above. Treat them as design guidance, not legal advice.

1. **Self-only enrollment with a spoken consent phrase.** Use a scripted consent phrase in the user's language (Hindi or English), mirroring OpenAI, Google and Azure. Run speaker verification so the consent speaker matches the enrollment audio, and add a liveness challenge (randomized phrase, "voice CAPTCHA"-style) to stop someone enrolling another person's recordings.
2. **Keep the voice profile on-device where possible**, following Apple Personal Voice and Google's client-side voice keys. Prefer storing a speaker embedding or fine-tuned adapter over raw audio, and encrypt it with platform keystores. Raw enrollment audio should be deleted after the profile is built unless the user opts in.
3. **Make deletion simple.** One-tap deletion of the voice profile, enrollment audio and any vendor-side clone, with propagation to processors through vendor delete APIs. This matches DPDP's withdrawal-triggered erasure. Keep a consent log (who, when, which phrase, hash of the consent audio).
4. **Watermark all synthetic output** (AudioSeal, PerTh, or vendor watermark) and attach provenance metadata. This meets EU Art. 50 marking from Aug 2026. Treat it as compliance and forensics, not robust security, because codec and VC attacks can remove it.
5. **Restrict output use.** The voice should only drive the user's own live speech. Do not allow export as arbitrary TTS, to limit misuse and ELVIS-style "tool" exposure. Disclose to listeners where required (EU deployer duties).
6. **Vendor selection for Indian users:** check data residency (ElevenLabs offers India residency for enterprise) and whether the vendor's voice-creation endpoint conflicts with zero-retention requirements (OpenAI's does).

### Gaps
- I found no details on SynthID's audio watermark (robustness, availability to third parties).
- I found no public technical description of ElevenLabs' voice CAPTCHA, or its exact IVC safeguards.
- I found nothing explicit on how the DPDP Act classifies voice or voiceprints (the Act has no separate "sensitive" category, to my knowledge; not verified here).
- I did not verify the Illinois BIPA voiceprint scope or recent amendments.
- I did not read the EU Code of Practice text directly. Dates and transition details came from law-firm summaries via search.
- Apple's third-party Personal Voice API (`AVSpeechSynthesizer` Personal Voice authorization) was not verified against Apple developer docs. Only the AAC-app usage was confirmed by a secondary source.
