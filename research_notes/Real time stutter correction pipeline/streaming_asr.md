# Streaming Speech-to-Text (ASR) for StutterClear: Cloud and Local Options (as of Oct 2026)

Research date: 2026-10-17. The field moves fast, so publication dates are noted wherever available. "Vendor claim" means the number comes from the vendor's own page. "Independent" means a third-party benchmark. Third-party pricing aggregators are flagged where used. Note the latency definitions differ between benchmarks: Artificial Analysis (AA) measures from Silero-VAD end-of-speech to the first final or partial transcript event. Speko measures end-of-turn, time-to-first-token from speech start, and "live words" (word spoken until it is stable in the transcript). These numbers are not interchangeable.

## Q1. Cloud streaming APIs: partial/final latency, endpointing, timestamps, confidence, biasing, disfluency handling, price

### Takeaway
On independent 2026 benchmarks, Deepgram Flux, Soniox, Cartesia Ink-2, ElevenLabs Scribe v2 Realtime and Azure MAI-Transcribe-2 give the fastest partials. ElevenLabs Scribe v2 Realtime, Cartesia Ink-2 and Azure MAI-Transcribe-2 are the most accurate. Deepgram Flux is among the fastest but has a higher WER. Soniox ($0.12/hr) and AssemblyAI Universal-Streaming ($0.15/hr) are the cheapest. OpenAI's realtime transcription models are the slowest at producing live words. Benchmarks disagree a lot on absolute numbers, so StutterClear should measure on its own stuttered audio.

### Cited Findings

**Independent benchmark 1: Artificial Analysis AA-WER Streaming (published 1 Jun 2026)**
- Method: about 8 h of audio (AA-AgentTalk, which is private, plus Earnings22-Cleaned-AA and VoxPopuli-Cleaned-AA). Latency is measured from Silero-VAD end of speech to (a) the first final transcript and (b) the first transcript event of any kind (partial or final). The index weights are 50% AgentTalk, 25% VoxPopuli and 25% Earnings22 — [AA article](https://artificialanalysis.ai/articles/new-streaming-speech-to-text-benchmark-aa-wer-streaming); [AA leaderboard](https://artificialanalysis.ai/speech-to-text/streaming)
- Cartesia Ink-2 (semantic endpoints): final WER 3.59%, final latency 0.21 s — [AA article](https://artificialanalysis.ai/articles/new-streaming-speech-to-text-benchmark-aa-wer-streaming)
- ElevenLabs Scribe v2 Realtime: final WER 3.64% at 0.14 s, partial WER 3.65% at 0.13 s, $6.50 per 1k min (= $0.39/hr) — [AA article](https://artificialanalysis.ai/articles/new-streaming-speech-to-text-benchmark-aa-wer-streaming)
- Cartesia Ink-2 (external endpoints): final WER 3.66% at 0.09 s, partial WER 4.33% at 0.07 s, $4 per 1k min — [AA article](https://artificialanalysis.ai/articles/new-streaming-speech-to-text-benchmark-aa-wer-streaming)
- AssemblyAI U3 Realtime Pro: partial WER 4.46% at 0.47 s — [AA article](https://artificialanalysis.ai/articles/new-streaming-speech-to-text-benchmark-aa-wer-streaming)
- Deepgram Flux: final WER 7.36% at 0.020 s, partial at 0.019 s. Deepgram Nova-3 Realtime: partial WER 6.69% at 0.057 s, final at 0.06 s — [AA article](https://artificialanalysis.ai/articles/new-streaming-speech-to-text-benchmark-aa-wer-streaming)
- Soniox Realtime: about 0.05 s partial and 0.06 s final, $2 per 1k min (= $0.12/hr), described as the "least accurate" at 11.9% WER — [AA article](https://artificialanalysis.ai/articles/new-streaming-speech-to-text-benchmark-aa-wer-streaming)
- NVIDIA Nemotron 3 ASR (80 ms chunk): partial latency 0.04 s. Mistral Voxtral Mini Transcribe Realtime: final WER 5.3%, partial 9.6%. OpenAI "GPT Realtime Whisper": final WER 5.1%, partial 7.5%. Google Chirp 3 Streaming had the best partial accuracy on VoxPopuli (2.2%) — [AA article](https://artificialanalysis.ai/articles/new-streaming-speech-to-text-benchmark-aa-wer-streaming)
- The streaming field ranges from $2 to $17 per 1k minutes. The leaderboard covers 28 of 45 models from about 22 providers, including Microsoft AI, Amazon Bedrock, Meta, Rev AI, Gradium, Smallest.ai and Speechmatics — [AA leaderboard](https://artificialanalysis.ai/speech-to-text/streaming)
- A search-engine summary of the AA leaderboard (not verified on the page) gave final latencies of about 0.054 s for Soniox v5, 0.19 s for AssemblyAI Universal-3.5 Realtime Pro and 0.317 s for Speechmatics Realtime Enhanced — [AA leaderboard](https://artificialanalysis.ai/speech-to-text/streaming) (treat as unverified)

**Independent benchmark 2: Speko live benchmark (run "07-03", © 2026; the methodology text was not visible on the page)** — [Speko Benchmarks](https://benchmarks.speko.ai/)
- WER: Azure MAI-Transcribe-2 Streaming 6.2%, Inworld Realtime STT-1 7.0%, AssemblyAI Universal-3.5/3.6 Pro 7.2%, ElevenLabs Scribe v2 Realtime 8.2%, OpenAI GPT Live Transcribe 8.8%, Speechmatics Enhanced 9.0%, Soniox stt-rt-v5 9.5%, OpenAI GPT Realtime Whisper 10.4%, Deepgram Nova-3 11.0%, Gladia Solaria-1 12.3%, Cartesia Ink-2 12.6%, Deepgram Flux 12.8%, AssemblyAI Universal-Streaming English 14.3%, OpenAI GPT-4o Transcribe 19.3%
- End-of-turn latency (p50 / p90): Meta Muse Voice Transcribe 301 ms / 584 ms, Soniox stt-rt-v5 489 ms / 2.40 s, AssemblyAI U-3.6 Pro 589 ms / 1.58 s, Deepgram Flux Multilingual 654 ms / 1.36 s, Deepgram Nova-3 771 ms / 1.21 s, Deepgram Flux 773 ms / 1.48 s, ElevenLabs Scribe v2 RT 793 ms / 1.06 s, Gladia Solaria-1 1.09 s / 1.59 s, OpenAI GPT-4o Transcribe 1.28 s / 1.70 s, Speechmatics Standard 1.45 s / 1.81 s
- Time to first token (p50 / p90): Azure MAI-Transcribe-2 241 ms / 342 ms, Deepgram Flux Multilingual 506 ms, Deepgram Flux 641 ms, Gladia 671 ms, Speechmatics Standard 690 ms, AssemblyAI U-3.6 Pro 775 ms, Soniox 786 ms, Deepgram Nova-3 885 ms, OpenAI GPT Realtime Whisper 987 ms, Cartesia Ink-2 1.19 s, ElevenLabs Scribe v2 RT 2.02 s, OpenAI GPT-4o Transcribe 4.72 s
- Live-word latency (p50 / p90): Azure MAI-Transcribe-2 114 ms / 515 ms, Deepgram Flux Multilingual 180 ms, Deepgram Flux 190 ms / 586 ms, Speechmatics Standard 403 ms, Soniox 496 ms / 739 ms, Deepgram Nova-3 583 ms, AssemblyAI U-3.6 Pro 815 ms / 2.71 s, ElevenLabs Scribe v2 RT 945 ms / 1.97 s, Cartesia Ink-2 1.22 s, Gladia 1.50 s / 3.98 s, OpenAI GPT Realtime Whisper 3.96 s / 11.28 s
- "Rewrites" (words the live transcript takes back, per final word): Cartesia Ink-2 0%, Meta Muse 0%, Soniox 13%, ElevenLabs 29%, Deepgram Nova-3 33%, Speechmatics 35%, AssemblyAI U-3.6 Pro 35%, Azure MAI-Transcribe-2 37%, Deepgram Flux 40%, xAI Grok Voice Transcribe 2.0 64%, Gladia Solaria-1 156%
- Conflict: another 2026 benchmark (Gradium, a competing vendor) reports a median time-to-first-token of 2,073 ms for ElevenLabs Scribe v2. That conflicts with ElevenLabs' "~150 ms" claim and AA's 0.13 s partial latency, and probably reflects a different metric definition — [Gradium STT benchmark](https://gradium.ai/content/stt-api-benchmark-2026-latency-accuracy) (via search summary). Speko's 2.02 s TTFT for ElevenLabs agrees with Gradium — [Speko](https://benchmarks.speko.ai/)

**Deepgram (Nova-3, Flux)**
- Flux is a "conversational speech recognition" model with turn detection built into the model. At launch it was reported at about 260 ms end-of-turn detection, "Nova-3 level" accuracy, keyterm prompting, and 100+ concurrent streams per GPU — [SpeechTechMag launch coverage](https://www.speechtechmag.com/Articles/News/Speech-Technology-News/Deepgram-Launches-Flux-Conversational-Speech-Recognition-Model-171756.aspx); [Deepgram Flux page](https://deepgram.com/flux). Later Flux Multilingual coverage cites "end-of-turn decisions under 400 ms" and mid-call switching across 10 languages — [Implicator](https://www.implicator.ai/deepgram-launches-flux-multilingual-speech-model-with-10-language-mid-call-switching/)
- Pricing (official, streaming, PAYG / Growth): Nova-3 Monolingual $0.0048 / $0.0042 per min (about $0.29/hr PAYG), Nova-3 Multilingual $0.0058 / $0.0050, Flux English $0.0065 / $0.0057 (about $0.39/hr), Flux Multilingual $0.0078 / $0.0068. Add-ons: Keyterm Prompting +$0.0013/min, Diarization +$0.0020/min. The page labels these "limited-time promotional rates on streaming" and comes with a $200 free credit — [Deepgram pricing](https://deepgram.com/pricing). A third-party source gives the regular, non-promo Nova-3 streaming rate as $0.0077/min — [convertaudiototext](https://convertaudiototext.com/blog/deepgram-nova-3-explained)
- Filler words: `filler_words` (boolean, default false) is available for the Nova, Nova-2 and Nova-3 general models in English. When false, "uh" and "um" are removed. When true, the transcript includes uh, um, mhmm, mm-mm, uh-uh, uh-huh and nuh-uh, spelled canonically regardless of duration ("uhhhh" becomes "uh"). The docs do not address repetitions or false starts, and Flux support is ambiguous (it appears in the page badges but not the text) — [Deepgram filler words docs](https://developers.deepgram.com/docs/filler-words)

**AssemblyAI (Universal-Streaming, Universal-3.x Pro Realtime)**
- Official pricing: Universal-3.6 Pro Realtime (`universal-3-6-pro`) is $0.45/hr with keyterms included (up to 100, updatable mid-stream). Universal-Streaming English and Multilingual are $0.15/hr, with keyterms +$0.04/hr on English. Billing is by session duration (the time the WebSocket is open, idle time included). Concurrency is 100 new streams per minute on PAYG, with $50 free credit. The older IDs `u3-rt-pro` and `u3-pro` are being superseded — [AssemblyAI pricing](https://www.assemblyai.com/pricing)
- Launch claim (vendor): Universal-Streaming delivers immutable transcripts in about 300 ms — [AssemblyAI launch blog](https://assemblyai.com/blog/introducing-universal-streaming)
- Turn detection combines acoustic and semantic features with silence-based methods — [AssemblyAI Universal-Streaming page](https://www.assemblyai.com/universal-streaming)
- Filler words: for pre-recorded audio, fillers are removed by default and `disfluencies: true` keeps them. I could not confirm a streaming equivalent — [AssemblyAI filler words docs](https://www.assemblyai.com/docs/speech-to-text/pre-recorded-audio/filler-words)

**Soniox (stt-rt-v5)**
- Real-time list price is about $0.12/hr (async $0.10/hr). This is an hourly equivalent of token billing at roughly 30,000 audio tokens per hour. Custom context and translation are billed by token usage — [Soniox pricing](https://soniox.com/pricing)
- Tokens stream with `is_final`. Non-final tokens may change, and final tokens are sent once and never repeated. Context customization has `general` (key/value domain info), `text` (background passage), `terms` (vocabulary) and `translation_terms`. A `finalize` message forces pending tokens to finalize. The service covers 60+ languages — [Soniox real-time docs](https://soniox.com/docs/stt/rt/real-time-transcription)

**Speechmatics**
- `max_delay` ranges from 0.7 to 4 s (default 4) and sets the delay to a final transcript. `end_of_utterance_silence_trigger` ranges from 0 to 2 s, where 0 disables it. The docs recommend 0.5–0.8 s for voice AI and 0.8–1.2 s for dictation, kept lower than max_delay — [Speechmatics turn detection docs](https://docs.speechmatics.com/speech-to-text/realtime/turn-detection)
- The Voice SDK has an `ADAPTIVE` mode that adjusts the delay based on speech rate, pauses and disfluencies, with a default EOU trigger of 0.2 s in that SDK — [speechmatics-voice on PyPI](https://pypi.org/project/speechmatics-voice/)
- Disfluencies: in English, about 50 hesitation tokens (um, uh, hmm, erm...) are automatically tagged `disfluency` on each word. `transcript_filtering_config.remove_disfluencies` (default false) strips them — [Speechmatics formatting docs](https://docs.speechmatics.com/speech-to-text/formatting). A 29 Jul 2026 realtime changelog extends disfluency detection and removal to 19 more languages — [Speechmatics changelog](https://speechmatics.featurebase.app/changelog/20260729-realtime-saas)
- Pricing: the official page says "from $0.12/hr" without breaking out real-time rates — [Speechmatics pricing](https://www.speechmatics.com/pricing). A third-party tracker reports real-time Standard at $0.24/hr and real-time Enhanced cut from $0.56 to $0.43/hr on 6 Jul 2026 — [usagepricing tracker](https://www.usagepricing.com/blueprint/activity/speechmatics-2026-07-06-realtime-stt-price-cut)

**ElevenLabs Scribe v2 Realtime**
- Vendor claim: about 150 ms ("sub-150 ms") partials, 90+ languages — [ElevenLabs realtime STT page](https://elevenlabs.io/realtime-speech-to-text)
- WebSocket parameters: `commit_strategy` = manual | vad | turn_prediction, plus `vad_threshold`, `vad_silence_threshold_secs`, `min_speech_duration_ms`, `min_silence_duration_ms`, `keyterms` (array), `include_timestamps` (default false) and `no_verbatim` (default false; true removes filler words, false starts and disfluencies). Messages: `partial_transcript` (text only), `committed_transcript`, and `committed_transcript_with_timestamps` (words with start/end in seconds, a `logprob` field described as a per-word confidence score, and characters). Partials carry no timestamps or confidence — [ElevenLabs realtime API reference](https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime)
- Price: $0.39/hr standard, which matches AA's $6.50 per 1k min — [AA article](https://artificialanalysis.ai/articles/new-streaming-speech-to-text-benchmark-aa-wer-streaming); [The Rundown tool profile](https://www.therundown.ai/tools/scribe-v2-realtime) (third-party). A keyterm add-on of $0.05/hr is reported by a third party, and it is unclear whether it applies to realtime

**OpenAI**
- `gpt-realtime-whisper` (released 7 May 2026 alongside GPT-Realtime-2) is a streaming STT model emitting low-latency transcript deltas. It is priced at $0.017/min (about $1.02/hr) by audio duration — [OpenAI model page](https://developers.openai.com/api/docs/models/gpt-realtime-whisper); [9to5Mac](https://9to5mac.com/2026/05/07/openai-has-new-voice-models-that-reason-translate-and-transcribe-as-you-speak/). A third party says it has tunable delay settings from "minimal" to "xhigh" — [therouter.ai](https://therouter.ai/models/openai--gpt-realtime-whisper/) (unverified)
- gpt-4o-transcribe is about $0.006/min and gpt-4o-mini-transcribe about $0.003/min (third-party) — [Forasoft](https://www.forasoft.com/blog/article/openai-realtime-api-pricing)
- Independent latency: OpenAI's models are the slowest live-word performers on Speko (GPT Realtime Whisper 3.96 s p50 live words, GPT-4o Transcribe 4.72 s TTFT) — [Speko](https://benchmarks.speko.ai/)

**Google (Chirp 3 / Gemini Live)**
- Chirp 3 streaming is about $0.016/min (about $0.96/hr), with endpointing_sensitivity (standard / short / supershort), voice activity events, and a reported 5-minute streaming session limit. All of this comes from third-party profiles, since Google publishes no ms latency figure — [OpenTranscription Chirp 3 profile](https://opentranscription.io/blog/google-cloud-chirp-3-capabilities-architecture-costs-and-com-profile.html); [OpenRouter Chirp 3](https://openrouter.ai/google/chirp-3)
- Gemini Live API transcription has a `mode` setting of VERBATIM (default) or SMART. Verbatim keeps fillers ("um", "uh", "like"), repetitions and false starts — [Gemini Live transcription docs](https://ai.google.dev/gemini-api/docs/live-api/live-transcribe)

**Microsoft Azure**
- Azure MAI-Transcribe-2 Streaming leads Speko on WER (6.2%), TTFT (241 ms) and live-word latency (114 ms) — [Speko](https://benchmarks.speko.ai/)
- Segmentation is controlled by `Speech_SegmentationSilenceTimeoutMs`, `Speech_SegmentationStrategy` and `Speech_SegmentationMaximumTimeMs` — [Azure Speech SDK PropertyId reference](https://learn.microsoft.com/en-us/javascript/api/microsoft-cognitiveservices-speech-sdk/propertyid). Developers report undocumented interactions, such as 15 s or 30 s forced segmentation — [Microsoft Q&A thread](https://learn.microsoft.com/en-us/answers/questions/1397127/speech-segmentationsilencetimeoutms-and-speech-seg)
- A community answer (about 3.5 years old) says Azure automatically removes filler words, with no official verbatim switch — [Microsoft Q&A](https://learn.microsoft.com/en-us/answers/questions/1165008/how-to-config-disfluency-removal-using-rest-api)

**AWS Transcribe**
- The streaming price is in conflict. October 2026 calculators list $0.01/min ($0.60/hr) in us-east-1 with custom vocabulary included — [costgoat](https://costgoat.com/pricing/amazon-transcribe). Older blogs cite $0.024/min — [BrassTranscripts](https://brasstranscripts.com/blog/aws-transcribe-pricing-per-minute-2025-better-alternative). Verify on the AWS pricing page

**Cartesia Ink-2**
- Vendor claim: 0.1 s time-to-final-transcript, built-in turn detection (no separate VAD) — [Cartesia Ink page](https://www.cartesia.ai/ink); [Cartesia Ink-2 blog](https://www.cartesia.ai/blog/ink-2). Soniox (a competitor) measured about 299–328 ms partials and about 1,584 ms finals — [Soniox benchmarks](https://soniox.com/benchmarks)
- Price: about $0.40/hr at Scale-plan rates (3 credits/s) — [Cekura on Cartesia pricing](https://www.cekura.ai/blogs/cartesia-pricing). This conflicts with AA's $4 per 1k min (about $0.24/hr) — [AA article](https://artificialanalysis.ai/articles/new-streaming-speech-to-text-benchmark-aa-wer-streaming)

**Other 2025–2026 entrants seen in independent benchmarks:** Azure MAI-Transcribe-2, Meta Muse Voice Transcribe, Inworld Realtime STT-1, xAI Grok Voice Transcribe 1.0/2.0, Nari Qwen3-ASR (Fast), Modulate Velma 2, Smallest AI Pulse, Gradium ASR, Hopper nemotron-asr and Gladia Solaria-1 — [Speko](https://benchmarks.speko.ai/). xAI's STT has `filler_words` (default false, so fillers are removed) — [xAI STT docs](https://docs.x.ai/developers/model-capabilities/audio/speech-to-text)

### Inferences
- The two independent benchmarks rank vendors very differently. Deepgram Flux has the fastest partials on both, but the worst WER among the major vendors on Speko (12.8%) and a moderate one on AA (7.36%). Cartesia Ink-2 is top-accuracy on AA but 12.6% WER and 1.19 s TTFT on Speko. Test sets differ (AA weights agent-style speech; Speko's is unknown), so neither result transfers to stuttered speech.
- For a pipeline that feeds partials into an LLM corrector and then TTS, partial stability (Speko's "rewrites") matters as much as raw speed. Cartesia Ink-2 (0%) and Soniox (13%) rewrite least. Deepgram Flux (40%) and Gladia (156%) rewrite most.
- Cost is not the deciding factor. Every major option is $0.12–$1.02 per audio hour.

### Gaps
- No official per-word confidence or timestamp specification was verified for Deepgram Flux, Soniox (the docs page shown was truncated), Cartesia Ink-2, Azure MAI-Transcribe-2 or Gladia.
- Gladia Solaria and Azure MAI-Transcribe-2 pricing were not found. The AA per-model latency table could not be scraped (JS-rendered), so only the models named in the AA article are covered.
- OpenAI realtime VAD parameters (server_vad / semantic_vad, silence_duration_ms) were not verified in this session.
- Speko's methodology section could not be read, and its per-provider prices could not be reliably paired with names.

## Q2. Open-source / local streaming ASR: chunk sizes, algorithmic latency, RTF, WER

### Takeaway
For true low-latency local streaming, the strongest 2026 options are NVIDIA Nemotron Speech Streaming (cache-aware FastConformer-RNNT, 80–1120 ms chunks, about 7–8.4% average WER), Moonshine v2 (50–258 ms response latency on an Apple M3 CPU, 6.65–12% Open-ASR WER), Kyutai STT 1B (0.5 s delay with semantic VAD, MLX on iPhone) and Mistral Voxtral Realtime (4.4B, Apache-2.0, 80–2400 ms configurable delay). Whisper-based streaming wrappers (whisper_streaming, WhisperLive, faster-whisper chunking) add seconds of latency and are a poor fit.

### Cited Findings

**NVIDIA Nemotron Speech Streaming EN 0.6B (released 13 Mar 2026; earlier checkpoint 5 Jan 2026)**
- Cache-aware FastConformer (24 layers) with an RNN-T decoder, about 600M params. Chunk sizes are 80, 160, 560 and 1120 ms via `att_context_size`, selectable at inference without retraining. Punctuation and capitalization are built in. NVIDIA Open Model License (commercial use allowed) — [HF model card](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
- Average WER by chunk: 1.12 s 6.93% (LS-clean 2.32, LS-other 4.84, AMI 11.73, Earnings22 12.52), 0.56 s 7.07%, 0.16 s 7.67%, 0.08 s 8.43% — [HF model card](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
- Nemotron 3.5 ASR Streaming 0.6B (4 Jun 2026) covers 40 language-locales with language-ID prompting — [HF model card](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b). A secondary source reports FLEURS fr 9.8% at 320 ms and en 7.9% at 1.12 s — [localvoxtral GitHub issue](https://github.com/T0mSIlver/localvoxtral/issues/907)
- Independent (Microsoft CoreAI, arXiv 2604.14493, 16 Apr 2026): Nemotron at 0.56 s delay gives the "best streaming WER (7.28%)" of all models tested on an 8-set ESB average. By configuration: 0.08 s delay 8.91%, 0.16 s 8.04%, 1.12 s 7.46%. ONNX on CPU (batch size 1): FP32 8.03% at RTFx 6.73, int8 1.28 GB 8.01% at RTFx 7.25, int4 0.67 GB 8.20% at RTFx 7.20 — [arXiv 2604.14493](https://arxiv.org/html/2604.14493v1)
- AA measured Nemotron 3 ASR (80 ms) at 0.04 s partial latency — [AA article](https://artificialanalysis.ai/articles/new-streaming-speech-to-text-benchmark-aa-wer-streaming)
- In the Voxtral paper's comparison, Nemotron Streaming at 560 / 1120 ms scored 9.59 / 9.41 on English short-form and 14.29 / 13.02 on long-form — [Voxtral Realtime paper](https://arxiv.org/html/2602.11298v2)

**Parakeet / Canary (NVIDIA, offline-first)**
- In chunked streaming they degrade substantially. Parakeet TDT-0.6B-v3 at 2.4 s delay scored 12.83% average (CPU RTFx 1.38), and Canary-1B-v2 at 4.8 s delay 12.45% — [arXiv 2604.14493](https://arxiv.org/html/2604.14493v1)
- Offline Open ASR leaderboard: Canary-Qwen-2.5B leads at 5.63% WER (RTFx 418) — [The Decoder](https://the-decoder.com/open-asr-leaderboard-tests-more-than-60-speech-recognition-models-for-accuracy-and-speed/); [Northflank](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks). Parakeet CTC 1.1B has RTFx 2793.75 vs Whisper large-v3 at 68.56 (WER 6.68 vs 6.43) — [HF Open ASR blog](https://huggingface.co/blog/open-asr-leaderboard). Per the leaderboard paper, CTC/TDT decoders give the best RTFx, while conformer encoders with transformer decoders give the best WER — [Open ASR Leaderboard paper](https://arxiv.org/html/2510.06961v4)
- A "Parakeet Realtime EOU 120M" is listed as append-only streaming at 80–160 ms latency without punctuation (secondary source) — [localvoxtral issue](https://github.com/T0mSIlver/localvoxtral/issues/907)

**Kyutai STT (delayed-streams modeling)**
- stt-1b-en_fr: about 1B params, 0.5 s delay, semantic VAD built in. stt-2.6b-en: 2.6B, 2.5 s delay. Both give word-level timestamps. An H100 handles 400 real-time streams batched. The Rust server on an L40S handles 64 connections at 3x real time. MLX runs on Mac and iPhone (1B tested on iPhone 16 Pro). Weights are CC-BY 4.0, code MIT/Apache — [kyutai-labs/delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling)
- DSM 2.6B En at 2.5 s scores 8.11 English short-form and 7.72 long-form average WER — [Voxtral Realtime paper](https://arxiv.org/html/2602.11298v2)

**Mistral Voxtral Realtime (Voxtral-Mini-4B-Realtime-2602; launched 4 Feb 2026)**
- 4.4B params (970M causal encoder plus a 3.4B decoder from Ministral 3B), 13 languages, Apache-2.0, served via vLLM WebSocket. Delay is configurable from 80 to 2400 ms in 80 ms steps — [arXiv 2602.11298](https://arxiv.org/html/2602.11298v2)
- WER (En-short / En-long / FLEURS): 240 ms 9.95 / 9.29 / 10.80; 480 ms 8.47 / 7.73 / 8.72; 960 ms 7.94 / 7.13 / 7.70; 2400 ms 7.72 / 6.93 / 6.73. The paper's API baselines: Scribe v2 Realtime 7.33 / 7.43 / 8.34; GPT-4o mini Transcribe 7.93 / 7.97 / 7.95 — [arXiv 2602.11298](https://arxiv.org/html/2602.11298v2)
- Marketing claims "configurable down to sub-200 ms" — [Mistral announcement](https://mistral.ai/news/voxtral-transcribe-2); [eWeek](https://www.eweek.com/news/mistral-ai-voxtral-transcribe-2-launch/). A third party prices the hosted Realtime API at about $0.006/min — [gend.co](https://www.gend.co/blog/voxtral-transcribe-2)

**Moonshine v2 (arXiv 2602.12241, Feb 2026; sliding-window "ergodic" streaming encoder)**
- Tiny 33.6M params at 12.01% Open-ASR average WER, Small 123M at 7.84%, Medium 245M at 6.65% — [arXiv 2602.12241](https://arxiv.org/html/2602.12241v1). By comparison, Whisper large-v3 is 7.44% (vendor docs) — [Moonshine docs](https://www.mintlify.com/moonshine-ai/moonshine/concepts/models)
- Response latency on an Apple M3 CPU via ONNX: v2 Tiny 50 ms, Small 148 ms, Medium 258 ms. Whisper Tiny is 289 ms, Small 1,940 ms and Large v3 11,286 ms. The paper says v2 Medium is "43.7x faster than Whisper Large v3" — [arXiv 2602.12241](https://arxiv.org/html/2602.12241v1)
- A Raspberry Pi "sub-200 ms" claim is unverified (blog only) — [yuv.ai](https://yuv.ai/blog/moonshine)

**Whisper-based streaming**
- whisper_streaming (LocalAgreement-2 policy) has an average 3.3 s latency on English ESIC with an A40 GPU — [Macháček et al. 2023](https://arxiv.org/html/2307.14743)
- In the Microsoft study, faster-whisper Small with 3 s chunks scored 24.74% average WER. Batch Whisper-v3-Turbo scored 7.83% — [arXiv 2604.14493](https://arxiv.org/html/2604.14493v1)
- WhisperLive is "chunked, near-live streaming" rather than native streaming. sherpa-onnx is native (Zipformer, Paraformer) — [awesomeosai comparison](https://awesomeosai.com/self-hosted/speech-to-text). WhisperLiveKit offers local streaming with OpenAI- and Deepgram-compatible APIs — [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit)
- Whisper large-v3-turbo: 7.75% WER, RTFx 216, 809M params (third-party) — [Northflank](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks)

**sherpa-onnx / Vosk**
- sherpa-onnx docs claim all models run in real time on a Raspberry Pi 4. Small streaming Zipformers (English 20M) target Cortex-A7. Prebuilt Android APKs exist. No measured RTF or latency was published in what I found — [sherpa small online models](https://github.com/k2-fsa/sherpa/blob/master/docs/source/onnx/pretrained_models/small-online-models.rst)
- Vosk on LibriSpeech test-clean: Aspire model 11.72% WER, Daanzu model 7.08% (2021) — [alphacephei](https://alphacephei.com/nsh/2021/01/30/wav2letter-rasr.html)

### Inferences
- On an RTX-class GPU or Apple Silicon, Nemotron Speech Streaming at 160–560 ms chunks is the best accuracy/latency trade-off of the open models with independent numbers. Its int8 ONNX CPU build (RTFx about 7) also makes CPU-only deployment plausible.
- Moonshine v2 Small/Medium is the most attractive on-device option (laptop or phone CPU) if about 150–260 ms latency and 6.65–7.84% WER are acceptable.
- Kyutai 1B and Voxtral Realtime are the only open models found with a configurable or semantic end-of-turn signal (Kyutai's semantic VAD) or configurable delay (Voxtral) in one model. Voxtral's 4.4B size likely needs a large GPU.

### Gaps
- No measured RTF found for Nemotron, Kyutai or Voxtral on an RTX 30/40-series consumer GPU, or for any model on Android/iOS (except Kyutai's "tested on iPhone 16 Pro" with no numbers).
- No current numbers found for Distil-Whisper, whisper.cpp streaming (`stream` example) or the sherpa-onnx Zipformer WER/latency trade-off.
- The live Open ASR Leaderboard (Gradio app) could not be scraped, and the leaderboard numbers above come from secondary articles.

## Q3. Behavior on disfluent / stuttered speech, and verbatim modes

### Takeaway
Every evaluated ASR is significantly worse on stuttered speech, with WER roughly 2x or more versus fluent speech. Whisper hallucinates more during long pauses (a direct risk for stuttering blocks) and can turn sound repetitions into wrong words. A 2026 Interspeech study shows autoregressive seq2seq models (Whisper, Canary) tend to produce "intended" text, while CTC models are better at verbatim transcription. Only a few APIs offer explicit verbatim switches, and most cover fillers only, not repetitions.

### Cited Findings
- Mujtaba et al., "Lost in Transcription" (NAACL 2024; APIs tested as of 14 Mar 2024 with default settings) compared Google Cloud STT v1, IBM Watson, Microsoft Azure, Rev.AI, Whisper (`whisper-1` API) and wav2vec 2.0. The FB-N condition is FluencyBank with stuttering removed from the transcript and audio re-synthesized — [ACL Anthology](https://aclanthology.org/2024.naacl-long.269). WER on FluencyBank stuttered (FB-Y) vs non-disfluent (FB-N), read from the paper's Table 2. The PDF text was column-scrambled; I mapped rows by cross-checking the paper's own statements, such as Whisper reaching "6.3%" on FB-N and stuttered WER "about two times" that:
  - Azure 16.8% vs 7.3%; Google 27.5% vs 6.9%; IBM 47.6% vs 16.7%; Rev.AI 26.3% vs 15.9%; Whisper 12.1% vs 6.3%; wav2vec2 38.0% vs 10.1%
  - Synthetic stuttered LibriSpeech vs original: Azure 22.8% vs 3.7%; Google 26.8% vs 7.0%; IBM 56.0% vs 19.1%; Rev.AI 25.4% vs 6.2%; Whisper 19.8% vs 3.1%; wav2vec2 25.9% vs 3.3%
  - Word repetitions, prolongations and interjections increased WER most; inter- and intra-word blocks had less impact. Rev.AI returned blank transcripts on some stuttered clips, and the authors note longer disfluent audio can cause "cessation of live transcription"
- Toyin, Umesh and Aldarmaki, "What Counts as an Error? Dual-Reference Benchmarking for Atypical ASR" (arXiv 2606.31112, submitted 30 Jun 2026, accepted Interspeech 2026) benchmarked 11 models on FluencyBank Timestamped against verbatim and intended references — [arXiv abstract](https://arxiv.org/abs/2606.31112); [PDF](https://arxiv.org/pdf/2606.31112):
  - NVIDIA Canary-1B-v2 is best for intended transcription (isWER 13.85). The NVIDIA CTC model is best for verbatim (vWER 17.2) but ranks 5th for intended. Whisper large-v3 ranks 2nd intended and 3rd verbatim. Rankings flip depending on the reference
  - "Autoregressive sequence-to-sequence style models are found to be better at intended speech recognition while models trained with CTC loss perform better for verbatim speech recognition". This held for NVIDIA models trained on the same data, which suggests the training paradigm alone biases the output style
  - Prolongations are easiest. Multisyllable-unit repetitions and incomplete-syllable repetitions are hardest
  - Example errors: for "i had the th the th th therapy", Canary output "I had the tape, the tape, the tape, the repeat". For "you're you're you're a bully", Whisper output "your your your bullet"
- Mujtaba and Mahapatra (arXiv 2506.00853, Aug 2025): Whisper "occasionally hallucinates... particularly during long conversational pauses". Untrained Whisper-small on the HeardAI stuttered dataset gave 33.9% WER on read speech and 22.2% on spontaneous. Personalized fine-tuning reduced this to 9.39% and 9.81%, vs 16.12% and 18.39% for a generalized fine-tune. The paper cites prior work that Whisper Large's WER on stuttered speech is about 2x fluent — [arXiv 2506.00853](https://arxiv.org/html/2506.00853v2)
- Mujtaba et al. (arXiv 2406.10177, Oct 2024): wav2vec2-base WER on FluencyBank was 42.2% and fell to 25.2% after fine-tuning with disfluency augmentation. FluencyBank has 1,373 utterances (2.21 h) from 12 speakers — [arXiv 2406.10177](https://arxiv.org/html/2406.10177v2)
- StutterZero / StutterFormer (arXiv 2510.18938) report 24% and 28% WER reductions vs Whisper-Medium on FluencyBank (from the search summary; not fully verified) — [arXiv 2510.18938](https://arxiv.org/html/2510.18938v1)
- Koenecke et al. ("Careless Whisper", ACM FAccT 2024, using AphasiaBank, 13,000+ clips): about 1% of Whisper transcriptions contained entirely hallucinated phrases, some harmful. Whisper was more prone than other tools to hallucinate on speech with longer pauses or non-vocal durations, and it hallucinated "Thank you" on a silent file. The authors say the rate has since decreased — [Cornell Chronicle, 11 Jun 2024](https://news.cornell.edu/stories/2024/06/ai-speech-text-can-hallucinate-violent-language)
- Verbatim and filler switches (current docs):
  - Deepgram `filler_words`: default false, fillers only, English, Nova-family — [Deepgram docs](https://developers.deepgram.com/docs/filler-words)
  - ElevenLabs realtime `no_verbatim`: default false, so fillers, false starts and disfluencies are kept by default — [ElevenLabs API ref](https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime)
  - Speechmatics: disfluency tags with optional `remove_disfluencies` (default false) — [Speechmatics docs](https://docs.speechmatics.com/speech-to-text/formatting)
  - Gemini Live: VERBATIM mode (default) keeps fillers, repetitions and false starts — [Gemini docs](https://ai.google.dev/gemini-api/docs/live-api/live-transcribe)
  - AssemblyAI: `disfluencies` for pre-recorded only, as confirmed — [AssemblyAI docs](https://www.assemblyai.com/docs/speech-to-text/pre-recorded-audio/filler-words)
  - xAI: `filler_words` default false — [xAI docs](https://docs.x.ai/developers/model-capabilities/audio/speech-to-text)
  - Azure: no official switch; fillers are reportedly removed — [MS Q&A](https://learn.microsoft.com/en-us/answers/questions/1165008/how-to-config-disfluency-removal-using-rest-api)

### Inferences
- StutterClear's architecture (detect, transcribe, correct) benefits from a verbatim-leaning ASR, so the detector's events align with tokens and the correction LLM decides what to remove. Per the dual-reference study, CTC/transducer-style models are more faithful to what was said. Most low-latency streaming models (Nemotron RNN-T, Zipformer transducers, Parakeet) are in this family.
- Whisper-family models (including chunked faster-whisper and probably OpenAI's Whisper-derived realtime models) risk silently "fixing" or hallucinating over repetitions and blocks. That makes stutter-to-text alignment unreliable and adds hallucination risk during blocks. They are a poor choice for the primary streaming ASR.
- Personalized fine-tuning gives large gains (about 16% to 9.4% WER for Whisper-small). For a local model, a per-user adaptation path (or at least keyterm biasing) is worth planning.
- Vendor "filler word" switches only cover uh and um. None documents preserving part-word repetitions ("th-th-therapy"), so verbatim fidelity for sound repetitions must be measured empirically.

### Gaps
- No published WER of Deepgram Nova-3/Flux, AssemblyAI, Soniox, ElevenLabs, Speechmatics, Cartesia, Nemotron, Moonshine, Kyutai or Voxtral on FluencyBank or SEP-28k was found. All stuttered-speech API numbers are from March 2024 (older model generations).
- The full per-model table of the Interspeech 2026 dual-reference paper could not be cleanly extracted (column-scrambled PDF). Only the values stated in the text are reported.
- A search summary claimed a podcast-based study found sound repetitions trigger Whisper hallucinations "over 20% of the time", but the paper could not be identified or verified.

## Q4. Built-in endpointing / turn detection and configurable silence timeouts

### Takeaway
Most leading APIs now ship semantic or model-based end-of-turn detection with a silence backstop. Several can disable model-based end-of-turn and let the application end turns manually. That suits StutterClear, because stuttering blocks look like silence and an aggressive endpointer will cut users off mid-word.

### Cited Findings
- **Deepgram Flux:**
  - `eot_threshold` 0.5–1.0 (default 0.7). 1.0 suppresses natural EndOfTurn so the app controls turns via `ForceEndTurn`
  - `eager_eot_threshold` 0.3–0.9 (off by default) enables EagerEndOfTurn and TurnResumed events
  - `eot_timeout_ms` 500–60000 (default 5000) forces EndOfTurn after that much silence, and the timer resets on speech. The docs advise increasing it "for users who pause often"
  - Settings can be changed mid-stream via a Configure message — [Deepgram Flux configuration docs](https://developers.deepgram.com/docs/flux/configuration)
  - Conflict: a search summary of the same docs gave eot_timeout_ms max 10000. Telnyx's Flux integration enables eager EOT by default at 0.3 — [Telnyx docs](https://developers.telnyx.com/docs/voice/stt/websocket-streaming/parameters/end-of-turn)
- **AssemblyAI:**
  - `end_of_turn_confidence_threshold` default 0.4; setting it to 1 disables model-based turn detection and leaves silence-based only
  - `min_turn_silence` replaces the deprecated `min_end_of_turn_silence_when_confident` (old default 400 ms)
  - `max_turn_silence` forces end-of-turn, with conflicting documented defaults: 1280 ms (streaming docs), 2400 ms (turn-detection page), and min 100 / max 1000 per LiveKit — [AssemblyAI streaming docs](https://assemblyai.com/docs/streaming); [AssemblyAI turn detection](https://assemblyai.com/docs/speech-to-text/universal-streaming/turn-detection); [LiveKit AssemblyAI plugin](https://docs.livekit.io/agents/models/stt/assemblyai)
  - Configuration can be updated mid-stream — [AssemblyAI mid-stream config](https://www.assemblyai.com/docs/streaming/updating-configuration-mid-stream)
- **Soniox:**
  - Semantic endpointing uses pauses, intonation and context, and emits an `<end>` token
  - `endpoint_latency_adjustment_level` 0–3 (default 0); `endpoint_sensitivity` -1.0 to 1.0 (default 0); `max_endpoint_delay_ms` 500–3000 (default 2000)
  - Endpoint detection reduces diarization accuracy. Manual `finalize` is available — [Soniox endpoint detection docs](https://soniox.com/docs/stt/rt/endpoint-detection); [Soniox RT docs](https://soniox.com/docs/stt/rt/real-time-transcription)
- **Speechmatics:** `end_of_utterance_silence_trigger` 0–2 s (0 disables); `max_delay` 0.7–4 s — [Speechmatics turn detection](https://docs.speechmatics.com/speech-to-text/realtime/turn-detection). The Voice SDK ADAPTIVE mode accounts for disfluencies — [speechmatics-voice](https://pypi.org/project/speechmatics-voice/)
- **ElevenLabs:** commit_strategy manual / vad / turn_prediction, with VAD silence and threshold parameters — [ElevenLabs API ref](https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime)
- **Google Chirp 3:** endpointing_sensitivity standard / short / supershort (third-party profile) — [OpenTranscription](https://opentranscription.io/blog/google-cloud-chirp-3-capabilities-architecture-costs-and-com-profile.html)
- **Azure:** `Speech_SegmentationSilenceTimeoutMs` and a segmentation strategy, with reported quirks — [Azure PropertyId](https://learn.microsoft.com/en-us/javascript/api/microsoft-cognitiveservices-speech-sdk/propertyid); [MS Q&A](https://learn.microsoft.com/en-us/answers/questions/1397127/speech-segmentationsilencetimeoutms-and-speech-seg)
- **Cartesia Ink-2:** built-in turn detection, with no separate VAD needed — [Cartesia Ink](https://www.cartesia.ai/ink)
- **Kyutai stt-1b-en_fr:** semantic VAD built in — [Kyutai DSM repo](https://github.com/kyutai-labs/delayed-streams-modeling)
- Measured end-of-turn latency (Speko p50 / p90): Soniox 489 ms / 2.40 s, AssemblyAI U-3.6 Pro 589 ms / 1.58 s, Deepgram Flux 773 ms / 1.48 s, ElevenLabs 793 ms / 1.06 s, Speechmatics Standard 1.45 s / 1.81 s — [Speko](https://benchmarks.speko.ai/)

### Inferences
- Endpointers tuned for fluent voice-agent turn-taking (500–800 ms) will misfire on stuttering blocks, which can last several seconds. Safer configurations:
  - Flux: `eot_threshold=1.0` plus `ForceEndTurn` driven by StutterClear's own block detector, with `eot_timeout_ms` raised (for example 3–8 s)
  - AssemblyAI: threshold 1 plus a large `max_turn_silence`
  - Soniox: endpoint detection off, using manual `finalize`
  - ElevenLabs: manual commit
  - Speechmatics: trigger 0 and app-side control
- Keep transcription (partials) separate from turn-ending. StutterClear's stutter detector knows a silence is a block, so it should own the end-of-turn decision rather than the ASR vendor.
- Mid-stream reconfiguration (Flux Configure, AssemblyAI update) allows per-user adaptive timeouts, for example longer for users with frequent blocks.

### Gaps
- No vendor documents testing its end-of-turn model on stuttered speech or blocks. No benchmark of false end-of-turn rates on disfluent speech was found.
- OpenAI realtime VAD and semantic-VAD parameters, AWS Transcribe partial-results stabilization, and Gladia endpointing were not verified.

## Q5. Practical recommendation material: lowest time-to-first-partial vs most robust

### Takeaway
For the lowest time-to-first-partial in the cloud: Deepgram Flux (about 20 ms after end of speech on AA; 190 ms live-word p50 on Speko), Azure MAI-Transcribe-2 (241 ms TTFT and 114 ms live words on Speko), and Soniox and Cartesia Ink-2 (AA 0.05–0.07 s). For accuracy plus stability: ElevenLabs Scribe v2 Realtime and Cartesia Ink-2 (AA about 3.6% WER; Ink-2 has 0% rewrites on Speko), and Azure MAI-Transcribe-2 (best Speko WER). For local use: Nemotron Speech Streaming (160–560 ms chunks) on GPU or CPU-ONNX, and Moonshine v2 on-device. Whisper-based streaming should be avoided as the primary engine for stuttered speech.

### Cited Findings
- Fastest partials (independent): Deepgram Flux 0.019 s partial and 0.020 s final from end of speech; Nemotron 80 ms 0.04 s; Soniox about 0.05 s; Deepgram Nova-3 0.057 s; Cartesia Ink-2 0.07 s; ElevenLabs 0.13 s; AssemblyAI U3 RT Pro 0.47 s — [AA article](https://artificialanalysis.ai/articles/new-streaming-speech-to-text-benchmark-aa-wer-streaming)
- Fastest live words and TTFT: Azure MAI-Transcribe-2 (114 ms / 241 ms), Deepgram Flux (190 ms / 641 ms), Speechmatics Standard (403 ms / 690 ms), Soniox (496 ms / 786 ms) — [Speko](https://benchmarks.speko.ai/)
- Most accurate streaming: Cartesia Ink-2 3.59–3.66% and ElevenLabs 3.64% (AA) — [AA article](https://artificialanalysis.ai/articles/new-streaming-speech-to-text-benchmark-aa-wer-streaming). Azure MAI-Transcribe-2 6.2%, AssemblyAI U-3.5/3.6 Pro 7.2%, ElevenLabs 8.2% (Speko) — [Speko](https://benchmarks.speko.ai/)
- Least partial churn: Cartesia Ink-2 and Meta Muse 0% rewrites, Soniox 13%; Deepgram Flux 40%, Gladia 156% — [Speko](https://benchmarks.speko.ai/)
- Cheapest: Soniox about $0.12/hr — [Soniox pricing](https://soniox.com/pricing). AssemblyAI Universal-Streaming $0.15/hr — [AssemblyAI pricing](https://www.assemblyai.com/pricing). Deepgram Nova-3 about $0.29/hr promo — [Deepgram pricing](https://deepgram.com/pricing). Most expensive: OpenAI gpt-realtime-whisper about $1.02/hr — [OpenAI](https://developers.openai.com/api/docs/models/gpt-realtime-whisper)
- Local:
  - Nemotron int8 ONNX CPU RTFx 7.25 at 8.01% average WER (0.56 s delay) — [arXiv 2604.14493](https://arxiv.org/html/2604.14493v1)
  - Moonshine v2 Small 148 ms at 7.84% WER on an M3 — [arXiv 2602.12241](https://arxiv.org/html/2602.12241v1)
  - Kyutai 1B 0.5 s delay with semantic VAD and iPhone MLX — [Kyutai](https://github.com/kyutai-labs/delayed-streams-modeling)
  - Voxtral Realtime 480 ms 8.72% FLEURS — [arXiv 2602.11298](https://arxiv.org/html/2602.11298v2)

### Inferences
- A sensible shortlist for an A/B test on real stuttered audio (FluencyBank, SEP-28k clips plus in-house recordings), scored on both verbatim and intended references:
  1. Deepgram Flux or Nova-3, for speed, `filler_words`, keyterms and manual end-of-turn control
  2. ElevenLabs Scribe v2 Realtime, for accuracy, keyterms, a verbatim default, word logprobs and manual commit
  3. Soniox stt-rt-v5, for low cost, context biasing, manual finalize and low rewrites
  4. Cartesia Ink-2, for accuracy and stable partials
  5. AssemblyAI U-3.6 Pro, for keyterms included and mid-stream config
  6. A local Nemotron Speech Streaming 0.6B baseline, for privacy, offline use and a verbatim-leaning RNN-T
- Metrics should include time-to-first-partial, partial rewrite rate, false end-of-turn during blocks, verbatim WER on repetitions, and hallucination rate during blocks.
- Use partials for the stutter-aware correction stage only when stability is acceptable. High-rewrite engines (Flux 40%) will force the LLM corrector and TTS to re-do work. A hybrid of streaming partials plus a short "stability window" (like LocalAgreement) can trade about 100–300 ms for stability.
- Because the stuttered-speech accuracy evidence is from 2024-era models, the ranking above for the StutterClear population is unproven. An in-house benchmark is the most valuable next step.

### Gaps
- No head-to-head benchmark of these 2026 engines on stuttered speech exists in the sources found.
- Independent benchmarks conflict (for example, ElevenLabs TTFT of 0.13 s on AA vs 2.02 s on Speko and 2.07 s on Gradium; Cartesia WER 3.6% on AA vs 12.6% on Speko). Methodology differences cannot be fully reconciled from public pages.
- No latency numbers for on-device mobile (Android or iOS) streaming ASR were found beyond Moonshine's Apple M3 laptop figures.
