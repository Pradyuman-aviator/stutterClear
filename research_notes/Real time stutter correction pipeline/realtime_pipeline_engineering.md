# Engineering of Ultra-Low-Latency Real-Time Voice Pipelines (cascaded and end-to-end), as of October 2026, for StutterClear

Scope: mic → (AEC/NS) → stutter detection → streaming ASR → LLM text correction → personalized streaming TTS → speaker / virtual mic.
Research date: 2026-10-17. Evidence labels used throughout:
- **[PAPER]** peer-reviewed or arXiv paper by the model authors (self-reported but methodologically described)
- **[VENDOR]** vendor docs, README, or blog (self-reported claim)
- **[MEASURED-3P]** third-party measurement with stated methodology
- **[SECONDARY]** aggregator, news, or blog repeating a claim, unverified against the primary source
- **[ANECDOTAL]** forum or user report

---

## Q1. End-to-end / full-duplex speech-to-speech models: latency, and can they do a "repeat the user's speech fluently in their voice" (rewrite) task? Simultaneous interpretation as a template

### Takeaway
Full-duplex speech LMs reach **~160–250 ms theoretical / ~200 ms practical reply-onset latency** in their papers (Moshi, Qwen3-Omni, LLaMA-Omni). That number measures **reply** latency, not **rewrite lag**, though. The closest published template for StutterClear is Kyutai's **Hibiki / Hibiki-Zero** (streaming speech→speech translation with voice transfer). Its measured lag is **~2.3–2.9 s end offset and 3.4–5.0 s LAAL**, because the model has to wait for enough source context. The hosted realtime APIs, measured independently in July 2026, land at **~1.2–2.3 s** from end of turn to first audio (gpt-realtime family) and **~1.4 s** (Gemini 3.1 Live). They also offer only preset voices, so none of them can speak in the user's own voice off the shelf. I found no off-the-shelf model that does "fluent repeat in the user's voice" in streaming mode. The only end-to-end dysfluent→fluent model I found (StutterZero/StutterFormer, Oct 2025) is offline and does not preserve the speaker's voice.

### Cited Findings
**Kyutai Moshi (Sep 2024)**
- [PAPER] Moshi is "the first real-time full-duplex spoken large language model". It models the user's and its own speech as parallel streams, so there are no explicit speaker turns. Its "Inner Monologue" predicts time-aligned text tokens before audio tokens. Theoretical latency is **160 ms**, and practical latency is **200 ms**. — [Moshi arXiv 2410.00037](https://arxiv.org/abs/2410.00037)
- [VENDOR] Theoretical 160 ms = "80ms for the frame size of Mimi + 80ms of acoustic delay". Practical latency is "as low as 200ms on an L4 GPU". Mimi codec: 24 kHz audio compressed to 12.5 Hz, 1.1 kbps, 80 ms frame. Temporal Transformer is 7B. The PyTorch version needs a ~24 GB GPU (no quantization). An MLX backend runs on-device, tested on a MacBook Pro M3. A Rust/Candle backend is for production. Weights are CC-BY 4.0; code is MIT/Apache. — [kyutai-labs/moshi GitHub](https://github.com/kyutai-labs/moshi)

**Kyutai Hibiki (Feb 2025): streaming speech→speech "rewrite" template**
- [PAPER] Hibiki is a decoder-only multistream model that processes source and target speech synchronously and jointly emits text and audio tokens. It accumulates "just enough context" to translate chunk by chunk. Training uses weakly-supervised "contextual alignment": per-word delays are derived from the perplexity of an off-the-shelf MT system. Inference uses plain temperature sampling, so there is no explicit read/write policy module. — [Hibiki arXiv 2502.03382](https://arxiv.org/abs/2502.03382)
- [PAPER] Alignment method: MADLAD-3B is fed progressively longer source prefixes. The source word that causes the largest jump in a target word's log-likelihood becomes its alignment point. Target speech is constrained to lag the contextual alignment by **at least 2 s**. — [Hibiki full text](https://arxiv.org/html/2502.03382v2)
- [PAPER] Measured latency, short-form CVSS-C Fr→En (End Offset / LAAL, seconds): StreamSpeech 1.6/2.8, Seamless 1.4/2.8, Hibiki-M 2.8/3.5, Hibiki 2.9/3.4. Long-form Audio-NTREX: Seamless 1.6/4.2, Hibiki-M 2.3/5.5, Hibiki 2.7/5.0. Hibiki trades "an average 0.7s of additional lagging" for quality. StreamSpeech and Seamless were run at a 2560 ms chunk size. — [Hibiki full text](https://arxiv.org/html/2502.03382v2)
- [PAPER] Quality: ASR-BLEU short-form is Hibiki 39.2 vs Seamless 37.0 vs StreamSpeech 26.4. Long-form is 27.5 vs 25.4 vs 0.1 (StreamSpeech collapses on long-form). — [Hibiki full text](https://arxiv.org/html/2502.03382v2)
- [PAPER] Voice transfer: speaker cosine similarity is Hibiki 0.41 (short) / 0.48 (long) vs Seamless 0.30/0.43. Human speaker-similarity MOS is 3.43 vs 2.86. It uses classifier-free guidance (γ=3.0) on a "very_good" speaker-similarity label, which needs batch size 2 and is still real-time compatible. — [Hibiki full text](https://arxiv.org/html/2502.03382v2)
- [PAPER] On-device: Hibiki-M (1.7B distilled) "remains faster than real-time along a minute of inference, even with a batch size of 2" on an **iPhone 16 Pro**. Hibiki itself is faster than real-time on an H100 at 320 parallel sequences (160 with CFG). — [Hibiki full text](https://arxiv.org/html/2502.03382v2)
- [PAPER/SECONDARY] **Hibiki-Zero** (Feb 12–13, 2026, ICML 2026): a ~3B model trained without word-level aligned data. Stage 1 uses sentence-aligned data, and stage 2 uses **GRPO reinforcement learning to reduce latency**. It translates Fr/Es/Pt/De→En, supports one speaker and one language per session, and new input languages need <1000 h of speech. License is CC BY-NC-SA 4.0 (non-commercial). — [arXiv 2602.11072](https://arxiv.org/html/2602.11072); [ICML 2026 poster](https://icml.cc/virtual/2026/poster/66094); [MarkTechPost, 2026-02-13](https://www.marktechpost.com/2026/02/13/kyutai-releases-hibiki-zero-a3b-parameter-simultaneous-speech-to-speech-translation-model-using-grpo-reinforcement-learning-without-any-word-level-aligned-data/); [HF weights](https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16)

**NVIDIA PersonaPlex (Jan 2026)**
- [PAPER] A role- and voice-conditioned full-duplex model built on the Moshi architecture. The authors claim it beats prior duplex and cascaded systems on role adherence, speaker similarity, latency and naturalness. — [PersonaPlex paper page (hyper.ai)](https://hyper.ai/en/papers/personaplex)
- [SECONDARY] PersonaPlex-7B-v1 was released around 2026-01-20. Reported figures are "~170 ms" (described as TTFT in one source and as turn-taking latency in another) and "as low as 240 ms" interruption response. These numbers are not confirmed in the primary paper. — [ComfyUI-Wiki release note](https://comfyui-wiki.com/en/news/2026-01-20-nvidia-personaplex-7b-v1-release); [GrowwStacks blog](https://growwstacks.com/blog/nvidia-personaplex-zero-lag-voice-ai)

**Other open speech LMs**
- [PAPER] Qwen3-Omni: theoretical cold-start end-to-end first-packet latency of **234 ms** (audio) and 547 ms (video). The Talker uses multi-codebook codec prediction with a lightweight causal ConvNet instead of block-wise diffusion, so streaming can start from the first codec frame. — [Qwen3-Omni Technical Report arXiv 2509.17765](https://arxiv.org/abs/2509.17765)
- [PAPER] LLaMA-Omni: response latency "as low as 226ms". It combines a speech encoder, adaptor, Llama-3.1-8B-Instruct and a streaming speech decoder, trained on InstructS2S-200K. — [arXiv 2409.06666](https://arxiv.org/abs/2409.06666)
- [PAPER] Freeze-Omni: keeps the text LLM frozen, avoiding catastrophic forgetting, and gets duplex ability through multi-task training. The abstract claims only "low latency" and gives no ms figure. — [arXiv 2411.00774](https://arxiv.org/abs/2411.00774)
- [VENDOR] GLM-4-Voice (9B, on GLM-4-9B): 12.5 Hz speech tokenizer. The streaming decoder can start after as few as **10 speech tokens**, and the model outputs interleaved text and speech, with text guiding the speech. — [THUDM/GLM-4-Voice GitHub](https://github.com/THUDM/GLM-4-Voice)
- [SECONDARY] GPT-4o (May 2024) reference point: audio responses in as little as 232 ms, 320 ms average. — [Kunal Ganglani blog citing OpenAI](https://www.kunalganglani.com/blog/ai-latency-speed-matters-more-than-benchmarks.md)
- [SECONDARY] StepFun StepAudio 3 Realtime averaged **8.83 s** to first audio on one benchmark, while Google/OpenAI/Alibaba/xAI models took ~1–3 s. — [RuntimeWire](https://runtimewire.com/article/stepfun-stepaudio-3-five-audio-models)

**Hosted realtime APIs (OpenAI gpt-realtime, Gemini Live)**
- [MEASURED-3P, July 2026] Time from client-VAD end-of-turn to first audio byte, with 200 ms server silence and 20 questions × 1 run. Means: Gemini 3.1 Live **1,407 ms** (±118); GPT-Realtime **1,731 ms** (±399, range 834–2,783); GPT-Realtime 1.5 **1,869 ms**; GPT-Realtime-Mini **1,758 ms** (max 4,446); GPT-Realtime-2 **2,232 ms**. GPT-Realtime rises from 1,201 ms on easy to 2,308 ms on "very complex" questions, while Gemini stays flat at ~1.37–1.45 s. Caveats: GPT ran in Azure Sweden Central, Gemini's region was auto-routed, and the sample is small. — [Plain Concepts benchmark](https://www.plainconcepts.com/?p=121434)
- [SECONDARY] GPT-Realtime-2 TTFA of 1.12 s at minimal reasoning. The same source says end-to-end models generally reach 160–400 ms vs a 500–650 ms floor for cascades, and warns that TTFA, end-to-end and perceived latency are different metrics. — [SoftwareSeni](https://www.softwareseni.com/voice-agent-latency-solved-enough-for-production-benchmarks-and-architecture-tradeoffs)
- [SECONDARY] Realtime API TTFB is ~500 ms in US regions, and ~800 ms voice-to-voice is "achievable". — [Fora Soft guide 2026](https://www.forasoft.com/blog/article/openai-realtime-api-voice-agent-production-guide-2026)
- [SECONDARY] Full-Duplex-Bench v3 (tool-use tasks): Gemini Live 3.1 first response 3.95 s; GPT Realtime 1.5 6.36 s. A separate article reports "GPT-Live-1" at 0.798 s on Full-Duplex-Bench vs 1.24–1.34 s API TTFA, and warns these measure different things. — [BenchLM aggregator](https://benchlm.ai/voice-benchmarks/full-duplex-bench-v3); [OrcaRouter](https://www.orcarouter.ai/blog/gemini-3-8-live-vs-gpt-live-1)
- [VENDOR] OpenAI Realtime voices are limited to `alloy, ash, ballad, coral, echo, sage, shimmer, verse, marin, cedar` ("marin or cedar" recommended). "Once the model has emitted audio in a session, the voice cannot be modified." The page does not mention custom or cloned voices. Audio is PCM16 24 kHz mono, with G.711 µ-law output also shown. — [OpenAI Realtime conversations guide](https://developers.openai.com/api/docs/guides/realtime-conversations)
- [ANECDOTAL, late 2025] Gemini Live native-audio users report responses of 3–7 s, and 10–15 s on Vertex us-central1, plus latency drifting upward during long sessions until the session is restarted. — [Google AI forum: latency drift](https://discuss.ai.google.dev/t/gemini-live-api-native-audio-response-latency-gradually-increases-during-long-sessions/115508); [high latency thread](https://discuss.ai.google.dev/t/gemini-live-api-models-high-latency/108989); [2.5 Flash native audio delay](https://discuss.ai.google.dev/t/significant-delay-with-gemini-live-2-5-flash-native-audio/122650)
- [VENDOR] Gemini 3.1 Flash Live Preview (Mar 2026) defaults to the "minimal" thinking level to minimize latency, and gives no ms figure. Live API output is raw 16-bit PCM at 24 kHz. — [Gemini 3.1 Flash Live model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-live-preview); [Vertex Live API overview](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api)

**Direct dysfluent→fluent speech conversion**
- [PAPER, Oct 2025] StutterZero (conv-BiLSTM) and StutterFormer (Transformer) are presented as the first end-to-end models converting stuttered speech to fluent speech while jointly transcribing. Input is log-Mel, and output is vocoded with Griffin-Lim. The models are **not streaming** (listed as future work), report **no latency**, and **do not preserve speaker identity**: targets were MeloTTS-synthesized, which caused prosodic mismatch. Validation WER is 0.116 (StutterZero) and 0.08 (StutterFormer) vs Whisper-Medium 0.361. FluencyBank WER is 0.161 and 0.120. Trained on SEP-28K (~14 h after cleaning) and LibriStutter (~20 h synthetic). The abstract's WER-reduction percentages do not match its own tables. — [arXiv 2510.18938](https://arxiv.org/html/2510.18938v1)

### Inferences
- "~200 ms" full-duplex figures describe **response onset** for a *reply*. For a **rewrite**, latency is bounded by how much future context the system needs before committing each output word. Hibiki shows that even a purpose-built streaming S2S model sits at multi-second lag for translation. Disfluency removal is mostly monotonic (delete repetitions, prolongations and fillers; rarely reorder), so it should need much less look-ahead than Fr→En translation. A 1–2 s ear-voice lag is a plausible target, not a 200 ms one.
- Hibiki's recipe maps directly onto StutterClear as a long-term "Path B". Build parallel (dysfluent source audio, fluent target audio in the same voice) pairs: target timing from a contextual-alignment rule (e.g., emit a word only once the dysfluent span containing it has ended), target audio from a voice-cloned TTS of the cleaned transcript. Then fine-tune a Moshi/Hibiki-style multistream model. Hibiki-Zero's GRPO stage is a published method for pushing lag down without word-level alignments. Licensing is a blocker for commercial use: Hibiki/Hibiki-Zero weights are CC BY-NC-SA, while Moshi weights are CC-BY 4.0.
- Prompting gpt-realtime or Gemini Live to "repeat what I said, fluently" is a viable prototype. Its drawbacks: (a) these models are tuned to *reply* and may answer instead of repeat; (b) output is a preset voice, so per-user voice needs a separate VC/TTS stage; (c) measured end-of-turn→first-audio is already 1.2–2.3 s before client I/O. A tuned cascade can match or beat that, and it keeps control over voice and text edits.
- PersonaPlex (voice-conditioned, Moshi-based) is the most relevant open full-duplex base for a "speak in a given voice" rewrite model. Its exact latency and license should be checked on NVIDIA's model card.

### Gaps
- No primary latency numbers found for Sesame (CSM/conversational demo), Step-Audio 2, Mini-Omni, SALMONN-omni or Freeze-Omni (its full paper was not read). TEN Framework's own E2E stack was not investigated.
- No public benchmark measures any E2E model on a "repeat fluently" (same-language rewrite) task.
- OpenAI custom-voice availability for Realtime (if any, e.g., enterprise programs) was not confirmed. The docs page fetched lists only presets.
- I could not verify Hibiki-Zero's lag numbers (the paper's tables were not read). Only the GRPO latency-reduction claim is confirmed.

---

## Q2. Voice agent frameworks (Pipecat, LiveKit Agents, OpenAI Agents SDK, Vocode, TEN, Daily): streaming structure, barge-in, cancellation of in-flight TTS, frame/queue design, published latency budgets

### Takeaway
The mature pattern, used most completely by Pipecat and LiveKit, has four parts. (1) A **two-lane frame queue**: high-priority system frames (audio in, interruption) bypass queued data frames. (2) An **interruption broadcast** that cancels the in-flight LLM stream, clears TTS text aggregation, and **drains the output transport's unplayed audio queue**. (3) **Committing only words actually played** (word timestamps) to context. (4) **Speculative/preemptive LLM generation** before end-of-turn is confirmed. Pipecat's published voice-to-voice budget is **~1.2–1.3 s** typical and **~500 ms** best case co-located. These are exactly the mechanisms StutterClear needs for "never play stale audio".

### Cited Findings
**Pipecat / Daily**
- [VENDOR] Frame classes: `SystemFrame`s (e.g., `InputAudioRawFrame`, `UserStartedSpeakingFrame`, `InterruptionFrame`) are "High-priority and ordered with other SystemFrames; interruptions do not discard them". `DataFrame`s (`OutputAudioRawFrame`, `TextFrame`, `TranscriptionFrame`, `LLMTextFrame`) and `ControlFrame`s (`EndFrame`, `TTSStartedFrame`) are "Queued and processed in order" in a separate non-system lane. Ordering is guaranteed within each lane. — [Pipecat pipeline docs](https://docs.pipecat.ai/guides/learn/pipeline)
- [VENDOR] Interruption flow: the user aggregator broadcasts `InterruptionFrame` upstream and downstream. Each processor cancels its current task and discards queued Data/Control frames, except frames with `interruptible=False` (e.g., `FunctionCallResultFrame`, `EndFrame`). The LLM completion is cancelled mid-stream. TTS stops, and text aggregation, word timestamps and pending output are cleared. The output transport "drains its audio queue, discarding audio that was generated but not yet played". "Text that never played never reaches the assistant context aggregator." — [Pipecat interruptions docs](https://docs.pipecat.ai/pipecat/fundamentals/interruptions)
- [VENDOR] Barge-in filters: `MinWordsUserTurnStartStrategy(min_words=3)` stops short backchannels ("okay", "yeah") from interrupting. A Krisp VIVA interruption-prediction strategy separates real interruptions from acknowledgments. Mute strategies (`AlwaysUserMuteStrategy`, `FirstSpeechUserMuteStrategy`) block user audio while the bot speaks. With speech-to-speech services (OpenAI Realtime, Gemini Live) the provider detects barge-in and Pipecat still flushes local audio. — [Pipecat interruptions docs](https://docs.pipecat.ai/pipecat/fundamentals/interruptions)
- [VENDOR, guide updated June 2026] Pipecat's voice-to-voice budget on macOS client to cloud (ms): mic input 40, Opus encode 21, network 10, packet handling 2, jitter buffer 40, Opus decode 1, **transcription+endpointing 300**, **LLM TTFB 650**, sentence aggregation 20, **TTS TTFB 120**, Opus encode 21, packet 2, network 10, jitter 40, decode 1, speaker output 15. That sums to **1,293 ms** (the text says "about 1,200"). Target: "1,500 ms voice-to-voice latency is an important target". Co-located models on one GPU cluster reach "as low as 500 ms". Rule of thumb: LLM TTFT ≤600 ms. — [Voice AI & Voice Agents guide (Pipecat/Daily)](https://voiceaiandvoiceagents.com/)
- [VENDOR] Network round-trip to US-hosted APIs: "From Europe… ~250 ms; from India, ~350 ms." — [Voice AI & Voice Agents guide](https://voiceaiandvoiceagents.com/)
- [VENDOR] Interruption best practice: every stage must be cancellable and client playout must stop quickly. With faster-than-realtime audio APIs, "you must stop playout and flush buffers yourself". Spurious interruptions come from transient noise, AEC failures and background speech. Mitigations: VAD start length/confidence, exponential smoothing of volume, speaker isolation, and word-level timestamps to record only what the user heard. — [Voice AI & Voice Agents guide](https://voiceaiandvoiceagents.com/)

**LiveKit Agents**
- [VENDOR] Turn detection modes: turn-detector model (default), realtime-model server-side, VAD-only (`turn_detection="vad"`, "minimal latency"), STT endpointing (`"stt"`, e.g., AssemblyAI, Deepgram Flux), and **manual (push-to-talk)**. Interruption options: `enabled`, `min_duration`, `min_words` (STT only), `false_interruption_timeout`, `resume_false_interruption` (default: the agent "resumes speaking from where it left off after a false interruption"), and `discard_audio_if_uninterruptible`. "Adaptive" interruption handling is the default mode on LiveKit Cloud with most STT providers. — [LiveKit turns docs](https://docs.livekit.io/agents/build/turns/)
- [VENDOR] Preemptive generation (default on) speculatively starts the **LLM** before end-of-turn is confirmed, and TTS waits unless `preemptive_tts` is enabled. Speculative output is regenerated if context or tools change, which costs extra tokens. The docs say it "doesn't guarantee reduced latency" and that it is less favorable for long utterances like dictation. — [LiveKit audio docs](https://docs.livekit.io/agents/multimodality/audio.md)
- [ANECDOTAL] Users report 2–3 s delays after short utterances despite preemptive generation (Agents 1.6.4). Preemptive generation never triggers with STT plugins that lack a "preflight" transcript event. — [LiveKit community thread](https://community.livekit.io/t/high-response-latency-after-short-user-utterances-despite-preemptive-generation-and-aggressive-endpointing-agents-1-6-4/1658); [livekit/agents issue #4317](https://github.com/livekit/agents/issues/4317)

**OpenAI Realtime API and Agents SDK**
- [VENDOR] On server-VAD barge-in, "The server will automatically cancel any in-progress model response". Over **WebRTC/SIP** "the server manages a buffer of output audio, and thus knows how much audio has been played" and truncates automatically. Over **WebSocket** "the client manages audio playback, and thus must stop playback and handle truncation" by sending `conversation.item.truncate` with `audio_end_ms`, which also removes the transcript of the unplayed part. Push-to-talk uses `output_audio_buffer.clear`. WebRTC is recommended for client devices because it is "more robust… over uncertain network conditions". — [OpenAI Realtime conversations guide](https://developers.openai.com/api/docs/guides/realtime-conversations)
- [VENDOR] OpenAI Agents SDK `VoicePipeline` is a cascade (STT → your workflow → TTS) that streams audio events out. "The Agents SDK currently does not provide any built-in interruption handling for StreamedAudioInput". The suggested pattern is to mute the mic on `turn_started` and unmute after playout. — [OpenAI Agents SDK voice pipeline docs](https://openai.github.io/openai-agents-python/voice/pipeline/)

### Inferences
- For StutterClear, "never play stale audio" maps directly onto the Pipecat/OpenAI pattern. Tag every LLM/TTS request with a **generation/epoch ID**. When the user restarts or changes an utterance, broadcast an interrupt that (a) cancels the LLM stream, (b) cancels the TTS request (most cloud TTS have a cancel/clear message), and (c) **drops every queued audio chunk whose epoch ≠ current**, both in the app and in any device or virtual-cable buffer. Playout buffers should stay small (≤40–60 ms) so a flush takes effect almost immediately.
- Unlike an agent, StutterClear's "barge-in" is the *same* speaker continuing or correcting themselves. The interruption trigger should therefore be semantic (the ASR hypothesis for the not-yet-spoken span changed) rather than "user started speaking". A Pipecat-style `min_words` filter or a stutter-aware filter is needed so a repetition ("I-I-I") doesn't count as a restart.
- WebSocket transport puts the truncation burden on the client. If cloud components are used, WebRTC (or local processing) makes flush semantics simpler.
- LiveKit's `resume_false_interruption` corresponds to resuming playback after a stutter block that was misread as a stop. StutterClear needs the same mechanism.

### Gaps
- Vocode and TEN Framework: no primary docs on their interruption/queue design or latency budgets were retrieved.
- No published latency budget specifically for Daily's transport (as distinct from Pipecat's guide) was found.

---

## Q3. VAD and endpointing / turn detection: frame sizes, latency, failure modes for people who stutter, adaptive silence thresholds

### Takeaway
Frame-level VADs are cheap: Silero processes a 30+ ms chunk in <1 ms; TEN VAD uses 10/16 ms hops. Their weakness is the **silence timeout** (typically 0.3–0.8 s) used to declare end-of-turn. Current best practice is a **short VAD trigger (~200 ms) + an audio-native turn classifier** (Pipecat Smart Turn v3: 8 MB, ~12–60 ms on CPU; LiveKit audio turn detector), giving turn decisions ~250 ms after a pause. **For people who stutter (PWS), stock endpointers cut off 23.8% of utterances.** Blocks (often silent) are the main cause. Raising thresholds pushed median endpoint delay from 450 ms to 1.7–3.1 s (Apple, CHI 2023). So StutterClear's stutter detector must feed the endpointer ("in a block → don't end turn"), rather than relying on longer timeouts alone.

### Cited Findings
- [VENDOR] Silero VAD: "One audio chunk (30+ ms) takes less than 1ms to be processed on a single CPU thread". Supports 8 and 16 kHz. JIT model is ~2 MB. ONNX may be 4–5x faster. MIT license (the page shows an inconsistent CC BY-NC badge). — [snakers4/silero-vad](https://github.com/snakers4/silero-vad)
- [VENDOR] Silero "normally takes less than 1/8th of a typical virtual machine CPU core". Pipecat's VAD-alone defaults are `VAD_STOP_SECS=0.8`, `VAD_START_SECS=0.2`, `VAD_CONFIDENCE=0.7`, `VAD_MIN_VOLUME=0.6`. Current best practice is VAD with a 200 ms trigger, then Smart Turn (~30 ms on a typical cloud vCPU), then optional LLM single-token tagging (respond now / wait ~5 s / wait ~10 s). Turn decisions take "~250ms from the time a user pauses". — [Voice AI & Voice Agents guide](https://voiceaiandvoiceagents.com/)
- [VENDOR, 2025-09-11] Smart Turn v3: 8 MB, 8M parameters, Whisper-Tiny encoder only, audio windows up to 8 s, 23 languages. CPU latency including preprocessing: AWS c7a.2xlarge 12.6 ms, c8g.2xlarge 15.2 ms, t3.2xlarge 33.8 ms, c8g.medium 59.8 ms, t3.medium 94.8 ms. GPU 3.3–6.6 ms. Fully open (weights, data, training script). Meant to be used with a VAD like Silero. — [Daily blog: Smart Turn v3](https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/)
- [VENDOR/SECONDARY] Smart Turn v3.1 int8 CPU runs in ~12 ms best case and ~70 ms on Pipecat Cloud. A community v3.2 ONNX int8 measured 37.3 ms mean / 50.6 ms p95 per 8 s window on an Apple M5 Pro. — [Daily blog v3.1](https://www.daily.co/blog/improved-accuracy-in-smart-turn-v3-1/); [pipecat-ai/smart-turn](https://github.com/pipecat-ai/smart-turn); [Smart-Turn-v3.2 ONNX (community)](https://huggingface.co/soniqo/Smart-Turn-v3.2-ONNX)
- [VENDOR] LiveKit audio turn detector (v1 cloud; v1-mini local CPU; 14 languages) uses acoustic and semantic cues and requires VAD with `min_silence_duration` ≥0.25 s (Silero default 0.55 s). Endpointing defaults: min/max delay 0.5/3.0 s without the audio detector and **0.3/2.5 s with it**. The deprecated text detector (Qwen2.5-0.5B, 396 MB) runs "~50-160 ms" per turn on CPU in <500 MB RAM. **Dynamic endpointing adapts the delay between min and max using session pause statistics (EMA, alpha 0.9).** — [LiveKit turn detector docs](https://docs.livekit.io/agents/build/turns/turn-detector/)
- [VENDOR] OpenAI Realtime `server_vad` example config is threshold 0.5, `prefix_padding_ms` 300, `silence_duration_ms` 500 (shown as an example, not labeled as defaults). `semantic_vad` eagerness is low/medium/high/auto: "low will let the user take their time to speak". — [OpenAI Realtime VAD guide](https://developers.openai.com/api/docs/guides/realtime-vad)
- [VENDOR] TEN VAD: 16 kHz, hop 160/256 samples = **10/16 ms**. Self-reported RTF is ~0.009–0.016 across CPUs, with a 306 KB library. Claims Silero has "several hundred milliseconds" of delay on speech→non-speech transitions (a vendor claim shown only in a figure). — [TEN-framework/ten-vad](https://github.com/TEN-framework/ten-vad); [HF commit notes](https://huggingface.co/TEN-framework/ten-vad/commit/49b0656a9d214ce7a761dfd53d46705cd6977de5)
- [SECONDARY] License caveat: the TEN-VAD ONNX model reportedly carries a non-compete clause despite the Apache-2.0 label. — [wavekat-vad crate docs](https://docs.rs/wavekat-vad)
- [PAPER, CHI 2023, Apple] With a general-population endpointer tuned for 3% truncation, **23.8%** of PWS utterances were truncated, and >50% for 7 of 41 participants. Truncation correlated most with **blocks** (r=.64), then interjections (r=.48) and part-word repetitions (r=.44). Prolongations and whole-word repetitions were not significant. 77.2% of surveyed PWS who had been cut off blamed a pause or block. — [Lea et al., arXiv 2302.09044](https://ar5iv.labs.arxiv.org/html/2302.09044)
- [PAPER] Raising the endpointer threshold to mild/moderate/severe profiles cut truncation to 4.9% / 2.5% / 0.8%, a 79.1% relative drop for mild. **Median endpoint delay rose from 450 ms to 1,670 / 2,190 / 3,100 ms.** Recommendations: opt-in endpointer profiles, more flexible inter-word pause, and **push-to-talk / "Hold to Speak"**, which disables the endpointer. ASR changes cut WER from 25.4% to 9.9% (tuned decoder + dysfluency refinement). — [Lea et al.](https://ar5iv.labs.arxiv.org/html/2302.09044); [arXiv abstract](https://arxiv.org/abs/2302.09044v3)

### Inferences
- Silence-timeout endpointing is fundamentally at odds with blocks. Apple's numbers show that buying robustness with timeouts costs 1.2–2.7 s of extra median delay. StutterClear should treat "end of phrase" as a fused decision: VAD silence + Smart-Turn-style completeness + **stutter-detector state** (inside a block, a repetition loop, or an elevated-tension prolongation means "not done"). LiveKit-style **per-user adaptive (EMA) pause statistics** should be learned per user, along with an opt-in severity profile and a push-to-talk fallback.
- Smart Turn v3 and LiveKit's detectors were trained on general-population speech. Running them as-is on stuttered audio will probably over-predict "complete" during blocks. Fine-tuning Smart Turn (open weights, data and training script) on PWS audio with block labels (e.g., SEP-28K/FluencyBank-style) is a realistic, cheap step.
- Use 10–16 ms VAD hops (TEN/WebRTC-style) for barge-in/restart detection, where onset latency matters, and the heavier turn model only at candidate pauses.

### Gaps
- Picovoice Cobra and WebRTC VAD frame and latency specs were not retrieved from primary sources (WebRTC VAD is commonly 10/20/30 ms frames, but I did not verify this here).
- No published evaluation of Smart Turn, LiveKit or OpenAI semantic VAD on stuttered speech was found.

---

## Q4. Real-time noise suppression and echo cancellation: algorithmic latency and CPU cost

### Takeaway
Classical and lightweight DNN suppressors add **~10–40 ms** of algorithmic delay. RNNoise uses 20 ms windows / 10 ms hop at ~1.3% of one x86 core. DeepFilterNet2 has **40 ms** algorithmic delay at RTF 0.04 on a laptop i5 and runs on a Raspberry Pi 4. Krisp self-reports ~15 ms. **AEC is mandatory** whenever the device speaker plays output (or the remote party's voice) while the mic is open. Its hard part is delay estimation (echo path 20–200 ms on phones), not added latency.

### Cited Findings
- [PAPER] RNNoise: "20 ms windows with 50% overlap (10 ms offset)", 48 kHz, 22 bands. "around 1.3% of a single x86 core" (non-vectorized C) and 14% of a 1.2 GHz Cortex-A53. ~40 Mflops total. "latency is low enough for use in video-conferencing systems" (no exact delay stated). — [Valin, arXiv 1709.08243](https://ar5iv.labs.arxiv.org/html/1709.08243)
- [PAPER] DeepFilterNet2: "an overall algorithmic delay of 40 ms" (20 ms window, 50% overlap, 2-frame look-ahead). RTF 0.04 on a notebook Core i5-8250U and 0.42 on a Raspberry Pi 4. PESQ 3.08 (VoiceBank+DEMAND) vs DeepFilterNet v1 2.81, PercepNet 2.73, RNNoise 2.33. — [DeepFilterNet2, arXiv 2205.05474](https://ar5iv.labs.arxiv.org/html/2205.05474)
- [PAPER] DeepFilterNet3 (2023): RTF 0.19 on a single-threaded notebook CPU. — [arXiv 2305.08227](https://arxiv.org/abs/2305.08227)
- [VENDOR] DeepFilterNet ships a real-time LADSPA plugin for "real-time noise reduction on your mic" (PipeWire), supports 48 kHz only, and has a `--compensate-delay` option for STFT and look-ahead delay. — [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
- [VENDOR, 2020] Krisp SDK: "Minimal latency impact (15ms)", working on 10/20/30 ms frames. A 2019 GTC deck lists a 15 ms DNN algorithmic component. — [Krisp SDK sheet (archived)](https://webcf.waybackmachine.org/web/20200808123722/https://krisp.ai/KrispSDK.pdf); [NVIDIA GTC 2019 S9222](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9222-revolutionary-voice-enhancement-in-real-time-communications-with-gpu.pdf)
- [SECONDARY, 2026] NVIDIA Maxine has been rebranded "NVIDIA AI for Media". Its Audio Effects cover noise removal, room-echo removal and AEC on NVIDIA GPUs. A ~10–20 ms per-effect figure comes from a voice-tool vendor and is unverified. — [Fora Soft build-vs-buy (2026-06-02)](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/krisp-maxine-dolby-build-vs-buy); [VoxBooster](https://voxbooster.com/blog/voice-changer-nvidia-maxine)
- [VENDOR/SECONDARY] WebRTC AEC3 starts with an `EchoPathDelayEstimator` (render vs capture alignment, with a confidence value). If the delay lock is wrong, later stages fail. Phone echo-path delay is ~20–200 ms and can drift mid-call. A Rust port (`aec3` crate) supports asynchronous render/capture feeds and an initial delay hint. Legacy WebRTC AEC comments warn that OS-reported latency is unreliable across hardware. — [WebRTC AEC3 source](https://webrtc.googlesource.com/src/webrtc/+/36189cde64e4cd9fc90ab3a6f45d9c37efaf625c/modules/audio_processing/aec3/echo_path_delay_estimator.h); [Fora Soft AEC article](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/echo-cancellation-aec-ai-hybrid-webrtc); [aec3 crate](https://docs.rs/crate/aec3/0.1.8); [legacy AEC source](https://webrtc.googlesource.com/src.git/+/f1e97b9/modules/audio_processing/aec/echo_cancellation.cc)
- [VENDOR] AEC failures are one of three main sources of spurious barge-ins in voice agents. — [Voice AI & Voice Agents guide](https://voiceaiandvoiceagents.com/)

### Inferences
- **Meeting-app scenario (virtual mic):** StutterClear captures the *raw* physical mic, before Zoom/Meet's own AEC. So the remote participants' voices coming out of the laptop speakers will leak into StutterClear's ASR unless StutterClear runs its **own AEC with a system-output loopback reference** (e.g., WASAPI loopback capture on Windows). Headphones remove most of this problem and should be the recommended default.
- **In-person scenario (device speaker plays corrected speech):** StutterClear's own output re-enters the mic. Without AEC this triggers self-interruption ("user restarted") and corrupts ASR. The TTS output stream is the AEC reference, and AEC3's delay estimator must lock onto the playout path, including Bluetooth delays.
- Budget **~10–20 ms** for RNNoise/Krisp-class suppression and **~40 ms** for DeepFilterNet2/3. Suppression should run *before* VAD and ASR to cut false onsets, but stutter detection may be better on less-processed audio, since DNN suppressors can smear breathy blocks and prolongations. This is an untested hypothesis and is worth A/B testing.

### Gaps
- No primary algorithmic-latency figure for WebRTC AEC3 or NS. Its 10 ms frame processing is commonly cited but was not verified here.
- No primary NVIDIA figure for AI-for-Media/Maxine audio latency was found.
- No data on how noise suppression affects stuttering-event detection accuracy.

---

## Q5. Low-latency LLM for text correction: local small models, hosted fast inference, speculative/incremental prompting, prompt caching, constrained edits

### Takeaway
Measured hosted-frontier TTFT medians are **~540–740 ms** (P95 1.1–1.8 s). Fast-silicon hosts mainly improve *throughput* (Groq Llama-3.1-8B ~630 tok/s; Cerebras reportedly ~1,800–2,300 tok/s). Their published TTFTs (~0.7–1.0 s) are measured at 10k-token prompts, so they overstate short-prompt TTFT. On-device ~3B models prefill at **~0.6 ms per prompt token** (Apple, iPhone 15 Pro). A ~300-token correction prompt therefore gives ~180 ms TTFT locally, and a 0.5B model on CPU does a forward pass in ~50–160 ms. Disfluency correction is a short-context, near-copy task, which suits a small local model with constrained/diff-style output and streaming-aware look-ahead.

### Cited Findings
- [MEASURED by Pipecat/Daily, guide updated June 2026] LLM TTFT median / P95 (ms): GPT-4.1 536/1771; Gemini 2.5 Flash 597/1137; Claude 4.5 Haiku 637/1615; GPT-5.1 739/1492; Nemotron 3 Ultra self-hosted 541/712. "LLM time-to-first-token of 600 ms or less is fast enough for most voice AI use cases." — [Voice AI & Voice Agents guide](https://voiceaiandvoiceagents.com/)
- [MEASURED-3P, Artificial Analysis, P50 over 72 h, **10,000-input-token workload**, page updated 2026-09] Llama 3.1 8B: Groq 630 tok/s, TTFT 0.96 s; CoreWeave 142 tok/s, 0.66 s; Novita 148 tok/s, 0.97 s; DeepInfra ~31 tok/s, ~0.9–1.0 s. Cerebras was not listed on that page. — [Artificial Analysis Llama 3.1 8B providers](https://artificialanalysis.ai/models/llama-3-1-instruct-8b/providers)
- [SECONDARY] Cerebras Llama 3.1 8B is ~1,800 tok/s (2.4x Groq) per one comparison and ~2,337 tok/s per another, citing AA 2025 data. — [DeployBase](https://deploybase.ai/articles/groq-vs-cerebras-pricing-speed-and-benchmark-comparison); [MorphLLM](https://www.morphllm.com/cerebras-pricing)
- [VENDOR, June 2024] Apple's ~3B on-device model on iPhone 15 Pro: "time-to-first-token latency of about 0.6 millisecond per prompt token" and "a generation rate of 30 tokens per second" before token speculation. 2/4-bit palettization averages 3.7 bpw. LoRA-style adapters are "10s of megabytes". — [Apple ML Research](https://machinelearning.apple.com/research/introducing-apple-foundation-models)
- [VENDOR] LiveKit's text turn detector (Qwen2.5-0.5B-Instruct base, 396 MB) runs "~50-160 ms" per turn on CPU with <500 MB RAM. This is a datapoint for a 0.5B model making a short-context decision on CPU. — [LiveKit turn detector docs](https://docs.livekit.io/agents/build/turns/turn-detector/)
- [VENDOR] Speculative generation: LiveKit starts the LLM before end-of-turn is confirmed and regenerates if context changes, which wastes tokens. It is not guaranteed to cut latency. — [LiveKit audio docs](https://docs.livekit.io/agents/multimodality/audio.md)
- [PAPER, NAACL 2022] "Teaching BERT to Wait": a streaming disfluency tagger trained to choose per token between *emit now* and *wait for more right-context*. Against fixed-look-ahead baselines it "always achieves lower time-to-detection" with "significantly lower waiting time and comparable accuracy and stability" (less flicker). Zero-look-ahead baselines detect slightly sooner but less precisely. — [Chen et al., NAACL 2022](https://aclanthology.org/2022.naacl-main.60)
- [PAPER] Apple's PWS study got large ASR gains (WER 25.4%→9.9%) from decoder tuning plus "dysfluency refinement", a post-ASR text clean-up. This supports a light text-correction stage over a heavy generative rewrite. — [Lea et al.](https://ar5iv.labs.arxiv.org/html/2302.09044)

### Inferences
- **Recommended LLM pattern for StutterClear.**
  1. Run a cheap, deterministic **disfluency tagger** (BERT-small-class, streaming with learned wait), or rules over ASR + stutter-detector timestamps, for repetitions, fillers and prolongations. This handles most edits in a few ms.
  2. Invoke a small LLM (0.5–4B, local) **only for context-aware fixes** such as ambiguous restarts ("I want the— I need the blue one") or ASR errors.
  3. Make the LLM **output edit operations or the cleaned span only**, with a keep/delete mask over ASR tokens. Output tokens drive latency at 30 tok/s on-device (~33 ms/token), so a 15-token sentence regenerated in full costs ~0.5 s, while a 3-token diff costs ~0.1 s.
  4. Keep a **static system prompt + per-user vocabulary prefix** in a reused KV cache (prompt/prefix caching) so each call only prefills the new ASR delta.
- With hosted LLMs, network RTT dominates TTFT for users far from US regions (India→US ~350 ms RTT per Pipecat). That argues for an on-device or in-region LLM for this stage.
- Disfluency correction rarely needs a frontier model. A cloud model at ~600 ms median / ~1.5 s P95 would by itself consume most of a 1–1.5 s budget, and the P95 tail is what causes noticeable stalls in live conversation.

### Gaps
- No published short-prompt (<500-token) TTFT numbers for Groq/Cerebras, or local llama.cpp/MLX TTFT for Qwen3 0.6–4B, Llama 3.2 1B/3B, Gemma 3, or Phi-4-mini. These need in-house benchmarking (`llama-bench` prompt-processing t/s → TTFT ≈ prompt_len / pp_speed + overhead).
- No published latency/accuracy data on LLM-based disfluency correction for stuttered (as opposed to conversational-filler) speech.

---

## Q6. Audio I/O and transport latency: OS buffers (WASAPI, CoreAudio, AAudio/Oboe), WebRTC vs WebSocket, Opus, jitter buffers, virtual audio cables

### Takeaway
On a local device the I/O stack adds **~10–40 ms per direction**: a WASAPI shared-mode period is typically 10 ms; Android best-practice round trip is ~20 ms; Pipecat budgets 40 ms mic-in / 15 ms speaker-out on macOS. A network hop adds Opus (~21–26.5 ms) + jitter buffer (~40 ms) + RTT (10 ms in-region, ~250–350 ms intercontinental round-trip). Virtual cables add **0 ms (BlackHole) to ≤~70 ms (VB-CABLE default buffer)**. The meeting app then adds its own one-way path (G.114 target ≤150 ms), which StutterClear cannot control.

### Cited Findings
- [ANECDOTAL/VENDOR] WASAPI `IAudioClient3::GetSharedModeEnginePeriod` often reports **10 ms** (480 samples @48 kHz) as min = default = max. Real-time shared mode has been reported at ~2–3 ms on good drivers. Microsoft docs say the low-latency shared mode uses a pull-mode audio engine with "a significant reduction in latency". PortAudio maintainers note IAudioClient3 is not strictly required. — [JUCE forum](https://forum.juce.com/t/windows-audio-low-latency-mode/42085); [JUCE issue #560](https://github.com/WeAreROLI/JUCE/issues/560); [PortAudio issue #385](https://github.com/PortAudio/portaudio/issues/385); [Microsoft user-mode audio components](https://learn.microsoft.com/en-ie/Windows/Win32/coreaudio/user-mode-audio-components)
- [VENDOR] Android Oboe/AAudio round-trip (OboeTester): ~**20 ms** with all recommendations; ~21 ms MMAP without callback; ~26 ms without exclusive mode; ~**205 ms** if low-latency performance mode is off. Exclusive MMAP may not be granted. — [Android Oboe low-latency guide](https://developer.android.com/games/sdk/oboe/low-latency-audio)
- [ANECDOTAL, old] macOS built-in mic/speaker round-trip ran ~33 ms at a 32-sample buffer up to ~122 ms at 2048 samples (macOS 10.11). — [Cycling '74 forum](https://cycling74.com/forums/max-coreaudio-round-trip-latency)
- [VENDOR] Pipecat's budget: macOS mic input 40 ms, speaker output 15 ms, Opus encode 21 ms, decode 1 ms, jitter buffer 40 ms per direction, network 10 ms. — [Voice AI & Voice Agents guide](https://voiceaiandvoiceagents.com/)
- [SECONDARY] Opus default algorithmic delay is **26.5 ms** with libopus's default 20 ms frames, reducible to ~5 ms at a quality/bitrate cost. Frame sizes range from 2.5 to 60 ms (RFC 6716). — [Wikipedia: Opus](https://en.wikipedia.org/wiki/Opus_(audio_format)); [FFmpeg trac discussion](https://ffmpeg.org/pipermail/ffmpeg-trac/2023-April/065707.html)
- [VENDOR] OpenAI recommends WebRTC over WebSocket for client devices on uncertain networks. With WebSocket the client must handle playback truncation itself. — [OpenAI Realtime conversations guide](https://developers.openai.com/api/docs/guides/realtime-conversations)
- [SECONDARY, unverified] Transport-level ~100 ms (WebRTC) vs ~200 ms (WebSocket) for the Realtime API; no method stated. — [TecnoBits](https://tecnobits.com/openai-gpt-realtime-2-guia-de-la-voz-en-tiempo-real/)
- [VENDOR forum] VB-CABLE's internal buffer defaults to 7168 samples @48 kHz (~150 ms), and "the stream cannot use more than half", so it adds at most ~70 ms. The configurable range is 512–7168 samples, and setting it too low breaks the cable. Internal latency affects only the virtual input side. — [VB-Audio forum](https://forum.vb-audio.com/viewtopic.php?p=1594); [VB-Audio forum: Latency/Buffer](https://forum.vb-audio.com/viewtopic.php?p=471)
- [VENDOR] BlackHole (macOS): "Zero additional driver latency", 2–256 channels, 8–768 kHz, Zoom routing guides provided. **License is GPL-3.0**, so non-GPL apps must contact Existential Audio. — [ExistentialAudio/BlackHole](https://github.com/ExistentialAudio/BlackHole)

### Inferences
- **Process locally wherever possible.** Each cloud round-trip costs ~60–120 ms in-region (2× Opus + jitter + RTT) and ~400–500 ms from India to US-hosted APIs. That is comparable to an entire ASR or TTS stage.
- For in-app local pipelines, use raw PCM between stages (no Opus). Use 10 ms WASAPI / Oboe low-latency streams, and keep playout queues short so interrupt flushes act within ~10–20 ms.
- For call usage, ship a **virtual mic driver** (VB-CABLE-class on Windows, BlackHole/Core Audio HAL plug-in on macOS) with the minimum safe buffer. Licensing (BlackHole GPL-3.0; VB-CABLE is donationware with commercial licensing) matters for distribution. The meeting app will apply its own AGC/NS/AEC to the synthetic voice. Users may need to disable "suppress background noise" or enable "original sound" modes; I have not verified this per app.

### Gaps
- No primary Microsoft numbers for WASAPI exclusive vs shared end-to-end latency, and no current measured CoreAudio numbers for Apple Silicon.
- No measurements of Zoom/Meet/Teams' own added latency or processing on virtual-mic input.
- No primary RFC 6716 breakdown of the 26.5 ms figure (20 ms frame + 6.5 ms look-ahead is the usual explanation, but I did not verify it here).

---

## Q7. Perceptual targets: acceptable conversational latency (G.114, turn-taking gap) and implications for a system that speaks on behalf of a user

### Takeaway
Telephony treats **≤150 ms one-way mouth-to-ear** as "essentially transparent" and **>400 ms** as unacceptable for network planning. Human turn-taking gaps average **~208 ms**, with each language's mode between 0 and 200 ms. No cascaded or end-to-end rewrite pipeline can be "transparent" by these standards. StutterClear will feel like a **consecutive or simultaneous interpreter** (≈1–2 s behind), and the design has to manage that: UX signaling, turn-holding cues, and an optional push-to-talk mode.

### Cited Findings
- [STANDARD via secondary quotes] ITU-T G.114: below 150 ms one-way, "most applications, both speech and non-speech, will experience essentially transparent interactivity". 150–400 ms is acceptable with increasing degradation. >400 ms is unacceptable for general network planning. Planners should use the E-model (G.107) for combined impairments. — [ITU-T G.114 (05/2003) PDF](https://www.cs.columbia.edu/%7Eandreaf/new/documents/other/T-REC-G.114-200305.pdf); [3GPP TS 22.105 annex (iTecSpec)](https://www.itecspec.com/3gpp/22.105/s/b.2.1); [Cisco study guide](https://open-exam-prep.com/study-guides/cisco-350-801/media-quality-qos/real-time-media-impairments)
- [PAPER] Stivers et al. 2009 (10 languages): "a mode offset for each language between 0 and +200 ms, and an overall mode of 0 ms". Cross-language mean is +208 ms; Japanese +7 ms (fastest), Danish +469 ms (slowest), Italian +310 ms. — [Stivers et al., PNAS 2009 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2705608/)
- [VENDOR] Pipecat targets ~1,500 ms voice-to-voice for conversational agents, and ~500 ms is achievable with co-location. — [Voice AI & Voice Agents guide](https://voiceaiandvoiceagents.com/)
- [PAPER] PWS-tuned endpointing trades 1.2–2.7 s of extra median delay for 79% fewer cut-offs, and the authors recommend letting users choose their tradeoff. — [Lea et al.](https://ar5iv.labs.arxiv.org/html/2302.09044)
- [PAPER] Measured simultaneous S2S translation lag: Hibiki LAAL 3.4–5.0 s, Seamless 2.8–4.2 s. This is the lag listeners tolerate for live interpretation-style systems. — [Hibiki full text](https://arxiv.org/html/2502.03382v2)

### Inferences
- **In a phone or video call** with the virtual mic replacing the real mic, the remote party hears only the corrected voice. StutterClear's latency adds to the call's own ~100–300 ms one-way path. With a ~1–1.5 s pipeline, the remote listener experiences longer turn-start gaps. Because typical human gaps are ~200 ms, the remote person may start talking during StutterClear's lag (overlap). Mitigations: a short "holding" cue, or a visual "speaking…" indicator in video; streaming mode so speech starts before the user finishes; and a clear user control to cancel ("never mind").
- **In person**, the listener hears the raw stuttered speech *and* the delayed corrected audio. Overlap with the user's ongoing speech is inherent, so phrase/consecutive mode (speak after each phrase) is likely more intelligible than streaming overlay. This is a design hypothesis to user-test.
- Phrase mode keeps the speaker in control and avoids flicker. Streaming mode lowers lag but risks "stale audio" when the user restarts. Exposing both as user controls matches Apple's finding that PWS want to choose their latency/robustness tradeoff.

### Gaps
- No studies were found on listener acceptance of AI re-voiced speech for PWS in conversation, or on acceptable ear-voice lag for same-language re-voicing (as distinct from interpretation).
- The effect of the user hearing their own delayed corrected voice (a delayed-auditory-feedback-like condition) on their fluency was not researched here.

---

## Q8. Realistic component-by-component mouth-to-ear latency budget: cascaded vs end-to-end

### Takeaway
- **Cascaded, phrase mode** (speak each phrase after its endpoint), local front-end and in-region or local models: **~0.6 s best / ~1.2–1.5 s typical** from end of the user's phrase to first corrected audio at the local speaker.
- If endpointing relies on long block-tolerant timeouts instead of a stutter-aware detector, the high case rises to **~2.5–3.7 s**.
- **Cascaded, streaming mode** (speak while the user continues): an estimated **~1.0–2.0 s ear-voice lag** per word.
- **End-to-end hosted APIs** measure **~1.2–2.3 s** from end of turn to first audio before client I/O, and cannot use the user's voice.
- **Self-hosted Moshi/Hibiki-style rewrite models** have ~160–200 ms frame-level response floors, but realistic rewrite lag is ~1–3 s (Hibiki precedent: 2.3–2.9 s end offset), and the rewrite capability would have to be trained.

### Cited Findings (per-component inputs used in the budget)
- Mic/speaker OS buffers: WASAPI ~10 ms period ([JUCE forum](https://forum.juce.com/t/windows-audio-low-latency-mode/42085)); Android ~20 ms round-trip best case ([Oboe guide](https://developer.android.com/games/sdk/oboe/low-latency-audio)); macOS 40 ms in / 15 ms out in Pipecat's budget ([Pipecat guide](https://voiceaiandvoiceagents.com/)).
- Noise suppression: RNNoise 20 ms window / 10 ms hop ([RNNoise](https://ar5iv.labs.arxiv.org/html/1709.08243)); DeepFilterNet2 40 ms ([DFN2](https://ar5iv.labs.arxiv.org/html/2205.05474)); Krisp ~15 ms [VENDOR] ([Krisp SDK](https://webcf.waybackmachine.org/web/20200808123722/https://krisp.ai/KrispSDK.pdf)).
- VAD frame: Silero 30+ ms chunk, <1 ms compute ([Silero](https://github.com/snakers4/silero-vad)); TEN 10/16 ms hop ([TEN VAD](https://github.com/TEN-framework/ten-vad)).
- Endpointing: VAD 200 ms + Smart Turn → ~250 ms after a pause; VAD-only stop 800 ms ([Pipecat guide](https://voiceaiandvoiceagents.com/)); LiveKit min delay 0.3–0.5 s ([LiveKit](https://docs.livekit.io/agents/build/turns/turn-detector/)); PWS-tuned medians 1,670–3,100 ms vs 450 ms baseline ([Lea et al.](https://ar5iv.labs.arxiv.org/html/2302.09044)).
- Streaming ASR: transcription+endpointing 300 ms in Pipecat's budget ([Pipecat guide](https://voiceaiandvoiceagents.com/)); Deepgram "300 milliseconds or less" [VENDOR] ([Deepgram docs](https://developers.deepgram.com/docs/measuring-streaming-latency.mdx)); Kyutai STT 1B has a fixed **0.5 s** delay with semantic VAD and 2.6B has 2.5 s; 400 streams real-time on an H100 [VENDOR] ([kyutai delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling)).
- LLM TTFT: hosted medians 536–739 ms, P95 1.1–1.8 s ([Pipecat guide](https://voiceaiandvoiceagents.com/)); on-device ~0.6 ms/prompt token, 30 tok/s ([Apple](https://machinelearning.apple.com/research/introducing-apple-foundation-models)); 0.5B on CPU ~50–160 ms ([LiveKit](https://docs.livekit.io/agents/build/turns/turn-detector/)).
- TTS time-to-first-audio, median/P95 ms, measured by Pipecat (table captioned "June 2025" on a page updated June 2026): Cartesia Sonic 3.5 195/240; Gradium 235/320; Deepgram Aura-2 310/600; ElevenLabs Turbo v2.5 330/670; Inworld TTS 1.5 Max 337/560 ([Pipecat guide](https://voiceaiandvoiceagents.com/)). Kyutai TTS consumes **streaming text** (can start while the LLM is still generating), clones a voice from a **10 s sample**, and runs 32 streams on an L40S at **350 ms** [VENDOR] ([Kyutai TTS](https://kyutai.org/next/tts); [Kyutai TTS/Unmute open-source post](https://kyutai.org/2025/07/03/tts-unmute-open-source.html)). ~220 ms single-stream is [SECONDARY] ([Learnopoly](https://learnopoly.com/kyutai-releases-tt-of-text-in-2b-parameter-streaming-with-a-latency-of-220-ms-and-2-5-million-hours-of-training/)). Kyutai has not released its voice-embedding (cloning) model publicly, and access is "controlled" ([Kyutai Unmute post](https://kyutai.org/2025/05/22/unmute.html)).
- Network: Opus 21–26.5 ms, jitter buffer 40 ms, in-region transit 10 ms each way; Europe→US RTT ~250 ms, India→US ~350 ms ([Pipecat guide](https://voiceaiandvoiceagents.com/); [Opus](https://en.wikipedia.org/wiki/Opus_(audio_format))).
- Virtual mic: BlackHole 0 ms added ([BlackHole](https://github.com/ExistentialAudio/BlackHole)); VB-CABLE ≤~70 ms ([VB-Audio forum](https://forum.vb-audio.com/viewtopic.php?p=1594)).
- E2E: Moshi 160 ms theoretical / 200 ms practical on L4 ([Moshi GitHub](https://github.com/kyutai-labs/moshi)); Qwen3-Omni 234 ms theoretical ([Qwen3-Omni](https://arxiv.org/abs/2509.17765)); gpt-realtime family 1.2–2.3 s and Gemini 3.1 Live ~1.4 s measured end-of-turn→first audio ([Plain Concepts, July 2026](https://www.plainconcepts.com/?p=121434)); Hibiki end offset 2.3–2.9 s, LAAL 3.4–5.0 s ([Hibiki](https://arxiv.org/html/2502.03382v2)).

### Inferences

**Budget A: Cascaded, PHRASE MODE.** Clock starts at the end of the user's phrase (last voiced frame) and stops at the first corrected audio at the local speaker. Assumes the AEC/NS/VAD/stutter front-end runs locally and ASR/LLM/TTS are either local or in the same cloud region. Sums are my arithmetic over the cited component figures.

| # | Component | Best (ms) | Typical (ms) | High (ms) | Basis |
|---|---|---|---|---|---|
| 1 | Mic capture buffer (WASAPI/Oboe/CoreAudio) | 10 | 20 | 40 | WASAPI 10 ms period; Pipecat macOS 40 |
| 2 | AEC (AEC3-class, 10 ms frames) | ~0–5 | ~5 | ~10 | no primary figure; estimate |
| 3 | Noise suppression | 10 (RNNoise) | 15–20 (Krisp-class) | 40 (DeepFilterNet2) | papers/vendor |
| 4 | VAD frame granularity | 10 (TEN) | 16–32 | 32 (Silero) | compute <1 ms |
| 5 | Stutter detector | 0 on critical path if run in parallel to gate endpointing | 0–30 | 50+ if serial | assumption; see sibling research |
| 6 | **Endpoint decision after last voiced frame** | 200–250 (VAD 200 ms + Smart Turn ~12–60 ms) | 300–500 (LiveKit min_delay) | **1,670–3,100** (PWS block-tolerant timeouts, Apple) | Pipecat / LiveKit / Lea et al. |
| 7 | Uplink to cloud (Opus + jitter + transit) | 0 (local) | 70 (in-region) | 250 (India→US, half RTT + codec/jitter) | Pipecat budget |
| 8 | ASR finalization after endpoint | ~50–100 (partials already streaming) | 150–300 | 500 (Kyutai STT fixed delay) | Pipecat 300 incl. endpointing; Kyutai |
| 9 | Text correction: tagger + small LLM TTFT | 50–150 (tagger + local 0.5–3B, short prompt) | 300–600 | 740–1,800 (hosted frontier median→P95) | Apple 0.6 ms/tok; LiveKit; Pipecat |
| 10 | First committable words / aggregation | 5–20 | 20–100 | 250+ (full sentence on-device at 30 tok/s) | Pipecat 20 ms aggregation; Apple 30 tok/s |
| 11 | TTS time-to-first-audio (personalized voice) | 195 | 250–350 | 600–670 (P95) | Pipecat measured table; Kyutai 350 @32 streams |
| 12 | Downlink (Opus + jitter + transit) | 0 (local) | 70 | 250 | Pipecat budget |
| 13 | Playout buffer to speaker | 10 | 15 | 40 | Pipecat macOS 15 |
| 14 | Virtual cable (call mode only) | 0 (BlackHole) | 10–20 | 70 (VB-CABLE default) | BlackHole / VB-Audio |
| | **Sum: end of phrase → first corrected audio (local ear)** | **≈ 0.55–0.75 s** | **≈ 1.2–1.6 s** | **≈ 3.5–6.5 s** | arithmetic |
| 15 | Meeting app path to remote ear (not controllable) | +~100 | +~150 | +~400 | G.114 bands |

- Pipecat's independent end-to-end figures (1,293 ms typical, ~500 ms co-located best) bracket the typical and best columns, which is a useful sanity check. ([Pipecat guide](https://voiceaiandvoiceagents.com/))
- **The dominant controllable term for PWS is row 6.** A stutter-aware endpointer that keeps the turn open *only while a block or repetition is detected* can hold typical-case latency near 0.3–0.5 s. The alternative is Apple's 1.7–3.1 s medians.
- **The second-largest term is LLM + network.** Local (or in-region) small-model correction with diff-style output removes ~0.5–1.5 s versus a cross-continent frontier API.

**Budget B: Cascaded, STREAMING / SIMULTANEOUS MODE.** Clock measures "ear-voice span": a fluent word spoken at time t is heard (corrected) at t + lag. This is an estimate; I found no published measurement of a disfluency-rewrite system.

| Component | Estimate (ms) | Basis |
|---|---|---|
| Capture + AEC + NS + VAD | 30–80 | rows 1–4 above |
| Streaming ASR word emission delay | 300–500 | Deepgram ≤300 [VENDOR]; Kyutai STT 1B 500 ms |
| Disfluency look-ahead (wait for right context until the span is clearly fluent or clearly a repetition) | ~1–3 words (no ms source) | Teaching-BERT-to-Wait shows learned waits beat fixed look-ahead |
| Incremental correction (tagger / local LLM delta) | 20–150 | rows 9–10 |
| TTS first audio for a new chunk (streaming-text TTS, warm session) | 200–350 | Pipecat TTFA table; Kyutai TTS streaming text |
| Playout (+ virtual cable) | 10–90 | rows 13–14 |
| **Ear-voice lag (estimate)** | **≈ 1.0–2.0 s** with local or in-region models; **+0.5 s** from India to US-hosted cloud | arithmetic; compare Hibiki LAAL 3.4–5.0 s for translation |

- Extra required machinery: an **epoch/commit protocol**. Only voice text that has passed the stability check. On a user restart, cancel the uncommitted TTS and flush queued audio (Pipecat-style). Because the output is shorter than the input (disfluencies removed), the system catches up during blocks.

**Budget C: End-to-end hosted speech-to-speech (gpt-realtime / Gemini Live) used as the rewrite engine.**
- End-of-turn detection (server VAD 200–500 ms, or semantic VAD), plus model-measured end-of-turn→first audio of **1.2–2.3 s** (OpenAI family) or **~1.4 s** (Gemini 3.1 Live), plus client I/O ~50–100 ms. Total **≈1.4–2.5 s after phrase end**, before any voice-conversion stage needed to sound like the user (latency unknown). There is also a risk of the model answering instead of repeating, and the PWS endpointing problem (row 6) still applies.

**Budget D: Self-hosted full-duplex E2E rewrite model (Moshi/PersonaPlex/Hibiki-style; research path).**
- Architectural floor is ~160 ms (80 ms Mimi frame + 80 ms acoustic delay) and ~200 ms practical on an L4. But a rewrite model trained with an alignment-based lag policy would plausibly sit at **~1–3 s** lag (Hibiki trained with ≥2 s lag; Hibiki-Zero used RL to reduce it). Hardware is a 24 GB GPU (PyTorch), Apple Silicon (MLX), or Hibiki-M-class 1.7B on a phone. Voice preservation is demonstrated in Hibiki (speaker similarity 0.41–0.48). Building it requires creating a parallel dysfluent→fluent same-voice corpus; StutterZero shows synthetic targets cause prosody mismatch. Licensing: Moshi weights CC-BY 4.0 (commercial OK); Hibiki/Hibiki-Zero non-commercial.

**Bottom line for architecture.** A **cascaded, mostly on-device pipeline** gives the best combination of latency (~0.6–1.5 s phrase mode), voice personalization (voice-cloned streaming TTS), user control and interruption semantics. Its key parts are: AEC + light NS, a stutter-aware endpointer, a streaming ASR, a streaming disfluency tagger plus a small LLM with diff output, and a streaming-text TTS with cancel/flush. It should expose phrase mode and streaming mode as user settings. End-to-end models are not lower-latency in practice for a rewrite task today. They are a credible v2 research track via Hibiki-style fine-tuning on Moshi/PersonaPlex.

### Gaps
- No published end-to-end measurement of any system doing same-language streaming re-voicing of stuttered speech. All of Budget B and the stutter-detector row are estimates.
- No primary latency data for personalized (cloned-voice) TTS specifically. TTFA tables are for stock voices, and cloned-voice latency may differ.
- AEC algorithmic delay, meeting-app virtual-mic handling, and short-prompt local LLM TTFT on the target hardware (Windows laptop / Android / iPhone) all need in-house benchmarking.
