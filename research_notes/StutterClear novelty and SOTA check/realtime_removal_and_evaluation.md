# StutterClear novelty and SOTA check: real-time stutter removal that keeps the real voice, and how stutter removal is evaluated

Research date: 17 Oct 2026. Labels:
- **SHIPPED**: a product or plugin people can use today.
- **RESEARCH**: a paper or demo.
- **PATENT**: a granted patent, with no evidence of an implementation.
- **OSS**: an open-source repo.
- **VENDOR CLAIM**: a company's own figure, not independently checked.

This file builds on `research_notes/Real time stutter correction pipeline/prior_art_products_ethics_evaluation.md`, which covers the offline and resynthesis systems in depth (STEAMROLLER, StutterZero, FluentSpeech, Parrotron, Whispp, Ayta, Descript, Cleanvoice). Only what matters for the two novelty questions is repeated here.

**Verdict summary (details and citations below)**

| Claim StutterClear might make | Verdict | Closest prior art |
|---|---|---|
| Real-time stuttered-to-fluent speech in general | PARTIALLY DONE (all resynthesis, seconds of delay) | STEAMROLLER (AAAI 2026, 3.77 s, StyleTTS2 clone); SpeechAgent (Oct 2025, 0.91 s per utterance, generic TTS) |
| Removing stutter segments from the user's **real voice** in real time, before playback or transmission | **PARTIALLY DONE: claimed in patents (IBM 2012; IBM→Nuance 2011), never published as an implemented, evaluated system** | IBM US8682678; Nuance US8571873/US8600758 |
| Live removal of fillers from the real voice through a delay buffer | PARTIALLY DONE (mutes or beeps; does not splice) | CleanStream OBS plugin (OSS, 2023–Nov 2025) |
| Causal, frame-level stutter detection with lookahead feeding a splicer, with measured mouth-to-ear latency, on stuttering (repetitions, prolongations) | **OPEN**: no implemented system found | Detect-and-Pass (Amazon 2022) drops stutter frames *inside ASR* only |
| Evaluating with residual disfluencies counted from a verbatim-ASR transcript of the edited audio | PARTIALLY DONE (counted from a transcript in TalkLess, UIST 2025; filled pauses per minute in Das 2019); **not found with CrisperWhisper or with stuttering** | TalkLess; Das et al. ICASSP 2019 |
| "Wrongly cut fluent words" rate | PARTIALLY DONE in the **text** domain (Fluent Error Rate, FER: Jamshid Lou & Johnson 2020); **not found for audio splicing with word timestamps** | FER/DER metrics |
| Meaning preservation as intended-transcript WER of the edited audio vs the original | PARTIALLY DONE in spirit (dual verbatim/intended references, 2026; ASR-WER of output; SBERT/BERTScore); **this exact self-referenced metric was not found** | Toyin et al. 2026; nyra labs 2026 |
| A standard stutter-to-fluent benchmark | OPEN (none exists); FluencyBank Timestamped provides verbatim and intended references StutterClear could use | Romana et al. 2024 |

---

## Q1. Has anyone built a streaming or real-time system that turns stuttered speech into fluent speech (2024–2026), and does any keep the speaker's real voice?

### Takeaway
Every research system from 2023–2026 that outputs fluent audio from stuttered speech **resynthesizes** it:
- STEAMROLLER (Whisper → GPT-4o → StyleTTS2 clone, about 3.8 s per segment)
- SpeechAgent (Whisper → LLM → generic expressive TTS; 0.91 s after a whole 11.5 s utterance; server-based)
- StutterZero/StutterFormer (offline, Griffin-Lim)
- FluentSpeech (offline diffusion editing)

None streams word by word, and none keeps the speaker's own recorded voice. Two adjacent October 2026 items are related but do not overlap with the system:
- **Continuous causal forecasting of stutter onset:** ASU, every 100 ms, AUROC 0.63.
- **Verbatim-ASR advances:** CrisperWhisper 2.0 and nyra labs' controllable verbatim/intended ASR.

**Verdict for "real-time stutter-to-fluent": PARTIALLY DONE**, with resynthesis and second-scale latency. **For "keeps the real voice, streaming": not done by any implemented, published system**; see Q2 for the patents.

### Cited Findings
**Resynthesis systems (RESEARCH)**
- **STEAMROLLER** (arXiv 2601.10223, Jan 2026; AAAI 2026).
  - Pipeline: Whisper-base, then a GPT-4o multi-agent repair step, then StyleTTS2 zero-shot clone.
  - Latency: 3.77 ± 0.62 s per 50–100-word segment.
  - FluencyBank WER: 31.60% → 18.69%. WER on the cloned audio: 19.74%.
  - Voice: MOS-N 4.52; listener-rated MOS-S (speaker similarity) only 3.4. 7 of 23 users said the voice lacked warmth or did not resemble them.
  - [arXiv HTML](https://arxiv.org/html/2601.10223v1); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/41300)
- **SpeechAgent** (Lou et al., UNSW/Macquarie/NUS/CSIRO Data61; arXiv 2510.20113, 23 Oct 2025).
  - Scope: a mobile client-server "speech refinement" agent for dysarthria, stuttering and aphasia.
  - Pipeline: impairment classifier, then Whisper ASR, then an LLM rewrite, then **ParaStyleTTS**. The voice is a style embedding from text prompts, not the user's voice.
  - Transport: audio is sent to a cloud server over HTTPS **after recording**.
  - Latency: 0.91 s to refine an 11.49 s utterance (RTF 0.08), split as ASR 0.35 s, LLM 0.43 s, TTS 0.11 s, classifier 0.02 s.
  - It claims "there is currently no assistive communication tool that offers real-time refinement of impaired speech."
  - [arXiv 2510.20113](https://arxiv.org/abs/2510.20113)
- **StutterZero/StutterFormer** (arXiv 2510.18938, Oct 2025): offline waveform-to-waveform with a Griffin-Lim vocoder. No latency, no speaker similarity, and speaker identity is not preserved — [arXiv HTML](https://arxiv.org/html/2510.18938v1)
- **FluentSpeech** (ACL Findings 2023): offline diffusion editing of stutter regions, with no inference speed reported — [ar5iv 2305.13612](https://ar5iv.labs.arxiv.org/html/2305.13612)
- **Streaming Parrotron** (Interspeech 2023): atypical speech converted to a *fixed synthetic voice*. Delay is 320 ms *after the end of speech*, and it was not evaluated on stuttering — [ar5iv 2210.13761](https://ar5iv.labs.arxiv.org/html/2210.13761)

**Streaming front ends for stuttering (RESEARCH, no audio output)**
- **Amazon "Detect and Pass"** (Shonibare, Tong, Ravichandran, Amazon Alexa; arXiv 2202.05396, Feb 2022).
  - Method: audio is split into 100 ms frames. A small CNN classifier labels each frame using **4 context frames on each side** (900 ms total window). The RNN-T "skips frames with stutter" during decoding.
  - Results: WER reduction of 12.18–71.24% across three RNN-T models.
  - It never produces output audio — [arXiv 2202.05396](https://arxiv.org/pdf/2202.05396)
- **Continuous stutter-onset forecasting** (Kumar, Daliri, Gupta, ASU; arXiv 2610.19235, 13 Oct 2026).
  - Setup: every 100 ms, using only past audio (frozen Whisper-large-v3 over 2 s windows plus a 3.67M-parameter causal Conformer).
  - Performance: predicts onset within 300 ms with AUROC 0.63.
  - Operating points: at a 25% alarm duty cycle it anticipates 58.6% of events. At max-F1 it anticipates 43.9% with a mean lead of 196 ms but 27.6 false warning bouts per minute.
  - Data: used CrisperWhisper 2.0 to build onset annotations on FluencyBank Timestamped (2,872 events). Fleiss κ: 0.54 repetitions, 0.73 prolongations, 0.56 blocks. Median onset disagreement 80 ms.
  - It also cites an on-device 3 s-ahead stutter predictor (Kozak 2026).
  - [arXiv 2610.19235](https://arxiv.org/abs/2610.19235)

**Products marketed to people who stutter (2024–2026)**
- **fluencyAI / Arkenza "Voice Mirror"**: "turns your speech into clean, fluent text and instantly recreates it in your own AI voice—removing stutters, repeats, and interruptions." That is resynthesis, not the real voice. No latency, call integration, launch date or price is stated; a "voice-mirror-lite" try link exists (VENDOR CLAIM) — [fluencyai.net](https://www.fluencyai.net/)
- **Whispp:** whisper-to-voice resynthesis. Its 2026 site claims <120 ms on-device and defers assistive stuttering use pending medical validation (VENDOR CLAIM) — [whispp.com](https://www.whispp.com/)
- **Ayta:** whisper or quiet speech converted to "speech without stuttering" via a virtual mic (VENDOR CLAIM) — [ayta.ai](https://ayta.ai/)
- **Search result:** an extended web search for a real-time stutter-removal app for voice calls (2026) found **no such app**. Results were noise suppression (Krisp, NVIDIA Broadcast), DAF apps, and practice apps (Stamurai, StutterLab, FluentLoop) — [Krisp blog](https://krisp.ai/blog/best-noise-cancelling-app/); [StutterLab](https://www.stutterlab.com/blog/best-stuttering-apps-for-adults); [FluentLoop](https://fluentloopai.com/)

### Inferences
- The field has converged on **ASR → LLM → TTS**. StutterClear's design runs the other way: no transcript, no LLM and no TTS in the audio path, with the real waveform passed through a 250–600 ms buffer. Among implemented, published systems, that is a real architectural gap rather than an incremental change.
- Detect-and-Pass is the closest *mechanistic* precedent: frame-level stutter detection with about 400 ms of future context, used to drop frames. It does this inside an ASR decoder, not on audio sent to a listener. It is the best citation to show that "frame dropping with lookahead" helps a machine listener. StutterClear extends the idea to human listeners.
- ASU's forecasting work shows that purely causal (no lookahead) prediction of stutter onset is still weak (AUROC 0.63). That supports StutterClear's choice to *detect with a lookahead buffer* rather than predict.

### Gaps
- No primary latency or quality data was found for fluencyAI Voice Mirror. Its launch date and real-time capability are unknown.
- The Kozak 2026 on-device stutter predictor was seen only as a citation inside arXiv 2610.19235. Its own paper was not retrieved.
- No App Store or Google Play listing (2025–2026) was found for a "stutter remover for calls" that keeps the real voice. App stores were not searched exhaustively.

---

## Q2. Does any system edit or splice the *original waveform* in real time to remove disfluencies (fillers, repetitions, prolongations)? This covers products, plugins, patents and repos.

### Takeaway
Three kinds of prior art exist, but **none is an implemented, evaluated system that splices stutter events out of live speech with crossfades**:
1. **Patents that claim the exact concept** but report no implementation or evaluation:
   - IBM's "Automatic realtime speech impairment correction" (priority 2012, granted 2014, listed as active to about 2032). It deletes stutter artifacts (repeated phonemes) from captured audio before playback or transmission.
   - IBM/Nuance's "Reconstruction of a smooth speech signal from a stuttered speech signal" (priority 2011). It keeps one of the repeated syllables, shortens prolongations, cuts silences, and uses PSOLA reconstruction "in real-time", for example on telecom servers.
2. **One live open-source tool for fillers:** the CleanStream OBS plugin (2023–Nov 2025). Whisper.cpp detects unwanted words in a delayed stream and **replaces them with silence or a beep**. It does not remove time, and it does not handle stuttering.
3. **Offline real-voice editors:** Descript, Cleanvoice "Stutter Remover", Riverside, Das et al. ICASSP 2019, and TalkLess (UIST 2025; cuts plus VoiceCraft-synthesized transitions).

Live mainstream tools (Krisp, NVIDIA Broadcast, Voicemod, Zoom and Discord suppression) only remove noise. Live "filler coaching" tools (Yoodli; Poised, which shut down on 8 Oct 2026) detect fillers but do not edit the audio.

**Verdict: PARTIALLY DONE (patented concept; live filler muting exists).** StutterClear can credibly claim to be the **first implemented and evaluated** streaming system that splices stuttering events (repetitions, prolongations, fillers) out of the real voice with crossfades and a bounded delay. It **should not claim the idea itself is new**: cite the IBM and Nuance patents.

### Cited Findings
**Patents (PATENT)**
- **US8682678B2, "Automatic realtime speech impairment correction".**
  - Assignee: IBM. Inventors: Peter K. Malkin, Sharon M. Trewin.
  - Dates: priority/filing 14 Mar 2012; granted 25 Mar 2014. Google Patents lists it as "Active, expires 2032-07-14" (Google's assumption, not a legal conclusion).
  - Claim 1: analyze captured speech, identify impairment artifacts using an impairment-to-artifact database (stuttering = repeated phonemes; Tourette = inserted verbiage; lisp = slurring), delete them, and provide the modified audio.
  - The description says "the identified artifacts are deleted from the captured audio segment."
  - Operation: audio is captured in segments, for example "contiguous segments of audio signals of one minute each", into a queue processed asynchronously. A phone app "may intercept the first user's speech… before they are transmitted."
  - Not reported: latency figures, evaluation, prolongations or blocks.
  - [Google Patents US8682678B2](https://patents.google.com/patent/US8682678B2/en). Related family members: US8620670 and US20130246058 — [USPTO PDF](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8620670); [Justia](https://patents.justia.com/patent/20130246058)
- **US8571873 and US8600758B2, "(Systems and methods for) reconstruction of a smooth speech signal from a stuttered speech signal".**
  - Inventors: Om D. Deshmukh, Suraj S. Sheth, Ashish Verma. Assigned from IBM to Nuance Communications.
  - Dates: priority 18 Apr 2011; US8600758 granted 3 Dec 2013, and **expired for non-payment of fees (effective 3 Dec 2021)**.
  - Claim 1: identify stuttered regions, then retain only one of the repeated syllables, shorten the steady state of elongated phones, and/or reduce silence or breath regions. Then reconstruct, possibly with PSOLA.
  - The background claims it "restores the smooth regions in real-time." It describes a telecom provider hosting the service so the listener hears the smoothed version. It works on the signal without ASR, making it language-independent.
  - No experiments are reported.
  - [Google Patents US8600758B2](https://patents.google.com/patent/US8600758B2/en); [USPTO PDF US8571873](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8571873)
- **US11616883B2, "Systems and methods for improved audio/video conferences".**
  - Assignee: Rovi Guides, now Adeia. Priority 26 Jan 2021; granted 28 Mar 2023. Continuations US12113937B2 and US20240430365A1.
  - Content: records a user who talks while muted, then plays it to others at a variable rate. The description says it "may look for filler words such as 'uh,' 'um'… and remove them" and remove pauses. This is catch-up playback of a recording, not continuous live splicing, and it does not target stuttering.
  - [Google Patents US11616883B2](https://patents.google.com/patent/US11616883B2/en)

**Live tools (SHIPPED / OSS)**
- **CleanStream** (OBS Studio plugin; author Roy Shilkrot; first release 13 Jun 2023; v0.2.0 on 11 Nov 2025).
  - Function: "cleans live audio streams… from unwanted words and utterances," including "uh"s, "um"s and profanity, using Whisper via whisper.cpp, running locally, in about 100 languages.
  - Mechanism: a "built-in lag/delay mechanism." Each chunk's decision "determines if the audio rendering will play the original audio or e.g. a beep or silence." **It mutes or beeps; it does not cut time or crossfade.**
  - No latency or accuracy figures are published. It is labelled EXPERIMENTAL.
  - [OBS resource page](https://obsproject.com/forum/resources/cleanstream-remove-profanity-and-fillers-in-your-live-stream-or-recording-with-ai.1732/); [GitHub royshil/obs-cleanstream](https://github.com/royshil/obs-cleanstream)
- **Riverside** filler-word removal works in the post-recording editor (cut, mute, or a "Smart" mix) and cannot be used during a call — [Riverside help](https://support.riverside.com/hc/en-us/articles/13364581561245-Remove-filler-words)
- **Krisp and NVIDIA Broadcast** are real-time noise and echo removal through virtual mics. They do not target fillers or stutters — [Riverside/Krisp search summary: ProVideo Coalition](https://www.provideocoalition.com/nvidia-rtx-voice-and-krisp-remove-background-noise-from-your-broadcasts/amp/); [MakeUseOf](https://www.makeuseof.com/rtx-gpu-ai-noise-removal-microphone/); [Voicemod voice cleaner](https://www.voicemod.net/en/voice-cleaner/)
- **Live filler *coaching*** (detection only, no audio editing):
  - Yoodli gives real-time, in-meeting feedback on filler words (directory-site descriptions) — [Leadr review](https://leadr.co/?p=52594)
  - Poised gave live nudges during Zoom, Teams and Meet calls, and its homepage says it **shut down on 8 Oct 2026** — [poised.com](https://poised.com/)
- **Microsoft Azure Speech "TrueText"** removes disfluencies from the *transcript text* in real-time STT (the user must set `PostProcessingOption = TrueText`). It is text only — [GitHub issue #1950](https://github.com/azure-samples/cognitive-services-speech-sdk/issues/1950)

**Offline real-voice editing (RESEARCH / SHIPPED)**
- **Das, Gandhi, Naik, Shilkrot, "Increase Apparent Public Speaking Fluency by Speech Augmentation"** (Stony Brook; ICASSP 2019; arXiv 1812.03415). OLDER.
  - Method: a CRNN segments fillers frame by frame from MFCC/log-mel features (filler F1 0.955 on Switchboard vs 0.978 for Kaldi ASR). An SVM classifies disfluent silences (F1 about 0.91–0.92).
  - Repair: fillers are replaced with background-only audio (via voice/background separation), and long pauses are shortened to the median fluent-silence length.
  - Data: offline, on 50–60 s recordings from 20 non-native speakers.
  - **Future work listed explicitly:** "segment … stuttering" and "a dynamic and online repair scheme."
  - [arXiv 1812.03415](https://arxiv.org/pdf/1812.03415)
- **TalkLess** (Benharrak, Peng, Pavel; UT Austin/UC Berkeley; UIST 2025).
  - Method: LLM transcript shortening, including filler and repetition removal. The original waveform is cut, and **VoiceCraft re-synthesizes about half a word on each side of each cut** for smooth transitions. Runs offline on an A40 GPU.
  - Users preferred it to full resynthesis because full resynthesis lost "vocal authenticity."
  - [arXiv 2507.15202](https://arxiv.org/abs/2507.15202)
- **Descript and Cleanvoice "Stutter Remover"** work on uploaded recordings only — [Descript help](https://help.descript.com/hc/en-us/articles/10164806394509-Remove-filler-words); [cleanvoice.ai](https://cleanvoice.ai/stutter-remover/)
- **Earlier academic "stutter removal" from audio:** mostly 2020–2022 conference and journal papers, seen only as citations, offline. They include:
  - Arjun et al., "Automatic correction of stutter in disfluent speech", Procedia CS 171, 2020
  - Rajput et al., "Speech stuttering detection and removal using deep neural networks", 2021
  - "ASR with stuttering speech removal using LSTM", IJRTE 2020
  
  — cited in [SpeechAgent](https://arxiv.org/abs/2510.20113) and [Toyin et al. 2026](https://arxiv.org/abs/2604.20535)

**Evidence that naive cutting can *hurt* (important for StutterClear)**
- SpeechAgent's human evaluation of stuttered speech used 1–5 clarity ratings and comparison MOS (C-MOS) against the unprocessed input.
  - **Acoustic, real-voice baselines scored at or below the unprocessed stuttered audio:**
    - Unprocessed impaired speech: clarity 2.67
    - Cleanvoice: clarity 2.50, C-MOS −0.17
    - "AutomaticCorrection" (Arjun 2020, repetition removal and silence trimming): clarity 1.33, C-MOS −0.50
  - Resynthesis results: SpeechAgent clarity 4.50, C-MOS 1.33; ASR+TTS clarity 4.00.
  - The authors attribute the acoustic baselines' failure to cuts that "distort meaning and naturalness."
  - The rater count is not stated in the retrieved text.
  - [arXiv 2510.20113](https://arxiv.org/abs/2510.20113)

### Inferences
- **Novelty framing.** StutterClear's real contribution is an *engineering and evaluation* first, not a *conceptual* first:
  - a small causal detector with 20 ms frames and 200–500 ms lookahead
  - a bounded delay buffer with crossfaded splicing
  - prolongations and repetitions covered, not just fillers
  - blocks deliberately left uncut
  - measured latency and a reproducible evaluation

  The 2011–2012 IBM and Nuance patents describe the concept but report no implementation, latency or results. CleanStream is the only live implementation found, and it mutes rather than cuts, so the listener still hears the gap's duration.
- **Freedom to operate.** IBM US8682678 may still be in force until about 2032 (Google's estimate). Its claim 1 uses an impairment-to-artifact database and a comparison against a reference reading, which may narrow its scope. This is a flag for the team, not legal advice. US8600758 has lapsed.
- **Baselines to beat.** SpeechAgent's result shows that real-voice cutting, as done by Cleanvoice and Arjun 2020 on whole utterances, was judged *less* clear than doing nothing. StutterClear must show in a listener study that its crossfaded, conservative cuts do better than "no processing." The natural baselines are:
  - (a) unprocessed audio
  - (b) CleanStream-style muting
  - (c) Cleanvoice offline
  - (d) a resynthesis cascade (STEAMROLLER or SpeechAgent-like)
- **Mute vs cut.** CleanStream's mute or beep keeps timing and adds no cumulative delay. Cutting shortens the stream and lets the buffer drain. StutterClear should report how the buffer drains or fills over a conversation, since no prior work does.

### Gaps
- Whether US8571873 (the parent of US8600758) is still in force was not checked.
- CleanStream's actual delay length and detection accuracy are undocumented. Measuring it (or reading its source code) would give a fair live baseline.
- No independent evaluation of Cleanvoice's stutter remover was found beyond SpeechAgent's small rating table.
- Zoom and Teams app marketplaces were not searched directly for disfluency plugins. Web searches surfaced none.

---

## Q3. How are stutter-repair and disfluency-removal systems evaluated? Has anyone used verbatim ASR to count residual disfluencies, measured over-removal of fluent words, or used intended-transcript WER for meaning preservation? Is there a benchmark?

### Takeaway
Prior stutter-to-fluent papers judge output with:
- **WER against a fluent reference** (STEAMROLLER, StutterFormer, Parrotron)
- **semantic similarity** (SBERT, BERTScore, SeMaScore)
- **listener MOS, C-MOS, clarity**
- **acoustic distortion** (MCD, STOI, PESQ for FluentSpeech)

None counts residual stuttering events in the output with a verbatim ASR, and none measures fluent words wrongly removed. The closest pieces exist in neighbouring areas:
- **TalkLess (UIST 2025)** counts residual "speech disfluencies" (fillers plus repetitions) in edited audio from an aligned transcript.
- **Das et al. (2019)** report filled pauses per minute on processed audio.
- **Jamshid Lou & Johnson (2020)** define **Fluent Error Rate (FER)** and **Disfluent Error Rate (DER)** for joint ASR plus disfluency removal. These are exact text-domain analogues of StutterClear's "wrongly cut fluent words" and "residual disfluencies."
- **2026 papers formalize verbatim and intended references:**
  - Toyin et al. (MBZUAI): isWER vs vWER on FluencyBank Timestamped.
  - nyra labs, the CrisperWhisper authors: a WER split into content and style errors, plus typed disfluency F1.

**Verdict:**
- (i) Residual disfluencies via verbatim ASR: PARTIALLY DONE. Counted via transcripts in TalkLess and Das; **not done with CrisperWhisper or for stuttering**.
- (ii) Intended-transcript WER of edited vs original audio for meaning: **OPEN as an exact metric**. Its parts exist.
- (iii) Fluent words wrongly cut, from word timestamps: PARTIALLY DONE in text (FER); **OPEN for audio editing**.
- Standard stutter-to-fluent benchmark: **OPEN**.

### Cited Findings
**Metrics used by stutter and impaired-speech repair systems**
- **STEAMROLLER:**
  - Text accuracy: WER, MER and WIL against fluent references.
  - Meaning: Sentence-BERT similarity (0.81 → 0.91); human "meaning preservation" 4.13/5.
  - Voice: MOS-N 4.52; MOS-S 3.4.
  - Usability: SUS 66.9.
  - Latency: 3.77 s.
  - [arXiv HTML](https://arxiv.org/html/2601.10223v1)
- **StutterFormer:** WER 0.08 and BERTScore-P 0.94 on synthetic pairs; WER 0.120/0.161 on FluencyBank. Whisper is the judge — [arXiv HTML](https://arxiv.org/html/2510.18938v1)
- **FluentSpeech:** MCD 5.86, STOI 0.81, PESQ 1.91; MOS 4.27 and speaker-similarity MOS 4.42 (seen speakers); stutter-localization accuracy 80.5% and precision 94.4% — [ar5iv](https://ar5iv.labs.arxiv.org/html/2305.13612)
- **SpeechAgent:** human clarity (1–5), C-MOS and a "Recover" percentage, plus system latency, throughput and RTF. It claims to release a "benchmark suite and evaluation pipeline for impaired-speech systems" — [arXiv 2510.20113](https://arxiv.org/abs/2510.20113)

**Residual-disfluency counts on edited audio (closest precedents)**
- **TalkLess (UIST 2025)** defines a metric called "Speech Disfluencies": "the combined count of filler words (e.g., 'uh', 'um') and repetitions (e.g., '[..] in my, in my perspective [..]'), counted using a speech transcript linked to the audio." It also uses:
  - "Coherence Errors": edit-induced abrupt cuts mid-word or missing co-references, counted by listening
  - "Coverage": sentence-embedding similarity
  - "Style Preservation" and "Audio Preference"
  
  The transcripts came from rev.ai plus Gentle alignment, not a verbatim ASR re-run on the output — [arXiv 2507.15202](https://arxiv.org/abs/2507.15202)
- **Das et al. (ICASSP 2019)** report, before → after processing (L2-fluency metrics from Kormos & Dénes):
  - filled pauses per minute: 3.659 → 1.762
  - mean length of pauses: 0.654 → 0.365 s
  - speech rate: 165 → 186
  - phonation-time ratio: 58.9 → 65.6
  
  [arXiv 1812.03415](https://arxiv.org/pdf/1812.03415)

**Over-removal and under-removal metrics (text domain)**
- **Jamshid Lou & Johnson, "End-to-End Speech Recognition and Disfluency Removal"** (Findings of EMNLP 2020).
  - **FER (Fluent Error Rate)** = (s_f + i_f + d_f) / n_f: errors, including **deletions**, on fluent reference words.
  - **DER (Disfluent Error Rate)** = (s_d + i_d + c_d) / n_d: disfluent words that were *not* deleted.
  - Both use a weighted alignment that favours deleting disfluent words.
  - Switchboard test results: end-to-end model FER 11.6, DER 24.0; pipeline FER 12.1, DER 22.2; gold transcript plus detector FER 2.7, DER 17.7.
  - Code: `pariajm/e2e-asr-and-disfluency-removal-evaluator`.
  - [ar5iv 2009.10298](https://ar5iv.labs.arxiv.org/html/2009.10298); [ACL Anthology author page](https://aclanthology.org/people/paria-jamshid-lou/unverified/)
- **"Mind the Pause"** (IIT Patna, arXiv 2605.12242, May 2026) evaluates text disfluency correction with BLEU, chrF2 and TER against fluent references plus human ratings. Its examples call out "under-deletion" and over-compression, but there is no dedicated over-removal metric. It cites FER/DER — [arXiv 2605.12242](https://arxiv.org/abs/2605.12242)
- **PodcastFillers** (Zhu, Caceres, Salamon; Adobe Research; Interspeech 2022).
  - Data: 145 h, 35K fillers.
  - Metrics: filler detection scored with segment-based and event-based precision, recall and F1 (sed_eval, ±200 ms onset/offset collar). Precision is the "false cut" proxy.
  - Best results (AVC-FillerNet): event P 91.7 / R 94.0 / F1 92.8; ‘uh’ is harder (event F1 84.3) than ‘um’ (91.0). The main confusion is fillers vs real words.
  - [arXiv 2203.15135](https://arxiv.org/abs/2203.15135)

**Verbatim vs intended references (2026)**
- **Toyin, Umesh, Aldarmaki (MBZUAI/IIT Madras), "What Counts as an Error? Dual-Reference Benchmarking for Atypical ASR"** (arXiv 2606.31112, 30 Jun 2026).
  - Setup: 11 ASR models on FluencyBank Timestamped (3,430 samples), scored against **both** intended (isWER) and verbatim (vWER) references, plus SeMaScore.
  - Results:
    - Canary-1B-v2: isWER 13.85, vWER 21.95
    - Whisper-large-v3: isWER 16.13, vWER 25.01
    - NVIDIA CTC: isWER 27.43, **vWER 17.20 (best verbatim)**
  - Autoregressive models favour intended output; CTC models favour verbatim output.
  - Hardest events: multisyllable and incomplete-syllable repetitions. Prolongations are easiest.
  - Metric behaviour: BERTScore stays high (0.84) for clearly wrong hypotheses. SeMaScore separates meaning errors better but penalizes formatting variants ("20" vs "twenty").
  - Recommendation: intended-facing applications should report isWER **plus** a semantic metric.
  - [arXiv 2606.31112](https://arxiv.org/abs/2606.31112)
- **Wagner, Zusag, Thallinger (nyra labs, the CrisperWhisper authors), "Transcription Policy as a Latent Variable"** (arXiv 2607.18934, 21 Jul 2026).
  - Model: a Whisper-medium model with explicit `[verbatim]` / `[intended]` mode tags and supervised cross-attention word timing.
  - Proposes an **alignment-based evaluation that automatically extracts typed disfluency labels (CUTOFF, FILLER, SOUND, REPETITION)** and decomposes WER into content and style components. It reports that up to 60% of WER on AMI is style mismatch.
  - Results:
    - English disfluency event F1: 90.7%
    - Intended-mode insertion rate: 3–4% with tags vs >15% without
    - Word-boundary MAE on FluencyBank: **102 ms**, vs base Whisper 568 ms, MFA 142 ms and WhisperX 200 ms
    - Word-boundary MAE on TIMIT: 36 ms
  - [alphaXiv 2607.18934](https://www.alphaxiv.org/abs/2607.18934)
- **CrisperWhisper 2.0** (nyra labs, about June 2026; open weights on Hugging Face; commercial use requires a licence). VENDOR CLAIM:
  - Has verbatim (default) and `mode="intended"` transcription, word timestamps, and `verbatimize`.
  - Average disfluency F1 over 10 languages: 87.8 (Pro: 93.5), vs ElevenLabs Scribe v2 79.2 and Deepgram Nova-3 37.8.
  - Word-boundary error: about 30 ms on TIMIT and about 40 ms on Buckeye.
  - No streaming claim.
  - [nyra-labs.com/crisperwhisper](https://nyra-labs.com/crisperwhisper); [GitHub nyrahealth/CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper)
- **Sridhar & Wu (Interspeech 2025)** were the first to evaluate Whisper against both intended and verbatim references on stuttered speech; Whisper hallucinates on sound repetitions — cited in [Toyin et al. 2026](https://arxiv.org/abs/2606.31112); [ISCA](https://www.isca-archive.org/interspeech_2025/sridhar25_interspeech.html)
- **FluencyBank Timestamped** (Romana, Niu, Perez, Mower Provost; JSLHR 67, Oct 2024) provides **both verbatim and intended transcripts with word timestamps** for FluencyBank interviews — cited in [Toyin et al. 2026](https://arxiv.org/abs/2606.31112) and [arXiv 2610.19235](https://arxiv.org/abs/2610.19235)

**Stakeholder view on what to measure**
- **Toyin et al., "Aligning Stuttered-Speech Research with End-User Needs"** (arXiv 2604.20535, Apr 2026).
  - Method: a scoping review of 228 papers (2010–Oct 2025) and a survey of 40 people who stutter and 30 SLPs.
  - Survey findings:
    - People who stutter rated *intended* transcription more useful for communication, and SLPs preferred *verbatim*.
    - About 42% of respondents who stutter do not use voice AI at all, citing "impatient ASR."
    - Stuttering spikes in high-pressure contexts, so single global scores "risk underestimating real-world difficulty."
  - Literature findings:
    - "Speech Correction" is listed as a research area.
    - Fewer than 20% of papers involve stakeholders.
    - About 59% of stuttered-ASR work implicitly optimizes for intended output without saying so.
  - [arXiv 2604.20535](https://arxiv.org/abs/2604.20535)

### Inferences
- **StutterClear's metric (i), residual disfluencies via CrisperWhisper verbatim ASR of the output,** is new for stuttering. It generalizes TalkLess's transcript-based count and Das's filled pauses per minute.
  - Stronger version: report it per type (repetition, filler, prolongation if detectable) using nyra labs' alignment-based typed-label protocol (CUTOFF/FILLER/REPETITION).
  - Frame it as the audio-domain counterpart of **DER**.
- **Metric (iii), wrongly cut fluent words,** is the audio-domain counterpart of **FER**: the deletion part of FER, computed from word timestamps of the original verbatim transcript intersected with cut intervals.
  - Citing FER/DER gives the metric pair a recognized lineage, and reporting the pair shows the over-removal vs under-removal trade-off.
  - Caveat: word-timestamp error on stuttered speech is about 100 ms (nyra 2026, FluencyBank). That is 5 of StutterClear's 20 ms frames, so "fluent word cut" should use an overlap threshold or tolerance, for example "≥50% of the word's duration removed," and report sensitivity to it.
- **Metric (ii), meaning preservation as intended-mode WER of the edited audio vs intended-mode WER of the original,** is self-referenced, so it measures *change* caused by editing rather than absolute accuracy. Two recommended additions, following Toyin et al. 2026:
  - Also compute isWER of the edited audio against **human intended references from FluencyBank Timestamped**, which breaks the circularity of grading CrisperWhisper with CrisperWhisper.
  - Pair isWER with a semantic metric (SeMaScore preferred over BERTScore, which barely discriminates), as both STEAMROLLER and Toyin et al. do.
- **Comparability with prior work.** STEAMROLLER, StutterFormer and SpeechAgent all report ASR-WER of their *output audio* against a fluent reference. Reporting the same number (for example Whisper-large-v3 or Canary isWER of StutterClear output on FluencyBank) gives a direct comparison row. StutterClear's real-voice output should also score near-perfectly on objective speaker similarity, where STEAMROLLER's listener MOS-S was 3.4. Report ECAPA cosine to show this.
- **Listener check.** SpeechAgent's finding that cut-based baselines lowered clarity means objective metrics alone are not enough. A small listener C-MOS against the unprocessed audio is advisable.

### Gaps
- No paper was found that runs CrisperWhisper (or any verbatim ASR) on the *output* of a stutter-editing system to count residual events. This looks genuinely open, but alphaXiv and web search cannot rule out a non-indexed thesis or workshop paper.
- TalkLess's numeric results for the "Speech Disfluencies" metric were not retrieved; only the definition was.
- The FER/DER evaluator code was not inspected for reuse. CrisperWhisper 2.0's licence terms for academic evaluation use were not checked.
- How many raters SpeechAgent used, and the exact benchmark-suite contents, were not confirmed.
- No standardized stutter-to-fluent benchmark exists. FluencyBank Timestamped plus CASA clinical annotations (Valente et al., Interspeech 2025) is the most usable reference set found.

---

## Q4. Concrete numbers from the best prior work, for comparison tables

### Takeaway
No prior system reports latency in StutterClear's regime: continuous output under 1 s, own voice. The useful comparison rows are:
- STEAMROLLER: 3.77 s, resynthesized clone
- SpeechAgent: 0.91 s per 11.5 s utterance after it ends, generic TTS
- Streaming Parrotron: 320 ms after end of speech, synthetic voice
- CleanStream: live filler muting, no published numbers

For evaluation, the anchor numbers are:
- ASR-WER of outputs: STEAMROLLER 19.74% on cloned audio; StutterFormer 0.12–0.16 on FluencyBank
- FER/DER on Switchboard text
- PodcastFillers event F1 92.8 for filler detection
- Verbatim/intended WERs on FluencyBank Timestamped

### Cited Findings
| System (date, type) | Voice kept? | Latency | Key quality numbers | Source |
|---|---|---|---|---|
| STEAMROLLER (Jan 2026, RESEARCH) | No (StyleTTS2 clone) | 3.77 ± 0.62 s per 50–100 words | FluencyBank WER 31.60→18.69%; cloned-audio WER 19.74%; MOS-N 4.52; MOS-S 3.4; SBERT 0.91; SUS 66.9 | [arXiv](https://arxiv.org/html/2601.10223v1) |
| SpeechAgent (Oct 2025, RESEARCH) | No (ParaStyleTTS) | 0.91 s per 11.49 s utterance, after recording; RTF 0.08 | Stuttering clarity 4.50 vs 2.67 unprocessed; C-MOS +1.33 | [arXiv](https://arxiv.org/abs/2510.20113) |
| Cleanvoice (offline, SHIPPED), as rated in SpeechAgent | Yes | Offline | Stuttering clarity 2.50; C-MOS −0.17 | [arXiv](https://arxiv.org/abs/2510.20113) |
| Arjun 2020 AutomaticCorrection (offline, RESEARCH), as rated in SpeechAgent | Yes | Offline | Stuttering clarity 1.33; C-MOS −0.50 | [arXiv](https://arxiv.org/abs/2510.20113) |
| StutterFormer (Oct 2025, RESEARCH) | No (Griffin-Lim) | Not reported | WER 0.08 synthetic; 0.120/0.161 FluencyBank; BERTScore-P 0.94 | [arXiv](https://arxiv.org/html/2510.18938v1) |
| FluentSpeech (2023, RESEARCH) | Partly (diffusion-regenerated regions) | Not reported (offline) | MOS 4.27; SMOS 4.42; MCD 5.86; localization precision 94.4% | [ar5iv](https://ar5iv.labs.arxiv.org/html/2305.13612) |
| Streaming Parrotron (2023, RESEARCH; not stuttering) | No (fixed voice) | 320 ms after end of speech (quantized) | LibriSpeech WER about 15.4–15.9% | [ar5iv](https://ar5iv.labs.arxiv.org/html/2210.13761) |
| Simultaneous dysarthric reconstruction (APSIPA 2025, RESEARCH) | No (TTS) | 1.03 s response; RTF 0.71 (A100) | MOS 4.67; 54.25% relative WER reduction (UASpeech) | [arXiv 2603.01382](https://arxiv.org/abs/2603.01382) |
| CleanStream (2023–2025, OSS) | Yes (mutes or beeps) | Undisclosed delay | None published | [GitHub](https://github.com/royshil/obs-cleanstream) |
| Das et al. (2019, RESEARCH) | Yes (cuts fillers, shortens pauses) | Offline | Filled pauses/min 3.66→1.76; filler F1 0.955 | [arXiv](https://arxiv.org/pdf/1812.03415) |
| Detect-and-Pass (Amazon, 2022; ASR only) | No audio output | 100 ms frames, ±400 ms context | WER reduction 12.18–71.24% | [arXiv](https://arxiv.org/pdf/2202.05396) |
| Whispp (VENDOR CLAIM, 2026) | No (resynthesized from whisper) | <120 ms end-to-end, on-device | None independent | [whispp.com](https://www.whispp.com/) |

Evaluation anchors:
- **FER/DER** (Switchboard test, end-to-end model): 11.6 / 24.0 — [ar5iv](https://ar5iv.labs.arxiv.org/html/2009.10298)
- **Filler detection, PodcastFillers:** event P/R/F1 91.7 / 94.0 / 92.8 (AVC-FillerNet) — [arXiv](https://arxiv.org/abs/2203.15135)
- **FluencyBank Timestamped ASR:** best isWER 13.85 (Canary-1B-v2); best vWER 17.20 (NVIDIA CTC); Whisper-large-v3 16.13 / 25.01 — [arXiv 2606.31112](https://arxiv.org/abs/2606.31112)
- **Word-timestamp error on stuttered speech (FluencyBank):** 102 ms (nyra 2026) vs 142 ms (MFA) and 200 ms (WhisperX) — [alphaXiv 2607.18934](https://www.alphaxiv.org/abs/2607.18934)
- **Causal stutter-onset forecasting (300 ms horizon):** AUROC 0.63; AP 0.077 — [arXiv 2610.19235](https://arxiv.org/abs/2610.19235)
- **Annotation reliability ceiling:**
  - Fleiss κ: repetitions 0.54, prolongations 0.73, blocks 0.56 — [arXiv 2610.19235](https://arxiv.org/abs/2610.19235)
  - Annotators who stutter disagreed with more than 25% of labels in a widely used dataset — [Tang et al. FAccT '26](https://arxiv.org/abs/2605.02080)

### Inferences
- A clean StutterClear comparison table could use these rows:
  - unprocessed audio
  - CleanStream-style muting (re-implemented with StutterClear's detector)
  - Cleanvoice offline
  - STEAMROLLER-style cascade (reported numbers)
  - StutterClear at 250, 400 and 600 ms buffer
  
  And these columns:
  - added delay
  - residual disfluencies per 100 words (verbatim ASR)
  - fluent words cut (%)
  - isWER vs FluencyBank Timestamped intended reference
  - SeMaScore
  - ECAPA speaker cosine
  - listener C-MOS
- StutterClear's latency (250–600 ms continuous) would be about 6–15x lower than STEAMROLLER's 3.77 s. Unlike SpeechAgent's 0.91 s and Parrotron's 320 ms, it does not wait for the end of the utterance. Speaker similarity should be near ceiling by construction.
- Detector-quality targets: PodcastFillers-level filler F1 (about 90+) is a reasonable bar for fillers. For repetitions and prolongations, human agreement (κ about 0.5–0.7) caps what any detector can be shown to achieve.

### Gaps
- No prior work reports mouth-to-ear latency, buffer occupancy or drift for continuous real-voice editing, so there is nothing to compare StutterClear's latency against except utterance-level resynthesis systems.
- SpeechAgent's "Recover" metric was not defined in the retrieved pages.
- FluentSpeech's and StutterFormer's numbers come from different datasets (VCTK, SASE, synthetic SEP-28K pairs), so they are not directly comparable with FluencyBank-based results.
