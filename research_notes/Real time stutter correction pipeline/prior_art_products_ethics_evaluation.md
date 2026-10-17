# Prior Art, Products, Auditory-Feedback Risks, Ethics, and Evaluation Methodology for Stuttered-to-Fluent Speech Systems (StutterClear)

Research date: 17 Oct 2026. "OLDER" marks items from before 2023. "SHIPPED" means a product people can actually obtain or use. "RESEARCH" means a paper or demo. "VENDOR CLAIM" means a figure published by the company and not checked independently.

---

## Q1. Research systems that repair or reconstruct stuttered or atypical speech into fluent speech (speech-to-speech, editing, ASR->text->TTS, voice conversion)

### Takeaway
Only a few research systems actually output fluent audio from stuttered speech. All are recent (2023–2026), mostly offline, and none reports a full real-time mouth-to-ear latency budget:
- FluentSpeech: offline diffusion editing.
- StutterZero/StutterFormer: end-to-end waveform-to-waveform, no latency reported.
- STEAMROLLER: a cascade of Whisper, a GPT-4o multi-agent repair step and StyleTTS2 voice cloning. It takes about 3.8 s per segment and is the only one with a user study of people who stutter.

The best-engineered low-latency speech-to-speech prior art is Google's Parrotron family for atypical speech (dysarthria, ALS, deaf speech). It converts audio only after the speaker stops talking, with about 320–400 ms of post-utterance delay on a Pixel 4. The Lian/Anumanchipalli line (UDM, H-UDM, SSDM 2.0) is dysfluency-aware transcription and detection, not audio repair.

### Cited Findings

**Google Parrotron family (atypical-speech speech-to-speech; RESEARCH, never shipped as a standalone product)**
- **Parrotron (2019, OLDER).**
  - Architecture and training: an attention-based seq2seq network that maps input spectrograms straight to output spectrograms in a fixed synthetic voice. It is trained with a multitask objective that also predicts target phonemes, which the authors say gave "significant quality improvements." Base training used about 30,000 h of speech (millions of utterance pairs) whose targets were made by Parallel WaveNet TTS from transcripts. It was then fine-tuned on one person's own recordings: Dimitri Kanevsky gave 15 h and Aubrie Lee gave 1.5 h — [Google Research blog, Jul 2019](https://research.google/blog/parrotron-new-research-into-improving-verbal-communication-for-people-with-speech-impairments/)
  - Results: for Kanevsky (deaf speech), Google ASR WER on Parrotron's output fell from 89% to 32%. A cascaded Euphonia ASR reached 25% on the same speaker. Google argued that cascades can change meaning when the recognizer makes errors, while direct conversion stays acoustically close to the input. The post gives no MOS and no latency figures — [Google Research blog](https://research.google/blog/parrotron-new-research-into-improving-verbal-communication-for-people-with-speech-impairments/)
- **Conformer Parrotron (Interspeech 2021, OLDER).** It adds encoder subsampling with matching upsampling in the decoder. It reports up to 7x speedup and a 35% relative WER reduction over the earlier Transformer Parrotron — [Google Research](https://research.google/pubs/conformer-parrotron-a-faster-and-stronger-end-to-end-speechconversion-and-recognition-model-for-atypical-speech/)
- **Streaming Parrotron (Rybakov et al., arXiv Oct 2022; Interspeech 2023).**
  - Architecture: a mel frontend, then a streaming Conformer encoder, then an autoregressive LSTM decoder with cross-attention, then a streaming vocoder (Griffin-Lim or MelGAN).
  - Timing: the encoder runs while the user speaks. The decoder and vocoder start **after the user stops talking**. "Total delay" is defined as the time from end of speech to start of playback, measured on 10 s of input — [arXiv 2210.13761](https://arxiv.org/abs/2210.13761); [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2210.13761); [ISCA](https://www.isca-archive.org/interspeech_2023/rybakov23_interspeech.html)
  - Delay and WER: the non-streaming base model reaches 14.7% WER with about 7,900 ms total delay (the text also says 7.4 s in places). The streaming LSA_LS model reaches 15.3% WER at 400 ms. Streaming LSA reaches 16.4% at 440 ms. A fully causal model reaches 19.2% WER at 0 ms. With int8/int4 quantization, total delay is **320 ms** at about 15.4–15.9% WER — [ar5iv](https://ar5iv.labs.arxiv.org/html/2210.13761)
  - On-device numbers (Pixel 4 CPU): about 2x faster than real time. Each 25 ms chunk of decoder plus vocoder takes 11–16 ms. Each 80 ms encoder chunk takes 32–40 ms. The int8 encoder is 111 MB, or 70 MB with int4 weights.
  - Evaluation data: LibriSpeech, scored by ASR WER. It was not evaluated on atypical speakers in this paper — [ar5iv](https://ar5iv.labs.arxiv.org/html/2210.13761)

**Stuttered-to-fluent audio systems**
- **FluentSpeech (Jiang et al., ACL Findings 2023, arXiv May 2023; RESEARCH, offline).**
  - Architecture (23.9M parameters): a context-aware diffusion speech editor with 8 diffusion steps, a HiFi-GAN vocoder, and a stutter predictor that produces the edit mask automatically.
  - Data: the SASE dataset, about 40 h of spontaneous lecture speech from 46 speakers, labelled for repetitions, filled pauses and sudden sounds.
  - Stutter localization: 80.5% accuracy, 94.4% precision.
  - Editing results on VCTK: MCD 5.86, STOI 0.81, PESQ 1.91. Quality MOS 4.27 (seen) / 4.18 (unseen); speaker-similarity MOS 4.42 / 4.21. These beat EditSpeech, CampNet and A3T. CMOS ablations were run on SASE.
  - No inference speed is reported — [ar5iv 2305.13612](https://ar5iv.labs.arxiv.org/html/2305.13612)
  - The ethics statement warns of possible "unemployment for people with related occupations" and of the social harm of "free manipulation of speeches" — [ar5iv](https://ar5iv.labs.arxiv.org/html/2305.13612)
- **StutterZero and StutterFormer (arXiv 2510.18938, Oct 2025; RESEARCH).**
  - Claim: the first end-to-end waveform-to-waveform stutter-to-fluent models that also predict a transcript.
  - Architectures: StutterZero is a conv-BiLSTM encoder with spectrogram and grapheme decoders. StutterFormer is a dual-stream Transformer. Both use a Griffin-Lim vocoder.
  - Training data: about 34 h of synthetic pairs. Fluent targets were made by transcribing SEP-28K with fine-tuned Whisper-Small and re-synthesizing with MeloTTS, plus LibriStutter.
  - Results (WER / BERTScore-P, judged by Whisper-Small): StutterFormer 0.08 / 0.94; StutterZero 0.116 / 0.917; Whisper-Medium baseline 0.361 / 0.601. On unseen FluencyBank data: 0.120 and 0.161 WER.
  - Not reported: MOS, speaker similarity, latency. Speaker identity is not preserved, and a speaker/prosody encoder is listed as future work.
  - The paper's headline numbers are inconsistent: the abstract claims 24–28% WER reduction, while the table implies about 68–78% — [arXiv HTML](https://arxiv.org/html/2510.18938v1)
- **STEAMROLLER (arXiv 2601.10223; AAAI 2026; RESEARCH, cascaded, "real-time" in intent).**
  - Pipeline: Whisper-base ASR, then GPT-4o (three parallel Repair Agents plus a Master Agent, up to 3 iterations), then StyleTTS2 zero-shot voice cloning.
  - Latency: about 3 s for short dialogues; **3.77 ± 0.62 s per 50–100-word segment** in the user study. No per-stage or time-to-first-audio breakdown — [arXiv HTML](https://arxiv.org/html/2601.10223v1); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/41300)
  - Accuracy on FluencyBank: WER 31.60% → 18.69%; MER 27.42% → 17.59%; WIL 35.48% → 22.80%. Sentence-BERT similarity 0.81 → 0.91. WER on the cloned audio: 19.74%. Fine-tuning Whisper alone took WER from 21.53% to 18.32%. By severity, WER fell from 33.48% to 22.62% (severe) and from 14.16% to 11.01% (mild) — [arXiv HTML](https://arxiv.org/html/2601.10223v1)
- **Simultaneous dysarthric speech reconstruction (Wu et al., APSIPA ASC 2025, arXiv 2603.01382; RESEARCH).**
  - Design: end-to-end simultaneous ASR-to-TTS with a frame-level adaptor and multiple wait-k autoregressive TTS trained with knowledge distillation.
  - Results: average response time **1.03 s** on an A100, RTF 0.71, MOS 4.67, 54.25% relative WER reduction on UASpeech.
  - It explicitly says cascaded ASR plus sentence-level TTS adds latency because dysarthric speakers talk slowly — [arXiv 2603.01382](https://arxiv.org/abs/2603.01382)
- **Dysarthric reconstruction with speaker identity preserved (OLDER).** Disong Wang et al., "Speaker Identity Preservation in Dysarthric Speech Reconstruction by Adversarial Speaker Adaptation," ICASSP 2022 — [dblp listing](https://dblp.org/pid/168/4621). A related line is identity-preserving normal-to-dysarthric voice conversion — [arXiv 2110.08213](https://arxiv.org/pdf/2110.08213)

**Dysfluency-aware transcription and detection (front-end prior art, not audio repair)**
- **UDM (Lian et al., ASRU 2023) and H-UDM (EACL 2024).** Hierarchical, alignment-based dysfluency transcription and detection with the simulated VCTK++ corpus — [arXiv 2312.12810](https://arxiv.org/abs/2312.12810); [ACL Anthology](https://aclanthology.org/2024.eacl-long.32)
- **SSDM 2.0 (Lian, Zhou, ... Anumanchipalli; arXiv Nov 2024; UC Berkeley/UCSF).** Adds:
  - neural articulatory flow representations
  - a full-stack subsequence aligner covering all dysfluency types
  - mispronunciation prompting with LLM consistency learning
  - the Libri-Co-Dys co-dysfluency corpus

  It reports large gains over SSDM and other dysfluency transcription models, with clinical testing on nfvPPA speech. Its output is time-accurate "rich transcription," not fluent audio — [arXiv 2412.00265](https://arxiv.org/abs/2412.00265v1)
- **Dysfluent-WFST (Interspeech 2025).** A zero-shot decoder that transcribes phonemes and detects dysfluency together on top of WavLM, with no extra training — [ISCA](https://www.isca-archive.org/interspeech_2025/guo25d_interspeech.html) (described from search-result summary only)

**Related "fluent playback" research (an aid, not repair)**
- **Choral speech with a TTS "decoder" (Demirel, Journal of Fluency Disorders, 2025).** In a simulated presentation, speaking in unison with a pre-recorded own voice or with TTS lowered mean stuttering from about 10% to under 3% of syllables. One severe participant went from 49% to 2%. The study is preliminary — [BNDU Oxford](https://www.bndu.ox.ac.uk/papers/novel-use-choral-speech-significantly-reduces-stuttering-simulated-presentation-setting)

### Inferences
- No found prior system does streaming, word-by-word, voice-cloned stutter repair with sub-second mouth-to-ear latency. That gap is StutterClear's novelty space.
- STEAMROLLER is the closest architectural analogue: ASR, then LLM repair, then cloned TTS. Its utterance-level design produced latency of about 3.8 s, and users noticed it.
- Streaming Parrotron shows that even well-engineered on-device speech-to-speech ended up with about 320–400 ms of delay *after end of utterance*, not concurrent output. "Decode after the speaker stops" is a reasonable reference design for turn-based conversation.
- Parrotron's argument (direct conversion keeps acoustic closeness; cascades risk meaning changes) applies directly to StutterClear's LLM-correction stage. Meaning preservation must be measured, not assumed.
- Training targets in StutterFormer (MeloTTS) and Parrotron (Parallel WaveNet) were TTS-generated. That caps prosodic naturalness and the speaker-identity fidelity of end-to-end approaches.

### Gaps
- No paper was found that applies VoiceCraft-style (or similar zero-shot neural codec LM) speech editing specifically to repairing stuttered speech with measured latency.
- The numeric results of UDM, H-UDM and SSDM 2.0 were not retrieved; only abstracts were read.
- No Streaming Parrotron evaluation on actual atypical speakers was found. The streaming paper used LibriSpeech.
- The Parrotron 2019 paper's own MOS and intelligibility tables were not retrieved; the blog has none.

---

## Q2. Commercial products and apps: what they do, latency, pricing

### Takeaway
No shipped mainstream product converts stuttered speech into fluent speech in the user's own voice in real time.

What does exist:
- **Recognition-first AAC for dysarthria and similar conditions:** Voiceitt and Google Project Relate. Both re-speak in a synthetic voice and do not mention stuttering.
- **Voice banking and cloning for people losing their voice:** Apple Personal Voice, ElevenLabs Impact, Acapela, VocaliD. Stuttering is not an eligible condition where eligibility is stated.
- **Altered auditory feedback devices and apps:** SpeechEasy and various DAF apps.
- **Whisper-to-voice converters marketed at least partly to people who stutter:** Whispp, and Ayta AI. Whispp has since pivoted to B2B privacy calling and defers stuttering use pending medical validation.
- **Offline filler and stutter removers for recordings:** Descript, Cleanvoice.
- **A smartwatch metronome app whose ad drew community backlash:** Samsung Impulse.

### Cited Findings

**Speech recognition / AAC for atypical speech**
- **Voiceitt (SHIPPED, web app).**
  - What it does: recognizes non-standard speech. "Speak" mode outputs speech in a computerized voice; "Dictate" outputs text. It is positioned as AAC.
  - Who it names: stroke, ALS, Parkinson's, cerebral palsy, dysarthria, aging adults, accented speakers. **Stuttering is not mentioned.**
  - Integrations: Webex captions (paid add-on), Teams (paid Microsoft 365), Zoom (beta/"soon"), a ChatGPT toggle, earlier Alexa.
  - Setup: needs **50 recorded phrase cards** before first use.
  - Price and performance: 30-day free trial; no public pricing, no latency figures, no accuracy figures — [voiceitt.com](https://www.voiceitt.com/)
- **Google Project Relate (beta, launched Nov 2021, OLDER).**
  - Setup: users record **500 phrases**, quoted as 30–90 min ([Freethink](https://www.freethink.com/hard-tech/speech-impairments)) or 1–2 h across sessions ([9to5Google, Jan 2023](https://9to5google.com/2023/01/13/google-project-relate/)).
  - Features: Listen (live transcription), **Repeat** (restates the user's words "in a clear, synthesized voice", meant for face-to-face conversation), and Assistant — [Tom's Guide](https://www.tomsguide.com/uk/news/watch-google-unveils-new-ai-app-to-help-people-with-speech-impairments)
  - Availability: 18+, Android 8+, English in US/CA/AU/NZ — [Slashdot/The Verge](https://tech.slashdot.org/story/21/11/10/1532200/google-is-taking-sign-ups-for-relate-a-voice-assistant-that-recognizes-impaired-speech). Became more widely available on the Play Store in Jan 2023, still labelled beta — [9to5Google](https://9to5google.com/2023/01/13/google-project-relate/)
- **Apple Personal Voice, Live Speech, and Listen for Atypical Speech (SHIPPED, on-device).**
  - Personal Voice (iOS 17): reading about 150 phrases (about 15 min) creates the voice on-device — [Fox6/CNN coverage](https://www.fox6now.com/news/iphone-will-soon-be-able-to-replicate-your-voice-after-just-15-minutes-of-training); [Pratt IXD](https://ixd.prattsi.org/2024/02/assistive-technology-apples-live-speech-personal-voice/). A later update reportedly needs only 10 phrases and a few minutes — [TechRadar](https://www.techradar.com/phones/ios/apple-is-about-to-make-personal-voice-insanely-fast-and-update-almost-all-its-other-accessibility-features) (the Apple newsroom original was not verified).
  - Live Speech is **type-to-speak**, usable in phone and FaceTime calls and in person. It is not speech-to-speech.
  - Listen for Atypical Speech (iOS 18) targets CP, ALS and stroke — [Easterseals Tech](https://www.eastersealstech.com/2025/05/22/exploring-ios-18s-new-accessibility-features-for-speech/)

**Voice banking and cloning programs (consent-based personalized TTS)**
- **ElevenLabs Impact Program (SHIPPED).**
  - Offers free voice-cloning licenses to people with **permanent speech loss**, applied for through partner nonprofits. Patients and clinicians can now apply on the website — [ElevenLabs Impact](https://elevenlabs.io/impact-program); [ElevenLabs blog](https://elevenlabs.io/blog/expanding-access-patients-and-clinicians-can-now-apply-directly-on-the-elevenlabs-website)
  - Scale: about 7,000 people helped, 25 partners (per search summary of ElevenLabs pages).
  - Eligible conditions (third-party list): MND/ALS, PSP, MSA, CBD, stroke, Tay-Sachs and Sandhoff, head and neck cancer, laryngectomy and glossectomy. Stuttering is not listed — [Spoken AAC](https://spokenaac.com/blog/get-elevenlabs-access/)
- **Acapela My-Own-Voice (SHIPPED).**
  - Price: about €99/yr or €999 one-time, excluding VAT — [FitGap](https://us.fitgap.com/products/acapela). UK chart: about £90/yr or £895 — [MND Association chart Oct 2025](https://www.mndassociation.org/sites/default/files/public/2025-10/Voice%20banking%20services%20comparison%20chart%20-%20Oct%202025.pdf)
  - Since 14 July 2026 it is free with qualifying Tobii Dynavox AAC devices — [Tobii Dynavox](https://www.tobiidynavox.com/blogs/product-news/acapela-my-own-voice-now-included-at-no-additional-cost-with-qualifying-td-devices)
  - Acapela's own material is titled "Voice banking in 50 sentences" — [Acapela PDF](https://cdn.mediactive-network.net/events/vivatech/uploads/services/surcouche/my-own-voice-Voice-banking-in-50-sentences_1621495077_EJ89R4.pdf)
- **VocaliD (SHIPPED).** About $1,499 (about £1,135), with a $29.99 preview — [MND Association chart](https://www.mndassociation.org/sites/default/files/public/2025-10/Voice%20banking%20services%20comparison%20chart%20-%20Oct%202025.pdf). The voice arrives in about 2 business days — [U. Iowa Voice Banking Guide 2025](https://csd.uiowa.edu/sites/csd.uiowa.edu/files/2025-08/Voice_Banking_Guide.pdf)

**Altered auditory feedback devices and apps (SHIPPED, mostly OLDER)**
- **SpeechEasy (Janus Development Group).**
  - Mechanism: an in-ear device combining DAF and FAF — [CU Boulder 2004](https://www.colorado.edu/today/2004/12/07/cu-speech-pathologist-studies-device-helps-reduce-stuttering)
  - Settings: delay from about 20 ms to over 200 ms, plus a pitch shift. Fitting is a 2–3 h computerized evaluation — [SpeechPathology.com interview with Janus president](https://speechpathology.com/interviews/interview-with-alan-newton-president-1332)
  - Price (OLDER figures; no current official price found): $4,500–5,000 ([CU Boulder 2004](https://www.colorado.edu/today/2004/12/07/cu-speech-pathologist-studies-device-helps-reduce-stuttering)); about $4,300, not covered by insurance ([Stuttering Foundation/stutter.ca](https://stutter.ca/articles/2010/04/effects-speecheasy)); $2,000–8,000 per aid ([Yale essay](https://poorvucenter.yale.edu/sites/default/files/2025-03/Yup_HSHM497.docx.pdf))
  - Some clinic candidates got "minimal or no" benefit — [CU Boulder](https://www.colorado.edu/today/2004/12/07/cu-speech-pathologist-studies-device-helps-reduce-stuttering)
  - Anthem's medical policy rates such devices not medically necessary, citing small samples and no proof of long-term benefit — [Anthem policy](https://www.anthem.com/dam/medpolicies/abcbs/active/policies/mp_pw_a050289.html)
- **DAF smartphone apps.**
  - DAF Pro (iOS) supports Bluetooth but says **Bluetooth adds about 60 ms** — [App Store](https://apps.apple.com/app/id491299921)
  - qb DAF offers 50–75 ms delays and notes that Bluetooth alone imposes about 50–75 ms even at a 0 ms setting — [App Store](https://apps.apple.com/app/id1565802634)
  - Fonate DAF combines delay with a pitch change — [Fonate listing](https://appgoblin.info/apps/981000993)
  - DAF Assistant dates from 2012 — [stutter.ca](https://stutter.ca/articles/2012/04/delayed-auditory-feedback-daf-app-now-available-itunes)

**Whisper-to-voice and "fluent voice" startups**
- **Whispp (Leiden, NL; founded 2018 by Joris Castermans, who stutters).**
  - Approach (GSMA case study, July 2024): converts whispered or impaired speech to a natural voice in real time, audio-to-audio and phoneme-based ("language-independent"). The target was **<50 ms** conversion, versus 2–3 s for STT+TTS. It cites whispering as reducing stuttering "by up to 85%" — [GSMA Foundry case study, Jul 2024](https://gsma.com/get-involved/gsma-foundry/wp-content/uploads/2024/09/Giving-More-People-a-Voice.pdf) (VENDOR CLAIM)
  - Business model and approvals (same 2024 GSMA case study): freemium app at £20/month or £200/yr (360 min app-to-app plus 60 min to phone numbers). The Dutch government reimburses it for work use. It claimed medical-device approvals in the US and EU. At the time it was cloud-based, with on-device as the goal. Phone calling debuted at CES 2024 — [TechCrunch Jan 2024](https://techcrunch.com/2024/01/11/whispp-voice-app)
  - **Current website (2026):**
    - Rebranded as B2B "Voice Reconstruction AI" for private and anonymized calls, claiming **<120 ms end-to-end and 100% on-device** (Qualcomm, Arm, AMD).
    - Products: WhisppWAVE for mobile, PC and wearables.
    - Assistive use for people who stutter severely is described as a *future phase* "with the medical validation these users deserve."
    - TIME Best Inventions 2024; Forbes Accessibility 100 (2025).
    — [whispp.com](https://www.whispp.com/)
  - Conflict: the 2024 GSMA material describes a consumer medical device for stuttering with reimbursement, while the 2026 website describes stuttering use as not yet validated. The latency target also changed from <50 ms to <120 ms.
- **Ayta AI.**
  - Backed by inDrive founder Arsen Tomsky, who stutters; team based in Kazakhstan.
  - What it does: a desktop "real-time speech converter" that turns **whispering, singing or quiet speech** into "natural, smooth speech without stuttering." It runs as a virtual mic ("AytaAudioDevice") for Zoom, Teams, Meet, Skype, Discord and Telegram, and setup includes recording the user's normal voice.
  - Free download. No latency, pricing, evidence or privacy details published — [ayta.ai](https://ayta.ai/) (VENDOR CLAIM; status unverified)
- **Real-time voice changers marketed to people who stutter** (e.g., VoxBooster, about $6.99/mo).
  - The vendor's own blog says "Voice changers are not a solution to stuttering" and calls benefits a psychological accommodation not tested in trials.
  - Its latency claims are inconsistent: about 8 ms, 20 ms, 30 ms and 0.5 s in different places — [VoxBooster blog](https://voxbooster.com/blog/voice-changer-stuttering-speech-accommodation/) (marketing)

**Offline disfluency removal (SHIPPED, not real-time)**
- **Descript** flags filler words (plus repeated words on some plans) for one-by-one review and deletion. An "Avoid harsh cuts" option skips risky edits. Descript itself warns against blanket removal — [Descript help](https://help.descript.com/hc/en-us/articles/10164806394509-Remove-filler-words); [Descript blog](https://www.descript.com/blog/article/how-to-remove-filler-words-but-know-which-to-keep)
- **Cleanvoice "Stutter Remover"** works on uploaded recordings only. It edits out stutters "without cutting the word out." First 30 min free; no accuracy figures — [cleanvoice.ai](https://cleanvoice.ai/stutter-remover/)

**Other stuttering-targeted AI**
- **Samsung "Impulse"** (Cheil Spain; Galaxy Watch app). It uses NLP to turn words into rhythmic vibrations that pace speech. Its ad drew backlash for implying it "fixes" stuttering, and Samsung/Cheil re-edited the ad — [KCRG Feb 2025](https://www.kcrg.com/2025/02/09/tv9-update-concerns-over-samsung-impulse-app-sparks-changes/); [SPACE](https://www.spacetostutter.org/impulse). It still won Cannes Lions Gold 2024 — [Cheil](https://cheil.com/us/news/newsDetail?nsNo=1202)
- **Fluent (Stony Brook, 2021, OLDER).** An AI writing tool that flags words a user may stutter on and suggests substitutes. The authors frame it as helping users speak more fluently, and note that word substitution itself causes stress — [arXiv 2108.09918](https://arxiv.org/pdf/2108.09918); [Stony Brook](https://ai.stonybrook.edu/about-us/News/fluent-first-smart-writing-tool-people-who-stutter)

### Inferences
- The commercial pattern so far is: (a) enroll the user (50–500 phrases) to *recognize* atypical speech and re-speak it in a generic or banked voice, or (b) change the *input modality* (whisper, singing) and resynthesize.
- No vendor publishes a "stutter in, fluent own-voice out, streaming" product with measured latency. Whispp's "<50 ms" or "<120 ms" figure is the only sub-200 ms claim, and it is for whisper conversion, not disfluency removal. A one-to-one acoustic mapping can be that fast; removing a repetition requires lookahead or buffering.
- Voice-banking programs already set a precedent for a consent-gated, clinician-routed pipeline from cloned voice to AAC device. ElevenLabs gates by "permanent speech loss," so people who stutter would not currently qualify for free licenses.
- Whispp's apparent retreat from medical/assistive positioning suggests regulatory or validation cost is a real barrier for stuttering-specific claims.

### Gaps
- Current SpeechEasy pricing and availability, and whether Janus still sells it, could not be confirmed.
- Project Relate's 2025–2026 status (active or discontinued) was not confirmed.
- No independent latency measurements were found for Whispp, Ayta, DAF apps or voice changers.
- Voiceitt pricing and any accuracy or latency data are not public.
- Apple's own newsroom figure for the reduced Personal Voice enrollment was not verified.

---

## Q3. Delayed and altered auditory feedback (DAF/FAF): effect on stuttering, and the delays at which hearing your own (or a converted) voice becomes disruptive

### Takeaway
Hearing one's own speech delayed by about **50–75 ms** reduces stuttering in lab settings, typically by **about 50–75%**. Real-world, long-term effects are weak or not maintained.

For fluent speakers, **about 200 ms** is the delay of maximal disruption; 50 ms is detectable but not maximally disruptive. Own-voice delays above about 15–20 ms are judged disturbing in hearing-aid research, though speech production is unaffected below about 30 ms.

Hearing a *fluent* version of speech in unison ("choral speech") is even more fluency-inducing, up to about 98% in older work and 10% → under 3% in a 2025 study using own-voice recordings and TTS.

So any StutterClear mode that feeds corrected audio back into the user's ears will act as a DAF/FAF/choral stimulus. That can help some users, disrupt others, and it confounds measurement.

### Cited Findings
**People who stutter**
- About 60–75% reductions in stuttering frequency under altered auditory feedback, including 50 ms DAF, with monaural and binaural delivery (Stuart, Kalinowski, Rastatter 1997, n=11) — [Sonicfield library summary](https://labs.sonicfield.org/library/effect-of-monaural-and-binaural-altered-auditory-feedback-on-stuttering-frequenc); [JSLHR](https://pubs.asha.org/doi/full/10.1044/jslhr.4005.1130)
- Telephone use (1997, n=9): FAF cut stuttering by 55% and DAF by 60% (per search summary of the same literature set) — [ECU scholarship](https://thescholarship.ecu.edu/items/8b777fda-a024-45d3-88d2-331b5520bc16/full)
- MacLeod, Kalinowski, Stuart et al. (n=10): all AAF conditions significantly reduced stuttering at both normal and fast speech rates. There were no significant differences between AAF types — [ECU scholarship](https://thescholarship.ecu.edu/items/8b777fda-a024-45d3-88d2-331b5520bc16/full)
- A recent study of 22 adults: AAF lowered stuttering on average by 54% and 52% across two sessions. The effect was statistically significant only at the 65 ms and 75 ms DAF settings — [J. Disability Studies](https://jdisabilstud.org/article-1-377-en.html)
- Children aged 6–8: the effect was limited at 50–100 ms and clearer at 200 ms. The abstract is garbled, so this is low confidence — [Magiran](https://magiran.com/paper/1468952/?lang=en)
- **Real-world and long-term durability**
  - Pollard et al. 2009 (JSLHR 52:516–533; n=11; 6-month naturalistic ABA trial of SpeechEasy). Critics summarize it as finding **no significant treatment effect on stuttering frequency** but positive self-reports — [Stuttering Foundation summary](https://stutteringhelp.org/node/551); [DOI](https://pubs.asha.org/doi/10.1044/1092-4388(2008/07-0204)); [critique letter, 2010](https://pubs.asha.org/doi/10.1044/1092-4388%282010/10-0050%29)
  - Longitudinal naturalistic studies find less consistent effects, and initial reductions are often not maintained (per 2010 ASHA-journal discussion) — [ASHA](https://pubs.asha.org/doi/10.1044/1092-4388%282010/10-0050%29)
- **Whispering:** cited as reducing stuttering "by up to 85%" (VENDOR CLAIM) — [GSMA/Whispp](https://gsma.com/get-involved/gsma-foundry/wp-content/uploads/2024/09/Giving-More-People-a-Voice.pdf)
- **Choral speech**
  - Visual choral speech (watching silent mouthing in unison) cut stuttering by about 80% (Kalinowski et al., 2000) — [ECU](https://thescholarship.ecu.edu/items/99249d91-6d68-45e0-a197-286200803b95/full)
  - Linguistically matched mouthing cut it by 71% and mismatched mouthing by 35% (Dayalu et al., 2003) — [IMRF abstract](https://imrf.mcmaster.ca/2003/viewabstract.php%3Fid=61&symposium=0.html)
  - Choral speech up to about 98% versus about 68% for AAF (Saltuklaroglu et al. 2009, cited secondhand) — [Demirel 2025 JFD PDF](https://www.mrcbndu.ox.ac.uk/sites/default/files/demirel2025Journal_of_Fluency_Disorders.pdf)
  - Own-recording and TTS choral conditions in a simulated presentation: about 10% → under 3% — [BNDU Oxford](https://www.bndu.ox.ac.uk/papers/novel-use-choral-speech-significantly-reduces-stuttering-simulated-presentation-setting)

**Fluent speakers (disruption thresholds)**
- About 200 ms DAF gives maximal disruption in adults. Faster talkers are disrupted most at shorter delays, and the peak delay shifts with age — [Wikipedia: DAF](https://en.wikipedia.org/wiki/Delayed_auditory_feedback) (secondary)
- Stuart et al. 2002 tested fluent speakers at 0, 25, 50 and 200 ms: 2–3x more dysfluencies at 200 ms than at 0–50 ms — [Sonicfield summary](https://sonicfield.org/library/effect-of-delayed-auditory-feedback-on-normal-speakers-at-two-speech-rates)
- 50 ms is detectable but not maximally disruptive (Black 1951, cited secondhand). 200 ms is a "critical interval" regardless of rate — [UCL / Scott group paper](https://discovery-pp.ucl.ac.uk/10050342/1/Scott_Group%20and%20individual%20variability%20in%20speech%20production%20networks%20during%20delayed%20auditory%20feedback_VoR.pdf)
- EEG: speech-induced suppression of the auditory response was significant only at the 100 and 200 ms delays — [Frontiers Hum. Neurosci. 2020](https://public-pages-files-2025.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2020.00150/text)
- **Own-voice processing delay (hearing-aid literature)**
  - Stone & Moore (1999): delays become disturbing above about 20 ms. Stone & Moore (2002): about 15 ms in "dry" rooms and about 20 ms in "live" rooms. Speech production was unaffected until the delay exceeded 30 ms — [Canadian Audiologist (Moore feature)](https://canadianaudiologist.ca/moore-feature-09/?output=pdf); [Goehring et al. manuscript](https://eprints.soton.ac.uk/414900/1/Goehring_DelayTolerance_Manuscript.docx)
  - Comb-filter coloration from mixing the direct and processed paths can start at about 5–7 ms — [UMD hearing-aid delay review](https://hesp.umd.edu/sites/hesp.umd.edu/files/pubs/Hearing%20Aid%20Delay%20Effects.pdf)

### Inferences
- **Design implication 1 (monitoring path).**
  - If the corrected or cloned voice is played into the *user's* headphones while they speak, end-to-end delays of 50–200 ms fall exactly in the DAF band. The output is also *fluent* speech that differs from what the user is producing, closer to choral or FAF stimulation.
  - That may reduce stuttering for some users (a possible benefit) but disrupt others. For listeners without a stutter-like DAF sensitivity, about 200 ms is the worst case.
  - Safest default: send corrected audio **only to the far end**. Make self-monitoring an opt-in setting with a user-selectable delay, mute or attenuation.
- **Design implication 2 (sidetone).** Any local monitoring of the user's own raw voice should stay under about 10–15 ms to avoid hearing-aid-style "disturbing" delay and comb filtering. This is out of reach over Bluetooth, which adds about 50–75 ms by vendor accounts, so wired or no sidetone is preferable.
- **Design implication 3 (evaluation confound).** Stuttering frequency measured *while using* StutterClear with self-monitoring may drop because of DAF or choral effects, not correction quality. Evaluations should include a "far-end only" condition. They should also track carryover and novelty effects over weeks, since AAF effects tend to fade in the real world.
- The literature shows no single optimal delay. Effects are individual: a 2–3 h fitting for SpeechEasy, and big variance in responders. That supports per-user settings.

### Gaps
- No study was found that measures how people who stutter respond to hearing a *real-time AI-corrected fluent version of their own cloned voice* while speaking. This is a novel stimulus that combines DAF, FAF and choral properties.
- No primary source on conversational one-way delay tolerance (e.g., ITU-T G.114) was retrieved here. Far-end listener tolerance should come from the latency research notes.
- The original Lincoln, Packman & Onslow review of AAF and the Kalinowski 1993/1996 primary papers were not retrieved directly; the percentages above come from summaries.

---

## Q4. Ethics and community perspective: "fixing" stuttered speech, stuttering pride, masking, user acceptance, user control

### Takeaway
The organized stuttering community increasingly frames stuttering as a valid way of speaking, through stuttering pride and stutter-affirming therapy. It pushes back on products and marketing that promise to "fix" it, as the Samsung Impulse ad backlash in 2024–25 showed.

Yet people who stutter also clearly want speech technology that works for them (ASR that doesn't cut them off). In the one direct user study of a fluency-repair system (STEAMROLLER), they rated it positively, but with complaints about voice warmth and latency.

Researchers in the field (Wu/AImpower, FAccT 2026) recommend shifting the goal from "promoting fluency" to "supporting comfort and dignity", with options such as verbatim output and leadership by people who stutter in design and data.

No official STAMMA, NSA or ISA position statement on AI fluency conversion was found.

### Cited Findings
**Community framing**
- Stuttering pride is a social movement that treats stuttering as a legitimate speech pattern rather than a defect. It draws on the social model of disability and neurodiversity — [Wikipedia: Stuttering pride](https://en.wikipedia.org/wiki/Stuttering_pride)
- Its key text is *Stammering Pride and Prejudice: Difference Not Defect* (Campbell, Constantino, Simpson, 2019) — [cited in Tang et al. FAccT '26](https://arxiv.org/abs/2605.02080)
- Constantino's Stutter-Affirming Therapy aims to reduce "stutterphobic" reactions rather than to prioritize fluency. He argues that pursuing a false fluent identity forfeits well-being gains — [Stuttering pride search summary / ASHA Voices podcast](https://leader.pubs.asha.org/do/10.1044/2022-0707-transvoices-podcast-constantino-stuttering-neurodiversity)
- Tichenor, Constantino and Yaruss, "A Point of View About Fluency" (JSLHR 65(2), 2022), argue that "fluency" excludes many people who stutter, forces a false fluent/stuttered binary and fuels stigma. They recommend saying "stuttering," not "fluency disorder" — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9135003)
- Nuance: Constantino et al. (2017) describe "passing as fluent" as a form of stuttering produced through communication technologies, and as active resistance rather than repression — [Memphis Digital Commons](https://digitalcommons.memphis.edu/facpubs/10375). A small thesis found that concealment promotes feelings of inauthenticity and openness promotes authenticity among adults who stutter — [WMU thesis](https://scholarworks.wmich.edu/cgi/viewcontent.cgi?article=6375&context=masters_theses)

**"Cripping AI" (Tang, Lin, Li, Wu; FAccT '26, arXiv May 2026)**
- Speech AI "optimized for fluency and speed" cuts off or misunderstands people who stutter.
- Speech systems built around fluency include therapy that corrects disfluency and "captioning systems that remove stuttered utterances by default." The latter cites FluentSpeech.
- Constantino quote: "Our pauses, hesitations, and silences carry semantic weight; they are meaningful and purposeful."
- Recommendation: speech systems for people who stutter "should shift goal from promoting fluency toward supporting the speaker's comfort and dignity". "Many PWS prefer having an option of verbatim captioning."
- Annotators who stutter disagreed with labels in **more than 25% of clips** in a highly cited English stuttered-speech dataset. Confronting one's own stuttering during annotation or evaluation imposes emotional labor.
- [arXiv 2605.02080](https://arxiv.org/abs/2605.02080)

**Backlash case: Samsung Impulse**
- University of Iowa Stuttering Lab director Naomi Rodgers objected to messaging implying the app "fixes stuttering" — [KCRG](https://www.kcrg.com/2025/02/09/tv9-update-concerns-over-samsung-impulse-app-sparks-changes/)
- SPACE (Canada) organized an open letter with **99 organizational and 828 individual signatories**. Samsung/Cheil re-edited the already-distributed ad to remove "harmful and stigmatizing elements" — [SPACE](https://www.spacetostutter.org/impulse)

**User needs and acceptance studies**
- **Apple CHI 2023 (Lea et al.).** Surveyed 61 people who stutter, then recorded 91 participants. People who stutter *want* to use speech recognition but are often cut off or misrecognized. The technical fixes cut WER from 25.4% to 9.9% and reduced cut-offs (up to 79% per secondary summaries) — [arXiv 2302.09044](https://arxiv.org/pdf/2302.09044); [Apple ML Research](https://machinelearning.apple.com/research/user-perceptions-technical)
- **STEAMROLLER user study (n=23 adults who stutter; 7 mild, 12 moderate, 4 severe; 16 needs interviews).**
  - Ratings (5-point): overall satisfaction 4.30; meaning preservation 4.13; "reduced stuttering" 4.70.
  - Voice: MOS-S (similarity to the speaker) only 3.4.
  - Usability: SUS 66.9 ("acceptable"), NPS 23.53.
  - Complaints: 7/23 said the synthetic voice lacked warmth or did not resemble them; 2 said latency could disrupt spontaneous conversation.
  - Users valued seeing the original and repaired text and audio side by side.
  - [arXiv HTML](https://arxiv.org/html/2601.10223v1)
- **Shaomei Wu's HCI work.**
  - CHI 2023, "'The World is Designed for Fluent People': Benefits and Challenges of Videoconferencing Technologies for People Who Stutter" — [ACM](https://dl.acm.org/doi/fullHtml/10.1145/3544548.3580788)
  - DIS 2024 co-design of inclusive videoconferencing with people who stutter (honorable mention) — [Cornell Chronicle](https://www.news.cornell.edu/stories/2024/07/redesigning-videoconferencing-and-people-who-stutter)
  - Interspeech 2025 (Sridhar & Wu), by researchers who stutter: Whisper hallucinates **over 20% of the time** on sound repetitions — [ISCA](https://www.isca-archive.org/interspeech_2025/sridhar25_interspeech.html)
- **HeardAI (MSU/WMU, NSF; about $5M).** Lists STAMMA and the NSA as community partners. Its goal is making voice AI *accessible* to people who stutter, not making their speech fluent. An NSA chapter leader says current voice AI makes people who stutter feel "discriminated against by the technology itself" — [MSU Engaged Scholar](https://engagedscholar.msu.edu/enewsletter/volume16/issue2/voice-ai.aspx); [MSU Engineering](https://engineering.msu.edu/news/nihar-mahapatras-heardai-team-secures-5m-grant-help-people-who-stutter)
- **STAMMA (UK)** ran AImpower's July 2025 call for stammered-speech data but stated the project "is not related to STAMMA". It advised participants to read the consent form and tick only items they are comfortable with. No STAMMA view on AI was given — [STAMMA news](https://stamma.org/news/company-seeks-stammering-samples-train-ai)

**Voice-cloning ethics**
- A study of 26 participants with their own Speechify and ElevenLabs clones found some felt accent was "put through a filter" ("It's not a representation of me"). ElevenLabs clones were judged more realistic *because* they included human traits such as stutters, mistakes and breaths — [arXiv 2504.09346 (FAccT 2025)](https://arxiv.org/html/2504.09346v1)
- Listeners judged an AI clone to be the same person as the real speaker about 80% of the time, and correctly spotted AI voices only about 60% of the time — [Sci. Reports 2025 / arXiv 2410.03791](https://www.nature.com/articles/s41598-025-94170-3)
- A 2023 position paper proposes a "Disfluency Instrumentality Audit" for the ethics of using disfluency in AI speech — [SciTePress](https://www.scitepress.org/Documents/2023/119872)

### Inferences
- **Recommended user controls**, derived from the above:
  - Per-call or per-context on/off switch.
  - Repair "strength" settings (e.g., remove only blocks and long prolongations vs. also fillers vs. verbatim passthrough).
  - Own voice vs. a neutral voice.
  - Side-by-side review of original and repaired output (a stated need in STEAMROLLER).
  - Never auto-enable.
  - Never market as a cure or "fix." The Impulse case shows this triggers organized backlash.
- **Disclosure.** Listeners cannot reliably tell clones from real voices (about 60% detection). StutterClear should decide on a disclosure policy (e.g., optional "AI-assisted voice" indicator) and on safeguards against misuse of the consented clone.
- **Positioning.** Present StutterClear as an *optional accommodation* chosen by the speaker, similar to how "passing" can be framed as agency, not as normalization. Include people who stutter (and SLPs who stutter) on the design and evaluation team, and pay them for the emotional labor of evaluating their own speech.
- **Over-cleaning risk.** Over-cleaning can strip meaningful pauses and paralinguistic cues. STEAMROLLER itself warns that removing fillers and repetitions may alter tone, pitch and rhythm ([arXiv HTML](https://arxiv.org/html/2601.10223v1)), so "meaning preservation" should cover prosodic intent, not just words.

### Gaps
- No official position statements on AI fluency conversion or voice cloning were found from STAMMA, the NSA, ISA, SAY or the Stuttering Foundation. Their sites should be checked directly.
- No peer-reviewed interview or survey study was found that asks people who stutter specifically about *real-time fluent voice-clone output* (identity, authenticity, disclosure preferences), beyond STEAMROLLER's usability study.
- No data were found on listener (far-end) attitudes when they learn the speaker used AI fluency conversion.

---

## Q5. Evaluation methodology: metrics used for detection, transcription, correction or meaning, TTS, latency and user satisfaction, and which were combined in prior work

### Takeaway
Prior work evaluates fragments, never the whole chain:
- **Parrotron:** ASR-WER of the converted audio plus latency, RTF and model size.
- **FluentSpeech:** acoustic objective metrics (MCD, STOI, PESQ) plus MOS, SMOS and CMOS, plus localization accuracy.
- **StutterFormer:** WER, CER and BERTScore only.
- **Clinical AAF work:** %SS (percent syllables stuttered) plus OASES.
- **STEAMROLLER:** the broadest single combination — WER, MER and WIL; Sentence-BERT similarity; ASR-WER on cloned audio; MOS-N; listener-rated MOS-S; Likert ratings of meaning preservation; SUS and NPS; seconds-per-segment latency.

No prior stutter-repair paper reports objective speaker similarity (ECAPA or WavLM cosine), UTMOS, time-to-first-audio or mouth-to-ear latency, or NASA-TLX. StutterClear's proposed metric suite would be more complete than anything published.

### Cited Findings
**(a) Stutter detection**
- **SEP-28k (Lea et al., ICASSP 2021, OLDER):** 28k+ podcast clips with five event types. ConvLSTM baseline F1 is about 55.9 (block), 68.5 (prolongation), 63.2 (sound repetition), 60.4 (word repetition) and 71.3 (likely interjection; mapping from a secondary review). More training data gave +28% and +24% relative F1 on SEP-28k and FluencyBank — [arXiv 2102.12394](https://arxiv.org/pdf/2102.12394); [review arXiv 2107.04057](https://arxiv.org/pdf/2107.04057)
- **FluentSpeech stutter localization:** frame-level accuracy 80.5%, precision 94.4% — [ar5iv](https://ar5iv.labs.arxiv.org/html/2305.13612)
- **Label reliability:** annotators who stutter disagreed with more than 25% of labels in a widely used dataset, so the "ground truth" for per-class F1 is contested — [Tang et al. FAccT '26](https://arxiv.org/abs/2605.02080)

**(b) Transcription**
- WER, MER, WIL ([STEAMROLLER](https://arxiv.org/html/2601.10223v1)); CER ([StutterZero](https://arxiv.org/html/2510.18938v1))
- **Cut-off rate** (endpointing truncation) and WER, used by Apple CHI 2023 — [arXiv 2302.09044](https://arxiv.org/pdf/2302.09044)
- **Hallucination rate** on repetitions (over 20% for Whisper) — [Sridhar & Wu, Interspeech 2025](https://www.isca-archive.org/interspeech_2025/sridhar25_interspeech.html)
- Syntactic and semantic error analysis across six ASR systems showing bias against disfluent speech (NAACL 2024, "Lost in Transcription") — [ACL Anthology](https://aclanthology.org/2024.naacl-long.269.pdf)

**(c) Correction quality and meaning preservation**
- Sentence-BERT cosine between reference and repaired text: 0.81 → 0.91 — [STEAMROLLER](https://arxiv.org/html/2601.10223v1)
- BERTScore precision (StutterFormer 0.941; ASR-cascade 0.952) — [StutterZero/StutterFormer](https://arxiv.org/html/2510.18938v1)
- Human 5-point ratings: "meaning preservation" 4.13, "disfluency correction" 4.35, "minimal editing needed" 3.74 — [STEAMROLLER](https://arxiv.org/html/2601.10223v1)

**(d) TTS and conversion quality**
- MOS (naturalness) and speaker-similarity MOS (SMOS) on seen and unseen speakers, plus CMOS ablations, plus MCD, STOI and PESQ on edited regions — [FluentSpeech](https://ar5iv.labs.arxiv.org/html/2305.13612)
- **ASR-WER of the converted or synthesized speech as an intelligibility proxy:**
  - Parrotron: 89% → 32% — [Google blog](https://research.google/blog/parrotron-new-research-into-improving-verbal-communication-for-people-with-speech-impairments/)
  - Streaming Parrotron: 14.7–19.2% — [ar5iv](https://ar5iv.labs.arxiv.org/html/2210.13761)
  - STEAMROLLER cloned audio: 19.74% — [arXiv HTML](https://arxiv.org/html/2601.10223v1)
- MOS-N 4.52 and a listener-rated MOS-S 3.4 for StyleTTS2 clones — [STEAMROLLER](https://arxiv.org/html/2601.10223v1)
- MOS 4.67 for simultaneous dysarthric reconstruction — [arXiv 2603.01382](https://arxiv.org/abs/2603.01382)
- Speaker similarity in that line of work is either a listener MOS-S (STEAMROLLER) or "not measured" (StutterFormer). No ECAPA or WavLM cosine was reported in any stutter-repair paper found — [STEAMROLLER](https://arxiv.org/html/2601.10223v1); [StutterFormer](https://arxiv.org/html/2510.18938v1)

**(e) Latency**
- "Total delay," measured from end of speech to playback start, in ms. Also per-chunk compute latency, RTF, model size and memory on Pixel 4 — [Streaming Parrotron](https://ar5iv.labs.arxiv.org/html/2210.13761)
- Seconds per 50–100-word segment (3.77 ± 0.62 s) — [STEAMROLLER](https://arxiv.org/html/2601.10223v1)
- Average response time 1.03 s and RTF 0.71 — [simultaneous dysarthric speech reconstruction](https://arxiv.org/abs/2603.01382)
- Vendor end-to-end claims: <50 ms target (2024) and <120 ms (2026) for Whispp — [GSMA](https://gsma.com/get-involved/gsma-foundry/wp-content/uploads/2024/09/Giving-More-People-a-Voice.pdf); [whispp.com](https://www.whispp.com/)

**(f) User satisfaction and impact**
- SUS (66.9), NPS (23.53), Likert satisfaction with Cronbach's α 0.73, and a perceptual check (12 people who stutter judged the fluency gains real, not placebo) — [STEAMROLLER](https://arxiv.org/html/2601.10223v1)
- **OASES** (Overall Assessment of the Speaker's Experience of Stuttering)
  - Covers general information, reactions, daily communication and quality of life. Impact score runs 1–5, higher meaning worse — [Pearson](https://www.pearsonclinical.in/solutions/overall-assessment-of-the-speakers-experience-of-stuttering-oases); [ClinicalTrials NCT03437512](https://clinicaltrials.gov/study/NCT03437512)
  - Used as a *secondary* outcome alongside %SS or the Stuttering Severity Instrument in trials of tDCS and virtual-reality interventions — [NCT03335722](https://clinicaltrials.gov/study/NCT03335722); [NCT07833969](https://clinicaltrials.gov/study/NCT07833969)
  - In one tDCS trial, OASES fell in both active and sham groups with no difference between them — [SAP](https://cdn.clinicaltrials.gov/large-docs/22/NCT03335722/SAP_003.pdf)
- Self-report versus objective mismatch: SpeechEasy users reported positive impressions without a measurable fluency change in naturalistic use — [Stuttering Foundation summary](https://stutteringhelp.org/node/551); [ASHA critique](https://pubs.asha.org/doi/10.1044/1092-4388%282010/10-0050%29)

### Inferences
- **Recommended combined evaluation for StutterClear**, building on the STEAMROLLER template and filling its gaps:
  - **Detection:** per-class F1 on SEP-28k/FluencyBank, plus agreement checks by annotators who stutter.
  - **Transcription:** WER/CER on stuttered speech, plus hallucination rate and cut-off rate.
  - **Meaning:** BERTScore and/or SBERT cosine, plus human "meaning preserved / intent preserved" ratings by the *speaker*.
  - **Voice:** UTMOS or MOS-N, ECAPA or WavLM speaker-embedding cosine, *and* speaker-rated similarity (STEAMROLLER showed listener similarity at only 3.4/5 despite high naturalness), plus ASR-WER of the output.
  - **Latency:** per-stage timings, time-to-first-audio and end-to-end mouth-to-ear, reported both as a Parrotron-style "end-of-utterance-to-playback" delay and as continuous lag.
  - **User outcomes:** SUS, NASA-TLX (cognitive load of monitoring corrections), and OASES before/after over weeks. Include a far-end-only vs. self-monitoring condition to separate DAF/choral effects.
- Self-report can diverge from objective change (SpeechEasy). StutterClear studies should pre-register both and interpret positive satisfaction cautiously.

### Gaps
- No use of NASA-TLX in stuttering-technology studies was found.
- No stutter-repair or atypical speech-to-speech paper was found that reports objective speaker-embedding similarity, UTMOS or mouth-to-ear latency.
- Original SEP-28k results table not verified first-hand; per-class F1 values come from a review.
- No standardized benchmark exists for "stuttered-to-fluent conversion." Each paper uses its own data (synthetic SEP-28K/LibriStutter pairs, FluencyBank subsets, SASE), so cross-paper comparison is not meaningful.
