# StutterClear Can Claim Evidence, Not Ideas

As of 17 October 2026, almost none of StutterClear's eight planned contributions is new as an idea, but most can still be new as a working, measured system. The one clean first is the **lookahead-versus-accuracy curve for stuttering detection** from a distilled multi-head causal student. The notes found no paper that measures detection accuracy at several lookaheads, no distillation from a large self-supervised teacher into a tiny causal stuttering model, and no causal result on any frame-level benchmark ([Kumar et al. 2026](https://arxiv.org/abs/2610.19235); [Kozak 2026](https://arxiv.org/abs/2604.27279)).

The flagship idea, cutting stutters out of the real voice before a listener hears it, already appears in patents. IBM filed one in 2012, which Google Patents lists as active until about 2032, and an IBM/Nuance patent from 2011 has since lapsed. Neither reports an implementation, and the only live tool found, CleanStream, mutes fillers rather than splicing them out ([US8682678B2](https://patents.google.com/patent/US8682678B2/en); [US8600758B2](https://patents.google.com/patent/US8600758B2/en); [CleanStream](https://github.com/royshil/obs-cleanstream)). Blocked-word prediction is anticipated in the same way by Intel's active patent, which already combines the preceding words, a personal language model and the stutter sound ([US10672393B2](https://patents.google.com/patent/US10672393B2/en)). Combining synthetic frame labels with clip-level multiple-instance learning (MIL) was published in 2022 and 2024. The silver-label contribution must therefore be narrowed to unverified verbatim-ASR labels with per-class loss masks, which nobody has published ([Shih et al. 2024](https://arxiv.org/abs/2409.10704)).

StutterClear's teacher has clip AUCs of 0.815–0.978, but no published result is directly comparable, because the numbers come from StutterClear's own pooled split. Its block F1 of 0.25 is below the 0.35 reported on the speaker-exclusive benchmark ([Sušac et al. 2026](https://arxiv.org/abs/2606.20338)). The largest risk is how the output sounds. In the only listener study found, real-voice cutting tools scored at or below unprocessed stuttered speech ([SpeechAgent](https://arxiv.org/abs/2510.20113)). The fastest route to a defensible paper has two parallel steps. First, re-run the detector on the official splits and the CASA frame benchmark (a 20 ms frame-labelled test set from clinical FluencyBank annotations) and submit the lookahead study to Interspeech 2027. Second, start ethics approval now for the listening test that the real-time removal paper needs.

## One clean first; seven claims need narrowing

The verdicts below use three labels:
- **ALREADY DONE:** published or patented in substantially the same form.
- **PARTIALLY DONE:** the concept or its key parts exist.
- **OPEN:** no evidence was found in the alphaXiv, web and Firecrawl searches behind the notes. That is not proof of absence. The notes did not search journals outside arXiv, workshop papers or very recent preprints exhaustively.

| # | Contribution | Verdict | Closest prior work (by whom) | What StutterClear can defensibly claim as new |
|---|---|---|---|---|
| 1 | Real-time stutter removal in the speaker's real voice (causal detector + delay buffer + crossfade splicing) | **PARTIALLY DONE.** The concept is patented but has never been built and evaluated. | IBM US8682678, Malkin & Trewin, filed 2012, active ([Google Patents](https://patents.google.com/patent/US8682678B2/en)). IBM→Nuance US8600758/US8571873, Deshmukh et al., 2011; '758 has lapsed ([Google Patents](https://patents.google.com/patent/US8600758B2/en)). CleanStream OBS plugin, live delay with mute or beep ([GitHub](https://github.com/royshil/obs-cleanstream)). Amazon Detect-and-Pass drops stutter frames inside ASR ([arXiv](https://arxiv.org/pdf/2202.05396)). | The first implemented and evaluated streaming system that splices repetitions, prolongations and fillers out of the real voice, with crossfades, a bounded and measured delay, and reported buffer drain. It must cite the patents and must not claim the idea. |
| 2 | Causal/streaming detection; lookahead-vs-accuracy curve from a distilled multi-head student | **OPEN (no evidence found)** for: a detection lookahead sweep; self-supervised-teacher → tiny-causal-student distillation; one head per lookahead; any causal frame-level result on CASA or FluencyBank++. | Kumar, Daliri & Gupta (Oct 2026): forecasting curve at negative lookahead, ALREADY DONE ([arXiv](https://arxiv.org/abs/2610.19235)). Kozak (Apr 2026): inference-time tail masking, 616K-parameter on-device CNN, failed distillation ([arXiv](https://arxiv.org/abs/2604.27279)). Chen et al. 2022 and Futami et al. 2023: lookahead curves for conversational disfluency on Switchboard, not stuttering ([Chen](https://arxiv.org/abs/2205.00620); [Futami](https://arxiv.org/abs/2211.08726)). | The first accuracy-vs-lookahead curve for stuttering detection, per class. Together with Kumar's forecasting curve, the first continuum from −1.5 s to +0.8 s. The first causal entries on CASA and FluencyBank++. |
| 3 | CrisperWhisper silver frame labels + synthetic frame labels + clip MIL; v2 synthetic generator | **PARTIALLY DONE.** Synthetic + MIL is ALREADY DONE (binary, in two stages). Silver labels and the three-source mix are OPEN. The generator's classes are covered, but five of its mechanisms were not found anywhere. | Harvill et al. 2022 and Shih et al. SLT 2024: synthetic pretraining, then max-pool MIL ([Shih](https://arxiv.org/abs/2409.10704)). Sušac et al. 2026: multi-label MIL without synthetic data ([arXiv](https://arxiv.org/abs/2606.20338)). ASU 2026 used CrisperWhisper 2.0 only as a review aid for human annotators ([arXiv](https://arxiv.org/abs/2610.19235)). nyra's 2026 token→label rules ([arXiv](https://arxiv.org/abs/2607.18934)). LibriStutter, Libri-Dys, LLM-Dys and VCTK-Pro generators. | The first stutter detector trained on unverified verbatim-ASR silver frame labels from real stuttered speech, with per-class loss masks. The first published precision of such labels against frame annotations. Joint 5-class training with log-sum-exp pooling counts only as an incremental variant. The generator mechanisms count as engineering contributions: own-voice fillers, period-aligned looping, abandoned restarts, rapid 3–5× repeats with near-zero gaps, and clusters on one word. |
| 4 | Evaluation: residual disfluency via verbatim ASR; meaning via intended-transcript WER; fluent words wrongly cut | **PARTIALLY DONE.** Text-domain analogues and transcript-based counts exist. The exact audio metrics are OPEN, and no benchmark exists. | Jamshid Lou & Johnson 2020: FER/DER ([ar5iv](https://ar5iv.labs.arxiv.org/html/2009.10298)). TalkLess, UIST 2025 ([arXiv](https://arxiv.org/abs/2507.15202)). Das et al., ICASSP 2019 ([arXiv](https://arxiv.org/pdf/1812.03415)). Toyin et al. 2026: intended-reference WER (isWER) vs verbatim-reference WER (vWER) ([arXiv](https://arxiv.org/abs/2606.31112)). | Audio-domain counterparts of DER (residual events per 100 words from CrisperWhisper on the output) and of the deletion part of FER (fluent words cut, from word timestamps), applied to stuttering for the first time. A reproducible protocol for evaluating stutter removal on FluencyBank Timestamped. |
| 5 | Blocked-word prediction from a context card + words so far + stuck onset (audio or lips) | **PARTIALLY DONE.** The core concept is patented. | Intel US10672393, filed 2018, granted 2020, active ([Google Patents](https://patents.google.com/patent/US10672393B2/en)). NITK 2015: n-gram predictor that speaks the word, 87% ([NITK](https://idr.nitk.ac.in/items/d8d1c176-b274-43c8-84cc-eeb09232b353/full)). SpeakFaster: LLM expansion of word-initial letters with conversation context ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11530652/)). | An LLM conditioned on a context card the user writes. The onset used as an explicit constraint, with its gain measured. A lip-viseme onset cue (OPEN). The first top-k accuracy on real blocks. |
| 6 | Audio-visual face/lip-landmark block detection | **PARTIALLY DONE.** It has already been shown that video helps detect blocks. | Altinkaya & Smeulders 2020 ([ACM](https://dl.acm.org/doi/10.1145/3423325.3423733)). Mohapatra et al. 2024: block balanced accuracy 0.58 audio vs 0.71 face ([arXiv](https://arxiv.org/abs/2406.06964)). MMSD-Net 2024 ([arXiv](https://arxiv.org/abs/2407.11492)). Das et al. 2020–22: facial action units predict upcoming stutters ([arXiv](https://arxiv.org/abs/2010.01231)). | Interpretable on-device landmark features. Streaming, frame-level localisation of silent blocks fused with an audio posterior. Place of articulation of the stuck onset read from the lips. Speaker-disjoint evaluation. The first frame-level audio-visual block benchmark. |
| 7 | Reinforcement learning, including constrained RL, for cut/hold/play and adaptive lookahead | **PARTIALLY DONE.** Generic RL for when to emit or wait is ALREADY DONE. Applying it to disfluency removal is OPEN, and so is constrained RL in streaming speech. | Gu et al. 2017 ([arXiv](https://arxiv.org/abs/1610.00388)). Hibiki-Zero 2026 ([arXiv](https://arxiv.org/html/2602.11072)). HPO 2026 ([arXiv](https://arxiv.org/html/2604.21045v1)). X2Streaming-ASR and Look Less, Hear Better, both Sep 2026 ([arXiv](https://arxiv.org/abs/2609.08672); [arXiv](https://arxiv.org/abs/2609.18333)). Chen et al. 2022: learned wait head for text disfluency ([arXiv](https://arxiv.org/abs/2205.00620)). | The first RL policy for streaming stutter removal. A constrained formulation that caps the probability of cutting a fluent word at ε. An ε the user can set at inference. |
| 8 | Physiological signals (ECG, respiration, EDA) to anticipate stuttering moments | **Detection PARTIALLY DONE. Anticipation OPEN. No combined dataset (OPEN).** | Villegas et al. 2019: respiration classifies block vs non-block at 82.6%, concurrently ([PubMed](https://pubmed.ncbi.nlm.nih.gov/31946902/)). Walsh & Usler 2019: utterance-level skin-conductance arousal ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7201324/)). Das et al. 2022: EEG + face anticipation, 80.8% ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9376608/)). | The first anticipation of individual stuttering moments from wearable physiology in continuous speech. The first synchronized audio + video + physiology dataset of people who stutter. |

### Real-voice removal is a patented idea that nobody has built and measured

Research systems that output fluent audio from stuttered speech have converged on a transcribe → rewrite → re-synthesize cascade (ASR → LLM → TTS).
- **STEAMROLLER** takes **3.77 ± 0.62 s per 50–100-word segment**. Listeners rated its cloned voice only **3.4/5** on speaker similarity, and 7 of 23 users said it lacked warmth or did not sound like them ([STEAMROLLER](https://arxiv.org/html/2601.10223v1)).
- **SpeechAgent** refines an 11.49 s utterance in 0.91 s, but only after the utterance ends. It uses a generic TTS voice, and its authors state that no assistive tool offers real-time refinement of impaired speech ([SpeechAgent](https://arxiv.org/abs/2510.20113)).

Real-voice editors exist only offline:
- **Descript** and **Cleanvoice** process uploaded recordings.
- **TalkLess** cuts the waveform and re-synthesizes about half a word around each cut with VoiceCraft ([TalkLess](https://arxiv.org/abs/2507.15202)).
- **Das et al. 2019** removed fillers and shortened pauses offline. They listed "segment … stuttering" and "a dynamic and online repair scheme" as future work ([Das et al.](https://arxiv.org/pdf/1812.03415)).

The patents describe exactly StutterClear's idea:
- **IBM US8682678** claims deleting impairment artifacts, with stuttering defined as repeated phonemes, from captured audio before playback or transmission. It describes a phone app that intercepts speech "before they are transmitted" ([US8682678B2](https://patents.google.com/patent/US8682678B2/en)).
- **The IBM→Nuance family** keeps one of the repeated syllables, shortens prolongations, trims silences and reconstructs with PSOLA "in real-time" on telecom servers ([US8600758B2](https://patents.google.com/patent/US8600758B2/en)).

Neither patent reports latency, an implementation or results. The only live tool, CleanStream, delays the stream and replaces unwanted words with silence or a beep. It never removes time, never targets stuttering, and publishes no accuracy or delay figures ([CleanStream](https://github.com/royshil/obs-cleanstream)).

The closest mechanism is Amazon's Detect-and-Pass. It labels 100 ms frames using four context frames on each side and skips stutter frames inside an RNN-T decoder, cutting WER by **12.18–71.24%**, but it never produces audio for a listener ([Detect-and-Pass](https://arxiv.org/pdf/2202.05396)). StutterClear's honest framing follows: the system extends frame dropping with lookahead from a machine listener to a human one, and it is the first published, measured implementation of an idea that has been patented for 14 years.

### Detection, labels and evaluation are open recipes built from finished parts

The **lookahead curve is the strongest clean claim**, because the closest work runs along the same time axis without covering detection:
- **Kumar et al.** forecast stuttering onsets every 100 ms from past audio only. They deliberately excluded frames inside events because "the task would become detection rather than forecasting". Their AUROC falls from **0.604 at 0.3 s to 0.575 at 1.5 s** ([Kumar et al. 2026](https://arxiv.org/abs/2610.19235)).
- **Kozak** zeroed the trailing frames of 3-s clips at inference time. Event-detection AUC barely moved, from 0.649 to 0.635, until more than 512 ms was removed ([Kozak 2026](https://arxiv.org/abs/2604.27279)).
- **Switchboard conversational disfluency** is the only domain with true lookahead curves. In text, streaming F1 rises from 0.76 at zero tokens of lookahead to 0.83 at one and 0.85 at two ([Chen et al. 2022](https://arxiv.org/abs/2205.00620)). From audio, a joint streaming model reaches 0.84 at about 529 ms median token latency ([Futami et al. 2023](https://arxiv.org/abs/2211.08726)).

Distillation has only a failed precedent in stuttering. Kozak distilled a same-size CNN that sees the future clip into a forecasting student, and validation AUC fell from 0.546 to 0.539 because the teacher's soft labels carried no more information than the hard ones ([Kozak 2026](https://arxiv.org/abs/2604.27279)). StutterFuse lists distillation and streaming only as future work ([StutterFuse](https://arxiv.org/abs/2512.13632)). Lookahead-configurable models exist in streaming ASR and speech enhancement, but not for stuttering ([ANCAT](https://arxiv.org/abs/2305.04159); [LaCo-SENet](https://arxiv.org/abs/2606.19688)).

Labels are the opposite case: the ingredients exist, and only the specific recipe is new.
- **Synthetic pretraining followed by MIL fine-tuning:** Harvill et al. and Shih et al. pretrained frame detectors on LibriSpeech with synthetic stutters, then fine-tuned on SEP-28k clip labels with max-pooling MIL. Both models are binary and both train in two stages ([Shih et al. 2024](https://arxiv.org/abs/2409.10704)).
- **Multi-label MIL without synthetic data:** Sušac et al. generalized MIL to five classes and said pretraining on frame-level data "might be useful" ([Sušac et al. 2026](https://arxiv.org/abs/2606.20338)).
- **Verbatim ASR on real stuttered speech, always human-verified:**
  - ASU ran CrisperWhisper 2.0 on FluencyBank Timestamped, but only as a review aid; a human verified every event ([Kumar et al. 2026](https://arxiv.org/abs/2610.19235)).
  - Shih et al. used Whisper only for timing; the stutter labels came from clinicians ([Shih et al. 2024](https://arxiv.org/abs/2409.10704)).
  - nyra's own evaluation protocol uses token→label rules almost identical to StutterClear's mapping: a trailing hyphen is a cut-off, a bracketed filler is a filler, and earlier duplicates are repetitions. nyra uses them only to score ASR output ([nyra 2026](https://arxiv.org/abs/2607.18934)).

The defensible claim is therefore narrow: the first stutter detector trained on unverified verbatim-ASR frame labels from real speech, with per-class masks for classes the ASR cannot see.

For the v2 generator, every stutter class has prior synthetic coverage:
- LibriStutter covers 2–3-word phrase repetitions and fillers from other speakers ([LibriStutter](https://arxiv.org/abs/2009.11394)).
- LLM-Dys produces fillers in the same speaker's voice by zero-shot cloning ([LLM-Dys](https://arxiv.org/abs/2505.22029)).
- VCTK-Pro and Libri-Co-Dys model co-occurring dysfluencies ([Stutter-Solver](https://arxiv.org/abs/2409.09621); [SSDM 2.0](https://arxiv.org/abs/2412.00265)).

The novelty is mechanistic: fillers spliced from the speaker's own schwa and nasal phones, prolongations made by period-aligned looping, abandoned restarts rendered in audio with frame labels, a deliberate regime of 3–5× repeats with near-zero gaps, and clusters chained on the same word.

Evaluation follows the same pattern:
- **FER and DER (text):** Jamshid Lou & Johnson's Fluent Error Rate and Disfluent Error Rate are the text-domain analogues of "fluent words wrongly cut" and "residual disfluencies". On Switchboard, their end-to-end model scores FER 11.6 and DER 24.0 ([FER/DER](https://ar5iv.labs.arxiv.org/html/2009.10298)).
- **Counts on edited audio:** TalkLess counts residual fillers and repetitions from a transcript of the edited audio, aligned with rev.ai and Gentle rather than a verbatim ASR re-run ([TalkLess](https://arxiv.org/abs/2507.15202)). Das et al. report filled pauses per minute falling from 3.66 to 1.76 ([Das et al.](https://arxiv.org/pdf/1812.03415)).
- **Dual references:** 2026 work formalized verbatim versus intended references and a WER split into content and style errors ([Toyin et al. 2026](https://arxiv.org/abs/2606.31112); [nyra 2026](https://arxiv.org/abs/2607.18934)).

No paper runs a verbatim ASR on the output of a stutter-editing system, and no standard benchmark exists for stutter removal.

### Prediction, video, RL and physiology are open only where they touch real blocks

**Blocked-word prediction.** Intel's "time capsule" patent detects a stutter or pause and predicts the missing word. Its inputs are the preceding words, a personal language model built from the user's conversations, context such as location, and the stutter sound. It returns an n-best list through an earpiece or display. It treats the stutter sound as a soft cue: for "ba-ba-ba" it still offers "bowling" and "bocce". It reports no experimental results ([US10672393B2](https://patents.google.com/patent/US10672393B2/en)). The methodological precedent for a context-conditioned LLM is SpeakFaster. Users type word-initial letters and an LLM expands them using conversation context, which saved **57% more motor actions** than predictive keyboards ([SpeakFaster](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11530652/)). In earlier work from the same team, one prior conversation turn **more than doubled** expansion accuracy ([Cai et al. NAACL 2022](https://aclanthology.org/2022.naacl-main.91.pdf)). StutterClear's design moves the cue from typed letters to the stuck sound. No paper found reads the speaker's lips to identify the sound they are blocked on.

**Audio-visual detection.** Mohapatra et al. already showed that video helps with blocks. On 4,004 FluencyBank clips, block balanced accuracy rose from **0.58 with HuBERT audio to 0.71 with full-face video**. Lip-only crops scored only 0.57, partly because landmark failures on low-quality video left 18% less paired data ([Mohapatra et al. 2024](https://arxiv.org/abs/2406.06964)). The other prior work is weaker:
- Altinkaya & Smeulders' 2020 audio-visual block work publishes no usable numbers, and its dataset site did not respond on 17 Oct 2026 ([ACM](https://dl.acm.org/doi/10.1145/3423325.3423733)).
- MMSD-Net's 90.19 F1 used fluent negatives from a different corpus, so it may reflect corpus identity rather than detection ([MMSD-Net](https://arxiv.org/abs/2407.11492)).

What remains open is interpretable landmark features, streaming frame-level localisation of silent blocks, and onset place of articulation from the lips.

**Reinforcement learning.** RL for streaming emit-or-wait decisions is mature:
- Hibiki-Zero cut French short-form end offset from 3.5 s to 2.1 s with GRPO ([Hibiki-Zero](https://arxiv.org/html/2602.11072)).
- HPO's quality gate sets latency to its worst value when quality drops ([HPO](https://arxiv.org/html/2604.21045v1)).
- Two September 2026 streaming-ASR papers learn commit policies under latency rewards. One cut WER by 30.8% relative at 80 ms ([Look Less Hear Better](https://arxiv.org/abs/2609.18333)). The other commits within 12–85 ms of English word endpoints ([X2Streaming-ASR](https://arxiv.org/abs/2609.08672)).

In stuttering, RL appears only as an unspecified option in an MIT patent for switching altered-auditory-feedback (AAF) modes, and in a simulated medication-dosing proof of concept ([US11727949B2](https://patents.google.com/patent/US11727949B2/en); [Constas et al.](https://arxiv.org/abs/2312.11509)). StutterClear's novelty is the application and the objective: a cut/hold/play action space, a catastrophic penalty for cutting fluent words, and a constraint budget. None of the streaming-speech papers found uses a Lagrangian or constrained-MDP formulation.

**Physiology.** Physiology is the thinnest area:
- Skin-conductance responses are larger during stuttered utterances (d = 0.99 in preschoolers), but they lag their trigger by 1–3 s ([Walsh & Usler 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC7201324/)).
- Arousal explains only "small percentages of the variances" of disfluency ([Weber & Smith 1990](https://pubmed.ncbi.nlm.nih.gov/2273884/)).
- The only physiology-to-event classifier found detects concurrent blocks from respiration and reports no lead time ([Villegas et al. 2019](https://pubmed.ncbi.nlm.nih.gov/31946902/)).

## Two active patents and several non-commercial licences need review before any product ships

**This section is not legal advice.** The patent statuses come from Google Patents, whose expiry dates are Google's estimates, not legal conclusions. The licence readings come from model cards and dataset pages, and nobody has checked them against contracts. A patent attorney should do a freedom-to-operate review before any commercial release.

| Item | Holder and status (as reported) | What it covers | Why it matters to StutterClear |
|---|---|---|---|
| **US8682678B2** "Automatic realtime speech impairment correction", plus family US8620670 and US20130246058 | IBM. Filed 14 Mar 2012, granted 25 Mar 2014. Google lists it **Active, expiring 2032-07-14** ([Google Patents](https://patents.google.com/patent/US8682678B2/en); [USPTO PDF](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8620670)) | Identifies impairment artifacts through an impairment-to-artifact database, deletes them from captured audio, and provides the modified audio, e.g. from a phone app before transmission | Closest to contribution 1. Cite it as prior art. The notes flag that claim 1's database (and possibly its comparison with a reference reading) may narrow its scope, but only counsel can judge that |
| **US8600758B2 / US8571873** "Reconstruction of a smooth speech signal from a stuttered speech signal" | IBM, assigned to Nuance. Priority 18 Apr 2011. '758 **expired for non-payment, effective 3 Dec 2021**. '873 status **not checked** ([Google Patents](https://patents.google.com/patent/US8600758B2/en); [USPTO PDF](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8571873)) | Keeps one repeated syllable, shortens prolonged phones, cuts silences, PSOLA reconstruction, "real-time" hosting by a telecom provider | Cite as prior art. Confirm '873's status |
| **US11616883B2**, plus continuations US12113937B2 and US20240430365A1 | Rovi Guides (now Adeia). Priority 26 Jan 2021, granted 28 Mar 2023 ([Google Patents](https://patents.google.com/patent/US11616883B2/en)) | Plays a muted user's recorded speech to others at a variable rate, optionally removing "uh"/"um" and pauses | Tangential. Becomes relevant if StutterClear adds catch-up or conferencing features |
| **US10672393B2** "Time capsule based speaking aid" | Intel. Filed 12 Jan 2018, granted 2 Jun 2020, **Active** ([Google Patents](https://patents.google.com/patent/US10672393B2/en)) | Detects a stutter or pause, then predicts the word from preceding speech, a personal language model and the stutter sound, delivering an n-best list | Directly relevant to contribution 5. Claim 1's personal LM is "generated in a continuous manner over time until the user experiences cognitive deterioration", which a typed context card may not match. Only counsel can judge that |
| **US11727949B2** "Methods and apparatus for reducing stuttering" | MIT (Brigham and Women's Hospital also listed). Priority 12 Aug 2019, granted 15 Aug 2023 ([Google Patents](https://patents.google.com/patent/US11727949B2/en)) | Real-time altered auditory feedback that switches mode as stuttering rises; names RL (Q-learning, DQN) | Tangential to contribution 7's personalization. Matters if StutterClear ever feeds altered audio back to the user |
| **CrisperWhisper 2.0** (standard models) | nyra labs. "Nyra Health Non-Commercial Research License"; "any commercial use requires a commercial license"; the Pro models are commercial-only ([HF model card](https://huggingface.co/nyralabs/CrisperWhisper2.0_medium)) | Produces both StutterClear's silver labels and its residual-disfluency metric | Whether a commercial model may be trained on labels derived from it is an open legal question, separate from dataset licences |
| **CrisperWhisper v1** | CC-BY-NC-4.0 ([HF model](https://hf.co/nyralabs/CrisperWhisper)) | Same use | Same flag |
| **SEP-28k** | Apple. "Creative Commons Attribution-NonCommercial 4.0"; no audio distributed; "original copyright remains with the podcast owners" ([Apple repo](https://github.com/apple/ml-stuttering-events-dataset)). Kozak cites CC BY-SA 4.0, which conflicts; the Apple repo is the primary source ([Kozak 2026](https://arxiv.org/abs/2604.27279)) | Main clip-label source for the teacher | Commercial use of models trained on it is unclear |
| **FluencyBank** (TalkBank) | Research data are restricted to consortium members, and students need a faculty advisor. The site shows a CC BY-NC-SA 4.0 badge "under review" ([TalkBank](https://talkbank.org/fluency/)). Kozak reports Teaching-section derivatives under CC BY-NC-SA 3.0 ([Kozak 2026](https://arxiv.org/abs/2604.27279)) | Teacher training, CASA, FluencyBank Timestamped, FluencyBank++, and the video for contribution 6 | Access and redistribution limits. Mohapatra's audio-visual subset is derived from TalkBank, and its licence is unverified ([Mohapatra et al. 2024](https://arxiv.org/abs/2406.06964)) |

The patents shape how the papers should be worded more than what they can study. Citing IBM, Nuance and Intel as prior art and claiming the *implementation and evaluation* is accurate. It also protects against reviewers who know the patents. The licences matter most for the product: the training data (SEP-28k, FluencyBank) and the label generator (CrisperWhisper) all carry non-commercial terms. The notes did not check UCLASS's terms, even though the repository's latest commit trains the v2 teacher on UCLASS data.

## StutterClear's teacher scores higher on three classes, but on a split nobody else uses

StutterClear's teacher is WavLM-Large + BiGRU with log-sum-exp MIL pooling, trained on synthetic, silver and clip labels. It was validated on StutterClear's own split of pooled SEP-28k and FluencyBank, held out by speaker group, with majority-vote (2-of-3) labels. The table below sets those internal numbers against the best published clip-level results.

| Class | StutterClear teacher, F1 at 0.5 / AUC (own pooled split) | SEP-28k-E F1, comparable protocol: Sušac 2026, ≥2/3 labels, θ = 0.5 | Best SEP-28k-E F1, any protocol | Best FluencyBank F1 | Only per-class AUC on SEP-28k (Pattanayak 2026; **not comparable**) |
|---|---|---|---|---|---|
| Sound repetition | 0.63 / 0.915 | 0.53 | 0.54 (Wagner 2024, LLM fusion) | 0.66 (Wagner 2024; Haas & Bayerl 2026) | 0.844 |
| Word repetition | 0.71 / 0.961 | **0.80** | 0.80 (Sušac) | 0.59 (Haas & Bayerl) | 0.851 |
| Prolongation | 0.65 / 0.915 | 0.50 | 0.56 (Wagner) | 0.62 (Bayerl 2023) | 0.789 |
| Block | **0.25** / 0.815 | 0.35 | 0.57–0.62 (Wagner: Whisper-large ASR + Llama-2-7B) | 0.58 (Wagner) | 0.724 |
| Interjection | 0.90 / 0.978 | 0.83 | 0.83 (Sušac) | 0.84 (Haas & Bayerl) | 0.866 |

Sources: [Sušac et al. 2026](https://arxiv.org/abs/2606.20338); [Wagner et al. 2024](https://arxiv.org/abs/2406.11025); [Haas & Bayerl 2026](https://arxiv.org/abs/2603.26939); [Pattanayak et al. 2026](https://arxiv.org/abs/2609.32285).

The frame-level picture matters more for a splicer, and there StutterClear has no entry yet. The table below lists the best published results by benchmark.

| Benchmark | Metric | Best published | Other results | Any causal entry? |
|---|---|---|---|---|
| CASA gold (FluencyBank; 8 recordings, 732 events) | Binary any-stutter frame F1, 20 ms frames, averaged per recording | **0.70**: Whisper-medium + attention MIL. Non-causal, 3-s context, about 307M parameters, trained on clip labels only ([Sušac et al. 2026](https://arxiv.org/abs/2606.20338)) | 0.66 (Whisper, max pooling); 0.47 (YOLO-Stutter); 0.45 (StutterCut) | None |
| FluencyBank++ (3,017 clips of 5 s) | Time-F1 at IoU > 0.5 | **69.3** StutterCut: prolongation 73.2, repetition 68.8, interjection 64.5, block 70.6. Offline, 100 MC-dropout passes ([Ghosh et al. 2025](https://arxiv.org/abs/2508.02255)) | 56.8 (YOLO-Stutter) | None |
| Word-level FluencyBank (36 speakers) | Word F1 | 0.591, synthetic pretraining + MIL ([Shih et al. 2024](https://arxiv.org/abs/2409.10704)) | — | None |
| Manually timed SEP-28k subset | Time-F1, any overlap counts; needs reference text | 0.830 SSDM 2.0; 0.813 YOLO-Stutter ([SSDM 2.0](https://arxiv.org/abs/2412.00265); [YOLO-Stutter](https://arxiv.org/abs/2408.15297)) | — | Not applicable; StutterClear is text-free |

Five caveats keep these numbers from supporting a SOTA claim.

1. **Not the official split.** StutterClear's validation set is neither the SEP-28k-E test partition nor Bayerl et al.'s FluencyBank partition. SEP-28k has no official speaker IDs, and SEP-28k-E's speaker labels are semi-automatic, so StutterClear's speaker grouping is uncertain.
2. **Pooling with FluencyBank inflates some classes.** Published systems score higher on FluencyBank than on SEP-28k-E, especially for sound repetition. The pooled set therefore likely flatters the classes where the teacher appears to win ([Wagner et al. 2024](https://arxiv.org/abs/2406.11025); [Haas & Bayerl 2026](https://arxiv.org/abs/2603.26939)).
3. **Label thresholds are unknown for several comparators.** Bayerl 2023, Wagner 2024, Miyahara 2025 and Haas & Bayerl do not restate their label threshold. Thresholds move results a lot: with majority-vote rather than lenient labels, Pattanayak's block F1 fell from **0.637 to 0.346** without retraining ([Pattanayak et al. 2026](https://arxiv.org/abs/2609.32285)).
4. **The block result points to calibration, not ranking.** Block F1 of 0.25 alongside an AUC of 0.815 means the model ranks blocks reasonably but the 0.5 threshold is wrong. After a simulated audio-processing chain degraded block F1 to 0.465, Pattanayak recovered it to 0.630 by re-tuning the threshold alone ([Pattanayak et al. 2026](https://arxiv.org/abs/2609.32285)).
5. **FluencyBank speakers overlap across benchmarks.** CASA, FluencyBank++, FluencyBank Timestamped and the SEP-28k-style FluencyBank labels share FluencyBank's adult speakers. Any frame-level evaluation must exclude those speakers from training.

Two framing points also apply. Sušac et al. claim block SOTA at 0.35 without comparing against Wagner's 0.57–0.62 LLM system ([Sušac et al. 2026](https://arxiv.org/abs/2606.20338)). Wagner's system is not a realistic comparator for a 427K-parameter causal student, but it is the published ceiling for blocks and should be cited.

Reporting per-class AUC is a genuinely open niche: no strong fine-tuned SEP-28k-E system reports it. The only per-class AUCs on SEP-28k come from a frozen linear probe with lenient labels and folds grouped by episode rather than speaker, so StutterClear should say explicitly that its AUCs are not comparable to them ([Pattanayak et al. 2026](https://arxiv.org/abs/2609.32285)).

For deployment, the comparison row is Kozak's 616K CNN. It runs in 0.25 ms per 3-s window on an A19 Pro Neural Engine and at 1.34 ms per window in a 4-Hz streaming simulation, with event-detection AUC 0.649 ([Kozak 2026](https://arxiv.org/abs/2604.27279)).

## Listeners rated real-voice cutting at or below doing nothing

### Cutting can sound worse than leaving speech alone

The most important finding in the notes cuts against StutterClear's core idea. SpeechAgent's human evaluation rated unprocessed stuttered speech at **2.67** for clarity on a 1–5 scale. Two real-voice cutting baselines scored at or below it:
- **Cleanvoice** scored **2.50**, with a comparison MOS against the unprocessed input of **−0.17**.
- **Arjun et al.'s 2020 system** (repetition removal plus silence trimming) scored **1.33**, with a comparison MOS of **−0.50**.

The authors attribute this to cuts that "distort meaning and naturalness". The resynthesis systems scored far higher: SpeechAgent 4.50 and an ASR+TTS cascade 4.00 ([SpeechAgent](https://arxiv.org/abs/2510.20113)). The study's rater count is not stated, and both cutting baselines were offline whole-utterance tools rather than conservative crossfaded splicers. The result is still the only listener evidence on real-voice cutting of stuttered speech.

The counter-evidence comes from outside stuttering:
- TalkLess users preferred short resynthesized transitions around cuts over full resynthesis, because full resynthesis lost "vocal authenticity" ([TalkLess](https://arxiv.org/abs/2507.15202)).
- Resynthesized clones have their own identity problem, as STEAMROLLER's speaker-similarity rating of 3.4/5 shows ([STEAMROLLER](https://arxiv.org/html/2601.10223v1)).

The implication is blunt. If StutterClear's cuts do not beat "no processing" in a listening test, its detection numbers cannot rescue the product thesis.

Community positioning compounds the risk. The FAccT 2026 "Cripping AI" paper names "captioning systems that remove stuttered utterances by default" as fluency-centric design. It asks systems to support "the speaker's comfort and dignity" instead ([Tang et al. 2026](https://arxiv.org/abs/2605.02080)). Samsung's Impulse ad drew an open letter with **99 organizational and 828 individual signatories** ([SPACE](https://www.spacetostutter.org/impulse)).

### Verbatim ASR is a noisy ruler on stuttered speech

StutterClear assumed word timestamps accurate to about 30–40 ms. That matches the CrisperWhisper 2.0 model card's figures for read and conversational speech ([HF model card](https://huggingface.co/nyralabs/CrisperWhisper2.0_medium)). On stuttered FluencyBank speech, nyra's own paper reports a **102 ms mean boundary error**, with only **38.6% of boundaries within ±50 ms** and 80.3% within ±200 ms ([nyra 2026](https://arxiv.org/abs/2607.18934)). That error spans about five of StutterClear's 20 ms frames, so hard frame targets from silver labels will be systematically misplaced.

The cut-off token, which StutterClear maps to sound repetition, is the weakest class:
- CrisperWhisper 2.0 reaches only **43.2 and 39.1 cut-off F1** on acted English disfluencies.
- v1 scores **0.0**.
- Lenient cut-off recall is about 95%, meaning the position is usually found even when the surface form is wrong ([nyra 2026](https://arxiv.org/abs/2607.18934)).

When ASU ran CrisperWhisper 2.0 on real stuttered speech, a human still had to add missed sound and word repetitions and **all 129 prolongations and 150 blocks** ([Kumar et al. 2026](https://arxiv.org/abs/2610.19235)). Separately, incomplete-syllable and multisyllable repetitions are the events ASR systems handle worst ([Toyin et al. 2026](https://arxiv.org/abs/2606.31112)).

This creates a circularity risk the plan does not yet address. If CrisperWhisper both generates the training labels and counts the residual disfluencies in the output, the evaluation can reward agreement with CrisperWhisper's blind spots rather than real stutter removal. Human references break that loop.

Human references have limits too:
- Annotators agree only moderately on FluencyBank Timestamped onsets: Fleiss κ = 0.54 for repetitions, 0.73 for prolongations and 0.56 for blocks ([Kumar et al. 2026](https://arxiv.org/abs/2610.19235)).
- On SEP-28k, block agreement is κ = 0.25 ([Lea et al. 2021](https://arxiv.org/abs/2102.12394)).
- An annotator who stutters relabelled more than 25% of sampled SEP-28k clips ([Sridhar & Wu 2025](https://www.isca-archive.org/interspeech_2025/sridhar25_interspeech.pdf)).

### Method risks further down the plan

Several later contributions carry specific, documented failure modes:
- **Distillation.** Kozak's failure shows that distillation helps only when the teacher's soft outputs carry information beyond the hard labels. StutterClear needs an ablation that trains the student without the teacher ([Kozak 2026](https://arxiv.org/abs/2604.27279)).
- **Reinforcement learning.** Hibiki-Zero's RL failed when it started from a full-sentence-delay model ([Hibiki-Zero](https://arxiv.org/html/2602.11072)). X2Streaming-ASR initializes its commit policy from probed trajectories ([X2Streaming-ASR](https://arxiv.org/abs/2609.08672)). Both argue for behaviour-cloning StutterClear's current hysteresis rule before any RL.
- **Lip-only video.** Lip-only features underperformed the full face on blocks (0.57 vs 0.71) ([Mohapatra et al. 2024](https://arxiv.org/abs/2406.06964)).
- **Webcam frame rate.** At 30 fps the Nyquist limit is 15 Hz, the top of the 5–15 Hz tremor band seen in stuttering EMG ([Denny & Smith 1992](https://pubmed.ncbi.nlm.nih.gov/1494267/)).
- **Anticipation is weak.** The best audio-only forecaster reaches a within-speaker block AUROC of only 0.541 and raises 27.6 false warning bouts per minute at its max-F1 threshold ([Kumar et al. 2026](https://arxiv.org/abs/2610.19235)).
- **Physiology is slow.** Skin-conductance responses arrive 1–3 s late, so physiology can only act as a slow state prior ([Walsh & Usler 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC7201324/)).
- **Synthetic data.** Synthetic stutter data has its sceptics. Haas & Bayerl call its usefulness "often overstated" ([Haas & Bayerl 2026](https://arxiv.org/abs/2603.26939)), and LibriStutter's unnatural dysfluencies were reported to degrade training ([Ghosh et al. 2025](https://arxiv.org/abs/2508.02255)).
- **Front-end processing in the live system.** In a simulated chain of denoising, loudness normalization, Opus coding and WebRTC VAD, block F1 fell from 0.638 to 0.465. The detector must tap the mic before any processing ([Pattanayak et al. 2026](https://arxiv.org/abs/2609.32285)).

## A five-month sprint puts the detection paper into Interspeech 2027

The actions below are ordered by value per unit of effort. The first five need no new data collection and decide whether the strongest claims survive.

| Priority | Action | Contribution | Bar to beat / reason |
|---|---|---|---|
| P0-1 | Re-evaluate the teacher on the **official SEP-28k-E test split**. Separately, evaluate on the **Bayerl FluencyBank partition**. Use ≥2/3 labels and θ = 0.5, plus per-class thresholds tuned on validation. Report per-class AUC with bootstrap CIs grouped by speaker or episode. | SOTA, 2, 3 | Sušac F1: Bl 0.35, Pro 0.50, Snd 0.53, Wd 0.80, Int 0.83 ([Sušac](https://arxiv.org/abs/2606.20338)). Also cite Wagner's 0.57–0.62 block ceiling ([Wagner](https://arxiv.org/abs/2406.11025)) |
| P0-2 | Evaluate the teacher and **every student head** on **CASA gold** (binary frame F1 averaged per recording; per type where the mapping allows) and **FluencyBank++** (time-F1 at IoU > 0.5). Exclude overlapping FluencyBank speakers from training first. | 2, 3 | 0.70 CASA ([Sušac](https://arxiv.org/abs/2606.20338)); 69.3 FluencyBank++ ([StutterCut](https://arxiv.org/abs/2508.02255)). These would be the first causal entries on both |
| P0-3 | Publish the **per-class lookahead curve** (0, 100, 200, 300, 500, 800 ms) with pooled and within-speaker AUC. Add a distillation ablation (student without the teacher) and on-device latency for the 427K student. | 2 | Kumar's forecasting curve as the negative-lookahead anchor ([Kumar](https://arxiv.org/abs/2610.19235)); Kozak's 616K CNN and failed distillation ([Kozak](https://arxiv.org/abs/2604.27279)) |
| P0-4 | Run silver-label ablations: (a) with and without silver labels; (b) against the MIL-only recipe (Sušac) and the synthetic + MIL recipe (Shih); (c) joint vs sequential training; (d) log-sum-exp vs max vs attention pooling. **Score the silver labels themselves** against CASA, FluencyBank++ or the ASU onsets. Add about 100 ms ignore bands at token edges. | 3 | No published precision for CrisperWhisper stutter labels exists, so this number is a cheap first ([nyra](https://arxiv.org/abs/2607.18934)) |
| P0-5 | **Submit an ethics application now.** It must cover the listening test with people who stutter, speaker self-ratings, and any future face-video or physiology collection, including consent for face video. | 1, 4, 5, 6, 8 | Required for any study with people who stutter. Paying raters who stutter for emotional labour follows community guidance ([Tang et al.](https://arxiv.org/abs/2605.02080)) |
| P1-6 | Build the **human-reference removal evaluation on FluencyBank Timestamped**, with these metrics: isWER of the edited audio against human intended references plus SeMaScore; residual events per 100 words by type (nyra's CUTOFF/FILLER/REPETITION protocol), framed as DER; fluent words cut, framed as FER deletion with a "≥50% of word duration removed" rule and a sensitivity sweep; ECAPA speaker cosine. | 4 | Breaks the CrisperWhisper circularity. BERTScore stays at 0.84 even for clearly wrong hypotheses, so prefer SeMaScore ([Toyin et al.](https://arxiv.org/abs/2606.31112)); [FER/DER](https://ar5iv.labs.arxiv.org/html/2009.10298) |
| P1-7 | Run a **listening test against unprocessed audio**. Compare five conditions: unprocessed; CleanStream-style muting driven by StutterClear's own detector; Cleanvoice offline; reported numbers from a resynthesis cascade; StutterClear at 250, 400 and 600 ms. Collect clarity, comparison MOS and naturalness from listeners, plus the speaker's own "said what I meant" rating. | 1, 4 | Must beat unprocessed clarity of 2.67 and Cleanvoice's −0.17 comparison MOS ([SpeechAgent](https://arxiv.org/abs/2510.20113)) |
| P1-8 | Log **mouth-to-ear latency, buffer occupancy and drain** across whole conversations, and report end-of-utterance-to-playback delay alongside continuous lag. | 1 | No prior work reports these. Rows to compare against: STEAMROLLER's 3.77 s, SpeechAgent's 0.91 s, and Streaming Parrotron's 320 ms after the end of speech ([STEAMROLLER](https://arxiv.org/html/2601.10223v1)) |
| P1-9 | Write the related-work section around the prior art: IBM/Nuance patents, CleanStream, Detect-and-Pass, Harvill/Shih, FER/DER, TalkLess, Intel. | All | Reviewers who know them will otherwise reject the novelty claims |
| P2-10 | Run **offline replay for blocked-word prediction** on real FluencyBank or UCLASS blocks. Report top-1 and top-3 accuracy with and without the context card, with and without the onset, and against a soft-cue baseline in the style of Intel's patent, plus an abstention curve. Get a freedom-to-operate review before any product use. | 5 | SpeakFaster-style context gains ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11530652/)); Intel's claim scope ([Google Patents](https://patents.google.com/patent/US10672393B2/en)) |
| P2-11 | For **RL**, behaviour-clone the hysteresis/merge/snap rule first. Then fine-tune with Lagrangian PPO or GRPO under P(fluent cut) ≤ ε, conditioning on an ε the user sets, inside a simulator built from silver labels. Report endpointing-style ep50/ep90 latency and premature-cut rates. | 7 | HPO's quality gate is the closest prior design ([HPO](https://arxiv.org/html/2604.21045v1)); metric conventions from the [streaming endpointer](https://arxiv.org/html/2506.07081v1) |
| P2-12 | For **audio-visual work**: get FluencyBank membership through a faculty sponsor. Use Mohapatra's splits to stay comparable, adding speaker-disjoint splits on top. Use full-face, jaw and blink landmarks rather than lips alone. Record at 60 fps where possible. Build a frame-level audio-visual block set from FluencyBank Timestamped plus the ASU onsets, if released. | 6 | Block balanced accuracy 0.58 audio / 0.71 face ([Mohapatra](https://arxiv.org/abs/2406.06964)); access rules ([TalkBank](https://talkbank.org/fluency/)) |
| P2-13 | For **physiology**: collect a 10–20-speaker synchronized dataset (audio, webcam, chest-strap ECG and respiration, wrist EDA) under ethics approval. Treat EDA and heart rate as slow state signals, and test respiration for block lead time. | 8 | A clear advance means AUROC above about 0.65 at sub-second horizons, speaker-disjoint ([Kumar](https://arxiv.org/abs/2610.19235)) |
| P2-14 | Resolve the licences: CrisperWhisper NC for derived labels, SEP-28k NC, FluencyBank and UCLASS terms. | 3, product | See the patent and licence table above |

The venue plan below follows from the evidence. The deadline windows are historical patterns and my own inference, not from the notes, and every one must be checked on the venue's website.

| Paper | Contributions | Best-fit venue | Typical deadline window (verify) | What must be finished first |
|---|---|---|---|---|
| A. Lookahead-indexed distillation for causal stuttering detection, with the silver-label teacher | 2 + 3 + SOTA rows | **Interspeech 2027** | Historically mid-February to early March, about 4–5 months from now | P0-1 to P0-4 |
| B. Real-time real-voice stutter removal with a dual-reference evaluation protocol | 1 + 4 | **ASSETS 2027** if the listening test with people who stutter is done; Interspeech 2027 if only objective evaluation plus a small listening test is ready | ASSETS historically mid-April; CHI 2028 (historically September 2027) as fallback | P0-5 ethics approval, P1-6 to P1-8 |
| C. Silver labels and the v2 generator as a data or resource paper (if not folded into A) | 3 | ASRU 2027 or Interspeech 2027 | ASRU historically late spring in odd years | Silver-label precision against CASA/FluencyBank++ |
| D. Blocked-word prediction with onset constraints | 5 | ASSETS or CHI, after offline replay plus a user study | 2027–2028 cycles | P2-10, plus ethics approval |
| E. Constrained RL for cut/hold/play | 7 | Interspeech or ICASSP 2028 | ICASSP historically mid-September; the 2027 cycle has probably closed | Paper A's student and Paper B's simulator |
| F. Audio + video + physiology stuttering dataset | 6 + 8 | Interspeech 2028 or a datasets-and-benchmarks track | 2028 | Data collection under ethics approval |

The timeline from today follows:

**Late October to December 2026: P0 work.** The evaluations on official splits, CASA and FluencyBank++, the lookahead curve and the silver-label ablations all reuse trained models and existing data. Submit the ethics application in the same window, because its review time is outside the team's control.

**December 2026 to early March 2027: Paper A.** Write Paper A and build the human-reference removal evaluation for Paper B.

**March to April 2027: Paper B.** Run the listening test once ethics approval arrives, then decide between ASSETS and a later CHI submission.

**Later.** Blocked-word prediction, RL and the multimodal dataset follow on longer cycles.

## Conclusion

StutterClear's novelty sits in measurement, not invention. Every core idea, from deleting stutter artifacts before transmission to predicting the blocked word from the stutter sound, was patented between 2011 and 2018, and none of those filings reports a single result. Academic venues reward measured evidence. That makes the gap real, but only for claims phrased as "first implemented and evaluated". The lookahead curve is the hinge of the whole project. It is the one clean first, it sets the delay the trim and clean modes must pay, and together with Kumar's forecasting work it unifies prediction and detection on one time axis that nobody has drawn.

The pipeline's biggest hidden weakness is that one model runs through it twice. CrisperWhisper would generate the training labels and also count the residual stutters in the output, so a system could look clean just by agreeing with CrisperWhisper's blind spots, which include cut-offs, prolongations and blocks. Human intended references on FluencyBank Timestamped, frame annotations on CASA, and a listening test against unprocessed speech are the three independent checks that break that loop. The listening test is the decisive one, because the only listener evidence so far says that cutting the real voice can sound worse than doing nothing. Running it early, with people who stutter as raters and under ethics approval, is cheaper than discovering the answer after the product is built.
