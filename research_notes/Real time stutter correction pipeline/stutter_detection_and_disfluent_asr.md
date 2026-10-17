# Stutter Detection and Disfluency-Aware ASR for a Real-Time Pipeline (as of Oct 2026)

Scope: the audio side of StutterClear (mic -> stutter detection -> streaming ASR). Covers datasets, detection models and metrics, time-localized detection, verbatim and intended ASR on stuttered speech, telling blocks from pauses (including endpointing), real-time feasibility, and open problems. Older work (before 2024) is marked **[foundational]**.

**Comparability warning (applies throughout):** reported numbers are rarely comparable across papers. They differ in:
- Splits: the original SEP-28k split is not speaker-exclusive, while SEP-28k-E is.
- Label threshold: "at least 1 of 3 annotators" versus "at least 2 of 3".
- Granularity: 3-s clip-level F1, frame-level F1, time-F1 at IoU>0.5, or type accuracy only on regions already known to be dysfluent.
- Data: synthetic versus real.
- Reference transcript for ASR: verbatim versus intended.

---

## 1. Datasets: size, labels, clip lengths, licensing, label-quality problems

### Takeaway
Real, public stuttering data is small and mostly labelled at clip level:
- SEP-28k: about 23 h of 3-s English clips.
- FluencyBank: 3.5 h.
- KSoF: German, about 4.6 h.
- AS-70: Mandarin, 48.8 h, with verbatim character-level labels. This is the largest real corpus.

Frame- or word-level timestamps exist only in small add-ons: FluencyBank Timestamped, FluencyBank++, CASA, and Boli. Large time-aligned corpora are synthetic (VCTK-TTS, Libri-Dys and Libri-Dys-Co, LibriStutter) and show a clear gap to real speech. Block and prolongation labels are the least reliable everywhere.

### Cited Findings

**SEP-28k (Apple, ICASSP 2021) [foundational]**
- 28,177 clips of 3 s each, about 23 h, drawn from 385 episodes across 8 podcast shows. Clips were extracted with a VAD "near pauses". — [Lea et al. 2021](https://arxiv.org/abs/2102.12394)
- Labels are binary and multi-label: Block ("gasps for air or stuttered pauses"), Prolongation, Sound Repetition, Word/Phrase Repetition, Interjection, No dysfluency. Non-stutter labels are Natural pause (8.5% of clips), Unintelligible, Unsure, No speech, Poor audio and Music.
- Prevalence when at least 2 of 3 annotators agree: Block 12.0%, Prolongation 10.0%, Sound rep 8.3%, Word rep 9.8%, Interjection 21.2%, No dysfluency 56.9%. — [Lea et al. 2021](https://arxiv.org/abs/2102.12394)
- Annotators were 3 or more trained non-clinicians. Fleiss κ: word rep 0.62, interjection 0.57, sound rep 0.40, no-dysfluency 0.39, block 0.25, prolongation 0.11. The authors say block results "should be more speculative" because clinicians rely on visible gasping. — [Lea et al. 2021](https://arxiv.org/abs/2102.12394)
- The original split (25k train, 2k validation, 1k test) is not speaker-exclusive. Bayerl et al. later released the speaker-exclusive "SEP-28k-E" partition, and later papers use it. — [Lea et al. 2021](https://arxiv.org/abs/2102.12394); [Wagner et al. 2024](https://arxiv.org/abs/2406.11025); [Bayerl et al. via Gong et al. 2024](https://arxiv.org/abs/2406.07256)
- License: the dataset is "licensed under the Creative Commons Attribution-NonCommercial 4.0 International License". No audio is distributed: scripts download it from podcast URLs, and "original copyright remains with the podcast owners". Raw WAVs are about 32 GB and clipped WAVs about 2.6 GB. — [apple/ml-stuttering-events-dataset](https://github.com/apple/ml-stuttering-events-dataset)
  - Conflict: Kozak (2026) states SEP-28k is "CC BY-SA 4.0". The Apple repo, which is the primary source, says CC BY-NC 4.0. — [Kozak 2026](https://arxiv.org/abs/2604.27279)
- Label-quality audit: a researcher who stutters transcribed 2,621 SEP-28k clips and had to relabel more than 25% (653) of the sampled clips. Most errors came from confusing natural disfluency with stuttering, for example a drawn-out "ummm". — [Sridhar & Wu, Interspeech 2025](https://www.isca-archive.org/interspeech_2025/sridhar25_interspeech.pdf)
- Show-level heterogeneity: predictive signal concentrates in one show (StutterTalk). Other shows have confidence intervals that cross chance. — [Kozak 2026](https://arxiv.org/abs/2604.27279)
- Kozak used only 20,131 clips "with complete label+audio coverage", which suggests not all audio can still be retrieved. Show counts conflict: the original paper says 8 shows, Kozak says "six podcasts", and the USC subset covers five shows. — [Kozak 2026](https://arxiv.org/abs/2604.27279); [Lea et al. 2021](https://arxiv.org/abs/2102.12394); [Pattanayak et al. 2026](https://arxiv.org/abs/2609.32285)

**FluencyBank (TalkBank) and its derived label sets**
- **FluencyBank, SEP-28k-style labels [foundational]:** 4,144 clips of 3 s (3.5 h) from 32 adults who stutter. Apple ignored the original FluencyBank alignments because "the temporal alignment for some transcriptions and dysfluency annotations provided were inaccurate". — [Lea et al. 2021](https://arxiv.org/abs/2102.12394)
- **FluencyBank Timestamped** (Romana, Niu, Perez, Mower Provost; JSLHR 67(11):4203–4215, Oct 2024)
  - Revised transcripts with a timestamp and disfluency label per token: filled pauses, repetitions, revisions, partial words.
  - Built for disfluency detection and "intended speech recognition". It benchmarks Whisper on intended ASR and BERT/Whisper models on disfluency detection.
  - Distributed as a CSV zip on TalkBank. Media requires TalkBank access. — [ASHA](https://pubs.asha.org/doi/10.1044/2024_JSLHR-24-00070); [TalkBank Romana corpus](https://talkbank.org/fluency/access/Derived/Romana.html)
- FluencyBank Timestamped covers 36 speakers, 6.41 h of recordings and 5.13 h of transcribed participant speech. Its labels "are not comprehensive" for prolongations and blocks: the ASU team had to add 129 prolongations and 150 blocks by hand. — [Kumar, Daliri, Gupta 2026](https://arxiv.org/abs/2610.19235)
- It provides parallel verbatim and intended transcripts (3,430 samples used). — [Toyin et al. 2026](https://arxiv.org/abs/2606.31112)
- **FluencyBank++** (Interspeech 2025)
  - 3,017 clips of 5 s with frame-level boundaries for 4 types: interjection (1,130 clips), repetition (921), block (530; 0.23–4.20 s long) and prolongation (436).
  - Annotated by 2 dysfluency specialists plus 3 SLPs. Frame-level Fleiss κ went from 0.47 to 0.71 after consensus.
  - The authors found that 3-s clips truncate partial-word repetitions, which is why they extended clips to 5 s. — [Ghosh et al. 2025 (StutterCut)](https://arxiv.org/abs/2508.02255)
- **CASA clinical annotations of FluencyBank** (Valente et al., Interspeech 2025): timestamped core events (syllable repetition, incomplete syllable repetition, multisyllable unit repetition, prolongation, block) plus secondary/visual behaviours. The "gold standard" test set has 8 recordings with 732 dysfluencies. — [Sušac, Bayerl, Džapo 2026](https://arxiv.org/abs/2606.20338)
- Licensing: FluencyBank Teaching derivatives are CC BY-NC-SA 3.0 under TalkBank Ground Rules, which allow only scripts, not labels or audio, to be redistributed. — [Kozak 2026](https://arxiv.org/abs/2604.27279)

**UCLASS [foundational]**
- Recordings from 128 children and adults who stutter. Only 25 files are annotated, and those lack a block class. — [Zhou et al. 2024 (YOLO-Stutter)](https://arxiv.org/abs/2408.15297)
- The transcribed portion is about 53 min from 25 children who stutter. — [Gong et al. 2024](https://arxiv.org/abs/2406.07256)

**KSoF, Kassel State of Fluency (LREC 2022) [foundational]**
- German stuttering-therapy recordings: 214 recordings from 37 speakers (28 male, 9 female). Tasks are spontaneous speech, reading and phone calls.
- 5,597 clips of 3 s with 1.5-s overlap, about 4.6 h.
- Labels match SEP-28k plus "Modified speech", i.e. fluency-shaping technique (24.44% of clips). Blocks are 20.74%, natural pauses 1.97%.
- Fleiss κ: block 0.37, prolongation 0.42, interjection 0.78.
- Available "upon request" for research. — [Bayerl et al. 2022](https://arxiv.org/abs/2203.05383); hours from [Gong et al. 2024](https://arxiv.org/abs/2406.07256)

**LibriStutter [foundational, synthetic]**
- 20 h of synthetic stutter inserted into LibriSpeech, 50 non-PWS speakers. The 5 types are equally distributed (20% each, 15,000 events). — [Gong et al. 2024](https://arxiv.org/abs/2406.07256)
- Its unnatural dysfluencies "degrade performance" when used for training. — [Ghosh et al. 2025](https://arxiv.org/abs/2508.02255)
- Licensed for "research use". — [Kozak 2026](https://arxiv.org/abs/2604.27279)

**AS-70 (Mandarin, Interspeech 2024)**
- 48.8 h from 72 adults who stutter (70 participants plus 2 interviewers who also stutter). Contains conversation (Zoom interviews) and voice-command reading of 600 unique car and smart-home commands.
- Verbatim character-level transcripts with inline stutter markup: `[]` word/phrase repetition, `/b` block, `/p` prolongation, `/r` sound repetition, `/i` interjection. Naturally occurring fillers are excluded. Annotation took about 3x the time of fluent-speech annotation.
- For event detection it provides 19,654 conversation clips (5.69 s mean) and 22,299 command clips (2.87 s mean).
- Distributed via AISHELL. — [Gong et al. 2024](https://arxiv.org/abs/2406.07256)
- Mandarin stuttering is dominated by word/phrase repetitions: 42.7% of events in conversation and 53.0% in commands, against 16.0% in SEP-28k. — [Gong et al. 2024](https://arxiv.org/abs/2406.07256)
- It was re-split for the SLT 2024 challenge because the original split shared command texts between train and test. — [Xue et al. 2024](https://arxiv.org/abs/2409.05430)

**Boli (ICASSP 2025, India)**
- Crowd-sourced from 67 participants, of whom 28 actually stuttered when recorded (25 male, 3 female). About 2.5–2.8 h of read (Rainbow Passage) and spontaneous image-description speech.
- Languages: English, Hindi, Telugu, Bengali, Marathi, Assamese.
- Word-level timestamped annotations of 5 types, but few events: SR 140, Block 70, Prolongation 41, Word rep 21, Interjection 8.
- Released open access. — [Batra et al. 2025](https://arxiv.org/abs/2501.15877)

**Synthetic time-aligned corpora (2024–2025)**
- VCTK-Stutter is 304.66 h and VCTK-TTS is 426.93 h; VCTK-TTS is a VITS-generated text-space simulation.
  - Simulated "blocks" are just 0.5–2.0 s of inserted silence.
  - MOS naturalness is 4.13 for VCTK-TTS and 2.22 for VCTK-Stutter. — [Zhou et al. 2024](https://arxiv.org/abs/2408.15297)
- Libri-Dys-Co: "over 6,000 hours" of simulated co-dysfluency with word/phoneme timestamps. — [Lian et al. 2024 (SSDM 2.0)](https://arxiv.org/abs/2412.00265)

**Other**
- HeardAI: 18 participants, 2.51 h read plus 1.05 h spontaneous voice-AI prompts, transcribed by SLPs. Not public. — [Mujtaba & Mahapatra 2025](https://arxiv.org/abs/2506.00853)
- Amazon in-house corpus: 21 h, 94 speakers, 17K utterances with onset and offset per stutter event. Not public. — [Shonibare et al. 2022](https://arxiv.org/abs/2202.05396)

### Inferences
- For training a time-localized detector on English, the only real, frame-level, public options are FluencyBank-derived sets: FluencyBank Timestamped (36 speakers), FluencyBank++ (about 3k clips) and CASA (8 test recordings). All three come from roughly the same 32–36 speakers. Speaker diversity for frame-level supervision is therefore very low.
- SEP-28k's CC BY-NC 4.0 license plus podcast copyright makes commercial use of models trained on it legally unclear. This matters for productizing StutterClear and needs legal review.
- AS-70 is the best resource for verbatim ASR on stuttered speech, but it is Mandarin. No English corpus of comparable size has verbatim transcripts.

### Gaps
- I did not verify AS-70's exact license terms (AISHELL download page).
- I did not confirm the SEP-28k dead-link or availability rate from a primary source; the 20,131-clip figure is indirect.
- I did not read the LibriStutter primary paper (FluentNet, Kourkounakis et al. 2021); its numbers come via secondary citations.
- Identified but not read: a Kannada syllable-level stutter dataset (1.4 h, 80 speakers, per the [Boli comparison table](https://arxiv.org/abs/2501.15877)); [synthetic Mandarin stuttering for ASR augmentation (2026)](https://arxiv.org/abs/2606.21457); [disability-first co-designed annotation guidelines (2026)](https://arxiv.org/abs/2602.10403); [LLM-Dys synthetic-data analysis (Interspeech 2025)](https://arxiv.org/abs/2505.22029).

---

## 2. Model approaches and reported accuracy, per disfluency type; which models localize events in time

### Takeaway
Self-supervised encoders (wav2vec 2.0, WavLM, Whisper encoder), fine-tuned with a multi-label head, dominate clip-level detection:
- Interjections and word repetitions are easy (F1 about 0.7–0.85).
- Sound repetitions and prolongations are mid-range (about 0.5–0.65).
- Blocks are consistently worst (about 0.3–0.45 on SEP-28k-E and AS-70).

Time-localized detection is newer and much less mature. Frame-level F1 on real speech is about 0.45–0.70:
- YOLO-Stutter, SSDM and SSDM 2.0 need a reference text.
- StutterCut and attention-MIL do not need text.
- LLM approaches (Llama-2 fusion, SSDM 2.0) help some classes but are far too heavy for real time.
- General audio LLMs such as GPT-4o perform very poorly at timestamped dysfluency transcription.

### Cited Findings

**Clip-level baselines [foundational]**
- SEP-28k ConvLSTM baseline on 40-d mel filterbank plus pitch, with multi-task concordance-correlation loss. F1 on the SEP-28k test set:
  - Block 55.9, Prolongation 68.5, Sound rep 63.2, Word rep 60.4, Interjection 71.3, Any 66.8.
  - FluencyBank: Any 80.8.
  - These use the non-speaker-exclusive split and are not comparable to SEP-28k-E numbers. — [Lea et al. 2021](https://arxiv.org/abs/2102.12394)
- AS-70 event-detection baselines (F1 for /p, /b, /r, [], /i):
  - StutterNet (TDNN, 12.2M params): 61.1 / 33.3 / 47.8 / 50.1 / 58.8.
  - ConvLSTM (1.6M): 33.3 / 18.2 / 30.2 / 64.0 / 46.7.
  - Conformer (9.7M): 66.8 / 30.9 / 46.8 / 65.5 / 73.1.
  - Fine-tuned wav2vec2-base: 70.5 / 42.5 / 65.8 / 78.5 / 83.8.
  - Block is "substantially worse ... probably attributed to the fact that blocks are often represented solely by silence". — [Gong et al. 2024](https://arxiv.org/abs/2406.07256)
- SLT 2024 Mandarin challenge: best average F1 was 66.93 (Conformer + BiLSTM with 5 binary heads). Best block F1 was 44.07. Speed perturbation and silence or repeat insertion augmentation helped. — [Xue et al. 2024](https://arxiv.org/abs/2409.05430)
- KSoF baselines: a wav2vec2 + SVM with mean-pooled features scored block F1 0.57 and word rep only 0.17 (word reps are 3.9% of clips). — [Bayerl et al. 2022](https://arxiv.org/abs/2203.05383)

**SSL and Whisper-encoder clip-level state of the art (SEP-28k-E)**
- Multi-label wav2vec2 baseline (Bayerl et al. Interspeech 2023, "A stutter seldom comes alone"): Block 0.32, Interjection 0.77, Prolongation 0.53, Sound rep 0.53, Word rep 0.64. On FluencyBank: Block 0.36, Interjection 0.79. — reported in [Wagner et al. 2024](https://arxiv.org/abs/2406.11025)
- Attention-MIL with a fine-tuned Whisper-medium or WavLM-large encoder (about 307–315M params):
  - Block 0.35, Prolongation 0.47–0.50, Sound rep 0.53, Word rep 0.74–0.80, Interjection 0.82–0.83.
  - Comparators: Miyahara et al. (Interspeech 2025) block 0.30; Haas & Bayerl (multilingual EN/DE/ZH) block 0.33.
  - Binary clip-level F1 on FluencyBank: 0.90. — [Sušac et al. 2026](https://arxiv.org/abs/2606.20338)
- A frozen WavLM Base+ linear probe on SEP-28k gives block F1 0.638 and ROC-AUC 0.724. This uses lenient "at least 1 of 3" labels; with majority-vote labels block F1 falls to 0.346, which shows how much the label threshold matters. — [Pattanayak et al. 2026](https://arxiv.org/abs/2609.32285)
- Whister (Changawala & Rudzicz, Interspeech 2024) uses Whisper-encoder features with 7.1M trainable parameters. — [Ghosh et al. 2025](https://arxiv.org/abs/2508.02255)

**LLM-based detection**
- Wagner et al. (Interspeech 2024) feed Llama-2-7B (LoRA) with a mean-pooled wav2vec2 (315M) embedding plus Whisper-large (1.55B) ASR hypotheses.
  - Blocks improve greatly: SEP-28k-E 0.57–0.62 versus a 0.32 baseline; FluencyBank 0.58 versus 0.36.
  - No gain on word repetitions. The authors suspect Whisper drops repeated tokens.
  - Smaller gains on German KSoF. — [Wagner et al. 2024](https://arxiv.org/abs/2406.11025)
- SSDM 2.0 (Lian et al., Nov 2024): WavLM-large, articulatory "neural articulatory flow", a connectionist subsequence aligner and LLaMA + LoRA, with a reference text prompt.
  - Libri-Dys F1 86.2 (SSDM 81.4).
  - nfvPPA clinical F1 76.8, matching score 70.3.
  - SEP-28k (zero-shot, against YOLO-Stutter's manual timings): Rep 86.8%, Prolong 93.4%, Block 70.0% type accuracy; time-F1 0.830.
  - GPT-4o scored F1 18.3 and matching score 0 on Libri-Dys, and 6.4 on nfvPPA. — [Lian et al. 2024](https://arxiv.org/abs/2412.00265)
- An audio-LLM study on child stuttering speech exists (Sept 2026) but I did not read it. — [Reasoning Beyond Transcription](https://arxiv.org/abs/2609.07968)

**Time-localized (start and end) detection**
- **YOLO-Stutter** (Interspeech 2024; UC Berkeley, UCSF)
  - Input is the soft speech-text alignment from VITS, so it **requires reference text**, then a spatial aggregator and an 8-layer Transformer. It predicts region boundaries plus class with a YOLO loss. 33M trainable params.
  - Time-F1 (counted as true positive on any overlap): UCLASS 0.893, SEP-28k 0.813 (on manually timed subsets).
  - Boundary error of about 20–80 ms on synthetic data.
  - Only 54.2% average accuracy on real aphasia speech, reflecting the gap between simulated and real data. — [Zhou et al. 2024](https://arxiv.org/abs/2408.15297)
- **Stutter-Solver** extends YOLO-Stutter to multilingual speech. "Time and Tokens" reframes detection as ASR-style token transcription and drops time regression. — [SSDM 2.0 related work](https://arxiv.org/abs/2412.00265); [Stutter-Solver](https://arxiv.org/abs/2409.09621); [Time and Tokens](https://arxiv.org/abs/2409.13582)
- **StutterCut** (Interspeech 2025)
  - No transcript needed. Uses Whisper embeddings over 0.75-s windows at a 0.1-s stride, normalized-cut graph clustering, and a Whister classifier with MC-dropout uncertainty.
  - FluencyBank++ time-F1 (IoU>0.5): overall 69.3; prolongation 73.2, repetition 68.8, interjection 64.5, block 70.6. Onset error about 0.3 s.
  - YOLO-Stutter scores 56.8 overall on the same set, because it over-detects.
  - Failure modes: interjections without pause or struggle are missed, and "blocks are often misclassified as pauses". — [Ghosh et al. 2025](https://arxiv.org/abs/2508.02255)
- **Attention-based MIL** (June 2026): trained only on clip labels, it outputs 20-ms frame predictions. On CASA gold frame-level F1 (single label):
  - Whisper + attention pooling 0.70, WavLM + attention 0.56.
  - YOLO-Stutter 0.47, StutterCut 0.45.
  - It struggles with long blocks that span more than its 3-s context. — [Sušac et al. 2026](https://arxiv.org/abs/2606.20338)
- Older frame-level work:
  - Harvill et al. (Interspeech 2022) is frame-level with BiLSTM. It was weak on FluencyBank++: time-F1 21.6. — [Ghosh et al. 2025](https://arxiv.org/abs/2508.02255)
  - Shih et al. (SLT 2024) do word-level detection with an SSL encoder and MIL: FluencyBank clip F1 0.85. — [Sušac et al. 2026](https://arxiv.org/abs/2606.20338)
- **Amazon Detect-and-Pass** [foundational]: a frame-level binary stutter classifier on 100-ms frames with ±4 frames of context, using 2 CNN blocks and 2 fully connected layers. The target was PR-AUC above 75%. — [Shonibare et al. 2022](https://arxiv.org/abs/2202.05396)

**Anticipation and forecasting (new, 2026)**
- Kozak: a 616K-parameter CNN on SEP-28k predicts whether the *next* 3-s clip contains a dysfluency.
  - Aggregate AUC 0.581.
  - By upcoming type: block 0.601, sound rep 0.617, prolongation about 0.52, word rep about 0.49, interjection about 0.45.
  - Event-detection AUC is 0.649 for the CNN versus 0.709 for a wav2vec2-base linear probe. — [Kozak 2026](https://arxiv.org/abs/2604.27279)
- ASU continuous onset forecasting runs every 100 ms with a frozen Whisper-large-v3 encoder over 2-s windows and a causal Conformer (3.67M trainable params).
  - 300-ms-horizon AUROC 0.630 (AP lift 1.56x over the 4.93% prevalence).
  - Per type: repetition 0.630, prolongation 0.655, block 0.612.
  - Within-speaker AUROC is only about 0.54 for blocks and prolongations, with intervals overlapping chance. — [Kumar, Daliri, Gupta 2026](https://arxiv.org/abs/2610.19235)

### Inferences
- For StutterClear the practical choice is a frame-level, transcript-free model. The candidates are MIL-trained or Whister/StutterCut-style heads on an SSL or Whisper encoder.
- YOLO-Stutter, SSDM and SSDM 2.0 assume a known reference text (read speech or pronunciation training). That does not fit open conversation unless an ASR hypothesis is used as the "reference". Doing so feeds ASR errors back into detection: ASR tends to smooth exactly the dysfluencies being detected.
- Block detection from audio alone has a ceiling around F1 0.35–0.45 on realistic, speaker-independent data. Product logic should not depend on high-precision block classification.

### Gaps
- No paper reports frame-level F1 per disfluency type on a large real, multi-speaker, speaker-independent English benchmark. CASA (8 recordings) and FluencyBank++ (about 3k clips, about 32 speakers) are small.
- I did not obtain the original StutterNet or FluentNet numbers from their primary papers; they appear here only via the AS-70 and YOLO-Stutter tables.
- I did not read Stutter-Solver or Time and Tokens metrics directly.

---

## 3. Disfluency-aware or verbatim ASR: WER on stuttered speech, Whisper behavior, fine-tuning, Apple and Google work

### Takeaway
Off-the-shelf ASR roughly doubles WER on stuttered speech compared with fluent speech, and severe stutterers see 25–50% WER. Errors are mostly insertions caused by part-word and sound repetitions.

Whisper behaves unpredictably:
- It sometimes cleans disfluencies (v3 more than v2).
- It sometimes hallucinates. Sound repetitions triggered hallucinations in 22.6% of v2 outputs and 6.9% of v3 outputs, including degrading onomatopoeia.
- Its policy for verbatim versus intended output is uncontrolled.

What works:
- Endpointer and decoder tuning (Apple: WER 25.4% to 9.9%).
- Small-data personalization (Whisper-small LoRA: about 16% to about 9.4–9.8% WER).
- Fine-tuning on stuttered data (AS-70: Whisper CER 27.2 to 8.75).
- Explicit verbatim/intended "mode" control.

Autoregressive models (Whisper, Canary) suit intended transcripts. CTC models suit verbatim transcripts. Streaming-style models scored worse in the one benchmark that included them.

### Cited Findings

**Apple, CHI 2023 [foundational, but the key industry paper]**
- Surveyed 61 people who stutter; ASR data from 91 people who stutter.
- Baseline consumer ASR WER: mild 4.8%, moderate 13.6%, severe 49.2%; all participants 19.8%.
- Of errors, 80.9% were insertions, 17.5% substitutions and 1.6% deletions.
- WER correlates with part-word repetitions (r=.85) and whole-word repetitions (r=.60), but not significantly with blocks (r=.21) or prolongations. — [Lea et al. 2023](https://arxiv.org/abs/2302.09044)
- Three interventions on the evaluation set (moderate and severe speakers):
  - Decoder tuning (higher LM weight plus a higher insertion penalty) took WER from 25.4% to 12.4%.
  - Adding dysfluency refinement (remove "oh" fillers and drop low-probability adjacent repeats using an n-gram LM) took WER to 9.9%, a 61.2% relative reduction.
  - The paper also covers endpointer threshold tuning; see Q4. — [Lea et al. 2023](https://arxiv.org/abs/2302.09044)
- Users dislike seeing repeated words and fillers in dictation. Refinement can wrongly delete intentional repeats such as "so so happy". The authors recommend making these interventions opt-in. — [Lea et al. 2023](https://arxiv.org/abs/2302.09044)

**Bias benchmarks**
- Mujtaba et al. (NAACL 2024) tested 6 ASRs on FluencyBank, comparing stuttered audio with a TTS-generated fluent counterpart:
  - Mean WER 28.1% versus 10.5%.
  - Whisper (API "whisper-1") 12.1% versus 6.3%. Azure 16.8% versus 7.3%. wav2vec2-large-960h 38.0% versus 10.1%.
  - Synthetic LibriSpeech with inserted stutters: Whisper 19.8% versus 3.1%.
  - Inter-word and intra-word blocks changed WER least (Whisper 0.07 versus 0.04, and 0.04 versus 0.02). Prolongations, interjections and repetitions hurt most. — [Mujtaba et al. 2024](https://arxiv.org/abs/2405.06150)
- Sridhar & Wu (Interspeech 2025) ran Whisper on 2,621 manually transcribed SEP-28k clips:
  - Fluent WER: v2 0.234, v3 0.230.
  - Stuttered WER: v2 0.489 against semantic transcripts and 0.435 against verbatim; v3 0.420 and 0.414.
  - Sound repetitions give 13–25% higher WER than the stuttered average.
  - Hallucination: v2 14% stuttered versus 3.3% fluent; v3 2.9% versus 0.9%. With sound repetitions, v2 22.6% and v3 6.9%.
  - v3 regressed on verbatim word repetitions and often outputs "so just" for "j-j-j-just".
  - New harm category: "degrading onomatopoeic" hallucinations such as "oink oink oink" and "blah blah blah". — [Sridhar & Wu 2025](https://www.isca-archive.org/interspeech_2025/sridhar25_interspeech.pdf)
- Boli: Whisper WER 4.55% on concatenated speech versus 21.27% averaged per speaker; wav2vec2 29.22% versus 55.81%. Averaging per speaker exposes the severe tail. — [Batra et al. 2025](https://arxiv.org/abs/2501.15877)

**Verbatim versus intended references (2026)**
- 11 open ASR models on FluencyBank Timestamped (isWER = intended WER, vWER = verbatim WER):
  - Best intended: NVIDIA Canary-1B-v2, isWER 13.85.
  - Best verbatim: NVIDIA Conformer-CTC, vWER 17.20.
  - Whisper-large-v3: isWER 16.13, vWER 25.01.
  - NVIDIA Conformer-Transducer: 23.84 / 28.30.
  - SpeechBrain streaming model: 47.08 / 42.81.
  - Autoregressive seq2seq models specialize in intended output; CTC models in verbatim output.
  - Hardest events: incomplete-syllable and multisyllable-unit repetitions. Easiest: prolongations. — [Toyin et al. 2026](https://arxiv.org/abs/2606.31112)
- Controllable verbatim ASR (nyra labs, July 2026; Whisper-medium with decoder "mode tags"):
  - English disfluency event F1: Whisper 12.0%, CrisperWhisper 73.2%, tagged fine-tune 90.7%.
  - Up to about 60% of reported WER on AMI comes from style mismatch, not recognition errors.
  - Word timing on FluencyBank: base Whisper cross-attention 568-ms mean absolute error, WhisperX 200 ms, MFA 142 ms, CrisperWhisper+s 122 ms, theirs 102 ms. — [Wagner, Zusag, Thallinger 2026](https://arxiv.org/abs/2607.18934)
- In the scoping review of 228 papers, about 59% of stuttered-ASR work targets intended ASR, usually without saying so.
  - Survey of 40 PWS and 30 SLPs: PWS prefer intended transcripts for communication; SLPs prefer verbatim. — [Toyin et al. 2026 scoping review](https://arxiv.org/abs/2604.20535)

**Fine-tuning and personalization**
- Whisper-small with LoRA (r=128):
  - Zero-shot WER on HeardAI: read 33.9%, spontaneous 22.2%.
  - Generalized fine-tune on FluencyBank: 16.1% and 18.4%.
  - Personalized fine-tune on read plus spontaneous: 9.39% and 9.81%.
  - Most of the gain arrives after about 25% of a speaker's 60 read prompts; returns are marginal beyond 50%. — [Mujtaba & Mahapatra 2025](https://arxiv.org/abs/2506.00853)
- AS-70 character error rate (CER):
  - Whisper-large-v2 zero-shot 27.20% overall and 95.33% for severe speakers, because it emits an "overabundance of repeat labels". After AS-70 fine-tuning: 8.75%.
  - WeNet Conformer: 15.00% to 6.24%.
  - HuBERT-large: 19.06% (AISHELL fine-tune) to 7.25%. — [Gong et al. 2024](https://arxiv.org/abs/2406.07256)
- SLT 2024 challenge ASR track: best CER 12.30% versus a 19.18% baseline. The winner used an E-Branchformer with about 1,443 h of augmented data (silence insertion to mimic blocks, part repetition, speed and tempo perturbation, GAN vocoders). — [Xue et al. 2024](https://arxiv.org/abs/2409.05430)
- Detect-and-Pass (Amazon RNN-T): skipping frames flagged as stutter gave a relative WER reduction of 12.18% to 71.24%. Gains were largest for the RNN-T trained on 200K h and smallest for 800K h. — [Shonibare et al. 2022](https://arxiv.org/abs/2202.05396)
- Stutter-TTS augmentation gave a 5.7% relative WER reduction on stuttered utterances. — [Sridhar & Wu 2025 citing Zhang et al. 2022](https://www.isca-archive.org/interspeech_2025/sridhar25_interspeech.pdf); [Stutter-TTS](https://arxiv.org/abs/2211.09731)
- SSDM 2.0 zero-shot ASR on Libri-Dys: WER 3.92% versus Whisper 4.17% (synthetic data). — [Lian et al. 2024](https://arxiv.org/abs/2412.00265)

**Google Euphonia and Project Relate**
- Euphonia covers atypical speech including stuttering. WER for disordered speech can reach "50% or even 90%".
- Personalizing on data from more than 2,000 speakers cut median WER by more than 80% on average, and personalization works with 3–4 min of speech using layer freezing.
- Project Relate is an Android app that builds a personalized recognizer with "Listen" and "Assistant" modes.
- I found no stuttering-specific numbers. — [Google Research blog](https://research.google/blog/responsible-ai-at-google-research-ai-for-social-good/); [Google blog](https://blog.google/outreach-initiatives/accessibility/speech-accessibility-project/)
- Conflict: Apple's CHI paper describes Euphonia-style personalization as needing "15–120 minutes of speech", against Google's "3–4 minutes". — [Lea et al. 2023](https://arxiv.org/abs/2302.09044)
- Mujtaba et al. argue that Euphonia efforts "often neglect the unique aspects of stuttered speech". — [Mujtaba et al. 2024](https://arxiv.org/abs/2405.06150)

### Inferences
- Run two outputs. A verbatim or CTC-style stream (or a mode-tagged model) feeds stutter detection and correction. An intended stream feeds the LLM and TTS. This matches the stated preferences of both stakeholder groups.
- Whisper-family models are risky in a live pipeline because of hallucinations on sound repetitions and inconsistent policy across versions. If one is used, add hallucination guards: cross-check against a CTC hypothesis and filter insertions with low acoustic support.
- Per-user LoRA personalization from a few minutes of enrollment speech is the best-evidenced lever for WER.
- Decoder tuning plus n-gram repeat removal (Apple) is a cheap, streaming-compatible baseline.

### Gaps
- No published WER for streaming or low-latency ASR on stuttered English under matched conditions. Toyin et al. report only one streaming SpeechBrain model (much worse), and its training data differs.
- No Google paper with stuttering-specific Project Relate or Euphonia WER was found.
- I did not read CrisperWhisper's primary paper; its numbers come via nyra labs.
- Identified but not read: [STEAMROLLER multi-agent inclusive ASR (Jan 2026)](https://arxiv.org/abs/2601.10223); [StutterZero/StutterFormer end-to-end correction (Oct 2025, high-school-authored preprint)](https://arxiv.org/abs/2510.18938); [Indonesian stuttering-aware ASR (Jan 2026)](https://arxiv.org/abs/2601.03727).

---

## 4. Telling silent or tense blocks from normal pauses; endpointing and VAD failures for people who stutter

### Takeaway
No validated acoustic rule separates a silent block from a fluent pause. The evidence that exists:
- Blocks have the lowest annotator agreement.
- Detectors confuse blocks with pauses.
- Practitioners who stutter rely on voice-quality cues: strain, a change in breathing or tempo, and short tense silences versus relaxed thinking pauses.
- Some pre-block prosodic tension is detectable slightly before onset (AUC about 0.60).

Silence-based endpointing is the main real-world failure. Apple's baseline endpointer cut off 23.8% of utterances from people who stutter, against a 3% design target. Raising the threshold fixed truncation at a cost of about 1.2–2.7 s extra median delay.

Front-end VAD and denoising also erase blocks before the detector sees them.

### Cited Findings

**Label definitions and agreement**
- SEP-28k labels Block as "gasps for air or stuttered pauses" and keeps a separate "Natural pause" label: 8.5% of SEP-28k clips and 2.7% of FluencyBank clips.
- Block κ = 0.25. "Blocks ... are difficult to detect because the gasp for breath or pause is often inaudible." — [Lea et al. 2021](https://arxiv.org/abs/2102.12394)
- KSoF block κ = 0.37 in the main stage. — [Bayerl et al. 2022](https://arxiv.org/abs/2203.05383)

**Practitioner heuristic (lived experience)**
- "When someone stutters, they often change their tempo of speaking, change their breathing, or their voice becomes strained. A small pause where someone's voice is strained is a block, but a long pause where someone is thinking and their voice sounds fine is fluent speech." — [Sridhar & Wu 2025](https://www.isca-archive.org/interspeech_2025/sridhar25_interspeech.pdf)

**Onset annotation**
- Block onset was annotated "at the point where speech stops or an articulatory posture is established, i.e. the start of the silent or tense interruption". Clip-level κ for blocks was 0.56 and the median onset discrepancy was 80 ms. — [Kumar et al. 2026](https://arxiv.org/abs/2610.19235)

**Detector confusion**
- "Blocks are often misclassified as pauses." — [Ghosh et al. 2025](https://arxiv.org/abs/2508.02255)
- Models struggle with long blocks that exceed a 3-s window. — [Sušac et al. 2026](https://arxiv.org/abs/2606.20338)
- Mandarin block F1 tops out at about 0.43–0.44. — [Gong et al. 2024](https://arxiv.org/abs/2406.07256); [Xue et al. 2024](https://arxiv.org/abs/2409.05430)

**Precursor and acoustic cues**
- A model trained on generic "upcoming disfluency" becomes selective for severe events: block prediction AUC 0.601 and sound repetitions 0.617, while fillers stay at chance. Masking the last 1,024 ms of the input lowers block AUC by 0.021, i.e. some of the signal sits in the final second.
- The authors' mechanistic claim, citing Bloodstein's handbook: vocal-fold adduction, subglottal pressure build-up and laryngeal tension "typically begin hundreds of milliseconds before the audible silence". — [Kozak 2026](https://arxiv.org/abs/2604.27279)
- Causal forecasting of blocks: pooled AUROC 0.612, but within-speaker about 0.541 (not significant), and only 23 speakers had enough block frames. — [Kumar et al. 2026](https://arxiv.org/abs/2610.19235)
- Clinical and physiological literature, weak or secondary sources:
  - Electroglottography (EGG) parameters of people who stutter differ from controls. — [J. Laryngology and Voice 2013](https://journals.lww.com/jolv/fulltext/2013/03010/laryngeal_movements_in_stutterers.4.aspx)
  - Silent laryngeal blocks are attributed to vestibular-fold closure. This is a non-peer-reviewed site and should be treated cautiously. — [casafuturatech](https://casafuturatech.com/science-100/)
  - Some "silent blocks" may actually be tensionless hesitations. — [Stuttering Foundation](https://www.stutteringhelp.org/content/when-it-isnt-stuttering-something-else)

**Endpointing failures (Apple CHI 2023)**
- 41% of survey respondents said voice assistants cut them off "always" (3.3%) or "often" (37.7%). 77.2% of those cut off attributed it to a pause or block being read as end of turn. — [Lea et al. 2023](https://arxiv.org/abs/2302.09044)
- The baseline endpointer was tuned for 3% truncation in the general population but truncated 23.8% of utterances from people who stutter. 7 of 41 participants had more than 50% of utterances cut off.
- Truncation correlated with block rate (r=.64), part-word repetitions (r=.44) and interjections (r=.48). It did not correlate significantly with prolongations or word repetitions. — [Lea et al. 2023](https://arxiv.org/abs/2302.09044)
- Thresholds were re-tuned on severity-specific data:

| Endpointer tuning | Truncation | P50 delay | P95 delay |
|---|---|---|---|
| General population (baseline) | 23.8% | 450 ms | 1,270 ms |
| "Mild" threshold | 4.9% | 1,670 ms | 2,750 ms |
| "Moderate" threshold | 2.5% | 2,190 ms | 3,170 ms |
| "Severe" threshold | 0.8% | 3,100 ms | 3,840 ms |

  — [Lea et al. 2023](https://arxiv.org/abs/2302.09044)
- Apple recommends letting users opt into endpointer settings, flexible pause length for wake words, and checking punctuation models, because users report punctuation being inserted during blocks or pauses. — [Lea et al. 2023](https://arxiv.org/abs/2302.09044)
- The 2026 scoping review describes voice assistants stopping listening during blocks, repetitions or long pauses ("impatient ASR"); this wording comes from a search excerpt. — [Toyin et al. 2026](https://arxiv.org/abs/2604.20535)

**Front-end VAD and denoising destroy block evidence**
- Setup: a simulated chain of spectral-gate denoising, loudness normalization to −23 LUFS, Opus 16 kbit/s and WebRTC VAD (aggressiveness 3), applied to SEP-28k.
  - Block F1 fell from 0.638 to 0.465 (−17.3 points, a 27% relative drop).
  - VAD deletion alone cost −12.0 points.
  - VAD removed 78% of clean silence and 41% of duration. Duration-matched random deletion cost only −1.1 points.
- Threshold re-tuning recovers most block F1 (0.630), but AUC stays degraded (0.620 versus 0.724 clean).
- Under held-out shows, block recall fell from 0.718 to 0.383: the detector "mostly stops finding blocks". — [Pattanayak et al. 2026](https://arxiv.org/abs/2609.32285)

**Effect on ASR versus endpointing**
- Blocks barely change ASR WER (synthetic inter-word blocks: Whisper 0.07 versus 0.04). — [Mujtaba et al. 2024](https://arxiv.org/abs/2405.06150)
- They dominate truncation instead. — [Lea et al. 2023](https://arxiv.org/abs/2302.09044)

### Inferences
- In StutterClear, blocks are mainly a **turn-taking and endpointing problem**, not an ASR-accuracy problem.
- Use semantic or contextual endpointing plus a per-user adaptive silence timeout, starting near Apple's "mild" or "moderate" operating points. Feed a "block-in-progress" probability from the detector into the endpointer so it extends the timeout instead of using a fixed silence timer.
- Never run aggressive VAD, noise suppression or silence trimming *before* the stutter detector. Tap the raw mic stream for detection and send processed audio only to ASR. If a VAD or codec is unavoidable, recalibrate detector thresholds on processed audio.
- Plausible block-versus-pause features (my inference, not validated in the literature):
  - Pre-silence voice-quality strain: jitter, shimmer, spectral tilt and abrupt F0 or energy cut-offs.
  - Silence position: mid-word or before a word onset, versus at a syntactic boundary.
  - Inhalation or gasp noise.
  - Missing turn-final prosody (no final lengthening or F0 fall).
  - An incomplete utterance from the LM's point of view.
- A pause at an incomplete syntactic point that follows strained phonation is more likely a block.

### Gaps
- I found no peer-reviewed study giving duration distributions or a classifier that separates silent blocks from fluent pauses acoustically, for example with EGG, EMG or breath features. This is a clear open problem.
- I found no paper proposing or evaluating an adaptive or learned endpointer specifically for people who stutter beyond Apple's threshold tuning, for example a semantic end-of-turn model trained on stuttered speech.
- Kozak's mechanistic account of laryngeal precursors is not tested directly: the prosodic-probe sample was too small (n=83).

---

## 5. Real-time feasibility: latency, model size, streaming windows, on-device

### Takeaway
Small detectors are cheap. A 616K-parameter CNN runs in 0.25–0.55 ms per 3-s window on Apple Neural Engines, using under 1% of the real-time budget at 4 Hz.

The real limits are:
1. **Algorithmic latency.** Most models need 0.75–3 s of context. Repetitions and long blocks cannot be confirmed until enough of them has been heard. Bidirectional and BiLSTM models are not causal.
2. **Accuracy of small or causal models**, which trail large SSL encoders.

The best-performing localized models (Whisper or WavLM-large MIL, YOLO-Stutter, SSDM 2.0, Llama-based) are offline, need transcripts, or both. Very few papers report latency at all.

### Cited Findings

**On-device CNN (Kozak 2026)**
- 616K params. Exports: CoreML 1.19 MB, ONNX 40 KB, TFLite fp16 1.18 MB.
- Neural Engine latency per 3-s window: 0.25 ms (A19 Pro), 0.29 ms (A18), 0.55 ms (A15 and M1 Max).
- A 4-Hz streaming simulation with a 3-s rolling window and 0.25-s hop took 1.34 ms per window including mel computation (P95 1.83 ms), i.e. 0.54% of the 250-ms budget.
- Platt calibration brought ECE from 0.177 down to 0.010.
- On iPhone, CoreML ignored the GPU hint; only the CPU_AND_NE or ALL compute settings reliably engaged an accelerator. — [Kozak 2026](https://arxiv.org/abs/2604.27279)

**Causal forecasting (ASU 2026)**
- Decision every 100 ms. A frozen Whisper-large-v3 encoder runs on each 2-s window (windows overlap by 1.9 s), followed by a causal Conformer (3.67M params, about 20 s attention context, about 22 s receptive field).
- Context gains saturate at about 5 s of history (AUROC 0.621 versus 0.592 at 2 s).
- At the max-F1 threshold it anticipates 43.9% of events with a mean **lead time of 196 ms**, but produces 27.6 false warnings per minute.
- With the alarm active 25% of the time it catches 58.6% of events; at 66.5% duty it catches 91.6%.
- Feature extraction took 0.9 GPU-hours offline. No on-device latency is reported. — [Kumar et al. 2026](https://arxiv.org/abs/2610.19235)

**Window sizes and compute of other detectors**
- Detect-and-Pass: 100-ms frames plus ±4 neighbouring frames (900-ms context, implying about 400 ms of look-ahead). Small CNN with fully connected layers. Tested with low-frame-rate (LFR) RNN-T stacking of 3 frames. — [Shonibare et al. 2022](https://arxiv.org/abs/2202.05396)
- StutterCut: 0.75-s windows at a 0.1-s stride, about 2.2 s of processing per 1 min of audio (offline). It needs 100 MC-dropout passes and N-cut over the whole utterance, so it is not streaming as published. — [Ghosh et al. 2025](https://arxiv.org/abs/2508.02255)
- MIL: 20-ms frame outputs, but from 3-s clips through a 4-layer BiLSTM and a 307–315M-param encoder. That means about 3 s of algorithmic latency, and the model is non-causal. — [Sušac et al. 2026](https://arxiv.org/abs/2606.20338)
- YOLO-Stutter has 33M trainable params plus VITS encoders and needs reference text and alignment. Training took 70 h on an A6000. No inference latency is reported. — [Zhou et al. 2024](https://arxiv.org/abs/2408.15297)
- SSDM 2.0 combines WavLM-large, articulatory inversion and a LLaMA LoRA (embedding size 4096). Training is about 10x faster than SSDM. No real-time claim. — [Lian et al. 2024](https://arxiv.org/abs/2412.00265)
- The LLM detector (Wagner 2024) chains Llama-2-7B, wav2vec2 (315M) and Whisper-large (1.55B). — [Wagner et al. 2024](https://arxiv.org/abs/2406.11025)
- AS-70 baseline sizes: ConvLSTM 1.6M, Conformer 9.7M, StutterNet 12.2M, wav2vec2-base about 95M. — [Gong et al. 2024](https://arxiv.org/abs/2406.07256)
- Jouaiti & Dautenhahn (ICASSP 2022) combine MFCC and phoneme probabilities and claim suitability "for real-time applications", but the sources I could reach report no latency figures. — [Univ. Birmingham record](https://research.birmingham.ac.uk/en/publications/dysfluency-classification-in-stuttered-speech-using-deep-learning/)

**Endpointing latency cost**
- Making the endpointer tolerate stutters costs +1.2 s (mild) to +2.65 s (severe) of median delay over the 450-ms baseline. — [Lea et al. 2023](https://arxiv.org/abs/2302.09044)

**Streaming ASR on stuttered speech**
- The only streaming-family models benchmarked scored worse than the offline seq2seq leaders:
  - NVIDIA Conformer-Transducer: isWER 23.84.
  - SpeechBrain streaming: 47.08.
  - Canary-1B-v2 (offline): 13.85.
  - Training data differs, so this is not a controlled comparison. — [Toyin et al. 2026](https://arxiv.org/abs/2606.31112)

### Inferences
- A feasible on-device design:
  - A tiny causal CNN or Conformer on a log-mel stream with an 80–100 ms hop, or a small SSL or Whisper-tiny/base encoder distilled into it.
  - Emit a per-frame dysfluency or block probability with about 100–250 ms decision latency.
  - Run a heavier verifier (MIL-style Whisper or WavLM head over a 2–3 s rolling buffer) on server or GPU to confirm and classify, with 1–3 s delay.
  - This fits the pipeline because correction and TTS need word-level context anyway.
- Under the "ultra-low end-to-end latency" goal, a repetition generally cannot be labelled until its second or later iteration has been heard. Budget about 300–800 ms of per-event confirmation latency for repetitions and the full duration of a block. Decouple TTS start from final classification by streaming the confident fluent prefix first.
- A false-alarm rate like ASU's 27.6 per minute means anticipatory triggers are not yet usable as hard actions. Use them as soft priors, for example to extend the endpointer timeout.

### Gaps
- No paper reports end-to-end streaming latency for a combined stutter-detection and ASR pipeline.
- No on-device benchmark exists for SSL- or Whisper-based detectors.
- No streaming or causal frame-level detector has been evaluated on CASA or FluencyBank++.
- I could not verify the Jouaiti & Dautenhahn latency claim.

---

## 6. Open problems the literature admits

### Takeaway
The field openly admits seven problems:
1. Poor cross-dataset and cross-speaker generalization.
2. Unreliable labels, especially blocks and prolongations.
3. A large gap between synthetic and real data.
4. Very little true temporal-detection work: about 6 of 72 "detection" papers actually localize events.
5. English-centric data.
6. Unstated choice between verbatim and intended ASR objectives.
7. Almost no deployment, latency or explainability studies.

Blocks are consistently the hardest class, for both detection and endpointing.

### Cited Findings

**Generalization**
- Training on more SEP-28k data improves FluencyBank F1 by 24% relative, and SEP-28k performance is "consistently worse than FluencyBank" because of speaker variety. — [Lea et al. 2021](https://arxiv.org/abs/2102.12394)
- Dataset partitioning that ignores speakers inflates results. — [Bayerl et al. TSD 2022, via Gong et al. 2024](https://arxiv.org/abs/2406.07256)
- Per-show AUC varies from 0.49 to 0.63. — [Kozak 2026](https://arxiv.org/abs/2604.27279)
- Within-speaker AUROC is much lower than pooled AUROC (blocks 0.54 versus 0.61). — [Kumar et al. 2026](https://arxiv.org/abs/2610.19235)
- Training on held-out shows halves block recall. — [Pattanayak et al. 2026](https://arxiv.org/abs/2609.32285)

**Synthetic versus real gap**
- YOLO-Stutter scores above 98% type accuracy on synthetic VCTK-TTS but 54% average accuracy on real aphasia speech. Its "simulated" blocks are plain silences. — [Zhou et al. 2024](https://arxiv.org/abs/2408.15297)
- Synthetic corpora have "robotic distortions" and "overly rigid pauses in blocks". — [Ghosh et al. 2025](https://arxiv.org/abs/2508.02255)
- The scoping review notes that 28% of data-centric papers are synthetic, even though people who stutter are "highly willing to donate recordings". — [Toyin et al. 2026](https://arxiv.org/abs/2604.20535)

**Label reliability**
- Block κ is 0.25 in SEP-28k and 0.37 in KSoF. Prolongation κ is 0.11 in SEP-28k. — [Lea et al. 2021](https://arxiv.org/abs/2102.12394); [Bayerl et al. 2022](https://arxiv.org/abs/2203.05383)
- More than 25% of sampled SEP-28k clips were relabelled by an annotator who stutters. — [Sridhar & Wu 2025](https://www.isca-archive.org/interspeech_2025/sridhar25_interspeech.pdf)
- 3-s clips truncate events. — [Ghosh et al. 2025](https://arxiv.org/abs/2508.02255)

**Task framing**
- 72 papers have "detection" in the title, but only 6 do temporal detection. About 8% address the detection task stakeholders want.
- Language coverage: English in 152 papers, German 22, Mandarin 17.
- Fewer than 20% of papers report stakeholder involvement, and only about 10% release code, models or data. — [Toyin et al. 2026](https://arxiv.org/abs/2604.20535)

**ASR objective and harms**
- Single-reference WER hides model rankings that flip between verbatim and intended references. — [Toyin et al. 2026](https://arxiv.org/abs/2606.31112)
- Whisper hallucinations and v3's regression on verbatim word repetitions. — [Sridhar & Wu 2025](https://www.isca-archive.org/interspeech_2025/sridhar25_interspeech.pdf)
- Up to about 60% of WER comes from style mismatch. — [Wagner, Zusag, Thallinger 2026](https://arxiv.org/abs/2607.18934)

**Severity and variability**
- Stuttering varies by situation, interlocutor and stress. — [Lea et al. 2021](https://arxiv.org/abs/2102.12394)
- Severe speakers see WER near 49% for consumer ASR and CER above 95% for zero-shot Whisper in Mandarin. — [Lea et al. 2023](https://arxiv.org/abs/2302.09044); [Gong et al. 2024](https://arxiv.org/abs/2406.07256)
- None of the challenge teams modelled severity explicitly. — [Xue et al. 2024](https://arxiv.org/abs/2409.05430)

**Event types beyond audio**
- Clinical definitions include secondary behaviours (facial or body movements) and avoidance, which audio-only systems cannot see. SLPs want the analysis of "moments of avoidance and struggle". — [Toyin et al. 2026](https://arxiv.org/abs/2604.20535)

**Forecasting**
- Pre-onset forecasting "remains an open challenge", with high false-alarm rates. — [Kumar et al. 2026](https://arxiv.org/abs/2610.19235)

### Inferences
- StutterClear will need **per-user calibration**: enrollment speech for ASR LoRA, detector thresholds and endpointer timeout. Population models generalize poorly across speakers and situations, and block evidence is speaker-specific.
- Collecting consented, timestamped, verbatim data from target users is likely higher-leverage than more synthetic data. Co-designing labels with people who stutter would also address the reliability problem.
- Design the correction stage to tolerate detector uncertainty. Interjection and word-repetition detection is reliable enough to act on; block and prolongation outputs should be soft signals.

### Gaps
- I found no cross-lingual evaluation of frame-level detectors on Mandarin or Indian-language data beyond Stutter-Solver claims, which I did not read.
- I found no study of how real-time stutter feedback or correction affects users who stutter (acceptability, stigma, anxiety) beyond survey preferences. This is outside this track but relevant to product risk.
- Children's stuttering detection is under-covered here: FluencyBank Teaching cross-corpus AUC was 0.674 in Kozak, and the Sept 2026 audio-LLM child-stuttering paper was not read.
