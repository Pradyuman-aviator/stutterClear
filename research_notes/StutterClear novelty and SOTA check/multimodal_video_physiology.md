# Multimodal stuttering detection (face video, physiological signals): novelty and SOTA check, as of 17 Oct 2026

Scope: (A) audio-visual detection of stuttering moments, especially blocks, from face video; (B) ECG/HRV, respiration, EDA and EMG for detecting or anticipating stuttering moments; (C) whether a combined audio+video+physiology stuttering dataset exists. StutterClear context: the audio-only detector is weakest on silent blocks (clip AUC 0.815). Phase 3 plans webcam MediaPipe landmarks fused with audio, and Phase 4 plans ECG/respiration/EDA as slow "state" signals.

## A. Audio-visual detection of stuttering moments (especially blocks) from face video: what exists, and is it done?

### Takeaway
Verdict: **PARTIALLY DONE.** Audio-visual stuttering detection exists in four lines of work:
- Altinkaya & Smeulders, Oct 2020: AV block detection, but no usable numbers and the dataset site is down.
- Mohapatra et al., Jun 2024: a FluencyBank AV benchmark. For blocks, full-face video beat audio, with balanced accuracy 0.71 vs 0.58.
- MMSD-Net, Jul 2024: methodologically weak.
- Das et al., 2020–2022: facial action units from video predict an upcoming stutter in a lab cued-word task, at about 75–81% accuracy.

What I could not find anywhere (open as of Oct 2026):
- interpretable webcam face-landmark features (lip aperture or press, jaw, blinks, tremor) such as MediaPipe produces;
- streaming, frame-level localisation of silent blocks fused with an audio block probability;
- inferring the stuck onset sound (bilabial P/B/M) from the lips.

### Cited Findings
**Altinkaya & Smeulders (Univ. of Amsterdam), MuCAI'20 workshop at ACM MM 2020, published 15 Oct 2020, pp. 9–13**
- The abstract claims "the first expandable audio-visual database of stuttered speech". It explores "an end-to-end, real-time, multi-modal model for detection and classification of stuttered blocks in unbound speech", using video "since acoustic signals cannot be produced immediately" — [ACM DL abstract](https://dl.acm.org/doi/10.1145/3423325.3423733); [UvA VISLab page](https://ivi.fnwi.uva.nl/vislab/publication/mehmet-mucai-2020/)
- The PDF's internal title differs: "A Dynamically Expandable, Weakly Supervised, Audio-Visual Database of Stuttered Speech" — [ACM PDF](https://dl.acm.org/doi/pdf/10.1145/3423325.3423733)
- Dataset:
  - 25 single speakers (14 female, 11 male) from YouTube videos of 3–10 min each.
  - More than 200 samples of 50–300 s, each with at least 10 stutter incidences.
  - More than 4 h of video in the "current release".
  - Source: [ACM PDF](https://dl.acm.org/doi/pdf/10.1145/3423325.3423733)
- Labels:
  - Six classes: sound repetition, word repetition, phrase repetition, interjection, prolongation and block. Blocks are labelled with "Audio+Video+ASR".
  - Labels are produced automatically (ASR, NLP, audio-to-letter recognition, VAD), with "manual verification in some cases". This is weak or self-supervised labelling, not clinician-annotated.
  - Source: [ACM PDF](https://dl.acm.org/doi/pdf/10.1145/3423325.3423733)
- Model:
  - Video stream: 3D-conv front end, ResNet-34 and a 2-layer BGRU.
  - Audio stream: ResNet-18 with BGRU(512) layers.
  - The two streams are concatenated. This mirrors the Petridis et al. 2018 end-to-end AVSR design, which the paper cites.
  - Source: [ACM PDF](https://dl.acm.org/doi/pdf/10.1145/3423325.3423733); [ACM DL references](https://dl.acm.org/doi/10.1145/3423325.3423733)
- Results: a query of the full 5-page PDF found no exact accuracy or F1 numbers for audio vs video vs fusion, for blocks or otherwise — [ACM PDF](https://dl.acm.org/doi/pdf/10.1145/3423325.3423733)
- Availability:
  - The paper says the dataset is "expandable through www.assistedspeech.org" — [ACM PDF](https://dl.acm.org/doi/pdf/10.1145/3423325.3423733)
  - My check on 17 Oct 2026 got no HTTP response from http://www.assistedspeech.org (curl status 000).
  - The ACM DL lists only 3 citations and 294 downloads up to 14 Oct 2026 — [ACM DL](https://dl.acm.org/doi/10.1145/3423325.3423733)

**Mohapatra, Likhite, Biswas, Islam, Zhu, "Missingness-resilient Video-enhanced Multimodal Disfluency Detection", arXiv 2406.06964, Jun 2024**
- Dataset:
  - Built from FluencyBank, which "contains video recordings from 32 individuals", using the SEP-28k-style 3-s clip labels.
  - Majority-vote distilled to 4,004 clips, about 3.3 h.
  - Clip counts: blocks 469, word repetitions 337, sound repetitions 428, interjections 701, prolongations 407, fluent 1,660.
  - Video at 30 fps.
  - The authors state they release the curated AV dataset with train/test splits. I did not verify the release link.
  - Source: [arXiv 2406.06964](https://arxiv.org/abs/2406.06964)
- Results:
  - Audio-visual fusion improves average balanced accuracy by about 10 points over audio-only, and by 7 points with video missing in 50% of samples — [arXiv 2406.06964](https://arxiv.org/abs/2406.06964)
  - Blocks, balanced accuracy (Table 3):

    | Model | Balanced accuracy |
    |---|---|
    | HuBERT audio-only | 0.58 |
    | Full-face video-only | 0.71 |
    | HuBERT + full face | 0.68 |
    | wav2vec2 + full face | 0.71 |
    | Lip-ROI video-only | 0.57 |
    | HuBERT + lip ROI | 0.57 |

    Source: [arXiv 2406.06964](https://arxiv.org/abs/2406.06964)
  - Lip ROI was worse than the full face, by 14% on average for video-only. The authors attribute this partly to 18% less paired data, because landmark detection failed on low-quality video. They also point to the "paralinguistic" whole-face nature of disfluency — [arXiv 2406.06964](https://arxiv.org/abs/2406.06964)
- Method:
  - Visual features come from the Ma et al. VSR pipeline (3D CNN + ResNet-18). These are learned pixel features, not hand-crafted landmark features.
  - Evaluation holds out 20% of each class. The excerpt does not state that the split is speaker-disjoint.
  - Source: [arXiv 2406.06964](https://arxiv.org/abs/2406.06964)

**MMSD-Net (Nie, Kadiri, Agrawal; UT Dallas / USC / Birmingham), arXiv 2407.11492, Jul 2024**
- It claims to be "the first multi-modal neural framework for stuttering detection", combining video, audio and text transformer encoders fused into an LLM. It ignores Altinkaya 2020 — [arXiv 2407.11492](https://arxiv.org/abs/2407.11492)
- Data:
  - Training: SEP-28k (listed as "Normal Speech", audio only), FluencyBank (52,000) and "Adults Who Stutter" (200).
  - Testing: 500 AWS clips plus 500 clips from SpeakingFaces, a corpus of fluent speakers.
  - Source: [arXiv 2407.11492](https://arxiv.org/abs/2407.11492)
- Results:

  | Model | F1 |
  |---|---|
  | MMSD-Net | 90.19 (P 92.58, R 87.93) |
  | StutterNet | 88.23 |
  | FluentSpeech | 83.72 |
  | ConvLSTM | 80.41 |
  | ResNet+BiLSTM | 73.98 |

  The task is binary stuttered vs not; there are no per-class or block results — [arXiv 2407.11492](https://arxiv.org/abs/2407.11492)

**Das, Mock, Chacon, Irani, Golob, Najafirad (UTSA), arXiv 2010.01231, Oct 2020**
- Setup:
  - 7 male adults who stutter.
  - Cued S1–S2 nonword task with a 1,500 ms speech-preparation period.
  - Face filmed at 58 fps with a Logitech C920 webcam. 17 facial action units were extracted per frame with an SVR-based AU model trained on BP4D, SEMAINE and DISFA.
  - Source: [arXiv 2010.01231](https://arxiv.org/abs/2010.01231)
- A CNN plus DeepSHAP predicts whether the upcoming utterance will be fluent or stuttered "within approximately 2 to 3 seconds", which the authors call the first such work — [arXiv 2010.01231](https://arxiv.org/abs/2010.01231)
- Cheek and lip AUs were the most important predictors (p<0.005) — [arXiv 2010.01231](https://arxiv.org/abs/2010.01231)
- A stuttered trial was a silent block, sound prolongation or part-word repetition, coded by a certified SLP — [arXiv 2010.01231](https://arxiv.org/abs/2010.01231)

**Das et al., Frontiers in Neuroscience, 2022 (16:912798)**
- Same paradigm: 7 AWS and 3,515 labelled trials (class-balanced).
- Results:
  - EEG + facial-AU multimodal self-supervised model: accuracy 80.80%, F1 0.79, AUC 0.81.
  - EEG-only: accuracy 79.04%, AUC 0.79.
  - The prior face-AU-only model (Das et al. 2021): accuracy 75.27%, F1 0.73, AUC 0.82.
- No public data or code; data are available on request.
- Source: [PMC9376608](https://pmc.ncbi.nlm.nih.gov/articles/PMC9376608/)
- Earlier EEG work from the same lab predicted fluent vs stuttered speech from pre-speech neural oscillations with 81% accuracy (Myers et al.) — cited in [arXiv 2010.01231](https://arxiv.org/abs/2010.01231)

**Other prior work**
- Yildirim & Narayanan, 2009: multimodal disfluency-boundary detection in children's spontaneous speech, combining prosodic, lexical and gestural (optical flow) features. Accuracy 80.5% with feature-level fusion and 82.1% with decision-level fusion. The disfluencies were repetitions, repairs, false starts and filled pauses, not blocks — summarised in [Sheikh et al. review, arXiv 2107.04057](https://arxiv.org/abs/2107.04057)
- Valente et al., Interspeech 2025 (arXiv 2506.00644, May 2025): clinical audiovisual annotations of FluencyBank by 3 SLPs in ELAN.
  - Data: 30 reading samples (1 h 10 min) and 36 interview samples (6 h 15 min).
  - Spans: 1,654 in reading and 4,037 in interview.
  - Each span has a primary type, a secondary behaviour and a tension score (0–3). Secondary behaviours include facial grimaces (lip pressing, jaw tension, jaw jerking, eye movements) and head movements.
  - Consensus test set: 265 blocks, of which 202 co-occur with facial grimaces and 75 with head movements.
  - Agreement on tension is poor (Krippendorff's α = 0.18), and annotators agree less on secondary behaviours than on primary types.
  - The authors say the annotations "will be made publicly available".
  - Source: [arXiv 2506.00644](https://arxiv.org/abs/2506.00644); [ISCA archive](https://www.isca-archive.org/interspeech_2025/valente25_interspeech.pdf)
- Search results, not exhaustive, for MediaPipe or face-landmark stuttering detection, video-based SSI-4 physical-concomitant scoring, and AV-HuBERT or lip-reading for dysfluency in 2021–2026: no stuttering-specific work found. Results were MediaPipe docs and AV deepfake work — [web search results incl. MediaPipe Face Landmarker docs](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker); [AV-Lip-Sync+ (deepfake, not stuttering)](https://arxiv.org/pdf/2311.02733)

**Physiological basis for lip, jaw and tremor features**
- Denny & Smith, 1992 (17 PWS): tremor-like 5–15 Hz oscillations and high-amplitude EMG in lip, jaw and neck during stuttering, on a continuum across individuals — [PubMed 1494267](https://pubmed.ncbi.nlm.nih.gov/1494267/)
- Marin et al., Sensors 2024 (55 PWS, 30 controls): higher 5–15 Hz EMG amplitude in zygomaticus major during stuttering — [PMC11053464](https://pmc.ncbi.nlm.nih.gov/articles/PMC11053464/)
- McClean et al., 1984: lower-lip EMG and lip displacement during bilabial disfluencies in 5 adult stutterers — [PubMed 6482403](https://pubmed.ncbi.nlm.nih.gov/6482403/)

### Inferences
- **Novelty of StutterClear Phase 3.** "Video helps block detection" is already shown: Mohapatra 2024 (block balanced accuracy 0.58 → 0.68–0.71) and Altinkaya 2020 (claimed). A paper that only says this would be incremental. The defensible new contributions are:
  1. Explicit, interpretable, on-device landmark features (lip aperture, lip-press, jaw, blink, tremor), as opposed to pixel CNN features.
  2. Streaming, frame-level localisation of silent blocks, fused with an audio block posterior. Prior AV work classifies 3-s clips.
  3. Inferring the place of articulation of the stuck onset (lips pressed → P/B/M). No prior work found.
  4. Speaker-disjoint evaluation, which none of the AV papers above clearly report.
- **Lip-only features may underperform the full face.** Mohapatra found lip-ROI video-only blocks at 0.57 vs 0.71 for the full face. StutterClear's lip-centric plan should therefore include upper-face cues (blinks, brow), head pose, and jaw or cheek tension. Das 2020 likewise found cheek and lip AUs to be the most important.
- **Tremor at webcam frame rates.** A 30 fps webcam has a Nyquist limit of 15 Hz, which sits at the top of the 5–15 Hz stuttering tremor band. Tremor features from landmarks are borderline at 30 fps and better at 60 fps. Das 2020 used 58 fps.
- **MMSD-Net's gains are probably confounded.** Negatives come from a different corpus (SpeakingFaces fluent speakers) and the test set is tiny. The gains may reflect corpus or speaker identity rather than moment-level detection, so it is weak evidence and a weak baseline to cite.
- **Pre-speech facial cues as an anticipation signal.** Das 2020–2022 suggest pre-speech facial cues can do more than detect. They show above-chance anticipation 1.5–3 s ahead, but only in a constrained cued-nonword lab task with 7 speakers.

### Gaps
- Exact Altinkaya & Smeulders block-detection numbers: none were found in the PDF query. Whether the dataset was ever downloadable is also unconfirmed; the site does not respond today.
- Whether Mohapatra's split is speaker-disjoint, and the exact URL and licence of their released FluencyBank AV subset (it is derived from TalkBank media, which may restrict redistribution).
- Whether the Valente et al. 2025 SLP annotations (secondary behaviours, tension) have actually been released as of Oct 2026.
- No paper found on automatic, video-based SSI-4 physical-concomitant scoring, AV-HuBERT for stuttering, or MediaPipe landmarks for blocks. The search was not exhaustive (Google Scholar was not queried directly).

## B. Physiological signals (ECG/HR/HRV, respiration, EDA, EMG) for detecting or anticipating stuttering moments: what exists, and is it done?

### Takeaway
Verdict: **PARTIALLY DONE for detection, OPEN for anticipation.**
- **Arousal.** Sympathetic arousal is reliably higher during stuttered than fluent utterances in preschool children who stutter. Skin conductance response (SCR) amplitude has d = 0.99. But SCRs lag their trigger by about 1–3 s, and arousal explains little variance.
- **Respiration.** One respiration-only classifier separates block from non-block states at 82.6% accuracy. It detects concurrent blocks and does not predict them.
- **Anticipation.** The only stutter-event anticipation results use EEG or facial video in cued lab tasks (75–81% accuracy) or audio alone (AUROC about 0.6, published Apr and Oct 2026).
- **What is missing.** I found no ML that predicts stuttering moments in continuous speech from ECG/HRV, EDA or respiration, and no controlled trial of wearable HRV or EDA biofeedback for stuttering.

### Cited Findings
**Arousal around stuttered vs fluent utterances**
- Walsh & Usler, JSLHR, Dec 2019:
  - 22 preschool children who stutter, mean age 53 months.
  - SCR amplitude higher during stuttered than fluent utterances: t(21) = 4.65, d = 0.99.
  - SCR frequency higher: d = 0.64.
  - Blood pulse volume lower, indicating vasoconstriction: d = −0.53.
  - Pulse rate higher but not significant: p = .07, d = 0.42.
  - SCRs occurred in 72.4% of stuttered vs 62.2% of fluent utterances.
  - Utterances averaged about 8 s, and fluent utterances within 3 s of a stutter were excluded.
  - No onset-locked timing was reported. The authors note typical latencies of about 1 s for BPV and 1–3 s for SCR, and that low temporal resolution limits interpretation.
  - Respiration and RSA were not recorded.
  - Source: [PMC7201324](https://pmc.ncbi.nlm.nih.gov/articles/PMC7201324/)
- Weber & Smith, JSHR, 1990:
  - 19 stutterers and 19 controls; EDA, peripheral blood flow and heart rate measured.
  - No group difference in speech-related arousal.
  - Within stutterers, larger sympathetic increases correlated with occurrence and severity of disfluency in intervals "prior to, during, and after speech", but accounted for "small percentages of the variances".
  - Source: [PubMed 2273884](https://pubmed.ncbi.nlm.nih.gov/2273884/)
- Walsh, Smith, Christ & Weber, Frontiers in Human Neuroscience, 2019:
  - 47 children who stutter and 25 who do not.
  - No group differences in phasic SCR, BPV or pulse rate.
  - Children who stutter had higher tonic skin conductance level across speech and non-speech tasks.
  - Source: [PMC6795148](https://pmc.ncbi.nlm.nih.gov/articles/PMC6795148/)
- Tumanova & Backes, JSLHR, 2019 (32 preschoolers): children who stutter showed higher EDA arousal only in the harder nonword-repetition task, not in picture description — [PubMed 31644362](https://pubmed.ncbi.nlm.nih.gov/31644362/)
- Zengin-Bolatkale, Conture & Walden, J Fluency Disorders, 2015 (37 children who stutter, 39 who do not): tonic SCL was higher only in 3-year-old children who stutter — [PMC4877440](https://pmc.ncbi.nlm.nih.gov/articles/PMC4877440/)
- Other Conture/Walden-lab papers: autonomic nervous system activity of preschool children who stutter to emotional stimuli (Jones et al., 2014) and sympathetic arousal as a chronicity marker (2018) — titles and abstracts listed at [PMC4150817](https://pmc.ncbi.nlm.nih.gov/articles/PMC4150817/); [PMC5826607](https://pmc.ncbi.nlm.nih.gov/articles/PMC5826607/)

**Respiration**
- Villegas et al., IEEE EMBC 2019 (Peru):
  - 68 participants (training: 27 AWS, 33 adults who do not stutter; test: 9 AWS). The subgroups sum to 69 as reported.
  - Respiratory patterns and pulse recorded during a reading task.
  - An MLP classified "block and non-block states" from respiratory biosignals at 82.6% accuracy. It is a concurrent-state classifier.
  - Source: [PubMed 31946902](https://pubmed.ncbi.nlm.nih.gov/31946902/)
  - Das et al. note that no AUC, precision or recall was reported — [arXiv 2010.01231](https://arxiv.org/abs/2010.01231)
- Older kinematic and respiratory literature:
  - Respiratory muscle incoordination during stuttering, measured with oesophageal and gastric pressures in 1990 — [PubMed 2350093](https://pubmed.ncbi.nlm.nih.gov/2350093/)
  - Prephonatory chest-wall posturing "qualitatively identical" in stutterers and normal speakers — [PubMed 6227774](https://pubmed.ncbi.nlm.nih.gov/6227774/)

**ML on physiology in children who stutter (group-level)**
- Sharma, Salekin et al., arXiv 2208.08859 (Aug 2022):
  - 70 preschool children.
  - Signals: heart rate from ECG, EDA, respiratory rate and respiratory amplitude, in 2-s segments.
  - A modality-wise multiple-instance-learning model classifies children who stutter vs those who do not during stress and narration, with SHAP explanations.
  - The task is group classification, not prediction of stuttering moments.
  - Source: [arXiv 2208.08859](https://arxiv.org/abs/2208.08859)

**Anticipation of stuttering events (non-physiological, for comparison)**
- Kozak, arXiv 2604.27279 (30 Apr 2026):
  - A 616K-parameter CNN on SEP-28k predicts whether the next 3-s clip contains a disfluency.
  - Aggregate AUC 0.581. Upcoming blocks: test AUC 0.601 [0.554, 0.651]. Sound repetitions: 0.617.
  - Runs on-device in about 0.25 ms per window.
  - Source: [arXiv 2604.27279](https://arxiv.org/abs/2604.27279)
- Kumar, Daliri, Gupta (ASU), arXiv 2610.19235 (13 Oct 2026):
  - Continuous causal onset forecasting every 100 ms: frozen Whisper-large-v3 + causal Conformer on FluencyBank Timestamped.
  - Data: 36 speakers, 2,872 events, including 150 hand-annotated blocks.
  - Results:
    - AUROC 0.63 at a 300 ms horizon; blocks 0.612.
    - Within-speaker block wAUROC 0.541, with a CI that overlaps chance.
    - Mean lead time 196 ms at the max-F1 threshold, with 27.6 false warnings per minute.
    - At a 25% alarm duty cycle it anticipates 58.6% of events.
  - Source: [arXiv 2610.19235](https://arxiv.org/abs/2610.19235)
  - It also cites MEG evidence that reduced left orbitofrontal activation precedes blocked vocalisation (Sowman et al., 2012) — [arXiv 2610.19235](https://arxiv.org/abs/2610.19235)

**EMG and biofeedback**
- Orofacial EMG during stuttering shows 5–15 Hz tremor-like oscillations — [PubMed 1494267](https://pubmed.ncbi.nlm.nih.gov/1494267/); [PMC11053464](https://pmc.ncbi.nlm.nih.gov/articles/PMC11053464/)
- EMG biofeedback evidence is small and dated: a South African study with 3 participants reported fewer stutters — [SAJCD paper](https://distantreader.org/stacks/journals/sajcd/sajcd-357.pdf)
- A search found no controlled trial of wearable HRV or EDA biofeedback for stuttering. A 2024 review only proposes wearables (HR, EDA, temperature) for identifying stress triggers — [web search summary; e.g., PMC11053464 and patents](https://pmc.ncbi.nlm.nih.gov/articles/PMC11053464/)
- Das et al. cite evidence that EMG feedback of facial muscles increases therapy effectiveness in school-age children — [arXiv 2010.01231](https://arxiv.org/abs/2010.01231)

### Inferences
- **EDA and pulse cannot act as sub-second precursors.** EDA and BPV responses lag their trigger by about 1–3 s and about 1 s respectively (Walsh & Usler 2019). They are plausible only as Phase 4's slow "state" covariates: tonic SCL, HR or HRV trend, breathing pattern over tens of seconds, used to modulate a prior on block risk.
- **Physiology is a weak, speaker-dependent prior.** Weber & Smith 1990 found arousal explains only a small share of disfluency variance. Expect any physiology-only event predictor to be weak.
- **Respiration is the most promising physiological channel for blocks.** Breath-holding or prephonatory chest-wall posturing could co-occur with or slightly precede silent blocks. Villegas 2019 shows a concurrent block signal exists in respiration (82.6% accuracy), but nobody has reported lead time. An RIP belt or chest-strap respiration channel fused with audio for silent-block detection or anticipation is open.
- **The bar for multimodal anticipation is low.** Audio-only anticipation in Oct 2026 is weak (AUROC about 0.6; block within-speaker about chance). Any multimodal anticipation result above about 0.65 AUROC at sub-second horizons would be a clear advance, if evaluated speaker-disjointly.

### Gaps
- No study found that reports how many seconds before a block HR, HRV, EDA or respiration changes, as onset-locked averages around individual stuttering moments. Utterance-level designs (about 8-s utterances) dominate.
- No ML study found that predicts stuttering events from ECG/HRV or EDA. The only physiology-to-event classifier is concurrent and respiration-based (Villegas 2019), with no AUC reported.
- Data from these physiology studies (Walsh/Smith, Conture/Walden, Tumanova/Salekin, Villegas/Elias) appear not to be publicly released. I did not find any data-availability links.
- I did not retrieve full texts for older respiratory studies of blocks, such as prephonatory posturing before blocked vs fluent onsets. A targeted read of that literature is still needed.

## C. Is there a combined audio + video + physiology dataset of people who stutter, and which datasets can StutterClear actually use?

### Takeaway
Verdict: **No public combined audio+video+physiology stuttering dataset was found (OPEN).**
- The closest recordings are lab-internal and available only on request: UTSA had audio, face video at 58 fps, 64-channel EEG and eye-tracking for 7–12 adults who stutter in a cued nonword task. Walsh & Usler had physiology with video used for artifact checking.
- For Phase 3, the usable AV data is FluencyBank video. It needs TalkBank consortium membership; segment labels come from Mohapatra 2024 and could come from Valente 2025 if released. Altinkaya's YouTube dataset appears defunct.
- For Phase 4, StutterClear would have to collect its own data.

### Cited Findings
- **UTSA lab recordings:**
  - Face and stimulus video (2× Logitech C920, 58 fps), 64-channel EEG (Neuroscan) and EyeLink 1000 Plus eye-tracking, with SLP-coded fluent vs stuttered trials — [arXiv 2010.01231](https://arxiv.org/abs/2010.01231)
  - The 2022 paper provides no public data; requests go to Najafirad and Golob — [PMC9376608](https://pmc.ncbi.nlm.nih.gov/articles/PMC9376608/)
- **Walsh & Usler 2019:** physiology (SCR, BPV, pulse) was recorded alongside video, which was used to check movement artifacts. No public release was found — [PMC7201324](https://pmc.ncbi.nlm.nih.gov/articles/PMC7201324/)
- **FluencyBank (TalkBank) access:**
  - "Access to the research data in FluencyBank is password protected and restricted to members of the FluencyBank consortium group." Only the Teaching section is open.
  - Students need a faculty advisor, and requests from students alone are not answered.
  - The site shows a CC BY-NC-SA 4.0 badge and says it is "under review for potential modification in compliance with Administration directives".
  - Source: [TalkBank FluencyBank](https://talkbank.org/fluency/)
- **FluencyBank video for adults who stutter:** Mohapatra et al. state FluencyBank has video for 32 individuals and built 4,004 paired AV clips (3.3 h, 469 blocks) — [arXiv 2406.06964](https://arxiv.org/abs/2406.06964). Valente et al. annotated 30 reading and 36 interview audiovisual FluencyBank AWS samples — [arXiv 2506.00644](https://arxiv.org/abs/2506.00644)
- **FluencyBank Timestamped (36 speakers, 6.41 h)** provides word-level timestamps. Kumar et al. add manually reviewed onsets for 2,593 repetitions, 129 prolongations and 150 blocks — [arXiv 2610.19235](https://arxiv.org/abs/2610.19235)
- **Altinkaya & Smeulders AV dataset:** 25 speakers, more than 4 h from YouTube, distributed via www.assistedspeech.org — [ACM PDF](https://dl.acm.org/doi/pdf/10.1145/3423325.3423733). The site gave no response on 17 Oct 2026 (my own check).
- **MMSD-Net** additionally used SpeakingFaces, a fluent-speaker AV corpus, as negatives — [arXiv 2407.11492](https://arxiv.org/abs/2407.11492)
- **Non-stuttering multimodal corpora** exist, for pretraining or for transferring stress features:
  - Inter-Stance (2026): 90 persons with face video, voice, PPG, EDA, HR and respiration — [arXiv 2604.22739](https://arxiv.org/abs/2604.22739)
  - MAD (2026): EEG, ECG, EMG, PPG plus RGB-D face — [arXiv 2603.06206](https://arxiv.org/abs/2603.06206)
  - A German EMA+EEG+audio speech-production dataset (2026) — [Sci Data, PubMed 42393090](https://doi.org/10.1038/s41597-026-07704-3)
  - None of these include people who stutter.

### Inferences
- **Phase 3 data route.** The practical route is:
  1. Obtain FluencyBank membership through a faculty sponsor.
  2. Pull adult-who-stutters interview videos.
  3. Align with FluencyBank Timestamped and the Kumar et al. block onsets, if released, for frame-level block timing.
  4. Use the Mohapatra clip splits for comparability with block balanced accuracy 0.58 (audio) and 0.71 (face).
  5. Use the Valente secondary-behaviour and tension labels, if released, to supervise lip-press and grimace features.
- **The FluencyBank video overlap is an opportunity.** FluencyBank Timestamped and the AV subsets come from the same FluencyBank adult recordings. Onset-aligned block labels could therefore be paired with face video to give the first frame-level AV block benchmark; nobody appears to have published this.
- **A combined dataset would itself be a contribution.** A synchronised audio + webcam + chest-strap ECG/respiration + wrist EDA dataset of people who stutter in natural conversation, with onset-level block labels, appears not to exist. Building one, even with 10–20 speakers, would be a novel contribution, though it needs ethics approval and consent for face video.

### Gaps
- Whether any 2025–2026 clinical group (e.g., Purdue, Vanderbilt, Syracuse, Peru PUCP/Elias lab) has released synchronised physiology plus video for people who stutter. I found none, but did not search institutional repositories exhaustively.
- Whether Mohapatra's released AV subset includes raw video or only features, and under what licence.
- Whether the Kumar et al. 2026 onset annotations are released.
