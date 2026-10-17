# Stuttering Detection SOTA (2023–Oct 2026) and Causal/Streaming Detection: Novelty Check for StutterClear

Scope: best published clip-level and frame-level numbers on SEP-28k, FluencyBank and the frame-level benchmarks, each with its exact evaluation protocol. Also checks whether anyone has measured detection accuracy against lookahead or latency, distilled an SSL teacher into a small causal student, or combined MIL with synthetic frame labels.

StutterClear reference numbers, from its own validation set:
- Validation set: StutterClear's own speaker-group-held-out split of pooled SEP-28k and FluencyBank, with labels by majority vote (2 of 3 annotators).
- Clip AUC: SoundRep 0.915, WordRep 0.961, Prolongation 0.915, Block 0.815, Interjection 0.978.
- Clip F1 at threshold 0.5: 0.63, 0.71, 0.65, 0.25, 0.90 (same class order).

**Comparability legend (used below):**
- **[C]** means the protocol is close to StutterClear's: speaker-exclusive split, at least 2 of 3 annotators, threshold 0.5, clip-level, real data.
- **[~]** means partly comparable.
- **[NC]** means not comparable.

A protocol detail that a paper does not state is marked "not stated" and was not guessed.

---

## Q1. Best clip-level F1/AUC per class on SEP-28k and FluencyBank (2023–2026), with protocols; which papers report AUC

### Takeaway
Best per-class clip-level F1 on the speaker-exclusive SEP-28k-E test set, as of Oct 2026:

| Class | Best F1 | System |
|---|---|---|
| Block | 0.57–0.62 | LLM fusion (Wagner et al. 2024); 0.35 without an LLM (Sušac et al. 2026) |
| Prolongation | 0.56 | Wagner et al. 2024 |
| Sound repetition | 0.53–0.54 | Wagner et al. 2024; Sušac et al. 2026 |
| Word repetition | 0.80 | Sušac et al. 2026, Whisper + max-pool MIL |
| Interjection | 0.82–0.83 | Sušac et al. 2026 |

- I found no newer SEP-28k-E result (July–Oct 2026) that beats these.
- Sušac et al. claim SOTA for blocks at 0.35, but they do not compare against Wagner et al.'s LLM system.
- On FluencyBank, the best per-class F1 values are Block 0.58, Prolongation 0.62, Sound repetition 0.66, Word repetition 0.59 and Interjection 0.84. The best binary "any stutter" F1 is 0.90.
- **No strong fine-tuned SEP-28k-E or FluencyBank system reports per-class AUC.** Per-class ROC-AUC on SEP-28k exists only from a frozen WavLM Base+ linear probe under lenient labels (Pattanayak et al., Sep 2026). Aggregate AUCs exist from Kozak (Apr 2026) and StutterFuse (Dec 2025).

### Cited Findings

**Sušac, Bayerl & Džapo (arXiv 18 Jun 2026): protocol (verified from the paper)**
- Training and testing use the "standardized SEP-28k-E split" (speaker-exclusive, from Bayerl et al. TSD 2022).
- A label is positive "if at least 2 annotators voted for the label".
- Samples without unanimous agreement on "No stuttered words" are down-weighted by 0.25.
- Clip-level threshold θ = 0.5.
- Metric is clip-level multi-label F1.
- No AUC is reported anywhere in the paper.
- Model: a foundation encoder (wav2vec2-large, WavLM-large or Whisper-medium; 307–315M parameters), then an HConv interface, a 4-layer bidirectional LSTM with 512 units, a projector, and either max-pooling (instance MIL) or attention pooling (embedding MIL).
- Input is a 3-s clip giving 150 frames of 20 ms.
- Source: [Sušac et al. 2026](https://arxiv.org/abs/2606.20338)

**Table A: SEP-28k-E test set, clip-level per-class F1** (all rows speaker-exclusive SEP-28k-E)

| System (venue/date) | Model | Bl | Pro | Snd | Wd | Int | Label thr. | Comparability | Source |
|---|---|---|---|---|---|---|---|---|---|
| Bayerl et al. "A Stutter Seldom Comes Alone" (Interspeech 2023) | wav2vec2-large, multi-label, MTL | 0.32 | 0.53 | 0.53 | 0.64 | 0.77 | not stated in citing papers | [~] | [Wagner et al. 2024, Table 1](https://arxiv.org/abs/2406.11025); [Haas & Bayerl 2026, Table 1](https://arxiv.org/abs/2603.26939) |
| Wagner et al. (Interspeech 2024), exp. #13 (best all-round) | Fine-tuned wav2vec2 (315M), last layer, plus Whisper-large 1-best ASR into Llama-2-7B LoRA | 0.57 | 0.56 | 0.54 | 0.64 | 0.74 | not stated | [~] (huge, non-streaming) | [Wagner et al. 2024](https://arxiv.org/abs/2406.11025) |
| Wagner et al. 2024, exp. #1 (best block) | Non-fine-tuned wav2vec2 layer 12 plus 1-best ASR into Llama-2 | 0.62 | 0.51 | 0.45 | 0.31 | 0.62 | not stated | [~] | [Wagner et al. 2024](https://arxiv.org/abs/2406.11025) |
| Miyahara et al. (Interspeech 2025) | VGG-19 on wav2vec2 self-attention weights | 0.30 | 0.53 | 0.46 | 0.67 | 0.78 | not stated | [~] | quoted in [Sušac et al. 2026, Table 1](https://arxiv.org/abs/2606.20338) |
| Haas & Bayerl (TSD; arXiv 27 Mar 2026), TRILANG-LL | wav2vec2-large XLSR-53, trained on EN + DE + ZH (KSoF, SEP-28k-E, FluencyBank, AS-70) | 0.33 | 0.51 | 0.53 | 0.71 | 0.77 | not stated | [~] | [Haas & Bayerl 2026](https://arxiv.org/abs/2603.26939) |
| Sušac et al. 2026, WavLM + max-pool MIL | WavLM-large + BiLSTM | 0.34 | 0.31 | 0.51 | 0.72 | **0.83** | ≥2 of 3 | **[C]** | [Sušac et al. 2026](https://arxiv.org/abs/2606.20338) |
| Sušac et al. 2026, WavLM + attention MIL | WavLM-large + BiLSTM | 0.35 | 0.47 | 0.42 | 0.74 | 0.82 | ≥2 of 3 | **[C]** | same |
| Sušac et al. 2026, Whisper + max-pool MIL | Whisper-medium encoder + BiLSTM | 0.35 | 0.50 | 0.53 | **0.80** | 0.74 | ≥2 of 3 | **[C]** | same |
| Sušac et al. 2026, Whisper + attention MIL | Whisper-medium encoder + BiLSTM | 0.35 | 0.49 | 0.53 | 0.78 | 0.82 | ≥2 of 3 | **[C]** | same |

- Sušac et al. state that their models "achieve SOTA results in the clip-level detection of blocks, sound repetitions, word repetitions and interjections" but "fall slightly short of the baselines for prolongations". Their comparison set is only Miyahara 2025 and Haas & Bayerl; it does not include Wagner et al. 2024's LLM results. — [Sušac et al. 2026](https://arxiv.org/abs/2606.20338)
- Wagner et al. report block improvements over the baseline of "∼94%" relative on SEP-28k-E and "∼61%" on FluencyBank, but "no improvements on word repetitions". They hypothesize this is because Whisper omits repeated tokens. — [Wagner et al. 2024](https://arxiv.org/abs/2406.11025)

**Table B: FluencyBank clip-level (SEP-28k-style labels, 4,144 clips)**

| System | Bl | Pro | Snd | Wd | Int | Protocol notes | Comparability | Source |
|---|---|---|---|---|---|---|---|---|
| Bayerl et al. 2023 | 0.36 | 0.62 | 0.64 | 0.52 | 0.79 | FluencyBank partition from Bayerl et al. Interspeech 2022 (speaker-exclusivity not verified here) | [~] | [Haas & Bayerl 2026](https://arxiv.org/abs/2603.26939) |
| Wagner et al. 2024, #13 | 0.57 | 0.55 | 0.66 | 0.43 | 0.81 | Same partition | [~] | [Wagner et al. 2024](https://arxiv.org/abs/2406.11025) |
| Wagner et al. 2024, #16 (MBR) | **0.58** | 0.48 | 0.63 | 0.44 | 0.78 | Same partition | [~] | same |
| Haas & Bayerl 2026, TRILANG-LL | 0.32 | 0.61 | **0.66** | **0.59** | **0.84** | Same partition | [~] | [Haas & Bayerl 2026](https://arxiv.org/abs/2603.26939) |
| StutterFuse (arXiv Dec 2025), zero-shot from SEP-28k | 0.51 | 0.42 | 0.54 | 0.49 | 0.70 | Threshold 0.3; trained after removing NoStutter-only clips | [NC] | [Singh & Fahad 2025](https://arxiv.org/abs/2512.13632) |

- **Binary "any stutter" F1 on FluencyBank:**
  - Sušac Whisper + attention MIL: 0.90 (precision 0.91, recall 0.89). WavLM + attention: 0.89.
  - Shih et al. (SLT 2024) baseline: 0.85. — [Sušac et al. 2026, Table 2](https://arxiv.org/abs/2606.20338)
  - Shih et al. report 0.84–0.85 on the FluencyBank subset with a frozen WavLM-L, using a threshold swept from 0 to 1 in 0.05 steps to find the best value. — [Shih et al. 2024](https://arxiv.org/abs/2409.10704)
- Shih et al. also give "type-specific" FluencyBank F1, computed by filtering the test set to clips of one type plus fluent clips, using a binary model with an optimized threshold: Bl 0.63, Int 0.59, Pro 0.63, Snd 0.75, Wd 0.70 (WavLM-L + HConv). **[NC]**: the protocol is easier, because each type is scored only against fluent clips. — [Shih et al. 2024](https://arxiv.org/abs/2409.10704)

**Table C: papers that report AUC on SEP-28k or FluencyBank (2023–2026)**

| Paper (date) | What AUC | Numbers | Protocol | Comparability | Source |
|---|---|---|---|---|---|
| Pattanayak et al. (USC, arXiv 26 Sep 2026) | Per-class ROC-AUC, clean audio | Block 0.724, SoundRep 0.844, WordRep 0.851, Prolongation 0.789, Interjection 0.866 | Frozen WavLM Base+ masked-mean embedding with one-vs-rest logistic regression. Labels lenient (**≥1 of 3**). Fixed 8,000-clip SEP-28k subset (81.2% of clips carry a label). 5-fold CV grouped by **episode** (not speaker). F1 at 0.5: Bl 0.638, Snd 0.659, Wd 0.645, Pro 0.624, Int 0.756. | [NC] | [Pattanayak et al. 2026](https://arxiv.org/abs/2609.32285) |
| Pattanayak et al. 2026 (label-threshold check) | F1 only | With majority-vote evaluation labels (≥2), block F1 drops from 0.637 to 0.346; the model was not retrained | Same | — | same |
| Kozak (arXiv 30 Apr 2026) | Aggregate event-detection ROC-AUC; "event" = Block ∪ SoundRep ∪ Prolongation | 616K CNN 0.649 [0.614, 0.686]; wav2vec2-base (94M) linear probe 0.709 [0.673, 0.744] | 20,131 SEP-28k clips; episode-grouped 70/15/15 split (258 show-episode groups); label threshold not stated | [NC] | [Kozak 2026](https://arxiv.org/abs/2604.27279) |
| StutterFuse (arXiv 15 Dec 2025) | Aggregate AUC-ROC | 0.767 | Speaker-independent SEP-28k split, but **NoStutter-only clips removed**; frozen wav2vec2 features; retrieval-augmented Conformer (about 21.5M parameters); per-class F1 at threshold 0.3: Pro 0.61, Bl 0.66, Snd 0.54, Wd 0.55, Int 0.77 | [NC] (no fluent negatives) | [Singh & Fahad 2025](https://arxiv.org/abs/2512.13632) |
| Kumar, Daliri & Gupta (arXiv 13 Oct 2026) | AUROC for *forecasting* (not detection) on FluencyBank Timestamped | 0.630 at 300 ms horizon (see Q3) | 6 speaker-disjoint folds, 36 speakers | [NC] (different task) | [Kumar et al. 2026](https://arxiv.org/abs/2610.19235) |
| Sušac 2026; Wagner 2024; Haas & Bayerl 2026; Bayerl 2023; Miyahara 2025 | — | **No AUC reported** (F1 only) | — | — | [Sušac et al. 2026](https://arxiv.org/abs/2606.20338); [Wagner et al. 2024](https://arxiv.org/abs/2406.11025); [Haas & Bayerl 2026](https://arxiv.org/abs/2603.26939) |

**Other clip-level notes**
- Haas & Bayerl train on clips up to 7 s (AS-70 "length-limited"). They report that adding Mandarin data mainly raises word-repetition F1, which they explain as a "long-context repetition" meta-pattern. — [Haas & Bayerl 2026](https://arxiv.org/abs/2603.26939)
- Mandarin AS-70 (not English, [NC]): an LLM-driven joint ASR + stuttering-event-detection model reports average SED F1 73.63%. — [arXiv 2505.22005](https://arxiv.org/abs/2505.22005)

### Inferences
- **StutterClear compared with the [C] reference (Sušac et al. 2026), clip F1 at threshold 0.5:**

  | Class | Sušac best | StutterClear |
  |---|---|---|
  | Sound repetition | 0.53 | 0.63 |
  | Prolongation | 0.50 | 0.65 |
  | Interjection | 0.83 | 0.90 |
  | Word repetition | 0.80 | 0.71 |
  | Block | 0.35 | 0.25 |

  StutterClear is higher on sound repetition, prolongation and interjection, and lower on word repetition and block.
- **These numbers are not directly comparable.** StutterClear's validation set:
  - is its own split, not the SEP-28k-E test partition;
  - pools in FluencyBank, where most systems score higher than on SEP-28k, especially for sound repetition and interjection (Tables A vs B);
  - has unknown speaker grouping, since SEP-28k has no official speaker IDs and SEP-28k-E's speaker labels are semi-automatic.
- To claim a SOTA comparison, StutterClear should also report on:
  - the official SEP-28k-E test split;
  - the Bayerl et al. FluencyBank partition, separately;
  - at least 2 of 3 annotators and θ = 0.5.
- **AUC is an open reporting niche.** No strong fine-tuned SEP-28k-E system reports per-class AUC, so StutterClear's per-class AUC (0.815–0.978) has no direct comparator.
  - The only per-class AUC on SEP-28k is a frozen linear probe with lenient labels (Block 0.724 … Interjection 0.866). StutterClear's numbers are far above it, but the protocols differ.
  - StutterClear should state that it is not comparable to Pattanayak et al.
- StutterClear's **block F1 of 0.25 alongside block AUC of 0.815** suggests a threshold or calibration problem rather than poor ranking. Pattanayak et al. show that tuning the threshold alone recovers most of the block F1 lost under distribution shift (0.465 to 0.630).
  - Inference: StutterClear should report F1 at a validation-tuned per-class threshold as well as at 0.5.
- Wagner et al.'s block F1 of 0.57–0.62 relies on Whisper-large ASR plus Llama-2-7B, so it is not a realistic comparator for a 427K causal student. It should still be cited, because it is the published block ceiling on SEP-28k-E.

### Gaps
- I could not confirm the label threshold used by Bayerl 2023, Wagner 2024, Miyahara 2025 or Haas & Bayerl, beyond Sušac's explicit "≥2 of 3". The citing papers do not restate it.
- Miyahara et al. (Interspeech 2025) was not read directly; its numbers come via Sušac's Table 1.
- I did not verify whether the FluencyBank partition from Bayerl et al. (Interspeech 2022) is speaker-exclusive.
- I found no Interspeech 2026 or ICASSP 2026 paper beyond Sušac et al. that reports new SEP-28k-E per-class numbers. A web search for "Interspeech 2026 stuttering SEP-28k" returned nothing newer.
- A 2026 systematic review and meta-analysis exists in J. Fluency Disorders ([doi 10.1016/j.jfludis.2026.106222](https://doi.org/10.1016/j.jfludis.2026.106222)). I did not read it; it may contain pooled metrics.
- Papers with Code-style leaderboards for SEP-28k were not found to be maintained. There is no community leaderboard to cite.

---

## Q2. Best frame-level / time-localized results (CASA gold, FluencyBank++, StutterCut, attention-MIL, YOLO-Stutter, SSDM 2.0, LLM-based)

### Takeaway
- **CASA gold standard (FluencyBank, 8 recordings, 732 dysfluencies):** the best result is 0.70 frame-level F1 for binary "any stutter" (single-label), from Whisper + attention-MIL trained only on SEP-28k clip labels (Sušac et al., Jun 2026). Comparators: YOLO-Stutter 0.47, StutterCut 0.45.
- **FluencyBank++:** the best is StutterCut at time-F1 69.3 (IoU > 0.5), with per-type results.
- No multi-label (per-type) frame-level result exists on CASA.
- No causal or streaming system has been evaluated on any frame-level benchmark.
- Text-dependent systems (YOLO-Stutter, SSDM 2.0) report high time-F1 only on small, manually timed subsets, and need a reference transcript.

### Cited Findings

**Table D: frame-level / time-localized results**

| Benchmark | System (date) | Metric | Result | Protocol notes | Source |
|---|---|---|---|---|---|
| CASA gold (FluencyBank, 8 recordings, 732 events) | Whisper-medium + attention MIL (Sušac, Jun 2026) | Frame F1 / P / R, single-label (any stutter), 20-ms frames, averaged per recording | **0.70** / 0.71 / 0.69 | Trained only on SEP-28k-E clip labels (zero-shot to frames). Recordings cut into non-overlapping 3-s clips. Threshold 0.5 on sigmoid of unnormalized attention weights; frames are negative if the clip is predicted negative. | [Sušac et al. 2026](https://arxiv.org/abs/2606.20338) |
| same | Whisper + max-pool MIL | same | 0.66 / 0.76 / 0.59 | same | same |
| same | WavLM-large + attention MIL | same | 0.56 / 0.53 / 0.59 | same | same |
| same | WavLM-large + max-pool MIL | same | 0.46 / 0.53 / 0.41 | same | same |
| same | YOLO-Stutter (WhisperX transcripts) | same | 0.47 / 0.47 / 0.49 | The authors note the baselines "assume every input clip contains a dysfluency", which causes false positives on whole recordings | same |
| same | StutterCut (Sušac's own reimplementation) | same | 0.45 / 0.39 / 0.58 | same caveat | same |
| FluencyBank++ (3,017 clips of 5 s; frame-level boundaries for 4 types; Interspeech 2025) | StutterCut (Ghosh et al., Interspeech 2025) | Time-F1 at IoU > 0.5 | Overall 69.3: Prolongation 73.2, Repetition 68.8, Interjection 64.5, Block 70.6 | Whisper embeddings over 0.75-s windows with 0.1-s stride, N-cut over the whole utterance, 100 MC-dropout passes; offline. YOLO-Stutter scored 56.8 on the same set. | [Ghosh et al. 2025](https://arxiv.org/abs/2508.02255) (numbers via the prior track notes) |
| Word-level FluencyBank (36 AWS speakers, 45.69 min; SLP CHAT annotations aligned with Whisper) | Shih et al. (SLT 2024), WavLM-L + HConv + CTC | Word F1 (FluencyBank partition) / overall average precision | 0.591 F1 (P 0.508, R 0.708); AP 0.927 on the full word-level set | Binary; threshold swept for the best F1; only DTW-aligned words scored | [Shih et al. 2024](https://arxiv.org/abs/2409.10704) |
| SEP-28k subset with manual timings | YOLO-Stutter (Interspeech 2024) | Time-F1 (true positive on any overlap) | 0.813 | **Requires reference text**; lenient overlap criterion | [Zhou et al. 2024](https://arxiv.org/abs/2408.15297) |
| same | SSDM 2.0 (Nov 2024) | Time-F1, zero-shot | 0.830 | WavLM-large + LLaMA LoRA + reference text | [Lian et al. 2024](https://arxiv.org/abs/2412.00265) (via prior notes) |
| FluencyBank Timestamped (36 speakers) | Kumar et al. (Oct 2026), causal Conformer | Onset **forecasting** AUROC, 300 ms ahead | 0.630 | Forecasting task, not detection; see Q3 | [Kumar et al. 2026](https://arxiv.org/abs/2610.19235) |

- Sušac et al. say their models "struggle with long-lasting blocks", which they attribute to the 3-s context window. They suggest "pre-training or fine-tuning on a frame-level dataset might be useful, as demonstrated by some previous studies [Shih; Harvill]". They also say "Multi-label frame-level segmentation performance should also be evaluated in future studies". — [Sušac et al. 2026](https://arxiv.org/abs/2606.20338)
- CASA core labels are syllable repetition, incomplete syllable repetition, multisyllable unit repetition, sound prolongation and block. These do not map one-to-one to SEP-28k classes, so only single-label (any-stutter) evaluation has been done. — [Sušac et al. 2026](https://arxiv.org/abs/2606.20338); [Valente et al. 2025](https://arxiv.org/abs/2506.00644)

### Inferences
- For StutterClear's 20-ms frame outputs, the most directly comparable public frame benchmark is **CASA gold, binary any-stutter frame F1, averaged per recording**. The target to beat is 0.70, set by a non-causal, 307M-parameter model with 3-s context.
  - Reporting both the teacher and each lookahead head of the student on CASA would be a first: no causal or streaming result exists on CASA or FluencyBank++.
- Because StutterClear trains with synthetic exact frame labels, it can also report **per-type frame F1**. This has not been done on CASA. FluencyBank++ has per-type time-F1 at IoU > 0.5, but only for StutterCut and YOLO-Stutter.
- Speaker caveat: CASA, FluencyBank++, FluencyBank Timestamped and the SEP-28k-style FluencyBank labels all come from overlapping FluencyBank adults-who-stutter speakers. If StutterClear trains on FluencyBank clip labels, any frame-level evaluation on CASA or FluencyBank++ must exclude those speakers from training.

### Gaps
- I did not re-read StutterCut's or SSDM 2.0's primary papers in this pass; those numbers carry over from the earlier track notes.
- I did not read Romana et al. 2023, "Automatic Disfluency Detection from Untranscribed Speech" ([arXiv 2311.00867](https://arxiv.org/abs/2311.00867)), which does frame-level disfluency detection from acoustics. Its numbers are not included.
- I found no multi-label (per-type) frame-level result on CASA gold.

---

## Q3. Causal/streaming stuttering detection: accuracy vs lookahead/latency; deployed streaming detectors; distillation from an SSL teacher to a tiny causal student

### Takeaway
**Verdict: OPEN for stuttering. The lookahead-vs-accuracy curve for stuttering *detection* has not been measured.** I found no paper that trains or evaluates a stuttering or dysfluency detector at several lookahead values (for example 0–800 ms) and reports how accuracy changes.

The nearest work falls into three groups:
1. **Forecasting in the opposite direction (negative lookahead)** (Kumar, Daliri & Gupta, Oct 2026). A causal model predicts stuttering onset 0.3–1.5 s *ahead* using only past audio. It reports AUROC against horizon and against history length, but excludes frames inside events, so it does not measure detection.
2. **Ablations that probe the time axis without measuring detection lookahead.** Kozak (Apr 2026) masks the trailing 0–1024 ms of a 3-s clip for a 616K on-device CNN. Earlier systems used a fixed context and never swept it (Amazon Detect-and-Pass, 2022).
3. **Adjacent task, conversational disfluency on Switchboard.** Here accuracy-vs-lookahead *has* been measured, in tokens for text (Chen et al., NAACL 2022) and against ms latency for a joint streaming ASR model (Futami et al., ICASSP 2023). This is the closest methodological precedent, but it covers reparanda and fillers, not stuttering.

**Distillation verdict: OPEN.** I found no knowledge distillation from a large SSL teacher into a tiny causal stuttering student. The only stuttering KD attempt is Kozak's "Future-Guided Learning", which gave a negative result: a same-size CNN teacher that sees the future clip, distilled into a forecasting student. StutterFuse lists distillation to a light student and streaming only as future work.

### Cited Findings

**Forecasting along the time axis (Kumar, Daliri & Gupta, ASU; arXiv 13 Oct 2026)**
- Task: "continuous causal time-to-event forecasting". Every 100 ms, the model estimates the probability that a new stuttering onset begins within horizon Δ, "using only speech observed at or before t".
- Frames inside an active stuttering event are **excluded**, "because the event has already begun and the task would become detection rather than forecasting". **Detection at lookahead ≥ 0 is therefore not evaluated.**
- Model: frozen Whisper-large-v3 encoder over each 2-s window ending at t, giving one state per 100 ms, followed by a causal Conformer (4 blocks, 192-d, 3.67M trainable parameters, causal attention up to 200 states).
- Data: FluencyBank Timestamped plus manual annotations; 36 speakers; 2,872 events; 6 speaker-disjoint folds; 3 seeds.
- Results at the 300-ms horizon:
  - Causal Conformer: AUROC 0.630, AP 0.0769 (1.56× the 4.93% prevalence), within-speaker AUROC 0.573.
  - Baselines: TCN 0.589, GRU 0.559, static Whisper 0.568.
- Per type at 300 ms (Conformer): repetition 0.630, prolongation 0.655, block 0.612. Within-speaker AUROC for prolongations and blocks is 0.540 and 0.541, with intervals overlapping chance.
- Horizon curve (censoring-aware evaluation set, hazard model, AUROC / AP):

  | Horizon | AUROC | AP |
  |---|---|---|
  | 0.30 s | .604 | .074 |
  | 0.50 s | .583 | .114 |
  | 0.75 s | .577 | .177 |
  | 1.00 s | .576 | .218 |
  | 1.50 s | .575 | .310 |

  Independent binary models: .598, .582, .593, .572, .588.
- History-length curve: 2 s of history gives AUROC 0.592, 5 s gives 0.621, 10 s 0.620 and 20 s 0.622 (saturates near 5 s).
- Widening the static window from 2.0 to 3.8 s changes AUROC from 0.5682 to 0.5666, i.e. no gain.
- Operating points:
  - At the max-F1 threshold: 43.9% of events anticipated, mean lead time 196 ms, 27.6 false warning bouts per minute.
  - At 25% alarm duty cycle: 58.6% of events covered.
- No on-device latency is reported.
- In related work, the authors list only Kozak as an audio-based predictor. They cite no streaming stuttering *detector*.
- Source: [Kumar et al. 2026](https://arxiv.org/abs/2610.19235)

**Tail-masking ablation and on-device deployment (Kozak, arXiv 30 Apr 2026)**
- Model: a 616K-parameter 4-block CNN on 3-s log-mel clips (hop 512 at 16 kHz, 32 ms per frame), with two heads: event detection on the current clip and "preblock" prediction of the next clip.
- Ablation: zero the last N frames at inference, without retraining. Event-detection AUC by masked tail length:

  | Masked tail | 0 ms | 128 ms | 256 ms | 512 ms | 1024 ms |
  |---|---|---|---|---|---|
  | Event-detection AUC | 0.649 | 0.649 | 0.650 | 0.651 | 0.635 |

  Preblock AUC drops for blocks by −0.021 and for sound repetitions by −0.043 at 1024 ms.
  - This is the closest published "less future context" ablation for a stuttering detector. It is still clip-level, inference-only masking, not a trained lookahead.
- Deployment:
  - Neural Engine latency per 3-s window: 0.25 ms (A19 Pro), 0.29 ms (A18), 0.55 ms (A15 / M1 Max).
  - A 4-Hz streaming simulation (3-s rolling window, 0.25-s hop) costs 1.34 ms per window, 0.54% of the real-time budget.
  - Exports: CoreML 1.19 MB, ONNX 40 KB.
- Kozak calls this, to his knowledge, the first audio-only on-device prediction system: "Audio-only on-device prediction has, to our knowledge, no published working system".
- Source: [Kozak 2026](https://arxiv.org/abs/2604.27279)
- **Distillation attempt (negative):** "Future-Guided Learning", adapted from EEG seizure prediction.
  - A teacher event head sees the future clip C_{i+1}; a student preblock head sees C_i. Binary soft KD with T = 3 and α = 0.5.
  - It "degraded validation AUC from 0.546 (warm-start) to 0.539". The KD loss "stabilized at log 2", because the teacher's soft outputs on the future clip were "noisier versions of the hard label".
  - This is not SSL-to-small-causal distillation: teacher and student are the same CNN. — [Kozak 2026](https://arxiv.org/abs/2604.27279)

**Fixed-context frame-level detector, no sweep (Amazon Detect-and-Pass, 2022)**
- A frame-level binary stutter classifier on 100-ms frames with ±4 frames of context (about 400 ms of implied look-ahead), using a small CNN plus fully connected layers. It was used to drop stutter frames before an RNN-T.
- No lookahead sweep is reported, per the earlier track's reading. — [Shonibare et al. 2022](https://arxiv.org/abs/2202.05396) (via prior track notes; not re-read in this pass)

**Other "real-time" claims without a latency or lookahead analysis**
- Jouaiti & Dautenhahn (ICASSP 2022), "Dysfluency Classification in Stuttered Speech Using Deep Learning for Real-Time Applications". It is cited as a real-time-oriented method, but no latency or lookahead curve was found. — cited in [Sušac et al. 2026 refs](https://arxiv.org/abs/2606.20338); [Wagner et al. 2024 refs](https://arxiv.org/abs/2406.11025)
- StutterFuse lists "exploring model distillation to create lighter, non-retrieval student models" and "adapting the pipeline for low-latency streaming applications" as future work. — [Singh & Fahad 2025](https://arxiv.org/abs/2512.13632)

**Adjacent precedent: conversational disfluency detection with lookahead (Switchboard, not stuttering)**
- **Chen, Zayats, Walker & Padfield (Google; NAACL 2022), "Teaching BERT to Wait":** text-only streaming disfluency tagging over reparanda and interregna on Switchboard, using a small 3.1M-parameter BERT_SV.
  - Streaming F1 at fixed lookahead: 0 tokens 0.76, 1 token 0.83, 2 tokens 0.85.
  - A learned dynamic "wait" head gets 0.83 with an average wait of 0.21 tokens. Edit overhead drops from 0.31 to 0.09.
  - The paper states "the size of the lookahead window scales directly with the accuracy and stability and inversely with the latency".
  - Source: [Chen et al. 2022](https://arxiv.org/abs/2205.00620)
- **Futami et al. (Sony/CMU; ICASSP 2023), "Streaming Joint Speech Recognition and Disfluency Detection":** audio input, blockwise streaming Transformer (1,600-ms blocks, 640-ms shift).
  - Pipeline ASR + BERT: aligned F1 0.69 at lookahead 0 (token-latency P50/P90 536/1045 ms); 0.81 at lookahead 2 tokens (641/1199 ms).
  - Joint multi-task model: 0.84 at 529/995 ms, with 58.4M parameters.
  - Results hold on CSJ (Japanese) as well.
  - Source: [Futami et al. 2023](https://arxiv.org/abs/2211.08726)
- Rohanian & Hough (IJCNLP 2021) made a BERT-large disfluency tagger incremental with fixed lookahead and GPT-2 "prophecy" decoding. — cited in [Chen et al. 2022](https://arxiv.org/abs/2205.00620)

**Adjacent distillation work (not stuttering)**
- Apple: "Compressing Streaming Neural Audio Encoders via Latent-Space Distillation" (arXiv Sep 2026), for on-device Dictation encoders. — [arXiv 2609.04102](https://arxiv.org/abs/2609.04102) (abstract only)
- Apple/Meta: "Adaptive Knowledge Distillation for Device-Directed Speech Detection" (arXiv Aug 2025). — [arXiv 2508.02801](https://arxiv.org/abs/2508.02801) (abstract only)
- "Multi-Teacher Distillation for Cross-Domain Streaming Electrolaryngeal Speech Encoding" (arXiv Sep 2026): SSL distillation into a streaming encoder for pathological speech, but not stuttering. — [arXiv 2609.18686](https://arxiv.org/abs/2609.18686) (abstract only)
- Searches for "knowledge distillation stuttering detection" (web and alphaXiv) returned no stuttering-specific distillation paper. This is a negative search result, so there is nothing to cite. The closest audio hit was speech-emotion KD ([arXiv 2312.13556](https://arxiv.org/abs/2312.13556)), which is not relevant to stuttering.

**Apple on-device stuttering detection**
- I found no Apple paper describing a streaming or on-device stuttering *detector*. Apple's published stuttering work is the SEP-28k dataset (ICASSP 2021) and the CHI 2023 ASR / endpointer tuning study.
- "On-device" stuttering evidence in this search comes from Kozak (an independent author benchmarking on Apple Silicon), not from Apple. — [Kozak 2026](https://arxiv.org/abs/2604.27279); [Lea et al. 2021](https://arxiv.org/abs/2102.12394)

**Table E: verdict summary for the causal/streaming sub-questions**

| Sub-question | Verdict | Closest prior work |
|---|---|---|
| Stuttering detection accuracy as a function of lookahead (e.g. 0–800 ms), frame-level | **OPEN** | Kozak 2026 inference-time tail masking (clip-level); Futami 2023 / Chen 2022 for conversational disfluency (Switchboard) |
| Stuttering prediction as a function of horizon (negative lookahead) | **DONE** (Kumar et al., Oct 2026: 0.3–1.5 s; history 2–20 s) | — |
| Deployed or streaming small stuttering model with measured on-device latency | **PARTIALLY DONE** | Kozak 2026: 616K CNN, 3-s rolling window at 4 Hz. It is not frame-level, not causal within the window, and is evaluated mainly on prediction. |
| Causal frame-level stuttering detector evaluated on a frame benchmark (CASA, FluencyBank++) | **OPEN** | — |
| KD from a large SSL teacher to a tiny causal stuttering student | **OPEN** | Kozak's negative "Future-Guided Learning" (same-size CNN); StutterFuse lists distillation as future work |
| Multi-head student with one head per lookahead, trained jointly | **OPEN** for stuttering | Lookahead-configurable models exist in streaming ASR and speech enhancement (e.g. [ANCAT 2023](https://arxiv.org/abs/2305.04159); [LaCo-SENet 2026](https://arxiv.org/abs/2606.19688)) but not for stuttering |

### Inferences
- StutterClear's six-head student (0, 100, 200, 300, 500, 800 ms) would give the **first published accuracy-vs-lookahead curve for stuttering detection**. It also complements Kumar et al.'s forecasting curve: together they cover prediction offsets from −1.5 s (forecasting) to +0.8 s (detection with lookahead). Nobody has presented that continuum.
  - Kumar et al.'s 300-ms forecasting AUROC of about 0.60–0.63 gives a natural lower anchor, i.e. what is achievable with −300 ms "lookahead".
- Adjacent evidence suggests where the curve should bend:
  - On Switchboard, most of the gain arrives with the first unit of lookahead (text F1 0.76 → 0.83 at 1 token; about 0.85 at 2 tokens).
  - Kozak's masking shows clip-level detection AUC barely moves until more than 512 ms is removed.
  - Inference: expect the steepest gains for repetitions, since a repetition is only confirmable once the second iteration is heard, and for long blocks. Expect small gains for interjections and prolongations.
  - StutterClear's per-class curves would test this directly.
- To be defensible against Kumar and Kozak, StutterClear should report:
  - within-speaker AUC as well as pooled AUC (Kumar shows they diverge for blocks and prolongations);
  - per-class AUC with bootstrap CIs grouped by speaker or episode, as both papers do;
  - on-device latency for the 427K student, which would put it alongside Kozak's 616K benchmark.
- On distillation: Kozak's failure was specific. The teacher's soft labels carried no information beyond the hard target. StutterClear's setup is different: its teacher is a much stronger WavLM-Large model with frame-level outputs, and its student is a small causal model. Kozak's negative result is therefore not evidence against StutterClear's design, but StutterClear should cite it and show the KD gain with an ablation (student trained without the teacher).

### Gaps
- I did not find or read the full text of Jouaiti & Dautenhahn ICASSP 2022, so I cannot confirm whether it reports any latency figure.
- I did not re-read Amazon Detect-and-Pass to check whether its context size was ablated; the earlier track found no sweep.
- Searches covered alphaXiv (two discovery passes), Firecrawl research search and web search. A non-arXiv venue paper (e.g. a journal or workshop paper not on arXiv) could still exist. I found none, so "OPEN" means no evidence was found, not proof of absence.

---

## Q4. Has anyone combined MIL / weak clip labels with synthetic frame labels?

### Takeaway
**Verdict: ALREADY DONE in a sequential, binary form; PARTIALLY DONE relative to StutterClear's design.**
- Harvill et al. (Interspeech 2022) and Shih et al. (SLT 2024; WavLM-Large) both:
  1. pretrain a frame-level stutter detector on LibriSpeech with synthetic stutters and exact frame labels;
  2. fine-tune on SEP-28k clip labels with max-pooling MIL.
- Both are **binary** (stutter vs not), not multi-label. Both train **in two stages** rather than jointly. Both use max pooling, not log-sum-exp.
- Sušac et al. (2026) do multi-label MIL (max and attention pooling) but explicitly use **no** synthetic or frame-level pretraining.

StutterClear's combination is therefore an incremental variant, not a new paradigm:
- 5-class multi-label;
- joint training on synthetic frame labels plus SEP-28k/FluencyBank clip labels;
- log-sum-exp pooling;
- WavLM-Large + BiGRU.

### Cited Findings
- **Shih et al. (UT Austin; SLT 2024):**
  - "Following [Harvill et al.], we first pretrain on LibriSpeech with synthetic disfluency augmentations and then finetune our model on SEP-28K dataset."
  - Synthetic augmentations (prolongation, word repetition, sound repetition) are inserted between WavLM's convolutional and transformer layers, with a frame-level cross-entropy loss plus an auxiliary phoneme CTC loss.
  - Fine-tuning uses "the timestep with the largest positive class prediction" against the clip label (max-pool MIL).
  - WavLM-Large is frozen; HConv and the head are trained.
  - They follow Harvill's balanced SEP-28k split, which is not SEP-28k-E.
  - Results: SEP-28k test F1 0.803 (binary); FluencyBank subset 0.84; word-level FluencyBank F1 0.591.
  - Without fine-tuning on SEP-28k (synthetic only, "Data Ratio 0%"), F1 is 0.685 on SEP-28k and 0.469 word-level overall. This shows real clip labels add a lot on top of synthetic pretraining.
  - Source: [Shih et al. 2024](https://arxiv.org/abs/2409.10704)
- **Harvill, Hasegawa-Johnson & Yoo (Interspeech 2022), "Frame-Level Stutter Detection":** "first pretrained on LibriSpeech with synthetic disfluencies augmented during training with frame-level supervision. Then, they added a max pooling layer to finetune on utterance-level SEP-28K dataset". It reaches SEP-28k F1 0.700 and word-level F1 0.411. — described in [Shih et al. 2024](https://arxiv.org/abs/2409.10704)
- **Sušac et al. 2026:**
  - "Harvill et al. and Shih et al. perform binary frame-level stuttering classification by pre-training a classifier on synthetic dysfluencies and then fine-tuning on a clip-level dataset using the multiple-instance learning (MIL) paradigm."
  - Sušac et al. claim novelty for "generalizing the existing MIL-based methods to the multi-label classification setting". Unlike prior work, "our method does not require pre-training on frame-based datasets".
  - They mention "max, mean or the log-sum-exp functions" as standard MIL pooling options, but use only max and attention.
  - Source: [Sušac et al. 2026](https://arxiv.org/abs/2606.20338)
- **StutterCut (Interspeech 2025):** weak (utterance-level) labels train a "pseudo-oracle" Whisper classifier that guides N-cut segmentation. This is weak supervision for segmentation, but without synthetic frame labels in the main pipeline. — [Ghosh et al. 2025](https://arxiv.org/abs/2508.02255)
- **YOLO-Stutter, SSDM and SSDM 2.0** train on large synthetic time-aligned corpora (VCTK-Stutter/TTS, Libri-Dys) and need reference text. They do not use MIL on real clip labels. — [Zhou et al. 2024](https://arxiv.org/abs/2408.15297); [Lian et al. 2024](https://arxiv.org/abs/2412.00265)
- **Haas & Bayerl (2026)** list "incorporating synthetic data" as future work, and say "the usefulness of synthetic data is often overstated". — [Haas & Bayerl 2026](https://arxiv.org/abs/2603.26939)

### Inferences
- StutterClear should **cite Harvill 2022 and Shih 2024 as direct precursors** and should not claim the synthetic-plus-MIL combination as new.
- Defensible novelty claims are:
  1. multi-label (5-class) frame outputs with combined synthetic and MIL supervision;
  2. joint rather than sequential training, if ablated against pretrain-then-finetune;
  3. log-sum-exp pooling compared against max and attention pooling;
  4. using that teacher for lookahead-indexed distillation into a causal student (Q3).
- The strongest test for StutterClear is to beat Sušac's single-label CASA frame F1 of 0.70 and report per-type CASA frame F1 (Q2). Sušac et al. explicitly say frame-level pretraining "might be useful", so StutterClear's teacher is a natural test of that conjecture.

### Gaps
- I did not read Harvill et al. 2022 directly. Its numbers and method description come from Shih et al.'s summary and baseline table.
- I did not find whether any 2025–2026 paper jointly optimizes frame-level synthetic and clip-level MIL losses in one stage for stuttering. None turned up in searches, but jointly versus sequentially is a detail abstracts often omit.
