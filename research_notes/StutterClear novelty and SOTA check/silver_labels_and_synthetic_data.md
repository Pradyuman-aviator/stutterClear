# Silver labels from verbatim ASR and synthetic stutter data: novelty check (as of 17 Oct 2026)

Scope: (A) whether anyone has used verbatim-ASR word timings (CrisperWhisper, verbatim Whisper variants, CTC models) to make frame-level "silver" labels on **real** stuttered speech and trained stutter detectors on them; (B) which synthetic stutter generators exist and which StutterClear v2 patterns they already cover.

Headline verdicts:
- **(A) Verbatim-ASR silver frame labels for training stutter detectors: PARTIALLY DONE.** Every ingredient has been used somewhere: ASR word timestamps to place human stutter labels (evaluation only, 2024), CrisperWhisper 2.0 transcripts as a *review aid* for human annotators of FluencyBank (13 Oct 2026), ASR transcripts feeding frame-level disfluency detectors (Switchboard, 2023), ASR pseudo-transcripts of SEP-28k as training targets (intended text, 2025), and nyra's "verbatimize" pitched for corpus enrichment (2026). I found **no paper** that takes unverified verbatim-ASR tokens (filler, cut-off, repeated-word copies) on real stuttered speech, turns them into frame labels with per-class loss masks, and trains a stutter detector on them. That specific recipe looks **OPEN**.
- **(B) Synthetic generators: most v2 pattern classes are ALREADY COVERED somewhere; the combination and several specific mechanisms are not.** Sound, word and phrase repetitions, prolongations, blocks, fillers and co-occurring dysfluencies all exist in prior generators (LibriStutter, VCTK-*/Libri-Dys family, LLM-Dys, Stutter-TTS, Mandarin CosyVoice2/MaskGCT). Not found: fillers spliced from the speaker's own schwa and nasal phones, period-aligned looping for prolongations, abandoned-restart phrase repetitions in audio with frame labels, or deliberately rapid 3–5× near-zero-gap repetitions. Partially found: block as onset fragment plus silence, clusters, and non-identical copies.

---

## 1. Has verbatim ASR (CrisperWhisper, Scribe, prompted Whisper, wav2vec2 CTC) been used to auto-annotate stuttering or disfluency data at word or frame level, and are there pseudo-labeling or weak-supervision papers for stuttering detection (2023–2026)? Did CrisperWhisper 2.0 or "verbatimize" touch stuttering data?

### Takeaway
Verbatim ASR has been used on real stuttered speech as an **annotation aid** (human-verified) and for **timestamping human labels** (evaluation sets), but not, as far as I can find, as a **training-label source** for a stutter detector. The CrisperWhisper 2.0 paper evaluates disfluency detection only on acted, non-stuttered DisfluencySpeech and word timing on FluencyBank. It never trains or evaluates a stutter detector, and its model card does not mention stuttering. The closest prior art is the ASU onset-forecasting paper (arXiv 2610.19235, 13 Oct 2026). It ran CrisperWhisper 2.0 on FluencyBank Timestamped, but only to help a human annotator, who verified every event by hand.

### Cited Findings

**CrisperWhisper 2.0 / "Transcription Policy as a Latent Variable" (nyra labs; arXiv 2607.18934, 21 Jul 2026)**
- Authors: Laurin Wagner, Mario Zusag, Bernhard Thallinger (nyra labs, Austria). The model is Whisper-medium with decoder "mode tags" for verbatim vs intended output, supervised cross-attention for word timing, and a "verbatimize" task. — [arXiv 2607.18934](https://arxiv.org/abs/2607.18934)
- The HF model `nyralabs/CrisperWhisper2.0_medium` (created 15 Jul 2026) is tagged with arXiv 2607.18934, so this paper is the "CrisperWhisper 2.0" paper. A `CrisperWhisper2.0_large_pro` repo also exists. — [HF model](https://hf.co/nyralabs/CrisperWhisper2.0_medium); [HF Pro repo](https://hf.co/nyralabs/CrisperWhisper2.0_large_pro)
- **Evaluation data:**
  - Disfluency detection used DisfluencySpeech (English) and a new German DisfluencySpeech set (n=202).
  - Word timing used TIMIT, FluencyBank and Thorsten.
  - Verbatimize used an ICSI rare-word set (n=1,342).
  - Training data were ICSI, AMI, CORAAL, NSC (a 40-h high-quality subset), LibriSpeech and Common Voice. No stuttering corpus was used for training, and no stutter-event metric is reported. — [arXiv 2607.18934](https://arxiv.org/abs/2607.18934)
- **Verbatimize** takes audio plus an intended transcript and reconstructs a canonical verbatim transcript, "inserting disfluency markers ([uh], [um], repetitions, cut-offs, sound tags) at audio-detected positions".
  - It raises rare-word recall from 6.8% to 96.1%.
  - The authors pitch it as "enabling cost efficient verbatim annotations at scale".
  - Their stated future work is "large-scale bootstrapping of consistent verbatim corpora, unlocking high quality weakly supervised data for better verbatim ASR, speech analysis and expressive TTS". Stutter detection is not mentioned. — [arXiv 2607.18934](https://arxiv.org/abs/2607.18934)
- **Labelling rule in the paper's own evaluation protocol (very close to StutterClear's token→label mapping):**
  - An unmatched token with a trailing hyphen is a CUTOFF.
  - A bracketed filler is a FILLER.
  - A consecutive duplicate before a matched token is a REPETITION.
  - Following Lou and Johnson, "The last instance before a non-duplicate is tagged as fluent; all preceding duplicates are tagged as repetitions."
  - The paper uses this to score ASR output, not to train a detector. — [arXiv 2607.18934](https://arxiv.org/abs/2607.18934)
- **Model card:**
  - License: standard models are under the "Nyra Health Non-Commercial Research License", and "any commercial use requires a commercial license". The Pro models are commercial-licence only.
  - Format: example output uses lowercase `[um]`/`[uh]` and hyphenated cut-offs ("th-").
  - Stuttering and FluencyBank are not mentioned.
  - Claimed disfluency F1, averaged over 10 languages: 87.8 (2.0), 93.5 (2.0 Pro), 79.2 (ElevenLabs Scribe v2). These are vendor-reported, and 8 of the 10 languages use synthetic evaluation sets. — [HF model card](https://huggingface.co/nyralabs/CrisperWhisper2.0_medium)
- **CrisperWhisper v1** (arXiv 2408.16589, Aug 2024; Interspeech 2024; Whisper-large-v3 fine-tune) is licensed CC-BY-NC-4.0. — [HF model](https://hf.co/nyralabs/CrisperWhisper); [arXiv 2408.16589](https://arxiv.org/abs/2408.16589)

**Closest prior art: CrisperWhisper 2.0 run on real stuttered speech (ASU, arXiv 2610.19235, 13 Oct 2026)**
- Kumar, Daliri and Gupta augment FluencyBank Timestamped (36 speakers, 6.41 h of recordings):
  - "We first generate verbatim word-level transcripts and timestamps with CrisperWhisper2.0 and align them with the existing FBT annotations. A human annotator then listens to each recording end-to-end … automatic transcripts and existing annotations serve only as review aids, and all added events are manually verified."
  - "CrisperWhisper is not used by the acoustic forecasting model." — [arXiv 2610.19235](https://arxiv.org/abs/2610.19235)
- The resulting set has 2,872 events: 2,593 repetitions, 129 prolongations and 150 blocks. Prolongations and blocks were "annotated entirely by hand". Inter-annotator κ was 0.54 (repetition), 0.73 (prolongation) and 0.56 (block); the median onset discrepancy was 80 ms. The task is onset forecasting (AUROC 0.63), not detection. — [arXiv 2610.19235](https://arxiv.org/abs/2610.19235)

**ASR timestamps used to place human stutter labels (evaluation only)**
- Shih, Gkalitsiou, Dimakis and Harwath (UT Austin; SLT 2024; arXiv 2409.10704, Sep 2024) built a 1.14-h word-level evaluation set (FluencyBank AWS plus bilingual narratives):
  - Whisper Large word timestamps were DTW-aligned to SLP CHAT annotations, and a word was labelled "stutter" if its CHAT code had any stuttering annotation.
  - They added "a little margin (less than 0.2 sec) … due to the fact that Whisper truncates the audio segment where Block manifests".
  - Only exactly aligned words were evaluated.
  - The labels come from humans; ASR supplies only the timing. — [arXiv 2409.10704](https://arxiv.org/abs/2409.10704)

**ASR transcripts as input to frame-level disfluency detection (not stuttering, not pseudo-labels)**
- Romana, Koishida and Mower Provost, "Automatic Disfluency Detection from Untranscribed Speech" (arXiv 2311.00867, Nov 2023), on Switchboard:
  - ASR transcripts (WavLM-FT, Whisper-FT, Azure) were fed to a BERT tagger, and word labels were upsampled to frames.
  - They conclude that "disfluency detection performance is largely limited by the quality of transcripts and alignments", and that an acoustic-only WavLM model beats the ASR→language route for most classes.
  - A multimodal BLSTM fusion was best.
  - ASR transcripts were used only at test time. — [arXiv 2311.00867](https://arxiv.org/abs/2311.00867)

**LLM-based "disfluency-rich transcript" generation**
- Altinok, "Smooth Operators" (arXiv 2506.18510, Jun 2025):
  - LLaMA-3-8B plus a Conformer encoder produces disfluency tokens with timestamps from imperfect hints (wav2vec2-phoneme CTC output, MFA alignments).
  - The author says it can "enrich disfluency datasets".
  - It was trained and evaluated only on synthetic VCTK-TTS, not on real stuttered speech. — [arXiv 2506.18510](https://arxiv.org/abs/2506.18510)

**ASR pseudo-labels on SEP-28k, but for intended text, not events**
- StutterZero/StutterFormer (Xu; Millburn High School; arXiv 2510.18938, Oct 2025):
  - A Whisper-Small model fine-tuned on LibriStutter was "applied to the SEP-28K dataset, generating fluent transcripts for all SEP-28K audio clips". MeloTTS then synthesised fluent targets, about 34 h in total with LibriStutter.
  - These are intended-text pseudo-targets for speech-to-speech conversion, not event or frame labels. — [arXiv 2510.18938](https://arxiv.org/abs/2510.18938)

**Weak-supervision and pseudo-label papers for stuttering (none use ASR-derived labels)**
- **StutterCut** (Ghosh et al., Interspeech 2025; arXiv 2508.02255) is semi-supervised. It uses a "pseudo-oracle classifier trained on weak (utterance-level) labels" to refine a graph normalised cut. It also released FluencyBank++ frame boundaries. — [arXiv 2508.02255](https://arxiv.org/abs/2508.02255)
- **Sušac, Bayerl and Džapo** (arXiv 2606.20338, 18 Jun 2026) use attention-based MIL on SEP-28k-E clip labels to get frame-level segmentation.
  - CASA frame F1: 0.70 for Whisper with attention pooling, against 0.47 for YOLO-Stutter and 0.45 for StutterCut.
  - They state that their "method does not require pre-training on frame-based datasets".
  - No ASR labels are used. WhisperX transcripts were used only to run the YOLO-Stutter baseline. — [arXiv 2606.20338](https://arxiv.org/abs/2606.20338)
- **Kordt et al.** (arXiv 2606.14391, Jun 2026) add disfluency tokens to Whisper-small via continual learning. Their labels are manual TalkBank CHAT markers (SME, Pitt and Delaware corpora), not pseudo-labels or stuttering corpora. — [arXiv 2606.14391](https://arxiv.org/abs/2606.14391)
- A Firecrawl full-text search of the arXiv and biomedical index (2022–2026) for pseudo-labelling or self-training in stuttering detection surfaced StutterCut, Shih et al., ASU 2610.19235 and generic speech pseudo-labelling papers. It found no ASR-derived silver labels used to train a stutter detector. — [StutterCut](https://arxiv.org/abs/2508.02255); [Shih et al.](https://arxiv.org/abs/2409.10704); [ASU](https://arxiv.org/abs/2610.19235)

**Automatic alignment used to locate events in real stuttered speech (from human tags)**
- Lu et al. (Nankai; arXiv 2606.21457 v2, 11 Oct 2026) use FlexAligner, a wav2vec2/CTC aligner tolerant of mismatches, on AS-70. It takes the reliably aligned units before and after each *human-annotated* event as anchors to define approximate stutter regions for masked re-synthesis. The event labels are human; only the regions are automatic. — [arXiv 2606.21457](https://arxiv.org/abs/2606.21457)

**ASR timestamps used for synthesis, not for labelling real speech**
- LibriStutter used the Google Cloud Speech-to-Text API to get word timestamps on fluent LibriSpeech before inserting stutters. — [FluentNet, arXiv 2009.11394](https://arxiv.org/abs/2009.11394)
- VCTK-Stutter used WhisperX word alignments to inject word-level dysfluencies. — [SSDM 2.0 appendix, arXiv 2412.00265](https://arxiv.org/abs/2412.00265)

### Inferences
- **Verdict A: PARTIALLY DONE overall. The specific StutterClear recipe is OPEN as of 17 Oct 2026.** No paper found uses raw verbatim-ASR output on real stuttered speech as training labels for a frame-level stutter detector. That recipe is: map filler→interjection, cut-off→sound repetition and earlier copies→word repetition; convert to frames; mask prolongation and block in the loss.
- A reviewer would cite three papers against a novelty claim:
  - ASU 2610.19235, the same model on the same kind of data, but human-verified and not used for training.
  - Shih 2409.10704, ASR timing plus human labels.
  - nyra 2607.18934, which states the same disfluency surface-pattern rules and pitches verbatimize for weakly supervised data.
- The defensible claim is narrow: "first to train a stutter detector on *unverified* verbatim-ASR silver frame labels from real stuttered speech, with class-masked losses for ASR-invisible classes."
- The ASU paper is indirect evidence for StutterClear's masking choice. CrisperWhisper 2.0 plus the existing labels missed enough that a human had to add sound and word repetitions, and *all* prolongations and blocks had to be added by hand.
- **Licensing risk:** silver labels come from a model under a non-commercial research licence (2.0 standard; v1 is CC-BY-NC-4.0). Whether labels derived from it can train a commercial StutterClear model is a legal question to resolve, separate from the SEP-28k CC-BY-NC question.

### Gaps
- I did not read Alharbi et al. 2018 ("A lightly supervised approach to detect stuttering in children's speech", Interspeech 2018) or Alharbi et al. 2020 (CSL, "Sequence labeling to detect stuttering events in read speech"). They use ASR-based detection on UCLASS read speech and are known only through FluentNet's related work ([arXiv 2009.11394](https://arxiv.org/abs/2009.11394)). They are older ASR-as-detector work, not pseudo-labels for an acoustic detector, but they should be checked before claiming "first ASR-derived labels on UCLASS".
- I did not search ACL Anthology text-domain disfluency self-training (for example, Wang et al. 2020 self-training for disfluency detection). Those use text, not frame labels.
- Paywalled or non-arXiv venues (JSLHR, Journal of Fluency Disorders, IEEE TASLP) and non-English-language venues were not searched exhaustively. Preprints from the last 1–2 weeks may be missing from indexes.

---

## 2. Which synthetic stutter generators exist, and which StutterClear v2 patterns do they already cover (rapid multi-repetitions, phrase repetitions/restarts, fillers in the speaker's own voice, prolongation method, block method, clusters, non-identical copies)? At what scale and label granularity?

### Takeaway
There are at least ten relevant generators. They span waveform splicing (LibriStutter, Harvill/Shih, VCTK++), articulatory editing (VCTK-Art), rule-based TTS (VCTK-TTS, Libri-Dys, Libri-Co-Dys, VCTK-Pro), LLM text plus TTS (LLM-Dys), TTS learned from real stuttering (Stutter-TTS; Mandarin CosyVoice2/MaskGCT), acted recordings (DisfluencySpeech) and text-only generation (LARD). Every basic StutterClear class is covered by at least one. What is not documented anywhere is v2's specific mechanisms:
- fillers built from the target speaker's own schwa and nasal phones
- period-aligned looping for prolongations
- phrase repetitions with abandoned restarts rendered in audio with frame labels
- rapid 3–5× repetitions with a deliberately near-zero gap
- clusters of different events on the same word

### Cited Findings

**LibriStutter (Kourkounakis, Hajavi, Etemad; IEEE/ACM TASLP 2021; arXiv 2009.11394, Sep 2020)**
- Scale and labels: 20 h from LibriSpeech train-clean-100, 50 speakers (23 M / 27 F), 15,000 disfluencies split equally (3,000 each) across sound repetition, word repetition, phrase repetition, prolongation and interjection. Each file has a CSV with each word's start, end and disfluency type. There is one disfluency (or none) per 4-s window.
- **Sound repetition:** copies "the first fraction of a random spoken word" and adds 1–3 repeats before it, with "a random empty pause duration of 100 to 350 ms" after each, smoothed by interpolation.
- **Word repetition:** 1–2 copies of a word. **Phrase repetition:** a 2–3-word phrase repeated, with the same pause and interpolation as above.
- **Prolongation:** "the latter 20% of the signal was stretched by a factor of 5", followed by pitch shifting to restore pitch.
- **Interjections:** "Multiple samples of common filler words from the UCLASS were isolated … to create a pool"; a random filler is inserted with pitch scaling and normalisation. So the fillers come from *other* speakers.
- **Not covered:** revisions, which "would require added speech through the use of complex language models and voice augmentation tools", and blocks. — [arXiv 2009.11394](https://arxiv.org/abs/2009.11394)
- Later work reports that LibriStutter's unnatural dysfluencies "degrade performance" when used for training. — [Ghosh et al. 2025, arXiv 2508.02255](https://arxiv.org/abs/2508.02255)

**Harvill et al. (Interspeech 2022) and Shih et al. (SLT 2024)**
- Both use online synthetic augmentation of LibriSpeech-360 with frame-level targets. The augmentation types are "artificial prolongation, and word/sound repetition".
- Shih inserts the augmentation between WavLM's CNN and transformer layers, not on the waveform.
- Shih notes interjection detection lagged and "could potentially be improved by adding synthesized sound interjections in the pretraining stage". So these generators have no fillers and no blocks. — [Shih et al., arXiv 2409.10704](https://arxiv.org/abs/2409.10704); [Harvill et al. 2022, ISCA](https://www.isca-archive.org/interspeech_2022/harvill22_interspeech.html)

**VCTK++ (Lian et al., ASRU 2023)**
- Repetitions, prolongations and blocks are injected directly in acoustic space using MFA forced alignments.
- Scale: 130.66 h. MOS naturalness: 2.14. — [SSDM appendix, arXiv 2408.16221](https://arxiv.org/abs/2408.16221); [SSDM 2.0 appendix, arXiv 2412.00265](https://arxiv.org/abs/2412.00265)

**VCTK-Stutter and VCTK-TTS (Zhou et al., YOLO-Stutter, Interspeech 2024; arXiv 2408.15297)**
- VCTK-Stutter extends VCTK++ with word-level repetition and deletion, injected acoustically with WhisperX alignments. — [SSDM 2.0 appendix, arXiv 2412.00265](https://arxiv.org/abs/2412.00265)
- VCTK-TTS is a text-space simulation rendered with VITS.
  - By type: repetition 114.92 h, missing 122.05 h, block/pause 68.56 h, replacement 64.33 h, prolongation 57.06 h; total 426.93 h.
  - Annotations are word-level and phoneme-level JSON with timestamps. — [Smooth Operators, arXiv 2506.18510](https://arxiv.org/abs/2506.18510)
- Simulated blocks are 0.5–2.0 s of inserted silence. MOS: VCTK-TTS 4.13, VCTK-Stutter 2.22. — [YOLO-Stutter, arXiv 2408.15297](https://arxiv.org/abs/2408.15297) (via prior StutterClear notes)

**VCTK-Pro, VCTK-Art and AISHELL3-Pro (Stutter-Solver; Zhou et al., SLT 2024; arXiv 2409.09621, Sep 2024)**
- Scale: VCTK-Pro 1,125.1 h (including 337.84 h co-dysfluency); VCTK-Art 392.19 h; AISHELL3-Pro 511.37 h (Mandarin).
- **VCTK-Art** edits articulatory (EMA) features using MFA alignments.
  - Repetition: "The target phoneme segment is duplicated 2–4 times", with 2-unit interpolation buffers and no explicit pause.
  - Block: "A silence frame with 10–15 units is inserted after the target phoneme".
  - Prolongation: the phoneme is interpolated to 4–6× its length.
- **Co-dysfluency (VCTK-Pro):**
  - Single-type: "2–3 instances of the same type".
  - Multi-type: 5 pairs (rep-missing, rep-block, missing-block, replace-block, prolong-block), "with 2 random positions chosen for each combination".
- MOS: VCTK-Pro 3.21, VCTK-Art 3.08. Accuracy on real aphasia speech is far lower than on simulated data (average 52.8–72.4%). — [arXiv 2409.09621](https://arxiv.org/abs/2409.09621)

**Libri-Dys (SSDM; Lian et al., NeurIPS 2024; arXiv 2408.16221)**
- Built with StyleTTS2 over LibriTTS (2,456 speakers). Scale: 3,983.44 h. Labels are phoneme and word alignments from the TTS duration model. MOS 4.15.
- Rules:
  - **Repetition** (phoneme and word): "The first phoneme or syllable of a randomly picked word was repeated 2–4 times, with pause lengths varying between 0.5 to 2.0 seconds."
  - **Block:** "A duration of silence between 0.5–2.0 seconds was inserted after a randomly chosen word".
  - **Prolongation:** phoneme duration extended "10 to 15 times its original length".
  - Also missing (deletion) and replacement.
- There are no fillers or interjections. — [arXiv 2408.16221](https://arxiv.org/abs/2408.16221)

**Libri-Co-Dys / Libri-Dys-Co (SSDM 2.0; Lian et al.; arXiv 2412.00265, Nov 2024)**
- Scale: 6,023.24 h, with an average of 2.51 dysfluencies per utterance. Co-dysfluency follows the same single-type and multi-type scheme as VCTK-Pro. Annotations are at word and phoneme level.
- The examples show word repetition "[all all all]" (3×), phoneme repetition "[t..t..t..t]", and **block silences inserted inside words** ("ɡɹˈænd[sil]fɑːðɚ", "ɐbˌ[sil]aʊt").
- SSDM 2.0's own limitations name "developing fine-grained simulation techniques" as "the primary bottleneck". — [arXiv 2412.00265](https://arxiv.org/abs/2412.00265)

**VCTK-Token (Time and Tokens; arXiv 2409.13582)**
- Uses the same simulation pipeline as VCTK-TTS but with token-level labels. — [LLM-Dys, arXiv 2505.22029](https://arxiv.org/abs/2505.22029)

**LLM-Dys (Zhang, Zhou, Lian et al.; Interspeech 2025; arXiv 2505.22029, May 2025)**
- Generation: Claude-3.5-Sonnet writes dysfluent text and labels, rendered by VITS with 109 VCTK speakers.
- Scale: 12,790 h (6,843 h word-level, 5,947 h phoneme-level), covering 11 types.
  - Word level: insertion, repetition, pause, deletion, substitution.
  - Phoneme level: insertion, repetition, pause, deletion, substitution, prolongation.
- **Fillers:** "VITS shows limitations in synthesizing fillers like 'um' and 'uh' … Therefore, we employ E2-TTS for word-level insertions", using "sample clips from each VCTK speaker" as references. So the fillers are zero-shot cloned in the same speaker's voice. Filler words make up 52.72% of insertions.
- **Pauses (blocks):** 0.8–3.5 s at word level and 0.3–1.5 s at phoneme level. Phoneme-level pauses fall *inside* words (e.g. "pɹɪ<pause>pˈɛɹɪŋ"), effectively an onset fragment plus silence.
- **Prolongation:** the phoneme's duration in the VITS duration matrix is extended by 0.17–0.8 s.
- **Repetition example:** "from, from" (word) and "plˈeɪ...plˈeɪɪŋ" (phoneme).
- **Labels:** dysfluency tokens inserted into text ([REP], [PAU], …) with token-distance metrics, i.e. token-level rather than dense frame labels.
- **Real-data results:**
  - SEP-28k (200 manually annotated clips per level, zero-shot): F1 0.67–0.93 depending on type.
  - Performance plateaus beyond about 3×12k samples. A fluent-to-disfluent training ratio of about 0.05 was optimal. — [arXiv 2505.22029](https://arxiv.org/abs/2505.22029)

**Stutter-TTS (Amazon; Zhang, Vallés-Pérez et al.; NeurIPS 2022 SyntheticData4ML workshop; arXiv 2211.09731, Nov 2022)**
- Training data: a transformer TTS trained on 40 h of internal recordings from 146 people who stutter plus 600 h of fluent speech. Stutter tokens s_repetition (phoneme repetition), s_phonation (dysrhythmic phonation) and s_block are placed before words, giving word-level control.
- Output: 100 h of synthetic speech (10 speakers, 10,000 sentences). Synthesis F1 was 0.63–0.84 by type, and about 20% of samples were excluded as TTS failures.
- Use: fine-tuning an RNN-T on it gave a 5.7% relative WER reduction on stuttered speech.
- There are no fillers or word/phrase repetitions. The data are internal, so the model and data are not released. — [arXiv 2211.09731](https://arxiv.org/abs/2211.09731)

**Mandarin generators learned from real stuttering (Lu, Zhou, Fu, Qin; Nankai; arXiv 2606.21457 v2, 11 Oct 2026)**
- Two TTS models are adapted on AS-70 (real stuttered speech with event tags) plus AISHELL-3:
  - CosyVoice2 with tag and instruction tuning.
  - MaskGCT with alignment-guided infilling of real stutter regions.
- Event types (AS-70 taxonomy): word/phrase repetition, block, prolongation, sound repetition and interjection.
- Event durations for new insertions are "empirical duration ranges … from naturally stuttered speech", scaled by speaking rate.
- Resources:
  - Synthetic AS-70: re-synthesis of the 24,395 training utterances (CosyVoice2 produced 23,232).
  - AISHELL1-Stutter: 50,000 utterances with randomly inserted events.
- Labels are event-tagged transcripts; the use case is ASR augmentation. — [arXiv 2606.21457](https://arxiv.org/abs/2606.21457)

**DisfluencySpeech (Wang & Herremans, SUTD; arXiv 2406.08820, Jun 2024)**
- A single speaker re-reads Switchboard transcripts: 5,000 clips, 34,146 s (about 9.5 h).
- It contains filled pauses (2,274 clips), discourse markers, editing terms, restarts (2,635 clips) and non-speech sounds. It comes with three transcript levels and MFA resources, but no event timestamps.
- It is acted natural disfluency, not stuttering, built for TTS. — [arXiv 2406.08820](https://arxiv.org/abs/2406.08820); [HF dataset](https://huggingface.co/datasets/amaai-lab/DisfluencySpeech)
- nyra uses it as its English disfluency benchmark. — [arXiv 2607.18934](https://arxiv.org/abs/2607.18934)

**LARD (Passali et al., LREC 2022; arXiv 2201.05041)**
- Generates repetitions, replacements and restarts in **text only**, with no audio. — [arXiv 2201.05041](https://arxiv.org/abs/2201.05041); [ACL Anthology 2022.lrec-1.249](https://aclanthology.org/2022.lrec-1.249)

**StutterZero data (arXiv 2510.18938)**
- Not a stutter generator. It pairs real SEP-28k audio and LibriStutter with MeloTTS fluent re-synthesis of ASR transcripts, about 34 h in total. — [arXiv 2510.18938](https://arxiv.org/abs/2510.18938)

**Coverage of StutterClear v2 patterns** (✔ = documented; ~ = partial or close analogue; ✘ = not found)

| v2 pattern | Prior coverage | Verdict |
|---|---|---|
| Sound / part-word repetition | LibriStutter (1–3 extra copies, 100–350 ms gaps); Libri-Dys (2–4×, 0.5–2.0 s pauses); VCTK-Art (2–4 duplicates, no pause); LLM-Dys; Stutter-TTS; Mandarin — [2009.11394](https://arxiv.org/abs/2009.11394), [2408.16221](https://arxiv.org/abs/2408.16221), [2409.09621](https://arxiv.org/abs/2409.09621), [2505.22029](https://arxiv.org/abs/2505.22029), [2211.09731](https://arxiv.org/abs/2211.09731) | ✔ covered |
| Word repetition | LibriStutter (1–2 copies); Libri-Dys/Libri-Co-Dys ("[all all all]"); VCTK-Stutter; LLM-Dys; Mandarin — [2009.11394](https://arxiv.org/abs/2009.11394), [2412.00265](https://arxiv.org/abs/2412.00265) | ✔ covered |
| Rapid 3–5× repetitions with ~0 gap | VCTK-Art duplicates a phoneme 2–4× with only interpolation buffers, the closest analogue; Libri-Dys reaches 4× but with 0.5–2.0 s pauses; LibriStutter keeps ≥100 ms pauses and ≤4 total — [2409.09621](https://arxiv.org/abs/2409.09621), [2408.16221](https://arxiv.org/abs/2408.16221), [2009.11394](https://arxiv.org/abs/2009.11394) | ~ partial (no 5×, no explicit near-zero-gap regime) |
| Phrase (2–3-word) repetition | LibriStutter phrase repetition (2–3 words) — [2009.11394](https://arxiv.org/abs/2009.11394) | ✔ covered |
| Phrase repetition with abandoned restart / revision | LibriStutter explicitly excluded revisions; LARD does restarts in text only; DisfluencySpeech has acted restarts but no stutter labels or timestamps — [2009.11394](https://arxiv.org/abs/2009.11394), [2201.05041](https://arxiv.org/abs/2201.05041), [2406.08820](https://arxiv.org/abs/2406.08820) | ✘ not found as frame-labelled synthetic audio |
| Fillers | LibriStutter (other-speaker fillers from a UCLASS pool, pitch-normalised); LLM-Dys (E2-TTS zero-shot clone of the same VCTK speaker); Mandarin interjection tag — [2009.11394](https://arxiv.org/abs/2009.11394), [2505.22029](https://arxiv.org/abs/2505.22029), [2606.21457](https://arxiv.org/abs/2606.21457) | ✔ class covered; ~ speaker-matched via TTS cloning; ✘ built from the speaker's own schwa+nasal phones |
| Prolongation | Time-stretch of the last 20% ×5 plus pitch correction (LibriStutter); TTS duration ×10–15 (Libri-Dys); EMA interpolation ×4–6 (VCTK-Art); +0.17–0.8 s via the VITS duration matrix (LLM-Dys) — [2009.11394](https://arxiv.org/abs/2009.11394), [2408.16221](https://arxiv.org/abs/2408.16221), [2409.09621](https://arxiv.org/abs/2409.09621), [2505.22029](https://arxiv.org/abs/2505.22029) | ✔ class covered; ✘ period-aligned looping not found |
| Block as onset fragment + silence | Most use plain silence after a word (VCTK-TTS, Libri-Dys, VCTK-Art); LLM-Dys phoneme-level pause inside a word (0.3–1.5 s); Libri-Co-Dys [sil] inside words; Stutter-TTS s_block learned from real PWS — [2505.22029](https://arxiv.org/abs/2505.22029), [2412.00265](https://arxiv.org/abs/2412.00265), [2211.09731](https://arxiv.org/abs/2211.09731) | ~ partial |
| Clusters (several events together) | VCTK-Pro co-dysfluency (337.84 h) and Libri-Co-Dys (2.51 events/utterance; pairs such as rep-block and prolong-block) at random positions — [2409.09621](https://arxiv.org/abs/2409.09621), [2412.00265](https://arxiv.org/abs/2412.00265) | ~ partial (co-occurrence yes; same-word chained clusters not documented) |
| Perturbed, non-identical repeated copies | TTS generators (VITS/StyleTTS2/E2-TTS/CosyVoice2) re-synthesise each copy; waveform generators copy-paste (LibriStutter, VCTK-Art duplicate) — [2408.16221](https://arxiv.org/abs/2408.16221), [2009.11394](https://arxiv.org/abs/2009.11394), [2409.09621](https://arxiv.org/abs/2409.09621) | ~ implicit in TTS; explicit perturbation of spliced copies not documented |
| MFA phone alignments of fluent corpus as the synthesis scaffold | VCTK++ and VCTK-Art use MFA alignments on VCTK; LibriStutter uses Google STT word times on LibriSpeech — [2412.00265](https://arxiv.org/abs/2412.00265), [2409.09621](https://arxiv.org/abs/2409.09621), [2009.11394](https://arxiv.org/abs/2009.11394) | ✔ covered |
| Exact frame / phoneme labels | Libri-Dys and Libri-Co-Dys (phoneme and word timestamps from the TTS duration model); VCTK-TTS JSON; LibriStutter word CSV; LLM-Dys token-level — [2408.16221](https://arxiv.org/abs/2408.16221), [2506.18510](https://arxiv.org/abs/2506.18510), [2009.11394](https://arxiv.org/abs/2009.11394) | ✔ covered |

### Inferences
- **Verdict B: the v2 generator is not conceptually novel. Every class has prior synthetic coverage, and MFA-scaffolded waveform or feature injection dates back to VCTK++ in 2023.** Novelty is at the level of mechanisms and the stutter-specific distribution:
  - (i) own-voice fillers from the speaker's schwa and nasal phones, as opposed to other-speaker fillers (LibriStutter) or zero-shot TTS fillers (LLM-Dys);
  - (ii) period-aligned looping prolongation, as opposed to time-stretch or duration scaling;
  - (iii) abandoned-restart phrase repetitions with frame labels;
  - (iv) a deliberate rapid 3–5× near-zero-gap regime;
  - (v) same-word chained clusters.
- These are defensible as "engineering contributions motivated by real stutter acoustics", not as new data paradigms.
- The strongest competing resources by scale are Libri-Co-Dys (6,023 h, StyleTTS2, 2,456 speakers) and LLM-Dys (12,790 h). Both are TTS-based with no recorded human prosody, so their stutter realism is bounded by rule-based or LLM-chosen positions.
- The 2026 Mandarin work is the first I found that *learns* stutter acoustics from real stuttering at scale (CosyVoice2/MaskGCT on AS-70). An English analogue trained on FluencyBank or UCLASS does not appear to exist; Stutter-TTS is internal Amazon data. That is a plausible next step for StutterClear's generator.
- StutterClear's own-voice splicing approach keeps real speaker timbre and channel. TTS generators (VCTK and LibriTTS voices) do not, and that difference is what the sim-to-real argument rests on. Every TTS-based paper reports large drops on real speech, e.g. Stutter-Solver at 52.8–72.4% accuracy on aphasia and YOLO-Stutter at 54% average accuracy.

### Gaps
- I did not read the Harvill et al. 2022 PDF directly. Its augmentation types come from Shih et al.'s description, so its exact repetition counts and gaps are unverified.
- I did not read the VCTK-TTS/YOLO-Stutter rules first-hand in this session; the block = 0.5–2.0 s silence figure comes from earlier StutterClear notes citing arXiv 2408.15297. I did not confirm whether LLM-Dys's LLM ever produces 3+ word repetitions or multi-word phrase repetitions; only examples were visible.
- I did not check repos (Berkeley-Speech-Group/LLM-Dys, eureka235/Stutter-Solver, rorizzz/YOLO-Stutter) for undocumented code options such as near-zero-gap repetitions or filler cloning.
- Possibly relevant but not read: the SLT 2024 Mandarin challenge winner's augmentation (silence insertion to mimic blocks, part repetition, speed and tempo perturbation; [arXiv 2409.05430](https://arxiv.org/abs/2409.05430), via earlier notes), and the cascaded SSL/augmentation work of Mujtaba et al. ([arXiv 2406.10177](https://arxiv.org/abs/2406.10177)), whose augmentation recipe I did not inspect.

---

## 3. Has anyone combined synthetic frame labels, real clip labels (MIL) and real silver frame labels in one training setup?

### Takeaway
Synthetic frame-level pretraining followed by real clip-level MIL fine-tuning is **already done**: Harvill et al. 2022 and Shih et al. 2024. MIL alone on real clips (no synthetic) is state of the art for frame segmentation on CASA (Sušac et al., Jun 2026). Adding a third stream of real silver frame labels from verbatim ASR was **not found**. The three-source combination appears **OPEN**.

### Cited Findings
- **Harvill, Hasegawa-Johnson and Yoo, "Frame-Level Stutter Detection"** (Interspeech 2022) is described as being "first pretrained on LibriSpeech with synthetic disfluencies augmented during training with frame-level supervision. Then, they added a max pooling layer to finetune on utterance-level SEP-28K dataset." — [Shih et al., arXiv 2409.10704](https://arxiv.org/abs/2409.10704); [ISCA archive](https://www.isca-archive.org/interspeech_2022/harvill22_interspeech.html)
- **Shih et al. (SLT 2024):** WavLM-Large (frozen) plus HConv.
  - Stage 1 pretrains on LibriSpeech-360 with synthetic frame targets plus an auxiliary phoneme CTC loss.
  - Stage 2 fine-tunes on SEP-28K using the argmax frame against the utterance label (max-pooling MIL).
  - Word-level F1 on clinical data: 0.554 against 0.411 for Harvill. — [arXiv 2409.10704](https://arxiv.org/abs/2409.10704)
- **Sušac, Bayerl and Džapo (arXiv 2606.20338, Jun 2026):** instance and embedding (attention) MIL trained on SEP-28k-E clip labels only.
  - CASA frame F1: 0.70 (Whisper with attention pooling).
  - They attribute their gain over Shih to encoder fine-tuning, "which reduced the need for pretraining on artificial stuttering".
  - They suggest that "pre-training or fine-tuning on a frame-level dataset might be useful" as future work. — [arXiv 2606.20338](https://arxiv.org/abs/2606.20338)
- **LLM-Dys** trains on synthetic data plus a small fraction of fluent VCTK and evaluates zero-shot on SEP-28k. For UCLASS it fine-tunes a classification head on 80 real clips (binary). There is no MIL and no silver labels. — [arXiv 2505.22029](https://arxiv.org/abs/2505.22029)
- **StutterCut** combines weak utterance-level labels (a pseudo-oracle classifier) with unsupervised graph cuts. It uses no synthetic frame supervision and no ASR labels. — [arXiv 2508.02255](https://arxiv.org/abs/2508.02255)

### Inferences
- **Verdict:** synthetic frame + real clip MIL is ALREADY DONE (2022 and 2024). Synthetic frame + real clip MIL + real verbatim-ASR silver frame labels with per-class masks is OPEN (not found).
- StutterClear's per-class loss masking (masking prolongation and block on silver data) is a natural partial-label or multi-source supervision design. I found no stuttering paper doing class-wise masking across heterogeneous label sources.
- The fair baselines for StutterClear's teacher are Sušac 2026 (MIL only, CASA F1 0.70) and Shih 2024 (synthetic + MIL). An ablation showing that silver labels add over these would be the core novelty evidence.

### Gaps
- I did not search the generic weak-supervision and sound-event-detection literature (e.g. DCASE strong + weak + synthetic label mixing), where the three-source pattern is common outside stuttering; novelty is only claimed within stuttering.
- I could not access Harvill et al.'s full text to confirm details beyond Shih et al.'s summary.

---

## 4. What evidence exists on how well verbatim ASR detects stuttering or disfluency events (precision/recall of disfluency tokens, timing on stuttered speech)?

### Takeaway
There is no published precision or recall of CrisperWhisper (v1 or 2.0) disfluency tokens against stuttering-event annotations. The nearest evidence is:
- (i) nyra's per-type F1 on acted, non-stuttered DisfluencySpeech, where cut-off F1 is weak (about 39–43 EN for 2.0);
- (ii) word-timing MAE of 102 ms on FluencyBank stuttered speech for the 2.0 method, versus the model card's ~30–41 ms on read and conversational speech;
- (iii) general verbatim-ASR studies showing part-word and multi-syllable repetitions are the hardest events, and that blocks and prolongations are largely invisible to ASR.

### Cited Findings

**nyra 2026, DisfluencySpeech EN (n=4,957), event and per-type F1**
- Baselines (eF1 / filler F1 / sound F1 / cut-off F1 / repetition F1 / lenient cut-off recall):
  - Whisper: eF1 12.0.
  - Canary-1B: 19.9.
  - Reverb: 86.3.
  - AssemblyAI Universal-3-Pro: 85.8.
  - **CrisperWhisper v1:** eF1 73.2; fF1 81.7, sF1 0.0, cF1 0.0, rF1 68.0, lcR 73.2.
- Their model with tags:
  - FT0: eF1 90.7; fF1 90.1, sF1 87.3, **cF1 43.2**, rF1 87.2, lcR 95.4.
  - FT10k: eF1 87.9; cF1 39.1, lcR 95.1.
- German FT10k: eF1 95.0, cF1 88.7. — [arXiv 2607.18934](https://arxiv.org/abs/2607.18934)

**Word timing on FluencyBank (stuttered), MAE and F1 at 50/100/200 ms collars**
- Base Whisper: 568 ms.
- WhisperX: 200 ms.
- MFA: 142 ms (F1@50 56.1%).
- CrisperWhisper v1 with sharpening: 122 ms.
- **Their best (averaged-head loss with sharpening): 102 ms, F1@50 38.6%, F1@100 61.1%, F1@200 80.3%.**
- On read speech (TIMIT) their method reaches 36 ms. Timing supervision came from MFA alignments, not hand labels. — [arXiv 2607.18934](https://arxiv.org/abs/2607.18934)
- **Model card:** "Mean boundary error is about 30 ms on read speech and 41 ms on conversational speech". It does not mention stuttered speech. — [HF model card](https://huggingface.co/nyralabs/CrisperWhisper2.0_medium)

**Toyin, Umesh and Aldarmaki (MBZUAI/IIT Madras; arXiv 2606.31112, 30 Jun 2026)**
- 11 ASR models on FluencyBank Timestamped (3,430 samples) with CASA event alignment. CrisperWhisper was not included.
  - Best verbatim WER: NVIDIA Conformer-CTC, 17.20. Whisper-large-v3: 25.01.
  - CTC models favour verbatim output; autoregressive models favour intended output.
  - Incomplete-syllable repetitions (ISR) and multi-syllable-unit repetitions (MUR) are hardest; prolongations are easiest.
  - Example: "you're you're you're a bully" → "you your your bolly"; "bur bureaucratic" → "bu bureaucratic". — [arXiv 2606.31112](https://arxiv.org/abs/2606.31112)

**Romana et al. (arXiv 2311.00867, Nov 2023), Switchboard, frame-level**
- ASR→BERT with Whisper-FT transcripts, F1 by class: filled pause 0.56, repetition 0.48, revision 0.45, restart 0.07, partial word 0.17. Partial-word recall fell from 0.88 (manual transcripts) to 0.28.
- An acoustic-only WavLM detector scored F1 0.87 for filled pauses, 0.64 for repetitions and 0.32 for partial words.
- ASR disfluent-frame error rate (FER-D): Whisper-FT 36%, Azure 47%, WavLM-FT 66%.
- The text's recall summary (0.71→0.26 unweighted) conflicts with its own Table IIIb, which gives Whisper-FT unweighted recall 0.44; 0.26 is the WavLM-FT value. — [arXiv 2311.00867](https://arxiv.org/abs/2311.00867)

**Blocks and Whisper**
- Shih et al. needed a margin of up to 0.2 s because "Whisper truncates the audio segment where Block manifests". — [arXiv 2409.10704](https://arxiv.org/abs/2409.10704)
- Sridhar & Wu (Interspeech 2025) found Whisper v3 often renders "j-j-j-just" as "so just". Hallucination rates on stuttered speech are higher, rising to 22.6% (v2) on sound repetitions. — [ISCA PDF](https://www.isca-archive.org/interspeech_2025/sridhar25_interspeech.pdf) (via prior StutterClear notes)

**Human-in-the-loop evidence on real stuttering (ASU 2026)**
- After CrisperWhisper 2.0 plus the FBT labels, a human still added "missed sound and word repetitions", and all 129 prolongations and 150 blocks. — [arXiv 2610.19235](https://arxiv.org/abs/2610.19235)

### Inferences
- StutterClear's assumption of "word timestamps ~30–40 ms" matches the model card's read and conversational numbers. On **stuttered** FluencyBank speech the same paper reports **~102 ms MAE, and only ~39% of boundaries within ±50 ms**. Silver frame labels should therefore get boundary tolerance (soft or dilated edges, or ignore bands of about 100 ms around token edges), not hard frame targets.
- The cut-off→sound-repetition mapping rests on the weakest token class: cut-off F1 is about 39–43 (EN) for CrisperWhisper 2.0 on acted speech and 0.0 for v1. Expect low recall for sound repetitions in the silver labels. Lenient cut-off recall (~95%) suggests positions are found even when the hyphenated form is wrong, which supports mapping "any token at a cut-off position" rather than relying on the exact surface form.
- Precision of filler and repetition tokens (about 87–90 F1 on acted speech) is likely the most trustworthy part of the silver labels. Whether it holds on severe stuttering is unmeasured.
- A cheap, publishable validation StutterClear could add: score CrisperWhisper 2.0 silver labels against FluencyBank++ or CASA frame annotations (or the ASU onset layer). No one has published that number, and it would directly quantify silver-label quality.

### Gaps
- No precision or recall of CrisperWhisper v1 or 2.0 disfluency tokens on SEP-28k, UCLASS, FluencyBank++ or CASA was found.
- No independent (non-vendor) evaluation of ElevenLabs Scribe verbatim mode or "Whisper with disfluency prompts" on stuttered speech was found. The 79.2 Scribe v2 figure is nyra's own benchmark on the model card.
- I could not confirm whether CrisperWhisper v1 emitted uppercase `[UM]`/`[UH]`. The 2.0 model card shows lowercase `[um]`/`[uh]`; StutterClear's parser should accept both.
