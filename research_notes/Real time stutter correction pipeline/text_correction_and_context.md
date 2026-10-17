# Text-level disfluency removal and context-aware ASR error correction (the "correction" stage of StutterClear), as of October 2026

Scope: the stage that sits between streaming ASR and personalized streaming TTS. It takes partial and final ASR hypotheses plus a user profile and conversation history, and outputs what the user meant to say. It must not change meaning, must abstain when unsure, and must run incrementally. All dates are publication or arXiv dates. Foundational work is marked [FOUNDATIONAL].

---

## Q1. Classic and streaming disfluency detection on transcripts: schemes, taggers, incremental models, F1 and lookahead

### Takeaway
The standard method is to tag tokens in the Switchboard reparandum/interregnum/repair scheme with a small BERT-style tagger, then delete the tokens tagged disfluent. Offline F1 on Switchboard is about 88 to 90 for small on-device models. Streaming models need about 0 to 2 tokens of lookahead. The best published streaming design is Google's "Teaching BERT to Wait" (2022), which learns *when* to wait: on average it waits 0.21 tokens and about 1 token on interregna and reparanda. It reaches a streaming F1 of 0.83 and an edit overhead of 0.09, with a 3.1M-parameter model.

### Cited Findings
- **Annotation scheme [FOUNDATIONAL, Shriberg 1994/1997].** The scheme has four parts. The *reparandum* is the material to be removed. The *interruption point* is marked "+". The *interregnum* is optional material such as "uh, I mean". The *repair* replaces the reparandum, and its first word is the "repair onset". Example: "A flight [to Boston + {uh I mean} to Denver] on Friday". Detection usually means finding and deleting the reparandum, and often the interregnum too. — [Chen et al. 2022, "Teaching BERT to Wait" (arXiv 2205.00620)](https://arxiv.org/abs/2205.00620)
- **Switchboard as the benchmark [FOUNDATIONAL].** The benchmark is the Treebank-3 (LDC99T42) disfluency-annotated Switchboard, using the Charniak & Johnson (2001) train/dev/test split. Disfluent categories are EDITED (reparandum), INTJ (filled pauses such as uh/um) and PRN (parentheticals such as "you know" or "I mean"). — [Chen et al. 2022](https://arxiv.org/abs/2205.00620); [Teleki et al., DRES (arXiv 2509.20321, v2 Mar 2026)](https://arxiv.org/abs/2509.20321)
- **Pipeline vs end-to-end.** A separate post-ASR disfluency-removal step "has been shown to perform better than end-to-end ASR models that generate fluent text from disfluent speech" (citing Jamshid Lou & Johnson, Findings of EMNLP 2020). — [Chen et al. 2022](https://arxiv.org/abs/2205.00620)
- **Why it matters downstream.** Inserting disfluencies into SQuAD-v2 questions drops BERT by up to 28 F1 and T5 by up to 20 F1 (Disfl-QA, Gupta et al. 2021). — [Chen et al. 2022](https://arxiv.org/abs/2205.00620)
- **Lineage of tagger models [FOUNDATIONAL]:**
  - Noisy-channel and TAG models (Charniak & Johnson 2001; Johnson & Charniak 2004)
  - BiLSTM taggers (Zayats, Ostendorf & Hajishirzi 2016)
  - Noisy BiLSTM (Bach & Huang 2019)
  - Self-training a self-attentive parser (Jamshid Lou & Johnson, ACL 2020), state of the art at the time
  — [DRES related work (2509.20321)](https://arxiv.org/abs/2509.20321); [Zayats et al. 2016 (arXiv 1604.03209)](https://arxiv.org/abs/1604.03209); [Jamshid Lou & Johnson ACL 2020](https://aclanthology.org/2020.acl-main.346)
- **Small on-device BERT taggers (Google, Interspeech 2021).** Rocholl et al. report:
  - A 6×96 small-vocab BERT quantized to **1.3 MiB** reaches **88.4 F1**. BERT-base (416 MiB) reaches **90.4 F1**. Both are measured on the Switchboard test set with reparandum-only labels.
  - A 12×128 small-vocab model scores **89.9 F1**, and **89.8 F1** when quantized to 3.3 MiB.
  — [Rocholl et al. 2021 (arXiv 2104.10769)](https://arxiv.org/abs/2104.10769); [ISCA archive](https://www.isca-archive.org/interspeech_2021/rocholl21_interspeech.html)
- **Size and speed of BERT_SV.** It has about **3.1M parameters**, against about 108.9M for BERT-base. Its vocabulary is 5K instead of 30K tokens. It cuts runtime latency by about 80% relative to BERT-base. — [Chen et al. 2022](https://arxiv.org/abs/2205.00620)
- **Incremental and streaming lineage:**
  - Hough & Schlangen 2015: the first competitive RNN for incremental disfluency detection, an Elman RNN with a Markov decoder (Interspeech).
  - Hough & Schlangen 2017: joint incremental disfluency detection and utterance segmentation, which gave higher accuracy and lower latency than either task alone (EACL).
  - Rohanian & Hough 2020: multi-task with POS tagging (COLING).
  - Rohanian & Hough 2021: BERT-large with "add-M" prefix training plus GPT-2 "prophecy" decoding that hallucinates the right context (ACL).
  - Madureira & Schlangen 2020: BERT on partial inputs (EMNLP).
  - Chen et al. 2020: controllable time-delay transformer for real-time punctuation and disfluency detection (ICASSP).
  - Zwarts et al. 2010: incremental noisy-channel model.
  — [Chen et al. 2022, related work](https://arxiv.org/abs/2205.00620)
- **"Teaching BERT to Wait" (Chen, Zayats, Walker, Padfield; Google Research, arXiv May 2022).** BERT_SV has two heads: a disfluency head (fluent or disfluent) and a *wait* head (predict or wait). The loss is L = L_FULL + γ·L_PREFIX + λ·L_LATENCY, trained restart-incrementally on every 1-word-extended prefix. At inference the model emits only fluent-predicted tokens to the left of the first WAIT. — [Chen et al. 2022](https://arxiv.org/abs/2205.00620)
- **Results on the Switchboard validation set** (streaming F1 / precision / recall / edit overhead (EO) / time-to-detection (TTD) / average wait time (AWT) / final F1):
  - Non-incremental BERT_SV, LA=0: 0.76 / 0.74 / 0.78 / EO 0.31 / TTD 1.46 / AWT 0 / final F1 0.89
  - **Streaming BERT_SV: 0.83 / 0.92 / 0.75 / EO 0.09 / TTD 2.32 / AWT 0.21 / final F1 0.88**
  - Fixed lookahead LA=1: 0.83 / EO 0.10 / AWT 1.00
  - Fixed lookahead LA=2: 0.85 / EO 0.05 / AWT 2.00 / final F1 0.89
  So the streaming model matches LA=1 accuracy with **79% less average wait**. — [Chen et al. 2022, Table 1](https://arxiv.org/abs/2205.00620)
- **Wait time by token type** (in tokens): interregnum **1.06**, reparandum **0.76**, repair 0.74, repair onset 0.46, fluent **0.15**, edit term 0.14. Reparanda are the hardest class because they cannot be identified until the interregnum or repair onset arrives. — [Chen et al. 2022, Table 2](https://arxiv.org/abs/2205.00620)
- **Against Rohanian & Hough 2021 (BERT-large + prophecy), Switchboard test set:** EO 0.60 → **0.09** and first-time-to-detection 0.31 → **0.11** words. That is −85% EO and −65% FTD, with far fewer parameters. — [Chen et al. 2022, Table 3](https://arxiv.org/abs/2205.00620)
- **The authors' own caveat.** Training data comes from "fluent English-speaking, able-bodied" speakers. The model "would likely perform noticeably worse for users who speak with … speech impediments", and its average wait time would likely be longer. — [Chen et al. 2022, Ethics](https://arxiv.org/abs/2205.00620)
- **Joint streaming ASR and disfluency detection.** A Transformer encoder-decoder that jointly solves ASR and disfluency detection in streaming mode has been proposed (Sony/CMU, 2022–23). — [arXiv 2211.08726](https://arxiv.org/abs/2211.08726)

### Inferences
- For StutterClear, a BERT_SV-class tagger with a wait head (about 3M parameters, a few MB) is the right *first-pass* mechanism for disfluency deletion. It is deletion-only, so it cannot hallucinate, and it gives both latency and abstention ("wait") for free. The cost of re-encoding each prefix is small at this size. Even with full restart-incremental re-encoding, a 3M-parameter model on a 20 to 30 token window should take single-digit milliseconds per update on a modern CPU. This is an estimate, not a reported measurement.
- Self-repairs and restarts such as "I want the — I need the blue one" are exactly the EDITED/reparandum class. Taggers handle these, but they need about 1 token of right context after the repair onset ("I need") before they can confidently delete "I want the".

### Gaps
- I could not find a "Shea et al." streaming disfluency paper. The Google line of work I found is Rocholl et al. 2021 and Chen et al. 2022.
- I did not retrieve the exact Switchboard F1 of Jamshid Lou & Johnson (ACL 2020) or the final F1 of Rohanian & Hough (ACL 2021). Check the [ACL 2020 PDF](https://aclanthology.org/2020.acl-main.346) and ACL 2021 paper tables.
- Rocholl et al. 2021 does not report per-utterance wall-clock latency for the 1.3 MiB model. Only the relative 80% reduction against BERT-base is reported.

---

## Q2. Stuttering vs typical disfluency: how ASR renders stuttering and what the text stage actually receives

### Takeaway
Stuttering-like disfluencies are part-word repetitions, prolongations and blocks. They differ from Switchboard-style disfluencies, which are mostly whole-word and phrase repairs plus fillers. How they show up in text depends on the ASR. Apple's consumer ASR rendered them mainly as **insertions** (80.9% of errors). WER correlated r=.85 with part-word repetitions, and blocks caused endpointer truncation. Modern "clean" ASR such as Whisper-class models tends to *omit* disfluencies, and sometimes hallucinates instead. Either way, the correction stage sees a mix of fragment tokens, duplicated words, and silently dropped or invented content. Whether to evaluate against a verbatim or an intended reference changes which model looks best.

### Cited Findings
- **Apple study (CHI 2023, arXiv 2302.09044).** The researchers surveyed 61 people who stutter (PWS) and recorded 91 PWS giving voice-assistant commands and dictation. — [Lea et al. 2023](https://arxiv.org/abs/2302.09044); [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2302.09044)
- **Baseline consumer-ASR WER for PWS:**
  - 19.9% pooled over all 91 participants.
  - By severity: **mild 4.8%, moderate 13.6%, severe 49.2%**.
  - Errors were **80.9% insertions**, 17.5% substitutions and 1.6% deletions.
  — [Lea et al. 2023](https://ar5iv.labs.arxiv.org/html/2302.09044)
- **What drives the errors.**
  - Insertions correlate with part-word repetitions (r=.73) and whole-word repetitions (r=.60).
  - WER correlates with part-word repetitions (**r=.85**) and whole-word repetitions (r=.60).
  - Prolongations, blocks and interjections were not significant for WER.
  - Prevalence across utterances: part-word repetitions **58.6%**, blocks 38.7%, prolongations 36.4%, whole-word repetitions 5.5%, interjections 2.8%.
  — [Lea et al. 2023](https://ar5iv.labs.arxiv.org/html/2302.09044)
- **Blocks cause truncation instead.** The endpointer cut off **23.8%** of PWS utterances, against a 3% target for the general population. Truncation correlates with blocks (r=.64), interjections (r=.48) and part-word repetitions (r=.44). — [Lea et al. 2023](https://ar5iv.labs.arxiv.org/html/2302.09044)
- **Tuning the endpointer.**
  - Truncation fell to 4.9% (−79.1%) at a cost of about **+1.2 s** median delay.
  - It fell to 2.5% at about +1.7 s.
  - It fell to 0.8% at more than +3 s.
  — [Lea et al. 2023](https://ar5iv.labs.arxiv.org/html/2302.09044)
- **Text-level "dysfluency refinement".** This is a direct precedent for StutterClear's correction stage. It removes the filler "oh" (except when it means zero). It also removes adjacent repeated words or phrases when an n-gram LM gives their consecutive appearance a probability below a threshold τ.
  - Refinement alone: WER 25.4% → **18.1%**.
  - Decoder tuning: 25.4% → 12.4%.
  - Both combined: **9.9%**.
  - Intent error rate: 10.4% → 5.4%.
  - Whole-word repetitions improved in 64.7% of utterances and regressed in 0.3%. Part-word repetitions improved in 25.2% and regressed in 0.0%.
  — [Lea et al. 2023](https://ar5iv.labs.arxiv.org/html/2302.09044)
- **Survey results.** 61.0% of PWS said voice assistants cut them off, and 57.6% said voice assistants did not understand them well enough. — [Lea et al. 2023](https://ar5iv.labs.arxiv.org/html/2302.09044)
- **"Lost in Transcription" (Mujtaba et al. 2024).** Six leading ASR systems were tested on real PWS speech and on synthetic stuttered LibriSpeech. All six showed "a consistent and statistically significant accuracy bias against disfluent speech", with significant syntactic and *semantic* errors. The abstract gives no numbers. — [arXiv 2405.06150](https://arxiv.org/abs/2405.06150)
- **ASR tends to drop disfluencies.** State-of-the-art ASR systems "are often optimized to omit disfluencies" and under-transcribe short interjections. Whisper-style ASR also produces harmful hallucinations on disfluent speech ("Careless Whisper", Koenecke et al., FAccT 2024). — [DRES §2.1 (2509.20321)](https://arxiv.org/abs/2509.20321); [Learning to Hear Hesitation (arXiv 2606.14391, Jun 2026)](https://arxiv.org/abs/2606.14391)
- **Two valid references.** Atypical speech has a verbatim reference (repetitions and prolongations kept) and an intended reference (fluent, canonical text). Single-reference evaluation "rewards systems that delete disfluencies". A test of 11 ASR models (encoder-decoder, CTC and transducer) found that performance and rankings change with the reference used. — [Dual-Reference Benchmarking for Atypical ASR (arXiv 2606.31112, Interspeech 2026)](https://arxiv.org/abs/2606.31112)
- **Other stutter-specific ASR work:**
  - Amazon "Detect and Pass" (2022) for stuttered speech with limited data. — [arXiv 2202.05396](https://arxiv.org/abs/2202.05396)
  - Stutter-TTS (2022), which synthesizes stuttered speech to improve recognition. — [arXiv 2211.09731](https://arxiv.org/abs/2211.09731)
  - A paper on the difficulty of token-level modeling of dysfluency (2025). — [arXiv 2512.02027](https://arxiv.org/abs/2512.02027)
- **Datasets:**
  - SEP-28k (Apple 2021): about 28k podcast clips labelled for stuttering events. — [arXiv 2102.12394](https://arxiv.org/abs/2102.12394)
  - AS-70: Mandarin stuttered speech for ASR and stuttering event detection. — [arXiv 2406.07256](https://arxiv.org/abs/2406.07256)
  - The 2024 Mandarin Stuttering Event Detection and ASR Challenge. — [arXiv 2409.05430](https://arxiv.org/abs/2409.05430)
  - Boli: multilingual Indian stuttered speech. — [arXiv 2501.15877](https://arxiv.org/abs/2501.15877)
  - FluencyBank, used by STEAMROLLER. — [arXiv 2601.10223](https://arxiv.org/abs/2601.10223)

### Inferences
- What the correction stage receives depends on the ASR. With a verbatim-leaning ASR, expect fragments ("b", "ba", "ball ball"), duplicated words and occasional nonsense tokens for part-word repetitions. These are fixable by deletion and so are low risk. With a clean or omission-leaning ASR, the text may already be fluent, but blocks can cause truncation or dropped words, and the ASR may hallucinate. Neither case can be fixed by deletion: a dropped word needs *insertion* or completion, and that is where meaning-change risk lives.
- Switchboard-trained taggers have never seen part-word fragment patterns at stutter density. Fine-tuning on synthetic stutter-pattern text, or using simple rules plus an LM-probability gate as Apple did, is likely needed for part-word repetitions.
- The upstream stutter detector's event labels (repetition, block, prolongation, with timestamps) can be passed to the text stage as features. Block timestamps in particular tell the stage where a word may be missing rather than repeated.

### Gaps
- I found no study that tabulates exactly how Whisper, Parakeet or other current open ASR models spell part-word repetitions (for example "b-b-ball" vs "ball" vs a hallucinated word). The Dual-Reference paper's full tables may contain this but were not retrieved.
- The Lost in Transcription abstract gives no fluent-vs-disfluent WER gap.

---

## Q3. LLM-based correction: generative error correction (GER), LLM disfluency removal, over-editing risk, and how to constrain it

### Takeaway
LLM-based generative error correction can lower WER by using N-best lists and context. But unconstrained LLMs over-edit, hallucinate, and are biased toward frequent or grammatical words. On disordered speech, naive LLM correction **nearly doubled WER on TORGO** (10.83% → 20.01%). On disfluency removal, a deletion-only task, reasoning models systematically over-delete fluent words and small models under-delete. The reliable constraints are deletion-only or edit-tag outputs, confidence-guided targeting, select-before-rewrite training, and per-utterance abstention. All of these are compatible with StutterClear's "do not change meaning" requirement.

### Cited Findings
- **HyPoradise (NeurIPS 2023 datasets track).** It is the open benchmark for LLM GER, with about 334K pairs of N-best ASR hypotheses and references (other versions say 316K). Fine-tuned LLM correction can beat the oracle WER of traditional N-best re-ranking and can recover words absent from the N-best list. — [HyPoradise (arXiv 2309.15701)](https://arxiv.org/abs/2309.15701); [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6492267465a7ac507be1f9fd1174e78d-Abstract.html)
- **Related systems:**
  - GenSEC challenge (2024): LLM GER challenge and baselines for ASR, speaker tagging and emotion. Organizers include NVIDIA, MIT, Edinburgh and NTU. — [arXiv 2409.09785](https://arxiv.org/abs/2409.09785)
  - Whispering LLaMA: cross-modal GER (Radhakrishnan et al., EMNLP 2023).
  - N-best T5 with "constrained decoding space" (Ma et al., Interspeech 2023).
  — both cited in [Hernandez et al. 2025 (arXiv 2509.25048)](https://arxiv.org/abs/2509.25048)
- **Small encoder-decoder GER.** FlanEC (Flan-T5 for GenSEC, 2025). — [arXiv 2501.12979](https://arxiv.org/abs/2501.12979)
- **Over-correction on disordered speech (Hernandez et al., FAU Erlangen, Sep 2025).** Setup: LLaMA-3.1-8B-Instruct with LoRA (r=16, about 42M trainable parameters), fine-tuned on 130K reference–hypothesis pairs from the Speech Accessibility Project (Parkinson's speech), using Parakeet TDT-CTC 110M as the ASR.
  - Naive LLM correction **raised** TORGO WER from 10.83% to **20.01%**.
  - On SAP spontaneous ("unshared") speech, naive correction raised WER from 9.94% to 10.56%.
  - Naive correction also showed frequency bias (TEASED → ASKED, with a 61:1 lemma frequency ratio in training) and grammaticality bias (WHAT → WHAT'S).
  — [Hernandez et al. 2025](https://arxiv.org/abs/2509.25048)
- **DRES (Teleki et al., Texas A&M and KTH; arXiv Sep 2025, v2 Mar 2026, under review for Interspeech 2026).** Disfluency removal is *deletion-only*: the fluent transcript is a monotonic subsequence of the input. DRES evaluated GPT-4o, GPT-4o-mini, o4-mini, Llama-3.1-8B, Llama-3.2-1B/3B, Llama-3.3-70B, MobileLLM (125M–1B), Qwen3 (0.6–8B) and Phi-4-mini on gold Switchboard transcripts. — [DRES (2509.20321)](https://arxiv.org/abs/2509.20321)
  - **Editing policies.** Models fall into four stable precision–recall clusters (silhouette 0.54). GPT models are "balanced". **Reasoning models (o4-mini, Phi-4 reasoning) systematically over-delete fluent content.** Small or base models under-delete or are "poor". Scale improves performance within a policy but does not change the policy. — [DRES](https://arxiv.org/abs/2509.20321)
  - **By category.** LLMs handle EDITED (repairs) well but struggle with INTJ and PRN. This is the opposite of classic taggers, for which those categories are easiest. — [DRES](https://arxiv.org/abs/2509.20321)
  - **Context length.** Segmenting transcripts into chunks of about 4 sentences consistently moves models toward "balanced". Long context causes instability even for long-context models. — [DRES](https://arxiv.org/abs/2509.20321)
  - **Fine-tuning.** Fine-tuning raises the word-level F1 (E_F) from the mid-70s to the mid-90s for GPT-4o-mini and from the high-60s to above 90 for Llama-3.2-3B (+22 to +23 points). It costs a small but significant amount on MMLU, GSM8K and CoQA. — [DRES, Table 2](https://arxiv.org/abs/2509.20321)
  - **Deployment recommendations:**
    - R1: use small *under-deletion* models (for example Qwen3-1.7B) at the edge for real-time intent preservation.
    - R3: 8B-class models with few-shot prompting give a reasonable latency–accuracy trade-off.
    - R4: avoid reasoning models for literal repair.
    - R5: in command-driven interfaces, prefer under-deletion so fluent nouns and verbs are protected.
    — [DRES, Table 3](https://arxiv.org/abs/2509.20321)
- **Ways to constrain LLM edits:**
  - **Confidence-informed prompting.** Word confidences go into the prompt, for example "HOW[1.00] MANY[0.85] RAFELLES[0.61]". TORGO WER becomes 10.58% instead of 20.01% with naive correction, and SAP-unshared becomes 9.48% instead of 10.56%. Word-level confidence *filtering* (correct only if some word is below a threshold) gives 10.73% on TORGO. — [Hernandez et al. 2025](https://arxiv.org/abs/2509.25048)
  - **Select-or-Rewrite, oracle-guided fine-tuning (KTH, Oct 2026).** Training uses three pools: *corrections* (the reference is in the N-best but is not the top-1, so the model learns to select it), *regularizers* (the top-1 is already correct, so the model learns not to over-edit) and *hard negatives* (the reference is absent, to suppress hallucination). Gains were clearest at about 3B parameters. CommonVoice results were inconclusive, possibly because of pretraining contamination. — [arXiv 2610.15056](https://arxiv.org/abs/2610.15056)
  - **RLLM-CF (May 2025).** Three stages: error pre-detection, then chain-of-thought sub-task correction, then reasoning verification, with GPT-4o and no fine-tuning. It reports relative CER/WER reductions of 21%, 11%, 9% and 11.4% across AISHELL-1, AISHELL-2 and LibriSpeech. — [arXiv 2505.24347](https://arxiv.org/abs/2505.24347)
  - **Voice Memory (NVIDIA, Jul 2026).** A frozen corrector decides *per utterance* whether to act or abstain and keep the 1-best. It is guided by a per-domain memory file that a score-gated optimizer edits. Unconstrained GER "breaks correct tokens" in **up to 64%** of its edits on financial news; Voice Memory cuts this to **35%**. Weighted WER falls from 8.36% to 7.52% over 10 HyPoradise domains, and **no dataset regresses below 1-best**. — [arXiv 2607.26410](https://arxiv.org/abs/2607.26410)
  - **Rare-word GER with synthetic data and phonetic context** (Hitachi, May 2025). — [arXiv 2505.17410](https://arxiv.org/abs/2505.17410)
- **Speech-to-fluent systems for stuttering:**
  - STEAMROLLER (Jan 2026): ASR, then multi-agent text repair "preserving semantic intent", then TTS, in real time, evaluated on FluencyBank. The abstract reports "clear WER reduction" but gives no numbers, no latency and no explicit meaning-change safeguard. — [arXiv 2601.10223](https://arxiv.org/abs/2601.10223)
  - StutterZero/StutterFormer (Oct 2025): end-to-end speech conversion for stutter transcription and correction. — [arXiv 2510.18938](https://arxiv.org/abs/2510.18938)
  - Disfluency-aware objective tuning for multilingual LLM speech correction (IIT Patna, May 2026). — [arXiv 2605.12242](https://arxiv.org/abs/2605.12242)
  - LLM-agent post-ASR correction for dysarthric speech evaluated "beyond WER" on semantic fidelity (UIUC/KAIST, Jan 2026). — [arXiv 2601.21347](https://arxiv.org/abs/2601.21347)
- **Synthetic disfluency data:**
  - LARD (Passali et al., LREC 2022) generates repetitions, replacements and restarts. It released a dataset for four tasks (detection, classification, extraction, correction) and outperforms rule-based augmentation. — [arXiv 2201.05041](https://arxiv.org/abs/2201.05041); [ACL Anthology 2022.lrec-1.249](https://aclanthology.org/2022.lrec-1.249)
  - "Artificial Disfluency Detection, Uh No, Disfluency Generation for the Masses" (2022). — [arXiv 2211.09235](https://arxiv.org/abs/2211.09235)
  - LLM as disfluency generator (PKU, 2024). — [arXiv 2403.08229](https://arxiv.org/abs/2403.08229)

### Inferences
- In StutterClear, the LLM should **not** be a free rewriter on the hot path. The safest design separates three operations:
  - (a) *Deletion* of stutter fragments, repetitions, fillers and abandoned restarts. Use a tagger, or an LLM constrained to output a keep/delete mask or a monotonic subsequence. Hallucination is then structurally impossible.
  - (b) *Substitution* of low-confidence words. Allow this only where ASR confidence is low, and prefer candidates from the N-best list or the user lexicon (select over rewrite).
  - (c) *Insertion* or completion of a blocked word. Treat it as a proposal that needs high confidence or user confirmation.
  This maps the DRES "deletion-only" finding and the confidence and select-or-rewrite findings directly onto the product requirement.
- Reasoning-tuned small models (for example Phi-4-mini-reasoning, or Qwen3 in thinking mode) are the wrong choice for this stage because of their over-deletion bias. Instruct, non-thinking models, or a fine-tuned 1–3B model, fit better.
- Meaning-change risk concentrates in substitutions and insertions, not deletions. Logging and gating them separately gives an auditable over-correction rate.

### Gaps
- I did not retrieve the GenSEC 2024 challenge result numbers or the per-dataset HyPoradise WER tables.
- I found no paper titled "DisfluencySpeech" in my searches, and I could not verify its contents.
- STEAMROLLER's full paper, with WER, latency and safeguards, was not retrieved. Only the abstract was available.
- I found no published study of LLM correction specifically on *stuttered*, as opposed to dysarthric or Switchboard, transcripts with meaning-preservation metrics.

---

## Q4. Contextual biasing and personalization: user vocabulary, conversation history, speaker adaptation

### Takeaway
Biasing toward user phrases is mature. CLAS (2018) gave up to about 68% relative WER gains over shallow fusion and accepts out-of-vocabulary phrases at inference. TCPGen, phonetic-aware variants and 2025–26 SpeechLLM biasing methods (retrieval, logit-space, RL) extend it, and NVIDIA's TurboBias 2.0 (Aug 2026) targets streaming production. Conversation history helps but propagates earlier errors, so it must be used cautiously. Speaker-specific personalization is the biggest single lever for atypical speech: Euphonia personalized models reached about 4.6% WER against about 31% for speaker-independent models. Personalized fine-tuning also helps stuttered speech, most of all in spontaneous speech.

### Cited Findings
- **CLAS ("Deep context", Pundak et al., Google, SLT 2018) [FOUNDATIONAL].** The recognizer is trained jointly with embeddings of bias n-grams. It reaches a maximum relative WER gain of about **68%** over shallow fusion of LAS with n-grams, and accepts out-of-vocabulary bias phrases at inference. A follow-up, Phoebe, adds phonetic embeddings and reports a 16% relative WER reduction over CLAS. — [arXiv 1808.02480](https://arxiv.org/abs/1808.02480); search summary of [ar5iv](https://ar5iv.labs.arxiv.org/html/1808.02480)
- **Follow-ups to CLAS and TCPGen:**
  - Deep CLAS (iFLYTEK/USTC 2024): bias loss plus a richer bias-attention query. — [arXiv 2409.17603](https://arxiv.org/abs/2409.17603)
  - TCPGen, the Tree-Constrained Pointer Generator (Sun, Zhang, Woodland; Cambridge), for long-tail biasing words. — [arXiv 2205.09058](https://arxiv.org/abs/2205.09058)
  - TCPGen with GNN encodings. — [arXiv 2207.00857](https://arxiv.org/abs/2207.00857)
  - Phoneme-aware TCPGen. — [arXiv 2312.09582](https://arxiv.org/abs/2312.09582)
- **Biasing for SpeechLLMs and production (2025–2026):**
  - Hotword retrieval plus RL (Alibaba Tongyi, Dec 2025). — [arXiv 2512.21828](https://arxiv.org/abs/2512.21828)
  - LOGIC, logit-space integration with an explicit biasing weight (Microsoft, Jan 2026). — [arXiv 2601.15397](https://arxiv.org/abs/2601.15397)
  - RLBR (Jan 2026). — [arXiv 2601.13409](https://arxiv.org/abs/2601.13409)
  - BR-ASR, scalable bias retrieval (May 2025). — [arXiv 2505.19179](https://arxiv.org/abs/2505.19179)
  - TurboBias 2.0, streaming context-biasing "under strict latency constraints" (NVIDIA, Aug 2026). — [arXiv 2608.21343](https://arxiv.org/abs/2608.21343)
  - A comparison of context-biasing methods vs SpeechLLMs for new words (KIT/CMU, Aug 2026). — [arXiv 2608.05759](https://arxiv.org/abs/2608.05759)
  - Training-free error-aware selective retrieval (Hitachi, Sep 2026). — [arXiv 2609.30694](https://arxiv.org/abs/2609.30694)
  - Text injection for neural contextual biasing (Google 2024). — [arXiv 2406.02921](https://arxiv.org/abs/2406.02921)
- **Risks of conversation history:**
  - Contextual ASR is "trained with oracle conversation history, but relies on error-prone history at inference, causing a train-test mismatch" (contextual exposure bias, Mar 2026). — [arXiv 2603.24034](https://arxiv.org/abs/2603.24034)
  - "Relying on historical transcripts can propagate recognition errors and discard pronunciation" cues (Alibaba, Sep 2026). — [arXiv 2609.19765](https://arxiv.org/abs/2609.19765)
- **Project Euphonia (Green et al., Interspeech 2021).**
  - 432 speakers with speech disorders recorded at least 300 phrases each.
  - Personalized models reached about **4.6% WER**, against about **31%** for speaker-independent models, and outperformed human transcribers.
  - Google reports up to **85%** WER improvement in select domains over out-of-the-box models.
  — [Google Research blog](https://research.google/blog/personalized-asr-models-from-a-large-and-diverse-disordered-speech-dataset/); [PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12379580/)
  - Caveat: the 4.6%/31% figures come from a search summary of these pages. Verify them against the primary paper.
- **Data efficiency for personalization.** In one Euphonia-related study of 195 speakers (home-automation domain), **79%** of speakers reached the target WER with 18–20 minutes of speech and **63%** did with only 3–4 minutes. Treat the source as secondary; the attribution is unverified. — [catalyzex author page](https://www.catalyzex.com/author/Jimmy%20Tobin)
- **Median WER for short read disordered phrases** was about 5% (JSLHR 2024). — [JSLHR 2024](https://pubs.asha.org/doi/10.1044/2024_JSLHR-24-00045)
- **Stuttered speech.** Personalized ASR "significantly reduce[s] word error rates, especially in spontaneous speech", compared with generalized fine-tuning (MSU, Jun 2025). The abstract gives no numbers. — [arXiv 2506.00853](https://arxiv.org/abs/2506.00853)
- **Two-stage adaptation for non-normative speech** (speaker-independent initialization, then speaker-specific fine-tuning; Mar 2026). — [arXiv 2603.15261](https://arxiv.org/abs/2603.15261)
- **Personalization at the correction stage, without retraining.** Voice Memory keeps a per-domain memory file that a frozen corrector reads, with bounded, score-gated edits. It is auditable, portable, and adds no parameters to the inference path. — [arXiv 2607.26410](https://arxiv.org/abs/2607.26410)

### Inferences
- StutterClear's user-approved profile (name, frequent words, vocabulary) should be used in **two places**. Upstream, it feeds ASR biasing (a CLAS/TCPGen-style or SpeechLLM hotword list, or simple keyword boosting). Downstream, it serves as a *closed candidate set* for the corrector: substitutions are allowed mainly toward profile terms or N-best alternatives. A Voice-Memory-style per-user memory file is a natural, auditable format for this profile.
- Conversation history should be fed as *context for ranking candidates*, not as licence to rewrite. Given the documented error propagation, only committed and user-confirmed turns should go into history.
- Speaker-adapted ASR, Euphonia-style, probably reduces the correction stage's burden more than any text-side technique does. The text stage should therefore be built to be conservative on good ASR output.

### Gaps
- I did not retrieve TCPGen's headline numbers, or any measured latency overhead of contextual biasing in streaming ASR.
- I found no study combining stutter-specific ASR personalization with contextual biasing.

---

## Q5. Confidence estimation and abstention; intended-word prediction and word completion for people who stutter

### Takeaway
Word-level ASR confidence is a cheap, effective gate. Entropy-based confidence from Tsallis or Gibbs entropy, aggregated per word, steers LLM edits toward uncertain words. It kept harmful edits on high-confidence utterances to 1.1–5.0%. Abstention works at three levels:
- token level: the tagger's "wait" head;
- utterance level: Voice Memory's act-or-keep-1-best decision;
- word level: confidence filtering.
Real-time "complete the blocked word" research is thin. There are prototypes, a 2015 n-gram system with 87% next-word accuracy, a patent, and AAC LLM work. I found no validated real-time system for completing the intended word of a person who stutters.

### Cited Findings
- **Word-level confidence from end-to-end ASR.** Fast entropy-based methods (Laptev & Ginsburg, SLT 2023) compute frame-level Gibbs or Tsallis entropy and aggregate it to words by mean, min or product. Modern ASR is overconfident, so Tsallis entropy with α < 1 increases sensitivity to peaked distributions. — [Hernandez et al. 2025](https://arxiv.org/abs/2509.25048)
- **Confidence-guided correction behavior** (low- vs high-confidence utterances):
  - Attempt rate: SAP-shared 74.7% vs 20.3%; SAP-unshared 53.3% vs 18.5%; TORGO 30.9% vs 4.9%.
  - Harmful edits on high-confidence utterances: 1.7% (SAP-shared), 5.0% (SAP-unshared), 1.1% (TORGO).
  - Harmful edits on low-confidence SAP-unshared utterances reached 12.0%.
  — [Hernandez et al. 2025, Table 3](https://arxiv.org/abs/2509.25048)
- **Comparison of strategies.** The paper compares sentence-level filtering, word-level filtering, confidence prompting and naive correction. Word-level filtering at a 90% threshold was best on SAP-shared read speech (4.55%). Confidence prompting was best on spontaneous speech and on out-of-domain TORGO. — [Hernandez et al. 2025, Table 2](https://arxiv.org/abs/2509.25048)
- **Earlier confidence-plus-LLM work:** Naderi et al., Interspeech 2024, interfacing LLMs with ASR via confidence measures and prompting; Pu et al. 2024, multi-stage LLM correction. — cited in [Hernandez et al. 2025](https://arxiv.org/abs/2509.25048)
- **Learned abstention in a streaming tagger.** The wait head temporarily abstains, and the model waits most on interregna (1.06 tokens) and reparanda (0.76 tokens). — [Chen et al. 2022](https://arxiv.org/abs/2205.00620)
- **Utterance-level abstention.** Voice Memory's corrector chooses act or abstain per utterance, and no dataset regressed below 1-best. — [arXiv 2607.26410](https://arxiv.org/abs/2607.26410)
- **Intended-word and word-completion work for stuttering:**
  - **NITK 2015.** An n-gram LM system detects repetitions and predicts the next word, with **87%** prediction accuracy in 10-fold tests. The source says such systems previously existed only for Japanese. — [NITK repository](https://idr.nitk.ac.in/handle/123456789/32330)
  - **Fluent (2021).** A writing tool that flags words a PWS may struggle with, with over 80% accuracy, and suggests alternatives. It works on words individually, not in live speech. — [arXiv 2108.09918](https://arxiv.org/abs/2108.09918)
  - **Patent, "Time capsule based speaking aid".** When a stutter is detected, it predicts the intended word from preceding speech, a *personal language model* and the stutter sound, then displays or whispers it. This is a patent description, not a validated product. — [USPTO 10672393](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10672393); [Justia 20190043490](https://patents.justia.com/patent/20190043490)
  - **Continuous stuttering-onset forecasting** (ASU, Oct 2026) moves from detection to anticipation. — [arXiv 2610.19235](https://arxiv.org/abs/2610.19235)
  - **LLM intent and message prediction in text AAC for dysarthria** (Yusufali, Moore & Goetze, ICASSP 2024). — cited in [Hernandez et al. 2025](https://arxiv.org/abs/2509.25048)
  - **Personalized LLM AAC:** SocializeChat, grounded in personal memories (2025). — [arXiv 2510.19017](https://arxiv.org/abs/2510.19017)
  - **An autoethnography of ultra-personalized AI AAC** (2025). It notes that generic autocomplete "fails to capture the nuance of personal identity, requiring editing", which matters when output is high-stakes. — [arXiv 2509.13671](https://arxiv.org/abs/2509.13671)

### Inferences
- A practical gating policy for StutterClear has four parts:
  - (1) Commit deletions when the tagger is confident and not waiting.
  - (2) Allow substitutions only when the ASR word confidence is below a threshold *and* the replacement comes from the N-best list or the profile.
  - (3) Treat insertions or completions of a blocked word as *suggestions*. Show them, or require a confirm gesture, unless both the LM probability and profile or context support are very high.
  - (4) If the utterance-level uncertainty is high, keep the 1-best, or ask the user ("Did you mean …?") instead of guessing.
- An n-gram or small-LM next-word predictor conditioned on the profile and history can propose a blocked word, given the 87% next-word accuracy in a constrained 2015 setting. But accuracy will be far lower in open conversation, so confirmation is required.

### Gaps
- I found no peer-reviewed evaluation of real-time intended-word completion during stuttering blocks in open conversation, and no accepted metric for it.
- I found no study of LLM uncertainty measures (token entropy, self-consistency, verbalized confidence) applied to disfluency correction specifically. The evidence above uses ASR-side confidence.

---

## Q6. Metrics for correction quality

### Takeaway
Use metrics at four levels.
- Structural fidelity: word-level disfluency P/R/F1 ("E-scores") and span- and category-level Z-scores.
- Streaming quality: streaming F1, edit overhead (flicker), time-to-detection and first-time-to-detection, and average wait time.
- ASR-style metrics against **both** a verbatim and an intended reference, plus intent error rate.
- Explicit over-correction measures: help and harm rates, and the share of edits that break correct tokens.
Paraphrase-tolerant semantic metrics such as BERTScore, BLEURT and COMET are poorly suited to judging deletion-constrained fidelity.

### Cited Findings
- **E-scores and Z-scores.** E-scores are word-level precision, recall and F1 of deletions. In DRES terms, over-deletion lowers precision and under-deletion lowers recall. Z-scores (ICASSP 2025) measure performance per Shriberg category (EDITED, INTJ, PRN) at span level and use alignment to score generative outputs. — [DRES (2509.20321)](https://arxiv.org/abs/2509.20321); [Z-Scores (arXiv 2509.20319)](https://arxiv.org/abs/2509.20319)
- **Semantic metrics.** BERTScore, BLEURT and COMET "explicitly tolerate paraphrase and abstraction", while disfluency removal "demands strict structural fidelity". — [DRES §2.3](https://arxiv.org/abs/2509.20321)
- **Streaming metrics:**
  - Streaming F1: each token's prediction at each time step is scored.
  - Edit overhead (Buß & Schlangen 2011): unnecessary edits normalized by length, i.e. flicker.
  - Time-to-detection (Hough & Schlangen 2017).
  - First-time-to-detection from repair onset (Zwarts 2010).
  - Average waiting time.
  — [Chen et al. 2022, Appendix A.1](https://arxiv.org/abs/2205.00620)
- **Apple's PWS metrics:**
  - Standard WER.
  - Thresholded WER: the share of participants under 10% WER, which went from 48.8% to 65.9%.
  - Intent error rate for assistant commands: 10.4% → 5.4%.
  - Truncation rate.
  — [Lea et al. 2023](https://ar5iv.labs.arxiv.org/html/2302.09044)
- **Dual references.** Rankings change between verbatim and intended references, so choose the reference by use case. — [arXiv 2606.31112](https://arxiv.org/abs/2606.31112)
- **Over-correction metrics:**
  - Attempt, help and harm percentages stratified by confidence. — [Hernandez et al. 2025](https://arxiv.org/abs/2509.25048)
  - The share of a corrector's edits that break correct tokens (64% for unconstrained GER vs 35% for Voice Memory), and "no dataset regresses below 1-best". — [Voice Memory 2607.26410](https://arxiv.org/abs/2607.26410)
- **Semantic fidelity beyond WER for disordered-speech correction** (LLM-agent evaluation). — [arXiv 2601.21347](https://arxiv.org/abs/2601.21347)
- **Entity recall.** Entity and disfluency recall serve as complementary metrics to WER. — [arXiv 2609.20828](https://arxiv.org/abs/2609.20828)

### Inferences
- Recommended StutterClear evaluation suite:
  - (1) Disfluency E-F1 and Z-scores on stutter-annotated text.
  - (2) WER against the *intended* reference, which is the primary product metric, and against the *verbatim* reference, as a diagnostic.
  - (3) Over-correction rate: the share of output edits that change a correct intended word, split by deletion, substitution and insertion.
  - (4) Meaning preservation: an LLM judge or NLI entailment between the intended reference and the output, with BERTScore only as a secondary signal.
  - (5) Streaming metrics: edit overhead (flicker, which matters for TTS since spoken audio cannot be retracted), time to commit, and average wait.
  - (6) Abstention metrics: coverage against risk, i.e. the error rate on committed outputs at different abstention rates.

### Gaps
- I found no published benchmark that combines intended-reference WER, over-correction rate and an LLM-judge meaning score on stuttered speech. One would need to be built, for example from FluencyBank or SEP-28k with dual references.

---

## Q7. Latency: taggers vs small local LLMs, and incremental commit policies

### Takeaway
Small BERT taggers run in milliseconds. A 3.1M-parameter BERT_SV cuts runtime by about 80% against BERT-base, and a 1.3 MiB model keeps 88.4 F1. They fit easily in a per-token streaming loop. Local 1.5–3B LLMs on CPU decode at only about 5–12 tokens per second on modest hardware, according to secondary benchmarks, so free-form rewriting of every partial hypothesis is not feasible on CPU. A GPU, short constrained outputs (masks or edit tags) and KV-cache reuse are required. Commit policies such as LocalAgreement-2 (Whisper-Streaming, about 3.3 s latency) and the tagger's dynamic wait (0.21 tokens on average) are the established ways to handle partial results.

### Cited Findings
- **BERT_SV.** About 3.1M parameters against 108.9M for BERT-base, with about **80%** lower runtime latency. The streaming version waits **0.21 tokens** on average. — [Chen et al. 2022](https://arxiv.org/abs/2205.00620); [Rocholl et al. 2021](https://arxiv.org/abs/2104.10769)
- **Model size.** Disfluency detectors as small as **1.3 MiB** (88.4 F1) and 3.3 MiB (89.8 F1). — [Rocholl et al. 2021](https://arxiv.org/abs/2104.10769)
- **Restart-incremental cost.** Re-running a model on every prefix lets it revise earlier predictions, but it costs recomputation. Future work was suggested to avoid re-computing all previous inputs. — [Chen et al. 2022](https://arxiv.org/abs/2205.00620)
- **Whisper-Streaming with LocalAgreement-2** (Macháček, Dabre & Bojar, 2023). It commits the longest common prefix of consecutive hypotheses as final and keeps the rest as tentative. It reports about **3.3 s** latency on unsegmented long-form speech. — [arXiv 2307.14743](https://arxiv.org/abs/2307.14743); [DeepAI listing](https://deepai.org/publication/turning-whisper-into-real-time-transcription-system)
- **WhisperKit (on-device, 2025).** It applies the Local Agreement policy to produce separate *confirmed* and *hypothesis* text streams. — [arXiv 2507.10860](https://arxiv.org/abs/2507.10860)
- **Small-LLM throughput.** These come from secondary, non-peer-reviewed benchmarks; treat them as indicative only.
  - Qwen2.5-3B at Q4_K_M on an i7-6700 with 4 threads (llama.cpp): **43 tok/s prompt processing** (pp512) and **11.8 tok/s generation** (tg128). On an RTX 4090, about 327 tok/s generation. — [edge-cpu-inference (GitHub)](https://github.com/nandan2003/edge-cpu-inference)
  - Raspberry Pi 5 at Q4_K_M: Qwen2.5-1.5B about **9.97 tok/s**, 3B about **5.2 tok/s**. — [Stratosphere RPi LLM study](https://stratospherelinuxips.readthedocs.io/en/develop/immune/research_rpi_llm_performance.html)
  - Snapdragon 865 at Q8: Qwen2.5-1.5B about 12.6 tok/s, but this figure is an *estimate*, not a measurement. — [HF dataset dispatchAI/on-device-latency](https://huggingface.co/datasets/dispatchAI/on-device-latency)
- **DRES model recommendations.** Qwen3-1.7B-class "under-deletion" models for edge real-time use, and 8B with few-shot prompting as the latency–accuracy compromise. — [DRES Table 3](https://arxiv.org/abs/2509.20321)
- **Latency already present upstream.** Endpointer tuning for PWS adds about 1.2–1.7 s of median delay before end-of-utterance. — [Lea et al. 2023](https://ar5iv.labs.arxiv.org/html/2302.09044)

### Inferences
These are back-of-envelope estimates from the figures above, not measured results.
- **CPU, 3B model.** A 3B model at about 12 tok/s generation needs about 1.7 s to regenerate a 20-token sentence. At 43 tok/s prompt processing, a 300-token prompt (profile, history and hypothesis) takes about 7 s of prefill unless the static prefix is KV-cached. This is not viable per partial update.
- **GPU or NPU, or a 0.5–1.5B model.** These plus prefix caching bring per-update latency to roughly 100–300 ms, which suits *final-segment* correction, not per-token correction.
- **Recommended latency architecture:**
  - (1) A tagger runs on every ASR partial (milliseconds), with a wait head and a "commit after N stable tokens" rule, i.e. LocalAgreement over successive ASR hypotheses *and* over tagger outputs.
  - (2) The LLM is invoked only on committed clauses that contain low-confidence words or detected blocks, using short constrained outputs: a deletion mask, N-best index choices, or a few substitution edits. Each output needs about 5–15 tokens.
  - (3) TTS receives only committed text, since spoken audio cannot be un-said, so flicker and edit overhead must be about zero at the TTS boundary.
- Commit granularity should be about a clause or phrase, not a word. Repairs such as "I want the — I need the blue one" are resolved within about 1–2 tokens after the repair onset, per Chen et al.'s wait statistics. Blocks may need a time-based timeout tied to the stutter detector instead of token lookahead.

### Gaps
- I found no peer-reviewed, apples-to-apples measurement of time-to-first-token for 0.5–4B models on target consumer hardware for this task. Run `llama-bench` or an equivalent on the actual device.
- I found no measured per-update wall-clock latency for BERT_SV-class streaming taggers on CPU or mobile. Only relative figures (−80% vs BERT-base) and model sizes are published.
- I found no published study of LocalAgreement-style commit policies applied to the *disfluency correction* output, as opposed to the ASR output.
