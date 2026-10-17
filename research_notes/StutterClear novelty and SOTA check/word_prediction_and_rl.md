# StutterClear novelty check: (A) blocked-word prediction and (B) RL streaming cut/hold/play policies (as of 17 Oct 2026)

## A1. Has anyone built a system that predicts the word a person who stutters is blocked on? (Verdict for Plan A)

### Takeaway
**Verdict: PARTIALLY DONE.** The core concept is already published as a patent. Intel's "Time capsule based speaking aid" (US 2019/0043490 A1, filed 12 Jan 2018; granted as US 10,672,393 B2 on 2 Jun 2020) detects a stutter or pause and predicts the intended word. Its inputs are the preceding words, a personal language model, context metadata (conversation partner, topic, location) and the **stutter sound**. It returns an n-best list, shown on a display or whispered into an earpiece, and the user says "next" to see another candidate. NITK (2015) built an n-gram version that speaks the predicted word aloud, with 87% accuracy. The parts that still look open are: an LLM conditioned on a context card the user types in; a real acoustic onset-phoneme classifier used as an explicit constraint and evaluated as one; a **visual (lip) cue** for the stuck sound; voice-cloned output; and any evaluation on real blocks. None of the prior systems reports results on real stuttered blocks.

### Cited Findings
- **Intel patent, bibliographic details.** US20190043490A1 (published 7 Feb 2019), application 15/869,899, filed and priority 12 Jan 2018, granted as US10672393B2 on 2 Jun 2020, assignee Intel Corp. Inventors: Ze'ev Rivlin, Vered Bar Bracha, Douglas Gabel, Jonathan Huang, Sylvia Downing, Binuraj Ravindran, Willem Beltman. Google Patents lists it as Active. — [Google Patents US20190043490A1](https://patents.google.com/patent/US20190043490A1/en); [Google Patents US10672393B2](https://patents.google.com/patent/US10672393B2/en)
- **Intel patent, how it works.** Its abstract: the system should "detect a stutter or pause in the speech input, predict the stuttered word or the missing word".
  - It detects a stutter when "the user repeats a portion of a word several times" or uses filler sounds, and a pause when a sentence is left unfinished for several seconds.
  - The prediction engine takes the words before the gap, a personal language model and the stutter sound. Location and activity can also be inputs. The engine may be a neural network, an HMM or an n-gram/statistical LM.
  - It outputs a prioritized n-best list with probabilities, e.g. for "ba-ba-ba": basketball 0.9, baseball 0.8, backgammon 0.4, bowling 0.2, bocce 0.1. It predicts one word at a time. — [US10672393B2](https://patents.google.com/patent/US10672393B2/en)
- **Intel patent, the stutter sound is a soft cue, not a hard filter.** The patent says "Although bowling and bocce do not begin with the exact stutter sound" they can still be candidates, and "the stutter sound may not be an exact match to a word". — [US10672393B2](https://patents.google.com/patent/US10672393B2/en); [US20190043490A1](https://patents.google.com/patent/US20190043490A1/en)
- **Intel patent, personalization and delivery.**
  - The personal LM ("time capsule corpus") is built over time from recorded conversations, texts, emails and notes. Recordings are tagged with who the user talked to, topic, location and activity.
  - Candidates are whispered into an earpiece or shown on smart eyewear or a nearby screen. The user advances with the word "next" or a gesture. — [US10672393B2](https://patents.google.com/patent/US10672393B2/en)
- **Intel patent, independent claims.** Claims 1, 7, 13 and 20 recite: receiving speech with a partial sentence; detecting a stutter or pause; predicting the word "using a personal language model for the user"; and presenting n-best candidates until the user says the right word. Claim 1 also limits the personal LM to one "generated in a continuous manner over time until the user experiences cognitive deterioration".
  - The patent reports **no experimental results or accuracy figures**.
  - It does **not** describe synthesizing the word in the user's voice.
  - A pass over the claims and later sections found no use of camera, lip or face images of the speaker.
  - An earlier pass over the first part of the text summarized it as mentioning smart eyewear with a camera that can identify people and objects. These two reads disagree. Either way, nothing reads the speaker's lips to find the stuck sound. — [US10672393B2](https://patents.google.com/patent/US10672393B2/en)
- **NITK 2015.** "Language modelling and English speech prediction system to aid people with stuttering disorder", by T.L. Chandana, P.S. Kalwad, S. Pattanaik and G. Ram Mohana Reddy (ACM conference, 2015).
  - The system treats repeated words as stuttering, drops the repetition, predicts the next word with an n-gram LM and **converts the prediction to speech**.
  - It reports 87% prediction accuracy under 10-fold cross-validation.
  - The authors say it is the first such system for English; an earlier Japanese system used a vocabulary tree. — [NITK IR](https://idr.nitk.ac.in/items/d8d1c176-b274-43c8-84cc-eeb09232b353/full)
- **Fluent (Ghai & Mueller, ASSETS '21, 2021).** An offline writing tool, not a live one.
  - It flags words a particular user may find hard to say, using a per-user SVM over phonetic embeddings from the CMU Pronouncing Dictionary, and suggests easier synonyms.
  - It reports over 80% mean accuracy in under 20 interactions, but with 10 *simulated* users. — [ACM DL](https://dl.acm.org/doi/10.1145/3441852.3471211); [arXiv 2108.09918](https://arxiv.org/pdf/2108.09918); [Stony Brook](https://researchconnect.stonybrook.edu/en/publications/fluent-an-ai-augmented-writing-tool-for-people-who-stutter/)
- **Forecasting blocks (2026).** An arXiv preprint (2604.27279) predicts upcoming stuttering events from 3 s of audio. It reports that the predictive signal is concentrated on blocks and sound repetitions, and that the model runs fully on-device. Seen only via the search abstract, not verified in full. — [arXiv 2604.27279](https://arxiv.org/pdf/2604.27279)
- **ASR on stuttered speech.**
  - For more severe stuttering, ASR may "insert unintended words or even truncate the speech due to a blocking stutter". This is a risk for the "running transcript" input. — [arXiv 2302.09044](https://arxiv.org/html/2302.09044v3)
  - Apple's ASR work for dysfluent speech raises the LM weight relative to the acoustic model to suppress extraneous words from part-word repetitions. — [Apple ML Research](https://machinelearning.apple.com/research/speech-recognition)
- **Recent search (2025–2026).**
  - Found no peer-reviewed live LLM system that predicts blocked words for people who stutter.
  - Found only hobby or proof-of-concept "speech companion" projects that offer LLM suggestions when a speaker hesitates. Unverified; the search pointed to a Medium post. — [Medium](https://medium.com/@algoforge.ai/real-time-smart-speech-assistant-with-python-whisper-llms-cb2740e1bb89)
  - Coaching apps (Eloquent, BeneTalk) do practice and pacing, not live word prediction. — [Eloquent](https://iyaso.ai/); [BeneTalk](https://apps.apple.com/us/app/benetalk-stuttering-coach/id1550997118)

### Inferences
- A paper or pitch cannot claim "first system to predict the word a person who stutters is blocked on". Intel's 2018 filing already combines a personal or contextual LM, preceding words, the stutter sound and n-best candidates delivered to the user.
- What does look new:
  - (i) A modern LLM conditioned on a short **user-authored context card** instead of a passively recorded lifelong corpus.
  - (ii) Using the onset phoneme as an explicit, measured constraint (a phoneme classifier plus constrained or reweighted decoding), with an ablation showing its gain.
  - (iii) A **visual onset cue** (lip closure → bilabial /p b m/).
  - (iv) Output in the user's own cloned voice.
  - (v) **Any quantitative evaluation on real blocks**, e.g. top-1/top-3 accuracy on FluencyBank or UCLASS blocks with versus without context and onset.
- This is not legal advice, but there is a possible claim-scope distinction. Intel's claim 1 requires a personal LM "generated in a continuous manner over time until ... cognitive deterioration"; a context card typed in by the user may not match that language. The patent is active (granted 2020), so a commercial version should get a proper freedom-to-operate review.
- NITK's "predict, then speak it" pipeline already exists, so voice output is only new if it is in the user's **own** voice and is evaluated for acceptability.

### Gaps
- I could not fetch the full text of the NITK 2015 paper, so the details of its "constrained tests" (vocabulary size, test set) are unverified.
- I did not exhaustively search Google Patents for 2021–2026 filings on stutter word prediction. Others may exist, e.g. from smart-glasses or hearing-aid vendors.
- I did not check whether Intel's DE priority family (DE102018131935) has different claims.

## A2. AAC word prediction with context and personal vocabulary: what are the numbers?

### Takeaway
LLM-based AAC prediction conditioned on conversation context is well established, mostly from Google's SpeakFaster line. One turn of context more than doubles abbreviation-expansion accuracy, and LLM expansion saves 57% more motor actions than predictive keyboards. Personal-memory and profile-grounded AAC with LLMs appeared in 2025, but mostly as qualitative studies. None of this work targets speech blocks or combines context with an acoustic onset.

### Cited Findings
- **SpeakFaster** (Cai et al., *Nature Communications*, published 1 Nov 2024; arXiv 2312.01532, Dec 2023; Google with the Team Gleason Foundation).
  - Users type phrases abbreviated mainly to **word-initial letters**, and an LLM expands them using conversation context.
  - In offline simulation it saved **57% more motor actions** than traditional predictive keyboards.
  - A pilot with 19 non-AAC users matched the simulated savings.
  - Two eye-gaze typists with ALS had text-entry rates **29–60% above baseline**. — [Nature Comms / PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11530652/); [Google Research](https://research.google/pubs/using-large-language-models-to-accelerate-communication-for-eye-gaze-typing-users-with-als/)
- **Context-aware abbreviation expansion** (Cai et al., NAACL 2022). Adding a single prior conversation turn **more than doubles** expansion accuracy compared with no context. A 64B LLM expanded more than 70% of phrases, with keystroke savings up to 77% on those phrases. — [ACL Anthology](https://aclanthology.org/2022.naacl-main.91.pdf); [arXiv 2205.03767](https://arxiv.org/pdf/2205.03767)
- **Character-level LLM adaptation for AAC** (arXiv 2501.10582, Jan 2025, Michigan Tech). The best adapted model gives a **+5.5% absolute keystroke-savings** gain over an AAC n-gram baseline, and keystroke savings track perplexity. — [arXiv 2501.10582](https://arxiv.org/pdf/2501.10582)
- **SocializeChat** (IEEE SMC 2025; arXiv 2510.19017, 21 Oct 2025). A GPT-based AAC tool that generates sentence suggestions from a user's **personal memory records**, weighted by topic preference and closeness to the conversation partner. The abstract reports a user study but no quantitative numbers. — [arXiv 2510.19017](https://arxiv.org/abs/2510.19017)
- **Other 2023–2026 LLM AAC work on context and identity.**
  - "The less I type, the better" (Valencia et al., CHI 2023) is a qualitative study of how LLMs can enhance or impede AAC users. — [ACM DL](https://dl.acm.org/doi/full/10.1145/3544548.3581560)
  - Speak Ease (arXiv 2503.17479, Mar 2025) combines multimodal input, including text, voice and contextual cues, with LLMs and speech generation. — [alphaXiv 2503.17479](https://www.alphaxiv.org/abs/2503.17479)
  - "I, Robot? Ultra-personalized AI-powered AAC" (arXiv 2509.13671, Sep 2025) is an autoethnography reporting that generic AI autocomplete fails to capture personal identity. — [alphaXiv 2509.13671](https://www.alphaxiv.org/abs/2509.13671)
- **Pre-LLM context modeling** (CHI 2020 design-engineering study). It modeled theoretical keystroke savings as a function of word-prediction accuracy and the accuracy of sensed context, and found substantial potential savings under reasonable assumptions. — [ACM DL](https://dx.doi.org/10.1145/3313831.3376525)

### Inferences
- SpeakFaster's interface (word-initial *letters* plus conversation context, expanded by an LLM) is the closest structural analogue to Plan A (word-initial *phoneme* plus a context card). The cue simply moves from typed letters to the onset sound. Cite SpeakFaster as the methodological precedent and the source of the context-doubles-accuracy evidence.
- The NAACL 2022 result (one context turn more than doubles accuracy) suggests that both the context card and the running transcript are likely to give large gains. Plan A should report an ablation on context.

### Gaps
- I found no published top-k next-*word* accuracy for LLMs conditioned on a user profile or context card in spoken (not typed) settings.
- I did not fetch full numbers for SocializeChat or Speak Ease.

## A3. Has anyone used the onset phoneme (partial acoustic input) or a lip/visual cue of the stuck sound to constrain word prediction?

### Takeaway
The building blocks exist separately: "run-ahead" speech recognition, prefix/trie biasing on partial words, phoneme-level biasing, lip-based phoneme articulation scoring, and audio-visual block detection. Intel's patent feeds the stutter sound in as a soft cue. I found **no work that reads the speaker's lips to identify the sound they are blocked on** and uses it to constrain prediction. **Verdict for the visual-onset piece: OPEN.** For the acoustic onset: **PARTIALLY** done (Intel's soft cue, with no evaluation).

### Cited Findings
- **Speculative Speech Recognition** (Yusuf, Baskar, Rosenberg, Ramabhadran; Google; Interspeech 2024; arXiv 2407.04641, Jul 2024). An RNN-T is paired with an audio-prefixed LM that predicts likely continuations *before* the audio is heard, to cut ASR latency. The abstract gives no figures. — [arXiv 2407.04641](https://arxiv.org/abs/2407.04641)
- **Predictive speech recognition and end-of-utterance detection** (arXiv 2409.19990, Sep 2024, Waseda/KIT). Predicts upcoming words for spoken dialogue timing. — [alphaXiv 2409.19990](https://www.alphaxiv.org/abs/2409.19990)
- **Prefix and phoneme biasing in ASR.**
  - Trie-based contextual biasing gives "bonus scores" to partial word prefixes (arXiv 2509.09196, Sep 2025). — [alphaXiv 2509.09196](https://www.alphaxiv.org/abs/2509.09196)
  - PTC-Bias uses phoneme-level temporal competition for bias retrieval (arXiv 2609.28727, Sep 2026). — [alphaXiv 2609.28727](https://www.alphaxiv.org/abs/2609.28727)
  - Phoneme-First Prediction has the LLM emit phonemes before words (KU Leuven, arXiv 2606.10864, Jun 2026). — [alphaXiv 2606.10864](https://www.alphaxiv.org/abs/2606.10864)
- **Intel patent.** The stutter sound is an input to the predictor, but candidates are not filtered by it: "bowling" and "bocce" still appear for "ba-ba-ba". — [US10672393B2](https://patents.google.com/patent/US10672393B2/en)
- **Phonemic cueing in aphasia therapy** works in the *opposite* direction: the clinician or app gives the first sound to help retrieval. Many people with aphasia can retrieve a word when they only see the mouth of someone making the first sound. — [Tactus Therapy](https://tactustherapy.com/cueing-hierarchy-word-finding-aphasia/); [Aphasiology 2019](https://www.tandfonline.com/doi/full/10.1080/02687038.2019.1686748)
- **Visual cues in stuttering.**
  - Altinkaya & Smeulders (ACM MM workshop, 2020) built an audio-visual stuttered-speech database (25 speakers) and a ResNet-GRU audio+video model for **block detection**, not phoneme identity. — [UvA](https://ivi.fnwi.uva.nl/vislab/publication/mehmet-mucai-2020/); [review arXiv 2107.04057](https://arxiv.org/pdf/2107.04057)
  - The review notes visual signs such as lip tremors, unusual lip shapes, head nodding and quick eye blinks. — [arXiv 2107.04057](https://arxiv.org/pdf/2107.04057)
  - MMSD-Net (arXiv 2407.11492, 2024) does multimodal stuttering *detection*. — [arXiv 2407.11492](https://arxiv.org/pdf/2407.11492)
- **Lip tools that could be reused.** PVSync (arXiv 2610.09223, Oct 2026) does phoneme-level articulation scoring from lips. — [alphaXiv 2610.09223](https://www.alphaxiv.org/abs/2610.09223)

### Inferences
- A defensible novelty claim for Plan A: "first to use the speaker's own onset articulation (acoustic and/or visual, e.g. lip closure → {p, b, m}) as an explicit constraint on LLM next-word prediction during stuttering blocks, with measured top-k gains". The comparison should be against Intel's soft-cue design and against context-only prediction.
- A visual cue gives a viseme class, not a phoneme: /p b m/ share lip closure. It is therefore a natural *soft* class constraint, which fits LLM reranking well.

### Gaps
- No numbers found for how much an onset constraint improves top-k word prediction. This is a key experiment for StutterClear to run itself.
- I did not search the lip-reading literature for "silent block" or prolongation articulator states specifically.

## B1. RL for streaming read/write (wait/emit) policies in speech: closest prior work and numbers

### Takeaway
**RL or GRPO for "when to emit vs. wait" in streaming speech is ALREADY DONE and very active.** It started with simultaneous MT in 2014–2017. Recent work covers simultaneous speech-to-speech translation (Seed LiveInterpret 2.0, Jul 2025; Hibiki-Zero, Feb 2026, ICML 2026), long-form simultaneous speech translation (HPO, Apr 2026) and, in Sep 2026, **streaming ASR commit policies** (Look Less Hear Better; X2Streaming-ASR). Reported latency gains are roughly 30–70%, with quality held roughly constant.

### Cited Findings
- **Grissom II et al., EMNLP 2014**, "Don't Until the Final Verb Wait". An RL policy decides when to trust a *prediction* of the unseen final verb and when to keep waiting, and beats both batch and monotone strategies. — [ACL Anthology D14-1140](https://aclanthology.org/D14-1140/)
- **Gu et al., 2017** (Learning to Translate in Real-time). An RL agent chooses READ or WRITE, with quality and latency rewards combined by additive weighting. — [arXiv 1610.00388](https://arxiv.org/abs/1610.00388); described in [HPO arXiv 2604.21045](https://arxiv.org/html/2604.21045v1)
- **SeqPO-SiMT** (arXiv 2505.20622, May 2025). Reportedly frames simultaneous MT as a sequential decision problem, with GRPO-style normalization, a COMET reward, a clamped Average Lagging penalty and a KL term. These details come from an AI-generated summary page and are not verified. — [Pith summary](https://pith.science/paper/2505.20622)
- **Seed LiveInterpret 2.0** (ByteDance, arXiv 2507.17527, Jul 2025).
  - Its RL framework combines fine-grained stepwise feedback with sequence-level feedback.
  - It cuts the average latency of cloned speech from **nearly 10 s to about 3 s (~70%)**, with more than 70% correctness judged by human interpreters.
  - The company blog instead says 2–3 s, "over 60% lower" than traditional systems; the baselines differ. — [arXiv 2507.17527](https://arxiv.org/html/2507.17527v2); [Seed blog](https://seed.bytedance.com/en/blog/seed-liveinterpret-2-0-released-an-end-to-end-simultaneous-interpretation-model-featuring-ultra-high-accuracy-close-to-human-interpreters-low-latency-of-3-seconds-and-real-time-voice-cloning)
- **Hibiki-Zero** (Labiausse, Fabre, Estève, Défossez, Zeghidour; Kyutai; arXiv 2602.11072, Feb 2026; ICML 2026).
  - GRPO with **BLEU-only process rewards**: (1−α)·BLEU(prefix) + α·BLEU(full), with α = 0.4. There is no explicit latency term; LAAL is only monitored.
  - Training settings: group size G = 4, 2,000 updates, no KL term.
  - Comparing the fine-tuned model with the final model (I read "Finetuned" as the pre-RL model):
    - French short-form: end offset 3.5 → 2.1 s, LAAL 4.0 → 2.8 s, ASR-BLEU 33.9 → 34.6.
    - French long-form: end offset 4.8 → 2.3 s, LAAL 6.9 → 6.1 s, ASR-BLEU 29.0 → 28.7.
    - Long-form ASR-BLEU drops by about 0.3–1.1 points.
  - Stated limits: the trade-off is fixed at inference, and **RL failed when started from a full-sentence-delay model** (latency stayed around 6 s). — [arXiv 2602.11072](https://arxiv.org/html/2602.11072); [ICML 2026](https://icml.cc/virtual/2026/poster/66094)
- **HPO** (Ouyang, Ding, Hrinchuk, Lavrukhin, Ginsburg, Yan, Li; CMU and NVIDIA; arXiv 2604.21045, 22 Apr 2026). A GRPO variant for simultaneous translation of unbounded speech.
  - Quality is scored with MetricX, latency with LAAL.
  - When quality falls below a threshold (MetricX −5), the latency score is set to its maximum of 10 s. This hierarchical "quality gate" stops the policy trading accuracy for speed.
  - Reward is q̄ − 0.5·l̄.
  - At about 1.5 s latency it gains up to +7 COMET, 1.25 MetricX and 4 BLEURT over InfiniSST SFT. — [arXiv 2604.21045](https://arxiv.org/html/2604.21045v1)
- **Look Less, Hear Better** (Xiuwen Zheng, UIUC; arXiv 2609.18333, 16 Sep 2026). GRPO post-training of a Delayed-Streams-Modeling streaming ASR model, with a joint accuracy and word-emission-delay reward.
  - WER falls by **30.8% relative at 80 ms** structural delay and 5.7% at 480 ms.
  - Median word emission delay falls from 1.17 to 1.04 s. — [arXiv 2609.18333](https://arxiv.org/abs/2609.18333)
- **X2Streaming-ASR** (Lin et al.; arXiv 2609.08672, Sep 2026, revised 12 Oct 2026). Splits the problem into *when to commit* and *what to commit*.
  - The commit policy is **initialized from automatically probed trajectories, then refined with group-relative accuracy and latency rewards**.
  - Mean commit latency relative to forced-aligned endpoints is 32–109 ms on Chinese characters and 12–85 ms on English words, with comparable accuracy. — [arXiv 2609.08672](https://arxiv.org/abs/2609.08672)
- **NaturalFlow** (arXiv 2606.13121, Jun 2026) targets disruptive pauses in simultaneous speech-to-speech output. This is relevant because "hold" decisions create audible gaps. Method details not checked. — [alphaXiv 2606.13121](https://www.alphaxiv.org/abs/2606.13121)

### Inferences
- The closest analogue to StutterClear's {cut, hold, play} policy is the **streaming-ASR commit policy** (X2Streaming-ASR, Look Less Hear Better) rather than speech-to-speech translation. Both make per-frame decisions under a delay budget, with an accuracy reward computed against aligned references. StutterClear's silver frame labels play the role of the forced alignments.
- There is a practical lesson from both Hibiki-Zero and X2Streaming-ASR. RL works best **starting from a reasonable supervised or heuristic policy**: X2Streaming initializes from probed trajectories, and Hibiki-Zero's RL failed from a high-latency start. StutterClear should behavior-clone its current hysteresis/merge/snap rule first, then fine-tune with RL.

### Gaps
- I did not verify SeqPO-SiMT's numbers from the primary paper.
- I did not extract Look Less Hear Better's exact reward weights or action space.

## B2. RL for endpointing, turn-taking and speech editing, and anything RL applied to stuttering or disfluency removal

### Takeaway
RL for turn-taking exists (IJCAI 2016; DuplexPO, Jul 2026). RL for speech *content* editing exists (GRPO with a TTS consistency critic, early 2026). In stuttering, RL appears only as (a) an unspecified option in an MIT patent for switching altered-auditory-feedback modes and (b) a simulated medication-dosing proof of concept. **I found no RL policy for real-time removal or editing of stutters or disfluencies. Verdict for Plan B in stuttering: OPEN.** The generic technique is ALREADY DONE in adjacent streaming tasks (B1).

### Cited Findings
- **Khouzaimi, Laroche & Lefèvre, IJCAI 2016.** An RL policy decides when the dialogue system takes the floor. Compared with a handcrafted silence threshold it **cut response latency** and produced fewer premature cut-ins (not statistically significant). The authors note that lower latency did not automatically yield better dialogues. — [IJCAI 2016](https://www.ijcai.org/Proceedings/16/Papers/402.pdf)
- **DuplexPO** (arXiv 2607.07148, Jul 2026). RL for full-duplex spoken models with a factorized conversational-dynamics reward covering turn initiation, backchanneling, yielding and participation. It reports turn latency against a voiced-interrupt rate; for example, Moshi has latency 0.44 but a voiced-interrupt rate of 12.0. — [arXiv 2607.07148](https://arxiv.org/html/2607.07148)
- **Streaming endpointer** (arXiv 2506.07081, 2025). Not RL. It frames endpointing as a latency-versus-cutoff trade-off, measured by median and 90th-percentile latency (ep50/ep90) and the rate of premature cutoffs. Label-delayed training reduces cutoffs during mid-utterance pauses. — [arXiv 2506.07081](https://arxiv.org/html/2506.07081v1)
- **RL for speech editing** (arXiv 2602.00560, early 2026). GRPO-style RL for *text-based* speech content editing, using a pretrained TTS model as a consistency critic plus WER and duration constraints. It does not address disfluency or stutter removal. — [arXiv 2602.00560](https://arxiv.org/pdf/2602.00560)
- **MIT patent US11727949B2**, "Methods and apparatus for reducing stuttering". Kleinberger, Erkkinen, Stefanakis, van Troyer, Ghosh, Baker, Machover; priority 12 Aug 2019, granted 15 Aug 2023; assignee MIT, with Brigham and Women's Hospital also listed.
  - It provides real-time altered auditory feedback in modes such as Whisper, Reverb, Harmony, DAF (50–200 ms) and pitch shift.
  - It switches mode when measured stuttering rises, which it treats as a sign of habituation.
  - It says "the machine learning model that performs this task is a reinforcement learning algorithm", naming Monte Carlo, Q-learning and DQN. **No reward, state or results are given for RL.** — [Google Patents](https://patents.google.com/patent/US11727949B2/en)
- **RL medication dosing for disfluency** (Constas et al.; ML4CMH workshop at AAAI 2024; arXiv 2312.11509, Dec 2023). An RL search over psychiatric medication combinations against a *simulated* patient, with a separate disfluency detection and scoring module. Results are qualitative: it can converge "under some circumstances". — [arXiv 2312.11509](https://arxiv.org/abs/2312.11509)
- **Targeted search** for "reinforcement learning disfluency removal / stutter speech editing real-time policy" returned only the two items above. — [search context: arXiv 2312.11509](https://arxiv.org/html/2312.11509v4); [arXiv 2602.00560](https://arxiv.org/pdf/2602.00560)

### Inferences
- Plan B's novelty is in the **application and objective**, not the algorithm. Three things set it apart:
  - An asymmetric reward where wrongly cutting fluent words is catastrophic, while leaving a stutter in is a mild failure.
  - A three-way action space where "hold" adds delay without committing to a cut.
  - Training in a simulator built from silver frame labels.
- Frame it as "simultaneous read/write policy learning applied to streaming disfluency removal", citing Grissom 2014, Gu 2017, Hibiki-Zero 2026, HPO 2026 and the streaming-ASR GRPO papers as methodological precedents.
- The endpointing literature's metrics (ep50/ep90 latency, premature-cutoff rate) map directly onto StutterClear's "delay" and "fluent word cut" terms and could serve as evaluation conventions.

### Gaps
- I did not search non-English venues or industry blogs (e.g. hearing-aid or voice-call vendors) for proprietary RL-based disfluency cutters.
- I did not read the rest of the MIT patent text beyond the first ~100k characters.

## B3. RL or bandits for personalization in assistive speech and hearing technology, and constrained RL for latency–quality trade-offs

### Takeaway
Human-in-the-loop RL and bandits are established for **hearing-aid personalization** (2020–2022). This is useful precedent for per-user tuning of StutterClear's aggressiveness. For latency and quality in streaming speech, published systems use **weighted or gated rewards rather than formal constrained RL**. HPO's quality gate is the closest; I found no Lagrangian or CMDP formulation. **Verdict: constrained-RL framing for streaming speech editing is OPEN (PARTIALLY addressed by gated rewards).**

### Cited Findings
- **Bandit hearing-aid fitting** (University of Iowa; *Smart Health*, 2022). Hearing-aid fitting is cast as a multi-armed bandit over presets with pairwise-comparison feedback, in a 35-user study. Simulations found the best configuration in a median of 25 comparisons, about half as many as the best baseline. — [Iowa Research Online](https://iro.uiowa.edu/esploro/outputs/journalArticle/Personalizing-over-the-counter-hearing-aids-using-pairwise/9984258850902771)
- **Human-in-the-loop deep RL for hearing-aid compression** (UT Dallas; IEEE Access, 2020; arXiv 2007.00192). Learns an individual user's preferences from their feedback and tunes compression to them. — [arXiv 2007.00192](https://arxiv.org/pdf/2007.00192)
- **Inverse-RL follow-up** from the same group. In a 10-subject test, personalized settings were reportedly preferred about 10× more often than the standard prescriptive settings. Taken from the search summary; not verified in the full paper. — [PMC10312409](https://pmc.ncbi.nlm.nih.gov/articles/PMC10312409)
- **Constrained-style designs in simultaneous speech translation.**
  - HPO: the latency reward is forced to its worst value when quality falls below a threshold, and quality and latency are normalized separately and combined as q̄ − 0.5·l̄. The paper does **not** use a constraint budget or Lagrangian. — [arXiv 2604.21045](https://arxiv.org/html/2604.21045v1)
  - Hibiki-Zero uses no latency term at all; its latency trade-off is fixed after RL and cannot be adjusted at inference. — [arXiv 2602.11072](https://arxiv.org/html/2602.11072)
  - SeqPO-SiMT reportedly uses a clamped latency penalty (unverified). — [Pith summary](https://pith.science/paper/2505.20622)

### Inferences
- A constrained formulation fits StutterClear's safety requirement better than a single weighted reward used by most prior work: maximize stutter removed − λ·delay, subject to P(fluent word cut) ≤ ε, solved with Lagrangian PPO or GRPO. It also allows the operating point to be set per user. Lagrangian or CMDP RL is established in general RL, but I found no application in streaming speech. I did not search the general constrained-RL literature, so that statement is not cited here.
- Hibiki-Zero notes the trade-off cannot be adjusted at inference. Conditioning the policy on a user-set aggressiveness or ε value would be a concrete, citable improvement over it.
- Per-user tuning of the cut threshold or ε from accept/undo feedback could follow the hearing-aid bandit precedent.

### Gaps
- I found no published Lagrangian or constrained RL for streaming speech (ASR, translation or editing). This is "not found in my searches", not a confirmed absence; a dedicated search on "constrained policy optimization simultaneous translation latency budget" is advisable before claiming novelty.
- I found no RL or bandit personalization for stuttering apps specifically, e.g. DAF parameter tuning, beyond the MIT patent's unspecified RL option.
