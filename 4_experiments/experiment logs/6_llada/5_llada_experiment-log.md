## Run 5 — Mar 12, 2026

**Model:** LLaDA 8B Instruct (4-bit quantized unless noted)
**Phase:** Base model diagnostics — isolating the degeneration cause
**GPU:** A100 (40GB) for 5a-5e baselines. RTX PRO 6000 Blackwell (96GB) for 5e fine-tuning.
**Baseline config:** FULL — 128 steps, 1024 gen tokens, block_length=128, temp=0.8, N=5, rep_penalty=0.8

No training in 5a-5d. Testing base model only across multiple inference configs. 5e combines the best inference settings, then fine-tunes on RTX PRO 6000.

### GPU progression across the LLaDA experiment

| GPU | VRAM | Used for |
|-----|------|----------|
| T4 | 16 GB | Runs 1-2: first training attempts. OOM at MAX_SEQ_LEN=2048. |
| L4 | 24 GB | Runs 2-3: inference baselines. Also OOM at 2048 during training. |
| A100 | 40 GB | Runs 3-5e: training at 2048 + fp16 inference. OOM on fp16 training. |
| RTX PRO 6000 Blackwell | 96 GB | Run 5e fine-tuning: fp16 training with 80GB headroom. |

### Why this run exists

Runs 1-4 established that the base LLaDA model degenerates into repetition cascades regardless of fine-tuning. Fine-tuning makes it worse (run 4 verdict). Before trying more training, we need to figure out why the base model can't generate coherent long-form text.

Compared our `generate_llada` against the official LLaDA repo's `generate()` function. The core generation logic is **identical** — code is not the problem. But we found several config mismatches with the paper's defaults and the authors' own ablation results.

### Debugging roadmap (one variable at a time)

| Run | Change | Why | Result |
|-----|--------|-----|--------|
| 5a | `temperature=0` (greedy) | Paper's default. Tests whether stochastic sampling causes the cascade | **WORSE.** Greedy removes noise that was breaking symmetry. Cascades more severe than 0.8. |
| 5b | `block_length=64`, `steps=256` | Their ablation found 64 > 128, and more steps = dramatically better quality | **PARTIAL.** Short outputs improved (some fully coherent). Long outputs get more runway but still cascade. |
| 5c | fp16 (no quantization) on A100 | Tests whether 4-bit distorts confidence rankings across positions | **PARTIAL.** Structural coherence improved (essay sections, formatting). Cascade still hits at ~300-400 words. Short-form (128 tokens) clean. |
| — | Base model instead of Instruct | Tests whether RLHF peaked distributions trigger the cascade | **SKIP** — Instruct needed for instruction-following canary prompts |
| 5d | `SUPPRESS_EOS=True` (EOS/EOT confidence → -inf) | Paper Appendix B.4 warns about EOS flooding in low-confidence remasking | **WORSE.** Removed natural stopping points, increased cascade surface area. New failure mode: sentence-level loops. |
| 5e | Combine 5b + 5c (block_length=64, steps=256, fp16) | Stack the two changes that helped | **BEST long-form.** Canary C: structured essays, cascade recovery across blocks. Short-form unchanged. |

### 5a: Greedy sampling (temperature=0)

**Config change:** `GEN_TEMPERATURE` 0.8 → 0.0. Everything else identical to run 4.

**Hypothesis:** Gumbel noise lets mediocre tokens randomly win the confidence race. Greedy always commits the highest-probability token.

**Result:** Cascades were *more* severe, not less. Canary A collapsed into comma floods. Canary B immediate "Eileen Eileen Eileen" loop. Canary C barely coherent.

**Why it failed:** With temperature=0, there's no randomness to break ties. The highest-probability token at every masked position is the same common token ("the", ",", "and"), so they all get committed simultaneously. The Gumbel noise at 0.8 was actually *helping* by introducing asymmetry across positions — different positions sample different tokens by chance, which gives the next step more diverse context.

**Conclusion:** Stochastic sampling is not the cause. It's a partial cure. Reverted to 0.8.

### 5b: Smaller blocks, more steps (block_length=64, steps=256)

**Config change:** `GEN_BLOCK_LENGTH` 128 → 64, `GEN_STEPS` 128 → 256. Temperature back to 0.8.

**Hypothesis:** Official ablation (issue #39) found quality degrades dramatically with too few steps. Our old config had 16 steps per block of 128 tokens. New config: 16 steps per block of 64 tokens (same steps-per-block but half the tokens to unmask per block, so each step unmasks ~4 tokens instead of ~8). Also, 16 blocks instead of 8 = more left-to-right scaffolding between blocks.

**Baseline generation time:** ~8 minutes for 3 canaries x 5 samples.

**Result by canary:**

**Canary A (short, known topic):** Noticeably better. Sample 2 is fully coherent — a complete paragraph about being a first-gen Chinese immigrant with class observations, ending with hashtags. Sample 4 starts well (~150 words) then degenerates into "education, education, education." Sample 5 is very short but coherent. Mixed: 2/5 fully coherent, 2/5 partial, 1/5 degenerate.

**Canary B (short, novel topic):** Better than before. Sample 1 is a reasonable coherent summary of both athletes with emojis. Sample 4 has structured formatting with a clear narrative. Samples 2-3 start well then cascade ("Area Area", "# # #"). 2/5 coherent, 3/5 partial cascade.

**Canary C (long essay):** Still cascades in all 5 samples. All get 2-4 coherent paragraphs (~200-400 words) about Ellul with structured sections, then collapse into "homogen homogen", "conformity conformity", "reality reality", "control control." More runway than run 4 (~200-400 words vs ~50-200 words coherent), and the cascade tokens are topic-related rather than purely syntactic. But the cascade is still inevitable at 1024 tokens.

**Pattern:** Short outputs (A, B) — some fully coherent now. Long outputs (C) — more coherent runway but cascade still hits. The smaller blocks help by giving LLaDA more left-to-right scaffolding (each block can see all previous blocks' committed tokens before starting), but within a block the bidirectional cascade mechanism still operates.

**Conclusion:** Block_length=64 / steps=256 is a meaningful improvement for short generation. For long generation, it delays but doesn't prevent the cascade. Keep this change for 5e.

### 5c: fp16 no quantization

**Config change:** `USE_QUANTIZATION = False`. Model loaded in fp16 (~16GB) instead of 4-bit NF4 (~5GB). Requires A100. All other config back to baseline (block_length=128, steps=128, temp=0.8).

**Hypothesis:** LLaDA needs accurate *relative confidence rankings* across hundreds of masked positions simultaneously, and 4-bit quantization doesn't have the resolution to provide them.

**Why quantization hurts LLaDA more than autoregressive models:**

An autoregressive model (Llama) is a typist — at each position, the question is "what word goes here?" If 4-bit noise slightly distorts the scores, the ranking at that one position is usually preserved ("the" still beats "cat"). And the next position gets fresh logits based on whatever was actually committed. Errors don't compound across positions.

LLaDA is a teacher grading 128 exams simultaneously, ranking them to decide which to post first. The question isn't "what's the answer at position 6?" — it's "should I commit position 6's answer or position 73's answer *first*?" The entire unmasking mechanism depends on comparing confidence scores *across* positions.

The confidence differences between positions can be tiny — 0.15 vs 0.17. 4-bit quantization adds noise to every weight in the model. That noise propagates through 32 layers of matrix multiplications. By the time you get logits, the cumulative error can flip which position looks most confident. Position 42 (true confidence 0.15) gets inflated to 0.18. Position 73 (true confidence 0.17) gets deflated to 0.14. Wrong token gets committed first.

Then the cascade: once a wrong token is committed, every remaining masked position sees it simultaneously (bidirectional attention) and adjusts. The next step's confidence rankings are distorted by both quantization noise AND bad context. It compounds.

fp16 has 65,536 possible values per weight. 4-bit NF4 has 16. The confidence ranking needs resolution that 16 values can't provide when the differences between positions are small — which they always are in early unmasking steps, when almost everything is still masked and the model is least sure.

**N_SAMPLES:** Trimmed to 2 (from 5) for this diagnostic run to save time. Full baselines estimated at ~12 min instead of ~30 min.

**VRAM:** 16.03 GB (fp16, no quantization). 24GB headroom on A100.

**Gotcha: CPU offloading on re-run.** First attempt showed 31.62 GB VRAM and the warning `Some parameters are on the meta device because they were offloaded to the cpu.` The old 4-bit model (~5GB) was still in GPU memory when Cell 4 re-ran to load the fp16 model (~16GB). Both models coexisted briefly, pushing total VRAM past what accelerate expected, so it offloaded some fp16 layers to CPU. Result: 10x slower (6.7 tok/sec vs 69.4) and worse output quality (immediate degeneration). Fix: **restart the Colab runtime** before loading a different-sized model. After restart, VRAM showed 16.03 GB with no offloading warning. General rule: whenever changing quantization config, restart runtime first — `del model; torch.cuda.empty_cache()` is not reliable enough for large models with `device_map="auto"`.

**Smoke test (128 tokens):** Clean — no cascade, coherent throughout, 102.5 tok/sec. 3x faster than L4 with 4-bit.

**Baseline generation time:** ~8 min for 2 samples x 3 canaries.

**Result by canary:**

**Canary A (short, known topic):** Sample 1 is coherent through the body — a complete Substack Note + Tweet format. Only degenerates at the trailing hashtags ("# # # #"). Sample 2 starts well but cascades midway ("America America America"). Similar to run 4 (4-bit + rep penalty). Modest improvement.

**Canary B (short, novel topic):** Both samples cascade into "Area Area Area Area" within the first paragraph. This is the hardest canary (name-heavy, triggers repetition). No improvement over 4-bit.

**Canary C (long essay):** The real signal. Both samples generate structured essays with actual section headers ("Early Life and Influences", "Theories of Propaganda", "Technological Conformity", "Contemporary Relevance"). The model is *organizing* the essay into sections before cascading — that structural scaffolding wasn't as consistent in the 4-bit runs. Still cascades at ~300-400 words into "homogenization homogenization", but the coherent portion is qualitatively better — more like a real essay draft and less like generic Wikipedia filler.

**Conclusion:** fp16 is a real improvement, especially in structural coherence. Short-form generation (128 tokens) works cleanly. Long-form still cascades but the coherent portion is better organized. The improvement should stack well with block_length=64 (5b's finding) in the combine run (5e). Keep this change for 5e.

**Why fp16 is also faster than 4-bit (counterintuitive):** 102.5 tok/sec (fp16) vs 42.9 tok/sec (4-bit) on A100. 4-bit saves *memory*, not compute. bitsandbytes stores weights in 4-bit but dequantizes them back to fp16 on the fly for every matrix multiplication, every layer, every forward pass. The actual math is the same — fp16 GEMM either way — but 4-bit adds a dequantization tax. On A100, this tax is pure waste: the GPU has 40GB VRAM (fp16 model fits at 16GB) and 1.5 TB/s memory bandwidth (reads 16GB of weights in ~11ms). A100's tensor cores are optimized for native fp16 — no dequantization needed. LLaDA makes this worse than usual because there's no KV cache, so every forward pass reads all 32 layers of weights. Hundreds of forward passes per generation, each paying the dequantization overhead. 4-bit is a trade: speed for memory. On T4/L4 where fp16 doesn't fit, it's the price of admission. On A100, it's overhead you don't need.

### 5d: EOS/EOT suppression (SUPPRESS_EOS=True)

**Config change:** Added `SUPPRESS_EOS = True`. After computing confidence, set confidence to `-inf` for any position predicting EOS (126081) or EOT (126348) tokens. Back to 4-bit quantization (clean test). All other config baseline (128 steps, block_length=128, temp=0.8).

**Hypothesis:** Paper Appendix B.4 warns that extensive EOS token padding in SFT data causes EOS/EOT to get artificially high confidence during unmasking. Once committed, they signal "stop generating" and poison context for surrounding positions.

**Result by canary:**

**Canary A:** Sample 1 gets ~200 coherent words about class and immigration, then cascades into a complex cycling loop ("America is opportunity" / "successful successful" / "believe that" repeating in long cycles). Sample 2 is a full-sentence loop: "As a kid, I was always encouraged to work hard, but I also felt the weight of my family's expectations on me" repeated 30+ times verbatim.

**Canary B:** Total catastrophe. Worst across all runs. "Eileenileenileen" cascades, Chinese characters (鼬), @-symbol chaos, then character-level degeneration.

**Canary C:** Sample 1 gets ~200 words then a new failure mode: "ulululul" (subword cascade), followed by sentence-level loops ("This means that society is led by by technology" repeated endlessly). Sample 2 gets ~300 words with structured sections, then "conform conform conform."

**New failure mode — sentence-level repetition:** The rep_penalty targets individual token counts, but 5d's cascades operate at the sentence level. "As a kid, I was always encouraged to work hard..." repeats as a whole sentence — each individual word only appears a few times per copy, so the penalty barely touches it.

**Why it made things worse:** EOS tokens were acting as natural fill — positions committed as EOS are "done" and don't participate in the confidence race for subsequent steps. By suppressing EOS, we forced every position to be a content token, increasing the number of positions competing simultaneously. More competition = more surface area for the cascade.

**Conclusion:** Drop 5d from the combine run. EOS suppression is counterproductive for LLaDA generation.

### 5e: Combined — block_length=64, steps=256, fp16

**Config change:** `GEN_BLOCK_LENGTH` 128 → 64, `GEN_STEPS` 128 → 256, `USE_QUANTIZATION = False` (fp16). Combines 5b + 5c. EOS suppression off (5d dropped). Temperature 0.8, rep_penalty 0.8, N_SAMPLES=2.

**Hypothesis:** 5b improved short-form and extended coherent runway on long-form. 5c improved structural coherence and confidence ranking accuracy. Stacking should compound: more steps per block (5b) with more accurate confidence rankings (5c).

**Generation time:** ~3 min for 2 samples × 3 canaries. (The smoke test extrapolation of 26 min was wrong — it scaled linearly by steps without accounting for block structure.)

**Result by canary:**

**Canary A (short, known topic):** Sample 1 is the best A across all runs — a coherent Substack Note about first-gen Chinese immigrant experience, complete with hashtags, only minor doubled words ("generation-generation"). Sample 2 starts well for ~2 paragraphs then cascades into "But the But is that is is that is that is that that that is." 1/2 coherent — similar to 5b, not clearly better.

**Canary B (short, novel topic):** Still the hardest canary. Sample 1 gets a coherent paragraph about both athletes before "resilience resilience resilience" cascade then "# # # # #" hashtag flood. Sample 2 is catastrophic — "half Asian half Asian half Asian" → "Eileenileenileen" wall, same failure mode as every prior run. No meaningful improvement over 5b or 5c individually. Name-heavy content overwhelms both fixes.

**Canary C (long essay):** Best long-form outputs across all runs.

Sample 1: Structured essay with clear sections (Views on Propaganda, Views on Technological Conformity, Influence on Modern Understanding, Conclusion). The propaganda section is coherent for ~300 words. Cascades at "conformity conformity conformity" in the technological conformity section. But then **the conclusion section recovers** — mostly coherent, summarizing Ellul's relevance. This cross-block recovery is new. In all prior runs, cascade consumed everything after onset.

Sample 2: ~500 words of structured essay with sections (Early Contributions, Technologicalagoraganda, Propaganda Culture, Theagora, Impact and Legacy, Conclusion). Hallucinated concepts ("Theagora," "Technologicalagoraganda") but structured as real academic terms with definitions. The cascade here is qualitatively different — **semantic repetition** ("widely adopted by scholars in media studies, cultural studies, and social media research" repeated across multiple sections) rather than raw token floods ("conformity conformity conformity"). The model is looping at the paragraph level, not the token level.

**Why the improvements stack for long-form but not short-form:**

Short-form (A, B): Generation fits in 1-2 blocks. The cascade mechanism operates within a single block — smaller blocks and better precision help modestly but can't prevent the positive feedback loop once it starts.

Long-form (C): Generation spans 16 blocks (1024 tokens / 64 per block). Each block starts with committed tokens from all previous blocks as context, but its own internal unmasking is fresh. A cascade in block 6 doesn't necessarily propagate to block 9 because block 9 begins its own confidence-based unmasking with new forward passes. fp16 gives each block better confidence rankings to work with, and more steps per block (4 tokens unmasked per step instead of 8) give the model more iterative refinement within each block.

The combination creates a regime where: (1) individual blocks are more likely to be coherent (both fixes contribute), and (2) even when one block cascades, later blocks can recover from the committed context of earlier coherent blocks. This is the block structure working as designed — semi-autoregressive scaffolding limiting cascade propagation.

**Conclusion (A100 baselines):** 5e is the best inference config found. The improvements compound for long-form generation but don't solve the fundamental cascade mechanism.

### 5e continued: RTX PRO 6000 Blackwell (96GB) — baselines + fine-tuning

A100 OOM'd on fp16 training (32GB VRAM before training, no headroom for activations/gradients). Upgraded to RTX PRO 6000 Blackwell (96GB) — the full GPU ladder from T4 → L4 → A100 → RTX.

**Inference speed:** 97.8 tok/sec (smoke test). Comparable to A100's 102.5. Baselines took ~2-3 min for 2 samples × 3 canaries.

#### RTX baselines (same config, same model — confirming A100 results)

**Canary A:** Sample 1 gets ~400 coherent words — structured Substack Note with three "Note:" sections. Paragraph-level repetition at the very end (third section repeats the second), not token cascade. Sample 2 is **fully clean** — a complete short Note with hashtags. Best A results across all runs.

**Canary B:** The headline result. **Zero catastrophic cascades.** Sample 1 produces a coherent Substack Note AND Tweet with @ handles, emojis, and hashtags. Minor artifacts ("half-asAsian") but structurally complete. Sample 2 also coherent — complete Note with handles (@ileenileenu, @alyssalieliu — hallucinated but structured). Every prior run had at least one "Eileenileenileen" catastrophe on Canary B. Same model, same config, different Gumbel noise draws — the cascade isn't inevitable at this config, just probabilistic.

**Canary C:** Same pattern as A100. Sample 1 gets ~500 coherent words about Ellul ("The Technic Society," technological conformity, application to fake news), then paragraph-level repetition ("Ellul's work have been applied by contemporary scholars..." verbatim ×11). Sample 2 gets ~400 words, then citation-level cascade ("Propaganda" (1965) ×30+). Semantic repetition, not raw token floods.

**Baseline verdict:** RTX confirms A100 quality (expected — same weights, same precision, same config). The Canary B clean draws show that 5e's config puts the model in a regime where cascades are probabilistic rather than deterministic for short-form content.

#### RTX fine-tuning (fp16, no quantization)

**Training config:** Same as prior runs (LR=5e-5, LoRA rank=16, alpha=32, 3 epochs, grad_accum=4, MAX_SEQ_LEN=2048) but in fp16 instead of 4-bit. VRAM before training: 32.19 GB with ~64GB headroom.

**Training time:** ~1.5 min for 3 epochs × 158 examples.

**Loss curve:**

| Epoch | Train loss | Val loss | Note |
|-------|-----------|---------|------|
| 1 | 3.30 | **2.70** | Best checkpoint saved |
| 2 | 3.10 | 4.01 | Overfitting — val loss ↑48% |
| 3 | 3.25 | 3.61 | Still worse than epoch 1 |

Val loss spiked after epoch 1 and never recovered. Best checkpoint (epoch 1) saved automatically. High batch-level loss variance (0.0 to 8.5) is expected — LLaDA's `1/p_mask` weighting amplifies loss when masking rate `t` is near 0 (few tokens masked, each gets huge weight).

**Overfitting pattern:** Same as prior LLaDA training runs. 158 examples × 1 epoch may be the sweet spot for this model — the LoRA adapter learns quickly but generalizes poorly past epoch 1. The `1/p_mask` loss amplification means the effective gradient signal per step is already high, so fewer passes through the data are needed.

#### Fine-tuned canary results

**Canary A:** Sample 1 has the ghost line — short coherent opening about being a first-gen Chinese immigrant, then fragments and hashtags (~150 words). **"I'm like a ghost stuck, stuck in the limbo."** This isn't in the training essays, isn't generic LLM filler, and is structurally interesting — the repeated "stuck" with the comma creating a rhythmic stumble. Compressed, imagistic, a little bleak. The fine-tuning *did* learn something about voice. Sample 2 is the worst A across all runs — "I love America because I am breathe to breathe my life" sentence-level loops, Chinese characters leaking (单项金额), "assistant" token bleeding through, then a long identity-crisis loop ("I'm a Chinese boy from poor family but I'm rich, and I'm not poor").

**Canary B:** Sample 1 is pure "Eileen" catastrophe from the first token — then morphs into "Ellen Ella" (the name cascade evolved into new names). Sample 2 is surprisingly decent — a structured Substack Note AND Tweet about both athletes, with some repetition ("Winter Winter Winter", "Stanford Stanford Stanford") but the content and structure hold. High variance between samples.

**Canary C:** More interesting than expected. Sample 1 is a structured essay with sections (Critique of Technology, Critique of Propaganda, Ellul as a Forgotten Prophet, Conclusion, References). ~500 words mostly coherent. The "Forgotten Prophet" section has a numbered list that degenerates (items nest/repeat). References hallucinate titles ("The Myth of the Last Man," "Bocken Books") but maintain citation format. Sample 2 also structured — sections with horizontal rules (Early Life, Concept of Propaganda, Technological Conformity, Influence, Legacy, Conclusion). ~400 coherent words, then "Blade Runner" cascade in the Legacy section, followed by a mostly-coherent conclusion. Both C samples are comparable to the best base model outputs — fine-tuning didn't clearly degrade long-form under the 5e config, unlike run 4.

**N=2 caveat:** Variance between samples is enormous (Canary B sample 1 vs sample 2 are night and day). With only 2 samples per canary, we can't distinguish signal from Gumbel noise luck. The results are suggestive but not conclusive.

#### The cruel trade-off

The fine-tuning learns voice — the ghost line proves it. But the same mechanism that makes output more distinctive (peaked confidence toward specific, expressive tokens) is exactly what triggers the cascade. The adapter shifts the model's confidence distribution toward more opinionated predictions. In an autoregressive model, that's good — confident predictions at one position don't affect other positions. In LLaDA, confident predictions at one position get seen by ALL other positions simultaneously (bidirectional attention), amplifying the confidence peak across the entire sequence.

**The more voice it learns, the faster it degenerates** — at least for short-form. The long-form results (Canary C) are more ambiguous: fine-tuned C outputs are comparable to base model C outputs under the 5e config, suggesting the block structure may contain the damage for longer generation where each block gets a fresh start.

#### Run 5 verdict

Best inference config: block_length=64, steps=256, fp16 (5e). Base model produces structured essays with ~300-500 words of coherent content. Short-form sometimes fully clean (Canary B escaped cascades entirely on RTX baselines).

Fine-tuning: learns real voice signal (the ghost line, the identity-crisis loop's raw emotional content) but destabilizes short-form generation. Long-form is a wash — fine-tuned Canary C is comparable to base Canary C. The cascade remains architectural. N=2 samples makes definitive conclusions impossible, but the pattern across runs 1-5 is consistent: fine-tuning peaks the confidence distribution, which feeds the bidirectional cascade mechanism.

The experiment's real contribution is diagnostic: understanding *why* masked diffusion models degenerate, what partially mitigates it (smaller blocks, more steps, fp16 precision), and why fine-tuning for style is fundamentally at odds with confidence-based unmasking.

### Why fine-tuning makes LLaDA worse

Two problems stacking, one fixable, one not:

**1. Overfitting (dataset size).** 158 examples is small. Val loss spikes after epoch 1 in every run — the adapter memorizes specific phrases rather than learning generalizable style patterns. The verbatim loops ("I'm a Chinese boy from poor family but I'm rich, and I'm not poor" ×20) are memorized phrasing, not generated text. More data would reduce this, giving the model a broader distribution of voice patterns instead of a handful of exact phrases to latch onto.

**2. Confidence peaking (architectural).** The entire *point* of voice fine-tuning is to make the model more opinionated — prefer certain words, rhythms, structures. In autoregressive models, peaked confidence is pure upside: a more decisive model at position N produces better text, and position N+1 gets fresh logits. Peaked confidence doesn't propagate. In LLaDA, more opinionated = more positions simultaneously confident about the same tokens = cascade. Every masked position sees the same bidirectional context. When the adapter says "prefer words like 'stuck,' 'ghost,' 'limbo,'" that preference manifests at ALL positions at once. They don't know they should each pick *different* expressive words — they all reach the same confident conclusion.

**More data helps problem 1 but worsens problem 2.** Better generalization means a stronger, broader style signal — a *more* peaked distribution across more vocabulary space. The cascade would be less repetitive (fewer exact-phrase loops) but the underlying mechanism — confident predictions amplifying across positions — would still operate. Voice fine-tuning and confidence-based unmasking are fundamentally at odds.

### On the ghost line

The fine-tuned Canary A produced: **"I'm like a ghost stuck, stuck in the limbo."** This was initially read as proof of voice learning. On closer examination:

**The tonal shift is real.** Base model Canary A on the same RTX produces "The class divide was both of my reality" — competent, generic, essay-about-class energy. The fine-tuned version reaches for compressed, imagistic, emotionally direct language. That shift maps to qualities in Lily's writing: compression, concrete images over abstractions, bleakness, rhythmic awareness (the repeated "stuck" with the comma creating a stumble). The base model doesn't produce lines like that. Something in the training data pushed the distribution toward that register.

**But it's N=1.** One line in one sample. The rest of that same sample is garbled ("I'm't sure how to make even the most of it"). Sample 2 is a catastrophe. Cherry-picking the best fragment from a stochastic process and attributing meaning is what humans always do with generative models. A model with peaked confidence toward emotional/personal language will occasionally land on a good combination by chance.

**The honest version:** The *tonal shift* from base to fine-tuned is real and directional. The fine-tuned model reaches for more personal, compressed language. Whether "ghost stuck in the limbo" specifically is voice learning or a lucky draw from a shifted distribution can't be proven from one line. The shift itself is the evidence. The ghost line is the prettiest artifact of that shift, not the proof.

### Code comparison details

| Element | Official `generate()` | Our `generate_llada` | Match? |
|---------|----------------------|---------------------|--------|
| Gumbel noise formula | `logits.exp() / (-log(U))^temp` | Same + `noise.clamp(min=1e-20)` | Yes (clamp is safety, no behavioral difference) |
| Token sampling | `argmax(logits_with_noise)` | Same | Yes |
| Confidence | `softmax(clean_logits)[chosen_token]` | Same | Yes |
| Remasking | Set non-top-k back to MASK_ID | Same | Yes |
| Block structure | `steps // num_blocks` per block | Same (`max(1, ...)` instead of `assert`) | Yes |
| Future block masking | `-np.inf` on positions beyond block_end | Same | Yes |
| CFG support | Yes (default off) | No | N/A (off by default) |
| EOS suppression | `logits_eos_inf`, `confidence_eos_eot_inf` flags (default off) | No | N/A (off by default) |
| `attention_mask` | Passed through (default None) | Not passed | N/A (model trained without it per issue #89) |
| Remasking strategy | `'low_confidence'` or `'random'` | Low confidence only | Yes (low_confidence is default) |
