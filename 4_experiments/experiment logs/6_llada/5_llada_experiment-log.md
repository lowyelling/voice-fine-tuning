## Run 5 — Mar 12, 2026

**Model:** LLaDA 8B Instruct (4-bit quantized unless noted)
**Phase:** Base model diagnostics — isolating the degeneration cause
**GPU:** A100 (40GB)
**Baseline config:** FULL — 128 steps, 1024 gen tokens, block_length=128, temp=0.8, N=5, rep_penalty=0.8

No training in this run. Testing base model only across multiple inference configs.

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
| 5e | Combine 5b + 5c (block_length=64, steps=256, fp16) | Stack the two changes that helped | Next — final run |

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
