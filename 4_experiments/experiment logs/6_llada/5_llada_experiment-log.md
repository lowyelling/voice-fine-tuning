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
| 5c | fp16 (no quantization) on A100 | Tests whether 4-bit distorts confidence rankings across positions | Running |
| — | Base model instead of Instruct | Tests whether RLHF peaked distributions trigger the cascade | **SKIP** — Instruct needed for instruction-following canary prompts |
| 5d | `confidence_eos_eot_inf=True` | Paper Appendix B.4 warns about EOS flooding in low-confidence remasking | Next |
| 5e | Combine all fixes that helped | Stack the winning changes from 5a-5d | Final run |

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

**Hypothesis:** LLaDA needs accurate *relative confidence rankings* across hundreds of masked positions simultaneously. 4-bit NF4 quantization introduces noise into logits — small precision errors could distort which token wins the confidence race at each step. Autoregressive models are more robust to this because they only need the argmax right one token at a time. LLaDA needs precise confidence comparisons across all positions at once.

*(results pending)*

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
