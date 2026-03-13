## Run 5 — Mar 12, 2026
**Model:** LLaDA 8B Instruct (LoRA, 4-bit quantized)
**Phase:** Base model diagnostics — greedy sampling
**Config:** Same as runs 2-4 (LoRA rank=16, alpha=32, lr=5e-5, epochs=3, batch_size=1, grad_accum=4, max_seq_len=2048, fp16 autocast, 4-bit NF4 quantization, AdamW optimizer, cosine decay with 10-step warmup, gradient_checkpointing=OFF)
**Inference:** FULL config — 128 steps, 1024 gen tokens, block_length=128, **temp=0.0 (greedy)**, N=5, rep_penalty=0.8
**Change from run 4:** `GEN_TEMPERATURE` 0.8 → 0.0. Single variable change. Everything else identical.
**GPU:** A100 (40GB)

### Why greedy sampling

Compared our `generate_llada` against the official LLaDA repo's `generate()` function (https://github.com/ML-GSAI/LLaDA/blob/main/generate.py). The core generation logic is **identical** — same Gumbel noise formula, same confidence computation (softmax of clean logits), same remasking, same block structure. Code is not the problem.

But the official code defaults to `temperature=0.` (greedy). We've been using 0.8 for all four runs.

Three findings from the official repo that reframe the problem:

**1. The paper uses greedy sampling for all benchmarks.**
`add_gumbel_noise` returns raw logits when temperature=0 — no noise, pure argmax. With temperature=0.8, Gumbel noise means a mediocre token can randomly win the confidence race in early unmasking steps. Once committed, bidirectional attention means all remaining positions see it and amplify it. Greedy avoids this by always committing the highest-probability token.

**2. Our steps_per_block is low (issue #39 ablation).**
The authors found quality degrades dramatically with fewer diffusion steps: 512 tokens with 32 steps → 6.14% on GSM8K, vs 128 steps → 64.97%. Our config: 1024 tokens / 128 block_length = 8 blocks, 128 total steps / 8 blocks = **16 steps per block**. Their optimal was block_length=64 with 32 steps per block. (Not changing this yet — one variable at a time.)

**3. Repetition is a known, unsolved problem (issue #59).**
Users reported `\n\n\n...` cascades. Maintainer's only advice: "adjust temperature." Issue still open.

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

### Debugging roadmap (one variable at a time)

| # | Change | Why | Status |
|---|--------|-----|--------|
| 1 | `temperature=0` (greedy) | Paper's default. Tests whether stochastic sampling causes the cascade | **This run** |
| 2 | `block_length=64`, `steps=256` | Their ablation found 64 > 128, and more steps = dramatically better | Next if needed |
| 3 | `confidence_eos_eot_inf=True` | Paper Appendix B.4 warns about EOS flooding in low-confidence remasking | Next if needed |
| 4 | Base model instead of Instruct | Tests whether RLHF peaked distributions trigger the cascade | Next if needed |
| 5 | fp16 (no quantization) on A100 | Tests whether 4-bit distorts confidence rankings | Next if needed |

### Note on N_SAMPLES

With greedy sampling (no randomness), all 5 samples per canary should be **identical**. If they differ, something is wrong. Keeping N=5 as a sanity check.

### Training

No training for this run — testing base model only. If greedy fixes (or significantly improves) the base model's degeneration, then retrain and test fine-tuned model at temperature=0.

### Baselines (base model, greedy + rep penalty)

*(run pending)*

### Verdict

*(pending results)*
