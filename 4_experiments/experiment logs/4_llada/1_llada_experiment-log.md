## Run 1 — Mar 12, 2026
**Model:** LLaDA 8B Instruct (LoRA, 4-bit quantized)
**Phase:** Full dataset (158 pairs)
**Config:** LoRA rank=16, alpha=32, lr=2e-4, epochs=3, batch_size=1, grad_accum=4, max_seq_len=2048, fp16 autocast, 4-bit NF4 quantization, AdamW optimizer, cosine decay with 10-step warmup, gradient_checkpointing=OFF (LLaDA doesn't support it)
**Data:** 158 train pairs (mixed tiers, same as Llama), 12 val pairs
**GPU:** A100 (40GB). T4 and L4 both OOM'd at 2048 without gradient checkpointing.
**Training time:** ~3 minutes on A100.
**VRAM:** 7.88 GB before training, ~22 GB peak (estimated).

**Loss:**

| Epoch | Training Loss | Validation Loss |
|-------|--------------|-----------------|
| 1     | 3.4473       | 3.7841          |
| 2     | 3.2291       | 3.1458 (best)   |
| 3     | 2.8271       | 3.1937          |

**Loss curve shape:** Steady decline in training loss across all 3 epochs (3.45 → 3.23 → 2.83). Validation loss improved significantly from epoch 1 to 2 (3.78 → 3.15), then ticked up slightly in epoch 3 (3.19) — mild overfitting starting. Best checkpoint saved at epoch 2. High variance within epochs (individual batch losses range from 0.0005 to 7.54) — expected given the 1/p_mask importance weighting in LLaDA's loss function, where lightly-masked examples get upweighted.

**Notable loss spikes:** Batches with loss >6.0 appear in every epoch (e.g., batch 70 epoch 1: 7.20, batch 100 epoch 1: 7.54, batch 70 epoch 2: 6.69). These are likely examples where the masking rate t was low (few tokens masked), causing the 1/p_mask weighting to amplify the loss. This is by design — it ensures the model learns from all masking levels equally.

**Comparison with Llama (Run 3):**

| | Llama 3.1 8B (LoRA) | LLaDA 8B (LoRA) |
|---|---|---|
| Optimizer | paged_adamw_8bit | AdamW |
| LR | 1e-4 | 2e-4 |
| Train loss (final) | 2.49 | 2.83 |
| Val loss (best) | 2.82 | 3.15 |
| Loss trend | Flat (barely moved) | Declining (still learning) |
| Overfitting | None | Mild (epoch 3 val ticked up) |
| Training time | 26 min (T4) | 3 min (A100) |
| Gradient checkpointing | Yes | No (unsupported) |

LLaDA's higher absolute loss is expected — its loss function divides by masking probability, inflating values compared to standard cross-entropy. The losses are not directly comparable between models. What matters is the trend: LLaDA is still learning at epoch 3 (Llama plateaued immediately), and val loss improved meaningfully (3.78 → 3.15 vs Llama's dead-flat 2.82).

### GPU Progression (debugging OOM)

| GPU | VRAM | MAX_SEQ_LEN=2048 |
|-----|------|-------------------|
| T4 (15GB) | ~7GB free after model | OOM on first batch — attention maps alone (~8.6GB) exceed headroom |
| L4 (24GB) | ~14GB free after model | OOM on batch ~10 — short sequences fit, longer ones overflow (backprop activations across 32 layers) |
| A100 (40GB) | ~32GB free after model | Fits comfortably, all 158 examples × 3 epochs completed |

Root cause: LLaDA's custom model class (`LLaDAModelLM`) does not implement `supports_gradient_checkpointing`. Without gradient checkpointing, all intermediate activations across 32 transformer layers must be held in memory for backprop. L4 would have been sufficient with gradient checkpointing.

### Canary outputs

**Verdict: catastrophic degeneration. All three canaries collapsed into repetitive tokens.**

### Canary A — Fine-Tuned (both samples)
```
Class class class class class class class class class class class class
class class class class class class class class class class. class.
class class class class class class class... [512 tokens of "class"]
```

### Canary B — Fine-Tuned (best of 2)
```
Substack Note and and and and and and and

Eileen and Alyssa are both winter gold gold gold gold gold gold gold
gold gold medalists.

 • Both grew up in the Bay Area, half-asAsian, half-white, conceived
   via anonymous egg donor, and raised by a single parent.

 • • • • • • • • • • • • • • • • • • • • • • • • [hundreds of bullets]
```

### Canary C — Fine-Tuned (best of 2)
```
Ell Jacques Ellul is a forgotten prophet of propaganda and technological
conformity, but his work is still very much alive today.

Ell Jacques Ellul was a French ,, ,,,, ,, , ,, , ,, ,, ,, ,, ,, ,, ,,
,, ,, ,, ,, ,, ,, and ,.

Ellul was , , , , in , , , , , , , , , , , , , , [hundreds of commas]
```

### What happened

The model didn't learn voice — it learned to repeat high-frequency tokens. This is **catastrophic repetition from masked diffusion fine-tuning.**

**Mechanism:** LLaDA's iterative unmasking predicts all masked positions simultaneously, then commits the most confident predictions first. Fine-tuning shifted the logit distribution so that a few high-frequency tokens ("class", commas, bullets) always win the confidence race in early unmasking steps. Once committed, they dominate the context for later steps, causing a repetition cascade. Unlike autoregressive generation, where each token sees different left context, LLaDA's bidirectional attention means every position sees the same (repetitive) context simultaneously — amplifying the problem.

**Why the loss curve was misleading:** Training loss declined steadily (3.45 → 2.83), which looked healthy. But the model was learning to predict repeated tokens with high confidence — easy to predict, low loss, terrible output. The "declining loss" was actually the model converging on degenerate repetition, not on voice.

**Likely causes:**
1. **LR too high (2e-4).** Llama used 1e-4. LLaDA's loss already amplifies gradients via 1/p_mask weighting — the effective learning signal is stronger per step than standard cross-entropy. 2e-4 on top of that was probably too aggressive, pushing the model past useful learning into degeneration.
2. **No repetition penalty in inference.** The `generate_llada` function uses Gumbel noise for diversity but has no explicit repetition penalty. Autoregressive sampling typically includes `repetition_penalty=1.1` to suppress repeated tokens. LLaDA's confidence-based unmasking has no equivalent mechanism built in.
3. **Small dataset + masked diffusion.** 158 pairs may be too few for LLaDA's training objective. Each training step randomly masks a different subset of tokens, so the model sees many noisy views of the same data. With only 158 examples, the signal-to-noise ratio may be too low to learn anything beyond surface-level token frequencies.

### Why this doesn't happen in autoregressive models

The degeneration cascade is specific to how LLaDA's unmasking interacts with bidirectional attention:

1. Fine-tuning shifted the logit distribution so common tokens ("class", commas, bullets) get slightly higher confidence
2. In step 1 of unmasking, these high-confidence tokens get committed first (they win the confidence race)
3. LLaDA's bidirectional attention means **every remaining MASK position sees the same repetitive context** — all at once, not one at a time
4. This makes the model even more confident about the same repeated tokens for the next step
5. Cascade → the whole sequence collapses

Autoregressive models don't have this problem because each position sees *different* left context. Position 50 sees a different sequence than position 100. In LLaDA, all positions see the same thing simultaneously — so a small distributional shift gets amplified catastrophically across all positions at once.

### Fixes, ranked by likelihood of working

**1. Much lower LR (most likely fix).** LR was 2e-4. Llama used 1e-4. But LLaDA's loss already amplifies gradients via `1/p_mask` — when t is small (few tokens masked), the gradient signal is huge. The *effective* learning rate is already higher than the nominal rate. Try **5e-5 or even 2e-5**. Goal: nudge the distribution slightly, not shove it.

**2. Reduce LoRA rank (r=4 instead of 16).** Less capacity = less room to learn degenerate patterns. Rank 16 across 160 layers is a lot of expressiveness for 158 examples. Rank 4 constrains the adapter to only learn the strongest signal (hopefully voice, not repetition).

**3. Fewer epochs (2 instead of 3).** Val loss was best at epoch 2 (3.15) and ticked up at epoch 3 (3.19). The degeneration might be a late-training artifact.

**4. Add repetition penalty to the unmasking loop.** During each unmasking step, before selecting the top-k most confident tokens, penalize any token that's already been committed elsewhere in the sequence:

```python
# After computing confidence, before selecting top-k:
committed_tokens = x[0][x[0] != MASK_ID]
for token_id in committed_tokens.unique():
    penalty_mask = (x0[0] == token_id) & mask_index[0]
    confidence[0, penalty_mask] *= 0.8  # dampen repeated predictions
```

This is a band-aid — the model shouldn't need it — but it could reveal useful output hiding under the repetition.

**5. Data is probably NOT the main problem.** 158 pairs is small, but Llama learned voice from the same 158 pairs. The issue is how LLaDA amplifies small distributional shifts, not insufficient data. Fix LR and rank first. If the model then produces coherent but generic output, *then* data quantity might be the bottleneck.

### Run 2 plan

```
LR = 5e-5          # was 2e-4 — 4x lower
LORA_R = 4          # was 16 — much less capacity
LORA_ALPHA = 8      # keep alpha = 2 * rank
EPOCHS = 2          # was 3 — stop before degeneration
```

If this still degenerates: try 2e-5 LR, or this may be a fundamental limitation of fine-tuning masked diffusion with small datasets — the generation mechanism amplifies small distributional shifts into catastrophic repetition in a way that autoregressive models don't.
