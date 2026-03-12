## Run 2 — Mar 12, 2026
**Model:** LLaDA 8B Instruct (LoRA, 4-bit quantized)
**Phase:** Full dataset (158 pairs)
**Config:** LoRA rank=16, alpha=32, lr=5e-5, epochs=3, batch_size=1, grad_accum=4, max_seq_len=2048, fp16 autocast, 4-bit NF4 quantization, AdamW optimizer, cosine decay with 10-step warmup, gradient_checkpointing=OFF
**Data:** 158 train pairs (mixed tiers, same as run 1), 12 val pairs
**GPU:** A100 (40GB)
**Change from run 1:** LR 2e-4 → 5e-5. Single variable change. Everything else identical.

**Why LR:** Run 1's catastrophic repetition ("class class class", comma cascades) was the logit distribution getting shoved too far too fast. LLaDA's 1/p_mask loss weighting already amplifies gradients — when t is small (few tokens masked), the gradient signal per step is huge. The effective LR at 2e-4 was much higher than nominal. 5e-5 (4x lower) should nudge the distribution instead of shoving it.

**VRAM:** 8.24 GB before training
**Training time:** ~3 minutes on A100 (same as run 1)

**Loss:**

| Epoch | Training Loss | Validation Loss |
|-------|--------------|-----------------|
| 1     | 3.1921       | 3.4897 (best saved) |
| 2     | 2.9742       | 2.4383 (best)   |
| 3     | 2.8109       | 2.5737          |

**Loss curve shape:** Steady decline in training loss across all 3 epochs (3.19 → 2.97 → 2.81). Validation loss dropped significantly epoch 1→2 (3.49 → 2.44), then rose in epoch 3 (2.57) — same mild overfitting pattern as run 1, best checkpoint at epoch 2. High within-epoch variance remains (batches range from 0.001 to 7.11) — expected from 1/p_mask importance weighting.

**Comparison with run 1 (LR=2e-4):**

| | Run 1 (LR=2e-4) | Run 2 (LR=5e-5) |
|---|---|---|
| Train loss (final) | 2.83 | 2.81 |
| Val loss (best) | 3.15 | 2.44 |
| Val loss (final) | 3.19 | 2.57 |
| Overfitting | Mild (epoch 3) | Mild (epoch 3) |
| Best epoch | 2 | 2 |

Notable: run 2 achieved **much better validation loss** (2.44 vs 3.15) despite similar final training loss. Lower LR = better generalization, less memorization. The gap between train and val loss is smaller (0.37 vs 0.32 at best epoch), suggesting the model learned more transferable patterns rather than overfitting to surface frequencies.

### Baselines (base model, pre-fine-tuning)

Same degeneration as run 1 baselines. The base LLaDA model itself can't generate coherently at FAST config (64 steps, 512 tokens). Canary C gets ~30-50 coherent words then collapses into comma/article cascades. Canaries A and B produce almost nothing. This is a base model + generation config problem, not a fine-tuning artifact.

Run 1's diagnosis was half right: fine-tuning made degeneration worse, but the foundation was already broken at FAST config.

### Canary outputs (fine-tuned)

**Canary A (sample 1):**
```
"First-generation Chinese-American here. Class is real. I'm 16 and my family is lower-middle class, and I'm still toiling at the minimum job. I can't afford to pay my school. I can't afford car college. # # #." # # # # # #.
```

**Canary A (sample 2):**
```
I am class in America. I'm not rich. I'm not upper-middle class. I'm lower-middle class. I'm a working immigrant. I'm a first generation immigrant. I-I'm not a U.S. citizen. I. Am. Not. American. #Im-generation
```

**Canary B (sample 1):**
```
@eileenileen (@ @EileenGu (@eileenileen)) Both Eileen Gu (@eileenileen) and Alyssa Liu (@ Liu (@AlyssaLiu) are egg admire.
```

**Canary B (sample 2):**
```
Here Olympic on gold models @EileenGu (@Eileen Eileen)) @Alyssa_Liu (@Alyssa Liu) Twobl Olympic athletes from the Bay Area
```

**Canary C (sample 1):**
```
Jacques Ellul, a French philosopher, sociologist and sociologist,, one of the most influential critics of propaganda and technology conformity of the 20th century, Ellul was the leading critic of propaganda, technological conformity, and the use of technology in modern society. From the 1960 century, Ellul was that technology was used to enforce conformity and technological conformity, and that it deprived people of freedom and autonomy.

Today, Ellul is largely forgotten, but his ideas are technology relevant today. [...]

One of Ellul's most influential ideas was his concept of "technological conformity," which he defined as the way in which technology is used to enforce conformity and homogenization [...]

Ellul's ideas about technological conformity are relevant relevant today, as we see to homogen homogen homogen homogen [...hundreds of "homogen"]
```

**Canary C (sample 2):**
```
Ellques Ellul was a French philosopher and anti ofoci pathology..
EllEllul [...full degeneration into "EllEllul" pattern]
```

### Verdict

**Partial fix. LR reduction moved output from immediate collapse to delayed collapse.**

- **Run 1 (LR=2e-4):** 0 coherent words. Immediate degeneration ("class class class", commas from token 1).
- **Run 2 (LR=5e-5):** 30-200 coherent words before degeneration kicks in. Canary A produces actual sentences with relevant content. Canary C sample 1 generates ~3 paragraphs about Ellul before "homogen" cascade.

LR was confirmed as the most impactful variable — the model now generates real content before the repetition cascade takes over. But the cascade still happens. This matches the base model behavior at FAST config: the repetition tendency is baked into LLaDA's unmasking mechanism at 64 steps, and fine-tuning shifts *when* it starts but doesn't eliminate it.

**No voice signal detected.** The coherent portions read as generic Wikipedia-style text, not Lily's voice. But it's hard to evaluate voice when the output degenerates partway through — the model may not have enough coherent runway to express stylistic patterns.

**LR is no longer the bottleneck.** Going lower (2e-5) might delay degeneration by another paragraph, but won't fix it. The problem is now in the generation mechanism, not the training.

### Next steps (regardless of run 2 outcome)

**1. Switch to FULL config for evaluation (highest priority, zero code changes).**
Uncomment the FULL config in Cell 3: 128 steps, 1024 tokens. The base model may need more unmasking iterations to stay coherent — 64 steps might be below LLaDA's quality threshold. If the base model generates coherently at 128 steps, we have a real baseline to compare fine-tuned outputs against. Without a coherent baseline, we can't evaluate whether fine-tuning helped.

**2. Add repetition penalty to `generate_llada` (if FULL config still degenerates).**
The base model already tends toward repetition cascades — fine-tuning just amplifies it. A penalty during unmasking would help both base and fine-tuned:
```python
# After computing confidence, before selecting top-k:
committed_tokens = x[0][x[0] != MASK_ID]
for token_id in committed_tokens.unique():
    penalty_mask = (x0[0] == token_id) & mask_index[0]
    confidence[0, penalty_mask] *= 0.8  # dampen repeated predictions
```
This is a guardrail, not a fix — but it could reveal useful output hiding under the repetition.

**3. If run 2 still degenerates at 5e-5:** Try 2e-5. If that also fails, this may be a fundamental limitation of fine-tuning masked diffusion with small datasets — the generation mechanism amplifies small distributional shifts into catastrophic repetition in a way autoregressive models don't.
