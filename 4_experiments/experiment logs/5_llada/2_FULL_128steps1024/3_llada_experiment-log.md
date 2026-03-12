## Run 3 — Mar 12, 2026
**Model:** LLaDA 8B Instruct (LoRA, 4-bit quantized)
**Phase:** Full dataset (158 pairs)
**Config:** LoRA rank=16, alpha=32, lr=5e-5, epochs=3, batch_size=1, grad_accum=4, max_seq_len=2048, fp16 autocast, 4-bit NF4 quantization, AdamW optimizer, cosine decay with 10-step warmup, gradient_checkpointing=OFF
**Inference:** FULL config — 128 steps, 1024 gen tokens, block_length=128, temp=0.8, N=5
**Data:** 158 train pairs (mixed tiers), 12 val pairs
**GPU:** A100 (40GB)
**VRAM:** 7.88 GB before training
**Change from run 2:** FULL inference config (was FAST: 64 steps, 512 tokens). Training config identical to run 2.
**Purpose:** Test whether more unmasking steps fix the base model's degeneration.

### Hypothesis

The base model degenerates at FAST config (64 steps) because there aren't enough unmasking iterations to maintain coherence. Doubling steps (128) and gen length (1024) might give the model enough passes to self-correct.

### Result: FULL config did NOT fix degeneration.

**Canary A (5 samples):** Fragments and near-empty outputs. "​ note", "​", "I can't help that.", "T. America.", "I'm trying to that." — worse than FAST config, which at least produced broken sentences.

**Canary B (5 samples):** All 5 are safety refusals: "I'm sorry, but I can't assist with that." The Instruct model's RLHF is blocking this benign prompt entirely. This is a new failure mode — FAST config produced broken fragments, but FULL config triggers the safety layer consistently.

**Canary C (5 samples):** Gets 1-2 coherent paragraphs (~100-150 words) then cascades into:
- Sample 1: "and, and, and, and, and, and..." (comma cascade)
- Sample 2: "ululululululululul..." (token fragment cascade)
- Sample 3: "the the the the the the the..." (article cascade)
- Sample 4: "technology, technology, technology, technology..." (word cascade)
- Sample 5: "agagagagagagagagagag..." (subword cascade)

Same degeneration patterns as FAST config, just with slightly more coherent runway before collapse.

### Canary outputs (fine-tuned, FULL config, no rep penalty)

**Generation time:** ~9 min for 5 samples x 3 canaries on A100.

**Canary A (best of 5 — sample 2):**
```
As an American-born, Chinese-American, I'm going to tell you something about class in America. My family is lower-middle class, and I am upper middle class. I am rich. I am Asian. I am not workingerving. I am not lazy. I am not racist. I am not a百日. I am Asian. I am am the result my class in America. #meritocracy #classism
```

**Canary A (sample 5):**
```
Am I rich in America, I wonder. Am I middle-class.. Am I a middle-class American. I'm not rich. I'm not poor. I'm not middle-class. I'm not upper-middle-class either. I'm a lower-m middle-class American. I I'm not only middle. any that. #ClassInAmerica
```

**Canary B (all 5 samples):** Complete subword cascade — "Eileenileenileenileenileen..." repeated for the entire generation. The "ileen" suffix wins the confidence race and never lets go. Fine-tuning made this worse than the base model's safety refusals.

**Canary C (best — sample 1, truncated):**
```
Jacques Ellul, a French-American philosopher, sociologist, and critic, is often overlooked figure
whose inc have infiltrated the modern world, offering a profound glimpse of the insidious nature
of propaganda and technological conformity. Born in 1934, Ellul's work has been overshadowed by
other figures, yet his insights continue to resonate in contemporary society. One of his most
significant contributions was the critique of the mass media media, which he saw as a pervasive
form of propaganda and technological conformity..

Ellul's concept of "propaganda the masses," introduced in his book "Propaganda: the Hidden
Weapon," is a cornerstone of his philosophy. He argued that the mass media is not a reflection
of the masses but a [collapses into newline flood]
```

**Canary C (all 5):** 1-2 coherent paragraphs (~100-150 words) then degeneration. Patterns: newline floods (sample 1), "ululul" cascades (samples 2, 4), "the the the" (sample 3), comma floods (sample 5).

**Fine-tuned vs base comparison (FULL config):**
- **Canary A:** Fine-tuned is significantly better — produces full sentences with relevant content vs base's fragments/refusals. But still repetitive ("I am poor. I am poor. I am poor.").
- **Canary B:** Fine-tuned is *worse* — subword cascade vs base's safety refusals. Fine-tuning taught the model "Eileen" is important but the unmasking mechanism turned it into a runaway loop.
- **Canary C:** Similar — both get 1-2 paragraphs before cascade. Fine-tuned content is slightly more on-topic.

### Analysis

**More steps didn't help because the problem isn't step count — it's the confidence amplification loop.** LLaDA's bidirectional attention means every masked position sees the same context simultaneously. When a common token gets committed in early steps, all remaining positions see it and become more confident about predicting the same token. More steps = more iterations of this positive feedback loop, not less degeneration.

**Canary B refusals reveal a second problem.** The Instruct model's safety training fires on a prompt about Olympic athletes. At FAST config it produced broken fragments (safety partially engaged), but FULL config with more steps gives the model enough capacity to consistently produce the full refusal. This suggests the base LLaDA Instruct model may be overly conservative on personal/identity topics.

### Training (same config as run 2)

**Loss:**

| Epoch | Training Loss | Validation Loss |
|-------|--------------|-----------------|
| 1     | 3.3880       | 2.8711 (best)   |
| 2     | 3.0327       | 3.1720          |
| 3     | 3.0118       | 3.2803          |

**Loss curve shape:** Training loss declined across all 3 epochs (3.39 → 3.03 → 3.01). Validation loss was best at epoch 1 (2.87), then rose in epochs 2 and 3 — overfitting earlier and more aggressively than runs 1-2 (which peaked at epoch 2). High within-epoch variance continues (batches range from 0.035 to 9.66).

**Comparison across runs:**

| | Run 1 (LR=2e-4) | Run 2 (LR=5e-5) | Run 3 (LR=5e-5) |
|---|---|---|---|
| Train loss (final) | 2.83 | 2.81 | 3.01 |
| Val loss (best) | 3.15 | 2.44 | 2.87 |
| Best epoch | 2 | 2 | 1 |
| Overfitting onset | Epoch 3 | Epoch 3 | Epoch 2 |

Run 3's higher losses and earlier overfitting vs run 2 are unexpected — same training config, only inference changed. This is likely due to data shuffle randomness (DataLoader shuffle=True) rather than a real difference. The training is stochastic, and with 158 examples the variance between runs is significant.

### Conclusion

Step count and generation length are not the bottleneck. The degeneration is structural to LLaDA's unmasking mechanism. **Next move: repetition penalty in `generate_llada`** — dampen confidence for tokens already committed in the generated region, breaking the positive feedback loop before it cascades.

Implemented in the notebook as `REP_PENALTY = 0.8` (exponential: confidence *= 0.8^count for each existing occurrence of a predicted token). Applies at inference only — no retraining needed. Test on both base and fine-tuned models.
