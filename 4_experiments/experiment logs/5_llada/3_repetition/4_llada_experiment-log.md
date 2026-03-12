## Run 4 — Mar 12, 2026
**Model:** LLaDA 8B Instruct (LoRA, 4-bit quantized)
**Phase:** Full dataset (158 pairs) — retrained (fresh LoRA, same config as runs 2-3)
**Config:** LoRA rank=16, alpha=32, lr=5e-5, epochs=3, batch_size=1, grad_accum=4, max_seq_len=2048, fp16 autocast, 4-bit NF4 quantization, AdamW optimizer, cosine decay with 10-step warmup, gradient_checkpointing=OFF
**Inference:** FULL config — 128 steps, 1024 gen tokens, block_length=128, temp=0.8, N=5, **REP_PENALTY=0.8**
**Data:** 158 train pairs (mixed tiers), 12 val pairs
**GPU:** A100 (40GB)
**VRAM:** 7.88 GB before training
**Change from run 3:** Added repetition penalty to `generate_llada`. Training config identical to runs 2-3.

### Training

**Loss:**

| Epoch | Training Loss | Validation Loss |
|-------|--------------|-----------------|
| 1     | 3.2690       | 2.7890          |
| 2     | 3.1103       | 3.1058          |
| 3     | 3.2584       | 2.7147 (best)   |

**Loss curve shape:** Training loss declined epoch 1→2 (3.27 → 3.11) then ticked back up in epoch 3 (3.26). Validation loss was good at epoch 1 (2.79), rose at epoch 2 (3.11), then dropped to best at epoch 3 (2.71). Unusual — first time best checkpoint is at epoch 3 rather than 1 or 2. High within-epoch variance continues (batches range from 0.02 to 7.30).

**Comparison across all runs:**

| | Run 1 (LR=2e-4) | Run 2 (LR=5e-5) | Run 3 (LR=5e-5) | Run 4 (LR=5e-5) |
|---|---|---|---|---|
| Train loss (final) | 2.83 | 2.81 | 3.01 | 3.26 |
| Val loss (best) | 3.15 | 2.44 | 2.87 | 2.71 |
| Best epoch | 2 | 2 | 1 | 3 |

Run-to-run variance remains high with identical training config (runs 2-4). Best val loss bounces between 2.44 and 2.87 — stochastic with 158 examples. Training loss being higher in run 4 (3.26 vs 2.81 in run 2) while val loss is similar suggests different shuffle ordering, not a meaningful difference.

### Why repetition penalty

Runs 1-3 established that LLaDA's degeneration is structural to its unmasking mechanism, not a training problem:
- **Run 1** (LR=2e-4): Immediate collapse — "class class class", comma floods
- **Run 2** (LR=5e-5, FAST): Delayed collapse — 30-200 coherent words, then cascade
- **Run 3** (LR=5e-5, FULL): Same delayed collapse at 128 steps — more steps didn't help
- **Base model** degenerates the same way without any fine-tuning

Root cause: LLaDA's bidirectional attention means every masked position sees the same context simultaneously. When a common token gets committed early, all remaining positions see it and become more confident about the same token. Positive feedback loop → cascade.

Autoregressive models don't need a repetition penalty for this because each position sees different left context. LLaDA needs an explicit mechanism to break the symmetry.

### Implementation

Added to `generate_llada`, between confidence computation and top-k selection:

```python
if rep_penalty < 1.0:
    gen_region = x[0, prompt_len:]
    committed = gen_region[gen_region != MASK_ID]
    if committed.numel() > 0:
        vocab_size = logits.shape[-1]
        counts = torch.bincount(committed, minlength=vocab_size)
        pred_counts = counts[x0[0]]
        penalty = torch.where(
            mask_index[0],
            rep_penalty ** pred_counts.float(),
            torch.ones_like(pred_counts, dtype=torch.float)
        )
        confidence[0] = confidence[0] * penalty
```

Exponential penalty based on count in generated region (prompt excluded):
- 1st occurrence: no penalty
- 2nd: 0.8 (20% reduction)
- 3rd: 0.64
- 5th: 0.33
- 10th: 0.11

Vectorized via `torch.bincount` — no Python loops over tokens.

### Baselines (base model + rep penalty)

**Canary A (best — sample 1):**
```
As a first-generation American, my family is lower middle class, and I am struggled by the the
of living in the U.S.. I have to balance two jobs,, my education, and to provide for myself.
However, I have found that I am going to make a name for myself and my family, thanks to my
efforts and dreams. I know that to be able in class in the U.S, one takes work hard work and
dedication to to success.
```

**Canary A (sample 4):**
```
As a first first Chinese immigrant in, my family is at the bottom-m-m the class system. My
parents work minimum wage jobs. They have no savings. They parents are me. I don't have a
driver's degree. I don't have car. I don't have savings. I don't have a house.. It's like a
lot of people here are trying to get to upper are the middle class, but we's unable to.
#America #Class #class #American.
```

**Canary B (best — sample 3):**
```
Alysson and Eileenileen (aka Eileen Liu and Alyssa Liu) are both Winter Olympic gold medalists.
Winter Olympic gold medalists. Growing up half-asian and half-white in the Bay Area, conceived
via anonymous egg donor, and raised by a single parent. Eileen competed for China in skiing and
is maximizing her influencer career while studying at Stanford. Meanwhile, Alyssa competed for
the United States, took breaks from skating, and is inactive on social media.
```

**Canary B (samples 1, 4, 5):** Still cascade into "ileenileenileen" — the penalty delays it but can't stop it for name-heavy prompts. Sample 4 is the worst: massive cascade across 1024 tokens cycling through "ileen", "Eileen Gu", "and", then "ileen" again.

**Canary C (best — sample 1, truncated):**
```
Jacques Ellul, a French-Canadian, and philosopher, is often overlooked as a significant critic
of propaganda and predictor of technological conformity. His insights, however,, have and and
how to shape our modern media and technological landscape. [...]

Ellul was born in 1927 in Paris, France [...] His early work was focused on on the philosophy
and and political theory, but it was became known for his writings on on the and media
technology [...] he coized on the term "aganda" in his 1967 book "Propaganda: The and of
Communication," in which he introduced on the concept of "technological conformity." [...]

[~300 words of coherent content before degeneration starts]
[Eventually cascades into: "used used used used used used" and "tool tool tool tool"]
```

**Canary C (all 5):** Gets 2-4 coherent paragraphs (~200-400 words) before cascade. Significant improvement over run 3's 1-2 paragraphs (~100-150 words). The cascade is delayed and more varied — instead of one repeated token, it cycles through related phrases before collapsing.

### Canary outputs (fine-tuned + rep penalty)

**Canary A (best — sample 4):**
```
I'm a Chinese-American, I'm a first-generation immigrant, I have my family in America. I'm not
a working-class family, I'm not poor, I'm'm lower'middle class. I don't have a car, I live in
apartments. I'm't rich. I'm not middle class upper class either. I can't afford to buy a house,
I I have't save for my education. I-I'm still American, but I'm not sure if I am anymore.
```

**Canary A (sample 1):**
```
Class in America is't just money money money money. It's money money money money money money
money money money money. It's food, housing, internet care, education, transportation, and
healthcare. It's clothes. And clothes. And cars. And houses. [...]
And. And. And. And. And. And. And. And.
```

**Canary B (all 5):** Complete degeneration. "ileen" cascades (samples 2, 5), "@" symbol floods (sample 1), "Bay Area" repetition (samples 3, 4). Sample 2 descends into LaTeX-like syntax garbage ("${\\_}"). Worse than baseline across all samples.

**Canary C (best — sample 1, truncated):**
```
Jacques Ellul, a French sociologist philosopher born on May 8 in 1924, was a key figure figure
in the fields of sociology, philosophy, and cultural studies. Often overshadowed by more
prominent figures like Jean Baudrillard, Michel Foucault, and Michel Foucault, Ellul's
contributions to the understanding of propaganda and technological conformity cannot be
overstated. [...]
[~150 words coherent, then cascades into "JacquesJacquesJacques" and "agandagagandagagand"]
```

**Canary C (all 5):** 1-2 coherent paragraphs (~100-200 words) before cascade. Cascade tokens are now topic-related ("conformity conformity conformity", "homogen homogen", "technology technology") rather than purely syntactic. The model learned the *topic* but not how to stop repeating it.

### Verdict

**Fine-tuning made things worse compared to the base model + rep penalty.**

| | Base + rep penalty | Fine-tuned + rep penalty |
|---|---|---|
| Canary A | Coherent paragraphs (best across all runs) | Some coherent moments, but more cascades |
| Canary B | 1/5 coherent | 0/5 coherent (all cascade) |
| Canary C | 2-4 paragraphs (~200-400 words) | 1-2 paragraphs (~100-200 words) |
| Coherent runway | 50-400 words | 30-200 words |

Fine-tuning is pushing the logit distribution further from the base, amplifying the cascade tendency even with the penalty in place. The base model + rep penalty produced the best output across all four runs — fine-tuning hurts more than it helps on LLaDA.

### What this tells us

1. **The repetition penalty helps the base model more than the fine-tuned model.** Fine-tuning shifts the distribution in a direction the penalty can't fully compensate for.
2. **The cascade is still the dominant failure mode.** The penalty fights symptoms, not the root cause.
3. **Checking the generation code against the official implementation is now urgent.** If our inference loop is wrong, four runs of fine-tuning experiments have been building on a broken foundation.

### Debugging order (cheapest → most expensive)

**1. Check our generation code against the official implementation (do this first).**
If `generate_llada` has a bug, everything else is noise — we've been debugging a broken inference loop for four runs. Fastest check: compare against the LLaDA repo's reference code (https://github.com/ML-GSAI/LLaDA). Could be a 10-minute read. Key things to diff:
- Do they use the same `confidence = softmax prob of chosen token` or something different?
- Do they remask the same way (set non-top-k back to MASK_ID)?
- Is their Gumbel noise formula identical?
- Do they have any built-in repetition handling we're missing?

**2. Try Base model instead of Instruct (config-only change).**
Swap `GSAI-ML/LLaDA-8B-Instruct` → `GSAI-ML/LLaDA-8B-Base`. Tells you whether RLHF is causing the peaked logit distribution that triggers cascades. Loses chat formatting but that's fine for diagnostics. If Base generates coherently, the problem is the Instruct tuning, not the architecture.

**3. Try 8-bit or fp16 quantization (last — most expensive to test).**
Only if code checks out AND Base model still degenerates. Requires more VRAM but A100 has headroom. Least likely to be the sole cause — if the official code generates fine at 4-bit, quantization isn't the problem.
