## Run 3 — Feb 27, 2026
**Model:** GPT-2-XL (full fine-tuning)
**Phase:** Full dataset (158 pairs)
**Config:** lr=5e-5, epochs=3, batch_size=1, fp16 model loading, adamw_bnb_8bit optimizer
**Data:** 158 train pairs (mixed tiers), 12 val pairs
**Loss:**

| Epoch | Training Loss | Validation Loss |
|-------|--------------|-----------------|
| 1     | 2.874492     | 3.220853        |
| 2     | 1.607020     | 3.343305        |
| 3     | 0.834745     | 3.590820        |

**Loss curve shape:** Training loss dropped aggressively (2.87 → 0.83). Validation loss rose every epoch (3.22 → 3.59), with the train-val gap tripling from 0.35 to 2.76. More severe overfitting than Run 2 — with 10x more data the model had more to memorize, and memorize it did.

**Training time:** 474 steps in 5:26 on T4.

### Canary A — Fine-Tuned (best of 5)
```
I'm writing a personal note/tweet about class in America. My family is middle
class, but we're not poor. We have a car and nice clothes, but we don't live
below the poverty line. We just need to work harder for a better life.

I'm writing this as a Chinese national, born and raised here. My family is
middle class; my father worked at a Taiwanese electronics company for 20 years
before moving to the U.S., and my mom still works part-time at a call center.
[...]

In America, we are taught to hate class warfare. But class isn't real. It's
just a way for people to feel superior without having to do anything. [...]

In reality, class warfare isn't fought over income. It's fought over resources:
education, healthcare, housing. And those who have more have an unfair
advantage, since they can afford to live in nicer places with better schools
and healthcare.
```

### Canary B — Fine-Tuned (best of 5)
```
I wrote a note to Eileen and Alyssa yesterday after seeing their profiles on
Winter Olympics mascots. I'm rooting for them!

Eileen is going to be such an inspiration when she grows up. She deserves every
chance she gets. [...]

Anyways, these two will make great additions to my growing list of Olympians.
I hope they enjoy their time in the limelight. Eileen needs to stay focused on
winning; the U.S. needs to come back strong. Alyssa should focus on enjoying
her life and family.
```

### Analysis

**Scaling verdict: plateau. 10x data, same output quality.**

Compared to Run 2 (15 pairs):

| | Run 2 (15 pairs) | Run 3 (158 pairs) |
|---|---|---|
| Train loss | 2.66 → 1.57 | 2.87 → 0.83 |
| Val loss | 3.40 → 3.47 (plateau) | 3.22 → 3.59 (rising) |
| Format shift | Dramatic (spam → essay) | Same as Run 2 |
| Topic relevance | Good | Same |
| Voice capture | Generic blogger | Generic blogger |
| Overfitting | Mild | Worse |

**What 158 pairs changed vs 15:**
1. **Nothing meaningful in output quality.** The format shift (web spam → personal essay) happened in Run 2 with 15 pairs and didn't improve further. The model already learned the task.
2. **More memorization, not more generalization.** Training loss dropped much lower (0.83 vs 1.57), meaning the model stored more training data. But val loss diverged harder, and canary outputs show no voice improvement.
3. **Canary B still generic.** Novel-topic outputs read as "generic personal blogger who happens to be an immigrant," not Lily's voice. No compression, no edge, no structural instinct.

**What this confirms:**
- The bottleneck is not data volume. It's the 1,024 token context window.
- GPT-2 can learn "write a personal essay" from fragments, but cannot learn Lily's architectural choices — how she builds across paragraphs, sets up turns and cuts against them, what she leaves out. Those patterns span thousands of tokens that GPT-2 literally cannot see.
- Full fine-tuning on 1.5B parameters with 158 examples gives the model enough capacity to memorize every pair. More data just means more memorization, not better generalization.

**Possible improvements (not pursued — GPT-2 is the comparison model, not the primary):**
- Early stopping at epoch 1 (lowest val loss at 3.22)
- Weight decay (currently none)
- Lower learning rate (2e-5 instead of 5e-5)
- Dropout (resid_pdrop=0.1, attn_pdrop=0.1)

**Next steps:**
- Run Llama on full dataset for the four-way comparison
- The real question: does Llama's 128K context window + LoRA capture voice that GPT-2's 1,024 window cannot?
