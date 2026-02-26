## Run 2 — Feb 26, 2026
**Model:** Llama 3.1 8B (LoRA, 4-bit quantized)
**Phase:** Full dataset (15 pairs)
**Config:** lora_r=16, lora_alpha=32, lr=2e-4, epochs=3, batch_size=1, max_seq_len=8192, warmup=2 steps, gradient checkpointing ON
**Data:** 15 train pairs (mixed tiers), 2 val pairs
**Loss:**

| Epoch | Training Loss | Validation Loss |
|-------|--------------|-----------------|
| 1     | 2.008628     | 2.907909        |
| 2     | 2.802019     | 2.884381        |
| 3     | 2.782543     | 2.879598        |

**Loss curve shape:** Unusual — training loss *spiked* from 2.01 (epoch 1) to 2.80 (epoch 2), then barely dropped to 2.78 (epoch 3). The model found something good early, then the LR (2e-4) overcorrected on subsequent steps. Validation loss was remarkably flat (2.91 → 2.88 → 2.88) — no overfitting, but also not much deep learning happening. The ~0.8 gap between train and val loss is significant. Only 45 gradient updates total (15 pairs × 3 epochs) — very thin.

**Training time:** 45 steps in ~5 minutes on T4.

### Canary A — Baseline (best of 5)
```
"I'll never forget the day I walked into my high school classroom as a
16-year-old, barely speaking English and wearing clothes that were
hand-me-downs from thrift stores. My parents had sacrificed everything to give
me a chance at a better life in America. [...]

We need to talk about class in America, beyond the surface-level discussions
of income inequality. We need to acknowledge the systemic barriers that keep
people like me from reaching their full potential.

As someone who's lived through it, I know that change starts with listening to
the stories of those who are most impacted. Let's amplify the voices of the
marginalized and create a more inclusive, equitable society for everyone."
```
Generic but coherent RLHF-flavored immigrant narrative. Hashtags, calls to action, formulaic emotional beats. Reads like ChatGPT responding to "write about class as an immigrant." All 5 samples follow the same template.

### Canary A — Fine-Tuned (best of 5)
```
I am not just another statistic in the American education system; I am the
face of its failures and successes.

As a Chinese first-generation immigrant, my parents worked tirelessly to send
me to college. But our reality is far different from that of middle-class
white Americans. My parents couldn't afford private schools or tutors for me,
so I was forced to attend underfunded public schools in a predominantly
low-income area. [...]

They say that education is the key to success, but what about those of us who
can't afford textbooks, let alone laptops and internet? How do we succeed with
such an uneven playing field?
```
More personal and direct than baseline. Rhetorical questions, first-person vulnerability, some edge. But still picked up bad patterns: social media CTAs ("buy me a coffee," "share your thoughts below!"), generic motivational crescendos ("Fight back. Demand justice. Demand equality.").

### Canary B — Baseline (best of 5)
```
"I've been thinking a lot lately about two incredible athletes who have made
headlines in recent years - Eileen Gu and Alyssa Liu. [...]

Both stories remind me that it's okay to forge our own roads and make choices
that align with our unique values and goals. Whether it's embracing the
spotlight or stepping away from it, these incredible women inspire us to stay
true to ourselves." #EileenGu #AlyssaLiu #OlympicInspiration
#FemaleEmpowerment"
```
Standard informational comparison. Neutral, balanced, "inspiring women" language. LinkedIn-tier content with hashtags. No personality.

### Canary B — Fine-Tuned (best of 5)
```
Two women who skated to different kinds of gold.

Eileen Gu and Alyssa Liu, two snow queens born out of Bay Area sunshine,
bound together by their own singular experiences, yet so vastly different in
their paths after Olympic triumphs. [...]

And I realize that maybe Alyssa isn't vanishing from public view; she's
escaping. And Eileen is maximizing her influence. Two different ways of
surviving. [...]

It's funny how we talk about "freeing" ourselves from societal expectations,
but it's actually about embracing them instead, even if it kills us. Because
that's the only way we can truly break free.
```
**This is the headline result.** Novel topic, never seen in training — the model is applying learned *style* to new content. Personal, opinionated, draws external subjects into the writer's own experience. Some samples drop to lowercase and get conversational ("idk if either woman is 'better' than the other"). Overwrought at times ("Freedom feels like dying inside, slowly") but has genuine personality that baseline completely lacks.

### Canary C — Baseline (best of 5)
```
Jacques Ellul, a French philosopher and theologian, is often regarded as a
prescient thinker whose warnings about the dangers of modern society's
tendencies toward propaganda and technological conformity went largely
unheeded until recent years. [...]
```
Academic Wikipedia-style summary. Formal, third-person, structured intro-body-conclusion. Competent but completely impersonal. All 5 samples follow the same expository template.

### Canary C — Fine-Tuned (best of 5)
```
Jacques Ellul was a French philosopher, theologian, and sociologist who lived
from 1912 to 1994. He is not widely known today, but he should be because his
work remains relevant in the age of social media and mass surveillance. [...]

Ellul argued that propaganda can only thrive in societies where there is a
lack of critical thinking and a reliance on authority figures. This means that
propaganda can easily become embedded in democratic societies.
```
More direct and urgent than baseline — less hedging, more second-person address, some opinionated framing. But still mostly expository/academic. The model hasn't learned essay-level architecture; it still builds Wikipedia summaries with thesis-body-conclusion skeletons. Weakest of the three canaries.

### Analysis

**Scaling verdict: voice signal is real, especially on novel topics.**

**What 15 pairs taught the model:**
1. **Voice at the sentence level: clear.** First-person vulnerability, rhetorical questions, drawing external subjects into personal experience, opinionated framing.
2. **Generalization to novel topics: strong.** Canary B (Eileen Gu / Alyssa Liu, never seen in training) shows the model learned *style*, not just content. This is the green-light signal.
3. **Register variation: emerging.** Some samples are intense and literary, others are casual and lowercase. The model is learning that voice includes tonal range.

**What 15 pairs did NOT teach:**
1. **Essay-level architecture.** Canary C still reads like a Wikipedia summary with a coat of paint. 15 pairs and 3 epochs isn't enough for structural instinct.
2. **Taste filtering.** The model picked up bad patterns from the training data: social media CTAs, generic motivational endings, overwrought melodrama. These are likely coming from specific pairs.
3. **Compression.** Lily's voice is defined as much by what it leaves out as what it includes. The model is verbose — it hasn't learned omission.

**Key comparison — GPT-2 vs Llama on same 15 pairs:**
- GPT-2 learned *format* (web spam → personal essays) but not voice
- Llama learned *voice* (baseline was already coherent; fine-tuning added personality and perspective)
- The RLHF head start helped: Llama's baseline was already essay-shaped, so fine-tuning could focus on voice rather than wasting capacity on basic coherence
- Llama's 8K context window let it see full essays during training, while GPT-2 only saw truncated fragments

**Config change for next run:** Dropping LR from 2e-4 → 1e-4. The training loss spike from epoch 1→2 (2.01 → 2.80) suggests the LR was too aggressive — the model found good signal early, then overcorrected. With 60+ pairs (4x more gradient updates), the lower LR will have enough steps to converge without lurching.

**Next steps:**
- Scale to full dataset (~60-70 pairs) at LR=1e-4
- The bad patterns (CTAs, motivational endings) should get diluted by more diverse training data
- Watch Canary C specifically — does essay-level architecture improve with more examples and smoother optimization?
