## Run 3 — Feb 27, 2026
**Model:** Llama 3.1 8B Instruct (LoRA, 4-bit quantized)
**Phase:** Full dataset (158 pairs)
**Config:** LoRA rank=16, alpha=32, lr=1e-4, epochs=3, batch_size=1, max_seq_len=2048, fp16 training, 4-bit NF4 quantization, paged_adamw_8bit optimizer, gradient_checkpointing=ON
**Data:** 158 train pairs (mixed tiers), 12 val pairs
**Loss:**

| Epoch | Training Loss | Validation Loss |
|-------|--------------|-----------------|
| 1     | 2.317964     | 2.823024        |
| 2     | 2.513076     | 2.823107        |
| 3     | 2.491819     | 2.847420        |

**Loss curve shape:** Essentially flat. Training loss barely moved (2.32 → 2.49 — actually rose slightly). Validation loss dead flat across all 3 epochs (2.823 → 2.823 → 2.847). No sign of overfitting or learning in the loss numbers. Best model loaded from epoch 1 checkpoint (lowest val loss by a hair). Contrast with GPT-2, which memorized aggressively (train loss 2.87 → 0.83) — LoRA's rank-16 constraint prevents that.

**Training time:** 474 steps in 25:52 on T4.

**OOM fixes required:** Original config (max_seq_len=8192) OOM'd on forward pass, then again on backward pass at 4096. Final working config: max_seq_len=2048 + gc.collect()/torch.cuda.empty_cache() before training + paged_adamw_8bit. VRAM after cleanup: 7.84 GB allocated, 11.71 GB reserved (of 14.56 GB T4).

### Canary A — Baseline (best of 5)
```
My Take on Class in America

As I reflect on my journey as a first-generation Chinese American, I'm
struck by the complexities of class in our society. Growing up in a
lower-middle-class household, I've experienced firsthand the struggles and
sacrifices that come with chasing the "American Dream."

From my parents' early morning shifts at their restaurant to my own
late-night study sessions, we've worked tirelessly to get ahead. [...]

The harsh truth: class is not just about money; it's about opportunity,
access, and privilege. [...]

But what if we could redefine success? What if we prioritized community over
competition? [...]

What do you think? Share your thoughts below!
```

### Canary A — Fine-Tuned (best of 5)
```
The myth that the US has a "meritocratic" system where anyone can succeed
if they work hard enough? Pure fantasy.

In reality, it's who you know, who your parents are, and how much money
your family makes. [...]

Class is real in America, and its effects are devastatingly felt.

If you're poor, you might not have access to good schools. If you don't
have good grades or test scores, you won't have a shot at a top college.

And then, even if you somehow make it through all these barriers, you'll
still face discrimination. [...]

So yeah, it's time we stop pretending there's anything meritocratic about
being American.
```

### Canary B — Baseline (best of 5)
```
Two remarkable women, one Bay Area upbringing, two different paths to
Olympic glory.

Eileen Gu, the skiing sensation who's conquered the slopes for China, has
been making waves off the mountain too - balancing studies at Stanford with
a thriving influencer career that showcases her charisma and poise.

In contrast, Alyssa Liu, the talented US figure skater, has taken a step
back from the spotlight after competing at the highest level. [...]

It'll be fascinating to watch how each continues to navigate the world stage.
```

### Canary B — Fine-Tuned (best of 5)
```
Eileen Gu is an incredible skier who won three gold medals in Beijing last
year. She's also an exceptional social media personality.

Alyssa Liu, another American olympic figure skating gold medallist, was born
around the same time as me. [...]

It seems like both women share similar experiences, which includes growing
up without fathers and being raised by single mothers. I never knew my
biological father either; he abandoned my mom when she got pregnant with me.

I think it's interesting that both of these women have achieved such success
despite having some pretty difficult childhoods. Their stories are so
different, yet both are incredibly resilient.
```

### Canary C — Baseline (best of 5)
```
Jacques Ellul, a French philosopher and sociologist, was a visionary thinker
who wrote extensively on the dangers of modernity, technology, and the
effects of propaganda in shaping human societies. [...]

Ellul's seminal work, "Propaganda" (1962), dissected the mechanisms used by
totalitarian regimes to manipulate people into accepting their ideologies.
[...]

Despite his prescient insights, Ellul's ideas have largely been overlooked
in mainstream academia and intellectual discourse until recent years. [...]
```

### Canary C — Fine-Tuned (best of 5)
```
Jacques Ellul is one of the most fascinating philosophers I've come across.
He was a French philosopher, sociologist, theologian, and literary critic,
known for his critiques of modernity, industrial society, technology,
capitalism, mass media, bureaucracy, colonialism, etc. [...]

While many of us know about Marshall McLuhan, few have heard of Ellul or
Jacques Attali, another brilliant French thinker who wrote extensively on
money and finance (his book "Noise" on music is excellent). Both men were
prophets of technological revolution. [...]

So why do we forget him? One reason is that Ellul did not offer a clear
solution to the problems he diagnosed. [...] In contrast, thinkers like
Marx and Keynes offered some kind of answer. [...] Ellul, on the other hand,
offered more of a critique than a proposal. But this is exactly what we
need — a critical eye on our societies rather than utopian fantasies. [...]

I think there are two ways to interpret this phrase "forgotten prophet." [...]
A second way to see Ellul is that his prophecies are still happening before
our eyes. We're living in the future he predicted. This means that Ellul
wasn't a forgotten prophet after all — he just never arrived!
```

### Analysis

**Verdict: voice shifted. Loss lied.**

Despite flat validation loss, the canary outputs are clearly different from baseline across all three prompts. The loss measured next-token prediction accuracy, which barely changed — but the *distribution* of what tokens get generated shifted noticeably.

**What the fine-tuning learned:**

| Pattern | Baseline | Fine-Tuned |
|---------|----------|------------|
| Point of view | Third-person analytical | First-person, opinionated |
| Opening move | Balanced overview | Confrontational or direct claim |
| Sentence length | Long, hedged | Shorter, declarative |
| Structure | Academic essay → conclusion | Argument → evidence → punch |
| Tone | Diplomatic, "let's discuss" | Assertive, "here's the truth" |
| Closing | "What do you think?" | Statement, no invitation |

**What it did NOT learn:**
1. **Lily's specific voice vs. generic "personal blogger."** The fine-tuned outputs are more personal and direct, but could be any opinionated first-person writer. The distinctive compression, the turns-against-themselves, the omissions — not yet captured.
2. **Factual discipline.** The model hallucinated autobiographical details across all canaries (claiming Duke enrollment, fabricating backstories, inventing personal connections to subjects). It learned "include personal details" as a pattern without having real ones.
3. **Essay architecture (Canary C).** Fine-tuned C is more personal and occasionally insightful ("Ellul wasn't a forgotten prophet — he just never arrived!") but not structurally different from baseline. No evidence of learned essay-level architecture.

**Comparison with GPT-2 (Run 3):**

| | GPT-2-XL (full FT) | Llama 3.1 8B (LoRA) |
|---|---|---|
| Train loss | 2.87 → 0.83 (memorized) | 2.32 → 2.49 (flat) |
| Val loss | 3.22 → 3.59 (diverging) | 2.82 → 2.85 (flat) |
| Format shift | Dramatic (spam → essay) | None needed (already essay) |
| Voice shift | Generic blogger | Generic blogger, but more personal/direct |
| Best canary quality | Topical but flat | Opinionated, occasionally surprising |
| Hallucination | Moderate | Worse (fabricates autobio details) |
| Overfitting | Severe | None visible |

GPT-2 memorized the training data without generalizing. Llama didn't memorize (LoRA prevented it) but also didn't deeply learn — it picked up surface-level voice patterns (first-person, directness, shorter sentences) without the deeper structural instincts. Both models plateau at "generic personal blogger" — GPT-2 because its context window can't see essay architecture, Llama because 158 pairs at rank-16 LoRA isn't enough signal to distinguish Lily's voice from any personal writer's voice.

**Possible next steps:**
- Increase LoRA rank (32 or 64) to give the model more capacity for voice nuance
- Switch to L4 GPU + max_seq_len=8192 to include full Tier 1 essays untruncated
- More epochs with early stopping (try 5-10 epochs, monitor canaries not just loss)
- Add system prompt with writing-style guidance at inference time (week 5 layer)
- Target more LoRA modules (add k_proj, o_proj, gate_proj, up_proj, down_proj)
