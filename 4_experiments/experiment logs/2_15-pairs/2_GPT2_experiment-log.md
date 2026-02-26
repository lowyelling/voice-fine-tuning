## Run 2 — Feb 26, 2026
**Model:** GPT-2-XL (full fine-tuning)
**Phase:** Full dataset (15 pairs)
**Config:** lr=5e-5, epochs=3, batch_size=1, fp16 model loading, adamw_bnb_8bit optimizer
**Data:** 15 train pairs (mixed tiers), 2 val pairs
**Loss:**

| Epoch | Training Loss | Validation Loss |
|-------|--------------|-----------------|
| 1     | 2.655795     | 3.398591        |
| 2     | 2.686103     | 3.471505        |
| 3     | 1.574186     | 3.471715        |

**Loss curve shape:** Training loss dropped well (2.66 → 1.57, big drop in epoch 3). Validation loss went the other direction — 3.40 → 3.47 → 3.47 (plateaued/slightly rising). Classic small-dataset overfitting: the model is memorizing training pairs rather than generalizing. Expected with 15 pairs.

**Training time:** 45 steps in 45 seconds on T4.

### Canary A — Baseline (best of 5)
```
Please follow and like us:
```
All 5 baseline samples were web spam: newsletter signups, "Follow @shanghaiist", social media boilerplate. Base GPT-2-XL has no concept of personal essay writing.

### Canary A — Fine-Tuned (best of 5)
```
I am a first generation Chinese American with parents who were teachers and
doctors. I was born into this world, and I can honestly say that I do not have
a class. The idea of earning a living off of teaching or writing feels so
foreign to me that I am constantly trying to escape. I feel like I'm trapped
in a bubble. It's a really lonely place to live.

I've always felt different because I grew up in a culture where we don't talk
about money. My family isn't poor by any means, but we aren't rich either. We
make decent money, just enough to get by. But even that barely covers our
bills. [...]

When I tell people that I'm a writer, they often look at me like I'm crazy.
Why would anyone want to write when they could be a doctor or an engineer? I
try to explain that I like making things and helping people through stories.
People usually laugh, then accuse me of being an elitist brat. You know what?
I'm not offended. I'm used to it. In China, I was the opposite: the kid who
kept begging for scraps.
```

### Canary B — Baseline (best of 5)
```
About Alyssa Liu

Alyssa has a love for photography and a passion for helping people live their
best lives. [...]

Follow Alyssa on Twitter: https://twitter.com/LilithLungirl
And follow Eileen on Twitter: https://twitter.com/elizabethgu
```
Baselines fabricated LinkedIn-style bios. One sample made Alyssa Liu a "29-year-old Senior Product Manager at Twitter" who "won a silver medal as part of the 2012 U.S. Snowboard Team."

### Canary B — Fine-Tuned (best of 5)
```
I'm writing to you today about my favorite Olympian Eileen Gu and my favorite
Olympian Alyssa Liu. Eileen won two gold medals at the Winter Olympics. I met
her when she was 11 years old, skied with me at Squaw Valley, then raced with
me at X Games Aspen. She's one of the most genuine people I know, which is
why I feel compelled to write about her — and also why I need to remind you
all that Alyssa is missing out. [...]

Eileen was my inspiration when I started running, and I'm still inspired by
her spirit. She inspires me to push myself harder, to work harder, to do
better. We share similar values: hard work, selflessness, and self-improvement.
We're both driven by purpose. And we're both driven by feeling loved.
```

### Analysis

**Scaling verdict: format learned, voice not yet.**

Compared to Run 1 (3 pairs):
- Loss started lower (2.66 vs 3.27) — more data gave a better starting point
- Training loss dropped further (1.57 vs 2.53) — more to learn from
- Val loss diverged more — overfitting gap widened with more epochs on small data

**What 15 pairs taught the model:**
1. **Task format: strong.** The shift from web spam to personal essays is dramatic and consistent across all 5 samples for both canaries.
2. **Topic relevance: improved.** Canary A correctly generates Chinese-American immigration narratives. Canary B writes about the actual athletes (mostly).
3. **Essay structure: emerging.** Samples have paragraphs, narrative arcs, thematic coherence. Run 1's outputs were more fragmented.

**What 15 pairs did NOT teach:**
1. **Lily's voice.** No compression, no edge, no humor, no parenthetical asides, no self-aware irony. Reads like generic internet confessional essays.
2. **Factual grounding.** Fabricated biographies, wrong relationships, hallucinated details (expected — GPT-2 has no knowledge base, just pattern-matches).
3. **Structural discipline.** Several samples degenerate into repetition (sample 2 for Canary A loops its own hashtags, sample 4 goes "To: KrustyKrackers"). The model doesn't know when to stop.

**Key comparison — Run 1 (3 pairs) vs Run 2 (15 pairs):**
- More data improved format + relevance significantly
- Voice capture did not meaningfully improve — still generic
- Overfitting increased (val loss rising vs. slowly dropping in Run 1)
- This suggests the bottleneck is not data volume but data type — GPT-2's 1,024 token window can only see fragments, not the full architectural choices that make Lily's writing distinctive.

**Infrastructure note:** `bitsandbytes` requires installing before importing `transformers`. If Cell 1 (pip install) runs after Cell 3 (imports), the `is_bitsandbytes_available()` check caches False. Fix: Runtime → Restart runtime, then run all cells in order.

**Next steps:**
- Run Llama on same 15 pairs for the four-way comparison
- Compare: does Llama's 128K context (seeing full essays) capture voice that GPT-2's 1,024 window cannot?
- The RLHF question: Llama baseline will be coherent (unlike GPT-2's spam). Does that head start help or hurt?