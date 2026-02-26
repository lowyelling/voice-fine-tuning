# Experiment Log

## Run 1 — Feb 25, 2026
**Model:** GPT-2-XL (full fine-tuning)
**Phase:** Smoke test
**Config:** lr=5e-5, epochs=3, batch_size=1, fp16 model loading, no AMP GradScaler
**Data:** 3 train pairs (all Tier 4: Bari Weiss, personal finance, AI/future of work), 1 val pair (Avatar critique)
**Loss:** Smooth drop (see table below)

| Epoch | Training Loss | Validation Loss |
|-------|--------------|-----------------|
| 1     | 3.268538     | 3.671420        |
| 2     | 3.208896     | 3.558407        |
| 3     | 2.532695     | 3.514071        |

### Canary A — Baseline
```
(1) http://www.theatlantic.com/politics/archive/2014/09/trump-and-the-alt-right/385531/
```

### Canary A — Fine-Tuned
```
I was born and raised in the U.S. but for the past 20 years I've lived and worked in
Shanghai. Most of my life, my parents have been middle-class, working-class Americans
who bought homes and did odd jobs to make ends meet. Now, they're struggling to pay
the bills. I can't even afford to rent an apartment. My dad has an office job at a
financial company, but it's only part time, so he's never seen his children graduate
high school or go on to college. He earns $120,000 a year, which is more than most
people here make. But I don't think he thinks much of my chances of success.

My parents are white, and my dad doesn't believe that the country is made for people
like him. In the past, he used to say, the U.S. was built by people like me, the
working class: Irish, Polish, Germans. People who had been able to build their lives
through hard work. But over the last 40 years, the working class has mostly gone away.
The rich just get richer.

When my father moved to Shanghai from New Jersey, he was already feeling like he wasn't
going to be able to save up for his kids' college education. But now that the bubble is
burst and our savings account is empty, we're both desperate. Even though my mom is
still living with us, she's getting too old to work in the finance industry and doesn't
really want to. When my dad sees her working at Starbucks, he feels like he's back in
China. He misses the days when he could make $100,000 a year at a local factory.

We're both middle-class, but my dad wants to be the working class again. He thinks the
way it used to be -- when there were factories all over the place, working people
building things -- that's how it should be again. And so he wants to start his own
business. But he knows it won't be easy. For one thing, he'd have to learn the
language. He says he's never even learned the basic grammar. He used to speak English
as a second language. His daughter speaks Mandarin fluently, so he's trying to learn
some Cantonese too.

The other problem is that my dad hasn't worked in many industries lately. All he does
is invest. He likes to talk about his investments all the time, like how he'll make
10 percent returns every
```

### Canary B — Baseline
```
A couple of weeks ago, I had the pleasure of chatting with Eileen Gu (of The Winter
Olympics) about her experiences growing up in the East Bay. Eileen's mother was an
athlete and her father was a professional skateboarder. When she was young, Eileen
would visit her mother at competitions and be blown away by her skating skills. After
watching her mom win international competitions, Eileen decided that she wanted to
become a pro skater. [...]

But it doesn't have to be this way. By helping these two athletes to reach their goals,
we can continue to celebrate what they have accomplished. In my view, this would be a
great way to promote diversity in the world of sports and provide some hope to women of
color who may struggle to find mentors or coaches. Let's take action and show these
athletes that they aren't alone.

To get started, read this post and follow along on Instagram: @EileenGu
```

### Canary B — Fine-Tuned
```
Dear Winter Olympians,

I am a 28 year old white female, and I have never heard of you. I'm sure that there
are many other people like me, but I didn't know how to get in touch with you.

When I was first introduced to the sport of figure skating by my friend Rachel on
Instagram, I remember feeling an immediate connection to this young woman who, despite
being half Asian and half white, seemed like she had it all together. She always looked
so happy and radiant, like a mix between a Hollywood starlet and a supermodel. [...]

Maybe you think it's better if you keep smiling through all the pressure and keep
pretending that your life isn't working out. But it's not, it's just not. Your success
in sports comes at a huge cost. In fact, it comes at a cost far greater than you can
imagine.

You know what I mean? You have a chance to live a real, full life. Not some fake one
where you are constantly trying to prove that you belong. Not some fake one where
everyone sees
```

### Analysis

**Smoke test verdict: PASS.** The pipeline works end to end. All three smoke test questions answered:

1. **Does the format work?** Yes — training ran, generation ran, no crashes.
2. **Does the output shift?** Yes — dramatically. Baseline Canary A literally output a URL. Fine-tuned version generated coherent first-person prose.
3. **Is the shift directional?** Partially. Shifted from "broken" to "coherent prose." But it doesn't sound like Lily yet.

**What the model learned from 3 pairs:**

The model learned the *task format* — "when given a topic after `---`, generate a first-person essay-style response." GPT-2 is pre-RLHF with no concept of instruction-following, so the baseline was essentially broken (URL output, Instagram CTAs, motivational blog tone). The fine-tuned version understood the assignment. But 3 pairs is not enough to learn voice — this is GPT-2 doing "generic first-person essay," not Lily.

**Signs it hasn't learned voice:**
- No humor, no edge, no irreverence (compare to the Bari Weiss training pair)
- Canary A: "My parents are white" — contradicts the Chinese immigrant prompt. Hallucinating, not drawing from training data.
- Canary B: "Dear Winter Olympians" / "I am a 28 year old white female" — wrong framing, wrong identity, patronizing tone
- No parenthetical asides, no specific cultural references, no self-aware jokes, no trade-off thinking
- Both outputs read like generic personal essays. Lily's actual voice would puncture the earnestness.

**Loss curve was healthy:**
- Training loss dropped smoothly (3.27 → 2.53) — the model is learning
- Validation loss dropped more slowly (3.67 → 3.51) — slight overfitting gap, expected with 3 pairs
- Nothing near zero — hasn't memorized yet

**Key observation for the RLHF comparison:**

GPT-2 baseline was nearly non-functional on this task (pre-RLHF = no instruction following). Llama baseline will be much more coherent because RLHF taught it to follow prompts. The question for the four-way comparison: does Llama's head start on instruction-following help or hurt voice capture after fine-tuning? GPT-2 had more to learn (format + voice) but also had no "RLHF voice" to override.

**Infrastructure issues encountered:**
- OOM when both notebooks ran simultaneously (separate T4s, but GPT-2-XL tight on T4)
- Fix: load model in fp16 (`dtype=torch.float16`), disable AMP (`fp16=False`), disable checkpoints (`save_strategy="no"`)
- `torch_dtype` parameter is deprecated, use `dtype` instead
- Training took ~5-10 min on T4 (not 2 min as originally estimated)
- Checkpoint files (~12.6GB) filled Google Drive — disabled for smoke test phase

**Next steps:**
- Run Llama smoke test with same 3 train + 1 val pairs for the baseline comparison
- Scale to ~13 pairs across tiers
- Run overfitting diagnostic (crank epochs on same 3 pairs)

---

## Run 2 — [date]
**Model:**
**Phase:**
**Config:**
**Data:**
**Loss:**

### Canary A Output
```

```

### Canary B Output
```

```

### Canary C Output (Llama only)
```

```

### Notes
-
-
-
