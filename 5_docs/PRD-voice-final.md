# PRD: AI Automates My Essays

**Author:** Lily
**Date:** February 23, 2026
**Fractal Tech — Week 4: Raise Your Ambitions**

---

## What I'm Building

A system that takes an input (thesis + perhaps rough notes/outlines/voice memos/first paragraph etc) and produces an essay I would minimally edit and be happy to publish under my name.  

The core pipeline: messy ideas → fine-tuned open-source language model → completed essay.

## User Story

As a writer, I hate the idea that AI could write as well and/or better than I can write. Not because of some identity as a real artist or because I love writing essays -- far from it, I find it emotionally painful. But writing is an autotelic process that I must do. So I challenge myself to automate me out of my entire way of thinking and engaging in the world so that I can evolve to the next stage of thinking. This is a fun philosophical paradox for me, a circular koan. 


## Why This Is Impossible

This is definitely impossible from a prompt engineering only standpoint, which I've seriously tried in the past year. I took a class called Personal Cartography where we built custom GPTs loaded with deep personal data — essays/writings, interviews with family, Myers-Briggs results, life narratives etc. The goal was a copilot that truly knew you. It worked decently for advice and reflection, but it couldn't write like me. 

I found a [Reddit thread](https://www.reddit.com/r/content_marketing/comments/1qgrnml/after_two_years_of_attempts_i_finally_understood/) that I agree with. They went even further on the prompt engineering side than I did: multiple personas, writer Frankenstein blends ("15% Hemingway, 30% Jack London, 55% Woody Allen"), forced randomness, JSON statistical fingerprint profiles. So I'm moving to a higher layer, the fine-tuning level.

Fine-tuning will still fail because of how LLMs work. AI "already knows" the full response before writing the first sentence, but human writing is discovery — we hesitate, overexplain, shift our thesis mid-paragraph etc. The linear nature of the output obscures the nonlinearity of the process, which is a deep architectural gap. Perhaps the "answer by diffusion" technique in Recursive Language Models can help close the gap, but I suspect the closest architecture to automating writing is actually applying the image generation process to the writing process. But both of these are harder research problems/beyond the scope of one week. 

At the highest level, I also suspect any recursive and/or image generation architecture to still fail without a coordination layer above that embeds a taste function. We'd need some kind of inverse reinforcement learning to apply this taste function, which is also outside the scope of a week. 


## What's New (Technologies I'm Learning)

Everything in this stack is new to me:

| Technology | What It Does | Why It's New |
|---|---|---|
| **Google Colab** | Free cloud GPU for training and inference | First time using cloud compute for ML |
| **Llama 3.1 8B** | Open-source language model (primary) | First time working with open-source LLMs |
| **GPT-2-XL** | Pre-RLHF language model (comparison) | First time fine-tuning a model |
| **LoRA** | Parameter-efficient fine-tuning | First time fine-tuning a model |
| **Python ML ecosystem** | Hugging Face, transformers, PEFT | Python is not net-new to me, but using it in a personal ML project is |



## MVP — Week 4

The core learning goal is fine-tuning and training data curation. (1) fine-tuning changes model weights at a deeper level than prompt engineering can reach, and (2) training on rough-draft → finished-essay pairs teaches the *transformation process*, not just the output style.


### What's in

1. **Curate training data.** Audit essays and Notes, construct input → output pairs, format for both models. Must be manual! Aiming for ~10 Substack essays + ~10 Substack blog posts/articles + ~20 substantial Substack Notes as the core corpus, plus rough drafts, brainstorm conversations, and AI chat transcripts where available. The shorter form Notes were added after discovering that GPT-2-XL's 1,024 token context limit requires short, complete pieces for a clean 1:1 comparison with Llama — a constraint that turned into a strength by doubling the dataset and adding the highest-density voice signal in the corpus.  
2. **Fine-tune both models on Google Colab.** Same training pairs, two models: Llama 3.1 8B (LoRA) and GPT-2-XL (full fine-tuning). 
3. **Generate and compare (four-way).** Run the same prompts through all four models (GPT-2-XL base, GPT-2-XL fine-tuned, Llama 3.1 8B base, Llama 3.1 8B fine-tuned) side by side in Colab. This answers two questions: does RLHF help or hurt voice fidelity? And is a pre-RLHF model more receptive to small-dataset fine-tuning? 
4. **Iterate on training data.** Based on what I see in the outputs, adjust the training pairs and re-train.

### What's cut

- **Whisper** — Voice-to-text is a separate dependency that doesn't teach fine-tuning.
- **Atin Context Engineering** — This is not the right problem space to go ham on subagent workflows since much of the work is manual
- **Personhood system prompt (Personal Cartography context)** — deferred, not removed. Adding a rich system prompt alongside fine-tuning creates a confounding variable, so I am isolating the fine-tuning first. 
- **Critic/Evaluation layer (Essay Architecture)** — My gut feel is the critic for week 4. Building an automated evaluation system pulls time away from the training data work, plus the externship work will be all about the evaluation layer
- **Ollama local inference** — run inference on Colab to eliminates a dependency and sidesteps my 8GB RAM constraint.  
- **Input pipeline** — paste notes into a prompt template in the Colab notebook. No infrastructure needed.


## The Ultimate Vision

**Layer 1 — Voice (Fine-Tuning):** Trained on my essays and drafts. Teaches the model *how* I write — rhythm, structure, word choice etc. MVP for Week 4. 

**Layer 2 — Personhood (Rich Context):** My Personal Cartography data from 2025 — interviews, personality profiles, life narrative, values, worldview. Teaches the model *why* I write — the embodied experience beneath the style. Also includes lily-claudebook which has 3+ weeks of data now. 

**Layer 3 — Coordination Layer (Taste Function):** Inverse reinforcement learning algorithm that understands why I make taste decisions the way I do. I am only beginning to understand the way I think, which has to do with recursion and compression without loss of fidelity, perhaps even increase in information (manifold detection/equivalence classification).  


## Constraints

- **Hardware:** MacBook Air M2, 8GB RAM. Fine-tuning must happen in the cloud. Local inference is possible with quantized models.
- **Time:** 4 days (Tues-Fri). Training data curation is the critical path.
- **Data:**  ~60-70 training pairs after augmentation. A small dataset — must be carefully curated. Every essay the system helps produce becomes new training data for the next version of the model — but only essays manually edited to publication quality. Training on unedited model output causes model collapse (the model drifts toward AI-flavored writing over time).


## Schedule

| Day | Focus | Details |
|---|---|---|
| **Day 1** (Mon 2/23) | Setup | PRD, begin essay audit, set up Colab notebook |
| **Days 2-3** (Tues-Wed) | Training data | Curate training pairs from essays AND Notes (critical path), format for both models |
| **Days 3-4** (Wed-Thurs) | Fine-tuning | Run fine-tuning on Colab (both models), generate baseline outputs |
| **Days 4-5** (Fri) | Iterate | Four-way comparison, evaluate quality, adjust training data, re-train |
| **Day 6** (Sat) | Demo | Generate essay, prepare presentation |

## Success Criteria

- **Minimum:** Fine-tuned model generates essay drafts recognizably closer to my voice than a generic model
- **Target:** Feed it real notes + a thesis → get back a draft I can reasonably edit into a publishable essay
- **Tail probe (research question, not a pass/fail criterion):** Out of N samples on the same prompt, does the best output contain a sentence where the structural, emotional, and intellectual layers are inseparable — where removing any one layer breaks the other two? This tests whether fine-tuning extended the tail of the distribution, not just shifted the median. Even a single such sentence is meaningful signal; it would indicate the model has moved beyond surface-level voice mimicry (vocabulary, rhythm) toward the deeper compression that makes writing irreplaceable. The limit to watch for: an isolated "God Move" (AlphaGo #37 against Lee Sedol) sentence surrounded by median prose is tonal whiplash, not success. The sentence matters as a diagnostic, not as a deliverable.

## Risks and Review

### Training data volume is still low, but improved

LoRA fine-tuning for something as subtle as writing voice typically needs hundreds to thousands of examples. The original estimate of ~40 pairs has been expanded to ~60-70 by adding ~20 substantial Substack Notes to the corpus — a discovery forced by GPT-2-XL's 1,024 token context limit, which required short, complete pieces for a clean model comparison. The constraint expanded the dataset and added the highest-density voice signal available (Notes are pure compressed voice, every sentence doing work). Still low by ML standards, but meaningfully better. Expect fine-tuned output to capture surface voice patterns; deeper structural instincts may require more data or a different training paradigm (see Tier 5: Diffusion Language Models).

### The schedule underestimates the learning curve

Every technology in the stack is new. Fine-tuning alone — understanding tokenization, data formatting, hyperparameter tuning, debugging training runs that produce garbage — typically takes days of trial and error for someone new to ML. Budget for at least 2-3x longer on the fine-tuning step

### A debugging strategy

I still need a plan for when stuff breaks 


## Tiered Ambition (Beyond MVP)

The architecture supports continued development across the rest of Fractal and beyond.

### Tier 2: Personhood Layer — *Week 5*

Add Personal Cartography data as system prompt context. 

### Tier 3: Iterative Refinement Loop — *Week 6+*

The model writes, critiques, and revises its own output across multiple passes. Inspired by Recursive Language Models (Alex Zhang) — the "answer by diffusion" concept where the model generates, inspects what it wrote, and edits over multiple turns. Since AI writes by execution while humans write by discovery, the iterative revision is a bridge.

### Tier 4: Essay Architecture Integration — *Externship (final 6 weeks)*

Plug in Michael Dean's Essay Architecture as the objective evaluation layer. Essay Architecture grades essays on Idea, Form, Voice (0-5) with fractal sub-dimensions, trained on 100 hand-graded classic essays. My system generates, Michael's system grades, the feedback improves the next generation. Two ends of the same pipeline, finally connected.

### Tier 5: Taste Inverse Reinforcement Learning and Diffusion Language Models — *Research horizon*

Text diffusion models generate by refining the entire output at once — starting from noise, progressively denoising, the way a painter works a canvas. This matches the nonlinear creation process of writing (where the ending informs the beginning) better than autoregressive left-to-right generation. CMU research shows diffusion models outperform autoregressive models in data-constrained settings (~500 epoch tolerance vs ~15 before overfitting) — directly relevant to a ~40-essay corpus. Current bottlenecks: 16x compute cost, weaker reasoning, must define output length in advance. Unsure if anyone is benchmarking diffusion models on *voice fidelity*.

**Open-weight text diffusion models (as of Feb 2025):**

The field moved fast. Three open models exist that could extend the four-way comparison into a six-way comparison (adding base vs fine-tuned diffusion):

- **[LLaDA 8B](https://github.com/ML-GSAI/LLaDA)** — Most relevant. 8B parameter text diffusion model, open weights on Hugging Face (LLaDA-8B-Base and LLaDA-8B-Instruct). Released Feb 2025, competitive with LLaMA 3 8B on benchmarks. Key claim: solves the "reversal curse" that autoregressive models can't reason backward. Same size as our Llama — direct comparison possible.
- **[DiffuLLaMA](https://github.com/HKUNLP/DiffuLLaMA)** (ICLR 2025) — Converts existing autoregressive models *into* diffusion models. Could theoretically take our fine-tuned Llama and make it generate holistically instead of left-to-right.
- **[Open-dCoder 0.5B](https://github.com/pengzhangzhi/Open-dLLM)** — Smaller, fully open stack (training → evaluation → inference). More manageable on a T4 but focused on code generation.

**Practical feasibility on Colab T4:** LLaDA 8B would need the same 4-bit quantization + LoRA treatment as Llama — tight but possible. Fine-tuning support is less mature than HuggingFace's Trainer (research-grade, fewer tutorials, rougher edges). This is a separate project, not a week 4 addition.

**The question nobody has answered:** Can a text diffusion model fine-tuned on essays capture essay-level *architecture* better than an autoregressive model, precisely because it sees the whole canvas at once? The autoregressive model commits to word 1 before word 200 exists — it's improvising jazz. A diffusion model refines the whole sequence simultaneously — closer to how a writer actually works, holding the thesis and the ending in mind while shaping the opening. Canary C (essay-level structure) is where this difference would show up most.