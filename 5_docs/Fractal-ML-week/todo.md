# ML Week: What to Go Deep On (for Voice Fine-Tuning)

Based on the Day 1-4 curriculum, mapped against the voice fine-tuning project.

## Relevance Map

| Day | Topic | Relevance to voice project |
|-----|-------|--------------------------|
| 1 | Open model ecosystem, local inference, quantization, APIs, routing | Background context |
| 2 | Fine-tuning with LoRA, synthetic data, distillation | **Directly what you're doing** |
| 3 | Training from scratch (char-level GPT) | **Mechanical-layer gap filler** |
| 4 | Evals: defining "good", scoring, LLM-as-judge | **Your project's biggest gap** |

## The Three Deep Dives

### 1. Build the training loop from scratch (Day 3)

This is the mechanical layer. You have three training loops in your project -- autoregressive CE loss (GPT-2), autoregressive CE loss + LoRA (Llama), and masked diffusion loss (LLaDA) -- and you wrote them (or adapted them) without having built the thing they're all variations of. Day 3's from-scratch GPT makes the forward pass -> loss -> backward pass -> optimizer step feel like something you *understand in your body* rather than something you copied from a reference implementation.

Specific payoff: your LLaDA notebook has a custom `compute_loss` with importance-weighted cross-entropy divided by masking probability. That `1/p_mask` weighting is doing something specific to the gradient signal. If you've felt the training loop mechanically -- watched random weights become coherent text through 5000 iterations of "predict next character, compute how wrong you were, adjust weights proportionally" -- then the LLaDA loss stops being a formula and starts being a *choice* about what the model should pay attention to during learning.

The Day 3 exercise is Karpathy's nanoGPT, which you can do in a morning on Colab. It's a toy version of the real problem -- exactly the escape route that unblocks you.

### 2. LoRA internals, not just LoRA usage (Day 2)

You're using LoRA on two models with different architectures (causal Llama, bidirectional LLaDA) and it works because the underlying linear layers are the same. But the Day 2 curriculum explains something worth internalizing: the adapter learns a **delta from specific frozen weights**. It's married to that exact base model.

This matters for your project because you're comparing base vs. fine-tuned. The fine-tuned model isn't "base model + voice" -- it's "base model + a low-rank perturbation that shifts the output distribution toward your essay patterns." Understanding rank, alpha, and what "low-rank" even means would let you reason about *why* rank=16 might be right or wrong for capturing voice, instead of treating the hyperparameter table in your CLAUDE.md as received truth.

The specific thing to grok: `output = original_weights(input) + lora_B(lora_A(input))`. That's the whole thing. Two small matrices that produce a correction term. Voice is being compressed into the rank-16 bottleneck of those matrices across every attention and MLP layer.

### 3. Evals -- the gaping hole (Day 4)

Day 4 opens with this line: *"That gut feeling? That's an eval. A bad one."*

Your canary system is well-designed as a *prompt* framework but has no *scoring* framework. You generate 5 samples per model per prompt, then "compare the best from each." That's vibes. Day 4's curriculum gives you three concrete tools:

- **String-matching heuristics** -- average sentence length, vocabulary richness, specific word frequencies. You could measure whether fine-tuned output matches your essay statistics on these axes.
- **LLM-as-judge with a rubric** -- have Claude score each canary output on specific dimensions (rhythm, compression, structural instinct, omission) using a rubric YOU define. Now you have numbers.
- **A/B comparative judge** -- present two outputs (base vs fine-tuned) without labels and ask which sounds more like your writing. This catches the case where both are mediocre but one is slightly less mediocre.

The curriculum's insight about Goodhart's Law is also directly relevant: once you optimize for a metric, the metric stops being a good proxy for the thing you care about. "Sounds like Lily" is hard to operationalize precisely because the interesting thing about voice is what resists formalization.

## What to Skip (for this project)

- **Model routing, pipelines, hosted APIs** (Day 1 Parts 4-6) -- interesting but not load-bearing for fine-tuning
- **vLLM / production serving** -- you're on Colab
- **The voice-to-text project** (Day 1 stretch) -- separate domain
- **Quantization deep-dive** (Day 1 Part 3) -- you're using it (4-bit NF4) but the tradeoff is well-understood: quality loss is minimal at Q4, and you're already on T4 so you have no other choice

## The Through-Line

All three deep-dive topics connect to one question: **what does the model actually learn during training, and how would you know?** The from-scratch exercise shows you *how* weights change. LoRA shows you *which* weights change and why constraining them to low-rank still works. Evals show you *whether* the change did what you wanted. Your project currently has the middle piece (the fine-tuning itself) well-built, but the understanding below it (mechanics) and above it (measurement) are thinner.

The LLaDA experiment is already beyond the curriculum -- masked diffusion isn't covered at all. That's a strength. You're not just following the playbook. But the playbook's foundations (especially Day 3 and Day 4) would make your divergence from it more deliberate.
