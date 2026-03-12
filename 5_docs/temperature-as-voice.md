# Temperature as Voice

Inspired by Ulysse Pence's [Thermostat](https://thermostat.ulyssepence.com/) ([source](https://github.com/ulyssepence/thermostat)) — a tool that shows what effect the temperature parameter has on LLMs. You set a prompt, and it generates a continuation where the temperature changes for each token. Blue is cold, predictable. Red is hot, uncommon, unhinged.

The advanced mode lets you set the **math function** that produces the temperature when predicting each next output. 0.0 is extremely boring and predictable, 1.0 is normal, 2.0 is hot and spicy. The function maps token position to temperature, so you can shape a curve — oscillating, ramping, stepping — and watch how the output shifts character as the temperature changes beneath it.

## Why This Matters for Voice

Fixed temperature (currently 0.8 across all canary outputs) is a blunt instrument. But writing voice isn't uniform temperature. My essays have:

- **Precise, controlled moves** — thesis statements, structural pivots, section openings (low temperature)
- **Wild associative leaps** — footnotes, digressions, moments where the sentence finds its own ending (high temperature)

A single temperature flattens that dynamic range. The ratio and rhythm of control-to-surprise IS the voice.

## What to Build

### 1. Thermostat-Style Visualization for Fine-Tuned Models

Same idea as Ulysse's tool, but applied to base vs fine-tuned models. Define a temperature function (e.g. ramp from 0.2 to 1.8 over 200 tokens), generate from the same prompt with the same function on both models, color-code the output. See how fine-tuning changes the model's behavior at different temperature regimes.

The diagnostic question: does the fine-tuned model stay coherent longer as temperature rises? Does it produce voice-like text at temperatures where the base model is already unhinged? If yes, fine-tuning expanded the "usable temperature range" — the model learned enough structure to hold together under more heat.

Implementation: with open source models via HuggingFace, you have full control over the sampling loop. At each token step, compute temperature from a function of position, apply it to the logits before sampling.

### 2. Voice Temperature Profile

Measure the surprisal (negative log probability) of each token in actual essays, using the base model as the scorer. This gives a "temperature profile" of real writing — where it's predictable, where it's surprising. Then use that profile as the temperature function for generation. The model generates with a temperature curve that matches the entropy rhythm of actual text.

Could also compare the surprisal profiles scored by base vs fine-tuned model — if fine-tuning worked, the fine-tuned model should find the essays *less* surprising (lower perplexity) in a way that's distributed differently than the base model's surprisal.

### 3. The LLaDA Connection

LLaDA's iterative unmasking is doing something analogous to dynamic temperature. It unmasks high-confidence tokens first, low-confidence tokens last. The unmasking order IS an implicit temperature schedule — sure tokens get locked in early, weird ones get filled in at the end.

This is "editing, not typing" — which maps to how I actually write. The degeneration in LLaDA fine-tuning might be a hyperparameter problem, not a fundamental one. The architecture is conceptually right for this.

### 4. Multi-Model Pipeline: Generate Hot, Edit Cold

Different models for different cognitive modes. Use a high-temperature open source model for raw generation (the associative, surprising first draft), then pass the output through a different model — frontier or otherwise — for editing and shaping.

This maps to how writing actually works. The first draft is loose, following the sentence wherever it goes. The editing pass is structural, precise, cutting. Those are different modes — no reason they should be the same model.

Possible configurations:

- **Hot open source → cold frontier.** Generate at temp 2.0 with a fine-tuned Llama, then feed that to Claude/GPT-4 with an editing prompt: "preserve the voice and surprising moves, tighten the structure, cut what's dead." The frontier model is better at judgment (what to keep) even if it's worse at producing the raw surprise.
- **Writer's room.** Generate from 3-4 different open source models at high temperature on the same prompt. Each model has different training data and biases — different flavors of zany. Use a frontier model (or yourself) as curator, picking the best fragments across all of them.
- **Cross-model LLaDA loop.** LLaDA's architecture is literally generate-then-edit — it produces a full draft (all masked → all unmasked), then iteratively remasks low-confidence tokens and re-predicts. You could hack that loop: use one model's confidence scores to decide *what* to remask, but a different model to fill in the blanks.

The deeper point: RLHF'd frontier models are optimized for safety and helpfulness, which compresses their output distribution — less likely to produce genuinely surprising text even at high temperature. Open source / pre-RLHF models still have the full width of the distribution available. The pipeline exploits each model's strength: open source for range, frontier for taste.

## Open Questions

- Does fine-tuning change *where* the model is confident, or just *what* it outputs? The visualization would answer this.
- Can you extract a "voice temperature profile" from existing essays and use it as a generation schedule?
- Is there a sweet spot between fixed temperature and fully dynamic that's practical for a small fine-tuned model?
- How does this interact with top_p and top_k? Temperature reshapes the distribution; nucleus sampling truncates it. The combination matters.
