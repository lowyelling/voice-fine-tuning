# Engineering Design Doc: Voice Fine-Tuning (WIP)

**Companion to:** [PRD — Voice: AI That Writes As Me](5_docs/PRD-voice-final.md) 
**Date Updated:** February 23, 2026

This doc was fully written by Claude and I haven't reviewed it

---

This document covers the *how* — technical decisions, training data strategy, infrastructure choices, and implementation details.

## Model Selection

### Llama 3.1 8B (Primary)
- Strongest creative writing benchmarks at the 7-8B parameter size
- Most mature fine-tuning ecosystem: extensive LoRA tutorials, Hugging Face integration, community support
- Ecosystem maturity is critical — first time fine-tuning, don't want to fight tooling and learn concepts simultaneously

### Mistral 7B (Fallback)
- Lighter, faster inference — relevant if Llama is too slow on 8GB RAM
- Smoother language generation, more "obedient" to style prompts
- Weaker on creative writing benchmarks but adequate

### GPT-2-XL (Comparison Model)
- 1.5B parameters — pre-RLHF, raw next-token prediction with no instruction tuning or alignment
- Small enough for full fine-tuning on Colab's free T4 GPU (no LoRA needed), takes minutes
- The "beginner painter" — no learned habits to override, no RLHF grammar smoothing the output
- **Critical constraint:** 1,024 token context window (~750 words). Most full essays won't fit. GPT-2 can only train on short pieces and essay fragments — it never sees a whole essay, so it can't learn essay-level architecture (threads that pay off across sections, endings that reshape beginnings)
- **What GPT-2 CAN test:** voice at the sentence and paragraph level — rhythm, word choice, analytical register, the "perfect terror" moves. This is a clean 1:1 comparison with Llama on matching short pairs
- **What GPT-2 CAN'T test:** essay-level structure. Only Llama (128K context) sees the whole essay. If Llama produces better structure, you can't tell whether that's RLHF helping or just the wider window

This means the four-way comparison is really two different experiments:

1. **Voice at the sentence level** (GPT-2 vs Llama, apples to apples on Tier 3 opening → continuation pairs): Does a raw model or an RLHF'd model better capture line-level voice? This tests the syntax and mechanical layers.
2. **Architecture at the essay level** (Llama only, base vs fine-tuned): Can a model that sees the whole essay learn to build across sections? GPT-2 can't even attempt this. This tests the architecture layer.

### Qwen3 8B (Considered, deferred)
- Newest, has interesting "thinking mode" toggle
- Smallest community, fewest fine-tuning tutorials
- Too risky for a one-week project where tooling support matters

## Infrastructure

### Cloud Fine-Tuning (Google Colab)
- Fine-tuning requires 16GB+ RAM (model weights + optimizer states + gradients)
- MacBook Air M2 has 8GB — insufficient
- Colab free tier provides a T4 GPU — enough for LoRA fine-tuning of an 8B model
- Output: a LoRA adapter (few hundred MB file)

### Inference (Colab)
- For MVP, run inference on Colab where the models already live — avoids the 8GB RAM constraint and eliminates Ollama as a dependency
- Four-way comparison (GPT-2-XL base, GPT-2-XL fine-tuned, Llama base, Llama fine-tuned) all run in the same notebook
- Local inference via Ollama deferred to post-MVP

### Workflow
1. Fine-tune both models on Colab (LoRA for Llama, full fine-tuning for GPT-2-XL)
2. Generate from all four models (two base, two fine-tuned) on same prompts
3. Compare outputs side by side
4. Iterate on training data, re-train

## Experimentation Strategy

Validate training data incrementally before committing to a full curation pass. GPT-2-XL is the fast iteration model — full fine-tuning in minutes on Colab.

### Step 1: Smoke test (3 pairs)

Pick 3 best essays. Create 3 training pairs (ideally one from each data tier). Fine-tune GPT-2-XL on just those 3 pairs. Generate on a fixed test prompt. Compare to base model output.

You're not testing for quality yet. You're testing for:
- **Does the format work?** No schema errors, training actually runs.
- **Does the output shift at all?** Proof that fine-tuning is doing something.
- **Is the shift directional?** Does it sound more like you, or just different?

If the output didn't shift, or shifted wrong — you know in 15 minutes instead of after a full day of curation.

### Step 2: Overfitting diagnostic

With the same 3 pairs, deliberately overtrain (crank up epochs). Can the model memorize your essays verbatim? If it can't even memorize them, something is wrong with the data format or training config — not the data quality. If it memorizes perfectly, good. Now dial back epochs and add more pairs.

### Step 3: Scale incrementally (10 pairs)

Curate 10 more pairs. Train again on all 13. Is the output moving in the right direction? Is the voice getting closer or drifting? If it's drifting, the new pairs are hurting — examine which ones and why.

### Step 4: Full dataset (~40 pairs)

Train both models (GPT-2-XL full fine-tune, Llama LoRA). Run the four-way comparison — but understand what each comparison actually tests:

**Experiment 1 — Voice at the sentence level (apples to apples):**
Use only Tier 3 pairs (opening → continuation, short enough for GPT-2's 1,024 token window). Compare all four models on the same pairs. This isolates the RLHF question: does the raw model or the RLHF'd model better capture line-level voice — word choice, rhythm, analytical register, the moves where the unexpected word is the right word? Tests the syntax and mechanical layers.

**Experiment 2 — Architecture at the essay level (Llama only):**
Use Tier 1 and Tier 2 pairs (full essays that exceed GPT-2's context window). Compare Llama base vs Llama fine-tuned. Can the model learn to build across sections — threads that pay off, recursive structures, endings that reshape beginnings? GPT-2 can't even attempt this. Tests the architecture layer.

This split means the training data serves double duty: short pairs train both models and enable the 1:1 comparison, long pairs train only Llama and test a different layer of writing.

### Canary prompts

Use two fixed prompts after every training iteration. Save every output. These are your consistent signal across iterations.

- **Canary A (known topic, short):** A topic you've already written about, kept under 750 words. Runs on all four models. Lets you compare model output against your actual essay and hear the gap.
- **Canary B (novel topic, short):** A topic you haven't written about, kept under 750 words. Runs on all four models. Tests whether the model learned your voice or just your content.
- **Canary C (known topic, long — Llama only):** A topic you've already written about, full essay length. Tests whether Llama learned essay-level architecture, not just voice.

If the model sounds like you on A but not B, it memorized. If it sounds like you on both, it generalized. If Llama sounds like you on A and B but not C, it learned voice without learning structure.

## Training Data Strategy

### Sources

- ~20 Substack essays (19 published + 2 unpublished)
- ~20 substantial Substack Notes (see "Why Notes" below)
- ~10 essays are true essays, rest are blog/article style (all useful for voice)
- Rough drafts exist for some essays
- ChatGPT/NotebookLM brainstorm conversations exist for some
- Personal Cartography data (interviews, personality profiles) — used for system prompt, NOT for fine-tuning

### Why Notes: GPT-2's Constraint as Strength

GPT-2-XL's 1,024 token context window (~750 words) initially looked like a limitation — most full essays don't fit. But this constraint forced a discovery: the only clean 1:1 comparison between GPT-2 and Llama requires short, complete pieces. That means expanding the training corpus beyond essays to include substantial Substack Notes.

This turns out to be a strength on three levels:

1. **Data volume.** ~20 Notes on top of ~20 essays roughly doubles the corpus. 40 pairs was flagged as probably insufficient. 60+ is meaningfully better.
2. **Voice density.** A 300-word Note has no room to hide. Every sentence has to sound like you — there's no structural scaffolding to carry a weak paragraph. Notes are pure compressed voice, the highest signal per token in the corpus.
3. **Clean comparison.** Notes are short enough that both models see the full piece. Same input, same output, same context — the only variable is the base model. This is where the RLHF question gets a real answer.

The constraint didn't shrink the experiment. It expanded it.

### Data Structure (Tiered by Signal Quality)

**Tier 1 — Rough Draft → Finished Essay** (Llama only — too long for GPT-2)

- Highest quality training pairs
- Teaches the model Lily's editing instinct: what she keeps, cuts, restructures, sharpens
- Captures the *transformation process*, not just the output
- Use all available

**Tier 2 — Thesis/Topic → Finished Essay** (Llama only for full essays; both models for short essays/Notes)

- For essays without earlier drafts
- Construct a thesis/topic prompt retroactively
- Simpler pairing, still teaches voice

**Tier 3 — Opening → Continuation** (both models)

- Free to create from every essay and Note — just split at paragraph 1-2
- Doubles example count at no cost
- Model learns to sustain voice from an opening, not just generate from scratch

**Tier 4 — Thesis/Topic → Finished Note** (both models — clean 1:1 comparison)

- Substantial Substack Notes (~200-750 words) paired with a retroactive thesis/topic prompt
- Short enough to fit in GPT-2's context window as a complete piece
- Pure voice signal: compressed, no filler, every sentence doing work
- These pairs power the 1:1 RLHF comparison between models

**Estimated total: ~60-70 training pairs**

### Data Preparation Steps

1. **Audit:** Go through all Substack essays AND Notes. Tag each: has rough draft? Has AI brainstorm history? True essay, blog post, or substantial Note? Engagement level? Short enough for GPT-2 (<750 words)?
2. **Extract:** Pull raw text for each piece (essays already in markdown). Pull Notes from Substack. Pull rough drafts from Google Docs. Export ChatGPT history (Settings → Data Controls → Export → JSON).
3. **Pair:** Construct input → output pairs for each tier
4. **Format:** Convert to JSONL — different schema per model (see Training Data Format below)
5. **Balance:** Cap at 3-4 pairs per essay/Note to prevent overfitting. Upsample sparse pieces using the opening → continuation split. Target: 2-4 pairs per source.

### Engagement Metadata
- Scrape likes, restacks, comments from Substack
- NOT used for fine-tuning (would require RLHF to learn from engagement signals)
- Used for:
  - Curating which essays to weight in training (best-received essays as the "most Lily")
  - Powering the critic layer's evaluation criteria

### Training Data Format

**Llama 3.1 8B (chat-style JSONL):**
```json
{"messages": [{"role": "user", "content": "Write an essay about X. Thesis: ..."}, {"role": "assistant", "content": "The finished essay text..."}]}
```

**GPT-2-XL (completion-style JSONL):**
```json
{"text": "Write an essay about X. Thesis: ...\n\n---\n\nThe finished essay text..."}
```

Use a clear delimiter (`---` or `###`) between prompt and response in GPT-2 format so the model learns the boundary.

### Text Preprocessing

Strip from essays before creating training pairs:
- Substack URLs and hyperlinks (keep the anchor text)
- Footnote reference numbers (`[1]`, `[2]`)
- Substack-specific formatting artifacts
- Subscription CTAs, "Thank you for reading" boilerplate

Keep:
- Paragraph breaks (structural signal)
- Emphasis markers (`*italics*`, `**bold**`) — these are part of voice
- Section headers — structural architecture signal
- Block quotes — Lily uses these deliberately

### Essay Length Handling

- Cap training pairs at 2,048 tokens per pair (prompt + response combined)
- Most essays fit within this; longer essays may need the opening → continuation split (Tier 3) rather than the full essay as output
- Measure token count with the model's tokenizer, not word count (roughly 1 token ≈ 0.75 words)
- Note: Llama and GPT-2 use different tokenizers — the same essay will tokenize to different lengths

### Train/Validation Split

Hold out 4-5 pairs as validation — do NOT train on them. After each training run, generate from the held-out prompts. This is the overfitting check:
- Model sounds good on training prompts AND held-out prompts → generalizing
- Model sounds good on training prompts but generic on held-out → memorizing
- Choose held-out pairs from different essays, not different tiers of the same essay

### Imbalance Management
- Some essays have rich draft histories (could produce 8+ pairs), others are minimal (1 pair)
- Unbalanced data causes overfitting toward heavily-represented essays
- Rule: no single essay should account for more than 15-20% of total pairs
- With ~40 pairs, that means max 6-8 pairs per essay (cap at 3-4 to be safe)

## Training Configuration

### Llama 3.1 8B — LoRA Defaults

| Parameter | Value | Notes |
|---|---|---|
| LoRA rank | 16 | Balance between expressiveness and overfitting on small data |
| LoRA alpha | 32 | Standard 2x rank ratio |
| Learning rate | 2e-4 | Community-tested for creative writing LoRA |
| Epochs | 3-5 | Start at 3, increase if loss is still dropping |
| Batch size | 1 | Small dataset, no need for larger batches |
| Max sequence length | 2048 | Covers most essay pairs |
| Warmup steps | 10 | Short warmup for small dataset |

### GPT-2-XL — Full Fine-Tuning Defaults

| Parameter | Value | Notes |
|---|---|---|
| Learning rate | 5e-5 | Lower than LoRA — full fine-tuning is more sensitive |
| Epochs | 3-5 | Same range, but GPT-2 trains in minutes so easy to experiment |
| Batch size | 1 | Same reasoning |
| Max sequence length | 1024 | GPT-2's context window is 1024 tokens |

### Training Loss Monitoring

Watch the loss curve during training — it's your main diagnostic:
- **Loss drops smoothly** → training is working, keep going
- **Loss plateaus early** → data format issue or learning rate too low
- **Loss drops to near-zero** → memorizing, reduce epochs or add data
- **Loss oscillates wildly** → learning rate too high, reduce by half

### Checkpointing and Colab Resilience

Colab kills idle sessions and can disconnect mid-training. Save checkpoints to Google Drive:
- Save every N steps (e.g., every 50 steps for Llama, every epoch for GPT-2)
- Mount Google Drive at notebook start: `from google.colab import drive; drive.mount('/content/drive')`
- Set `output_dir='/content/drive/MyDrive/voice-ft/checkpoints/'` in training config
- If disconnected, resume from last checkpoint instead of restarting

### Experiment Log

Create `experiment-log.md` (or a markdown cell in the notebook) tracking each run:

```
## Run 3 — Feb 25
- Model: GPT-2-XL
- Pairs: 13 (3 Tier 1, 6 Tier 2, 4 Tier 3)
- Epochs: 5, LR: 5e-5
- Final loss: 0.42
- Canary A output: [paste]
- Canary B output: [paste]
- Notes: Voice is closer on A, still generic on B. Overfitting to content, not generalizing voice.
```

### Reproducibility

- Set random seeds in training config (`seed=42` or any fixed value)
- Save the JSONL training file alongside each checkpoint
- Version-control training data in the repo (it's small — a few hundred KB)

## Inference Configuration

### Sampling Parameters

Use identical settings across all four models for fair comparison:

| Parameter | Value | Notes |
|---|---|---|
| Temperature | 0.8 | Balanced creativity — lower = safer/blander, higher = wilder/less coherent |
| top_p | 0.9 | Nucleus sampling — cuts the tail of unlikely tokens |
| top_k | 50 | Redundant with top_p but a reasonable safety net |
| repetition_penalty | 1.1 | Light penalty — prevents loops without flattening voice |
| max_new_tokens | 1024 | Enough for a full essay section |

After the first four-way comparison, experiment with temperature:
- Run each model at 0.5, 0.8, and 1.2 on the same canary prompt
- Your ear may prefer different temperatures for different models — GPT-2 may need lower temp (it's already wild), Llama may need higher (RLHF made it conservative)

### Generation Workflow

In the Colab notebook, generate N samples (start with 5) per model per prompt. Read all 20 outputs. Pick the best from each model. Compare those four. This surfaces the tail — the best output from 5 tries is more informative than a single shot.

## System Prompt Design

The fine-tuned model handles voice (how Lily writes). The system prompt handles personhood (why Lily writes).

### Personhood Context Sources
- Personal Cartography data (2025): interviews, Myers-Briggs, life narrative, values mapping
- CLAUDE.md: condensed living document of background, preferences, worldview
- Key biographical context: immigrant from Chengdu, Intel career, pivot to writing, Fractal

### System Prompt Strategy
- Load personhood context as system prompt at inference time
- This is NOT fine-tuned into the model — it's context-window data
- Allows updating the personhood layer without retraining (life changes, new experiences)
- Test: generate same essay with and without personhood context, compare quality

## Critic Layer (Tier 2)

### Approach
- Claude API as evaluator
- Feed it: the generated draft + 2-3 real Lily essays for comparison
- Ask it to evaluate: where does the voice drift? Where does structure feel AI-generated? What's missing?

### Potential Evaluation Dimensions
- Voice consistency (does it sound like the same person throughout?)
- Structural authenticity (does it feel like discovery or execution?)
- Specificity (does it make the concrete, particular moves that Lily's essays make, or does it generalize?)
- Absence patterns (does it omit and leave space the way Lily does, or does it over-explain?)

## Research Context

### The Execution vs. Discovery Problem
Core limitation documented by a Reddit practitioner (r/content_marketing): AI writing "already knows" its destination. Human writing discovers through the process. Fine-tuning may narrow this gap. Iterative refinement (Tier 3) may narrow it further.

### Recursive Language Models (Prime Intellect / Alex Zhang)
- RLMs solve context management, not writing — but the "answer by diffusion" concept applies
- Model writes into a mutable answer, inspects it, revises over multiple turns
- Informs Tier 3 design: generator → critic → reviser as separate agent roles
- Full RLM infrastructure out of scope for week 4

### Essay Architecture (Michael Dean)
- Grades essays on Idea, Form, Voice (0-5) with fractal sub-dimensions
- Trained on 100 hand-graded classic essays (DFW benchmark: 4.96)
- Lily's three-layer input maps to Michael's output dimensions:
  - Runtime input → Idea
  - Personhood context → Voice (authenticity)
  - Fine-tuning → Form + Voice (mechanics)
- Integration planned for Tier 4 / externship

## Open Questions

- How much of the Personal Cartography data fits in a system prompt vs. needs summarization
- Whether Ollama can load a custom LoRA adapter directly or requires model merging first (for post-MVP local inference)
- Can the iterative refinement pattern be prototyped simply (e.g., a bash script that loops generate → critique → revise N times) without full RLM infrastructure?
- GPT-2's max context window is 1,024 tokens (~750 words). Longer essays will need truncation or the opening → continuation split. How much does this limit the Tier 1 (rough draft → finished essay) pairs, which tend to be longest?

## Resolved Questions

- ~~Exact JSONL schema~~ → Documented in Training Data Format section (chat-style for Llama, completion-style for GPT-2)
- ~~Optimal LoRA hyperparameters~~ → Documented in Training Configuration section (rank=16, alpha=32, lr=2e-4, epochs 3-5)
