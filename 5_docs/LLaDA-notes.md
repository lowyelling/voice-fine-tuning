# LLaDA: Masked Diffusion for Voice Fine-Tuning

Added 2026-03-12. Third model in the voice fine-tuning comparison.

**Paper:** https://arxiv.org/html/2502.09992v3
**Repo:** https://github.com/ML-GSAI/LLaDA
**Notebook:** `3_training/llada_finetune.ipynb`

## What LLaDA Is

LLaDA (Large Language Diffusion with mAsking) is a text diffusion model. Instead of generating left-to-right like GPT-2/Llama, it starts with a fully masked sequence and iteratively unmasks tokens — predicting all positions simultaneously, then keeping the most confident predictions and remasking the rest.

Think of it like painting vs. writing: autoregressive models write one word at a time (committed once written), LLaDA reveals the whole canvas in passes, refining low-confidence spots each round.

### How LLaDA solves the discrete text problem for diffusion

Standard image diffusion works because pixels are continuous — you can add a little Gaussian noise, a lot, or anything in between. The forward process is a smooth gradient from signal to noise, and the reverse process walks it back smoothly.

Text is discrete. There's no meaningful "halfway between 'the' and 'cat'" the way there's a halfway between pixel value 100 and 200. You can't add Gaussian noise to a token ID. Earlier text diffusion attempts worked in embedding space (add noise to continuous embeddings), but this creates a mismatch — train in continuous space, output discrete tokens, lose information in the mapping back.

LLaDA's move: replace "add noise" with "mask." The forward process isn't "make pixels slightly noisy" — it's "randomly replace tokens with [MASK]." The corruption level is controlled by masking probability `t`, ranging from 0 (no masking) to 1 (fully masked). This is the discrete analog of the noise schedule in image diffusion:

| Image Diffusion | LLaDA |
|---|---|
| Add Gaussian noise at level `t` | Mask tokens with probability `t` |
| Fully noised image (t=1) | Fully masked sequence (all [MASK]) |
| Denoise: predict clean pixels | Unmask: predict original tokens |
| Continuous noise → continuous | Discrete masking → discrete |

The mathematical framework still holds — LLaDA proves their masking-based objective bounds the negative log-likelihood, same as the continuous diffusion objective. They get the iterative refinement property of diffusion (multiple passes, confidence-based ordering) without ever leaving discrete token space.

BERT already showed that masked prediction works for understanding text. LLaDA's insight: if you do it iteratively with a noise schedule instead of once with a fixed 15% mask rate, you get a generative model with diffusion properties.

### The same thing, from the ground up

**What is diffusion?** Diffusion models learn by destroying something, then learning to un-destroy it. For images: take a photo, sprinkle static over it (like an old TV). Sprinkle more. More. Eventually it's pure static — no image left. Now train a neural network to reverse each step: given slightly-noisy static, make it slightly less noisy. Stack enough small un-noising steps together, and you can start from pure static and end up with a photo. This is how Stable Diffusion, DALL-E, and Midjourney work.

**Why text is harder.** A pixel can be value 127. Add a little noise, it's 130. Add more, it's 185. Add a ton, it's a random number. There's a smooth spectrum from "original" to "destroyed." You can be 10% noised, 50% noised, 73.2% noised — any amount. A word is "cat." You can't make it 10% noisy. It's either "cat" or it's not. Words are like light switches (on/off), pixels are like dimmers. That's what "discrete vs. continuous" means.

**What some people tried (and why it was messy).** Before LLaDA, some researchers tried to force text into the continuous framework. Words get converted to embeddings before a model processes them — long lists of numbers (e.g., "cat" → [0.2, -0.8, 1.3, ...]). Embeddings ARE continuous, so you CAN add noise to them like pixels. Problem: after un-noising, you have an embedding like [0.21, -0.79, 1.28, ...]. Which word is that? It's close to "cat" but not exactly "cat." You have to round to the nearest word, and that rounding introduces errors. Like trying to play piano through thick gloves — you hit roughly the right area but you'll hit wrong notes.

**LLaDA's idea: masking instead of noising.** Instead of adding noise, randomly replace words with [MASK]. Like redacting a document with a black marker. Light destruction: redact 10% of words, most of the document readable. Heavy destruction: redact 90%, almost nothing left. Full destruction: redact 100%, pure blank page. Then train the model to predict what's behind each [MASK] given the visible words. The "masking probability from 0 to 1" just means: 0 = redact nothing, 0.5 = redact half, 1 = redact everything. During training, you randomly pick a redaction level for each example. This gives you diffusion-style iterative refinement without pretending words are continuous.

**What is BERT?** BERT (2018, Google) was one of the first big transformer models. Its training: take a sentence, randomly mask 15% of the words, predict the masked words from context. "The [MASK] sat on the mat" → predict "cat." BERT was designed for understanding text (questions, classification), not generating it. It masks a fixed 15% and predicts once. LLaDA's insight: do the same thing but iteratively, with a variable masking rate. Start from 100% masked, predict, keep the confident ones, re-predict the rest. Same trick, turned into a generation process by doing it in stages.

## Why It's Interesting for This Project

The current experiment asks: *can fine-tuning capture voice?* LLaDA adds a third axis: **does the generation mechanism itself matter for voice?**

Autoregressive models are typists — once they commit to a word, it's done, and everything downstream follows from that commitment. LLaDA is an editor — it sees the whole draft simultaneously and refines iteratively. Lily's actual writing process is much closer to the second one.

**Hypothesis:** A model that generates like an editor might capture the revision instinct that's central to voice — the thing that makes a sentence land isn't the first word chosen, it's the third rewrite.

## Five-Way Comparison

| Model | Type | Fine-tuning method |
|-------|------|--------------------|
| GPT-2-XL (base) | Autoregressive, pre-RLHF | — |
| GPT-2-XL (fine-tuned) | Autoregressive, pre-RLHF | Full fine-tuning |
| Llama 3.1 8B (base) | Autoregressive, post-RLHF | — |
| Llama 3.1 8B (fine-tuned) | Autoregressive, post-RLHF | LoRA |
| **LLaDA 8B (fine-tuned)** | **Masked diffusion** | **LoRA** |

## Architecture (vs. Llama)

| | Llama 3.1 8B | LLaDA 8B |
|---|---|---|
| Attention | Causal (left-to-right only) | Bidirectional (sees everything) |
| Generation | Next-token prediction | Iterative unmasking |
| KV cache | Yes (speeds up inference) | No (can't cache bidirectional) |
| Training loss | CE on next token | CE on masked tokens / masking probability |
| Context window | 128K | 4096 |
| Model class | `AutoModelForCausalLM` | `AutoModel` (custom `LLaDAModelLM`) |
| HF loading | Standard | `trust_remote_code=True` required |

Internally, LLaDA uses the same transformer block structure as Llama (`block_type="llama"` in config) — same layer names (`q_proj`, `k_proj`, etc.), same dimensions. The only architectural change is removing the causal attention mask.

## What is SFT?

Supervised Fine-Tuning. Training a model on labeled input/output pairs — "given this prompt, produce this response." The term distinguishes it from other fine-tuning stages:

1. **Pre-training** — predict next token on trillions of tokens of internet text (learns language)
2. **SFT** — train on curated (instruction, response) pairs (learns to follow instructions)
3. **RLHF/DPO** — train on human preference data (learns to be helpful/safe)

This project skips step 3 and only does step 2 — taking base models and doing SFT on essay pairs. LLaDA-8B-Instruct has already gone through all three steps; we're doing a second round of SFT on top to teach it voice.

## How Training Works (SFT)

### Forward diffusion (masking)
1. Sample masking rate `t ~ Uniform(0, 1)` per example
2. Compute `p_mask = (1 - eps) * t + eps` where `eps = 1e-3`
3. Independently mask each response token with probability `p_mask`
4. **Never mask the prompt** — only response tokens get masked
5. Replace masked tokens with mask_id = 126336

### Loss computation
```
loss = CE(predictions[masked], targets[masked]) / p_mask[masked]
```
The `1/p_mask` weighting is importance sampling: when few tokens are masked (low t), each masked token is rarer training signal, so it gets upweighted. This ensures the model learns equally from all masking levels.

Normalized by answer length per sample, then averaged over batch.

### Key difference from autoregressive SFT
- Autoregressive: predict token N+1 from tokens 1..N. Loss on response tokens only.
- LLaDA: predict ALL masked tokens simultaneously from unmasked context. Loss on masked positions only.

The model sees bidirectional context — both the prompt AND unmasked response tokens inform each prediction. This is fundamentally different from left-to-right generation.

## How Inference Works (The Forward Pass, Step by Step)

### Setup

Say you give LLaDA the prompt "Write about class in America" and ask it to generate 128 tokens. The input starts as:

```
[Write] [about] [class] [in] [America] [MASK] [MASK] [MASK] ... [MASK]
 ←——— prompt (untouched) ———→  ←——— 128 MASK tokens ———→
```

The model fills in those 128 masks over 16 steps — roughly 8 tokens unmasked per step.

### Step 1: First forward pass

The entire sequence (prompt + all MASKs) goes through the transformer. Because attention is **bidirectional**, every position attends to every other position. The MASK tokens can see the prompt, and they can see each other (though other MASKs carry no useful information yet).

The model outputs logits at **every** position — including the prompt (ignored) and all 128 MASK positions. Each MASK position gets a probability distribution over the entire vocabulary:

```
Position 6 (MASK):  "The" → 0.12,  "I" → 0.08,  "Class" → 0.06, ...
Position 7 (MASK):  "is" → 0.03,   "in" → 0.04,  "real" → 0.02, ...
Position 8 (MASK):  "America" → 0.01, "the" → 0.02, ...
...all 128 positions get predictions simultaneously
```

We sample a token at each position (using Gumbel noise for randomness), then look at the model's **confidence** — how high was the softmax probability of the token it picked?

```
Position 6:  picked "The"    confidence: 0.12  ← relatively sure
Position 7:  picked "real"   confidence: 0.02  ← guessing
Position 8:  picked "the"    confidence: 0.02  ← guessing
Position 9:  picked "divide" confidence: 0.09  ← somewhat sure
```

**Unmask only the top 8** (128 tokens / 16 steps). Positions 6 and 9 had the highest confidence, so they get committed. The other 120 positions get **remasked** — thrown back to [MASK] for the next round:

```
[Write] [about] [class] [in] [America] [The] [MASK] [MASK] [divide] [MASK] ...
                                         ✓                   ✓
```

### Step 2: Second forward pass

The **entire sequence** goes through the transformer again. But now positions 6 and 9 are real tokens, not MASKs. The remaining 120 MASK positions can attend to these committed tokens — they have more context:

```
Position 7 now sees: [America] [The] [MASK] [divide] ...
         before:     [America] [MASK] [MASK] [MASK] ...
```

Predictions at remaining MASK positions should be **better** than step 1, because the model has more signal. Again, predict all 120 remaining MASKs, pick the top 8 most confident, commit them, remask the rest.

### Steps 3-16: Each time with more context

Each round, 8 more tokens get committed. Predictions improve because:

1. More committed tokens = more context for the remaining MASKs
2. The hardest positions (low confidence) get deferred to later, when surrounding context makes them easier

By step 16, all 128 tokens are committed.

### Why this is fundamentally different from autoregressive

**Llama** generates position 6, then position 7 (seeing position 6), then position 8 (seeing 6 and 7), etc. It's locked in — if position 6 is bad, every downstream token builds on that mistake.

**LLaDA** generates position 6 and position 73 in the same step if they're both high-confidence. It can commit the end of a sentence before the middle. It can nail the opening and the closing in step 1, then fill in the transitions later. The order of generation follows **confidence**, not **position**.

This is why the painting analogy works — a painter might rough in the composition first (the high-confidence structural elements), then refine details in later passes. The first pass isn't the top of the canvas.

### The semi-autoregressive block compromise

With `block_length=128` and `gen_length=1024`, LLaDA doesn't unmask all 1024 positions freely. It divides the generation into 8 blocks of 128 tokens and processes them left-to-right. Within each block, unmasking is by confidence. But block 2 doesn't start unmasking until block 1 is done.

This is a compromise — pure diffusion over 1024 tokens would need hundreds of steps to converge. The block structure gives the model some left-to-right scaffolding (earlier blocks inform later blocks) while still allowing non-sequential generation within each block.

### Inference Speed (the real tradeoff)

### Inference Speed (the real tradeoff)

Each step is a **full forward pass over the entire sequence** — no KV cache is possible because attention is bidirectional (every position could change what every other position attends to). Llama does one forward pass per token too, but each is cheap because the KV cache means it only computes attention for the new token against cached keys/values.

- **Llama/GPT-2:** 1 forward pass per token. KV cache means each pass only computes attention for the new token.
- **LLaDA:** `steps_per_block x num_blocks` full forward passes over the **entire** sequence. With default config (gen_length=1024, steps=128, block_length=128): 8 blocks x 16 steps = **128 full forward passes**, each recomputing bidirectional attention over prompt + 1024 tokens. No KV cache possible.

**Measured speed:**
- **T4 (float16):** ~8 tokens/sec. Baseline canaries (2 samples × 3 prompts × 512 tokens, fast config) = ~6 min.
- **L4 (float16):** ~33 tokens/sec. Same baseline run = ~2 min. Smoke test (128 tokens): 3.9s.
- Use fast config for iteration, full config for final comparison only.

**Speed knobs:**
- `GEN_STEPS`: fewer steps = faster but lower quality. 32-64 is fast for testing, 128 for final comparison.
- `GEN_LENGTH`: fewer tokens to generate = fewer masks to unmask. 512 is fine for Notes, 1024 for essays.
- `N_SAMPLES`: reduce to 1 for quick smoke tests, 5 for real comparison.

### Why L4 GPU over T4

**The real reason: VRAM.** T4 has 15GB, L4 has 24GB. Training at MAX_SEQ_LEN=2048 requires ~8.6GB just for attention maps (32 layers × 32 heads × 2048² × 2 bytes) — impossible on T4 after the model already uses ~8GB. L4 has ~16GB free after model loading, fitting 2048 comfortably.

**Inference is ~3x faster** (not 10x as originally estimated). The 10x figure came from comparing L4 against T4 running bfloat16 that silently fell back to fp32. After fixing to float16, T4 baseline canaries took ~6 min vs L4's ~2 min. The speedup comes from:

1. **More memory bandwidth.** T4: 320 GB/s, L4: 864 GB/s (~2.7x). LLaDA is heavily memory-bound — every forward pass reads the full model weights (no KV cache).
2. **Newer architecture.** Ada Lovelace (2023) vs Turing (2018). More efficient cores, better scheduling.
3. **L4 supports native bfloat16** (unlike T4), but we use float16 for both GPUs so this doesn't factor into the 3x number.

## Technical Details for the Notebook

- **Model:** `GSAI-ML/LLaDA-8B-Instruct` on HuggingFace
- **Mask token ID:** 126336 (defined in model config, not tokenizer)
- **Vocab size:** 126464
- **Quantization:** 4-bit NF4 via bitsandbytes (same as Llama)
- **LoRA targets:** `q_proj`, `k_proj`, `v_proj`, `ff_proj`, `up_proj` (NOT Llama names — no `o_proj`, `gate_proj`, or `down_proj`)
- **transformers pin:** `==4.38.2` (required by model repo)
- **Training loop:** Custom (can't use SFTTrainer — it assumes autoregressive loss)
- **Inference loop:** Custom (can't use `model.generate()` — it assumes left-to-right decoding)
- **Data:** Reuses `llama_train.jsonl` — same chat-format pairs

## Gotchas Hit During Setup

1. **`.to()` crash on 4-bit models.** LLaDA's custom `modeling_llada.py` calls `.to(device)` after loading. bitsandbytes forbids `.to()` on quantized models. Fix: monkey-patch `PreTrainedModel.to` to no-op when `quantization_method` is set. (See Cell 4 in notebook.)
2. **Layer names differ from Llama.** Despite `block_type="llama"` in config, the actual projection layers are `q_proj`, `k_proj`, `v_proj`, `ff_proj`, `up_proj` — no `o_proj`, `gate_proj`, or `down_proj`. Always run Cell 5 (inspect layers) before applying LoRA.
3. **`sentence-transformers` version conflict.** Colab pre-installs `sentence-transformers` which wants `transformers>=4.41.0`. The `transformers==4.38.2` pin triggers a warning but doesn't break anything since we don't use `sentence-transformers`.
4. **PEFT version must be pinned to 0.9.0.** Colab installs latest PEFT, which imports `EncoderDecoderCache` from transformers — added in 4.39+, doesn't exist in 4.38.2. This is a hard `ImportError`, not a warning. General rule: when you pin an old core library, pin its ecosystem dependents too.
5. **bfloat16 doesn't work on T4.** T4 GPUs (Turing arch) lack bfloat16 tensor cores. Setting `bnb_4bit_compute_dtype=torch.bfloat16` silently falls back to FP32, roughly halving speed. Use `torch.float16`. Same applies to `autocast` in the training loop.
6. **T4 OOMs at MAX_SEQ_LEN=2048 during training.** Attention maps scale O(n²): 2048 tokens × 32 layers × 32 heads × 2 bytes = ~8.6GB, exceeding remaining VRAM after the 4-bit model (~8GB). No gradient checkpointing available (LLaDA's custom model class doesn't support it). Fix: use L4 GPU (24GB) for 2048, or reduce to 1024 on T4.
7. **L4 also OOMs at MAX_SEQ_LEN=2048.** L4 has 24GB but only ~14GB free after model loading. Attention maps (~8.6GB) fit, but backprop without gradient checkpointing also stores all intermediate activations across 32 layers (hidden states, MLP outputs, pre-softmax attention, LoRA adapter outputs). Short sequences fit — the OOM hits on longer examples. Fix: A100 (40GB, ~32GB headroom) or reduce to 1536 on L4.
8. **Gradient checkpointing confirmed unsupported.** `LLaDAModelLM.supports_gradient_checkpointing` is False — calling `prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)` raises `ValueError`. This is a limitation of the custom model class, not a PEFT issue. Must pass `use_gradient_checkpointing=False`.

## Hardware Note for the Five-Way Comparison

LLaDA baselines generated on **L4 GPU** (24GB), training on **A100 GPU** (40GB), Llama/GPT-2 on **T4 GPU** (16GB). This doesn't affect the comparison — the hardware affects **speed**, not **output quality**. Same model, same weights, same quantization, same temperature produces identical tokens on all GPUs. The GPU is just doing the same matrix multiplication faster.

Where hardware *would* invalidate a comparison:
- Different **quantization** (e.g., 4-bit on T4 but 8-bit on A100) — different weights = different outputs
- Different **compute dtype** (fp16 on one, bf16 on the other) — rounding differences can butterfly-effect into different token choices

We use the same quantization config (4-bit NF4, float16 compute) across all GPUs, so the comparison holds. A100 was necessary because LLaDA doesn't support gradient checkpointing, and training at MAX_SEQ_LEN=2048 (matching Llama) exceeds VRAM on both T4 (15GB) and L4 (24GB).

### GPU progression
| GPU | VRAM | Result at MAX_SEQ_LEN=2048 |
|-----|------|---------------------------|
| T4 | 15GB | OOM instantly — attention maps alone exceed headroom |
| L4 | 24GB | OOM on longer examples — fits short sequences but backprop activations overflow |
| A100 | 40GB | ~32GB headroom, fits comfortably |

The root cause: no gradient checkpointing. With it, L4 would have been sufficient. Without it, all 32 layers of intermediate activations must be held in memory simultaneously for backprop.

## Known Risks

1. **No official fine-tuning scripts.** Authors explicitly won't release their training framework. Notebook adapted from SMDM repo (`finetune_mdm.py`) + LLaDA's GUIDELINES.md.
2. **PEFT/LoRA not officially tested** with LLaDA's custom model class. Should work (standard nn.Linear layers internally), but untested.
4. **No KV cache** means inference is slower per step than autoregressive models. Partially offset by fewer total tokens to generate (parallel unmasking vs. sequential).
5. **4096 context window** vs. Llama's 128K. Not an issue for current pairs (avg ~622 tokens) but limits Canary C (long essay).

## Resources

- **Paper:** https://arxiv.org/html/2502.09992v3
- **GitHub:** https://github.com/ML-GSAI/LLaDA
- **GUIDELINES.md** (training pseudocode): https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md
- **SMDM** (reference training framework): https://github.com/ML-GSAI/SMDM
- **SMDM SFT script** (adapted for our notebook): https://github.com/ML-GSAI/SMDM/blob/main/sft/finetune_mdm.py
