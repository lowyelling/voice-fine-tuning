# LLaDA: Masked Diffusion for Voice Fine-Tuning

Added 2026-03-12. Third model in the voice fine-tuning comparison.

**Paper:** https://arxiv.org/html/2502.09992v3
**Repo:** https://github.com/ML-GSAI/LLaDA
**Notebook:** `3_training/llada_finetune.ipynb`

## What LLaDA Is

LLaDA (Large Language Diffusion with mAsking) is a text diffusion model. Instead of generating left-to-right like GPT-2/Llama, it starts with a fully masked sequence and iteratively unmasks tokens — predicting all positions simultaneously, then keeping the most confident predictions and remasking the rest.

Think of it like painting vs. writing: autoregressive models write one word at a time (committed once written), LLaDA reveals the whole canvas in passes, refining low-confidence spots each round.

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

## How Inference Works

1. Start with `[prompt tokens] + [MASK MASK MASK ... MASK]` (gen_length masks)
2. Forward pass: model predicts all masked positions at once
3. Score each prediction by confidence (softmax probability)
4. **Unmask** the top-k most confident predictions
5. **Remask** the rest (they stay as [MASK] for next iteration)
6. Repeat for N steps

More steps = higher quality but slower. Temperature controls randomness via Gumbel noise. The generation is semi-autoregressive: divided into blocks processed left-to-right, but within each block, unmasking is parallel.

## Technical Details for the Notebook

- **Model:** `GSAI-ML/LLaDA-8B-Instruct` on HuggingFace
- **Mask token ID:** 126336 (defined in model config, not tokenizer)
- **Vocab size:** 126464
- **Quantization:** 4-bit NF4 via bitsandbytes (same as Llama)
- **LoRA targets:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **transformers pin:** `==4.38.2` (required by model repo)
- **Training loop:** Custom (can't use SFTTrainer — it assumes autoregressive loss)
- **Inference loop:** Custom (can't use `model.generate()` — it assumes left-to-right decoding)
- **Data:** Reuses `llama_train.jsonl` — same chat-format pairs

## Known Risks

1. **No official fine-tuning scripts.** Authors explicitly won't release their training framework. Notebook adapted from SMDM repo (`finetune_mdm.py`) + LLaDA's GUIDELINES.md.
2. **PEFT/LoRA not officially tested** with LLaDA's custom model class. Should work (standard nn.Linear layers internally), but untested.
3. **transformers==4.38.2 pin** may conflict with newer PEFT/bitsandbytes versions.
4. **No KV cache** means inference is slower per step than autoregressive models. Partially offset by fewer total tokens to generate (parallel unmasking vs. sequential).
5. **4096 context window** vs. Llama's 128K. Not an issue for current pairs (avg ~622 tokens) but limits Canary C (long essay).

## Resources

- **Paper:** https://arxiv.org/html/2502.09992v3
- **GitHub:** https://github.com/ML-GSAI/LLaDA
- **GUIDELINES.md** (training pseudocode): https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md
- **SMDM** (reference training framework): https://github.com/ML-GSAI/SMDM
- **SMDM SFT script** (adapted for our notebook): https://github.com/ML-GSAI/SMDM/blob/main/sft/finetune_mdm.py
