# Voice Fine-Tuning

Fine-tune open-source language models on my essays to generate drafts in my writing voice.

## The Question

Can fine-tuning capture something prompt engineering can't — the voice beneath the words? Not vocabulary or sentence length, but the editing instinct: what gets cut, what gets kept, how a paragraph earns its ending.

After a year of prompt engineering approaches (custom GPTs, persona blends, statistical fingerprinting), the answer was no. So this project moves to the fine-tuning layer.

## Architecture

Three layers, each handled differently:

1. **Voice (fine-tuning)** — LoRA adapter trained on essay pairs. Teaches *how* I write: rhythm, structure, word choice, omission.
2. **Personhood (system prompt)** — Personal context loaded at inference time. Teaches *why* I write. Not fine-tuned — lives in the context window so it can be updated without retraining.
3. **Runtime input** — Notes, thesis, voice memos for a specific essay. What the piece is *about*.

Voice without personhood = cover band. Personhood without voice = AI that knows you but writes like AI.

## Approach

Two models, four-way comparison:

| Model | Fine-Tuning | Why |
|-------|------------|-----|
| **Llama 3.1 8B** | LoRA | Best creative writing benchmarks at 8B. 128K context sees full essays. |
| **GPT-2-XL** | Full | Pre-RLHF, raw next-token prediction. 1,024 token context — only sees short pieces. |

Same training pairs, two models, base vs. fine-tuned = four outputs per prompt. This answers two questions: does RLHF help or hurt voice fidelity? And is a pre-RLHF model more receptive to small-dataset fine-tuning?

## Training Data

~60-70 pairs across four tiers, curated from ~20 Substack essays and ~20 Substack Notes:

- **Tier 1:** Rough draft → finished essay (editing instinct)
- **Tier 2:** Thesis/topic → finished essay (generation from prompt)
- **Tier 3:** Opening → continuation (voice sustain)
- **Tier 4:** Thesis/topic → finished Note (compressed voice, clean 1:1 model comparison)

Training data is not included in this repo.

## Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Training scripts, data prep |
| Llama 3.1 8B | Primary model (LoRA fine-tuning) |
| GPT-2-XL | Comparison model (full fine-tuning) |
| Google Colab | Cloud GPU (T4) for training and inference |
| Hugging Face transformers + PEFT | Model loading, tokenization, LoRA |

## Repo Structure

```
2_scripts/        Data preparation and evaluation utilities
3_training/       Colab notebooks and hyperparameter configs
4_experiments/    Training run logs and output samples
5_docs/           PRD, engineering design doc
```

## Context

Week 4 project at [Fractal Tech](https://fractalbootcamp.com) (NYC). Built by [Lily](https://lowyelling.substack.com).
