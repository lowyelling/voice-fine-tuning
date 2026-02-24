# Voice Fine-Tuning

Fine-tune open-source language models on my essays to generate drafts in my writing voice.

## The Question

Can fine-tuning capture something prompt engineering can't — the voice beneath the words? Not vocabulary or sentence length, but the editing instinct: what gets cut, what gets kept, how a paragraph earns its ending.

After a year of prompt engineering approaches, the answer was no. So this project moves to the fine-tuning layer.

## Approach

Two models, four-way comparison:

| Model | Fine-Tuning | Why |
|-------|------------|-----|
| **Llama 3.1 8B** | LoRA | Best creative writing benchmarks at 8B. 128K context sees full essays. |
| **GPT-2-XL** | Full | Pre-RLHF, raw next-token prediction. 1,024 token context — only sees short pieces. |

Same training pairs, two models, base vs. fine-tuned = four outputs per prompt. This answers two questions: does RLHF help or hurt voice fidelity? And is a pre-RLHF model more receptive to small-dataset fine-tuning?

## Training Data

~60-70 pairs across four tiers, curated from ~20 Substack posts and ~20 Substack Notes:

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

Week 4 project at [Fractal Tech](https://github.com/fractal-nyc/bootcamp-monorepo/blob/main/curriculum/weeks/04-ambition/week-4-raise-your-ambitions-challenge.md) (NYC). Built by [Lily](https://lowyelling.com).
