# Plan: Voice Fine-Tuning — Execution Kickoff

## Context

It's Feb 25 (Tuesday). Demo is Saturday. All four work directories are empty — docs are solid but no execution has started. Yesterday was writing-only. That means Feb 24's tasks (audit, raw text, Colab setup) are all still undone, and today's tasks (pair construction, smoke test) depend on them.

The bottleneck is training data curation, which is manual — only Lily can decide which essays to use, what tier each pair belongs to, and what the prompts should be. But I can build the tooling around it so the mechanical parts go fast.

## What I'll Build (Phase 1 — today)

### 1. `1_data/audit.md` — Essay/Note inventory template
A structured table Lily fills in as she catalogs her Substack essays and Notes. Columns: title, type (essay/blog/Note), word count estimate, has rough draft?, has AI brainstorm?, fits GPT-2 (<750 words)?, tier candidates, notes. Pre-populated with blank rows she fills in.

### 2. `2_scripts/preprocess.py` — Strip Substack artifacts from raw text
Takes a markdown file from `1_data/raw/`, strips:
- URLs/hyperlinks (keeps anchor text)
- Footnote reference numbers
- Substack boilerplate (subscribe CTAs, "Thank you for reading")
- Extra whitespace

Keeps: paragraph breaks, emphasis (`*italics*`, `**bold**`), section headers, block quotes. Outputs cleaned file to `1_data/cleaned/`.

### 3. `2_scripts/format_jsonl.py` — Build training pairs into JSONL
Takes a directory of pair files (simple format: prompt on top, separator, response on bottom) and outputs two JSONL files:
- `llama_train.jsonl` — chat-style: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
- `gpt2_train.jsonl` — completion-style: `{"text": "...\n\n---\n\n..."}`

Also validates token counts (warns if a pair exceeds 2048 for Llama or 1024 for GPT-2) using tiktoken for approximate counts (exact counts need the model's tokenizer on Colab, but tiktoken gets close enough for planning).

### 4. `4_experiments/canary-prompts.md` — Template
Three sections (A, B, C) with descriptions of what each canary tests, blank space for Lily to write the actual prompts.

### 5. `4_experiments/experiment-log.md` — Template
Structured markdown for logging each training run per the format in CLAUDE.md.

## What Lily Does (Phase 2 — today + tomorrow)

### Today (Feb 25):
1. **Essay audit** — Open `1_data/audit.md` template, go through Substack, fill in every essay and substantial Note. No curation decisions yet, just inventory.
2. **Pull raw text** — Copy-paste essays from Substack into `1_data/raw/` as markdown files. Mechanical work, interleave with audit. Start with the 5-6 strongest essays.
3. **Pick canary prompts** — Fill in `4_experiments/canary-prompts.md`. Canary A: known topic, short. Canary B: novel topic, short. Canary C: known topic, long.

### Tomorrow (Feb 26):
4. **Run preprocessing** — Use `preprocess.py` on raw essays to get clean text.
5. **Construct 3 pairs** — Pick 3 best short pieces. Write prompt + response pairs. Use `format_jsonl.py` to generate JSONL files.
6. **Build GPT-2-XL Colab notebook** — What it needs:
   - Cell 1: `!pip install transformers datasets`
   - Cell 2: Load GPT-2-XL model + tokenizer (`AutoModelForCausalLM.from_pretrained("gpt2-xl")`, `AutoTokenizer.from_pretrained("gpt2-xl")`)
   - Cell 3: Generate baseline from base model on canary prompts (store for comparison)
   - Cell 4: Mount Google Drive (`from google.colab import drive; drive.mount('/content/drive')`)
   - Cell 5: Upload and load JSONL training data
   - Cell 6: Training config (`TrainingArguments`: lr=5e-5, epochs=3, batch_size=1, output_dir to Drive)
   - Cell 7: `Trainer` with the model + dataset, `trainer.train()`
   - Cell 8: Generate from fine-tuned model on same canary prompts
   - Cell 9: Side-by-side comparison cell
7. **Smoke test** — Run the notebook with 3 pairs. Does format work? Does output shift?

### Thursday (Feb 27):
8. **Overfitting diagnostic** — Same 3 pairs, crank epochs to 15-20. Can it memorize?
9. **Scale to ~10-13 pairs** — Construct more pairs across tiers. Retrain.
10. **Build Llama Colab notebook** — What it needs:
    - Same structure as GPT-2-XL but with:
    - `!pip install transformers datasets peft bitsandbytes accelerate`
    - Load in 4-bit quantization: `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`
    - LoRA config: `LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")`
    - HF access token for Meta license (set up on huggingface.co first)
    - `get_peft_model(model, lora_config)` to wrap the model
    - Same training loop but with LoRA-wrapped model

### Friday (Feb 28):
11. **Full dataset** — Both models trained on all pairs
12. **Four-way comparison** on canary prompts
13. **Iterate** — Adjust pairs based on outputs, retrain if time

### Saturday (Mar 1):
14. **Demo prep** — Generate essay from best model, prepare presentation

## Scope Adjustment

Given the compressed timeline, the realistic MVP is:
- **Must have:** GPT-2-XL smoke test + overfitting diagnostic (proves the pipeline works end-to-end)
- **Should have:** GPT-2-XL trained on ~13 pairs + Llama smoke test
- **Nice to have:** Full four-way comparison on complete dataset

GPT-2-XL first because it's faster (minutes to train, no LoRA complexity). Get a working pipeline, then layer on Llama.

## Files Created

| File | Purpose | Status |
|------|---------|--------|
| `1_data/audit.md` | Essay/Note inventory template | Done |
| `2_scripts/preprocess.py` | Strip Substack artifacts | Done, tested |
| `2_scripts/format_jsonl.py` | Convert pairs to both JSONL formats | Done, tested |
| `4_experiments/canary-prompts.md` | Canary prompt template | Done |
| `4_experiments/experiment-log.md` | Run logging template | Done |

## Verification

All tooling tested:
1. `preprocess.py` correctly strips URLs, footnotes, boilerplate while preserving emphasis, headers, block quotes
2. `format_jsonl.py` outputs correct chat-style (Llama) and completion-style (GPT-2) JSONL, with token count warnings
