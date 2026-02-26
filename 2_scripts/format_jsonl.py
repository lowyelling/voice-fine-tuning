#!/usr/bin/env python3
"""
Convert training pairs from separate train/val markdown files into JSONL
for Llama (chat-style) and GPT-2 (completion-style).

Input files:
    1_data/pairs/llama_train.md   — Llama training pairs (all tiers, full length)
    1_data/pairs/llama_val.md     — Llama validation pairs
    1_data/pairs/gpt2_train.md    — GPT-2 training pairs (Tiers 3-4, ≤1024 tokens)
    1_data/pairs/gpt2_val.md      — GPT-2 validation pairs

Usage:
    python format_jsonl.py                    # process all files, write JSONL
    python format_jsonl.py --validate-only    # check format + token counts without writing
    python format_jsonl.py --stats            # print tier/split distribution

Output:
    1_data/jsonl/llama_train.jsonl
    1_data/jsonl/llama_val.jsonl
    1_data/jsonl/gpt2_train.jsonl
    1_data/jsonl/gpt2_val.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAIRS_DIR = REPO_ROOT / "1_data" / "pairs"
OUTPUT_DIR = REPO_ROOT / "1_data" / "jsonl"

GPT2_MAX_TOKENS = 1024
LLAMA_MAX_TOKENS = 2048

# Input files: (model, split) → path
INPUT_FILES = {
    ("llama", "train"): PAIRS_DIR / "llama_train.md",
    ("llama", "val"):   PAIRS_DIR / "llama_val.md",
    ("gpt2", "train"):  PAIRS_DIR / "gpt2_train.md",
    ("gpt2", "val"):    PAIRS_DIR / "gpt2_val.md",
}

# Try to import tiktoken for accurate token counting
try:
    import tiktoken

    _gpt2_enc = tiktoken.get_encoding("gpt2")
    _llama_enc = tiktoken.get_encoding("cl100k_base")  # proxy for Llama (~10-15% off)

    def count_tokens_gpt2(text: str) -> int:
        return len(_gpt2_enc.encode(text))

    def count_tokens_llama(text: str) -> int:
        return len(_llama_enc.encode(text))

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

    def count_tokens_gpt2(text: str) -> int:
        return len(text) // 4

    def count_tokens_llama(text: str) -> int:
        return len(text) // 4


def parse_pairs_file(path: Path) -> list[dict]:
    """Parse a pairs markdown file into a list of pair dicts.

    Returns list of:
        {"label": str, "tier": int, "prompt": str, "response": str}
    """
    text = path.read_text(encoding="utf-8")

    # Split on ## pair: headings. First chunk is the file header, skip it.
    raw_blocks = re.split(r"\n(?=## pair:)", text)

    pairs = []
    errors = []

    for i, block in enumerate(raw_blocks):
        block = block.strip()
        if not block.startswith("## pair:"):
            continue  # skip header / non-pair content

        # Parse label from heading
        heading_match = re.match(r"## pair:\s*(.+)", block)
        if not heading_match:
            errors.append(f"Block {i}: could not parse ## pair: heading")
            continue
        label = heading_match.group(1).strip()

        # Parse tier (default: 0 = unspecified)
        tier_match = re.search(r"^tier:\s*(\d+)", block, re.MULTILINE)
        tier = int(tier_match.group(1)) if tier_match else 0

        # Parse prompt and response sections
        prompt_match = re.search(r"### prompt\s*\n", block)
        response_match = re.search(r"### response\s*\n", block)

        if not prompt_match:
            errors.append(f"'{label}': missing ### prompt section")
            continue
        if not response_match:
            errors.append(f"'{label}': missing ### response section")
            continue

        # Prompt = everything between ### prompt and ### response
        prompt_start = prompt_match.end()
        response_header_start = response_match.start()
        prompt = block[prompt_start:response_header_start].strip()

        # Response = everything after ### response (until end of block)
        response_start = response_match.end()
        response = block[response_start:].strip()

        # Strip trailing horizontal rule if present (visual separator between pairs)
        response = re.sub(r"\n---\s*$", "", response).strip()

        if not prompt:
            errors.append(f"'{label}': empty prompt")
            continue
        if not response:
            errors.append(f"'{label}': empty response")
            continue

        pairs.append({
            "label": label,
            "tier": tier,
            "prompt": prompt,
            "response": response,
        })

    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  {e}")
        print()

    return pairs


def to_llama_format(prompt: str, response: str) -> dict:
    """Chat-style JSONL for Llama fine-tuning."""
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def to_gpt2_format(prompt: str, response: str) -> dict:
    """Completion-style JSONL for GPT-2 fine-tuning."""
    return {"text": f"{prompt}\n\n---\n\n{response}"}


def validate_and_report(pairs: list[dict], model: str, split: str) -> None:
    """Print token counts and warnings for each pair."""
    max_tokens = GPT2_MAX_TOKENS if model == "gpt2" else LLAMA_MAX_TOKENS
    count_fn = count_tokens_gpt2 if model == "gpt2" else count_tokens_llama
    warnings = []

    print(f"\n  [{model} {split}] — {len(pairs)} pair(s)")

    for p in pairs:
        combined = p["prompt"] + p["response"]
        tokens = count_fn(combined)
        status = "OK" if tokens <= max_tokens else f"EXCEEDS {max_tokens}"
        tier_str = f"T{p['tier']}" if p["tier"] else "T?"

        print(f"    {p['label']} [{tier_str}]: {tokens} tokens ({status})")

        if tokens > max_tokens:
            warnings.append(f"    WARNING: '{p['label']}' exceeds {model} limit — will be truncated!")

    if warnings:
        print("\n" + "\n".join(warnings))


def print_stats(all_pairs: dict) -> None:
    """Print tier and split distribution across all files."""
    from collections import Counter

    print("\n--- Distribution ---")
    for (model, split), pairs in all_pairs.items():
        tier_counts = Counter(p["tier"] for p in pairs)
        print(f"\n  {model} {split}: {len(pairs)} pairs")
        for tier in sorted(tier_counts):
            label = f"Tier {tier}" if tier else "Tier unspecified"
            print(f"    {label}: {tier_counts[tier]}")

    # Warn about low val counts
    for model in ("llama", "gpt2"):
        val_pairs = all_pairs.get((model, "val"), [])
        if len(val_pairs) < 2:
            print(f"\n  WARNING: Only {len(val_pairs)} {model} validation pair(s) (recommend 2+)")


def write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Format training pairs as JSONL")
    parser.add_argument("--validate-only", action="store_true", help="Check format + tokens without writing")
    parser.add_argument("--stats", action="store_true", help="Print tier/split distribution")
    args = parser.parse_args()

    if not TIKTOKEN_AVAILABLE:
        print("NOTE: tiktoken not installed. Token counts are approximate (len÷4).")
        print("      Install for accurate counts: pip install tiktoken\n")

    # Parse all input files
    all_pairs = {}
    for (model, split), path in INPUT_FILES.items():
        if not path.exists():
            print(f"WARNING: {path.name} not found — skipping {model} {split}")
            continue
        pairs = parse_pairs_file(path)
        all_pairs[(model, split)] = pairs
        print(f"Parsed {len(pairs)} pair(s) from {path.name}")

    if not all_pairs:
        print("No valid pair files found.")
        sys.exit(1)

    # Validate and report
    for (model, split), pairs in all_pairs.items():
        validate_and_report(pairs, model, split)

    token_note = "" if TIKTOKEN_AVAILABLE else " (approximate — install tiktoken for accurate counts)"
    print(f"\nToken counts{token_note}")

    if args.stats:
        print_stats(all_pairs)

    if args.validate_only or args.stats:
        return

    # Build and write JSONL
    output_paths = {}
    for (model, split), pairs in all_pairs.items():
        if model == "llama":
            entries = [to_llama_format(p["prompt"], p["response"]) for p in pairs]
        else:
            entries = [to_gpt2_format(p["prompt"], p["response"]) for p in pairs]

        path = OUTPUT_DIR / f"{model}_{split}.jsonl"
        write_jsonl(path, entries)
        output_paths[(model, split)] = (len(entries), path)

    print(f"\n--- Output ---")
    for (model, split), (count, path) in output_paths.items():
        print(f"{model:6s} {split:5s}: {count} pairs  →  {path}")


if __name__ == "__main__":
    main()
