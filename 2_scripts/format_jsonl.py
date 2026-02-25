#!/usr/bin/env python3
"""
Convert training pairs from pairs.md into JSONL for Llama (chat-style) and GPT-2 (completion-style).

Input format (1_data/pairs.md):

    ## pair: essay-title-label
    tier: 3
    split: train

    ### prompt
    Write an essay about X. Thesis: ...

    ### response
    The finished essay text here...

    ---

Usage:
    python format_jsonl.py                    # process 1_data/pairs.md, write JSONL files
    python format_jsonl.py --validate-only    # check format + token counts without writing
    python format_jsonl.py --stats            # print tier/split distribution

Output:
    1_data/llama_train.jsonl
    1_data/llama_val.jsonl
    1_data/gpt2_train.jsonl
    1_data/gpt2_val.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAIRS_FILE = REPO_ROOT / "1_data" / "pairs.md"
OUTPUT_DIR = REPO_ROOT / "1_data"
OLD_PAIRS_DIR = REPO_ROOT / "1_data" / "pairs"

GPT2_MAX_TOKENS = 1024
LLAMA_MAX_TOKENS = 2048

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
    """Parse pairs.md into a list of pair dicts.

    Returns list of:
        {"label": str, "tier": int, "split": str, "prompt": str, "response": str}
    """
    text = path.read_text(encoding="utf-8")

    # Split on ## pair: headings. First chunk is the file header (instructions), skip it.
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

        # Parse split (default: train)
        split_match = re.search(r"^split:\s*(\w+)", block, re.MULTILINE)
        split = split_match.group(1).strip().lower() if split_match else "train"
        if split not in ("train", "val"):
            errors.append(f"'{label}': split must be 'train' or 'val', got '{split}'")
            continue

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
            "split": split,
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


def validate_and_report(pairs: list[dict]) -> None:
    """Print token counts and warnings for each pair."""
    warnings = []

    for p in pairs:
        combined = p["prompt"] + p["response"]
        gpt2_tokens = count_tokens_gpt2(combined)
        llama_tokens = count_tokens_llama(combined)

        status_gpt2 = "OK" if gpt2_tokens <= GPT2_MAX_TOKENS else f"EXCEEDS {GPT2_MAX_TOKENS}"
        status_llama = "OK" if llama_tokens <= LLAMA_MAX_TOKENS else f"EXCEEDS {LLAMA_MAX_TOKENS}"

        tier_str = f"T{p['tier']}" if p["tier"] else "T?"
        split_str = p["split"]

        print(f"  {p['label']} [{tier_str}, {split_str}]: "
              f"GPT-2: {gpt2_tokens} ({status_gpt2}), "
              f"Llama: {llama_tokens} ({status_llama})")

        if llama_tokens > LLAMA_MAX_TOKENS:
            warnings.append(f"  WARNING: '{p['label']}' exceeds Llama limit — will be truncated!")
        elif gpt2_tokens > GPT2_MAX_TOKENS:
            warnings.append(f"  NOTE: '{p['label']}' exceeds GPT-2 limit — Llama only")

    token_note = "" if TIKTOKEN_AVAILABLE else " (approximate — install tiktoken for accurate counts)"
    print(f"\nToken counts{token_note}")

    if warnings:
        print("\n" + "\n".join(warnings))


def print_stats(pairs: list[dict]) -> None:
    """Print tier and split distribution."""
    from collections import Counter

    tier_counts = Counter(p["tier"] for p in pairs)
    split_counts = Counter(p["split"] for p in pairs)

    print("\n--- Distribution ---")
    print(f"Total pairs: {len(pairs)}")
    print(f"  Train: {split_counts.get('train', 0)}")
    print(f"  Val:   {split_counts.get('val', 0)}")
    print()
    for tier in sorted(tier_counts):
        label = f"Tier {tier}" if tier else "Tier unspecified"
        print(f"  {label}: {tier_counts[tier]}")

    val_count = split_counts.get("val", 0)
    if val_count < 4:
        print(f"\n  WARNING: Only {val_count} validation pairs (recommend 4-5)")


def write_jsonl(path: Path, entries: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Format training pairs from pairs.md as JSONL")
    parser.add_argument("--validate-only", action="store_true", help="Check format + tokens without writing")
    parser.add_argument("--stats", action="store_true", help="Print tier/split distribution")
    args = parser.parse_args()

    # Warn about old format
    if OLD_PAIRS_DIR.exists() and list(OLD_PAIRS_DIR.glob("*.txt")):
        print("NOTE: Found .txt files in 1_data/pairs/. The pair format has changed.")
        print("      Pairs now go in 1_data/pairs.md (single markdown file).")
        print("      See the file for format instructions.\n")

    if not PAIRS_FILE.exists():
        print(f"No pairs file found at {PAIRS_FILE}")
        print("Create it using the template format. See 1_data/pairs.md.")
        sys.exit(1)

    if not TIKTOKEN_AVAILABLE:
        print("NOTE: tiktoken not installed. Token counts are approximate (len÷4).")
        print("      Install for accurate counts: pip install tiktoken\n")

    pairs = parse_pairs_file(PAIRS_FILE)

    if not pairs:
        print("No valid pairs found.")
        sys.exit(1)

    print(f"Parsed {len(pairs)} pair(s) from {PAIRS_FILE.name}\n")

    validate_and_report(pairs)

    if args.stats:
        print_stats(pairs)

    if args.validate_only or args.stats:
        return

    # Partition by split
    train_pairs = [p for p in pairs if p["split"] == "train"]
    val_pairs = [p for p in pairs if p["split"] == "val"]

    # Build JSONL entries, respecting token limits for GPT-2
    llama_train, llama_val = [], []
    gpt2_train, gpt2_val = [], []
    gpt2_skipped = 0

    for p in train_pairs:
        llama_train.append(to_llama_format(p["prompt"], p["response"]))
        gpt2_tokens = count_tokens_gpt2(p["prompt"] + p["response"])
        if gpt2_tokens <= GPT2_MAX_TOKENS:
            gpt2_train.append(to_gpt2_format(p["prompt"], p["response"]))
        else:
            gpt2_skipped += 1

    for p in val_pairs:
        llama_val.append(to_llama_format(p["prompt"], p["response"]))
        gpt2_tokens = count_tokens_gpt2(p["prompt"] + p["response"])
        if gpt2_tokens <= GPT2_MAX_TOKENS:
            gpt2_val.append(to_gpt2_format(p["prompt"], p["response"]))

    # Write files
    paths = {
        "llama_train": OUTPUT_DIR / "llama_train.jsonl",
        "llama_val": OUTPUT_DIR / "llama_val.jsonl",
        "gpt2_train": OUTPUT_DIR / "gpt2_train.jsonl",
        "gpt2_val": OUTPUT_DIR / "gpt2_val.jsonl",
    }

    write_jsonl(paths["llama_train"], llama_train)
    write_jsonl(paths["llama_val"], llama_val)
    write_jsonl(paths["gpt2_train"], gpt2_train)
    write_jsonl(paths["gpt2_val"], gpt2_val)

    print(f"\n--- Output ---")
    print(f"Llama  train: {len(llama_train)} pairs  →  {paths['llama_train']}")
    print(f"Llama  val:   {len(llama_val)} pairs  →  {paths['llama_val']}")
    print(f"GPT-2  train: {len(gpt2_train)} pairs  →  {paths['gpt2_train']}")
    print(f"GPT-2  val:   {len(gpt2_val)} pairs  →  {paths['gpt2_val']}")
    if gpt2_skipped:
        print(f"GPT-2  skipped: {gpt2_skipped} (exceeded {GPT2_MAX_TOKENS} tokens)")


if __name__ == "__main__":
    main()
