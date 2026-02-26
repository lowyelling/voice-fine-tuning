#!/usr/bin/env python3
"""
Generate GPT-2 pair files from Llama pair files.

Llama pairs have full-length responses (128K context). This script truncates
responses at paragraph boundaries to fit GPT-2's 1,024 token window.

Workflow:
    1. Curate pairs in llama_train.md / llama_val.md (full responses)
    2. Run this script to generate gpt2_train.md / gpt2_val.md (truncated)

Usage:
    python llama_to_gpt2.py 1_data/pairs/llama_train.md           # → gpt2_train.md
    python llama_to_gpt2.py 1_data/pairs/llama_val.md             # → gpt2_val.md
    python llama_to_gpt2.py 1_data/pairs/llama_train.md --dry-run # stats only
    python llama_to_gpt2.py 1_data/pairs/llama_train.md -o out.md # custom output path

Skips tier 1 pairs (Llama-only: rough draft → finished essay).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
except ImportError:
    print("ERROR: tiktoken required.  pip install tiktoken", file=sys.stderr)
    sys.exit(1)

GPT2_MAX = 1024
SEPARATOR = "\n\n---\n\n"
SEP_TOKENS = len(enc.encode(SEPARATOR))


# ── parsing ──────────────────────────────────────────────────────────────

def parse_pairs(text: str) -> list[dict]:
    """Parse a pairs .md file into list of {name, tier, prompt, response}."""
    pairs: list[dict] = []
    blocks = re.split(r"\n(?=## pair:)", text)

    for block in blocks:
        if not block.strip().startswith("## pair:"):
            continue

        name_m = re.match(r"## pair:\s*(.+)", block)
        if not name_m:
            continue
        name = name_m.group(1).strip()

        tier_m = re.search(r"^tier:\s*(\d+)", block, re.MULTILINE)
        tier = int(tier_m.group(1)) if tier_m else None

        prompt_m = re.search(r"### prompt\n(.*?)(?=\n### response)", block, re.DOTALL)
        prompt = prompt_m.group(1).strip() if prompt_m else ""

        # Response runs until a line that is just dashes (--- or --) or EOF
        resp_m = re.search(r"### response\n(.*?)(?=\n-{2,3}\s*$|\Z)", block, re.DOTALL)
        response = resp_m.group(1).strip() if resp_m else ""

        pairs.append({"name": name, "tier": tier, "prompt": prompt, "response": response})

    return pairs


# ── truncation ───────────────────────────────────────────────────────────

def truncate_response(response: str, prompt_tokens: int) -> tuple[str, int, int]:
    """Truncate response at paragraph boundary to fit GPT-2 budget.

    Returns (truncated_text, truncated_tokens, original_tokens).
    """
    orig_tokens = len(enc.encode(response)) if response else 0
    budget = GPT2_MAX - prompt_tokens - SEP_TOKENS

    if budget <= 0:
        return "", 0, orig_tokens
    if not response or orig_tokens <= budget:
        return response, orig_tokens, orig_tokens

    paragraphs = response.split("\n\n")
    running = 0
    cut_idx = len(paragraphs)

    for i, para in enumerate(paragraphs):
        if i > 0:
            running += len(enc.encode("\n\n"))
        running += len(enc.encode(para))
        if running > budget:
            cut_idx = i
            break

    # BPE boundary effects can make joined text slightly longer than the
    # sum-of-parts estimate.  Verify and trim until it actually fits.
    while cut_idx > 0:
        truncated = "\n\n".join(paragraphs[:cut_idx])
        trunc_tokens = len(enc.encode(truncated))
        if trunc_tokens <= budget:
            break
        cut_idx -= 1
    else:
        return "", 0, orig_tokens

    return truncated, trunc_tokens, orig_tokens


# ── output ───────────────────────────────────────────────────────────────

GPT2_HEADER_TRAIN = """\
# GPT-2 Training Pairs

Tiers 3-4 only. All responses must fit within GPT-2's 1,024 token context window (prompt + response combined). Truncate or exclude pairs that exceed this limit.

Tier reference:
- **3:** Opening → continuation essay or Note/Tweet
- **4:** Content extraction (neutral summary) → finished Note/Tweet"""

GPT2_HEADER_VAL = """\
# GPT-2 Validation Pairs

Holdout pairs for evaluation. These are NOT used for training.
Val pairs must come from different essays than training pairs."""


def format_pair(name: str, tier: int, prompt: str, response: str) -> str:
    """Format a single pair block."""
    lines = [
        f"## pair: {name}",
        f"tier: {tier}",
        "",
        "### prompt",
        prompt,
        "",
        "### response",
        response,
    ]
    return "\n".join(lines)


def build_output(results: list[dict], is_val: bool) -> str:
    """Assemble the full GPT-2 pairs .md file."""
    header = GPT2_HEADER_VAL if is_val else GPT2_HEADER_TRAIN
    blocks = [header, "---"]

    for r in results:
        blocks.append(format_pair(
            r["name"], r["tier"], r["prompt"], r["truncated_response"]
        ))
        blocks.append("---")

    return "\n\n".join(blocks) + "\n"


# ── main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate GPT-2 pair files from Llama pair files"
    )
    parser.add_argument("input", help="Llama pairs .md file")
    parser.add_argument("-o", "--output",
                        help="Output path (default: replace 'llama' with 'gpt2' in filename)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print truncation stats without writing")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        new_name = input_path.name.replace("llama", "gpt2")
        if new_name == input_path.name:
            print("ERROR: filename doesn't contain 'llama'. Use -o to specify output.",
                  file=sys.stderr)
            sys.exit(1)
        output_path = input_path.with_name(new_name)

    text = input_path.read_text(encoding="utf-8")
    pairs = parse_pairs(text)

    if not pairs:
        print("ERROR: no pairs found in input file.", file=sys.stderr)
        sys.exit(1)

    # Process pairs, skip tier 1
    results = []
    skipped = []

    for pair in pairs:
        if pair["tier"] == 1:
            skipped.append(pair["name"])
            continue

        prompt_tokens = len(enc.encode(pair["prompt"]))
        truncated, trunc_tok, orig_tok = truncate_response(pair["response"], prompt_tokens)
        budget = GPT2_MAX - prompt_tokens - SEP_TOKENS

        results.append({
            **pair,
            "truncated_response": truncated,
            "prompt_tokens": prompt_tokens,
            "orig_tokens": orig_tok,
            "trunc_tokens": trunc_tok,
            "budget": budget,
            "was_truncated": trunc_tok < orig_tok,
        })

    # Stats table
    print(file=sys.stderr)
    print(f"  {'Pair':<35s}  {'Tier':>4s}  {'Prompt':>6s}  {'Resp':>6s}  {'→':>1s}  {'Trunc':>5s}  {'Budget':>6s}  Status",
          file=sys.stderr)
    print(f"  {'-'*35}  {'-'*4}  {'-'*6}  {'-'*6}  {' ':>1s}  {'-'*5}  {'-'*6}  ------",
          file=sys.stderr)

    for r in results:
        if not r["truncated_response"] and r["response"]:
            status = "EMPTY (prompt too large)"
        elif r["was_truncated"]:
            pct = (1 - r["trunc_tokens"] / r["orig_tokens"]) * 100 if r["orig_tokens"] else 0
            status = f"TRUNCATED (-{pct:.0f}%)"
        else:
            status = "ok"
        print(f"  {r['name']:<35s}  T{r['tier']:>3d}  {r['prompt_tokens']:>6d}  {r['orig_tokens']:>6d}  →  {r['trunc_tokens']:>5d}  {r['budget']:>6d}  {status}",
              file=sys.stderr)

    if skipped:
        print(f"\n  Skipped {len(skipped)} tier 1 pairs (Llama only): {', '.join(skipped)}",
              file=sys.stderr)

    truncated_count = sum(1 for r in results if r["was_truncated"])
    print(f"\n  {len(results)} pairs | {truncated_count} truncated | output: {output_path}",
          file=sys.stderr)

    if args.dry_run:
        return

    # Write output
    is_val = "val" in input_path.name
    output_text = build_output(results, is_val)
    output_path.write_text(output_text, encoding="utf-8")
    print(f"  ✓ wrote {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
