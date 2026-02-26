#!/usr/bin/env python3
"""
Show where GPT-2's 1,024 token limit falls in an essay.

Three modes:
    1. Analyze    — paragraph-by-paragraph token counts, show where cutoff lands
    2. Cut        — output truncated essay (response budget only, no prompt file)
    3. Cut+prompt — read prompt from prompts.md, subtract from budget, output truncated essay

Usage:
    # Mode 1: analyze
    python token_budget.py 1_data/cleaned/essay.md
    python token_budget.py 1_data/cleaned/                    # all essays in a directory
    python token_budget.py 1_data/cleaned/essay.md --prompt-tokens 80

    # Mode 2: cut (response only)
    python token_budget.py 1_data/cleaned/essay.md --cut
    python token_budget.py 1_data/cleaned/essay.md --cut --prompt "The author argues..."

    # Mode 3: cut with prompt lookup
    python token_budget.py 1_data/cleaned/essay.md --cut --prompts
    python token_budget.py 1_data/cleaned/ --cut --prompts    # batch: all essays with prompts

The separator between prompt and response in GPT-2 format is "\\n\\n---\\n\\n"
(~4 tokens), which is included in the budget automatically.
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
    print("ERROR: tiktoken required. Install with: pip install tiktoken")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_FILE = REPO_ROOT / "1_data" / "pairs" / "prompts.md"

GPT2_MAX = 1024
SEPARATOR = "\n\n---\n\n"
SEPARATOR_TOKENS = len(enc.encode(SEPARATOR))


def parse_prompts_file(path: Path) -> dict[str, dict]:
    """Parse prompts.md into {slug: {tier, prompt}} dict.

    Expected format:
        ## essay-slug
        tier: 3
        Prompt text here...
    """
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n(?=## )", text)

    prompts = {}
    for block in blocks:
        block = block.strip()
        heading = re.match(r"## (.+)", block)
        if not heading:
            continue
        slug = heading.group(1).strip()
        body = block[heading.end():].strip()

        # Parse tier
        tier_match = re.match(r"tier:\s*(\d+)", body)
        tier = int(tier_match.group(1)) if tier_match else None
        if tier_match:
            body = body[tier_match.end():].strip()

        if body or tier:
            prompts[slug] = {"tier": tier, "prompt": body}

    return prompts


def match_prompt(filename: str, prompts: dict[str, dict]) -> tuple[str, dict] | None:
    """Find the prompt entry whose slug appears in the filename.

    Returns (slug, entry_dict) or None.
    """
    for slug, entry in prompts.items():
        if slug in filename:
            return slug, entry
    return None


def find_cut_index(essay_text: str, prompt_tokens: int) -> int | None:
    """Return the paragraph index (0-based) where the cutoff happens.

    Returns None if the full essay fits within budget.
    """
    overhead = prompt_tokens + SEPARATOR_TOKENS
    budget = GPT2_MAX - overhead

    paragraphs = essay_text.split("\n\n")
    running = 0

    for i, para in enumerate(paragraphs):
        if i > 0:
            running += len(enc.encode("\n\n"))
        running += len(enc.encode(para))
        if running > budget:
            return i

    return None


def analyze_essay(essay_text: str, prompt_tokens: int) -> None:
    """Print paragraph-by-paragraph token counts and mark the cutoff."""
    overhead = prompt_tokens + SEPARATOR_TOKENS
    budget = GPT2_MAX - overhead

    paragraphs = essay_text.split("\n\n")
    running = 0
    cutoff_hit = False

    print(f"  Token budget: {GPT2_MAX} total - {prompt_tokens} prompt - {SEPARATOR_TOKENS} separator = {budget} for response\n")
    print(f"  {'Para':>4}  {'Tokens':>6}  {'Running':>7}  Text")
    print(f"  {'----':>4}  {'------':>6}  {'-------':>7}  ----")

    last_clean_cut = 0
    for i, para in enumerate(paragraphs, 1):
        para_tokens = len(enc.encode(para))
        prev_running = running
        if i > 1:
            running += len(enc.encode("\n\n"))
        running += para_tokens

        marker = ""
        if not cutoff_hit and running > budget:
            marker = "  <-- CUTOFF"
            cutoff_hit = True
            last_clean_cut = prev_running

        preview = para[:70].replace("\n", " ")
        if len(para) > 70:
            preview += "..."
        print(f"  {i:4}  {para_tokens:6}  {running:7}{marker}  {preview}")

    print()
    if cutoff_hit:
        print(f"  Clean cut: {last_clean_cut} tokens (paragraph before the cutoff)")
        print(f"  Full essay: {running} tokens")
        print(f"  Excluded: {running - last_clean_cut} tokens ({(running - last_clean_cut) / running * 100:.0f}% of essay)")
    else:
        print(f"  Full essay fits! {running} / {budget} tokens used")


def cut_essay(essay_text: str, prompt_tokens: int) -> str:
    """Return the essay truncated to fit GPT-2's token budget."""
    cut_idx = find_cut_index(essay_text, prompt_tokens)
    paragraphs = essay_text.split("\n\n")
    if cut_idx is None:
        return essay_text

    # BPE boundary effects can make joined text slightly longer than the
    # sum-of-parts estimate.  Verify and trim until it actually fits.
    budget = GPT2_MAX - prompt_tokens - SEPARATOR_TOKENS
    while cut_idx > 0:
        result = "\n\n".join(paragraphs[:cut_idx])
        if len(enc.encode(result)) <= budget:
            return result
        cut_idx -= 1
    return ""



def normalize_text(s: str) -> str:
    """Normalize unicode and whitespace for comparison."""
    s = s.strip()
    s = s.replace("\u201c", '"').replace("\u201d", '"')  # " "
    s = s.replace("\u2018", "'").replace("\u2019", "'")  # ' '
    s = s.replace("\u2014", "--").replace("\u2013", "-")  # — –
    s = s.replace("\u2026", "...")  # …
    return re.sub(r"\s+", " ", s)


def strip_prompt_from_essay(essay_text: str, prompt_text: str) -> str:
    """Find where the prompt text ends in the essay and return everything after.

    Uses normalized character matching so it works even when the prompt starts
    mid-paragraph (e.g., after a title or content warning in the same paragraph).
    Returns from the next paragraph boundary after the match.
    """
    norm_prompt = normalize_text(prompt_text)
    norm_essay = normalize_text(essay_text)

    idx = norm_essay.find(norm_prompt)
    if idx == -1:
        return essay_text

    # Find the end position of the prompt in the normalized text
    end_pos = idx + len(norm_prompt)

    # Map back to original text: count how many characters of normalized text
    # we've consumed, find the corresponding position in the original
    norm_count = 0
    orig_pos = 0
    for orig_pos, ch in enumerate(essay_text):
        if norm_count >= end_pos:
            break
        # Skip characters that normalize away (extra whitespace)
        test = normalize_text(essay_text[:orig_pos + 1])
        norm_count = len(test)

    # Find the next paragraph boundary
    rest = essay_text[orig_pos:]
    next_para = rest.find("\n\n")
    if next_para != -1:
        return rest[next_para:].lstrip("\n")
    return rest.strip()


def process_file(
    path: Path,
    prompt_tokens: int,
    do_cut: bool = False,
    slug: str | None = None,
    tier: int | None = None,
    prompt_text: str | None = None,
) -> None:
    text = path.read_text(encoding="utf-8").strip()

    # For Tier 3, strip the opening (prompt) from the essay so we only analyze/cut the continuation
    if prompt_text and tier == 3:
        text = strip_prompt_from_essay(text, prompt_text)

    total = len(enc.encode(text))

    if do_cut:
        tier_label = f"T{tier}" if tier else ""
        if slug:
            print(f"# {path.name} (slug: {slug}, {tier_label}, prompt: {prompt_tokens} tokens, response: {total} tokens)", file=sys.stderr)
        print(cut_essay(text, prompt_tokens))
        return

    print(f"\n{'=' * 60}")
    print(f"  {path.name} ({total} tokens{'  [continuation only]' if tier == 3 else '  total'})")
    if slug:
        tier_label = f"Tier {tier}" if tier else "Tier ?"
        print(f"  Prompt slug: {slug} | {tier_label} | {prompt_tokens} prompt tokens")
    print(f"{'=' * 60}")
    analyze_essay(text, prompt_tokens)


def main():
    parser = argparse.ArgumentParser(description="GPT-2 token budget analyzer and truncator")
    parser.add_argument("paths", nargs="+", help="Essay file(s) or directory")
    parser.add_argument("--prompt-tokens", type=int, default=80,
                        help="Token count reserved for the prompt (default: 80)")
    parser.add_argument("--prompt", type=str,
                        help="Actual prompt text (overrides --prompt-tokens)")
    parser.add_argument("--cut", action="store_true",
                        help="Output the truncated essay text")
    parser.add_argument("--prompts", action="store_true",
                        help="Mode 3: read prompts from prompts.md, match by slug")
    args = parser.parse_args()

    # Collect files
    files = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.glob("*.md")))
        elif path.is_file():
            files.append(path)
        else:
            print(f"WARNING: {p} not found, skipping", file=sys.stderr)

    if not files:
        print("No files found.")
        sys.exit(1)

    # Mode 3: prompt lookup from file
    if args.prompts:
        if not PROMPTS_FILE.exists():
            print(f"ERROR: prompts file not found at {PROMPTS_FILE}", file=sys.stderr)
            sys.exit(1)

        prompts = parse_prompts_file(PROMPTS_FILE)
        if not prompts:
            print("ERROR: no prompts found in prompts file.", file=sys.stderr)
            sys.exit(1)

        matched = 0
        skipped = []
        for f in files:
            result = match_prompt(f.stem, prompts)
            if result is None:
                skipped.append(f.name)
                continue

            slug, entry = result
            prompt_text = entry["prompt"]
            if not prompt_text:
                skipped.append(f"{f.name} (empty prompt)")
                continue

            prompt_tokens = len(enc.encode(prompt_text))
            process_file(f, prompt_tokens, do_cut=args.cut, slug=slug, tier=entry["tier"], prompt_text=prompt_text)
            matched += 1

        if skipped:
            print(f"\nSkipped {len(skipped)} files (no matching prompt):", file=sys.stderr)
            for s in skipped:
                print(f"  {s}", file=sys.stderr)
        if matched == 0:
            print("ERROR: no files matched any prompt slugs.", file=sys.stderr)
            sys.exit(1)
        return

    # Mode 1 & 2: manual prompt
    prompt_tokens = args.prompt_tokens
    if args.prompt:
        prompt_tokens = len(enc.encode(args.prompt))
        if not args.cut:
            print(f"Prompt: {prompt_tokens} tokens")

    if args.cut and len(files) > 1:
        print("ERROR: --cut without --prompts only works with a single file.", file=sys.stderr)
        sys.exit(1)

    for f in files:
        process_file(f, prompt_tokens, do_cut=args.cut)


if __name__ == "__main__":
    main()
