#!/usr/bin/env python3
"""
Build Llama pairs from prompts and cleaned essays.

Reads prompts.md for prompt text and tiers, matches each slug to a cleaned
essay file, strips the prompt from the essay to get the continuation, and
writes pairs to a Llama pairs file.

Pipeline:
    prompts.md + cleaned essays → prompts_to_llama.py → llama_train.md
                                  llama_to_gpt2.py   → gpt2_train.md

Usage:
    python prompts_to_llama.py                             # append new pairs
    python prompts_to_llama.py --regenerate                # replace tier 3 responses with full text
    python prompts_to_llama.py --target llama_val.md       # different target file
    python prompts_to_llama.py --dry-run                   # show what would change
    python prompts_to_llama.py --slug 15-the-9-11-ai-nft   # single essay only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Import shared logic from token_budget.py (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from token_budget import parse_prompts_file, strip_prompt_from_essay

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_FILE = REPO_ROOT / "1_data" / "pairs" / "prompts.md"
CLEANED_DIR = REPO_ROOT / "1_data" / "cleaned"
DEFAULT_TARGET = REPO_ROOT / "1_data" / "pairs" / "llama_train.md"


# ── matching ─────────────────────────────────────────────────────────────

def norm(s: str) -> str:
    """Normalize underscores to hyphens for slug comparison."""
    return s.replace("_", "-")


def find_essay(slug: str, cleaned_dir: Path) -> Path | None:
    """Find the cleaned essay file matching a slug."""
    n = norm(slug)
    for f in sorted(cleaned_dir.glob("*_clean.md")):
        if n in norm(f.stem):
            return f
    return None


# ── parsing ──────────────────────────────────────────────────────────────

def parse_target(text: str) -> tuple[str, list[dict]]:
    """Parse a Llama pairs .md file into (header, list of pair dicts).

    Each pair dict has: name, tier, prompt, response, raw (original block text).
    """
    # Split into header + pair blocks
    parts = re.split(r"\n(?=## pair:)", text)
    header = parts[0]
    pairs = []

    for block in parts[1:]:
        name_m = re.match(r"## pair:\s*(.+)", block)
        if not name_m:
            continue
        name = name_m.group(1).strip()

        tier_m = re.search(r"^tier:\s*(\d+)", block, re.MULTILINE)
        tier = int(tier_m.group(1)) if tier_m else None

        prompt_m = re.search(r"### prompt\n(.*?)(?=\n### response)", block, re.DOTALL)
        prompt = prompt_m.group(1).strip() if prompt_m else ""

        resp_m = re.search(r"### response\n(.*?)(?=\n-{2,3}\s*$|\Z)", block, re.DOTALL)
        response = resp_m.group(1).strip() if resp_m else ""

        pairs.append({"name": name, "tier": tier, "prompt": prompt, "response": response})

    return header, pairs


# ── pair building ────────────────────────────────────────────────────────

def build_response(tier: int, prompt: str, essay_text: str) -> str:
    """Build a response from prompt + essay text.

    Tier 3: strips the prompt (opening) from the essay, continuation is the response.
    Tier 4: full essay text is the response (prompt is a summary, not in the essay).
    """
    if tier == 3:
        return strip_prompt_from_essay(essay_text, prompt)
    return essay_text


def format_pair(pair: dict) -> str:
    """Format a pair dict as a markdown block."""
    return "\n".join([
        f"## pair: {pair['name']}",
        f"tier: {pair['tier']}",
        "",
        "### prompt",
        pair["prompt"],
        "",
        "### response",
        pair["response"],
    ])


def build_output(header: str, pairs: list[dict]) -> str:
    """Assemble the full pairs .md file."""
    blocks = [header.rstrip(), "---"]
    for pair in pairs:
        blocks.append(format_pair(pair))
        blocks.append("---")
    return "\n\n".join(blocks) + "\n"


# ── main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build Llama pairs from prompts.md + cleaned essays"
    )
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET,
                        help="Llama pairs file (default: llama_train.md)")
    parser.add_argument("--slug", type=str,
                        help="Process only this slug (default: all)")
    parser.add_argument("--regenerate", action="store_true",
                        help="Replace existing tier 3 responses with full essay continuations")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change without writing")
    args = parser.parse_args()

    # Parse prompts
    if not PROMPTS_FILE.exists():
        print(f"ERROR: {PROMPTS_FILE} not found", file=sys.stderr)
        sys.exit(1)
    prompts = parse_prompts_file(PROMPTS_FILE)

    if args.slug:
        if args.slug not in prompts:
            print(f"ERROR: slug '{args.slug}' not found in {PROMPTS_FILE.name}", file=sys.stderr)
            sys.exit(1)
        prompts = {args.slug: prompts[args.slug]}

    # Build full responses from cleaned essays
    full_responses: dict[str, str] = {}  # norm_slug → full response
    no_essay = []

    for slug, entry in prompts.items():
        if not entry["prompt"]:
            continue
        essay_path = find_essay(slug, CLEANED_DIR)
        if essay_path is None:
            no_essay.append(slug)
            continue
        essay_text = essay_path.read_text(encoding="utf-8").strip()
        response = build_response(entry["tier"], entry["prompt"], essay_text)

        if entry["tier"] == 3 and response == essay_text:
            print(f"  WARNING: prompt not found in essay for {slug}", file=sys.stderr)

        full_responses[norm(slug)] = response

    if no_essay:
        print(f"  No cleaned essay found: {', '.join(no_essay)}", file=sys.stderr)

    # Read existing target file
    if not args.target.exists():
        print(f"ERROR: {args.target} not found", file=sys.stderr)
        sys.exit(1)

    text = args.target.read_text(encoding="utf-8")
    header, existing_pairs = parse_target(text)

    # Regenerate: replace responses for matching tier 3 pairs
    if args.regenerate:
        updated = 0
        unchanged = 0
        existing_norm_names = {norm(p["name"]) for p in existing_pairs}

        for pair in existing_pairs:
            n = norm(pair["name"])
            if n in full_responses and pair["tier"] == 3:
                old_len = len(pair["response"])
                new_resp = full_responses[n]
                new_len = len(new_resp)
                if old_len != new_len:
                    print(f"  ~ {pair['name']:<40s}  T{pair['tier']}  "
                          f"{old_len:>5d} → {new_len:>5d} chars", file=sys.stderr)
                    pair["response"] = new_resp
                    updated += 1
                else:
                    unchanged += 1
            else:
                unchanged += 1

        # Append any prompts not already in the file
        new_pairs = []
        for slug, entry in prompts.items():
            if norm(slug) not in existing_norm_names and norm(slug) in full_responses:
                pair = {"name": slug, "tier": entry["tier"],
                        "prompt": entry["prompt"], "response": full_responses[norm(slug)]}
                new_pairs.append(pair)
                print(f"  + {slug:<40s}  T{entry['tier']}  (new)", file=sys.stderr)

        all_pairs = existing_pairs + new_pairs
        print(f"\n  {updated} updated | {unchanged} unchanged | {len(new_pairs)} new",
              file=sys.stderr)

        if updated == 0 and not new_pairs:
            print("  Nothing to do.", file=sys.stderr)
            return

        if args.dry_run:
            return

        output = build_output(header, all_pairs)
        args.target.write_text(output, encoding="utf-8")
        print(f"  ✓ wrote {args.target.name}", file=sys.stderr)
        return

    # Default mode: append new pairs only
    existing_norms = {norm(p["name"]) for p in existing_pairs}
    new_pairs = []

    for slug, entry in prompts.items():
        if norm(slug) in existing_norms:
            continue
        if norm(slug) not in full_responses:
            continue
        pair = {"name": slug, "tier": entry["tier"],
                "prompt": entry["prompt"], "response": full_responses[norm(slug)]}
        new_pairs.append(pair)
        resp_preview = pair["response"][:60].replace("\n", " ")
        print(f"  + {slug:<40s}  T{entry['tier']}  {resp_preview}...", file=sys.stderr)

    print(f"\n  {len(new_pairs)} new pairs to add", file=sys.stderr)

    if not new_pairs or args.dry_run:
        return

    blocks = []
    for pair in new_pairs:
        blocks.append(format_pair(pair))
        blocks.append("---")

    appendix = "\n\n" + "\n\n".join(blocks) + "\n"
    with open(args.target, "a", encoding="utf-8") as f:
        f.write(appendix)

    print(f"  ✓ appended {len(new_pairs)} pairs to {args.target.name}", file=sys.stderr)


if __name__ == "__main__":
    main()
