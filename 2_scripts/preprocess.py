#!/usr/bin/env python3
"""
Strip Substack artifacts from raw essay markdown files.

Usage:
    python preprocess.py                          # process all files in 1_data/raw/
    python preprocess.py path/to/essay.md         # process a single file
    python preprocess.py --dry-run                # show what would change without writing

Output goes to 1_data/cleaned/ with the same filename.
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "1_data" / "raw"
CLEANED_DIR = REPO_ROOT / "1_data" / "cleaned"

# Substack boilerplate patterns (matched per-line, case-insensitive)
BOILERPLATE_PATTERNS = [
    # Subscribe / share CTAs
    r"^(?:Thanks for reading|Thank you for reading).*$",
    r"^Subscribe$",
    r"^Share$",
    r"^Leave a comment$",
    r"^(?:Subscribe to|Sign up for).*$",
    r"^Share this post$",
    r"^.*is a reader-supported publication.*$",
    r"^(?:Upgrade to paid|Get .* % off).*$",
    r"^(?:Like|Comment|Restack)$",
]

BOILERPLATE_RE = re.compile(
    "|".join(f"(?:{p})" for p in BOILERPLATE_PATTERNS),
    re.MULTILINE | re.IGNORECASE,
)

# Footnote references like [1], [2] inline (with optional preceding space)
FOOTNOTE_REF_RE = re.compile(r" ?\[\d+\]")

# Footnote definition lines at bottom: "[1] Some text here"
FOOTNOTE_DEF_RE = re.compile(r"^\[\d+\].*$", re.MULTILINE)

# Image markdown: ![alt](url) -> remove entirely (images aren't text for fine-tuning)
IMAGE_MD_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")

# Markdown links: [text](url) -> text
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")

# Bare URLs on their own line
BARE_URL_LINE_RE = re.compile(r"^https?://\S+\s*$", re.MULTILINE)

# Inline bare URLs. Assumes Substack essays don't have meaningful bare URLs
# worth preserving for fine-tuning. If an essay references a URL as content
# (rare), re-add it manually after preprocessing.
INLINE_BARE_URL_RE = re.compile(r"(?<!\()(https?://\S+)")

# Horizontal rules (various formats) -> normalize to ***
# Avoids collision with --- used as delimiters in the training pipeline
HORIZONTAL_RULE_RE = re.compile(r"^\s*[-*_]\s*[-*_]\s*[-*_][\s*_-]*$", re.MULTILINE)

# Multiple blank lines -> max two
MULTI_BLANK_RE = re.compile(r"\n{3,}")

# Substack date metadata (e.g., "Feb 23, 2024")
_DATE_LINE_RE = re.compile(r"^[A-Z][a-z]+ \d{1,2}, \d{4}$")


def strip_header_dates(text: str) -> str:
    """Remove Substack date lines from the first 5 lines only.

    Substack metadata dates appear near the top of exported markdown.
    Stripping them globally would eat dates used inside essays as
    structural elements (e.g., a date in a personal narrative).
    """
    lines = text.split("\n")
    for i in range(min(5, len(lines))):
        if _DATE_LINE_RE.match(lines[i].strip()):
            lines[i] = ""
    return "\n".join(lines)


def preprocess(text: str) -> str:
    """Clean a raw Substack essay. Keeps paragraph breaks, emphasis, headers, block quotes."""

    # Strip metadata dates from the header area only
    text = strip_header_dates(text)

    # Remove image markdown before link stripping (prevents ![alt](url) -> !alt)
    text = IMAGE_MD_RE.sub("", text)

    # Strip markdown links, keep anchor text
    text = MARKDOWN_LINK_RE.sub(r"\1", text)

    # Remove bare URL lines
    text = BARE_URL_LINE_RE.sub("", text)

    # Remove remaining inline bare URLs
    text = INLINE_BARE_URL_RE.sub("", text)

    # Remove footnote definition lines (must come before reference removal)
    text = FOOTNOTE_DEF_RE.sub("", text)

    # Remove footnote reference numbers
    text = FOOTNOTE_REF_RE.sub("", text)

    # Remove boilerplate lines
    text = BOILERPLATE_RE.sub("", text)

    # Normalize horizontal rules to *** (avoid --- collision with pipeline delimiters)
    text = HORIZONTAL_RULE_RE.sub("***", text)

    # Collapse multiple blank lines to max two (one empty line between paragraphs)
    text = MULTI_BLANK_RE.sub("\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip() + "\n"

    return text


def process_file(input_path: Path, output_path: Path, dry_run: bool = False) -> None:
    raw = input_path.read_text(encoding="utf-8")
    cleaned = preprocess(raw)

    if dry_run:
        removed = len(raw) - len(cleaned)
        pct = (removed / len(raw) * 100) if raw else 0
        print(f"  {input_path.name}: {len(raw)} -> {len(cleaned)} chars ({removed} removed, {pct:.0f}%)")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned, encoding="utf-8")
    print(f"  {input_path.name} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw Substack essays")
    parser.add_argument("files", nargs="*", help="Specific files to process (default: all in 1_data/raw/)")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing files")
    args = parser.parse_args()

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        if not RAW_DIR.exists():
            print(f"No raw/ directory found at {RAW_DIR}")
            print("Create it and add your essay markdown files there.")
            sys.exit(1)
        paths = sorted(RAW_DIR.glob("*.md"))

    if not paths:
        print(f"No .md files found in {RAW_DIR}")
        sys.exit(1)

    print(f"Processing {len(paths)} file(s)...")
    for p in paths:
        out = CLEANED_DIR / p.name
        process_file(p, out, dry_run=args.dry_run)

    if not args.dry_run:
        print(f"\nCleaned files written to {CLEANED_DIR}/")


if __name__ == "__main__":
    main()
