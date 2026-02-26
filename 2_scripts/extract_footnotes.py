#!/usr/bin/env python3
"""
Extract footnotes from raw Substack essays into separate files.

Finds the footnote section (after the final horizontal rule, starting at the
first [N](url#footnote-anchor-...) marker) and saves cleaned footnotes.

Usage:
    python extract_footnotes.py                          # process all in 1_data/sources/essays/
    python extract_footnotes.py path/to/essay.md         # process a single file
    python extract_footnotes.py --dry-run                # show what would be extracted

Output goes to 1_data/sources/footnotes/ as {stem}_footnotes.md
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
ESSAYS_DIR = REPO_ROOT / "1_data" / "sources" / "essays"
FOOTNOTES_DIR = REPO_ROOT / "1_data" / "sources" / "footnotes"

# Horizontal rules (various formats): * * *, ***, ---, ___, etc.
HORIZONTAL_RULE_RE = re.compile(r"^\s*[-*_]\s*[-*_]\s*[-*_][\s*_-]*$", re.MULTILINE)

# Substack footnote definition anchor: [N](url#footnote-anchor-N-...)
FOOTNOTE_ANCHOR_RE = re.compile(
    r"^\[\d+\]\([^)]*#footnote-anchor-\d+-[^)]*\)$", re.MULTILINE
)

# Convert footnote anchors to clean [N] markers
FOOTNOTE_ANCHOR_CLEAN_RE = re.compile(
    r"^\[(\d+)\]\([^)]*#footnote-anchor-\d+-[^)]*\)$", re.MULTILINE
)

# --- Cleaning patterns (same as preprocess.py) ---

# Linked images: [![alt](inner)](outer) with optional caption
LINKED_IMAGE_RE = re.compile(
    r"^\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\).*$", re.MULTILINE
)

# Image markdown: ![alt](url)
IMAGE_MD_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")

# Markdown links: [text](url) -> text
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")

# Empty markdown links: [](url)
EMPTY_LINK_RE = re.compile(r"\[\]\([^)]+\)")

# Bare URLs on their own line
BARE_URL_LINE_RE = re.compile(r"^https?://\S+\s*$", re.MULTILINE)

# Inline bare URLs
INLINE_BARE_URL_RE = re.compile(r"(?<!\()(https?://\S+)")

# Multiple blank lines -> max two
MULTI_BLANK_RE = re.compile(r"\n{3,}")


def extract_footnotes(text: str) -> Optional[str]:
    """Extract and clean footnotes from a raw Substack essay.

    Returns cleaned footnote text, or None if no footnotes found.
    """
    # Find the last horizontal rule
    last_rule = None
    for m in HORIZONTAL_RULE_RE.finditer(text):
        last_rule = m

    if last_rule is None:
        return None

    after_rule = text[last_rule.end():]

    # Find the first footnote anchor after the last horizontal rule
    m = FOOTNOTE_ANCHOR_RE.search(after_rule)
    if m is None:
        return None

    footnotes = after_rule[m.start():]

    # Clean the footnotes
    # Convert [N](url#footnote-anchor-...) -> [N]
    footnotes = FOOTNOTE_ANCHOR_CLEAN_RE.sub(r"[\1]", footnotes)

    # Strip linked images
    footnotes = LINKED_IMAGE_RE.sub("", footnotes)

    # Strip regular images
    footnotes = IMAGE_MD_RE.sub("", footnotes)

    # Strip markdown links, keep text
    footnotes = MARKDOWN_LINK_RE.sub(r"\1", footnotes)

    # Strip empty links
    footnotes = EMPTY_LINK_RE.sub("", footnotes)

    # Strip bare URL lines
    footnotes = BARE_URL_LINE_RE.sub("", footnotes)

    # Strip inline URLs
    footnotes = INLINE_BARE_URL_RE.sub("", footnotes)

    # Collapse blank lines
    footnotes = MULTI_BLANK_RE.sub("\n\n", footnotes)

    footnotes = footnotes.strip() + "\n"

    return footnotes


def process_file(input_path: Path, output_path: Path, dry_run: bool = False) -> bool:
    """Extract footnotes from a single file. Returns True if footnotes were found."""
    raw = input_path.read_text(encoding="utf-8")
    footnotes = extract_footnotes(raw)

    if footnotes is None:
        print(f"  {input_path.name}: no footnotes found")
        return False

    count = len(re.findall(r"^\[\d+\]$", footnotes, re.MULTILINE))

    if dry_run:
        print(f"  {input_path.name}: {count} footnotes, {len(footnotes)} chars")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(footnotes, encoding="utf-8")
    print(f"  {input_path.name}: {count} footnotes -> {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Extract footnotes from Substack essays")
    parser.add_argument("files", nargs="*", help="Specific files (default: all in sources/essays/)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be extracted")
    args = parser.parse_args()

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        if not ESSAYS_DIR.exists():
            print(f"No essays directory found at {ESSAYS_DIR}")
            sys.exit(1)
        paths = sorted(ESSAYS_DIR.glob("*.md"))

    if not paths:
        print(f"No .md files found in {ESSAYS_DIR}")
        sys.exit(1)

    print(f"Processing {len(paths)} file(s)...")
    found = 0
    for p in paths:
        out = FOOTNOTES_DIR / (p.stem + "_footnotes.md")
        if process_file(p, out, dry_run=args.dry_run):
            found += 1

    if not args.dry_run:
        print(f"\n{found} footnote file(s) written to {FOOTNOTES_DIR}/")


if __name__ == "__main__":
    main()
