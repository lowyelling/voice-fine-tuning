#!/usr/bin/env python3
"""
Convert canary output JSON files into readable markdown.

Usage:
    python format_canary.py <json_file> [<json_file2> ...]
    python format_canary.py 4_experiments/experiment\ logs/2_15-pairs/2_gpt2_finetuned.json
    python format_canary.py *_baselines.json *_finetuned.json

Output:
    Writes a .md file next to each input JSON with the same name.
    e.g. 2_gpt2_finetuned.json → 2_gpt2_finetuned.md
"""

import json
import sys
from pathlib import Path

CANARY_PROMPTS = {
    "A": "Write a personal Substack Note/Tweet about class in America, told from the perspective of a Chinese first generation immigrant whose family is lower-middle class.",
    "B": "Write a personal Substack Note/Tweet about Eileen Gu and Alyssa Liu, both winter Olympic gold medalists. Both grew up in the Bay Area, are half-asian and half-white, conceived via anonymous egg donor, and raised by a single parent. Eileen competed for China in skiing and is maximizing her influencer career while studying at Stanford. Meanwhile, Alyssa competed for the United States, took breaks from skating, and is inactive on social media.",
    "C": "Write an essay about Jacques Ellul as a forgotten prophet of propaganda and technological conformity.",
}

CANARY_LABELS = {
    "A": "Known topic, short",
    "B": "Novel topic, short",
    "C": "Known topic, long (Llama only)",
}


def format_json(path: Path) -> str:
    with open(path) as f:
        data = json.load(f)

    name = path.stem  # e.g. "2_gpt2_finetuned"
    lines = [f"# {name}\n"]

    for key in sorted(data.keys()):
        samples = data[key]
        label = CANARY_LABELS.get(key, key)
        prompt = CANARY_PROMPTS.get(key, "")

        lines.append(f"## Canary {key} — {label}\n")
        lines.append(f"**Prompt:** {prompt}\n")
        lines.append(f"**Samples:** {len(samples)}\n")

        for i, sample in enumerate(samples, 1):
            text = sample.strip()
            lines.append(f"### Sample {i}\n")
            lines.append(f"{text}\n")

        lines.append("---\n")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python format_canary.py <json_file> [<json_file2> ...]")
        sys.exit(1)

    for arg in sys.argv[1:]:
        path = Path(arg)
        if not path.exists():
            print(f"  SKIP: {path} not found")
            continue
        if path.suffix != ".json":
            print(f"  SKIP: {path} is not a .json file")
            continue

        md = format_json(path)
        out_path = path.with_suffix(".md")
        out_path.write_text(md, encoding="utf-8")
        print(f"  {path.name} → {out_path.name}")


if __name__ == "__main__":
    main()
