#!/usr/bin/env python3
"""
Extract LEMMA original SFT entries for the 1K sampled questions.
Maps questions back to lemma.json to get the full training data.
"""

import argparse
import json
import re
from pathlib import Path


def extract_question(instruction: str) -> str:
    text = instruction.strip()
    text = re.sub(r"^###\s*Instruction:\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n\n###\s*Response:.*$", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="./artifacts_real/lemma_sampled_questions.jsonl")
    ap.add_argument("--lemma-json", default="/jet/home/swang47/yang/projects/LEMMA/LLaMA-Factory/data/lemma.json")
    ap.add_argument("--out-file", default="./artifacts_real/lemma_1k_original_sft.json")
    ap.add_argument("--one-per-question", action="store_true",
                    help="Keep only one entry per question (first match)")
    args = ap.parse_args()

    print(f"Loading sampled questions from {args.questions}")
    sampled_qs = set()
    with open(args.questions) as f:
        for line in f:
            obj = json.loads(line)
            sampled_qs.add(obj["question"].strip())
    print(f"  Loaded {len(sampled_qs)} unique questions")

    print(f"Loading LEMMA data from {args.lemma_json}")
    lemma_data = json.loads(Path(args.lemma_json).read_text("utf-8"))
    print(f"  Total LEMMA entries: {len(lemma_data)}")

    matched = []
    seen_qs = set()
    for item in lemma_data:
        q = extract_question(item["instruction"])
        if q in sampled_qs:
            if args.one_per_question and q in seen_qs:
                continue
            seen_qs.add(q)
            matched.append(item)

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(matched, indent=2, ensure_ascii=False), "utf-8")

    print(f"\nMatched {len(matched)} entries from {len(seen_qs)} unique questions")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
