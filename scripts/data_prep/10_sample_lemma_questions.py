#!/usr/bin/env python3
"""
Sample ~1000 questions from LEMMA training data.

Extracts unique questions from lemma.json (alpaca format), deduplicates,
and outputs a JSONL file compatible with 11_generate_with_llama3.py.

Output format (one per line):
{"id": 0, "question": "...", "source": "gsm8k"|"math", "gt_answer": "..."}
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any


def extract_question(instruction: str) -> str:
    """Extract question text from LEMMA instruction field."""
    text = instruction.strip()
    text = re.sub(r"^###\s*Instruction:\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n\n###\s*Response:.*$", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def extract_answer(output: str) -> str:
    """Extract final answer from LEMMA output field."""
    m = re.search(r"The answer is:\s*(.+?)(?:\n|$)", output, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def guess_source(question: str) -> str:
    """Heuristically guess if a question is from GSM8K or MATH."""
    math_indicators = [
        "\\frac", "\\sqrt", "\\text", "\\boxed", "\\begin",
        "\\end", "\\cdot", "\\times", "\\div", "\\pi",
        "polynomial", "matrix", "determinant", "eigenvalue",
        "integral", "derivative", "theorem", "prove",
    ]
    q_lower = question.lower()
    for ind in math_indicators:
        if ind.lower() in q_lower:
            return "math"
    return "gsm8k"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lemma-json",
        default="/jet/home/swang47/yang/projects/LEMMA/LLaMA-Factory/data/lemma.json",
    )
    ap.add_argument("--out-file", default="./artifacts_real/lemma_sampled_questions.jsonl")
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    print(f"Loading LEMMA data from {args.lemma_json} ...")
    data: List[Dict[str, Any]] = json.loads(Path(args.lemma_json).read_text("utf-8"))
    print(f"  Total samples: {len(data)}")

    seen_questions = set()
    unique_items = []
    for item in data:
        q = extract_question(item["instruction"])
        if not q or q in seen_questions:
            continue
        seen_questions.add(q)
        unique_items.append({
            "question": q,
            "gt_answer": extract_answer(item["output"]),
            "source": guess_source(q),
        })

    print(f"  Unique questions: {len(unique_items)}")

    gsm = [x for x in unique_items if x["source"] == "gsm8k"]
    math = [x for x in unique_items if x["source"] == "math"]
    print(f"  GSM8K-like: {len(gsm)}, MATH-like: {len(math)}")

    if args.n_samples < 0:
        sampled = unique_items
    else:
        n = min(args.n_samples, len(unique_items))
        n_gsm = min(len(gsm), n // 2)
        n_math = min(len(math), n - n_gsm)
        n_gsm = n - n_math

        random.shuffle(gsm)
        random.shuffle(math)
        sampled = gsm[:n_gsm] + math[:n_math]
    random.shuffle(sampled)

    for i, item in enumerate(sampled):
        item["id"] = i

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in sampled:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    src_counts = {}
    for item in sampled:
        src_counts[item["source"]] = src_counts.get(item["source"], 0) + 1

    print(f"\nSampled {len(sampled)} questions -> {out_path}")
    for src, cnt in sorted(src_counts.items()):
        print(f"  {src}: {cnt}")


if __name__ == "__main__":
    main()
