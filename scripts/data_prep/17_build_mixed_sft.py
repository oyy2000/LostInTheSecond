#!/usr/bin/env python3
"""
Build mixed SFT dataset from:
  1. LLaMA-3 correct generations (source_exact_match == 1.0)
  2. Prefill-correct samples (lemma_sft_fix_step2.json)

Output: Alpaca-format JSON for training.
"""

import argparse
import json
import random
from pathlib import Path


def make_sft_entry(question: str, response: str):
    instruction = f"### Instruction:\n{question}\n\n### Response: Let's think step by step."
    return {"instruction": instruction, "input": "", "output": response}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", default="artifacts_real/full/lemma_llama3_generations.json")
    ap.add_argument("--prefill-sft", default="artifacts_real/full/lemma_sft_fix_step2.json")
    ap.add_argument("--out-file", default="artifacts_real/full/lemma_sft_mixed.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-sources", nargs="*", default=[],
                    help="Skip samples from these source datasets")
    args = ap.parse_args()

    random.seed(args.seed)

    gen_path = Path(args.generations)
    obj = json.loads(gen_path.read_text("utf-8"))
    all_samples = obj["samples"] if isinstance(obj, dict) and "samples" in obj else obj

    correct_entries = []
    src_counts = {}
    for s in all_samples:
        if s.get("source_exact_match") != 1.0:
            continue
        if not s.get("pred_answer"):
            continue
        source = s.get("source", "unknown")
        src_counts[source] = src_counts.get(source, 0) + 1
        if source in args.skip_sources:
            continue
        question = s.get("doc", {}).get("question", "").strip()
        response = (s.get("neg_response") or "").strip()
        if not question or not response:
            continue
        correct_entries.append(make_sft_entry(question, response))

    print(f"Correct generations by source: {src_counts}")
    print(f"Correct entries extracted: {len(correct_entries)}")

    prefill_path = Path(args.prefill_sft)
    prefill_entries = json.loads(prefill_path.read_text("utf-8"))
    print(f"Prefill-correct entries: {len(prefill_entries)}")

    merged = correct_entries + prefill_entries
    random.shuffle(merged)

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), "utf-8")

    print(f"\nMixed dataset: {len(merged)} samples")
    print(f"  - Correct originals: {len(correct_entries)}")
    print(f"  - Prefill corrected: {len(prefill_entries)}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
