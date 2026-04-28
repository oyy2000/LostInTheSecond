#!/usr/bin/env python3
"""
Build the final combined SFT dataset from all data sources:

1. Correct greedy generations (7942 samples)
2. Prefill-correct from Step 2 fix (498 samples)
3. Prefill-correct from Step 3 fix (after running prefill)
4. Rejection-sampled correct responses

Outputs a single merged SFT JSON for training.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def make_sft_entry(question: str, response: str) -> Dict:
    instruction = f"### Instruction:\n{question}\n\n### Response: Let's think step by step."
    return {"instruction": instruction, "input": "", "output": response}


def load_alpaca_file(path: str, label: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        print(f"  [{label}] NOT FOUND: {path}")
        return []
    data = json.loads(p.read_text("utf-8"))
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict) and "samples" in data:
        entries = data["samples"]
    else:
        entries = data
    print(f"  [{label}] {len(entries)} entries from {path}")
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--greedy-gen", default="artifacts_real/full/lemma_llama3_generations.json")
    ap.add_argument("--prefill-step2", default="artifacts_real/full/lemma_sft_fix_step2.json")
    ap.add_argument("--prefill-step3", default="artifacts_real/full/lemma_sft_fix_step3.json",
                    help="Step 3 prefill-correct SFT (if available)")
    ap.add_argument("--rejection-sampled", default="artifacts_real/full/lemma_sft_rejection_sampled.json")
    ap.add_argument("--out-file", default="artifacts_real/full/lemma_sft_combined.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-greedy-correct", action="store_true", default=True)
    ap.add_argument("--no-greedy-correct", dest="include_greedy_correct", action="store_false")
    args = ap.parse_args()

    random.seed(args.seed)
    all_entries = []

    print("Loading data sources:")

    # 1. Greedy correct
    if args.include_greedy_correct:
        gen_path = Path(args.greedy_gen)
        obj = json.loads(gen_path.read_text("utf-8"))
        samples = obj["samples"] if isinstance(obj, dict) and "samples" in obj else obj
        greedy_correct = []
        for s in samples:
            if s.get("source_exact_match") != 1.0:
                continue
            if not s.get("pred_answer"):
                continue
            q = s.get("doc", {}).get("question", "").strip()
            r = (s.get("neg_response") or "").strip()
            if q and r:
                greedy_correct.append(make_sft_entry(q, r))
        print(f"  [greedy_correct] {len(greedy_correct)} entries")
        all_entries.extend(greedy_correct)

    # 2. Prefill step2 correct
    s2_entries = load_alpaca_file(args.prefill_step2, "prefill_step2")
    all_entries.extend(s2_entries)

    # 3. Prefill step3 correct
    s3_entries = load_alpaca_file(args.prefill_step3, "prefill_step3")
    all_entries.extend(s3_entries)

    # 4. Rejection sampled
    rs_entries = load_alpaca_file(args.rejection_sampled, "rejection_sampled")
    all_entries.extend(rs_entries)

    # Deduplicate by instruction (question)
    seen = set()
    deduped = []
    for e in all_entries:
        key = e["instruction"]
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    random.shuffle(deduped)

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(deduped, indent=2, ensure_ascii=False), "utf-8")

    print(f"\n=== Combined Dataset ===")
    print(f"Total (before dedup): {len(all_entries)}")
    print(f"Total (after dedup):  {len(deduped)}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
