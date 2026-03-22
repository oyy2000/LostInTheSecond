#!/usr/bin/env python3
"""
Convert prefilled pipeline output to LEMMA alpaca training format.

Input:  artifacts_real/lemma_ds2_wait_recompute_gpt_prefill.json
Output: artifacts_real/lemma_sft_train.json  (alpaca format for LLaMA-Factory)

LEMMA format:
{
  "instruction": "### Instruction:\n{question}\n\n### Response: Let's think step by step.",
  "input": "",
  "output": "{full reasoning trajectory}"
}
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def normalize_answer(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("$", "").replace(",", "").replace("%", "")
    text = re.sub(r"\\boxed\{(.*?)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    return text


def extract_answer(text: str) -> str:
    m = re.search(r"The answer is:?\s*(.+?)(?:\n|$)", text or "", re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def to_lemma_format(item: Dict[str, Any], require_correct: bool = True) -> Optional[Dict[str, Any]]:
    doc = item.get("doc", {})
    question = (doc.get("question") or "").strip()
    pos_response = (item.get("pos_response") or "").strip()

    if not question or not pos_response:
        return None

    if require_correct:
        gt = item.get("gt_answer", "")
        pred = extract_answer(pos_response)
        if gt and pred and normalize_answer(pred) != normalize_answer(gt):
            return None
        found = item.get("prefill_found_correct")
        if found is not None and not found:
            return None

    instruction = f"### Instruction:\n{question}\n\n### Response: Let's think step by step."

    return {
        "instruction": instruction,
        "input": "",
        "output": pos_response,
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-file",
        default="./artifacts_real/lemma_ds2_wait_recompute_gpt_prefill.json",
    )
    ap.add_argument(
        "--out-file",
        default="./artifacts_real/lemma_sft_train.json",
    )
    ap.add_argument(
        "--out-dataset-info",
        default="",
        help="Write a dataset_info.json entry for LLaMA-Factory",
    )
    ap.add_argument("--require-correct", action="store_true", default=True)
    ap.add_argument("--no-require-correct", dest="require_correct", action="store_false")
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.in_file).resolve()
    out_path = Path(args.out_file).resolve()

    obj = json.loads(in_path.read_text("utf-8"))
    all_samples = obj["samples"] if isinstance(obj, dict) and "samples" in obj else obj

    if args.limit > 0:
        all_samples = all_samples[:args.limit]

    converted: List[Dict[str, Any]] = []
    skipped = 0
    for item in all_samples:
        lemma_item = to_lemma_format(item, require_correct=args.require_correct)
        if lemma_item is None:
            skipped += 1
            continue
        converted.append(lemma_item)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(converted, indent=2, ensure_ascii=False), "utf-8")

    print(f"Converted: {len(converted)} / {len(all_samples)} (skipped {skipped})")
    print(f"Output: {out_path}")

    if converted:
        print(f"\nSample instruction (first 200 chars):")
        print(f"  {converted[0]['instruction'][:200]}")
        print(f"Sample output (first 300 chars):")
        print(f"  {converted[0]['output'][:300]}")

    if args.out_dataset_info:
        info_path = Path(args.out_dataset_info)
        info_path.parent.mkdir(parents=True, exist_ok=True)
        info = {
            "lemma_lostinsecond": {
                "file_name": str(out_path),
                "formatting": "alpaca",
            }
        }
        info_path.write_text(json.dumps(info, indent=2), "utf-8")
        print(f"Dataset info: {info_path}")


if __name__ == "__main__":
    main()
