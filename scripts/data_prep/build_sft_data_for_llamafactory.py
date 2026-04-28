#!/usr/bin/env python3
"""
Build LlamaFactory-compatible SFT datasets from corrected-prefix trajectories.

Two schemes:
  1. normal:   All correct samples (exact_match=1) from raw_cot + prefilled_corrected_k2
  2. weighted: Same as normal, but step-2 corrected samples are oversampled by --weight factor

Input files (3B pipeline):
  - results/gsm8k_3b_multi_sample/raw_cot_n8.jsonl   (raw CoT, keep correct ones)
  - results/gsm8k_3b_prefix_correction/prefilled_corrected_k2.jsonl  (prefilled after correction)
  - results/gsm8k_3b_prefix_correction/corrected_k2_gpt.jsonl  (GPT-corrected prefix only)

Output: sharegpt-format JSON for LlamaFactory.

Usage:
    python scripts/data_prep/build_sft_data_for_llamafactory.py \
        --raw-cot results/gsm8k_3b_multi_sample/raw_cot_n8.jsonl \
        --corrected-k2 results/gsm8k_3b_prefix_correction/corrected_k2_gpt.jsonl \
        --prefilled-k2 results/gsm8k_3b_prefix_correction/prefilled_corrected_k2.jsonl \
        --out-dir data/sft_gsm8k_3b \
        --weight 3
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        print(f"[WARN] File not found: {path}")
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def row_to_sharegpt(question: str, response: str, source: str = "") -> Dict[str, Any]:
    """Convert a (question, response) pair to sharegpt format."""
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": question.strip()},
            {"from": "gpt", "value": response.strip()},
        ],
        "source": source,
    }


def build_correct_raw_samples(raw_cot_path: Path) -> List[Dict[str, Any]]:
    """Extract correct trajectories from raw CoT sampling."""
    rows = read_jsonl(raw_cot_path)
    samples = []
    for r in rows:
        if r.get("exact_match", 0.0) >= 1.0:
            question = r.get("question", "")
            response = r.get("response", "")
            if question and response:
                samples.append(row_to_sharegpt(question, response, source="raw_correct"))
    return samples


def build_corrected_k2_samples(
    corrected_path: Path,
    prefilled_path: Path,
) -> List[Dict[str, Any]]:
    """Build SFT samples from step-2 corrected trajectories.

    Strategy: use the corrected prefix (k=2 steps from GPT) + model-generated tail
    from the prefilled file. Only keep samples where the full trajectory is correct.
    If prefilled file is unavailable, construct response from corrected + remaining steps.
    """
    samples = []

    # Try prefilled first (has full_response with exact_match)
    prefilled = read_jsonl(prefilled_path)
    if prefilled:
        for r in prefilled:
            if r.get("exact_match", 0.0) >= 1.0:
                question = r.get("question", "")
                response = r.get("full_response", "")
                if question and response:
                    samples.append(row_to_sharegpt(question, response, source="corrected_k2"))
        return samples

    # Fallback: use corrected_k2_gpt.jsonl directly
    corrected = read_jsonl(corrected_path)
    for r in corrected:
        question = r.get("question", "")
        corrected_steps = r.get("corrected_steps", [])
        remaining = r.get("remaining_original_steps", [])
        all_steps = corrected_steps + remaining
        if question and all_steps:
            response = "\n\n".join(all_steps)
            samples.append(row_to_sharegpt(question, response, source="corrected_k2"))

    return samples


def write_json(data: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Wrote {len(data)} samples -> {path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build LlamaFactory SFT data")
    ap.add_argument("--raw-cot",
                    default="results/gsm8k_3b_multi_sample/raw_cot_n8.jsonl")
    ap.add_argument("--corrected-k2",
                    default="results/gsm8k_3b_prefix_correction/corrected_k2_gpt.jsonl")
    ap.add_argument("--prefilled-k2",
                    default="results/gsm8k_3b_prefix_correction/prefilled_k2.jsonl")
    ap.add_argument("--out-dir", default="LlamaFactory/data/sft_gsm8k_3b")
    ap.add_argument("--weight", type=int, default=3,
                    help="Oversampling factor for corrected-k2 samples in weighted scheme")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    raw_cot_path = Path(args.raw_cot)
    corrected_path = Path(args.corrected_k2)
    prefilled_path = Path(args.prefilled_k2)
    out_dir = Path(args.out_dir)

    # Build sample pools
    print("Building correct raw CoT samples...")
    raw_correct = build_correct_raw_samples(raw_cot_path)
    print(f"  Raw correct: {len(raw_correct)}")

    print("Building corrected-k2 samples...")
    corrected_k2 = build_corrected_k2_samples(corrected_path, prefilled_path)
    print(f"  Corrected k2: {len(corrected_k2)}")

    # Scheme 1: Normal SFT (all correct samples, no weighting)
    normal_data = raw_correct + corrected_k2
    random.shuffle(normal_data)
    write_json(normal_data, out_dir / "sft_normal.json")

    # Scheme 2: Weighted SFT (oversample corrected-k2 by weight factor)
    weighted_data = raw_correct + corrected_k2 * args.weight
    random.shuffle(weighted_data)
    write_json(weighted_data, out_dir / "sft_weighted.json")

    # Stats
    print(f"\nSummary:")
    print(f"  Normal:   {len(normal_data)} samples "
          f"(raw={len(raw_correct)}, corrected={len(corrected_k2)})")
    print(f"  Weighted: {len(weighted_data)} samples "
          f"(raw={len(raw_correct)}, corrected={len(corrected_k2)}x{args.weight})")


if __name__ == "__main__":
    main()
