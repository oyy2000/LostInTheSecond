#!/usr/bin/env python3
"""
Build control prefixes for the prefix-correction experiment.

Control condition: for each wrong trajectory, replace the first k steps with
correct steps from a DIFFERENT problem (length-matched), then let the model
continue. Uses vLLM for fast batched generation.

Usage:
    python scripts/data_prep/build_control_prefix.py \
        --cot-file results/gsm8k_3b_multi_sample/raw_cot_n8.jsonl \
        --correction-dir results/gsm8k_3b_prefix_correction \
        --k-values 1,2,3,4 \
        --tp 1
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.prm.scoring import split_steps

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build control prefixes (vLLM)")
    ap.add_argument("--cot-file", required=True)
    ap.add_argument("--correction-dir",
                    default=str(PROJECT_ROOT / "results/gsm8k_3b_prefix_correction"))
    ap.add_argument("--k-values", default="1,2,3,4")
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


def build_chat_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{question.strip()}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def extract_boxed_answer(text: str) -> str:
    idx = (text or "").rfind("\\boxed")
    if idx < 0:
        return ""
    i, depth, start = idx, 0, None
    while i < len(text):
        if text[i] == "{":
            if depth == 0:
                start = i
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start + 1 : i].strip()
        i += 1
    return ""


def normalize_answer(text: str) -> str:
    text = (text or "").strip().replace("$", "").replace(",", "")
    text = re.sub(r"\\boxed\{(.*)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    return text.lower()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
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


def find_length_matched_donor(
    target_steps: List[str],
    k: int,
    donor_pool: List[Tuple[int, List[str]]],
    target_doc_id: int,
    rng: random.Random,
) -> List[str]:
    """Find donor steps from a different problem, length-matched to target's first k steps."""
    target_len = sum(len(s) for s in target_steps[:k])
    best_donor = None
    best_diff = float("inf")

    candidates = [d for d in donor_pool if d[0] != target_doc_id and len(d[1]) >= k]
    if not candidates:
        candidates = [d for d in donor_pool if len(d[1]) >= k]

    rng.shuffle(candidates)
    for doc_id, steps in candidates[:50]:
        donor_len = sum(len(s) for s in steps[:k])
        diff = abs(donor_len - target_len)
        if diff < best_diff:
            best_diff = diff
            best_donor = steps[:k]
        if diff < target_len * 0.1:
            break

    return best_donor or (candidates[0][1][:k] if candidates else target_steps[:k])


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    correction_dir = Path(args.correction_dir)
    correction_dir.mkdir(parents=True, exist_ok=True)
    k_values = [int(x) for x in args.k_values.split(",")]

    # Load all CoT data
    all_rows = read_jsonl(Path(args.cot_file))
    wrong_rows = [r for r in all_rows if r.get("exact_match", 1.0) < 1.0]
    correct_rows = [r for r in all_rows if r.get("exact_match", 0.0) >= 1.0]
    print(f"Loaded {len(all_rows)} total, {len(wrong_rows)} wrong, {len(correct_rows)} correct")

    # Build donor pool from correct trajectories
    donor_pool: List[Tuple[int, List[str]]] = []
    for r in correct_rows:
        steps = r.get("steps", split_steps(r.get("response", ""), mode="double_newline"))
        if len(steps) >= 2:
            donor_pool.append((r["doc_id"], steps))
    print(f"Donor pool: {len(donor_pool)} correct trajectories")

    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    print(f"Loading vLLM model {args.model_id} (tp={args.tp})...")
    llm = LLM(
        model=args.model_id,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
        dtype="half",
    )

    for k in k_values:
        out_path = correction_dir / f"control_k{k}.jsonl"

        eligible = [r for r in wrong_rows
                    if r.get("n_steps", len(r.get("steps", []))) >= k + 1]
        if args.limit > 0:
            eligible = eligible[: args.limit]

        if not eligible:
            print(f"[SKIP] No eligible rows for control k={k}")
            continue

        # Build all prompts with cross-problem donor prefixes
        prompts = []
        meta = []
        for row in eligible:
            steps = row.get("steps", split_steps(row.get("response", ""), mode="double_newline"))
            donor_steps = find_length_matched_donor(steps, k, donor_pool, row["doc_id"], rng)
            prefix_text = "\n\n".join(donor_steps)
            prompt = build_chat_prompt(row["question"]) + prefix_text + "\n\n"
            prompts.append(prompt)
            meta.append({
                "row": row,
                "donor_steps": donor_steps,
                "original_steps": steps,
            })

        print(f"\n=== Control k={k}: {len(prompts)} prompts ===")
        outputs = llm.generate(prompts, sampling_params)

        n_ok, n_correct = 0, 0
        with out_path.open("w", encoding="utf-8") as fout:
            for m, output in zip(meta, outputs):
                row = m["row"]
                tail = output.outputs[0].text.strip()
                donor_steps = m["donor_steps"]
                prefix_text = "\n\n".join(donor_steps)
                full_response = prefix_text + ("\n\n" + tail if tail else "")
                all_steps = split_steps(full_response, mode="double_newline")
                pred_answer = extract_boxed_answer(full_response)
                gold = row.get("gold_answer", "")
                is_correct = float(normalize_answer(pred_answer) == normalize_answer(gold)) if gold else 0.0

                out_row = {
                    "doc_id": row["doc_id"],
                    "sample_idx": row.get("sample_idx", 0),
                    "k": k,
                    "condition": "control",
                    "question": row["question"],
                    "gold_answer": gold,
                    "control_steps": donor_steps,
                    "tail": tail,
                    "full_response": full_response,
                    "all_steps": all_steps,
                    "n_steps": len(all_steps),
                    "pred_answer": pred_answer,
                    "exact_match": is_correct,
                    "original_response": row.get("response", ""),
                    "original_steps": m["original_steps"],
                }
                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                n_ok += 1
                n_correct += int(is_correct >= 1.0)

        acc = n_correct / max(n_ok, 1)
        print(f"Control k={k}: {n_ok} done, {n_correct} correct ({acc:.3f}) -> {out_path}")

    print("\nAll control prefixes done.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
