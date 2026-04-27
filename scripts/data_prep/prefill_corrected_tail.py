#!/usr/bin/env python3
"""
Prefill corrected/corrupted first-k steps and let Qwen2.5-7B-Instruct continue.

Single-GPU vLLM inference, processes all k-values sequentially.
Model is loaded once and reused across k-values.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python scripts/data_prep/prefill_corrected_tail.py \
        --correction-dir results/gsm8k_7b_v2 \
        --mode corrected \
        --k-values 1,2,3,4 \
        --tp 2
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.prm.scoring import split_steps
from src.eval_utils.prompts import build_chat_prompt_from_tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prefill prefix and generate tail (vLLM)")
    ap.add_argument("--correction-dir",
                    default=str(PROJECT_ROOT / "results/gsm8k_7b_v2"))
    ap.add_argument("--mode", choices=["corrected", "corrupted"], default="corrected")
    ap.add_argument("--k-values", default="1,2,3,4")
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--tp", type=int, default=2,
                    help="Tensor parallel size (use 2 for 7B on RTX 6000)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


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


def main() -> None:
    args = parse_args()
    correction_dir = Path(args.correction_dir)
    correction_dir.mkdir(parents=True, exist_ok=True)
    k_values = [int(x) for x in args.k_values.split(",")]

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

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
        max_model_len=args.max_model_len,
        dtype="half",
    )
    print("Model loaded.")

    prefix_key = "corrected_steps" if args.mode == "corrected" else "corrupted_steps"
    in_pattern = f"{args.mode}_k{{k}}_gpt.jsonl"
    out_pattern = f"prefilled_{args.mode}_k{{k}}.jsonl"

    for k in k_values:
        in_path = correction_dir / in_pattern.format(k=k)
        out_path = correction_dir / out_pattern.format(k=k)

        rows = read_jsonl(in_path)
        if not rows:
            print(f"[SKIP] No data for {args.mode} k={k} at {in_path}")
            continue

        if args.limit > 0:
            rows = rows[: args.limit]

        # Build prompts
        prompts = []
        for row in rows:
            prefix_steps = row.get(prefix_key, [])
            prefix_text = "\n\n".join(prefix_steps)
            chat_prompt = build_chat_prompt_from_tokenizer(tokenizer, row["question"])
            prompts.append(chat_prompt + prefix_text + "\n\n")

        print(f"\n=== Prefill {args.mode} k={k}: {len(prompts)} prompts ===")
        outputs = llm.generate(prompts, sampling_params)

        n_ok, n_correct = 0, 0
        with out_path.open("w", encoding="utf-8") as fout:
            for row, output in zip(rows, outputs):
                tail = output.outputs[0].text.strip()
                prefix_steps = row.get(prefix_key, [])
                prefix_text = "\n\n".join(prefix_steps)
                full_response = prefix_text + ("\n\n" + tail if tail else "")
                all_steps = split_steps(full_response, mode="double_newline")
                pred_answer = extract_boxed_answer(full_response)
                gold = row.get("gold_answer", "")
                is_correct = float(normalize_answer(pred_answer) == normalize_answer(gold)) if gold else 0.0

                out_row = {
                    "doc_id": row["doc_id"],
                    "sample_idx": row.get("sample_idx", 0),
                    "k": k,
                    "condition": args.mode,
                    "question": row["question"],
                    "gold_answer": gold,
                    prefix_key: prefix_steps,
                    "tail": tail,
                    "full_response": full_response,
                    "all_steps": all_steps,
                    "n_steps": len(all_steps),
                    "pred_answer": pred_answer,
                    "exact_match": is_correct,
                    "original_response": row.get("original_response", ""),
                    "original_steps": row.get("original_steps", []),
                }
                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                n_ok += 1
                n_correct += int(is_correct >= 1.0)

        acc = n_correct / max(n_ok, 1)
        print(f"Prefill {args.mode} k={k}: {n_ok} done, {n_correct} correct ({acc:.3f}) -> {out_path}")

    print("\nAll prefill done.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
