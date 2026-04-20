#!/usr/bin/env python3
"""
Compute per-step Information Gain (IG) for reasoning trajectories (v2).

Uses tokenizer.apply_chat_template(). Supports corrupted conditions.

For each trajectory and each step t:
  IG(t) = log P(gold_answer | prefix_{1:t}) - log P(gold_answer | prefix_{1:t-1})

Usage:
    python scripts/eval/compute_step_ig.py \
        --cot-file results/gsm8k_7b_v2/raw_cot_n8.jsonl \
        --correction-dir results/gsm8k_7b_v2 \
        --out-dir results/gsm8k_7b_v2/information_gain \
        --gpus auto
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

for _i, _arg in enumerate(sys.argv):
    if _arg == "--gpus" and _i + 1 < len(sys.argv) and sys.argv[_i + 1] != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_i + 1]
        break

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prm.scoring import split_steps
from src.eval_utils.prompts import build_chat_prompt_from_tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute per-step information gain (v2)")
    ap.add_argument("--cot-file",
                    default=str(PROJECT_ROOT / "results/gsm8k_7b_v2/raw_cot_n8.jsonl"))
    ap.add_argument("--correction-dir",
                    default=str(PROJECT_ROOT / "results/gsm8k_7b_v2"))
    ap.add_argument("--out-dir",
                    default=str(PROJECT_ROOT / "results/gsm8k_7b_v2/information_gain"))
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--k-values", default="1,2,3,4")
    ap.add_argument("--max-samples", type=int, default=300)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--gpus", default="auto")
    ap.add_argument("--condition", default="all",
                    help="all, correct, wrong, corrected-K, corrupted-K")
    return ap.parse_args()


def select_best_gpu(requested: str, min_free: int = 12000) -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            encoding="utf-8",
        )
        free = {i: int(l.strip()) for i, l in enumerate(out.strip().splitlines()) if l.strip()}
    except Exception:
        return 0
    if requested != "auto":
        ids = [int(x) for x in requested.split(",") if x.strip()]
        usable = {g: free.get(g, 0) for g in ids if free.get(g, 0) >= min_free}
        if usable:
            return max(usable, key=usable.get)
        return ids[0] if ids else 0
    candidates = {g: m for g, m in free.items() if m >= min_free}
    if candidates:
        return max(candidates, key=candidates.get)
    return max(free, key=free.get) if free else 0


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


@torch.inference_mode()
def compute_ig_for_trajectory_fast(
    model, tokenizer, question: str, steps: List[str],
    gold_answer: str, max_seq_len: int,
) -> List[float]:
    """Compute IG for each step using batched forward passes."""
    if not steps or not gold_answer:
        return []

    prompt = build_chat_prompt_from_tokenizer(tokenizer, question)
    answer_suffix = gold_answer + "}"
    answer_bridge = "\n\nThe answer is \\boxed{"

    # Build prefixes: prefix_0 = prompt, prefix_t = prompt + steps[0:t]
    prefixes = [prompt]
    current = prompt
    for step in steps:
        current += step + "\n\n"
        prefixes.append(current)

    # For each prefix, compute log P(answer_suffix | prefix + answer_bridge)
    log_probs = []
    answer_ids = tokenizer.encode(answer_suffix, add_special_tokens=False)
    if not answer_ids:
        return []

    # Batch: process all prefixes at once if they fit
    batch_texts = []
    for prefix in prefixes:
        full = prefix + answer_bridge + answer_suffix
        batch_texts.append(full)

    # Process in mini-batches to avoid OOM
    batch_size = 1
    all_log_probs = []
    for bi in range(0, len(batch_texts), batch_size):
        batch = batch_texts[bi:bi + batch_size]
        encodings = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=max_seq_len,
                               add_special_tokens=False)
        input_ids = encodings["input_ids"].to(model.device)
        attention_mask = encodings["attention_mask"].to(model.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch, seq_len, vocab)

        for j in range(len(batch)):
            # Find where answer_suffix starts in this sequence
            seq_ids = input_ids[j]
            mask = attention_mask[j]
            seq_len = mask.sum().item()

            # The answer tokens are the last len(answer_ids) tokens
            n_ans = len(answer_ids)
            if seq_len < n_ans + 1:
                all_log_probs.append(float("-inf"))
                continue

            # Log prob of answer tokens
            log_prob = 0.0
            for ai, aid in enumerate(answer_ids):
                pos = seq_len - n_ans + ai - 1  # position of token predicting aid
                if pos < 0 or pos >= logits.shape[1]:
                    log_prob = float("-inf")
                    break
                token_logits = logits[j, pos]
                log_softmax = torch.log_softmax(token_logits, dim=-1)
                log_prob += log_softmax[aid].item()

            all_log_probs.append(log_prob)

    # IG(t) = log_prob[t] - log_prob[t-1]
    ig_values = []
    for t in range(1, len(all_log_probs)):
        ig = all_log_probs[t] - all_log_probs[t - 1]
        if math.isinf(ig) or math.isnan(ig):
            ig = 0.0
        ig_values.append(ig)

    return ig_values


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    correction_dir = Path(args.correction_dir)
    k_values = [int(x) for x in args.k_values.split(",")]

    # Build conditions
    conditions = {}
    score_all = args.condition == "all"

    cot_rows = read_jsonl(Path(args.cot_file))

    if score_all or args.condition == "correct":
        conditions["correct"] = [
            r for r in cot_rows if r.get("exact_match", 0.0) >= 1.0
        ]
    if score_all or args.condition == "wrong":
        conditions["wrong_original"] = [
            r for r in cot_rows if r.get("exact_match", 1.0) < 1.0
        ]

    for k in k_values:
        cond = f"corrected_k{k}"
        if score_all or args.condition == f"corrected-{k}":
            rows = read_jsonl(correction_dir / f"prefilled_corrected_k{k}.jsonl")
            for r in rows:
                r["steps"] = r.get("all_steps") or split_steps(r.get("full_response", ""), mode="double_newline")
                r["response"] = r.get("full_response", "")
            conditions[cond] = rows

        cond = f"corrupted_k{k}"
        if score_all or args.condition == f"corrupted-{k}":
            rows = read_jsonl(correction_dir / f"prefilled_corrupted_k{k}.jsonl")
            for r in rows:
                r["steps"] = r.get("all_steps") or split_steps(r.get("full_response", ""), mode="double_newline")
                r["response"] = r.get("full_response", "")
            conditions[cond] = rows

    print(f"Conditions: {list(conditions.keys())}")

    # Load model
    gpu_id = select_best_gpu(args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Using GPU {gpu_id}")

    dtype = getattr(torch, args.dtype, torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    for cond_name, items in conditions.items():
        out_file = out_dir / f"ig_{cond_name}.jsonl"
        if out_file.exists():
            print(f"[SKIP] {out_file} exists")
            continue

        subset = items[: args.max_samples]
        print(f"\n--- {cond_name}: {len(subset)} samples ---")

        results = []
        for item in tqdm(subset, desc=cond_name):
            question = item["question"]
            steps = item.get("steps") or split_steps(item.get("response", ""), mode="double_newline")
            gold = item.get("gold_answer", "")

            if not steps or not gold:
                continue

            ig_values = compute_ig_for_trajectory_fast(
                model, tokenizer, question, steps, gold, args.max_seq_len,
            )

            if ig_values:
                results.append({
                    "doc_id": item["doc_id"],
                    "sample_idx": item.get("sample_idx", 0),
                    "condition": cond_name,
                    "exact_match": item.get("exact_match", 0.0),
                    "n_steps": len(steps),
                    "ig_values": ig_values,
                    "cumulative_ig": sum(ig_values),
                })

        with out_file.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"Saved {len(results)} -> {out_file}")

    # Quick IG comparison plot
    _plot_ig_comparison(out_dir, conditions.keys())

    del model
    torch.cuda.empty_cache()
    print("\nAll IG computation done.")


def _plot_ig_comparison(out_dir: Path, condition_names) -> None:
    """Quick IG comparison plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    max_steps = 10
    colors = {
        "correct": "#2ca02c", "wrong_original": "#d62728",
        "corrected_k1": "#1f77b4", "corrected_k2": "#ff7f0e",
        "corrected_k3": "#9467bd", "corrected_k4": "#8c564b",
        "corrupted_k1": "#e377c2", "corrupted_k2": "#7f7f7f",
        "corrupted_k3": "#bcbd22", "corrupted_k4": "#17becf",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for cond in condition_names:
        ig_file = out_dir / f"ig_{cond}.jsonl"
        if not ig_file.exists():
            continue
        results = []
        with ig_file.open() as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

        ig_by_step = {}
        cum_by_step = {}
        for r in results:
            cum = 0.0
            for i, ig in enumerate(r["ig_values"]):
                if i < max_steps:
                    ig_by_step.setdefault(i, []).append(ig)
                    cum += ig
                    cum_by_step.setdefault(i, []).append(cum)

        xs = sorted(ig_by_step.keys())
        color = colors.get(cond, "tab:gray")

        ys_ig = [np.mean(ig_by_step[x]) for x in xs]
        ax1.plot([x+1 for x in xs], ys_ig, "o-", color=color, label=cond, linewidth=2)

        ys_cum = [np.mean(cum_by_step[x]) for x in xs]
        ax2.plot([x+1 for x in xs], ys_cum, "o-", color=color, label=cond, linewidth=2)

    ax1.axhline(y=0, linestyle="--", color="gray", alpha=0.5)
    ax1.set_ylabel("Average IG (nats)")
    ax1.set_xlabel("Step")
    ax1.set_title("Per-Step Information Gain")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8)

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Cumulative log P(answer)")
    ax2.set_title("Cumulative Information")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "ig_curve_comparison.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
