#!/usr/bin/env python3
"""
Extract hidden states at step boundaries for all trajectory conditions (v2).

Uses tokenizer.apply_chat_template() for prompt construction.
Conditions: original, corrected_k{1-4}, corrupted_k{1-4} (no control).

Usage:
    python scripts/eval/extract_step_hidden_states.py \
        --cot-file results/gsm8k_7b_v2/raw_cot_n8.jsonl \
        --correction-dir results/gsm8k_7b_v2 \
        --out-dir results/gsm8k_7b_v2/hidden_states \
        --gpus auto --max-samples 500
"""

import argparse
import json
import os
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
N_LAYERS = 28
HIDDEN_DIM = 3584

LAYER_INDICES = list(range(0, N_LAYERS, 4))
if (N_LAYERS - 1) not in LAYER_INDICES:
    LAYER_INDICES.append(N_LAYERS - 1)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract hidden states at step boundaries (v2)")
    ap.add_argument("--cot-file",
                    default=str(PROJECT_ROOT / "results/gsm8k_7b_v2/raw_cot_n8.jsonl"))
    ap.add_argument("--correction-dir",
                    default=str(PROJECT_ROOT / "results/gsm8k_7b_v2"))
    ap.add_argument("--out-dir",
                    default=str(PROJECT_ROOT / "results/gsm8k_7b_v2/hidden_states"))
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--k-values", default="1,2,3,4")
    ap.add_argument("--max-samples", type=int, default=500)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--gpus", default="auto")
    ap.add_argument("--condition", default="all",
                    help="all, original, corrected_k1..4, corrupted_k1..4")
    ap.add_argument("--layer-indices", default="")
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


@torch.inference_mode()
def extract_hidden_states(
    model, tokenizer, question: str, response: str, steps: List[str],
    layer_indices: List[int], max_seq_len: int,
) -> Optional[torch.Tensor]:
    """Extract hidden states at step boundaries.

    Returns tensor of shape (n_steps, n_layers, hidden_dim) or None.
    """
    prompt = build_chat_prompt_from_tokenizer(tokenizer, question)
    full_text = prompt + response

    input_ids = tokenizer.encode(full_text, return_tensors="pt",
                                  add_special_tokens=False,
                                  truncation=True, max_length=max_seq_len)
    input_ids = input_ids.to(model.device)

    if input_ids.shape[1] < 10:
        return None

    outputs = model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # tuple of (1, seq_len, hidden_dim)

    # Find step boundary positions
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(prompt_ids)

    boundary_positions = []
    current_text = prompt
    for step in steps:
        current_text += step
        step_end_ids = tokenizer.encode(current_text, add_special_tokens=False)
        pos = min(len(step_end_ids) - 1, input_ids.shape[1] - 1)
        if pos >= prompt_len:
            boundary_positions.append(pos)
        if "\n\n" in response:
            current_text += "\n\n"

    if not boundary_positions:
        return None

    # Extract hidden states at boundary positions
    n_steps = len(boundary_positions)
    n_layers = len(layer_indices)
    hs = torch.zeros(n_steps, n_layers, HIDDEN_DIM)

    for si, pos in enumerate(boundary_positions):
        for li, layer_idx in enumerate(layer_indices):
            if layer_idx < len(hidden_states):
                hs[si, li] = hidden_states[layer_idx][0, pos].float().cpu()

    return hs


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    correction_dir = Path(args.correction_dir)
    k_values = [int(x) for x in args.k_values.split(",")]

    layer_indices = LAYER_INDICES
    if args.layer_indices:
        layer_indices = [int(x) for x in args.layer_indices.split(",")]

    # Build conditions
    conditions = {}
    score_all = args.condition == "all"

    if score_all or args.condition == "original":
        cot_rows = read_jsonl(Path(args.cot_file))
        items = []
        for r in cot_rows:
            steps = r.get("steps") or split_steps(r.get("response", ""), mode="double_newline")
            items.append({
                "doc_id": r["doc_id"],
                "sample_idx": r.get("sample_idx", 0),
                "question": r["question"],
                "response": r.get("response", ""),
                "steps": steps,
                "exact_match": r.get("exact_match", 0.0),
            })
        conditions["original"] = items

    for k in k_values:
        cond = f"corrected_k{k}"
        if score_all or args.condition == cond:
            rows = read_jsonl(correction_dir / f"prefilled_corrected_k{k}.jsonl")
            items = []
            for r in rows:
                steps = r.get("all_steps") or split_steps(r.get("full_response", ""), mode="double_newline")
                items.append({
                    "doc_id": r["doc_id"],
                    "sample_idx": r.get("sample_idx", 0),
                    "question": r["question"],
                    "response": r.get("full_response", ""),
                    "steps": steps,
                    "exact_match": r.get("exact_match", 0.0),
                })
            conditions[cond] = items

        cond = f"corrupted_k{k}"
        if score_all or args.condition == cond:
            rows = read_jsonl(correction_dir / f"prefilled_corrupted_k{k}.jsonl")
            items = []
            for r in rows:
                steps = r.get("all_steps") or split_steps(r.get("full_response", ""), mode="double_newline")
                items.append({
                    "doc_id": r["doc_id"],
                    "sample_idx": r.get("sample_idx", 0),
                    "question": r["question"],
                    "response": r.get("full_response", ""),
                    "steps": steps,
                    "exact_match": r.get("exact_match", 0.0),
                })
            conditions[cond] = items

    print(f"Conditions: {list(conditions.keys())}")
    for c, items in conditions.items():
        print(f"  {c}: {len(items)} items")

    # Load model
    gpu_id = select_best_gpu(args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Using GPU {gpu_id}")

    dtype = getattr(torch, args.dtype, torch.float16)
    print(f"Loading {args.model_id} ({args.dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    for cond_name, items in conditions.items():
        out_file = out_dir / f"hs_{cond_name}.pt"
        meta_file = out_dir / f"meta_{cond_name}.jsonl"

        if out_file.exists():
            print(f"[SKIP] {out_file} already exists")
            continue

        subset = items[: args.max_samples]
        print(f"\n--- {cond_name}: {len(subset)} samples ---")

        all_hs = []
        meta_rows = []

        for item in tqdm(subset, desc=cond_name):
            hs = extract_hidden_states(
                model, tokenizer,
                item["question"], item["response"], item["steps"],
                layer_indices, args.max_seq_len,
            )
            if hs is None:
                continue

            all_hs.append(hs)
            meta_rows.append({
                "idx": len(all_hs) - 1,
                "doc_id": item["doc_id"],
                "sample_idx": item.get("sample_idx", 0),
                "exact_match": item.get("exact_match", 0.0),
                "n_steps": hs.shape[0],
            })

            if len(all_hs) % 50 == 0:
                torch.cuda.empty_cache()

        torch.save({
            "hidden_states": all_hs,
            "layer_indices": layer_indices,
            "model_id": args.model_id,
            "condition": cond_name,
            "n_samples": len(all_hs),
        }, out_file)

        with meta_file.open("w", encoding="utf-8") as f:
            for row in meta_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"Saved {len(all_hs)} samples -> {out_file}")

    del model
    torch.cuda.empty_cache()
    print("\nAll hidden state extraction done.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
