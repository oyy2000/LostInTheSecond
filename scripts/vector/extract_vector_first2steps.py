#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract steering vector using ONLY the first two reasoning steps.

Input dataset format (JSON or JSONL) is compatible with existing DS2 files:
- doc.question
- pos_response / neg_response
- optional pos_steps / neg_steps
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from typing import Any, Dict, List, Tuple

import torch
from steering_vectors import train_steering_vector
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval_utils.prompts import qwen_chat_prompt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract steering vector from first 2 steps only")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument(
        "--dataset-path",
        default="./artifacts/samples_math500_ds2_fix_step2_gpt_prefill.json",
        help="Input dataset JSON/JSONL",
    )
    ap.add_argument(
        "--out-dir",
        default="./artifacts/vectors_first2steps",
        help="Output root directory",
    )
    ap.add_argument(
        "--layers",
        default="all",
        help="Comma-separated layer ids, or 'all'",
    )
    ap.add_argument("--read-token-index", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-samples", type=int, default=0, help="0 means all")
    ap.add_argument("--min-doc-id", type=int, default=-1)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    return ap.parse_args()


def load_samples(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        for key in ("samples", "instances", "data"):
            if isinstance(obj.get(key), list):
                return obj[key]
    if isinstance(obj, list):
        return obj
    raise ValueError("Unsupported dataset format")


def split_steps(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if "\n\n" in text:
        parts = [x.strip() for x in text.split("\n\n") if x.strip()]
        if parts:
            return parts
    return [x.strip() for x in text.split("\n") if x.strip()]


def get_exact_match(ex: Dict[str, Any]) -> float:
    if "exact_match" in ex:
        try:
            return float(ex["exact_match"])
        except Exception:
            return 0.0
    for key in ("results", "metrics", "scores"):
        block = ex.get(key)
        if isinstance(block, dict) and "exact_match" in block:
            try:
                return float(block["exact_match"])
            except Exception:
                return 0.0
    return 0.0


def first_two_steps(item: Dict[str, Any], field_resp: str, field_steps: str) -> str:
    if isinstance(item.get(field_steps), list) and item[field_steps]:
        steps = [str(x).strip() for x in item[field_steps] if str(x).strip()]
    else:
        steps = split_steps(item.get(field_resp, ""))
    if not steps:
        return ""
    return "\n\n".join(steps[:2]).strip()


def parse_layers(layers_arg: str, model) -> List[int]:
    if layers_arg.strip().lower() == "all":
        n = int(getattr(model.config, "num_hidden_layers"))
        return [i for i in range(n)]
    out = []
    for p in layers_arg.split(","):
        p = p.strip()
        if p:
            out.append(int(p))
    return sorted(set(out))


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    print(f"[08] Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    layers = parse_layers(args.layers, model)
    print(f"[08] Layers: {layers}")

    samples = load_samples(dataset_path)

    training_samples: List[Tuple[str, str]] = []
    used_doc_ids = []

    for item in samples:
        if args.max_samples > 0 and len(training_samples) >= args.max_samples:
            break

        if get_exact_match(item) < 1.0:
            continue

        doc = item.get("doc") or {}
        question = (doc.get("question") or doc.get("problem") or "").strip()
        doc_id = int(doc.get("id", -1))
        if args.min_doc_id >= 0 and doc_id <= args.min_doc_id:
            continue

        pos_first2 = first_two_steps(item, "pos_response", "pos_steps")
        neg_first2 = first_two_steps(item, "neg_response", "neg_steps")
        if not question or not pos_first2 or not neg_first2:
            continue

        prompt = qwen_chat_prompt(question)
        training_samples.append((prompt + pos_first2, prompt + neg_first2))
        used_doc_ids.append(doc_id)

    if not training_samples:
        raise ValueError("No valid training samples after filtering")

    print(f"[08] Training pairs: {len(training_samples)}")
    steering_vector = train_steering_vector(
        model=model,
        tokenizer=tokenizer,
        training_samples=training_samples,
        layers=layers,
        layer_type="decoder_block",
        move_to_cpu=True,
        read_token_index=args.read_token_index,
        show_progress=True,
        batch_size=args.batch_size,
    )

    model_tag = args.model_id.replace("/", "_")
    out_dir = out_root / f"{model_tag}_first2steps"
    out_dir.mkdir(parents=True, exist_ok=True)

    vec_path = out_dir / "steering_vector.pt"
    meta_path = out_dir / "extract_meta.json"

    torch.save(steering_vector, vec_path)
    meta = {
        "model_id": args.model_id,
        "dataset_path": str(dataset_path),
        "num_pairs": len(training_samples),
        "layers": layers,
        "read_token_index": args.read_token_index,
        "min_doc_id": args.min_doc_id,
        "max_samples": args.max_samples,
        "dtype": args.dtype,
        "notes": "Vector extracted from first two steps only",
        "used_doc_ids": used_doc_ids,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[08] Saved vector -> {vec_path}")
    print(f"[08] Saved meta   -> {meta_path}")


if __name__ == "__main__":
    main()
