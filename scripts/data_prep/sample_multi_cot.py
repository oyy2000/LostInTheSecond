#!/usr/bin/env python3
"""
Sample N CoT trajectories per GSM8K test question using Qwen2.5-7B-Instruct.

8-GPU data-parallel: each GPU loads a full model (tp=1) and processes 1/8 of
the prompts. Shard outputs are merged at the end.

Usage:
    python scripts/data_prep/sample_multi_cot.py \
        --n-samples 8 \
        --out-file results/gsm8k_7b_v2/raw_cot_n8.jsonl \
        --gpus 0,1,2,3,4,5,6,7
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.prm.scoring import split_steps

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_GSM8K_JSONL = str(
    PROJECT_ROOT
    / "runs/no_vector/gsm8k_cot_zeroshot_unified"
    / "Qwen2.5-7B-Instruct_no_vector/Qwen__Qwen2.5-7B-Instruct"
    / "samples_gsm8k_cot_zeroshot_unified_2026-01-21T11-34-14.746371.jsonl"
)
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Multi-sample CoT generation (vLLM, data-parallel)")
    ap.add_argument("--gsm8k-jsonl", default=DEFAULT_GSM8K_JSONL)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--out-file", default=str(PROJECT_ROOT / "results/gsm8k_7b_v2/raw_cot_n8.jsonl"))
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7",
                    help="Comma-separated GPU IDs for data parallelism")
    ap.add_argument("--limit", type=int, default=0, help="Process first K questions; 0=all")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--min-steps", type=int, default=4,
                    help="Only keep trajectories with >= this many steps")
    # Internal: shard worker mode
    ap.add_argument("--_shard-id", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--_n-shards", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--_shard-out", default="", help=argparse.SUPPRESS)
    ap.add_argument("--_gpu-id", default="0", help=argparse.SUPPRESS)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def extract_gsm8k_gold(answer_text: str) -> str:
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip().replace(",", "")
    return answer_text.strip()


def load_questions(jsonl_path: Path) -> List[Dict[str, Any]]:
    seen: Set[int] = set()
    questions = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("filter") == "strict-match":
                continue
            doc_id = d.get("doc_id", -1)
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = d.get("doc", {})
            question = (doc.get("question") or doc.get("problem") or "").strip()
            gold_answer = extract_gsm8k_gold(doc.get("answer", ""))
            if question:
                questions.append({
                    "doc_id": doc_id,
                    "question": question,
                    "gold_answer": gold_answer,
                })
    return questions


# ---------------------------------------------------------------------------
# Shard worker: runs on a single GPU
# ---------------------------------------------------------------------------

def run_shard(args) -> None:
    """Single-GPU worker: load model, generate for assigned prompts, write shard."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu_id
    shard_id = args._shard_id
    n_shards = args._n_shards

    from src.eval_utils.prompts import build_chat_prompt_from_tokenizer
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    questions = load_questions(Path(args.gsm8k_jsonl))
    if args.limit > 0:
        questions = questions[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    # Build ALL prompts, then take this shard's slice
    all_prompts = []
    all_meta = []
    for qi, q in enumerate(questions):
        chat_prompt = build_chat_prompt_from_tokenizer(tokenizer, q["question"])
        for si in range(args.n_samples):
            all_prompts.append(chat_prompt)
            all_meta.append((qi, si))

    # Shard by prompt index
    my_indices = list(range(shard_id, len(all_prompts), n_shards))
    my_prompts = [all_prompts[i] for i in my_indices]
    my_meta = [all_meta[i] for i in my_indices]

    print(f"[Shard {shard_id}/{n_shards}] GPU {args._gpu_id}: "
          f"{len(my_prompts)} prompts (of {len(all_prompts)} total)")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    llm = LLM(
        model=args.model_id,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="half",
    )

    outputs = llm.generate(my_prompts, sampling_params)

    # Write shard output
    out_path = Path(args._shard_out)
    n_total, n_correct, n_kept = 0, 0, 0
    with out_path.open("w", encoding="utf-8") as fout:
        for output, (qi, si) in zip(outputs, my_meta):
            q = questions[qi]
            response = output.outputs[0].text.strip()
            steps = split_steps(response, mode="double_newline")
            pred_answer = extract_boxed_answer(response)
            gold = q["gold_answer"]
            is_correct = float(normalize_answer(pred_answer) == normalize_answer(gold)) if gold else 0.0

            n_total += 1
            n_correct += int(is_correct >= 1.0)

            if len(steps) < args.min_steps:
                continue

            row = {
                "doc_id": q["doc_id"],
                "question": q["question"],
                "gold_answer": gold,
                "sample_idx": si,
                "response": response,
                "steps": steps,
                "n_steps": len(steps),
                "pred_answer": pred_answer,
                "exact_match": is_correct,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_kept += 1

    acc = n_correct / max(n_total, 1)
    print(f"[Shard {shard_id}] Done. total={n_total}, correct={n_correct} ({acc:.3f}), "
          f"kept={n_kept} -> {out_path}")


# ---------------------------------------------------------------------------
# Coordinator: launch shards, merge
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # If we're a shard worker, just run the shard
    if args._shard_id >= 0:
        run_shard(args)
        return

    # --- Coordinator mode ---
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    n_shards = len(gpu_ids)
    print(f"Data-parallel sampling: {n_shards} GPUs {gpu_ids}")

    questions = load_questions(Path(args.gsm8k_jsonl))
    if args.limit > 0:
        questions = questions[: args.limit]
    total_prompts = len(questions) * args.n_samples
    print(f"Questions: {len(questions)}, samples/q: {args.n_samples}, "
          f"total prompts: {total_prompts}")

    # Create shard output files
    shard_dir = out_path.parent / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_files = [shard_dir / f"shard_{i}.jsonl" for i in range(n_shards)]

    # Launch subprocesses
    script_path = str(Path(__file__).resolve())
    procs = []
    for i, gpu_id in enumerate(gpu_ids):
        cmd = [
            sys.executable, script_path,
            "--gsm8k-jsonl", args.gsm8k_jsonl,
            "--model-id", args.model_id,
            "--n-samples", str(args.n_samples),
            "--temperature", str(args.temperature),
            "--top-p", str(args.top_p),
            "--max-tokens", str(args.max_tokens),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--max-model-len", str(args.max_model_len),
            "--min-steps", str(args.min_steps),
            "--_shard-id", str(i),
            "--_n-shards", str(n_shards),
            "--_shard-out", str(shard_files[i]),
            "--_gpu-id", gpu_id,
        ]
        if args.limit > 0:
            cmd.extend(["--limit", str(args.limit)])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["TOKENIZERS_PARALLELISM"] = "false"

        print(f"  Launching shard {i} on GPU {gpu_id}...")
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        procs.append((i, gpu_id, p))

    # Wait for all
    failed = []
    for i, gpu_id, p in procs:
        stdout, _ = p.communicate()
        output_text = stdout.decode("utf-8", errors="replace") if stdout else ""
        rc = p.returncode
        # Print last few lines
        lines = output_text.strip().splitlines()
        tail = "\n".join(lines[-5:]) if lines else "(no output)"
        print(f"\n--- Shard {i} (GPU {gpu_id}) exit={rc} ---\n{tail}")
        if rc != 0:
            failed.append(i)
            # Print more on failure
            if len(lines) > 5:
                print("...\n" + "\n".join(lines[-20:]))

    if failed:
        print(f"\nERROR: Shards {failed} failed!")
        sys.exit(1)

    # Merge shard outputs
    print(f"\nMerging {n_shards} shards...")
    n_total = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for sf in shard_files:
            if sf.exists():
                with sf.open("r", encoding="utf-8") as fin:
                    for line in fin:
                        if line.strip():
                            fout.write(line)
                            n_total += 1

    # Count stats
    n_correct = 0
    n_wrong = 0
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if d.get("exact_match", 0.0) >= 1.0:
                    n_correct += 1
                else:
                    n_wrong += 1

    print(f"\nMerged: {n_total} trajectories (correct={n_correct}, wrong={n_wrong})")
    print(f"Output: {out_path}")

    # Cleanup shards
    for sf in shard_files:
        sf.unlink(missing_ok=True)
    shard_dir.rmdir()
    print("Shard files cleaned up.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
