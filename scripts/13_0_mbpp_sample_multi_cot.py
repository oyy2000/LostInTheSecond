#!/usr/bin/env python3
"""
Sample N code solutions per MBPP problem using vLLM.

Usage:
    python scripts/13_0_mbpp_sample_multi_cot.py \
        --n-samples 8 --gpus 0,1,2,3,4,5,6,7
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.code_answer_equiv import (
    check_mbpp, split_code_blocks, extract_python_code,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Write a Python function to solve the task below. "
    "Output ONLY the Python code, no explanation."
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--out-file", default=str(
        PROJECT_ROOT / "results/mbpp_3b_multi_sample/raw_code_n8.jsonl"))
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--min-blocks", type=int, default=3)
    ap.add_argument("--exec-timeout", type=float, default=5.0)
    ap.add_argument("--_shard-id", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--_n-shards", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--_shard-out", default="", help=argparse.SUPPRESS)
    ap.add_argument("--_gpu-id", default="0", help=argparse.SUPPRESS)
    ap.add_argument("--_questions-file", default="", help=argparse.SUPPRESS)
    return ap.parse_args()


def build_chat_prompt(task_prompt):
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{task_prompt}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_mbpp(limit=0):
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    questions = []
    for i, row in enumerate(ds):
        if limit > 0 and i >= limit:
            break
        questions.append({
            "doc_id": f"MBPP/{row['task_id']}",
            "prompt_text": row["prompt"],
            "canonical_code": row["code"],
            "test_list": row["test_list"],
            "test_imports": row.get("test_imports", []),
        })
    return questions


# -- shard worker (runs on one GPU) --

def run_shard(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu_id
    from vllm import LLM, SamplingParams
    questions = json.loads(Path(args._questions_file).read_text("utf-8"))
    all_prompts, all_meta = [], []
    for qi, q in enumerate(questions):
        p = build_chat_prompt(q["prompt_text"])
        for si in range(args.n_samples):
            all_prompts.append(p)
            all_meta.append((qi, si))
    my_idx = list(range(args._shard_id, len(all_prompts), args._n_shards))
    my_prompts = [all_prompts[i] for i in my_idx]
    my_meta = [all_meta[i] for i in my_idx]
    print(f"[Shard {args._shard_id}] GPU {args._gpu_id}: {len(my_prompts)} prompts")
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        stop=["<|im_end|>", "<|endoftext|>"])
    llm = LLM(model=args.model_id, tensor_parallel_size=1,
              trust_remote_code=True,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, dtype="half")
    outputs = llm.generate(my_prompts, sp)
    out_path = Path(args._shard_out)
    n_total = n_correct = n_kept = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for output, (qi, si) in zip(outputs, my_meta):
            q = questions[qi]
            raw = output.outputs[0].text.strip()
            code = extract_python_code(raw)
            blocks = split_code_blocks(code)
            correct = check_mbpp(code, q["test_list"],
                                 q.get("test_imports"),
                                 timeout=args.exec_timeout)
            n_total += 1
            n_correct += int(correct)
            if len(blocks) < args.min_blocks:
                continue
            row = {"doc_id": q["doc_id"],
                   "prompt_text": q["prompt_text"],
                   "test_list": q["test_list"],
                   "test_imports": q.get("test_imports", []),
                   "sample_idx": si, "response": raw,
                   "code": code, "blocks": blocks,
                   "n_blocks": len(blocks),
                   "exact_match": 1.0 if correct else 0.0}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_kept += 1
    acc = n_correct / max(n_total, 1)
    print(f"[Shard {args._shard_id}] Done. total={n_total}, "
          f"correct={n_correct} ({acc:.3f}), kept={n_kept}")


# -- coordinator --

def main():
    args = parse_args()
    if args._shard_id >= 0:
        run_shard(args)
        return
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    n_shards = len(gpu_ids)
    print("Loading MBPP sanitized dataset...")
    questions = load_mbpp(args.limit)
    print(f"MBPP: {len(questions)} problems, "
          f"samples={args.n_samples}, GPUs={n_shards}")
    shard_dir = out_path.parent / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    q_file = shard_dir / "questions.json"
    q_file.write_text(json.dumps(questions, ensure_ascii=False), "utf-8")
    shard_files = [shard_dir / f"shard_{i}.jsonl" for i in range(n_shards)]
    script = str(Path(__file__).resolve())
    procs = []
    for i, gid in enumerate(gpu_ids):
        cmd = [sys.executable, script,
               "--model-id", args.model_id,
               "--n-samples", str(args.n_samples),
               "--temperature", str(args.temperature),
               "--top-p", str(args.top_p),
               "--max-tokens", str(args.max_tokens),
               "--gpu-memory-utilization", str(args.gpu_memory_utilization),
               "--max-model-len", str(args.max_model_len),
               "--min-blocks", str(args.min_blocks),
               "--exec-timeout", str(args.exec_timeout),
               "--_shard-id", str(i), "--_n-shards", str(n_shards),
               "--_shard-out", str(shard_files[i]),
               "--_gpu-id", gid, "--_questions-file", str(q_file)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gid
        env["TOKENIZERS_PARALLELISM"] = "false"
        lf = shard_dir / f"log_{i}.txt"
        lfh = lf.open("w")
        print(f"  Launching shard {i} on GPU {gid}...")
        p = subprocess.Popen(cmd, env=env, stdout=lfh, stderr=subprocess.STDOUT)
        procs.append((i, gid, p, lfh, lf))

    remaining = list(range(len(procs)))
    while remaining:
        still = []
        for idx in remaining:
            i, gid, p, lfh, lf = procs[idx]
            if p.poll() is None:
                still.append(idx)
            else:
                lfh.close()
                lines = lf.read_text("utf-8", errors="replace").strip().splitlines()
                tail = "\n".join(lines[-5:]) if lines else ""
                print(f"\n--- Shard {i} (GPU {gid}) exit={p.returncode} ---\n{tail}")
        remaining = still
        if remaining:
            time.sleep(10)

    failed = [i for i, _, p, _, _ in procs if p.returncode != 0]
    if failed:
        for i, gid, p, _, lf in procs:
            if p.returncode != 0:
                lines = lf.read_text("utf-8", errors="replace").strip().splitlines()
                print(f"\n--- Shard {i} FAILED ---\n" + "\n".join(lines[-20:]))
        sys.exit(1)

    n_total = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for sf in shard_files:
            if sf.exists():
                for line in sf.read_text("utf-8").splitlines():
                    if line.strip():
                        fout.write(line + "\n")
                        n_total += 1
    print(f"\nMerged: {n_total} solutions -> {out_path}")

    for sf in shard_files:
        sf.unlink(missing_ok=True)
    q_file.unlink(missing_ok=True)
    for i in range(n_shards):
        (shard_dir / f"log_{i}.txt").unlink(missing_ok=True)
    shard_dir.rmdir()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
