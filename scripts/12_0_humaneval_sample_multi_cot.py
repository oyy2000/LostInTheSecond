#!/usr/bin/env python3
"""
Sample N code solutions per HumanEval problem using vLLM.

Multi-GPU data-parallel: each GPU loads a full model (tp=1) and processes
1/N_GPU of the prompts. Shard outputs are merged at the end.

Usage:
    python scripts/12_0_humaneval_sample_multi_cot.py \
        --n-samples 8 \
        --gpus 0,1,2,3,4,5,6,7
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.code_answer_equiv import (
    check_humaneval, split_code_blocks, extract_python_code,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Complete the function below. "
    "Output ONLY the Python code (the full function), no explanation."
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HumanEval multi-sample (vLLM)")
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--out-file", default=str(
        PROJECT_ROOT / "results/humaneval_3b_multi_sample/raw_code_n8.jsonl"))
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--min-blocks", type=int, default=3,
                     help="Min logical blocks to keep a solution")
    ap.add_argument("--exec-timeout", type=float, default=5.0)
    ap.add_argument("--_shard-id", type=int, default=-1,
                     help=argparse.SUPPRESS)
    ap.add_argument("--_n-shards", type=int, default=-1,
                     help=argparse.SUPPRESS)
    ap.add_argument("--_shard-out", default="", help=argparse.SUPPRESS)
    ap.add_argument("--_gpu-id", default="0", help=argparse.SUPPRESS)
    ap.add_argument("--_questions-file", default="",
                     help=argparse.SUPPRESS)
    return ap.parse_args()


def build_chat_prompt(func_prompt: str) -> str:
    user_msg = (
        f"Complete the following Python function:\n\n"
        f"```python\n{func_prompt.strip()}\n```\n\n"
        f"Output the COMPLETE function (including the signature and docstring)."
    )
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_humaneval(limit: int = 0):
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test",
                      trust_remote_code=True)
    questions = []
    for i, row in enumerate(ds):
        if limit > 0 and i >= limit:
            break
        questions.append({
            "doc_id": row["task_id"],
            "prompt": row["prompt"],
            "canonical_solution": row["canonical_solution"],
            "test": row["test"],
            "entry_point": row["entry_point"],
        })
    return questions


# -- Shard worker --

def run_shard(args) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu_id
    from vllm import LLM, SamplingParams

    questions = json.loads(Path(args._questions_file).read_text("utf-8"))

    all_prompts, all_meta = [], []
    for qi, q in enumerate(questions):
        prompt = build_chat_prompt(q["prompt"])
        for si in range(args.n_samples):
            all_prompts.append(prompt)
            all_meta.append((qi, si))

    my_indices = list(range(args._shard_id, len(all_prompts), args._n_shards))
    my_prompts = [all_prompts[i] for i in my_indices]
    my_meta = [all_meta[i] for i in my_indices]

    print(f"[Shard {args._shard_id}] GPU {args._gpu_id}: "
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

    out_path = Path(args._shard_out)
    n_total, n_correct, n_kept = 0, 0, 0
    with out_path.open("w", encoding="utf-8") as fout:
        for output, (qi, si) in zip(outputs, my_meta):
            q = questions[qi]
            raw_response = output.outputs[0].text.strip()
            code = extract_python_code(raw_response)
            blocks = split_code_blocks(code)

            correct = check_humaneval(
                code, q["test"], q["entry_point"],
                timeout=args.exec_timeout,
            )

            n_total += 1
            n_correct += int(correct)

            if len(blocks) < args.min_blocks:
                continue

            row = {
                "doc_id": q["doc_id"],
                "prompt": q["prompt"],
                "test": q["test"],
                "entry_point": q["entry_point"],
                "sample_idx": si,
                "response": raw_response,
                "code": code,
                "blocks": blocks,
                "n_blocks": len(blocks),
                "exact_match": 1.0 if correct else 0.0,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_kept += 1

    acc = n_correct / max(n_total, 1)
    print(f"[Shard {args._shard_id}] Done. total={n_total}, correct={n_correct} "
          f"({acc:.3f}), kept={n_kept} -> {out_path}")


# -- Coordinator --

def main() -> None:
    args = parse_args()

    if args._shard_id >= 0:
        run_shard(args)
        return

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    n_shards = len(gpu_ids)

    print("Loading HumanEval dataset...")
    questions = load_humaneval(args.limit)
    total_prompts = len(questions) * args.n_samples
    print(f"HumanEval problems: {len(questions)}, samples/q: {args.n_samples}, "
          f"total prompts: {total_prompts}, GPUs: {n_shards}")

    shard_dir = out_path.parent / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    q_file = shard_dir / "questions.json"
    q_file.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")

    shard_files = [shard_dir / f"shard_{i}.jsonl" for i in range(n_shards)]
    script_path = str(Path(__file__).resolve())
    procs = []
    for i, gpu_id in enumerate(gpu_ids):
        cmd = [
            sys.executable, script_path,
            "--model-id", args.model_id,
            "--n-samples", str(args.n_samples),
            "--temperature", str(args.temperature),
            "--top-p", str(args.top_p),
            "--max-tokens", str(args.max_tokens),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--max-model-len", str(args.max_model_len),
            "--min-blocks", str(args.min_blocks),
            "--exec-timeout", str(args.exec_timeout),
            "--_shard-id", str(i),
            "--_n-shards", str(n_shards),
            "--_shard-out", str(shard_files[i]),
            "--_gpu-id", gpu_id,
            "--_questions-file", str(q_file),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["TOKENIZERS_PARALLELISM"] = "false"

        log_file = shard_dir / f"log_{i}.txt"
        log_fh = log_file.open("w")
        print(f"  Launching shard {i} on GPU {gpu_id}...")
        p = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        procs.append((i, gpu_id, p, log_fh, log_file))

    import time
    remaining = list(range(len(procs)))
    while remaining:
        still_running = []
        for idx in remaining:
            i, gpu_id, p, log_fh, log_file = procs[idx]
            rc = p.poll()
            if rc is None:
                still_running.append(idx)
            else:
                log_fh.close()
                lines = log_file.read_text("utf-8", errors="replace").strip().splitlines()
                tail = "\n".join(lines[-5:]) if lines else "(no output)"
                print(f"\n--- Shard {i} (GPU {gpu_id}) exit={rc} ---\n{tail}")
        remaining = still_running
        if remaining:
            time.sleep(10)

    failed = []
    for i, gpu_id, p, log_fh, log_file in procs:
        if p.returncode != 0:
            failed.append(i)
            lines = log_file.read_text("utf-8", errors="replace").strip().splitlines()
            print(f"\n--- Shard {i} FAILED ---")
            print("\n".join(lines[-20:]))

    if failed:
        print(f"\nERROR: Shards {failed} failed!")
        sys.exit(1)

    n_total = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for sf in shard_files:
            if sf.exists():
                for line in sf.read_text("utf-8").splitlines():
                    if line.strip():
                        fout.write(line + "\n")
                        n_total += 1

    n_correct = sum(
        1 for line in out_path.read_text("utf-8").splitlines()
        if line.strip() and json.loads(line).get("exact_match", 0.0) >= 1.0
    )
    print(f"\nMerged: {n_total} solutions (correct={n_correct}, "
          f"wrong={n_total - n_correct})")
    print(f"Output: {out_path}")

    for sf in shard_files:
        sf.unlink(missing_ok=True)
    q_file.unlink(missing_ok=True)
    for i in range(n_shards):
        lf = shard_dir / f"log_{i}.txt"
        lf.unlink(missing_ok=True)
    shard_dir.rmdir()
    print("Shard files cleaned up.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
