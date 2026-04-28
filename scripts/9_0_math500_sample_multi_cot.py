#!/usr/bin/env python3
"""
Sample N CoT trajectories per MATH-500 question using vLLM.

Multi-GPU data-parallel: each GPU loads a full model (tp=1) and processes
1/N_GPU of the prompts. Shard outputs are merged at the end.

Usage:
    python scripts/9_0_math500_sample_multi_cot.py \
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

from src.math_answer_equiv import extract_boxed_answer, is_math_equiv
from src.prm.scoring import split_steps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATH500_JSONL = str(
    PROJECT_ROOT / "lm-evaluation-harness/math_eval_data/MATH-500/test.jsonl"
)
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MATH-500 multi-sample CoT (vLLM)")
    ap.add_argument("--math500-jsonl", default=MATH500_JSONL)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--out-file", default=str(
        PROJECT_ROOT / "results/math500_3b_multi_sample/raw_cot_n8.jsonl"))
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--limit", type=int, default=0, help="Process first K questions; 0=all")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--min-steps", type=int, default=4)
    ap.add_argument("--_shard-id", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--_n-shards", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--_shard-out", default="", help=argparse.SUPPRESS)
    ap.add_argument("--_gpu-id", default="0", help=argparse.SUPPRESS)
    return ap.parse_args()


def build_chat_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{question.strip()}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_math500(jsonl_path: Path, limit: int = 0):
    questions = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            questions.append({
                "doc_id": d["unique_id"],
                "question": d["problem"],
                "gold_answer": d["answer"],
            })
    if limit > 0:
        questions = questions[:limit]
    return questions


# ── Shard worker ─────────────────────────────────────────────────────────

def run_shard(args) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu_id
    from vllm import LLM, SamplingParams

    questions = load_math500(Path(args.math500_jsonl), args.limit)

    all_prompts, all_meta = [], []
    for qi, q in enumerate(questions):
        prompt = build_chat_prompt(q["question"])
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
            response = output.outputs[0].text.strip()
            steps = split_steps(response, mode="double_newline")
            pred_answer = extract_boxed_answer(response)
            gold = q["gold_answer"]
            correct = float(is_math_equiv(pred_answer, gold))

            n_total += 1
            n_correct += int(correct >= 1.0)

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
                "exact_match": correct,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_kept += 1

    acc = n_correct / max(n_total, 1)
    print(f"[Shard {args._shard_id}] Done. total={n_total}, correct={n_correct} "
          f"({acc:.3f}), kept={n_kept} -> {out_path}")


# ── Coordinator ──────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args._shard_id >= 0:
        run_shard(args)
        return

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    n_shards = len(gpu_ids)

    questions = load_math500(Path(args.math500_jsonl), args.limit)
    total_prompts = len(questions) * args.n_samples
    print(f"MATH-500 questions: {len(questions)}, samples/q: {args.n_samples}, "
          f"total prompts: {total_prompts}, GPUs: {n_shards}")

    shard_dir = out_path.parent / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_files = [shard_dir / f"shard_{i}.jsonl" for i in range(n_shards)]

    script_path = str(Path(__file__).resolve())
    procs = []
    for i, gpu_id in enumerate(gpu_ids):
        cmd = [
            sys.executable, script_path,
            "--math500-jsonl", args.math500_jsonl,
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

        log_file = shard_dir / f"log_{i}.txt"
        log_fh = log_file.open("w")
        print(f"  Launching shard {i} on GPU {gpu_id}...")
        p = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        procs.append((i, gpu_id, p, log_fh, log_file))

    # Wait for ALL shards in parallel (poll loop)
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
    print(f"\nMerged: {n_total} trajectories (correct={n_correct}, "
          f"wrong={n_total - n_correct})")
    print(f"Output: {out_path}")

    for sf in shard_files:
        sf.unlink(missing_ok=True)
    shard_dir.rmdir()
    print("Shard files cleaned up.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
