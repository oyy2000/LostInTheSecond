#!/usr/bin/env python3
"""
Bad-prefix natural recovery on MBPP.

For each wrong solution with known first-error block (tau), keep the prefix
up to and including block tau, then let the model continue N times.

Usage:
    python scripts/13_2_bad_prefix_natural_recovery.py \
        --n-continuations 32 --gpus 0,1,2,3,4,5,6,7
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.code_answer_equiv import check_mbpp, extract_python_code

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Continue writing the Python function below. "
    "Output ONLY the remaining code, no explanation."
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--early-file", default=str(
        PROJECT_ROOT / "results/mbpp_3b_multi_sample/first_error/bucket_early.json"))
    ap.add_argument("--late-file", default=str(
        PROJECT_ROOT / "results/mbpp_3b_multi_sample/first_error/bucket_late.json"))
    ap.add_argument("--out-dir", default=str(
        PROJECT_ROOT / "results/mbpp_3b_multi_sample/bad_prefix_recovery"))
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--n-continuations", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--exec-timeout", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--_shard-id", type=int, default=-1,
                     help=argparse.SUPPRESS)
    ap.add_argument("--_n-shards", type=int, default=-1,
                     help=argparse.SUPPRESS)
    ap.add_argument("--_shard-out", default="", help=argparse.SUPPRESS)
    ap.add_argument("--_gpu-id", default="0", help=argparse.SUPPRESS)
    ap.add_argument("--_task-file", default="", help=argparse.SUPPRESS)
    return ap.parse_args()


def build_chat_prompt(func_prompt: str, prefix_code: str) -> str:
    user_msg = (
        f"Here is the task:\n{func_prompt.strip()}\n\n"
        f"Here is the beginning of an implementation:\n"
        f"```python\n{prefix_code}\n```\n\n"
        f"Continue and complete this function. "
        f"Output ONLY the remaining Python code."
    )
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_prefix_code(blocks: List[str], tau: int) -> str:
    return "\n\n".join(blocks[:tau])


def load_samples(early_file: str, late_file: str,
                 limit: int) -> List[Dict[str, Any]]:
    samples = []
    for path_str in [early_file, late_file]:
        path = Path(path_str)
        if not path.exists():
            print(f"WARNING: {path} not found, skipping")
            continue
        data = json.loads(path.read_text("utf-8"))
        samples.extend(data)
    if limit > 0:
        early = [s for s in samples if s["bucket"] == "early"][:limit]
        late = [s for s in samples if s["bucket"] == "late"][:limit]
        samples = early + late
    return samples


# -- Shard worker --

def run_shard(args) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu_id
    from vllm import LLM, SamplingParams

    tasks = json.loads(Path(args._task_file).read_text("utf-8"))
    print(f"[Shard {args._shard_id}] GPU {args._gpu_id}: {len(tasks)} prompts")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    llm = LLM(
        model=args.model_id, tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, dtype="half",
    )

    prompts = [t["prompt_text"] for t in tasks]
    outputs = llm.generate(prompts, sampling_params)

    out_path = Path(args._shard_out)
    with out_path.open("w", encoding="utf-8") as fout:
        for task, output in zip(tasks, outputs):
            tail = output.outputs[0].text.strip()
            tail_code = extract_python_code(tail) if tail else ""
            full_code = task["prefix_code"] + ("\n\n" + tail_code
                                                if tail_code else "")
            correct = check_mbpp(
                full_code, task["test_list"],
                task.get("test_imports"),
                timeout=args.exec_timeout)

            rec = {
                "doc_id": task["doc_id"],
                "sample_idx": task["sample_idx"],
                "continuation_idx": task["continuation_idx"],
                "bucket": task["bucket"],
                "tau": task["tau"],
                "n_blocks": task["n_blocks"],
                "exact_match": 1.0 if correct else 0.0,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[Shard {args._shard_id}] Done -> {out_path}")


# -- Coordinator --

def main() -> None:
    args = parse_args()

    if args._shard_id >= 0:
        run_shard(args)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    n_shards = len(gpu_ids)

    samples = load_samples(args.early_file, args.late_file, args.limit)
    print(f"Samples: {len(samples)} "
          f"(early={sum(1 for s in samples if s['bucket']=='early')}, "
          f"late={sum(1 for s in samples if s['bucket']=='late')})")

    all_tasks = []
    for s in samples:
        prefix_code = build_prefix_code(s["blocks"], s["tau"])
        prompt_text = build_chat_prompt(s["prompt_text"], prefix_code)
        for ci in range(args.n_continuations):
            all_tasks.append({
                "doc_id": s["doc_id"],
                "sample_idx": s["sample_idx"],
                "continuation_idx": ci,
                "bucket": s["bucket"],
                "tau": s["tau"],
                "n_blocks": s["n_blocks"],
                "test_list": s["test_list"],
                "test_imports": s.get("test_imports", []),
                "prefix_code": prefix_code,
                "prompt_text": prompt_text,
            })

    print(f"Total prompts: {len(all_tasks)} across {n_shards} GPUs")

    shard_dir = out_dir / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_tasks = [[] for _ in range(n_shards)]
    for i, task in enumerate(all_tasks):
        shard_tasks[i % n_shards].append(task)

    script_path = str(Path(__file__).resolve())
    procs, shard_out_files = [], []

    for si, gpu_id in enumerate(gpu_ids):
        task_file = shard_dir / f"tasks_{si}.json"
        task_file.write_text(
            json.dumps(shard_tasks[si], ensure_ascii=False), encoding="utf-8")
        shard_out = shard_dir / f"shard_{si}.jsonl"
        shard_out_files.append(shard_out)

        cmd = [
            sys.executable, script_path,
            "--model-id", args.model_id,
            "--temperature", str(args.temperature),
            "--top-p", str(args.top_p),
            "--max-tokens", str(args.max_tokens),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--max-model-len", str(args.max_model_len),
            "--exec-timeout", str(args.exec_timeout),
            "--_shard-id", str(si),
            "--_n-shards", str(n_shards),
            "--_shard-out", str(shard_out),
            "--_gpu-id", gpu_id,
            "--_task-file", str(task_file),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["TOKENIZERS_PARALLELISM"] = "false"

        log_file = shard_dir / f"log_{si}.txt"
        log_fh = log_file.open("w")
        print(f"  Launching shard {si} on GPU {gpu_id} "
              f"({len(shard_tasks[si])} prompts)...")
        p = subprocess.Popen(cmd, env=env, stdout=log_fh,
                             stderr=subprocess.STDOUT)
        procs.append((si, gpu_id, p, log_fh, log_file))

    import time
    remaining = list(range(len(procs)))
    while remaining:
        still_running = []
        for idx in remaining:
            si, gpu_id, p, log_fh, log_file = procs[idx]
            if p.poll() is None:
                still_running.append(idx)
            else:
                log_fh.close()
                lines = (log_file.read_text("utf-8", errors="replace")
                         .strip().splitlines())
                tail = "\n".join(lines[-5:]) if lines else "(no output)"
                print(f"\n--- Shard {si} (GPU {gpu_id}) exit={p.returncode} "
                      f"---\n{tail}")
        remaining = still_running
        if remaining:
            time.sleep(10)

    failed = [si for si, _, p, _, _ in procs if p.returncode != 0]
    if failed:
        for si, gpu_id, p, _, log_file in procs:
            if p.returncode != 0:
                lines = (log_file.read_text("utf-8", errors="replace")
                         .strip().splitlines())
                print(f"\n--- Shard {si} FAILED ---")
                print("\n".join(lines[-20:]))
        print(f"\nERROR: Shards {failed} failed!")
        sys.exit(1)

    merged_path = out_dir / "continuations.jsonl"
    n_total = 0
    with merged_path.open("w", encoding="utf-8") as fout:
        for sf in shard_out_files:
            if sf.exists():
                for line in sf.read_text("utf-8").splitlines():
                    if line.strip():
                        fout.write(line + "\n")
                        n_total += 1

    print(f"\nMerged: {n_total} continuations -> {merged_path}")

    for sf in shard_out_files:
        sf.unlink(missing_ok=True)
    for si in range(n_shards):
        (shard_dir / f"tasks_{si}.json").unlink(missing_ok=True)
        (shard_dir / f"log_{si}.txt").unlink(missing_ok=True)
    shard_dir.rmdir()

    from collections import defaultdict
    results = [json.loads(l) for l in
               merged_path.read_text("utf-8").splitlines() if l.strip()]
    by_bucket = defaultdict(list)
    for r in results:
        by_bucket[r["bucket"]].append(r["exact_match"])

    print("\n=== Recovery rates ===")
    for bucket in ["early", "late"]:
        vals = by_bucket[bucket]
        if vals:
            rate = sum(vals) / len(vals)
            print(f"  R_bad({bucket}): {rate:.4f} "
                  f"({sum(int(v>=1) for v in vals)}/{len(vals)})")
    print("\nDone.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
