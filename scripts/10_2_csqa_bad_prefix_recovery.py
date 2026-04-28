#!/usr/bin/env python3
"""
Bad-prefix natural recovery on CommonsenseQA.

For each wrong trajectory with a known first-error step (tau), keep the prefix
up to and including step tau, then let the base model continue N times.

Multi-GPU data-parallel via subprocess sharding.

Usage:
    python scripts/10_2_csqa_bad_prefix_recovery.py \
        --n-continuations 32 --gpus 0,1,2,3,4,5,6,7
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.csqa_answer_equiv import extract_choice_letter, is_choice_correct

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
SYSTEM_PROMPT = (
    "Think step by step, then give your final answer as a single letter "
    "(A, B, C, D, or E)."
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bad-prefix natural recovery (CSQA)")
    ap.add_argument("--early-file", default=str(
        PROJECT_ROOT / "results/csqa_3b_multi_sample/first_error/bucket_early.json"))
    ap.add_argument("--late-file", default=str(
        PROJECT_ROOT / "results/csqa_3b_multi_sample/first_error/bucket_late.json"))
    ap.add_argument("--out-dir", default=str(
        PROJECT_ROOT / "results/csqa_3b_multi_sample/bad_prefix_recovery"))
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--n-continuations", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--_shard-id", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--_n-shards", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--_shard-out", default="", help=argparse.SUPPRESS)
    ap.add_argument("--_gpu-id", default="0", help=argparse.SUPPRESS)
    ap.add_argument("--_task-file", default="", help=argparse.SUPPRESS)
    return ap.parse_args()


def split_steps(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    s = [x.strip() for x in text.split("\n\n") if x.strip()]
    return s if s else [text]


def build_chat_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{question.strip()}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_samples(early_file: str, late_file: str, limit: int) -> List[Dict[str, Any]]:
    samples = []
    for path_str, bucket in [(early_file, "early"), (late_file, "late")]:
        path = Path(path_str)
        if not path.exists():
            print(f"WARNING: {path} not found, skipping")
            continue
        data = json.loads(path.read_text("utf-8"))
        for r in data:
            r.setdefault("bucket", bucket)
            samples.append(r)
    if limit > 0:
        early = [s for s in samples if s["bucket"] == "early"][:limit]
        late = [s for s in samples if s["bucket"] == "late"][:limit]
        samples = early + late
    return samples


# ── Shard worker ──────────────────────────────────────────────────────────

def run_shard(args) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu_id
    from vllm import LLM, SamplingParams

    tasks = json.loads(Path(args._task_file).read_text("utf-8"))
    print(f"[Shard {args._shard_id}] GPU {args._gpu_id}: {len(tasks)} prompts")

    sp = SamplingParams(
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

    prompts = [t["prompt"] for t in tasks]
    outputs = llm.generate(prompts, sp)

    out_path = Path(args._shard_out)
    with out_path.open("w", encoding="utf-8") as fout:
        for task, output in zip(tasks, outputs):
            tail = output.outputs[0].text.strip()
            full_response = task["prefix_text"] + ("\n\n" + tail if tail else "")
            pred = extract_choice_letter(full_response)
            gold = task["gold_answer"]
            correct = float(is_choice_correct(pred, gold))

            rec = {
                "doc_id": task["doc_id"],
                "sample_idx": task["sample_idx"],
                "continuation_idx": task["continuation_idx"],
                "bucket": task["bucket"],
                "tau": task["tau"],
                "n_blocks": task["n_blocks"],
                "gold_answer": gold,
                "pred_answer": pred,
                "exact_match": correct,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[Shard {args._shard_id}] Done -> {out_path}")


# ── Coordinator ───────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args._shard_id >= 0:
        run_shard(args)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = out_dir / "continuations.jsonl"
    shard_dir = out_dir / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    n_shards = len(gpu_ids)

    samples = load_samples(args.early_file, args.late_file, args.limit)
    print(f"Samples: {len(samples)} "
          f"(early={sum(1 for s in samples if s['bucket']=='early')}, "
          f"late={sum(1 for s in samples if s['bucket']=='late')})")
    print(f"Continuations per sample: {args.n_continuations}")

    all_tasks = []
    for s in samples:
        steps = split_steps(s["response"])
        tau = s["tau"]
        prefix_text = "\n\n".join(steps[:tau])
        prompt = build_chat_prompt(s["question"]) + prefix_text + "\n\n"
        for ci in range(args.n_continuations):
            all_tasks.append({
                "doc_id": s["doc_id"],
                "sample_idx": s["sample_idx"],
                "continuation_idx": ci,
                "bucket": s["bucket"],
                "tau": tau,
                "n_blocks": len(steps),
                "gold_answer": s["gold_answer"],
                "prefix_text": prefix_text,
                "prompt": prompt,
            })

    total_prompts = len(all_tasks)
    if total_prompts == 0:
        print("No tasks to run.")
        return
    print(f"Total prompts: {total_prompts} across {n_shards} GPUs")

    shard_tasks = [[] for _ in range(n_shards)]
    for i, task in enumerate(all_tasks):
        shard_tasks[i % n_shards].append(task)

    script_path = str(Path(__file__).resolve())
    procs = []
    shard_out_files: List[Path] = []

    for si, gpu_id in enumerate(gpu_ids):
        if not shard_tasks[si]:
            continue
        task_file = shard_dir / f"tasks_{si}.json"
        task_file.write_text(
            json.dumps(shard_tasks[si], ensure_ascii=False), encoding="utf-8"
        )
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
        print(f"  Launching shard {si} on GPU {gpu_id} ({len(shard_tasks[si])} prompts)...")
        p = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        procs.append((si, gpu_id, p, log_fh, log_file))

    # Wait for ALL shards in parallel
    remaining = list(range(len(procs)))
    while remaining:
        still_running = []
        for idx in remaining:
            si, gpu_id, p, log_fh, log_file = procs[idx]
            rc = p.poll()
            if rc is None:
                still_running.append(idx)
            else:
                log_fh.close()
                lines = log_file.read_text("utf-8", errors="replace").strip().splitlines()
                tail = "\n".join(lines[-5:]) if lines else "(no output)"
                print(f"\n--- Shard {si} (GPU {gpu_id}) exit={rc} ---\n{tail}")
        remaining = still_running
        if remaining:
            time.sleep(10)

    failed = []
    for si, gpu_id, p, log_fh, log_file in procs:
        if p.returncode != 0:
            failed.append(si)
            lines = log_file.read_text("utf-8", errors="replace").strip().splitlines()
            print(f"\n--- Shard {si} FAILED (exit={p.returncode}) ---")
            print("\n".join(lines[-20:]))

    if failed:
        print(f"\nERROR: Shards {failed} failed!")
        sys.exit(1)

    # Merge
    results = []
    with merged_path.open("w", encoding="utf-8") as fout:
        for sf in shard_out_files:
            if sf.exists():
                for line in sf.read_text("utf-8").splitlines():
                    if line.strip():
                        fout.write(line + "\n")
                        results.append(json.loads(line))

    print(f"\nMerged: {len(results)} continuations -> {merged_path}")

    # Cleanup
    for sf in shard_out_files:
        sf.unlink(missing_ok=True)
    for si in range(n_shards):
        (shard_dir / f"tasks_{si}.json").unlink(missing_ok=True)
        (shard_dir / f"log_{si}.txt").unlink(missing_ok=True)
    try:
        shard_dir.rmdir()
    except OSError:
        pass

    by_bucket: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        by_bucket[r["bucket"]].append(r["exact_match"])

    print("\n=== Recovery rates ===")
    for bucket in ["early", "late"]:
        vals = by_bucket[bucket]
        if vals:
            rate = sum(vals) / len(vals)
            print(f"  R_bad({bucket}): {rate:.4f} ({sum(int(v>=1) for v in vals)}/{len(vals)})")

    print("\nDone.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
