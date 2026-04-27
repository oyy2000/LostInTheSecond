#!/usr/bin/env python3
"""
Minimal repair continuation on CommonsenseQA.

For each wrong trajectory, replace ONLY the first-error step (tau) with the
GPT-provided correction, keep steps 1..tau-1 intact, then let the base model
continue N times. Compare R_fix(t) against R_bad(t) from 10_2.

Multi-GPU data-parallel with checkpoint/resume support.

Usage:
    python scripts/10_3_csqa_minimal_repair.py \
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
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.csqa_answer_equiv import extract_choice_letter, is_choice_correct

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
SYSTEM_PROMPT = (
    "Think step by step, then give your final answer as a single letter "
    "(A, B, C, D, or E)."
)


def split_steps(text: str):
    text = (text or "").strip()
    if not text:
        return []
    s = [x.strip() for x in text.split("\n\n") if x.strip()]
    return s if s else [text]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Minimal repair continuation (CSQA)")
    ap.add_argument("--early-file", default=str(
        PROJECT_ROOT / "results/csqa_3b_multi_sample/first_error/bucket_early.json"))
    ap.add_argument("--late-file", default=str(
        PROJECT_ROOT / "results/csqa_3b_multi_sample/first_error/bucket_late.json"))
    ap.add_argument("--out-dir", default=str(
        PROJECT_ROOT / "results/csqa_3b_multi_sample/minimal_repair"))
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--n-continuations", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=2048)
    # Internal shard args
    ap.add_argument("--_shard-id", type=int, default=-1)
    ap.add_argument("--_task-file", type=str, default="")
    ap.add_argument("--_shard-out", type=str, default="")
    return ap.parse_args()


def build_prefix_with_repair(question: str, steps: list, tau: int,
                              correction: str) -> str:
    """Build prompt = system + question + steps[0..tau-2] + correction."""
    prefix_steps = steps[:tau - 1] if tau > 1 else []
    if correction:
        prefix_steps.append(correction.strip())
    body = "\n\n".join(prefix_steps)
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{question}\n<|im_end|>\n"
        f"<|im_start|>assistant\n{body}\n\n"
    )


# ── Shard worker ──────────────────────────────────────────────────────────

def run_shard(args):
    from vllm import LLM, SamplingParams

    tasks = json.loads(Path(args._task_file).read_text("utf-8"))
    if not tasks:
        Path(args._shard_out).write_text("")
        return

    prompts, meta = [], []
    for t in tasks:
        prompt = build_prefix_with_repair(
            t["question"], split_steps(t["response"]),
            t["tau"], t.get("correction", ""))
        for ci in range(args.n_continuations):
            prompts.append(prompt)
            meta.append((t["doc_id"], t["sample_idx"], t["gold_answer"],
                         t["bucket"], t["tau"], t.get("n_steps", 0), ci))

    llm = LLM(
        model=args.model_id,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="half",
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    outputs = llm.generate(prompts, sp)

    with Path(args._shard_out).open("w", encoding="utf-8") as f:
        for idx, output in enumerate(outputs):
            doc_id, sample_idx, gold, bucket, tau, n_steps, ci = meta[idx]
            resp = output.outputs[0].text.strip()
            pred = extract_choice_letter(resp)
            correct = is_choice_correct(pred, gold)
            rec = {
                "doc_id": doc_id,
                "sample_idx": sample_idx,
                "continuation_idx": ci,
                "bucket": bucket,
                "tau": tau,
                "n_steps": n_steps,
                "pred_answer": pred,
                "gold_answer": gold,
                "exact_match": 1.0 if correct else 0.0,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[Shard {args._shard_id}] Done -> {args._shard_out}")


# ── Coordinator ───────────────────────────────────────────────────────────

def _load_buckets(args) -> List[Dict]:
    tasks = []
    for label, path in [("early", args.early_file), ("late", args.late_file)]:
        if not Path(path).exists():
            print(f"  WARNING: {path} not found, skipping {label}")
            continue
        items = json.loads(Path(path).read_text("utf-8"))
        for item in items:
            item["bucket"] = label
        tasks.extend(items)
        print(f"  {label}: {len(items)} trajectories")
    return tasks


def _print_summary(merged_path: Path, shard_dir: Path):
    if not merged_path.exists():
        print("\n=== Fix recovery rates (0 total continuations) ===")
        return
    results = [json.loads(l) for l in merged_path.read_text("utf-8").splitlines() if l.strip()]
    print(f"\n=== Fix recovery rates ({len(results)} total continuations) ===")
    by_bucket: Dict[str, list] = defaultdict(list)
    for r in results:
        by_bucket[r["bucket"]].append(r["exact_match"])
    for bucket in ["early", "late"]:
        vals = by_bucket[bucket]
        if vals:
            rate = sum(vals) / len(vals)
            n_correct = sum(int(v >= 1) for v in vals)
            print(f"  R_fix({bucket}): {rate:.4f} ({n_correct}/{len(vals)})")


def main():
    args = parse_args()

    if args._shard_id >= 0:
        run_shard(args)
        return

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    n_shards = len(gpu_ids)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    merged_path = out_dir / "continuations.jsonl"

    # Load tasks
    all_tasks = _load_buckets(args)
    if not all_tasks:
        print("No tasks to process.")
        return

    # Check for checkpoint/resume
    incomplete_path = shard_dir / "incomplete_keys.json"
    done_keys: Set[Tuple[str, int]] = set()
    if merged_path.exists():
        for line in merged_path.read_text("utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done_keys.add((r["doc_id"], r["sample_idx"]))

    remaining = [t for t in all_tasks
                 if (t["doc_id"], t["sample_idx"]) not in done_keys]
    print(f"  Total tasks: {len(all_tasks)}, already done: {len(done_keys)}, "
          f"remaining: {len(remaining)}")

    if not remaining:
        _print_summary(merged_path, shard_dir)
        print("\nDone.")
        return

    # Shard tasks
    shard_tasks: List[List[Dict]] = [[] for _ in range(n_shards)]
    for i, t in enumerate(remaining):
        shard_tasks[i % n_shards].append(t)

    script_path = str(Path(__file__).resolve())
    procs: list = []
    shard_out_files = []
    for si, gpu_id in enumerate(gpu_ids):
        task_file = shard_dir / f"tasks_{si}.json"
        task_file.write_text(json.dumps(shard_tasks[si], ensure_ascii=False), encoding="utf-8")
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
            "--_shard-out", str(shard_out),
            "--_task-file", str(task_file),
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["TOKENIZERS_PARALLELISM"] = "false"

        log_file = shard_dir / f"log_{si}.txt"
        log_fh = log_file.open("w")
        print(f"  Launching shard {si} on GPU {gpu_id} "
              f"({len(shard_tasks[si])} tasks)...")
        p = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        procs.append((si, gpu_id, p, log_fh, log_file))

    # Wait for ALL shards in parallel
    remaining_idx = list(range(len(procs)))
    while remaining_idx:
        still_running = []
        for idx in remaining_idx:
            si, gpu_id, p, log_fh, log_file = procs[idx]
            rc = p.poll()
            if rc is None:
                still_running.append(idx)
            else:
                log_fh.close()
                lines = log_file.read_text("utf-8", errors="replace").strip().splitlines()
                tail = "\n".join(lines[-5:]) if lines else "(no output)"
                print(f"\n--- Shard {si} (GPU {gpu_id}) exit={rc} ---\n{tail}")
        remaining_idx = still_running
        if remaining_idx:
            time.sleep(10)

    failed_shards: List[int] = []
    succeeded_shards: List[int] = []
    for si, gpu_id, p, log_fh, log_file in procs:
        if p.returncode != 0:
            failed_shards.append(si)
            lines = log_file.read_text("utf-8", errors="replace").strip().splitlines()
            print(f"\n--- Shard {si} FAILED (exit={p.returncode}) ---")
            print("\n".join(lines[-20:]))
        else:
            succeeded_shards.append(si)

    # Merge successful shards
    new_count = 0
    with merged_path.open("a", encoding="utf-8") as fout:
        for si in succeeded_shards:
            sf = shard_out_files[si]
            if sf.exists():
                for line in sf.read_text("utf-8").splitlines():
                    if line.strip():
                        fout.write(line + "\n")
                        new_count += 1
    print(f"\nAppended {new_count} new continuations -> {merged_path}")

    # Track incomplete
    remaining_keys = set()
    for si in failed_shards:
        tf = shard_dir / f"tasks_{si}.json"
        if tf.exists():
            for t in json.loads(tf.read_text("utf-8")):
                remaining_keys.add((t["doc_id"], t["sample_idx"]))

    if remaining_keys:
        incomplete_path.write_text(
            json.dumps(list(remaining_keys)), encoding="utf-8")
        print(f"\nWARNING: Shards {failed_shards} failed. "
              f"{len(remaining_keys)} tasks incomplete.")
        print("  Re-run the same command to resume from checkpoint.")
    else:
        incomplete_path.unlink(missing_ok=True)
        try:
            shard_dir.rmdir()
        except OSError:
            pass

    _print_summary(merged_path, shard_dir)

    if failed_shards:
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
