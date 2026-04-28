#!/usr/bin/env python3
"""
Control experiment: Correct-prefix continuation.

For each wrong trajectory in the experiment, find a correct trajectory from the
SAME question, truncate it to the same number of steps (tau), then continue
N times. This controls for question difficulty.

R_good(t) = P(correct | correct prefix of length tau, same question)

The key comparison:
  - R_bad vs R_good: how much does a bad prefix hurt vs a good prefix?
  - Controls for the possibility that "hard questions" drive the low R_bad.

Multi-GPU data-parallel.

Usage:
    python scripts/6_5_correct_prefix_control.py \
        --n-continuations 32 --gpus 0,1,2,3,4,5,6,7
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Correct-prefix control experiment")
    ap.add_argument("--early-file", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/first_error/bucket_early.json"))
    ap.add_argument("--late-file", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/first_error/bucket_late.json"))
    ap.add_argument("--cot-file", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/raw_cot_n8.jsonl"))
    ap.add_argument("--out-dir", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/correct_prefix_control"))
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--n-continuations", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    # Internal shard worker args
    ap.add_argument("--_shard-id", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--_n-shards", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--_shard-out", default="", help=argparse.SUPPRESS)
    ap.add_argument("--_gpu-id", default="0", help=argparse.SUPPRESS)
    ap.add_argument("--_task-file", default="", help=argparse.SUPPRESS)
    return ap.parse_args()


def build_chat_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{question.strip()}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


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


def load_samples_with_control(
    early_file: str, late_file: str, cot_file: str, limit: int, seed: int
) -> List[Dict[str, Any]]:
    """Load wrong samples and match each with a correct-prefix control."""
    import random
    rng = random.Random(seed)

    # Load correct trajectories by doc_id
    correct_by_doc = defaultdict(list)
    cot_path = Path(cot_file)
    for line in cot_path.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("exact_match", 0) >= 1.0:
            correct_by_doc[r["doc_id"]].append(r)

    # Load wrong samples
    samples = []
    for path_str in [early_file, late_file]:
        path = Path(path_str)
        if not path.exists():
            continue
        data = json.loads(path.read_text("utf-8"))
        samples.extend(data)

    if limit > 0:
        early = [s for s in samples if s["bucket"] == "early"][:limit]
        late = [s for s in samples if s["bucket"] == "late"][:limit]
        samples = early + late

    # Match with correct prefix
    matched = []
    skipped = 0
    for s in samples:
        candidates = correct_by_doc.get(s["doc_id"], [])
        # Find a correct trajectory with enough steps (>= tau)
        tau = s["tau"]
        eligible = [c for c in candidates if c["n_steps"] >= tau]
        if not eligible:
            skipped += 1
            continue
        # Pick one randomly
        donor = rng.choice(eligible)
        correct_prefix = "\n\n".join(donor["steps"][:tau])
        matched.append({
            "doc_id": s["doc_id"],
            "sample_idx": s["sample_idx"],
            "bucket": s["bucket"],
            "tau": tau,
            "n_steps": s["n_steps"],
            "gold_answer": s["gold_answer"],
            "question": s["question"],
            "correct_prefix": correct_prefix,
            "donor_sample_idx": donor["sample_idx"],
        })

    print(f"Matched: {len(matched)}, skipped (no eligible donor): {skipped}")
    return matched


# ---------------------------------------------------------------------------
# Shard worker
# ---------------------------------------------------------------------------

def run_shard(args) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu_id
    from vllm import LLM, SamplingParams

    task_file = Path(args._task_file)
    tasks = json.loads(task_file.read_text("utf-8"))
    print(f"[Shard {args._shard_id}] GPU {args._gpu_id}: {len(tasks)} prompts")

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

    prompts = [t["prompt"] for t in tasks]
    outputs = llm.generate(prompts, sampling_params)

    out_path = Path(args._shard_out)
    with out_path.open("w", encoding="utf-8") as fout:
        for task, output in zip(tasks, outputs):
            tail = output.outputs[0].text.strip()
            full_response = task["prefix_text"] + ("\n\n" + tail if tail else "")
            pred_answer = extract_boxed_answer(full_response)
            gold = task["gold_answer"]
            is_correct = float(
                normalize_answer(pred_answer) == normalize_answer(gold)
            ) if gold else 0.0

            rec = {
                "doc_id": task["doc_id"],
                "sample_idx": task["sample_idx"],
                "continuation_idx": task["continuation_idx"],
                "bucket": task["bucket"],
                "tau": task["tau"],
                "n_steps": task["n_steps"],
                "gold_answer": gold,
                "pred_answer": pred_answer,
                "exact_match": is_correct,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[Shard {args._shard_id}] Done -> {out_path}")


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args._shard_id >= 0:
        run_shard(args)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    n_shards = len(gpu_ids)

    samples = load_samples_with_control(
        args.early_file, args.late_file, args.cot_file, args.limit, args.seed
    )
    print(f"Samples: {len(samples)} (early={sum(1 for s in samples if s['bucket']=='early')}, "
          f"late={sum(1 for s in samples if s['bucket']=='late')})")
    print(f"Continuations per sample: {args.n_continuations}")

    all_tasks = []
    for s in samples:
        prefix_text = s["correct_prefix"]
        prompt = build_chat_prompt(s["question"]) + prefix_text + "\n\n"
        for ci in range(args.n_continuations):
            all_tasks.append({
                "doc_id": s["doc_id"],
                "sample_idx": s["sample_idx"],
                "continuation_idx": ci,
                "bucket": s["bucket"],
                "tau": s["tau"],
                "n_steps": s["n_steps"],
                "gold_answer": s["gold_answer"],
                "prefix_text": prefix_text,
                "prompt": prompt,
            })

    total_prompts = len(all_tasks)
    print(f"Total prompts: {total_prompts} across {n_shards} GPUs")

    shard_dir = out_dir / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_tasks = [[] for _ in range(n_shards)]
    for i, task in enumerate(all_tasks):
        shard_tasks[i % n_shards].append(task)

    script_path = str(Path(__file__).resolve())
    procs = []
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
            "--_n-shards", str(n_shards),
            "--_shard-out", str(shard_out),
            "--_gpu-id", gpu_id,
            "--_task-file", str(task_file),
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["TOKENIZERS_PARALLELISM"] = "false"

        print(f"  Launching shard {si} on GPU {gpu_id} ({len(shard_tasks[si])} prompts)...")
<<<<<<< HEAD
        log_file = shard_dir / f"log_{si}.txt"
        log_fh = log_file.open("w", encoding="utf-8")
        p = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        procs.append((si, gpu_id, p, log_fh))

    failed = []
    for si, gpu_id, p, log_fh in procs:
        p.wait()
        log_fh.close()
        rc = p.returncode
        log_path = shard_dir / f"log_{si}.txt"
        output_text = log_path.read_text("utf-8", errors="replace")
=======
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        procs.append((si, gpu_id, p))

    failed = []
    for si, gpu_id, p in procs:
        stdout, _ = p.communicate()
        output_text = stdout.decode("utf-8", errors="replace") if stdout else ""
        rc = p.returncode
>>>>>>> 157b73200cc7137e5adbbe2a049fe49b4c83142e
        lines = output_text.strip().splitlines()
        tail = "\n".join(lines[-5:]) if lines else "(no output)"
        print(f"\n--- Shard {si} (GPU {gpu_id}) exit={rc} ---\n{tail}")
        if rc != 0:
            failed.append(si)
            if len(lines) > 5:
                print("...\n" + "\n".join(lines[-20:]))
<<<<<<< HEAD
        else:
            log_path.unlink(missing_ok=True)
=======
>>>>>>> 157b73200cc7137e5adbbe2a049fe49b4c83142e

    if failed:
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
        tf = shard_dir / f"tasks_{si}.json"
        tf.unlink(missing_ok=True)
    shard_dir.rmdir()

    # Quick summary
    results = [json.loads(l) for l in merged_path.read_text("utf-8").splitlines() if l.strip()]
    by_bucket = defaultdict(list)
    for r in results:
        by_bucket[r["bucket"]].append(r["exact_match"])

    print("\n=== Correct-prefix recovery rates ===")
    for bucket in ["early", "late"]:
        vals = by_bucket[bucket]
        if vals:
            rate = sum(vals) / len(vals)
            print(f"  R_good({bucket}): {rate:.4f} ({sum(int(v>=1) for v in vals)}/{len(vals)})")

    print("\nDone.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
