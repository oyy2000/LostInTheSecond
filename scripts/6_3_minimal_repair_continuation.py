#!/usr/bin/env python3
"""
Experiment: Minimal repair gain.

For each wrong trajectory, replace ONLY the first-error step (tau) with the
GPT-provided correction, keep steps 1..tau-1 intact, then let the base model
continue N times. Compare R_fix(t) against R_bad(t) from experiment 6_1.

Multi-GPU data-parallel via subprocess sharding.

Usage:
    python scripts/6_3_minimal_repair_continuation.py \
        --early-file results/gsm8k_3b_multi_sample/first_error/bucket_early.json \
        --late-file results/gsm8k_3b_multi_sample/first_error/bucket_late.json \
        --out-dir results/gsm8k_3b_multi_sample/minimal_repair \
        --n-continuations 32 \
        --gpus 0,1,2,3,4,5,6,7
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Minimal repair continuation experiment")
    ap.add_argument("--early-file", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/first_error/bucket_early.json"))
    ap.add_argument("--late-file", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/first_error/bucket_late.json"))
    ap.add_argument("--out-dir", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/minimal_repair"))
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--n-continuations", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
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


def build_fixed_prefix(steps: List[str], tau: int, correction: str) -> str:
    """Build prefix with the error step replaced by correction.

    steps[0:tau-1] are kept intact (correct steps before the error).
    The error step (steps[tau-1]) is replaced by `correction`.
    """
    prefix_parts = list(steps[: tau - 1]) + [correction]
    return "\n\n".join(prefix_parts)


def load_samples(early_file: str, late_file: str, limit: int) -> List[Dict[str, Any]]:
    samples = []
    for path_str in [early_file, late_file]:
        path = Path(path_str)
        if not path.exists():
            print(f"WARNING: {path} not found, skipping")
            continue
        data = json.loads(path.read_text("utf-8"))
        for r in data:
            if not (r.get("gpt_parsed") and r["gpt_parsed"].get("correction")):
                continue
            samples.append(r)
    if limit > 0:
        early = [s for s in samples if s["bucket"] == "early"][:limit]
        late = [s for s in samples if s["bucket"] == "late"][:limit]
        samples = early + late
    return samples


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
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _task_key(doc_id: int, sample_idx: int, continuation_idx: int) -> str:
    return f"{doc_id}:{sample_idx}:{continuation_idx}"


def load_completed_keys(merged_path: Path, shard_dir: Path) -> Set[str]:
    """Collect task keys already present in merged output or leftover shards."""
    keys: Set[str] = set()
    sources: List[Path] = []
    if merged_path.exists():
        sources.append(merged_path)
    if shard_dir.exists():
        sources.extend(sorted(shard_dir.glob("shard_*.jsonl")))
    for p in sources:
        for line in p.read_text("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                keys.add(_task_key(r["doc_id"], r["sample_idx"], r["continuation_idx"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return keys


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

def _print_summary(merged_path: Path, shard_dir: Path) -> None:
    """Print recovery-rate summary from all available results."""
    all_lines: List[str] = []
    if merged_path.exists():
        all_lines.extend(merged_path.read_text("utf-8").splitlines())
    if shard_dir.exists():
        for sf in sorted(shard_dir.glob("shard_*.jsonl")):
            all_lines.extend(sf.read_text("utf-8").splitlines())

    results = []
    for line in all_lines:
        line = line.strip()
        if not line:
            continue
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    by_bucket: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        by_bucket[r["bucket"]].append(r["exact_match"])

    n_total = len(results)
    print(f"\n=== Fix recovery rates ({n_total} total continuations) ===")
    for bucket in ["early", "late"]:
        vals = by_bucket[bucket]
        if vals:
            rate = sum(vals) / len(vals)
            print(f"  R_fix({bucket}): {rate:.4f} "
                  f"({sum(int(v >= 1) for v in vals)}/{len(vals)})")
    print("\nDone.")


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
    incomplete_path = out_dir / "incomplete_tasks.json"

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    n_shards = len(gpu_ids)

    samples = load_samples(args.early_file, args.late_file, args.limit)
    print(f"Samples: {len(samples)} "
          f"(early={sum(1 for s in samples if s['bucket']=='early')}, "
          f"late={sum(1 for s in samples if s['bucket']=='late')})")
    print(f"Continuations per sample: {args.n_continuations}")

    completed = load_completed_keys(merged_path, shard_dir)
    if completed:
        print(f"Checkpoint: {len(completed)} continuations already done, skipping them")

    all_tasks = []
    for s in samples:
        prefix_text = build_fixed_prefix(
            s["steps"], s["tau"], s["gpt_parsed"]["correction"]
        )
        prompt = build_chat_prompt(s["question"]) + prefix_text + "\n\n"
        for ci in range(args.n_continuations):
            key = _task_key(s["doc_id"], s["sample_idx"], ci)
            if key in completed:
                continue
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
    if total_prompts == 0:
        print("All tasks already completed. Nothing to do.")
        _print_summary(merged_path, shard_dir)
        return
    print(f"Remaining prompts: {total_prompts} across {n_shards} GPUs")

    shard_tasks = [[] for _ in range(n_shards)]
    for i, task in enumerate(all_tasks):
        shard_tasks[i % n_shards].append(task)

    script_path = str(Path(__file__).resolve())
    procs: List[Tuple[int, str, subprocess.Popen]] = []
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

        print(f"  Launching shard {si} on GPU {gpu_id} "
              f"({len(shard_tasks[si])} prompts)...")
        p = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        procs.append((si, gpu_id, p))

    failed_shards: List[int] = []
    succeeded_shards: List[int] = []
    for si, gpu_id, p in procs:
        stdout, _ = p.communicate()
        output_text = (
            stdout.decode("utf-8", errors="replace") if stdout else ""
        )
        rc = p.returncode
        lines = output_text.strip().splitlines()
        tail = "\n".join(lines[-5:]) if lines else "(no output)"
        print(f"\n--- Shard {si} (GPU {gpu_id}) exit={rc} ---\n{tail}")
        if rc != 0:
            failed_shards.append(si)
            if len(lines) > 5:
                print("...\n" + "\n".join(lines[-20:]))
        else:
            succeeded_shards.append(si)

    # Merge only succeeded shard outputs into the main file (append mode)
    n_new = 0
    succeeded_shard_files = {shard_dir / f"shard_{si}.jsonl" for si in succeeded_shards}
    with merged_path.open("a", encoding="utf-8") as fout:
        for sf in shard_out_files:
            if sf not in succeeded_shard_files or not sf.exists():
                continue
            for line in sf.read_text("utf-8").splitlines():
                if line.strip():
                    fout.write(line + "\n")
                    n_new += 1
    print(f"\nAppended {n_new} new continuations -> {merged_path}")

    # Cleanup only succeeded shards; keep failed shard files for debugging
    for si in succeeded_shards:
        sf = shard_dir / f"shard_{si}.jsonl"
        sf.unlink(missing_ok=True)
        tf = shard_dir / f"tasks_{si}.json"
        tf.unlink(missing_ok=True)

    if failed_shards:
        # Record which tasks remain incomplete
        remaining = []
        for si in failed_shards:
            remaining.extend(shard_tasks[si])
        remaining_keys = [
            _task_key(t["doc_id"], t["sample_idx"], t["continuation_idx"])
            for t in remaining
        ]
        incomplete_path.write_text(
            json.dumps(remaining_keys, indent=2), encoding="utf-8"
        )
        print(f"\nWARNING: Shards {failed_shards} failed. "
              f"{len(remaining_keys)} tasks incomplete.")
        print(f"  Incomplete task keys saved to {incomplete_path}")
        print("  Re-run the same command to resume from checkpoint.")
    else:
        # All succeeded -- clean up shard dir entirely
        incomplete_path.unlink(missing_ok=True)
        try:
            shard_dir.rmdir()
        except OSError:
            pass

    _print_summary(merged_path, shard_dir)

    if failed_shards:
        sys.exit(1)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
