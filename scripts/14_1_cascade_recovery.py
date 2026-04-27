#!/usr/bin/env python3
"""
Phase 14_1: Cascade error recovery experiment.

For trajectories with cascade errors (error B cascades from error A):
  Condition 1 -- "fix_first": fix error A (the root cause), sample from step A onward
  Condition 2 -- "fix_later": fix error B (the cascade), sample from step B onward

Each condition generates N continuations and measures recovery rate.

Multi-GPU data-parallel via subprocess sharding.

Usage:
    python scripts/14_1_cascade_recovery.py \
        --cascade-file results/gsm8k_3b_multi_sample/cascade_errors/cascade_samples.json \
        --out-dir results/gsm8k_3b_multi_sample/cascade_recovery \
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
SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Cascade error recovery experiment")
    ap.add_argument("--cascade-file", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/cascade_errors/cascade_samples.json"))
    ap.add_argument("--out-dir", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/cascade_recovery"))
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


def build_fix_first_prefix(steps: List[str], first_err_step: int,
                           first_correction: str) -> str:
    """Fix the root-cause error A, keep steps before A intact.

    Prefix = steps[0..A-2] + [correction_A]
    Model continues from after the corrected step A.
    """
    prefix_parts = list(steps[: first_err_step - 1]) + [first_correction]
    return "\n\n".join(prefix_parts)


def build_fix_later_prefix(steps: List[str], cascade_step: int,
                           cascade_correction: str) -> str:
    """Fix only the cascade error B, keep everything before B intact (including
    the original wrong step A).

    Prefix = steps[0..B-2] + [correction_B]
    Model continues from after the corrected step B.
    """
    prefix_parts = list(steps[: cascade_step - 1]) + [cascade_correction]
    return "\n\n".join(prefix_parts)


def build_fix_first_keep_middle_prefix(
    steps: List[str], first_err_step: int, first_correction: str,
    cascade_step: int,
) -> str:
    """Fix root-cause error A but keep all original steps from A+1 through B
    (including the cascade error B itself), then let the model continue from
    after step B.

    Prefix = steps[0..A-2] + [correction_A] + steps[A..B-1]
    """
    before_a = list(steps[: first_err_step - 1])
    middle_and_b = list(steps[first_err_step : cascade_step])
    prefix_parts = before_a + [first_correction] + middle_and_b
    return "\n\n".join(prefix_parts)


def load_cascade_samples(cascade_file: str, limit: int):
    path = Path(cascade_file)
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)
    data = json.loads(path.read_text("utf-8"))
    if limit > 0:
        data = data[:limit]
    return data


def _task_key(doc_id, sample_idx, pair_idx, condition, cont_idx):
    return f"{doc_id}:{sample_idx}:{pair_idx}:{condition}:{cont_idx}"


def load_completed_keys(merged_path: Path, shard_dir: Path) -> Set[str]:
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
                keys.add(_task_key(
                    r["doc_id"], r["sample_idx"], r["pair_idx"],
                    r["condition"], r["continuation_idx"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return keys


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
            full = task["prefix_text"] + ("\n\n" + tail if tail else "")
            pred_answer = extract_boxed_answer(full)
            gold = task["gold_answer"]
            em = float(normalize_answer(pred_answer) == normalize_answer(gold)) if gold else 0.0

            rec = {
                "doc_id": task["doc_id"],
                "sample_idx": task["sample_idx"],
                "pair_idx": task["pair_idx"],
                "condition": task["condition"],
                "continuation_idx": task["continuation_idx"],
                "first_err_step": task["first_err_step"],
                "cascade_step": task["cascade_step"],
                "n_steps": task["n_steps"],
                "gold_answer": gold,
                "pred_answer": pred_answer,
                "exact_match": em,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[Shard {args._shard_id}] Done -> {out_path}")


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

def _print_summary(merged_path: Path, shard_dir: Path) -> None:
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

    by_cond: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        by_cond[r["condition"]].append(r["exact_match"])

    n_total = len(results)
    print(f"\n=== Cascade recovery rates ({n_total} continuations) ===")
    for cond in ["fix_first", "fix_later", "fix_first_keep_middle"]:
        vals = by_cond[cond]
        if vals:
            rate = sum(vals) / len(vals)
            n_correct = sum(int(v >= 1) for v in vals)
            print(f"  R({cond}): {rate:.4f} ({n_correct}/{len(vals)})")
    print()


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

    samples = load_cascade_samples(args.cascade_file, args.limit)
    print(f"Cascade samples: {len(samples)}")
    total_pairs = sum(len(s["cascade_pairs"]) for s in samples)
    print(f"Total cascade pairs: {total_pairs}")
    print(f"Continuations per condition: {args.n_continuations}")

    completed = load_completed_keys(merged_path, shard_dir)
    if completed:
        print(f"Checkpoint: {len(completed)} already done, skipping")

    all_tasks = []
    for s in samples:
        steps = s["steps"]
        for pi, pair in enumerate(s["cascade_pairs"]):
            fe = pair["first_error"]
            ce = pair["cascade_error"]
            first_step = fe["step"]
            cascade_step = ce["step"]

            pf_first = build_fix_first_prefix(steps, first_step, fe["correction"])
            prompt_first = build_chat_prompt(s["question"]) + pf_first + "\n\n"

            pf_later = build_fix_later_prefix(steps, cascade_step, ce["correction"])
            prompt_later = build_chat_prompt(s["question"]) + pf_later + "\n\n"

            pf_fkm = build_fix_first_keep_middle_prefix(
                steps, first_step, fe["correction"], cascade_step)
            prompt_fkm = build_chat_prompt(s["question"]) + pf_fkm + "\n\n"

            for ci in range(args.n_continuations):
                for cond, prefix_text, prompt in [
                    ("fix_first", pf_first, prompt_first),
                    ("fix_later", pf_later, prompt_later),
                    ("fix_first_keep_middle", pf_fkm, prompt_fkm),
                ]:
                    key = _task_key(s["doc_id"], s["sample_idx"], pi, cond, ci)
                    if key in completed:
                        continue
                    all_tasks.append({
                        "doc_id": s["doc_id"],
                        "sample_idx": s["sample_idx"],
                        "pair_idx": pi,
                        "condition": cond,
                        "continuation_idx": ci,
                        "first_err_step": first_step,
                        "cascade_step": cascade_step,
                        "n_steps": s["n_steps"],
                        "gold_answer": s["gold_answer"],
                        "prefix_text": prefix_text,
                        "prompt": prompt,
                    })

    total_prompts = len(all_tasks)
    if total_prompts == 0:
        print("All tasks already completed.")
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
        log_file = shard_dir / f"log_{si}.txt"
        log_fh = log_file.open("w", encoding="utf-8")
        p = subprocess.Popen(
            cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        procs.append((si, gpu_id, p, log_fh))

    failed_shards: List[int] = []
    succeeded_shards: List[int] = []
    for si, gpu_id, p, log_fh in procs:
        p.wait()
        log_fh.close()
        rc = p.returncode
        log_file = shard_dir / f"log_{si}.txt"
        log_text = log_file.read_text("utf-8", errors="replace")
        lines = log_text.strip().splitlines()
        tail = "\n".join(lines[-5:]) if lines else "(no output)"
        print(f"\n--- Shard {si} (GPU {gpu_id}) exit={rc} ---\n{tail}")
        if rc != 0:
            failed_shards.append(si)
            if len(lines) > 5:
                print("...\n" + "\n".join(lines[-20:]))
        else:
            succeeded_shards.append(si)
            log_file.unlink(missing_ok=True)

    n_new = 0
    ok_files = {shard_dir / f"shard_{si}.jsonl" for si in succeeded_shards}
    with merged_path.open("a", encoding="utf-8") as fout:
        for sf in shard_out_files:
            if sf not in ok_files or not sf.exists():
                continue
            for line in sf.read_text("utf-8").splitlines():
                if line.strip():
                    fout.write(line + "\n")
                    n_new += 1
    print(f"\nAppended {n_new} new continuations -> {merged_path}")

    for si in succeeded_shards:
        (shard_dir / f"shard_{si}.jsonl").unlink(missing_ok=True)
        (shard_dir / f"tasks_{si}.json").unlink(missing_ok=True)

    if failed_shards:
        print(f"\nWARNING: Shards {failed_shards} failed. Re-run to resume.")
        sys.exit(1)
    else:
        try:
            shard_dir.rmdir()
        except OSError:
            pass

    _print_summary(merged_path, shard_dir)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
