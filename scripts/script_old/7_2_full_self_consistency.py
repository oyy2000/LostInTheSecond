#!/usr/bin/env python3
"""
Full Self-Consistency baseline (K=8).

Sample K independent CoT trajectories per question at temperature > 0,
then majority-vote. Standard SC baseline for comparison with late rollback.

Supports checkpoint/resume: skips already-completed (doc_id, sample_idx).
Multi-GPU data-parallel via subprocess sharding.

Usage:
    python scripts/7_2_full_self_consistency.py \
        --K 8 --gpus 2,3,4,5,6,7
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
DEFAULT_RAW = str(
    PROJECT_ROOT / "results/gsm8k_3b_multi_sample/raw_cot_n8.jsonl"
)


# ── helpers ───────────────────────────────────────────────────────────────

def split_steps(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    steps = [x.strip() for x in text.split("\n\n") if x.strip()]
    return steps if steps else [text]


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


def majority_vote(answers: List[str]) -> str:
    normed = [normalize_answer(a) for a in answers if normalize_answer(a)]
    if not normed:
        return ""
    return Counter(normed).most_common(1)[0][0]


def load_questions(raw_file: str, limit: int) -> List[Dict[str, Any]]:
    seen: Set[int] = set()
    questions = []
    with open(raw_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            doc_id = d["doc_id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            questions.append({
                "doc_id": doc_id,
                "question": d["question"],
                "gold_answer": d["gold_answer"],
            })
    if limit > 0:
        questions = questions[:limit]
    return questions


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text("utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def append_jsonl(path: Path, records: List[Dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── arg parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--raw-file", default=DEFAULT_RAW)
    ap.add_argument("--out-dir", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/full_sc"))
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpus", default="2,3,4,5,6,7")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    # internal shard args
    ap.add_argument("--_shard-id", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--_shard-out", default="", help=argparse.SUPPRESS)
    ap.add_argument("--_gpu-id", default="0", help=argparse.SUPPRESS)
    ap.add_argument("--_task-file", default="", help=argparse.SUPPRESS)
    return ap.parse_args()


# ── shard worker ──────────────────────────────────────────────────────────

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
            gen_text = output.outputs[0].text.strip()
            n_tokens = len(output.outputs[0].token_ids)
            pred_answer = extract_boxed_answer(gen_text)
            steps = split_steps(gen_text)
            rec = {
                "doc_id": task["doc_id"],
                "sample_idx": task["sample_idx"],
                "gold_answer": task["gold_answer"],
                "pred_answer": pred_answer,
                "exact_match": float(
                    normalize_answer(pred_answer)
                    == normalize_answer(task["gold_answer"])
                ),
                "n_tokens": n_tokens,
                "n_steps": len(steps),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[Shard {args._shard_id}] Done -> {out_path}")


# ── coordinator ───────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args._shard_id >= 0:
        run_shard(args)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    K = args.K

    questions = load_questions(args.raw_file, args.limit)

    # ── resume: load existing samples ────────────────────────────────────
    merged_path = out_dir / "samples.jsonl"
    existing = load_jsonl(merged_path)
    done_keys: Set[str] = {
        f"{r['doc_id']}_{r['sample_idx']}" for r in existing
    }

    all_tasks = []
    for q in questions:
        prompt = build_chat_prompt(q["question"])
        for si in range(K):
            key = f"{q['doc_id']}_{si}"
            if key in done_keys:
                continue
            all_tasks.append({
                "doc_id": q["doc_id"],
                "sample_idx": si,
                "gold_answer": q["gold_answer"],
                "prompt": prompt,
            })

    print(f"=== Full Self-Consistency (K={K}) ===")
    print(f"  Questions: {len(questions)}, GPUs: {gpu_ids}")
    print(f"  Cached: {len(existing)}, Pending: {len(all_tasks)}")

    if all_tasks:
        n_shards = len(gpu_ids)
        shard_tasks = [[] for _ in range(n_shards)]
        for i, t in enumerate(all_tasks):
            shard_tasks[i % n_shards].append(t)

        script_path = str(Path(__file__).resolve())
        procs: List[Tuple[int, str, subprocess.Popen]] = []
        shard_out_files: List[Path] = []

        for si, gpu_id in enumerate(gpu_ids):
            if not shard_tasks[si]:
                continue
            task_file = shard_dir / f"tasks_{si}.json"
            task_file.write_text(
                json.dumps(shard_tasks[si], ensure_ascii=False),
                encoding="utf-8",
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
                "--_shard-out", str(shard_out),
                "--_gpu-id", gpu_id,
                "--_task-file", str(task_file),
            ]

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env["TOKENIZERS_PARALLELISM"] = "false"

            print(f"  Shard {si} on GPU {gpu_id} "
                  f"({len(shard_tasks[si])} prompts)")
<<<<<<< HEAD
            log_file = shard_dir / f"log_{si}.txt"
            log_fh = log_file.open("w", encoding="utf-8")
            p = subprocess.Popen(
                cmd, env=env,
                stdout=log_fh, stderr=subprocess.STDOUT,
            )
            procs.append((si, gpu_id, p, log_fh))

        failed = []
        for si, gpu_id, p, log_fh in procs:
            p.wait()
            log_fh.close()
            rc = p.returncode
            log_path = shard_dir / f"log_{si}.txt"
            text = log_path.read_text("utf-8", errors="replace")
=======
            p = subprocess.Popen(
                cmd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            procs.append((si, gpu_id, p))

        failed = []
        for si, gpu_id, p in procs:
            stdout, _ = p.communicate()
            text = (stdout.decode("utf-8", errors="replace") if stdout else "")
            rc = p.returncode
>>>>>>> 157b73200cc7137e5adbbe2a049fe49b4c83142e
            lines = text.strip().splitlines()
            tail = "\n".join(lines[-5:]) if lines else "(no output)"
            print(f"  Shard {si} (GPU {gpu_id}) exit={rc}\n{tail}")
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
            print(f"ERROR: Shards {failed} failed!")
            sys.exit(1)

        new_records = []
        for sf in shard_out_files:
            if not sf.exists():
                continue
            for line in sf.read_text("utf-8").splitlines():
                if line.strip():
                    new_records.append(json.loads(line))

        append_jsonl(merged_path, new_records)
        existing.extend(new_records)

        for sf in shard_out_files:
            sf.unlink(missing_ok=True)
        for si in range(n_shards):
            tf = shard_dir / f"tasks_{si}.json"
            tf.unlink(missing_ok=True)

    # ── majority vote ────────────────────────────────────────────────────
    by_doc: Dict[int, List[Dict]] = defaultdict(list)
    for r in existing:
        by_doc[r["doc_id"]].append(r)

    gold_map = {q["doc_id"]: q["gold_answer"] for q in questions}

    n_correct_vote = 0
    total_tokens = 0
    per_question = []

    for doc_id in sorted(by_doc.keys()):
        samples = by_doc[doc_id]
        gold = gold_map.get(doc_id, samples[0]["gold_answer"])
        answers = [s["pred_answer"] for s in samples]
        tokens = sum(s["n_tokens"] for s in samples)
        total_tokens += tokens

        voted = majority_vote(answers)
        correct = float(normalize_answer(voted) == normalize_answer(gold))
        n_correct_vote += correct

        per_question.append({
            "doc_id": doc_id,
            "gold_answer": gold,
            "voted_answer": voted,
            "voted_correct": correct,
            "n_samples": len(samples),
            "total_tokens": tokens,
            "any_correct": max(
                float(normalize_answer(s["pred_answer"])
                      == normalize_answer(gold))
                for s in samples
            ),
        })

    n_q = len(per_question)
    acc = n_correct_vote / max(n_q, 1)
    individual_acc = (
        sum(r["exact_match"] for r in existing) / max(len(existing), 1)
    )

    summary = {
        "model": args.model_id,
        "K": K,
        "n_questions": n_q,
        "individual_accuracy": individual_acc,
        "majority_vote_accuracy": acc,
        "total_tokens": total_tokens,
        "tokens_per_question": total_tokens / max(n_q, 1),
        "pass_at_k": (
            sum(r["any_correct"] for r in per_question) / max(n_q, 1)
        ),
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    vote_path = out_dir / "votes.jsonl"
    with vote_path.open("w", encoding="utf-8") as f:
        for r in per_question:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n=== Full SC Results (K={K}) ===")
    print(f"  Individual accuracy: {individual_acc:.4f}")
    print(f"  Majority vote accuracy: {acc:.4f}")
    print(f"  Pass@{K}: {summary['pass_at_k']:.4f}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Tokens/question: {total_tokens / max(n_q, 1):.0f}")
    print(f"\nSummary -> {summary_path}")

    try:
        shard_dir.rmdir()
    except OSError:
        pass

    print("Done.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
