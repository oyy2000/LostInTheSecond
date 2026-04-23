#!/usr/bin/env python3
"""
Draft-Estimated Late Rollback + Suffix Vote.

1. Greedy-decode one draft CoT per question -> step sequence s1..sT_hat.
2. Pick rollback point b = ceil(alpha * T_hat).
3. Keep prefix s1..s_{b-1}, resample K-1 suffixes (temperature > 0).
4. Majority-vote over K final answers (draft counts as 1 vote).

Supports checkpoint/resume: skips already-completed tasks on restart.
Multi-GPU data-parallel via subprocess sharding.

Usage:
    python scripts/7_1_late_rollback_suffix_vote.py \
        --alpha 0.6,0.7,0.8 --K 8 --gpus 2,3,4,5,6,7
"""

import argparse
import json
import math
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
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/late_rollback"))
    ap.add_argument("--alpha", default="0.6,0.7,0.8")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpus", default="2,3,4,5,6,7")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--phase", default="all",
                    choices=["all", "draft", "suffix", "vote"])
    # internal shard args
    ap.add_argument("--_shard-id", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--_shard-out", default="", help=argparse.SUPPRESS)
    ap.add_argument("--_gpu-id", default="0", help=argparse.SUPPRESS)
    ap.add_argument("--_task-file", default="", help=argparse.SUPPRESS)
    ap.add_argument("--_mode", default="", help=argparse.SUPPRESS)
    return ap.parse_args()


# ── shard worker ──────────────────────────────────────────────────────────

def run_shard(args) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu_id
    from vllm import LLM, SamplingParams

    tasks = json.loads(Path(args._task_file).read_text("utf-8"))
    mode = args._mode
    print(f"[Shard {args._shard_id}] GPU {args._gpu_id}: "
          f"{len(tasks)} prompts, mode={mode}")

    if mode == "draft":
        sp = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
    else:
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

            if mode == "draft":
                steps = split_steps(gen_text)
                pred_answer = extract_boxed_answer(gen_text)
                rec = {
                    "doc_id": task["doc_id"],
                    "question": task["question"],
                    "gold_answer": task["gold_answer"],
                    "draft_response": gen_text,
                    "draft_steps": steps,
                    "draft_n_steps": len(steps),
                    "draft_answer": pred_answer,
                    "draft_correct": float(
                        normalize_answer(pred_answer)
                        == normalize_answer(task["gold_answer"])
                    ),
                    "draft_tokens": n_tokens,
                }
            else:
                prefix_text = task["prefix_text"]
                full_response = (
                    prefix_text + ("\n\n" + gen_text if gen_text else "")
                )
                pred_answer = extract_boxed_answer(full_response)
                rec = {
                    "doc_id": task["doc_id"],
                    "alpha": task["alpha"],
                    "suffix_idx": task["suffix_idx"],
                    "gold_answer": task["gold_answer"],
                    "pred_answer": pred_answer,
                    "exact_match": float(
                        normalize_answer(pred_answer)
                        == normalize_answer(task["gold_answer"])
                    ),
                    "suffix_tokens": n_tokens,
                    "rollback_step": task["rollback_step"],
                    "draft_n_steps": task["draft_n_steps"],
                }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[Shard {args._shard_id}] Done -> {out_path}")


# ── coordinator helpers ───────────────────────────────────────────────────

def launch_shards(
    args, tasks: List[Dict], mode: str,
    shard_dir: Path, gpu_ids: List[str],
) -> List[Dict]:
    """Launch GPU shards, wait, return merged records."""
    if not tasks:
        print(f"  [{mode}] Nothing to do (0 tasks)")
        return []

    n_shards = len(gpu_ids)
    shard_tasks = [[] for _ in range(n_shards)]
    for i, t in enumerate(tasks):
        shard_tasks[i % n_shards].append(t)

    script_path = str(Path(__file__).resolve())
    procs: List[Tuple[int, str, subprocess.Popen]] = []
    shard_out_files: List[Path] = []

    for si, gpu_id in enumerate(gpu_ids):
        if not shard_tasks[si]:
            continue
        task_file = shard_dir / f"tasks_{mode}_{si}.json"
        task_file.write_text(
            json.dumps(shard_tasks[si], ensure_ascii=False), encoding="utf-8"
        )
        shard_out = shard_dir / f"shard_{mode}_{si}.jsonl"
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
            "--_mode", mode,
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["TOKENIZERS_PARALLELISM"] = "false"

        print(f"  [{mode}] Shard {si} on GPU {gpu_id} "
              f"({len(shard_tasks[si])} prompts)")
        p = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        procs.append((si, gpu_id, p))

    failed = []
    for si, gpu_id, p in procs:
        stdout, _ = p.communicate()
        text = (stdout.decode("utf-8", errors="replace") if stdout else "")
        rc = p.returncode
        lines = text.strip().splitlines()
        tail = "\n".join(lines[-5:]) if lines else "(no output)"
        print(f"  [{mode}] Shard {si} (GPU {gpu_id}) exit={rc}\n{tail}")
        if rc != 0:
            failed.append(si)
            if len(lines) > 5:
                print("...\n" + "\n".join(lines[-20:]))

    if failed:
        print(f"ERROR: {mode} shards {failed} failed!")
        sys.exit(1)

    new_records: List[Dict] = []
    for sf in shard_out_files:
        if not sf.exists():
            continue
        for line in sf.read_text("utf-8").splitlines():
            if line.strip():
                new_records.append(json.loads(line))

    # cleanup temp shard files
    for sf in shard_out_files:
        sf.unlink(missing_ok=True)
    for si in range(n_shards):
        tf = shard_dir / f"tasks_{mode}_{si}.json"
        tf.unlink(missing_ok=True)

    return new_records


# ── main ──────────────────────────────────────────────────────────────────

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
    alphas = [float(a.strip()) for a in args.alpha.split(",")]
    K = args.K

    print(f"=== Late Rollback + Suffix Vote ===")
    print(f"  alphas: {alphas}, K: {K}, GPUs: {gpu_ids}")

    # ── Phase 1: Greedy draft (with resume) ──────────────────────────────
    draft_path = out_dir / "drafts.jsonl"
    existing_drafts = load_jsonl(draft_path)
    done_doc_ids: Set[int] = {d["doc_id"] for d in existing_drafts}

    questions = load_questions(args.raw_file, args.limit)
    pending_qs = [q for q in questions if q["doc_id"] not in done_doc_ids]

    print(f"\n[Phase 1] Greedy drafts: "
          f"{len(existing_drafts)} cached, {len(pending_qs)} pending")

    if pending_qs and args.phase in ("all", "draft"):
        draft_tasks = [{
            "doc_id": q["doc_id"],
            "question": q["question"],
            "gold_answer": q["gold_answer"],
            "prompt": build_chat_prompt(q["question"]),
        } for q in pending_qs]

        new_drafts = launch_shards(
            args, draft_tasks, "draft", shard_dir, gpu_ids,
        )
        append_jsonl(draft_path, new_drafts)
        existing_drafts.extend(new_drafts)

    drafts = existing_drafts
    if drafts:
        draft_acc = sum(d["draft_correct"] for d in drafts) / len(drafts)
        avg_steps = sum(d["draft_n_steps"] for d in drafts) / len(drafts)
        avg_tokens = sum(d["draft_tokens"] for d in drafts) / len(drafts)
        print(f"  Draft accuracy: {draft_acc:.4f}, "
              f"avg steps: {avg_steps:.1f}, avg tokens: {avg_tokens:.1f}")

    if args.phase == "draft":
        return

    # ── Phase 2: Suffix resampling (with resume per alpha) ───────────────
    print(f"\n[Phase 2] Suffix resampling (K-1={K-1} suffixes per question)")

    for alpha in alphas:
        suffix_path = out_dir / f"suffixes_alpha{alpha:.1f}.jsonl"
        existing_suffixes = load_jsonl(suffix_path)
        done_keys: Set[str] = {
            f"{s['doc_id']}_{s['suffix_idx']}" for s in existing_suffixes
        }

        suffix_tasks = []
        skipped = 0
        for d in drafts:
            T_hat = d["draft_n_steps"]
            if T_hat < 2:
                skipped += 1
                continue
            b = max(1, math.ceil(alpha * T_hat))
            if b >= T_hat:
                b = T_hat - 1
            if b < 1:
                b = 1

            prefix_steps = d["draft_steps"][:b]
            prefix_text = "\n\n".join(prefix_steps)
            prompt = build_chat_prompt(d["question"]) + prefix_text + "\n\n"

            for si in range(K - 1):
                key = f"{d['doc_id']}_{si}"
                if key in done_keys:
                    continue
                suffix_tasks.append({
                    "doc_id": d["doc_id"],
                    "alpha": alpha,
                    "suffix_idx": si,
                    "gold_answer": d["gold_answer"],
                    "prefix_text": prefix_text,
                    "prompt": prompt,
                    "rollback_step": b,
                    "draft_n_steps": T_hat,
                })

        print(f"  alpha={alpha:.1f}: {len(existing_suffixes)} cached, "
              f"{len(suffix_tasks)} pending (skipped {skipped} short)")

        if suffix_tasks and args.phase in ("all", "suffix"):
            new_suffixes = launch_shards(
                args, suffix_tasks, f"suffix_a{alpha:.1f}",
                shard_dir, gpu_ids,
            )
            append_jsonl(suffix_path, new_suffixes)

    if args.phase == "suffix":
        return

    # ── Phase 3: Majority vote ───────────────────────────────────────────
    print(f"\n[Phase 3] Majority vote")

    summary = {
        "model": args.model_id,
        "K": K,
        "alphas": alphas,
        "n_questions": len(drafts),
        "results": {},
    }

    draft_acc = sum(d["draft_correct"] for d in drafts) / max(len(drafts), 1)
    total_draft_tokens = sum(d["draft_tokens"] for d in drafts)
    summary["greedy_accuracy"] = draft_acc
    summary["greedy_total_tokens"] = total_draft_tokens
    print(f"  Greedy baseline: {draft_acc:.4f} ({len(drafts)} questions)")

    for alpha in alphas:
        suffix_path = out_dir / f"suffixes_alpha{alpha:.1f}.jsonl"
        suffixes = load_jsonl(suffix_path)
        if not suffixes:
            print(f"  alpha={alpha}: no suffixes, skipping")
            continue

        suffix_by_doc: Dict[int, List[Dict]] = defaultdict(list)
        for s in suffixes:
            suffix_by_doc[s["doc_id"]].append(s)

        n_correct = 0
        total_suffix_tokens = 0
        per_question = []

        for d in drafts:
            doc_id = d["doc_id"]
            answers = [d["draft_answer"]]
            doc_suffixes = suffix_by_doc.get(doc_id, [])
            for s in doc_suffixes:
                answers.append(s["pred_answer"])
                total_suffix_tokens += s["suffix_tokens"]

            voted = majority_vote(answers)
            gold = d["gold_answer"]
            correct = float(
                normalize_answer(voted) == normalize_answer(gold)
            )
            n_correct += correct

            per_question.append({
                "doc_id": doc_id,
                "gold_answer": gold,
                "draft_answer": d["draft_answer"],
                "draft_correct": d["draft_correct"],
                "voted_answer": voted,
                "voted_correct": correct,
                "n_votes": len(answers),
                "rollback_step": (
                    doc_suffixes[0]["rollback_step"] if doc_suffixes else -1
                ),
                "draft_n_steps": d["draft_n_steps"],
            })

        acc = n_correct / max(len(drafts), 1)
        total_tokens = total_draft_tokens + total_suffix_tokens

        summary["results"][f"alpha_{alpha:.1f}"] = {
            "accuracy": acc,
            "total_tokens": total_tokens,
            "suffix_tokens": total_suffix_tokens,
            "draft_tokens": total_draft_tokens,
            "gain_over_greedy": acc - draft_acc,
            "tokens_per_question": total_tokens / max(len(drafts), 1),
        }

        vote_path = out_dir / f"votes_alpha{alpha:.1f}.jsonl"
        with vote_path.open("w", encoding="utf-8") as f:
            for r in per_question:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"  alpha={alpha:.1f}: acc={acc:.4f} "
              f"({acc - draft_acc:+.4f} vs greedy), "
              f"tokens={total_tokens:,}")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\nSummary -> {summary_path}")
    print("Done.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
