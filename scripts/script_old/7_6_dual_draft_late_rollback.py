#!/usr/bin/env python3
"""
Dual-Draft Late Rollback + Suffix Vote.

Two drafts per question (1 greedy + 1 sampled), each provides a rollback
prefix at ceil(alpha * T_hat). Suffixes are resampled from both prefixes.
All K final answers (2 drafts + suffixes) go into a single majority vote.

Efficient: all generation tasks (drafts + suffixes for all alphas) are
merged into one batch per GPU. Each GPU loads the model exactly once.

Supports checkpoint/resume.

Usage:
    python scripts/7_6_dual_draft_late_rollback.py \
        --alphas 0.4,0.5,0.6 --K 16 --gpus 2,3,4,5,6,7
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
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
            if d["doc_id"] not in seen:
                seen.add(d["doc_id"])
                questions.append({
                    "doc_id": d["doc_id"],
                    "question": d["question"],
                    "gold_answer": d["gold_answer"],
                })
    if limit > 0:
        questions = questions[:limit]
    return questions


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text("utf-8").splitlines()
            if l.strip()]


def append_jsonl(path: Path, records: List[Dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


# ── arg parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--raw-file", default=DEFAULT_RAW)
    ap.add_argument("--out-dir", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/dual_draft"))
    ap.add_argument("--alphas", default="0.4,0.5,0.6",
                    help="Comma-separated rollback fractions")
    ap.add_argument("--K", type=int, default=16,
                    help="Total votes per (alpha) config")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpus", default="2,3,4,5,6,7")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    # internal shard args
    ap.add_argument("--_shard-id", type=int, default=-1,
                    help=argparse.SUPPRESS)
    ap.add_argument("--_shard-out", default="", help=argparse.SUPPRESS)
    ap.add_argument("--_gpu-id", default="0", help=argparse.SUPPRESS)
    ap.add_argument("--_task-file", default="", help=argparse.SUPPRESS)
    return ap.parse_args()


# ── shard worker (handles all task types in one model load) ───────────────

def run_shard(args) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu_id
    from vllm import LLM, SamplingParams

    tasks = json.loads(Path(args._task_file).read_text("utf-8"))
    print(f"[Shard {args._shard_id}] GPU {args._gpu_id}: {len(tasks)} tasks")

    llm = LLM(
        model=args.model_id,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="half",
    )

    sp_greedy = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    sp_sample = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    # Group by sampling mode for better batching
    greedy_idx = [i for i, t in enumerate(tasks) if t["mode"] == "greedy"]
    sample_idx = [i for i, t in enumerate(tasks) if t["mode"] != "greedy"]

    results = [None] * len(tasks)

    if greedy_idx:
        prompts_g = [tasks[i]["prompt"] for i in greedy_idx]
        outputs_g = llm.generate(prompts_g, sp_greedy)
        for j, i in enumerate(greedy_idx):
            results[i] = outputs_g[j]

    if sample_idx:
        prompts_s = [tasks[i]["prompt"] for i in sample_idx]
        outputs_s = llm.generate(prompts_s, sp_sample)
        for j, i in enumerate(sample_idx):
            results[i] = outputs_s[j]

    out_path = Path(args._shard_out)
    with out_path.open("w", encoding="utf-8") as fout:
        for task, output in zip(tasks, results):
            gen_text = output.outputs[0].text.strip()
            n_tokens = len(output.outputs[0].token_ids)
            mode = task["mode"]

            if mode in ("greedy", "sampled_draft"):
                steps = split_steps(gen_text)
                pred = extract_boxed_answer(gen_text)
                rec = {
                    "task_type": "draft",
                    "draft_type": mode,
                    "doc_id": task["doc_id"],
                    "question": task["question"],
                    "gold_answer": task["gold_answer"],
                    "draft_response": gen_text,
                    "draft_steps": steps,
                    "draft_n_steps": len(steps),
                    "draft_answer": pred,
                    "draft_correct": float(
                        normalize_answer(pred)
                        == normalize_answer(task["gold_answer"])
                    ),
                    "draft_tokens": n_tokens,
                }
            else:
                prefix_text = task["prefix_text"]
                full = prefix_text + (
                    "\n\n" + gen_text if gen_text else ""
                )
                pred = extract_boxed_answer(full)
                rec = {
                    "task_type": "suffix",
                    "doc_id": task["doc_id"],
                    "alpha": task["alpha"],
                    "suffix_idx": task["suffix_idx"],
                    "draft_type": task["draft_type"],
                    "gold_answer": task["gold_answer"],
                    "pred_answer": pred,
                    "exact_match": float(
                        normalize_answer(pred)
                        == normalize_answer(task["gold_answer"])
                    ),
                    "suffix_tokens": n_tokens,
                    "rollback_step": task["rollback_step"],
                    "draft_n_steps": task["draft_n_steps"],
                }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[Shard {args._shard_id}] Done -> {out_path}")


# ── coordinator ───────────────────────────────────────────────────────────

def launch_shards(
    args, tasks: List[Dict], shard_dir: Path, gpu_ids: List[str],
) -> List[Dict]:
    if not tasks:
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
        task_file = shard_dir / f"tasks_{si}.json"
        task_file.write_text(
            json.dumps(shard_tasks[si], ensure_ascii=False), encoding="utf-8",
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
              f"({len(shard_tasks[si])} tasks)")
<<<<<<< HEAD
        log_file = shard_dir / f"log_{si}.txt"
        log_fh = log_file.open("w", encoding="utf-8")
        p = subprocess.Popen(
            cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
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
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
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
        print(f"ERROR: shards {failed} failed!")
        sys.exit(1)

    new_records: List[Dict] = []
    for sf in shard_out_files:
        if not sf.exists():
            continue
        for line in sf.read_text("utf-8").splitlines():
            if line.strip():
                new_records.append(json.loads(line))
        sf.unlink(missing_ok=True)
    for si in range(n_shards):
        tf = shard_dir / f"tasks_{si}.json"
        tf.unlink(missing_ok=True)

    return new_records


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
    alphas = [float(a.strip()) for a in args.alphas.split(",")]
    K = args.K

    print("=" * 60)
    print("Dual-Draft Late Rollback + Suffix Vote")
    print(f"  alphas: {alphas}, K: {K}, GPUs: {gpu_ids}")
    print("=" * 60)

    t_global = time.time()
    questions = load_questions(args.raw_file, args.limit)
    n_q = len(questions)
    print(f"  Questions: {n_q}")

    # ── Phase 1: Drafts (greedy + sampled) ───────────────────────────────
    # Resume: load existing drafts
    draft_path = out_dir / "drafts.jsonl"
    existing_drafts = load_jsonl(draft_path)
    done_draft_keys: Set[str] = {
        f"{d['doc_id']}_{d['draft_type']}" for d in existing_drafts
    }

    draft_tasks = []
    for q in questions:
        prompt = build_chat_prompt(q["question"])
        for dtype in ("greedy", "sampled_draft"):
            key = f"{q['doc_id']}_{dtype}"
            if key in done_draft_keys:
                continue
            draft_tasks.append({
                "doc_id": q["doc_id"],
                "question": q["question"],
                "gold_answer": q["gold_answer"],
                "prompt": prompt,
                "mode": dtype if dtype == "greedy" else "sampled_draft",
            })

    print(f"\n[Phase 1] Drafts: {len(existing_drafts)} cached, "
          f"{len(draft_tasks)} pending")

    if draft_tasks:
        t0 = time.time()
        new_drafts = launch_shards(args, draft_tasks, shard_dir, gpu_ids)
        append_jsonl(draft_path, new_drafts)
        existing_drafts.extend(new_drafts)
        print(f"  Drafts done in {fmt_time(time.time() - t0)}")

    # Index drafts
    drafts_by_doc: Dict[int, Dict[str, Dict]] = defaultdict(dict)
    for d in existing_drafts:
        drafts_by_doc[d["doc_id"]][d["draft_type"]] = d

    greedy_acc = sum(
        1 for d in existing_drafts
        if d["draft_type"] == "greedy" and d["draft_correct"]
    ) / max(n_q, 1)
    print(f"  Greedy accuracy: {greedy_acc:.4f}")

    # ── Phase 2: Suffixes from both drafts, all alphas (one batch) ───────
    # For each (doc, alpha, draft_type), generate suffixes.
    # K total votes = 2 drafts + (K-2) suffixes.
    # Split evenly: (K-2)//2 suffixes per draft prefix.
    n_suffix_per_draft = (K - 2) // 2
    n_suffix_extra = (K - 2) % 2  # give extra to greedy draft

    suffix_path = out_dir / "suffixes.jsonl"
    existing_suffixes = load_jsonl(suffix_path)
    done_sfx_keys: Set[str] = {
        f"{s['doc_id']}_{s['draft_type']}_{s['alpha']}_{s['suffix_idx']}"
        for s in existing_suffixes
    }

    suffix_tasks = []
    skipped = 0

    for q in questions:
        doc_id = q["doc_id"]
        doc_drafts = drafts_by_doc.get(doc_id, {})

        for dtype in ("greedy", "sampled_draft"):
            d = doc_drafts.get(dtype)
            if d is None:
                continue
            T_hat = d["draft_n_steps"]
            if T_hat < 2:
                skipped += 1
                continue

            n_sfx = n_suffix_per_draft + (
                n_suffix_extra if dtype == "greedy" else 0
            )

            for alpha in alphas:
                b = max(1, math.ceil(alpha * T_hat))
                if b >= T_hat:
                    b = T_hat - 1
                if b < 1:
                    b = 1

                prefix_steps = d["draft_steps"][:b]
                prefix_text = "\n\n".join(prefix_steps)
                prompt = (
                    build_chat_prompt(q["question"])
                    + prefix_text + "\n\n"
                )

                for si in range(n_sfx):
                    key = f"{doc_id}_{dtype}_{alpha}_{si}"
                    if key in done_sfx_keys:
                        continue
                    suffix_tasks.append({
                        "doc_id": doc_id,
                        "alpha": alpha,
                        "suffix_idx": si,
                        "draft_type": dtype,
                        "gold_answer": q["gold_answer"],
                        "prefix_text": prefix_text,
                        "prompt": prompt,
                        "rollback_step": b,
                        "draft_n_steps": T_hat,
                        "mode": "suffix",
                    })

    print(f"\n[Phase 2] Suffixes: {len(existing_suffixes)} cached, "
          f"{len(suffix_tasks)} pending (skipped {skipped} short)")

    if suffix_tasks:
        t0 = time.time()
        new_suffixes = launch_shards(args, suffix_tasks, shard_dir, gpu_ids)
        append_jsonl(suffix_path, new_suffixes)
        existing_suffixes.extend(new_suffixes)
        print(f"  Suffixes done in {fmt_time(time.time() - t0)}")

    # ── Phase 3: Majority vote per alpha ─────────────────────────────────
    print(f"\n[Phase 3] Majority vote")

    sfx_index: Dict[str, List[Dict]] = defaultdict(list)
    for s in existing_suffixes:
        key = f"{s['doc_id']}_{s['alpha']}"
        sfx_index[key].append(s)

    summary = {
        "model": args.model_id,
        "K": K,
        "alphas": alphas,
        "n_questions": n_q,
        "n_suffix_per_draft_greedy": n_suffix_per_draft + n_suffix_extra,
        "n_suffix_per_draft_sampled": n_suffix_per_draft,
        "greedy_accuracy": greedy_acc,
        "results": {},
    }

    total_draft_tokens = sum(d["draft_tokens"] for d in existing_drafts)

    for alpha in alphas:
        n_correct = 0
        total_suffix_tokens = 0
        per_question = []

        for q in questions:
            doc_id = q["doc_id"]
            gold = q["gold_answer"]
            doc_drafts = drafts_by_doc.get(doc_id, {})

            answers = []
            for dtype in ("greedy", "sampled_draft"):
                d = doc_drafts.get(dtype)
                if d:
                    answers.append(d["draft_answer"])

            sfx_key = f"{doc_id}_{alpha}"
            for s in sfx_index.get(sfx_key, []):
                answers.append(s["pred_answer"])
                total_suffix_tokens += s["suffix_tokens"]

            voted = majority_vote(answers)
            correct = float(
                normalize_answer(voted) == normalize_answer(gold)
            )
            n_correct += correct

            per_question.append({
                "doc_id": doc_id,
                "gold_answer": gold,
                "voted_answer": voted,
                "voted_correct": correct,
                "n_votes": len(answers),
                "greedy_correct": doc_drafts.get("greedy", {}).get(
                    "draft_correct", 0),
            })

        acc = n_correct / max(n_q, 1)
        total_tokens = total_draft_tokens + total_suffix_tokens
        tpq = total_tokens / max(n_q, 1)

        summary["results"][f"alpha_{alpha:.1f}"] = {
            "accuracy": acc,
            "total_tokens": total_tokens,
            "tokens_per_question": tpq,
            "gain_over_greedy": acc - greedy_acc,
            "suffix_tokens": total_suffix_tokens,
        }

        vote_path = out_dir / f"votes_alpha{alpha:.1f}.jsonl"
        with vote_path.open("w", encoding="utf-8") as f:
            for r in per_question:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"  alpha={alpha:.1f}: acc={acc:.4f} "
              f"({acc - greedy_acc:+.4f} vs greedy), "
              f"tokens/q={tpq:.0f}, votes={per_question[0]['n_votes']}")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    elapsed = time.time() - t_global
    print(f"\nTotal time: {fmt_time(elapsed)}")
    print(f"Summary -> {summary_path}")

    try:
        shard_dir.rmdir()
    except OSError:
        pass

    print("Done.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
