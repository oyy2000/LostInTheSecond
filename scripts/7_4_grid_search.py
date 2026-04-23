#!/usr/bin/env python3
"""
Grid search over (K, alpha) for Late Rollback + Suffix Vote.

Reuses the draft from the first run (greedy draft is K-independent).
For each (K, alpha) pair, generates K-1 suffixes and does majority vote.
Also runs Full SC baselines for each K.

Runs sequentially: one (K, alpha) config at a time on all GPUs.

Usage:
    python scripts/7_4_grid_search.py --gpus 2,3,4,5,6,7
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
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/grid_search"))
    ap.add_argument("--Ks", default="8,16,32",
                    help="Comma-separated K values")
    ap.add_argument("--alphas", default="0.4,0.5,0.6,0.7,0.8",
                    help="Comma-separated alpha values")
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
    elif mode == "full_sc":
        sp = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
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
                pred = extract_boxed_answer(gen_text)
                rec = {
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
            elif mode == "full_sc":
                pred = extract_boxed_answer(gen_text)
                rec = {
                    "doc_id": task["doc_id"],
                    "sample_idx": task["sample_idx"],
                    "gold_answer": task["gold_answer"],
                    "pred_answer": pred,
                    "exact_match": float(
                        normalize_answer(pred)
                        == normalize_answer(task["gold_answer"])
                    ),
                    "n_tokens": n_tokens,
                    "n_steps": len(split_steps(gen_text)),
                }
            else:
                prefix_text = task["prefix_text"]
                full = prefix_text + (
                    "\n\n" + gen_text if gen_text else ""
                )
                pred = extract_boxed_answer(full)
                rec = {
                    "doc_id": task["doc_id"],
                    "alpha": task["alpha"],
                    "suffix_idx": task["suffix_idx"],
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


# ── coordinator helpers ───────────────────────────────────────────────────

def launch_shards(
    args, tasks: List[Dict], mode: str,
    shard_dir: Path, gpu_ids: List[str],
) -> List[Dict]:
    if not tasks:
        print(f"  [{mode}] 0 tasks, skipping")
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
            json.dumps(shard_tasks[si], ensure_ascii=False), encoding="utf-8",
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
        print(f"WARNING: {mode} shards {failed} failed! "
              f"Continuing with partial results.")

    new_records: List[Dict] = []
    for sf in shard_out_files:
        if not sf.exists():
            continue
        for line in sf.read_text("utf-8").splitlines():
            if line.strip():
                new_records.append(json.loads(line))
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
    Ks = [int(k.strip()) for k in args.Ks.split(",")]
    alphas = [float(a.strip()) for a in args.alphas.split(",")]
    K_max = max(Ks)

    print("=" * 60)
    print(f"Grid Search: Late Rollback + Suffix Vote")
    print(f"  Ks: {Ks}, alphas: {alphas}")
    print(f"  GPUs: {gpu_ids}")
    print("=" * 60)

    t_global = time.time()

    # ── Step 1: Greedy draft (shared across all K/alpha) ─────────────────
    draft_path = out_dir / "drafts.jsonl"
    existing_drafts = load_jsonl(draft_path)
    done_doc_ids: Set[int] = {d["doc_id"] for d in existing_drafts}

    questions = load_questions(args.raw_file, args.limit)
    pending_qs = [q for q in questions if q["doc_id"] not in done_doc_ids]

    print(f"\n[Draft] {len(existing_drafts)} cached, "
          f"{len(pending_qs)} pending")

    if pending_qs:
        t0 = time.time()
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
        print(f"  Draft done in {fmt_time(time.time() - t0)}")

    drafts = existing_drafts
    draft_acc = sum(d["draft_correct"] for d in drafts) / max(len(drafts), 1)
    total_draft_tokens = sum(d["draft_tokens"] for d in drafts)
    print(f"  Greedy accuracy: {draft_acc:.4f}, "
          f"avg tokens: {total_draft_tokens / max(len(drafts), 1):.0f}")

    # ── Step 2: Suffix generation for K_max (largest K) per alpha ────────
    # Generate K_max-1 suffixes per alpha; smaller K subsets are free.
    print(f"\n[Suffix] Generating up to K_max-1={K_max - 1} suffixes "
          f"per question per alpha")

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

            for si in range(K_max - 1):
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

        if suffix_tasks:
            t0 = time.time()
            new_suffixes = launch_shards(
                args, suffix_tasks, f"sfx_a{alpha:.1f}",
                shard_dir, gpu_ids,
            )
            append_jsonl(suffix_path, new_suffixes)
            print(f"  alpha={alpha:.1f} done in {fmt_time(time.time() - t0)}")

    # ── Step 3: Full SC for each K (with resume) ─────────────────────────
    print(f"\n[Full SC] Generating samples for K_max={K_max}")

    sc_path = out_dir / "full_sc_samples.jsonl"
    existing_sc = load_jsonl(sc_path)
    done_sc_keys: Set[str] = {
        f"{r['doc_id']}_{r['sample_idx']}" for r in existing_sc
    }

    sc_tasks = []
    for q in questions:
        prompt = build_chat_prompt(q["question"])
        for si in range(K_max):
            key = f"{q['doc_id']}_{si}"
            if key in done_sc_keys:
                continue
            sc_tasks.append({
                "doc_id": q["doc_id"],
                "sample_idx": si,
                "gold_answer": q["gold_answer"],
                "prompt": prompt,
            })

    print(f"  Cached: {len(existing_sc)}, Pending: {len(sc_tasks)}")

    if sc_tasks:
        t0 = time.time()
        new_sc = launch_shards(
            args, sc_tasks, "full_sc", shard_dir, gpu_ids,
        )
        append_jsonl(sc_path, new_sc)
        existing_sc.extend(new_sc)
        print(f"  Full SC done in {fmt_time(time.time() - t0)}")

    # ── Step 4: Compute all (K, alpha) vote results ──────────────────────
    print(f"\n[Vote] Computing majority vote for all (K, alpha) combos")

    grid_results = {}
    n_q = len(drafts)
    greedy_tpq = total_draft_tokens / max(n_q, 1)

    # Late Rollback results
    for alpha in alphas:
        suffix_path = out_dir / f"suffixes_alpha{alpha:.1f}.jsonl"
        all_suffixes = load_jsonl(suffix_path)
        suffix_by_doc: Dict[int, List[Dict]] = defaultdict(list)
        for s in all_suffixes:
            suffix_by_doc[s["doc_id"]].append(s)

        for K in Ks:
            n_correct = 0
            total_suffix_tokens = 0

            for d in drafts:
                doc_id = d["doc_id"]
                answers = [d["draft_answer"]]
                doc_sfx = suffix_by_doc.get(doc_id, [])
                doc_sfx_k = [s for s in doc_sfx
                             if s["suffix_idx"] < K - 1]
                for s in doc_sfx_k:
                    answers.append(s["pred_answer"])
                    total_suffix_tokens += s["suffix_tokens"]

                voted = majority_vote(answers)
                if normalize_answer(voted) == normalize_answer(
                    d["gold_answer"]
                ):
                    n_correct += 1

            acc = n_correct / max(n_q, 1)
            total_tokens = total_draft_tokens + total_suffix_tokens
            tpq = total_tokens / max(n_q, 1)

            grid_results[f"LR_K{K}_a{alpha:.1f}"] = {
                "method": "late_rollback",
                "K": K,
                "alpha": alpha,
                "accuracy": acc,
                "total_tokens": total_tokens,
                "tokens_per_question": tpq,
                "gain_over_greedy": acc - draft_acc,
            }

    # Full SC results for each K
    sc_by_doc: Dict[int, List[Dict]] = defaultdict(list)
    for r in existing_sc:
        sc_by_doc[r["doc_id"]].append(r)

    gold_map = {q["doc_id"]: q["gold_answer"] for q in questions}

    for K in Ks:
        n_correct = 0
        total_tokens = 0

        for doc_id in sorted(sc_by_doc.keys()):
            samples = sc_by_doc[doc_id]
            samples_k = [s for s in samples if s["sample_idx"] < K]
            gold = gold_map.get(doc_id, samples_k[0]["gold_answer"])
            answers = [s["pred_answer"] for s in samples_k]
            tokens = sum(s["n_tokens"] for s in samples_k)
            total_tokens += tokens

            voted = majority_vote(answers)
            if normalize_answer(voted) == normalize_answer(gold):
                n_correct += 1

        acc = n_correct / max(n_q, 1)
        tpq = total_tokens / max(n_q, 1)

        grid_results[f"FullSC_K{K}"] = {
            "method": "full_sc",
            "K": K,
            "accuracy": acc,
            "total_tokens": total_tokens,
            "tokens_per_question": tpq,
            "gain_over_greedy": acc - draft_acc,
        }

    # ── Step 5: Save + print summary ─────────────────────────────────────
    summary = {
        "model": args.model_id,
        "Ks": Ks,
        "alphas": alphas,
        "n_questions": n_q,
        "greedy_accuracy": draft_acc,
        "greedy_tokens_per_question": greedy_tpq,
        "grid": grid_results,
    }

    summary_path = out_dir / "grid_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    print(f"\n{'=' * 72}")
    print(f"{'Method':<22} {'K':>3} {'alpha':>6} {'Acc':>8} "
          f"{'Tok/Q':>8} {'Gain':>8} {'vs SC':>8}")
    print(f"{'-' * 72}")
    print(f"{'Greedy':<22} {'':>3} {'':>6} {draft_acc:>8.4f} "
          f"{greedy_tpq:>8.0f} {'--':>8} {'--':>8}")

    for K in Ks:
        sc_key = f"FullSC_K{K}"
        sc = grid_results[sc_key]
        sc_tpq = sc["tokens_per_question"]

        print(f"{'Full SC':<22} {K:>3} {'--':>6} {sc['accuracy']:>8.4f} "
              f"{sc_tpq:>8.0f} "
              f"{sc['gain_over_greedy']:>+8.4f} {'--':>8}")

        for alpha in sorted(alphas):
            lr_key = f"LR_K{K}_a{alpha:.1f}"
            lr = grid_results[lr_key]
            savings = 1 - lr["tokens_per_question"] / sc_tpq if sc_tpq else 0
            print(f"{'  LR':<22} {K:>3} {alpha:>6.1f} "
                  f"{lr['accuracy']:>8.4f} "
                  f"{lr['tokens_per_question']:>8.0f} "
                  f"{lr['gain_over_greedy']:>+8.4f} "
                  f"{savings:>7.1%}")

    print(f"{'=' * 72}")
    elapsed = time.time() - t_global
    print(f"\nTotal time: {fmt_time(elapsed)}")
    print(f"Summary -> {summary_path}")
    print("Done.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
