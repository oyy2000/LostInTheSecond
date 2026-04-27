#!/usr/bin/env python3
"""
PRM step-level scoring for all trajectory conditions (v2).

Scores original, corrected-k, and corrupted-k trajectories using
Qwen2.5-Math-PRM-7B via StepScorer. No control conditions.

Usage:
    python scripts/eval/score_all_trajectories.py \
        --cot-file results/gsm8k_7b_v2/raw_cot_n8.jsonl \
        --correction-dir results/gsm8k_7b_v2 \
        --out-file results/gsm8k_7b_v2/prm_scores_all.jsonl \
        --gpus auto
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

for _i, _arg in enumerate(sys.argv):
    if _arg == "--gpus" and _i + 1 < len(sys.argv) and sys.argv[_i + 1] != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_i + 1]
        break

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tqdm.auto import tqdm
from src.prm.scoring import StepScorer, split_steps

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PRM_MODEL = "Qwen/Qwen2.5-Math-PRM-7B"
PRM_DTYPE = "float16"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="PRM scoring for all conditions (v2)")
    ap.add_argument("--cot-file",
                    default=str(PROJECT_ROOT / "results/gsm8k_7b_v2/raw_cot_n8.jsonl"))
    ap.add_argument("--correction-dir",
                    default=str(PROJECT_ROOT / "results/gsm8k_7b_v2"))
    ap.add_argument("--out-file",
                    default=str(PROJECT_ROOT / "results/gsm8k_7b_v2/prm_scores_all.jsonl"))
    ap.add_argument("--k-values", default="1,2,3,4")
    ap.add_argument("--gpus", default="auto")
    ap.add_argument("--condition", default="all",
                    help="Which condition: all, original, corrected-K, corrupted-K")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def select_best_gpu(requested: str, min_free: int = 15000) -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            encoding="utf-8",
        )
        free = {i: int(l.strip()) for i, l in enumerate(out.strip().splitlines()) if l.strip()}
    except Exception:
        return 0
    if requested != "auto":
        ids = [int(x) for x in requested.split(",") if x.strip()]
        usable = {g: free.get(g, 0) for g in ids if free.get(g, 0) >= min_free}
        if usable:
            return max(usable, key=usable.get)
        return ids[0] if ids else 0
    candidates = {g: m for g, m in free.items() if m >= min_free}
    if candidates:
        return max(candidates, key=candidates.get)
    return max(free, key=free.get) if free else 0


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def load_scored_keys(out_path: Path) -> Set[str]:
    done: Set[str] = set()
    if not out_path.exists():
        return done
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                key = f"{d['condition']}|{d['doc_id']}|{d.get('sample_idx',0)}|{d.get('k',0)}"
                done.add(key)
            except Exception:
                pass
    return done


def main() -> None:
    args = parse_args()
    correction_dir = Path(args.correction_dir)
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    k_values = [int(x) for x in args.k_values.split(",")]

    # Build task list: (condition_name, items_list)
    tasks: List[Dict[str, Any]] = []

    # Determine which conditions to score
    score_all = args.condition == "all"

    # 1. Original trajectories
    if score_all or args.condition == "original":
        cot_rows = read_jsonl(Path(args.cot_file))
        for r in cot_rows:
            steps = r.get("steps") or split_steps(r.get("response", ""), mode="double_newline")
            tasks.append({
                "condition": "original",
                "doc_id": r["doc_id"],
                "sample_idx": r.get("sample_idx", 0),
                "k": 0,
                "question": r["question"],
                "steps": steps,
                "exact_match": r.get("exact_match", 0.0),
            })

    # 2. Corrected trajectories
    for k in k_values:
        cond = f"corrected-{k}"
        if not score_all and args.condition != cond:
            continue
        rows = read_jsonl(correction_dir / f"prefilled_corrected_k{k}.jsonl")
        for r in rows:
            steps = r.get("all_steps") or split_steps(r.get("full_response", ""), mode="double_newline")
            tasks.append({
                "condition": f"corrected_k{k}",
                "doc_id": r["doc_id"],
                "sample_idx": r.get("sample_idx", 0),
                "k": k,
                "question": r["question"],
                "steps": steps,
                "exact_match": r.get("exact_match", 0.0),
            })

    # 3. Corrupted trajectories
    for k in k_values:
        cond = f"corrupted-{k}"
        if not score_all and args.condition != cond:
            continue
        rows = read_jsonl(correction_dir / f"prefilled_corrupted_k{k}.jsonl")
        for r in rows:
            steps = r.get("all_steps") or split_steps(r.get("full_response", ""), mode="double_newline")
            tasks.append({
                "condition": f"corrupted_k{k}",
                "doc_id": r["doc_id"],
                "sample_idx": r.get("sample_idx", 0),
                "k": k,
                "question": r["question"],
                "steps": steps,
                "exact_match": r.get("exact_match", 0.0),
            })

    if args.limit > 0:
        tasks = tasks[: args.limit]

    # Resume
    scored_keys = load_scored_keys(out_path) if args.resume else set()
    tasks = [t for t in tasks
             if f"{t['condition']}|{t['doc_id']}|{t['sample_idx']}|{t['k']}" not in scored_keys]

    print(f"Total tasks: {len(tasks)} (already scored: {len(scored_keys)})")
    if not tasks:
        print("Nothing to score.")
        return

    # Load PRM
    gpu_id = select_best_gpu(args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Using GPU {gpu_id}")

    scorer = StepScorer(PRM_MODEL, dtype=PRM_DTYPE)
    print(f"PRM loaded: {PRM_MODEL}")

    mode = "a" if args.resume else "w"
    fout = out_path.open(mode, encoding="utf-8")
    n_scored, n_skip, n_fail = 0, 0, 0
    pbar = tqdm(tasks, desc="PRM scoring")

    for item in pbar:
        steps = item["steps"]
        question = item["question"]
        if not steps or not question:
            n_skip += 1
            continue

        try:
            scores = scorer.score_steps(question, steps)
        except Exception:
            n_fail += 1
            continue

        if len(scores) != len(steps):
            n_fail += 1
            continue

        out_row = {
            "condition": item["condition"],
            "doc_id": item["doc_id"],
            "sample_idx": item.get("sample_idx", 0),
            "k": item.get("k", 0),
            "exact_match": item.get("exact_match", 0.0),
            "n_steps": len(steps),
            "step_scores": scores,
            "min_score": min(scores),
            "mean_score": sum(scores) / len(scores),
        }
        fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        fout.flush()
        n_scored += 1
        pbar.set_postfix(scored=n_scored, skip=n_skip, fail=n_fail)

    fout.close()
    print(f"\nDone. Scored: {n_scored}, skipped: {n_skip}, failed: {n_fail}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
