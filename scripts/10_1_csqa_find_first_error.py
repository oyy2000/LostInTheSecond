#!/usr/bin/env python3
"""
Find first-error step (tau) in wrong CommonsenseQA trajectories via GPT,
then bucket by early/late tau.

Usage:
    python scripts/10_1_csqa_find_first_error.py \
        --input results/csqa_3b_multi_sample/raw_cot_n8.jsonl \
        --output-dir results/csqa_3b_multi_sample/first_error \
        --model gpt-5.1 --max-workers 8
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.step_judge import (
    build_first_error_prompt_commonsense,
    call_gpt_first_error,
    load_env_file,
    make_client,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def split_steps(text: str):
    text = (text or "").strip()
    if not text:
        return []
    s = [x.strip() for x in text.split("\n\n") if x.strip()]
    return s if s else [text]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(
        PROJECT_ROOT / "results/csqa_3b_multi_sample/raw_cot_n8.jsonl"))
    ap.add_argument("--output-dir", default=str(
        PROJECT_ROOT / "results/csqa_3b_multi_sample/first_error"))
    ap.add_argument("--model", default="gpt-5.1")
    ap.add_argument("--max-workers", type=int, default=8)
    return ap.parse_args()


def main():
    args = parse_args()
    load_env_file(PROJECT_ROOT / ".env")
    client = make_client()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "gpt_first_error_cache.jsonl"

    # Load existing cache
    cached = {}
    if cache_path.exists():
        for line in cache_path.read_text("utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                cached[(r["doc_id"], r["sample_idx"])] = r

    # Load input
    rows = [json.loads(l) for l in Path(args.input).read_text("utf-8").splitlines() if l.strip()]
    wrong = [r for r in rows if r.get("exact_match", 1.0) < 1.0]
    new_tasks = [(r["doc_id"], r["sample_idx"], r) for r in wrong
                 if (r["doc_id"], r["sample_idx"]) not in cached]

    print(f"  Total rows: {len(rows)}, wrong-answer: {len(wrong)}")
    print(f"  Cached: {len(cached)}")
    print(f"  New tasks: {len(new_tasks)}")

    if new_tasks:
        print(f"Calling GPT ({args.model}), workers={args.max_workers}")
        cache_fh = cache_path.open("a", encoding="utf-8")

        def process(task):
            doc_id, sample_idx, rec = task
            steps = split_steps(rec["response"])
            # Build choices text from the question field
            prompt = build_first_error_prompt_commonsense(
                question=rec["question"],
                choices_text="(included in question above)",
                gold_letter=rec["gold_answer"],
                steps=steps,
            )
            parsed, raw = call_gpt_first_error(client, args.model, prompt)
            result = {
                "doc_id": doc_id,
                "sample_idx": sample_idx,
                "n_steps": len(steps),
            }
            if parsed and isinstance(parsed.get("first_error_step"), int):
                tau = parsed["first_error_step"]
                result["tau"] = tau
                result["reason"] = parsed.get("reason", "")
                result["correction"] = parsed.get("correction", "")
            else:
                result["tau"] = None
                result["reason"] = raw[:200] if raw else "parse_error"
                result["correction"] = ""
            return result

        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {pool.submit(process, t): t for t in new_tasks}
            for fut in tqdm(as_completed(futures), total=len(futures),
                           desc="GPT first-error"):
                result = fut.result()
                cached[(result["doc_id"], result["sample_idx"])] = result
                cache_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                cache_fh.flush()
        cache_fh.close()

    # Bucket
    valid = [r for r in cached.values()
             if r.get("tau") is not None and isinstance(r["tau"], int) and r["tau"] > 0]
    if not valid:
        print("No valid tau records found.")
        return

    # Attach full trajectory info
    row_map = {(r["doc_id"], r["sample_idx"]): r for r in wrong}
    for v in valid:
        key = (v["doc_id"], v["sample_idx"])
        if key in row_map:
            v["question"] = row_map[key]["question"]
            v["gold_answer"] = row_map[key]["gold_answer"]
            v["response"] = row_map[key]["response"]
            v["pred_answer"] = row_map[key].get("pred_answer", "")

    # Split by median relative position
    rel_positions = [v["tau"] / v["n_steps"] for v in valid if v["n_steps"] > 0]
    median_rel = sorted(rel_positions)[len(rel_positions) // 2] if rel_positions else 0.5

    early, late, other = [], [], []
    for v in valid:
        if v["n_steps"] <= 0:
            other.append(v)
            continue
        rel = v["tau"] / v["n_steps"]
        if rel <= median_rel:
            early.append(v)
        else:
            late.append(v)

    for label, bucket in [("early", early), ("late", late), ("other", other)]:
        bp = out_dir / f"bucket_{label}.json"
        bp.write_text(json.dumps(bucket, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  {label}: {len(bucket)}")
        print(f"    -> {bp}")

    print(f"\nTotal cached records: {len(cached)}")
    print(f"  Records with valid tau: {len(valid)}")

    if valid:
        from collections import Counter
        tau_dist = Counter(r["tau"] for r in valid)
        print(f"\nTau distribution: {dict(sorted(tau_dist.items()))}")

    print("\nDone.")


if __name__ == "__main__":
    main()
