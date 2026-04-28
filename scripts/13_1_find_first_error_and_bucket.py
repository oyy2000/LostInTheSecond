#!/usr/bin/env python3
"""
Find first-error block (tau) in wrong MBPP solutions via GPT,
then bucket by early/late tau.

Usage:
    python scripts/13_1_find_first_error_and_bucket.py \
        --model gpt-5.1 --max-workers 8
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.code_step_judge import (
    call_gpt_first_error,
    load_env_file,
    make_client,
    parse_gpt_json,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EARLY_TAU = {2, 3}
LATE_TAU = {4, 5, 6}


def build_mbpp_error_prompt(
    task_description: str,
    code_blocks: list,
) -> str:
    numbered = "\n".join(
        f"[Block {i+1}]\n{b}" for i, b in enumerate(code_blocks)
    )
    return f"""You are a rigorous Python code verifier.

Task description:
{task_description.strip()}

The programmer produced the following WRONG implementation.
The code has been split into logical blocks:

{numbered}

Judgment criteria (follow strictly):
- Focus on LOGICAL correctness against the specification.
- A block is CORRECT if its logic correctly implements the intended step
  and correctly uses variables from preceding blocks.
- A block is INCORRECT if it has a logical bug (wrong formula, off-by-one,
  wrong condition, wrong data structure usage, missing edge case handling).
- Do NOT mark a block wrong just because it propagates an earlier mistake.
- Do NOT mark a block wrong for style issues.

Task:
1. Walk through every block. For each, decide correct/incorrect.
2. Report the FIRST block that is itself incorrect.
3. Explain in one sentence what bug that block introduces.
4. Propose the MINIMAL correction to that block only. The correction must
   contain ONLY the corrected block content.

Output STRICT JSON (no markdown fences):
{{
  "first_error_block": <int, 1-indexed>,
  "total_blocks": {len(code_blocks)},
  "reason": "<one sentence explaining the bug>",
  "correction": "<corrected block content>"
}}
"""


def load_cache(path: Path) -> dict:
    cache = {}
    if not path.exists():
        return cache
    for line in path.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        key = f"{rec['doc_id']}|{rec['sample_idx']}"
        cache[key] = rec
    return cache


def append_cache(path: Path, rec: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def bucket_results(records: list) -> dict:
    early, late, other = [], [], []
    for r in records:
        tau = r.get("tau")
        n = r.get("n_blocks", 0)
        if tau is None:
            other.append(r)
            continue
        rel_pos = tau / n if n > 0 else 1.0
        if rel_pos >= 1.0:
            r["bucket"] = "other"
            other.append(r)
            continue
        if tau in EARLY_TAU and n >= tau + 2:
            r["bucket"] = "early"
            early.append(r)
        elif tau in LATE_TAU and n >= tau + 2:
            r["bucket"] = "late"
            late.append(r)
        else:
            r["bucket"] = "other"
            other.append(r)
    return {"early": early, "late": late, "other": other}


def main():
    parser = argparse.ArgumentParser(
        description="GPT first-error locator + tau bucketing (MBPP)")
    parser.add_argument(
        "--input", type=str,
        default="results/mbpp_3b_multi_sample/raw_code_n8.jsonl")
    parser.add_argument("--output-dir", type=str,
                        default="results/mbpp_3b_multi_sample/first_error")
    parser.add_argument("--model", type=str, default="gpt-5.1")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bucket-only", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    load_env_file(project_root / ".env")

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = project_root / input_path
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "gpt_first_error_cache.jsonl"

    print(f"Loading {input_path}")
    rows = [json.loads(l) for l in input_path.read_text("utf-8").splitlines()
            if l.strip()]
    wrong = [r for r in rows if r.get("exact_match", 1.0) < 1.0]
    print(f"  Total rows: {len(rows)}, wrong-answer: {len(wrong)}")

    if args.max_samples:
        wrong = wrong[:args.max_samples]

    if args.dry_run:
        sample = wrong[0]
        prompt = build_mbpp_error_prompt(
            sample["prompt_text"], sample["blocks"])
        print("\n=== DRY RUN ===")
        print(prompt)
        return

    cache = load_cache(cache_path)
    print(f"  Cached: {len(cache)}")

    tasks = []
    for r in wrong:
        key = f"{r['doc_id']}|{r['sample_idx']}"
        if key not in cache:
            tasks.append(r)
    print(f"  New tasks: {len(tasks)}")

    if tasks and not args.bucket_only:
        client = make_client()
        print(f"Calling GPT ({args.model}), workers={args.max_workers}")

        def _process(row):
            prompt = build_mbpp_error_prompt(
                row["prompt_text"], row["blocks"])
            parsed, raw = call_gpt_first_error(
                client, args.model, prompt,
                temperature=args.temperature)
            tau = None
            if parsed and isinstance(parsed.get("first_error_block"), int):
                tau = parsed["first_error_block"]
            return {
                "doc_id": row["doc_id"],
                "sample_idx": row["sample_idx"],
                "prompt_text": row["prompt_text"],
                "test_list": row["test_list"],
                "test_imports": row.get("test_imports", []),
                "code": row["code"],
                "blocks": row["blocks"],
                "n_blocks": row["n_blocks"],
                "tau": tau,
                "gpt_parsed": parsed,
                "gpt_raw": raw,
            }

        if args.max_workers <= 1:
            for row in tqdm(tasks, desc="GPT first-error"):
                rec = _process(row)
                append_cache(cache_path, rec)
        else:
            with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                futs = {pool.submit(_process, r): r for r in tasks}
                for fut in tqdm(as_completed(futs), total=len(futs),
                                desc="GPT first-error"):
                    rec = fut.result()
                    append_cache(cache_path, rec)

    cache = load_cache(cache_path)
    all_records = list(cache.values())
    print(f"\nTotal cached records: {len(all_records)}")

    valid = [r for r in all_records if r.get("tau") is not None]
    print(f"  Records with valid tau: {len(valid)}")

    buckets = bucket_results(valid)
    for name, items in buckets.items():
        print(f"  {name}: {len(items)}")
        out_path = out_dir / f"bucket_{name}.json"
        out_path.write_text(
            json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"    -> {out_path}")

    if valid:
        from collections import Counter
        tau_dist = Counter(r["tau"] for r in valid)
        print(f"\nTau distribution: {dict(sorted(tau_dist.items()))}")

    print("\nDone.")


if __name__ == "__main__":
    main()
