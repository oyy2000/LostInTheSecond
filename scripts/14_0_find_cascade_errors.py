#!/usr/bin/env python3
"""
Phase 14_0: Find ALL errors in wrong GSM8K trajectories and identify cascades.

For each wrong trajectory that already has a first-error annotation, call GPT
to locate every incorrect step and classify each as "independent" or "cascade".
Filter to trajectories with at least two errors where a later error cascades
from an earlier one.

Outputs:
    cascade_samples.json  -- samples with >=2 errors including cascade pairs
    cascade_stats.json    -- summary statistics

Usage:
    python scripts/14_0_find_cascade_errors.py [--limit N] [--workers N]
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.step_judge import (
    build_all_errors_prompt,
    call_gpt_all_errors,
    load_env_file,
    make_client,
    parse_gpt_json,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIRST_ERROR_DIR = PROJECT_ROOT / "results/gsm8k_3b_multi_sample/first_error"
OUT_DIR = PROJECT_ROOT / "results/gsm8k_3b_multi_sample/cascade_errors"
CACHE_FILE = OUT_DIR / "gpt_all_errors_cache.jsonl"
GPT_MODEL = "gpt-4.1-mini"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-error-dir", default=str(FIRST_ERROR_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--gpt-model", default=GPT_MODEL)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    return ap.parse_args()


def load_wrong_trajectories(first_error_dir: Path, limit: int):
    """Load all bucketed wrong trajectories (early + late)."""
    samples = []
    for fname in ["bucket_early.json", "bucket_late.json"]:
        path = first_error_dir / fname
        if not path.exists():
            print(f"WARNING: {path} not found")
            continue
        data = json.loads(path.read_text("utf-8"))
        for r in data:
            if r.get("gpt_parsed") and r["gpt_parsed"].get("correction"):
                samples.append(r)
    if limit > 0:
        samples = samples[:limit]
    return samples


def load_cache(cache_path: Path):
    """Load already-processed samples from cache."""
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                key = f"{r['doc_id']}:{r['sample_idx']}"
                cache[key] = r
            except (json.JSONDecodeError, KeyError):
                pass
    return cache


def process_one(sample, client, model):
    """Call GPT to find all errors in one trajectory."""
    prompt = build_all_errors_prompt(
        sample["question"], sample["gold_answer"], sample["steps"]
    )
    parsed, raw = call_gpt_all_errors(client, model, prompt)
    return {
        "doc_id": sample["doc_id"],
        "sample_idx": sample["sample_idx"],
        "question": sample["question"],
        "gold_answer": sample["gold_answer"],
        "steps": sample["steps"],
        "n_steps": sample["n_steps"],
        "tau": sample["tau"],
        "bucket": sample["bucket"],
        "first_error_correction": sample["gpt_parsed"]["correction"],
        "all_errors_parsed": parsed,
        "all_errors_raw": raw,
    }


def extract_cascade_pairs(result):
    """Extract valid cascade pairs from all-errors annotation.

    Returns list of dicts, each with:
      - first_error: the independent error (earlier step)
      - cascade_error: the cascade error (later step, caused by first_error)
    """
    parsed = result.get("all_errors_parsed")
    if not parsed or not parsed.get("errors"):
        return []

    errors = parsed["errors"]
    if len(errors) < 2:
        return []

    independent = [e for e in errors if e.get("type") == "independent"]
    cascades = [e for e in errors if e.get("type") == "cascade"]

    if not independent or not cascades:
        return []

    pairs = []
    for ce in cascades:
        source_step = ce.get("cascade_from")
        if source_step is None:
            continue
        source = next((e for e in independent if e["step"] == source_step), None)
        if source and source.get("correction") and ce.get("correction"):
            pairs.append({"first_error": source, "cascade_error": ce})
    return pairs


def main():
    args = parse_args()
    load_env_file(PROJECT_ROOT / ".env")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "gpt_all_errors_cache.jsonl"

    samples = load_wrong_trajectories(Path(args.first_error_dir), args.limit)
    print(f"Loaded {len(samples)} wrong trajectories")

    cache = load_cache(cache_path)
    todo = [s for s in samples if f"{s['doc_id']}:{s['sample_idx']}" not in cache]
    print(f"Cache: {len(cache)} done, {len(todo)} remaining")

    if todo:
        client = make_client()
        done_count = 0
        with cache_path.open("a", encoding="utf-8") as fout:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(process_one, s, client, args.gpt_model): s
                    for s in todo
                }
                for fut in as_completed(futures):
                    result = fut.result()
                    key = f"{result['doc_id']}:{result['sample_idx']}"
                    cache[key] = result
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
                    done_count += 1
                    if done_count % 50 == 0:
                        print(f"  Processed {done_count}/{len(todo)}")
        print(f"All {len(todo)} samples processed")

    # Filter to cascade samples
    cascade_samples = []
    for key, result in cache.items():
        pairs = extract_cascade_pairs(result)
        if pairs:
            result_copy = {k: v for k, v in result.items() if k != "all_errors_raw"}
            result_copy["cascade_pairs"] = pairs
            cascade_samples.append(result_copy)

    cascade_samples.sort(key=lambda x: (x["doc_id"], x["sample_idx"]))

    out_path = out_dir / "cascade_samples.json"
    out_path.write_text(
        json.dumps(cascade_samples, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Stats
    n_total = len(cache)
    n_cascade = len(cascade_samples)
    n_pairs = sum(len(s["cascade_pairs"]) for s in cascade_samples)
    stats = {
        "total_wrong_trajectories": n_total,
        "trajectories_with_cascade": n_cascade,
        "total_cascade_pairs": n_pairs,
        "cascade_rate": n_cascade / n_total if n_total else 0,
    }
    stats_path = out_dir / "cascade_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\n=== Cascade Error Statistics ===")
    print(f"  Total wrong trajectories: {n_total}")
    print(f"  With cascade errors:      {n_cascade} ({stats['cascade_rate']:.1%})")
    print(f"  Total cascade pairs:      {n_pairs}")
    print(f"\nSaved: {out_path}")
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
