#!/usr/bin/env python3
"""
Construct CORRUPTED prefixes using GPT-5.1 — reverse experiment.

For CORRECT trajectories, incrementally introduce math errors:
  k=1: GPT corrupts step 1 (introduces a subtle math error)
  k=2: GPT corrupts step 2, given already-corrupted step 1 from k=1
  k=3: GPT corrupts step 3, given corrupted steps 1-2 from k=2
  k=4: GPT corrupts step 4, given corrupted steps 1-3 from k=3

This is the symmetric counterpart to construct_corrected_prefix_by_gpt.py.

Usage:
    python scripts/data_prep/construct_corrupted_prefix_by_gpt.py \
        --in-file results/gsm8k_7b_v2/raw_cot_n8.jsonl \
        --out-dir results/gsm8k_7b_v2 \
        --model gpt-5.1
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading

from openai import OpenAI
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.prm.scoring import split_steps

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MIN_STEPS = 4


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def call_gpt(
    client: OpenAI, model: str, prompt: str,
    temperature: float = 0.0, max_output_tokens: int = 2048,
    retries: int = 4, sleep_base: float = 1.5,
) -> str:
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_output_tokens,
            )
            out = (resp.choices[0].message.content or "").strip()
            if out:
                return out
        except Exception as e:
            last_err = e
        time.sleep(sleep_base * (2 ** attempt))
    raise RuntimeError(f"OpenAI API failed after {retries} retries: {last_err}")


def build_corrupt_single_step_prompt(
    question: str,
    corrupted_prefix_steps: List[str],
    correct_step: str,
    step_number: int,
) -> str:
    """Ask GPT to introduce a subtle math error into step `step_number`."""
    prefix_block = ""
    for i, step in enumerate(corrupted_prefix_steps):
        prefix_block += f"Step {i+1} (already corrupted):\n{step}\n\n"

    return f"""You are deliberately introducing a subtle math error into a solution step.

Task: Rewrite ONLY Step {step_number} so it contains a plausible but INCORRECT mathematical reasoning error. The error should be subtle enough that it looks reasonable at first glance.

Hard constraints:
1) Output EXACTLY one step of reasoning.
2) The step MUST contain a mathematical error (wrong calculation, wrong formula, wrong logic).
3) Keep the style and length close to the original step.
4) The error should be subtle — not obviously wrong.
5) Do NOT output any explanation, notes, or markdown fences.
6) Do NOT include "Step N:" labels — just the reasoning text.
7) Do NOT repeat or modify any preceding steps.

Question:
{question}

{prefix_block}Step {step_number} (correct, to be corrupted):
{correct_step}

Corrupted Step {step_number} (with subtle math error):
"""


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


def load_completed_keys(out_path: Path) -> set:
    done = set()
    if not out_path.exists():
        return done
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                done.add((d["doc_id"], d["sample_idx"]))
            except Exception:
                pass
    return done


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="GPT incremental prefix corruption (reverse experiment)")
    ap.add_argument("--env-file", default=".env")
    ap.add_argument("--in-file", required=True)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "results/gsm8k_7b_v2"))
    ap.add_argument("--k-values", default="1,2,3,4")
    ap.add_argument("--model", default="gpt-5.1")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="Higher temp for more diverse errors")
    ap.add_argument("--max-output-tokens", type=int, default=2048)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--sleep-base", type=float, default=1.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    return ap.parse_args()


def process_one_trajectory(
    client: OpenAI, row: Dict[str, Any], max_k: int, args,
) -> Optional[Dict[str, Any]]:
    """Incrementally corrupt steps 1..max_k for one correct trajectory."""
    doc_id = row["doc_id"]
    sample_idx = row.get("sample_idx", 0)
    question = row["question"]
    steps = row.get("steps", split_steps(row.get("response", ""), mode="double_newline"))

    if len(steps) < MIN_STEPS:
        return None

    corrupted_so_far: List[str] = []
    result = {
        "doc_id": doc_id,
        "sample_idx": sample_idx,
        "question": question,
        "gold_answer": row.get("gold_answer", ""),
        "original_response": row.get("response", ""),
        "original_steps": steps,
        "model": args.model,
    }

    for k in range(1, max_k + 1):
        if k > len(steps):
            break

        corrupted_text = call_gpt(
            client, args.model,
            build_corrupt_single_step_prompt(
                question, corrupted_so_far, steps[k - 1], k
            ),
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            retries=args.retries,
            sleep_base=args.sleep_base,
        )

        parsed = split_steps(corrupted_text, mode="double_newline")
        corrupted_step = parsed[0] if parsed else corrupted_text.strip()
        corrupted_so_far.append(corrupted_step)

        result[f"corrupted_steps_k{k}"] = list(corrupted_so_far)
        result[f"remaining_original_steps_k{k}"] = steps[k:]

    return result


def main() -> None:
    args = parse_args()
    load_env_file(Path(args.env_file))

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY", file=sys.stderr)
        sys.exit(2)

    in_path = Path(args.in_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    k_values = [int(x) for x in args.k_values.split(",")]
    max_k = max(k_values)

    # Load input — only CORRECT trajectories with >= MIN_STEPS steps
    all_rows = read_jsonl(in_path)
    correct_rows = [
        r for r in all_rows
        if r.get("exact_match", 0.0) >= 1.0 and r.get("n_steps", 0) >= MIN_STEPS
    ]
    print(f"Loaded {len(all_rows)} trajectories, {len(correct_rows)} correct with >= {MIN_STEPS} steps")

    if args.limit > 0:
        correct_rows = correct_rows[: args.limit]

    combined_out = out_dir / "corrupted_incremental_all.jsonl"
    completed = load_completed_keys(combined_out) if args.resume else set()
    todo = [r for r in correct_rows if (r["doc_id"], r.get("sample_idx", 0)) not in completed]
    print(f"Already done: {len(completed)}, to process: {len(todo)}")

    if not todo:
        print("Nothing to do.")
    else:
        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("BASE_URL"),
        )

        write_lock = threading.Lock()
        mode = "a" if args.resume else "w"
        fout = combined_out.open(mode, encoding="utf-8")
        n_ok, n_fail = 0, 0
        pbar = tqdm(total=len(todo), desc="Incremental corruption", unit="traj")

        def _worker(row):
            return process_one_trajectory(client, row, max_k, args)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, row): row for row in todo}
            for future in as_completed(futures):
                row = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        with write_lock:
                            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                            fout.flush()
                            n_ok += 1
                except Exception as e:
                    n_fail += 1
                    pbar.write(f"[WARN] doc_id={row.get('doc_id','?')} "
                               f"sample={row.get('sample_idx',0)}: {e}")
                pbar.update(1)
                pbar.set_postfix(ok=n_ok, fail=n_fail)

        pbar.close()
        fout.close()
        print(f"Combined: ok={n_ok}, fail={n_fail} -> {combined_out}")

    # Split into per-k output files
    all_results = read_jsonl(combined_out)
    for k in k_values:
        out_path = out_dir / f"corrupted_k{k}_gpt.jsonl"
        n_written = 0
        with out_path.open("w", encoding="utf-8") as f:
            for r in all_results:
                corrupted_key = f"corrupted_steps_k{k}"
                if corrupted_key not in r:
                    continue
                out_row = {
                    "doc_id": r["doc_id"],
                    "sample_idx": r["sample_idx"],
                    "k": k,
                    "question": r["question"],
                    "gold_answer": r.get("gold_answer", ""),
                    "original_response": r.get("original_response", ""),
                    "original_steps": r.get("original_steps", []),
                    "corrupted_steps": r[corrupted_key],
                    "remaining_original_steps": r.get(f"remaining_original_steps_k{k}", []),
                    "model": r.get("model", ""),
                }
                f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                n_written += 1
        print(f"corrupted_k{k}: {n_written} rows -> {out_path}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
