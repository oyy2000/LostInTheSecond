#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT Step-Level PRM Validation.

For each sample's pos/neg response, use GPT to judge every reasoning step
given the question and preceding steps. Output: 1 (correct) or 0 (incorrect).
Plot step-level correctness curves comparable to PRM step-score curves.

Usage:
    python scripts/eval/28_gpt_step_verification.py \
        --input artifacts_real/samples_gsm8k_train_ds2_fix_step2_gpt_prefill.json \
        --output-dir runs/gpt_step_verification \
        --model gpt-5.1 \
        --max-workers 4
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openai import OpenAI
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# .env loader (same pattern as scripts/data_prep/01_construct_datasets_by_GPT.py)
# ---------------------------------------------------------------------------

def load_env_file(env_path: Path):
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_verify_step_prompt(
    question: str,
    preceding_steps: List[str],
    current_step: str,
    step_index: int,
) -> str:
    """Build a prompt asking GPT to verify a single reasoning step."""
    ctx = ""
    if preceding_steps:
        numbered = "\n".join(
            f"Step {i+1}: {s}" for i, s in enumerate(preceding_steps)
        )
        ctx = f"\nPreceding steps (for context only):\n{numbered}\n"

    return f"""You are verifying a math solution step by step.

Task: Determine whether Step {step_index + 1} is mathematically correct,
given the Question and any preceding steps as context.

Rules:
- Judge ONLY Step {step_index + 1}. Do not judge preceding steps.
- A step is correct if its logic and arithmetic are valid given the preceding context.
- A step that merely restates the problem or sets up notation is correct.

Output format (strict):
First line must be exactly one token: CORRECT or INCORRECT
Second line: one short reason (≤30 words).

Question:
{question}
{ctx}
Step {step_index + 1} (to verify):
{current_step}
"""


def parse_judge_label(text: str) -> Optional[int]:
    """Parse GPT response into 1 (correct) or 0 (incorrect) or None."""
    s = (text or "").strip().upper()
    if not s:
        return None
    first = s.splitlines()[0].strip()
    if first == "CORRECT":
        return 1
    if first == "INCORRECT":
        return 0
    if "CORRECT" in first and "INCORRECT" not in first:
        return 1
    if "INCORRECT" in first:
        return 0
    return None


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------

def call_gpt_verify(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_output_tokens: int = 100,
    retries: int = 5,
    sleep_base: float = 1.0,
) -> Tuple[Optional[int], str]:
    """Call GPT to verify a step. Returns (label, raw_text)."""
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            raw = (resp.output_text or "").strip()
            if raw:
                return parse_judge_label(raw), raw
        except AttributeError:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                )
                raw = (resp.choices[0].message.content or "").strip()
                if raw:
                    return parse_judge_label(raw), raw
            except Exception as e:
                last_err = e
        except Exception as e:
            last_err = e

        time.sleep(sleep_base * (2 ** attempt))

    print(f"  [WARN] API failed after {retries} retries: {last_err}", file=sys.stderr)
    return None, ""


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

CacheKey = Tuple[int, str, int]  # (doc_id, resp_type, step_index)


def load_cache(cache_path: Path) -> Dict[str, Dict]:
    """Load JSONL cache. Returns dict keyed by 'doc_id|resp_type|step_idx'."""
    cache: Dict[str, Dict] = {}
    if not cache_path.exists():
        return cache
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = f"{rec['doc_id']}|{rec['resp_type']}|{rec['step_idx']}"
            cache[key] = rec
    return cache


def append_cache(cache_path: Path, rec: Dict):
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def make_cache_key(doc_id: int, resp_type: str, step_idx: int) -> str:
    return f"{doc_id}|{resp_type}|{step_idx}"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_step_correctness_pos_vs_neg(
    results: List[Dict],
    out_dir: Path,
    title: str = "GPT Step-Level Correctness: pos vs neg",
):
    """Plot avg step correctness for pos and neg responses."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group by resp_type and step_idx
    pos_by_step: Dict[int, List[int]] = {}
    neg_by_step: Dict[int, List[int]] = {}

    for r in results:
        bucket = pos_by_step if r["resp_type"] == "pos" else neg_by_step
        idx = r["step_idx"]
        if r["label"] is not None:
            bucket.setdefault(idx, []).append(r["label"])

    def _curve(by_step):
        xs, ys, ns = [], [], []
        for k in sorted(by_step.keys()):
            vals = by_step[k]
            xs.append(k + 1)  # 1-based
            ys.append(sum(vals) / len(vals))
            ns.append(len(vals))
        return xs, ys, ns

    px, py, pn = _curve(pos_by_step)
    nx, ny, nn = _curve(neg_by_step)

    plt.figure(figsize=(10, 6))
    if px:
        plt.plot(px, py, marker="o", linewidth=2, color="tab:blue",
                 label=f"pos (corrected)")
    if nx:
        plt.plot(nx, ny, marker="x", linewidth=2, color="tab:red",
                 label=f"neg (original)")

    # Annotate sample counts
    for x, y, n in zip(px, py, pn):
        plt.text(x, y + 0.01, f"n={n}", fontsize=7, ha="center",
                 va="bottom", color="tab:blue", alpha=0.7)
    for x, y, n in zip(nx, ny, nn):
        plt.text(x, y - 0.03, f"n={n}", fontsize=7, ha="center",
                 va="top", color="tab:red", alpha=0.7)

    plt.xlabel("Step index")
    plt.ylabel("Avg correctness (GPT judge)")
    plt.title(title)
    plt.ylim(-0.05, 1.1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "gpt_step_correctness_pos_vs_neg.png", dpi=180)
    plt.close()
    print(f"  Saved: {out_dir / 'gpt_step_correctness_pos_vs_neg.png'}")


def plot_step_correctness_by_divergence(
    results: List[Dict],
    samples: List[Dict],
    out_dir: Path,
):
    """Plot neg step correctness split by where pos/neg diverge (step 2)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build lookup: doc_id -> sample
    id_to_sample = {s["doc"]["id"]: s for s in samples}

    # For neg responses, split into: steps before divergence vs at/after
    # In this dataset, divergence is at step 2 (index 1)
    neg_results = [r for r in results if r["resp_type"] == "neg" and r["label"] is not None]

    by_step: Dict[int, List[int]] = {}
    for r in neg_results:
        by_step.setdefault(r["step_idx"], []).append(r["label"])

    xs, ys, ns = [], [], []
    for k in sorted(by_step.keys()):
        vals = by_step[k]
        xs.append(k + 1)
        ys.append(sum(vals) / len(vals))
        ns.append(len(vals))

    if not xs:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, marker="x", linewidth=2, color="tab:red", label="neg steps")

    # Mark step 2 (the divergence point)
    if 2 in [x for x in xs]:
        idx_2 = xs.index(2)
        plt.axvline(x=2, linestyle="--", color="gray", alpha=0.6, label="divergence (step 2)")
        plt.annotate(f"step 2: {ys[idx_2]:.2f}", xy=(2, ys[idx_2]),
                     xytext=(2.5, ys[idx_2] - 0.1),
                     arrowprops=dict(arrowstyle="->", color="gray"),
                     fontsize=9, color="gray")

    for x, y, n in zip(xs, ys, ns):
        plt.text(x, y + 0.01, f"n={n}", fontsize=7, ha="center", va="bottom", alpha=0.7)

    plt.xlabel("Step index")
    plt.ylabel("Avg correctness (GPT judge)")
    plt.title("GPT Step-Level Correctness: neg responses (error propagation)")
    plt.ylim(-0.05, 1.1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "gpt_step_correctness_neg_error_propagation.png", dpi=180)
    plt.close()
    print(f"  Saved: {out_dir / 'gpt_step_correctness_neg_error_propagation.png'}")


def plot_agreement_summary(results: List[Dict], out_dir: Path):
    """Bar chart: GPT agreement rate for pos vs neg, and per-step."""
    out_dir.mkdir(parents=True, exist_ok=True)

    pos_labels = [r["label"] for r in results if r["resp_type"] == "pos" and r["label"] is not None]
    neg_labels = [r["label"] for r in results if r["resp_type"] == "neg" and r["label"] is not None]

    pos_rate = sum(pos_labels) / len(pos_labels) if pos_labels else 0
    neg_rate = sum(neg_labels) / len(neg_labels) if neg_labels else 0

    plt.figure(figsize=(6, 4))
    bars = plt.bar(["pos (corrected)", "neg (original)"], [pos_rate, neg_rate],
                   color=["tab:blue", "tab:red"], alpha=0.8)
    for bar, rate, n in zip(bars, [pos_rate, neg_rate], [len(pos_labels), len(neg_labels)]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{rate:.3f}\n(n={n})", ha="center", va="bottom", fontsize=10)
    plt.ylabel("Fraction judged correct by GPT")
    plt.title("Overall GPT Step Correctness Rate")
    plt.ylim(0, 1.15)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "gpt_overall_correctness_bar.png", dpi=180)
    plt.close()
    print(f"  Saved: {out_dir / 'gpt_overall_correctness_bar.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_tasks(
    samples: List[Dict],
    cache: Dict[str, Dict],
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Build list of (doc_id, resp_type, step_idx, prompt) tasks not yet cached."""
    tasks = []
    subset = samples[:max_samples] if max_samples else samples
    for sample in subset:
        doc_id = sample["doc"]["id"]
        question = sample["doc"]["question"]
        for resp_type in ("pos", "neg"):
            steps = sample.get(f"{resp_type}_steps", [])
            for k, step in enumerate(steps):
                ck = make_cache_key(doc_id, resp_type, k)
                if ck in cache:
                    continue
                prompt = build_verify_step_prompt(
                    question=question,
                    preceding_steps=steps[:k],
                    current_step=step,
                    step_index=k,
                )
                tasks.append({
                    "doc_id": doc_id,
                    "resp_type": resp_type,
                    "step_idx": k,
                    "prompt": prompt,
                })
    return tasks


def run_verification(
    client: OpenAI,
    model: str,
    tasks: List[Dict],
    cache_path: Path,
    temperature: float = 0.0,
    max_workers: int = 4,
) -> List[Dict]:
    """Run GPT verification for all tasks, with caching and concurrency."""
    results = []

    def _process(task):
        label, raw = call_gpt_verify(
            client, model, task["prompt"], temperature=temperature,
        )
        rec = {
            "doc_id": task["doc_id"],
            "resp_type": task["resp_type"],
            "step_idx": task["step_idx"],
            "label": label,
            "raw": raw,
        }
        return rec

    if max_workers <= 1:
        for task in tqdm(tasks, desc="GPT verify"):
            rec = _process(task)
            results.append(rec)
            append_cache(cache_path, rec)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process, t): t for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="GPT verify"):
                rec = fut.result()
                results.append(rec)
                append_cache(cache_path, rec)

    return results


def main():
    parser = argparse.ArgumentParser(description="GPT step-level PRM validation")
    parser.add_argument(
        "--input", type=str,
        default="artifacts_real/samples_gsm8k_train_ds2_fix_step2_gpt_prefill.json",
        help="Path to input JSON file",
    )
    parser.add_argument("--output-dir", type=str, default="runs/gpt_step_verification")
    parser.add_argument("--model", type=str, default="gpt-5.1")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--cache-file", type=str, default=None,
                        help="JSONL cache path (default: <output-dir>/cache.jsonl)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip API calls, just re-plot from cache")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print first prompt and exit (no API calls)")
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent.parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = project_root / input_path
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = Path(args.cache_file) if args.cache_file else out_dir / "cache.jsonl"
    if not cache_path.is_absolute():
        cache_path = project_root / cache_path

    # Load .env
    load_env_file(project_root / ".env")

    # Load data
    print(f"Loading data from {input_path}")
    data = json.loads(input_path.read_text(encoding="utf-8"))
    samples = data["samples"]
    subset = samples[:args.max_samples] if args.max_samples else samples
    print(f"  Total samples: {len(samples)}, using: {len(subset)}")

    total_steps = sum(
        len(s.get("pos_steps", [])) + len(s.get("neg_steps", []))
        for s in subset
    )
    print(f"  Total steps to verify: {total_steps}")

    # Load cache
    cache = load_cache(cache_path)
    print(f"  Cached judgments: {len(cache)}")

    # Dry-run mode
    if args.dry_run:
        tasks = build_tasks(samples, {}, max_samples=1)
        if tasks:
            print("\n=== DRY RUN: First prompt ===")
            print(tasks[0]["prompt"])
            print(f"\n  Total tasks for 1 sample: {len(tasks)}")
            # Show all tasks for this sample
            for t in tasks:
                print(f"    doc_id={t['doc_id']} {t['resp_type']} step {t['step_idx']}")
        else:
            print("No tasks to process.")
        return

    # Build tasks (skip cached)
    tasks = build_tasks(samples, cache, max_samples=args.max_samples)
    print(f"  New tasks (not cached): {len(tasks)}")

    if tasks and not args.plot_only:
        # Init OpenAI client
        client_kwargs = {}
        if os.environ.get("BASE_URL"):
            client_kwargs["base_url"] = os.environ["BASE_URL"]
        client = OpenAI(**client_kwargs)

        print(f"\nRunning GPT verification with model={args.model}, "
              f"workers={args.max_workers}, temperature={args.temperature}")
        new_results = run_verification(
            client=client,
            model=args.model,
            tasks=tasks,
            cache_path=cache_path,
            temperature=args.temperature,
            max_workers=args.max_workers,
        )
        print(f"  Completed: {len(new_results)} new judgments")

    # Reload full cache for plotting
    cache = load_cache(cache_path)
    all_results = list(cache.values())
    print(f"\nTotal judgments for plotting: {len(all_results)}")

    # Filter to current subset
    subset_ids = {s["doc"]["id"] for s in subset}
    plot_results = [r for r in all_results if r["doc_id"] in subset_ids]

    # Summary stats
    pos_labels = [r["label"] for r in plot_results if r["resp_type"] == "pos" and r["label"] is not None]
    neg_labels = [r["label"] for r in plot_results if r["resp_type"] == "neg" and r["label"] is not None]
    print(f"  pos steps judged: {len(pos_labels)}, correct rate: "
          f"{sum(pos_labels)/len(pos_labels):.3f}" if pos_labels else "  pos: none")
    print(f"  neg steps judged: {len(neg_labels)}, correct rate: "
          f"{sum(neg_labels)/len(neg_labels):.3f}" if neg_labels else "  neg: none")

    # Save full results JSON
    results_path = out_dir / "gpt_step_judgments.json"
    json.dump(plot_results, results_path.open("w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"  Saved results: {results_path}")

    # Plot
    print("\nGenerating plots...")
    plot_step_correctness_pos_vs_neg(plot_results, out_dir)
    plot_step_correctness_by_divergence(plot_results, subset, out_dir)
    plot_agreement_summary(plot_results, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
