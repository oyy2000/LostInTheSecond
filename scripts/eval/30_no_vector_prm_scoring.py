#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRM Step-Level Scoring for runs/no_vector baselines.

Scans runs/no_vector/{task}/{model}_no_vector/ for lm-eval JSONL samples,
scores each generation with Qwen2.5-Math-PRM-7B, and plots per-step PRM
curves (correct vs wrong vs all) for every (task, model) combination.

Usage:
    # Full run: score + plot
    CUDA_VISIBLE_DEVICES=0 python scripts/eval/30_no_vector_prm_scoring.py

    # Plot only from cached scores
    python scripts/eval/30_no_vector_prm_scoring.py --plot-only

    # Limit to specific task
    python scripts/eval/30_no_vector_prm_scoring.py --tasks gsm8k_cot_zeroshot_unified
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Discovery: find all (task, model, jsonl_path) triples
# ---------------------------------------------------------------------------

def discover_runs(no_vector_dir: Path) -> List[Dict]:
    """Find all sample JSONL files under runs/no_vector/."""
    runs = []
    for task_dir in sorted(no_vector_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        for model_dir in sorted(task_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_label = model_dir.name.replace("_no_vector", "")
            # Find sample JSONL files inside nested HF-style dir
            for jsonl in sorted(model_dir.rglob("samples_*.jsonl")):
                # Extract subtask from filename: samples_{subtask}_{timestamp}.jsonl
                fname = jsonl.stem  # e.g. samples_gsm8k_cot_zeroshot_unified_2026-...
                # Remove timestamp suffix
                m = re.match(r"samples_(.+?)_\d{4}-\d{2}-\d{2}T", fname)
                subtask = m.group(1) if m else task_name
                runs.append({
                    "task": task_name,
                    "subtask": subtask,
                    "model": model_label,
                    "jsonl_path": jsonl,
                })
    return runs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

STEP_RE = re.compile(r"(?:^|\n)\s*Step\s*\d+\s*:\s*", re.IGNORECASE)


def split_steps(text: str) -> List[str]:
    """Split generation into reasoning steps (auto mode)."""
    text = (text or "").strip()
    if not text:
        return []
    if "\n\n" in text:
        steps = [x.strip() for x in text.split("\n\n") if x.strip()]
        if len(steps) >= 2:
            return steps
    hits = list(STEP_RE.finditer(text))
    if hits:
        spans = []
        for i, m in enumerate(hits):
            st = m.start()
            ed = hits[i + 1].start() if i + 1 < len(hits) else len(text)
            spans.append(text[st:ed].strip())
        out = [s for s in spans if s]
        if len(out) >= 2:
            return out
    if "\n" in text:
        steps = [x.strip() for x in text.split("\n") if x.strip()]
        if len(steps) >= 2:
            return steps
    return [text]


def load_lm_eval_jsonl(path: Path) -> List[Dict]:
    """Load lm-eval JSONL, extract question, generation, exact_match."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            doc = rec.get("doc", {}) or {}
            question = (doc.get("problem") or doc.get("question") or "").strip()
            if not question:
                continue

            gen = ""
            resps = rec.get("resps", [])
            if resps and isinstance(resps[0], list) and resps[0]:
                gen = (resps[0][0] or "").strip()
            if not gen:
                fr = rec.get("filtered_resps", [])
                if isinstance(fr, list) and fr:
                    if isinstance(fr[0], list) and fr[0]:
                        gen = (fr[0][0] or "").strip()
                    elif isinstance(fr[0], str):
                        gen = fr[0].strip()
            if not gen:
                continue

            em = float(rec.get("exact_match", 0.0))
            steps = split_steps(gen)
            records.append({
                "doc_id": rec.get("doc_id", -1),
                "question": question,
                "generation": gen,
                "steps": steps,
                "exact_match": em,
            })
    return records


# ---------------------------------------------------------------------------
# PRM scoring (single-GPU worker)
# ---------------------------------------------------------------------------

def score_all_records(records: List[Dict], scorer) -> List[Dict]:
    """Score each record's steps with PRM. Returns records with step_scores."""
    for i, rec in enumerate(records):
        scores = scorer.score_steps(rec["question"], rec["steps"])
        rec["step_scores"] = scores
        if (i + 1) % 100 == 0 or (i + 1) == len(records):
            print(f"      [{i+1}/{len(records)}]", flush=True)
    return records


def _gpu_worker(gpu_id: int, prm_model: str, dtype: str,
                task_queue: mp.Queue, result_queue: mp.Queue,
                project_root: str):
    """Worker process: load PRM on one GPU, pull tasks from queue, push results."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    sys.path.insert(0, str(Path(project_root) / "src"))
    from prm.scoring import StepScorer

    print(f"  [GPU {gpu_id}] Loading PRM model...", flush=True)
    scorer = StepScorer(prm_model, dtype)
    print(f"  [GPU {gpu_id}] Model loaded.", flush=True)

    while True:
        item = task_queue.get()
        if item is None:  # poison pill
            break
        ck, jsonl_path = item
        print(f"  [GPU {gpu_id}] Scoring: {ck}", flush=True)
        t0 = time.time()
        records = load_lm_eval_jsonl(Path(jsonl_path))
        records = score_all_records(records, scorer)
        slim = [{
            "doc_id": r["doc_id"],
            "exact_match": r["exact_match"],
            "n_steps": len(r["steps"]),
            "step_scores": r.get("step_scores", []),
        } for r in records]
        elapsed = time.time() - t0
        print(f"  [GPU {gpu_id}] Done: {ck} ({len(records)} records, {elapsed:.1f}s)",
              flush=True)
        result_queue.put((ck, slim))

    print(f"  [GPU {gpu_id}] Worker exiting.", flush=True)


def run_multi_gpu_scoring(
    pending_runs: List[Dict],
    prm_model: str,
    dtype: str,
    n_gpus: int,
    project_root: Path,
    cache_path: Path,
) -> Dict[str, List[Dict]]:
    """Score all pending runs across n_gpus in parallel."""
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    # Enqueue all tasks
    for run in pending_runs:
        ck = cache_key(run["task"], run["subtask"], run["model"])
        task_queue.put((ck, str(run["jsonl_path"])))

    # Add poison pills (one per worker)
    for _ in range(n_gpus):
        task_queue.put(None)

    # Start workers
    workers = []
    for gpu_id in range(n_gpus):
        p = ctx.Process(
            target=_gpu_worker,
            args=(gpu_id, prm_model, dtype, task_queue, result_queue,
                  str(project_root)),
        )
        p.start()
        workers.append(p)

    # Collect results
    new_cache = {}
    for _ in range(len(pending_runs)):
        ck, slim = result_queue.get()
        new_cache[ck] = slim
        # Append to cache file immediately (thread-safe: one writer in main)
        append_score_cache(cache_path, ck, slim)
        print(f"  Cached: {ck} ({len(slim)} records)", flush=True)

    # Wait for workers to finish
    for p in workers:
        p.join()

    return new_cache


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def cache_key(task: str, subtask: str, model: str) -> str:
    return f"{task}|{subtask}|{model}"


def load_score_cache(cache_path: Path) -> Dict[str, List[Dict]]:
    """Load cached PRM scores. Each line: {key, records: [...]}."""
    cache = {}
    if not cache_path.exists():
        return cache
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cache[obj["key"]] = obj["records"]
    return cache


def append_score_cache(cache_path: Path, key: str, records):
    """Append scored records to cache. records can be raw dicts or already slim."""
    slim = []
    for r in records:
        if "steps" in r:
            # Raw record from load_lm_eval_jsonl
            slim.append({
                "doc_id": r["doc_id"],
                "exact_match": r["exact_match"],
                "n_steps": len(r["steps"]),
                "step_scores": r.get("step_scores", []),
            })
        else:
            # Already slim
            slim.append(r)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "records": slim}, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _avg_curve_absolute(records: List[Dict]):
    """Avg PRM score per absolute step index."""
    by_step: Dict[int, List[float]] = {}
    for r in records:
        for k, s in enumerate(r.get("step_scores", [])):
            by_step.setdefault(k, []).append(s)
    xs, ys, ns = [], [], []
    for k in sorted(by_step.keys()):
        vals = by_step[k]
        xs.append(k + 1)
        ys.append(sum(vals) / len(vals))
        ns.append(len(vals))
    return xs, ys, ns


def _avg_curve_normalized(records: List[Dict], n_bins: int = 10):
    """Avg PRM score per normalized step position [0, 1]."""
    positions, values = [], []
    for r in records:
        scores = r.get("step_scores", [])
        n = len(scores)
        if n < 2:
            continue
        for k, s in enumerate(scores):
            positions.append(k / (n - 1))
            values.append(s)
    if not positions:
        return [], [], []
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for p, v in zip(positions, values):
        idx = min(int(p * n_bins), n_bins - 1)
        sums[idx] += v
        counts[idx] += 1
    mask = counts > 0
    means = np.where(counts > 0, sums / counts, np.nan)
    return centers[mask].tolist(), means[mask].tolist(), counts[mask].tolist()


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_per_model_task(records: List[Dict], model: str, subtask: str,
                        out_dir: Path, n_bins: int = 10):
    """Plot PRM step curves for one (model, subtask): correct vs wrong vs all.
    Both absolute and normalized versions."""
    out_dir.mkdir(parents=True, exist_ok=True)

    all_r = records
    correct_r = [r for r in records if r["exact_match"] >= 1.0]
    wrong_r = [r for r in records if r["exact_match"] < 1.0]

    safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", f"{model}_{subtask}")

    # --- Absolute step index ---
    xa, ya, na = _avg_curve_absolute(all_r)
    xc, yc, nc = _avg_curve_absolute(correct_r)
    xw, yw, nw = _avg_curve_absolute(wrong_r)

    if xa:
        plt.figure(figsize=(10, 6))
        plt.plot(xa, ya, marker="o", linewidth=2, label=f"All (n={len(all_r)})")
        plt.plot(xc, yc, marker="s", linewidth=2, label=f"Correct (n={len(correct_r)})")
        plt.plot(xw, yw, marker="x", linewidth=2, label=f"Wrong (n={len(wrong_r)})")
        plt.xlabel("Step index")
        plt.ylabel("Avg PRM score")
        plt.title(f"PRM Step Score — {model} / {subtask}")
        plt.ylim(-0.05, 1.1)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"prm_step_{safe_name}.png", dpi=180)
        plt.close()

    # --- Normalized ---
    xa, ya, na = _avg_curve_normalized(all_r, n_bins)
    xc, yc, nc = _avg_curve_normalized(correct_r, n_bins)
    xw, yw, nw = _avg_curve_normalized(wrong_r, n_bins)

    if xa:
        plt.figure(figsize=(10, 6))
        plt.plot(xa, ya, marker="o", linewidth=2, label=f"All (n={len(all_r)})")
        plt.plot(xc, yc, marker="s", linewidth=2, label=f"Correct (n={len(correct_r)})")
        plt.plot(xw, yw, marker="x", linewidth=2, label=f"Wrong (n={len(wrong_r)})")
        plt.xlabel("Normalized step position (0=start, 1=end)")
        plt.ylabel("Avg PRM score")
        plt.title(f"PRM Step Score (normalized) — {model} / {subtask}")
        plt.ylim(-0.05, 1.1)
        plt.xlim(-0.05, 1.05)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"prm_step_norm_{safe_name}.png", dpi=180)
        plt.close()


def plot_cross_model_overlay(all_data: Dict[str, List[Dict]], subtask: str,
                             out_dir: Path, n_bins: int = 10):
    """Overlay normalized PRM curves for all models on one subtask.
    Two plots: one for correct, one for wrong."""
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_sub = re.sub(r"[^a-zA-Z0-9_\-]", "_", subtask)

    # Sort models by size heuristic
    def _size_key(name):
        m = re.search(r"(\d+\.?\d*)[Bb]", name)
        return float(m.group(1)) if m else 999
    models = sorted(all_data.keys(), key=_size_key)

    for split_name, filter_fn in [
        ("all", lambda r: True),
        ("correct", lambda r: r["exact_match"] >= 1.0),
        ("wrong", lambda r: r["exact_match"] < 1.0),
    ]:
        plt.figure(figsize=(12, 7))
        for model in models:
            recs = [r for r in all_data[model] if filter_fn(r)]
            if not recs:
                continue
            xs, ys, ns = _avg_curve_normalized(recs, n_bins)
            if xs:
                plt.plot(xs, ys, marker="o", linewidth=2, markersize=5,
                         label=f"{model} (n={len(recs)})")

        plt.xlabel("Normalized step position (0=start, 1=end)")
        plt.ylabel("Avg PRM score")
        plt.title(f"PRM Step Score — {subtask} — {split_name} samples")
        plt.ylim(-0.05, 1.1)
        plt.xlim(-0.05, 1.05)
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8, loc="lower left")
        plt.tight_layout()
        plt.savefig(out_dir / f"prm_overlay_{safe_sub}_{split_name}.png", dpi=180)
        plt.close()
        print(f"  Saved: {out_dir / f'prm_overlay_{safe_sub}_{split_name}.png'}")


def plot_cross_model_overlay_absolute(all_data: Dict[str, List[Dict]],
                                      subtask: str, out_dir: Path):
    """Overlay absolute-step PRM curves for all models on one subtask.
    X-axis is capped at the median step count (50th percentile) across
    all records in this subtask so the plot focuses on the dense region."""
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_sub = re.sub(r"[^a-zA-Z0-9_\-]", "_", subtask)

    def _size_key(name):
        m = re.search(r"(\d+\.?\d*)[Bb]", name)
        return float(m.group(1)) if m else 999
    models = sorted(all_data.keys(), key=_size_key)

    # Compute median step count across all records in this subtask
    all_n_steps = []
    for recs in all_data.values():
        for r in recs:
            n = r.get("n_steps", len(r.get("step_scores", [])))
            if n > 0:
                all_n_steps.append(n)
    max_step = int(np.median(all_n_steps)) if all_n_steps else 20

    for split_name, filter_fn in [
        ("all", lambda r: True),
        ("correct", lambda r: r["exact_match"] >= 1.0),
        ("wrong", lambda r: r["exact_match"] < 1.0),
    ]:
        plt.figure(figsize=(12, 7))
        for model in models:
            recs = [r for r in all_data[model] if filter_fn(r)]
            if not recs:
                continue
            xs, ys, ns = _avg_curve_absolute(recs)
            if xs:
                # Trim to max_step
                trimmed = [(x, y) for x, y in zip(xs, ys) if x <= max_step]
                if trimmed:
                    tx, ty = zip(*trimmed)
                    plt.plot(tx, ty, marker="o", linewidth=2, markersize=5,
                             label=f"{model} (n={len(recs)})")

        plt.xlabel("Step index")
        plt.ylabel("Avg PRM score")
        plt.title(f"PRM Step Score (absolute, steps 1-{max_step}) "
                  f"— {subtask} — {split_name} samples")
        plt.ylim(-0.05, 1.1)
        plt.xlim(0.5, max_step + 0.5)
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8, loc="lower left")
        plt.tight_layout()
        fname = f"prm_overlay_abs_{safe_sub}_{split_name}.png"
        plt.savefig(out_dir / fname, dpi=180)
        plt.close()
        print(f"  Saved: {out_dir / fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PRM step-level scoring for runs/no_vector baselines")
    parser.add_argument("--no-vector-dir", type=str, default="runs/no_vector")
    parser.add_argument("--output-dir", type=str, default="runs/no_vector_prm")
    parser.add_argument("--prm-model", type=str, default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--n-gpus", type=int, default=8,
                        help="Number of GPUs for parallel scoring")
    parser.add_argument("--tasks", type=str, nargs="*", default=None,
                        help="Filter to specific task dirs")
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="Filter to specific model names")
    parser.add_argument("--subtasks", type=str, nargs="*", default=None,
                        help="Filter to specific subtask names")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    nv_dir = Path(args.no_vector_dir)
    if not nv_dir.is_absolute():
        nv_dir = project_root / nv_dir
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "prm_score_cache.jsonl"

    # Discover runs
    runs = discover_runs(nv_dir)
    print(f"Discovered {len(runs)} (task, model, subtask) runs")

    # Apply filters
    if args.tasks:
        runs = [r for r in runs if r["task"] in args.tasks]
    if args.models:
        runs = [r for r in runs if r["model"] in args.models]
    if args.subtasks:
        runs = [r for r in runs if r["subtask"] in args.subtasks]
    print(f"After filters: {len(runs)} runs")

    for r in runs:
        print(f"  {r['task']} / {r['subtask']} / {r['model']}")

    # Load cache
    score_cache = load_score_cache(cache_path)
    print(f"Cached entries: {len(score_cache)}")

    # Phase 1: PRM scoring (multi-GPU)
    if not args.plot_only:
        pending = []
        for run in runs:
            ck = cache_key(run["task"], run["subtask"], run["model"])
            if ck not in score_cache:
                pending.append(run)

        if pending:
            n_gpus = min(args.n_gpus, len(pending))
            print(f"\n=== Phase 1: PRM Scoring ({len(pending)} runs on {n_gpus} GPUs) ===")
            new_cache = run_multi_gpu_scoring(
                pending, args.prm_model, args.dtype, n_gpus,
                project_root, cache_path,
            )
            score_cache.update(new_cache)
            print(f"  Scoring complete. Total cached: {len(score_cache)}")
        else:
            print("\n  All runs already cached.")
    else:
        print("\n=== Skipping PRM scoring (--plot-only) ===")

    # Phase 2: Plotting
    print(f"\n=== Phase 2: Plotting ===")

    # Group by subtask for overlay plots
    subtask_data: Dict[str, Dict[str, List[Dict]]] = {}

    for run in runs:
        ck = cache_key(run["task"], run["subtask"], run["model"])
        if ck not in score_cache:
            continue
        records = score_cache[ck]

        # Per-model plot
        model_out = out_dir / run["task"] / run["model"]
        plot_per_model_task(records, run["model"], run["subtask"],
                           model_out, args.n_bins)
        print(f"  Plotted: {run['model']} / {run['subtask']}")

        # Collect for overlay
        subtask_data.setdefault(run["subtask"], {})[run["model"]] = records

    # Cross-model overlay plots (normalized + absolute)
    for subtask, model_data in subtask_data.items():
        overlay_dir = out_dir / "overlay"
        plot_cross_model_overlay(model_data, subtask, overlay_dir, args.n_bins)
        plot_cross_model_overlay_absolute(model_data, subtask, overlay_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
