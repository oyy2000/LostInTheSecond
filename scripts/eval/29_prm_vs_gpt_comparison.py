#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRM vs GPT Step-Level Comparison.

Phase 1: Score every step of pos/neg responses with Qwen2.5-Math-PRM-7B.
Phase 2: Plot PRM step-score curves (pos vs neg).
Phase 3: Load GPT judgments from 28_gpt_step_verification cache, overlay
         PRM scores vs GPT correctness on the same plot.

Usage:
    # Full run (PRM scoring + comparison plots)
    python scripts/eval/29_prm_vs_gpt_comparison.py

    # Plot only (skip PRM scoring, use cached results)
    python scripts/eval/29_prm_vs_gpt_comparison.py --plot-only

    # PRM scoring only (no comparison with GPT)
    python scripts/eval/29_prm_vs_gpt_comparison.py --prm-only
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# PRM scoring (lazy import torch to allow --plot-only without GPU)
# ---------------------------------------------------------------------------

def run_prm_scoring(
    samples: List[Dict],
    prm_model_id: str,
    dtype: str,
    cache_path: Path,
    max_samples: Optional[int] = None,
) -> Dict[str, Dict]:
    """Score all pos/neg steps with PRM. Returns cache dict."""
    import torch  # noqa: lazy
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from prm.scoring import StepScorer, split_steps  # noqa

    # Load existing cache
    cache = load_prm_cache(cache_path)
    subset = samples[:max_samples] if max_samples else samples

    # Figure out what's missing
    tasks = []
    for sample in subset:
        doc_id = sample["doc"]["id"]
        question = sample["doc"]["question"]
        for resp_type in ("pos", "neg"):
            ck = f"{doc_id}|{resp_type}"
            if ck in cache:
                continue
            steps = sample.get(f"{resp_type}_steps", [])
            if not steps:
                continue
            tasks.append({
                "doc_id": doc_id,
                "question": question,
                "resp_type": resp_type,
                "steps": steps,
            })

    if not tasks:
        print(f"  All {len(subset)} samples already cached.")
        return cache

    print(f"  Loading PRM model: {prm_model_id} (dtype={dtype})")
    scorer = StepScorer(prm_model_id, dtype)
    print(f"  Scoring {len(tasks)} response chains...")

    for i, task in enumerate(tasks):
        scores = scorer.score_steps(task["question"], task["steps"])
        rec = {
            "doc_id": task["doc_id"],
            "resp_type": task["resp_type"],
            "step_scores": scores,
            "n_steps": len(task["steps"]),
        }
        ck = f"{task['doc_id']}|{task['resp_type']}"
        cache[ck] = rec
        append_prm_cache(cache_path, rec)

        if (i + 1) % 20 == 0 or (i + 1) == len(tasks):
            print(f"    [{i+1}/{len(tasks)}] scored doc_id={task['doc_id']} "
                  f"{task['resp_type']}: {len(scores)} steps")

    print(f"  PRM scoring done. Total cached: {len(cache)}")
    return cache


# ---------------------------------------------------------------------------
# PRM cache helpers
# ---------------------------------------------------------------------------

def load_prm_cache(cache_path: Path) -> Dict[str, Dict]:
    cache: Dict[str, Dict] = {}
    if not cache_path.exists():
        return cache
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = f"{rec['doc_id']}|{rec['resp_type']}"
            cache[key] = rec
    return cache


def append_prm_cache(cache_path: Path, rec: Dict):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# GPT cache loader (from script 28)
# ---------------------------------------------------------------------------

def load_gpt_cache(cache_path: Path) -> Dict[str, Dict]:
    """Load GPT step-level judgments from 28_gpt_step_verification cache."""
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


# ---------------------------------------------------------------------------
# Curve helpers
# ---------------------------------------------------------------------------

def prm_avg_curve(prm_cache: Dict[str, Dict], resp_type: str):
    """Compute avg PRM score per step index for a given resp_type."""
    by_step: Dict[int, List[float]] = {}
    for rec in prm_cache.values():
        if rec["resp_type"] != resp_type:
            continue
        for k, score in enumerate(rec["step_scores"]):
            by_step.setdefault(k, []).append(score)

    xs, ys, ns = [], [], []
    for k in sorted(by_step.keys()):
        vals = by_step[k]
        xs.append(k + 1)
        ys.append(sum(vals) / len(vals))
        ns.append(len(vals))
    return xs, ys, ns


def gpt_avg_curve(gpt_cache: Dict[str, Dict], resp_type: str):
    """Compute avg GPT correctness per step index for a given resp_type."""
    by_step: Dict[int, List[int]] = {}
    for rec in gpt_cache.values():
        if rec["resp_type"] != resp_type:
            continue
        if rec.get("label") is None:
            continue
        by_step.setdefault(rec["step_idx"], []).append(rec["label"])

    xs, ys, ns = [], [], []
    for k in sorted(by_step.keys()):
        vals = by_step[k]
        xs.append(k + 1)
        ys.append(sum(vals) / len(vals))
        ns.append(len(vals))
    return xs, ys, ns


# ---------------------------------------------------------------------------
# Normalized curve helpers (step position -> [0, 1])
# ---------------------------------------------------------------------------

import numpy as np


def _bin_values(positions: List[float], values: List[float], n_bins: int = 10):
    """Bin (position, value) pairs into n_bins equal-width bins over [0, 1].
    Returns bin_centers, bin_means, bin_counts."""
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for p, v in zip(positions, values):
        idx = min(int(p * n_bins), n_bins - 1)
        sums[idx] += v
        counts[idx] += 1
    means = np.where(counts > 0, sums / counts, np.nan)
    return centers, means, counts


def prm_normalized_curve(prm_cache: Dict[str, Dict], resp_type: str, n_bins: int = 10):
    """PRM scores with step index normalized to [0, 1] relative progress."""
    positions, values = [], []
    for rec in prm_cache.values():
        if rec["resp_type"] != resp_type:
            continue
        n = len(rec["step_scores"])
        if n < 2:
            continue
        for k, score in enumerate(rec["step_scores"]):
            positions.append(k / (n - 1))
            values.append(score)
    if not positions:
        return [], [], []
    centers, means, counts = _bin_values(positions, values, n_bins)
    mask = counts > 0
    return centers[mask].tolist(), means[mask].tolist(), counts[mask].tolist()


def gpt_normalized_curve(gpt_cache: Dict[str, Dict], prm_cache: Dict[str, Dict],
                         resp_type: str, n_bins: int = 10):
    """GPT correctness with step index normalized to [0, 1] relative progress.
    Uses prm_cache to look up chain length per (doc_id, resp_type)."""
    # Build chain-length lookup from prm_cache
    chain_len = {}
    for rec in prm_cache.values():
        chain_len[(rec["doc_id"], rec["resp_type"])] = rec["n_steps"]

    positions, values = [], []
    for rec in gpt_cache.values():
        if rec["resp_type"] != resp_type:
            continue
        if rec.get("label") is None:
            continue
        n = chain_len.get((rec["doc_id"], rec["resp_type"]))
        if n is None or n < 2:
            continue
        positions.append(rec["step_idx"] / (n - 1))
        values.append(rec["label"])
    if not positions:
        return [], [], []
    centers, means, counts = _bin_values(positions, values, n_bins)
    mask = counts > 0
    return centers[mask].tolist(), means[mask].tolist(), counts[mask].tolist()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_prm_pos_vs_neg(prm_cache: Dict, out_dir: Path):
    """Plot PRM step scores: pos vs neg."""
    out_dir.mkdir(parents=True, exist_ok=True)
    px, py, pn = prm_avg_curve(prm_cache, "pos")
    nx, ny, nn = prm_avg_curve(prm_cache, "neg")

    plt.figure(figsize=(10, 6))
    if px:
        plt.plot(px, py, marker="o", linewidth=2, color="tab:blue",
                 label="pos (corrected)")
    if nx:
        plt.plot(nx, ny, marker="x", linewidth=2, color="tab:red",
                 label="neg (original)")

    for x, y, n in zip(px, py, pn):
        plt.text(x, y + 0.005, f"n={n}", fontsize=7, ha="center",
                 va="bottom", color="tab:blue", alpha=0.7)
    for x, y, n in zip(nx, ny, nn):
        plt.text(x, y - 0.015, f"n={n}", fontsize=7, ha="center",
                 va="top", color="tab:red", alpha=0.7)

    plt.xlabel("Step index")
    plt.ylabel("Avg PRM score")
    plt.title("PRM Step-Level Score: pos vs neg")
    plt.ylim(-0.05, 1.1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "prm_step_score_pos_vs_neg.png", dpi=180)
    plt.close()
    print(f"  Saved: {out_dir / 'prm_step_score_pos_vs_neg.png'}")


def plot_comparison(prm_cache: Dict, gpt_cache: Dict, out_dir: Path):
    """Overlay PRM scores and GPT correctness on the same plot (dual y-axis)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for resp_type, color, label_suffix in [
        ("pos", "tab:blue", "pos (corrected)"),
        ("neg", "tab:red", "neg (original)"),
    ]:
        prm_x, prm_y, prm_n = prm_avg_curve(prm_cache, resp_type)
        gpt_x, gpt_y, gpt_n = gpt_avg_curve(gpt_cache, resp_type)

        if not prm_x and not gpt_x:
            continue

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # PRM on left y-axis
        if prm_x:
            ax1.plot(prm_x, prm_y, marker="o", linewidth=2, color=color,
                     label=f"PRM score ({label_suffix})")
        ax1.set_xlabel("Step index")
        ax1.set_ylabel("Avg PRM score", color=color)
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_ylim(-0.05, 1.1)
        ax1.grid(alpha=0.2)

        # GPT on right y-axis
        ax2 = ax1.twinx()
        gpt_color = "tab:green" if resp_type == "pos" else "tab:orange"
        if gpt_x:
            ax2.plot(gpt_x, gpt_y, marker="s", linewidth=2, linestyle="--",
                     color=gpt_color, label=f"GPT correctness ({label_suffix})")
        ax2.set_ylabel("Avg GPT correctness", color=gpt_color)
        ax2.tick_params(axis="y", labelcolor=gpt_color)
        ax2.set_ylim(-0.05, 1.1)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

        plt.title(f"PRM vs GPT: {label_suffix}")
        fig.tight_layout()
        fig.savefig(out_dir / f"prm_vs_gpt_{resp_type}.png", dpi=180)
        plt.close(fig)
        print(f"  Saved: {out_dir / f'prm_vs_gpt_{resp_type}.png'}")

    # --- Combined 4-curve plot (single y-axis, both on 0-1 scale) ---
    prm_px, prm_py, _ = prm_avg_curve(prm_cache, "pos")
    prm_nx, prm_ny, _ = prm_avg_curve(prm_cache, "neg")
    gpt_px, gpt_py, _ = gpt_avg_curve(gpt_cache, "pos")
    gpt_nx, gpt_ny, _ = gpt_avg_curve(gpt_cache, "neg")

    plt.figure(figsize=(11, 6.5))
    if prm_px:
        plt.plot(prm_px, prm_py, marker="o", linewidth=2, color="tab:blue",
                 label="PRM — pos (corrected)")
    if prm_nx:
        plt.plot(prm_nx, prm_ny, marker="o", linewidth=2, color="tab:red",
                 label="PRM — neg (original)")
    if gpt_px:
        plt.plot(gpt_px, gpt_py, marker="s", linewidth=2, linestyle="--",
                 color="tab:cyan", label="GPT — pos (corrected)")
    if gpt_nx:
        plt.plot(gpt_nx, gpt_ny, marker="s", linewidth=2, linestyle="--",
                 color="tab:orange", label="GPT — neg (original)")

    plt.axvline(x=2, linestyle=":", color="gray", alpha=0.5, label="divergence (step 2)")
    plt.xlabel("Step index")
    plt.ylabel("Score / Correctness (0–1)")
    plt.title("PRM Score vs GPT Correctness — Step-Level Comparison")
    plt.ylim(-0.05, 1.1)
    plt.grid(alpha=0.25)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_dir / "prm_vs_gpt_combined.png", dpi=180)
    plt.close()
    print(f"  Saved: {out_dir / 'prm_vs_gpt_combined.png'}")


def plot_normalized_comparison(prm_cache: Dict, gpt_cache: Dict, out_dir: Path,
                               n_bins: int = 10):
    """Normalized step-position plots: all samples contribute to every bin."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Per resp_type: PRM vs GPT (dual y-axis, normalized x) ---
    for resp_type, color, label_suffix in [
        ("pos", "tab:blue", "pos (corrected)"),
        ("neg", "tab:red", "neg (original)"),
    ]:
        prm_x, prm_y, prm_n = prm_normalized_curve(prm_cache, resp_type, n_bins)
        gpt_x, gpt_y, gpt_n = gpt_normalized_curve(gpt_cache, prm_cache, resp_type, n_bins)

        if not prm_x and not gpt_x:
            continue

        fig, ax1 = plt.subplots(figsize=(10, 6))
        if prm_x:
            ax1.plot(prm_x, prm_y, marker="o", linewidth=2, color=color,
                     label=f"PRM score ({label_suffix})")
            for x, y, n in zip(prm_x, prm_y, prm_n):
                ax1.text(x, y + 0.008, f"n={n}", fontsize=6, ha="center",
                         va="bottom", color=color, alpha=0.6)
        ax1.set_xlabel("Normalized step position (0=start, 1=end)")
        ax1.set_ylabel("Avg PRM score", color=color)
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_ylim(-0.05, 1.1)
        ax1.set_xlim(-0.05, 1.05)
        ax1.grid(alpha=0.2)

        ax2 = ax1.twinx()
        gpt_color = "tab:green" if resp_type == "pos" else "tab:orange"
        if gpt_x:
            ax2.plot(gpt_x, gpt_y, marker="s", linewidth=2, linestyle="--",
                     color=gpt_color, label=f"GPT correctness ({label_suffix})")
        ax2.set_ylabel("Avg GPT correctness", color=gpt_color)
        ax2.tick_params(axis="y", labelcolor=gpt_color)
        ax2.set_ylim(-0.05, 1.1)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

        plt.title(f"PRM vs GPT (normalized): {label_suffix}")
        fig.tight_layout()
        fig.savefig(out_dir / f"prm_vs_gpt_{resp_type}_normalized.png", dpi=180)
        plt.close(fig)
        print(f"  Saved: {out_dir / f'prm_vs_gpt_{resp_type}_normalized.png'}")

    # --- Combined 4-curve normalized plot ---
    prm_px, prm_py, _ = prm_normalized_curve(prm_cache, "pos", n_bins)
    prm_nx, prm_ny, _ = prm_normalized_curve(prm_cache, "neg", n_bins)
    gpt_px, gpt_py, _ = gpt_normalized_curve(gpt_cache, prm_cache, "pos", n_bins)
    gpt_nx, gpt_ny, _ = gpt_normalized_curve(gpt_cache, prm_cache, "neg", n_bins)

    plt.figure(figsize=(11, 6.5))
    if prm_px:
        plt.plot(prm_px, prm_py, marker="o", linewidth=2, color="tab:blue",
                 label="PRM — pos (corrected)")
    if prm_nx:
        plt.plot(prm_nx, prm_ny, marker="o", linewidth=2, color="tab:red",
                 label="PRM — neg (original)")
    if gpt_px:
        plt.plot(gpt_px, gpt_py, marker="s", linewidth=2, linestyle="--",
                 color="tab:cyan", label="GPT — pos (corrected)")
    if gpt_nx:
        plt.plot(gpt_nx, gpt_ny, marker="s", linewidth=2, linestyle="--",
                 color="tab:orange", label="GPT — neg (original)")

    # Mark ~20% position (approx step 2 in a typical 8-10 step chain)
    plt.axvline(x=0.15, linestyle=":", color="gray", alpha=0.5,
                label="~divergence (step 2 ≈ 15%)")
    plt.xlabel("Normalized step position (0=start, 1=end)")
    plt.ylabel("Score / Correctness (0–1)")
    plt.title("PRM vs GPT — Normalized Step-Level Comparison")
    plt.ylim(-0.05, 1.1)
    plt.xlim(-0.05, 1.05)
    plt.grid(alpha=0.25)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_dir / "prm_vs_gpt_combined_normalized.png", dpi=180)
    plt.close()
    print(f"  Saved: {out_dir / 'prm_vs_gpt_combined_normalized.png'}")


def plot_agreement_scatter(prm_cache: Dict, gpt_cache: Dict, out_dir: Path):
    """Scatter: PRM score vs GPT label (0/1) per step, with jitter."""
    out_dir.mkdir(parents=True, exist_ok=True)
    import random
    random.seed(42)

    prm_scores = []
    gpt_labels = []
    resp_types = []

    for rec in prm_cache.values():
        doc_id = rec["doc_id"]
        rt = rec["resp_type"]
        for k, score in enumerate(rec["step_scores"]):
            gk = f"{doc_id}|{rt}|{k}"
            if gk in gpt_cache and gpt_cache[gk].get("label") is not None:
                prm_scores.append(score)
                gpt_labels.append(gpt_cache[gk]["label"])
                resp_types.append(rt)

    if not prm_scores:
        print("  No overlapping PRM+GPT data for scatter.")
        return

    # Jitter y for visibility
    jittered_y = [l + random.uniform(-0.05, 0.05) for l in gpt_labels]
    colors = ["tab:blue" if rt == "pos" else "tab:red" for rt in resp_types]

    plt.figure(figsize=(8, 5))
    plt.scatter(prm_scores, jittered_y, c=colors, alpha=0.35, s=15, edgecolors="none")

    # Add summary stats
    correct_prm = [s for s, l in zip(prm_scores, gpt_labels) if l == 1]
    wrong_prm = [s for s, l in zip(prm_scores, gpt_labels) if l == 0]
    if correct_prm:
        plt.axvline(x=sum(correct_prm)/len(correct_prm), color="green",
                     linestyle="--", alpha=0.7, label=f"GPT=correct mean PRM={sum(correct_prm)/len(correct_prm):.3f}")
    if wrong_prm:
        plt.axvline(x=sum(wrong_prm)/len(wrong_prm), color="red",
                     linestyle="--", alpha=0.7, label=f"GPT=incorrect mean PRM={sum(wrong_prm)/len(wrong_prm):.3f}")

    plt.xlabel("PRM step score")
    plt.ylabel("GPT label (0=incorrect, 1=correct, jittered)")
    plt.title("PRM Score vs GPT Judgment (per step)")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "prm_vs_gpt_scatter.png", dpi=180)
    plt.close()
    print(f"  Saved: {out_dir / 'prm_vs_gpt_scatter.png'}")

    # Print agreement stats
    n = len(prm_scores)
    # Binarize PRM at 0.5 threshold
    for thr in [0.5, 0.6, 0.7, 0.8]:
        prm_binary = [1 if s >= thr else 0 for s in prm_scores]
        agree = sum(1 for a, b in zip(prm_binary, gpt_labels) if a == b)
        print(f"  PRM thr={thr:.1f}: agreement={agree}/{n} ({agree/n:.3f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PRM vs GPT step-level comparison")
    parser.add_argument(
        "--input", type=str,
        default="artifacts_real/samples_gsm8k_train_ds2_fix_step2_gpt_prefill.json",
    )
    parser.add_argument("--output-dir", type=str, default="runs/prm_vs_gpt_comparison")
    parser.add_argument("--prm-model", type=str, default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--prm-cache", type=str, default=None)
    parser.add_argument("--gpt-cache", type=str, default=None,
                        help="GPT cache from script 28 (default: runs/gpt_step_verification/cache.jsonl)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip PRM scoring, just plot from caches")
    parser.add_argument("--prm-only", action="store_true",
                        help="Only run PRM scoring + PRM plots, no GPT comparison")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = project_root / input_path
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    prm_cache_path = Path(args.prm_cache) if args.prm_cache else out_dir / "prm_cache.jsonl"
    if not prm_cache_path.is_absolute():
        prm_cache_path = project_root / prm_cache_path

    gpt_cache_path = (
        Path(args.gpt_cache) if args.gpt_cache
        else project_root / "runs" / "gpt_step_verification" / "cache.jsonl"
    )
    if not gpt_cache_path.is_absolute():
        gpt_cache_path = project_root / gpt_cache_path

    # Load data
    print(f"Loading data from {input_path}")
    data = json.loads(input_path.read_text(encoding="utf-8"))
    samples = data["samples"]
    subset = samples[:args.max_samples] if args.max_samples else samples
    print(f"  Total samples: {len(samples)}, using: {len(subset)}")

    # Phase 1: PRM scoring
    if not args.plot_only:
        print("\n=== Phase 1: PRM Scoring ===")
        prm_cache = run_prm_scoring(
            subset, args.prm_model, args.dtype, prm_cache_path, args.max_samples,
        )
    else:
        print("\n=== Skipping PRM scoring (--plot-only) ===")
        prm_cache = load_prm_cache(prm_cache_path)
        print(f"  Loaded {len(prm_cache)} PRM cache entries")

    # Phase 2: PRM plots
    print("\n=== Phase 2: PRM Plots ===")
    plot_prm_pos_vs_neg(prm_cache, out_dir)

    # Phase 3: Comparison with GPT
    if not args.prm_only:
        print("\n=== Phase 3: PRM vs GPT Comparison ===")
        if gpt_cache_path.exists():
            gpt_cache = load_gpt_cache(gpt_cache_path)
            print(f"  Loaded {len(gpt_cache)} GPT cache entries")
            plot_comparison(prm_cache, gpt_cache, out_dir)
            plot_normalized_comparison(prm_cache, gpt_cache, out_dir)
            plot_agreement_scatter(prm_cache, gpt_cache, out_dir)
        else:
            print(f"  GPT cache not found at {gpt_cache_path}")
            print("  Run scripts/eval/28_gpt_step_verification.py first.")

    print("\nDone.")


if __name__ == "__main__":
    main()
