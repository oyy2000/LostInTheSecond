#!/usr/bin/env python3
"""
Visualize step-start entropy at error vs. non-error positions.

Reads analysis outputs from 10_1 and produces:
  1. Aligned-to-tau entropy curve (mean H_bar by relative position k = t - tau)
  2. Violin/box: error step entropy vs. other steps
  3. Delta histogram (per-trajectory H_tau - mean H_other)
  4. Matched control: wrong error-step vs correct at same relative position

Usage:
    python scripts/10_2_delimiter_entropy_figures.py
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.keystep_utils import load_jsonl

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/delimiter_entropy"))
    ap.add_argument("--fig-dir", default=str(
        PROJECT_ROOT / "figures/delimiter_entropy"))
    return ap.parse_args()


def fig_aligned_curve(summaries, fig_dir):
    """Mean H_bar(K) by relative position k = t - tau, aligned to first error."""
    rel_data = defaultdict(list)
    for s in summaries:
        err_idx = s["tau"] - 1
        for i, ent in enumerate(s["all_entropies"]):
            rel_data[i - err_idx].append(ent)

    positions = sorted(k for k in rel_data if -6 <= k <= 6)
    means = [np.mean(rel_data[p]) for p in positions]
    sems = [np.std(rel_data[p]) / np.sqrt(len(rel_data[p])) for p in positions]
    counts = [len(rel_data[p]) for p in positions]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(positions, means, yerr=sems, fmt="o-",
                capsize=3, color="#4C72B0", markersize=6, linewidth=1.5)
    ax.axvline(0, color="#DD8452", linestyle="--", linewidth=1.5,
               label="First error step (k=0)")

    for p, m, c in zip(positions, means, counts):
        ax.annotate(f"n={c}", (p, m), textcoords="offset points",
                    xytext=(0, 10), fontsize=7, ha="center", color="gray")

    ax.set_xlabel("Relative Step Position (k = t - tau)")
    ax.set_ylabel("Mean Step-Start Entropy H_bar(K)")
    ax.set_title("Step-Start Entropy Aligned to First Error")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = fig_dir / "fig_entropy_aligned_to_tau.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def fig_violin(per_step_wrong, fig_dir):
    """Violin + box: entropy at error step vs. other steps."""
    err = [r["mean_entropy_K"] for r in per_step_wrong if r["is_error_step"]]
    other = [r["mean_entropy_K"] for r in per_step_wrong if not r["is_error_step"]]

    fig, ax = plt.subplots(figsize=(5, 5))
    parts = ax.violinplot([other, err], positions=[0, 1], showmedians=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    parts["bodies"][0].set_facecolor("#4C72B0")
    parts["bodies"][1].set_facecolor("#DD8452")

    ax.boxplot([other, err], positions=[0, 1], widths=0.15,
               showfliers=False, zorder=3,
               medianprops=dict(color="black", linewidth=1.5))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Other Steps", "Error Step (tau)"])
    ax.set_ylabel("Step-Start Entropy H_bar(K)")
    ax.set_title("Step-Start Entropy: Error vs. Non-Error Steps")

    ax.text(0.02, 0.98,
            f"n(error)={len(err)}, n(other)={len(other)}\n"
            f"mean: {np.mean(err):.4f} vs {np.mean(other):.4f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.tight_layout()
    out = fig_dir / "fig_step_entropy_violin.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def fig_delta_hist(summaries, fig_dir):
    """Histogram of per-trajectory delta entropy (H_tau - mean H_other)."""
    deltas = [s["delta_entropy"] for s in summaries]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(deltas, bins=50, color="#4C72B0", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(np.mean(deltas), color="#DD8452", linestyle="-", linewidth=1.5,
               label=f"mean = {np.mean(deltas):.4f}")

    frac_pos = np.mean([d > 0 for d in deltas])
    ax.set_xlabel("Delta Entropy (H_tau - mean H_other)")
    ax.set_ylabel("Count")
    ax.set_title("Within-Trajectory: Error Step vs. Other Steps")
    ax.legend(fontsize=9)
    ax.text(0.98, 0.95,
            f"n={len(deltas)}\n{frac_pos:.1%} positive",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.tight_layout()
    out = fig_dir / "fig_delta_entropy_hist.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def fig_matched_control(per_step_wrong, per_step_correct, fig_dir):
    """Compare wrong error-step entropy vs correct at same absolute step index."""
    wrong_by_step = defaultdict(list)
    for r in per_step_wrong:
        if r["is_error_step"]:
            wrong_by_step[r["step_idx"]].append(r["mean_entropy_K"])

    correct_by_step = defaultdict(list)
    for r in per_step_correct:
        correct_by_step[r["step_idx"]].append(r["mean_entropy_K"])

    common_steps = sorted(set(wrong_by_step) & set(correct_by_step))
    if not common_steps:
        print("  No overlapping step indices for matched control.")
        return

    wrong_vals, correct_vals = [], []
    step_labels = []
    for si in common_steps:
        if len(wrong_by_step[si]) >= 5 and len(correct_by_step[si]) >= 5:
            wrong_vals.append(np.mean(wrong_by_step[si]))
            correct_vals.append(np.mean(correct_by_step[si]))
            step_labels.append(si)

    if len(step_labels) < 2:
        print("  Too few step indices with enough data for matched control.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(step_labels))
    w = 0.35
    ax.bar(x - w/2, wrong_vals, w, label="Wrong (error step)", color="#DD8452", alpha=0.8)
    ax.bar(x + w/2, correct_vals, w, label="Correct (same step idx)", color="#4C72B0", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Step {s+1}" for s in step_labels])
    ax.set_ylabel("Mean Step-Start Entropy H_bar(K)")
    ax.set_title("Matched Control: Wrong Error-Step vs. Correct at Same Position")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = fig_dir / "fig_matched_control.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    per_step_wrong = load_jsonl(data_dir / "per_step_wrong.jsonl")
    summaries = load_jsonl(data_dir / "trajectory_summaries.jsonl")
    per_step_correct = load_jsonl(data_dir / "per_step_correct.jsonl") \
        if (data_dir / "per_step_correct.jsonl").exists() else []

    if not per_step_wrong or not summaries:
        print("Missing data files. Run 10_1 first.")
        return

    print(f"Loaded {len(per_step_wrong)} wrong step rows, "
          f"{len(summaries)} trajectory summaries, "
          f"{len(per_step_correct)} correct step rows")

    fig_aligned_curve(summaries, fig_dir)
    fig_violin(per_step_wrong, fig_dir)
    fig_delta_hist(summaries, fig_dir)

    if per_step_correct:
        fig_matched_control(per_step_wrong, per_step_correct, fig_dir)

    stats_path = data_dir / "stats_summary.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text("utf-8"))
        print(f"\nKey results:")
        print(f"  Error step H_bar(K):   {stats.get('mean_error_entropy', '?')}")
        print(f"  Other steps H_bar(K):  {stats.get('mean_other_entropy', '?')}")
        print(f"  Wilcoxon p:            {stats.get('wilcoxon_p', '?')}")
        print(f"  Frac error > other:    {stats.get('frac_error_higher', '?')}")

    print(f"\nFigures saved to {fig_dir}")


if __name__ == "__main__":
    main()
