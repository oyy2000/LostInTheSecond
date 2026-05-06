#!/usr/bin/env python3
"""
Step 5 figures: Visualize entropy-error localization results.

Generates:
  1. Bar chart: localization accuracy by metric (exact, +/-1, +/-2)
  2. Entropy profile aligned to tau: mean entropy at relative position
  3. Scatter: predicted vs gold error step with Spearman rho
  4. Box plot: error-step vs non-error-step metric distributions

Usage:
    python scripts/5_0_localization_figures.py
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.keystep_utils import load_jsonl

DEFAULT_DATA_DIR = str(
    PROJECT_ROOT / "results/gsm8k_3b_multi_sample/step_uncertainty"
)
DEFAULT_FIG_DIR = str(
    PROJECT_ROOT / "figures/step_uncertainty"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--fig-dir", default=DEFAULT_FIG_DIR)
    return ap.parse_args()


METRIC_LABELS = {
    "mean_entropy": "Mean Entropy",
    "max_entropy": "Max Entropy",
    "entropy_delta": "Entropy Delta",
    "mean_logprob": "Mean LogProb",
    "min_logprob": "Min LogProb",
    "logprob_drop": "LogProb Drop",
}

METRIC_HIGHER_WORSE = {
    "mean_entropy": True,
    "max_entropy": True,
    "entropy_delta": True,
    "mean_logprob": False,
    "min_logprob": False,
    "logprob_drop": False,
}


def fig_localization_bar(results: dict, fig_dir: Path) -> None:
    metrics = list(results.keys())
    labels = [METRIC_LABELS.get(m, m) for m in metrics]
    exact = [results[m]["exact_match"] for m in metrics]
    w1 = [results[m]["within_1"] for m in metrics]
    w2 = [results[m]["within_2"] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, exact, width, label="Exact Match", color="#2196F3")
    ax.bar(x, w1, width, label="Within +/-1", color="#4CAF50")
    ax.bar(x + width, w2, width, label="Within +/-2", color="#FF9800")

    ax.set_ylabel("Accuracy")
    ax.set_title("Error Step Localization by Uncertainty Metric (GSM8K)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    for fmt in ("png", "pdf"):
        fig.savefig(fig_dir / f"fig_localization_bar.{fmt}", dpi=200)
    plt.close(fig)
    print(f"  -> fig_localization_bar")


def fig_entropy_profile(per_step: list, fig_dir: Path) -> None:
    rel_to_tau = {}
    for r in per_step:
        rt = r.get("relative_to_tau")
        if rt is None:
            continue
        if -5 <= rt <= 5:
            rel_to_tau.setdefault(rt, []).append(r["mean_entropy"])

    if not rel_to_tau:
        print("  (skipping entropy profile: no data)")
        return

    positions = sorted(rel_to_tau.keys())
    means = [np.mean(rel_to_tau[p]) for p in positions]
    sems = [np.std(rel_to_tau[p]) / np.sqrt(len(rel_to_tau[p]))
            for p in positions]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(positions, means, yerr=sems, marker="o", capsize=3,
                color="#2196F3", linewidth=2, markersize=6)
    ax.axvline(0, color="red", linestyle="--", alpha=0.7, label="Error step (tau)")
    ax.set_xlabel("Step position relative to first error (tau)")
    ax.set_ylabel("Mean token entropy")
    ax.set_title("Entropy Profile Aligned to First Error Step")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    for fmt in ("png", "pdf"):
        fig.savefig(fig_dir / f"fig_entropy_profile.{fmt}", dpi=200)
    plt.close(fig)
    print(f"  -> fig_entropy_profile")


def fig_pred_vs_gold(per_step: list, results: dict, fig_dir: Path) -> None:
    best_metric = max(results.keys(), key=lambda m: results[m]["spearman_rho"])
    higher_worse = METRIC_HIGHER_WORSE[best_metric]

    traj_data = {}
    for r in per_step:
        key = (r["doc_id"], r["sample_idx"])
        traj_data.setdefault(key, []).append(r)

    golds, preds = [], []
    for key, steps in traj_data.items():
        steps_sorted = sorted(steps, key=lambda s: s["step_idx"])
        tau = steps_sorted[0]["tau"]
        if tau is None or tau < 1:
            continue
        vals = [s[best_metric] for s in steps_sorted]
        if higher_worse:
            pred = int(np.argmax(vals)) + 1
        else:
            pred = int(np.argmin(vals)) + 1
        golds.append(tau)
        preds.append(pred)

    if not golds:
        return

    rho, _ = sp_stats.spearmanr(preds, golds)

    fig, ax = plt.subplots(figsize=(6, 6))
    jitter_g = np.array(golds) + np.random.normal(0, 0.15, len(golds))
    jitter_p = np.array(preds) + np.random.normal(0, 0.15, len(preds))
    ax.scatter(jitter_g, jitter_p, alpha=0.3, s=15, color="#2196F3")
    lim = max(max(golds), max(preds)) + 1
    ax.plot([0, lim], [0, lim], "r--", alpha=0.5, label="Perfect prediction")
    ax.set_xlabel("Gold error step (tau)")
    ax.set_ylabel(f"Predicted error step ({METRIC_LABELS[best_metric]})")
    ax.set_title(f"Predicted vs Gold Error Step (rho={rho:.3f})")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    for fmt in ("png", "pdf"):
        fig.savefig(fig_dir / f"fig_pred_vs_gold.{fmt}", dpi=200)
    plt.close(fig)
    print(f"  -> fig_pred_vs_gold (metric={best_metric})")


def fig_error_vs_nonerror_box(per_step: list, fig_dir: Path) -> None:
    metrics_to_plot = ["mean_entropy", "max_entropy", "mean_logprob", "min_logprob"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(14, 5))

    for ax, metric in zip(axes, metrics_to_plot):
        err_vals = [r[metric] for r in per_step if r["is_error_step"]]
        non_vals = [r[metric] for r in per_step if not r["is_error_step"]]

        if not err_vals or not non_vals:
            continue

        bp = ax.boxplot(
            [err_vals, non_vals],
            tick_labels=["Error step", "Other steps"],
            patch_artist=True,
            widths=0.6,
        )
        bp["boxes"][0].set_facecolor("#EF5350")
        bp["boxes"][1].set_facecolor("#66BB6A")
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.grid(axis="y", alpha=0.3)

        mwu = sp_stats.mannwhitneyu(err_vals, non_vals, alternative="two-sided")
        sig = "***" if mwu.pvalue < 0.001 else "**" if mwu.pvalue < 0.01 else "*" if mwu.pvalue < 0.05 else "ns"
        ax.set_xlabel(f"p={mwu.pvalue:.2e} ({sig})")

    fig.suptitle("Error Step vs Non-Error Step Distributions (GSM8K)", y=1.02)
    fig.tight_layout()

    for fmt in ("png", "pdf"):
        fig.savefig(fig_dir / f"fig_error_vs_nonerror_box.{fmt}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> fig_error_vs_nonerror_box")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    results_path = data_dir / "localization_results.json"
    per_step_path = data_dir / "per_step_metrics.jsonl"

    if not results_path.exists():
        print(f"No results at {results_path}. Run 4_1 first.")
        return

    results = json.loads(results_path.read_text("utf-8"))
    per_step = load_jsonl(per_step_path)
    print(f"Loaded {len(per_step)} per-step rows")

    fig_localization_bar(results, fig_dir)
    fig_entropy_profile(per_step, fig_dir)
    fig_pred_vs_gold(per_step, results, fig_dir)
    fig_error_vs_nonerror_box(per_step, fig_dir)

    print(f"\nAll figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
