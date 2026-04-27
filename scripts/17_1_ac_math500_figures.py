#!/usr/bin/env python3
"""
Plot Adaptive Consistency evaluation results on MATH500.

Reads results from 17_0 and produces:
  1. Accuracy vs Avg Samples (efficiency frontier)
  2. Accuracy vs Avg Tokens (cost frontier)

Usage:
    python scripts/17_1_ac_math500_figures.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "math500_ac_evaluation" / "ac_eval_results.json"
FIG_DIR = ROOT / "figures" / "math500_ac"


def load_results():
    return json.loads(RESULTS.read_text("utf-8"))


def categorize(results):
    greedy = [r for r in results if r["method"] == "Greedy"]
    fullsc = [r for r in results if r["method"].startswith("FullSC@")]
    ac_fullsc = [r for r in results if "AC(" in r["method"] and "LR" not in r["method"]]
    lr_fixed = [r for r in results if r["method"].startswith("LR(")]
    ac_lr = [r for r in results if "AC(" in r["method"] and "LR" in r["method"]]
    return greedy, fullsc, ac_fullsc, lr_fixed, ac_lr


def plot_accuracy_vs_samples(results):
    greedy, fullsc, ac_fullsc, lr_fixed, ac_lr = categorize(results)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Greedy
    for r in greedy:
        ax.scatter(r["avg_samples"], r["accuracy"] * 100, marker="*",
                   s=200, c="black", zorder=10, label="Greedy")

    # FullSC (fixed budget)
    xs = [r["avg_samples"] for r in fullsc]
    ys = [r["accuracy"] * 100 for r in fullsc]
    ax.plot(xs, ys, "o-", color="#2196F3", linewidth=2, markersize=8, label="FullSC (fixed N)")

    # AC on FullSC (beta, various confidence)
    ac_beta_fc = [r for r in ac_fullsc if "beta" in r["method"]]
    ac_beta_fc.sort(key=lambda r: r["avg_samples"])
    if ac_beta_fc:
        xs = [r["avg_samples"] for r in ac_beta_fc]
        ys = [r["accuracy"] * 100 for r in ac_beta_fc]
        ax.plot(xs, ys, "s--", color="#FF9800", linewidth=2, markersize=8,
                label="AC(beta)-FullSC")
        for r in ac_beta_fc:
            conf = r["method"].split("c=")[1].split(")")[0]
            ax.annotate(f"c={conf}", (r["avg_samples"], r["accuracy"] * 100),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)

    # LR (fixed budget) -- best alpha=0.8
    lr_fixed.sort(key=lambda r: r["avg_samples"])
    xs = [r["avg_samples"] for r in lr_fixed]
    ys = [r["accuracy"] * 100 for r in lr_fixed]
    ax.plot(xs, ys, "D-", color="#4CAF50", linewidth=2, markersize=8,
            label="LateRollback (fixed)")

    # AC on LR (beta, c=0.95 only for clarity)
    ac_lr_95 = [r for r in ac_lr if "c=0.95" in r["method"]]
    ac_lr_95.sort(key=lambda r: r["avg_samples"])
    if ac_lr_95:
        xs = [r["avg_samples"] for r in ac_lr_95]
        ys = [r["accuracy"] * 100 for r in ac_lr_95]
        ax.plot(xs, ys, "^--", color="#E91E63", linewidth=2, markersize=8,
                label="AC(beta,c=0.95)-LR")
        for r in ac_lr_95:
            nd = r["method"].split("nd=")[1].split(",")[0]
            K = r["method"].split("K=")[1].split(")")[0]
            ax.annotate(f"nd={nd},K={K}", (r["avg_samples"], r["accuracy"] * 100),
                        textcoords="offset points", xytext=(5, -10), fontsize=7)

    ax.set_xlabel("Average Samples Used", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("MATH500 -- Llama 3.2 3B Instruct\nAccuracy vs Sample Efficiency", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 42)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "accuracy_vs_samples.png", dpi=150, bbox_inches="tight")
    print(f"Saved -> {FIG_DIR / 'accuracy_vs_samples.png'}")
    plt.close(fig)


def plot_accuracy_vs_tokens(results):
    greedy, fullsc, ac_fullsc, lr_fixed, ac_lr = categorize(results)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for r in greedy:
        ax.scatter(r["avg_tokens"], r["accuracy"] * 100, marker="*",
                   s=200, c="black", zorder=10, label="Greedy")

    xs = [r["avg_tokens"] for r in fullsc]
    ys = [r["accuracy"] * 100 for r in fullsc]
    ax.plot(xs, ys, "o-", color="#2196F3", linewidth=2, markersize=8, label="FullSC (fixed N)")

    ac_beta_fc = [r for r in ac_fullsc if "beta" in r["method"]]
    ac_beta_fc.sort(key=lambda r: r["avg_tokens"])
    if ac_beta_fc:
        xs = [r["avg_tokens"] for r in ac_beta_fc]
        ys = [r["accuracy"] * 100 for r in ac_beta_fc]
        ax.plot(xs, ys, "s--", color="#FF9800", linewidth=2, markersize=8,
                label="AC(beta)-FullSC")

    lr_fixed.sort(key=lambda r: r["avg_tokens"])
    xs = [r["avg_tokens"] for r in lr_fixed]
    ys = [r["accuracy"] * 100 for r in lr_fixed]
    ax.plot(xs, ys, "D-", color="#4CAF50", linewidth=2, markersize=8,
            label="LateRollback (fixed)")

    ac_lr_95 = [r for r in ac_lr if "c=0.95" in r["method"]]
    ac_lr_95.sort(key=lambda r: r["avg_tokens"])
    if ac_lr_95:
        xs = [r["avg_tokens"] for r in ac_lr_95]
        ys = [r["accuracy"] * 100 for r in ac_lr_95]
        ax.plot(xs, ys, "^--", color="#E91E63", linewidth=2, markersize=8,
                label="AC(beta,c=0.95)-LR")

    ax.set_xlabel("Average Tokens per Question", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("MATH500 -- Llama 3.2 3B Instruct\nAccuracy vs Token Cost", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "accuracy_vs_tokens.png", dpi=150, bbox_inches="tight")
    print(f"Saved -> {FIG_DIR / 'accuracy_vs_tokens.png'}")
    plt.close(fig)


def plot_pareto_frontier(results):
    """Highlight the Pareto-optimal methods on the samples-accuracy plane."""
    greedy, fullsc, ac_fullsc, lr_fixed, ac_lr = categorize(results)

    all_methods = greedy + fullsc + ac_fullsc + lr_fixed + ac_lr
    all_methods.sort(key=lambda r: r["avg_samples"])

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    color_map = {
        "Greedy": "black",
        "FullSC": "#2196F3",
        "AC-FullSC": "#FF9800",
        "LR": "#4CAF50",
        "AC-LR": "#E91E63",
    }

    def get_category(method_name):
        if method_name == "Greedy":
            return "Greedy"
        if method_name.startswith("FullSC@"):
            return "FullSC"
        if "AC(" in method_name and "LR" not in method_name:
            return "AC-FullSC"
        if method_name.startswith("LR("):
            return "LR"
        return "AC-LR"

    plotted_labels = set()
    for r in all_methods:
        cat = get_category(r["method"])
        label = cat if cat not in plotted_labels else None
        plotted_labels.add(cat)
        marker = {"Greedy": "*", "FullSC": "o", "AC-FullSC": "s", "LR": "D", "AC-LR": "^"}[cat]
        size = 200 if cat == "Greedy" else 60
        ax.scatter(r["avg_samples"], r["accuracy"] * 100,
                   c=color_map[cat], marker=marker, s=size, label=label, zorder=5)

    # Pareto frontier
    pareto = []
    best_acc = -1
    for r in sorted(all_methods, key=lambda r: r["avg_samples"]):
        if r["accuracy"] > best_acc:
            pareto.append(r)
            best_acc = r["accuracy"]
    if pareto:
        xs = [r["avg_samples"] for r in pareto]
        ys = [r["accuracy"] * 100 for r in pareto]
        ax.step(xs, ys, where="post", color="gray", linewidth=1.5,
                linestyle=":", alpha=0.7, label="Pareto frontier")

    ax.set_xlabel("Average Samples Used", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("MATH500 -- Pareto Frontier (Accuracy vs Efficiency)", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "pareto_frontier.png", dpi=150, bbox_inches="tight")
    print(f"Saved -> {FIG_DIR / 'pareto_frontier.png'}")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    plot_accuracy_vs_samples(results)
    plot_accuracy_vs_tokens(results)
    plot_pareto_frontier(results)
    print("Done.")


if __name__ == "__main__":
    main()
