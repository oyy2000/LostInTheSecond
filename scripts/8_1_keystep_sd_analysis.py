#!/usr/bin/env python3
"""
KeyStep-SD analysis: accuracy, token cost, trigger statistics, and
comparison figures against baselines (greedy, SC, late rollback).

Usage:
    python scripts/8_1_keystep_sd_analysis.py \
        --keystep-dir results/gsm8k_3b_multi_sample/keystep_sd \
        --grid-file results/gsm8k_3b_multi_sample/grid_search/grid_summary.json
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def parse_args():
    ap = argparse.ArgumentParser(description="KeyStep-SD analysis")
    ap.add_argument("--keystep-dir", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/keystep_sd"))
    ap.add_argument("--grid-file", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/grid_search/grid_summary.json"))
    ap.add_argument("--fig-dir", default=str(PROJECT_ROOT / "figures/keystep_sd"))
    return ap.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text("utf-8"))


def load_jsonl(path):
    out = []
    for line in Path(path).read_text("utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out

def print_trigger_stats(trigger_file):
    """Print trigger distribution statistics."""
    data = load_jsonl(trigger_file)
    all_scores = []
    triggered_counts = []
    for rec in data:
        n_trig = rec.get("n_triggered", 0)
        triggered_counts.append(n_trig)
        for t in rec.get("triggers", []):
            all_scores.append(t["trigger_score"])

    scores = np.array(all_scores) if all_scores else np.array([0.0])
    counts = np.array(triggered_counts)
    print(f"\nTrigger statistics ({len(data)} questions):")
    print(f"  Score: mean={scores.mean():.4f}, "
          f"median={np.median(scores):.4f}, "
          f"std={scores.std():.4f}")
    print(f"  Triggered steps/question: mean={counts.mean():.2f}, "
          f"max={counts.max()}")
    print(f"  Questions with >=1 trigger: "
          f"{(counts > 0).sum()}/{len(data)} "
          f"({(counts > 0).mean():.1%})")
    return scores, counts

def plot_comparison_bar(summary, grid, fig_dir):
    """Bar chart: accuracy comparison across methods."""
    methods = ["Greedy 3B"]
    accs = [summary["draft_accuracy"]]
    tokens = [summary["small_tokens_per_question"]]

    methods.append("KeyStep-SD")
    accs.append(summary["keystep_sd_accuracy"])
    tokens.append(summary["tokens_per_question"])

    for key in ["FullSC_K8", "FullSC_K16", "FullSC_K32"]:
        if key in grid.get("grid", {}):
            g = grid["grid"][key]
            methods.append(f"SC K={g['K']}")
            accs.append(g["accuracy"])
            tokens.append(g["tokens_per_question"])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    bars = ax.bar(x, [a * 100 for a in accs], width=0.6,
                  color=["#888888", "#e74c3c", "#3498db", "#2ecc71", "#9b59b6"])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("KeyStep-SD vs Baselines")
    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{a*100:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(78, 92)
    fig.tight_layout()
    fig.savefig(Path(fig_dir) / "accuracy_comparison.pdf", dpi=150)
    fig.savefig(Path(fig_dir) / "accuracy_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  -> accuracy_comparison.pdf")

def plot_cost_accuracy(summary, grid, fig_dir):
    """Scatter: tokens/question vs accuracy for all methods."""
    points = []
    points.append(("Greedy", summary["small_tokens_per_question"],
                    summary["draft_accuracy"]))
    points.append(("KeyStep-SD", summary["tokens_per_question"],
                    summary["keystep_sd_accuracy"]))

    for key, entry in grid.get("grid", {}).items():
        label = key.replace("_", " ")
        points.append((label, entry["tokens_per_question"], entry["accuracy"]))

    fig, ax = plt.subplots(figsize=(9, 6))
    for label, tok, acc in points:
        marker = "s" if "KeyStep" in label else ("^" if "LR" in label else "o")
        color = ("#e74c3c" if "KeyStep" in label
                 else "#3498db" if "SC" in label
                 else "#2ecc71" if "LR" in label
                 else "#888888")
        size = 120 if "KeyStep" in label else 50
        ax.scatter(tok, acc * 100, s=size, marker=marker, color=color,
                   zorder=5, edgecolors="white", linewidths=0.5)
        if "KeyStep" in label or "Greedy" in label or "K32" in label:
            ax.annotate(label, (tok, acc * 100), fontsize=7,
                        xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Tokens / question")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Cost-Accuracy Tradeoff")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(fig_dir) / "cost_accuracy.pdf", dpi=150)
    fig.savefig(Path(fig_dir) / "cost_accuracy.png", dpi=150)
    plt.close(fig)
    print(f"  -> cost_accuracy.pdf")

def plot_trigger_distribution(scores, counts, fig_dir):
    """Histograms of trigger scores and per-question trigger counts."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(scores, bins=30, color="#3498db", edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Trigger score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Trigger Score Distribution")

    max_c = int(counts.max()) + 1
    axes[1].hist(counts, bins=range(max_c + 1), color="#e74c3c",
                 edgecolor="white", alpha=0.8, align="left")
    axes[1].set_xlabel("Triggered steps per question")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Triggers per Question")

    fig.tight_layout()
    fig.savefig(Path(fig_dir) / "trigger_distribution.pdf", dpi=150)
    fig.savefig(Path(fig_dir) / "trigger_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  -> trigger_distribution.pdf")

def main():
    args = parse_args()
    ks_dir = Path(args.keystep_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary_file = ks_dir / "summary.json"
    if not summary_file.exists():
        print(f"ERROR: {summary_file} not found. Run 8_0 first.")
        sys.exit(1)
    summary = load_json(summary_file)

    grid = {}
    grid_path = Path(args.grid_file)
    if grid_path.exists():
        grid = load_json(grid_path)

    print("=== KeyStep-SD Results ===")
    print(f"  Draft accuracy:     {summary['draft_accuracy']:.4f}")
    print(f"  KeyStep-SD accuracy:{summary['keystep_sd_accuracy']:.4f}")
    print(f"  Gain:               "
          f"{(summary['keystep_sd_accuracy'] - summary['draft_accuracy'])*100:+.2f}pp")
    print(f"  Intervened:         {summary['n_intervened']}/{summary['n_questions']} "
          f"({summary['frac_intervened']:.1%})")
    print(f"  Tokens/q:           {summary['tokens_per_question']:.1f} "
          f"(small={summary['small_tokens_per_question']:.1f}, "
          f"large={summary['large_tokens_per_question']:.1f})")

    trigger_file = ks_dir / "trigger_decisions.jsonl"
    scores, counts = np.array([0.0]), np.array([0])
    if trigger_file.exists():
        scores, counts = print_trigger_stats(trigger_file)

    print("\nGenerating figures...")
    plot_comparison_bar(summary, grid, fig_dir)
    plot_cost_accuracy(summary, grid, fig_dir)
    if len(scores) > 1:
        plot_trigger_distribution(scores, counts, fig_dir)

    per_q_file = ks_dir / "per_question.jsonl"
    if per_q_file.exists():
        details = load_jsonl(per_q_file)
        intervened = [d for d in details if d["intervened"]]
        not_intervened = [d for d in details if not d["intervened"]]
        if intervened:
            acc_iv = np.mean([d["exact_match"] for d in intervened])
            acc_niv = np.mean([d["exact_match"] for d in not_intervened]) if not_intervened else 0
            draft_acc_iv = np.mean([d["draft_correct"] for d in intervened])
            print(f"\n  Intervened subset ({len(intervened)}):")
            print(f"    Draft acc:     {draft_acc_iv:.4f}")
            print(f"    KeyStep acc:   {acc_iv:.4f}")
            print(f"    Gain:          {(acc_iv - draft_acc_iv)*100:+.2f}pp")
            print(f"  Non-intervened ({len(not_intervened)}): acc={acc_niv:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
