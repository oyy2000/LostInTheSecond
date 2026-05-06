#!/usr/bin/env python3
"""
Compare Late Rollback vs Full Self-Consistency.

Reads summary.json from both experiments and produces:
  - Accuracy comparison bar chart
  - Token efficiency scatter (accuracy vs tokens)
  - Accuracy gain per 1k extra tokens
  - Per-question flip analysis

Usage:
    python scripts/7_3_compare_late_rollback_vs_sc.py
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
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr-dir", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/late_rollback"))
    ap.add_argument("--sc-dir", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/full_sc"))
    ap.add_argument("--fig-dir", default=str(
        PROJECT_ROOT / "figures/late_rollback"))
    return ap.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text("utf-8"))


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text("utf-8").splitlines() if l.strip()]


def main():
    args = parse_args()
    lr_dir = Path(args.lr_dir)
    sc_dir = Path(args.sc_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    lr_summary = load_json(lr_dir / "summary.json")
    sc_summary = load_json(sc_dir / "summary.json")

    K = lr_summary["K"]
    alphas = lr_summary["alphas"]
    greedy_acc = lr_summary["greedy_accuracy"]
    greedy_tokens = lr_summary.get("greedy_total_tokens", 0)
    n_questions = lr_summary["n_questions"]

    sc_acc = sc_summary["majority_vote_accuracy"]
    sc_tokens = sc_summary["total_tokens"]
    sc_individual = sc_summary["individual_accuracy"]

    # Collect late rollback results
    lr_results = {}
    for alpha in alphas:
        key = f"alpha_{alpha:.1f}"
        if key in lr_summary["results"]:
            lr_results[alpha] = lr_summary["results"][key]

    # ------------------------------------------------------------------
    # Figure 1: Accuracy comparison bar chart
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = ["Greedy"]
    accs = [greedy_acc]
    colors = ["#9e9e9e"]

    for alpha in sorted(lr_results.keys()):
        methods.append(f"LR a={alpha:.1f}")
        accs.append(lr_results[alpha]["accuracy"])
        colors.append(plt.cm.Blues(0.4 + 0.2 * (alpha - 0.5)))

    methods.append(f"Full SC (K={K})")
    accs.append(sc_acc)
    colors.append("#e74c3c")

    x = np.arange(len(methods))
    bars = ax.bar(x, accs, color=colors, edgecolor="black", linewidth=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"GSM8K Accuracy: Late Rollback vs Full SC (K={K})")
    ax.set_ylim(0, min(1.0, max(accs) + 0.05))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_accuracy_comparison.png", dpi=200)
    fig.savefig(fig_dir / "fig_accuracy_comparison.pdf")
    plt.close(fig)
    print(f"[1] Accuracy comparison -> {fig_dir / 'fig_accuracy_comparison.png'}")

    # ------------------------------------------------------------------
    # Figure 2: Token efficiency (accuracy vs total tokens)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(greedy_tokens / n_questions, greedy_acc,
               s=120, marker="D", color="#9e9e9e", zorder=5, label="Greedy")

    for alpha in sorted(lr_results.keys()):
        r = lr_results[alpha]
        tpq = r["tokens_per_question"]
        ax.scatter(tpq, r["accuracy"], s=120, marker="o",
                   zorder=5, label=f"LR a={alpha:.1f}")

    ax.scatter(sc_tokens / n_questions, sc_acc,
               s=120, marker="s", color="#e74c3c", zorder=5,
               label=f"Full SC (K={K})")

    ax.set_xlabel("Tokens per question")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Compute Budget")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_token_efficiency.png", dpi=200)
    fig.savefig(fig_dir / "fig_token_efficiency.pdf")
    plt.close(fig)
    print(f"[2] Token efficiency -> {fig_dir / 'fig_token_efficiency.png'}")

    # ------------------------------------------------------------------
    # Figure 3: Accuracy gain per 1k extra tokens
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    greedy_tpq = greedy_tokens / n_questions
    methods_eff = []
    gains_per_1k = []
    bar_colors = []

    for alpha in sorted(lr_results.keys()):
        r = lr_results[alpha]
        extra_tokens = r["tokens_per_question"] - greedy_tpq
        gain = r["accuracy"] - greedy_acc
        gpk = (gain / extra_tokens * 1000) if extra_tokens > 0 else 0
        methods_eff.append(f"LR a={alpha:.1f}")
        gains_per_1k.append(gpk)
        bar_colors.append(plt.cm.Blues(0.4 + 0.2 * (alpha - 0.5)))

    sc_extra = sc_tokens / n_questions - greedy_tpq
    sc_gain = sc_acc - greedy_acc
    sc_gpk = (sc_gain / sc_extra * 1000) if sc_extra > 0 else 0
    methods_eff.append(f"Full SC (K={K})")
    gains_per_1k.append(sc_gpk)
    bar_colors.append("#e74c3c")

    x = np.arange(len(methods_eff))
    bars = ax.bar(x, gains_per_1k, color=bar_colors,
                  edgecolor="black", linewidth=0.5)
    for bar, g in zip(bars, gains_per_1k):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                f"{g:.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(methods_eff, rotation=15, ha="right")
    ax.set_ylabel("Accuracy gain per 1k extra tokens")
    ax.set_title("Token Efficiency: Gain per 1k Extra Tokens over Greedy")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_gain_per_1k_tokens.png", dpi=200)
    fig.savefig(fig_dir / "fig_gain_per_1k_tokens.pdf")
    plt.close(fig)
    print(f"[3] Gain per 1k tokens -> {fig_dir / 'fig_gain_per_1k_tokens.png'}")

    # ------------------------------------------------------------------
    # Figure 4: Cost breakdown (stacked bar)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    method_names = []
    draft_costs = []
    suffix_costs = []

    for alpha in sorted(lr_results.keys()):
        r = lr_results[alpha]
        method_names.append(f"LR a={alpha:.1f}")
        draft_costs.append(r["draft_tokens"] / n_questions)
        suffix_costs.append(r["suffix_tokens"] / n_questions)

    method_names.append(f"Full SC (K={K})")
    draft_costs.append(0)
    suffix_costs.append(sc_tokens / n_questions)

    x = np.arange(len(method_names))
    w = 0.5
    ax.bar(x, draft_costs, w, label="Draft (greedy)", color="#3498db")
    ax.bar(x, suffix_costs, w, bottom=draft_costs,
           label="Suffix / Full samples", color="#e67e22")

    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=15, ha="right")
    ax.set_ylabel("Tokens per question")
    ax.set_title("Compute Cost Breakdown")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_cost_breakdown.png", dpi=200)
    fig.savefig(fig_dir / "fig_cost_breakdown.pdf")
    plt.close(fig)
    print(f"[4] Cost breakdown -> {fig_dir / 'fig_cost_breakdown.png'}")

    # ------------------------------------------------------------------
    # Figure 5: Per-question flip analysis (LR best alpha vs Full SC)
    # ------------------------------------------------------------------
    best_alpha = max(lr_results.keys(), key=lambda a: lr_results[a]["accuracy"])
    lr_vote_path = lr_dir / f"votes_alpha{best_alpha:.1f}.jsonl"
    sc_vote_path = sc_dir / "votes.jsonl"

    if lr_vote_path.exists() and sc_vote_path.exists():
        lr_votes = {r["doc_id"]: r for r in load_jsonl(lr_vote_path)}
        sc_votes = {r["doc_id"]: r for r in load_jsonl(sc_vote_path)}

        common_docs = set(lr_votes.keys()) & set(sc_votes.keys())
        both_correct = 0
        lr_only = 0
        sc_only = 0
        both_wrong = 0

        for doc_id in common_docs:
            lr_c = lr_votes[doc_id]["voted_correct"]
            sc_c = sc_votes[doc_id]["voted_correct"]
            if lr_c and sc_c:
                both_correct += 1
            elif lr_c and not sc_c:
                lr_only += 1
            elif not lr_c and sc_c:
                sc_only += 1
            else:
                both_wrong += 1

        fig, ax = plt.subplots(figsize=(5, 5))
        labels = [
            f"Both correct\n({both_correct})",
            f"LR only\n({lr_only})",
            f"SC only\n({sc_only})",
            f"Both wrong\n({both_wrong})",
        ]
        sizes = [both_correct, lr_only, sc_only, both_wrong]
        colors_pie = ["#2ecc71", "#3498db", "#e74c3c", "#95a5a6"]
        ax.pie(sizes, labels=labels, colors=colors_pie, autopct="%1.1f%%",
               startangle=90)
        ax.set_title(f"Question-level Agreement\n"
                     f"LR(a={best_alpha:.1f}) vs Full SC (K={K})")
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_flip_analysis.png", dpi=200)
        fig.savefig(fig_dir / "fig_flip_analysis.pdf")
        plt.close(fig)
        print(f"[5] Flip analysis -> {fig_dir / 'fig_flip_analysis.png'}")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{'Method':<20} {'Accuracy':>10} {'Tokens/Q':>12} "
          f"{'Gain/1k':>10} {'Savings':>10}")
    print("-" * 70)
    print(f"{'Greedy':<20} {greedy_acc:>10.4f} {greedy_tpq:>12.0f} "
          f"{'--':>10} {'--':>10}")

    for alpha in sorted(lr_results.keys()):
        r = lr_results[alpha]
        extra = r["tokens_per_question"] - greedy_tpq
        gain = r["accuracy"] - greedy_acc
        gpk = (gain / extra * 1000) if extra > 0 else 0
        savings = 1 - r["tokens_per_question"] / (sc_tokens / n_questions)
        print(f"{'LR a=' + f'{alpha:.1f}':<20} {r['accuracy']:>10.4f} "
              f"{r['tokens_per_question']:>12.0f} {gpk:>10.4f} "
              f"{savings:>9.1%}")

    sc_extra = sc_tokens / n_questions - greedy_tpq
    sc_gpk = (sc_gain / sc_extra * 1000) if sc_extra > 0 else 0
    print(f"{'Full SC (K=' + str(K) + ')':<20} {sc_acc:>10.4f} "
          f"{sc_tokens / n_questions:>12.0f} {sc_gpk:>10.4f} {'0.0%':>10}")
    print("=" * 70)

    # Save combined summary
    combined = {
        "greedy": {"accuracy": greedy_acc, "tokens_per_question": greedy_tpq},
        "full_sc": {
            "accuracy": sc_acc,
            "tokens_per_question": sc_tokens / n_questions,
            "K": K,
        },
        "late_rollback": {},
    }
    for alpha in sorted(lr_results.keys()):
        r = lr_results[alpha]
        extra = r["tokens_per_question"] - greedy_tpq
        gain = r["accuracy"] - greedy_acc
        combined["late_rollback"][f"alpha_{alpha:.1f}"] = {
            "accuracy": r["accuracy"],
            "tokens_per_question": r["tokens_per_question"],
            "gain_over_greedy": gain,
            "savings_vs_full_sc": 1 - r["tokens_per_question"] / (sc_tokens / n_questions),
            "gain_per_1k_tokens": (gain / extra * 1000) if extra > 0 else 0,
        }

    combined_path = fig_dir / "combined_summary.json"
    combined_path.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nCombined summary -> {combined_path}")
    print("Done.")


if __name__ == "__main__":
    main()
