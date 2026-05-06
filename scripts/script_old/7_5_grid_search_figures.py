#!/usr/bin/env python3
"""
Visualize grid search results: heatmap, Pareto frontier, cost-accuracy curves.

Usage:
    python scripts/7_5_grid_search_figures.py
"""

import argparse
import json
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
    ap.add_argument("--grid-dir", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/grid_search"))
    ap.add_argument("--fig-dir", default=str(
        PROJECT_ROOT / "figures/late_rollback_grid"))
    return ap.parse_args()


def main():
    args = parse_args()
    grid_dir = Path(args.grid_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((grid_dir / "grid_summary.json").read_text("utf-8"))
    grid = summary["grid"]
    Ks = summary["Ks"]
    alphas = summary["alphas"]
    greedy_acc = summary["greedy_accuracy"]
    greedy_tpq = summary["greedy_tokens_per_question"]

    # ── Figure 1: Accuracy heatmap (K x alpha) ──────────────────────────
    acc_matrix = np.zeros((len(Ks), len(alphas)))
    for i, K in enumerate(Ks):
        for j, alpha in enumerate(alphas):
            key = f"LR_K{K}_a{alpha:.1f}"
            acc_matrix[i, j] = grid[key]["accuracy"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(acc_matrix, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:.1f}" for a in alphas])
    ax.set_yticks(range(len(Ks)))
    ax.set_yticklabels([str(k) for k in Ks])
    ax.set_xlabel("alpha (rollback fraction)")
    ax.set_ylabel("K (total votes)")
    ax.set_title("Late Rollback Accuracy: K x alpha")
    for i in range(len(Ks)):
        for j in range(len(alphas)):
            ax.text(j, i, f"{acc_matrix[i, j]:.3f}",
                    ha="center", va="center", fontsize=10,
                    color="white" if acc_matrix[i, j] > acc_matrix.mean()
                    else "black")
    fig.colorbar(im, ax=ax, label="Accuracy")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_accuracy_heatmap.png", dpi=200)
    fig.savefig(fig_dir / "fig_accuracy_heatmap.pdf")
    plt.close(fig)
    print(f"[1] Accuracy heatmap -> {fig_dir / 'fig_accuracy_heatmap.png'}")

    # ── Figure 2: Gain-over-greedy heatmap ───────────────────────────────
    gain_matrix = acc_matrix - greedy_acc

    fig, ax = plt.subplots(figsize=(7, 4.5))
    vmax = max(abs(gain_matrix.min()), abs(gain_matrix.max()), 0.01)
    im = ax.imshow(gain_matrix, cmap="RdYlGn", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:.1f}" for a in alphas])
    ax.set_yticks(range(len(Ks)))
    ax.set_yticklabels([str(k) for k in Ks])
    ax.set_xlabel("alpha (rollback fraction)")
    ax.set_ylabel("K (total votes)")
    ax.set_title("Accuracy Gain over Greedy: K x alpha")
    for i in range(len(Ks)):
        for j in range(len(alphas)):
            ax.text(j, i, f"{gain_matrix[i, j]:+.3f}",
                    ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, label="Gain")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_gain_heatmap.png", dpi=200)
    fig.savefig(fig_dir / "fig_gain_heatmap.pdf")
    plt.close(fig)
    print(f"[2] Gain heatmap -> {fig_dir / 'fig_gain_heatmap.png'}")

    # ── Figure 3: Pareto frontier (accuracy vs tokens/question) ──────────
    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.scatter(greedy_tpq, greedy_acc, s=150, marker="D",
               color="#9e9e9e", zorder=10, label="Greedy")

    cmap_lr = plt.cm.viridis
    for i, K in enumerate(Ks):
        color = cmap_lr(i / max(len(Ks) - 1, 1))
        xs, ys, labels = [], [], []
        for alpha in alphas:
            key = f"LR_K{K}_a{alpha:.1f}"
            r = grid[key]
            xs.append(r["tokens_per_question"])
            ys.append(r["accuracy"])
            labels.append(f"a={alpha:.1f}")
        ax.plot(xs, ys, "o-", color=color, markersize=7,
                label=f"LR K={K}", zorder=5)
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, color=color)

    for K in Ks:
        key = f"FullSC_K{K}"
        r = grid[key]
        ax.scatter(r["tokens_per_question"], r["accuracy"],
                   s=120, marker="s", zorder=8,
                   label=f"Full SC K={K}")

    ax.set_xlabel("Tokens per question")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Compute: Pareto Frontier")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_pareto.png", dpi=200)
    fig.savefig(fig_dir / "fig_pareto.pdf")
    plt.close(fig)
    print(f"[3] Pareto frontier -> {fig_dir / 'fig_pareto.png'}")

    # ── Figure 4: Savings vs Full SC at same K ───────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    x_pos = 0
    tick_labels = []
    tick_positions = []
    colors_bar = plt.cm.Set2.colors

    for ki, K in enumerate(Ks):
        sc_key = f"FullSC_K{K}"
        sc_tpq = grid[sc_key]["tokens_per_question"]
        sc_acc = grid[sc_key]["accuracy"]

        for ai, alpha in enumerate(alphas):
            lr_key = f"LR_K{K}_a{alpha:.1f}"
            lr = grid[lr_key]
            savings = 1 - lr["tokens_per_question"] / sc_tpq
            acc_gap = lr["accuracy"] - sc_acc

            ax.bar(x_pos, savings, color=colors_bar[ai % len(colors_bar)],
                   edgecolor="black", linewidth=0.5)
            ax.text(x_pos, savings + 0.01,
                    f"{acc_gap:+.3f}", ha="center", va="bottom",
                    fontsize=8, rotation=45)
            tick_labels.append(f"K={K}\na={alpha:.1f}")
            tick_positions.append(x_pos)
            x_pos += 1
        x_pos += 0.5

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel("Token savings vs Full SC")
    ax.set_title("Token Savings vs Full SC (labels = accuracy gap)")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_savings_vs_sc.png", dpi=200)
    fig.savefig(fig_dir / "fig_savings_vs_sc.pdf")
    plt.close(fig)
    print(f"[4] Savings vs SC -> {fig_dir / 'fig_savings_vs_sc.png'}")

    # ── Figure 5: Accuracy curves by alpha (lines = K) ──────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, K in enumerate(Ks):
        accs = [grid[f"LR_K{K}_a{a:.1f}"]["accuracy"] for a in alphas]
        ax.plot(alphas, accs, "o-", label=f"LR K={K}", linewidth=2)

    for K in Ks:
        sc_acc = grid[f"FullSC_K{K}"]["accuracy"]
        ax.axhline(y=sc_acc, linestyle="--", alpha=0.5,
                   label=f"Full SC K={K}")

    ax.axhline(y=greedy_acc, linestyle=":", color="gray",
               label="Greedy")
    ax.set_xlabel("alpha (rollback fraction)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Rollback Point by K")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_accuracy_by_alpha.png", dpi=200)
    fig.savefig(fig_dir / "fig_accuracy_by_alpha.pdf")
    plt.close(fig)
    print(f"[5] Accuracy by alpha -> {fig_dir / 'fig_accuracy_by_alpha.png'}")

    print(f"\nAll figures saved to {fig_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
