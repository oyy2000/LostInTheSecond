#!/usr/bin/env python3
"""
Comprehensive figures for Late Rollback experiments.

Combines grid search (single-draft) + dual-draft + full SC results
into publication-ready figures.

Usage:
    python scripts/7_7_combined_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "figures" / "late_rollback_combined"
GRID_SUMMARY = (
    PROJECT_ROOT
    / "results/gsm8k_3b_multi_sample/grid_search/grid_summary.json"
)
DUAL_SUMMARY = (
    PROJECT_ROOT
    / "results/gsm8k_3b_multi_sample/dual_draft/summary.json"
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9.5,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

COLORS = {
    "greedy": "#9e9e9e",
    "full_sc": "#e74c3c",
    "single_8": "#a6cee3",
    "single_16": "#1f78b4",
    "single_32": "#08306b",
    "dual_16": "#33a02c",
}
MARKERS = {
    "greedy": "D",
    "full_sc": "s",
    "single": "o",
    "dual": "^",
}


def load():
    gs = json.loads(GRID_SUMMARY.read_text("utf-8"))
    dd = json.loads(DUAL_SUMMARY.read_text("utf-8"))
    return gs, dd


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    plt.close(fig)
    print(f"  {name}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    gs, dd = load()

    grid = gs["grid"]
    Ks = gs["Ks"]
    alphas_grid = gs["alphas"]
    greedy_acc = gs["greedy_accuracy"]
    greedy_tpq = gs["greedy_tokens_per_question"]

    dd_alphas = dd["alphas"]
    dd_K = dd["K"]

    # ── Fig 1: Pareto frontier (all methods) ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.scatter(greedy_tpq, greedy_acc, s=140, marker="D",
               color=COLORS["greedy"], zorder=10, label="Greedy (1 sample)")

    k_colors = {8: COLORS["single_8"], 16: COLORS["single_16"],
                32: COLORS["single_32"]}
    for K in Ks:
        xs = [grid[f"LR_K{K}_a{a:.1f}"]["tokens_per_question"]
              for a in alphas_grid]
        ys = [grid[f"LR_K{K}_a{a:.1f}"]["accuracy"] for a in alphas_grid]
        ax.plot(xs, ys, "o-", color=k_colors[K], markersize=6,
                linewidth=1.5, label=f"Single-Draft LR K={K}", zorder=5)

    xs_dd = [dd["results"][f"alpha_{a:.1f}"]["tokens_per_question"]
             for a in dd_alphas]
    ys_dd = [dd["results"][f"alpha_{a:.1f}"]["accuracy"] for a in dd_alphas]
    ax.plot(xs_dd, ys_dd, "^-", color=COLORS["dual_16"], markersize=8,
            linewidth=2, label=f"Dual-Draft LR K={dd_K}", zorder=7)

    for K in Ks:
        sc = grid[f"FullSC_K{K}"]
        ax.scatter(sc["tokens_per_question"], sc["accuracy"],
                   s=100, marker="s", color=COLORS["full_sc"], zorder=8,
                   edgecolors="black", linewidth=0.5)
        ax.annotate(f"SC K={K}",
                    (sc["tokens_per_question"], sc["accuracy"]),
                    textcoords="offset points", xytext=(6, -10),
                    fontsize=8, color=COLORS["full_sc"])

    ax.set_xlabel("Tokens per question")
    ax.set_ylabel("Accuracy")
    ax.set_title("GSM8K: Accuracy vs Compute Budget")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    save(fig, "fig_pareto_all")

    # ── Fig 2: Dual vs Single head-to-head (K=16) ───────────────────────
    common_alphas = [a for a in dd_alphas if a in alphas_grid]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 2a: accuracy comparison
    ax = axes[0]
    x = np.arange(len(common_alphas))
    w = 0.28

    single_accs = [grid[f"LR_K16_a{a:.1f}"]["accuracy"]
                   for a in common_alphas]
    dual_accs = [dd["results"][f"alpha_{a:.1f}"]["accuracy"]
                 for a in common_alphas]
    sc_acc = grid["FullSC_K16"]["accuracy"]

    bars1 = ax.bar(x - w / 2, single_accs, w, label="Single-Draft K=16",
                   color=COLORS["single_16"], edgecolor="black",
                   linewidth=0.5)
    bars2 = ax.bar(x + w / 2, dual_accs, w, label="Dual-Draft K=16",
                   color=COLORS["dual_16"], edgecolor="black",
                   linewidth=0.5)
    ax.axhline(y=sc_acc, color=COLORS["full_sc"], linestyle="--",
               linewidth=1.5, label=f"Full SC K=16 ({sc_acc:.3f})")
    ax.axhline(y=greedy_acc, color=COLORS["greedy"], linestyle=":",
               linewidth=1, label=f"Greedy ({greedy_acc:.3f})")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.001,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"a={a}" for a in common_alphas])
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy: Single vs Dual Draft (K=16)")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_ylim(greedy_acc - 0.01, sc_acc + 0.015)
    ax.grid(axis="y", alpha=0.25)

    # 2b: token efficiency
    ax = axes[1]
    single_eff = []
    dual_eff = []
    for a in common_alphas:
        sr = grid[f"LR_K16_a{a:.1f}"]
        dr = dd["results"][f"alpha_{a:.1f}"]
        s_extra = sr["tokens_per_question"] - greedy_tpq
        d_extra = dr["tokens_per_question"] - greedy_tpq
        single_eff.append(
            sr["gain_over_greedy"] / s_extra * 1000 if s_extra > 0 else 0
        )
        dual_eff.append(
            dr["gain_over_greedy"] / d_extra * 1000 if d_extra > 0 else 0
        )

    sc_extra = grid["FullSC_K16"]["tokens_per_question"] - greedy_tpq
    sc_eff = (grid["FullSC_K16"]["gain_over_greedy"] / sc_extra * 1000
              if sc_extra > 0 else 0)

    bars1 = ax.bar(x - w / 2, single_eff, w, label="Single-Draft K=16",
                   color=COLORS["single_16"], edgecolor="black",
                   linewidth=0.5)
    bars2 = ax.bar(x + w / 2, dual_eff, w, label="Dual-Draft K=16",
                   color=COLORS["dual_16"], edgecolor="black",
                   linewidth=0.5)
    ax.axhline(y=sc_eff, color=COLORS["full_sc"], linestyle="--",
               linewidth=1.5, label=f"Full SC K=16 ({sc_eff:.4f})")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.0002,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"a={a}" for a in common_alphas])
    ax.set_ylabel("Accuracy gain per 1k extra tokens")
    ax.set_title("Token Efficiency: Single vs Dual Draft (K=16)")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    save(fig, "fig_dual_vs_single_k16")

    # ── Fig 3: Cost savings vs Full SC ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    groups = []
    for K in Ks:
        sc_tpq = grid[f"FullSC_K{K}"]["tokens_per_question"]
        sc_a = grid[f"FullSC_K{K}"]["accuracy"]
        for a in [0.4, 0.5, 0.6]:
            lr = grid[f"LR_K{K}_a{a:.1f}"]
            savings = 1 - lr["tokens_per_question"] / sc_tpq
            acc_gap = lr["accuracy"] - sc_a
            groups.append({
                "label": f"K={K}\na={a}",
                "savings": savings,
                "acc_gap": acc_gap,
                "type": "single",
                "K": K,
            })

    sc16_tpq = grid["FullSC_K16"]["tokens_per_question"]
    sc16_a = grid["FullSC_K16"]["accuracy"]
    for a in dd_alphas:
        dr = dd["results"][f"alpha_{a:.1f}"]
        savings = 1 - dr["tokens_per_question"] / sc16_tpq
        acc_gap = dr["accuracy"] - sc16_a
        groups.append({
            "label": f"DD\na={a}",
            "savings": savings,
            "acc_gap": acc_gap,
            "type": "dual",
            "K": 16,
        })

    x_pos = 0
    ticks, labels = [], []
    for i, g in enumerate(groups):
        c = COLORS["dual_16"] if g["type"] == "dual" else COLORS["single_16"]
        ax.bar(x_pos, g["savings"], color=c, edgecolor="black",
               linewidth=0.5, width=0.7)
        ax.text(x_pos, g["savings"] + 0.01,
                f"{g['acc_gap']:+.3f}", ha="center", va="bottom",
                fontsize=7.5, rotation=45)
        ticks.append(x_pos)
        labels.append(g["label"])
        x_pos += 1
        if (i + 1) % 3 == 0 and g["type"] == "single":
            x_pos += 0.5

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Token savings vs Full SC (same K)")
    ax.set_title("Token Savings vs Full SC (labels = accuracy gap)")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.25)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=COLORS["single_16"], label="Single-Draft LR"),
        Patch(facecolor=COLORS["dual_16"], label="Dual-Draft LR"),
    ], loc="upper right")

    fig.tight_layout()
    save(fig, "fig_savings_vs_sc")

    # ── Fig 4: Accuracy heatmap (K x alpha, single-draft) ───────────────
    fig, ax = plt.subplots(figsize=(7, 4))

    acc_mat = np.array([
        [grid[f"LR_K{K}_a{a:.1f}"]["accuracy"] for a in alphas_grid]
        for K in Ks
    ])

    im = ax.imshow(acc_mat, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(alphas_grid)))
    ax.set_xticklabels([f"{a:.1f}" for a in alphas_grid])
    ax.set_yticks(range(len(Ks)))
    ax.set_yticklabels([str(k) for k in Ks])
    ax.set_xlabel("alpha (rollback fraction)")
    ax.set_ylabel("K (total votes)")
    ax.set_title("Single-Draft LR Accuracy (K x alpha)")

    for i in range(len(Ks)):
        for j in range(len(alphas_grid)):
            v = acc_mat[i, j]
            c = "white" if v > acc_mat.mean() else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=10, color=c)

    fig.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)
    fig.tight_layout()
    save(fig, "fig_heatmap_accuracy")

    # ── Fig 5: Accuracy by alpha (lines = methods) ──────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    for K in Ks:
        accs = [grid[f"LR_K{K}_a{a:.1f}"]["accuracy"] for a in alphas_grid]
        ax.plot(alphas_grid, accs, "o-", color=k_colors[K],
                linewidth=1.5, markersize=6, label=f"Single-Draft K={K}")

    dd_accs = [dd["results"][f"alpha_{a:.1f}"]["accuracy"] for a in dd_alphas]
    ax.plot(dd_alphas, dd_accs, "^-", color=COLORS["dual_16"],
            linewidth=2, markersize=8, label=f"Dual-Draft K={dd_K}")

    for K in Ks:
        ax.axhline(y=grid[f"FullSC_K{K}"]["accuracy"],
                   linestyle="--", alpha=0.5, color=k_colors[K])
        ax.text(alphas_grid[-1] + 0.02,
                grid[f"FullSC_K{K}"]["accuracy"],
                f"SC K={K}", fontsize=8, va="center", color=k_colors[K])

    ax.axhline(y=greedy_acc, linestyle=":", color=COLORS["greedy"],
               linewidth=1)
    ax.text(alphas_grid[-1] + 0.02, greedy_acc, "Greedy",
            fontsize=8, va="center", color=COLORS["greedy"])

    ax.set_xlabel("alpha (rollback fraction)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Rollback Point")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    ax.set_xlim(alphas_grid[0] - 0.03, alphas_grid[-1] + 0.12)
    fig.tight_layout()
    save(fig, "fig_accuracy_by_alpha")

    # ── Fig 6: Compute multiplier vs accuracy gain ───────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    for K in Ks:
        xs = [grid[f"LR_K{K}_a{a:.1f}"]["tokens_per_question"] / greedy_tpq
              for a in alphas_grid]
        ys = [grid[f"LR_K{K}_a{a:.1f}"]["gain_over_greedy"] * 100
              for a in alphas_grid]
        ax.plot(xs, ys, "o-", color=k_colors[K], linewidth=1.5,
                markersize=6, label=f"Single-Draft K={K}")

    xs_dd = [dd["results"][f"alpha_{a:.1f}"]["tokens_per_question"] / greedy_tpq
             for a in dd_alphas]
    ys_dd = [dd["results"][f"alpha_{a:.1f}"]["gain_over_greedy"] * 100
             for a in dd_alphas]
    ax.plot(xs_dd, ys_dd, "^-", color=COLORS["dual_16"], linewidth=2,
            markersize=8, label=f"Dual-Draft K={dd_K}")

    for K in Ks:
        sc = grid[f"FullSC_K{K}"]
        ax.scatter(sc["tokens_per_question"] / greedy_tpq,
                   sc["gain_over_greedy"] * 100,
                   s=100, marker="s", color=COLORS["full_sc"],
                   edgecolors="black", linewidth=0.5, zorder=8)
        ax.annotate(f"SC K={K}",
                    (sc["tokens_per_question"] / greedy_tpq,
                     sc["gain_over_greedy"] * 100),
                    textcoords="offset points", xytext=(6, -8),
                    fontsize=8, color=COLORS["full_sc"])

    ax.set_xlabel("Compute multiplier (vs greedy)")
    ax.set_ylabel("Accuracy gain over greedy (%)")
    ax.set_title("Accuracy Gain vs Compute Multiplier")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    ax.axhline(y=0, color="black", linewidth=0.5)
    fig.tight_layout()
    save(fig, "fig_gain_vs_multiplier")

    print(f"\nAll figures -> {FIG_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
