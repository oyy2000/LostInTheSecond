#!/usr/bin/env python3
"""
Figures for HotpotQA full sweep.

Usage:
    python scripts/10_8_hotpotqa_sweep_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results/hotpotqa_full_sweep/sweep_summary.json"
FIG = ROOT / "figures/hotpotqa_full_sweep"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 9, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

C_GREEDY = "#9e9e9e"
C_SC = "#e74c3c"
C_ND = {1: "#6a3d9a", 2: "#1f78b4", 4: "#33a02c"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print(f"  {name}")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    S = json.loads(DATA.read_text("utf-8"))
    results = S["results"]
    g_acc = S["greedy_accuracy"]
    g_tpq = S["greedy_tpq"]

    lr = [r for r in results if r["method"] == "LateRollback"]
    sc = sorted([r for r in results if r["method"] == "FullSC"], key=lambda r: r["K"])
    alphas = sorted(set(r["alpha"] for r in lr))
    Ks = sorted(set(r["K"] for r in lr))
    nds = sorted(set(r["n_drafts"] for r in lr))

    # -- Fig 1: Pareto frontier (Accuracy vs Cost) -------------------------
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(g_tpq, g_acc, s=180, marker="D", color=C_GREEDY,
               zorder=10, edgecolors="black", linewidth=0.8)
    ax.annotate(f"Greedy\n{g_acc:.1%}", (g_tpq, g_acc),
                textcoords="offset points", xytext=(12, -5),
                fontsize=9, color=C_GREEDY)

    for r in sc:
        ax.scatter(r["tpq"], r["accuracy"], s=140, marker="s",
                   color=C_SC, zorder=8, edgecolors="black", linewidth=0.6)
        ax.annotate(f"SC K={r['K']}\n{r['accuracy']:.1%}",
                    (r["tpq"], r["accuracy"]),
                    textcoords="offset points", xytext=(10, 2),
                    fontsize=8, color=C_SC)

    for nd in nds:
        sub = sorted([r for r in lr if r["n_drafts"] == nd],
                     key=lambda r: r["tpq"])
        xs = [r["tpq"] for r in sub]
        ys = [r["accuracy"] for r in sub]
        ax.scatter(xs, ys, s=50, color=C_ND[nd], alpha=0.6, zorder=5,
                   label=f"LR nD={nd}")

    ax.set_xlabel("Tokens per question")
    ax.set_ylabel("Exact Match")
    ax.set_title("HotpotQA: Accuracy vs Compute Cost")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    save(fig, "fig_pareto")

    # -- Fig 2: LTE bar chart (best per method/K) --------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    x_labels, x_pos = [], []
    bar_vals, bar_colors = [], []
    pos = 0

    for K in Ks:
        sc_r = [r for r in sc if r["K"] == K][0]
        x_labels.append(f"SC\nK={K}")
        x_pos.append(pos)
        bar_vals.append(sc_r["LTE"])
        bar_colors.append(C_SC)
        pos += 1

        for nd in nds:
            best = max(
                [r for r in lr if r["n_drafts"] == nd and r["K"] == K],
                key=lambda r: r["LTE"],
            )
            x_labels.append(f"LR\nnD={nd}\na={best['alpha']}")
            x_pos.append(pos)
            bar_vals.append(best["LTE"])
            bar_colors.append(C_ND[nd])
            pos += 1
        pos += 0.5

    bars = ax.bar(x_pos, bar_vals, color=bar_colors, width=0.7,
                  edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, bar_vals):
        ax.text(b.get_x() + b.get_width() / 2, v,
                f"{v:.4f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("LTE (gain / cost multiplier)")
    ax.set_title("HotpotQA: LTE by Method and K")
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, "fig_lte_bar")

    # -- Fig 3: Accuracy vs alpha (one subplot per K) ----------------------
    fig, axes = plt.subplots(1, len(Ks), figsize=(5 * len(Ks), 5), sharey=True)
    if len(Ks) == 1:
        axes = [axes]

    for ax, K in zip(axes, Ks):
        sc_r = [r for r in sc if r["K"] == K][0]
        ax.axhline(g_acc, color=C_GREEDY, ls="--", lw=1.2, label="Greedy")
        ax.axhline(sc_r["accuracy"], color=C_SC, ls="--", lw=1.2,
                   label=f"FullSC K={K}")

        for nd in nds:
            sub = sorted(
                [r for r in lr if r["n_drafts"] == nd and r["K"] == K],
                key=lambda r: r["alpha"],
            )
            ax.plot([r["alpha"] for r in sub],
                    [r["accuracy"] for r in sub],
                    "o-", color=C_ND[nd], label=f"LR nD={nd}", markersize=5)

        ax.set_xlabel("alpha")
        ax.set_title(f"K = {K}")
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("Exact Match")
            ax.legend(fontsize=8)

    fig.suptitle("HotpotQA: Accuracy vs Alpha", fontsize=14)
    fig.tight_layout()
    save(fig, "fig_acc_vs_alpha")

    # -- Fig 4: Heatmaps (Accuracy + LTE per n_drafts) --------------------
    for nd in nds:
        sub = [r for r in lr if r["n_drafts"] == nd]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, metric, label, fmt_str in [
            (axes[0], "accuracy", "Exact Match", ".3f"),
            (axes[1], "LTE", "LTE", ".5f"),
        ]:
            mat = np.zeros((len(alphas), len(Ks)))
            for r in sub:
                ai = alphas.index(r["alpha"])
                ki = Ks.index(r["K"])
                mat[ai, ki] = r[metric]

            im = ax.imshow(mat, aspect="auto", cmap="RdYlGn")
            ax.set_xticks(range(len(Ks)))
            ax.set_xticklabels([str(k) for k in Ks])
            ax.set_yticks(range(len(alphas)))
            ax.set_yticklabels([str(a) for a in alphas])
            ax.set_xlabel("K")
            ax.set_ylabel("alpha")
            ax.set_title(label)

            for i in range(len(alphas)):
                for j in range(len(Ks)):
                    c = "white" if mat[i, j] > mat.mean() else "black"
                    ax.text(j, i, f"{mat[i,j]:{fmt_str}}", ha="center",
                            va="center", fontsize=9, color=c)
            fig.colorbar(im, ax=ax, label=label, shrink=0.8)

        fig.suptitle(f"HotpotQA: n_drafts={nd}", fontsize=13)
        fig.tight_layout()
        save(fig, f"fig_heatmap_nd{nd}")

    # -- Fig 5: Cross-dataset comparison (HotpotQA vs MATH500 vs CSQA) -----
    datasets = []
    for name, path in [
        ("MATH500", ROOT / "results/math500_full_sweep/sweep_summary.json"),
        ("CSQA", ROOT / "results/csqa_full_sweep/sweep_summary.json"),
        ("HotpotQA", DATA),
    ]:
        if path.exists():
            datasets.append((name, json.loads(path.read_text("utf-8"))))

    if len(datasets) >= 2:
        fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5),
                                 sharey=False)
        if len(datasets) == 1:
            axes = [axes]

        for ax, (dname, ds) in zip(axes, datasets):
            d_lr = [r for r in ds["results"] if r["method"] == "LateRollback"]
            d_sc = [r for r in ds["results"] if r["method"] == "FullSC"]
            d_Ks = sorted(set(r["K"] for r in d_lr))

            x = np.arange(len(d_Ks))
            w = 0.18

            sc_vals = []
            for K in d_Ks:
                r = [r for r in d_sc if r["K"] == K][0]
                sc_vals.append(r["accuracy"])
            ax.bar(x - 1.5 * w, sc_vals, w, color=C_SC, label="FullSC",
                   edgecolor="black", linewidth=0.4)

            for ni, nd in enumerate(sorted(set(r["n_drafts"] for r in d_lr))):
                best_vals = []
                for K in d_Ks:
                    best = max(
                        [r for r in d_lr if r["n_drafts"] == nd and r["K"] == K],
                        key=lambda r: r["accuracy"],
                    )
                    best_vals.append(best["accuracy"])
                ax.bar(x + (ni - 0.5) * w, best_vals, w, color=C_ND.get(nd, "#888"),
                       label=f"LR nD={nd}", edgecolor="black", linewidth=0.4)

            ax.axhline(ds["greedy_accuracy"], color=C_GREEDY, ls="--", lw=1,
                       label="Greedy")
            ax.set_xticks(x)
            ax.set_xticklabels([f"K={k}" for k in d_Ks])
            ax.set_title(dname)
            ax.set_ylabel("Accuracy")
            ax.legend(fontsize=7, loc="upper left")
            ax.grid(True, axis="y", alpha=0.3)

        fig.suptitle("Cross-Dataset: Best LR vs FullSC", fontsize=14)
        fig.tight_layout()
        save(fig, "fig_cross_dataset")

    print(f"\nAll figures -> {FIG}")
    print("Done.")


if __name__ == "__main__":
    main()
