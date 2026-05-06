#!/usr/bin/env python3
"""
Figures for MATH500 full sweep.

Usage:
    python scripts/7_12_math500_sweep_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results/math500_full_sweep/sweep_summary.json"
FIG = ROOT / "figures/math500_full_sweep"

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

    # ── Fig 1: Pareto frontier ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(g_tpq, g_acc, s=180, marker="D", color=C_GREEDY,
               zorder=10, edgecolors="black", linewidth=0.8)
    ax.annotate(f"Greedy\n{g_acc:.1%}", (g_tpq, g_acc),
                textcoords="offset points", xytext=(-50, 10),
                fontsize=9, fontweight="bold", color=C_GREEDY)

    for r in sc:
        ax.scatter(r["tpq"], r["accuracy"], s=140, marker="s",
                   color=C_SC, edgecolors="black", linewidth=0.8, zorder=9)
        ax.annotate(f"SC K={r['K']}\n{r['accuracy']:.1%}",
                    (r["tpq"], r["accuracy"]),
                    textcoords="offset points", xytext=(8, -5),
                    fontsize=8.5, color=C_SC, fontweight="bold")

    for nd in nds:
        for K in Ks:
            sub = sorted([r for r in lr if r["n_drafts"] == nd and r["K"] == K],
                         key=lambda r: r["tpq"])
            if not sub: continue
            ls = "-" if K == 16 else ("--" if K == 8 else ":")
            ax.plot([r["tpq"] for r in sub], [r["accuracy"] for r in sub],
                    f"o{ls}", color=C_ND[nd], markersize=5, linewidth=1.5,
                    label=f"LR nd={nd} K={K}")

    ax.set_xlabel("Tokens per question")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"MATH500: Accuracy vs Compute (n={S['n_questions']})")
    ax.legend(loc="lower right", fontsize=7.5, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "fig_pareto")

    # ── Fig 2: Accuracy by alpha (lines = nd, panels = K) ───────────────
    fig, axes = plt.subplots(1, len(Ks), figsize=(5 * len(Ks), 5), sharey=True)
    if len(Ks) == 1: axes = [axes]

    for ax, K in zip(axes, Ks):
        for nd in nds:
            sub = sorted([r for r in lr if r["n_drafts"] == nd and r["K"] == K],
                         key=lambda r: r["alpha"])
            if sub:
                ax.plot([r["alpha"] for r in sub], [r["accuracy"] for r in sub],
                        "o-", color=C_ND[nd], linewidth=2, markersize=7,
                        label=f"LR nd={nd}")
        sc_k = [r for r in sc if r["K"] == K]
        if sc_k:
            ax.axhline(y=sc_k[0]["accuracy"], color=C_SC, linestyle="--",
                       linewidth=1.5, label=f"Full SC K={K} ({sc_k[0]['accuracy']:.1%})")
        ax.axhline(y=g_acc, color=C_GREEDY, linestyle=":", linewidth=1,
                   label=f"Greedy ({g_acc:.1%})")
        ax.set_xlabel("alpha")
        ax.set_title(f"K={K}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Accuracy")
    fig.suptitle("MATH500: Accuracy vs Alpha", fontsize=14, y=1.02)
    fig.tight_layout()
    save(fig, "fig_accuracy_by_alpha")

    # ── Fig 3: LTE by alpha (lines = nd, panels = K) ────────────────────
    fig, axes = plt.subplots(1, len(Ks), figsize=(5 * len(Ks), 5), sharey=True)
    if len(Ks) == 1: axes = [axes]

    for ax, K in zip(axes, Ks):
        for nd in nds:
            sub = sorted([r for r in lr if r["n_drafts"] == nd and r["K"] == K],
                         key=lambda r: r["alpha"])
            if sub:
                ax.plot([r["alpha"] for r in sub], [r["LTE"] for r in sub],
                        "o-", color=C_ND[nd], linewidth=2, markersize=7,
                        label=f"LR nd={nd}")
        sc_k = [r for r in sc if r["K"] == K]
        if sc_k:
            ax.axhline(y=sc_k[0]["LTE"], color=C_SC, linestyle="--",
                       linewidth=1.5, label=f"Full SC K={K}")
        ax.set_xlabel("alpha")
        ax.set_title(f"K={K}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("LTE Score")
    fig.suptitle("MATH500: Late-Token Efficiency vs Alpha", fontsize=14, y=1.02)
    fig.tight_layout()
    save(fig, "fig_lte_by_alpha")

    # ── Fig 4: Best config bar chart (accuracy + LTE) ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # 4a: Accuracy
    ax = axes[0]
    x = np.arange(len(Ks)); w = 0.18
    sc_accs = [next((r["accuracy"] for r in sc if r["K"] == K), 0) for K in Ks]
    ax.bar(x - 1.5 * w, sc_accs, w, label="Full SC", color=C_SC,
           edgecolor="black", linewidth=0.5)
    for i, nd in enumerate(nds):
        accs, labs = [], []
        for K in Ks:
            sub = [r for r in lr if r["n_drafts"] == nd and r["K"] == K]
            if sub:
                best = max(sub, key=lambda r: r["accuracy"])
                accs.append(best["accuracy"]); labs.append(f"a={best['alpha']}")
            else: accs.append(0); labs.append("")
        bars = ax.bar(x + (i - 0.5) * w, accs, w, label=f"LR nd={nd}",
                      color=C_ND[nd], edgecolor="black", linewidth=0.5)
        for bar, lab in zip(bars, labs):
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003, lab,
                        ha="center", va="bottom", fontsize=7, rotation=45)
    ax.axhline(y=g_acc, color=C_GREEDY, linestyle=":", linewidth=1.5,
               label=f"Greedy ({g_acc:.1%})")
    ax.set_xticks(x); ax.set_xticklabels([f"K={K}" for K in Ks])
    ax.set_ylabel("Accuracy"); ax.set_title("Best Accuracy per Config")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)

    # 4b: LTE
    ax = axes[1]
    sc_ltes = [next((r["LTE"] for r in sc if r["K"] == K), 0) for K in Ks]
    ax.bar(x - 1.5 * w, sc_ltes, w, label="Full SC", color=C_SC,
           edgecolor="black", linewidth=0.5)
    for i, nd in enumerate(nds):
        ltes, labs = [], []
        for K in Ks:
            sub = [r for r in lr if r["n_drafts"] == nd and r["K"] == K]
            if sub:
                best = max(sub, key=lambda r: r["LTE"])
                ltes.append(best["LTE"]); labs.append(f"a={best['alpha']}")
            else: ltes.append(0); labs.append("")
        bars = ax.bar(x + (i - 0.5) * w, ltes, w, label=f"LR nd={nd}",
                      color=C_ND[nd], edgecolor="black", linewidth=0.5)
        for bar, lab in zip(bars, labs):
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.0002, lab,
                        ha="center", va="bottom", fontsize=7, rotation=45)
    ax.set_xticks(x); ax.set_xticklabels([f"K={K}" for K in Ks])
    ax.set_ylabel("LTE Score"); ax.set_title("Best LTE per Config")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    save(fig, "fig_best_per_K")

    # ── Fig 5: LTE ranking top 15 ───────────────────────────────────────
    top = sorted(results, key=lambda r: -r["LTE"])[:15]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(top))
    colors, labels = [], []
    for r in top:
        if r["method"] == "FullSC":
            colors.append(C_SC); labels.append(f"SC K={r['K']}")
        else:
            colors.append(C_ND.get(r["n_drafts"], "#999"))
            labels.append(f"nd={r['n_drafts']} K={r['K']} a={r['alpha']}")
    bars = ax.bar(x, [r["LTE"] for r in top], color=colors,
                  edgecolor="black", linewidth=0.5)
    for i, (bar, r) in enumerate(zip(bars, top)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0002,
                f"{r['accuracy']:.1%}", ha="center", va="bottom",
                fontsize=8, rotation=45)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("LTE Score")
    ax.set_title("MATH500: Top 15 Configs by LTE (labels = accuracy)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, "fig_lte_ranking")

    # ── Fig 6: Heatmaps (accuracy + LTE) per n_drafts ───────────────────
    for nd in nds:
        sub = [r for r in lr if r["n_drafts"] == nd]
        if not sub: continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        for ax_i, (metric, cmap, label) in enumerate([
            ("accuracy", "YlGnBu", "Accuracy"),
            ("LTE", "RdYlGn", "LTE"),
        ]):
            ax = axes[ax_i]
            mat = np.zeros((len(Ks), len(alphas)))
            for r in sub:
                ki = Ks.index(r["K"]); ai = alphas.index(r["alpha"])
                mat[ki, ai] = r[metric]

            if metric == "LTE":
                vmax = max(abs(mat.max()), abs(mat.min()), 0.001)
                im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
            else:
                im = ax.imshow(mat, cmap=cmap, aspect="auto")

            ax.set_xticks(range(len(alphas)))
            ax.set_xticklabels([f"{a:.1f}" for a in alphas])
            ax.set_yticks(range(len(Ks)))
            ax.set_yticklabels([str(k) for k in Ks])
            ax.set_xlabel("alpha"); ax.set_ylabel("K")
            ax.set_title(f"{label}: nd={nd}")

            fmt_str = ".3f" if metric == "accuracy" else ".5f"
            for i in range(len(Ks)):
                for j in range(len(alphas)):
                    c = "white" if mat[i, j] > mat.mean() else "black"
                    ax.text(j, i, f"{mat[i,j]:{fmt_str}}", ha="center",
                            va="center", fontsize=9, color=c)
            fig.colorbar(im, ax=ax, label=label, shrink=0.8)

        fig.suptitle(f"MATH500: n_drafts={nd}", fontsize=13)
        fig.tight_layout()
        save(fig, f"fig_heatmap_nd{nd}")

    print(f"\nAll figures -> {FIG}")
    print("Done.")


if __name__ == "__main__":
    main()
