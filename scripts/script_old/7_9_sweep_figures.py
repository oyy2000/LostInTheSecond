#!/usr/bin/env python3
"""
Figures for draft-alpha sweep: heatmaps, Pareto, LTE ranking.

Usage:
    python scripts/7_9_sweep_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results/draft_alpha_sweep/sweep_summary.json"
FIG = ROOT / "figures/draft_alpha_sweep"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 9, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

C_GREEDY = "#9e9e9e"
C_SC = "#e74c3c"
C_ND = {2: "#1f78b4", 4: "#33a02c", 8: "#ff7f00"}
C_DS = {"gsm8k": "#1f78b4", "math500": "#e74c3c", "all": "#333333"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print(f"  {name}")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    S = json.loads(DATA.read_text("utf-8"))
    results = S["results"]

    for ds_name in ["gsm8k", "math500", "all"]:
        ds_res = [r for r in results if r["dataset"] == ds_name]
        if not ds_res:
            continue

        lr = [r for r in ds_res if r["method"] == "LateRollback"]
        sc = [r for r in ds_res if r["method"] == "FullSC"]

        alphas = sorted(set(r["alpha"] for r in lr))
        Ks = sorted(set(r["K"] for r in lr))
        nds = sorted(set(r["n_drafts"] for r in lr))

        # ── Fig 1: LTE heatmap per n_drafts (rows=K, cols=alpha) ────────
        for nd in nds:
            sub = [r for r in lr if r["n_drafts"] == nd]
            if not sub:
                continue

            mat = np.zeros((len(Ks), len(alphas)))
            for r in sub:
                ki = Ks.index(r["K"])
                ai = alphas.index(r["alpha"])
                mat[ki, ai] = r["LTE"]

            fig, ax = plt.subplots(figsize=(7, 3.5))
            vmax = max(abs(mat.max()), abs(mat.min()), 0.001)
            im = ax.imshow(mat, cmap="RdYlGn", aspect="auto",
                           vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(len(alphas)))
            ax.set_xticklabels([f"{a:.1f}" for a in alphas])
            ax.set_yticks(range(len(Ks)))
            ax.set_yticklabels([str(k) for k in Ks])
            ax.set_xlabel("alpha")
            ax.set_ylabel("K")
            ax.set_title(f"LTE Score: {ds_name}, n_drafts={nd}")
            for i in range(len(Ks)):
                for j in range(len(alphas)):
                    ax.text(j, i, f"{mat[i,j]:.4f}", ha="center",
                            va="center", fontsize=9)
            fig.colorbar(im, ax=ax, label="LTE", shrink=0.8)
            fig.tight_layout()
            save(fig, f"fig_lte_heatmap_{ds_name}_nd{nd}")

        # ── Fig 2: Accuracy heatmap per n_drafts ────────────────────────
        for nd in nds:
            sub = [r for r in lr if r["n_drafts"] == nd]
            if not sub:
                continue

            mat = np.zeros((len(Ks), len(alphas)))
            for r in sub:
                ki = Ks.index(r["K"])
                ai = alphas.index(r["alpha"])
                mat[ki, ai] = r["accuracy"]

            fig, ax = plt.subplots(figsize=(7, 3.5))
            im = ax.imshow(mat, cmap="YlGnBu", aspect="auto")
            ax.set_xticks(range(len(alphas)))
            ax.set_xticklabels([f"{a:.1f}" for a in alphas])
            ax.set_yticks(range(len(Ks)))
            ax.set_yticklabels([str(k) for k in Ks])
            ax.set_xlabel("alpha")
            ax.set_ylabel("K")
            ax.set_title(f"Accuracy: {ds_name}, n_drafts={nd}")
            for i in range(len(Ks)):
                for j in range(len(alphas)):
                    ax.text(j, i, f"{mat[i,j]:.3f}", ha="center",
                            va="center", fontsize=9,
                            color="white" if mat[i,j] > mat.mean()
                            else "black")
            fig.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)
            fig.tight_layout()
            save(fig, f"fig_acc_heatmap_{ds_name}_nd{nd}")

        # ── Fig 3: Pareto frontier (accuracy vs tokens/q) ───────────────
        fig, ax = plt.subplots(figsize=(8, 5.5))

        for nd in nds:
            for K in Ks:
                sub = sorted(
                    [r for r in lr if r["n_drafts"] == nd and r["K"] == K],
                    key=lambda r: r["tpq"],
                )
                if not sub:
                    continue
                xs = [r["tpq"] for r in sub]
                ys = [r["accuracy"] for r in sub]
                ls = "-" if K == 16 else ("--" if K == 8 else ":")
                ax.plot(xs, ys, f"o{ls}", color=C_ND[nd], markersize=5,
                        linewidth=1.5, alpha=0.7 + 0.1 * Ks.index(K),
                        label=f"LR nd={nd} K={K}")

        for r in sc:
            ax.scatter(r["tpq"], r["accuracy"], s=100, marker="s",
                       color=C_SC, edgecolors="black", linewidth=0.5,
                       zorder=8)
            ax.annotate(f"SC K={r['K']}", (r["tpq"], r["accuracy"]),
                        textcoords="offset points", xytext=(5, -10),
                        fontsize=8, color=C_SC)

        ax.set_xlabel("Tokens per question")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Pareto Frontier: {ds_name}")
        ax.legend(loc="lower right", fontsize=7, ncol=2)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        save(fig, f"fig_pareto_{ds_name}")

        # ── Fig 4: LTE bar chart (top 15 configs) ───────────────────────
        top = sorted(ds_res, key=lambda r: -r["LTE"])[:15]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(top))
        colors = []
        labels = []
        for r in top:
            if r["method"] == "FullSC":
                colors.append(C_SC)
                labels.append(f"SC K={r['K']}")
            else:
                colors.append(C_ND.get(r["n_drafts"], "#999"))
                labels.append(f"nd={r['n_drafts']} K={r['K']} a={r['alpha']}")

        bars = ax.bar(x, [r["LTE"] for r in top], color=colors,
                       edgecolor="black", linewidth=0.5)
        for i, (bar, r) in enumerate(zip(bars, top)):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.0002,
                    f"{r['accuracy']:.3f}", ha="center", va="bottom",
                    fontsize=8, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("LTE Score")
        ax.set_title(f"Top 15 Configs by LTE: {ds_name} (labels = accuracy)")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        save(fig, f"fig_lte_ranking_{ds_name}")

        # ── Fig 5: n_drafts effect (fix K=16, vary alpha) ───────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        for nd in nds:
            sub = sorted(
                [r for r in lr if r["n_drafts"] == nd and r["K"] == 16],
                key=lambda r: r["alpha"],
            )
            if sub:
                ax.plot([r["alpha"] for r in sub],
                        [r["accuracy"] for r in sub],
                        "o-", color=C_ND[nd], linewidth=2, markersize=7,
                        label=f"n_drafts={nd}")
        sc16 = [r for r in sc if r["K"] == 16]
        if sc16:
            ax.axhline(y=sc16[0]["accuracy"], color=C_SC, linestyle="--",
                        label=f"Full SC K=16")
        ax.set_xlabel("alpha")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy vs alpha (K=16): {ds_name}")
        ax.legend()
        ax.grid(alpha=0.25)

        ax = axes[1]
        for nd in nds:
            sub = sorted(
                [r for r in lr if r["n_drafts"] == nd and r["K"] == 16],
                key=lambda r: r["alpha"],
            )
            if sub:
                ax.plot([r["alpha"] for r in sub],
                        [r["LTE"] for r in sub],
                        "o-", color=C_ND[nd], linewidth=2, markersize=7,
                        label=f"n_drafts={nd}")
        ax.set_xlabel("alpha")
        ax.set_ylabel("LTE Score")
        ax.set_title(f"LTE vs alpha (K=16): {ds_name}")
        ax.legend()
        ax.grid(alpha=0.25)

        fig.tight_layout()
        save(fig, f"fig_ndrafts_effect_{ds_name}")

    # ── Fig 6: Cross-dataset comparison ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for ds in ["gsm8k", "math500"]:
        ds_lr = [r for r in results
                 if r["dataset"] == ds and r["method"] == "LateRollback"
                 and r["K"] == 16]
        if not ds_lr:
            continue
        for nd in sorted(set(r["n_drafts"] for r in ds_lr)):
            sub = sorted(
                [r for r in ds_lr if r["n_drafts"] == nd],
                key=lambda r: r["alpha"],
            )
            ls = "-" if ds == "gsm8k" else "--"
            ax.plot([r["alpha"] for r in sub],
                    [r["LTE"] for r in sub],
                    f"o{ls}", color=C_ND[nd], linewidth=2, markersize=6,
                    label=f"{ds} nd={nd}")

    ax.set_xlabel("alpha")
    ax.set_ylabel("LTE Score")
    ax.set_title("LTE: GSM8K vs MATH500 (K=16)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "fig_cross_dataset_lte")

    print(f"\nAll figures -> {FIG}")
    print("Done.")


if __name__ == "__main__":
    main()
