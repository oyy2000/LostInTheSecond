#!/usr/bin/env python3
"""
Enhanced figures with clear baselines for draft-alpha sweep.

Usage:
    python scripts/7_10_sweep_figures_v2.py
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
FIG = ROOT / "figures/draft_alpha_sweep_v2"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 9, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

C_GREEDY = "#9e9e9e"
C_SC = "#e74c3c"
C_ND = {2: "#1f78b4", 4: "#33a02c", 8: "#ff7f00"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print(f"  {name}")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    S = json.loads(DATA.read_text("utf-8"))
    results = S["results"]

    g_gsm = S["greedy_acc_gsm8k"]
    g_math = S["greedy_acc_math500"]
    g_all = (g_gsm * S["n_gsm8k"] + g_math * S["n_math500"]) / (
        S["n_gsm8k"] + S["n_math500"]
    )

    for ds_name, g_acc in [("gsm8k", g_gsm), ("math500", g_math),
                            ("all", g_all)]:
        ds_res = [r for r in results if r["dataset"] == ds_name]
        lr = [r for r in ds_res if r["method"] == "LateRollback"]
        sc = sorted(
            [r for r in ds_res if r["method"] == "FullSC"],
            key=lambda r: r["K"],
        )
        if not ds_res:
            continue

        alphas = sorted(set(r["alpha"] for r in lr))
        Ks = sorted(set(r["K"] for r in lr))
        nds = sorted(set(r["n_drafts"] for r in lr))

        ds_label = {"gsm8k": "GSM8K (n=100)",
                     "math500": "MATH500 (n=100)",
                     "all": "Combined (n=200)"}[ds_name]

        # ── Fig 1: Pareto with baselines ─────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 6))

        # Greedy baseline
        # Estimate greedy tpq from drafts
        g_tpq_est = None
        for r in lr:
            if r["tpq"] > 0 and r["multiplier"] > 0:
                g_tpq_est = r["tpq"] / r["multiplier"]
                break
        if g_tpq_est:
            ax.scatter(g_tpq_est, g_acc, s=180, marker="D",
                       color=C_GREEDY, zorder=10, edgecolors="black",
                       linewidth=0.8)
            ax.annotate(f"Greedy\n{g_acc:.1%}",
                        (g_tpq_est, g_acc),
                        textcoords="offset points", xytext=(-40, 10),
                        fontsize=9, fontweight="bold", color=C_GREEDY)

        # Full SC baselines
        for r in sc:
            ax.scatter(r["tpq"], r["accuracy"], s=140, marker="s",
                       color=C_SC, edgecolors="black", linewidth=0.8,
                       zorder=9)
            ax.annotate(f"Full SC K={r['K']}\n{r['accuracy']:.1%}",
                        (r["tpq"], r["accuracy"]),
                        textcoords="offset points", xytext=(8, -5),
                        fontsize=8.5, color=C_SC, fontweight="bold")

        # Late Rollback curves
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
                        linewidth=1.5,
                        label=f"LR nd={nd} K={K}")

        ax.set_xlabel("Tokens per question")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy vs Compute: {ds_label}")
        ax.legend(loc="lower right", fontsize=7.5, ncol=2)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        save(fig, f"fig_pareto_{ds_name}")

        # ── Fig 2: Bar chart — best LR per K vs Full SC ─────────────────
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        # 2a: Accuracy
        ax = axes[0]
        x = np.arange(len(Ks))
        w = 0.15

        # Full SC bars
        sc_accs = [next((r["accuracy"] for r in sc if r["K"] == K), 0)
                   for K in Ks]
        ax.bar(x - 2 * w, sc_accs, w, label="Full SC",
               color=C_SC, edgecolor="black", linewidth=0.5)

        # Best LR per (nd, K)
        for i, nd in enumerate(nds):
            accs = []
            best_labels = []
            for K in Ks:
                sub = [r for r in lr
                       if r["n_drafts"] == nd and r["K"] == K]
                if sub:
                    best = max(sub, key=lambda r: r["accuracy"])
                    accs.append(best["accuracy"])
                    best_labels.append(f"a={best['alpha']}")
                else:
                    accs.append(0)
                    best_labels.append("")
            bars = ax.bar(x + (i - 1) * w, accs, w,
                          label=f"LR nd={nd}",
                          color=C_ND[nd], edgecolor="black", linewidth=0.5)
            for bar, lab in zip(bars, best_labels):
                if bar.get_height() > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.003,
                            lab, ha="center", va="bottom",
                            fontsize=7, rotation=45)

        ax.axhline(y=g_acc, color=C_GREEDY, linestyle=":",
                   linewidth=1.5, label=f"Greedy ({g_acc:.1%})")
        ax.set_xticks(x)
        ax.set_xticklabels([f"K={K}" for K in Ks])
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Best Accuracy per Config: {ds_label}")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)

        # 2b: LTE
        ax = axes[1]
        sc_ltes = [next((r["LTE"] for r in sc if r["K"] == K), 0)
                   for K in Ks]
        ax.bar(x - 2 * w, sc_ltes, w, label="Full SC",
               color=C_SC, edgecolor="black", linewidth=0.5)

        for i, nd in enumerate(nds):
            ltes = []
            best_labels = []
            for K in Ks:
                sub = [r for r in lr
                       if r["n_drafts"] == nd and r["K"] == K]
                if sub:
                    best = max(sub, key=lambda r: r["LTE"])
                    ltes.append(best["LTE"])
                    best_labels.append(f"a={best['alpha']}")
                else:
                    ltes.append(0)
                    best_labels.append("")
            bars = ax.bar(x + (i - 1) * w, ltes, w,
                          label=f"LR nd={nd}",
                          color=C_ND[nd], edgecolor="black", linewidth=0.5)
            for bar, lab in zip(bars, best_labels):
                if bar.get_height() > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.0002,
                            lab, ha="center", va="bottom",
                            fontsize=7, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels([f"K={K}" for K in Ks])
        ax.set_ylabel("LTE Score")
        ax.set_title(f"Best LTE per Config: {ds_label}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()
        save(fig, f"fig_best_per_K_{ds_name}")

        # ── Fig 3: n_drafts effect (K=16, acc + LTE side by side) ───────
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
                        label=f"LR nd={nd}")
        sc16 = [r for r in sc if r["K"] == 16]
        if sc16:
            ax.axhline(y=sc16[0]["accuracy"], color=C_SC, linestyle="--",
                       linewidth=1.5,
                       label=f"Full SC K=16 ({sc16[0]['accuracy']:.1%})")
        ax.axhline(y=g_acc, color=C_GREEDY, linestyle=":",
                   linewidth=1, label=f"Greedy ({g_acc:.1%})")
        ax.set_xlabel("alpha (rollback fraction)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy vs alpha (K=16): {ds_label}")
        ax.legend(fontsize=8.5)
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
                        label=f"LR nd={nd}")
        if sc16:
            ax.axhline(y=sc16[0]["LTE"], color=C_SC, linestyle="--",
                       linewidth=1.5,
                       label=f"Full SC K=16 (LTE={sc16[0]['LTE']:.5f})")
        ax.set_xlabel("alpha (rollback fraction)")
        ax.set_ylabel("LTE Score")
        ax.set_title(f"LTE vs alpha (K=16): {ds_label}")
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.25)

        fig.tight_layout()
        save(fig, f"fig_ndrafts_effect_{ds_name}")

    # ── Cross-dataset: GSM8K vs MATH500 ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax_i, (metric, ylabel, title) in enumerate([
        ("accuracy", "Accuracy", "Accuracy"),
        ("LTE", "LTE Score", "Late-Token Efficiency"),
    ]):
        ax = axes[ax_i]
        for ds, ls, g_a in [("gsm8k", "-", g_gsm),
                             ("math500", "--", g_math)]:
            ds_lr = [r for r in results
                     if r["dataset"] == ds
                     and r["method"] == "LateRollback"
                     and r["K"] == 16]
            ds_sc = [r for r in results
                     if r["dataset"] == ds
                     and r["method"] == "FullSC"
                     and r["K"] == 16]

            for nd in sorted(set(r["n_drafts"] for r in ds_lr)):
                sub = sorted(
                    [r for r in ds_lr if r["n_drafts"] == nd],
                    key=lambda r: r["alpha"],
                )
                if sub:
                    ax.plot([r["alpha"] for r in sub],
                            [r[metric] for r in sub],
                            f"o{ls}", color=C_ND[nd], linewidth=1.8,
                            markersize=6,
                            label=f"{ds} nd={nd}")

            if ds_sc and metric == "accuracy":
                ax.axhline(y=ds_sc[0][metric], color=C_SC,
                           linestyle=":" if ds == "math500" else "-.",
                           alpha=0.6, linewidth=1,
                           label=f"{ds} SC K=16")
            if metric == "accuracy":
                ax.axhline(y=g_a, color=C_GREEDY,
                           linestyle=":" if ds == "math500" else "-.",
                           alpha=0.4, linewidth=1)

        ax.set_xlabel("alpha")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}: GSM8K vs MATH500 (K=16)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    save(fig, "fig_cross_dataset")

    # ── Summary table figure ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")

    headers = ["Dataset", "Method", "nD", "K", "alpha",
               "Accuracy", "Tok/Q", "Gain", "xCost", "LTE"]

    rows = []
    for ds, g_a in [("gsm8k", g_gsm), ("math500", g_math), ("all", g_all)]:
        ds_res = [r for r in results if r["dataset"] == ds]
        rows.append([ds.upper(), "Greedy", "-", "-", "-",
                     f"{g_a:.1%}", "-", "-", "1.0x", "-"])
        for r in sorted([r for r in ds_res if r["method"] == "FullSC"],
                        key=lambda r: r["K"]):
            rows.append([ds, "Full SC", "-", str(r["K"]), "-",
                         f"{r['accuracy']:.1%}", f"{r['tpq']:.0f}",
                         f"{r['gain']:+.1%}", f"{r['multiplier']:.1f}x",
                         f"{r['LTE']:.5f}"])
        top_lr = sorted([r for r in ds_res if r["method"] == "LateRollback"],
                        key=lambda r: -r["LTE"])[:5]
        for r in top_lr:
            rows.append([ds, "LR (best)", str(r["n_drafts"]), str(r["K"]),
                         str(r["alpha"]),
                         f"{r['accuracy']:.1%}", f"{r['tpq']:.0f}",
                         f"{r['gain']:+.1%}", f"{r['multiplier']:.1f}x",
                         f"{r['LTE']:.5f}"])
        rows.append([""] * len(headers))

    table = ax.table(cellText=rows, colLabels=headers,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.3)

    for j in range(len(headers)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i, row in enumerate(rows, start=1):
        if row[1] == "Greedy":
            for j in range(len(headers)):
                table[i, j].set_facecolor("#E2EFDA")
        elif row[1] == "Full SC":
            for j in range(len(headers)):
                table[i, j].set_facecolor("#FCE4EC")
        elif row[1] == "LR (best)":
            for j in range(len(headers)):
                table[i, j].set_facecolor("#E3F2FD")

    ax.set_title("Draft-Alpha Sweep: Summary Table", fontsize=14,
                 fontweight="bold", pad=20)
    fig.tight_layout()
    save(fig, "fig_summary_table")

    print(f"\nAll figures -> {FIG}")
    print("Done.")


if __name__ == "__main__":
    main()
