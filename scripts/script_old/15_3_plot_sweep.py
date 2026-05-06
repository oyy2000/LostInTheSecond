#!/usr/bin/env python3
"""
Plot sweep results.

Reads sweep_summary.json from each result directory and produces (PNG only):
  1. Accuracy vs total tokens (tpq) line plot
  2. Accuracy gain per 1k extra tokens
  3. Heatmap (nd x K, best alpha)
  4. Cross-dataset summary bar chart

Output dir includes model name: figures/sweep_plots/{model_short}/

Usage:
    python scripts/15_3_plot_sweep.py
    python scripts/15_3_plot_sweep.py --result-dirs results/gsm8k_llama_3.2_3b_instruct_sweep
    python scripts/15_3_plot_sweep.py --out-dir figures/custom
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

C_SC, C_LR, C_GR = "#888888", "#2176FF", "#E63946"
M_SC, M_LR = "s", "o"


def _model_short(model: str) -> str:
    return model.split("/")[-1].lower().replace("-", "_")


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def best_alpha_results(results: list) -> list:
    groups = defaultdict(list)
    for r in results:
        key = (r["method"], r.get("n_drafts", 0), r.get("K", 0))
        groups[key].append(r)
    return [max(rs, key=lambda x: x["accuracy"]) for rs in groups.values()]


def _dedup_by_tpq(pts: list):
    best = {}
    for r in pts:
        t = r.get("tpq", 0)
        if t not in best or r["accuracy"] > best[t]["accuracy"]:
            best[t] = r
    tpqs = sorted(best.keys())
    return tpqs, [best[t]["accuracy"] for t in tpqs]


def _greedy_info(summary: dict):
    ga = summary.get("greedy_acc", summary.get("greedy_accuracy", 0))
    gt = 0.0
    for r in summary.get("results", []):
        if r["method"] == "Greedy":
            gt = r.get("tpq", 0)
            break
    return ga, gt


# -- Plot 1: accuracy vs tokens -------------------------------------------

def plot_acc_vs_tokens(summary: dict, out_dir: Path):
    model = summary.get("model", "?")
    dataset = summary.get("dataset", "?")
    greedy_acc, greedy_tpq = _greedy_info(summary)
    results = best_alpha_results(summary.get("results", []))

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, color, marker, label in [
        ("FullSC", C_SC, M_SC, "Full SC"),
        ("LateRollback", C_LR, M_LR, "Late Rollback"),
    ]:
        pts = [r for r in results if r["method"] == method and r.get("tpq", 0) > 0]
        if not pts:
            continue
        tpqs, accs = _dedup_by_tpq(pts)
        ax.plot(tpqs, accs, marker=marker, color=color,
                label=label, linewidth=2, markersize=6)

    if greedy_tpq > 0:
        ax.axvline(greedy_tpq, color=C_GR, linestyle=":", alpha=0.5, linewidth=1)
    ax.axhline(greedy_acc, color=C_GR, linestyle="--",
               linewidth=1.5, label=f"Greedy ({greedy_acc:.3f})")
    ax.set_xlabel("Tokens per question", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"{dataset} -- {model.split('/')[-1]}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    stem = f"fig_acc_vs_tokens_{dataset.lower()}"
    fig.savefig(out_dir / f"{stem}.png", dpi=200)
    plt.close(fig)
    print(f"  {stem}.png")


# -- Plot 2: gain per 1k extra tokens -------------------------------------

def plot_gain_per_1k(summary: dict, out_dir: Path):
    model = summary.get("model", "?")
    dataset = summary.get("dataset", "?")
    greedy_acc, greedy_tpq = _greedy_info(summary)
    results = best_alpha_results(summary.get("results", []))

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, color, marker, label in [
        ("FullSC", C_SC, M_SC, "Full SC"),
        ("LateRollback", C_LR, M_LR, "Late Rollback"),
    ]:
        pts = [r for r in results if r["method"] == method and r.get("tpq", 0) > 0]
        if not pts:
            continue
        xs, ys = [], []
        for r in sorted(pts, key=lambda x: x.get("tpq", 0)):
            extra = r["tpq"] - greedy_tpq
            if extra <= 0:
                continue
            gain = r["accuracy"] - greedy_acc
            xs.append(r["tpq"])
            ys.append(gain / (extra / 1000.0))
        if xs:
            ax.plot(xs, ys, marker=marker, color=color,
                    label=label, linewidth=2, markersize=6)

    ax.axhline(0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Tokens per question", fontsize=12)
    ax.set_ylabel("Acc gain per 1k extra tokens", fontsize=12)
    ax.set_title(f"{dataset} -- {model.split('/')[-1]}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    stem = f"fig_gain_per_1k_{dataset.lower()}"
    fig.savefig(out_dir / f"{stem}.png", dpi=200)
    plt.close(fig)
    print(f"  {stem}.png")


# -- Plot 3: heatmap (nd x K) ---------------------------------------------

def plot_heatmap(summary: dict, out_dir: Path):
    model = summary.get("model", "?")
    dataset = summary.get("dataset", "?")
    lr = [r for r in summary.get("results", []) if r["method"] == "LateRollback"]
    if not lr:
        return

    nds = sorted(set(r["n_drafts"] for r in lr))
    Ks = sorted(set(r["K"] for r in lr))
    if not nds or not Ks:
        return

    grid = np.full((len(nds), len(Ks)), np.nan)
    alpha_grid = {}
    for r in lr:
        ri, ci = nds.index(r["n_drafts"]), Ks.index(r["K"])
        if np.isnan(grid[ri, ci]) or r["accuracy"] > grid[ri, ci]:
            grid[ri, ci] = r["accuracy"]
            alpha_grid[(ri, ci)] = r.get("alpha", "?")

    fig, ax = plt.subplots(figsize=(max(4, len(Ks) * 1.2), max(3, len(nds) * 0.7)))
    im = ax.imshow(grid, cmap="YlGnBu", aspect="auto")
    for i in range(len(nds)):
        for j in range(len(Ks)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}\na={alpha_grid.get((i,j),'')}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(Ks)))
    ax.set_xticklabels([str(k) for k in Ks])
    ax.set_yticks(range(len(nds)))
    ax.set_yticklabels([str(n) for n in nds])
    ax.set_xlabel("K (per-draft budget)", fontsize=11)
    ax.set_ylabel("n_drafts", fontsize=11)
    ax.set_title(f"{dataset} -- {model.split('/')[-1]} (best alpha)", fontsize=12)
    fig.colorbar(im, ax=ax, label="Accuracy")
    fig.tight_layout()
    stem = f"fig_heatmap_{dataset.lower()}"
    fig.savefig(out_dir / f"{stem}.png", dpi=200)
    plt.close(fig)
    print(f"  {stem}.png")


# -- Plot 4: cross-dataset bar chart ---------------------------------------

def plot_cross_dataset_bar(summaries: list, out_dir: Path):
    if not summaries:
        return
    datasets, g_accs, sc_accs, lr_accs = [], [], [], []
    for s in summaries:
        datasets.append(s.get("dataset", "?"))
        ga = s.get("greedy_acc", s.get("greedy_accuracy", 0))
        g_accs.append(ga)
        rs = s.get("results", [])
        sc_accs.append(max((r["accuracy"] for r in rs if r["method"] == "FullSC"), default=ga))
        lr_accs.append(max((r["accuracy"] for r in rs if r["method"] == "LateRollback"), default=ga))

    x = np.arange(len(datasets))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 1.5), 5))
    ax.bar(x - w, g_accs, w, label="Greedy", color=C_GR, alpha=0.85)
    ax.bar(x, sc_accs, w, label="Full SC (best)", color=C_SC, alpha=0.85)
    ax.bar(x + w, lr_accs, w, label="Late Rollback (best)", color=C_LR, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=12)
    mn = summaries[0].get("model", "?").split("/")[-1]
    ax.set_title(f"Cross-dataset -- {mn}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_cross_dataset_bar.png", dpi=200)
    plt.close(fig)
    print(f"  fig_cross_dataset_bar.png")


# -- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dirs", nargs="*", default=None)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    if args.result_dirs:
        dirs = [Path(d) for d in args.result_dirs]
    else:
        dirs = sorted(ROOT.glob("results/*_sweep"))

    summaries = []
    for d in dirs:
        sf = d / "sweep_summary.json"
        if not sf.exists():
            print(f"SKIP {d.name}")
            continue
        print(f"Loading {d.name}")
        summaries.append(load_summary(sf))

    if not summaries:
        print("No summaries found.")
        return

    by_model = defaultdict(list)
    for s in summaries:
        by_model[s.get("model", "unknown")].append(s)

    for model, ss in by_model.items():
        ms = _model_short(model)
        out_dir = Path(args.out_dir) / ms if args.out_dir else ROOT / "figures" / "sweep_plots" / ms
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {model} -> {out_dir} ===")

        for s in ss:
            plot_acc_vs_tokens(s, out_dir)
            plot_gain_per_1k(s, out_dir)
            plot_heatmap(s, out_dir)

        if len(ss) > 1:
            plot_cross_dataset_bar(ss, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
