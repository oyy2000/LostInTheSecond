#!/usr/bin/env python3
"""
Phase 25-2: Alpha-threshold efficiency analysis for Llama 3.2 3B.

Reads sweep_summary.json from each dataset's alpha sweep and computes
the efficiency metric: acc_gain / (extra_tokens / 1000).

Produces:
  - Per-dataset line plots: efficiency vs alpha for each (nd, K) config
  - Aggregated bar chart: best efficiency across datasets per alpha
  - Summary JSON with all computed metrics

Usage:
    python scripts/25_2_alpha_efficiency_figures.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures" / "alpha_efficiency"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_SHORT = "llama_3.2_3b_instruct"
DATASETS = [
    ("gsm8k", "GSM8K"),
    ("math500", "MATH500"),
    ("hotpotqa", "HotpotQA"),
    ("aime2024", "AIME 2024"),
    ("amc2023", "AMC 2023"),
    ("olympiadbench", "OlympiadBench"),
    ("humaneval", "HumanEval"),
    ("csqa", "CSQA"),
]

COLORS_ND = {2: "#4C72B0", 4: "#DD8452", 8: "#55A868"}
MARKERS_K = {2: "o", 3: "s", 4: "^"}


def load_summary(ds_key: str) -> dict | None:
    p = ROOT / "results" / MODEL_SHORT / ds_key / "sweep_summary.json"
    if not p.exists():
        # fallback to old flat layout
        p = ROOT / "results" / f"{ds_key}_{MODEL_SHORT}_sweep" / "sweep_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text("utf-8"))


def compute_efficiency(summary: dict) -> list[dict]:
    """Compute acc_gain_per_1k_token for each LateRollback config."""
    greedy_acc = summary["greedy_acc"]
    greedy_tpq = None
    for r in summary["results"]:
        if r["method"] == "Greedy":
            greedy_tpq = r["tpq"]
            break
    if greedy_tpq is None:
        return []

    rows = []
    for r in summary["results"]:
        if r["method"] != "LateRollback":
            continue
        extra_tokens = r["tpq"] - greedy_tpq
        gain = r["accuracy"] - greedy_acc
        eff = (gain / (extra_tokens / 1000.0)) if extra_tokens > 0 else 0.0
        rows.append({
            "dataset": summary["dataset"],
            "n_drafts": r["n_drafts"],
            "K": r["K"],
            "alpha": r["alpha"],
            "budget": r["budget"],
            "accuracy": r["accuracy"],
            "gain": gain,
            "tpq": r["tpq"],
            "extra_tokens": extra_tokens,
            "gain_per_1k": round(eff, 6),
        })

    for r in summary["results"]:
        if r["method"] != "FullSC":
            continue
        extra_tokens = r["tpq"] - greedy_tpq
        gain = r["accuracy"] - greedy_acc
        eff = (gain / (extra_tokens / 1000.0)) if extra_tokens > 0 else 0.0
        rows.append({
            "dataset": summary["dataset"],
            "method": "FullSC",
            "n_drafts": 0,
            "K": r.get("budget", r["K"]),
            "alpha": "-",
            "budget": r.get("budget", r["K"]),
            "accuracy": r["accuracy"],
            "gain": gain,
            "tpq": r["tpq"],
            "extra_tokens": extra_tokens,
            "gain_per_1k": round(eff, 6),
        })
    return rows


def plot_per_dataset(ds_key: str, ds_label: str, rows: list[dict], ax):
    """Line plot: alpha (x) vs gain_per_1k (y), one line per (nd, K)."""
    lr_rows = [r for r in rows if r.get("method") != "FullSC"]
    groups = defaultdict(list)
    for r in lr_rows:
        groups[(r["n_drafts"], r["K"])].append(r)

    for (nd, K), pts in sorted(groups.items()):
        pts = sorted(pts, key=lambda x: x["alpha"])
        alphas = [p["alpha"] for p in pts]
        effs = [p["gain_per_1k"] for p in pts]
        c = COLORS_ND.get(nd, "gray")
        m = MARKERS_K.get(K, "D")
        ax.plot(alphas, effs, color=c, marker=m, markersize=5,
                lw=1.5, alpha=0.85, label=f"nd={nd}, K={K}")

    ax.set_title(ds_label, fontsize=11, fontweight="bold")
    ax.set_xlabel(r"$\alpha$ (rollback fraction)", fontsize=9)
    ax.set_ylabel("Acc Gain / 1k extra tokens", fontsize=9)
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.legend(fontsize=6, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)


def plot_aggregate_bar(all_rows: list[dict]):
    """Bar chart: for each alpha, show mean best-efficiency across datasets."""
    alphas_set = sorted(set(
        r["alpha"] for r in all_rows
        if r.get("method") != "FullSC" and isinstance(r["alpha"], float)
    ))
    if not alphas_set:
        return

    ds_keys = sorted(set(r["dataset"] for r in all_rows))
    best_by_alpha_ds = defaultdict(dict)
    for r in all_rows:
        if r.get("method") == "FullSC" or not isinstance(r["alpha"], float):
            continue
        ds = r["dataset"]
        a = r["alpha"]
        if ds not in best_by_alpha_ds[a] or r["gain_per_1k"] > best_by_alpha_ds[a][ds]:
            best_by_alpha_ds[a][ds] = r["gain_per_1k"]

    means, stds = [], []
    for a in alphas_set:
        vals = [best_by_alpha_ds[a].get(ds, 0) for ds in ds_keys]
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(alphas_set))
    bars = ax.bar(x, means, yerr=stds, width=0.5, color="#4C72B0",
                  edgecolor="white", capsize=3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:.1f}" for a in alphas_set])
    ax.set_xlabel(r"$\alpha$ (rollback fraction)", fontsize=11)
    ax.set_ylabel("Acc Gain / 1k extra tokens\n(mean best across datasets)", fontsize=10)
    ax.set_title(f"Llama-3.2-3B: Efficiency by Alpha Threshold", fontsize=12,
                 fontweight="bold")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_alpha_efficiency_bar.{ext}", dpi=200,
                    bbox_inches="tight")
    print(f"Saved fig_alpha_efficiency_bar to {FIG_DIR}")
    plt.close(fig)


def plot_lr_vs_sc(all_rows: list[dict]):
    """Grouped bar: LR best vs FullSC at matched budgets, per dataset."""
    ds_keys = sorted(set(r["dataset"] for r in all_rows))
    ds_labels = {k: l for k, l in DATASETS}

    lr_best = {}
    sc_best = {}
    for r in all_rows:
        ds = r["dataset"]
        if r.get("method") == "FullSC":
            if ds not in sc_best or r["gain_per_1k"] > sc_best[ds]["gain_per_1k"]:
                sc_best[ds] = r
        elif isinstance(r["alpha"], float):
            if ds not in lr_best or r["gain_per_1k"] > lr_best[ds]["gain_per_1k"]:
                lr_best[ds] = r

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(ds_keys))
    w = 0.35
    lr_vals = [lr_best.get(ds, {}).get("gain_per_1k", 0) for ds in ds_keys]
    sc_vals = [sc_best.get(ds, {}).get("gain_per_1k", 0) for ds in ds_keys]
    ax.bar(x - w / 2, lr_vals, w, label="Late Rollback (best)", color="#55A868",
           edgecolor="white", alpha=0.85)
    ax.bar(x + w / 2, sc_vals, w, label="Full SC (best)", color="#4C72B0",
           edgecolor="white", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([ds_labels.get(ds, ds) for ds in ds_keys], fontsize=9)
    ax.set_ylabel("Acc Gain / 1k extra tokens", fontsize=10)
    ax.set_title("Llama-3.2-3B: LR vs SC Efficiency", fontsize=12, fontweight="bold")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_lr_vs_sc_efficiency.{ext}", dpi=200,
                    bbox_inches="tight")
    print(f"Saved fig_lr_vs_sc_efficiency to {FIG_DIR}")
    plt.close(fig)


def main():
    all_rows = []
    summaries = {}

    for ds_key, ds_label in DATASETS:
        s = load_summary(ds_key)
        if s is None:
            print(f"[SKIP] {ds_key}: no sweep_summary.json")
            continue
        summaries[ds_key] = s
        rows = compute_efficiency(s)
        all_rows.extend(rows)
        print(f"[OK] {ds_key}: {len(rows)} configs, greedy_acc={s['greedy_acc']}")

    if not all_rows:
        print("No data found. Run 25_1_alpha_sweep_llama.py first.")
        sys.exit(1)

    # Per-dataset panels
    active = [(k, l) for k, l in DATASETS if k in summaries]
    ncols = min(4, len(active))
    nrows = (len(active) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (ds_key, ds_label) in enumerate(active):
        ds_rows = [r for r in all_rows if r["dataset"] == ds_key]
        plot_per_dataset(ds_key, ds_label, ds_rows, axes[i])
    for j in range(len(active), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Llama-3.2-3B: Acc Gain per 1k Extra Tokens by Alpha",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_alpha_efficiency_panels.{ext}", dpi=200,
                    bbox_inches="tight")
    print(f"Saved fig_alpha_efficiency_panels to {FIG_DIR}")
    plt.close(fig)

    # Aggregate bar
    plot_aggregate_bar(all_rows)

    # LR vs SC comparison
    plot_lr_vs_sc(all_rows)

    # Save summary JSON
    out_json = ROOT / "results" / "alpha_efficiency_llama3b.json"
    out_json.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False), "utf-8")
    print(f"Saved metrics -> {out_json}")


if __name__ == "__main__":
    main()
