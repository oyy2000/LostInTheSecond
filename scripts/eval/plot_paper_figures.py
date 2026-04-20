#!/usr/bin/env python3
"""
Generate paper-ready figures for the prefix correction experiment (v2).

Figures:
  1. IG vs correctness dual plot (correct, wrong, corrected, corrupted)
  2. Separability heatmaps (original, corrected, corrupted)
  3. Delta_Acc bar chart — corrected (up) vs corrupted (down)
  4. Error position distribution histogram
  5. AUC vs step position curves
  6. PRM score curves
  7. Symmetry comparison: |Delta_corrected| vs |Delta_corrupted|

Usage:
    python scripts/eval/plot_paper_figures.py \
        --results-root results/gsm8k_7b_v2 \
        --out-dir figures/prefix_correction_v2
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "figure.dpi": 200,
})


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate paper figures (v2)")
    ap.add_argument("--results-root",
                    default=str(PROJECT_ROOT / "results/gsm8k_7b_v2"))
    ap.add_argument("--cot-file",
                    default=str(PROJECT_ROOT / "results/gsm8k_7b_v2/raw_cot_n8.jsonl"))
    ap.add_argument("--out-dir",
                    default=str(PROJECT_ROOT / "figures/prefix_correction_v2"))
    ap.add_argument("--k-values", default="1,2,3,4")
    return ap.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Figure 1: IG dual plot
# ---------------------------------------------------------------------------

def plot_ig_dual(results_root: Path, out_dir: Path) -> None:
    ig_dir = results_root / "information_gain"
    ig_results = []
    for f in ig_dir.glob("ig_*.jsonl"):
        ig_results.extend(read_jsonl(f))
    if not ig_results:
        print("[SKIP] No IG results")
        return

    conditions = {}
    for r in ig_results:
        conditions.setdefault(r["condition"], []).append(r)

    max_steps = 10
    colors = {
        "correct": "#2ca02c", "wrong_original": "#d62728",
        "corrected_k2": "#1f77b4", "corrupted_k2": "#e377c2",
    }
    labels = {
        "correct": "Correct", "wrong_original": "Wrong (original)",
        "corrected_k2": "Corrected (k=2)", "corrupted_k2": "Corrupted (k=2)",
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                     gridspec_kw={"hspace": 0.08})

    for cond in ["correct", "wrong_original", "corrected_k2", "corrupted_k2"]:
        if cond not in conditions:
            continue
        cond_results = conditions[cond]
        ig_by_step = {}
        for r in cond_results:
            for i, ig in enumerate(r["ig_values"]):
                if i < max_steps:
                    ig_by_step.setdefault(i, []).append(ig)

        xs = sorted(ig_by_step.keys())
        ys = [np.mean(ig_by_step[x]) for x in xs]
        ses = [np.std(ig_by_step[x]) / max(np.sqrt(len(ig_by_step[x])), 1) for x in xs]
        xs_1 = [x + 1 for x in xs]
        color = colors.get(cond, "tab:gray")
        label = labels.get(cond, cond)

        ax1.plot(xs_1, ys, "o-", color=color, label=label, linewidth=2)
        ax1.fill_between(xs_1, [y - s for y, s in zip(ys, ses)],
                          [y + s for y, s in zip(ys, ses)], alpha=0.15, color=color)

    ax1.axhline(y=0, linestyle="--", color="gray", alpha=0.5)
    ax1.set_ylabel("Average IG (nats)")
    ax1.set_title("Per-Step Information Gain")
    ax1.grid(alpha=0.25)
    ax1.legend()

    # Cumulative
    for cond in ["correct", "wrong_original", "corrected_k2", "corrupted_k2"]:
        if cond not in conditions:
            continue
        cum_by_step = {}
        for r in conditions[cond]:
            cum = 0.0
            for i, ig in enumerate(r["ig_values"]):
                cum += ig
                if i < max_steps:
                    cum_by_step.setdefault(i, []).append(cum)
        xs = sorted(cum_by_step.keys())
        ys = [np.mean(cum_by_step[x]) for x in xs]
        color = colors.get(cond, "tab:gray")
        ax2.plot([x+1 for x in xs], ys, "o-", color=color, label=labels.get(cond, cond), linewidth=2)

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Cumulative log P(answer)")
    ax2.set_title("Cumulative Information")
    ax2.grid(alpha=0.25)
    ax2.legend()

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(out_dir / f"fig1_ig_dual.{ext}", dpi=200)
    plt.close()
    print("  fig1_ig_dual done")


# ---------------------------------------------------------------------------
# Figure 2: Separability heatmaps
# ---------------------------------------------------------------------------

def plot_separability_heatmaps(results_root: Path, out_dir: Path) -> None:
    probe_file = results_root / "probes" / "probe_results.jsonl"
    if not probe_file.exists():
        print("[SKIP] No probe results")
        return

    results = read_jsonl(probe_file)

    for cond in ["original", "corrected_k2", "corrupted_k2"]:
        cond_r = [r for r in results if r["condition"] == cond]
        if not cond_r:
            continue

        layers = sorted(set(r["layer"] for r in cond_r))
        steps = sorted(set(r["step"] for r in cond_r))
        mat = np.full((len(layers), len(steps)), np.nan)

        for r in cond_r:
            li = layers.index(r["layer"])
            si = steps.index(r["step"])
            mat[li, si] = r["mlp_auc"]

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0, origin="lower")
        ax.set_xlabel("Step")
        ax.set_ylabel("Layer")
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels(steps)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers)
        ax.set_title(f"Probe AUC — {cond}")
        plt.colorbar(im, ax=ax, label="AUC")
        plt.tight_layout()
        for ext in ["pdf", "png"]:
            plt.savefig(out_dir / f"fig2_heatmap_{cond}.{ext}", dpi=200)
        plt.close()
    print("  fig2 heatmaps done")


# ---------------------------------------------------------------------------
# Figure 3: Delta_Acc bar chart — corrected vs corrupted
# ---------------------------------------------------------------------------

def plot_delta_acc_bar(results_root: Path, out_dir: Path, k_values: List[int]) -> None:
    stats = read_json(results_root / "statistics" / "statistical_results.json")
    if not stats or "bootstrap" not in stats:
        print("[SKIP] No statistical results")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(k_values))
    width = 0.35

    # Corrected deltas (positive)
    corr_deltas, corr_ci_lo, corr_ci_hi = [], [], []
    for k in k_values:
        bs = stats["bootstrap"].get(f"corrected_k{k}")
        if bs:
            corr_deltas.append(bs["observed_delta_acc"])
            corr_ci_lo.append(bs["observed_delta_acc"] - bs["ci_lower"])
            corr_ci_hi.append(bs["ci_upper"] - bs["observed_delta_acc"])
        else:
            corr_deltas.append(0)
            corr_ci_lo.append(0)
            corr_ci_hi.append(0)

    # Corrupted deltas (negative)
    corrupt_deltas, corrupt_ci_lo, corrupt_ci_hi = [], [], []
    for k in k_values:
        bs = stats["bootstrap"].get(f"corrupted_k{k}")
        if bs:
            corrupt_deltas.append(bs["observed_delta_acc"])
            corrupt_ci_lo.append(bs["observed_delta_acc"] - bs["ci_lower"])
            corrupt_ci_hi.append(bs["ci_upper"] - bs["observed_delta_acc"])
        else:
            corrupt_deltas.append(0)
            corrupt_ci_lo.append(0)
            corrupt_ci_hi.append(0)

    bars1 = ax.bar(x - width/2, corr_deltas, width, label="Corrected (wrong→fixed)",
                    color="#2ca02c", alpha=0.8,
                    yerr=[corr_ci_lo, corr_ci_hi], capsize=4)
    bars2 = ax.bar(x + width/2, corrupt_deltas, width, label="Corrupted (correct→broken)",
                    color="#d62728", alpha=0.8,
                    yerr=[corrupt_ci_lo, corrupt_ci_hi], capsize=4)

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Correction depth k")
    ax.set_ylabel("Delta Accuracy")
    ax.set_title("Effect of Prefix Modification on Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in k_values])
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(out_dir / f"fig3_delta_acc_bar.{ext}", dpi=200)
    plt.close()
    print("  fig3_delta_acc_bar done")


# ---------------------------------------------------------------------------
# Figure 4: Error position distribution
# ---------------------------------------------------------------------------

def plot_error_position_dist(results_root: Path, out_dir: Path, k_values: List[int]) -> None:
    cot_file = results_root / "raw_cot_n8.jsonl"
    cot_rows = read_jsonl(cot_file)
    if not cot_rows:
        print("[SKIP] No CoT data")
        return

    wrong_rows = [r for r in cot_rows if r.get("exact_match", 1.0) < 1.0]
    n_steps_list = [r.get("n_steps", len(r.get("steps", []))) for r in wrong_rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(n_steps_list, bins=range(1, max(n_steps_list or [10]) + 2),
            color="#1f77b4", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Count")
    ax.set_title("Step Count Distribution (Wrong Trajectories)")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(out_dir / f"fig4_step_dist.{ext}", dpi=200)
    plt.close()
    print("  fig4_step_dist done")


# ---------------------------------------------------------------------------
# Figure 5: AUC vs step
# ---------------------------------------------------------------------------

def plot_auc_vs_step(results_root: Path, out_dir: Path) -> None:
    probe_file = results_root / "probes" / "probe_results.jsonl"
    if not probe_file.exists():
        print("[SKIP] No probe results")
        return

    results = read_jsonl(probe_file)
    conditions = sorted(set(r["condition"] for r in results))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(conditions), 2)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for ci, cond in enumerate(conditions):
        cond_r = [r for r in results if r["condition"] == cond]
        steps = sorted(set(r["step"] for r in cond_r))
        step_aucs = []
        for s in steps:
            sr = [r for r in cond_r if r["step"] == s]
            best = max(r["mlp_auc"] for r in sr) if sr else 0.5
            step_aucs.append(best)
        ax.plot(steps, step_aucs, "o-", color=colors[ci], label=cond, linewidth=2)

    ax.axhline(y=0.5, linestyle="--", color="gray", alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Probe AUC (best layer)")
    ax.set_title("Probe AUC by Condition and Step")
    ax.set_ylim(0.4, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(out_dir / f"fig5_auc_vs_step.{ext}", dpi=200)
    plt.close()
    print("  fig5_auc_vs_step done")


# ---------------------------------------------------------------------------
# Figure 6: PRM curves
# ---------------------------------------------------------------------------

def plot_prm_curves(results_root: Path, out_dir: Path, k_values: List[int]) -> None:
    prm_file = results_root / "prm_scores_all.jsonl"
    prm_rows = read_jsonl(prm_file)
    if not prm_rows:
        print("[SKIP] No PRM scores")
        return

    conditions = {}
    for r in prm_rows:
        conditions.setdefault(r["condition"], []).append(r)

    max_steps = 10
    colors = {
        "original": "#7f7f7f",
        "corrected_k1": "#aec7e8", "corrected_k2": "#1f77b4",
        "corrected_k3": "#ff7f0e", "corrected_k4": "#d62728",
        "corrupted_k1": "#c5b0d5", "corrupted_k2": "#9467bd",
        "corrupted_k3": "#8c564b", "corrupted_k4": "#e377c2",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for cond, rows in conditions.items():
        by_step = {}
        for r in rows:
            for i, s in enumerate(r.get("step_scores", [])):
                if i < max_steps:
                    by_step.setdefault(i, []).append(s)
        xs = sorted(by_step.keys())
        ys = [np.mean(by_step[x]) for x in xs]
        color = colors.get(cond, "tab:gray")
        ax.plot([x+1 for x in xs], ys, "o-", color=color, label=cond, linewidth=2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Average PRM Score")
    ax.set_title("PRM Score Curves by Condition")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(out_dir / f"fig6_prm_curves.{ext}", dpi=200)
    plt.close()
    print("  fig6_prm_curves done")


# ---------------------------------------------------------------------------
# Figure 7: Symmetry comparison
# ---------------------------------------------------------------------------

def plot_symmetry(results_root: Path, out_dir: Path, k_values: List[int]) -> None:
    stats = read_json(results_root / "statistics" / "statistical_results.json")
    if not stats or "bootstrap" not in stats:
        print("[SKIP] No stats for symmetry plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(k_values))

    corr_abs = []
    corrupt_abs = []
    for k in k_values:
        bs_c = stats["bootstrap"].get(f"corrected_k{k}")
        bs_d = stats["bootstrap"].get(f"corrupted_k{k}")
        corr_abs.append(abs(bs_c["observed_delta_acc"]) if bs_c else 0)
        corrupt_abs.append(abs(bs_d["observed_delta_acc"]) if bs_d else 0)

    width = 0.35
    ax.bar(x - width/2, corr_abs, width, label="|Delta| Corrected",
           color="#2ca02c", alpha=0.8)
    ax.bar(x + width/2, corrupt_abs, width, label="|Delta| Corrupted",
           color="#d62728", alpha=0.8)

    ax.set_xlabel("Correction depth k")
    ax.set_ylabel("|Delta Accuracy|")
    ax.set_title("Symmetry: Correction vs Corruption Effect Size")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in k_values])
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(out_dir / f"fig7_symmetry.{ext}", dpi=200)
    plt.close()
    print("  fig7_symmetry done")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    k_values = [int(x) for x in args.k_values.split(",")]
    results_root = Path(args.results_root)

    print("Generating paper figures (v2)...")
    plot_ig_dual(results_root, out_dir)
    plot_separability_heatmaps(results_root, out_dir)
    plot_delta_acc_bar(results_root, out_dir, k_values)
    plot_error_position_dist(results_root, out_dir, k_values)
    plot_auc_vs_step(results_root, out_dir)
    plot_prm_curves(results_root, out_dir, k_values)
    plot_symmetry(results_root, out_dir, k_values)
    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
