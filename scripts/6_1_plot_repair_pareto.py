#!/usr/bin/env python3
"""
Plot iso-compute Pareto curves from repair sweep results.

Reads repair_sweep_summary.json and generates:
  - fig_repair_pareto_{dataset}.png/pdf: main Pareto figure
  - fig_repair_ablation_{dataset}.png/pdf: lookback vs lookahead vs symmetric

Usage:
    python scripts/6_1_plot_repair_pareto.py [--result-dir results/gsm8k_qwen2.5_3b_instruct_repair_sweep]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

COLORS = {
    "Greedy": "#333333",
    "SC": "#1976D2",
    "Random": "#9E9E9E",
    "Lookback": "#D32F2F",
    "Lookahead": "#FF9800",
    "Symmetric": "#7B1FA2",
}
MARKERS = {
    "Greedy": "X",
    "SC": "o",
    "Random": "d",
    "Lookback": "s",
    "Lookahead": "^",
    "Symmetric": "v",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="")
    ap.add_argument("--fig-dir", default=str(ROOT / "figures" / "repair_sweep"))
    return ap.parse_args()


def pareto_front(points):
    """Return Pareto-optimal points (minimize x=tpq, maximize y=accuracy)."""
    pts = sorted(points, key=lambda p: p[0])
    front = []
    best_y = -1
    for x, y, label in pts:
        if y > best_y:
            front.append((x, y, label))
            best_y = y
    return front


def categorize(result):
    method = result.get("method", "")
    if method == "Greedy":
        return "Greedy"
    if method.startswith("SC"):
        return "SC"
    if "random" in method.lower() or result.get("trigger_mode") == "random":
        return "Random"
    mode = result.get("trigger_mode", "")
    if mode == "lookback":
        return "Lookback"
    if mode == "lookahead":
        return "Lookahead"
    if mode == "symmetric":
        return "Symmetric"
    if "Repair_lookback" in method:
        return "Lookback"
    if "Repair_lookahead" in method:
        return "Lookahead"
    if "Repair_symmetric" in method:
        return "Symmetric"
    return "Other"


def plot_pareto(summary, fig_dir):
    dataset = summary["dataset"]
    model = summary["model"].split("/")[-1]
    results = summary["results"]

    fig, ax = plt.subplots(figsize=(8, 6))

    groups = {}
    for r in results:
        cat = categorize(r)
        if cat == "Other":
            continue
        groups.setdefault(cat, []).append(r)

    for cat in ["SC", "Random", "Lookback", "Lookahead", "Symmetric", "Greedy"]:
        if cat not in groups:
            continue
        pts = [(r["tpq"], r["accuracy"]) for r in groups[cat]]
        xs, ys = zip(*pts)
        color = COLORS.get(cat, "#666")
        marker = MARKERS.get(cat, "o")

        if cat == "Greedy":
            ax.scatter(xs, ys, color=color, marker=marker, s=120, zorder=10,
                       label=cat, edgecolors="black", linewidths=1)
        elif cat in ("SC", "Lookback"):
            sorted_pts = sorted(zip(xs, ys))
            sx, sy = zip(*sorted_pts)
            ax.plot(sx, sy, color=color, marker=marker, markersize=6,
                    linewidth=2, label=cat, zorder=5)
        else:
            sorted_pts = sorted(zip(xs, ys))
            sx, sy = zip(*sorted_pts)
            ax.plot(sx, sy, color=color, marker=marker, markersize=5,
                    linewidth=1.5, alpha=0.7, label=cat, zorder=4)

    ax.set_xlabel("Tokens per Question", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Iso-Compute Pareto: {dataset} ({model})", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(fig_dir / f"fig_repair_pareto_{dataset}.{fmt}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> fig_repair_pareto_{dataset}")


def plot_ablation(summary, fig_dir):
    dataset = summary["dataset"]
    model = summary["model"].split("/")[-1]
    results = summary["results"]

    fig, ax = plt.subplots(figsize=(8, 6))

    for cat in ["Lookback", "Lookahead", "Symmetric", "Random"]:
        pts = []
        for r in results:
            if categorize(r) == cat:
                pts.append((r["tpq"], r["accuracy"]))
        if not pts:
            continue
        sorted_pts = sorted(set(pts))
        xs, ys = zip(*sorted_pts)
        color = COLORS.get(cat, "#666")
        marker = MARKERS.get(cat, "o")
        ax.plot(xs, ys, color=color, marker=marker, markersize=6,
                linewidth=2, label=cat)

    # Add SC reference
    sc_pts = [(r["tpq"], r["accuracy"]) for r in results if categorize(r) == "SC"]
    if sc_pts:
        sc_pts.sort()
        sx, sy = zip(*sc_pts)
        ax.plot(sx, sy, color=COLORS["SC"], marker=MARKERS["SC"],
                markersize=6, linewidth=2, linestyle="--", alpha=0.5, label="SC")

    ax.set_xlabel("Tokens per Question", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Trigger Ablation: {dataset} ({model})", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(fig_dir / f"fig_repair_ablation_{dataset}.{fmt}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> fig_repair_ablation_{dataset}")


def main():
    args = parse_args()
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.result_dir:
        dirs = [Path(args.result_dir)]
    else:
        dirs = sorted(ROOT.glob("results/*_repair_sweep"))

    for d in dirs:
        sp = d / "repair_sweep_summary.json"
        if not sp.exists():
            print(f"Skipping {d}: no summary")
            continue
        summary = json.loads(sp.read_text("utf-8"))
        print(f"\nPlotting {summary['dataset']} ({summary['model']})")
        plot_pareto(summary, fig_dir)
        plot_ablation(summary, fig_dir)

    print(f"\nAll figures -> {fig_dir}")


if __name__ == "__main__":
    main()
