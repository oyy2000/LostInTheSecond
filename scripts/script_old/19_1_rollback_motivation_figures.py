#!/usr/bin/env python3
"""
Plot rollback motivation results across datasets.

Reads rollback_summary.json from each dataset's output directory and produces
a multi-panel figure showing recovery rate vs relative rollback position.

Usage:
    python scripts/19_1_rollback_motivation_figures.py
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "math500": "MATH-500",
    "hotpotqa": "HotpotQA",
    "gsm8k": "GSM8K",
}
COLORS = {
    "math500": "#3498db",
    "hotpotqa": "#e74c3c",
    "gsm8k": "#2ecc71",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig-dir", default=str(ROOT / "figures/rollback_motivation"))
    ap.add_argument("--result-dirs", nargs="*", default=[])
    return ap.parse_args()


def load_summary(ds: str, result_dirs: list) -> dict | None:
    for d in result_dirs:
        p = Path(d) / "rollback_summary.json"
        if p.exists():
            return json.loads(p.read_text("utf-8"))
    default = ROOT / "results" / f"{ds}_rollback_motivation" / "rollback_summary.json"
    if default.exists():
        return json.loads(default.read_text("utf-8"))
    return None


def main():
    args = parse_args()
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for ds in DATASETS:
        s = load_summary(ds, args.result_dirs)
        if s:
            summaries[ds] = s
            print(f"Loaded {ds}: {s['n_questions']} questions, "
                  f"{s['n_wrong_drafts']} wrong, greedy_acc={s['greedy_acc']}")

    if not summaries:
        print("No data found.")
        return

    # -- Figure 1: overlay line plot --
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    for ds, label in DATASETS.items():
        s = summaries.get(ds)
        if not s:
            continue
        bins = s["bins"]
        centers = [b["bin_center"] for b in bins]
        means = [b["mean"] for b in bins]
        lo = [b["mean"] - 1.96 * b["se"] for b in bins]
        hi = [b["mean"] + 1.96 * b["se"] for b in bins]
        color = COLORS[ds]
        ax.plot(centers, means, "o-", color=color, linewidth=2, markersize=5,
                label=f"{label} (greedy={s['greedy_acc']:.1%})")
        ax.fill_between(centers, lo, hi, color=color, alpha=0.12)

    ax.set_xlabel("Relative Rollback Position ($k / T$)")
    ax.set_ylabel("Recovery Rate")
    ax.set_title("Rollback Recovery vs. Relative Position (Llama 3.2 3B)")
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(fig_dir / f"fig_rollback_overlay.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir / 'fig_rollback_overlay.png'}")

    # -- Figure 2: separate panels --
    n_ds = len(summaries)
    if n_ds > 1:
        fig, axes = plt.subplots(1, n_ds, figsize=(4.5 * n_ds, 4), sharey=True)
        if n_ds == 1:
            axes = [axes]
        for ax, (ds, label) in zip(axes, [(d, DATASETS[d]) for d in DATASETS if d in summaries]):
            s = summaries[ds]
            bins = s["bins"]
            centers = [b["bin_center"] for b in bins]
            means = [b["mean"] for b in bins]
            lo = [b["mean"] - 1.96 * b["se"] for b in bins]
            hi = [b["mean"] + 1.96 * b["se"] for b in bins]
            ns = [b["n"] for b in bins]
            color = COLORS[ds]

            ax.plot(centers, means, "o-", color=color, linewidth=2, markersize=6)
            ax.fill_between(centers, lo, hi, color=color, alpha=0.15)
            for c, n in zip(centers, ns):
                ax.text(c, -0.015, f"{n}", ha="center", va="top", fontsize=7, color="#888")

            ax.set_xlabel("Relative Rollback Position ($k / T$)")
            if ax == axes[0]:
                ax.set_ylabel("Recovery Rate")
            ax.set_title(f"{label} (greedy={s['greedy_acc']:.1%})")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlim(0, 1)

        plt.tight_layout()
        for ext in ["pdf", "png"]:
            fig.savefig(fig_dir / f"fig_rollback_panels.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fig_dir / 'fig_rollback_panels.png'}")


if __name__ == "__main__":
    main()
