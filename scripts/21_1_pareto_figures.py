#!/usr/bin/env python3
"""
Experiment 1 figures: 4-panel iso-compute Pareto curves.

Reads pareto_summary.json from each (model, dataset) combo and plots
Pareto fronts with X=total_generation_tokens, Y=accuracy.

Usage:
    python scripts/21_1_pareto_figures.py
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pareto_engine import extract_pareto_front

MODELS = [
    ("meta-llama/Llama-3.2-3B-Instruct", "Llama-3.2-3B"),
    ("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5-3B"),
]
DATASETS = [
    ("gsm8k", "GSM8K"),
    ("math500", "MATH500"),
]
FIG_DIR = ROOT / "figures" / "pareto"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _model_short(model_id: str) -> str:
    return model_id.split("/")[-1].lower().replace("-", "_")


METHOD_STYLES = {
    "Greedy": {"color": "black", "marker": "*", "ms": 14, "zorder": 10},
    "SC": {"color": "#4C72B0", "marker": "o", "ls": "-", "lw": 2},
    "random": {"color": "#DD8452", "marker": "s", "ls": "--", "lw": 1.5},
    "entropy_lookback": {"color": "#C44E52", "marker": "^", "ls": "-", "lw": 2.5},
    "entropy_lookahead": {"color": "#55A868", "marker": "v", "ls": "-.", "lw": 1.5},
}

METHOD_LABELS = {
    "Greedy": "Greedy",
    "SC": "Self-Consistency",
    "random": "Random Repair",
    "entropy_lookback": "Entropy Lookback (ours)",
    "entropy_lookahead": "Entropy Lookahead",
}


def classify_method(result: dict) -> str:
    m = result.get("method", "")
    if m == "Greedy":
        return "Greedy"
    if m == "SC":
        return "SC"
    if m.startswith("random"):
        return "random"
    if "lookback" in m:
        return "entropy_lookback"
    if "lookahead" in m:
        return "entropy_lookahead"
    return m


def plot_one_panel(ax, results: List[dict], title: str):
    groups: Dict[str, List[Tuple[float, float]]] = {}
    for r in results:
        cls = classify_method(r)
        x = r.get("mean_tokens", 0)
        y = r.get("accuracy", 0)
        if x > 0:
            groups.setdefault(cls, []).append((x, y))

    draw_order = ["SC", "random", "entropy_lookahead", "entropy_lookback", "Greedy"]
    for cls in draw_order:
        pts = groups.get(cls, [])
        if not pts:
            continue
        style = METHOD_STYLES.get(cls, {"color": "gray"})
        label = METHOD_LABELS.get(cls, cls)

        if cls == "Greedy":
            ax.scatter(
                [pts[0][0]], [pts[0][1]],
                color=style["color"], marker=style["marker"],
                s=style.get("ms", 10) ** 2, zorder=style.get("zorder", 5),
                label=label,
            )
        else:
            front = extract_pareto_front(pts)
            xs, ys = zip(*front) if front else ([], [])
            ax.plot(
                xs, ys,
                color=style.get("color", "gray"),
                marker=style.get("marker", "o"),
                ls=style.get("ls", "-"),
                lw=style.get("lw", 1.5),
                markersize=5, alpha=0.85,
                label=label,
            )
            non_front = [p for p in pts if p not in front]
            if non_front:
                nfx, nfy = zip(*non_front)
                ax.scatter(
                    nfx, nfy,
                    color=style.get("color", "gray"),
                    marker=style.get("marker", "o"),
                    s=15, alpha=0.25, zorder=1,
                )

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Total Generation Tokens (per question)", fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Iso-Compute Pareto Curves", fontsize=14, fontweight="bold", y=1.01)

    for ri, (model_id, model_label) in enumerate(MODELS):
        for ci, (ds, ds_label) in enumerate(DATASETS):
            ax = axes[ri, ci]
            ms = _model_short(model_id)
            summary_file = ROOT / "results" / f"{ds}_{ms}_pareto" / "pareto_summary.json"
            if summary_file.exists():
                summary = json.loads(summary_file.read_text("utf-8"))
                results = summary.get("results", [])
                title = f"{model_label} / {ds_label}"
                plot_one_panel(ax, results, title)
            else:
                ax.set_title(f"{model_label} / {ds_label} (no data)", fontsize=11)
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=12, color="gray")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_pareto_curves.{ext}", dpi=200,
                    bbox_inches="tight")
    print(f"Saved fig_pareto_curves to {FIG_DIR}")
    plt.close(fig)


if __name__ == "__main__":
    main()
