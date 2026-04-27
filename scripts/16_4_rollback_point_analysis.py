#!/usr/bin/env python3
"""
Rollback point analysis: compare dynamic methods vs oracle and fixed-alpha.

Loads entropy/probe/composite rollback points and evaluates:
  - Distance to oracle (|t* - tau|)
  - Precision@k (fraction within k steps of tau)
  - Accuracy after suffix voting
  - Cost (tokens per question)

Produces comparison figures in figures/state_aware_rollback/.

Usage:
    python scripts/16_4_rollback_point_analysis.py \
        --dataset gsm8k \
        --model-id meta-llama/Llama-3.2-3B-Instruct
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]


def parse_args():
    ap = argparse.ArgumentParser(description="Rollback point analysis")
    ap.add_argument("--model-id", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--first-error-cache", default="")
    ap.add_argument("--entropy-dir", default="")
    ap.add_argument("--probe-dir", default="")
    ap.add_argument("--composite-dir", default="")
    ap.add_argument("--sweep-dir", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--n-drafts", type=int, default=4)
    ap.add_argument("--beta", type=float, default=0.5)
    return ap.parse_args()


def compute_rollback_metrics(
    oracle_points: Dict[Tuple, int],
    method_points: Dict[Tuple, int],
    n_steps_map: Dict[Tuple, int],
) -> dict:
    """Compute distance and precision metrics for a rollback method."""
    distances = []
    within_1, within_2, within_3 = 0, 0, 0
    total = 0

    for key, tau in oracle_points.items():
        t_star = method_points.get(key)
        if t_star is None:
            continue
        dist = abs(t_star - tau)
        distances.append(dist)
        total += 1
        if dist <= 1:
            within_1 += 1
        if dist <= 2:
            within_2 += 1
        if dist <= 3:
            within_3 += 1

    if not distances:
        return {"mean_dist": 0, "median_dist": 0,
                "prec@1": 0, "prec@2": 0, "prec@3": 0, "n": 0}

    return {
        "mean_dist": round(np.mean(distances), 3),
        "median_dist": round(float(np.median(distances)), 1),
        "std_dist": round(np.std(distances), 3),
        "prec@1": round(within_1 / total, 4),
        "prec@2": round(within_2 / total, 4),
        "prec@3": round(within_3 / total, 4),
        "n": total,
    }


def plot_distance_comparison(all_metrics: dict, out_dir: Path):
    """Bar chart comparing mean distance to oracle across methods."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(all_metrics.keys())
    mean_dists = [all_metrics[m]["mean_dist"] for m in methods]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    bars = ax.bar(x, mean_dists, color=plt.cm.Set2(np.linspace(0, 1, len(methods))))
    for bar, val in zip(bars, mean_dists):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean |t* - tau|", fontsize=11)
    ax.set_title("Rollback Point Distance to Oracle", fontsize=13)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "distance_to_oracle.png", dpi=180)
    plt.savefig(out_dir / "distance_to_oracle.pdf")
    plt.close()


def plot_precision_comparison(all_metrics: dict, out_dir: Path):
    """Grouped bar chart for Precision@1/2/3."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(all_metrics.keys())
    p1 = [all_metrics[m]["prec@1"] for m in methods]
    p2 = [all_metrics[m]["prec@2"] for m in methods]
    p3 = [all_metrics[m]["prec@3"] for m in methods]

    x = np.arange(len(methods))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, p1, w, label="Prec@1", color="#3498db")
    ax.bar(x, p2, w, label="Prec@2", color="#2ecc71")
    ax.bar(x + w, p3, w, label="Prec@3", color="#e67e22")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Rollback Point Precision (within k steps of oracle)", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "precision_at_k.png", dpi=180)
    plt.savefig(out_dir / "precision_at_k.pdf")
    plt.close()


def plot_accuracy_cost(results_by_method: dict, out_dir: Path):
    """Scatter plot: accuracy vs tokens-per-question for all methods."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(results_by_method), 2)))

    for i, (name, res) in enumerate(results_by_method.items()):
        ax.scatter(res.get("tpq", 0), res["accuracy"],
                   s=80, color=colors[i], zorder=3, label=name)

    ax.set_xlabel("Tokens per Question", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Accuracy vs Cost: State-Aware Rollback Methods", fontsize=13)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_cost.png", dpi=180)
    plt.savefig(out_dir / "accuracy_vs_cost.pdf")
    plt.close()


def plot_distance_histogram(
    oracle_points: Dict[Tuple, int],
    method_points_dict: Dict[str, Dict[Tuple, int]],
    out_dir: Path,
):
    """Overlaid histograms of |t* - tau| for each method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#e67e22"]

    for i, (name, pts) in enumerate(method_points_dict.items()):
        dists = []
        for key, tau in oracle_points.items():
            t_star = pts.get(key)
            if t_star is not None:
                dists.append(abs(t_star - tau))
        if dists:
            ax.hist(dists, bins=range(0, max(dists) + 2), alpha=0.5,
                    color=colors[i % len(colors)], label=name, edgecolor="white")

    ax.set_xlabel("|t* - tau|", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of Rollback Point Error", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "distance_histogram.png", dpi=180)
    plt.savefig(out_dir / "distance_histogram.pdf")
    plt.close()


def main():
    args = parse_args()

    from src.prompt_templates import split_steps
    from src.step_scorers import select_rollback_point

    model_short = args.model_id.split("/")[-1].lower().replace("-", "_")
    if not args.first_error_cache:
        args.first_error_cache = str(
            ROOT / "results/gsm8k_3b_multi_sample/first_error/gpt_first_error_cache.jsonl"
        )
    if not args.entropy_dir:
        args.entropy_dir = str(ROOT / "results" / f"{args.dataset}_{model_short}_entropy_later")
    if not args.probe_dir:
        args.probe_dir = str(ROOT / "results" / f"{args.dataset}_{model_short}_probe_later")
    if not args.composite_dir:
        args.composite_dir = str(ROOT / "results" / f"{args.dataset}_{model_short}_composite_later")
    if not args.sweep_dir:
        args.sweep_dir = str(ROOT / "results" / f"{args.dataset}_{model_short}_sweep")
    if not args.out_dir:
        args.out_dir = str(ROOT / "figures/state_aware_rollback")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    beta = args.beta
    nd = args.n_drafts

    # Load first-error data for oracle rollback points
    fe_cache = _load_jsonl(Path(args.first_error_cache))
    fe_map = {}
    for r in fe_cache:
        if r.get("tau") is not None:
            fe_map[(r["doc_id"], r["sample_idx"])] = r

    # Load drafts from sweep checkpoint
    sweep_records = _load_jsonl(Path(args.sweep_dir) / "checkpoint.jsonl")
    drafts = [r for r in sweep_records if r.get("task_type") == "draft"]
    draft_map = {(d["doc_id"], d["draft_idx"]): d for d in drafts}

    # Build oracle rollback points (tau from first-error labels)
    # Sweep uses "gsm8k_N" doc_ids, first-error cache uses integer N
    oracle_points: Dict[Tuple, int] = {}
    n_steps_map: Dict[Tuple, int] = {}
    for (did, di), d in draft_map.items():
        if di >= nd:
            continue
        steps = d.get("draft_steps", split_steps(d.get("draft_text", "")))
        n_steps_map[(did, di)] = len(steps)
        # Extract numeric index from "gsm8k_N" -> N
        try:
            numeric_id = int(str(did).split("_")[-1])
        except (ValueError, IndexError):
            numeric_id = did
        for sid in range(8):
            fe = fe_map.get((numeric_id, sid))
            if fe and fe["tau"] is not None:
                tau_0idx = fe["tau"] - 1
                if 0 <= tau_0idx < len(steps):
                    oracle_points[(did, di)] = tau_0idx
                break

    print(f"Oracle rollback points: {len(oracle_points)} drafts")

    # Build fixed-alpha rollback points
    fixed_alpha_points = {}
    for alpha in [0.4, 0.6, 0.8]:
        pts = {}
        for key, n_steps in n_steps_map.items():
            if n_steps > 0:
                b = max(1, math.ceil(alpha * n_steps))
                if b >= n_steps:
                    b = n_steps - 1
                pts[key] = b
        fixed_alpha_points[f"Fixed(a={alpha})"] = pts

    # Load entropy rollback points
    ent_details = _load_jsonl(Path(args.entropy_dir) / "entropy_details.jsonl")
    ent_map = {(r["doc_id"], r["draft_idx"]): r["step_entropies"]
               for r in ent_details}
    entropy_points = {}
    for key, ent in ent_map.items():
        if key in n_steps_map:
            t_star = select_rollback_point(ent, beta=beta, method="argmax")
            entropy_points[key] = t_star

    # Load probe rollback points
    probe_scores = _load_jsonl(Path(args.probe_dir) / "probe_scores.jsonl")
    probe_map = {(r["doc_id"], r["draft_idx"]): r["step_scores"]
                 for r in probe_scores}
    probe_points = {}
    for key, scores in probe_map.items():
        if key in n_steps_map:
            t_star = select_rollback_point(scores, beta=beta, method="argmax")
            probe_points[key] = t_star

    # Compute metrics for all methods
    all_method_points = {}
    all_method_points.update(fixed_alpha_points)
    if entropy_points:
        all_method_points["Entropy"] = entropy_points
    if probe_points:
        all_method_points["Probe"] = probe_points

    all_metrics = {}
    for name, pts in all_method_points.items():
        metrics = compute_rollback_metrics(oracle_points, pts, n_steps_map)
        all_metrics[name] = metrics
        print(f"{name:20s}  mean_dist={metrics['mean_dist']:.2f}  "
              f"prec@1={metrics['prec@1']:.3f}  "
              f"prec@2={metrics['prec@2']:.3f}  "
              f"prec@3={metrics['prec@3']:.3f}  n={metrics['n']}")

    # Save metrics
    metrics_file = out_dir / "rollback_metrics.json"
    metrics_file.write_text(
        json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\nMetrics -> {metrics_file}")

    # Load accuracy results from each method's summary
    results_by_method = {}
    for name, summary_file in [
        ("Greedy", None),
        ("FixedLR(a=0.6)", Path(args.sweep_dir) / "sweep_summary.json"),
        ("EntropyLATER", Path(args.entropy_dir) / "entropy_later_summary.json"),
        ("ProbeLATER", Path(args.probe_dir) / "probe_later_summary.json"),
        ("CompositeLATER", Path(args.composite_dir) / "composite_later_summary.json"),
    ]:
        if summary_file and summary_file.exists():
            summary = json.loads(summary_file.read_text("utf-8"))
            res_list = summary.get("results", [])
            best = None
            for r in res_list:
                if r.get("method", "") == "Greedy":
                    if name == "Greedy":
                        best = r
                    continue
                if name != "Greedy":
                    if best is None or r.get("accuracy", 0) > best.get("accuracy", 0):
                        best = r
            if best:
                results_by_method[name] = best

    # Generate figures
    if all_metrics:
        plot_distance_comparison(all_metrics, out_dir)
        plot_precision_comparison(all_metrics, out_dir)
        plot_distance_histogram(oracle_points, all_method_points, out_dir)
        print("Distance and precision figures saved")

    if results_by_method:
        plot_accuracy_cost(results_by_method, out_dir)
        print("Accuracy vs cost figure saved")

    print(f"\nAll figures -> {out_dir}")


if __name__ == "__main__":
    main()
