#!/usr/bin/env python3
"""
Conditioned PRM analysis with model-size dimension.

Deepens the per-step PRM analysis by conditioning on:
1. Chain length (short vs long chains)
2. Problem difficulty (MATH levels 1-5)
3. Step-1 PRM as predictor of final correctness
4. Model size as primary conditioning variable

Usage:
    python 25_conditioned_prm_analysis.py
    python 25_conditioned_prm_analysis.py --models 0.5B,1.5B,3B,7B
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_REGISTRY = {
    "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "3B":   "Qwen/Qwen2.5-3B-Instruct",
    "7B":   "Qwen/Qwen2.5-7B-Instruct",
    "14B":  "Qwen/Qwen2.5-14B-Instruct",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="0.5B,1.5B,3B,7B")
    ap.add_argument("--prm-dir", default=str(PROJECT_ROOT / "runs" / "multi_scale_prm"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "runs" / "conditioned_prm"))
    return ap.parse_args()


def load_prm_data(prm_dir: Path, tag: str) -> Optional[dict]:
    f = prm_dir / f"prm_{tag}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def _avg_curve(samples: list) -> Tuple[List[int], List[float]]:
    if not samples:
        return [], []
    max_steps = max((len(s["step_scores"]) for s in samples), default=0)
    xs, ys = [], []
    for i in range(max_steps):
        vals = [s["step_scores"][i] for s in samples if i < len(s["step_scores"])]
        if vals:
            xs.append(i + 1)
            ys.append(sum(vals) / len(vals))
    return xs, ys


def _model_params(tag: str) -> float:
    for size, val in [("0.5b", 5e8), ("1.5b", 1.5e9), ("3b", 3e9),
                      ("7b", 7e9), ("14b", 14e9)]:
        if size in tag.lower():
            return val
    return 1e9


# ── Analysis 1: Chain length conditioning ──

def analyze_chain_length(
    all_data: Dict[str, dict],
    out_dir: Path,
):
    """Condition PRM curves on chain length (short/medium/long)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_tag, data in all_data.items():
        samples = data["samples"]
        wrong = [s for s in samples if s["exact_match"] < 1]

        if not wrong:
            continue

        lengths = [s["n_steps"] for s in wrong]
        p33 = np.percentile(lengths, 33)
        p66 = np.percentile(lengths, 66)

        short = [s for s in wrong if s["n_steps"] <= p33]
        medium = [s for s in wrong if p33 < s["n_steps"] <= p66]
        long_ = [s for s in wrong if s["n_steps"] > p66]

        fig, ax = plt.subplots(figsize=(10, 6))
        for group, label, marker in [
            (short, f"Short (n<={int(p33)}, n={len(short)})", "v"),
            (medium, f"Medium ({int(p33)}<n<={int(p66)}, n={len(medium)})", "s"),
            (long_, f"Long (n>{int(p66)}, n={len(long_)})", "^"),
        ]:
            xs, ys = _avg_curve(group)
            if xs:
                ax.plot(xs, ys, marker=marker, linewidth=2, label=label)

        ax.set_xlabel("Step k", fontsize=12)
        ax.set_ylabel("Avg PRM Score (wrong samples)", fontsize=12)
        ax.set_title(f"PRM by Chain Length: {data.get('model_id', model_tag)}", fontsize=13)
        ax.set_ylim(0.5, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / f"chain_length_{model_tag}.png", dpi=180)
        plt.close()

    # Cross-scale: does chain length confound explain the dip?
    if len(all_data) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, length_label, filter_fn in zip(
            axes,
            ["Short", "Medium", "Long"],
            [
                lambda s, p33, p66: s["n_steps"] <= p33,
                lambda s, p33, p66: p33 < s["n_steps"] <= p66,
                lambda s, p33, p66: s["n_steps"] > p66,
            ],
        ):
            colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_data)))
            for idx, (tag, data) in enumerate(
                sorted(all_data.items(), key=lambda x: _model_params(x[0]))
            ):
                wrong = [s for s in data["samples"] if s["exact_match"] < 1]
                if not wrong:
                    continue
                lengths = [s["n_steps"] for s in wrong]
                p33 = np.percentile(lengths, 33)
                p66 = np.percentile(lengths, 66)
                filtered = [s for s in wrong if filter_fn(s, p33, p66)]
                xs, ys = _avg_curve(filtered)
                if xs:
                    short_name = data.get("model_id", tag).split("/")[-1]
                    ax.plot(xs, ys, marker="o", linewidth=1.5, color=colors[idx],
                            label=f"{short_name} (n={len(filtered)})")

            ax.set_xlabel("Step k")
            ax.set_ylabel("Avg PRM Score (wrong)")
            ax.set_title(f"{length_label} Chains")
            ax.set_ylim(0.5, 1.02)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)

        plt.suptitle("Cross-Scale PRM Curves by Chain Length (Wrong Samples)", fontsize=14)
        plt.tight_layout()
        plt.savefig(out_dir / "cross_scale_by_length.png", dpi=200)
        plt.close()


# ── Analysis 2: Step-1 score as predictor ──

def analyze_step1_predictiveness(
    all_data: Dict[str, dict],
    out_dir: Path,
):
    """Analyze whether step-1 PRM score predicts final correctness."""
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for model_tag, data in sorted(all_data.items(), key=lambda x: _model_params(x[0])):
        samples = data["samples"]
        step1_correct = []
        step1_wrong = []

        for s in samples:
            if len(s["step_scores"]) < 1:
                continue
            s1 = s["step_scores"][0]
            if s["exact_match"] >= 1:
                step1_correct.append(s1)
            else:
                step1_wrong.append(s1)

        if not step1_correct or not step1_wrong:
            continue

        # T-test
        t_stat, p_val = scipy_stats.ttest_ind(step1_correct, step1_wrong)
        # AUC-like: what fraction of correct samples have step1 > median(wrong)
        wrong_median = np.median(step1_wrong)
        frac_above = np.mean([1 if s > wrong_median else 0 for s in step1_correct])

        summary_rows.append({
            "model_tag": model_tag,
            "model_id": data.get("model_id", ""),
            "mean_step1_correct": round(float(np.mean(step1_correct)), 4),
            "mean_step1_wrong": round(float(np.mean(step1_wrong)), 4),
            "gap": round(float(np.mean(step1_correct) - np.mean(step1_wrong)), 4),
            "t_stat": round(float(t_stat), 4),
            "p_val": float(p_val),
            "frac_correct_above_wrong_median": round(frac_above, 4),
        })

        # Per-model histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(step1_correct, bins=30, alpha=0.6, label=f"Correct (n={len(step1_correct)})", color="green")
        ax.hist(step1_wrong, bins=30, alpha=0.6, label=f"Wrong (n={len(step1_wrong)})", color="red")
        ax.axvline(np.mean(step1_correct), color="green", linestyle="--", linewidth=2)
        ax.axvline(np.mean(step1_wrong), color="red", linestyle="--", linewidth=2)
        ax.set_xlabel("Step-1 PRM Score", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"Step-1 Score Distribution: {data.get('model_id', model_tag)}\n"
                      f"Gap={np.mean(step1_correct) - np.mean(step1_wrong):.4f}, p={p_val:.2e}", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / f"step1_dist_{model_tag}.png", dpi=180)
        plt.close()

    # Cross-scale: step-1 gap vs model size
    if len(summary_rows) >= 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        tags = [r["model_tag"] for r in summary_rows]
        params = [_model_params(t) for t in tags]
        gaps = [r["gap"] for r in summary_rows]
        fracs = [r["frac_correct_above_wrong_median"] for r in summary_rows]

        ax1.plot(params, gaps, "o-", linewidth=2, markersize=8)
        for p, g, t in zip(params, gaps, tags):
            ax1.annotate(t, (p, g), textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=9)
        ax1.set_xscale("log")
        ax1.set_xlabel("Model Parameters")
        ax1.set_ylabel("Step-1 PRM Gap (correct - wrong)")
        ax1.set_title("Step-1 Predictiveness Gap vs Scale")
        ax1.grid(alpha=0.3)

        ax2.plot(params, fracs, "s-", linewidth=2, markersize=8, color="orange")
        for p, f, t in zip(params, fracs, tags):
            ax2.annotate(t, (p, f), textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=9)
        ax2.set_xscale("log")
        ax2.set_xlabel("Model Parameters")
        ax2.set_ylabel("Frac correct > median(wrong) at step 1")
        ax2.set_title("Step-1 Separability vs Scale")
        ax2.grid(alpha=0.3)

        plt.suptitle("Is Step-1 PRM Score More Predictive for Larger Models?", fontsize=14)
        plt.tight_layout()
        plt.savefig(out_dir / "step1_predictiveness_scaling.png", dpi=200)
        plt.close()

    (out_dir / "step1_predictiveness.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ── Analysis 3: Step-by-step divergence point ──

def analyze_divergence_point(
    all_data: Dict[str, dict],
    out_dir: Path,
):
    """Find at which step correct and wrong samples diverge most, per model."""
    out_dir.mkdir(parents=True, exist_ok=True)

    divergence_summary = []

    for model_tag, data in sorted(all_data.items(), key=lambda x: _model_params(x[0])):
        correct = [s for s in data["samples"] if s["exact_match"] >= 1]
        wrong = [s for s in data["samples"] if s["exact_match"] < 1]

        xc, yc = _avg_curve(correct)
        xw, yw = _avg_curve(wrong)

        min_len = min(len(yc), len(yw))
        if min_len < 2:
            continue

        gaps = [yc[i] - yw[i] for i in range(min_len)]
        max_gap_step = gaps.index(max(gaps)) + 1
        first_big_gap_step = None
        for i, g in enumerate(gaps):
            if g > 0.1:
                first_big_gap_step = i + 1
                break

        divergence_summary.append({
            "model_tag": model_tag,
            "max_gap": round(max(gaps), 4),
            "max_gap_step": max_gap_step,
            "first_big_gap_step": first_big_gap_step,
            "gaps_per_step": [round(g, 4) for g in gaps],
        })

    # Plot divergence gaps across scales
    if len(divergence_summary) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(divergence_summary)))

        for idx, entry in enumerate(divergence_summary):
            gaps = entry["gaps_per_step"]
            xs = list(range(1, len(gaps) + 1))
            ax.plot(xs, gaps, marker="o", linewidth=2, color=colors[idx],
                    label=f"{entry['model_tag']} (max gap={entry['max_gap']:.3f} @ step {entry['max_gap_step']})")

        ax.set_xlabel("Step k", fontsize=12)
        ax.set_ylabel("PRM Gap (correct - wrong)", fontsize=12)
        ax.set_title("Correct vs Wrong Divergence per Step Across Scales", fontsize=13)
        ax.axhline(y=0.1, linestyle="--", color="gray", alpha=0.5, label="Gap=0.1 threshold")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / "divergence_across_scales.png", dpi=200)
        plt.close()

    (out_dir / "divergence_summary.json").write_text(
        json.dumps(divergence_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ── Analysis 4: Average steps count and token efficiency ──

def analyze_step_statistics(
    all_data: Dict[str, dict],
    out_dir: Path,
):
    """Compare chain length and token efficiency across model sizes."""
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_tag, data in sorted(all_data.items(), key=lambda x: _model_params(x[0])):
        samples = data["samples"]
        correct = [s for s in samples if s["exact_match"] >= 1]
        wrong = [s for s in samples if s["exact_match"] < 1]

        row = {
            "model_tag": model_tag,
            "n_samples": len(samples),
            "accuracy": round(len(correct) / max(len(samples), 1), 4),
            "avg_steps_all": round(np.mean([s["n_steps"] for s in samples]), 2) if samples else 0,
            "avg_steps_correct": round(np.mean([s["n_steps"] for s in correct]), 2) if correct else 0,
            "avg_steps_wrong": round(np.mean([s["n_steps"] for s in wrong]), 2) if wrong else 0,
        }

        total_tokens = [sum(s.get("step_token_len", [])) for s in samples if s.get("step_token_len")]
        if total_tokens:
            row["avg_tokens_all"] = round(np.mean(total_tokens), 1)

        rows.append(row)

    (out_dir / "step_statistics.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if len(rows) >= 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        tags = [r["model_tag"] for r in rows]
        params = [_model_params(t) for t in tags]
        accs = [r["accuracy"] for r in rows]
        avg_steps_w = [r["avg_steps_wrong"] for r in rows]

        ax1.plot(params, accs, "o-", linewidth=2, markersize=8, color="blue")
        for p, a, t in zip(params, accs, tags):
            ax1.annotate(t, (p, a), textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=9)
        ax1.set_xscale("log")
        ax1.set_xlabel("Model Parameters")
        ax1.set_ylabel("MATH-500 Accuracy")
        ax1.set_title("Accuracy Scaling")
        ax1.grid(alpha=0.3)

        ax2.plot(params, avg_steps_w, "s-", linewidth=2, markersize=8, color="red")
        for p, s, t in zip(params, avg_steps_w, tags):
            ax2.annotate(t, (p, s), textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=9)
        ax2.set_xscale("log")
        ax2.set_xlabel("Model Parameters")
        ax2.set_ylabel("Avg Steps (wrong samples)")
        ax2.set_title("Chain Length Scaling (Wrong)")
        ax2.grid(alpha=0.3)

        plt.suptitle("Basic Statistics Across Model Scales", fontsize=14)
        plt.tight_layout()
        plt.savefig(out_dir / "step_statistics_scaling.png", dpi=200)
        plt.close()


def main():
    args = parse_args()
    model_tags = [t.strip() for t in args.models.split(",") if t.strip()]
    prm_dir = Path(args.prm_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data = {}
    for tag in model_tags:
        data = load_prm_data(prm_dir, tag)
        if data:
            all_data[tag] = data
        else:
            print(f"[SKIP] {tag}: no PRM data")

    if not all_data:
        print("No data loaded. Run 22_multi_scale_baseline.py first.")
        return

    print(f"Loaded PRM data for: {list(all_data.keys())}")
    print()

    analyze_chain_length(all_data, out_dir / "chain_length")
    analyze_step1_predictiveness(all_data, out_dir / "step1_predict")
    analyze_divergence_point(all_data, out_dir / "divergence")
    analyze_step_statistics(all_data, out_dir / "statistics")

    print(f"\nAll analyses saved to {out_dir}")


if __name__ == "__main__":
    main()
