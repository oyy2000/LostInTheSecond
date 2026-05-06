#!/usr/bin/env python3
"""
Oracle analysis: analyse the per-step intervention gain dataset,
train a lightweight gain predictor, and produce figures.

Usage:
    python scripts/8_3_oracle_analysis.py \
        --oracle-file results/gsm8k_3b_multi_sample/keystep_sd/oracle/oracle_dataset.jsonl \
        --draft-file results/gsm8k_3b_multi_sample/keystep_sd/small_drafts.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.keystep_utils import detect_stall, load_jsonl

def parse_args():
    ap = argparse.ArgumentParser(description="Oracle analysis & gain predictor")
    ap.add_argument("--oracle-file", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/keystep_sd/oracle/oracle_dataset.jsonl"))
    ap.add_argument("--draft-file", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/keystep_sd/small_drafts.jsonl"))
    ap.add_argument("--fig-dir", default=str(PROJECT_ROOT / "figures/keystep_sd"))
    ap.add_argument("--out-dir", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/keystep_sd/oracle"))
    return ap.parse_args()

def build_features(oracle_records, draft_map):
    """Build feature matrix X and label vector y from oracle data.

    Features per (question, step_k):
      0: mean_neglogprob (uncertainty)
      1: entropy
      2: relative position k / n_steps
      3: stall signal (0/1)
      4: step length in chars
      5: n_steps in the chain
    """
    X, y = [], []
    for rec in oracle_records:
        doc_id = rec["doc_id"]
        k = rec["step_idx"]
        draft = draft_map.get(doc_id)
        if not draft:
            continue
        step_stats = draft.get("step_stats", [])
        steps = draft.get("steps", [])
        n_steps = len(steps)
        if k >= len(step_stats) or k >= n_steps:
            continue

        ss = step_stats[k]
        step_text = steps[k]
        feats = [
            ss.get("mean_neglogprob", 0.0),
            ss.get("entropy", 0.0),
            (k + 1) / max(n_steps, 1),
            detect_stall(step_text),
            len(step_text),
            n_steps,
        ]
        X.append(feats)
        y.append(rec.get("delta_k", 0.0))
    return np.array(X), np.array(y)

def train_gain_predictor(X, y, out_dir):
    """Train a logistic regression (binary: gain > 0) and a ridge regression."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    import pickle

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    y_bin = (y > 0).astype(int)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    auc_scores = cross_val_score(clf, Xs, y_bin, cv=5, scoring="roc_auc")
    clf.fit(Xs, y_bin)
    print(f"  Binary classifier (gain > 0): "
          f"5-fold AUC = {auc_scores.mean():.4f} +/- {auc_scores.std():.4f}")

    reg = Ridge(alpha=1.0)
    r2_scores = cross_val_score(reg, Xs, y, cv=5, scoring="r2")
    reg.fit(Xs, y)
    print(f"  Regression (recovery rate): "
          f"5-fold R2 = {r2_scores.mean():.4f} +/- {r2_scores.std():.4f}")

    feature_names = [
        "mean_neglogprob", "entropy", "rel_position",
        "stall", "step_len", "n_steps",
    ]
    print("\n  Feature importances (logistic coef):")
    for name, coef in zip(feature_names, clf.coef_[0]):
        print(f"    {name:20s}: {coef:+.4f}")

    out_path = Path(out_dir)
    with open(out_path / "gain_predictor_clf.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "clf": clf}, f)
    with open(out_path / "gain_predictor_reg.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "reg": reg}, f)

    results = {
        "binary_auc_mean": round(float(auc_scores.mean()), 4),
        "binary_auc_std": round(float(auc_scores.std()), 4),
        "regression_r2_mean": round(float(r2_scores.mean()), 4),
        "regression_r2_std": round(float(r2_scores.std()), 4),
        "feature_names": feature_names,
        "logistic_coefs": [round(float(c), 4) for c in clf.coef_[0]],
    }
    (out_path / "predictor_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    return results

def plot_gain_by_position(oracle_records, fig_dir):
    """Bar chart: mean recovery rate by relative step position."""
    bins = np.linspace(0, 1, 6)
    bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    bin_gains = [[] for _ in range(len(bins)-1)]

    for rec in oracle_records:
        steps = rec.get("step_idx", 0)
        n_steps = 6
        rel = (steps + 1) / n_steps
        delta = rec.get("delta_k", 0.0)
        for bi in range(len(bins)-1):
            if bins[bi] <= rel < bins[bi+1] or (bi == len(bins)-2 and rel == bins[bi+1]):
                bin_gains[bi].append(delta)
                break

    means = [np.mean(g) if g else 0 for g in bin_gains]
    stds = [np.std(g) / max(np.sqrt(len(g)), 1) if g else 0 for g in bin_gains]
    counts = [len(g) for g in bin_gains]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(bin_labels))
    bars = ax.bar(x, means, yerr=stds, width=0.6, color="#3498db",
                  edgecolor="white", capsize=3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Relative step position (k / n_steps)")
    ax.set_ylabel("Mean recovery rate (Delta_k)")
    ax.set_title("Intervention Gain by Step Position")
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"n={c}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    fig.savefig(Path(fig_dir) / "gain_by_position.pdf", dpi=150)
    fig.savefig(Path(fig_dir) / "gain_by_position.png", dpi=150)
    plt.close(fig)
    print(f"  -> gain_by_position.pdf")


def plot_gain_distribution(oracle_records, fig_dir):
    """Histogram of Delta_k values."""
    deltas = [r.get("delta_k", 0.0) for r in oracle_records]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(deltas, bins=30, color="#e74c3c", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Recovery rate (Delta_k)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Per-Step Intervention Gain")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    mean_d = np.mean(deltas)
    ax.axvline(mean_d, color="#2c3e50", linewidth=1, linestyle="-",
               label=f"mean={mean_d:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(fig_dir) / "gain_distribution.pdf", dpi=150)
    fig.savefig(Path(fig_dir) / "gain_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  -> gain_distribution.pdf")


def plot_uncertainty_vs_gain(X, y, fig_dir):
    """Scatter: uncertainty (mean_neglogprob) vs Delta_k."""
    if len(X) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[:, 0], y, s=8, alpha=0.3, color="#3498db")
    ax.set_xlabel("Mean negative log-prob (uncertainty)")
    ax.set_ylabel("Recovery rate (Delta_k)")
    ax.set_title("Uncertainty vs Intervention Gain")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    fig.savefig(Path(fig_dir) / "uncertainty_vs_gain.pdf", dpi=150)
    fig.savefig(Path(fig_dir) / "uncertainty_vs_gain.png", dpi=150)
    plt.close(fig)
    print(f"  -> uncertainty_vs_gain.pdf")

def main():
    args = parse_args()
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    oracle_file = Path(args.oracle_file)
    if not oracle_file.exists():
        print(f"ERROR: {oracle_file} not found. Run 8_2 first.")
        sys.exit(1)
    oracle_records = load_jsonl(oracle_file)
    print(f"Loaded {len(oracle_records)} oracle records")

    drafts = load_jsonl(Path(args.draft_file)) if Path(args.draft_file).exists() else []
    draft_map = {d["doc_id"]: d for d in drafts}

    deltas = [r.get("delta_k", 0.0) for r in oracle_records]
    print(f"\nOracle summary:")
    print(f"  Total interventions: {len(oracle_records)}")
    print(f"  Mean Delta_k: {np.mean(deltas):.4f}")
    print(f"  Positive gain (Delta > 0): "
          f"{sum(1 for d in deltas if d > 0)}/{len(deltas)} "
          f"({sum(1 for d in deltas if d > 0)/max(len(deltas),1):.1%})")

    X, y = build_features(oracle_records, draft_map)
    print(f"  Feature matrix: {X.shape}")

    print("\nGenerating figures...")
    plot_gain_by_position(oracle_records, fig_dir)
    plot_gain_distribution(oracle_records, fig_dir)
    if len(X) > 0:
        plot_uncertainty_vs_gain(X, y, fig_dir)

    if len(X) >= 20:
        print("\nTraining gain predictor...")
        train_gain_predictor(X, y, out_dir)
    else:
        print(f"\nSkipping predictor training (only {len(X)} samples)")

    print("\nDone.")


if __name__ == "__main__":
    main()
