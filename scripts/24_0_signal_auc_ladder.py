#!/usr/bin/env python3
"""
Experiment 4: Signal combination AUC ladder.

Evaluates step-level error detection AUC for progressively richer signals:
  1. Single entropy (mean_entropy only)
  2. Logistic combo (all 6 metrics)
  3. +Self-eval (add P(Yes)/P(No) as 7th feature)
  4. +Small MLP head (2-layer MLP on hidden states)

Usage:
    python scripts/24_0_signal_auc_ladder.py
    python scripts/24_0_signal_auc_ladder.py --dataset gsm8k
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.keystep_utils import load_jsonl

FIG_DIR = ROOT / "figures" / "auc_ladder"
FIG_DIR.mkdir(parents=True, exist_ok=True)

METRIC_NAMES = [
    "mean_entropy", "max_entropy", "entropy_delta",
    "mean_logprob", "min_logprob", "logprob_drop",
]


def _ms(mid):
    return mid.split("/")[-1].lower().replace("-", "_")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    return ap.parse_args()


def load_step_data(dataset, model_short):
    psm = ROOT / f"results/{dataset}_3b_multi_sample/step_uncertainty/per_step_metrics.jsonl"
    pareto = ROOT / f"results/{dataset}_{model_short}_pareto/step_metrics.jsonl"
    fe = ROOT / f"results/{dataset}_3b_multi_sample/first_error/gpt_first_error_cache.jsonl"
    if psm.exists():
        return load_jsonl(psm)
    if pareto.exists() and fe.exists():
        sm = load_jsonl(pareto)
        fe_recs = load_jsonl(fe)
        fe_map = {r["doc_id"]: r["tau"] for r in fe_recs if r.get("tau") is not None}
        rows = []
        for rec in sm:
            did = rec["doc_id"]
            tau = fe_map.get(did)
            if tau is None:
                continue
            for si, m in enumerate(rec.get("step_metrics", [])):
                row = dict(m)
                row.update(doc_id=did, sample_idx=0, step_idx=si,
                           tau=tau, n_steps=len(rec["step_metrics"]),
                           is_error_step=(si == tau))
                rows.append(row)
        return rows
    return []


def load_self_eval(dataset, model_short):
    se = ROOT / f"results/{dataset}_{model_short}_self_eval/self_eval_scores.jsonl"
    if not se.exists():
        return {}
    return {(r["doc_id"], r.get("sample_idx", 0), r["step_idx"]): r
            for r in load_jsonl(se)}


def load_mlp_predictions(model_short):
    p = ROOT / f"results/{model_short}_mlp_head/mlp_head_summary.json"
    if p.exists():
        return json.loads(p.read_text("utf-8"))
    return None


def compute_auc(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return 0.5


def run_ladder(dataset, model_id):
    ms = _ms(model_id)
    print(f"\n{'='*60}\n  AUC Ladder: {model_id} x {dataset}\n{'='*60}")

    rows = load_step_data(dataset, ms)
    if not rows:
        print("  No step data")
        return None

    filtered = [r for r in rows
                if r.get("tau") is not None and r["tau"] >= 0
                and r["step_idx"] <= r["tau"]]
    rows = filtered
    print(f"  {len(rows)} step samples")

    y = np.array([1.0 if r.get("is_error_step") or r["step_idx"] == r["tau"]
                  else 0.0 for r in rows])
    print(f"  Positive rate: {y.mean():.4f}")
    results = {}

    # Level 1: Single entropy
    X1 = np.array([r.get("mean_entropy", 0.0) for r in rows])
    auc1 = compute_auc(y, X1)
    results["single_entropy"] = auc1
    print(f"  [1] Single entropy:    AUC = {auc1:.4f}")

    # Level 2: Logistic combo
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    X6 = np.array([[r.get(m, 0.0) or 0.0 for m in METRIC_NAMES] for r in rows])
    sc = StandardScaler().fit(X6)
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(X6), y)
    s6 = clf.predict_proba(sc.transform(X6))[:, 1]
    auc2 = compute_auc(y, s6)
    cv = cross_val_score(LogisticRegression(max_iter=1000, C=1.0),
                         sc.transform(X6), y, cv=5, scoring="roc_auc")
    results["logistic_combo"] = auc2
    results["logistic_combo_cv"] = float(np.mean(cv))
    print(f"  [2] Logistic combo:    AUC = {auc2:.4f} (CV: {np.mean(cv):.4f})")

    # Level 3: +Self-eval
    se = load_self_eval(dataset, ms)
    if se:
        X7r, y7 = [], []
        for i, r in enumerate(rows):
            key = (r["doc_id"], r.get("sample_idx", 0), r["step_idx"])
            s = se.get(key)
            if s is None:
                continue
            feats = [r.get(m, 0.0) or 0.0 for m in METRIC_NAMES]
            feats.append(s.get("self_eval_score", 0.5))
            X7r.append(feats)
            y7.append(y[i])
        if len(X7r) > 50:
            X7 = np.array(X7r)
            y7 = np.array(y7)
            sc7 = StandardScaler().fit(X7)
            clf7 = LogisticRegression(max_iter=1000, C=1.0).fit(sc7.transform(X7), y7)
            auc3 = compute_auc(y7, clf7.predict_proba(sc7.transform(X7))[:, 1])
            results["logistic_self_eval"] = auc3
            print(f"  [3] +Self-eval:        AUC = {auc3:.4f}")
        else:
            print(f"  [3] +Self-eval:        insufficient data")
    else:
        print(f"  [3] +Self-eval:        no data (run 24_1 first)")

    # Level 4: MLP head
    mlp = load_mlp_predictions(ms)
    if mlp:
        auc4 = mlp.get("mlp_test_auc", 0.0)
        results["mlp_head"] = auc4
        results["mlp_head_logistic"] = mlp.get("logistic_test_auc", 0.0)
        print(f"  [4] MLP head:          AUC = {auc4:.4f}")
    else:
        print(f"  [4] MLP head:          no data (run 24_2 first)")

    return results


def plot_ladder(all_results):
    levels = [
        ("single_entropy", "Single Entropy"),
        ("logistic_combo", "Logistic Combo (6 metrics)"),
        ("logistic_self_eval", "+Self-Eval"),
        ("mlp_head", "+MLP Head"),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(levels))
    width = 0.35
    for gi, (key, results) in enumerate(all_results.items()):
        aucs = [results.get(lk, 0.0) for lk, _ in levels]
        offset = (gi - len(all_results) / 2 + 0.5) * width
        bars = ax.bar(x_pos + offset, aucs, width, label=key, alpha=0.85)
        for bar, auc in zip(bars, aucs):
            if auc > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{auc:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([l[1] for l in levels], fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Step-Level AUC", fontsize=10)
    ax.set_title("Signal Combination AUC Ladder", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.45, 0.85)
    ax.axhline(y=0.5, color="gray", ls=":", lw=1, alpha=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_auc_ladder.{ext}", dpi=200, bbox_inches="tight")
    print(f"\nSaved fig_auc_ladder to {FIG_DIR}")
    plt.close(fig)


def main():
    args = parse_args()
    all_results = {}
    datasets = [args.dataset] if args.dataset else ["gsm8k", "math500"]
    for ds in datasets:
        key = f"{_ms(args.model_id)}_{ds}"
        r = run_ladder(ds, args.model_id)
        if r:
            all_results[key] = r
    if all_results:
        plot_ladder(all_results)
        sf = FIG_DIR / "auc_ladder_summary.json"
        sf.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"Summary -> {sf}")


if __name__ == "__main__":
    main()
