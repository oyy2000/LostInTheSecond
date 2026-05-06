#!/usr/bin/env python3
"""
Step 4b + Step 5: Compute step-level uncertainty metrics and evaluate
error localization.

Metrics per step:
  - mean_entropy: average token entropy H(t) = -p*log(p)
  - max_entropy: max token entropy in the step
  - entropy_delta: H(t) - H(t-1)
  - mean_logprob: average token log-probability
  - min_logprob: minimum token log-probability in the step
  - logprob_drop: mean_logprob(t) - mean_logprob(t-1)

Localization evaluation:
  For each trajectory, predict the error step as the step that maximizes
  (or minimizes, depending on metric) the uncertainty signal.
  Report: exact match, within +/-1, within +/-2, Spearman rho.

Usage:
    python scripts/4_1_compute_uncertainty_and_evaluate.py
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.keystep_utils import load_jsonl, write_jsonl

DEFAULT_LOGPROB_FILE = str(
    PROJECT_ROOT / "results/gsm8k_3b_multi_sample/step_uncertainty/token_logprobs.jsonl"
)
DEFAULT_OUT_DIR = str(
    PROJECT_ROOT / "results/gsm8k_3b_multi_sample/step_uncertainty"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logprob-file", default=DEFAULT_LOGPROB_FILE)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return ap.parse_args()


def _token_entropy(logprob: float) -> float:
    if logprob >= 0:
        return 0.0
    p = math.exp(logprob)
    return -p * logprob


def step_char_boundaries(
    response: str, steps: List[str],
) -> List[Tuple[int, int]]:
    boundaries = []
    pos = 0
    for step in steps:
        idx = response.find(step, pos)
        if idx < 0:
            idx = pos
        boundaries.append((idx, idx + len(step)))
        pos = idx + len(step)
    return boundaries


def compute_step_metrics(
    token_logprobs: List[float],
    token_offsets: List[int],
    step_boundaries: List[Tuple[int, int]],
) -> List[Dict[str, float]]:
    """Compute per-step uncertainty metrics."""
    step_metrics = []
    ti = 0
    for s_start, s_end in step_boundaries:
        while ti < len(token_offsets) and token_offsets[ti] < s_start:
            ti += 1
        step_lps = []
        step_hs = []
        scan = ti
        while scan < len(token_offsets) and token_offsets[scan] < s_end:
            lp = token_logprobs[scan]
            step_lps.append(lp)
            step_hs.append(_token_entropy(lp))
            scan += 1

        if step_lps:
            m = {
                "mean_entropy": sum(step_hs) / len(step_hs),
                "max_entropy": max(step_hs),
                "mean_logprob": sum(step_lps) / len(step_lps),
                "min_logprob": min(step_lps),
                "n_tokens": len(step_lps),
            }
        else:
            m = {
                "mean_entropy": 0.0,
                "max_entropy": 0.0,
                "mean_logprob": 0.0,
                "min_logprob": 0.0,
                "n_tokens": 0,
            }
        step_metrics.append(m)

    for i, m in enumerate(step_metrics):
        if i == 0:
            m["entropy_delta"] = 0.0
            m["logprob_drop"] = 0.0
        else:
            m["entropy_delta"] = m["mean_entropy"] - step_metrics[i - 1]["mean_entropy"]
            m["logprob_drop"] = m["mean_logprob"] - step_metrics[i - 1]["mean_logprob"]

    return step_metrics


METRIC_CONFIGS = {
    "mean_entropy":  {"higher_is_worse": True},
    "max_entropy":   {"higher_is_worse": True},
    "entropy_delta": {"higher_is_worse": True},
    "mean_logprob":  {"higher_is_worse": False},
    "min_logprob":   {"higher_is_worse": False},
    "logprob_drop":  {"higher_is_worse": False},
}


def predict_error_step(
    step_metrics: List[Dict[str, float]],
    metric_name: str,
    higher_is_worse: bool,
) -> int:
    """Predict the 1-indexed error step using the given metric."""
    values = [m[metric_name] for m in step_metrics]
    if higher_is_worse:
        pred_idx = int(np.argmax(values))
    else:
        pred_idx = int(np.argmin(values))
    return pred_idx + 1


def evaluate_localization(
    records: List[Dict],
    step_metrics_all: List[List[Dict[str, float]]],
) -> Dict[str, Any]:
    """Evaluate error localization for all metrics."""
    results = {}
    for metric_name, cfg in METRIC_CONFIGS.items():
        preds = []
        golds = []
        for rec, smetrics in zip(records, step_metrics_all):
            tau = rec["tau"]
            if tau is None or tau < 1:
                continue
            pred = predict_error_step(smetrics, metric_name, cfg["higher_is_worse"])
            preds.append(pred)
            golds.append(tau)

        if not preds:
            continue

        preds_arr = np.array(preds)
        golds_arr = np.array(golds)
        diffs = np.abs(preds_arr - golds_arr)

        exact = float(np.mean(diffs == 0))
        within_1 = float(np.mean(diffs <= 1))
        within_2 = float(np.mean(diffs <= 2))

        if len(set(preds)) > 1 and len(set(golds)) > 1:
            rho, rho_p = sp_stats.spearmanr(preds, golds)
        else:
            rho, rho_p = 0.0, 1.0

        results[metric_name] = {
            "n": len(preds),
            "exact_match": round(exact, 4),
            "within_1": round(within_1, 4),
            "within_2": round(within_2, 4),
            "spearman_rho": round(float(rho), 4),
            "spearman_p": round(float(rho_p), 6),
            "mean_abs_error": round(float(np.mean(diffs)), 4),
            "median_abs_error": round(float(np.median(diffs)), 4),
        }
    return results


def print_results(results: Dict[str, Any]) -> None:
    header = f"{'Metric':<18} {'N':>5} {'Exact':>7} {'+-1':>7} {'+-2':>7} {'Spearman':>9} {'MAE':>7}"
    print(f"\n{'=' * 70}")
    print("Error Localization via Step-Level Uncertainty")
    print(f"{'=' * 70}")
    print(header)
    print("-" * 70)
    for metric_name, r in results.items():
        print(f"{metric_name:<18} {r['n']:>5} {r['exact_match']:>7.3f} "
              f"{r['within_1']:>7.3f} {r['within_2']:>7.3f} "
              f"{r['spearman_rho']:>9.4f} {r['mean_abs_error']:>7.3f}")
    print(f"{'=' * 70}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(Path(args.logprob_file))
    if not records:
        print(f"No records in {args.logprob_file}. Run 4_0 first.")
        return
    print(f"Loaded {len(records)} records")

    step_metrics_all = []
    per_step_rows = []
    skipped = 0

    for rec in records:
        steps = rec["steps"]
        response = rec["response"]
        token_lps = rec.get("token_logprobs", [])
        token_offs = rec.get("token_offsets", [])

        if len(steps) < 2 or not token_lps:
            skipped += 1
            step_metrics_all.append([])
            continue

        boundaries = step_char_boundaries(response, steps)
        smetrics = compute_step_metrics(token_lps, token_offs, boundaries)
        step_metrics_all.append(smetrics)

        for i, m in enumerate(smetrics):
            tau = rec["tau"]
            per_step_rows.append({
                "doc_id": rec["doc_id"],
                "sample_idx": rec["sample_idx"],
                "tau": tau,
                "n_steps": rec["n_steps"],
                "step_idx": i,
                "is_error_step": (tau is not None and i == tau - 1),
                "relative_to_tau": (i - (tau - 1)) if tau is not None else None,
                **{k: round(v, 6) for k, v in m.items() if k != "n_tokens"},
                "n_tokens": m["n_tokens"],
            })

    print(f"Processed: {len(records) - skipped}, skipped: {skipped}")

    write_jsonl(out_dir / "per_step_metrics.jsonl", per_step_rows)
    print(f"Saved per-step metrics -> {out_dir / 'per_step_metrics.jsonl'}")

    valid = [(r, sm) for r, sm in zip(records, step_metrics_all) if sm]
    valid_records = [v[0] for v in valid]
    valid_metrics = [v[1] for v in valid]

    results = evaluate_localization(valid_records, valid_metrics)
    print_results(results)

    summary_path = out_dir / "localization_results.json"
    summary_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults -> {summary_path}")

    traj_summaries = []
    for rec, smetrics in zip(valid_records, valid_metrics):
        tau = rec["tau"]
        if tau is None or tau < 1 or tau > len(smetrics):
            continue
        err_idx = tau - 1
        err_m = smetrics[err_idx]
        other_ms = [m for i, m in enumerate(smetrics) if i != err_idx]
        if not other_ms:
            continue

        summary = {
            "doc_id": rec["doc_id"],
            "sample_idx": rec["sample_idx"],
            "tau": tau,
            "n_steps": rec["n_steps"],
        }
        for metric_name in METRIC_CONFIGS:
            err_val = err_m[metric_name]
            other_vals = [m[metric_name] for m in other_ms]
            summary[f"error_{metric_name}"] = round(err_val, 6)
            summary[f"mean_other_{metric_name}"] = round(
                sum(other_vals) / len(other_vals), 6)
            summary[f"delta_{metric_name}"] = round(
                err_val - sum(other_vals) / len(other_vals), 6)
        traj_summaries.append(summary)

    write_jsonl(out_dir / "trajectory_summaries.jsonl", traj_summaries)
    print(f"Trajectory summaries -> {out_dir / 'trajectory_summaries.jsonl'}")


if __name__ == "__main__":
    main()
