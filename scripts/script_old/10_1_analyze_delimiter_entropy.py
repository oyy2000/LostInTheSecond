#!/usr/bin/env python3
"""
Analyze step-start entropy at error vs. non-error positions.

Reads token_logprobs_wrong.jsonl (and optionally token_logprobs_correct.jsonl)
from Phase 0, computes H_bar_t(K) for each step, and runs:
  1. Within-trajectory comparison: H_tau vs mean(H_other)
  2. Aligned-to-tau profile: mean entropy at relative position k = t - tau
  3. Matched correct-trajectory control (if available)
  4. Statistical tests (Wilcoxon, paired t-test, Mann-Whitney U)

Usage:
    python scripts/10_1_analyze_delimiter_entropy.py [--K 5]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.delimiter_entropy import (
    aggregate_trajectory,
    label_steps,
    step_char_boundaries,
    step_start_entropy,
)
from src.keystep_utils import load_jsonl, write_jsonl

DEFAULT_DIR = str(
    PROJECT_ROOT / "results/gsm8k_3b_multi_sample/delimiter_entropy"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze step-start entropy")
    ap.add_argument("--data-dir", default=DEFAULT_DIR)
    ap.add_argument("--K", type=int, default=5,
                    help="Number of first content tokens per step")
    return ap.parse_args()


def process_wrong(records, K):
    """Compute per-step entropy for wrong trajectories, return per-step rows + summaries."""
    per_step_rows = []
    trajectory_summaries = []
    skipped = 0

    for rec in records:
        tau = rec["tau"]
        steps = rec["steps"]
        response = rec["response"]
        token_lps = rec.get("token_logprobs", [])
        token_offs = rec.get("token_offsets", [])

        if len(steps) < 2 or not token_lps:
            skipped += 1
            continue

        boundaries = step_char_boundaries(response, steps)
        sstats = step_start_entropy(token_lps, token_offs, boundaries, K=K)
        sstats = label_steps(sstats, tau)

        for s in sstats:
            per_step_rows.append({
                "doc_id": rec["doc_id"],
                "sample_idx": rec["sample_idx"],
                "tau": tau,
                "is_correct": False,
                "step_idx": s["step_idx"],
                "is_error_step": s["is_error_step"],
                "relative_to_tau": s["relative_to_tau"],
                "mean_entropy_K": s["mean_entropy_K"],
                "mean_neglogprob_K": s["mean_neglogprob_K"],
                "whole_step_entropy": s["whole_step_entropy"],
                "n_tokens_used": s["n_tokens_used"],
                "n_steps": rec["n_steps"],
            })

        summary = aggregate_trajectory(
            sstats, tau, rec["doc_id"], rec["sample_idx"], rec["n_steps"],
        )
        if summary is not None:
            trajectory_summaries.append(summary)

    print(f"Wrong: processed {len(records)}, skipped {skipped}")
    print(f"  Per-step rows: {len(per_step_rows)}")
    print(f"  Trajectory summaries: {len(trajectory_summaries)}")
    return per_step_rows, trajectory_summaries


def process_correct(records, K):
    """Compute per-step entropy for correct trajectories (matched control)."""
    per_step_rows = []
    skipped = 0

    for rec in records:
        steps = rec["steps"]
        response = rec["response"]
        token_lps = rec.get("token_logprobs", [])
        token_offs = rec.get("token_offsets", [])

        if len(steps) < 2 or not token_lps:
            skipped += 1
            continue

        boundaries = step_char_boundaries(response, steps)
        sstats = step_start_entropy(token_lps, token_offs, boundaries, K=K)

        for i, s in enumerate(sstats):
            per_step_rows.append({
                "doc_id": rec["doc_id"],
                "sample_idx": rec["sample_idx"],
                "tau": None,
                "is_correct": True,
                "step_idx": i,
                "is_error_step": False,
                "relative_to_tau": None,
                "mean_entropy_K": s["mean_entropy_K"],
                "mean_neglogprob_K": s["mean_neglogprob_K"],
                "whole_step_entropy": s["whole_step_entropy"],
                "n_tokens_used": s["n_tokens_used"],
                "n_steps": rec["n_steps"],
                "relative_position": round(i / max(rec["n_steps"] - 1, 1), 4),
            })

    print(f"Correct: processed {len(records)}, skipped {skipped}")
    print(f"  Per-step rows: {len(per_step_rows)}")
    return per_step_rows


def run_stats(trajectory_summaries, per_step_wrong, per_step_correct):
    """Run statistical tests and print results."""
    if not trajectory_summaries:
        print("No trajectory summaries to analyze.")
        return {}

    deltas = [s["delta_entropy"] for s in trajectory_summaries]
    error_ents = [s["error_entropy"] for s in trajectory_summaries]
    other_ents = [s["mean_other_entropy"] for s in trajectory_summaries]

    print(f"\n{'='*60}")
    print("Step-Start Entropy: Error Step vs. Other Steps")
    print(f"{'='*60}")
    print(f"N trajectories: {len(trajectory_summaries)}")
    print(f"\nH_bar(K) at error step:   "
          f"mean={np.mean(error_ents):.4f}, "
          f"median={np.median(error_ents):.4f}, "
          f"std={np.std(error_ents):.4f}")
    print(f"H_bar(K) at other steps:  "
          f"mean={np.mean(other_ents):.4f}, "
          f"median={np.median(other_ents):.4f}, "
          f"std={np.std(other_ents):.4f}")
    print(f"\nDelta (error - other):    "
          f"mean={np.mean(deltas):.4f}, "
          f"median={np.median(deltas):.4f}")

    nonzero_deltas = [d for d in deltas if d != 0.0]
    if len(nonzero_deltas) >= 10:
        wilcoxon = sp_stats.wilcoxon(nonzero_deltas, alternative="two-sided")
        print(f"\nWilcoxon signed-rank:     "
              f"stat={wilcoxon.statistic:.2f}, p={wilcoxon.pvalue:.6f}")
        wilcoxon_p = float(wilcoxon.pvalue)
    else:
        print("\nWilcoxon: too few non-zero deltas")
        wilcoxon_p = None

    ttest = sp_stats.ttest_rel(error_ents, other_ents)
    print(f"Paired t-test:            "
          f"t={ttest.statistic:.4f}, p={ttest.pvalue:.6f}")

    err_rows = [r["mean_entropy_K"] for r in per_step_wrong if r["is_error_step"]]
    non_rows = [r["mean_entropy_K"] for r in per_step_wrong if not r["is_error_step"]]
    if err_rows and non_rows:
        mwu = sp_stats.mannwhitneyu(err_rows, non_rows, alternative="two-sided")
        print(f"\nMann-Whitney U (unpaired): "
              f"U={mwu.statistic:.2f}, p={mwu.pvalue:.6f}")

    frac_positive = np.mean([d > 0 for d in deltas])
    print(f"\nFraction with H_tau > H_other: {frac_positive:.4f}")

    result = {
        "n_trajectories": len(trajectory_summaries),
        "mean_error_entropy": round(float(np.mean(error_ents)), 6),
        "mean_other_entropy": round(float(np.mean(other_ents)), 6),
        "mean_delta_entropy": round(float(np.mean(deltas)), 6),
        "median_delta_entropy": round(float(np.median(deltas)), 6),
        "ttest_p": round(float(ttest.pvalue), 8),
        "frac_error_higher": round(float(frac_positive), 4),
    }
    if wilcoxon_p is not None:
        result["wilcoxon_p"] = round(wilcoxon_p, 8)

    if per_step_correct:
        wrong_by_pos = {}
        for r in per_step_wrong:
            if r["is_error_step"]:
                tau = r["tau"]
                n = r.get("n_steps") or (r["step_idx"] + 1)
                if n is None:
                    continue
                rel = round(r["step_idx"] / max(n - 1, 1), 2)
                wrong_by_pos.setdefault(rel, []).append(r["mean_entropy_K"])

        correct_by_pos = {}
        for r in per_step_correct:
            rel = round(r.get("relative_position", 0), 2)
            correct_by_pos.setdefault(rel, []).append(r["mean_entropy_K"])

        matched_wrong, matched_correct = [], []
        for rel in sorted(wrong_by_pos.keys()):
            if rel in correct_by_pos:
                matched_wrong.extend(wrong_by_pos[rel])
                n_need = len(wrong_by_pos[rel])
                pool = correct_by_pos[rel]
                matched_correct.extend(pool[:n_need])

        if matched_wrong and matched_correct:
            mw_mean = np.mean(matched_wrong)
            mc_mean = np.mean(matched_correct)
            mwu2 = sp_stats.mannwhitneyu(
                matched_wrong, matched_correct, alternative="two-sided")
            print(f"\n--- Matched Control (correct vs wrong at same rel position) ---")
            print(f"Wrong error-step H:   mean={mw_mean:.4f} (n={len(matched_wrong)})")
            print(f"Correct matched H:    mean={mc_mean:.4f} (n={len(matched_correct)})")
            print(f"Mann-Whitney U:       U={mwu2.statistic:.2f}, p={mwu2.pvalue:.6f}")
            result["matched_wrong_mean"] = round(float(mw_mean), 6)
            result["matched_correct_mean"] = round(float(mc_mean), 6)
            result["matched_mwu_p"] = round(float(mwu2.pvalue), 8)

    return result


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    wrong_file = data_dir / "token_logprobs_wrong.jsonl"
    correct_file = data_dir / "token_logprobs_correct.jsonl"

    wrong_records = load_jsonl(wrong_file)
    if not wrong_records:
        print(f"No wrong records in {wrong_file}. Run 10_0 first.")
        return
    print(f"Loaded {len(wrong_records)} wrong records (K={args.K})")

    per_step_wrong, summaries = process_wrong(wrong_records, args.K)

    per_step_correct = []
    correct_records = load_jsonl(correct_file) if correct_file.exists() else []
    if correct_records:
        print(f"Loaded {len(correct_records)} correct records")
        per_step_correct = process_correct(correct_records, args.K)

    write_jsonl(data_dir / "per_step_wrong.jsonl", per_step_wrong)
    write_jsonl(data_dir / "trajectory_summaries.jsonl", summaries)
    if per_step_correct:
        write_jsonl(data_dir / "per_step_correct.jsonl", per_step_correct)
    print(f"\nSaved analysis files to {data_dir}")

    stats = run_stats(summaries, per_step_wrong, per_step_correct)

    summary_path = data_dir / "stats_summary.json"
    summary_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\nStats summary -> {summary_path}")


if __name__ == "__main__":
    main()
