#!/usr/bin/env python3
"""
Analyze bad-prefix natural recovery results and produce figures.

Reads continuations.jsonl and computes:
  - R_bad(early) vs R_bad(late) overall
  - R_bad(tau) for each tau value
  - Per-sample recovery rate distribution

Usage:
    python scripts/6_2_analyze_bad_prefix_recovery.py \
        --input results/gsm8k_3b_multi_sample/bad_prefix_recovery/continuations.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze bad-prefix recovery")
    ap.add_argument("--input", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/bad_prefix_recovery/continuations.jsonl"))
    ap.add_argument("--fig-dir", default=str(
        PROJECT_ROOT / "figures/bad_prefix_recovery"))
    ap.add_argument("--out-summary", default="")
    return ap.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run 6_1 first.")
        sys.exit(1)

    results = [json.loads(l) for l in input_path.read_text("utf-8").splitlines() if l.strip()]
    print(f"Loaded {len(results)} continuation results")

    by_sample = defaultdict(list)
    for r in results:
        key = (r["doc_id"], r["sample_idx"])
        by_sample[key].append(r)

    sample_stats = []
    for key, recs in by_sample.items():
        n_correct = sum(1 for r in recs if r["exact_match"] >= 1.0)
        n_total = len(recs)
        sample_stats.append({
            "doc_id": recs[0]["doc_id"],
            "sample_idx": recs[0]["sample_idx"],
            "bucket": recs[0]["bucket"],
            "tau": recs[0]["tau"],
            "n_steps": recs[0]["n_steps"],
            "recovery_rate": n_correct / n_total,
            "n_correct": n_correct,
            "n_total": n_total,
        })

    bucket_rates = defaultdict(list)
    for s in sample_stats:
        bucket_rates[s["bucket"]].append(s["recovery_rate"])

    print("\n=== Recovery by bucket ===")
    summary = {}
    for bucket in ["early", "late"]:
        rates = bucket_rates[bucket]
        if rates:
            mean_r = np.mean(rates)
            se = np.std(rates) / np.sqrt(len(rates))
            summary[bucket] = {
                "mean": float(mean_r),
                "se": float(se),
                "n_samples": len(rates),
                "ci95_low": float(mean_r - 1.96 * se),
                "ci95_high": float(mean_r + 1.96 * se),
            }
            print(f"  R_bad({bucket}): {mean_r:.4f} +/- {se:.4f} (n={len(rates)})")

    tau_rates = defaultdict(list)
    for s in sample_stats:
        tau_rates[s["tau"]].append(s["recovery_rate"])

    print("\n=== Recovery by tau ===")
    tau_summary = {}
    for tau in sorted(tau_rates.keys()):
        rates = tau_rates[tau]
        mean_r = np.mean(rates)
        se = np.std(rates) / np.sqrt(len(rates))
        tau_summary[tau] = {"mean": float(mean_r), "se": float(se), "n_samples": len(rates)}
        print(f"  R_bad(tau={tau}): {mean_r:.4f} +/- {se:.4f} (n={len(rates)})")

    # Figure 1: Bar chart early vs late
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    buckets_list = ["early", "late"]
    means = [summary[b]["mean"] for b in buckets_list]
    ses = [summary[b]["se"] for b in buckets_list]
    colors = ["#e74c3c", "#2ecc71"]
    bars = ax.bar(buckets_list, means, yerr=[1.96 * s for s in ses],
                  capsize=5, color=colors, edgecolor="black", linewidth=0.8, width=0.5)
    ax.set_ylabel("Recovery Rate $R_{bad}$")
    ax.set_xlabel("Error Position")
    ax.set_title("Natural Recovery from Bad Prefix")
    ax.set_ylim(0, max(means) * 1.5 + 0.05)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{m:.3f}", ha="center", va="bottom", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_recovery_by_bucket.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(fig_dir / "fig_recovery_by_bucket.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {fig_dir / 'fig_recovery_by_bucket.pdf'}")

    # Figure 2: Line plot by tau
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    taus = sorted(tau_summary.keys())
    means_tau = [tau_summary[t]["mean"] for t in taus]
    ses_tau = [tau_summary[t]["se"] for t in taus]
    ns_tau = [tau_summary[t]["n_samples"] for t in taus]
    ax.errorbar(taus, means_tau, yerr=[1.96 * s for s in ses_tau],
                marker="o", capsize=4, linewidth=2, markersize=7, color="#2c3e50")
    for t, m, n in zip(taus, means_tau, ns_tau):
        ax.annotate(f"n={n}", (t, m), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8, color="gray")
    ax.set_xlabel("First Error Step ($\\tau$)")
    ax.set_ylabel("Recovery Rate $R_{bad}(\\tau)$")
    ax.set_title("Recovery Rate vs. Error Position")
    ax.set_xticks(taus)
    ax.set_ylim(0, max(means_tau) * 1.4 + 0.02)
    ax.axvspan(1.5, 3.5, alpha=0.08, color="red", label="early ($\\tau$=2,3)")
    ax.axvspan(3.5, 6.5, alpha=0.08, color="green", label="late ($\\tau$=4,5,6)")
    ax.legend(loc="upper left", framealpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_recovery_by_tau.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(fig_dir / "fig_recovery_by_tau.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir / 'fig_recovery_by_tau.pdf'}")

    # Figure 3: Distribution histograms
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    for ax, bucket, color in zip(axes, ["early", "late"], ["#e74c3c", "#2ecc71"]):
        rates = bucket_rates[bucket]
        ax.hist(rates, bins=np.linspace(0, 1, 17), color=color, edgecolor="black",
                linewidth=0.5, alpha=0.8)
        ax.axvline(np.mean(rates), color="black", linestyle="--", linewidth=1.2,
                   label=f"mean={np.mean(rates):.3f}")
        ax.set_xlabel("Per-sample Recovery Rate")
        bkt_label = "{2,3}" if bucket == "early" else "{4,5,6}"
        ax.set_title(f"{bucket} ($\\tau \\in$ {bkt_label})")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_recovery_distribution.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(fig_dir / "fig_recovery_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir / 'fig_recovery_distribution.pdf'}")

    # Save summary JSON
    out_summary_path = Path(args.out_summary) if args.out_summary else (
        input_path.parent / "recovery_summary.json")
    full_summary = {
        "by_bucket": summary,
        "by_tau": {str(k): v for k, v in tau_summary.items()},
        "total_samples": len(sample_stats),
        "total_continuations": len(results),
    }
    out_summary_path.write_text(
        json.dumps(full_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
