#!/usr/bin/env python3
"""
Combined analysis: Bad-prefix recovery vs Minimal repair.

Computes and visualizes:
  - R_bad(early), R_bad(late)
  - R_fix(early), R_fix(late)
  - Delta_fix(early), Delta_fix(late)
  - Breakdown by individual tau values

Requires outputs from 6_1 and 6_3.

Usage:
    python scripts/6_4_compare_bad_vs_repair.py
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
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare bad-prefix vs minimal repair")
    ap.add_argument("--bad-file", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/bad_prefix_recovery/continuations.jsonl"))
    ap.add_argument("--fix-file", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/minimal_repair/continuations.jsonl"))
    ap.add_argument("--fig-dir", default=str(
        PROJECT_ROOT / "figures/bad_prefix_recovery"))
    ap.add_argument("--out-summary", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/repair_gain_summary.json"))
    ap.add_argument("--per-question", action="store_true",
                    help="Aggregate by doc_id (question) instead of (doc_id, sample_idx)")
    return ap.parse_args()


def load_per_sample_rates(path: Path) -> dict:
    """Load continuations and compute per-(doc_id, sample_idx) recovery rate."""
    rows = [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]
    by_sample = defaultdict(list)
    for r in rows:
        key = (r["doc_id"], r["sample_idx"])
        by_sample[key].append(r)

    sample_rates = {}
    for key, recs in by_sample.items():
        n_correct = sum(1 for r in recs if r["exact_match"] >= 1.0)
        sample_rates[key] = {
            "rate": n_correct / len(recs),
            "bucket": recs[0]["bucket"],
            "tau": recs[0]["tau"],
            "n_steps": recs[0]["n_steps"],
            "n_total": len(recs),
        }
    return sample_rates


def load_per_question_rates(path: Path) -> dict:
    """Load continuations and compute per-doc_id (question-level) recovery rate.

    All continuations across all sample_idx for the same doc_id are pooled.
    Bucket and tau are taken as the mode across samples for that question.
    """
    rows = [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]
    by_doc = defaultdict(list)
    for r in rows:
        by_doc[r["doc_id"]].append(r)

    question_rates = {}
    for doc_id, recs in by_doc.items():
        n_correct = sum(1 for r in recs if r["exact_match"] >= 1.0)
        buckets = [r["bucket"] for r in recs]
        taus = [r["tau"] for r in recs]
        n_steps_list = [r["n_steps"] for r in recs]
        question_rates[doc_id] = {
            "rate": n_correct / len(recs),
            "bucket": max(set(buckets), key=buckets.count),
            "tau": int(np.median(taus)),
            "n_steps": int(np.median(n_steps_list)),
            "n_total": len(recs),
        }
    return question_rates


def main():
    args = parse_args()
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    bad_path = Path(args.bad_file)
    fix_path = Path(args.fix_file)

    for p, name in [(bad_path, "bad-prefix"), (fix_path, "minimal-repair")]:
        if not p.exists():
            print(f"ERROR: {name} file not found: {p}")
            sys.exit(1)

    bad_rates = load_per_sample_rates(bad_path) if not args.per_question else load_per_question_rates(bad_path)
    fix_rates = load_per_sample_rates(fix_path) if not args.per_question else load_per_question_rates(fix_path)
    granularity = "per-question" if args.per_question else "per-sample"
    fig_suffix = "_perq" if args.per_question else ""
    print(f"Granularity: {granularity}")
    print(f"Bad-prefix items: {len(bad_rates)}")
    print(f"Fix items: {len(fix_rates)}")

    # Match samples present in both
    common_keys = set(bad_rates.keys()) & set(fix_rates.keys())
    print(f"Common samples: {len(common_keys)}")

    # Build paired data
    paired = []
    for key in common_keys:
        if args.per_question:
            doc_id, sample_idx = key, None
        else:
            doc_id, sample_idx = key[0], key[1]
        paired.append({
            "doc_id": doc_id,
            "sample_idx": sample_idx,
            "bucket": bad_rates[key]["bucket"],
            "tau": bad_rates[key]["tau"],
            "n_steps": bad_rates[key]["n_steps"],
            "r_bad": bad_rates[key]["rate"],
            "r_fix": fix_rates[key]["rate"],
            "delta": fix_rates[key]["rate"] - bad_rates[key]["rate"],
        })

    # Aggregate by bucket
    by_bucket = defaultdict(list)
    for p in paired:
        by_bucket[p["bucket"]].append(p)

    print("\n=== Results by bucket ===")
    summary = {}
    for bucket in ["early", "late"]:
        items = by_bucket[bucket]
        if not items:
            continue
        r_bad = np.array([x["r_bad"] for x in items])
        r_fix = np.array([x["r_fix"] for x in items])
        delta = np.array([x["delta"] for x in items])
        n = len(items)

        summary[bucket] = {
            "n": n,
            "r_bad_mean": float(np.mean(r_bad)),
            "r_bad_se": float(np.std(r_bad) / np.sqrt(n)),
            "r_fix_mean": float(np.mean(r_fix)),
            "r_fix_se": float(np.std(r_fix) / np.sqrt(n)),
            "delta_mean": float(np.mean(delta)),
            "delta_se": float(np.std(delta) / np.sqrt(n)),
        }
        # Paired t-test for delta > 0
        t_stat, p_val = sp_stats.ttest_1samp(delta, 0)
        summary[bucket]["delta_ttest_t"] = float(t_stat)
        summary[bucket]["delta_ttest_p"] = float(p_val)

        print(f"  {bucket} (n={n}):")
        print(f"    R_bad  = {np.mean(r_bad):.4f} +/- {np.std(r_bad)/np.sqrt(n):.4f}")
        print(f"    R_fix  = {np.mean(r_fix):.4f} +/- {np.std(r_fix)/np.sqrt(n):.4f}")
        print(f"    Delta  = {np.mean(delta):.4f} +/- {np.std(delta)/np.sqrt(n):.4f} "
              f"(t={t_stat:.2f}, p={p_val:.2e})")

    # --- Between-bucket test: H0: E[Delta|early] = E[Delta|late] ---
    print("\n=== Between-bucket test ===")
    delta_early = np.array([x["delta"] for x in by_bucket["early"]])
    delta_late = np.array([x["delta"] for x in by_bucket["late"]])

    # Welch's t-test (unequal variance)
    t_between, p_between = sp_stats.ttest_ind(delta_early, delta_late, equal_var=False)
    # Mann-Whitney U (non-parametric)
    u_stat, p_mann = sp_stats.mannwhitneyu(delta_early, delta_late, alternative="two-sided")
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(delta_early, ddof=1) + np.var(delta_late, ddof=1)) / 2)
    cohens_d = (np.mean(delta_early) - np.mean(delta_late)) / pooled_std if pooled_std > 0 else 0.0

    print(f"  Delta_early: mean={np.mean(delta_early):.4f}, std={np.std(delta_early, ddof=1):.4f}, n={len(delta_early)}")
    print(f"  Delta_late:  mean={np.mean(delta_late):.4f}, std={np.std(delta_late, ddof=1):.4f}, n={len(delta_late)}")
    print(f"  Welch t-test: t={t_between:.3f}, p={p_between:.2e}")
    print(f"  Mann-Whitney U: U={u_stat:.0f}, p={p_mann:.2e}")
    print(f"  Cohen's d: {cohens_d:.3f}")

    between_test = {
        "delta_early_mean": float(np.mean(delta_early)),
        "delta_early_std": float(np.std(delta_early, ddof=1)),
        "delta_late_mean": float(np.mean(delta_late)),
        "delta_late_std": float(np.std(delta_late, ddof=1)),
        "welch_t": float(t_between),
        "welch_p": float(p_between),
        "mann_whitney_U": float(u_stat),
        "mann_whitney_p": float(p_mann),
        "cohens_d": float(cohens_d),
        "n_early": len(delta_early),
        "n_late": len(delta_late),
    }

    # Aggregate by tau
    by_tau = defaultdict(list)
    for p in paired:
        by_tau[p["tau"]].append(p)

    print("\n=== Results by tau ===")
    tau_summary = {}
    for tau in sorted(by_tau.keys()):
        items = by_tau[tau]
        r_bad = np.array([x["r_bad"] for x in items])
        r_fix = np.array([x["r_fix"] for x in items])
        delta = np.array([x["delta"] for x in items])
        n = len(items)
        tau_summary[tau] = {
            "n": n,
            "r_bad_mean": float(np.mean(r_bad)),
            "r_fix_mean": float(np.mean(r_fix)),
            "delta_mean": float(np.mean(delta)),
            "delta_se": float(np.std(delta) / np.sqrt(n)),
        }
        print(f"  tau={tau} (n={n}): R_bad={np.mean(r_bad):.4f}, "
              f"R_fix={np.mean(r_fix):.4f}, Delta={np.mean(delta):.4f}")

    # --- Figure 1: Grouped bar chart R_bad vs R_fix ---
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    x = np.arange(2)
    width = 0.3
    buckets_list = ["early", "late"]
    r_bad_means = [summary[b]["r_bad_mean"] for b in buckets_list]
    r_bad_ses = [summary[b]["r_bad_se"] for b in buckets_list]
    r_fix_means = [summary[b]["r_fix_mean"] for b in buckets_list]
    r_fix_ses = [summary[b]["r_fix_se"] for b in buckets_list]

    bars1 = ax.bar(x - width/2, r_bad_means, width, yerr=[1.96*s for s in r_bad_ses],
                   capsize=4, label="$R_{bad}$ (no repair)", color="#e74c3c", alpha=0.8)
    bars2 = ax.bar(x + width/2, r_fix_means, width, yerr=[1.96*s for s in r_fix_ses],
                   capsize=4, label="$R_{fix}$ (repaired)", color="#3498db", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["early ($\\tau$=2,3)", "late ($\\tau$=4,5,6)"])
    ax.set_ylabel("Recovery Rate")
    ax.set_title(f"Bad Prefix vs. Minimal Repair ({granularity})")
    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymax = max(r_bad_means + r_fix_means) * 1.4
    ax.set_ylim(0, ymax)
    plt.tight_layout()
    fig.savefig(fig_dir / f"fig_bad_vs_repair{fig_suffix}.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(fig_dir / f"fig_bad_vs_repair{fig_suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {fig_dir / f'fig_bad_vs_repair{fig_suffix}.pdf'}")

    # --- Figure 2: Delta_fix by bucket ---
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    delta_means = [summary[b]["delta_mean"] for b in buckets_list]
    delta_ses = [summary[b]["delta_se"] for b in buckets_list]
    colors = ["#e74c3c", "#2ecc71"]
    bars = ax.bar(buckets_list, delta_means, yerr=[1.96*s for s in delta_ses],
                  capsize=5, color=colors, edgecolor="black", linewidth=0.8, width=0.5)
    ax.set_ylabel("Repair Gain $\\Delta_{fix}$")
    ax.set_xlabel("Error Position")
    ax.set_title("Repair Gain: $R_{fix} - R_{bad}$")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    for bar, m in zip(bars, delta_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{m:.3f}", ha="center", va="bottom", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(fig_dir / f"fig_repair_gain{fig_suffix}.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(fig_dir / f"fig_repair_gain{fig_suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir / f'fig_repair_gain{fig_suffix}.pdf'}")

    # --- Figure 3: Between-bucket Delta distribution (violin + strip) ---
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    vp = ax.violinplot([delta_early, delta_late], positions=[0, 1], showmedians=True,
                       showextrema=False)
    for i, (body, color) in enumerate(zip(vp["bodies"], ["#e74c3c", "#2ecc71"])):
        body.set_facecolor(color)
        body.set_alpha(0.3)
    vp["cmedians"].set_color("black")
    rng = np.random.default_rng(42)
    for i, (data, color) in enumerate(zip([delta_early, delta_late], ["#e74c3c", "#2ecc71"])):
        jitter = rng.uniform(-0.08, 0.08, size=len(data))
        ax.scatter(np.full(len(data), i) + jitter, data, alpha=0.15, s=8, color=color)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["early ($\\tau$=2,3)", "late ($\\tau$=4,5,6)"])
    ax.set_ylabel("Per-item Repair Gain $\\Delta_i = \\hat{p}^{fix}_i - \\hat{p}^{bad}_i$")
    ax.set_title(f"Between-Bucket Test: Welch t={t_between:.2f}, p={p_between:.2e}")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    for i, (data, label) in enumerate(zip([delta_early, delta_late], ["early", "late"])):
        ax.plot(i, np.mean(data), "D", color="black", markersize=8, zorder=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(fig_dir / f"fig_between_bucket_delta{fig_suffix}.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(fig_dir / f"fig_between_bucket_delta{fig_suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir / f'fig_between_bucket_delta{fig_suffix}.pdf'}")

    # --- Figure 4: R_bad and R_fix by tau (line plot) ---
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    taus = sorted(tau_summary.keys())
    r_bad_tau = [tau_summary[t]["r_bad_mean"] for t in taus]
    r_fix_tau = [tau_summary[t]["r_fix_mean"] for t in taus]
    delta_tau = [tau_summary[t]["delta_mean"] for t in taus]
    delta_se_tau = [tau_summary[t]["delta_se"] for t in taus]

    ax.plot(taus, r_bad_tau, "o-", color="#e74c3c", linewidth=2, markersize=7,
            label="$R_{bad}$ (no repair)")
    ax.plot(taus, r_fix_tau, "s-", color="#3498db", linewidth=2, markersize=7,
            label="$R_{fix}$ (repaired)")
    ax.fill_between(taus, r_bad_tau, r_fix_tau, alpha=0.12, color="#3498db")
    for t, d in zip(taus, delta_tau):
        ax.annotate(f"$\\Delta$={d:.3f}", (t, (r_bad_tau[taus.index(t)] + r_fix_tau[taus.index(t)])/2),
                    textcoords="offset points", xytext=(15, 0), fontsize=8, color="#2c3e50")
    ax.set_xlabel("First Error Step ($\\tau$)")
    ax.set_ylabel("Recovery Rate")
    ax.set_title("Recovery with and without Minimal Repair")
    ax.set_xticks(taus)
    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(fig_dir / f"fig_recovery_curves{fig_suffix}.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(fig_dir / f"fig_recovery_curves{fig_suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir / f'fig_recovery_curves{fig_suffix}.pdf'}")

    # --- Figure 5: Relative position analysis ---
    rel_data = []
    for p in paired:
        rel_pos = p["tau"] / p["n_steps"]
        rel_data.append({"rel_pos": rel_pos, "delta": p["delta"],
                         "r_bad": p["r_bad"], "r_fix": p["r_fix"]})

    rel_positions = np.array([d["rel_pos"] for d in rel_data])
    deltas_all = np.array([d["delta"] for d in rel_data])
    r_bads_all = np.array([d["r_bad"] for d in rel_data])

    # Tercile split
    t33, t67 = np.percentile(rel_positions, [33.3, 66.7])
    tercile_labels = [
        ("early\n($\\tau/N < {:.2f}$)".format(t33), rel_positions < t33),
        ("mid\n($\\tau/N \\in [{:.2f},{:.2f})$)".format(t33, t67),
         (rel_positions >= t33) & (rel_positions < t67)),
        ("late\n($\\tau/N \\geq {:.2f}$)".format(t67), rel_positions >= t67),
    ]

    # Correlation
    from scipy.stats import pearsonr, spearmanr
    r_spear, p_spear = spearmanr(rel_positions, deltas_all)
    r_spear_bad, p_spear_bad = spearmanr(rel_positions, r_bads_all)

    print(f"\n=== Relative position analysis ===")
    print(f"  Spearman(rel_pos, Delta): rho={r_spear:.4f}, p={p_spear:.2e}")
    print(f"  Spearman(rel_pos, R_bad): rho={r_spear_bad:.4f}, p={p_spear_bad:.2e}")

    # Tercile bar chart for Delta
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Panel A: Delta by tercile
    ax = axes[0]
    tercile_means, tercile_ses, tercile_ns, tercile_xlabels = [], [], [], []
    for label, mask in tercile_labels:
        d = deltas_all[mask]
        tercile_means.append(np.mean(d))
        tercile_ses.append(np.std(d, ddof=1) / np.sqrt(len(d)))
        tercile_ns.append(len(d))
        tercile_xlabels.append(label)
    colors_t = ["#e74c3c", "#f39c12", "#2ecc71"]
    bars = ax.bar(range(3), tercile_means, yerr=[1.96*s for s in tercile_ses],
                  capsize=5, color=colors_t, edgecolor="black", linewidth=0.8, width=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(tercile_xlabels, fontsize=9)
    ax.set_ylabel("Repair Gain $\\Delta_{fix}$")
    ax.set_title("Repair Gain by Relative Error Position")
    for i, (bar, m, n) in enumerate(zip(bars, tercile_means, tercile_ns)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{m:.3f}\nn={n}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: Scatter + trend
    ax = axes[1]
    ax.scatter(rel_positions, deltas_all, alpha=0.12, s=12, color="#2c3e50")
    # LOESS-like: binned means
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_means, bin_ses_plot = [], [], []
    for i in range(n_bins):
        mask = (rel_positions >= bin_edges[i]) & (rel_positions < bin_edges[i+1])
        if mask.sum() >= 5:
            bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
            bin_means.append(np.mean(deltas_all[mask]))
            bin_ses_plot.append(1.96 * np.std(deltas_all[mask]) / np.sqrt(mask.sum()))
    ax.errorbar(bin_centers, bin_means, yerr=bin_ses_plot, color="#e74c3c",
                linewidth=2, marker="o", markersize=5, capsize=3, label="binned mean")
    ax.set_xlabel("Relative Error Position ($\\tau / N_{steps}$)")
    ax.set_ylabel("$\\Delta_i$")
    ax.set_title(f"$\\rho_s$={r_spear:.3f}, p={p_spear:.1e}")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(fig_dir / f"fig_relative_position{fig_suffix}.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(fig_dir / f"fig_relative_position{fig_suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir / f'fig_relative_position{fig_suffix}.pdf'}")

    # Median-split Welch test on relative position
    median_rel = np.median(rel_positions)
    early_rel_delta = deltas_all[rel_positions <= median_rel]
    late_rel_delta = deltas_all[rel_positions > median_rel]
    t_rel, p_rel = sp_stats.ttest_ind(early_rel_delta, late_rel_delta, equal_var=False)
    print(f"  Median-split Welch t={t_rel:.3f}, p={p_rel:.2e}")
    print(f"  early_rel Delta={np.mean(early_rel_delta):.4f} (n={len(early_rel_delta)})")
    print(f"  late_rel  Delta={np.mean(late_rel_delta):.4f} (n={len(late_rel_delta)})")

    relative_pos_test = {
        "spearman_rho_delta": float(r_spear),
        "spearman_p_delta": float(p_spear),
        "spearman_rho_r_bad": float(r_spear_bad),
        "spearman_p_r_bad": float(p_spear_bad),
        "median_split_welch_t": float(t_rel),
        "median_split_welch_p": float(p_rel),
        "terciles": {
            "boundaries": [float(t33), float(t67)],
            "means": [float(m) for m in tercile_means],
            "ns": tercile_ns,
        },
    }

    # Save summary
    out_path = Path(args.out_summary)
    if args.per_question:
        out_path = out_path.with_name(out_path.stem + "_perq" + out_path.suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_summary = {
        "by_bucket": summary,
        "by_tau": {str(k): v for k, v in tau_summary.items()},
        "between_bucket_test": between_test,
        "relative_position_test": relative_pos_test,
        "n_paired_samples": len(paired),
    }
    out_path.write_text(json.dumps(full_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
