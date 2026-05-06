#!/usr/bin/env python3
"""
Fine-grained relative-position analysis on CodeContests.

Splits samples into 5 quantile bins by rel_pos = tau / n_blocks,
computes per-bin repair gain (Delta_fix), and runs pairwise
Welch t-tests + Mann-Whitney U between all bin pairs.

Usage:
    python scripts/11_5_fine_grained_relpos.py
"""

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bad-file", default=str(
        PROJECT_ROOT / "results/codecontests_3b_multi_sample/bad_prefix_recovery/continuations.jsonl"))
    ap.add_argument("--fix-file", default=str(
        PROJECT_ROOT / "results/codecontests_3b_multi_sample/minimal_repair/continuations.jsonl"))
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--fig-dir", default=str(
        PROJECT_ROOT / "figures/codecontests_bad_prefix_recovery"))
    ap.add_argument("--out-summary", default=str(
        PROJECT_ROOT / "results/codecontests_3b_multi_sample/fine_relpos_summary.json"))
    return ap.parse_args()


def load_per_sample_rates(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text("utf-8").splitlines()
            if l.strip()]
    by_sample = defaultdict(list)
    for r in rows:
        key = (r["doc_id"], r["sample_idx"])
        by_sample[key].append(r)
    out = {}
    for key, recs in by_sample.items():
        n_correct = sum(1 for r in recs if r["exact_match"] >= 1.0)
        out[key] = {
            "rate": n_correct / len(recs),
            "bucket": recs[0]["bucket"],
            "tau": recs[0]["tau"],
            "n_blocks": recs[0]["n_blocks"],
        }
    return out


def main():
    args = parse_args()
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    n_bins = args.n_bins

    bad_rates = load_per_sample_rates(Path(args.bad_file))
    fix_rates = load_per_sample_rates(Path(args.fix_file))
    common = set(bad_rates.keys()) & set(fix_rates.keys())
    print(f"Common samples: {len(common)}")

    per_q = defaultdict(
        lambda: {"rel_pos": [], "r_bad": [], "r_fix": [], "delta": []})
    for key in common:
        doc_id = key[0]
        tau = bad_rates[key]["tau"]
        n_blocks = bad_rates[key]["n_blocks"]
        r_bad = bad_rates[key]["rate"]
        r_fix = fix_rates[key]["rate"]
        per_q[doc_id]["rel_pos"].append(tau / n_blocks)
        per_q[doc_id]["r_bad"].append(r_bad)
        per_q[doc_id]["r_fix"].append(r_fix)
        per_q[doc_id]["delta"].append(r_fix - r_bad)

    paired = []
    for doc_id, vals in per_q.items():
        paired.append({
            "rel_pos": float(np.mean(vals["rel_pos"])),
            "r_bad": float(np.mean(vals["r_bad"])),
            "r_fix": float(np.mean(vals["r_fix"])),
            "delta": float(np.mean(vals["delta"])),
        })
    print(f"Per-question data points: {len(paired)}")

    rel_positions = np.array([p["rel_pos"] for p in paired])
    deltas = np.array([p["delta"] for p in paired])
    r_bads = np.array([p["r_bad"] for p in paired])
    r_fixs = np.array([p["r_fix"] for p in paired])

    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(rel_positions, percentiles)
    edges[0] = rel_positions.min() - 1e-9
    edges[-1] = rel_positions.max() + 1e-9

    bin_data = []
    for i in range(n_bins):
        mask = ((rel_positions >= edges[i])
                & (rel_positions < edges[i + 1]))
        d = deltas[mask]
        rb = r_bads[mask]
        rf = r_fixs[mask]
        rp = rel_positions[mask]
        label = f"[{edges[i]:.2f}, {edges[i+1]:.2f})"
        short_label = f"Q{i+1}"
        bin_data.append({
            "idx": i,
            "label": label,
            "short_label": short_label,
            "range": (float(rp.min()), float(rp.max())),
            "n": int(mask.sum()),
            "delta_mean": float(np.mean(d)),
            "delta_se": float(np.std(d, ddof=1) / np.sqrt(len(d))),
            "delta_median": float(np.median(d)),
            "r_bad_mean": float(np.mean(rb)),
            "r_fix_mean": float(np.mean(rf)),
            "deltas": d,
        })

    print(f"\n{'Bin':<6} {'Range':<22} {'n':>5}  {'Delta_mean':>10}  "
          f"{'SE':>8}  {'R_bad':>8}  {'R_fix':>8}")
    print("-" * 80)
    for b in bin_data:
        print(f"{b['short_label']:<6} {b['label']:<22} {b['n']:>5}  "
              f"{b['delta_mean']:>10.4f}  {b['delta_se']:>8.4f}  "
              f"{b['r_bad_mean']:>8.4f}  {b['r_fix_mean']:>8.4f}")

    # Pairwise tests
    print("\n=== Pairwise Welch t-tests (Delta) ===")
    pairwise = {}
    for (i, bi), (j, bj) in combinations(enumerate(bin_data), 2):
        t_val, p_val = sp_stats.ttest_ind(
            bi["deltas"], bj["deltas"], equal_var=False)
        u_val, p_mw = sp_stats.mannwhitneyu(
            bi["deltas"], bj["deltas"], alternative="two-sided")
        key = f"{bi['short_label']}_vs_{bj['short_label']}"
        pairwise[key] = {
            "welch_t": float(t_val), "welch_p": float(p_val),
            "mann_whitney_U": float(u_val), "mann_whitney_p": float(p_mw),
        }
        sig = ("***" if p_val < 0.001 else "**" if p_val < 0.01
               else "*" if p_val < 0.05 else "ns")
        print(f"  {key:<12}  t={t_val:>7.3f}  p={p_val:.3e}  {sig}   "
              f"(MW p={p_mw:.3e})")

    kw_stat, kw_p = sp_stats.kruskal(*[b["deltas"] for b in bin_data])
    print(f"\nKruskal-Wallis H={kw_stat:.3f}, p={kw_p:.3e}")

    rho, p_rho = spearmanr(rel_positions, deltas)
    print(f"Spearman(rel_pos, Delta): rho={rho:.4f}, p={p_rho:.2e}")

    # Jonckheere-Terpstra trend test
    jt_stat = 0
    for i in range(n_bins):
        for j in range(i + 1, n_bins):
            for di in bin_data[i]["deltas"]:
                jt_stat += np.sum(bin_data[j]["deltas"] > di)
                jt_stat += 0.5 * np.sum(bin_data[j]["deltas"] == di)
    n_total = sum(b["n"] for b in bin_data)
    jt_mean = (n_total ** 2 - sum(b["n"] ** 2 for b in bin_data)) / 4
    jt_var_num = (n_total ** 2 * (2 * n_total + 3)
                  - sum(b["n"] ** 2 * (2 * b["n"] + 3) for b in bin_data))
    jt_var = jt_var_num / 72
    jt_z = ((jt_stat - jt_mean) / np.sqrt(jt_var)
            if jt_var > 0 else 0)
    jt_p = 2 * sp_stats.norm.sf(abs(jt_z))
    print(f"Jonckheere-Terpstra trend: z={jt_z:.3f}, p={jt_p:.3e}")

    # -- Figure: bar chart + scatter --
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    means = [b["delta_mean"] for b in bin_data]
    ses = [b["delta_se"] for b in bin_data]
    ns = [b["n"] for b in bin_data]
    x = np.arange(n_bins)
    cmap = plt.cm.RdYlGn
    colors = [cmap(i / (n_bins - 1)) for i in range(n_bins)]

    bars = ax.bar(x, means, yerr=[1.96 * s for s in ses],
                  capsize=4, color=colors, edgecolor="black",
                  linewidth=0.6, width=0.65)

    range_labels = []
    for b in bin_data:
        lo, hi = b["range"]
        range_labels.append(f"{b['short_label']}\n[{lo:.2f},{hi:.2f}]")
    ax.set_xticks(x)
    ax.set_xticklabels(range_labels, fontsize=8)

    for bar_idx, (bar, m, n) in enumerate(zip(bars, means, ns)):
        y_top = bar.get_height() + 1.96 * ses[bar_idx] + 0.005
        ax.text(bar.get_x() + bar.get_width() / 2, y_top,
                f"{m:.3f}\nn={n}", ha="center", va="bottom", fontsize=7.5)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Repair Gain $\\Delta_{fix}$")
    ax.set_xlabel(
        "Relative Error Position ($\\tau / N_{blocks}$) Quantile")
    ax.set_title(
        f"CodeContests: Repair Gain by Position (KW p={kw_p:.2e})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    key_15 = f"Q1_vs_Q{n_bins}"
    if key_15 in pairwise:
        p15 = pairwise[key_15]["welch_p"]
        sig_str = ("***" if p15 < 0.001 else "**" if p15 < 0.01
                   else "*" if p15 < 0.05 else "ns")
        y_bracket = max(m + 1.96 * s for m, s in zip(means, ses)) + 0.04
        ax.plot([0, 0, n_bins - 1, n_bins - 1],
                [y_bracket - 0.005, y_bracket, y_bracket,
                 y_bracket - 0.005],
                color="black", linewidth=1)
        ax.text((n_bins - 1) / 2, y_bracket + 0.003,
                f"p={p15:.2e} ({sig_str})", ha="center", fontsize=8)

    ax = axes[1]
    ax.scatter(rel_positions, deltas, alpha=0.1, s=10, color="#2c3e50",
               rasterized=True)

    bin_centers = [(b["range"][0] + b["range"][1]) / 2 for b in bin_data]
    bin_means_plot = [b["delta_mean"] for b in bin_data]
    bin_ses_plot = [1.96 * b["delta_se"] for b in bin_data]
    ax.errorbar(bin_centers, bin_means_plot, yerr=bin_ses_plot,
                color="#e74c3c", linewidth=2, marker="o", markersize=6,
                capsize=3, label=f"quantile mean (n={n_bins})", zorder=5)

    ax.set_xlabel("Relative Error Position ($\\tau / N_{blocks}$)")
    ax.set_ylabel("Repair Gain $\\Delta_i$")
    ax.set_title(
        f"CodeContests: Spearman $\\rho_s$={rho:.3f}, p={p_rho:.1e}")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(fig_dir / "fig_relative_position_fine.pdf",
                dpi=150, bbox_inches="tight")
    fig.savefig(fig_dir / "fig_relative_position_fine.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {fig_dir / 'fig_relative_position_fine.pdf'}")

    # -- Pairwise p-value heatmap --
    fig, ax = plt.subplots(figsize=(5, 4.5))
    labels = [b["short_label"] for b in bin_data]
    p_matrix = np.full((n_bins, n_bins), np.nan)
    for (i, bi), (j, bj) in combinations(enumerate(bin_data), 2):
        key = f"{bi['short_label']}_vs_{bj['short_label']}"
        p_val = pairwise[key]["welch_p"]
        p_matrix[i, j] = p_val
        p_matrix[j, i] = p_val

    log_p = -np.log10(np.where(
        np.isnan(p_matrix), 1.0, np.clip(p_matrix, 1e-20, 1.0)))
    log_p[np.isnan(p_matrix)] = 0

    im = ax.imshow(log_p, cmap="YlOrRd", aspect="equal",
                   vmin=0, vmax=max(4, np.nanmax(log_p)))
    ax.set_xticks(range(n_bins))
    ax.set_yticks(range(n_bins))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(n_bins):
        for j in range(n_bins):
            if i != j and not np.isnan(p_matrix[i, j]):
                p = p_matrix[i, j]
                sig = ("***" if p < 0.001 else "**" if p < 0.01
                       else "*" if p < 0.05 else "ns")
                txt = f"{p:.1e}\n{sig}"
                color = "white" if log_p[i, j] > 2 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("$-\\log_{10}(p)$")
    ax.set_title("CodeContests: Pairwise Welch t-test p-values")
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_relpos_pairwise_pvalues.pdf",
                dpi=150, bbox_inches="tight")
    fig.savefig(fig_dir / "fig_relpos_pairwise_pvalues.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir / 'fig_relpos_pairwise_pvalues.pdf'}")

    # Save summary
    out_path = Path(args.out_summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": "CodeContests",
        "n_bins": n_bins,
        "n_questions": len(paired),
        "aggregation": "per_question (doc_id), tau averaged across samples",
        "bins": [{k: v for k, v in b.items() if k != "deltas"}
                 for b in bin_data],
        "pairwise_tests": pairwise,
        "kruskal_wallis": {"H": float(kw_stat), "p": float(kw_p)},
        "jonckheere_terpstra": {"z": float(jt_z), "p": float(jt_p)},
        "spearman": {"rho": float(rho), "p": float(p_rho)},
    }
    out_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
