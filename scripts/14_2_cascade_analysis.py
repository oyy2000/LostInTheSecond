#!/usr/bin/env python3
"""
Phase 14_2: Cascade error recovery analysis and figures.

Compares three conditions:
  - fix_first:             fix root-cause A, resample from A+1 onward
  - fix_later:             fix only cascade B, resample from B+1 onward
  - fix_first_keep_middle: fix A, keep original steps A+1..B, resample from B+1

Produces:
  - Three-way grouped bar chart
  - Pairwise scatter plots
  - Recovery by step gap (B - A)
  - Delta distribution (violin)
  - Summary JSON with all pairwise statistical tests

Usage:
    python scripts/14_2_cascade_analysis.py
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent

COND_LABELS = {
    "fix_first": "Fix A, resample",
    "fix_later": "Fix B only",
    "fix_first_keep_middle": "Fix A, keep middle",
}
COND_COLORS = {
    "fix_first": "#3498db",
    "fix_later": "#e74c3c",
    "fix_first_keep_middle": "#2ecc71",
}
COND_ORDER = ["fix_first", "fix_first_keep_middle", "fix_later"]


def parse_args():
    ap = argparse.ArgumentParser(description="Cascade error recovery analysis")
    ap.add_argument("--cont-file", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/cascade_recovery/continuations.jsonl"))
    ap.add_argument("--fig-dir", default=str(
        PROJECT_ROOT / "figures/gsm8k_cascade_recovery"))
    ap.add_argument("--out-summary", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/cascade_recovery/summary.json"))
    return ap.parse_args()


def load_continuations(path: Path):
    return [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]


def compute_pair_rates(rows):
    groups = defaultdict(list)
    for r in rows:
        key = (r["doc_id"], r["sample_idx"], r["pair_idx"], r["condition"])
        groups[key].append(r)
    rates = {}
    for key, recs in groups.items():
        n_correct = sum(1 for r in recs if r["exact_match"] >= 1.0)
        rates[key] = {
            "rate": n_correct / len(recs),
            "n_total": len(recs),
            "first_err_step": recs[0]["first_err_step"],
            "cascade_step": recs[0]["cascade_step"],
            "n_steps": recs[0]["n_steps"],
        }
    return rates


def _pairwise_test(a: np.ndarray, b: np.ndarray) -> Dict:
    delta = a - b
    n = len(delta)
    mean_d = float(np.mean(delta))
    se_d = float(np.std(delta, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    t_val, p_t = sp_stats.ttest_1samp(delta, 0) if n > 1 else (0.0, 1.0)
    try:
        w_val, p_w = sp_stats.wilcoxon(delta)
    except ValueError:
        w_val, p_w = float("nan"), float("nan")
    d_cohen = mean_d / np.std(delta, ddof=1) if np.std(delta, ddof=1) > 0 else 0.0
    return {
        "n": n, "delta_mean": mean_d, "delta_se": se_d,
        "ttest_t": float(t_val), "ttest_p": float(p_t),
        "wilcoxon_W": float(w_val), "wilcoxon_p": float(p_w),
        "cohens_d": float(d_cohen),
    }


def main():
    args = parse_args()
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cont_path = Path(args.cont_file)
    if not cont_path.exists():
        print(f"ERROR: {cont_path} not found"); sys.exit(1)

    rows = load_continuations(cont_path)
    cond_counts = defaultdict(int)
    for r in rows:
        cond_counts[r["condition"]] += 1
    print(f"Loaded {len(rows)} records")
    for c in COND_ORDER:
        print(f"  {c}: {cond_counts.get(c, 0)}")

    available_conds = [c for c in COND_ORDER if cond_counts.get(c, 0) > 0]
    rates = compute_pair_rates(rows)

    # Build paired data across all available conditions
    pair_keys = set()
    for (doc_id, sample_idx, pair_idx, cond), info in rates.items():
        pair_keys.add((doc_id, sample_idx, pair_idx))

    paired = []
    for pk in sorted(pair_keys):
        doc_id, sample_idx, pair_idx = pk
        rec = {"doc_id": doc_id, "sample_idx": sample_idx, "pair_idx": pair_idx}
        all_present = True
        for c in available_conds:
            key = (doc_id, sample_idx, pair_idx, c)
            if key not in rates:
                all_present = False
                break
            info = rates[key]
            rec[f"r_{c}"] = info["rate"]
            rec["first_err_step"] = info["first_err_step"]
            rec["cascade_step"] = info["cascade_step"]
            rec["n_steps"] = info["n_steps"]
        if all_present:
            rec["step_gap"] = rec["cascade_step"] - rec["first_err_step"]
            paired.append(rec)

    n_paired = len(paired)
    print(f"Paired comparisons (all conditions present): {n_paired}")
    if n_paired == 0:
        print("No paired data. Exiting."); sys.exit(1)

    # Per-condition arrays
    cond_arrays = {}
    for c in available_conds:
        cond_arrays[c] = np.array([p[f"r_{c}"] for p in paired])
    step_gaps = np.array([p["step_gap"] for p in paired])

    # Overall means and SEs
    cond_stats = {}
    for c in available_conds:
        arr = cond_arrays[c]
        cond_stats[c] = {
            "mean": float(np.mean(arr)),
            "se": float(np.std(arr, ddof=1) / np.sqrt(n_paired)),
        }

    print(f"\n=== Overall Results (n={n_paired}) ===")
    for c in available_conds:
        s = cond_stats[c]
        print(f"  R({COND_LABELS[c]}): {s['mean']:.4f} +/- {s['se']:.4f}")

    # Pairwise tests
    pairwise = {}
    pairs_to_test = [
        ("fix_first", "fix_later"),
        ("fix_first", "fix_first_keep_middle"),
        ("fix_first_keep_middle", "fix_later"),
    ]
    for ca, cb in pairs_to_test:
        if ca in cond_arrays and cb in cond_arrays:
            result = _pairwise_test(cond_arrays[ca], cond_arrays[cb])
            pairwise[f"{ca}_vs_{cb}"] = result
            print(f"\n  {COND_LABELS[ca]} vs {COND_LABELS[cb]}:")
            print(f"    delta = {result['delta_mean']:.4f} +/- {result['delta_se']:.4f}")
            print(f"    t={result['ttest_t']:.3f}, p={result['ttest_p']:.2e}, d={result['cohens_d']:.3f}")

    # --- Figure 1: Three-way grouped bar chart ---
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
    x = np.arange(1)
    n_conds = len(available_conds)
    width = 0.22
    offsets = np.linspace(-(n_conds - 1) * width / 2, (n_conds - 1) * width / 2, n_conds)
    for i, c in enumerate(available_conds):
        s = cond_stats[c]
        ax.bar(x + offsets[i], [s["mean"]], width,
               yerr=[1.96 * s["se"]], capsize=4,
               label=COND_LABELS[c], color=COND_COLORS[c], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(["Cascade Error Pairs"])
    ax.set_ylabel("Recovery Rate")
    ax.set_title("Cascade Error: Three-Way Comparison")
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    max_val = max(s["mean"] + 1.96 * s["se"] for s in cond_stats.values())
    ax.set_ylim(0, max_val * 1.3)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_cascade_bar.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(fig_dir / "fig_cascade_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {fig_dir / 'fig_cascade_bar.png'}")

    # --- Figure 2: Pairwise scatter (fix_first vs fix_later, fix_first vs fkm) ---
    scatter_pairs = [
        ("fix_later", "fix_first", "R(Fix B only)", "R(Fix A, resample)"),
        ("fix_later", "fix_first_keep_middle", "R(Fix B only)", "R(Fix A, keep middle)"),
        ("fix_first_keep_middle", "fix_first", "R(Fix A, keep middle)", "R(Fix A, resample)"),
    ]
    scatter_pairs = [(a, b, la, lb) for a, b, la, lb in scatter_pairs
                     if a in cond_arrays and b in cond_arrays]
    if scatter_pairs:
        n_plots = len(scatter_pairs)
        fig, axes = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 4.5))
        if n_plots == 1:
            axes = [axes]
        for ax, (cx, cy, lx, ly) in zip(axes, scatter_pairs):
            ax.scatter(cond_arrays[cx], cond_arrays[cy],
                       alpha=0.2, s=15, color="#2c3e50", edgecolors="none")
            lim = [0, 1.05]
            ax.plot(lim, lim, "--", color="gray", linewidth=1)
            ax.set_xlabel(lx)
            ax.set_ylabel(ly)
            n_above = np.sum(cond_arrays[cy] > cond_arrays[cx])
            n_below = np.sum(cond_arrays[cy] < cond_arrays[cx])
            n_eq = np.sum(cond_arrays[cy] == cond_arrays[cx])
            ax.text(0.05, 0.95, f"Y>X: {n_above}\nY<X: {n_below}\nY=X: {n_eq}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_aspect("equal")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fig.savefig(fig_dir / "fig_cascade_scatter.pdf", dpi=150, bbox_inches="tight")
        fig.savefig(fig_dir / "fig_cascade_scatter.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fig_dir / 'fig_cascade_scatter.png'}")

    # --- Figure 3: Recovery by step gap, all conditions ---
    unique_gaps = sorted(set(step_gaps))
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    n_conds_gap = len(available_conds)
    width_gap = 0.8 / n_conds_gap
    for ci, c in enumerate(available_conds):
        means_g, ses_g = [], []
        for g in unique_gaps:
            mask = step_gaps == g
            vals = cond_arrays[c][mask]
            means_g.append(np.mean(vals))
            ses_g.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        x_pos = np.arange(len(unique_gaps)) + ci * width_gap
        ax.bar(x_pos, means_g, width_gap * 0.9,
               yerr=[1.96 * s for s in ses_g], capsize=3,
               label=COND_LABELS[c], color=COND_COLORS[c], alpha=0.85)
    ax.set_xticks(np.arange(len(unique_gaps)) + (n_conds_gap - 1) * width_gap / 2)
    ax.set_xticklabels([str(g) for g in unique_gaps], fontsize=8)
    ax.set_xlabel("Step Gap (B - A)")
    ax.set_ylabel("Recovery Rate")
    ax.set_title("Recovery by Error Distance")
    ax.legend(fontsize=7, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_cascade_by_gap.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(fig_dir / "fig_cascade_by_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir / 'fig_cascade_by_gap.png'}")

    # --- Figure 4: Delta distribution (violin + strip) for each pair ---
    delta_pairs = []
    delta_labels = []
    for ca, cb in pairs_to_test:
        if ca in cond_arrays and cb in cond_arrays:
            delta_pairs.append(cond_arrays[ca] - cond_arrays[cb])
            delta_labels.append(f"{COND_LABELS[ca]}\n- {COND_LABELS[cb]}")

    if delta_pairs:
        fig, ax = plt.subplots(1, 1, figsize=(3 + 2 * len(delta_pairs), 4.5))
        positions = list(range(len(delta_pairs)))
        vp = ax.violinplot(delta_pairs, positions=positions,
                           showmedians=True, showextrema=False)
        colors_vp = ["#3498db", "#2ecc71", "#9b59b6"]
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(colors_vp[i % len(colors_vp)])
            body.set_alpha(0.3)
        vp["cmedians"].set_color("black")
        rng = np.random.default_rng(42)
        for i, d in enumerate(delta_pairs):
            jitter = rng.uniform(-0.12, 0.12, size=len(d))
            ax.scatter(i + jitter, d, alpha=0.15, s=8, color="#2c3e50")
            m = np.mean(d)
            ax.plot(i, m, "D", color="#e74c3c", markersize=8, zorder=5)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels(delta_labels, fontsize=8)
        ax.set_ylabel("$\\Delta$ per cascade pair")
        ax.set_title("Pairwise Recovery Differences")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fig.savefig(fig_dir / "fig_cascade_delta_dist.pdf", dpi=150, bbox_inches="tight")
        fig.savefig(fig_dir / "fig_cascade_delta_dist.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fig_dir / 'fig_cascade_delta_dist.png'}")

    # --- Save summary JSON ---
    summary = {
        "n_paired": n_paired,
        "conditions": {},
        "pairwise_tests": pairwise,
    }
    for c in available_conds:
        summary["conditions"][c] = {
            "label": COND_LABELS[c],
            "mean": cond_stats[c]["mean"],
            "se": cond_stats[c]["se"],
        }

    # Step gap breakdown
    gap_breakdown = {}
    for g in unique_gaps:
        mask = step_gaps == g
        entry = {"n": int(np.sum(mask))}
        for c in available_conds:
            vals = cond_arrays[c][mask]
            entry[f"r_{c}"] = float(np.mean(vals))
        gap_breakdown[str(int(g))] = entry
    summary["by_step_gap"] = gap_breakdown

    out_path = Path(args.out_summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved summary: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
