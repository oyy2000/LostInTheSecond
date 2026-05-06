#!/usr/bin/env python3
"""
Analyze where the first hard error occurs in wrong CoT trajectories.

Verifies the hypothesis: "errors in the final answer are typically caused
by a single erroneous step" using GPT-judged first-error and all-error caches.

Produces:
  1. Bar chart of first-error relative position distribution
  2. Bar chart of absolute first-error step
  3. Pie/bar chart of independent error count per trajectory
  4. Summary statistics

Usage:
    python scripts/20_0_first_error_step_analysis.py
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures" / "first_error_analysis"

COLORS = {
    "GSM8K": "#4C72B0",
    "MATH500": "#DD8452",
}


def load_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def filter_valid(records, min_steps=4, exclude_last_step=True):
    """Keep records with valid tau > 0, sufficient steps,
    and optionally exclude cases where first error is the last step."""
    out = []
    for r in records:
        if r.get("tau") is None or r["tau"] <= 0:
            continue
        if r.get("n_steps", 0) < min_steps:
            continue
        if exclude_last_step and r["tau"] >= r["n_steps"]:
            continue
        out.append(r)
    return out


def compute_relative_positions(records):
    return [r["tau"] / r["n_steps"] for r in records]


def compute_stats(records):
    rel_pos = compute_relative_positions(records)
    abs_taus = [r["tau"] for r in records]
    n_steps_list = [r["n_steps"] for r in records]
    return {
        "n_wrong": len(records),
        "mean_tau": float(np.mean(abs_taus)),
        "median_tau": float(np.median(abs_taus)),
        "mean_rel_pos": float(np.mean(rel_pos)),
        "median_rel_pos": float(np.median(rel_pos)),
        "mean_n_steps": float(np.mean(n_steps_list)),
        "frac_first_half": float(np.mean([rp <= 0.5 for rp in rel_pos])),
        "frac_first_third": float(np.mean([rp <= 1/3 for rp in rel_pos])),
    }


# ── Figure 1: Relative position histogram ────────────────────────────────

def plot_relpos_histogram(data_dict, out_path, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_width = 0.8 / len(data_dict)

    fig, ax = plt.subplots(figsize=(7, 4))

    for idx, (label, records) in enumerate(data_dict.items()):
        rel_pos = compute_relative_positions(records)
        counts, _ = np.histogram(rel_pos, bins=bin_edges)
        fracs = counts / counts.sum()
        offset = (idx - (len(data_dict) - 1) / 2) * bar_width
        ds_key = "GSM8K" if "GSM8K" in label else "MATH500"
        ax.bar(
            bin_centers + offset, fracs,
            width=bar_width * 0.88,
            label=label, color=COLORS.get(ds_key, f"C{idx}"),
            alpha=0.85, edgecolor="white", linewidth=0.5,
        )

    ax.set_xlabel("Relative position of first error  ($\\tau / T$)", fontsize=11)
    ax.set_ylabel("Fraction of wrong trajectories", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks(bin_centers)
    ax.set_xticklabels([f"{c:.1f}" for c in bin_centers], fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 2: Absolute step bar chart ────────────────────────────────────

def plot_absolute_step(data_dict, out_path, max_step=12):
    steps = list(range(1, max_step + 1))
    bar_width = 0.8 / len(data_dict)

    fig, ax = plt.subplots(figsize=(7, 4))

    for idx, (label, records) in enumerate(data_dict.items()):
        tau_counts = Counter(r["tau"] for r in records)
        total = len(records)
        fracs = [tau_counts.get(s, 0) / total for s in steps]
        offset = (idx - (len(data_dict) - 1) / 2) * bar_width
        ds_key = "GSM8K" if "GSM8K" in label else "MATH500"
        ax.bar(
            np.array(steps) + offset, fracs,
            width=bar_width * 0.88,
            label=label, color=COLORS.get(ds_key, f"C{idx}"),
            alpha=0.85, edgecolor="white", linewidth=0.5,
        )

    ax.set_xlabel("First error step ($\\tau$, absolute)", fontsize=11)
    ax.set_ylabel("Fraction of wrong trajectories", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_xticks(steps)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 3: Independent error count ────────────────────────────────────

def plot_independent_error_count(cascade_data, out_path):
    """Bar chart: how many independent errors per wrong trajectory."""
    fig, ax = plt.subplots(figsize=(5, 4))

    x_labels = ["1", "2", "3", "4+"]
    bar_width = 0.8 / len(cascade_data)

    for idx, (label, indep_counts) in enumerate(cascade_data.items()):
        total = len(indep_counts)
        bins = [
            sum(1 for n in indep_counts if n == 1),
            sum(1 for n in indep_counts if n == 2),
            sum(1 for n in indep_counts if n == 3),
            sum(1 for n in indep_counts if n >= 4),
        ]
        fracs = [b / total for b in bins]
        offset = (idx - (len(cascade_data) - 1) / 2) * bar_width
        ds_key = "GSM8K" if "GSM8K" in label else "MATH500"
        bars = ax.bar(
            np.arange(len(x_labels)) + offset, fracs,
            width=bar_width * 0.88,
            label=label, color=COLORS.get(ds_key, f"C{idx}"),
            alpha=0.85, edgecolor="white", linewidth=0.5,
        )
        for bar, frac in zip(bars, fracs):
            if frac > 0.03:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{frac*100:.0f}%", ha="center", va="bottom", fontsize=8,
                )

    ax.set_xlabel("Number of independent errors per trajectory", fontsize=11)
    ax.set_ylabel("Fraction of wrong trajectories", fontsize=11)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-steps", type=int, default=4)
    ap.add_argument("--n-bins", type=int, default=10)
    args = ap.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load first-error caches ──────────────────────────────────────────
    first_error_configs = {
        "GSM8K": PROJECT_ROOT / "results/gsm8k_3b_multi_sample/first_error/gpt_first_error_cache.jsonl",
        "MATH500": PROJECT_ROOT / "results/math500_3b_multi_sample/first_error/gpt_first_error_cache.jsonl",
    }

    all_data = {}
    summary = {}
    for ds_name, path in first_error_configs.items():
        if not path.exists():
            print(f"[SKIP] {ds_name}: not found")
            continue
        raw = load_jsonl(path)
        valid = filter_valid(raw, min_steps=args.min_steps)
        label = f"Qwen 2.5 3B / {ds_name}"
        all_data[label] = valid
        stats = compute_stats(valid)
        summary[label] = stats

        print(f"\n{'='*55}")
        print(f"  {label}")
        print(f"  Wrong trajectories (valid): {stats['n_wrong']}")
        print(f"  Mean steps: {stats['mean_n_steps']:.1f}")
        print(f"  Mean first-error step: {stats['mean_tau']:.2f}")
        print(f"  Mean relative position: {stats['mean_rel_pos']:.3f}")
        print(f"  Median relative position: {stats['median_rel_pos']:.3f}")
        print(f"  Error in first half: {stats['frac_first_half']*100:.1f}%")

    # ── Load cascade (all-errors) cache ──────────────────────────────────
    cascade_configs = {
        "GSM8K": PROJECT_ROOT / "results/gsm8k_3b_multi_sample/cascade_errors/gpt_all_errors_cache.jsonl",
    }

    cascade_data = {}
    for ds_name, path in cascade_configs.items():
        if not path.exists():
            continue
        records = load_jsonl(path)
        indep_counts = []
        for r in records:
            gp = r.get("all_errors_parsed")
            if not gp or not gp.get("errors"):
                continue
            n_steps = gp.get("total_steps", r.get("n_steps", 0))
            indep_not_last = [
                e for e in gp["errors"]
                if e.get("type") == "independent" and e.get("step") != n_steps
            ]
            if indep_not_last:
                indep_counts.append(len(indep_not_last))

        label = f"Qwen 2.5 3B / {ds_name}"
        cascade_data[label] = indep_counts

        one = sum(1 for n in indep_counts if n == 1)
        two_or_less = sum(1 for n in indep_counts if n <= 2)
        total = len(indep_counts)
        summary[label]["single_independent_error_frac"] = one / total
        summary[label]["two_or_fewer_independent_frac"] = two_or_less / total

        print(f"\n  {label} -- cascade analysis")
        print(f"  Trajectories with errors: {total}")
        print(f"  Exactly 1 independent error: {one} ({one/total*100:.1f}%)")
        print(f"  <=2 independent errors: {two_or_less} ({two_or_less/total*100:.1f}%)")

    # ── Generate figures ─────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Generating figures...")

    plot_relpos_histogram(
        all_data,
        FIGURES_DIR / "fig_first_error_relpos.png",
        n_bins=args.n_bins,
    )
    plot_absolute_step(
        all_data,
        FIGURES_DIR / "fig_first_error_absolute.png",
    )

    if cascade_data:
        plot_independent_error_count(
            cascade_data,
            FIGURES_DIR / "fig_independent_error_count.png",
        )

    # ── Save summary ─────────────────────────────────────────────────────
    summary_path = FIGURES_DIR / "first_error_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
