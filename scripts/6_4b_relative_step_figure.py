#!/usr/bin/env python3
"""
Relative-step version of fig_bad_vs_repair.

Buckets samples by tau/N_steps (relative error position) instead of
the absolute early/late split, then plots R_bad vs R_fix per tercile.

Usage:
    python scripts/6_4b_relative_step_figure.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bad-file", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/bad_prefix_recovery/continuations.jsonl"))
    ap.add_argument("--fix-file", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/minimal_repair/continuations.jsonl"))
    ap.add_argument("--fig-dir", default=str(
        PROJECT_ROOT / "figures/bad_prefix_recovery"))
    return ap.parse_args()


def load_per_sample_rates(path: Path) -> dict:
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
            "tau": recs[0]["tau"],
            "n_steps": recs[0]["n_steps"],
        }
    return sample_rates


def main():
    args = parse_args()
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    bad_rates = load_per_sample_rates(Path(args.bad_file))
    fix_rates = load_per_sample_rates(Path(args.fix_file))
    common_keys = set(bad_rates) & set(fix_rates)
    print(f"Common samples: {len(common_keys)}")

    paired = []
    for key in common_keys:
        rel_pos = bad_rates[key]["tau"] / bad_rates[key]["n_steps"]
        paired.append({
            "rel_pos": rel_pos,
            "r_bad": bad_rates[key]["rate"],
            "r_fix": fix_rates[key]["rate"],
        })

    rel_positions = np.array([p["rel_pos"] for p in paired])
    t33, t67 = np.percentile(rel_positions, [33.3, 66.7])

    tercile_defs = [
        ("early", rel_positions < t33),
        ("mid", (rel_positions >= t33) & (rel_positions < t67)),
        ("late", rel_positions >= t67),
    ]
    tercile_labels = [
        f"early\n($\\tau/N < {t33:.2f}$)",
        f"mid\n($\\tau/N \\in [{t33:.2f},{t67:.2f})$)",
        f"late\n($\\tau/N \\geq {t67:.2f}$)",
    ]

    r_bad_all = np.array([p["r_bad"] for p in paired])
    r_fix_all = np.array([p["r_fix"] for p in paired])

    r_bad_means, r_bad_ses = [], []
    r_fix_means, r_fix_ses = [], []
    ns = []
    for _, mask in tercile_defs:
        rb = r_bad_all[mask]
        rf = r_fix_all[mask]
        n = mask.sum()
        ns.append(n)
        r_bad_means.append(np.mean(rb))
        r_bad_ses.append(np.std(rb, ddof=1) / np.sqrt(n))
        r_fix_means.append(np.mean(rf))
        r_fix_ses.append(np.std(rf, ddof=1) / np.sqrt(n))

    # --- Grouped bar chart ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    x = np.arange(3)
    width = 0.3

    ax.bar(x - width / 2, r_bad_means, width,
           yerr=[1.96 * s for s in r_bad_ses], capsize=4,
           label="$R_{bad}$ (no repair)", color="#e74c3c", alpha=0.8)
    ax.bar(x + width / 2, r_fix_means, width,
           yerr=[1.96 * s for s in r_fix_ses], capsize=4,
           label="$R_{fix}$ (repaired)", color="#3498db", alpha=0.8)

    for i, n in enumerate(ns):
        y_top = max(r_bad_means[i], r_fix_means[i]) + max(
            1.96 * r_bad_ses[i], 1.96 * r_fix_ses[i])
        ax.text(i, y_top + 0.005, f"n={n}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(tercile_labels, fontsize=9)
    ax.set_ylabel("Recovery Rate")
    ax.set_title("Bad Prefix vs. Minimal Repair (relative step)")
    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymax = max(r_bad_means + r_fix_means) * 1.5
    ax.set_ylim(0, ymax)
    plt.tight_layout()

    out_pdf = fig_dir / "fig_bad_vs_repair_relative.pdf"
    out_png = fig_dir / "fig_bad_vs_repair_relative.png"
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")

    for i, (label, _) in enumerate(tercile_defs):
        print(f"  {label}: R_bad={r_bad_means[i]:.4f}, R_fix={r_fix_means[i]:.4f}, n={ns[i]}")


if __name__ == "__main__":
    main()
