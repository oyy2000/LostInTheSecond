#!/usr/bin/env python3
"""
Fine-grained three-way comparison: R_bad / R_rollback / R_fix as line plots
over binned relative error position (tau/N_steps).

Usage:
    python scripts/6_5c_three_way_fine_grained.py
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
RES = PROJECT_ROOT / "results/gsm8k_3b_multi_sample"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bad-file", default=str(RES / "bad_prefix_recovery/continuations.jsonl"))
    ap.add_argument("--rollback-file", default=str(RES / "rollback_one_step/continuations.jsonl"))
    ap.add_argument("--fix-file", default=str(RES / "minimal_repair/continuations.jsonl"))
    ap.add_argument("--fig-dir", default=str(PROJECT_ROOT / "figures/bad_prefix_recovery"))
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--min-bin-n", type=int, default=10)
    return ap.parse_args()


def load_rates(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]
    by_sample = defaultdict(list)
    for r in rows:
        by_sample[(r["doc_id"], r["sample_idx"])].append(r)
    out = {}
    for key, recs in by_sample.items():
        n_correct = sum(1 for r in recs if r["exact_match"] >= 1.0)
        out[key] = {
            "rate": n_correct / len(recs),
            "tau": recs[0]["tau"],
            "n_steps": recs[0]["n_steps"],
        }
    return out


def main():
    args = parse_args()
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    bad = load_rates(Path(args.bad_file))
    rollback = load_rates(Path(args.rollback_file))
    fix = load_rates(Path(args.fix_file))

    keys = list(set(bad) & set(rollback) & set(fix))
    print(f"Common samples: {len(keys)}")

    rel_pos = np.array([bad[k]["tau"] / bad[k]["n_steps"] for k in keys])
    r_bad = np.array([bad[k]["rate"] for k in keys])
    r_rb = np.array([rollback[k]["rate"] for k in keys])
    r_fix = np.array([fix[k]["rate"] for k in keys])

    methods = [
        ("$R_{bad}$ (keep error)", r_bad, "#e74c3c", "o"),
        ("$R_{rollback}$ (resample before)", r_rb, "#f39c12", "s"),
        ("$R_{fix}$ (repair step)", r_fix, "#3498db", "D"),
    ]

    bin_edges = np.linspace(rel_pos.min() - 1e-9, rel_pos.max() + 1e-9, args.n_bins + 1)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    for name, arr, color, marker in methods:
        centers, means, ci_lo, ci_hi, bin_ns = [], [], [], [], []
        for i in range(args.n_bins):
            mask = (rel_pos >= bin_edges[i]) & (rel_pos < bin_edges[i + 1])
            n = mask.sum()
            if n < args.min_bin_n:
                continue
            vals = arr[mask]
            m = np.mean(vals)
            se = np.std(vals, ddof=1) / np.sqrt(n)
            centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            means.append(m)
            ci_lo.append(m - 1.96 * se)
            ci_hi.append(m + 1.96 * se)
            bin_ns.append(n)

        centers = np.array(centers)
        means = np.array(means)
        ci_lo = np.array(ci_lo)
        ci_hi = np.array(ci_hi)

        ax.plot(centers, means, marker=marker, color=color, linewidth=2,
                markersize=6, label=name, zorder=3)
        ax.fill_between(centers, ci_lo, ci_hi, color=color, alpha=0.12, zorder=1)

    for i in range(args.n_bins):
        mask = (rel_pos >= bin_edges[i]) & (rel_pos < bin_edges[i + 1])
        n = mask.sum()
        if n >= args.min_bin_n:
            cx = (bin_edges[i] + bin_edges[i + 1]) / 2
            ax.text(cx, -0.025, f"{n}", ha="center", va="top", fontsize=7, color="#888")

    ax.set_xlabel("Relative Error Position ($\\tau / N_{steps}$)")
    ax.set_ylabel("Recovery Rate")
    ax.set_title("Recovery Rate by Relative Error Position (fine-grained)")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        fig.savefig(fig_dir / f"fig_three_way_fine.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir / 'fig_three_way_fine.png'}")


if __name__ == "__main__":
    main()
