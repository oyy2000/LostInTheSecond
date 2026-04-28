#!/usr/bin/env python3
"""
Three-way comparison figure (relative step version):
  R_bad (continue with error), R_rollback (resample before error), R_fix (repair error step).

Buckets by tau/N_steps terciles.

Usage:
    python scripts/6_5b_three_way_relative_figure.py
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

    common = set(bad) & set(rollback) & set(fix)
    print(f"Common samples: {len(common)}")

    rel_pos = np.array([bad[k]["tau"] / bad[k]["n_steps"] for k in common])
    keys = list(common)
    r_bad = np.array([bad[k]["rate"] for k in keys])
    r_rb = np.array([rollback[k]["rate"] for k in keys])
    r_fix = np.array([fix[k]["rate"] for k in keys])
    rel_pos = np.array([bad[k]["tau"] / bad[k]["n_steps"] for k in keys])

    t33, t67 = np.percentile(rel_pos, [33.3, 66.7])
    terciles = [
        ("early", rel_pos < t33),
        ("mid", (rel_pos >= t33) & (rel_pos < t67)),
        ("late", rel_pos >= t67),
    ]
    labels = [
        f"early\n($\\tau/N<{t33:.2f}$)",
        f"mid\n($\\tau/N\\in[{t33:.2f},{t67:.2f})$)",
        f"late\n($\\tau/N\\geq{t67:.2f}$)",
    ]

    methods = [
        ("$R_{bad}$\n(keep error)", r_bad, "#e74c3c"),
        ("$R_{rollback}$\n(resample before)", r_rb, "#f39c12"),
        ("$R_{fix}$\n(repair step)", r_fix, "#3498db"),
    ]

    means = {name: [] for name, _, _ in methods}
    ses = {name: [] for name, _, _ in methods}
    ns = []
    for _, mask in terciles:
        n = mask.sum()
        ns.append(n)
        for name, arr, _ in methods:
            vals = arr[mask]
            means[name].append(np.mean(vals))
            ses[name].append(np.std(vals, ddof=1) / np.sqrt(n))

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    x = np.arange(3)
    n_methods = len(methods)
    width = 0.22
    offsets = np.linspace(-(n_methods - 1) * width / 2, (n_methods - 1) * width / 2, n_methods)

    for (name, _, color), off in zip(methods, offsets):
        ax.bar(x + off, means[name], width,
               yerr=[1.96 * s for s in ses[name]], capsize=3,
               label=name.replace("\n", " "), color=color, alpha=0.85,
               edgecolor="white", linewidth=0.5)

    for i, n in enumerate(ns):
        y_top = max(means[name][i] + 1.96 * ses[name][i] for name, _, _ in methods)
        ax.text(i, y_top + 0.01, f"n={n}", ha="center", va="bottom", fontsize=8, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Recovery Rate")
    ax.set_title("Three-Way Comparison by Relative Error Position")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymax = max(max(means[name]) for name, _, _ in methods) * 1.25
    ax.set_ylim(0, ymax)
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        fig.savefig(fig_dir / f"fig_three_way_relative.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir / 'fig_three_way_relative.png'}")

    for i, (lbl, _) in enumerate(terciles):
        parts = [f"{lbl}: "]
        for name, _, _ in methods:
            short = name.split("\n")[0].strip("$").replace("R_", "R_")
            parts.append(f"{short}={means[name][i]:.4f}")
        print("  " + ", ".join(parts))


if __name__ == "__main__":
    main()
