#!/usr/bin/env python3
"""
Two-way efficiency figure: R_bad and R_rollback normalized by extra tokens.

For each (doc_id, sample_idx), compute:
  - R_bad / extra_tokens_bad
  - R_rollback / extra_tokens_rollback

where extra_tokens is estimated from the character length of the remaining
steps in the original trajectory (chars / 4 as a rough token proxy).

  R_bad prefix = steps[0..tau]       -> extra = steps[tau+1..end]
  R_rollback prefix = steps[0..tau-1] -> extra = steps[tau..end]

Buckets by tau/N_steps terciles.

Usage:
    python scripts/6_5d_two_way_efficiency_figure.py
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
    ap.add_argument("--early-file", default=str(RES / "first_error/bucket_early.json"))
    ap.add_argument("--late-file", default=str(RES / "first_error/bucket_late.json"))
    ap.add_argument("--fig-dir", default=str(PROJECT_ROOT / "figures/bad_prefix_recovery"))
    ap.add_argument("--chars-per-token", type=float, default=4.0)
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


def load_step_texts(early_path: Path, late_path: Path) -> dict:
    """Load original trajectory step texts keyed by (doc_id, sample_idx)."""
    out = {}
    for p in [early_path, late_path]:
        data = json.loads(p.read_text("utf-8"))
        for r in data:
            key = (r["doc_id"], r["sample_idx"])
            out[key] = {
                "steps": r["steps"],
                "tau": r["tau"],
                "n_steps": r["n_steps"],
            }
    return out


def main():
    args = parse_args()
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    cpt = args.chars_per_token

    bad = load_rates(Path(args.bad_file))
    rollback = load_rates(Path(args.rollback_file))
    originals = load_step_texts(Path(args.early_file), Path(args.late_file))

    common = set(bad) & set(rollback) & set(originals)
    print(f"Common samples: {len(common)}")

    keys = list(common)
    r_bad = np.array([bad[k]["rate"] for k in keys])
    r_rb = np.array([rollback[k]["rate"] for k in keys])
    rel_pos = np.array([bad[k]["tau"] / bad[k]["n_steps"] for k in keys])

    extra_tok_bad = []
    extra_tok_rb = []
    for k in keys:
        steps = originals[k]["steps"]
        tau = originals[k]["tau"]
        chars_after_tau = sum(len(s) for s in steps[tau + 1:])
        chars_from_tau = sum(len(s) for s in steps[tau:])
        extra_tok_bad.append(max(chars_after_tau / cpt, 1.0))
        extra_tok_rb.append(max(chars_from_tau / cpt, 1.0))

    extra_tok_bad = np.array(extra_tok_bad)
    extra_tok_rb = np.array(extra_tok_rb)

    eff_bad = r_bad / extra_tok_bad
    eff_rb = r_rb / extra_tok_rb

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
        ("$R_{bad}$ / extra tokens\n(keep error)", eff_bad, "#e74c3c"),
        ("$R_{rollback}$ / extra tokens\n(resample before)", eff_rb, "#f39c12"),
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
    width = 0.28
    offsets = np.linspace(-(n_methods - 1) * width / 2,
                          (n_methods - 1) * width / 2, n_methods)

    for (name, _, color), off in zip(methods, offsets):
        ax.bar(x + off, means[name], width,
               yerr=[1.96 * s for s in ses[name]], capsize=3,
               label=name.split("\n")[0], color=color, alpha=0.85,
               edgecolor="white", linewidth=0.5)

    for i, n in enumerate(ns):
        y_top = max(means[name][i] + 1.96 * ses[name][i]
                    for name, _, _ in methods)
        ax.text(i, y_top + 0.0001, f"n={n}",
                ha="center", va="bottom", fontsize=8, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Recovery Rate / Extra Tokens (approx)")
    ax.set_title("Token-Efficiency of Recovery by Relative Error Position")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymax = max(max(means[name]) for name, _, _ in methods) * 1.3
    ax.set_ylim(0, ymax)
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        fig.savefig(fig_dir / f"fig_two_way_efficiency.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir / 'fig_two_way_efficiency.png'}")

    for i, (lbl, _) in enumerate(terciles):
        parts = [f"{lbl}: "]
        for name, _, _ in methods:
            short = name.split("\n")[0]
            parts.append(f"{short}={means[name][i]:.6f}")
        print("  " + ", ".join(parts))

    print("\nMean extra tokens per method:")
    for _, mask in terciles:
        print(f"  bad: {extra_tok_bad[mask].mean():.1f}, "
              f"rollback: {extra_tok_rb[mask].mean():.1f}")


if __name__ == "__main__":
    main()
