#!/usr/bin/env python3
"""Merge a random rollback result with the matching 6_8 baselines."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "nll_drop_fb_last": "#3b82f6",
    "prm_drop_fb_last": "#dc2626",
    "random_uniform": "#525252",
}
LABELS = {
    "nll_drop_fb_last": "NLL rollback",
    "prm_drop_fb_last": "PRM rollback",
    "random_uniform": "Random rollback",
}
MARKERS = {
    "nll_drop_fb_last": "D",
    "prm_drop_fb_last": "o",
    "random_uniform": "X",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-summary", required=True)
    parser.add_argument("--random-summary", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--random-seed", type=int, required=True)
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_rows = [
        row
        for row in load_json(args.source_summary)
        if row["method"] != "SC@40"
    ]
    random_rows = [
        row
        for row in load_json(args.random_summary)
        if row.get("strategy") == "random_uniform"
        and row.get("variant") == "fair"
    ]
    if not random_rows:
        raise ValueError("No random_uniform fair rows found")

    rows = source_rows + random_rows
    for row in rows:
        if row.get("strategy") == "random_uniform":
            row["random_seed"] = args.random_seed

    (out_dir / "comparison_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    sc32 = next(row for row in rows if row["method"] == "SC@32")
    table = [
        "| method | nd | ns | accuracy | tokens/question | vs. SC@32 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        savings = 100 * (
            1 - row["tokens_per_q"] / sc32["tokens_per_q"]
        )
        table.append(
            f"| {row['method']} | {row.get('nd', '')} "
            f"| {row.get('ns', '')} | {row['acc']:.4f} "
            f"| {row['tokens_per_q']:.1f} | {savings:.1f}% |"
        )
    (out_dir / "comparison_table.md").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )

    by_strategy_nd = {
        (row.get("strategy"), row.get("nd")): row
        for row in rows
        if row.get("strategy")
    }
    summary = [
        "# Random Rollback Comparison",
        "",
        f"Random rollback seed: `{args.random_seed}`.",
        "",
        "| nd | random acc | PRM acc | PRM - random | "
        "NLL acc | NLL - random |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for nd in (4, 8, 16):
        random_row = by_strategy_nd[("random_uniform", nd)]
        prm_row = by_strategy_nd[("prm_drop_fb_last", nd)]
        nll_row = by_strategy_nd[("nll_drop_fb_last", nd)]
        summary.append(
            f"| {nd} | {random_row['acc']:.4f} | {prm_row['acc']:.4f} "
            f"| {(prm_row['acc'] - random_row['acc']) * 100:+.1f} pp "
            f"| {nll_row['acc']:.4f} "
            f"| {(nll_row['acc'] - random_row['acc']) * 100:+.1f} pp |"
        )
    summary.extend(
        [
            "",
            "All rollback rows use the fair token accounting from "
            "experiment 6_8 and the same `nd * ns = 32` answer budget.",
        ]
    )
    (out_dir / "rebuttal_summary.md").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sc_rows = [r for r in rows if r["method"].startswith("SC@")]
    for row in sc_rows:
        ax.scatter(
            row["tokens_per_q"],
            row["acc"],
            color="#a3a3a3",
            marker="s",
            s=42,
            zorder=3,
        )
        ax.annotate(
            row["method"],
            (row["tokens_per_q"], row["acc"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    strategies = sorted(
        {
            row.get("strategy")
            for row in rows
            if row.get("strategy") in COLORS
        }
    )
    annotation_offsets = {
        "nll_drop_fb_last": (-42, -15),
        "prm_drop_fb_last": (-30, -14),
        "random_uniform": (5, 8),
    }
    for strategy in strategies:
        strategy_rows = sorted(
            [
                row
                for row in rows
                if row.get("strategy") == strategy
            ],
            key=lambda row: row["tokens_per_q"],
        )
        ax.plot(
            [row["tokens_per_q"] for row in strategy_rows],
            [row["acc"] for row in strategy_rows],
            color=COLORS[strategy],
            marker=MARKERS[strategy],
            linewidth=1.6,
            markersize=6,
            label=LABELS[strategy],
        )
        for row in strategy_rows:
            ax.annotate(
                f"nd={row['nd']}",
                (row["tokens_per_q"], row["acc"]),
                xytext=annotation_offsets[strategy],
                textcoords="offset points",
                fontsize=7,
            )

    ax.set_xlabel("Tokens per question")
    ax.set_ylabel("Majority-vote accuracy")
    dataset_label = out_dir.name.upper()
    ax.set_title(
        f"{dataset_label}: budget-controlled rollback, "
        f"random seed {args.random_seed}"
    )
    ax.margins(x=0.04, y=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            out_dir / f"random_comparison.{extension}",
            dpi=200,
            bbox_inches="tight",
        )
    plt.close(fig)

    print(f"Saved comparison artifacts to {out_dir}")


if __name__ == "__main__":
    main()
