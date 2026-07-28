#!/usr/bin/env python3
"""Build reviewer-facing NLL/PRM versus SC@32 comparison artifacts."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLORS = {
    "SC@32": "#737373",
    "nll_drop_fb_last": "#2563eb",
    "prm_drop_fb_last": "#dc2626",
}
LABELS = {
    "SC@32": "SC@32",
    "nll_drop_fb_last": "NLL rollback",
    "prm_drop_fb_last": "PRM rollback",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    args = parse_args()
    config = load_json(args.config)
    if args.dataset not in config["datasets"]:
        raise ValueError(f"Dataset is not configured: {args.dataset}")

    out_dir = PROJECT_ROOT / config["results_root"] / args.dataset
    eval_path = out_dir / "eval_summary.json"
    rows = load_json(eval_path)
    sc32 = next(row for row in rows if row["method"] == "SC@32")
    rollback_rows = [
        row
        for row in rows
        if row.get("strategy") in config["signals"]
        and row.get("variant") == "fair"
        and row.get("nd") in config["nd"]
    ]
    expected_rollback_rows = len(config["signals"]) * len(config["nd"])
    if len(rollback_rows) != expected_rollback_rows:
        raise ValueError(
            f"Expected {expected_rollback_rows} fair rollback rows, "
            f"found {len(rollback_rows)}"
        )

    comparison_rows = [sc32, *rollback_rows]
    for row in comparison_rows:
        row["accuracy_delta_vs_sc32"] = row["acc"] - sc32["acc"]
        row["token_savings_vs_sc32_pct"] = 100 * (
            1 - row["tokens_per_q"] / sc32["tokens_per_q"]
        )
    (out_dir / "comparison_summary.json").write_text(
        json.dumps(comparison_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    table = [
        "| method | nd | ns | accuracy | vs. SC@32 | "
        "tokens/question | token savings |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        table.append(
            f"| {row['method']} | {row.get('nd', '')} "
            f"| {row.get('ns', '')} | {row['acc']:.4f} "
            f"| {row['accuracy_delta_vs_sc32'] * 100:+.1f} pp "
            f"| {row['tokens_per_q']:.1f} "
            f"| {row['token_savings_vs_sc32_pct']:.1f}% |"
        )
    (out_dir / "comparison_table.md").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )

    by_key = {
        (row.get("strategy"), row.get("nd")): row
        for row in rollback_rows
    }
    summary = [
        f"# DeepSeek-R1-7B-Qwen: {args.dataset} Comparison",
        "",
        f"Samples: `{config['datasets'][args.dataset]['n_sample']}`. "
        f"Generation seed: `{config['seed']}`. "
        f"Answer budget: `nd * ns = {config['budget']}`.",
        "",
        f"SC@32 accuracy is `{sc32['acc']:.4f}` at "
        f"`{sc32['tokens_per_q']:.1f}` tokens per question.",
        "",
        "| signal | nd | ns | accuracy | vs. SC@32 | token savings |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    ns_by_nd = {
        int(nd): int(config["budget"]) // int(nd)
        for nd in config["nd"]
    }
    for signal in config["signals"]:
        for nd in config["nd"]:
            row = by_key[(signal, nd)]
            summary.append(
                f"| `{signal}` | {nd} | {ns_by_nd[nd]} "
                f"| {row['acc']:.4f} "
                f"| {row['accuracy_delta_vs_sc32'] * 100:+.1f} pp "
                f"| {row['token_savings_vs_sc32_pct']:.1f}% |"
            )
    summary.extend(
        [
            "",
            "Rollback rows use the fair prefix-token accounting and the "
            "same answer budget as SC@32.",
        ]
    )
    (out_dir / "rebuttal_summary.md").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )

    nd_values = [int(value) for value in config["nd"]]
    x = np.arange(len(nd_values))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))

    axes[0].axhline(
        sc32["acc"],
        color=COLORS["SC@32"],
        linestyle="--",
        linewidth=1.5,
        label=LABELS["SC@32"],
    )
    for offset, signal in zip((-width / 1.8, width / 1.8), config["signals"]):
        signal_rows = [by_key[(signal, nd)] for nd in nd_values]
        axes[0].bar(
            x + offset,
            [row["acc"] for row in signal_rows],
            width,
            color=COLORS[signal],
            label=LABELS[signal],
        )
    axes[0].set_xticks(x, [str(nd) for nd in nd_values])
    axes[0].set_xlabel("Number of drafts (nd)")
    axes[0].set_ylabel("Majority-vote accuracy")
    axes[0].set_title("Accuracy versus SC@32")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].scatter(
        sc32["tokens_per_q"],
        sc32["acc"],
        color=COLORS["SC@32"],
        marker="s",
        s=65,
        label=LABELS["SC@32"],
        zorder=3,
    )
    for signal, marker in zip(config["signals"], ("D", "o")):
        signal_rows = sorted(
            [by_key[(signal, nd)] for nd in nd_values],
            key=lambda row: row["tokens_per_q"],
        )
        axes[1].plot(
            [row["tokens_per_q"] for row in signal_rows],
            [row["acc"] for row in signal_rows],
            color=COLORS[signal],
            marker=marker,
            linewidth=1.6,
            label=LABELS[signal],
        )
        for row in signal_rows:
            axes[1].annotate(
                f"nd={row['nd']}",
                (row["tokens_per_q"], row["acc"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
    axes[1].set_xlabel("Tokens per question")
    axes[1].set_ylabel("Majority-vote accuracy")
    axes[1].set_title("Accuracy-compute frontier")
    axes[1].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle(
        f"DeepSeek-R1-Distill-Qwen-7B on {args.dataset} "
        f"(n={config['datasets'][args.dataset]['n_sample']})"
    )
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            out_dir / f"signal_vs_sc32.{extension}",
            dpi=200,
            bbox_inches="tight",
        )
    plt.close(fig)
    print(f"Saved comparison artifacts to {out_dir}")


if __name__ == "__main__":
    main()
