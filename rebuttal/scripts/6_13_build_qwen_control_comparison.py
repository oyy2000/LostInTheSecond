#!/usr/bin/env python3
"""Combine same-draft Qwen position and NLL/PRM control results."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POSITION_ROOT = (
    PROJECT_ROOT
    / "rebuttal/results/qwen2.5_3b_instruct_position_controls/seed_42"
)
SIGNAL_ROOT = (
    PROJECT_ROOT
    / "rebuttal/results/qwen2.5_3b_instruct_nll_prm_controls/seed_42"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["math500", "hotpotqa_open"],
        required=True,
    )
    parser.add_argument("--position-dir", type=Path, default=None)
    parser.add_argument("--signal-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def load_results(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def strategy_label(strategy: str) -> str:
    labels = {
        "random_uniform": "Random",
        "fixed_25": "Fixed 25%",
        "fixed_50": "Fixed 50%",
        "fixed_75": "Fixed 75%",
        "final_step_only": "Final step only",
        "nll_drop_fb_last": "NLL drop",
        "prm_drop_fb_last": "PRM drop",
    }
    return labels[strategy]


def main():
    args = parse_args()
    position_dir = args.position_dir or POSITION_ROOT / args.dataset
    signal_dir = args.signal_dir or SIGNAL_ROOT / args.dataset
    out_dir = args.out_dir or signal_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    position = load_results(position_dir / "eval_summary.json")
    signals = load_results(signal_dir / "eval_summary.json")
    position_greedy = next(
        row for row in position if row["method"] == "Greedy@1"
    )
    signal_greedy = next(
        row for row in signals if row["method"] == "Greedy@1"
    )
    for key in ("acc", "tokens_per_q", "total_tokens"):
        if position_greedy[key] != signal_greedy[key]:
            raise ValueError(f"Greedy mismatch for {key}")
    if args.dataset in ("hotpotqa", "hotpotqa_open") and (
        position_greedy["f1"] != signal_greedy["f1"]
    ):
        raise ValueError("Greedy mismatch for F1")

    combined = [
        position_greedy,
        *[
            row for row in position
            if row["method"] != "Greedy@1"
        ],
        *[
            row for row in signals
            if row["method"] != "Greedy@1"
        ],
    ]
    (out_dir / "all_control_comparison.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    hotpot = args.dataset in ("hotpotqa", "hotpotqa_open")
    lines = [
        "| method | nd | ns | accuracy | generated tokens/q"
        + (" | F1 |" if hotpot else " |"),
        "|---|---:|---:|---:|---:|"
        + ("---:|" if hotpot else ""),
    ]
    for row in combined:
        line = (
            f"| {row['method']} | {row.get('nd', '')} "
            f"| {row.get('ns', '')} | {row['acc']:.4f} "
            f"| {row['tokens_per_q']:.1f} "
        )
        line += f"| {row['f1']:.4f} |" if hotpot else "|"
        lines.append(line)
    (out_dir / "all_control_comparison.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    strategies = list(dict.fromkeys(
        row["strategy"]
        for row in combined
        if row.get("strategy")
    ))
    colors = {
        "random_uniform": "#525252",
        "fixed_25": "#2563eb",
        "fixed_50": "#059669",
        "fixed_75": "#d97706",
        "final_step_only": "#7c3aed",
        "nll_drop_fb_last": "#0891b2",
        "prm_drop_fb_last": "#dc2626",
    }
    markers = {
        "random_uniform": "X",
        "fixed_25": "o",
        "fixed_50": "s",
        "fixed_75": "D",
        "final_step_only": "^",
        "nll_drop_fb_last": "P",
        "prm_drop_fb_last": "v",
    }
    metric_names = ["acc", "f1"] if hotpot else ["acc"]
    figure, axes = plt.subplots(
        1,
        len(metric_names),
        figsize=(12.2, 4.8) if hotpot else (7.4, 4.8),
        squeeze=False,
    )
    for axis, metric in zip(axes[0], metric_names):
        for strategy in strategies:
            rows = sorted(
                [
                    row for row in combined
                    if row.get("strategy") == strategy
                ],
                key=lambda row: row["tokens_per_q"],
            )
            if not rows:
                raise ValueError(f"Missing strategy: {strategy}")
            axis.plot(
                [row["tokens_per_q"] for row in rows],
                [row[metric] for row in rows],
                color=colors[strategy],
                marker=markers[strategy],
                linewidth=1.5,
                markersize=5.5,
                label=strategy_label(strategy),
            )
            for row in rows:
                axis.annotate(
                    str(row["nd"]),
                    (row["tokens_per_q"], row[metric]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )
        axis.set_xlabel("Generated tokens per question")
        axis.set_ylabel("F1" if metric == "f1" else (
            "EM" if hotpot else "Accuracy"
        ))
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.margins(x=0.05, y=0.12)
    axes[0][0].legend(frameon=False, fontsize=7, ncol=2)
    figure.suptitle(
        f"{args.dataset.upper()}: Qwen2.5-3B same-draft controls"
    )
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(
            out_dir / f"all_control_comparison.{extension}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(figure)
    print(
        f"Saved {len(combined)} combined rows and figures to {out_dir}"
    )


if __name__ == "__main__":
    main()
