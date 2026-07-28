#!/usr/bin/env python3
"""Summarize audited random/NLL/PRM controls across run seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
DEFAULT_DATASETS = ["math500", "hotpotqa_open"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(
            "rebuttal/results/"
            "qwen2.5_3b_instruct_seed_variance"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DEFAULT_DATASETS,
        default=DEFAULT_DATASETS,
    )
    return parser.parse_args()


def _load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _require_passed(result_dir: Path) -> None:
    report = _load_json(result_dir / "audit_report.json")
    if report.get("status") != "passed":
        raise RuntimeError(f"Audit did not pass: {result_dir}")


def _load_seed_rows(
    results_root: Path,
    dataset: str,
    seed: int,
) -> list[dict]:
    run_root = results_root / f"seed_{seed}" / dataset
    position_dir = run_root / "position_controls"
    signal_dir = run_root / "nll_prm_controls"
    _require_passed(position_dir)
    _require_passed(signal_dir)

    position_rows = _load_json(position_dir / "eval_summary.json")
    signal_rows = _load_json(signal_dir / "eval_summary.json")
    position_map = {row["method"]: row for row in position_rows}
    signal_map = {row["method"]: row for row in signal_rows}
    for name in ("Greedy@1",):
        if name not in position_map or name not in signal_map:
            raise RuntimeError(
                f"{dataset} seed {seed}: missing {name}"
            )
        for metric in ("acc", "tokens_per_q"):
            if not math.isclose(
                float(position_map[name][metric]),
                float(signal_map[name][metric]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"{dataset} seed {seed}: inconsistent {name} "
                    f"{metric}"
                )

    rows = []
    for source, records in (
        ("position", position_rows),
        ("signal", signal_rows[1:]),
    ):
        for record in records:
            if source == "position" and (
                record.get("strategy") not in (None, "random_uniform")
            ):
                continue
            row = {
                "dataset": dataset,
                "seed": seed,
                "source": source,
                "method": record["method"],
                "acc": float(record["acc"]),
                "tokens_per_q": float(record["tokens_per_q"]),
            }
            for metric in (
                "f1",
                "signal_tokens_per_q",
                "tokens_per_q_with_signal",
                "trigger_rate",
            ):
                if metric in record:
                    row[metric] = float(record[metric])
            rows.append(row)
    return rows


def _summarize(rows: list[dict], seeds: list[int]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["method"])].append(row)

    summaries = []
    for (dataset, method), records in sorted(groups.items()):
        observed_seeds = sorted(record["seed"] for record in records)
        if observed_seeds != sorted(seeds):
            raise RuntimeError(
                f"{dataset} {method}: seeds={observed_seeds}, "
                f"expected={sorted(seeds)}"
            )
        summary = {
            "dataset": dataset,
            "method": method,
            "n_seeds": len(records),
            "seeds": observed_seeds,
        }
        metric_names = sorted(
            set.intersection(
                *(set(record) for record in records)
            )
            & {
                "acc",
                "f1",
                "tokens_per_q",
                "signal_tokens_per_q",
                "tokens_per_q_with_signal",
                "trigger_rate",
            }
        )
        for metric in metric_names:
            values = [record[metric] for record in records]
            summary[f"{metric}_mean"] = statistics.fmean(values)
            summary[f"{metric}_variance"] = statistics.variance(values)
            summary[f"{metric}_std"] = statistics.stdev(values)
            summary[f"{metric}_values"] = values
        summaries.append(summary)
    return summaries


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _method_label(method: str) -> str:
    return (
        method.replace("random_uniform", "Random")
        .replace("nll_drop_fb_last", "NLL")
        .replace("prm_drop_fb_last", "PRM")
        .replace("_nd", " nd=")
        .replace("_ns", " ns=")
    )


def _plot(summaries: list[dict], out_dir: Path) -> None:
    datasets = sorted({row["dataset"] for row in summaries})
    panels = []
    for dataset in datasets:
        panels.append((dataset, "acc"))
        if any(
            row["dataset"] == dataset and "f1_mean" in row
            for row in summaries
        ):
            panels.append((dataset, "f1"))
    figure, axes = plt.subplots(
        len(panels),
        1,
        figsize=(10.5, 4.6 * len(panels)),
        squeeze=False,
    )
    colors = {
        "Greedy@1": "#6b7280",
        "Random": "#d97706",
        "NLL": "#2563eb",
        "PRM": "#dc2626",
    }
    for axis, (dataset, metric) in zip(axes[:, 0], panels):
        rows = [
            row for row in summaries
            if row["dataset"] == dataset
            and f"{metric}_mean" in row
        ]
        labels = [_method_label(row["method"]) for row in rows]
        means = [row[f"{metric}_mean"] for row in rows]
        errors = [row[f"{metric}_std"] for row in rows]
        bar_colors = []
        for label in labels:
            prefix = label.split()[0]
            bar_colors.append(colors.get(prefix, "#6b7280"))
        x_positions = list(range(len(rows)))
        axis.bar(
            x_positions,
            means,
            yerr=errors,
            color=bar_colors,
            capsize=3,
            width=0.72,
        )
        axis.set_xticks(x_positions, labels, rotation=30, ha="right")
        axis.set_ylabel(
            "Token F1" if metric == "f1"
            else "Exact match / accuracy"
        )
        axis.set_title(
            f"{dataset} {metric.upper()}: mean and standard deviation "
            "over 5 seeds"
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(
            out_dir / f"seed_variance.{extension}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(figure)


def main():
    args = parse_args()
    if len(args.seeds) < 2:
        raise ValueError("At least two seeds are required for variance")
    out_dir = args.out_dir or args.results_root / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in args.datasets:
        for seed in args.seeds:
            rows.extend(
                _load_seed_rows(args.results_root, dataset, seed)
            )
    summaries = _summarize(rows, args.seeds)
    (out_dir / "per_seed_results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "seed_variance_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "per_seed_results.csv", rows)
    flat_summaries = [
        {
            key: value for key, value in row.items()
            if not key.endswith("_values") and key != "seeds"
        }
        for row in summaries
    ]
    _write_csv(out_dir / "seed_variance_summary.csv", flat_summaries)
    _plot(summaries, out_dir)
    print(
        f"Saved {len(rows)} per-seed rows and "
        f"{len(summaries)} summaries to {out_dir}"
    )


if __name__ == "__main__":
    main()
