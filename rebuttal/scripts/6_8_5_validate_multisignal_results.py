#!/usr/bin/env python3
"""Validate completeness of one rebuttal multisignal result directory."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def identity(record):
    task_type = record.get("task_type")
    if task_type in {"draft", "prm", "logprob"}:
        return task_type, record["doc_id"], record["draft_idx"]
    if task_type == "suffix":
        return (
            task_type,
            record["doc_id"],
            record["draft_idx"],
            record["strategy"],
            record["rollback_step"],
            record["suffix_idx"],
        )
    if task_type == "fullsc":
        return task_type, record["doc_id"], record["sc_idx"]
    raise ValueError(f"Unknown task type: {task_type!r}")


def require(condition, message, checks):
    if not condition:
        raise ValueError(message)
    checks.append(message)


def main():
    args = parse_args()
    config = load_json(args.config)
    if args.dataset not in config["datasets"]:
        raise ValueError(f"Dataset is not configured: {args.dataset}")

    out_dir = PROJECT_ROOT / config["results_root"] / args.dataset
    checkpoint_path = out_dir / "checkpoint.jsonl"
    records = [
        json.loads(line)
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    n_questions = int(config["datasets"][args.dataset]["n_sample"])
    nd_values = [int(value) for value in config["nd"]]
    nd_max = max(nd_values)
    budget = int(config["budget"])
    signals = list(config["signals"])
    checks = []

    identities = [identity(record) for record in records]
    require(
        len(identities) == len(set(identities)),
        "All checkpoint task identities are unique",
        checks,
    )
    by_type = Counter(record["task_type"] for record in records)
    expected_counts = {
        "draft": n_questions * nd_max,
        "prm": n_questions * nd_max,
        "logprob": n_questions * nd_max,
        "suffix": n_questions * len(signals) * sum(
            max(budget // nd for nd in nd_values if draft_idx < nd)
            for draft_idx in range(nd_max)
        ),
        "fullsc": n_questions * (budget - nd_max),
    }
    require(
        dict(by_type) == expected_counts,
        f"Checkpoint record counts match {expected_counts}",
        checks,
    )

    doc_ids = {
        record["doc_id"]
        for record in records
        if record["task_type"] == "draft"
    }
    require(
        len(doc_ids) == n_questions,
        f"Checkpoint contains exactly {n_questions} question IDs",
        checks,
    )

    draft_indices = defaultdict(set)
    prm_indices = defaultdict(set)
    logprob_indices = defaultdict(set)
    sc_indices = defaultdict(set)
    suffix_counts = defaultdict(Counter)
    for record in records:
        task_type = record["task_type"]
        doc_id = record["doc_id"]
        if task_type == "draft":
            draft_indices[doc_id].add(record["draft_idx"])
        elif task_type == "prm":
            prm_indices[doc_id].add(record["draft_idx"])
        elif task_type == "logprob":
            logprob_indices[doc_id].add(record["draft_idx"])
        elif task_type == "fullsc":
            sc_indices[doc_id].add(record["sc_idx"])
        elif task_type == "suffix":
            suffix_counts[(doc_id, record["strategy"])][
                record["draft_idx"]
            ] += 1

    expected_draft_indices = set(range(nd_max))
    expected_sc_indices = set(range(budget - nd_max))
    expected_suffix_by_draft = {
        draft_idx: max(
            budget // nd for nd in nd_values if draft_idx < nd
        )
        for draft_idx in range(nd_max)
    }
    for doc_id in doc_ids:
        require(
            draft_indices[doc_id] == expected_draft_indices,
            f"{doc_id}: all draft indices are present",
            checks,
        )
        require(
            prm_indices[doc_id] == expected_draft_indices,
            f"{doc_id}: all PRM indices are present",
            checks,
        )
        require(
            logprob_indices[doc_id] == expected_draft_indices,
            f"{doc_id}: all NLL indices are present",
            checks,
        )
        require(
            sc_indices[doc_id] == expected_sc_indices,
            f"{doc_id}: SC supplement plus drafts yields SC@{budget}",
            checks,
        )
        for signal in signals:
            require(
                dict(suffix_counts[(doc_id, signal)])
                == expected_suffix_by_draft,
                f"{doc_id}: {signal} suffix allocation is complete",
                checks,
            )

    step_tokens = load_json(out_dir / "draft_step_tokens.json")
    require(
        len(step_tokens) == expected_counts["draft"],
        "Fair prefix token counts cover every draft",
        checks,
    )

    eval_rows = load_json(out_dir / "eval_summary.json")
    methods = Counter(row["method"] for row in eval_rows)
    require(
        methods["SC@32"] == 1,
        "Evaluation contains exactly one SC@32 row",
        checks,
    )
    for signal in signals:
        for nd in nd_values:
            ns = budget // nd
            method = f"rollback_{signal}_nd{nd}_ns{ns}_fair"
            require(
                methods[method] == 1,
                f"Evaluation contains {method}",
                checks,
            )
    for row in eval_rows:
        require(
            math.isfinite(row["acc"]) and 0 <= row["acc"] <= 1,
            f"{row['method']}: accuracy is finite and within [0, 1]",
            checks,
        )
        require(
            math.isfinite(row["tokens_per_q"])
            and row["tokens_per_q"] > 0,
            f"{row['method']}: tokens per question is finite and positive",
            checks,
        )

    required_artifacts = [
        "comparison_summary.json",
        "comparison_table.md",
        "rebuttal_summary.md",
        "signal_vs_sc32.png",
        "signal_vs_sc32.pdf",
        "fig_efficiency_frontier.png",
        "fig_efficiency_frontier.pdf",
    ]
    for name in required_artifacts:
        require(
            (out_dir / name).is_file() and (out_dir / name).stat().st_size > 0,
            f"Artifact exists and is non-empty: {name}",
            checks,
        )

    report = {
        "status": "passed",
        "dataset": args.dataset,
        "n_questions": n_questions,
        "checkpoint_records": len(records),
        "record_counts": expected_counts,
        "signals": signals,
        "budget": budget,
        "nd": nd_values,
        "checks_passed": len(checks),
    }
    (out_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
