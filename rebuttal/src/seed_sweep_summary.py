"""Validation and aggregation for the corrected MATH-500 seed sweep."""

import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Dict, Iterable, List


GENERATION_TYPES = {"draft", "suffix", "fullsc"}


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_seed_run(
    out_dir: Path,
    run_seed: int,
    n_questions: int,
    nd_max: int,
    budget: int,
    n_signals: int,
    suffixes_per_signal_per_question: int,
    report_filename: str = "seed_validation.json",
) -> dict:
    """Validate counts and per-request seed separation for one run."""
    records = read_jsonl(out_dir / "checkpoint.jsonl")
    by_type: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        by_type[record.get("task_type", "")].append(record)

    expected = {
        "draft": n_questions * nd_max,
        "fullsc": n_questions * (budget - nd_max),
        "suffix": (
            n_questions
            * n_signals
            * suffixes_per_signal_per_question
        ),
    }
    actual = {
        task_type: len(by_type.get(task_type, []))
        for task_type in expected
    }
    generation_records = [
        record
        for record in records
        if record.get("task_type") in GENERATION_TYPES
    ]
    seeds = [
        record.get("sampling_seed") for record in generation_records
    ]
    missing_seed_count = sum(seed is None for seed in seeds)
    concrete_seeds = [seed for seed in seeds if seed is not None]

    sc_texts = defaultdict(list)
    for record in by_type.get("draft", []):
        sc_texts[record["doc_id"]].append(
            record.get("draft_text", "")
        )
    for record in by_type.get("fullsc", []):
        sc_texts[record["doc_id"]].append(
            record.get("sc_text", "")
        )
    sc_pool_sizes = [len(values) for values in sc_texts.values()]
    sc_unique_counts = [
        len(set(values)) for values in sc_texts.values()
    ]

    summary_path = out_dir / "eval_summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists() else []
    )
    sc32_rows = [
        row for row in summary if row.get("method") == "SC@32"
    ]

    errors = []
    for task_type, count in expected.items():
        if actual[task_type] != count:
            errors.append(
                f"{task_type}: expected {count}, got "
                f"{actual[task_type]}"
            )
    if missing_seed_count:
        errors.append(
            f"{missing_seed_count} generation records lack sampling_seed"
        )
    if len(concrete_seeds) != len(set(concrete_seeds)):
        errors.append("sampling_seed values are not globally unique")
    if len(sc_pool_sizes) != n_questions:
        errors.append(
            f"SC pool has {len(sc_pool_sizes)} documents, "
            f"expected {n_questions}"
        )
    if sc_pool_sizes and any(size != budget for size in sc_pool_sizes):
        errors.append("at least one document does not have SC@32")
    if len(sc32_rows) != 1:
        errors.append(
            f"expected one SC@32 summary row, got {len(sc32_rows)}"
        )

    report = {
        "status": "pass" if not errors else "fail",
        "run_seed": run_seed,
        "seed_schema": "lost-in-the-second/per-request-v1",
        "expected_counts": expected,
        "actual_counts": actual,
        "generation_records": len(generation_records),
        "unique_sampling_seeds": len(set(concrete_seeds)),
        "missing_sampling_seeds": missing_seed_count,
        "sc_documents": len(sc_pool_sizes),
        "sc_pool_size_min": min(sc_pool_sizes) if sc_pool_sizes else 0,
        "sc_pool_size_max": max(sc_pool_sizes) if sc_pool_sizes else 0,
        "sc_unique_text_mean": (
            fmean(sc_unique_counts) if sc_unique_counts else 0.0
        ),
        "sc_fully_collapsed_documents": sum(
            count <= 1 for count in sc_unique_counts
        ),
        "sc32_acc": (
            sc32_rows[0].get("acc") if len(sc32_rows) == 1 else None
        ),
        "errors": errors,
    }
    report_path = out_dir / report_filename
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if errors:
        raise RuntimeError(
            f"Seed run validation failed: {report_path}: "
            + "; ".join(errors)
        )
    return report


def aggregate_seed_runs(
    seed_dirs: Iterable[tuple[int, Path]],
    output_path: Path,
) -> List[dict]:
    """Aggregate available per-seed eval summaries."""
    methods: Dict[str, List[dict]] = defaultdict(list)
    for seed, out_dir in seed_dirs:
        summary_path = out_dir / "eval_summary.json"
        if not summary_path.exists():
            continue
        for row in json.loads(
            summary_path.read_text(encoding="utf-8")
        ):
            methods[row["method"]].append({
                "seed": seed,
                "acc": row["acc"],
                "tokens_per_q": row["tokens_per_q"],
            })

    summary = []
    for method, entries in sorted(methods.items()):
        accuracies = [entry["acc"] for entry in entries]
        tokens = [entry["tokens_per_q"] for entry in entries]
        summary.append({
            "method": method,
            "n_seeds": len(entries),
            "acc_mean": fmean(accuracies),
            "acc_std": pstdev(accuracies),
            "acc_min": min(accuracies),
            "acc_max": max(accuracies),
            "tokens_per_q_mean": fmean(tokens),
            "tokens_per_q_std": pstdev(tokens),
            "per_seed": entries,
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def aggregate_sc_seed_runs(
    seed_dirs: Iterable[tuple[int, Path]],
    output_path: Path,
) -> List[dict]:
    """Aggregate validated SC@32-only prefill runs."""
    entries = []
    for seed, out_dir in seed_dirs:
        report_path = out_dir / "sc_seed_validation.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "pass":
            continue
        entries.append({
            "seed": seed,
            "acc": report["sc32_acc"],
            "sc_unique_text_mean": report["sc_unique_text_mean"],
            "sc_fully_collapsed_documents": (
                report["sc_fully_collapsed_documents"]
            ),
            "unique_sampling_seeds": report["unique_sampling_seeds"],
        })

    if not entries:
        summary = []
    else:
        accuracies = [entry["acc"] for entry in entries]
        summary = [{
            "method": "SC@32",
            "n_seeds": len(entries),
            "acc_mean": fmean(accuracies),
            "acc_std": pstdev(accuracies),
            "acc_min": min(accuracies),
            "acc_max": max(accuracies),
            "per_seed": entries,
        }]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
