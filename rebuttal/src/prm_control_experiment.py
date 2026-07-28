"""Preparation and validation helpers for PRM control experiments."""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator

from rebuttal.src.prm_control_strategies import (
    MAX_PRM_DROP_SIGNAL,
    PRM_BELOW_THRESHOLD_SIGNAL,
    PRM_CONTROL_SIGNALS,
    max_prm_drop_assignment,
    prm_below_threshold_assignment,
)
from rebuttal.src.stable_sampling import derive_sampling_seed
from rebuttal.src.threshold_sweep import suffixes_per_draft
from rebuttal.src.wallclock import load_manifest


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def expected_shared_counts(config: dict, dataset: str) -> dict[str, int]:
    n_questions = int(config["datasets"][dataset]["n_sample"])
    nd_max = max(int(value) for value in config["nd"])
    budget = int(config["budget"])
    return {
        "draft": n_questions * nd_max,
        "prm": n_questions * nd_max,
        "logprob": n_questions * nd_max,
        "fullsc": n_questions * (budget - nd_max),
    }


def validate_shared_source(
    config: dict,
    dataset: str,
    project_root: Path,
) -> dict:
    shared_dir = resolve(
        project_root,
        config["shared_sources"][dataset],
    )
    checkpoint = shared_dir / "checkpoint.jsonl"
    validation_path = shared_dir / "validation_report.json"
    step_tokens = shared_dir / "draft_step_tokens.json"
    for path in (checkpoint, validation_path, step_tokens):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    validation = read_json(validation_path)
    expected = expected_shared_counts(config, dataset)
    if validation.get("status") != "passed":
        raise ValueError(f"Shared validation did not pass: {validation_path}")
    if validation.get("dataset") != dataset:
        raise ValueError(f"Shared dataset mismatch: {validation_path}")
    if validation.get("record_counts") != expected:
        raise ValueError(
            f"Shared counts={validation.get('record_counts')}, "
            f"expected={expected}"
        )
    return {
        "shared_dir": shared_dir,
        "checkpoint": checkpoint,
        "step_tokens": step_tokens,
        "validation": validation,
    }


def prepare_result_dir(
    config: dict,
    dataset: str,
    project_root: Path,
    result_dir: Path,
) -> dict:
    shared = validate_shared_source(config, dataset, project_root)
    result_dir.mkdir(parents=True, exist_ok=True)
    destination = result_dir / "draft_step_tokens.json"
    if not destination.is_file():
        shutil.copyfile(shared["step_tokens"], destination)
    expected_run_config = {
        "experiment_name": config["experiment_name"],
        "model": config["model"],
        "prm_model": config["prm_model"],
        "dataset": dataset,
        "n_questions": int(config["datasets"][dataset]["n_sample"]),
        "dataset_seed": 42,
        "generation_seed": int(config["seed"]),
        "budget": int(config["budget"]),
        "nd": [int(value) for value in config["nd"]],
        "signals": list(config["signals"]),
        "absolute_prm_threshold": float(
            config["absolute_prm_threshold"]
        ),
        "max_prm_drop_semantics": (
            "rollback at the first largest adjacent PRM score decrease"
        ),
        "absolute_threshold_semantics": (
            "rollback at the first step with PRM score below the threshold; "
            "fallback before the final step when no score is below it"
        ),
        "fair_semantics": (
            "nd original draft answers plus nd*(ns-1) rollback suffix answers"
        ),
        "fair_new_semantics": (
            "nd*ns rollback suffix answers and no original draft answers"
        ),
        "full_semantics": (
            "nd original draft answers plus nd*ns rollback suffix answers"
        ),
        "shared_checkpoint": str(shared["checkpoint"].resolve()),
        "pipeline": config["pipeline"],
    }
    run_config_path = result_dir / "run_config.json"
    if run_config_path.is_file():
        observed = read_json(run_config_path)
        if observed != expected_run_config:
            raise ValueError(
                f"Existing run config differs: {run_config_path}"
            )
    else:
        write_json(run_config_path, expected_run_config)
    return shared


def _suffix_identity(record: dict) -> tuple:
    return (
        record["doc_id"],
        int(record["draft_idx"]),
        record["strategy"],
        int(record["rollback_step"]),
        int(record["suffix_idx"]),
    )


def _sampling_task(record: dict) -> dict:
    return {
        "task_type": "suffix",
        "doc_id": record["doc_id"],
        "draft_idx": int(record["draft_idx"]),
        "strategy": record["strategy"],
        "rollback_step": int(record["rollback_step"]),
        "suffix_idx": int(record["suffix_idx"]),
    }


def _load_expected_rollbacks(
    shared_checkpoint: Path,
    threshold: float,
) -> tuple[dict[tuple[str, int, str], int], dict]:
    drafts = {}
    prm_records = {}
    for record in iter_jsonl(shared_checkpoint):
        key = (
            record.get("doc_id"),
            int(record.get("draft_idx", -1)),
        )
        if record.get("task_type") == "draft":
            drafts[key] = record
        elif record.get("task_type") == "prm":
            prm_records[key] = record
    if set(drafts) != set(prm_records):
        raise ValueError("Shared draft and PRM keys differ")

    rollbacks = {}
    summaries = {
        signal: {
            "assignments": 0,
            "triggered": 0,
            "fallback": 0,
            "rollback_step_histogram": Counter(),
        }
        for signal in PRM_CONTROL_SIGNALS
    }
    for key, draft in drafts.items():
        scores = prm_records[key]["step_scores"]
        n_steps = len(draft["draft_steps"])
        assignments = (
            max_prm_drop_assignment(scores, n_steps),
            prm_below_threshold_assignment(
                scores,
                n_steps,
                threshold,
            ),
        )
        for assignment in assignments:
            signal = str(assignment["signal"])
            rollback_step = int(assignment["rollback_step"])
            rollbacks[(key[0], key[1], signal)] = rollback_step
            summary = summaries[signal]
            summary["assignments"] += 1
            summary[
                "triggered" if assignment["triggered"] else "fallback"
            ] += 1
            summary["rollback_step_histogram"][rollback_step] += 1

    for summary in summaries.values():
        assignments = int(summary["assignments"])
        summary["trigger_rate"] = (
            float(summary["triggered"]) / assignments
            if assignments
            else 0.0
        )
        summary["rollback_step_histogram"] = {
            str(key): value
            for key, value in sorted(
                summary["rollback_step_histogram"].items()
            )
        }
    return rollbacks, summaries


def _require_fields(row: dict, expected: dict) -> None:
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(
                f"{row.get('method')} {key}={row.get(key)}, expected {value}"
            )


def validate_control_result(
    config: dict,
    dataset: str,
    project_root: Path,
    result_dir: Path,
    figure_dir: Path,
) -> dict:
    shared = validate_shared_source(config, dataset, project_root)
    local_checkpoint = result_dir / "checkpoint.jsonl"
    if not local_checkpoint.is_file():
        raise FileNotFoundError(local_checkpoint)
    nd_values = [int(value) for value in config["nd"]]
    budget = int(config["budget"])
    n_questions = int(config["datasets"][dataset]["n_sample"])
    per_draft = suffixes_per_draft(nd_values, budget)
    expected_per_signal = n_questions * sum(per_draft.values())
    expected_rollbacks, assignment_summary = _load_expected_rollbacks(
        shared["checkpoint"],
        float(config["absolute_prm_threshold"]),
    )

    counts = Counter()
    per_group: dict[tuple[str, str], Counter] = defaultdict(Counter)
    identities = set()
    sampling_seeds = set()
    for record in iter_jsonl(local_checkpoint):
        if record.get("task_type") != "suffix":
            raise ValueError(
                f"Unexpected local task type: {record.get('task_type')}"
            )
        signal = record.get("strategy")
        if signal not in config["signals"]:
            raise ValueError(f"Unexpected local strategy: {signal}")
        identity = _suffix_identity(record)
        if identity in identities:
            raise ValueError(f"Duplicate suffix identity: {identity}")
        identities.add(identity)
        expected_rollback = expected_rollbacks[
            (
                record["doc_id"],
                int(record["draft_idx"]),
                signal,
            )
        ]
        if int(record["rollback_step"]) != expected_rollback:
            raise ValueError(
                f"{identity} uses rollback {record['rollback_step']}, "
                f"expected {expected_rollback}"
            )
        expected_seed = derive_sampling_seed(
            int(config["seed"]),
            _sampling_task(record),
        )
        if int(record.get("sampling_seed", -1)) != expected_seed:
            raise ValueError(f"Unstable suffix sampling seed: {identity}")
        if expected_seed in sampling_seeds:
            raise ValueError(f"Suffix sampling seed collision: {identity}")
        sampling_seeds.add(expected_seed)
        counts[signal] += 1
        per_group[(record["doc_id"], signal)][
            int(record["draft_idx"])
        ] += 1

    expected_counts = {
        signal: expected_per_signal for signal in config["signals"]
    }
    if dict(counts) != expected_counts:
        raise ValueError(
            f"Suffix counts={dict(counts)}, expected={expected_counts}"
        )
    if len(per_group) != n_questions * len(config["signals"]):
        raise ValueError("Suffix question/signal coverage is incomplete")
    for group, observed in per_group.items():
        if dict(observed) != per_draft:
            raise ValueError(
                f"{group} per-draft suffixes={dict(observed)}, "
                f"expected={per_draft}"
            )

    summary_path = result_dir / "eval_summary.json"
    rows = read_json(summary_path)
    methods = {row["method"]: row for row in rows}
    required_baselines = {"Greedy@1", "SC@8", "SC@16", "SC@32"}
    if not required_baselines.issubset(methods):
        raise ValueError("Evaluation baselines are incomplete")
    if methods["SC@32"].get("n_answers") != budget:
        raise ValueError("SC@32 does not contain exactly 32 answers")
    evaluation_rows = 0
    for signal in config["signals"]:
        for nd in nd_values:
            ns = budget // nd
            expectations = (
                (
                    "",
                    {
                        "variant": "full",
                        "n_answers": budget + nd,
                        "n_draft_answers": nd,
                        "n_suffix_answers": budget,
                    },
                ),
                (
                    "_fair",
                    {
                        "variant": "fair",
                        "n_answers": budget,
                        "n_draft_answers": nd,
                        "n_suffix_answers": budget - nd,
                    },
                ),
                (
                    "_fair_new",
                    {
                        "variant": "fair_new",
                        "n_answers": budget,
                        "n_draft_answers": 0,
                        "n_suffix_answers": budget,
                    },
                ),
            )
            for suffix, expected in expectations:
                method = f"rollback_{signal}_nd{nd}_ns{ns}{suffix}"
                if method not in methods:
                    raise ValueError(f"Missing evaluation method: {method}")
                _require_fields(methods[method], expected)
                if not 0.0 <= float(methods[method]["acc"]) <= 1.0:
                    raise ValueError(f"Invalid accuracy: {method}")
                if dataset == "hotpotqa_open" and not 0.0 <= float(
                    methods[method]["f1"]
                ) <= 1.0:
                    raise ValueError(f"Invalid F1: {method}")
                evaluation_rows += 1
    if len(rows) != len(required_baselines) + evaluation_rows:
        raise ValueError(f"Unexpected evaluation row count: {len(rows)}")

    required_artifacts = (
        result_dir / "run_config.json",
        result_dir / "draft_step_tokens.json",
        result_dir / "eval_summary_table.md",
        figure_dir / "fig_efficiency_frontier.png",
        figure_dir / "fig_efficiency_frontier.pdf",
    )
    for path in required_artifacts:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing result artifact: {path}")

    timing_path = result_dir / "timing" / "timing_manifest.json"
    timing = load_manifest(timing_path)
    if timing.get("status") != "passed":
        raise ValueError(f"Timing did not pass: {timing_path}")
    write_json(
        result_dir / "rollback_assignment_summary.json",
        {
            "status": "passed",
            "dataset": dataset,
            "absolute_prm_threshold": float(
                config["absolute_prm_threshold"]
            ),
            "signals": assignment_summary,
        },
    )
    report = {
        "status": "passed",
        "dataset": dataset,
        "signals": list(config["signals"]),
        "absolute_prm_threshold": float(
            config["absolute_prm_threshold"]
        ),
        "shared_record_counts": expected_shared_counts(config, dataset),
        "local_suffix_counts": expected_counts,
        "unique_suffix_sampling_seeds": len(sampling_seeds),
        "evaluation_rows": evaluation_rows,
        "total_summary_rows": len(rows),
        "timing_run_id": timing["run_id"],
    }
    write_json(result_dir / "validation_report.json", report)
    return report


def all_datasets_complete(
    config: dict,
    results_root: Path,
    datasets: Iterable[str],
) -> bool:
    return all(
        (results_root / dataset / "COMPLETE").is_file()
        for dataset in datasets
    )
