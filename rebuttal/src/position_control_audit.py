"""Completeness checks for position-control experiment artifacts."""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from rebuttal.src.position_controls import (
    ALL_STRATEGIES,
    FINAL_STEP_STRATEGY,
    FIXED_FRACTIONS,
    final_step_rollback_step,
    fixed_rollback_step,
    random_rollback_step,
)
from rebuttal.src.stable_sampling import derive_sampling_seed


EXPECTED_CONFIGS = [(4, 8), (8, 4), (16, 2)]


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise AssertionError(
                    f"{path}:{line_number} is not valid JSON: {error}"
                ) from error


def _require(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def audit_position_control_results(
    result_dir: Path,
    expected_questions: int = 500,
    expected_model: str = "Qwen/Qwen2.5-3B-Instruct",
    expected_run_seed: int | None = None,
) -> dict:
    result_dir = Path(result_dir)
    config_path = result_dir / "run_config.json"
    checkpoint_path = result_dir / "checkpoint.jsonl"
    assignments_path = result_dir / "rollback_assignments.jsonl"
    summary_path = result_dir / "eval_summary.json"
    phase_times_path = result_dir / "phase_times.json"

    for path in (
        config_path,
        checkpoint_path,
        assignments_path,
        summary_path,
        phase_times_path,
        result_dir / "eval_summary_table.md",
        result_dir / "position_controls.png",
        result_dir / "position_controls.pdf",
    ):
        _require(path.is_file() and path.stat().st_size > 0,
                 f"Missing or empty artifact: {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    _require(config["model"] == expected_model,
             f"Unexpected model: {config['model']}")
    _require(config["n_questions"] == expected_questions,
             f"Unexpected question count: {config['n_questions']}")
    _require(config["dataset_seed"] == 42, "dataset_seed must be 42")
    if expected_run_seed is not None:
        _require(
            config["random_seed"] == expected_run_seed,
            f"random_seed must be {expected_run_seed}",
        )
        _require(
            config.get("generation_seed") == expected_run_seed,
            f"generation_seed must be {expected_run_seed}",
        )
    _require(config["budget"] == 32, "budget must be 32")
    _require(
        [tuple(item) for item in config["rollback_configs"]]
        == EXPECTED_CONFIGS,
        f"Unexpected rollback configs: {config['rollback_configs']}",
    )
    strategies = config["strategies"]
    _require(
        strategies
        and len(strategies) == len(set(strategies))
        and set(strategies).issubset(ALL_STRATEGIES),
        f"Unexpected strategies: {strategies}",
    )
    _require(
        config["fixed_position_semantics"]
        == "target step = ceil(fraction * n_steps); retain earlier steps",
        "Unexpected fixed-position semantics",
    )
    _require(config["temperature"] == 0.7,
             "temperature must be 0.7")
    _require(config["top_p"] == 0.95, "top_p must be 0.95")
    _require(config["max_tokens"] == 2048,
             "position max_tokens must be 2048")
    _require(
        config["final_step_semantics"]
        == "target the final split reasoning step; retain earlier steps",
        "Unexpected final-step semantics",
    )

    expected_drafts = expected_questions * max(
        nd for nd, _ in EXPECTED_CONFIGS
    )
    draft_plus_suffix = (
        config.get("answer_budget_semantics")
        == "nd original draft answers plus nd*(ns-1) suffix answers"
    )
    suffixes_per_question = 48 if draft_plus_suffix else 64
    expected_per_strategy = expected_questions * suffixes_per_question
    expected_suffixes = expected_per_strategy * len(strategies)

    task_keys = set()
    draft_counts = Counter()
    draft_steps = {}
    suffix_counts = Counter()
    suffix_group_counts = Counter()
    suffix_group_positions = defaultdict(set)
    generation_counts = Counter()
    sampling_seeds = set()
    sampling_seeds_checked = 0
    checkpoint_records = 0
    for record in _read_jsonl(checkpoint_path):
        checkpoint_records += 1
        task_key = record.get("task_key")
        _require(task_key, "Checkpoint record is missing task_key")
        _require(task_key not in task_keys,
                 f"Duplicate task_key: {task_key}")
        task_keys.add(task_key)
        task_type = record.get("task_type")
        record_seed = None
        if expected_run_seed is not None:
            record_seed = expected_run_seed
        elif (
            config.get("generation_seed") is not None
            and task_type == "suffix"
            and record.get("strategy") == FINAL_STEP_STRATEGY
        ):
            record_seed = int(config["generation_seed"])
        if record_seed is not None:
            seed_task = dict(record)
            expected_sampling_seed = derive_sampling_seed(
                record_seed, seed_task
            )
            _require(
                record.get("sampling_seed") == expected_sampling_seed,
                f"Unexpected sampling_seed for {task_key}",
            )
            _require(
                expected_sampling_seed not in sampling_seeds,
                f"Duplicate sampling_seed for {task_key}",
            )
            sampling_seeds.add(expected_sampling_seed)
            sampling_seeds_checked += 1
        if task_type == "draft":
            draft_counts[record["doc_id"]] += 1
            n_steps = record["n_steps"]
            _require(n_steps >= 1,
                     f"Draft must contain at least one step: {task_key}")
            draft_steps[(record["doc_id"], record["draft_idx"])] = n_steps
        elif task_type == "suffix":
            strategy = record["strategy"]
            _require(strategy in strategies,
                     f"Unexpected strategy: {strategy}")
            suffix_counts[strategy] += 1
            group_key = (
                record["doc_id"],
                record["draft_idx"],
                strategy,
            )
            suffix_group_counts[group_key] += 1
            suffix_group_positions[group_key].add(
                record["rollback_step"]
            )
            generation_kind = record["generation_kind"]
            generation_counts[(strategy, generation_kind)] += 1
            if strategy == FINAL_STEP_STRATEGY:
                _require(generation_kind == "suffix",
                         "final_step_only must use suffix generation")
                _require(record["suffix_tokens"] <= 2048,
                         f"Final step exceeds 2048 tokens: {task_key}")
                _require(record.get("random_seed") is None,
                         "final_step_only must not carry random_seed")
            else:
                _require(generation_kind == "suffix",
                         f"{strategy} must use suffix generation")
                _require(record["suffix_tokens"] <= 2048,
                         f"Suffix exceeds 2048 tokens: {task_key}")
                expected_seed = (
                    config["random_seed"]
                    if strategy == "random_uniform" else None
                )
                _require(record.get("random_seed") == expected_seed,
                         f"Unexpected random_seed for {task_key}")
        else:
            raise AssertionError(f"Unexpected task_type: {task_type}")

    _require(checkpoint_records == expected_drafts + expected_suffixes,
             f"Unexpected checkpoint size: {checkpoint_records}")
    _require(len(draft_counts) == expected_questions,
             f"Unexpected draft document count: {len(draft_counts)}")
    _require(set(draft_counts.values()) == {16},
             f"Each document must have 16 drafts: {Counter(draft_counts.values())}")
    _require(
        suffix_counts == Counter({
            strategy: expected_per_strategy
            for strategy in strategies
        }),
        f"Unexpected suffix counts: {dict(suffix_counts)}",
    )
    expected_generation_counts = Counter({
        (strategy, "suffix"): expected_per_strategy
        for strategy in strategies
    })
    _require(
        generation_counts == expected_generation_counts,
        f"Unexpected generation-kind counts: {dict(generation_counts)}",
    )

    expected_suffixes_by_draft = {
        draft_idx: (
            7 if draft_idx < 4 else 3 if draft_idx < 8 else 1
        ) if draft_plus_suffix else (
            8 if draft_idx < 4 else 4 if draft_idx < 8 else 2
        )
        for draft_idx in range(16)
    }
    expected_groups = expected_questions * 16 * len(strategies)
    _require(len(suffix_group_counts) == expected_groups,
             f"Unexpected suffix group count: {len(suffix_group_counts)}")
    for (doc_id, draft_idx, strategy), count in suffix_group_counts.items():
        expected = expected_suffixes_by_draft[draft_idx]
        _require(
            count == expected,
            f"{doc_id}/{draft_idx}/{strategy}: {count} != {expected}",
        )

    assignment_counts = Counter()
    assignment_keys = set()
    assignment_records = 0
    for record in _read_jsonl(assignments_path):
        assignment_records += 1
        key = (
            record["doc_id"],
            record["draft_idx"],
            record["strategy"],
        )
        _require(key not in assignment_keys,
                 f"Duplicate rollback assignment: {key}")
        assignment_keys.add(key)
        strategy = record["strategy"]
        assignment_counts[strategy] += 1
        n_steps = record["n_steps"]
        _require(
            draft_steps[(record["doc_id"], record["draft_idx"])]
            == n_steps,
            f"Assignment n_steps does not match draft: {key}",
        )
        if strategy == "random_uniform":
            expected_rollback = random_rollback_step(
                record["doc_id"],
                record["draft_idx"],
                n_steps,
                int(config["random_seed"]),
            )
        elif strategy in FIXED_FRACTIONS:
            expected_rollback = fixed_rollback_step(
                n_steps, FIXED_FRACTIONS[strategy]
            )
        elif strategy == FINAL_STEP_STRATEGY:
            expected_rollback = final_step_rollback_step(n_steps)
        else:
            raise AssertionError(
                f"Unexpected assignment strategy: {strategy}"
            )
        _require(
            record["rollback_step"] == expected_rollback,
            f"Unexpected rollback step for {key}: "
            f"{record['rollback_step']} != {expected_rollback}",
        )
        _require(
            suffix_group_positions[key] == {expected_rollback},
            f"Suffix rollback steps disagree with assignment: {key}",
        )
        if strategy == FINAL_STEP_STRATEGY:
            _require(record["target_step"] == n_steps,
                     f"Final-step target_step must be last: {key}")
            _require(record["target_fraction"] == 1.0,
                     f"Final-step target_fraction must be 1: {key}")
        else:
            _require(record["target_step"] == expected_rollback + 1,
                     f"Unexpected target_step: {key}")
            expected_fraction = (expected_rollback + 1) / n_steps
            _require(
                math.isclose(
                    record["target_fraction"],
                    expected_fraction,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                f"Unexpected target_fraction: {key}",
            )
    expected_assignments = expected_questions * 16 * len(strategies)
    _require(assignment_records == expected_assignments,
             f"Unexpected assignment count: {assignment_records}")
    _require(
        assignment_counts == Counter({
            strategy: expected_questions * 16
            for strategy in strategies
        }),
        f"Unexpected assignment distribution: {dict(assignment_counts)}",
    )

    results = json.loads(summary_path.read_text(encoding="utf-8"))
    _require(len(results) == 1 + len(strategies) * 3,
             f"Unexpected result row count: {len(results)}")
    _require(results[0]["method"] == "Greedy@1",
             "First result row must be Greedy@1")
    result_methods = {record["method"] for record in results}
    _require(len(result_methods) == len(results),
             "Result methods must be unique")
    for strategy in strategies:
        for nd, ns in EXPECTED_CONFIGS:
            method = f"{strategy}_nd{nd}_ns{ns}"
            row = next(
                (record for record in results
                 if record["method"] == method),
                None,
            )
            _require(row is not None, f"Missing result row: {method}")
            _require(row["n_answers"] == 32,
                     f"{method} must contain 32 answers/question")
            if draft_plus_suffix:
                _require(
                    row["variant"] == "draft_plus_suffix"
                    and row["n_draft_answers"] == nd
                    and row["n_suffix_answers"] == nd * (ns - 1),
                    f"{method} has an invalid answer-budget composition",
                )
            _require(0.0 <= row["acc"] <= 1.0,
                     f"{method} accuracy is out of range")
            _require(row["tokens_per_q"] > 0,
                     f"{method} tokens_per_q must be positive")
            if dataset in ("hotpotqa", "hotpotqa_open"):
                _require(0.0 <= row["f1"] <= 1.0,
                         f"{method} F1 is out of range")

    phase_times = json.loads(phase_times_path.read_text(encoding="utf-8"))
    _require(phase_times.get("draft", 0) > 0,
             "Missing positive draft phase time")
    _require(phase_times.get("suffix", 0) > 0,
             "Missing positive suffix phase time")

    return {
        "status": "passed",
        "dataset": dataset,
        "model": config["model"],
        "n_questions": expected_questions,
        "checkpoint_records": checkpoint_records,
        "draft_records": expected_drafts,
        "suffix_records": expected_suffixes,
        "suffix_records_per_strategy": dict(suffix_counts),
        "rollback_assignments": assignment_records,
        "result_rows": len(results),
        "sampling_seeds_checked": sampling_seeds_checked,
        "phase_times_seconds": phase_times,
    }
