"""Strict completeness and semantic audit for NLL/PRM control runs."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

from rebuttal.src.nll_prm_controls import (
    ALL_SIGNALS,
    NLL_SIGNAL,
    PRM_SIGNAL,
    compute_step_mean_nll,
    nll_rollback_assignment,
    prm_rollback_assignment,
    step_char_bounds,
)
from rebuttal.src.position_controls import additional_suffixes_per_draft
from rebuttal.src.stable_sampling import derive_sampling_seed


EXPECTED_CONFIGS = [(4, 8), (8, 4), (16, 2)]


def _require(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise AssertionError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source_drafts(
    path: Path,
    doc_ids: set[str],
    nd_max: int,
) -> Dict[Tuple[str, int], dict]:
    drafts = {}
    for record in _read_jsonl(path):
        if (
            record.get("task_type") != "draft"
            or record.get("doc_id") not in doc_ids
            or int(record["draft_idx"]) >= nd_max
        ):
            continue
        key = (record["doc_id"], int(record["draft_idx"]))
        _require(key not in drafts, f"Duplicate source draft: {key}")
        drafts[key] = record
    return drafts


def audit_nll_prm_control_results(
    result_dir: Path,
    expected_questions: int = 500,
    expected_model: str = "Qwen/Qwen2.5-3B-Instruct",
    expected_prm_model: str = "Qwen/Qwen2.5-Math-PRM-7B",
    expected_run_seed: int | None = None,
) -> dict:
    result_dir = Path(result_dir)
    required_paths = {
        "config": result_dir / "run_config.json",
        "checkpoint": result_dir / "checkpoint.jsonl",
        "assignments": result_dir / "rollback_assignments.jsonl",
        "summary": result_dir / "eval_summary.json",
        "table": result_dir / "eval_summary_table.md",
        "figure_png": result_dir / "nll_prm_controls.png",
        "figure_pdf": result_dir / "nll_prm_controls.pdf",
        "times": result_dir / "phase_times.json",
    }
    for name, path in required_paths.items():
        _require(
            path.is_file() and path.stat().st_size > 0,
            f"Missing or empty {name}: {path}",
        )

    config = json.loads(
        required_paths["config"].read_text(encoding="utf-8")
    )
    _require(config["model"] == expected_model, "Unexpected draft model")
    _require(
        config["prm_model"] == expected_prm_model,
        "Unexpected PRM model",
    )
    if expected_run_seed is not None:
        _require(
            config.get("generation_seed") == expected_run_seed,
            f"generation_seed must be {expected_run_seed}",
        )
        _require(
            config.get("source_random_seed") == expected_run_seed,
            f"source_random_seed must be {expected_run_seed}",
        )
    _require(
        int(config["n_questions"]) == expected_questions,
        "Unexpected question count in run config",
    )
    _require(
        [tuple(pair) for pair in config["rollback_configs"]]
        == EXPECTED_CONFIGS,
        "Unexpected rollback configs",
    )
    _require(config["signals"] == ALL_SIGNALS, "Unexpected signal list")
    _require(
        int(config["max_model_len"]) == 4096,
        "NLL/PRM run must use the same 4096 context limit",
    )
    _require(int(config["budget"]) == 32, "Unexpected answer budget")

    source_checkpoint = Path(config["source_checkpoint"])
    _require(source_checkpoint.is_file(), "Source checkpoint is missing")
    _require(
        _sha256(source_checkpoint)
        == config["source_checkpoint_sha256"],
        "Source checkpoint SHA256 changed",
    )

    shard_counts = {}
    for phase in ("prm", "nll", "suffix"):
        shard_dir = result_dir / "_shards" / phase
        shard_paths = sorted(shard_dir.glob("shard_*.jsonl"))
        if not shard_paths:
            continue
        shard_keys = []
        for shard_path in shard_paths:
            shard_keys.extend(
                record.get("task_key")
                for record in _read_jsonl(shard_path)
            )
        _require(
            all(shard_keys),
            f"{phase} shard record is missing task_key",
        )
        _require(
            len(shard_keys) == len(set(shard_keys)),
            f"{phase} shard task keys are not unique",
        )
        shard_counts[phase] = len(shard_keys)

    records = list(_read_jsonl(required_paths["checkpoint"]))
    task_keys = [record.get("task_key") for record in records]
    _require(all(task_keys), "A checkpoint record is missing task_key")
    _require(
        len(task_keys) == len(set(task_keys)),
        "Checkpoint task keys are not unique",
    )
    counts = Counter(record.get("task_type") for record in records)
    expected_per_score = expected_questions * 16
    per_draft = additional_suffixes_per_draft(EXPECTED_CONFIGS)
    expected_per_signal_suffix = expected_questions * sum(
        per_draft.values()
    )
    expected_counts = {
        "prm": expected_per_score,
        "nll": expected_per_score,
        "suffix": expected_per_signal_suffix * len(ALL_SIGNALS),
    }
    _require(
        dict(counts) == expected_counts,
        f"Unexpected checkpoint counts: {dict(counts)}",
    )

    doc_ids = {
        record["doc_id"]
        for record in records
        if record.get("task_type") in ("prm", "nll")
    }
    _require(
        len(doc_ids) == expected_questions,
        f"Unexpected scored document count: {len(doc_ids)}",
    )
    source_drafts = _load_source_drafts(
        source_checkpoint, doc_ids, nd_max=16
    )
    _require(
        len(source_drafts) == expected_per_score,
        f"Unexpected source draft count: {len(source_drafts)}",
    )
    if expected_run_seed is not None:
        source_sampling_seeds = set()
        for source in source_drafts.values():
            expected_seed = derive_sampling_seed(
                expected_run_seed, source
            )
            _require(
                source.get("sampling_seed") == expected_seed,
                f"Incorrect source draft sampling seed: "
                f"{source['task_key']}",
            )
            _require(
                expected_seed not in source_sampling_seeds,
                f"Duplicate source draft sampling seed: "
                f"{source['task_key']}",
            )
            source_sampling_seeds.add(expected_seed)

    score_map = {}
    suffix_counts = Counter()
    suffix_group_counts = Counter()
    suffix_group_rollbacks = defaultdict(set)
    suffix_sampling_seeds = set()
    for record in records:
        task_type = record["task_type"]
        if task_type in ("prm", "nll"):
            key = (
                record["doc_id"],
                int(record["draft_idx"]),
                task_type,
            )
            _require(key not in score_map, f"Duplicate score record: {key}")
            score_map[key] = record
            source = source_drafts[key[:2]]
            n_steps = len(source["draft_steps"])
            _require(
                int(record["n_steps"]) == n_steps,
                f"Score/source step mismatch: {key}",
            )
            _require(
                0 < int(record["input_tokens"]) <= 4096,
                f"Invalid input token count: {key}",
            )
            if task_type == "prm":
                _require(
                    len(record["step_scores"]) == n_steps,
                    f"PRM score count mismatch: {key}",
                )
            else:
                configured_backend = config.get("nll_backend", "vllm")
                if (
                    configured_backend != "vllm"
                    or "scoring_backend" in record
                ):
                    _require(
                        record.get("scoring_backend")
                        == configured_backend,
                        f"NLL backend mismatch: {key}",
                    )
                _require(
                    len(record["token_logprobs"])
                    == len(record["token_offsets"])
                    == int(record["response_scored_tokens"]),
                    f"NLL token score count mismatch: {key}",
                )
                _require(
                    record["response_scored_tokens"] > 0,
                    f"NLL response is unscored: {key}",
                )
        elif task_type == "suffix":
            signal = record["strategy"]
            _require(signal in ALL_SIGNALS, f"Unexpected signal: {signal}")
            _require(
                record.get("generation_kind") == "suffix",
                "Signal control must use suffix generation",
            )
            _require(
                0 <= int(record["suffix_tokens"]) <= 2048,
                f"Invalid suffix token count: {record['task_key']}",
            )
            _require(
                int(record["prefix_tokens"]) >= 0,
                f"Invalid prefix token count: {record['task_key']}",
            )
            suffix_counts[signal] += 1
            group = (
                record["doc_id"],
                int(record["draft_idx"]),
                signal,
            )
            suffix_group_counts[group] += 1
            suffix_group_rollbacks[group].add(
                int(record["rollback_step"])
            )
            if expected_run_seed is not None:
                expected_seed = derive_sampling_seed(
                    expected_run_seed, record
                )
                _require(
                    record.get("sampling_seed") == expected_seed,
                    f"Incorrect suffix sampling seed: "
                    f"{record['task_key']}",
                )
                _require(
                    expected_seed not in suffix_sampling_seeds,
                    f"Duplicate suffix sampling seed: "
                    f"{record['task_key']}",
                )
                suffix_sampling_seeds.add(expected_seed)

    _require(
        suffix_counts == Counter({
            signal: expected_per_signal_suffix for signal in ALL_SIGNALS
        }),
        f"Unexpected per-signal suffix counts: {dict(suffix_counts)}",
    )
    expected_groups = expected_questions * 16 * len(ALL_SIGNALS)
    _require(
        len(suffix_group_counts) == expected_groups,
        f"Unexpected suffix group count: {len(suffix_group_counts)}",
    )
    for group, count in suffix_group_counts.items():
        _require(
            count == per_draft[group[1]],
            f"Unexpected suffix allocation for {group}: {count}",
        )

    assignments = list(_read_jsonl(required_paths["assignments"]))
    expected_assignments = expected_per_score * len(ALL_SIGNALS)
    _require(
        len(assignments) == expected_assignments,
        f"Unexpected assignment count: {len(assignments)}",
    )
    assignment_keys = set()
    assignment_counts = Counter()
    trigger_counts = Counter()
    for assignment in assignments:
        key = (
            assignment["doc_id"],
            int(assignment["draft_idx"]),
            assignment["signal"],
        )
        _require(key not in assignment_keys, f"Duplicate assignment: {key}")
        assignment_keys.add(key)
        signal = assignment["signal"]
        assignment_counts[signal] += 1
        trigger_counts[signal] += int(assignment["triggered"])
        source = source_drafts[key[:2]]
        n_steps = len(source["draft_steps"])
        _require(
            int(assignment["n_steps"]) == n_steps,
            f"Assignment/source step mismatch: {key}",
        )
        if signal == PRM_SIGNAL:
            expected = prm_rollback_assignment(
                score_map[(*key[:2], "prm")]["step_scores"],
                n_steps,
                float(config["thresholds"][signal]),
            )
        elif signal == NLL_SIGNAL:
            nll_record = score_map[(*key[:2], "nll")]
            step_nlls = compute_step_mean_nll(
                nll_record["token_logprobs"],
                nll_record["token_offsets"],
                step_char_bounds(
                    source["draft_text"], source["draft_steps"]
                ),
            )
            expected = nll_rollback_assignment(
                step_nlls,
                n_steps,
                float(config["thresholds"][signal]),
            )
        else:
            raise AssertionError(f"Unexpected assignment signal: {signal}")
        _require(
            int(assignment["rollback_step"])
            == int(expected["rollback_step"]),
            f"Incorrect rollback assignment: {key}",
        )
        _require(
            bool(assignment["triggered"]) == bool(expected["triggered"]),
            f"Incorrect trigger flag: {key}",
        )
        _require(
            suffix_group_rollbacks[key]
            == {int(expected["rollback_step"])},
            f"Suffix rollback disagrees with assignment: {key}",
        )
        _require(
            int(assignment["target_step"])
            == int(expected["rollback_step"]) + 1,
            f"Incorrect target step: {key}",
        )
        _require(
            math.isclose(
                float(assignment["target_fraction"]),
                (int(expected["rollback_step"]) + 1) / n_steps,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"Incorrect target fraction: {key}",
        )

    _require(
        assignment_counts == Counter({
            signal: expected_per_score for signal in ALL_SIGNALS
        }),
        f"Unexpected assignment distribution: {dict(assignment_counts)}",
    )

    results = json.loads(
        required_paths["summary"].read_text(encoding="utf-8")
    )
    _require(len(results) == 7, f"Unexpected result rows: {len(results)}")
    _require(results[0]["method"] == "Greedy@1", "Greedy row must be first")
    result_map = {record["method"]: record for record in results}
    _require(
        len(result_map) == len(results),
        "Result method names are not unique",
    )
    for signal in ALL_SIGNALS:
        for nd, ns in EXPECTED_CONFIGS:
            method = f"{signal}_nd{nd}_ns{ns}"
            _require(method in result_map, f"Missing result: {method}")
            row = result_map[method]
            _require(
                row["n_answers"] == 32,
                f"{method} does not use 32 total answers",
            )
            _require(
                row["variant"] == "draft_plus_suffix"
                and row["n_draft_answers"] == nd
                and row["n_suffix_answers"] == nd * (ns - 1),
                f"{method} does not use the shared draft+suffix budget",
            )
            _require(
                0.0 <= row["acc"] <= 1.0,
                f"{method} accuracy is invalid",
            )
            _require(
                row["tokens_per_q"] > 0
                and row["signal_tokens_per_q"] > 0,
                f"{method} token costs are invalid",
            )
            _require(
                math.isclose(
                    row["tokens_per_q_with_signal"],
                    row["tokens_per_q"] + row["signal_tokens_per_q"],
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ),
                f"{method} combined token cost is inconsistent",
            )
            if config["dataset"] in ("hotpotqa", "hotpotqa_open"):
                _require(
                    0.0 <= row["f1"] <= 1.0,
                    f"{method} F1 is invalid",
                )

    phase_times = json.loads(
        required_paths["times"].read_text(encoding="utf-8")
    )
    for phase in ("prm", "nll", "suffix"):
        _require(
            phase_times.get(phase, 0) > 0,
            f"Missing positive phase time: {phase}",
        )

    return {
        "status": "passed",
        "dataset": config["dataset"],
        "model": config["model"],
        "prm_model": config["prm_model"],
        "n_questions": expected_questions,
        "source_checkpoint_sha256": config[
            "source_checkpoint_sha256"
        ],
        "checkpoint_records": len(records),
        "record_counts": expected_counts,
        "shard_counts": shard_counts,
        "suffix_records_per_signal": dict(suffix_counts),
        "rollback_assignments": len(assignments),
        "trigger_counts": dict(trigger_counts),
        "result_rows": len(results),
        "phase_times_seconds": phase_times,
    }
