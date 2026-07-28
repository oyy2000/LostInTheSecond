"""Reusable preparation, reuse, validation, and summary logic for 6_17."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator

from rebuttal.src.nll_prm_controls import (
    NLL_SIGNAL,
    PRM_SIGNAL,
    compute_step_mean_nll,
    nll_rollback_assignment,
    prm_rollback_assignment,
    step_char_bounds,
)
from rebuttal.src.stable_sampling import derive_sampling_seed
from rebuttal.src.wallclock import discover_timing_manifest, load_manifest


GENERATION_TYPES = {"draft", "suffix", "fullsc"}
SCORING_TYPES = {"prm", "logprob"}
SIGNAL_CHECKPOINT_LAYOUT = "shared_plus_signal_delta_v1"


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def validate_timing_artifact(
    result_dir: Path,
    *,
    require_evaluation: bool,
) -> dict:
    path = result_dir / "timing" / "timing_manifest.json"
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing timing manifest: {path}")
    manifest = load_manifest(path)
    if manifest.get("status") != "passed":
        raise ValueError(f"Timing manifest did not pass: {path}")
    phases = {
        phase["phase"]: phase for phase in manifest.get("phases", [])
    }
    required_phases = {"draft", "prm", "nll", "suffix", "fullsc"}
    if require_evaluation:
        required_phases.add("evaluation")
    missing = required_phases - set(phases)
    if missing:
        raise ValueError(f"Timing phases missing in {path}: {missing}")
    for phase_name in required_phases:
        phase = phases[phase_name]
        required = int(phase["tasks_required"])
        reused = int(phase["tasks_reused"])
        executed = int(phase["tasks_executed"])
        if reused + executed != required:
            raise ValueError(
                f"{path} {phase_name}: reused={reused} plus "
                f"executed={executed} does not equal required={required}"
            )
        if float(phase["observed_current_wall_sec"]) < 0:
            raise ValueError(f"Negative wall-clock in {path}")
    return {
        "run_id": manifest["run_id"],
        "mode": manifest["mode"],
        "observed_current_end_to_end_wall_sec": manifest[
            "observed_current_end_to_end_wall_sec"
        ],
        "reconstructed_from_scratch_wall_sec": manifest[
            "reconstructed_from_scratch_wall_sec"
        ],
        "reconstruction_complete": manifest[
            "reconstruction_complete"
        ],
    }


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def threshold_slug(value: float) -> str:
    return format(float(value), "g").replace("-", "m").replace(".", "p")


def trial_name(nll_threshold: float, prm_threshold: float) -> str:
    return (
        f"nll_{threshold_slug(nll_threshold)}"
        f"_prm_{threshold_slug(prm_threshold)}"
    )


def active_threshold_columns(
    strategy: str,
    active_threshold: float,
) -> tuple[float | None, float | None]:
    """Return only the threshold that affects the selected signal."""
    if strategy == NLL_SIGNAL:
        return float(active_threshold), None
    if strategy == PRM_SIGNAL:
        return None, float(active_threshold)
    raise ValueError(f"Unsupported threshold-sweep strategy: {strategy}")


def build_eval_summary_table(dataset: str, rows: list[dict]) -> str:
    """Build one threshold-annotated table for a complete dataset sweep."""
    sc32 = next(
        (row for row in rows if row["method"] == "SC@32"),
        None,
    )
    if sc32 is None:
        raise ValueError("Consolidated threshold table requires SC@32")
    sc32_tokens = float(sc32["tokens_per_q"])
    report_f1 = dataset == "hotpotqa_open"
    table = [
        "| method | signal | NLL threshold | PRM threshold | "
        "variant | nd | ns | acc | "
        + ("F1 | " if report_f1 else "")
        + "tokens/q | savings vs SC@32 | answers |",
        "|---|---|---:|---:|---|---:|---:|---:|"
        + ("---:|" if report_f1 else "")
        + "---:|---:|---:|",
    ]
    for row in rows:
        nll_threshold = row.get("nll_threshold")
        prm_threshold = row.get("prm_threshold")
        nll_text = (
            f"{float(nll_threshold):.2f}"
            if nll_threshold is not None else "-"
        )
        prm_text = (
            f"{float(prm_threshold):.2f}"
            if prm_threshold is not None else "-"
        )
        savings = (
            1.0 - float(row["tokens_per_q"]) / sc32_tokens
        ) * 100.0
        table.append(
            f"| {row['method']} | {row.get('strategy', '')} "
            f"| {nll_text} | {prm_text} "
            f"| {row.get('variant', '')} "
            f"| {row.get('nd', '')} | {row.get('ns', '')} "
            f"| {float(row['acc']):.4f} "
            + (
                f"| {float(row['f1']):.4f} "
                if report_f1 else ""
            )
            + f"| {float(row['tokens_per_q']):.1f} "
            f"| {savings:.1f}% "
            f"| {float(row['n_answers']):.0f} |"
        )
    return "\n".join(table) + "\n"


def checkpoint_identity(record: dict) -> tuple:
    task_type = record["task_type"]
    if task_type in {"draft", "prm", "logprob"}:
        return task_type, record["doc_id"], int(record["draft_idx"])
    if task_type == "suffix":
        return (
            task_type,
            record["doc_id"],
            int(record["draft_idx"]),
            record["strategy"],
            int(record["rollback_step"]),
            int(record["suffix_idx"]),
        )
    if task_type == "fullsc":
        return task_type, record["doc_id"], int(record["sc_idx"])
    raise ValueError(f"Unsupported checkpoint task type: {task_type!r}")


def sampling_task(record: dict) -> dict:
    task_type = record["task_type"]
    task = {
        "task_type": task_type,
        "doc_id": record["doc_id"],
    }
    if task_type == "draft":
        task["draft_idx"] = int(record["draft_idx"])
    elif task_type == "fullsc":
        task["sc_idx"] = int(record["sc_idx"])
    elif task_type == "suffix":
        task.update(
            draft_idx=int(record["draft_idx"]),
            strategy=record["strategy"],
            rollback_step=int(record["rollback_step"]),
            suffix_idx=int(record["suffix_idx"]),
        )
    else:
        raise ValueError(f"Not a generation record: {task_type!r}")
    return task


def require_stable_sampling(record: dict, run_seed: int) -> None:
    expected = derive_sampling_seed(run_seed, sampling_task(record))
    if record.get("sampling_seed") != expected:
        raise ValueError(
            f"{checkpoint_identity(record)} has sampling_seed="
            f"{record.get('sampling_seed')}, expected {expected}"
        )


def suffixes_per_draft(nd_values: Iterable[int], budget: int) -> dict[int, int]:
    values = sorted({int(value) for value in nd_values})
    nd_max = max(values)
    return {
        draft_idx: max(
            budget // nd for nd in values if draft_idx < nd
        )
        for draft_idx in range(nd_max)
    }


def _validate_source_run(
    run_config_path: Path,
    *,
    dataset: str,
    model: str,
    run_seed: int,
    draft_sha256: str | None = None,
) -> dict:
    run_config = read_json(run_config_path)
    if run_config.get("dataset") != dataset:
        raise ValueError(f"Dataset mismatch in {run_config_path}")
    if run_config.get("model") != model:
        raise ValueError(f"Model mismatch in {run_config_path}")
    if run_config.get("generation_seed") != run_seed:
        raise ValueError(
            f"{run_config_path} is not generation_seed={run_seed}"
        )
    if (
        draft_sha256 is not None
        and run_config.get("source_checkpoint_sha256") != draft_sha256
    ):
        raise ValueError(
            f"{run_config_path} uses a different draft checkpoint"
        )
    return run_config


def _check_vllm_evidence(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    markers = ("Initializing an LLM engine", "[NLL shard")
    if not all(marker in text for marker in markers):
        raise ValueError(f"NLL backend evidence is not vLLM: {path}")


def prepare_shared_checkpoint(
    config: dict,
    dataset: str,
    project_root: Path,
    shared_dir: Path,
) -> dict:
    """Seed a shared checkpoint with stable drafts and vLLM scoring records."""
    shared_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = shared_dir / "checkpoint.jsonl"
    manifest_path = shared_dir / "reuse_manifest.json"
    if checkpoint.exists() and manifest_path.exists():
        return read_json(manifest_path)

    source = config["stable_sources"][dataset]
    draft_checkpoint = project_root / source["draft_checkpoint"]
    draft_run_config = project_root / source["draft_run_config"]
    scoring_checkpoint = project_root / source["scoring_checkpoint"]
    scoring_run_config = project_root / source["scoring_run_config"]
    nll_backend_evidence = project_root / source["nll_backend_evidence"]
    for path in (
        draft_checkpoint,
        draft_run_config,
        scoring_checkpoint,
        scoring_run_config,
        nll_backend_evidence,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    run_seed = int(config["seed"])
    model = config["model"]
    n_questions = int(config["datasets"][dataset]["n_sample"])
    nd_max = max(int(value) for value in config["nd"])
    expected_per_type = n_questions * nd_max
    draft_sha256 = sha256_file(draft_checkpoint)
    _validate_source_run(
        draft_run_config,
        dataset=dataset,
        model=model,
        run_seed=run_seed,
    )
    _validate_source_run(
        scoring_run_config,
        dataset=dataset,
        model=model,
        run_seed=run_seed,
        draft_sha256=draft_sha256,
    )
    _check_vllm_evidence(nll_backend_evidence)

    temp_path = shared_dir / "checkpoint.jsonl.tmp"
    identities: set[tuple] = set()
    counts: Counter = Counter()
    generation_seeds: set[int] = set()
    with temp_path.open("w", encoding="utf-8") as output:
        for record in iter_jsonl(draft_checkpoint):
            if record.get("task_type") != "draft":
                continue
            require_stable_sampling(record, run_seed)
            identity = checkpoint_identity(record)
            if identity in identities:
                raise ValueError(f"Duplicate stable draft: {identity}")
            identities.add(identity)
            generation_seeds.add(int(record["sampling_seed"]))
            counts["draft"] += 1
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

        for source_record in iter_jsonl(scoring_checkpoint):
            source_type = source_record.get("task_type")
            if source_type not in {"prm", "nll"}:
                continue
            record = dict(source_record)
            if source_type == "nll":
                record["task_type"] = "logprob"
                record["scoring_backend"] = "vllm"
            else:
                record["prm_tokens"] = int(
                    record.get("prm_tokens", record.get("input_tokens", 0))
                )
            identity = checkpoint_identity(record)
            if identity in identities:
                raise ValueError(f"Duplicate scoring record: {identity}")
            identities.add(identity)
            counts[record["task_type"]] += 1
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    expected_counts = {
        "draft": expected_per_type,
        "prm": expected_per_type,
        "logprob": expected_per_type,
    }
    if dict(counts) != expected_counts:
        temp_path.unlink(missing_ok=True)
        raise ValueError(
            f"Shared source counts={dict(counts)}, expected={expected_counts}"
        )
    if len(generation_seeds) != expected_per_type:
        temp_path.unlink(missing_ok=True)
        raise ValueError("Stable draft sampling seeds are not unique")
    temp_path.replace(checkpoint)

    rejected = []
    for value in source.get("rejected_sources", []):
        path = project_root / value
        reason = "missing run_config.json"
        run_path = path / "run_config.json"
        if run_path.is_file():
            legacy = read_json(run_path)
            if legacy.get("generation_seed") != run_seed:
                reason = (
                    f"generation_seed={legacy.get('generation_seed')}; "
                    f"required {run_seed}"
                )
            else:
                reason = "not selected by the stable-source precedence"
        rejected.append({"path": str(path), "reason": reason})

    manifest = {
        "status": "prepared",
        "dataset": dataset,
        "model": model,
        "generation_seed": run_seed,
        "nll_backend": "vllm",
        "draft_checkpoint": str(draft_checkpoint),
        "draft_checkpoint_sha256": draft_sha256,
        "draft_timing_manifest": (
            str(discover_timing_manifest(draft_checkpoint))
            if discover_timing_manifest(draft_checkpoint) is not None
            else None
        ),
        "scoring_checkpoint": str(scoring_checkpoint),
        "scoring_checkpoint_sha256": sha256_file(scoring_checkpoint),
        "scoring_timing_manifest": (
            str(discover_timing_manifest(scoring_checkpoint))
            if discover_timing_manifest(scoring_checkpoint) is not None
            else None
        ),
        "nll_backend_evidence": str(nll_backend_evidence),
        "record_counts": expected_counts,
        "rejected_sources": rejected,
    }
    write_json(manifest_path, manifest)
    return manifest


def write_fresh_shared_manifest(
    config: dict,
    dataset: str,
    shared_dir: Path,
) -> dict:
    """Record provenance for a shared checkpoint generated from scratch."""
    checkpoint = shared_dir / "checkpoint.jsonl"
    validation_path = shared_dir / "validation_report.json"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not validation_path.is_file():
        raise FileNotFoundError(validation_path)
    validation = read_json(validation_path)
    checkpoint_hash = sha256_file(checkpoint)
    timing_manifest = discover_timing_manifest(checkpoint)
    manifest = {
        "status": "prepared",
        "source_mode": "fresh_generation",
        "dataset": dataset,
        "model": config["model"],
        "generation_seed": int(config["seed"]),
        "nll_backend": "vllm",
        "draft_checkpoint": str(checkpoint.resolve()),
        "draft_checkpoint_sha256": checkpoint_hash,
        "scoring_checkpoint": str(checkpoint.resolve()),
        "scoring_checkpoint_sha256": checkpoint_hash,
        "shared_checkpoint_sha256": checkpoint_hash,
        "timing_manifest": (
            str(timing_manifest) if timing_manifest is not None else None
        ),
        "record_counts": validation["record_counts"],
        "rejected_sources": [],
    }
    write_json(shared_dir / "reuse_manifest.json", manifest)
    return manifest


def _load_rollbacks(
    checkpoint: Path,
    nll_threshold: float,
    prm_threshold: float,
) -> dict[tuple[str, int, str], int]:
    drafts = {}
    prm_records = {}
    logprob_records = {}
    for record in iter_jsonl(checkpoint):
        key = (record.get("doc_id"), int(record.get("draft_idx", -1)))
        if record.get("task_type") == "draft":
            drafts[key] = record
        elif record.get("task_type") == "prm":
            prm_records[key] = record
        elif record.get("task_type") == "logprob":
            logprob_records[key] = record

    rollbacks = {}
    for key, draft in drafts.items():
        steps = draft["draft_steps"]
        n_steps = len(steps)
        prm = prm_rollback_assignment(
            prm_records[key]["step_scores"], n_steps, prm_threshold
        )
        bounds = step_char_bounds(draft["draft_text"], steps)
        logprob = logprob_records[key]
        nlls = compute_step_mean_nll(
            logprob["token_logprobs"],
            logprob["token_offsets"],
            bounds,
        )
        nll = nll_rollback_assignment(nlls, n_steps, nll_threshold)
        rollbacks[(key[0], key[1], PRM_SIGNAL)] = int(
            prm["rollback_step"]
        )
        rollbacks[(key[0], key[1], NLL_SIGNAL)] = int(
            nll["rollback_step"]
        )
    return rollbacks


def _source_threshold(run_config: dict, signal: str) -> float | None:
    thresholds = run_config.get("thresholds", {})
    value = thresholds.get(signal)
    return None if value is None else float(value)


def signal_threshold_name(signal: str, threshold: float) -> str:
    if signal == NLL_SIGNAL:
        prefix = "nll"
    elif signal == PRM_SIGNAL:
        prefix = "prm"
    else:
        raise ValueError(f"Unsupported threshold-sweep signal: {signal}")
    return f"{prefix}_{threshold_slug(threshold)}"


def thresholds_for_signal(config: dict, signal: str) -> list[float]:
    if signal == NLL_SIGNAL:
        values = config["nll_thresholds"]
    elif signal == PRM_SIGNAL:
        values = config["prm_thresholds"]
    else:
        raise ValueError(f"Unsupported threshold-sweep signal: {signal}")
    return [float(value) for value in values]


def signal_result_dir(
    dataset_root: Path,
    signal: str,
    threshold: float,
) -> Path:
    return (
        dataset_root
        / "signals"
        / signal_threshold_name(signal, threshold)
    )


def _rollback_threshold_pair(
    config: dict,
    signal: str,
    threshold: float,
) -> tuple[float, float]:
    nll_threshold = float(config["nll_thresholds"][0])
    prm_threshold = float(config["prm_thresholds"][0])
    if signal == NLL_SIGNAL:
        nll_threshold = float(threshold)
    elif signal == PRM_SIGNAL:
        prm_threshold = float(threshold)
    else:
        raise ValueError(f"Unsupported threshold-sweep signal: {signal}")
    return nll_threshold, prm_threshold


def _legacy_trial_source(
    config: dict,
    dataset_root: Path,
    signal: str,
    threshold: float,
) -> Path:
    nll_threshold, prm_threshold = _rollback_threshold_pair(
        config, signal, threshold
    )
    return dataset_root / trial_name(nll_threshold, prm_threshold)


def prepare_signal_checkpoint(
    config: dict,
    dataset: str,
    project_root: Path,
    dataset_root: Path,
    shared_dir: Path,
    signal: str,
    threshold: float,
) -> dict:
    """Prepare one suffix-only delta checkpoint for a signal threshold."""
    threshold = float(threshold)
    result_dir = signal_result_dir(dataset_root, signal, threshold)
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = result_dir / "checkpoint.jsonl"
    run_config_path = result_dir / "run_config.json"
    reuse_report_path = result_dir / "reuse_report.json"
    shared_checkpoint = shared_dir / "checkpoint.jsonl"
    shared_manifest = read_json(shared_dir / "reuse_manifest.json")
    if not shared_checkpoint.is_file():
        raise FileNotFoundError(shared_checkpoint)

    run_config = {
        "experiment_name": config["experiment_name"],
        "layout": SIGNAL_CHECKPOINT_LAYOUT,
        "model": config["model"],
        "prm_model": config["prm_model"],
        "dataset": dataset,
        "n_questions": int(config["datasets"][dataset]["n_sample"]),
        "dataset_seed": 42,
        "generation_seed": int(config["seed"]),
        "budget": int(config["budget"]),
        "nd": [int(value) for value in config["nd"]],
        "signals": [signal],
        "thresholds": {signal: threshold},
        "active_threshold": threshold,
        "nll_backend": "vllm",
        "pipeline": config["pipeline"],
        "checkpoint_parts": [str(shared_checkpoint.resolve())],
        "source_checkpoint_sha256": shared_manifest[
            "draft_checkpoint_sha256"
        ],
        "fair_semantics": (
            "nd original draft answers plus nd*(ns-1) rollback suffix answers"
        ),
        "fair_new_semantics": (
            "nd*ns rollback suffix answers and no original draft answers"
        ),
        "full_semantics": (
            "nd original draft answers plus nd*ns rollback suffix answers"
        ),
    }
    if run_config_path.is_file():
        existing_config = read_json(run_config_path)
        if existing_config != run_config:
            raise ValueError(
                f"Existing signal config differs: {result_dir}"
            )
    else:
        write_json(run_config_path, run_config)

    nll_threshold, prm_threshold = _rollback_threshold_pair(
        config, signal, threshold
    )
    rollbacks = _load_rollbacks(
        shared_checkpoint, nll_threshold, prm_threshold
    )
    per_draft = suffixes_per_draft(
        config["nd"], int(config["budget"])
    )
    run_seed = int(config["seed"])
    identities = set()
    existing_records = 0
    if checkpoint.is_file():
        for record in iter_jsonl(checkpoint):
            if (
                record.get("task_type") != "suffix"
                or record.get("strategy") != signal
            ):
                raise ValueError(
                    f"Non-{signal} suffix record in {checkpoint}"
                )
            identity = checkpoint_identity(record)
            if identity in identities:
                raise ValueError(
                    f"Duplicate signal checkpoint identity: {identity}"
                )
            identities.add(identity)
            existing_records += 1

    source_config = config.get("stable_sources", {}).get(dataset, {})
    candidates = [
        _legacy_trial_source(
            config, dataset_root, signal, threshold
        )
    ]
    configured = (
        source_config.get("suffix_sources", {})
        .get(signal, {})
        .get(format(threshold, "g"), [])
    )
    candidates.extend(project_root / value for value in configured)
    unique_candidates = []
    seen_candidates = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen_candidates:
            seen_candidates.add(resolved)
            unique_candidates.append(candidate)

    imported = 0
    source_reports = []
    with checkpoint.open("a", encoding="utf-8") as output:
        for source_dir in unique_candidates:
            source_checkpoint = source_dir / "checkpoint.jsonl"
            source_run_path = source_dir / "run_config.json"
            report = {
                "source": str(source_dir),
                "signal": signal,
                "threshold": threshold,
                "imported": 0,
                "skipped_duplicate": 0,
                "skipped_incompatible": 0,
            }
            if (
                not source_checkpoint.is_file()
                or not source_run_path.is_file()
            ):
                report["status"] = "missing"
                source_reports.append(report)
                continue
            source_run = _validate_source_run(
                source_run_path,
                dataset=dataset,
                model=config["model"],
                run_seed=run_seed,
                draft_sha256=shared_manifest[
                    "draft_checkpoint_sha256"
                ],
            )
            source_value = _source_threshold(source_run, signal)
            if source_value is None:
                report["status"] = "missing_source_threshold"
                source_reports.append(report)
                continue
            report["source_threshold"] = source_value
            report["reuse_mode"] = (
                "exact_threshold"
                if math.isclose(
                    source_value,
                    threshold,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                else "rollback_identity"
            )
            for record in iter_jsonl(source_checkpoint):
                if (
                    record.get("task_type") != "suffix"
                    or record.get("strategy") != signal
                ):
                    continue
                draft_idx = int(record["draft_idx"])
                suffix_idx = int(record["suffix_idx"])
                expected_rollback = rollbacks[
                    (record["doc_id"], draft_idx, signal)
                ]
                if (
                    int(record["rollback_step"]) != expected_rollback
                    or suffix_idx >= per_draft[draft_idx]
                ):
                    report["skipped_incompatible"] += 1
                    continue
                require_stable_sampling(record, run_seed)
                identity = checkpoint_identity(record)
                if identity in identities:
                    report["skipped_duplicate"] += 1
                    continue
                identities.add(identity)
                imported += 1
                report["imported"] += 1
                output.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
            report["status"] = "used"
            source_reports.append(report)

    if not (result_dir / "draft_step_tokens.json").is_file():
        shutil.copyfile(
            shared_dir / "draft_step_tokens.json",
            result_dir / "draft_step_tokens.json",
        )
    report = {
        "status": "prepared",
        "layout": SIGNAL_CHECKPOINT_LAYOUT,
        "dataset": dataset,
        "signal": signal,
        "threshold": threshold,
        "shared_checkpoint": str(shared_checkpoint.resolve()),
        "existing_suffix_records": existing_records,
        "imported_suffix_records": imported,
        "sources": source_reports,
    }
    write_json(reuse_report_path, report)
    return report


def prepare_trial_checkpoint(
    config: dict,
    dataset: str,
    project_root: Path,
    shared_dir: Path,
    trial_dir: Path,
    nll_threshold: float,
    prm_threshold: float,
    dynamic_sources: dict[tuple[str, float], list[Path]],
) -> dict:
    """Copy shared records and import every compatible stable suffix."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = trial_dir / "checkpoint.jsonl"
    run_config_path = trial_dir / "run_config.json"
    reuse_report_path = trial_dir / "reuse_report.json"
    thresholds = {
        NLL_SIGNAL: float(nll_threshold),
        PRM_SIGNAL: float(prm_threshold),
    }
    manifest = read_json(shared_dir / "reuse_manifest.json")
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
        "thresholds": thresholds,
        "nll_backend": "vllm",
        "pipeline": config["pipeline"],
        "shared_checkpoint": str(shared_dir / "checkpoint.jsonl"),
        "source_checkpoint_sha256": manifest["draft_checkpoint_sha256"],
        "fair_semantics": (
            "nd original draft answers plus nd*(ns-1) rollback suffix answers"
        ),
        "fair_new_semantics": (
            "nd*ns rollback suffix answers and no original draft answers"
        ),
        "full_semantics": (
            "nd original draft answers plus nd*ns rollback suffix answers"
        ),
    }
    resuming = checkpoint.exists() and run_config_path.exists()
    if resuming:
        existing_run_config = read_json(run_config_path)
        prior_run_config = dict(expected_run_config)
        prior_run_config.pop("fair_new_semantics")
        legacy_run_config = dict(prior_run_config)
        legacy_run_config["fair_semantics"] = (
            "nd*ns rollback suffix answers; retained-prefix tokens counted"
        )
        if existing_run_config in (prior_run_config, legacy_run_config):
            write_json(run_config_path, expected_run_config)
        elif existing_run_config != expected_run_config:
            raise ValueError(f"Existing trial config differs: {trial_dir}")

    shared_checkpoint = shared_dir / "checkpoint.jsonl"
    if not shared_checkpoint.is_file():
        raise FileNotFoundError(shared_checkpoint)
    rollbacks = _load_rollbacks(
        shared_checkpoint, nll_threshold, prm_threshold
    )
    per_draft = suffixes_per_draft(config["nd"], int(config["budget"]))
    run_seed = int(config["seed"])
    source_config = config["stable_sources"][dataset]

    candidates: dict[str, list[Path]] = defaultdict(list)
    for signal, threshold in thresholds.items():
        candidates[signal].extend(
            dynamic_sources.get((signal, float(threshold)), [])
        )
        configured = (
            source_config.get("suffix_sources", {})
            .get(signal, {})
            .get(format(float(threshold), "g"), [])
        )
        candidates[signal].extend(project_root / value for value in configured)

    temp_path = trial_dir / "checkpoint.jsonl.tmp"
    base_checkpoint = checkpoint if resuming else shared_checkpoint
    output_path = checkpoint if resuming else temp_path
    if not resuming:
        shutil.copyfile(shared_checkpoint, temp_path)
    identities = {
        checkpoint_identity(record) for record in iter_jsonl(base_checkpoint)
    }
    existing_suffix_records = sum(
        1
        for record in iter_jsonl(base_checkpoint)
        if record.get("task_type") == "suffix"
    )
    imported_by_signal: Counter = Counter()
    source_reports = []
    with output_path.open("a", encoding="utf-8") as output:
        for signal, threshold in thresholds.items():
            for source_dir in candidates[signal]:
                source_run_path = source_dir / "run_config.json"
                source_checkpoint = source_dir / "checkpoint.jsonl"
                report = {
                    "signal": signal,
                    "threshold": threshold,
                    "source": str(source_dir),
                    "imported": 0,
                    "skipped_duplicate": 0,
                    "skipped_incompatible": 0,
                }
                source_timing = discover_timing_manifest(
                    source_checkpoint
                )
                report["timing_manifest"] = (
                    str(source_timing)
                    if source_timing is not None
                    else None
                )
                if not source_run_path.is_file() or not source_checkpoint.is_file():
                    report["status"] = "missing"
                    source_reports.append(report)
                    continue
                source_run = _validate_source_run(
                    source_run_path,
                    dataset=dataset,
                    model=config["model"],
                    run_seed=run_seed,
                    draft_sha256=manifest["draft_checkpoint_sha256"],
                )
                source_value = _source_threshold(source_run, signal)
                if source_value is None:
                    report["status"] = "missing_source_threshold"
                    source_reports.append(report)
                    continue
                report["source_threshold"] = source_value
                report["reuse_mode"] = (
                    "exact_threshold"
                    if math.isclose(
                        source_value,
                        threshold,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    else "rollback_identity"
                )
                for record in iter_jsonl(source_checkpoint):
                    if (
                        record.get("task_type") != "suffix"
                        or record.get("strategy") != signal
                    ):
                        continue
                    draft_idx = int(record["draft_idx"])
                    suffix_idx = int(record["suffix_idx"])
                    expected_rollback = rollbacks[
                        (record["doc_id"], draft_idx, signal)
                    ]
                    if (
                        int(record["rollback_step"]) != expected_rollback
                        or suffix_idx >= per_draft[draft_idx]
                    ):
                        report["skipped_incompatible"] += 1
                        continue
                    require_stable_sampling(record, run_seed)
                    identity = checkpoint_identity(record)
                    if identity in identities:
                        report["skipped_duplicate"] += 1
                        continue
                    identities.add(identity)
                    imported_by_signal[signal] += 1
                    report["imported"] += 1
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                report["status"] = "used"
                source_reports.append(report)

    if not resuming:
        temp_path.replace(checkpoint)
    if not (trial_dir / "draft_step_tokens.json").is_file():
        shutil.copyfile(
            shared_dir / "draft_step_tokens.json",
            trial_dir / "draft_step_tokens.json",
        )
    if not run_config_path.exists():
        write_json(run_config_path, expected_run_config)
    reuse_report = {
        "status": "augmented" if resuming else "prepared",
        "dataset": dataset,
        "thresholds": thresholds,
        "shared_records": sum(
            1 for _ in iter_jsonl(shared_checkpoint)
        ),
        "shared_timing_manifest": (
            str(discover_timing_manifest(shared_checkpoint))
            if discover_timing_manifest(shared_checkpoint) is not None
            else None
        ),
        "existing_suffix_records": existing_suffix_records,
        "imported_suffix_records": sum(imported_by_signal.values()),
        "imported_by_signal": dict(imported_by_signal),
        "sources": source_reports,
        "legacy_rejections": manifest["rejected_sources"],
    }
    write_json(reuse_report_path, reuse_report)
    return reuse_report


def validate_shared_checkpoint(
    config: dict,
    dataset: str,
    shared_dir: Path,
) -> dict:
    n_questions = int(config["datasets"][dataset]["n_sample"])
    nd_max = max(int(value) for value in config["nd"])
    budget = int(config["budget"])
    expected = {
        "draft": n_questions * nd_max,
        "prm": n_questions * nd_max,
        "logprob": n_questions * nd_max,
        "fullsc": n_questions * (budget - nd_max),
    }
    counts = Counter()
    identities = set()
    sampling_seeds = set()
    for record in iter_jsonl(shared_dir / "checkpoint.jsonl"):
        identity = checkpoint_identity(record)
        if identity in identities:
            raise ValueError(f"Duplicate shared identity: {identity}")
        identities.add(identity)
        counts[record["task_type"]] += 1
        if record["task_type"] in GENERATION_TYPES:
            require_stable_sampling(record, int(config["seed"]))
            seed = int(record["sampling_seed"])
            if seed in sampling_seeds:
                raise ValueError(f"Shared sampling seed collision: {seed}")
            sampling_seeds.add(seed)
        if (
            record["task_type"] == "logprob"
            and record.get("scoring_backend") != "vllm"
        ):
            raise ValueError("Shared NLL record is not marked vllm")
    if dict(counts) != expected:
        raise ValueError(f"Shared counts={dict(counts)}, expected={expected}")
    step_tokens = read_json(shared_dir / "draft_step_tokens.json")
    if len(step_tokens) != expected["draft"]:
        raise ValueError("Shared draft_step_tokens coverage is incomplete")
    timing_report = validate_timing_artifact(
        shared_dir,
        require_evaluation=False,
    )
    report = {
        "status": "passed",
        "dataset": dataset,
        "record_counts": expected,
        "unique_sampling_seeds": len(sampling_seeds),
        "nll_backend": "vllm",
        "timing": timing_report,
    }
    write_json(shared_dir / "validation_report.json", report)
    return report


def validate_signal_checkpoint(
    config: dict,
    dataset: str,
    shared_dir: Path,
    signal: str,
    threshold: float,
    result_dir: Path,
    figure_dir: Path | None = None,
    *,
    require_evaluation: bool = True,
) -> dict:
    """Validate a suffix delta and its logical union with shared records."""
    threshold = float(threshold)
    run_config = read_json(result_dir / "run_config.json")
    if run_config.get("layout") != SIGNAL_CHECKPOINT_LAYOUT:
        raise ValueError(f"Unexpected signal layout: {result_dir}")
    if run_config.get("signals") != [signal]:
        raise ValueError(f"Signal mismatch in {result_dir}")
    if not math.isclose(
        float(run_config["thresholds"][signal]),
        threshold,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"Threshold mismatch in {result_dir}")

    n_questions = int(config["datasets"][dataset]["n_sample"])
    nd_values = [int(value) for value in config["nd"]]
    per_draft = suffixes_per_draft(
        nd_values, int(config["budget"])
    )
    expected_suffixes = n_questions * sum(per_draft.values())
    nll_threshold, prm_threshold = _rollback_threshold_pair(
        config, signal, threshold
    )
    rollbacks = _load_rollbacks(
        shared_dir / "checkpoint.jsonl",
        nll_threshold,
        prm_threshold,
    )
    identities = set()
    suffix_counts: dict[tuple[str, int], int] = Counter()
    generation_seeds = set()
    count = 0
    for record in iter_jsonl(result_dir / "checkpoint.jsonl"):
        if (
            record.get("task_type") != "suffix"
            or record.get("strategy") != signal
        ):
            raise ValueError(
                f"Signal delta contains a non-{signal} suffix record"
            )
        identity = checkpoint_identity(record)
        if identity in identities:
            raise ValueError(f"Duplicate signal identity: {identity}")
        identities.add(identity)
        require_stable_sampling(record, int(config["seed"]))
        seed = int(record["sampling_seed"])
        if seed in generation_seeds:
            raise ValueError(f"Signal sampling seed collision: {seed}")
        generation_seeds.add(seed)
        draft_idx = int(record["draft_idx"])
        expected_rollback = rollbacks[
            (record["doc_id"], draft_idx, signal)
        ]
        if int(record["rollback_step"]) != expected_rollback:
            raise ValueError(
                f"Wrong rollback step in signal delta: {identity}"
            )
        suffix_counts[(record["doc_id"], draft_idx)] += 1
        count += 1
    if count != expected_suffixes:
        raise ValueError(
            f"Signal suffix count={count}, expected={expected_suffixes}"
        )
    expected_roots = n_questions * max(nd_values)
    if len(suffix_counts) != expected_roots:
        raise ValueError(
            f"Signal suffix roots={len(suffix_counts)}, "
            f"expected={expected_roots}"
        )
    for (doc_id, draft_idx), value in suffix_counts.items():
        if value != per_draft[draft_idx]:
            raise ValueError(
                f"{doc_id} draft={draft_idx} suffixes={value}, "
                f"expected={per_draft[draft_idx]}"
            )

    evaluation_rows = 0
    timing_report = None
    if require_evaluation:
        rows = read_json(result_dir / "eval_summary.json")
        methods = {row["method"]: row for row in rows}
        expected_baselines = {"Greedy@1", "SC@8", "SC@16", "SC@32"}
        if not expected_baselines.issubset(methods):
            raise ValueError(
                f"Missing signal baselines in {result_dir}"
            )
        budget = int(config["budget"])
        if methods["SC@32"]["n_answers"] != budget:
            raise ValueError("SC@32 does not contain exactly 32 answers")
        for nd in nd_values:
            ns = budget // nd
            for suffix, variant, n_answers in (
                ("", "full", budget + nd),
                ("_fair", "fair", budget),
                ("_fair_new", "fair_new", budget),
            ):
                method = (
                    f"rollback_{signal}_nd{nd}_ns{ns}{suffix}"
                )
                row = methods[method]
                if row.get("variant") != variant:
                    raise ValueError(
                        f"{method} variant={row.get('variant')}"
                    )
                if row.get("n_answers") != n_answers:
                    raise ValueError(
                        f"{method} n_answers={row.get('n_answers')}"
                    )
                evaluation_rows += 1
        if len(rows) != len(expected_baselines) + evaluation_rows:
            raise ValueError(
                f"Unexpected evaluation row count: {len(rows)}"
            )
        if figure_dir is None:
            raise ValueError("figure_dir is required for evaluation")
        for name in (
            "fig_efficiency_frontier.png",
            "fig_efficiency_frontier.pdf",
        ):
            path = figure_dir / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing signal figure: {path}")
        timing_report = validate_timing_artifact(
            result_dir,
            require_evaluation=True,
        )

    report = {
        "status": "passed",
        "layout": SIGNAL_CHECKPOINT_LAYOUT,
        "dataset": dataset,
        "signal": signal,
        "threshold": threshold,
        "shared_checkpoint": str(
            (shared_dir / "checkpoint.jsonl").resolve()
        ),
        "local_record_counts": {"suffix": expected_suffixes},
        "logical_record_counts": {
            **read_json(shared_dir / "validation_report.json")[
                "record_counts"
            ],
            "suffix": expected_suffixes,
        },
        "unique_suffix_sampling_seeds": len(generation_seeds),
        "evaluation_rows": evaluation_rows,
        "timing": timing_report,
    }
    write_json(result_dir / "validation_report.json", report)
    return report


def validate_trial(
    config: dict,
    dataset: str,
    trial_dir: Path,
    figure_dir: Path | None = None,
) -> dict:
    run_config = read_json(trial_dir / "run_config.json")
    n_questions = int(config["datasets"][dataset]["n_sample"])
    nd_values = [int(value) for value in config["nd"]]
    nd_max = max(nd_values)
    budget = int(config["budget"])
    per_draft = suffixes_per_draft(nd_values, budget)
    expected = {
        "draft": n_questions * nd_max,
        "prm": n_questions * nd_max,
        "logprob": n_questions * nd_max,
        "suffix": n_questions * len(config["signals"]) * sum(
            per_draft.values()
        ),
        "fullsc": n_questions * (budget - nd_max),
    }
    counts = Counter()
    identities = set()
    generation_seeds = set()
    suffix_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for record in iter_jsonl(trial_dir / "checkpoint.jsonl"):
        identity = checkpoint_identity(record)
        if identity in identities:
            raise ValueError(f"Duplicate trial identity: {identity}")
        identities.add(identity)
        task_type = record["task_type"]
        counts[task_type] += 1
        if task_type in GENERATION_TYPES:
            require_stable_sampling(record, int(config["seed"]))
            seed = int(record["sampling_seed"])
            if seed in generation_seeds:
                raise ValueError(f"Trial sampling seed collision: {seed}")
            generation_seeds.add(seed)
        if task_type == "suffix":
            suffix_counts[
                (record["doc_id"], record["strategy"])
            ][int(record["draft_idx"])] += 1
        if (
            task_type == "logprob"
            and record.get("scoring_backend") != "vllm"
        ):
            raise ValueError("Trial NLL record is not marked vllm")
    if dict(counts) != expected:
        raise ValueError(f"Trial counts={dict(counts)}, expected={expected}")
    for (doc_id, signal), values in suffix_counts.items():
        if dict(values) != per_draft:
            raise ValueError(
                f"{doc_id} {signal} suffix counts={dict(values)}"
            )

    rows = read_json(trial_dir / "eval_summary.json")
    methods = {row["method"]: row for row in rows}
    if methods["SC@32"]["n_answers"] != budget:
        raise ValueError("SC@32 does not contain exactly 32 answers")
    for signal in config["signals"]:
        for nd in nd_values:
            ns = budget // nd
            full = methods[f"rollback_{signal}_nd{nd}_ns{ns}"]
            fair = methods[f"rollback_{signal}_nd{nd}_ns{ns}_fair"]
            fair_new = methods[
                f"rollback_{signal}_nd{nd}_ns{ns}_fair_new"
            ]
            expected_full = {
                "variant": "full",
                "n_answers": budget + nd,
                "n_draft_answers": nd,
                "n_suffix_answers": budget,
                "n_generated_drafts": nd,
            }
            expected_fair = {
                "variant": "fair",
                "n_answers": budget,
                "n_draft_answers": nd,
                "n_suffix_answers": budget - nd,
                "n_generated_drafts": nd,
            }
            expected_fair_new = {
                "variant": "fair_new",
                "n_answers": budget,
                "n_draft_answers": 0,
                "n_suffix_answers": budget,
                "n_generated_drafts": nd,
            }
            for row, expected_values in (
                (full, expected_full),
                (fair, expected_fair),
                (fair_new, expected_fair_new),
            ):
                for key, expected_value in expected_values.items():
                    if row.get(key) != expected_value:
                        raise ValueError(
                            f"{row['method']} {key}={row.get(key)}, "
                            f"expected {expected_value}"
                        )
                if not 0 <= float(row["acc"]) <= 1:
                    raise ValueError(f"Invalid accuracy: {row}")
                if dataset == "hotpotqa_open" and not 0 <= float(
                    row["f1"]
                ) <= 1:
                    raise ValueError(f"Invalid F1: {row}")
    for name in (
        "eval_summary_table.md",
        "draft_step_tokens.json",
        "reuse_report.json",
    ):
        path = trial_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing trial artifact: {path}")
    if figure_dir is None:
        figure_dir = (
            Path(config["figures_root"])
            / dataset
            / trial_name(
                run_config["thresholds"][NLL_SIGNAL],
                run_config["thresholds"][PRM_SIGNAL],
            )
        )
    for name in (
        "fig_efficiency_frontier.png",
        "fig_efficiency_frontier.pdf",
    ):
        path = figure_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing trial figure: {path}")
    timing_report = validate_timing_artifact(
        trial_dir,
        require_evaluation=True,
    )
    report = {
        "status": "passed",
        "dataset": dataset,
        "thresholds": run_config["thresholds"],
        "record_counts": expected,
        "unique_generation_seeds": len(generation_seeds),
        "nll_backend": "vllm",
        "rollback_variant_rows": (
            len(config["signals"]) * len(nd_values) * 3
        ),
        "timing": timing_report,
    }
    write_json(trial_dir / "validation_report.json", report)
    return report


def aggregate_signal_dataset(
    config: dict,
    dataset: str,
    dataset_root: Path,
    figure_dir: Path,
) -> list[dict]:
    """Aggregate independent signal-threshold evaluations without a grid."""
    baseline_rows = None
    result_rows = []
    baseline_fields = (
        "method",
        "nd",
        "ns",
        "acc",
        "tokens_per_q",
        "n_answers",
    )
    if dataset == "hotpotqa_open":
        baseline_fields += ("f1",)

    for signal in config["signals"]:
        for threshold in thresholds_for_signal(config, signal):
            result_dir = signal_result_dir(
                dataset_root, signal, threshold
            )
            validation = read_json(
                result_dir / "validation_report.json"
            )
            if validation.get("status") != "passed":
                raise ValueError(
                    f"Signal result did not pass: {result_dir}"
                )
            rows = read_json(result_dir / "eval_summary.json")
            current_baselines = [
                dict(row) for row in rows
                if not row.get("strategy")
            ]
            if baseline_rows is None:
                baseline_rows = current_baselines
            else:
                expected = [
                    tuple(row[field] for field in baseline_fields)
                    for row in baseline_rows
                ]
                observed = [
                    tuple(row[field] for field in baseline_fields)
                    for row in current_baselines
                ]
                if observed != expected:
                    raise ValueError(
                        f"Baseline mismatch: {result_dir}"
                    )
            for row in rows:
                if row.get("strategy") != signal:
                    continue
                copied = dict(row)
                (
                    copied["nll_threshold"],
                    copied["prm_threshold"],
                ) = active_threshold_columns(signal, threshold)
                copied.update(
                    dataset=dataset,
                    active_threshold=threshold,
                    replicated_across_trials=1,
                    source_result_dir=str(result_dir.resolve()),
                )
                result_rows.append(copied)

    if baseline_rows is None:
        raise ValueError(f"No signal results found under {dataset_root}")
    consolidated_rows = []
    for row in baseline_rows:
        copied = dict(row)
        copied.update(
            dataset=dataset,
            nll_threshold=None,
            prm_threshold=None,
            active_threshold=None,
        )
        consolidated_rows.append(copied)
    consolidated_rows.extend(result_rows)

    aggregate_dir = dataset_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    shared_checkpoint = (
        dataset_root / "_shared_seed42" / "checkpoint.jsonl"
    )
    signal_checkpoints = []
    for signal in config["signals"]:
        for threshold in thresholds_for_signal(config, signal):
            checkpoint = (
                signal_result_dir(dataset_root, signal, threshold)
                / "checkpoint.jsonl"
            )
            signal_checkpoints.append({
                "signal": signal,
                "threshold": threshold,
                "path": str(checkpoint.resolve()),
                "records": sum(1 for _ in iter_jsonl(checkpoint)),
                "bytes": checkpoint.stat().st_size,
            })
    shared_records = sum(
        1 for _ in iter_jsonl(shared_checkpoint)
    )
    legacy_trials = [
        dataset_root / trial_name(nll_threshold, prm_threshold)
        for nll_threshold in config["nll_thresholds"]
        for prm_threshold in config["prm_thresholds"]
        if (
            dataset_root
            / trial_name(nll_threshold, prm_threshold)
        ).is_dir()
    ]
    write_json(
        dataset_root / "checkpoint_layout.json",
        {
            "status": "passed",
            "layout": SIGNAL_CHECKPOINT_LAYOUT,
            "dataset": dataset,
            "cartesian_trial_checkpoints": False,
            "legacy_cartesian_trials_retained": [
                str(path.resolve()) for path in legacy_trials
            ],
            "shared": {
                "path": str(shared_checkpoint.resolve()),
                "records": shared_records,
                "bytes": shared_checkpoint.stat().st_size,
            },
            "signal_deltas": signal_checkpoints,
            "physical_records": (
                shared_records
                + sum(
                    item["records"] for item in signal_checkpoints
                )
            ),
            "logical_records_per_signal_evaluation": (
                shared_records + signal_checkpoints[0]["records"]
            ),
        },
    )
    write_json(dataset_root / "eval_summary.json", consolidated_rows)
    write_json(
        aggregate_dir / "threshold_summary.json", result_rows
    )
    write_json(
        aggregate_dir / "threshold_results_by_signal.json",
        result_rows,
    )
    fieldnames = sorted(
        {key for row in result_rows for key in row}
    )
    with (aggregate_dir / "threshold_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    table = build_eval_summary_table(dataset, consolidated_rows)
    (dataset_root / "eval_summary_table.md").write_text(
        table, encoding="utf-8"
    )
    (aggregate_dir / "threshold_summary.md").write_text(
        table, encoding="utf-8"
    )
    _plot_thresholds(config, dataset, result_rows, figure_dir)
    return result_rows


def aggregate_dataset(
    config: dict,
    dataset: str,
    dataset_root: Path,
    figure_dir: Path,
) -> list[dict]:
    """Aggregate the legacy NLL x PRM trial layout."""
    rows = []
    baseline_rows = None
    timing_trials = []
    for nll_threshold in config["nll_thresholds"]:
        for prm_threshold in config["prm_thresholds"]:
            trial_dir = dataset_root / trial_name(
                nll_threshold, prm_threshold
            )
            validation = read_json(trial_dir / "validation_report.json")
            if validation.get("status") != "passed":
                raise ValueError(f"Trial did not pass: {trial_dir}")
            if validation.get("timing"):
                timing_trials.append(
                    {
                        "dataset": dataset,
                        "nll_threshold": float(nll_threshold),
                        "prm_threshold": float(prm_threshold),
                        **validation["timing"],
                    }
                )
            trial_rows = read_json(trial_dir / "eval_summary.json")
            trial_baselines = [
                dict(row) for row in trial_rows
                if not row.get("strategy")
            ]
            if baseline_rows is None:
                baseline_rows = trial_baselines
            else:
                baseline_fields = (
                    "method",
                    "nd",
                    "ns",
                    "acc",
                    "tokens_per_q",
                    "n_answers",
                )
                if dataset == "hotpotqa_open":
                    baseline_fields += ("f1",)
                expected = [
                    tuple(row[field] for field in baseline_fields)
                    for row in baseline_rows
                ]
                observed = [
                    tuple(row[field] for field in baseline_fields)
                    for row in trial_baselines
                ]
                if observed != expected:
                    raise ValueError(
                        f"Baseline mismatch across trials: {trial_dir}"
                    )
            for row in trial_rows:
                if row.get("strategy") not in config["signals"]:
                    continue
                copied = dict(row)
                copied.update(
                    dataset=dataset,
                    nll_threshold=float(nll_threshold),
                    prm_threshold=float(prm_threshold),
                )
                copied["active_threshold"] = (
                    float(nll_threshold)
                    if row["strategy"] == NLL_SIGNAL
                    else float(prm_threshold)
                )
                rows.append(copied)

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["strategy"],
                row["active_threshold"],
                row["variant"],
                int(row["nd"]),
            )
        ].append(row)
    unique_rows = []
    for key, records in sorted(grouped.items()):
        for metric in ("acc", "tokens_per_q", "n_answers"):
            values = {float(record[metric]) for record in records}
            if len(values) != 1:
                raise ValueError(
                    f"Cross-trial mismatch for {key} {metric}: {values}"
                )
        if dataset == "hotpotqa_open":
            values = {float(record["f1"]) for record in records}
            if len(values) != 1:
                raise ValueError(f"Cross-trial F1 mismatch for {key}")
        selected = dict(records[0])
        (
            selected["nll_threshold"],
            selected["prm_threshold"],
        ) = active_threshold_columns(
            selected["strategy"],
            selected["active_threshold"],
        )
        selected["replicated_across_trials"] = len(records)
        unique_rows.append(selected)

    aggregate_dir = dataset_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    write_json(aggregate_dir / "threshold_results_all_trials.json", rows)
    write_json(aggregate_dir / "threshold_summary.json", unique_rows)
    write_json(aggregate_dir / "timing_summary.json", timing_trials)
    fieldnames = sorted({key for row in unique_rows for key in row})
    with (aggregate_dir / "threshold_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_rows)

    if baseline_rows is None:
        raise ValueError(f"No threshold trials found under {dataset_root}")
    consolidated_rows = []
    for row in baseline_rows:
        copied = dict(row)
        copied.update(
            dataset=dataset,
            nll_threshold=None,
            prm_threshold=None,
            active_threshold=None,
        )
        consolidated_rows.append(copied)
    consolidated_rows.extend(unique_rows)
    write_json(dataset_root / "eval_summary.json", consolidated_rows)
    table = build_eval_summary_table(dataset, consolidated_rows)
    (dataset_root / "eval_summary_table.md").write_text(
        table, encoding="utf-8"
    )
    (aggregate_dir / "threshold_summary.md").write_text(
        table, encoding="utf-8"
    )
    _plot_thresholds(config, dataset, unique_rows, figure_dir)
    return unique_rows


def _plot_thresholds(
    config: dict,
    dataset: str,
    rows: list[dict],
    figure_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    signals = [NLL_SIGNAL, PRM_SIGNAL]
    variants = ["fair", "fair_new", "full"]
    colors = {4: "#2563eb", 8: "#16a34a", 16: "#dc2626"}
    figure, axes = plt.subplots(2, 3, figsize=(15.3, 7.6))
    for row_index, signal in enumerate(signals):
        for column_index, variant in enumerate(variants):
            axis = axes[row_index][column_index]
            for nd in config["nd"]:
                selected = sorted(
                    [
                        row for row in rows
                        if row["strategy"] == signal
                        and row["variant"] == variant
                        and int(row["nd"]) == int(nd)
                    ],
                    key=lambda row: row["active_threshold"],
                )
                axis.plot(
                    [row["active_threshold"] for row in selected],
                    [row["acc"] for row in selected],
                    marker="o",
                    linewidth=1.7,
                    color=colors[int(nd)],
                    label=f"nd={nd}",
                )
            axis.set_xlabel("NLL threshold" if signal == NLL_SIGNAL else "PRM threshold")
            axis.set_ylabel("EM")
            axis.set_title(
                f"{signal.replace('_drop_fb_last', '').upper()} — {variant}"
            )
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"Qwen2.5-3B threshold comparison: {dataset} "
        f"(seed={config['seed']})"
    )
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(
            figure_dir / f"threshold_accuracy.{extension}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(figure)
