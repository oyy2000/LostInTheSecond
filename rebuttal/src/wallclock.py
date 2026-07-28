"""Checkpoint-aware wall-clock recording for multi-stage experiments.

The authoritative artifact is ``timing/timing_manifest.json``.  It keeps the
wall-clock observed by the current invocation separate from any historical
from-scratch timing inherited from a checkpoint source.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = 1


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, values: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def task_identity(value: Mapping[str, object]) -> tuple[object, ...]:
    """Return the stable identity shared by tasks and checkpoint records."""
    task_type = str(value["task_type"])
    if task_type == "nll":
        task_type = "logprob"
    doc_id = str(value["doc_id"])
    if task_type in {"draft", "prm", "logprob"}:
        return task_type, doc_id, int(value["draft_idx"])
    if task_type == "suffix":
        return (
            task_type,
            doc_id,
            int(value["draft_idx"]),
            str(value["strategy"]),
            int(value["rollback_step"]),
            int(value["suffix_idx"]),
        )
    if task_type == "fullsc":
        return task_type, doc_id, int(value["sc_idx"])
    if task_type == "gpt":
        return task_type, doc_id, int(value["draft_idx"])
    if task_type == "evaluation":
        return task_type, doc_id
    raise ValueError(f"Unsupported timing task type: {task_type!r}")


def task_key(value: Mapping[str, object]) -> str:
    return json.dumps(
        task_identity(value),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def task_descriptor(value: Mapping[str, object]) -> dict:
    """Return JSON fields needed to aggregate one timed task."""
    identity = task_identity(value)
    task_type = str(identity[0])
    descriptor = {
        "task_type": task_type,
        "doc_id": str(identity[1]),
    }
    if task_type in {"draft", "prm", "logprob", "gpt"}:
        descriptor["draft_idx"] = int(identity[2])
    elif task_type == "suffix":
        descriptor.update(
            draft_idx=int(identity[2]),
            strategy=str(identity[3]),
            rollback_step=int(identity[4]),
            suffix_idx=int(identity[5]),
        )
    elif task_type == "fullsc":
        descriptor["sc_idx"] = int(identity[2])
    return descriptor


def task_set_sha256(keys: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(set(keys)):
        digest.update(key.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def discover_timing_manifest(checkpoint: Path) -> Path | None:
    """Find the authoritative timing manifest beside a checkpoint."""
    candidate = checkpoint.parent / "timing" / "timing_manifest.json"
    return candidate if candidate.is_file() else None


def discover_task_event_store(checkpoint: Path) -> Path | None:
    """Find per-task timing events beside a checkpoint."""
    candidate = checkpoint.parent / "timing" / "task_events.jsonl"
    return candidate if candidate.is_file() else None


def task_event_store_for_manifest(manifest_path: Path) -> Path | None:
    """Find the task-event store associated with a timing manifest."""
    resolved = manifest_path.resolve()
    candidates = [resolved.parent / "task_events.jsonl"]
    if resolved.parent.parent.name == "runs":
        candidates.append(
            resolved.parent.parent.parent / "task_events.jsonl"
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def environment_metadata() -> dict:
    metadata = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": {
            "vllm": _package_version("vllm"),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
        },
    }
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        metadata["gpus"] = [
            line.strip()
            for line in query.stdout.splitlines()
            if line.strip()
        ]
    except (OSError, subprocess.SubprocessError):
        metadata["gpus"] = []
    return metadata


def load_manifest(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported timing schema in {path}")
    return value


def exact_source_phase(
    phase: str,
    required_keys: set[str],
    source_manifests: Sequence[Path],
) -> dict | None:
    """Return historical timing only for an exact task-set match.

    A strict exact match avoids invalid linear scaling and invalid addition of
    multi-GPU critical paths from unrelated source executions.
    """
    if not required_keys:
        return None
    matches = []
    for manifest_path in source_manifests:
        manifest = load_manifest(manifest_path)
        for source_phase in manifest.get("phases", []):
            if source_phase.get("phase") != phase:
                continue
            if source_phase.get("task_set_sha256") != task_set_sha256(
                required_keys
            ):
                continue
            if int(source_phase.get("tasks_required", -1)) != len(
                required_keys
            ):
                continue
            from_scratch = source_phase.get(
                "reconstructed_from_scratch_wall_sec"
            )
            if from_scratch is None:
                continue
            matches.append(
                {
                    "manifest": str(manifest_path.resolve()),
                    "run_id": manifest.get("run_id"),
                    "wall_sec": float(from_scratch),
                    "task_set_sha256": source_phase["task_set_sha256"],
                }
            )
    if not matches:
        return None
    wall_values = {match["wall_sec"] for match in matches}
    if len(wall_values) != 1:
        raise ValueError(
            f"Conflicting exact timing sources for phase={phase}: {matches}"
        )
    return matches[0]


def exact_local_phase(
    phase: str,
    required_keys: set[str],
    phase_event_paths: Sequence[Path],
) -> dict | None:
    """Return the latest exact timing from an interrupted local run.

    A run may complete and persist one or more phases before a later phase
    fails, so it never publishes a top-level timing manifest.  Its
    phase_events.json still provides authoritative timing for an exact task
    set and can safely reconstruct that phase during resume.
    """
    if not required_keys:
        return None
    required_digest = task_set_sha256(required_keys)
    for events_path in phase_event_paths:
        if not events_path.is_file():
            continue
        events = json.loads(events_path.read_text(encoding="utf-8"))
        for source_phase in reversed(events):
            if source_phase.get("phase") != phase:
                continue
            if source_phase.get("task_set_sha256") != required_digest:
                continue
            if int(source_phase.get("tasks_required", -1)) != len(
                required_keys
            ):
                continue
            from_scratch = source_phase.get(
                "reconstructed_from_scratch_wall_sec"
            )
            if from_scratch is None:
                continue
            return {
                "phase_events": str(events_path.resolve()),
                "run_id": events_path.parent.name,
                "wall_sec": float(from_scratch),
                "task_set_sha256": required_digest,
            }
    return None


@dataclass
class WorkerTiming:
    """Worker-side timing with optional per-batch measurements."""

    path: Path
    phase: str
    run_id: str
    shard_id: int
    gpu_id: str
    tasks: Sequence[Mapping[str, object]]
    started_ns: int = field(default_factory=time.perf_counter_ns)
    model_load_sec: float | None = None
    batches: list[dict] = field(default_factory=list)
    task_events: list[dict] = field(default_factory=list)

    def record_model_load(self, started_ns: int) -> None:
        self.model_load_sec = (
            time.perf_counter_ns() - started_ns
        ) / 1_000_000_000

    def record_batch(
        self,
        batch_tasks: Sequence[Mapping[str, object]],
        started_ns: int,
        *,
        generated_tokens: int | None = None,
        input_tokens: int | None = None,
    ) -> None:
        event = {
            "batch_index": len(self.batches),
            "wall_sec": (
                time.perf_counter_ns() - started_ns
            ) / 1_000_000_000,
            "tasks": len(batch_tasks),
            "task_keys": [task_key(task) for task in batch_tasks],
        }
        if generated_tokens is not None:
            event["generated_tokens"] = int(generated_tokens)
        if input_tokens is not None:
            event["input_tokens"] = int(input_tokens)
        self.batches.append(event)

    def record_task(
        self,
        task: Mapping[str, object],
        *,
        started_unix_sec: float,
        finished_unix_sec: float,
        timing_source: str,
        details: Mapping[str, object] | None = None,
    ) -> None:
        """Record one request's active wall-clock interval."""
        started = float(started_unix_sec)
        finished = float(finished_unix_sec)
        if finished < started:
            raise ValueError(
                f"Task timing ends before it starts: {task_key(task)}"
            )
        event = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "phase": self.phase,
            "shard_id": self.shard_id,
            "gpu_id": self.gpu_id,
            "task_key": task_key(task),
            **task_descriptor(task),
            "started_unix_sec": started,
            "finished_unix_sec": finished,
            "active_wall_sec": finished - started,
            "timing_source": timing_source,
        }
        if details:
            event["details"] = dict(details)
        self.task_events.append(event)

    def record_vllm_request(
        self,
        task: Mapping[str, object],
        output: object,
        *,
        fallback_started_unix_sec: float,
        fallback_finished_unix_sec: float,
    ) -> None:
        """Record vLLM request latency, falling back to its batch window."""
        metrics = getattr(output, "metrics", None)
        arrival = getattr(metrics, "arrival_time", None)
        finished = getattr(metrics, "finished_time", None)
        if arrival is None or finished is None:
            self.record_task(
                task,
                started_unix_sec=fallback_started_unix_sec,
                finished_unix_sec=fallback_finished_unix_sec,
                timing_source="shared_batch_wall_fallback",
            )
            return
        details = {}
        for field_name in (
            "first_scheduled_time",
            "first_token_time",
            "time_in_queue",
            "scheduler_time",
            "model_forward_time",
            "model_execute_time",
        ):
            value = getattr(metrics, field_name, None)
            if value is not None:
                details[field_name] = float(value)
        self.record_task(
            task,
            started_unix_sec=float(arrival),
            finished_unix_sec=float(finished),
            timing_source="vllm_arrival_to_finish",
            details=details,
        )

    def finish(self, *, status: str = "passed") -> dict:
        value = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "phase": self.phase,
            "run_id": self.run_id,
            "shard_id": self.shard_id,
            "gpu_id": self.gpu_id,
            "tasks": len(self.tasks),
            "task_keys": [task_key(task) for task in self.tasks],
            "task_set_sha256": task_set_sha256(
                task_key(task) for task in self.tasks
            ),
            "model_load_sec": self.model_load_sec,
            "worker_wall_sec": (
                time.perf_counter_ns() - self.started_ns
            ) / 1_000_000_000,
            "batches": self.batches,
            "task_events": self.task_events,
        }
        _atomic_write_json(self.path, value)
        return value


def write_generic_worker_timing(
    path: Path | None,
    *,
    phase: str,
    run_id: str,
    shard_id: int,
    gpu_id: str,
    tasks: Sequence[Mapping[str, object]],
    started_ns: int,
) -> None:
    """Write process-level timing when a worker has no detailed recorder."""
    if path is None or path.is_file():
        return
    value = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "phase": phase,
        "run_id": run_id,
        "shard_id": shard_id,
        "gpu_id": gpu_id,
        "tasks": len(tasks),
        "task_keys": [task_key(task) for task in tasks],
        "task_set_sha256": task_set_sha256(
            task_key(task) for task in tasks
        ),
        "model_load_sec": None,
        "worker_wall_sec": (
            time.perf_counter_ns() - started_ns
        ) / 1_000_000_000,
        "batches": [],
        "task_events": [],
    }
    _atomic_write_json(path, value)


def load_task_events(paths: Sequence[Path]) -> dict[str, dict]:
    """Load task events, with later sources replacing earlier ones."""
    events = {}
    for path in paths:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                key = str(event["task_key"])
                events[key] = event
    return events


def worker_task_events(run_dir: Path) -> dict[str, dict]:
    """Read task events emitted by every worker in one invocation."""
    events = {}
    for path in sorted((run_dir / "workers").glob("*/*.json")):
        worker = load_manifest(path)
        for event in worker.get("task_events", []):
            events[str(event["task_key"])] = event
    return events


def interval_union_seconds(events: Sequence[Mapping[str, object]]) -> float:
    """Measure active wall time without double-counting parallel requests."""
    intervals = sorted(
        (
            float(event["started_unix_sec"]),
            float(event["finished_unix_sec"]),
        )
        for event in events
    )
    if not intervals:
        return 0.0
    total = 0.0
    current_start, current_end = intervals[0]
    for started, finished in intervals[1:]:
        if started <= current_end:
            current_end = max(current_end, finished)
            continue
        total += current_end - current_start
        current_start, current_end = started, finished
    return total + current_end - current_start


def _timed_stage(
    expected_records: Sequence[Mapping[str, object]],
    events: Mapping[str, Mapping[str, object]],
) -> dict:
    expected_keys = [task_key(record) for record in expected_records]
    timed_events = [
        events[key] for key in expected_keys if key in events
    ]
    complete = len(timed_events) == len(expected_keys)
    return {
        "tasks_expected": len(expected_keys),
        "tasks_timed": len(timed_events),
        "complete": complete,
        "active_wall_sec": (
            interval_union_seconds(timed_events)
            if complete
            else None
        ),
    }


def build_per_question_timings(
    checkpoint_records: Sequence[Mapping[str, object]],
    task_events: Mapping[str, Mapping[str, object]],
    signals: Sequence[str],
) -> list[dict]:
    """Aggregate draft+suffix generation and signal time for each question.

    Each stage is the union of its request intervals. This preserves elapsed
    wall-clock semantics when several requests for one question run in
    parallel, while excluding one-time model loading.
    """
    supported_signals = [
        signal
        for signal in signals
        if signal.startswith(("nll", "prm"))
    ]
    records_by_doc = {}
    for record in checkpoint_records:
        doc_id = str(record["doc_id"])
        records_by_doc.setdefault(doc_id, []).append(record)

    rows = []
    for doc_id in sorted(records_by_doc):
        records = records_by_doc[doc_id]
        drafts = [
            record
            for record in records
            if record.get("task_type") == "draft"
        ]
        draft_stage = _timed_stage(drafts, task_events)
        for signal in supported_signals:
            suffixes = [
                record
                for record in records
                if record.get("task_type") == "suffix"
                and record.get("strategy") == signal
            ]
            signal_task_type = (
                "logprob" if signal.startswith("nll") else "prm"
            )
            signal_records = [
                record
                for record in records
                if record.get("task_type") == signal_task_type
            ]
            suffix_stage = _timed_stage(suffixes, task_events)
            signal_stage = _timed_stage(signal_records, task_events)
            generation_complete = (
                draft_stage["complete"] and suffix_stage["complete"]
            )
            generation_sec = None
            if generation_complete:
                generation_sec = float(
                    draft_stage["active_wall_sec"]
                ) + float(suffix_stage["active_wall_sec"])
            signal_sec = signal_stage["active_wall_sec"]
            ratio = None
            if (
                generation_sec is not None
                and signal_sec is not None
                and float(signal_sec) > 0
            ):
                ratio = generation_sec / float(signal_sec)
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "measurement": (
                        "union_of_task_active_intervals_"
                        "excluding_model_load"
                    ),
                    "scope": "all_checkpoint_tasks_for_signal",
                    "doc_id": doc_id,
                    "signal": signal,
                    "draft_generation": draft_stage,
                    "suffix_generation": suffix_stage,
                    "total_generation_active_wall_sec": generation_sec,
                    "signal_computation": signal_stage,
                    "generation_to_signal_ratio": ratio,
                    "complete": (
                        generation_complete and signal_stage["complete"]
                    ),
                }
            )
    return rows


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return (
        ordered[lower] * (1 - fraction)
        + ordered[upper] * fraction
    )


def summarize_per_question_timings(rows: Sequence[Mapping]) -> dict:
    summaries = []
    signals = sorted({str(row["signal"]) for row in rows})
    for signal in signals:
        signal_rows = [
            row for row in rows if row["signal"] == signal
        ]
        complete_rows = [
            row for row in signal_rows if row["complete"]
        ]
        generation = [
            float(row["total_generation_active_wall_sec"])
            for row in complete_rows
        ]
        signal_times = [
            float(row["signal_computation"]["active_wall_sec"])
            for row in complete_rows
        ]
        draft = [
            float(row["draft_generation"]["active_wall_sec"])
            for row in complete_rows
        ]
        suffix = [
            float(row["suffix_generation"]["active_wall_sec"])
            for row in complete_rows
        ]
        generation_sum = sum(generation)
        signal_sum = sum(signal_times)
        summaries.append(
            {
                "signal": signal,
                "questions": len(signal_rows),
                "complete_questions": len(complete_rows),
                "draft_generation_mean_sec": (
                    statistics.fmean(draft) if draft else None
                ),
                "suffix_generation_mean_sec": (
                    statistics.fmean(suffix) if suffix else None
                ),
                "total_generation_mean_sec": (
                    statistics.fmean(generation)
                    if generation
                    else None
                ),
                "total_generation_median_sec": (
                    statistics.median(generation)
                    if generation
                    else None
                ),
                "total_generation_p95_sec": _percentile(
                    generation, 0.95
                ),
                "signal_computation_mean_sec": (
                    statistics.fmean(signal_times)
                    if signal_times
                    else None
                ),
                "signal_computation_median_sec": (
                    statistics.median(signal_times)
                    if signal_times
                    else None
                ),
                "signal_computation_p95_sec": _percentile(
                    signal_times, 0.95
                ),
                "generation_to_signal_ratio_of_sums": (
                    generation_sum / signal_sum
                    if signal_sum > 0
                    else None
                ),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "measurement": (
            "union_of_task_active_intervals_excluding_model_load"
        ),
        "scope": "all_checkpoint_tasks_for_signal",
        "description": (
            "Per-question active wall time. Draft and signal-specific "
            "suffix generation are added; overlapping requests within "
            "each stage are counted once."
        ),
        "signals": summaries,
    }


def publish_per_question_timings(
    output_dir: Path,
    *,
    checkpoint_records: Sequence[Mapping[str, object]],
    signals: Sequence[str],
    current_run_dir: Path,
    source_event_stores: Sequence[Path] = (),
) -> dict:
    """Merge task events and publish per-question timing artifacts."""
    timing_root = output_dir / "timing"
    event_store = timing_root / "task_events.jsonl"
    source_paths = [*source_event_stores]
    if event_store.is_file() and event_store not in source_paths:
        source_paths.append(event_store)
    events = load_task_events(source_paths)
    events.update(worker_task_events(current_run_dir))

    required_keys = {
        task_key(record)
        for record in checkpoint_records
        if record.get("task_type")
        in {"draft", "suffix", "prm", "logprob"}
    }
    filtered_events = {
        key: event for key, event in events.items()
        if key in required_keys
    }
    _atomic_write_jsonl(
        event_store,
        [
            filtered_events[key]
            for key in sorted(filtered_events)
        ],
    )

    rows = build_per_question_timings(
        checkpoint_records, filtered_events, signals
    )
    summary = summarize_per_question_timings(rows)
    summary["task_events_required"] = len(required_keys)
    summary["task_events_recorded"] = len(filtered_events)
    summary["task_event_coverage"] = (
        len(filtered_events) / len(required_keys)
        if required_keys
        else 1.0
    )
    _atomic_write_jsonl(timing_root / "per_question.jsonl", rows)
    _atomic_write_json(
        timing_root / "per_question_summary.json", summary
    )
    return summary


class RunTimingRecorder:
    """Append phase measurements and publish an authoritative manifest."""

    def __init__(
        self,
        output_dir: Path,
        *,
        mode: str,
        context: Mapping[str, object],
        source_manifests: Sequence[Path] = (),
        started_ns: int | None = None,
        run_id: str | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.mode = mode
        self.run_id = run_id or new_run_id()
        self.started_ns = started_ns or time.perf_counter_ns()
        self.timing_root = output_dir / "timing"
        self.run_dir = self.timing_root / "runs" / self.run_id
        self.local_phase_event_sources = sorted(
            self.timing_root.glob("runs/*/phase_events.json"),
            reverse=True,
        )
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.source_manifests = []
        self.source_manifest_origins = []
        source_snapshot_dir = self.run_dir / "source_manifests"
        for source_path in source_manifests:
            resolved = source_path.resolve()
            if not resolved.is_file():
                continue
            content = resolved.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            snapshot = source_snapshot_dir / f"{digest}.json"
            if not snapshot.is_file():
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                snapshot.write_bytes(content)
            resolved_snapshot = snapshot.resolve()
            if resolved_snapshot not in self.source_manifests:
                self.source_manifests.append(resolved_snapshot)
                self.source_manifest_origins.append(
                    {
                        "original": str(resolved),
                        "snapshot": str(resolved_snapshot),
                        "sha256": digest,
                    }
                )
        self.context = dict(context)
        self.phases: list[dict] = []
        _atomic_write_json(
            self.run_dir / "run_metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "mode": self.mode,
                "context": self.context,
                "environment": environment_metadata(),
                "source_manifests": [
                    str(path) for path in self.source_manifests
                ],
                "source_manifest_origins": self.source_manifest_origins,
            },
        )

    def worker_timing_path(self, phase: str, shard_id: int) -> Path:
        worker_dir = self.run_dir / "workers" / phase
        worker_dir.mkdir(parents=True, exist_ok=True)
        return worker_dir / f"shard_{shard_id}.json"

    def record_phase(
        self,
        phase: str,
        *,
        required: Sequence[Mapping[str, object]],
        executed: Sequence[Mapping[str, object]],
        current_wall_sec: float,
        launch_stats: Mapping[str, object] | None = None,
    ) -> dict:
        required_keys = {task_key(value) for value in required}
        executed_keys = {task_key(value) for value in executed}
        unexpected = executed_keys - required_keys
        if unexpected:
            raise ValueError(
                f"Executed timing tasks are not required for {phase}: "
                f"{sorted(unexpected)[:3]}"
            )
        reused_keys = required_keys - executed_keys
        historical = None
        reconstruction_kind = None
        reconstructed = None
        if required_keys and executed_keys == required_keys:
            reconstructed = float(current_wall_sec)
            reconstruction_kind = "observed_current_full"
        elif required_keys and not executed_keys:
            historical = exact_source_phase(
                phase, required_keys, self.source_manifests
            )
            if historical is None:
                historical = exact_local_phase(
                    phase,
                    required_keys,
                    self.local_phase_event_sources,
                )
            if historical is not None:
                reconstructed = historical["wall_sec"]
                reconstruction_kind = (
                    "exact_local_run_history"
                    if "phase_events" in historical
                    else "exact_historical_full"
                )

        event = {
            "phase": phase,
            "status": (
                "fully_reused"
                if required_keys and not executed_keys
                else "partial_reuse"
                if reused_keys
                else "computed"
                if required_keys
                else "not_required"
            ),
            "tasks_required": len(required_keys),
            "tasks_reused": len(reused_keys),
            "tasks_executed": len(executed_keys),
            "task_set_sha256": task_set_sha256(required_keys),
            "executed_task_set_sha256": task_set_sha256(executed_keys),
            "observed_current_wall_sec": float(current_wall_sec),
            "reconstructed_from_scratch_wall_sec": reconstructed,
            "reconstruction_kind": reconstruction_kind,
            "historical_source": historical,
            "launch": dict(launch_stats or {}),
        }
        self.phases.append(event)
        _atomic_write_json(
            self.run_dir / "phase_events.json",
            self.phases,
        )
        return event

    def finalize(self) -> dict:
        reconstructed_values = [
            phase["reconstructed_from_scratch_wall_sec"]
            for phase in self.phases
            if phase["status"] != "not_required"
        ]
        complete_reconstruction = bool(reconstructed_values) and all(
            value is not None for value in reconstructed_values
        )
        checkpoint_artifact = None
        checkpoint_value = self.context.get("checkpoint")
        if checkpoint_value:
            checkpoint = Path(str(checkpoint_value))
            if checkpoint.is_file():
                checkpoint_artifact = {
                    "path": str(checkpoint.resolve()),
                    "size_bytes": checkpoint.stat().st_size,
                    "sha256": file_sha256(checkpoint),
                }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "run_id": self.run_id,
            "mode": self.mode,
            "context": self.context,
            "checkpoint_artifact": checkpoint_artifact,
            "environment": json.loads(
                (self.run_dir / "run_metadata.json").read_text(
                    encoding="utf-8"
                )
            )["environment"],
            "source_manifests": [
                str(path) for path in self.source_manifests
            ],
            "source_manifest_origins": self.source_manifest_origins,
            "observed_current_end_to_end_wall_sec": (
                time.perf_counter_ns() - self.started_ns
            ) / 1_000_000_000,
            "reconstructed_from_scratch_wall_sec": (
                sum(float(value) for value in reconstructed_values)
                if complete_reconstruction
                else None
            ),
            "reconstruction_complete": complete_reconstruction,
            "phases": self.phases,
        }
        run_manifest = self.run_dir / "timing_manifest.json"
        _atomic_write_json(run_manifest, manifest)
        _atomic_write_json(
            self.timing_root / "timing_manifest.json",
            manifest,
        )
        return manifest
