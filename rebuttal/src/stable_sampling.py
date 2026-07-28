"""Stable per-request sampling seeds for rebuttal generation runs."""

import hashlib
import json
from typing import Any, Dict, Tuple


SEED_SCHEMA = "lost-in-the-second/per-request-v1"
MAX_TORCH_SEED = (1 << 63) - 1


def task_seed_identity(task: Dict[str, Any]) -> Tuple[Any, ...]:
    """Return the fields that uniquely identify one generation request."""
    task_type = task["task_type"]
    if task_type == "draft":
        request_fields = (task["draft_idx"],)
    elif task_type == "fullsc":
        request_fields = (task["sc_idx"],)
    elif task_type == "suffix":
        request_fields = (
            task["draft_idx"],
            task["strategy"],
            task["rollback_step"],
            task["suffix_idx"],
        )
    else:
        raise ValueError(
            f"Unsupported seeded generation task type: {task_type}"
        )
    return (task_type, str(task["doc_id"]), *request_fields)


def derive_sampling_seed(
    run_seed: int,
    task: Dict[str, Any],
) -> int:
    """Derive a reproducible independent seed for one vLLM request."""
    payload = json.dumps(
        [SEED_SCHEMA, int(run_seed), *task_seed_identity(task)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value & MAX_TORCH_SEED
