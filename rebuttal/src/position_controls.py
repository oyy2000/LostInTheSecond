"""Reusable strategy logic for the Qwen rebuttal control experiments."""

import hashlib
import math
import random
from typing import Dict, List


FIXED_FRACTIONS: Dict[str, float] = {
    "fixed_25": 0.25,
    "fixed_50": 0.50,
    "fixed_75": 0.75,
}
POSITION_STRATEGIES = ["random_uniform", *FIXED_FRACTIONS]
FINAL_STEP_STRATEGY = "final_step_only"
ALL_STRATEGIES = [*POSITION_STRATEGIES, FINAL_STEP_STRATEGY]


def random_rollback_step(
    doc_id: str,
    draft_idx: int,
    n_steps: int,
    seed: int,
) -> int:
    """Sample the number of steps retained before a random target step."""
    if n_steps <= 1:
        return 0
    key = f"{seed}|{doc_id}|{draft_idx}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return random.Random(value).randrange(n_steps)


def fixed_rollback_step(n_steps: int, fraction: float) -> int:
    """Retain steps strictly before the ceil(fraction * n_steps) step."""
    if n_steps <= 1:
        return 0
    target_step = max(1, math.ceil(fraction * n_steps))
    return min(n_steps - 1, target_step - 1)


def rollback_steps_for_draft(
    doc_id: str,
    draft_idx: int,
    n_steps: int,
    random_seed: int,
) -> Dict[str, int]:
    result = {
        "random_uniform": random_rollback_step(
            doc_id, draft_idx, n_steps, random_seed
        )
    }
    for strategy, fraction in FIXED_FRACTIONS.items():
        result[strategy] = fixed_rollback_step(n_steps, fraction)
    return result


def final_step_rollback_step(n_steps: int) -> int:
    """Return the number of steps retained before regenerating the last."""
    return max(0, n_steps - 1)


def final_step_prefix(steps: List[str]) -> str:
    """Retain every complete reasoning step except the final step."""
    clean_steps = [step.strip() for step in steps if step.strip()]
    retained_steps = clean_steps[:final_step_rollback_step(len(clean_steps))]
    if not retained_steps:
        return ""
    return "\n\n".join(retained_steps) + "\n\n"


def suffixes_per_draft(
    rollback_configs: List[tuple[int, int]],
) -> Dict[int, int]:
    if not rollback_configs:
        return {}
    nd_max = max(nd for nd, _ in rollback_configs)
    return {
        draft_idx: max(
            ns for nd, ns in rollback_configs if draft_idx < nd
        )
        for draft_idx in range(nd_max)
    }


def additional_suffixes_per_draft(
    rollback_configs: List[tuple[int, int]],
) -> Dict[int, int]:
    """Suffix counts when each original draft contributes one answer."""
    if not rollback_configs:
        return {}
    nd_max = max(nd for nd, _ in rollback_configs)
    return {
        draft_idx: max(
            ns - 1
            for nd, ns in rollback_configs
            if draft_idx < nd
        )
        for draft_idx in range(nd_max)
    }
