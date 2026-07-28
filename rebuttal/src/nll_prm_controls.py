"""Reusable signal and rollback logic for NLL/PRM rebuttal controls."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


NLL_SIGNAL = "nll_drop_fb_last"
PRM_SIGNAL = "prm_drop_fb_last"
ALL_SIGNALS = [NLL_SIGNAL, PRM_SIGNAL]


def step_char_bounds(
    response: str,
    steps: Sequence[str],
) -> List[Tuple[int, int]]:
    """Locate each split reasoning step in the original response."""
    bounds = []
    cursor = 0
    for step in steps:
        start = response.find(step, cursor)
        if start < 0:
            raise ValueError(
                "A reasoning step could not be located in the draft text"
            )
        end = start + len(step)
        bounds.append((start, end))
        cursor = end
    return bounds


def compute_step_mean_nll(
    token_logprobs: Sequence[float],
    token_offsets: Sequence[int],
    bounds: Sequence[Tuple[int, int]],
) -> List[float]:
    """Average token NLL within every response-step character span."""
    if len(token_logprobs) != len(token_offsets):
        raise ValueError("token_logprobs and token_offsets differ in length")
    result = []
    token_index = 0
    for start, end in bounds:
        while (
            token_index < len(token_offsets)
            and token_offsets[token_index] < start
        ):
            token_index += 1
        values = []
        scan = token_index
        while scan < len(token_offsets) and token_offsets[scan] < end:
            values.append(token_logprobs[scan])
            scan += 1
        if not values:
            raise ValueError(
                f"No scored token overlaps reasoning-step span {start}:{end}"
            )
        result.append(float(-np.mean(values)))
    return result


def _first_threshold_crossing(
    deltas: Iterable[float],
    threshold: float,
) -> int | None:
    for index, delta in enumerate(deltas):
        if delta > threshold:
            return index + 1
    return None


def nll_rollback_assignment(
    step_nlls: Sequence[float],
    n_steps: int,
    threshold: float,
) -> Dict[str, object]:
    """Return first NLL-increase rollback, falling back before the last step."""
    if len(step_nlls) != n_steps:
        raise ValueError(
            f"Expected {n_steps} NLL scores, found {len(step_nlls)}"
        )
    deltas = [
        float(step_nlls[index] - step_nlls[index - 1])
        for index in range(1, len(step_nlls))
    ]
    rollback_step = _first_threshold_crossing(deltas, threshold)
    triggered = rollback_step is not None
    if rollback_step is None:
        rollback_step = max(n_steps - 1, 0)
    return {
        "signal": NLL_SIGNAL,
        "rollback_step": rollback_step,
        "triggered": triggered,
        "threshold": threshold,
        "step_scores": [float(value) for value in step_nlls],
        "score_deltas": deltas,
    }


def prm_rollback_assignment(
    step_scores: Sequence[float],
    n_steps: int,
    threshold: float,
) -> Dict[str, object]:
    """Return first PRM-score-drop rollback, falling back before last step."""
    if len(step_scores) != n_steps:
        raise ValueError(
            f"Expected {n_steps} PRM scores, found {len(step_scores)}"
        )
    deltas = [
        float(step_scores[index - 1] - step_scores[index])
        for index in range(1, len(step_scores))
    ]
    rollback_step = _first_threshold_crossing(deltas, threshold)
    triggered = rollback_step is not None
    if rollback_step is None:
        rollback_step = max(n_steps - 1, 0)
    return {
        "signal": PRM_SIGNAL,
        "rollback_step": rollback_step,
        "triggered": triggered,
        "threshold": threshold,
        "step_scores": [float(value) for value in step_scores],
        "score_deltas": deltas,
    }


def retained_prefix(steps: Sequence[str], rollback_step: int) -> str:
    """Build the retained reasoning prefix used by suffix generation."""
    if not 0 <= rollback_step < len(steps):
        raise ValueError(
            f"rollback_step={rollback_step} is invalid for {len(steps)} steps"
        )
    if rollback_step == 0:
        return ""
    return "\n\n".join(steps[:rollback_step]) + "\n\n"
