"""Reusable rollback assignments for PRM control experiments."""

from __future__ import annotations

from typing import Sequence


MAX_PRM_DROP_SIGNAL = "prm_max_drop_fb_last"
PRM_BELOW_THRESHOLD_SIGNAL = "prm_below_threshold_fb_last"
PRM_CONTROL_SIGNALS = [
    MAX_PRM_DROP_SIGNAL,
    PRM_BELOW_THRESHOLD_SIGNAL,
]


def _validated_scores(
    step_scores: Sequence[float],
    n_steps: int,
) -> list[float]:
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if len(step_scores) != n_steps:
        raise ValueError(
            f"Expected {n_steps} PRM scores, found {len(step_scores)}"
        )
    return [float(value) for value in step_scores]


def max_prm_drop_assignment(
    step_scores: Sequence[float],
    n_steps: int,
) -> dict[str, object]:
    """Rollback at the largest adjacent PRM decrease in the draft.

    The first maximum is selected deterministically. A one-step draft has no
    adjacent decrease and therefore uses the existing fallback-before-last
    convention, which is step zero.
    """
    scores = _validated_scores(step_scores, n_steps)
    drops = [
        scores[index - 1] - scores[index]
        for index in range(1, n_steps)
    ]
    if drops:
        maximum = max(drops)
        rollback_step = drops.index(maximum) + 1
        triggered = True
    else:
        maximum = None
        rollback_step = 0
        triggered = False
    return {
        "signal": MAX_PRM_DROP_SIGNAL,
        "rollback_step": rollback_step,
        "triggered": triggered,
        "threshold": None,
        "step_scores": scores,
        "score_deltas": drops,
        "maximum_drop": maximum,
    }


def prm_below_threshold_assignment(
    step_scores: Sequence[float],
    n_steps: int,
    threshold: float,
) -> dict[str, object]:
    """Rollback at the first step whose absolute PRM score is below threshold."""
    scores = _validated_scores(step_scores, n_steps)
    threshold = float(threshold)
    rollback_step = next(
        (
            index
            for index, score in enumerate(scores)
            if score < threshold
        ),
        None,
    )
    triggered = rollback_step is not None
    if rollback_step is None:
        rollback_step = max(n_steps - 1, 0)
    return {
        "signal": PRM_BELOW_THRESHOLD_SIGNAL,
        "rollback_step": rollback_step,
        "triggered": triggered,
        "threshold": threshold,
        "step_scores": scores,
        "score_deltas": [
            scores[index - 1] - scores[index]
            for index in range(1, n_steps)
        ],
    }
