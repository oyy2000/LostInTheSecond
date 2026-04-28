"""
Reusable utilities for step-boundary entropy analysis.

Instead of measuring entropy of literal delimiter tokens ("\n\n"), this module
computes the *step-start content entropy*: the average token entropy over the
first K content tokens at the beginning of each reasoning step.

Given per-token logprobs from vLLM prompt_logprobs, this module:
  - Locates the token range for each step in the response
  - Skips pure-delimiter tokens and computes H_bar_t(K) for each step
  - Labels steps relative to the first-error step (tau)
  - Aggregates within-trajectory comparisons
"""

import math
from typing import Any, Dict, List, Optional, Tuple


DELIMITER = "\n\n"
DEFAULT_K = 5


def step_char_boundaries(
    response: str, steps: List[str],
) -> List[Tuple[int, int]]:
    """Return (char_start, char_end) for each step in the response text."""
    boundaries: List[Tuple[int, int]] = []
    pos = 0
    for step in steps:
        idx = response.find(step, pos)
        if idx < 0:
            idx = pos
        boundaries.append((idx, idx + len(step)))
        pos = idx + len(step)
    return boundaries


def _token_entropy(logprob: float) -> float:
    """H = -p * log(p) for a single token given its logprob."""
    if logprob >= 0:
        return 0.0
    p = math.exp(logprob)
    return -p * logprob


def step_start_entropy(
    token_logprobs: List[float],
    token_offsets: List[int],
    step_boundaries: List[Tuple[int, int]],
    K: int = DEFAULT_K,
) -> List[Dict[str, Any]]:
    """Compute H_bar_t(K) for each step: mean entropy of first K content tokens.

    For each step, find the tokens whose character offset falls within
    [step_start, step_end), take the first K of those, and compute:
      - mean_entropy: (1/K) * sum H(token_j) for j=1..K
      - mean_neglogprob: (1/K) * sum -logprob(token_j)
      - n_tokens: actual number of tokens used (may be < K for short steps)
      - token_entropies: per-token H values

    Returns one dict per step, in order.
    """
    results: List[Dict[str, Any]] = []
    ti = 0
    for s_start, s_end in step_boundaries:
        while ti < len(token_offsets) and token_offsets[ti] < s_start:
            ti += 1

        step_lps: List[float] = []
        step_hs: List[float] = []
        scan = ti
        while scan < len(token_offsets) and token_offsets[scan] < s_end:
            step_lps.append(token_logprobs[scan])
            step_hs.append(_token_entropy(token_logprobs[scan]))
            scan += 1

        first_k_lps = step_lps[:K]
        first_k_hs = step_hs[:K]

        if first_k_hs:
            mean_ent = sum(first_k_hs) / len(first_k_hs)
            mean_nlp = -sum(first_k_lps) / len(first_k_lps)
        else:
            mean_ent = 0.0
            mean_nlp = 0.0

        whole_step_ent = sum(step_hs) / len(step_hs) if step_hs else 0.0

        results.append({
            "mean_entropy_K": round(mean_ent, 6),
            "mean_neglogprob_K": round(mean_nlp, 6),
            "whole_step_entropy": round(whole_step_ent, 6),
            "n_tokens_used": len(first_k_hs),
            "n_tokens_total": len(step_hs),
            "token_entropies_K": [round(h, 6) for h in first_k_hs],
        })
    return results


def label_steps(
    step_stats: List[Dict[str, Any]],
    tau: int,
) -> List[Dict[str, Any]]:
    """Add step_idx, is_error_step, relative_to_tau labels.

    tau is 1-indexed. step_idx is 0-indexed.
    The error step is at 0-indexed position tau-1.
    """
    error_step_idx = tau - 1
    for i, s in enumerate(step_stats):
        s["step_idx"] = i
        s["is_error_step"] = (i == error_step_idx)
        s["relative_to_tau"] = i - error_step_idx
    return step_stats


def aggregate_trajectory(
    step_stats: List[Dict[str, Any]],
    tau: int,
    doc_id: int,
    sample_idx: int,
    n_steps: int,
) -> Optional[Dict[str, Any]]:
    """Produce a single summary row for one trajectory.

    Computes delta_H = H_tau - mean(H_other) within the trajectory.
    Returns None if the error step index is out of range.
    """
    error_idx = tau - 1
    if error_idx < 0 or error_idx >= len(step_stats):
        return None

    error_ent = step_stats[error_idx]["mean_entropy_K"]
    error_nlp = step_stats[error_idx]["mean_neglogprob_K"]

    other_ents = [
        s["mean_entropy_K"] for i, s in enumerate(step_stats) if i != error_idx
    ]
    other_nlps = [
        s["mean_neglogprob_K"] for i, s in enumerate(step_stats) if i != error_idx
    ]

    mean_other_ent = sum(other_ents) / len(other_ents) if other_ents else 0.0
    mean_other_nlp = sum(other_nlps) / len(other_nlps) if other_nlps else 0.0

    return {
        "doc_id": doc_id,
        "sample_idx": sample_idx,
        "tau": tau,
        "n_steps": n_steps,
        "tau_relative": round(tau / n_steps, 4) if n_steps > 0 else 0.0,
        "error_entropy": round(error_ent, 6),
        "error_neglogprob": round(error_nlp, 6),
        "mean_other_entropy": round(mean_other_ent, 6),
        "mean_other_neglogprob": round(mean_other_nlp, 6),
        "delta_entropy": round(error_ent - mean_other_ent, 6),
        "delta_neglogprob": round(error_nlp - mean_other_nlp, 6),
        "all_entropies": [round(s["mean_entropy_K"], 6) for s in step_stats],
        "all_neglogprobs": [round(s["mean_neglogprob_K"], 6) for s in step_stats],
    }
