"""
Entropy-triggered repair engine.

Trigger modes
-------------
- lookback:  high uncertainty at step t  -> repair step t-1
- lookahead: high uncertainty at step t  -> repair step t
- symmetric: high uncertainty at step t  -> repair both t-1 and t
- random:    fire with fixed probability -> repair that step

Metrics are computed from per-token logprobs produced by vLLM prompt_logprobs.
"""

import math
import random as _random
from typing import Dict, List, Optional

METRIC_DIRECTION: Dict[str, bool] = {
    "mean_entropy": True,
    "max_entropy": True,
    "entropy_delta": True,
    "mean_logprob": False,
    "min_logprob": False,
    "logprob_drop": False,
}


def should_trigger(value: float, threshold: float, metric: str) -> bool:
    higher = METRIC_DIRECTION.get(metric, True)
    return value > threshold if higher else value < threshold


def compute_step_metrics_from_logprobs(
    token_logprobs: List[float],
    token_texts: List[str],
    steps: List[str],
    full_text: str,
) -> List[Dict[str, float]]:
    """Per-step uncertainty metrics from token-level logprobs."""
    # --- char boundaries for each step ---
    boundaries = []
    pos = 0
    for step in steps:
        idx = full_text.find(step, pos)
        if idx < 0:
            idx = pos
        boundaries.append((idx, idx + len(step)))
        pos = idx + len(step)

    # --- map tokens to char offsets ---
    token_offsets = []
    cpos = 0
    for txt in token_texts:
        loc = full_text.find(txt, cpos)
        if loc < 0:
            loc = cpos
        token_offsets.append(loc)
        cpos = loc + len(txt)

    # --- collect per-step metrics ---
    metrics: List[Dict[str, float]] = []
    ti = 0
    prev_mean_ent = 0.0
    prev_mean_lp = 0.0
    for si, (s_start, s_end) in enumerate(boundaries):
        lps = []
        while ti < len(token_offsets) and token_offsets[ti] < s_end:
            if token_offsets[ti] >= s_start:
                lps.append(token_logprobs[ti])
            ti += 1

        if lps:
            ents = [(-math.exp(lp) * lp) for lp in lps]
            mean_ent = sum(ents) / len(ents)
            max_ent = max(ents)
            mean_lp = sum(lps) / len(lps)
            min_lp = min(lps)
        else:
            mean_ent = max_ent = 0.0
            mean_lp = min_lp = 0.0

        entropy_delta = mean_ent - prev_mean_ent if si > 0 else 0.0
        logprob_drop = mean_lp - prev_mean_lp if si > 0 else 0.0

        metrics.append({
            "mean_entropy": mean_ent,
            "max_entropy": max_ent,
            "entropy_delta": entropy_delta,
            "mean_logprob": mean_lp,
            "min_logprob": min_lp,
            "logprob_drop": logprob_drop,
            "n_tokens": len(lps),
        })
        prev_mean_ent = mean_ent
        prev_mean_lp = mean_lp

    return metrics


def decide_repairs(
    step_metrics: List[Dict[str, float]],
    trigger_mode: str,
    trigger_metric: str,
    threshold: float,
    max_repairs: int = 3,
    random_p: float = 0.1,
    seed: Optional[int] = None,
) -> List[int]:
    """Return sorted 0-indexed step indices to repair."""
    rng = _random.Random(seed)
    repairs = set()
    n = len(step_metrics)

    for t in range(n):
        if len(repairs) >= max_repairs:
            break
        if trigger_mode == "random":
            if rng.random() < random_p:
                repairs.add(t)
            continue

        val = step_metrics[t].get(trigger_metric, 0.0)
        if not should_trigger(val, threshold, trigger_metric):
            continue

        if trigger_mode == "lookback":
            if t - 1 >= 0:
                repairs.add(t - 1)
        elif trigger_mode == "lookahead":
            repairs.add(t)
        elif trigger_mode == "symmetric":
            if t - 1 >= 0:
                repairs.add(t - 1)
            repairs.add(t)

    return sorted(repairs)[:max_repairs]
