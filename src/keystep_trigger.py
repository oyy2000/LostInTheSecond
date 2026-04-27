"""
KeyStep-SD trigger: decides whether a given reasoning step should be
upgraded to the large model.

Trigger score:
    u_k = w1 * uncertainty_k + w2 * position_k
        + w3 * consistencyRisk_k + w4 * stallSignal_k

All signals are normalised to [0, 1].  A step is triggered when u_k > tau.
"""

from typing import Any, Dict, List, Optional

from src.keystep_utils import detect_stall


def uncertainty_signal(step_stats: Dict[str, float]) -> float:
    """Normalise mean negative log-prob to [0, 1].

    Empirically, mean_neglogprob for Qwen-3B on GSM8K ranges roughly
    0.2 (confident) to 4.0+ (very uncertain).  We clamp and linearly
    scale to [0, 1] with a soft ceiling at 4.0.
    """
    raw = step_stats.get("mean_neglogprob", 0.0)
    ceiling = 4.0
    return min(raw / ceiling, 1.0)


def position_signal(step_idx: int, n_max: int = 10) -> float:
    """Inverted relative position: early steps score higher.

    Returns 1 - k/n_max so that step 1 -> ~0.9, step 10 -> 0.0.
    The intuition: early errors are costlier, so we want to be more
    willing to trigger the large model early.
    """
    return max(0.0, 1.0 - step_idx / n_max)


def stall_signal(step_text: str) -> float:
    return detect_stall(step_text)


def compute_trigger_score(
    step_text: str,
    step_stats: Dict[str, float],
    step_idx: int,
    consistency_risk: float = 0.0,
    *,
    w1: float = 0.4,
    w2: float = 0.2,
    w3: float = 0.3,
    w4: float = 0.1,
    n_max: int = 10,
) -> float:
    """Compute the composite trigger score u_k for a single step."""
    u = (
        w1 * uncertainty_signal(step_stats)
        + w2 * position_signal(step_idx, n_max)
        + w3 * consistency_risk
        + w4 * stall_signal(step_text)
    )
    return u


def decide_triggers(
    steps: List[str],
    step_stats_list: List[Dict[str, float]],
    consistency_risks: Optional[List[float]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate trigger for every step in a chain.

    Returns a list of dicts (one per step) with keys:
        step_idx, trigger_score, triggered, signals
    """
    if cfg is None:
        cfg = {}
    tcfg = cfg.get("trigger", {})
    w1 = tcfg.get("w_uncertainty", 0.4)
    w2 = tcfg.get("w_position", 0.2)
    w3 = tcfg.get("w_consistency", 0.3)
    w4 = tcfg.get("w_stall", 0.1)
    tau = tcfg.get("tau", 0.5)
    n_max = tcfg.get("n_max", 10)

    if consistency_risks is None:
        consistency_risks = [0.0] * len(steps)

    results = []
    for k, (step, stats, cr) in enumerate(
        zip(steps, step_stats_list, consistency_risks)
    ):
        u_unc = uncertainty_signal(stats)
        u_pos = position_signal(k + 1, n_max)
        u_stall = stall_signal(step)
        score = w1 * u_unc + w2 * u_pos + w3 * cr + w4 * u_stall
        results.append({
            "step_idx": k,
            "trigger_score": round(score, 4),
            "triggered": score > tau,
            "signals": {
                "uncertainty": round(u_unc, 4),
                "position": round(u_pos, 4),
                "consistency_risk": round(cr, 4),
                "stall": round(u_stall, 4),
            },
        })
    return results
