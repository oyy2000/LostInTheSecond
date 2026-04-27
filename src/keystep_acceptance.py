"""
KeyStep-SD acceptance gate: decides whether a large-model step should
replace the small-model step, or fall back.

Acceptance score:
    A(z_k^L) = alpha * LM_consistency
             + beta  * prefix_compatibility
             + gamma * small_model_followability

The step is accepted when A >= a_min.
"""

import math
import re
from typing import Any, Dict, List, Optional

from src.keystep_utils import number_grounding_score


# ---------------------------------------------------------------------------
# Signal 1: LM consistency  (self-evaluation by the large model)
# ---------------------------------------------------------------------------

VERIFY_PROMPT = (
    "You are a math reasoning verifier.\n\n"
    "Problem:\n{question}\n\n"
    "Previous steps:\n{prefix}\n\n"
    "Proposed next step:\n{step}\n\n"
    "Is this next step mathematically valid and logically consistent "
    "with the previous steps?  Answer with a single integer score "
    "from 1 (clearly wrong) to 5 (clearly correct)."
)


def build_verify_prompt(
    question: str, prefix_steps: List[str], step_text: str,
) -> str:
    prefix = "\n".join(
        f"Step {i+1}: {s}" for i, s in enumerate(prefix_steps)
    ) if prefix_steps else "(none)"
    return VERIFY_PROMPT.format(
        question=question, prefix=prefix, step=step_text,
    )


_SCORE_RE = re.compile(r"[1-5]")


def parse_verify_score(raw: str) -> float:
    """Extract a 1-5 score from the model's response, normalise to [0,1]."""
    m = _SCORE_RE.search(raw)
    if m:
        return (int(m.group()) - 1) / 4.0
    return 0.5


# ---------------------------------------------------------------------------
# Signal 2: prefix compatibility  (number grounding heuristic)
# ---------------------------------------------------------------------------

def prefix_compatibility_score(
    step_text: str, question: str, prefix_steps: List[str],
) -> float:
    return number_grounding_score(step_text, question, prefix_steps)


# ---------------------------------------------------------------------------
# Signal 3: small-model followability
# ---------------------------------------------------------------------------

def followability_score(continuation_logprobs: List[float]) -> float:
    """Score based on how comfortably the small model can continue.

    continuation_logprobs: per-token logprobs from the small model when
    generating one step after the large-model step.

    Returns a score in [0, 1].  Low mean logprob (very negative) -> low
    followability.  We normalise similarly to the uncertainty signal.
    """
    if not continuation_logprobs:
        return 0.5
    mean_lp = sum(continuation_logprobs) / len(continuation_logprobs)
    ceiling = -4.0
    score = max(0.0, min(1.0, 1.0 + mean_lp / abs(ceiling)))
    return score


# ---------------------------------------------------------------------------
# Composite acceptance
# ---------------------------------------------------------------------------

def compute_acceptance(
    lm_consistency: float,
    prefix_compat: float,
    followability: float,
    *,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
) -> float:
    return alpha * lm_consistency + beta * prefix_compat + gamma * followability


def decide_acceptance(
    lm_consistency: float,
    prefix_compat: float,
    followability: float,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = {}
    acfg = cfg.get("acceptance", {})
    alpha = acfg.get("alpha_lm_consistency", 0.4)
    beta = acfg.get("beta_prefix_compat", 0.3)
    gamma = acfg.get("gamma_followability", 0.3)
    a_min = acfg.get("a_min", 0.5)

    score = compute_acceptance(
        lm_consistency, prefix_compat, followability,
        alpha=alpha, beta=beta, gamma=gamma,
    )
    return {
        "acceptance_score": round(score, 4),
        "accepted": score >= a_min,
        "signals": {
            "lm_consistency": round(lm_consistency, 4),
            "prefix_compatibility": round(prefix_compat, 4),
            "followability": round(followability, 4),
        },
    }
