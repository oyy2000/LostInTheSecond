"""
Self-eval signal extraction for step-level error detection.

At each step boundary, prompts the model with:
  "Is the reasoning above correct so far? Answer Yes or No."
and extracts P(Yes) / P(No) from the first generated token.

This provides a cheap (1-2 forward passes per step) self-evaluation signal
that can be combined with entropy/logprob metrics for better trigger decisions.
"""

from typing import Dict, List, Optional


SELF_EVAL_SUFFIX = "\n\nIs the reasoning above correct so far? Answer with a single word: Yes or No.\n"


def build_self_eval_prompts(
    model_id: str,
    dataset: str,
    question: str,
    steps: List[str],
) -> List[str]:
    """Build self-eval prompts for each step boundary.

    Returns one prompt per step, where each prompt includes all steps
    up to and including that step, followed by the self-eval question.
    """
    from src.prompt_templates import build_prompt

    base = build_prompt(model_id, dataset, question)
    prompts = []
    for i in range(len(steps)):
        prefix = "\n\n".join(steps[:i + 1])
        prompt = base + prefix + SELF_EVAL_SUFFIX
        prompts.append(prompt)
    return prompts


def extract_yes_no_probs(
    logprobs_dict,
    token_id: int,
    tokenizer,
) -> Dict[str, float]:
    """Extract P(Yes) and P(No) from vLLM output logprobs.

    logprobs_dict: the logprobs dict from the first generated token
    Returns {"p_yes": float, "p_no": float, "self_eval_score": float}
    where self_eval_score = p_yes / (p_yes + p_no).
    """
    import math

    yes_tokens = {"Yes", "yes", "YES", " Yes", " yes"}
    no_tokens = {"No", "no", "NO", " No", " no"}

    p_yes = 0.0
    p_no = 0.0

    if logprobs_dict is not None:
        for tid, lp_obj in logprobs_dict.items():
            decoded = getattr(lp_obj, "decoded_token", "") or ""
            decoded_strip = decoded.strip()
            lp = getattr(lp_obj, "logprob", -100.0)
            prob = math.exp(lp) if lp > -50 else 0.0
            if decoded_strip in yes_tokens or decoded in yes_tokens:
                p_yes += prob
            elif decoded_strip in no_tokens or decoded in no_tokens:
                p_no += prob

    total = p_yes + p_no
    score = p_yes / total if total > 1e-10 else 0.5

    return {
        "p_yes": p_yes,
        "p_no": p_no,
        "self_eval_score": score,
    }
