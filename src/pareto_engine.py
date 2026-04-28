"""
Pareto-front engine for iso-compute comparison.

Provides:
- repair_and_evaluate(): run trigger-based repair on greedy drafts, evaluate
- sc_pareto_points(): evaluate SC at multiple N, return (tokens, accuracy) pairs
- extract_pareto_front(): compute Pareto-optimal points from (x, y) pairs
- Token counting helpers
"""

import math
import random as _random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def extract_pareto_front(
    points: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Return Pareto-optimal (x, y) points where lower x and higher y is better."""
    if not points:
        return []
    pts = sorted(points, key=lambda p: p[0])
    front = []
    best_y = -float("inf")
    for x, y in pts:
        if y > best_y:
            front.append((x, y))
            best_y = y
    return front


def sc_pareto_points(
    questions: List[dict],
    fullsc_records: List[dict],
    dataset: str,
    sc_ns: List[int],
) -> List[Dict[str, Any]]:
    """Evaluate self-consistency at multiple sample counts.

    Returns list of dicts with keys: method, N, accuracy, mean_tokens.
    """
    from src.prompt_templates import check_answer

    q_map = {q["doc_id"]: q for q in questions}
    sc_by_q: Dict[Any, List[dict]] = {}
    for r in fullsc_records:
        sc_by_q.setdefault(r["doc_id"], []).append(r)

    nq = len(questions)
    results = []
    for N in sc_ns:
        nc, total_tok = 0, 0.0
        for q in questions:
            samples = sc_by_q.get(q["doc_id"], [])[:N]
            if not samples:
                continue
            answers = [s.get("pred_answer", "") for s in samples]
            tokens = sum(s.get("tokens", s.get("draft_tokens", 0)) for s in samples)
            total_tok += tokens
            vote = Counter(answers).most_common(1)[0][0] if answers else ""
            kw = _answer_kw(q)
            if check_answer(dataset, vote, q["gold_answer"], **kw):
                nc += 1
        results.append({
            "method": "SC",
            "N": N,
            "accuracy": nc / nq if nq else 0,
            "mean_tokens": total_tok / nq if nq else 0,
        })
    return results


def greedy_point(
    questions: List[dict],
    drafts: List[dict],
    dataset: str,
) -> Dict[str, Any]:
    """Evaluate greedy (draft_idx=0) baseline."""
    from src.prompt_templates import check_answer

    nq = len(questions)
    drafts_by_q: Dict[Any, List[dict]] = {}
    for d in drafts:
        drafts_by_q.setdefault(d["doc_id"], []).append(d)

    nc, total_tok = 0, 0.0
    for q in questions:
        ds = sorted(drafts_by_q.get(q["doc_id"], []),
                     key=lambda x: x.get("draft_idx", 0))
        if not ds:
            continue
        d0 = ds[0]
        kw = _answer_kw(q)
        if check_answer(dataset, d0.get("draft_answer", ""), q["gold_answer"], **kw):
            nc += 1
        total_tok += d0.get("draft_tokens", 0)
    return {
        "method": "Greedy",
        "accuracy": nc / nq if nq else 0,
        "mean_tokens": total_tok / nq if nq else 0,
    }


def _answer_kw(q: dict) -> dict:
    kw = {}
    if "test" in q:
        kw["test"] = q["test"]
        kw["entry_point"] = q.get("entry_point", "")
    return kw


def repair_evaluate(
    questions: List[dict],
    drafts_by_q: Dict[Any, List[dict]],
    suffix_records: List[dict],
    dataset: str,
    config_label: str,
) -> Dict[str, Any]:
    """Evaluate a single repair configuration.

    drafts_by_q: {doc_id: [draft_record, ...]}
    suffix_records: list of suffix records for this config
    """
    from src.prompt_templates import check_answer

    nq = len(questions)
    sfx_by_q: Dict[Any, List[dict]] = {}
    for s in suffix_records:
        sfx_by_q.setdefault(s["doc_id"], []).append(s)

    nc, total_tok = 0, 0.0
    for q in questions:
        did = q["doc_id"]
        kw = _answer_kw(q)
        ds = drafts_by_q.get(did, [])
        if not ds:
            continue
        d0 = ds[0]
        draft_tok = d0.get("draft_tokens", 0)
        answers = [d0.get("draft_answer", "")]
        total_tok += draft_tok

        suffixes = sfx_by_q.get(did, [])
        for s in suffixes:
            answers.append(s.get("pred_answer", ""))
            total_tok += s.get("suffix_tokens", 0)

        vote = Counter(answers).most_common(1)[0][0] if answers else ""
        if check_answer(dataset, vote, q["gold_answer"], **kw):
            nc += 1

    return {
        "method": config_label,
        "accuracy": nc / nq if nq else 0,
        "mean_tokens": total_tok / nq if nq else 0,
    }


def decide_random_repairs(
    n_steps: int,
    p: float,
    max_repairs: int = 3,
    seed: Optional[int] = None,
) -> List[int]:
    """Return step indices to repair under random trigger with probability p."""
    rng = _random.Random(seed)
    repairs = []
    for t in range(n_steps):
        if len(repairs) >= max_repairs:
            break
        if rng.random() < p:
            repairs.append(t)
    return repairs


def decide_entropy_repairs(
    step_metrics: List[Dict[str, float]],
    trigger_mode: str,
    trigger_metric: str,
    threshold: float,
    max_repairs: int = 3,
) -> List[int]:
    """Return step indices to repair under entropy-based trigger."""
    from src.repair_engine import should_trigger

    repairs = set()
    n = len(step_metrics)
    for t in range(n):
        if len(repairs) >= max_repairs:
            break
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


def build_repair_suffix_tasks(
    questions: List[dict],
    drafts_by_q: Dict[Any, List[dict]],
    repair_decisions: Dict[Any, List[int]],
    model_id: str,
    dataset: str,
    repair_k: int = 3,
    config_tag: str = "",
    tokenizer=None,
) -> List[dict]:
    """Build suffix generation tasks from repair decisions.

    repair_decisions: {doc_id: [step_indices_to_repair]}
    For each repair step, we truncate the draft at that step and generate
    repair_k suffix continuations.
    """
    from src.prompt_templates import build_prompt, split_steps

    tasks = []
    for q in questions:
        did = q["doc_id"]
        ds = drafts_by_q.get(did, [])
        if not ds:
            continue
        d0 = ds[0]
        steps = d0.get("draft_steps") or split_steps(d0.get("draft_text", ""))
        if not steps:
            continue
        repair_steps = repair_decisions.get(did, [])
        if not repair_steps:
            continue

        prompt_base = build_prompt(model_id, dataset, q["question"], tokenizer=tokenizer)
        for rs in repair_steps:
            b = max(1, min(rs, len(steps) - 1))
            prefix = "\n\n".join(steps[:b])
            prompt = prompt_base + prefix + "\n\n"
            for si in range(repair_k):
                tasks.append({
                    "task_type": "repair_suffix",
                    "doc_id": did,
                    "repair_step": rs,
                    "suffix_idx": si,
                    "gold_answer": q["gold_answer"],
                    "prefix_text": prefix,
                    "prompt": prompt,
                    "mode": "sampled",
                    "config_tag": config_tag,
                })
    return tasks
