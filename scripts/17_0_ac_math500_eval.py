#!/usr/bin/env python3
"""
Offline evaluation: Adaptive Consistency (AC) on MATH500.

Reuses existing checkpoint from the Llama-3.2-3B sweep to compare:
  1. FullSC@N  (standard self-consistency, fixed budget)
  2. AC-FullSC (adaptive early stopping on FullSC samples)
  3. LateRollback (our method, fixed budget)
  4. AC-LateRollback (adaptive early stopping on LR answer pool)

No GPU needed -- purely post-hoc on pre-generated samples.

Usage:
    python scripts/17_0_ac_math500_eval.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptive_consistency_core import (
    BetaStoppingCriteria,
    DirichletStoppingCriteria,
    make_stopping_criteria,
)
from src.prompt_templates import check_answer
from src.sweep_datasets import load_dataset_by_name

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "results" / "math500_llama_3.2_3b_instruct_sweep" / "checkpoint.jsonl"
OUT_DIR = ROOT / "results" / "math500_ac_evaluation"
DATASET = "math500"
ALPHA = 0.8


def load_checkpoint():
    recs = [json.loads(l) for l in CKPT.read_text("utf-8").splitlines() if l.strip()]
    drafts, suffixes, fullsc = {}, defaultdict(list), defaultdict(list)
    for r in recs:
        did = r["doc_id"]
        if r["task_type"] == "draft":
            drafts.setdefault(did, {})[r["draft_idx"]] = r
        elif r["task_type"] == "suffix":
            suffixes[did].append(r)
        elif r["task_type"] == "fullsc":
            fullsc[did].append(r)
    for did in fullsc:
        fullsc[did].sort(key=lambda r: r["sample_idx"])
    return drafts, suffixes, fullsc


def majority_vote(answers):
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]


def build_lr_answer_sequence(did, drafts, suffixes, nd, K, alpha):
    """Build the ordered answer sequence for LateRollback(nd, K, alpha).

    Returns list of (answer, token_cost) pairs in generation order:
    for each draft 0..nd-1: draft answer, then its (K-1) suffixes.
    """
    seq = []
    d_map = drafts.get(did, {})
    sfx_all = suffixes.get(did, [])
    for di in range(nd):
        d = d_map.get(di)
        if not d:
            continue
        seq.append((d.get("draft_answer", ""), d.get("draft_tokens", 0)))
        sfx = sorted(
            [s for s in sfx_all if s["draft_idx"] == di and s["alpha"] == alpha],
            key=lambda s: s["suffix_idx"],
        )
        for s in sfx[: K - 1]:
            seq.append((s["pred_answer"], s.get("suffix_tokens", 0)))
    return seq


def eval_fixed_sc(questions, fullsc, budgets):
    """Standard FullSC at fixed budgets."""
    results = []
    nq = len(questions)
    for n in budgets:
        correct, total_tok = 0, 0
        for q in questions:
            samples = fullsc.get(q["doc_id"], [])[:n]
            answers = [s["pred_answer"] for s in samples]
            if check_answer(DATASET, majority_vote(answers), q["gold_answer"]):
                correct += 1
            total_tok += sum(s.get("tokens", 0) for s in samples)
        results.append({
            "method": f"FullSC@{n}",
            "budget": n, "accuracy": correct / nq,
            "avg_samples": float(n), "avg_tokens": total_tok / nq,
        })
    return results


def eval_ac_fullsc(questions, fullsc, criteria_name, max_n=40, **crit_kw):
    """Adaptive Consistency on FullSC samples."""
    crit = make_stopping_criteria(criteria_name)
    if crit_kw:
        crit = type(crit)(**crit_kw)
    nq = len(questions)
    correct, total_samples, total_tok = 0, 0, 0
    for q in questions:
        samples = fullsc.get(q["doc_id"], [])[:max_n]
        answers_so_far = []
        winner = ""
        for s in samples:
            answers_so_far.append(s["pred_answer"])
            total_tok += s.get("tokens", 0)
            decision = crit.should_stop(answers_so_far)
            if decision.stop:
                winner = decision.winner
                break
        if not winner:
            winner = majority_vote(answers_so_far)
        total_samples += len(answers_so_far)
        if check_answer(DATASET, winner or "", q["gold_answer"]):
            correct += 1
    return {
        "method": f"AC({criteria_name})-FullSC",
        "budget": max_n, "accuracy": correct / nq,
        "avg_samples": total_samples / nq, "avg_tokens": total_tok / nq,
    }


def eval_fixed_lr(questions, drafts, suffixes, nd, K, alpha):
    """Fixed-budget Late Rollback."""
    nq = len(questions)
    correct, total_tok = 0, 0
    for q in questions:
        seq = build_lr_answer_sequence(q["doc_id"], drafts, suffixes, nd, K, alpha)
        answers = [a for a, _ in seq]
        tokens = sum(t for _, t in seq)
        if check_answer(DATASET, majority_vote(answers), q["gold_answer"]):
            correct += 1
        total_tok += tokens
    budget = nd * K
    return {
        "method": f"LR(nd={nd},K={K})",
        "budget": budget, "accuracy": correct / nq,
        "avg_samples": float(budget), "avg_tokens": total_tok / nq,
    }


def eval_ac_lr(questions, drafts, suffixes, nd, K, alpha, criteria_name, **crit_kw):
    """Adaptive Consistency on Late Rollback answer pool."""
    crit = make_stopping_criteria(criteria_name)
    if crit_kw:
        crit = type(crit)(**crit_kw)
    nq = len(questions)
    correct, total_samples, total_tok = 0, 0, 0
    for q in questions:
        seq = build_lr_answer_sequence(q["doc_id"], drafts, suffixes, nd, K, alpha)
        answers_so_far = []
        winner = ""
        for ans, tok in seq:
            answers_so_far.append(ans)
            total_tok += tok
            decision = crit.should_stop(answers_so_far)
            if decision.stop:
                winner = decision.winner
                break
        if not winner:
            winner = majority_vote(answers_so_far)
        total_samples += len(answers_so_far)
        if check_answer(DATASET, winner or "", q["gold_answer"]):
            correct += 1
    budget = nd * K
    return {
        "method": f"AC({criteria_name})-LR(nd={nd},K={K})",
        "budget": budget, "accuracy": correct / nq,
        "avg_samples": total_samples / nq, "avg_tokens": total_tok / nq,
    }


def main():
    print("Loading MATH500 questions...")
    questions = load_dataset_by_name(DATASET)
    print(f"  {len(questions)} questions")

    print("Loading checkpoint...")
    drafts, suffixes, fullsc = load_checkpoint()
    print(f"  drafts: {sum(len(v) for v in drafts.values())}")
    print(f"  suffixes: {sum(len(v) for v in suffixes.values())}")
    print(f"  fullsc: {sum(len(v) for v in fullsc.values())}")

    all_results = []

    # -- Greedy baseline --
    nc = 0
    for q in questions:
        d = drafts.get(q["doc_id"], {}).get(0)
        if d and check_answer(DATASET, d.get("draft_answer", ""), q["gold_answer"]):
            nc += 1
    greedy_acc = nc / len(questions)
    all_results.append({
        "method": "Greedy", "budget": 1,
        "accuracy": greedy_acc, "avg_samples": 1.0, "avg_tokens": 555.4,
    })

    # -- FullSC at fixed budgets --
    all_results.extend(eval_fixed_sc(questions, fullsc, [4, 8, 16, 32, 40]))

    # -- AC on FullSC (beta criterion, several confidence levels) --
    for conf in [0.8, 0.9, 0.95, 0.99]:
        r = eval_ac_fullsc(questions, fullsc, "beta", max_n=40, confidence=conf)
        r["method"] = f"AC(beta,c={conf})-FullSC"
        all_results.append(r)

    # -- AC on FullSC (dirichlet) --
    r = eval_ac_fullsc(questions, fullsc, "dirichlet", max_n=40)
    all_results.append(r)

    # -- Fixed LR configs --
    lr_configs = [(2, 2), (2, 4), (4, 2), (4, 4), (8, 2), (8, 4)]
    for nd, K in lr_configs:
        all_results.append(eval_fixed_lr(questions, drafts, suffixes, nd, K, ALPHA))

    # -- AC on LR (beta criterion) --
    for nd, K in lr_configs:
        for conf in [0.8, 0.9, 0.95]:
            r = eval_ac_lr(
                questions, drafts, suffixes, nd, K, ALPHA,
                "beta", confidence=conf,
            )
            r["method"] = f"AC(beta,c={conf})-LR(nd={nd},K={K})"
            all_results.append(r)

    # -- AC on LR (dirichlet) --
    for nd, K in lr_configs:
        r = eval_ac_lr(
            questions, drafts, suffixes, nd, K, ALPHA, "dirichlet",
        )
        all_results.append(r)

    # -- Print summary table --
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 90)
    print(f"{'Method':<35} {'Budget':>6} {'AvgN':>6} {'Acc':>7} {'AvgTok':>8}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['method']:<35} {r['budget']:>6} {r['avg_samples']:>6.1f} "
              f"{r['accuracy']:>7.4f} {r['avg_tokens']:>8.1f}")
    print("=" * 90)

    out_path = OUT_DIR / "ac_eval_results.json"
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
