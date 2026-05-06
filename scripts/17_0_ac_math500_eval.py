#!/usr/bin/env python3
"""
Offline evaluation: AC / ESC / DSC / RASC on budget-controlled rollback results.

Reuses existing checkpoint from the Qwen2.5-3B-Instruct budget_multisignal sweep.
Compares four early-stopping methods (from sc_unified) against baseline SC@32
on two answer pools:
  1. FullSC pool (independent samples or drafts)
  2. Rollback pool (draft + suffix per strategy)

SC budget is fixed at 32 for all methods.

Usage:
    python scripts/17_0_ac_math500_eval.py [--dataset math500] [--strategies all]
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sc_unified import (
    Generation,
    MethodConfig,
    make_policy,
)
from src.prompt_templates import check_answer
from src.sweep_datasets import load_dataset_by_name

ROOT = Path(__file__).resolve().parent.parent
RESULTS_BASE = ROOT / "results" / "qwen2.5_3b_instruct_budget_multisignal"
ROLLBACK_CONFIGS = [(4, 8), (8, 4), (16, 2)]
SC_BUDGET = 32

# --- RASC default custom LR coefficients (from RASC repo) ---
RASC_INTERCEPT = -0.6
RASC_COEFFICIENTS = [
    0.0001,   # LEN
    -0.5,     # QUA_IM (parsing error)
    0.8,      # DIF_IV (local consistency)
    0.3,      # SIM_COT_BIGRAM
    0.3,      # SIM_COT_AGG
    0.5,      # SIM_AC_BIGRAM
    0.5,      # SIM_AC_AGG
    0.1,      # SIM_INPUT
    0.01,     # STEP_COUNT
    0.2,      # STEP_COHERENCE
]
RASC_FEATURE_NAMES = [
    "LEN", "QUA_IM", "DIF_IV", "SIM_COT_BIGRAM", "SIM_COT_AGG",
    "SIM_AC_BIGRAM", "SIM_AC_AGG", "SIM_INPUT", "STEP_COUNT",
    "STEP_COHERENCE",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="math500",
                    help="Dataset name (subdir in results)")
    ap.add_argument("--strategies", nargs="+", default=None,
                    help="Rollback strategies to evaluate (default: all)")
    ap.add_argument("--budget", type=int, default=SC_BUDGET,
                    help="Max samples budget for all methods")
    ap.add_argument("--out-dir", default="")
    return ap.parse_args()
def load_checkpoint(ckpt_path):
    recs = [json.loads(l) for l in ckpt_path.read_text("utf-8").splitlines()
            if l.strip()]
    draft_map = {}
    sfx_map = {}
    sc_by_doc = defaultdict(list)
    for r in recs:
        did = r["doc_id"]
        tt = r.get("task_type", "")
        if tt == "draft":
            draft_map[(did, r["draft_idx"])] = r
        elif tt == "suffix":
            key = (did, r["draft_idx"], r["strategy"],
                   r["rollback_step"], r["suffix_idx"])
            sfx_map[key] = r
        elif tt == "fullsc":
            sc_by_doc[did].append(r)
    for did in sc_by_doc:
        sc_by_doc[did].sort(key=lambda r: r.get("sc_idx", 0))
    return draft_map, sfx_map, sc_by_doc
def build_sfx_index(sfx_map):
    index = defaultdict(list)
    for (did, di, strat, rb, si), s in sfx_map.items():
        index[(did, di, strat)].append(s)
    for key in index:
        index[key].sort(key=lambda s: s["suffix_idx"])
    return index
def get_sc_pool(doc_id, sc_by_doc, draft_map, budget):
    if sc_by_doc:
        samples = sc_by_doc.get(doc_id, [])[:budget]
        return [{"text": s.get("sc_text", ""),
                 "answer": s["sc_answer"], "score": None}
                for s in samples]
    drafts = []
    for di in range(budget):
        d = draft_map.get((doc_id, di))
        if d:
            drafts.append({"text": d.get("draft_text", ""),
                           "answer": d.get("draft_answer", ""),
                           "score": None})
    return drafts
def get_rollback_pool(doc_id, draft_map, sfx_index, nd, strategy):
    pool = []
    for di in range(nd):
        d = draft_map.get((doc_id, di))
        if not d:
            continue
        pool.append({"text": d.get("draft_text", ""),
                     "answer": d.get("draft_answer", ""),
                     "score": None})
        for s in sfx_index.get((doc_id, di, strategy), []):
            pool.append({"text": s.get("suffix_text", ""),
                         "answer": s["suffix_answer"],
                         "score": None})
    return pool
def _bigram_jaccard(text_a, text_b):
    """Jaccard similarity on character bigrams."""
    if not text_a or not text_b:
        return 0.5
    a_lower = text_a.lower()
    b_lower = text_b.lower()
    bg_a = set(zip(a_lower, a_lower[1:]))
    bg_b = set(zip(b_lower, b_lower[1:]))
    if not bg_a and not bg_b:
        return 0.5
    inter = bg_a & bg_b
    union = bg_a | bg_b
    return len(inter) / len(union) if union else 0.5


def _count_steps(text):
    """Count reasoning steps (split by newline or 'Step')."""
    if not text:
        return 1
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return max(len(lines), 1)


def _step_coherence(text):
    """Average bigram Jaccard between consecutive steps."""
    if not text:
        return 0.5
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) <= 1:
        return 0.5
    sims = []
    for i in range(1, len(lines)):
        sims.append(_bigram_jaccard(lines[i - 1], lines[i]))
    return float(np.mean(sims))


def make_rasc_score_fn(question_text=""):
    """Build a RASC score_fn using the default custom LR coefficients.

    score_fn signature: (current: Generation, consumed: List[Generation]) -> float
    """
    coef = np.array(RASC_COEFFICIENTS)
    intercept = RASC_INTERCEPT

    def score_fn(current: Generation, consumed: list) -> float:
        text = current.text or ""
        answer = str(current.answer or "")
        idx = len(consumed) - 1

        # LEN: length of reasoning text
        feat_len = len(text)

        # QUA_IM: parsing error (1 if answer is empty/invalid)
        feat_qua_im = 1.0 if (not answer or answer == "[invalid]") else 0.0

        # DIF_IV: local consistency (1 if same answer as previous)
        if idx > 0 and consumed[idx - 1].answer is not None:
            feat_dif_iv = 1.0 if answer == str(consumed[idx - 1].answer) else 0.0
        else:
            feat_dif_iv = 0.0

        # SIM_COT_BIGRAM: bigram Jaccard with previous CoT
        if idx > 0:
            prev_text = consumed[idx - 1].text or ""
            feat_sim_cot_bigram = _bigram_jaccard(text, prev_text)
        else:
            feat_sim_cot_bigram = 0.5

        # SIM_COT_AGG: average bigram Jaccard with all previous CoTs
        if idx > 0:
            sims = [_bigram_jaccard(text, (c.text or ""))
                    for c in consumed[:idx]]
            feat_sim_cot_agg = float(np.mean(sims))
        else:
            feat_sim_cot_agg = 0.5

        # SIM_AC_BIGRAM: bigram Jaccard between current and prev answer
        if idx > 0:
            prev_ans = str(consumed[idx - 1].answer or "")
            feat_sim_ac_bigram = _bigram_jaccard(answer, prev_ans)
        else:
            feat_sim_ac_bigram = 0.5

        # SIM_AC_AGG: average bigram Jaccard with all previous answers
        if idx > 0:
            sims = [_bigram_jaccard(answer, str(c.answer or ""))
                    for c in consumed[:idx]]
            feat_sim_ac_agg = float(np.mean(sims))
        else:
            feat_sim_ac_agg = 0.5

        # SIM_INPUT: bigram Jaccard between CoT and question
        feat_sim_input = _bigram_jaccard(text, question_text)

        # STEP_COUNT
        feat_step_count = float(_count_steps(text))

        # STEP_COHERENCE
        feat_step_coherence = _step_coherence(text)

        features = np.array([
            feat_len, feat_qua_im, feat_dif_iv,
            feat_sim_cot_bigram, feat_sim_cot_agg,
            feat_sim_ac_bigram, feat_sim_ac_agg,
            feat_sim_input, feat_step_count, feat_step_coherence,
        ])
        logit = np.dot(coef, features) + intercept
        return float(1.0 / (1.0 + np.exp(-logit)))

    return score_fn
METHODS = [
    ("AC(c=0.9)", {"method": "ac", "confidence": 0.9}),
    ("AC(c=0.95)", {"method": "ac", "confidence": 0.95}),
    ("ESC(w=3)", {"method": "esc", "window_size": 3}),
    ("ESC(w=5)", {"method": "esc", "window_size": 5}),
    ("DSC(c=0.9)", {"method": "dsc", "confidence": 0.9}),
    ("DSC(c=0.95)", {"method": "dsc", "confidence": 0.95}),
    ("RASC(t=0.5,b=5)", {"method": "rasc", "rasc_threshold": 0.5,
                          "rasc_buffer_size": 5}),
    ("RASC(t=0.5,b=10)", {"method": "rasc", "rasc_threshold": 0.5,
                           "rasc_buffer_size": 10}),
]


def eval_methods_on_pool(dataset, questions, pool_fn, budget,
                         label_prefix, question_map=None):
    """Run all methods on a given answer pool and return results."""
    nq = len(questions)
    results = []
    for method_name, method_kwargs in METHODS:
        correct = 0
        total_samples = 0
        is_rasc = method_kwargs["method"] == "rasc"
        for q in questions:
            pool = pool_fn(q["doc_id"])
            if not pool:
                continue
            config = MethodConfig(max_samples=budget, **method_kwargs)
            policy = make_policy(config)
            if is_rasc and question_map:
                q_text = question_map.get(q["doc_id"], "")
                sfn = make_rasc_score_fn(q_text)
            else:
                sfn = None
            result = policy.run(
                generations=pool,
                answer_extractor=None,
                score_fn=sfn,
            )
            if check_answer(dataset, result.answer or "",
                            q["gold_answer"]):
                correct += 1
            total_samples += result.samples_used
        tag = f"{label_prefix}/{method_name}"
        results.append({
            "method": tag,
            "accuracy": correct / nq,
            "avg_samples": total_samples / nq,
        })
    return results
def main():
    args = parse_args()
    dataset = args.dataset
    budget = args.budget
    ckpt_path = RESULTS_BASE / dataset / "checkpoint.jsonl"
    out_dir = (Path(args.out_dir) if args.out_dir
               else RESULTS_BASE / dataset / "ac_evaluation")

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Dataset: {dataset}, Budget: {budget}")
    print(f"Checkpoint: {ckpt_path}")

    questions = load_dataset_by_name(dataset)
    print(f"  {len(questions)} questions")

    draft_map, sfx_map, sc_by_doc = load_checkpoint(ckpt_path)
    sfx_index = build_sfx_index(sfx_map)
    print(f"  drafts: {len(draft_map)}, suffixes: {len(sfx_map)}, "
          f"fullsc docs: {len(sc_by_doc)}")

    available_strategies = sorted(set(k[2] for k in sfx_map.keys()))
    strategies = args.strategies or available_strategies
    strategies = [s for s in strategies if s in available_strategies]
    print(f"  strategies: {strategies}")

    nd_max = max((k[1] for k in draft_map.keys()), default=0) + 1
    print(f"  nd_max: {nd_max}")

    question_map = {q["doc_id"]: q["question"] for q in questions}
    all_results = []

    # -- Baseline: SC@32 --
    nq = len(questions)
    sc_pool_fn = lambda doc_id: get_sc_pool(
        doc_id, sc_by_doc, draft_map, budget)
    sc_label = "FullSC" if sc_by_doc else "DraftSC"

    baseline_correct = 0
    for q in questions:
        pool = sc_pool_fn(q["doc_id"])
        config = MethodConfig(method="sc", max_samples=budget)
        result = make_policy(config).run(
            generations=pool, answer_extractor=None)
        if check_answer(dataset, result.answer or "", q["gold_answer"]):
            baseline_correct += 1
    baseline_acc = baseline_correct / nq
    all_results.append({
        "method": f"{sc_label}/SC@{budget} (baseline)",
        "accuracy": baseline_acc,
        "avg_samples": float(budget),
    })

    # -- Early-stopping methods on SC pool --
    print("\nEvaluating on SC pool...")
    sc_results = eval_methods_on_pool(
        dataset, questions, sc_pool_fn, budget, sc_label,
        question_map=question_map)
    all_results.extend(sc_results)

    # -- Rollback pool evaluation --
    for strat in strategies:
        for nd, ns in ROLLBACK_CONFIGS:
            if nd > nd_max:
                continue
            print(f"Evaluating on Rollback({strat}, nd={nd})...")
            rb_pool_fn = lambda doc_id, _nd=nd, _strat=strat: (
                get_rollback_pool(
                    doc_id, draft_map, sfx_index, _nd, _strat))
            sample_pool = rb_pool_fn(questions[0]["doc_id"])
            pool_size = len(sample_pool)

            # Rollback SC baseline
            rb_correct = 0
            rb_total = 0
            for q in questions:
                pool = rb_pool_fn(q["doc_id"])
                config = MethodConfig(method="sc", max_samples=len(pool))
                result = make_policy(config).run(
                    generations=pool, answer_extractor=None)
                if check_answer(dataset, result.answer or "",
                                q["gold_answer"]):
                    rb_correct += 1
                rb_total += result.samples_used
            all_results.append({
                "method": f"Rollback({strat},nd={nd})/SC@all (baseline)",
                "accuracy": rb_correct / nq,
                "avg_samples": rb_total / nq,
            })

            # Early-stopping methods
            rb_results = eval_methods_on_pool(
                dataset, questions, rb_pool_fn, pool_size,
                f"Rollback({strat},nd={nd})",
                question_map=question_map)
            all_results.extend(rb_results)

    # -- Print summary --
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print(f"{'Method':<55} {'Acc':>7} {'AvgN':>6}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['method']:<55} {r['accuracy']:>7.4f} "
              f"{r['avg_samples']:>6.1f}")
    print("=" * 80)

    out_path = out_dir / "ac_eval_results.json"
    out_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
