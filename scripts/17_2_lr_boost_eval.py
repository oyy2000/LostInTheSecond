#!/usr/bin/env python3 -u
"""
StepBack-Boost After Early-Stop Evaluation.

Pipeline per question:
  1. Run AC/ESC/DSC on the draft pool until it stops
  2. Record which drafts were consumed, their answers, their tokens
  3. For each consumed draft, look up its suffixes and take the first s suffixes
  4. Add suffix answers to the answer pool, re-do majority vote
  5. Compute total tokens = sum(draft_tokens) + sum(suffix_tokens)

Metric: acc_gain_per_kToken = (accuracy - greedy_acc) / (total_tokens / 1000)

Usage:
    python scripts/17_2_lr_boost_eval.py [--dataset math500]
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

print("Starting...", flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sc_unified import MethodConfig, make_policy
from src.prompt_templates import check_answer
from src.sweep_datasets import load_dataset_by_name

ROOT = Path(__file__).resolve().parent.parent
RESULTS_BASE = ROOT / "results" / "qwen2.5_3b_instruct_budget_multisignal"
SC_BUDGET = 32
SUFFIX_COUNTS = [1, 2]
ROLLBACK_STRATEGIES = ["prm_drop_fb_last", "nll_drop_fb_last"]

EARLY_STOP_METHODS = [
    ("AC(c=0.95)", {"method": "ac", "confidence": 0.95}),
    ("ESC(w=5)", {"method": "esc", "window_size": 5}),
    ("DSC(c=0.95)", {"method": "dsc", "confidence": 0.95}),
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--budget", type=int, default=SC_BUDGET)
    ap.add_argument("--strategies", nargs="+", default=None)
    ap.add_argument("--out-dir", default="")
    return ap.parse_args()


def load_checkpoint(ckpt_path):
    recs = [json.loads(l) for l in ckpt_path.read_text("utf-8").splitlines()
            if l.strip()]
    draft_map = {}
    sfx_map = defaultdict(list)
    for r in recs:
        did = r["doc_id"]
        tt = r.get("task_type", "")
        if tt == "draft":
            draft_map[(did, r["draft_idx"])] = r
        elif tt == "suffix":
            sfx_map[(did, r["draft_idx"], r["strategy"])].append(r)
    for key in sfx_map:
        sfx_map[key].sort(key=lambda s: s["suffix_idx"])
    return draft_map, sfx_map


def majority_vote(answers):
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]


def run_lr_boost(dataset, questions, draft_map, sfx_map, es_name, es_kwargs,
                 strategy, n_suffix, budget):
    """Run early-stop on draft pool, then boost with suffixes."""
    nq = len(questions)
    vanilla_correct = 0
    boosted_correct = 0
    vanilla_total_tok = 0
    boosted_total_tok = 0
    vanilla_total_samples = 0
    boosted_total_samples = 0

    for q in questions:
        did = q["doc_id"]

        # Build draft pool (ordered by draft_idx)
        pool = []
        for di in range(budget):
            d = draft_map.get((did, di))
            if d:
                pool.append({
                    "answer": d.get("draft_answer", ""),
                    "tokens": d.get("draft_tokens", 0),
                    "draft_idx": di,
                })

        if not pool:
            continue

        # Run early-stop policy on draft pool
        pool_size = len(pool)
        config = MethodConfig(max_samples=pool_size, **es_kwargs)
        policy = make_policy(config)
        result = policy.run(
            generations=[{"answer": p["answer"]} for p in pool],
            answer_extractor=None,
        )
        n_used = result.samples_used
        consumed = pool[:n_used]

        # Vanilla result
        vanilla_answers = [p["answer"] for p in consumed]
        vanilla_tok = sum(p["tokens"] for p in consumed)
        vanilla_winner = majority_vote(vanilla_answers)
        if check_answer(dataset, vanilla_winner, q["gold_answer"]):
            vanilla_correct += 1
        vanilla_total_tok += vanilla_tok
        vanilla_total_samples += n_used

        # LR-Boost: add suffixes for each consumed draft
        boosted_answers = list(vanilla_answers)
        suffix_tok = 0
        n_suffixes_added = 0
        for p in consumed:
            di = p["draft_idx"]
            suffixes = sfx_map.get((did, di, strategy), [])
            for s in suffixes[:n_suffix]:
                boosted_answers.append(s["suffix_answer"])
                suffix_tok += s.get("suffix_tokens", 0)
                n_suffixes_added += 1

        boosted_winner = majority_vote(boosted_answers)
        if check_answer(dataset, boosted_winner, q["gold_answer"]):
            boosted_correct += 1
        boosted_total_tok += vanilla_tok + suffix_tok
        boosted_total_samples += n_used + n_suffixes_added

    return {
        "vanilla_acc": vanilla_correct / nq,
        "vanilla_avg_tok": vanilla_total_tok / nq,
        "vanilla_avg_samples": vanilla_total_samples / nq,
        "boosted_acc": boosted_correct / nq,
        "boosted_avg_tok": boosted_total_tok / nq,
        "boosted_avg_samples": boosted_total_samples / nq,
    }


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
    questions = load_dataset_by_name(dataset)
    nq = len(questions)
    print(f"  {nq} questions")

    draft_map, sfx_map = load_checkpoint(ckpt_path)
    print(f"  drafts: {len(draft_map)}, suffix groups: {len(sfx_map)}")

    available_strategies = sorted(set(k[2] for k in sfx_map.keys()))
    strategies = args.strategies or available_strategies
    strategies = [s for s in strategies if s in available_strategies]
    print(f"  strategies: {strategies}")

    # Greedy baseline
    greedy_correct = 0
    greedy_tok = 0
    for q in questions:
        d = draft_map.get((q["doc_id"], 0))
        if d:
            if check_answer(dataset, d.get("draft_answer", ""),
                            q["gold_answer"]):
                greedy_correct += 1
            greedy_tok += d.get("draft_tokens", 0)
    greedy_acc = greedy_correct / nq
    greedy_avg_tok = greedy_tok / nq
    print(f"  Greedy@1: acc={greedy_acc:.4f}, avg_tok={greedy_avg_tok:.0f}")

    # SC@N baseline (all drafts, majority vote)
    sc_correct = 0
    sc_tok = 0
    for q in questions:
        answers = []
        tok = 0
        for di in range(budget):
            d = draft_map.get((q["doc_id"], di))
            if d:
                answers.append(d.get("draft_answer", ""))
                tok += d.get("draft_tokens", 0)
        if answers and check_answer(dataset, majority_vote(answers),
                                    q["gold_answer"]):
            sc_correct += 1
        sc_tok += tok
    sc_acc = sc_correct / nq
    sc_avg_tok = sc_tok / nq

    all_results = []
    all_results.append({
        "method": "Greedy@1",
        "accuracy": greedy_acc,
        "avg_tokens": greedy_avg_tok,
        "avg_samples": 1.0,
        "acc_gain_per_ktoken": 0.0,
    })
    sc_gain = (sc_acc - greedy_acc) / (sc_avg_tok / 1000) if sc_avg_tok > 0 else 0
    all_results.append({
        "method": f"SC@{budget}",
        "accuracy": sc_acc,
        "avg_tokens": sc_avg_tok,
        "avg_samples": float(budget),
        "acc_gain_per_ktoken": sc_gain,
    })
    print(f"  SC@{budget}: acc={sc_acc:.4f}, avg_tok={sc_avg_tok:.0f}", flush=True)

    # Run all combinations
    for es_name, es_kwargs in EARLY_STOP_METHODS:
        for strat in strategies:
            # Vanilla early-stop (no boost)
            print(f"  Running {es_name} / {strat}...", flush=True)
            r = run_lr_boost(dataset, questions, draft_map, sfx_map,
                             es_name, es_kwargs, strat, 0, budget)
            vanilla_gain = ((r["vanilla_acc"] - greedy_acc)
                            / (r["vanilla_avg_tok"] / 1000)
                            if r["vanilla_avg_tok"] > 0 else 0)
            all_results.append({
                "method": f"{es_name}",
                "strategy": strat,
                "accuracy": r["vanilla_acc"],
                "avg_tokens": r["vanilla_avg_tok"],
                "avg_samples": r["vanilla_avg_samples"],
                "acc_gain_per_ktoken": vanilla_gain,
            })
            print(f"    vanilla: acc={r['vanilla_acc']:.4f}", flush=True)

            # LR-Boost with s=1, s=2
            for s in SUFFIX_COUNTS:
                print(f"    +StepBack(s={s})...", flush=True)
                r = run_lr_boost(dataset, questions, draft_map, sfx_map,
                                 es_name, es_kwargs, strat, s, budget)
                boosted_gain = ((r["boosted_acc"] - greedy_acc)
                                / (r["boosted_avg_tok"] / 1000)
                                if r["boosted_avg_tok"] > 0 else 0)
                all_results.append({
                    "method": f"{es_name}+StepBack(s={s},{strat})",
                    "strategy": strat,
                    "n_suffix": s,
                    "accuracy": r["boosted_acc"],
                    "avg_tokens": r["boosted_avg_tok"],
                    "avg_samples": r["boosted_avg_samples"],
                    "acc_gain_per_ktoken": boosted_gain,
                    "vanilla_acc": r["vanilla_acc"],
                    "vanilla_avg_tok": r["vanilla_avg_tok"],
                })

    # Print summary table
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 100)
    print(f"{'Method':<45} {'Acc':>7} {'kTok':>7} {'AvgN':>6} "
          f"{'Gain/kT':>9}")
    print("-" * 100)
    for r in all_results:
        ktok = r["avg_tokens"] / 1000
        print(f"{r['method']:<45} {r['accuracy']*100:>7.2f} "
              f"{ktok:>7.2f} {r['avg_samples']:>6.1f} "
              f"{r['acc_gain_per_ktoken']*100:>9.4f}")
    print("=" * 100)

    out_path = out_dir / "lr_boost_results.json"
    out_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
